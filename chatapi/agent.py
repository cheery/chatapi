"""Coding agent: connects to chatapi server, runs the agent loop."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from . import chatfmt, protocol
from .chatfmt import CFMessage
from .client import ChatClient

MAX_TOOL_OUTPUT = 100_000


@dataclass
class StreamEvent:
    kind: str  # "chunk", "tool_call", "tool_result", "done", "error"
    text: str = ""
    tool_name: str = ""
    call_id: str = ""
    meta: dict = field(default_factory=dict)


def _bash_schema() -> str:
    return json.dumps({
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"},
            "timeout": {"type": "integer", "description": "Optional timeout in seconds"},
        },
        "required": ["command"],
    })


def _read_schema() -> str:
    return json.dumps({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path to the file to read"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "integer", "description": "Maximum number of lines to read"},
        },
        "required": ["path"],
    })


def _edit_schema() -> str:
    return json.dumps({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path to the file"},
            "oldText": {"type": "string", "description": "Exact text to find. Must be unique in the file."},
            "newText": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "oldText", "newText"],
    })


def _write_schema() -> str:
    return json.dumps({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    })


def build_tools() -> list[CFMessage]:
    return [
        chatfmt.tool("bash", _bash_schema(), description="Execute a bash command."),
        chatfmt.tool("read", _read_schema(), description="Read file contents."),
        chatfmt.tool("edit", _edit_schema(), description="Make surgical edits to files."),
        chatfmt.tool("write", _write_schema(), description="Create or overwrite files."),
    ]


async def _simple_request(
    client: ChatClient, name: str, *args: str, payload: bytes = b"",
) -> protocol.Message:
    await client.send(name, *args, payload=payload)
    return await client.recv()


def _resolve_path(working_dir: str, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(working_dir) / p


class CodingAgent:
    def __init__(
        self,
        client: ChatClient,
        session_id: str,
        model: str,
        working_dir: str,
    ):
        self.client = client
        self.session_id = session_id
        self.model = model
        self.working_dir = os.path.abspath(working_dir)

    @classmethod
    async def create(
        cls,
        *,
        socket_path: str | None = None,
        tcp_host: str | None = None,
        tcp_port: int | None = None,
        model: str | None = None,
        working_dir: str = ".",
        system_prompt: str = "",
    ) -> CodingAgent:
        if socket_path:
            client = await ChatClient.connect_unix(socket_path)
        elif tcp_host and tcp_port:
            client = await ChatClient.connect_tcp(tcp_host, tcp_port)
        else:
            raise ValueError("specify --socket or --tcp")

        await client.handshake()

        if model is None:
            mid = client.next_id()
            resp = await _simple_request(client, "default_model?", str(mid), "chat")
            if resp.name == "refuse!":
                await client.bye()
                raise RuntimeError("server has no default model")
            model = resp.args[2] if len(resp.args) > 2 else resp.args[1]

        mid = client.next_id()
        resp = await _simple_request(client, "chat?", str(mid))
        if resp.name == "refuse!":
            await client.bye()
            raise RuntimeError("server refused to create session")
        session_id = resp.args[1]

        mid = client.next_id()
        resp = await _simple_request(client, "model?", str(mid), session_id, model)
        if resp.name == "refuse!":
            await client.bye()
            raise RuntimeError(f"server refused model {model!r}")

        messages = [chatfmt.system(system_prompt)] + build_tools()
        mid = client.next_id()
        payload = chatfmt.encode_file(messages)
        resp = await _simple_request(client, "message?", str(mid), session_id, payload=payload)
        if resp.name == "refuse!":
            await client.bye()
            raise RuntimeError("server refused initial message")

        return cls(client, session_id, model, working_dir)

    async def send_user_message(self, text: str) -> AsyncIterator[StreamEvent]:
        msg = chatfmt.user(text)
        mid = self.client.next_id()
        payload = chatfmt.encode_file([msg])
        resp = await _simple_request(
            self.client, "message?", str(mid), self.session_id, payload=payload,
        )
        if resp.name == "refuse!":
            yield StreamEvent(kind="error", text="server refused user message")
            return
        async for event in self._run_loop():
            yield event

    async def _run_loop(self) -> AsyncIterator[StreamEvent]:
        while True:
            mid = self.client.next_id()
            chunks: list[CFMessage] = []
            current_tag: str | None = None

            async for msg in self.client.request(
                "complete?", str(mid), self.session_id, payload=b""
            ):
                if msg.name == "complete*!":
                    try:
                        chunk = chatfmt.decode_message(msg.payload)
                    except ValueError:
                        continue
                    if chunk.tag != "_":
                        current_tag = chunk.tag
                    chunks.append(chunk)
                    yield StreamEvent(
                        kind="chunk",
                        text=chunk.body or "",
                        meta={"tag": current_tag or chunk.tag},
                    )
                elif msg.name in (
                    "refuse!", "context_limit_reached!",
                    "context_limit_unknown!", "aborted!",
                ):
                    yield StreamEvent(kind="error", text=msg.name)
                    return

            blocks = chatfmt.merge_chunks(chunks)
            call_blocks = [b for b in blocks if b.tag == "call"]

            if not call_blocks:
                yield StreamEvent(kind="done")
                return

            ret_messages: list[CFMessage] = []
            for cb in call_blocks:
                call_id = str(cb.args[0])
                tool_name = str(cb.args[1])
                try:
                    tool_input = json.loads(cb.body or "{}")
                except json.JSONDecodeError:
                    tool_input = {}
                yield StreamEvent(
                    kind="tool_call",
                    tool_name=tool_name,
                    call_id=call_id,
                    text=json.dumps(tool_input),
                )
                result_content, is_error = await self._execute_tool(tool_name, tool_input)
                yield StreamEvent(
                    kind="tool_result",
                    tool_name=tool_name,
                    call_id=call_id,
                    text=result_content,
                )
                ret_messages.append(
                    chatfmt.ret(call_id, result_content, error=is_error)
                )

            mid = self.client.next_id()
            payload = chatfmt.encode_file(ret_messages)
            resp = await _simple_request(
                self.client, "message?", str(mid), self.session_id, payload=payload,
            )
            if resp.name == "refuse!":
                yield StreamEvent(kind="error", text="server refused tool results")
                return

    async def _execute_tool(self, name: str, args: dict) -> tuple[str, bool]:
        if name == "bash":
            return await self._tool_bash(args)
        elif name == "read":
            return self._tool_read(args)
        elif name == "edit":
            return self._tool_edit(args)
        elif name == "write":
            return self._tool_write(args)
        else:
            return f"Unknown tool: {name}", True

    async def _tool_bash(self, args: dict) -> tuple[str, bool]:
        command = args.get("command", "")
        timeout = args.get("timeout")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.working_dir,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "Timed out", True
            output = stdout.decode("utf-8", errors="replace")
            if len(output) > MAX_TOOL_OUTPUT:
                output = output[:MAX_TOOL_OUTPUT] + "\n... (truncated)"
            return output, proc.returncode != 0
        except Exception as e:
            return str(e), True

    def _tool_read(self, args: dict) -> tuple[str, bool]:
        path = args.get("path", "")
        offset = args.get("offset")
        limit = args.get("limit")
        full = _resolve_path(self.working_dir, path)
        try:
            lines = full.read_text().splitlines(keepends=True)
            start = (offset - 1) if offset else 0
            end = (start + limit) if limit else len(lines)
            content = "".join(lines[start:end])
            return content, False
        except Exception as e:
            return str(e), True

    def _tool_edit(self, args: dict) -> tuple[str, bool]:
        path = args.get("path", "")
        old_text = args.get("oldText", "")
        new_text = args.get("newText", "")
        if not old_text:
            return "oldText is empty", True
        full = _resolve_path(self.working_dir, path)
        try:
            content = full.read_text()
            count = content.count(old_text)
            if count == 0:
                return "oldText not found", True
            if count > 1:
                return f"oldText found {count} times, must be unique", True
            content = content.replace(old_text, new_text, 1)
            full.write_text(content)
            return "ok", False
        except Exception as e:
            return str(e), True

    def _tool_write(self, args: dict) -> tuple[str, bool]:
        path = args.get("path", "")
        content = args.get("content", "")
        full = _resolve_path(self.working_dir, path)
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            return "ok", False
        except Exception as e:
            return str(e), True

    async def close(self) -> None:
        mid = self.client.next_id()
        try:
            await self.client.send("end?", str(mid), self.session_id, payload=b"")
            await self.client.recv()
        except Exception:
            pass
        await self.client.bye()
