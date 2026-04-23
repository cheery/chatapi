"""Anthropic SDK adapter for the chatapi server."""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

import anthropic

from . import chatfmt
from .backend import Backend, ContextLimitError, ContextLimitUnknown, ModelInfo
from .chatfmt import CFMessage

DEFAULT_CHAT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 8192
# Extended-thinking budget. Models without thinking support will reject this;
# the caller can adjust the model selection or disable thinking in the future.
DEFAULT_THINKING_BUDGET = 1024


def _to_anthropic(
    messages: list[CFMessage],
) -> tuple[str | None, list[dict], list[dict]]:
    """Translate chatfmt messages into (system, messages, tools).

    Groups consecutive assistant-side blocks (assistant text + call) into one
    assistant turn with list-form content, and consecutive ret blocks into
    one user turn with tool_result blocks, matching Anthropic's API.
    """
    system_parts: list[str] = []
    tools: list[dict] = []
    out: list[dict] = []
    pending_assistant: list[dict] | None = None
    pending_rets: list[dict] | None = None

    def flush_rets() -> None:
        nonlocal pending_rets
        if pending_rets is not None:
            out.append({"role": "user", "content": pending_rets})
            pending_rets = None

    def flush_assistant() -> None:
        nonlocal pending_assistant
        if pending_assistant is None:
            return
        blocks = pending_assistant
        pending_assistant = None
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            out.append({"role": "assistant", "content": blocks[0]["text"]})
        else:
            out.append({"role": "assistant", "content": blocks})

    for m in messages:
        body = m.body or ""
        if m.tag == "system":
            flush_rets()
            flush_assistant()
            system_parts.append(body)
        elif m.tag == "tool":
            flush_rets()
            flush_assistant()
            if not m.args:
                raise ValueError("tool message requires a name arg")
            tool_def: dict = {
                "name": m.args[0],
                "input_schema": json.loads(body) if body else {},
            }
            if "description" in m.kwargs:
                tool_def["description"] = m.kwargs["description"]
            tools.append(tool_def)
        elif m.tag == "user":
            flush_rets()
            flush_assistant()
            out.append({"role": "user", "content": body})
        elif m.tag == "assistant":
            flush_rets()
            if pending_assistant is None:
                pending_assistant = []
            pending_assistant.append({"type": "text", "text": body})
        elif m.tag == "call":
            flush_rets()
            if len(m.args) < 2:
                raise ValueError("call message requires id and name args")
            if pending_assistant is None:
                pending_assistant = []
            pending_assistant.append({
                "type": "tool_use",
                "id": m.args[0],
                "name": m.args[1],
                "input": json.loads(body) if body else {},
            })
        elif m.tag == "ret":
            flush_assistant()
            if not m.args:
                raise ValueError("ret message requires id arg")
            if pending_rets is None:
                pending_rets = []
            block: dict = {
                "type": "tool_result",
                "tool_use_id": m.args[0],
                "content": body,
            }
            if m.meta.get("error"):
                block["is_error"] = True
            pending_rets.append(block)
        elif m.tag == "think":
            # TODO(v0): chatfmt 'think' messages are dropped on the way to Anthropic.
            continue
        else:
            raise ValueError(f"unsupported chatfmt tag for Anthropic: {m.tag}")

    flush_rets()
    flush_assistant()

    system = "\n\n".join(system_parts) if system_parts else None
    return system, out, tools


class AnthropicBackend(Backend):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        kwargs: dict = {"api_key": api_key or os.environ.get("ANTHROPIC_API_KEY")}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def list_models(self, flavor: str | None) -> list[ModelInfo]:
        if flavor not in (None, "", "chat"):
            return []
        out: list[ModelInfo] = []
        async for m in self._client.models.list():
            out.append(ModelInfo(name=m.id, flavor="chat"))
        return out

    def default_model(self, flavor: str | None) -> ModelInfo | None:
        if flavor not in (None, "", "chat"):
            return None
        return ModelInfo(name=DEFAULT_CHAT_MODEL, flavor="chat")

    async def stream_complete(
        self, model: str, messages: list[CFMessage], usage_out: dict | None = None,
    ) -> AsyncIterator[CFMessage]:
        system, msgs, tools = _to_anthropic(messages)
        kwargs: dict = {
            "model": model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": msgs,
            "thinking": {"type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET},
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "content_block_start":
                        cb = getattr(event, "content_block", None)
                        cbtype = getattr(cb, "type", None)
                        if cbtype == "text":
                            yield CFMessage(tag="assistant", body="")
                        elif cbtype == "thinking":
                            yield CFMessage(tag="think", body="")
                        elif cbtype == "tool_use":
                            yield CFMessage(
                                tag="call",
                                args=[cb.id, cb.name],
                                body="",
                            )
                    elif etype == "content_block_delta":
                        delta = event.delta
                        dtype = getattr(delta, "type", None)
                        if dtype == "thinking_delta":
                            yield chatfmt.cont(content=delta.thinking)
                        elif dtype == "text_delta":
                            yield chatfmt.cont(content=delta.text)
                        elif dtype == "input_json_delta":
                            yield chatfmt.cont(content=delta.partial_json)
                        # signature_delta and other delta types are ignored for v0.
                    elif etype == "message_delta" and usage_out is not None:
                        usage = getattr(event, "usage", None)
                        out_tokens = getattr(usage, "output_tokens", None)
                        if out_tokens is not None:
                            usage_out["output_tokens"] = out_tokens
        except anthropic.BadRequestError as e:
            msg = str(e).lower()
            if "context" in msg or "too long" in msg or "max_tokens" in msg:
                raise ContextLimitError(str(e)) from e
            raise

    async def context_limit(
        self, model: str, messages: list[CFMessage]
    ) -> tuple[int, int]:
        # TODO(v0): Anthropic API doesn't expose used/total tokens for an
        # in-progress chat session synchronously. Surface unknown to the client.
        raise ContextLimitUnknown(model)
