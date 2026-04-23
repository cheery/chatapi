"""Textual TUI for the chatapi coding agent."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from .agent import CodingAgent, StreamEvent

DEFAULT_SYSTEM_PROMPT = """\
You are an expert coding assistant. You help users with coding tasks by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands
- edit: Make surgical edits to files
- write: Create or overwrite files

Guidelines:
- Use bash for file operations like ls, grep, find
- Use read to examine files before editing
- Use edit for precise changes (old text must match exactly)
- Use write only for new files or complete rewrites
- When summarizing your actions, output plain text directly - do NOT use cat or bash to display what you did
- Be concise in your responses
- Show file paths clearly when working with files
"""

_CSS = """
Screen {
    layout: vertical;
}
#conversation {
    height: 1fr;
    scrollbar-size: 1 1;
    padding: 0 1;
}
#input-bar {
    dock: bottom;
    height: 3;
    margin: 0 1;
}
.msg-user {
    color: $text;
    background: $surface;
    padding: 0 1;
    margin: 1 0;
}
.msg-assistant {
    color: $text;
    padding: 0 1;
    margin: 1 0;
}
.msg-think {
    color: $text-disabled;
    background: $surface-darken-1;
    padding: 0 1;
    margin: 1 0;
}
.msg-tool {
    color: $warning;
    padding: 0 1;
}
.msg-error {
    color: $error;
    padding: 0 1;
}
"""


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class AgentApp(App):
    TITLE = "chatapi-agent"

    CSS = _CSS

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "abort", "Abort"),
    ]

    def __init__(
        self,
        socket_path: str | None = None,
        tcp_host: str | None = None,
        tcp_port: int | None = None,
        model: str | None = None,
        working_dir: str = ".",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        super().__init__()
        self.socket_path = socket_path
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.model = model
        self.working_dir = working_dir
        self.system_prompt = system_prompt
        self.agent: CodingAgent | None = None
        self._agent_task: asyncio.Task | None = None
        self._assistant_widget: Static | None = None
        self._think_widget: Static | None = None
        self._assistant_text = ""
        self._think_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="conversation")
        yield Input(id="input-bar", placeholder="Type a message...")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self.agent = await CodingAgent.create(
                socket_path=self.socket_path,
                tcp_host=self.tcp_host,
                tcp_port=self.tcp_port,
                model=self.model,
                working_dir=self.working_dir,
                system_prompt=self.system_prompt,
            )
        except Exception as e:
            self._add_message("error", f"Failed to connect: {e}")
            return
        self.query_one("#input-bar", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.agent is None:
            return
        event.input.value = ""
        self._add_message("user", text)
        input_widget = event.input
        input_widget.disabled = True
        self._agent_task = asyncio.create_task(
            self._run_agent_loop(text, input_widget)
        )

    async def _run_agent_loop(self, text: str, input_widget: Input) -> None:
        self._assistant_text = ""
        self._think_text = ""
        self._assistant_widget = None
        self._think_widget = None
        try:
            async for event in self.agent.send_user_message(text):
                await self._handle_event(event)
        except asyncio.CancelledError:
            self._add_message("error", "Aborted")
        except Exception as e:
            self._add_message("error", str(e))
        finally:
            input_widget.disabled = False
            input_widget.focus()

    async def _handle_event(self, event: StreamEvent) -> None:
        if event.kind == "chunk":
            tag = event.meta.get("tag", "_")
            if tag in ("assistant", "_") and self._assistant_widget is not None:
                self._assistant_text += event.text
                self._assistant_widget.update(Markdown(self._assistant_text))
                self._scroll_to_bottom()
            elif tag in ("think",) or (tag == "_" and self._think_widget is not None):
                self._think_text += event.text
                if self._think_widget:
                    self._think_widget.update(self._think_text)
                    self._scroll_to_bottom()
            elif tag == "assistant":
                self._assistant_text = event.text
                self._assistant_widget = self._add_message("assistant", Markdown(self._assistant_text))
                self._scroll_to_bottom()
            elif tag == "think":
                self._think_text = event.text
                self._think_widget = self._add_message("think", self._think_text)
                self._scroll_to_bottom()
        elif event.kind == "tool_call":
            self._add_message(
                "tool", f">> {event.tool_name}: {_truncate(event.text, 200)}"
            )
            self._scroll_to_bottom()
        elif event.kind == "tool_result":
            self._add_message(
                "tool", f"<< {event.tool_name}: {_truncate(event.text, 300)}"
            )
            self._scroll_to_bottom()
        elif event.kind == "done":
            pass
        elif event.kind == "error":
            self._add_message("error", event.text)
            self._scroll_to_bottom()

    def _add_message(self, role: str, content) -> Static:
        conv = self.query_one("#conversation", VerticalScroll)
        widget = Static(content, classes=f"msg-{role}")
        conv.mount(widget)
        return widget

    def _scroll_to_bottom(self) -> None:
        conv = self.query_one("#conversation", VerticalScroll)
        conv.scroll_end(animate=False)

    def action_abort(self) -> None:
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            self._agent_task = None


def main() -> None:
    parser = argparse.ArgumentParser(prog="chatapi-agent")
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--socket", help="unix socket path")
    conn.add_argument("--tcp", help="HOST:PORT (loopback)")
    parser.add_argument("--model", help="model name (vendor/model)")
    parser.add_argument("-C", "--dir", default=".", help="working directory")
    parser.add_argument("--prompt", default=None, help="system prompt file (default: built-in)")
    args = parser.parse_args()

    tcp_host = tcp_port = None
    if args.tcp:
        host, _, port_s = args.tcp.rpartition(":")
        tcp_host = host
        tcp_port = int(port_s)

    prompt = DEFAULT_SYSTEM_PROMPT
    if args.prompt:
        prompt = Path(args.prompt).read_text()

    app = AgentApp(
        socket_path=args.socket,
        tcp_host=tcp_host,
        tcp_port=tcp_port,
        model=args.model,
        working_dir=args.dir,
        system_prompt=prompt,
    )
    app.run()


if __name__ == "__main__":
    main()
