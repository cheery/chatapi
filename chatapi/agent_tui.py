"""Textual TUI for the chatapi coding agent."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TabbedContent,
    TabPane,
)

from . import chatfmt
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
.msg-tool-summary {
    color: $warning;
    padding: 0 1;
    margin: 0 0;
}
.msg-error {
    color: $error;
    padding: 0 1;
}
#tools-view {
    height: 1fr;
    scrollbar-size: 1 1;
    padding: 0 1;
}
.tool-entry {
    color: $text;
    padding: 0;
    margin: 0;
}
.tool-separator {
    color: $text-disabled;
    padding: 0;
    margin: 0;
}

/* Model picker modal */
ModelPickerScreen {
    align: center middle;
}
#model-picker {
    width: 60;
    height: 20;
    background: $surface;
    border: thick $accent;
    padding: 1 2;
}
#model-picker Label {
    text-align: center;
    margin-bottom: 1;
}
#model-picker ListView {
    height: 1fr;
}

/* File dialog modal */
FileDialogScreen {
    align: center middle;
}
#file-dialog {
    width: 60;
    height: 8;
    background: $surface;
    border: thick $accent;
    padding: 1 2;
}
#file-dialog Input {
    margin-top: 1;
}
"""


class ModelPickerScreen(ModalScreen[str | None]):
    BINDINGS = [("escape", "dismiss_none", "Cancel")]

    def __init__(self, models: list[str]):
        super().__init__()
        self.models = models

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="model-picker"):
            yield Label("Select Model")
            yield ListView(
                *[ListItem(Label(m)) for m in self.models]
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self.models):
            self.dismiss(self.models[idx])
        else:
            self.dismiss(None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class FileDialogScreen(ModalScreen[str | None]):
    BINDINGS = [("escape", "dismiss_none", "Cancel")]

    def __init__(self, title: str, default: str = ""):
        super().__init__()
        self._title = title
        self._default = default

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="file-dialog"):
            yield Label(self._title)
            yield Input(value=self._default, id="file-input")

    def on_mount(self) -> None:
        self.query_one("#file-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class AgentApp(App):
    TITLE = "chatapi-agent"

    CSS = _CSS

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "stop_agent", "Stop"),
        ("ctrl+m", "pick_model", "Model"),
        ("ctrl+s", "save_session", "Save"),
        ("ctrl+l", "load_session", "Load"),
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
        self._pending_tool_calls: list[StreamEvent] = []
        self._pending_tool_results: list[StreamEvent] = []
        self._tool_counter = 0
        self._needs_separator = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Chat"):
                yield VerticalScroll(id="conversation")
            with TabPane("Tools"):
                yield VerticalScroll(id="tools-view")
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
            self._add_chat("error", f"Failed to connect: {e}")
            return
        self.query_one("#input-bar", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "input-bar":
            return
        text = event.value.strip()
        if not text or self.agent is None:
            return
        event.input.value = ""
        self._add_chat("user", text)
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
        self._pending_tool_calls = []
        self._pending_tool_results = []
        self._tool_counter = 0
        self._needs_separator = False
        try:
            async for event in self.agent.send_user_message(text):
                await self._handle_event(event)
        except asyncio.CancelledError:
            self._add_chat("error", "Stopped")
        except Exception as e:
            self._add_chat("error", str(e))
        finally:
            self._flush_tool_summary()
            input_widget.disabled = False
            input_widget.focus()

    async def _handle_event(self, event: StreamEvent) -> None:
        if event.kind == "chunk":
            tag = event.meta.get("tag", "_")
            if tag == "assistant":
                if self._needs_separator:
                    self._assistant_text += "\n\n"
                    self._needs_separator = False
                self._assistant_text += event.text
                if self._assistant_widget is None:
                    self._assistant_widget = self._add_chat("assistant", Markdown(self._assistant_text))
                else:
                    self._assistant_widget.update(Markdown(self._assistant_text))
                self._scroll_chat()
            elif tag == "think":
                self._think_text += event.text
                if self._think_widget is None:
                    self._think_widget = self._add_chat("think", self._think_text)
                else:
                    self._think_widget.update(self._think_text)
                self._scroll_chat()
        elif event.kind == "tool_call":
            self._needs_separator = True
            self._tool_counter += 1
            self._pending_tool_calls.append(event)
        elif event.kind == "tool_result":
            self._pending_tool_results.append(event)
        elif event.kind == "done":
            self._flush_tool_summary()
        elif event.kind == "error":
            self._flush_tool_summary()
            self._add_chat("error", event.text)
            self._scroll_chat()

    def _flush_tool_summary(self) -> None:
        if not self._pending_tool_calls:
            return
        n = len(self._pending_tool_calls)
        self._add_chat("tool-summary", f"[{n}] tool call{'s' if n != 1 else ''}")
        self._append_tool_details()
        self._pending_tool_calls = []
        self._pending_tool_results = []
        self._scroll_chat()

    def _append_tool_details(self) -> None:
        tools_view = self.query_one("#tools-view", VerticalScroll)
        tools_view.mount(Static("---", classes="tool-separator"))
        for i, call in enumerate(self._pending_tool_calls):
            result = self._pending_tool_results[i] if i < len(self._pending_tool_results) else None
            idx = self._tool_counter - len(self._pending_tool_calls) + i + 1
            header = f"[{idx}] {call.tool_name}"
            call_body = call.text
            tools_view.mount(Static(header, classes="tool-entry"))
            tools_view.mount(Static(call_body, classes="tool-entry"))
            tools_view.mount(Static("---", classes="tool-separator"))
            if result:
                result_text = result.text
                if len(result_text) > 2000:
                    result_text = result_text[:2000] + "\n..."
                tools_view.mount(Static(result_text, classes="tool-entry"))
            tools_view.mount(Static("---", classes="tool-separator"))
        tools_view.scroll_end(animate=False)

    def _add_chat(self, role: str, content) -> Static:
        conv = self.query_one("#conversation", VerticalScroll)
        widget = Static(content, classes=f"msg-{role}")
        conv.mount(widget)
        return widget

    def _scroll_chat(self) -> None:
        conv = self.query_one("#conversation", VerticalScroll)
        conv.scroll_end(animate=False)

    async def action_stop_agent(self) -> None:
        if self._agent_task and not self._agent_task.done():
            if self.agent:
                try:
                    await self.agent.abort()
                except Exception:
                    pass
            self._agent_task.cancel()
            self._agent_task = None

    async def action_pick_model(self) -> None:
        if self.agent is None:
            return
        try:
            models = await self.agent.list_models()
        except Exception as e:
            self._add_chat("error", f"Failed to list models: {e}")
            return
        if not models:
            self._add_chat("error", "No models available")
            return

        def _on_pick(result: str | None) -> None:
            if result:
                asyncio.create_task(self._switch_model(result))

        self.push_screen(ModelPickerScreen(models), _on_pick)

    async def _switch_model(self, model: str) -> None:
        if self.agent is None:
            return
        try:
            await self.agent.set_model(model)
            self._add_chat("tool-summary", f"Switched to model: {model}")
        except Exception as e:
            self._add_chat("error", f"Failed to switch model: {e}")

    async def action_save_session(self) -> None:
        if self.agent is None:
            return

        def _on_save(result: str | None) -> None:
            if result:
                path = result if result.endswith(".cfmt") else result + ".cfmt"
                try:
                    self.agent.save_session(path)
                except Exception:
                    pass

        self.push_screen(FileDialogScreen("Save session (.cfmt)", "session.cfmt"), _on_save)

    async def action_load_session(self) -> None:
        if self.agent is None:
            return

        def _on_load(result: str | None) -> None:
            if result:
                path = result if result.endswith(".cfmt") else result + ".cfmt"
                asyncio.create_task(self._do_load(path))

        self.push_screen(FileDialogScreen("Load session (.cfmt)"), _on_load)

    async def _do_load(self, path: str) -> None:
        if self.agent is None:
            return
        try:
            await self.agent.load_session(path)
        except Exception:
            pass


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
