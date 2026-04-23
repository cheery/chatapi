"""Rich-based CLI for the chatapi coding agent."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

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

HELP_TEXT = """[bold]commands[/]
  /model [name]   set model, or pick interactively when no name is given
  /models         list available models
  /save [path]    save session to .cfmt (default: session.cfmt)
  /load <path>    load session from a .cfmt file
  /help           show this help
  /quit           exit (Ctrl+D also works)

[dim]While a turn is streaming, Ctrl+C aborts it and returns you to the prompt.[/]"""

_ANSI_DIM_ITALIC = "\033[2;3m"
_ANSI_RESET = "\033[0m"


def _fmt_tool_input(text: str):
    text = text or ""
    try:
        obj = json.loads(text)
    except Exception:
        return Text(text.strip() or "{}")
    pretty = json.dumps(obj, indent=2, ensure_ascii=False)
    return Syntax(
        pretty,
        "json",
        theme="ansi_dark",
        background_color="default",
        word_wrap=True,
    )


class AgentCLI:
    def __init__(self, agent: CodingAgent, console: Console):
        self.agent = agent
        self.console = console
        self._current_task: asyncio.Task | None = None
        self._stream_tag: str | None = None

    async def run(self) -> None:
        self._banner()
        while True:
            try:
                text = await asyncio.to_thread(self._prompt)
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if await self._handle_command(text):
                    break
                continue
            await self._run_turn(text)

    def _banner(self) -> None:
        self.console.print(
            Panel.fit(
                Text.from_markup(
                    f"[bold cyan]chatapi-agent[/] connected\n"
                    f"  model: [green]{self.agent.model}[/]\n"
                    f"  cwd:   [dim]{self.agent.working_dir}[/]\n"
                    "  Type [bold]/help[/] for commands."
                ),
                border_style="cyan",
            )
        )

    def _prompt(self) -> str:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return input("> ")

    async def _handle_command(self, text: str) -> bool:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in ("/quit", "/exit"):
            return True
        if cmd == "/help":
            self.console.print(HELP_TEXT)
            return False
        if cmd == "/models":
            await self._list_models()
            return False
        if cmd == "/model":
            await self._choose_model(arg)
            return False
        if cmd == "/save":
            self._save(arg or "session.cfmt")
            return False
        if cmd == "/load":
            if not arg:
                self.console.print("[red]/load requires a path[/]")
                return False
            await self._load(arg)
            return False
        self.console.print(f"[red]unknown command: {cmd}[/]  (try /help)")
        return False

    async def _list_models(self) -> list[str]:
        try:
            models = await self.agent.list_models()
        except Exception as e:
            self.console.print(f"[red]list_models failed: {e}[/]")
            return []
        if not models:
            self.console.print("[yellow]no models available[/]")
            return []
        for i, m in enumerate(models):
            marker = "[green]*[/]" if m == self.agent.model else " "
            self.console.print(f"  {marker} [{i:>2}] {m}")
        return models

    async def _choose_model(self, name: str) -> None:
        if name:
            await self._set_model(name)
            return
        models = await self._list_models()
        if not models:
            return
        try:
            raw = await asyncio.to_thread(input, "select (name or index)> ")
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return
        raw = raw.strip()
        if not raw:
            return
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(models):
                await self._set_model(models[idx])
            else:
                self.console.print("[red]index out of range[/]")
            return
        if raw in models:
            await self._set_model(raw)
        else:
            self.console.print(f"[red]not a known model: {raw}[/]")

    async def _set_model(self, name: str) -> None:
        try:
            await self.agent.set_model(name)
            self.console.print(f"[green]model set to {name}[/]")
        except Exception as e:
            self.console.print(f"[red]set_model failed: {e}[/]")

    def _save(self, path: str) -> None:
        if not path.endswith(".cfmt"):
            path += ".cfmt"
        try:
            self.agent.save_session(path)
            self.console.print(f"[green]saved session to {path}[/]")
        except Exception as e:
            self.console.print(f"[red]save failed: {e}[/]")

    async def _load(self, path: str) -> None:
        if not path.endswith(".cfmt"):
            path += ".cfmt"
        try:
            await self.agent.load_session(path)
            self.console.print(f"[green]loaded session from {path}[/]")
        except Exception as e:
            self.console.print(f"[red]load failed: {e}[/]")

    async def _run_turn(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(self._consume(text))
        self._current_task = task

        handler_installed = False
        try:
            loop.add_signal_handler(
                signal.SIGINT,
                lambda: task.cancel() if not task.done() else None,
            )
            handler_installed = True
        except (NotImplementedError, ValueError):
            pass

        try:
            await task
        except asyncio.CancelledError:
            self._end_stream_if_open()
            self.console.print("[yellow]aborted[/]")
            try:
                await self.agent.abort()
            except Exception:
                pass
        finally:
            self._current_task = None
            if handler_installed:
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except Exception:
                    pass

    async def _consume(self, text: str) -> None:
        self._stream_tag = None
        tool_counter = 0
        pending: dict[str, int] = {}

        self.console.rule(style="bright_black")

        try:
            async for event in self.agent.send_user_message(text):
                if event.kind == "chunk":
                    tag = event.meta.get("tag", "_")
                    if tag not in ("assistant", "think"):
                        continue
                    if self._stream_tag != tag:
                        self._end_stream_if_open()
                        self._begin_stream(tag)
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
                elif event.kind == "tool_call":
                    self._end_stream_if_open()
                    tool_counter += 1
                    pending[event.call_id] = tool_counter
                    self.console.print(
                        Panel(
                            _fmt_tool_input(event.text),
                            title=Text.from_markup(
                                f"[bold yellow]→ call[/]  [cyan]{event.tool_name}[/]  [dim]#{tool_counter}[/]"
                            ),
                            border_style="yellow",
                            expand=False,
                        )
                    )
                elif event.kind == "tool_result":
                    idx = pending.get(event.call_id, "?")
                    result = event.text or "(empty)"
                    truncated = len(result) > 3000
                    if truncated:
                        result = result[:3000]
                    body = Text(result.rstrip() or "(empty)")
                    if truncated:
                        body.append("\n... (truncated)", style="dim")
                    self.console.print(
                        Panel(
                            body,
                            title=Text.from_markup(
                                f"[bold blue]← result[/]  [dim]#{idx}[/]"
                            ),
                            border_style="blue",
                            expand=False,
                        )
                    )
                elif event.kind == "done":
                    self._end_stream_if_open()
                elif event.kind == "error":
                    self._end_stream_if_open()
                    self.console.print(f"[red]error: {event.text}[/]")
        finally:
            self._end_stream_if_open()

    def _begin_stream(self, tag: str) -> None:
        if tag == "assistant":
            self.console.print("[bold green]assistant[/]")
        else:
            self.console.print("[dim italic]think[/]")
            sys.stdout.write(_ANSI_DIM_ITALIC)
            sys.stdout.flush()
        self._stream_tag = tag

    def _end_stream_if_open(self) -> None:
        if self._stream_tag is None:
            return
        if self._stream_tag == "think":
            sys.stdout.write(_ANSI_RESET)
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._stream_tag = None


async def _amain(args: argparse.Namespace) -> int:
    console = Console()
    tcp_host = tcp_port = None
    if args.tcp:
        host, _, port_s = args.tcp.rpartition(":")
        tcp_host = host
        try:
            tcp_port = int(port_s)
        except ValueError:
            console.print(f"[red]invalid --tcp spec: {args.tcp!r}[/]")
            return 2

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.prompt:
        try:
            system_prompt = Path(args.prompt).read_text()
        except OSError as e:
            console.print(f"[red]failed to read prompt file: {e}[/]")
            return 2

    try:
        agent = await CodingAgent.create(
            socket_path=args.socket,
            tcp_host=tcp_host,
            tcp_port=tcp_port,
            model=args.model,
            working_dir=args.dir,
            system_prompt=system_prompt,
        )
    except Exception as e:
        console.print(f"[red]failed to connect: {e}[/]")
        return 1

    cli = AgentCLI(agent, console)
    try:
        await cli.run()
    finally:
        try:
            await agent.close()
        except Exception:
            pass
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="chatapi-agent")
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--socket", help="unix socket path")
    conn.add_argument("--tcp", help="HOST:PORT (loopback)")
    parser.add_argument("--model", help="model name (vendor/model)")
    parser.add_argument("-C", "--dir", default=".", help="working directory")
    parser.add_argument("--prompt", default=None, help="system prompt file (default: built-in)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
