"""Asyncio server speaking the chatapi protocol."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

from . import chatfmt, protocol
from .backend import Backend, ContextLimitError, ContextLimitUnknown
from .session import ChatSession, SessionRegistry
from .wire import read_frame, write_frame

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "0"


class _Sender:
    """Serializes writes onto a single StreamWriter."""

    def __init__(self, writer: asyncio.StreamWriter):
        self._writer = writer
        self._lock = asyncio.Lock()

    async def send(self, name: str, *args: str, payload: bytes = b"") -> None:
        body = protocol.encode(protocol.Message(name=name, args=args, payload=payload))
        async with self._lock:
            await write_frame(self._writer, body)


class Connection:
    def __init__(self, backend: Backend, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.backend = backend
        self.reader = reader
        self.writer = writer
        self.sender = _Sender(writer)
        self.sessions = SessionRegistry()
        # Map of in-flight request message-id -> task. Used by abort?.
        self.inflight: dict[int, asyncio.Task] = {}
        self._handshake_done = False

    # --- entry point ---

    async def serve(self) -> None:
        try:
            await self._handshake()
            if not self._handshake_done:
                return
            await self._main_loop()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception:
            log.exception("connection error")
        finally:
            await self._shutdown()

    async def _handshake(self) -> None:
        body = await read_frame(self.reader)
        msg = protocol.decode(body)
        if msg.name != "version?":
            await self.sender.send("not_supported!", "0", payload=PROTOCOL_VERSION.encode())
            return
        client_version = msg.payload.decode("utf-8", errors="replace")
        if client_version != PROTOCOL_VERSION:
            await self.sender.send("not_supported!", "0", payload=PROTOCOL_VERSION.encode())
            return
        await self.sender.send("version!", "0", payload=b"")
        self._handshake_done = True

    async def _main_loop(self) -> None:
        while True:
            try:
                body = await read_frame(self.reader)
            except asyncio.IncompleteReadError:
                return
            try:
                msg = protocol.decode(body)
            except ValueError as e:
                log.warning("malformed frame: %s", e)
                continue

            if msg.name == "bye":
                return

            handler = _HANDLERS.get(msg.name)
            if handler is None:
                await self._refuse(msg, "unknown call")
                continue

            # Run request handlers as tasks so long-running streams (complete?)
            # don't block subsequent requests on the same connection.
            task = asyncio.create_task(self._run(handler, msg))
            mid = msg.message_id
            if mid is not None:
                self.inflight[mid] = task
                task.add_done_callback(lambda _t, k=mid: self.inflight.pop(k, None))
            session = self._session(msg)
            if session is not None and mid is not None and msg.name == "complete?":
                session.inflight[mid] = task
                task.add_done_callback(lambda _t, s=session, k=mid: s.inflight.pop(k, None))

    async def _run(self, handler: Callable[["Connection", protocol.Message], Awaitable[None]], msg: protocol.Message) -> None:
        try:
            await handler(self, msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("handler %s failed", msg.name)
            try:
                await self._refuse(msg, "internal error")
            except Exception:
                pass

    async def _shutdown(self) -> None:
        for task in list(self.inflight.values()):
            task.cancel()
        if self.inflight:
            await asyncio.gather(*self.inflight.values(), return_exceptions=True)
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass

    # --- helpers ---

    async def _refuse(self, msg: protocol.Message, reason: str) -> None:
        mid = msg.args[0] if msg.args else "0"
        sid = msg.args[1] if len(msg.args) > 1 else ""
        await self.sender.send("refuse!", mid, sid, payload=reason.encode("utf-8"))

    def _session(self, msg: protocol.Message) -> ChatSession | None:
        if len(msg.args) < 2:
            return None
        return self.sessions.get(msg.args[1])


# --- handler functions ---

async def _h_version(conn: Connection, msg: protocol.Message) -> None:
    # Subsequent version? after handshake is unexpected.
    await conn._refuse(msg, "version handshake already completed")


async def _h_supported(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    # Only one supported version in v0; emit it as the terminating non-stream response.
    await conn.sender.send("supported!", mid, payload=PROTOCOL_VERSION.encode())


async def _h_models(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    flavor = msg.payload.decode("utf-8") or None
    models = await conn.backend.list_models(flavor)
    if not models:
        await conn.sender.send("models!", mid, "", flavor or "", payload=b"")
        return
    *stream, last = models
    for m in stream:
        await conn.sender.send("models*!", mid, m.name, m.flavor, payload=b"")
    await conn.sender.send("models!", mid, last.name, last.flavor, payload=b"")


async def _h_default_model(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    flavor = msg.args[2] if len(msg.args) > 2 else None
    if flavor == "":
        flavor = None
    info = conn.backend.default_model(flavor)
    if info is None:
        await conn._refuse(msg, "no default model")
        return
    await conn.sender.send("default_model!", mid, info.name, info.flavor, payload=b"")


async def _h_chat(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    session = conn.sessions.create()
    await conn.sender.send("chat!", mid, session.id, payload=b"")


async def _h_model(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    session = conn._session(msg)
    if session is None:
        await conn._refuse(msg, "unknown session")
        return
    if len(msg.args) < 3:
        await conn._refuse(msg, "missing model name")
        return
    name = msg.args[2]
    session.model = name
    await conn.sender.send("model!", mid, session.id, name, payload=b"")


async def _h_message_stream(conn: Connection, msg: protocol.Message) -> None:
    session = conn._session(msg)
    if session is None:
        await conn._refuse(msg, "unknown session")
        return
    session.pending_request += msg.payload


async def _h_message(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    session = conn._session(msg)
    if session is None:
        await conn._refuse(msg, "unknown session")
        return
    session.pending_request += msg.payload
    buffered = bytes(session.pending_request)
    session.pending_request.clear()
    if buffered:
        try:
            decoded = chatfmt.decode_file(buffered)
        except ValueError as e:
            await conn._refuse(msg, f"malformed chatfmt: {e}")
            return
        session.messages.extend(decoded)
    await conn.sender.send("message!", mid, session.id, payload=b"")


async def _h_complete(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    session = conn._session(msg)
    if session is None:
        await conn._refuse(msg, "unknown session")
        return
    if session.model is None:
        await conn._refuse(msg, "no model selected")
        return
    if msg.payload:
        try:
            extra = chatfmt.decode_file(msg.payload)
        except ValueError as e:
            await conn._refuse(msg, f"malformed chatfmt: {e}")
            return
        session.messages.extend(extra)

    # Each backend yields (tag, delta). The first chunk for each block carries
    # the block's tag (and may carry args/kwargs/meta in the future); follow-on
    # deltas in the same block are emitted with tag '_' so the client can
    # accumulate body and meta into the open block. Session history records
    # the merged blocks via chatfmt.merge_chunks.
    chunks: list[chatfmt.CFMessage] = []
    usage: dict = {}
    t0 = time.monotonic()
    try:
        async for tag, delta in conn.backend.stream_complete(
            session.model, session.messages, usage_out=usage,
        ):
            if chunks and chunks[-1].tag != chatfmt.CONT_TAG and chunks[-1].tag == tag:
                chunk_msg = chatfmt.cont(content=delta)
            elif chunks and chunks[-1].tag == chatfmt.CONT_TAG and _last_block_tag(chunks) == tag:
                chunk_msg = chatfmt.cont(content=delta)
            else:
                chunk_msg = chatfmt.CFMessage(tag=tag, body=delta)
            chunks.append(chunk_msg)
            await conn.sender.send(
                "complete*!", mid, session.id,
                payload=chatfmt.encode_message(chunk_msg),
            )
    except ContextLimitError:
        await conn.sender.send("context_limit_reached!", mid, session.id, payload=b"")
        return
    except asyncio.CancelledError:
        # abort? cancelled this task; emit aborted! for the original message.
        await conn.sender.send("aborted!", mid, session.id, payload=b"")
        raise

    if chunks:
        at_iso = datetime.now(timezone.utc).isoformat(timespec="minutes")
        meta_kwargs: dict = {"_at": at_iso, "_time": time.monotonic() - t0}
        if "output_tokens" in usage:
            meta_kwargs["_tokens"] = usage["output_tokens"]
        trailing = chatfmt.cont(**meta_kwargs)
        chunks.append(trailing)
        await conn.sender.send(
            "complete*!", mid, session.id,
            payload=chatfmt.encode_message(trailing),
        )

    session.messages.extend(chatfmt.merge_chunks(chunks))
    await conn.sender.send("complete!", mid, session.id, payload=b"")


def _last_block_tag(chunks: list[chatfmt.CFMessage]) -> str | None:
    for c in reversed(chunks):
        if c.tag != chatfmt.CONT_TAG:
            return c.tag
    return None


async def _h_abort(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    session = conn._session(msg)
    sid = session.id if session else (msg.args[1] if len(msg.args) > 1 else "")
    # abort? has no target message-id arg in the spec; it cancels all in-flight
    # completions on the named session (v0 has at most one).
    if session is not None:
        for task in list(session.inflight.values()):
            task.cancel()
    await conn.sender.send("abort!", mid, sid, payload=b"")


async def _h_end(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    session = conn._session(msg)
    if session is None:
        await conn._refuse(msg, "unknown session")
        return
    conn.sessions.end(session.id)
    await conn.sender.send("end!", mid, session.id, payload=b"")


async def _h_context_limit(conn: Connection, msg: protocol.Message) -> None:
    mid = msg.args[0]
    session = conn._session(msg)
    if session is None:
        await conn._refuse(msg, "unknown session")
        return
    if session.model is None:
        await conn._refuse(msg, "no model selected")
        return
    try:
        used, total = await conn.backend.context_limit(session.model, session.messages)
    except ContextLimitUnknown:
        await conn.sender.send("context_limit_unknown!", mid, session.id, payload=b"")
        return
    await conn.sender.send("context_limit!", mid, session.id, str(used), str(total), payload=b"")


_HANDLERS: dict[str, Callable[[Connection, protocol.Message], Awaitable[None]]] = {
    "version?": _h_version,
    "supported?": _h_supported,
    "models?": _h_models,
    "default_model?": _h_default_model,
    "chat?": _h_chat,
    "model?": _h_model,
    "message*?": _h_message_stream,
    "message?": _h_message,
    "complete?": _h_complete,
    "abort?": _h_abort,
    "end?": _h_end,
    "context_limit?": _h_context_limit,
}


# --- top-level server lifecycle ---

async def serve_unix(backend: Backend, path: str) -> asyncio.AbstractServer:
    async def handler(reader, writer):
        await Connection(backend, reader, writer).serve()
    return await asyncio.start_unix_server(handler, path=path)


async def serve_tcp(backend: Backend, host: str, port: int) -> asyncio.AbstractServer:
    async def handler(reader, writer):
        await Connection(backend, reader, writer).serve()
    return await asyncio.start_server(handler, host=host, port=port)
