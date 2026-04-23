"""Minimal async client helper for chatapi."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from . import protocol
from .wire import read_frame, write_frame


class ChatClient:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        # Client uses even message ids; 0 is reserved for the version handshake.
        self._next_id = 2

    @classmethod
    async def connect_unix(cls, path: str) -> "ChatClient":
        reader, writer = await asyncio.open_unix_connection(path)
        return cls(reader, writer)

    @classmethod
    async def connect_tcp(cls, host: str, port: int) -> "ChatClient":
        reader, writer = await asyncio.open_connection(host, port)
        return cls(reader, writer)

    def next_id(self) -> int:
        mid = self._next_id
        self._next_id += 2
        return mid

    async def send(self, name: str, *args: str, payload: bytes = b"") -> None:
        body = protocol.encode(protocol.Message(name=name, args=args, payload=payload))
        await write_frame(self.writer, body)

    async def recv(self) -> protocol.Message:
        body = await read_frame(self.reader)
        return protocol.decode(body)

    async def request(self, name: str, *args: str, payload: bytes = b"") -> AsyncIterator[protocol.Message]:
        await self.send(name, *args, payload=payload)
        while True:
            msg = await self.recv()
            yield msg
            if msg.kind in ("response", "bare"):
                return
            if msg.name in ("aborted!", "refuse!", "context_limit_reached!", "context_limit_unknown!"):
                return

    async def handshake(self, version: str = "0") -> protocol.Message:
        await self.send("version?", "0", payload=version.encode("utf-8"))
        return await self.recv()

    async def bye(self, reason: str = "") -> None:
        await self.send("bye", payload=reason.encode("utf-8"))
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except Exception:
            pass
