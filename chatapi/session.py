"""Chat session state and registry."""
from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field

from .chatfmt import CFMessage


@dataclass
class ChatSession:
    id: str
    model: str | None = None
    messages: list[CFMessage] = field(default_factory=list)
    pending_request: bytearray = field(default_factory=bytearray)
    inflight: dict[int, asyncio.Task] = field(default_factory=dict)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(self) -> ChatSession:
        sid = secrets.token_hex(4)
        s = ChatSession(id=sid)
        self._sessions[sid] = s
        return s

    def get(self, sid: str) -> ChatSession | None:
        return self._sessions.get(sid)

    def end(self, sid: str) -> ChatSession | None:
        return self._sessions.pop(sid, None)

    def all(self) -> list[ChatSession]:
        return list(self._sessions.values())
