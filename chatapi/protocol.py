"""chatapi wire-message structure: name+suffix, args, payload."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RS = 0x1E
GS = 0x1D

Kind = Literal["request", "stream_request", "response", "stream_response", "bare"]

_SUFFIXES: list[tuple[str, Kind]] = [
    ("*?", "stream_request"),
    ("*!", "stream_response"),
    ("?", "request"),
    ("!", "response"),
]


def _check_arg(s: str) -> None:
    for ch in s:
        if ord(ch) < 0x20:
            raise ValueError(f"arg contains forbidden control char {ord(ch):#x}")


@dataclass
class Message:
    name: str
    args: tuple[str, ...] = ()
    payload: bytes = b""

    @property
    def kind(self) -> Kind:
        for suffix, k in _SUFFIXES:
            if self.name.endswith(suffix):
                return k
        return "bare"

    @property
    def base_name(self) -> str:
        for suffix, _ in _SUFFIXES:
            if self.name.endswith(suffix):
                return self.name[: -len(suffix)]
        return self.name

    @property
    def message_id(self) -> int | None:
        if not self.args:
            return None
        try:
            return int(self.args[0])
        except ValueError:
            return None


def encode(msg: Message) -> bytes:
    _check_arg(msg.name)
    parts = [msg.name.encode("utf-8")]
    for a in msg.args:
        _check_arg(a)
        parts.append(a.encode("utf-8"))
    out = bytearray()
    for i, p in enumerate(parts):
        if i > 0:
            out.append(RS)
        out += p
    out.append(GS)
    out += msg.payload
    return bytes(out)


def decode(body: bytes) -> Message:
    gs_pos = body.find(GS)
    if gs_pos < 0:
        raise ValueError("message has no GS separator")
    arg_section = body[:gs_pos]
    payload = body[gs_pos + 1 :]
    if not arg_section:
        raise ValueError("message has no name")
    raw_args = arg_section.split(bytes([RS]))
    name = raw_args[0].decode("utf-8")
    args = tuple(a.decode("utf-8") for a in raw_args[1:])
    return Message(name=name, args=args, payload=payload)
