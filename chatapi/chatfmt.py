"""Encode/decode chatfmt messages."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .wire import decode_vlq_bytes, encode_vlq

DC1 = 0x11
DC2 = 0x12
DC3 = 0x13
DC4 = 0x14
FS = 0x1C
GS = 0x1D
RS = 0x1E
US = 0x1F
ESC = 0x5C  # backslash

_VALUE_LEAD = {RS, DC1, DC2, DC3, DC4}
_TEXT_TERMINATORS = {US, RS, DC1, DC2, DC3, DC4, GS, FS}
_BODY_TERMINATORS = {US, FS}
_NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")


@dataclass
class CFMessage:
    tag: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    body: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def make(tag: str, *args: Any, **kwargs: Any) -> CFMessage:
    body = kwargs.pop("content", None)
    fields: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k.startswith("_"):
            meta[k[1:]] = v
        else:
            fields[k] = v
    return CFMessage(tag=tag, args=list(args), kwargs=fields, body=body, meta=meta)


def system(content: str) -> CFMessage:
    return CFMessage(tag="system", body=content)


def user(content: str) -> CFMessage:
    return CFMessage(tag="user", body=content)


def assistant(content: str) -> CFMessage:
    return CFMessage(tag="assistant", body=content)


def think(content: str) -> CFMessage:
    return CFMessage(tag="think", body=content)


CONT_TAG = "_"


def cont(content: str | None = None, **meta: Any) -> CFMessage:
    """Build a streaming continuation chunk (tag '_').

    Body is appended to the parent block's body; keyword args (which must
    start with '_') become meta entries to merge into the parent's meta.
    """
    msg = CFMessage(tag=CONT_TAG)
    if content is not None:
        msg.body = content
    for k, v in meta.items():
        if not k.startswith("_"):
            raise ValueError(f"continuation accepts only meta kwargs (start with _): {k}")
        msg.meta[k[1:]] = v
    return msg


def merge_chunks(chunks) -> list[CFMessage]:
    """Fold a sequence of streamed chunks into complete blocks.

    A chunk with a non-'_' tag opens a new block, carrying its tag, args,
    kwargs, body, and meta. A '_' chunk extends the most recent block:
    its body is appended and its meta is merged in (last write wins).
    """
    blocks: list[CFMessage] = []
    for chunk in chunks:
        if chunk.tag == CONT_TAG:
            if not blocks:
                raise ValueError("continuation chunk arrived before any block")
            current = blocks[-1]
            if chunk.body is not None:
                current.body = (current.body or "") + chunk.body
            for k, v in chunk.meta.items():
                current.meta[k] = v
        else:
            blocks.append(
                CFMessage(
                    tag=chunk.tag,
                    args=list(chunk.args),
                    kwargs=dict(chunk.kwargs),
                    body=chunk.body,
                    meta=dict(chunk.meta),
                )
            )
    return blocks


def _check_name(name: str) -> None:
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid chatfmt name: {name!r}")


def _check_text(s: str) -> None:
    for ch in s:
        if ord(ch) in _TEXT_TERMINATORS:
            raise ValueError(f"text contains forbidden control char {ord(ch):#x}")


def _encode_value(v: Any) -> bytes:
    # bool must come before int, since bool is a subclass of int.
    if v is ...:
        return bytes([DC4])
    if v is None:
        return bytes([DC4]) + b"null"
    if isinstance(v, bool):
        return bytes([DC4]) + (b"true" if v else b"false")
    if isinstance(v, int):
        return bytes([DC1]) + str(v).encode("utf-8")
    if isinstance(v, float):
        return bytes([DC2]) + repr(v).encode("utf-8")
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        return bytes([DC3]) + encode_vlq(len(b)) + b
    if isinstance(v, str):
        _check_text(v)
        return bytes([RS]) + v.encode("utf-8")
    raise TypeError(f"unsupported chatfmt value type: {type(v).__name__}")


def _encode_body(text: str) -> bytes:
    out = bytearray()
    for byte in text.encode("utf-8"):
        if byte == ESC:
            out += b"\\5c"
        elif byte < 0x20 and byte not in (0x09, 0x0A, 0x0D):
            out += f"\\{byte:02x}".encode("ascii")
        else:
            out.append(byte)
    return bytes(out)


def _decode_body(data: bytes) -> str:
    out = bytearray()
    i = 0
    while i < len(data):
        b = data[i]
        if b == ESC:
            if i + 2 >= len(data):
                raise ValueError("truncated body escape")
            try:
                out.append(int(data[i + 1 : i + 3].decode("ascii"), 16))
            except ValueError as e:
                raise ValueError("invalid body escape") from e
            i += 3
        else:
            out.append(b)
            i += 1
    return out.decode("utf-8")


def encode_message(msg: CFMessage) -> bytes:
    _check_name(msg.tag)
    out = bytearray(msg.tag.encode("utf-8"))
    for v in msg.args:
        out += _encode_value(v)
    for k, v in msg.kwargs.items():
        _check_name(k)
        out.append(US)
        out += k.encode("utf-8")
        out += _encode_value(v)
    if msg.body is not None or msg.meta:
        out.append(GS)
        if msg.body is not None:
            out += _encode_body(msg.body)
        for k, v in msg.meta.items():
            _check_name(k)
            out.append(US)
            out += k.encode("utf-8")
            out += _encode_value(v)
    return bytes(out)


def encode_file(messages: list[CFMessage]) -> bytes:
    out = bytearray()
    for m in messages:
        out += encode_message(m)
        out.append(FS)
    return bytes(out)


class _Cursor:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def peek(self) -> int | None:
        if self.pos >= len(self.data):
            return None
        return self.data[self.pos]

    def read(self) -> int:
        b = self.data[self.pos]
        self.pos += 1
        return b


def _read_text(cur: _Cursor) -> str:
    start = cur.pos
    while cur.pos < len(cur.data) and cur.data[cur.pos] not in _TEXT_TERMINATORS:
        cur.pos += 1
    return cur.data[start : cur.pos].decode("utf-8")


def _read_value(cur: _Cursor) -> Any:
    lead = cur.read()
    if lead == RS:
        return _read_text(cur)
    if lead == DC1:
        return int(_read_text(cur))
    if lead == DC2:
        return float(_read_text(cur))
    if lead == DC3:
        length, consumed = decode_vlq_bytes(cur.data, cur.pos)
        cur.pos += consumed
        b = cur.data[cur.pos : cur.pos + length]
        if len(b) != length:
            raise ValueError("blob shorter than declared length")
        cur.pos += length
        return bytes(b)
    if lead == DC4:
        text = _read_text(cur)
        if text == "":
            return ...
        if text == "true":
            return True
        if text == "false":
            return False
        if text == "null":
            return None
        raise ValueError(f"invalid DC4 sentinel: {text!r}")
    raise ValueError(f"unexpected value lead: {lead:#x}")


def _read_keyword(cur: _Cursor) -> str:
    start = cur.pos
    while cur.pos < len(cur.data) and cur.data[cur.pos] not in _VALUE_LEAD:
        cur.pos += 1
    name = cur.data[start : cur.pos].decode("utf-8")
    _check_name(name)
    return name


def decode_message(data: bytes) -> CFMessage:
    cur = _Cursor(data)
    # tag: bare ascii name up to first control char.
    while cur.pos < len(cur.data) and cur.data[cur.pos] not in _TEXT_TERMINATORS:
        cur.pos += 1
    tag = data[: cur.pos].decode("utf-8")
    _check_name(tag)
    msg = CFMessage(tag=tag)
    # positional values: leads RS/DC1/DC2/DC3/DC4
    while cur.peek() in _VALUE_LEAD:
        msg.args.append(_read_value(cur))
    # keyword fields: US keyword value
    while cur.peek() == US:
        cur.read()
        k = _read_keyword(cur)
        msg.kwargs[k] = _read_value(cur)
    # optional body section
    if cur.peek() == GS:
        cur.read()
        body_start = cur.pos
        while cur.pos < len(cur.data) and cur.data[cur.pos] not in _BODY_TERMINATORS:
            cur.pos += 1
        msg.body = _decode_body(data[body_start : cur.pos])
        while cur.peek() == US:
            cur.read()
            k = _read_keyword(cur)
            msg.meta[k] = _read_value(cur)
    if cur.pos != len(cur.data):
        raise ValueError(f"trailing bytes after message at offset {cur.pos}")
    return msg


def decode_file(data: bytes) -> list[CFMessage]:
    if not data:
        return []
    msgs = []
    start = 0
    for i, b in enumerate(data):
        if b == FS:
            msgs.append(decode_message(data[start:i]))
            start = i + 1
    if start != len(data):
        # No trailing FS — treat remaining as last message.
        msgs.append(decode_message(data[start:]))
    return msgs
