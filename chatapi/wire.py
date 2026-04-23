"""VLQ + length-prefixed frame I/O for chatapi."""
from __future__ import annotations

import asyncio

VLQ_MAX_BITS = 28
VLQ_MAX = (1 << VLQ_MAX_BITS) - 1


def encode_vlq(n: int) -> bytes:
    if n < 0:
        raise ValueError("VLQ value must be non-negative")
    if n > VLQ_MAX:
        raise ValueError(f"VLQ value exceeds {VLQ_MAX_BITS}-bit limit")
    vlq = bytearray()
    vlq.append(n & 127)
    while n > 127:
        n >>= 7
        vlq.append(128 | (n & 127))
    vlq.reverse()
    return bytes(vlq)


def decode_vlq_bytes(data: bytes, offset: int = 0) -> tuple[int, int]:
    value = 0
    i = offset
    end = min(offset + 4, len(data))
    while i < end:
        b = data[i]
        value = (value << 7) | (b & 127)
        i += 1
        if b & 128 == 0:
            return value, i - offset
    raise ValueError("VLQ exceeds 4 bytes or truncated input")


async def read_vlq(reader: asyncio.StreamReader) -> int:
    value = 0
    for _ in range(4):
        chunk = await reader.readexactly(1)
        b = chunk[0]
        value = (value << 7) | (b & 127)
        if b & 128 == 0:
            return value
    raise ValueError("VLQ exceeds 4-byte limit")


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    length = await read_vlq(reader)
    if length == 0:
        return b""
    return await reader.readexactly(length)


async def write_frame(writer: asyncio.StreamWriter, body: bytes) -> None:
    writer.write(encode_vlq(len(body)))
    writer.write(body)
    await writer.drain()
