import asyncio
import os
import pytest

from chatapi.wire import (
    VLQ_MAX,
    decode_vlq_bytes,
    encode_vlq,
    read_frame,
    read_vlq,
    write_frame,
)


@pytest.mark.parametrize("n", [0, 1, 127, 128, 200, 16383, 16384, 2**21 - 1, 2**21, VLQ_MAX])
def test_vlq_roundtrip(n):
    encoded = encode_vlq(n)
    decoded, consumed = decode_vlq_bytes(encoded)
    assert decoded == n
    assert consumed == len(encoded)


def test_vlq_known_encodings():
    assert encode_vlq(0) == b"\x00"
    assert encode_vlq(127) == b"\x7f"
    assert encode_vlq(128) == b"\x81\x00"
    assert encode_vlq(200) == b"\x81\x48"


def test_vlq_overflow():
    with pytest.raises(ValueError):
        encode_vlq(VLQ_MAX + 1)
    with pytest.raises(ValueError):
        encode_vlq(-1)


def test_vlq_decode_overrun():
    with pytest.raises(ValueError):
        decode_vlq_bytes(b"\x80\x80\x80\x80")


def _pipe():
    """Return (reader, writer) pair backed by an in-memory transport."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader(loop=loop)
    protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
    transport = _MemTransport(protocol)
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    return reader, writer


class _MemTransport(asyncio.Transport):
    def __init__(self, protocol):
        super().__init__()
        self._protocol = protocol
        self._closing = False

    def write(self, data):
        self._protocol.data_received(bytes(data))

    def close(self):
        if not self._closing:
            self._closing = True
            self._protocol.eof_received()

    def is_closing(self):
        return self._closing

    def get_write_buffer_size(self):
        return 0


@pytest.mark.asyncio
async def test_frame_roundtrip():
    reader, writer = _pipe()
    payload = os.urandom(523)
    await write_frame(writer, payload)
    got = await read_frame(reader)
    assert got == payload


@pytest.mark.asyncio
async def test_frame_empty():
    reader, writer = _pipe()
    await write_frame(writer, b"")
    got = await read_frame(reader)
    assert got == b""


@pytest.mark.asyncio
async def test_read_vlq_async():
    reader, writer = _pipe()
    writer.write(encode_vlq(123456))
    await writer.drain()
    assert await read_vlq(reader) == 123456
