import pytest

from chatapi.protocol import Message, decode, encode


def _roundtrip(msg):
    assert decode(encode(msg)) == msg


def test_request_roundtrip():
    msg = Message(name="version?", args=("0", "0"), payload=b"")
    _roundtrip(msg)
    assert msg.kind == "request"
    assert msg.base_name == "version"
    assert msg.message_id == 0


def test_stream_request():
    msg = Message(name="message*?", args=("4", "abc"), payload=b"some bytes")
    _roundtrip(msg)
    assert msg.kind == "stream_request"
    assert msg.base_name == "message"


def test_stream_response():
    msg = Message(name="complete*!", args=("8", "abc"), payload=b"hi")
    _roundtrip(msg)
    assert msg.kind == "stream_response"


def test_response():
    msg = Message(name="end!", args=("14", "abc"), payload=b"")
    _roundtrip(msg)
    assert msg.kind == "response"


def test_bare_bye():
    msg = Message(name="bye", args=(), payload=b"")
    _roundtrip(msg)
    assert msg.kind == "bare"
    assert msg.message_id is None


def test_bye_with_reason():
    msg = Message(name="bye", args=(), payload=b"unsupported version 0")
    _roundtrip(msg)


def test_payload_can_contain_any_byte():
    msg = Message(name="data!", args=("2",), payload=bytes(range(256)))
    _roundtrip(msg)


def test_arg_with_control_char_rejected():
    with pytest.raises(ValueError):
        encode(Message(name="bad", args=("a\x1fb",)))


def test_decode_missing_gs_rejected():
    with pytest.raises(ValueError):
        decode(b"version?\x1e0")


def test_decode_empty_name_rejected():
    with pytest.raises(ValueError):
        decode(b"\x1d")
