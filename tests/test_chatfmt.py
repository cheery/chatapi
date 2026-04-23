import pytest

from chatapi.chatfmt import (
    CFMessage,
    assistant,
    decode_file,
    decode_message,
    encode_file,
    encode_message,
    make,
    system,
    user,
)


def _roundtrip(msg):
    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert decoded == msg


def test_simple_messages_roundtrip():
    _roundtrip(system("you are helpful"))
    _roundtrip(user("hello"))
    _roundtrip(assistant("hi there"))


def test_body_with_whitespace_passes_through():
    msg = user("line1\nline2\twith tab\rand cr")
    encoded = encode_message(msg)
    # \n \t \r should be literal in the encoded body
    assert b"\nline2" in encoded
    assert b"\twith" in encoded
    _roundtrip(msg)


def test_body_escapes_low_bytes_and_backslash():
    msg = assistant("a\x01b\x05\\c")
    encoded = encode_message(msg)
    assert b"\\01" in encoded
    assert b"\\05" in encoded
    assert b"\\5c" in encoded
    _roundtrip(msg)


def test_args_all_value_types():
    msg = make(
        "demo",
        "text",
        42,
        3.5,
        b"\x00\x01raw",
        True,
        False,
        None,
        ...,
    )
    _roundtrip(msg)


def test_kwargs_and_meta():
    msg = make("demo", id=7, name="alice", _trace="abc", _retry=2)
    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert decoded.kwargs == {"id": 7, "name": "alice"}
    assert decoded.meta == {"trace": "abc", "retry": 2}


def test_invalid_name_rejected():
    with pytest.raises(ValueError):
        encode_message(CFMessage(tag="1bad"))


def test_text_with_control_char_rejected():
    with pytest.raises(ValueError):
        encode_message(CFMessage(tag="t", args=["bad\x1ftext"]))


def test_file_roundtrip():
    msgs = [system("sys"), user("hi"), assistant("yo")]
    encoded = encode_file(msgs)
    assert encoded.endswith(b"\x1c")
    assert decode_file(encoded) == msgs


def test_file_without_trailing_fs():
    msgs = [user("a"), assistant("b")]
    encoded = encode_file(msgs).rstrip(b"\x1c")
    assert decode_file(encoded) == msgs


def test_blob_with_control_bytes():
    blob = bytes(range(32))
    msg = make("data", blob)
    _roundtrip(msg)


def test_bool_not_confused_with_int():
    msg = make("flag", True, 1)
    decoded = decode_message(encode_message(msg))
    assert decoded.args[0] is True
    assert decoded.args[1] == 1
    assert isinstance(decoded.args[1], int) and not isinstance(decoded.args[1], bool)
