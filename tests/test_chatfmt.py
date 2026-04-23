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


# --- streaming continuation (tag '_') ---


from chatapi.chatfmt import CONT_TAG, cont, merge_chunks


def test_cont_builder_basic():
    c = cont(content="more")
    assert c.tag == CONT_TAG
    assert c.body == "more"
    assert c.meta == {}


def test_cont_builder_with_meta():
    c = cont(content="x", _tokens=12, _stop="end_turn")
    assert c.meta == {"tokens": 12, "stop": "end_turn"}


def test_cont_rejects_non_meta_kwarg():
    with pytest.raises(ValueError):
        cont(content="x", random="nope")


def test_merge_chunks_appends_body():
    chunks = [
        make("assistant", content="Hel"),
        cont(content="lo, "),
        cont(content="world!"),
    ]
    blocks = merge_chunks(chunks)
    assert len(blocks) == 1
    assert blocks[0].tag == "assistant"
    assert blocks[0].body == "Hello, world!"


def test_merge_chunks_accumulates_meta():
    chunks = [
        make("assistant", content="hi", _model="m1"),
        cont(content=" there", _input_tokens=5),
        cont(_output_tokens=12),
        cont(_input_tokens=7),  # later writes win
    ]
    blocks = merge_chunks(chunks)
    assert blocks[0].body == "hi there"
    assert blocks[0].meta == {
        "model": "m1",
        "input_tokens": 7,
        "output_tokens": 12,
    }


def test_merge_chunks_opens_new_block_on_tag_change():
    chunks = [
        make("think", content="thinking"),
        cont(content=" more"),
        make("assistant", content="answer"),
        cont(content="!"),
    ]
    blocks = merge_chunks(chunks)
    assert [(b.tag, b.body) for b in blocks] == [
        ("think", "thinking more"),
        ("assistant", "answer!"),
    ]


def test_merge_chunks_preserves_first_block_args_kwargs():
    chunks = [
        make("tool_use", "calculator", lang="python", _ref="r1"),
        cont(content="2+2"),
    ]
    blocks = merge_chunks(chunks)
    assert blocks[0].args == ["calculator"]
    assert blocks[0].kwargs == {"lang": "python"}
    assert blocks[0].meta == {"ref": "r1"}
    assert blocks[0].body == "2+2"


def test_merge_chunks_isolates_block_from_chunk():
    """Mutating a returned block must not change the input chunk."""
    chunk = make("assistant", content="x", _tag="a")
    blocks = merge_chunks([chunk])
    blocks[0].body = "mutated"
    blocks[0].meta["tag"] = "b"
    assert chunk.body == "x"
    assert chunk.meta == {"tag": "a"}


def test_merge_chunks_continuation_before_block_raises():
    with pytest.raises(ValueError):
        merge_chunks([cont(content="oops")])


def test_cont_chunk_roundtrips_on_wire():
    c = cont(content="hi", _tokens=42)
    decoded = decode_message(encode_message(c))
    assert decoded == c
