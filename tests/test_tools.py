"""Tests for tool/call/ret chatfmt messages and the Anthropic adapter."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import AsyncIterator

import pytest

from chatapi import chatfmt
from chatapi.backend import Backend, ContextLimitUnknown, ModelInfo
from chatapi.chatfmt import CFMessage, call, cont, merge_chunks, ret, tool
from chatapi.client import ChatClient
from chatapi.server import serve_unix


# --- chatfmt round-trip tests ---


def _roundtrip(msg):
    encoded = chatfmt.encode_message(msg)
    decoded = chatfmt.decode_message(encoded)
    assert decoded == msg


def test_tool_roundtrip():
    schema = json.dumps({"type": "object", "properties": {"q": {"type": "string"}}})
    _roundtrip(tool("search", schema, description="Search the web"))


def test_tool_minimal():
    _roundtrip(tool("echo", "{}"))


def test_call_roundtrip():
    _roundtrip(call("toolu_01", "search", '{"q":"hello"}'))


def test_call_empty_body():
    _roundtrip(call("toolu_02", "echo"))


def test_ret_roundtrip():
    _roundtrip(ret("toolu_01", '{"result": 42}'))


def test_ret_error():
    msg = ret("toolu_01", "timeout", error=True)
    assert msg.meta == {"error": True}
    _roundtrip(msg)


def test_tool_file_roundtrip():
    schema = json.dumps({"type": "object"})
    msgs = [
        chatfmt.system("you are helpful"),
        tool("search", schema, description="Search"),
        chatfmt.user("find something"),
        call("toolu_01", "search", '{"q":"x"}'),
        ret("toolu_01", '{"hits":[]}'),
        chatfmt.assistant("here are the results"),
    ]
    encoded = chatfmt.encode_file(msgs)
    decoded = chatfmt.decode_file(encoded)
    assert decoded == msgs


def test_merge_chunks_with_call():
    chunks = [
        CFMessage(tag="assistant", body="Let me search. "),
        call("toolu_01", "search", '{"q":'),
        cont(content='"ferrofluid"}'),
    ]
    blocks = merge_chunks(chunks)
    assert len(blocks) == 2
    assert blocks[0].tag == "assistant"
    assert blocks[0].body == "Let me search. "
    assert blocks[1].tag == "call"
    assert blocks[1].args == ["toolu_01", "search"]
    assert blocks[1].body == '{"q":"ferrofluid"}'


def test_merge_chunks_multiple_calls():
    chunks = [
        call("id1", "f", '{"a":1}'),
        call("id2", "g", '{"b":2}'),
    ]
    blocks = merge_chunks(chunks)
    assert len(blocks) == 2
    assert blocks[0].args == ["id1", "f"]
    assert blocks[1].args == ["id2", "g"]


def test_json_with_backslash_in_body():
    body = json.dumps({"path": "C:\\Users\\test"})
    msg = call("id1", "read", body)
    _roundtrip(msg)


# --- Anthropic adapter _to_anthropic tests ---


def _translate(messages):
    pytest.importorskip("anthropic")
    from chatapi.anthropic_backend import _to_anthropic
    return _to_anthropic(messages)


def test_to_anthropic_extracts_tools():
    schema = json.dumps({"type": "object", "properties": {"q": {"type": "string"}}})
    msgs = [
        chatfmt.system("be helpful"),
        tool("search", schema, description="Search"),
        chatfmt.user("hello"),
    ]
    system, api_msgs, tools = _translate(msgs)
    assert system == "be helpful"
    assert len(tools) == 1
    assert tools[0]["name"] == "search"
    assert tools[0]["description"] == "Search"
    assert json.loads(schema) == tools[0]["input_schema"]
    assert api_msgs == [{"role": "user", "content": "hello"}]


def test_to_anthropic_call_maps_to_tool_use():
    msgs = [
        chatfmt.user("hi"),
        chatfmt.assistant("Let me look."),
        call("toolu_01", "search", '{"q":"x"}'),
    ]
    _, api_msgs, tools = _translate(msgs)
    assert tools == []
    assert len(api_msgs) == 2
    assert api_msgs[0] == {"role": "user", "content": "hi"}
    assert api_msgs[1]["role"] == "assistant"
    assert api_msgs[1]["content"] == [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"q": "x"}},
    ]


def test_to_anthropic_ret_maps_to_tool_result():
    msgs = [
        chatfmt.user("hi"),
        chatfmt.assistant("Let me look."),
        call("toolu_01", "search", '{"q":"x"}'),
        ret("toolu_01", '{"hits":[]}'),
    ]
    _, api_msgs, _ = _translate(msgs)
    # assistant turn + user turn with tool_result
    assert len(api_msgs) == 3
    assert api_msgs[0]["role"] == "user"
    assert api_msgs[1]["role"] == "assistant"
    assert api_msgs[2]["role"] == "user"
    assert api_msgs[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": '{"hits":[]}'},
    ]


def test_to_anthropic_ret_error():
    msgs = [
        call("toolu_01", "f", "{}"),
        ret("toolu_01", "timeout", error=True),
    ]
    _, api_msgs, _ = _translate(msgs)
    assert api_msgs[-1]["content"][0]["is_error"] is True


def test_to_anthropic_pure_text_assistant():
    msgs = [
        chatfmt.user("hi"),
        chatfmt.assistant("hello!"),
    ]
    _, api_msgs, _ = _translate(msgs)
    assert api_msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]


# --- e2e with tool call round-trip ---


class ToolFakeBackend(Backend):
    """Fake backend that simulates a tool call then a text response."""

    def __init__(self):
        self.calls: list[tuple[str, list[CFMessage]]] = []

    async def list_models(self, flavor):
        return [ModelInfo("fake", "chat")]

    def default_model(self, flavor):
        return ModelInfo("fake", "chat")

    async def stream_complete(self, model, messages, usage_out=None):
        self.calls.append((model, list(messages)))
        has_tools = any(m.tag == "tool" for m in messages)
        has_ret = any(m.tag == "ret" for m in messages)

        if has_tools and not has_ret:
            # First turn: model decides to call a tool.
            yield CFMessage(tag="think", body="thinking...")
            yield call("toolu_01", "search", json.dumps({"q": "ferrofluid"}))
        elif has_ret:
            # Second turn: model sees result and responds.
            yield CFMessage(tag="assistant", body="Here are the results.")
        else:
            yield CFMessage(tag="assistant", body="no tools")

        if usage_out is not None:
            usage_out["output_tokens"] = 5

    async def context_limit(self, model, messages):
        raise ContextLimitUnknown(model)


@pytest.fixture
async def tool_server():
    backend = ToolFakeBackend()
    tmp = tempfile.mkdtemp()
    sock = os.path.join(tmp, "chatapi.sock")
    server = await serve_unix(backend, sock)
    try:
        yield sock, backend
    finally:
        server.close()
        await server.wait_closed()
        if os.path.exists(sock):
            os.unlink(sock)
        os.rmdir(tmp)


async def _drain(client_iter):
    out = []
    async for m in client_iter:
        out.append(m)
    return out


async def test_tool_round_trip(tool_server):
    sock, backend = tool_server
    client = await ChatClient.connect_unix(sock)
    await client.handshake("0")

    # Create session
    msgs = await _drain(client.request("chat?", "2", payload=b""))
    sid = msgs[-1].args[1]
    await _drain(client.request("model?", "4", sid, "fake", payload=b""))

    # Send: system + tool declaration + user message
    schema = json.dumps({"type": "object", "properties": {"q": {"type": "string"}}})
    request_msgs = [
        chatfmt.system("you are helpful"),
        tool("search", schema, description="Search"),
        chatfmt.user("Tell me about ferrofluid"),
    ]
    await _drain(client.request(
        "message?", "6", sid,
        payload=chatfmt.encode_file(request_msgs),
    ))

    # First completion: should yield a call block
    msgs = await _drain(client.request("complete?", "8", sid, payload=b""))
    chunks = [chatfmt.decode_message(m.payload) for m in msgs if m.name == "complete*!"]
    blocks = merge_chunks(chunks)

    # Find the call block
    call_blocks = [b for b in blocks if b.tag == "call"]
    assert len(call_blocks) == 1
    assert call_blocks[0].args[1] == "search"
    assert json.loads(call_blocks[0].body) == {"q": "ferrofluid"}

    # Client sends ret back
    ret_msgs = [ret("toolu_01", json.dumps({"hits": ["ferrofluid article"]}))]
    await _drain(client.request(
        "message?", "10", sid,
        payload=chatfmt.encode_file(ret_msgs),
    ))

    # Second completion: model responds with text
    msgs = await _drain(client.request("complete?", "12", sid, payload=b""))
    chunks = [chatfmt.decode_message(m.payload) for m in msgs if m.name == "complete*!"]
    blocks = merge_chunks(chunks)
    assistant_blocks = [b for b in blocks if b.tag == "assistant"]
    assert len(assistant_blocks) == 1
    assert assistant_blocks[0].body == "Here are the results."

    # Backend received the full history including tool/call/ret
    _, history = backend.calls[-1]
    assert any(m.tag == "ret" for m in history)
    assert any(m.tag == "call" for m in history)

    await client.bye()
