"""End-to-end smoke test of the server with a fake backend (no network)."""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import AsyncIterator

import pytest

from chatapi import chatfmt
from chatapi.backend import Backend, ContextLimitUnknown, ModelInfo
from chatapi.client import ChatClient
from chatapi.server import serve_unix


class FakeBackend(Backend):
    def __init__(self):
        self.calls: list[tuple[str, list[chatfmt.CFMessage]]] = []

    async def list_models(self, flavor):
        return [ModelInfo("fake-fast", "chat"), ModelInfo("fake-smart", "chat")]

    def default_model(self, flavor):
        return ModelInfo("fake-smart", "chat")

    async def stream_complete(self, model, messages) -> AsyncIterator[str]:
        self.calls.append((model, list(messages)))
        for chunk in ("hel", "lo, ", "world!"):
            yield chunk

    async def context_limit(self, model, messages):
        raise ContextLimitUnknown(model)


@pytest.fixture
async def server_socket():
    backend = FakeBackend()
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


async def test_full_round_trip(server_socket):
    sock, backend = server_socket
    client = await ChatClient.connect_unix(sock)

    # handshake
    resp = await client.handshake("0")
    assert resp.name == "version!"

    # models?
    msgs = await _drain(client.request("models?", "2", payload=b"chat"))
    names = [m.args[1] for m in msgs if m.args[1]]
    assert names == ["fake-fast", "fake-smart"]
    assert msgs[-1].name == "models!"

    # default_model?
    msgs = await _drain(client.request("default_model?", "4", "chat", payload=b""))
    assert msgs[-1].name == "default_model!"
    assert msgs[-1].args[1] == "fake-smart"

    # chat?
    msgs = await _drain(client.request("chat?", "6", payload=b""))
    sid = msgs[-1].args[1]

    # model?
    msgs = await _drain(client.request("model?", "8", sid, "fake-smart", payload=b""))
    assert msgs[-1].name == "model!"

    # message?
    user_blob = chatfmt.encode_file([chatfmt.user("hi")])
    msgs = await _drain(client.request("message?", "10", sid, payload=user_blob))
    assert msgs[-1].name == "message!"

    # complete?
    msgs = await _drain(client.request("complete?", "12", sid, payload=b""))
    deltas = [m.payload.decode() for m in msgs if m.name == "complete*!"]
    assert "".join(deltas) == "hello, world!"
    assert msgs[-1].name == "complete!"
    # backend received the user message
    model, history = backend.calls[-1]
    assert model == "fake-smart"
    assert history[-1].tag == "user"
    assert history[-1].body == "hi"

    # context_limit? -> unknown
    msgs = await _drain(client.request("context_limit?", "14", sid, payload=b""))
    assert msgs[-1].name == "context_limit_unknown!"

    # end?
    msgs = await _drain(client.request("end?", "16", sid, payload=b""))
    assert msgs[-1].name == "end!"

    await client.bye()


async def test_unknown_call_refuses(server_socket):
    sock, _ = server_socket
    client = await ChatClient.connect_unix(sock)
    await client.handshake("0")
    msgs = await _drain(client.request("nonsense?", "2", payload=b""))
    assert msgs[-1].name == "refuse!"
    await client.bye()


async def test_complete_aborts_cleanly(server_socket):
    sock, _ = server_socket

    class SlowBackend(FakeBackend):
        async def stream_complete(self, model, messages):
            for chunk in ("a", "b", "c"):
                yield chunk
                await asyncio.sleep(0.05)

    # Replace backend by spinning up a fresh server with a slow one.
    slow = SlowBackend()
    tmp = tempfile.mkdtemp()
    sock2 = os.path.join(tmp, "slow.sock")
    server = await serve_unix(slow, sock2)
    try:
        client = await ChatClient.connect_unix(sock2)
        await client.handshake("0")

        # set up session
        msgs = await _drain(client.request("chat?", "2", payload=b""))
        sid = msgs[-1].args[1]
        await _drain(client.request("model?", "4", sid, "fake-smart", payload=b""))

        # start completion + abort after first chunk
        await client.send("complete?", "6", sid, payload=b"")
        first = await client.recv()
        assert first.name == "complete*!"
        await client.send("abort?", "8", sid, payload=b"")

        # expect abort! and aborted! (in either order)
        seen = []
        for _ in range(4):
            try:
                m = await asyncio.wait_for(client.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                break
            seen.append(m.name)
            if "aborted!" in seen and "abort!" in seen:
                break
        assert "abort!" in seen
        assert "aborted!" in seen

        await client.bye()
    finally:
        server.close()
        await server.wait_closed()
        if os.path.exists(sock2):
            os.unlink(sock2)
        os.rmdir(tmp)
