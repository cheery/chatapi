"""Scripted single round-trip against a local chatapi server.

Usage:
    python examples/simple_client.py [/path/to/socket]

Default socket path is /tmp/chatapi.sock.
"""
from __future__ import annotations

import asyncio
import sys

from chatapi import chatfmt
from chatapi.client import ChatClient


async def main(socket_path: str) -> None:
    client = await ChatClient.connect_unix(socket_path)

    print(">> handshake")
    resp = await client.handshake("0")
    print(f"   <- {resp.name}{resp.args}")
    if resp.name != "version!":
        print("server rejected handshake")
        await client.bye()
        return

    print(">> models? chat")
    mid = client.next_id()
    models: list[tuple[str, str]] = []
    async for m in client.request("models?", str(mid), payload=b"chat"):
        if m.name in ("models*!", "models!") and m.args[1]:
            models.append((m.args[1], m.args[2]))
    print(f"   <- {len(models)} model(s)")
    for name, flavor in models[:5]:
        print(f"      - {name} ({flavor})")
    if len(models) > 5:
        print(f"      ...and {len(models) - 5} more")

    print(">> default_model? chat")
    mid = client.next_id()
    default_name = None
    async for m in client.request("default_model?", str(mid), "chat", payload=b""):
        if m.name == "default_model!":
            default_name = m.args[1]
    print(f"   <- {default_name}")

    chosen = default_name or (models[0][0] if models else None)
    if chosen is None:
        print("no model available; aborting")
        await client.bye()
        return

    print(">> chat?")
    mid = client.next_id()
    session_id = None
    async for m in client.request("chat?", str(mid), payload=b""):
        if m.name == "chat!":
            session_id = m.args[1]
    assert session_id is not None
    print(f"   <- session {session_id}")

    print(f">> model? {chosen}")
    mid = client.next_id()
    async for m in client.request("model?", str(mid), session_id, chosen, payload=b""):
        print(f"   <- {m.name}{m.args}")

    print(">> message? user(\"Say hello in one short sentence.\")")
    user_msg = chatfmt.encode_file([chatfmt.user("Say hello in one short sentence.")])
    mid = client.next_id()
    async for m in client.request("message?", str(mid), session_id, payload=user_msg):
        print(f"   <- {m.name}{m.args}")

    print(">> complete?")
    mid = client.next_id()
    print("   <- ", end="", flush=True)
    pieces: list[str] = []
    async for m in client.request("complete?", str(mid), session_id, payload=b""):
        if m.name == "complete*!":
            chunk = m.payload.decode("utf-8", errors="replace")
            pieces.append(chunk)
            print(chunk, end="", flush=True)
        elif m.name == "complete!":
            print()  # newline
        elif m.name in ("aborted!", "refuse!", "context_limit_reached!"):
            print(f"\n   !! {m.name} {m.args} {m.payload!r}")
    print(f"   (assembled {len(''.join(pieces))} chars)")

    print(">> end?")
    mid = client.next_id()
    async for m in client.request("end?", str(mid), session_id, payload=b""):
        print(f"   <- {m.name}{m.args}")

    print(">> bye")
    await client.bye()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/chatapi.sock"
    asyncio.run(main(path))
