"""Run the chatapi server over a unix socket and/or loopback TCP."""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import logging
import os

from .anthropic_backend import AnthropicBackend
from .server import serve_tcp, serve_unix


def _parse_tcp(spec: str) -> tuple[str, int]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError("TCP spec must be HOST:PORT")
    host, _, port_text = spec.rpartition(":")
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise argparse.ArgumentTypeError(f"TCP host must be an IP literal, got {host!r}")
    if not addr.is_loopback:
        raise argparse.ArgumentTypeError(
            "refusing non-loopback TCP bind: chatapi has no auth in v0"
        )
    try:
        port = int(port_text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid TCP port: {port_text!r}")
    return host, port


async def _run(args: argparse.Namespace) -> None:
    backend = AnthropicBackend()
    servers = []
    if args.socket:
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        servers.append(await serve_unix(backend, args.socket))
        logging.info("listening on unix:%s", args.socket)
    if args.tcp:
        host, port = args.tcp
        servers.append(await serve_tcp(backend, host, port))
        logging.info("listening on tcp:%s:%d", host, port)
    if not servers:
        raise SystemExit("specify --socket and/or --tcp")
    try:
        await asyncio.gather(*(s.serve_forever() for s in servers))
    except asyncio.CancelledError:
        pass
    finally:
        for s in servers:
            s.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="chatapi-server")
    parser.add_argument("--socket", help="unix socket path to listen on")
    parser.add_argument("--tcp", type=_parse_tcp, help="HOST:PORT (loopback only)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
