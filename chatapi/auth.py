"""Loader for ~/.chatapi/auth.json."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("~/.chatapi/auth.json").expanduser()
_VENDOR_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Vendor:
    name: str
    api: str
    key: str
    api_url: str | None = None


def load(path: Path | str | None = None) -> list[Vendor]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"auth file not found: {p}")
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top-level must be an object mapping vendor name to entry")
    out: list[Vendor] = []
    for name, entry in raw.items():
        if not _VENDOR_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid vendor name {name!r}: must match [A-Za-z0-9_.-]+")
        if not isinstance(entry, dict):
            raise ValueError(f"vendor {name!r}: entry must be an object")
        api = entry.get("api")
        key = entry.get("key")
        if not isinstance(api, str) or not api:
            raise ValueError(f"vendor {name!r}: missing or invalid 'api'")
        if not isinstance(key, str) or not key:
            raise ValueError(f"vendor {name!r}: missing or invalid 'key'")
        api_url = entry.get("api_url")
        if api_url is not None and (not isinstance(api_url, str) or not api_url):
            raise ValueError(f"vendor {name!r}: 'api_url' must be a non-empty string")
        out.append(Vendor(name=name, api=api, key=key, api_url=api_url))
    return out


def ensure_secure(path: Path | str | None = None) -> None:
    """Best-effort warn (or raise) if the auth file is world-readable."""
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        # Not fatal — some environments use group-shared keyrings — but loud.
        import logging

        logging.getLogger(__name__).warning(
            "auth file %s has permissive mode %o; consider chmod 600", p, mode
        )
