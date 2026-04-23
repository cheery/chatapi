"""Auth-file loader + MultiBackend prefix dispatch."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from chatapi import auth, multi_backend
from chatapi.backend import Backend, ContextLimitUnknown, ModelInfo
from chatapi.chatfmt import CFMessage, user
from chatapi.multi_backend import MultiBackend


def _write_auth(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "auth.json"
    p.write_text(json.dumps(data))
    return p


def test_auth_load_basic(tmp_path):
    p = _write_auth(tmp_path, {
        "personal": {"api": "anthropic", "key": "k1"},
        "work": {"api": "anthropic", "key": "k2"},
    })
    vendors = auth.load(p)
    assert [v.name for v in vendors] == ["personal", "work"]
    assert vendors[0].api == "anthropic"
    assert vendors[1].key == "k2"


def test_auth_invalid_name(tmp_path):
    p = _write_auth(tmp_path, {"bad name!": {"api": "anthropic", "key": "k"}})
    with pytest.raises(ValueError):
        auth.load(p)


def test_auth_missing_fields(tmp_path):
    p = _write_auth(tmp_path, {"v": {"api": "anthropic"}})
    with pytest.raises(ValueError):
        auth.load(p)


def test_auth_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        auth.load(tmp_path / "nope.json")


# --- MultiBackend ---


class _Stub(Backend):
    def __init__(self, models, default, deltas, tag):
        self._models = models
        self._default = default
        self._deltas = deltas
        self._tag = tag
        self.last_call = None

    async def list_models(self, flavor):
        return list(self._models)

    def default_model(self, flavor):
        return self._default

    async def stream_complete(self, model, messages):
        self.last_call = (model, list(messages))
        for d in self._deltas:
            yield f"{self._tag}:{d}"

    async def context_limit(self, model, messages):
        raise ContextLimitUnknown(model)


def _multi():
    a = _Stub([ModelInfo("fast"), ModelInfo("smart")], ModelInfo("smart"), ["a", "b"], "A")
    b = _Stub([ModelInfo("alt")], ModelInfo("alt"), ["x"], "B")
    return MultiBackend({"v1": a, "v2": b}), a, b


@pytest.mark.asyncio
async def test_list_models_prefixes_names():
    multi, _, _ = _multi()
    names = [m.name for m in await multi.list_models("chat")]
    assert names == ["v1/fast", "v1/smart", "v2/alt"]


def test_default_model_prefixed():
    multi, _, _ = _multi()
    info = multi.default_model("chat")
    assert info is not None and info.name == "v1/smart"


@pytest.mark.asyncio
async def test_stream_complete_dispatches_by_prefix():
    multi, a, b = _multi()
    chunks = []
    async for c in multi.stream_complete("v2/alt", [user("hi")]):
        chunks.append(c)
    assert chunks == ["B:x"]
    assert b.last_call is not None and b.last_call[0] == "alt"
    assert a.last_call is None


@pytest.mark.asyncio
async def test_unknown_vendor_raises():
    multi, _, _ = _multi()
    with pytest.raises(KeyError):
        async for _ in multi.stream_complete("missing/foo", []):
            pass


@pytest.mark.asyncio
async def test_naked_model_name_rejected():
    multi, _, _ = _multi()
    with pytest.raises(ValueError):
        async for _ in multi.stream_complete("no-prefix", []):
            pass


def test_build_skips_unknown_api(tmp_path, caplog):
    multi_backend.register("dummy", lambda key: _Stub([ModelInfo("only")], ModelInfo("only"), ["d"], "D"))
    vendors = [
        auth.Vendor("good", "dummy", "k"),
        auth.Vendor("bad", "made-up", "k"),
    ]
    backend = multi_backend.build(vendors)
    assert backend.vendors() == ["good"]


def test_build_no_vendors_raises():
    with pytest.raises(RuntimeError):
        multi_backend.build([])
