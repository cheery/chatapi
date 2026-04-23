"""Aggregate multiple per-vendor backends and dispatch by name prefix."""
from __future__ import annotations

import logging
from typing import AsyncIterator, Callable

from .auth import Vendor
from .backend import Backend, ContextLimitUnknown, ModelInfo
from .chatfmt import CFMessage

log = logging.getLogger(__name__)

VENDOR_SEP = "/"

# Mapping of "api" name in auth.json -> factory(vendor) -> Backend.
BackendFactory = Callable[[Vendor], Backend]
_REGISTRY: dict[str, BackendFactory] = {}


def register(api: str, factory: BackendFactory) -> None:
    _REGISTRY[api] = factory


def _builtin_registry() -> None:
    # Lazy import so the test suite can stub this without importing anthropic.
    from .anthropic_backend import AnthropicBackend

    register("anthropic", lambda v: AnthropicBackend(api_key=v.key, base_url=v.api_url))


_builtin_registry()


def build(vendors: list[Vendor]) -> "MultiBackend":
    backends: dict[str, Backend] = {}
    for v in vendors:
        factory = _REGISTRY.get(v.api)
        if factory is None:
            log.warning("vendor %s: unknown api %r, skipping", v.name, v.api)
            continue
        backends[v.name] = factory(v)
        log.info("vendor %s ready (api=%s, base_url=%s)", v.name, v.api, v.api_url or "default")
    if not backends:
        raise RuntimeError("no usable vendors found in auth.json")
    return MultiBackend(backends)


def _split(name: str) -> tuple[str, str]:
    if VENDOR_SEP not in name:
        raise ValueError(f"model name {name!r} must be prefixed with vendor (vendor{VENDOR_SEP}model)")
    vendor, _, model = name.partition(VENDOR_SEP)
    return vendor, model


class MultiBackend(Backend):
    def __init__(self, backends: dict[str, Backend]):
        self._backends = backends

    def vendors(self) -> list[str]:
        return list(self._backends.keys())

    def _resolve(self, name: str) -> tuple[Backend, str]:
        vendor, native = _split(name)
        backend = self._backends.get(vendor)
        if backend is None:
            raise KeyError(f"unknown vendor {vendor!r}")
        return backend, native

    async def list_models(self, flavor: str | None) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for vendor, backend in self._backends.items():
            try:
                models = await backend.list_models(flavor)
            except Exception as e:
                log.warning("vendor %s: list_models failed: %s", vendor, e)
                continue
            for m in models:
                out.append(ModelInfo(name=f"{vendor}{VENDOR_SEP}{m.name}", flavor=m.flavor))
        return out

    def default_model(self, flavor: str | None) -> ModelInfo | None:
        for vendor, backend in self._backends.items():
            info = backend.default_model(flavor)
            if info is not None:
                return ModelInfo(name=f"{vendor}{VENDOR_SEP}{info.name}", flavor=info.flavor)
        return None

    async def stream_complete(
        self, model: str, messages: list[CFMessage]
    ) -> AsyncIterator[tuple[str, str]]:
        backend, native = self._resolve(model)
        async for chunk in backend.stream_complete(native, messages):
            yield chunk

    async def context_limit(
        self, model: str, messages: list[CFMessage]
    ) -> tuple[int, int]:
        backend, native = self._resolve(model)
        return await backend.context_limit(native, messages)
