"""Backend abstraction for completion providers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from .chatfmt import CFMessage


class ContextLimitError(Exception):
    """Raised by a backend when the request exceeds the model's context window."""


class ContextLimitUnknown(Exception):
    """Raised by a backend that cannot report context usage for a session."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    flavor: str = "chat"


class Backend(Protocol):
    async def list_models(self, flavor: str | None) -> list[ModelInfo]: ...

    def default_model(self, flavor: str | None) -> ModelInfo | None: ...

    def stream_complete(
        self, model: str, messages: list[CFMessage], usage_out: dict | None = None,
    ) -> AsyncIterator[CFMessage]:
        """Yield chatfmt chunks. A chunk with a non-'_' tag opens a new
        block (carrying its tag, args, kwargs, and optionally an initial
        body). A chunk with tag '_' (see chatfmt.cont) extends the most
        recent open block — its body is appended and its meta merged in.

        If usage_out is supplied, the backend may write known usage figures
        into it (e.g. {"output_tokens": N}) for the server to surface on the
        wire. Backends that cannot report usage simply leave it empty."""
        ...

    async def context_limit(
        self, model: str, messages: list[CFMessage]
    ) -> tuple[int, int]:
        """Return (used, total) tokens for the given session, or raise ContextLimitUnknown."""
        ...
