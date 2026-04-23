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
        self, model: str, messages: list[CFMessage]
    ) -> AsyncIterator[str]: ...

    async def context_limit(
        self, model: str, messages: list[CFMessage]
    ) -> tuple[int, int]:
        """Return (used, total) tokens for the given session, or raise ContextLimitUnknown."""
        ...
