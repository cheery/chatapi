"""Anthropic SDK adapter for the chatapi server."""
from __future__ import annotations

import os
from typing import AsyncIterator

import anthropic

from .backend import Backend, ContextLimitError, ContextLimitUnknown, ModelInfo
from .chatfmt import CFMessage

DEFAULT_CHAT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 8192
# Extended-thinking budget. Models without thinking support will reject this;
# the caller can adjust the model selection or disable thinking in the future.
DEFAULT_THINKING_BUDGET = 1024


def _to_anthropic(messages: list[CFMessage]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        body = m.body or ""
        if m.tag == "system":
            system_parts.append(body)
        elif m.tag == "user":
            out.append({"role": "user", "content": body})
        elif m.tag == "assistant":
            out.append({"role": "assistant", "content": body})
        elif m.tag == "think":
            # TODO(v0): chatfmt 'think' messages are dropped on the way to Anthropic.
            continue
        else:
            raise ValueError(f"unsupported chatfmt tag for Anthropic: {m.tag}")
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


class AnthropicBackend(Backend):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        kwargs: dict = {"api_key": api_key or os.environ.get("ANTHROPIC_API_KEY")}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def list_models(self, flavor: str | None) -> list[ModelInfo]:
        if flavor not in (None, "", "chat"):
            return []
        out: list[ModelInfo] = []
        async for m in self._client.models.list():
            out.append(ModelInfo(name=m.id, flavor="chat"))
        return out

    def default_model(self, flavor: str | None) -> ModelInfo | None:
        if flavor not in (None, "", "chat"):
            return None
        return ModelInfo(name=DEFAULT_CHAT_MODEL, flavor="chat")

    async def stream_complete(
        self, model: str, messages: list[CFMessage]
    ) -> AsyncIterator[tuple[str, str]]:
        system, msgs = _to_anthropic(messages)
        kwargs: dict = {
            "model": model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": msgs,
            "thinking": {"type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET},
        }
        if system is not None:
            kwargs["system"] = system
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "thinking_delta":
                        yield ("think", delta.thinking)
                    elif dtype == "text_delta":
                        yield ("assistant", delta.text)
                    # signature_delta and other delta types are ignored for v0.
        except anthropic.BadRequestError as e:
            msg = str(e).lower()
            if "context" in msg or "too long" in msg or "max_tokens" in msg:
                raise ContextLimitError(str(e)) from e
            raise

    async def context_limit(
        self, model: str, messages: list[CFMessage]
    ) -> tuple[int, int]:
        # TODO(v0): Anthropic API doesn't expose used/total tokens for an
        # in-progress chat session synchronously. Surface unknown to the client.
        raise ContextLimitUnknown(model)
