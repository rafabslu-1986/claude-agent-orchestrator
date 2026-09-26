"""Thin wrapper around the Anthropic SDK.

Kept deliberately small so it can be swapped for a fake/mock implementation
in tests without touching any orchestration logic. Every other module talks
to Claude only through this interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-5"


@dataclass
class ClaudeResponse:
    """Normalized response shape used throughout the codebase.

    text: concatenated text content (may be empty if the turn was pure tool_use)
    tool_calls: list of {"id": str, "name": str, "input": dict}
    stop_reason: raw stop_reason from the API ("end_turn", "tool_use", ...)
    raw: the original SDK message object, kept for debugging
    """

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None
    usage: dict[str, int] | None = None
    """{"input_tokens": int, "output_tokens": int} when the underlying SDK
    response carries usage (every real API call does); None only for
    hand-built responses in tests that don't set it. Etapa 14's per-session
    budget guardrail is built entirely on this field.
    """


class ClaudeClient:
    """Wraps anthropic.Anthropic().messages.create(...) behind a stable API.

    Requires ANTHROPIC_API_KEY in the environment for live calls. Every
    class in this project depends on this wrapper (not on the SDK directly),
    so tests substitute a fake client and never make network calls.
    """

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazily constructed so import-time never requires a key

    def _get_sdk_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Export it or pass api_key= "
                    "explicitly to ClaudeClient(). Tests do not need this: they "
                    "inject a fake client instead."
                )
            import anthropic  # imported lazily so the package is optional for tests

            helicone_key = os.environ.get("HELICONE_API_KEY")
            if helicone_key:
                headers = {"Helicone-Auth": f"Bearer {helicone_key}"}
                # LGPD (Etapa 11): customer messages are personal data, and this
                # proxy sits directly in the request path -- Helicone doesn't
                # receive a "copy" that could be redacted separately, it receives
                # the exact bytes on their way to Claude. Redacting first would
                # mean redacting what Claude sees too, breaking every specialist.
                # The fix is to not retain the content at all: cost and latency
                # (the whole point of Etapa 8) are computed from request timing
                # and token-usage metadata, not from the prompt/response bodies,
                # so omitting them costs nothing but the ability to read
                # customer messages back later in the Helicone dashboard -- which
                # is exactly what should not be there by default. Set
                # HELICONE_LOG_CONTENT=true to opt back in for local debugging.
                if os.environ.get("HELICONE_LOG_CONTENT", "").lower() not in ("1", "true"):
                    headers["Helicone-Omit-Request"] = "true"
                    headers["Helicone-Omit-Response"] = "true"
                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    base_url="https://anthropic.helicone.ai",
                    default_headers=headers,
                )
            else:
                self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def send(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> ClaudeResponse:
        """Send one turn to Claude and normalize the response."""
        client = self._get_sdk_client()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            extra_body={"temperature": temperature},
        )
        if tools:
            kwargs["tools"] = tools

        raw = client.messages.create(**kwargs)
        return self._normalize(raw)

    @staticmethod
    def _normalize(raw: Any) -> ClaudeResponse:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in raw.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        usage_obj = getattr(raw, "usage", None)
        usage = (
            {
                "input_tokens": getattr(usage_obj, "input_tokens", 0),
                "output_tokens": getattr(usage_obj, "output_tokens", 0),
            }
            if usage_obj is not None
            else None
        )

        return ClaudeResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=getattr(raw, "stop_reason", "end_turn"),
            raw=raw,
            usage=usage,
        )
