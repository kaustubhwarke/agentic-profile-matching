"""LLM factory.

Centralises construction of the Claude chat model so every component (graph
nodes, tools, conversational agent) shares one consistent, configured client.

Implementation note: Claude Opus 4.8 / 4.7 reject sampling parameters
(``temperature``/``top_p``/``top_k``) with a 400. We therefore never set
``temperature`` — langchain-anthropic omits it from the request when unset —
and steer generation depth via the ``effort`` output-config parameter instead.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from profile_matching.config import Settings, get_settings


def build_chat_model(settings: Settings | None = None, *, max_tokens: int | None = None) -> ChatAnthropic:
    """Construct a configured :class:`ChatAnthropic` chat model.

    Args:
        settings: Optional settings override (defaults to the cached singleton).
        max_tokens: Optional per-call override of the configured output cap.
    """
    settings = settings or get_settings()

    # `effort` (output_config) is supported on Opus 4.5+ and Sonnet 4.6 but
    # NOT on Haiku 4.5 or older Sonnets (those return a 400). Only attach it
    # for models known to support it.
    model_kwargs: dict = {}
    model = settings.llm_model.lower()
    if "opus-4" in model or "sonnet-4-6" in model or "fable-5" in model:
        model_kwargs["output_config"] = {"effort": settings.llm_effort}

    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=max_tokens or settings.llm_max_tokens,
        # Pass the raw secret value; langchain-anthropic wraps it internally.
        api_key=settings.anthropic_api_key.get_secret_value(),
        timeout=120,
        max_retries=3,
        model_kwargs=model_kwargs,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> ChatAnthropic:
    """Return a process-wide cached chat model built from default settings."""
    return build_chat_model()
