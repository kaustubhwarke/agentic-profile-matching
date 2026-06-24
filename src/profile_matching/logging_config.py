"""Structured, environment-aware logging configuration.

Call :func:`configure_logging` once at process startup (CLI / Streamlit entry
points do this). Library modules should simply ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from profile_matching.config import get_settings

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter suitable for log aggregation systems."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(force: bool = False) -> None:
    """Configure root logging once, honouring ``APM_LOG_LEVEL`` / ``APM_LOG_JSON``."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    handler = logging.StreamHandler(stream=sys.stderr)

    if settings.log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # Silence noisy third-party libraries.
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    configure_logging()
    return logging.getLogger(name)
