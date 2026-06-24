"""Centralised, validated application configuration.

All configuration is sourced from environment variables (optionally via a
`.env` file) using ``pydantic-settings``. This keeps secrets out of source
control and gives a single, typed, import-anywhere settings object.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EffortLevel = Literal["low", "medium", "high", "max"]


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APM_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Anthropic / Claude -------------------------------------------------
    # NOTE: read without the APM_ prefix because the Anthropic SDK and
    # langchain-anthropic both expect the canonical ANTHROPIC_API_KEY name.
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="ANTHROPIC_API_KEY",
        description="Anthropic API key used by langchain-anthropic.",
    )

    # --- LLM ----------------------------------------------------------------
    llm_model: str = Field(
        default="claude-opus-4-8",
        description="Claude model id. Latest, most capable by default.",
    )
    llm_max_tokens: int = Field(default=8000, ge=512, le=128000)
    llm_effort: EffortLevel = Field(default="high")

    # --- Embeddings / RAG ---------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local sentence-transformers embedding model.",
    )
    vector_store_dir: Path = Field(default=Path("./chroma_db"))
    collection_name: str = Field(default="resumes")
    retrieval_top_k: int = Field(default=10, ge=1, le=100)

    # --- Data locations -----------------------------------------------------
    resume_dir: Path = Field(default=Path("./data/resumes"))
    job_dir: Path = Field(default=Path("./data/jobs"))
    report_dir: Path = Field(default=Path("./reports"))

    # --- Agent behaviour ----------------------------------------------------
    max_refinement_loops: int = Field(default=5, ge=1, le=20)
    checkpoint_db: Path = Field(default=Path("./checkpoints/graph_state.sqlite"))

    # --- Multi-round screening (Part C) -------------------------------------
    # Round 1 retrieves up to this many resumes from the index ("from 100").
    screen_pool_size: int = Field(default=100, ge=1, le=1000)
    # Round 1 advances this many candidates to deep analysis ("top 10").
    screen_shortlist_size: int = Field(default=10, ge=1, le=100)
    # Borderline band: a finalist scoring in [low, high) is "borderline" and
    # receives improvement suggestions.
    borderline_low: int = Field(default=50, ge=0, le=100)
    borderline_high: int = Field(default=70, ge=0, le=100)

    # --- Observability ------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    # --- Derived helpers ----------------------------------------------------
    @property
    def has_api_key(self) -> bool:
        """True when a non-empty Anthropic API key is configured."""
        return bool(self.anthropic_api_key.get_secret_value().strip())

    def ensure_runtime_dirs(self) -> None:
        """Create directories the application writes to at runtime."""
        for path in (
            self.vector_store_dir,
            self.report_dir,
            self.checkpoint_db.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
