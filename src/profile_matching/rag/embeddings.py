"""Embedding model factory.

Uses a local sentence-transformers model via ``langchain-huggingface``. This
runs entirely offline (no per-call API cost) and keeps resume text — which may
be sensitive PII — on the machine rather than shipping it to a remote embedding
service.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from profile_matching.config import get_settings

if TYPE_CHECKING:  # avoid importing heavy ML deps at module import time
    from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> "HuggingFaceEmbeddings":
    """Return a process-wide cached embedding model.

    The model is downloaded once to the local Hugging Face cache and reused.
    The import is deferred so that modules which never embed (e.g. unit tests
    that stub the vector store) don't pay the cost of loading torch.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
