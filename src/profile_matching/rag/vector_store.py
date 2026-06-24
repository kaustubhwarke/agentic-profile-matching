"""Persistent vector store for resume retrieval (Milestone 2 foundation).

Wraps a persistent Chroma collection behind a small, domain-specific facade so
the rest of the application never touches the raw LangChain/Chroma API. The
facade returns :class:`Candidate` domain objects, not raw ``Document`` objects.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from profile_matching.config import Settings, get_settings
from profile_matching.logging_config import get_logger
from profile_matching.models.domain import Candidate
from profile_matching.rag.embeddings import get_embeddings

logger = get_logger(__name__)


class ResumeVectorStore:
    """Domain facade over a persistent Chroma collection of resumes."""

    def __init__(self, settings: Settings | None = None) -> None:
        # Deferred import: keeps chromadb out of the import path for callers
        # (and tests) that never construct a real store.
        from langchain_chroma import Chroma

        self._settings = settings or get_settings()
        self._settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=self._settings.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(self._settings.vector_store_dir),
        )

    # --- Write path ---------------------------------------------------------
    def add_documents(self, documents: list[Document]) -> int:
        """Index ``documents`` (resumes). Returns the number indexed."""
        if not documents:
            return 0
        ids = [doc.metadata.get("candidate_id", str(i)) for i, doc in enumerate(documents)]
        self._store.add_documents(documents=documents, ids=ids)
        logger.info("Indexed %d resume document(s).", len(documents))
        return len(documents)

    def count(self) -> int:
        """Return the number of indexed resumes."""
        try:
            return self._store._collection.count()  # noqa: SLF001 - no public API yet
        except Exception:  # pragma: no cover - defensive
            return 0

    # --- Read path ----------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[Candidate]:
        """Semantic search returning ranked :class:`Candidate` objects."""
        k = top_k or self._settings.retrieval_top_k
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        candidates: list[Candidate] = []
        for doc, score in results:
            candidates.append(
                Candidate(
                    candidate_id=doc.metadata.get("candidate_id", "unknown"),
                    name=doc.metadata.get("name", ""),
                    content=doc.page_content,
                    relevance_score=round(float(score), 4),
                )
            )
        logger.debug("Search '%s' → %d candidate(s).", query[:60], len(candidates))
        return candidates

    def get_candidate(self, candidate_id: str) -> Candidate | None:
        """Fetch a single candidate by id, or ``None`` if absent."""
        data = self._store.get(ids=[candidate_id])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        if not documents:
            return None
        meta = metadatas[0] if metadatas else {}
        return Candidate(
            candidate_id=candidate_id,
            name=meta.get("name", ""),
            content=documents[0],
        )


@lru_cache(maxsize=1)
def get_vector_store() -> ResumeVectorStore:
    """Return a process-wide cached :class:`ResumeVectorStore`."""
    return ResumeVectorStore()
