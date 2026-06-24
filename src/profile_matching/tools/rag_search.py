"""RAG search tool (Milestone 2).

Exposes semantic resume retrieval to the agent as a callable tool.
"""

from __future__ import annotations

from langchain_core.tools import tool

from profile_matching.logging_config import get_logger
from profile_matching.rag.vector_store import get_vector_store

logger = get_logger(__name__)


@tool
def search_resumes(query: str, top_k: int = 5) -> str:
    """Semantically search the resume database for matching candidates.

    Use this to find candidates by skills, experience, or any free-text
    criteria (e.g. "React engineer with 3+ years and GraphQL").

    Args:
        query: Natural-language description of the desired candidate profile.
        top_k: Maximum number of candidates to return (1-25).

    Returns:
        A formatted list of matching candidates with ids, similarity scores,
        and resume previews.
    """
    top_k = max(1, min(int(top_k), 25))
    candidates = get_vector_store().search(query, top_k=top_k)
    if not candidates:
        return "No candidates found. The resume index may be empty — run ingestion first."

    blocks: list[str] = [f"Found {len(candidates)} candidate(s) for: {query}\n"]
    for rank, cand in enumerate(candidates, start=1):
        blocks.append(
            f"[{rank}] id={cand.candidate_id} | name={cand.name or 'N/A'} "
            f"| relevance={cand.relevance_score:.3f}\n{cand.preview(500)}\n"
        )
    return "\n".join(blocks)
