"""`compare_candidates` tool — head-to-head comparison of multiple candidates."""

from __future__ import annotations

from langchain_core.tools import tool

from profile_matching.llm import get_chat_model
from profile_matching.rag.vector_store import get_vector_store


@tool
def compare_candidates(candidate_ids: list[str], role_context: str = "") -> str:
    """Compare two or more candidates head-to-head.

    Args:
        candidate_ids: The ids of the candidates to compare (e.g. from search_resumes).
        role_context: Optional description of the role to compare them against.

    Returns:
        A structured side-by-side comparison highlighting relative strengths,
        gaps, and a recommended ordering with justification.
    """
    if len(candidate_ids) < 2:
        return "Provide at least two candidate ids to compare."

    store = get_vector_store()
    profiles: list[str] = []
    missing: list[str] = []
    for cid in candidate_ids:
        candidate = store.get_candidate(cid)
        if candidate is None:
            missing.append(cid)
            continue
        profiles.append(f"### Candidate id={cid} ({candidate.name or 'N/A'})\n{candidate.content}")

    if not profiles:
        return f"None of the requested candidates were found: {', '.join(candidate_ids)}"

    note = f"(Not found, skipped: {', '.join(missing)})\n\n" if missing else ""
    llm = get_chat_model()
    message = llm.invoke(
        [
            (
                "system",
                "You are a recruiter performing a rigorous head-to-head candidate "
                "comparison. Compare the candidates dimension by dimension (skills, "
                "depth of experience, relevant impact, gaps). Conclude with a ranked "
                "ordering and a one-line justification per candidate. Cite evidence.",
            ),
            (
                "user",
                f"Role context: {role_context or '(general comparison)'}\n\n"
                + "\n\n".join(profiles),
            ),
        ]
    )
    content = message.content if isinstance(message.content, str) else str(message.content)
    return note + content
