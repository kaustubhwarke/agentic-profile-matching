"""`generate_interview_questions` tool — screening questions for one candidate."""

from __future__ import annotations

from langchain_core.tools import tool

from profile_matching.agent.services import make_interview_questions
from profile_matching.rag.vector_store import get_vector_store


@tool
def generate_interview_questions(candidate_id: str, role_context: str = "") -> str:
    """Generate targeted screening questions for a specific candidate.

    Args:
        candidate_id: The id of the candidate (e.g. from search_resumes).
        role_context: Optional description of the role being screened for.

    Returns:
        A set of screening questions tailored to validate the candidate's
        strengths and probe their likely gaps.
    """
    candidate = get_vector_store().get_candidate(candidate_id)
    if candidate is None:
        return f"Candidate '{candidate_id}' was not found in the resume database."
    return make_interview_questions(candidate, role_context)
