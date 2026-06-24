"""Graph node implementations for the matching state machine (Part A).

Each node is a pure-ish function ``(AgentState) -> dict`` returning a partial
state update. Nodes never mutate the incoming state in place; LangGraph merges
the returned dict into the running state via the channel reducers.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from profile_matching.agent import services
from profile_matching.config import get_settings
from profile_matching.logging_config import get_logger
from profile_matching.models.domain import JobRequirements
from profile_matching.models.state import AgentState
from profile_matching.rag.vector_store import get_vector_store

logger = get_logger(__name__)


def parse_jd(state: AgentState) -> dict:
    """Normalise the incoming job description and record the run start."""
    jd = (state.get("job_description") or "").strip()
    if not jd:
        return {
            "messages": [AIMessage(content="No job description supplied; nothing to match.")],
            "is_complete": True,
        }
    logger.info("parse_jd: %d chars", len(jd))
    return {
        "job_description": jd,
        "refinement_count": state.get("refinement_count", 0),
        "messages": [AIMessage(content="Parsed job description. Extracting requirements…")],
    }


def extract_requirements(state: AgentState) -> dict:
    """Extract structured requirements, folding in any human feedback."""
    jd = state["job_description"]
    feedback = (state.get("feedback") or "").strip()
    if feedback:
        jd = f"{jd}\n\nADDITIONAL REFINEMENT FROM HIRING MANAGER:\n{feedback}"

    requirements: JobRequirements = services.extract_requirements_from_text(jd)
    msg = (
        f"Identified {len(requirements.must_have)} must-have and "
        f"{len(requirements.nice_to_have)} nice-to-have requirements."
    )
    return {"requirements": requirements, "messages": [AIMessage(content=msg)]}


def search_resumes(state: AgentState) -> dict:
    """Retrieve a candidate pool from the vector store using the requirements."""
    requirements = state["requirements"]
    query = requirements.as_search_query()
    candidates = get_vector_store().search(query)
    msg = f"Retrieved {len(candidates)} candidate(s) from the resume database."
    logger.info("search_resumes: %s", msg)
    return {"candidates": candidates, "messages": [AIMessage(content=msg)]}


def rank_candidates(state: AgentState) -> dict:
    """Score and rank the retrieved candidates against the requirements."""
    candidates = state.get("candidates", [])
    requirements = state["requirements"]
    if not candidates:
        return {
            "messages": [AIMessage(content="No candidates to rank.")],
            "is_complete": True,
        }
    ranked = services.rank_candidates(candidates, requirements)
    top = ranked[0]
    msg = f"Ranked {len(ranked)} candidate(s). Top: {top.name or top.candidate_id} ({top.overall_score}/100)."
    return {"report": services.MatchReport(job_title=requirements.title, ranked=ranked), "messages": [AIMessage(content=msg)]}


def generate_report(state: AgentState) -> dict:
    """Synthesise the executive-summary report for the ranked shortlist."""
    requirements = state["requirements"]
    report = state.get("report")
    ranked = report.ranked if report else []
    full_report = services.synthesize_report(requirements, ranked)
    return {
        "report": full_report,
        "messages": [AIMessage(content=full_report.summary)],
    }


def human_feedback(state: AgentState) -> dict:
    """Pause for human feedback (human-in-the-loop refinement).

    Uses LangGraph's ``interrupt`` primitive: execution halts here and the
    caller resumes with ``Command(resume=<feedback string>)``. An empty string
    or an approval keyword ends the loop.
    """
    from langgraph.types import interrupt  # local import keeps node import-light

    report = state.get("report")
    payload = {
        "summary": report.summary if report else "",
        "top_candidates": [
            {"id": s.candidate_id, "name": s.name, "score": s.overall_score, "rec": s.recommendation}
            for s in (report.top(3) if report else [])
        ],
        "prompt": (
            "Review the shortlist. Reply with refined criteria to re-rank, "
            "or 'approve' to finish."
        ),
    }
    feedback = interrupt(payload)
    feedback = (feedback or "").strip() if isinstance(feedback, str) else str(feedback)
    return {
        "feedback": feedback,
        "refinement_count": state.get("refinement_count", 0) + 1,
    }


def route_after_feedback(state: AgentState) -> str:
    """Decide whether to refine again or finish, after human feedback."""
    settings = get_settings()
    feedback = (state.get("feedback") or "").strip().lower()
    count = state.get("refinement_count", 0)

    approval_words = {"approve", "approved", "accept", "done", "ok", "looks good", "yes", ""}
    if feedback in approval_words or count >= settings.max_refinement_loops:
        return "finalize"
    return "refine"


def finalize(state: AgentState) -> dict:
    """Mark the run complete."""
    return {
        "is_complete": True,
        "feedback": "",  # clear so a re-run starts clean
        "messages": [AIMessage(content="Matching complete. Shortlist finalised.")],
    }
