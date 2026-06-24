"""Multi-round screening graph (Part C — Advanced Capabilities).

Implements the three-round screening workflow plus explainability:

    START → extract_requirements → retrieve_pool → initial_screen (round 1)
          → deep_analysis (round 2) → final_recommendation (round 3)
          → compile_report → END

* **Round 1 (initial screen):** retrieve up to ``APM_SCREEN_POOL_SIZE`` resumes
  ("from 100") and fast-triage to the top ``APM_SCREEN_SHORTLIST_SIZE`` ("top 10").
* **Round 2 (deep analysis):** full, reasoned scoring of the shortlist.
* **Round 3 (final round):** hire / no-hire / borderline recommendation per
  finalist, with improvement suggestions for borderline candidates.

The compiled report (``ScreeningReport``) provides the explainability surface:
detailed per-candidate strengths, gaps, and improvement suggestions.

This graph is linear and non-interactive (no human-in-the-loop), so it needs no
checkpointer — distinct from the Part-A matching graph.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from profile_matching.agent import services
from profile_matching.config import get_settings
from profile_matching.logging_config import get_logger
from profile_matching.models.domain import ScreeningReport
from profile_matching.models.state import ScreeningState
from profile_matching.rag.vector_store import get_vector_store

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def _extract_requirements(state: ScreeningState) -> dict:
    requirements = services.extract_requirements_from_text(state["job_description"])
    return {"requirements": requirements}


def _retrieve_pool(state: ScreeningState) -> dict:
    settings = get_settings()
    query = state["requirements"].as_search_query()
    pool = get_vector_store().search(query, top_k=settings.screen_pool_size)
    logger.info("Round 1 retrieval: %d candidate(s).", len(pool))
    return {"pool": pool}


def _initial_screen(state: ScreeningState) -> dict:
    settings = get_settings()
    shortlist = services.initial_screen(
        state.get("pool", []), state["requirements"], settings.screen_shortlist_size
    )
    return {"shortlist": shortlist}


def _deep_analysis(state: ScreeningState) -> dict:
    finalists = services.rank_candidates(state.get("shortlist", []), state["requirements"])
    return {"finalists": finalists}


def _final_recommendation(state: ScreeningState) -> dict:
    assessments = services.final_recommendations(state.get("finalists", []), state["requirements"])
    return {"assessments": assessments}


def _compile_report(state: ScreeningState) -> dict:
    requirements = state["requirements"]
    finalists = state.get("finalists", [])
    # Reuse the executive-summary synthesiser for a consistent narrative voice.
    summary = services.synthesize_report(requirements, finalists).summary if finalists else ""
    report = ScreeningReport(
        job_title=requirements.title,
        pool_size=len(state.get("pool", [])),
        shortlist_ids=[c.candidate_id for c in state.get("shortlist", [])],
        finalists=finalists,
        assessments=state.get("assessments", []),
        summary=summary,
    )
    return {"report": report}


# ---------------------------------------------------------------------------
# Graph assembly + driver
# ---------------------------------------------------------------------------
def build_screening_graph():
    """Build and compile the multi-round screening graph."""
    graph = StateGraph(ScreeningState)
    graph.add_node("extract_requirements", _extract_requirements)
    graph.add_node("retrieve_pool", _retrieve_pool)
    graph.add_node("initial_screen", _initial_screen)
    graph.add_node("deep_analysis", _deep_analysis)
    graph.add_node("final_recommendation", _final_recommendation)
    graph.add_node("compile_report", _compile_report)

    graph.add_edge(START, "extract_requirements")
    graph.add_edge("extract_requirements", "retrieve_pool")
    graph.add_edge("retrieve_pool", "initial_screen")
    graph.add_edge("initial_screen", "deep_analysis")
    graph.add_edge("deep_analysis", "final_recommendation")
    graph.add_edge("final_recommendation", "compile_report")
    graph.add_edge("compile_report", END)
    return graph.compile()


class ScreeningPipeline:
    """Drives the multi-round screening graph to a :class:`ScreeningReport`."""

    def __init__(self) -> None:
        self._graph = build_screening_graph()

    def run(self, job_description: str) -> ScreeningReport:
        """Run all three rounds for a job description and return the report."""
        result = self._graph.invoke({"job_description": job_description})
        report = result.get("report")
        return report if report is not None else ScreeningReport()
