"""The LangGraph agent state.

This is the single source of truth that flows through every node of the
matching graph. It satisfies the assignment's "Agent State Design" requirement:

* **Conversation history** — ``messages`` (reducer-merged via ``add_messages``).
* **Job requirements understanding** — ``job_description`` + ``requirements``.
* **Candidate shortlist and reasoning** — ``candidates`` + ``report``.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from profile_matching.models.domain import (
    Candidate,
    CandidateScore,
    FinalAssessment,
    JobRequirements,
    MatchReport,
    ScreeningReport,
)


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the matching graph.

    ``total=False`` allows nodes to return partial updates; LangGraph merges
    each returned dict into the running state (channel-wise).
    """

    # --- Conversation history (append-only via reducer) ---------------------
    messages: Annotated[list, add_messages]

    # --- Job requirements understanding -------------------------------------
    job_description: str
    requirements: JobRequirements

    # --- Candidate shortlist + reasoning ------------------------------------
    candidates: list[Candidate]
    report: MatchReport

    # --- Human-in-the-loop control ------------------------------------------
    feedback: str
    refinement_count: int
    is_complete: bool


class ScreeningState(TypedDict, total=False):
    """State threaded through the Part-C multi-round screening graph.

    Distinct from :class:`AgentState`: the screening pipeline is a linear,
    non-interactive multi-round process (initial screen → deep analysis →
    final recommendation), so it carries round-by-round artefacts rather than
    a human-feedback loop.
    """

    job_description: str
    requirements: JobRequirements

    # Round 1 — initial screen
    pool: list[Candidate]            # retrieved candidate pool (up to ~100)
    shortlist: list[Candidate]       # advanced to round 2 (top ~10)

    # Round 2 — deep analysis
    finalists: list[CandidateScore]  # detailed scores for the shortlist

    # Round 3 — final recommendation
    assessments: list[FinalAssessment]

    # Compiled output
    report: ScreeningReport
