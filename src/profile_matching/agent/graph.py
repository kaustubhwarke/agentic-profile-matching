"""The matching state machine (Part A).

Graph structure (satisfies the required workflow):

    START → parse_jd → extract_requirements → search_resumes →
    rank_candidates → generate_report → human_feedback ─┐
                                                         │ (refine)
            ▲────────────────────────────────────────────┘
            └── extract_requirements (loops with new feedback)
    human_feedback ──(approve)──► finalize → END

The ``human_feedback`` node uses LangGraph's ``interrupt`` primitive, so the
graph pauses for human input and resumes with ``Command(resume=...)``. A
checkpointer is therefore mandatory for the interrupt to persist across the
pause; we default to an in-memory saver.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from profile_matching.agent import nodes
from profile_matching.models.state import AgentState


def build_matching_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the matching state machine.

    Args:
        checkpointer: Optional persistence backend. Defaults to ``MemorySaver``.
            Required for the human-in-the-loop interrupt to function.

    Returns:
        A compiled, runnable LangGraph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("parse_jd", nodes.parse_jd)
    graph.add_node("extract_requirements", nodes.extract_requirements)
    graph.add_node("search_resumes", nodes.search_resumes)
    graph.add_node("rank_candidates", nodes.rank_candidates)
    graph.add_node("generate_report", nodes.generate_report)
    graph.add_node("human_feedback", nodes.human_feedback)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "parse_jd")
    graph.add_edge("parse_jd", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")
    graph.add_edge("generate_report", "human_feedback")

    # Human-feedback loop: refine (back to requirements) or finalize.
    graph.add_conditional_edges(
        "human_feedback",
        nodes.route_after_feedback,
        {"refine": "extract_requirements", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
