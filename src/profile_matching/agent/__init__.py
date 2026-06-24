"""Agent package: prompts, graph nodes, the matching state machine, and chat agent."""

from __future__ import annotations

from profile_matching.agent.conversation import ConversationalAgent, build_conversational_agent
from profile_matching.agent.graph import build_matching_graph
from profile_matching.agent.pipeline import MatchingPipeline, PipelineResult
from profile_matching.agent.screening import ScreeningPipeline, build_screening_graph

__all__ = [
    "ConversationalAgent",
    "MatchingPipeline",
    "PipelineResult",
    "ScreeningPipeline",
    "build_conversational_agent",
    "build_matching_graph",
    "build_screening_graph",
]
