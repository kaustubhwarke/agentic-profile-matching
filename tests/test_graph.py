"""End-to-end tests for the matching state machine (Part A).

LLM services and the vector store are stubbed, so these exercise the graph
topology, state threading, and the human-in-the-loop interrupt/resume cycle
without any network calls.
"""

from __future__ import annotations

import pytest

from profile_matching.agent import nodes
from profile_matching.agent.pipeline import MatchingPipeline


@pytest.fixture
def patched(monkeypatch, stub_services, fake_store):
    """Wire the stubbed services + fake store into the graph nodes."""
    monkeypatch.setattr(nodes, "get_vector_store", lambda: fake_store)
    return True


def test_pipeline_runs_to_human_feedback(tmp_settings, patched) -> None:
    pipeline = MatchingPipeline()
    result = pipeline.start("Hiring a Senior Frontend Engineer (React).", thread_id="t1")

    assert result.awaiting_feedback is True
    assert result.report is not None
    assert result.report.ranked  # candidates were scored
    # Highest score first.
    assert result.report.ranked[0].overall_score >= result.report.ranked[-1].overall_score


def test_pipeline_approve_finalizes(tmp_settings, patched) -> None:
    pipeline = MatchingPipeline()
    pipeline.start("Senior Frontend Engineer", thread_id="t2")
    result = pipeline.resume("approve", thread_id="t2")

    assert result.awaiting_feedback is False
    assert result.report is not None
    assert result.report.summary


def test_pipeline_refine_loops_then_finalizes(tmp_settings, patched) -> None:
    pipeline = MatchingPipeline()
    pipeline.start("Senior Frontend Engineer", thread_id="t3")

    # One refinement loop: supply new criteria → pauses again for feedback.
    refined = pipeline.resume("also require GraphQL", thread_id="t3")
    assert refined.awaiting_feedback is True
    assert refined.report is not None

    # Approve to finish.
    done = pipeline.resume("approve", thread_id="t3")
    assert done.awaiting_feedback is False


def test_pipeline_terminates_at_max_refinements(tmp_settings, patched) -> None:
    pipeline = MatchingPipeline()
    pipeline.start("Senior Frontend Engineer", thread_id="t4")
    # tmp_settings caps refinement loops at 3; keep refining past that.
    result = None
    for _ in range(6):
        result = pipeline.resume("keep refining", thread_id="t4")
        if not result.awaiting_feedback:
            break
    assert result is not None and result.awaiting_feedback is False
