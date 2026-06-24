"""Tests for the human-feedback routing logic (no LLM)."""

from __future__ import annotations

import pytest

from profile_matching.agent.nodes import route_after_feedback


@pytest.mark.parametrize("feedback", ["approve", "Approved", "done", "ok", "looks good", ""])
def test_approval_finalizes(tmp_settings, feedback: str) -> None:
    state = {"feedback": feedback, "refinement_count": 1}
    assert route_after_feedback(state) == "finalize"


def test_substantive_feedback_refines(tmp_settings) -> None:
    state = {"feedback": "require GraphQL and 5+ years", "refinement_count": 1}
    assert route_after_feedback(state) == "refine"


def test_max_loops_forces_finalize(tmp_settings) -> None:
    # tmp_settings sets APM_MAX_REFINEMENT_LOOPS=3
    state = {"feedback": "keep refining", "refinement_count": 3}
    assert route_after_feedback(state) == "finalize"
