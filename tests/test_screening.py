"""Tests for the multi-round screening workflow (Part C).

Services and the vector store are stubbed; these exercise the three-round graph
topology, state threading, and the explainability report.
"""

from __future__ import annotations

import pytest

from profile_matching.agent import screening
from profile_matching.agent.screening import ScreeningPipeline
from profile_matching.models.domain import ScreeningReport


@pytest.fixture
def wired(monkeypatch, stub_services, fake_store):
    monkeypatch.setattr(screening, "get_vector_store", lambda: fake_store)
    return fake_store


def test_screening_runs_all_three_rounds(tmp_settings, wired) -> None:
    report: ScreeningReport = ScreeningPipeline().run("Senior Frontend Engineer (React).")

    # Round 1: pool retrieved and shortlisted.
    assert report.pool_size == 2
    assert report.shortlist_ids  # advanced to round 2
    # Round 2: deep analysis produced scores.
    assert report.finalists
    # Round 3: an assessment per finalist.
    assert len(report.assessments) == len(report.finalists)


def test_screening_classifies_hire_and_borderline(tmp_settings, wired) -> None:
    report = ScreeningPipeline().run("Senior Frontend Engineer")
    decisions = {a.candidate_id: a.decision for a in report.assessments}
    # Alice scores 90 (hire); Bob scores 65 (borderline) under the stub.
    assert decisions["alice"] == "hire"
    assert decisions["bob"] == "borderline"


def test_borderline_candidates_get_improvement_suggestions(tmp_settings, wired) -> None:
    report = ScreeningPipeline().run("Senior Frontend Engineer")
    for assessment in report.borderline():
        assert assessment.improvement_suggestions, "borderline candidates must get suggestions"


def test_screening_report_renders_markdown(tmp_settings, wired) -> None:
    report = ScreeningPipeline().run("Senior Frontend Engineer")
    md = report.render_markdown()
    assert "# Screening report" in md
    assert "Final recommendations" in md
    assert "Improvement suggestions" in md  # borderline candidate present


def test_initial_screen_caps_shortlist(tmp_settings, monkeypatch) -> None:
    """With a large pool, round 1 advances at most shortlist_size candidates."""
    from profile_matching.agent import services
    from tests.conftest import FakeVectorStore
    from profile_matching.models.domain import Candidate

    big_pool = [
        Candidate(candidate_id=f"c{i}", name=f"C{i}", content=f"resume {i}", relevance_score=1.0 - i / 100)
        for i in range(40)
    ]
    store = FakeVectorStore(big_pool)
    monkeypatch.setattr(screening, "get_vector_store", lambda: store)
    # Stub the LLM-backed services used downstream.
    monkeypatch.setattr(services, "extract_requirements_from_text", lambda jd, llm=None: __import__(
        "profile_matching.models.domain", fromlist=["JobRequirements"]
    ).JobRequirements(title="Role"))
    monkeypatch.setattr(services, "initial_screen", lambda c, r, k, llm=None: c[:k])
    monkeypatch.setattr(services, "rank_candidates", lambda c, r, llm=None: [])
    monkeypatch.setattr(services, "final_recommendations", lambda f, r, llm=None: [])
    monkeypatch.setattr(services, "synthesize_report", lambda r, f, llm=None: __import__(
        "profile_matching.models.domain", fromlist=["MatchReport"]
    ).MatchReport())

    report = ScreeningPipeline().run("Role")
    # tmp_settings leaves screen_shortlist_size at its default (10).
    assert len(report.shortlist_ids) == 10
    assert report.pool_size == 40
