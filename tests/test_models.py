"""Unit tests for the domain models."""

from __future__ import annotations

from profile_matching.models.domain import (
    Candidate,
    JobRequirements,
    MatchReport,
    RequirementItem,
)
from tests.conftest import make_score


def test_as_search_query_includes_requirements(sample_requirements: JobRequirements) -> None:
    query = sample_requirements.as_search_query()
    assert "Senior Frontend Engineer" in query
    assert "React" in query
    assert "TypeScript" in query
    assert "3+ years experience" in query


def test_as_search_query_falls_back_to_summary() -> None:
    reqs = JobRequirements(summary="Just a summary")
    assert reqs.as_search_query() == "Just a summary"


def test_candidate_preview_truncates() -> None:
    cand = Candidate(candidate_id="x", content="y" * 1000)
    preview = cand.preview(limit=100)
    assert len(preview) <= 101
    assert preview.endswith("…")


def test_match_report_top_orders_by_score() -> None:
    report = MatchReport(
        ranked=[
            make_score("a", "A", 60),
            make_score("b", "B", 95),
            make_score("c", "C", 80),
        ]
    )
    top = report.top(2)
    assert [s.candidate_id for s in top] == ["b", "c"]


def test_requirement_item_weight_bounds() -> None:
    item = RequirementItem(text="React", weight=5)
    assert 1 <= item.weight <= 5
