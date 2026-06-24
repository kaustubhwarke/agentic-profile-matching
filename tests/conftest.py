"""Shared pytest fixtures.

Tests are designed to run without an Anthropic API key or any model download:
LLM-backed services and the vector store are stubbed. Pure logic (domain
models, sandbox security, routing) is exercised directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from profile_matching.config import get_settings
from profile_matching.models.domain import (
    Candidate,
    CandidateScore,
    FinalAssessment,
    JobRequirements,
    MatchReport,
    RequirementItem,
)


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point all runtime directories at a temp dir and reset the settings cache."""
    resume_dir = tmp_path / "resumes"
    job_dir = tmp_path / "jobs"
    report_dir = tmp_path / "reports"
    for d in (resume_dir, job_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("APM_RESUME_DIR", str(resume_dir))
    monkeypatch.setenv("APM_JOB_DIR", str(job_dir))
    monkeypatch.setenv("APM_REPORT_DIR", str(report_dir))
    monkeypatch.setenv("APM_VECTOR_STORE_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("APM_CHECKPOINT_DB", str(tmp_path / "ckpt.sqlite"))
    monkeypatch.setenv("APM_MAX_REFINEMENT_LOOPS", "3")

    get_settings.cache_clear()
    settings = get_settings()
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def sample_requirements() -> JobRequirements:
    return JobRequirements(
        title="Senior Frontend Engineer",
        summary="Build the customer web app.",
        must_have=[
            RequirementItem(text="React", weight=5),
            RequirementItem(text="TypeScript", weight=5),
            RequirementItem(text="3+ years experience", category="experience", weight=4),
        ],
        nice_to_have=[RequirementItem(text="Webpack", weight=2)],
        min_years_experience=3,
    )


@pytest.fixture
def sample_candidates() -> list[Candidate]:
    return [
        Candidate(candidate_id="alice", name="Alice", content="React TypeScript 6 years", relevance_score=0.9),
        Candidate(candidate_id="bob", name="Bob", content="React JavaScript 2 years", relevance_score=0.7),
    ]


class FakeVectorStore:
    """In-memory stand-in for :class:`ResumeVectorStore`."""

    def __init__(self, candidates: list[Candidate]) -> None:
        self._by_id = {c.candidate_id: c for c in candidates}
        self._candidates = candidates

    def count(self) -> int:
        return len(self._candidates)

    def search(self, query: str, top_k: int | None = None):
        return self._candidates[: top_k or len(self._candidates)]

    def get_candidate(self, candidate_id: str):
        return self._by_id.get(candidate_id)


@pytest.fixture
def fake_store(sample_candidates: list[Candidate]) -> FakeVectorStore:
    return FakeVectorStore(sample_candidates)


def make_score(candidate_id: str, name: str, score: int) -> CandidateScore:
    rec = "strong_yes" if score >= 85 else "yes" if score >= 70 else "maybe" if score >= 50 else "no"
    return CandidateScore(
        candidate_id=candidate_id,
        name=name,
        overall_score=score,
        must_have_met=["React"],
        must_have_missing=[] if score >= 70 else ["TypeScript depth"],
        strengths=["Strong React"],
        gaps=[] if score >= 70 else ["Limited TypeScript"],
        reasoning="stubbed reasoning",
        recommendation=rec,
    )


@pytest.fixture
def stub_services(monkeypatch: pytest.MonkeyPatch, sample_requirements, sample_candidates):
    """Replace LLM-backed services with deterministic stubs (no network)."""
    from profile_matching.agent import nodes, services

    def fake_extract(jd: str, llm=None) -> JobRequirements:
        return sample_requirements

    def fake_rank(candidates, requirements, llm=None) -> list[CandidateScore]:
        scores = [
            make_score(c.candidate_id, c.name, 90 - i * 25)
            for i, c in enumerate(candidates)
        ]
        return sorted(scores, key=lambda s: s.overall_score, reverse=True)

    def fake_report(requirements, ranked, llm=None) -> MatchReport:
        return MatchReport(
            job_title=requirements.title,
            ranked=ranked,
            summary="Stubbed executive summary.",
        )

    def fake_initial_screen(candidates, requirements, shortlist_size, llm=None):
        return candidates[:shortlist_size]

    def fake_final(finalists, requirements, llm=None) -> list[FinalAssessment]:
        out: list[FinalAssessment] = []
        for s in finalists:
            if s.overall_score >= 70:
                decision = "hire"
            elif s.overall_score >= 50:
                decision = "borderline"
            else:
                decision = "no_hire"
            out.append(
                FinalAssessment(
                    candidate_id=s.candidate_id,
                    name=s.name,
                    decision=decision,
                    confidence=s.overall_score,
                    rationale="stubbed rationale",
                    improvement_suggestions=["Deepen TypeScript"] if decision == "borderline" else [],
                )
            )
        return out

    monkeypatch.setattr(services, "extract_requirements_from_text", fake_extract)
    monkeypatch.setattr(services, "rank_candidates", fake_rank)
    monkeypatch.setattr(services, "synthesize_report", fake_report)
    monkeypatch.setattr(services, "initial_screen", fake_initial_screen)
    monkeypatch.setattr(services, "final_recommendations", fake_final)
    return services
