"""Test scenarios — 5+ conversation flows (submission requirement).

Each scenario maps to a distinct user journey through the system. LLM calls and
the vector store are stubbed so the flows are deterministic and offline.

Flows covered:
  1. JD → shortlist → approve            (happy-path pipeline)
  2. JD → refine criteria → re-rank      (iterative refinement)
  3. "Find candidates with X"            (semantic search tool)
  4. "Compare the top candidates"        (head-to-head comparison tool)
  5. "Generate interview questions"      (screening tool)
  6. Empty index handling                (graceful degradation)
  7. "Run a full screening for this JD"  (Part C multi-round screening tool)
  8. Hire / no-hire decisioning          (Part C explainability)
"""

from __future__ import annotations

import pytest

from profile_matching.agent import nodes, screening
from profile_matching.agent.pipeline import MatchingPipeline
from profile_matching.tools import comparison, interview, rag_search
from profile_matching.tools import screening as screening_tool
from tests.conftest import FakeVectorStore


class _FakeLLM:
    """Minimal stand-in for a chat model: returns a canned text message."""

    def __init__(self, text: str) -> None:
        self._text = text

    def invoke(self, _messages):
        from langchain_core.messages import AIMessage

        return AIMessage(content=self._text)


@pytest.fixture
def wired(monkeypatch, stub_services, fake_store):
    monkeypatch.setattr(nodes, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(rag_search, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(comparison, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(interview, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(screening, "get_vector_store", lambda: fake_store)
    return fake_store


# --- Flow 1: happy path -----------------------------------------------------
def test_flow_jd_to_shortlist_approve(tmp_settings, wired) -> None:
    pipeline = MatchingPipeline()
    result = pipeline.start("Senior Frontend Engineer needing React.", thread_id="f1")
    assert result.report.ranked
    final = pipeline.resume("approve", thread_id="f1")
    assert final.awaiting_feedback is False
    assert "Alice" in {s.name for s in final.report.ranked}


# --- Flow 2: iterative refinement ------------------------------------------
def test_flow_refine_then_approve(tmp_settings, wired) -> None:
    pipeline = MatchingPipeline()
    pipeline.start("Senior Frontend Engineer", thread_id="f2")
    refined = pipeline.resume("Find me candidates with React and 3+ years", thread_id="f2")
    assert refined.awaiting_feedback is True  # re-ranked, awaiting next decision
    done = pipeline.resume("approve", thread_id="f2")
    assert done.awaiting_feedback is False


# --- Flow 3: semantic search ------------------------------------------------
def test_flow_find_candidates(tmp_settings, wired) -> None:
    out = rag_search.search_resumes.invoke(
        {"query": "React engineer with 3+ years", "top_k": 5}
    )
    assert "Alice" in out
    assert "id=alice" in out


# --- Flow 4: head-to-head comparison ---------------------------------------
def test_flow_compare_candidates(tmp_settings, wired, monkeypatch) -> None:
    monkeypatch.setattr(
        comparison, "get_chat_model", lambda: _FakeLLM("Alice > Bob: more React depth.")
    )
    out = comparison.compare_candidates.invoke(
        {"candidate_ids": ["alice", "bob"], "role_context": "Frontend"}
    )
    assert "Alice" in out


def test_flow_compare_requires_two(tmp_settings, wired) -> None:
    out = comparison.compare_candidates.invoke({"candidate_ids": ["alice"]})
    assert "at least two" in out.lower()


# --- Flow 5: interview questions -------------------------------------------
def test_flow_interview_questions(tmp_settings, wired, monkeypatch) -> None:
    from profile_matching.agent import services

    monkeypatch.setattr(
        services, "make_interview_questions", lambda cand, ctx, llm=None: "Q1. Explain React hooks."
    )
    out = interview.generate_interview_questions.invoke(
        {"candidate_id": "alice", "role_context": "Frontend"}
    )
    assert "React" in out


def test_flow_interview_unknown_candidate(tmp_settings, wired) -> None:
    out = interview.generate_interview_questions.invoke({"candidate_id": "ghost"})
    assert "not found" in out.lower()


# --- Flow 6: empty index graceful degradation ------------------------------
def test_flow_empty_index(tmp_settings, monkeypatch) -> None:
    empty = FakeVectorStore([])
    monkeypatch.setattr(rag_search, "get_vector_store", lambda: empty)
    out = rag_search.search_resumes.invoke({"query": "anything"})
    assert "empty" in out.lower() or "no candidates" in out.lower()


# --- Flow 7: full multi-round screening (Part C) ---------------------------
def test_flow_run_full_screening(tmp_settings, wired) -> None:
    out = screening_tool.screen_candidates.invoke(
        {"job_description": "Senior Frontend Engineer needing React and TypeScript."}
    )
    assert "Screening report" in out
    assert "Final recommendations" in out


# --- Flow 8: hire / no-hire decisioning + improvement suggestions ----------
def test_flow_hire_decisions(tmp_settings, wired) -> None:
    report = screening.ScreeningPipeline().run("Senior Frontend Engineer")
    decisions = {a.candidate_id: a.decision for a in report.assessments}
    assert decisions.get("alice") == "hire"
    assert decisions.get("bob") == "borderline"
    # Borderline candidate carries actionable improvement suggestions.
    bob = next(a for a in report.assessments if a.candidate_id == "bob")
    assert bob.improvement_suggestions
