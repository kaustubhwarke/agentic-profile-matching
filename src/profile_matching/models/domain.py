"""Domain models — the typed vocabulary of the matching domain.

These Pydantic models are used both as structured-output schemas for the LLM
(``with_structured_output``) and as the canonical in-memory representation that
flows through the agent state and the tools.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


class RequirementItem(BaseModel):
    """A single, atomic requirement extracted from a job description."""

    text: str = Field(description="The requirement, phrased concisely.")
    category: Literal["skill", "experience", "education", "responsibility", "other"] = Field(
        default="skill", description="Coarse classification of the requirement."
    )
    weight: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Relative importance from 1 (minor) to 5 (critical).",
    )


class JobRequirements(BaseModel):
    """Structured requirements parsed from a job description."""

    title: str = Field(default="", description="The role/job title.")
    summary: str = Field(default="", description="One-paragraph summary of the role.")
    must_have: list[RequirementItem] = Field(
        default_factory=list, description="Hard requirements a candidate must satisfy."
    )
    nice_to_have: list[RequirementItem] = Field(
        default_factory=list, description="Desirable but non-blocking requirements."
    )
    min_years_experience: float = Field(
        default=0.0, ge=0, description="Minimum total years of relevant experience."
    )

    def as_search_query(self) -> str:
        """Render the requirements into a dense-retrieval query string."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        parts.extend(item.text for item in self.must_have)
        parts.extend(item.text for item in self.nice_to_have)
        if self.min_years_experience:
            parts.append(f"{self.min_years_experience:g}+ years experience")
        return " ; ".join(parts) if parts else self.summary


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """A candidate resume retrieved from the vector store."""

    candidate_id: str = Field(description="Stable identifier (resume filename stem).")
    name: str = Field(default="", description="Candidate display name, if parsed.")
    content: str = Field(description="Full resume text.")
    relevance_score: float = Field(
        default=0.0, description="Raw semantic similarity from retrieval (higher is closer)."
    )

    def preview(self, limit: int = 600) -> str:
        """Return a truncated preview of the resume content."""
        text = self.content.strip()
        return text if len(text) <= limit else text[:limit] + "…"


class CandidateScore(BaseModel):
    """The agent's reasoned assessment of one candidate against requirements."""

    candidate_id: str = Field(description="Identifier of the candidate being scored.")
    name: str = Field(default="", description="Candidate display name.")
    overall_score: int = Field(
        ge=0, le=100, description="Holistic fit score from 0 to 100."
    )
    must_have_met: list[str] = Field(
        default_factory=list, description="Must-have requirements clearly satisfied."
    )
    must_have_missing: list[str] = Field(
        default_factory=list, description="Must-have requirements not evidenced."
    )
    strengths: list[str] = Field(default_factory=list, description="Notable strengths.")
    gaps: list[str] = Field(default_factory=list, description="Notable gaps or risks.")
    reasoning: str = Field(description="Concise justification for the score.")
    recommendation: Literal["strong_yes", "yes", "maybe", "no"] = Field(
        description="Screening recommendation."
    )


class MatchReport(BaseModel):
    """A human-readable report summarising a full matching run."""

    job_title: str = Field(default="")
    ranked: list[CandidateScore] = Field(default_factory=list)
    summary: str = Field(default="", description="Executive summary of the shortlist.")

    def top(self, n: int = 3) -> list[CandidateScore]:
        """Return the top ``n`` candidates by overall score."""
        return sorted(self.ranked, key=lambda c: c.overall_score, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Part C — Multi-round screening & explainability
# ---------------------------------------------------------------------------

HireDecision = Literal["hire", "no_hire", "borderline"]


class ScreenSelection(BaseModel):
    """Structured output of the initial (round-1) screen.

    The model returns the candidate ids it advances to deep analysis, in
    priority order.
    """

    selected_ids: list[str] = Field(
        default_factory=list, description="Candidate ids advanced to round 2, best first."
    )
    reasoning: str = Field(default="", description="Brief rationale for the shortlist.")


class FinalAssessment(BaseModel):
    """Round-3 hire/no-hire decision with explainability for one finalist."""

    candidate_id: str = Field(description="Identifier of the assessed candidate.")
    name: str = Field(default="")
    decision: HireDecision = Field(description="Final screening decision.")
    confidence: int = Field(ge=0, le=100, description="Confidence in the decision (0-100).")
    rationale: str = Field(description="Justification grounded in the candidate's evidence.")
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable suggestions; populated especially for borderline candidates.",
    )


class FinalAssessments(BaseModel):
    """Wrapper enabling a single structured call to assess all finalists."""

    items: list[FinalAssessment] = Field(default_factory=list)


class ScreeningReport(BaseModel):
    """Full output of a multi-round screening run (Part C)."""

    job_title: str = Field(default="")
    pool_size: int = Field(default=0, description="Number of resumes considered in round 1.")
    shortlist_ids: list[str] = Field(
        default_factory=list, description="Candidate ids advanced from round 1 to round 2."
    )
    finalists: list[CandidateScore] = Field(
        default_factory=list, description="Round-2 deep-analysis scores."
    )
    assessments: list[FinalAssessment] = Field(
        default_factory=list, description="Round-3 hire/no-hire assessments."
    )
    summary: str = Field(default="", description="Executive summary across all rounds.")

    def hires(self) -> list[FinalAssessment]:
        """Finalists with a 'hire' decision."""
        return [a for a in self.assessments if a.decision == "hire"]

    def borderline(self) -> list[FinalAssessment]:
        """Finalists flagged borderline (need improvement before a yes)."""
        return [a for a in self.assessments if a.decision == "borderline"]

    def _score_for(self, candidate_id: str) -> CandidateScore | None:
        return next((s for s in self.finalists if s.candidate_id == candidate_id), None)

    def render_markdown(self) -> str:
        """Render a detailed, explainable match report (Part C explainability)."""
        lines: list[str] = [f"# Screening report — {self.job_title or 'Role'}", ""]
        lines.append(
            f"Considered **{self.pool_size}** resumes → shortlisted "
            f"**{len(self.shortlist_ids)}** → assessed **{len(self.assessments)}**."
        )
        if self.summary:
            lines += ["", "## Executive summary", "", self.summary]

        lines += ["", "## Final recommendations", ""]
        decision_rank = {"hire": 0, "borderline": 1, "no_hire": 2}
        for assessment in sorted(self.assessments, key=lambda a: decision_rank.get(a.decision, 3)):
            score = self._score_for(assessment.candidate_id)
            header = f"### {assessment.name or assessment.candidate_id} — {assessment.decision.upper()}"
            if score is not None:
                header += f" ({score.overall_score}/100)"
            lines += [header, "", f"_Confidence: {assessment.confidence}%_", "", assessment.rationale, ""]
            if score is not None:
                lines.append(f"**Strengths:** {', '.join(score.strengths) or '—'}")
                lines.append(f"**Gaps:** {', '.join(score.gaps) or '—'}")
                lines.append(
                    f"**Must-have met:** {', '.join(score.must_have_met) or '—'}  \n"
                    f"**Must-have missing:** {', '.join(score.must_have_missing) or '—'}"
                )
            if assessment.improvement_suggestions:
                lines += ["", "**Improvement suggestions:**"]
                lines += [f"- {s}" for s in assessment.improvement_suggestions]
            lines.append("")
        return "\n".join(lines)
