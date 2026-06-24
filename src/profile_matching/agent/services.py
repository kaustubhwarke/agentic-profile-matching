"""Reusable LLM-backed domain services.

These pure functions encapsulate every LLM interaction in the domain so that
both the graph nodes (Part A) and the conversational tools (Part B) share one
implementation. Each function is small, typed, and independently testable.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from profile_matching.agent import prompts
from profile_matching.config import get_settings
from profile_matching.llm import get_chat_model
from profile_matching.logging_config import get_logger
from profile_matching.models.domain import (
    Candidate,
    CandidateScore,
    FinalAssessment,
    FinalAssessments,
    JobRequirements,
    MatchReport,
    ScreenSelection,
)

logger = get_logger(__name__)


def _format_requirements(items) -> str:
    if not items:
        return "  (none)"
    return "\n".join(f"  - [w{item.weight}] {item.text}" for item in items)


def extract_requirements_from_text(
    job_description: str, llm: ChatAnthropic | None = None
) -> JobRequirements:
    """Parse a free-text job description into structured :class:`JobRequirements`."""
    llm = llm or get_chat_model()
    structured = llm.with_structured_output(JobRequirements)
    result = structured.invoke(
        [
            ("system", prompts.EXTRACT_REQUIREMENTS_SYSTEM),
            ("user", prompts.EXTRACT_REQUIREMENTS_USER.format(job_description=job_description)),
        ]
    )
    logger.info(
        "Extracted requirements: %d must-have, %d nice-to-have.",
        len(result.must_have),
        len(result.nice_to_have),
    )
    return result


def score_candidate(
    candidate: Candidate,
    requirements: JobRequirements,
    llm: ChatAnthropic | None = None,
) -> CandidateScore:
    """Score a single candidate against requirements with reasoning."""
    llm = llm or get_chat_model()
    structured = llm.with_structured_output(CandidateScore)
    score = structured.invoke(
        [
            ("system", prompts.SCORE_CANDIDATE_SYSTEM),
            (
                "user",
                prompts.SCORE_CANDIDATE_USER.format(
                    title=requirements.title or "(unspecified)",
                    min_years=requirements.min_years_experience,
                    must_have=_format_requirements(requirements.must_have),
                    nice_to_have=_format_requirements(requirements.nice_to_have),
                    candidate_id=candidate.candidate_id,
                    resume=candidate.content,
                ),
            ),
        ]
    )
    # Preserve provenance even if the model omits identity fields.
    score.candidate_id = candidate.candidate_id
    if not score.name:
        score.name = candidate.name
    return score


def rank_candidates(
    candidates: list[Candidate],
    requirements: JobRequirements,
    llm: ChatAnthropic | None = None,
) -> list[CandidateScore]:
    """Score and rank a list of candidates (descending by overall score)."""
    llm = llm or get_chat_model()
    scores = [score_candidate(candidate, requirements, llm) for candidate in candidates]
    scores.sort(key=lambda s: s.overall_score, reverse=True)
    return scores


def synthesize_report(
    requirements: JobRequirements,
    ranked: list[CandidateScore],
    llm: ChatAnthropic | None = None,
) -> MatchReport:
    """Produce an executive-summary :class:`MatchReport` from ranked candidates."""
    llm = llm or get_chat_model()
    ranked_block = "\n".join(
        f"- {s.name or s.candidate_id} (id={s.candidate_id}): score={s.overall_score}, "
        f"rec={s.recommendation}; gaps={'; '.join(s.gaps) or 'none'}"
        for s in ranked
    ) or "(no candidates)"

    summary_msg = llm.invoke(
        [
            ("system", prompts.REPORT_SYSTEM),
            (
                "user",
                prompts.REPORT_USER.format(
                    title=requirements.title or "(unspecified)",
                    ranked_block=ranked_block,
                ),
            ),
        ]
    )
    summary = summary_msg.content if isinstance(summary_msg.content, str) else str(summary_msg.content)
    return MatchReport(job_title=requirements.title, ranked=ranked, summary=summary)


def initial_screen(
    candidates: list[Candidate],
    requirements: JobRequirements,
    shortlist_size: int,
    llm: ChatAnthropic | None = None,
) -> list[Candidate]:
    """Round 1 — fast triage of a large pool down to the top ``shortlist_size``.

    Performed in a single LLM call (compact previews) for cost efficiency. Falls
    back to retrieval order if the model returns nothing usable.
    """
    if len(candidates) <= shortlist_size:
        return candidates

    llm = llm or get_chat_model()
    by_id = {c.candidate_id: c for c in candidates}
    pool_block = "\n".join(
        f"- id={c.candidate_id} | {c.name or 'N/A'} | {c.preview(280)}" for c in candidates
    )
    structured = llm.with_structured_output(ScreenSelection)
    try:
        selection: ScreenSelection = structured.invoke(
            [
                ("system", prompts.INITIAL_SCREEN_SYSTEM.format(shortlist_size=shortlist_size)),
                (
                    "user",
                    prompts.INITIAL_SCREEN_USER.format(
                        title=requirements.title or "(unspecified)",
                        must_have=_format_requirements(requirements.must_have),
                        pool=pool_block,
                        shortlist_size=shortlist_size,
                    ),
                ),
            ]
        )
        ordered = [by_id[cid] for cid in selection.selected_ids if cid in by_id]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Initial screen failed (%s); falling back to retrieval order.", exc)
        ordered = []

    if not ordered:
        ordered = candidates[:shortlist_size]
    logger.info("Initial screen: %d pool → %d shortlist.", len(candidates), len(ordered[:shortlist_size]))
    return ordered[:shortlist_size]


def final_recommendations(
    finalists: list[CandidateScore],
    requirements: JobRequirements,
    llm: ChatAnthropic | None = None,
) -> list[FinalAssessment]:
    """Round 3 — hire/no-hire decision + improvement suggestions for each finalist."""
    if not finalists:
        return []

    settings = get_settings()
    llm = llm or get_chat_model()
    scorecards = "\n\n".join(
        f"- id={s.candidate_id} | {s.name or 'N/A'} | score={s.overall_score} "
        f"| rec={s.recommendation}\n"
        f"  strengths: {', '.join(s.strengths) or 'none'}\n"
        f"  gaps: {', '.join(s.gaps) or 'none'}\n"
        f"  must-have missing: {', '.join(s.must_have_missing) or 'none'}\n"
        f"  reasoning: {s.reasoning}"
        for s in finalists
    )
    structured = llm.with_structured_output(FinalAssessments)
    result: FinalAssessments = structured.invoke(
        [
            (
                "system",
                prompts.FINAL_RECOMMENDATION_SYSTEM.format(
                    borderline_low=settings.borderline_low,
                    borderline_high=settings.borderline_high,
                ),
            ),
            (
                "user",
                prompts.FINAL_RECOMMENDATION_USER.format(
                    title=requirements.title or "(unspecified)",
                    scorecards=scorecards,
                ),
            ),
        ]
    )

    # Restore provenance / names from the scorecards in case the model omits them.
    score_by_id = {s.candidate_id: s for s in finalists}
    assessments: list[FinalAssessment] = []
    for item in result.items:
        if item.candidate_id in score_by_id and not item.name:
            item.name = score_by_id[item.candidate_id].name
        assessments.append(item)

    # Ensure every finalist has an assessment, even if the model skipped one.
    assessed_ids = {a.candidate_id for a in assessments}
    for score in finalists:
        if score.candidate_id not in assessed_ids:
            assessments.append(_fallback_assessment(score, settings))
    return assessments


def _fallback_assessment(score: CandidateScore, settings) -> FinalAssessment:
    """Derive a deterministic assessment from a score when the LLM omits one."""
    if score.must_have_missing or score.overall_score < settings.borderline_low:
        decision = "no_hire"
    elif score.overall_score < settings.borderline_high:
        decision = "borderline"
    else:
        decision = "hire"
    return FinalAssessment(
        candidate_id=score.candidate_id,
        name=score.name,
        decision=decision,
        confidence=score.overall_score,
        rationale=score.reasoning or "Derived from deep-analysis scorecard.",
        improvement_suggestions=(
            [f"Close gap: {g}" for g in score.gaps] if decision == "borderline" else []
        ),
    )


def make_interview_questions(
    candidate: Candidate,
    role_context: str,
    llm: ChatAnthropic | None = None,
) -> str:
    """Generate targeted screening questions for a candidate."""
    llm = llm or get_chat_model()
    message = llm.invoke(
        [
            ("system", prompts.INTERVIEW_SYSTEM),
            (
                "user",
                prompts.INTERVIEW_USER.format(
                    role_context=role_context or "(general screening)",
                    resume=candidate.content,
                ),
            ),
        ]
    )
    return message.content if isinstance(message.content, str) else str(message.content)
