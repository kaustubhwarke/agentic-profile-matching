"""`screen_candidates` tool — run the full multi-round screening (Part C).

Exposes the three-round screening workflow (initial screen → deep analysis →
final hire/no-hire recommendation) to the conversational agent.
"""

from __future__ import annotations

from langchain_core.tools import tool

from profile_matching.agent.screening import ScreeningPipeline


@tool
def screen_candidates(job_description: str) -> str:
    """Run a full multi-round candidate screening for a job description.

    Round 1 fast-screens the whole resume pool to a shortlist; round 2 performs a
    deep, reasoned analysis of the shortlist; round 3 produces a hire / no-hire /
    borderline recommendation per finalist, with improvement suggestions for
    borderline candidates.

    Args:
        job_description: The full job description text.

    Returns:
        A detailed, explainable screening report (Markdown): executive summary,
        per-candidate strengths and gaps, decisions, and improvement suggestions.
    """
    report = ScreeningPipeline().run(job_description)
    if not report.assessments and not report.finalists:
        return (
            "Screening produced no finalists. The resume index may be empty — "
            "run ingestion first (python -m scripts.ingest)."
        )
    return report.render_markdown()
