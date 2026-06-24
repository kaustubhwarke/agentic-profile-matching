"""Run the end-to-end matching pipeline non-interactively (auto-approve).

Useful for smoke-testing the full Part-A workflow and for CI demos.

Usage:
    python -m scripts.run_pipeline [path/to/job_description.txt]

If no path is given, the first job description in the configured job directory
is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

from profile_matching.agent.pipeline import MatchingPipeline
from profile_matching.config import get_settings
from profile_matching.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def _resolve_jd(argv: list[str]) -> str:
    settings = get_settings()
    if len(argv) > 1:
        return Path(argv[1]).read_text(encoding="utf-8", errors="ignore")
    jobs = sorted(settings.job_dir.glob("*.txt")) if settings.job_dir.exists() else []
    if not jobs:
        raise SystemExit(
            "No job description found. Pass a path, or run "
            "`python -m scripts.generate_sample_data` first."
        )
    return jobs[0].read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    configure_logging()
    get_settings().ensure_runtime_dirs()
    jd = _resolve_jd(sys.argv)

    pipeline = MatchingPipeline()
    result = pipeline.start(jd, thread_id="batch")

    # Auto-approve the human-in-the-loop step for non-interactive runs.
    while result.awaiting_feedback:
        result = pipeline.resume("approve", thread_id="batch")

    report = result.report
    if report is None:
        print("Pipeline produced no report.")
        return

    print("=" * 72)
    print(f"JOB: {report.job_title or '(untitled)'}")
    print("=" * 72)
    print(report.summary)
    print("-" * 72)
    for rank, score in enumerate(report.ranked, start=1):
        print(
            f"{rank}. {score.name or score.candidate_id}: "
            f"{score.overall_score}/100 ({score.recommendation})"
        )


if __name__ == "__main__":
    main()
