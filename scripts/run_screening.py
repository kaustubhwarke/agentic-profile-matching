"""Run the multi-round screening workflow non-interactively (Part C).

Usage:
    python -m scripts.run_screening [path/to/job_description.txt]

If no path is given, the first job description in the configured job directory
is used. Prints the full, explainable screening report.
"""

from __future__ import annotations

import sys
from pathlib import Path

from profile_matching.agent.screening import ScreeningPipeline
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

    report = ScreeningPipeline().run(jd)
    print("=" * 72)
    print(
        f"Round 1: {report.pool_size} resumes → {len(report.shortlist_ids)} shortlisted | "
        f"Round 2: {len(report.finalists)} deep-analysed | "
        f"Round 3: {len(report.hires())} hire / {len(report.borderline())} borderline"
    )
    print("=" * 72)
    print(report.render_markdown())


if __name__ == "__main__":
    main()
