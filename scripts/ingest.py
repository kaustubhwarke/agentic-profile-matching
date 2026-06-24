"""Build or refresh the resume vector index.

Usage:
    python -m scripts.ingest
"""

from __future__ import annotations

from profile_matching.config import get_settings
from profile_matching.logging_config import configure_logging, get_logger
from profile_matching.rag.ingestion import ingest_resumes

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    settings = get_settings()
    settings.ensure_runtime_dirs()
    count = ingest_resumes(settings)
    if count == 0:
        print(
            "No resumes were indexed. Generate samples first:\n"
            "  python -m scripts.generate_sample_data"
        )
    else:
        print(f"Indexed {count} resume(s) into '{settings.collection_name}'.")


if __name__ == "__main__":
    main()
