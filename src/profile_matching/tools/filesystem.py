"""Sandboxed file-system tools (Milestone 1).

These tools let the agent inspect job descriptions, resumes, and write match
reports. Every path supplied by the model is **untrusted** — each operation is
confined to an allow-list of root directories (resume, job, and report dirs).
Path traversal (``..``, absolute escapes, symlinks) is rejected before any I/O.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from profile_matching.config import get_settings
from profile_matching.logging_config import get_logger

logger = get_logger(__name__)


def _allowed_roots() -> list[Path]:
    settings = get_settings()
    roots = [settings.resume_dir, settings.job_dir, settings.report_dir]
    return [r.resolve() for r in roots]


def _resolve_within_sandbox(raw_path: str, *, for_write: bool = False) -> Path:
    """Resolve ``raw_path`` and verify it stays within an allowed root.

    Raises:
        ValueError: if the resolved path escapes every allowed root.
    """
    roots = _allowed_roots()
    candidate = Path(raw_path)

    # If a bare name or relative path is given, anchor it to the first matching
    # root that actually contains it; otherwise anchor to the resume dir.
    resolved_candidates: list[Path] = []
    if candidate.is_absolute():
        resolved_candidates.append(candidate.resolve())
    else:
        resolved_candidates.extend((root / candidate).resolve() for root in roots)

    for resolved in resolved_candidates:
        for root in roots:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            if for_write:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            return resolved

    raise ValueError(
        f"Access denied: '{raw_path}' is outside the permitted directories "
        f"({', '.join(str(r) for r in roots)})."
    )


@tool
def list_files(directory: str = "") -> str:
    """List files available to the agent.

    Args:
        directory: Optional sub-path relative to an allowed root. Empty lists
            all allowed roots (resumes, jobs, reports).

    Returns:
        A newline-delimited listing of files, or an error message.
    """
    try:
        if directory:
            target = _resolve_within_sandbox(directory)
            entries = sorted(p.name for p in target.iterdir()) if target.is_dir() else [target.name]
            return "\n".join(entries) or "(empty)"

        lines: list[str] = []
        for root in _allowed_roots():
            lines.append(f"# {root.name}/")
            if root.exists():
                lines.extend(f"  {p.name}" for p in sorted(root.iterdir()) if p.is_file())
            else:
                lines.append("  (missing)")
        return "\n".join(lines)
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"


@tool
def read_file(path: str) -> str:
    """Read the contents of a text file (resume or job description).

    Args:
        path: File name or relative path within an allowed directory.

    Returns:
        The file's text content, or an error message.
    """
    try:
        resolved = _resolve_within_sandbox(path)
        if not resolved.is_file():
            return f"Error: '{path}' is not a readable file."
        return resolved.read_text(encoding="utf-8", errors="ignore")
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file (used to persist match reports).

    Args:
        path: Destination file name or relative path within an allowed directory.
        content: Text to write.

    Returns:
        A confirmation message, or an error message.
    """
    try:
        resolved = _resolve_within_sandbox(path, for_write=True)
        resolved.write_text(content, encoding="utf-8")
        logger.info("Wrote %d bytes to %s", len(content), resolved)
        return f"Wrote {len(content)} characters to {resolved.name}."
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"
