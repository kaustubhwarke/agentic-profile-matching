"""Agent tools.

Exposes the full tool surface available to the conversational agent:

* File-system tools (Milestone 1): :func:`list_files`, :func:`read_file`, :func:`write_file`.
* RAG search tool (Milestone 2): :func:`search_resumes`.
* Domain tools: :func:`extract_requirements`, :func:`compare_candidates`,
  :func:`generate_interview_questions`.
* Advanced (Part C): :func:`screen_candidates` — full multi-round screening.
"""

from __future__ import annotations

from profile_matching.tools.comparison import compare_candidates
from profile_matching.tools.filesystem import list_files, read_file, write_file
from profile_matching.tools.interview import generate_interview_questions
from profile_matching.tools.rag_search import search_resumes
from profile_matching.tools.requirements import extract_requirements
from profile_matching.tools.screening import screen_candidates


def all_tools() -> list:
    """Return every tool, ready to bind to the conversational agent."""
    return [
        list_files,
        read_file,
        write_file,
        search_resumes,
        extract_requirements,
        compare_candidates,
        generate_interview_questions,
        screen_candidates,
    ]


__all__ = [
    "all_tools",
    "compare_candidates",
    "extract_requirements",
    "generate_interview_questions",
    "list_files",
    "read_file",
    "screen_candidates",
    "search_resumes",
    "write_file",
]
