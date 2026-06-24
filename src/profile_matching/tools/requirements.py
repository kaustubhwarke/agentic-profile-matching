"""`extract_requirements` tool — parse a JD into must-have vs nice-to-have."""

from __future__ import annotations

from langchain_core.tools import tool

from profile_matching.agent.services import extract_requirements_from_text


@tool
def extract_requirements(job_description: str) -> str:
    """Parse a job description into structured must-have vs nice-to-have requirements.

    Args:
        job_description: The full job description text.

    Returns:
        A human-readable breakdown of the role's requirements, including weights
        and the minimum years of experience.
    """
    reqs = extract_requirements_from_text(job_description)
    lines = [f"Title: {reqs.title or '(unspecified)'}"]
    if reqs.summary:
        lines.append(f"Summary: {reqs.summary}")
    lines.append(f"Minimum experience: {reqs.min_years_experience:g} years\n")

    lines.append("MUST-HAVE:")
    lines.extend(f"  - [weight {i.weight}] {i.text} ({i.category})" for i in reqs.must_have)
    if not reqs.must_have:
        lines.append("  (none)")

    lines.append("\nNICE-TO-HAVE:")
    lines.extend(f"  - [weight {i.weight}] {i.text} ({i.category})" for i in reqs.nice_to_have)
    if not reqs.nice_to_have:
        lines.append("  (none)")

    return "\n".join(lines)
