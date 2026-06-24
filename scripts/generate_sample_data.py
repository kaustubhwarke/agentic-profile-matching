"""Generate synthetic resumes and job descriptions for demos and tests.

Produces a configurable number of diverse resumes spanning several role
archetypes plus a couple of job descriptions. Existing static samples in
``data/`` are preserved; this only adds files that do not already exist.

Usage:
    python -m scripts.generate_sample_data            # default: 100 resumes
    python -m scripts.generate_sample_data --count 24
"""

from __future__ import annotations

import argparse
import random

from profile_matching.config import get_settings
from profile_matching.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

FIRST_NAMES = [
    "Alice", "Bob", "Carla", "David", "Elena", "Farid", "Grace", "Hiro",
    "Ines", "Jamal", "Kira", "Liam", "Maya", "Noah", "Omar", "Priya",
    "Quinn", "Rosa", "Sven", "Tara", "Uma", "Viktor", "Wendy", "Xin",
    "Yara", "Zane",
]
LAST_NAMES = [
    "Nguyen", "Patel", "Garcia", "Smith", "Kowalski", "Haddad", "Okafor",
    "Tanaka", "Rossi", "Andersson", "Mehta", "Cohen", "Silva", "Khan",
]

ARCHETYPES = [
    {
        "title": "Senior Frontend Engineer",
        "skills": ["React", "TypeScript", "Redux", "GraphQL", "Jest", "Webpack", "CSS"],
        "domain": "consumer web applications",
    },
    {
        "title": "Backend Engineer",
        "skills": ["Python", "Django", "PostgreSQL", "Redis", "Docker", "REST APIs", "Celery"],
        "domain": "high-throughput payment systems",
    },
    {
        "title": "Full-Stack Engineer",
        "skills": ["React", "Node.js", "TypeScript", "MongoDB", "AWS", "GraphQL"],
        "domain": "SaaS platforms",
    },
    {
        "title": "Data Engineer",
        "skills": ["Python", "Spark", "Airflow", "SQL", "Kafka", "Snowflake", "dbt"],
        "domain": "analytics pipelines",
    },
    {
        "title": "Machine Learning Engineer",
        "skills": ["Python", "PyTorch", "TensorFlow", "MLflow", "Kubernetes", "NLP"],
        "domain": "recommendation systems",
    },
    {
        "title": "DevOps Engineer",
        "skills": ["Kubernetes", "Terraform", "AWS", "CI/CD", "Prometheus", "Go", "Docker"],
        "domain": "cloud infrastructure",
    },
]

RESUME_TEMPLATE = """{name}
{title} | {years} years of experience

SUMMARY
{title} with {years} years building {domain}. Proven track record delivering
production systems and mentoring engineers.

TECHNICAL SKILLS
{skills}

EXPERIENCE
{company_a} — {title} ({y1} years)
  - Led development of {domain} using {lead_skill} and {second_skill}.
  - Improved system reliability and reduced latency through profiling and refactoring.
  - Collaborated cross-functionally with product and design.

{company_b} — Software Engineer ({y2} years)
  - Built and maintained features across the stack using {third_skill}.
  - Wrote automated tests and participated in code review.

EDUCATION
B.S. in Computer Science
"""

COMPANIES = ["Northwind", "Acme Corp", "Globex", "Initech", "Umbrella", "Hooli", "Pied Piper"]

SAMPLE_JOBS = {
    "frontend_react_role.txt": """Senior Frontend Engineer

We are hiring a Senior Frontend Engineer to build our customer-facing web app.

Must have:
- 3+ years of professional experience with React and TypeScript
- Strong understanding of state management (Redux or similar)
- Experience consuming REST or GraphQL APIs
- Solid testing discipline (Jest, React Testing Library)

Nice to have:
- Experience with performance optimization and Webpack
- Familiarity with CI/CD pipelines
- Design-system / accessibility experience

You will own features end to end, mentor junior engineers, and partner closely
with design and product.
""",
    "data_engineer_role.txt": """Data Engineer

We're looking for a Data Engineer to build and own our analytics pipelines.

Must have:
- 4+ years building data pipelines in Python
- Strong SQL and data modelling skills
- Experience with a workflow orchestrator (Airflow or similar)
- Experience with a distributed processing framework (Spark)

Nice to have:
- Kafka / streaming experience
- Snowflake and dbt
- Infrastructure-as-code exposure
""",
}


def _make_resume(rng: random.Random, archetype: dict) -> tuple[str, str]:
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    years = rng.randint(2, 12)
    y1 = max(1, years // 2)
    y2 = max(1, years - y1)
    skills = archetype["skills"]
    rng.shuffle(skills)
    selected = skills[: rng.randint(4, len(skills))]
    body = RESUME_TEMPLATE.format(
        name=name,
        title=archetype["title"],
        years=years,
        domain=archetype["domain"],
        skills="\n".join(f"  - {s}" for s in selected),
        company_a=rng.choice(COMPANIES),
        company_b=rng.choice(COMPANIES),
        y1=y1,
        y2=y2,
        lead_skill=selected[0],
        second_skill=selected[1 % len(selected)],
        third_skill=selected[2 % len(selected)],
    )
    return name, body


def generate(count: int, seed: int = 42) -> int:
    """Generate ``count`` resumes and the sample job descriptions."""
    settings = get_settings()
    settings.resume_dir.mkdir(parents=True, exist_ok=True)
    settings.job_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    created = 0
    for i in range(count):
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        name, body = _make_resume(rng, dict(archetype))  # copy so shuffle is isolated
        slug = f"{name.lower().replace(' ', '_')}_{i:03d}.txt"
        path = settings.resume_dir / slug
        if path.exists():
            continue
        path.write_text(body, encoding="utf-8")
        created += 1

    for filename, content in SAMPLE_JOBS.items():
        path = settings.job_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    logger.info("Generated %d new resume(s) in %s.", created, settings.resume_dir)
    return created


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate synthetic resumes and JDs.")
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of resumes to generate (default 100, matching Part C scale).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    created = generate(args.count, args.seed)
    print(f"Created {created} new resume file(s).")


if __name__ == "__main__":
    main()
