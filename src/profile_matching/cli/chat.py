"""Conversational CLI for the agentic profile matching system.

Two modes in one REPL:

* **Chat mode (default)** — free-form natural-language queries handled by the
  tool-calling conversational agent (Part B).
* **Pipeline mode** (`/pipeline`) — runs the structured matching state machine
  (Part A) end-to-end against a job description, pausing for your feedback at
  the human-in-the-loop step and re-ranking on refinement.

Slash commands:
    /pipeline [path]   Run the matching pipeline (Part A). With a path, the JD is
                       read from that file; otherwise paste the JD then end with EOF.
    /screen [path]     Run the multi-round screening (Part C): initial screen →
                       deep analysis → hire/no-hire recommendation.
    /history           Show the conversation history.
    /help              Show this help.
    /quit, /exit       Leave.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from profile_matching.agent.conversation import ConversationalAgent
from profile_matching.agent.pipeline import MatchingPipeline, PipelineResult
from profile_matching.agent.screening import ScreeningPipeline
from profile_matching.config import get_settings
from profile_matching.logging_config import configure_logging, get_logger
from profile_matching.rag.vector_store import get_vector_store

console = Console()
logger = get_logger(__name__)


def _preflight() -> bool:
    """Validate configuration and index state before starting the REPL."""
    settings = get_settings()
    if not settings.has_api_key:
        console.print(
            "[bold red]ANTHROPIC_API_KEY is not set.[/] Copy .env.example to .env "
            "and add your key.",
        )
        return False
    settings.ensure_runtime_dirs()
    try:
        count = get_vector_store().count()
    except Exception as exc:  # pragma: no cover - environment dependent
        console.print(f"[yellow]Warning: vector store unavailable ({exc}).[/]")
        return True
    if count == 0:
        console.print(
            "[yellow]The resume index is empty.[/] Run [bold]make seed && make ingest[/] "
            "(or [bold]python -m scripts.ingest[/]) to populate it.",
        )
    else:
        console.print(f"[green]Resume index ready: {count} candidate(s).[/]")
    return True


def _read_jd(arg: str) -> str:
    """Resolve a job description from a file path or multi-line paste."""
    if arg:
        path = Path(arg.strip())
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
        console.print(f"[yellow]No file at {path}; treating the argument as the JD text.[/]")
        return arg
    console.print("[dim]Paste the job description. Finish with a line containing only 'EOF'.[/]")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)


def _render_result(result: PipelineResult) -> None:
    if result.report and result.report.summary:
        console.print(Panel(Markdown(result.report.summary), title="Executive summary"))
    if result.report and result.report.ranked:
        console.print("[bold]Ranked shortlist:[/]")
        for rank, score in enumerate(result.report.ranked, start=1):
            console.print(
                f"  {rank}. {score.name or score.candidate_id} "
                f"[cyan]({score.overall_score}/100, {score.recommendation})[/] "
                f"— gaps: {', '.join(score.gaps) or 'none'}"
            )


def _run_pipeline(arg: str, thread_id: str) -> None:
    """Drive the matching pipeline with interactive human feedback."""
    jd = _read_jd(arg)
    if not jd.strip():
        console.print("[red]No job description provided.[/]")
        return

    pipeline = MatchingPipeline()
    console.print("[dim]Running matching pipeline…[/]")
    result = pipeline.start(jd, thread_id=thread_id)
    _render_result(result)

    while result.awaiting_feedback:
        console.print(
            "\n[bold]Human-in-the-loop:[/] enter refined criteria to re-rank, "
            "or 'approve' to finish."
        )
        feedback = Prompt.ask("[bold green]feedback[/]")
        result = pipeline.resume(feedback, thread_id=thread_id)
        _render_result(result)

    console.print("[green]Pipeline complete.[/]")


def _run_screening(arg: str) -> None:
    """Run the multi-round screening workflow (Part C)."""
    jd = _read_jd(arg)
    if not jd.strip():
        console.print("[red]No job description provided.[/]")
        return

    console.print("[dim]Running multi-round screening (this scores each finalist)…[/]")
    with console.status("[dim]screening…[/]"):
        report = ScreeningPipeline().run(jd)

    if not report.finalists and not report.assessments:
        console.print("[yellow]No finalists — is the resume index populated?[/]")
        return

    console.print(
        f"[dim]Round 1: {report.pool_size} resumes → {len(report.shortlist_ids)} shortlisted; "
        f"Round 2: {len(report.finalists)} deep-analysed; "
        f"Round 3: {len(report.hires())} hire, {len(report.borderline())} borderline.[/]"
    )
    console.print(Panel(Markdown(report.render_markdown()), title="Screening report"))


def run_repl() -> None:
    """Run the interactive REPL."""
    configure_logging()
    console.print(
        Panel.fit(
            "[bold]Agentic Profile Matching[/]\n"
            "Chat naturally, or run [bold]/pipeline[/] (match + refine) "
            "or [bold]/screen[/] (multi-round screening).\n"
            "[dim]/help for commands, /quit to exit.[/]",
            border_style="blue",
        )
    )
    if not _preflight():
        sys.exit(1)

    agent = ConversationalAgent(thread_id="cli-session")

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]you[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not user_input:
            continue

        command, _, arg = user_input.partition(" ")
        command = command.lower()

        if command in {"/quit", "/exit"}:
            console.print("[dim]Goodbye.[/]")
            break
        if command == "/help":
            console.print(Markdown(__doc__ or ""))
            continue
        if command == "/history":
            for msg in agent.history():
                role = getattr(msg, "type", "?")
                console.print(f"[dim]{role}:[/] {str(msg.content)[:300]}")
            continue
        if command == "/pipeline":
            _run_pipeline(arg, thread_id="cli-pipeline")
            continue
        if command == "/screen":
            _run_screening(arg)
            continue

        # Default: conversational agent.
        try:
            with console.status("[dim]thinking…[/]"):
                response = agent.send(user_input)
            console.print(Panel(Markdown(response), title="assistant", border_style="green"))
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("Agent error")
            console.print(f"[red]Error: {exc}[/]")


def main() -> None:
    """Console-script entry point."""
    run_repl()


if __name__ == "__main__":
    main()
