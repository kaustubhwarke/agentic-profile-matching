"""High-level driver for the matching state machine.

Wraps the compiled graph and its interrupt/resume lifecycle behind a small API
that the CLI, Streamlit UI, and batch scripts all share.
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from profile_matching.agent.graph import build_matching_graph
from profile_matching.logging_config import get_logger
from profile_matching.models.domain import MatchReport

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Outcome of a pipeline step.

    Either the run paused for human feedback (``awaiting_feedback`` True, with
    ``interrupt_payload`` populated), or it completed (``report`` populated).
    """

    report: MatchReport | None
    awaiting_feedback: bool
    interrupt_payload: dict | None = None


class MatchingPipeline:
    """Drives the matching graph, including the human-in-the-loop refinement."""

    def __init__(self, checkpointer: BaseCheckpointSaver | None = None) -> None:
        self._graph = build_matching_graph(checkpointer)

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _interpret(self, result: dict, thread_id: str) -> PipelineResult:
        """Translate a raw graph result into a :class:`PipelineResult`.

        Detects a pending human-feedback interrupt two ways for cross-version
        robustness: the ``__interrupt__`` key in the invoke output, and (as a
        fallback) pending tasks with interrupts on the state snapshot.
        """
        payload = self._extract_interrupt(result, thread_id)
        if payload is not None:
            return PipelineResult(
                report=self._current_report(thread_id),
                awaiting_feedback=True,
                interrupt_payload=payload,
            )
        return PipelineResult(report=result.get("report"), awaiting_feedback=False)

    def _extract_interrupt(self, result: dict, thread_id: str):
        """Return the interrupt payload if the graph is paused, else ``None``."""
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            first = interrupts[0]
            return getattr(first, "value", first)

        snapshot = self._graph.get_state(self._config(thread_id))
        if snapshot is None or not getattr(snapshot, "next", None):
            return None
        for task in getattr(snapshot, "tasks", ()) or ():
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                first = task_interrupts[0]
                return getattr(first, "value", first)
        return None

    def _current_report(self, thread_id: str) -> MatchReport | None:
        snapshot = self._graph.get_state(self._config(thread_id))
        return snapshot.values.get("report") if snapshot else None

    def start(self, job_description: str, thread_id: str = "default") -> PipelineResult:
        """Run the pipeline from a job description until the first pause/end."""
        result = self._graph.invoke(
            {"job_description": job_description, "refinement_count": 0},
            config=self._config(thread_id),
        )
        return self._interpret(result, thread_id)

    def resume(self, feedback: str, thread_id: str = "default") -> PipelineResult:
        """Resume a paused run with human feedback (refine or approve)."""
        result = self._graph.invoke(
            Command(resume=feedback),
            config=self._config(thread_id),
        )
        return self._interpret(result, thread_id)
