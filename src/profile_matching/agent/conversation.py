"""Conversational agent (Part B).

A tool-calling ReAct agent that handles natural-language recruiting queries,
iterative refinement, and explanations. It is backed by a checkpointer so the
full conversation history persists across turns (keyed by ``thread_id``),
satisfying the "track conversation history" and "iterative refinement"
requirements.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from profile_matching.agent.prompts import CONVERSATION_SYSTEM
from profile_matching.llm import get_chat_model
from profile_matching.logging_config import get_logger
from profile_matching.tools import all_tools

logger = get_logger(__name__)


def build_conversational_agent(checkpointer: BaseCheckpointSaver | None = None):
    """Build a compiled ReAct agent bound to the recruiting tool surface.

    The system-prompt keyword of ``create_react_agent`` was renamed across
    LangGraph versions (``state_modifier`` → ``prompt``). We try the modern name
    and transparently fall back, so the agent builds on any pinned 0.2.x/0.3.x.
    """
    model = get_chat_model()
    tools = all_tools()
    saver = checkpointer or MemorySaver()
    try:
        return create_react_agent(
            model=model, tools=tools, prompt=CONVERSATION_SYSTEM, checkpointer=saver
        )
    except TypeError:
        return create_react_agent(
            model=model, tools=tools, state_modifier=CONVERSATION_SYSTEM, checkpointer=saver
        )


class ConversationalAgent:
    """Ergonomic wrapper around the compiled ReAct agent.

    Maintains a stable ``thread_id`` so every ``send`` shares one persisted
    conversation. Construct one per user session.
    """

    def __init__(
        self,
        thread_id: str = "default",
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._agent = build_conversational_agent(checkpointer)
        self._config = {"configurable": {"thread_id": thread_id}}

    def send(self, message: str) -> str:
        """Send a user message and return the agent's final text response."""
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=self._config,
        )
        final: BaseMessage = result["messages"][-1]
        return final.content if isinstance(final.content, str) else str(final.content)

    def stream_events(self, message: str) -> Iterator[BaseMessage]:
        """Stream agent messages (tool calls, tool results, final answer) as they occur."""
        for chunk in self._agent.stream(
            {"messages": [HumanMessage(content=message)]},
            config=self._config,
            stream_mode="values",
        ):
            messages = chunk.get("messages", [])
            if messages:
                yield messages[-1]

    def history(self) -> list[BaseMessage]:
        """Return the persisted conversation history for this thread."""
        snapshot = self._agent.get_state(self._config)
        return snapshot.values.get("messages", []) if snapshot else []

    @staticmethod
    def is_final_answer(message: BaseMessage) -> bool:
        """True when ``message`` is the assistant's final (non-tool-calling) answer."""
        return isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
