"""Streamlit chat UI for the agentic profile matching system.

Run with:  streamlit run src/profile_matching/app/streamlit_app.py

Provides a conversational chat surface (Part B) plus a sidebar "Matching
pipeline" panel that runs the structured workflow (Part A) with a
human-in-the-loop refinement box.
"""

from __future__ import annotations

import uuid

import streamlit as st

from profile_matching.agent.conversation import ConversationalAgent
from profile_matching.agent.pipeline import MatchingPipeline
from profile_matching.agent.screening import ScreeningPipeline
from profile_matching.config import get_settings
from profile_matching.logging_config import configure_logging
from profile_matching.rag.vector_store import get_vector_store

configure_logging()

st.set_page_config(page_title="Agentic Profile Matching", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    if "agent" not in st.session_state:
        st.session_state.agent = ConversationalAgent(thread_id=st.session_state.thread_id)
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list[(role, text)]
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "awaiting_feedback" not in st.session_state:
        st.session_state.awaiting_feedback = False
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "screening_report" not in st.session_state:
        st.session_state.screening_report = None


_init_state()
settings = get_settings()


# ---------------------------------------------------------------------------
# Sidebar — status + matching pipeline
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Status")
    if not settings.has_api_key:
        st.error("ANTHROPIC_API_KEY not set.")
    try:
        index_count = get_vector_store().count()
        st.metric("Indexed resumes", index_count)
        if index_count == 0:
            st.warning("Run ingestion: `python -m scripts.ingest`")
    except Exception as exc:  # pragma: no cover
        st.warning(f"Vector store unavailable: {exc}")
    st.caption(f"Model: `{settings.llm_model}` · effort `{settings.llm_effort}`")

    st.divider()
    st.header("🧭 Matching pipeline")
    jd_text = st.text_area("Job description", height=200, placeholder="Paste a JD here…")
    if st.button("Run matching", use_container_width=True, disabled=not jd_text.strip()):
        st.session_state.pipeline = MatchingPipeline()
        with st.spinner("Extracting requirements, searching, ranking…"):
            result = st.session_state.pipeline.start(
                jd_text, thread_id=st.session_state.thread_id + "-pipe"
            )
        st.session_state.pipeline_result = result
        st.session_state.awaiting_feedback = result.awaiting_feedback

    if st.session_state.pipeline_result is not None:
        result = st.session_state.pipeline_result
        report = result.report
        if report and report.summary:
            st.subheader("Executive summary")
            st.write(report.summary)
        if report and report.ranked:
            st.subheader("Ranked shortlist")
            for rank, score in enumerate(report.ranked, start=1):
                with st.expander(
                    f"{rank}. {score.name or score.candidate_id} — "
                    f"{score.overall_score}/100 ({score.recommendation})"
                ):
                    st.markdown(f"**Reasoning:** {score.reasoning}")
                    st.markdown(f"**Strengths:** {', '.join(score.strengths) or '—'}")
                    st.markdown(f"**Gaps:** {', '.join(score.gaps) or '—'}")
                    st.markdown(
                        f"**Must-have met:** {', '.join(score.must_have_met) or '—'}  \n"
                        f"**Must-have missing:** {', '.join(score.must_have_missing) or '—'}"
                    )

        if st.session_state.awaiting_feedback:
            st.info("Human-in-the-loop: refine the criteria or approve.")
            feedback = st.text_input("Refinement (or type 'approve')", key="pipe_feedback")
            if st.button("Submit feedback", use_container_width=True):
                with st.spinner("Re-ranking…"):
                    result = st.session_state.pipeline.resume(
                        feedback or "approve",
                        thread_id=st.session_state.thread_id + "-pipe",
                    )
                st.session_state.pipeline_result = result
                st.session_state.awaiting_feedback = result.awaiting_feedback
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — multi-round screening (Part C)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.header("🔬 Multi-round screening")
    st.caption("Initial screen → deep analysis → hire/no-hire recommendation.")
    screen_jd = st.text_area("Job description ", height=160, key="screen_jd", placeholder="Paste a JD…")
    if st.button("Run screening", use_container_width=True, disabled=not screen_jd.strip()):
        with st.spinner("Screening across three rounds…"):
            st.session_state.screening_report = ScreeningPipeline().run(screen_jd)

    sr = st.session_state.screening_report
    if sr is not None:
        st.caption(
            f"Round 1: {sr.pool_size} → {len(sr.shortlist_ids)} shortlisted · "
            f"Round 2: {len(sr.finalists)} analysed · "
            f"Round 3: {len(sr.hires())} hire / {len(sr.borderline())} borderline"
        )
        with st.expander("Full screening report", expanded=True):
            st.markdown(sr.render_markdown())


# ---------------------------------------------------------------------------
# Main — conversational chat
# ---------------------------------------------------------------------------
st.title("🧭 Agentic Profile Matching")
st.caption(
    "Ask in natural language — e.g. *“Find candidates with React and 3+ years”*, "
    "*“Compare the top 3 side by side”*, *“Why did Alice rank higher than Bob?”*"
)

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Message the recruiting assistant…")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = st.session_state.agent.send(prompt)
            except Exception as exc:  # pragma: no cover
                reply = f"Error: {exc}"
        st.markdown(reply)
    st.session_state.chat.append(("assistant", reply))
