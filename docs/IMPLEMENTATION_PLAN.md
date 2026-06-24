# Implementation Plan

A phased plan for delivering the Agentic Profile Matching system, mapped to the
assignment's Part A / Part B and the submission guidelines. All phases are
**complete** in this repository; status is tracked per phase.

---

## Guiding principles

- **Enterprise-grade**: layered packages, typed configuration, structured
  logging, security-by-default, automated tests.
- **Production-ready**: configurable via environment, resilient (retries, loop
  caps, graceful degradation), persistent state and index.
- **Industry standards**: LangGraph for orchestration, Pydantic for contracts,
  RAG for grounded retrieval, `src/` layout, pinned dependencies.
- **Requirement-complete**: every Part-A and Part-B item has a concrete home.

---

## Phase 0 — Foundations  ✅

| Task | Output |
|---|---|
| Project scaffolding (`src/` layout, packaging) | `pyproject.toml`, `requirements.txt` |
| Typed configuration | `config.py` (`pydantic-settings`) |
| Structured logging | `logging_config.py` |
| LLM factory (Claude, no `temperature`, `effort`) | `llm.py` |
| `.gitignore`, `.env.example`, `Makefile` | repo root |

## Phase 1 — Domain model  ✅

| Task | Output |
|---|---|
| Requirements, candidate, score, report models | `models/domain.py` |
| LangGraph agent state (history / requirements / shortlist) | `models/state.py` |

## Phase 2 — RAG / knowledge base (Milestone 2 foundation)  ✅

| Task | Output |
|---|---|
| Local embeddings (sentence-transformers) | `rag/embeddings.py` |
| Persistent Chroma vector store facade | `rag/vector_store.py` |
| Resume ingestion (txt/md/pdf/docx) | `rag/ingestion.py` |
| Ingestion + sample-data scripts | `scripts/ingest.py`, `scripts/generate_sample_data.py` |

## Phase 3 — Tools (Part A: tool surface)  ✅

| Task | Output |
|---|---|
| Sandboxed file-system tools (Milestone 1) | `tools/filesystem.py` |
| RAG search tool (Milestone 2) | `tools/rag_search.py` |
| `extract_requirements` | `tools/requirements.py` |
| `compare_candidates` | `tools/comparison.py` |
| `generate_interview_questions` | `tools/interview.py` |
| Shared LLM-backed services | `agent/services.py`, `agent/prompts.py` |

## Phase 4 — Matching state machine (Part A: agent architecture)  ✅

| Task | Output |
|---|---|
| Graph nodes + routing | `agent/nodes.py` |
| StateGraph assembly (required topology) | `agent/graph.py` |
| Human-in-the-loop via `interrupt` | `agent/nodes.py` + `graph.py` |
| Pipeline driver (start/resume) | `agent/pipeline.py` |
| Non-interactive runner | `scripts/run_pipeline.py` |

## Phase 5 — Conversational agent (Part B: interactive features)  ✅

| Task | Output |
|---|---|
| ReAct agent bound to tools, with memory | `agent/conversation.py` |
| NL queries + iterative refinement + explanation | system prompt + tool surface |

## Phase 6 — Multi-round screening & explainability (Part C)  ✅

| Task | Output |
|---|---|
| Part-C domain models | `models/domain.py` (`ScreenSelection`, `FinalAssessment`, `ScreeningReport`) |
| Screening state | `models/state.py` (`ScreeningState`) |
| Round-1 initial screen (top 10 from 100) | `agent/services.py` (`initial_screen`) + `APM_SCREEN_*` |
| Round-2 deep analysis | `agent/services.py` (`rank_candidates`) |
| Round-3 hire/no-hire + improvement suggestions | `agent/services.py` (`final_recommendations`) |
| Three-round screening graph + driver | `agent/screening.py` |
| Explainable report (strengths/gaps/suggestions) | `ScreeningReport.render_markdown()` |
| Exposed as a tool | `tools/screening.py` (`screen_candidates`) |
| Batch runner | `scripts/run_screening.py` |

## Phase 7 — Interfaces (submission: chat interface)  ✅

| Task | Output |
|---|---|
| Rich CLI (chat + `/pipeline` + `/screen`) | `cli/chat.py` |
| Streamlit web UI (chat + pipeline + screening panels) | `app/streamlit_app.py` |

## Phase 8 — Testing (submission: 5+ conversation flows)  ✅

| Task | Output |
|---|---|
| Unit tests (models, fs sandbox, ingestion, routing) | `tests/test_*.py` |
| Integration tests (matching graph + screening graph) | `tests/test_graph.py`, `tests/test_screening.py` |
| 8 conversation-flow scenarios (A/B/C) | `tests/scenarios/test_conversation_flows.py` |
| Offline test fixtures (stub LLM + store) | `tests/conftest.py` |

## Phase 9 — Documentation  ✅

| Task | Output |
|---|---|
| Architecture / HLD / LLD | `ARCHITECTURE.md`, `HLD.md`, `LLD.md` |
| Project structure | `PROJECT_STRUCTURE.md` |
| This plan | `IMPLEMENTATION_PLAN.md` |
| Usage + state-machine diagram | `README.md` |

---

## Verification checklist

- [x] Sources byte-compile (`python -m compileall src scripts tests`).
- [x] Graph topology matches the required workflow.
- [x] Agent state tracks history, requirements, and shortlist+reasoning.
- [x] All six required tools implemented and bound to the agent.
- [x] Conversational interface supports NL queries and iterative refinement.
- [x] **Part C: multi-round screening (round 1 → 2 → 3) implemented as a graph.**
- [x] **Part C: explainable report with strengths/gaps + borderline improvement suggestions.**
- [x] CLI (`/pipeline`, `/screen`) and Streamlit (pipeline + screening) interfaces.
- [x] 8 conversation-flow scenarios + unit/integration tests (offline).
- [x] Configuration, logging, security, and `.gitignore` in place.

> **Note on running the test suite here:** this sandbox's package index is a
> credential-gated private mirror, so dependencies could not be installed to
> execute `pytest` in-place. All sources compile cleanly, and the suite is
> designed to run offline (LLM + vector store stubbed) once dependencies are
> installed via `pip install -e ".[dev]"`.

---

## Future enhancements (out of scope)

- Parallelised round-2 scoring (async fan-out) for very large shortlists.
- Managed/distributed vector DB and a database-backed LangGraph checkpointer.
- AuthN/Z, multi-tenancy, and ATS integration.
- Resume PII redaction prior to LLM scoring.
