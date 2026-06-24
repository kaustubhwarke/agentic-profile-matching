# Architecture

This document describes the architecture of the Agentic Profile Matching
system: its components, how data flows through them, and the design decisions
behind them.

---

## 1. Overview

The system is an **agentic recruiting assistant**. Given a job description and a
corpus of resumes, it extracts structured requirements, retrieves candidates via
semantic search, ranks them with reasoned scoring, produces a decision-ready
report, and refines the shortlist through a human-in-the-loop loop (Part A). A
parallel **conversational agent** answers free-form recruiting questions over the
same knowledge base (Part B). A dedicated **multi-round screening** pipeline
triages a large pool to a shortlist, deeply analyses finalists, and issues
explainable hire/no-hire recommendations (Part C).

It is built on four pillars:

1. **LangGraph** — deterministic agent orchestration (the matching state machine
   and the ReAct conversational agent).
2. **Claude** (`claude-opus-4-8` via `langchain-anthropic`) — requirement
   extraction, candidate scoring, report synthesis, and conversation.
3. **RAG** — a persistent Chroma vector store of resume embeddings
   (sentence-transformers, local/offline).
4. **Typed domain models** (Pydantic) — a shared, validated vocabulary used both
   as LLM structured-output schemas and as the in-memory representation.

---

## 2. Component view

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Interfaces                                    │
│   cli/chat.py  (Rich REPL)            app/streamlit_app.py  (web chat)     │
└───────────────┬───────────────────────────────────┬───────────────────────┘
                │                                     │
                ▼                                     ▼
┌───────────────────────┐  ┌──────────────────────────┐  ┌────────────────────┐
│ agent/pipeline.py      │  │ agent/conversation.py    │  │ agent/screening.py │
│ MatchingPipeline (A)   │  │ ConversationalAgent (B)  │  │ ScreeningPipeline  │
│ drives matching graph  │  │ ReAct + tool surface     │  │ 3-round graph (C)  │
└──────────┬─────────────┘  └────────────┬─────────────┘  └─────────┬──────────┘
           │                             │                          │
           ▼                             ▼                          ▼
┌───────────────────────┐  ┌──────────────────────────┐  ┌────────────────────┐
│ agent/graph.py+nodes.py│  │ tools/ (filesystem, rag, │  │ ScreeningState     │
│ StateGraph(AgentState) │  │ requirements, comparison,│  │ rounds 1/2/3 nodes │
│                        │  │ interview, screening)    │  │                    │
└──────────┬─────────────┘  └────────────┬─────────────┘  └─────────┬──────────┘
           │                             │                          │
           └─────────────────┬───────────────────────────┬──────────┘
                             ▼
              ┌────────────────────────────────────────┐
              │   agent/services.py                     │
              │   extract / score / rank / report /      │
              │   interview  (all LLM calls funnel here) │
              └───────┬─────────────────────────┬────────┘
                      ▼                          ▼
        ┌──────────────────────┐   ┌────────────────────────────┐
        │ llm.py (ChatAnthropic)│   │ rag/ (embeddings, store,   │
        │ Claude Opus 4.8       │   │ ingestion) → Chroma        │
        └──────────────────────┘   └────────────────────────────┘

        config.py · logging_config.py  (cross-cutting)
```

### Layers

| Layer | Modules | Responsibility |
|---|---|---|
| **Interface** | `cli/`, `app/` | User I/O; never contain business logic. |
| **Orchestration** | `agent/graph.py`, `agent/nodes.py`, `agent/pipeline.py`, `agent/conversation.py`, `agent/screening.py` | Control flow: the matching state machine, the ReAct loop, and the multi-round screening graph. |
| **Tools** | `tools/` | The agent's callable capabilities (LangChain `@tool`s). |
| **Domain services** | `agent/services.py` | Every LLM interaction in one testable place. |
| **Knowledge / RAG** | `rag/` | Embeddings, vector store, ingestion. |
| **Models** | `models/` | Typed domain objects + agent state. |
| **Platform** | `config.py`, `logging_config.py`, `llm.py` | Configuration, logging, the LLM factory. |

---

## 3. The matching state machine (Part A)

`agent/graph.py` compiles a `StateGraph(AgentState)` with the required topology:

```
START → parse_jd → extract_requirements → search_resumes →
rank_candidates → generate_report → human_feedback ──(refine)──► extract_requirements
                                                    └─(approve)──► finalize → END
```

- **parse_jd** — normalises the incoming JD, initialises the refinement counter.
- **extract_requirements** — `services.extract_requirements_from_text` →
  structured `JobRequirements` (must-have vs nice-to-have, weights, min years).
  Folds in any human feedback from a prior loop.
- **search_resumes** — turns requirements into a query and retrieves a candidate
  pool from the vector store.
- **rank_candidates** — `services.rank_candidates` scores each candidate against
  the requirements with explicit reasoning and a recommendation.
- **generate_report** — `services.synthesize_report` writes an executive summary.
- **human_feedback** — calls LangGraph `interrupt(...)`, pausing the graph and
  surfacing the shortlist; resumes with `Command(resume=<feedback>)`.
- **route_after_feedback** — approval keywords (or hitting
  `APM_MAX_REFINEMENT_LOOPS`) → `finalize`; anything else → loop to
  `extract_requirements`.

A **checkpointer** (default `MemorySaver`) is mandatory for the interrupt to
persist across the pause; `MatchingPipeline` drives the
`start()` → (pause) → `resume()` cycle.

### Agent state

`models/state.py` defines `AgentState`, satisfying the "Agent State Design"
requirement:

| Requirement | Field | Reducer |
|---|---|---|
| Conversation history | `messages` | `add_messages` (append-merge) |
| Job-requirements understanding | `job_description`, `requirements` | last-write |
| Candidate shortlist + reasoning | `candidates`, `report` | last-write |
| HITL control | `feedback`, `refinement_count`, `is_complete` | last-write |

---

## 4. The conversational agent (Part B)

`agent/conversation.py` builds a ReAct agent (`langgraph.prebuilt.create_react_agent`)
bound to the full tool surface and backed by a checkpointer keyed on
`thread_id`, so conversation history persists across turns. Its system prompt
directs it to: search before answering, re-rank and **explain changes** on
refinement, and ground "why X over Y" explanations in retrieved evidence.

**Two complementary surfaces, one knowledge base:** the structured graph is
ideal for the canonical JD→report workflow with auditable steps; the ReAct agent
is ideal for open-ended exploration and iterative refinement. Both call the same
`services.py` and the same vector store, so behaviour is consistent.

---

## 5. Multi-round screening (Part C)

`agent/screening.py` compiles a second, linear `StateGraph(ScreeningState)` for
the advanced screening workflow:

```
START → extract_requirements → retrieve_pool → initial_screen (round 1)
      → deep_analysis (round 2) → final_recommendation (round 3)
      → compile_report → END
```

- **retrieve_pool** — retrieves up to `APM_SCREEN_POOL_SIZE` (default 100)
  resumes from the vector store ("from 100").
- **initial_screen (round 1)** — `services.initial_screen` performs a single,
  cost-efficient LLM triage call over compact previews and advances the top
  `APM_SCREEN_SHORTLIST_SIZE` (default 10) candidates ("top 10"). Falls back to
  retrieval order if the model returns nothing usable.
- **deep_analysis (round 2)** — `services.rank_candidates` runs full, reasoned
  scoring on the shortlist (per-candidate `CandidateScore`).
- **final_recommendation (round 3)** — `services.final_recommendations` issues a
  `hire` / `no_hire` / `borderline` decision per finalist with a confidence and
  **improvement suggestions for borderline candidates**. A deterministic
  fallback derives a decision from the score if the model omits a finalist.
- **compile_report** — assembles a `ScreeningReport` whose `render_markdown()`
  is the explainability surface (executive summary + per-candidate strengths,
  gaps, must-have coverage, and improvement suggestions).

This graph is **linear and non-interactive** (no human-in-the-loop), so it needs
no checkpointer — a deliberate contrast with the Part-A matching graph. It is
driven by `ScreeningPipeline.run()` and exposed both as the `screen_candidates`
tool and as a batch script (`scripts/run_screening.py`).

The borderline band (`APM_BORDERLINE_LOW`..`APM_BORDERLINE_HIGH`, default
50..70) is configuration-driven and shared between the round-3 prompt and the
deterministic fallback.

---

## 6. Tool surface

| Tool | Module | Notes |
|---|---|---|
| `list_files`, `read_file`, `write_file` | `tools/filesystem.py` | **Sandboxed** to an allow-list of dirs; path-traversal rejected. |
| `search_resumes` | `tools/rag_search.py` | Semantic retrieval (Milestone 2). |
| `extract_requirements` | `tools/requirements.py` | Must-have vs nice-to-have. |
| `compare_candidates` | `tools/comparison.py` | Head-to-head, evidence-cited. |
| `generate_interview_questions` | `tools/interview.py` | Validate strengths, probe gaps. |
| `screen_candidates` | `tools/screening.py` | Full Part-C multi-round screening (returns the explainable report). |

---

## 7. Data flow

**Ingestion (offline):** `data/resumes/*` → `rag/ingestion.py` parses
(`.txt/.md/.pdf/.docx`) → `Document`s with stable metadata → embeddings →
persistent Chroma collection.

**Matching (online, Part A):** JD → `extract_requirements` → `as_search_query()`
→ vector search → per-candidate scoring → ranked `MatchReport` → human feedback →
(refine | finalize).

**Conversation (online, Part B):** user message → ReAct agent → tool calls
(search/compare/interview/extract/screen/fs) → grounded answer → persisted to thread.

**Screening (online, Part C):** JD → `extract_requirements` → retrieve pool
(≤100) → round-1 triage (→ top 10) → round-2 deep scoring → round-3 hire/no-hire
+ suggestions → `ScreeningReport.render_markdown()`.

---

## 8. Cross-cutting concerns

- **Configuration** — `config.py` (`pydantic-settings`), one typed, cached
  `Settings` object; secrets via `SecretStr`; never hard-coded.
- **Logging** — `logging_config.py`, text or JSON, level-controlled; noisy
  third-party loggers quieted.
- **Security** — file tools confined to an allow-list with traversal rejection;
  resume PII stays local (offline embeddings); API key only via env/`SecretStr`.
- **Resilience** — LLM client retries (`max_retries=3`); refinement-loop cap
  prevents runaway loops; empty-index and unknown-candidate paths degrade
  gracefully.
- **Performance** — cached singletons (`Settings`, embeddings, vector store, chat
  model); lazy import of heavy ML deps (torch/chromadb) so non-RAG paths stay
  light. The round-1 screen is a **single** LLM call over the whole pool (not one
  call per resume), bounding cost as the corpus grows.

---

## 9. Key design decisions

| Decision | Rationale |
|---|---|
| **LangGraph state machine** for the core flow | The assignment mandates the graph; explicit nodes give auditability and a natural HITL interrupt point. |
| **Separate ReAct agent** for conversation | Open-ended NL queries fit a tool-calling loop better than a fixed pipeline. |
| **All LLM calls in `services.py`** | One testable seam; nodes and tools never duplicate prompt logic. |
| **Pydantic structured outputs** | Deterministic, validated extraction/scoring; same types flow through state. |
| **Local sentence-transformers embeddings** | Offline, no per-call cost, keeps resume PII on-box. |
| **Chroma persistent store** | Simple, embedded, production-capable vector DB; no external service to operate for a demo. |
| **`claude-opus-4-8`, no `temperature`, `effort=high`** | Latest/most capable model; Opus 4.8 rejects sampling params — depth is steered via `effort`. |
| **Lazy heavy imports** | Fast CLI cold-start; unit tests run without torch/chromadb. |
| **Separate linear graph for screening (Part C)** | The 3-round screen is deterministic and non-interactive; a dedicated graph keeps it free of the matching graph's HITL machinery and easy to reason about. |
| **Round-1 triage as one LLM call** | Screens 100→10 cheaply; deep (per-candidate) scoring is reserved for the 10 finalists. |
| **Config-driven borderline band** | Explainability/improvement-suggestion behaviour is tunable without code changes. |

See [`HLD.md`](HLD.md) and [`LLD.md`](LLD.md) for the high- and low-level design.
