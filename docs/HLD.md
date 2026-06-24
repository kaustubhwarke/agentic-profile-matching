# High-Level Design (HLD)

## 1. Purpose & scope

The Agentic Profile Matching system helps a hiring manager find, rank, screen,
and compare candidates against a job description. This HLD describes the system
at the subsystem level: responsibilities, interactions, interfaces, and
non-functional characteristics. Module- and class-level detail is in
[`LLD.md`](LLD.md).

In scope: requirement extraction, semantic candidate retrieval, reasoned
ranking, report generation, human-in-the-loop refinement, conversational
querying, CLI and web interfaces.

Out of scope (future): ATS integration, authentication/multi-tenancy,
distributed/managed vector DB, resume PII redaction pipeline.

---

## 2. Context diagram

```
        ┌─────────────┐        job description / NL queries        ┌──────────────┐
        │ Hiring      │ ─────────────────────────────────────────▶ │  Agentic     │
        │ manager     │ ◀───────────────────────────────────────── │  Profile     │
        └─────────────┘     shortlist / report / answers           │  Matching    │
                                                                     │  system      │
        ┌─────────────┐        resume corpus (files)                └──────┬───────┘
        │ Resume      │ ───────────────────────────────────────────────────┘
        │ store (fs)  │
        └─────────────┘                       │ inference
                                               ▼
                                       ┌───────────────┐
                                       │ Anthropic API │  (Claude Opus 4.8)
                                       └───────────────┘
```

External dependencies: the Anthropic API (LLM) and a local Hugging Face model
cache (embeddings). No other external services are required.

---

## 3. Subsystems

| # | Subsystem | Responsibility | Key modules |
|---|---|---|---|
| 1 | **Interface** | Capture user input, render results | `cli/chat.py`, `app/streamlit_app.py` |
| 2 | **Matching orchestration** | Run the structured JD→report workflow with HITL (Part A) | `agent/graph.py`, `agent/nodes.py`, `agent/pipeline.py` |
| 3 | **Conversational orchestration** | Tool-calling NL agent with memory (Part B) | `agent/conversation.py` |
| 3b | **Screening orchestration** | Run the 3-round screening + explainability (Part C) | `agent/screening.py` |
| 4 | **Tooling** | The agent's callable capabilities | `tools/*` |
| 5 | **Domain services** | All LLM-backed operations | `agent/services.py`, `agent/prompts.py` |
| 6 | **Knowledge / RAG** | Embed, store, retrieve resumes | `rag/*` |
| 7 | **Domain model** | Typed entities + agent state | `models/*` |
| 8 | **Platform** | Config, logging, LLM factory | `config.py`, `logging_config.py`, `llm.py` |

---

## 4. Subsystem interactions

### 4.1 Structured matching (Part A)

1. Interface hands a JD to `MatchingPipeline.start()`.
2. The compiled graph runs `parse_jd → extract_requirements → search_resumes →
   rank_candidates → generate_report` and pauses at `human_feedback`.
3. The pipeline returns the report + the interrupt payload to the interface.
4. The interface collects feedback and calls `MatchingPipeline.resume()`.
5. `route_after_feedback` either loops to `extract_requirements` (re-rank on new
   criteria) or proceeds to `finalize`.

### 4.2 Conversation (Part B)

1. Interface forwards an NL message to `ConversationalAgent.send()`.
2. The ReAct agent plans, calls tools (search/compare/interview/extract/fs),
   observes results, and produces a grounded answer.
3. The full exchange is persisted under the session `thread_id` (checkpointer),
   enabling multi-turn refinement and "explain the change" behaviour.

### 4.3 Multi-round screening (Part C)

1. Interface (or the `screen_candidates` tool) hands a JD to
   `ScreeningPipeline.run()`.
2. The screening graph extracts requirements, retrieves a pool of up to 100
   resumes, fast-triages to the top 10 (round 1), deeply scores those finalists
   (round 2), and issues hire/no-hire/borderline decisions with improvement
   suggestions (round 3).
3. `compile_report` returns a `ScreeningReport`; `render_markdown()` produces the
   explainable, decision-ready document.

All three paths converge on `agent/services.py` (LLM ops) and `rag/`
(retrieval), guaranteeing consistent reasoning across surfaces.

---

## 5. Interface contracts (high level)

| Interface | Operation | Input | Output |
|---|---|---|---|
| `MatchingPipeline` | `start(jd, thread_id)` | JD text | `PipelineResult` (report, awaiting_feedback, interrupt payload) |
| `MatchingPipeline` | `resume(feedback, thread_id)` | feedback text | `PipelineResult` |
| `ConversationalAgent` | `send(message)` | NL text | answer text |
| `ScreeningPipeline` | `run(jd)` | JD text | `ScreeningReport` (rounds 1–3 + assessments) |
| `ResumeVectorStore` | `search(query, top_k)` | query | `list[Candidate]` |
| Tools | `@tool` callables | typed args | string result for the LLM |

---

## 6. Data design (high level)

- **Resume corpus**: files in `data/resumes/` (`.txt/.md/.pdf/.docx`).
- **Vector index**: persistent Chroma collection (`APM_COLLECTION_NAME`) of
  resume embeddings + metadata (`candidate_id`, `name`, `source`).
- **Domain entities**: `JobRequirements`, `Candidate`, `CandidateScore`,
  `MatchReport`, plus Part-C `ScreenSelection`, `FinalAssessment`,
  `ScreeningReport` (see [`LLD.md`](LLD.md)).
- **Agent state**: `AgentState` (matching) and `ScreeningState` (screening)
  threaded through their respective graphs; conversation history via
  `add_messages`.
- **Checkpoints**: LangGraph checkpointer (in-memory by default; SQLite path
  available via config for durability).

---

## 7. Non-functional requirements

| Attribute | Approach |
|---|---|
| **Security** | Sandboxed file tools (allow-list + traversal rejection); secrets via `SecretStr`/env; local embeddings keep resume PII on-box. |
| **Reliability** | LLM retries; refinement-loop cap; graceful empty-index / not-found handling. |
| **Observability** | Structured logging (text/JSON), level-controlled; optional LangSmith tracing via env. |
| **Configurability** | All knobs via env (`APM_*`); typed, validated settings. |
| **Testability** | LLM and vector store are stubbable; pure logic isolated; offline test suite. |
| **Performance** | Cached singletons; lazy heavy imports; top-K retrieval bounds LLM scoring cost. |
| **Maintainability** | Layered packages; single LLM seam; prompts centralised. |
| **Portability** | Pure Python; embedded vector DB; runs locally on Windows/macOS/Linux. |

---

## 8. Deployment view

A single Python process hosts both the CLI and (separately) the Streamlit app.
The vector store persists to disk (`APM_VECTOR_STORE_DIR`). The only network
dependency is the Anthropic API. For production hardening, the Chroma store can
be swapped for a managed vector DB and the in-memory checkpointer for the SQLite
(or a database-backed) saver — both are isolated behind their factories.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM cost on large candidate pools | `APM_RETRIEVAL_TOP_K` bounds the pool scored per run. |
| Runaway refinement loops | `APM_MAX_REFINEMENT_LOOPS` cap forces finalize. |
| Resume PII leaving the box | Local embeddings; only resume text relevant to a query reaches the LLM. |
| Model/API drift | Single LLM factory; model id and effort are config-driven. |
| Heavy ML deps slow startup | Lazy imports of torch/chromadb. |
