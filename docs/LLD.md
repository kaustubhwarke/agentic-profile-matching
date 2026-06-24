# Low-Level Design (LLD)

Module-, class-, and contract-level design. Pairs with [`HLD.md`](HLD.md)
(subsystem view) and [`ARCHITECTURE.md`](ARCHITECTURE.md) (component view).

---

## 1. Package map

```
profile_matching/
├── config.py            Settings (pydantic-settings) + get_settings()
├── logging_config.py    configure_logging(), get_logger()
├── llm.py               build_chat_model(), get_chat_model()
├── models/
│   ├── domain.py        RequirementItem, JobRequirements, Candidate,
│   │                    CandidateScore, MatchReport, + Part C:
│   │                    ScreenSelection, FinalAssessment, FinalAssessments,
│   │                    ScreeningReport
│   └── state.py         AgentState, ScreeningState (TypedDict)
├── rag/
│   ├── embeddings.py    get_embeddings()  [lazy HF import]
│   ├── vector_store.py  ResumeVectorStore, get_vector_store()  [lazy Chroma]
│   └── ingestion.py     load_resume_documents(), ingest_resumes()
├── tools/
│   ├── filesystem.py    list_files, read_file, write_file  (sandboxed)
│   ├── rag_search.py    search_resumes
│   ├── requirements.py  extract_requirements
│   ├── comparison.py    compare_candidates
│   ├── interview.py     generate_interview_questions
│   ├── screening.py     screen_candidates  (Part C)
│   └── __init__.py      all_tools()
├── agent/
│   ├── prompts.py       all prompt templates
│   ├── services.py      extract / score / rank / report / interview /
│   │                    initial_screen / final_recommendations
│   ├── nodes.py         matching graph node functions + routing
│   ├── graph.py         build_matching_graph()
│   ├── pipeline.py      MatchingPipeline, PipelineResult
│   ├── conversation.py  ConversationalAgent, build_conversational_agent()
│   └── screening.py     build_screening_graph(), ScreeningPipeline  (Part C)
├── cli/chat.py          Rich REPL
└── app/streamlit_app.py Streamlit UI
```

---

## 2. Domain models (`models/domain.py`)

All are Pydantic v2 `BaseModel`s; the requirement/score models double as LLM
structured-output schemas.

### `RequirementItem`
| Field | Type | Notes |
|---|---|---|
| `text` | `str` | The requirement. |
| `category` | `Literal[skill,experience,education,responsibility,other]` | default `skill`. |
| `weight` | `int` 1–5 | importance. |

### `JobRequirements`
| Field | Type |
|---|---|
| `title` | `str` |
| `summary` | `str` |
| `must_have` | `list[RequirementItem]` |
| `nice_to_have` | `list[RequirementItem]` |
| `min_years_experience` | `float ≥ 0` |

Method `as_search_query() -> str` renders a dense-retrieval query from title +
requirements + min years (falls back to `summary`).

### `Candidate`
`candidate_id`, `name`, `content` (full resume), `relevance_score` (retrieval
similarity). Method `preview(limit)` truncates content.

### `CandidateScore`
`candidate_id`, `name`, `overall_score` (0–100), `must_have_met`,
`must_have_missing`, `strengths`, `gaps`, `reasoning`,
`recommendation` (`strong_yes|yes|maybe|no`).

### `MatchReport`
`job_title`, `ranked: list[CandidateScore]`, `summary`. Method `top(n)` returns
the n highest-scoring candidates.

### Part C models
- `ScreenSelection` — round-1 structured output: `selected_ids: list[str]`
  (best-first), `reasoning`.
- `FinalAssessment` — `candidate_id`, `name`, `decision`
  (`hire|no_hire|borderline`), `confidence` 0–100, `rationale`,
  `improvement_suggestions: list[str]`.
- `FinalAssessments` — `items: list[FinalAssessment]` (wrapper for one structured
  round-3 call).
- `ScreeningReport` — `job_title`, `pool_size`, `shortlist_ids`,
  `finalists: list[CandidateScore]`, `assessments: list[FinalAssessment]`,
  `summary`. Methods: `hires()`, `borderline()`, `render_markdown()` (the
  explainability surface).

---

## 3. Agent state (`models/state.py`)

`AgentState(TypedDict, total=False)`:

| Channel | Type | Reducer | Maps to requirement |
|---|---|---|---|
| `messages` | `Annotated[list, add_messages]` | append-merge | conversation history |
| `job_description` | `str` | last-write | — |
| `requirements` | `JobRequirements` | last-write | job-requirements understanding |
| `candidates` | `list[Candidate]` | last-write | shortlist |
| `report` | `MatchReport` | last-write | shortlist + reasoning |
| `feedback` | `str` | last-write | HITL control |
| `refinement_count` | `int` | last-write | HITL control |
| `is_complete` | `bool` | last-write | HITL control |

`total=False` lets each node return a partial update dict that LangGraph merges
channel-wise.

**`ScreeningState`** (Part C) is a separate `TypedDict(total=False)` for the
linear screening graph: `job_description`, `requirements`, `pool: list[Candidate]`
(round-1 retrieval), `shortlist: list[Candidate]` (advanced to round 2),
`finalists: list[CandidateScore]` (round-2 deep scores),
`assessments: list[FinalAssessment]` (round-3), and `report: ScreeningReport`.
It has no `messages`/feedback channels — the screening flow is non-interactive.

---

## 4. Domain services (`agent/services.py`)

Pure functions; each accepts an optional `llm` override for testing.

| Function | Signature | Behaviour |
|---|---|---|
| `extract_requirements_from_text` | `(jd: str, llm?) -> JobRequirements` | `llm.with_structured_output(JobRequirements)`. |
| `score_candidate` | `(Candidate, JobRequirements, llm?) -> CandidateScore` | structured scoring; restores provenance ids. |
| `rank_candidates` | `(list[Candidate], JobRequirements, llm?) -> list[CandidateScore]` | scores each, sorts desc by `overall_score`. |
| `synthesize_report` | `(JobRequirements, list[CandidateScore], llm?) -> MatchReport` | executive-summary text. |
| `make_interview_questions` | `(Candidate, role_context, llm?) -> str` | screening questions. |
| `initial_screen` | `(list[Candidate], JobRequirements, shortlist_size, llm?) -> list[Candidate]` | **Part C round 1** — one LLM triage call; falls back to retrieval order. |
| `final_recommendations` | `(list[CandidateScore], JobRequirements, llm?) -> list[FinalAssessment]` | **Part C round 3** — hire/no-hire + suggestions; deterministic `_fallback_assessment` ensures every finalist is assessed. |

All prompts live in `agent/prompts.py`. This module is the single seam where the
LLM is invoked for domain operations; nodes and tools call it.

---

## 5. Graph nodes (`agent/nodes.py`)

Each node: `(AgentState) -> dict` (partial update). Returns never mutate input.

| Node | Reads | Writes |
|---|---|---|
| `parse_jd` | `job_description` | normalised JD, `refinement_count=0`, status message |
| `extract_requirements` | `job_description`, `feedback` | `requirements` (feedback folded in) |
| `search_resumes` | `requirements` | `candidates` (via `get_vector_store().search`) |
| `rank_candidates` | `candidates`, `requirements` | `report` (ranked scores) |
| `generate_report` | `requirements`, `report` | `report` with `summary` |
| `human_feedback` | `report` | `feedback`, `refinement_count += 1` (via `interrupt`) |
| `finalize` | — | `is_complete=True`, clears `feedback` |

`route_after_feedback(state) -> "refine" | "finalize"`: approval keywords
(`approve/approved/accept/done/ok/looks good/yes/""`) **or**
`refinement_count >= APM_MAX_REFINEMENT_LOOPS` → `finalize`; else `refine`.

---

## 6. Graph assembly (`agent/graph.py`)

`build_matching_graph(checkpointer=None) -> CompiledGraph`:

- Adds the seven nodes; wires the linear edges `START→parse_jd→…→human_feedback`.
- Conditional edges from `human_feedback`:
  `{"refine": "extract_requirements", "finalize": "finalize"}`; `finalize→END`.
- Compiles with the supplied checkpointer or `MemorySaver()` (required for the
  `interrupt` to persist across the pause).

---

## 7. Pipeline driver (`agent/pipeline.py`)

`PipelineResult` (dataclass): `report: MatchReport|None`,
`awaiting_feedback: bool`, `interrupt_payload: dict|None`.

`MatchingPipeline`:
- `start(jd, thread_id) -> PipelineResult` — invokes the graph with
  `{job_description, refinement_count:0}` under the thread config.
- `resume(feedback, thread_id) -> PipelineResult` — invokes with
  `Command(resume=feedback)`.
- `_interpret(result, thread_id)` — detects `result["__interrupt__"]`; when
  present, returns `awaiting_feedback=True` plus the current report read from the
  graph state snapshot.

---

## 8. Conversational agent (`agent/conversation.py`)

`build_conversational_agent(checkpointer=None)` → `create_react_agent(model,
tools=all_tools(), prompt=CONVERSATION_SYSTEM, checkpointer=...)`.

`ConversationalAgent(thread_id, checkpointer=None)`:
- `send(message) -> str` — invoke, return final message text.
- `stream_events(message) -> Iterator[BaseMessage]` — stream values.
- `history() -> list[BaseMessage]` — read persisted thread state.
- `is_final_answer(message)` — AIMessage without tool calls.

---

## 8a. Screening graph & pipeline (`agent/screening.py`, Part C)

`build_screening_graph() -> CompiledGraph` — a linear `StateGraph(ScreeningState)`:
`START → extract_requirements → retrieve_pool → initial_screen → deep_analysis →
final_recommendation → compile_report → END`. No checkpointer (non-interactive).

Node mapping:

| Node | Service / source | Writes |
|---|---|---|
| `_extract_requirements` | `services.extract_requirements_from_text` | `requirements` |
| `_retrieve_pool` | `get_vector_store().search(top_k=screen_pool_size)` | `pool` |
| `_initial_screen` | `services.initial_screen(..., screen_shortlist_size)` | `shortlist` |
| `_deep_analysis` | `services.rank_candidates` | `finalists` |
| `_final_recommendation` | `services.final_recommendations` | `assessments` |
| `_compile_report` | assembles `ScreeningReport` (+ `synthesize_report` summary) | `report` |

`ScreeningPipeline.run(job_description) -> ScreeningReport` invokes the graph and
returns the compiled report (empty `ScreeningReport` if nothing produced).

---

## 9. Tools (`tools/`)

LangChain `@tool` functions; docstrings are the LLM-facing tool descriptions.

### Filesystem (`filesystem.py`) — security-critical
`_allowed_roots()` = resume/job/report dirs (resolved). `_resolve_within_sandbox(path, for_write)`:
resolves the path (anchoring relative paths to each root), and verifies the
result is `relative_to` an allowed root — else raises `ValueError("Access
denied")`. Rejects `..` traversal, absolute escapes, and out-of-root access.
`list_files`, `read_file`, `write_file` return strings (or `"Error: …"`).

### `rag_search.search_resumes(query, top_k=5)`
Clamps `top_k` to 1–25; formats `Candidate`s with id, name, relevance, preview;
explicit empty-index message.

### `requirements.extract_requirements(job_description)`
Calls the service; renders must-have/nice-to-have with weights.

### `comparison.compare_candidates(candidate_ids, role_context="")`
Requires ≥2 ids; fetches each from the store; reports any not found; LLM
produces an evidence-cited ranking.

### `interview.generate_interview_questions(candidate_id, role_context="")`
Fetches candidate; returns "not found" message if absent; else service output.

### `screening.screen_candidates(job_description)` (Part C)
Runs `ScreeningPipeline().run(jd)` and returns `report.render_markdown()`;
reports an empty-index message when no finalists are produced.

`all_tools()` returns the full list (including `screen_candidates`) bound to the
conversational agent.

---

## 10. RAG layer (`rag/`)

- `embeddings.get_embeddings()` — cached `HuggingFaceEmbeddings`
  (`normalize_embeddings=True`); **lazy import** of `langchain_huggingface`.
- `vector_store.ResumeVectorStore` — facade over Chroma (lazy `langchain_chroma`
  import); `add_documents`, `count`, `search` (returns `Candidate`s via
  `similarity_search_with_relevance_scores`), `get_candidate(id)`.
  `get_vector_store()` is a cached singleton.
- `ingestion.load_resume_documents()` — reads `.txt/.md/.pdf/.docx`, derives
  `candidate_id` (filename stem) and `name` (first non-empty line); skips empty
  / unsupported. `ingest_resumes()` loads + indexes.

---

## 11. Platform modules

- `config.Settings` — `pydantic-settings`, `env_prefix="APM_"`, `ANTHROPIC_API_KEY`
  read without prefix; `has_api_key`, `ensure_runtime_dirs()`.
  `get_settings()` is `lru_cache`d.
- `logging_config` — `configure_logging(force)`, `get_logger(name)`; JSON or
  text formatter; quiets noisy libraries.
- `llm.build_chat_model()` — `ChatAnthropic(model, max_tokens,
  api_key=secret.get_secret_value(), timeout=120, max_retries=3)`; attaches
  `output_config.effort` only for models that support it (Opus 4.x / Sonnet 4.6
  / Fable 5); **never sets `temperature`** (Opus 4.8 rejects it).
  `get_chat_model()` is cached.
- Part-C settings: `screen_pool_size` (default 100), `screen_shortlist_size`
  (default 10), `borderline_low`/`borderline_high` (default 50/70).

---

## 12. Error handling & edge cases

| Case | Handling |
|---|---|
| Empty JD | `parse_jd` short-circuits with `is_complete=True`. |
| Empty candidate pool | `rank_candidates` returns early; search tool reports empty index. |
| Unknown candidate id | comparison/interview tools return a clear message. |
| Path traversal / escape | filesystem tools raise → `"Error: Access denied …"`. |
| Refinement never approved | loop cap forces `finalize`. |
| Missing API key | interfaces warn at startup (`Settings.has_api_key`). |
| Tool failures (I/O) | caught, returned as `"Error: …"` strings the LLM can react to. |

---

## 13. Testing strategy

- **Unit**: domain models (`test_models.py`), filesystem sandbox security
  (`test_filesystem_tools.py`), ingestion loading (`test_ingestion.py`), routing
  (`test_routing.py`).
- **Integration**: full matching state machine (`test_graph.py`) and the
  multi-round screening graph (`test_screening.py`) — both with stubbed services
  + fake store.
- **Scenario**: 8 conversation flows incl. Part C
  (`tests/scenarios/test_conversation_flows.py`).
- LLM and vector store are stubbed via `conftest.py` fixtures
  (`stub_services`, `fake_store`, `tmp_settings`) — the suite runs offline.
