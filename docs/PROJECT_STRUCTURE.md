# Project Structure

The package/file hierarchy of the Agentic Profile Matching system, with the
role of each file. The project uses a standard `src/` layout.

```
agentic-profile-matching/
├── README.md                     Overview, quick start, usage, state diagram, req. mapping
│
├── docs/                         Project documentation
│   ├── ARCHITECTURE.md           System architecture (components, data flow, decisions)
│   ├── HLD.md                    High-level design (subsystems, NFRs, deployment)
│   ├── LLD.md                    Low-level design (modules, classes, contracts)
│   ├── IMPLEMENTATION_PLAN.md    Phased delivery plan + status + verification checklist
│   └── PROJECT_STRUCTURE.md      This file
│
├── pyproject.toml                Packaging, dependencies, tool config (ruff/mypy/pytest)
├── requirements.txt              Pinned runtime dependencies
├── .env.example                  Documented environment configuration template
├── .gitignore                    Ignore rules (venvs, secrets, vector store, checkpoints…)
├── Makefile                      Developer task runner (install/seed/ingest/chat/ui/test…)
│
├── data/                         Sample corpus (ships with the repo)
│   ├── jobs/                     Sample job descriptions
│   │   ├── frontend_react_role.txt
│   │   └── data_engineer_role.txt
│   └── resumes/                  Sample resumes (one candidate per file)
│       ├── alice_nguyen.txt          (senior frontend — strong React/TS)
│       ├── bob_patel.txt             (junior frontend — 2 yrs)
│       ├── carla_garcia.txt          (full-stack)
│       ├── david_smith.txt           (backend)
│       ├── elena_rossi.txt           (data engineer)
│       └── farid_haddad.txt          (ML engineer)
│
├── scripts/                      Operational entry points
│   ├── __init__.py
│   ├── generate_sample_data.py   Synthesize diverse resumes + JDs (default 100)
│   ├── ingest.py                 Build / refresh the resume vector index
│   ├── run_pipeline.py           Run the full matching pipeline non-interactively (Part A)
│   └── run_screening.py          Run the multi-round screening non-interactively (Part C)
│
├── src/profile_matching/         The application package
│   ├── __init__.py               Version + package docstring
│   ├── py.typed                  PEP 561 typing marker
│   ├── config.py                 Settings (pydantic-settings) + get_settings()
│   ├── logging_config.py         configure_logging(), get_logger() — text/JSON
│   ├── llm.py                    Claude chat-model factory (no temperature; effort)
│   │
│   ├── models/                   Typed domain vocabulary
│   │   ├── __init__.py
│   │   ├── domain.py             RequirementItem, JobRequirements, Candidate,
│   │   │                         CandidateScore, MatchReport + Part C:
│   │   │                         ScreenSelection, FinalAssessment, ScreeningReport
│   │   └── state.py              AgentState + ScreeningState (TypedDicts)
│   │
│   ├── rag/                      Retrieval-Augmented Generation layer
│   │   ├── __init__.py
│   │   ├── embeddings.py         get_embeddings() — local sentence-transformers (lazy)
│   │   ├── vector_store.py       ResumeVectorStore facade over Chroma (lazy)
│   │   └── ingestion.py          load_resume_documents(), ingest_resumes()
│   │
│   ├── tools/                    The agent's tool surface
│   │   ├── __init__.py           all_tools()
│   │   ├── filesystem.py         list_files, read_file, write_file (SANDBOXED) — Milestone 1
│   │   ├── rag_search.py         search_resumes — Milestone 2
│   │   ├── requirements.py       extract_requirements
│   │   ├── comparison.py         compare_candidates
│   │   ├── interview.py          generate_interview_questions
│   │   └── screening.py          screen_candidates — multi-round screening (Part C)
│   │
│   ├── agent/                    Orchestration (the brains)
│   │   ├── __init__.py
│   │   ├── prompts.py            All prompt templates (extraction/scoring/report/…)
│   │   ├── services.py           extract / score / rank / report / interview /
│   │   │                         initial_screen / final_recommendations (LLM seam)
│   │   ├── nodes.py              Graph node functions + route_after_feedback
│   │   ├── graph.py              build_matching_graph() — the state machine (Part A)
│   │   ├── pipeline.py           MatchingPipeline / PipelineResult (start/resume driver)
│   │   ├── conversation.py       ConversationalAgent — ReAct agent (Part B)
│   │   └── screening.py          build_screening_graph() + ScreeningPipeline (Part C)
│   │
│   ├── cli/                      Command-line interface
│   │   ├── __init__.py
│   │   └── chat.py               Rich REPL (chat + /pipeline + /screen + /history) — entry point
│   │
│   └── app/                      Web interface
│       ├── __init__.py
│       └── streamlit_app.py      Streamlit chat UI + matching-pipeline + screening panels
│
└── tests/                        Test suite (runs offline; LLM + store stubbed)
    ├── __init__.py
    ├── conftest.py               Fixtures: tmp_settings, fake_store, stub_services
    ├── test_models.py            Domain-model unit tests
    ├── test_filesystem_tools.py  Sandbox security + behaviour tests
    ├── test_ingestion.py         Resume-loading tests
    ├── test_routing.py           Human-feedback routing tests
    ├── test_graph.py             Matching state-machine integration (incl. HITL) — Part A
    ├── test_screening.py         Multi-round screening graph integration — Part C
    └── scenarios/
        ├── __init__.py
        └── test_conversation_flows.py   8 conversation-flow scenarios (A/B/C)
```

## Runtime-generated paths (git-ignored)

| Path | Created by | Purpose |
|---|---|---|
| `chroma_db/` | `scripts/ingest.py` | Persistent vector index. |
| `checkpoints/` | the graph (if SQLite saver configured) | LangGraph checkpoints. |
| `reports/` | `write_file` tool | Persisted match reports. |
| `logs/` | logging (if file handler added) | Logs. |
| `.venv/` | you | Virtual environment. |
| `.env` | you (from `.env.example`) | Secrets & config. |

## Entry points

| Command | Target |
|---|---|
| `profile-matching` (console script) | `profile_matching.cli.chat:main` |
| `python -m profile_matching.cli.chat` | CLI REPL |
| `streamlit run src/profile_matching/app/streamlit_app.py` | Web UI |
| `python -m scripts.generate_sample_data` | Seed data |
| `python -m scripts.ingest` | Build index |
| `python -m scripts.run_pipeline [jd_path]` | Batch matching pipeline (Part A) |
| `python -m scripts.run_screening [jd_path]` | Batch multi-round screening (Part C) |
