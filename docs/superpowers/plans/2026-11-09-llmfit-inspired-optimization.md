# LLMFit-inspired Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add execution variants, utilization-based fit, confidence metadata, hardware simulation, and an optional llmfit text-model adapter to the existing LangGraph recommendation Agent.

**Architecture:** The LangGraph flow remains intact while the selection node delegates to a local fit engine and an optional fail-open llmfit client. Hardware overrides are applied before environment routing, and the same additive response fields are rendered in the existing frontend.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, LangGraph, urllib, vanilla HTML/CSS/JavaScript, pytest, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-09-llmfit-inspired-optimization-design.md`

## Global Constraints

- Existing request bodies without overrides must remain valid.
- llmfit is optional and must never make the recommendation endpoint unavailable.
- llmfit is only eligible for text, code, and embedding stages.
- Simulated hardware must bypass remote llmfit results.
- No new Node build or mandatory runtime dependency.
- All new estimates must expose their confidence and basis.
- The ten primary capability families must each contain at least ten distinct local catalog models.
- API and frontend candidate limits must accept values from 1 through 10 while retaining a default of 3.

## Review Focus

- An unreachable or malformed llmfit service falls back to local candidates.
- A video-generation task never receives a vision-understanding candidate from llmfit.
- Zero available RAM/VRAM does not divide by zero and produces an unsupported fit.
- A hardware override changes routing and result hardware without mutating the detected inventory.
- Existing clients that omit all new fields continue to receive a successful response.

---

### Task 1: Hardware simulation and additive schemas

**Files:**
- Modify: `backend/schemas.py`
- Modify: `backend/agents/model_selector/state.py`
- Modify: `backend/agents/model_selector/nodes.py`
- Modify: `backend/agents/model_selector/graph.py`
- Test: `tests/test_graph.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `HardwareOverride`, `apply_hardware_override()`, and effective hardware in graph state.
- Preserves: current automatic detection when no override is provided.

- [ ] Write failing tests for simulated 24GB and 2×48GB server overrides.
- [ ] Run targeted tests and verify RED.
- [ ] Implement validated override merging before environment routing.
- [ ] Add `allow_offload` and `use_llmfit` preferences with backward-compatible defaults.
- [ ] Run targeted tests and verify GREEN.

### Task 2: Execution variants, utilization fit, and confidence

**Files:**
- Create: `backend/agents/model_selector/fit_engine.py`
- Modify: `backend/agents/model_selector/state.py`
- Modify: `backend/agents/model_selector/matcher.py`
- Test: `tests/test_fit_engine.py`
- Modify: `tests/test_matcher.py`

**Interfaces:**
- Consumes: effective `HardwareInventory`, `ModelSpec`, and `allow_offload`.
- Produces: selected execution variant, utilization ratios, confidence, basis, and fit tier.

- [ ] Write failing tests for 60/85/98% boundaries, zero resources, quantized selection, and CPU offload.
- [ ] Run targeted tests and verify RED.
- [ ] Implement pure variant generation and fit evaluation.
- [ ] Integrate the selected estimate into candidate ranking and explanations.
- [ ] Run targeted tests and verify GREEN.

### Task 3: Optional llmfit text-model adapter

**Files:**
- Create: `backend/agents/model_selector/llmfit_client.py`
- Modify: `backend/agents/model_selector/nodes.py`
- Modify: `backend/agents/model_selector/graph.py`
- Modify: `.env.example`
- Test: `tests/test_llmfit_client.py`
- Modify: `tests/test_graph.py`

**Interfaces:**
- Consumes: `LLMFIT_BASE_URL`, eligible task stage, effective hardware, and result limit.
- Produces: normalized `CandidateRecommendation` values or an empty list on failure.

- [ ] Write failing parser and fail-open tests using an injected HTTP fetcher.
- [ ] Run targeted tests and verify RED.
- [ ] Implement stable-field parsing and use-case mapping.
- [ ] Inject the adapter into the LangGraph selection node and merge/deduplicate candidates.
- [ ] Verify simulated and non-text stages bypass the adapter.
- [ ] Run targeted tests and verify GREEN.

### Task 4: Expanded offline catalog and ten-candidate API

**Files:**
- Create: `backend/agents/model_selector/models_extended.json`
- Modify: `backend/agents/model_selector/catalog.py`
- Modify: `backend/schemas.py`
- Modify: `backend/agents/model_selector/nodes.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_matcher.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: existing `ModelSpec` validation and matcher ranking.
- Produces: a merged, unique local catalog with at least ten models for each primary capability family and a request limit of 10.

- [ ] Write failing coverage tests for ten distinct models in each primary category.
- [ ] Write a failing API test requesting ten candidates for a simulated high-resource server.
- [ ] Run targeted tests and verify RED.
- [ ] Load and validate both core and extended catalog resources with duplicate-ID protection.
- [ ] Add the missing model records with conservative resource estimates and modality-specific runtimes.
- [ ] Raise schema and graph-node limits from 5 to 10.
- [ ] Run targeted tests and verify GREEN.

### Task 5: Frontend simulation controls and estimate display

**Files:**
- Modify: `backend/static/index.html`
- Modify: `backend/static/app.css`
- Modify: `backend/static/app.js`
- Modify: `tests/test_frontend.py`
- Modify: `.superpowers/sdd/2026-11-09-model-navigator-frontend/browser_flow_check.py`

**Interfaces:**
- Consumes: additive request and response fields from Tasks 1–3.
- Produces: environment preset requests and visible source/mode/utilization/confidence details.

- [ ] Add failing route/browser checks for the target-environment control and displayed estimate metadata.
- [ ] Run checks and verify RED.
- [ ] Implement presets and request payload generation.
- [ ] Render new candidate metadata with safe DOM operations.
- [ ] Add a ten-candidate option to the candidate-count selector.
- [ ] Verify desktop/mobile success and retry states.

### Task 6: Documentation and final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Produces: setup instructions for optional llmfit and hardware simulation.

- [ ] Document default fallback behavior, `LLMFIT_BASE_URL`, and simulation presets.
- [ ] Run the full pytest suite, compileall, and JavaScript syntax check.
- [ ] Restart Uvicorn and complete one detected-hardware and one simulated-server browser submission.
- [ ] Perform final whole-change review and record any rulings or deferred minor findings.

