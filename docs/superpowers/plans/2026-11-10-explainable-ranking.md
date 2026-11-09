# Explainable Preference Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the weak additive score with a hard-gated, explainable multi-dimensional ranking whose weights materially change for speed, balanced, and quality preferences.

**Architecture:** A pure `scoring.py` module computes relative component scores and applies preference weights after local and optional llmfit candidates are merged. The matcher remains responsible for model fit construction, while the frontend renders the additive score breakdown returned by the API.

**Tech Stack:** Python 3.12, Pydantic, FastAPI, LangGraph, vanilla HTML/CSS/JavaScript, pytest, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-10-explainable-ranking-design.md`

## Global Constraints

- Viable local candidates always rank above unsupported candidates.
- Every component and final score is an integer from 0 through 100.
- Preference weights match the approved table and sum to 100.
- Existing API clients remain valid because all response changes are additive.
- The frontend keeps the current visual language and requires no Node build.

## Review Focus

- A quality preference must not move an unsupported model above a runnable model.
- Missing parameter-size metadata must still produce finite, differentiated component scores.
- A one-candidate stage must not divide by zero during relative normalization.
- llmfit candidates without local catalog metadata must still receive a complete score breakdown.
- Tool stages such as FFmpeg must remain stable and readable.

---

### Task 1: Pure preference scoring engine

**Files:**
- Create: `backend/agents/model_selector/scoring.py`
- Modify: `backend/agents/model_selector/state.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `CandidateRecommendation`, optional `ModelSpec`, `HardwareInventory`, and preference.
- Produces: `rank_candidates(...) -> list[CandidateRecommendation]` with score breakdown and ranking reasons.

- [ ] Write failing tests for exact weights, bounded components, missing metadata, one candidate, and the compatibility gate.
- [ ] Run the tests and verify RED.
- [ ] Implement relative speed/quality signals and weighted scoring.
- [ ] Run the tests and verify GREEN.

### Task 2: Matcher integration and preference behavior

**Files:**
- Modify: `backend/agents/model_selector/matcher.py`
- Modify: `tests/test_matcher.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: Task 1 `rank_candidates`.
- Produces: preference-sensitive ordering and additive API fields.

- [ ] Write failing tests showing different speed/quality rankings for text and image generation.
- [ ] Add a failing test that API candidates expose the complete breakdown.
- [ ] Run targeted tests and verify RED.
- [ ] Replace the legacy additive sort with `rank_candidates`.
- [ ] Run matcher and API tests and verify GREEN.

### Task 3: Frontend score explanation

**Files:**
- Modify: `backend/static/index.html`
- Modify: `backend/static/app.css`
- Modify: `backend/static/app.js`
- Modify: `tests/test_frontend.py`
- Create: `.superpowers/sdd/2026-11-10-explainable-ranking/browser_check.py`

**Interfaces:**
- Consumes: Task 2 additive candidate fields.
- Produces: total score, preference label, component bars, and ranking explanations.

- [ ] Add failing frontend checks for score and breakdown elements.
- [ ] Run checks and verify RED.
- [ ] Implement the score panel with safe DOM operations.
- [ ] Run static checks and Playwright preference-switch flow.

### Task 4: Documentation and final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Produces: documented score semantics and preference weights.

- [ ] Document score dimensions, hard compatibility gate, and heuristic limitations.
- [ ] Run full pytest, compileall, JavaScript syntax check, and browser flow.
- [ ] Restart ports 8000 and 8765 and verify live preference-sensitive requests.
- [ ] Perform final self-review and record rulings.
