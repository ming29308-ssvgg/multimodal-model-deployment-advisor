# Model Navigator Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished FastAPI-served frontend that accepts a task and visualizes the hardware-aware model recommendation response.

**Architecture:** FastAPI serves a static HTML/CSS/JS application from the same origin as the API. The client calls the existing recommendation route, renders an Agent narrative and structured comparisons, and draws an original pixel world map on canvas.

**Tech Stack:** FastAPI, vanilla HTML5, CSS, JavaScript Canvas, pytest, Playwright-based local web testing.

**Spec:** `docs/superpowers/specs/2026-11-09-model-navigator-frontend-design.md`

## Global Constraints

- REQUIRED DESIGN SUB-SKILL: use `frontend-design` before authoring the interface.
- Keep `/docs` and `/api/v1/model-selector/recommend` unchanged.
- No Node build, CDN, external font, or runtime dependency on the reference image.
- The page must work at 1440px desktop and 390px mobile widths.
- Form submission must expose loading, success, and retryable error states.
- Respect `prefers-reduced-motion`.

## Review Focus

- `/` must return the page while `/docs` remains accessible.
- Empty task submission must be blocked client-side and remain validated server-side.
- API errors must restore the form and show an actionable message.
- Long Chinese model/reason text must wrap without horizontal overflow.
- Mobile layout must preserve readable task input, candidate comparison, and commands.

---

### Task 1: Static frontend delivery

**Files:**
- Modify: `backend/main.py`
- Create: `backend/static/index.html`
- Test: `tests/test_frontend.py`

**Interfaces:**
- Produces: `GET /` HTML page and `/static/*` asset delivery.
- Preserves: existing `/health`, `/docs`, and API routes.

- [ ] Write failing route tests for `/`, static CSS/JS, and preserved `/docs`.
- [ ] Run targeted tests and verify RED.
- [ ] Mount StaticFiles and serve the index.
- [ ] Add the semantic HTML shell.
- [ ] Run targeted tests and verify GREEN.

### Task 2: Visual system and responsive layout

**Files:**
- Create: `backend/static/app.css`
- Modify: `backend/static/index.html`

**Interfaces:**
- Consumes: semantic page shell.
- Produces: dark pixel-world hero, task console, result panels, candidate comparison, loading/error states.

- [ ] Implement the frontend-design visual direction without external assets.
- [ ] Add desktop, tablet, mobile, reduced-motion, focus, and overflow rules.
- [ ] Render desktop and mobile screenshots in a real browser and inspect layout.

### Task 3: API integration and result rendering

**Files:**
- Create: `backend/static/app.js`
- Modify: `tests/test_frontend.py`

**Interfaces:**
- Consumes: `POST /api/v1/model-selector/recommend`.
- Produces: rendered Agent answer, hardware inventory, pipeline, candidates, deployment plan, warnings, and commands.

- [ ] Add browser-level failing behavior checks for input submission and visible candidate output.
- [ ] Implement deterministic canvas world map and API request lifecycle.
- [ ] Implement safe DOM rendering without raw API HTML injection.
- [ ] Verify success, empty input, and API error behavior.

### Task 4: Documentation and end-to-end verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Produces: startup and frontend usage instructions.

- [ ] Document `/`, `/docs`, and the page workflow.
- [ ] Run the full pytest suite and compileall.
- [ ] Start Uvicorn and complete a real browser submission.
- [ ] Capture desktop and mobile screenshots for visual review.

