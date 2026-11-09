# Model Navigator Frontend Design

## Intent

Create an original one-page interface for the existing model recommendation Agent. The visual direction is inspired by the supplied dark pixel-world reference: black space, low-resolution world map particles, fluorescent green accents, large typography, and terminal-like system feedback.

## Goals

- Serve the page directly from the existing FastAPI application at `/`.
- Let the user enter a natural-language task, choose speed/balanced/quality, and submit it to `/api/v1/model-selector/recommend`.
- Present an Agent-style narrative answer plus detected hardware, task pipeline, concrete model candidates, resource reasons, tradeoffs, warnings, and deployment commands.
- Work without a separate Node/Vite build or external CDN.
- Provide desktop and mobile responsive layouts, keyboard access, loading state, empty/error state, and reduced-motion handling.
- Preserve `/docs` and the existing API.

## Visual Direction

- Original product name: `MODEL//NAV`.
- Palette: near-black `#050706`, graphite gray, soft white, fluorescent green `#2eea8a`.
- Background: a canvas-generated, simplified pixel world map with deterministic gray and green points; do not copy or embed the supplied screenshot.
- Typography: system display and monospace stacks, oversized Chinese hero statement, narrow uppercase labels, deliberate tracking.
- Shape language: square corners, hairline borders, clipped corners, instrument-panel dividers, no generic rounded dashboard cards.
- Motion: subtle map pulse, scan line, staggered result reveal; disabled under `prefers-reduced-motion`.

## Page Structure

1. Navigation: brand, API online indicator, link to Swagger.
2. Hero: statement “让每个任务，找到真正跑得动的模型”.
3. Task console: textarea, example prompt shortcuts, quality selector, candidate count, submit button.
4. Result header: Agent answer and detected environment.
5. Pipeline strip: one item per classified task stage.
6. Candidate comparison: up to five concrete model cards per stage.
7. Deployment plan: steps, warnings, next actions, commands.
8. Footer: local-only notice and API path.

## Interaction

- Submission uses `fetch` with the existing request schema.
- While loading, controls are disabled and the system status changes to “SCANNING HARDWARE”.
- Successful responses scroll results into view and render structured output.
- Failure restores controls and displays a clear retryable error.
- Example task buttons populate the textarea but do not submit automatically.

## Architecture

- `backend/static/index.html`: semantic page shell and templates.
- `backend/static/app.css`: all visual styling and responsive behavior.
- `backend/static/app.js`: map rendering, form state, API request, and result rendering.
- `backend/main.py`: mount `/static` and return the index at `/`.

## Non-goals

- No actual model inference/chat endpoint in this phase.
- No user authentication, history persistence, file uploads, or external analytics.
- No dependency on the supplied image file at runtime.

## Approaches

- Chosen: vanilla HTML/CSS/JS served by FastAPI. It is self-contained, easy to run in the current project, and avoids a second dev server.
- Rejected for this phase: React/Vite, because the interface does not yet need a component build pipeline and would complicate local startup.
- Rejected for this phase: server-rendered Jinja results, because the existing JSON API already supports a richer interactive client.

