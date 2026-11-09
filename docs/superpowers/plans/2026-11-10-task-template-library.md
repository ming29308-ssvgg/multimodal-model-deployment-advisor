# Task Template Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace narrow example chips with a categorized task-template library and complete the general audio multi-speaker pipeline.

**Architecture:** The classifier receives a small rule extension for audio diarization. The frontend keeps template data in HTML, uses tabs to filter cards, and continues filling the existing textarea without adding a framework.

**Tech Stack:** Python 3.12, FastAPI, vanilla HTML/CSS/JavaScript, pytest, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-10-task-template-library-design.md`

## Global Constraints

- Free-form task input remains primary and editable.
- Templates only advertise supported classifier behavior.
- Tabs are keyboard accessible and responsive.
- No new runtime dependency or frontend build step.

## Review Focus

- Ordinary audio transcription must not require the word “meeting”.
- Audio diarization must not require video input.
- Switching tabs must hide unrelated templates and preserve click-to-fill.
- Mobile layout must remain usable.
- Loading state must disable all template controls.

---

### Task 1: Complete the audio pipeline

**Files:**
- Modify: `backend/agents/model_selector/task_classifier.py`
- Modify: `tests/test_task_classifier.py`

- [ ] Add failing tests for audio transcription + diarization + summary.
- [ ] Verify RED.
- [ ] Extend the audio branch without changing video behavior.
- [ ] Verify GREEN.

### Task 2: Build the categorized template library

**Files:**
- Modify: `backend/static/index.html`
- Modify: `backend/static/app.css`
- Modify: `backend/static/app.js`
- Modify: `tests/test_frontend.py`

- [ ] Add failing structure checks for five categories and representative templates.
- [ ] Verify RED.
- [ ] Implement tabs, template cards, accessibility, and loading-state handling.
- [ ] Verify GREEN.

### Task 3: Browser verification and delivery

**Files:**
- Create: `.superpowers/sdd/2026-11-10-task-template-library/browser_check.py`
- Modify: `README.md`

- [ ] Verify tab switching and audio template click-to-fill in a browser.
- [ ] Submit the multi-speaker audio template and verify the three-stage pipeline.
- [ ] Run the full test suite and syntax checks.
- [ ] Restart ports 8000 and 8765.
