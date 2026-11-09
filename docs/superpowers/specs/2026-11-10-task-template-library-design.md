# Task Template Library Design

## Intent

Replace three narrow example chips with a complete, understandable task
template library that reflects the Agent's real text, image, audio, video, and
multimodal capabilities.

## Chosen Design

The task console gains five modality tabs:

- Text & code
- Image
- Audio
- Video
- Multimodal

Each tab displays concise template cards with a title and one-line description.
Clicking a card fills the editable task textarea. The selected tab is keyboard
accessible and uses the existing black/green visual system.

## Template Coverage

- Text & code: writing/summary, knowledge retrieval, translation, code.
- Image: understanding, OCR, generation.
- Audio: transcription, audio understanding, speech synthesis, multi-speaker
  transcription and summary.
- Video: understanding, generation, transcription/subtitles.
- Multimodal: video speech summary and combined image/text analysis.

## Backend Completion

An audio request containing transcription plus speaker separation must produce:

1. speech-to-text;
2. speaker diarization;
3. optional summarization.

This removes the previous meeting/video-only assumption.

## Non-goals

- Building a large searchable template marketplace.
- Adding templates for capabilities the classifier cannot execute.
- Replacing free-form user input.
