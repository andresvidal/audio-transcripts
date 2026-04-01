from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import BaseTranscriber, Segment, TranscriptResult

DEFAULT_MODEL = "gemini-2.0-flash"

_TRANSCRIPTION_PROMPT = (
    "Transcribe the audio file exactly as spoken. "
    "Output only the transcript text with no additional commentary. "
    "Preserve punctuation and paragraph breaks where natural."
)


class GeminiTranscriber(BaseTranscriber):
    """Cloud transcription via Google Gemini API (multimodal).

    Can transcribe audio and optionally summarize or answer questions in one
    call. No native speaker diarization — returns a single transcript block.

    Uses the new google-genai SDK (File API for upload → generate_content).

    Requirements:
        pip install google-genai
        GOOGLE_API_KEY environment variable (or api_key argument)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, language)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for the gemini backend. "
                "Set it in your .env file or pass api_key=..."
            )
        self.prompt = prompt or _TRANSCRIPTION_PROMPT
        self._client = None

    @property
    def backend_name(self) -> str:
        return "gemini"

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def transcribe(self, file_path: Path, status_callback=None) -> TranscriptResult:
        client = self._get_client()

        # Upload audio via the File API
        uploaded = client.files.upload(path=str(file_path))

        # Poll until the file is ready
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            raise RuntimeError(
                f"Gemini file processing failed for '{file_path.name}'. "
                "Check the file format and try again."
            )

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[uploaded, self.prompt],
            )
            text = (response.text or "").strip()
        finally:
            # Always clean up uploaded file to avoid storage accumulation
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

        # Gemini does not return timestamps; wrap as a single segment
        duration: float = getattr(uploaded, "duration_seconds", None) or 0.0
        segments = [Segment(id=0, start=0.0, end=duration, text=text, speaker=None)]

        return TranscriptResult(
            source_file=file_path,
            duration_seconds=duration,
            language=self.language or "unknown",
            backend=self.backend_name,
            model=self.model,
            diarization=False,
            transcribed_at=datetime.now(timezone.utc),
            segments=segments,
            speakers_detected=[],
        )
