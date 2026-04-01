from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import BaseTranscriber, Segment, TranscriptResult

# OpenAI Whisper API hard limit
_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB
_CHUNK_SECONDS = 10 * 60  # 10-minute chunks for large files


class OpenAITranscriber(BaseTranscriber):
    """Cloud transcription via OpenAI Whisper API.

    Handles files larger than 25 MB by splitting them into chunks with ffmpeg
    and stitching the segments back together with correct timestamps.

    No speaker diarization — the API returns a single transcript block.

    Requirements:
        pip install openai
        OPENAI_API_KEY environment variable (or api_key argument)
    """

    def __init__(
        self,
        model: str = "whisper-1",
        language: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, language)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for the openai backend. "
                "Set it in your .env file or pass api_key=..."
            )
        self._client = None

    @property
    def backend_name(self) -> str:
        return "openai"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _transcribe_file(self, audio_path: Path) -> list[dict]:
        """Send one audio file to the API and return its segment dicts."""
        client = self._get_client()
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=self.language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        return response.segments or []

    def _split_audio(self, file_path: Path, chunk_dir: Path) -> list[Path]:
        """Split audio into fixed-duration chunks using ffmpeg."""
        pattern = str(chunk_dir / "chunk_%04d.mp3")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(file_path),
                "-f", "segment",
                "-segment_time", str(_CHUNK_SECONDS),
                "-c", "copy",
                pattern,
            ],
            check=True,
            capture_output=True,
        )
        return sorted(chunk_dir.glob("chunk_*.mp3"))

    def transcribe(self, file_path: Path, status_callback=None) -> TranscriptResult:
        raw_segments: list[dict] = []

        if file_path.stat().st_size > _MAX_FILE_BYTES:
            with tempfile.TemporaryDirectory() as tmp:
                chunks = self._split_audio(file_path, Path(tmp))
                time_offset = 0.0
                for chunk in chunks:
                    chunk_segs = self._transcribe_file(chunk)
                    for s in chunk_segs:
                        s["start"] = s.get("start", 0.0) + time_offset
                        s["end"] = s.get("end", 0.0) + time_offset
                    if chunk_segs:
                        time_offset = chunk_segs[-1]["end"]
                    raw_segments.extend(chunk_segs)
        else:
            raw_segments = self._transcribe_file(file_path)

        segments: list[Segment] = [
            Segment(
                id=i,
                start=s.get("start", 0.0),
                end=s.get("end", 0.0),
                text=s.get("text", "").strip(),
                speaker=None,
            )
            for i, s in enumerate(raw_segments)
        ]

        duration = segments[-1].end if segments else 0.0

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
