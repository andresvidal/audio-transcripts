from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import BaseTranscriber, Segment, TranscriptResult

DEFAULT_MODEL = "openai/whisper-large-v3-turbo"


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class HuggingFaceTranscriber(BaseTranscriber):
    """Transcription via HuggingFace Transformers ASR pipeline.

    Supports any Whisper-compatible model on the Hub.
    Returns chunk-level timestamps but no word-level timestamps or diarization.

    Requirements:
        pip install transformers torch torchaudio
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, language)
        self.device = device or _detect_device()
        self._pipe = None

    @property
    def backend_name(self) -> str:
        return "huggingface"

    def _load_pipeline(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            device=self.device,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            return_timestamps=True,
        )

    def transcribe(self, file_path: Path, status_callback=None) -> TranscriptResult:
        import torchaudio

        self._load_pipeline()

        # Determine duration without loading the full waveform into memory
        info = torchaudio.info(str(file_path))
        duration = info.num_frames / info.sample_rate

        generate_kwargs: dict = {}
        if self.language:
            generate_kwargs["language"] = self.language

        result = self._pipe(str(file_path), generate_kwargs=generate_kwargs)

        raw_chunks = result.get("chunks", [])
        segments: list[Segment] = []

        for i, chunk in enumerate(raw_chunks):
            ts = chunk.get("timestamp") or (0.0, 0.0)
            start = ts[0] if ts[0] is not None else 0.0
            end = ts[1] if ts[1] is not None else start
            segments.append(
                Segment(
                    id=i,
                    start=start,
                    end=end,
                    text=chunk.get("text", ""),
                    speaker=None,
                )
            )

        # Fallback: wrap the entire transcript as a single segment
        if not segments and result.get("text"):
            segments = [
                Segment(id=0, start=0.0, end=duration, text=result["text"], speaker=None)
            ]

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
