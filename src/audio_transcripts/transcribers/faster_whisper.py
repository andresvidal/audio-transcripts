from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import BaseTranscriber, Segment, TranscriptResult, Word


def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type) for the best available hardware."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
        # faster-whisper does not support MPS natively; fall back to cpu
        if torch.backends.mps.is_available():
            return "cpu", "int8"
    except ImportError:
        pass
    return "cpu", "int8"


class FasterWhisperTranscriber(BaseTranscriber):
    """Fast local transcription via faster-whisper (CTranslate2 backend).

    Produces word-level timestamps but no speaker diarization.
    Use WhisperX when speaker labels are needed.

    Requirements:
        pip install faster-whisper
    """

    def __init__(
        self,
        model: str = "large-v3",
        language: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, language)
        detected_device, detected_compute = _detect_device()
        self.device = device or detected_device
        self.compute_type = compute_type or detected_compute
        self._model = None

    @property
    def backend_name(self) -> str:
        return "faster-whisper"

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model, device=self.device, compute_type=self.compute_type
        )

    def transcribe(self, file_path: Path, status_callback=None) -> TranscriptResult:
        self._load_model()

        segments_gen, info = self._model.transcribe(
            str(file_path),
            language=self.language,
            word_timestamps=True,
            beam_size=5,
        )

        segments: list[Segment] = []
        for i, seg in enumerate(segments_gen):
            words = [
                Word(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    score=w.probability,
                )
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(
                    id=i,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=None,
                    words=words,
                )
            )

        return TranscriptResult(
            source_file=file_path,
            duration_seconds=info.duration,
            language=info.language,
            backend=self.backend_name,
            model=self.model,
            diarization=False,
            transcribed_at=datetime.now(timezone.utc),
            segments=segments,
            speakers_detected=[],
        )
