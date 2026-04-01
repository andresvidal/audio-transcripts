from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


@dataclass
class Word:
    word: str
    start: float
    end: float
    score: float = 1.0


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptResult:
    source_file: Path
    duration_seconds: float
    language: str
    backend: str
    model: str
    diarization: bool
    transcribed_at: datetime
    segments: list[Segment]
    speakers_detected: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        lines = []
        for seg in self.segments:
            line = seg.text.strip()
            if seg.speaker and self.diarization:
                line = f"{seg.speaker}: {line}"
            lines.append(line)
        return "\n".join(lines)

    def apply_speaker_names(self, name_map: dict[str, str]) -> None:
        """Replace SPEAKER_XX IDs with human-readable names in all segments."""
        if not name_map:
            return
        for seg in self.segments:
            if seg.speaker and seg.speaker in name_map:
                seg.speaker = name_map[seg.speaker]
        self.speakers_detected = [name_map.get(s, s) for s in self.speakers_detected]


class BaseTranscriber(ABC):
    """Abstract base class for all transcription backends."""

    def __init__(self, model: str, language: Optional[str] = None, **kwargs):
        self.model = model
        self.language = language

    @abstractmethod
    def transcribe(
        self,
        file_path: Path,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptResult:
        """Transcribe the given audio file and return a TranscriptResult."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for this backend, e.g. 'whisperx'."""
