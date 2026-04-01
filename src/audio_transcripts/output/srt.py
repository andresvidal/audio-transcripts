from __future__ import annotations

from pathlib import Path

from ..transcribers.base import TranscriptResult


def _ts(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(result: TranscriptResult, output_path: Path) -> None:
    """Write a SubRip (.srt) subtitle file with optional speaker prefixes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.segments, start=1):
            text = seg.text.strip()
            if seg.speaker and result.diarization:
                text = f"{seg.speaker}: {text}"
            f.write(f"{i}\n")
            f.write(f"{_ts(seg.start)} --> {_ts(seg.end)}\n")
            f.write(f"{text}\n\n")
