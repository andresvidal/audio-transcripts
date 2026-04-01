from __future__ import annotations

from pathlib import Path

from ..transcribers.base import TranscriptResult


def _ts(seconds: float) -> str:
    """Format seconds as WebVTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_vtt(result: TranscriptResult, output_path: Path) -> None:
    """Write a WebVTT (.vtt) subtitle file.

    Uses <v SPEAKER_XX> voice tags when diarization data is present,
    which is compatible with browsers and QuickTime.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in result.segments:
            text = seg.text.strip()
            if seg.speaker and result.diarization:
                text = f"<v {seg.speaker}>{text}</v>"
            f.write(f"{_ts(seg.start)} --> {_ts(seg.end)}\n")
            f.write(f"{text}\n\n")
