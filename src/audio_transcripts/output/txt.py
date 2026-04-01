from __future__ import annotations

from pathlib import Path

from ..transcribers.base import TranscriptResult


def write_txt(result: TranscriptResult, output_path: Path) -> None:
    """Write a plain-text transcript, prefixing each line with the speaker label."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in result.segments:
            line = seg.text.strip()
            if seg.speaker and result.diarization:
                line = f"{seg.speaker}: {line}"
            f.write(line + "\n")
