from __future__ import annotations

import json
from pathlib import Path

from ..transcribers.base import TranscriptResult


def write_json(result: TranscriptResult, output_path: Path) -> None:
    """Write a structured JSON transcript following the schema in SPEC.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source_file": str(result.source_file),
        "duration_seconds": result.duration_seconds,
        "language": result.language,
        "backend": result.backend,
        "model": result.model,
        "diarization": result.diarization,
        "speakers_detected": result.speakers_detected,
        "transcribed_at": result.transcribed_at.isoformat(),
        "text": result.text,
        "segments": [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text.strip(),
                "words": [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "score": w.score,
                    }
                    for w in seg.words
                ],
            }
            for seg in result.segments
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
