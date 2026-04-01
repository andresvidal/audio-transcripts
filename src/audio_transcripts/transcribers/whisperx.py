from __future__ import annotations

import hashlib
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .base import BaseTranscriber, Segment, TranscriptResult, Word


def _detect_device() -> str:
    """Return the best available device, falling back to cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # whisperx does not fully support MPS; use cpu on Apple Silicon
        if torch.backends.mps.is_available():
            return "cpu"
    except ImportError:
        pass
    return "cpu"


class WhisperXTranscriber(BaseTranscriber):
    """Transcription + forced alignment + speaker diarization via WhisperX.

    This is the default backend. It runs fully locally and labels every
    transcript segment with the speaker who said it (SPEAKER_00, SPEAKER_01, …).

    Requirements:
        pip install whisperx torch torchaudio

    Speaker diarization additionally requires an HF_TOKEN environment variable
    with access to pyannote/speaker-diarization-3.1 (free to request at
    https://huggingface.co/pyannote/speaker-diarization-3.1).
    """

    def __init__(
        self,
        model: str = "large-v3",
        language: Optional[str] = None,
        device: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        diarize: bool = True,
        hf_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, language)
        self.device = device or _detect_device()
        self.compute_type = "int8" if self.device == "cpu" else "float16"
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.diarize = diarize
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._model = None
        if self.diarize:
            self._check_diarization_access()

    def _check_diarization_access(self) -> None:
        """Warn early if HF_TOKEN is missing or lacks access to gated pyannote models."""
        GATED_MODELS = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/segmentation-3.0",
            "pyannote/speaker-diarization-community-1",
        ]
        if not self.hf_token:
            warnings.warn(
                "\n[WhisperX] HF_TOKEN is not set — speaker diarization will be skipped.\n"
                "  1. Get a free token at https://huggingface.co/settings/tokens\n"
                "  2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  3. Accept model terms at https://huggingface.co/pyannote/segmentation-3.0\n"
                "  4. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                "  5. Add HF_TOKEN=hf_... to your .env file\n"
                "  Output will be transcribed but without SPEAKER_XX labels.",
                stacklevel=2,
            )
            return
        try:
            import urllib.request
            denied = []
            for model_id in GATED_MODELS:
                req = urllib.request.Request(
                    f"https://huggingface.co/api/models/{model_id}",
                    headers={"Authorization": f"Bearer {self.hf_token}"},
                )
                try:
                    urllib.request.urlopen(req, timeout=5)
                except Exception as e:
                    code = getattr(getattr(e, "code", None), "__str__", lambda: str(e))()
                    denied.append(f"  - {model_id} ({code})")
            if denied:
                warnings.warn(
                    "\n[WhisperX] HF_TOKEN is set but access is denied for:\n"
                    + "\n".join(denied) + "\n"
                    "  Accept the model license at each URL above, then retry.\n"
                    "  Speaker diarization will be skipped for this run.",
                    stacklevel=2,
                )
        except Exception:
            pass  # network unavailable — skip preflight silently

    @property
    def backend_name(self) -> str:
        return "whisperx"

    def _cache_key(self, file_path: Path) -> str:
        """Stable key based on file content hash + model + language (Whisper step only)."""
        h = hashlib.sha256()
        h.update(file_path.read_bytes())
        h.update(self.model.encode())
        h.update((self.language or "").encode())
        return h.hexdigest()[:16]

    def _full_cache_key(self, file_path: Path) -> str:
        """Stable key for the complete diarized result (includes diarization params)."""
        h = hashlib.sha256()
        h.update(file_path.read_bytes())
        h.update(self.model.encode())
        h.update((self.language or "").encode())
        h.update(str(self.diarize).encode())
        h.update(str(self.min_speakers).encode())
        h.update(str(self.max_speakers).encode())
        return h.hexdigest()[:16]

    def _cache_dir(self) -> Path:
        d = Path.cwd() / ".cache" / "whisperx"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cache_path(self, file_path: Path) -> Path:
        return self._cache_dir() / f"{file_path.stem}_{self._cache_key(file_path)}.json"

    def _full_cache_path(self, file_path: Path) -> Path:
        return self._cache_dir() / f"{file_path.stem}_{self._full_cache_key(file_path)}.full.json"

    def _load_cache(self, file_path: Path) -> Optional[dict]:
        p = self._cache_path(file_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, file_path: Path, result: dict) -> None:
        with open(self._cache_path(file_path), "w", encoding="utf-8") as f:
            json.dump(result, f)

    def _load_full_cache(self, file_path: Path) -> Optional[dict]:
        p = self._full_cache_path(file_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_full_cache(self, file_path: Path, payload: dict) -> None:
        with open(self._full_cache_path(file_path), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import whisperx

        self._model = whisperx.load_model(
            self.model,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
        )

    def transcribe(
        self,
        file_path: Path,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptResult:
        import whisperx

        def _status(msg: str) -> None:
            if status_callback:
                status_callback(msg)

        # ── Full cache hit: skip all ML (Whisper + align + diarize) ──────────
        full_cached = self._load_full_cache(file_path)
        if full_cached:
            _status("full cache hit — skipping all ML steps…")
            return self._result_from_full_cache(file_path, full_cached)

        _status("loading model…")
        self._load_model()

        _status("loading audio…")
        audio = whisperx.load_audio(str(file_path))
        duration = len(audio) / 16_000  # whisperx resamples to 16 kHz

        # Step 1 — transcribe (with Whisper-only cache)
        cached = self._load_cache(file_path)
        if cached:
            _status("transcription cache hit — skipping Whisper…")
            result = cached
            detected_language: str = result.get("language") or self.language or "unknown"
        else:
            _status("transcribing (Whisper)…")
            result = self._model.transcribe(audio, batch_size=16, language=self.language)
            detected_language = result.get("language") or self.language or "unknown"
            self._save_cache(file_path, result)
            _status("transcription cached ✓")

        # Step 2 — forced alignment (word-level timestamps)
        _status(f"aligning words ({detected_language})…")
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language, device=self.device
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
        except Exception as exc:
            warnings.warn(
                f"Word alignment failed for '{detected_language}': {exc}. "
                "Proceeding without word-level timestamps."
            )

        # Step 3 — speaker diarization
        did_diarize = False
        if self.diarize:
            if not self.hf_token:
                warnings.warn(
                    "HF_TOKEN is not set — skipping speaker diarization. "
                    "Set HF_TOKEN to enable speaker labels."
                )
            else:
                try:
                    _status("diarizing speakers (pyannote)…")
                    from whisperx.diarize import DiarizationPipeline
                    diarize_pipeline = DiarizationPipeline(
                        model_name="pyannote/speaker-diarization-3.1",
                        token=self.hf_token, device=self.device
                    )
                    diarize_kwargs: dict = {}
                    if self.min_speakers is not None:
                        diarize_kwargs["min_speakers"] = self.min_speakers
                    if self.max_speakers is not None:
                        diarize_kwargs["max_speakers"] = self.max_speakers

                    diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
                    _status("assigning speakers to segments…")
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    did_diarize = True
                except Exception as exc:
                    _status(f"diarization failed: {exc}")
                    warnings.warn(
                        f"Speaker diarization failed: {exc}. "
                        "Proceeding without speaker labels."
                    )

        # Build typed result
        segments: list[Segment] = []
        for i, seg in enumerate(result.get("segments", [])):
            words = [
                Word(
                    word=w.get("word", ""),
                    start=w.get("start", seg["start"]),
                    end=w.get("end", seg["end"]),
                    score=w.get("score", 1.0),
                )
                for w in seg.get("words", [])
            ]
            segments.append(
                Segment(
                    id=i,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    speaker=seg.get("speaker"),
                    words=words,
                )
            )

        speakers_detected = sorted(
            {seg.speaker for seg in segments if seg.speaker is not None}
        )

        transcript = TranscriptResult(
            source_file=file_path,
            duration_seconds=duration,
            language=detected_language,
            backend=self.backend_name,
            model=self.model,
            diarization=did_diarize,
            transcribed_at=datetime.now(timezone.utc),
            segments=segments,
            speakers_detected=speakers_detected,
        )

        # Save full cache so future runs with --speaker-names / new formats are instant
        self._save_full_cache(file_path, self._result_to_full_cache(transcript))
        return transcript

    def _result_to_full_cache(self, result: TranscriptResult) -> dict:
        """Serialize a TranscriptResult to the full-cache dict format."""
        return {
            "duration_seconds": result.duration_seconds,
            "language": result.language,
            "diarization": result.diarization,
            "speakers_detected": result.speakers_detected,
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end, "score": w.score}
                        for w in seg.words
                    ],
                }
                for seg in result.segments
            ],
        }

    def _result_from_full_cache(self, file_path: Path, payload: dict) -> TranscriptResult:
        """Reconstruct a TranscriptResult from the full-cache dict."""
        segments = [
            Segment(
                id=seg["id"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg.get("speaker"),
                words=[
                    Word(word=w["word"], start=w["start"], end=w["end"], score=w["score"])
                    for w in seg.get("words", [])
                ],
            )
            for seg in payload.get("segments", [])
        ]
        return TranscriptResult(
            source_file=file_path,
            duration_seconds=payload["duration_seconds"],
            language=payload["language"],
            backend=self.backend_name,
            model=self.model,
            diarization=payload["diarization"],
            transcribed_at=datetime.now(timezone.utc),
            segments=segments,
            speakers_detected=payload["speakers_detected"],
        )
