# Audio Transcripts — Project Specification

## Overview

A modular CLI tool that scans a folder for audio files and generates transcripts using
state-of-the-art speech-to-text models. The design prioritizes **local, speaker-aware
transcription** — every transcript is tied to the speaker who said it. WhisperX is the
default backend, providing word-level timestamps and speaker diarization out of the box.
Cloud API backends are available as fallbacks when local compute is unavailable.

---

## Goals

- List all audio files in a target folder (recursive optional)
- Transcribe each file and **label every segment by speaker** (SPEAKER_00, SPEAKER_01, …)
- Output transcripts in one or more formats (txt, json, srt, vtt) with speaker labels
- Skip already-transcribed files (idempotent / resumable)
- Be reusable via CLI and importable as a Python library
- Run entirely locally by default — no data leaves the machine

---

## Supported Audio Formats

| Extension | Description                   |
|-----------|-------------------------------|
| `.mp3`    | MPEG Audio Layer III          |
| `.wav`    | Waveform Audio                |
| `.m4a`    | MPEG-4 Audio                  |
| `.mp4`    | MPEG-4 (audio extracted)      |
| `.ogg`    | Ogg Vorbis                    |
| `.flac`   | Free Lossless Audio Codec     |
| `.aac`    | Advanced Audio Codec          |
| `.webm`   | WebM Audio                    |
| `.opus`   | Opus Audio                    |

---

## Transcription Backends

The tool supports multiple backends via a common interface (`BaseTranscriber`). Backends
are ordered by priority — **WhisperX is the default** because it produces speaker-labeled
transcripts. Cloud backends do not support diarization and produce speaker-less output.

### 1. WhisperX ⭐ default — local, speaker-aware

- **Library**: [`whisperx`](https://github.com/m-bain/whisperX) — installed from git (PyPI release pins `ctranslate2==4.4.0` which has no Python 3.12+ wheel)
- **Model**: `openai/whisper-large-v3` + `pyannote/speaker-diarization-3.1`
- **Cost**: **Free** — runs locally; diarization pipeline requires a free HuggingFace token and acceptance of three gated model licenses:
  - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
  - [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
- **Pros**: Word-level timestamps, speaker diarization (who said what), forced alignment, fully local, content-hash cache
- **Diarization**: Yes — each segment tagged with `SPEAKER_00`, `SPEAKER_01`, etc.
- **Use when**: Any multi-speaker audio; meetings, interviews, call recordings

### 2. Faster-Whisper (local, fast — no diarization)

- **Library**: [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
- **Model**: `large-v3` or `large-v3-turbo` via CTranslate2
- **Cost**: **Free** — runs locally; one-time model download (~3 GB for `large-v3`); compute cost depends on your hardware
- **Pros**: 4× faster than original Whisper, low VRAM with int8 quantization, CPU-friendly
- **Diarization**: No — single-speaker output only
- **Use when**: Single-speaker recordings (lectures, voice memos) where speed matters more than speaker labels

### 3. Hugging Face Transformers (local, flexible — no diarization)

- **Library**: `transformers` + `torch`
- **Models**:
  - `openai/whisper-large-v3-turbo` (fast, accurate)
  - `distil-whisper/distil-large-v3` (2× faster, minimal accuracy loss)
- **Cost**: **Free** — runs locally; HF Inference API (cloud fallback) offers a free tier (~1,000 requests/month) and paid at ~$0.06/CPU·hr or ~$0.60/GPU·hr via serverless endpoints
- **Pros**: Easiest to experiment with different HF models
- **Diarization**: No — use WhisperX for speaker labels
- **Use when**: Trying new models from the Hub without extra tooling

### 4. OpenAI API (cloud)

- **Library**: `openai` SDK
- **Model**: `whisper-1` (or `gpt-4o-audio-preview` for chat-based transcription)
- **Cost**:
  - `whisper-1`: **$0.006 / minute** of audio (~$0.36/hr)
  - `gpt-4o-audio-preview`: **$0.10 / 1K audio input tokens** (≈ $0.06/minute; includes conversational context)
  - File size limit: 25 MB per request (tool handles chunking automatically)
- **Pros**: No local compute needed, very accurate
- **Diarization**: No — cloud API returns a single transcript block
- **Use when**: Occasional use, no GPU, files are small-to-medium, single-speaker content

### 5. Google Gemini API (cloud, multimodal)

- **Library**: `google-generativeai`
- **Model**: `gemini-2.0-flash` or `gemini-2.5-pro`
- **Cost**:
  - `gemini-2.0-flash`: **$0.001 / 1K audio seconds** (~$0.06/hr) — most economical
  - `gemini-2.5-pro`: **$0.0035 / 1K audio seconds** (~$0.21/hr) — highest accuracy, up to 2 hr audio per request
  - Both have a **free tier**: 1,500 requests/day (`flash`) or 50 requests/day (`pro`) at no cost
- **Pros**: Can transcribe + summarize + answer questions in one call
- **Diarization**: No — but prompt engineering can produce rough speaker separation for simple 2-speaker audio
- **Use when**: Want enriched output (summary, topics, action items) and diarization is not required

---

## Architecture

```
audio-transcripts/
├── SPEC.md
├── README.md
├── pyproject.toml           # packaging + dependencies
├── .env.example             # API key template
├── Dockerfile               # Python 3.12 + ffmpeg + all deps
├── docker-compose.yml       # volumes, env, default command
├── transcribe.py            # CLI entry point
├── .cache/
│   └── whisperx/            # content-hash cache (gitignored)
│       ├── <stem>_<hash>.json       # Whisper-only segments
│       └── <stem>_<hash>.full.json  # aligned + diarized segments
└── src/
    └── audio_transcripts/
        ├── __init__.py
        ├── scanner.py        # file discovery
        ├── transcribers/
        │   ├── __init__.py
        │   ├── base.py       # BaseTranscriber ABC
        │   ├── faster_whisper.py
        │   ├── whisperx.py
        │   ├── huggingface.py
        │   ├── openai_api.py
        │   └── google_gemini.py
        ├── output/
        │   ├── __init__.py
        │   ├── txt.py
        │   ├── json_writer.py
        │   ├── srt.py
        │   └── vtt.py
        └── pipeline.py       # orchestrates scan → transcribe → write
```

---

## CLI Interface

```bash
# Basic usage — transcribe all audio in a folder (WhisperX default, speaker-labeled)
python transcribe.py ./audio-files

# Explicitly choose backend
python transcribe.py /path/to/audio --backend whisperx          # default: speaker-labeled
python transcribe.py /path/to/audio --backend faster-whisper    # faster, no speakers
python transcribe.py /path/to/audio --backend huggingface --model distil-whisper/distil-large-v3
python transcribe.py /path/to/audio --backend openai
python transcribe.py /path/to/audio --backend gemini

# Output formats (can combine)
python transcribe.py /path/to/audio --output-dir ./transcripts --formats txt srt json

# Recursive folder scan
python transcribe.py /path/to/audio --recursive

# Skip already-done files
python transcribe.py /path/to/audio --skip-existing   # (default: on)

# Language hint (speeds up models)
python transcribe.py /path/to/audio --language en

# Dry run — list files only, no transcription
python transcribe.py /path/to/audio --dry-run

# Whisper model size (for local backends)
python transcribe.py /path/to/audio --model large-v3-turbo

# Speaker diarization options (WhisperX only)
python transcribe.py /path/to/audio --min-speakers 2 --max-speakers 5
python transcribe.py /path/to/audio --no-diarize          # skip diarization, faster

# Concurrency (for API backends)
python transcribe.py /path/to/audio --workers 4
```

---

## Output Formats

| Format | Description                                                        |
|--------|--------------------------------------------------------------------|
| `txt`  | Speaker-labeled plain text: `SPEAKER_00: Hello, welcome…`         |
| `json` | Structured: segments, timestamps, speaker labels, word confidence  |
| `srt`  | SubRip subtitles with speaker prefix per cue                       |
| `vtt`  | WebVTT subtitles with speaker voice tags (`<v SPEAKER_00>`)        |

Output files are named after the source audio file:
```
audio-file.mp3  →  audio-file.txt / audio-file.srt / audio-file.json
```

---

## Transcript JSON Schema

```json
{
  "source_file": "/path/to/audio.mp3",
  "duration_seconds": 3612.5,
  "language": "en",
  "backend": "whisperx",
  "model": "large-v3",
  "diarization": true,
  "speakers_detected": ["SPEAKER_00", "SPEAKER_01"],
  "transcribed_at": "2026-03-31T14:22:00Z",
  "text": "SPEAKER_00: Hello, welcome to the meeting.\nSPEAKER_01: Thanks for having me.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "speaker": "SPEAKER_00",
      "text": "Hello, welcome to the meeting.",
      "words": [
        { "word": "Hello", "start": 0.0, "end": 0.5, "score": 0.98 },
        { "word": "welcome", "start": 0.6, "end": 1.1, "score": 0.97 }
      ]
    },
    {
      "id": 1,
      "start": 4.5,
      "end": 6.8,
      "speaker": "SPEAKER_01",
      "text": "Thanks for having me.",
      "words": [
        { "word": "Thanks", "start": 4.5, "end": 4.9, "score": 0.99 }
      ]
    }
  ]
}
```

---

## Dependencies

### Core (always required)

| Package         | Purpose                          |
|-----------------|----------------------------------|
| `click`         | CLI argument parsing             |
| `rich`          | Progress bars, colored output    |
| `ffmpeg-python` | Audio format conversion via ffmpeg |
| `python-dotenv` | .env file loading                |

### Local backends

| Package                 | Backend              |
|-------------------------|----------------------|
| `faster-whisper`        | Faster-Whisper       |
| `whisperx`              | WhisperX             |
| `transformers`          | HuggingFace          |
| `torch`                 | HuggingFace / WhisperX |
| `torchaudio`            | Audio loading        |

### Cloud backends

| Package              | Backend      |
|----------------------|--------------|
| `openai`             | OpenAI API   |
| `google-generativeai`| Gemini API   |

### System dependency

```
ffmpeg  # must be installed via brew/apt
```

---

## Environment Variables (.env)

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
HF_TOKEN=hf_...           # required for WhisperX diarization (three gated pyannote models)
```

---

## Key Design Decisions

1. **Speaker-first output**: The default pipeline (WhisperX) always attempts diarization. Every `txt`, `srt`, `vtt`, and `json` output prefixes each segment with `SPEAKER_XX:`. Pass `--no-diarize` to skip when only one speaker is present.

2. **Backend abstraction**: All backends implement `BaseTranscriber.transcribe(file_path) -> TranscriptResult`. Swapping backends requires only a `--backend` flag change. Backends that do not support diarization return `speaker=None` on each segment.

3. **Idempotency**: Before transcribing, the pipeline checks if an output file already exists for that source. Configurable via `--skip-existing` / `--overwrite`.

4. **Chunked processing for large files**: Audio longer than a configurable threshold (default 30 min) is split and merged to avoid memory issues and API size limits. Speaker labels are preserved across chunk boundaries.

5. **ffmpeg preprocessing**: All audio is normalized to 16kHz mono WAV before feeding into local models (Whisper requirement), handled transparently.

6. **Progress persistence**: A lightweight `manifest.json` in the output dir tracks completion state, so interrupted runs resume from where they left off.

7. **Speaker name mapping**: An optional `--speaker-names` flag accepts a JSON map (`{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}`) to replace raw IDs with real names in all output files.

8. **No GPU required**: WhisperX and Faster-Whisper both run on CPU with `int8` quantization. GPU (CUDA / MPS on Apple Silicon) is detected and used automatically when available, significantly reducing processing time.

9. **Two-layer content-hash cache** (WhisperX only): Results are cached in `.cache/whisperx/` keyed by SHA-256 of the audio file bytes + model + language (+ diarization params for the full cache). Two layers:
   - *Whisper cache* (`*.json`): raw segments from the Whisper pass. Reused on re-runs to skip the slowest step.
   - *Full cache* (`*.full.json`): aligned and diarized segments. On a full cache hit all ML is skipped entirely — re-runs with `--overwrite --speaker-names` complete in ~1–2 seconds regardless of audio length.
   Cache is invalidated automatically when the audio file content, model, language, or diarization parameters change.

10. **HuggingFace token preflight**: On startup, `WhisperXTranscriber` verifies that `HF_TOKEN` is set and that the token has access to all required gated models. A clear, actionable warning (with fix steps) is printed before any audio is processed if access is missing.

---

## Implementation Phases

### Phase 1 — Core with Speaker Diarization (MVP)
- [x] `scanner.py`: list audio files, dry-run output
- [x] `BaseTranscriber` ABC with `speaker` field on every segment
- [x] **WhisperX backend** (default) — transcription + diarization in one pass
- [x] `txt` output writer with `SPEAKER_XX:` prefixes
- [x] CLI with `click` + `rich` progress bar
- [x] `--skip-existing`, `--recursive`, `--language`, `--dry-run`
- [x] `--min-speakers`, `--max-speakers`, `--no-diarize`

### Phase 2 — Additional Local Backend
- [x] Faster-Whisper backend (single-speaker fallback, faster)
- [x] HuggingFace Transformers backend
- [x] `--speaker-names` JSON map flag

### Phase 3 — Rich Output + Cloud Backends
- [x] `json`, `srt`, `vtt` output writers (all with speaker labels)
- [x] OpenAI API backend (with file chunking for >25 MB)
- [x] Google Gemini backend
- [x] Manifest-based resume

### Phase 4 — Polish
- [x] `pyproject.toml` packaging (`pip install .`)
- [x] README with examples
- [x] Docker image (`Dockerfile` + `docker-compose.yml`)
- [ ] Unit tests for scanner, diarization output, and writers

### Phase 5 — Performance & UX
- [x] Stage-aware progress display (`rich.Live` with per-file + per-stage bars)
- [x] Two-layer content-hash cache (Whisper-only + full diarized result)
- [x] Instant re-label: `--overwrite --speaker-names` hits full cache, no ML (~1–2 s)
- [x] HuggingFace token preflight check with actionable error messages
- [x] Default output directory fixed to project root (`./transcripts/`)

---

## Example Workflow

### Option A — Docker (recommended)

```bash
# 1. Copy env file and set HF_TOKEN, API keys
cp .env.example .env

# 2. Build image (first build ~10–15 min; cached on subsequent runs)
docker compose build

# 3. Transcribe ./audio-files → ./transcripts (WhisperX default)
docker compose run --rm transcribe

# Custom flags work the same as the local CLI
docker compose run --rm transcribe \
  /audio --backend faster-whisper --language en -f txt -f srt
```

### Option B — Local virtualenv

```bash
# 1. Create and activate a virtual environment
#    Use Python 3.12 — whisperx requires Python <3.14
python3.12 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 2. Install local backends (faster-whisper, transformers, torch)
pip install -e ".[local]"

# 3. Install WhisperX separately from git (PyPI release pins ctranslate2==4.4.0
#    which has no Python 3.13+ wheel; the git version has no such restriction)
pip install git+https://github.com/m-bain/whisperX.git

# Cloud backends (OpenAI + Gemini)
pip install -e ".[cloud]"

# Transcribe a folder with speaker diarization (default)
python transcribe.py ./audio-files \
  --backend whisperx \
  --model large-v3 \
  --language en \
  --formats txt srt json \
  --output-dir ./transcripts \
  --min-speakers 2 --max-speakers 6 \
  --recursive

# Remap speaker IDs to real names on first run
python transcribe.py ./audio-files \
  --speaker-names '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}' \
  --output-dir ./transcripts

# Instantly re-label with different names (full cache hit — no ML, ~1–2 s)
python transcribe.py ./audio-files --overwrite \
  --speaker-names '{"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}'

# Output
transcripts/
  meeting-2025-01-10.txt        # SPEAKER_00: Hello...\nSPEAKER_01: Thanks...
  meeting-2025-01-10.srt        # subtitles with speaker prefix per cue
  meeting-2025-01-10.json       # full segment + word-level data
  ...
  manifest.json
```
