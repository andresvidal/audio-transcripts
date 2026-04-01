# Audio Transcripts

Local-first audio transcription with **speaker diarization** — every line of output is
tied to the person who said it. Powered by WhisperX (default), Faster-Whisper, HuggingFace
Transformers, OpenAI Whisper API, and Google Gemini.

**Features:**
- Speaker-labeled output (`SPEAKER_00:`, `SPEAKER_01:`, … or custom names)
- Word-level timestamps and alignment
- Transcription cache — Whisper and diarization results are cached by content hash; re-runs are instant
- Multiple output formats: `txt`, `json`, `srt`, `vtt`
- Manifest tracking — already-done files are skipped automatically
- Preflight check for HuggingFace token and gated model access

## Requirements

- Python 3.12 (`whisperx` requires `<3.14`)
- `ffmpeg` installed system-wide:
  ```bash
  brew install ffmpeg          # macOS
  sudo apt install ffmpeg      # Ubuntu/Debian
  ```

## Docker (recommended — no Python setup required)

Docker handles Python 3.12, ffmpeg, and all dependencies automatically.

```bash
# 1. Copy env file and fill in your tokens
cp .env.example .env

# 2. Build the image (first build: ~10–15 min, downloads torch + whisperx)
docker compose build

# 3. Run — transcribes everything in ./audio-files, writes output to ./transcripts
docker compose run --rm transcribe

# Custom flags
docker compose run --rm transcribe \
  /audio --backend faster-whisper --language en -f txt -f srt

# Pass speaker count hints
docker compose run --rm transcribe \
  /audio --min-speakers 2 --max-speakers 4 -f txt -f json
```

> **Model caching**: the `hf-model-cache` Docker volume persists downloaded models
> (~1–3 GB) across runs so they are only downloaded once.

## Local Install

Requires Python 3.12 and `ffmpeg` installed on the host.

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 2. Install local backends (faster-whisper, transformers, torch)
pip install -e ".[local]"

# 3. Install WhisperX from git (the PyPI release pins ctranslate2==4.4.0
#    which has no Python 3.12+ wheel; the git version has no such restriction)
pip install git+https://github.com/m-bain/whisperX.git

# Cloud backends (OpenAI + Gemini)
pip install -e ".[cloud]"
```

## Setup

Copy `.env.example` to `.env` and fill in your tokens:

```bash
cp .env.example .env
```

For speaker diarization (WhisperX default), accept the terms for these three gated models
on HuggingFace (free, just requires an account):

1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/pyannote/speaker-diarization-community-1

Then create a token at https://huggingface.co/settings/tokens and set `HF_TOKEN=hf_...`
in your `.env`.

## Usage

```bash
# Activate the venv first (if not already active)
source .venv/bin/activate

# Transcribe a folder — WhisperX default, speaker-labeled output
# (defaults: --model small, -f txt -f srt)
python transcribe.py ./audio-files

# Available model sizes (whisperx / faster-whisper), fastest → most accurate:
#   tiny | base | small | medium | large-v2 | large-v3 | large-v3-turbo
python transcribe.py ./audio-files --model medium   # balanced
python transcribe.py ./audio-files --model large-v3  # best accuracy

# Specify output formats (default: txt, json)
python transcribe.py ./audio-files -f txt -f srt -f json

# Hint the number of speakers for better diarization
python transcribe.py ./audio-files --min-speakers 2 --max-speakers 4

# Custom output directory
python transcribe.py ./audio-files --output-dir ./my-transcripts

# Remap SPEAKER_XX IDs to real names
python transcribe.py ./audio-files \
  --speaker-names '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}'

# Re-label an already-transcribed folder instantly (uses full cache, no ML)
python transcribe.py ./audio-files --overwrite \
  --speaker-names '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}'

# Target a single file instead of a whole folder
python transcribe.py ./audio-files/interview.m4a -f txt -f srt

# Re-label a single file with speaker names
python transcribe.py ./audio-files/interview.m4a --overwrite \
  --speaker-names '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}'

# Single-speaker, no diarization (faster)
python transcribe.py ./audio-files --backend faster-whisper

# Scan sub-folders recursively
python transcribe.py ./audio-files --recursive

# List discovered files without transcribing
python transcribe.py ./audio-files --dry-run
```

## Google Colab quick start

> **Before you start:** go to Runtime → Change runtime type and select a **T4 GPU** (or any available GPU).

1. Clone + install

   Run these three cells **in order**. Splitting them avoids the session restart that
   occurs when torch is reinstalled mid-cell.

   **Cell 1 — system deps** (ffmpeg is usually pre-installed in Colab, but this is a no-op if so):
   ```bash
   !apt-get install -y ffmpeg
   ```

   **Cell 2 — core package** (click, rich, dotenv only — no torch):
   ```bash
   %cd /content
   !git clone https://github.com/andresvidal/audio-transcripts.git
   %cd /content/audio-transcripts
   !pip install -e "."
   ```

   **Cell 3 — WhisperX** (installs torch/torchaudio compatible versions automatically):
   ```bash
   !pip install git+https://github.com/m-bain/whisperX.git
   ```

   > **Why split?** `pip install -e ".[local]"` re-installs torch, which conflicts with
   > Colab's pre-installed torch and triggers a session restart. Installing the core
   > package first (no torch) then letting WhisperX manage its own torch version avoids this.

2. Set secrets from Colab userdata (or other secret manager)
```bash
from google.colab import userdata
import os

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
```

3. Put the audio file(s) in /content/audio-files (upload via Colab UI or copy from Drive)
```bash
!mkdir -p /content/audio-files /content/transcripts
# e.g. (if you copied to drive):
# !cp /content/drive/MyDrive/audio/* /content/audio-files
```

4. Transcribe with whisperx small and txt/srt output and save to /content/transcripts.
```bash
!python transcribe.py /content/audio-files --output-dir /content/transcripts --backend whisperx --model small --language en -f txt -f srt
```

5. Re-label from cache with friendly speaker names
```bash
!python transcribe.py /content/audio-files --output-dir /content/transcripts --overwrite \
  --speaker-names '{"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}' -f txt -f srt
```

## Caching

WhisperX results are cached locally in `.cache/whisperx/` by a SHA-256 content hash of
the audio file. There are two cache layers:

| Layer | Stores | Cache key |
|---|---|---|
| Whisper cache (`*.json`) | Raw transcription segments | file bytes + model + language |
| Full cache (`*.full.json`) | Aligned + diarized segments | file bytes + model + language + diarize + speaker hints |

On the **first run**, all three steps run (Whisper → align → diarize) and both caches are
saved. On **subsequent runs**, the full cache is hit and all ML steps are skipped entirely
— a typical re-run takes under 2 seconds regardless of audio length.

**Changing speaker names does not invalidate the cache.** Use `--overwrite` to regenerate
output files from the cache with different names:

```bash
# First time — runs ML, takes a few minutes
python transcribe.py ./audio-files -f txt -f srt

# Instantly regenerate with real names (no ML, ~1–2 s)
python transcribe.py ./audio-files --overwrite \
  --speaker-names '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}'
```

The cache is stored in `.cache/` at the project root (gitignored). Delete it to force a
full re-run:

```bash
rm -rf .cache/whisperx/
```

## Backends

| Backend | Diarization | Cost | Best for |
|---|---|---|---|
| `whisperx` ⭐ | ✅ Yes | Free (local) | Any multi-speaker audio |
| `faster-whisper` | ❌ No | Free (local) | Single-speaker, fast |
| `huggingface` | ❌ No | Free (local) | Experimenting with HF models |
| `openai` | ❌ No | $0.006/min | No GPU, small files |
| `gemini` | ❌ No | ~$0.06/hr | Transcribe + summarize |

## Models

Applies to `whisperx` and `faster-whisper` backends. Pass with `--model <size>`.

| Model | Params | Speed (M1 CPU) | Accuracy | Notes |
|---|---|---|---|---|
| `tiny` | 39 M | ~3–5 min / 30 min audio | Low | Quick tests only |
| `base` | 74 M | ~5–8 min / 30 min audio | Low–Medium | |
| `small` ⭐ | 244 M | ~10–20 min / 30 min audio | Medium | **Default** — good balance |
| `medium` | 769 M | ~30–50 min / 30 min audio | High | Recommended upgrade |
| `large-v2` | 1.5 B | ~60–100 min / 30 min audio | Very high | |
| `large-v3` | 1.5 B | ~60–120 min / 30 min audio | Best | Latest OpenAI model |
| `large-v3-turbo` | 809 M | ~20–35 min / 30 min audio | Very high | Distilled large-v3, fast + accurate |

> Speeds are approximate for Apple M1 CPU. GPU (CUDA) is typically 5–10× faster.
> `large-v3-turbo` is the best accuracy-to-speed tradeoff if `small` isn't accurate enough.

## Output formats

| Format | Description |
|---|---|
| `txt` | Plain text with `SPEAKER_00: …` prefixes (or remapped names) |
| `json` | Full structured output with segments, timestamps, word scores, speaker IDs |
| `srt` | SubRip subtitles (video players) |
| `vtt` | WebVTT subtitles (web / QuickTime) |

### Example `txt` output

```
SPEAKER_00: Hey, welcome to the show.
SPEAKER_01: Thanks for having me.
SPEAKER_00: So let's start with the basics.
```

With `--speaker-names '{"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}'`:

```
Host: Hey, welcome to the show.
Guest: Thanks for having me.
Host: So let's start with the basics.
```

See [SPEC.md](SPEC.md) for full specification and design decisions.

## Troubleshooting

### Speaker diarization fails with 403 / access denied

You need to accept the terms for all three gated pyannote models (see [Setup](#setup)).
Each model page has an "Agree and access repository" button. The token must have **read**
permission.

### `torchcodec is not installed correctly` warning on startup

`torchcodec` is pulled in transitively by `torchaudio` but is not needed by whisperx or
pyannote. Uninstall it to silence the warning permanently:

```bash
pip uninstall torchcodec -y
```

### `Lightning automatically upgraded your loaded checkpoint` notice

The Lightning upgrade tool is broken with PyTorch 2.6+ (a `weights_only` incompatibility).
This notice is suppressed automatically — no action needed.

### Transcription is very slow (M1/M2 Mac)

WhisperX does not support MPS (Apple Silicon GPU) and falls back to CPU. Expected speeds:

| Model | 30 min audio on M1 CPU |
|---|---|
| `large-v3` | 60–120 min |
| `medium` | 30–50 min |
| `small` ⭐ default | 10–20 min |
| `tiny` / `base` | 3–8 min |

Use `--model small` or `--model medium` for faster results:

```bash
python transcribe.py ./audio-files --model medium --language en
```

