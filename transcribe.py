#!/usr/bin/env python3
"""Audio Transcription CLI — entry point.

Usage:
    python transcribe.py FOLDER [OPTIONS]
"""
from __future__ import annotations

import logging
import warnings
# Suppress torchcodec warning from pyannote (message starts with \n, use module filter)
warnings.filterwarnings("ignore", module="pyannote.audio.core.io")
# Suppress Lightning checkpoint upgrade notice: upgrade tool is broken with PyTorch 2.6
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

import json
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

# Allow running as `python transcribe.py` from the project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()

console = Console()

_BACKENDS = ["whisperx", "faster-whisper", "huggingface", "openai", "gemini"]
_FORMATS = ["txt", "json", "srt", "vtt"]
_DEFAULT_BACKEND = "whisperx"
_DEFAULT_FORMATS = ("txt",)

_DEFAULT_MODEL: dict[str, str] = {
    "whisperx": "large-v3",
    "faster-whisper": "large-v3",
    "huggingface": "openai/whisper-large-v3-turbo",
    "openai": "whisper-1",
    "gemini": "gemini-2.0-flash",
}


def _build_transcriber(
    backend: str,
    model: str,
    language: str | None,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
):
    """Instantiate the requested transcription backend."""
    from audio_transcripts.transcribers.faster_whisper import FasterWhisperTranscriber
    from audio_transcripts.transcribers.google_gemini import GeminiTranscriber
    from audio_transcripts.transcribers.huggingface import HuggingFaceTranscriber
    from audio_transcripts.transcribers.openai_api import OpenAITranscriber
    from audio_transcripts.transcribers.whisperx import WhisperXTranscriber

    common = dict(model=model, language=language)

    if backend == "whisperx":
        return WhisperXTranscriber(
            **common,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(**common)
    if backend == "huggingface":
        return HuggingFaceTranscriber(**common)
    if backend == "openai":
        return OpenAITranscriber(**common)
    if backend == "gemini":
        return GeminiTranscriber(**common)

    raise ValueError(f"Unknown backend: {backend!r}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--backend", "-b",
    default=_DEFAULT_BACKEND, show_default=True,
    type=click.Choice(_BACKENDS),
    help="Transcription backend.",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Model name/size (overrides the per-backend default).",
)
@click.option(
    "--language", "-l",
    default=None,
    help="Language code hint, e.g. 'en', 'es'. Skips auto-detection.",
)
@click.option(
    "--formats", "-f",
    multiple=True,
    default=_DEFAULT_FORMATS,
    type=click.Choice(_FORMATS),
    show_default=True,
    help="Output format(s). Repeat to combine: -f txt -f srt -f json",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Destination for transcript files (default: ./transcripts/).",
)
@click.option(
    "--recursive/--no-recursive",
    default=False, show_default=True,
    help="Descend into sub-folders.",
)
@click.option(
    "--skip-existing/--overwrite",
    default=True, show_default=True,
    help="Skip files already recorded in manifest.json.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List discovered audio files without transcribing.",
)
@click.option(
    "--min-speakers",
    default=None, type=int,
    help="Minimum speaker count hint (WhisperX only).",
)
@click.option(
    "--max-speakers",
    default=None, type=int,
    help="Maximum speaker count hint (WhisperX only).",
)
@click.option(
    "--no-diarize",
    is_flag=True,
    help="Disable speaker diarization even when using WhisperX.",
)
@click.option(
    "--speaker-names",
    default=None,
    help='JSON map of speaker IDs to names: \'{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}\'',
)
def main(
    folder: Path,
    backend: str,
    model: str | None,
    language: str | None,
    formats: tuple[str, ...],
    output_dir: Path | None,
    recursive: bool,
    skip_existing: bool,
    dry_run: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    no_diarize: bool,
    speaker_names: str | None,
) -> None:
    """Transcribe audio files in FOLDER and write speaker-labeled transcripts."""

    resolved_model = model or _DEFAULT_MODEL[backend]
    resolved_output = output_dir or (Path.cwd() / "transcripts")
    resolved_formats = list(formats) or list(_DEFAULT_FORMATS)

    # Parse speaker name map
    speaker_map: dict[str, str] | None = None
    if speaker_names:
        try:
            speaker_map = json.loads(speaker_names)
        except json.JSONDecodeError as exc:
            console.print(f"[red]--speaker-names is not valid JSON: {exc}[/red]")
            sys.exit(1)

    # Print run summary
    console.rule("[bold]Audio Transcripts")
    console.print(f"  [bold]Folder:[/bold]   {folder}")
    console.print(f"  [bold]Backend:[/bold]  {backend}  ({resolved_model})")
    console.print(f"  [bold]Output:[/bold]   {resolved_output}")
    console.print(f"  [bold]Formats:[/bold]  {', '.join(resolved_formats)}")
    if backend == "whisperx" and not no_diarize:
        cache_dir = Path.cwd() / ".cache" / "whisperx"
        console.print(f"  [bold]Cache:[/bold]    {cache_dir}")
    if backend == "whisperx" and not no_diarize:
        if not os.environ.get("HF_TOKEN"):
            console.print(
                "  [yellow]⚠  HF_TOKEN not set — speaker diarization will be skipped.[/yellow]\n"
                "  [dim]Set HF_TOKEN in .env or request access at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1[/dim]"
            )
    console.print()

    # Build transcriber (skip for dry-run to avoid loading heavy models)
    transcriber = None
    if not dry_run:
        try:
            transcriber = _build_transcriber(
                backend=backend,
                model=resolved_model,
                language=language,
                diarize=not no_diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except ImportError as exc:
            console.print(f"[red]Missing dependency for backend '{backend}':[/red] {exc}")
            console.print(
                "[dim]Install with:[/dim]  pip install -e '.[local]'  "
                "[dim]or[/dim]  pip install -e '.[cloud]'"
            )
            sys.exit(1)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)

    from audio_transcripts.pipeline import run_pipeline

    run_pipeline(
        folder=folder,
        output_dir=resolved_output,
        transcriber=transcriber,
        formats=resolved_formats,
        recursive=recursive,
        skip_existing=skip_existing,
        dry_run=dry_run,
        speaker_names=speaker_map,
    )


if __name__ == "__main__":
    main()
