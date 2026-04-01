from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .output.json_writer import write_json
from .output.srt import write_srt
from .output.txt import write_txt
from .output.vtt import write_vtt
from .scanner import scan_audio_files
from .transcribers.base import BaseTranscriber, TranscriptResult


def _is_interactive_terminal() -> bool:
    """Return True only when stdout is a real TTY that supports cursor movement.

    Rich's Live display uses ANSI cursor-movement codes (e.g. move-up N lines)
    to overwrite previous frames.  These codes are silently ignored in:
      - Colab / Jupyter (!python subprocess)
      - piped output  (python transcribe.py | tee log.txt)
      - Docker without -t

    Colab's !-shell allocates a pseudo-TTY for subprocesses, so isatty()
    incorrectly returns True there.  We detect Colab explicitly via env vars
    that are always set in its runtime.
    """
    import os
    import sys

    if os.environ.get("COLAB_BACKEND_VERSION") or os.environ.get("COLAB_RELEASE_TAG"):
        return False
    return sys.stdout.isatty()


console = Console()

FORMAT_WRITERS = {
    "txt": (write_txt, ".txt"),
    "json": (write_json, ".json"),
    "srt": (write_srt, ".srt"),
    "vtt": (write_vtt, ".vtt"),
}

MANIFEST_NAME = "manifest.json"


def _load_manifest(output_dir: Path) -> dict:
    path = output_dir / MANIFEST_NAME
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "files": {}}


def _save_manifest(output_dir: Path, manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _already_done(manifest: dict, key: str, formats: list[str]) -> bool:
    entry = manifest["files"].get(key)
    if not entry or entry.get("status") != "done":
        return False
    return all(fmt in entry.get("outputs", []) for fmt in formats)


def run_pipeline(
    folder: Path,
    output_dir: Path,
    transcriber: Optional[BaseTranscriber],
    formats: list[str],
    recursive: bool = False,
    skip_existing: bool = True,
    dry_run: bool = False,
    speaker_names: Optional[dict[str, str]] = None,
    files: Optional[list[Path]] = None,
) -> None:
    """Scan *folder* for audio files and transcribe each one.

    Args:
        folder: Root directory containing audio files.
        output_dir: Where to write transcript files and manifest.json.
        transcriber: Backend to use. May be None only when dry_run=True.
        formats: List of output format keys, e.g. ['txt', 'srt'].
        recursive: Descend into sub-directories.
        skip_existing: Skip files already present in the manifest.
        dry_run: Only list discovered files; do not transcribe.
        speaker_names: Optional map from SPEAKER_XX to human names.
        files: If given, process only these files instead of scanning folder.
    """
    audio_files = files if files is not None else scan_audio_files(folder, recursive=recursive)

    if not audio_files:
        console.print(f"[yellow]No audio files found in {folder}[/yellow]")
        return

    # ── Dry run: display file list and exit ──────────────────────────────────
    if dry_run:
        table = Table(title=f"Audio files in {folder}", show_lines=True)
        table.add_column("#", style="dim", justify="right")
        table.add_column("File")
        table.add_column("Size", justify="right")
        for i, f in enumerate(audio_files, 1):
            size_mb = f.stat().st_size / (1024 * 1024)
            table.add_row(str(i), str(f.relative_to(folder)), f"{size_mb:.1f} MB")
        console.print(table)
        console.print(f"\n[bold]{len(audio_files)}[/bold] audio file(s) found.")
        return

    # ── Transcription run ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(output_dir)

    done_count = 0
    skip_count = 0
    error_files: list[tuple[str, str]] = []

    if not _is_interactive_terminal():
        # ── Non-TTY (Colab, pipe, Docker): plain per-file status lines ──────
        # Rich's Live relies on ANSI cursor-movement codes (move-up N lines) to
        # overwrite the previous frame.  Colab ignores those codes even when
        # invoked via !python (subprocess), causing every refresh to print a
        # new line.  isatty()=False is the reliable signal to avoid Live.
        total = len(audio_files)
        for i, audio_file in enumerate(audio_files, 1):
            key = str(audio_file)

            if skip_existing and _already_done(manifest, key, formats):
                console.print(f"  [dim]skip[/dim]  {audio_file.name}")
                skip_count += 1
                continue

            console.print(f"  [{i}/{total}] [bold blue]{audio_file.name}[/bold blue]")

            def _stage(msg: str) -> None:  # noqa: E306
                if msg:
                    console.print(f"        [dim]{msg}[/dim]")

            try:
                result: TranscriptResult = transcriber.transcribe(  # type: ignore[union-attr]
                    audio_file, status_callback=_stage
                )

                if speaker_names:
                    result.apply_speaker_names(speaker_names)

                stem = audio_file.stem
                for fmt in formats:
                    writer_fn, ext = FORMAT_WRITERS[fmt]
                    writer_fn(result, output_dir / (stem + ext))

                manifest["files"][key] = {
                    "status": "done",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "backend": transcriber.backend_name,  # type: ignore[union-attr]
                    "model": transcriber.model,  # type: ignore[union-attr]
                    "diarization": result.diarization,
                    "speakers_detected": result.speakers_detected,
                    "outputs": formats,
                }
                _save_manifest(output_dir, manifest)
                done_count += 1
                console.print(f"  [green]✓[/green]  {audio_file.name}")

            except Exception as exc:
                console.print(f"  [red]✗[/red]  {audio_file.name}: {exc}")
                error_files.append((audio_file.name, str(exc)))
                manifest["files"][key] = {
                    "status": "error",
                    "error": str(exc),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                _save_manifest(output_dir, manifest)

    else:
        # ── Terminal: rich Live display with animated progress bar ────────────
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total} files"),
            TimeElapsedColumn(),
            console=console,
        )
        stage_progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("  [dim]{task.description}[/dim]"),
            TimeElapsedColumn(),
            console=console,
        )

        with Live(console=console, refresh_per_second=8) as live:
            from rich.console import Group

            def _refresh(status: str = "") -> None:
                live.update(Group(progress, stage_progress))

            file_task = progress.add_task("Transcribing…", total=len(audio_files))
            stage_task = stage_progress.add_task("", total=None)

            for audio_file in audio_files:
                key = str(audio_file)
                progress.update(file_task, description=f"[bold blue]{audio_file.name}")

                if skip_existing and _already_done(manifest, key, formats):
                    console.print(f"  [dim]skip[/dim]  {audio_file.name}")
                    skip_count += 1
                    progress.advance(file_task)
                    continue

                def _stage(msg: str) -> None:  # noqa: E306
                    stage_progress.update(stage_task, description=msg)
                    _refresh()

                try:
                    result: TranscriptResult = transcriber.transcribe(  # type: ignore[union-attr]
                        audio_file, status_callback=_stage
                    )

                    if speaker_names:
                        result.apply_speaker_names(speaker_names)

                    stem = audio_file.stem
                    for fmt in formats:
                        writer_fn, ext = FORMAT_WRITERS[fmt]
                        writer_fn(result, output_dir / (stem + ext))

                    manifest["files"][key] = {
                        "status": "done",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "backend": transcriber.backend_name,  # type: ignore[union-attr]
                        "model": transcriber.model,  # type: ignore[union-attr]
                        "diarization": result.diarization,
                        "speakers_detected": result.speakers_detected,
                        "outputs": formats,
                    }
                    _save_manifest(output_dir, manifest)
                    done_count += 1
                    stage_progress.update(stage_task, description="")
                    console.print(f"  [green]✓[/green]     {audio_file.name}")

                except Exception as exc:
                    console.print(f"  [red]✗[/red]     {audio_file.name}: {exc}")
                    error_files.append((audio_file.name, str(exc)))
                    manifest["files"][key] = {
                        "status": "error",
                        "error": str(exc),
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    _save_manifest(output_dir, manifest)
                    stage_progress.update(stage_task, description="")

                progress.advance(file_task)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    parts = [f"[bold green]{done_count} transcribed[/bold green]"]
    if skip_count:
        parts.append(f"[dim]{skip_count} skipped[/dim]")
    if error_files:
        parts.append(f"[bold red]{len(error_files)} failed[/bold red]")
    console.print("  ".join(parts))

    if error_files:
        for name, err in error_files:
            console.print(f"  [red]{name}[/red]: {err}")
