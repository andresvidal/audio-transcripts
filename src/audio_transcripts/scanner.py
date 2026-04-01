from __future__ import annotations

from pathlib import Path

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".aac", ".webm", ".opus"}
)


def scan_audio_files(folder: Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of audio files found in *folder*.

    Args:
        folder: Directory to search.
        recursive: When True, descend into sub-directories.

    Returns:
        Sorted list of Path objects for every discovered audio file.
    """
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
