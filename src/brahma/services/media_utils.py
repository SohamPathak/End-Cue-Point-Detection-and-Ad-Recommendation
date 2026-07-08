"""Shared media helpers built on ffprobe/ffmpeg and a stable video hash.

Kept in one place so outro detection and the compositor reuse the same probing
logic rather than duplicating ffprobe calls.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from brahma.exceptions import CompositionError

_BLACK_RE = re.compile(r"black_start:(?P<start>[\d.]+)\s+black_end:(?P<end>[\d.]+)")


def video_hash(path: str | Path) -> str:
    """Return a stable hash identifying a video file.

    Uses path + size + mtime (cheap and stable) rather than hashing full bytes,
    which would be slow for large videos.

    Args:
        path: Path to the video.

    Returns:
        A short hex digest.
    """
    p = Path(path)
    stat = p.stat()
    key = f"{p.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def probe_duration(path: str | Path) -> float:
    """Return the container duration of a media file in seconds.

    Args:
        path: Path to the media file.

    Returns:
        Duration in seconds.

    Raises:
        CompositionError: If ffprobe fails or returns no duration.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = json.loads(out.stdout)["format"]["duration"]
        return float(duration)
    except (subprocess.CalledProcessError, KeyError, ValueError) as exc:
        raise CompositionError(f"ffprobe failed for {path}: {exc}") from exc


def probe_video_props(path: str | Path) -> dict[str, float]:
    """Return basic video-stream properties: width, height, fps.

    Args:
        path: Path to the video.

    Returns:
        Mapping with ``width``, ``height`` and ``fps`` (frames per second).

    Raises:
        CompositionError: If ffprobe fails.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(out.stdout)["streams"][0]
        num, _, den = stream["r_frame_rate"].partition("/")
        fps = float(num) / float(den) if den and float(den) else float(num)
        return {
            "width": float(stream["width"]),
            "height": float(stream["height"]),
            "fps": fps,
        }
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as exc:
        raise CompositionError(f"ffprobe video props failed for {path}: {exc}") from exc


def probe_video_codec(path: str | Path) -> str:
    """Return the video stream's codec name (e.g. ``h264``), or ``""`` on failure."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def has_audio_stream(path: str | Path) -> bool:
    """Return whether the media file contains at least one audio stream."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(out.stdout.strip())


def detect_black_segments(
    path: str | Path,
    *,
    min_duration: float = 0.1,
    pixel_threshold: float = 0.10,
) -> list[tuple[float, float]]:
    """Return ``(black_start, black_end)`` segments via ffmpeg blackdetect.

    Fade-to-black transitions are strong, cheap delimiters for outros/credits.

    Args:
        path: Path to the video.
        min_duration: Minimum black duration to report (seconds).
        pixel_threshold: Blackness threshold (0..1); higher is more permissive.

    Returns:
        A list of ``(start, end)`` tuples, in chronological order.
    """
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-vf",
            f"blackdetect=d={min_duration}:pix_th={pixel_threshold}",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    segments: list[tuple[float, float]] = []
    for match in _BLACK_RE.finditer(proc.stderr):
        segments.append((float(match["start"]), float(match["end"])))
    return segments


def extract_frame(
    path: str | Path, timestamp_sec: float, *, max_width: int | None = None
) -> bytes:
    """Extract a single JPEG frame at ``timestamp_sec``.

    Args:
        path: Path to the video.
        timestamp_sec: Seek time in seconds.
        max_width: If set, downscale the frame to at most this width (keeping
            aspect). Smaller frames upload faster and cost fewer vision tokens.

    Returns:
        Raw JPEG bytes of the frame.

    Raises:
        CompositionError: If ffmpeg fails to produce a frame.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
    ]
    if max_width:
        # -1 keeps aspect; only downscale (never upscale past the source width).
        cmd += ["-vf", f"scale='min({max_width},iw)':-1"]
    cmd += ["-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True)
        if not out.stdout:
            raise CompositionError(
                f"No frame extracted at {timestamp_sec}s from {path}"
            )
        return out.stdout
    except subprocess.CalledProcessError as exc:
        raise CompositionError(f"ffmpeg frame extraction failed: {exc}") from exc


def extract_frames(
    path: str | Path,
    timestamps: list[float],
    *,
    max_width: int | None = None,
    max_workers: int = 8,
) -> list[tuple[float, bytes]]:
    """Extract several frames concurrently.

    Frame extraction is independent per timestamp and I/O-bound, so running the
    per-frame ffmpeg processes in a small thread pool cuts wall-clock time
    substantially versus a serial loop. Frames that fail to extract are skipped.

    Args:
        path: Path to the video.
        timestamps: Seek times in seconds.
        max_width: Optional downscale width passed to :func:`extract_frame`.
        max_workers: Max concurrent ffmpeg processes.

    Returns:
        ``(timestamp, jpeg_bytes)`` pairs in the original timestamp order, for
        the frames that extracted successfully.
    """

    def _one(ts: float) -> tuple[float, bytes] | None:
        try:
            return ts, extract_frame(path, ts, max_width=max_width)
        except CompositionError:
            return None

    workers = max(1, min(max_workers, len(timestamps)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_one, timestamps))
    return [r for r in results if r is not None]
