"""Gemini-only outro (end-credits) detection with a coarse→fine frame grid.

Rationale: credits realistically never exceed ~5 minutes, so we scan only a
bounded tail window. Gemini is the detector — CV (blackdetect/scenedetect) is
kept purely as an offline eval reference (see ``scripts/eval_outro.py``), not in
this request path.

Two passes keep both precision and token cost bounded:
1. **Coarse** — sample the window every ``coarse_step_sec`` and ask which frame
   first shows the outro.
2. **Fine** — resample every ``fine_step_sec`` between the coarse pick and the
   frame just before it, and ask again for the precise boundary.

NEVER-EARLY guarantee: the model returns the first frame that *clearly* shows the
outro (so the true start is at or before that timestamp), we never subtract, and
an optional ``safety_bias_sec`` can push the reported start later. Being a second
late is invisible; being early would overlay the ad on real content.

The result is cached per video hash, so detection runs once per video.
"""

from __future__ import annotations

import json
from pathlib import Path

from brahma.clients.gemini import get_gemini_client, parse_json_response
from brahma.config import AppConfig, get_config
from brahma.exceptions import OutroDetectionError
from brahma.models import OutroResult
from brahma.prompts import render_prompt
from brahma.services.media_utils import extract_frame, probe_duration, video_hash


def _cache_path(config: AppConfig, vhash: str) -> Path:
    """Return the on-disk cache path for a video hash."""
    cache_dir = config.abs_path(config.paths.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"outro_{vhash}.json"


def _grid(start: float, end: float, step: float) -> list[float]:
    """Return timestamps from ``start`` to ``end`` inclusive, spaced by ``step``."""
    if step <= 0:
        return [start]
    times: list[float] = []
    t = start
    while t < end:
        times.append(round(t, 3))
        t += step
    times.append(round(end, 3))
    # De-dup while preserving order.
    seen: set[float] = set()
    unique: list[float] = []
    for x in times:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique


def _ask_vision(
    video_path: str, timestamps: list[float], duration: float
) -> tuple[int, float, list[float]]:
    """Extract frames at ``timestamps`` and ask Gemini which starts the outro.

    Returns:
        ``(index, confidence, kept_timestamps)`` where ``index`` is a 0-based
        index into ``kept_timestamps``, or ``-1`` if none show the outro.
    """
    frames: list[bytes] = []
    kept: list[float] = []
    for ts in timestamps:
        try:
            frames.append(extract_frame(video_path, ts))
            kept.append(ts)
        except OutroDetectionError:
            continue
    if not frames:
        raise OutroDetectionError("Could not extract any frames for detection.")

    listing = "\n".join(
        f"Image {i + 1}: timestamp {ts:.2f}s" for i, ts in enumerate(kept)
    )
    prompt = render_prompt("outro_detection", duration=duration, listing=listing)
    raw = get_gemini_client().generate_vision(prompt, frames)
    choice = parse_json_response(raw)
    picked = int(choice["image_index"])  # 1-based, or 0 for "none"
    confidence = float(choice.get("confidence", 0.5))
    if picked <= 0:
        return -1, confidence, kept
    return min(picked - 1, len(kept) - 1), confidence, kept


def detect_outro(
    video_path: str,
    config: AppConfig | None = None,
    *,
    force: bool = False,
) -> OutroResult:
    """Detect the outro cuepoint for a video (Gemini coarse→fine), cached per video.

    Args:
        video_path: Path to the video file.
        config: Optional config override.
        force: If ``True``, ignore any cached result and re-detect.

    Returns:
        The :class:`OutroResult` with the outro start timestamp.

    Raises:
        OutroDetectionError: If detection fails entirely.
    """
    cfg = config or get_config()
    vhash = video_hash(video_path)
    cache_file = _cache_path(cfg, vhash)

    if not force and cache_file.is_file():
        return OutroResult.model_validate_json(cache_file.read_text(encoding="utf-8"))

    duration = probe_duration(video_path)
    window_start = max(0.0, duration - cfg.outro.max_outro_lookback_sec)

    method = "gemini-coarse-fine"
    has_outro = True
    try:
        # Coarse pass over the bounded tail window.
        coarse_ts = _grid(window_start, duration - 0.1, cfg.outro.coarse_step_sec)
        c_idx, c_conf, c_kept = _ask_vision(video_path, coarse_ts, duration)

        if c_idx < 0:
            # No outro anywhere in the window — the video simply ends (e.g. a
            # screen recording / raw clip). Append the ad at the very end rather
            # than burning it over real content.
            start_sec, confidence, method, has_outro = (
                duration,
                c_conf,
                "no-outro (appended at end)",
                False,
            )
        else:
            # Fine pass brackets the coarse pick on BOTH sides: from the previous
            # coarse frame up to the NEXT one. Searching only backward could never
            # correct a coarse pick that landed too early (on late content that
            # merely looks end-like); including the forward half lets the model
            # push the boundary later onto the first genuine outro frame.
            prev = c_kept[c_idx - 1] if c_idx > 0 else window_start
            nxt = c_kept[c_idx + 1] if c_idx + 1 < len(c_kept) else duration - 0.1
            fine_ts = _grid(prev, nxt, cfg.outro.fine_step_sec)
            f_idx, f_conf, f_kept = _ask_vision(video_path, fine_ts, duration)
            if f_idx < 0:
                # Fine pass now sees NO outro in the bracket → the coarse frame was
                # a false positive on content. Fall back to the next coarse frame
                # if there is one (later, never earlier); else no outro.
                if c_idx + 1 < len(c_kept):
                    start_sec, confidence = c_kept[c_idx + 1], c_conf
                else:
                    start_sec, confidence, method, has_outro = (
                        duration,
                        c_conf,
                        "no-outro (appended at end)",
                        False,
                    )
            else:
                start_sec, confidence = f_kept[f_idx], f_conf
    except (OutroDetectionError, json.JSONDecodeError, KeyError, ValueError) as exc:
        # On failure, append at the end (never risk an early cut over content).
        start_sec, confidence, method, has_outro = (
            duration,
            0.3,
            f"fallback ({exc})",
            False,
        )

    if has_outro:
        # NEVER-EARLY: only ever push the start later, never earlier.
        start_sec = start_sec + max(0.0, cfg.outro.safety_bias_sec)
        # Clamp implausibly short / out-of-range outros.
        start_sec = min(start_sec, max(0.0, duration - cfg.outro.min_outro_sec))

    result = OutroResult(
        video_hash=vhash,
        start_sec=round(start_sec, 3),
        video_duration_sec=round(duration, 3),
        has_outro=has_outro,
        method=method,
        confidence=confidence,
    )
    cache_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
