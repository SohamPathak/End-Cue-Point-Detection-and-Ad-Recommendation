"""Offline evaluation harness for the outro detector.

CV signals (ffmpeg blackdetect + PySceneDetect) are NOT used in production — the
detector is Gemini-only. Here they serve as a cheap *reference* to measure the
Gemini detector's error, plus optional hand-labelled ground truth.

The key metric is the **signed error** ``pred - reference``:
  - positive  = LATE  (safe — ad appears a bit into the credits)
  - negative  = EARLY (bad — ad would overlay real content)

Usage:
    python scripts/eval_outro.py VIDEO [--truth SECONDS] [--json]

Example:
    python scripts/eval_outro.py "sample_video/my_video.mp4" --truth 559.5
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from scenedetect import ContentDetector, SceneManager, open_video

from brahma.config import get_config
from brahma.services.media_utils import (
    detect_black_segments,
    probe_duration,
)
from brahma.services.outro_detection import detect_outro


def cv_reference(video_path: str) -> dict[str, Optional[float]]:
    """Compute CV reference boundaries near the end of the video.

    Returns:
        Mapping with the last fade-to-black start and the last scene cut.
    """
    cfg = get_config().outro
    duration = probe_duration(video_path)
    window_start = max(0.0, duration - cfg.max_outro_lookback_sec)

    black = detect_black_segments(
        video_path,
        min_duration=cfg.eval_black_min_duration,
        pixel_threshold=cfg.eval_black_pixel_threshold,
    )
    black_in_window = [s for s, _ in black if window_start <= s < duration - 1.0]

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=cfg.eval_scene_threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    cuts = [
        sc[0].get_seconds()
        for sc in scene_manager.get_scene_list()
        if sc[0].get_seconds() >= window_start
    ]

    return {
        "last_black_start": min(black_in_window) if black_in_window else None,
        "last_scene_cut": min(cuts) if cuts else None,
        "duration": duration,
    }


def evaluate(video_path: str, truth: Optional[float]) -> dict[str, object]:
    """Run the Gemini detector and compare against CV + optional ground truth."""
    ref = cv_reference(video_path)
    result = detect_outro(video_path, force=True)
    pred = result.start_sec

    # Prefer explicit ground truth; else fall back to the fade-to-black reference.
    reference = truth if truth is not None else ref["last_black_start"]
    signed_error = round(pred - reference, 3) if reference is not None else None

    return {
        "video": video_path,
        "prediction_sec": pred,
        "confidence": result.confidence,
        "method": result.method,
        "cv_reference": ref,
        "ground_truth_sec": truth,
        "reference_used_sec": reference,
        "signed_error_sec": signed_error,
        "verdict": _verdict(signed_error),
    }


def _verdict(signed_error: Optional[float]) -> str:
    """Human-readable verdict for a signed error (late is safe, early is bad)."""
    if signed_error is None:
        return "no-reference"
    if signed_error < -0.5:
        return "EARLY (bad — would overlay content)"
    if signed_error <= 2.0:
        return "GOOD (on-time / safely late)"
    return "LATE (safe but missed some credits)"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the outro detector.")
    parser.add_argument("video", help="Path to the video file.")
    parser.add_argument(
        "--truth", type=float, default=None, help="Hand-labelled outro start (s)."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    report = evaluate(args.video, args.truth)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Video           : {report['video']}")
    print(
        f"Prediction      : {report['prediction_sec']}s "
        f"(conf {report['confidence']}, {report['method']})"
    )
    print(f"CV reference     : {report['cv_reference']}")
    print(f"Reference used   : {report['reference_used_sec']}s")
    print(f"Signed error     : {report['signed_error_sec']}s")
    print(f"Verdict          : {report['verdict']}")


if __name__ == "__main__":
    main()
