"""Compositor smoke tests on synthetic clips (real ffmpeg)."""

from __future__ import annotations

import subprocess

from brahma.services.compositor import composite_ad
from brahma.services.media_utils import has_audio_stream, probe_duration
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_short_ad_crossfades_back_to_tail(config, synthetic_video, synthetic_ad):
    # base=8s, outro_start=2, delay=2 -> ad_appear=4; ad=2s -> tail_start=6 < end
    out = composite_ad(
        synthetic_video, synthetic_ad, 2.0, "smoke_tail", config, force=True
    )
    dur = probe_duration(out)
    # pre(4s) + ad(2s) then xfade back into ~2s of tail, minus 1s overlap ≈ 7s
    assert 6.0 < dur < 8.5


@requires_ffmpeg
def test_long_ad_plays_to_completion(
    config, synthetic_video, synthetic_ad, monkeypatch
):
    # Force a long-ad scenario: outro near the end so no tail remains.
    out = composite_ad(
        synthetic_video, synthetic_ad, 7.5, "smoke_long", config, force=True
    )
    dur = probe_duration(out)
    # ad_appear=min(7.5+2, 8)=8 (clamped) + 2s ad = ~10s (extends past original 8s)
    assert dur > 8.0


@requires_ffmpeg
def test_output_is_cached(config, synthetic_video, synthetic_ad):
    first = composite_ad(
        synthetic_video, synthetic_ad, 2.0, "smoke_cache", config, force=True
    )
    # Second call without force must return the same cached path, no re-render.
    second = composite_ad(synthetic_video, synthetic_ad, 2.0, "smoke_cache", config)
    assert first == second


@requires_ffmpeg
def test_silent_base_video_gets_ad_appended(config, synthetic_ad, tmp_path):
    """A silent base video (e.g. a screen recording) must still composite.

    The no-outro case appends the ad at the end (outro_start == duration); the
    compositor must synthesise silence for the base and still produce audio.
    """
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:d=4:r=25",
            "-pix_fmt",
            "yuv420p",
            str(silent),
        ],
        check=True,
    )
    assert has_audio_stream(str(silent)) is False
    out = composite_ad(
        str(silent), synthetic_ad, 4.0, "silent_append", config, force=True
    )
    # 4s silent video + 2s ad appended (start == duration → no tail).
    assert probe_duration(out) > 4.0
    assert has_audio_stream(out) is True
