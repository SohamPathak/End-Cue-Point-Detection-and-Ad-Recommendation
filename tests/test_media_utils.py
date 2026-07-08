"""Tests for media helpers against synthetic ffmpeg clips."""

from __future__ import annotations

import pytest

from brahma.exceptions import CompositionError
from brahma.services import media_utils
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_probe_duration(synthetic_video):
    assert media_utils.probe_duration(synthetic_video) == pytest.approx(8.0, abs=0.5)


@requires_ffmpeg
def test_probe_video_props(synthetic_video):
    props = media_utils.probe_video_props(synthetic_video)
    assert props["width"] == 320
    assert props["height"] == 240
    assert props["fps"] == pytest.approx(25.0)


@requires_ffmpeg
def test_has_audio_stream(synthetic_video):
    assert media_utils.has_audio_stream(synthetic_video) is True


@requires_ffmpeg
def test_extract_frame_returns_jpeg(synthetic_video):
    data = media_utils.extract_frame(synthetic_video, 1.0)
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker


@requires_ffmpeg
def test_detect_black_segments_on_solid_color(synthetic_video):
    # A solid red clip has no black segments.
    assert media_utils.detect_black_segments(synthetic_video) == []


def test_probe_duration_missing_file_raises():
    with pytest.raises(CompositionError):
        media_utils.probe_duration("/nonexistent/file.mp4")
