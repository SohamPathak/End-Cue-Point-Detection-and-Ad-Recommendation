"""Tests for outro-detection logic with vision mocked (no Vertex, no real video)."""

from __future__ import annotations

import pytest

from brahma.clients.gemini import parse_json_response
from brahma.services import outro_detection


def test_grid_spacing_inclusive():
    grid = outro_detection._grid(0.0, 10.0, 5.0)
    assert grid[0] == 0.0
    assert grid[-1] == 10.0
    assert 5.0 in grid


def test_grid_dedups_endpoint():
    grid = outro_detection._grid(0.0, 10.0, 10.0)
    assert grid == [0.0, 10.0]


def test_parse_json_response_strips_fences():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('{"b": 2}') == {"b": 2}


@pytest.fixture
def stub_media(monkeypatch):
    """Stub media helpers so no real video is needed."""
    monkeypatch.setattr(outro_detection, "probe_duration", lambda p: 600.0)
    monkeypatch.setattr(outro_detection, "video_hash", lambda p: "stubhash")
    monkeypatch.setattr(outro_detection, "extract_frame", lambda p, t: b"jpeg")


def test_coarse_fine_never_early(config, stub_media, monkeypatch, tmp_path):
    """The reported start must equal the vision-picked frame (never earlier)."""
    monkeypatch.setattr(config.paths, "cache_dir", str(tmp_path))

    calls = {"n": 0}

    def fake_ask(video_path, timestamps, duration):
        calls["n"] += 1
        # Always "pick" the last candidate frame in the given grid.
        return len(timestamps) - 1, 0.9, timestamps

    monkeypatch.setattr(outro_detection, "_ask_vision", fake_ask)
    result = outro_detection.detect_outro("v.mp4", config, force=True)
    # Two passes (coarse then fine) must have run.
    assert calls["n"] == 2
    # Never before window start, never after the min-outro clamp.
    assert result.start_sec <= 600.0 - config.outro.min_outro_sec
    assert result.method == "gemini-coarse-fine"


def test_no_outro_appends_at_end(config, stub_media, monkeypatch, tmp_path):
    """If vision sees no outro (e.g. a screen recording), append the ad at the end.

    The ad must NOT be placed over real content, so start_sec == duration and
    has_outro is False.
    """
    monkeypatch.setattr(config.paths, "cache_dir", str(tmp_path))
    monkeypatch.setattr(outro_detection, "_ask_vision", lambda *a, **k: (-1, 0.4, a[1]))
    result = outro_detection.detect_outro("v.mp4", config, force=True)
    assert result.has_outro is False
    assert result.start_sec == pytest.approx(600.0)  # == duration → appended
    assert "no-outro" in result.method
