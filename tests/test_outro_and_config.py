"""Tests for config loading, prompts, media hashing, and the outro cache."""

from __future__ import annotations

from brahma.config import get_config
from brahma.models import OutroResult
from brahma.prompts import render_prompt
from brahma.services import outro_detection
from brahma.services.media_utils import video_hash
from tests.conftest import requires_ffmpeg


def test_config_loads_expected_shape(config):
    assert config.recommender.strategy in {"rule", "llm"}
    assert {a.id for a in config.ads} == {"umbrella", "sunscreen"}
    assert any(f.name == "weather" for f in config.enabled_features())


def test_prompt_renders_with_placeholders():
    rendered = render_prompt("outro_detection", duration=100.0, listing="Image 1: 90s")
    assert "100.00s" in rendered
    assert "Image 1: 90s" in rendered
    assert "NEVER TOO EARLY" in rendered  # never-early rule is present


@requires_ffmpeg
def test_video_hash_is_stable(synthetic_video):
    assert video_hash(synthetic_video) == video_hash(synthetic_video)


@requires_ffmpeg
def test_outro_cache_hit_skips_detection(synthetic_video, monkeypatch):
    """A cached result must be returned without invoking vision."""
    cfg = get_config()
    vhash = video_hash(synthetic_video)
    cache_file = outro_detection._cache_path(cfg, vhash)
    cached = OutroResult(
        video_hash=vhash,
        start_sec=42.0,
        video_duration_sec=100.0,
        method="cached-test",
        confidence=0.99,
    )
    cache_file.write_text(cached.model_dump_json(), encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("vision must not be called on a cache hit")

    monkeypatch.setattr(outro_detection, "_ask_vision", fail)
    result = outro_detection.detect_outro(synthetic_video)
    assert result.start_sec == 42.0
    assert result.method == "cached-test"
    cache_file.unlink(missing_ok=True)


def test_never_early_bias_is_non_negative(config):
    """Safety bias must never subtract time (never-early guarantee)."""
    assert config.outro.safety_bias_sec >= 0.0
