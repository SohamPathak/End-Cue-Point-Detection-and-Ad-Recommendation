"""Tests for the feature collector, weather feature, and orchestrator wiring."""

from __future__ import annotations

import pytest

from brahma.features.base import RequestContext
from brahma.features.collector import FeatureCollector
from brahma.models import (
    OutroResult,
    WeatherCondition,
    WeatherResult,
)


@pytest.fixture
def fake_weather(monkeypatch):
    """Patch the weather API to return a deterministic result (no network)."""

    def _apply(condition: WeatherCondition, code: int):
        result = WeatherResult(
            location_query="X",
            location_resolved="Testville, Nowhere",
            latitude=1.0,
            longitude=2.0,
            condition=condition,
            weather_code=code,
        )
        monkeypatch.setattr(
            "brahma.features.weather_feature.get_weather",
            lambda location, cfg=None: result,
        )
        return result

    return _apply


def test_collector_produces_weather_feature(config, fake_weather):
    fake_weather(WeatherCondition.RAINY, 61)
    fs = FeatureCollector(config).collect(RequestContext(location="anywhere"))
    feature = fs.get("weather")
    assert feature is not None
    assert feature.value == "rainy"
    assert feature.source == "api"
    assert "weather_result" in feature.detail


def test_unknown_provider_raises(config, monkeypatch):
    from brahma.config import FeatureConfig

    bad = FeatureConfig(name="mystery", provider="does_not_exist", source="api")
    monkeypatch.setattr(config, "features", [bad])
    with pytest.raises(KeyError):
        FeatureCollector(config)


def test_orchestrator_run_end_to_end(config, fake_weather, monkeypatch, tmp_path):
    """orchestrator.run wires phases together using mocked outro + composite."""
    fake_weather(WeatherCondition.SUNNY, 0)

    outro = OutroResult(
        video_hash="abc",
        start_sec=100.0,
        video_duration_sec=200.0,
        method="test",
        confidence=0.9,
    )
    # Avoid heavy work: stub detection + compositing.
    monkeypatch.setattr("brahma.agent.orchestrator.detect_outro", lambda *a, **k: outro)
    fake_out = tmp_path / "out.mp4"
    fake_out.write_text("x")
    monkeypatch.setattr(
        "brahma.agent.orchestrator.composite_ad", lambda *a, **k: str(fake_out)
    )

    from brahma.agent.orchestrator import Orchestrator

    result = Orchestrator(config).run("video.mp4", "Testville")
    assert result.recommendation.ad_id == "sunscreen"
    assert result.weather.condition is WeatherCondition.SUNNY
    assert result.output_video_path == str(fake_out)
    assert "end cue point was detected at 1:40" in result.reasoning  # 100s → m:ss


def test_prepare_video_delegates_to_detect(config, monkeypatch):
    outro = OutroResult(
        video_hash="h",
        start_sec=1.0,
        video_duration_sec=2.0,
        method="t",
        confidence=1.0,
    )
    monkeypatch.setattr("brahma.agent.orchestrator.detect_outro", lambda *a, **k: outro)
    from brahma.agent.orchestrator import Orchestrator

    assert Orchestrator(config).prepare_video("v.mp4").start_sec == 1.0
