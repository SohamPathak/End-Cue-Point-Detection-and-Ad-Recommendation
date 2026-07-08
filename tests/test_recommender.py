"""Tests for the feature layer and recommender."""

from __future__ import annotations

import pytest

from brahma.models import Feature, FeatureSet
from brahma.recommender.recommender import Recommender


def _fs(weather: str) -> FeatureSet:
    return FeatureSet(
        features=[
            Feature(name="weather", value=weather, source="api", dtype="categorical")
        ]
    )


def test_rule_strategy_rainy_selects_umbrella(config):
    rec = Recommender(config).recommend(_fs("rainy"))
    assert rec.ad_id == "umbrella"
    assert rec.strategy == "rule"
    assert rec.scores["umbrella"] == 1.0


def test_rule_strategy_sunny_selects_sunscreen(config):
    rec = Recommender(config).recommend(_fs("sunny"))
    assert rec.ad_id == "sunscreen"


def test_no_matching_feature_raises(config):
    from brahma.exceptions import RecommendationError

    with pytest.raises(RecommendationError):
        Recommender(config).recommend(_fs("snowy"))


def test_llm_strategy_falls_back_to_rule_on_error(config, monkeypatch):
    """If the LLM strategy errors, the recommender degrades to the rule result."""
    monkeypatch.setattr(config.recommender, "strategy", "llm")
    monkeypatch.setattr(config.agent, "enabled", True)

    def boom(*args, **kwargs):
        raise RuntimeError("no vertex in tests")

    monkeypatch.setattr("brahma.recommender.recommender.get_gemini_client", boom)
    rec = Recommender(config).recommend(_fs("rainy"))
    assert rec.ad_id == "umbrella"  # rule fallback still works
    assert "fallback" in rec.reason.lower()


def test_agent_disabled_forces_rule(config, monkeypatch):
    """agent.enabled=False forces the deterministic rule even if strategy=llm."""
    monkeypatch.setattr(config.recommender, "strategy", "llm")
    monkeypatch.setattr(config.agent, "enabled", False)
    rec = Recommender(config).recommend(_fs("sunny"))
    assert rec.strategy == "rule"
    assert rec.ad_id == "sunscreen"
