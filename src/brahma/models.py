"""Domain data-transfer objects shared across the pipeline.

These are intentionally provider-agnostic: services return them, the recommender
consumes them, and the UI renders them. Adding a new signal means adding a
``Feature`` — no change to these envelope types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class WeatherCondition(str, Enum):
    """Reduced weather buckets the demo supports."""

    SUNNY = "sunny"
    RAINY = "rainy"


class GeoCandidate(BaseModel):
    """A single geocoding match for a free-text location query.

    Surfaced to the user so they can disambiguate (e.g. Bangalore, India vs.
    Bangalore Town, Pakistan) instead of the system silently picking the first.
    """

    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = Field("", description="State/region, when available.")

    @property
    def display(self) -> str:
        """Human-readable label like 'Bangalore, Karnataka, India'."""
        parts = [self.name, self.admin1, self.country]
        return ", ".join(p for p in parts if p)


class WeatherResult(BaseModel):
    """Resolved weather for a location (raw output of the weather API service)."""

    location_query: str = Field(..., description="Raw location string from the user.")
    location_resolved: str = Field(
        ..., description="Human-readable resolved place name."
    )
    latitude: float
    longitude: float
    condition: WeatherCondition
    weather_code: int = Field(..., description="Raw WMO weather code from Open-Meteo.")


class Feature(BaseModel):
    """A single signal value produced by a ``FeatureProvider``.

    The recommender reasons over a set of these. ``value`` is deliberately typed
    as ``Any`` so future numeric/embedding features fit the same envelope.
    """

    name: str
    value: Any
    source: str = Field(..., description="api | static | db | tool")
    dtype: str = Field(..., description="categorical | numeric | text | ...")
    detail: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific context for explanations."
    )


class FeatureSet(BaseModel):
    """The full set of features collected for one recommendation request."""

    features: list[Feature] = Field(default_factory=list)

    def get(self, name: str) -> Optional[Feature]:
        """Return the feature with ``name``, or ``None`` if absent."""
        return next((f for f in self.features if f.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        """Flatten to ``{name: value}`` for prompts and rule logic."""
        return {f.name: f.value for f in self.features}


class AdRecommendation(BaseModel):
    """The recommender's decision for one request."""

    ad_id: str
    ad_path: str
    reason: str = Field(..., description="Human-readable justification for the choice.")
    strategy: str = Field(..., description="rule | llm — how the decision was made.")
    experiment_id: str
    scores: dict[str, float] = Field(
        default_factory=dict, description="Optional per-ad scores for observability."
    )


class OutroResult(BaseModel):
    """Detected outro cuepoint for a video (cached per video hash)."""

    video_hash: str
    start_sec: float = Field(
        ...,
        description=(
            "Timestamp where the outro begins. When ``has_outro`` is False this "
            "equals the video duration (the ad is appended at the end)."
        ),
    )
    video_duration_sec: float
    has_outro: bool = Field(
        True,
        description="False when the video has no ending sequence (e.g. a raw clip).",
    )
    method: str = Field(
        ..., description="How the cuepoint was found (cv+vision, vision, fallback)."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class PipelineResult(BaseModel):
    """End-to-end result surfaced to the UI."""

    location_query: str
    weather: WeatherResult
    outro: OutroResult
    recommendation: AdRecommendation
    output_video_path: str
    reasoning: str = Field(..., description="Overall narrative shown to the user.")
