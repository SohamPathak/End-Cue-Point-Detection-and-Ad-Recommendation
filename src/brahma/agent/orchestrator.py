"""Pipeline orchestrator — coordinates the three phases into one result.

Phases:
  1. **Video prep** (once per video, cached): detect the outro cuepoint.
  2. **Per-location** (re-runs per location): collect features → recommend an ad.
  3. **Composite**: burn the chosen ad into the outro.

The orchestrator is a deterministic coordinator. The "agentic" reasoning lives in
its collaborators — the vision-based outro detector and the recommender (whose
``strategy`` may be ``llm``). ``agent.enabled`` in config forces the recommender
to the deterministic ``rule`` strategy for a fully reproducible run.
"""

from __future__ import annotations

from brahma.config import AppConfig, get_config
from brahma.features.base import RequestContext
from brahma.features.collector import FeatureCollector
from brahma.models import (
    AdRecommendation,
    FeatureSet,
    GeoCandidate,
    OutroResult,
    PipelineResult,
    WeatherResult,
)
from brahma.recommender.recommender import Recommender
from brahma.services.compositor import composite_ad
from brahma.services.outro_detection import detect_outro


class Orchestrator:
    """Runs the end-to-end weather-aware outro ad-insertion pipeline."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialise the orchestrator and its collaborators.

        Args:
            config: Optional config override.
        """
        self._config = config or get_config()
        self._collector = FeatureCollector(self._config)
        self._recommender = Recommender(self._config)

    def prepare_video(self, video_path: str, *, force: bool = False) -> OutroResult:
        """Phase 1: detect (and cache) the outro cuepoint for a video.

        Args:
            video_path: Path to the video.
            force: Re-detect even if a cached result exists.

        Returns:
            The detected :class:`OutroResult`.
        """
        return detect_outro(video_path, self._config, force=force)

    def recommend(
        self, location: str, geo: GeoCandidate | None = None
    ) -> tuple[FeatureSet, AdRecommendation]:
        """Phase 2: collect features for a location and recommend an ad.

        Args:
            location: Free-text location.
            geo: An optional pre-resolved geocoding candidate (the user's
                disambiguation choice) so the location is not re-guessed.

        Returns:
            The collected :class:`FeatureSet` and the :class:`AdRecommendation`.
        """
        context = RequestContext(location=location, geo=geo)
        features = self._collector.collect(context)
        recommendation = self._recommender.recommend(features)
        return features, recommendation

    def run(
        self,
        video_path: str,
        location: str,
        *,
        geo: GeoCandidate | None = None,
        outro: OutroResult | None = None,
        force_outro: bool = False,
        force_composite: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline for a video + location.

        Args:
            video_path: Path to the source video.
            location: Free-text location.
            geo: Optional pre-resolved geocoding candidate (user's choice).
            outro: A pre-computed outro result (Phase 1 cache) to reuse; if
                ``None`` it is detected here.
            force_outro: Force outro re-detection.
            force_composite: Force re-compositing even if cached.

        Returns:
            The complete :class:`PipelineResult`.
        """
        outro = outro or self.prepare_video(video_path, force=force_outro)
        features, recommendation = self.recommend(location, geo)

        output_path = composite_ad(
            video_path,
            self._config.abs_path(recommendation.ad_path).as_posix(),
            outro.start_sec,
            recommendation.ad_id,
            self._config,
            force=force_composite,
        )

        weather = self._extract_weather(features, location)
        reasoning = self._build_reasoning(weather, outro, recommendation)
        return PipelineResult(
            location_query=location,
            weather=weather,
            outro=outro,
            recommendation=recommendation,
            output_video_path=output_path,
            reasoning=reasoning,
        )

    @staticmethod
    def _extract_weather(features: FeatureSet, location: str) -> WeatherResult:
        """Recover the full WeatherResult stored in the weather feature detail."""
        feature = features.get("weather")
        if feature is None or "weather_result" not in feature.detail:
            raise ValueError("Weather feature missing from the collected feature set.")
        return WeatherResult.model_validate(feature.detail["weather_result"])

    def _build_reasoning(
        self,
        weather: WeatherResult,
        outro: OutroResult,
        recommendation: AdRecommendation,
    ) -> str:
        """Compose a short human-readable narrative of the decision."""
        delay = self._config.compositor.ad_start_delay_sec
        weather_part = (
            f"Weather in {weather.location_resolved} is "
            f"'{weather.condition.value}' (WMO code {weather.weather_code}). "
            f"{recommendation.reason} "
        )
        if not outro.has_outro:
            return (
                f"{weather_part}"
                f"No end cue point (outro/credits) was detected in this video "
                f"(confidence {outro.confidence:.0%}); to avoid overlaying real "
                f"content, the '{recommendation.ad_id}' ad is appended at the end."
            )
        mm, ss = divmod(int(round(outro.start_sec)), 60)
        return (
            f"{weather_part}"
            f"The end cue point was detected at {mm}:{ss:02d} "
            f"(confidence {outro.confidence:.0%}, method {outro.method}); "
            f"the '{recommendation.ad_id}' ad is burned in full-screen starting "
            f"{delay:.0f}s into it."
        )
