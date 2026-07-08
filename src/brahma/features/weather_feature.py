"""Weather feature provider — wraps the Open-Meteo service as a signal.

This is the single enabled feature today. It shows the pattern every future
provider follows: take the request context, produce a :class:`Feature` whose
``detail`` carries enough context for the recommender to explain itself.
"""

from __future__ import annotations

from brahma.config import AppConfig, FeatureConfig, get_config
from brahma.features.base import FeatureProvider, RequestContext
from brahma.models import Feature, GeoCandidate
from brahma.services.weather_api import get_weather, weather_for_candidate


class WeatherFeatureProvider(FeatureProvider):
    """Provides the ``weather`` categorical feature (sunny | rainy)."""

    key = "weather"

    def __init__(
        self, definition: FeatureConfig, config: AppConfig | None = None
    ) -> None:
        """Initialise with the feature definition and optional config.

        Args:
            definition: The feature's config entry.
            config: Optional config override.
        """
        super().__init__(definition)
        self._config = config or get_config()

    def collect(self, context: RequestContext) -> Feature:
        """Fetch weather for the request location and wrap it as a feature.

        If the context carries a pre-resolved ``geo`` candidate (the user's
        disambiguation choice), it is used directly so the location is never
        re-guessed; otherwise the free-text ``location`` is geocoded.

        Args:
            context: The request context (uses ``context.location`` and an
                optional ``geo`` :class:`GeoCandidate` in ``context.extra``).

        Returns:
            A categorical ``weather`` feature carrying the full weather detail.
        """
        geo = context.extra.get("geo")
        if isinstance(geo, GeoCandidate):
            weather = weather_for_candidate(
                geo, self._config, location_query=context.location
            )
        else:
            weather = get_weather(context.location, self._config)
        return Feature(
            name=self.definition.name,
            value=weather.condition.value,
            source=self.definition.source,
            dtype=self.definition.dtype,
            detail={
                "location_resolved": weather.location_resolved,
                "weather_code": weather.weather_code,
                "latitude": weather.latitude,
                "longitude": weather.longitude,
                # Full result kept for the pipeline (UI shows it).
                "weather_result": weather.model_dump(mode="json"),
            },
        )
