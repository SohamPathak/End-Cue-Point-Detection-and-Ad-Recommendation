"""Feature collector — runs all enabled providers and assembles a FeatureSet.

Providers run concurrently (they are independent I/O-bound signals). Adding a
signal is a two-line change: register the class here and enable it in
``config.yaml``. The recommender is handed the resulting :class:`FeatureSet`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Type

from brahma.config import AppConfig, get_config
from brahma.features.base import FeatureProvider, RequestContext
from brahma.features.weather_feature import WeatherFeatureProvider
from brahma.models import Feature, FeatureSet

# Registry: config ``provider`` key -> provider class.
# Extend here as new providers are implemented.
PROVIDER_REGISTRY: dict[str, Type[FeatureProvider]] = {
    WeatherFeatureProvider.key: WeatherFeatureProvider,
}


class FeatureCollector:
    """Builds feature providers from config and collects their signals."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Instantiate providers for every enabled feature definition.

        Args:
            config: Optional config override.

        Raises:
            KeyError: If a config feature references an unknown provider.
        """
        self._config = config or get_config()
        self._providers: list[FeatureProvider] = []
        for definition in self._config.enabled_features():
            provider_cls = PROVIDER_REGISTRY.get(definition.provider)
            if provider_cls is None:
                raise KeyError(
                    f"Unknown feature provider {definition.provider!r}. "
                    f"Registered: {sorted(PROVIDER_REGISTRY)}"
                )
            self._providers.append(provider_cls(definition))

    def collect(self, context: RequestContext) -> FeatureSet:
        """Run all enabled providers concurrently and return their features.

        Args:
            context: The request context passed to every provider.

        Returns:
            A :class:`FeatureSet` with one feature per enabled provider.
        """
        if not self._providers:
            return FeatureSet(features=[])

        with ThreadPoolExecutor(max_workers=len(self._providers)) as pool:
            features: list[Feature] = list(
                pool.map(lambda p: p.collect(context), self._providers)
            )
        return FeatureSet(features=features)
