"""Feature-provider abstraction — the extensibility seam of the system.

A *feature* is any signal that can influence the ad recommendation. Today there
is one (weather); tomorrow there may be many (time-of-day, geo demographics,
trending topics, user profile). Each is a :class:`FeatureProvider` that turns a
request context into a :class:`Feature`. The recommender never changes when a
new provider is added — it just reasons over whatever features it receives.
"""

from __future__ import annotations

import abc
from typing import Any

from brahma.config import FeatureConfig
from brahma.models import Feature


class RequestContext:
    """Inputs available to every provider for one recommendation request."""

    def __init__(self, location: str, **extra: Any) -> None:
        """Store the request inputs.

        Args:
            location: The user-supplied location string.
            **extra: Additional context future providers may need.
        """
        self.location = location
        self.extra = extra


class FeatureProvider(abc.ABC):
    """Base class for all feature providers.

    Subclasses declare a ``key`` matching ``provider`` in ``config.yaml`` and
    implement :meth:`collect`.
    """

    #: Registry key; must match ``FeatureConfig.provider``.
    key: str

    def __init__(self, definition: FeatureConfig) -> None:
        """Bind the provider to its declarative definition.

        Args:
            definition: The feature's config entry.
        """
        self.definition = definition

    @abc.abstractmethod
    def collect(self, context: RequestContext) -> Feature:
        """Produce this provider's feature for the given request.

        Args:
            context: The request context.

        Returns:
            The computed :class:`Feature`.
        """
        raise NotImplementedError
