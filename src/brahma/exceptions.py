"""Central exception hierarchy for the Brahma pipeline.

All layers raise these so the UI and orchestrator can fail fast with a clear,
typed error instead of leaking library-specific exceptions.
"""

from __future__ import annotations


class BrahmaError(Exception):
    """Base class for all application errors."""


class ConfigError(BrahmaError):
    """Raised when configuration is missing or invalid."""


class WeatherError(BrahmaError):
    """Raised when weather cannot be resolved for a location."""


class OutroDetectionError(BrahmaError):
    """Raised when the outro cuepoint cannot be determined."""


class RecommendationError(BrahmaError):
    """Raised when no ad can be recommended for the collected features."""


class CompositionError(BrahmaError):
    """Raised when the video composition (ffmpeg) fails."""
