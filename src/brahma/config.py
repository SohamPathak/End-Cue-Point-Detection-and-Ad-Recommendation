"""Typed, cached configuration loaded from ``config.yaml`` + environment.

Environment variables (see ``.env.example``) hold only secrets/paths; all
behavioural knobs live in ``configs/config.yaml``. Access the singleton via
``get_config()``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiConfig(BaseModel):
    """Vertex AI / Gemini settings."""

    auth: str = "vertex"
    location: str = "us-central1"
    orchestrator_model: str = "gemini-2.5-flash"
    vision_model: str = "gemini-2.5-flash"
    request_timeout_sec: int = 60


class WeatherConfig(BaseModel):
    """Open-Meteo endpoints and rain classification."""

    provider: str = "open-meteo"
    geocoding_url: str
    forecast_url: str
    rainy_codes: list[int]
    geocode_candidates: int = 5
    geocode_aliases: dict[str, str] = Field(default_factory=dict)


class FeatureConfig(BaseModel):
    """Declarative definition of one feature/signal."""

    name: str
    provider: str
    source: str = "api"
    enabled: bool = True
    dtype: str = "categorical"


class RecommenderConfig(BaseModel):
    """Recommender strategy — fixed per experiment."""

    strategy: str = "rule"  # rule | llm
    llm_model: str = "gemini-2.5-flash"
    experiment_id: str = "exp-weather-v1"


class AdConfig(BaseModel):
    """One ad in the catalog with matching metadata."""

    id: str
    path: str
    tags: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class OutroConfig(BaseModel):
    """Outro-detection tuning (Gemini-only coarse→fine)."""

    max_outro_lookback_sec: float = 300.0
    coarse_step_sec: float = 10.0
    fine_step_sec: float = 1.0
    min_outro_sec: float = 3.0
    safety_bias_sec: float = 0.0
    # CV knobs — used only by the offline eval harness, not the request path.
    eval_scene_threshold: float = 27.0
    eval_black_min_duration: float = 0.1
    eval_black_pixel_threshold: float = 0.10


class CompositorConfig(BaseModel):
    """ffmpeg burn-in settings."""

    ad_start_delay_sec: float = 2.0
    fade_duration_sec: float = 1.0
    scale_mode: str = "fit"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 20


class AgentConfig(BaseModel):
    """Orchestrator behaviour."""

    enabled: bool = True


class PathsConfig(BaseModel):
    """Filesystem locations (relative to project root)."""

    output_dir: str = "outputs"
    cache_dir: str = "outputs/cache"
    sample_video_dir: str = "sample_video"
    prompts_dir: str = "configs/prompts"
    logo: str = "BRAHMA_AI_Logo.jpg"


class EnvSettings(BaseSettings):
    """Secrets and paths sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_application_credentials: str = Field(
        "configs/bq_creds.json", alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    vertex_location: Optional[str] = Field(None, alias="VERTEX_LOCATION")
    brahma_config: str = Field("configs/config.yaml", alias="BRAHMA_CONFIG")


class AppConfig(BaseModel):
    """Root configuration object for the whole app."""

    gemini: GeminiConfig
    weather: WeatherConfig
    features: list[FeatureConfig]
    recommender: RecommenderConfig
    ads: list[AdConfig]
    outro: OutroConfig
    compositor: CompositorConfig
    agent: AgentConfig
    paths: PathsConfig

    # Populated from env, not YAML.
    creds_path: str
    project_root: Path

    def enabled_features(self) -> list[FeatureConfig]:
        """Return only the features marked enabled."""
        return [f for f in self.features if f.enabled]

    def ad_by_id(self, ad_id: str) -> Optional[AdConfig]:
        """Look up an ad by its catalog id."""
        return next((a for a in self.ads if a.id == ad_id), None)

    def abs_path(self, path: str) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(path)
        return p if p.is_absolute() else self.project_root / p


def _project_root() -> Path:
    """Locate the project root (dir containing ``configs/``)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir():
            return parent
    return Path.cwd()


@functools.lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache the application configuration.

    Returns:
        The fully-parsed :class:`AppConfig`.

    Raises:
        FileNotFoundError: If the YAML config file cannot be found.
    """
    env = EnvSettings()
    root = _project_root()

    config_path = root / env.brahma_config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    # Env overrides YAML for the Vertex region when provided.
    if env.vertex_location:
        raw.setdefault("gemini", {})["location"] = env.vertex_location

    return AppConfig(
        **raw,
        creds_path=str(root / env.google_application_credentials),
        project_root=root,
    )


def reset_config_cache() -> None:
    """Clear the cached config (used in tests)."""
    get_config.cache_clear()
    os.environ.pop("_BRAHMA_CACHE_BUST", None)
