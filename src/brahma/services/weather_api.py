"""Weather lookup via the keyless Open-Meteo API.

Two steps: geocode a free-text location to lat/lon, then fetch the current
weather code and reduce it to the demo's ``sunny``/``rainy`` bucket. This is a
plain service — the recommender consumes it through ``WeatherFeatureProvider``.
"""

from __future__ import annotations

from typing import Any

import httpx

from brahma.config import AppConfig, get_config
from brahma.exceptions import WeatherError
from brahma.models import GeoCandidate, WeatherCondition, WeatherResult


def _classify(weather_code: int, rainy_codes: list[int]) -> WeatherCondition:
    """Reduce a WMO weather code to the supported buckets."""
    return (
        WeatherCondition.RAINY
        if weather_code in rainy_codes
        else WeatherCondition.SUNNY
    )


def _query_geocoder(
    client: httpx.Client, name: str, cfg: AppConfig
) -> list[dict[str, Any]]:
    """Call Open-Meteo geocoding for one name and return raw result dicts."""
    resp = client.get(
        cfg.weather.geocoding_url,
        params={
            "name": name,
            "count": cfg.weather.geocode_candidates,
            "language": "en",
            "format": "json",
        },
    )
    resp.raise_for_status()
    return resp.json().get("results") or []


def _expand_query_names(location: str, cfg: AppConfig) -> list[str]:
    """Expand a free-text query into ordered names to try against the geocoder.

    Open-Meteo's geocoder is name-exact, so "Udupi, Karnataka" returns nothing.
    We therefore also try the leading comma-segment ("Udupi") and any configured
    alias. Order is preserved and de-duplicated (case-insensitively):

    1. a known alias for the raw query (canonical name leads, e.g. Bombay→Mumbai);
    2. the raw query as typed;
    3. the leading comma-segment (drops trailing "state, country" qualifiers).
    """
    raw = location.strip()
    ordered = [raw]
    alias = cfg.weather.geocode_aliases.get(raw.lower())
    if alias:
        ordered.insert(0, alias)
    if "," in raw:
        head = raw.split(",", 1)[0].strip()
        if head:
            ordered.append(head)
            # An alias may also apply to the bare leading segment.
            head_alias = cfg.weather.geocode_aliases.get(head.lower())
            if head_alias:
                ordered.insert(0, head_alias)

    seen: set[str] = set()
    result: list[str] = []
    for name in ordered:
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result


def geocode_candidates(
    location: str, config: AppConfig | None = None, *, limit: int | None = None
) -> list[GeoCandidate]:
    """Return geocoding matches for a free-text location, for user disambiguation.

    The caller (UI) chooses between the matches rather than the system silently
    picking the first hit — e.g. "Bangalore" would otherwise resolve to Pakistan.
    Because Open-Meteo's geocoder is name-exact, a small config-driven alias map
    (``weather.geocode_aliases``) is *also* queried and merged, so renamed cities
    (Bangalore→Bengaluru, Bombay→Mumbai, …) still appear as options.

    Args:
        location: Free-text place query.
        config: Optional config override.
        limit: Max candidates to return (defaults to config value).

    Returns:
        A list of :class:`GeoCandidate`, best match first, de-duplicated by
        coordinate (may be empty).

    Raises:
        WeatherError: If the geocoding request fails.
    """
    cfg = config or get_config()
    cap = limit or cfg.weather.geocode_candidates
    names = _expand_query_names(location, cfg)

    raw: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=15.0) as client:
            for name in names:
                raw.extend(_query_geocoder(client, name, cfg))
    except httpx.HTTPError as exc:
        raise WeatherError(f"Geocoding request failed: {exc}") from exc

    seen: set[tuple[float, float]] = set()
    candidates: list[GeoCandidate] = []
    for r in raw:
        key = (round(r["latitude"], 4), round(r["longitude"], 4))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            GeoCandidate(
                name=r.get("name", location),
                latitude=r["latitude"],
                longitude=r["longitude"],
                country=r.get("country", ""),
                admin1=r.get("admin1", ""),
            )
        )
    return candidates[:cap]


def _fetch_weather_code(latitude: float, longitude: float, cfg: AppConfig) -> int:
    """Return the current WMO weather code for a coordinate."""
    try:
        with httpx.Client(timeout=15.0) as client:
            fc = client.get(
                cfg.weather.forecast_url,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current_weather": True,
                },
            )
            fc.raise_for_status()
            current = fc.json().get("current_weather")
    except httpx.HTTPError as exc:
        raise WeatherError(f"Weather API request failed: {exc}") from exc
    if not current:
        raise WeatherError(f"No current weather for ({latitude}, {longitude}).")
    return int(current["weathercode"])


def weather_for_candidate(
    candidate: GeoCandidate,
    config: AppConfig | None = None,
    *,
    location_query: str | None = None,
) -> WeatherResult:
    """Fetch the weather bucket for an already-resolved geocoding candidate.

    Use this after the user has disambiguated the location, so we never re-guess.

    Args:
        candidate: The chosen geocoding match.
        config: Optional config override.
        location_query: The original free-text query (for provenance).

    Returns:
        A :class:`WeatherResult` for that exact coordinate.
    """
    cfg = config or get_config()
    code = _fetch_weather_code(candidate.latitude, candidate.longitude, cfg)
    return WeatherResult(
        location_query=location_query or candidate.display,
        location_resolved=candidate.display,
        latitude=candidate.latitude,
        longitude=candidate.longitude,
        condition=_classify(code, cfg.weather.rainy_codes),
        weather_code=code,
    )


def get_weather(location: str, config: AppConfig | None = None) -> WeatherResult:
    """Resolve a free-text location and return its current weather bucket.

    Convenience path that takes the best geocoding match. The UI prefers
    :func:`geocode_candidates` + :func:`weather_for_candidate` so the user can
    disambiguate ambiguous names (e.g. Bangalore in India vs. Pakistan).

    Args:
        location: Free-text place (e.g. "London", "Tokyo").
        config: Optional config override (defaults to the app config).

    Returns:
        A :class:`WeatherResult` with the resolved place and sunny/rainy bucket.

    Raises:
        WeatherError: If the location cannot be resolved or the API fails.
    """
    cfg = config or get_config()
    candidates = geocode_candidates(location, cfg, limit=1)
    if not candidates:
        raise WeatherError(f"Could not resolve location: {location!r}")
    return weather_for_candidate(candidates[0], cfg, location_query=location)
