"""Tests for the weather service (Open-Meteo), with HTTP mocked."""

from __future__ import annotations

import httpx
import pytest

from brahma.exceptions import WeatherError
from brahma.models import WeatherCondition
from brahma.services import weather_api


def _mock_transport(geocode_results, forecast_code):
    """Build an httpx MockTransport for the two Open-Meteo calls."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding" in str(request.url):
            return httpx.Response(200, json={"results": geocode_results})
        return httpx.Response(
            200, json={"current_weather": {"weathercode": forecast_code}}
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def patch_client(monkeypatch):
    """Patch httpx.Client so weather_api uses our mock transport."""

    def _apply(geocode_results, forecast_code):
        transport = _mock_transport(geocode_results, forecast_code)
        real_client = httpx.Client

        def factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr(weather_api.httpx, "Client", factory)

    return _apply


def test_rainy_code_maps_to_rainy(patch_client):
    patch_client(
        [{"name": "Mumbai", "country": "India", "latitude": 19.0, "longitude": 72.8}],
        95,
    )
    result = weather_api.get_weather("Mumbai")
    assert result.condition is WeatherCondition.RAINY
    assert result.location_resolved == "Mumbai, India"


def test_clear_code_maps_to_sunny(patch_client):
    patch_client(
        [
            {
                "name": "Phoenix",
                "country": "United States",
                "latitude": 33.4,
                "longitude": -112.0,
            }
        ],
        0,
    )
    result = weather_api.get_weather("Phoenix")
    assert result.condition is WeatherCondition.SUNNY


def test_unresolvable_location_raises(patch_client):
    patch_client([], 0)
    with pytest.raises(WeatherError):
        weather_api.get_weather("Nowhereville XYZ")


def test_resolved_name_includes_admin1(patch_client):
    patch_client(
        [
            {
                "name": "Bengaluru",
                "admin1": "Karnataka",
                "country": "India",
                "latitude": 12.9,
                "longitude": 77.6,
            }
        ],
        0,
    )
    result = weather_api.get_weather("Bengaluru")
    assert result.location_resolved == "Bengaluru, Karnataka, India"


def test_geocode_candidates_dedups_by_coordinate(patch_client):
    dup = {"name": "X", "country": "Y", "latitude": 1.0, "longitude": 2.0}
    patch_client([dup, dict(dup)], 0)
    cands = weather_api.geocode_candidates("X")
    assert len(cands) == 1


def test_geocode_alias_is_searched(monkeypatch, config):
    """A known alias (bombay→Mumbai) is queried in addition to the raw name."""
    monkeypatch.setattr(config.weather, "geocode_aliases", {"bombay": "Mumbai"})
    queried: list[str] = []

    def fake_query(client, name, cfg):
        queried.append(name)
        return [
            {"name": name, "latitude": 1.0, "longitude": len(name), "country": "IN"}
        ]

    monkeypatch.setattr(weather_api, "_query_geocoder", fake_query)
    weather_api.geocode_candidates("Bombay", config)
    # Canonical alias searched first, then the raw query.
    assert queried[0] == "Mumbai"
    assert "Bombay" in queried


def test_comma_query_falls_back_to_leading_segment(monkeypatch, config):
    """A "City, State" query also tries the bare "City" (Open-Meteo is name-exact)."""
    monkeypatch.setattr(config.weather, "geocode_aliases", {})
    queried: list[str] = []

    def fake_query(client, name, cfg):
        queried.append(name)
        # Only the bare leading segment resolves, mimicking Open-Meteo.
        if name == "Udupi":
            return [
                {
                    "name": "Udupi",
                    "admin1": "Karnataka",
                    "country": "India",
                    "latitude": 13.3,
                    "longitude": 74.7,
                }
            ]
        return []

    monkeypatch.setattr(weather_api, "_query_geocoder", fake_query)
    cands = weather_api.geocode_candidates("Udupi, Karnataka", config)
    assert queried == ["Udupi, Karnataka", "Udupi"]
    assert cands and cands[0].display == "Udupi, Karnataka, India"


def test_weather_for_candidate_uses_coordinate(patch_client):
    from brahma.models import GeoCandidate

    patch_client([], 61)  # geocoding not called; forecast returns rainy code 61
    cand = GeoCandidate(
        name="Testville", latitude=10.0, longitude=20.0, country="Nowhere"
    )
    result = weather_api.weather_for_candidate(cand, location_query="q")
    assert result.condition is WeatherCondition.RAINY
    assert result.location_resolved == "Testville, Nowhere"
    assert result.location_query == "q"
