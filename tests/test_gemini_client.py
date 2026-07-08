"""Tests for the Gemini client construction and helpers (SDK mocked)."""

from __future__ import annotations

import pytest

from brahma.clients import gemini


def test_missing_creds_raises(config, monkeypatch):
    monkeypatch.setattr(config, "creds_path", "/nonexistent/creds.json")
    with pytest.raises(FileNotFoundError):
        gemini.GeminiClient(config)


def test_validate_credentials_missing_file(config, monkeypatch):
    monkeypatch.setattr(config, "creds_path", "/nope/creds.json")
    ok, msg = gemini.validate_credentials(config)
    assert ok is False
    assert "not found" in msg


def test_validate_credentials_invalid_json(config, monkeypatch, tmp_path):
    bad = tmp_path / "creds.json"
    bad.write_text("{not json")
    monkeypatch.setattr(config, "creds_path", str(bad))
    ok, msg = gemini.validate_credentials(config)
    assert ok is False
    assert "not valid JSON" in msg


def test_validate_credentials_missing_fields(config, monkeypatch, tmp_path):
    partial = tmp_path / "creds.json"
    partial.write_text('{"type": "service_account"}')
    monkeypatch.setattr(config, "creds_path", str(partial))
    ok, msg = gemini.validate_credentials(config)
    assert ok is False
    assert "missing fields" in msg


def test_validate_credentials_ok_without_ping(config, monkeypatch, tmp_path):
    good = tmp_path / "creds.json"
    good.write_text(
        '{"type": "service_account", "project_id": "p", '
        '"private_key": "k", "client_email": "e@x.iam"}'
    )
    monkeypatch.setattr(config, "creds_path", str(good))
    ok, msg = gemini.validate_credentials(config)
    assert ok is True
    assert "'p'" in msg


def test_client_reads_project_and_calls_sdk(config, monkeypatch, tmp_path):
    """Client init reads project_id from creds and builds a Vertex genai client."""
    creds = tmp_path / "creds.json"
    creds.write_text('{"project_id": "proj-123", "type": "service_account"}')
    monkeypatch.setattr(config, "creds_path", str(creds))

    captured = {}

    class FakeModels:
        def generate_content(self, model, contents, config=None):  # noqa: A002
            captured["model"] = model
            return type("R", (), {"text": "  hello  "})()

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = FakeModels()

    monkeypatch.setattr(gemini.genai, "Client", FakeClient)
    monkeypatch.setattr(
        gemini.service_account.Credentials,
        "from_service_account_file",
        staticmethod(lambda path, scopes: "FAKE_CREDS"),
    )

    client = gemini.GeminiClient(config)
    assert captured["project"] == "proj-123"
    assert captured["vertexai"] is True

    out = client.generate_text("hi")
    assert out == "hello"  # stripped
    assert captured["model"] == config.gemini.orchestrator_model
