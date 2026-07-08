"""Thin Gemini client backed by Vertex AI + a GCP service account.

Auth uses only the service-account JSON (path from env); ``project_id`` is read
from that file, so no secret values are ever hard-coded or committed. The client
is cached so the whole app shares one authenticated instance.
"""

from __future__ import annotations

import functools
import json
import warnings
from pathlib import Path
from typing import Any, Optional

# google-genai 0.8.0 emits a benign Pydantic warning from its own models; it is
# not actionable on our side, so silence just that message.
warnings.filterwarnings(
    "ignore",
    message="<built-in function any> is not a Python type.*",
    category=UserWarning,
)

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from google.oauth2 import service_account  # noqa: E402

from brahma.config import AppConfig, get_config  # noqa: E402

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class GeminiClient:
    """Wrapper over the google-genai Vertex client with text + vision helpers."""

    def __init__(self, config: AppConfig) -> None:
        """Initialise the Vertex-backed client from service-account creds.

        Args:
            config: The application configuration.

        Raises:
            FileNotFoundError: If the credentials file is missing.
        """
        creds_path = Path(config.creds_path)
        if not creds_path.is_file():
            raise FileNotFoundError(
                f"Service-account creds not found: {creds_path}. "
                "Copy configs/creds.example.json and fill it, or set "
                "GOOGLE_APPLICATION_CREDENTIALS."
            )

        with creds_path.open("r", encoding="utf-8") as fh:
            project_id = json.load(fh)["project_id"]

        credentials = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            str(creds_path), scopes=_SCOPES
        )

        self._config = config
        self._client = genai.Client(
            vertexai=True,
            project=project_id,
            location=config.gemini.location,
            credentials=credentials,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate a text response.

        Args:
            prompt: The user prompt.
            model: Model id override; defaults to the orchestrator model.
            temperature: Sampling temperature.
            system_instruction: Optional system prompt.

        Returns:
            The model's text output (stripped).
        """
        response = self._client.models.generate_content(
            model=model or self._config.gemini.orchestrator_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
            ),
        )
        return (response.text or "").strip()

    def generate_vision(
        self,
        prompt: str,
        images: list[bytes],
        *,
        mime_type: str = "image/jpeg",
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response conditioned on one or more images.

        Args:
            prompt: The instruction/question about the images.
            images: Raw image bytes, in order.
            mime_type: MIME type of the images.
            model: Model id override; defaults to the vision model.
            temperature: Sampling temperature.

        Returns:
            The model's text output (stripped).
        """
        parts: list[types.Part] = [
            types.Part.from_bytes(data=img, mime_type=mime_type) for img in images
        ]
        parts.append(types.Part.from_text(text=prompt))
        response = self._client.models.generate_content(
            model=model or self._config.gemini.vision_model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return (response.text or "").strip()


@functools.lru_cache(maxsize=1)
def get_gemini_client() -> GeminiClient:
    """Return the shared, lazily-constructed Gemini client."""
    return GeminiClient(get_config())


# Fields a GCP service-account JSON must contain to be usable for Vertex auth.
REQUIRED_CRED_FIELDS = ("type", "project_id", "private_key", "client_email")


def validate_credentials(
    config: AppConfig | None = None, *, ping: bool = False
) -> tuple[bool, str]:
    """Validate the service-account credentials without raising.

    Checks, in order: file exists → valid JSON → required fields present →
    (optionally) a live Vertex call succeeds. Intended for the UI's creds gate.

    Args:
        config: Optional config override.
        ping: If ``True``, make a tiny live Gemini call to confirm the project
            has Vertex AI enabled and the key works end-to-end.

    Returns:
        ``(ok, message)`` — ``ok`` is ``False`` with a human-readable reason if
        the credentials are missing/invalid/unusable.
    """
    cfg = config or get_config()
    path = Path(cfg.creds_path)
    if not path.is_file():
        return False, f"Credentials file not found at {path}."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Credentials file is not valid JSON: {exc}"

    missing = [f for f in REQUIRED_CRED_FIELDS if not data.get(f)]
    if missing:
        return False, f"Credentials file is missing fields: {', '.join(missing)}."

    if not ping:
        return True, f"Credentials look valid (project '{data['project_id']}')."

    try:
        get_gemini_client.cache_clear()
        client = get_gemini_client()
        client.generate_text("ping", temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - surface any auth/enablement error
        return False, f"Vertex AI call failed (check API enablement / key): {exc}"
    return True, f"Vertex AI reachable (project '{data['project_id']}')."


def parse_json_response(text: str) -> Any:
    """Parse a JSON object from an LLM response, tolerating markdown fences.

    Args:
        text: Raw model output that should contain a JSON value.

    Returns:
        The parsed JSON value.

    Raises:
        json.JSONDecodeError: If the text is not valid JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())
