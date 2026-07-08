"""Shared pytest fixtures.

Tests avoid live network / Vertex calls: the Gemini client is mocked and weather
HTTP is patched. Synthetic ffmpeg clips exercise the real compositor and media
helpers without needing the bundled assets.
"""

from __future__ import annotations

import subprocess

import pytest

from brahma.config import get_config


@pytest.fixture(scope="session")
def config():
    """The application config (loaded once)."""
    return get_config()


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> str:
    """A short base video (red, tone) for compositor/media tests."""
    path = tmp_path_factory.mktemp("media") / "base.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=8:r=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:duration=8",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return str(path)


@pytest.fixture(scope="session")
def synthetic_ad(tmp_path_factory) -> str:
    """A short ad clip (blue, different tone)."""
    path = tmp_path_factory.mktemp("media") / "ad.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=2:r=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=800:duration=2",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return str(path)


def _ffmpeg_available() -> bool:
    """Whether ffmpeg is on PATH (compositor/media tests need it)."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg not installed"
)
