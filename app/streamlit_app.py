"""Minimalistic Streamlit UI for the Brahma AI end-cue-point ad-insertion pipeline.

Flow mirrors the two-phase design:
  * A **credentials gate** validates the Vertex service account up front and, if
    it is missing/invalid, lets the user paste or upload one before proceeding.
  * Selecting/uploading a video runs **Phase 1** (outro detection) once and
    caches the result — changing the location never re-detects the outro.
  * A location is resolved to explicit candidates so the user **disambiguates**
    (e.g. Bangalore, India vs. Bangalore Town, Pakistan) — never a silent guess.
  * Submitting runs **Phase 2** (weather → recommend → composite) and shows the
    original vs. ad-baked video side by side.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import streamlit as st

from brahma.clients.gemini import get_gemini_client, validate_credentials
from brahma.config import get_config
from brahma.exceptions import BrahmaError, WeatherError
from brahma.models import GeoCandidate, OutroResult
from brahma.services.weather_api import geocode_candidates

if TYPE_CHECKING:
    from brahma.agent.orchestrator import Orchestrator


def _logo_path() -> str | None:
    """Return the brand logo path if it exists, else None."""
    cfg = get_config()
    path = cfg.abs_path(cfg.paths.logo)
    return str(path) if path.is_file() else None


st.set_page_config(
    page_title="Brahma AI — Weather-Aware End Cue Point Ads",
    page_icon=_logo_path() or "🎬",
    layout="wide",
)

# Brand-neutral surface/ink tokens (light + dark), applied to the custom panels.
_STYLES = """
<style>
:root {
  --brahma-surface: #fcfcfb; --brahma-ink: #0b0b0b; --brahma-ink-2: #52514e;
  --brahma-border: rgba(11,11,11,0.10); --brahma-accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --brahma-surface: #1a1a19; --brahma-ink: #ffffff; --brahma-ink-2: #c3c2b7;
    --brahma-border: rgba(255,255,255,0.12); --brahma-accent: #3987e5;
  }
}
.brahma-panel {
  border: 1px solid var(--brahma-border); border-radius: 12px;
  background: var(--brahma-surface); padding: 16px 18px; margin: 8px 0 4px;
}
.brahma-panel-title {
  font-size: 0.8rem; font-weight: 600; color: var(--brahma-ink-2);
  text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 12px;
}
.brahma-stat-row { display: flex; flex-wrap: wrap; gap: 12px; }
.brahma-stat {
  flex: 1 1 140px; min-width: 130px; border: 1px solid var(--brahma-border);
  border-radius: 10px; padding: 12px 14px; background: var(--brahma-surface);
}
.brahma-stat-label {
  font-size: 0.72rem; color: var(--brahma-ink-2); margin-bottom: 4px;
  text-transform: uppercase; letter-spacing: 0.03em;
}
.brahma-stat-value {
  font-size: 1.15rem; font-weight: 700; color: var(--brahma-ink);
  line-height: 1.25; word-break: break-word;   /* never truncate (e.g. method id) */
}
</style>
"""


@st.cache_resource
def _flush_caches_on_start() -> bool:
    """Clear the on-disk detection/composite cache once per server process.

    ``@st.cache_resource`` runs this exactly once when the server boots (not on
    every rerun), so each fresh launch starts with a clean cache while still
    reusing results within a running session.
    """
    cfg = get_config()
    for sub in (cfg.paths.cache_dir, cfg.paths.output_dir):
        target = cfg.abs_path(sub)
        if target.is_dir():
            for item in target.glob("*"):
                if item.is_file() and item.suffix in {".json", ".mp4"}:
                    item.unlink(missing_ok=True)
    return True


@st.cache_resource
def _orchestrator() -> "Orchestrator":
    """Return a process-wide orchestrator (Vertex client is initialised once)."""
    # Lazy import so the app still renders the creds gate if Vertex is misconfigured.
    from brahma.agent.orchestrator import Orchestrator

    return Orchestrator()


def _list_sample_videos() -> list[Path]:
    """List bundled sample videos."""
    cfg = get_config()
    sample_dir = cfg.abs_path(cfg.paths.sample_video_dir)
    if not sample_dir.is_dir():
        return []
    return sorted(p for p in sample_dir.iterdir() if p.suffix.lower() == ".mp4")


def _persist_upload(uploaded: object) -> str:
    """Save an uploaded video to a stable temp path and return it.

    Streamlit re-runs the whole script on every interaction, so this must be
    idempotent per upload — otherwise each rerun would write a new random file,
    changing ``video_path`` and invalidating the cached end-cue-point detection.
    We key the temp file by the upload's stable ``file_id`` and reuse it.
    """
    file_id = getattr(uploaded, "file_id", None) or getattr(uploaded, "name", "upload")
    suffix = Path(uploaded.name).suffix or ".mp4"  # type: ignore[attr-defined]
    dest = Path(tempfile.gettempdir()) / f"brahma_upload_{file_id}{suffix}"
    if not dest.exists():
        with dest.open("wb") as fh:
            shutil.copyfileobj(uploaded, fh)  # type: ignore[arg-type]
    return str(dest)


def _reset_video_state() -> None:
    """Clear per-video derived state when the source video changes."""
    for key in ("outro", "video_path", "result"):
        st.session_state.pop(key, None)


def _credentials_gate() -> bool:
    """Validate Vertex credentials; if invalid, render a fill-in form.

    Returns:
        ``True`` when credentials are valid and the app may proceed.
    """
    ok, message = validate_credentials()
    if ok:
        return True

    st.error(f"🔑 Gemini / Vertex AI credentials required — {message}")
    st.markdown(
        "Provide a **GCP service-account JSON** (with the **Vertex AI API "
        "enabled**). Only the file is needed; nothing is committed."
    )
    cfg = get_config()
    dest = Path(cfg.creds_path)

    tab_upload, tab_paste = st.tabs(["Upload JSON file", "Paste JSON"])
    with tab_upload:
        up = st.file_uploader("Service-account JSON", type=["json"], key="creds_up")
        if up is not None and st.button("Save uploaded credentials"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(up.getvalue())
            _revalidate_and_rerun()
    with tab_paste:
        pasted = st.text_area("Paste the JSON contents", height=200, key="creds_paste")
        if pasted and st.button("Save pasted credentials"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(pasted, encoding="utf-8")
            _revalidate_and_rerun()

    st.caption(f"Credentials are read from: `{dest}` (configurable via .env).")
    return False


def _revalidate_and_rerun() -> None:
    """Clear caches so the new creds are picked up, then rerun."""
    get_gemini_client.cache_clear()
    _orchestrator.clear()
    ok, message = validate_credentials()
    if ok:
        st.success(message)
    else:
        st.error(message)
    st.rerun()


class _SidebarInputs(NamedTuple):
    """User inputs gathered from the sidebar."""

    location: str
    geo: GeoCandidate | None
    redetect: bool
    run_clicked: bool


def _render_sidebar() -> _SidebarInputs:
    """Render the sidebar controls and return the user's inputs."""
    with st.sidebar:
        st.header("1 · Choose a video")
        samples = _list_sample_videos()
        sample_names = ["— upload my own —"] + [p.name for p in samples]
        choice = st.selectbox("Sample video", sample_names, index=1 if samples else 0)

        video_path: str | None = None
        if choice == "— upload my own —":
            uploaded = st.file_uploader("Upload an MP4", type=["mp4", "mov", "m4v"])
            if uploaded is not None:
                video_path = _persist_upload(uploaded)
        else:
            video_path = str(next(p for p in samples if p.name == choice))

        if video_path and st.session_state.get("video_path") != video_path:
            _reset_video_state()
            st.session_state["video_path"] = video_path

        # Detection is a video operation, so its control lives with the video
        # section — only meaningful once a video is chosen.
        redetect = st.button(
            "Detect end cue point",
            help="Detect (or re-detect) the end cue point for the selected video",
            disabled=video_path is None,
        )

        st.divider()
        st.header("2 · Enter a location")
        location = st.text_input("Anywhere in the world", value="London")
        geo = _resolve_location(location)
        run_clicked = st.button(
            "Generate ad-baked video", type="primary", disabled=geo is None
        )

    return _SidebarInputs(
        location=location, geo=geo, redetect=redetect, run_clicked=run_clicked
    )


def main() -> None:
    """Render the app."""
    _flush_caches_on_start()  # runs once per server process
    st.markdown(_STYLES, unsafe_allow_html=True)
    logo = _logo_path()
    if logo:
        st.image(logo, width=280)
    else:
        st.title("🎬 Brahma AI")
    st.caption(
        "Weather-Aware End Cue Point Ads — detect a video's end cue point, read "
        "the weather for any location, and burn the matching ad into the credits."
    )

    if not _credentials_gate():
        return

    inp = _render_sidebar()
    location, geo, redetect, run_clicked = (
        inp.location,
        inp.geo,
        inp.redetect,
        inp.run_clicked,
    )

    if not st.session_state.get("video_path"):
        st.info("👈 Pick a sample video or upload one to begin.")
        return

    video_path = st.session_state["video_path"]
    orch = _orchestrator()

    # ---- Phase 1: end-cue-point detection ------------------------------------
    # Only run when the user asks: the "Detect" button, or implicitly when they
    # click "Generate" without a cached result. Never auto-detect on page load.
    need_detect = redetect or (run_clicked and "outro" not in st.session_state)
    if need_detect:
        with st.spinner("Detecting end cue point (once per video)…"):
            try:
                detected = orch.prepare_video(video_path, force=bool(redetect))
                st.session_state["outro"] = detected.model_dump()
            except BrahmaError as exc:
                st.error(f"End cue point detection failed: {exc}")
                return

    if "outro" not in st.session_state:
        st.info(
            "Click **Detect end cue point** to analyse the selected video, or "
            "**Generate** to detect and composite in one step."
        )
        return

    outro = OutroResult.model_validate(st.session_state["outro"])
    _render_outro_banner(outro)

    # ---- Phase 2: weather → recommend → composite -----------------------------
    if run_clicked and geo is not None:
        try:
            st.session_state["result"] = _run_with_progress(
                orch, video_path, location, geo, outro
            )
        except BrahmaError as exc:
            st.error(f"Pipeline failed: {exc}")
            return

    _render_result(video_path, st.session_state.get("result"))


def _run_with_progress(
    orch: Orchestrator,
    video_path: str,
    location: str,
    geo: GeoCandidate,
    outro: OutroResult,
) -> dict:
    """Run Phase 2 with a visible progress bar.

    Compositing re-encodes the video (tens of seconds), so we surface staged
    progress instead of an indefinite spinner. The heavy work happens inside the
    single ``run`` call; the bar frames it with clear before/after states.
    """
    bar = st.progress(10, text=f"Reading weather for “{geo.display}”…")
    bar.progress(
        60, text="Choosing the ad and burning it into the end cue point (encoding)…"
    )
    result = orch.run(video_path, location, geo=geo, outro=outro).model_dump()
    bar.progress(100, text="Done.")
    bar.empty()
    return result


def _resolve_location(location: str) -> GeoCandidate | None:
    """Fetch geocoding candidates and let the user disambiguate.

    Args:
        location: Free-text location query.

    Returns:
        The chosen :class:`GeoCandidate`, or ``None`` if unresolved.
    """
    if not location.strip():
        return None
    try:
        candidates = geocode_candidates(location)
    except WeatherError as exc:
        st.warning(f"Could not look up location: {exc}")
        return None
    if not candidates:
        st.warning(f"No places found for “{location}”. Try a more specific name.")
        return None

    labels = [c.display for c in candidates]
    if len(candidates) > 1:
        st.caption("Multiple matches — pick the intended place:")
    idx = st.radio(
        "Matches",
        options=range(len(candidates)),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
    )
    return candidates[idx]


def _fmt_mmss(seconds: float) -> str:
    """Format seconds as ``m:ss`` (e.g. 560.6 → "9:20")."""
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _stat_card(label: str, value: str) -> str:
    """Return HTML for one bounded stat tile (value wraps, never truncates)."""
    return (
        '<div class="brahma-stat">'
        f'<div class="brahma-stat-label">{label}</div>'
        f'<div class="brahma-stat-value">{value}</div>'
        "</div>"
    )


def _render_outro_banner(outro: OutroResult) -> None:
    """Show the detected end cue point inside a bounded, styled card row.

    Handles the no-cue-point case (e.g. a screen recording): the ad is appended
    at the end rather than burned over content, and the panel says so.
    """
    conf_pct = f"{outro.confidence:.0%}"
    length = _fmt_mmss(outro.video_duration_sec)
    if outro.has_outro:
        stats = (
            ("End cue point at", _fmt_mmss(outro.start_sec)),
            ("Video length", length),
            ("Confidence", conf_pct),
            ("Method", outro.method),
        )
    else:
        stats = (
            ("End cue point", "None detected"),
            ("Ad placement", "Appended at end"),
            ("Video length", length),
            ("Confidence", conf_pct),
        )
    cards = "".join(_stat_card(label, value) for label, value in stats)
    st.markdown(
        '<div class="brahma-panel">'
        '<div class="brahma-panel-title">Detected end cue point</div>'
        f'<div class="brahma-stat-row">{cards}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if not outro.has_outro:
        st.info(
            "No end cue point (outro/credits) was found in this video, so the ad "
            "will be **appended at the end** instead of overlaid on real content."
        )


def _render_result(video_path: str, result: dict | None) -> None:
    """Render the side-by-side original vs. ad-baked output and the rationale."""
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.video(video_path)
    with right:
        st.subheader("Ad baked into end cue point")
        if result:
            st.video(result["output_video_path"])
        else:
            st.info("Choose a location and click **Generate** to see the output.")

    if not result:
        return

    st.divider()
    st.subheader("Why this ad?")
    st.write(result["reasoning"])

    weather = result["weather"]
    rec = result["recommendation"]
    cards = "".join(
        _stat_card(label, value)
        for label, value in (
            ("Location", weather["location_resolved"]),
            ("Weather", weather["condition"].title()),
            ("Ad", rec["ad_id"].title()),
        )
    )
    st.markdown(
        '<div class="brahma-panel">'
        '<div class="brahma-panel-title">Decision</div>'
        f'<div class="brahma-stat-row">{cards}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Decision detail (features, scores, experiment)"):
        st.json(
            {
                "experiment_id": rec["experiment_id"],
                "strategy": rec["strategy"],
                "scores": rec["scores"],
                "weather_code": weather["weather_code"],
            }
        )


if __name__ == "__main__":
    main()
