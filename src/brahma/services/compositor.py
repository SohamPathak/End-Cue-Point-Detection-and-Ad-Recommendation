"""ffmpeg compositor — burns an ad full-screen into a video's outro.

Timeline (given ``outro_start`` and ``ad_start_delay_sec``)::

    ad_appear = outro_start + delay
    [0 .............. ad_appear)      original content + first seconds of credits
    [ad_appear ...... ad_appear+ad)   AD, full-screen, with the ad's own audio
    [ad_appear+ad ... video_end)      remaining original credits (if any), restored

Cases:
  * **Short ad** (ends before the video does): after the ad, crossfade back to
    the remaining credits with their original audio.
  * **Long ad** (would overrun the video end): let the ad play to completion; the
    output simply extends past the original end. No trimming.

Output is cached by ``(video_hash, ad_id)`` so switching locations that map to an
already-rendered ad reuses the file instead of re-encoding.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from brahma.config import AppConfig, get_config
from brahma.exceptions import CompositionError
from brahma.services.media_utils import (
    has_audio_stream,
    probe_duration,
    probe_video_props,
    video_hash,
)


def _output_path(config: AppConfig, vhash: str, ad_id: str) -> Path:
    """Return the cached output path for a (video, ad) pair."""
    out_dir = config.abs_path(config.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"composited_{vhash}_{ad_id}.mp4"


def composite_ad(
    video_path: str,
    ad_path: str,
    outro_start_sec: float,
    ad_id: str,
    config: AppConfig | None = None,
    *,
    force: bool = False,
) -> str:
    """Burn the ad into the video's outro and return the output path.

    Args:
        video_path: Source video.
        ad_path: Ad clip to burn in.
        outro_start_sec: Detected outro cuepoint (seconds).
        ad_id: Catalog id of the ad (used for the cache key).
        config: Optional config override.
        force: If ``True``, re-render even if a cached output exists.

    Returns:
        Path to the composited MP4.

    Raises:
        CompositionError: If ffmpeg fails.
    """
    cfg = config or get_config()
    ccfg = cfg.compositor
    vhash = video_hash(video_path)
    out_path = _output_path(cfg, vhash, ad_id)

    if not force and out_path.is_file():
        return str(out_path)

    props = probe_video_props(video_path)
    width, height, fps = int(props["width"]), int(props["height"]), props["fps"]
    video_dur = probe_duration(video_path)
    ad_dur = probe_duration(ad_path)

    ad_appear = min(outro_start_sec + ccfg.ad_start_delay_sec, video_dur)
    fade = ccfg.fade_duration_sec
    tail_start = ad_appear + ad_dur  # where remaining credits would resume
    has_tail = tail_start < video_dur - fade  # enough credits left to fade back to

    # Normalise every branch to identical geometry / fps / timebase / sample
    # format so concat and xfade never hit a mismatch (xfade in particular
    # rejects differing input timebases).
    vnorm = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"fps={fps},format=yuv420p,settb=AVTB"
    )
    ad_scale = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"fps={fps},format=yuv420p,settb=AVTB"
    )
    anorm = "aformat=sample_rates=44100:channel_layouts=stereo"
    ad_has_audio = has_audio_stream(ad_path)
    base_has_audio = has_audio_stream(video_path)

    filt: list[str] = []
    # [pre]  = original content up to ad_appear (video + audio)
    filt.append(f"[0:v]trim=0:{ad_appear:.3f},setpts=PTS-STARTPTS,{vnorm}[prev]")
    if base_has_audio:
        filt.append(f"[0:a]atrim=0:{ad_appear:.3f},asetpts=PTS-STARTPTS,{anorm}[prea]")
    else:
        # Silent base video (e.g. a screen recording) → synthesise silence so the
        # audio concat stays balanced with the video.
        filt.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{ad_appear:.3f}[prea]")
    # [adv]/[ada] = normalised ad
    filt.append(f"[1:v]{ad_scale}[adv]")
    if ad_has_audio:
        filt.append(f"[1:a]asetpts=PTS-STARTPTS,{anorm}[ada]")
    else:
        # Silent ad → synthesise silence for the concat to stay a/v balanced.
        filt.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{ad_dur:.3f}[ada]")

    if has_tail:
        # [post] = remaining credits after the ad, with a crossfade back.
        filt.append(
            f"[0:v]trim={tail_start:.3f}:{video_dur:.3f},setpts=PTS-STARTPTS,{vnorm}[postv]"
        )
        tail_dur = video_dur - tail_start
        if base_has_audio:
            filt.append(
                f"[0:a]atrim={tail_start:.3f}:{video_dur:.3f},"
                f"asetpts=PTS-STARTPTS,{anorm}[posta]"
            )
        else:
            filt.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{tail_dur:.3f}[posta]")
        # Concat pre+ad, then xfade into the credits tail.
        filt.append("[prev][adv]concat=n=2:v=1:a=0[v01]")
        filt.append("[prea][ada]concat=n=2:v=0:a=1[a01]")
        xfade_offset = ad_appear + ad_dur - fade
        filt.append(
            f"[v01][postv]xfade=transition=fade:duration={fade:.3f}:"
            f"offset={xfade_offset:.3f}[vout]"
        )
        filt.append(f"[a01][posta]acrossfade=d={fade:.3f}[aout]")
    else:
        # Long ad (or ad reaches the end): pre + ad, no tail.
        filt.append("[prev][adv]concat=n=2:v=1:a=0[vout]")
        filt.append("[prea][ada]concat=n=2:v=0:a=1[aout]")

    filter_complex = ";".join(filt)
    tmp_out = out_path.with_suffix(".mp4.tmp")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(ad_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        ccfg.video_codec,
        "-crf",
        str(ccfg.crf),
        "-c:a",
        ccfg.audio_codec,
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(tmp_out),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise CompositionError(
            f"ffmpeg composition failed: {exc.stderr.strip()[-800:]}"
        ) from exc

    tmp_out.replace(out_path)
    return str(out_path)
