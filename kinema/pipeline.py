"""Render orchestrator.

Pairwise pipeline:
  1. Render each input image as a still mp4 clip of length D
  2. Walk the clips, xfading each into a growing "accumulator" mp4
  3. Mux audio with -shortest

Each ffmpeg invocation has at most 2 video inputs, so memory is bounded
no matter how many images the user provides. Slower than a single-shot
filter_complex (lots of re-encodes), but reliable inside the worker's
4GB / 2-CPU envelope.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from kinema.titles import ASPECT_DIMS, render_title_card
from kinema.transitions import TransitionSpec, sample_transition

logger = logging.getLogger(__name__)


@dataclass
class Recipe:
    name: str
    description: str
    transitions: list[dict]

    @classmethod
    def load(cls, path: Path) -> "Recipe":
        data = yaml.safe_load(path.read_text())
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            transitions=data["transitions"],
        )


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    return float(json.loads(out)["format"]["duration"])


def _check_inputs(sec_per_image: float, transition_seconds: float) -> None:
    if sec_per_image <= transition_seconds:
        raise ValueError("sec_per_image must exceed transition duration")


def _run_ffmpeg(cmd: list[str], *, label: str) -> None:
    """Run ffmpeg, surface stderr tail on failure (worker log_tail loses scroll)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-30:])
        raise RuntimeError(
            f"ffmpeg {label} failed (exit {proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr (tail): {tail}"
        )


def _render_still_clip(image: Path, duration: float, width: int, height: int, out: Path) -> None:
    """Image → fixed-length .mp4 with normalized aspect/fps/format."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,format=yuv420p,fps=30"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-t", f"{duration:.3f}", "-i", str(image),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    _run_ffmpeg(cmd, label=f"still_clip {image.name}")


def _xfade_pair(
    accum: Path, clip: Path, spec: TransitionSpec, offset: float, out: Path,
) -> None:
    """xfade `clip` onto the tail of `accum`. offset = accum_duration - spec.duration."""
    out.parent.mkdir(parents=True, exist_ok=True)
    filt = spec.filter_str("0:v", "1:v", offset)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(accum), "-i", str(clip),
        "-filter_complex", f"{filt}[vout]",
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    _run_ffmpeg(cmd, label=f"xfade {spec.name} → {out.name}")


def _mux_audio(video: Path, audio: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ]
    _run_ffmpeg(cmd, label="mux audio")


def run_pipeline(
    *,
    recipe_path: Path,
    image_paths: list[Path],
    audio_path: Path,
    out_path: Path,
    aspect: str = "16:9",
    sec_per_image: float = 1.5,
    title_text: str | None = None,
    seed: int | None = None,
    workdir: Path | None = None,
) -> Path:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")
    if aspect not in ASPECT_DIMS:
        raise ValueError(f"unsupported aspect: {aspect}")

    recipe = Recipe.load(recipe_path)
    rng = random.Random(seed)
    width, height = ASPECT_DIMS[aspect]

    workdir = workdir or out_path.parent / ".kinema-tmp"
    workdir.mkdir(parents=True, exist_ok=True)

    inputs: list[Path] = list(image_paths)
    if title_text:
        title_png = render_title_card(title_text, aspect, workdir / "title.png")
        inputs = [title_png, *inputs]

    if len(inputs) < 2:
        raise ValueError(f"need at least 2 images (incl. title card), got {len(inputs)}")

    audio_seconds = _ffprobe_duration(audio_path)
    plan_T = float((recipe.transitions[0].get("params") or {}).get("duration", 0.5))
    _check_inputs(sec_per_image, plan_T)

    logger.info(
        "rendering: recipe=%s images=%d audio=%.1fs aspect=%s sec/image=%.2f",
        recipe.name, len(inputs), audio_seconds, aspect, sec_per_image,
    )

    # 1) Render each image as a fixed-length clip.
    clips_dir = workdir / "clips"
    clips: list[Path] = []
    for i, img in enumerate(inputs):
        clip = clips_dir / f"clip_{i:04d}.mp4"
        _render_still_clip(img, sec_per_image, width, height, clip)
        clips.append(clip)
        if i % 20 == 0:
            logger.info("clips rendered: %d/%d", i + 1, len(inputs))

    # 2) Chain xfades pairwise — accumulator grows by (D - T) per step.
    accum = clips[0]
    accum_duration = sec_per_image
    accum_dir = workdir / "accum"
    for i in range(1, len(clips)):
        spec = sample_transition(recipe.transitions, rng)
        offset = max(0.0, accum_duration - spec.duration)
        next_accum = accum_dir / f"accum_{i:04d}.mp4"
        _xfade_pair(accum, clips[i], spec, offset, next_accum)
        # Free space: drop the previous accumulator (clips kept for debugging).
        if accum != clips[0]:
            try: accum.unlink()
            except OSError: pass
        accum = next_accum
        accum_duration = accum_duration + sec_per_image - spec.duration
        if i % 10 == 0:
            logger.info("xfades: %d/%d (accum %.1fs)", i, len(clips) - 1, accum_duration)

    # 3) Mux audio.
    _mux_audio(accum, audio_path, out_path)

    # Cleanup intermediates.
    for p in (workdir / "clips", workdir / "accum"):
        shutil.rmtree(p, ignore_errors=True)

    return out_path
