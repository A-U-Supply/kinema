"""Render orchestrator.

Given a recipe, image paths, audio path, and target aspect/duration, build an
ffmpeg command that:
  1. ingests each image as a still video segment of length D
  2. chains xfade transitions sampled from the recipe's pool, with each
     transition's `offset` accumulating along the timeline
  3. muxes the audio, trimming to the shorter of audio/video
  4. encodes to H.264 + AAC mp4
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from kinema.titles import ASPECT_DIMS, render_title_card
from kinema.transitions import sample_transition

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


# Hard cap to keep the filter_complex graph + decoder allocations within the
# worker's 4GB / 2-CPU envelope. Real-world: 117 simultaneous image inputs
# OOM-killed ffmpeg silently. ~40 has comfortable headroom.
_MAX_IMAGES_IN_GRAPH = 40


def _check_inputs(sec_per_image: float, transition_seconds: float) -> None:
    if sec_per_image <= transition_seconds:
        raise ValueError("sec_per_image must exceed transition duration")


def _build_filter_graph(
    n_images: int,
    width: int,
    height: int,
    sec_per_image: float,
    recipe: Recipe,
    rng: random.Random,
) -> tuple[str, str]:
    """Returns (filter_complex_str, final_label)."""
    parts = []
    # Normalize each input to consistent format/size/sar/fps.
    for i in range(n_images):
        parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,format=yuv420p,fps=30[v{i}]"
        )

    # Chain xfades. After fading v0+v1 the result is "x1"; then x1+v2 → "x2"; etc.
    prev_label = "v0"
    cumulative_offset = 0.0
    for i in range(n_images - 1):
        spec = sample_transition(recipe.transitions, rng)
        # Each segment plays for sec_per_image; transitions overlap by spec.duration.
        # offset_i = (i+1) * (sec_per_image - spec.duration)
        cumulative_offset += sec_per_image - spec.duration
        next_label = f"x{i+1}" if i < n_images - 2 else "vout"
        snippet = spec.filter_str(prev_label, f"v{i+1}", cumulative_offset)
        parts.append(f"{snippet}[{next_label}]")
        prev_label = next_label

    # Edge case: only 2 images → final label is vout from the single xfade above.
    # Edge case: only 1 image → no transition; alias v0 → vout.
    if n_images == 1:
        # `null` is the pass-through video filter; `copy` is for stream copy
        # outside of filter graphs and crashes here.
        parts.append("[v0]null[vout]")

    return ";".join(parts), "vout"


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

    # Title card prepended as image 0.
    inputs: list[Path] = list(image_paths)
    if title_text:
        title_png = render_title_card(title_text, aspect, workdir / "title.png")
        inputs = [title_png, *inputs]

    audio_seconds = _ffprobe_duration(audio_path)
    plan_T = float((recipe.transitions[0].get("params") or {}).get("duration", 0.5))
    _check_inputs(sec_per_image, plan_T)

    # Use exactly what was provided; cap for ffmpeg sanity. The user controls
    # video length via image_count (or by picking more images). Audio is
    # trimmed/clipped to whichever side ends first via -shortest.
    if len(inputs) > _MAX_IMAGES_IN_GRAPH:
        logger.warning("trimming %d images to %d to fit ffmpeg envelope",
                       len(inputs), _MAX_IMAGES_IN_GRAPH)
        images = inputs[:_MAX_IMAGES_IN_GRAPH]
    else:
        images = inputs
    if len(images) < 2:
        raise ValueError(f"need at least 2 images, got {len(images)}")

    video_seconds = len(images) * sec_per_image - (len(images) - 1) * plan_T
    logger.info(
        "rendering: recipe=%s images=%d → %.1fs video, %.1fs audio, aspect=%s",
        recipe.name, len(images), video_seconds, audio_seconds, aspect,
    )

    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    for img in images:
        cmd += ["-loop", "1", "-t", f"{sec_per_image:.3f}", "-i", str(img)]
    cmd += ["-i", str(audio_path)]

    filter_graph, final_label = _build_filter_graph(
        len(images), width, height, sec_per_image, recipe, rng
    )

    cmd += [
        "-filter_complex", filter_graph,
        "-map", f"[{final_label}]",
        "-map", f"{len(images)}:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]

    logger.info("ffmpeg cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Surface ffmpeg's last 60 lines so worker log_tail shows the real cause.
        tail = "\n".join((proc.stderr or "").splitlines()[-60:])
        raise RuntimeError(
            f"ffmpeg failed with exit {proc.returncode}\n"
            f"--- filter_complex ---\n{filter_graph}\n"
            f"--- ffmpeg stderr (tail) ---\n{tail}"
        )
    return out_path
