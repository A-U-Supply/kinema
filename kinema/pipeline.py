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
        f"setsar=1,format=yuv420p,fps=24"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-t", f"{duration:.3f}", "-i", str(image),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    _run_ffmpeg(cmd, label=f"still_clip {image.name}")


def _render_chunk(
    clips: list[Path], specs: list[TransitionSpec], sec_per_image: float, out: Path,
) -> float:
    """Chain `clips` with `specs` xfades in a single filter_complex.
    Returns the chunk's video duration."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(clips)
    assert len(specs) == n - 1, f"need {n-1} transitions for {n} clips, got {len(specs)}"

    # Build inputs and filter graph. Each clip is already normalized; just
    # alias [{i}:v] → [v{i}] so the xfade chain is uniform.
    parts = [f"[{i}:v]null[v{i}]" for i in range(n)]
    cumulative_offset = 0.0
    prev = "v0"
    for i, spec in enumerate(specs):
        cumulative_offset += sec_per_image - spec.duration
        next_label = f"x{i+1}" if i < n - 2 else "vout"
        parts.append(f"{spec.filter_str(prev, f'v{i+1}', cumulative_offset)}[{next_label}]")
        prev = next_label
    if n == 1:
        parts.append("[v0]null[vout]")

    filter_complex = ";".join(parts)
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for c in clips:
        cmd += ["-i", str(c)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    _run_ffmpeg(cmd, label=f"chunk {out.name} ({n} clips)")

    # Chunk duration = sum(sec_per_image) - sum(transition_durations)
    return sec_per_image * n - sum(s.duration for s in specs)


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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
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
        # Texture the 3D block letters with a random source image.
        src_for_title = rng.choice(image_paths) if image_paths else None
        title_png = render_title_card(
            title_text, aspect, workdir / "title.png",
            source_image=src_for_title, seed=seed,
        )
        inputs = [title_png, *inputs]

    if len(inputs) < 2:
        raise ValueError(f"need at least 2 images (incl. title card), got {len(inputs)}")

    audio_seconds = _ffprobe_duration(audio_path)
    # Use the recipe's mean transition duration as the planning baseline.
    durs = [float((t.get("params") or {}).get("duration", 0.5)) for t in recipe.transitions]
    plan_T = sum(durs) / max(1, len(durs))
    _check_inputs(sec_per_image, plan_T)

    # Cycle inputs to fill the audio length so the whole track plays.
    # Pairwise rendering keeps memory bounded so this is safe even at N=200+.
    target_images = max(2, int((audio_seconds + plan_T) / max(0.01, sec_per_image - plan_T)) + 1)
    if len(inputs) < target_images:
        cycled: list[Path] = []
        i = 0
        while len(cycled) < target_images:
            cycled.append(inputs[i % len(inputs)])
            i += 1
        inputs = cycled

    logger.info(
        "rendering: recipe=%s images=%d audio=%.1fs aspect=%s sec/image=%.2f plan_T=%.2f",
        recipe.name, len(inputs), audio_seconds, aspect, sec_per_image, plan_T,
    )

    # 1) Render each input image into a still mp4 clip.
    clips_dir = workdir / "clips"
    clips: list[Path] = []
    for i, img in enumerate(inputs):
        clip = clips_dir / f"clip_{i:04d}.mp4"
        _render_still_clip(img, sec_per_image, width, height, clip)
        clips.append(clip)
        if (i + 1) % 20 == 0:
            logger.info("clips rendered: %d/%d", i + 1, len(inputs))

    # 2) Chunked xfade chaining: render CHUNK_SIZE clips at a time in one
    # filter_complex (small N, OK for memory), then chain the chunk videos
    # pairwise. Drastically fewer ffmpeg invocations than pure pairwise.
    CHUNK_SIZE = 6
    chunks: list[tuple[Path, float]] = []  # (path, duration)
    chunk_dir = workdir / "chunks"
    for ci in range(0, len(clips), CHUNK_SIZE):
        group = clips[ci : ci + CHUNK_SIZE]
        chunk_path = chunk_dir / f"chunk_{ci // CHUNK_SIZE:04d}.mp4"
        if len(group) == 1:
            shutil.copy2(group[0], chunk_path)
            chunk_dur = sec_per_image
        else:
            chunk_specs = [sample_transition(recipe.transitions, rng) for _ in range(len(group) - 1)]
            chunk_dur = _render_chunk(group, chunk_specs, sec_per_image, chunk_path)
        chunks.append((chunk_path, chunk_dur))
        logger.info("chunk %d/%d → %.1fs", ci // CHUNK_SIZE + 1, (len(clips) + CHUNK_SIZE - 1) // CHUNK_SIZE, chunk_dur)

    # 3) Chain chunks pairwise with one xfade between each adjacent pair.
    accum, accum_duration = chunks[0]
    accum_dir = workdir / "accum"
    for i in range(1, len(chunks)):
        next_clip, next_dur = chunks[i]
        spec = sample_transition(recipe.transitions, rng)
        offset = max(0.0, accum_duration - spec.duration)
        next_accum = accum_dir / f"accum_{i:04d}.mp4"
        _xfade_pair(accum, next_clip, spec, offset, next_accum)
        if accum != chunks[0][0]:
            try: accum.unlink()
            except OSError: pass
        accum = next_accum
        accum_duration = accum_duration + next_dur - spec.duration
        logger.info("inter-chunk xfade %d/%d (accum %.1fs)", i, len(chunks) - 1, accum_duration)

    # 4) Mux audio.
    _mux_audio(accum, audio_path, out_path)

    # Cleanup intermediates.
    for p in (workdir / "clips", workdir / "chunks", workdir / "accum"):
        shutil.rmtree(p, ignore_errors=True)

    return out_path
