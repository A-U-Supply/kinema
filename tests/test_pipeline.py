"""End-to-end smoke test: render a short mp4 from generated fixtures.

Skipped on hosts without ffmpeg. The Docker image always satisfies it.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from PIL import Image

from kinema.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (320, 180), color=color).save(path, "PNG")


def _make_silent_wav(path: Path, seconds: float, sample_rate: int = 22050) -> None:
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))


@ffmpeg_required
def test_smooth_fade_renders_mp4(tmp_path: Path) -> None:
    images = []
    for i, color in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]):
        p = tmp_path / f"img{i}.png"
        _make_image(p, color)
        images.append(p)

    audio = tmp_path / "silence.wav"
    _make_silent_wav(audio, seconds=4.0)

    out = tmp_path / "out.mp4"
    run_pipeline(
        recipe_path=REPO_ROOT / "recipes" / "smooth-fade.yaml",
        image_paths=images,
        audio_path=audio,
        out_path=out,
        aspect="16:9",
        sec_per_image=1.0,
        title_text="Smoke Test",
        seed=42,
        workdir=tmp_path / "work",
    )

    assert out.exists() and out.stat().st_size > 1000

    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name",
        "-of", "json", str(out),
    ])
    info = json.loads(probe)
    streams = {s["codec_type"]: s["codec_name"] for s in info["streams"]}
    assert streams.get("video") == "h264"
    assert streams.get("audio") == "aac"
    assert float(info["format"]["duration"]) > 1.0
