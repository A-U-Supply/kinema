"""Title-card rendering.

Default style: 3D isometric block letters textured with one of the user's
source images, via the vendored protease-bots `render_block_word`. Falls
back to plain Pillow text when no source image is available.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kinema.vendor.protease.block_letter_bot import render_block_word

logger = logging.getLogger(__name__)

ASPECT_DIMS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}

_FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_fallback_font(size: int):
    for path in _FALLBACK_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _render_plain(text: str, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = h // 8
    font = _load_fallback_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    while text_w > w * 0.85 and font_size > 24:
        font_size = int(font_size * 0.9)
        font = _load_fallback_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - text_w) // 2 - bbox[0]
    y = (h - text_h) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(240, 240, 240))
    return img


def _wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    length = 0
    for word in words:
        added = length + (1 if cur else 0) + len(word)
        if cur and added > max_chars:
            lines.append(" ".join(cur))
            cur, length = [word], len(word)
        else:
            cur.append(word)
            length = added
    if cur:
        lines.append(" ".join(cur))
    return lines or [text]


def _block_letter_card(
    text: str, w: int, h: int, source_image: Path, rng: random.Random,
) -> Image.Image:
    """Render text via vendored protease-bots block-letter renderer, then
    composite onto a black canvas of the target aspect."""
    src = Image.open(source_image).convert("RGB")
    src_arr = np.array(src)

    lines = [ln.upper() for ln in _wrap(text, max_chars=12)]
    line_imgs: list[np.ndarray] = []
    # Per-line: rotate font seed so each line can have its own face — looks
    # busier but actually it's simpler if all lines share a font.
    font_size = max(80, h // (3 * max(2, len(lines))))
    depth = max(20, font_size // 4)
    for line in lines:
        try:
            arr = render_block_word(
                line, src_arr,
                font_size=font_size,
                depth_px=depth,
                angle_deg=rng.uniform(20, 40),
                letter_spacing=int(font_size * 0.06),
            )
        except Exception as e:  # noqa: BLE001 — fall back gracefully
            logger.warning("block_letter failed (%s); falling back to plain text", e)
            return _render_plain(text, w, h)
        # render_block_word returns RGBA-ish 4ch, we need RGB.
        if arr.shape[-1] == 4:
            # Composite over black using its alpha.
            alpha = arr[..., 3:4].astype(np.float32) / 255.0
            rgb = arr[..., :3].astype(np.float32)
            arr = (rgb * alpha).clip(0, 255).astype(np.uint8)
        line_imgs.append(arr)

    # Stack lines vertically with a small gap.
    gap = max(8, font_size // 8)
    total_h = sum(im.shape[0] for im in line_imgs) + gap * (len(line_imgs) - 1)
    max_w = max(im.shape[1] for im in line_imgs)
    stacked = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    cursor = 0
    for im in line_imgs:
        x0 = (max_w - im.shape[1]) // 2
        stacked[cursor:cursor + im.shape[0], x0:x0 + im.shape[1]] = im
        cursor += im.shape[0] + gap

    # Scale to fit ~80% of canvas while preserving aspect.
    sh, sw = stacked.shape[:2]
    scale = min((w * 0.85) / sw, (h * 0.85) / sh)
    new_w, new_h = max(1, int(sw * scale)), max(1, int(sh * scale))
    stacked_img = Image.fromarray(stacked).resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (w, h), color=(0, 0, 0))
    canvas.paste(stacked_img, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


def render_title_card(
    text: str,
    aspect: str,
    out_path: Path,
    *,
    source_image: Path | None = None,
    seed: int | None = None,
) -> Path:
    if aspect not in ASPECT_DIMS:
        raise ValueError(f"unsupported aspect: {aspect}")
    w, h = ASPECT_DIMS[aspect]
    rng = random.Random(seed)

    if source_image is not None and source_image.exists():
        img = _block_letter_card(text, w, h, source_image, rng)
    else:
        img = _render_plain(text, w, h)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path
