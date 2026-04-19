"""Title-card rendering with Pillow.

Outputs a single PNG sized to the target aspect; the pipeline treats it as the
first image in the sequence (gets the same display + transition treatment as
any other still).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASPECT_DIMS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}

_DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _DEFAULT_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_title_card(text: str, aspect: str, out_path: Path) -> Path:
    if aspect not in ASPECT_DIMS:
        raise ValueError(f"unsupported aspect: {aspect}")
    w, h = ASPECT_DIMS[aspect]

    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Wrap manually — pick a font size that lets the text occupy ~70% width.
    font_size = h // 8
    font = _load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    while text_w > w * 0.85 and font_size > 24:
        font_size = int(font_size * 0.9)
        font = _load_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x = (w - text_w) // 2 - bbox[0]
    y = (h - text_h) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(240, 240, 240))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path
