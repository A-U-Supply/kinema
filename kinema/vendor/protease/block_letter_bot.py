"""3D block letter texture bot.

Fetches 3 source images and maps each as a texture onto isometric 3D block
letters spelling a word chosen randomly from a list.

For each letter:
  - Front face: source image texture, masked to the letter glyph outline
  - Top face:   source image texture, dimmed (simulates top lighting)
  - Right face: source image texture, darker (simulates shadow side)

UV coordinates are computed per-face using parallelogram inversion so the
texture wraps correctly onto each angled surface.

Posts 3 images.
"""
import argparse
import logging
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def wrap_text(text, max_chars=15):
    """Wrap text at word boundaries, returning a list of line strings."""
    words = text.split()
    lines, current, length = [], [], 0
    for word in words:
        added = length + (1 if current else 0) + len(word)
        if current and added > max_chars:
            lines.append(' '.join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length = added
    if current:
        lines.append(' '.join(current))
    return lines


WORD_LIST = [
    "PULSE", "WAVE", "FLUX", "ECHO", "LOOP",
    "VOID", "APEX", "NEON", "GRID", "HAZE",
    "BEAM", "CORE", "EDGE", "FLOW", "GLOW",
    "HEAT", "IRIS", "KNOT", "LENS", "NOVA",
    "PRISM", "SHIFT", "SPARK", "TRACE", "VEIL",
]

# Font catalog: (display_name, [path_candidates]) — multiple paths handle
# different distro layouts for the same font.
_BUNDLED_FONT_DIR = str(Path(__file__).parent / "fonts")

_FONT_CATALOG = [
    ("Liberation Sans Bold", [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ]),
    ("Liberation Serif Bold", [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSerif-Bold.ttf",
    ]),
    ("DejaVu Sans Bold", [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]),
    ("DejaVu Serif Bold", [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf",
    ]),
    ("FreeSans Bold", [
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/freefont/FreeSansBold.ttf",
    ]),
    ("FreeSerif Bold", [
        "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
        "/usr/share/fonts/freefont/FreeSerifBold.ttf",
    ]),
    ("Ubuntu Bold", [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/ubuntu/Ubuntu-B.ttf",
    ]),
    # macOS — wide variety: sans, serif, condensed, monospace
    ("Impact",            ["/System/Library/Fonts/Supplemental/Impact.ttf"]),
    ("Georgia Bold",      ["/System/Library/Fonts/Supplemental/Georgia Bold.ttf"]),
    ("Times New Roman Bold", ["/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"]),
    ("Courier New Bold",  ["/System/Library/Fonts/Supplemental/Courier New Bold.ttf"]),
    ("DIN Condensed Bold",["/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"]),
    ("DIN Alternate Bold",["/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"]),
    ("Trebuchet MS Bold", ["/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf"]),
    ("Arial Black",       ["/System/Library/Fonts/Supplemental/Arial Black.ttf"]),
    ("Verdana Bold",      ["/System/Library/Fonts/Supplemental/Verdana Bold.ttf"]),
    ("Arial Narrow Bold", ["/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf"]),
    # Bundled Google Fonts (committed to /fonts in repo — work locally and in CI)
    ("Anton",          [f"{_BUNDLED_FONT_DIR}/Anton-Regular.ttf"]),
    ("Alfa Slab One",  [f"{_BUNDLED_FONT_DIR}/AlfaSlabOne-Regular.ttf"]),
    ("Bangers",        [f"{_BUNDLED_FONT_DIR}/Bangers-Regular.ttf"]),
    ("Black Ops One",  [f"{_BUNDLED_FONT_DIR}/BlackOpsOne-Regular.ttf"]),
    ("Boogaloo",       [f"{_BUNDLED_FONT_DIR}/Boogaloo-Regular.ttf"]),
    ("Lobster",        [f"{_BUNDLED_FONT_DIR}/Lobster-Regular.ttf"]),
    ("Pacifico",       [f"{_BUNDLED_FONT_DIR}/Pacifico-Regular.ttf"]),
    ("Righteous",      [f"{_BUNDLED_FONT_DIR}/Righteous-Regular.ttf"]),
    ("Russo One",      [f"{_BUNDLED_FONT_DIR}/RussoOne-Regular.ttf"]),
    ("Squada One",     [f"{_BUNDLED_FONT_DIR}/SquadaOne-Regular.ttf"]),
    # Linux apt fonts (installed by workflow)
    ("Open Sans Bold",  ["/usr/share/fonts/truetype/open-sans/OpenSans-Bold.ttf"]),
    ("Lato Bold",       ["/usr/share/fonts/truetype/lato/Lato-Bold.ttf"]),
    ("Lato Black",      ["/usr/share/fonts/truetype/lato/Lato-Black.ttf"]),
    ("Arimo Bold",      ["/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"]),
    ("Tinos Bold",      ["/usr/share/fonts/truetype/croscore/Tinos-Bold.ttf"]),
    ("Cousine Bold",    ["/usr/share/fonts/truetype/croscore/Cousine-Bold.ttf"]),
    ("Carlito Bold",    ["/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf"]),
    ("Caladea Bold",    ["/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf"]),
    ("Roboto Bold",     ["/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
                         "/usr/share/fonts/truetype/roboto/unhinted/Roboto-Bold.ttf"]),
    ("Roboto Black",    ["/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Black.ttf",
                         "/usr/share/fonts/truetype/roboto/unhinted/Roboto-Black.ttf"]),
]


def _available_fonts():
    """Return list of (name, path) for every catalog font present on disk."""
    available = []
    for name, paths in _FONT_CATALOG:
        for p in paths:
            if Path(p).exists():
                available.append((name, p))
                break
    return available


def _load_font(size, font_path=None):
    candidates = [font_path] if font_path else []
    for _name, paths in _FONT_CATALOG:
        candidates.extend(paths)
    for p in candidates:
        if p is None:
            continue
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            continue
    logger.warning("No truetype font found, falling back to default")
    return ImageFont.load_default()


def _face_uv(xx, yy, origin, vec_a, vec_b):
    """Compute (u, v) for every pixel relative to parallelogram face.

    Face is: origin + u·vec_a + v·vec_b,  u,v ∈ [0, 1].
    Returns u, v (full-image arrays) and boolean inside mask.
    """
    det = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]
    if abs(det) < 1e-6:
        return np.zeros_like(xx), np.zeros_like(yy), np.zeros(xx.shape, bool)
    dx = xx - origin[0]
    dy = yy - origin[1]
    u = (dx * vec_b[1] - dy * vec_b[0]) / det
    v = (dy * vec_a[0] - dx * vec_a[1]) / det
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    return u, v, inside


def _paint_face(output, src_f, xx, yy, origin, vec_a, vec_b, brightness,
                glyph_mask=None, glyph_W=None, glyph_H=None,
                src_x_min=0, src_x_max=None, src_y_min=0, src_y_max=None,
                gamma=1.0):
    """Sample source image onto a parallelogram face and write into output.

    src_x_min/src_x_max and src_y_min/src_y_max define the square crop of
    the source image to sample from, one crop per letter.
    """
    u, v, inside = _face_uv(xx, yy, origin, vec_a, vec_b)
    if not inside.any():
        return

    if glyph_mask is not None:
        u_px = (u * (glyph_W - 1)).clip(0, glyph_W - 1).astype(np.int32)
        v_px = (v * (glyph_H - 1)).clip(0, glyph_H - 1).astype(np.int32)
        inside = inside & glyph_mask[v_px, u_px]
        if not inside.any():
            return

    sh, sw = src_f.shape[:2]
    if src_x_max is None:
        src_x_max = sw - 1
    if src_y_max is None:
        src_y_max = sh - 1
    sx = (src_x_min + u[inside] * (src_x_max - src_x_min)).clip(0, sw - 1).astype(np.int32)
    sy = (src_y_min + v[inside] * (src_y_max - src_y_min)).clip(0, sh - 1).astype(np.int32)
    rgb = (src_f[sy, sx] * brightness).clip(0, 255)
    if gamma != 1.0:
        rgb = (rgb / 255.0) ** gamma * 255.0
    output[inside] = np.concatenate([rgb, np.full((len(rgb), 1), 255.0)], axis=1)


def _fill_edge_gaps(mask):
    """Fill gaps between consecutive rows of edge pixels.

    For curved/diagonal letter edges, consecutive rows can have edge pixels
    several columns apart (e.g. up to 7px at the top of an O).  The extrusion
    of these sparse rows leaves transparent stripes.

    For each pair of consecutive rows, matches each edge pixel in row r to its
    nearest-column neighbour in row r+1 and fills all intermediate columns in
    row r+1 as bridge pixels, ensuring extrusion strips are contiguous.

    Apply transposed (mask.T → fill → .T) for top_edge, where the gap
    direction is along columns instead of rows.
    """
    H, W = mask.shape
    out = mask.copy()
    for r in range(H - 1):
        cur = np.where(mask[r])[0]
        nxt = np.where(mask[r + 1])[0]
        if not len(cur) or not len(nxt):
            continue
        for c in cur:
            nearest = nxt[np.argmin(np.abs(nxt - c))]
            lo, hi = min(c, nearest), max(c, nearest)
            if hi > lo:
                out[r + 1, lo:hi] = True
    return out


def _paint_edge_extrusion(output, src_f, x0, y0, edge_mask, iso_dx, iso_dy,
                          brightness, src_x_min, src_x_max, src_y_min, src_y_max, face):
    """Paint extruded edge faces following all letter contours including inner holes.

    edge_mask: (H, W) boolean — True at every edge pixel to extrude.
      For face='right': pixels where glyph has no glyph to the right.
      For face='top':   pixels where glyph has no glyph above.

    Both outer boundaries and inner hole boundaries are included, so counters
    in letters like O, B, P, R get correctly extruded inner walls.
    """
    sh, sw = src_f.shape[:2]
    n_steps = max(abs(iso_dx), abs(iso_dy), 1) + 1
    ts = np.linspace(0, 1, n_steps)

    ys, xs = np.where(edge_mask)   # all edge pixel row/col indices
    if not len(ys):
        return

    H, W = edge_mask.shape

    # Output pixel coords: (n_edges, n_steps)
    px = np.round(x0 + xs[:, None] + ts[None, :] * iso_dx).astype(int)
    py = np.round(y0 + ys[:, None] - ts[None, :] * iso_dy).astype(int)

    if face == 'right':
        # t → source x (depth into extrusion), glyph row → source y
        src_x = np.clip(src_x_min + ts * (src_x_max - src_x_min), 0, sw - 1).astype(int)
        src_y = np.clip(src_y_min + ys / max(H - 1, 1) * (src_y_max - src_y_min),
                        0, sh - 1).astype(int)
        sx = np.broadcast_to(src_x[None, :], px.shape)
        sy = np.broadcast_to(src_y[:, None], px.shape)
    else:  # face == 'top'
        # glyph col → source x, t → source y (depth into extrusion)
        src_x = np.clip(src_x_min + xs / max(W - 1, 1) * (src_x_max - src_x_min),
                        0, sw - 1).astype(int)
        src_y = np.clip(src_y_min + ts * (src_y_max - src_y_min), 0, sh - 1).astype(int)
        sx = np.broadcast_to(src_x[:, None], px.shape)
        sy = np.broadcast_to(src_y[None, :], px.shape)

    mask = (px >= 0) & (px < output.shape[1]) & (py >= 0) & (py < output.shape[0])
    rgb = (src_f[sy[mask], sx[mask]] * brightness).clip(0, 255)
    output[py[mask], px[mask]] = np.concatenate([rgb, np.full((len(rgb), 1), 255.0)], axis=1)


def render_block_word(word, src_arr, font_size=200, depth_px=70, angle_deg=30,
                      letter_spacing=12, shade_top=0.78, shade_right=0.52,
                      font_path=None):
    """Render the source image as isometric 3D block letter texture.

    Args:
        word:           text to render
        src_arr:        uint8 (H, W, 3) source image
        font_size:      letter height in pixels (default 200)
        depth_px:       extrusion depth (default 70)
        angle_deg:      isometric angle in degrees (default 30)
        letter_spacing: pixels between letters (default 12)
        shade_top:      top face brightness (default 0.78)
        shade_right:    right face brightness (default 0.52)

    Returns:
        uint8 (out_H, out_W, 3) rendered image
    """
    font = _load_font(font_size, font_path)

    rad = math.radians(angle_deg)
    iso_dx = int(depth_px * math.cos(rad))   # rightward component of extrusion
    iso_dy = int(depth_px * math.sin(rad))   # upward component (positive = up)

    # Measure characters
    chars = list(word)
    char_data = []
    for c in chars:
        bbox = font.getbbox(c)           # (left, top, right, bottom)
        W = bbox[2] - bbox[0]
        H = bbox[3] - bbox[1]
        char_data.append((c, bbox, W, H))

    total_w = sum(d[2] for d in char_data) + letter_spacing * max(0, len(chars) - 1)
    max_h   = max(d[3] for d in char_data)

    pad    = 40
    # Use abs so the canvas is large enough for any extrusion direction.
    # iso_dx or iso_dy can be negative when angle_deg is outside [0°, 90°].
    out_w  = total_w + 2 * abs(iso_dx) + 2 * pad
    out_h  = max_h   + abs(iso_dy) + 2 * pad

    output = np.zeros((out_h, out_w, 4), dtype=np.float32)
    src_f  = src_arr.astype(np.float32)

    xs = np.arange(out_w, dtype=np.float64)
    ys = np.arange(out_h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    x_cursor = abs(iso_dx) + pad   # room on both sides for any extrusion direction
    y0_base  = abs(iso_dy) + pad

    vec_back = np.array([iso_dx, -iso_dy], dtype=float)  # extrusion direction

    # Collect front faces; paint edge extrusions first, then front face on top
    front_faces = []

    n_letters = len(char_data)
    src_sh, src_sw = src_f.shape[:2]

    # Find the largest square tile size that yields at least n_letters tiles
    n_needed = max(n_letters, 1)
    lo, hi = 1, min(src_sw, src_sh)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if (src_sw // mid) * (src_sh // mid) >= n_needed:
            lo = mid
        else:
            hi = mid - 1
    tile_size = lo
    cols = src_sw // tile_size
    rows = src_sh // tile_size

    # Shuffle all tile origins so each letter gets a random region
    tile_origins = [(r * tile_size, c * tile_size)
                    for r in range(rows) for c in range(cols)]
    random.shuffle(tile_origins)

    for i, (c, bbox, W, H) in enumerate(char_data):
        x0 = x_cursor
        y0 = y0_base

        vec_right = np.array([W, 0.0])
        vec_down  = np.array([0.0, H])

        # Skip non-renderable characters (e.g. space has H=0)
        if W == 0 or H == 0:
            x_cursor += W + letter_spacing
            continue

        # Glyph mask for front face
        glyph_img = Image.new('L', (W, H), 0)
        ImageDraw.Draw(glyph_img).text((-bbox[0], -bbox[1]), c, font=font, fill=255)
        glyph_mask = np.array(glyph_img) > 128

        # Horizontal edge: outermost pixel per row in the extrusion's x direction.
        # Only the outermost pixel is used to avoid phantom inner-hole layers
        # (e.g. inside O, B, P).
        horiz_edge = np.zeros_like(glyph_mask)
        for r in range(H):
            row_cols = np.where(glyph_mask[r])[0]
            if len(row_cols):
                # iso_dx >= 0 → extrusion goes right → show right face (outermost right)
                # iso_dx <  0 → extrusion goes left  → show left face  (outermost left)
                horiz_edge[r, row_cols[-1] if iso_dx >= 0 else row_cols[0]] = True

        # Vertical edge: outermost pixel per column in the extrusion's y direction.
        # Inner hole edges on the same side as the extrusion are also included
        # (e.g. ceiling of O's counter is visible when extrusion goes up).
        vert_edge = np.zeros_like(glyph_mask)
        if iso_dy >= 0:
            # Extrusion goes up → show top face: topmost pixel per column
            above = np.zeros_like(glyph_mask)
            above[1:, :] = glyph_mask[:-1, :]
            vert_edge = glyph_mask & ~above
        else:
            # Extrusion goes down → show bottom face: bottommost pixel per column
            below = np.zeros_like(glyph_mask)
            below[:-1, :] = glyph_mask[1:, :]
            vert_edge = glyph_mask & ~below

        # Fill column/row gaps so curved edges extrude without transparent stripes
        horiz_edge = _fill_edge_gaps(horiz_edge)
        vert_edge  = _fill_edge_gaps(vert_edge.T).T

        # Assign a random tile from the grid to this letter
        ty, tx = tile_origins[i]
        x_min = tx
        x_max = tx + tile_size - 1
        y_min = ty
        y_max = ty + tile_size - 1

        # Paint edge extrusions (horiz then vert; front face will overdraw)
        _paint_edge_extrusion(output, src_f, x0, y0, horiz_edge, iso_dx, iso_dy,
                              shade_right, x_min, x_max, y_min, y_max, face='right')
        _paint_edge_extrusion(output, src_f, x0, y0, vert_edge, iso_dx, iso_dy,
                              shade_top, x_min, x_max, y_min, y_max, face='top')

        # Front face: top-left corner → right → down, masked to glyph
        front_faces.append(((x0, y0), vec_right, vec_down, glyph_mask, W, H,
                             x_min, x_max, y_min, y_max))

        x_cursor += W + letter_spacing

    for origin, va, vb, gm, gW, gH, xmn, xmx, ymn, ymx in front_faces:
        _paint_face(output, src_f, xx, yy, origin, va, vb, 1.0, gm, gW, gH, xmn, xmx, ymn, ymx,
                    gamma=0.65)

    return output.clip(0, 255).astype(np.uint8)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="3D block letter texture bot")
    parser.add_argument("--source-channel", default="image-gen")
    parser.add_argument("--post-channel",   default="img-junkyard")
    parser.add_argument("--output-dir", type=Path, default=Path("./block-letter-bot-output"))
    parser.add_argument("--text-channel", default="song-titles",
                        help="Slack channel to fetch text from (default: song-titles)")
    parser.add_argument("--custom-text", default="",
                        help="Manually entered text; if non-empty, skips Slack fetch entirely")
    parser.add_argument("--words", default="",
                        help="Comma-separated fallback word list (used if --text-channel is empty)")
    parser.add_argument("--font-size",  type=int, default=200)
    parser.add_argument("--max-chars",  type=int, default=15,
                        help="Max characters per wrapped line (default 15)")
    parser.add_argument("--depth",      type=int, default=70,
                        help="Extrusion depth in pixels (default 70)")
    parser.add_argument("--angle",     type=float, default=30.0,
                        help="Isometric angle in degrees (default 30)")
    parser.add_argument("--font-name", default="random",
                        help="Font name from catalog, or 'random' to pick automatically (default)")
    parser.add_argument("--no-post",   action="store_true")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    from slack_fetcher import fetch_random_images, fetch_random_message_texts
    from slack_poster import post_collages

    if args.custom_text.strip():
        # Manual text overrides Slack fetch entirely; treat each line as a candidate
        word_list = [
            line.strip().upper()
            for line in args.custom_text.strip().splitlines()
            if line.strip()
        ]
    elif args.text_channel:
        raw_titles = fetch_random_message_texts(token, args.text_channel, 50)
        word_list = []
        for t in raw_titles:
            # Add every non-empty line as a separate candidate
            for line in t.split("\n"):
                # Skip lines containing URLs
                if re.search(r"https?://|www\.", line, re.IGNORECASE):
                    continue
                cleaned = re.sub(r"[^A-Za-z\s]", "", line).strip().upper()
                # Skip lines that are a single unbroken run (likely a URL residue or code)
                if cleaned and len(cleaned.split()) >= 1 and max(len(w) for w in cleaned.split()) <= 20:
                    word_list.append(cleaned)
        if not word_list:
            word_list = [w.strip().upper() for w in args.words.split(",") if w.strip()]
    else:
        word_list = [w.strip().upper() for w in args.words.split(",") if w.strip()]

    if not word_list:
        word_list = WORD_LIST

    source_dir = args.output_dir / "source"
    out_dir    = args.output_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    available_fonts = _available_fonts()
    if args.font_name and args.font_name != "random":
        entry = next((e for e in _FONT_CATALOG if e[0] == args.font_name), None)
        chosen_font_path = next((p for p in entry[1] if Path(p).exists()), None) if entry else None
        if chosen_font_path:
            chosen_font_name = args.font_name
        else:
            logger.warning(f"Font '{args.font_name}' not found on disk, falling back to random")
            chosen_font_name, chosen_font_path = random.choice(available_fonts) if available_fonts else ("default", None)
    elif available_fonts:
        chosen_font_name, chosen_font_path = random.choice(available_fonts)
    else:
        chosen_font_name, chosen_font_path = "default", None
        logger.warning("No catalog fonts found, using PIL default")
    logger.info(f"Using font: {chosen_font_name}")

    logger.info(f"Fetching 3 images from #{args.source_channel}...")
    source_paths = list(fetch_random_images(token, args.source_channel, 3, source_dir))

    output_paths = []
    for i, path in enumerate(source_paths, start=1):
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        src_h, src_w = arr.shape[:2]

        title = random.choice(word_list)
        rows  = wrap_text(title, max_chars=args.max_chars)
        logger.info(f"Image {i}: {src_w}×{src_h}, title='{title}' → {len(rows)} row(s)")

        # Each row gets a horizontal band of the source image
        n_rows = len(rows)
        row_images = []
        for j, row_text in enumerate(rows):
            y0 = int(j / n_rows * src_h)
            y1 = int((j + 1) / n_rows * src_h)
            band = arr[y0:y1, :]
            logger.info(f"  Row {j + 1}: '{row_text}'")
            result = render_block_word(row_text, band, font_size=args.font_size,
                                       depth_px=args.depth, angle_deg=args.angle,
                                       font_path=chosen_font_path)
            row_images.append(result)

        # Stack rows vertically, left-aligned, 20px gap
        gap     = 20
        max_w   = max(r.shape[1] for r in row_images)
        total_h = sum(r.shape[0] for r in row_images) + gap * (n_rows - 1)
        canvas  = np.zeros((total_h, max_w, 4), dtype=np.uint8)
        y = 0
        for r in row_images:
            canvas[y:y + r.shape[0], :r.shape[1]] = r
            y += r.shape[0] + gap

        dest = out_dir / f"block_letter_result_{i}.png"
        Image.fromarray(canvas).save(dest)
        logger.info(f"  Saved {dest.name} ({canvas.shape[1]}×{canvas.shape[0]})")
        output_paths.append(dest)

    if not args.no_post:
        post_collages(token, args.post_channel, output_paths,
                      bot_name="block-letter-bot", threaded=False)
        logger.info(f"Posted {len(output_paths)} files to #{args.post_channel}")
    else:
        logger.info(f"Saved to {out_dir} (--no-post)")


if __name__ == "__main__":
    main()
