"""FFmpeg filter-graph builders for image-to-image transitions.

Each builder returns a filter snippet (sans output label) that consumes two
input streams [a][b] and produces a single output stream. The pipeline module
appends the output label and stitches snippets into a chain.

v0 is FFmpeg-only. The transition vocabulary:

  xfade            — one named xfade transition (any of XFADE_MODES)
  mask             — sample from a curated list of mask-style xfade modes
                     (wipes, slides, slices, circle/vert/horz opens)
  glitch           — xfade with an explicit "glitchy" mode subset (pixelize,
                     hlslice, vuslice, distance, dissolve, fadegrays)
  tween            — minterpolate motion-compensated frame blend layered on
                     a base xfade. Slower; keeps everything CPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

# Full set of xfade transition names available in ffmpeg ≥ 4.3
XFADE_MODES = [
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite",
    "radial", "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circleopen", "circleclose", "vertopen", "vertclose",
    "horzopen", "horzclose", "dissolve", "pixelize",
    "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "hblur", "fadegrays", "wipetl", "wipetr", "wipebl", "wipebr",
    "squeezeh", "squeezev", "zoomin",
]

# Curated subset that "feels like" a mask transition — directional reveals,
# slices, and shape-based opens.
MASK_MODES = [
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "wipetl", "wipetr", "wipebl", "wipebr",
    "slideleft", "slideright", "slideup", "slidedown",
    "circleopen", "circleclose", "circlecrop",
    "rectcrop", "vertopen", "vertclose", "horzopen", "horzclose",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "diagtl", "diagtr", "diagbl", "diagbr",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
]

# Curated subset that "feels like" a glitch transition — abrupt or pixelated.
GLITCH_MODES = [
    "pixelize", "hlslice", "hrslice", "vuslice", "vdslice",
    "distance", "dissolve", "fadegrays", "fadeblack", "fadewhite",
    "squeezeh", "squeezev",
]


@dataclass
class TransitionSpec:
    """One sampled transition with concrete params, ready to render."""
    builder: Callable[..., str]
    duration: float
    params: dict
    name: str

    def filter_str(self, a_label: str, b_label: str, offset: float) -> str:
        return self.builder(a_label, b_label, self.duration, offset, **self.params)


def xfade(a: str, b: str, duration: float, offset: float, *, mode: str = "fade") -> str:
    return f"[{a}][{b}]xfade=transition={mode}:duration={duration:.3f}:offset={offset:.3f}"


def tween(a: str, b: str, duration: float, offset: float, *, fps: int = 30, base: str = "fade") -> str:
    """Motion-compensated tween via minterpolate. CPU-heavy; v1 may swap in
    RIFE/FILM via Modal. Pin fps + timebase at the tail so it's concatenable
    with other xfades (minterpolate rewrites timebase to 1/fps*N, which
    crashes subsequent xfade steps expecting 1/fps)."""
    return (
        f"[{a}][{b}]xfade=transition={base}:duration={duration:.3f}:offset={offset:.3f},"
        f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,"
        f"fps={fps},settb=AVTB,setpts=PTS-STARTPTS"
    )


# Aliases so recipes can name "mask" and "glitch" — the actual mode is sampled
# from a curated pool unless explicitly overridden.
def mask(a: str, b: str, duration: float, offset: float, *, mode: str | None = None) -> str:
    return xfade(a, b, duration, offset, mode=mode or "wipeleft")


def glitch(a: str, b: str, duration: float, offset: float, *, mode: str | None = None) -> str:
    return xfade(a, b, duration, offset, mode=mode or "pixelize")


BUILDERS: dict[str, Callable] = {
    "xfade": xfade,
    "mask": mask,
    "glitch": glitch,
    "tween": tween,
}


# Per-clip visual effects — applied during still_clip render BEFORE transitions.
# Each value is an ffmpeg video-filter chain snippet that transforms a single
# video stream. The pipeline prepends it to the scale+pad chain.
CLIP_EFFECTS: dict[str, str] = {
    "none": "null",
    "negate": "negate",
    "edges": "edgedetect=mode=colormix:high=0.2",
    "edges_hard": "edgedetect=mode=canny:low=0.1:high=0.4",
    "threshold": (
        "lutyuv='y=if(gt(val,128),255,0)':u=128:v=128"
    ),
    "posterize": "lutyuv='y=(floor(val/32))*32':u='(floor(val/32))*32':v='(floor(val/32))*32'",
    "hue_shift": "hue=h=180",
    "hue_drift": "hue=h='mod(t*60,360)'",  # rotate 60°/sec
    "saturate": "eq=saturation=2.0",
    "desaturate": "hue=s=0",
    "mono_red": "lutrgb=r='val':g=0:b=0",
    "mono_green": "lutrgb=r=0:g='val':b=0",
    "mono_blue": "lutrgb=r=0:g=0:b='val'",
    "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
    "emboss": "convolution='-2 -1 0 -1 1 1 0 1 2:-2 -1 0 -1 1 1 0 1 2:-2 -1 0 -1 1 1 0 1 2:-2 -1 0 -1 1 1 0 1 2'",
    "solarize": "lutrgb='r=if(gt(val,128),255-val,val)':g='if(gt(val,128),255-val,val)':b='if(gt(val,128),255-val,val)'",
    "high_contrast": "eq=contrast=2.0",
    "low_contrast": "eq=contrast=0.5:brightness=0.1",
    "vibrance": "eq=saturation=1.6:contrast=1.3",
    "chromashift": "split=3[r][g][b];[r]lutrgb=g=0:b=0[rr];[g]lutrgb=r=0:b=0[gg];[b]lutrgb=r=0:g=0[bb];[rr][gg]blend=all_mode=addition[rg];[rg][bb]blend=all_mode=addition",
    "blur": "boxblur=8:2",
    "pixelate": "scale=iw/8:ih/8,scale=iw*8:ih*8:flags=neighbor",
}


def pick_clip_effect(clip_effects: list[dict] | None, index: int) -> str:
    """Choose an effect for clip `index`. `clip_effects` is recipe config:
      [{"every": 2, "effect": "negate"}, {"every": 3, "effect": "edges"}]
    Later rules override earlier ones. Returns an ffmpeg vf chain or "null"."""
    if not clip_effects:
        return "null"
    chosen = "null"
    for rule in clip_effects:
        every = int(rule.get("every", 1))
        offset = int(rule.get("offset", 0))
        effect = rule.get("effect", "none")
        if every < 1:
            continue
        if (index - offset) % every == 0 and (index - offset) >= 0:
            chosen = CLIP_EFFECTS.get(effect, "null")
    return chosen

# Pools each "alias" type samples from when its mode is "random".
_RANDOM_POOLS: dict[str, list[str]] = {
    "xfade": XFADE_MODES,
    "mask": MASK_MODES,
    "glitch": GLITCH_MODES,
}


def sample_transition(pool: list[dict], rng: random.Random) -> TransitionSpec:
    """Pick one transition from a recipe's pool (weighted)."""
    if not pool:
        raise ValueError("transition pool is empty")
    weights = [float(t.get("weight", 1.0)) for t in pool]
    chosen = rng.choices(pool, weights=weights, k=1)[0]
    ttype = chosen["type"]
    builder = BUILDERS.get(ttype)
    if builder is None:
        raise ValueError(f"unknown transition type: {ttype}")

    params = dict(chosen.get("params") or {})
    duration = float(params.pop("duration", 0.5))

    # mode="random" → sample from this type's curated pool (or full XFADE_MODES for xfade).
    mode = params.get("mode")
    if mode == "random":
        pool_for_type = _RANDOM_POOLS.get(ttype, XFADE_MODES)
        params["mode"] = rng.choice(pool_for_type)

    return TransitionSpec(builder=builder, duration=duration, params=params, name=ttype)
