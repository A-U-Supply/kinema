"""Unit tests for transition sampling and filter-graph building.

These tests don't shell out to ffmpeg — they only verify the strings we'd send.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from kinema.pipeline import Recipe, _check_inputs
from kinema.transitions import (
    BUILDERS,
    MASK_MODES,
    GLITCH_MODES,
    XFADE_MODES,
    sample_transition,
    xfade,
    mask,
    glitch,
    tween,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_xfade_emits_canonical_filter() -> None:
    s = xfade("a", "b", 0.5, 1.25, mode="circleopen")
    assert s == "[a][b]xfade=transition=circleopen:duration=0.500:offset=1.250"


def test_tween_chains_minterpolate() -> None:
    s = tween("a", "b", 0.4, 2.0, fps=24, base="dissolve")
    assert s.startswith("[a][b]xfade=transition=dissolve:duration=0.400:offset=2.000")
    assert "minterpolate=fps=24:mi_mode=mci" in s


def test_sample_transition_respects_weights() -> None:
    pool = [
        {"type": "xfade", "weight": 0.0, "params": {"mode": "fade", "duration": 0.5}},
        {"type": "mask", "weight": 1.0, "params": {"mode": "wipeleft", "duration": 0.5}},
    ]
    rng = random.Random(0)
    for _ in range(20):
        spec = sample_transition(pool, rng)
        assert spec.name == "mask"


def test_sample_transition_random_mode_picks_from_curated_pool() -> None:
    rng = random.Random(7)
    seen_modes: set[str] = set()
    pool = [{"type": "mask", "params": {"mode": "random", "duration": 0.5}}]
    for _ in range(50):
        spec = sample_transition(pool, rng)
        seen_modes.add(spec.params["mode"])
    # Every sampled mode must come from the curated mask pool.
    assert seen_modes <= set(MASK_MODES)
    # And we should see variety, not just one.
    assert len(seen_modes) > 1


def test_glitch_random_pool_constraint() -> None:
    rng = random.Random(13)
    pool = [{"type": "glitch", "params": {"mode": "random", "duration": 0.3}}]
    for _ in range(30):
        spec = sample_transition(pool, rng)
        assert spec.params["mode"] in GLITCH_MODES


def test_unknown_transition_type_raises() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="unknown transition type"):
        sample_transition([{"type": "diffuse", "params": {}}], rng)


def test_empty_pool_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        sample_transition([], random.Random(0))


@pytest.mark.parametrize(
    "path",
    sorted((REPO_ROOT / "recipes").glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_shipped_recipes_load_and_sample(path: Path) -> None:
    """Every file in recipes/ must parse and produce a sampleable transition
    of a known type. This catches typos in new recipes before they ship."""
    recipe = Recipe.load(path)
    assert recipe.name
    rng = random.Random(0)
    spec = sample_transition(recipe.transitions, rng)
    assert spec.duration > 0
    assert spec.name in BUILDERS


def test_check_inputs_rejects_overlong_transitions() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        _check_inputs(sec_per_image=0.5, transition_seconds=0.5)


def test_check_inputs_accepts_normal_durations() -> None:
    _check_inputs(sec_per_image=1.5, transition_seconds=0.5)  # no raise


