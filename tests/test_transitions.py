"""Unit tests for transition sampling and filter-graph building.

These tests don't shell out to ffmpeg — they only verify the strings we'd send.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from kinema.pipeline import Recipe, _build_filter_graph, _check_inputs
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
    "name,path",
    [
        ("smooth-fade", REPO_ROOT / "recipes" / "smooth-fade.yaml"),
        ("mask-bleed", REPO_ROOT / "recipes" / "mask-bleed.yaml"),
        ("glitch-cut", REPO_ROOT / "recipes" / "glitch-cut.yaml"),
        ("hybrid-wash", REPO_ROOT / "recipes" / "hybrid-wash.yaml"),
    ],
)
def test_shipped_recipes_load_and_sample(name: str, path: Path) -> None:
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


def test_build_filter_graph_has_correct_offsets_and_labels() -> None:
    recipe = Recipe.load(REPO_ROOT / "recipes" / "smooth-fade.yaml")
    rng = random.Random(0)
    graph, final = _build_filter_graph(
        n_images=4, width=320, height=180, sec_per_image=1.0, recipe=recipe, rng=rng
    )
    assert final == "vout"
    # Three transitions for four images; each xfade should have a monotonically
    # increasing offset of 0.5, 1.0, 1.5 (sec_per_image=1.0, duration=0.5 →
    # offset_i = (i+1)*(D-T) = (i+1)*0.5).
    assert "offset=0.500" in graph
    assert "offset=1.000" in graph
    assert "offset=1.500" in graph
    assert "[vout]" in graph
    # Each input was normalized.
    for i in range(4):
        assert f"[v{i}]" in graph
