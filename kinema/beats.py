"""Beat detection for beat-synced recipes.

Given an audio file, return a list of beat timestamps (seconds) that the
pipeline can use as image-cut boundaries. Detection via librosa's
onset-aware beat tracker.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa

logger = logging.getLogger(__name__)


def detect_beats(audio_path: Path, *, skip: int = 1) -> list[float]:
    """Return beat timestamps in seconds, every `skip`-th beat.

    skip=1 → every beat
    skip=2 → every other beat (half-time, doubles image duration)
    skip=4 → every bar (4/4)
    """
    if skip < 1:
        raise ValueError(f"skip must be >= 1, got {skip}")
    # sr=22050 is fine for beat tracking — doubling it doesn't improve onset
    # detection meaningfully and slows load 2x.
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats: list[float] = [float(b) for b in beat_frames]
    if skip > 1:
        beats = beats[::skip]
    # tempo is a numpy array in newer librosa — coerce to scalar safely.
    try:
        tempo_val = float(tempo.item() if hasattr(tempo, "item") else tempo)
    except (TypeError, ValueError):
        tempo_val = 0.0
    logger.info("detected %d beats (tempo ~%.1f BPM, skip=%d)", len(beats), tempo_val, skip)
    return beats


def beat_intervals(audio_path: Path, *, skip: int = 1, min_dur: float = 0.15) -> list[float]:
    """Return per-image durations matching the beats — length N gives N images.

    The first image plays from 0 to beats[0]; subsequent images play for the
    gap to the next beat. Any interval below `min_dur` is merged into the
    prior image to avoid impossibly short cuts.
    """
    beats = detect_beats(audio_path, skip=skip)
    if not beats:
        return []
    # Build raw intervals: [beat_0, beat_1-beat_0, beat_2-beat_1, ...]
    raw: list[float] = [beats[0]]
    for i in range(1, len(beats)):
        raw.append(beats[i] - beats[i - 1])

    # Merge too-short intervals with the next one.
    merged: list[float] = []
    carry = 0.0
    for d in raw:
        d += carry
        if d < min_dur:
            carry = d
        else:
            merged.append(d)
            carry = 0.0
    if carry > 0 and merged:
        merged[-1] += carry
    elif carry > 0:
        merged.append(carry)
    return merged
