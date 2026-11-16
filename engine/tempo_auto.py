"""Find a song's pulse (the slow beat the map counts in) and how each beat splits: 4 for straight songs, 3 for swung or 12/8, 6 for slow 6/8.

These were chosen by hand for every earlier song; this does it from the audio, checked against those hand choices (see tempo_check.py).
"""
import numpy as np
from scipy import signal

import grid_tools

LOW, HIGH = 52.0, 115.0
BINS = 24


def comb(env, bpm):
    """How strongly onsets line up on a grid at this tempo: 1.0 is an even spread, higher is locked."""
    period = 60 * grid_tools.FPS / bpm
    bins = 64
    which = ((np.arange(len(env)) / period) % 1.0 * bins).astype(int)
    fold = np.bincount(which, weights=env, minlength=bins) / (np.bincount(which, minlength=bins) + 1e-9)
    return fold.max() / (fold.mean() + 1e-9)


def best_tempo(env):
    grid = np.arange(50.0, 210.0, 0.05)
    scores = np.array([comb(env, b) for b in grid])
    # a tempo also counts the support of its half and double, so the beat wins over a busy hi-hat or a slow bar
    support = scores + 0.5 * np.interp(grid * 2, grid, scores, right=0) + 0.5 * np.interp(grid / 2, grid, scores, left=0)
    return float(grid[int(np.argmax(support))]), grid, scores


def subdivision_strength(env, beats):
    fold = np.zeros(BINS)
    for a, b in zip(beats, beats[1:]):
        i, j = int((a - grid_tools.CALIBRATION_S) * grid_tools.FPS), int((b - grid_tools.CALIBRATION_S) * grid_tools.FPS)
        if j <= i + BINS // 2:
            continue
        part = env[i:j]
        where = (np.arange(len(part)) / len(part) * BINS).astype(int)
        fold += np.bincount(where, weights=part, minlength=BINS) / (np.bincount(where, minlength=BINS) + 1e-9)
    fold /= fold.max() + 1e-9
    near = lambda k: float(fold[max(0, k - 1):k + 2].max())
    return {"quarter": (near(6) + near(18)) / 2, "half": near(12), "third": (near(8) + near(16)) / 2, "fold": fold}


def detect(mono):
    """mono: 22.05 kHz float audio. Returns pulse_bpm, per_beat, map_beats_per_pulse and the evidence."""
    env = grid_tools.onset_envelope(mono)
    tempo, grid, scores = best_tempo(env)
    near = lambda b: max(comb(env, x) for x in np.arange(b * 0.99, b * 1.01, 0.05))
    # a 12/8 song can lock onto every other swung eighth, 1.5x its beat; its real beat then explains the onsets better than the tempo found
    # (My Dearest Darling: 2.50 against 1.82; on every straight song checked the 2/3 tempo scored at least 20% lower)
    if near(tempo * 2 / 3) > near(tempo) * 1.1:
        tempo = tempo * 2 / 3
    pulse = tempo
    while pulse >= HIGH:
        pulse /= 2
    while pulse < LOW:
        pulse *= 2
    beats = grid_tools.track_beats(env, pulse)
    fitted, _ = grid_tools.fitted_beats(beats)
    pulse = float(60 / np.median(np.diff(fitted)))
    s = subdivision_strength(env, fitted)
    triplet = s["third"] > s["quarter"] * 1.15
    per_beat, per_pulse = (3, 1) if triplet else (4, 1)
    # a slow pulse whose beat splits into two groups of three is 6/8: count it in sixes, and give the map twice the beats
    if triplet and pulse < 62 and s["half"] > 0.6 * s["third"]:
        per_beat, per_pulse = 6, 2
    return {"pulse_bpm": round(pulse, 2), "per_beat": per_beat, "map_beats_per_pulse": per_pulse, "raw_tempo": round(tempo, 2),
            "strength": {k: round(v, 2) for k, v in s.items() if k != "fold"}}
