"""Find a song's pulse (the slow beat the map counts in) and how each beat splits: 4 for straight songs, 3 for swung or 12/8, 6 for slow 6/8.

These were chosen by hand for every earlier song; this does it from the audio, checked against those hand choices (see tempo_check.py).
"""
import numpy as np
from scipy import signal

import grid_tools

LOW, HIGH = 52.0, 115.0
# bumped when detection changes, so cached tempos are found again
VERSION = 2
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


WINDOW_S, HOP_S = 20.0, 5.0
# two parts count as different tempos only when they differ by more than this, and last at least MIN_SECTION_S
CHANGE = 1.04
MIN_SECTION_S = 25.0
REAL_CHANGE = 1.5


def local_tempo(env, guide):
    """The strongest tempo in a stretch of the song, moved to the octave nearest the whole song's pulse, and how clearly it locks."""
    grid = np.arange(50.0, 210.0, 0.1)
    scores = np.array([comb(env, b) for b in grid])
    support = scores + 0.5 * np.interp(grid * 2, grid, scores, right=0) + 0.5 * np.interp(grid / 2, grid, scores, left=0)
    tempo = float(grid[int(np.argmax(support))])
    while tempo >= guide * 1.414:
        tempo /= 2
    while tempo < guide / 1.414:
        tempo *= 2
    return tempo, float(scores.max())


def tempo_sections(env, guide):
    """Parts of the song at different tempos, as [{"start_s", "pulse_bpm"}]. One part for almost every song; MY EYES has two (119 BPM, then 147 from 2:26.5)."""
    fps = grid_tools.FPS
    win, hop = int(WINDOW_S * fps), int(HOP_S * fps)
    starts = list(range(0, max(1, len(env) - win + 1), hop))
    rows = []
    for a in starts:
        part = env[a:a + win]
        if part.mean() < 0.2 * env.mean():
            continue
        tempo, clarity = local_tempo(part, guide)
        rows.append((a / fps, tempo, clarity))
    if len(rows) < 3:
        return [{"start_s": 0.0, "pulse_bpm": guide}]
    # runs of windows that agree, ignoring a window that disagrees with both neighbours
    tempos = np.array([r[1] for r in rows])
    smooth = np.array([np.median(tempos[max(0, i - 1):i + 2]) for i in range(len(tempos))])
    groups = [[0]]
    for i in range(1, len(rows)):
        reference = np.median(smooth[groups[-1]])
        if max(smooth[i], reference) / min(smooth[i], reference) > CHANGE:
            groups.append([i])
        else:
            groups[-1].append(i)
    # too short to be a real part: joins its neighbour
    merged = []
    for g in groups:
        span = rows[g[-1]][0] - rows[g[0]][0] + WINDOW_S
        if merged and (span < MIN_SECTION_S or max(np.median(smooth[g]), np.median(smooth[merged[-1]])) / min(np.median(smooth[g]), np.median(smooth[merged[-1]])) <= CHANGE):
            merged[-1] += g
        else:
            merged.append(list(g))
    if len(merged) > 1 and rows[merged[0][-1]][0] - rows[merged[0][0]][0] + WINDOW_S < MIN_SECTION_S:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    fps = grid_tools.FPS
    near = lambda part, bpm: max(comb(part, x) for x in np.arange(bpm * 0.995, bpm * 1.005, 0.05))
    bounds = [rows[g[0]][0] for g in merged] + [len(env) / fps]
    bounds[0] = 0.0
    # a real tempo change: the part's own tempo fits its beats far better than the whole song's does. MY EYES' first half: 2.56 times better.
    # Swung songs only seem to change, because triplets make neighbouring tempos fit too: there it was at most 1.05.
    labels = []
    for k, g in enumerate(merged):
        part = env[int(bounds[k] * fps):int(bounds[k + 1] * fps)]
        pulse = float(np.median(tempos[g]))
        own = near(part, pulse)
        whole = max(near(part, guide), near(part, guide * 2), near(part, guide / 2))
        labels.append(round(pulse, 1) if own >= REAL_CHANGE * whole else None)
    parts = []
    for k, g in enumerate(merged):
        pulse = float(np.median(tempos[g])) if labels[k] is not None else guide
        if parts and abs(parts[-1]["pulse_bpm"] / pulse - 1) < CHANGE - 1:
            continue
        parts.append({"start_s": bounds[k], "pulse_bpm": pulse})
    parts[0]["start_s"] = 0.0
    # where the new beat takes over: halfway into the first stretch where it clearly fits better than the old one.
    # A beatless bridge before it stays with the old tempo (MY EYES: the switch at 2:26.5, after a quiet bridge)
    for k in range(1, len(parts)):
        before, after = parts[k - 1]["pulse_bpm"], parts[k]["pulse_bpm"]
        lo, hi = max(parts[k - 1]["start_s"], parts[k]["start_s"] - WINDOW_S - HOP_S), parts[k]["start_s"] + WINDOW_S
        ts = np.arange(lo, hi, 0.25)
        lead = np.array([near(env[int(t * fps):int((t + 6) * fps)], after) - near(env[int(t * fps):int((t + 6) * fps)], before) for t in ts])
        first = int(np.argmax(lead > 0.5 * lead.max())) if lead.max() > 0 else 0
        parts[k]["start_s"] = round(float(ts[first] + 3.0), 2)
    return parts


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
    sections = tempo_sections(env, pulse)
    if len(sections) > 1:
        beats = grid_tools.track_beats_sections(env, sections)
    else:
        beats = grid_tools.track_beats(env, pulse)
    fitted, _ = grid_tools.fitted_beats(beats)
    # each part's tempo, from its own tracked beats
    for k, section in enumerate(sections):
        end = sections[k + 1]["start_s"] if k + 1 < len(sections) else fitted[-1] + 1
        inside = fitted[(fitted >= section["start_s"]) & (fitted < end)]
        if len(inside) > 4:
            section["pulse_bpm"] = round(float(60 / np.median(np.diff(inside))), 2)
    pulse = float(60 / np.median(np.diff(fitted)))
    s = subdivision_strength(env, fitted)
    triplet = s["third"] > s["quarter"] * 1.15
    per_beat, per_pulse = (3, 1) if triplet else (4, 1)
    # a slow pulse whose beat splits into two groups of three is 6/8: count it in sixes, and give the map twice the beats
    if triplet and pulse < 62 and s["half"] > 0.6 * s["third"]:
        per_beat, per_pulse = 6, 2
    out = {"pulse_bpm": round(pulse, 2), "per_beat": per_beat, "map_beats_per_pulse": per_pulse, "raw_tempo": round(tempo, 2),
           "strength": {k: round(v, 2) for k, v in s.items() if k != "fold"}}
    if len(sections) > 1:
        out["tempo_sections"] = sections
    return out
