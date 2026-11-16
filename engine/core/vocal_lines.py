"""Load the separated vocal track, and find its lines by the real rests between them."""
import numpy as np

import syllables as syl

REST_S = 0.35
QUIET_BELOW_MEDIAN_DB = 12


def load_vocal(project):
    return np.load(project / "analysis" / "stems" / "vocals_22k.npy").astype(np.float32) / 32768


def find_lines(vocal, section):
    """Spans of singing or rapping, split at real rests in the vocal track, in seconds."""
    start, end = section["first_beat_s"], section["last_grid_beat_s"]
    db, times = syl.loudness_curve(vocal, start, end)
    inside = (times >= start) & (times <= end)
    active_level = np.median(db[inside & (db > -55)]) if np.any(inside & (db > -55)) else -30
    quiet = db < active_level - QUIET_BELOW_MEDIAN_DB
    lines = []
    i = 0
    n = len(db)
    line_start = None
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) / syl.FPS >= REST_S and line_start is not None:
                lines.append((line_start, times[i]))
                line_start = None
            i = j
        else:
            if line_start is None:
                line_start = times[i]
            i += 1
    if line_start is not None:
        lines.append((line_start, times[-1]))
    return [(max(a, start), min(b, end)) for a, b in lines if b > start and a < end and b - a >= 0.3]
