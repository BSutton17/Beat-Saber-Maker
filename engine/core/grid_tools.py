"""Find a song's beat grid, including songs played by hand that drift.

A beat tracker follows the beats, then runs of beats that fit one steady tempo become one tempo region. A locked song ends up as one region,
and a drifting one as several, so the grid stays on the beat from start to end. A step is one subdivision of the beat: 4 for straight songs,
3 or 6 for songs with a triplet or 6/8 feel.
"""
import numpy as np
from scipy import signal

SR, HOP = 22050, 128
FPS = SR / HOP
# community maps' notes sit about 40 ms after the onset-envelope peak
CALIBRATION_S = 0.040
# the most a beat may sit away from a steady tempo before a new region starts. About what a player can feel.
MAX_DRIFT_S = 0.028
MIN_REGION_BEATS = 8


def onset_envelope(mono):
    freqs, _, spec = signal.stft(mono, fs=SR, nperseg=1024, noverlap=1024 - HOP, boundary=None, padded=False)
    log_mag = np.log1p(np.abs(spec[freqs < 8000]) * 100)
    flux = np.concatenate([[0], np.maximum(0, np.diff(log_mag, axis=1)).sum(axis=0)])
    n = int(FPS)
    flux = np.maximum(0, flux - signal.fftconvolve(flux, np.ones(n) / n, mode="same"))
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    return np.convolve(flux, kernel / kernel.sum(), mode="same")


def track_beats(env, bpm, tightness=300.0):
    """Ellis's dynamic programming beat tracker: strong onsets, with each beat close to the expected spacing."""
    period = 60 * FPS / bpm
    score = env / (env.std() + 1e-9)
    best = np.copy(score)
    back = np.full(len(env), -1)
    low, high = int(period * 0.82), int(period * 1.22)
    lags = np.arange(low, high + 1)
    cost = -tightness * np.log(lags / period) ** 2
    for t in range(high, len(env)):
        candidates = best[t - lags] + cost
        i = int(np.argmax(candidates))
        if candidates[i] > 0:
            best[t] = score[t] + candidates[i]
            back[t] = t - lags[i]
    t = int(np.argmax(best[-int(period * 2):])) + len(best) - int(period * 2)
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1], float) / FPS + CALIBRATION_S


def track_beats_sections(env, sections):
    """Beats for a song whose tempo changes, like MY EYES (119 BPM, then 147 from 2:26.5). sections: [{"start_s", "pulse_bpm"}, ...] in order.
    Each part is tracked at its own tempo, so the tracker never stretches one tempo over the other."""
    starts = [float(s["start_s"]) for s in sections] + [len(env) / FPS + 1]
    out = []
    for section, start, end in zip(sections, starts, starts[1:]):
        a, b = int(start * FPS), min(len(env), int(end * FPS))
        if b - a < FPS * 4:
            continue
        beats = track_beats(env[a:b], float(section["pulse_bpm"])) + a / FPS
        period = 60 / float(section["pulse_bpm"])
        # where two parts meet, a beat too close to the last one is dropped
        if out:
            beats = beats[beats > out[-1] + 0.6 * period]
        out.extend(beats.tolist())
    return np.array(out)


def steady_regions(beats):
    """Greedy: the longest runs that fit one steady tempo. Returns (first index, last index, slope, intercept)."""
    regions = []
    i = 0
    n = len(beats)
    while i < n - 1:
        j = min(i + MIN_REGION_BEATS, n - 1)
        best = None
        while j <= n - 1:
            k = np.arange(i, j + 1)
            slope, intercept = np.polyfit(k, beats[i:j + 1], 1)
            error = np.abs(beats[i:j + 1] - (slope * k + intercept))
            # a few misplaced beats shouldn't break a tempo that otherwise holds
            if np.percentile(error, 90) > MAX_DRIFT_S or error.max() > 2.5 * MAX_DRIFT_S:
                break
            best = (i, j, float(slope), float(intercept))
            j += 1
        if best is None:
            j = min(i + MIN_REGION_BEATS, n - 1)
            k = np.arange(i, j + 1)
            slope, intercept = np.polyfit(k, beats[i:j + 1], 1)
            best = (i, j, float(slope), float(intercept))
        regions.append(best)
        i = best[1]
    return regions


def onset_support(env, times):
    """How strongly the sound hits on these beat times, on average."""
    idx = ((np.asarray(times) - CALIBRATION_S) * FPS).astype(int)
    idx = idx[(idx > 2) & (idx < len(env) - 3)]
    if not len(idx):
        return 0.0
    return float(np.mean(np.max(np.stack([env[idx + d] for d in range(-2, 3)]), axis=0)))


def steady_line(beats, env):
    """The single steady tempo line, near these beats' tempo and phase, that lands on the sound best. Returns (how well it lands, its beats)."""
    if len(beats) < 8:
        return -1.0, beats
    t0, t1 = beats[0], beats[-1]
    p0 = float(np.median(np.diff(beats)))
    best = (-1.0, beats)
    for p in p0 * (1 + np.linspace(-0.006, 0.006, 25)):
        angle = 2 * np.pi * (beats % p) / p
        phase0 = (np.angle(np.mean(np.exp(1j * angle))) % (2 * np.pi)) / (2 * np.pi) * p
        for phase in phase0 + np.linspace(-0.06, 0.06, 25):
            start = t0 - ((t0 - phase) % p)
            if start < t0 - p / 2:
                start += p
            line = np.arange(start, t1 + p / 2, p)
            score = onset_support(env, line)
            if score > best[0]:
                best = (score, line)
    return best


def straighten(beats, env, sections=None, chunk=32):
    """The tracker sometimes latches onto an off-beat sound for a few bars (MY EYES: two 8-beat runs, a third of a beat out). Produced songs hold one tempo, so for
    each tempo part the best steady line is tried, and every 32-beat stretch keeps whichever lands on the sound better: the line, or the tracked beats.
    Songs that really drift, played live, keep their tracked beats, because the line fits them worse. Checked against the beat grids of 27 human maps
    the engine never tuned on: the worst 10% of beats went from 28 ms to 8 ms off (typical song), and no song got more than 10 ms worse."""
    fitted, _ = fitted_beats(beats)
    starts = [float(s["start_s"]) for s in sections] if sections else [0.0]
    ends = starts[1:] + [float("inf")]
    out = []
    for s0, s1 in zip(starts, ends):
        part = fitted[(fitted >= s0) & (fitted < s1)]
        score, line = steady_line(part, env)
        if score < 0:
            out.append(part)
            continue
        half = 0.5 * float(np.median(np.diff(part)))
        edges = list(range(0, len(part), chunk)) + [len(part)]
        for a, b in zip(edges, edges[1:]):
            mine = part[a:b]
            if b < len(part):
                theirs = line[(line >= part[a] - half) & (line < part[b] - half)]
            else:
                theirs = line[line >= part[a] - half]
            out.append(theirs if len(theirs) and onset_support(env, theirs) > onset_support(env, mine) else mine)
    joined = np.sort(np.concatenate(out))
    # where stretches meet, two beats closer than half a beat are one beat
    return joined[np.concatenate([[True], np.diff(joined) > 0.5 * np.median(np.diff(joined))])]


def fitted_beats(beats):
    """Beat times with each steady region's fit in place of the tracker's jitter, joined so that time always moves forward."""
    regions = steady_regions(beats)
    out = np.array(beats, float)
    for first, last, slope, intercept in regions:
        out[first:last + 1] = slope * np.arange(first, last + 1) + intercept
    # the shared beat between two regions sits halfway between their fits
    for (_, last, slope, intercept), (first, _, next_slope, next_intercept) in zip(regions, regions[1:]):
        out[last] = 0.5 * (slope * last + intercept + next_slope * first + next_intercept)
    # a region plays at one tempo, so its beats must lie exactly on the straight line between its two ends
    for first, last, _, _ in regions:
        out[first:last + 1] = np.linspace(out[first], out[last], last - first + 1)
    return out, regions


def steps_from_beats(beats, per_beat):
    """Every step's time, evenly splitting each beat."""
    index = np.arange((len(beats) - 1) * per_beat + 1) / per_beat
    return np.interp(index, np.arange(len(beats)), beats)


def bar_phase(values, per_beat, beats_per_bar=4):
    """Which beat starts the bar, from where heavy low hits fall."""
    per_step = values[::per_beat]
    scores = [float(per_step[k::beats_per_bar].mean()) for k in range(beats_per_bar)]
    return int(np.argmax(scores)), scores
