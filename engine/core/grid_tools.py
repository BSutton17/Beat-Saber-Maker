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
