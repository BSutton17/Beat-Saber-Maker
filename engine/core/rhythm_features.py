"""Measure separated stems at each 1/4 slow-pulse step: the inputs to the rhythm model."""
import os

import numpy as np
from scipy import signal

import syllables

SR = 22050
NAMES = ["syllable", "syllable_before", "syllable_after", "vocal_level", "vocal_onset", "kick", "kick_before", "kick_after", "snare_hat",
         "snare_hat_before", "snare_hat_after", "bass", "other", "on_half_beat", "on_beat", "on_two_beats"]


def band_flux(audio, low, high):
    freqs, _, spec = signal.stft(audio, fs=SR, nperseg=512, noverlap=512 - 128, boundary=None, padded=False)
    band = (freqs >= low) & (freqs < high)
    log_mag = np.log1p(np.abs(spec[band]) * 200)
    flux = np.concatenate([[0], np.maximum(0, np.diff(log_mag, axis=1)).mean(axis=0)])
    scale = np.percentile(flux, 98) + 1e-9
    return np.minimum(flux / scale, 2.0)


def sample(curve, times, tolerance=0.03):
    fps = SR / 128
    centre = ((times - 256 / SR) * fps).astype(int)
    reach = int(tolerance * fps)
    out = np.zeros(len(times))
    for i, c in enumerate(centre):
        a, b = max(0, c - reach), min(len(curve), c + reach + 1)
        if a < b:
            out[i] = curve[a:b].max()
    return out


def stems_at(folder, step_times):
    load = lambda name: np.load(os.path.join(folder, f"{name}_22k.npy")).astype(np.float32) / 32768
    vocal, drums, bass, other = load("vocals"), load("drums"), load("bass"), load("other")
    syll = np.zeros(len(step_times))
    period = step_times[1] - step_times[0]
    for row in syllables.detect(vocal, min_prominence=3):
        i = int(round((row["time_s"] - step_times[0]) / period))
        if 0 <= i < len(step_times):
            error = abs(row["time_s"] - step_times[i])
            if error <= 0.07:
                syll[i] = max(syll[i], min(row["prominence"] / 20, 1.5) * (1 - error / 0.1))
    db, times = syllables.loudness_curve(vocal)
    level = np.interp(step_times, times, np.maximum(db, -60)) / 60 + 1
    onset = sample(band_flux(vocal, 150, 5000), step_times)
    kick = sample(band_flux(drums, 0, 200), step_times)
    snare = sample(band_flux(drums, 200, 11000), step_times)
    bass_on = sample(band_flux(bass, 0, 400), step_times)
    other_on = sample(band_flux(other, 200, 8000), step_times)
    before = lambda x: np.concatenate([[0], x[:-1]])
    after = lambda x: np.concatenate([x[1:], [0]])
    k = np.arange(len(step_times))
    return np.column_stack([syll, before(syll), after(syll), level, onset, kick, before(kick), after(kick), snare, before(snare), after(snare),
                            bass_on, other_on, k % 2 == 0, k % 4 == 0, k % 8 == 0]).astype(np.float32)
