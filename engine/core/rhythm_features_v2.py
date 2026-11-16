"""Measure separated stems at each step of a song: the inputs to the v2 rhythm model.

The first 16 columns match v1, so v1 can be tested on the same data. The rest are for what v1 couldn't see: singing and acoustic songs lean on chord changes
and phrase starts more than drum hits, and mappers place fewer notes in quiet parts. Steps may be unevenly spaced, so hand-played songs that drift are measured correctly.
"""
import os

import numpy as np
from scipy import signal

import syllables
from rhythm_features import band_flux, sample

SR = 22050
NAMES_V1 = ["syllable", "syllable_before", "syllable_after", "vocal_level", "vocal_onset", "kick", "kick_before", "kick_after", "snare_hat",
            "snare_hat_before", "snare_hat_after", "bass", "other", "on_half_beat", "on_beat", "on_two_beats"]
NAMES = NAMES_V1 + ["chord_change", "phrase_start", "energy", "drum_level", "vocal_active", "kick_no_vocal", "snare_no_vocal", "other_no_vocal", "chord_no_vocal", "other_before", "other_after", "syllable_on_beat"]
ACTIVE_WINDOW_S = 1.0


def metric_flags(strength, per_beat):
    """Every step's place in the beat, lined up with where the music actually hits. Counting from the first step instead would put "the beat" somewhere arbitrary.

    Straight songs split 2, 4, 8 steps, and 6/8 songs split 3, 6, 12.
    """
    k = np.arange(len(strength))
    beat_at = max(range(per_beat), key=lambda o: strength[o::per_beat].mean())
    two_at = max((beat_at, beat_at + per_beat), key=lambda o: strength[o::per_beat * 2].mean())
    place = (k - beat_at) % per_beat
    # a swung beat has no even half: its off-beat is the last of the three parts, long-short, so that's what plays the half beat's role
    half_beat = np.isin(place, (0, 2)) if per_beat == 3 else place % (per_beat // 2) == 0
    return half_beat, place == 0, (k - two_at) % (per_beat * 2) == 0


def rms_curve(audio, window_s=0.5):
    """Loudness in dB, about every 10 ms."""
    hop = 220
    size = int(window_s * SR)
    power = signal.fftconvolve(audio.astype(np.float64) ** 2, np.ones(size) / size, mode="same")[::hop]
    return 10 * np.log10(np.maximum(power, 1e-10)), np.arange(len(power)) * hop / SR


def chord_change(music, times):
    """How much the notes being played change across each step, from a 12-note profile of the bass and other stems."""
    freqs, frame_times, spec = signal.stft(music, fs=SR, nperseg=4096, noverlap=4096 - 1024, boundary=None, padded=False)
    keep = (freqs >= 80) & (freqs <= 2000)
    pitch = (np.round(12 * np.log2(freqs[keep] / 440.0)) % 12).astype(int)
    power = np.abs(spec[keep]) ** 2
    chroma = np.zeros((12, power.shape[1]))
    for note in range(12):
        chroma[note] = power[pitch == note].sum(axis=0)
    chroma = np.sqrt(chroma)
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    fps = 1 / (frame_times[1] - frame_times[0])
    reach = max(1, int(0.35 * fps))
    out = np.zeros(len(times))
    centres = np.searchsorted(frame_times, times)
    for i, c in enumerate(centres):
        before, after = chroma[:, max(0, c - reach):c], chroma[:, c:c + reach]
        if before.shape[1] and after.shape[1]:
            a, b = before.mean(axis=1), after.mean(axis=1)
            out[i] = 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return np.minimum(out / (np.percentile(out, 98) + 1e-9), 1.5)


def stems_at(folder, step_times, per_beat=4):
    step_times = np.asarray(step_times, float)
    load = lambda name: np.load(os.path.join(folder, f"{name}_22k.npy")).astype(np.float32) / 32768
    vocal, drums, bass, other = load("vocals"), load("drums"), load("bass"), load("other")
    syll = np.zeros(len(step_times))
    for row in syllables.detect(vocal, min_prominence=3):
        i = int(np.searchsorted(step_times, row["time_s"]))
        i = min(max(i, 1), len(step_times) - 1)
        i = i - 1 if abs(row["time_s"] - step_times[i - 1]) < abs(row["time_s"] - step_times[i]) else i
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
    half, beat, two = metric_flags(kick + snare + bass_on + other_on, per_beat)
    # context around each step
    length = min(len(vocal), len(drums), len(bass), len(other))
    mix = vocal[:length] + drums[:length] + bass[:length] + other[:length]
    mix_db, level_times = rms_curve(mix)
    top = np.percentile(mix_db, 95)
    energy = np.clip(np.interp(step_times, level_times, mix_db - top) / 40 + 1, 0, 1.1)
    drum_db, _ = rms_curve(drums[:length])
    drum_level = np.clip(np.interp(step_times, level_times, drum_db - top) / 40 + 1, 0, 1.1)
    vocal_db, _ = rms_curve(vocal[:length])
    sung = np.percentile(vocal_db, 90)
    active_curve = vocal_db > sung - 18
    fps = 1 / (level_times[1] - level_times[0])
    size = int(ACTIVE_WINDOW_S * fps)
    active = np.interp(step_times, level_times, signal.fftconvolve(active_curve.astype(float), np.ones(size) / size, mode="same"))
    active = np.clip(active, 0, 1)
    # a phrase start: the voice now, after being quiet just before
    now = np.interp(step_times, times, db)
    earlier = np.interp(step_times - 0.3, times, db)
    phrase = np.clip((now - earlier) / 30, 0, 1) * (earlier < sung - 18)
    chord = chord_change(bass[:length] + other[:length], step_times)
    no_vocal = 1 - active
    return np.column_stack([syll, before(syll), after(syll), level, onset, kick, before(kick), after(kick), snare, before(snare), after(snare),
                            bass_on, other_on, half, beat, two,
                            chord, phrase, energy, drum_level, active, kick * no_vocal, snare * no_vocal, other_on * no_vocal, chord * no_vocal, before(other_on), after(other_on), syll * half]).astype(np.float32)
