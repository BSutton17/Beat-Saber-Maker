"""Find syllables in a separated vocal track.

A syllable is a peak in vocal loudness. Its time is the steepest rise before that peak, which is about where a listener
hears the word start. Prominence is how far it rises above the dip before it, and stressed syllables score higher.
"""
import numpy as np
from scipy import signal

SR = 22050
HOP = 128
NFFT = 512
FPS = SR / HOP


def loudness_curve(vocal, start_s=0.0, end_s=None):
    a = max(0, int((start_s - 1) * SR))
    b = len(vocal) if end_s is None else min(len(vocal), int((end_s + 1) * SR))
    freqs, _, spec = signal.stft(vocal[a:b], fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    band = (freqs >= 150) & (freqs <= 5000)
    power = (np.abs(spec[band]) ** 2).sum(axis=0)
    kernel = signal.windows.hann(9)
    smooth = np.convolve(power, kernel / kernel.sum(), mode="same")
    db = 10 * np.log10(smooth + 1e-10)
    times = (a + NFFT / 2) / SR + np.arange(len(db)) * HOP / SR
    return db, times


def detect(vocal, start_s=0.0, end_s=None, min_prominence=6.0, min_gap_s=0.08, floor_db=-45.0):
    db, times = loudness_curve(vocal, start_s, end_s)
    peaks, props = signal.find_peaks(db, height=floor_db, prominence=min_prominence, distance=max(1, int(min_gap_s * FPS)))
    rise = np.diff(db, prepend=db[0])
    rows = []
    for peak, left, prominence in zip(peaks, props["left_bases"], props["prominences"]):
        reach = max(left, peak - int(0.15 * FPS))
        steepest = reach + int(np.argmax(rise[reach:peak + 1]))
        when = float(times[steepest])
        rows.append({"time_s": when, "peak_s": float(times[peak]), "prominence": float(prominence), "level_db": float(db[peak])})
    rows = [r for r in rows if r["time_s"] >= start_s and (end_s is None or r["time_s"] <= end_s)]
    return rows
