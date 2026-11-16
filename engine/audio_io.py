"""Audio decoding without MATLAB: soundfile if it's installed, otherwise the ffmpeg the app ships with (BSM_FFMPEG)."""
import os
import shutil
import subprocess

import numpy as np


def ffmpeg_path():
    return os.environ.get("BSM_FFMPEG") or shutil.which("ffmpeg")


def decode_channels(path):
    """Returns (samples x channels float32, sample rate)."""
    try:
        import soundfile
        data, rate = soundfile.read(path, dtype="float32", always_2d=True)
        return data, rate
    except ImportError:
        pass
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("no audio decoder: set BSM_FFMPEG or install soundfile")
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace").stderr
    rate = 44100
    for part in probe.split(","):
        if part.strip().endswith(" Hz"):
            rate = int(part.strip().split()[0])
            break
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(rate), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy(), rate


def decode(path):
    """Same shape as AI Mapper's metrics.decode: mono float32 and the sample rate."""
    data, rate = decode_channels(str(path))
    return data.mean(axis=1).astype(np.float32), rate
