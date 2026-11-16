"""Split the concert into vocals, drums, bass and other with Hybrid Demucs from torchaudio.

Usage: python -I separate_stems.py <song.ogg> <output folder>
Saves 22.05 kHz mono 16-bit arrays, because later steps only need timing and these sync through OneDrive.
"""
import ctypes
import os
import sys
import time

import numpy as np
import torch
import torchaudio

DLL_DIR = r"C:\Program Files\MATLAB\R2025a\bin\win64"
SEGMENT_S = 10.0
OVERLAP_S = 1.0


class Info(ctypes.Structure):
    _fields_ = [("frames", ctypes.c_int64), ("samplerate", ctypes.c_int), ("channels", ctypes.c_int),
                ("format", ctypes.c_int), ("sections", ctypes.c_int), ("seekable", ctypes.c_int)]


def decode(path):
    os.add_dll_directory(DLL_DIR)
    sf = ctypes.CDLL(os.path.join(DLL_DIR, "libsndfile-1.dll"))
    sf.sf_open.restype = ctypes.c_void_p
    sf.sf_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(Info)]
    sf.sf_readf_float.restype = ctypes.c_int64
    sf.sf_readf_float.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    sf.sf_close.argtypes = [ctypes.c_void_p]
    info = Info()
    handle = sf.sf_open(path.encode("utf-8"), 0x10, ctypes.byref(info))
    if not handle:
        raise RuntimeError(f"could not open {path}")
    buf = np.zeros((info.frames, info.channels), dtype=np.float32)
    got = sf.sf_readf_float(handle, buf.ctypes.data_as(ctypes.c_void_p), info.frames)
    sf.sf_close(handle)
    return buf[:got], info.samplerate


def main():
    song, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    audio, rate = decode(song)
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    print("loading model", flush=True)
    model = bundle.get_model().eval()
    names = list(model.sources)
    print("sources", names, flush=True)
    wave = torch.from_numpy(audio.T).float()
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    if rate != bundle.sample_rate:
        wave = torchaudio.transforms.Resample(rate, bundle.sample_rate)(wave)
        rate = bundle.sample_rate
    mean, std = wave.mean(), wave.std()
    wave = (wave - mean) / std

    chunk = int(SEGMENT_S * rate)
    overlap = int(OVERLAP_S * rate)
    step = chunk - overlap
    total = wave.shape[1]
    result = torch.zeros(len(names), 2, total)
    weight = torch.zeros(total)
    fade = torch.ones(chunk)
    ramp = torch.linspace(0, 1, overlap)
    fade[:overlap] = ramp
    fade[-overlap:] = ramp.flip(0)
    started = time.time()
    starts = list(range(0, total, step))
    with torch.no_grad():
        for n, a in enumerate(starts):
            z = min(a + chunk, total)
            piece = wave[:, a:z].unsqueeze(0)
            sources = model(piece)[0]
            window = fade[:z - a]
            if a == 0:
                window = window.clone()
                window[:overlap] = 1
            if z == total:
                window = window.clone()
                window[-min(overlap, z - a):] = 1
            result[:, :, a:z] += sources * window
            weight[a:z] += window
            elapsed = time.time() - started
            print(f"chunk {n + 1}/{len(starts)}, audio {z / rate:.0f} s, elapsed {elapsed:.0f} s, "
                  f"estimated left {elapsed / (n + 1) * (len(starts) - n - 1):.0f} s", flush=True)
            if z == total:
                break
    result = result / weight.clamp(min=1e-6) * std + mean

    resampler = torchaudio.transforms.Resample(rate, 22050)
    for i, name in enumerate(names):
        mono = resampler(result[i].mean(dim=0)).numpy()
        data = np.clip(mono * 32767, -32768, 32767).astype(np.int16)
        np.save(os.path.join(out, f"{name}_22k.npy"), data)
        rms = float(np.sqrt(np.mean((mono.astype(np.float64)) ** 2)))
        print(name, "saved, samples", len(data), "rms", round(rms, 4), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
