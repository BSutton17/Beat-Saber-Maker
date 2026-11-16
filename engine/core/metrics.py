"""Read-only measurement of Beat Saber maps (v2, v3, v4). Writes results to the path given."""
import ctypes
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from collections import Counter

import numpy as np
from scipy import signal

DLL_DIR = r"C:\Program Files\MATLAB\R2025a\bin\win64"
UP = {0, 4, 5}
DOWN = {1, 6, 7}
ANGLE = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}


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
        return None, None
    buf = np.zeros((info.frames, info.channels), dtype=np.float32)
    got = sf.sf_readf_float(handle, buf.ctypes.data_as(ctypes.c_void_p), info.frames)
    sf.sf_close(handle)
    return buf[:got].mean(axis=1), info.samplerate


def onset_env(mono, sr):
    hop = 128 * sr // 22050
    nfft = 1024 * sr // 22050
    freqs, _, spec = signal.stft(mono, fs=sr, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    mag = np.log1p(np.abs(spec[freqs < 8000]) * 100).astype(np.float32)
    flux = np.concatenate([[0], np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)])
    n = int(sr / hop)
    flux = np.maximum(0, flux - signal.fftconvolve(flux, np.ones(n) / n, mode="same"))
    return flux, sr / hop


class Source:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path) if path.lower().endswith(".zip") else None
        self.dir = None if self.zip else path
        self.names = {n.lower(): n for n in self.zip.namelist()} if self.zip else {
            n.lower(): n for n in os.listdir(path)}

    def read(self, name):
        key = name.lower()
        if key not in self.names:
            raise FileNotFoundError(name)
        if self.zip:
            return self.zip.read(self.names[key])
        with open(os.path.join(self.dir, self.names[key]), "rb") as fh:
            return fh.read()

    def json(self, name):
        return json.loads(self.read(name).decode("utf-8-sig"))


def difficulties(src):
    info = src.json("Info.dat")
    if "_beatsPerMinute" in info:
        bpm = info["_beatsPerMinute"]
        song = info["_songFilename"]
        env = info.get("_environmentName")
        offset_ms = info.get("_songTimeOffset", 0)
        rows = []
        for group in info["_difficultyBeatmapSets"]:
            if group["_beatmapCharacteristicName"] != "Standard":
                continue
            for d in group["_difficultyBeatmaps"]:
                rows.append({"name": d["_difficulty"], "file": d["_beatmapFilename"], "njs": d["_noteJumpMovementSpeed"],
                             "offset": d["_noteJumpStartBeatOffset"], "light": None})
        return dict(bpm=bpm, song=song, env=env, offset_ms=offset_ms, audio=None, rows=rows)
    bpm = info["audio"]["bpm"]
    rows = []
    for d in info["difficultyBeatmaps"]:
        if d["characteristic"] != "Standard":
            continue
        rows.append({"name": d["difficulty"], "file": d["beatmapDataFilename"], "njs": d["noteJumpMovementSpeed"],
                     "offset": d["noteJumpStartBeatOffset"], "light": d.get("lightshowDataFilename")})
    envs = info.get("environmentNames") or [None]
    return dict(bpm=bpm, song=info["audio"]["songFilename"], env=envs[0], offset_ms=0,
                audio=info["audio"].get("audioDataFilename"), rows=rows)


def load_diff(src, row):
    d = src.json(row["file"])
    notes, events, bombs, walls, bpm_changes = [], [], 0, 0, []
    if "colorNotes" in d and "colorNotesData" in d:
        data = d["colorNotesData"]
        for n in d["colorNotes"]:
            e = data[n.get("i", 0)]
            notes.append((n.get("b", 0), e.get("c", 0), e.get("x", 0), e.get("y", 0), e.get("d", 0)))
        bombs = len(d.get("bombNotes", []))
        walls = len(d.get("obstacles", []))
        arcs, chains = len(d.get("arcs", [])), len(d.get("chains", []))
        if row["light"]:
            try:
                l = src.json(row["light"])
                ed = l.get("basicEventsData", [])
                for e in l.get("basicEvents", []):
                    m = ed[e.get("i", 0)]
                    events.append((e.get("b", 0), m.get("t", 0), m.get("i", 0), m.get("f", 1)))
                cd = l.get("colorBoostEventsData", [])
                for e in l.get("colorBoostEvents", []):
                    events.append((e.get("b", 0), 5, cd[e.get("i", 0)].get("b", 0), 1))
            except FileNotFoundError:
                pass
    elif "colorNotes" in d:
        for n in d["colorNotes"]:
            notes.append((n.get("b", 0), n.get("c", 0), n.get("x", 0), n.get("y", 0), n.get("d", 0)))
        bombs = len(d.get("bombNotes", []))
        walls = len(d.get("obstacles", []))
        arcs, chains = len(d.get("sliders", [])), len(d.get("burstSliders", []))
        for e in d.get("basicBeatmapEvents", []):
            events.append((e.get("b", 0), e.get("et", 0), e.get("i", 0), e.get("f", 1)))
        for e in d.get("colorBoostBeatmapEvents", []):
            events.append((e.get("b", 0), 5, 1 if e.get("o") else 0, 1))
        bpm_changes = [(e["b"], e["m"]) for e in d.get("bpmEvents", [])]
    else:
        for n in d.get("_notes", []):
            t = n.get("_type", 0)
            if t in (0, 1):
                notes.append((n["_time"], t, n["_lineIndex"], n["_lineLayer"], n["_cutDirection"]))
            elif t == 3:
                bombs += 1
        walls = len(d.get("_obstacles", []))
        arcs, chains = len(d.get("_sliders", [])), 0
        for e in d.get("_events", []):
            if e.get("_type") == 100:
                bpm_changes.append((e["_time"], e.get("_floatValue", 0)))
            else:
                events.append((e["_time"], e["_type"], e.get("_value", 0), e.get("_floatValue", 1)))
        cd = d.get("_customData") or {}
        for e in cd.get("_BPMChanges", cd.get("_bpmChanges", [])) or []:
            bpm_changes.append((e["_time"], e["_BPM"]))
    return sorted(notes), sorted(events), bombs, walls, arcs, chains, sorted(bpm_changes)


def time_map(base_bpm, changes):
    """Beat to seconds, treating change times as raw file beats."""
    points = [(0.0, 0.0, base_bpm)]
    for b, m in changes:
        if m <= 0:
            continue
        pb, pt, pm = points[-1]
        points.append((b, pt + (b - pb) * 60 / pm, m))
    bs = np.array([p[0] for p in points])
    ts = np.array([p[1] for p in points])
    ms = np.array([p[2] for p in points])

    def conv(beat):
        i = np.searchsorted(bs, beat, side="right") - 1
        i = np.maximum(i, 0)
        return ts[i] + (beat - bs[i]) * 60 / ms[i]

    return conv


def swing_metrics(notes, sec):
    """Group each hand's notes into swings, then classify the angle between swings."""
    result = {"angles": Counter(), "resets": 0, "fast_resets": 0, "swings": 0, "dots": 0, "gaps": []}
    swing_times = []
    for hand in (0, 1):
        hn = [n for n in notes if n[1] == hand]
        prev = None
        prev_t = None
        for n in hn:
            t = sec(n[0])
            if prev_t is not None and t - prev_t < 0.05:
                continue
            d = n[4]
            result["swings"] += 1
            swing_times.append(t)
            if d == 8:
                result["dots"] += 1
                prev, prev_t = None, t
                continue
            if prev is not None:
                diff = abs(ANGLE[d] - ANGLE[prev]) % 360
                diff = min(diff, 360 - diff)
                result["angles"][diff] += 1
                if diff <= 45:
                    result["resets"] += 1
                    if t - prev_t < 0.45:
                        result["fast_resets"] += 1
                result["gaps"].append(t - prev_t)
            prev, prev_t = d, t
    return result, np.array(sorted(swing_times))


def precision(notes):
    c = Counter()
    for n in notes:
        f = n[0] % 1
        for name, step in (("1/1", 1), ("1/2", 0.5), ("1/4", 0.25), ("1/3", 1 / 3), ("1/8", 0.125), ("1/6", 1 / 6), ("1/16", 0.0625)):
            k = f / step
            if abs(k - round(k)) * step < 0.012:
                c[name] += 1
                break
        else:
            c["other"] += 1
    return c


def variety(notes):
    tokens = [(n[1], n[2], n[3], n[4]) for n in notes]
    if len(tokens) < 5:
        return None
    c = Counter(tokens)
    total = sum(c.values())
    ent = -sum(v / total * math.log2(v / total) for v in c.values())
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    gc = Counter(grams)
    top = gc.most_common(1)[0][1] / len(grams)
    return {"token_entropy_bits": round(ent, 2), "unique_4gram_share": round(len(gc) / len(grams), 3),
            "top_4gram_share": round(top, 3), "unique_tokens": len(c)}


def analyse(src, meta, row, env_info, window=None):
    notes, events, bombs, walls, arcs, chains, changes = load_diff(src, row)
    if window:
        notes = [n for n in notes if window[0] <= n[0] <= window[1]]
        events = [e for e in events if window[0] <= e[0] <= window[1]]
    bpm = meta["bpm"]
    if meta["audio"]:
        try:
            ad = src.json(meta["audio"])
            sr = ad.get("songFrequency") or 44100
            regions = ad.get("bpmData", [])
            if len(regions) > 1:
                changes = [(r["sb"], (r["eb"] - r["sb"]) / max(r["ei"] - r["si"], 1) * sr * 60) for r in regions]
        except FileNotFoundError:
            pass
    sec = time_map(bpm, changes)
    if not notes:
        return None
    t = np.array([sec(n[0]) for n in notes])
    dur = float(t[-1] - t[0]) or 1.0
    njs, off = row["njs"], row["offset"]
    # game formula for half jump duration
    bps = bpm / 60
    hjd = 4.0
    while njs and njs * (1 / bps) * hjd > 17.999:
        hjd /= 2
    hjd = max(hjd + off, 0.25)
    jd = njs / bps * 2 * hjd
    rt = jd / (2 * njs) * 1000 if njs else None

    windows = np.arange(t[0], t[-1], 4)
    counts = np.array([((t >= w) & (t < w + 8)).sum() / 8 for w in windows]) if len(windows) else np.array([0])
    nonzero = counts[counts > 0]
    sw, swing_t = swing_metrics(notes, sec)
    angle_total = sum(sw["angles"].values()) or 1
    x = Counter(n[2] for n in notes)
    y = Counter(n[3] for n in notes)
    d = Counter(n[4] for n in notes)
    total = len(notes)
    beats = Counter(round(n[0], 3) for n in notes)
    doubles = sum(1 for n in notes if beats[round(n[0], 3)] >= 2)
    cross = sum(1 for n in notes if (n[1] == 0 and n[2] >= 2) or (n[1] == 1 and n[2] <= 1))
    face = sum(1 for n in notes if n[2] in (1, 2) and n[3] == 1)
    up_row = sum(1 for n in notes if n[3] == 2)

    out = {
        "diff": row["name"], "notes": total, "njs": njs, "offset": off, "hjd_beats": round(hjd, 3), "jd_m": round(jd, 2),
        "rt_ms": round(rt) if rt else None, "nps_mean": round(total / dur, 2), "nps_p95_8s": round(float(np.percentile(nonzero, 95)), 2),
        "nps_max_8s": round(float(counts.max()), 2), "nps_cv_8s": round(float(nonzero.std() / nonzero.mean()), 2),
        "swings_per_s": round(sw["swings"] / dur, 2), "bombs": bombs, "walls": walls, "arcs": arcs, "chains": chains,
        "bpm_changes": len(changes), "lanes_pct": [round(x[i] / total * 100, 1) for i in range(4)],
        "layers_pct": [round(y[i] / total * 100, 1) for i in range(3)],
        "dir_cardinal_vertical_pct": round((d[0] + d[1]) / total * 100, 1), "dir_horizontal_pct": round((d[2] + d[3]) / total * 100, 1),
        "dir_diagonal_pct": round((d[4] + d[5] + d[6] + d[7]) / total * 100, 1), "dir_dot_pct": round(d[8] / total * 100, 1),
        "doubles_pct": round(doubles / total * 100, 1), "cross_pct": round(cross / total * 100, 1), "face_pct": round(face / total * 100, 1),
        "top_row_pct": round(up_row / total * 100, 1), "left_hand_pct": round(sum(1 for n in notes if n[1] == 0) / total * 100, 1),
        "angle_share_pct": {str(k): round(sw["angles"][k] / angle_total * 100, 1) for k in (0, 45, 90, 135, 180)},
        "fastest_same_hand_ms": round(min(sw["gaps"]) * 1000) if sw["gaps"] else None,
        "resets_per_100_swings": round(sw["resets"] / max(sw["swings"], 1) * 100, 1),
        "fast_resets_per_100_swings": round(sw["fast_resets"] / max(sw["swings"], 1) * 100, 1),
        "precision_pct": {k: round(v / total * 100, 1) for k, v in precision(notes).most_common()},
        "variety": variety(notes),
    }

    # alignment of notes to audio onsets, against the same note set shifted by half a 1/16 grid and by random lag
    if env_info is not None:
        env, fr, song_dur = env_info
        offset_s = meta["offset_ms"] / 1000
        ut = np.unique(np.round(t + offset_s, 3))
        ut = ut[(ut > 0.2) & (ut < song_dur - 0.2)]
        if len(ut):
            def strength(ts):
                tol = int(0.035 * fr)
                vals = []
                for s in ts:
                    c = int(s * fr)
                    a, z = max(0, c - tol), min(len(env), c + tol + 1)
                    vals.append(env[a:z].max() if z > a else 0)
                return float(np.mean(vals))

            rng = np.random.default_rng(7)
            base = strength(ut)
            null = [strength(ut + rng.uniform(0.09, 0.31)) for _ in range(40)]
            tol = int(0.035 * fr)
            thresh = np.percentile(env, 90)
            hit = np.mean([env[max(0, int(s * fr) - tol):int(s * fr) + tol + 1].max() >= thresh for s in ut])
            out["onset_strength_vs_shifted"] = round(base / (np.mean(null) + 1e-9), 2)
            out["notes_on_top10pct_onset_pct"] = round(float(hit) * 100, 1)
            # best constant offset of the whole note set, to expose sync drift
            lags = np.arange(-0.08, 0.0801, 0.005)
            scores = [strength(ut + l) for l in lags]
            out["best_global_offset_ms"] = round(float(lags[int(np.argmax(scores))] * 1000))

    # lighting
    if events:
        secs = np.array([sec(e[0]) for e in events])
        span = max(float(secs.max() - secs.min()), 1.0)
        types = Counter(e[1] for e in events)
        lightish = [e for e in events if e[1] in (0, 1, 2, 3, 4, 6, 7, 10, 11)]
        vals = Counter()
        for e in lightish:
            v = e[2]
            cls = "off" if v == 0 else {1: "on", 2: "flash", 3: "fade", 0: "transition"}[v % 4]
            vals[cls] += 1
        nt = np.unique(np.round(t, 3))
        lt = np.array([sec(e[0]) for e in lightish]) if lightish else np.array([])
        near = float(np.mean([np.min(np.abs(nt - s)) < 0.03 for s in np.unique(np.round(lt, 3))])) if len(lt) else 0
        # flashes: per-lane off/on or colour-flash transitions per second, peak in a 1 second window
        rates = []
        for lane in sorted(set(e[1] for e in lightish)):
            lane_t = np.array(sorted(sec(e[0]) for e in lightish if e[1] == lane))
            if len(lane_t) > 3:
                c = np.searchsorted(lane_t, lane_t + 1.0) - np.arange(len(lane_t))
                rates.append(int(c.max()))
        # seconds with 4+ lanes on an off or flash event at nearly the same time, a proxy for full-field flashing
        flashy = [sec(e[0]) for e in lightish if e[2] == 0 or e[2] % 4 == 2]
        fl = np.array(sorted(flashy))
        field_fps = 0
        if len(fl):
            uniq = np.unique(np.round(fl, 2))
            c = np.searchsorted(uniq, uniq + 1.0) - np.arange(len(uniq))
            field_fps = int(c.max())
        ev = {"events": len(events), "ev_per_s": round(len(events) / span, 2), "types": dict(sorted(types.items())),
              "value_class_pct": {k: round(v / max(len(lightish), 1) * 100, 1) for k, v in vals.most_common()},
              "light_instants_on_a_note_pct": round(near * 100, 1), "peak_lane_events_per_s": max(rates) if rates else 0,
              "peak_distinct_flash_instants_per_s": field_fps, "boosts": types.get(5, 0)}
        out["lighting"] = ev
    return out


def main():
    jobs = json.loads(sys.argv[1]) if sys.argv[1].startswith("[") else json.load(open(sys.argv[1], encoding="utf-8"))
    results = []
    for job in jobs:
        src = Source(job["path"])
        meta = difficulties(src)
        env_info = None
        try:
            raw = src.read(meta["song"])
            fd, tmp = tempfile.mkstemp(suffix=".ogg")
            os.close(fd)
            with open(tmp, "wb") as fh:
                fh.write(raw)
            mono, sr = decode(tmp)
            os.unlink(tmp)
            if mono is not None:
                env, fr = onset_env(mono, sr)
                env_info = (env, fr, len(mono) / sr)
        except Exception as exc:  # audio is optional for the note metrics
            print("audio failed", job["label"], exc, file=sys.stderr)
        rec = {"label": job["label"], "bpm": meta["bpm"], "env": meta["env"], "offset_ms": meta["offset_ms"],
               "song_s": round(env_info[2], 1) if env_info else None, "diffs": []}
        for row in meta["rows"]:
            r = analyse(src, meta, row, env_info, job.get("window"))
            if r:
                rec["diffs"].append(r)
        results.append(rec)
        print("done", job["label"], [d["diff"] for d in rec["diffs"]], file=sys.stderr)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
