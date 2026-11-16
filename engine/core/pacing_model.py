"""Pacing model: how busy a map should be in each part of a song, relative to the song's own average, learned from human maps.

rhythm_v2 says which steps are likely notes, but the builder took one note budget for the whole song and filled every rest past max_rest_s, so calm parts got
as many notes as loud ones. The user found MY EYES (Travis Scott) "wayyyy too busy" in its mellow first half. This learns, from the human maps rhythm_v2 trained
on, how note density rises and falls with how busy the sound is: syllables, drum hits, other instruments, loudness, each against the song's own average.
Density is counted per beat, so a faster part gets more notes a second without anything learned for it.

    python pacing_model.py add <map zip> <stems folder> <key> <name>   measure one more human map (kept in analysis/pacing_extra.npz)
    python pacing_model.py evaluate                                      each map scored by a model trained without it
    python pacing_model.py save                                          writes models/pacing_v1/weights.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from rhythm_features_v2 import NAMES

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = ROOT / "analysis" / "ref2_features.npz"
META_FILE = ROOT / "analysis" / "ref2_meta.json"
EXTRA_FILE = ROOT / "analysis" / "pacing_extra.npz"
EXTRA_META = ROOT / "analysis" / "pacing_extra.json"
MODEL_FOLDER = ROOT / "models" / "pacing_v1"
WINDOW_BEATS = 8
HOP_BEATS = 2
RIDGE = 3.0
# which measures the saved model uses. Loudness against the song's own average won a held-out comparison on 36 human maps: median fit +0.52 against +0.45 to
# +0.47 for richer sets (voice, drums, all 11), and worse than a flat budget on 10 maps against 13 to 14. One number learned from 36 maps can't overfit one song.
USE = ["energy_mean"]
# a step counts as busy in each part of the sound when its measure passes these
BUSY = {"syllable": 0.3, "vocal_onset": 0.5, "kick": 0.5, "snare_hat": 0.5, "other": 0.5, "bass": 0.5, "chord_change": 0.5}
LEVELS_MEANS = ("energy", "drum_level", "vocal_active", "vocal_level")


def use_columns(rel, names):
    return rel[:, [names.index(c) for c in USE]]


def window_features(x, per_beat):
    """For windows of WINDOW_BEATS beats across the song: (window centre steps, features, names). Each feature is a share of busy steps or a mean level,
    compared with the song's own typical value, so a part is busy for this song whatever the genre."""
    size, hop = WINDOW_BEATS * per_beat, HOP_BEATS * per_beat
    n = len(x)
    columns = np.array([(x[:, NAMES.index(k)] > v).astype(float) for k, v in BUSY.items()] + [x[:, NAMES.index(k)] for k in LEVELS_MEANS]).T
    names = [f"{k}_rate" for k in BUSY] + [f"{k}_mean" for k in LEVELS_MEANS]
    if n <= size:
        centres = np.array([n // 2])
    else:
        centres = np.arange(size // 2, n - size // 2 + 1, hop)
    rows = np.array([columns[max(0, c - size // 2):c + size // 2].mean(axis=0) for c in centres])
    whole = columns.mean(axis=0)
    return centres, np.log((rows + 0.05) / (whole + 0.05)), names


def window_density(labels, centres, per_beat):
    size = WINDOW_BEATS * per_beat
    return np.array([labels[max(0, c - size // 2):c + size // 2].mean() for c in centres])


def song_rows(x, t, y, per_beat):
    hits = np.flatnonzero(y)
    if len(hits) < 20:
        return None
    # only where the map plays, from a little before its first note to its last
    a, b = max(0, hits[0] - 2 * per_beat), min(len(y), hits[-1] + 2 * per_beat + 1)
    x, y = x[a:b], y[a:b]
    centres, rel, names = window_features(x, per_beat)
    rel = use_columns(rel, names)
    density = window_density(y, centres, per_beat)
    mean = y.mean()
    target = np.log((density + 0.04) / (mean + 0.04))
    return rel, target


def fit(rows):
    x = np.vstack([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    # every map counts the same however long it is
    w = np.concatenate([np.full(len(r[1]), 1.0 / len(r[1])) for r in rows])
    xm = np.hstack([x, np.ones((len(x), 1))])
    a = xm.T @ (xm * w[:, None]) + RIDGE * np.diag([1.0] * x.shape[1] + [0.0]) / len(rows)
    return np.linalg.solve(a, xm.T @ (y * w))


def predict(coef, rel):
    return np.hstack([rel, np.ones((len(rel), 1))]) @ coef


def load():
    """Every human map: key -> (x, t, labels by level, per_beat, meta)."""
    data = np.load(FEATURES_FILE)
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    songs = {}
    for k, v in meta.items():
        if v["group"] == "test_same_song":
            continue
        songs[k] = (data[f"{k}|x"], data[f"{k}|t"], {lv: data[f"{k}|{lv}"] for lv in v["levels"]}, v["per_beat"], v)
    if EXTRA_FILE.exists():
        extra = np.load(EXTRA_FILE)
        for k, v in json.loads(EXTRA_META.read_text(encoding="utf-8")).items():
            songs[k] = (extra[f"{k}|x"], extra[f"{k}|t"], {lv: extra[f"{k}|{lv}"] for lv in v["levels"]}, v["per_beat"], v)
    return songs


def collect(songs, levels=("Hard", "Expert")):
    out = {}
    for k, (x, t, labels, per_beat, v) in songs.items():
        for level in levels:
            if level in labels:
                r = song_rows(x, t, labels[level], per_beat)
                if r is not None:
                    out[(k, level)] = r
    return out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0


def trainable(v):
    return v["group"] != "test_yours"


def evaluate():
    songs = load()
    rows = collect(songs)
    names = window_features(next(iter(songs.values()))[0], 4)[2]
    results = []
    for (k, level), r in rows.items():
        # never trained on the map it's scored on
        coef = fit([v for (kk, _), v in rows.items() if kk != k and trainable(songs[kk][4])])
        pred = predict(coef, r[0])
        loud = r[0][:, USE.index("energy_mean")]
        results.append((songs[k][4]["name"], level, corr(pred, r[1]), corr(loud, r[1])))
        print(f"{songs[k][4]['name'][:42]:42s} {level:6s} model {results[-1][2]:+.2f}   loudness only {results[-1][3]:+.2f}")
    model = np.array([r[2] for r in results])
    loud = np.array([r[3] for r in results])
    print(f"\nmedian correlation with the human map's density, part by part: model {np.median(model):+.2f}, loudness only {np.median(loud):+.2f}, a flat budget 0.00")
    print(f"maps where the model beats loudness: {int((model > loud).sum())} of {len(model)}")
    coef = fit([v for (kk, _), v in rows.items() if trainable(songs[kk][4])])
    print("weights:", {n: round(float(c), 3) for n, c in zip(USE + ["bias"], coef)})
    return results


def save():
    songs = load()
    rows = collect(songs)
    coef = fit([v for (kk, _), v in rows.items() if trainable(songs[kk][4])])
    names = window_features(next(iter(songs.values()))[0], 4)[2]
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    (MODEL_FOLDER / "weights.json").write_text(json.dumps({"window_beats": WINDOW_BEATS, "hop_beats": HOP_BEATS, "busy": BUSY, "features": USE, "coef": [float(c) for c in coef],
                                                          "trained_on": sorted({songs[k][4]["name"] for (k, _) in rows if trainable(songs[k][4])})}, indent=1), encoding="utf-8")
    print("saved", MODEL_FOLDER / "weights.json")


def pacing(x, per_beat):
    """For a new song: how many times the song's average density each step should get, from the saved model. Mean about 1."""
    model = json.loads((MODEL_FOLDER / "weights.json").read_text(encoding="utf-8"))
    centres, rel, names = window_features(x, per_beat)
    rel = rel[:, [names.index(c) for c in model["features"]]]
    curve = np.exp(predict(np.array(model["coef"]), rel))
    return np.interp(np.arange(len(x)), centres, curve)


def add(zip_path, stems, key, name):
    import rhythm_model_v2
    from rhythm_features_v2 import stems_at
    grid = rhythm_model_v2.map_grid(zip_path)
    song = Path(zip_path).parent / "song.ogg"
    hardest = [lv for lv in rhythm_model_v2.LEVELS if lv in grid["labels"]][-1]
    shift = rhythm_model_v2.sound_shift(song, grid["times"][grid["labels"][hardest]])
    times = grid["times"] + shift
    keep = times >= 0
    x = stems_at(stems, times[keep], grid["per_beat"])
    store = dict(np.load(EXTRA_FILE)) if EXTRA_FILE.exists() else {}
    meta = json.loads(EXTRA_META.read_text(encoding="utf-8")) if EXTRA_META.exists() else {}
    store[f"{key}|x"] = x
    store[f"{key}|t"] = times[keep].astype(np.float32)
    for level, y in grid["labels"].items():
        store[f"{key}|{level}"] = y[keep]
    meta[key] = {"group": "rap", "name": name, "per_beat": grid["per_beat"], "levels": list(grid["labels"]), "bpm_changes": grid["bpm_changes"], "shift_s": shift}
    np.savez_compressed(EXTRA_FILE, **store)
    EXTRA_META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"added {key}: {name}, per_beat {grid['per_beat']}, levels {list(grid['labels'])}, tempo changes {grid['bpm_changes']}, shift {shift * 1000:.0f} ms, steps {int(keep.sum())}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "add":
        add(*sys.argv[2:6])
    else:
        {"evaluate": evaluate, "save": save}[command]()
