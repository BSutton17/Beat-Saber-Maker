"""Learn which sounds people choose to place notes on, per difficulty and across genres, from well-rated maps and the user's own.

Only timing is learned, so no mapper's patterns can be copied. The same method as v1: every map is tested on a model trained without it.

Usage:
  python rhythm_model_v2.py extract <refmaps folder>   measure the reference songs and save the results
  python rhythm_model_v2.py evaluate                   test each map on models trained without it, against v1
  python rhythm_model_v2.py save                       write the trained models to models/rhythm_v2
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy import signal
from sklearn.metrics import roc_auc_score

import grid_tools
import metrics
from mapfiles import PROJECT, WIP
from rhythm_features_v2 import NAMES, NAMES_V1, stems_at

FEATURES_FILE = PROJECT / "analysis" / "ref2_features.npz"
META_FILE = PROJECT / "analysis" / "ref2_meta.json"
MODEL_FOLDER = PROJECT / "models" / "rhythm_v2"
LEVELS = ("Easy", "Normal", "Hard", "Expert")
# the user's own maps: never the songs being mapped. Their Pick Up the Phone is only used as a test.
YOURS = {"quebec": ("Quebec", "stems_refs/quebec"), "love_galore": ("Love Galore", "stems_refs/love_galore"), "chemicals": ("Chemicals", "stems_refs/chemicals"), "eleven_eleven": ("Eleven Eleven", "stems_refs/eleven_eleven")}
TEST_ONLY_YOURS = {"pick_up_the_phone": ("Travis Scott Concert", "stems_songs/Pick Up the Phone")}
TEST_ONLY_GROUPS = {"test_same_song"}
YOUR_WEIGHT = 2.0
# notes this far from the sound get their measurements taken where they really are
LARGE_SHIFT_S = 0.04
# a map whose notes don't sit on any one grid can't teach timing
MIN_ON_GRID = 0.55
# notes aren't forced onto steps scoring below what this share of human notes score. Forcing notes past the model's confidence caused the worst builds.
# Lower difficulties are held under Expert's density instead, so they only need a low floor to avoid truly empty moments.
FLOOR_PERCENTILE = {"Easy": 5, "Normal": 5, "Hard": 5, "Expert": 10}


def grid_fit(beats, per_map_beat):
    """How well notes fit a grid, allowing for a constant shift, since many mappers' grids don't start on a whole beat. Returns (share on the grid, shift in steps)."""
    steps = np.asarray(beats) * per_map_beat
    angle = 2 * np.pi * (steps % 1.0)
    phase = (np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()) / (2 * np.pi)) % 1.0
    error = np.abs((steps - phase + 0.5) % 1.0 - 0.5)
    return float(np.mean(error < 0.12)), float(phase)


def subdivision(beats, bpm):
    """Choose the grid from where the notes actually sit. Returns (steps per map beat, per_beat for the flags, is it triplet, share on the grid, shift in steps)."""
    pulse = bpm
    while pulse >= 115:
        pulse /= 2
    straight = 4 * pulse / bpm
    share, phase = grid_fit(beats, straight)
    best = (straight, 4, False, share, phase)
    for per_map_beat, per_beat in ((6.0, 6), (3.0, 3), (1.5, 3), (0.75, 3)):
        if 60 / bpm / per_map_beat < 0.13:
            continue
        triple_share, triple_phase = grid_fit(beats, per_map_beat)
        # a triplet grid has more places for notes to land, so it has to win clearly
        if triple_share > best[3] + 0.12:
            best = (per_map_beat, per_beat, True, triple_share, triple_phase)
        break
    return best


def sound_shift(song, note_times):
    """If a map's notes sit noticeably away from the sound, measure the sound where the notes really are. Only the notes' own times can tell: on repeating music, every step looks like a beat."""
    audio, rate = metrics.decode(str(song))
    mono = signal.resample_poly(audio, grid_tools.SR, rate).astype(np.float32)
    env = grid_tools.onset_envelope(mono)
    shifts = np.arange(-0.12, 0.121, 0.01)
    place = lambda t: np.clip((t * grid_tools.FPS).astype(int), 0, len(env) - 1)
    strength = [env[place(note_times - grid_tools.CALIBRATION_S + s)].mean() for s in shifts]
    best = float(shifts[int(np.argmax(strength))])
    return best if abs(best) >= LARGE_SHIFT_S else 0.0


def map_grid(path):
    """Steps and note labels for each difficulty of one map, through any tempo changes."""
    src = metrics.Source(path)
    meta = metrics.difficulties(src)
    rows = {}
    for row in meta["rows"]:
        rows[row["name"]] = row
    # Expert+ stands in for Expert only if there's no Expert
    if "Expert" not in rows and "ExpertPlus" in rows:
        rows["Expert"] = rows["ExpertPlus"]
    loaded = {name: metrics.load_diff(src, rows[name]) for name in LEVELS if name in rows}
    if not loaded:
        return None
    hardest = loaded[[n for n in LEVELS if n in loaded][-1]]
    changes = hardest[6]
    if any(m <= 0 for _, m in changes):
        return None
    bpm = meta["bpm"]
    every_beat = sorted({n[0] for name in loaded for n in loaded[name][0]})
    if not every_beat:
        return None
    per_map_beat, per_beat, triple, on_grid, phase = subdivision([n[0] for n in hardest[0]], bpm)
    first = max(0, int(min(every_beat) * per_map_beat) - 8)
    last = int(np.ceil(max(every_beat) * per_map_beat)) + 8
    steps = np.arange(first, last)
    conv = metrics.time_map(bpm, changes)
    times = np.array([float(conv((s + phase) / per_map_beat)) for s in steps]) + meta["offset_ms"] / 1000
    labels = {}
    stats = {}
    for name, loaded_diff in loaded.items():
        notes = loaded_diff[0]
        y = np.zeros(len(steps), bool)
        for b in {n[0] for n in notes}:
            i = int(round(b * per_map_beat - phase)) - first
            if 0 <= i < len(y):
                y[i] = True
        labels[name] = y
        stats[name] = {"njs": rows[name]["njs"], "offset": rows[name]["offset"]}
    return {"times": times, "labels": labels, "per_beat": per_beat, "triple": triple, "bpm": bpm, "per_map_beat": per_map_beat, "stats": stats, "bpm_changes": len(changes), "on_grid": on_grid, "phase": phase, "song": meta["song"]}


def extract(root):
    root = Path(root)
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    jobs = [(k, str(root / k / "map.zip"), str(root / k / "stems"), v["group"], f"{v['song']} ({v['mapper']})", 1.0) for k, v in catalog.items()]
    jobs += [(k, str(WIP / v[0]), str(PROJECT / "analysis" / v[1]), "yours", v[0], YOUR_WEIGHT) for k, v in YOURS.items()]
    jobs += [(k, str(WIP / v[0]), str(PROJECT / "analysis" / v[1]), "test_yours", "your Pick Up the Phone", 0.0) for k, v in TEST_ONLY_YOURS.items()]
    store, meta = {}, {}
    for key, path, stems, group, name, weight in jobs:
        if not (Path(stems) / "vocals_22k.npy").exists():
            print("no stems yet, skipping:", key, name, flush=True)
            continue
        grid = map_grid(path)
        if grid is None:
            print("no usable grid:", key, name, flush=True)
            continue
        # a test map is kept whatever its grid, since nothing learns from it
        if grid["on_grid"] < MIN_ON_GRID and weight > 0 and group not in TEST_ONLY_GROUPS:
            print(f"notes don't sit on one grid ({grid['on_grid'] * 100:.0f}%), skipping: {key} {name}", flush=True)
            continue
        song = Path(path).parent / "song.ogg" if path.endswith(".zip") else Path(path) / grid["song"]
        hardest = [level for level in LEVELS if level in grid["labels"]][-1]
        shift = sound_shift(song, grid["times"][grid["labels"][hardest]])
        grid["times"] = grid["times"] + shift
        keep = grid["times"] >= 0
        x = stems_at(stems, grid["times"][keep], grid["per_beat"])
        store[f"{key}|x"] = x
        store[f"{key}|t"] = grid["times"][keep].astype(np.float32)
        for level, y in grid["labels"].items():
            store[f"{key}|{level}"] = y[keep]
        meta[key] = {"group": group, "name": name, "weight": weight, "per_beat": grid["per_beat"], "triple": grid["triple"], "bpm": grid["bpm"], "per_map_beat": grid["per_map_beat"], "stats": grid["stats"], "bpm_changes": grid["bpm_changes"], "levels": list(grid["labels"]), "on_grid": grid["on_grid"], "shift_s": shift}
        step_ms = np.median(np.diff(grid["times"][keep])) * 1000
        print(f"{key}: {name}, {group}, bpm {grid['bpm']}, step {step_ms:.0f} ms{' triplet' if grid['triple'] else ''}, on grid {grid['on_grid'] * 100:.0f}%, shift {shift * 1000:.0f} ms, " + ", ".join(f"{level} {y[keep].mean() * 100:.0f}%" for level, y in grid["labels"].items()), flush=True)
    np.savez_compressed(FEATURES_FILE, **store)
    META_FILE.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load():
    return np.load(FEATURES_FILE), json.loads(META_FILE.read_text(encoding="utf-8"))


def train_keys(meta):
    return [k for k, v in meta.items() if v["group"] not in TEST_ONLY_GROUPS and v["group"] != "test_yours"]


def fit(data, meta, keys, level, columns, kind="linear"):
    rows = [k for k in keys if level in meta[k]["levels"]]
    x = np.vstack([data[f"{k}|x"][:, columns] for k in rows])
    y = np.concatenate([data[f"{k}|{level}"] for k in rows])
    # each map counts the same however long it is, and the user's count double because their taste is the target
    w = np.concatenate([np.full(len(data[f"{k}|{level}"]), meta[k]["weight"] * 1000 / len(data[f"{k}|{level}"])) for k in rows])
    if kind == "linear":
        m = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    else:
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.05, l2_regularization=1.0)
    m.fit(x, y, sample_weight=w)
    return m


def v1_scores(x):
    v1 = json.loads((PROJECT / "models" / "rhythm_v1" / "weights.json").read_text(encoding="utf-8"))
    return x[:, :len(NAMES_V1)] @ np.array(v1["coefficients"]) + v1["intercept"]


def evaluate():
    data, meta = load()
    all_columns = list(range(len(NAMES)))
    keys = train_keys(meta)
    print("Each map scored by a model that never saw it: 0.50 is guessing, 1.00 is perfect.\n")
    results = {}
    for level in LEVELS:
        print(f"=== {level}")
        print(f"{'map':<52} {'group':<15} {'v1':>6} {'v2':>6} {'v2 no-vocal':>12}")
        for left in keys:
            if level not in meta[left]["levels"]:
                continue
            others = [k for k in keys if k != left]
            x, y = data[f"{left}|x"], data[f"{left}|{level}"]
            if y.sum() < 10 or (~y).sum() < 10:
                continue
            model = fit(data, meta, others, level, all_columns)
            score = model.predict_proba(x)[:, 1]
            old = roc_auc_score(y, v1_scores(x))
            new = roc_auc_score(y, score)
            quiet = x[:, NAMES.index("vocal_active")] < 0.3
            no_vocal = roc_auc_score(y[quiet], score[quiet]) if quiet.sum() > 30 and 5 < y[quiet].sum() < quiet.sum() - 5 else float("nan")
            results.setdefault(level, []).append((meta[left]["group"], old, new, no_vocal, left))
            print(f"{meta[left]['name'][:50]:<52} {meta[left]['group']:<15} {old:>6.3f} {new:>6.3f} {no_vocal:>12.3f}")
        rows = results.get(level, [])
        if rows:
            for group in sorted({r[0] for r in rows}):
                part = [r for r in rows if r[0] == group]
                print(f"  average {group:<15} n={len(part):<3} v1 {np.mean([r[1] for r in part]):.3f}  v2 {np.mean([r[2] for r in part]):.3f}")
            print(f"  average all             n={len(rows):<3} v1 {np.mean([r[1] for r in rows]):.3f}  v2 {np.mean([r[2] for r in rows]):.3f}\n")
    print("=== never used in training")
    for key, entry in meta.items():
        if key in keys:
            continue
        for level in entry["levels"]:
            x, y = data[f"{key}|x"], data[f"{key}|{level}"]
            model = fit(data, meta, keys, level, all_columns)
            print(f"{entry['name'][:44]:<46} {level:<7} v1 {roc_auc_score(y, v1_scores(x)):.3f}  v2 {roc_auc_score(y, model.predict_proba(x)[:, 1]):.3f}")
    print("\nWhat the Expert model weighs:")
    model = fit(data, meta, keys, "Expert", all_columns)
    scale = np.vstack([data[f"{k}|x"] for k in keys]).std(axis=0)
    for name, c in sorted(zip(NAMES, model.coef_[0] * scale), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<18} {c:+.2f}")


def goosebumps_measures():
    """The user's favourite section: its notes from the main map, measured on the concert's stems."""
    folder = WIP / "Travis Scott Fortnite Concert AI"
    audio = json.loads((folder / "AudioData.dat").read_text(encoding="utf-8"))
    regions = audio["bpmData"]
    marks = [regions[0]["sb"]] + [r["eb"] for r in regions]
    samples = [regions[0]["si"]] + [r["ei"] for r in regions]
    beats = np.arange(399, 470, 0.25)
    times = np.interp(beats, marks, samples) / audio["songFrequency"]
    notes = json.loads((folder / "ExpertStandard.dat").read_text(encoding="utf-8"))["colorNotes"]
    placed = {round(n["b"] * 4) for n in notes if 399 <= n["b"] < 470}
    labels = np.array([round(b * 4) in placed for b in beats])
    return stems_at(str(PROJECT / "analysis" / "stems"), times, 4), labels, times


def save():
    data, meta = load()
    keys = train_keys(meta)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    out = {"name": "rhythm_v2", "feature_names": NAMES, "step": "one subdivision of the beat", "trained_on": {k: meta[k]["name"] for k in keys}, "levels": {}}
    goose_x, goose_y, goose_t = goosebumps_measures()
    for level in LEVELS:
        model = fit(data, meta, keys, level, list(range(len(NAMES))))
        rows = [k for k in keys if level in meta[k]["levels"]]
        placed = np.concatenate([model.predict_proba(data[f"{k}|x"])[:, 1][data[f"{k}|{level}"]] for k in rows])
        floor = float(np.percentile(placed, FLOOR_PERCENTILE[level]))
        out["levels"][level] = {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "maps": len(rows), "floor": floor}
        scores = model.predict_proba(goose_x)[:, 1]
        print(f"{level}: trained on {len(rows)} maps, floor {floor:.3f}; on the user's loved Goosebumps notes: {roc_auc_score(goose_y, scores):.3f}, with {np.mean(scores > floor) * 100:.0f}% of steps above the floor ({np.sum(scores > floor) / (goose_t[-1] - goose_t[0]):.2f} a second; the map has {goose_y.sum() / (goose_t[-1] - goose_t[0]):.2f})")
    print(f"v1 on the same Goosebumps notes: {roc_auc_score(goose_y, v1_scores(goose_x)):.3f}")
    (MODEL_FOLDER / "weights.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", MODEL_FOLDER / "weights.json")


def score(measures, level):
    """How likely a mapper at this difficulty is to place a note on each step. Needs only NumPy."""
    weights = json.loads((MODEL_FOLDER / "weights.json").read_text(encoding="utf-8"))["levels"][level]
    raw = np.asarray(measures, dtype=np.float64) @ np.array(weights["coefficients"]) + weights["intercept"]
    return 1 / (1 + np.exp(-raw))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "extract":
        extract(sys.argv[2])
    elif command == "evaluate":
        evaluate()
    elif command == "save":
        save()
