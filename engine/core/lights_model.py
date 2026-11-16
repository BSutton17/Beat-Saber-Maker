"""Learn how good lighters follow a song's energy, from well-rated maps and the user's own.

It measures how bright the show is and how busy it is, at each level of a song's own energy, which lights do the work, what kinds of events they use, and how fast they react
when a song suddenly gets bigger. Only these overall habits are kept, so no one's lightshow can be copied.

Usage:
  python lights_model.py extract <refmaps folder>   measure the lighting of every reference map
  python lights_model.py report                     print what good lighters do, and save it to models/lights_v1
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal, stats

import metrics
from mapfiles import PROJECT, WIP
from rhythm_features import band_flux
from rhythm_features_v2 import rms_curve

HOP_S = 0.1
FEATURES_FILE = PROJECT / "analysis" / "lights_measures.json"
MODEL_FOLDER = PROJECT / "models" / "lights_v1"
LIGHT_LANES = (0, 1, 2, 3, 4, 6, 7, 10, 11)
FADE_S = 1.0
YOURS = {"quebec": ("Quebec", "stems_refs/quebec"), "love_galore": ("Love Galore", "stems_refs/love_galore"), "chemicals": ("Chemicals", "stems_refs/chemicals"), "eleven_eleven": ("Eleven Eleven", "stems_refs/eleven_eleven")}
ENERGY_BINS = 5
MIN_EVENTS = 150
SERIES_FILE = PROJECT / "analysis" / "lights_series.npz"
PART_NAMES = ["bright_loudness", "brightness", "instruments", "vocals", "drums", "drum_hits", "bass"]
SMOOTH_S = 2.0


def song_energy(stems, duration):
    """How big the music is, about every 0.1 s: overall loudness, how much the drums are in, and how full the music under the voice is."""
    load = lambda name: np.load(Path(stems) / f"{name}_22k.npy").astype(np.float32) / 32768
    vocal, drums, bass, other = load("vocals"), load("drums"), load("bass"), load("other")
    length = min(map(len, (vocal, drums, bass, other)))
    grid = np.arange(0, min(duration, length / 22050), HOP_S)
    mix_db, times = rms_curve(vocal[:length] + drums[:length] + bass[:length] + other[:length], 1.0)
    drum_db, _ = rms_curve(drums[:length], 1.0)
    band_db, _ = rms_curve(bass[:length] + other[:length], 1.0)
    top = np.percentile(mix_db, 95)
    return {"grid": grid, "mix": np.clip(np.interp(grid, times, mix_db - top) / 40 + 1, 0, 1.1), "drums": np.clip(np.interp(grid, times, drum_db - top) / 40 + 1, 0, 1.1), "band": np.clip(np.interp(grid, times, band_db - top) / 40 + 1, 0, 1.1)}


def combined(energy):
    """The energy measure used everywhere: mostly loudness, with extra weight on drums coming in."""
    return 0.5 * energy["mix"] + 0.3 * energy["drums"] + 0.2 * energy["band"]


def intensity_parts(stems, duration):
    """What might make a part feel big, about every 0.1 s, each measured against the song's own typical level.

    Plain loudness can't do this: Slow Dancing In the Dark's quiet verses have the heaviest bass, so they measure as loud, while its big parts are brighter and fuller.
    """
    load = lambda name: np.load(Path(stems) / f"{name}_22k.npy").astype(np.float32) / 32768
    vocal, drums, bass, other = load("vocals"), load("drums"), load("bass"), load("other")
    length = min(map(len, (vocal, drums, bass, other)))
    vocal, drums, bass, other = vocal[:length], drums[:length], bass[:length], other[:length]
    grid = np.arange(0, min(duration, length / 22050), HOP_S)
    mix = vocal + drums + bass + other
    freqs, frame_times, spec = signal.stft(mix, fs=22050, nperseg=2048, noverlap=2048 - 512, boundary=None, padded=False)
    power = np.abs(spec) ** 2
    bright_db = 10 * np.log10(power[freqs >= 300].sum(axis=0) + 1e-10)
    centroid = (power * freqs[:, None]).sum(axis=0) / (power.sum(axis=0) + 1e-9)
    size = max(1, int(SMOOTH_S * 22050 / 512))
    smooth = lambda x: signal.fftconvolve(x, np.ones(size) / size, mode="same")
    parts = [np.interp(grid, frame_times, smooth(bright_db)), np.interp(grid, frame_times, np.log(smooth(centroid) + 1))]
    for stem in (other, vocal, drums):
        db, times = rms_curve(stem, SMOOTH_S)
        parts.append(np.interp(grid, times, np.maximum(db, -70)))
    hits = band_flux(drums, 0, 11000)
    hit_times = np.arange(len(hits)) * 128 / 22050
    parts.append(np.interp(grid, hit_times, signal.fftconvolve((hits > 0.5).astype(float), np.ones(int(SMOOTH_S * 22050 / 128)) / int(SMOOTH_S * 22050 / 128), mode="same")))
    db, times = rms_curve(bass, SMOOTH_S)
    parts.append(np.interp(grid, times, np.maximum(db, -70)))
    out = np.column_stack(parts)
    middle, spread = np.median(out, axis=0), np.percentile(out, 90, axis=0) - np.percentile(out, 10, axis=0) + 1e-6
    return np.clip((out - middle) / spread, -3, 3), grid


def learned_intensity(parts, weights):
    return parts @ np.asarray(weights)


def brightness(events, conv, grid):
    """How lit the show is at each moment, from 0 to 1, by following each light through its events."""
    lanes = {}
    for beat, lane, value, strength in events:
        if lane in LIGHT_LANES:
            lanes.setdefault(lane, []).append((float(conv(beat)), value, strength if strength is not None else 1.0))
    if not lanes:
        return np.zeros(len(grid))
    total = np.zeros(len(grid))
    for lane, rows in lanes.items():
        rows.sort()
        level = np.zeros(len(grid))
        times = np.array([r[0] for r in rows])
        where = np.searchsorted(times, grid, side="right") - 1
        for i, w in enumerate(where):
            if w < 0:
                continue
            t, value, strength = rows[w]
            age = grid[i] - t
            s = min(float(strength), 1.5)
            if value == 0:
                level[i] = 0
            elif value in (1, 5, 9, 4, 8, 12):
                level[i] = s
            elif value in (2, 6, 10):
                level[i] = s * max(0.7, 1.2 - age)
            elif value in (3, 7, 11):
                level[i] = s * max(0.0, 1 - age / FADE_S)
        total += level
    return np.clip(total / max(5, len(lanes)), 0, 1.2)


def rate_curve(events, conv, grid, lanes):
    """Events per second, smoothed over 2 seconds."""
    times = np.array([float(conv(e[0])) for e in events if e[1] in lanes])
    if len(times) == 0:
        return np.zeros(len(grid))
    counts = np.histogram(times, bins=np.append(grid, grid[-1] + HOP_S))[0].astype(float) / HOP_S
    size = int(2.0 / HOP_S)
    return np.convolve(counts, np.ones(size) / size, mode="same")


def measure(path, stems, name, group, weight):
    src = metrics.Source(path)
    meta = metrics.difficulties(src)
    rows = {r["name"]: r for r in meta["rows"]}
    choice = next((rows[n] for n in ("Expert", "ExpertPlus", "Hard", "Normal", "Easy") if n in rows), None)
    if choice is None:
        return None
    notes, events, *_ , changes = metrics.load_diff(src, choice)
    lights = [e for e in events if e[1] in LIGHT_LANES]
    if len(lights) < MIN_EVENTS:
        return None
    conv = metrics.time_map(meta["bpm"], changes)
    duration = float(conv(max(e[0] for e in events)))
    energy = song_energy(stems, duration + 2)
    grid = energy["grid"]
    value = combined(energy)
    lit = brightness(events, conv, grid)
    rate = rate_curve(events, conv, grid, LIGHT_LANES)
    parts, part_grid = intensity_parts(stems, duration + 2)
    count = min(len(part_grid), len(lit))
    series = {"parts": parts[:count:5].astype(np.float32), "lit": lit[:count:5].astype(np.float32), "rate": rate[:count:5].astype(np.float32)}
    edges = np.quantile(value, np.linspace(0, 1, ENERGY_BINS + 1))
    which = np.clip(np.searchsorted(edges, value, side="right") - 1, 0, ENERGY_BINS - 1)
    by_lit = [float(lit[which == b].mean()) if (which == b).any() else 0.0 for b in range(ENERGY_BINS)]
    by_rate = [float(rate[which == b].mean()) if (which == b).any() else 0.0 for b in range(ENERGY_BINS)]
    # how they react: find where the music gets a lot bigger, and read the show around those moments
    size = int(4 / HOP_S)
    before = np.convolve(value, np.ones(size) / size, mode="full")[:len(value)]
    after = np.convolve(value, np.ones(size) / size, mode="full")[size - 1:][:len(value)]
    rise = after - before
    moments = []
    threshold = max(0.1, np.percentile(rise, 97))
    i = int(8 / HOP_S)
    while i < len(rise) - int(8 / HOP_S):
        if rise[i] >= threshold and rise[i] == rise[max(0, i - 40):i + 40].max():
            moments.append(i)
            i += int(16 / HOP_S)
        else:
            i += 1
    reach = int(6 / HOP_S)
    around_lit = [lit[m - reach:m + reach + 1] for m in moments if m - reach >= 0 and m + reach + 1 <= len(lit)]
    around_rate = [rate[m - reach:m + reach + 1] for m in moments if m - reach >= 0 and m + reach + 1 <= len(rate)]
    # the first time the show reaches halfway between its level before and after a rise
    lags = []
    for m in moments:
        if m - reach < 0 or m + reach >= len(lit):
            continue
        base = lit[m - 30:m - 5].mean()
        top = lit[m + 5:m + 40].mean()
        if top - base < 0.05:
            continue
        crossing = np.flatnonzero(lit[m - 20:m + 40] >= base + 0.5 * (top - base))
        if len(crossing):
            lags.append((crossing[0] - 20) * HOP_S)
    values = [e[2] for e in lights]
    kinds = {"off": sum(v == 0 for v in values), "on": sum(v in (1, 5, 9) for v in values), "flash": sum(v in (2, 6, 10) for v in values), "fade": sum(v in (3, 7, 11) for v in values), "transition": sum(v in (4, 8, 12) for v in values)}
    lanes = {str(l): sum(e[1] == l for e in lights) for l in LIGHT_LANES}
    return {"name": name, "group": group, "weight": weight, "events": len(lights), "duration": duration, "seconds": duration, "energy_vs_lit": float(stats.spearmanr(value, lit).statistic), "energy_vs_rate": float(stats.spearmanr(value, rate).statistic),
            "lit_by_energy": by_lit, "rate_by_energy": by_rate, "kinds": kinds, "lanes": lanes, "around_lit": np.mean(around_lit, axis=0).tolist() if around_lit else None, "around_rate": np.mean(around_rate, axis=0).tolist() if around_rate else None,
            "lags": lags, "moments": len(moments), "rings_per_min": sum(e[1] == 8 for e in events) * 60 / duration, "zooms_per_min": sum(e[1] == 9 for e in events) * 60 / duration, "laser_speed_per_min": sum(e[1] in (12, 13) for e in events) * 60 / duration, "boost_per_min": sum(e[1] == 5 for e in events) * 60 / duration,
            "strength_mean": float(np.mean([e[3] if e[3] is not None else 1.0 for e in lights if e[2] != 0])), "dynamic_range": float(max(by_lit) - min(by_lit)), "series": series}


def extract(root):
    root = Path(root)
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    results = {}
    arrays = {}
    jobs = [(k, str(root / k / "map.zip"), str(root / k / "stems"), f"{v['song']} ({v['mapper']})", v["group"], 1.0) for k, v in catalog.items() if (root / k / "stems" / "vocals_22k.npy").exists()]
    jobs += [(k, str(WIP / v[0]), str(PROJECT / "analysis" / v[1]), v[0], "yours", 2.0) for k, v in YOURS.items()]
    for key, path, stems, name, group, weight in jobs:
        row = measure(path, stems, name, group, weight)
        if row is None:
            print("too few basic light events, skipped:", name, flush=True)
            continue
        for name_, values_ in row.pop("series").items():
            arrays[f"{key}|{name_}"] = values_
        results[key] = row
        print(f"{name[:44]:<46} {group:<15} lights {row['events']:>5}  energy vs brightness {row['energy_vs_lit']:+.2f}  energy vs busyness {row['energy_vs_rate']:+.2f}  rises {row['moments']}", flush=True)
    FEATURES_FILE.write_text(json.dumps(results), encoding="utf-8")
    np.savez_compressed(SERIES_FILE, **arrays)


def standard(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def fit_intensity(arrays, keys, weights):
    """The mix of measures that best predicts how bright and busy human lighters make each moment."""
    x = np.vstack([arrays[f"{k}|parts"] for k in keys])
    y = np.concatenate([standard(arrays[f"{k}|lit"]) + standard(arrays[f"{k}|rate"]) for k in keys])
    w = np.concatenate([np.full(len(arrays[f"{k}|lit"]), weights[k] * 200 / len(arrays[f"{k}|lit"])) for k in keys])
    scale = np.sqrt(w)[:, None]
    return np.linalg.solve((x * scale).T @ (x * scale) + 1.0 * np.eye(x.shape[1]), (x * scale).T @ (y * scale[:, 0]))


def test_intensity(rows):
    """Each song's lights predicted by a recipe learned without it, against plain loudness and against each measure alone."""
    arrays = np.load(SERIES_FILE)
    keys = [k for k, v in rows.items() if f"{k}|parts" in arrays]
    weights = {k: rows[k]["weight"] for k in keys}
    print(f"{'map':<46} {'group':<15} {'recipe':>7} {'loudness':>9}")
    scores, plain = [], []
    for left in keys:
        recipe = fit_intensity(arrays, [k for k in keys if k != left], weights)
        target = standard(arrays[f"{left}|lit"]) + standard(arrays[f"{left}|rate"])
        ours = stats.spearmanr(arrays[f"{left}|parts"] @ recipe, target).statistic
        old = rows[left]["energy_vs_lit"]
        scores.append(ours)
        plain.append(old)
        print(f"{rows[left]['name'][:44]:<46} {rows[left]['group']:<15} {ours:>7.2f} {old:>9.2f}")
    print(f"median, 1.00 is a perfect match: learned recipe {np.median(scores):.2f}, plain loudness {np.median(plain):.2f}")
    for i, name in enumerate(PART_NAMES):
        print(f"  {name:<16} alone: {np.median([stats.spearmanr(arrays[f'{k}|parts'][:, i], standard(arrays[f'{k}|lit']) + standard(arrays[f'{k}|rate'])).statistic for k in keys]):+.2f}")
    recipe = fit_intensity(arrays, keys, weights)
    print("the recipe:", {n: round(float(c), 3) for n, c in zip(PART_NAMES, recipe)})
    return recipe, float(np.median(scores)), float(np.median(plain))


def report():
    rows = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
    rows = {k: v for k, v in rows.items() if v["group"] != "test_same_song"}
    recipe, recipe_score, plain_score = test_intensity(rows)
    weights = np.array([v["weight"] for v in rows.values()])
    average = lambda field: np.average([v[field] for v in rows.values()], axis=0, weights=weights)
    print(f"{len(rows)} maps\n")
    print("energy vs brightness, each song's correlation: median", round(float(np.median([v["energy_vs_lit"] for v in rows.values()])), 2), "; energy vs busyness: median", round(float(np.median([v["energy_vs_rate"] for v in rows.values()])), 2))
    print("how lit, quietest fifth to loudest fifth:", np.round(average("lit_by_energy"), 3))
    print("events per second, quietest fifth to loudest fifth:", np.round(average("rate_by_energy"), 2))
    # each song against its own average, so loud shows don't swamp quiet ones
    relative = np.array([np.array(v["rate_by_energy"]) / (np.mean(v["rate_by_energy"]) + 1e-9) for v in rows.values()])
    print("busyness relative to the song's own average:", np.round(np.average(relative, axis=0, weights=weights), 2))
    relative_lit = np.array([np.array(v["lit_by_energy"]) / (np.mean(v["lit_by_energy"]) + 1e-9) for v in rows.values()])
    print("brightness relative to the song's own average:", np.round(np.average(relative_lit, axis=0, weights=weights), 2))
    rates = np.array([v["events"] / v["seconds"] for v in rows.values()])
    print("light events per second over a whole song: 25/50/75%", np.round(np.percentile(rates, [25, 50, 75]), 2))
    kinds = {k: sum(v["kinds"][k] * v["weight"] for v in rows.values()) for k in ("off", "on", "flash", "fade", "transition")}
    total = sum(kinds.values())
    print("what kinds of events:", {k: round(n / total, 3) for k, n in kinds.items()})
    lanes = {}
    for v in rows.values():
        for lane, n in v["lanes"].items():
            lanes[lane] = lanes.get(lane, 0) + n * v["weight"] / v["events"]
    all_lanes = sum(lanes.values())
    print("which lights do the work:", {k: round(n / all_lanes, 3) for k, n in lanes.items()})
    for field in ("rings_per_min", "zooms_per_min", "laser_speed_per_min", "boost_per_min", "strength_mean"):
        print(f"{field}: 25/50/75%", np.round(np.percentile([v[field] for v in rows.values()], [25, 50, 75]), 2))
    around = [np.array(v["around_rate"]) / (np.mean(v["around_rate"]) + 1e-9) for v in rows.values() if v["around_rate"]]
    around_lit = [np.array(v["around_lit"]) / (np.mean(v["around_lit"]) + 1e-9) for v in rows.values() if v["around_lit"]]
    print(f"\naround the moment a song gets bigger, from {len(around)} songs, each second from -6 s to +6 s:")
    print("  busyness:", np.round(np.mean(around, axis=0)[::10], 2))
    print("  brightness:", np.round(np.mean(around_lit, axis=0)[::10], 2))
    lags = np.concatenate([v["lags"] for v in rows.values()]) if any(v["lags"] for v in rows.values()) else np.array([])
    print(f"how long the show takes to reach halfway up, from {len(lags)} rises: median {np.median(lags):+.1f} s, 25-75% {np.percentile(lags, 25):+.1f} to {np.percentile(lags, 75):+.1f} s")
    groups = sorted({v["group"] for v in rows.values()})
    for group in groups:
        part = [v for v in rows.values() if v["group"] == group]
        print(f"  {group:<16} n={len(part):<3} busyness {np.round(np.mean([np.array(v['rate_by_energy']) for v in part], axis=0), 2)}  rate {np.mean([v['events'] / v['seconds'] for v in part]):.2f}/s  energy vs brightness {np.median([v['energy_vs_lit'] for v in part]):+.2f}")
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    model = {"name": "lights_v1", "maps": {k: v["name"] for k, v in rows.items()}, "relative_rate_by_energy": np.average(relative, axis=0, weights=weights).tolist(), "relative_lit_by_energy": np.average(relative_lit, axis=0, weights=weights).tolist(), "median_events_per_second": float(np.median(rates)),
             "events_per_second_25_75": np.percentile(rates, [25, 75]).tolist(), "kinds": {k: n / total for k, n in kinds.items()}, "lanes": {k: n / all_lanes for k, n in lanes.items()}, "rings_per_min": float(np.median([v["rings_per_min"] for v in rows.values()])), "zooms_per_min": float(np.median([v["zooms_per_min"] for v in rows.values()])), "laser_speed_per_min": float(np.median([v["laser_speed_per_min"] for v in rows.values()])), "reaction_lag_s": float(np.median(lags)) if len(lags) else None, "intensity_parts": PART_NAMES, "intensity_weights": recipe.tolist(), "left_out_match": recipe_score, "plain_loudness_match": plain_score}
    (MODEL_FOLDER / "weights.json").write_text(json.dumps(model, indent=1), encoding="utf-8")
    print("\nsaved", MODEL_FOLDER / "weights.json")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    elif sys.argv[1] == "report":
        report()
