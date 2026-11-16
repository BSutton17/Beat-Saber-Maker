"""Pick a map's environment, colours and lighting style from the song and its cover.

Environment: a nearest-match model. Each song is measured (tempo, swing, how much of the sound is drums, bass, voice and other instruments, brightness, major or minor,
how much the loudness moves, how clearly the beat stands out) and compared with songs whose environment was chosen by hand: the earlier AI Mapper songs, and every
environment the user picks by hand in this app, which are saved to models/style_v1/your_picks.json so it learns their taste.
Colours: the cover's most vivid colours, kept vivid and never pale (pale boosts washed Like Him out to white), with a mood palette when the cover is grey.
"""
import colorsys
import json
from pathlib import Path

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "style_v1"
STEMS = ("drums", "bass", "vocals", "other")
SR = 22050
FEATURES = ("pulse", "swing", "drums", "bass", "vocals", "other", "brightness", "minor", "dynamics", "clarity")
# how much each measure counts when comparing songs
FEATURE_WEIGHT = {"pulse": 1.0, "swing": 1.2, "drums": 1.4, "bass": 1.0, "vocals": 0.6, "other": 1.0, "brightness": 0.8, "minor": 0.8, "dynamics": 0.7, "clarity": 1.0}

# the environments the lighting engine knows, with what each looks like, shared with the app's picker
ENVIRONMENTS = json.loads((HERE / "environments.json").read_text(encoding="utf-8"))


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def key_minor(mix):
    """How much more the song looks minor than major, from its chroma (Krumhansl-Kessler key profiles)."""
    f, _, spec = signal.stft(mix[::2], fs=SR / 2, nperseg=4096, noverlap=2048)
    power = np.abs(spec).mean(axis=1)
    keep = (f > 55) & (f < 2000)
    pitch = np.round(12 * np.log2(f[keep] / 440.0)).astype(int) % 12
    chroma = np.bincount(pitch, weights=power[keep], minlength=12)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    corr = lambda profile: max(np.corrcoef(chroma, np.roll(profile, k))[0, 1] for k in range(12))
    return float(corr(minor) - corr(major))


def song_features(stems_dir, pulse, per_beat):
    import grid_tools
    import tempo_auto
    stems_dir = Path(stems_dir)
    parts = {name: np.load(stems_dir / f"{name}_22k.npy").astype(np.float32) / 32768 for name in STEMS}
    n = min(len(p) for p in parts.values())
    mix = sum(p[:n] for p in parts.values())
    level = {name: rms(parts[name][:n]) for name in STEMS}
    total = sum(level.values())
    f, _, spec = signal.stft(mix[::4], fs=SR / 4, nperseg=1024)
    power = np.abs(spec).mean(axis=1)
    brightness = float((f * power).sum() / (power.sum() + 1e-9))
    window = SR * 2
    loud = np.array([20 * np.log10(rms(mix[i:i + window])) for i in range(0, n - window, window)])
    loud = loud[loud > loud.max() - 40]
    env = grid_tools.onset_envelope(mix)
    clarity = float(np.log(tempo_auto.comb(env, pulse)))
    return {"pulse": float(pulse), "swing": 1.0 if per_beat in (3, 6) else 0.0, **{name: level[name] / total for name in STEMS},
            "brightness": brightness, "minor": key_minor(mix), "dynamics": float(np.std(loud)), "clarity": clarity}


def examples():
    rows = json.loads((MODEL / "examples.json").read_text(encoding="utf-8"))["songs"]
    picks = MODEL / "your_picks.json"
    if picks.exists():
        rows += json.loads(picks.read_text(encoding="utf-8"))["songs"]
    return rows


def recommend_environment(features, rows=None):
    """Every environment ranked, nearest first, with the song that decided it and why."""
    rows = rows if rows is not None else examples()
    table = np.array([[r["features"][k] for k in FEATURES] for r in rows])
    spread = table.std(axis=0) + 1e-6
    weight = np.array([FEATURE_WEIGHT[k] for k in FEATURES])
    song = np.array([features[k] for k in FEATURES])
    distance = np.sqrt((((table - song) / spread * weight) ** 2).sum(axis=1))
    # the user's own picks count a little closer than the starting examples
    distance = distance * np.array([0.85 if r.get("source") == "you" else 1.0 for r in rows])
    best = {}
    for r, d in sorted(zip(rows, distance), key=lambda x: x[1]):
        best.setdefault(r["environment"], (float(d), r))
    ranked = sorted(best.items(), key=lambda x: x[1][0])
    # far from every example: the classic environment is the safe choice
    if ranked and ranked[0][1][0] > FAR and ranked[0][0] != "DefaultEnvironment":
        ranked = [("DefaultEnvironment", (ranked[0][1][0], None))] + [r for r in ranked if r[0] != "DefaultEnvironment"]
    out = []
    for env, (d, row) in ranked:
        if row:
            why = f"Closest to {row['song']}, which is on {ENVIRONMENTS[env]['name']}. This song is {describe(features)}."
        else:
            why = f"Not close to any song it knows (this one is {describe(features)}), so the classic environment."
        out.append({"environment": env, "distance": round(d, 2), "why": why})
    seen = {o["environment"] for o in out}
    out += [{"environment": env, "distance": None, "why": ""} for env in ENVIRONMENTS if env not in seen]
    return out


FAR = 6.0


def describe(f):
    words = ["fast" if f["pulse"] >= 88 else "slow" if f["pulse"] < 66 else "mid-tempo"]
    if f["swing"]:
        words.append("swung")
    if f["drums"] > 0.3:
        words.append("drum-heavy")
    elif f["drums"] < 0.16:
        words.append("light on drums")
    if f["bass"] > 0.3:
        words.append("bass-heavy")
    if f["other"] > 0.38:
        words.append("led by instruments")
    if f["minor"] > 0.05:
        words.append("minor-key")
    elif f["minor"] < -0.05:
        words.append("major-key")
    if f["brightness"] > 1500:
        words.append("bright")
    elif f["brightness"] < 900:
        words.append("dark-toned")
    return ", ".join(words)


def drive(f):
    """0 for a soft ballad, 1 for a hard-hitting track."""
    pulse = np.clip((f["pulse"] - 55) / 45, 0, 1)
    drums = np.clip((f["drums"] - 0.12) / 0.25, 0, 1)
    clarity = np.clip((f["clarity"] - 0.6) / 1.4, 0, 1)
    return float(0.25 * pulse + 0.45 * drums + 0.3 * clarity)


def light_style(f):
    """The lighting engine's style settings for this song: soft songs glow and breathe, hard ones flash. Beat-locked, as the user asked: lights follow the beat, or the voice where there are no drums, never a mix."""
    d = drive(f)
    style = {"soft": d < 0.62, "rate": round(0.6 + 0.5 * d, 2), "speed": round(0.5 + 0.5 * d, 2), "glow": 0.7, "highest": 4, "colorful": d >= 0.62, "beat_locked": True}
    if d < 0.32:
        style.update({"highest": 3, "boost": False, "lowest": 1})
    return style


def hexcolor(rgb):
    return "".join(f"{int(round(max(0, min(1, c)) * 255)):02X}" for c in rgb) + "FF"


def vivid(h, s, v, min_s=0.7, min_v=0.85):
    return colorsys.hsv_to_rgb(h % 1.0, max(s, min_s), max(v, min_v))


def cover_hues(cover_path):
    """The cover's vivid colours as (hue, saturation, value, share), most present first."""
    from PIL import Image
    img = Image.open(cover_path).convert("RGB").resize((80, 80))
    px = np.asarray(img, dtype=np.float32).reshape(-1, 3) / 255
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px])
    keep = (hsv[:, 1] > 0.3) & (hsv[:, 2] > 0.25)
    if keep.mean() < 0.04:
        return []
    hsv = hsv[keep]
    bins = 24
    weight = hsv[:, 1] * hsv[:, 2]
    which = (hsv[:, 0] * bins).astype(int) % bins
    hist = np.bincount(which, weights=weight, minlength=bins)
    hist = np.convolve(np.concatenate([hist[-1:], hist, hist[:1]]), [0.25, 0.5, 0.25], mode="valid")
    out, used = [], []
    for b in np.argsort(hist)[::-1]:
        if hist[b] <= 0.03 * hist.sum():
            break
        if any(min(abs(b - o), bins - abs(b - o)) < 2 for o in used):
            continue
        sel = which == b
        if not sel.any():
            continue
        used.append(int(b))
        out.append((float(np.median(hsv[sel, 0])), float(np.median(hsv[sel, 1])), float(np.median(hsv[sel, 2])), float(hist[b] / hist.sum())))
    return out


MOOD_HUES = {"dark": (0.68, 0.78), "warm": (0.07, 0.95), "party": (0.9, 0.13), "cool": (0.55, 0.75)}
SABERS = {"saberAColor": "C81414FF", "saberBColor": "288ED2FF", "obstaclesColor": "FF3030FF"}


def palette(cover_path, f):
    """Two light colours and their boosts from the cover, with the default red and blue sabers. Returns the colours and where they came from."""
    hues = cover_hues(cover_path) if cover_path else []
    d = drive(f)
    if hues:
        h0, s0, v0, _ = hues[0]
        # the second colour: the next vivid colour far enough round the wheel to tell apart
        second = next(((h, s, v) for h, s, v, _ in hues[1:] if min(abs(h - h0), 1 - abs(h - h0)) >= 0.12), None)
        if second is None:
            second = (h0 + (0.5 if d >= 0.55 else 0.14), s0, v0)
        h1, s1, v1 = second
        source = "the cover's colours"
    else:
        mood = "party" if d >= 0.6 else "dark" if f["minor"] > 0.03 else "warm" if f["other"] > 0.35 else "cool"
        h0, h1 = MOOD_HUES[mood]
        s0 = s1 = 0.85
        v0 = v1 = 0.9
        source = f"a {mood} mood palette, because the cover has no strong colours"
    # boosts move each colour round the wheel and push it brighter, never paler; harder songs move further for more contrast
    turn = 0.04 + 0.06 * d
    colors = {"environmentColor0": hexcolor(vivid(h0, s0, v0)), "environmentColor1": hexcolor(vivid(h1, s1, v1)),
              "environmentColor0Boost": hexcolor(vivid(h0 - turn, s0, v0, 0.8, 0.95)), "environmentColor1Boost": hexcolor(vivid(h1 + turn, s1, v1, 0.8, 0.95)), **SABERS}
    return colors, source


def save_pick(song, features, environment):
    """Remember an environment the user chose by hand, so later recommendations learn from it."""
    picks = MODEL / "your_picks.json"
    data = json.loads(picks.read_text(encoding="utf-8")) if picks.exists() else {"about": "Environments you picked by hand in Beat Saber Maker. The recommender learns from these.", "songs": []}
    data["songs"] = [r for r in data["songs"] if r["song"] != song] + [{"song": song, "environment": environment, "features": features, "source": "you"}]
    picks.write_text(json.dumps(data, indent=1), encoding="utf-8")
