"""Guide levels: red dot notes the user places by hand in ChroMapper as a metronome, saying "a note here".

The user marked MY EYES' fast half this way in Expert+ (16 Nov 2026): a red dot on every quarter beat, with a few gaps left on purpose. When a level they rebuild
has red dots, every dot becomes a real note at that moment (the engine picks the hand, direction and position), and nothing else goes near them. A long stretch
with no dots at all (UNDONE_S or more) is a part they haven't done yet, so the engine fills it as usual; a shorter gap between dots is a rest they meant.

The dots are saved with the build (guides.json in the song's record folder), because the rebuilt level holds real notes instead of dots. Red dots added to the
built level later join the saved ones on the next rebuild.
"""
import json
from pathlib import Path

import numpy as np

RED, DOT = 0, 8
# a stretch with no dots this long, in seconds, hasn't been done yet. The user's own gaps on purpose in MY EYES were at most 1.2 s.
UNDONE_S = 8.0
# filled parts keep this many beats away from the dots, so they don't crowd the hand-marked part
EDGE_BEATS = 1.0
SAVED = "guides.json"


def is_marker(note):
    return note.get("c") == RED and note.get("d") == DOT


def notes_of(data):
    """Every colour note in a level file as {"b", "c", "d"}, from any of the v2, v3 and v4 formats."""
    if "_notes" in data:
        return [{"b": n["_time"], "c": n.get("_type"), "d": n.get("_cutDirection")} for n in data["_notes"] if n.get("_type") in (0, 1)]
    table = data.get("colorNotesData")
    out = []
    for n in data.get("colorNotes", []):
        meta = table[n.get("i", 0)] if table is not None else n
        out.append({"b": n.get("b", 0), "c": meta.get("c"), "d": meta.get("d")})
    return out


def without_markers(data):
    """The level with its red dots taken out, for comparing it with the last build."""
    data = json.loads(json.dumps(data))
    if "_notes" in data:
        data["_notes"] = [n for n in data["_notes"] if not (n.get("_type") == RED and n.get("_cutDirection") == DOT)]
        return data
    if "colorNotes" in data:
        table = data.get("colorNotesData")
        meta = (lambda n: table[n.get("i", 0)]) if table is not None else (lambda n: n)
        data["colorNotes"] = [n for n in data["colorNotes"] if not is_marker(meta(n))]
    return data


def beats_to_seconds(folder, beats):
    """Seconds for beats in the folder's map, read the way the game reads them: through AudioData.dat's tempo regions, or Info.dat's one tempo."""
    beats = np.asarray(beats, float)
    audio = folder / "AudioData.dat"
    if audio.exists():
        data = json.loads(audio.read_text(encoding="utf-8-sig"))
        regions, rate = data.get("bpmData") or [], data.get("songFrequency", 44100)
        if regions:
            ends = [r["sb"] for r in regions] + [regions[-1]["eb"]]
            samples = [r["si"] for r in regions] + [regions[-1]["ei"]]
            return np.interp(beats, ends, samples) / rate
    info = json.loads((folder / "Info.dat").read_text(encoding="utf-8-sig"))
    bpm = info.get("audio", {}).get("bpm") or info.get("_beatsPerMinute")
    return beats * 60.0 / float(bpm)


def find(folder, record, levels):
    """{level: sorted marker seconds} for the chosen levels that have red dots now or had them at the last build.
    Also returns {level: file name} for levels whose file has notes besides dots that this tool didn't make, which a rebuild would wipe out."""
    saved_file = record / SAVED
    saved = json.loads(saved_file.read_text(encoding="utf-8")) if saved_file.exists() else {}
    found, mixed = {}, {}
    for level in levels:
        name = f"{level}Standard.dat"
        times = list(saved.get(level, []))
        path = folder / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            notes = notes_of(data)
            dots = [n["b"] for n in notes if is_marker(n)]
            if dots:
                times += [float(t) for t in beats_to_seconds(folder, dots)]
                # notes besides the dots, in a level this tool never built, are the user's own work
                if len(dots) < len(notes) and not (record / name).exists():
                    mixed[level] = name
        if times:
            found[level] = sorted(set(round(t, 4) for t in times))
    return found, mixed


def save(record, guides):
    record.mkdir(parents=True, exist_ok=True)
    path = record / SAVED
    if guides:
        path.write_text(json.dumps({"about": "Red dot markers from the user's guide levels, in song seconds. Each becomes a note on rebuild.", **guides}, indent=1), encoding="utf-8")
    elif path.exists():
        path.unlink()


def plan(marker_seconds, times, per_beat):
    """Where the markers land on the step grid, and which other steps may still get notes (the undone parts).
    Returns (marker steps, allowed mask, report)."""
    marks = np.asarray(sorted(marker_seconds), float)
    step_s = float(np.median(np.diff(times)))
    beat_s = step_s * per_beat
    nearest = np.clip(np.searchsorted(times, marks), 1, len(times) - 1)
    nearest = np.where(np.abs(times[nearest - 1] - marks) <= np.abs(times[nearest] - marks), nearest - 1, nearest)
    off_grid = int(np.sum(np.abs(times[nearest] - marks) > 0.3 * step_s))
    steps = sorted(set(nearest.tolist()))
    allowed = np.zeros(len(times), bool)
    edges = [-np.inf, *marks.tolist(), np.inf]
    filled = []
    for a, b in zip(edges, edges[1:]):
        start, end = (times[0] if a == -np.inf else a), (times[-1] if b == np.inf else b)
        if end - start < UNDONE_S:
            continue
        low = start if a == -np.inf else a + EDGE_BEATS * beat_s
        high = end if b == np.inf else b - EDGE_BEATS * beat_s
        allowed |= (times >= low) & (times <= high)
        filled.append([round(float(start), 1), round(float(end), 1)])
    report = {"markers": len(marks), "on_grid_steps": len(steps), "off_grid": off_grid, "merged": len(marks) - len(steps), "filled_s": filled}
    return set(steps), allowed, report
