"""Make a map for one song in config/batch2_songs.json: any difficulties, its own environment and colours, and lighting that follows its mood.

    python map_song_v2.py "Nice For What"            refuses to write into an existing folder
    python map_song_v2.py "Nice For What" --replace  rebuilds, only if the folder still matches this tool's last build

It needs the song's stems in analysis/stems_songs, from separate_stems.py. It uses the song file at the top of CustomWIPLevels, or the copy in the map folder once that's gone.
The grid follows the beat tracker, so hand-played songs that drift stay on the beat. Always playtest: the user's ear overrides every number here.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import signal

import grid_tools
import guides
import lights_v2
import metrics
import notes_v2
import rhythm_model_v2
import vocal_lines
from mapfiles import PROJECT, WIP
from rhythm_features_v2 import NAMES, stems_at

CONFIG = PROJECT / "config" / "batch2_songs.json"
LEVEL_ORDER = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
# Expert+ is built by Expert's model and rules (there is no Expert+ model), a little faster: the user set 16 for their own MY EYES Expert+
BASE_LEVEL = {"ExpertPlus": "Expert"}
LEVEL_CHANGES = {"ExpertPlus": {"njs": 16.0}}
RANK = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}
SABERS = {"saberAColor": "C81414FF", "saberBColor": "288ED2FF", "obstaclesColor": "FF3030FF"}
PLACES = 4


def jump_offset(base_bpm, njs, reaction_s):
    """The note jump offset that gives this reaction time."""
    beat_s = 60 / base_bpm
    half_jump = 4.0
    while njs * beat_s * half_jump > 17.999:
        half_jump /= 2
    return round(reaction_s / beat_s - half_jump, 3)


def table(rows):
    """Split rows into the v4 item list and its shared data list."""
    data, index, items = [], {}, []
    for beat, meta in rows:
        key = json.dumps(meta, sort_keys=True)
        if key not in index:
            index[key] = len(data)
            data.append(meta)
        items.append({"b": beat, "i": index[key]})
    return items, data


def build_regions(fitted, regions, duration, rate, per_pulse):
    """Tempo regions in map beats: a lead-in, the steady runs, and an outro. Returns the regions, and how many map beats the lead-in takes."""
    period = (fitted[-1] - fitted[0]) / (len(fitted) - 1) / per_pulse
    lead_beats = max(1, int(round(fitted[0] / period)))
    out = [{"si": 0, "ei": int(round(fitted[0] * rate)), "sb": 0, "eb": lead_beats}]
    for first, last, _, _ in regions:
        out.append({"si": int(round(fitted[first] * rate)), "ei": int(round(fitted[last] * rate)), "sb": lead_beats + first * per_pulse, "eb": lead_beats + last * per_pulse})
    tail = max(1, int(round((duration - fitted[-1]) / period)))
    out.append({"si": out[-1]["ei"], "ei": int(round(duration * rate)) - 1, "sb": out[-1]["eb"], "eb": out[-1]["eb"] + tail})
    return out, lead_beats


def beat_to_seconds(beats, regions, rate):
    """Map beats back to song seconds, the way the game reads the tempo regions."""
    ends = [r["sb"] for r in regions] + [regions[-1]["eb"]]
    samples = [r["si"] for r in regions] + [regions[-1]["ei"]]
    return np.interp(beats, ends, samples) / rate


def download_cover(url, out):
    subprocess.run(["curl", "-s", "-L", "-A", "map-tool", "-o", str(out), url], check=True)
    if out.read_bytes()[:3] != b"\xff\xd8\xff":
        sys.exit(f"the cover from {url} isn't a JPEG")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def by_value(path, markers=True):
    """A map file's content, ignoring what ChroMapper changes when it only re-saves: number formatting, list order, and its own editor notes.
    With markers=False, red dot guide markers (guides.py) are left out too."""
    def plain(value):
        if isinstance(value, (bool, str)) or value is None:
            return value
        if isinstance(value, (int, float)):
            return round(float(value), 3) + 0.0
        if isinstance(value, list):
            return sorted((plain(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
        return {k: plain(v) for k, v in sorted(value.items()) if k not in ("customData", "songDuration")}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not markers:
        data = guides.without_markers(data)
    if "colorBoostEvents" in data:
        last = {}
        for e in data["colorBoostEvents"]:
            last[round(float(e["b"]), 3)] = data["colorBoostEventsData"][e.get("i", 0)].get("b")
        data["colorBoostEvents"] = [[b, v] for b, v in sorted(last.items())]
        data.pop("colorBoostEventsData", None)
    # arcs and chains point into the note-data table, which ChroMapper renumbers when it saves, so they're compared by what they point at.
    # ChroMapper also writes a blue arc's tail with the red colour, so a tail's colour is left out
    notes_data = data.get("colorNotesData", [])
    pointed = lambda i: notes_data[i] if 0 <= i < len(notes_data) else {"missing": i}
    if "arcs" in data:
        arc_data = data.get("arcsData") or [{}]
        data["arcs"] = [{"hb": a.get("hb", 0), "tb": a.get("tb", 0), "hr": a.get("hr", 0), "tr": a.get("tr", 0), "head": pointed(a.get("hi", 0)),
                         "tail": {k: v for k, v in pointed(a.get("ti", 0)).items() if k != "c"}, "shape": arc_data[min(a.get("ai", 0), len(arc_data) - 1)]} for a in data["arcs"]]
        data.pop("arcsData", None)
    if "chains" in data:
        chain_data = data.get("chainsData") or [{}]
        data["chains"] = [{"hb": c.get("hb", 0), "tb": c.get("tb", 0), "hr": c.get("hr", 0), "tr": c.get("tr", 0), "head": pointed(c.get("i", 0)),
                           "shape": chain_data[min(c.get("ci", 0), len(chain_data) - 1)]} for c in data["chains"]]
        data.pop("chainsData", None)
    for items, table in (("colorNotes", "colorNotesData"), ("basicEvents", "basicEventsData")):
        if items in data:
            data[items] = [{"b": e["b"], **data[table][e.get("i", 0)]} for e in data[items]]
            data.pop(table)
    return plain(data)


def main():
    name = sys.argv[1]
    replace = "--replace" in sys.argv
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))[name]
    out = WIP / cfg["folder"]
    record = PROJECT / "versions" / "songs" / f"{cfg['folder']} v1"
    # a folder is only rebuilt if nobody has edited it since the last build
    if out.exists():
        if not replace:
            sys.exit(f"refusing to write into an existing folder: {out}")
        # red dots added since are guide markers (guides.py), not edits
        for kept in record.glob("*.dat"):
            current = out / kept.name
            if current.exists() and digest(current) != digest(kept) and by_value(current, markers=False) != by_value(kept, markers=False):
                sys.exit(f"{current} has changed since this tool built it. Check for hand edits before replacing it.")
    # the chosen levels' red dot markers, read before anything is overwritten, in seconds
    marked, mixed = guides.find(out, record, [l for l in LEVEL_ORDER if l in cfg["difficulties"]]) if out.exists() else ({}, {})
    if mixed:
        sys.exit(f"{out / next(iter(mixed.values()))} has red dot markers and other notes this tool didn't make, which a rebuild would replace. Remove the other notes, or make this map under another folder name.")
    raw = WIP / cfg["raw"]
    source = raw if raw.exists() else out / cfg["song_file"]
    stems = PROJECT / "analysis" / "stems_songs" / name
    audio, rate = metrics.decode(str(source))
    frames = len(audio)
    duration = frames / rate
    mono = signal.resample_poly(audio, grid_tools.SR, rate).astype(np.float32)
    # a song whose tempo changes lists its parts in tempo_sections, each tracked at its own tempo
    envelope = grid_tools.onset_envelope(mono)
    raw_beats = grid_tools.track_beats_sections(envelope, cfg["tempo_sections"]) if cfg.get("tempo_sections") else grid_tools.track_beats(envelope, cfg["pulse_bpm"])
    # where the tracker slipped off the beat, a steady tempo line that fits the sound better takes over (grid_tools.straighten); a song can turn this off
    if cfg.get("straighten", True) and cfg.get("version", 1) >= 2:
        raw_beats = grid_tools.straighten(raw_beats, envelope, cfg.get("tempo_sections"))
    fitted, steady = grid_tools.fitted_beats(raw_beats)
    per_beat, per_pulse = cfg["per_beat"], cfg["map_beats_per_pulse"]
    times = grid_tools.steps_from_beats(fitted, per_beat)
    measures = stems_at(stems, times, per_beat)
    offset_beat, _ = grid_tools.bar_phase(measures[:, NAMES.index("kick")] + measures[:, NAMES.index("bass")], per_beat)
    offset = offset_beat * per_beat
    vocal = np.load(stems / "vocals_22k.npy").astype(np.float32) / 32768
    lines = vocal_lines.find_lines(vocal, {"first_beat_s": float(fitted[0]), "last_grid_beat_s": float(fitted[-1])})
    spans = [(int(np.searchsorted(times, a)), min(int(np.searchsorted(times, b)), len(times) - 1)) for a, b in lines]
    regions, lead_beats = build_regions(fitted, steady, duration, rate, per_pulse)
    to_beat = lambda step: round(lead_beats + step / per_beat * per_pulse, PLACES)
    # the grid must read back as the times it was built from
    check = beat_to_seconds(np.array([to_beat(k) for k in range(0, len(times), per_beat)]), regions, rate)
    worst = np.abs(check - times[::per_beat]).max()
    if worst > 0.03:
        sys.exit(f"the tempo regions read back {worst * 1000:.0f} ms away from the beat grid")

    weights = json.loads((rhythm_model_v2.MODEL_FOLDER / "weights.json").read_text(encoding="utf-8"))
    base_bpm = round(float(np.median([60 / r[2] for r in steady for _ in range(r[1] - r[0])])) * per_pulse, 3)
    report = {"song": name, "source_bpm_regions": len(steady), "base_bpm": base_bpm, "steps_per_pulse": per_beat, "bar_offset_beats": offset_beat, "difficulties": {}}
    beatmaps, rows_info = {}, []
    # what Expert finds in this song sets the others' density, even when Expert isn't one of the song's difficulties
    expert = notes_v2.make_notes(rhythm_model_v2.score(measures, "Expert"), measures, times, per_beat, spans, notes_v2.DIFFICULTY["Expert"], notes_v2.DIFFICULTY["Expert"]["per_second"], weights["levels"]["Expert"]["floor"], offset)
    expert_moments = np.unique([n["step"] for n in expert])
    expert_rate = len(expert_moments) / (times[expert_moments[-1]] - times[expert_moments[0]])
    # a song can pin the reference its lower difficulties were built from, so later Expert changes don't move them
    expert_rate = cfg.get("expert_density", expert_rate)
    report["expert_density_found"] = round(float(expert_rate), 2)
    # the lighting's sections also tell the patterns how big each part is
    sections, energy = lights_v2.plan(stems, fitted, duration, cfg["style"])
    levels = np.zeros(len(times), int)
    for a, b, section_level in sections:
        levels[a * per_beat:b * per_beat + 1] = section_level
    for level in [l for l in LEVEL_ORDER if l in cfg["difficulties"]]:
        base = BASE_LEVEL.get(level, level)
        # a song may adjust a level's settings, e.g. {"Expert": {"combos": true, "max_rest_s": 0.6, "per_second": 4.0}} for combo areas; songs without it build as before.
        # Expert+ takes Expert's adjustments unless it has its own.
        # engine version 2 (Beat Saber Maker) also paces the notes by the song's calm and busy parts (models/pacing_v1)
        song_settings = cfg.get("settings", {})
        settings = {**notes_v2.DIFFICULTY[base], **LEVEL_CHANGES.get(level, {}), **({"pacing": True} if cfg.get("version", 1) >= 2 else {}), **song_settings.get(level, song_settings.get(base, {}))}
        scores = rhythm_model_v2.score(measures, base)
        floor = weights["levels"][base]["floor"]
        wanted = min(settings["per_second"], expert_rate * notes_v2.SHARE_OF_EXPERT[base]) * cfg.get("density", {}).get(level, 1.0)
        chains, arcs = [], []
        wants_patterns = cfg.get("patterns")
        # a guide level (red dot markers) always goes through the pattern engine, which keeps every marker
        guide = guides.plan(marked[level], times, per_beat) if level in marked else None
        if guide is not None or (wants_patterns is True and base in ("Normal", "Hard", "Expert")) or (isinstance(wants_patterns, list) and level in wants_patterns):
            made, chains, arcs = notes_v2.make_notes_v3(scores, measures, times, per_beat, spans, settings, wanted if base != "Expert" else settings["per_second"], floor, offset, levels, base, guide=guide)
        else:
            made = expert if level == "Expert" else notes_v2.make_notes(scores, measures, times, per_beat, spans, settings, wanted, floor, offset)
        meta_of = lambda n: {"x": n["x"], "y": n["y"], "c": n["c"], "d": n["d"], "a": 0}
        rows = sorted(((to_beat(n["step"]), meta_of(n)) for n in made), key=lambda r: (r[0], r[1]["c"], r[1]["y"]))
        items, data = table(rows)
        for item in items:
            item["r"] = 0
        index_of = {json.dumps(m, sort_keys=True): i for i, m in enumerate(data)}
        meta_index = lambda n: index_of[json.dumps(meta_of(n), sort_keys=True)]
        chain_items, chain_data = [], []
        for head, tail_step, tx, ty in chains:
            chain_data.append({"tx": tx, "ty": ty, "c": 4, "s": 0.5})
            chain_items.append({"hb": to_beat(made[head]["step"]), "tb": to_beat(tail_step), "hr": 0, "tr": 0, "i": meta_index(made[head]), "ci": len(chain_data) - 1})
        arc_items = [{"hb": to_beat(made[h]["step"]), "tb": to_beat(made[t]["step"]), "hr": 0, "tr": 0, "hi": meta_index(made[h]), "ti": meta_index(made[t]), "ai": 0} for h, t in arcs]
        njs = settings["njs"]
        jump = jump_offset(base_bpm, njs, settings["reaction_s"])
        beatmaps[f"{level}Standard.dat"] = {"version": "4.1.0", "colorNotes": items, "colorNotesData": data, "bombNotes": [], "bombNotesData": [], "arcs": arc_items, "arcsData": [{"m": 1.0, "tm": 1.0, "a": 0}] if arc_items else [], "chains": chain_items, "chainsData": chain_data, "obstacles": [], "obstaclesData": [], "njsEvents": [], "njsEventData": []}
        rows_info.append({"characteristic": "Standard", "difficulty": level, "beatmapAuthors": {"mappers": [], "lighters": []}, "environmentNameIdx": 0, "beatmapColorSchemeIdx": 0, "noteJumpMovementSpeed": njs, "noteJumpStartBeatOffset": jump, "beatmapDataFilename": f"{level}Standard.dat", "lightshowDataFilename": "Lightshow.dat"})
        moments = sorted({n["step"] for n in made})
        seconds = times[moments]
        report["difficulties"][level] = {"chains": len(chain_items), "arcs": len(arc_items), "notes": len(made), "moments": len(moments), "per_second": round(len(made) / (seconds[-1] - seconds[0]), 2), "first_note_s": round(float(seconds[0]), 2), "longest_rest_s": round(float(np.diff(seconds).max()), 2), "njs": njs, "offset": jump}
        if guide is not None:
            # how many markers got a note (a marker can only miss one if no hand could reach it)
            report["difficulties"][level]["guide"] = {**guide[2], "missed": len(guide[0] - set(moments))}

    events, boosts = lights_v2.generate(times, measures, per_beat, sections, cfg["style"], cfg["environment"], offset)
    light_rows = [(to_beat(step), {"t": lane, "i": value, "f": strength}) for step, lane, value, strength in events]
    lights, lights_data = table(light_rows)
    # one boost state per beat: the last one set there wins, as ChroMapper keeps only one
    latest = {}
    for step, on in boosts:
        latest[to_beat(step)] = 1 if on else 0
    boost_rows = sorted(latest.items())
    report["sections"] = [{"start_s": round(float(fitted[a]), 1), "level": level} for a, b, level in sections]
    # a separate Lightshow mode, its own tab in ChroMapper, with its own environment, colours and show, and no notes
    show_mode = cfg.get("lightshow_mode")
    environments, schemes = [cfg["environment"]], [{"useOverride": True, "overrideNotes": True, "overrideLights": True, "colorSchemeName": cfg["title"], **SABERS, **cfg["colors"]}]
    extra_files = {}
    if show_mode:
        environments.append(show_mode["environment"])
        schemes.append({"useOverride": True, "overrideNotes": True, "overrideLights": True, "colorSchemeName": f"{cfg['title']} lightshow", **SABERS, **show_mode["colors"]})
        show_sections, _ = lights_v2.plan(stems, fitted, duration, show_mode["style"], bar_offset=offset_beat)
        show_events, show_boosts = lights_v2.generate(times, measures, per_beat, show_sections, show_mode["style"], show_mode["environment"], offset)
        show_lights, show_data = table([(to_beat(step), {"t": lane, "i": value, "f": strength}) for step, lane, value, strength in show_events])
        latest_show = {}
        for step, on in show_boosts:
            latest_show[to_beat(step)] = 1 if on else 0
        extra_files["LightshowKaleidoscope.dat"] = {"version": "4.0.0", "basicEvents": show_lights, "basicEventsData": show_data, "colorBoostEvents": [{"b": b, "i": i} for b, i in sorted(latest_show.items())], "colorBoostEventsData": [{"b": 0}, {"b": 1}],
                                                    "waypoints": [], "waypointsData": [], "indexFilters": [], "eventBoxGroups": [], "lightColorEventBoxes": [], "lightColorEvents": [], "lightRotationEventBoxes": [], "lightRotationEvents": [],
                                                    "lightTranslationEventBoxes": [], "lightTranslationEvents": [], "fxEventBoxes": [], "floatFxEvents": [], "basicEventTypesWithKeywords": {"d": []}, "useNormalEventsAsCompatibleEvents": True}
        extra_files["ExpertLightshow.dat"] = {"version": "4.1.0", "colorNotes": [], "colorNotesData": [], "bombNotes": [], "bombNotesData": [], "arcs": [], "arcsData": [], "chains": [], "chainsData": [], "obstacles": [], "obstaclesData": [], "njsEvents": [], "njsEventData": []}
        rows_info.append({"characteristic": "Lightshow", "difficulty": "Expert", "beatmapAuthors": {"mappers": [], "lighters": []}, "environmentNameIdx": 1, "beatmapColorSchemeIdx": 1, "noteJumpMovementSpeed": 15.0, "noteJumpStartBeatOffset": 0,
                          "beatmapDataFilename": "ExpertLightshow.dat", "lightshowDataFilename": "LightshowKaleidoscope.dat"})
        report["lightshow_mode"] = {"environment": show_mode["environment"], "light_events": len(show_lights), "sections": [{"start_s": round(float(fitted[a]), 1), "level": level} for a, b, level in show_sections]}
    report["light_events"] = len(lights)
    biggest = max(sections, key=lambda s: (s[2], s[1] - s[0]))
    preview = max(0.0, round(float(fitted[biggest[0]]) - 1, 1))
    song_file = cfg["song_file"]
    info = {"version": "4.0.1", "song": {"title": cfg["title"], "subTitle": cfg["subtitle"], "author": cfg["artist"]},
            "audio": {"songFilename": song_file, "songDuration": round(duration, 3), "audioDataFilename": "AudioData.dat", "bpm": base_bpm, "lufs": 0, "previewStartTime": preview, "previewDuration": 10},
            "songPreviewFilename": song_file, "coverImageFilename": "cover.jpg", "environmentNames": environments,
            "colorSchemes": schemes,
            "difficultyBeatmaps": rows_info, "customData": {"editors": {"lastEditedBy": "AI Mapper", "AI Mapper": {"version": "0.4.0"}}}}
    audio_data = {"version": "4.0.0", "songChecksum": "", "songSampleCount": frames - 1, "songFrequency": rate, "bpmData": regions, "lufsData": []}
    lightshow = {"version": "4.0.0", "basicEvents": lights, "basicEventsData": lights_data, "colorBoostEvents": [{"b": b, "i": i} for b, i in boost_rows], "colorBoostEventsData": [{"b": 0}, {"b": 1}],
                 "waypoints": [], "waypointsData": [], "indexFilters": [], "eventBoxGroups": [], "lightColorEventBoxes": [], "lightColorEvents": [], "lightRotationEventBoxes": [], "lightRotationEvents": [],
                 "lightTranslationEventBoxes": [], "lightTranslationEvents": [], "fxEventBoxes": [], "floatFxEvents": [], "basicEventTypesWithKeywords": {"d": []}, "useNormalEventsAsCompatibleEvents": True}

    out.mkdir(parents=True, exist_ok=True)
    for file_name, data in (("Info.dat", info), ("AudioData.dat", audio_data), ("Lightshow.dat", lightshow), *beatmaps.items(), *extra_files.items()):
        (out / file_name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    if source != out / song_file:
        shutil.copy2(source, out / song_file)
    if not (out / "cover.jpg").exists():
        download_cover(cfg["cover"], out / "cover.jpg")
    names = ["Info.dat", song_file, "cover.jpg", "AudioData.dat", "Lightshow.dat", *beatmaps, *extra_files]
    with zipfile.ZipFile(out / f"{cfg['folder']}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file_name in names:
            archive.write(out / file_name, file_name)
    record.mkdir(parents=True, exist_ok=True)
    for stale in record.glob("*.dat"):
        if stale.name not in ("Info.dat", "AudioData.dat", "Lightshow.dat", *beatmaps, *extra_files):
            stale.unlink()
    for file_name in ("Info.dat", "AudioData.dat", "Lightshow.dat", *beatmaps, *extra_files):
        shutil.copy2(out / file_name, record / file_name)
    # the built guide levels hold notes now, not dots, so their markers are kept here for the next rebuild (levels not built this time keep theirs)
    saved_guides = json.loads((record / guides.SAVED).read_text(encoding="utf-8")) if (record / guides.SAVED).exists() else {}
    saved_guides.pop("about", None)
    guides.save(record, {**saved_guides, **marked})
    (record / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
