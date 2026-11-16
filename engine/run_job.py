"""The one entry point the app calls. It reads a job file and prints one JSON object per line: progress, then a result or an error.

    python run_job.py <job.json>

A job is {"mode": "analyze" | "build", "name": map folder name, "title", "subtitle", "artist", "audio": .ogg path, "cover": image path,
          "difficulties": [...], "environment": "auto" | name, "colors": "auto" | {...}, "tempo": "auto" | {...},
          "patterns": bool, "combos": "auto" | bool, "lightshow": bool, "replace": bool}

analyze: separates the stems (cached), finds the tempo, and recommends an environment, colours and lighting style.
build: does the same, then writes the map with the current builder (CURRENT.json) into the output folder, ChroMapper's CustomWIPLevels.
"""
import contextlib
import hashlib
import io
import json
import runpy
import shutil
import sys
import traceback
from pathlib import Path

import boot  # noqa: F401  (must come first)
import audio_io
import mapfiles
import style_engine
import tempo_auto

HERE = Path(__file__).resolve().parent
LEVELS = ("Easy", "Normal", "Hard", "Expert")
COMBO_SETTINGS = {"combos": True, "max_rest_s": 0.6, "per_second": 4.0}


def emit(kind, **data):
    # straight to the real output, even while the engine's own prints are being caught
    sys.__stdout__.write(json.dumps({"type": kind, **data}) + "\n")
    sys.__stdout__.flush()


def progress(stage, message, pct=None):
    emit("progress", stage=stage, message=message, pct=pct)


class LineWriter(io.TextIOBase):
    """Catches what the engine prints, so it can't break the one-JSON-per-line output; each line goes to a handler."""
    def __init__(self, handle):
        self.handle, self.buffer, self.lines = handle, "", []

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.lines.append(line)
            self.handle(line)
        return len(text)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key_of(job):
    """The engine's name for a song: its stems and settings are kept under this, so renaming the map doesn't redo the slow stem split."""
    return job.get("song_id") or job["name"]


def stems_for(job):
    """Split the song into vocals, drums, bass and other with Demucs, unless this exact audio was split before."""
    folder = HERE / "analysis" / "stems_songs" / key_of(job)
    stamp = folder / "source.sha256"
    audio_hash = file_hash(job["audio"])
    if stamp.exists() and stamp.read_text().strip() == audio_hash and all((folder / f"{s}_22k.npy").exists() for s in style_engine.STEMS):
        progress("stems", "Using the stems from last time", 100)
        return folder, audio_hash
    progress("stems", "Loading the stem separator (Demucs). The first run downloads its model, about 300 MB", 0)
    import separate_stems
    separate_stems.decode = audio_io.decode_channels

    def on_line(line):
        if line.startswith("chunk "):
            done, total = line.split()[1].rstrip(",").split("/")
            progress("stems", f"Separating vocals, drums, bass and instruments: part {done} of {total}", round(100 * int(done) / int(total)))
    old_argv = sys.argv
    sys.argv = ["separate_stems.py", job["audio"], str(folder)]
    try:
        with contextlib.redirect_stdout(LineWriter(on_line)):
            separate_stems.main()
    finally:
        sys.argv = old_argv
    stamp.write_text(audio_hash)
    return folder, audio_hash


def tempo_for(job, folder, audio_hash):
    if isinstance(job.get("tempo"), dict):
        t = job["tempo"]
        return {"pulse_bpm": float(t["pulse_bpm"]), "per_beat": int(t["per_beat"]), "map_beats_per_pulse": int(t.get("map_beats_per_pulse", 2 if int(t["per_beat"]) == 6 else 1)), "source": "yours"}
    cache = folder / "tempo.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        # a tempo found by an older detector is found again (version 2 finds tempo changes, like MY EYES at 2:26)
        if saved.get("audio") == audio_hash and saved.get("detector") == tempo_auto.VERSION:
            return saved["tempo"]
    progress("tempo", "Finding the tempo and beat")
    import numpy as np
    from scipy import signal
    import grid_tools
    mono, rate = audio_io.decode(job["audio"])
    found = tempo_auto.detect(signal.resample_poly(mono, grid_tools.SR, rate).astype(np.float32))
    found["source"] = "detected"
    cache.write_text(json.dumps({"audio": audio_hash, "detector": tempo_auto.VERSION, "tempo": found}))
    return found


def analyze(job):
    folder, audio_hash = stems_for(job)
    tempo = tempo_for(job, folder, audio_hash)
    progress("style", "Choosing the environment and colours")
    features = style_engine.song_features(folder, tempo["pulse_bpm"], tempo["per_beat"])
    colors, color_source = style_engine.palette(job.get("cover"), features)
    drive = style_engine.drive(features)
    return {"tempo": tempo, "features": features, "description": style_engine.describe(features), "drive": round(drive, 2),
            "environments": style_engine.recommend_environment(features), "colors": colors, "color_source": color_source,
            "light_style": style_engine.light_style(features), "combos_auto": drive >= 0.62}


def write_cover(source, target):
    """A square 512 px JPEG, cropped from the middle, as Beat Saber expects."""
    from PIL import Image
    img = Image.open(source).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img.crop((left, top, left + side, top + side)).resize((512, 512), Image.LANCZOS).save(target, "JPEG", quality=92)


def keep_jump_settings(out, name, kept, report):
    """Put the user's own jump speed and offset back on the rebuilt difficulties, then rewrite Info.dat in the zip too."""
    import zipfile
    info_path = out / "Info.dat"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for row in info["difficultyBeatmaps"]:
        njs, offset = kept.get((row["characteristic"], row["difficulty"]), (None, None))
        if njs is not None:
            row["noteJumpMovementSpeed"] = njs
        if offset is not None:
            row["noteJumpStartBeatOffset"] = offset
            if row["characteristic"] == "Standard" and row["difficulty"] in report.get("difficulties", {}):
                report["difficulties"][row["difficulty"]].update({"njs": row["noteJumpMovementSpeed"], "offset": offset, "kept_your_jump": True})
    info_path.write_text(json.dumps(info, separators=(",", ":")), encoding="utf-8")
    package = out / f"{name}.zip"
    if package.exists():
        with zipfile.ZipFile(package) as old:
            names = old.namelist()
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
            for file_name in names:
                archive.write(out / file_name, file_name)


def build(job):
    found = analyze(job)
    name = job["name"]
    out = mapfiles.WIP / name
    record = HERE / "versions" / "songs" / f"{name} v1"
    # never touch a folder this program didn't make: it could be one of the user's own maps
    if out.exists() and any(out.iterdir()):
        if not record.exists():
            raise UserError(f'A map folder called "{name}" already exists in {mapfiles.WIP} and wasn\'t made by Beat Saber Maker. Pick another name so it isn\'t overwritten.')
        if not job.get("replace"):
            raise UserError(f'"{name}" already exists from an earlier build. Rebuild it from its own song under Recent songs, or pick another folder name.')
    environment = found["environments"][0]["environment"] if job.get("environment", "auto") == "auto" else job["environment"]
    if environment not in style_engine.ENVIRONMENTS:
        raise UserError(f"Unknown environment {environment}")
    colors = found["colors"] if job.get("colors", "auto") == "auto" else {**style_engine.SABERS, **job["colors"]}
    style = found["light_style"]
    combos = found["combos_auto"] if job.get("combos", "auto") == "auto" else bool(job["combos"])
    difficulties = [level for level in LEVELS if level in job["difficulties"]]
    if not difficulties:
        raise UserError("Pick at least one difficulty")
    tempo = found["tempo"]
    cfg = {"title": job["title"], "subtitle": job.get("subtitle", ""), "artist": job.get("artist", ""), "raw": str(Path(job["audio"]).resolve()), "song_file": "song.ogg",
           "folder": name, "pulse_bpm": tempo["pulse_bpm"], "per_beat": tempo["per_beat"], "map_beats_per_pulse": tempo["map_beats_per_pulse"], "difficulties": difficulties,
           "cover": "", "environment": environment, "colors": colors, "style": style, "patterns": bool(job.get("patterns", True)),
           "mood": found["description"], "made_by": "Beat Saber Maker",
           # engine version 2: notes paced by the song's calm and busy parts, and the beat grid straightened where the tracker slipped
           "version": 2}
    if tempo.get("tempo_sections"):
        cfg["tempo_sections"] = tempo["tempo_sections"]
    if combos and "Expert" in difficulties:
        cfg["settings"] = {"Expert": COMBO_SETTINGS}
    if job.get("lightshow"):
        cfg["lightshow_mode"] = {"environment": "KaleidoscopeEnvironment", "colors": colors, "style": {**style, "colorful": True, "beat_locked": True}}
    config_file = HERE / "config" / "batch2_songs.json"
    config_file.parent.mkdir(exist_ok=True)
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    config[key_of(job)] = cfg
    config_file.write_text(json.dumps(config, indent=1), encoding="utf-8")
    if job.get("environment", "auto") != "auto":
        style_engine.save_pick(job["title"], found["features"], environment)

    created = not out.exists()
    # jump speed and offset set by hand in ChroMapper are kept through a rebuild. Info.dat is rewritten from these settings anyway, so a change there alone
    # (jump settings, or ChroMapper's own re-save) doesn't stop the rebuild; edited notes or lights still do
    kept_jumps = {}
    if not created and (out / "Info.dat").exists() and (record / "Info.dat").exists():
        old_info = json.loads((out / "Info.dat").read_text(encoding="utf-8-sig"))
        kept_jumps = {(r["characteristic"], r["difficulty"]): (r.get("noteJumpMovementSpeed"), r.get("noteJumpStartBeatOffset")) for r in old_info.get("difficultyBeatmaps", [])}
        shutil.copy2(record / "Info.dat", out / "Info.dat")
    out.mkdir(parents=True, exist_ok=True)
    progress("build", "Placing notes and lights with the current model")
    if job.get("cover"):
        write_cover(job["cover"], out / "cover.jpg")
    elif not (out / "cover.jpg").exists():
        from PIL import Image
        Image.new("RGB", (512, 512), "#" + colors["environmentColor0"][:6]).save(out / "cover.jpg", "JPEG")
    current = json.loads((HERE / "CURRENT.json").read_text(encoding="utf-8"))
    builder_name = current["builder"]
    old_argv = sys.argv
    sys.argv = [f"{builder_name}.py", key_of(job), "--replace"]
    captured = LineWriter(lambda line: None)
    try:
        builder = __import__(builder_name)
        with contextlib.redirect_stdout(captured):
            builder.main()
    except SystemExit as stop:
        if created:
            shutil.rmtree(out, ignore_errors=True)
        text = str(stop)
        if "has changed since this tool built it" in text:
            text = f"{Path(text.split(' has changed')[0]).name} was edited (in ChroMapper?) after this map was made, so nothing was overwritten. To keep your edits, make the new version under another folder name."
        raise UserError(f"The map builder stopped: {text}")
    except Exception:
        if created:
            shutil.rmtree(out, ignore_errors=True)
        raise
    finally:
        sys.argv = old_argv
    report = json.loads(next(line for line in reversed(captured.lines) if line.startswith("{")))
    if kept_jumps:
        keep_jump_settings(out, name, kept_jumps, report)

    progress("check", "Checking the swing flow")
    check = LineWriter(lambda line: None)
    if cfg["patterns"]:
        sys.argv = ["check_patterns.py", key_of(job)]
        try:
            with contextlib.redirect_stdout(check):
                runpy.run_path(str(HERE / "core" / "check_patterns.py"), run_name="__main__")
        except SystemExit:
            pass
        except Exception as problem:
            check.lines.append(f"check could not run: {problem}")
        finally:
            sys.argv = old_argv
    return {"folder": str(out), "zip": str(out / f"{name}.zip"), "environment": environment, "environment_name": style_engine.ENVIRONMENTS[environment]["name"],
            "colors": colors, "combos": combos, "lightshow": bool(job.get("lightshow")), "tempo": tempo, "report": report,
            "checks": [line.strip() for line in check.lines if line.strip().startswith(tuple(difficulties)) or "PASS" in line or "PROBLEM" in line],
            "model": {"builder": builder_name, "models": current["models"], "synced": current["synced"]}}


class UserError(Exception):
    pass


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        result = analyze(job) if job.get("mode") == "analyze" else build(job)
        emit("result", mode=job.get("mode", "build"), **result)
    except UserError as problem:
        emit("error", message=str(problem))
        sys.exit(1)
    except Exception as problem:
        emit("error", message=f"{type(problem).__name__}: {problem}", trace=traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
