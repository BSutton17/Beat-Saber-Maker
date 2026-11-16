"""Read-only checks for maps from the pattern engine: swing parity per hand, turn angles, hands tangling, and valid chains and arcs. Reads the zip, as the headset would.

    python check_patterns.py "Like Him" [more songs]
"""
import json
import sys
import zipfile
from collections import defaultdict

import numpy as np

from mapfiles import PROJECT, WIP
from notes_v2 import DIFFICULTY
from patterns_v3 import BLOCKED, DOWNWARD, FOREHAND, TANGLE_S, turn

failures = 0
config = json.loads((PROJECT / "config" / "batch2_songs.json").read_text(encoding="utf-8"))
for song in sys.argv[1:]:
    cfg = config[song]
    with zipfile.ZipFile(WIP / cfg["folder"] / f"{cfg['folder']}.zip") as archive:
        info = json.loads(archive.read("Info.dat"))
        audio = json.loads(archive.read("AudioData.dat"))
        maps = {row["difficulty"]: json.loads(archive.read(row["beatmapDataFilename"])) for row in info["difficultyBeatmaps"] if row["characteristic"] == "Standard"}
    regions = audio["bpmData"]
    marks = [regions[0]["sb"]] + [r["eb"] for r in regions]
    samples = [regions[0]["si"]] + [r["ei"] for r in regions]
    seconds = lambda b: float(np.interp(b, marks, samples)) / audio["songFrequency"]
    print(f"\n=== {song} | {info['environmentNames']} | colour override {info['colorSchemes'][0].get('useOverride')}")
    for level, data in maps.items():
        meta = data["colorNotesData"]
        notes = sorted((seconds(n["b"]), meta[n["i"]]["c"], meta[n["i"]]["x"], meta[n["i"]]["y"], meta[n["i"]]["d"]) for n in data["colorNotes"])
        problems = defaultdict(int)
        reset = DIFFICULTY[level]["reset_s"]
        for hand in (0, 1):
            swings = []
            for t, c, x, y, d in notes:
                if c != hand:
                    continue
                if swings and abs(swings[-1][0] - t) < 0.01:
                    # a stack is one swing
                    if swings[-1][4] != d:
                        problems["stack with mixed directions"] += 1
                    continue
                swings.append((t, c, x, y, d))
            for (t0, _, x0, y0, d0), (t1, _, x1, y1, d1) in zip(swings, swings[1:]):
                if t1 - t0 >= reset:
                    continue
                if (d0 in FOREHAND[hand]) == (d1 in FOREHAND[hand]):
                    problems["parity broken (two forehands or two backhands in a row)"] += 1
                elif turn(d0, d1) < 135:
                    problems["turn under 135 degrees"] += 1
        for t, c, x, y, d in notes:
            if d in DOWNWARD and y == 2:
                problems["down swing on the top row"] += 1
            if (x, y) in BLOCKED:
                problems["note in the middle of the middle row"] += 1
        lefts = [(t, x) for t, c, x, y, d in notes if c == 0]
        rights = [(t, x) for t, c, x, y, d in notes if c == 1]
        # each note against where the other hand actually is: its latest note at or before this one
        def tangled(mine, theirs, mine_is_left):
            count = 0
            for t, x in mine:
                before = [(t2, x2) for t2, x2 in theirs if t2 <= t + 1e-6]
                if before and t - before[-1][0] < TANGLE_S and ((x >= before[-1][1]) if mine_is_left else (x <= before[-1][1])):
                    count += 1
            return count
        tangles = tangled(lefts, rights, True) + tangled(rights, lefts, False)
        far = sum(1 for t, c, x, y, d in notes if (c == 0 and x == 3) or (c == 1 and x == 0))
        if far:
            problems["crossover to the far lane"] += far
        if tangles:
            problems["hands tangled (left at or right of right within 0.5 s)"] += tangles
        for chain in data["chains"]:
            head = meta[chain["i"]]
            tail = data["chainsData"][chain["ci"]]
            if not (0 <= tail["tx"] <= 3 and 0 <= tail["ty"] <= 2) or chain["tb"] <= chain["hb"] or not any(abs(n["b"] - chain["hb"]) < 1e-6 and n["i"] == chain["i"] for n in data["colorNotes"]):
                problems["bad chain"] += 1
        for arc in data["arcs"]:
            if meta[arc["hi"]]["c"] != meta[arc["ti"]]["c"] or arc["tb"] <= arc["hb"]:
                problems["bad arc"] += 1
        moments = {}
        for t, c, x, y, d in notes:
            moments.setdefault(round(t, 3), []).append(c)
        doubles = sum(1 for v in moments.values() if 0 in v and 1 in v) / len(moments)
        crossed = sum(1 for t, c, x, y, d in notes if (c == 0 and x >= 2) or (c == 1 and x <= 1)) / len(notes)
        positions = len({(c, x, y) for t, c, x, y, d in notes})
        print(f"  {level:<7} {len(notes)} notes | doubles {doubles * 100:.0f}% of moments | crossovers {crossed * 100:.0f}% | positions {positions} | chains {len(data['chains'])} | arcs {len(data['arcs'])} | " + ("PASS" if not problems else "PROBLEMS " + dict(problems).__repr__()))
        failures += bool(problems)
sys.exit(1 if failures else 0)
