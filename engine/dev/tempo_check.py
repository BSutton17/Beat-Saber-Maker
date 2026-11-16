"""Check tempo_auto against the hand-picked tempos of earlier songs. Run from the AI Mapper tools folder's Python."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

TOOLS = Path(sys.argv[1])
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grid_tools, metrics, tempo_auto  # noqa

WIP = TOOLS.parent.parent / "Beat Saber_Data" / "CustomWIPLevels"
known = {k: (v["pulse_bpm"], v["per_beat"], v["map_beats_per_pulse"], v["folder"]) for k, v in json.loads((TOOLS.parent / "config" / "batch2_songs.json").read_text(encoding="utf-8")).items()}
known.update({"Can't Feel My Legs": (70, 4, 1, "Can't Feel My Legs AI"), "Slow Dancing In the Dark": (89, 4, 1, "Slow Dancing In the Dark AI"), "Treehouse": (73.51, 4, 1, "Treehouse AI")})
for name, (pulse, per_beat, per_pulse, folder) in known.items():
    info = json.loads((WIP / folder / "Info.dat").read_text(encoding="utf-8-sig"))
    song = info["audio"]["songFilename"] if "audio" in info else info["_songFilename"]
    audio, rate = metrics.decode(str(WIP / folder / song))
    mono = signal.resample_poly(audio.mean(axis=1) if audio.ndim > 1 else audio, grid_tools.SR, rate).astype(np.float32)
    got = tempo_auto.detect(mono)
    same_grid = abs(got["pulse_bpm"] * got["per_beat"] - pulse * per_beat) / (pulse * per_beat) < 0.02
    print(f"{name:26s} hand {pulse:6.2f}/{per_beat}/{per_pulse}  auto {got['pulse_bpm']:6.2f}/{got['per_beat']}/{got['map_beats_per_pulse']}  {'OK ' if same_grid else 'XX '} raw {got['raw_tempo']} {got['strength']}", flush=True)
