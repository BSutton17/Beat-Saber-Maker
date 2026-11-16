"""Build models/style_v1/examples.json from the songs whose environments were chosen by hand in AI Mapper (config/batch2_songs.json).

    python dev/style_train.py "<AI Mapper folder>"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import boot  # noqa: E402,F401
import style_engine  # noqa: E402

ABOUT = "Songs whose environment was chosen by hand, with their measurements. The environment recommender compares new songs with these."
source = Path(sys.argv[1])
config = json.loads((source / "config" / "batch2_songs.json").read_text(encoding="utf-8"))
rows = []
for name, cfg in config.items():
    stems = source / "analysis" / "stems_songs" / name
    f = style_engine.song_features(stems, cfg["pulse_bpm"], cfg["per_beat"])
    rows.append({"song": name, "environment": cfg["environment"], "why": cfg.get("why_environment", ""), "features": {k: round(v, 4) for k, v in f.items()}, "source": "AI Mapper"})
    print(f"{name:24s} {cfg['environment']:24s} drive {style_engine.drive(f):.2f} | {style_engine.describe(f)} | " + " ".join(f"{k}={v:.2f}" for k, v in f.items()), flush=True)
style_engine.MODEL.mkdir(parents=True, exist_ok=True)
(style_engine.MODEL / "examples.json").write_text(json.dumps({"about": ABOUT, "songs": rows}, indent=1), encoding="utf-8")
# with one song per environment, leaving a song out can't give its own environment back, so this shows what comes next instead
for r in rows:
    ranked = style_engine.recommend_environment(r["features"], [o for o in rows if o is not r])
    print("without", r["song"], "->", [(x["environment"], x["distance"]) for x in ranked[:3]])
