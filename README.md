# Beat Saber Maker

Make Beat Saber maps from a song and a cover image, or just a YouTube link, straight into ChroMapper's folder. It runs locally, on your computer only.

## What it does

1. **Song**: paste a YouTube link (its audio is downloaded and converted to `.ogg`, and its thumbnail becomes the cover), or upload an audio file and a cover image.
2. **Analysis**: Demucs splits the song into vocals, drums, bass and other instruments. The program then finds the tempo and whether the beat is straight, swung or 6/8.
3. **Look**: an environment and colours are suggested. Switch either one to **Manual** to choose yourself. Every environment has a picture so you can see what you're picking.
4. **Map**: pick Easy, Normal, Hard and/or Expert, then **Make map**. Notes come from the rhythm model, lights from the lighting model, and the map is written in ChroMapper's v4 format with a zip for the Quest.

## Start it

```
npm install
npm run setup     # first time on a computer: Python check, yt-dlp, environment pictures, config.json
npm start         # builds the page and opens http://localhost:5175
```

`npm run dev` runs the page with hot reload on http://localhost:5173 while you work on the code.

Setup looks for a Python 3.11+ that already has numpy, scipy, scikit-learn, Pillow, torch and torchaudio. If there isn't one, it creates `engine/.venv` and installs `engine/requirements.txt` into it, which is a large download because of torch. The first stem split also downloads the Demucs model, about 300 MB.

Set the maps folder in **Settings** to ChroMapper's `Beat Saber_Data\CustomWIPLevels`.

## The models

Everything that makes a map lives in `engine/` and is part of this repository, so a fresh clone on another computer works:

| Part | Where | What it does |
|---|---|---|
| Rhythm model | `engine/models/rhythm_v2` | where notes go, per difficulty |
| Lighting model | `engine/models/lights_v1` | how intense each moment feels, for mood lighting |
| Environment model | `engine/models/style_v1` | nearest match to songs whose environment was chosen by hand; learns from your manual picks (`your_picks.json`) |
| Mapping engine | `engine/core` | patterns, flow, lights and the map writer |
| Tempo | `engine/tempo_auto.py` | pulse and subdivision; matches the hand-chosen grid on all 10 earlier songs |
| Colours | `engine/style_engine.py` | vivid colours from the cover, or a mood palette when the cover is grey |

**Keeping the model current.** `engine/core` and `engine/models` are copies from the AI Mapper project. When a newer model is made there, use **Settings → Update model**, or run `npm run sync-model`. It copies the builder, every module it imports and the model folders named in `AI Mapper/models/CURRENT.json`, so the program never needs rewriting for a new model. Commit afterwards to carry the new model to other computers.

## Folders

- `server/`: the Node and Express server (song intake, job queue, settings).
- `client/`: the React page (Vite).
- `engine/`: the Python engine. `run_job.py` is the only entry point the server calls.
- `work/` (not committed): songs, stems and jobs.

## Notes

- Only download songs you have the right to use. YouTube's terms restrict downloading.
- The environment pictures come from the BSMG wiki and are the game's own images, so setup downloads them rather than keeping them in the repository.
- Maps aren't playtested by the program: always play them.
