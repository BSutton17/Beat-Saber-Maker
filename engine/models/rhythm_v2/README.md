# Rhythm model v2

Saved 2026-11-07, because the user pointed out that v1 only knew rap. It estimates how likely a mapper is to place a note on each step of a song, with one model for each of Easy, Normal, Hard and Expert. It only learns timing, so it can't copy anyone's patterns.

- `weights.json`: the models, as plain numbers, with a confidence floor for each level.
- `training_measurements.npz`, `training_meta.json`: what it trained on, so it can be retrained without the songs.

## Trained on

The user's Quebec, Love Galore and Chemicals, counted double, and well-rated human maps from BeatSaver in country and folk, soul and vintage ballads, R&B, pop, indie and rock, and rap. `trained_on` in `weights.json` has the list. Maps whose notes don't sit on one grid were left out, since they can't teach timing: Take Me To Church, Almost, Paradise, Blackbird and Eleven Eleven.

## How good it is

Each map scored by a model trained without it. 0.50 is guessing and 1.00 is perfect. Full results: `analysis/rhythm_v2_evaluation.txt`.

| Level | v1 | v2 | Maps |
|---|---|---|---|
| Easy | 0.79 | 0.93 | 11 |
| Normal | 0.82 | 0.90 | 13 |
| Hard | 0.80 | 0.89 | 14 |
| Expert | 0.78 | 0.83 | 20 |

v1 was trained on Quebec and Love Galore, so its 0.81 on those isn't a fair comparison with v2's 0.77, which never saw them. Tests never used in training: Nice For What (Duddzy) 0.71, the user's Pick Up the Phone 0.63, whose notes sit about 100 ms from the sound. No training map was on a triplet grid, so swung songs are the least tested.

## Bugs found while building it, all of which v1 also had

- The beat flags counted from wherever a map's notes began, so "on the beat" was off on many maps. `metric_flags` in `tools/rhythm_features_v2.py` now finds the beat from the music, in training and on new songs alike.
- Half of the maps' notes sit on a grid with a constant shift. Measuring the audio at whole steps missed them entirely, so `subdivision` in `tools/rhythm_model_v2.py` finds each map's shift from its notes.
- A sound-shift check run on every grid step, not on the notes, locks onto the wrong subdivision on repeating music. Scores far below 0.50 mean something like this is broken, not that the model is weak.

## Using it

`tools/map_song_v2.py` does everything, from `config/batch2_songs.json`. To retrain: `python rhythm_model_v2.py extract <refmaps folder>`, `evaluate`, then `save`. `tools/research/find_genre_maps.py` and `fetch_genre_refs.py` fetched the training maps. The playtest overrides the metrics.
