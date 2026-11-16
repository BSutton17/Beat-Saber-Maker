# Pacing model v1

Saved 2026-11-16. The user found the MY EYES map (Travis Scott) "wayyyy too busy" in its calm first half, and said maps should calm down in calm parts, pick up in exciting ones, and follow the lyrics better, without being overfitted to that one song.

## What it is

One learned number. A part's share of the notes rises with how loud it is against the song's own average: note density ∝ (relative loudness)^1.22, counted per beat over 8-beat windows. Counting per beat means a faster part, like MY EYES after its tempo change, gets more notes a second without anything learned for it.

`tools/pacing_model.py` holds it (`evaluate`, `save`, `add`). `notes_v2.py` uses it only when a song's settings include `"pacing": true`, which `map_song_v2.py` adds for configs with `"version": 2`. Beat Saber Maker sets that, and older configs build exactly as before.

## How it is used, and what was tested and left out

All measured on 22 human Expert maps (the rhythm_v2 training maps, the user's Quebec, Love Galore and Pick Up the Phone, and aromix's MY EYES). Each map was scored with a pacing model trained without it.

| Choice | Result | Kept? |
|---|---|---|
| Loudness only vs richer sets (voice, drums, all 11 measures) | Loudness fits best (median +0.52 vs +0.45 to +0.47) and is worse than a flat budget on the fewest maps (10/36 vs 13–14) | loudness only |
| Calm parts may rest longer (up to 2.5x), by loudness or by drum-and-voice activity | With combos: median fit to human density +0.43 → +0.50; MY EYES +0.45 → +0.86; Quebec +0.32 → +0.70; placement unchanged (F1 0.950 → 0.953) | yes |
| Combo runs only where drums and voice are at least as busy as usual for the song | Human maps have more back-to-back runs in busier parts within a song (16 of 21 songs). Ice On My Baby's approved combo Expert: 737 notes then, 715 now | yes |
| Note quotas per part | No gain; slightly worse without combos | off (`PACE_QUOTA`) |
| Tilting note choice toward busy parts | No gain | off (`PACE_BIAS`) |
| A 0.3 s minimum gap in calm parts | Over-corrected (calm back-to-back 0% vs humans 31%); lowered F1 and Love Galore's fit | off (`CALM_MIN_GAP_S`) |

The human MY EYES map (BeatSaver 38dd9, aromix, Expert+, mixed reviews: 59 up, 44 down) was added at the user's suggestion for pacing only, never for note placement. Its measurements are in `analysis/pacing_extra.npz`.

## Known limits

Our calmest third still gets about 0.90x the average density, against 0.74x on human maps. Each fix tried for that cost something else. Nobody has played a paced map yet: the playtest overrides these numbers.
