# Lighting model v1

Saved 2026-11-07. The user asked for lighting that reflects a song's mood. Their example: in Slow Dancing In the Dark, the lights came on partway through the exciting parts, when they should be dark in the slow moments and explode in the intense ones.

`weights.json` holds the habits of good lighters, measured on 28 maps. `tools/lights_model.py` measures them and `tools/lights_v2.py` uses them. Only overall habits are kept, so no one's lightshow can be copied.

## What was learned

- **Loudness doesn't tell you what feels big.** Slow Dancing's quiet verses have its heaviest bass, so they measure loud. A recipe fitted to what human lighters follow matches their lighting at 0.40 on songs left out of the fit, against 0.25 for plain loudness, where 1.00 is a perfect match. Drum hits count most. It's far from perfect, and that's the honest limit.
- **Lighters act right on the change.** Around the moment a song gets bigger, their events drop to about 0.64 of normal in the second before, then jump to about 1.5 within 2 seconds. So the engine holds a beat dark and hits on the first beat. The show reaches halfway up a median of 0.4 s after the rise.
- **Most lighters vary less than this user wants.** On average their quietest fifth is only about half as busy as their loudest. The user's own Quebec goes from about 1 to 5 events a second, and they asked for dark quiet parts, so `LEVEL_RATE` follows them, not the average.

## How the engine works

It cuts a song into sections by intensity and gives each a level from 0 (nearly dark) to 4 (everything on). Levels set the budget of events, and each light has a job: centre for kicks, back for snares, rings for instruments, lasers for vocals. `ENVIRONMENTS` lists which lanes do each job per environment, from the BSMG wiki and UGEcko's environment data. Each song's mood is in `config/batch2_songs.json`: colours, softness, and the top level it may reach.

## Not verified

Nobody has seen these lightshows yet. How environment-specific lanes behave is from documentation only, not tested: Interscope's car lanes (8, 16, 17), Gaga's coil heights (12, 13, 16-19) and Billie's lanes. Billie's rain (lane 8) is deliberately never triggered. Light brightness values below 1 and the colour schemes depend on the game honouring them.
