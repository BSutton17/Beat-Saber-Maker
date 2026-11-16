"""Lighting that follows a song's mood: dark and slow in the quiet parts, exploding on the exact beat the big parts begin.

The song is cut into sections by its energy, and each section gets a level from 0 (nearly dark) to 4 (everything on). Within a section, each light has a job, and the strongest moments of that
job's instrument get events, up to the section's budget. The budgets come from the user's request and from what good lighters do, in models/lights_v1. A section's first beat is where its level starts:
the old engine's lights came from hits inside bars, so they only built up partway through a big part, which the user noticed in Slow Dancing In the Dark.
"""
import json

import numpy as np

from lights_model import HOP_S, MODEL_FOLDER, intensity_parts
from rhythm_features_v2 import NAMES

SYLLABLE, VOCAL_ONSET, KICK, SNARE, BASS, OTHER, CHORD, PHRASE, ENERGY = (NAMES.index(n) for n in ("syllable", "vocal_onset", "kick", "snare_hat", "bass", "other", "chord_change", "phrase_start", "energy"))
# a section counts as having a beat to follow when its drums hit this hard on the beat, on average
DRUMS_PRESENT = 0.35
OFF, BLUE_ON, BLUE_FLASH, BLUE_FADE, BLUE_EASE, RED_ON, RED_FLASH, RED_FADE, RED_EASE, WHITE_ON, WHITE_FLASH, WHITE_FADE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
# light events per second at each level, before a song's own rate. The quiet end is the user's request, darker than most lighters go. The top matches the busy end of well-rated maps.
LEVEL_RATE = [0.12, 0.55, 1.6, 3.2, 4.8]
LEVEL_STRENGTH = [0.3, 0.5, 0.85, 1.0, 1.1]
LEVEL_SPEED = [0, 1, 2, 3, 6]
# the shortest gap between events on one light at each level, in seconds
LEVEL_GAP = [2.0, 1.0, 0.45, 0.22, 0.16]
# who does the work at each level
LEVEL_SHARE = [{"back": 1.0}, {"back": 0.45, "main": 0.2, "left": 0.175, "right": 0.175}, {"center": 0.25, "back": 0.2, "main": 0.2, "left": 0.175, "right": 0.175},
               {"center": 0.24, "back": 0.14, "main": 0.28, "left": 0.17, "right": 0.17}, {"center": 0.22, "back": 0.13, "main": 0.27, "left": 0.19, "right": 0.19}]
# what each job listens to: drums first, and instruments where a song has no drums
JOB_SOUND = {"center": (KICK, BASS), "back": (SNARE, CHORD), "main": (OTHER, SNARE), "left": (SYLLABLE, VOCAL_ONSET), "right": (SYLLABLE, VOCAL_ONSET)}
MIN_SECTION_BEATS = 8
# how much a change has to explain before it becomes a new section, with a song's usual range of intensity counted as 1
SECTION_PENALTY = 0.45
# a jump of this many levels is an explosion
EXPLOSION_JUMP = 2

# which lanes do each job in each environment, from the mapping wiki and the game's environment data
ENVIRONMENTS = {
    "DefaultEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    "BigMirrorEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": None, "speed": (12, 13)},
    # Panic's ring lights are much brighter than others, so they're held down
    "PanicEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13), "main_strength": 0.7},
    # K/DA: bottom and top lights, left and right lasers, centre lights, and laser speed; no rings. Its right lasers also light the middle of the arrows.
    "KDAEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": None, "zoom": None, "speed": (12, 13)},
    "KaleidoscopeEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": None},
    # the gates, plus lasers on 6 and 7. Lane 8 drives the cars, and 16 and 17 their hydraulics.
    "InterscopeEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2, 6], "right": [3, 7], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13), "hydraulics": (16, 17)},
    # the water channel lights, the rays, and the sun or moon on 4, with lasers on 10 and 11 and more water on 6 and 7. 8 would start the rain, so it's left alone.
    "BillieEnvironment": {"jobs": {"back": [0, 6], "main": [1, 7], "left": [2, 10], "right": [3, 11], "center": [4]}, "spin": None, "zoom": None, "speed": (12, 13), "always_lit": [4]},
    # the auroras on 0 and 1, the tower lights on 2, 3, 6, 7, 10 and 11, and the tower coils' height on 12, 13 and 16-19
    "GagaEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2, 6, 10], "right": [3, 7, 11], "center": [4]}, "spin": None, "zoom": None, "speed": None, "coils": (12, 13, 16, 17, 18, 19), "always_lit": [0]},
    # the rest use the standard five lanes; they differ in which triggers they have (BSMG wiki, basic lighting). The game ignores events for lights an environment doesn't have.
    "TriangleEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    # Nice's "left" and "right" lasers are below and above the track
    "NiceEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    # Imagine Dragons' side lights on 0 show the opposite colour of the event
    "DragonsEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    "MonstercatEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": None, "speed": (12, 13)},
    "CrabRaveEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": None, "speed": (12, 13)},
    "RocketEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": None, "zoom": None, "speed": (12, 13)},
    "OriginsEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": None, "speed": (12, 13)},
    "GreenDayEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    "GreenDayGrenadeEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": None, "zoom": None, "speed": (12, 13)},
    "TimbalandEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    # FitBeat's ring lasers on 0 show the opposite colour of the event
    "FitBeatEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    # Linkin Park: floor and ceiling lights; 8 switches the laser mode
    "LinkinParkEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": None, "speed": (12, 13)},
    # BTS: the logo, centre lasers, side lasers and runway; 8 spins the towers and 9 moves the pillars
    "BTSEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    # Skrillex: lights on 0-4 plus lasers on 6 and 7; 8 and 9 spin its two sets of rings
    "SkrillexEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2, 6], "right": [3, 7], "center": [4]}, "spin": 8, "zoom": 9, "speed": (12, 13)},
    "HalloweenEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": None, "zoom": None, "speed": (12, 13)},
    # Weave, Fall Out Boy and EDM are group-lighting environments: only some of their lights answer these basic events, and they have no triggers.
    # Fall Out Boy's value lane picks the video clip, so it gets no speed events.
    "WeaveEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": None, "zoom": None, "speed": None},
    "PyroEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": None, "zoom": None, "speed": None},
    "EDMEnvironment": {"jobs": {"back": [0], "main": [1], "left": [2], "right": [3], "center": [4]}, "spin": None, "zoom": None, "speed": None},
}


def beat_energy(stems, beats, duration):
    """How intense the song feels at each beat, by the recipe learned from human lighters, with the song's usual range running from 0 to 1."""
    parts, _ = intensity_parts(stems, duration)
    weights = json.loads((MODEL_FOLDER / "weights.json").read_text(encoding="utf-8"))["intensity_weights"]
    value = parts @ np.array(weights)
    out = np.zeros(len(beats))
    for i, start in enumerate(beats):
        end = beats[i + 1] if i + 1 < len(beats) else start + (beats[i] - beats[i - 1])
        a, b = int(start / HOP_S), max(int(start / HOP_S) + 1, int(end / HOP_S))
        part = value[a:b]
        out[i] = part.mean() if len(part) else (value[-1] if len(value) else 0.0)
    low, high = np.percentile(out, 8), np.percentile(out, 92)
    return (out - low) / (high - low + 1e-9)


def split_sections(energy, min_beats=MIN_SECTION_BEATS, penalty=SECTION_PENALTY, forced=()):
    """Cut the song where its energy really changes: the fewest sections that explain the energy well. Forced cuts, such as where one song in a medley starts, always stay."""
    if forced:
        cuts = sorted({0, len(energy), *[c for c in forced if 0 < c < len(energy)]})
        out = []
        for a, b in zip(cuts, cuts[1:]):
            out += [(a + x, a + y) for x, y in split_sections(energy[a:b], min(min_beats, b - a), penalty)]
        return out
    n = len(energy)
    sums = np.concatenate([[0], np.cumsum(energy)])
    squares = np.concatenate([[0], np.cumsum(energy ** 2)])

    def cost(a, b):
        total = sums[b] - sums[a]
        return squares[b] - squares[a] - total * total / (b - a)

    best = np.full(n + 1, np.inf)
    best[0] = 0
    back = np.zeros(n + 1, int)
    for end in range(min_beats, n + 1):
        for start in range(0, end - min_beats + 1):
            if best[start] == np.inf:
                continue
            value = best[start] + cost(start, end) + penalty
            if value < best[end]:
                best[end] = value
                back[end] = start
    cuts = []
    at = n
    while at > 0:
        cuts.append(at)
        at = back[at]
    cuts = [0] + cuts[::-1]
    return list(zip(cuts, cuts[1:]))


def sharpen(sections, energy, reach=2, fixed=(), bar_offset=None):
    """Move each cut to the beat where the energy actually jumps, so a big part starts on its first beat. Given the bar lines, a cut prefers a bar's first beat, where drops land."""
    if len(sections) < 2:
        return sections
    cuts = [s[0] for s in sections] + [sections[-1][1]]
    for i in range(1, len(cuts) - 1):
        if cuts[i] in fixed:
            continue
        options = range(max(cuts[i - 1] + 2, cuts[i] - reach), min(cuts[i + 1] - 2, cuts[i] + reach) + 1)
        rising = energy[cuts[i]:cuts[i + 1]].mean() >= energy[cuts[i - 1]:cuts[i]].mean()

        def jump(k):
            change = energy[k:k + 2].mean() - energy[max(0, k - 2):k].mean()
            downbeat = 0.15 if bar_offset is not None and (k - bar_offset) % 4 == 0 else 0.0
            return (change if rising else -change) + downbeat

        cuts[i] = max(options, key=jump, default=cuts[i])
    return list(zip(cuts, cuts[1:]))


def section_levels(sections, energy, style):
    """Give each section a level from where its energy sits in this song, within the range this song's mood allows."""
    means = np.array([energy[a:b].mean() for a, b in sections])
    places = np.clip(means, 0, 1)
    edges = style.get("level_edges", (0.18, 0.42, 0.68, 0.88))
    levels = np.searchsorted(edges, places)
    lowest, highest = style.get("lowest", 0), style.get("highest", 4)
    # the song's single biggest moment reaches the top level, when the mood allows it
    levels[int(np.argmax(means))] = max(levels[int(np.argmax(means))], min(highest, 3))
    return np.clip(levels, lowest, highest).astype(int).tolist()


def merge_sections(sections, levels):
    merged = []
    for (a, b), level in zip(sections, levels):
        if merged and merged[-1][2] == level:
            merged[-1] = (merged[-1][0], b, level)
        else:
            merged.append((a, b, level))
    return merged


def plan(stems, beats, duration, style, forced=(), bar_offset=None):
    """The whole song's sections as (first beat, end beat, level)."""
    energy = beat_energy(stems, beats, duration)
    sections = split_sections(energy, forced=forced)
    if style.get("beat_locked"):
        # a light show snaps its changes to bar lines, but never moves a forced cut such as a song's start in a medley
        sections = sharpen(sections, energy, reach=3, fixed=set(forced), bar_offset=bar_offset)
    elif not forced:
        sections = sharpen(sections, energy)
    return merge_sections(sections, section_levels(sections, energy, style)), energy


def beat_section(add, env, span, step_times, measures, per_beat, offset_steps, level, soft, style):
    """One section of a beat-locked show. With a beat to follow, every light keeps a steady pattern on the beat grid, growing with the level.
    With no drums, the lights follow the singing instead, snapped to the grid. Never both at once, as the user asked."""
    half_beat = (lambda k: (k - offset_steps) % per_beat in (0, 2)) if per_beat == 3 else (lambda k: (k - offset_steps) % (per_beat // 2) == 0)
    on_beat = lambda k: (k - offset_steps) % per_beat == 0
    beat_number = lambda k: ((k - offset_steps) // per_beat) % 4
    bar = lambda k: (k - offset_steps) // (4 * per_beat)
    beats = [k for k in span if on_beat(k)]
    if not beats:
        return
    drums = float(np.mean([max(measures[k, KICK], measures[k, SNARE]) for k in beats]))
    strength = LEVEL_STRENGTH[level]
    main_strength = env.get("main_strength", 1.0)

    def colour(k, job):
        phrase = bar(k) // 4
        if level <= 1:
            return BLUE_FADE
        flip = (phrase + (job in ("back", "main")) + (beat_number(k) % 2 if job in ("left", "right") else 0)) % 2
        return RED_FADE if flip else BLUE_FADE

    def fire(k, job, power=1.0):
        if measures[k, ENERGY] < 0.3:
            return
        value = colour(k, job)
        if level == 4 and not soft and beat_number(k) == 0 and on_beat(k):
            value = RED_FLASH if value == RED_FADE else BLUE_FLASH
        for lane in env["jobs"][job]:
            add(k, lane, value, strength * power * (main_strength if job == "main" else 1.0))

    if drums >= DRUMS_PRESENT or level >= 3:
        for k in span:
            if not half_beat(k):
                continue
            n = beat_number(k)
            if on_beat(k):
                if level >= 2 or n == 0:
                    fire(k, "back" if level <= 1 else "center", 1.0 if n == 0 else 0.85)
                if level >= 2 and n in (1, 3):
                    fire(k, "back")
                if level == 1 and n == 2:
                    fire(k, "main", 0.8)
                if level >= 3:
                    fire(k, "left" if n % 2 == 0 else "right")
                if level == 4:
                    fire(k, "back")
                    fire(k, "right" if n % 2 == 0 else "left", 0.8)
                if level == 2 and n == 0:
                    fire(k, "main")
            elif level >= 3:
                # the off-beats keep the rings moving in big parts
                fire(k, "main", 0.75)
    else:
        # no beat to follow, so the lasers answer the singing, snapped to the nearest half beat, and the back light marks each bar
        # quieter parts answer less of the singing: only line starts, and further apart
        threshold, spacing = {0: (0.6, 1.6), 1: (0.5, 0.8)}.get(level, (0.45, 0.4))
        sung = [k for k in span if half_beat(k) and (measures[k, PHRASE] if level == 0 else measures[k, PHRASE] + 0.6 * measures[k, SYLLABLE]) > threshold]
        last = -1e9
        for k in sung:
            if step_times[k] - last >= spacing:
                fire(k, "left" if (bar(k) + beat_number(k)) % 2 == 0 else "right")
                last = step_times[k]
            if measures[k, PHRASE] > 0.5 and level >= 2:
                fire(k, "center", 0.8)
        for k in beats:
            if beat_number(k) == 0:
                fire(k, "back", 0.8)
    # motion on the bar: Kaleidoscope has only spin and zoom to move, so bigger parts use them a lot
    if env.get("spin") is not None:
        every = {1: 8, 2: 2, 3: 1, 4: 0.5}.get(level)
        if every:
            for k in range(span[0], span[-1] + 1, max(1, int(every * 4 * per_beat))):
                add(k, env["spin"], 1)
    if env.get("zoom") is not None and level >= 3:
        every = 4 if level == 3 else 2
        for k in range(span[0], span[-1] + 1, every * 4 * per_beat):
            add(k, env["zoom"], 1)


def pick_moments(strength, times, budget, min_gap):
    """The strongest moments, spaced apart, up to the budget."""
    chosen = []
    for i in np.argsort(strength)[::-1]:
        if len(chosen) >= budget or strength[i] <= 0:
            break
        if all(abs(times[i] - times[c]) >= min_gap for c in chosen):
            chosen.append(int(i))
    return sorted(chosen)


def generate(step_times, measures, per_beat, sections, style, environment, offset_steps=0):
    """Events as (step, lane, value, strength), and colour boosts as (step, on).

    offset_steps is where bars start. In a medley each song starts its bars differently, so it may also be a function giving the bar start for a section's first step.
    """
    bar_start_for = offset_steps if callable(offset_steps) else (lambda first: offset_steps)
    env = ENVIRONMENTS[environment]
    rate_scale = style.get("rate", 1.0)
    bar_steps = 4 * per_beat
    events = []
    boosts = []
    total = len(step_times)
    soft = style.get("soft", False)
    use_boost = style.get("boost", True)
    speed_scale = style.get("speed", 1.0)
    previous = None
    dark_before = []

    def add(step, lane, value, strength=1.0):
        events.append((int(step), int(lane), int(value), round(float(strength), 3)))

    light_lanes = [lane for lanes in env["jobs"].values() for lane in lanes]
    main_strength = env.get("main_strength", 1.0)
    for first_beat, end_beat, level in sections:
        first, last = first_beat * per_beat, min(end_beat * per_beat, total - 1)
        if last <= first:
            continue
        span = np.arange(first, last)
        seconds = step_times[last] - step_times[first]
        offset_steps = bar_start_for(first)
        explosion = previous is not None and level - previous >= EXPLOSION_JUMP and level >= 3
        falling = previous is not None and level < previous
        # the moment before a big part is held dark, so the hit lands
        if explosion and first - per_beat >= 0:
            dark_before.append(first)
            for lane in light_lanes:
                add(first - per_beat, lane, OFF)
        if explosion or (level == 4 and previous != 4):
            for lanes in env["jobs"].values():
                for lane in lanes:
                    add(first, lane, RED_FLASH if lane % 2 else BLUE_FLASH, 1.2)
            if env.get("spin") is not None:
                add(first, env["spin"], 1)
            if env.get("zoom") is not None:
                add(first, env["zoom"], 1)
        elif falling:
            # let what was lit die away rather than cut
            for lane in light_lanes:
                add(first, lane, BLUE_FADE if level >= 1 else OFF, LEVEL_STRENGTH[level])
        elif previous is None:
            # start clean and dark, so nothing lights before the music does
            for lane in light_lanes:
                add(first, lane, OFF)
        # a mood that keeps one light glowing through quiet parts, like a sun or an aurora
        if level <= 2:
            for lane in env.get("always_lit", []):
                add(first + 1, lane, BLUE_EASE, LEVEL_STRENGTH[level] * style.get("glow", 0.7))
        if style.get("beat_locked"):
            beat_section(add, env, span, step_times, measures, per_beat, offset_steps, level, soft, style)
            share_items = []
        else:
            share_items = LEVEL_SHARE[level].items()
        # how many events each job gets here
        budget = LEVEL_RATE[level] * rate_scale * seconds
        for job, share in share_items:
            lanes = env["jobs"][job]
            sound, fallback = JOB_SOUND[job]
            strength = np.maximum(measures[span, sound], 0.7 * measures[span, fallback])
            if job in ("left", "right") and level <= 2:
                # lasers answer the start of each sung line in quiet parts, not every syllable
                strength = measures[span, PHRASE] + 0.3 * measures[span, SYLLABLE]
            if job == "right":
                # so the two sides answer each other, not fire together
                strength = strength * (1 + 0.5 * ((span // (2 * per_beat)) % 2))
            elif job == "left":
                strength = strength * (1 + 0.5 * (((span // (2 * per_beat)) + 1) % 2))
            moments = pick_moments(strength, step_times[span], int(round(budget * share)), LEVEL_GAP[level])
            for m in moments:
                step = int(span[m])
                bar = (step - offset_steps) // bar_steps
                phrase = bar // 4
                # quiet parts stay on one colour, and bigger parts bring in the other
                if level <= 1:
                    value = style.get("quiet_fade", BLUE_FADE)
                elif level == 2:
                    value = RED_FADE if phrase % 2 and job in ("left", "right") else BLUE_FADE
                else:
                    value = RED_FADE if (phrase + (job in ("back", "main"))) % 2 else BLUE_FADE
                    if level == 4 and not soft and (step - offset_steps) % bar_steps == 0:
                        value = RED_FLASH if value == RED_FADE else BLUE_FLASH
                power = LEVEL_STRENGTH[level] * (main_strength if job == "main" else 1.0)
                for lane in lanes:
                    add(step, lane, value, power)
                    if lane in env.get("always_lit", []) and level <= 2:
                        # return to the glow afterwards
                        add(min(step + per_beat, last), lane, BLUE_EASE, LEVEL_STRENGTH[level] * style.get("glow", 0.7))
        # motion, once a bar
        for step in range(first, last, bar_steps):
            if env.get("speed") is not None:
                speed = int(round(LEVEL_SPEED[level] * speed_scale))
                for lane in env["speed"]:
                    add(step, lane, speed)
            if env.get("coils") is not None:
                height = [0, 1, 2, 4, 7][level]
                for lane in env["coils"]:
                    add(step, lane, height)
        if env.get("spin") is not None and not style.get("beat_locked"):
            every = {2: 4, 3: 2, 4: 1}.get(level)
            if every:
                for step in range(first + every * bar_steps, last, every * bar_steps):
                    add(step, env["spin"], 1)
        if env.get("hydraulics") is not None and level >= 3:
            hits = pick_moments(measures[span, KICK], step_times[span], int(seconds * (0.5 if level == 3 else 1.0)), 0.6)
            for k, m in enumerate(hits):
                add(int(span[m]), env["hydraulics"][k % 2], 1 if k % 4 < 2 else 0)
        if use_boost:
            if style.get("colorful") and level in (2, 3):
                # a colourful show swaps colour pairs in mid-size parts too, every eight bars
                for k, step in enumerate(range(first, last, 8 * bar_steps)):
                    boosts.append((step, k % 2 == 1))
            elif level == 4:
                # the biggest parts swap between both colour pairs every four bars, so all four colours keep coming round
                for k, step in enumerate(range(first, last, 4 * bar_steps)):
                    boosts.append((step, k % 2 == 0))
            else:
                boosts.append((first, False))
        previous = level
    # the song ends in the dark
    for lane in light_lanes:
        add(max(0, total - 2), lane, BLUE_FADE, 0.5)
    # nothing may fire in the held-dark beat before a big hit, and a light does one thing at a time: a hit beats anything else
    keep = {}
    for event in events:
        step, lane, value, _ = event
        if lane in light_lanes and any(mark - per_beat < step < mark for mark in dark_before):
            continue
        rank = 2 if value in (BLUE_FLASH, RED_FLASH) else (1 if value == OFF else 0)
        if (step, lane) not in keep or rank >= keep[(step, lane)][0]:
            keep[(step, lane)] = (rank, event)
    return sorted(event for _, event in keep.values()), boosts
