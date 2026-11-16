"""Notes for any difficulty, on straight or triplet grids, and on songs whose tempo drifts.

The flow rules are the ones the user has liked since the first build, from note_engine.py. What's new is that every limit is in seconds, not steps, so a 6/8 song and a straight one
get the same physical feel, and the rhythm comes from the v2 models, one for each difficulty. Targets come from analysis/difficulty_style.json: well-rated maps and the user's own.
"""
import numpy as np

from note_engine import (DOWN, DOWN_FAMILY, Hand, LEFT, RIGHT, TO_LEFT, TO_RIGHT, UP, UP_FAMILY, expert_position, inward, note, outward, pick)
from rhythm_features_v2 import NAMES

SYLLABLE, KICK, SNARE, CHORD, PHRASE, ENERGY = (NAMES.index(n) for n in ("syllable", "kick", "snare_hat", "chord_change", "phrase_start", "energy"))
VOCAL_ONSET_COLUMN = NAMES.index("vocal_onset")
# per_second: note moments a second. min_gap_s: the shortest gap between moments. line_gap_s: the longest gap allowed inside a sung line, so the notes don't seem to stop while the singing carries on.
# max_rest_s: the longest gap while there's still music, since notes that stop feel wrong even when the voice has only paused. double_share: most of the notes that may be two-handed hits. reset_s: a rest this long lets both arms return, so the next swing may go down again. Expert's numbers are what the user loved.
DIFFICULTY = {
    "Easy": {"max_rest_s": 4.0, "per_second": 0.95, "min_gap_s": 0.55, "line_gap_s": 3.2, "double_every_beats": 4, "double_room_s": 1.1, "double_share": 0.24, "top_share": 0.02, "reset_s": 2.2, "direction_mix": ("straight",) * 6 + ("outward",), "njs": 10.0, "reaction_s": 1.25},
    "Normal": {"max_rest_s": 3.0, "per_second": 1.5, "min_gap_s": 0.36, "line_gap_s": 2.2, "double_every_beats": 4, "double_room_s": 0.8, "double_share": 0.2, "top_share": 0.03, "reset_s": 1.5, "direction_mix": ("straight",) * 5 + ("outward",), "njs": 11.0, "reaction_s": 1.15},
    "Hard": {"max_rest_s": 2.4, "per_second": 2.1, "min_gap_s": 0.26, "line_gap_s": 1.5, "double_every_beats": 4, "double_room_s": 0.6, "double_share": 0.15, "top_share": 0.06, "reset_s": 1.1, "direction_mix": ("straight", "straight", "straight", "straight", "outward"), "njs": 12.5, "reaction_s": 0.95},
    "Expert": {"max_rest_s": 2.0, "per_second": 3.0, "min_gap_s": 0.13, "line_gap_s": 1.0, "double_every_beats": 2, "double_room_s": 0.3, "double_share": 0.12, "top_share": 0.13, "reset_s": 0.9, "direction_mix": ("straight", "straight", "straight", "outward", "inward"), "njs": 15.0, "reaction_s": 0.711},
}
# how a difficulty compares with what Expert found in the same song, from well-rated maps: Easy 1.0, Normal 1.5, Hard 2.2 and Expert 3.0 moments a second
SHARE_OF_EXPERT = {"Easy": 0.34, "Normal": 0.52, "Hard": 0.74, "Expert": 1.0}
# nothing before this, so the player has time to get ready as the song starts
FIRST_NOTE_SECONDS = 3.0
STRAY_OPENING_MOMENTS, STRAY_OPENING_REST = 6, 4.0
QUIET_ENERGY = 0.42
FAST_HAND_S = 0.36
# pacing (models/pacing_v1): each part's share of the notes follows how loud it is against the rest of the song, counted over this many seconds around each step
PACE_WINDOW_S = 4.0
PACE_RANGE = (0.25, 3.0)
# whether pace also caps how many notes each part may take (False: pace only lets calm parts rest longer and keeps combos to busy parts)
PACE_QUOTA = False
# how strongly a part's pace tilts which steps are chosen: score x pace ** PACE_BIAS (0 = not at all)
PACE_BIAS = 0.0
# a calm part may rest up to this many times longer than the level's usual longest rest
CALM_REST_STRETCH = 2.5
# in clearly calm parts (below CALM_BELOW of the song's usual loudness or drum-and-voice activity), notes are at least this far apart, so they settle into a
# steady pulse instead of back-to-back pairs. Human Expert maps use far fewer back-to-back notes where the drums and voice are quiet. 0 turns it off.
CALM_MIN_GAP_S = 0.0
# tested on 22 human Expert maps and left off: pacing alone already brings back-to-back notes in calm parts to 26% (humans 31%); this rule took them to 0% and
# lowered placement accuracy (F1 0.953 to 0.944) and the fit on the user's Love Galore (+0.78 to +0.66)
CALM_BELOW = 0.8


def gap_at(settings, pace, activity, k):
    """The shortest gap allowed around step k."""
    gap = settings["min_gap_s"]
    if CALM_MIN_GAP_S and pace is not None and activity is not None and min(float(pace[k]), float(activity[k])) < CALM_BELOW:
        gap = max(gap, CALM_MIN_GAP_S)
    return gap


def pace_for(measures, per_beat):
    """How busy each step's part of the song should be, against the song's average, from the pacing model."""
    import pacing_model
    return np.clip(pacing_model.pacing(measures, per_beat), *PACE_RANGE)


def activity_for(measures, times, window_s=4.0):
    """How much the drums and voice are doing around each step (snare or hi-hat hits and vocal onsets a second), against the song's own median.
    In 21 human Expert maps, busier parts like this have more back-to-back runs within a song (16 of 21 songs); loudness can't see it: MY EYES' calm
    and hard halves are equally loud, but its hard half has about three times the drum and vocal hits."""
    hit = ((measures[:, SNARE] > 0.5) | (measures[:, VOCAL_ONSET_COLUMN] > 0.5)).astype(float)
    cum = np.concatenate([[0.0], np.cumsum(hit)])
    low = np.searchsorted(times, times - window_s / 2)
    high = np.searchsorted(times, times + window_s / 2)
    rate = (cum[high] - cum[low]) / window_s
    playing = measures[:, ENERGY] >= QUIET_ENERGY
    typical = float(np.median(rate[playing])) if playing.any() else float(np.median(rate))
    return rate / max(typical, 1e-6)


def stretch(pace, k, activity=None):
    """How much longer rests may be at step k: 1 in average or busy parts, up to CALM_REST_STRETCH in the calmest, by loudness or by how little is happening."""
    if pace is None:
        return 1.0
    calm = float(pace[k]) if activity is None else min(float(pace[k]), float(activity[k]))
    return float(min(CALM_REST_STRETCH, 1.0 / max(calm, 1.0 / CALM_REST_STRETCH)))


def choose_steps(scores, times, measures, spans, settings, per_second, floor, pace=None, activity=None):
    """The steps people would most likely map, likeliest first, spaced apart, then enough to keep sung lines going.
    Given a pace, the song's notes are shared out by it: no part takes more than its share, so calm parts stay calm and busy parts get more."""
    usable = (times >= FIRST_NOTE_SECONDS) & (measures[:, ENERGY] >= QUIET_ENERGY)
    wanted = int(round(per_second * (times[-1] - max(times[0], FIRST_NOTE_SECONDS))))
    min_gap = settings["min_gap_s"]
    chosen = []
    taken = np.zeros(len(times), bool)
    if pace is not None:
        pace = pace / max(float(pace[usable].mean()), 1e-9) if usable.any() else pace
        share = np.where(usable, pace, 0.0)
        expected = np.concatenate([[0.0], np.cumsum(wanted * share / max(share.sum(), 1e-9))])
        low_edge = np.searchsorted(times, times - PACE_WINDOW_S / 2)
        high_edge = np.searchsorted(times, times + PACE_WINDOW_S / 2)
        quota = expected[high_edge] - expected[low_edge]
        if PACE_BIAS:
            scores = scores * pace ** PACE_BIAS
    for step in np.argsort(scores)[::-1].tolist():
        if len(chosen) >= wanted or scores[step] <= floor:
            break
        if not usable[step]:
            continue
        if pace is not None and PACE_QUOTA and taken[low_edge[step]:high_edge[step]].sum() + 1 > quota[step] + 0.5:
            continue
        here_gap = gap_at(settings, pace, activity, step)
        low = int(np.searchsorted(times, times[step] - here_gap + 1e-6))
        high = int(np.searchsorted(times, times[step] + here_gap - 1e-6))
        if taken[low:high].any():
            continue
        chosen.append(step)
        taken[step] = True
    line_gap = settings["line_gap_s"]
    for first, last in spans:
        inside = sorted(s for s in np.flatnonzero(taken) if first <= s <= last)
        anchors = sorted(set([first] + inside + [last]))
        for a, b in zip(anchors, anchors[1:]):
            at = a
            while times[b] - times[at] > line_gap * stretch(pace, at, activity):
                gap_here = line_gap * stretch(pace, at, activity)
                reach_low = int(np.searchsorted(times, times[at] + max(min_gap, 0.55 * gap_here)))
                reach_high = int(np.searchsorted(times, times[at] + gap_here))
                options = [k for k in range(reach_low, min(reach_high + 1, b)) if usable[k] and times[b] - times[k] >= gap_at(settings, pace, activity, k) and times[k] - times[at] >= gap_at(settings, pace, activity, k)]
                if not options:
                    break
                best = max(options, key=lambda k: scores[k])
                taken[best] = True
                at = best
    return np.flatnonzero(fill_rests(taken, scores, times, usable, settings, pace, activity)).tolist()


def fill_rests(taken, scores, times, usable, settings, pace=None, activity=None):
    """Add the likeliest steps wherever the notes rest too long, including before the first note, so long as there's music to play them on.
    Given a pace, calm parts may rest longer (stretch), so the map breathes where the song does."""
    start = next((k for k in range(len(times)) if usable[k]), 0)
    marks = sorted(set([start] + np.flatnonzero(taken).tolist() + [len(times) - 1]))
    for a, b in zip(marks, marks[1:]):
        at = a
        while times[b] - times[at] > settings["max_rest_s"] * stretch(pace, at, activity):
            max_rest = settings["max_rest_s"] * stretch(pace, at, activity)
            low = int(np.searchsorted(times, times[at] + 0.6 * max_rest))
            high = int(np.searchsorted(times, times[at] + max_rest))
            options = [k for k in range(low, min(high + 1, b)) if usable[k] and times[b] - times[k] >= gap_at(settings, pace, activity, k) and times[k] - times[at] >= gap_at(settings, pace, activity, k)]
            if not options:
                # silence or too close to the next note: jump past it
                at = max(high, at + 1)
                if at >= b:
                    break
                continue
            best = max(options, key=lambda k: scores[k])
            taken[best] = True
            at = best
    return taken


def without_stray_opening(steps, times):
    """A few notes and then a long rest at the start reads as the map stopping suddenly, so the map starts after that rest instead."""
    start = 0
    for a, b in zip(steps[:STRAY_OPENING_MOMENTS], steps[1:]):
        if times[b] - times[a] >= STRAY_OPENING_REST:
            start = b
    return [s for s in steps if s >= start]


def plan_doubles(steps, scores, measures, times, per_beat, settings, offset):
    """Doubles land on strong hits on the beat, with room around them, after an even number of single swings so both arms are going the same way."""
    every = settings["double_every_beats"] * per_beat
    room = settings["double_room_s"]
    share = settings["double_share"]
    strong = np.percentile(scores[steps], 45) if steps else 1.0
    hit = (measures[:, KICK] >= 0.7) | (measures[:, CHORD] >= 0.8) | (measures[:, PHRASE] >= 0.35)
    kept, doubles = [], set()
    singles = 0
    skip_until = -1.0
    for step in steps:
        if times[step] < skip_until:
            continue
        if kept and times[step] - times[kept[-1]] >= settings["reset_s"]:
            singles = 0
        wanted = (step - offset) % every == 0 and scores[step] >= strong and bool(hit[step])
        if share is not None and 2 * (len(doubles) + 1) > share * (len(kept) + len(doubles) + 2):
            wanted = False
        # a single swing right before a strong hit can give way to it, which also keeps the arms matched if it leaves an even count
        tight = wanted and bool(kept) and times[step] - times[kept[-1]] < room and kept[-1] not in doubles and (singles - 1) % 2 == 0
        if tight:
            kept.pop()
            singles -= 1
        if wanted and (not kept or times[step] - times[kept[-1]] >= room or tight) and singles % 2 == 0:
            kept.append(step)
            doubles.add(step)
            singles = 0
            skip_until = times[step] + room
            continue
        kept.append(step)
        singles += 1
    return kept, doubles


def guided(steps, guide):
    """With a guide (guides.plan: marker steps and where else notes may go), the markers are the notes, plus the engine's own only in the undone parts."""
    if guide is None:
        return steps
    markers, allowed = guide[0], guide[1]
    return sorted(set(markers) | {s for s in steps if allowed[s]})


def make_notes_v3(scores, measures, times, per_beat, spans, settings, per_second, floor, offset, levels, level_name, guide=None):
    """The same rhythm, placed by the pattern engine. Returns (notes, chains, arcs). A guide's marker steps are always kept (see guided)."""
    import patterns_v3
    pace = pace_for(measures, per_beat) if settings.get("pacing") else None
    activity = activity_for(measures, times) if settings.get("pacing") else None
    steps = choose_steps(scores, times, measures, spans, settings, per_second, floor, pace, activity)
    steps = without_stray_opening(steps, times)
    profile = patterns_v3.PATTERNS[level_name]
    streams = (set(), [])
    if settings.get("combos"):
        # with pacing (engine version 2), the song's frantic parts become streams, a note on every step; version-1 songs build as before
        if settings.get("pacing"):
            streams = patterns_v3.stream_steps(measures, times, levels, per_beat, offset, activity)
        steps = patterns_v3.add_combo_runs(steps, measures, times, levels, pace=pace, activity=activity, streams=streams)
    steps = guided(steps, guide)
    keep = guide[0] if guide is not None else ()
    steps, doubles = patterns_v3.plan_doubles(steps, scores, measures, times, per_beat, settings, profile, levels, offset, streams=streams[0], keep=keep)
    return patterns_v3.place(steps, doubles, measures, times, levels, settings, profile, offset)


def make_notes(scores, measures, times, per_beat, spans, settings, per_second, floor, offset=0):
    pace = pace_for(measures, per_beat) if settings.get("pacing") else None
    activity = activity_for(measures, times) if settings.get("pacing") else None
    steps = choose_steps(scores, times, measures, spans, settings, per_second, floor, pace, activity)
    steps = without_stray_opening(steps, times)
    steps, doubles = plan_doubles(steps, scores, measures, times, per_beat, settings, offset)
    bar_steps = 4 * per_beat
    hands = [Hand(LEFT), Hand(RIGHT)]
    out = []
    lead = LEFT
    used = {}
    top_share, mix, reset_s = settings["top_share"], settings["direction_mix"], settings["reset_s"]
    previous = None
    for step in steps:
        bar = (step - offset) // bar_steps
        position = (step - offset) % bar_steps
        variant = (bar // 2) % 4
        snare = measures[step, SNARE]
        if previous is not None and times[step] - times[previous] >= reset_s:
            for hand in hands:
                hand.next_family = DOWN_FAMILY
                hand.last_dir = None
        previous = step
        if step in doubles and hands[LEFT].next_family == hands[RIGHT].next_family:
            family = hands[LEFT].next_family
            wide = pick([True, True, False], "wide", variant, position)
            lane_left, lane_right = (0, 3) if wide else (1, 2)
            row = 1 if wide and family == UP_FAMILY and pick([False, False, True], "double-row", variant, position, bar % 8) else 0
            for hand, lane in ((hands[LEFT], lane_left), (hands[RIGHT], lane_right)):
                options = [d for d in hand.direction_options(135) if d != inward(hand.side, family)] or hand.direction_options(135)
                preferred = outward(hand.side, family)
                direction = preferred if preferred in options and wide and top_share > 0.05 else (UP if family == UP_FAMILY else DOWN)
                if direction not in hand.direction_options(135):
                    direction = options[0]
                out.append(note(step, hand.side, lane, row, direction))
                hand.swing(direction, lane, step, row)
            lead = LEFT if lead == RIGHT else RIGHT
            continue
        hand = hands[lead]
        lead = LEFT if lead == RIGHT else RIGHT
        family = hand.next_family
        straight = UP if family == UP_FAMILY else DOWN
        options = hand.direction_options(135)
        choices = {"straight": straight, "outward": outward(hand.side, family), "inward": inward(hand.side, family)}
        weights = [choices[kind] for kind in mix]
        weights = [d for d in weights if d in options] or options
        direction = pick(weights, "dir", variant, position, hand.side, bar % 8)
        inner_lane = 1 if hand.side == LEFT else 2
        candidates = expert_position(hand.side, family, variant, position in (per_beat, 3 * per_beat) and snare > 1.3)
        if sum(1 for n in out if n["y"] == 2) + 1 > top_share * (len(out) + 1):
            candidates = [c for c in candidates if c[1] != 2] or [(inner_lane, 0)]
        fewest = min(used.get((hand.side, cx, cy, direction), 0) for cx, cy in candidates)
        x, y = pick([c for c in candidates if used.get((hand.side, c[0], c[1], direction), 0) == fewest], "pos", variant, position, hand.side, bar % 8)
        if x == inner_lane and direction == inward(hand.side, family):
            direction = straight
        # keep fast same-hand moves short
        if hand.last_step is not None and times[step] - times[hand.last_step] <= FAST_HAND_S:
            if abs(x - hand.lane) > 1:
                x = hand.lane + (1 if x > hand.lane else -1)
            if abs(y - hand.row) > 1:
                y = 1
        if family == DOWN_FAMILY and y == 2:
            y = 0
        if x in (1, 2) and y == 1:
            y = 0
        out.append(note(step, hand.side, x, y, direction))
        used[(hand.side, x, y, direction)] = used.get((hand.side, x, y, direction), 0) + 1
        hand.swing(direction, x, step, y)
    return out
