"""Notes for any difficulty, on straight or triplet grids, and on songs whose tempo drifts.

The flow rules are the ones the user has liked since the first build, from note_engine.py. What's new is that every limit is in seconds, not steps, so a 6/8 song and a straight one
get the same physical feel, and the rhythm comes from the v2 models, one for each difficulty. Targets come from analysis/difficulty_style.json: well-rated maps and the user's own.
"""
import numpy as np

from note_engine import (DOWN, DOWN_FAMILY, Hand, LEFT, RIGHT, TO_LEFT, TO_RIGHT, UP, UP_FAMILY, expert_position, inward, note, outward, pick)
from rhythm_features_v2 import NAMES

SYLLABLE, KICK, SNARE, CHORD, PHRASE, ENERGY = (NAMES.index(n) for n in ("syllable", "kick", "snare_hat", "chord_change", "phrase_start", "energy"))
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


def choose_steps(scores, times, measures, spans, settings, per_second, floor):
    """The steps people would most likely map, likeliest first, spaced apart, then enough to keep sung lines going."""
    usable = (times >= FIRST_NOTE_SECONDS) & (measures[:, ENERGY] >= QUIET_ENERGY)
    wanted = int(round(per_second * (times[-1] - max(times[0], FIRST_NOTE_SECONDS))))
    min_gap = settings["min_gap_s"]
    chosen = []
    taken = np.zeros(len(times), bool)
    for step in np.argsort(scores)[::-1].tolist():
        if len(chosen) >= wanted or scores[step] <= floor:
            break
        if not usable[step]:
            continue
        low = int(np.searchsorted(times, times[step] - min_gap + 1e-6))
        high = int(np.searchsorted(times, times[step] + min_gap - 1e-6))
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
            while times[b] - times[at] > line_gap:
                reach_low = int(np.searchsorted(times, times[at] + max(min_gap, 0.55 * line_gap)))
                reach_high = int(np.searchsorted(times, times[at] + line_gap))
                options = [k for k in range(reach_low, min(reach_high + 1, b)) if usable[k] and times[b] - times[k] >= min_gap]
                if not options:
                    break
                best = max(options, key=lambda k: scores[k])
                taken[best] = True
                at = best
    return np.flatnonzero(fill_rests(taken, scores, times, usable, settings)).tolist()


def fill_rests(taken, scores, times, usable, settings):
    """Add the likeliest steps wherever the notes rest too long, including before the first note, so long as there's music to play them on."""
    max_rest = settings["max_rest_s"]
    start = next((k for k in range(len(times)) if usable[k]), 0)
    marks = sorted(set([start] + np.flatnonzero(taken).tolist() + [len(times) - 1]))
    for a, b in zip(marks, marks[1:]):
        at = a
        while times[b] - times[at] > max_rest:
            low = int(np.searchsorted(times, times[at] + 0.6 * max_rest))
            high = int(np.searchsorted(times, times[at] + max_rest))
            options = [k for k in range(low, min(high + 1, b)) if usable[k] and times[b] - times[k] >= settings["min_gap_s"]]
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


def make_notes_v3(scores, measures, times, per_beat, spans, settings, per_second, floor, offset, levels, level_name):
    """The same rhythm, placed by the pattern engine. Returns (notes, chains, arcs)."""
    import patterns_v3
    steps = choose_steps(scores, times, measures, spans, settings, per_second, floor)
    steps = without_stray_opening(steps, times)
    profile = patterns_v3.PATTERNS[level_name]
    if settings.get("combos"):
        steps = patterns_v3.add_combo_runs(steps, measures, times, levels)
    steps, doubles = patterns_v3.plan_doubles(steps, scores, measures, times, per_beat, settings, profile, levels, offset)
    return patterns_v3.place(steps, doubles, measures, times, levels, settings, profile, offset)


def make_notes(scores, measures, times, per_beat, spans, settings, per_second, floor, offset=0):
    steps = choose_steps(scores, times, measures, spans, settings, per_second, floor)
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
