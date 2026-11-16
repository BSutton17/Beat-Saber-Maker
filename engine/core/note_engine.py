"""Note engine for Expert and Hard: a model trained on well-rated maps chooses the 1/4-beat rhythm, and flow rules hold on every note.

A step is 1/4 of a beat, and a bar is 4 beats (16 steps). Hard uses the same model, so it follows the same lyrics, with room between swings
and simpler placement. Easy will be designed separately once it's wanted.
"""
import hashlib

import numpy as np

import vocal_lines

LEFT, RIGHT = 0, 1
UP, DOWN, TO_LEFT, TO_RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = range(8)
ANGLE = {UP: 90, DOWN: 270, TO_LEFT: 180, TO_RIGHT: 0, UP_LEFT: 135, UP_RIGHT: 45, DOWN_LEFT: 225, DOWN_RIGHT: 315}
UP_FAMILY = (UP, UP_LEFT, UP_RIGHT)
DOWN_FAMILY = (DOWN, DOWN_LEFT, DOWN_RIGHT)


def turn(a, b):
    diff = abs(ANGLE[a] - ANGLE[b]) % 360
    return min(diff, 360 - diff)


def pick(options, *key):
    """A repeatable choice: the same musical situation gets the same answer unless the variant changes."""
    digest = hashlib.sha256(repr(key).encode()).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def strengths(features, step):
    kick = max(features["sub"][step], features["low"][step])
    snare = features["mid"][step]
    hat = features["high"][step]
    return kick, snare, hat


def salience(features, step):
    kick, snare, hat = strengths(features, step)
    return 1.0 * kick + 1.0 * snare + 0.4 * hat


# Rate is note moments per second, and Hard also stays under a share of what Expert found in the song. min_gap is the fewest steps between moments, so
# nothing in Hard needs quick 1/4 runs. double_room is the steps kept clear around a double, so the same hand always has time. double_share caps doubles
# as a share of notes: the user's liked maps are at 5-9%, up to 24% on Love Galore. Top_share is the most of the notes that may sit on the top row.
# Hard's values are a first guess until the user plays one.
DIFFICULTY = {
    "Expert": {"per_second": 3.2, "instrument_rate": 2.2, "min_gap": 1, "double_every": 8, "double_room": 2, "double_share": 0.12, "top_share": 0.13, "direction_mix": ("straight", "straight", "straight", "outward", "inward")},
    "Hard": {"per_second": 2.2, "share_of_expert": 0.72, "instrument_rate": 1.5, "min_gap": 2, "double_every": 16, "double_room": 4, "double_share": 0.10, "top_share": 0.06, "direction_mix": ("straight", "straight", "straight", "straight", "outward")},
}
# After a rest this long both hands can be back in place, so the next swing may be a down again. Goosebumps, at 65 BPM, used a 1 beat rest, which is 0.92 s.
# Counting in beats instead would reset constantly in faster or sparser songs, breaking the arm flow.
RESET_SECONDS = 0.9

# ---- rhythm ----

STEPS_PER_BAR = 16
# the model's score at the density the user loved in Goosebumps. Other songs may not go below it just to reach a density.
SCORE_FLOOR = 0.37
SYLLABLE, VOCAL_LEVEL, VOCAL_ONSET, KICK, SNARE, BASS, OTHER = 0, 3, 4, 5, 8, 11, 12
NO_VOCALS_BELOW = 0.1
CLEAR_SOUND = 0.7
FINALE_CLEAR_SOUND = 0.5


def instrument_score(measures):
    return np.maximum(np.maximum(measures[:, KICK], measures[:, SNARE]), 0.85 * np.maximum(measures[:, BASS], measures[:, OTHER]))


def instrument_steps(measures, offset, period, per_second, finale_step=None, finale_per_second=None):
    """Bars with no vocals follow the clearest instrument hits on half-beats. The model was trained on rap, so it can't judge these."""
    total = len(measures) - 1
    level = instrument_score(measures)
    chosen = []
    bar_seconds = STEPS_PER_BAR / 4 * period
    for start in range(offset % STEPS_PER_BAR - STEPS_PER_BAR, total + 1, STEPS_PER_BAR):
        bar = [s for s in range(start, start + STEPS_PER_BAR) if 0 <= s <= total]
        if not bar or measures[bar, VOCAL_LEVEL].mean() >= NO_VOCALS_BELOW:
            continue
        finale = finale_step is not None and start >= finale_step
        rate = finale_per_second if finale else per_second
        budget = int(round(rate * bar_seconds))
        clear = FINALE_CLEAR_SOUND if finale else CLEAR_SOUND
        half_beats = [s for s in bar if (s - offset) % 2 == 0 and level[s] >= clear]
        picked = []
        for s in sorted(half_beats, key=lambda s: -level[s]):
            if len(picked) >= budget:
                break
            picked.append(s)
        chosen += picked
    return chosen


def choose_steps(scores, per_second, period, spans, floor=SCORE_FLOOR, min_gap=1):
    """Take the steps people would most likely map, across the song, so busy lines get more notes than sparse ones.

    Inside a vocal line, no gap may pass 2 beats, so the notes do not appear to stop while the vocals carry on.
    """
    wanted = int(round(per_second * (len(scores) - 1) * period / 4))
    if min_gap > 1:
        # likeliest first, skipping anything too close, until the count is met
        chosen = set()
        for step in np.argsort(scores)[::-1].tolist():
            if len(chosen) >= wanted or scores[step] <= floor:
                break
            if all(other not in chosen for other in range(step - min_gap + 1, step + min_gap)):
                chosen.add(step)
    else:
        threshold = max(np.sort(scores)[::-1][min(wanted, len(scores) - 1)], floor)
        chosen = set(np.flatnonzero(scores > threshold).tolist())
    for first, last in spans:
        inside = sorted(s for s in chosen if first <= s <= last)
        anchors = sorted(set([first] + inside + [last]))
        for x, y in zip(anchors, anchors[1:]):
            s = x + 4
            while y - s >= 2:
                best = max(range(s - 1, min(s + 2, y)), key=lambda k: scores[k])
                chosen.add(best)
                s = best + 4
    return sorted(chosen)


def spaced_out(steps, scores, min_gap):
    """Keep the likeliest steps, dropping any that land too close to one already kept."""
    if min_gap <= 1:
        return steps
    kept = []
    for step in sorted(steps, key=lambda s: -scores[s]):
        if all(abs(step - other) >= min_gap for other in kept):
            kept.append(step)
    return sorted(kept)


def plan_doubles(steps, scores, features, high, offset=0, allowed=None, every=8, room=2, share=None):
    """Put doubles on strong kicks every 2 beats, clearing 1/4 beat either side so that both hands have time.

    A double needs an even number of single swings since the last reset, so that both hands are going the same way.
    """
    kept = []
    doubles = set()
    singles = 0
    skip_until = None
    for i, step in enumerate(steps):
        if skip_until is not None and step < skip_until:
            continue
        if kept and step - kept[-1] >= 4:
            singles = 0
        index = min(step, len(features["sub"]) - 1)
        wanted = (step - offset) % every == 0 and scores[step] >= high and strengths(features, index)[0] > 1.0
        if allowed is not None:
            wanted = (step - offset) % every == 0 and bool(allowed[step])
        # notes so far, counting doubles twice, with this one added
        if share is not None and 2 * (len(doubles) + 1) > share * (len(kept) + len(doubles) + 2):
            wanted = False
        if wanted:
            tight = room == 2 and bool(kept) and step - kept[-1] == 1 and kept[-1] not in doubles
            count = singles - 1 if tight else singles
            clear = not kept or step - kept[-1] >= room or tight
            if clear and count % 2 == 0:
                if tight:
                    kept.pop()
                kept.append(step)
                doubles.add(step)
                singles = 0
                skip_until = step + room
                continue
        kept.append(step)
        singles += 1
    return kept, doubles


def build_events(section, vocal, scores, measures, features, per_second, instruments, settings):
    period = 60 / section["bpm"]
    start = section["first_beat_s"]
    spans = [(int(np.ceil((a - start) / period * 4)), int(np.floor((b - start) / period * 4))) for a, b in vocal_lines.find_lines(vocal, section)]
    offset = section.get("bar_offset_beats", 0) * 4
    steps = choose_steps(scores, per_second, period, spans, min_gap=settings["min_gap"])
    syllable = measures[:, 0]
    # instrumental sections can have no steps above the floor, and then the instrument rule supplies them all
    high = np.percentile(scores[steps], 40) if steps else 1.0
    allowed = None
    if instruments:
        finale = instruments.get("finale_step")
        extra = instrument_steps(measures, offset, period, instruments["per_second"], finale, instruments.get("finale_per_second"))
        steps = sorted(set(steps) | set(extra))
        # stems decide doubles here: a real kick with a likely note, or a big instrument hit in a bar with no vocals
        no_vocals = np.zeros(len(scores), bool)
        no_vocals[extra] = True
        allowed = (measures[:, KICK] >= 0.7) & (scores >= high) | (no_vocals & (instrument_score(measures) >= 1.1))
    steps = spaced_out(steps, scores, settings["min_gap"])
    steps, doubles = plan_doubles(steps, scores, features, high, offset, allowed, settings["double_every"], settings["double_room"], settings["double_share"])
    events = [{"step": s, "score": float(scores[s]), "kind": "syllable" if syllable[s] > 0.2 else "sound", "double": s in doubles} for s in steps]
    for before, event, after in zip([None] + events[:-1], events, events[1:] + [None]):
        event["gap_before"] = event["step"] - before["step"] if before else 999
        event["gap_after"] = after["step"] - event["step"] if after else 999
    return events


# ---- flow ----

class Hand:
    def __init__(self, side):
        self.side = side
        self.next_family = DOWN_FAMILY
        self.last_dir = None
        self.last_step = None
        self.lane = 1 if side == LEFT else 2
        self.row = 0

    def direction_options(self, min_turn):
        options = list(self.next_family)
        if self.last_dir is None:
            return options
        return [d for d in options if turn(self.last_dir, d) >= min_turn]

    def swing(self, direction, lane, step, row=0):
        self.last_dir = direction
        self.lane = lane
        self.row = row
        self.last_step = step
        self.next_family = UP_FAMILY if direction in DOWN_FAMILY else DOWN_FAMILY


def outward(side, family):
    if family == DOWN_FAMILY:
        return DOWN_LEFT if side == LEFT else DOWN_RIGHT
    return UP_LEFT if side == LEFT else UP_RIGHT


def inward(side, family):
    if family == DOWN_FAMILY:
        return DOWN_RIGHT if side == LEFT else DOWN_LEFT
    return UP_RIGHT if side == LEFT else UP_LEFT


def expert_position(side, family, variant, big):
    """Candidate placements. The centre squares of the middle row are never used, and down swings never start on the top row."""
    inner = 1 if side == LEFT else 2
    outer = 0 if side == LEFT else 3
    low = [(inner, 0), (outer, 0)]
    if family == DOWN_FAMILY:
        if big:
            return [(outer, 1), (outer, 0)]
        return {0: low + [(outer, 1)], 1: [(inner, 0), (outer, 1)], 2: [(outer, 0), (outer, 1), (inner, 0)],
                3: [(inner, 0), (outer, 0), (outer, 1)]}[variant % 4]
    if big:
        return [(outer, 0), (outer, 1)]
    return {0: low + [(outer, 1)], 1: [(outer, 0), (outer, 1), (inner, 2)], 2: [(inner, 0), (outer, 2), (outer, 1)],
            3: [(outer, 1), (inner, 0), (outer, 2)]}[variant % 4]


def note(step, side, x, y, d):
    return {"step": step, "c": side, "x": x, "y": y, "d": d}


def expert_notes(features, section, vocal, scores, measures, per_second=3.2, instruments=None, settings=DIFFICULTY["Expert"]):
    """Notes for one section. Other difficulties pass their own settings from DIFFICULTY."""
    events = build_events(section, vocal, scores, measures, features, per_second, instruments, settings)
    step_seconds = 60 / section["bpm"] / 4
    top_share, direction_mix = settings["top_share"], settings["direction_mix"]
    offset = section.get("bar_offset_beats", 0) * 4
    hands = [Hand(LEFT), Hand(RIGHT)]
    out = []
    lead = LEFT
    used = {}
    for event in events:
        step = event["step"]
        bar = (step - offset) // STEPS_PER_BAR
        position = (step - offset) % STEPS_PER_BAR
        variant = (bar // 2) % 4
        index = min(step, len(features["sub"]) - 1)
        kick, snare, _ = strengths(features, index)
        # a long enough rest gives both hands time to return
        if event["gap_before"] * step_seconds >= RESET_SECONDS:
            for hand in hands:
                hand.next_family = DOWN_FAMILY
                hand.last_dir = None
        if event["double"] and hands[LEFT].next_family == hands[RIGHT].next_family:
            family = hands[LEFT].next_family
            wide = pick([True, True, False], "wide", variant, position)
            lane_left, lane_right = (0, 3) if wide else (1, 2)
            row = 1 if wide and family == UP_FAMILY and pick([False, False, True], "double-row", variant, position, bar % 8) else 0
            for hand, lane in ((hands[LEFT], lane_left), (hands[RIGHT], lane_right)):
                options = [d for d in hand.direction_options(135) if d != inward(hand.side, family)] or hand.direction_options(135)
                preferred = outward(hand.side, family)
                direction = preferred if preferred in options and wide else (UP if family == UP_FAMILY else DOWN)
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
        weights = [choices[kind] for kind in direction_mix]
        weights = [d for d in weights if d in options] or options
        direction = pick(weights, "dir", variant, position, hand.side, bar % 8)
        inner_lane = 1 if hand.side == LEFT else 2
        candidates = expert_position(hand.side, family, variant, position in (4, 12) and snare > 1.3)
        # keep the top row an occasional reach, which also protects the shoulders
        if sum(1 for n in out if n["y"] == 2) + 1 > top_share * (len(out) + 1):
            candidates = [c for c in candidates if c[1] != 2] or [(inner_lane, 0)]
        fewest = min(used.get((hand.side, cx, cy, direction), 0) for cx, cy in candidates)
        x, y = pick([c for c in candidates if used.get((hand.side, c[0], c[1], direction), 0) == fewest], "pos", variant, position, hand.side, bar % 8)
        # an inward diagonal only makes sense from the outside of the grid
        if x == inner_lane and direction == inward(hand.side, family):
            direction = straight
        # keep fast same-hand moves short
        if hand.last_step is not None and step - hand.last_step <= 2:
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


# ---- lights ----

def light_events(features, bar_steps=16, offset=0, look=None):
    """Lane jobs follow the report: kick on center, snare on back lasers, hats on rings, phrases on lasers.

    Goosebumps uses the defaults. A look changes the colors, energy and boost for a song's section.
    """
    look = look or {}
    mode = look.get("mode", "full")
    climax = look.get("climax_step")
    white_fade = 11
    events = []
    boosts = []
    total = len(features["sub"])
    red_on, red_fade = 5, 7
    blue_on, blue_fade = 1, 3
    for step in range(total):
        bar = (step - offset) // bar_steps
        position = (step - offset) % bar_steps
        kick, snare, hat = strengths(features, step)
        phrase = bar // 4
        fade = red_fade if phrase % 2 == 0 else blue_fade
        other = blue_fade if phrase % 2 == 0 else red_fade
        if look.get("swap"):
            fade, other = other, fade
        if look.get("blue"):
            fade, other = blue_fade, (white_fade if phrase % 2 == 0 else blue_fade)
        if mode == "sparse":
            if position == 0 and bar % 2 == 0:
                events.append((step, 0, other))
            if position == 0 and kick > 1.0:
                events.append((step, 4, fade))
            continue
        if mode == "calm":
            if kick > 1.2:
                events.append((step, 4, fade))
            if position % 4 == 0 and hat > 0.5:
                events.append((step, 1, fade))
            if position % 8 == 0 or features["vocal"][step] > 1.0:
                events.append((step, 2 if step % 3 == 0 else 3, other))
            if position == 0:
                events.append((step, 12, 1))
                events.append((step, 13, 1))
                events.append((step, 9, 0))
            continue
        if kick > 1.0:
            events.append((step, 4, fade))
        if position in (4, 12) and snare > 1.0:
            events.append((step, 0, other))
        if (step % 2 == 0 and hat > 0.5) or hat > 1.0:
            events.append((step, 1, other if position % 4 == 0 else fade))
        # lasers follow the estimated vocal syllables
        if position % 8 == 0 or features["vocal"][step] > 0.8:
            lane = 2 if step % 3 == 0 else 3
            events.append((step, lane, fade))
        if position == 0:
            low_speed, high_speed = look.get("speed", (2, 4))
            if climax is not None and step >= climax:
                low_speed, high_speed = 5, 7
            speed = low_speed if bar % 4 else high_speed
            events.append((step, 12, speed))
            events.append((step, 13, speed))
            events.append((step, 9, 0))
        if position == 0 and bar % 2 == 0:
            events.append((step, 8, 0))
        if look.get("boost", "cycle") == "cycle" and position == 0 and bar % 8 == 0:
            boosts.append((step, (bar // 8) % 2 == 1))
    if look.get("boost") == "on":
        boosts.append((0, True))
    if climax is not None:
        boosts.append((climax, True))
    return events, boosts
