"""More varied, more fun note patterns, for songs where steady up-down swings felt too safe.

The user said Like Him, Nice For What and Pick Up the Phone were mostly "up down up down" with the occasional two-handed swing. Measured against well-rated Expert maps
(tools/research/pattern_study.py), those maps never moved a hand more than one lane, never crossed over, used 10 of the grid's positions where good maps use about 19,
and had two-handed hits on 3-5% of moments where good maps have 13-30%.

This keeps every flow rule the user has liked, and decides each note by weighing every valid choice against targets from those maps. Bigger tools go on bigger moments,
since "if everything is emphasized, nothing is emphasized" (BSMG intermediate mapping guide): a hand travels further, doubles get wider, and stacks and chains appear when
the music hits harder. Arcs, which the user uses in their own Quebec and Love Galore, link notes across long held vocals.

Parity: a forehand (palm) swing for the left hand is down, down-left, down-right or right, and its backhand is up, up-left, up-right or left. The right hand mirrors it.
Swings alternate forehand and backhand, and each swing turns at least 135 degrees from the hand's last, except after a rest.
"""
import hashlib

import numpy as np

from rhythm_features_v2 import NAMES

LEFT, RIGHT = 0, 1
UP, DOWN, TO_LEFT, TO_RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = range(8)
ANGLE = {UP: 90, DOWN: 270, TO_LEFT: 180, TO_RIGHT: 0, UP_LEFT: 135, UP_RIGHT: 45, DOWN_LEFT: 225, DOWN_RIGHT: 315}
FOREHAND = {LEFT: (DOWN, DOWN_LEFT, DOWN_RIGHT, TO_RIGHT), RIGHT: (DOWN, DOWN_LEFT, DOWN_RIGHT, TO_LEFT)}
BACKHAND = {LEFT: (UP, UP_LEFT, UP_RIGHT, TO_LEFT), RIGHT: (UP, UP_LEFT, UP_RIGHT, TO_RIGHT)}
DOWNWARD = (DOWN, DOWN_LEFT, DOWN_RIGHT)
SIDE = (TO_LEFT, TO_RIGHT)
KICK, SNARE, SYLLABLE, PHRASE, VOCAL_ACTIVE, OTHER, VOCAL_ONSET, ENERGY_COLUMN = (NAMES.index(n) for n in ("kick", "snare_hat", "syllable", "phrase_start", "vocal_active", "other", "vocal_onset", "energy"))
# face notes block the view of what's coming
BLOCKED = {(1, 1), (2, 1)}
FAST_HAND_S = 0.36
# a hand needs this long to travel two lanes, or three
TRAVEL_S = {2: 0.42, 3: 0.6}
# the two hands mustn't tangle: within this long of each other, the left hand stays left of the right
TANGLE_S = 0.5

# targets per difficulty, from well-rated Expert maps (median lane moves 26% none, 55% one, 10% two or more; crossovers 13%; side swings 4%; doubles 18% of moments),
# held back a little, since the user's taste is natural flow. The lower levels follow the BSMG guide: Hard keeps stacks short and crossovers simple, and Normal has none.
PATTERNS = {
    "Expert": {"stay": 0.25, "travel": 0.13, "cross": 0.05, "side": 0.04, "middle": 0.24, "top": 0.12, "stack": 0.02, "chains_per_min": 1.6, "arcs_per_min": 2.5, "doubles_by_level": (0.03, 0.06, 0.12, 0.18, 0.24)},
    "Hard": {"stay": 0.28, "travel": 0.08, "cross": 0.0, "side": 0.02, "middle": 0.22, "top": 0.06, "stack": 0.01, "chains_per_min": 0.0, "arcs_per_min": 2.0, "doubles_by_level": (0.02, 0.05, 0.09, 0.13, 0.17)},
    "Normal": {"stay": 0.3, "travel": 0.05, "cross": 0.0, "side": 0.0, "middle": 0.2, "top": 0.03, "stack": 0.0, "chains_per_min": 0.0, "arcs_per_min": 1.5, "doubles_by_level": (0.02, 0.05, 0.08, 0.12, 0.15)},
}


def chance(*key):
    """A repeatable number between 0 and 1: the same musical moment gets the same answer every build."""
    return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:4], "big") / 2 ** 32


def turn(a, b):
    diff = abs(ANGLE[a] - ANGLE[b]) % 360
    return min(diff, 360 - diff)


def outward(hand, direction):
    """Whether a diagonal points away from the middle."""
    if hand == LEFT:
        return direction in (UP_LEFT, DOWN_LEFT)
    return direction in (UP_RIGHT, DOWN_RIGHT)


def inward(hand, direction):
    if hand == LEFT:
        return direction in (UP_RIGHT, DOWN_RIGHT)
    return direction in (UP_LEFT, DOWN_LEFT)


def crossed(hand, x):
    return x >= 2 if hand == LEFT else x <= 1


def too_far(hand, x):
    """A crossover only reaches the nearest lane on the other side."""
    return x == 3 if hand == LEFT else x == 0


class Hand:
    def __init__(self, side):
        self.side = side
        self.forehand_next = True
        self.last_dir = None
        self.x = 1 if side == LEFT else 2
        self.y = 0
        self.t = -9.0
        self.recent = []
        self.used = {}

    def family(self):
        return FOREHAND[self.side] if self.forehand_next else BACKHAND[self.side]

    def swing(self, direction, x, y, t):
        self.last_dir = direction
        self.x, self.y, self.t = x, y, t
        self.forehand_next = direction not in FOREHAND[self.side]
        self.recent = (self.recent + [(x, y, direction)])[-6:]
        self.used[(x, y)] = self.used.get((x, y), 0) + 1

    def reset(self):
        self.forehand_next = True
        self.last_dir = None


def valid_spot(hand, x, y, direction, allow_cross):
    if (x, y) in BLOCKED:
        return False
    # the user's rule: no down swings on the top row
    if direction in DOWNWARD and y == 2:
        return False
    if direction in SIDE and y == 2:
        return False
    if crossed(hand, x) and (too_far(hand, x) or not (allow_cross and direction in (UP, DOWN) and y == 0)):
        return False
    inner = 1 if hand == LEFT else 2
    # an inward diagonal only makes sense from the outside of the grid
    if inward(hand, direction) and x in (inner, 3 - inner) and not crossed(hand, x):
        return False
    if inward(hand, direction) and crossed(hand, x):
        return False
    # side swings sit on the hand's own side, on the bottom two rows
    if direction in SIDE and crossed(hand, x):
        return False
    return True


class Shares:
    """Running tallies, so each choice can lean towards whatever is under its target."""
    def __init__(self):
        self.counts = {k: 0 for k in ("moves", "stay", "travel", "cross", "side", "middle", "top", "notes")}

    def share(self, key, of="notes"):
        return self.counts[key] / max(1, self.counts[of])

    def add(self, move, x, y, direction, hand):
        if move is not None:
            self.counts["moves"] += 1
            self.counts["stay"] += move == 0
            self.counts["travel"] += move >= 2
        self.counts["notes"] += 1
        self.counts["cross"] += crossed(hand, x)
        self.counts["side"] += direction in SIDE
        self.counts["middle"] += y == 1
        self.counts["top"] += y == 2


def emphasis(measures, step, level):
    """How hard the music hits here, from 0 to 1, combining the section's size and the hit itself."""
    hit = min(1.0, 0.5 * max(measures[step, KICK], measures[step, SNARE]) + 0.4 * measures[step, PHRASE] + 0.2 * measures[step, SYLLABLE])
    return 0.55 * level / 4 + 0.45 * hit


def single_choice(hand, other, t, e, profile, shares, step, allow_cross):
    """Weigh every valid placement for one hand's next swing and take the best."""
    options = []
    gap = t - hand.t
    for direction in hand.family():
        if hand.last_dir is not None and turn(hand.last_dir, direction) < 135:
            continue
        for x in range(4):
            for y in range(3):
                if not valid_spot(hand.side, x, y, direction, allow_cross and e > 0.55):
                    continue
                move = abs(x - hand.x)
                if gap < FAST_HAND_S and (move > 1 or abs(y - hand.y) > 1):
                    continue
                if move >= 2 and gap < TRAVEL_S[min(move, 3)]:
                    continue
                # don't tangle with the other hand
                if t - other.t < TANGLE_S and ((hand.side == LEFT and x >= other.x) or (hand.side == RIGHT and x <= other.x)):
                    continue
                score = 0.0
                # lean towards whatever is under its target
                stay_gap = profile["stay"] - shares.share("stay", "moves")
                travel_gap = profile["travel"] - shares.share("travel", "moves")
                score += 3 * (stay_gap if move == 0 else -0.5 * stay_gap)
                score += 4 * travel_gap * (move >= 2)
                score += 4 * (profile["cross"] - shares.share("cross")) * crossed(hand.side, x)
                score += 4 * (profile["side"] - shares.share("side")) * (direction in SIDE)
                score += 3 * (profile["middle"] - shares.share("middle")) * (y == 1)
                score += 4 * (profile["top"] - shares.share("top")) * (y == 2) - 1.5 * (y == 2 and e < 0.4)
                # bigger moments get bigger movement, and quiet ones stay compact
                score += (e - 0.45) * 1.2 * min(move, 2) + (e - 0.5) * 0.6 * (direction not in (UP, DOWN))
                # a straight swing reads easiest, so it stays the backbone
                score += 0.35 * (direction in (UP, DOWN)) + 0.15 * outward(hand.side, direction)
                # don't repeat the exact last spot too much, and keep each hand mostly on its own side
                score -= 0.4 * sum(1 for r in hand.recent[-4:] if r == (x, y, direction))
                score -= 0.25 * abs(x - (0.5 if hand.side == LEFT else 2.5))
                # spread across the grid: a spot this hand has rarely used gets a lift
                score += 0.6 * (1 - hand.used.get((x, y), 0) / max(1, max(hand.used.values(), default=1)))
                score += 0.2 * chance("single", step, hand.side, x, y, direction)
                options.append((score, x, y, direction, move))
    if not options:
        return None
    return max(options)


DOUBLE_SHAPES = [((0, 0), (3, 0)), ((1, 0), (2, 0)), ((0, 1), (3, 1)), ((0, 0), (3, 1)), ((0, 1), (3, 0)), ((1, 0), (3, 0)), ((0, 0), (2, 0)), ((0, 2), (3, 2)), ((1, 2), (2, 2))]


def double_choice(hands, t, e, step):
    """Both hands at once: a shape and directions that suit each hand's flow, wider when the music hits harder."""
    left, right = hands
    options = []
    for (lx, ly), (rx, ry) in DOUBLE_SHAPES:
        for ld in left.family():
            for rd in right.family():
                if ld in SIDE or rd in SIDE:
                    continue
                if left.last_dir is not None and turn(left.last_dir, ld) < 135:
                    continue
                if right.last_dir is not None and turn(right.last_dir, rd) < 135:
                    continue
                if not (valid_spot(LEFT, lx, ly, ld, False) and valid_spot(RIGHT, rx, ry, rd, False)):
                    continue
                # pointing outwards from both top corners is an uncomfortable double
                if ly == 2 and ry == 2 and outward(LEFT, ld) and outward(RIGHT, rd):
                    continue
                gap_l, gap_r = t - left.t, t - right.t
                # neither hand may land on the other's side of where that hand just was
                if (gap_r < TANGLE_S and lx >= right.x) or (gap_l < TANGLE_S and rx <= left.x):
                    continue
                if (abs(lx - left.x) > 1 and gap_l < TRAVEL_S[2]) or (abs(rx - right.x) > 1 and gap_r < TRAVEL_S[2]):
                    continue
                width = rx - lx
                score = (e - 0.5) * 0.8 * width + 0.3 * (ld in (UP, DOWN) and rd in (UP, DOWN)) + 0.25 * (outward(LEFT, ld) and outward(RIGHT, rd))
                score -= 0.6 * (ly == 2) * (e < 0.7) + 0.4 * (ly != ry)
                score -= 0.5 * sum(1 for r in left.recent[-4:] if r[:2] == (lx, ly))
                score += 0.3 * chance("double", step, lx, ly, rx, ry, ld, rd)
                options.append((score, (lx, ly, ld), (rx, ry, rd)))
    return max(options) if options else None


# combos only where the drums and voice are at least as busy as usual for the song, and the part isn't quiet
COMBO_ACTIVITY = 1.0
COMBO_PACE = 0.9


def add_combo_runs(steps, measures, times, levels, max_run=8, pace=None, activity=None):
    """Combo areas: where the voice or the drums keep moving, fill the space between notes with back-to-back steps, so busy parts play as continuous runs.

    Only on the step grid, never finer, and each run stops after max_run notes for a breath. The user asked for these on Expert: fewer gaps, more combos, not harder cuts.
    """
    chosen = set(steps)
    busy = (measures[:, SYLLABLE] > 0.2) | (measures[:, VOCAL_ONSET] > 0.5) | (measures[:, SNARE] > 0.5) | (measures[:, OTHER] > 0.7)
    ordered = sorted(chosen)
    for a, b in zip(ordered, ordered[1:]):
        if not 2 <= b - a <= 3 or levels[a] < 2:
            continue
        for k in range(a + 1, b):
            # given a pace, only parts that aren't quiet and where the drums and voice are busy for this song get combos, so calm parts stay calm
            if busy[k] and measures[k, ENERGY_COLUMN] >= 0.42 and (pace is None or pace[k] >= COMBO_PACE) and (activity is None or activity[k] >= COMBO_ACTIVITY):
                chosen.add(k)
    # break any run that grew past max_run, so the player gets a breath
    out, run = [], 0
    for k in sorted(chosen):
        run = run + 1 if out and k - out[-1] == 1 else 1
        if run > max_run:
            run = 0
            continue
        out.append(k)
    return out


def plan_doubles(steps, scores, measures, times, per_beat, settings, profile, levels, offset=0):
    """Doubles on strong hits on the beat, more of them in bigger sections, after an even number of single swings so both arms are going the same way."""
    room = settings["double_room_s"]
    hit = (measures[:, KICK] >= 0.6) | (measures[:, SNARE] >= 0.9) | (measures[:, PHRASE] >= 0.35)
    kept, doubles = [], set()
    per_level = {}
    singles = 0
    skip_until = -1.0
    for step in steps:
        if times[step] < skip_until:
            continue
        level = levels[step]
        seen, made = per_level.get(level, (0, 0))
        if kept and times[step] - times[kept[-1]] >= settings["reset_s"]:
            singles = 0
        wanted = (step - offset) % per_beat == 0 and bool(hit[step]) and made < profile["doubles_by_level"][level] * (seen + 1)
        tight = wanted and bool(kept) and times[step] - times[kept[-1]] < room and kept[-1] not in doubles and (singles - 1) % 2 == 0
        # in bigger sections a strong double may replace the single just before it, so both arms line up for it
        tight = tight or (wanted and level >= 2 and bool(kept) and kept[-1] not in doubles and singles % 2 == 1 and scores[step] > scores[kept[-1]])
        if tight:
            kept.pop()
            singles -= 1
        if wanted and (not kept or times[step] - times[kept[-1]] >= room or tight) and singles % 2 == 0:
            kept.append(step)
            doubles.add(step)
            per_level[level] = (seen + 1, made + 1)
            singles = 0
            skip_until = times[step] + room
            continue
        kept.append(step)
        per_level[level] = (seen + 1, made)
        singles += 1
    return kept, doubles


def place(steps, doubles, measures, times, levels, settings, profile, offset=0):
    """Notes, chains and arcs for chosen steps. Returns (notes, chains, arcs): notes as dicts, chains as (head index, tail step, tail x, tail y), arcs as (head index, tail index)."""
    hands = [Hand(LEFT), Hand(RIGHT)]
    shares = Shares()
    notes = []
    lead = LEFT
    previous = None
    for step in steps:
        t = times[step]
        level = levels[step]
        e = emphasis(measures, step, level)
        if previous is not None and t - times[previous] >= settings["reset_s"]:
            for hand in hands:
                hand.reset()
        previous = step
        if step in doubles and hands[LEFT].forehand_next == hands[RIGHT].forehand_next:
            choice = double_choice(hands, t, e, step)
            if choice:
                _, (lx, ly, ld), (rx, ry, rd) = choice
                for hand, (x, y, d) in ((hands[LEFT], (lx, ly, ld)), (hands[RIGHT], (rx, ry, rd))):
                    shares.add(abs(x - hand.x), x, y, d, hand.side)
                    notes.append({"step": step, "c": hand.side, "x": x, "y": y, "d": d})
                    hand.swing(d, x, y, t)
                lead = 1 - lead
                continue
        hand, other = hands[lead], hands[1 - lead]
        choice = single_choice(hand, other, t, e, profile, shares, step, profile["cross"] > 0)
        if choice is None:
            # nothing fits this hand's flow here, so the other hand takes it, rather than the note disappearing
            hand, other = other, hand
            choice = single_choice(hand, other, t, e, profile, shares, step, profile["cross"] > 0)
            if choice is None:
                continue
        lead = 1 - hand.side
        _, x, y, d, move = choice
        shares.add(move if hand.t > 0 else None, x, y, d, hand.side)
        notes.append({"step": step, "c": hand.side, "x": x, "y": y, "d": d})
        # on the hardest hits, a single becomes a two-note stack, played in one swing
        if profile["stack"] > 0 and e > 0.8 and d in (UP, DOWN) and x in (0, 3) and y == 0 and shares.counts["notes"] * profile["stack"] > sum(1 for n in notes if n.get("stacked")):
            notes.append({"step": step, "c": hand.side, "x": x, "y": 1, "d": d, "stacked": True})
        hand.swing(d, x, y, t)
    return notes, plan_chains(notes, measures, times, levels, profile), plan_arcs(notes, measures, times, levels, profile)


def plan_chains(notes, measures, times, levels, profile):
    """A chain on a big hit that's followed by a burst of sound, where the swing has room to carry through."""
    if profile["chains_per_min"] <= 0:
        return []
    minutes = (times[notes[-1]["step"]] - times[notes[0]["step"]]) / 60
    budget = int(profile["chains_per_min"] * minutes)
    candidates = []
    by_hand = {LEFT: [n for n in notes if n["c"] == LEFT], RIGHT: [n for n in notes if n["c"] == RIGHT]}
    moments = {}
    for n in notes:
        moments.setdefault(n["step"], []).append(n)
    for hand, rows in by_hand.items():
        for a, b in zip(rows, rows[1:]):
            if len(moments[a["step"]]) > 1 or levels[a["step"]] < 3 or a["y"] != 1 and a["d"] in (UP, DOWN):
                continue
            # the tail follows the swing, one square on
            tail = {UP: (a["x"], a["y"] + 1), DOWN: (a["x"], a["y"] - 1), TO_LEFT: (a["x"] - 1, a["y"]), TO_RIGHT: (a["x"] + 1, a["y"])}.get(a["d"])
            if tail is None or not (0 <= tail[0] <= 3 and 0 <= tail[1] <= 2) or tail in BLOCKED:
                continue
            gap = times[b["step"]] - times[a["step"]]
            if gap < 0.55:
                continue
            following = measures[a["step"] + 1:a["step"] + 3, OTHER].max(initial=0) + measures[a["step"] + 1:a["step"] + 3, SNARE].max(initial=0)
            candidates.append((following, a, tail))
    chosen = []
    for _, head, tail in sorted(candidates, key=lambda c: -c[0])[:budget]:
        chosen.append((notes.index(head), head["step"] + 1, tail[0], tail[1]))
    return chosen


def plan_arcs(notes, measures, times, levels, profile):
    """Arcs link a hand's notes across a long held vocal, so a sustained note looks sustained. Kept sparse, as the guide warns they clutter the view."""
    if profile["arcs_per_min"] <= 0:
        return []
    minutes = (times[notes[-1]["step"]] - times[notes[0]["step"]]) / 60
    budget = int(profile["arcs_per_min"] * minutes)
    moments = {}
    for n in notes:
        moments.setdefault(n["step"], []).append(n)
    candidates = []
    for hand in (LEFT, RIGHT):
        rows = [n for n in notes if n["c"] == hand]
        for a, b in zip(rows, rows[1:]):
            gap = times[b["step"]] - times[a["step"]]
            if gap < 0.8 or gap > 3.0 or len(moments[a["step"]]) > 1 or len(moments[b["step"]]) > 1 or a.get("stacked") or b.get("stacked"):
                continue
            # the other hand may play once in between, on its own side, where the arc won't cross it
            between = [n for n in notes if a["step"] < n["step"] < b["step"]]
            if len(between) > 1 or any(n["c"] == hand or crossed(n["c"], n["x"]) for n in between):
                continue
            held = measures[a["step"]:b["step"], VOCAL_ACTIVE].mean() - 0.6 * measures[a["step"] + 1:b["step"], SYLLABLE].max(initial=0)
            if held > 0.5 and levels[a["step"]] <= 3:
                candidates.append((held, notes.index(a), notes.index(b)))
    chosen = sorted(candidates, key=lambda c: -c[0])[:budget]
    return [(h, tl) for _, h, tl in sorted(chosen, key=lambda c: c[1])]
