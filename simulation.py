import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

import sys
import os
import pickle


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from src.models.poisson import get_lambda


# FIFA World Cup 2026 Group Stages
world_cup_2026_groups = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

# Hosts are the only teams with a genuine home advantage at WC 2026.
HOST_NATIONS = {"USA", "Mexico", "Canada"}


def resolve_venue(team_a, team_b):
    """Decide the venue for a match.

    Returns (neutral, home_team):
      - exactly one host  -> not neutral, that host is the home side
      - neither / both hosts -> neutral, no advantage to anyone
    """
    a_host = team_a in HOST_NATIONS
    b_host = team_b in HOST_NATIONS
    if a_host ^ b_host:                       # exactly one host
        return False, (team_a if a_host else team_b)
    return True, None

ROUND_OF_32 = [
    (("A", 1), ("B", 2)),
    (("C", 1), ("D", 2)),
    (("E", 1), ("F", 2)),
    (("G", 1), ("H", 2)),
    (("I", 1), ("J", 2)),
    (("K", 1), ("L", 2)),

    (("B", 1), ("A", 2)),
    (("D", 1), ("C", 2)),
    (("F", 1), ("E", 2)),
    (("H", 1), ("G", 2)),
    (("J", 1), ("I", 2)),
    (("L", 1), ("K", 2)),
]


def build_match_prob_cache(teams, model, save_path="models/prob_cache.pkl"):
    """
    Builds and caches match probabilities for all team pairs.
    Uses symmetry + disk persistence for speed.
    """
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)

    cache = {}

    for t1, t2 in combinations(teams, 2):

        neutral, home = resolve_venue(t1, t2)

        if neutral:
            probs = model.predict(t1, t2, neutral=True)   # symmetric
        else:
            # Orient so the host is the home side, then store both directions
            # flipped so the host stays advantaged whichever order it's
            # looked up in during simulation.
            opp = t2 if home == t1 else t1
            host_probs = model.predict(home, opp, neutral=False)
            p_home = np.array([
                host_probs.get("home_win", 0),   # host wins
                host_probs.get("draw", 0),
                host_probs.get("away_win", 0),    # opponent wins
            ], dtype=float)
            p_home = np.clip(p_home, 0, 1)
            p_home = p_home / p_home.sum() if p_home.sum() else np.array([0.33, 0.34, 0.33])
            cache[(home, opp)] = p_home
            cache[(opp, home)] = np.array([p_home[2], p_home[1], p_home[0]])
            continue

        p = np.array([
            probs.get("home_win", 0),
            probs.get("draw", 0),
            probs.get("away_win", 0)
        ], dtype=float)

        p = np.clip(p, 0, 1)

        if p.sum() == 0:
            p = np.array([0.33, 0.34, 0.33])
        else:
            p = p / p.sum()

        cache[(t1, t2)] = p
        cache[(t2, t1)] = np.array([p[2], p[1], p[0]])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(cache, f)

    return cache


# ===========================================================================
# OPTIMIZED SIMULATION CORE
#
# Speed-ups vs the original:
#  - np.random.choice (which validates its args on every call) is replaced by a
#    single uniform draw compared against precomputed cumulative probabilities.
#  - Per-pair Poisson lambdas + outcome CDFs are precomputed ONCE into a match
#    cache, so get_lambda / exp / resolve_venue are not called inside the loop.
#  - Knockout matches skip goal sampling (they only need a winner).
#  - Progression is accumulated from per-stage team lists instead of building a
#    full {team: {stage: 0}} dict every single tournament.
# ===========================================================================

# Module-level aliases avoid attribute lookups in the hot loop.
_random = np.random.random
_poisson = np.random.poisson


def build_match_cache(teams, prob_cache, elo_dict, beta_home, beta_away):
    """Precompute, for every ordered pair, the data the match sampler needs:
        (cdf0, cdf1, lam1, lam2)
    where cdf0 = P(team1 win), cdf1 = P(team1 win)+P(draw), and lam1/lam2 are the
    venue-aware expected goals. Built once per run_simulations call."""
    mc = {}
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            p = prob_cache.get((t1, t2))
            if p is None:
                continue
            cdf0 = float(p[0])
            cdf1 = float(p[0] + p[1])

            neutral, home = resolve_venue(t1, t2)
            if neutral:
                lam1, lam2 = get_lambda(t1, t2, elo_dict, beta_home, beta_away, neutral=True)
            elif home == t1:
                lam1, lam2 = get_lambda(t1, t2, elo_dict, beta_home, beta_away, neutral=False)
            else:
                lam2, lam1 = get_lambda(t2, t1, elo_dict, beta_home, beta_away, neutral=False)

            mc[(t1, t2)] = (cdf0, cdf1, float(lam1), float(lam2))
    return mc


def simulate_match_fast(t1, t2, mc, need_goals=True):
    """Sample one match. Returns (result, g1, g2) where result is t1, t2, or 'draw'.
    Goals are only sampled when need_goals (group stage); knockout skips them."""
    cdf0, cdf1, lam1, lam2 = mc[(t1, t2)]
    r = _random()

    if not need_goals:
        if r < cdf0:
            return t1, 0, 0
        elif r < cdf1:
            return "draw", 0, 0
        return t2, 0, 0

    g1 = _poisson(lam1)
    g2 = _poisson(lam2)
    if r < cdf0:                       # team1 win
        return t1, max(g1, g2 + 1), min(g1, g2)
    elif r < cdf1:                     # draw
        g = min(g1, g2)
        return "draw", g, g
    return t2, min(g1, g2), max(g1 + 1, g2)   # team2 win


# Backwards-compatible wrapper (kept in case anything calls it directly).
def simulate_match_hybrid(team1, team2, prob_cache, elo_dict=None, beta_home=None, beta_away=None):
    p = prob_cache[(team1, team2)]
    cdf0 = float(p[0]); cdf1 = float(p[0] + p[1])
    r = _random()
    if elo_dict is not None:
        neutral, home = resolve_venue(team1, team2)
        if neutral:
            lam1, lam2 = get_lambda(team1, team2, elo_dict, beta_home, beta_away, neutral=True)
        elif home == team1:
            lam1, lam2 = get_lambda(team1, team2, elo_dict, beta_home, beta_away, neutral=False)
        else:
            lam2, lam1 = get_lambda(team2, team1, elo_dict, beta_home, beta_away, neutral=False)
        g1 = _poisson(lam1); g2 = _poisson(lam2)
    else:
        g1, g2 = 1, 1
    if r < cdf0:
        return team1, max(g1, g2 + 1), min(g1, g2)
    elif r < cdf1:
        g = min(g1, g2); return "draw", g, g
    return team2, min(g1, g2), max(g1 + 1, g2)


def simulate_group(group_teams, mc):
    table = {team: {"points": 0, "gd": 0, "goals": 0} for team in group_teams}

    for team_a, team_b in combinations(group_teams, 2):
        result, g1, g2 = simulate_match_fast(team_a, team_b, mc, need_goals=True)

        table[team_a]["goals"] += g1
        table[team_b]["goals"] += g2
        table[team_a]["gd"] += g1 - g2
        table[team_b]["gd"] += g2 - g1

        if result == team_a:
            table[team_a]["points"] += 3
        elif result == team_b:
            table[team_b]["points"] += 3
        else:
            table[team_a]["points"] += 1
            table[team_b]["points"] += 1

    return sorted(
        table.items(),
        key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["goals"]),
        reverse=True
    )


def normalize_standings(sorted_items):
    return [
        {"team": team, "points": s["points"], "gd": s["gd"], "goals": s["goals"]}
        for team, s in sorted_items
    ]


def simulate_group_stage(groups, mc):
    group_results = {}
    third_place = []
    for group_name, teams in groups.items():
        standings = normalize_standings(simulate_group(teams, mc))
        group_results[group_name] = standings
        third_place.append({"group": group_name, "team": standings[2]["team"],
                            "stats": standings[2]})
    return group_results, third_place


def get_best_third_places(third_place):
    return sorted(
        third_place,
        key=lambda x: (x["stats"]["points"], x["stats"]["gd"], x["stats"]["goals"]),
        reverse=True
    )[:8]


def get_knockout_teams(group_results, best_thirds):
    teams = []
    for group, standings in group_results.items():
        teams.append(standings[0]["team"])
        teams.append(standings[1]["team"])
    for item in best_thirds:
        teams.append(item["team"])
    assert len(teams) == 32
    return teams


def seed_knockout(teams, elo_dict):
    return sorted(teams, key=lambda t: elo_dict.get(t, 1500), reverse=True)


def seeding_order(n):
    """Standard single-elimination seed order for a power-of-two field."""
    assert n & (n - 1) == 0, "n must be a power of two"
    order = [1, 2]
    while len(order) < n:
        m = len(order) * 2 + 1
        nxt = []
        for s in order:
            nxt.append(s)
            nxt.append(m - s)
        order = nxt
    return order


_SEED_ORDER_32 = seeding_order(32)


def bracket_seed_teams(teams, elo_dict):
    ranked = seed_knockout(teams, elo_dict)
    return [ranked[s - 1] for s in _SEED_ORDER_32]


def create_bracket(teams):
    return [(teams[i], teams[-i - 1]) for i in range(len(teams) // 2)]


def simulate_knockout_round(teams, mc):
    winners = []
    for i in range(0, len(teams), 2):
        t1, t2 = teams[i], teams[i + 1]
        result, _, _ = simulate_match_fast(t1, t2, mc, need_goals=False)
        if result == "draw":
            winners.append(t1 if _random() < 0.5 else t2)   # penalties
        else:
            winners.append(result)
    return winners


def simulate_tournament(groups, mc, elo_dict):
    group_results, third_place = simulate_group_stage(groups, mc)
    best_thirds = get_best_third_places(third_place)
    knockout_teams = get_knockout_teams(group_results, best_thirds)

    seeded = bracket_seed_teams(knockout_teams, elo_dict)

    r32 = simulate_knockout_round(seeded, mc)
    r16 = simulate_knockout_round(r32, mc)
    qf = simulate_knockout_round(r16, mc)
    sf = simulate_knockout_round(qf, mc)
    final = simulate_knockout_round(sf, mc)
    winner = final[0]

    bracket = {
        "R32": {"matches": [(seeded[i], seeded[i + 1]) for i in range(0, 32, 2)], "winners": r32},
        "R16": {"matches": [(r32[i], r32[i + 1]) for i in range(0, 16, 2)], "winners": r16},
        "QF": {"matches": [(r16[i], r16[i + 1]) for i in range(0, 8, 2)], "winners": qf},
        "SF": {"matches": [(qf[i], qf[i + 1]) for i in range(0, 4, 2)], "winners": sf},
        "Final": {"matches": [(sf[0], sf[1])], "winners": final},
    }

    stage_teams = {
        "R32": knockout_teams,   # reached R32
        "R16": r32,              # won R32
        "QF": r16,
        "SF": qf,
        "Final": sf,
        "Winner": final,
    }
    return winner, stage_teams, group_results, bracket


def run_simulations(n, groups, prob_cache, elo_dict, beta_home, beta_away):
    all_teams = [t for g in groups.values() for t in g]

    # Build the per-pair cache ONCE (this is the big win).
    mc = build_match_cache(all_teams, prob_cache, elo_dict, beta_home, beta_away)

    winners = []
    progression_totals = {
        t: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0, "Winner": 0}
        for t in all_teams
    }

    # Keep ONE sample tournament per distinct champion. After the loop we show
    # the one belonging to the most likely champion, so the displayed bracket
    # is internally consistent (the champion appears in its own Final) and
    # looks realistic (it ends with the favourite, not a random upset).
    sample_bracket = {}
    sample_groups = {}
    first_bracket = None
    first_group_results = None

    for i in range(n):
        winner, stage_teams, group_results, bracket = simulate_tournament(groups, mc, elo_dict)
        winners.append(winner)

        for stage, tlist in stage_teams.items():
            for t in tlist:
                progression_totals[t][stage] += 1

        if i == 0:
            first_bracket = bracket
            first_group_results = group_results
        if winner not in sample_bracket:
            sample_bracket[winner] = bracket
            sample_groups[winner] = group_results

    results = dict(Counter(winners))

    # Representative tournament = one that the MODAL champion actually won.
    if results:
        modal_champion = max(results, key=results.get)
        rep_bracket = sample_bracket.get(modal_champion, first_bracket)
        rep_groups = sample_groups.get(modal_champion, first_group_results)
    else:
        rep_bracket, rep_groups = first_bracket, first_group_results

    return results, progression_totals, rep_groups, rep_bracket


def normalize_progression(progression_totals, n):
    df = pd.DataFrame(progression_totals).T
    return df / n