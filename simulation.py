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


def simulate_match_hybrid(team1, team2, prob_cache, elo_dict=None, beta_home=None, beta_away=None):

    p = prob_cache[(team1, team2)]

    outcome = np.random.choice(
        ["home_win", "draw", "away_win"],
        p=p
    )

    if elo_dict is not None:
        neutral, home = resolve_venue(team1, team2)
        if neutral:
            lam1, lam2 = get_lambda(team1, team2, elo_dict, beta_home, beta_away, neutral=True)
        elif home == team1:
            lam1, lam2 = get_lambda(team1, team2, elo_dict, beta_home, beta_away, neutral=False)
        else:
            # host is team2 -> give the venue boost to team2's rate
            lam2, lam1 = get_lambda(team2, team1, elo_dict, beta_home, beta_away, neutral=False)
        g1 = np.random.poisson(lam1)
        g2 = np.random.poisson(lam2)
    else:
        g1, g2 = 1, 1

    if outcome == "home_win":
        return team1, max(g1, g2 + 1), min(g1, g2)
    elif outcome == "away_win":
        return team2, min(g1, g2), max(g1 + 1, g2)
    else:
        g = min(g1, g2)
        return "draw", g, g


def simulate_group(group_teams, prob_cache, elo_dict, beta_home, beta_away):

    table = {
        team: {"points": 0, "gd": 0, "goals": 0}
        for team in group_teams
    }

    for team_a, team_b in combinations(group_teams, 2):

        result, g1, g2 = simulate_match_hybrid(
            team_a, team_b,
            prob_cache,
            elo_dict,
            beta_home,
            beta_away
        )

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
        {
            "team": team,
            "points": stats["points"],
            "gd": stats["gd"],
            "goals": stats["goals"]
        }
        for team, stats in sorted_items
    ]


def simulate_group_stage(groups, prob_cache, elo_dict, beta_home, beta_away):

    group_results = {}
    third_place = []

    for group_name, teams in groups.items():

        standings_raw = simulate_group(
            teams,
            prob_cache,
            elo_dict,
            beta_home,
            beta_away
        )

        standings = normalize_standings(standings_raw)
        group_results[group_name] = standings

        third_place.append({
            "group": group_name,
            "team": standings[2]["team"],
            "stats": standings[2]
        })

    return group_results, third_place


def get_best_third_places(third_place):

    ranked = sorted(
        third_place,
        key=lambda x: (
            x["stats"]["points"],
            x["stats"]["gd"],
            x["stats"]["goals"]
        ),
        reverse=True
    )

    return ranked[:8]


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
    """Standard single-elimination seed order for a power-of-two field.

    Returns 1-indexed seeds arranged so that adjacent pairs are
    (1 vs n), (2 vs n-1), ... and stronger seeds are kept maximally
    apart: #1 and #2 can only meet in the final, the top 4 land in
    four different quarters, and so on.
    """
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


def bracket_seed_teams(teams, elo_dict):
    """Order the 32 qualifiers into a properly seeded bracket.

    Teams are ranked by Elo (1 = strongest), then placed into the standard
    bracket positions so the field is spread instead of pairing #1 vs #2.
    """
    ranked = seed_knockout(teams, elo_dict)          # index 0 = strongest
    order = seeding_order(len(ranked))               # 1-indexed seeds in bracket order
    return [ranked[s - 1] for s in order]


def create_bracket(teams):
    return [(teams[i], teams[-i - 1]) for i in range(len(teams) // 2)]


def simulate_knockout_round(teams, prob_cache, elo_dict, beta_home, beta_away):

    matches = [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]
    winners = []

    for t1, t2 in matches:
        result, _, _ = simulate_match_hybrid(
            t1, t2,
            prob_cache,
            elo_dict,
            beta_home,
            beta_away
        )

        if result == "draw":
            winners.append(np.random.choice([t1, t2]))  # penalties
        else:
            winners.append(result)

    return winners


def simulate_tournament(groups, prob_cache, elo_dict, beta_home, beta_away):

    progression = {
        team: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0, "Winner": 0}
        for team in [t for g in groups.values() for t in g]
    }

    # ---------------------------
    # GROUP STAGE
    # ---------------------------
    group_results, third_place = simulate_group_stage(
        groups, prob_cache, elo_dict, beta_home, beta_away
    )

    best_thirds = get_best_third_places(third_place)
    knockout_teams = get_knockout_teams(group_results, best_thirds)

    for t in knockout_teams:
        progression[t]["R32"] = 1

    # Proper bracket seeding: spread the field (#1 vs #32, ...) instead of
    # pairing the two strongest teams together in the Round of 32.
    seeded = bracket_seed_teams(knockout_teams, elo_dict)

    bracket = {}

    # ---------------------------
    # R32
    # ---------------------------
    r32 = simulate_knockout_round(seeded, prob_cache, elo_dict, beta_home, beta_away)
    r32_matches = [(seeded[i], seeded[i + 1]) for i in range(0, 32, 2)]
    bracket["R32"] = {"matches": r32_matches, "winners": r32}

    for t in r32:
        progression[t]["R16"] += 1

    # ---------------------------
    # R16
    # ---------------------------
    r16 = simulate_knockout_round(r32, prob_cache, elo_dict, beta_home, beta_away)
    r16_matches = [(r32[i], r32[i + 1]) for i in range(0, 16, 2)]
    bracket["R16"] = {"matches": r16_matches, "winners": r16}

    for t in r16:
        progression[t]["QF"] += 1

    # ---------------------------
    # QF
    # ---------------------------
    qf = simulate_knockout_round(r16, prob_cache, elo_dict, beta_home, beta_away)
    qf_matches = [(r16[i], r16[i + 1]) for i in range(0, 8, 2)]
    bracket["QF"] = {"matches": qf_matches, "winners": qf}

    for t in qf:
        progression[t]["SF"] += 1

    # ---------------------------
    # SF
    # ---------------------------
    sf = simulate_knockout_round(qf, prob_cache, elo_dict, beta_home, beta_away)
    sf_matches = [(qf[i], qf[i + 1]) for i in range(0, 4, 2)]
    bracket["SF"] = {"matches": sf_matches, "winners": sf}

    for t in sf:
        progression[t]["Final"] += 1

    # ---------------------------
    # FINAL
    # ---------------------------
    final = simulate_knockout_round(sf, prob_cache, elo_dict, beta_home, beta_away)
    final_match = [(sf[0], sf[1])]
    bracket["Final"] = {"matches": final_match, "winners": final}

    winner = final[0]

    for t in final:
        progression[t]["Winner"] += 1

    return winner, progression, group_results, bracket


def run_simulations(n, groups, prob_cache, elo_dict, beta_home, beta_away):

    winners = []

    progression_totals = {
        team: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0, "Winner": 0}
        for team in [t for g in groups.values() for t in g]
    }

    first_bracket = None
    first_group_results = None

    for i in range(n):

        winner, progression, group_results, bracket = simulate_tournament(
            groups,
            prob_cache,
            elo_dict,
            beta_home,
            beta_away
        )

        winners.append(winner)

        for team in progression:
            for stage in progression[team]:
                progression_totals[team][stage] += progression[team][stage]

        if i == 0:
            first_bracket = bracket
            first_group_results = group_results

    return dict(Counter(winners)), progression_totals, first_group_results, first_bracket


def normalize_progression(progression_totals, n):
    df = pd.DataFrame(progression_totals).T
    return df / n
