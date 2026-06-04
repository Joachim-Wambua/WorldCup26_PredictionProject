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

    # ✅ LOAD FROM DISK (instant)
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)

    cache = {}

    for t1, t2 in combinations(teams, 2):

        probs = model.predict(t1, t2)

        p = np.array([
            probs.get("home_win", 0),
            probs.get("draw", 0),
            probs.get("away_win", 0)
        ], dtype=float)

        # ✅ CLEAN + NORMALIZE
        p = np.clip(p, 0, 1)

        if p.sum() == 0:
            p = np.array([0.33, 0.34, 0.33])
        else:
            p = p / p.sum()

        # ✅ STORE BOTH DIRECTIONS (CRITICAL)
        cache[(t1, t2)] = p
        cache[(t2, t1)] = np.array([p[2], p[1], p[0]])

    # ✅ SAVE TO DISK
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

    # goals (optional)
    if elo_dict is not None:
        lam1, lam2 = get_lambda(team1, team2, elo_dict, beta_home, beta_away)
        g1 = np.random.poisson(lam1)
        g2 = np.random.poisson(lam2)
    else:
        g1, g2 = 1, 1

    if outcome == "home_win":
        return team1, max(g1, g2+1), min(g1, g2)
    elif outcome == "away_win":
        return team2, min(g1, g2), max(g1+1, g2)
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

# SIMULATE ALL GROUPS
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

    # top 2 from each group
    for group, standings in group_results.items():
        teams.append(standings[0]["team"])
        teams.append(standings[1]["team"])

    # add best 3rd
    for item in best_thirds:
        teams.append(item["team"])

    assert len(teams) == 32

    return teams

def seed_knockout(teams, elo_dict):
    return sorted(teams, key=lambda t: elo_dict.get(t, 1500), reverse=True)


def create_bracket(teams):
    return [(teams[i], teams[-i-1]) for i in range(len(teams)//2)]

def simulate_knockout_round(teams, prob_cache, elo_dict, beta_home, beta_away):

    matches = [(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]
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
        team: {"R32":0,"R16":0,"QF":0,"SF":0,"Final":0,"Winner":0}
        for group in groups.values() 
        for team in group
    }

    group_results, third_place = simulate_group_stage(
        groups, prob_cache, elo_dict, beta_home, beta_away
    )

    best_thirds = get_best_third_places(third_place)
    knockout_teams = get_knockout_teams(group_results, best_thirds)

    for t in knockout_teams:
        progression[t]["R32"] = 1

    seeded = seed_knockout(knockout_teams, elo_dict)

    r32 = simulate_knockout_round(seeded, prob_cache, elo_dict, beta_home, beta_away)
    for t in r32:
        progression[t]["R16"] += 1

    r16 = simulate_knockout_round(r32, prob_cache, elo_dict, beta_home, beta_away)
    for t in r16:
        progression[t]["QF"] += 1

    qf = simulate_knockout_round(r16, prob_cache, elo_dict, beta_home, beta_away)
    for t in qf:
        progression[t]["SF"] += 1

    sf = simulate_knockout_round(qf, prob_cache, elo_dict, beta_home, beta_away)
    for t in sf:
        progression[t]["Final"] += 1

    final = simulate_knockout_round(sf, prob_cache, elo_dict, beta_home, beta_away)
    for t in final:
        progression[t]["Winner"] += 1

    return final[0], progression, group_results

def run_simulations(n, groups, prob_cache, elo_dict, beta_home, beta_away):

    winners = []
    progression_totals = {
        team: {"R32":0,"R16":0,"QF":0,"SF":0,"Final":0,"Winner":0}
        for team in [t for g in groups.values() for t in g]
    }

    for _ in range(n):

        winner, progression, group_results = simulate_tournament(
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
        
        if _ == 0:
            first_group_results = group_results

    return dict(Counter(winners)), progression_totals, first_group_results

# Helper function to normalize progression totals into probabilities
def normalize_progression(progression_totals, n):
    df = pd.DataFrame(progression_totals).T
    return df / n

