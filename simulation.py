import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

import sys
import os


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

def simulate_match(team1, team2, elo_dict, beta_home, beta_away):
    """
    Simulate a single match using Poisson goal model.
    """

    lam_a, lam_b = get_lambda(
        team1, team2,
        elo_dict,
        beta_home,
        beta_away
    )

    goals_a = np.random.poisson(lam_a)
    goals_b = np.random.poisson(lam_b)

    if goals_a > goals_b:
        return team1
    elif goals_b > goals_a:
        return team2
    else:
        return np.random.choice([team1, team2])


def simulate_group(group_teams, elo_dict, beta_home, beta_away):

    table = {
        team: {"points": 0, "gd": 0, "goals": 0}
        for team in group_teams
    }

    for team_a, team_b in combinations(group_teams, 2):

        lam_a, lam_b = get_lambda(
            team_a, team_b,
            elo_dict,
            beta_home,
            beta_away
        )

        goals_a = np.random.poisson(lam_a)
        goals_b = np.random.poisson(lam_b)

        table[team_a]["goals"] += goals_a
        table[team_b]["goals"] += goals_b

        table[team_a]["gd"] += goals_a - goals_b
        table[team_b]["gd"] += goals_b - goals_a

        if goals_a > goals_b:
            table[team_a]["points"] += 3
        elif goals_b > goals_a:
            table[team_b]["points"] += 3
        else:
            table[team_a]["points"] += 1
            table[team_b]["points"] += 1

    return sorted(
        table.items(),
        key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["goals"]),
        reverse=True
    )


# SIMULATE ALL GROUPS
def simulate_group_stage(groups, elo_dict, beta_home, beta_away):

    group_results = {}
    third_place = []

    for group_name, teams in groups.items():

        standings = simulate_group(
            teams, elo_dict, beta_home, beta_away
        )

        group_results[group_name] = standings

        # store 3rd place
        third_place.append((group_name, standings[2]))

    return group_results, third_place

def get_best_third_places(third_place):

    ranked = sorted(
        third_place,
        key=lambda x: (
            x[1][1]["points"],
            x[1][1]["gd"],
            x[1][1]["goals"]
        ),
        reverse=True
    )

    return ranked[:8]

def get_knockout_teams(group_results, best_thirds):

    teams = []

    # top 2 from each group
    for group, standings in group_results.items():
        teams.append(standings[0][0])
        teams.append(standings[1][0])

    # add best 3rd
    for group_name, (team, stats) in best_thirds:
        teams.append(team)

    assert len(teams) == 32

    return teams

def seed_knockout(teams, elo_dict):
    return sorted(teams, key=lambda t: elo_dict.get(t, 1500), reverse=True)


def create_bracket(teams):
    return [(teams[i], teams[-i-1]) for i in range(len(teams)//2)]

def simulate_knockout_round(teams, elo_dict, beta_home, beta_away):

    matches = [(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]

    winners = []

    for t1, t2 in matches:
        winners.append(simulate_match(t1, t2, elo_dict, beta_home, beta_away))

    return winners

def simulate_tournament(groups, elo_dict, beta_home, beta_away):

    progression = {team: {"R32":0,"R16":0,"QF":0,"SF":0,"Final":0,"Winner":0}
                   for group in groups.values() for team in group}

    # --- GROUP STAGE ---
    group_results, third_place = simulate_group_stage(
        groups, elo_dict, beta_home, beta_away
    )

    # --- BEST 3rd ---
    best_thirds = get_best_third_places(third_place)

    # --- 32 TEAMS ---
    knockout_teams = get_knockout_teams(group_results, best_thirds)

    for t in knockout_teams:
        progression[t]["R32"] = 1

    # --- STRUCTURED BRACKET ---
    seeded = seed_knockout(knockout_teams, elo_dict)

    # --- KNOCKOUTS ---
    r32 = simulate_knockout_round(seeded, elo_dict, beta_home, beta_away)
    for team in r32:
        progression[team]["R16"] += 1

    r16 = simulate_knockout_round(r32, elo_dict, beta_home, beta_away)
    for team in r16:
        progression[team]["QF"] += 1

    qf = simulate_knockout_round(r16, elo_dict, beta_home, beta_away)
    for team in qf:
        progression[team]["SF"] += 1
    sf = simulate_knockout_round(qf, elo_dict, beta_home, beta_away)
    for team in sf:
        progression[team]["Final"] += 1
    final = simulate_knockout_round(sf, elo_dict, beta_home, beta_away)
    for team in final:
        progression[team]["Winner"] += 1

    # Return winner
    return final[0], progression

def run_simulations(n, groups, elo_dict, beta_home, beta_away):
    winners = []
    progression_totals = {}

    for team in [t for g in groups.values() for t in g]:
        progression_totals[team] = {
            "R32":0,"R16":0,"QF":0,"SF":0,"Final":0,"Winner":0
        }

    for _ in range(n):

        winner, progression = simulate_tournament(
            groups,
            elo_dict,
            beta_home,
            beta_away
        )

        winners.append(winner)

        for team in progression:
            for stage in progression[team]:
                progression_totals[team][stage] += progression[team][stage]

    return dict(Counter(winners)), progression_totals

# Helper function to normalize progression totals into probabilities
def normalize_progression(progression_totals, n):
    df = pd.DataFrame(progression_totals).T
    return df / n

