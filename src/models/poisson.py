import numpy as np
import pandas as pd
import statsmodels.api as sm

# Global storage for trained parameters
beta_home = None
beta_away = None


def train_model(df):
    """
    Trains Poisson regression model ONCE.
    """

    global beta_home, beta_away

    df = df.copy()
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    # HOME model
    X_home = sm.add_constant(df["elo_diff"])
    y_home = df["home_goals"]

    model_home = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()

    # AWAY model
    X_away = sm.add_constant(-df["elo_diff"])
    y_away = df["away_goals"]

    model_away = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()

    beta_home = model_home.params.values
    beta_away = model_away.params.values

    return beta_home, beta_away


def get_lambda(team_a, team_b, elo_dict, beta_home, beta_away):
    """
    Central function used by simulation.
    Converts Elo difference → expected goals.
    """

    elo_a = elo_dict.get(team_a, 1700)
    elo_b = elo_dict.get(team_b, 1700)

    elo_diff = elo_a - elo_b

    lam_home = np.exp(beta_home[0] + beta_home[1] * elo_diff)
    lam_away = np.exp(beta_away[0] + beta_away[1] * (-elo_diff))

    return lam_home, lam_away