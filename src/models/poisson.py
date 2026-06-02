import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss
from scipy.stats import poisson

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

def match_outcome_probs(lam_home, lam_away, max_goals=10):
    """
    Convert expected goals into:
    [P(Home Win), P(Draw), P(Away Win)]
    """

    p_home, p_draw, p_away = 0, 0, 0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):

            p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)

            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    return np.array([p_home, p_draw, p_away])

def predict_proba_df(df, elo_dict, beta_home, beta_away):
    """
    Generate probability predictions for a dataframe.
    Requires:
    - HOME_team
    - away_team
    """

    probs = []

    for _, row in df.iterrows():

        team_home = row["home_team"]
        team_away = row["away_team"]

        lam_home, lam_away = get_lambda(
            team_home, team_away, elo_dict, beta_home, beta_away
        )

        p = match_outcome_probs(lam_home, lam_away)
        probs.append(p)

    return np.array(probs)

def evaluate_poisson(df, y_true, elo_dict, beta_home, beta_away):
    """
    Evaluate Poisson model using:
    - Log Loss
    - Brier Score
    """

    probs = predict_proba_df(df, elo_dict, beta_home, beta_away)

    # Log Loss
    ll = log_loss(y_true, probs)

    # Brier Score (multiclass)
    y_onehot = np.eye(3)[y_true]
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

    return ll, brier