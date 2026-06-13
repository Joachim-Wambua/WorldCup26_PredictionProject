import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

# Global storage for trained parameters
beta_home = None
beta_away = None

# Dixon-Coles low-score correction strength. Negative -> more draws (0-0, 1-1)
# and fewer 1-0 / 0-1, which matches real football better than independent
# Poisson. ~-0.10 to -0.15 is typical for international fixtures; call
# fit_rho() on your training data to get a data-driven value and pass it in.
DEFAULT_RHO = -0.13


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


def get_lambda(team_a, team_b, elo_dict, beta_home, beta_away, neutral=False):
    """
    Central function used by simulation.
    Converts Elo difference -> expected goals.

    The home-field effect is encoded in the GAP between the two fitted
    intercepts (home teams historically score more than away teams at the
    same Elo gap). For a NEUTRAL venue we replace both intercepts with their
    midpoint, so equal-Elo teams get equal expected goals and the only
    asymmetry left is genuine Elo strength.

    neutral=False (default) -> team_a is the home side and keeps the venue
                               advantage. This is the right setting for
                               TRAINING/EVALUATION on real home/away matches,
                               and for a host playing in its own country.
    neutral=True            -> no venue advantage to either side.
    """

    elo_a = elo_dict.get(team_a, 1700)
    elo_b = elo_dict.get(team_b, 1700)

    elo_diff = elo_a - elo_b

    if neutral:
        # Shared baseline = midpoint of the two fitted intercepts.
        baseline = (beta_home[0] + beta_away[0]) / 2.0
        lam_home = np.exp(baseline + beta_home[1] * elo_diff)
        lam_away = np.exp(baseline + beta_away[1] * (-elo_diff))
    else:
        lam_home = np.exp(beta_home[0] + beta_home[1] * elo_diff)
        lam_away = np.exp(beta_away[0] + beta_away[1] * (-elo_diff))

    return lam_home, lam_away


# ---------------------------------------------------------------------------
# DIXON-COLES SCORELINE MODEL
# ---------------------------------------------------------------------------
def _dc_tau(lam_home, lam_away, rho, max_goals):
    """Dixon-Coles correction matrix tau[i, j] over a goal grid.

    Only the four lowest-score cells are adjusted; everything else is 1.0.
    """
    tau = np.ones((max_goals + 1, max_goals + 1))
    if rho == 0.0:
        return tau
    tau[0, 0] = 1.0 - lam_home * lam_away * rho
    tau[0, 1] = 1.0 + lam_home * rho
    tau[1, 0] = 1.0 + lam_away * rho
    tau[1, 1] = 1.0 - rho
    return tau


def score_matrix(lam_home, lam_away, rho=0.0, max_goals=10):
    """Joint scoreline probability matrix M[i, j] = P(home=i, away=j).

    Independent Poisson when rho=0; with rho<0 it applies the Dixon-Coles
    low-score correction (more 0-0 / 1-1, fewer 1-0 / 0-1). Vectorised and
    renormalised so it sums to 1 over the truncated grid.
    """
    i = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(i, lam_home), poisson.pmf(i, lam_away))
    M = M * _dc_tau(lam_home, lam_away, rho, max_goals)
    M = np.clip(M, 0.0, None)
    total = M.sum()
    return M / total if total else M


def match_outcome_probs(lam_home, lam_away, max_goals=10, rho=0.0):
    """
    Convert expected goals into:
    [P(Home Win), P(Draw), P(Away Win)]

    Vectorised, with optional Dixon-Coles correction via rho.
    """
    M = score_matrix(lam_home, lam_away, rho=rho, max_goals=max_goals)
    p_home = np.tril(M, -1).sum()   # i > j
    p_draw = np.trace(M)            # i == j
    p_away = np.triu(M, 1).sum()    # i < j
    return np.array([p_home, p_draw, p_away])


def fit_rho(df, elo_dict, beta_home, beta_away, bounds=(-0.2, 0.2)):
    """Maximum-likelihood estimate of the Dixon-Coles rho on training data.

    Holds the Poisson lambdas (from get_lambda) fixed and finds the rho that
    best explains the observed low-score cells. Run this once after training
    and feed the result wherever you pass rho (e.g. the match predictor).
    """
    rows = df[["home_team", "away_team", "home_goals", "away_goals"]].to_numpy()

    def neg_log_lik(rho):
        ll = 0.0
        for ht, at, hg, ag in rows:
            lh, la = get_lambda(ht, at, elo_dict, beta_home, beta_away)
            hg, ag = int(hg), int(ag)
            base = poisson.pmf(hg, lh) * poisson.pmf(ag, la)
            tau = 1.0
            if hg == 0 and ag == 0:
                tau = 1.0 - lh * la * rho
            elif hg == 0 and ag == 1:
                tau = 1.0 + lh * rho
            elif hg == 1 and ag == 0:
                tau = 1.0 + la * rho
            elif hg == 1 and ag == 1:
                tau = 1.0 - rho
            ll += np.log(max(base * tau, 1e-12))
        return -ll

    res = minimize_scalar(neg_log_lik, bounds=bounds, method="bounded")
    return float(res.x)


def most_likely_scoreline(lam_home, lam_away, rho=0.0, max_goals=10, outcome=None):
    """Return ((home_goals, away_goals), probability).

    outcome=None    -> the single most likely exact score.
    outcome in {'home','draw','away'} -> most likely score of that type, so
        the predicted score can be made consistent with a favoured outcome.
    """
    M = score_matrix(lam_home, lam_away, rho=rho, max_goals=max_goals)
    n = M.shape[0]
    best, best_p = (1, 1), -1.0
    for a in range(n):
        for b in range(n):
            if outcome == "home" and not a > b:
                continue
            if outcome == "draw" and not a == b:
                continue
            if outcome == "away" and not a < b:
                continue
            if M[a, b] > best_p:
                best_p, best = M[a, b], (a, b)
    return best, float(best_p)


def predict_proba_df(df, elo_dict, beta_home, beta_away, rho=0.0):
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

        p = match_outcome_probs(lam_home, lam_away, rho=rho)
        probs.append(p)

    return np.array(probs)


def evaluate_poisson(df, y_true, elo_dict, beta_home, beta_away, rho=0.0):
    """
    Evaluate Poisson model using:
    - Log Loss
    - Brier Score
    """

    probs = predict_proba_df(df, elo_dict, beta_home, beta_away, rho=rho)

    # Log Loss
    ll = log_loss(y_true, probs)

    # Brier Score (multiclass)
    y_onehot = np.eye(3)[y_true]
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

    return ll, brier