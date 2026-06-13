import pandas as pd
import numpy as np

from src.models.poisson import (
    train_model,
    get_lambda,
    match_outcome_probs,
    score_matrix,
    DEFAULT_RHO,
)
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from scipy.optimize import minimize_scalar


# -----------------------------
# FEATURE ENGINEERING (single source of truth)
# -----------------------------
def build_features(df, elo_dict, beta_home, beta_away):
    df = df.copy()

    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    def get_lambdas(row):
        lam_home, lam_away = get_lambda(
            row["home_team"],
            row["away_team"],
            elo_dict,
            beta_home,
            beta_away
        )
        return pd.Series([lam_home, lam_away])

    df[["lambda_home", "lambda_away"]] = df.apply(get_lambdas, axis=1)

    df["result"] = df.apply(
        lambda r: 2 if r["home_goals"] > r["away_goals"]
        else 1 if r["home_goals"] < r["away_goals"]
        else 0,
        axis=1
    )

    features = [
        "elo_home",
        "elo_away",
        "elo_diff",
        "lambda_home",
        "lambda_away"
    ]

    return df, features


# -----------------------------
# HYBRID MODEL
# -----------------------------
class HybridWorldCupModel:
    def __init__(self, blend_weight=1.0, rho=DEFAULT_RHO):
        self.beta_home = None
        self.beta_away = None
        self.xgb_model = None
        self.elo_dict = None
        self.features = None

        # --- new knobs (all have safe getattr fallbacks in predict) ---
        # temperature for probability calibration (1.0 = no scaling)
        self.temperature = 1.0
        # blend weight on the XGBoost head vs the Dixon-Coles Poisson head.
        # 1.0 = pure XGBoost (original behaviour). 0.7-0.85 mixes in the
        # Poisson scoreline model, which calibrates draws better.
        self.blend_weight = blend_weight
        # Dixon-Coles low-score correction used by the Poisson head.
        self.rho = rho

    # -------------------------
    # TRAINING PIPELINE
    # -------------------------
    def fit(self, train_df, elo_dict, test_df=None, calibrate=True):
        """
        Train hybrid model on already-split data.

        calibrate=True fits a temperature on a held-out slice of the TRAIN
        set so the simulator samples from well-calibrated probabilities
        instead of XGBoost's typically over-confident raw outputs.
        """

        self.elo_dict = elo_dict

        # 1. Poisson (train only)
        self.beta_home, self.beta_away = train_model(train_df)

        # 2. Feature engineering
        train_df, self.features = build_features(
            train_df, elo_dict, self.beta_home, self.beta_away
        )
        if test_df is not None:
            test_df, _ = build_features(
                test_df, elo_dict, self.beta_home, self.beta_away
            )

        X = train_df[self.features]
        y = train_df["result"]

        # 3. XGB — regularised + shallower. The signal is essentially 1-D
        #    (elo_diff), so a deep, unregularised forest just memorises noise.
        self.xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            gamma=0.5,
            eval_metric="mlogloss",
        )

        # 3b. Calibration split carved from TRAIN (keeps test_df clean for
        #     honest reporting). Train on the fit slice, learn temperature on
        #     the held-out slice, then refit on all of train and reuse T.
        if calibrate and len(X) > 50:
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            self.xgb_model.fit(X_fit, y_fit)
            self.temperature = self._fit_temperature(X_cal, y_cal)
            self.xgb_model.fit(X, y)   # refit on everything, keep T
        else:
            self.xgb_model.fit(X, y)
            self.temperature = 1.0

        # 4. Optional evaluation (reflects the calibrated probabilities)
        if test_df is not None:
            preds = self._calibrated_proba(test_df[self.features])
            print("📊 TEST LOG LOSS:", log_loss(test_df["result"], preds, labels=[0, 1, 2]))
            print("📊 TEST ACC:", accuracy_score(test_df["result"], np.argmax(preds, axis=1)))

        print(f"✅ Model trained (temperature={self.temperature:.2f}, "
              f"blend_weight={self.blend_weight:.2f})")

    # -------------------------
    # CALIBRATION HELPERS
    # -------------------------
    def _fit_temperature(self, X_val, y_val):
        """Temperature scaling: one parameter T minimising validation log loss.
        T>1 softens over-confident probabilities; T<1 sharpens."""
        raw = self.xgb_model.predict_proba(X_val)
        logp = np.log(np.clip(raw, 1e-9, 1.0))

        def nll(T):
            s = logp / T
            s = s - s.max(axis=1, keepdims=True)
            e = np.exp(s)
            p = e / e.sum(axis=1, keepdims=True)
            return log_loss(y_val, p, labels=[0, 1, 2])

        res = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
        return float(res.x)

    def _apply_temperature(self, probs):
        """Apply temperature to an [N, 3] probability array."""
        T = getattr(self, "temperature", 1.0) or 1.0
        if T == 1.0:
            return probs
        logp = np.log(np.clip(probs, 1e-9, 1.0))
        s = logp / T
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def _calibrated_proba(self, X):
        return self._apply_temperature(self.xgb_model.predict_proba(X))

    # -------------------------
    # EVALUATION
    # -------------------------
    def evaluate(self, test_df):
        test_df, _ = build_features(
            test_df, self.elo_dict, self.beta_home, self.beta_away
        )
        X_test = test_df[self.features]
        y_test = test_df["result"]

        probs = self._calibrated_proba(X_test)

        ll = log_loss(y_test, probs, labels=[0, 1, 2])
        acc = accuracy_score(y_test, np.argmax(probs, axis=1))
        brier = np.mean(np.sum((probs - np.eye(3)[y_test]) ** 2, axis=1))

        print("\n📊 FINAL EVALUATION")
        print("Log Loss:", ll)
        print("Accuracy:", acc)
        print("Brier:", brier)

        return {"log_loss": ll, "accuracy": acc, "brier": brier}

    # -------------------------
    # PREDICTION PIPELINE
    # -------------------------
    def _predict_oriented(self, home_team, away_team):
        """Calibrated XGBoost prediction with home_team in the home slot."""
        elo_home = self.elo_dict.get(home_team, 1500)
        elo_away = self.elo_dict.get(away_team, 1500)
        elo_diff = elo_home - elo_away

        lam_home, lam_away = get_lambda(
            home_team, away_team, self.elo_dict, self.beta_home, self.beta_away
        )

        X = pd.DataFrame([{
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_diff,
            "lambda_home": lam_home,
            "lambda_away": lam_away,
        }])

        probs = self._calibrated_proba(X)[0]   # [draw, away, home]
        return {
            "draw": float(probs[0]),
            "away_win": float(probs[1]),
            "home_win": float(probs[2]),
        }

    def _dc_probs(self, team1, team2, neutral):
        """Dixon-Coles Poisson W/D/L head, oriented to team1 (home slot)."""
        rho = getattr(self, "rho", DEFAULT_RHO)
        lam1, lam2 = get_lambda(
            team1, team2, self.elo_dict, self.beta_home, self.beta_away, neutral=neutral
        )
        p = match_outcome_probs(lam1, lam2, rho=rho)   # [home, draw, away]
        return {"home_win": float(p[0]), "draw": float(p[1]), "away_win": float(p[2])}

    def predict(self, team1, team2, neutral=True):
        """Match outcome probabilities for team1 (home slot) vs team2.

        neutral=True  -> symmetrised (cancels the home-slot bias).
        neutral=False -> team1 is the home side (a host at home).

        If blend_weight < 1, the calibrated XGBoost head is blended with the
        Dixon-Coles Poisson head, which models draws and scorelines better.
        """
        if not neutral:
            xgb_p = self._predict_oriented(team1, team2)
        else:
            ab = self._predict_oriented(team1, team2)   # team1 as home
            ba = self._predict_oriented(team2, team1)   # team2 as home
            hw = (ab["home_win"] + ba["away_win"]) / 2.0
            aw = (ab["away_win"] + ba["home_win"]) / 2.0
            dr = (ab["draw"] + ba["draw"]) / 2.0
            t = hw + aw + dr or 1.0
            xgb_p = {"home_win": hw / t, "draw": dr / t, "away_win": aw / t}

        w = getattr(self, "blend_weight", 1.0)
        if w >= 1.0:
            return xgb_p

        dc_p = self._dc_probs(team1, team2, neutral)
        out = {k: w * xgb_p[k] + (1.0 - w) * dc_p[k] for k in xgb_p}
        s = sum(out.values()) or 1.0
        return {k: v / s for k, v in out.items()}

    # -------------------------
    # SIMULATION
    # -------------------------
    def simulate_match(self, team1, team2, neutral=True):
        probs = self.predict(team1, team2, neutral=neutral)
        outcome = np.random.choice(
            ["away", "draw", "home"],
            p=[probs["away_win"], probs["draw"], probs["home_win"]]
        )
        if outcome == "home":
            return team1
        elif outcome == "away":
            return team2
        return np.random.choice([team1, team2])

    def sample_scoreline(self, team1, team2, neutral=True, max_goals=10):
        """Draw a full scoreline (g1, g2) from the Dixon-Coles joint matrix.

        Lets the simulator sample OUTCOME and GOALS together and consistently
        (instead of an XGBoost outcome plus independently-sampled Poisson
        goals that can disagree). Wire this into simulation.py if you want
        group goal-difference / goals-for to match the win probabilities.
        """
        rho = getattr(self, "rho", DEFAULT_RHO)
        lam1, lam2 = get_lambda(
            team1, team2, self.elo_dict, self.beta_home, self.beta_away, neutral=neutral
        )
        M = score_matrix(lam1, lam2, rho=rho, max_goals=max_goals)
        flat = M.flatten()
        flat = flat / flat.sum()
        idx = np.random.choice(len(flat), p=flat)
        g1, g2 = divmod(int(idx), M.shape[1])
        return int(g1), int(g2)

    # -------------------------
    # SAVE / LOAD
    # -------------------------
    def save(self, path="models/hybrid.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="models/hybrid.pkl"):
        return joblib.load(path)