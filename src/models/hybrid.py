import pandas as pd
import numpy as np

from src.models.poisson import train_model, get_lambda
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score


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
    def __init__(self):
        self.beta_home = None
        self.beta_away = None
        self.xgb_model = None
        self.elo_dict = None
        self.features = None

    # -------------------------
    # TRAINING PIPELINE
    # -------------------------
    def fit(self, train_df, elo_dict, test_df=None):
        """
        Train hybrid model on already-split data
        """

        self.elo_dict = elo_dict

        # -------------------------
        # 1. Train Poisson (train only)
        # -------------------------
        self.beta_home, self.beta_away = train_model(train_df)

        # -------------------------
        # 2. Feature engineering
        # -------------------------
        train_df, self.features = build_features(
            train_df,
            elo_dict,
            self.beta_home,
            self.beta_away
        )

        if test_df is not None:
            test_df, _ = build_features(
                test_df,
                elo_dict,
                self.beta_home,
                self.beta_away
            )

        # -------------------------
        # 3. Train XGB
        # -------------------------
        self.xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="mlogloss"
        )

        self.xgb_model.fit(
            train_df[self.features],
            train_df["result"]
        )

        # -------------------------
        # 4. Optional evaluation
        # -------------------------
        if test_df is not None:
            preds = self.xgb_model.predict_proba(test_df[self.features])

            print("📊 TEST LOG LOSS:",
                log_loss(test_df["result"], preds))

            print("📊 TEST ACC:",
                accuracy_score(test_df["result"], np.argmax(preds, axis=1)))

        print("✅ Model trained")


    # -------------------------
    # EVALUATION FUNCTION (external use)
    # -------------------------
    def evaluate(self, test_df):

        test_df, _ = build_features(
            test_df,
            self.elo_dict,
            self.beta_home,
            self.beta_away
        )

        X_test = test_df[self.features]
        y_test = test_df["result"]

        probs = self.xgb_model.predict_proba(X_test)

        ll = log_loss(y_test, probs)
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
    def predict(self, team1, team2):

        elo_home = self.elo_dict.get(team1, 1500)
        elo_away = self.elo_dict.get(team2, 1500)
        elo_diff = elo_home - elo_away

        lam_home, lam_away = get_lambda(
            team1, team2,
            self.elo_dict,
            self.beta_home,
            self.beta_away
        )

        X = pd.DataFrame([{
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_diff,
            "lambda_home": lam_home,
            "lambda_away": lam_away
        }])

        probs = self.xgb_model.predict_proba(X)[0]

        return {
            "away_win": float(probs[1]),
            "draw": float(probs[0]),
            "home_win": float(probs[2])
        }

    # -------------------------
    # SIMULATION
    # -------------------------
    def simulate_match(self, team1, team2):

        probs = self.predict(team1, team2)

        outcome = np.random.choice(
            ["away", "draw", "home"],
            p=[probs["away_win"], probs["draw"], probs["home_win"]]
        )

        if outcome == "home":
            return team1
        elif outcome == "away":
            return team2
        return np.random.choice([team1, team2])

    # -------------------------
    # SAVE / LOAD
    # -------------------------
    def save(self, path="models/hybrid.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="models/hybrid.pkl"):
        return joblib.load(path)