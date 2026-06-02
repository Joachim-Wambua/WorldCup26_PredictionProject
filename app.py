import streamlit as st
import pandas as pd
import numpy as np
import os, sys

# ---------------------------
# PATH SETUP
# ---------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.models.poisson import train_model, get_lambda
from src.models.hybrid import HybridWorldCupModel
from simulation import run_simulations, world_cup_2026_groups


# ---------------------------
# CACHING (IMPORTANT FOR STREAMLIT)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/matches_with_features.csv")
    final_elo = pd.read_csv("data/processed/final_elo.csv")
    return df, final_elo


@st.cache_resource
def load_hybrid_model():
    return HybridWorldCupModel.load("models/hybrid.pkl")


# ---------------------------
# LOAD DATA
# ---------------------------
df, final_elo = load_data()

elo_dict = dict(zip(final_elo["team"], final_elo["elo"]))

model = load_hybrid_model()

# Poisson params ONLY for simulation layer
# Cache poisson params
@st.cache_resource
def get_poisson_params(df):
    return train_model(df)

beta_home, beta_away = get_poisson_params(df)


# ---------------------------
# UI CONFIG
# ---------------------------
st.set_page_config(page_title="World Cup Simulator", layout="wide")

st.title("⚽ World Cup 2026 Simulator Dashboard")


# =========================================================
# SECTION 1: HYBRID MATCH PREDICTOR
# =========================================================
st.header("⚽ Hybrid Match Predictor")

teams = sorted(elo_dict.keys())

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Home Team", teams, key="t1")

with col2:
    team2 = st.selectbox("Away Team", teams, key="t2")

if model is None:
    st.error("Hybrid model not loaded")
    st.stop()

# --- Expected Goals (Poisson layer insight) ---
lam_home, lam_away = get_lambda(
    team1,
    team2,
    elo_dict,
    model.beta_home,
    model.beta_away
)

colA, colB = st.columns(2)

with colA:
    st.metric(team1, f"{lam_home:.2f} xG")

with colB:
    st.metric(team2, f"{lam_away:.2f} xG")


# --- Hybrid prediction ---
if st.button("Predict Match (Hybrid)"):
    probs = model.predict(team1, team2)

    # Map Labels to Team Names
    display_probs = {
        f"{team1} win": probs["home_win"],
        "Draw": probs["draw"],
        f"{team2} win": probs["away_win"]
    }

    st.subheader("Match Outcome Probabilities")

    prob_df = pd.DataFrame.from_dict(
        display_probs,
        orient="index",
        columns=["probability"]
    )

    prob_df = prob_df.sort_values("probability", ascending=False)

    st.bar_chart(prob_df)

    # Optional: cleaner JSON-style display
    st.write(display_probs)

    winner = max(display_probs, key=display_probs.get)

    st.success(f"Most likely outcome: {winner} ({display_probs[winner]:.1%})")

# =========================================================
# SECTION 2: TOURNAMENT SIMULATION
# =========================================================
st.header("🏆 Monte Carlo Tournament Simulation")

n_sims = st.sidebar.slider(
    "Simulations",
    500,
    10000,
    2000,
    step=500
)

run = st.sidebar.button("Run Simulation")


# ---------------------------
# RUN SIMULATION (COMPUTE ONLY)
# ---------------------------
if run:

    with st.spinner("Simulating tournaments..."):

        results, progression = run_simulations(
            n_sims,
            world_cup_2026_groups,
            elo_dict,
            beta_home,
            beta_away
        )

    # ✅ STORE EVERYTHING
    st.session_state["results"] = results
    st.session_state["progression"] = progression
    st.session_state["n_sims"] = n_sims

    st.success("Simulation complete!")


# ---------------------------
# DISPLAY RESULTS (ALWAYS IF AVAILABLE)
# ---------------------------
if "results" in st.session_state:

    results = st.session_state["results"]
    progression = st.session_state["progression"]
    n_sims = st.session_state["n_sims"]

    # ---------------------------
    # WINNER PROBABILITIES
    # ---------------------------
    results_df = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["wins"]
    )

    results_df["probability"] = results_df["wins"] / n_sims
    results_df = results_df.sort_values("probability", ascending=False)

    st.subheader("🏆 Winner Probabilities")

    st.bar_chart(results_df.head(10)["probability"])
    st.dataframe(results_df.head(20))

    # ---------------------------
    # PROGRESSION
    # ---------------------------
    st.subheader("📊 Tournament Progression")

    prog_df = pd.DataFrame(progression).T
    prog_df = prog_df.div(n_sims)

    st.dataframe(
        prog_df.sort_values("Winner", ascending=False).head(20)
    )

# =========================================================
# 🔎 SECTION 3: TEAM EXPLORER
# =========================================================
st.header("🔎 Team Explorer")

team = st.selectbox("Select Team", sorted(elo_dict.keys()), key="team_view")

prog_df = None

if "progression" in st.session_state:

    n_sims = st.session_state["n_sims"]

    prog_df = pd.DataFrame(st.session_state["progression"]).T
    prog_df = prog_df.div(n_sims)


if prog_df is not None and team in prog_df.index:

    st.write(f"### Progression probabilities: {team}")
    st.bar_chart(prog_df.loc[team])

else:
    st.info("Run a simulation to see team progression probabilities.")