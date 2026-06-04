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
from simulation import run_simulations, world_cup_2026_groups, build_match_prob_cache

PRIMARY = "#0B1F3A"     # deep navy
ACCENT = "#E10600"      # FIFA red
GOLD = "#F5C518"        # trophy gold
LIGHT = "#F4F6F8"
CARD = "#111827"

st.markdown(f"""
    <style>
        .hero {{
            background: linear-gradient(90deg, {PRIMARY}, #1C3D73);
            padding: 20px;
            border-radius: 12px;
            color: white;
        }}
        .hero h1 {{
            font-size: 42px;
            margin-bottom: 0;
        }}
        .hero p {{
            margin-top: 5px;
            opacity: 0.85;
        }}
    </style>

    <div class="hero">
        <h1>FIFA World Cup 2026 Simulator</h1>
        <p>AI-powered hybrid engine • Monte Carlo simulations • Live tournament projection</p>
    </div>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_prob_cache(_teams):
    model = HybridWorldCupModel.load("models/hybrid.pkl")
    return build_match_prob_cache(_teams, model, save_path="models/prob_cache.pkl")

# ---------------------------
# LOAD DATA + ELO RANKINGS + TEAM INFO
# ---------------------------
df, final_elo = load_data()

elo_dict = dict(zip(final_elo["team"], final_elo["elo"]))
teams = sorted({
    team
    for group in world_cup_2026_groups.values()
    for team in group
})

model = load_hybrid_model()
prob_cache = load_prob_cache(tuple(teams))


# Poisson params ONLY for simulation layer
# Cache poisson params
@st.cache_resource
def get_poisson_params(df):
    return train_model(df)

beta_home, beta_away = get_poisson_params(df)

# SHOW WORLD CUP 26 GROUP STAGES
def render_group_table(group_name, standings):
    df = pd.DataFrame(standings).sort_values(
        ["points", "gd", "goals"],
        ascending=False
    )

    st.markdown(f"### Group {group_name}")

    styled = df.style \
        .background_gradient(subset=["points"], cmap="Blues") \
        .format(precision=0)

    st.dataframe(styled, use_container_width=True)

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
with st.sidebar:
    st.header("⚙️ Simulation Controls")

    n_sims = st.slider("Number of Simulations", 100, 10000, 1000, step=100)

    show_live = st.toggle("🎥 Live Simulation Mode", value=False)

    run = st.button("🚀 Run Simulation")

# ---------------------------
# RUN SIMULATION (COMPUTE ONLY)
# ---------------------------
if run:
    with st.spinner("Simulating World Cup 2026..."):

        results, progression, group_results = run_simulations(
            n_sims,
            world_cup_2026_groups,
            prob_cache,
            elo_dict,
            beta_home,
            beta_away
        )

    # ✅ STORE EVERYTHING
    st.session_state["results"] = results
    st.session_state["progression"] = progression
    st.session_state["n_sims"] = n_sims
    st.session_state["group_results"] = group_results

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


# ---------------------------
# DISPLAY RESULTS (ALWAYS IF AVAILABLE)
# ---------------------------
if "group_results" in st.session_state:

    st.subheader("📊 Group Stage Standings")

    group_results = st.session_state["group_results"]

    cols = st.columns(4)

    for i, (group, standings) in enumerate(group_results.items()):
        with cols[i % 4]:
            render_group_table(group, standings)

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