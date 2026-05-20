import streamlit as st
import pandas as pd
import numpy as np
import os, sys
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from src.models.poisson import train_model

from simulation import (
    run_simulations,
    world_cup_2026_groups,
)

# Load Historical match data & Elo Scores
df = pd.read_csv("data/processed/matches_with_features.csv")
final_elo = pd.read_csv("data/processed/final_elo.csv")

# Model params
beta_home, beta_away = train_model(df)

# Create dict of elo scores
elo_dict = dict(zip(final_elo["team"], final_elo["elo"]))

st.set_page_config(page_title="World Cup Simulator", layout="wide")

st.title("⚽ World Cup 2026 Monte Carlo Simulator")

# ---------------------------
# Sidebar controls
# ---------------------------
n_sims = st.sidebar.slider("Simulations", 500, 10000, 2000, step=500)

run = st.sidebar.button("Run Simulation")

# ---------------------------
# Run simulation
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

    st.success("Simulation complete!")

    # ---------------------------
    # WINNER PROBABILITIES
    # ---------------------------
    df = pd.DataFrame.from_dict(results, orient="index", columns=["wins"])
    df["probability"] = df["wins"] / df["wins"].sum()
    df = df.sort_values("probability", ascending=False)

    top_10 = df.sort_values("probability", ascending=False).head(10)

    st.subheader("🏆 Winner Probabilities")

    st.bar_chart(top_10.set_index(top_10.index)["probability"])
    # st.bar_chart(top_10["probability"])

    st.dataframe(df.sort_values("probability", ascending=False).head(20))

    # ---------------------------
    # PROGRESSION HEATMAP DATA PREP
    # ---------------------------
    st.subheader("📊 Tournament Progression Overview")

    prog_df = pd.DataFrame(progression).T

    # normalize to probabilities
    prog_df = prog_df.div(prog_df.sum(axis=1), axis=0)

    st.dataframe(prog_df.sort_values("Winner", ascending=False).head(20))

    # ---------------------------
    # TEAM SELECTOR
    # ---------------------------
    st.subheader("🔎 Team Explorer")

    team = st.selectbox("Choose team", prog_df.index)

    st.write("Progression probabilities for:", team)

    st.bar_chart(prog_df.loc[team])