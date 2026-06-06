import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import textwrap

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

FLAG_URLS = {
    "Mexico": "https://flagcdn.com/w320/mx.png",
    "South Africa": "https://flagcdn.com/w320/za.png",
    "South Korea": "https://flagcdn.com/w320/kr.png",
    "Czechia": "https://flagcdn.com/w320/cz.png",

    "Canada": "https://flagcdn.com/w320/ca.png",
    "Bosnia and Herzegovina": "https://flagcdn.com/w320/ba.png",
    "Qatar": "https://flagcdn.com/w320/qa.png",
    "Switzerland": "https://flagcdn.com/w320/ch.png",

    "Brazil": "https://flagcdn.com/w320/br.png",
    "Morocco": "https://flagcdn.com/w320/ma.png",
    "Haiti": "https://flagcdn.com/w320/ht.png",
    "Scotland": "https://flagcdn.com/w320/gb-sct.png",

    "USA": "https://flagcdn.com/w320/us.png",
    "Paraguay": "https://flagcdn.com/w320/py.png",
    "Australia": "https://flagcdn.com/w320/au.png",
    "Türkiye": "https://flagcdn.com/w320/tr.png",

    "Germany": "https://flagcdn.com/w320/de.png",
    "Curacao": "https://flagcdn.com/w320/cw.png",
    "Ivory Coast": "https://flagcdn.com/w320/ci.png",
    "Ecuador": "https://flagcdn.com/w320/ec.png",

    "Netherlands": "https://flagcdn.com/w320/nl.png",
    "Japan": "https://flagcdn.com/w320/jp.png",
    "Sweden": "https://flagcdn.com/w320/se.png",
    "Tunisia": "https://flagcdn.com/w320/tn.png",

    "Belgium": "https://flagcdn.com/w320/be.png",
    "Egypt": "https://flagcdn.com/w320/eg.png",
    "Iran": "https://flagcdn.com/w320/ir.png",
    "New Zealand": "https://flagcdn.com/w320/nz.png",

    "Spain": "https://flagcdn.com/w320/es.png",
    "Cape Verde": "https://flagcdn.com/w320/cv.png",
    "Saudi Arabia": "https://flagcdn.com/w320/sa.png",
    "Uruguay": "https://flagcdn.com/w320/uy.png",

    "France": "https://flagcdn.com/w320/fr.png",
    "Senegal": "https://flagcdn.com/w320/sn.png",
    "Iraq": "https://flagcdn.com/w320/iq.png",
    "Norway": "https://flagcdn.com/w320/no.png",

    "Argentina": "https://flagcdn.com/w320/ar.png",
    "Algeria": "https://flagcdn.com/w320/dz.png",
    "Austria": "https://flagcdn.com/w320/at.png",
    "Jordan": "https://flagcdn.com/w320/jo.png",

    "Portugal": "https://flagcdn.com/w320/pt.png",
    "Congo DR": "https://flagcdn.com/w320/cd.png",
    "Uzbekistan": "https://flagcdn.com/w320/uz.png",
    "Colombia": "https://flagcdn.com/w320/co.png",

    "England": "https://flagcdn.com/w320/gb-eng.png",
    "Croatia": "https://flagcdn.com/w320/hr.png",
    "Ghana": "https://flagcdn.com/w320/gh.png",
    "Panama": "https://flagcdn.com/w320/pa.png",
}

st.markdown("""
<style>
.group-card {
    background-color: #0e1117;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.group-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
}

.team-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.team-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.team-flag img {
    width: 24px;
    border-radius: 3px;
}

.team-name {
    font-size: 14px;
}

.team-stats {
    display: flex;
    gap: 12px;
    font-size: 13px;
}

.qualify {
    background-color: rgba(0, 200, 120, 0.12);
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
.hero {{
    background: linear-gradient(135deg, {PRIMARY}, #1C3D73);
    padding: 30px;
    border-radius: 16px;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.hero-text h1 {{
    font-size: 40px;
    margin-bottom: 5px;
}}

.hero-text p {{
    opacity: 0.85;
    font-size: 16px;
}}

.hero img {{
    height: 80px;
}}

.match-card {{
    background: {CARD};
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    color: white;
}}

.team-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.team {{
    text-align: center;
}}

.team img {{
    width: 60px;
}}
</style>

<div class="hero">
    <div class="hero-text">
        <h1>FIFA World Cup 2026 Simulator</h1>
        <p>AI-powered Hybrid Engine • Monte Carlo Simulation • Broadcast Analytics</p>
    </div>
    <img src="https://assets.football-logos.cc/logos/tournaments/700x700/fifa-world-cup-2026--white.9ba8a004.png">
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

# Map Flag images to relevant countries
df["home_flag"] = df["home_team"].map(FLAG_URLS)
df["away_flag"] = df["away_team"].map(FLAG_URLS)

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
def render_group_card(group_name, standings):
    rows = []

    if isinstance(standings, dict):
        for team, stats in standings.items():
            rows.append({
                "team": team,
                "points": stats.get("points", 0),
                "gd": stats.get("gd", 0),
                "goals": stats.get("goals", 0),
            })
    else:
        for item in standings:
            team, stats = item if isinstance(item, tuple) else (item["team"], item)
            rows.append({
                "team": team,
                "points": stats.get("points", 0),
                "gd": stats.get("gd", 0),
                "goals": stats.get("goals", 0),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["points", "gd", "goals"], ascending=False).reset_index(drop=True)

    # Build rows separately (important)
    rows_html = ""

    for i, row in df.iterrows():
        team = row["team"]
        flag = get_flag(team)
        qualify_class = "team-row qualify" if i < 2 else "team-row"

        rows_html += f"""
<div class="{qualify_class}">
    <div class="team-left">
        <div class="team-flag">
            <img src="{flag}">
        </div>
        <div class="team-name">{team}</div>
    </div>
    <div class="team-stats">
        <div><b>{row['points']}</b> pts</div>
        <div>{row['gd']} GD</div>
        <div>{row['goals']} G</div>
    </div>
</div>
"""

    # ✅ Final card (NO indentation before <div)
    card_html = f"""<div class="group-card">
<div class="group-title">Group {group_name}</div>
{rows_html}
</div>"""

    st.markdown(card_html.strip(), unsafe_allow_html=True)
    # st.code(card_html)

# GET COUNTRY FLAG HELPER FUNCTION
def get_flag(team):
    return FLAG_URLS.get(team, "https://flagcdn.com/w320/un.png")

# ---------------------------
# UI CONFIG
# ---------------------------
st.set_page_config(page_title="World Cup Simulator", layout="wide")


# st.image(
#     "https://assets.football-logos.cc/logos/tournaments/700x700/fifa-world-cup-2026--white.9ba8a004.png",
#     width=250
# )
# st.title("World Cup 2026 Simulator Dashboard")

# =========================================================
# SECTION 1: HYBRID MATCH PREDICTOR
# =========================================================

teams = sorted({
    team
    for group in world_cup_2026_groups.values()
    for team in group
})

st.markdown("## World Cup 2026 Match Predictor")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Home Team", teams, key="t1")

with col2:
    team2 = st.selectbox("Away Team", teams, key="t2")


# --- MATCH CARD UI ---
st.markdown(f"""
<div class="match-card">
    <div class="team-row">
        <div class="team">
            <img src="{get_flag(team1)}">
            <p>{team1}</p>
        </div>
        <div>
            <h2>VS</h2>
        </div>
        <div class="team">
            <img src="{get_flag(team2)}">
            <p>{team2}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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
if st.button("Predict Match"):
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
    st.header("Simulation Controls")

    n_sims = st.slider("Slide to Select Number of Simulations", 100, 10000, 1000, step=100)

    show_live = st.toggle("Live Simulation Mode", value=False)

    run = st.button("Run Simulation")

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
            render_group_card(group, standings)

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

    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <img src="{get_flag(team)}" width="40">
            <h3 style="margin:0;">{team}</h3>
        </div>
        """, unsafe_allow_html=True)
    st.bar_chart(prog_df.loc[team])

else:
    st.info("Run a simulation to see team progression probabilities.")