import streamlit as st
import pandas as pd
import numpy as np
import os, sys

# ---------------------------
# PATH SETUP
# ---------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.append(PROJECT_ROOT)


def resolve_path(relpath):
    """Make data/model paths work no matter which directory the app is launched from."""
    for base in (APP_DIR, PROJECT_ROOT, os.getcwd()):
        candidate = os.path.join(base, relpath)
        if os.path.exists(candidate):
            return candidate
    return relpath


from src.models.poisson import train_model, get_lambda, score_matrix, DEFAULT_RHO
from src.models.hybrid import HybridWorldCupModel
from simulation import (
    run_simulations,
    world_cup_2026_groups,
    build_match_prob_cache,
    resolve_venue,
    HOST_NATIONS,
)
from knockout_bracket import render_broadcast_bracket

# ---------------------------
# UI CONFIG (must be the FIRST Streamlit command)
# ---------------------------
st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon=resolve_path("assets/worldcup2026.png"),
    layout="wide",
)

# ---------------------------
# THEME CONSTANTS
# ---------------------------
PRIMARY = "#0B1F3A"     # deep navy
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


def get_flag(team):
    return FLAG_URLS.get(team, "https://flagcdn.com/w320/un.png")


# ---------------------------
# GLOBAL CSS
# ---------------------------
st.markdown("""
<style>
/* ---- Tab navigation (self-contained dark nav bar) ----
   The tab bar carries its OWN navy background, so labels stay readable on ANY
   Streamlit theme (light or dark). This avoids prefers-color-scheme, which
   can't reliably detect Streamlit's rendered theme. Active tab = gold pill. */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0B1F3A;
    border: 1px solid rgba(245,197,24,0.20);
    border-radius: 12px;
    padding: 5px 6px;
    margin-bottom: 12px;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600; letter-spacing: 0.01em; padding: 8px 20px;
    border-radius: 8px;
    color: #cbd5e1 !important;                 /* light slate on the navy bar */
    background: transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: #ffffff !important; }
.stTabs [aria-selected="true"] {
    color: #0B1F3A !important;                 /* navy text ... */
    background: #F5C518 !important;            /* ... on a gold pill */
}
/* the gold pill marks the active tab, so hide the default sliding underline */
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }

/* ---- Group cards ---- */
.group-card {
    background: linear-gradient(160deg, #0d2143 0%, #0b1424 100%);
    border: 1px solid rgba(245,197,24,0.16);
    border-radius: 14px; padding: 14px 16px 10px; margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.30);
}
.group-title {
    font-size: 12px; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: #F5C518;
    margin-bottom: 10px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.10);
}
.team-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; padding: 7px 8px; border-radius: 8px; margin-bottom: 2px;
}
.team-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
.team-flag img {
    width: 24px; height: 16px; object-fit: cover; border-radius: 3px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.20); display: block;
}
.team-name { font-size: 14px; line-height: 1.2; color: #f1f5f9; }
.team-stats {
    display: flex; gap: 14px; font-size: 13px; color: #9fb0c3;
    flex: 0 0 auto; white-space: nowrap;
}
.team-stats b { color: #ffffff; font-weight: 700; }
.qualify { background: rgba(22,163,74,0.14); box-shadow: inset 3px 0 0 #16a34a; }
.qualify .team-name { color: #ffffff; font-weight: 600; }
.qualify .team-stats b { color: #86efac; }

/* ---- Match Predictor card ---- */
.pred-card {
    background: radial-gradient(900px 320px at 50% -40%, #17345F 0%, #0c1f3c 55%, #081326 100%);
    border: 1px solid rgba(245,197,24,0.18);
    border-radius: 18px; padding: 18px 22px 20px; color: #f8fafc;
    box-shadow: 0 10px 30px rgba(0,0,0,0.38);
}
.pred-venue {
    text-align: center; font-size: 11px; letter-spacing: 0.22em;
    text-transform: uppercase; color: rgba(245,197,24,0.9); margin-bottom: 14px;
}
.pred-main { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
.pred-side { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 0; }
.pred-side img {
    width: 58px; height: 39px; object-fit: cover; border-radius: 4px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.25);
}
.pred-tname { font-size: 15px; font-weight: 600; text-align: center; }
.pred-xg { font-size: 12px; color: #93a4ba; }
.pred-center { flex: 0 0 auto; text-align: center; padding: 0 6px; }
.pred-scoreline { font-size: 44px; font-weight: 800; line-height: 1; letter-spacing: 0.02em; }
.pred-scoreline span { color: rgba(245,197,24,0.8); padding: 0 6px; }
.pred-scoreprob { font-size: 11px; color: #93a4ba; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.12em; }
.pred-bar {
    display: flex; height: 26px; border-radius: 8px; overflow: hidden;
    margin: 18px 0 8px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
}
.pred-bar .seg { display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: #0b1424; white-space: nowrap; }
.pred-bar .seg1 { background: #38bdf8; }
.pred-bar .segd { background: #94a3b8; }
.pred-bar .seg2 { background: #fb7185; }
.pred-legend { display: flex; justify-content: center; gap: 18px; font-size: 12px; color: #cbd5e1; }
.pred-legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
.pred-legend .d1 { background: #38bdf8; }
.pred-legend .dd { background: #94a3b8; }
.pred-legend .d2 { background: #fb7185; }
.pred-alts { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }
.pred-alts .alt-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em; color: #7e90a6; }
.pred-chip { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 999px; padding: 4px 11px; font-size: 12px; color: #e2e8f0; }
.pred-chip b { color: #F5C518; }

/* ---- Hero ---- */
.hero {
    background: linear-gradient(135deg, #0B1F3A, #1C3D73);
    padding: 26px 30px; border-radius: 16px; color: white;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 6px;
}
.hero-text h1 { font-size: 36px; margin-bottom: 4px; }
.hero-text p { opacity: 0.85; font-size: 15px; }
.hero img { height: 76px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HERO
# ---------------------------
st.markdown("""
<div class="hero">
    <div class="hero-text">
        <h1>FIFA World Cup 2026 Simulator</h1>
        <p>AI-powered Hybrid Engine • Monte Carlo Simulation • Broadcast Analytics</p>
    </div>
    <img src="https://assets.football-logos.cc/logos/tournaments/700x700/fifa-world-cup-2026--white.9ba8a004.png">
</div>
""", unsafe_allow_html=True)


# ---------------------------
# CACHING + LOADING
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(resolve_path("data/processed/matches_with_features.csv"))
    final_elo = pd.read_csv(resolve_path("data/processed/final_elo.csv"))
    return df, final_elo


@st.cache_resource
def load_hybrid_model():
    return HybridWorldCupModel.load(resolve_path("models/hybrid.pkl"))


@st.cache_resource
def load_prob_cache(_model, _teams):
    save_path = os.path.join(resolve_path("models"), "prob_cache.pkl")
    return build_match_prob_cache(_teams, _model, save_path=save_path)


@st.cache_resource
def get_poisson_params(df):
    return train_model(df)


df, final_elo = load_data()
elo_dict = dict(zip(final_elo["team"], final_elo["elo"]))
teams = sorted({team for group in world_cup_2026_groups.values() for team in group})
model = load_hybrid_model()
prob_cache = load_prob_cache(model, tuple(teams))
beta_home, beta_away = get_poisson_params(df)


# ---------------------------
# SIMULATION RUNNER (single source of truth — called by either Run button)
# ---------------------------
def run_and_store(n):
    with st.spinner(f"Simulating World Cup 2026 — {n:,} tournaments..."):
        results, progression, group_results, bracket = run_simulations(
            n, world_cup_2026_groups, prob_cache, elo_dict, beta_home, beta_away
        )
    st.session_state["results"] = results
    st.session_state["progression"] = progression
    st.session_state["n_sims"] = n
    st.session_state["group_results"] = group_results
    st.session_state["bracket"] = bracket


SIM_HELP = ("More runs = smoother odds. 1,000 is plenty for a quick look; "
            "5,000 for stable probabilities.")


# ---------------------------
# GROUP STAGE CARD
# ---------------------------
def render_group_card(group_name, standings):
    rows = []
    if isinstance(standings, dict):
        for team, stats in standings.items():
            rows.append({"team": team, "points": stats.get("points", 0),
                         "gd": stats.get("gd", 0), "goals": stats.get("goals", 0)})
    else:
        for item in standings:
            team, stats = item if isinstance(item, tuple) else (item["team"], item)
            rows.append({"team": team, "points": stats.get("points", 0),
                         "gd": stats.get("gd", 0), "goals": stats.get("goals", 0)})

    table = pd.DataFrame(rows)
    table = table.sort_values(["points", "gd", "goals"], ascending=False).reset_index(drop=True)

    def fmt_gd(value):
        try:
            return f"{int(value):+d}"
        except (ValueError, TypeError):
            return str(value)

    rows_html = ""
    for i, row in table.iterrows():
        team = row["team"]
        flag = get_flag(team)
        qualify_class = "team-row qualify" if i < 2 else "team-row"
        rows_html += (
            f'<div class="{qualify_class}">'
            f'<div class="team-left">'
            f'<div class="team-flag"><img src="{flag}"></div>'
            f'<div class="team-name">{team}</div>'
            f'</div>'
            f'<div class="team-stats">'
            f'<div><b>{row["points"]}</b> pts</div>'
            f'<div>{fmt_gd(row["gd"])} GD</div>'
            f'<div>{row["goals"]} G</div>'
            f'</div>'
            f'</div>'
        )

    card_html = (
        f'<div class="group-card">'
        f'<div class="group-title">Group {group_name}</div>'
        f'{rows_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------
# MATCH PREDICTOR HELPERS
# ---------------------------
def venue_aware_lambdas(team1, team2):
    neutral, home = resolve_venue(team1, team2)
    if neutral:
        l1, l2 = get_lambda(team1, team2, elo_dict, model.beta_home, model.beta_away, neutral=True)
    elif home == team1:
        l1, l2 = get_lambda(team1, team2, elo_dict, model.beta_home, model.beta_away, neutral=False)
    else:
        l2, l1 = get_lambda(team2, team1, elo_dict, model.beta_home, model.beta_away, neutral=False)
    return l1, l2, neutral, home


def outcome_probs(team1, team2):
    p = prob_cache.get((team1, team2))
    if p is None:
        neutral, home = resolve_venue(team1, team2)
        pr = model.predict(team1, team2, neutral=neutral)
        p = np.array([pr["home_win"], pr["draw"], pr["away_win"]])
    return np.asarray(p, dtype=float)


def scoreline_matrix(l1, l2, max_goals=6):
    return score_matrix(l1, l2, rho=DEFAULT_RHO, max_goals=max_goals)


def top_scorelines(M, k=3):
    flat = M.flatten()
    idx = np.argsort(flat)[::-1][:k]
    out = []
    for f in idx:
        gi, gj = divmod(int(f), M.shape[1])
        out.append((gi, gj, float(M[gi, gj])))
    return out


def coherent_score(M, outcome):
    n = M.shape[0]
    best, best_p = (1, 1), -1.0
    for i in range(n):
        for j in range(n):
            if outcome == "home" and not i > j:
                continue
            if outcome == "draw" and not i == j:
                continue
            if outcome == "away" and not i < j:
                continue
            if M[i, j] > best_p:
                best_p, best = M[i, j], (i, j)
    return best, best_p


def pct_round(values):
    raw = [v * 100 for v in values]
    floors = [int(np.floor(x)) for x in raw]
    remainder = 100 - sum(floors)
    order = sorted(range(len(raw)), key=lambda k: raw[k] - floors[k], reverse=True)
    for k in range(remainder):
        floors[order[k]] += 1
    return floors


def render_match_predictor():
    st.subheader("Head-to-head predictor")
    st.caption("Pick two teams to see win/draw odds, a predicted scoreline, and expected goals. "
               "Updates instantly — no button needed.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, key="t1")
    with col2:
        away_default = 1 if len(teams) > 1 else 0
        team2 = st.selectbox("Team 2", teams, index=away_default, key="t2")

    if team1 == team2:
        st.warning("Pick two different teams to predict a match.")
        return

    l1, l2, neutral, home = venue_aware_lambdas(team1, team2)
    p1, pdraw, p2 = outcome_probs(team1, team2)

    M = scoreline_matrix(l1, l2)
    outcome = ["home", "draw", "away"][int(np.argmax([p1, pdraw, p2]))]
    (gs1, gs2), score_p = coherent_score(M, outcome)
    alts = top_scorelines(M, 3)
    ph, pd_, pa = pct_round([p1, pdraw, p2])

    venue_label = "Neutral venue" if neutral else f"{home} at home · host advantage applied"

    def seg(width, cls):
        label = f"{width}%" if width >= 8 else ""
        return f'<div class="seg {cls}" style="width:{width}%">{label}</div>'

    chips = "".join(f'<span class="pred-chip">{a}&ndash;{b} <b>{p:.0%}</b></span>' for a, b, p in alts)

    card = (
        '<div class="pred-card">'
        f'<div class="pred-venue">{venue_label}</div>'
        '<div class="pred-main">'
        f'<div class="pred-side"><img src="{get_flag(team1)}">'
        f'<div class="pred-tname">{team1}</div><div class="pred-xg">{l1:.2f} xG</div></div>'
        '<div class="pred-center">'
        f'<div class="pred-scoreline">{gs1}<span>&ndash;</span>{gs2}</div>'
        f'<div class="pred-scoreprob">predicted score · {score_p:.0%} likely</div>'
        '</div>'
        f'<div class="pred-side"><img src="{get_flag(team2)}">'
        f'<div class="pred-tname">{team2}</div><div class="pred-xg">{l2:.2f} xG</div></div>'
        '</div>'
        '<div class="pred-bar">'
        f'{seg(ph, "seg1")}{seg(pd_, "segd")}{seg(pa, "seg2")}'
        '</div>'
        '<div class="pred-legend">'
        f'<span><i class="dot d1"></i>{team1} win</span>'
        '<span><i class="dot dd"></i>Draw</span>'
        f'<span><i class="dot d2"></i>{team2} win</span>'
        '</div>'
        '<div class="pred-alts">'
        '<span class="alt-label">Other likely scores</span>'
        f'{chips}'
        '</div>'
        '</div>'
    )
    st.markdown(card, unsafe_allow_html=True)

    labels = {"home": f"{team1} to win", "draw": "Draw", "away": f"{team2} to win"}
    conf = max(ph, pd_, pa)
    st.caption(f"Most likely outcome: **{labels[outcome]}** ({conf}%) — predicted score {gs1}-{gs2}.")


# ---------------------------
# SIDEBAR — SIMULATION CONTROLS (unique keys: *_side)
# ---------------------------
with st.sidebar:
    st.markdown("### Simulation Controls")
    n_sims_side = st.slider(
        "Number of simulations", 100, 5000, 1000, step=100,
        key="n_sims_side", help=SIM_HELP,
    )
    run_side = st.button("▶  Run Simulation", use_container_width=True,
                         type="primary", key="run_side")
    st.caption("Each run plays out the whole tournament thousands of times.")

    st.divider()
    with st.expander("About this app"):
        st.markdown(
            "A hybrid **Elo (Team Strength Rating) + Poisson + XGBoost** engine runs a Monte Carlo "
            "simulation of FIFA World Cup 2026.\n\n"
            "- **Match Predictor** — head-to-head odds & predicted score\n"
            "- **Tournament Simulation** — Simulate the entire tournament. From group stage to knockout bracket & winner odds\n"
            "- **Team Explorer** — how far each team is likely to go"
        )

# Sidebar button runs the sim BEFORE the tabs render, so results show immediately.
if run_side:
    run_and_store(n_sims_side)


# ===========================================================================
# MAIN NAVIGATION — TABS
# ===========================================================================
tab_predict, tab_sim, tab_team = st.tabs([
    "Match Predictor",
    "Tournament Simulation",
    "Team Explorer",
])

# --- TAB 1: MATCH PREDICTOR ---
with tab_predict:
    if model is None:
        st.error("Hybrid model not loaded")
    else:
        render_match_predictor()

# --- TAB 2: TOURNAMENT SIMULATION ---
with tab_sim:
    if "results" not in st.session_state:
        st.info(
            "Set the number of simulations in the sidebar on the left or below, then click "
            "**▶ Run Simulation** to generate a knockout bracket, group tables, and title odds."
        )

        # In-tab controls use unique keys (*_tab) to avoid the DuplicateWidgetID clash.
        st.markdown("### Simulation Controls")
        n_sims_tab = st.slider(
            "Number of simulations", 100, 5000, 1000, step=100,
            key="n_sims_tab", help=SIM_HELP,
        )
        run_tab = st.button("▶  Run Simulation", use_container_width=True,
                            type="primary", key="run_tab")
        st.caption("Each run plays out the whole tournament thousands of times.")

        # This button is defined after the sidebar's run check, so trigger the
        # run here and rerun — the rerun then falls into the results branch.
        if run_tab:
            run_and_store(n_sims_tab)
            st.rerun()
    else:
        results = st.session_state["results"]
        progression = st.session_state["progression"]
        n_sims = st.session_state["n_sims"]
        group_results = st.session_state["group_results"]
        bracket = st.session_state["bracket"]

        # Headline champion (taken from the displayed bracket so it can't drift)
        final = bracket.get("Final", {})
        champ_list = final.get("winners") or []
        champion = champ_list[0] if champ_list else max(results, key=results.get)

        # --- Knockout bracket ---
        st.subheader("Knockout Bracket")
        render_broadcast_bracket(bracket, get_flag, champion=champion)
        st.caption(
            "One representative tournament won by the most likely champion — a single "
            "plausible path, not the full forecast (see Title Odds below)."
        )

        st.divider()

        # --- Group stage ---
        st.subheader("Group Stage Standings")
        st.caption("Top two of each group (green) advance, plus the eight best third-placed teams.")
        cols = st.columns(4)
        for i, (group, standings) in enumerate(group_results.items()):
            with cols[i % 4]:
                render_group_card(group, standings)

        st.divider()

        # --- Title odds ---
        st.subheader("Win Probability Percentage")
        results_df = pd.DataFrame.from_dict(results, orient="index", columns=["wins"])
        results_df["probability"] = results_df["wins"] / n_sims
        results_df = results_df.sort_values("probability", ascending=False)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.bar_chart(results_df.head(10)["probability"])
        with c2:
            st.dataframe(
                results_df.head(12).assign(
                    probability=lambda d: (d["probability"] * 100).round(1)
                )[["probability"]].rename(columns={"probability": "Win %"}),
                use_container_width=True,
            )

        st.divider()

        # --- Progression detail ---
        st.subheader("Tournament Progression")
        st.caption("Share of simulations in which each team reaches a given stage.")
        prog_df = pd.DataFrame(progression).T.div(n_sims)
        sort_col = "Winner" if "Winner" in prog_df.columns else prog_df.columns[-1]
        st.dataframe(
            (prog_df.sort_values(sort_col, ascending=False).head(20) * 100).round(1),
            use_container_width=True,
        )

# --- TAB 3: TEAM EXPLORER ---
with tab_team:
    st.subheader("Team Explorer")
    team = st.selectbox("Select a team", teams, key="team_view")

    prog_df = None
    if "progression" in st.session_state:
        prog_df = pd.DataFrame(st.session_state["progression"]).T.div(st.session_state["n_sims"])

    if prog_df is not None and team in prog_df.index:
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:10px; margin:4px 0 8px;">'
            f'<img src="{get_flag(team)}" width="40">'
            f'<h3 style="margin:0;">{team}</h3></div>',
            unsafe_allow_html=True,
        )
        st.caption("How far this team goes, across all simulations.")
        st.bar_chart(prog_df.loc[team])
    else:
        st.info("Run a simulation first (sidebar → ▶ Run Simulation) to see how far each team goes.")