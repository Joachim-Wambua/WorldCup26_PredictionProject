"""
Broadcast-style knockout bracket renderer for the WC26 simulator.

Why the old bracket broke
-------------------------
1. Matches were positioned with hand-tuned margins per round
   (.round-2 .match { margin: 100px 0 }), so they never lined up with the
   games that feed them, and the Final overflowed off the bottom.
2. The vertical connector (.round-col::before) spanned the FULL column
   height -> the floor-to-ceiling lines in the screenshot.
3. Three conflicting CSS blocks redefined .match / .team / .round-col
   (one even had `gap: spx`), so styles fought each other.

How this version works
----------------------
- The whole bracket has a fixed height. Every round column is a flex
  column with `justify-content: space-around`, and each PAIR of matches
  is its own flex box. The math works out so a pair's vertical centre
  always lands exactly on the match it feeds in the next round —
  no magic margins, ever.
- Connector lines are drawn per-pair (top 25% -> bottom 75% of the pair),
  so they only span between the two feeder matches.
- All classes are namespaced `bk-` so nothing collides with the hero /
  match-card CSS in app.py.

Usage (in app.py)
-----------------
    from knockout_bracket import render_broadcast_bracket

    if "bracket" in st.session_state:
        champion = max(st.session_state["results"],
                       key=st.session_state["results"].get)
        render_broadcast_bracket(
            st.session_state["bracket"],
            get_flag,
            champion=champion,
        )

Then delete from app.py:
- the big "# KNOCKOUT BRACKET" st.markdown CSS block
- the old render_bracket, render_match, render_side,
  render_broadcast_bracket and split_bracket functions
- (optional) the separate "Champion" markdown block — the trophy card
  in the centre column now shows the champion.
"""

import streamlit as st

_SIDE_ROUNDS = ["R32", "R16", "QF", "SF"]
_ROUND_LABELS = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-finals",
    "SF": "Semi-finals",
}

# ---------------------------------------------------------------------------
# THEME — WC26 broadcast: navy stadium backdrop, FIFA-red card spine,
# gold for winners / connectors / the Final.
# ---------------------------------------------------------------------------
_BRACKET_CSS = """
<style>
.bk-wrap{
  background:
    radial-gradient(1100px 540px at 50% -10%, #17345F 0%, #0B1F3A 52%, #071228 100%);
  border:1px solid rgba(245,197,24,.16);
  border-radius:18px;
  padding:18px 14px 26px;
  overflow-x:auto;
}
.bk-head{text-align:center;margin-bottom:4px;}
.bk-kicker{color:#F5C518;font-size:11px;letter-spacing:.32em;text-transform:uppercase;}
.bk-head h3{color:#fff;margin:2px 0 12px;font-size:22px;letter-spacing:.02em;}

.bk-container{display:flex;min-width:1500px;height:740px;}
.bk-side{display:flex;flex:1;}
.bk-right{flex-direction:row-reverse;}

.bk-round{flex:1;display:flex;flex-direction:column;min-width:168px;}
.bk-round-title{
  text-align:center;color:rgba(245,197,24,.85);
  font-size:10px;letter-spacing:.28em;text-transform:uppercase;
  padding:2px 0 8px;
}
.bk-body{flex:1;display:flex;flex-direction:column;justify-content:space-around;}

/* A pair = the two matches whose winners meet next round.
   Matches sit at 25% / 75% of the pair, so the connector spans exactly
   between them and the pair's midpoint feeds the next round. */
.bk-pair{
  flex:1;display:flex;flex-direction:column;justify-content:space-around;
  position:relative;margin:0 22px;
}
.bk-solo{justify-content:center;}
.bk-mw{position:relative;}

/* match card */
.bk-match{
  background:rgba(13,23,42,.92);
  border:1px solid rgba(255,255,255,.09);
  border-left:3px solid #E10600;
  border-radius:10px;
  padding:5px 8px;
  box-shadow:0 8px 18px rgba(0,0,0,.38);
}
.bk-right .bk-match{
  border-left:1px solid rgba(255,255,255,.09);
  border-right:3px solid #E10600;
}
.bk-team{
  display:flex;align-items:center;gap:8px;
  padding:3px 2px;color:#cbd5e1;font-size:13px;line-height:1.25;
  opacity:.6;white-space:nowrap;
}
.bk-right .bk-team{flex-direction:row-reverse;}
.bk-team img{
  width:22px;height:15px;object-fit:cover;border-radius:2px;
  box-shadow:0 0 0 1px rgba(255,255,255,.18);flex:0 0 auto;
}
.bk-team span{overflow:hidden;text-overflow:ellipsis;}
.bk-win{opacity:1;color:#F5C518;font-weight:700;}

/* connectors — gold, only between the two feeder matches */
.bk-left .bk-duo::after{
  content:"";position:absolute;right:-12px;top:25%;bottom:25%;
  width:2px;background:rgba(245,197,24,.4);
}
.bk-left .bk-duo .bk-mw::after{
  content:"";position:absolute;right:-12px;top:50%;
  width:12px;height:2px;background:rgba(245,197,24,.4);
}
.bk-left .bk-pair::before{
  content:"";position:absolute;right:-44px;top:50%;
  width:32px;height:2px;background:rgba(245,197,24,.4);
}
.bk-right .bk-duo::after{
  content:"";position:absolute;left:-12px;top:25%;bottom:25%;
  width:2px;background:rgba(245,197,24,.4);
}
.bk-right .bk-duo .bk-mw::after{
  content:"";position:absolute;left:-12px;top:50%;
  width:12px;height:2px;background:rgba(245,197,24,.4);
}
.bk-right .bk-pair::before{
  content:"";position:absolute;left:-44px;top:50%;
  width:32px;height:2px;background:rgba(245,197,24,.4);
}

/* the Final — centre column, gold glow, trophy, champion */
.bk-final{
  width:240px;display:flex;flex-direction:column;
  justify-content:center;align-items:center;gap:8px;padding:0 6px;
}
.bk-trophy{font-size:38px;filter:drop-shadow(0 0 14px rgba(245,197,24,.55));}
.bk-final-label{color:#F5C518;font-size:11px;letter-spacing:.34em;text-transform:uppercase;}
.bk-final .bk-match{
  width:100%;
  border:1px solid rgba(245,197,24,.55);
  box-shadow:0 0 26px rgba(245,197,24,.22);
}
.bk-final .bk-team{font-size:15px;justify-content:center;}
.bk-champ{margin-top:8px;text-align:center;color:#fff;}
.bk-champ img{width:44px;border-radius:4px;box-shadow:0 0 0 1px rgba(255,255,255,.25);}
.bk-champ-name{margin-top:4px;font-weight:700;color:#F5C518;letter-spacing:.04em;}

@media (prefers-reduced-motion: no-preference){
  .bk-match{animation:bkFade .45s ease both;}
  @keyframes bkFade{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}
}
</style>
"""


# ---------------------------------------------------------------------------
# data helpers — tolerant of (t1, t2) tuples, (t1, t2, winner) tuples,
# {"team1","team2","winner"} dicts, and rounds stored either as a plain
# list or as {"matches": [...]}.
# ---------------------------------------------------------------------------
def _matches_of(bracket, rnd):
    entry = bracket.get(rnd, [])
    if isinstance(entry, dict):
        return entry.get("matches", []) or []
    return entry or []


def _normalize(match):
    if isinstance(match, dict):
        return match.get("team1"), match.get("team2"), match.get("winner")
    if isinstance(match, (list, tuple)):
        if len(match) >= 3:
            return match[0], match[1], match[2]
        if len(match) == 2:
            return match[0], match[1], None
    return str(match), "", None


def _advancers(bracket, champion=None):
    """Winners per round = teams that appear in the next round."""
    out = {}
    order = _SIDE_ROUNDS + ["Final"]
    for i, rnd in enumerate(order[:-1]):
        nxt = set()
        for m in _matches_of(bracket, order[i + 1]):
            t1, t2, _ = _normalize(m)
            nxt.update([t1, t2])
        out[rnd] = nxt
    out["Final"] = {champion} if champion else set()
    return out


def _split(matches):
    mid = len(matches) // 2
    return matches[:mid], matches[mid:][::-1]  # right side mirrored


def _chunk_pairs(items):
    pairs, i = [], 0
    while i < len(items):
        pairs.append(items[i:i + 2])
        i += 2
    return pairs


# ---------------------------------------------------------------------------
# html builders (single-line strings — indented HTML trips up st.markdown)
# ---------------------------------------------------------------------------
def _team_html(team, get_flag, is_winner):
    cls = "bk-team bk-win" if is_winner else "bk-team"
    flag = get_flag(team) if team else ""
    name = team or "TBD"
    return f'<div class="{cls}"><img src="{flag}" alt=""><span>{name}</span></div>'


def _match_html(match, get_flag, round_winners):
    t1, t2, explicit = _normalize(match)
    def won(t):
        return t is not None and (t == explicit or t in round_winners)
    return (
        '<div class="bk-match">'
        + _team_html(t1, get_flag, won(t1))
        + _team_html(t2, get_flag, won(t2))
        + "</div>"
    )


def _side_html(side_matches, get_flag, winners_by_round, side):
    html = f'<div class="bk-side bk-{side}">'
    for rnd in _SIDE_ROUNDS:
        matches = side_matches.get(rnd, [])
        html += f'<div class="bk-round"><div class="bk-round-title">{_ROUND_LABELS[rnd]}</div><div class="bk-body">'
        for pair in _chunk_pairs(matches):
            pair_cls = "bk-pair bk-duo" if len(pair) == 2 else "bk-pair bk-solo"
            html += f'<div class="{pair_cls}">'
            for m in pair:
                html += '<div class="bk-mw">' + _match_html(m, get_flag, winners_by_round.get(rnd, set())) + "</div>"
            html += "</div>"
        html += "</div></div>"
    html += "</div>"
    return html


def _final_html(bracket, get_flag, winners_by_round, champion):
    finals = _matches_of(bracket, "Final")
    html = '<div class="bk-final"><div class="bk-trophy">🏆</div><div class="bk-final-label">Final</div>'
    if finals:
        html += _match_html(finals[0], get_flag, winners_by_round.get("Final", set()))
    if champion:
        html += (
            '<div class="bk-champ">'
            f'<img src="{get_flag(champion)}" alt="">'
            f'<div class="bk-champ-name">{champion}</div>'
            '<div style="font-size:11px;opacity:.7;">World Champions</div>'
            "</div>"
        )
    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def render_broadcast_bracket(bracket, get_flag, champion=None, title="Road to the Final"):
    """Render the full broadcast bracket.

    bracket  : dict with keys R32/R16/QF/SF/Final (lists or {"matches": [...]})
    get_flag : callable team-name -> flag url (your existing get_flag)
    champion : tournament winner, highlights the Final + champion card
    """
    left, right = {}, {}
    for rnd in _SIDE_ROUNDS:
        l, r = _split(_matches_of(bracket, rnd))
        left[rnd], right[rnd] = l, r

    winners = _advancers(bracket, champion)

    html = (
        '<div class="bk-wrap">'
        '<div class="bk-head"><div class="bk-kicker">FIFA World Cup 2026</div>'
        f"<h3>{title}</h3></div>"
        '<div class="bk-container">'
        + _side_html(left, get_flag, winners, "left")
        + _final_html(bracket, get_flag, winners, champion)
        + _side_html(right, get_flag, winners, "right")
        + "</div></div>"
    )

    st.markdown(_BRACKET_CSS, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)
