"""Visual system for the Streamlit shell.

Palette = the one already used by the chatbot's style.css, so the shell and the
embedded chatbot feel like one product. This CSS only styles the Streamlit shell;
the chatbot lives in its own iframe and is not affected by it.
"""
from __future__ import annotations

import html

import streamlit as st

NAVY = "#1B2A6B"
INDIGO = "#5B5BD6"
VIOLET = "#8B7CF0"
LAVENDER = "#C9C2FA"
PINK = "#F472B6"
MINT = "#2BB48A"
AMBER = "#F5B83D"
BG = "#F6F5FD"
TEXT = "#17214f"
MUTED = "#68709a"
BORDER = "#e4e3f1"

# class code -> colour (matches the real labels: No Diabetes / Prediabetes / Diabetes)
CLASS_COLORS = {"0": MINT, "1": AMBER, "2": PINK}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.stApp {{ background: {BG}; color: {TEXT}; }}
[data-testid="stHeader"] {{ background: rgba(246,245,253,.88); backdrop-filter: blur(8px); }}
[data-testid="stStatusWidget"], footer, [data-testid="stDecoration"] {{ display: none !important; visibility: hidden; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }}

h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -.01em; }}
a {{ color: {INDIGO}; }}

/* ---------- buttons ---------- */
.stButton > button, .stLinkButton > a {{
    border-radius: 12px; font-weight: 700; padding: .65rem 1.4rem; border: 1px solid {LAVENDER};
}}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {INDIGO}, {VIOLET}); color: #fff; border: 0;
    box-shadow: 0 10px 22px rgba(91,91,214,.25);
}}
.stButton > button:hover {{ filter: brightness(1.04); transform: translateY(-1px); }}

/* ---------- hero ---------- */
.hero {{
    border-radius: 26px; padding: 44px 48px; color: #fff; margin-bottom: 26px;
    background: radial-gradient(circle at 12% 18%, rgba(201,194,250,.28), transparent 30%),
                linear-gradient(135deg, {NAVY}, #273b84 55%, {INDIGO});
    box-shadow: 0 18px 45px rgba(27,42,107,.18);
}}
.hero .eyebrow {{ color: {LAVENDER}; font-weight: 700; letter-spacing: 1.2px; font-size: 13px; text-transform: uppercase; margin-bottom: 8px; }}
.hero h1 {{ color: #fff; margin: 0 0 12px; font-size: clamp(30px, 5vw, 50px); line-height: 1.1; }}
.hero p {{ margin: 0; max-width: 760px; font-size: 17px; line-height: 1.65; opacity: .92; }}

.page-title {{ margin: 0 0 4px; }}
.page-sub {{ color: {MUTED}; margin: 0 0 22px; font-size: 15px; }}

/* ---------- cards ---------- */
.card {{
    background: rgba(255,255,255,.96); border: 1px solid {BORDER}; border-radius: 20px;
    padding: 24px 26px; margin-bottom: 18px; box-shadow: 0 18px 45px rgba(27,42,107,.07);
}}
.card h3 {{ margin: 0 0 8px; font-size: 19px; }}
.card p, .card li {{ color: #38426e; line-height: 1.7; font-size: 15px; overflow-wrap: anywhere; }}
.card code {{ overflow-wrap: anywhere; }}
.card ul {{ margin: 6px 0 0; padding-left: 20px; }}
.kicker {{ color: {INDIGO}; font-size: 12px; font-weight: 800; letter-spacing: .8px; text-transform: uppercase; margin-bottom: 4px; }}

.grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin-bottom: 18px; }}
.stat {{ background: #fff; border: 1px solid {BORDER}; border-radius: 18px; padding: 18px 20px; box-shadow: 0 12px 30px rgba(27,42,107,.06); }}
.stat .label {{ color: {MUTED}; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }}
.stat .value {{ color: {NAVY}; font-size: 26px; font-weight: 800; margin-top: 4px; line-height: 1.15; overflow-wrap: anywhere; }}
.stat .note {{ color: {INDIGO}; font-size: 12px; margin-top: 4px; }}

.chip {{ display: inline-block; background: #eeeafd; color: {INDIGO}; border-radius: 999px; padding: 6px 13px; margin: 0 6px 8px 0; font-size: 13px; font-weight: 700; }}

/* ---------- workflow ---------- */
.flow {{ display: flex; flex-direction: column; align-items: center; gap: 0; }}
.flow .step {{ width: min(420px, 100%); text-align: center; background: #fff; border: 1px solid {LAVENDER}; border-radius: 14px; padding: 12px 16px; font-weight: 700; color: {NAVY}; }}
.flow .step small {{ display: block; font-weight: 500; color: {MUTED}; font-size: 12px; margin-top: 2px; }}
.flow .arrow {{ color: {VIOLET}; font-size: 22px; line-height: 1.2; }}

/* ---------- probability bars ---------- */
.prob-row {{ display: grid; grid-template-columns: 130px 1fr 64px; gap: 12px; align-items: center; margin: 10px 0; font-size: 14px; }}
.prob-track {{ background: #eeeafd; border-radius: 999px; height: 12px; overflow: hidden; }}
.prob-fill {{ height: 100%; border-radius: 999px; }}
.prob-row.top .name {{ font-weight: 800; color: {NAVY}; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 4px 12px; font-size: 12px; font-weight: 800; color: #fff; }}

.small {{ color: {MUTED}; font-size: 12.5px; line-height: 1.6; }}

table.clean {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
table.clean th {{ text-align: left; color: {MUTED}; font-size: 12px; text-transform: uppercase; letter-spacing: .5px; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
table.clean td {{ padding: 10px; border-bottom: 1px solid #f0eff8; color: {TEXT}; }}
.table-wrap {{ overflow-x: auto; }}

@media (max-width: 800px) {{
    .hero {{ padding: 30px 24px; border-radius: 20px; }}
    .hero p {{ font-size: 15px; }}
    .prob-row {{ grid-template-columns: 96px 1fr 52px; }}
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def e(text: object) -> str:
    return html.escape(str(text))


def hero(title: str, subtitle: str, eyebrow: str = "") -> None:
    eb = f'<div class="eyebrow">{e(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f'<div class="hero">{eb}<h1>{e(title)}</h1><p>{e(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<h1 class="page-title">{e(title)}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="page-sub">{e(subtitle)}</p>', unsafe_allow_html=True)


def stats(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, note)"""
    cells = "".join(
        f'<div class="stat"><div class="label">{e(l)}</div><div class="value">{e(v)}</div>'
        + (f'<div class="note">{e(n)}</div>' if n else "")
        + "</div>"
        for l, v, n in items
    )
    st.markdown(f'<div class="grid">{cells}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str, kicker: str = "") -> None:
    k = f'<div class="kicker">{e(kicker)}</div>' if kicker else ""
    st.markdown(f'<div class="card">{k}<h3>{e(title)}</h3>{body_html}</div>', unsafe_allow_html=True)


def chips(names: list[str]) -> str:
    return "".join(f'<span class="chip">{e(n)}</span>' for n in names)


def flow(steps: list[tuple[str, str]]) -> None:
    parts = []
    for i, (name, note) in enumerate(steps):
        if i:
            parts.append('<div class="arrow">↓</div>')
        small = f"<small>{e(note)}</small>" if note else ""
        parts.append(f'<div class="step">{e(name)}{small}</div>')
    st.markdown(f'<div class="card"><div class="flow">{"".join(parts)}</div></div>', unsafe_allow_html=True)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
