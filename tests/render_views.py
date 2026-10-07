"""Developer tool: run each page's Python against a tiny stand-in for the st.* API,
collect the HTML it emits, and screenshot it with the real theme CSS in Chromium.
This checks page logic + visual design; it is NOT a Streamlit run.   python tests/render_views.py
"""
import contextlib, runpy, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = []

class Stop(Exception): pass
class SS(dict):
    __getattr__ = dict.get
def _md(body, unsafe_allow_html=False, **k): OUT.append(body)
st = types.ModuleType("streamlit")
st.session_state = SS()
st.markdown = _md
st.write = lambda *a, **k: OUT.append("<div style='height:12px'></div>")
st.subheader = lambda t, **k: OUT.append(f"<h3 style='margin:22px 0 10px'>{t}</h3>")
st.error = lambda t, **k: OUT.append(f"<div style='color:#9b2458'>{t}</div>")
st.stop = lambda: (_ for _ in ()).throw(Stop())
st.button = lambda label, **k: (OUT.append(f"<button style='padding:12px 26px;border:0;border-radius:12px;color:#fff;font-weight:700;background:linear-gradient(135deg,#5B5BD6,#8B7CF0)'>{label}</button>") or False)
st.switch_page = lambda p: None
st.rerun = lambda: None
@contextlib.contextmanager
def _col(): OUT.append("<div class='col'>"); yield; OUT.append("</div>")
def columns(n, **k):
    OUT.append(f"<div style='display:grid;grid-template-columns:repeat({n},1fr);gap:16px'>")
    cols = [_col() for _ in range(n)]
    return cols
st.columns = columns
st.cache_resource = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
st.cache_data = st.cache_resource
st.secrets = {}
comp = types.ModuleType("streamlit.components.v1")
comp.declare_component = lambda name, path=None: (lambda **k: None)
sys.modules.update({"streamlit": st, "streamlit.components": types.ModuleType("streamlit.components"), "streamlit.components.v1": comp})
st.components = sys.modules["streamlit.components"]; st.components.v1 = comp

from core import theme
css = theme._CSS.replace("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');", "")

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    for name in ("home", "insights", "about", "prediction"):
        OUT.clear(); st.session_state.clear()
        if name == "prediction":   # simulate a finished prediction to render the result panel
            st.session_state["last_prediction"] = {"prediction_code": 2, "probabilities": {"0": 0.52, "1": 0.03, "2": 0.45}}
        try:
            runpy.run_path(str(ROOT / "views" / f"{name}.py"))
        except Stop:
            pass
        html = f"<meta charset=utf-8>{css}<body style='margin:0;background:#F6F5FD;font-family:Inter,system-ui,sans-serif'><div style='max-width:1180px;margin:0 auto;padding:24px 16px'>{''.join(OUT)}</div>"
        for w in (1280, 390):
            pg = b.new_page(viewport={"width": w, "height": 900})
            pg.set_content(html); pg.wait_for_timeout(200)
            ov = pg.evaluate("document.documentElement.scrollWidth")
            pg.screenshot(path=f"/tmp/view_{name}_{w}.png", full_page=True)
            print(f"{name:10s} {w}px  html={len(html):6d} chars  scrollWidth={ov} {'OVERFLOW' if ov>w+1 else 'ok'}")
            pg.close()
    b.close()
