"""Streamlit entry point:  streamlit run app.py"""
import streamlit as st

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from core import theme  # noqa: E402  (after set_page_config)

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/prediction.py", title="Diabetes Prediction", icon=":material/chat:", url_path="diabetes-prediction"),
    st.Page("views/insights.py", title="Model Insights", icon=":material/insights:", url_path="model-insights"),
    st.Page("views/about.py", title="About", icon=":material/info:", url_path="about"),
]

nav = st.navigation(pages, position="top")
theme.inject()

# The last prediction summary belongs to the chatbot session on the prediction page.
if nav.url_path != "diabetes-prediction":
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("chatbot_response", None)

import streamlit as st

st.write("Gemini key loaded:", bool(st.secrets.get("GEMINI_API_KEY")))
st.write("Gemini model:", st.secrets.get("GEMINI_MODEL", "NOT FOUND"))

nav.run()
