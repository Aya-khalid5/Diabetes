"""Cached, Streamlit-aware accessors. The heavy objects are created ONCE per server process."""
from __future__ import annotations

import streamlit as st

from .assistant import Assistant, from_secrets
from .predictor import Predictor


@st.cache_resource(show_spinner="Loading the trained model…")
def get_predictor() -> Predictor:
    return Predictor.load()


def get_assistant() -> Assistant:
    # Deliberately NOT cached: it is cheap (the Gemini client itself is cached per key),
    # and this way a key added/changed in Secrets is picked up without rebooting the app.
    return from_secrets(st.secrets)


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return get_predictor().meta
