"""Cached, Streamlit-aware accessors. The heavy objects are created ONCE per server process."""
from __future__ import annotations

import streamlit as st

from .assistant import Assistant, from_secrets
from .predictor import Predictor

@st.cache_resource(show_spinner="Loading the trained model…")
def get_predictor() -> Predictor:
    return Predictor.load()


@st.cache_resource(show_spinner=False)
def get_assistant() -> Assistant:
    return from_secrets(st.secrets)


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return get_predictor().meta
