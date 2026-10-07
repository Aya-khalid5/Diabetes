"""Streamlit custom component (v1, iframe) that hosts the ORIGINAL chatbot.

An iframe is used on purpose: it isolates the chatbot's own global CSS/JS
(body, *, .card, #ids ...) from Streamlit and vice-versa, so the chatbot renders
exactly as it did standalone. Components v2 and st.components.v1.html render
without that isolation (v2) or are one-way/deprecated (html).

Round trip:
  chatbot JS --setComponentValue({id, route, payload})--> returned here by chatbot()
  Python computes the answer, passes it back as the `response` argument
  chatbot JS receives it in the next `streamlit:render` event and resolves its fetch().
Because `key` is set, changing `response` re-renders the SAME iframe (no remount),
so the chat history on screen is preserved.
"""
from __future__ import annotations

import streamlit.components.v1 as components

from . import config

_component = components.declare_component("diabetes_chatbot", path=str(config.CHATBOT_DIR))


def chatbot(response: dict | None, key: str = "diabetes_chatbot_bridge"):
    """Render the chatbot. Returns the newest request sent by the chatbot (or None)."""
    return _component(response=response, key=key, default=None)
