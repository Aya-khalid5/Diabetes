import streamlit as st

from core import charts, services, theme
from core.bridge import handle_request
from core.chatbot_component import chatbot

ss = st.session_state

try:
    predictor = services.get_predictor()
    assistant = services.get_assistant()
except Exception:
    st.error("The prediction model could not be loaded right now. Please try again later.")
    st.stop()

theme.page_header(
    "Diabetes Prediction",
    "Answer the assistant below. Your answers are sent to the trained model and the result appears in the assistant.",
)

# ---------------------------------------------------------------------------
# The ORIGINAL chatbot, embedded unchanged. It returns the newest request it made
# (id / route / payload); we answer it with the real model and hand the answer back
# through the `response` argument on the next render.
# ---------------------------------------------------------------------------
request = chatbot(response=ss.get("chatbot_response"))

if isinstance(request, dict) and request.get("id") and request["id"] != ss.get("chatbot_last_id"):
    ss["chatbot_last_id"] = request["id"]
    result = handle_request(request, predictor, assistant)
    ss["chatbot_response"] = {"id": request["id"], **result}
    if request.get("route") == "/api/predict" and result["status"] == 200:
        ss["last_prediction"] = result["body"]
    st.rerun()

# ---------------------------------------------------------------------------
# Read-only summary of what the model returned (not a second form).
# ---------------------------------------------------------------------------
pred = ss.get("last_prediction")
if pred:
    names = predictor.meta["class_labels"]
    probs = pred["probabilities"]
    code = str(pred["prediction_code"])
    order = sorted(probs, key=lambda c: int(c))
    rows = [
        (names[c], probs[c], theme.CLASS_COLORS.get(c, theme.INDIGO), theme.pct(probs[c]))
        for c in order
    ]
    top = order.index(code)
    badge = f'<span class="badge" style="background:{theme.CLASS_COLORS.get(code, theme.INDIGO)}">{theme.e(names[code])}</span>'
    st.markdown(
        '<div class="card"><div class="kicker">Model output</div>'
        f"<h3>Predicted class: {badge}</h3>"
        + charts.bar_rows(rows, max_value=1.0, top_index=top)
        + '<p class="small">These are the Random Forest’s class probabilities (the share of its 150 trees voting '
        "for each class). They describe the model’s vote, not a calibrated personal risk percentage, and this tool "
        "is not a medical diagnosis.</p></div>",
        unsafe_allow_html=True,
    )
