import streamlit as st

from core import services, theme

try:
    meta = services.get_predictor().meta
except Exception:
    st.error("The model files could not be loaded. Please try again later.")
    st.stop()

ds, m = meta["dataset"], meta["metrics"]
rep = m["classification_report"]
total = ds["rows"]
no_diab_share = ds["class_counts"]["0"] / total

theme.hero(
    "Diabetes Prediction",
    "A machine-learning model that estimates diabetes status from everyday health indicators, "
    "paired with an AI health assistant that explains the result in plain language.",
    eyebrow="Better Data • Healthier Lives",
)

if st.button("Start Prediction", type="primary"):
    st.switch_page("views/prediction.py")

st.write("")
theme.stats(
    [
        ("Records", f"{total:,}", f"{ds['file']}"),
        ("Model inputs", f"{ds['model_features']}", f"from {ds['input_columns']} dataset columns"),
        ("Model", "Random Forest", f"{meta['model']['params']['n_estimators']} decision trees"),
        ("Test accuracy", theme.pct(m["accuracy"]), "stratified 20% hold-out"),
        ("ROC-AUC", f"{m['roc_auc_ovr_macro']:.2f}", "macro, one-vs-rest"),
    ]
)

c1, c2 = st.columns(2)
with c1:
    theme.card(
        "The problem",
        "<p>Diabetes is often detected late. Many of the strongest risk signals — body-mass index, age, blood "
        "pressure, cholesterol, general health, activity — are things people already know about themselves. "
        "This project explores how well a model can recognise diabetes and pre-diabetes from exactly those "
        "kinds of indicators.</p>",
        kicker="Why it matters",
    )
with c2:
    theme.card(
        "What the app does",
        "<p>You answer a short guided form in the assistant. Your answers are converted into the exact feature "
        "format the model was trained on, the trained Random Forest returns a prediction, and the assistant "
        "explains it and answers follow-up questions about your result.</p>",
        kicker="How it works",
    )

theme.card(
    "Please read this first",
    "<ul>"
    "<li>This is a <b>prediction and learning tool, not a medical diagnosis</b>. Do not start or stop any "
    "treatment based on it.</li>"
    f"<li>About {no_diab_share * 100:.0f}% of the records are “No Diabetes”, so the headline accuracy "
    "looks high partly because of that imbalance. The <b>Model Insights</b> page shows per-class results, "
    "including where the model is weak.</li>"
    "</ul>",
    kicker="Honest context",
)
