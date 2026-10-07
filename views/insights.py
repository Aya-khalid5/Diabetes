import streamlit as st

from core import charts, services, theme

try:
    meta = services.get_predictor().meta
except Exception:
    st.error("The model information could not be loaded. Please try again later.")
    st.stop()

ds, sp, mo, m = meta["dataset"], meta["split"], meta["model"], meta["metrics"]
rep, labels = m["classification_report"], meta["class_labels"]
classes = [str(c) for c in meta["classes"]]
color = theme.CLASS_COLORS
names = [labels[c] for c in classes]

theme.page_header(
    "Model Insights",
    "How the model was built and how it performs on data it never saw during training. Every number on this page was computed from the project's own data and model.",
)

# ------------------------------------------------------------------ Model
st.subheader("Model")
p = mo["params"]
theme.stats(
    [
        ("Algorithm", "Random Forest", "scikit-learn RandomForestClassifier"),
        ("Trees", str(p["n_estimators"]), f"max_depth = {p['max_depth']}  (fully grown)"),
        ("Class weighting", str(p["class_weight"]), "compensates for class imbalance"),
        ("Features", str(ds["model_features"]), f"{ds['input_columns']} columns after one-hot encoding"),
        ("Training / test rows", f"{sp['train_rows']:,} / {sp['test_rows']:,}", f"stratified split, seed {sp['random_state']}"),
    ]
)
st.markdown(
    f'<p class="small">Target: <code>{theme.e(ds["target"])}</code> — '
    + ", ".join(f"{c} = {theme.e(labels[c])}" for c in classes)
    + ". Training approach: one-hot encode categorical columns, stratified 80/20 split, fit the forest, evaluate on the "
    "20% hold-out. No scaling or imputation is used (tree-based model).</p>",
    unsafe_allow_html=True,
)
ver = meta["compact_verification"]
st.markdown(
    f'<p class="small">Deployment note: the app runs a lossless compact copy of the trained forest (same trees, splits and leaf values). '
    f'It was checked against the original scikit-learn model on all {ver["rows_checked"]:,} test rows: '
    f'{ver["label_mismatches"]} predictions differ.</p>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ Performance
st.subheader("Performance (held-out test set)")
macro, weighted = rep["macro avg"], rep["weighted avg"]
theme.stats(
    [
        ("Accuracy", theme.pct(m["accuracy"]), "all classes combined"),
        ("Precision (macro)", f"{macro['precision']:.2f}", f"weighted {weighted['precision']:.2f}"),
        ("Recall (macro)", f"{macro['recall']:.2f}", f"weighted {weighted['recall']:.2f}"),
        ("F1-score (macro)", f"{macro['f1-score']:.2f}", f"weighted {weighted['f1-score']:.2f}"),
        ("ROC-AUC (macro OvR)", f"{m['roc_auc_ovr_macro']:.3f}", "from predicted probabilities"),
    ]
)
rows = [
    (labels[c], color[c], f"{rep[c]['precision']:.2f}", f"{rep[c]['recall']:.2f}", f"{rep[c]['f1-score']:.2f}", f"{int(rep[c]['support']):,}")
    for c in classes
]
theme.card("Per-class results", charts.metrics_table(rows), kicker="Precision · Recall · F1")

pre = next((c for c in classes if labels[c].lower().startswith("pre")), None)
dia = classes[-1]
notes = [
    f"<li><b>Accuracy flatters this model.</b> {ds['class_counts']['0'] / ds['rows'] * 100:.0f}% of all records are "
    f"“{labels['0']}”, and the model is very good at that class (recall {rep['0']['recall']:.2f}). "
    "A model that mostly says “No Diabetes” already scores high accuracy.</li>",
    f"<li><b>“{labels[dia]}” is only partly detected:</b> when the model says {labels[dia]} it is right "
    f"{rep[dia]['precision'] * 100:.0f}% of the time, but it finds just {rep[dia]['recall'] * 100:.0f}% of the real cases.</li>",
]
if pre:
    notes.append(
        f"<li><b>“{labels[pre]}” is not recognised:</b> only {sp['class_counts_test'][pre]:,} of the "
        f"{sp['test_rows']:,} test rows belong to it, and the model's recall for it is {rep[pre]['recall']:.2f}.</li>"
    )
theme.card("How to read these numbers", "<ul>" + "".join(notes) + "</ul>", kicker="Important context")

# ------------------------------------------------------------------ Visualisations
st.subheader("Visualizations")

theme.card(
    "Confusion matrix",
    '<p class="small">Each row is the true class; cells show how the model classified those people. '
    "Outlined cells on the diagonal are correct predictions.</p>"
    + charts.confusion_matrix_html(m["confusion_matrix"], names),
    kicker="Test set",
)

left, right = st.columns(2)
with left:
    fi = m["feature_importance"][:12]
    mx = fi[0]["importance"]
    theme.card(
        "Feature importance (top 12)",
        '<p class="small">Mean decrease in impurity across all trees.</p>'
        + charts.bar_rows([(d["feature"], d["importance"], theme.INDIGO, f"{d['importance'] * 100:.1f}%") for d in fi], max_value=mx),
        kicker="What the model relies on",
    )
with right:
    cc = ds["class_counts"]
    theme.card(
        "Class distribution (full dataset)",
        f'<p class="small">{ds["rows"]:,} records.</p>'
        + charts.bar_rows([(labels[c], cc[c], color[c], f"{cc[c]:,} ({cc[c] / ds['rows'] * 100:.1f}%)") for c in classes], max_value=max(cc.values())),
        kicker="Imbalance",
    )

curves = [
    (labels[c], m["roc"][c]["fpr"], m["roc"][c]["tpr"], m["roc"][c]["auc"], color[c])
    for c in classes
]
theme.card(
    "ROC curves (one-vs-rest)",
    '<p class="small">Each curve treats one class as positive and the rest as negative. The dashed line is chance.</p>'
    + charts.roc_svg(curves),
    kicker="Ranking quality",
)

top_unasked = [
    d for d in m["feature_importance"][:8]
    if d["feature"] in {"PhysHlth_Days", "MentHlth_Days", "Income", "Fruits", "Education", "Veggies", "Sex"}
]
if top_unasked:
    theme.card(
        "Inputs the assistant does not ask for",
        "<p>The chatbot collects 11 answers. Several features that rank high above — "
        + ", ".join(f"<b>{theme.e(d['feature'])}</b>" for d in top_unasked)
        + " — are not asked, so the app fills them with the same fixed defaults the original backend used "
        "(see About). Results therefore reflect the 11 answers plus those defaults.</p>",
        kicker="Limitation",
    )
