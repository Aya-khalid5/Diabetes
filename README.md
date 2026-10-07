# Diabetes Prediction — Streamlit app

A three-class diabetes classifier (**No Diabetes / Prediabetes / Diabetes**) with an Arabic AI health
assistant. The original chatbot (HTML/CSS/JS) is embedded **unchanged** as the prediction screen and talks
to the real trained model.

## Features
- **Home** – overview, real dataset/model numbers, “Start Prediction” button
- **Diabetes Prediction** – the original chatbot; result comes from the trained model
- **Model Insights** – metrics, per-class results, confusion matrix, feature importance, ROC, class distribution
- **About** – problem, dataset, preprocessing, model, workflow, technologies

## Model
Random Forest (scikit-learn, 150 trees, `max_depth=None`, `class_weight="balanced_subsample"`), stratified 80/20 split,
`random_state=42` — identical to `diabetes.ipynb` / the old `main.py`. Trained on `Clean_diabetes_data.csv`
(253,680 rows, 45 features after one-hot encoding). Test accuracy 83.7%; see *Model Insights* for per-class results
(prediabetes is not recognised, diabetes recall is 18%). Not a medical tool.

**Deployment format.** The full scikit-learn forest has 13.8 M nodes (225 MB file, ~1.2 GB RAM), which exceeds
GitHub's 100 MB limit and is too heavy for Community Cloud. `train_model.py` therefore also writes a **lossless compact copy**
(`model/diabetes_rf_compact.npz`, ~21 MB, ~100 MB RAM): same trees, splits and leaf values, evaluated with numpy.
Before saving, the script compares it with scikit-learn on the whole test set and aborts if any prediction differs
(result: 0 of 50,736). The file is plain numpy, so there is no scikit-learn version/pickle problem on the server.

## Preprocessing note (differs from the old `main.py`)
The model was trained on 45 columns, 24 of which are one-hot versions of `BMI_Category`, `General_Health`, `Gender`,
`Age_Category`, `Education_Level`, `Income_Level`. The old backend left all 24 at 0 when predicting (a combination that never
occurs in training). The app now fills them exactly like the training data, using mappings read from the dataset
(`COMPLETE_DERIVED_FEATURES = True` in `core/config.py`; set `False` to reproduce the old behaviour).
Measured on 6,000 real held-out patients: old behaviour answered “No Diabetes” for 99.4% and caught 3% of true diabetes cases;
new behaviour flags 7% and catches 21% (matching the notebook's reported recall of 0.18).
The chatbot asks 11 questions; remaining inputs use the same fixed defaults as before (listed on the About page).

## Technologies
Python, pandas, NumPy, scikit-learn (training), Streamlit, HTML/CSS/JavaScript (chatbot), Google Gemini via `google-genai`.

## Project structure
```
app.py                     entry point (navigation + theme)
views/                     home.py · prediction.py · insights.py · about.py
components/chatbot/        ORIGINAL chatbot files + streamlit-bridge.js (see CHANGES.md)
core/                      predictor, bridge, assistant (Gemini), compact_forest, training, charts, theme, config
model/                     diabetes_rf_compact.npz, metadata.json (feature order, labels, metrics)
data/Clean_diabetes_data.csv
train_model.py             re-create model/ from the data
tests/                     unit tests + browser check
.streamlit/                config.toml, secrets.toml.example
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your Gemini key (optional)
streamlit run app.py
```
Without a Gemini key predictions still work; the assistant's explanation and follow-up chat show a notice instead.

To retrain (e.g. with your own library versions): `pip install -r requirements-train.txt && python train_model.py`.
Tests: `python -m unittest discover -s tests -v`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (do **not** commit `.streamlit/secrets.toml`; it is git-ignored).
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo and branch, main file `app.py`.
3. **Advanced settings → Secrets**: paste
   ```toml
   GEMINI_API_KEY = "your-key"
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   ```
4. Deploy.

## How the chatbot connects to the model
The chatbot runs in an isolated iframe (a Streamlit v1 custom component) so its CSS/JS behave exactly as before.
It still calls `fetch("/api/predict")` and `fetch("/api/chat")`; `streamlit-bridge.js` intercepts those two calls and sends
them to Python with the Streamlit component protocol. `core/bridge.py` answers with the same JSON the old FastAPI endpoints
returned: it validates the form, builds the 45-feature row, runs the trained model, asks Gemini for the explanation, and
returns the result, which the chatbot displays as before. The model is loaded once with `st.cache_resource`.
