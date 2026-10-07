"""Central configuration.

Everything here is copied from the original project (main.py / diabetes.ipynb).
Nothing in this file is a new modelling decision.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_PATH = ROOT / "data" / "Clean_diabetes_data.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "diabetes_rf_compact.npz"
METADATA_PATH = MODEL_DIR / "metadata.json"
CHATBOT_DIR = ROOT / "components" / "chatbot"

# --------------------------------------------------------------------------
# Training (identical to main.py and diabetes.ipynb)
# --------------------------------------------------------------------------
TARGET_COLUMNS = ["Diabetes_012", "Diabetes_Status"]
TARGET = "Diabetes_012"
TEST_SIZE = 0.20
RANDOM_STATE = 42
RF_PARAMS = dict(
    n_estimators=150,
    random_state=42,
    max_depth=None,
    class_weight="balanced_subsample",
    n_jobs=-1,
)

# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
# The chatbot only asks 11 questions. The remaining model inputs use the fixed
# defaults the original backend (main.py -> make_model_input) already used.
# NOTE: main.py named two of these "MentHlth" / "PhysHlth", but the trained
# model's columns are "MentHlth_Days" / "PhysHlth_Days"; the values (0) are the same.
DEFAULT_INPUTS = {
    "CholCheck": 1.0,
    "Fruits": 1.0,
    "Veggies": 1.0,
    "HvyAlcoholConsump": 0.0,
    "AnyHealthcare": 1.0,
    "NoDocbcCost": 0.0,
    "MentHlth_Days": 0.0,
    "PhysHlth_Days": 0.0,
    "Sex": 0.0,
    "Education": 5.0,
    "Income": 5.0,
}

# True  -> also fill the one-hot "derived category" columns (BMI_Category_*,
#          General_Health_*, Gender_*, Age_Category_*, Education_Level_*,
#          Income_Level_*) exactly the way the training data had them.
# False -> reproduce the ORIGINAL main.py behaviour, which left all 24 of those
#          columns at 0 (a combination that never occurs in the training data).
# See README ("Preprocessing note") for the measured difference.
COMPLETE_DERIVED_FEATURES = True

# --------------------------------------------------------------------------
# Arabic labels (verbatim from main.py)
# --------------------------------------------------------------------------
AGE_LABELS = {
    1: "18–24 سنة",
    2: "25–29 سنة",
    3: "30–34 سنة",
    4: "35–39 سنة",
    5: "40–44 سنة",
    6: "45–49 سنة",
    7: "50–54 سنة",
    8: "55–59 سنة",
    9: "60–64 سنة",
    10: "65–69 سنة",
    11: "70–74 سنة",
    12: "75–79 سنة",
    13: "80 سنة فأكثر",
}

GENERAL_HEALTH_LABELS = {
    1: "ممتازة",
    2: "جيدة جدًا",
    3: "جيدة",
    4: "مقبولة",
    5: "ضعيفة",
}

BMI_CATEGORIES = [
    (0, 18.5, "نقص في الوزن"),
    (18.5, 25, "وزن طبيعي"),
    (25, 30, "زيادة في الوزن"),
    (30, 35, "سمنة من الدرجة الأولى"),
    (35, 40, "سمنة من الدرجة الثانية"),
    (40, float("inf"), "سمنة شديدة"),
]

STATUS_TEXT = {
    0: "حالة سليمة / لا توجد مؤشرات مرتفعة ظاهرة في بيانات النموذج",
    1: "مؤشرات مرحلة ما قبل السكري (Prediabetes)",
    2: "مؤشرات إصابة بالسكري (Diabetes)",
}
