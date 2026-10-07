"""Real-model prediction service (no Streamlit imports, fully testable)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import config
from .compact_forest import CompactForest

log = logging.getLogger(__name__)


class InvalidInput(ValueError):
    """Raised with a user-friendly Arabic message when the form data is invalid."""


@dataclass
class PatientData:
    height_cm: float
    weight_kg: float
    age_category: int
    gen_hlth: int
    high_bp: int
    high_chol: int
    smoker: int
    stroke: int
    heart_disease: int
    phys_activity: int
    diff_walk: int

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "PatientData":
        """Same bounds as the original pydantic model in main.py."""

        def num(key: str, label: str, lo=None, hi=None, gt=None, integer=False):
            if key not in p or p[key] is None or p[key] == "":
                raise InvalidInput(f"البيانات ناقصة: {label}")
            try:
                v = float(p[key])
            except (TypeError, ValueError):
                raise InvalidInput(f"قيمة غير صحيحة: {label}")
            if v != v or v in (float("inf"), float("-inf")):
                raise InvalidInput(f"قيمة غير صحيحة: {label}")
            if gt is not None and not v > gt:
                raise InvalidInput(f"قيمة خارج النطاق المسموح: {label}")
            if lo is not None and v < lo:
                raise InvalidInput(f"قيمة خارج النطاق المسموح: {label}")
            if hi is not None and v > hi:
                raise InvalidInput(f"قيمة خارج النطاق المسموح: {label}")
            if integer:
                if v != int(v):
                    raise InvalidInput(f"قيمة غير صحيحة: {label}")
                return int(v)
            return v

        return cls(
            height_cm=num("height_cm", "الطول", gt=100, hi=250),
            weight_kg=num("weight_kg", "الوزن", gt=20, hi=300),
            age_category=num("age_category", "الفئة العمرية", 1, 13, integer=True),
            gen_hlth=num("gen_hlth", "الصحة العامة", 1, 5, integer=True),
            high_bp=num("high_bp", "ضغط الدم", 0, 1, integer=True),
            high_chol=num("high_chol", "الكوليسترول", 0, 1, integer=True),
            smoker=num("smoker", "التدخين", 0, 1, integer=True),
            stroke=num("stroke", "السكتة الدماغية", 0, 1, integer=True),
            heart_disease=num("heart_disease", "أمراض القلب", 0, 1, integer=True),
            phys_activity=num("phys_activity", "النشاط البدني", 0, 1, integer=True),
            diff_walk=num("diff_walk", "صعوبة المشي أو الحركة", 0, 1, integer=True),
        )


# ---------------------------------------------------------------- helpers
def bmi_category(bmi: float) -> str:
    for low, high, label in config.BMI_CATEGORIES:
        if low <= bmi < high:
            return label
    return "غير محدد"


def age_code_to_label(code: int) -> str:
    return config.AGE_LABELS.get(int(code), f"كود عمر غير معروف: {code}")


def yes_no(value: int) -> str:
    return "نعم" if int(value) == 1 else "لا"


def prediction_to_status(prediction: int) -> str:
    return config.STATUS_TEXT.get(int(prediction), config.STATUS_TEXT[2])


# ---------------------------------------------------------------- predictor
class Predictor:
    def __init__(self, model: CompactForest, metadata: dict[str, Any]):
        self.model = model
        self.meta = metadata
        self.feature_columns: list[str] = metadata["feature_columns"]
        self.accuracy: float = float(metadata["metrics"]["accuracy"])

    # -------- loading
    @classmethod
    def load(cls) -> "Predictor":
        meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        model = CompactForest.load(config.MODEL_PATH)
        if model.n_features != len(meta["feature_columns"]):
            raise RuntimeError("Model and metadata do not match; re-run train_model.py")
        return cls(model, meta)

    # -------- feature construction
    def _derived_onehots(self, row: dict[str, float]) -> dict[str, float]:
        """Rebuild the get_dummies columns exactly as they exist in the training data."""
        out: dict[str, float] = {}
        for label_col, spec in self.meta["derived_features"].items():
            src = row[spec["source"]]
            if spec["type"] == "map":
                label = spec["map"].get(str(int(src)))
                if label is None:
                    raise InvalidInput("قيمة غير مدعومة في بيانات النموذج.")
            else:  # thresholds
                idx = sum(src >= b for b in spec["bins"])
                label = spec["labels"][idx]
            out[f"{label_col}_{label}"] = 1.0
        return out

    def build_row(self, data: PatientData, bmi: float) -> pd.DataFrame:
        base: dict[str, float] = {
            "HighBP": float(data.high_bp),
            "HighChol": float(data.high_chol),
            "BMI": float(bmi),
            "Smoker": float(data.smoker),
            "Stroke": float(data.stroke),
            "HeartDiseaseorAttack": float(data.heart_disease),
            "PhysActivity": float(data.phys_activity),
            "GenHlth": float(data.gen_hlth),
            "DiffWalk": float(data.diff_walk),
            "Age": float(data.age_category),
            **config.DEFAULT_INPUTS,
        }
        row = dict(base)
        if config.COMPLETE_DERIVED_FEATURES:
            row.update(self._derived_onehots(base))
        # Exact training column order; anything not provided is 0 (as in main.py).
        frame = pd.DataFrame([row]).reindex(columns=self.feature_columns, fill_value=0)
        return frame.astype(float)

    # -------- prediction
    def predict(self, data: PatientData) -> dict[str, Any]:
        bmi = round(data.weight_kg / ((data.height_cm / 100) ** 2), 2)
        X = self.build_row(data, bmi)
        proba = self.model.predict_proba(X)[0]
        code = int(self.model.classes_[int(proba.argmax())])   # == RandomForestClassifier.predict
        probs = {str(int(c)): float(p) for c, p in zip(self.model.classes_, proba)}
        return {"bmi": bmi, "code": code, "probabilities": probs}

    def patient_context(self, data: PatientData, bmi: float, code: int) -> dict[str, Any]:
        return {
            "height_cm": round(data.height_cm, 1),
            "weight_kg": round(data.weight_kg, 1),
            "bmi": round(bmi, 1),
            "bmi_category": bmi_category(bmi),
            "age_category_code": data.age_category,
            "age_category": age_code_to_label(data.age_category),
            "general_health": config.GENERAL_HEALTH_LABELS[data.gen_hlth],
            "high_blood_pressure": yes_no(data.high_bp),
            "high_cholesterol": yes_no(data.high_chol),
            "smoker": yes_no(data.smoker),
            "stroke_history": yes_no(data.stroke),
            "heart_disease_history": yes_no(data.heart_disease),
            "physical_activity": yes_no(data.phys_activity),
            "difficulty_walking": yes_no(data.diff_walk),
            "ml_prediction_code": int(code),
            "ml_prediction": prediction_to_status(code),
            "model_accuracy_test_split": round(self.accuracy, 4),
        }
