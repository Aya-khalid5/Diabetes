"""Python half of the chatbot <-> model bridge (no Streamlit imports).

The original chatbot calls two endpoints:  POST /api/predict  and  POST /api/chat.
`handle_request` answers exactly those calls with the same JSON shapes that the
original FastAPI backend returned, so the chatbot's own JavaScript is untouched.

Return value: {"status": <http-like status>, "body": <json>}
Errors use {"detail": "<friendly message>"} - the field the chatbot already reads.
"""
from __future__ import annotations

import logging
from typing import Any

from .assistant import Assistant, ChatUnavailable
from .predictor import (
    InvalidInput,
    PatientData,
    Predictor,
    age_code_to_label,
    bmi_category,
    prediction_to_status,
)

log = logging.getLogger(__name__)

GENERIC_ERROR = "حدث خطأ غير متوقع. حاول مرة أخرى."
MODEL_ERROR = "تعذر إجراء التحليل حاليًا. حاول مرة أخرى بعد قليل."


def _error(status: int, detail: str) -> dict[str, Any]:
    return {"status": status, "body": {"detail": detail}}


def handle_predict(payload: dict, predictor: Predictor, assistant: Assistant) -> dict[str, Any]:
    try:
        data = PatientData.from_payload(payload)
    except InvalidInput as e:
        return _error(422, str(e))

    try:
        result = predictor.predict(data)
    except InvalidInput as e:
        return _error(422, str(e))
    except Exception:
        log.exception("prediction failed")
        return _error(500, MODEL_ERROR)

    bmi, code = result["bmi"], result["code"]
    context = predictor.patient_context(data, bmi, code)
    advice = assistant.initial_advice(context)

    return {
        "status": 200,
        "body": {
            "success": True,
            "bmi": bmi,
            "bmi_category": bmi_category(bmi),
            "age_label": age_code_to_label(data.age_category),
            "prediction_code": code,
            "status": prediction_to_status(code),
            "ai_advice": advice,
            "patient_context": context,
            "model_accuracy": round(predictor.accuracy, 4),
            # extra (ignored by the chatbot UI; used by the Streamlit result panel)
            "probabilities": result["probabilities"],
        },
    }


def handle_chat(payload: dict, assistant: Assistant) -> dict[str, Any]:
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return _error(422, "الرسالة فارغة.")
    if len(message) > 2000:
        return _error(422, "الرسالة طويلة جدًا (الحد الأقصى 2000 حرف).")
    context = payload.get("patient_context")
    if not isinstance(context, dict):
        return _error(422, "بيانات التحليل غير موجودة. أجرِ التحليل أولًا.")
    conversation = payload.get("conversation") or []
    if not isinstance(conversation, list):
        conversation = []

    try:
        answer = assistant.chat(context, conversation, message)
    except ChatUnavailable as e:
        return _error(e.status, e.message)
    except Exception:
        log.exception("chat failed")
        return _error(500, GENERIC_ERROR)
    return {"status": 200, "body": {"success": True, "answer": answer}}


def handle_request(req: dict, predictor: Predictor, assistant: Assistant) -> dict[str, Any]:
    """Dispatch one request coming from the chatbot iframe."""
    route = (req or {}).get("route")
    payload = (req or {}).get("payload") or {}
    if not isinstance(payload, dict):
        return _error(422, GENERIC_ERROR)
    if route == "/api/predict":
        return handle_predict(payload, predictor, assistant)
    if route == "/api/chat":
        return handle_chat(payload, assistant)
    return _error(404, "Not found")
