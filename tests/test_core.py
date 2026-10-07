"""Tests for everything that does not need Streamlit. Run:  python -m unittest discover -s tests -v"""
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from core import config
from core.assistant import Assistant, ChatUnavailable, build_chat_prompt, build_initial_prompt
from core.bridge import handle_request
from core.predictor import PatientData, Predictor

GOOD = dict(height_cm=170, weight_kg=70, age_category=7, gen_hlth=3, high_bp=1, high_chol=0,
            smoker=0, stroke=0, heart_disease=0, phys_activity=1, diff_walk=0)


class FakeGemini:
    class models:
        @staticmethod
        def generate_content(model, contents):
            class R: text = "رد تجريبي"
            FakeGemini.last = (model, contents)
            return R


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = Predictor.load()
        cls.A = Assistant(client=FakeGemini)
        cls.off = Assistant("")

    # ---------- model + features
    def test_feature_order_matches_training(self):
        row = self.P.build_row(PatientData.from_payload(GOOD), 24.22)
        self.assertEqual(list(row.columns), self.P.feature_columns)
        self.assertEqual(row.shape, (1, 45))

    def test_exactly_one_hot_per_group(self):
        row = self.P.build_row(PatientData.from_payload(GOOD), 24.22).iloc[0]
        for prefix in ("BMI_Category_", "General_Health_", "Gender_", "Age_Category_", "Education_Level_", "Income_Level_"):
            cols = [c for c in row.index if c.startswith(prefix)]
            self.assertEqual(row[cols].sum(), 1.0, prefix)

    def test_known_derivations(self):
        row = self.P.build_row(PatientData.from_payload({**GOOD, "age_category": 11, "gen_hlth": 1}), 31.0).iloc[0]
        self.assertEqual(row["BMI_Category_Obese"], 1.0)
        self.assertEqual(row["Age_Category_Older Adults"], 1.0)
        self.assertEqual(row["General_Health_Excellent"], 1.0)
        self.assertEqual(row["Gender_Female"], 1.0)          # Sex default 0 == Female in the data
        self.assertEqual(row["Education_Level_Some College"], 1.0)   # Education default 5
        self.assertEqual(row["Income_Level_Middle Income"], 1.0)     # Income default 5

    def test_bmi_thresholds(self):
        for bmi, col in [(18.4, "Underweight"), (18.5, "Normal"), (24.99, "Normal"), (25.0, "Overweight"), (29.9, "Overweight"), (30.0, "Obese")]:
            row = self.P.build_row(PatientData.from_payload(GOOD), bmi).iloc[0]
            self.assertEqual(row[f"BMI_Category_{col}"], 1.0, bmi)

    def test_probabilities_are_real_and_consistent(self):
        r = self.P.predict(PatientData.from_payload(GOOD))
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=9)
        self.assertEqual(r["code"], int(max(r["probabilities"], key=r["probabilities"].get)))

    def test_prediction_depends_on_inputs(self):
        lo = self.P.predict(PatientData.from_payload({**GOOD, "age_category": 2, "gen_hlth": 1, "high_bp": 0, "phys_activity": 1}))
        hi = self.P.predict(PatientData.from_payload({**GOOD, "weight_kg": 140, "age_category": 12, "gen_hlth": 5, "high_bp": 1, "high_chol": 1, "diff_walk": 1}))
        self.assertGreater(hi["probabilities"]["2"], lo["probabilities"]["2"])

    def test_metadata_matches_model(self):
        meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(meta["compact_verification"]["label_mismatches"], 0)
        self.assertEqual(self.P.model.n_features, len(meta["feature_columns"]))
        self.assertTrue(0.5 < meta["metrics"]["accuracy"] < 1)

    # ---------- bridge (same contract as the old FastAPI endpoints)
    def test_predict_endpoint_shape(self):
        out = handle_request({"route": "/api/predict", "payload": GOOD}, self.P, self.A)
        self.assertEqual(out["status"], 200)
        body = out["body"]
        for k in ("success", "bmi", "bmi_category", "age_label", "prediction_code", "status", "ai_advice", "patient_context", "model_accuracy"):
            self.assertIn(k, body)
        self.assertEqual(body["bmi"], round(70 / 1.7 ** 2, 2))
        self.assertEqual(body["status"], config.STATUS_TEXT[body["prediction_code"]])
        self.assertEqual(body["ai_advice"], "رد تجريبي")
        self.assertEqual(body["patient_context"]["ml_prediction_code"], body["prediction_code"])

    def test_predict_without_gemini_still_predicts(self):
        out = handle_request({"route": "/api/predict", "payload": GOOD}, self.P, self.off)
        self.assertEqual(out["status"], 200)
        self.assertIn("GEMINI_API_KEY", out["body"]["ai_advice"])

    def test_validation_messages(self):
        for bad in ({**GOOD, "height_cm": 90}, {**GOOD, "weight_kg": 10}, {**GOOD, "age_category": 14},
                    {**GOOD, "smoker": 2}, {k: v for k, v in GOOD.items() if k != "stroke"}, {**GOOD, "height_cm": "abc"}):
            out = handle_request({"route": "/api/predict", "payload": bad}, self.P, self.A)
            self.assertEqual(out["status"], 422)
            self.assertIsInstance(out["body"]["detail"], str)

    def test_model_failure_is_friendly(self):
        class Broken:
            accuracy = 0.8
            def predict(self, d): raise RuntimeError("secret internal detail")
        out = handle_request({"route": "/api/predict", "payload": GOOD}, Broken(), self.A)
        self.assertEqual(out["status"], 500)
        self.assertNotIn("secret", out["body"]["detail"])

    def test_chat_endpoint(self):
        ctx = handle_request({"route": "/api/predict", "payload": GOOD}, self.P, self.A)["body"]["patient_context"]
        out = handle_request({"route": "/api/chat", "payload": {"message": "لماذا؟", "patient_context": ctx,
                              "conversation": [{"role": "user", "content": "لماذا؟"}]}}, self.P, self.A)
        self.assertEqual(out, {"status": 200, "body": {"success": True, "answer": "رد تجريبي"}})
        self.assertIn("لماذا؟", FakeGemini.last[1])

    def test_chat_without_key_and_bad_input(self):
        out = handle_request({"route": "/api/chat", "payload": {"message": "hi", "patient_context": {}}}, self.P, self.off)
        self.assertEqual(out["status"], 503)
        self.assertEqual(handle_request({"route": "/api/chat", "payload": {"message": " ", "patient_context": {}}}, self.P, self.A)["status"], 422)
        self.assertEqual(handle_request({"route": "/nope", "payload": {}}, self.P, self.A)["status"], 404)

    def test_chat_gemini_error_hides_details(self):
        class Boom:
            class models:
                @staticmethod
                def generate_content(**k): raise RuntimeError("API key AIza-SECRET invalid")
        out = handle_request({"route": "/api/chat", "payload": {"message": "x", "patient_context": {}}}, self.P, Assistant(client=Boom))
        self.assertEqual(out["status"], 500)
        self.assertNotIn("SECRET", out["body"]["detail"])

    # ---------- prompts are the original ones
    def test_prompts_identical_to_original_main_py(self):
        orig = Path("/home/claude/orig/chatbot/chatbot/main.py")
        if not orig.exists():
            self.skipTest("original main.py not available")
        src = orig.read_text(encoding="utf-8")
        ctx = {"a": 1}
        # initial prompt text
        i = src.index('prompt = f"""') + len('prompt = f"""'); j = src.index('"""', i)
        self.assertEqual(build_initial_prompt(ctx), eval('f"""' + src[i:j] + '"""', {"context": ctx}))
        i = src.index('return f"""', src.index("def build_chat_prompt")) + len('return f"""'); j = src.index('"""', i)
        conv = [{"role": "user", "content": "س"}, {"role": "assistant", "content": "ج"}]
        history_text = "\n".join(f"{'المريض' if m['role']=='user' else 'المساعد'}: {m['content']}" for m in conv)
        expected = eval('f"""' + src[i:j] + '"""', {"patient_context": ctx, "history_text": history_text, "user_message": "سؤال"})
        self.assertEqual(build_chat_prompt(ctx, conv, "سؤال"), expected)


class ProtectedChatbot(unittest.TestCase):
    D = ROOT / "components" / "chatbot"
    O = Path("/home/claude/orig/chatbot/chatbot")

    def test_style_css_untouched(self):
        if not self.O.exists():
            self.skipTest("original not available")
        self.assertEqual(hashlib.sha256((self.D / "style.css").read_bytes()).hexdigest(),
                         hashlib.sha256((self.O / "style.css").read_bytes()).hexdigest())

    def test_html_only_gained_one_script_tag(self):
        if not self.O.exists():
            self.skipTest("original not available")
        a = (self.O / "index.html").read_text(encoding="utf-8").splitlines()
        b = (self.D / "index.html").read_text(encoding="utf-8").splitlines()
        self.assertEqual([l for l in b if l not in a], ['<script src="streamlit-bridge.js"></script>'])
        self.assertEqual([l for l in a if l not in b], [])

    def test_no_chatbot_logic_removed_from_script(self):
        if not self.O.exists():
            self.skipTest("original not available")
        a = (self.O / "script.js").read_text(encoding="utf-8").splitlines()
        b = (self.D / "script.js").read_text(encoding="utf-8").splitlines()
        removed = [l.strip() for l in a if l not in b]
        self.assertEqual(removed, ['const API_BASE = "http://127.0.0.1:8000";', 'showError(',
                                   '`${error.message} — تأكدي أن FastAPI يعمل على ${API_BASE}`', ');'])


class NoLeftovers(unittest.TestCase):
    def test_no_vercel_or_secrets_in_repo(self):
        pat = re.compile(r"vercel|NEXT_PUBLIC_API_URL|AIza[0-9A-Za-z_\-]{20,}", re.I)
        for f in ROOT.rglob("*"):
            if f.is_file() and f.suffix in {".py", ".js", ".html", ".css", ".txt", ".toml", ".md", ".json"} and "tests" not in f.parts and f.name != "metadata.json":
                text = f.read_text(encoding="utf-8", errors="ignore")
                hits = pat.findall(text)
                # README may say "no Vercel"; everything else must be clean
                if f.name == "README.md":
                    hits = [h for h in hits if h.lower() != "vercel"]
                self.assertFalse(hits, f"{f}: {hits}")


if __name__ == "__main__":
    unittest.main()
