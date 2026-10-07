"""End-to-end browser check of the chatbot <-> Python bridge (developer tool).

Streamlit itself is NOT started here. Instead a tiny host page implements the
Streamlit side of the documented v1 component protocol (componentReady /
setFrameHeight / setComponentValue  ->  streamlit:render with new args) and answers
requests with the REAL core.bridge.handle_request and the REAL trained model.
The chatbot files that run in the browser are the real ones from components/chatbot.

    pip install playwright && playwright install chromium
    python tests/browser_check.py [path-to-original-chatbot-folder]

If the second argument is given, the original standalone chatbot is screenshotted and
compared pixel-by-pixel with the embedded one.
"""
from __future__ import annotations

import json
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.assistant import Assistant
from core.bridge import handle_request
from core.predictor import PatientData, Predictor

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
ORIGINAL = Path(sys.argv[1]) if len(sys.argv) > 1 else None

HOST = """<!doctype html><meta charset=utf-8><body style="margin:0;background:#F6F5FD">
<iframe id=f src="/component/index.html" style="width:100%;border:0;height:0" scrolling="no"></iframe>
<script>
const f=document.getElementById('f'); window.__log=[];
function render(response){f.contentWindow.postMessage({type:'streamlit:render',args:{response:response},disabled:false,theme:{}},'*');}
window.addEventListener('message',async e=>{
  if(e.source!==f.contentWindow) return; const d=e.data; if(!d||!d.isStreamlitMessage) return;
  window.__log.push(d.type);
  if(d.type==='streamlit:componentReady') render(null);
  else if(d.type==='streamlit:setFrameHeight') f.style.height=d.height+'px';
  else if(d.type==='streamlit:setComponentValue'){
    const r=await fetch('/rpc',{method:'POST',body:JSON.stringify(d.value)}); const resp=await r.json();
    render(Object.assign({id:d.value.id},resp));
  }
});
</script>"""

PRED = Predictor.load()


class FakeGemini:
    class models:
        @staticmethod
        def generate_content(model, contents):
            class R:
                pass
            r = R()
            r.text = "ANSWER<<" + contents.strip().splitlines()[-1] + ">>"
            return r


ASSISTANT = Assistant(client=FakeGemini)
CALLS: list[dict] = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/host":
            return self._send(HOST.encode(), "text/html; charset=utf-8")
        for prefix, base in (("/component/", ROOT / "components" / "chatbot"), ("/original/", ORIGINAL)):
            if base and self.path.startswith(prefix):
                name = self.path[len(prefix):].split("?")[0] or "index.html"
                f = base / name
                if f.is_file():
                    return self._send(f.read_bytes(), (mimetypes.guess_type(name)[0] or "text/plain") + "; charset=utf-8")
        self._send(b"nf", "text/plain", 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CALLS.append(req)
        self._send(json.dumps(handle_request(req, PRED, ASSISTANT)).encode(), "application/json")


def main() -> int:
    from playwright.sync_api import sync_playwright

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    failures: list[str] = []

    def check(cond, msg):
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            failures.append(msg)

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.on("pageerror", lambda e: failures.append(f"page error: {e}"))
        pg.goto(base + "/host")
        fr = pg.frame_locator("#f")
        fr.locator("#analyzeBtn").wait_for(timeout=10000)
        pg.wait_for_timeout(400)

        # ---- 1. handshake + sizing
        log = pg.evaluate("window.__log")
        check(log and log[0] == "streamlit:componentReady", "componentReady is the first message")
        h0 = pg.evaluate("document.getElementById('f').offsetHeight")
        check(h0 > 300, f"iframe auto-sized from content (height={h0}px)")

        # ---- 2. visual identity with the original standalone chatbot
        if ORIGINAL:
            emb_h = pg.evaluate("document.getElementById('f').offsetHeight")
            pg.screenshot(path="/tmp/embedded.png", clip={"x": 0, "y": 0, "width": 1280, "height": min(emb_h, 880)})
            pg2 = b.new_page(viewport={"width": 1280, "height": 900})
            # block the dead localhost:8000 default; page is only screenshotted, not used
            pg2.goto(base + "/original/index.html")
            pg2.wait_for_timeout(400)
            pg2.screenshot(path="/tmp/original.png", clip={"x": 0, "y": 0, "width": 1280, "height": min(emb_h, 880)})
            from PIL import Image, ImageChops

            a, c = Image.open("/tmp/embedded.png").convert("RGB"), Image.open("/tmp/original.png").convert("RGB")
            diff = ImageChops.difference(a, c).getbbox()
            check(diff is None, "embedded chatbot is PIXEL-IDENTICAL to the original (1280px, initial state)")
            pg2.close()

        # ---- 3. validation still done by the chatbot itself
        fr.locator("#analyzeBtn").click()
        check("طولًا" in fr.locator("#formError").inner_text(), "empty form -> chatbot's own Arabic validation message")
        check(len(CALLS) == 0, "no request reaches Python for an invalid form")

        # ---- 4. real prediction through the bridge
        vals = dict(height=172, weight=95, ageCategory="9", genHlth="4", highBp="1", highChol="1", smoker="0",
                    stroke="0", heartDisease="0", physActivity="0", diffWalk="1")
        fr.locator("#height").fill(str(vals["height"]))
        fr.locator("#weight").fill(str(vals["weight"]))
        for k in ("ageCategory", "genHlth", "highBp", "highChol", "smoker", "stroke", "heartDisease", "physActivity", "diffWalk"):
            fr.locator("#" + k).select_option(vals[k])
        fr.locator("#analyzeBtn").click()
        fr.locator("#resultSection:not(.hidden)").wait_for(timeout=15000)
        expected = PRED.predict(PatientData(172, 95, 9, 4, 1, 1, 0, 0, 0, 0, 1))
        from core import config

        shown = fr.locator("#resultStatus").inner_text()
        check(shown == config.STATUS_TEXT[expected["code"]], f"result shown in chatbot == real model output ({expected['code']}: {shown})")
        check(fr.locator("#resultBmi").inner_text() == str(expected["bmi"]), f"BMI shown = {expected['bmi']}")
        check(fr.locator("#aiAdvice").inner_text().startswith("ANSWER<<"), "AI explanation comes back from Python (fake Gemini)")
        check(CALLS[-1]["route"] == "/api/predict" and CALLS[-1]["payload"]["weight_kg"] == 95, "request payload = chatbot form values")
        check(fr.locator("#chatSection:not(.hidden)").count() == 1, "chat section revealed after prediction")
        h1 = pg.evaluate("document.getElementById('f').offsetHeight")
        check(h1 > h0 + 200, f"iframe grew when results appeared ({h0}px -> {h1}px)")
        check(fr.locator("#analyzeBtn").is_enabled() and "تحليل" in fr.locator("#analyzeBtn").inner_text(), "analyze button restored")

        # ---- 5. follow-up chat, history, ordering
        fr.locator("#chatInput").fill("ما معنى الكوليسترول؟")
        fr.locator("#sendChatBtn").click()
        fr.locator(".message.assistant .bubble").nth(1).wait_for(timeout=15000)
        check("ما معنى الكوليسترول؟" in fr.locator(".message.assistant .bubble").nth(1).inner_text(), "chat answer rendered as assistant bubble")
        chat_call = [c for c in CALLS if c["route"] == "/api/chat"][-1]
        check(chat_call["payload"]["patient_context"]["ml_prediction_code"] == expected["code"], "chat request carries the patient context from the prediction")
        # two quick questions at once -> both answered, in order
        fr.locator(".quick-questions button").nth(0).click()
        fr.locator(".quick-questions button").nth(3).click()
        fr.locator(".message.assistant .bubble").nth(3).wait_for(timeout=15000)
        bubbles = fr.locator(".message.assistant .bubble").all_inner_texts()
        check(len(bubbles) == 4 and "لماذا ظهرت لي هذه النتيجة" in bubbles[2] and "BMI" in bubbles[3], "two rapid questions answered in order")
        hist = [c for c in CALLS if c["route"] == "/api/chat"][-1]["payload"]["conversation"]
        check(len(hist) >= 5 and hist[0]["role"] == "user", f"conversation history sent ({len(hist)} turns)")

        # ---- 6. backend validation error surfaces cleanly (height exactly 100 is rejected by the original backend)
        fr.locator("#height").fill("100")
        fr.locator("#analyzeBtn").click()
        fr.locator("#formError:not(.hidden)").wait_for(timeout=10000)
        err = fr.locator("#formError").inner_text()
        check("FastAPI" not in err and "127.0.0.1" not in err and len(err) > 3, f"error text is clean: «{err}»")

        # ---- 7. narrow (tablet / phone) widths keep working
        for w in (820, 390):
            pg.set_viewport_size({"width": w, "height": 900})
            pg.wait_for_timeout(300)
            sw = pg.evaluate("document.getElementById('f').contentDocument.documentElement.scrollWidth")
            check(sw <= w + 1, f"no horizontal overflow inside iframe at {w}px (scrollWidth={sw})")
        pg.set_viewport_size({"width": 1280, "height": 900})
        pg.wait_for_timeout(300)
        pg.screenshot(path="/tmp/embedded_result.png", full_page=True)
        b.close()

    print("\nFAILURES:" if failures else "\nALL BROWSER CHECKS PASSED", *failures, sep="\n  ")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
