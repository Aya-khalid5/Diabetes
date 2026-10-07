# Changes made to the original chatbot files

`style.css` — **unchanged** (byte-identical).

`index.html` — one line added before `script.js`:
```html
<script src="streamlit-bridge.js"></script>
```

`script.js` — two small edits:
1. `const API_BASE = "http://127.0.0.1:8000";` → `const API_BASE = "";` (no FastAPI host any more).
2. The error handler no longer appends "make sure FastAPI is running on …" (not true on Streamlit);
   it shows `error.message` only.

`streamlit-bridge.js` — new. Answers the chatbot's two existing `fetch("/api/predict")` and
`fetch("/api/chat")` calls through Streamlit (component protocol) and reports the iframe height.
`tests/test_core.py::ProtectedChatbot` fails if anything else in these files changes.
