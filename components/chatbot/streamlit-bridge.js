/*
 * streamlit-bridge.js
 *
 * Lets the ORIGINAL chatbot (index.html + script.js + style.css) talk to Python
 * when it is embedded in Streamlit as a custom component (iframe).
 *
 * The chatbot still calls   fetch("/api/predict")  and  fetch("/api/chat").
 * Those two calls are intercepted here and sent to Streamlit/Python with the
 * Streamlit component protocol; the answer comes back as a normal Response.
 * Nothing about the chatbot's look or logic is touched.
 *
 * Protocol (Streamlit custom components, API v1, no build step needed):
 *   iframe -> Streamlit : window.parent.postMessage({isStreamlitMessage:true, type, ...})
 *       streamlit:componentReady   {apiVersion: 1}
 *       streamlit:setFrameHeight   {height}
 *       streamlit:setComponentValue{value, dataType:"json"}
 *   Streamlit -> iframe : "message" event, data.type === "streamlit:render", data.args = {...}
 */
(function () {
    "use strict";

    var TIMEOUT_MS = 120000;
    var TIMEOUT_MESSAGE = "انتهت مهلة الاتصال. حاول مرة أخرى.";

    var nonce = Math.random().toString(36).slice(2, 10); // unique per iframe mount
    var seq = 0;
    var pending = {};                 // request id -> {resolve, timer}
    var queue = Promise.resolve();    // one request in flight at a time

    function send(type, data) {
        var msg = { isStreamlitMessage: true, type: type };
        for (var k in data) { if (Object.prototype.hasOwnProperty.call(data, k)) msg[k] = data[k]; }
        window.parent.postMessage(msg, "*");
    }

    // ---- Python -> chatbot -------------------------------------------------
    window.addEventListener("message", function (event) {
        var d = event.data;
        if (!d || d.type !== "streamlit:render") return;
        reportHeight();
        var resp = d.args && d.args.response;
        if (!resp || !resp.id) return;
        var p = pending[resp.id];
        if (!p) return;                     // stale/duplicate response, ignore
        clearTimeout(p.timer);
        delete pending[resp.id];
        p.resolve(resp);
    });

    // ---- chatbot -> Python -------------------------------------------------
    function call(route, payload) {
        return new Promise(function (resolve) {
            var id = nonce + "-" + (++seq);
            var timer = setTimeout(function () {
                delete pending[id];
                resolve({ status: 504, body: { detail: TIMEOUT_MESSAGE } });
            }, TIMEOUT_MS);
            pending[id] = { resolve: resolve, timer: timer };
            send("streamlit:setComponentValue", {
                value: { id: id, route: route, payload: payload },
                dataType: "json"
            });
        });
    }

    function enqueue(route, payload) {
        var run = function () { return call(route, payload); };
        queue = queue.then(run, run);
        return queue;
    }

    // Intercept ONLY the two API calls the chatbot makes; everything else is untouched.
    var realFetch = window.fetch ? window.fetch.bind(window) : null;
    window.fetch = function (url, options) {
        var path;
        try { path = new URL(String(url), window.location.href).pathname; }
        catch (e) { path = String(url); }
        var m = /\/api\/(predict|chat)$/.exec(path);
        if (!m) {
            return realFetch ? realFetch(url, options)
                             : Promise.reject(new Error("fetch unavailable"));
        }
        var payload = {};
        try { payload = JSON.parse((options && options.body) || "{}"); } catch (e) { /* keep {} */ }
        return enqueue("/api/" + m[1], payload).then(function (resp) {
            return new Response(JSON.stringify(resp.body || {}), {
                status: resp.status || 500,
                headers: { "Content-Type": "application/json" }
            });
        });
    };

    // ---- iframe height -----------------------------------------------------
    // style.css gives .page-shell "min-height: 100vh" (protected file, not edited),
    // so measuring the document would never shrink. Measure the footer instead:
    // it is the last visible element of the page.
    var lastHeight = 0;
    function reportHeight() {
        var f = document.querySelector("footer");
        var h = f
            ? Math.ceil(f.getBoundingClientRect().bottom + (window.pageYOffset || 0))
            : document.documentElement.scrollHeight;
        if (Math.abs(h - lastHeight) >= 1) {
            lastHeight = h;
            send("streamlit:setFrameHeight", { height: h });
        }
    }

    function startHeightWatch() {
        reportHeight();
        if (window.ResizeObserver) {
            var ro = new ResizeObserver(reportHeight);
            ro.observe(document.body);
            var c = document.querySelector(".container");
            if (c) ro.observe(c);
        }
        window.addEventListener("resize", reportHeight);
        if (document.fonts && document.fonts.ready) document.fonts.ready.then(reportHeight);
        window.addEventListener("load", reportHeight);
    }

    // ---- start ---------------------------------------------------------------
    send("streamlit:componentReady", { apiVersion: 1 });
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startHeightWatch);
    } else {
        startHeightWatch();
    }
})();
