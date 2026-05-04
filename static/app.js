// Phone-side runner. Captures rear camera, ships JPEG frames over WS, speaks
// recommendations through the Web Speech API.

const FRAME_INTERVAL_MS = 100;          // ~10 fps; pose pipeline can't keep up faster
const JPEG_QUALITY = 0.6;
const TARGET_WIDTH = 480;                // downscale before encode — bandwidth + latency
const TTS_COOLDOWN_MS = 8000;            // per-message dedupe

const els = {
  status: document.getElementById("status"),
  video: document.getElementById("preview"),
  canvas: document.getElementById("canvas"),
  recs: document.getElementById("recs"),
  elapsed: document.getElementById("elapsed"),
  spm: document.getElementById("spm"),
  strike: document.getElementById("strike"),
  start: document.getElementById("start"),
  stop: document.getElementById("stop"),
};

const state = {
  ws: null,
  stream: null,
  ticker: null,
  wakeLock: null,
  busy: false,
  lastSpoken: new Map(),
};

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.className = kind || "";
}

async function start() {
  els.start.disabled = true;
  setStatus("requesting camera…");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
      audio: false,
    });
  } catch (err) {
    setStatus("camera denied: " + err.message, "bad");
    els.start.disabled = false;
    return;
  }
  els.video.srcObject = state.stream;
  await els.video.play().catch(() => {});

  // Web Speech requires a user gesture to unlock — start() is that gesture.
  // Speak a silent priming utterance so subsequent server-driven speak() works.
  primeTTS();

  await acquireWakeLock();

  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws.binaryType = "arraybuffer";
  state.ws.onopen = () => {
    setStatus("connected", "ok");
    els.stop.disabled = false;
    state.ticker = setInterval(tick, FRAME_INTERVAL_MS);
  };
  state.ws.onmessage = (ev) => handleMessage(ev.data);
  state.ws.onclose = () => {
    setStatus("disconnected", "bad");
    cleanup();
  };
  state.ws.onerror = () => setStatus("ws error", "bad");
}

function stop() {
  if (state.ws) state.ws.close();
  cleanup();
  state.lastSpoken.clear();
  setStatus("idle");
}

function cleanup() {
  clearInterval(state.ticker);
  state.ticker = null;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  if (state.wakeLock) {
    state.wakeLock.release().catch(() => {});
    state.wakeLock = null;
  }
  els.start.disabled = false;
  els.stop.disabled = true;
}

async function tick() {
  if (state.busy || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (!els.video.videoWidth) return;
  state.busy = true;
  try {
    const w = TARGET_WIDTH;
    const h = Math.round((els.video.videoHeight / els.video.videoWidth) * w);
    if (els.canvas.width !== w || els.canvas.height !== h) {
      els.canvas.width = w;
      els.canvas.height = h;
    }
    const ctx = els.canvas.getContext("2d");
    ctx.drawImage(els.video, 0, 0, w, h);
    const blob = await new Promise((res) =>
      els.canvas.toBlob(res, "image/jpeg", JPEG_QUALITY),
    );
    if (!blob) return;
    const buf = await blob.arrayBuffer();
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(buf);
    }
  } finally {
    state.busy = false;
  }
}

function handleMessage(data) {
  let msg;
  try {
    msg = JSON.parse(data);
  } catch {
    return;
  }
  renderRecs(msg.recommendations || []);
  renderSummary(msg.summary || {});
  speakRecs(msg.recommendations || []);
}

function renderRecs(recs) {
  els.recs.innerHTML = "";
  if (!recs.length) {
    const div = document.createElement("div");
    div.className = "rec good";
    div.textContent = "Form looking good — keep going.";
    els.recs.appendChild(div);
    return;
  }
  for (const r of recs) {
    const div = document.createElement("div");
    div.className = "rec";
    div.textContent = r;
    els.recs.appendChild(div);
  }
}

function renderSummary(s) {
  els.elapsed.textContent = `${(s.elapsed_time || 0).toFixed(1)} s`;
  els.spm.textContent = (s.steps_per_minute || 0).toFixed(0);
  const strike = [s.left_foot_strike && "L", s.right_foot_strike && "R"]
    .filter(Boolean)
    .join("/") || "—";
  els.strike.textContent = strike;
}

function speakRecs(recs) {
  const now = Date.now();
  for (const [msg, ts] of state.lastSpoken) {
    if (now - ts > TTS_COOLDOWN_MS) state.lastSpoken.delete(msg);
  }
  for (const r of recs) {
    const last = state.lastSpoken.get(r) || 0;
    if (now - last < TTS_COOLDOWN_MS) continue;
    state.lastSpoken.set(r, now);
    const u = new SpeechSynthesisUtterance(r);
    u.rate = 1.05;
    speechSynthesis.speak(u);
  }
}

function primeTTS() {
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(" ");
  u.volume = 0;
  speechSynthesis.speak(u);
}

async function acquireWakeLock() {
  if (!("wakeLock" in navigator)) return;
  try {
    state.wakeLock = await navigator.wakeLock.request("screen");
  } catch {
    /* not granted, screen may dim — nothing fatal */
  }
}

// Re-acquire wake lock on visibility changes (iOS releases on background).
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && state.stream && !state.wakeLock) {
    acquireWakeLock();
  }
});

els.start.addEventListener("click", start);
els.stop.addEventListener("click", stop);
