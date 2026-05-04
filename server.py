"""Phone-as-camera streaming server.

Runs FastAPI + uvicorn over HTTPS. The phone browser opens `/`, captures camera
frames via getUserMedia, JPEG-encodes them, and pushes binary frames over the
`/ws` WebSocket. The server decodes each frame, runs the existing pose +
metrics pipeline via `FrameProcessor`, and pushes recommendation text back so
the phone speaks it through the Web Speech API.

Single connection at a time — the underlying pose models hold per-instance
state (BlazePose's running-mode timestamp, MoveNet's TFLite tensors) that is
not safe to share across coroutines. A second connection attempt is rejected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.config import Config
from core.frame_processor import FrameProcessor
from utils.tls import detect_lan_ip, ensure_self_signed

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent
_STATIC_DIR = _REPO_ROOT / "static"
_CERT_DIR = _REPO_ROOT / "tmp" / "certs"


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="Running Form Analyzer (server mode)")
    app.state.config = config
    app.state.connection_lock = asyncio.Lock()

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        # Reject a second simultaneous client — model state isn't sharable.
        if app.state.connection_lock.locked():
            await websocket.close(code=1013, reason="busy")
            return
        async with app.state.connection_lock:
            await websocket.accept()
            await _run_session(websocket, config)

    return app


async def _run_session(websocket: WebSocket, config: Config) -> None:
    processor = FrameProcessor(config, enable_audio=False)
    start = time.time()
    frames_in = 0
    last_log = start
    busy = False

    try:
        while True:
            data = await websocket.receive_bytes()
            if busy:
                # Drop the frame — back-pressure. We'd rather keep latency low
                # than queue stale frames.
                continue
            busy = True
            try:
                metrics, _angles = await asyncio.to_thread(
                    _process_jpeg, processor, data, time.time() - start
                )
            except Exception as exc:
                logger.exception("frame processing failed: %s", exc)
                continue
            finally:
                busy = False

            if metrics is None:
                continue

            frames_in += 1
            now = time.time()
            if now - last_log >= 2.0:
                fps = frames_in / max(1e-6, now - last_log)
                logger.info(
                    "frames=%d fps=%.1f recs=%d",
                    frames_in,
                    fps,
                    len(metrics.get("recommendations") or []),
                )
                frames_in = 0
                last_log = now

            await websocket.send_text(json.dumps(_payload(metrics)))
    except WebSocketDisconnect:
        pass
    finally:
        processor.close()


def _process_jpeg(
    processor: FrameProcessor, jpeg: bytes, current_time: float
):
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None
    return processor.process(frame, current_time)


def _payload(metrics: dict) -> dict:
    """Pick the small slice of metrics worth sending each frame.

    Most metrics are noisy frame-by-frame floats the phone has no use for; the
    recommendations list is the whole point of this loop. We also surface a
    couple of cadence numbers for the on-phone debug readout.
    """
    return {
        "recommendations": list(metrics.get("recommendations") or []),
        "summary": {
            "elapsed_time": float(metrics.get("elapsed_time", 0.0)),
            "steps_per_minute": float(metrics.get("steps_per_minute", 0.0)),
            "left_foot_strike": bool(metrics.get("left_foot_strike", False)),
            "right_foot_strike": bool(metrics.get("right_foot_strike", False)),
        },
    }


def serve(config: Config, host: str, port: int) -> None:
    import uvicorn

    cert, key = ensure_self_signed(_CERT_DIR)
    lan_ip = detect_lan_ip()
    print(f"\nPhone URL:  https://{lan_ip}:{port}/")
    print(f"Cert:       {cert}")
    print("First visit on the phone will show a self-signed warning — tap through.\n")

    app = create_app(config)
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
        log_level="info",
    )
