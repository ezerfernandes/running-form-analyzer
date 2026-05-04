from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.config import Config


@pytest.fixture
def config():
    return Config(side="right", model_type="blazepose", runner_height=182)


@pytest.fixture
def app(config):
    # Patch the model so the server doesn't try to instantiate BlazePose during
    # smoke tests (it would download a 5 MB model and load mediapipe). The
    # patch target is core.frame_processor because that's where the import
    # happens at FrameProcessor construction time.
    with patch("core.frame_processor.BlazePoseModel") as mock_model_cls, patch(
        "metrics.metrics.AudioFeedbackProvider"
    ):
        mock_model_cls.return_value = MagicMock(
            predict=MagicMock(return_value=None), close=MagicMock()
        )
        from server import create_app

        yield create_app(config)


def _jpeg_bytes():
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_index_serves_html(app):
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Running Form Analyzer" in r.text
    assert "/static/app.js" in r.text


def test_healthz(app):
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_static_app_js(app):
    client = TestClient(app)
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "WebSocket" in r.text


def test_websocket_processes_frame_and_responds(app):
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(_jpeg_bytes())
        msg = ws.receive_json()
        assert "recommendations" in msg
        assert "summary" in msg
        assert isinstance(msg["recommendations"], list)
        assert "elapsed_time" in msg["summary"]
        assert "steps_per_minute" in msg["summary"]


def test_second_websocket_rejected_while_first_open(app):
    # Single-connection invariant: model state isn't safe to share, so the
    # server slams the door on a second client until the first disconnects.
    client = TestClient(app)
    with client.websocket_connect("/ws"):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws"):
                pass
