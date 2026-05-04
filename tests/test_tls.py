import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from utils.tls import detect_lan_ip, ensure_self_signed


def test_detect_lan_ip_returns_string():
    ip = detect_lan_ip()
    assert isinstance(ip, str)
    assert ip.count(".") == 3


def test_detect_lan_ip_falls_back_when_offline():
    with patch("utils.tls.socket.socket") as mock_sock_cls:
        sock = MagicMock()
        sock.connect.side_effect = OSError("network unreachable")
        mock_sock_cls.return_value = sock
        assert detect_lan_ip() == "127.0.0.1"


def test_ensure_self_signed_idempotent(tmp_path: Path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("EXISTING-CERT")
    key.write_text("EXISTING-KEY")

    with patch("utils.tls.subprocess.run") as mock_run:
        c, k = ensure_self_signed(tmp_path)
        # Existing files mean openssl must not be invoked.
        mock_run.assert_not_called()
        assert c == cert
        assert k == key
        assert cert.read_text() == "EXISTING-CERT"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not installed")
def test_ensure_self_signed_creates_cert(tmp_path: Path):
    cert, key = ensure_self_signed(tmp_path)
    assert cert.exists()
    assert key.exists()
    assert "BEGIN CERTIFICATE" in cert.read_text()
