"""Self-signed TLS cert helper for LAN-only server mode.

Phone browsers refuse to grant camera access over plain HTTP from a non-
localhost origin, so server mode needs HTTPS even on a LAN. This module
generates an idempotent self-signed cert pinned to the laptop's current LAN
IP so the phone can hit `https://<ip>:<port>` directly.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Tuple


def detect_lan_ip() -> str:
    """Return the laptop's LAN IPv4 address.

    Uses the connect-to-public-IP trick: opens a UDP socket aimed at 8.8.8.8
    (no packet sent — UDP doesn't handshake) and reads the local endpoint the
    OS would route through. Falls back to 127.0.0.1 if nothing is reachable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_self_signed(cert_dir: Path) -> Tuple[Path, Path]:
    """Ensure cert.pem and key.pem exist in ``cert_dir``; create them if not.

    Returns (cert_path, key_path). Re-uses an existing pair when the LAN IP
    hasn't changed so the phone doesn't have to re-trust on every restart.
    Regenerates when the IP changes (different network) to avoid SAN mismatch.
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert = cert_dir / "cert.pem"
    key = cert_dir / "key.pem"
    ip_file = cert_dir / "ip.txt"

    lan_ip = detect_lan_ip()
    if cert.exists() and key.exists():
        if ip_file.exists() and ip_file.read_text().strip() == lan_ip:
            return cert, key
    san = f"subjectAltName=DNS:localhost,IP:127.0.0.1,IP:{lan_ip}"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "365",
            "-subj",
            "/CN=running-form-analyzer",
            "-addext",
            san,
        ],
        check=True,
        capture_output=True,
    )
    ip_file.write_text(lan_ip)
    return cert, key
