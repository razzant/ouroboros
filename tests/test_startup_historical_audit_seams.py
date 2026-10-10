"""The historical audit is explicit; server lifecycle owns no audit child."""
from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_boot_does_not_launch_a_historical_audit():
    server = (REPO / "server.py").read_text(encoding="utf-8")
    assert "_historical_audit" not in server
    from ouroboros.startup_historical_audit import main
    assert callable(main)
