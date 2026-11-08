"""The Summary's optional Consciousness row, as the server offers it (#1603).

The offer is decided by the install's own completion record: a setup that never completed is offered an
unticked choice (a fresh install, a first setup whose settings file already exists, a markerless older
install), and a completed setup is offered nothing, so it never repeats an old choice. The limits beside it
are the ones consciousness runs under now, read by the runtime's own readers. The page that carries it
writes nothing: neither settings nor state, and nothing in the completion save refers to it.
"""

from __future__ import annotations

import json
import pathlib

from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.settings_setup_contract import ONBOARDING_COMPLETED_KEY, build_setup_bootstrap, consciousness_offer


def _client(tmp_path: pathlib.Path) -> TestClient:
    from ouroboros.gateway.router import collect_routes

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    return TestClient(app)


def _tree(root: pathlib.Path) -> dict:
    return {str(path.relative_to(root)): (None if path.is_dir() else path.read_bytes()) for path in sorted(root.rglob("*"))}


def _initialized_state(root: pathlib.Path, stored: dict) -> None:
    """A primary copy of a completed initialization, as the existing consciousness tests seed it."""
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state" / "state.initialized.json").write_text(
        json.dumps({"initialization_id": "init-1", "phase": "complete"}), encoding="utf-8")
    (root / "state" / "state.json").write_text(json.dumps({"initialization_id": "init-1", **stored}), encoding="utf-8")


def test_offered_exactly_while_setup_has_never_completed():
    assert consciousness_offer({})["offered"] is True  # a brand-new install
    # A first setup whose settings already exist, and a markerless older install: an unticked choice, made now.
    assert consciousness_offer({"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9", "OUROBOROS_MODEL": "x"})["offered"] is True
    assert consciousness_offer({ONBOARDING_COMPLETED_KEY: "  "})["offered"] is True
    # A completed setup re-entering the wizard: no row, so no old choice can be repeated.
    assert consciousness_offer({ONBOARDING_COMPLETED_KEY: "2026-11-04T10:00:00+00:00"})["offered"] is False


def test_the_limits_are_the_runtime_readers_values(monkeypatch):
    for key in ("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "OUROBOROS_CONSCIOUSNESS_DAILY_USD", "OUROBOROS_CONSCIOUSNESS_MAX_TASKS"):
        monkeypatch.delenv(key, raising=False)
    assert consciousness_offer({}) == {"offered": True, "enabled": None, "autonomy": "act", "dailyUsd": 20.0, "maxTasks": 2}
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "observe")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "0")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "0")
    assert consciousness_offer({}, enabled=False) == {
        "offered": True, "enabled": False, "autonomy": "observe", "dailyUsd": 0.0, "maxTasks": 0}
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "bogus")  # the reader's own fallback, not a new level
    assert consciousness_offer({})["autonomy"] == "act"
    assert consciousness_offer({}, enabled="yes")["enabled"] is None


def test_the_bootstrap_carries_the_offer_and_the_save_contract_does_not():
    bootstrap = build_setup_bootstrap({}, "web", fresh_install=True, consciousness_enabled=True)
    assert bootstrap["consciousnessOffer"]["enabled"] is True
    assert build_setup_bootstrap({}, "web")["consciousnessOffer"]["enabled"] is None
    from ouroboros.gateway import contracts

    assert not any("onscious" in key for key in contracts.OnboardingCompleteRequest.__annotations__)


def test_the_page_reads_the_control_and_writes_nothing(monkeypatch, tmp_path):
    from ouroboros.gateway import onboarding_host

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {})
    before = _tree(tmp_path)
    with _client(tmp_path) as client:
        fresh = client.get("/api/onboarding/bootstrap").json()["bootstrap"]["consciousnessOffer"]
        page = client.get("/onboarding")
    assert fresh["offered"] is True and fresh["enabled"] is None  # no state yet: unknown, never "off"
    assert page.status_code == 200 and '"consciousnessOffer"' in page.text
    assert _tree(tmp_path) == before, "serving the offer authored settings or state"

    for stored, shown in (({"bg_consciousness_enabled": True}, True), ({"bg_consciousness_enabled": False}, False)):
        _initialized_state(tmp_path, stored)
        seeded = _tree(tmp_path)
        with _client(tmp_path) as client:
            offer = client.get("/api/onboarding/bootstrap").json()["bootstrap"]["consciousnessOffer"]
        assert offer["enabled"] is shown
        assert _tree(tmp_path) == seeded, "reading the control wrote state"

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {ONBOARDING_COMPLETED_KEY: "2026-11-04T10:00:00+00:00"})
    with _client(tmp_path) as client:
        assert client.get("/api/onboarding/bootstrap").json()["bootstrap"]["consciousnessOffer"]["offered"] is False


def _embedded_bootstrap(page: str) -> dict:
    marker = "window.__OURO_ONBOARDING_BOOTSTRAP__ = "
    return json.JSONDecoder().raw_decode(page, page.index(marker) + len(marker))[0]


def test_the_readiness_probe_body_is_the_same_live_setup(monkeypatch, tmp_path):
    """`GET /api/onboarding` answers a not-ready install with the wizard itself. Its body embeds the one live Setup
    bootstrap, so a control already on shows as on (no Start to offer) and an unreadable state stays unknown."""
    from ouroboros.gateway import onboarding_host

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {})
    for stored, shown in (({"bg_consciousness_enabled": True}, True), ({"bg_consciousness_enabled": False}, False)):
        _initialized_state(tmp_path, stored)
        seeded = _tree(tmp_path)
        with _client(tmp_path) as client:
            probe = client.get("/api/onboarding")
            bootstrap = client.get("/api/onboarding/bootstrap").json()["bootstrap"]
        assert probe.status_code == 200
        assert _embedded_bootstrap(probe.text) == bootstrap
        assert bootstrap["consciousnessOffer"]["enabled"] is shown
        assert _tree(tmp_path) == seeded, "the readiness probe wrote state"
    (tmp_path / "state" / "state.json").write_text("{not json", encoding="utf-8")
    with _client(tmp_path) as client:
        probe = client.get("/api/onboarding")
    assert probe.status_code == 200 and _embedded_bootstrap(probe.text)["consciousnessOffer"]["enabled"] is None


def test_an_unreadable_state_is_unknown_and_the_page_still_serves(monkeypatch, tmp_path):
    from ouroboros.gateway import onboarding_host

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {})
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "state.json").write_text("{not json", encoding="utf-8")
    with _client(tmp_path) as client:
        response = client.get("/api/onboarding/bootstrap")
    assert response.status_code == 200
    assert response.json()["bootstrap"]["consciousnessOffer"]["enabled"] is None
    assert (tmp_path / "state" / "state.json").read_text(encoding="utf-8") == "{not json"
