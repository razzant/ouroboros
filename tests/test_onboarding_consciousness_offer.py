"""The setup's Consciousness step and its carrier (owner decision 2026-11-20).

The step is offered while setup has never completed here and the state is not unreadable: with no state yet (the next
boot mints a first state) the switch starts on; a state that says off shows off; one that says on reads On already. The
page that carries it writes nothing. The choice rides the one completion write as a completion fact; the first state a
mint creates (first boot or owner Reset) is born with it, unless an installation Panic is pending, and where a state
already exists completion performs the owner's start: no command is sent, so Main shows no row. ``false`` and absence
change nothing. Every case goes through the real endpoint, the real settings write and the real ``init_state``.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.settings_setup_contract import (
    ONBOARDING_COMPLETED_KEY, ONBOARDING_CONSCIOUSNESS_KEY, build_setup_bootstrap, completion_facts, consciousness_offer,
)
from tests.test_onboarding_complete_endpoint import WIZARD_PAYLOAD, onboarding as onboarding


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


def test_offered_exactly_while_setup_has_never_completed_and_the_state_is_readable():
    assert consciousness_offer({})["offered"] is True  # a brand-new install
    # A first setup whose settings already exist, and a markerless older install: offered, decided now.
    assert consciousness_offer({"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9", "OUROBOROS_MODEL": "x"})["offered"] is True
    assert consciousness_offer({ONBOARDING_COMPLETED_KEY: "  "}, enabled=False)["offered"] is True
    # A completed setup re-entering the wizard, and a state that exists but cannot be read: no step.
    assert consciousness_offer({ONBOARDING_COMPLETED_KEY: "2026-11-04T10:00:00+00:00"})["offered"] is False
    assert consciousness_offer({}, readable=False)["offered"] is False
    assert build_setup_bootstrap({}, "web")["stepOrder"] == ["accounts", "models", "review_mode", "consciousness", "budget", "summary"]
    assert "consciousness" not in build_setup_bootstrap({}, "web", consciousness_readable=False)["stepOrder"]
    assert "consciousness" not in build_setup_bootstrap({ONBOARDING_COMPLETED_KEY: "t"}, "web")["stepOrder"]


def test_the_limits_are_the_runtime_readers_values(monkeypatch):
    for key in ("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "OUROBOROS_CONSCIOUSNESS_DAILY_USD", "OUROBOROS_CONSCIOUSNESS_MAX_TASKS"):
        monkeypatch.delenv(key, raising=False)
    from ouroboros.consciousness_view import FIRST_RUN, ON_WITHOUT_PLACES, POSITION_WORDS, first_run_words

    assert consciousness_offer({}) == {"offered": True, "enabled": None, "autonomy": "act", "dailyUsd": 20.0, "maxTasks": 2,
                                       "words": FIRST_RUN}  # the switch's words (owner 15.11, 5A)
    # The two positions of an install with no place yet, and a subscription's quota said where it is the one payer.
    assert (FIRST_RUN["off"], FIRST_RUN["on"]) == (POSITION_WORDS[False], ON_WITHOUT_PLACES)
    assert "quota" in FIRST_RUN["subscriptionCost"] and "API spending only" in FIRST_RUN["subscriptionCost"]
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "observe")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "0")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "0")
    assert consciousness_offer({}, enabled=False) == {
        "offered": True, "enabled": False, "autonomy": "observe", "dailyUsd": 0.0, "maxTasks": 0,
        "words": first_run_words("observe")}  # Advises answers nowhere: its first-run words say so (review r3)
    assert "answers in places" not in first_run_words("observe")["meaning"]
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "bogus")  # the reader's own fallback, not a new level
    assert consciousness_offer({})["autonomy"] == "act"
    assert consciousness_offer({}, enabled="yes")["enabled"] is None


def test_the_completion_carries_the_switch_and_only_on_is_a_fact():
    from ouroboros.gateway import contracts
    from ouroboros.settings_setup_contract import validate_setup_payload

    from typing import get_type_hints

    assert get_type_hints(contracts.OnboardingCompleteRequest)["consciousness"] is bool
    assert completion_facts({"consciousness": True}, "a")[ONBOARDING_CONSCIOUSNESS_KEY] == "on"
    for body in ({"consciousness": False}, {}, {"consciousness": "true"}):
        assert ONBOARDING_CONSCIOUSNESS_KEY not in completion_facts(body, "a")
    payload = {"OPENROUTER_API_KEY": "sk-or-v1-abcdefghijklmnop", "OUROBOROS_MODEL": "openai/gpt-5.6-luna"}
    assert validate_setup_payload({**payload, "consciousness": "yes"}, {})[1] == "consciousness must be true or false."
    assert validate_setup_payload({**payload, "consciousness": False}, {})[1] is None


def test_the_page_reads_the_control_and_writes_nothing(monkeypatch, tmp_path):
    from ouroboros.gateway import onboarding_host

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {})
    before = _tree(tmp_path)
    with _client(tmp_path) as client:
        fresh = client.get("/api/onboarding/bootstrap").json()["bootstrap"]
        page = client.get("/onboarding")
    # No state yet: offered with no control to show, so the switch starts on.
    assert fresh["consciousnessOffer"]["offered"] is True and fresh["consciousnessOffer"]["enabled"] is None
    assert "consciousness" in fresh["stepOrder"]
    assert page.status_code == 200 and '"consciousnessOffer"' in page.text
    assert _tree(tmp_path) == before, "serving the offer authored settings, state or a witness"

    for stored, shown in (({"bg_consciousness_enabled": True}, True), ({"bg_consciousness_enabled": False}, False)):
        _initialized_state(tmp_path, stored)
        seeded = _tree(tmp_path)
        with _client(tmp_path) as client:
            offer = client.get("/api/onboarding/bootstrap").json()["bootstrap"]["consciousnessOffer"]
        assert (offer["offered"], offer["enabled"]) == (True, shown)
        assert _tree(tmp_path) == seeded, "reading the control wrote state"

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {ONBOARDING_COMPLETED_KEY: "2026-11-04T10:00:00+00:00"})
    with _client(tmp_path) as client:
        assert client.get("/api/onboarding/bootstrap").json()["bootstrap"]["consciousnessOffer"]["offered"] is False


def test_only_a_root_the_next_boot_would_mint_counts_as_no_state_yet(monkeypatch, tmp_path):
    """An owner Reset's pending witness is a first state to come: offered. A completed witness whose copies are lost
    and a copy that cannot be read are a state that exists: no step, and nothing is written by asking."""
    from ouroboros.gateway import onboarding_host
    from supervisor.state_initialization import mark_pending

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {})
    mark_pending(tmp_path, origin="owner_reset")
    seeded = _tree(tmp_path)
    with _client(tmp_path) as client:
        bootstrap = client.get("/api/onboarding/bootstrap").json()["bootstrap"]
    assert (bootstrap["consciousnessOffer"]["offered"], bootstrap["consciousnessOffer"]["enabled"]) == (True, None)
    assert _tree(tmp_path) == seeded, "asking whether a first state is due wrote the witness"
    (tmp_path / "state" / "state.initialized.json").write_text(
        json.dumps({"initialization_id": "init-1", "phase": "complete", "completed_at": "t"}), encoding="utf-8")
    seeded = _tree(tmp_path)
    with _client(tmp_path) as client:
        lost = client.get("/api/onboarding/bootstrap").json()["bootstrap"]
    assert lost["consciousnessOffer"]["offered"] is False and "consciousness" not in lost["stepOrder"]
    assert _tree(tmp_path) == seeded


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


def test_an_unreadable_state_offers_no_step_and_the_page_still_serves(monkeypatch, tmp_path):
    from ouroboros.gateway import onboarding_host

    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {})
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "state.json").write_text("{not json", encoding="utf-8")
    with _client(tmp_path) as client:
        response = client.get("/api/onboarding/bootstrap")
    assert response.status_code == 200
    offer = response.json()["bootstrap"]["consciousnessOffer"]
    assert (offer["offered"], offer["enabled"]) == (False, None)
    assert "consciousness" not in response.json()["bootstrap"]["stepOrder"]
    assert (tmp_path / "state" / "state.json").read_text(encoding="utf-8") == "{not json"


# ---- The carrier: the real endpoint, the real settings write, the real mint ------------------------------------------


@pytest.fixture
def minted(onboarding, tmp_path, monkeypatch):
    """The completion endpoint and ``supervisor.state`` over one isolated root."""
    from supervisor import evolution_lifecycle as lifecycle
    from supervisor import queue, state

    prior_state, prior_queue = state.DRIVE_ROOT, queue.DRIVE_ROOT
    state.init(tmp_path)
    queue.init(tmp_path)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda: None)
    monkeypatch.setitem(lifecycle._STOP_LATCH, "stopped", False)
    yield onboarding
    state.init(prior_state)
    queue.init(prior_queue)


def _complete(harness, **fields) -> dict:
    response = harness.client.post("/api/onboarding/complete", json={**WIZARD_PAYLOAD, **fields})
    assert response.status_code == 200, response.text
    return response.json()


def _boot_control():
    from supervisor import state

    read = state.init_state()
    assert read.quality == "current", read
    return read.values["bg_consciousness_enabled"], read.values.get("bg_consciousness_stop_source")


@pytest.mark.parametrize("restart", [False, True], ids=["running-mode", "restart-required"])
def test_a_fresh_install_left_on_is_born_on_and_one_left_off_is_born_off(minted, tmp_path, restart):
    from ouroboros.config import get_runtime_mode

    mode = next(m for m in ("light", "advanced") if m != get_runtime_mode()) if restart else get_runtime_mode()
    receipt = _complete(minted, consciousness=True, OUROBOROS_RUNTIME_MODE=mode)
    assert receipt["restart_required"] is restart
    assert minted.saved()[ONBOARDING_CONSCIOUSNESS_KEY] == "on"
    assert not (tmp_path / "state" / "state.json").exists(), "completion writes no state; the mint carries the choice"
    assert _boot_control() == (True, None)
    # The other direction, in the same home: a switch turned off (or a page that sends nothing) is born off.
    shutil.rmtree(tmp_path / "state")
    minted.settings_path.unlink(missing_ok=True)
    _complete(minted, consciousness=False)
    assert not minted.saved().get(ONBOARDING_CONSCIOUSNESS_KEY)
    assert _boot_control() == (False, None)


def test_an_owner_reset_then_setup_left_on_is_born_on(minted, tmp_path):
    from ouroboros.consciousness import OWNER_STOP, _persist_stop
    from supervisor.state_initialization import mark_pending

    _complete(minted, consciousness=True)
    assert _boot_control() == (True, None)
    assert _persist_stop(OWNER_STOP) == ""  # the owner turned it off before the Reset
    for subdir in ("state", "memory", "logs", "task_results"):  # gateway/control.api_reset's wipe
        shutil.rmtree(tmp_path / subdir, ignore_errors=True)
    minted.settings_path.unlink(missing_ok=True)
    mark_pending(tmp_path, origin="owner_reset")
    _complete(minted, consciousness=True)
    assert _boot_control() == (True, None)


def test_an_older_install_with_an_owner_stop_stays_off_until_the_owner_turns_it_on(minted, tmp_path):
    from ouroboros.consciousness import OWNER_STOP, _persist_stop

    assert _boot_control() == (False, None)  # an install from before the step, born off
    assert _persist_stop(OWNER_STOP) == ""
    for body in ({"consciousness": False}, {}):  # the wizard clicked through with the switch as it showed, and an older page
        minted.settings_path.unlink(missing_ok=True)
        _complete(minted, **body)
        assert _boot_control() == (False, OWNER_STOP)
    minted.settings_path.unlink(missing_ok=True)
    _complete(minted, consciousness=True)  # the owner's move: the switch turned on where a state exists
    assert _boot_control() == (True, None)


def test_a_present_false_neither_stops_nor_clears_a_saved_on(minted, tmp_path):
    _complete(minted, consciousness=True)
    assert _boot_control() == (True, None)
    document = minted.saved()
    document.pop(ONBOARDING_COMPLETED_KEY)
    minted.settings_path.write_text(json.dumps(document), encoding="utf-8")
    _complete(minted, consciousness=False)
    assert minted.saved()[ONBOARDING_CONSCIOUSNESS_KEY] == "on"
    assert _boot_control() == (True, None)


def test_a_panic_before_the_first_mint_leaves_it_off_and_a_later_stop_wins_across_a_restart(minted, tmp_path):
    from ouroboros.consciousness import OWNER_STOP, _persist_stop

    _complete(minted, consciousness=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    assert _boot_control() == (False, None), "a pending Panic withholds the setup's choice"
    # Without the Panic the same root and document are born on; then the owner's Stop wins and survives a restart.
    shutil.rmtree(tmp_path / "state")
    assert _boot_control() == (True, None)
    assert _persist_stop(OWNER_STOP) == ""
    assert _boot_control() == (False, OWNER_STOP)


def test_a_start_that_cannot_land_is_reported_at_the_commit_boundary(minted, tmp_path):
    assert _boot_control() == (False, None)
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    minted.settings_path.unlink(missing_ok=True)
    response = minted.client.post("/api/onboarding/complete", json={**WIZARD_PAYLOAD, "consciousness": True})
    assert response.status_code == 500, response.text
    body = response.json()
    assert (body["saved"], body["post_commit_failed"]) == (True, "consciousness start")
    assert "Panic" in body["error"] and minted.saved()[ONBOARDING_CONSCIOUSNESS_KEY] == "on"
    assert minted.calls["supervisor"] == 1, "the start runs after every established post-save step"
