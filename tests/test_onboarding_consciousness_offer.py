"""The setup's Consciousness step and its carrier (owner decision 2026-11-20).

The step is offered only while the next boot mints a first state and setup has never completed here; its switch then
starts on. Where any state exists (off, on or unreadable) there is no step. The page that carries it writes nothing. The
choice rides the one completion write as a completion fact, and only while a first state is still to come (decided
before the write): ``true`` writes "on", ``false`` clears an earlier attempt's "on". The first state a mint creates (first
boot or owner Reset) is born with it unless an installation Panic is pending. Where a state exists the field changes
nothing, so setup never lifts an owner or Panic stop; no command is sent, so Main shows no row. Every case goes through
the real endpoint, the real settings write and the real ``init_state``.
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

def _forget_state(root):
    """Remove exactly the two state copies and the witness, as a new data root has none. Everything else under
    ``state/`` stays on every platform (the usage store this test's server holds open is no evidence of an earlier
    supervisor); a case that must also discard a Panic flag removes it itself."""
    for name in ("state.json", "state.last_good.json", "state.initialized.json"):
        (root / "state" / name).unlink(missing_ok=True)


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


def test_offered_exactly_while_a_first_state_is_to_come_and_setup_has_never_completed():
    assert consciousness_offer({})["offered"] is True  # a brand-new install
    # A first setup whose settings already exist: offered, decided now.
    assert consciousness_offer({"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9", "OUROBOROS_MODEL": "x"})["offered"] is True
    assert consciousness_offer({ONBOARDING_COMPLETED_KEY: "  "})["offered"] is True
    # A completed setup re-entering the wizard, and any install that has a state, whatever it says: no step.
    assert consciousness_offer({ONBOARDING_COMPLETED_KEY: "2026-11-04T10:00:00+00:00"})["offered"] is False
    for enabled in (False, True, None):
        assert consciousness_offer({}, enabled=enabled, first_state=False)["offered"] is False
    assert consciousness_offer({}, enabled=False, first_state=False)["enabled"] is False  # shown, never offered
    assert build_setup_bootstrap({}, "web")["stepOrder"] == [
        "accounts", "models", "review_mode", "consciousness", "skills", "budget", "summary"]
    assert "consciousness" not in build_setup_bootstrap({}, "web", consciousness_first_state=False)["stepOrder"]
    assert "consciousness" not in build_setup_bootstrap({ONBOARDING_COMPLETED_KEY: "t"}, "web")["stepOrder"]


def test_the_limits_are_the_runtime_readers_values(monkeypatch):
    for key in ("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "OUROBOROS_CONSCIOUSNESS_DAILY_USD", "OUROBOROS_CONSCIOUSNESS_MAX_TASKS"):
        monkeypatch.delenv(key, raising=False)
    from ouroboros.consciousness_view import FIRST_RUN, ON_WITHOUT_PLACES, POSITION_WORDS, first_run_words

    assert consciousness_offer({}) == {"offered": True, "enabled": None, "autonomy": "act", "dailyUsd": 15.0, "maxTasks": 2,
                                       "words": FIRST_RUN}  # the switch's words (owner 15.11, 5A)
    # The two positions of an install with no place yet, and a subscription's quota said where it is the one payer.
    assert (FIRST_RUN["off"], FIRST_RUN["on"]) == (POSITION_WORDS[False], ON_WITHOUT_PLACES)
    assert "quota" in FIRST_RUN["subscriptionCost"] and "allowance" not in FIRST_RUN["subscriptionCost"]
    # On, the wakes go on before a first reply; either position, Start begins the first conversation (owner 21.11, 1C):
    # no word ties the first message to the switch.
    assert "even before your first reply" in FIRST_RUN["meaning"] and "Either way, Start begins" in FIRST_RUN["meaning"]
    assert "only when you write" not in FIRST_RUN["meaning"]
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "observe")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "0")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "0")
    assert consciousness_offer({}, enabled=False) == {
        "offered": True, "enabled": None, "autonomy": "observe", "dailyUsd": 0.0, "maxTasks": 0,
        "words": first_run_words("observe")}  # Advises answers nowhere: its first-run words say so (review r3)
    assert "answers in places" not in first_run_words("observe")["meaning"]
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "bogus")  # the reader's own fallback, not a new level
    assert consciousness_offer({})["autonomy"] == "act"
    assert consciousness_offer({}, enabled="yes")["enabled"] is None


def test_the_completion_carries_the_switch_only_while_a_first_state_is_to_come():
    from ouroboros.gateway import contracts
    from ouroboros.settings_setup_contract import validate_setup_payload

    from typing import get_type_hints

    assert get_type_hints(contracts.OnboardingCompleteRequest)["consciousness"] is bool
    assert completion_facts({"consciousness": True}, "a", first_state_pending=True)[ONBOARDING_CONSCIOUSNESS_KEY] == "on"
    assert completion_facts({"consciousness": False}, "a", first_state_pending=True)[ONBOARDING_CONSCIOUSNESS_KEY] == ""
    for body in ({}, {"consciousness": "true"}):
        assert ONBOARDING_CONSCIOUSNESS_KEY not in completion_facts(body, "a", first_state_pending=True)
    for body in ({"consciousness": True}, {"consciousness": False}):  # a state exists: the field changes nothing
        assert ONBOARDING_CONSCIOUSNESS_KEY not in completion_facts(body, "a", first_state_pending=False)
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
            bootstrap = client.get("/api/onboarding/bootstrap").json()["bootstrap"]
        # A state exists, whatever it says: no step (Dashboard › Consciousness changes it); the control is shown.
        assert (bootstrap["consciousnessOffer"]["offered"], bootstrap["consciousnessOffer"]["enabled"]) == (False, shown)
        assert "consciousness" not in bootstrap["stepOrder"]
        assert _tree(tmp_path) == seeded, "reading the control wrote state"

    _forget_state(tmp_path)
    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {ONBOARDING_COMPLETED_KEY: "2026-11-04T10:00:00+00:00"})
    with _client(tmp_path) as client:
        assert client.get("/api/onboarding/bootstrap").json()["bootstrap"]["consciousnessOffer"]["offered"] is False


def test_only_a_root_the_next_boot_would_mint_counts_as_no_state_yet(monkeypatch, tmp_path):
    """An owner Reset's pending witness is a first state to come: offered. A completed witness whose copies are lost
    and a copy that cannot be read are a state that exists: no step, and nothing is written by asking. (Old copies a
    pending Reset sets aside: ``test_copies_a_pending_reset_sets_aside_count_as_no_state_yet``.)"""
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
    bootstrap, so a control already on shows as on (with no step) and an unreadable state stays unknown."""
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
    _forget_state(tmp_path)
    minted.settings_path.unlink(missing_ok=True)
    _complete(minted, consciousness=False)
    assert not minted.saved().get(ONBOARDING_CONSCIOUSNESS_KEY)
    assert _boot_control() == (False, None)


def test_a_state_minted_after_setup_owes_the_first_meeting_and_one_minted_before_does_not(minted, tmp_path):
    """My human's press of Start (the completion fact) is what a first state is born owing: one first meeting
    (``supervisor.first_meeting``), whatever the switch said. A root minted before any setup owes none, and a setup
    finished over a state that already exists adds none."""
    from supervisor import state
    from supervisor.first_meeting import FIRST_MEETING_KEY

    _complete(minted, consciousness=False)
    read = state.init_state()
    assert read.values[FIRST_MEETING_KEY] == {"state": "owed", "pressed_at": minted.saved()[ONBOARDING_COMPLETED_KEY]}
    _forget_state(tmp_path)
    minted.settings_path.unlink(missing_ok=True)
    assert FIRST_MEETING_KEY not in state.init_state().values  # no setup recorded when it was minted
    _complete(minted, consciousness=True)
    assert FIRST_MEETING_KEY not in state.init_state().values  # setup over an existing state: nothing owed


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


def test_a_field_sent_where_a_state_exists_changes_nothing_and_a_stale_page_lifts_no_stop(minted, tmp_path):
    """Review r1 (Opus M2, Astra C1): a second wizard tab still showing the fresh default finishes after the owner
    stopped consciousness. A state exists, so the field is ignored: the stop and its source stand, whatever it says."""
    from ouroboros.consciousness import OWNER_STOP, _persist_stop

    _complete(minted, consciousness=True)
    assert _boot_control() == (True, None)
    assert _persist_stop(OWNER_STOP) == ""
    for body in ({"consciousness": True}, {"consciousness": False}, {}):  # the stale tab, and an older page
        _complete(minted, **body)
        assert _boot_control() == (False, OWNER_STOP)
    # An install that predates the step (born off, no fact) reads the same: setup never changes an existing state.
    _forget_state(tmp_path)
    minted.settings_path.unlink(missing_ok=True)
    _complete(minted)
    assert _boot_control() == (False, None)
    _complete(minted, consciousness=True)
    assert not minted.saved().get(ONBOARDING_CONSCIOUSNESS_KEY), "a field where a state exists wrote no fact"
    assert _boot_control() == (False, None)


def test_a_panic_pending_at_finish_on_a_fresh_root_holds_through_a_second_finish(minted, tmp_path, monkeypatch):
    """Review r1 (Fable F1, Astra C1): Panic pressed before setup finished, so its flag waits for the first state. The
    supervisor that Finish starts mints the state and receives Panic before the request goes on; Finish answers 200,
    the state is off with Panic's source, and a second Finish with the untouched default lifts nothing."""
    import ouroboros.gateway.settings as gw_settings
    from ouroboros.installation_stop import PANIC_RECEIPT, panic_input, remove_received_panic
    from ouroboros.server_control import PANIC_CONTROL_KEYS, _panic_controls
    from supervisor import state

    def supervisor_receiving_panic(_request, _settings):  # init_state, then auto_resume_after_restart's Panic branch
        state.init_state()
        identity = panic_input(tmp_path)
        if identity:
            state.update_state(_panic_controls, confirm=PANIC_CONTROL_KEYS)
            state.update_state(lambda st: st.update({PANIC_RECEIPT: identity}))
            remove_received_panic(tmp_path, identity)
        return True

    monkeypatch.setattr(gw_settings, "_start_supervisor_if_needed_for_request", supervisor_receiving_panic)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    assert _complete(minted, consciousness=True)["ok"] is True
    assert not (tmp_path / "state" / "panic_stop.flag").exists(), "the supervisor received Panic during Finish"
    assert _boot_control() == (False, "panic")
    assert _complete(minted, consciousness=True)["ok"] is True  # the same wizard, finished again
    assert _boot_control() == (False, "panic")


def test_an_off_after_a_saved_on_is_the_first_states_choice_while_it_is_to_come(minted, tmp_path):
    """Review r1 (Fable F3): while no state exists the fact is the pending choice, so a later completion with the switch
    turned off clears the "on" an earlier one saved; the other direction, in the same home, turns it on again."""
    _complete(minted, consciousness=True)
    _complete(minted, consciousness=False)
    assert minted.saved()[ONBOARDING_CONSCIOUSNESS_KEY] == ""
    assert _boot_control() == (False, None)
    _forget_state(tmp_path)
    _complete(minted, consciousness=True)
    assert minted.saved()[ONBOARDING_CONSCIOUSNESS_KEY] == "on"
    assert _boot_control() == (True, None)


def test_copies_a_pending_reset_sets_aside_count_as_no_state_yet(minted, monkeypatch, tmp_path):
    """Review r1 (Astra M5): an old writer left valid copies of the previous identity after an owner Reset. Boot sets
    them aside and mints, so setup offers the step, nothing is written by asking, and the first state is born by the
    choice; a copy of the pending identity itself is adopted, so it is no first state."""
    from ouroboros.gateway import onboarding_host
    from supervisor.state_initialization import first_state_pending, mark_pending

    _complete(minted)
    assert _boot_control() == (False, None)
    shutil.copy(tmp_path / "state" / "state.json", tmp_path / "state" / "state.last_good.json")
    minted.settings_path.unlink(missing_ok=True)
    identity = mark_pending(tmp_path, origin="owner_reset")
    monkeypatch.setattr(onboarding_host, "load_settings", lambda: {})
    seeded = _tree(tmp_path)
    with _client(tmp_path) as client:
        offer = client.get("/api/onboarding/bootstrap").json()["bootstrap"]["consciousnessOffer"]
    assert (offer["offered"], offer["enabled"]) == (True, None)
    assert _tree(tmp_path) == seeded, "asking set copies aside or wrote the witness"
    _complete(minted, consciousness=True)
    assert _boot_control() == (True, None)
    assert json.loads((tmp_path / "state" / "state.json").read_text(encoding="utf-8"))["initialization_id"] == identity
    # The other direction: a copy of the pending identity itself is adopted, and an unreadable copy refuses.
    document = json.loads((tmp_path / "state" / "state.json").read_text(encoding="utf-8"))
    pending = mark_pending(tmp_path, origin="owner_reset")
    (tmp_path / "state" / "state.json").write_text(json.dumps({**document, "initialization_id": pending}), encoding="utf-8")
    assert first_state_pending(tmp_path) is False
    (tmp_path / "state" / "state.json").unlink()
    (tmp_path / "state" / "state.json").mkdir()
    assert first_state_pending(tmp_path) is False


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
    _forget_state(tmp_path)
    (tmp_path / "state" / "panic_stop.flag").unlink()
    assert _boot_control() == (True, None)
    assert _persist_stop(OWNER_STOP) == ""
    assert _boot_control() == (False, OWNER_STOP)
