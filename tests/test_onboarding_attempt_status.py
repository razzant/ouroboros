"""GET /api/onboarding/complete?attempt=<id> — reading back ONE completion attempt (#682).

The wizard makes the attempt id before it sends the completion, so it can still ask about
its own save when the answer is lost or outlives the writer bound (503
``settings_save_timeout``, ``saved: null``). The readiness probe cannot answer that: an
install whose older provider is still on disk passes it whatever happened to this save.

The server answers from two facts: the settings document names the attempt whose
completion write it holds (same write as the completion fact), and this process remembers
how the runs it served ended, around the writer's own ``CommitBoundary``. These tests drive
the REAL completion transaction against an isolated settings file.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.settings_setup_contract import ONBOARDING_ATTEMPT_KEY, ONBOARDING_COMPLETED_KEY
from tests.test_onboarding_complete_endpoint import WIZARD_PAYLOAD
from tests.test_onboarding_complete_endpoint import onboarding as onboarding  # noqa: F401  (the real-write fixture)

ATTEMPT_A = "a" * 32
ATTEMPT_B = "b" * 32
ATTEMPT_C = "c" * 32
API_ONLY = {**WIZARD_PAYLOAD, "subscriptionsConnected": False}


@pytest.fixture
def wizard(onboarding, monkeypatch):  # noqa: F811
    """The completion POST and the attempt GET over the same isolated settings file."""
    from ouroboros.gateway import onboarding as gw_onboarding
    from ouroboros.gateway import onboarding_host

    monkeypatch.setattr(onboarding_host, "_runs", OrderedDict())  # this process's memory starts empty
    app = Starlette(routes=[
        Route("/api/onboarding/complete", endpoint=gw_onboarding.api_onboarding_complete, methods=["POST"]),
        Route("/api/onboarding/complete", endpoint=onboarding_host.api_onboarding_attempt, methods=["GET"]),
    ])
    app.state.drive_root = onboarding.settings_path.parent
    app.state.repo_dir = onboarding.settings_path.parent / "repo"
    with TestClient(app) as client:
        onboarding.wizard = client
        yield onboarding


def _post(wizard, attempt, **extra):
    return wizard.wizard.post("/api/onboarding/complete", json={**API_ONLY, "attempt_id": attempt, **extra})


def _read(wizard, attempt):
    response = wizard.wizard.get("/api/onboarding/complete", params={"attempt": attempt})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()


def _wedge(monkeypatch, target, name):
    """Hold ``target.name`` until released; ``started``/``finished`` say where the episode is."""
    original = getattr(target, name)
    gate = {"release": threading.Event(), "started": threading.Event(), "finished": threading.Event()}

    def held(*args, **kwargs):
        gate["started"].set()
        gate["release"].wait(10)
        try:
            return original(*args, **kwargs)
        finally:
            gate["finished"].set()

    monkeypatch.setattr(target, name, held)
    return gate


def _wait_until(predicate, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "the abandoned episode did not finish"
        time.sleep(0.02)


def test_the_completion_write_carries_the_attempt_and_its_receipt_reads_back(wizard):
    response = _post(wizard, ATTEMPT_A)
    assert response.status_code == 200, response.text
    saved = wizard.saved()
    assert saved[ONBOARDING_ATTEMPT_KEY] == ATTEMPT_A and saved[ONBOARDING_COMPLETED_KEY]
    # The receipt is the attempt's own success envelope, byte-identical to the reply it replaces.
    assert _read(wizard, ATTEMPT_A) == {"saved": True, "receipt": response.json()}


def test_a_provider_saved_before_this_attempt_never_confirms_it(wizard):
    """The old Check status accepted the readiness 204. Here an existing install is READY (its earlier
    completion and provider are on disk), and the attempt that never reached the writer stays unknown."""
    from ouroboros.server_runtime import has_startup_ready_provider

    assert _post(wizard, ATTEMPT_A).status_code == 200
    assert has_startup_ready_provider(wizard.saved()), "the readiness probe would answer 204 here"
    assert _read(wizard, ATTEMPT_B) == {"saved": None}
    assert wizard.saved()[ONBOARDING_ATTEMPT_KEY] == ATTEMPT_A, "reading wrote nothing"


def test_a_save_that_outlives_its_wait_reads_unknown_then_stored_then_its_receipt(wizard, monkeypatch):
    """Before the commit the read stays unknown without claiming failure; after it the save is stored
    while its post-commit steps still run; only the finished episode yields the receipt."""
    import ouroboros.config as config
    import ouroboros.gateway.onboarding as gw_onboarding
    import ouroboros.gateway.settings as gw_settings

    monkeypatch.setattr(config, "get_settings_document_lock_timeout_sec", lambda: 0.2)
    before_commit = _wedge(monkeypatch, gw_onboarding, "_owner_write_settings")  # the one completion write
    after_commit = _wedge(monkeypatch, gw_settings, "_apply_settings_save_side_effects")
    try:
        response = _post(wizard, ATTEMPT_A, OUROBOROS_RUNTIME_MODE="pro")
        assert response.status_code == 503 and response.json()["saved"] is None
        assert response.json()["code"] == "settings_save_timeout"
        assert before_commit["started"].wait(10) and not wizard.settings_path.exists()
        assert _read(wizard, ATTEMPT_A) == {"saved": None}

        before_commit["release"].set()
        assert after_commit["started"].wait(10)
        assert wizard.saved()[ONBOARDING_ATTEMPT_KEY] == ATTEMPT_A
        assert _read(wizard, ATTEMPT_A) == {"saved": True}, "stored; completion and restart not yet confirmed"
    finally:
        before_commit["release"].set()
        after_commit["release"].set()
    assert after_commit["finished"].wait(10)
    _wait_until(lambda: "receipt" in _read(wizard, ATTEMPT_A))
    body = _read(wizard, ATTEMPT_A)
    assert body["saved"] is True and body["receipt"]["ok"] is True
    # The confirmed restart requirement is the receipt's: this process keeps the mode it booted with.
    assert config.get_runtime_mode() != "pro"
    assert body["receipt"]["runtime_mode"] == "pro" and body["receipt"]["restart_required"] is True


def test_retry_then_the_late_first_attempt_cannot_confirm_the_retry(wizard, monkeypatch):
    """A stays in its writer; the retry B is refused while A holds the document, and A landing
    afterwards confirms A only. A later save from another tab replaces A in turn."""
    import ouroboros.config as config
    import ouroboros.gateway.onboarding as gw_onboarding

    monkeypatch.setattr(config, "get_settings_document_lock_timeout_sec", lambda: 0.2)
    first = _wedge(monkeypatch, gw_onboarding, "_owner_write_settings")  # A holds the document, uncommitted
    try:
        assert _post(wizard, ATTEMPT_A).json()["saved"] is None
        assert first["started"].wait(10)
        retry = _post(wizard, ATTEMPT_B, OUROBOROS_RUNTIME_MODE="pro")
        assert retry.status_code == 503, retry.text
        assert retry.json() == {"error": retry.json()["error"], "saved": False, "code": "settings_busy"}
    finally:
        first["release"].set()
    assert first["finished"].wait(10)
    _wait_until(lambda: "receipt" in _read(wizard, ATTEMPT_A))
    assert wizard.saved()["OUROBOROS_RUNTIME_MODE"] == "advanced", "A's draft landed, not B's"
    assert _read(wizard, ATTEMPT_B) == {"saved": False, "error": retry.json()["error"], "code": "settings_busy"}

    other_tab = _post(wizard, ATTEMPT_C, OUROBOROS_RUNTIME_MODE="pro")
    assert other_tab.status_code == 200, other_tab.text
    assert _read(wizard, ATTEMPT_A) == {"saved": None}, "A's write is no longer the settings document"
    assert _read(wizard, ATTEMPT_C)["receipt"]["runtime_mode"] == "pro"


def test_a_refusal_before_the_write_reads_back_as_nothing_saved(wizard, monkeypatch):
    import ouroboros.config as config

    monkeypatch.setattr(config, "_acquire_settings_lock", lambda timeout=2.0: None)
    response = _post(wizard, ATTEMPT_A)
    assert response.status_code == 503 and response.json()["code"] == "settings_locked"
    assert _read(wizard, ATTEMPT_A) == {"saved": False, "error": response.json()["error"], "code": "settings_locked"}
    assert not wizard.settings_path.exists()


def test_a_failure_after_the_commit_reads_back_as_saved_with_its_stage(wizard, monkeypatch):
    import ouroboros.gateway.settings as gw_settings

    def broken(*_args, **_kwargs):
        raise RuntimeError("extension reload failed")

    monkeypatch.setattr(gw_settings, "_apply_settings_save_side_effects", broken)
    response = _post(wizard, ATTEMPT_A)
    assert response.status_code == 500 and response.json()["saved"] is True
    assert _read(wizard, ATTEMPT_A) == {"saved": True, "error": response.json()["error"], "post_commit_failed": "hot-reload"}


def test_an_attempt_this_process_did_not_serve_stays_stored_without_a_receipt(wizard, monkeypatch):
    """After a restart the document still names the attempt, but no receipt is invented for it."""
    from ouroboros.gateway import onboarding_host

    assert _post(wizard, ATTEMPT_A).status_code == 200
    monkeypatch.setattr(onboarding_host, "_runs", OrderedDict())
    assert _read(wizard, ATTEMPT_A) == {"saved": True}


def test_attempt_ids_are_validated_and_a_save_without_one_clears_the_older_id(wizard):
    bad = _post(wizard, "not an id")
    assert bad.status_code == 400 and bad.json()["saved"] is False
    assert not wizard.settings_path.exists()
    for query in ({}, {"attempt": ""}, {"attempt": "short"}, {"attempt": "x" * 129}):
        assert wizard.wizard.get("/api/onboarding/complete", params=query).status_code == 400, query

    assert _post(wizard, ATTEMPT_A).status_code == 200
    legacy = wizard.wizard.post("/api/onboarding/complete", json=API_ONLY)  # an older client sends no id
    assert legacy.status_code == 200, legacy.text
    assert wizard.saved()[ONBOARDING_ATTEMPT_KEY] == ""
    assert _read(wizard, ATTEMPT_A) == {"saved": None}, "the document no longer holds A's write"


def test_the_attempt_id_is_disk_only_and_no_generic_save_authors_or_clears_it(monkeypatch):
    import ouroboros.config as config
    from ouroboros.gateway.settings import _merge_settings_payload

    assert ONBOARDING_ATTEMPT_KEY in config.ENDPOINT_AUTHORED_SETTINGS
    assert ONBOARDING_ATTEMPT_KEY not in config.settings_env_keys()
    assert _merge_settings_payload({ONBOARDING_ATTEMPT_KEY: ATTEMPT_A}, {ONBOARDING_ATTEMPT_KEY: ""})[
        ONBOARDING_ATTEMPT_KEY] == ATTEMPT_A
    assert not _merge_settings_payload({}, {ONBOARDING_ATTEMPT_KEY: ATTEMPT_B}).get(ONBOARDING_ATTEMPT_KEY)
    monkeypatch.setenv(ONBOARDING_ATTEMPT_KEY, ATTEMPT_C)
    assert json.dumps(config.load_settings()).count(ATTEMPT_C) == 0, "an environment value never becomes the receipt"
