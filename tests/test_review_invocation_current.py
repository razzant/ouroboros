"""Review collection consumes the invocation projection without forensic replay."""
from __future__ import annotations

import pytest

from ouroboros import delegate_custody as c
from ouroboros import review_operation
from ouroboros.review_execution import ReviewRouteUnavailable
from ouroboros.review_session_custody import validate_producer_invocation
from tests._review_session_route_shared import _run_session_directly
from tests.test_custody_invocation_record import _legacy_rows, _publish, _request, _started
from tests.test_review_operation_collection import (
    PASS, SURFACE, TASK, _collect, _ctx, _observe, _session_run, _verdict, forbid_effects,  # noqa: F401
)

pytestmark = pytest.mark.usefixtures("forbid_effects")


def _history(root, *, multiple=False):
    request = _request(task_id=TASK, root_task_id=TASK, surface=SURFACE, slot_id="s", operation_id="op-session")
    rows = [request, {"type": c.START_FAILED, "invocation_id": "inv", "definite": True},
            {**request, "request": {"prompt": "repeated"}}]
    if multiple:
        rows.append({**_started("a"), "task_id": TASK})
    rows.append({**_started("b"), "task_id": TASK})
    for number, row in enumerate(rows):
        _publish(root, row, number)


@pytest.mark.parametrize("legacy", ["modern", "proven", "gap"])
@pytest.mark.parametrize("tokenless", [False, True])
def test_collection_never_replays_custody_chain(tmp_path, monkeypatch, legacy, tokenless):
    _history(tmp_path, multiple=legacy == "gap")
    if legacy != "modern":
        _legacy_rows(tmp_path, lose_bound_request=True)
    run = _session_run()
    if not tokenless:
        run["actors"][0]["pending_invocation_id"] = "inv"
    calls = _observe(monkeypatch, {"summary": {"state": "succeeded"},
                                   "primaryOutput": {"text": PASS, "truncated": False}})
    replays = []

    def forbidden(*args, **kwargs):
        replays.append(1)
        raise AssertionError("collection replayed the custody chain")

    monkeypatch.setattr(c, "custody_rows", forbidden)
    monkeypatch.setattr(c, "_iter_rows", forbidden)
    actor, fact = _collect(run, tmp_path, _ctx(tmp_path))
    gap = legacy == "gap" or (tokenless and legacy != "modern")
    if gap:
        assert actor["failure_code"] == "review_custody_legacy_gap"
        assert actor["usage"]["custody_state"] == "legacy_gap"
        assert fact == "unavailable" and calls == []
    else:
        assert actor["operation_state"] != "custody_lost"
        assert actor["usage"]["delegated_run_id"] == "b"
        assert _verdict(actor) == "PASS"
        assert fact == "collected"
        assert calls == [("handshake",), ("get_run", "b"), ("close",)]
    assert replays == []


@pytest.mark.parametrize("multiple", [False, True])
def test_completed_producer_uses_last_started_and_names_legacy_gap(tmp_path, multiple):
    _history(tmp_path, multiple=multiple)
    expected = {"task_id": TASK, "root_task_id": TASK, "surface": SURFACE,
                "slot_id": "s", "operation_id": "op-session"}
    payload = {"usage": {"delegated_run_id": "b"}}
    validate_producer_invocation(tmp_path, "inv", expected, payload, {}, error_source=False)
    _legacy_rows(tmp_path, lose_bound_request=True)
    if multiple:
        with pytest.raises(ValueError, match="legacy_gap"):
            validate_producer_invocation(tmp_path, "inv", expected, payload, {}, error_source=False)
    else:
        validate_producer_invocation(tmp_path, "inv", expected, payload, {}, error_source=False)


@pytest.mark.parametrize("surface", ["scope_review", "skill_review"])
def test_session_retry_never_replaces_a_legacy_gap(tmp_path, surface):
    _history(tmp_path, multiple=True)
    _legacy_rows(tmp_path)
    state = {"pending_invocation_id": "inv"}
    with pytest.raises(ReviewRouteUnavailable) as caught:
        _run_session_directly(tmp_path, retry_state=state, surface=surface)
    assert caught.value.code == "review_custody_legacy_gap"
    assert state == {"pending_invocation_id": "inv"}


def test_operation_lookup_keeps_first_matching_repeated_request(tmp_path):
    # The first request of an invocation can belong to another operation. The
    # old chain reader still selected it when a later REQUESTED matched this one.
    _publish(tmp_path, _request(operation_id="earlier"), 0)
    _publish(tmp_path, _request(), 1)
    _publish(tmp_path, _request("other"), 2)
    slot = type("Slot", (), {"slot_id": "slot"})()
    answer = review_operation._invocation_for_operation(tmp_path, slot, "operation", {})
    oracle = c.invocation_record(tmp_path, "inv", rows=list(c.custody_rows(tmp_path)))
    assert answer == {**oracle, "invocation_id": "inv"}
