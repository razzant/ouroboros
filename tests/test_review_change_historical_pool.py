"""Public change/preflight collection keeps the original operation across pool edits."""
import json
import threading
from pathlib import Path

import pytest

from ouroboros import review_custody
from ouroboros.observability import read_call_payload
from ouroboros.review_ledger import load_record, record_path
from ouroboros.review_state import load_state, make_repo_key
from ouroboros.tools.preflight_review import _handle_preflight_review
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_change import run_review_change
from tests._review_session_route_shared import (  # noqa: F401
    FakeGateway, _owned_gateway_uses_each_test_transport, _terminal_detail, fake_route,
)
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_review_change_end_to_end import staged_body  # noqa: F401


@pytest.mark.parametrize("surface", ["change", "preflight"])
@pytest.mark.parametrize("catalog_change", ["full", "removed", "emptied", "malformed", "ledger_gap"])
def test_public_collection_keeps_paid_request_and_operation(
        staged_body, fake_route, tmp_path, monkeypatch, surface, catalog_change):  # noqa: F811
    repo = Path(staged_body["repo"])
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "collection", task_id="original-review")
    ctx.task_attempt = 1
    set_review_pool(monkeypatch, pool_roster(pool_seat(
        "original-seat", "fake-review=fake-small", kind="agent_session", access="readonly")))
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    get_run, settle = FakeGateway.get_run, review_custody._settle_review_attempt
    timeout = review_custody._late_or_timeout_actor
    FakeGateway.detail = _terminal_detail("[]", conformance="passed")

    def delayed(self, *args, **kwargs):
        entered.set()
        assert release.wait(10)
        return get_run(self, *args, **kwargs)

    def completed(*args, **kwargs):
        settle(*args, **kwargs)
        settled.set()

    def admitted_timeout(*args, **kwargs):
        assert entered.wait(10)
        return timeout(*args, **kwargs)

    monkeypatch.setattr(FakeGateway, "get_run", delayed)
    monkeypatch.setattr(review_custody, "_settle_review_attempt", completed)
    monkeypatch.setattr(review_custody, "_late_or_timeout_actor", admitted_timeout)
    monkeypatch.setattr(review_custody, "_logical_timeout", lambda *_: .1)

    def call():
        if surface == "preflight":
            return json.loads(_handle_preflight_review(ctx, reviewer="original-seat", goal="same paid task"))
        return run_review_change(ctx, root="system_repo", subject="index", goal="same paid task")

    def attempt():
        return load_state(ctx.drive_root).filter_attempts(
            repo_key=make_repo_key(repo), tool_name="review_change", task_id=ctx.task_id)[-1]

    try:
        original = call()
        assert entered.is_set() and original["state"] == "pending", original
        [seat] = original["rows"]
        operation = attempt().triad_raw_results[0]["operation_id"]
        before = read_call_payload(ctx.drive_root, task_id=ctx.task_id, call_id=operation + "_prompt")
        [body] = [body for gateway in FakeGateway.instances for body in gateway.start_requests]
        assert body["mode"] == "ask" and body["access"] == "readonly"
        release.set()
        assert settled.wait(10)
        # Exercise a cold collector, not a live in-process memoized answer.
        review_custody._ACTIVE.clear()
        review_custody._NO_RESEND.clear()
        if catalog_change == "ledger_gap":
            assert attempt().triad_raw_results[0]["review_plan"]["parts"] == ["change", "coupling"]
            record_path(ctx.drive_root, original["record_id"]).unlink()
        if catalog_change == "malformed":
            monkeypatch.setenv("OUROBOROS_SUBAGENTS", "{broken")
        elif catalog_change == "emptied":
            # An emptied pool (now the Main seat) shapes only a NEW wave; the paid round is still collected.
            set_review_pool(monkeypatch, pool_roster())
        else:
            set_review_pool(monkeypatch, pool_roster(pool_seat(
                "original-seat" if catalog_change == "full" else "replacement-seat",
                "fake-review=different-model", kind="agent_session", access="full")))
        if surface == "preflight":
            from ouroboros.tools.commit_gate import commit_preflight_choice_error
            assert not commit_preflight_choice_error("original-seat", skipped=False, continuation=False,
                                                      ctx=ctx, goal="same paid task")
            assert commit_preflight_choice_error("missing-seat", skipped=False, continuation=False,
                                                   ctx=ctx, goal="a different new task")
        monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway",
                            lambda: pytest.fail("collection created a new paid session"))
        collected = call()
        assert collected["record_id"] == original["record_id"]
        assert collected["state"] == "settled", collected
        [row] = collected["rows"]
        assert row["seat_id"] == "original-seat"
        assert attempt().triad_raw_results[0]["operation_id"] == operation
        assert row["requested"]["model"] == seat["requested"]["model"]
        assert row["parts"] == seat["parts"] == ["change", "coupling"]
        assert [body] == [body for gateway in FakeGateway.instances for body in gateway.start_requests]
        assert read_call_payload(ctx.drive_root, task_id=ctx.task_id, call_id=operation + "_prompt") == before
        assert load_record(ctx.drive_root, original["record_id"])["revision"] >= (1 if catalog_change == "ledger_gap" else 2)
    finally:
        release.set()
        if entered.is_set():
            assert settled.wait(10)


@pytest.mark.parametrize("binding", ["recorded", "lost"])
@pytest.mark.parametrize("collection", ["repeat", "reference"])
def test_rejoin_keeps_the_paid_waves_history_binding_not_todays_heads(
        staged_body, fake_route, tmp_path, monkeypatch, binding, collection):  # noqa: F811
    """A late collection restores the history the paid wave was prepared with.

    A newer record of the same task and checkout exists by then; today's heads would
    name it as the old wave's predecessor. The rejoin reads the wave's saved dispute
    input, or, when that binding is gone, records an explicit gap with no ancestry.
    """
    import dataclasses

    from ouroboros.review_ledger import new_record_id, revise_record, write_record
    from ouroboros.review_state import update_state

    repo = Path(staged_body["repo"])
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "collection", task_id="original-review")
    ctx.task_attempt = 1
    set_review_pool(monkeypatch, pool_roster(pool_seat(
        "original-seat", "fake-review=fake-small", kind="agent_session", access="readonly")))
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    get_run, settle = FakeGateway.get_run, review_custody._settle_review_attempt
    timeout = review_custody._late_or_timeout_actor
    FakeGateway.detail = _terminal_detail("[]", conformance="passed")

    def delayed(self, *args, **kwargs):
        entered.set()
        assert release.wait(10)
        return get_run(self, *args, **kwargs)

    def completed(*args, **kwargs):
        settle(*args, **kwargs)
        settled.set()

    def admitted_timeout(*args, **kwargs):
        assert entered.wait(10)
        return timeout(*args, **kwargs)

    monkeypatch.setattr(FakeGateway, "get_run", delayed)
    monkeypatch.setattr(review_custody, "_settle_review_attempt", completed)
    monkeypatch.setattr(review_custody, "_late_or_timeout_actor", admitted_timeout)
    monkeypatch.setattr(review_custody, "_logical_timeout", lambda *_: .1)

    def call():
        return run_review_change(ctx, root="system_repo", subject="index", goal="same paid task")

    def attempts():
        return load_state(ctx.drive_root).filter_attempts(
            repo_key=make_repo_key(repo), tool_name="review_change", task_id=ctx.task_id)

    try:
        original = call()
        assert entered.is_set() and original["state"] == "pending", original
        paid = load_record(ctx.drive_root, original["record_id"])
        saved_input = paid["brief"]["dispute_input"]
        assert isinstance(saved_input, dict) and saved_input["previous_records"] == []
        release.set()
        assert settled.wait(10)
        review_custody._ACTIVE.clear()
        review_custody._NO_RESEND.clear()
        # A newer settled record of the same task and checkout: today's only head.
        newer_id = new_record_id()
        write_record(ctx.drive_root, {
            **paid, "record_id": newer_id, "revision": 1, "state": "settled",
            "brief": {**paid["brief"], "goal": "a newer round",
                      "dispute_input": {"previous_records": [original["record_id"]], "gaps": []}},
            "fingerprints": {**paid["fingerprints"], "retry_key": "a-newer-round", "reuse_key": "a-newer-round"}})
        [first] = attempts()
        update_state(ctx.drive_root, lambda state: state.record_attempt(dataclasses.replace(
            first, attempt=first.attempt + 1, review_record_id=newer_id, review_retry_key="a-newer-round",
            status="reviewed", late_result_pending=False, triad_raw_results=[])))
        from ouroboros.review_history import review_dispute_history
        heads = review_dispute_history(drive_root=ctx.drive_root, repo_root=repo, task_id=ctx.task_id,
                                       exclude_record_id=original["record_id"])["record_heads"]
        assert heads == [newer_id], "the hazard needs today's heads to name the newer record"
        if binding == "lost":
            def _drop(payload):
                payload["brief"] = {k: v for k, v in payload["brief"].items() if k != "dispute_input"}
                return payload
            revise_record(ctx.drive_root, original["record_id"], _drop)
        starts = [body for gateway in FakeGateway.instances for body in gateway.start_requests]
        monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway",
                            lambda: pytest.fail("collection created a new paid session"))
        if collection == "reference":
            from ouroboros.tools.review_collection import collect_change
            collected = collect_change(ctx, {"record_id": original["record_id"]})
        else:
            collected = call()
        assert collected["record_id"] == original["record_id"] and collected["state"] == "settled", collected
        record = load_record(ctx.drive_root, original["record_id"])
        restored = record["brief"]["dispute_input"]
        assert newer_id not in restored["previous_records"]
        assert record["fingerprints"]["retry_key"] == paid["fingerprints"]["retry_key"]
        assert starts == [body for gateway in FakeGateway.instances for body in gateway.start_requests]
        if binding == "recorded":
            assert restored == saved_input
        else:
            assert restored["previous_records"] == []
            assert [gap["reason"] for gap in restored["gaps"]] == ["paid wave has no saved dispute-input binding"]
    finally:
        release.set()
        if entered.is_set():
            assert settled.wait(10)


def test_a_refused_call_on_a_fresh_drive_writes_no_review_state(staged_body, tmp_path, monkeypatch):  # noqa: F811
    """The paid-round lookup runs before the pool is consulted; with no review state
    there is no open round to find, so a refused call leaves the drive as it was. The
    refusal is a reviewer the catalog does not hold (an empty pool reviews on the Main
    seat, so it no longer refuses by itself). The opposite direction (state holding an
    open round is still collected under an emptied pool) is
    ``test_public_collection_keeps_paid_request_and_operation``."""
    from ouroboros.review_state import _STATE_RELPATH
    from ouroboros.tools.review_change import _handle_review_change

    ctx = ToolContext(repo_dir=Path(staged_body["repo"]), drive_root=tmp_path / "fresh", task_id="fresh-review")
    set_review_pool(monkeypatch, pool_roster())
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway",
                        lambda: pytest.fail("a refused call started a session"))
    text = _handle_review_change(ctx, root="system_repo", subject="index", goal="nothing to review with",
                                 reviewers=["ghost"])
    assert "TOOL_ARG_ERROR (review_change)" in text and "is not an enabled catalog row" in text, text
    assert not (Path(ctx.drive_root) / _STATE_RELPATH).exists()
