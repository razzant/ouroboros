"""Explicit operator consumer over synthetic stores and fake HTTP only."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import openrouter_cost as cost, usage_accounting as usage, usage_store
from ouroboros.task_results import load_task_result, write_task_result
from scripts import reconcile_openrouter_cost as command

pytestmark = pytest.mark.serial  # Shared queue/capture globals and subprocess export consumers.
TARGET = {"provider": "openrouter", "base_url": "https://router.example/api/v1", "api_key": "test-original-key"}


def current(root, attempt_id):
    with usage_store.read(root) as txn:
        return txn.attempt(attempt_id)


@pytest.fixture
def env(tmp_path, monkeypatch):
    from supervisor import queue
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **k: False)
    monkeypatch.setattr(command, "_target", lambda: TARGET)
    monkeypatch.setattr("requests.get", lambda *a, **k: pytest.fail("unexpected HTTP"))
    return tmp_path


def attempt(root, *, state="settled", generation="generation-selected", task="child"):
    request = usage.AttemptRequest(
        model="vendor/unseen-model", provider="openrouter", drive_root=root,
        task_id=task, root_task_id="root", parent_task_id="root",
        reservation_usd=5.0, global_limit_usd=100,
        provider_receipt_binding=cost.binding_for_target(TARGET))
    payload = {"id": generation, "choices": [{"message": {"content": "kept answer"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 37, "completion_tokens": 11, "cached_tokens": 9}}
    if state == "settled":
        usage.execute_physical_attempt(request, lambda: payload,
                                       extractor=lambda p: (p["usage"], 2.0, False))
    else:
        def fail():
            usage.bind_provider_generation(generation)
            raise RuntimeError("physical failure")
        with pytest.raises(RuntimeError, match="physical failure"):
            usage.execute_physical_attempt(request, fail)
    attempt_id = usage.last_physical_attempt_capture().attempt_id
    for task_id in ("root", task):
        write_task_result(root, task_id, "cancelled", result="kept answer", root_task_id="root",
                          parent_task_id="root" if task_id != "root" else "")
    return current(root, attempt_id)


def receipt(root, row, price=0.25):
    return cost.retain_generation_receipt(root, row, {"data": {
        "id": row["provider_receipt_binding"]["generation_id"], "total_cost": price, "usage": 0.0}})


def invoke(root, row, capsys, *flags):
    code = command.main(["--data-root", str(root), "--attempt-id", row["attempt_id"], *flags])
    return code, json.loads(capsys.readouterr().out)["outcomes"][0]


def test_configured_target_reads_existing_key_without_settings_migration(tmp_path, monkeypatch):
    from ouroboros import config
    settings_path = tmp_path / "settings.json"
    original = b'{"OPENROUTER_API_KEY": "synthetic-original-key", "unrelated": "preserved"}'
    settings_path.write_bytes(original)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config, "load_settings", lambda: pytest.fail("no settings migrations in operator lookup"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-environment-key")
    target = command._target()
    assert target["provider"] == "openrouter" and target["api_key"] == "synthetic-original-key"
    assert target["base_url"] == "https://openrouter.ai/api/v1"
    assert settings_path.read_bytes() == original


def test_default_missing_store_never_creates_or_imports_history(tmp_path, capsys):
    assert command.main(["--data-root", str(tmp_path), "--attempt-id", "missing"]) == 1
    assert json.loads(capsys.readouterr().out)["outcomes"] == [
        {"attempt_id": "missing", "status": "store_unavailable"}]
    assert list(tmp_path.iterdir()) == []


def test_default_does_not_reimport_store_removed_during_inspection(env, capsys, monkeypatch):
    row = attempt(env)
    store_path = env / usage_store.STORE_REL
    held = usage_store.hold
    def removed_before_open(root, **kwargs):
        store_path.rename(store_path.with_suffix(".gone"))
        return held(root, **kwargs)
    monkeypatch.setattr(usage_store, "hold", removed_before_open)
    code, result = invoke(env, row, capsys)
    assert code == 1 and result["status"] == "error"
    assert not store_path.exists()


def test_apply_does_not_reimport_store_removed_after_inspection(env, capsys, monkeypatch):
    row = attempt(env)
    receipt(env, row)
    store_path = env / usage_store.STORE_REL
    locked = usage._locked
    def removed_before_apply(root, **kwargs):
        store_path.rename(store_path.with_suffix(".gone"))
        return locked(root, **kwargs)
    monkeypatch.setattr(usage, "_locked", removed_before_apply)
    code, result = invoke(env, row, capsys, "--apply")
    assert code == 1 and result["status"] == "error"
    assert not store_path.exists()


@pytest.mark.parametrize("price", [0.0, 3.25])
def test_default_and_apply_have_no_http_and_preserve_price_only_facts(env, capsys, price):
    row = attempt(env)
    receipt(env, row, price)
    code, inspected = invoke(env, row, capsys)
    assert code == 0 and inspected["status"] == "inspect" and inspected["receipt_cost_usd"] == price
    assert current(env, row["attempt_id"]) == row
    code, applied = invoke(env, row, capsys, "--apply")
    assert code == 0 and applied["status"] == "applied"
    final = current(env, row["attempt_id"])
    assert final["cost_usd"] == price and final["cost_final"] is True
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "model", "root_task_id", "parent_task_id"):
        assert final[key] == row[key]
    with usage_store.read(env) as txn:
        assert txn.open_attempts() == []  # Price uncertainty never enlarges ownership.
    code, duplicate = invoke(env, row, capsys, "--apply")
    assert code == 0 and duplicate["status"] == "duplicate"


def test_fetch_is_one_explicit_get_and_does_not_imply_apply(env, capsys, monkeypatch):
    row = attempt(env)
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        with usage_store.hold(env):
            pass  # No money transaction spans HTTP.
        return SimpleNamespace(status_code=200, headers={}, close=lambda: None,
                               json=lambda: {"data": {"id": "generation-selected", "total_cost": 0, "usage": 0.0}})
    monkeypatch.setattr("requests.get", get)
    code, fetched = invoke(env, row, capsys, "--fetch", "--attempt-id", row["attempt_id"])
    assert code == 0 and fetched["status"] == "retained" and len(calls) == 1
    assert calls[0][0] == TARGET["base_url"] + "/generation"
    assert calls[0][1]["params"] == {"id": "generation-selected"}
    assert calls[0][1]["allow_redirects"] is False and calls[0][1]["timeout"] == (5, 15)
    assert current(env, row["attempt_id"]) == row
    # Restart at receipt-before-ledger window: even --fetch --apply reads the
    # retained source first, with the original key now unavailable.
    monkeypatch.setattr(command, "_target", lambda: pytest.fail("retained price needs no key"))
    code, result = invoke(env, row, capsys, "--fetch", "--apply")
    assert code == 0 and result["status"] == "applied" and len(calls) == 1


@pytest.mark.parametrize(("status", "payload", "expected"), [
    (200, {"data": {"id": "generation-selected", "total_cost": 0, "usage": 0.0}}, "applied"),
    (200, {"data": {"id": "different", "total_cost": 0}}, "id_mismatch"),
    (200, {"data": {"id": "generation-selected"}}, "missing_price"),
    (200, {"data": {"id": "generation-selected", "total_cost": True}}, "invalid_price"),
    (200, {"data": {"id": "generation-selected", "total_cost": -1}}, "invalid_price"),
    (200, {"data": {"id": "generation-selected", "total_cost": float("inf")}}, "invalid_price"),
    (404, {"error": {"code": 404}}, "not_found"),
    (401, {}, "unauthorized"), (403, {}, "forbidden"), (429, {}, "rate_limited"),
])
def test_fetch_apply_real_consumer_outcomes(env, capsys, monkeypatch, status, payload, expected):
    row = attempt(env, state="unresolved")
    monkeypatch.setattr("requests.get", lambda *a, **k: SimpleNamespace(
        status_code=status, headers={"Retry-After": "3600"}, json=lambda: payload, close=lambda: None))
    code, result = invoke(env, row, capsys, "--fetch", "--apply")
    assert result["status"] == expected
    final = current(env, row["attempt_id"])
    if expected == "applied":
        assert code == 0 and final["cost_usd"] == 0 and final["cost_final"] is True
        assert result["state"] == "settled" and result["previous_state"] == "unresolved"
        assert final["physical_failure"] == row["physical_failure"]
    else:
        assert code == 1 and final == row


def test_changed_key_refuses_get_and_foreign_receipt_refuses_application(env, capsys, monkeypatch):
    row = attempt(env)
    monkeypatch.setattr(command, "_target", lambda: {**TARGET, "api_key": "test-rotated-key"})
    code, result = invoke(env, row, capsys, "--fetch", "--apply")
    assert code == 1 and result["status"] == "credential_mismatch"
    peer = attempt(env, generation="different-generation", task="peer")
    foreign = receipt(env, peer)
    assert cost.apply_retained_receipt(env, row["attempt_id"], foreign)["status"] == "ineligible"
    assert current(env, row["attempt_id"]) == row


@pytest.mark.parametrize("block", ["live", "post_task"])
def test_owner_custody_limits_still_apply(env, capsys, monkeypatch, block):
    row = attempt(env)
    receipt(env, row)
    if block == "live":
        monkeypatch.setattr("supervisor.queue.task_has_live_ownership", lambda *a, **k: True)
    elif block == "post_task":
        write_task_result(env, "child", "cancelled", root_phase_checkpoint={"post_task_synthesis": "running"})
    code, result = invoke(env, row, capsys, "--fetch", "--apply")
    assert code == 1 and result["status"] in {"owner_unsettled", "post_task_work_open"}
    assert current(env, row["attempt_id"]) == row


@pytest.mark.parametrize("source", ["provider_test", "capability_probe"])
@pytest.mark.parametrize("price", [0.0, 0.125])
def test_real_system_probe_scope_can_fetch_then_apply(env, capsys, monkeypatch, source, price):
    from ouroboros import llm_probe

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(env))
    monkeypatch.setattr(usage, "_reservation_cost", lambda _request: 5.0)
    target = {**TARGET, "resolved_model": "vendor/unseen-model", "usage_model": "vendor/unseen-model"}
    sends, gets = [], []

    def send(payload):
        sends.append(payload)
        return {"id": "generation-probe", "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}

    with usage.capture_attempt_ids() as attempt_ids:
        if source == "provider_test":
            llm_probe.accounted_one_shot(target, {"model": target["resolved_model"], "max_tokens": 1,
                "messages": [{"role": "user", "content": "OK"}]}, send, source=source)
        else:
            remote = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **p: send(p))))
            remote.with_options = lambda **_kwargs: remote
            client = SimpleNamespace(_resolve_remote_target=lambda _model: target, _get_remote_client=lambda _target: remote)
            assert llm_probe.probe_oversized_context(client, target["resolved_model"], "OK")["ok"]
    assert len(attempt_ids) == len(sends) == 1
    row = current(env, attempt_ids[0])
    assert row["task_id"] == row["root_task_id"] == f"system:{source}"
    assert row["non_task_operation"] is True and row["state"] == "settled"
    assert row["cost_usd"] is None and row["cost_final"] is False

    def get(url, **kwargs):
        gets.append((url, kwargs))
        return SimpleNamespace(status_code=200, headers={}, close=lambda: None,
                               json=lambda: {"data": {"id": "generation-probe", "total_cost": price}})

    monkeypatch.setattr("requests.get", get)
    code, fetched = invoke(env, row, capsys, "--fetch")
    assert code == 0 and fetched["status"] == "retained", fetched
    assert current(env, row["attempt_id"]) == row
    monkeypatch.setattr(command, "_target", lambda: pytest.fail("retained system price needs no key or GET"))
    code, applied = invoke(env, row, capsys, "--fetch", "--apply")
    assert code == 0 and applied["status"] == "applied"
    assert len(sends) == len(gets) == 1
    final = current(env, row["attempt_id"])
    assert final["cost_usd"] == price and final["cost_final"] is True
    assert final["task_id"] == f"system:{source}"
    # Existing corrupt evidence is an error, never a missing receipt that
    # grants another lookup. System scopes retain the same CAS validation.
    cost.call_manifest_path(env, row["task_id"], f"physical_{row['attempt_id']}_openrouter_price").write_text("unreadable")
    code, corrupt = invoke(env, row, capsys, "--fetch", "--apply")
    assert code == 1 and corrupt["status"] == "error"
    assert len(gets) == 1 and current(env, row["attempt_id"]) == final


@pytest.mark.parametrize("snapshot", ["running", "missing", "stale", "invalid", "empty",
                                      "direct", "direct_incomplete", "direct_invalid", "direct_empty"])
def test_standalone_cli_observes_selected_data_root_custody(env, snapshot):
    from ouroboros.test_environment import isolated_environment
    from ouroboros.utils import utc_now_iso

    selected = env / "selected"
    row = attempt(selected)
    receipt(selected, row)
    snapshot_path = selected / "state" / "queue_snapshot.json"
    observed = {"ts": utc_now_iso(), "running": [], "pending": []}
    if snapshot == "running":
        observed["running"] = [{"id": "child", "task": {"id": "child"}}]
    elif snapshot == "stale":
        observed["ts"] = "2000-01-01T00:00:00Z"
    if snapshot != "missing":
        snapshot_path.write_text("unreadable" if snapshot == "invalid" else json.dumps(observed))
    if snapshot.startswith("direct"):
        direct = {"ts": utc_now_iso(), "roots": [{"task_id": "child"}] if snapshot == "direct" else [],
                  "incomplete": snapshot == "direct_incomplete"}
        (selected / "state" / "direct_roots.json").write_text(
            "unreadable" if snapshot == "direct_invalid" else json.dumps(direct))
    # A different configured root has the opposite observation. Imported
    # supervisor maps in this fresh process are empty and cannot prove death.
    child_env = isolated_environment(env / "cli", command.REPO_ROOT, source=os.environ)
    configured = env / "configured"
    (configured / "state").mkdir(parents=True)
    applicable = snapshot in {"empty", "direct_empty"}
    (configured / "state" / "queue_snapshot.json").write_text(json.dumps({
        "ts": utc_now_iso(), "running": [{"id": "child"}] if applicable else [], "pending": []}))
    (configured / "state" / "direct_roots.json").write_text(json.dumps({
        "ts": utc_now_iso(), "roots": [{"task_id": "child"}] if applicable else [], "incomplete": False}))
    child_env["OUROBOROS_DATA_DIR"] = str(configured)
    process = subprocess.run([sys.executable, str(command.REPO_ROOT / "scripts" / "reconcile_openrouter_cost.py"),
        "--data-root", str(selected), "--attempt-id", row["attempt_id"], "--apply"],
        cwd=command.REPO_ROOT, env=child_env, capture_output=True, text=True, timeout=60)
    assert process.stdout.strip(), process.stderr
    result = json.loads(process.stdout.strip())["outcomes"][0]
    assert result["status"] == ("applied" if applicable else "owner_unsettled"), process.stderr
    assert process.returncode == (0 if applicable else 1)
    final = current(selected, row["attempt_id"])
    if applicable:
        assert final["cost_final"] is True and final["cost_usd"] == 0.25
    else:
        assert final == row


class _PriceReviewModel:
    """A real physical accounting send inside the real review coordinator."""

    def __init__(self, root, *, hold=False):
        self.root, self.row = root, None
        self.ready, self.release = threading.Event(), threading.Event()
        if not hold:
            self.release.set()

    def chat(self, **kwargs):
        tokens = {"prompt_tokens": 37, "completion_tokens": 11, "cached_tokens": 9}
        payload = {"id": "generation-review", "usage": tokens}
        request = usage.AttemptRequest(
            model=kwargs["model"], provider="openrouter", drive_root=self.root,
            reservation_usd=5.0, global_limit_usd=100,
            provider_receipt_binding=cost.binding_for_target(TARGET))
        usage.execute_physical_attempt(request, lambda: payload,
                                       extractor=lambda p: (p["usage"], 2.0, False))
        attempt_id = usage.last_physical_attempt_capture().attempt_id
        self.row = current(self.root, attempt_id)
        self.ready.set()
        assert self.release.wait(10), "fixture did not release review worker"
        return ({"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "fixture review"})},
                {**tokens, "ledger_attempt_ids": [attempt_id], "cost": 2.0, "cost_final": False,
                 "provider": "openrouter"})


def _review_context(root, *, non_task=False):
    return SimpleNamespace(task_id="price-review", task_attempt=1, drive_root=root, budget_drive_root=root,
                           task_lifecycle_bound=not non_task, task_metadata={}, pending_events=[], event_queue=None)


def _real_ownership(monkeypatch):
    from supervisor import queue as task_queue
    from supervisor.queue_transitions import task_has_live_ownership

    # The ordinary command fixture stubs ownership. These consumer regressions
    # must use the real durable operation and queue ownership predicates.
    monkeypatch.setattr(task_queue, "task_has_live_ownership", task_has_live_ownership)


@pytest.mark.parametrize("surface", ["skill_review", "task_acceptance"])
@pytest.mark.parametrize("non_task", [False, True])
def test_completed_review_attribution_allows_exact_price_recovery(env, capsys, monkeypatch, surface, non_task):
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    _real_ownership(monkeypatch)
    ctx = _review_context(env, non_task=non_task)
    write_task_result(env, ctx.task_id, "running")
    model = _PriceReviewModel(env)
    run = run_review_request(
        ReviewRequest(surface=surface, task_id=ctx.task_id, goal="review", retry_key="completed-price-review",
                      usage_attribution={"review_skill": "fixture-skill"} if surface == "skill_review" else {}),
        slots=[ReviewSlot(slot_id="price-seat", model="vendor/unseen-model")],
        drive_root=env, usage_ctx=ctx, llm=model)
    assert run.actors[0]["status"] == "ok"
    row = model.row
    assert row["review_slot_id"] == "price-seat" and row["review_wave_id"] == "completed-price-review"
    if surface == "skill_review":
        assert row["review_skill"] == "fixture-skill"
    assert bool(row.get("non_task_operation")) is non_task
    write_task_result(env, ctx.task_id, "completed", result="kept reviewed answer")
    before_task = load_task_result(env, ctx.task_id)
    receipt(env, row, 0.125)
    code, result = invoke(env, row, capsys, "--fetch", "--apply")
    assert code == 0 and result["status"] == "applied"
    final = current(env, row["attempt_id"])
    assert final["cost_usd"] == 0.125 and final["cost_final"] is True
    for key in ("review_slot_id", "review_wave_id", "review_skill", "prompt_tokens", "completion_tokens"):
        assert final.get(key) == row.get(key)
    assert load_task_result(env, ctx.task_id) == before_task


@pytest.mark.parametrize("non_task", [False, True])
def test_terminal_author_with_live_acceptance_operation_blocks_until_review_finishes(
        env, capsys, monkeypatch, non_task):
    from ouroboros import model_wait, review_operation
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    _real_ownership(monkeypatch)
    ctx = _review_context(env, non_task=non_task)
    write_task_result(env, ctx.task_id, "running")
    model = _PriceReviewModel(env, hold=True)
    operation = None
    try:
        with model_wait.task_model_wait_scope(
                task={"id": ctx.task_id, "_attempt": 1}, drive_root=env,
                event_queue=queue.Queue(), worker_slot_held=False):
            run_review_request(
                ReviewRequest(surface="task_acceptance", task_id=ctx.task_id, goal="review", subject="answer",
                              retry_key="live-price-review", drain_deadline=time.monotonic()),
                slots=[ReviewSlot(slot_id="price-seat", model="vendor/unseen-model", timeout_sec=20)],
                drive_root=env, usage_ctx=ctx, llm=model)
            assert model.ready.wait(5)
            operation = next(op for op in review_operation._LIVE.values() if op.task_id == ctx.task_id)
        write_task_result(env, ctx.task_id, "completed", result="kept reviewed answer")
        row = model.row
        assert row["state"] == "settled" and row["review_slot_id"] == "price-seat"
        assert bool(row.get("non_task_operation")) is non_task
        assert review_operation.task_has_live_review_operation(env, ctx.task_id)
        receipt(env, row, 0.125)
        code, blocked = invoke(env, row, capsys, "--fetch", "--apply")
        assert code == 1 and blocked["status"] == ("review_unsettled" if non_task else "owner_unsettled")
        assert current(env, row["attempt_id"]) == row
    finally:
        model.release.set()
        if operation is not None:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                pointer = (load_task_result(env, ctx.task_id).get("review_operations") or {}).get(operation.owner_id, {})
                if operation.closed and pointer.get("state") in {"closed", "unpublished", "collected"}:
                    break
                time.sleep(0.01)
            assert operation.closed
            assert pointer.get("state") in {"closed", "unpublished", "collected"}
            assert not review_operation.task_has_live_review_operation(env, ctx.task_id)
    # The same receipt becomes applicable after the operation closes, without
    # another generation or metadata GET and without losing review attribution.
    code, applied = invoke(env, row, capsys, "--fetch", "--apply")
    assert code == 0 and applied["status"] == "applied"
    final = current(env, row["attempt_id"])
    assert final["cost_usd"] == 0.125 and final["review_wave_id"] == "live-price-review"
    assert load_task_result(env, ctx.task_id)["status"] == "completed"


def test_retention_failure_never_applies_money(env, capsys, monkeypatch):
    row = attempt(env)
    monkeypatch.setattr("requests.get", lambda *a, **k: SimpleNamespace(
        status_code=200, headers={}, json=lambda: {"data": {"id": "generation-selected", "total_cost": 1}},
        close=lambda: None))
    monkeypatch.setattr(cost, "write_blob", lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic full disk")))
    code, result = invoke(env, row, capsys, "--fetch", "--apply")
    assert code == 1 and result["status"] == "error" and result["error_type"] == "OSError"
    assert current(env, row["attempt_id"]) == row


def test_ledger_projection_restart_and_export_reimport(env, capsys):
    from ouroboros.terminal_cost_reconciliation import reconcile_abandoned_usage
    row = attempt(env)
    before_calls = usage.usage_breakdown(env)["physical_calls"]
    receipt(env, row, 0.125)
    assert invoke(env, row, capsys, "--apply")[0] == 0
    # No provider network in existing maintenance. Its dirty-owner transaction
    # recovers a crash after the ledger write and before terminal projection.
    reconcile_abandoned_usage(env)
    result = load_task_result(env, "child")
    root_result = load_task_result(env, "root")
    assert result["cost_final"] is True and result["accounted_upper_bound_usd"] == 0.125
    assert root_result["accounted_upper_bound_usd_with_children"] == 0.125
    assert result["result"] == "kept answer" and result["status"] == "cancelled"
    summary = usage.usage_breakdown(env)
    assert summary["physical_calls"] == before_calls
    with usage_store.read(env) as txn:
        assert txn.dirty_owners() == []
    # The existing database needed no new columns/indexes: evidence lives in
    # extra, and the export/import owners preserve the current row's facts.
    final = current(env, row["attempt_id"])
    usage_store.export_journal(env)
    usage_store.migrate_from_journal(env)
    restored = current(env, row["attempt_id"])
    for key in ("cost_usd", "cost_final", "prompt_tokens", "cached_tokens", "provider_receipt_binding"):
        assert restored[key] == final[key]
    assert usage.usage_breakdown(env)["physical_calls"] == before_calls
    assert invoke(env, row, capsys, "--apply")[1]["status"] == "duplicate"
