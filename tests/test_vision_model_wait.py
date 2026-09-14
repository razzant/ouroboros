"""Controlled model engine in a real tracked child; never live auth/inference."""

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.serial

from ouroboros import usage_accounting as ua
from ouroboros.llm_claudexor import ClaudexorModelError, ClaudexorModelNotDispatched
from ouroboros.tools import vision, vision_process

MODEL = "claudexor::fixture=image-model"
ROUTE = {"source": "fixture", "model": "image-model", "credentialProfileId": "account-exact",
         "accountFingerprint": "fingerprint-exact"}
REF = {"resourceId": "resource-exact", "sha256": "sha256:" + "a" * 64, "sizeBytes": 100}


def _install_child_fixture(mode, events_path):
    """Patch only the gateway, retaining actual LLM/IPC/accounting code."""
    from ouroboros import llm_claudexor as transport, pricing
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def event(kind, **fields):
        with open(events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **fields}) + "\n")

    class Gateway:
        operation = ""
        polls = 0
        lost = False

        def upload_model_request(self, payload, *, idempotency_key):
            event("upload", key=idempotency_key, payload=payload)
            return REF

        def create_model_operation(self, ref, *, idempotency_key):
            event("create", key=idempotency_key)
            if not self.operation:
                self.operation = "operation-exact"
                event("generation", operation_id=self.operation)
            if mode == "lose_create" and not self.lost:
                self.lost = True
                raise ClaudexorUnavailable("daemon_unreachable", "controlled lost create reply")
            return self.detail()

        def detail(self):
            pending = mode in {"pending", "cancel"} or mode == "slow" and self.polls < 8
            not_started = mode in {"quota", "auth", "mixed"}
            refused = not_started or mode == "field"
            problem = {"code": {"quota": "subscription_window_exhausted", "auth": "auth_required",
                                "mixed": "credential_pool_exhausted", "field": "invalid_request"}.get(mode, "fixture_error"),
                       "message": "controlled refusal", "context": {"resetsAt": "2099-01-01T00:00:00Z"}}
            if mode == "field":
                problem["context"].update(httpStatus=400, vendorCode="string_above_max_length",
                                          parameter="instructions", providerMessage="private provider body")
            return {"id": self.operation, "state": "running" if pending else "failed" if refused else "succeeded",
                    "dispatch": {"state": "started" if pending else "not_started" if not_started else "response_received",
                                 "route": ROUTE},
                    "response": {"state": "absent"} if pending else {"state": "ready", "ref": REF},
                    "problem": problem if refused else None}

        def get_model_operation(self, operation_id, **kwargs):
            assert operation_id == self.operation
            self.polls += 1
            event("poll", operation_id=operation_id)
            return self.detail()

        def get_model_result(self, operation_id, **kwargs):
            detail = self.detail()
            value = {"outcome": "failed" if detail["problem"] else "completed", "route": ROUTE,
                     "message": {"role": "assistant", "content": "pixels read exactly once 🐍"},
                     "usage": {"input_tokens": 21, "output_tokens": 5},
                     "cost": {"knowledge": "unknown", "cashUsd": None}, "problem": detail["problem"]}
            return json.dumps(value).encode("utf-8")

        def acknowledge_model_result(self, operation_id, sha256):
            event("ack", operation_id=operation_id)
            return {"id": operation_id, "response": {"state": "acknowledged", "ref": REF}}

        def cancel_model_operation(self, operation_id, **kwargs):
            event("cancel", operation_id=operation_id)
            return self.detail()

        def close(self):
            event("close")

    gateway = Gateway()
    transport.ensure_owned_gateway = lambda: gateway
    transport.config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC = 0.01
    pricing._fetch_live_rows = lambda *_: pytest.fail("VLM fixture must never fetch live pricing")
    if mode == "checkpoint_fail":
        original = vision_process._private_json

        def fail_receipt(path, value):
            if path.name == "receipt.json":
                raise OSError("controlled IPC write failure")
            return original(path, value)

        vision_process._private_json = fail_receipt


@pytest.fixture
def child_fixture(tmp_path, monkeypatch):
    from ouroboros.tools import shell

    events = tmp_path / "events.jsonl"
    original = shell._tracked_subprocess_run
    state = {"mode": "success", "paths": [], "timeouts": []}

    def run(argv, **kwargs):
        payload = Path(argv[-1])
        state["paths"].append(payload.parent)
        state["timeouts"].append(kwargs["timeout"])
        if os.name != "nt":
            assert payload.stat().st_mode & 0o777 == 0o600
            assert payload.parent.stat().st_mode & 0o777 == 0o700
        child_argv = list(argv)
        child_argv[2] = (f"import runpy; runpy.run_path({__file__!r})['_install_child_fixture']"
                         f"({state['mode']!r}, {str(events)!r}); " + argv[2])
        try:
            return original(child_argv, **kwargs)
        finally:
            for path in payload.parent.iterdir():
                if os.name != "nt":
                    assert path.stat().st_mode & 0o777 == 0o600

    monkeypatch.setattr(shell, "_tracked_subprocess_run", run)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "data" / "settings.json"))
    monkeypatch.setattr(vision_process.config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.01)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path / "data", task_id="image-task", root_task_id="image-task")):
        yield state, events, tmp_path / "data"
    assert all(not path.exists() for path in state["paths"])
    assert not shell._active_subprocesses


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _call(client=None, **kwargs):
    return vision._vision_query_with_timeout(client, prompt="Describe only this image",
                                             images=[{"url": "data:image/png;base64,AAAA"}],
                                             model=MODEL, timeout=1.0, **kwargs)


def test_private_ipc_publication_never_exposes_an_empty_control_file(tmp_path, monkeypatch):
    actual_write = vision_process.atomic_write_json
    previous = []

    def observe(path, value):
        # This is the moment the concurrent child could read control. An empty
        # pre-created destination is invalid JSON and falsely cancels the call.
        previous.append(json.loads(path.read_text()) if path.exists() else None)
        actual_write(path, value)

    monkeypatch.setattr(vision_process, "atomic_write_json", observe)
    path = tmp_path / "control.json"
    first = {"receipt_id": "first", "reason": "cancelled"}
    second = {"receipt_id": "second", "reason": "deadline"}
    vision_process._private_json(path, first)
    vision_process._private_json(path, second)
    assert previous == [None, first]
    assert json.loads(path.read_text()) == second
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("mode", ["success", "slow", "lose_create", "checkpoint_fail"])
def test_real_child_retains_one_generation_and_unknown_cost(child_fixture, mode):
    state, events, root = child_fixture
    state["mode"] = mode
    text, usage = _call()
    assert text == "pixels read exactly once 🐍"
    assert usage["cost"] is None and usage["cost_final"] is False
    assert len(usage["ledger_attempt_ids"]) == 1
    assert ua.last_physical_attempt_capture().state == "settled"
    rows = _events(events)
    assert len([row for row in rows if row["kind"] == "generation"]) == 1
    assert len({row["key"] for row in rows if row["kind"] == "create"}) == 1
    assert any(row["kind"] == "ack" for row in rows)
    if mode == "slow":
        assert len([row for row in rows if row["kind"] == "poll"]) >= 8
        assert state["timeouts"][0] > 90
    request = next((root / "observability" / "calls" / "image-task").glob("*_model_request.json"))
    assert json.loads(request.read_text())["operation_id"] == "operation-exact"


@pytest.mark.parametrize("mode,code", [("quota", "subscription_window_exhausted"), ("auth", "auth_required"),
                                       ("mixed", "credential_pool_exhausted")])
def test_real_child_preserves_typed_refusal_receipt(child_fixture, mode, code):
    state, _, _ = child_fixture
    state["mode"] = mode
    with pytest.raises(ClaudexorModelNotDispatched) as caught:
        _call()
    error = caught.value
    assert error.code == code and error.reset_at == "2099-01-01T00:00:00Z"
    assert error.operation_id == "operation-exact" and error.route == ROUTE
    assert error.model_role == "vision"
    assert error.physical_attempt_capture.state == "released"
    assert error.ledger_attempt_ids == [error.physical_attempt_capture.attempt_id]


def test_real_child_reconstructs_provider_field_display_after_settlement(child_fixture):
    state, events, root = child_fixture
    state["mode"] = "field"
    with pytest.raises(ClaudexorModelError) as caught:
        _call()
    error = caught.value
    assert type(error) is ClaudexorModelError
    assert error.code == "invalid_request" and error.status_code == 400 and error.retryable is False
    assert error.operation_id == "operation-exact" and error.route == ROUTE and error.model_role == "vision"
    assert error.body == error.problem and error.problem["context"]["providerMessage"] == "private provider body"
    assert "provider_code=string_above_max_length, parameter=instructions" in error.display_message[:220]
    assert "private provider body" not in error.display_message
    assert str(error) == "invalid_request: controlled refusal"
    assert error.physical_attempt_capture.state == "settled"
    assert error.ledger_attempt_ids == [error.physical_attempt_capture.attempt_id]
    rows = _events(events)
    assert sum(row["kind"] == "generation" for row in rows) == sum(row["kind"] == "ack" for row in rows) == 1
    attempts = [json.loads(line) for line in (root / ua.LEDGER_REL).read_text().splitlines()]
    assert [row["state"] for row in attempts] == ["reserved", "dispatched", "settled"]


def test_parent_cancel_reaches_same_live_operation(child_fixture):
    state, events, _ = child_fixture
    state["mode"] = "cancel"

    def cancel_after_started():
        return "cancelled" if any(row["kind"] == "poll" for row in _events(events)) else None

    with pytest.raises(ClaudexorModelError) as caught:
        _call(model_poll_control=cancel_after_started)
    assert caught.value.control_reason == "cancelled"
    assert caught.value.operation_id == "operation-exact"
    assert caught.value.physical_attempt_capture.state == "unresolved"
    rows = _events(events)
    assert len([row for row in rows if row["kind"] == "generation"]) == 1
    assert {row["operation_id"] for row in rows if row["kind"] == "cancel"} == {"operation-exact"}


def test_dead_child_without_checkpoint_is_unknown_not_free(monkeypatch):
    from ouroboros.tools import shell

    monkeypatch.setattr(shell, "_tracked_subprocess_run", lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=137))
    with pytest.raises(ClaudexorModelError) as caught:
        _call()
    error = caught.value
    assert error.code == "model_outcome_unknown" and error.operation_id == "" and error.route == {}
    assert not isinstance(error, ClaudexorModelNotDispatched)
    assert not hasattr(error, "physical_attempt_capture")


def test_killed_child_retains_exact_operation_checkpoint(child_fixture):
    state, events, root = child_fixture
    state["mode"] = "pending"
    with pytest.raises(ClaudexorModelError) as caught:
        vision_process.run_vision_child(child_timeout=5, subscription=True, model_role="vision",
                                        model=MODEL, prompt="one image", images=[], timeout=0.02)
    error = caught.value
    assert error.code == "model_outcome_unknown" and error.operation_id == "operation-exact"
    assert error.physical_attempt_capture.state == "dispatched"
    custody = error.model_operation_custody
    assert custody["invocation_id"] == error.physical_attempt_capture.attempt_id
    assert custody["request_ref"] == REF
    manifest = json.loads(Path(custody["request_manifest_ref"]["path"]).read_text())
    assert manifest["operation_id"] == error.operation_id
    assert len([row for row in _events(events) if row["kind"] == "generation"]) == 1
    rows = [json.loads(line) for line in (root / ua.LEDGER_REL).read_text().splitlines()]
    assert rows[-1]["state"] == "dispatched" and not any(row["state"] == "released" for row in rows)


def test_strict_child_schema_rejects_false_capture_and_wrong_receipt():
    assert set(vision_process._CAPTURE["properties"]) == {field.name for field in fields(ua.PhysicalAttemptCapture)}
    assert set(vision_process._CONTEXT["properties"]) == {field.name for field in fields(ua.PhysicalAttemptContext)}
    capture = asdict(ua.PhysicalAttemptCapture("a", MODEL, "claudexor", "released", "opaque"))
    receipt = {"receipt_id": "ours", "custody": None, "capture": capture}
    vision_process._read_receipt(receipt, "ours")
    with pytest.raises(ValueError, match="another invocation"):
        vision_process._read_receipt(receipt, "theirs")
    capture["state"] = "free"
    with pytest.raises(Exception):
        vision_process._read_receipt(receipt, "ours")


def test_only_unsettled_image_call_retries_under_parent_wait(child_fixture, monkeypatch):
    from ouroboros.model_wait import task_model_wait_scope

    state, events, root = child_fixture
    state["mode"] = "quota"
    client = SimpleNamespace()
    waits = []
    with task_model_wait_scope(task={"id": "image-task"}, drive_root=root,
                               event_queue=None, worker_slot_held=True) as wait:
        def resume(actual_client, error, kwargs):
            assert actual_client is client and error.code == "subscription_window_exhausted"
            waits.append(error)
            state["mode"] = "success"
            return {**kwargs, "model_account_override": "replacement"}

        monkeypatch.setattr(wait, "wait", resume)
        text, usage = _call(client)
    assert text == "pixels read exactly once 🐍" and len(waits) == 1
    assert len(usage["ledger_attempt_ids"]) == 2
    uploads = [row["payload"] for row in _events(events) if row["kind"] == "upload"]
    assert len(uploads) == 2 and uploads[0]["messages"] == uploads[1]["messages"]
    assert uploads[1]["account"] == {"mode": "pin", "profileId": "replacement"}
    assert len(state["paths"]) == 2  # No tool/image-preparation replay; only the unsettled LLM call.


def test_subscription_tool_envelope_uses_existing_task_ceiling(monkeypatch):
    monkeypatch.setattr(vision, "_resolve_vlm_model", lambda *_a, **_k: MODEL)
    monkeypatch.setattr(vision, "_get_llm_client", lambda: None)
    monkeypatch.setattr(vision, "_vision_execution_window", lambda: 900)
    assert vision._vision_tool_timeout(None, {}) == 900 + 2 * vision.NESTED_SETTLEMENT_MARGIN_SEC
    monkeypatch.setattr(vision, "_resolve_vlm_model", lambda *_a, **_k: "api-model")
    assert vision._vision_tool_timeout(None, {}) == 0


def test_real_parent_quota_controller_polls_metadata_then_rejoins_image_call(child_fixture, monkeypatch):
    from ouroboros.model_wait import task_model_wait_scope

    state, events, root = child_fixture
    state["mode"] = "quota"
    catalog_calls = []

    def catalog(source, account, *, requested_model=None):
        catalog_calls.append((source, account))
        state["mode"] = "success"
        return {"source": source, "models": [{"id": "image-model"}]}

    client = SimpleNamespace(
        claudexor_model_sources=lambda: {"sources": [{"id": "fixture", "credentialHarness": "fixture"}]},
        claudexor_model_catalog=catalog,
    )
    resolutions, phases = [], []
    monkeypatch.setattr(vision, "_get_llm_client", lambda: client)
    monkeypatch.setattr(vision, "_resolve_vlm_model", lambda *_a, **_k: resolutions.append(MODEL) or MODEL)
    monkeypatch.setattr(vision, "emit_cognitive_operation_event", lambda *_a, **kw: phases.append(kw["phase"]))
    ctx = SimpleNamespace(task_metadata={}, deadline_ts=None, task_id="image-task", event_queue=None)
    with task_model_wait_scope(task={"id": "image-task"}, drive_root=root,
                               event_queue=None, worker_slot_held=True):
        text = vision._vlm_query(ctx, "Describe only this image", image_url="data:image/png;base64,AAAA")
    assert text == "pixels read exactly once 🐍" and catalog_calls == [("fixture", None)]
    assert resolutions == [MODEL] and phases == ["started", "finished"]
    assert len([row for row in _events(events) if row["kind"] == "upload"]) == 2
    persisted = json.loads((root / "task_results" / "image-task.json").read_text())
    wait, = persisted["model_waits"].values()
    assert wait["state"] == "resolved" and wait["resolution"] == "resource_available"
