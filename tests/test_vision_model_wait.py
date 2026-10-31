"""Controlled model engine in a real tracked child; never live auth/inference."""

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest
from tests._store_shared import read_result

pytestmark = pytest.mark.serial

from ouroboros import usage_accounting as ua
from ouroboros.llm_claudexor import ClaudexorModelError, ClaudexorModelNotDispatched
from ouroboros.tools import vision, vision_process
from tests._usage_store_testing import ledger_rows

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

        def operations(self, **_kwargs):
            return []  # Legacy model-operation catalog.

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
            pending = mode in {"pending", "cancel", "cancel_contended", "cancel_unreadable"} or mode == "slow" and self.polls < 8
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
    transport.ensure_owned_gateway = lambda **_kw: gateway
    transport.config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC = 0.01
    pricing._fetch_live_rows = lambda *_: pytest.fail("VLM fixture must never fetch live pricing")
    original_write = vision_process._private_json

    def record_receipt(path, value):
        original_write(path, value)
        if path.name == "receipt.json":
            event("custody", fields=sorted(value["custody"]), operation_id=value["custody"]["operation_id"])

    vision_process._private_json = record_receipt
    if mode in {"cancel_contended", "cancel_unreadable"}:
        real_read, refusals = Path.read_text, []

        def contended_read(self, *args, **kwargs):
            # Simulate a transient sharing refusal over intact published JSON.
            # The retained Windows failure did not identify its actual exception;
            # a refusal that never clears must still remain unreadable.
            if self.name == "control.json" and self.exists() and (mode == "cancel_unreadable" or not refusals):
                refusals.append(self)
                raise PermissionError(13, "controlled sharing violation", str(self))
            return real_read(self, *args, **kwargs)

        Path.read_text = contended_read
    if mode in {"checkpoint_fail", "checkpoint_retry"}:
        original = vision_process._private_json
        refused_once = False

        def fail_receipt(path, value):
            nonlocal refused_once
            if path.name == "receipt.json" and (mode == "checkpoint_fail"
                    or not refused_once and value["custody"]["operation_id"]):
                refused_once = True
                raise OSError("controlled IPC write failure")
            return original(path, value)

        vision_process._private_json = fail_receipt


@pytest.fixture
def child_fixture(tmp_path, monkeypatch):
    from ouroboros.tools import shell, shell_process

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
            if state.get("window_from_checkpoint"):
                with monkeypatch.context() as patch:
                    patch.setattr(shell_process, "subprocess", _window_from_checkpoint(events))
                    return original(child_argv, **kwargs)
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


def _checkpointed(events):
    """Whether the child has published its exact operation checkpoint (a complete custody row naming it)."""
    try:
        lines = events.read_text(encoding="utf-8").splitlines(keepends=True)
    except FileNotFoundError:
        return False
    for line in lines:
        if line.endswith("\n"):
            row = json.loads(line)
            if row["kind"] == "custody" and row["operation_id"]:
                return True
    return False


def _window_from_checkpoint(events, ceiling_sec=120):
    """The tracked runner's ``subprocess`` with one change: the child's execution window opens only once the child
    has published its operation checkpoint. A loaded runner can spend any fixed window on the child's imports
    before it creates the operation (#2113); the expiry, the kill and the parent's read stay the real ones."""

    class Popen(subprocess.Popen):
        def communicate(self, input=None, timeout=None):
            ceiling = time.monotonic() + ceiling_sec  # a bound for a child that never gets there, not a window
            while self.poll() is None and not _checkpointed(events) and time.monotonic() < ceiling:
                time.sleep(0.02)
            return super().communicate(input, timeout)

    return SimpleNamespace(**{**vars(subprocess), "Popen": Popen})


def _call(client=None, *, timeout=1.0, **kwargs):
    return vision._vision_query_with_timeout(client, prompt="Describe only this image",
                                             images=[{"url": "data:image/png;base64,AAAA"}],
                                             model=MODEL, timeout=timeout, **kwargs)


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


@pytest.mark.parametrize("mode", ["success", "slow", "lose_create", "checkpoint_fail", "checkpoint_retry"])
def test_real_child_retains_one_generation_and_unknown_cost(child_fixture, mode):
    state, events, root = child_fixture
    state["mode"] = mode
    # Lost-create recovery must survive ordinary cold imports on its first
    # control outage. The other cases keep 20ms to prove that transport bounds
    # do not become logical inference deadlines.
    text, usage = _call(timeout=1.0 if mode == "lose_create" else 0.02)
    assert text == "pixels read exactly once 🐍"
    assert usage["cost"] is None and usage["cost_final"] is False
    assert len(usage["ledger_attempt_ids"]) == 1
    assert ua.last_physical_attempt_capture().state == "settled"
    rows = _events(events)
    assert len([row for row in rows if row["kind"] == "generation"]) == 1
    assert len({row["key"] for row in rows if row["kind"] == "create"}) == 1
    assert any(row["kind"] == "ack" for row in rows)
    custody = [row for row in rows if row["kind"] == "custody"]
    if mode != "checkpoint_fail":
        assert [row["operation_id"] for row in custody] == ["", "operation-exact"]
        assert all(row["fields"] == ["invocation_id", "operation_id", "request_manifest_ref", "request_ref"] for row in custody)
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
    assert "provider_error_type=string_above_max_length, parameter=instructions" in error.display_message[:220]
    assert "private provider body" not in error.display_message
    assert str(error) == "invalid_request: controlled refusal"
    assert error.physical_attempt_capture.state == "settled"
    assert error.ledger_attempt_ids == [error.physical_attempt_capture.attempt_id]
    rows = _events(events)
    assert sum(row["kind"] == "generation" for row in rows) == sum(row["kind"] == "ack" for row in rows) == 1
    attempts = ledger_rows(root)
    assert [row["state"] for row in attempts] == ["settled"]


def _parent_cancel(child_fixture, mode, reason):
    state, events, _ = child_fixture
    state["mode"] = mode

    def cancel_after_started():
        return "cancelled" if any(row["kind"] == "poll" for row in _events(events)) else None

    with pytest.raises(ClaudexorModelError) as caught:
        _call(model_poll_control=cancel_after_started)
    assert caught.value.control_reason == reason
    assert caught.value.operation_id == "operation-exact"
    assert caught.value.physical_attempt_capture.state == "unresolved"
    rows = _events(events)
    assert len([row for row in rows if row["kind"] == "generation"]) == 1
    assert {row["operation_id"] for row in rows if row["kind"] == "cancel"} == {"operation-exact"}


def test_parent_cancel_reaches_same_live_operation(child_fixture):
    _parent_cancel(child_fixture, "cancel", "cancelled")


@pytest.mark.parametrize("mode,reason", [("cancel_contended", "cancelled"),
                                         ("cancel_unreadable", "child_control_unreadable")])
def test_only_a_transient_control_read_refusal_still_reaches_the_cancel(child_fixture, mode, reason):
    _parent_cancel(child_fixture, mode, reason)


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
    state["window_from_checkpoint"] = True  # the window expires half a second after the operation exists
    with pytest.raises(ClaudexorModelError) as caught:
        vision_process.run_vision_child(child_timeout=0.5, subscription=True, model_role="vision",
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
    rows = ledger_rows(root)
    assert rows[-1]["state"] == "dispatched" and not any(row["state"] == "released" for row in rows)


@pytest.mark.parametrize("reserve,bound", [(0, False), (65536, True)])
def test_strict_child_schema_rejects_false_capture_and_wrong_receipt(reserve, bound):
    assert set(vision_process._CAPTURE["properties"]) == {field.name for field in fields(ua.PhysicalAttemptCapture)}
    assert set(vision_process._CONTEXT["properties"]) == {field.name for field in fields(ua.PhysicalAttemptContext)}
    capture = asdict(ua.PhysicalAttemptCapture("a", MODEL, "claudexor", "released", "opaque"))
    capture["physical_context"] = asdict(ua.PhysicalAttemptContext(
        "owner_low", "low", "cold_estimate", "route", "round", None, None, False, False,
        view_kind="focus", response_reserve_tokens=reserve, window_witness_bound=bound))
    receipt = {"receipt_id": "ours", "custody": None, "capture": capture}
    vision_process._read_receipt(receipt, "ours")
    restored = vision_process._restore_capture(capture).physical_context
    assert restored.view_kind == "focus"
    assert restored.response_reserve_tokens == reserve and restored.window_witness_bound is bound
    for key, invalid in (("response_reserve_tokens", -1), ("response_reserve_tokens", "8192"),
                         ("response_reserve_tokens", True), ("window_witness_bound", "true")):
        saved = capture["physical_context"][key]
        capture["physical_context"][key] = invalid
        with pytest.raises(Exception):
            vision_process._read_receipt(receipt, "ours")
        capture["physical_context"][key] = saved
    capture["physical_context"]["view_kind"] = ["focus"]
    with pytest.raises(Exception):
        vision_process._read_receipt(receipt, "ours")
    capture["physical_context"]["view_kind"] = "focus"
    with pytest.raises(ValueError, match="another invocation"):
        vision_process._read_receipt(receipt, "theirs")
    capture["effort"] = ["not", "an", "evidence", "object"]
    with pytest.raises(Exception):
        vision_process._read_receipt(receipt, "ours")
    capture["effort"] = None
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
    from ouroboros.task_results import write_task_result

    state, events, root = child_fixture
    write_task_result(root, "image-task", "running", root_task_id="image-task")
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
    persisted = read_result(root, "image-task")
    wait, = persisted["model_waits"].values()
    assert wait["state"] == "resolved" and wait["resolution"] == "resource_available"


def _vlm_ctx():
    from ouroboros.tools.registry import BrowserState

    state = BrowserState()
    state.last_screenshot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    # The attribute the registry's result sidecar uses, so the typed result is observable.
    return SimpleNamespace(browser_state=state, event_queue=None, task_id="image-task", task_attempt=None,
                           _accumulated_usage={}, pending_events=[], **{_TOOL_RESULT_ATTR: None})


@pytest.mark.parametrize("tool,label", [("vlm_query", "VLM_QUERY_FAILED"), ("analyze_screenshot", "VLM_ANALYSIS_FAILED")])
def test_tool_result_carries_the_reconstructed_typed_provider_projection(child_fixture, monkeypatch, tool, label):
    """#763: the provider field the child's receipt carries reaches the tool's returned text."""
    from ouroboros.tools.tool_result import _published_tool_result

    state, _, _ = child_fixture
    state["mode"] = "field"
    monkeypatch.setattr(vision, "_get_llm_client", lambda: None)
    monkeypatch.setattr(vision, "_resolve_vlm_model", lambda *a, **k: MODEL)
    ctx = _vlm_ctx()
    text = (vision._vlm_query(ctx, prompt="Describe only this image", image_url="data:image/png;base64,AAAA")
            if tool == "vlm_query" else vision._analyze_screenshot(ctx, prompt="Describe only this image"))
    first = text.splitlines()[0]
    assert first.startswith(f"⚠️ {label}: ") and "provider_error_type=string_above_max_length" in first
    assert "parameter=instructions" in first and "invalid_request: controlled refusal" in first
    assert "private provider body" not in text  # the raw provider body never joins the projection
    published = _published_tool_result(ctx, None)
    assert published.text == text and published.code == "VLM_ERROR"
    # Machine metadata is unchanged; only the returned text gained the typed projection.
    assert dict(published.meta) == {"model": MODEL, "provider_status_code": 400}


@pytest.mark.parametrize("error,shown", [
    (RuntimeError("HTTP 400 image input is not supported"), "HTTP 400 image input is not supported"),
    (ClaudexorModelError({"code": "invalid_request", "message": "plain refusal"}), "invalid_request: plain refusal"),
])
def test_an_error_without_typed_fields_keeps_its_own_text(monkeypatch, error, shown):
    monkeypatch.setattr(vision, "_get_llm_client", lambda: None)
    monkeypatch.setattr(vision, "_resolve_vlm_model", lambda *a, **k: MODEL)
    monkeypatch.setattr(vision, "_vision_query_with_timeout", lambda *a, **k: (_ for _ in ()).throw(error))
    assert vision._vlm_query(_vlm_ctx(), prompt="p", image_url="data:image/png;base64,AAAA") == f"⚠️ VLM_QUERY_FAILED: {shown}"


def test_an_unknown_outcome_is_still_raised_not_turned_into_a_result(monkeypatch):
    unknown = ClaudexorModelError({"code": "provider_failed", "message": "lost",
                                   "context": {"vendorCode": "should_not_show"}}, unknown=True)
    monkeypatch.setattr(vision, "_get_llm_client", lambda: None)
    monkeypatch.setattr(vision, "_resolve_vlm_model", lambda *a, **k: MODEL)
    monkeypatch.setattr(vision, "_vision_query_with_timeout", lambda *a, **k: (_ for _ in ()).throw(unknown))
    with pytest.raises(ClaudexorModelError) as caught:
        vision._vlm_query(_vlm_ctx(), prompt="p", image_url="data:image/png;base64,AAAA")
    assert caught.value is unknown and "should_not_show" not in unknown.display_message
