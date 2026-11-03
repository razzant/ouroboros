"""Presence recovers a lost model answer inside the same turn and tool transcript.

Real loop, dispatcher, registry and monetary ledger; only provider sends are scripted.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import queue
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import loop, loop_transport, owner_mailbox, usage_accounting as ua
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
from ouroboros.presence_authority import presence_ceiling_payload
from tests.presence_legacy_helpers import legacy_registry as ToolRegistry
from tests.test_presence_completion import _call as finish
from tests.test_presence_runner import _admission
from tests.test_transport_death_retry import _events, _ledger
from tests.test_context_fit_integration import _plan
from tests._shared import saved_settings

PRIMARY = "claudexor::codex=primary-model"
FALLBACK = "claudexor::codex=fallback-model"
TASK = "presence-round-recovery"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    from ouroboros import context, fallback_cooldown
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.delenv(MODEL_ACCOUNTS_KEY, raising=False)
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: (
        {"model": task["model"], "provider": "claudexor"},
        SimpleNamespace(route_fp="fixture-" + task["model"], status="confirmed", stale=False,
                        window_tokens=900_000, source="advertised", source_id="codex")))
    (root / "state").mkdir(parents=True)
    return root


class Model:
    def __init__(self, root, *steps):
        self.root, self.steps, self.sends = root, list(steps), []

    def default_model(self):
        return PRIMARY

    def chat(self, **kwargs):
        self.sends.append({**deepcopy(kwargs), "selected_account": kwargs.get("model_account_override")
            if kwargs.get("model_account_override") is not None else model_role_option(MODEL_ACCOUNTS_KEY, kwargs["model_role"])})
        assert self.steps, "unexpected generation after the selected answer"
        step = self.steps.pop(0)

        def send():
            if callable(step):
                raise step()
            return deepcopy(step)

        request = ua.AttemptRequest(model=kwargs["model"], provider="openrouter", reservation_usd=1.0,
            drive_root=self.root, task_id=TASK, root_task_id=TASK, source="test.presence_round_recovery")
        response = ua.execute_physical_attempt(request, send,
            extractor=lambda _response: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True))
        return response, {"prompt_tokens": 1, "completion_tokens": 1}


def lost():
    return httpx.ReadError("model response lost after dispatch")


def run(root, model, *, registry=None, notes=None):
    registry = registry or ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx.is_direct_chat = True
    registry._ctx.task_metadata = {"presence": {"binding_id": "room-binding"}}
    registry._ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
    registry._ctx.context_fit_plan = replace(_plan(preferred="max", window=900_000), model=PRIMARY)
    notes = [] if notes is None else notes
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id=TASK, root_task_id=TASK)):
        result = loop.run_llm_loop(
            [{"role": "user", "content": "Read this room and answer."}], registry, model, root / "logs",
            lambda text, **_kw: notes.append(text), queue.Queue(), task_id=TASK, task_type="presence", drive_root=root)
    return (*result, registry)


@pytest.mark.parametrize("prior_tools", [False, True])
def test_configured_fallback_keeps_completed_tools_pins_and_ceiling(data_root, monkeypatch, prior_tools):
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "primary-pin", "fallback": ["fallback-pin"]}))
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_a, **_kw: pytest.fail("configured fallback precedes any probe"))
    registry = ToolRegistry(repo_dir=data_root, drive_root=data_root)
    completed = []
    registry.override_handler("chat_history", lambda _ctx, **_kw: completed.append("history") or "Retained room evidence")
    history = {"role": "assistant", "content": None, "tool_calls": [{"id": "history-before-loss", "type": "function",
        "function": {"name": "chat_history", "arguments": "{}"}}]}
    # The ordinary author-handover check keeps its one recovery nudge after tools.
    # The successor selects its answer after reading it, without re-running history.
    model = Model(data_root, *([history] if prior_tools else []), lost,
        *([{"role": "assistant", "content": "The earlier result supports the answer."}] if prior_tools else []),
        finish("message", "Recovered answer"))
    text, usage, trace, registry = run(data_root, model, registry=registry)

    assert text == "Recovered answer" and not model.steps
    assert [send["model"] for send in model.sends] == ([PRIMARY] if prior_tools else []) + [PRIMARY, FALLBACK] + (
        [FALLBACK] if prior_tools else [])
    assert [send["selected_account"] for send in model.sends] == (
        ["primary-pin"] if prior_tools else []) + ["primary-pin", "fallback-pin"] + (["fallback-pin"] if prior_tools else [])
    assert completed == (["history"] if prior_tools else [])
    assert [call["tool"] for call in trace["tool_calls"]] == (["chat_history"] if prior_tools else []) + ["presence_finish"]
    assert all(not call["is_error"] for call in trace["tool_calls"])
    previous_tools = [row for row in model.sends[int(prior_tools)]["messages"] if row.get("role") == "tool"]
    assert previous_tools == [row for row in model.sends[-1]["messages"] if row.get("role") == "tool"]
    if prior_tools:
        assert len(previous_tools) == 1 and "Retained room evidence" in previous_tools[0]["content"]
    schemas = [{item["function"]["name"] for item in send["tools"]} for send in model.sends]
    assert all(names == schemas[0] for names in schemas) and "run_shell" not in schemas[0]
    assert registry._ctx.task_contract["capability_ceiling"] == presence_ceiling_payload(_admission().capability_ceiling)
    rows = _ledger(data_root)
    assert [row["state"] for row in rows] == (["settled"] if prior_tools else []) + ["unresolved", "settled"] + (
        ["settled"] if prior_tools else [])
    old, selected = rows[int(prior_tools)], rows[-1]
    assert old["attempt_id"] != selected["attempt_id"] and old.get("cost_usd") is None
    assert usage["transport_recovery"]["previous_attempt"]["physical_attempt_id"] == old["attempt_id"]
    assert usage["transport_recovery"]["old_outcome"] == "unknown"
    assert TRANSPORT_DEATHS_KEY not in usage
    assert [row["retry_same_request"] for row in _events(data_root / "logs", "llm_api_error")] == [False]
    assert _events(data_root / "logs", "network_wait") == []


def test_same_operation_recoverable_never_dials_a_fallback(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: pytest.fail("no regeneration"))

    def recoverable():
        error = RuntimeError("accepted operation control read was lost")
        error.same_operation_recoverable = True
        error.operation_id = "accepted-operation"
        return error

    model = Model(data_root, recoverable)
    _text, usage, trace, _registry = run(data_root, model)
    assert len(model.sends) == 1 and _ledger(data_root)[0]["state"] == "unresolved"
    assert usage["_pending_transport_outcome"]["same_operation_recoverable"] is True
    assert usage["_pending_transport_outcome"]["operation_id"] == "accepted-operation"
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"


def test_stop_during_unknown_wait_preserves_one_attempt(data_root, monkeypatch):
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
    saved_settings(monkeypatch, data_root, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: {})

    def stop(_seconds, _wake):
        owner_mailbox.write_owner_message(data_root, REASON_OWNER_STOPPED_DIRECT_TURN, TASK,
            msg_id="presence-stop", kind=owner_mailbox.KIND_FINALIZE_NOW)
        return True

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", stop)
    model = Model(data_root, lost)
    _text, usage, _trace, _registry = run(data_root, model)
    assert len(model.sends) == len(_ledger(data_root)) == 1
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"


def test_known_budget_exhaustion_blocks_new_physical_fallback(data_root, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "1.5")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)

    def spent():
        ua.record_subscription_session("other-known-charge", drive_root=data_root, route="subscription",
            task_id="other", root_task_id="other", spend_usd=1.5)
        return lost()

    model = Model(data_root, spent, finish("message", "must not be sent"))
    _text, usage, _trace, _registry = run(data_root, model)
    rows = _ledger(data_root)
    assert [row["state"] for row in rows] == ["unresolved"]
    assert rows[0].get("cost_usd") is None
    assert usage["reason_code"] == "budget_exhausted"


def test_deadline_closing_after_lost_response_sends_no_fallback(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: pytest.fail("spent deadline"))
    registry = ToolRegistry(repo_dir=data_root, drive_root=data_root)

    def expired():
        registry._ctx.task_metadata["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        return lost()

    model = Model(data_root, expired)
    _text, usage, trace, _registry = run(data_root, model, registry=registry)
    assert len(model.sends) == len(_ledger(data_root)) == 1
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert usage["_pending_transport_outcome"]["outcome"] == "unknown"


@pytest.mark.parametrize("http_status", [0, 404])
def test_control_loss_rejoins_the_accepted_operation_without_another_post(data_root, monkeypatch, http_status):
    from ouroboros import llm_claudexor
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    invocation = llm_claudexor._ModelInvocation({"usage_model": PRIMARY}, {}, {"timeout": 1})
    invocation.operation_id, invocation.invocation_id = "accepted-operation", "physical-attempt"
    invocation.task_id, invocation.root, invocation.create_attempted = TASK, data_root, True
    ctx = SimpleNamespace(task_id=TASK, is_direct_chat=True, current_task_type="presence")
    monkeypatch.setattr(llm_claudexor, "current_model_wait",
        lambda: SimpleNamespace(tool_context=ctx, control_reason=lambda: None))
    clock, reads = [0.0], []
    monkeypatch.setattr(llm_claudexor, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))

    class Gateway:
        def get_model_operation(self, operation, **_kwargs):
            reads.append(operation)
            if len(reads) <= 3:
                raise ClaudexorUnavailable("daemon_unreachable" if not http_status else "http_404",
                    "control read failed", status_code=http_status)
            return {"id": operation, "state": "succeeded", "dispatch": {"state": "response_received"},
                "response": {"state": "ready", "ref": {"sha256": "exact"}}}

        def get_model_result(self, operation, **_kwargs):
            assert operation == "accepted-operation"
            return b'{"outcome":"completed","message":{"content":"same accepted answer"}}'

        def create_model_operation(self, *_args, **_kwargs):
            pytest.fail("second POST while the accepted operation is recoverable")

        def close(self):
            pass

    gateway = invocation.gateway = Gateway()
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    monkeypatch.setattr(invocation, "retain", lambda _raw: None)
    assert invocation.receive()["message"]["content"] == "same accepted answer"
    assert reads == ["accepted-operation"] * 4 and clock[0] > invocation.timeout
    assert invocation.operation_id == "accepted-operation"
    assert _events(data_root / "logs", "network_wait")[-1]["detail"] == "same_model_operation_rejoined"
