"""A running extension call's Stop (PluginAPI 2.6 ``get_stop_state``): one host-bound reading before a further action.

The observed failure: an extension sent the first part of a message, the owner pressed Stop, and the extension sent the
next part eight seconds later because nothing told the running call. The Stop here is recorded the way the gateway
records it (``cancel_intents.request_cancel``) over real task rows; only the extension's own steps are fixtures. Covered:
the reader and its latch on the call (decided again after the unlocked read), one loaded plugin serving two tasks,
contexts that prove no call, authority that cannot be read, the per-call child's binding and its transport field, the
host's typing of a call refused a further action, the receipts result review keeps, and the whole path through the
registry in process and out of process.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import pathlib
import threading
from types import SimpleNamespace

import pytest

from ouroboros import extension_plugin_api as plugin_api
from ouroboros.extension_registry_state import _PluginAPIConfig
from ouroboros.extension_surface_names import extension_surface_name
from ouroboros.tools.registry import ToolContext
from tests._extension_loader_shared import _add_fake_native_dep, _mark_isolated_deps_installed, _prepare_extension
from tests._extension_loader_shared import _clear_loader_state as _clear_loader_state  # autouse loader isolation

REPO = pathlib.Path(__file__).resolve().parents[1]
SKILL = "stopper"
NAME = extension_surface_name(SKILL, "send")
CONTINUE = {"status": "continue", "reason": "", "note": ""}
CANCELLED = {"status": "stop", "reason": "cancelled", "note": "Stop is in force for this task"}


@pytest.fixture
def task_ctx(tmp_path, monkeypatch):
    """A running managed task's own context over its persisted row (its Stop lives there)."""
    from ouroboros import config
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data)
    write_task_result(data, "root", "running", root_task_id="root")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=data)
    ctx.task_id = ctx.root_task_id = "root"
    ctx.task_lifecycle_bound = True
    return ctx


def _api(tmp_path, *, skill=SKILL, generation="2.6"):
    """One loaded skill's PluginAPI, shared by every call of every task, as the loader hands it to ``register``."""
    return plugin_api.PluginAPIImpl(_PluginAPIConfig(
        skill_name=skill, permissions=("tool",), env_allowlist=(), state_dir=tmp_path / "state" / skill,
        settings_reader=lambda: {}, drive_root=tmp_path, plugin_api_generation=generation))


@contextlib.contextmanager
def _call(ctx, *, skill=SKILL):
    """One host invocation of ``skill``'s tool bound as the registry binds it (``owner_pause.tool_handoff``), closed
    as that binding closes when the call returns."""
    from ouroboros.owner_pause import _TOOL_OPERATION
    from ouroboros.tool_custody import invocation_outcome

    name = extension_surface_name(skill, "send")
    outcome = invocation_outcome(ctx, name, f"op-{ctx.task_id or 'none'}-{skill}")
    token = _TOOL_OPERATION.set((ctx, name, outcome))
    try:
        yield outcome
    finally:
        with outcome["phase_lock"]:
            outcome["closed"] = True
        _TOOL_OPERATION.reset(token)


def _stop(ctx, task_id="root", **policy):
    """The owner's Stop as the gateway records it, before any sweep reaches the members."""
    from ouroboros.cancel_intents import request_cancel

    return request_cancel(ctx.drive_root, task_id, **{"reason": "owner Stop", "source": "test",
                                                      "requested_stop_policy": "immediate", **policy})


# --- the reader and its latch on the call --------------------------------------------------------------------------

@pytest.mark.parametrize("control", ["immediate", "finalize_then_cancel", "pause"])
def test_an_immediate_stop_refuses_the_next_action_and_holds_while_wrap_up_and_pause_refuse_nothing(
        tmp_path, task_ctx, control):
    from ouroboros import cancel_intents, owner_pause

    api = _api(tmp_path)
    with _call(task_ctx) as outcome:
        assert api.get_stop_state() == CONTINUE
        if control == "pause":  # an operation the tree already handed out finishes under a Pause
            owner_pause.install_fence(task_ctx.drive_root, "root", request_id="pause-1")
        else:  # a Wrap up lets the call finish its work
            _stop(task_ctx, requested_stop_policy=control)
        state = api.get_stop_state()
        if control != "immediate":
            assert state == CONTINUE and plugin_api.STEP_STOP not in outcome
        else:
            assert state == CANCELLED
            assert outcome[plugin_api.STEP_STOP] == {"reason": "cancelled", "note": CANCELLED["note"], "refusals": 1}
            # The intent settles (the row has not turned cancelled yet): a fresh read would say continue, the call's
            # latch keeps the refusal it handed out for the rest of the call.
            cancel_intents.settle_intent(task_ctx.drive_root, "root", outcome="test")
            assert api.get_stop_state() == CANCELLED and outcome[plugin_api.STEP_STOP]["refusals"] == 2
    if control == "pause":
        owner_pause.release_fence(task_ctx.drive_root, "root", reason="owner_resume")


@pytest.mark.parametrize("scope, refused", [("cascade", True), ("", False)])
def test_an_immediate_cascade_above_refuses_a_members_next_action_and_a_single_stop_above_does_not(
        tmp_path, task_ctx, scope, refused):
    from ouroboros.task_results import write_task_result

    write_task_result(task_ctx.drive_root, "child", "running", root_task_id="root", parent_task_id="root",
                      delegation_role="subagent")
    task_ctx.task_id = "child"
    with _call(task_ctx):
        _stop(task_ctx, "root", scope=scope)
        state = _api(tmp_path).get_stop_state()
    assert state == (CANCELLED if refused else CONTINUE)


def test_one_plugin_serving_two_tasks_refuses_only_the_stopped_tasks_call(tmp_path, task_ctx):
    from ouroboros.task_results import write_task_result

    write_task_result(task_ctx.drive_root, "other", "running", root_task_id="other")
    other = ToolContext(repo_dir=tmp_path, drive_root=task_ctx.drive_root)
    other.task_id = other.root_task_id = "other"
    other.task_lifecycle_bound = True
    api = _api(tmp_path)  # one object, as one loaded skill serves every task's calls
    first, recorded, answers = threading.Barrier(3), threading.Event(), {}

    def call(ctx):
        with _call(ctx) as outcome:
            answers[ctx.task_id] = [api.get_stop_state()]
            first.wait(10)
            recorded.wait(10)
            answers[ctx.task_id].append(api.get_stop_state())
            answers[ctx.task_id].append(plugin_api.STEP_STOP in outcome)

    threads = [threading.Thread(target=contextvars.Context().run, args=(call, ctx)) for ctx in (task_ctx, other)]
    for thread in threads:
        thread.start()
    first.wait(10)
    _stop(task_ctx, "root")
    recorded.set()
    for thread in threads:
        thread.join(10)
    assert answers["root"] == [CONTINUE, CANCELLED, True]
    assert answers["other"] == [CONTINUE, CONTINUE, False], "the other task's call goes on, its record untouched"


def test_a_context_without_a_call_of_this_skill_is_unbound_and_permits_nothing(tmp_path, task_ctx):
    api = _api(tmp_path)
    unbound = api.get_stop_state()
    assert unbound == {"status": "unbound", "reason": "unbound", "note": "no tool call of this skill runs in this context"}
    with _call(task_ctx, skill="another"):
        assert api.get_stop_state() == unbound, "another skill's call is not this skill's"
    with _call(task_ctx) as outcome:
        seen = []
        thread = threading.Thread(target=contextvars.Context().run, args=(lambda: seen.append(api.get_stop_state()),))
        thread.start()
        thread.join(10)
        assert seen == [unbound], "a thread the extension started without the call's context proves no call"
        assert plugin_api.STEP_STOP not in outcome


def test_a_context_copied_from_a_call_that_returned_is_refused_a_further_action(tmp_path, task_ctx):
    api = _api(tmp_path)
    with _call(task_ctx) as outcome:
        copied = contextvars.copy_context()
    state = copied.run(api.get_stop_state)
    assert state == {"status": "stop", "reason": "operation_already_returned",
                     "note": "this tool call has already returned"}
    assert plugin_api.STEP_STOP not in outcome, "a call already typed is not typed again"


@pytest.mark.parametrize("found", ["", "no_task"])
@pytest.mark.parametrize("meanwhile", ["returned", "typed", "refused"])
def test_the_calls_record_decides_again_after_the_read_whatever_the_read_found(tmp_path, task_ctx, monkeypatch,
                                                                                 meanwhile, found):
    """The read holds no lock. A call that returned while it read is refused (no typing would read a latch set now),
    and a refusal another thread of the same call latched meanwhile is answered: never a fresh ``continue``."""
    api = _api(tmp_path)
    with _call(task_ctx) as outcome:
        def read(source):
            if meanwhile == "typed":
                assert plugin_api.close_stop_binding(source) is None
                return found
            with outcome["phase_lock"]:
                if meanwhile == "returned":
                    outcome["closed"] = True
                else:
                    outcome[plugin_api.STEP_STOP] = {"reason": "cancelled", "note": CANCELLED["note"], "refusals": 1}
            return found  # this read itself saw no Stop

        monkeypatch.setattr(plugin_api, "_read_call_stop", read)
        state = api.get_stop_state()
    if meanwhile in {"returned", "typed"}:
        assert state == {"status": "stop", "reason": "operation_already_returned",
                         "note": "this tool call has already returned"}
        assert plugin_api.STEP_STOP not in outcome, "a call already typed is not typed again"
    else:
        assert state == CANCELLED and outcome[plugin_api.STEP_STOP]["refusals"] == 2


@pytest.mark.parametrize("task_id, managed, expected", [
    ("", False, ("continue", "no_task")),  # an operation run for no task
    ("standalone", False, ("continue", "no_task")),  # attribution only: no task lifecycle, no row
    ("ghost", True, ("stop", "stop_unreadable")),  # a managed task whose row is missing: unknown, never "no Stop"
])
def test_a_call_for_no_task_continues_and_a_managed_task_without_its_row_is_refused(
        tmp_path, task_ctx, task_id, managed, expected):
    _stop(task_ctx, "root", scope="cascade")  # an open cascade elsewhere does not break a call it cannot govern
    task_ctx.task_id, task_ctx.root_task_id, task_ctx.task_lifecycle_bound = task_id, task_id, managed
    with _call(task_ctx):
        state = _api(tmp_path).get_stop_state()
    assert (state["status"], state["reason"]) == expected and state["note"]


@pytest.mark.parametrize("managed, reason", [(True, "stop_unreadable"), (False, "no_task")])
def test_missing_task_id_keeps_its_lifecycle_binding_across_tool_transport(tmp_path, managed, reason):
    from ouroboros import extension_process_runner as runner

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_lifecycle_bound = managed
    wire = json.loads(json.dumps(runner._tool_context_payload(ctx)))
    child = runner._apply_tool_context_payload(ToolContext(repo_dir=tmp_path, drive_root=tmp_path), wire)
    assert child.task_lifecycle_bound is managed
    assert plugin_api._read_call_stop(ctx) == plugin_api._read_call_stop(child) == reason


@pytest.mark.parametrize("fault", ["projection", "busy row"])
def test_stop_authority_that_cannot_be_read_refuses_and_is_never_read_as_no_stop(tmp_path, task_ctx, monkeypatch,
                                                                                 fault):
    from ouroboros import task_results
    from ouroboros.store import StoreBusy

    if fault == "projection":
        (task_ctx.drive_root / "state").mkdir(exist_ok=True)
        (task_ctx.drive_root / "state" / "cancel_intents.json").write_text("{not json", encoding="utf-8")
    else:
        def busy(root, task_id, **kwargs):
            raise StoreBusy("the store is busy", reason="contention")

        monkeypatch.setattr(task_results, "read_document", busy)
    with _call(task_ctx):
        state = _api(tmp_path).get_stop_state()
    assert state == {"status": "stop", "reason": "stop_unreadable", "note": "this task's Stop could not be read"}


@pytest.mark.parametrize("managed, expected", [(True, ("stop", "stop_unreadable")), (False, ("continue", "no_task"))])
def test_a_task_without_its_lineage_is_never_read_as_no_stop_or_as_an_owner_stop(tmp_path, task_ctx, monkeypatch,
                                                                                  managed, expected):
    """Without the tree root, ``immediate_stop_pending`` reads only its fail-soft pending cancel, which takes this
    corrupt projection for "no Stop". The reader never asks it so: a managed task is refused as unreadable, and no
    owner Stop is invented; the shared predicate itself is unchanged."""
    from ouroboros import cancel_intents, owner_pause

    (task_ctx.drive_root / "state").mkdir(exist_ok=True)
    (task_ctx.drive_root / "state" / "cancel_intents.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(owner_pause, "_member_coordinates", lambda source: (str(task_ctx.drive_root), "", "root"))
    real, asked = cancel_intents.immediate_stop_pending, []

    def spy(*args, **kwargs):
        asked.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cancel_intents, "immediate_stop_pending", spy)
    task_ctx.task_lifecycle_bound = managed
    with _call(task_ctx):
        state = _api(tmp_path).get_stop_state()
    assert (state["status"], state["reason"]) == expected and state["note"]
    assert asked == [], "the predicate is never asked without the tree root that makes its projection read strict"


def test_the_row_is_read_after_the_projection_so_a_stop_settled_between_them_still_refuses(tmp_path, task_ctx,
                                                                                         monkeypatch):
    from ouroboros import cancel_intents
    from ouroboros.task_results import write_task_result

    data = task_ctx.drive_root
    _stop(task_ctx, "root")
    real = cancel_intents.immediate_stop_pending

    def settled_meanwhile(*args, **kwargs):
        write_task_result(data, "root", "cancelled", result="stopped")  # Stop custody ended the task...
        cancel_intents.settle_intent(data, "root", outcome="test")  # ...and retired its intent
        return real(*args, **kwargs)

    monkeypatch.setattr(cancel_intents, "immediate_stop_pending", settled_meanwhile)
    with _call(task_ctx):
        state = _api(tmp_path).get_stop_state()
    assert state == {"status": "stop", "reason": "task_cancelled", "note": "this task has already ended as cancelled"}
    assert "owner" not in state["note"], "a terminal status claims no cause"


def test_a_legacy_payload_is_refused_the_method(tmp_path, task_ctx):
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError

    with _call(task_ctx), pytest.raises(ExtensionRegistrationError, match='get_stop_state needs plugin_api "2.6"'):
        _api(tmp_path, generation="1.3").get_stop_state()


# --- the per-call child: its one binding and the transport field --------------------------------------------------

@pytest.mark.parametrize("mode", ["tool", "ws"])
def test_the_child_binds_only_a_tool_call_and_returns_its_refusal_beside_the_result_after_a_raise(
        tmp_path, task_ctx, monkeypatch, mode):
    from ouroboros import extension_process_runner as runner

    api, seen = _api(tmp_path), []

    async def steps(*_args):
        seen.append(api.get_stop_state())
        _stop(task_ctx, "root")
        seen.append(api.get_stop_state())
        raise RuntimeError("the extension raised after the refusal")

    monkeypatch.setattr(runner, "_load_child_extension", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_call_tool", steps)
    monkeypatch.setattr(runner, "_call_ws", steps)
    payload = {"mode": mode, "skill_name": SKILL, "surface": "send", "drive_root": str(task_ctx.drive_root),
               "repo_dir": str(tmp_path), "result_path": str(tmp_path / "result.json"),
               "ctx": runner._tool_context_payload(task_ctx)}
    (tmp_path / "input.json").write_text(json.dumps(payload), encoding="utf-8")
    runner._child_main(str(tmp_path / "input.json"))
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False and "raised after the refusal" in result["error"]
    if mode == "tool":
        assert seen == [CONTINUE, CANCELLED]
        assert result[plugin_api.STEP_STOP] == {"reason": "cancelled", "note": CANCELLED["note"], "refusals": 1}
    else:  # a WebSocket handler is no tool call: nothing to read, nothing reported
        assert [state["status"] for state in seen] == ["unbound", "unbound"] and plugin_api.STEP_STOP not in result
    assert api.get_stop_state()["status"] == "unbound", "the binding ended with the call: no later child inherits it"


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_async_handler_freezes_its_receipt_before_loop_callbacks(tmp_path, task_ctx, monkeypatch, child, raises):
    from ouroboros import extension_loader, extension_process_runner as runner
    from ouroboros.tools import extension_dispatch

    api, seen = _api(tmp_path), []
    release, workers = threading.Event(), []
    report, problem = {"phase": "handler"}, ValueError("handler phase")
    monkeypatch.setattr(plugin_api, "_read_call_stop", lambda _ctx: "cancelled")

    def after_return():
        seen.append(api.get_stop_state()["reason"])
        report["phase"] = "cleanup"
        problem.args = ("cleanup phase",)
        if not child:
            release.wait(15)  # keep the loop alive past the outer join

    async def handler():
        workers.append(threading.current_thread())
        seen.append(api.get_stop_state()["reason"])
        asyncio.get_running_loop().call_soon(after_return)
        if raises:
            raise problem
        return report

    descriptor = {"handler": handler, "wants_ctx": False, "skill": SKILL, "timeout_sec": 1}
    if child:
        monkeypatch.setattr(runner, "_load_child_extension", lambda *_a, **_k: None)
        monkeypatch.setattr(extension_loader, "get_tool", lambda _name: descriptor)
        path = tmp_path / "input.json"
        output = tmp_path / "result.json"
        path.write_text(json.dumps({"mode": "tool", "skill_name": SKILL, "surface": NAME,
                                    "drive_root": str(task_ctx.drive_root), "repo_dir": str(tmp_path),
                                    "result_path": str(output), "ctx": runner._tool_context_payload(task_ctx)}))
        runner._child_main(str(path))
        result = json.loads(output.read_text())
        assert api.get_stop_state()["reason"] == "unbound"
        assert result[plugin_api.STEP_STOP]["refusals"] == 1
        assert result["ok"] is (not raises)
        if raises:
            assert result["error"] == "ValueError: handler phase"
        else:
            assert result["result"] == {"phase": "handler"}
    else:
        monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
        monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
        with _call(task_ctx) as call:
            try:
                result = extension_dispatch._dispatch_extension_tool_result(task_ctx, NAME, descriptor, {}, handoff=call)
                assert not call["settled"] and call["local_holds"] == 1
            finally:
                release.set()
                for worker in workers:
                    worker.join(5)
            assert call["settled"]
            assert call[plugin_api.STEP_STOP]["refusals"] == 1
        assert result.meta["control_reason"] == "cancelled" and result.meta["refused_step"]
        if raises:
            assert result.code == "EXTENSION_ERROR" and "ValueError: handler phase" in result.text
            assert result.producer_text is None
        else:
            assert result.code == "OWNER_OPERATION_REFUSED" and result.producer_text == "{'phase': 'handler'}"
    assert seen == ["cancelled", "operation_already_returned"]


def test_the_childs_one_binding_refuses_a_second_and_leaves_nothing_after_its_release(tmp_path, task_ctx):
    api = _api(tmp_path)
    call = plugin_api.bind_child_tool_call(SKILL, task_ctx)
    try:
        with pytest.raises(RuntimeError, match="already serves a tool call"):
            plugin_api.bind_child_tool_call(SKILL, task_ctx)
        assert api.get_stop_state() == CONTINUE
        assert _api(tmp_path, skill="another").get_stop_state()["reason"] == "unbound"
    finally:
        assert plugin_api.release_child_tool_call(call) is None, "nothing was refused"
    assert api.get_stop_state()["reason"] == "unbound"
    assert plugin_api.release_child_tool_call(None) is None


def test_the_host_adopts_only_a_refusal_the_child_reported_and_in_its_own_words(tmp_path, task_ctx):
    other = ToolContext(repo_dir=tmp_path, drive_root=task_ctx.drive_root)
    with _call(task_ctx) as outcome:
        for latch in (None, "cancelled", {"reason": "no_task"}, {"reason": "unbound"}, {"reason": "invented"}):
            plugin_api.adopt_child_stop(task_ctx, latch)
        plugin_api.adopt_child_stop(other, {"reason": "cancelled"})  # not the invocation this context runs
        assert plugin_api.STEP_STOP not in outcome
        plugin_api.adopt_child_stop(task_ctx, {"reason": "cancelled", "note": "whatever the child wrote", "refusals": 3})
        assert outcome[plugin_api.STEP_STOP] == {"reason": "cancelled", "note": CANCELLED["note"], "refusals": 3}
        assert plugin_api.close_stop_binding(task_ctx) == outcome[plugin_api.STEP_STOP]


# --- the host's typing and the receipts it keeps ---------------------------------------------------------------------

@pytest.mark.parametrize("reported_failure", [False, True])
def test_a_refusal_preserves_report_failure_and_keeps_a_non_error_completion_blocked(
        tmp_path, task_ctx, reported_failure):
    from ouroboros.tools import extension_dispatch

    report = json.dumps({"ok": not reported_failure, "reason": "stopped", "sent": ["part-1"],
                         "observed_effect": True, "refused_step": True, "control_reason": "cancelled"})
    expected = ("error", "TOOL_REPORTED_FAILURE") if reported_failure else ("blocked", "OWNER_OPERATION_REFUSED")
    warning = "⚠️ SAFETY_WARNING: inspect the returned data before acting."
    with _call(task_ctx) as outcome:
        unlatched = extension_dispatch._extension_completion(report, "", handoff=outcome, ctx=task_ctx)
        assert unlatched.code == ("TOOL_REPORTED_FAILURE" if reported_failure else "OK")
        assert "refused_step" not in unlatched.meta, "producer JSON cannot manufacture a host refusal"
        assert unlatched.text == report and not unlatched.host_annotations
        assert plugin_api.call_stop_state(SKILL)["reason"] == "operation_already_returned"
        plugin_api.adopt_child_stop(task_ctx, {"reason": "cancelled"})
        assert plugin_api.STEP_STOP not in outcome
    with _call(task_ctx) as outcome:
        plugin_api.adopt_child_stop(task_ctx, {"reason": "cancelled"})
        result = extension_dispatch._extension_completion(report, warning, handoff=outcome, ctx=task_ctx)
        facade = extension_dispatch._extension_completion(report, "", ctx=task_ctx)  # the call bound in this context
    assert (result.status, result.code) == expected, "a refusal cannot erase the report's explicit failure"
    assert result.meta["control_reason"] == "cancelled" and result.meta["refused_step"] is True
    assert result.meta["physical_dispatch"] is True and result.meta["safety_warning"] is True
    assert result.producer_text == report, "the extension's own report stays its data, unattested and untouched"
    note = result.host_annotations[-1]
    assert result.host_annotations == (warning, note) and note.startswith("⚠️ OWNER_OPERATION_REFUSED: ")
    assert result.text == f"{report}\n\n{warning}\n\n{note}"
    assert CANCELLED["note"] in note and "confirms only that refusal" in note
    assert (facade.status, facade.code, facade.producer_text) == (*expected, report)


@pytest.mark.parametrize("status, code, bucket, phrase", [
    ("error", "EXTENSION_ERROR", "extension_error", "A tool step failed."),
    ("timeout", "EXTENSION_TIMEOUT", "timeout", "A tool step timed out."),
])
def test_failure_and_timeout_remain_unresolved_beside_the_host_refusal(task_ctx, status, code, bucket, phrase):
    from ouroboros._outcome_tool_errors import _classify_tool_errors
    from ouroboros.loop_tool_execution import _typed_result_metadata
    from ouroboros.tool_error_words import residual_tool_errors_phrase
    from ouroboros.tools import extension_dispatch

    failure = f"⚠️ TOOL_ERROR ({NAME}): extension tool failed: boom"
    original = extension_dispatch._extension_result(status, code, failure, timeout_sec=3, dispatched=True)
    with _call(task_ctx) as outcome:
        outcome["settled"] = False
        assert extension_dispatch._refused_failure(task_ctx, outcome, original) is original
        assert plugin_api.call_stop_state(SKILL)["reason"] == "operation_already_returned"
        assert outcome["settled"] is False and not outcome.get("local_extension_returned")
    with _call(task_ctx) as outcome:
        plugin_api.adopt_child_stop(task_ctx, {"reason": "cancelled"})
        refused = extension_dispatch._refused_failure(task_ctx, outcome, original)
    assert (refused.status, refused.code) == (status, code)
    assert dict(refused.meta) == {**original.meta, "refused_step": True, "control_reason": "cancelled"}
    assert refused.producer_text is None, "a raise or timeout has no producer receipt"
    assert refused.text == f"{failure}\n\n{refused.host_annotations[-1]}"
    assert refused.host_annotations[-1].startswith("⚠️ OWNER_OPERATION_REFUSED: ")

    for result in (original, refused):
        trace = {"tool_calls": [{"tool": NAME, "is_error": True, "args": {}, "result": result.text,
                                 **_typed_result_metadata(NAME, result.text, True, result)}]}
        buckets = _classify_tool_errors(trace)
        assert [item["status"] for item in buckets["unresolved"]] == [bucket]
        assert buckets["policy_denials"] == []
        assert residual_tool_errors_phrase({"unresolved_tool_errors": buckets["unresolved"],
                                            "cosmetic_tool_errors": buckets["cosmetic"]}) == phrase


@pytest.mark.parametrize("kind, status, code", [
    ("error", "error", "EXTENSION_ERROR"),
    ("timeout", "timeout", "EXTENSION_TIMEOUT"),
])
def test_child_failure_keeps_its_type_and_dispatch_provenance_after_refusal(
        task_ctx, monkeypatch, kind, status, code):
    from ouroboros import extension_process_runner as runner
    from ouroboros.tools import extension_dispatch

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_k: True)
    monkeypatch.setattr(extension_dispatch, "_replaced_since_bound", lambda *_a: False)

    def child(_tool, ctx, _args):
        plugin_api.adopt_child_stop(ctx, {"reason": "cancelled"})
        raise runner._mark_child_spawned(runner.ExtensionProcessError("child failed", failure_kind=kind))

    monkeypatch.setattr(runner, "dispatch_extension_tool_subprocess", child)
    descriptor = {"skill": SKILL, "out_of_process": True, "timeout_sec": 3,
                  "extension_generation": "generation", "content_hash": "payload"}
    with _call(task_ctx) as outcome:
        result = extension_dispatch._dispatch_extension_tool_result(task_ctx, NAME, descriptor, {}, handoff=outcome)
    assert (result.status, result.code) == (status, code)
    assert result.meta["physical_dispatch"] is True
    assert result.meta["extension_generation"] == "generation" and result.meta["content_hash"] == "payload"
    assert result.meta["control_reason"] == "cancelled" and result.meta["refused_step"] is True
    assert result.meta.get("timeout_sec") == (3 if kind == "timeout" else None)
    assert result.producer_text is None and "child failed" in result.text
    assert result.host_annotations[-1].startswith("⚠️ OWNER_OPERATION_REFUSED: ")


@pytest.mark.parametrize("reported_failure", [False, True])
def test_the_live_finished_frame_keeps_failure_and_host_refusal_separate(monkeypatch, reported_failure):
    """The step row reads a refused call by the host's mark before a reload too; nothing else of the meta goes live."""
    from ouroboros import loop_tool_execution as execution
    from ouroboros.tools import extension_dispatch

    frames: list = []
    monkeypatch.setattr(execution, "_emit_live_log", lambda _tools, payload: frames.append(payload))
    latched = {"phase_lock": threading.Lock(),
               plugin_api.STEP_STOP: {"reason": "cancelled", "note": CANCELLED["note"], "refusals": 1}}
    report = json.dumps({"ok": not reported_failure, "reason": "stopped", "sent": ["part-1"], "observed_effect": True})
    for typed in (extension_dispatch._extension_completion(report, "", handoff=latched),
                  extension_dispatch._extension_completion(report, "")):
        is_error = typed.status != "ok"
        meta = execution._typed_result_metadata(NAME, typed.text, is_error, typed)
        execution._emit_finished(SimpleNamespace(), {"tool": NAME, "invocation_id": "i"},
                                 {"result": typed.text, "is_error": is_error, "result_meta": meta}, 0.0)
    assert frames[0]["status"] == ("tool_reported_failure" if reported_failure else "blocked")
    assert frames[0]["is_error"] is True
    assert frames[0]["tool_result_meta"] == {"refused_step": True, "control_reason": "cancelled"}
    assert "tool_result_meta" not in frames[1], "a call the host refused nothing publishes no producer meta live"
    assert frames[1]["status"] == ("tool_reported_failure" if reported_failure else "ok")
    assert frames[1]["is_error"] is reported_failure


def test_result_review_reads_the_receipt_of_what_ran_before_the_refusal_and_nothing_from_a_raise(tmp_path,
                                                                                                monkeypatch):
    from ouroboros import loop_tool_execution as execution
    from ouroboros.result_material import _declared_effects
    from ouroboros.tools import extension_dispatch

    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    (drive / "logs").mkdir(parents=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="refused-source")
    ctx.task_metadata = {"task_id": ctx.task_id}
    latched = {"phase_lock": threading.Lock(),
               plugin_api.STEP_STOP: {"reason": "cancelled", "note": CANCELLED["note"], "refusals": 1}}
    report = json.dumps({"ok": False, "reason": "stopped", "sent": ["part-1"], "observed_effect": True})
    partial = extension_dispatch._extension_completion(report, "", handoff=latched)
    raised = extension_dispatch._refused_failure(None, latched, extension_dispatch._extension_result(
        "error", "EXTENSION_ERROR", f"⚠️ TOOL_ERROR ({NAME}): extension tool failed: boom", dispatched=True))
    monkeypatch.setattr(execution, "persist_call", lambda *a, **k: {})
    rows = []
    for call_id, result in (("call-partial", partial), ("call-raised", raised)):
        registry = SimpleNamespace(_ctx=ctx, CODE_TOOLS=frozenset(), execute_result=lambda *a, result=result: result)
        rows.append(execution._execute_single_tool(
            registry, {"id": call_id, "function": {"name": NAME, "arguments": "{}"}}, drive / "logs", ctx.task_id))
    trace = {"tool_calls": []}
    execution.process_tool_results(rows, [], trace, lambda *a, **k: None, SimpleNamespace(_ctx=ctx))
    ctx._execution_trace = trace
    assert [call["status"] for call in trace["tool_calls"]] == ["tool_reported_failure", "extension_error"]
    issues: list = []
    effects = _declared_effects(ctx, drive, ctx.task_id, issues)
    assert issues == []
    assert effects == [{"tool": NAME, "receipt": report, "host_notes": [partial.host_annotations[-1]]}]


# --- the whole path: the registry, in process and in a per-call child ---------------------------------------------

PARTS = '''import json


def _send(ctx, parts=2, stop=False, after_stop="ok"):
    sent = []
    for index in range(int(parts)):
        state = API.get_stop_state()  # the host's Stop, read last before each part starts, the first one included
        if state["status"] != "continue":
            if after_stop == "raise":
                raise RuntimeError("failed after refusal")
            return json.dumps({"ok": after_stop != "reported_failure", "reason": "stopped", "sent": sent, "host": state,
                               "observed_effect": bool(sent)})
        sent.append("part-%d" % (index + 1))
        if stop and index == 0:  # the owner presses Stop while the first part is on its way
            from ouroboros.cancel_intents import request_cancel
            request_cancel(ctx.drive_root, ctx.task_id, reason="owner Stop", source="test",
                           requested_stop_policy="immediate")
    return json.dumps({"ok": True, "sent": sent, "observed_effect": True})


def register(api):
    global API
    API = api
    api.register_tool("send", _send, description="Send a message in parts.",
                      schema={"type": "object", "properties": {"parts": {"type": "integer"},
                                                               "stop": {"type": "boolean"},
                                                               "after_stop": {"type": "string"}}})
'''


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("after_stop, status, code", [
    ("ok", "blocked", "OWNER_OPERATION_REFUSED"),
    ("reported_failure", "error", "TOOL_REPORTED_FAILURE"),
    ("raise", "error", "EXTENSION_ERROR"),
])
def test_a_message_in_parts_keeps_its_outcome_beside_the_refusal(
        tmp_path, monkeypatch, isolated, asynchronous, after_stop, status, code):
    from ouroboros import config, extension_loader
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path, SKILL, PARTS.replace("def _send(", "async def _send(") if asynchronous else PARTS,
        permissions=["tool"],
        extra_frontmatter='plugin_api: "2.6"\n' + ("dependencies:\n  - dummy_pkg\n" if isolated else ""))
    if isolated:
        _add_fake_native_dep(loaded)
        _mark_isolated_deps_installed(drive_root, loaded)
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None
    assert bool(extension_loader.get_tool(NAME).get("out_of_process")) is isolated
    monkeypatch.setattr(config, "DATA_DIR", drive_root)
    write_task_result(drive_root, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=REPO, drive_root=drive_root)
    registry.set_context(ToolContext(repo_dir=REPO, drive_root=drive_root, task_id="root",
                                     task_metadata={"root_task_id": "root", "work_kind": "owner"}))
    registry._ctx.task_lifecycle_bound = True

    whole = registry.execute_result(NAME, {"parts": 2})
    assert whole.code == "OK" and json.loads(whole.text)["sent"] == ["part-1", "part-2"], whole

    stopped = registry.execute_result(NAME, {"parts": 3, "stop": True, "after_stop": after_stop})
    assert (stopped.status, stopped.code) == (status, code), stopped
    if after_stop == "raise":
        assert stopped.producer_text is None and "failed after refusal" in stopped.text
    else:
        report = json.loads(stopped.producer_text)
        assert report["ok"] is (after_stop == "ok")
        assert report["sent"] == ["part-1"] and report["host"] == CANCELLED and report["observed_effect"] is True
        assert stopped.text.startswith(stopped.producer_text)
    assert stopped.meta["control_reason"] == "cancelled" and stopped.meta["refused_step"] is True
    assert stopped.meta["physical_dispatch"] is True
    assert stopped.host_annotations[-1].startswith("⚠️ OWNER_OPERATION_REFUSED: ")


@pytest.mark.parametrize("failure", ["", "helper_failure", "target_lost"])
def test_actual_key_keeps_stop_and_independent_failure_distinct(tmp_path, monkeypatch, task_ctx, failure):
    """Real host Stop and dispatcher; every desktop read/input uses mac's subprocess test double."""
    from tests._unix_computer_use import SAFARI, inputs, mac, receipt
    from ouroboros.owner_pause import tool_handoff
    from ouroboros.tools import extension_dispatch

    api = _api(tmp_path, skill="unix_computer_use")

    def on_input(_argv, device):
        _stop(task_ctx)
        if failure:  # an independent failure can occur after the same call latched a Stop
            assert api.get_stop_state()["status"] == "stop"
        if failure == "target_lost":
            device["front"] = SAFARI

    module, _plain, device = mac(
        tmp_path, monkeypatch, on_input=on_input,
        events=(1, "", "input helper failed") if failure == "helper_failure" else (0, "", ""))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_k: True)
    name = extension_surface_name("unix_computer_use", "key")
    descriptor = {"skill": "unix_computer_use", "handler": module._ComputerUse(api).key, "wants_ctx": False}
    with tool_handoff(task_ctx, name) as handoff:
        result = extension_dispatch._dispatch_extension_tool_result(
            task_ctx, name, descriptor, {"keys": "down down return", "app": "Keynote"}, handoff=handoff)
        assert handoff[plugin_api.STEP_STOP]["reason"] == "cancelled"
    report = receipt(result.producer_text)
    assert result.meta["physical_dispatch"] is True and result.meta["refused_step"] is True
    assert result.meta["control_reason"] == "cancelled"
    assert result.host_annotations[-1].startswith("⚠️ OWNER_OPERATION_REFUSED: ")
    assert report["observed_effect"] is True and report["steps_total"] == 3
    assert inputs(device) == [[{"op": "keys", "mods": [], "code": 125}]], "no second input, including on failure"
    if failure:
        assert (result.status, result.code) == ("error", "TOOL_REPORTED_FAILURE")
        assert report["ok"] is False and report.get("status") != "stopped"
        if failure == "helper_failure":
            assert report["steps_done"] == 0 and report["dispatched"] is True
            assert "input helper failed" in report["error"] and report["activation_done"] is True
        else:
            assert report["reason"] == "target_lost" and report["steps_done"] == 1
            assert report["dispatched"] is False
    else:
        assert (result.status, result.code) == ("blocked", "OWNER_OPERATION_REFUSED")
        assert report["status"] == report["reason"] == "stopped"
        assert "ok" not in report and "error" not in report
        assert report["steps_done"] == 1 and report["dispatched"] is False
        assert "cancelled" in report["note"] and "the next step was not sent" in report["note"]
