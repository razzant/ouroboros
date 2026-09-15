"""Real builtin refusals keep producer facts through the string handler ABI."""

from types import SimpleNamespace

import pytest

from ouroboros.tools.tool_result import (
    TOOL_CODE_SPECS,
    LegacyTextResultAdapter,
    ToolResult,
    _install_tool_result_sidecar,
    _published_tool_result,
    _restore_tool_result_sidecar,
)


def _call(ctx, function, *args, **kwargs):
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = function(ctx, *args, **kwargs)
        result = _published_tool_result(ctx, sentinel)
        assert isinstance(result, ToolResult)
        assert result.text == (text["message"] if isinstance(text, dict) else text)
        return result
    finally:
        _restore_tool_result_sidecar(token)


@pytest.mark.parametrize("producer", ["commit", "review_only"])
def test_empty_commit_message_is_an_argument_refusal_before_git(producer, monkeypatch):
    from ouroboros.tools import git, git_review_cycle

    monkeypatch.setattr(git, "_reset_commit_review_state", lambda _ctx: None)
    ctx = SimpleNamespace()
    function = git._repo_commit_push if producer == "commit" else git_review_cycle._run_non_committing_review_cycle
    result = _call(ctx, function, "")
    assert result.status == "error"
    assert result.code == "TOOL_ARG_ERROR"
    assert result.text == "⚠️ ERROR: commit_message must be non-empty."


@pytest.mark.parametrize("stored,code", [
    ({}, "LEGACY_UNAVAILABLE"),
    ({"status": "completed"}, "LEGACY_BLOCKED"),
    ({"status": "pending"}, "LEGACY_BLOCKED"),
])
def test_forwarding_refuses_unaddressable_tasks_before_mailbox_write(tmp_path, monkeypatch, stored, code):
    from ouroboros.tools import core
    import ouroboros.owner_mailbox as mailbox
    import ouroboros.task_status as task_status

    writes = []
    monkeypatch.setattr(core, "canonical_data_root", lambda _ctx: tmp_path)
    monkeypatch.setattr(task_status, "load_effective_task_result", lambda *_: stored)
    monkeypatch.setattr(mailbox, "write_task_message", lambda *_a, **_k: writes.append(1))
    result = _call(SimpleNamespace(drive_root=tmp_path), core._forward_to_worker, "missing-fixture", "hello")
    assert result.status != "ok"
    assert result.code == code
    assert writes == []


@pytest.mark.parametrize("function,arguments", [
    ("_update_scratchpad", {"content": ""}),
    ("_update_identity", {"content": "short"}),
    ("_send_user_message", {"text": ""}),
])
def test_invalid_cognitive_or_message_arguments_are_not_success(function, arguments):
    from ouroboros.tools import control_runtime

    result = _call(SimpleNamespace(current_chat_id=1), getattr(control_runtime, function), **arguments)
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")


@pytest.mark.parametrize("arguments,code", [
    ({}, "TOOL_ARG_ERROR"),
    ({"run_at": "not-a-date", "objective": "check"}, "TOOL_ARG_ERROR"),
    ({"run_at": "2099-01-01T00:00:00Z"}, "TOOL_ARG_ERROR"),
])
def test_followup_rejections_never_schedule(tmp_path, monkeypatch, arguments, code):
    from ouroboros.tools import followup
    from supervisor import queue

    writes = []
    monkeypatch.setattr(queue, "upsert_scheduled_task", lambda *_a, **_k: writes.append(1))
    ctx = SimpleNamespace(task_id="root", drive_root=tmp_path, task_metadata={})
    result = _call(ctx, followup._handle_schedule_followup, **arguments)
    assert (result.status, result.code) == ("error", code)
    assert writes == []


def test_knowledge_and_registry_argument_refusals_leave_files_untouched(tmp_path):
    from ouroboros.tools import knowledge, memory_tools

    ctx = SimpleNamespace(drive_root=tmp_path)
    for function, args in (
        (knowledge._knowledge_read, ("../private",)),
        (knowledge._knowledge_write, ("valid", "content", "invalid-mode")),
        (memory_tools._memory_update_registry, ("../private", "content")),
    ):
        result = _call(ctx, function, *args)
        assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert list(tmp_path.iterdir()) == []


def test_presence_contract_refusal_and_valid_completion_remain_distinct():
    from ouroboros.tools.presence import _finish_presence

    ctx = SimpleNamespace(task_contract={})
    result = _call(ctx, _finish_presence, "message", "hello")
    assert result.status == "unavailable"
    assert not hasattr(ctx, "_presence_completion")
    ctx.task_contract = {"capability_ceiling": {}}
    assert _finish_presence(ctx, "message", "hello").startswith("PRESENCE_COMPLETION_RECORDED")
    assert ctx._presence_completion == {"outcome": "message", "message": "hello"}


def test_real_producer_failure_survives_registry_dispatch(tmp_path, monkeypatch):
    from ouroboros.tools import control_runtime
    from ouroboros.tools.registry import ToolEntry, ToolRegistry
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.current_chat_id = 1
    registry.register(ToolEntry("fixture_builtin", {
        "name": "fixture_builtin", "description": "fixture",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }, lambda ctx: control_runtime._send_user_message(ctx, "")))
    result = registry.execute_result("fixture_builtin", {})
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert result.text == "⚠️ Empty message."


@pytest.mark.parametrize("text,code", [
    ("⚠️ WARNING: untracked files remain", "LEGACY_WARNING"),
    ("⚠️ REVIEW_BLOCKED: address findings", "REVIEW_BLOCKED"),
    ("⚠️ GIT_ERROR: inspect the refusal", "GIT_ERROR"),
])
def test_existing_warning_and_review_policy_are_not_blanket_reclassified(text, code):
    result = LegacyTextResultAdapter.from_text("fixture", text)
    assert (result.status, result.code, result.text) == ("ok", code, text)


# --- the external-executor family (owner Q8A) --------------------------------
#
# `delegate_start`/`delegate_wait`/`delegate_cancel`/`delegate_answer` speak a
# native `ToolResult` among themselves and project a `str` at their four
# registered entries. The incident these pin: `_fail` used to render
# `{"status": "refused", ...}` as a plain string, which the registry's legacy
# adapter classified as OK — so a refused wait/cancel (daemon unreachable, run
# not owned, containment fault, refused cancel) was recorded as a SUCCESSFUL
# tool call on the outcome axis, in the acceptance packet, and in the
# supervising task's own reasoning.


def _delegate_registry(tmp_path, monkeypatch, task_id="t-family"):
    """A real registry, with the family's four entries registered as production does."""
    from ouroboros.tools.registry import ToolRegistry
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = task_id
    registry._ctx.task_metadata = {"root_task_id": task_id, "parent_task_id": task_id}
    assert {"delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer"} <= set(
        registry._entries)
    return registry


# Each row is a call that refuses BEFORE any daemon work, so the dispatch
# contract is exercised without a transport at all.
_PRE_DAEMON_REFUSALS = [
    ("delegate_start", {"prompt": "do the work"}, "subagent_selection_required", "TOOL_ARG_ERROR"),
    ("delegate_wait", {"run_id": "run-1", "checkpoint_after_sec": 60},
     "checkpoint_requires_time_and_reason", "TOOL_ARG_ERROR"),
    ("delegate_cancel", {"run_id": ""}, "missing_run_id", "TOOL_ARG_ERROR"),
    ("delegate_answer", {"run_id": "", "interaction_id": "i-1", "answers": [{"question_id": "q"}]},
     "missing_run_id", "TOOL_ARG_ERROR"),
]


@pytest.mark.parametrize("tool,args,reason,code", _PRE_DAEMON_REFUSALS)
def test_every_delegate_verb_publishes_one_native_refusal_through_the_registry(
    tmp_path, monkeypatch, tool, args, reason, code,
):
    import json

    registry = _delegate_registry(tmp_path, monkeypatch)
    result = registry.execute_result(tool, dict(args))
    assert (result.status, result.code) == (TOOL_CODE_SPECS[code].status, code)
    # The handler ABI is unchanged: the published text IS the string the entry returned.
    assert registry.execute(tool, dict(args)) == result.text
    payload = json.loads(result.text)
    # The envelope is ADDITIVE: the domain reason keeps its own name and vocabulary.
    assert payload["status"] == "refused"
    assert payload["reason"] == reason
    assert payload["ok"] is False
    assert payload["host_code"] == code
    assert payload["tool"] == tool


def test_a_refused_delegate_call_is_no_longer_recorded_as_a_successful_call(tmp_path, monkeypatch):
    """The incident, end to end: the text-only reader and the typed reader agree."""
    from ouroboros._outcome_tool_errors import _classify_tool_errors
    from ouroboros.loop_tool_execution import _typed_execution_failure

    registry = _delegate_registry(tmp_path, monkeypatch)
    result = registry.execute_result("delegate_cancel", {"run_id": ""})
    assert result.code == "TOOL_ARG_ERROR"
    # Even a reader holding only the TEXT now sees the producer's own verdict.
    assert LegacyTextResultAdapter.from_text("delegate_cancel", result.text).status != "ok"
    assert _typed_execution_failure(True, result) is True
    buckets = _classify_tool_errors({"tool_calls": [{
        "tool": "delegate_cancel", "is_error": True,
        "status": TOOL_CODE_SPECS[result.code].outcome_bucket, "result": result.text,
    }]})
    assert [row["tool"] for row in buckets["unresolved"]] == ["delegate_cancel"]


@pytest.mark.parametrize("reason,code", [
    # Substrate: the daemon, the engine, custody or the run said no. Recorded,
    # never degrading — the host refused, the agent did not fail.
    ("daemon_unreachable", "TOOL_REPORTED_FAILURE"),
    ("run_ownership_unknown", "TOOL_REPORTED_FAILURE"),
    ("run_not_owned", "TOOL_REPORTED_FAILURE"),
    ("subscription_window_exhausted", "TOOL_REPORTED_FAILURE"),
    ("home_isolation_breach", "TOOL_REPORTED_FAILURE"),
    ("some_future_engine_code", "TOOL_REPORTED_FAILURE"),
    # Agent faults: the call itself was malformed or self-contradictory.
    ("empty_prompt", "TOOL_ARG_ERROR"),
    ("missing_run_id", "TOOL_ARG_ERROR"),
    ("retry_selector_conflict", "TOOL_ARG_ERROR"),
    ("checkpoint_requires_time_and_reason", "TOOL_ARG_ERROR"),
    ("configured_actor_route_mismatch", "TOOL_ARG_ERROR"),
])
def test_the_refusal_class_separates_a_substrate_no_from_a_malformed_call(reason, code):
    from ouroboros._outcome_tool_errors import _POLICY_DENIAL_STATUSES
    from ouroboros.delegate_shared import _fail

    result = _fail("delegate_wait", reason, "detail")
    assert result.code == code
    # Neither class may be a timeout or the generic tool error: a refusal did not
    # time out, and `TOOL_ERROR` would hide which of the two this was.
    assert result.code not in {"TOOL_TIMEOUT", "TOOL_ERROR"}
    bucket = TOOL_CODE_SPECS[result.code].outcome_bucket
    assert (bucket in _POLICY_DENIAL_STATUSES) is (code == "TOOL_REPORTED_FAILURE")


def test_a_host_note_composed_after_the_refusal_does_not_relabel_it():
    from ouroboros.delegate_shared import _fail
    from ouroboros.tools.tool_result import _compose_execute_result_result

    refusal = _fail("delegate_cancel", "run_not_owned", "another task owns it", run_id="run-x")
    composed = _compose_execute_result_result(
        "delegate_cancel", refusal, "auto-routed to the active room",
        "⚠️ SAFETY_WARNING: check the target",
    )
    assert composed.code == "TOOL_REPORTED_FAILURE"
    assert composed.meta["safety_warning"] is True and composed.meta["route_note"] is True
    assert composed.text.startswith(refusal.text)


def _own_run(tmp_path, run_id="run-1", task_id="t-family"):
    from ouroboros import delegate_custody as custody

    custody._CUSTODY.clear()
    custody._CUSTODY[run_id] = custody.RunCustody(
        run_id=run_id, task_id=task_id, route_id="some-route", model="m")
    return custody


@pytest.mark.parametrize("outcome,expected_code", [
    # A verified terminal receipt, and an ACCEPTED command whose run has not
    # stopped yet, are both successful observations of the control surface.
    ("confirmed", "OK"),
    ("requested", "OK"),
    # The daemon REFUSED the stop, or the stop could not be verified: the run may
    # still be live and mutating, so the call did not succeed.
    ("failed", "TOOL_REPORTED_FAILURE"),
    ("containment_fault_run_may_still_be_live", "TOOL_REPORTED_FAILURE"),
])
def test_cancel_outcomes_split_on_whether_the_run_may_still_be_live(
    tmp_path, monkeypatch, outcome, expected_code,
):
    import json

    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    custody = _own_run(tmp_path)
    monkeypatch.setattr(gw, "ClaudexorGateway",
                        lambda *a, **k: SimpleNamespace(handshake=lambda **_k: {}, close=lambda: None))
    monkeypatch.setattr(custody, "cancel_and_verify", lambda *_a, **_k: {
        "outcome": outcome, "accepted": outcome != "failed", "control_status": "s",
        "state": "running", "fault_reason": "", "detail": "d",
    })
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-family")
    result = delegate._delegate_cancel(ctx, "run-1", "stuck")
    custody._CUSTODY.clear()

    assert result.code == expected_code
    payload = json.loads(result.text)
    assert payload["status"] == outcome
    assert payload["run_may_still_be_live"] is (outcome != "confirmed")
    assert payload["note"] == delegate._CANCEL_NOTES[outcome]
    assert ("ok" in payload) is (expected_code != "OK")


def test_a_confirmed_cancel_over_a_settled_run_is_a_legitimate_no_op(tmp_path, monkeypatch):
    import json

    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    custody = _own_run(tmp_path)
    monkeypatch.setattr(gw, "ClaudexorGateway",
                        lambda *a, **k: SimpleNamespace(handshake=lambda **_k: {}, close=lambda: None))
    monkeypatch.setattr(custody, "cancel_and_verify", lambda *_a, **_k: {
        "outcome": "confirmed", "accepted": False, "control_status": "not_found",
        "state": "succeeded", "fault_reason": "", "detail": "already settled",
    })
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-family")
    result = delegate._delegate_cancel(ctx, "run-1")
    custody._CUSTODY.clear()

    assert result.code == "OK"
    assert json.loads(result.text)["accepted"] is False


@pytest.mark.parametrize("body,status,note_key", [
    ({"status": "delivered", "accepted": True}, "delivered", "delivered"),
    ({"status": "already_resolved", "accepted": False}, "already_resolved", "already_resolved"),
])
def test_answer_outcomes_keep_their_own_semantics_on_the_native_result(
    tmp_path, monkeypatch, body, status, note_key,
):
    import json

    import ouroboros.delegate_interactions as interactions
    from ouroboros.gateways import claudexor as gw

    _own_run(tmp_path)
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: SimpleNamespace(
        handshake=lambda **_k: {}, close=lambda: None,
        answer_interaction=lambda *_a, **_k: dict(body)))
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-family")
    result = interactions._delegate_answer(
        ctx, "run-1", "int-1", [{"question_id": "q1", "free_text": "yes"}])

    # The engine ANSWERED: an outcome is an observation, not a refusal — and
    # `already_resolved` still does not prove that THIS answer won.
    assert result.code == "OK"
    payload = json.loads(result.text)
    assert payload["status"] == status
    assert payload["note"] == interactions._ANSWER_NOTES[note_key]


def test_an_unsupported_engine_build_refuses_the_answer_typed(tmp_path, monkeypatch):
    import json

    import ouroboros.delegate_interactions as interactions
    from ouroboros.gateways import claudexor as gw

    _own_run(tmp_path)

    def _refuse(*_a, **_k):
        raise gw.ClaudexorUnavailable("engine_error", "no service", status_code=501)

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: SimpleNamespace(
        handshake=lambda **_k: {}, close=lambda: None, answer_interaction=_refuse))
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-family")
    result = interactions._delegate_answer(
        ctx, "run-1", "int-1", [{"question_id": "q1", "free_text": "yes"}])

    assert result.code == "TOOL_REPORTED_FAILURE"
    assert json.loads(result.text)["reason"] == "interaction_answers_unsupported"


def test_the_expired_wait_window_and_its_cache_horizon_note_stay_valid_json(tmp_path, monkeypatch):
    """The note is a FIELD of the window payload; appended after the JSON it made
    the whole result unparseable for every reader of this family."""
    import json

    import ouroboros.tools.control as control
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    _own_run(tmp_path)
    monkeypatch.setattr(control, "cache_horizon_note", lambda *_a, **_k: "the prompt cache expires soon")

    class _Alive:
        engine_version = "3.10.2"

        def handshake(self, **_kw): return {}
        def get_run(self, rid, *, timeout_sec=None):
            return {"lastSeq": 1, "summary": {"state": "running", "effectiveAccess": "readonly"}}
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Alive())
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-family")
    raw = delegate._delegate_wait(ctx, "run-1", wait_sec=1, since_seq=0)
    payload = json.loads(raw)          # the contract: still ONE JSON object
    assert payload["cache_horizon_note"] == "the prompt cache expires soon"
    assert payload["status"] in {"progress", "no_progress"}


def _wait_ctx(tmp_path, task_id="t-nanny"):
    return SimpleNamespace(
        task_id=task_id, task_attempt=1, drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        task_metadata={"root_task_id": task_id, "delegation_role": "subagent"},
    )


def _scripted_wait(ticks):
    """One ``wait_once`` replaying an exact sequence of observation TEXTS."""
    remaining = list(ticks)

    def wait_once(_ctx, run_id, *_args, **_kwargs):
        assert remaining, "the supervising loop asked for one tick too many"
        return remaining.pop(0)

    return wait_once, remaining


def test_a_quiet_window_renews_and_only_the_terminal_observation_wakes(tmp_path, monkeypatch):
    import json

    import ouroboros.delegate_supervision as supervision

    monkeypatch.setattr(supervision.time, "sleep", lambda _sec: None)
    wait_once, remaining = _scripted_wait([
        json.dumps({"status": "progress", "run_id": "run-1", "last_seq": 3}),
        json.dumps({"status": "no_progress", "run_id": "run-1", "last_seq": 3}),
        json.dumps({"status": "observation_pending", "run_id": "run-1",
                    "reason": "observation_read_timeout"}),
        json.dumps({"status": "succeeded", "run_id": "run-1", "last_seq": 9}),
    ])
    result = supervision.supervised_wait(_wait_ctx(tmp_path), "run-1", wait_once=wait_once)

    assert remaining == []                       # exactly four ticks, one wake
    # A terminal run is a SUCCESSFUL observation, whatever the leaf's own state.
    assert (result.status, result.code) == ("ok", "OK")
    payload = json.loads(result.text)
    assert payload["status"] == "succeeded"
    assert payload["supervision_wake_id"] == result.meta["supervision_wake_id"]


def test_a_terminal_failed_leaf_is_still_a_successful_observation(tmp_path):
    import json

    import ouroboros.delegate_supervision as supervision

    wait_once, _ = _scripted_wait([
        json.dumps({"status": "terminal", "state": "failed", "run_id": "run-1"}),
    ])
    result = supervision.supervised_wait(_wait_ctx(tmp_path, "t-failed"), "run-1", wait_once=wait_once)
    assert result.code == "OK"
    assert json.loads(result.text)["state"] == "failed"


def test_a_refused_observation_wakes_as_a_refusal_and_replays_as_one(tmp_path, monkeypatch):
    """Small wake, oversized fitted wake, failed spill — then a NEW process whose
    sidecar is gone replays the pending wake with the same class, source and ACK
    and without asking the daemon again."""
    import json

    import ouroboros.delegate_supervision as supervision
    from ouroboros.delegate_shared import _fail

    refusal_text = _fail("delegate_wait", "daemon_unreachable",
                         "the socket carried no answer", run_id="run-1").text
    wait_once, _ = _scripted_wait([refusal_text])
    ctx = _wait_ctx(tmp_path, "t-refused-wake")
    small = supervision.supervised_wait(ctx, "run-1", wait_once=wait_once)
    assert (small.status, small.code) == ("error", "TOOL_REPORTED_FAILURE")
    assert json.loads(small.text)["reason"] == "daemon_unreachable"

    # A NEW process: no sidecar, no gateway, only the durable supervision record.
    replay = supervision.supervised_wait(
        _wait_ctx(tmp_path, "t-refused-wake"), "run-1",
        wait_once=lambda *_a, **_k: pytest.fail("a pending wake must replay, not re-observe"))
    assert replay.code == "TOOL_REPORTED_FAILURE"
    assert replay.meta["supervision_wake_id"] == small.meta["supervision_wake_id"]
    assert supervision.acknowledge_pending_wake(ctx, replay.text) is True


def test_an_oversized_refused_wake_keeps_its_class_in_the_fitted_envelope(tmp_path, monkeypatch):
    import json

    import ouroboros.delegate_supervision as supervision
    import ouroboros.tool_capabilities as capabilities
    from ouroboros.delegate_shared import _fail

    monkeypatch.setattr(capabilities, "tool_result_limit", lambda _name: 2_000)
    refusal_text = _fail("delegate_wait", "home_isolation_breach",
                         "the engine did not contain the run: " + "x" * 4000,
                         run_id="run-1").text
    wait_once, _ = _scripted_wait([refusal_text])
    result = supervision.supervised_wait(
        _wait_ctx(tmp_path, "t-big-refusal"), "run-1", wait_once=wait_once)

    envelope = json.loads(result.text)        # still ONE valid JSON object
    assert len(result.text) <= 2_000
    assert envelope["wake_delivery"]["complete"] is False
    assert result.code == "TOOL_REPORTED_FAILURE"
    assert envelope["ok"] is False and envelope["host_code"] == "TOOL_REPORTED_FAILURE"


def test_a_failed_spill_keeps_the_exact_wake_and_its_class(tmp_path, monkeypatch):
    import json

    import ouroboros.artifacts as artifacts
    import ouroboros.delegate_supervision as supervision
    import ouroboros.tool_capabilities as capabilities
    from ouroboros.delegate_shared import _fail

    monkeypatch.setattr(capabilities, "tool_result_limit", lambda _name: 900)
    monkeypatch.setattr(artifacts, "store_actor_source_bytes",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("no room")))
    wait_once, _ = _scripted_wait([
        _fail("delegate_wait", "run_not_owned", "y" * 4000, run_id="run-1").text,
    ])
    result = supervision.supervised_wait(
        _wait_ctx(tmp_path, "t-spill-failed"), "run-1", wait_once=wait_once)

    # The EXACT wake is kept whole; the ordinary outer truncation fails the ack.
    assert len(result.text) > 900
    assert result.code == "TOOL_REPORTED_FAILURE"
    assert json.loads(result.text)["reason"] == "run_not_owned"


def _schema_1_state(tmp_path, task_id, payload):
    """A supervision record in the OLD writer's shape: no ``ok``, no ``host_code``."""
    import json

    path = tmp_path / "state" / "delegate_supervision" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": 1, "run_id": "run-1", "journal_cursor": 0, "status": "wake_pending",
        "pending_wake": {
            "wake_id": "legacy-wake", "attempt_key": "1", "payload": payload,
            "mailbox_ids": [], "seen_mailbox_ids": [], "interaction_ids": [],
            "created_at": "2026-08-30T00:00:00Z",
        },
    }, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("reason", ["daemon_unreachable", "run_not_owned"])
def test_two_schema_1_refusal_reasons_replay_as_one_recorded_class(tmp_path, reason):
    import json

    import ouroboros.delegate_supervision as supervision

    task_id = f"t-legacy-{reason}"
    _schema_1_state(tmp_path, task_id, {
        "status": "refused", "tool": "delegate_wait", "reason": reason,
        "detail": "written before the envelope existed", "run_id": "run-1",
        "supervision_wake_id": "legacy-wake",
    })
    result = supervision.supervised_wait(
        _wait_ctx(tmp_path, task_id), "run-1",
        wait_once=lambda *_a, **_k: pytest.fail("the stored wake replays, it is not re-observed"))

    assert result.code == "TOOL_REPORTED_FAILURE"
    payload = json.loads(result.text)
    # The stored body is replayed AS WRITTEN; only the two classification keys
    # are derived, and the domain reason keeps its own name.
    assert payload["reason"] == reason
    assert payload["detail"] == "written before the envelope existed"
    assert payload["ok"] is False


def test_the_word_error_inside_a_schema_1_success_is_not_a_refusal(tmp_path):
    import json

    import ouroboros.delegate_supervision as supervision

    _schema_1_state(tmp_path, "t-legacy-ok", {
        "status": "succeeded", "run_id": "run-1", "supervision_wake_id": "legacy-wake",
        "note": "the run reported: 0 errors, 1 warning; error budget untouched",
    })
    result = supervision.supervised_wait(
        _wait_ctx(tmp_path, "t-legacy-ok"), "run-1",
        wait_once=lambda *_a, **_k: pytest.fail("the stored wake replays"))

    assert (result.status, result.code) == ("ok", "OK")
    assert "ok" not in json.loads(result.text)


def test_the_private_cores_never_publish_and_the_entry_publishes_once(tmp_path, monkeypatch):
    """An early publish inside a core is silently discarded by the registry's
    equality gate — so the cores must not touch the published-result slot at all."""
    import ouroboros.delegate_supervision as supervision
    import ouroboros.tools.delegate as delegate
    from ouroboros.subagent_runtime import delegate_start_entry

    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-publish",
                          task_attempt=1, task_metadata={})
    for core, args in (
        (delegate._delegate_cancel, ("",)),
        (delegate_start_entry, ("do the work",)),
        (lambda _c, _r: supervision.supervised_wait(_c, _r, checkpoint_after_sec=5), ("run-1",)),
    ):
        sentinel = object()
        token = _install_tool_result_sidecar(ctx, sentinel)
        try:
            produced = core(ctx, *args)
            assert isinstance(produced, ToolResult)
            assert _published_tool_result(ctx, sentinel) is sentinel
        finally:
            _restore_tool_result_sidecar(token)

    registry = _delegate_registry(tmp_path, monkeypatch, task_id="t-publish")
    published = _call(registry._ctx, lambda _c: registry._entries["delegate_cancel"].handler(
        registry._ctx, run_id=""))
    assert published.code == "TOOL_ARG_ERROR"


def test_the_bootstrap_consumer_reads_the_native_start_result(tmp_path, monkeypatch):
    """A consumer that only ``json.loads``-ed a string would take the $0 unrun
    terminal over a run that may be live — the custody handle decides, not the text."""
    import json

    import ouroboros.subagent_runtime as runtime
    from ouroboros.delegate_shared import _fail, delegate_result
    from ouroboros.subagent_bootstrap import _pre_start_leaf

    def _leaf(result):
        monkeypatch.setattr(runtime, "delegate_start_entry", lambda *_a, **_k: result)
        ctx = SimpleNamespace(task_id="t-boot", drive_root=tmp_path,
                              budget_drive_root=str(tmp_path), task_metadata={},
                              _configured_actor_bootstrap={"selected_subagent_id": "s"})
        return ctx, _pre_start_leaf(ctx, {"id": "t-boot", "configured_subagent": {}}, {})

    ctx, started = _leaf(delegate_result({
        "status": "started", "run_id": "run-live", "invocation_id": "inv-1"}))
    assert json.loads(started)["status"] == "configured_session_started"
    assert ctx._configured_actor_bootstrap["physical_started"] is True

    # A post-POST unknown carries a custody handle: a run may be live, so the
    # model is woken instead of a second physical start being invited.
    ctx, unknown = _leaf(_fail("delegate_start", "daemon_unreachable", "transport died",
                               pending_invocation_id="inv-2"))
    assert json.loads(unknown)["status"] == "configured_session_startup_fault"
    assert getattr(ctx, "_configured_startup_refusal", None) is None

    ctx, definite = _leaf(_fail("delegate_start", "route_disabled", "the route is off",
                                definitely_unrun=True))
    assert definite == ""
    assert ctx._configured_startup_refusal["reason"] == "route_disabled"


def test_the_hold_consumer_resumes_on_a_native_leaf_wake(tmp_path, monkeypatch):
    """The unknown-provider hold: without migration every leaf wake would fail its
    acknowledgement and take the no-resend terminal instead of resuming."""
    import ouroboros.delegate_hold as hold
    from ouroboros.delegate_shared import _fail, delegate_result

    ctx = SimpleNamespace(task_id="t-hold-native", drive_root=tmp_path,
                          exact_model_route=True,
                          task_metadata={"configured_subagent": {"config_fingerprint": "fp"}})
    tools = SimpleNamespace(_ctx=ctx)
    hold.write_unknown_hold(ctx, "run-1", {"run_id": "run-1", "hold_cycles": 1})
    monkeypatch.setattr(hold, "acknowledge_pending_wake", lambda *_a, **_k: True)
    monkeypatch.setattr(hold, "supervised_wait", lambda *_a, **_k: delegate_result({
        "status": "succeeded", "run_id": "run-1", "supervision_wake_id": "w-native"}))
    messages: list = []
    assert hold.hold_step(tools, controls={}, messages=messages, drive_logs=tmp_path,
                          task_id="t-hold-native", emit_progress=lambda _t: None) == "resume"
    assert "[DELEGATED LEAF WAKE" in messages[-1]["content"]

    # A refused wait is a daemon statement, not a leaf wake: no paid resume round.
    hold.write_unknown_hold(ctx, "run-1", {"run_id": "run-1", "hold_cycles": 1})
    monkeypatch.setattr(hold, "supervised_wait", lambda *_a, **_k: _fail(
        "delegate_wait", "daemon_unreachable", "the socket carried no answer"))
    assert hold.hold_step(tools, controls={}, messages=[], drive_logs=tmp_path,
                          task_id="t-hold-native", emit_progress=lambda _t: None) == "terminal"


def test_the_wake_ack_is_keyed_on_the_published_wake_id(tmp_path, monkeypatch):
    """Hygiene: the acknowledgement belongs to the WAKE the result published, not
    to every call spelled ``delegate_wait``; outer truncation must still fail it."""
    import ouroboros.delegate_supervision as supervision
    from ouroboros.loop_tool_execution import process_tool_results

    ctx = _wait_ctx(tmp_path, "t-ack")
    wait_once, _ = _scripted_wait(['{"status": "succeeded", "run_id": "run-1", "last_seq": 4}'])
    wake = supervision.supervised_wait(ctx, "run-1", wait_once=wait_once)
    wake_id = wake.meta["supervision_wake_id"]

    def _run(result_text, meta):
        process_tool_results(
            [{"fn_name": "delegate_wait", "tool_call_id": "call-1", "result": result_text,
              "is_error": False, "args_for_log": {}, "tool_args": {},
              "result_meta": {"status": "ok", "tool_result_meta": meta}}],
            [], {"tool_calls": []}, emit_progress=lambda _m, *, incident=None: None,
            tools=SimpleNamespace(_ctx=ctx),
        )
        return supervision.supervision_checkpoint(ctx)

    # No published wake id: nothing is acknowledged, even on a delegate_wait row.
    assert _run(wake.text, {})["pending_wake"]["wake_id"] == wake_id
    # Truncated delivery with the id present: the ack still fails closed.
    assert _run(wake.text[:40], {"supervision_wake_id": wake_id})["pending_wake"]["wake_id"] == wake_id
    # The exact delivered transcript text acknowledges exactly this wake.
    settled = _run(wake.text, {"supervision_wake_id": wake_id})
    assert settled["pending_wake"] == {}
    assert settled["last_acknowledged_wake"]["wake_id"] == wake_id


@pytest.mark.parametrize("produced,code", [
    ("started", "OK"),
    ("refused", "TOOL_REPORTED_FAILURE"),
])
def test_exact_start_decorates_the_native_result_without_losing_its_class(
    tmp_path, monkeypatch, produced, code,
):
    """The decorator reads the producer's own payload and replaces the text; the
    deleted ``json.loads → TypeError → return result`` bypass used to drop the
    actor identity and the work-order source silently."""
    import json

    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros.delegate_shared import _fail, delegate_result

    core = (delegate_result({"status": "started", "run_id": "run-1"}) if produced == "started"
            else _fail("delegate_start", "queued_without_run_id", "no run id came back",
                       pending_invocation_id="inv-9"))
    monkeypatch.setattr(delegate, "_delegate_start", lambda *_a, **_k: core)
    snapshot = {
        "schema": 1, "selected_subagent_id": "session-builder",
        "config_fingerprint": "cfg-v1",
        "route": {"kind": "agent_session", "target_id": "some-route=weak"}, "effort": "low",
    }
    ctx = SimpleNamespace(task_id="t-exact", drive_root=tmp_path,
                          budget_drive_root=str(tmp_path), task_metadata={})
    result = runtime.exact_start(ctx, "brief", {
        "snapshot": snapshot,
        "work_order_source_request": {"schema": 1, "kind": "source_request"},
    })

    assert result.code == code
    payload = json.loads(result.text)
    assert payload["selected_subagent_id"] == "session-builder"
    assert payload["config_fingerprint"] == "cfg-v1"
    assert payload["work_order_source_request"] == {"schema": 1, "kind": "source_request"}
    assert payload["status"] == ("started" if produced == "started" else "refused")
