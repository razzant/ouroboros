"""#1573 consumer seams: the terminal acknowledgement means "appended in THIS attempt".

``delegate_run_terminal_acknowledged`` retires the owner's own-run health line at
the next context build of the same attempt. These checks drive the production
callers and the one same-attempt rebuild, not the helper alone:

* the loop's tool-result seam behind the real ``delegate_wait`` tool, the
  unknown-provider hold and the configured-session startup receipt each write the
  row only once the wake is already in the transcript they extend, and only with
  the attempt that transcript belongs to;
* an owner-wait planned restart resumes the SAME attempt through ``run_llm_loop``:
  the first request after it omits the own-run line exactly when the restored
  dialogue carries the terminal wake; otherwise the unread-result line names
  the output the body collected during the attempt, named at the loop's exit, without acknowledging a wake;
* the row survives the marker-filtered custody chain and the current closed-run
  receipt, so every custody reader replays the same acknowledgement.

No provider is called: model rounds are scripted at ``_dispatch_round_model`` and
the engine is the ``_Engine`` fixture behind ``ClaudexorGateway``.
"""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import context, loop
from ouroboros import delegate_custody as custody
from ouroboros.context import build_llm_messages
from ouroboros.memory import Memory
from ouroboros.tools.registry import ToolRegistry
from tests.test_delegated_run_discovery import (  # noqa: F401  (fixtures: autouse transport, geo)
    ROOT,
    _owned_gateway_uses_each_test_transport,
    _settle,
    _stage,
    _terminal_wake,
    _top,
)
from tests.test_delegated_run_discovery import geo as _geo

geo = _geo
TASK = "t-own"
OWN_LINE = "OWN DELEGATED RUN ENDED — run run-1"
UNREAD_LINE = "DELEGATED RESULT NEVER READ — run run-1"


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    """Any real model/catalog call fails the test; the route is a confirmed fixture window."""
    def refuse(*_a, **_k):
        raise AssertionError("ack-consumer fixture attempted a provider/catalog call")

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", refuse)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat_async", refuse)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", refuse)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task.get("model") or "fixture-model", "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a"),
    ))
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    custody._CUSTODY.clear()
    yield
    custody._CUSTODY.clear()


class _Env:
    """The minimum ``build_llm_messages`` reads: one repo prompt, one drive."""

    def __init__(self, root):
        self.repo_dir, self.drive_root = root / "repo", root
        for rel in ("repo/prompts", "repo/docs", "memory/knowledge", "logs", "state"):
            (root / rel).mkdir(parents=True, exist_ok=True)
        (root / "repo" / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
        (root / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")

    def drive_path(self, rel):
        return self.drive_root / rel

    def repo_path(self, rel):
        return self.repo_dir / rel


def _registry(tmp_path, attempt=1):
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = TASK, attempt
    ctx.task_metadata = {"root_task_id": TASK}
    return registry


def _fresh_messages(tmp_path, registry, attempt=1):
    """The context the agent builds for ``task['_attempt']`` (the real builder, real health core)."""
    task = {"id": TASK, "type": "task", "text": "supervise run-1", "_attempt": attempt}
    messages, _ = build_llm_messages(env=_Env(tmp_path), memory=Memory(drive_root=tmp_path),
                                     task=task, ctx=registry._ctx)
    return messages


def _system_text(messages):
    assert messages[0]["role"] == "system"
    return json.dumps(messages[0]["content"], ensure_ascii=False)


def _dialogue(messages):
    """The raw text each turn carries (JSON-encoding the list would escape a wake's own quotes)."""
    return "\n".join(row["content"] if isinstance(row.get("content"), str)
                     else json.dumps(row.get("content"), ensure_ascii=False) for row in messages)


def _loop_kwargs(tmp_path, registry, messages):
    return dict(messages=messages, tools=registry, llm=SimpleNamespace(default_model=lambda: "fixture-model"),
                drive_logs=tmp_path / "logs", emit_progress=lambda *_a, **_k: None,
                incoming_messages=queue.Queue(), task_id=TASK, drive_root=tmp_path)


def _acks(root):
    return [(row["run_id"], row["task_id"], row["attempt_key"])
            for row in custody.custody_rows(root) if row["type"] == custody.TERMINAL_ACKNOWLEDGED]


@pytest.fixture
def ack_spy(monkeypatch):
    """At the moment each acknowledgement row lands, what the named transcript already holds."""
    seen, real = [], custody.emit

    def emit(root, kind, payload):
        if kind == custody.TERMINAL_ACKNOWLEDGED and spy.transcript is not None:
            seen.append((payload.get("attempt_key"), json.dumps(spy.transcript, ensure_ascii=False)))
        return real(root, kind, payload)

    spy = SimpleNamespace(seen=seen, transcript=None)
    monkeypatch.setattr(custody, "emit", emit)
    return spy


# --- the loop's tool-result seam, then a same-attempt cold Resume -------------------

@pytest.mark.parametrize("waited", [True, False], ids=["wake-appended", "never-waited"])
def test_a_same_attempt_cold_resume_keeps_the_restored_wake_and_its_ack(tmp_path, monkeypatch, ack_spy, waited):
    from ouroboros.gateways import claudexor as gw
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from ouroboros.task_results import write_task_result

    engine = _settle(tmp_path, task_id=TASK)  # the sweep settled it while the owner was parked
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: engine)
    live = _registry(tmp_path)
    first = _fresh_messages(tmp_path, live)
    assert OWN_LINE in _system_text(first), "the attempt starts without the terminal payload"

    rounds = []

    def model(call, _disposition, **_kwargs):
        ack_spy.transcript = call.messages
        rounds.append(len(call.messages))
        if waited and len(rounds) == 1:
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-wait", "type": "function",
                "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": "run-1"})}}]}, 0.0
        return {"role": "assistant", "content": "parking for the owner", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", model)
    if not waited:
        # The body's collector retains the settled output during the attempt (#2205); the loop's
        # exit then names the never-read omission.
        from ouroboros.delegate_collection import collect_terminal

        collect_terminal(tmp_path, "run-1", gateway=engine)
    loop.run_llm_loop(**_loop_kwargs(tmp_path, live, first))
    transcript = live._ctx.messages
    tool_rows = [row for row in transcript if row.get("role") == "tool" and row.get("tool_call_id") == "call-wait"]
    if waited:
        from ouroboros.delegate_supervision import supervision_checkpoint

        wake_id = supervision_checkpoint(live._ctx)["runs"]["run-1"]["last_acknowledged_wake"]["wake_id"]
        assert len(tool_rows) == 1 and '"status": "terminal"' in tool_rows[0]["content"]
        assert wake_id in tool_rows[0]["content"]
        assert _acks(tmp_path) == [("run-1", TASK, "1")]
        # Post-append: when the row landed, the tool row naming its wake was already in the transcript.
        assert [attempt for attempt, _ in ack_spy.seen] == ["1"]
        assert wake_id in ack_spy.seen[0][1] and '"tool_call_id": "call-wait"' in ack_spy.seen[0][1]
        # Every custody reader keeps it: the marker-filtered chain and the current closed-run receipt.
        from ouroboros.delegate_custody_current import snapshot

        assert [row["type"] for row in custody._iter_rows(custody.event_log_path(tmp_path))
                ].count(custody.TERMINAL_ACKNOWLEDGED) == 1
        assert snapshot(tmp_path)["state"]["run-1"].terminal_ack_attempt == "1"
    else:
        assert tool_rows == [] and _acks(tmp_path) == []

    # The owner wait parks the transcript the loop actually built; a planned restart picks it up.
    live._ctx._owner_wait_requested = "quiz"
    write_task_result(tmp_path, TASK, "running")
    wait = checkpoint_owner_wait(live._ctx, transcript, {"tool_calls": []}, {"cost": 0.0}, 1, [], set())
    set_owner_wait(tmp_path, TASK, {**wait, "state": "waiting"})
    custody._CUSTODY.clear()
    assert OWN_LINE in _system_text(transcript), "the parked transcript keeps its attempt-start system view"

    cold = _registry(tmp_path, attempt=1)  # same attempt: the restart never mints a new one
    cold._ctx.owner_wait_resume = {**wait, "restart_transaction_id": "observed-restart"}
    cold._ctx.owner_wait_callback = lambda *_: None
    rebuilt = _fresh_messages(tmp_path, cold, attempt=1)
    assert OWN_LINE not in _system_text(rebuilt)
    assert (UNREAD_LINE in _system_text(rebuilt)) is not waited

    sent = []

    def resumed(call, _disposition, **_kwargs):
        sent.append(list(call.messages))
        return {"role": "assistant", "content": "continued", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", resumed)
    result, _usage, _trace = loop.run_llm_loop(**_loop_kwargs(tmp_path, cold, rebuilt))
    assert result == "continued" and len(sent) == 1
    request = sent[0]
    restored = _dialogue(request[1:])
    assert "continued from its saved owner wait after a planned restart" in restored
    if waited:
        # The restored dialogue carries the wake; the reprojected system view does not repeat it.
        assert wake_id in restored and '"status": "terminal"' in restored
        assert OWN_LINE not in _system_text(request)
        assert UNREAD_LINE not in _system_text(request)
    else:
        assert '"status": "terminal"' not in restored
        assert OWN_LINE not in _system_text(request) and UNREAD_LINE in _system_text(request)
    assert _acks(tmp_path) == ([("run-1", TASK, "1")] if waited else [])
    # A crash retry has a fresh transcript: a consumed result's own-run line
    # returns; a never-read result keeps its D7 line instead, without duplication.
    retried = _system_text(_fresh_messages(tmp_path, _registry(tmp_path, attempt=2), attempt=2))
    assert (OWN_LINE in retried) is waited
    assert (UNREAD_LINE in retried) is not waited


# --- the unknown-provider hold ---------------------------------------------------------

def _hold_registry(tmp_path):
    from ouroboros.delegate_supervision import write_unknown_hold

    registry = _registry(tmp_path)
    ctx = registry._ctx
    ctx.exact_model_route, ctx._configured_subagent_route_kind = True, "agent_session"
    ctx.task_metadata = {"root_task_id": TASK, "configured_subagent": {"config_fingerprint": "fp"}}
    write_unknown_hold(ctx, "run-1", {"run_id": "run-1", "hold_cycles": 1, "unknown_attempt_id": "u-1"})
    return registry


def _hold(registry, tmp_path, messages):
    from ouroboros.delegate_hold import hold_step

    return hold_step(registry, controls={}, messages=messages, drive_logs=tmp_path / "logs",
                     task_id=TASK, emit_progress=lambda *_a, **_k: None)


def test_the_hold_acknowledges_only_after_its_receipt_is_in_the_transcript(tmp_path, monkeypatch, ack_spy):
    import ouroboros.delegate_hold as delegate_hold

    _settle(tmp_path, task_id=TASK)
    registry = _hold_registry(tmp_path)
    # The hold's own supervising wait, with the leaf's terminal answer as the observation.
    monkeypatch.setattr(delegate_hold, "supervised_wait", lambda ctx, run_id: _terminal_wake(ctx, run_id))
    messages = [{"role": "user", "content": "supervise"}]
    ack_spy.transcript = messages
    assert _hold(registry, tmp_path, messages) == "resume"
    assert messages[-1]["content"].startswith("[DELEGATED LEAF WAKE / UNKNOWN-HOLD RESUME]")
    assert '"status": "terminal"' in messages[-1]["content"]
    assert _acks(tmp_path) == [("run-1", TASK, "1")]
    assert len(ack_spy.seen) == 1 and "[DELEGATED LEAF WAKE / UNKNOWN-HOLD RESUME]" in ack_spy.seen[0][1]


def test_a_hold_whose_acknowledgement_never_lands_drops_its_receipt_and_records_nothing(tmp_path, monkeypatch):
    import ouroboros.delegate_hold as delegate_hold

    _settle(tmp_path, task_id=TASK)
    registry = _hold_registry(tmp_path)
    monkeypatch.setattr(delegate_hold, "supervised_wait", lambda ctx, run_id: _terminal_wake(ctx, run_id))
    monkeypatch.setattr(delegate_hold.time, "sleep", lambda _s: None)
    # The supervision acknowledgement row is rejected by the event log: the real
    # acknowledgement refuses before the terminal row, so nothing may claim an append.
    real = custody.emit
    monkeypatch.setattr(custody, "emit", lambda root, kind, payload: False
                        if kind == "delegate_supervision_wake_acknowledged" else real(root, kind, payload))
    messages = [{"role": "user", "content": "supervise"}]
    assert _hold(registry, tmp_path, messages) == "terminal"
    assert messages == [{"role": "user", "content": "supervise"}], "the unacknowledged receipt is removed"
    assert _acks(tmp_path) == []
    assert OWN_LINE in _system_text(_fresh_messages(tmp_path, _registry(tmp_path)))


# --- the configured-session startup receipt (crash-retry adoption) ---------------------

def _crash_retry(tmp_path):
    """Attempt 1 took run-1's terminal wake, then its worker died; attempt 2 adopts the settled run."""
    import ouroboros.delegate_recovery as recovery
    from ouroboros.delegate_supervision import acknowledge_pending_wake
    from ouroboros.subagent_work_order import work_order_fingerprint
    from tests.test_available_subagents_core_followup import _session_row, _settings, _snapshot

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {"id": TASK, "type": "task", "text": "supervise run-1", "_attempt": 2,
            "configured_subagent": snapshot, "drive_root": str(tmp_path),
            "task_constraint": {}, "task_contract": {}}
    _settle(tmp_path, task_id=TASK, selected_subagent_id="session-builder",
            config_fingerprint=snapshot["config_fingerprint"],
            authority_fingerprint=recovery.authority_fingerprint_from_task(task),
            work_order_fingerprint=work_order_fingerprint(task))
    first = _registry(tmp_path, attempt=1)._ctx
    assert acknowledge_pending_wake(first, _terminal_wake(first).text)
    assert _acks(tmp_path) == [("run-1", TASK, "1")]
    handoff = recovery.prepare_handoff(tmp_path, task, cause=recovery.CAUSE_WORKER_CRASH,
                                       old_attempt=1, new_attempt=2, worker_id=1, exitcode=1)
    assert handoff["settled_terminal"] is True
    assert handoff["pending_wake"]["payload"]["status"] == "terminal"
    custody._CUSTODY.clear()
    return task


def _save_delivered_startup(registry, messages):
    from ouroboros.working_checkpoint import save_ready

    limit = SimpleNamespace(tools=registry, messages=messages, llm_trace={"tool_calls": []},
                            accumulated_usage={}, owner_msg_seen=set(), round_idx=1,
                            tool_schemas=[], budget_tail="tool")
    save_ready(limit)


def test_the_startup_receipt_acknowledges_the_replayed_wake_after_incorporated_save(
        tmp_path, ack_spy):
    from ouroboros.subagent_bootstrap import _adopt_recovery_handoff, append_startup_receipt

    task = _crash_retry(tmp_path)
    successor = _registry(tmp_path, attempt=2)
    # Bootstrap stages; the restored loop appends and saves before ACK.
    startup = _adopt_recovery_handoff(successor._ctx, task)
    assert json.loads(startup)["status"] == "configured_session_recovered_wake"
    messages = _fresh_messages(tmp_path, successor, attempt=2)
    ack_spy.transcript = messages
    append_startup_receipt(successor._ctx, messages, startup)
    assert _acks(tmp_path) == [("run-1", TASK, "1")]
    _save_delivered_startup(successor, messages)
    assert messages[-1]["content"].startswith("[CONFIGURED SESSION STARTUP / WAKE RECEIPT]")
    assert _acks(tmp_path) == [("run-1", TASK, "1"), ("run-1", TASK, "2")]
    assert len(ack_spy.seen) == 1 and ack_spy.seen[0][0] == "2"
    assert "[CONFIGURED SESSION STARTUP / WAKE RECEIPT]" in ack_spy.seen[0][1]
    # The next build of attempt 2 (a same-attempt cold Resume) no longer names the run.
    assert OWN_LINE not in _system_text(_fresh_messages(tmp_path, _registry(tmp_path, attempt=2), attempt=2))


def test_the_first_request_after_a_crash_retry_dates_its_health_snapshot(tmp_path):
    from ouroboros.subagent_bootstrap import _adopt_recovery_handoff, append_startup_receipt

    task = _crash_retry(tmp_path)
    successor = _registry(tmp_path, attempt=2)
    startup = _adopt_recovery_handoff(successor._ctx, task)
    messages = _fresh_messages(tmp_path, successor, attempt=2)
    append_startup_receipt(successor._ctx, messages, startup)
    assert _acks(tmp_path) == [("run-1", TASK, "1")]
    _save_delivered_startup(successor, messages)
    receipt = messages[-1]["content"]
    assert '"status": "terminal"' in receipt and '"run_id": "run-1"' in receipt
    # Health is captured before startup appends the receipt. Its historical claim
    # must remain true without a second context build or a pre-append ACK.
    health = _system_text(messages)
    assert OWN_LINE in health
    assert "at context capture, no terminal-wake acknowledgement was recorded for this attempt" in health
    assert "its terminal payload is not in this attempt's transcript" not in health
    assert OWN_LINE not in _system_text(_fresh_messages(tmp_path, successor, attempt=2))


# Foreign obligations share one pointer; its exact list preserves every owner/run.
# A foreign reader receives no instruction to read or acknowledge those runs.

def test_a_foreign_reader_sees_the_owners_unread_runs_on_one_line_without_a_read_instruction(geo, monkeypatch):
    from ouroboros.context import build_health_invariants

    _stage(geo, _top(geo, ROOT), monkeypatch, run_id="run-new")
    assert custody.record_started(geo.canonical, custody.RunCustody(
        run_id="run-old", task_id=ROOT, route_id="r", model="m", ledger_root=str(geo.canonical)))
    assert custody.emit(geo.canonical, custody.OUTPUT_SPILLED, {
        "run_id": "run-old", "task_id": ROOT, "artifact": "delegated_runs/run-old.json",
        "staged": True, "full_content": True, "sha256": "0" * 64})
    assert custody.emit(geo.canonical, custody.SETTLED, {"run_id": "run-old", "task_id": ROOT, "state": "succeeded"})
    custody._CUSTODY.clear()
    env = SimpleNamespace(drive_root=geo.canonical, drive_path=lambda rel="": geo.canonical / rel,
                          repo_path=lambda rel="": geo.repo / rel)
    health = build_health_invariants(env, task_id="intruder", attempt_key="1")
    rows = [item for item in health.splitlines() if "OTHER TASKS' DELEGATED WORK" in item]
    assert len(rows) == 1 and "full owner/run list" in rows[0]
    assert "Only the owner acknowledges results" in rows[0]
    assert "read_file" not in rows[0] and "NEVER READ" not in health
    # Resolve the exact list through the advertised reader as this foreign task.
    import re

    from ouroboros.tools.query_data import _query_data
    from ouroboros.tools.registry import ToolContext

    pointer = re.search(r"query_data\(op='blob', sha256='([a-f0-9]{64})'\)", rows[0])
    assert pointer
    intruder = ToolContext(repo_dir=geo.repo, drive_root=geo.canonical, task_id="intruder")
    record = json.loads(_query_data(intruder, op="blob", sha256=pointer.group(1)))
    assert record["status"] == "ok", record
    assert record["data"] == {ROOT: {"unread": ["run-new", "run-old"], "patches": [], "yours": []}}
    owner = build_health_invariants(env, task_id=ROOT, attempt_key="1")
    assert all(f"NEVER READ — run {run}" in owner for run in ("run-new", "run-old"))
    assert "read_file" in owner and "OTHER TASKS' DELEGATED WORK" not in owner
