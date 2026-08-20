"""Poltergeist phase B (B1): delegation-first nanny economics.

The incident: children dispatched onto the $0 delegated substrate burned $87 of
their OWN opus rounds co-building around successful runs, because (a) the child's
goal reached the run only if the nanny hand-copied it into every prompt, and
(b) the finalization nudge went permanently silent after the first successful
delegated run (old loop.py `if evidence.get("delegated_runs_succeeded"): return ""`).

Phase B replaces both: the task contract's objective/expected_output ride
STRUCTURALLY in the host-authored run `instructions`, and the silence is
proportional to the measured metered burn since the last delegated-run activity
(config thresholds; owner decision 2=B — reminders only, never a cap).
"""

import json
from types import SimpleNamespace

import pytest

from ouroboros.task_pacing import NANNY_FIRST_REMINDER_ROUNDS, NANNY_REMINDER_ROUNDS


@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )


# -- B1.1: the contract rides the host instructions ----------------------------


def _start_with_contract(tmp_path, monkeypatch, contract):
    import ouroboros.tools.delegate as delegate
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path,
        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
    )
    ctx.task_id = "t-nanny"
    ctx.task_metadata = {"root_task_id": "t-root", "parent_task_id": "t-root"}
    ctx.task_contract = dict(contract)

    seen = {}

    class _Stub:
        engine_version = "3.3.6"

        def handshake(self, **_kw): return {}
        def agent_capabilities(self):
            return {"harnesses": [{
                "id": "some-route", "enabled": True, "status": "ok",
                "accessProfilesSupported": ["readonly", "workspace_write"],
            }]}
        def quota_snapshots(self): return []
        def find_project_id(self, root): return "prj-existing"
        def register_project(self, root): raise AssertionError("must reuse the registration")
        def start_run(self, request, *, idempotency_key=""):
            seen["request"] = request
            return {"runId": "run-1", "runDir": "/tmp/run-1"}
        def close(self): pass

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    payload = json.loads(delegate._delegate_start(ctx, "run the tests"))
    delegate._CUSTODY.clear()
    assert payload["status"] == "started", payload
    return seen["request"]


def test_the_contract_objective_rides_the_run_instructions_structurally(tmp_path, monkeypatch):
    request = _start_with_contract(tmp_path, monkeypatch, {
        "objective": "Build the ghost-core module of the poltergeist game",
        "expected_output": "a verified module with passing tests",
    })
    instructions = request["instructions"]
    assert "HOST TASK OBJECTIVE" in instructions
    assert "ghost-core module" in instructions
    assert "HOST EXPECTED OUTPUT" in instructions
    assert "verified module with passing tests" in instructions
    # The nanny did NOT have to copy the contract into the prompt.
    assert "ghost-core" not in request["prompt"]
    # The prohibitions stay the opening statement of the channel.
    assert instructions.index("git commit") < instructions.index("HOST TASK OBJECTIVE")


def test_a_missing_contract_contributes_nothing(tmp_path, monkeypatch):
    request = _start_with_contract(tmp_path, monkeypatch, {})
    assert "HOST TASK OBJECTIVE" not in request["instructions"]
    assert "HOST EXPECTED OUTPUT" not in request["instructions"]


def test_retry_replays_the_stored_wire_body_not_a_rebuilt_one():
    """Byte-identical retry stays intact: the retry path replays the STORED
    canonical body under the stored key; the contract block is derived only on
    the fresh-start path (`_start_request` is never called on recovery)."""
    import inspect

    from ouroboros.tools import delegate

    src = inspect.getsource(delegate._delegate_start)
    recovering_branch = src.split("if recovering:", 1)[1].split("else:", 1)[0]
    assert "_assignment_instructions" not in recovering_branch
    assert "_start_request" not in recovering_branch
    # C1 extracted the stored-record read into `_resolve_retry_invocation`
    # (re-exported on the delegate surface); the recovering branch must go
    # through it, and the resolver itself is what replays the STORED body.
    assert "_resolve_retry_invocation" in recovering_branch
    assert 'record["request"]' in inspect.getsource(delegate._resolve_retry_invocation)


# -- B1.2/B1.3: proportional reminder ------------------------------------------


def _nanny_ctx(**extra):
    return SimpleNamespace(_nanny_route_dispatched=True, **extra)


def _delegate_call(name="delegate_start"):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_the_baseline_advances_on_delegate_verbs_only():
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _nanny_metered_since_delegate_activity

    ctx = _nanny_ctx()
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.5}, [_delegate_call()])
    assert _nanny_metered_since_delegate_activity(ctx) == (0, 0.0)

    # Five non-delegate rounds accrue against the baseline.
    for idx, cost in ((2, 1.0), (3, 1.5), (4, 2.0), (5, 2.5), (6, 3.0)):
        _note_nanny_delegate_activity(ctx, idx, {"cost": cost}, [
            {"id": "c", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ])
    rounds, cost = _nanny_metered_since_delegate_activity(ctx)
    assert rounds == 5
    assert cost == pytest.approx(2.5)

    # A delegate_wait call re-baselines the ROUND axis only (R2-5): watching is
    # not delegating, so the dollar axis stays cumulative across waits.
    _note_nanny_delegate_activity(ctx, 7, {"cost": 3.2}, [_delegate_call("delegate_wait")])
    rounds, cost = _nanny_metered_since_delegate_activity(ctx)
    assert rounds == 0
    assert cost == pytest.approx(2.7)

    # A real delegate verb (start/answer/cancel) resets BOTH axes.
    _note_nanny_delegate_activity(ctx, 8, {"cost": 3.4}, [_delegate_call("delegate_answer")])
    assert _nanny_metered_since_delegate_activity(ctx) == (0, 0.0)


def test_a_non_nanny_task_is_never_tracked_or_reminded():
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = SimpleNamespace(_nanny_route_dispatched=False)
    _note_nanny_delegate_activity(ctx, 5, {"cost": 9.0}, [])
    assert not hasattr(ctx, "_nanny_metered_progress")
    msgs: list = []
    tools = SimpleNamespace(_ctx=ctx)
    assert _maybe_inject_nanny_economics_reminder(5, msgs, tools, lambda *_: None) is False
    assert msgs == []


def test_the_reminder_fires_proportionally_and_rearms_without_a_cap():
    """Owner 2=B: no absolute cap — the reminder repeats each threshold-width of
    metered rounds for as long as the burn continues, and never blocks."""
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    fired: list = []
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.0}, [_delegate_call()])

    round_idx = 1
    for _ in range(3 * NANNY_REMINDER_ROUNDS):
        round_idx += 1
        _note_nanny_delegate_activity(ctx, round_idx, {"cost": 0.0}, [
            {"id": "c", "type": "function", "function": {"name": "run_command", "arguments": "{}"}},
        ])
        msgs: list = []
        if _maybe_inject_nanny_economics_reminder(round_idx, msgs, tools, lambda *_: None):
            fired.append(round_idx)
            joined = "\n".join(m.get("content", "") for m in msgs)
            assert "NANNY ECONOMICS REMINDER" in joined
            assert "reminder, not a stop" in joined
    # Fired repeatedly (proportional re-arming), roughly once per threshold width.
    assert len(fired) == 3
    assert fired[1] - fired[0] >= NANNY_REMINDER_ROUNDS


def test_the_reminder_never_makes_an_unconditional_zero_cost_claim():
    """BR1-3 regression pin (owner wording law: typed cost classes, never
    "free" unqualified): the reminder fires PRE-delegation, when the run's
    spend may settle billed/estimated/undisclosed — its text must carry the
    conditional phrasing (known-zero only on a settled $0 spend), never the
    old unconditional "runs at $0 marginal cost" assertion."""
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.0}, [_delegate_call()])
    round_idx = 1
    msgs: list = []
    while not msgs:
        round_idx += 1
        _note_nanny_delegate_activity(ctx, round_idx, {"cost": 0.0}, [
            {"id": "c", "type": "function", "function": {"name": "run_command", "arguments": "{}"}},
        ])
        _maybe_inject_nanny_economics_reminder(round_idx, msgs, tools, lambda *_: None)
    text = "\n".join(m.get("content", "") for m in msgs)
    assert "NANNY ECONOMICS REMINDER" in text
    # The conditional, typed form is present...
    assert "only when its settled spend reports $0" in text
    assert "estimated or undisclosed spend is never zero" in text
    # ...and the unconditional claim is gone: every "$0" in the reminder sits
    # inside the settled-spend conditional, never as a standing property.
    assert "runs at $0" not in text
    assert "substrate runs at" not in text


def test_below_threshold_stays_silent():
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.0}, [_delegate_call()])
    _note_nanny_delegate_activity(ctx, 2, {"cost": 0.1}, [])
    assert _maybe_inject_nanny_economics_reminder(2, [], tools, lambda *_: None) is False


def test_the_cost_axis_alone_can_trigger_the_reminder():
    """REAL-PATH cost firing (F1): a $2+ burn at round 2 must fire through the
    production gate — no cursor presets. The old single round-spacing gate
    (`round_idx - last_fired >= 8`) muted exactly this case, and the old test
    masked the mute by presetting `_nanny_reminder_round` below zero."""
    from ouroboros.task_pacing import NANNY_REMINDER_USD
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.0}, [_delegate_call()])
    # One expensive round: below the round threshold, at the dollar one.
    _note_nanny_delegate_activity(ctx, 2, {"cost": NANNY_REMINDER_USD}, [])
    msgs: list = []
    assert _maybe_inject_nanny_economics_reminder(2, msgs, tools, lambda *_: None) is True
    assert any("metered LLM rounds" in m.get("content", "") for m in msgs)


def test_the_reminder_stays_out_of_owner_chat_progress(tmp_path):
    """Owner decision (2026-08-15): the economics reminder is a model-facing
    user message plus a typed task_checkpoint event — emit_progress (chat ⚠️
    lines) stays silent, so the owner chat is not spammed mid-run."""
    import json

    from ouroboros.task_pacing import NANNY_REMINDER_USD
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.0}, [_delegate_call()])
    _note_nanny_delegate_activity(ctx, 2, {"cost": NANNY_REMINDER_USD}, [])
    msgs: list = []
    progress: list = []
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()
    assert _maybe_inject_nanny_economics_reminder(
        2, msgs, tools, progress.append,
        event_queue=None, task_id="t", drive_logs=drive_logs,
    ) is True
    assert progress == []
    assert any("NANNY ECONOMICS REMINDER" in m.get("content", "") for m in msgs)
    events = [json.loads(line) for line in
              (drive_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e.get("type") == "task_checkpoint"
               and e.get("checkpoint_kind") == "nanny_economics_reminder"
               for e in events)


def test_the_rearm_is_dual_axis_a_continuing_dollar_burn_refires_before_8_rounds():
    """F1 (five reviewers converged): after a firing, EITHER a further
    threshold-width of rounds OR a further threshold-width of dollars re-arms.
    A nanny burning $2 per round must hear the reminder again long before eight
    more rounds pass."""
    from ouroboros.task_pacing import NANNY_REMINDER_USD
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.0}, [_delegate_call()])
    _note_nanny_delegate_activity(ctx, 2, {"cost": NANNY_REMINDER_USD}, [])
    assert _maybe_inject_nanny_economics_reminder(2, [], tools, lambda *_: None) is True
    # One more round, another threshold-width of dollars: due again on the COST
    # axis alone (rounds-since-fire is 1, far under the round threshold).
    _note_nanny_delegate_activity(ctx, 3, {"cost": 2 * NANNY_REMINDER_USD}, [])
    assert _maybe_inject_nanny_economics_reminder(3, [], tools, lambda *_: None) is True
    # ...and with no further burn since that second firing, nothing re-fires.
    _note_nanny_delegate_activity(ctx, 4, {"cost": 2 * NANNY_REMINDER_USD}, [])
    assert _maybe_inject_nanny_economics_reminder(4, [], tools, lambda *_: None) is False


def test_delegate_activity_resets_the_fire_cursor():
    """F1 (gemini): a cooldown earned BEFORE delegate activity must not mute the
    reminder for burn that happens AFTER it — the fire cursor is cleared on the
    delegate verb, so the first post-activity threshold crossing fires with no
    spacing gate."""
    from ouroboros.task_pacing import NANNY_REMINDER_USD
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.0}, [_delegate_call()])
    _note_nanny_delegate_activity(ctx, 2, {"cost": NANNY_REMINDER_USD}, [])
    assert _maybe_inject_nanny_economics_reminder(2, [], tools, lambda *_: None) is True
    assert isinstance(ctx._nanny_reminder_mark, dict)
    # Delegate activity: baseline re-zeroes AND the fire cursor clears.
    _note_nanny_delegate_activity(ctx, 3, {"cost": NANNY_REMINDER_USD}, [_delegate_call("delegate_wait")])
    assert ctx._nanny_reminder_mark is None
    # Immediately after, one more expensive round crosses the cost threshold
    # again — and fires as a FIRST firing (no spacing gate), round 4 < 8.
    _note_nanny_delegate_activity(ctx, 4, {"cost": 2 * NANNY_REMINDER_USD}, [])
    assert _maybe_inject_nanny_economics_reminder(4, [], tools, lambda *_: None) is True


def test_a_ritual_wait_cannot_rebaseline_the_cost_axis():
    """R2-5 (fullcontext HIGH — the one genuine regression of the batch): the
    reviewer's probe shape, $0.24/round of co-building with a delegate_wait
    every 7 rounds, evaded the reminder forever because a wait re-zeroed BOTH
    baselines. A wait now advances only the ROUND baseline, so the dollar axis
    keeps accruing and fires at the threshold."""
    from ouroboros.task_pacing import NANNY_REMINDER_USD
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 0, {"cost": 0.0}, [_delegate_call()])
    fired: list = []
    cost = 0.0
    for round_idx in range(1, 25):
        cost += 0.24
        calls = [_delegate_call("delegate_wait")] if round_idx % 7 == 0 else []
        _note_nanny_delegate_activity(ctx, round_idx, {"cost": cost}, calls)
        msgs: list = []
        if _maybe_inject_nanny_economics_reminder(round_idx, msgs, tools, lambda *_: None):
            fired.append(round_idx)
            joined = "\n".join(m.get("content", "") for m in msgs)
            assert "since your last delegated-run activity" in joined
    assert fired, "the ritual wait must not silence the dollar axis"
    # $2 threshold crossed at round 9 ($2.16) despite the wait at round 7.
    assert fired[0] == 9
    assert NANNY_REMINDER_USD == pytest.approx(2.0)


def test_a_genuinely_holding_nanny_stays_quiet():
    """R2-5, the other direction: waits only, pennies per round — neither axis
    crosses, and the reminder never fires."""
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 0, {"cost": 0.0}, [_delegate_call()])
    cost = 0.0
    for round_idx in range(1, 31):
        cost += 0.01
        calls = [_delegate_call("delegate_wait")] if round_idx % 3 == 0 else []
        _note_nanny_delegate_activity(ctx, round_idx, {"cost": cost}, calls)
        assert _maybe_inject_nanny_economics_reminder(
            round_idx, [], tools, lambda *_: None) is False


def test_zero_baseline_reminder_says_since_task_start():
    """R2-7c: before the first delegate verb there is no 'last delegated-run
    activity' — the reminder measures from the task's start and says so."""
    from ouroboros.task_pacing import NANNY_REMINDER_USD
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    # No delegate verb ever called: the baseline is unset.
    _note_nanny_delegate_activity(ctx, 1, {"cost": NANNY_REMINDER_USD}, [])
    msgs: list = []
    assert _maybe_inject_nanny_economics_reminder(1, msgs, tools, lambda *_: None) is True
    joined = "\n".join(m.get("content", "") for m in msgs)
    assert "since this task started" in joined
    assert "since your last delegated-run activity" not in joined
    # The switch_model sanction rides the reminder text (R2-7b).
    assert "switch_model" in joined


def test_first_reminder_fires_early_with_zero_delegate_activity():
    """Owner-approved (2026-08-15): a harness-dispatched nanny that has made NO
    delegate-verb call hears its FIRST reminder at NANNY_FIRST_REMINDER_ROUNDS
    regardless of dollars — the live E2E's cheap children finished in 4-8
    rounds under $0.15 and never heard it."""
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    fired_at = None
    for round_idx in range(1, NANNY_REMINDER_ROUNDS + 1):
        # Pennies per round, never a delegate verb: the dollar axis stays cold.
        _note_nanny_delegate_activity(ctx, round_idx, {"cost": round_idx * 0.01}, [])
        msgs: list = []
        if _maybe_inject_nanny_economics_reminder(round_idx, msgs, tools, lambda *_: None):
            fired_at = round_idx
            joined = "\n".join(m.get("content", "") for m in msgs)
            assert "NANNY ECONOMICS REMINDER" in joined
            assert "since this task started" in joined
            break
    assert fired_at == NANNY_FIRST_REMINDER_ROUNDS


def test_no_early_fire_once_a_delegate_verb_happened():
    """The early first-fire is ONLY for a nanny with zero delegate activity:
    after any delegate verb the ordinary 8-round/$2 dual-axis thresholds apply
    unchanged, so round NANNY_FIRST_REMINDER_ROUNDS stays silent."""
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.0}, [_delegate_call()])
    fired: list = []
    for round_idx in range(2, 2 + NANNY_REMINDER_ROUNDS + 1):
        _note_nanny_delegate_activity(ctx, round_idx, {"cost": 0.05}, [])
        if _maybe_inject_nanny_economics_reminder(round_idx, [], tools, lambda *_: None):
            fired.append(round_idx)
    # Silent through the early horizon; the first fire needs the full
    # threshold-width of metered rounds since the delegate activity.
    assert fired == [1 + NANNY_REMINDER_ROUNDS]


def test_rearm_after_the_early_first_fire_uses_the_ordinary_thresholds():
    """Re-arms after the first firing are unchanged: after the early fire at
    round NANNY_FIRST_REMINDER_ROUNDS, the next one waits a further FULL
    threshold-width (8 rounds / $2) — the early constant governs only the
    first firing of a zero-delegation nanny."""
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder

    ctx = _nanny_ctx()
    tools = SimpleNamespace(_ctx=ctx)
    fired: list = []
    for round_idx in range(1, 3 * NANNY_REMINDER_ROUNDS):
        _note_nanny_delegate_activity(ctx, round_idx, {"cost": 0.0}, [])
        if _maybe_inject_nanny_economics_reminder(round_idx, [], tools, lambda *_: None):
            fired.append(round_idx)
    assert fired[0] == NANNY_FIRST_REMINDER_ROUNDS
    assert fired[1] - fired[0] == NANNY_REMINDER_ROUNDS
    assert fired[2] - fired[1] == NANNY_REMINDER_ROUNDS


def test_an_expensive_no_tool_round_still_counts(tmp_path):
    """F12 (sol #11): the loop marks metered progress after EVERY LLM response,
    including tool-less rounds — the same call the loop makes before its
    final-answer branch, with an empty tool list, must advance the progress
    mark while never advancing the delegate baseline."""
    from ouroboros.loop_nudges import _note_nanny_delegate_activity
    from ouroboros.loop_nudges import _nanny_metered_since_delegate_activity

    ctx = _nanny_ctx()
    _note_nanny_delegate_activity(ctx, 1, {"cost": 0.5}, [_delegate_call()])
    # Two no-tool rounds (tool_calls=[]), each expensive.
    _note_nanny_delegate_activity(ctx, 2, {"cost": 1.5}, [])
    _note_nanny_delegate_activity(ctx, 3, {"cost": 2.5}, [])
    rounds, cost = _nanny_metered_since_delegate_activity(ctx)
    assert rounds == 2
    assert cost == pytest.approx(2.0)


def test_a_fresh_dispatch_resets_every_economics_mark():
    """F4 (regressions lens): stale marks from a previous task on a reused
    context must not leak into the next dispatch — the reset clears the
    progress mark, the baseline AND the reminder fire cursor, so the new task
    starts with zero measured burn and an un-armed reminder."""
    from ouroboros.agent import reset_nanny_economics_marks
    from ouroboros.loop_nudges import _maybe_inject_nanny_economics_reminder, _nanny_metered_since_delegate_activity

    ctx = _nanny_ctx(
        _nanny_metered_progress={"round": 30, "cost": 9.0},
        _nanny_delegate_baseline={"round": 5, "cost": 1.0},
        _nanny_reminder_mark={"round": 30, "cost": 9.0},
        _nanny_finalization_injected=True,
    )
    reset_nanny_economics_marks(ctx, route_dispatched=True)
    assert ctx._nanny_route_dispatched is True
    assert ctx._nanny_finalization_injected is False
    assert _nanny_metered_since_delegate_activity(ctx) == (0, 0.0)
    tools = SimpleNamespace(_ctx=ctx)
    assert _maybe_inject_nanny_economics_reminder(31, [], tools, lambda *_: None) is False
    # And a non-harness dispatch resets the route latch off.
    reset_nanny_economics_marks(ctx, route_dispatched=False)
    assert ctx._nanny_route_dispatched is False


# -- finalization: proportional silence, not permanent silence ------------------


def _custody_drive(tmp_path):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _succeeded_run(drive, task_id="child-1"):
    from ouroboros import delegate_custody as custody

    assert custody.emit(drive, custody.STARTED, {
        "run_id": "run-1", "task_id": task_id, "route": "claude", "max_seconds": 300,
    })
    assert custody.emit(drive, custody.SETTLED, {
        "run_id": "run-1", "task_id": task_id, "route": "claude",
        "model": "claude-opus-5", "state": "succeeded", "cost_usd": 0.0,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
    })


def test_succeeded_run_with_a_big_metered_tail_gets_the_overrun_reminder(tmp_path):
    from ouroboros.loop_nudges import _maybe_inject_finalization_nudges

    drive = _custody_drive(tmp_path)
    _succeeded_run(drive)
    ctx = _nanny_ctx(
        _nanny_finalization_injected=False,
        _nanny_metered_progress={"round": 30, "cost": 9.0},
        _nanny_delegate_baseline={"round": 5, "cost": 1.0},
    )
    tools = SimpleNamespace(_ctx=ctx, available_tools=lambda: ["delegate_start", "delegate_wait"])
    msgs: list = []
    assert _maybe_inject_finalization_nudges(
        tools, drive, "child-1",
        {"reasoning_notes": [], "tool_calls": []}, "done", msgs, lambda *_: None,
    ) is True
    joined = "\n".join(m.get("content", "") for m in msgs)
    assert "NANNY_METERED_OVERRUN" in joined
    assert "25 of your own metered LLM rounds" in joined
    assert "$8.00" in joined
    assert "NANNY_DID_NOT_DELEGATE" not in joined


def test_succeeded_run_with_a_modest_tail_keeps_the_silence(tmp_path):
    from ouroboros.loop_nudges import _maybe_inject_finalization_nudges

    drive = _custody_drive(tmp_path)
    _succeeded_run(drive)
    ctx = _nanny_ctx(
        _nanny_finalization_injected=False,
        _nanny_metered_progress={"round": 8, "cost": 1.2},
        _nanny_delegate_baseline={"round": 6, "cost": 1.0},
    )
    tools = SimpleNamespace(_ctx=ctx, available_tools=lambda: ["delegate_start", "delegate_wait"])
    msgs: list = []
    assert _maybe_inject_finalization_nudges(
        tools, drive, "child-1",
        {"reasoning_notes": [], "tool_calls": []}, "done", msgs, lambda *_: None,
    ) is False
    assert msgs == []


# -- forced finalization: the proportional silence does not extend to forced exits


def _forced_ctx(tmp_path, **marks):
    return _nanny_ctx(drive_root=tmp_path, task_id="child-1", **marks)


def test_forced_wrapup_over_a_succeeded_run_carries_the_honest_spend_note(tmp_path):
    """F16 (grok): a forced exit (budget/rounds overrun) may not re-loop, so the
    honest-spend line rides the ONE forced prompt when the overrun condition
    holds — succeeded delegated runs used to silence the forced note entirely."""
    from ouroboros.loop_nudges import _forced_delegation_note

    drive = _custody_drive(tmp_path)
    _succeeded_run(drive)
    ctx = _forced_ctx(
        tmp_path,
        _nanny_metered_progress={"round": 30, "cost": 9.0},
        _nanny_delegate_baseline={"round": 5, "cost": 1.0},
    )
    note = _forced_delegation_note(ctx, {"tool_calls": []})
    assert "succeeded" in note
    assert "25 of your own metered LLM rounds" in note
    assert "$8.00" in note
    assert "honestly" in note


def test_forced_wrapup_over_a_succeeded_run_stays_silent_below_threshold(tmp_path):
    from ouroboros.loop_nudges import _forced_delegation_note

    drive = _custody_drive(tmp_path)
    _succeeded_run(drive)
    ctx = _forced_ctx(
        tmp_path,
        _nanny_metered_progress={"round": 8, "cost": 1.2},
        _nanny_delegate_baseline={"round": 6, "cost": 1.0},
    )
    assert _forced_delegation_note(ctx, {"tool_calls": []}) == ""


# -- B1.1 addendum: the assignment-field truncator is STRICT (F11) --------------


def test_assignment_field_truncation_is_strict_with_the_marker_inside_the_budget():
    """F11 (sol #9, probe 4050→4050): the generic preview helper's anti-waste
    floor let a small overflow pass whole and appended its marker BEYOND the
    limit. The assignment field is a bounded prompt-channel field: at 4000 it
    passes untouched, at 4001 it is cut WITH the marker inside 4000, and a
    multibyte tail is never severed mid-codepoint."""
    from ouroboros.utils import truncate_within_limit
    from ouroboros.tools.delegate import _ASSIGNMENT_FIELD_CHARS

    limit = _ASSIGNMENT_FIELD_CHARS
    assert limit == 4000

    exact = "a" * limit
    assert truncate_within_limit(exact, limit) == exact

    over_by_one = "a" * (limit + 1)
    out = truncate_within_limit(over_by_one, limit)
    assert len(out) <= limit
    assert "OMISSION NOTE" in out
    assert f"original length {limit + 1}" in out

    probe = "a" * 4050
    out = truncate_within_limit(probe, limit)
    assert len(out) <= limit, "the 4050→4050 passthrough is the exact probe defect"
    assert "OMISSION NOTE" in out

    unicode_text = "яё𐍈🚀" * 2000  # multibyte + astral-plane codepoints
    out = truncate_within_limit(unicode_text, limit)
    assert len(out) <= limit
    assert "OMISSION NOTE" in out
    out.encode("utf-8")  # a mid-codepoint cut would be impossible by slicing, prove it


def test_the_contract_block_rides_bounded_through_the_instructions(tmp_path, monkeypatch):
    """End to end: an oversized objective lands in the run instructions AT the
    strict bound, marker inside, never 4050 chars of field."""
    request = _start_with_contract(tmp_path, monkeypatch, {
        "objective": "O" * 4050,
        "expected_output": "ok",
    })
    instructions = request["instructions"]
    start = instructions.index("HOST TASK OBJECTIVE")
    end = instructions.index("HOST EXPECTED OUTPUT")
    field = instructions[start:end]
    # The whole labelled block stays within label + strict field budget (+ slack
    # for the label sentence and separators).
    assert "OMISSION NOTE" in field
    assert "O" * 4001 not in field
