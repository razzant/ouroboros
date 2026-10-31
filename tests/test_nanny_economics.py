"""Poltergeist phase B (B1): delegation-first nanny economics.

The incident: children dispatched onto the $0 delegated substrate burned $87 of
their OWN opus rounds co-building around successful runs, because (a) the child's
goal reached the run only if the nanny hand-copied it into every prompt, and
(b) the finalization nudge went permanently silent after the first successful
delegated run (old loop.py `if evidence.get("delegated_runs_succeeded"): return ""`).

Phase B replaced both: the task contract's objective/expected_output ride
STRUCTURALLY in the host-authored run `instructions`, and the burn since the last
delegated-run activity is measured (owner decision 2=B — never a cap). Since #1748
(owner 11 Oct: facts, no advice) that burn is a fact on every round's facts line with
its anchors named; the periodic reminder and the metered-overrun one-shot are gone.
"""

import json
from types import SimpleNamespace

import pytest



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


def _start_with_contract(
    tmp_path, monkeypatch, contract, *, prompt="run the tests", compiled_work_order=False,
):
    import ouroboros.tools.delegate as delegate
    from ouroboros import subagent_runtime
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
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, "t-root", "running", root_task_id="t-root")  # the child's real root
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
    snapshot = {
        "schema": 1,
        "selected_subagent_id": "economics-fixture",
        "config_fingerprint": "economics-fixture-v1",
        "route": {
            "kind": "agent_session",
            "target_id": "some-route=weak-model",
            "credential_profile_id": "",
        },
        "effort": "low",
        # A READ-ONLY child (local_readonly_subagent): technical access alone decides
        # the run shape now (#1609), so the selected session states it explicitly.
        "access": "readonly",
    }
    payload = json.loads(subagent_runtime.exact_start(ctx, prompt, {
        "snapshot": snapshot,
        "compiled_work_order": compiled_work_order,
    }).text)
    delegate._CUSTODY.clear()
    assert payload["status"] == "started", payload
    return seen["request"]


def test_the_contract_objective_rides_the_run_instructions_structurally(tmp_path, monkeypatch):
    request = _start_with_contract(tmp_path, monkeypatch, {
        "objective": "Build the ghost-core module of the poltergeist game",
        "expected_output": "a verified module with passing tests",
    })
    instructions = request["instructions"]
    assert "HOST TASK CONTRACT AUTHORITY" in instructions
    assert "ghost-core module" in instructions
    assert instructions.count("verified module with passing tests") == 1
    assert "verified module with passing tests" in instructions
    # The nanny did NOT have to copy the contract into the prompt.
    assert "ghost-core" not in request["prompt"]
    # The prohibitions stay the opening statement of the channel.
    assert instructions.index("git commit") < instructions.index("HOST TASK CONTRACT AUTHORITY")


def test_direct_start_carries_assignment_and_a_predecessor_brief(tmp_path, monkeypatch):
    context = " \nAPI_CONTEXT_EXACT\n "
    contract = {
        "objective": "delegate the Cat build",
        "context": context,
        "constraints": "Claudexor only; no native edits",
        "acceptance_claims": [{"id": "nested", "claim": "L1 asks L2 to spawn L3"}],
        "delegation_budget": {"may_delegate": True, "depth_remaining": 3,
                              "intent_note": "L1\nL2\nL3"},
        "allowed_resources": {"network": False},
        "predecessor_authority": {
            "source": {"kind": "task_result", "task_id": "cat-old"},
            "result": "Keep the existing tower",
            "origin_message_text": "Claudexor only; build three levels",
            "verification_receipts": [{"evidence": "OLD_RECEIPT_BODY"}],
            "task_contract": {
                "objective": "CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3",
                "context": "never use native/API fallback",
            },
        },
    }

    from ouroboros.task_results import write_task_result
    previous = contract["predecessor_authority"]
    original = json.loads(json.dumps(contract))
    write_task_result(tmp_path, "cat-old", "completed", result=previous["result"],
                      task_contract=previous["task_contract"])
    request = _start_with_contract(tmp_path, monkeypatch, contract)
    assert "OLD_RECEIPT_BODY" not in request["instructions"]
    marker = "HOST TASK CONTRACT AUTHORITY (normalized JSON; predecessor is a brief):\n"
    payload = request["instructions"].split(marker, 1)[1]
    normalized, end = json.JSONDecoder().raw_decode(payload)
    # The owner's words that caused the work follow the contract.
    assert payload[end:].startswith("\n\n") and "words of my human" in payload[end:]

    assert normalized["context"] == context
    assert normalized["constraints"] == contract["constraints"]
    assert normalized["acceptance_claims"][0]["claim"] == "L1 asks L2 to spawn L3"
    assert normalized["delegation_budget"]["intent_note"] == "L1\nL2\nL3"
    assert normalized["allowed_resources"]["network"] is False
    brief = normalized["predecessor_authority"]
    for key in ("source", "origin_message_text"):
        assert brief[key] == previous[key]
    for key in ("result", "task_contract", "verification_receipts"):
        assert key not in brief and brief["omitted_fields"][key] > 0
    assert contract == original  # the caller and its full predecessor data are untouched
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext
    saved = json.loads(_get_task_result(ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path),
        brief["source"]["task_id"], include_authority=True))["authority"]
    assert saved["result"] == previous["result"]
    assert saved["task_contract"]["objective"] == previous["task_contract"]["objective"]
    assert saved["task_contract"]["context"] == previous["task_contract"]["context"]
    assert "get_task_result" in payload[end:] and "launching task" in payload[end:]
    assert "delegate_answer" in payload[end:] and "input_required" in payload[end:]


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


# -- B1.2/B1.3: the measured burn and its anchors -------------------------------


def _nanny_ctx(**extra):
    return SimpleNamespace(_nanny_route_dispatched=True, **extra)


def _delegate_call(name="delegate_start"):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_the_baseline_advances_on_delegate_verbs_only():
    from ouroboros.loop import (
        _nanny_metered_since_delegate_activity,
        _note_nanny_delegate_activity,
    )

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

    # Supervision verbs (answer/cancel) share the wait semantics under the
    # sprint plan's D6 ("only delegate_start/schedule_subagent reset"; the
    # answer/cancel treatment is the operator's disclosed reading of that
    # plan): the round axis re-baselines, dollars stay cumulative —
    # supervising a run is not a new act of delegation.
    _note_nanny_delegate_activity(ctx, 8, {"cost": 3.4}, [_delegate_call("delegate_answer")])
    rounds, cost = _nanny_metered_since_delegate_activity(ctx)
    assert rounds == 0
    assert cost == pytest.approx(2.9)

    # Only a genuine act of delegation (delegate_start / schedule_subagent)
    # resets BOTH axes.
    _note_nanny_delegate_activity(ctx, 9, {"cost": 3.6}, [_delegate_call()])
    assert _nanny_metered_since_delegate_activity(ctx) == (0, 0.0)


def test_scheduling_a_child_is_a_real_act_of_delegation():
    # schedule_subagent is a genuine act of delegation (charter D6): it fully
    # resets the burn baseline. Coordination verbs are untracked entirely —
    # no observation flag exists to buy phrasing or silence.
    from ouroboros.loop import (
        _nanny_metered_since_delegate_activity,
        _note_nanny_delegate_activity,
    )

    ctx = _nanny_ctx()
    _note_nanny_delegate_activity(ctx, 3, {"cost": 0.4}, [
        _delegate_call("schedule_subagent"),
    ])
    assert not hasattr(ctx, "_nanny_coordination_activity")
    assert _nanny_metered_since_delegate_activity(ctx) == (0, 0.0)


def test_a_non_nanny_task_is_never_tracked_and_has_no_nanny_fact():
    from ouroboros.loop import _note_nanny_delegate_activity
    from ouroboros.nanny_pacing import nanny_burn_fact

    ctx = SimpleNamespace(_nanny_route_dispatched=False)
    _note_nanny_delegate_activity(ctx, 5, {"cost": 9.0}, [])
    assert not hasattr(ctx, "_nanny_metered_progress")
    assert nanny_burn_fact(ctx) == ""


def test_the_periodic_reminder_and_its_thresholds_are_gone():
    """#1748 (owner 11 Oct; 12 Aug "2=B" keeps no cap): no reminder producer, no
    8-round/$2/3-round thresholds, no fire cursor."""
    from ouroboros import loop, loop_nudges, nanny_pacing, task_pacing

    for module, name in ((loop, "_maybe_inject_nanny_economics_reminder"),
                         (loop_nudges, "_maybe_inject_nanny_economics_reminder"),
                         (nanny_pacing, "nanny_reminder_due"), (nanny_pacing, "nanny_burn_phrase"),
                         (task_pacing, "NANNY_REMINDER_ROUNDS"), (task_pacing, "NANNY_REMINDER_USD"),
                         (task_pacing, "NANNY_FIRST_REMINDER_ROUNDS")):
        assert not hasattr(module, name), name


def test_the_fact_names_both_anchors_as_requests():
    """Owner R2 correction: the two counters keep their different anchors and are
    named as what they are — the round anchor moves on any delegate verb CALL, the
    dollar anchor only on a delegate_start/schedule_subagent CALL; success is not implied."""
    from ouroboros.loop import _note_nanny_delegate_activity
    from ouroboros.nanny_pacing import nanny_burn_fact

    ctx = _nanny_ctx()
    _note_nanny_delegate_activity(ctx, 3, {"cost": 1.0}, [_delegate_call()])
    _note_nanny_delegate_activity(ctx, 6, {"cost": 2.0}, [_delegate_call("delegate_wait")])
    _note_nanny_delegate_activity(ctx, 9, {"cost": 3.5}, [])
    fact = nanny_burn_fact(ctx)
    assert fact == ("nanny: 3 own rounds since the last delegate verb call (round 6); own recorded cost "
                    "$2.50 since the last delegate_start/schedule_subagent call (round 3); these anchors are "
                    "requests, not confirmed runs")
    for advice in ("Consider", "delegate the remaining", "honestly", "reminder", "should"):
        assert advice not in fact, advice


def test_before_any_delegation_the_fact_counts_from_task_start():
    from ouroboros.loop import _note_nanny_delegate_activity
    from ouroboros.nanny_pacing import nanny_burn_fact

    ctx = _nanny_ctx()
    for round_idx in (1, 2, 3, 4):
        _note_nanny_delegate_activity(ctx, round_idx, {"cost": 0.25 * round_idx}, [])
    assert nanny_burn_fact(ctx) == ("nanny: no delegate_start/schedule_subagent call yet; since task start "
                                    "4 own rounds, own recorded cost $1.00")


def test_a_wait_without_a_start_keeps_the_dollar_anchor_at_task_start():
    from ouroboros.loop import _note_nanny_delegate_activity
    from ouroboros.nanny_pacing import nanny_burn_fact

    ctx = _nanny_ctx()
    _note_nanny_delegate_activity(ctx, 2, {"cost": 0.5}, [_delegate_call("delegate_wait")])
    _note_nanny_delegate_activity(ctx, 4, {"cost": 1.5}, [])
    fact = nanny_burn_fact(ctx)
    assert "2 own rounds since the last delegate verb call (round 2)" in fact
    assert "own recorded cost $1.50 since task start (no delegate_start/schedule_subagent call yet)" in fact


def test_a_host_started_run_seed_names_its_anchor():
    from ouroboros.agent import reset_nanny_economics_marks
    from ouroboros.loop import _note_nanny_delegate_activity
    from ouroboros.nanny_pacing import nanny_burn_fact

    ctx = SimpleNamespace()
    reset_nanny_economics_marks(ctx, route_dispatched=True, delegate_activity_seed=True)
    _note_nanny_delegate_activity(ctx, 2, {"cost": 0.4}, [])
    fact = nanny_burn_fact(ctx)
    assert "2 own rounds since the last delegate verb call (host-started run before round 1)" in fact
    # the seed anchors rounds only; its dollars are not an invented anchor value
    assert "own recorded cost $0.40 since task start (the host-started run records no cost anchor)" in fact


def test_missing_money_is_unknown_partial_is_a_subtotal_and_only_a_priced_zero_is_zero():
    """A missing price is never a $0 cost (``llm_pricing.add_usage``: an unpriced call adds
    no ``cost`` and sets ``cost_final`` False). Four cases, each from the usage carrier:
    nothing priced (cost absent, with or without cost_final) is unknown; priced plus
    unpriced is a subtotal that is not final; all priced is known — a confirmed
    $0.00 included. The progress mark keeps its ``{round, cost}`` shape."""
    from ouroboros.loop import _nanny_metered_since_delegate_activity, _note_nanny_delegate_activity
    from ouroboros.nanny_pacing import nanny_burn_fact

    def run(steps):
        usage: dict = {}
        ctx = _nanny_ctx(_accumulated_usage=usage)
        for round_idx, update, calls in steps:
            usage.update(update)
            _note_nanny_delegate_activity(ctx, round_idx, usage, calls)
        return ctx, nanny_burn_fact(ctx)

    # nothing priced, cost_final missing (no accumulation recorded at all)
    ctx, fact = run([(1, {}, [_delegate_call()]), (3, {}, [])])
    assert "own recorded cost unknown (no provider price recorded)" in fact and "$0.00" not in fact
    assert ctx._nanny_metered_progress == {"round": 3, "cost": None}
    assert _nanny_metered_since_delegate_activity(ctx) == (2, None)
    # nothing priced, cost_final False (unpriced calls recorded)
    _ctx, fact = run([(1, {}, [_delegate_call()]), (3, {"cost_final": False}, [])])
    assert "own recorded cost unknown (no provider price recorded)" in fact and "$0.00" not in fact
    # partly priced: a subtotal, not final
    _ctx, fact = run([(1, {"cost": 0.25}, [_delegate_call()]), (3, {"cost": 1.0, "cost_final": False}, [])])
    assert "own recorded cost $0.75 subtotal, not final (estimates included; unpriced calls not counted)" in fact
    # priced before the anchor? none: an anchor without a recorded price subtracts nothing
    _ctx, fact = run([(1, {}, [_delegate_call()]), (2, {"cost": 0.4}, [])])
    assert "own recorded cost $0.40 since the last delegate_start" in fact
    # a confirmed zero (e.g. a local route: cost 0.0, final) is a known $0.00
    _ctx, fact = run([(1, {"cost": 0.0}, [_delegate_call()]), (2, {"cost": 0.0, "cost_final": True}, [])])
    assert "own recorded cost $0.00 since" in fact and "unknown" not in fact and "subtotal" not in fact


def test_the_fact_reads_the_real_usage_fold_estimates_counted_unpriced_not():
    """The carrier is ``llm_pricing.add_usage``'s fold, not a handcrafted total: an
    estimated call's cost IS summed (and makes the total not final), an unpriced call adds
    nothing and makes it not final, only provider-priced calls stay final, and a task with
    nothing costed stays unknown."""
    from ouroboros.llm_pricing import add_usage
    from ouroboros.loop import _note_nanny_delegate_activity
    from ouroboros.nanny_pacing import nanny_burn_fact

    def run(calls):
        usage: dict = {}
        ctx = _nanny_ctx(_accumulated_usage=usage)
        for round_idx, call, tools in calls:
            add_usage(usage, {"prompt_tokens": 10, "completion_tokens": 5, **call})
            _note_nanny_delegate_activity(ctx, round_idx, usage, tools)
        return usage, nanny_burn_fact(ctx)

    priced, estimated, unpriced = {"cost": 0.25}, {"cost": 0.5, "cost_estimated": True}, {}
    usage, fact = run([(1, priced, [_delegate_call()]), (2, estimated, []), (3, unpriced, [])])
    assert usage["cost"] == pytest.approx(0.75) and usage["cost_final"] is False
    assert ("own recorded cost $0.50 subtotal, not final (estimates included; unpriced calls not counted) "
            "since the last delegate_start/schedule_subagent call (round 1)") in fact
    # an estimate alone is a recorded, non-final cost, never unknown and never final
    _usage, fact = run([(1, {"cost": 0.4, "cost_final": False}, [])])
    assert "own recorded cost $0.40 subtotal, not final" in fact and "unknown" not in fact
    # provider-priced only: final, no qualifier
    usage, fact = run([(1, priced, [_delegate_call()]), (2, priced, [])])
    assert "cost_final" not in usage
    assert "own recorded cost $0.25 since the last delegate_start" in fact and "not final" not in fact
    # unpriced only: no cost to sum, unknown rather than $0.00
    usage, fact = run([(1, unpriced, [_delegate_call()]), (2, unpriced, [])])
    assert "cost" not in usage and usage["cost_final"] is False
    assert "own recorded cost unknown (no provider price recorded)" in fact and "$0.00" not in fact


def test_an_expensive_no_tool_round_still_counts(tmp_path):
    """F12 (sol #11): the loop marks metered progress after EVERY LLM response,
    including tool-less rounds — the same call the loop makes before its
    final-answer branch, with an empty tool list, must advance the progress
    mark while never advancing the delegate baseline."""
    from ouroboros.loop import (
        _nanny_metered_since_delegate_activity,
        _note_nanny_delegate_activity,
    )

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
    progress mark and the baseline, so the new task starts with zero measured
    burn and a facts line counting from task start."""
    from ouroboros.agent import reset_nanny_economics_marks
    from ouroboros.loop import _nanny_metered_since_delegate_activity
    from ouroboros.nanny_pacing import nanny_burn_fact

    ctx = _nanny_ctx(
        _nanny_metered_progress={"round": 30, "cost": 9.0},
        _nanny_delegate_baseline={"round": 5, "cost": 1.0},
        _nanny_finalization_injected=True,
    )
    reset_nanny_economics_marks(ctx, route_dispatched=True)
    fresh = SimpleNamespace()
    reset_nanny_economics_marks(fresh, route_dispatched=True)
    assert not hasattr(fresh, "_nanny_reminder_mark")  # #1748: the reminder and its cursor are gone
    assert ctx._nanny_route_dispatched is True
    assert ctx._nanny_finalization_injected is False
    assert _nanny_metered_since_delegate_activity(ctx) == (0, None)  # nothing observed yet: unknown, not $0
    assert "no delegate_start/schedule_subagent call yet; since task start 0 own rounds" in nanny_burn_fact(ctx)
    # And a non-harness dispatch resets the route latch off.
    reset_nanny_economics_marks(ctx, route_dispatched=False)
    assert ctx._nanny_route_dispatched is False


# -- finalization: a succeeded run buys no round; the forced prompt states the burn


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


def test_succeeded_run_with_a_big_metered_tail_buys_no_overrun_round(tmp_path):
    """#1748: NANNY_METERED_OVERRUN is gone (base witness: the former
    ``test_succeeded_run_with_a_big_metered_tail_gets_the_overrun_reminder`` re-looped
    this exact state). The burn is on every round's facts line instead."""
    from ouroboros.loop import _maybe_inject_finalization_nudges

    drive = _custody_drive(tmp_path)
    _succeeded_run(drive)
    ctx = _nanny_ctx(
        _nanny_finalization_injected=False,
        _nanny_metered_progress={"round": 30, "cost": 9.0},
        _nanny_delegate_baseline={"round": 5, "cost": 1.0},
        drive_root=drive,
    )
    tools = SimpleNamespace(_ctx=ctx, available_tools=lambda: ["delegate_start", "delegate_wait"])
    msgs: list = []
    assert _maybe_inject_finalization_nudges(
        tools, drive, "child-1",
        {"reasoning_notes": [], "tool_calls": []}, "done", msgs, lambda *_: None,
    ) is False
    assert msgs == []


def _forced_ctx(tmp_path, **marks):
    return _nanny_ctx(drive_root=tmp_path, task_id="child-1", **marks)


def test_forced_wrapup_over_a_succeeded_run_states_the_burn_without_advice(tmp_path):
    """F16 (grok): a forced exit may not re-loop and carries no facts line, so the
    burn rides the ONE forced prompt as a fact — no threshold and no advice."""
    from ouroboros.loop import _forced_delegation_note

    drive = _custody_drive(tmp_path)
    _succeeded_run(drive)
    ctx = _forced_ctx(
        tmp_path,
        _nanny_metered_progress={"round": 30, "cost": 9.0},
        _nanny_delegate_baseline={"round": 5, "cost": 1.0},
    )
    note = _forced_delegation_note(ctx, {"tool_calls": []})
    assert "succeeded" in note and "25 own rounds" in note and "$8.00" in note
    assert "honestly" not in note and "Account" not in note
    # Below the old 8-round/$2 threshold the fact is still stated: no threshold remains.
    small = _forced_ctx(
        tmp_path,
        _nanny_metered_progress={"round": 8, "cost": 1.2},
        _nanny_delegate_baseline={"round": 6, "cost": 1.0},
    )
    assert "2 own rounds" in _forced_delegation_note(small, {"tool_calls": []})


def test_forced_wrapup_over_a_succeeded_run_with_no_own_burn_is_silent(tmp_path):
    from ouroboros.loop import _forced_delegation_note

    drive = _custody_drive(tmp_path)
    _succeeded_run(drive)
    ctx = _forced_ctx(
        tmp_path,
        _nanny_metered_progress={"round": 6, "cost": 1.0},
        _nanny_delegate_baseline={"round": 6, "cost": 1.0},
    )
    assert _forced_delegation_note(ctx, {"tool_calls": []}) == ""


# -- Configured-session work order wire budget ---------------------------------


def test_work_order_preserves_complete_fields_above_former_total_budget():
    from ouroboros.subagent_work_order import compile_external_work_order

    objective = "яё𐍈🚀\n" * 55_000 + "DECISIVE_OBJECTIVE_TAIL"
    rendered = compile_external_work_order({"id": "child", "objective": objective})
    assert len(objective) > 250_000
    assert objective in rendered and "OMISSION NOTE" not in rendered


def test_an_ordinary_contract_field_reaches_the_run_instructions_complete(tmp_path, monkeypatch):
    """End to end, the old per-field 4k prefix is gone from the wire."""
    request = _start_with_contract(tmp_path, monkeypatch, {
        "objective": "O" * 4050,
        "expected_output": "ok",
    })
    instructions = request["instructions"]
    assert "OMISSION NOTE" not in instructions
    assert instructions.count("O" * 4050) == 1


@pytest.mark.parametrize("objective", [
    "UNIQUE_ATOMIC_OBJECTIVE",
    "яё𐍈🚀\n" * 55_000 + "LARGE_COMPILED_OBJECTIVE_TAIL",
], ids=["short", "large-unicode"])
def test_atomic_compiled_work_order_sends_dynamic_brief_once(tmp_path, monkeypatch, objective):
    from ouroboros.subagent_work_order import compile_external_work_order

    task = {
        "id": "t-nanny",
        "objective": objective,
        "expected_output": "verified patch",
        "task_contract": {
            "objective": objective,
            "expected_output": "verified patch",
        },
    }
    work_order = compile_external_work_order(task)
    request = _start_with_contract(
        tmp_path,
        monkeypatch,
        task["task_contract"],
        prompt=work_order,
        compiled_work_order=True,
    )
    assert request["prompt"] == work_order
    assert request["prompt"].count(objective) == 1
    assert objective not in request["instructions"]
    assert "git commit" in request["instructions"]


def test_compiled_work_order_carries_exact_context_and_contract_tail():
    from ouroboros.subagent_work_order import compile_external_work_order

    context_tail = "API_CONTEXT_DECISIVE_TAIL"
    intent_tail = "NEST_L1_L2_L3"
    task = {
        "id": "t-authority",
        "objective": "research",
        "context": "c" * 700 + context_tail,
        "task_contract": {
            "objective": "research",
            "context": "c" * 700 + context_tail,
            "delegation_budget": {"intent_note": "i" * 700 + intent_tail},
        },
    }

    rendered = compile_external_work_order(task)

    assert context_tail in rendered
    assert intent_tail in rendered
    assert "TASK CONTRACT AUTHORITY" in rendered
    assert "chars omitted" not in rendered


def test_compiled_work_order_preserves_context_boundary_whitespace():
    from ouroboros.subagent_work_order import compile_external_work_order

    context = " \nEXACT_CONTEXT_WITH_BOUNDARIES\n "
    rendered = compile_external_work_order({
        "id": "t-boundaries",
        "objective": "research",
        "context": context,
        "task_contract": {"objective": "research", "context": context},
    })

    assert context in rendered


def test_authority_fingerprint_binds_every_normalized_contract_field(tmp_path):
    from ouroboros.delegate_recovery import authority_fingerprint_from_task

    base = {
        "id": "t-fingerprint",
        "drive_root": str(tmp_path),
        "task_contract": {
            "objective": "build",
            "context": "context-A",
            "constraints": "constraint-A",
            "delegation_budget": {"intent_note": "intent-A"},
        },
    }
    original = authority_fingerprint_from_task(base)
    for field, value in (
        ("context", "context-B"),
        ("constraints", "constraint-B"),
        ("delegation_budget", {"intent_note": "intent-B"}),
        ("predecessor_authority", {"source": {"task_id": "different"}}),
    ):
        changed = json.loads(json.dumps(base))
        changed["task_contract"][field] = value
        assert authority_fingerprint_from_task(changed) != original


def test_normalized_contract_rebuild_keeps_workspace_lineage_and_predecessor():
    from ouroboros.contracts.task_contract import build_task_contract

    normalized = build_task_contract({
        "id": "child", "workspace_root": "/tmp/project", "workspace_mode": "external",
        "parent_task_id": "parent", "root_task_id": "root", "session_id": "session",
        "delegation_role": "subagent", "predecessor_authority": {
            "source": {"task_id": "previous"},
            "task_contract": {"objective": "exact predecessor"},
        },
    })

    assert build_task_contract({"task_contract": normalized}) == normalized
