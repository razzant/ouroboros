"""The opt-in feasibility gate, the verified reset and the turn reserve they spend.

Split verbatim out of ``tests/test_osworld_cu_bridge.py`` by theme. This module owns the
gate verdict that fails open unless the answer is explicitly infeasible, the tool set the
gate phase may hold, the reset that is verified rather than assumed, and the per-task turn
accounting that keeps the gate's reserve honest.

These exercise the pure helpers only — no OSWorld VM, no Ouroboros server.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb
from devtools.benchmarks.osworld import (
    cu_bridge_budget,
    cu_bridge_gate,
    cu_bridge_runtime,
    cu_bridge_tool_policy,
)
from ouroboros.extension_loader import extension_surface_name


# The cu_bridge runner was split into owner leaves (v7 stream W). A seam like
# `_api` is now owned by one module and bound by name in the others, so a patch
# that reached only `rcb` would silently miss the leaf that actually calls it.
_CU_BRIDGE_MODULES = (
    rcb, cu_bridge_runtime, cu_bridge_tool_policy, cu_bridge_gate, cu_bridge_budget,
)

def _patch_bridge_seam(monkeypatch, name, value):
    """Patch a cu_bridge seam on EVERY module that binds it."""
    bound = [module for module in _CU_BRIDGE_MODULES if hasattr(module, name)]
    assert bound, name
    for module in bound:
        monkeypatch.setattr(module, name, value)

class _GateArgs:
    """Minimal stand-in for the parsed CLI namespace the gate helpers read."""

    def __init__(self, *, feasibility_gate: bool, task_timeout_sec: int = 3600,
                 data_dir: str = "/nonexistent-bench-data", max_steps: int = 0):
        self.feasibility_gate = feasibility_gate
        self.task_timeout_sec = task_timeout_sec
        # The gate poll reads the task's LIVE event log to enforce its turn
        # share, so the namespace carries the bench data dir like the real one.
        self.data_dir = data_dir
        self.max_steps = max_steps

@pytest.mark.parametrize(
    "latest,expected",
    [
        ({"result": "~/Desktop is empty; nothing to act on.\nINFEASIBLE"}, "INFEASIBLE"),
        ({"result": "The file is there.\nPROCEED"}, "PROCEED"),
        ({"result": "Cloudflare blocked the page.\nUNDETERMINED"}, "UNDETERMINED"),
        # Everything below must FAIL OPEN: the working phase still runs.
        ({"result": "a discussion that never states a verdict"}, "UNDETERMINED"),
        ({"result": "I weighed whether this is INFEASIBLE and decided it is not"}, "UNDETERMINED"),
        ({"status": "timeout"}, "UNDETERMINED"),
        ({}, "UNDETERMINED"),
        (None, "UNDETERMINED"),
        # The terminal answer field wins over the runtime result body.
        ({"final_answer": "PROCEED", "result": "INFEASIBLE"}, "PROCEED"),
    ],
)
def test_gate_verdict_fails_open_unless_explicitly_infeasible(latest, expected):
    assert rcb._gate_verdict(latest) == expected

def test_gate_verdict_reads_the_answer_not_a_recap_of_the_options():
    """Regression: reverse-scanning every line for a keyword read a model's own
    enumeration of the three options as its verdict, turning a PROCEED into a scored
    hard zero. Only the last line — what the prompt actually asks for — may decide."""
    recap = (
        "I inspected the desktop as instructed.\n\n"
        "Ruling out each option in turn:\n"
        "UNDETERMINED\n"
        "PROCEED\n"
        "INFEASIBLE\n\n"
        "None of those obstacles apply here: the file exists and the app supports the\n"
        "feature, so the task is clearly PROCEED.\n"
    )
    assert rcb._gate_verdict({"result": recap}) != "INFEASIBLE"

def test_gate_verdict_tolerates_formatting_but_not_prose():
    # Ordinary formatting of a real verdict is accepted.
    for ok in ("INFEASIBLE", "INFEASIBLE.", "**INFEASIBLE**", "`infeasible`"):
        assert rcb._gate_verdict({"result": ok}) == "INFEASIBLE", ok
    # A verdict embedded in a sentence is NOT a verdict: fail open instead of guessing.
    for not_a_verdict in ("the answer is INFEASIBLE", "INFEASIBLE, probably", ""):
        assert rcb._gate_verdict({"result": not_a_verdict}) != "INFEASIBLE", not_a_verdict

def test_gate_window_is_zero_when_disabled_and_floored_when_enabled():
    assert rcb._gate_window_sec(_GateArgs(feasibility_gate=False)) == 0.0
    assert rcb._gate_window_sec(_GateArgs(feasibility_gate=True, task_timeout_sec=3600)) == 900.0
    # Floor: a tiny task timeout must not shrink the phase to nothing.
    assert rcb._gate_window_sec(_GateArgs(feasibility_gate=True, task_timeout_sec=100)) == 60.0

def test_gate_claim_window_tracks_the_single_premise_round():
    """The gate occupies the claim holder BEFORE the working task. If its occupancy is not
    in the staleness bound, a second lane can reclaim a task the first is still working and
    both will score it. Since v6.81.1 the premise phase is exactly ONE round (the
    confirming challenger was removed: 20 invocations, 0 saves, 1 loss, and it confirmed
    every false kill — correlated errors, not an independent check), so the claim window
    must equal one gate window, not two."""
    from devtools.benchmarks.osworld.run_step_agent import claim_stale_sec

    args = _GateArgs(feasibility_gate=True, task_timeout_sec=3600)
    assert rcb._gate_claim_window_sec(args) == rcb._gate_window_sec(args) == 900.0
    base = claim_stale_sec(3600, 900, 900)
    assert base + rcb._gate_claim_window_sec(args) == base + 900.0
    assert rcb._gate_claim_window_sec(_GateArgs(feasibility_gate=False)) == 0.0, \
        "ungated runs unchanged"

def test_terminal_answer_text_prefers_final_answer_then_falls_back():
    assert rcb._terminal_answer_text({"final_answer": "done", "result": "other"}) == "done"
    # The documented fallback: the field that actually carries the text on this runner.
    assert rcb._terminal_answer_text({"final_answer": "", "result": "the real answer"}) == "the real answer"
    assert rcb._terminal_answer_text({"final_answer": "   ", "result": "x"}) == "x"
    assert rcb._terminal_answer_text({}) == ""
    assert rcb._terminal_answer_text(None) == ""

def test_gate_phase_removes_the_mutating_tools_and_keeps_the_reading_ones():
    normal = set(rcb._effective_disabled_tools(False))
    gated = set(rcb._effective_disabled_tools(False, gate_phase=True))
    assert normal < gated, "the gate phase must disable strictly more than the working phase"
    # NAMED literals, deliberately not derived from _GUI_ACTION_TOOLS: the v6.81.1 review
    # caught the aliases registered in the skill but missing from that set — the gate could
    # click through them. A test iterating the same incomplete set cannot catch that class,
    # so this list is the independent statement of what "mutating" means.
    mutating_tools = ("click", "double_click", "triple_click", "move", "left_click_drag",
                      "mouse_down", "mouse_up", "type_text", "key", "hold_key", "scroll")
    assert set(mutating_tools) == set(rcb._GUI_ACTION_TOOLS), \
        "a click alias was registered without updating _GUI_ACTION_TOOLS (or vice versa)"
    for mutating in mutating_tools:
        assert extension_surface_name(rcb.SKILL_NAME, mutating) in gated, mutating
        assert extension_surface_name(rcb.SKILL_NAME, mutating) not in normal, mutating
    # Observation and read-only probing must survive, or the phase cannot establish anything.
    for readable in ("screenshot", "window_list", "wait", "remote_exec"):
        assert extension_surface_name(rcb.SKILL_NAME, readable) not in gated, readable

def test_acceptance_claims_are_general_and_well_formed():
    """These travel to the reviewer that already runs. They must carry no task id, no
    application name and nothing about how the benchmark grades."""
    from ouroboros.contracts.task_contract import normalize_acceptance_claims

    claims = rcb._ACCEPTANCE_CLAIMS
    assert claims, "the panel runs either way; empty claims is what we are fixing"
    assert normalize_acceptance_claims(claims), "must survive the contract normalizer"
    blob = json.dumps(claims).lower()
    for forbidden in ("osworld", "evaluator", "gimp", "chrome", "libreoffice", "reward",
                      "infeasible task", "1 in 13"):
        assert forbidden not in blob, forbidden
    assert len({c["id"] for c in claims}) == len(claims), "claim ids must be unique"

class _FakeResetEnv:
    """DesktopEnv stand-in for _reset_verified: scripted setup outcomes per attempt.

    `plan` is a list of per-attempt behaviours: "ok" (setup succeeds), "silent"
    (reset returns but setup silently failed — the OSWorld fail-open path),
    "noshot" (no screenshot), "raise" (reset raises).
    """

    def __init__(self, plan, config=({"type": "download"},)):
        self.plan = list(plan)
        self.config = list(config)
        self.is_environment_used = False
        self.calls = 0
        self.used_flag_at_entry: list[bool] = []

    def reset(self, task_config=None):
        self.used_flag_at_entry.append(self.is_environment_used)
        behaviour = self.plan[min(self.calls, len(self.plan) - 1)]
        self.calls += 1
        # reset() always clears the flag after the revert, like the real one.
        self.is_environment_used = False
        if behaviour == "raise":
            raise RuntimeError("boot failed")
        if behaviour == "ok":
            self.is_environment_used = True
        self._behaviour = behaviour

    def _get_obs(self):
        return {"screenshot": b"" if self._behaviour == "noshot" else b"\x89PNG"}

def test_reset_verified_rejects_the_silent_setup_skip_and_recovers_on_retry():
    """Regression for the 2026-07-28 smoke: OSWorld's reset() skips ALL setup steps when
    the guest probe times out, raises nothing, and logs "Environment setup complete." The
    working phase then opens on a VM without the task's files. The postcondition is
    machine-checkable (`is_environment_used`), so the helper must reject such an attempt
    and succeed on a later healthy one."""
    env = _FakeResetEnv(["silent", "ok"])
    rec = rcb._reset_verified(env, {"config": env.config}, retries=3,
                              deadline=time.time() + 300, wait_after_sec=0,
                              sleep=lambda _s: None)
    assert rec["attempts"] == 2
    assert env.calls == 2

def test_reset_verified_forces_the_snapshot_revert_before_every_retry():
    """After a failed setup `is_environment_used` is False, and OSWorld's reset() then
    SKIPS the snapshot revert ("environment is clean") — an unforced retry would run
    setup on top of the partial state. The helper must force the flag True before the
    retry so the revert actually happens."""
    env = _FakeResetEnv(["silent", "silent", "ok"])
    rcb._reset_verified(env, {"config": env.config}, retries=3,
                        deadline=time.time() + 300, wait_after_sec=0,
                        sleep=lambda _s: None)
    assert env.used_flag_at_entry == [False, True, True]

def test_reset_verified_exhaustion_is_a_typed_infra_error_not_a_pass():
    env = _FakeResetEnv(["silent"])
    with pytest.raises(rcb.ResetUnverified) as exc:
        rcb._reset_verified(env, {"config": env.config}, retries=2,
                            deadline=time.time() + 300, wait_after_sec=0,
                            sleep=lambda _s: None)
    assert "silently failed" in str(exc.value)
    assert isinstance(exc.value.record.get("log_tail"), list)

def test_reset_verified_accepts_a_task_with_no_setup_config():
    """A task with an empty config never sets `is_environment_used`; that is OSWorld's
    documented behaviour, not a failure. Requiring the flag unconditionally would turn
    every no-setup task into an infra abort."""
    env = _FakeResetEnv(["silent"], config=())
    rec = rcb._reset_verified(env, {"config": []}, retries=1,
                              deadline=time.time() + 300, wait_after_sec=0,
                              sleep=lambda _s: None)
    assert rec["attempts"] == 1

def test_reset_verified_still_rejects_a_missing_screenshot():
    env = _FakeResetEnv(["noshot", "ok"])
    rec = rcb._reset_verified(env, {"config": env.config}, retries=3,
                              deadline=time.time() + 300, wait_after_sec=0,
                              sleep=lambda _s: None)
    assert rec["attempts"] == 2

def test_the_confirming_challenger_stays_removed():
    """v6.81.1 removed the second premise round. Its full-run ledger: 20 invocations,
    0 feasible tasks saved, 1 officially-infeasible task lost, 215 worker rounds burned,
    and it CONFIRMED all 4 of the gate's false kills — an identical-prompt re-read
    produces correlated errors, not an independent check. Guard the removal: the flow
    must post exactly ONE premise task per example and carry no challenger machinery."""
    assert not hasattr(rcb, "_kill_confirmed")
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    flow = src[src.index("claim_fd: int | None = None"):]
    assert flow.count("_gate_round(") == 1, "exactly one premise round per example"
    assert '"feasibility_gate_challenger": False' in src, \
        "the manifest must disclose the challenger's absence to cross-run readers"

def test_gate_cancel_unconfirmed_is_the_one_condition_that_may_not_fail_open():
    """A premise round whose cancel did not confirm leaves a zombie session sharing the
    lane's server and skill connection file — it would act on the same VM the worker is
    scored on. Detection must be exact: timeouts whose cancel DID confirm proceed."""
    assert rcb._gate_cancel_unconfirmed({"status": "timeout", "cancel_confirmed": False})
    assert rcb._gate_cancel_unconfirmed({"status": "timeout"})
    assert not rcb._gate_cancel_unconfirmed({"status": "timeout", "cancel_confirmed": True})
    assert not rcb._gate_cancel_unconfirmed({"status": "completed"})
    assert not rcb._gate_cancel_unconfirmed({})

def test_gate_round_posts_a_fresh_memory_gate_phase_task_and_reads_the_verdict(monkeypatch):
    posted = {}

    def fake_api(url, method, path, payload=None, timeout=None):
        if method == "POST" and path == "/api/tasks":
            posted.update(payload)
            return {"task_id": "gate-1"}
        if method == "GET":
            return {"status": "completed", "result": "the pack list has no such locale.\nINFEASIBLE",
                    "total_rounds": 4}
        raise AssertionError((method, path))

    _patch_bridge_seam(monkeypatch, "_api", fake_api)
    args = _GateArgs(feasibility_gate=True, task_timeout_sec=3600)
    args.allow_a11y = False
    args.ouroboros_url = "http://127.0.0.1:1"
    rec = rcb._gate_round(args.ouroboros_url, args, "change the UI language", role="gate")
    assert rec["verdict"] == "INFEASIBLE" and rec["role"] == "gate"
    assert rec["task_id"] == "gate-1" and rec["llm_rounds"] == 4
    # Independence and confinement travel in the payload itself.
    assert posted["memory_mode"] == "empty"
    assert set(rcb._effective_disabled_tools(False, gate_phase=True)) <= set(posted["disabled_tools"])

def test_gate_tool_trace_carries_full_args_for_the_offline_audit(tmp_path):
    """The read-only promise is auditable only if the sidecar carries every shell command
    VERBATIM: the GAIA leakage audit was blinded by exactly this (truncated previews on one
    arm). Rows from other tasks and non-skill tools must not leak into the trace."""
    from ouroboros.extension_loader import extension_name_prefix

    prefix = extension_name_prefix(rcb.SKILL_NAME)
    long_cmd = "find / -name '*.pak' " + "-o -name 'x' " * 120
    log_dir = tmp_path / "state" / "headless_tasks" / "gate42" / "data" / "logs"
    log_dir.mkdir(parents=True)
    rows = [
        {"type": "tool_call", "tool": prefix + "remote_exec", "args": {"command": long_cmd}},
        {"type": "tool_call", "tool": prefix + "screenshot", "args": {}, "is_error": False},
        {"type": "tool_call", "tool": "web_search", "args": {"q": "not a skill tool"}},
        {"type": "llm_round", "tool": prefix + "remote_exec"},
    ]
    (log_dir / "tools.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    trace = rcb._gate_tool_trace(tmp_path, "gate42")
    assert [t["tool"] for t in trace] == ["remote_exec", "screenshot"]
    assert trace[0]["args"]["command"] == long_cmd, "args must be verbatim, not a preview"
    assert rcb._gate_tool_trace(tmp_path, "") == []
    assert rcb._gate_tool_trace(tmp_path, "no-such-task") == []

def test_the_post_gate_reset_republishes_the_vm_endpoint():
    """The repair the v1 smoke actually needed. DockerProvider.revert_to_snapshot stops
    the container and start_emulator REALLOCATES ports, so the VM address changes on
    every reset. v1 published it once, before the gate (83/83 task dirs had bridge.json
    older than their gate record), so the working phase drove the pre-gate address —
    which another lane's container could already own. Pin the ordering: the post-gate
    reset must be followed by a target write and a _publish_target call, before the
    working task is created."""
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    post_gate = src.index('reset_diag["post_gate"]')
    republish = src.index("_publish_target(data_dir, target)", post_gate)
    worker_post = src.index('"acceptance_claims": _ACCEPTANCE_CLAIMS', post_gate)
    assert post_gate < republish < worker_post, \
        "the endpoint must be republished after the post-gate reset and before the worker starts"
    # And the target file the skill reads must be rewritten too, not just the sidecar.
    assert src.index("Path(args.target_file).expanduser().write_text(target", post_gate) < republish

def test_gate_preamble_is_a_rubric_not_an_exception_list():
    """The v6.81.0 false kills shared one shape: the gate judged whether the OUTCOME
    would be meaningful instead of whether the REQUESTED ACTION is performable. The fix
    is a semantic decomposition; pin its load-bearing steps so a later edit cannot
    quietly regress the prompt into an example list."""
    p = rcb.GATE_PREAMBLE
    for step in ("ACTION", "REFERENT", "BLOCKING", "ACQUISITION", "SAME-THING CHECK",
                 "CHECK, DO NOT ASSUME", "STORE-OR-RENDER", "PLACEHOLDERS"):
        assert step in p, step
    assert "When in doubt, answer UNDETERMINED" in p, "fail-open stays the default"
    # The forced two-round vision loop is gone: screenshots attach automatically.
    assert "view_image(path)" not in rcb.OSWORLD_PREAMBLE
    assert "attached" in rcb.OSWORLD_PREAMBLE.lower()

def test_gate_rubric_covers_named_mode_scope_and_prohibition():
    """Forensics: two gate PROCEEDs reinterpreted a named mode ('batch') and a launch
    scope (per-app vs per-folder) as working-phase details, and one prohibition
    ('without configuring X') was never verified as satisfiable — all three hide the
    premise in a modifier rather than a noun. Pin the 4d branch; the fail-open default
    must survive it."""
    p = rcb.GATE_PREAMBLE
    assert "NAMED MODE, SCOPE AND PROHIBITION" in p
    for phrase in ("MODE OF OPERATION", "APPLY SCOPE", "PROHIBITION"):
        assert phrase in p, phrase
    assert "When in doubt, answer UNDETERMINED" in p, "fail-open stays the default"

def _ns(**kw):
    from types import SimpleNamespace
    kw.setdefault("feasibility_gate", False)
    kw.setdefault("max_steps", 0)
    return SimpleNamespace(**kw)

def test_step_budget_uses_policy_turns_not_gui_actions():
    """A leaderboard step is one top-level policy turn: the official loop increments
    step_idx once per agent.predict() and executes every action that call emitted
    inside that step. The earlier 0.42-actions-per-round mapping compared a turn
    against an action. The declared budget must reserve the gate phase AND one
    tool-less terminal turn out of the claim, so a forced finalization is never
    step N+1."""
    b = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                         {"value": 85, "source": "settings"})
    assert b["step_semantics"] == "top_level_policy_turn"
    assert b["max_steps_claimed"] == 100 and b["enforced"] is True
    assert b["terminal_turn_reserve"] == 1
    assert b["gate_turn_reserve"] == rcb._GATE_TURN_RESERVE
    assert b["action_capable_round_cap"] == 100 - rcb._GATE_TURN_RESERVE - 1
    # Without the gate phase its reserve is not withheld.
    b2 = rcb._step_budget(_ns(max_steps=100), {"value": 99, "source": "settings"})
    assert b2["gate_turn_reserve"] == 0 and b2["action_capable_round_cap"] == 99
    # No claim -> nothing enforced, and the run is not comparable.
    b3 = rcb._step_budget(_ns(), {"value": 200, "source": "default"})
    assert b3["enforced"] is False and b3["max_steps_claimed"] is None

def test_a_step_claim_the_server_cannot_honor_is_refused_before_the_vm_boots():
    """Enforcement lives in the runtime round cap; the runner must PROVE that cap is
    at or below the declared budget before anything costs money. 'Most tasks finish
    early' is not a substitute — comparability is a per-task property."""
    import pytest

    over = rcb._step_budget(_ns(max_steps=100), {"value": 200, "source": "settings"})
    with pytest.raises(SystemExit, match="exceeds"):
        rcb._refuse_uncapped_step_claim(over)
    ok = rcb._step_budget(_ns(max_steps=100), {"value": 99, "source": "settings"})
    rcb._refuse_uncapped_step_claim(ok)  # must not raise
    # A claim so small the reserves swallow it is refused too.
    tiny = rcb._step_budget(_ns(max_steps=1, feasibility_gate=True), {"value": 1, "source": "env"})
    with pytest.raises(SystemExit, match="no working turns"):
        rcb._refuse_uncapped_step_claim(tiny)
    # An unenforced run is never refused (it simply is not comparable).
    rcb._refuse_uncapped_step_claim(rcb._step_budget(_ns(), {"value": 999, "source": "default"}))

def test_audit_reads_policy_turns_not_physical_calls():
    """The flat `total_rounds` on a task result is reconstructed from
    physical_calls — safety checks, acceptance reviewers and retries included —
    and on the v6.81.1 run it disagreed with the loop's own turn count on 344 of
    346 examples, running up to 13 higher. Auditing a step budget against it
    would mark compliant examples as overruns. Pin the loop field as the source
    and pin fail-closed behaviour when it is missing."""
    budget = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                              {"value": 85, "source": "settings"})
    # A result whose physical and policy counts deliberately differ.
    latest = {"total_rounds": 97, "loop_outcome": {"usage": {"total_rounds": 84}}}
    assert rcb._policy_turns(latest) == 84
    inside = rcb._audit_step_budget(budget, rcb._policy_turns(latest), 5)
    assert inside["policy_turns_used"] == 89 and inside["budget_fault"] is False
    assert inside["turn_source"] == "loop_outcome.usage.total_rounds"
    # Same example audited against the physical count would have been a fault.
    assert 97 + 5 > 100
    # Missing loop accounting fails CLOSED rather than coercing to zero.
    assert rcb._policy_turns({"total_rounds": 40}) is None
    blind = rcb._audit_step_budget(budget, rcb._policy_turns({"total_rounds": 40}), 3)
    assert blind["counts_available"] is False and blind["budget_fault"] is True
    # A real overrun is a harness fault, not a row-filtering criterion.
    over = rcb._audit_step_budget(budget, 99, 6)
    assert over["policy_turns_used"] == 105 and over["budget_fault"] is True
    assert "comparable" not in over
    # Undeclared budget: nothing to audit against, and that is stated.
    assert rcb._audit_step_budget(rcb._step_budget(_ns(), {"value": 200, "source": "default"}),
                                  5, 0)["audited"] is False

def test_gate_turns_are_enforced_per_task_from_the_live_event_log(tmp_path, monkeypatch):
    """The runtime round cap is SERVER-wide and the gate is a separate task, so a
    reserve that is only arithmetic lets the gate consume the worker's allowance.

    The enforcement must read the LIVE counter: `loop_outcome` is written only at
    finalization, so polling a running task for it yields None forever and any
    check built on it is dead code. `llm_round` events are emitted at the same
    statement that increments the loop's round counter, so counting them equals
    the turn count the task will eventually report."""
    task_id = "gate123"
    logs = tmp_path / "state" / "headless_tasks" / task_id / "data" / "logs"
    logs.mkdir(parents=True)
    events = logs / "events.jsonl"

    def _write_rounds(n: int) -> None:
        events.write_text("".join(
            json.dumps({"type": "llm_round", "task_id": task_id, "round": i + 1}) + "\n"
            for i in range(n)
        ), encoding="utf-8")

    _write_rounds(3)
    assert rcb._live_policy_turns(tmp_path, task_id) == 3
    # A finalization-only shape is NOT what the runtime serves while running.
    assert rcb._policy_turns({"status": "running", "total_rounds": 9}) is None

    calls = {"cancel": 0}
    polls = {"n": 0}

    def fake_api(url, method, path, payload=None, timeout=None):
        if path.endswith("/cancel"):
            calls["cancel"] += 1
            return {}
        polls["n"] += 1
        if calls["cancel"]:
            return {"status": "cancelled"}
        # The gate crosses its reserve between the first and second poll.
        _write_rounds(3 if polls["n"] < 2 else rcb._GATE_TURN_RESERVE)
        return {"status": "running"}

    orig_sleep = rcb.time.sleep
    _patch_bridge_seam(monkeypatch, "_api", fake_api)
    rcb.time.sleep = lambda s: None
    try:
        out = rcb._await_gate_task("http://x", task_id, time.time() + 3600,
                                   turn_budget=rcb._GATE_TURN_RESERVE, data_dir=tmp_path)
    finally:
        rcb.time.sleep = orig_sleep

    assert out["status"] == "turn_budget_exhausted"
    assert out["policy_turns"] == rcb._GATE_TURN_RESERVE
    assert calls["cancel"] == 1
    # An unconfirmed cancel of THIS status is a zombie premise session, exactly
    # like the timeout path — it must not fail open into the working phase.
    assert rcb._gate_cancel_unconfirmed({"status": "turn_budget_exhausted"}) is True
    assert rcb._gate_cancel_unconfirmed(
        {"status": "turn_budget_exhausted", "cancel_confirmed": True}) is False
    # No declared budget -> no per-task enforcement (unchanged legacy behaviour).
    assert rcb._gate_turn_budget(_ns(feasibility_gate=True)) == 0
    assert rcb._gate_turn_budget(_ns(max_steps=100, feasibility_gate=True)) == rcb._GATE_TURN_RESERVE
    # An unreadable log is UNKNOWN, never zero.
    assert rcb._live_policy_turns(tmp_path / "nope", task_id) is None

def test_unknown_gate_turns_keep_the_full_reserve(tmp_path):
    """UNKNOWN is not zero. If the gate's turn count cannot be read, granting
    claimed-1 turns would let the worker blow the declared total after an
    unmeasured gate — the audit would then call the already-scored campaign
    non-comparable. Fail closed: keep the worst-case reserve."""
    budget = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                              {"value": 99, "source": "settings"})
    assert rcb._worker_round_cap(budget, None) == 100 - rcb._GATE_TURN_RESERVE - 1

def test_unused_gate_reserve_is_returned_to_the_worker(tmp_path):
    """The static reserve is worst-case: the gate is budgeted 14 turns but spent a
    mean of 4 on the v6.83.0 run, so a flat max_steps-14-1 threw ~10 turns away on
    every example and 13 of 56 opus failures died at 89-92 turns INSIDE a 100-turn
    budget. Returning the unused reserve must keep the declared total intact."""
    budget = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                              {"value": 99, "source": "settings"})
    # Gate spent 4 -> worker may use 95, and 4 + 95 + 1 terminal == 100.
    assert rcb._worker_round_cap(budget, 4) == 95
    assert 4 + 95 + budget["terminal_turn_reserve"] == budget["max_steps_claimed"]
    # A gate that used its whole reserve leaves the old conservative number.
    assert rcb._worker_round_cap(budget, 14) == 85
    # No declared budget -> nothing to publish.
    assert rcb._worker_round_cap(rcb._step_budget(_ns(), {"value": 200, "source": "default"}), 4) is None

    # The cap is written where the server hot-reloads it from.
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({"OUROBOROS_MAX_ROUNDS": 99, "OTHER": "keep"}), encoding="utf-8")
    rec = rcb._publish_worker_round_cap(sp, 95)
    assert rec["applied"] is True and rec["previous"] == 99
    on_disk = json.loads(sp.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_MAX_ROUNDS"] == 95 and on_disk["OTHER"] == "keep"
    # An unwritable target is disclosed, never fatal (the stricter cap stays).
    bad = rcb._publish_worker_round_cap(tmp_path / "nope" / "settings.json", 95)
    assert bad["applied"] is False and "error" in bad

def test_a_gate_terminated_example_is_not_a_budget_fault():
    """A gate INFEASIBLE ends the example before the working phase, so the worker
    used exactly zero policy turns — a KNOWN count. Treating it as unknown made
    the fail-closed audit flag the very outcome the gate exists to produce
    (caught live on os/a462a795 minutes into the v6.83.0 run)."""
    budget = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                              {"value": 85, "source": "settings"})
    gated = rcb._audit_step_budget(budget, 0, 4, gate_expected=True)
    assert gated["budget_fault"] is False and gated["policy_turns_used"] == 4
    # A genuinely unknown worker count still fails closed.
    unknown = rcb._audit_step_budget(budget, None, 4, gate_expected=True)
    assert unknown["budget_fault"] is True
