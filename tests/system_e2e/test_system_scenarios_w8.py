"""S30–S31 — the exact mid-run budget pause and its owner-granted same-id Resume
(#1196), as a REAL consumer of the candidate (wave 8): the MANAGED root (S30) and
the DIRECT owner-chat turn (S31).

Owner Q1–Q10 (#1196): a task that meets a monetary rail AFTER work has been done
no longer buys a wrap-up call and ends ``failed/budget_exhausted`` — it PAUSES
exactly at a completed boundary, nonterminal, under its own task id, and only an
explicit owner Resume (never a budget increase) continues it. The unit suites
(``tests/test_budget_pause_exact*.py``) pin every seam against stubbed contexts;
THIS scenario drives the whole path on a real isolated server: a pooled worker,
the supervisor's queue transition, the durable task-result authority, the queue
snapshot, the owner HTTP surfaces (``/api/state``, ``/api/settings``,
``/api/tasks/{id}/resume``) and the loopback model — keyless, with the money made
REAL by a stub that reports a provider-final cost on every call.

DETERMINISM. Exhaustion is a function of the call count, not of time: every stub
completion costs the same known amount, the wallet is a small multiple of it, and
the task's in-task graceful ceiling is disabled through its own contract
(``budget_profile.cost_hard_stop_pct = 0``) so the ONE rail that can fire is the
ledger's global dispatch fence (``usage_accounting.reserve_attempt`` →
``BudgetExceeded`` → ``loop_budget._handle_budget_exceeded`` → exact pause on the
``dispatch_refused`` rail). Every synchronization point is durable-event polling
over ``ArtifactOracle`` readers; the one NEGATIVE claim (a budget increase wakes
nobody) is a bounded watch for any wake signal, which must stay silent.

WHAT S30 ASSERTS, in order:
1. PAUSED, NONTERMINAL, SAME ID — after several priced tool-calling rounds the
   durable row carries a ``budget_pause`` projection in state ``paused`` (pause_id,
   pause_generation 1, a stored ``source_ref`` the tree's own reader opens, the
   rail, the exact-continuation facts, ``physical_calls`` ≥ 2, a ``boundary``
   resume point) on a row whose status is ``scheduled`` with
   ``reason_code=budget_paused`` — NOT a terminal; the queue snapshot holds the
   task as a PENDING carrier under ``_budget_pause`` (``paused_exact_continuation``,
   the checkpoint naming the same pause) and not in RUNNING; ``events.jsonl`` has
   the ``budget_scope_paused`` row (park_state ``paused``, worker_event source) and
   NO ``task_done`` for the id; ``/api/state.active_chat_activities`` reports the
   root as ``managed_task`` in phase ``budget_paused``; the worker made no paid
   wrap-up call after the pause decision (no ``[BUDGET LIMIT]`` prompt ever reached
   the stub, and the stub's call count is frozen from the pause on);
2. RESUME WHILE STILL EXHAUSTED is the typed 409 ``budget_still_exhausted`` with
   ``action=increase_budget_then_resume``: no grant on the durable row, the carrier
   untouched;
3. A BUDGET INCREASE ALONE WAKES NOTHING — ``POST /api/settings`` raises
   ``TOTAL_BUDGET`` (read back from the settings file the server owns), and a
   bounded watch sees no wake signal: the row stays ``paused`` without a grant, the
   carrier stays PENDING, no model call, no resume event, the activity phase stays
   ``budget_paused``;
4. EXPLICIT RESUME after the increase is the typed 2xx grant (``exact_continuation``,
   a ``grant_id``, generation 1, the paused interval carried separately); the
   durable row records the SINGLE-USE grant and the loop CONSUMES it (state
   ``resumed``, ``grant.consumed_at``), the resumed model round carries the host's
   continuation notice and the pre-pause transcript, the task completes under the
   SAME id with ``task_done`` exactly once, and the ledger-derived rounds and spend
   are cumulative (never reset: the pause-time numbers are a lower bound of the
   terminal ones);
5. a REPEATED resume on the completed task is the typed 404 ``task_not_pending``.

S30's PUBLIC status plane (``GET /api/tasks/{id}`` and the status-filtered list,
both served by ``task_status.load_effective_task_result``) is pinned by its OWN
test after the lifecycle proof, so a projection defect there is a named red
instead of a masked lifecycle: a paused root must not read as ``running``.

S31 drives the SAME lifecycle for the owner's DIRECT chat turn over the /ws the SPA
opens: ordinary turns have no default cost stop (#1643), so the known-spend
``dispatch_refused`` rail fires; the live in-process actor ends
and the turn's own record is parked INLINE as the PENDING carrier with its
``_is_direct_chat`` lane fact; the census reports it as ``direct_chat`` /
``budget_paused``; the increase wakes nothing; the Resume grant refreshes no
planning threshold on this hard rail; a POOLED worker continues the checkpoint,
and the turn concludes in the chat: the durable
chat row and the keyed final frame over the same socket, the activity gone.

The default-lane tests pin the literals the claims depend on (the paid wrap-up
prompt head, the resume notice, its Q10 branch) and the priced stub's wire shape
(a reported cost on both the JSON and SSE envelopes).
"""

from __future__ import annotations

import json
import pathlib
import re
import urllib.request
import uuid

import pytest

from tests.system_e2e.harness import (
    LANE_MOCK,
    REPO_ROOT,
    ArtifactOracle,
    ScriptedStubModel,
    body_text,
    keyless_settings,
    require_lane,
    start_server,
    wait_durable_result,
    wait_until,
    ws_url,
)
# The wave-2 keepalive step: one author for the "keep listing the root" round.
from tests.system_e2e.test_system_scenarios_w2 import KEEPALIVE_STEP
# The wave-6 direct-chat glue (the SAME /ws the SPA opens, its frame collector and
# the activity/running readers): one author across the direct-turn scenarios.
from tests.system_e2e.test_system_scenarios_w6 import (
    S26_CHAT_ID,
    _WsFrames,
    _direct_activities,
    _running_task_ids,
)

from devtools.benchmarks.common.server_runner import _api, _api_status

# ---------------------------------------------------------------------------
# Money shape of the scenario. Every stub call costs S30_CALL_COST (provider
# FINAL); the wallet holds S30_TOTAL_BUDGET, so the ledger fence refuses the
# (S30_TOTAL_BUDGET / S30_CALL_COST + 1)-th send, whichever actor makes it.
# The per-task cap stays far above the wallet so the ROOT fence never binds
# and the pause is unambiguously the GLOBAL wallet's.
# ---------------------------------------------------------------------------
S30_CALL_COST = 0.25
S30_TOTAL_BUDGET = 1.0
S30_RAISED_BUDGET = 50.0
S30_PER_TASK_CAP = 10.0
# Longer than the rounds the wallet affords, so the pause is the budget rail and
# not the script's end; short enough that the resumed task finishes quickly.
S30_SCRIPT_STEPS = 10
# The negative watch: the supervisor assigns every 0.5 s (server.py loop), so a
# wake that was going to happen has happened many times over inside this window.
S30_WAKE_WATCH_SEC = 10.0

# Literals of the tree under test (the default-lane pin greps them out of the
# source so a drift is a named failure, not a silently mute negative claim):
#   FORCED_WRAPUP_MARKER — ouroboros/loop_budget.py: the prompt head of every
#     PAID best-effort wrap-up call the pre-#1196 rails used to buy;
#   RESUME_NOTICE_MARKER — ouroboros/budget_pause.py::resume_paused_loop: the
#     host notice appended to the transcript on a consumed grant.
#   REFRESH_SOURCE_LITERAL — the same notice's Q10 branch as written in the source
#     (a %-format); the resumed round of a GRACEFUL pause carries its rendering,
#     REFRESHED_NOTICE_MARKER, followed by the refreshed ceiling.
FORCED_WRAPUP_MARKER = "[BUDGET LIMIT]"
RESUME_NOTICE_MARKER = "This task continued from its budget pause after"
REFRESH_SOURCE_LITERAL = "'refreshed to $%.2f'"
REFRESHED_NOTICE_MARKER = "planning threshold refreshed to $"
LITERAL_SOURCES = {
    FORCED_WRAPUP_MARKER: "ouroboros/loop_budget.py",
    RESUME_NOTICE_MARKER.replace("budget pause", "{kind}"): "ouroboros/budget_pause.py",
    REFRESH_SOURCE_LITERAL: "ouroboros/budget_pause.py",
}

_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")


class PricedStubModel(ScriptedStubModel):
    """``ScriptedStubModel`` whose every completion reports a provider FINAL cost.

    The keyless stub has no price catalog (``pricing.get_pricing`` answers ``{}``
    for the openai-compatible route), so its calls settle as UNKNOWN money and can
    never exhaust a wallet. A budget scenario needs the ledger to see real
    dollars: the OpenAI-compatible transport honours a ``usage.cost`` the provider
    reports (``llm_openai_compatible``: a reported cost is final money, the catalog
    estimate runs only when it is absent), which is the OpenRouter wire shape.
    Every call — agent round, host periphery, review — costs the same known
    amount, so exhaustion is a deterministic function of the call count.
    """

    def __init__(self, script=None, *, cost_usd: float, **kwargs) -> None:
        super().__init__(script, **kwargs)
        self.cost_usd = float(cost_usd)

    def _completion(self, body: dict) -> dict:
        completion = super()._completion(body)
        completion["usage"] = {**completion["usage"], "cost": self.cost_usd}
        return completion


# ===========================================================================
# Default lane: the literal pins and the priced stub's wire shape.
# ===========================================================================


def test_w8_literals_still_exist_in_the_tree():
    """The two negative claims of S30 read literals of the tree: the paid wrap-up
    prompt head (must NEVER reach the stub after a pause) and the resume notice
    (MUST reach it on the resumed round). A literal that drifted upstream would
    make the first claim vacuously green and the second red by the wrong name."""
    # Runtime scenarios keep the concrete budget notice; source selects the
    # owner/authority/budget wording through the existing mapping. Pin those choices too.
    source = (REPO_ROOT / "ouroboros/budget_pause.py").read_text(encoding="utf-8")
    assert 'REASON_OWNER: "Pause"' in source
    assert 'REASON_AUTHORITY: f"technical pause (' in source
    assert 'str(row.get("reason") or ""), "budget pause")' in source
    for literal, relpath in LITERAL_SOURCES.items():
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert literal in source, f"{relpath} no longer carries {literal!r}"


def _post_completion(stub: ScriptedStubModel, payload: dict) -> bytes:
    req = urllib.request.Request(
        stub.base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def test_priced_stub_reports_a_final_cost_on_json_and_sse_envelopes():
    with PricedStubModel([dict(KEEPALIVE_STEP)], cost_usd=0.25) as stub:
        tool_body = {"messages": [{"role": "user", "content": "go"}],
                     "tools": [{"type": "function", "function": {"name": "list_files"}}]}
        plain = json.loads(_post_completion(stub, tool_body))
        assert plain["usage"]["cost"] == 0.25, plain["usage"]
        assert plain["usage"]["prompt_tokens"] == 10, plain["usage"]
        streamed = _post_completion(stub, {**tool_body, "stream": True}).decode("utf-8")
        frames = [json.loads(line[len("data: "):]) for line in streamed.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        usage_frames = [f for f in frames if isinstance(f.get("usage"), dict)]
        assert usage_frames and usage_frames[-1]["usage"]["cost"] == 0.25, frames
        assert stub.kinds() == ["agent", "final"]


# ===========================================================================
# Mock lane: S30 on a real isolated server.
# ===========================================================================


def _submit_priced_root(server, description: str) -> str:
    """Submit a root task whose contract DISABLES the in-task graceful ceiling.

    ``budget_profile.cost_hard_stop_pct = 0`` rides the task metadata into the
    task contract (``task_contract.build_task_contract``), so the only monetary
    rail left is the ledger's global dispatch fence — the scenario's rail.
    """
    created = _api(server.base_url, "POST", "/api/tasks", {
        "description": description,
        "memory_mode": "forked",
        "actor_id": "e2e-driver", "source": "e2e-driver",
        "timeout_sec": 1800,
        "metadata": {"source": "e2e-driver", "delegation_role": "root",
                     "budget_profile": {"cost_hard_stop_pct": 0}},
    }, timeout=60)
    task_id = str(created.get("task_id") or "")
    assert task_id, created
    return task_id


def _pending_carrier(oracle: ArtifactOracle, task_id: str) -> dict | None:
    """The task's PENDING row in the durable queue snapshot — the nested ``task``
    record (``queue_snapshot.persist_queue_snapshot`` keeps the scheduling facts
    on the outer row and the whole task, markers included, under ``task``)."""
    for row in oracle.queue_snapshot().get("pending") or []:
        if isinstance(row, dict) and str(row.get("id") or "") == task_id:
            task = row.get("task")
            return dict(task) if isinstance(task, dict) else {"id": task_id}
    return None


def _pause_row(oracle: ArtifactOracle, task_id: str) -> dict:
    row = oracle.task_result(task_id).get("budget_pause")
    return dict(row) if isinstance(row, dict) else {}


def _task_done_rows(oracle: ArtifactOracle, task_id: str) -> list:
    return [r for r in oracle.events("task_done") if str(r.get("task_id") or "") == task_id]


def _events_for(oracle: ArtifactOracle, event_type: str, task_id: str) -> list:
    return [r for r in oracle.events(event_type) if str(r.get("task_id") or "") == task_id]


def _managed_activity(server, task_id: str) -> dict | None:
    """The root's row in ``/api/state.active_chat_activities`` (the SPA's census)."""
    state = _api(server.base_url, "GET", "/api/state", timeout=30)
    for row in state.get("active_chat_activities") or []:
        if isinstance(row, dict) and str(row.get("activity_id") or "") == task_id:
            return row
    return None


def _settings_on_disk(server) -> dict:
    """The settings document the SERVER owns (the file ``start_server`` wrote and
    ``POST /api/settings`` rewrites) — the durable readback of an owner save."""
    return json.loads(pathlib.Path(server.settings_path).read_text(encoding="utf-8"))


def _resume(server, task_id: str) -> dict:
    """``POST /api/tasks/{id}/resume`` — the owner's Resume, typed at every status."""
    return _api_status(server.base_url, "POST", f"/api/tasks/{task_id}/resume", {}, timeout=120)


def _detail(server, task_id: str) -> dict:
    return _api(server.base_url, "GET", f"/api/tasks/{task_id}", timeout=30)


def _agent_rounds(stub: ScriptedStubModel) -> int:
    """Tool-bearing agent rounds the stub answered with a scripted tool call."""
    return sum(1 for kind in stub.kinds() if kind == "agent")


def _bodies_carrying(stub: ScriptedStubModel, marker: str) -> list:
    with stub._lock:
        calls = list(stub.calls)
    return [body for _kind, body in calls if marker in body_text(body)]


def _ledger_bucket(root, task_id: str) -> dict:
    """This task's bucket of the physical-attempt ledger (``state/usage_attempts.jsonl``
    under the budget drive root) through the tree's OWN reader — the authority
    ``reconstruct_task_cost`` and the terminal write derive ``total_rounds`` /
    ``accounted_upper_bound_usd`` from. Read-only."""
    from ouroboros.usage_accounting import usage_breakdown

    bucket = usage_breakdown(pathlib.Path(root), task_id=task_id)
    assert not bucket.get("integrity_degraded"), bucket
    return {"physical_calls": int(bucket.get("physical_calls") or 0),
            "accounted_usd": float(bucket.get("accounted_usd") or 0.0)}


def _usage_rows(oracle: ArtifactOracle, task_id: str) -> list:
    """The task's durable ``llm_usage`` rows (one per paid model round)."""
    return [r for r in oracle.events("llm_usage") if str(r.get("task_id") or "") == task_id]


def _submit_and_pause(server, oracle: ArtifactOracle) -> tuple[str, dict]:
    """Submit the priced root and wait for its durable ``paused`` budget_pause row.

    Returns ``(task_id, paused_row)``; the contract precondition of the rail choice
    is checked on the way (a silently dropped profile would re-enable the graceful
    ceiling and pause the task on a different rail with money still left).
    """
    from ouroboros.budget_pause import STATE_PAUSED

    task_id = _submit_priced_root(
        server, "Keep listing the repository root; the owner controls your budget.")
    assert wait_until(lambda: task_id in oracle.running_ids(), 120), (
        f"task {task_id} never reached the RUNNING set")
    contract = oracle.task_result(task_id).get("task_contract") or {}
    assert (contract.get("budget_profile") or {}).get("cost_hard_stop_pct") == 0, contract
    paused = wait_until(
        lambda: (_pause_row(oracle, task_id)
                 if _pause_row(oracle, task_id).get("state") == STATE_PAUSED else None),
        300)
    assert paused, (
        f"the task never reached a durable 'paused' budget_pause row: "
        f"{oracle.task_result(task_id)!r}")
    return task_id, paused


def _wake_signal(oracle: ArtifactOracle, server, stub: ScriptedStubModel, task_id: str,
                 calls_at_pause: int) -> str:
    """The FIRST sign that the paused task woke, or "" while it is still asleep.

    One predicate for the bounded negative watch: named, so a red watch says WHAT
    woke it instead of "something changed".
    """
    row = _pause_row(oracle, task_id)
    if str(row.get("state") or "") != "paused":
        return f"durable pause state moved to {row.get('state')!r}"
    if row.get("grant"):
        return f"a grant appeared on the durable row: {row.get('grant')!r}"
    if task_id in oracle.running_ids():
        return "the task is back in the queue snapshot's RUNNING set"
    carrier = _pending_carrier(oracle, task_id)
    if carrier is None or not isinstance(carrier.get("_budget_pause"), dict):
        return f"the PENDING carrier lost its _budget_pause marker: {carrier!r}"
    if carrier.get("_budget_pause_resume"):
        return f"a resume handoff appeared on the carrier: {carrier.get('_budget_pause_resume')!r}"
    if len(stub.kinds()) != calls_at_pause:
        return f"the model was called again ({len(stub.kinds())} calls, {calls_at_pause} at the pause)"
    if _events_for(oracle, "budget_task_explicitly_resumed", task_id):
        return "a budget_task_explicitly_resumed event was written"
    if _task_done_rows(oracle, task_id):
        return "a task_done event was written"
    return ""


@pytest.mark.integration
def test_s30_exact_budget_pause_then_owner_resume_continues_the_same_task_id(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.budget_pause import (
        RAIL_DISPATCH_REFUSED, RAIL_GLOBAL_EXHAUSTED, REASON_CODE, RESUME_POLICY,
        STATE_PAUSED, STATE_RESUME_GRANTED, STATE_RESUMED, STATUS_PAUSED_EXACT,
    )
    from ouroboros.task_results import STATUS_SCHEDULED

    root = tmp_path_factory.mktemp("s30")
    script = [dict(KEEPALIVE_STEP) for _ in range(S30_SCRIPT_STEPS)]
    with PricedStubModel(script, cost_usd=S30_CALL_COST,
                         final_answer="S30_FINAL: the paused task finished after its Resume.") as stub:
        settings = keyless_settings(
            stub, TOTAL_BUDGET=S30_TOTAL_BUDGET, OUROBOROS_PER_TASK_COST_USD=S30_PER_TASK_CAP)
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)

            # ---------------------------------------------------------------
            # 1. PAUSED, NONTERMINAL, SAME ID.
            # ---------------------------------------------------------------
            task_id, paused = _submit_and_pause(server, oracle)
            # Freeze the model's call ledger the moment the pause is durable: the
            # negative claims below are measured against this number.
            calls_at_pause = len(stub.kinds())
            agent_rounds_at_pause = _agent_rounds(stub)
            assert 2 <= agent_rounds_at_pause < S30_SCRIPT_STEPS, (
                "the pause must follow SEVERAL priced tool rounds and precede the script's end",
                stub.kinds())
            assert not stub.script_consumed(), "the script ran out: this was not a budget pause"

            # The durable pause row — the source of truth the queue marker locates.
            pause_id = str(paused.get("pause_id") or "")
            assert _HEX32_RE.match(pause_id), paused
            assert int(paused.get("pause_generation") or 0) == 1, paused
            assert paused.get("source_ref"), paused
            assert paused.get("rail") in {RAIL_DISPATCH_REFUSED, RAIL_GLOBAL_EXHAUSTED}, paused
            assert paused.get("scope") == "global", paused
            assert paused.get("exact_continuation") is True, paused
            assert paused.get("replay_safe") is False and paused.get("auto_resume") is False, paused
            assert paused.get("resume_policy") == RESUME_POLICY, paused
            assert int(paused.get("task_attempt") or 0) == 1, paused
            assert paused.get("is_direct_chat") is False, paused
            assert int(paused.get("physical_calls") or 0) >= 2, paused
            assert "grant" not in paused, paused
            assert float(paused["paused_at"]) <= float(paused["paused_confirmed_at"]), paused
            assert paused.get("pause_source") == "worker_event", paused
            point = paused.get("resume_point") or {}
            assert point.get("phase") == "boundary", point
            assert point.get("unanswered_tool_call_ids") == [], point
            assert point.get("unanswered_policy") == "not_re_executed_execution_unknown", point
            # The stored continuation is readable through the tree's OWN reader
            # and names this task, this pause and this attempt.
            stored_row = oracle.task_result(task_id)
            assert stored_row.get("status") == STATUS_SCHEDULED, stored_row.get("status")
            assert stored_row.get("reason_code") == REASON_CODE, stored_row.get("reason_code")
            assert (stored_row.get("resource_limit") or {}).get("status") == STATUS_PAUSED_EXACT, (
                stored_row.get("resource_limit"))
            carrier = _pending_carrier(oracle, task_id)
            assert carrier is not None, oracle.queue_snapshot()
            source_root = pathlib.Path(str(carrier.get("budget_drive_root") or server.data_root))
            state = json.loads(read_actor_source_bytes(source_root, task_id, paused["source_ref"]))
            assert state.get("task_id") == task_id and state.get("pause_id") == pause_id, (
                {k: state.get(k) for k in ("task_id", "pause_id", "task_attempt", "round_idx")})
            assert int(state.get("task_attempt") or 0) == 1, state.get("task_attempt")
            assert int(state.get("round_idx") or 0) >= 2, state.get("round_idx")

            # The queue snapshot: a PENDING carrier under _budget_pause, not RUNNING.
            assert task_id not in oracle.running_ids(), oracle.queue_snapshot().get("running")
            marker = carrier.get("_budget_pause")
            assert isinstance(marker, dict), carrier
            assert marker.get("status") == STATUS_PAUSED_EXACT, marker
            assert marker.get("exact_continuation") is True, marker
            assert marker.get("replay_safe") is False and marker.get("auto_resume") is False, marker
            assert marker.get("resume_policy") == RESUME_POLICY, marker
            assert marker.get("scope") == "global" and marker.get("root_task_id") == task_id, marker
            checkpoint = marker.get("checkpoint") or {}
            assert checkpoint.get("pause_id") == pause_id, checkpoint
            assert int(checkpoint.get("pause_generation") or 0) == 1, checkpoint
            assert checkpoint.get("source_ref") == paused.get("source_ref"), checkpoint
            assert not carrier.get("_is_direct_chat"), carrier
            assert not carrier.get("_budget_pause_resume") and not carrier.get("_budget_pause_hold"), carrier

            # No terminal: no task_done, and the owner-visible park event is there.
            assert _task_done_rows(oracle, task_id) == [], _task_done_rows(oracle, task_id)
            park_events = wait_until(
                lambda: [r for r in _events_for(oracle, "budget_scope_paused", task_id)
                         if r.get("pause_id") == pause_id] or None, 30)
            assert park_events, _events_for(oracle, "budget_scope_paused", task_id)
            assert park_events[-1].get("park_state") == STATE_PAUSED, park_events[-1]
            assert park_events[-1].get("pause_source") == "worker_event", park_events[-1]
            assert park_events[-1].get("status") == STATUS_PAUSED_EXACT, park_events[-1]
            assert not {"owner_visible", "toast_once"}.intersection(park_events[-1]), park_events[-1]

            # The owner's census: the root is a managed task in phase budget_paused.
            activity = wait_until(
                lambda: (_managed_activity(server, task_id)
                         if (_managed_activity(server, task_id) or {}).get("phase") == "budget_paused"
                         else None), 30)
            assert activity, _managed_activity(server, task_id)
            assert activity.get("kind") == "managed_task", activity
            assert int(activity.get("task_attempt") or 0) == 1, activity

            # The pause-time LEDGER numbers this task must never fall below again
            # (cumulative, not reset): the physical-attempt ledger, the durable
            # llm_usage rows and the pause row's own physical_calls agree. (The
            # PUBLIC detail's status/cost planes of a paused root are pinned by
            # test_s30_public_detail_of_a_paused_root_reports_the_pause_not_running —
            # its own test, so a projection defect there cannot mask this proof.)
            ledger_at_pause = _ledger_bucket(source_root, task_id)
            rounds_at_pause = ledger_at_pause["physical_calls"]
            spend_at_pause = ledger_at_pause["accounted_usd"]
            assert rounds_at_pause == int(paused.get("physical_calls") or 0) >= 2, (
                ledger_at_pause, paused.get("physical_calls"))
            assert rounds_at_pause == len(_usage_rows(oracle, task_id)) == agent_rounds_at_pause, (
                ledger_at_pause, len(_usage_rows(oracle, task_id)), agent_rounds_at_pause)
            assert spend_at_pause >= rounds_at_pause * S30_CALL_COST - 1e-9, ledger_at_pause
            assert all(r.get("cost") == S30_CALL_COST and r.get("cost_known") is True
                       for r in _usage_rows(oracle, task_id)), _usage_rows(oracle, task_id)

            # NO paid wrap-up after the pause decision: the forced best-effort
            # prompt never reached the model.
            assert _bodies_carrying(stub, FORCED_WRAPUP_MARKER) == [], (
                "a paid [BUDGET LIMIT] wrap-up call was made")

            # ---------------------------------------------------------------
            # 2. RESUME WHILE STILL EXHAUSTED: typed refusal, no grant.
            # ---------------------------------------------------------------
            refused = _resume(server, task_id)
            assert refused.get("status") == 409, refused
            assert (refused.get("body") or {}).get("error") == "budget_still_exhausted", refused
            assert (refused.get("body") or {}).get("action") == "increase_budget_then_resume", refused
            assert (refused.get("body") or {}).get("task_id") == task_id, refused
            after_refusal = _pause_row(oracle, task_id)
            assert after_refusal.get("state") == STATE_PAUSED and "grant" not in after_refusal, after_refusal
            assert after_refusal.get("pause_id") == pause_id, after_refusal
            assert _events_for(oracle, "budget_task_explicitly_resumed", task_id) == []
            carrier = _pending_carrier(oracle, task_id)
            assert carrier and isinstance(carrier.get("_budget_pause"), dict), carrier
            assert not carrier.get("_budget_pause_resume"), carrier

            # ---------------------------------------------------------------
            # 3. A BUDGET INCREASE ALONE WAKES NOTHING.
            # ---------------------------------------------------------------
            saved = _api_status(server.base_url, "POST", "/api/settings",
                                {"TOTAL_BUDGET": S30_RAISED_BUDGET}, timeout=120)
            assert saved.get("status") == 200, saved
            assert (saved.get("body") or {}).get("saved") is not False, saved
            on_disk = wait_until(
                lambda: (_settings_on_disk(server)
                         if float(_settings_on_disk(server).get("TOTAL_BUDGET") or 0) == S30_RAISED_BUDGET
                         else None), 30)
            assert on_disk, _settings_on_disk(server).get("TOTAL_BUDGET")
            # The bounded negative watch: the FIRST wake signal, or none.
            woke = wait_until(
                lambda: _wake_signal(oracle, server, stub, task_id, calls_at_pause) or None,
                S30_WAKE_WATCH_SEC)
            assert not woke, f"the budget increase woke the paused task: {woke}"
            assert len(stub.kinds()) == calls_at_pause, stub.kinds()
            assert (_managed_activity(server, task_id) or {}).get("phase") == "budget_paused", (
                _managed_activity(server, task_id))
            assert _task_done_rows(oracle, task_id) == []

            # ---------------------------------------------------------------
            # 4. EXPLICIT RESUME: single-use grant, same id, completes, cumulative.
            # ---------------------------------------------------------------
            granted = _resume(server, task_id)
            assert granted.get("status") == 200, granted
            grant_body = granted.get("body") or {}
            assert grant_body.get("ok") is True and grant_body.get("task_id") == task_id, grant_body
            assert grant_body.get("exact_continuation") is True, grant_body
            assert grant_body.get("root_task_id") == task_id, grant_body
            grant_id = str(grant_body.get("grant_id") or "")
            assert _HEX32_RE.match(grant_id), grant_body
            assert int(grant_body.get("grant_generation") or 0) == 1, grant_body
            assert float(grant_body.get("paused_duration_sec") or 0) >= S30_WAKE_WATCH_SEC, grant_body
            assert grant_body.get("eligible_descendants") == [] and grant_body.get("held_siblings") == [], grant_body
            resumed_events = wait_until(
                lambda: [r for r in _events_for(oracle, "budget_task_explicitly_resumed", task_id)
                         if r.get("grant_id") == grant_id] or None, 30)
            assert resumed_events and len(resumed_events) == 1, (
                _events_for(oracle, "budget_task_explicitly_resumed", task_id))
            assert resumed_events[0].get("exact_continuation") is True, resumed_events[0]
            assert resumed_events[0].get("same_generation") is True, resumed_events[0]
            assert resumed_events[0].get("selected_by") == "owner", resumed_events[0]

            # The grant is RECORDED on the durable row and then CONSUMED by the
            # loop (the resume_granted state is transient: a worker takes the row
            # within one assignment tick, so the durable proof of the grant is the
            # consumed grant carrying the response's own identity).
            consumed = wait_until(
                lambda: (_pause_row(oracle, task_id)
                         if _pause_row(oracle, task_id).get("state") == STATE_RESUMED else None), 180)
            assert consumed, _pause_row(oracle, task_id)
            assert consumed.get("pause_id") == pause_id, consumed
            assert int(consumed.get("resume_generation") or 0) == 1, consumed
            grant = consumed.get("grant") or {}
            assert grant.get("grant_id") == grant_id, grant
            assert grant.get("single_use") is True, grant
            assert int(grant.get("generation") or 0) == 1, grant
            assert grant.get("pause_id") == pause_id and int(grant.get("pause_generation") or 0) == 1, grant
            assert grant.get("consumed_at") and not grant.get("revoked_at"), grant
            assert grant.get("selected_by") == "owner", grant
            assert float(grant.get("paused_duration_sec") or 0) >= S30_WAKE_WATCH_SEC, grant
            assert grant.get("refresh_planning_threshold") is False, grant  # a HARD rail: no Q10 refresh
            assert STATE_RESUME_GRANTED != consumed.get("state")  # the grant did not stall undispatched

            # Completion under the SAME id.
            stored = wait_durable_result(oracle, task_id, timeout=600)
            assert stored.get("status") == "completed", stored
            assert "S30_FINAL" in str(stored.get("result") or ""), stored.get("result")
            done_rows = wait_until(lambda: _task_done_rows(oracle, task_id) or None, 60)
            assert done_rows and len(done_rows) == 1, done_rows
            assert done_rows[0].get("status") == "completed", done_rows[0]
            assert wait_until(lambda: task_id not in oracle.running_ids()
                              and _pending_carrier(oracle, task_id) is None, 60), oracle.queue_snapshot()
            assert wait_until(lambda: _managed_activity(server, task_id) is None, 60), (
                _managed_activity(server, task_id))
            # The pause projection survives the terminal write as history: resumed,
            # consumed, the same identities.
            final_pause = stored.get("budget_pause") or {}
            assert final_pause.get("state") == STATE_RESUMED and final_pause.get("pause_id") == pause_id, final_pause
            assert (final_pause.get("grant") or {}).get("grant_id") == grant_id, final_pause

            # The resumed round is a CONTINUATION: the model saw the host notice and
            # the pre-pause transcript (its earlier tool batches), never a fresh start.
            resumed_bodies = _bodies_carrying(stub, RESUME_NOTICE_MARKER)
            assert resumed_bodies, "no model round carried the resume notice: the task did not continue"
            first_resumed = resumed_bodies[0]
            prior_batches = sum(
                1 for m in first_resumed.get("messages") or []
                if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"))
            assert prior_batches >= agent_rounds_at_pause, (prior_batches, agent_rounds_at_pause)
            assert "were NOT reset" in body_text(first_resumed)
            assert _bodies_carrying(stub, FORCED_WRAPUP_MARKER) == [], (
                "a paid [BUDGET LIMIT] wrap-up call was made after all")
            assert stub.script_consumed(), "the resumed task never ran the script to its end"
            assert _agent_rounds(stub) == S30_SCRIPT_STEPS, stub.kinds()

            # Cumulative, never reset: the ledger of the SAME task id only grew across
            # the pause, the terminal row's ledger-derived numbers (written at the
            # terminal) sit between the pause-time and the live reading, and the
            # public detail of the COMPLETED task carries them under honest names.
            ledger_done = _ledger_bucket(source_root, task_id)
            rounds_done, spend_done = ledger_done["physical_calls"], ledger_done["accounted_usd"]
            assert rounds_done >= rounds_at_pause + S30_SCRIPT_STEPS - agent_rounds_at_pause + 1, (
                ledger_at_pause, ledger_done, agent_rounds_at_pause)  # every remaining round + the final
            assert spend_done >= spend_at_pause + S30_CALL_COST, (ledger_at_pause, ledger_done)
            usage_rows_done = _usage_rows(oracle, task_id)
            assert len(usage_rows_done) > len(_usage_rows(oracle, task_id)[:rounds_at_pause]) == rounds_at_pause
            stored_rounds = int(stored.get("total_rounds") or 0)
            stored_spend = stored.get("accounted_upper_bound_usd")
            assert rounds_at_pause < stored_rounds <= rounds_done, (rounds_at_pause, stored_rounds, rounds_done)
            assert stored_spend is not None and spend_at_pause < float(stored_spend) <= spend_done + 1e-9, (
                spend_at_pause, stored_spend, spend_done)
            detail_done = _detail(server, task_id)
            assert detail_done.get("status") == "completed", detail_done.get("status")
            assert int(detail_done.get("total_rounds") or 0) == stored_rounds, (
                detail_done.get("total_rounds"), stored_rounds)
            assert detail_done.get("accounted_upper_bound_usd") == stored_spend, (
                detail_done.get("accounted_upper_bound_usd"), stored_spend)

            # ---------------------------------------------------------------
            # 5. A REPEATED RESUME on the completed task is the typed refusal.
            # ---------------------------------------------------------------
            again = _resume(server, task_id)
            assert again.get("status") == 404, again
            assert (again.get("body") or {}).get("error") == "task_not_pending", again
            assert (again.get("body") or {}).get("task_id") == task_id, again
            assert len(_task_done_rows(oracle, task_id)) == 1
            assert len(_events_for(oracle, "budget_task_explicitly_resumed", task_id)) == 1
            assert oracle.task_result(task_id).get("status") == "completed"
            assert (oracle.task_result(task_id).get("budget_pause") or {}).get("grant", {}).get("grant_id") == grant_id
        finally:
            server.stop()


@pytest.mark.integration
def test_s30_public_detail_of_a_paused_root_reports_the_pause_not_running(
        e2e_clone, tmp_path_factory):
    """The PUBLIC status plane of an exactly paused root must agree with its durable
    authority, its queue row and its census entry.

    ``GET /api/tasks/{id}`` and the status-filtered ``GET /api/tasks?status=running``
    are both served by ``task_status.load_effective_task_result`` — the SPA's task
    detail and task list. For a forked-drive root the authority row (the server
    root's ``task_results/<id>.json``) says ``scheduled`` with
    ``reason_code=budget_paused`` and ``resource_limit.status=paused_exact_continuation``,
    the queue holds a PENDING ``_budget_pause`` carrier (never a RUNNING row) and
    ``/api/state`` reports phase ``budget_paused``. A public read that reports the
    task as ``running`` promises work that is not happening and hides the one
    control (Resume) the owner has. Kept as its OWN test, after the lifecycle proof,
    so a projection defect here is a named red instead of a masked lifecycle.
    """
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import REASON_CODE, STATUS_PAUSED_EXACT
    from ouroboros.task_results import STATUS_SCHEDULED

    root = tmp_path_factory.mktemp("s30_detail")
    script = [dict(KEEPALIVE_STEP) for _ in range(S30_SCRIPT_STEPS)]
    with PricedStubModel(script, cost_usd=S30_CALL_COST) as stub:
        settings = keyless_settings(
            stub, TOTAL_BUDGET=S30_TOTAL_BUDGET, OUROBOROS_PER_TASK_COST_USD=S30_PER_TASK_CAP)
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            task_id, _paused = _submit_and_pause(server, oracle)
            # The three surfaces that agree: authority row, queue snapshot, census.
            authority = oracle.task_result(task_id)
            assert authority.get("status") == STATUS_SCHEDULED, authority.get("status")
            assert authority.get("reason_code") == REASON_CODE, authority.get("reason_code")
            assert (authority.get("resource_limit") or {}).get("status") == STATUS_PAUSED_EXACT, (
                authority.get("resource_limit"))
            assert task_id not in oracle.running_ids(), oracle.queue_snapshot().get("running")
            carrier = _pending_carrier(oracle, task_id)
            assert carrier and isinstance(carrier.get("_budget_pause"), dict), carrier
            assert wait_until(
                lambda: (_managed_activity(server, task_id) or {}).get("phase") == "budget_paused", 30), (
                _managed_activity(server, task_id))
            # The public status plane, read ONCE into one fact so a red names all of it.
            detail = _detail(server, task_id)
            budget_root = pathlib.Path(str(carrier.get("budget_drive_root") or server.data_root))
            public = {
                "detail.status": detail.get("status"),
                "detail.reason_code": detail.get("reason_code"),
                "detail.resource_limit.status": (detail.get("resource_limit") or {}).get("status"),
                "detail.total_rounds": detail.get("total_rounds"),
                "detail.accounted_upper_bound_usd": detail.get("accounted_upper_bound_usd"),
                "listed_under_status=running": task_id in _running_task_ids(server),
                "authority.status": authority.get("status"),
                "authority.budget_pause.physical_calls": _paused.get("physical_calls"),
                "ledger": _ledger_bucket(budget_root, task_id),
                "census.phase": (_managed_activity(server, task_id) or {}).get("phase"),
            }
            assert public["listed_under_status=running"] is False, public
            assert public["detail.status"] == STATUS_SCHEDULED, public
            assert public["detail.reason_code"] == REASON_CODE, public
            assert public["detail.resource_limit.status"] == STATUS_PAUSED_EXACT, public
            # The same read's LEDGER plane: a paused root's public rounds/spend must
            # be its cumulative ledger, not the worker's pre-pause replica (or zero).
            assert int(public["detail.total_rounds"] or 0) == public["ledger"]["physical_calls"], public
            assert public["detail.accounted_upper_bound_usd"] is not None, public
        finally:
            server.stop()


# ===========================================================================
# S31 — the DIRECT owner-chat turn: same pause, same id, Resume finishes in chat.
# ===========================================================================

S31_MARKER = "S31_DIRECT_TURN_e2e_w8"
S31_PROBE = (f"[{S31_MARKER}] List the repository root and keep listing it; "
             "the owner controls your budget.")
S31_FINAL = "S31_FINAL: the paused chat turn finished after its Resume."
# Ordinary direct turns have no hidden half-wallet ceiling (#1643). This wallet
# reaches the known-spend dispatch fence after several rounds, including the
# turn's tool-less periphery (the proactive card namer).
S31_TOTAL_BUDGET = 2.0


def _direct_turn_id(oracle: ArtifactOracle, client_message_id: str) -> str:
    """The direct turn minted for *client_message_id*, from its durable
    ``task_received`` row (the lane fact and the origin link ride the task record)."""
    for row in oracle.events("task_received"):
        task = row.get("task") if isinstance(row.get("task"), dict) else {}
        origin = (task.get("metadata") or {}).get("origin_message_ref") or {}
        if task.get("_is_direct_chat") and str(origin.get("client_message_id") or "") == client_message_id:
            return str(task.get("id") or "")
    return ""


@pytest.mark.integration
def test_s31_direct_chat_turn_pauses_under_its_own_id_and_owner_resume_completes_in_chat(
        e2e_clone, tmp_path_factory):
    """The owner's DIRECT chat turn meets a monetary rail mid-turn and pauses the
    SAME way, under the SAME task id (#1196): its conversation-lane carrier (a
    worker process since #2654, never the server's own thread) reports the park as
    any pooled worker does (``pause_source=worker_event``), its RUNNING row ends and
    the turn's own task record is parked as the PENDING ``_budget_pause`` carrier
    with its ``_is_direct_chat`` lane fact (never a second live actor, never a
    terminal, never a paid wrap-up), the census reports it as ``direct_chat`` in
    phase ``budget_paused``; a budget increase alone wakes nothing; the existing
    Resume endpoint grants it once, a POOLED worker continues the checkpointed
    cognition cold (the host notice, no threshold refresh for the hard rail),
    and the turn concludes IN THE CHAT: the durable chat row and
    the keyed final frame over the same /ws the SPA opens, the activity gone."""
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import (
        RAIL_DISPATCH_REFUSED, REASON_CODE, RESUME_POLICY, STATE_PAUSED, STATE_RESUMED, STATUS_PAUSED_EXACT,
    )
    from ouroboros.task_results import STATUS_SCHEDULED

    root = tmp_path_factory.mktemp("s31")
    script = [dict(KEEPALIVE_STEP) for _ in range(S30_SCRIPT_STEPS)]
    with PricedStubModel(script, cost_usd=S30_CALL_COST, final_answer=S31_FINAL) as stub:
        settings = keyless_settings(
            stub, TOTAL_BUDGET=S31_TOTAL_BUDGET, OUROBOROS_PER_TASK_COST_USD=S30_PER_TASK_CAP)
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            from websockets.sync.client import connect as ws_connect

            client_message_id = f"e2e-s31-{uuid.uuid4().hex[:12]}"
            # proxy=None: the loopback /ws is never proxied (see S26).
            with ws_connect(ws_url(server), open_timeout=30, proxy=None) as ws, \
                    _WsFrames(ws) as frames:
                ws.send(json.dumps({
                    "type": "chat", "content": S31_PROBE, "chat_id": S26_CHAT_ID,
                    "client_message_id": client_message_id,
                }))
                task_id = wait_until(lambda: _direct_turn_id(oracle, client_message_id), 120)
                assert task_id, "the direct turn never wrote its task_received row"

                # ---------------------------------------------------------------
                # 1. The turn PAUSES under its own id; the actor ends; the record parks.
                # ---------------------------------------------------------------
                paused = wait_until(
                    lambda: (_pause_row(oracle, task_id)
                             if _pause_row(oracle, task_id).get("state") == STATE_PAUSED else None), 300)
                assert paused, f"the direct turn never reached a durable 'paused' row: {oracle.task_result(task_id)!r}"
                calls_at_pause = len(stub.kinds())
                agent_rounds_at_pause = _agent_rounds(stub)
                assert 2 <= agent_rounds_at_pause < S30_SCRIPT_STEPS, stub.kinds()
                assert not stub.script_consumed()
                pause_id = str(paused.get("pause_id") or "")
                assert _HEX32_RE.match(pause_id), paused
                assert paused.get("is_direct_chat") is True, paused
                assert paused.get("rail") == RAIL_DISPATCH_REFUSED, paused
                assert paused.get("exact_continuation") is True and paused.get("replay_safe") is False, paused
                assert paused.get("resume_policy") == RESUME_POLICY, paused
                # The carrier's own pause event parked it: the retired in-process park
                # (``direct_turn_inline_park``, supervisor/direct_pause.py) is gone with #2654.
                assert paused.get("pause_source") == "worker_event", paused
                assert int(paused.get("physical_calls") or 0) >= 2, paused
                assert (paused.get("resume_point") or {}).get("phase") == "boundary", paused.get("resume_point")
                assert "grant" not in paused, paused
                stored_row = oracle.task_result(task_id)
                assert stored_row.get("status") == STATUS_SCHEDULED, stored_row.get("status")
                assert stored_row.get("reason_code") == REASON_CODE, stored_row.get("reason_code")
                assert stored_row.get("_is_direct_chat") is True, stored_row.get("_is_direct_chat")
                assert int(stored_row.get("chat_id") or 0) == S26_CHAT_ID, stored_row.get("chat_id")
                assert (stored_row.get("resource_limit") or {}).get("status") == STATUS_PAUSED_EXACT, (
                    stored_row.get("resource_limit"))
                # The PENDING carrier is the turn's OWN record with its lane fact.
                carrier = _pending_carrier(oracle, task_id)
                assert carrier is not None, oracle.queue_snapshot()
                assert carrier.get("_is_direct_chat") is True, carrier
                assert int(carrier.get("chat_id") or 0) == S26_CHAT_ID, carrier
                assert S31_MARKER in str(carrier.get("text") or ""), carrier
                marker = carrier.get("_budget_pause")
                assert isinstance(marker, dict) and marker.get("status") == STATUS_PAUSED_EXACT, carrier
                assert marker.get("exact_continuation") is True, marker
                assert (marker.get("checkpoint") or {}).get("pause_id") == pause_id, marker
                assert task_id not in oracle.running_ids(), oracle.queue_snapshot().get("running")
                budget_root = pathlib.Path(str(carrier.get("budget_drive_root") or server.data_root))
                ledger_at_pause = _ledger_bucket(budget_root, task_id)
                assert ledger_at_pause["physical_calls"] == int(paused.get("physical_calls") or 0) >= 2, (
                    ledger_at_pause, paused.get("physical_calls"))
                # The park's own receipt is the owner event of the carrier's pause; no
                # disclosed inline executor ran the turn (``conversation_inline_executor``
                # names every execution on a server thread), so the actor that parked
                # it was the lane's worker process.
                park_events = wait_until(
                    lambda: [r for r in _events_for(oracle, "budget_scope_paused", task_id)
                             if r.get("pause_id") == pause_id] or None, 30)
                assert park_events and park_events[-1].get("pause_source") == "worker_event", park_events
                assert park_events[-1].get("park_state") == STATE_PAUSED, park_events[-1]
                inline = [r for r in oracle.supervisor_rows("conversation_inline_executor")
                          if str(r.get("task_id") or "") == task_id]
                assert inline == [], inline
                # No second live actor, no terminal, no paid wrap-up.
                assert wait_until(lambda: not _direct_activities(server, client_message_id), 30), (
                    "the live direct actor is still registered after the pause")
                activity = wait_until(
                    lambda: (_managed_activity(server, task_id)
                             if (_managed_activity(server, task_id) or {}).get("phase") == "budget_paused"
                             else None), 30)
                assert activity and activity.get("kind") == "direct_chat", _managed_activity(server, task_id)
                assert _task_done_rows(oracle, task_id) == []
                assert _bodies_carrying(stub, FORCED_WRAPUP_MARKER) == []
                # The turn had announced itself on the SAME /ws (typing frame, direct_chat).
                typing = wait_until(lambda: frames.find(type="typing", activity_id=task_id) or None, 30)
                assert typing and typing[0].get("kind") == "direct_chat", [f.get("type") for f in frames.frames]
                assert typing[0].get("client_message_id") == client_message_id, typing[0]

                # ---------------------------------------------------------------
                # 2. A BUDGET INCREASE alone wakes nothing.
                # ---------------------------------------------------------------
                saved = _api_status(server.base_url, "POST", "/api/settings",
                                    {"TOTAL_BUDGET": S30_RAISED_BUDGET}, timeout=120)
                assert saved.get("status") == 200, saved
                assert wait_until(
                    lambda: float(_settings_on_disk(server).get("TOTAL_BUDGET") or 0) == S30_RAISED_BUDGET, 30)
                woke = wait_until(
                    lambda: _wake_signal(oracle, server, stub, task_id, calls_at_pause) or None,
                    S30_WAKE_WATCH_SEC)
                assert not woke, f"the budget increase woke the paused direct turn: {woke}"
                assert (_managed_activity(server, task_id) or {}).get("phase") == "budget_paused"

                # ---------------------------------------------------------------
                # 3. EXPLICIT RESUME: a pooled worker continues the SAME turn, in chat.
                # ---------------------------------------------------------------
                granted = _resume(server, task_id)
                assert granted.get("status") == 200, granted
                grant_body = granted.get("body") or {}
                assert grant_body.get("ok") is True and grant_body.get("exact_continuation") is True, grant_body
                grant_id = str(grant_body.get("grant_id") or "")
                assert _HEX32_RE.match(grant_id) and int(grant_body.get("grant_generation") or 0) == 1, grant_body
                consumed = wait_until(
                    lambda: (_pause_row(oracle, task_id)
                             if _pause_row(oracle, task_id).get("state") == STATE_RESUMED else None), 180)
                assert consumed, _pause_row(oracle, task_id)
                grant = consumed.get("grant") or {}
                assert grant.get("grant_id") == grant_id and grant.get("consumed_at"), grant
                assert grant.get("single_use") is True and not grant.get("revoked_at"), grant
                # A hard dispatch fence resumes without inventing a graceful threshold.
                assert grant.get("refresh_planning_threshold") is False, grant
                stored = wait_durable_result(oracle, task_id, timeout=600)
                assert stored.get("status") == "completed", stored
                assert S31_FINAL in str(stored.get("result") or ""), stored.get("result")
                assert stored.get("_is_direct_chat") is True, stored.get("_is_direct_chat")
                assert (stored.get("budget_pause") or {}).get("pause_id") == pause_id, stored.get("budget_pause")
                done_rows = wait_until(lambda: _task_done_rows(oracle, task_id) or None, 60)
                assert done_rows and len(done_rows) == 1 and done_rows[0].get("status") == "completed", done_rows
                resumed_bodies = _bodies_carrying(stub, RESUME_NOTICE_MARKER)
                assert resumed_bodies, "no model round carried the resume notice: the turn did not continue"
                assert REFRESHED_NOTICE_MARKER not in body_text(resumed_bodies[0]), body_text(resumed_bodies[0])[-1500:]
                prior_batches = sum(
                    1 for m in resumed_bodies[0].get("messages") or []
                    if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"))
                assert prior_batches >= agent_rounds_at_pause, (prior_batches, agent_rounds_at_pause)
                assert _bodies_carrying(stub, FORCED_WRAPUP_MARKER) == []
                assert stub.script_consumed() and _agent_rounds(stub) == S30_SCRIPT_STEPS, stub.kinds()
                ledger_done = _ledger_bucket(budget_root, task_id)
                assert ledger_done["physical_calls"] > ledger_at_pause["physical_calls"], (ledger_at_pause, ledger_done)
                assert ledger_done["accounted_usd"] >= ledger_at_pause["accounted_usd"] + S30_CALL_COST, (
                    ledger_at_pause, ledger_done)

                # The chat CONCLUDED: the durable chat row under the turn id and the
                # keyed final frame over the same /ws; the activity and the running
                # list no longer name the turn.
                chat_rows = wait_until(
                    lambda: [r for r in oracle._jsonl("logs/chat.jsonl")
                             if str(r.get("task_id") or "") == task_id
                             and S31_FINAL in str(r.get("text") or "")] or None, 60)
                assert chat_rows, "the resumed turn's final never landed in chat.jsonl under its task id"
                assert chat_rows[-1].get("direction") == "out", chat_rows[-1]
                assert int(chat_rows[-1].get("chat_id") or 0) == S26_CHAT_ID, chat_rows[-1]
                final_frames = wait_until(
                    lambda: [f for f in frames.find(type="chat", task_id=task_id)
                             if not f.get("is_progress") and S31_FINAL in str(f.get("content") or "")] or None, 60)
                assert final_frames, "no keyed final frame for the resumed turn arrived over /ws"
                assert wait_until(lambda: _managed_activity(server, task_id) is None
                                  and not _direct_activities(server, client_message_id), 60)
                assert wait_until(lambda: task_id not in _running_task_ids(server)
                                  and task_id not in oracle.running_ids(), 60)

            # ---------------------------------------------------------------
            # 4. A REPEATED resume on the completed turn is the typed refusal.
            # ---------------------------------------------------------------
            again = _resume(server, task_id)
            assert again.get("status") == 404, again
            assert (again.get("body") or {}).get("error") == "task_not_pending", again
            assert len(_task_done_rows(oracle, task_id)) == 1
        finally:
            server.stop()


# ===========================================================================
# S32 — the pause survives the physical epoch: graceful SIGTERM -> boot -> paused -> Resume.
# ===========================================================================

S32_FINAL = "S32_FINAL: the paused task finished after the restart and its Resume."


def _boot_restore_rows(oracle: ArtifactOracle) -> list:
    return oracle.supervisor_rows("queue_restored_from_snapshot")


@pytest.mark.integration
def test_s32_exact_pause_survives_a_graceful_server_restart_and_resumes_after_it(
        e2e_clone, tmp_path_factory):
    """Owner contract (#1196): an exact budget pause survives a physical epoch.

    The lifespan teardown of a graceful SIGTERM settles interrupted RUNNING work
    as ``cancelled`` (``server_shutdown``) and drains PENDING; the paused carrier is
    NOT that work. Generation A pauses the root and is stopped gracefully: the
    ``server_shutdown`` row proves the teardown's custody step ran, and the task
    leaves the epoch untouched — no task_done, no cancel, the same ``paused`` row,
    the PENDING ``_budget_pause`` carrier still in the final snapshot, no further
    model round. Generation B (same clone, data root and stub) must park the SAME
    id again from that snapshot, unheld and un-dispatched: census ``budget_paused``,
    public detail ``scheduled``/``budget_paused`` (never ``running``), Resume still
    the typed 409 while the ledger is exhausted. Only the owner's Resume after the
    increase continues and completes the same id with cumulative rounds and spend.
    """
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import REASON_CODE, STATE_PAUSED, STATE_RESUMED, STATUS_PAUSED_EXACT
    from ouroboros.task_results import STATUS_SCHEDULED
    from supervisor.events_budget import budget_hold_fact

    root = tmp_path_factory.mktemp("s32")
    script = [dict(KEEPALIVE_STEP) for _ in range(S30_SCRIPT_STEPS)]
    with PricedStubModel(script, cost_usd=S30_CALL_COST, final_answer=S32_FINAL) as stub:
        settings = keyless_settings(
            stub, TOTAL_BUDGET=S30_TOTAL_BUDGET, OUROBOROS_PER_TASK_COST_USD=S30_PER_TASK_CAP)

        # ---------------------------------------------------------------
        # Generation A: pause, then a GRACEFUL stop (SIGTERM to the tree).
        # ---------------------------------------------------------------
        server = start_server(e2e_clone, root, settings)
        oracle = ArtifactOracle(server.data_root)
        try:
            task_id, paused = _submit_and_pause(server, oracle)
            pause_id = str(paused.get("pause_id") or "")
            assert _HEX32_RE.match(pause_id), paused
            agent_rounds_at_pause = _agent_rounds(stub)
            assert wait_until(
                lambda: (_managed_activity(server, task_id) or {}).get("phase") == "budget_paused", 30), (
                _managed_activity(server, task_id))
            carrier = _pending_carrier(oracle, task_id)
            assert carrier and isinstance(carrier.get("_budget_pause"), dict), carrier
            budget_root = pathlib.Path(str(carrier.get("budget_drive_root") or server.data_root))
            ledger_at_pause = _ledger_bucket(budget_root, task_id)
            assert ledger_at_pause["physical_calls"] == int(paused.get("physical_calls") or 0) >= 2, (
                ledger_at_pause, paused.get("physical_calls"))
        finally:
            server.stop()

        # The teardown's custody step RAN (the server_shutdown row is written after
        # kill_workers) and the pause was not what it settled.
        shutdown_rows = oracle.supervisor_rows("server_shutdown")
        assert shutdown_rows and shutdown_rows[-1].get("cause") == "external_signal", shutdown_rows
        cleanup_rows = oracle.supervisor_rows("zombie_prevention_cleanup")
        settled_by_shutdown = {
            tid for row in cleanup_rows
            for tid in list(row.get("drained_pending") or []) + list(row.get("orphaned_running") or [])
        }
        assert task_id not in settled_by_shutdown, cleanup_rows
        assert any(task_id in (row.get("retained_budget_paused") or []) for row in cleanup_rows), cleanup_rows
        after_stop = oracle.task_result(task_id)
        assert after_stop.get("status") == STATUS_SCHEDULED, after_stop.get("status")
        assert after_stop.get("reason_code") == REASON_CODE, after_stop.get("reason_code")
        assert (after_stop.get("budget_pause") or {}).get("state") == STATE_PAUSED, after_stop.get("budget_pause")
        assert (after_stop.get("budget_pause") or {}).get("pause_id") == pause_id, after_stop.get("budget_pause")
        assert "grant" not in (after_stop.get("budget_pause") or {}), after_stop.get("budget_pause")
        assert _task_done_rows(oracle, task_id) == [], _task_done_rows(oracle, task_id)
        assert task_id not in oracle.running_ids(), oracle.queue_snapshot().get("running")
        carrier = _pending_carrier(oracle, task_id)
        assert carrier and ((carrier.get("_budget_pause") or {}).get("checkpoint") or {}).get("pause_id") == pause_id, (
            carrier)
        assert not carrier.get("_budget_pause_hold") and not carrier.get("_budget_pause_resume"), carrier
        assert _agent_rounds(stub) == agent_rounds_at_pause, stub.kinds()
        assert _bodies_carrying(stub, RESUME_NOTICE_MARKER) == []
        assert _bodies_carrying(stub, FORCED_WRAPUP_MARKER) == []
        restores_before = len(_boot_restore_rows(oracle))

        # ---------------------------------------------------------------
        # Generation B: same clone, data root and stub; the boot re-parks the id.
        # ---------------------------------------------------------------
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            restore = wait_until(
                lambda: (_boot_restore_rows(oracle)[restores_before:] or None), 60)
            assert restore, "generation B wrote no queue_restored_from_snapshot row"
            assert int(restore[-1].get("restored_pending") or 0) >= 1, restore[-1]
            assert task_id not in (restore[-1].get("pending_parent_interrupted") or []), restore[-1]
            assert task_id not in (restore[-1].get("terminalized_running") or []), restore[-1]
            carrier = wait_until(
                lambda: (_pending_carrier(oracle, task_id)
                         if isinstance((_pending_carrier(oracle, task_id) or {}).get("_budget_pause"), dict)
                         else None), 60)
            assert carrier, oracle.queue_snapshot()
            assert ((carrier.get("_budget_pause") or {}).get("checkpoint") or {}).get("pause_id") == pause_id, carrier
            assert budget_hold_fact(carrier) is None, carrier  # restorable: unheld
            assert not carrier.get("_budget_pause_resume"), carrier
            assert task_id not in oracle.running_ids(), oracle.queue_snapshot().get("running")
            rebooted = oracle.task_result(task_id)
            assert rebooted.get("status") == STATUS_SCHEDULED and rebooted.get("reason_code") == REASON_CODE, (
                rebooted.get("status"), rebooted.get("reason_code"))
            assert (rebooted.get("budget_pause") or {}).get("state") == STATE_PAUSED, rebooted.get("budget_pause")
            assert (rebooted.get("budget_pause") or {}).get("pause_id") == pause_id, rebooted.get("budget_pause")
            assert _task_done_rows(oracle, task_id) == []
            activity = wait_until(
                lambda: (_managed_activity(server, task_id)
                         if (_managed_activity(server, task_id) or {}).get("phase") == "budget_paused"
                         else None), 60)
            assert activity and activity.get("kind") == "managed_task", _managed_activity(server, task_id)
            # The public status plane agrees with the authority across the epoch.
            detail = _detail(server, task_id)
            assert detail.get("status") == STATUS_SCHEDULED, detail.get("status")
            assert detail.get("reason_code") == REASON_CODE, detail.get("reason_code")
            assert (detail.get("resource_limit") or {}).get("status") == STATUS_PAUSED_EXACT, detail.get("resource_limit")
            assert int(detail.get("total_rounds") or 0) == ledger_at_pause["physical_calls"], (
                detail.get("total_rounds"), ledger_at_pause)
            assert task_id not in _running_task_ids(server)
            # Still exhausted (the ledger crossed the epoch too): the typed refusal, no grant.
            refused = _resume(server, task_id)
            assert refused.get("status") == 409, refused
            assert (refused.get("body") or {}).get("error") == "budget_still_exhausted", refused
            assert "grant" not in _pause_row(oracle, task_id), _pause_row(oracle, task_id)
            assert _agent_rounds(stub) == agent_rounds_at_pause, stub.kinds()

            # The owner's Resume after the increase: ONE grant, the same id completes.
            saved = _api_status(server.base_url, "POST", "/api/settings",
                                {"TOTAL_BUDGET": S30_RAISED_BUDGET}, timeout=120)
            assert saved.get("status") == 200, saved
            assert wait_until(
                lambda: float(_settings_on_disk(server).get("TOTAL_BUDGET") or 0) == S30_RAISED_BUDGET, 30)
            granted = _resume(server, task_id)
            assert granted.get("status") == 200, granted
            grant_body = granted.get("body") or {}
            assert grant_body.get("ok") is True and grant_body.get("exact_continuation") is True, grant_body
            grant_id = str(grant_body.get("grant_id") or "")
            assert _HEX32_RE.match(grant_id) and int(grant_body.get("grant_generation") or 0) == 1, grant_body
            consumed = wait_until(
                lambda: (_pause_row(oracle, task_id)
                         if _pause_row(oracle, task_id).get("state") == STATE_RESUMED else None), 180)
            assert consumed and consumed.get("pause_id") == pause_id, _pause_row(oracle, task_id)
            assert (consumed.get("grant") or {}).get("grant_id") == grant_id, consumed.get("grant")
            assert (consumed.get("grant") or {}).get("consumed_at"), consumed.get("grant")
            stored = wait_durable_result(oracle, task_id, timeout=600)
            assert stored.get("status") == "completed", stored
            assert "S32_FINAL" in str(stored.get("result") or ""), stored.get("result")
            done_rows = wait_until(lambda: _task_done_rows(oracle, task_id) or None, 60)
            assert done_rows and len(done_rows) == 1 and done_rows[0].get("status") == "completed", done_rows
            assert len(_events_for(oracle, "budget_task_explicitly_resumed", task_id)) == 1
            resumed_bodies = _bodies_carrying(stub, RESUME_NOTICE_MARKER)
            assert resumed_bodies, "no model round carried the resume notice: the task did not continue"
            prior_batches = sum(
                1 for m in resumed_bodies[0].get("messages") or []
                if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"))
            assert prior_batches >= agent_rounds_at_pause, (prior_batches, agent_rounds_at_pause)
            assert _bodies_carrying(stub, FORCED_WRAPUP_MARKER) == []
            assert stub.script_consumed() and _agent_rounds(stub) == S30_SCRIPT_STEPS, stub.kinds()
            ledger_done = _ledger_bucket(budget_root, task_id)
            assert ledger_done["physical_calls"] > ledger_at_pause["physical_calls"], (ledger_at_pause, ledger_done)
            assert ledger_done["accounted_usd"] >= ledger_at_pause["accounted_usd"] + S30_CALL_COST, (
                ledger_at_pause, ledger_done)
            assert int(stored.get("total_rounds") or 0) > ledger_at_pause["physical_calls"], (
                stored.get("total_rounds"), ledger_at_pause)
            assert wait_until(lambda: _pending_carrier(oracle, task_id) is None
                              and _managed_activity(server, task_id) is None, 60)
        finally:
            server.stop()
