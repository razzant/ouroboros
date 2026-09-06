"""Commit-gate wave admission (owner decision 2026-09-05, answer 2 = A).

The scope reviewer — the only constitutionally blocking seat — reserves its
budget FIRST, and the commit-gate wave (scope seats + triad seats) is admitted
all-or-nothing against every fence the ledger enforces at reservation — the
global TOTAL_BUDGET remainder and the task's root fence (the earlier wording
"against the task's current root fence" omitted the global axis; rc.14 audit
point 2) — BEFORE any paid seat is dispatched. A wave that does not fit is a
typed $0 pre-dispatch refusal naming the binding axis and the shortfall, never
a half-dispatched panel (the 4 September paid run: two triad seats held the
money, the third seat and the scope seat were refused mid-wave, and the commit
blocked with ~$5 of the $8 fence never spent).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.reviewer_window import ReviewerWindow
from ouroboros.tools import parallel_review, review, review_admission
from ouroboros.tools import scope_review as scope_mod
from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS

SCOPE_MODEL = "scope/model"
TRIAD_MODELS = ["triad/a", "triad/b"]
BOUNDS = {SCOPE_MODEL: 3.0, "triad/a": 1.0, "triad/b": 1.0}
ROOT = "wave-root"


def _scope_matrix() -> str:
    return json.dumps([
        {"item": item, "verdict": "PASS", "severity": "advisory",
         "reason": "checked the relevant code path and its consumers thoroughly"}
        for item in sorted(SCOPE_REQUIRED_ITEMS)
    ])


class LedgerLLM:
    """Every chat performs ONE real ledger attempt in the bound review scope
    (the exact seam the substrate's api executor drives), so the ledger order
    and the fence are the product's own, not a mock's."""

    def __init__(self, delays=None):
        self.calls = []
        self.delays = dict(delays or {})
        self.lock = threading.Lock()

    def chat(self, **kwargs):
        model = kwargs["model"]
        time.sleep(self.delays.get(model, 0.0))
        reply = {"content": _scope_matrix() if model == SCOPE_MODEL else "[]"}
        ua.execute_physical_attempt(
            ua.AttemptRequest(model=model, provider="test", reservation_usd=BOUNDS[model]),
            lambda: reply,
            extractor=lambda _r: ({"prompt_tokens": 4, "completion_tokens": 2}, 0.01, True),
        )
        with self.lock:
            self.calls.append(model)
        return reply, {"prompt_tokens": 4, "completion_tokens": 2, "cost": 0.01}


@pytest.fixture
def gate(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    ua._reset_task_cache_splits()
    ua._ROOT_ACCOUNTING_TELEMETRY.pop(ROOT, None)
    # The reservation math is the product's; only the price catalog is pinned
    # (no live pricing fetch under test).
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: BOUNDS[request.model])
    monkeypatch.setattr(scope_mod, "_scope_window", lambda *_a, **_k: ReviewerWindow(
        window_tokens=1_000_000, status="confirmed"))
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(parallel_review, "scope_reviewer_slots", lambda *_a, **_k: [
        SimpleNamespace(model=SCOPE_MODEL, slot_id="scope_slot_1", route=ReviewRouteKind.API_CHAT,
                        effort="", session_target="", session_profile="", subagent_id="",
                        retrieves=False),
    ])
    monkeypatch.setattr(review_admission, "prepare_scope_review", lambda *_a, **_k: ({
        "prompt": "SCOPE PACK " * 50, "session_task": "", "repo_dir": tmp_path,
        "scope_model_id": SCOPE_MODEL, "delegated": False, "slot_id": "scope_slot_1",
        "route": ReviewRouteKind.API_CHAT, "slot_effort": "", "session_target": "",
        "session_profile": "", "subagent_id": "", "context_manifest": {}, "stable_prefix_len": 0,
    }, None))
    row_plan = {
        "models": list(TRIAD_MODELS), "routes": [ReviewRouteKind.API_CHAT] * 2,
        "slot_ids": ["slot_1", "slot_2"], "efforts": ["", ""], "subagent_ids": ["", ""],
    }
    monkeypatch.setattr(review, "_prepare_unified_review", lambda *_a, **_k: ({
        "prompt": "TRIAD PACK " * 20, "stable_prefix_len": 0, "models": list(TRIAD_MODELS),
        "routes": [ReviewRouteKind.API_CHAT] * 2, "row_plan": row_plan, "session_task": "",
        "target_repo": tmp_path, "blocking_review": True,
    }, None, False))
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "get_review_enforcement", lambda: "blocking")
    return root


def _ctx(root, tmp_path):
    return SimpleNamespace(
        repo_dir=tmp_path, drive_root=root, task_id=ROOT,
        task_metadata={"root_task_id": ROOT, "budget_drive_root": str(root)},
        pending_events=[], _review_history=[], _review_advisory=[], _scope_review_history={},
        _review_iteration_count=0, _last_review_critical_findings=[], _review_degraded_reasons=[],
    )


def _run(root, tmp_path, monkeypatch, llm, *, fence: float, env_fence: float | None = None):
    """Run the commit gate with the task's usage scope bound on the orchestrator
    thread (``fence``: the root task's bound limit, the one admission uses) while
    the environment carries ``env_fence`` (a hot-reloaded setting; the same
    value unless a test pins the divergence)."""
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", str(fence if env_fence is None else env_fence))
    monkeypatch.setattr(review, "LLMClient", lambda: llm)
    monkeypatch.setattr(scope_mod, "LLMClient", lambda: llm)
    ctx = _ctx(root, tmp_path)
    scope = ua.UsageScope(drive_root=root, task_id=ROOT, root_task_id=ROOT, root_limit_usd=fence)
    with ua.usage_scope(scope):
        outcome = parallel_review.run_parallel_review(ctx, "wave admission commit")
    return ctx, outcome


def _ledger(root):
    path = root / ua.LEDGER_REL
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_fence_that_fits_the_whole_wave_dispatches_every_seat(gate, tmp_path, monkeypatch):
    llm = LedgerLLM()
    ctx, (review_err, scope_result, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert review_err is None, review_err
    assert scope_result.blocked is False and scope_result.status == "responded"
    assert sorted(llm.calls) == sorted([SCOPE_MODEL, *TRIAD_MODELS])
    rows = _ledger(gate)
    assert sorted(row["model"] for row in rows if row["state"] == "settled") == sorted(llm.calls)
    assert not [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]


def test_fence_that_fits_only_the_triad_refuses_the_wave_at_zero_dollars(gate, tmp_path, monkeypatch):
    """Scope $3 + triad $1 + $1 = $5 against a $4 fence: the triad alone would
    have fit (the 4 September failure shape), so the WHOLE wave is refused
    before dispatch — no seat reserved, no ledger row, every seat typed $0."""
    llm = LedgerLLM()
    ctx, (review_err, scope_result, block_reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=4.0)

    assert llm.calls == [] and _ledger(gate) == []
    assert review_err and "commit-gate review wave declined before dispatch ($0 spent)" in review_err
    assert "reservation upper bound $5.000000" in review_err
    assert "per-task budget fence $4.000000" in review_err
    assert "accounted=$0.000000 (of which $0.000000 is reserved by other in-flight attempts)" in review_err
    assert "remaining=$4.000000, shortfall=$1.000000" in review_err
    # The root axis binds; the global axis is disclosed beside it with its own remainder.
    assert "the global budget $100.000000 alone would leave $100.000000" in review_err
    assert "raise the per-task budget (OUROBOROS_PER_TASK_COST_USD)" in review_err
    # Every seat is named with its own bound, scope first.
    assert review_err.index("scope_review:scope_slot_1 scope/model $3.000000") < review_err.index(
        "multi_model_review:slot_1 triad/a $1.000000")
    assert block_reason == "review_wave_budget_insufficient"
    assert [r["status"] for r in ctx._last_triad_raw_results] == ["not_dispatched", "not_dispatched"]
    assert [r["slot_id"] for r in ctx._last_triad_raw_results] == ["slot_1", "slot_2"]
    assert scope_result.blocked is True and scope_result.status == "not_dispatched"
    assert scope_result.block_message.startswith("⚠️ SCOPE_REVIEW_BLOCKED: ")
    assert ctx._last_scope_raw_results[0]["status"] == "not_dispatched"
    assert any("triad_not_dispatched_budget_admission" in r for r in ctx._review_degraded_reasons)
    events = [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    assert len(events) == 1 and events[0]["surface"] == "commit_gate"
    assert events[0]["seats"] == ["scope_review:scope_slot_1", "multi_model_review:slot_1",
                                  "multi_model_review:slot_2"]
    assert events[0]["slot_bounds"] == [3.0, 1.0, 1.0]
    assert events[0]["binding_axis"] == "root" and events[0]["remaining_usd"] == 4.0
    assert (events[0]["global_limit_usd"], events[0]["global_remaining_usd"]) == (100.0, 100.0)


def test_global_budget_that_does_not_fit_refuses_the_wave_before_any_seat_reserves(gate, tmp_path, monkeypatch):
    """rc.14 audit point 2 (verified repro): root fence $10 fits the $5 wave, the
    global TOTAL_BUDGET $100 has $4 left after $96 settled under ANOTHER root.
    Before the fix admission read only the root axis, two seats reserved and
    paid, and the ledger's global check — the FIRST one ``reserve_attempt``
    runs — refused the third mid-wave. Now the wave is refused before any seat
    reserves, and the refusal names the global axis and its own knob, never the
    per-task fence the wave would have fit."""
    with ua.usage_scope(ua.UsageScope(drive_root=gate, task_id="other-root", root_task_id="other-root")):
        other = ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
        ua.mark_dispatched(other)
        ua.settle_attempt(other, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=96.0, cost_final=True)
    llm = LedgerLLM()
    ctx, (review_err, scope_result, block_reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert llm.calls == [] and len(_ledger(gate)) == 3  # the other root's attempt only
    assert review_err and "commit-gate review wave declined before dispatch ($0 spent)" in review_err
    assert "reservation upper bound $5.000000" in review_err
    assert "does not fit the global budget TOTAL_BUDGET $100.000000: accounted=$96.000000 across every task" in review_err
    assert "(of which $0.000000 is reserved by other in-flight attempts)" in review_err
    assert "remaining=$4.000000, shortfall=$1.000000" in review_err
    assert "the per-task budget fence $10.000000 alone would leave $10.000000" in review_err
    assert "raise TOTAL_BUDGET" in review_err and "OUROBOROS_PER_TASK_COST_USD" not in review_err
    assert block_reason == "review_wave_budget_insufficient"
    assert [r["status"] for r in ctx._last_triad_raw_results] == ["not_dispatched", "not_dispatched"]
    assert scope_result.blocked is True and scope_result.status == "not_dispatched"
    events = [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    assert len(events) == 1 and events[0]["binding_axis"] == "global"
    assert (events[0]["remaining_usd"], events[0]["global_remaining_usd"]) == (4.0, 4.0)
    assert (events[0]["global_limit_usd"], events[0]["global_accounted_usd"]) == (100.0, 96.0)
    assert (events[0]["limit_usd"], events[0]["accounted_usd"]) == (10.0, 0.0)


def test_review_wave_admission_binds_on_the_tighter_of_root_and_global_axes(gate, monkeypatch):
    """Direct pins of the two-axis contract: root and global remainders are
    read the way ``reserve_attempt`` will enforce them, the tighter one binds
    and is named, a non-positive TOTAL_BUDGET leaves the global axis unbounded,
    a caller with no root fence and no root rows is still bound by the global
    axis, and an internal exception keeps the fail-open skeleton."""
    wave = dict(root_task_id=ROOT, models=[SCOPE_MODEL, *TRIAD_MODELS], prompt_chars=10, task_id=ROOT)

    root_bound = ua.review_wave_admission(gate, root_limit_usd=4.0, global_limit_usd=10.0, **wave)
    assert (root_bound["fits"], root_bound["binding_axis"], root_bound["estimated_wave_usd"]) == (False, "root", 5.0)
    assert (root_bound["remaining_usd"], root_bound["global_remaining_usd"]) == (4.0, 10.0)
    global_bound = ua.review_wave_admission(gate, root_limit_usd=10.0, global_limit_usd=4.0, **wave)
    assert (global_bound["fits"], global_bound["binding_axis"]) == (False, "global")
    assert (global_bound["remaining_usd"], global_bound["limit_usd"], global_bound["global_limit_usd"]) == (
        4.0, 10.0, 4.0)
    assert (global_bound["global_accounted_usd"], global_bound["global_reserved_usd"]) == (0.0, 0.0)
    admitted = ua.review_wave_admission(gate, root_limit_usd=10.0, global_limit_usd=10.0, **wave)
    assert admitted["fits"] is True and admitted["remaining_usd"] == 10.0

    monkeypatch.setenv("TOTAL_BUDGET", "0")  # no finite global budget: the root axis alone
    root_only = ua.review_wave_admission(gate, root_limit_usd=4.0, **wave)
    assert (root_only["fits"], root_only["binding_axis"], root_only["remaining_usd"]) == (False, "root", 4.0)
    assert (root_only["global_limit_usd"], root_only["global_remaining_usd"]) == (None, None)
    unfenced_unbounded = ua.review_wave_admission(gate, **wave)  # no fence, no rows, no global budget
    assert unfenced_unbounded["fits"] is True and unfenced_unbounded["binding_axis"] is None
    monkeypatch.setenv("TOTAL_BUDGET", "4")
    unfenced = ua.review_wave_admission(gate, **wave)  # no root fence: the global axis still binds
    assert (unfenced["fits"], unfenced["binding_axis"], unfenced["limit_usd"]) == (False, "global", None)
    assert (unfenced["remaining_usd"], unfenced["global_limit_usd"]) == (4.0, 4.0)

    monkeypatch.setattr(ua, "usage_projection", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("ledger")))
    skeleton = ua.review_wave_admission(gate, root_limit_usd=4.0, global_limit_usd=4.0, **wave)
    assert skeleton["fits"] is True and skeleton["estimated_wave_usd"] is None
    assert skeleton["binding_axis"] is None and skeleton["global_remaining_usd"] is None


def test_refusal_names_money_held_by_other_in_flight_attempts(gate, tmp_path, monkeypatch):
    """The fence compares against settled + reserved holds: a refusal must say
    how much of `accounted` is a hold, not a bill."""
    with ua.usage_scope(ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0)):
        held = ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
        ua.mark_dispatched(held)  # an in-flight main-loop send: $1 upper bound held
        settled = ua.reserve_attempt(ua.AttemptRequest(model="triad/b", provider="test", source="main"))
        ua.mark_dispatched(settled)
        ua.settle_attempt(settled, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=2.5, cost_final=True)
    llm = LedgerLLM()
    ctx, (review_err, _scope, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=8.0)

    assert llm.calls == []
    assert "accounted=$3.500000 (of which $1.000000 is reserved by other in-flight attempts)" in review_err
    assert "remaining=$4.500000, shortfall=$0.500000" in review_err
    assert len(_ledger(gate)) == 5  # the two seeded attempts only (reserved/dispatched/settled rows)


def test_scope_reserves_before_the_triad_even_when_it_is_slower(gate, tmp_path, monkeypatch):
    """Scope-first ordering is enforced by the orchestrator, not by luck: the
    scope seat is deliberately the slow one, and the triad is still held until
    the scope reservation is on the ledger."""
    llm = LedgerLLM(delays={SCOPE_MODEL: 0.4})
    ctx, (review_err, scope_result, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert review_err is None and scope_result.status == "responded"
    reserved = [row for row in _ledger(gate) if row["state"] == "reserved"]
    assert [row["category"] for row in reserved][0] == "scope_review_review"
    assert [row["model"] for row in reserved] == [SCOPE_MODEL, *sorted(TRIAD_MODELS)] or \
        [row["model"] for row in reserved] == [SCOPE_MODEL, *reversed(sorted(TRIAD_MODELS))]
    assert reserved[0]["review_slot_id"] == "scope_slot_1"


def test_paid_seats_are_priced_seat_by_seat_scope_first(gate, tmp_path, monkeypatch):
    """The admission prices each seat with ITS pack and output reservation —
    the scope pack beside the triad pack — through the shared gate, one value
    per slot, and asks the gate as ONE wave."""
    seen = {}

    def _gate(ctx, *, surface, models, prompt_chars, max_completion_tokens, extra=None,
              categories="", slot_ids=""):
        seen.update(surface=surface, models=models, prompt_chars=prompt_chars,
                    max_completion_tokens=max_completion_tokens, extra=extra,
                    categories=categories, slot_ids=slot_ids)
        return None

    monkeypatch.setattr("ouroboros.tools.review_helpers.review_wave_budget_gate", _gate)
    from ouroboros.tools.review_multi_model import _review_output_budget

    llm = LedgerLLM()
    _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert seen["surface"] == "commit_gate"
    assert seen["models"] == [SCOPE_MODEL, *TRIAD_MODELS]
    scope_chars, triad_chars = seen["prompt_chars"][0], seen["prompt_chars"][1]
    assert seen["prompt_chars"] == [scope_chars, triad_chars, triad_chars]
    # The scope pack is measured as the exact message pair the substrate sends;
    # the triad pack carries the constitutional preamble + BIBLE ahead of it.
    assert scope_chars > len("SCOPE PACK " * 50) and triad_chars > len("TRIAD PACK " * 20) + 1000
    assert seen["max_completion_tokens"] == [100_000, _review_output_budget(), _review_output_budget()]
    # Every seat names the usage scope its substrate sends under (category + slot),
    # so its bound is read under the seat's own cache split, never the caller's.
    assert seen["categories"] == ["scope_review_review", "multi_model_review_review",
                                  "multi_model_review_review"]
    assert seen["slot_ids"] == ["scope_slot_1", "slot_1", "slot_2"]


def test_review_wave_admission_prices_per_slot_and_discloses_holds(gate, monkeypatch):
    requests = []
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: requests.append(request) or 0.5)
    admission = ua.review_wave_admission(
        gate, root_task_id="fresh-root", models=["prov/a", "prov/b"],
        prompt_chars=[400, 1600], max_completion_tokens=[1000, 2000], task_id="fresh-root",
        root_limit_usd=0.75,
    )
    assert [(r.prompt_tokens_estimate, r.max_completion_tokens, r.task_id) for r in requests] == [
        (100, 1000, "fresh-root"), (400, 2000, "fresh-root"),
    ]
    # No ledger row yet: the caller's bound fence is the limit, not a fail-open.
    assert admission["limit_usd"] == 0.75 and admission["accounted_usd"] == 0.0
    assert admission["estimated_wave_usd"] == 1.0 and admission["fits"] is False
    assert admission["slot_bounds"] == [0.5, 0.5] and admission["reserved_usd"] == 0.0
    # A scalar still broadcasts (the task-level callers are unchanged), and a
    # root with neither a ledger row nor a bound fence keeps failing open.
    requests.clear()
    scalar = ua.review_wave_admission(
        gate, root_task_id="fresh-root", models=["prov/a", "prov/b"], prompt_chars=400,
        root_limit_usd=10.0,
    )
    assert [r.prompt_tokens_estimate for r in requests] == [100, 100]
    assert [r.max_completion_tokens for r in requests] == [65536, 65536]
    assert scalar["fits"] is True and scalar["limit_usd"] == 10.0
    unfenced = ua.review_wave_admission(gate, root_task_id="fresh-root", models=["prov/a"], prompt_chars=400)
    assert unfenced["fits"] is True and unfenced["limit_usd"] is None


def test_wave_without_paid_seats_neither_waits_nor_refuses(gate, tmp_path, monkeypatch):
    """An all-retrieving wave rides the owner's subscription: nothing to price."""
    from ouroboros.tools.review_admission import admit_commit_gate_wave, commit_gate_paid_seats

    session_slot = SimpleNamespace(model="m/session", slot_id="scope_slot_1",
                                   route=ReviewRouteKind.AGENT_SESSION, subagent_id="")
    seats = commit_gate_paid_seats(
        {"prompt": "", "models": ["m/session"], "routes": [ReviewRouteKind.AGENT_SESSION],
         "row_plan": {"models": ["m/session"], "routes": [ReviewRouteKind.AGENT_SESSION]}},
        False, [{"slot": session_slot, "prepared": {"prompt": ""}, "final": None}],
    )
    assert seats == []
    ctx = _ctx(gate, tmp_path)
    assert admit_commit_gate_wave(ctx, seats) is None
    started = time.monotonic()
    parallel_review._await_scope_reservation(ctx, SimpleNamespace(done=lambda: False), seats, started)
    assert time.monotonic() - started < 0.5 and ctx.pending_events == []


# ---------------------------------------------------------------------------
# rc.14 audit findings (astra MAJOR 1-4, fable minors on e27bc3b5)
# ---------------------------------------------------------------------------

def _native_first_send_size(repo, *, surface, session_task, role_hint, output_contract, slot_id, model):
    """The executor's OWN opening send, measured by its own `_open_episode`."""
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    request = ReviewRequest(surface=surface, goal="g", task_id=ROOT, session_root=str(repo),
                            session_task=session_task, policy={"output_contract": output_contract})
    slot = ReviewSlot(slot_id=slot_id, model=model, effort="low", role_hint=role_hint,
                      route=ReviewRouteKind.API_CHAT, subagent_id="critic")
    executor = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id="op"), llm=None)
    return executor._open_episode(str(repo), str(repo))[3]


def test_native_episode_seats_are_paid_and_priced_by_their_first_send(gate, tmp_path):
    """Finding 1: a configured-subagent api row (native episode) reserves every
    round on the ledger exactly like a packet row, so it is a PAID seat priced
    by its first send — measured by the executor's own opening — while an
    agent-session row (subscription, ledger row written at settlement) is not."""
    from ouroboros.reviewer_slot_config import SCOPE_ROLE_HINT
    from ouroboros.tools.review_admission import commit_gate_paid_seats
    from ouroboros.tools.review_multi_model import TRIAD_ROLE_HINT, _review_output_budget
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT

    repo = tmp_path / "subject"
    repo.mkdir()
    native_scope = SimpleNamespace(model=SCOPE_MODEL, slot_id="scope_slot_1", route=ReviewRouteKind.API_CHAT,
                                   subagent_id="critic", retrieves=True)
    session_scope = SimpleNamespace(model="m/session", slot_id="scope_slot_2",
                                    route=ReviewRouteKind.AGENT_SESSION, subagent_id="", retrieves=True)
    scope_rows = [
        {"slot": native_scope, "final": None, "prepared": {
            "prompt": "SCOPE TASK", "session_task": "SCOPE TASK", "repo_dir": repo,
            "scope_model_id": SCOPE_MODEL, "stable_prefix_len": 0}},
        {"slot": session_scope, "final": None, "prepared": {"prompt": "", "session_task": "SCOPE TASK"}},
    ]
    triad = {
        "prompt": "TRIAD PACK", "stable_prefix_len": 0, "session_task": "TRIAD TASK", "target_repo": repo,
        "models": ["triad/a", "triad/native", "m/session"],
        "routes": [ReviewRouteKind.API_CHAT, ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION],
        "row_plan": {"slot_ids": ["slot_1", "slot_2", "slot_3"], "subagent_ids": ["", "critic", ""]},
    }
    seats = commit_gate_paid_seats(triad, False, scope_rows)

    assert [(s["surface"], s["slot_id"], s["model"]) for s in seats] == [
        ("scope_review", "scope_slot_1", SCOPE_MODEL),
        ("multi_model_review", "slot_1", "triad/a"),
        ("multi_model_review", "slot_2", "triad/native"),
    ]
    assert seats[0]["prompt_chars"] == _native_first_send_size(
        repo, surface="scope_review", session_task="SCOPE TASK", role_hint=SCOPE_ROLE_HINT,
        output_contract=scope_mod.SCOPE_RETRIEVING_OUTPUT_CONTRACT, slot_id="scope_slot_1", model=SCOPE_MODEL)
    assert seats[2]["prompt_chars"] == _native_first_send_size(
        repo, surface="multi_model_review", session_task="TRIAD TASK", role_hint=TRIAD_ROLE_HINT,
        output_contract=REVIEW_JSON_ARRAY_CONTRACT, slot_id="slot_2", model="triad/native")
    # A native first send carries instructions, the work-order AND the six tool
    # schemas (the packet row beside it carries the constitutional pack instead).
    from ouroboros.review_native_episode import native_episode_prompt, native_first_send_messages

    work_order_only = json.dumps(native_first_send_messages(native_episode_prompt(
        "multi_model_review", TRIAD_ROLE_HINT, "TRIAD TASK", REVIEW_JSON_ARRAY_CONTRACT, "slot_2")),
        ensure_ascii=False)
    assert seats[2]["prompt_chars"] > len(work_order_only) > 0 and seats[1]["prompt_chars"] > 0
    assert [s["max_completion_tokens"] for s in seats] == [100_000, _review_output_budget(), _review_output_budget()]


def test_scope_seat_is_measured_as_the_cached_block_pair_it_sends(gate, monkeypatch):
    """Fable minor: the scope send wraps the prompt in cached blocks at the
    recorded stable boundary; the admission measures THAT pair, not a plain
    system string, and the triad's user turn is one literal for both."""
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review_admission import commit_gate_paid_seats
    from ouroboros.tools.review_multi_model import TRIAD_USER_TURN

    prefix, dynamic = "STABLE GOVERNANCE " * 20, "DYNAMIC DIFF " * 5
    prompt = prefix + dynamic
    sent = {}

    class _StubLLM:
        def chat(self, **kwargs):
            sent["messages"] = kwargs["messages"]
            return {"content": _scope_matrix()}, {"prompt_tokens": 4, "completion_tokens": 2}

    original = rs.ReviewCoordinator.__init__
    monkeypatch.setattr(rs.ReviewCoordinator, "__init__",
                        lambda self, *, llm=None, drive_root=None, usage_ctx=None:
                        original(self, llm=_StubLLM(), drive_root=drive_root, usage_ctx=usage_ctx))
    ctx = SimpleNamespace(task_id=ROOT, event_queue=None, pending_events=[], drive_root=str(gate))
    token = scope_mod._SCOPE_STABLE_PREFIX_LEN.set(len(prefix))
    try:
        _raw, _usage, err = scope_mod._call_scope_llm(prompt, scope_model=SCOPE_MODEL, ctx=ctx)
    finally:
        scope_mod._SCOPE_STABLE_PREFIX_LEN.reset(token)
    assert err == ""
    expected = scope_mod.scope_api_messages(prompt, len(prefix))
    assert sent["messages"] == expected
    assert isinstance(expected[0]["content"], list) and expected[0]["content"][0].get("cache_control")

    slot = SimpleNamespace(model=SCOPE_MODEL, slot_id="scope_slot_1", route=ReviewRouteKind.API_CHAT, subagent_id="")
    seats = commit_gate_paid_seats(None, True, [{"slot": slot, "final": None, "prepared": {
        "prompt": prompt, "session_task": "", "scope_model_id": SCOPE_MODEL, "stable_prefix_len": len(prefix)}}])
    measured = json.dumps({"messages": sent["messages"]}, ensure_ascii=False, default=str)
    plain = json.dumps({"messages": [{"role": "system", "content": prompt},
                                     {"role": "user", "content": scope_mod.SCOPE_USER_TURN}]}, ensure_ascii=False)
    assert seats[0]["prompt_chars"] == len(measured) != len(plain)

    captured = {}

    def _fanout(ctx, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(review, "_handle_multi_model_review", _fanout)
    review._dispatch_unified_review(
        SimpleNamespace(task_id=ROOT, _review_history=[], _review_advisory=[], pending_events=[]),
        "m", {"blocking_review": True, "prompt": "p", "models": ["triad/a"], "stable_prefix_len": 0,
              "routes": [ReviewRouteKind.API_CHAT], "session_task": "", "target_repo": ".", "row_plan": {}})
    assert captured["content"] == TRIAD_USER_TURN


def test_each_seat_is_priced_under_its_own_cache_split_not_the_callers(tmp_path, monkeypatch):
    """Finding 2: the observed cache split is keyed by the SENDING scope
    (category + review slot). A warm split of the caller's own transcript must
    not price a reviewer seat's cold prefix — the seat's bound is the full
    write until the seat's OWN scope has observed a split. Real
    ``_reservation_cost``; only the price catalog is pinned."""
    from dataclasses import replace

    from ouroboros import pricing as pricing_mod
    from ouroboros.pricing import infer_provider_from_model

    class _P(tuple):
        tiers = ()

    model = "anthropic/claude-fable-5"
    monkeypatch.setattr(pricing_mod, "get_pricing", lambda **k: {model: _P((10.0, 1.0, 12.5, 50.0))})
    ua._reset_task_cache_splits()
    provider = infer_provider_from_model(model)
    caller = ua.UsageScope(drive_root=tmp_path, task_id=ROOT, root_task_id=ROOT, root_limit_usd=1000.0)
    seat_scope = replace(caller, category="scope_review_review", review_slot_id="scope_slot_1")
    kwargs = dict(root_task_id=ROOT, models=[model], prompt_chars=400_000, max_completion_tokens=1000,
                  task_id=ROOT, root_limit_usd=1000.0)
    with ua.usage_scope(caller):
        # The caller's transcript is warm (90% of the prompt read from cache).
        ua.stash_task_cache_split(ROOT, model, 90_000, provider=provider, ttl_seconds=300.0)
        callers_own = ua.review_wave_admission(tmp_path, **kwargs)["slot_bounds"][0]
        cold_seat = ua.review_wave_admission(
            tmp_path, categories="scope_review_review", slot_ids="scope_slot_1", **kwargs)["slot_bounds"][0]
        with ua.usage_scope(seat_scope):
            expected_cold = ua._reservation_cost(ua.AttemptRequest(
                model=model, provider=provider, prompt_tokens_estimate=100_000,
                max_completion_tokens=1000, task_id=ROOT))
            ua.stash_task_cache_split(ROOT, model, 90_000, provider=provider, ttl_seconds=300.0)
        warm_seat = ua.review_wave_admission(
            tmp_path, categories="scope_review_review", slot_ids="scope_slot_1", **kwargs)["slot_bounds"][0]
    assert cold_seat == pytest.approx(expected_cold)
    assert cold_seat > callers_own  # the caller's warm split never priced the seat
    assert warm_seat == pytest.approx(callers_own)  # the seat's OWN observed split does


def test_current_root_fence_governs_admission_over_the_ledgers_historical_minimum(gate, monkeypatch):
    """Finding 3: ``reserve_attempt`` enforces the CURRENT scope fence; the
    ledger projection carries the minimum of historical row limits. Admission
    must compare against the fence the reservation will use, whether it was
    raised or lowered since the earlier rows — the projection serves only a
    caller that binds no fence of its own."""
    with ua.usage_scope(ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0)):
        held = ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=3.0, cost_final=True)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 4.5)
    kwargs = dict(root_task_id=ROOT, models=["triad/a"], prompt_chars=10, task_id=ROOT)

    raised = ua.review_wave_admission(gate, root_limit_usd=50.0, **kwargs)
    assert (raised["limit_usd"], raised["accounted_usd"], raised["remaining_usd"]) == (50.0, 3.0, 47.0)
    assert raised["fits"] is True
    lowered = ua.review_wave_admission(gate, root_limit_usd=6.0, **kwargs)
    assert (lowered["limit_usd"], lowered["remaining_usd"], lowered["fits"]) == (6.0, 3.0, False)
    unfenced = ua.review_wave_admission(gate, **kwargs)  # no fence of its own: the ledger's $8 row
    assert (unfenced["limit_usd"], unfenced["remaining_usd"], unfenced["fits"]) == (8.0, 5.0, True)


def _reservation_ids() -> frozenset:
    return frozenset(r["attempt_id"] for r in (ua.last_root_accounting(ROOT) or {}).get("reservations") or [])


def test_scope_first_hold_observes_only_the_scope_seats_own_reservation(gate, tmp_path, monkeypatch):
    """Finding 4: the hold releases on the scope seat's OWN appended reservation
    (category + slot identity after the wave started) — a refresh, a
    settlement, a refused reservation or another seat's reservation never
    releases it — and a hold that ends without it is a typed event."""
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "NESTED_SETTLEMENT_MARGIN_SEC", 0.3)
    seats = [{"surface": "scope_review", "slot_id": "scope_slot_1", "model": SCOPE_MODEL,
              "prompt_chars": 10, "max_completion_tokens": 10}]
    never_done = SimpleNamespace(done=lambda: False)
    base = ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0)
    scope_seat = ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0,
                               category="scope_review_review", review_slot_id="scope_slot_1")
    triad_seat = ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0,
                               category="multi_model_review_review", review_slot_id="slot_1")

    with ua.usage_scope(base):
        earlier = ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
        ua.mark_dispatched(earlier)
    started = time.monotonic()
    known = _reservation_ids()   # the wave's baseline: everything reserved before it
    # Non-scope root telemetry updates after the wave started: none may release the hold.
    ua.refresh_root_accounting(gate, ROOT)
    with ua.usage_scope(base):
        ua.settle_attempt(earlier, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=0.5, cost_final=True)
    with ua.usage_scope(ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=0.75)):
        with pytest.raises(ua.BudgetExceeded):  # refused: pre-fence refresh only, no identity
            ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
    with ua.usage_scope(triad_seat):
        ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test"))
    assert not any(r["category"] == "scope_review_review"
                   for r in ua.last_root_accounting(ROOT)["reservations"])
    with ua.usage_scope(base):
        ctx = _ctx(gate, tmp_path)
        parallel_review._await_scope_reservation(ctx, never_done, seats, started, known_ids=known)
    assert time.monotonic() - started >= 0.3
    events = [e for e in ctx.pending_events if e.get("type") == "review_scope_lead_unobserved"]
    assert len(events) == 1 and events[0]["scope_slot_ids"] == ["scope_slot_1"]
    assert events[0]["scope_seat_done"] is False and events[0]["root_task_id"] == ROOT

    # The scope seat's own reservation, appended after the start, releases at once.
    started = time.monotonic()
    known = _reservation_ids()
    with ua.usage_scope(scope_seat):
        own = ua.reserve_attempt(ua.AttemptRequest(model=SCOPE_MODEL, provider="test"))
    identities = ua.last_root_accounting(ROOT)["reservations"]
    assert any(r["attempt_id"] == own.attempt_id and r["review_slot_id"] == "scope_slot_1" for r in identities)
    with ua.usage_scope(base):
        ctx = _ctx(gate, tmp_path)
        hold_started = time.monotonic()   # the pin times the HOLD, not the ledger append before it
        parallel_review._await_scope_reservation(ctx, never_done, seats, started, known_ids=known)
    assert time.monotonic() - hold_started < 0.25 and ctx.pending_events == []

    # A scope seat that finished without ever reserving (e.g. refused) ends the hold typed;
    # the earlier own reservation is in this wave's baseline, so it cannot release it —
    # an identity check, not a clock one (Windows' monotonic tick would merge the two).
    started = time.monotonic()
    known = _reservation_ids()
    with ua.usage_scope(base):
        ctx = _ctx(gate, tmp_path)
        parallel_review._await_scope_reservation(ctx, SimpleNamespace(done=lambda: True), seats, started, known_ids=known)
    events = [e for e in ctx.pending_events if e.get("type") == "review_scope_lead_unobserved"]
    assert len(events) == 1 and events[0]["scope_seat_done"] is True


def _seed_settled(root, *, fence: float, cost: float) -> None:
    """One settled main-loop row under ``fence`` (a ledger history row)."""
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id=ROOT, root_task_id=ROOT, root_limit_usd=fence)):
        row = ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
        ua.mark_dispatched(row)
        ua.settle_attempt(row, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=cost, cost_final=True)


def test_seats_reserve_against_exactly_the_fence_the_wave_was_admitted_with(gate, tmp_path, monkeypatch):
    """rc.14 audit (astra MAJOR on G1): admission prefers the caller scope's
    bound fence, so the seats must reserve against THAT fence — not against a
    setting hot-reloaded mid-turn that the reviewer threads would re-read from
    the environment once the executor transitions dropped the usage scope.
    Bound $50, environment $8, a $4 history row at $8: the $5 wave is admitted
    against $50 AND every seat's own ``reserve_attempt`` binds $50 — dispatched
    whole, never a partial paid wave."""
    _seed_settled(gate, fence=8.0, cost=4.0)
    llm = LedgerLLM()
    ctx, (review_err, scope_result, _reason, _adv) = _run(
        gate, tmp_path, monkeypatch, llm, fence=50.0, env_fence=8.0)

    assert review_err is None, review_err
    assert scope_result.blocked is False and scope_result.status == "responded"
    assert sorted(llm.calls) == sorted([SCOPE_MODEL, *TRIAD_MODELS])
    seats = [row for row in _ledger(gate) if row["state"] == "reserved" and row["source"] != "main"]
    assert sorted(row["model"] for row in seats) == sorted([SCOPE_MODEL, *TRIAD_MODELS])
    assert {row["root_limit_usd"] for row in seats} == {50.0}
    assert not [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    assert not [r for r in ctx._last_triad_raw_results if r.get("status") != "ok"] or all(
        "BudgetExceeded" not in str(r.get("error") or "") for r in ctx._last_triad_raw_results)


def test_bound_fence_that_does_not_fit_refuses_even_when_the_environment_is_roomier(gate, tmp_path, monkeypatch):
    """The converse: bound $8 (admission's fence), environment $50, a $4 history
    row: the $5 wave does not fit the bound fence and nothing is dispatched —
    the roomier setting never buys a seat the admitting fence refused."""
    _seed_settled(gate, fence=8.0, cost=4.0)
    llm = LedgerLLM()
    ctx, (review_err, scope_result, block_reason, _adv) = _run(
        gate, tmp_path, monkeypatch, llm, fence=8.0, env_fence=50.0)

    assert llm.calls == [] and len(_ledger(gate)) == 3  # the seeded row's reserved/dispatched/settled only
    assert review_err and "per-task budget fence $8.000000" in review_err
    assert "remaining=$4.000000, shortfall=$1.000000" in review_err
    assert block_reason == "review_wave_budget_insufficient"
    assert scope_result.status == "not_dispatched"


def test_scope_first_hold_ignores_a_sibling_tasks_same_named_scope_slot(gate, tmp_path, monkeypatch):
    """rc.14 audit (fable MINOR 1): two tasks under one root each run their own
    commit gate with the default ``scope_slot_1``; the sibling's reservation
    carries the right category, slot and time but ITS task id, and must not
    release this task's hold — only this task's own scope seat does."""
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "NESTED_SETTLEMENT_MARGIN_SEC", 0.3)
    seats = [{"surface": "scope_review", "slot_id": "scope_slot_1", "model": SCOPE_MODEL,
              "prompt_chars": 10, "max_completion_tokens": 10}]
    never_done = SimpleNamespace(done=lambda: False)
    base = ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0)
    sibling_seat = ua.UsageScope(drive_root=gate, task_id="sibling-task", root_task_id=ROOT, root_limit_usd=8.0,
                                 category="scope_review_review", review_slot_id="scope_slot_1")
    own_seat = ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0,
                             category="scope_review_review", review_slot_id="scope_slot_1")

    started = time.monotonic()
    with ua.usage_scope(sibling_seat):
        sibling = ua.reserve_attempt(ua.AttemptRequest(model=SCOPE_MODEL, provider="test"))
    identities = ua.last_root_accounting(ROOT)["reservations"]
    assert any(r["attempt_id"] == sibling.attempt_id and r["task_id"] == "sibling-task" for r in identities)
    with ua.usage_scope(base):
        ctx = _ctx(gate, tmp_path)
        parallel_review._await_scope_reservation(ctx, never_done, seats, started)
    assert time.monotonic() - started >= 0.3
    events = [e for e in ctx.pending_events if e.get("type") == "review_scope_lead_unobserved"]
    assert len(events) == 1 and events[0]["task_id"] == ROOT

    started = time.monotonic()
    with ua.usage_scope(own_seat):
        ua.reserve_attempt(ua.AttemptRequest(model=SCOPE_MODEL, provider="test"))
    with ua.usage_scope(base):
        ctx = _ctx(gate, tmp_path)
        hold_started = time.monotonic()   # the pin times the HOLD, not the ledger append before it
        parallel_review._await_scope_reservation(ctx, never_done, seats, started)
    assert time.monotonic() - hold_started < 0.25 and ctx.pending_events == []


def test_admission_that_raises_fails_open_loudly_and_typed(gate, tmp_path, monkeypatch, caplog):
    """rc.14 audit (fable MINOR 2): an exception inside the money admission keeps
    the owner's fail-open (the wave dispatches, unadmitted and without the
    scope-first hold) but is a warning plus ONE typed
    ``review_wave_admission_unavailable`` event naming the error — never a
    debug line indistinguishable from an admitted wave."""
    import logging

    def _boom(*_a, **_k):
        raise RuntimeError("no inspection tool schemas are projectable")

    monkeypatch.setattr(review_admission, "commit_gate_paid_seats", _boom)
    llm = LedgerLLM()
    with caplog.at_level(logging.WARNING, logger="ouroboros.tools.parallel_review"):
        ctx, (review_err, scope_result, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert review_err is None and scope_result.status == "responded"
    assert sorted(llm.calls) == sorted([SCOPE_MODEL, *TRIAD_MODELS])
    events = [e for e in ctx.pending_events if e.get("type") == "review_wave_admission_unavailable"]
    assert len(events) == 1
    assert events[0]["surface"] == "commit_gate" and events[0]["task_id"] == ROOT
    assert events[0]["error"] == "RuntimeError: no inspection tool schemas are projectable"
    assert not [e for e in ctx.pending_events if e.get("type") in {
        "review_wave_budget_insufficient", "review_scope_lead_unobserved"}]
    assert any("commit-gate wave admission unavailable (RuntimeError" in r.getMessage()
               and r.levelno == logging.WARNING for r in caplog.records)
