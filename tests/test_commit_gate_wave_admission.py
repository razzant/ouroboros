"""Commit-gate wave admission (owner decision 2026-09-05, rebased on #1487).

The commit-gate wave — ONE wave, every assigned seat (PR-3 B: a retrieving seat
answers both parts of one two-part brief, a packet seat the change) — is
admitted while KNOWN spend is below every fence the ledger enforces at
reservation — the global TOTAL_BUDGET, the task's current root and its original
billing group — the reservation's own rule (owner Q4-A, 2026-10-03). The seats'
summed reservation bounds are disclosed, never an earlier refusal: open holds of
other in-flight attempts are exposure, not spending. A wave declined at
admission (known spend already at a fence) is a typed $0 pre-dispatch refusal
naming the binding axis. A fence reached mid-wave refuses the remaining seats at
their own reservation, with truthful custody of what was sent (partial dispatch
is accepted overshoot behaviour, not hidden). There is no scope role and no
scope-first hold: every paid seat is priced before the first paid call, as one
wave.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.tools import parallel_review, review, review_admission
from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT, REVIEW_TWO_PART_OBJECT_CONTRACT
from tests._usage_store_testing import attempt_rows_in_start_order, ledger_rows

COUPLING_MODEL = "coupling/model"   # the retrieving seat, asked both parts (priced by its first send)
PACKET_MODELS = ["triad/a", "triad/b"]
BOUNDS = {COUPLING_MODEL: 3.0, "triad/a": 1.0, "triad/b": 1.0}
ROOT = "wave-root"
BRIEF = "TWO-PART BRIEF " * 50


def _coupling_matrix() -> list:
    return [{"item": item, "verdict": "PASS", "severity": "advisory",
             "reason": "checked the relevant code path and its consumers thoroughly"}
            for item in sorted(SCOPE_REQUIRED_ITEMS)]


def _two_part_answer() -> str:
    return json.dumps({"change": [], "change_clean": True, "coupling": _coupling_matrix()})


class LedgerLLM:
    """Every chat performs ONE real ledger attempt in the bound review scope
    (the exact seam the substrate's api executor drives), so the ledger order
    and the fence are the product's own, not a mock's. The retrieving seat
    answers contract B without a tool round; the packet seats answer ``[]``."""

    def __init__(self, delays=None):
        self.calls = []
        self.delays = dict(delays or {})
        self.lock = threading.Lock()

    def chat(self, **kwargs):
        model = kwargs["model"]
        time.sleep(self.delays.get(model, 0.0))
        reply = {"content": _two_part_answer() if model == COUPLING_MODEL else "[]"}
        ua.execute_physical_attempt(
            ua.AttemptRequest(model=model, provider="test", reservation_usd=BOUNDS[model]),
            lambda: reply,
            extractor=lambda _r: ({"prompt_tokens": 4, "completion_tokens": 2}, 0.01, True),
        )
        with self.lock:
            self.calls.append(model)
        return reply, {"prompt_tokens": 4, "completion_tokens": 2, "cost": 0.01}


class PricedLedgerLLM(LedgerLLM):
    """``LedgerLLM`` whose seats settle at chosen real prices. With ``first`` set,
    every other seat sends only after that seat has settled, so the order a test
    needs never depends on thread scheduling (the wave has no scope-first hold)."""

    def __init__(self, costs, first=None):
        super().__init__()
        self.costs, self.first, self.first_settled = dict(costs), first, threading.Event()

    def chat(self, **kwargs):
        model = kwargs["model"]
        if self.first and model != self.first:
            assert self.first_settled.wait(30), f"{self.first} never settled"
        reply = {"content": _two_part_answer() if model == COUPLING_MODEL else "[]"}
        price = self.costs.get(model, 0.01)
        try:
            ua.execute_physical_attempt(
                ua.AttemptRequest(model=model, provider="test", reservation_usd=BOUNDS[model]),
                lambda: reply,
                extractor=lambda _r: ({"prompt_tokens": 4, "completion_tokens": 2}, price, True),
            )
        finally:
            if model == self.first:
                self.first_settled.set()
        with self.lock:
            self.calls.append(model)
        return reply, {"prompt_tokens": 4, "completion_tokens": 2, "cost": price}


def _prepared(tmp_path) -> dict:
    """The one wave as ``_prepare_unified_review`` hands it to dispatch: one seat
    list with ``parts``, the retrieving seat carrying ITS brief and answer policy."""
    row_plan = {
        "models": [COUPLING_MODEL, *PACKET_MODELS], "routes": [ReviewRouteKind.API_CHAT] * 3,
        "slot_ids": ["slot_native", "slot_1", "slot_2"], "efforts": ["", "", ""], "subagent_ids": ["", "", ""],
        "session_targets": ["", "", ""], "session_profiles": ["", "", ""], "use_local": [None] * 3,
        "retrieves": [True, False, False], "parts": [("change", "coupling"), ("change",), ("change",)],
        "session_tasks": [BRIEF, "", ""],
        "session_policies": [{"output_contract": REVIEW_TWO_PART_OBJECT_CONTRACT}, None, None],
        "brief_shas": ["brief-sha", "", ""],
    }
    return {
        "prompt": "PACKET " * 20, "stable_prefix_len": 0, "models": list(row_plan["models"]),
        "routes": list(row_plan["routes"]), "row_plan": row_plan, "session_task": "",
        "target_repo": tmp_path, "blocking_review": True, "brief_texts": {"brief-sha": BRIEF},
    }


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
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(review, "_prepare_unified_review", lambda *_a, **_k: (_prepared(tmp_path), None, False))
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "get_review_enforcement", lambda: "blocking")
    return root


def _ctx(root, tmp_path):
    return SimpleNamespace(
        repo_dir=tmp_path, drive_root=root, task_id=ROOT,
        task_metadata={"root_task_id": ROOT, "budget_drive_root": str(root)},
        pending_events=[], _review_history=[], _review_advisory=[], _coupling_review_history={},
        _review_iteration_count=0, _last_review_critical_findings=[], _review_degraded_reasons=[],
    )


def _run(root, tmp_path, monkeypatch, llm, *, fence: float, env_fence: float | None = None):
    """Run the commit gate with the task's usage scope bound on the orchestrator
    thread (``fence``: the root task's bound limit, the one admission uses) while
    the environment carries ``env_fence`` (a hot-reloaded setting; the same
    value unless a test pins the divergence)."""
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", str(fence if env_fence is None else env_fence))
    monkeypatch.setattr(review, "LLMClient", lambda: llm)
    ctx = _ctx(root, tmp_path)
    scope = ua.UsageScope(drive_root=root, task_id=ROOT, root_task_id=ROOT, root_limit_usd=fence)
    with ua.usage_scope(scope):
        outcome = parallel_review.run_parallel_review(ctx, "wave admission commit")
    return ctx, outcome


def _ledger(root):
    return ledger_rows(root)


def test_fence_that_fits_the_whole_wave_dispatches_every_seat(gate, tmp_path, monkeypatch):
    llm = LedgerLLM()
    ctx, (review_err, coupling, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert review_err is None, review_err
    assert coupling.blocked is False and coupling.status == "responded" and coupling.verdict == "PASS"
    assert sorted(llm.calls) == sorted([COUPLING_MODEL, *PACKET_MODELS])
    rows = _ledger(gate)
    assert sorted(row["model"] for row in rows if row["state"] == "settled") == sorted(llm.calls)
    assert not [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    # One commit cycle, one wave: every seat under the same wave id (#1544).
    reserved = attempt_rows_in_start_order(gate)
    assert len({row["review_wave_id"] for row in reserved}) == 1 and reserved[0]["review_wave_id"]
    assert ctx._last_review_verdict["aggregate"] == "PASS"


def test_a_wave_larger_than_the_known_remainder_is_admitted_whole(gate, tmp_path, monkeypatch):
    """Retrieving seat $3 + packet $1 + $1 = $5 of worst case against a $4 fence with
    $0 known: the prospective sum is information, never an earlier refusal (#1487) —
    every seat is dispatched and each settles at its real price."""
    llm = LedgerLLM()
    ctx, (review_err, coupling, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=4.0)

    assert review_err is None, review_err
    assert coupling.blocked is False and coupling.status == "responded"
    assert sorted(llm.calls) == sorted([COUPLING_MODEL, *PACKET_MODELS])
    assert sorted(row["model"] for row in _ledger(gate) if row["state"] == "settled") == sorted(llm.calls)
    assert not [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]


def test_known_spend_at_the_fence_declines_the_wave_at_zero_dollars(gate, tmp_path, monkeypatch):
    """Known spend already at the $4 fence: the WHOLE wave is declined before dispatch —
    no seat reserved, no new ledger row, every seat typed $0."""
    _seed_settled(gate, fence=4.0, cost=4.0)
    before = _ledger(gate)
    llm = LedgerLLM()
    ctx, (review_err, coupling, block_reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=4.0)

    assert llm.calls == [] and _ledger(gate) == before
    assert review_err and "commit-gate review wave declined before dispatch ($0 spent)" in review_err
    assert "Known spend has reached the per-task budget fence $4.000000" in review_err
    assert "known spend=$4.000000 (plus $0.000000 of open holds, not counted)" in review_err
    assert "remaining=$0.000000" in review_err
    assert "reservation upper bound would have been $5.000000" in review_err
    # The root axis binds; the global axis is disclosed beside it with its own remainder.
    assert "the global budget $100.000000 alone would leave $96.000000" in review_err
    assert "raise the per-task budget (OUROBOROS_PER_TASK_COST_USD)" in review_err
    # Every seat is named with its own bound, in wave order.
    assert review_err.index("multi_model_review:slot_native coupling/model $3.000000") < review_err.index(
        "multi_model_review:slot_1 triad/a $1.000000")
    assert block_reason == "review_wave_budget_insufficient"
    assert [r["status"] for r in ctx._last_triad_raw_results] == ["not_dispatched"] * 3
    assert [r["slot_id"] for r in ctx._last_triad_raw_results] == ["slot_native", "slot_1", "slot_2"]
    assert coupling.status == "not_dispatched" and coupling.verdict == "not_performed"
    assert any("review_not_dispatched_budget_admission" in r for r in ctx._review_degraded_reasons)
    events = [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    assert len(events) == 1 and events[0]["surface"] == "commit_gate"
    assert events[0]["seats"] == ["multi_model_review:slot_native", "multi_model_review:slot_1",
                                  "multi_model_review:slot_2"]
    assert events[0]["slot_bounds"] == [3.0, 1.0, 1.0]
    assert events[0]["binding_axis"] == "root" and events[0]["remaining_usd"] == 0.0
    assert (events[0]["global_limit_usd"], events[0]["global_remaining_usd"]) == (100.0, 96.0)
    # The record carries the refusal text: NOT_DISPATCHED is the first aggregate branch.
    assert "reservation upper bound would have been $5.000000" in ctx._last_review_structured["wave_refusal"]


def test_global_known_spend_at_the_limit_declines_the_wave_before_any_seat_reserves(gate, tmp_path, monkeypatch):
    """rc.14 audit point 2, on the known-spend rule: the root fence $10 has room, the
    global TOTAL_BUDGET $100 is fully spent under ANOTHER root. The wave is declined
    before any seat reserves, and the refusal names the global axis and its own knob,
    never the per-task fence."""
    with ua.usage_scope(ua.UsageScope(drive_root=gate, task_id="other-root", root_task_id="other-root")):
        other = ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
        ua.mark_dispatched(other)
        ua.settle_attempt(other, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=100.0, cost_final=True)
    llm = LedgerLLM()
    ctx, (review_err, coupling, block_reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert llm.calls == [] and len(_ledger(gate)) == 1  # the other root's attempt only (one current row)
    assert review_err and "commit-gate review wave declined before dispatch ($0 spent)" in review_err
    assert "Known spend has reached the global budget TOTAL_BUDGET $100.000000: known spend=$100.000000 across every task" in review_err
    assert "(plus $0.000000 of open holds, not counted)" in review_err
    assert "remaining=$0.000000" in review_err
    assert "the per-task budget fence $10.000000 alone would leave $10.000000" in review_err
    assert "raise TOTAL_BUDGET" in review_err and "OUROBOROS_PER_TASK_COST_USD" not in review_err
    assert block_reason == "review_wave_budget_insufficient"
    assert [r["status"] for r in ctx._last_triad_raw_results] == ["not_dispatched"] * 3
    assert coupling.status == "not_dispatched"
    events = [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    assert len(events) == 1 and events[0]["binding_axis"] == "global"
    assert (events[0]["remaining_usd"], events[0]["global_remaining_usd"]) == (0.0, 0.0)
    assert (events[0]["global_limit_usd"], events[0]["global_known_usd"]) == (100.0, 100.0)
    assert (events[0]["limit_usd"], events[0]["known_usd"]) == (10.0, 0.0)


def test_a_fence_reached_mid_wave_refuses_the_remaining_seats_with_custody(gate, tmp_path, monkeypatch):
    """The wave is admitted at $0 known; the retrieving seat settles at the full $4
    fence before the packet seats send, so each packet seat is refused at its OWN
    reservation: nothing further is sent, and the outcome says what happened."""
    llm = PricedLedgerLLM(costs={COUPLING_MODEL: 4.0}, first=COUPLING_MODEL)
    ctx, (_review_err, coupling, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=4.0)

    assert llm.calls == [COUPLING_MODEL]
    assert coupling.status == "responded"
    rows = _ledger(gate)
    assert [(row["model"], row["state"]) for row in rows] == [(COUPLING_MODEL, "settled")]
    assert ua.usage_projection(gate, root_task_id=ROOT)["settled_usd"] == pytest.approx(4.0)
    packet = [r for r in ctx._last_triad_raw_results if r.get("slot_id") != "slot_native"]
    assert len(packet) == 2 and all(r.get("status") != "ok" for r in packet), packet
    assert all("budget" in json.dumps(r, default=str).lower() for r in packet), packet


def test_review_wave_admission_binds_on_the_tighter_of_root_and_global_axes(gate, monkeypatch):
    """Direct pins of the two-axis contract: root and global remainders are
    read the way ``reserve_attempt`` will enforce them, the tighter one binds
    and is named, a non-positive TOTAL_BUDGET leaves the global axis unbounded,
    a caller with no root fence and no root rows is still bound by the global
    axis, and an internal exception keeps the fail-open skeleton."""
    wave = dict(root_task_id=ROOT, models=[COUPLING_MODEL, *PACKET_MODELS], prompt_chars=10, task_id=ROOT)

    # The $5 estimate exceeds both remainders, and the wave still fits: known room decides.
    root_bound = ua.review_wave_admission(gate, root_limit_usd=4.0, global_limit_usd=10.0, **wave)
    assert (root_bound["fits"], root_bound["binding_axis"], root_bound["estimated_wave_usd"]) == (True, "root", 5.0)
    assert (root_bound["remaining_usd"], root_bound["global_remaining_usd"]) == (4.0, 10.0)
    global_bound = ua.review_wave_admission(gate, root_limit_usd=10.0, global_limit_usd=4.0, **wave)
    assert (global_bound["fits"], global_bound["binding_axis"]) == (True, "global")
    assert (global_bound["remaining_usd"], global_bound["limit_usd"], global_bound["global_limit_usd"]) == (
        4.0, 10.0, 4.0)
    assert (global_bound["global_known_usd"], global_bound["global_reserved_usd"]) == (0.0, 0.0)
    no_room = ua.review_wave_admission(gate, root_limit_usd=0.0, global_limit_usd=10.0, **wave)
    assert (no_room["fits"], no_room["binding_axis"], no_room["remaining_usd"]) == (False, "root", 0.0)

    monkeypatch.setenv("TOTAL_BUDGET", "0")  # no finite global budget: the root axis alone
    root_only = ua.review_wave_admission(gate, root_limit_usd=4.0, **wave)
    assert (root_only["fits"], root_only["binding_axis"], root_only["remaining_usd"]) == (True, "root", 4.0)
    assert (root_only["global_limit_usd"], root_only["global_remaining_usd"]) == (None, None)
    unfenced_unbounded = ua.review_wave_admission(gate, **wave)  # no fence, no rows, no global budget
    assert unfenced_unbounded["fits"] is True and unfenced_unbounded["binding_axis"] is None
    monkeypatch.setenv("TOTAL_BUDGET", "4")
    unfenced = ua.review_wave_admission(gate, **wave)  # no root fence: the global axis still binds
    assert (unfenced["fits"], unfenced["binding_axis"], unfenced["limit_usd"]) == (True, "global", None)
    assert (unfenced["remaining_usd"], unfenced["global_limit_usd"]) == (4.0, 4.0)

    monkeypatch.setattr(ua, "usage_projection", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("ledger")))
    skeleton = ua.review_wave_admission(gate, root_limit_usd=4.0, global_limit_usd=4.0, **wave)
    assert skeleton["fits"] is True and skeleton["estimated_wave_usd"] is None
    assert skeleton["binding_axis"] is None and skeleton["global_remaining_usd"] is None


def test_open_holds_alone_never_decline_the_wave(gate, tmp_path, monkeypatch):
    """A $1 in-flight main-loop send and $2.50 known under an $8 fence: the hold is
    exposure, not a bill, so the wave is admitted and dispatched (#1487)."""
    with ua.usage_scope(ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0)):
        held = ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
        ua.mark_dispatched(held)  # an in-flight main-loop send: $1 upper bound held
        settled = ua.reserve_attempt(ua.AttemptRequest(model="triad/b", provider="test", source="main"))
        ua.mark_dispatched(settled)
        ua.settle_attempt(settled, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=2.5, cost_final=True)
    llm = LedgerLLM()
    _ctx, (review_err, coupling, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=8.0)

    assert review_err is None and coupling.status == "responded"
    assert sorted(llm.calls) == sorted([COUPLING_MODEL, *PACKET_MODELS])


def test_refusal_names_open_holds_beside_known_spend(gate, tmp_path, monkeypatch):
    """Known spend at the $8 fence with a $1 hold still in flight: the refusal says how
    much is known spend and how much is an uncounted hold, never one blended number."""
    with ua.usage_scope(ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT, root_limit_usd=8.0)):
        held = ua.reserve_attempt(ua.AttemptRequest(model="triad/a", provider="test", source="main"))
        ua.mark_dispatched(held)
        settled = ua.reserve_attempt(ua.AttemptRequest(model="triad/b", provider="test", source="main"))
        ua.mark_dispatched(settled)
        ua.settle_attempt(settled, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=8.0, cost_final=True)
    llm = LedgerLLM()
    _ctx, (review_err, _coupling, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=8.0)

    assert llm.calls == []
    assert "known spend=$8.000000 (plus $1.000000 of open holds, not counted)" in review_err
    assert "remaining=$0.000000" in review_err
    assert len(_ledger(gate)) == 2  # the two seeded attempts only (one current row each)


def test_paid_seats_are_priced_seat_by_seat_before_the_first_paid_call(gate, tmp_path, monkeypatch):
    """The admission prices each paid seat with ITS send and output reservation
    — the retrieving seat's first send (its own two-part brief) beside the packet
    — through the shared gate, one value per slot, and asks the gate as ONE wave
    before any seat is called."""
    seen = {}
    order = []

    def _gate(ctx, *, surface, models, prompt_chars, max_completion_tokens, extra=None,
              categories="", slot_ids=""):
        order.append("admission")
        seen.update(surface=surface, models=models, prompt_chars=prompt_chars,
                    max_completion_tokens=max_completion_tokens, extra=extra,
                    categories=categories, slot_ids=slot_ids)
        return None

    monkeypatch.setattr("ouroboros.tools.review_helpers.review_wave_budget_gate", _gate)
    from ouroboros.tools.review_multi_model import _review_output_budget

    class _Ordered(LedgerLLM):
        def chat(self, **kwargs):
            order.append("paid_call")
            return super().chat(**kwargs)

    llm = _Ordered()
    _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert order and order[0] == "admission" and order.count("admission") == 1
    assert seen["surface"] == "commit_gate"
    assert seen["models"] == [COUPLING_MODEL, *PACKET_MODELS]
    native_chars, packet_chars = seen["prompt_chars"][0], seen["prompt_chars"][1]
    assert seen["prompt_chars"] == [native_chars, packet_chars, packet_chars]
    # The retrieving seat is measured as the executor's own first send (its brief,
    # instructions and tool schemas); the packet carries the constitutional
    # preamble + BIBLE ahead of the assembled change.
    assert native_chars > len(BRIEF) and packet_chars > len("PACKET " * 20) + 1000
    assert seen["max_completion_tokens"] == [_review_output_budget()] * 3
    # Every seat names the usage scope its substrate sends under (category + slot),
    # so its bound is read under the seat's own cache split, never the caller's.
    assert seen["categories"] == ["multi_model_review_review"] * 3
    assert seen["slot_ids"] == ["slot_native", "slot_1", "slot_2"]


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
    # No ledger row yet: the caller's bound fence is the limit, not a fail-open. The
    # $1.00 estimate above the $0.75 room is disclosed; known room decides (#1487).
    assert admission["limit_usd"] == 0.75 and admission["known_usd"] == 0.0
    assert admission["accounted_usd"] == 0.0
    assert admission["estimated_wave_usd"] == 1.0 and admission["fits"] is True
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


def test_wave_without_paid_seats_is_admitted_without_pricing(gate, tmp_path):
    """An all-session wave rides the owner's subscription: nothing to price."""
    from ouroboros.tools.review_admission import admit_commit_gate_wave, commit_gate_paid_seats

    seats = commit_gate_paid_seats(
        {"prompt": "", "models": ["m/session"], "routes": [ReviewRouteKind.AGENT_SESSION],
         "row_plan": {"models": ["m/session"], "routes": [ReviewRouteKind.AGENT_SESSION], "slot_ids": ["s"],
                      "retrieves": [True], "parts": [("change", "coupling")], "session_tasks": [BRIEF]}},
        False,
    )
    assert seats == []
    assert admit_commit_gate_wave(_ctx(gate, tmp_path), seats) is None


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
    """Finding 1: a retrieving api row (native episode) reserves every round on
    the ledger exactly like a packet row, so it is a PAID seat priced by its
    first send — the executor's own opening with the seat's OWN two-part brief —
    while an agent-session row (subscription, ledger row written at settlement)
    is not priced."""
    from ouroboros.tools.review_admission import commit_gate_paid_seats
    from ouroboros.tools.review_multi_model import TRIAD_ROLE_HINT, _review_output_budget

    repo = tmp_path / "subject"
    repo.mkdir()
    prepared = {
        "prompt": "PACKET", "stable_prefix_len": 0, "session_task": "", "target_repo": repo,
        "models": ["triad/a", "triad/native", "m/session", COUPLING_MODEL],
        "routes": [ReviewRouteKind.API_CHAT, ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION,
                   ReviewRouteKind.API_CHAT],
        "row_plan": {"slot_ids": ["slot_1", "slot_2", "slot_3", "slot_4"], "subagent_ids": ["", "critic", "", ""],
                     "retrieves": [False, True, True, True],
                     "parts": [("change",), ("change", "coupling"), ("change", "coupling"), ("coupling",)],
                     "session_tasks": ["", "BRIEF TWO", "BRIEF THREE", "BRIEF FOUR"]},
    }
    seats = commit_gate_paid_seats(prepared, False)

    assert [(s["surface"], s["slot_id"], s["model"]) for s in seats] == [
        ("multi_model_review", "slot_1", "triad/a"),
        ("multi_model_review", "slot_2", "triad/native"),
        ("multi_model_review", "slot_4", COUPLING_MODEL),
    ]
    assert seats[1]["prompt_chars"] == _native_first_send_size(
        repo, surface="multi_model_review", session_task="BRIEF TWO", role_hint=TRIAD_ROLE_HINT,
        output_contract=REVIEW_TWO_PART_OBJECT_CONTRACT, slot_id="slot_2", model="triad/native")
    assert seats[2]["prompt_chars"] == _native_first_send_size(
        repo, surface="multi_model_review", session_task="BRIEF FOUR", role_hint=TRIAD_ROLE_HINT,
        output_contract=REVIEW_TWO_PART_OBJECT_CONTRACT, slot_id="slot_4", model=COUPLING_MODEL)
    assert seats[1]["prompt_chars"] != seats[2]["prompt_chars"]  # each seat is priced with ITS brief
    # A native first send carries instructions, the work-order AND the six tool
    # schemas (the packet row beside it carries the constitutional pack instead).
    from ouroboros.review_native_episode import native_episode_prompt, native_first_send_messages

    work_order_only = json.dumps(native_first_send_messages(native_episode_prompt(
        "multi_model_review", TRIAD_ROLE_HINT, "BRIEF TWO", REVIEW_TWO_PART_OBJECT_CONTRACT, "slot_2")),
        ensure_ascii=False)
    assert seats[1]["prompt_chars"] > len(work_order_only) > 0 and seats[0]["prompt_chars"] > 0
    assert [s["max_completion_tokens"] for s in seats] == [_review_output_budget()] * 3
    # Nothing is priced before the wave is prepared, and an exited assembly prices nothing.
    assert commit_gate_paid_seats(prepared, True) == [] and commit_gate_paid_seats(None, False) == []


def test_the_triad_user_turn_is_one_literal_for_send_and_admission(monkeypatch):
    """The triad packet's user turn is a single literal, shared by the send and
    by the admission that measures it."""
    from ouroboros.tools.review_multi_model import TRIAD_USER_TURN

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
    seat_scope = replace(caller, category="multi_model_review_review", review_slot_id="slot_native")
    kwargs = dict(root_task_id=ROOT, models=[model], prompt_chars=400_000, max_completion_tokens=1000,
                  task_id=ROOT, root_limit_usd=1000.0)
    with ua.usage_scope(caller):
        # The caller's transcript is warm (90% of the prompt read from cache).
        ua.stash_task_cache_split(ROOT, model, 90_000, provider=provider, ttl_seconds=300.0)
        callers_own = ua.review_wave_admission(tmp_path, **kwargs)["slot_bounds"][0]
        cold_seat = ua.review_wave_admission(
            tmp_path, categories="multi_model_review_review", slot_ids="slot_native", **kwargs)["slot_bounds"][0]
        with ua.usage_scope(seat_scope):
            expected_cold = ua._reservation_cost(ua.AttemptRequest(
                model=model, provider=provider, prompt_tokens_estimate=100_000,
                max_completion_tokens=1000, task_id=ROOT))
            ua.stash_task_cache_split(ROOT, model, 90_000, provider=provider, ttl_seconds=300.0)
        warm_seat = ua.review_wave_admission(
            tmp_path, categories="multi_model_review_review", slot_ids="slot_native", **kwargs)["slot_bounds"][0]
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
    assert (raised["limit_usd"], raised["known_usd"], raised["remaining_usd"]) == (50.0, 3.0, 47.0)
    assert raised["fits"] is True
    lowered = ua.review_wave_admission(gate, root_limit_usd=6.0, **kwargs)
    assert (lowered["limit_usd"], lowered["remaining_usd"], lowered["fits"]) == (6.0, 3.0, True)
    at_known = ua.review_wave_admission(gate, root_limit_usd=3.0, **kwargs)  # lowered to the known spend
    assert (at_known["limit_usd"], at_known["remaining_usd"], at_known["fits"]) == (3.0, 0.0, False)
    unfenced = ua.review_wave_admission(gate, **kwargs)  # no fence of its own: the ledger's $8 row
    assert (unfenced["limit_usd"], unfenced["remaining_usd"], unfenced["fits"]) == (8.0, 5.0, True)


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
    An explicit owner amendment raises root AND group from $8 to $50.
    The environment stays $8; every seat retains the amended authority rather
    than mistaking an in-memory root fence for a group-cap amendment."""
    _seed_settled(gate, fence=8.0, cost=4.0)
    from ouroboros.task_results import write_task_result

    write_task_result(gate, ROOT, "running", root_task_id=ROOT,
                      acceptance_root_cap_amendments=[{
                          "accounting_root_task_id": ROOT, "new_cap_usd": 50.0,
                          "source_identity": "fixture-owner-amendment",
                          "source_ref": {"kind": "fixture_owner_decision"},
                          "recorded_at": "2026-10-07T00:00:00+00:00",
                      }])
    llm = LedgerLLM()
    ctx, (review_err, coupling, _reason, _adv) = _run(
        gate, tmp_path, monkeypatch, llm, fence=50.0, env_fence=8.0)

    assert review_err is None, review_err
    assert coupling.blocked is False and coupling.status == "responded"
    assert sorted(llm.calls) == sorted([COUPLING_MODEL, *PACKET_MODELS])
    seats = [row for row in _ledger(gate) if row["source"] != "main"]
    assert sorted(row["model"] for row in seats) == sorted([COUPLING_MODEL, *PACKET_MODELS])
    assert {row["root_limit_usd"] for row in seats} == {50.0}
    assert {row["billing_group_limit_usd"] for row in seats} == {50.0}
    assert {row["billing_group_limit_source"] for row in seats} == {"owner_amendment"}
    assert not [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    assert not [r for r in ctx._last_triad_raw_results if r.get("status") != "ok"] or all(
        "BudgetExceeded" not in str(r.get("error") or "") for r in ctx._last_triad_raw_results)


def test_raised_root_without_group_amendment_refuses_the_whole_wave(gate, tmp_path, monkeypatch):
    """#1599: durable group8 beats an in-memory root50, before any seat sends.

    Known spend has reached the original group's $8: the wave is declined although
    the in-memory root fence $50 has room.
    """
    _seed_settled(gate, fence=8.0, cost=8.0)
    before = _ledger(gate)
    llm = LedgerLLM()
    ctx, (review_err, coupling, reason, _adv) = _run(
        gate, tmp_path, monkeypatch, llm, fence=50.0, env_fence=8.0)

    assert llm.calls == [] and _ledger(gate) == before
    assert reason == "review_wave_budget_insufficient" and coupling.status == "not_dispatched"
    assert review_err and "remaining=$0.000000" in review_err
    event = next(e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient")
    assert event["binding_axis"] == "group" and event["limit_usd"] == 8.0
    assert "whole-work billing-group budget fence" in review_err
    assert "amend the original billing-group owner's cap explicitly" in review_err
    assert "raise the per-task budget" not in review_err
    assert [r["status"] for r in ctx._last_triad_raw_results] == ["not_dispatched"] * 3


@pytest.mark.parametrize("group_limit", [0.0, 8.0, None])
def test_wave_group_resolution_matches_reservation_without_pinning_authority(
    gate, monkeypatch, group_limit,
):
    from ouroboros.task_results import write_task_result, load_task_result

    binding = {"billing_group_id": "original-root", "billing_group_limit_usd": group_limit,
               "billing_group_limit_source": "fixture-carried"}
    write_task_result(gate, ROOT, "running", root_task_id=ROOT, billing_group=binding)
    before = load_task_result(gate, ROOT)
    with ua.usage_scope(ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT,
                                    root_limit_usd=50.0)):
        _, resolved = ua._merge_scope(ua.AttemptRequest(model="triad/a", provider="test"))
        admission = ua.review_wave_admission(
            gate, root_task_id=ROOT, task_id=ROOT, root_limit_usd=50.0,
            models=[COUPLING_MODEL, *PACKET_MODELS], prompt_chars=10)
    assert resolved.billing_group_id == "original-root"
    assert resolved.billing_group_limit_usd == group_limit
    assert admission["fits"] is (group_limit is None or group_limit > 0.0)  # known $0 decides
    assert load_task_result(gate, ROOT) == before and _ledger(gate) == []


def test_bound_fence_that_does_not_fit_refuses_even_when_the_environment_is_roomier(gate, tmp_path, monkeypatch):
    """The converse: bound $8 (admission's fence), environment $50, $8 of known
    history: the bound fence is reached and nothing is dispatched — the roomier
    setting never buys a seat the admitting fence refused."""
    _seed_settled(gate, fence=8.0, cost=8.0)
    llm = LedgerLLM()
    ctx, (review_err, coupling, block_reason, _adv) = _run(
        gate, tmp_path, monkeypatch, llm, fence=8.0, env_fence=50.0)

    assert llm.calls == [] and len(_ledger(gate)) == 1  # the seeded attempt only (one current row)
    assert review_err and "per-task budget fence $8.000000" in review_err
    assert "remaining=$0.000000" in review_err
    assert block_reason == "review_wave_budget_insufficient"
    assert coupling.status == "not_dispatched"


@pytest.mark.parametrize("non_task", [False, True])
def test_wave_resolution_preserves_explicit_host_operation_scope(gate, monkeypatch, non_task):
    """A declared host operation never borrows canonical task amendment authority."""
    from ouroboros.task_results import write_task_result

    write_task_result(gate, ROOT, "running", root_task_id=ROOT, billing_group={
        "billing_group_id": "original-root", "billing_group_limit_usd": 0.0,
        "billing_group_limit_source": "fixture-carried"})
    bound = ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT,
                          root_limit_usd=50.0, non_task_operation=non_task)
    with ua.usage_scope(bound):
        admission = ua.review_wave_admission(
            gate, root_task_id=ROOT, task_id=ROOT, root_limit_usd=50.0,
            models=[COUPLING_MODEL], prompt_chars=10)
    assert admission["fits"] is non_task
    assert _ledger(gate) == []


def test_wave_resolution_does_not_borrow_a_foreign_roots_explicit_group(gate, monkeypatch):
    from ouroboros.task_results import write_task_result

    write_task_result(gate, "target", "running", root_task_id="target", billing_group={
        "billing_group_id": "target-origin", "billing_group_limit_usd": 0.0,
        "billing_group_limit_source": "fixture-carried"})
    caller = ua.UsageScope(drive_root=gate, task_id=ROOT, root_task_id=ROOT,
                           root_limit_usd=50.0, billing_group_id="caller-origin",
                           billing_group_limit_usd=50.0, billing_group_limit_source="fixture")
    with ua.usage_scope(caller):
        admission = ua.review_wave_admission(
            gate, root_task_id="target", task_id="target", root_limit_usd=50.0,
            models=[COUPLING_MODEL], prompt_chars=10)
    assert admission["fits"] is False and admission["billing_group_id"] == "target-origin"
    assert admission["binding_axis"] == "group" and _ledger(gate) == []


def test_admission_that_raises_fails_open_loudly_and_typed(gate, tmp_path, monkeypatch, caplog):
    """rc.14 audit (fable MINOR 2): an exception inside the money admission keeps
    the owner's fail-open (the wave dispatches, unadmitted) but is a warning plus
    ONE typed ``review_wave_admission_unavailable`` event naming the error —
    never a debug line indistinguishable from an admitted wave."""
    import logging

    def _boom(*_a, **_k):
        raise RuntimeError("no inspection tool schemas are projectable")

    monkeypatch.setattr(review_admission, "commit_gate_paid_seats", _boom)
    llm = LedgerLLM()
    with caplog.at_level(logging.WARNING, logger="ouroboros.tools.parallel_review"):
        ctx, (review_err, coupling, _reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert review_err is None and coupling.status == "responded"
    assert sorted(llm.calls) == sorted([COUPLING_MODEL, *PACKET_MODELS])
    events = [e for e in ctx.pending_events if e.get("type") == "review_wave_admission_unavailable"]
    assert len(events) == 1
    assert events[0]["surface"] == "commit_gate" and events[0]["task_id"] == ROOT
    assert events[0]["error"] == "RuntimeError: no inspection tool schemas are projectable"
    assert not [e for e in ctx.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    assert any("commit-gate wave admission unavailable (RuntimeError" in r.getMessage()
               and r.levelno == logging.WARNING for r in caplog.records)


def test_a_packet_only_wave_is_admitted_and_then_not_performed(gate, tmp_path, monkeypatch):
    """The gate counts only a PASS/FAIL answer: a wave of packet seats alone is
    priced, admitted and dispatched, and then NOT_PERFORMED — no seat was asked
    the coupling question — so the commit blocks naming Part 2, not the money."""
    prepared = _prepared(tmp_path)
    plan = prepared["row_plan"]
    for key in list(plan):
        plan[key] = plan[key][1:]
    prepared["models"], prepared["routes"] = list(plan["models"]), list(plan["routes"])
    monkeypatch.setattr(review, "_prepare_unified_review", lambda *_a, **_k: (prepared, None, False))
    llm = LedgerLLM()
    ctx, (review_err, coupling, block_reason, _adv) = _run(gate, tmp_path, monkeypatch, llm, fence=10.0)

    assert sorted(llm.calls) == sorted(PACKET_MODELS)
    assert review_err and "NOT_PERFORMED" in review_err and "Part 2" in review_err
    assert block_reason == "coupling_not_performed" and coupling.status == "not_performed"
    assert ctx._last_review_verdict["per_question"] == {"change": "PASS", "coupling": "not_performed"}
    assert REVIEW_JSON_ARRAY_CONTRACT  # the packet seats answered contract A
