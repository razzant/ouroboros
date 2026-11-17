"""Reviewer agreement belongs to the counted seats of the admitted wave."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as ledger
from ouroboros import reviewer_slot_config as slots
from ouroboros.review_ledger import advisers_only, reusable_outcome
from tests.test_review_ledger import BOTH, CRITICAL, _answer, _facts, _plan, _raw
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool


@pytest.fixture
def roster(monkeypatch):
    value = pool_roster(*(pool_seat(sid, f"openai/{sid}") for sid in ("a", "b", "c")),
                        pool_seat("adviser", "openai/adviser", marked=False))
    set_review_pool(monkeypatch, value)
    return value


@pytest.mark.parametrize("additional", [False, True])
@pytest.mark.parametrize("signal", ["pending", "FAIL"])
def test_adviser_neither_vetoes_nor_keeps_wave_pending(additional, signal):
    rows = [_raw("a", "model-a", parts=BOTH), _raw("b", "model-b", parts=BOTH)]
    critic = _raw("critic", "model-c", "pending" if signal == "pending" else "responded", parts=BOTH)
    if signal == "pending":
        critic.update(operation_state="in_flight", late_result_pending=True)
    else:
        critic["answers"]["change"] = _answer("change", "FAIL", [CRITICAL])
    rows.append(critic)
    plans = [_plan(r["slot_id"], r["model_id"], parts=BOTH,
                   additional=additional and r["slot_id"] == "critic") for r in rows]
    record = ledger.build_wave_record(_facts(rows, rows=plans), surface="change").to_dict()
    assert record["verdict"]["aggregate"] == ("PASS" if additional else "NOT_PERFORMED" if signal == "pending" else "FAIL")
    assert record["state"] == ("pending" if not additional and signal == "pending" else "settled")
    assert record["rows"][-1]["additional"] is additional
    assert record["verdict"]["per_row"]["critic"] == signal
    assert record["panel"]["not_reached"] == [], "an answer still in flight is awaiting, not unreachable"
    if signal == "pending":
        from ouroboros.tools.git_review_cycle import _review_custody_pending
        ctx = SimpleNamespace(_last_triad_raw_results=rows, _last_review_structured={"rows": plans})
        assert _review_custody_pending(ctx)  # physical custody remains even when the adviser cannot hold
        assert _review_custody_pending(ctx, counted_only=True) is not additional


@pytest.mark.parametrize("count,answers,expected", [(1, 1, "PASS"), (2, 2, "PASS"),
                                                    (2, 1, "QUORUM_FAILED")])
def test_quorum_counts_the_seats_asked(count, answers, expected):
    raws = [_raw(str(i), f"model-{i}", "responded" if i < answers else "error", parts=BOTH)
            for i in range(count)]
    record = ledger.build_wave_record(_facts(raws), surface="change")
    assert record.verdict["aggregate"] == expected


@pytest.mark.parametrize(("surface", "strictness", "reason", "count"), [
    ("own_body", "advisory", "focused change", 1),
    ("skill", "advisory", "focused change", 1),
    ("plan_first", "cyber_pro", "", 1),
])
def test_strictness_composes_the_counted_panel(roster, strictness, reason, count, surface):
    panel = slots.compose_review_panel(["b", "adviser"], reason, surface=surface, strictness=strictness)
    assert len([seat for seat in panel.seats if not seat.additional]) == count
    assert panel.additional == ("adviser",)
    assert "widened_from" not in panel.facts and "reviewers_subset_ignored" not in panel.facts
    assert panel.facts["reason"] == reason and panel.facts["chosen_by"] == "ouroboros"
    assert [row["seat_id"] for row in panel.facts["pool"]] == ["a", "b", "c"]


@pytest.mark.parametrize("surface", ["own_body", "skill", "plan_first"])
def test_advisory_requires_a_counted_seat_and_a_reason(roster, surface):
    with pytest.raises(slots.PanelCompositionError, match="REVIEW_PANEL_REASON_REQUIRED"):
        slots.compose_review_panel(["b"], surface=surface, strictness="advisory")
    with pytest.raises(slots.PanelCompositionError, match="REVIEW_PANEL_EMPTY"):
        slots.compose_review_panel(["adviser"], "a critic", surface=surface, strictness="advisory")
    assert len(slots.compose_review_panel(surface=surface, strictness="advisory").seats) == 3


def test_optional_change_and_later_plan_remain_composed(roster):
    assert len(slots.compose_review_panel(["a"], surface="change", strictness="advisory").seats) == 1
    assert slots.compose_review_panel(["adviser"], surface="plan", strictness="advisory").additional == ("adviser",)


@pytest.mark.parametrize("surface", ["own_body", "change", "skill", "result", "result_required"])
def test_cyber_pro_may_seat_advisers_alone_and_every_other_strictness_still_refuses(roster, surface):
    """#3194: Cyber Pro composes freely, an adviser alone included; it stays heard, never counted."""
    panel = slots.compose_review_panel(["adviser"], "a second look", surface=surface, strictness="cyber_pro")
    assert [(seat.slot.slot_id, seat.additional) for seat in panel.seats] == [("adviser", True)]
    assert (panel.facts["seats"], panel.facts["additional_seats"], panel.facts["assigned"]) == (0, 1, [])
    with pytest.raises(slots.PanelCompositionError, match="REVIEW_PANEL_EMPTY"):  # every other Access: unchanged
        slots.compose_review_panel(["adviser"], "x", surface=surface, strictness="advisory")
    mixed = slots.compose_review_panel(["a", "adviser"], "x", surface=surface, strictness="cyber_pro")
    assert (mixed.facts["assigned"], mixed.additional) == (["a"], ("adviser",))
    with pytest.raises(slots.PanelCompositionError, match="REVIEWER_UNKNOWN"):
        slots.compose_review_panel(["nobody"], "x", surface=surface, strictness="cyber_pro")
    assert slots.compose_review_panel(surface=surface, strictness="cyber_pro").facts["seats"] == 3


def test_main_self_review_is_the_main_seat_actually_seated(monkeypatch):
    set_review_pool(monkeypatch, pool_roster(pool_seat("adviser", "openai/adviser", marked=False)))
    alone = slots.compose_review_panel(["adviser"], "a second look", surface="change", strictness="cyber_pro")
    assert (alone.facts["pool_empty"], alone.facts["main_self_review"], alone.facts["assigned"]) == (True, False, [])
    beside = slots.compose_review_panel(["main", "adviser"], "both", surface="change", strictness="cyber_pro")
    assert (beside.facts["main_self_review"], beside.facts["assigned"]) == (True, ["main"])
    assert slots.compose_review_panel(surface="change", strictness="cyber_pro").facts["main_self_review"] is True


def _adviser_row(status="responded", **extra):
    return _raw("adviser", "model-adviser", status, parts=BOTH, **extra)


def test_an_advisers_only_wave_reduces_to_no_counted_verdict_and_is_held_by_its_own_operation():
    """Zero counted seats: never PASS/FAIL and never a false NOT_DISPATCHED; per-seat answers stay. A
    ``review_change`` record waits for its adviser; a commit record keeps no counted hold."""
    answered = [_adviser_row(answers={"change": _answer("change", "FAIL", [CRITICAL]), "coupling": _answer("coupling")})]
    plans = [_plan("adviser", "model-adviser", parts=BOTH, additional=True)]
    record = ledger.build_wave_record(_facts(answered, rows=plans), surface="change").to_dict()
    verdict = record["verdict"]
    assert (verdict["aggregate"], verdict["reason"], record["state"]) == ("NOT_PERFORMED", "no_counted_reviewers", "settled")
    assert verdict["per_row"] == {"adviser": "FAIL"} and verdict["quorum"]["assigned"] == 0
    assert advisers_only(record) and reusable_outcome(record)

    running = [_adviser_row("pending", operation_state="in_flight", late_result_pending=True)]
    assert not ledger.counted_pending(ledger.build_rows(_facts(running, rows=plans))), "its semantics are unchanged"
    for surface, state in (("change", "pending"), ("result", "pending"), ("commit_gate", "settled")):
        held = ledger.build_wave_record(_facts(running, rows=plans, pending=True), surface=surface).to_dict()
        assert (held["state"], held["verdict"]["aggregate"]) == (state, "NOT_PERFORMED"), surface

    for unusable in (_adviser_row("error"), _adviser_row("unparseable")):
        settled = ledger.build_wave_record(_facts([unusable], rows=plans), surface="change").to_dict()
        assert settled["verdict"]["reason"] == "no_counted_reviewers" and not reusable_outcome(settled)
    refused = ledger.build_wave_record(_facts([], rows=plans, dispatch_refusal={"kind": "review_cycles_exhausted"}),
                                       surface="change").to_dict()
    assert refused["verdict"]["aggregate"] == "NOT_DISPATCHED"
    unsent = ledger.build_wave_record(_facts([_adviser_row("not_dispatched", operation_state="not_dispatched")],
                                             rows=plans), surface="change").to_dict()
    assert unsent["verdict"]["aggregate"] == "NOT_DISPATCHED"


def test_empty_pool_discloses_main_and_subscription_cost_is_never_invented(monkeypatch):
    set_review_pool(monkeypatch, pool_roster(pool_seat("adviser", "openai/adviser", marked=False)))
    panel = slots.compose_review_panel(surface="skill", strictness="advisory")
    assert panel.facts["pool"] == [] and panel.facts["main_self_review"] is True
    assert panel.facts["assigned"] == ["main"]
    assert slots.review_cost_facts(None, route="agent_session") == {"cost_usd": None, "cost_kind": "subscription"}
    assert slots.review_cost_facts(None) == {"cost_usd": None, "cost_kind": "unknown"}
    assert slots.review_cost_facts(0.025) == {"cost_usd": 0.025, "cost_kind": "measured"}


@pytest.mark.parametrize("counted", [False, True])
def test_only_counted_participants_keep_a_plan_ask_pending(counted):
    from ouroboros.plan_record import pending_asks
    ask = {"ask_id": "a1", "participants": [{"seat_id": "a", "owner_marked": True, "status": "answered"},
        {"seat_id": "critic", "owner_marked": counted, "status": "pending"}]}
    assert pending_asks({"asks": [ask]}) == ([ask] if counted else [])
    assert ask["participants"][1]["status"] == "pending", "the adviser's own custody stays visible"


def test_wave_snapshot_and_reuse_identity_survive_tick_edits(roster, monkeypatch):
    before = slots.compose_review_panel(["a"], "targeted", surface="own_body", strictness="advisory")
    full = slots.compose_review_panel(surface="own_body", strictness="advisory")
    assert before.assigned != full.assigned
    set_review_pool(monkeypatch, pool_roster(pool_seat("b", "openai/b")))
    with slots.composed_review_pool(before.seats):
        assert [row.slot_id for row in slots.review_pool_slots()] == ["a"]
    assert [row["seat_id"] for row in before.facts["pool"]] == ["a", "b", "c"]
    assert [row.slot_id for row in slots.review_pool_slots()] == ["b"]


def test_raising_an_adviser_to_counted_lapses_commit_reuse(roster, monkeypatch):
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint
    first = slots.compose_review_panel(["a", "adviser"], "targeted", surface="own_body", strictness="advisory")
    with slots.composed_review_pool(first.seats):
        old = commit_review_contract_fingerprint()
    set_review_pool(monkeypatch, pool_roster(pool_seat("a", "openai/a"), pool_seat("adviser", "openai/adviser")))
    full = slots.compose_review_panel(surface="own_body", strictness="advisory")
    with slots.composed_review_pool(full.seats):
        new = commit_review_contract_fingerprint()
    assert old and new and old != new


@pytest.mark.parametrize("additional", [False, True])
def test_commit_findings_do_not_promote_an_adviser_to_a_blocker(additional):
    from ouroboros.tools.review import _collect_review_findings
    ctx = SimpleNamespace()
    critical, warnings, _, raw = _collect_review_findings(ctx, [
        {"slot_id": "critic", "model": "openai/critic", "text": json.dumps([CRITICAL])}],
        {"slot_ids": ["critic"], "parts": [["change"]], "additional": [additional]})
    assert bool(critical) is not additional
    assert bool(warnings) is additional
    assert raw[0]["additional"] is additional and raw[0]["parsed_items"]


@pytest.mark.parametrize("additional", [False, True])
def test_skill_aggregator_hears_advisers_without_counting_them(additional):
    from ouroboros.review_ledger import seat_findings
    from ouroboros.skill_review_status import status_from_findings
    from ouroboros.triad_review import parse_seat_answers
    from tests._skill_review_shared import _make_actor, _pass_array_for_script_skill
    passing = _pass_array_for_script_skill()
    failing = passing.replace('"verdict": "PASS"', '"verdict": "FAIL"', 1)
    parsed = parse_seat_answers({"results": [
        {**_make_actor("openai/a", passing), "slot_id": "a"},
        {**_make_actor("openai/critic", failing), "slot_id": "critic", "additional": additional}]},
        {"a": ("skill",), "critic": ("skill",)})
    critical, advisory, extra = seat_findings([record.to_dict() for record in parsed.actor_records],
                                              {"critic"} if additional else set())
    assert status_from_findings([*critical, *advisory], "script") == ("clean" if additional else "blockers")
    assert (len(extra) == 1) is additional and any(row["verdict"] == "FAIL" for row in parsed.findings)
    if additional:  # Cyber Pro may seat the critic alone: its skill stays pending, its findings heard
        from ouroboros.tools.review_skill import outcome_status
        rows = [{"seat_id": "critic", "status": "responded", "parts": ["skill"], "additional": True,
                 "answers": {"skill": {"status": "responded", "verdict": "FAIL", "findings": extra}}}]
        alone = {"state": "settled", **ledger.reduce_verdict(rows), "findings": {"additional_findings": extra}}
        assert (alone["aggregate"], alone["reason"]) == ("NOT_PERFORMED", "no_counted_reviewers")
        status, findings, reason = outcome_status(alone)
        assert status == "pending" and [f.get("additional") for f in findings] == [True] and "NOT_PERFORMED" in reason


def test_owner_skill_job_defaults_to_whole_pool_and_freezes_tick_edits(tmp_path, monkeypatch, roster):
    from ouroboros.skill_review import SkillReviewOutcome
    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
    from tests._skill_review_shared import _build_skill, _make_ctx
    from tests.test_skill_review_runner import _reset_queue
    _reset_queue()
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    path = _build_skill(tmp_path)
    ctx = _make_ctx(tmp_path)
    def review(_ctx, skill_name):
        assert [s.slot_id for s in slots.review_pool_slots()] == ["a", "b", "c"]
        set_review_pool(monkeypatch, pool_roster(pool_seat("b", "openai/b")))
        assert [s.slot_id for s in slots.review_pool_slots()] == ["a", "b", "c"]
        return SkillReviewOutcome(skill_name=skill_name, status="blockers", error="a counted objection")
    try:
        result = run_skill_review_lifecycle_blocking(ctx, "weather", source="skills", review_impl=review, repo_path=str(path))
        assert result["panel"]["chosen_by"] == "owner" and result["panel"]["assigned"] == ["a", "b", "c"]
        assert [r["seat_id"] for r in result["panel"]["pool"]] == ["a", "b", "c"]
        assert result["status"] == "blockers"
    finally:
        _reset_queue()


def test_a_required_root_requires_a_counted_seat_and_a_reason_when_narrowed(roster):
    """Outside Cyber Pro a Required root refuses a narrowing with no reason and composes it
    once a reason is given; nothing widens a named subset."""
    with pytest.raises(slots.PanelCompositionError, match="REVIEW_PANEL_REASON_REQUIRED"):
        slots.compose_review_panel(["b"], surface="result_required", strictness="advisory")
    with pytest.raises(slots.PanelCompositionError, match="REVIEW_PANEL_EMPTY"):
        slots.compose_review_panel(["adviser"], "only a critic", surface="result_required", strictness="advisory")
    narrowed = slots.compose_review_panel(["b"], "the risky claim", surface="result_required", strictness="advisory")
    assert [seat.slot.slot_id for seat in narrowed.seats if not seat.additional] == ["b"]
    assert "widened_from" not in narrowed.facts and narrowed.facts["composition"] == "composed"


def test_auto_result_cyber_pro_and_omission_do_not_take_the_required_root_floor(roster):
    """The floor is the Required-root surface only. Auto (surface ``result``) and Cyber Pro
    keep the named rows; omitting reviewers, or the host seating them, is the whole pool."""
    auto = slots.compose_review_panel(["b"], surface="result", strictness="advisory")
    assert [seat.slot.slot_id for seat in auto.seats] == ["b"] and auto.facts["composition"] == "composed"
    cyber = slots.compose_review_panel(["b"], surface="result_required", strictness="cyber_pro")
    assert [seat.slot.slot_id for seat in cyber.seats] == ["b"]
    assert cyber.facts["reason_missing"] is True
    named_all = slots.compose_review_panel(["a", "b", "c"], surface="result_required", strictness="advisory")
    assert named_all.facts["reason_missing"] is False and sorted(named_all.facts["assigned"]) == ["a", "b", "c"]
    host = slots.compose_review_panel(surface="result_required", strictness="advisory", chosen_by="host")
    assert host.facts["chosen_by"] == "host"
    assert host.facts["composition"] == "full_pool" and host.facts["assigned"] == ["a", "b", "c"]
    assert slots.compose_review_panel(surface="result_required", strictness="advisory").facts["chosen_by"] == "ouroboros"


def test_the_result_surface_text_states_the_required_root_rule():
    from ouroboros.subagent_runtime import _REVIEW_SURFACES

    text = _REVIEW_SURFACES["result"]
    assert "Required root" in text and "chosen by the host" in text
    assert "composes freely" in text and "Off is refused" in text
