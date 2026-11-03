"""Configuration refusals, pending custody and reliable counted-completion notices."""
from contextlib import contextmanager

import pytest

from ouroboros import owner_mailbox, review_custody, review_ledger as ledger, task_custody
from ouroboros.review_records import ReviewActorRecord
from ouroboros.tools import plan_ask, plan_packet
from tests._skill_review_shared import _build_skill, _make_ctx
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_plan_review_event_route import _mailbox_entries
from tests.test_plan_task_tool import CLEAN, _record, _write, harness as _harness
from tests.test_review_ledger import BOTH, _facts, _raw
from tests.test_reviewer_panel_plan_r1 import PENDING, _adviser
from tests.test_skill_source_tools import _typed_call

harness = _harness


@pytest.mark.parametrize("surface", ["direct", "lifecycle", "install", "update"])
@pytest.mark.parametrize("problem", ["malformed_pool", "unknown_reviewer", "missing_reason", "invalid_type"])
def test_skill_pool_configuration_failures_are_not_argument_errors(tmp_path, monkeypatch, surface, problem):
    from ouroboros.skill_review import review_skill
    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(_build_skill(tmp_path)))
    set_review_pool(monkeypatch, pool_roster(pool_seat("counted", "openai/reviewer"),
                                           pool_seat("peer", "openai/peer")))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    if problem == "malformed_pool":
        monkeypatch.setenv("OUROBOROS_SUBAGENTS", "{broken")
    reviewers = {"malformed_pool": [], "unknown_reviewer": ["not-a-catalog-row"],
                 "missing_reason": ["counted"], "invalid_type": "counted"}[problem]
    ctx = _make_ctx(tmp_path)

    def no_effect(*args, **kwargs):
        raise AssertionError("a refused review must neither dispatch nor install")

    monkeypatch.setattr("ouroboros.tools.review._handle_multi_model_review", no_effect)
    monkeypatch.setattr("ouroboros.skill_source_install.install_item", no_effect)
    monkeypatch.setattr("ouroboros.skill_source_install.update_installed", no_effect)
    monkeypatch.setattr("ouroboros.skill_source_install.start_review", no_effect)
    if surface == "direct":
        outcome = review_skill(ctx, "weather", reviewers=reviewers)
        assert outcome.status == "pending"
        code, message = outcome.dispatch_refusal_code, outcome.error
    elif surface == "lifecycle":
        outcome = run_skill_review_lifecycle_blocking(ctx, "weather", source="tool", reviewers=reviewers)
        assert outcome["status"] == "pending"
        code, message = outcome.get("dispatch_refusal_code", ""), outcome["error"]
    else:
        arguments = ({"id": "community/weather", "reason": "weather needed"} if surface == "install"
                     else {"name": "weather", "mode": "replace"})
        outcome = _typed_call("skill_" + surface, ctx, **arguments, reviewers=reviewers)
        assert outcome.status == "error"
        code, message = outcome.code, outcome.text
    if problem == "malformed_pool":
        assert code == ("TOOL_ERROR" if surface in {"install", "update"} else ""), message
        assert "invalid review pool configuration blocks skill review" in message
        assert "not valid JSON" in message
    else:
        assert code == "TOOL_ARG_ERROR", message
        assert {"unknown_reviewer": "REVIEWER_UNKNOWN", "missing_reason": "REVIEW_PANEL_REASON_REQUIRED",
                "invalid_type": "REVIEW_PANEL_ARGUMENT"}[problem] in message
        assert "invalid review pool configuration" not in message


@pytest.mark.parametrize("custody_lost", [False, True])
def test_not_reached_agrees_with_pending_after_real_custody_and_ledger_projection(custody_lost):
    raw = _raw("s", "m/s", "error", parts=BOTH, operation_state="settled")
    if custody_lost:
        raw = review_custody._custody_lost_row("commit_review", 0, "controller no longer owns actor", raw)
        assert raw["late_result_pending"] and ledger.panel_not_reached([raw]) == []
    else:
        raw["failure_code"] = "provider_error"
    record = ledger.build_commit_gate_record(_facts([raw], panel={"pool": [{"seat_id": "s"}]})).to_dict()
    assert ledger.counted_pending(record["rows"]) is custody_lost
    assert record["state"] == ("pending" if custody_lost else "settled")
    assert record["panel"]["not_reached"] == ([] if custody_lost else [{"seat_id": "s", "cause": "provider_error"}])


@pytest.mark.parametrize("initial_lock", ["available", "refused", "raised"])
def test_later_adviser_callback_retries_only_an_undelivered_completion(harness, monkeypatch, initial_lock):
    _adviser(harness)
    sub = harness.install({"s1": PENDING, "critic": PENDING})
    ctx = harness.make_ctx()
    _write(ctx, ask=["s1", "critic"])
    request = sub.calls[0]["request"]
    slots = {slot.slot_id: slot for slot in sub.calls[0]["slots"]}
    actors = {sid: ReviewActorRecord(slot_id=sid, model=slot.model, status="ok", raw_text=CLEAN,
                                    operation_state="settled") for sid, slot in slots.items()}
    real_lock, real_write = task_custody.task_mail_lock, owner_mailbox.write_task_message
    attempts, results = [], []

    @contextmanager
    def unavailable_lock(*args, **kwargs):
        yield False

    def observe_write(*args, **kwargs):
        attempts.append(kwargs["msg_id"])
        if initial_lock == "raised" and len(attempts) == 1:
            results.append("raised")
            raise OSError("completion writer unavailable")
        result = real_write(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(owner_mailbox, "write_task_message", observe_write)
    if initial_lock == "refused":
        monkeypatch.setattr(task_custody, "task_mail_lock", unavailable_lock)
    plan_ask.settle_released_reply(ctx, request=request, task_id=ctx.task_id, actor=actors["s1"], slot=slots["s1"],
                                  settled_wave={"s1": "ok", "critic": ""})
    ask = _record(harness)["asks"][0]
    assert bool(ask.get("completion_announced")) is (initial_lock == "available")
    assert len(_mailbox_entries(harness.drive, ctx.task_id)) == int(initial_lock == "available")
    assert [p["status"] for p in ask["participants"]] == ["answered", "pending"]

    monkeypatch.setattr(task_custody, "task_mail_lock", real_lock)
    for _ in range(2):  # the ordinary later callback, then a repeated callback
        plan_ask.settle_released_reply(ctx, request=request, task_id=ctx.task_id, actor=actors["critic"], slot=slots["critic"],
                                      settled_wave={"s1": "ok", "critic": "ok"})
    ask = _record(harness)["asks"][0]
    assert ask["completion_announced"]
    assert all(p["status"] == "answered" and p["reply"]["prose"] == "Looks sound to me." for p in ask["participants"])
    [frame] = _mailbox_entries(harness.drive, ctx.task_id)
    assert frame["msg_id"] == f"plan-complete:{ctx.task_id}:{ask['ask_id']}"
    assert results == {"available": [True], "refused": [False, True], "raised": ["raised", True]}[initial_lock]
    assert attempts == [frame["msg_id"]] * len(results)


@pytest.mark.parametrize("counted", [False, True])
def test_completion_frame_reserves_still_owed_for_counted_participants(counted):
    text = plan_packet.settled_frame("a1", [
        {"seat_id": "finished", "status": "answered", "owner_marked": True},
        {"seat_id": "working", "status": "pending", "owner_marked": counted},
    ], [])
    assert "1 answered" in text
    assert ("1 still owed (working)" in text) is counted
    if counted:
        assert "advisers still working" not in text
    else:
        assert "counted participants settled" in text and "still owed" not in text
        assert "advisers still working: working" in text
        assert "not counted" in text and "replies land in the plan record" in text
        assert "plan_task() reads them" in text and "no further notice" in text


@pytest.mark.parametrize("tool", ["skill_install", "skill_update"])
def test_an_installer_value_error_is_not_reported_as_a_caller_argument_error(tmp_path, monkeypatch, tool):
    """Panel arguments are refused before the installer runs; an installer's own failure stays a tool failure."""
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(_build_skill(tmp_path)))
    set_review_pool(monkeypatch, pool_roster(pool_seat("counted", "openai/reviewer")))
    ctx = _make_ctx(tmp_path)

    def broken_installer(*args, **kwargs):
        raise ValueError("installer state is unreadable")

    monkeypatch.setattr("ouroboros.skill_source_install.install_item", broken_installer)
    monkeypatch.setattr("ouroboros.skill_source_install.update_installed", broken_installer)
    arguments = ({"id": "community/weather", "reason": "weather needed"} if tool == "skill_install"
                 else {"name": "weather", "mode": "replace"})
    with pytest.raises(ValueError, match="installer state is unreadable"):
        _typed_call(tool, ctx, **arguments)
