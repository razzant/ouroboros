"""A result review's outcome reaches its author before the final delivery (owner, 2 Nov).

Any settled outcome the author has not received whole, a clean PASS included, is handed back once with
every finding, and the author decides; one that a review tool already returned to it whole needs no second
turn. Receipt is exposure, never agreement. Proven at the public consumers: the real Main loop's finish path
over the real tool registry (Main is scripted at ``call_llm_with_retry``; the reviewers are the shared
offline seat seam), and the delivery step ``result_review_at_delivery``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop
import ouroboros.review_substrate as substrate
from ouroboros import review_ledger
from ouroboros import review_waiting as waiting
from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.review_result import result_review_at_delivery
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_result_review import CRITICAL, _floor_ctx, _produced_answer_protocol, _seam

ANSWER = "FINAL ANSWER: 2"
REMARK = "UNSEEN-REVIEW-FACT-2339: name the unit beside the number."
ADVISORY = json.dumps([{"item": "result_detail", "verdict": "FAIL", "severity": "advisory", "reason": REMARK}])


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _call(name, arguments, call_id):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def _finish(answer=ANSWER, **extra):
    return _call("finish_task", {"action": "finish", "answer": answer, **extra}, "complete")


def _holds(ctx):
    """The canonical ``completion_held`` rows: (round, origin, reason, subreason, answer sha)."""
    path = ctx.drive_root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [(row["round"], row["origin"], row["reason"], row["subreason"], row["answer_sha256"]) for row in rows
            if row.get("type") == "task_checkpoint" and row.get("checkpoint_kind") == "completion_held"]


def _main(ctx, monkeypatch, responses):
    """The real Main loop over ``ctx``: returns its delivered text, its trace and every request Main received."""
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    requests, script = [], iter(responses)

    def respond(_llm, messages, *_a, **_k):
        requests.append(json.loads(json.dumps(messages)))
        response = next(script)
        return (response() if callable(response) else response), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    text, _usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "How many?"}], registry, SimpleNamespace(default_model=lambda: "m"),
        ctx.drive_root / "logs", lambda _text, **_kw: None, queue.Queue(), task_id=ctx.task_id,
        drive_root=ctx.drive_root)
    return text, trace, requests


def _host_notices(request):
    return [str(row.get("content")) for row in request if row.get("role") == "user"]


# --- The real loop: an unreceived outcome is handed back first; a received one is not handed back again -----

@pytest.mark.parametrize("kind", ["clean", "advisory"])
def test_an_unreceived_pass_reaches_its_author_before_delivery(tmp_path, monkeypatch, kind):
    """The negative evidence of #2339: the host's Required review PASSes (clean, or with an advisory remark) and
    the result used to be delivered with nobody having read it. Now Main receives the whole outcome first and
    decides; its next finish delivers and buys nothing more."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot",
                        _seam(sends, {"t1": ADVISORY} if kind == "advisory" else {}))

    text, trace, requests = _main(ctx, monkeypatch, [{"content": ANSWER}, _finish()])

    assert text == ANSWER and len(requests) == 2 and len(sends) == 3, "one host wave; the decision buys none"
    handback = [notice for notice in _host_notices(requests[1]) if notice.startswith("[Result review]")]
    assert len(handback) == 1 and ": PASS." in handback[0], "the outcome reached Main before the delivery"
    assert (REMARK in handback[0]) is (kind == "advisory"), "an advisory remark rides the PASS whole"
    assert _holds(ctx) == [(1, "prose", "result_review", "pass", _sha(ANSWER))]
    [run] = trace["review_runs"]
    assert run["aggregate_signal"] == "PASS" and not run.get("informed_finish")


@pytest.mark.parametrize("seat, runtime", [("[]", "pro"), (ADVISORY, "pro"), (CRITICAL, "pro"), (CRITICAL, "cyber_pro")])
def test_an_outcome_the_review_tool_returned_whole_needs_no_second_turn(tmp_path, monkeypatch, seat, runtime):
    """Main asked for the review itself and the tool returned it whole into the transcript: its finish delivers
    at once (a FAIL in every Access as Main's own informed decision), with no ritual turn."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", runtime)
    _produced_answer_protocol(monkeypatch)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": seat}))

    text, trace, requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER}, "review"), _finish()])

    assert text == ANSWER and len(requests) == 2 and len(sends) == 3
    tool_rows = [row for row in requests[1] if row.get("role") == "tool"]
    [delivered] = [row["content"] for row in tool_rows if '"record_id"' in str(row["content"])]
    [run] = trace["review_runs"]
    assert run["record_id"] in delivered and _holds(ctx) == [], "no hold: the tool already returned it"
    assert not any(notice.startswith("[Result review]") for notice in _host_notices(requests[1]))
    assert bool(run.get("informed_finish")) is (seat == CRITICAL), "over findings: Main's own decision"


@pytest.mark.parametrize("mode, root", [("required", True), ("auto", True), ("auto", False)])
def test_a_stop_over_a_failed_result_the_tool_returned_buys_no_review_and_says_why(tmp_path, monkeypatch, mode, root):
    """The FAIL Main already read holds nothing, in Auto and for a child too: Main's explicit stop ends the task
    at once, buys no review and records why none ran."""
    ctx = _floor_ctx(tmp_path, monkeypatch, mode=mode, root=root)
    _produced_answer_protocol(monkeypatch)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))

    text, trace, requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER}, "review"),
        _call("finish_task", {"action": "stop", "answer": ANSWER, "rationale": "the reviewers' finding stands"},
              "stop")])

    assert len(requests) == 2 and len(sends) == 3 and _holds(ctx) == []
    assert text  # the explicit stop ends the task without buying another review
    assert trace["review_decision"]["trigger"] == "author_stop"


def test_a_reasoned_finish_of_the_fixed_answer_after_a_received_fail_delivers_it_at_once(tmp_path, monkeypatch):
    """Required + Advisory (owner, 8 Nov 2026): A's FAIL reached Main whole, and Main's explicit finish_task of the
    fixed B with its reason is its decision about exactly B: B is delivered at once, buys nothing and needs no
    identical resend. The decision stays B's: another answer, a new outcome or new words of the owner put the
    review before the author again."""
    from ouroboros.outcomes import _review_axis
    from ouroboros.review_waiting import outcome_states

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    fixed = "FINAL ANSWER: 3"
    text, trace, requests = _main(ctx, monkeypatch, [
        {"content": ANSWER}, _finish(fixed, rationale="Counted again: three findings are listed now.")])
    assert text == fixed and len(requests) == 2 and len(sends) == 3, "one wave; the fix buys none, no ritual resend"
    assert [row[2:4] for row in _holds(ctx)] == [("result_review", "findings")]
    [run] = trace["review_runs"]
    assert run["earlier_revision"] is True and run["informed_finish"]["rationale"].startswith("Counted again")
    assert run["informed_finish"]["subject_sha"] == run["binding_hash"] != run["reviewed_subject_sha"]
    assert _review_axis(trace)["author_finish"] == "author_finish"

    shown = []
    assert ctx._completion_selected["answer"] == fixed, "the decision B's delivery consumed is still on the context"
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 4", trace, shown.append) is True, "C is not B"
    assert "An earlier version of your result was reviewed" in shown[-1] and len(sends) == 3
    with monkeypatch.context() as patch:  # a genuinely new outcome under A's record, since the decision
        patch.setattr("ouroboros.review_waiting.outcome_states",
                      lambda record, real=outcome_states: [*real(record), "a-new-outcome#complete"])
        assert result_review_at_delivery(ctx, fixed, trace, shown.append) is True, "a new outcome reaches it first"
    ctx._owner_directives = [{"source": "owner_mailbox", "content": "Name the unit too.", "msg_id": "m-7"}]
    assert result_review_at_delivery(ctx, fixed, trace, shown.append) is True, "new words of the owner are asked about"
    assert "Name the unit too." in shown[-1] and "do they change what was asked?" in shown[-1] and len(sends) == 3


def test_a_same_byte_write_through_the_real_loop_buys_no_review_and_no_nudge(tmp_path, monkeypatch):
    """The real Main loop (8 Nov 2026 follow-up): rewriting a file with the bytes it already holds produces nothing,
    so Required buys no review, nothing is held, and the verify-before-done nudge (the effect projection's other
    reader) asks for nothing; the answer is delivered as written."""
    from tests.test_result_review import _baselined

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _baselined(ctx)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    answer = "app.py already says VALUE = 1; nothing needed changing."
    text, trace, requests = _main(ctx, monkeypatch, [
        _call("write_file", {"path": "app.py", "content": "VALUE = 1\n"}, "same"), {"content": answer}])
    assert text == answer and len(requests) == 2 and sends == [] and _holds(ctx) == []
    assert trace["tool_calls"][0]["tool_result_meta"]["operation_outcome"] == "completed_no_effect"
    assert trace["review_decision"]["trigger"] == "no_production"


# --- The delivery step -------------------------------------------------------------------------------

def test_every_finding_reaches_the_author_whole(tmp_path, monkeypatch):
    """No silent cut: a review with more than twelve findings is handed back with every one of them."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    many = [{"item": f"detail_{index}", "verdict": "FAIL", "severity": "advisory", "reason": f"UNIQUE-{index:02d}"}
            for index in range(13)]
    critical = [{"item": "missing_part", "verdict": "FAIL", "severity": "critical", "reason": "UNIQUE-CRITICAL"}]
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot",
                        _seam([], {"t1": json.dumps(many), "t2": json.dumps(critical)}))
    shown = []

    assert result_review_at_delivery(ctx, ANSWER, {"tool_calls": []}, shown.append) is True
    [text] = shown
    assert all(f"UNIQUE-{index:02d}" in text for index in range(13)) and "UNIQUE-CRITICAL" in text
    assert text.index("UNIQUE-CRITICAL") < text.index("UNIQUE-00"), "critical findings come first"


def _revise_seat(root, record_id, seat_id="t1", *, operation_id=None, row=None, outcome=None, finding=None,
                 bucket="advisory_findings"):
    """One seat as a later settlement of the same record leaves it (``review_ledger.settle_pending_payload``
    merges a later answer into its row; an explicit continuation would add a new operation): its row and
    canonical outcome take the given facts, ``finding`` joins ``bucket``, the rest of the record stays."""
    def mutate(record):
        rows = copy.deepcopy(record["rows"])
        seat = next(item for item in rows if item["seat_id"] == seat_id)
        seat.update(row or {})
        seat["outcome"] = {**seat["outcome"], **(outcome or {})}
        if operation_id:
            seat["operation_id"] = operation_id
            seat["outcome"].update(operation_id=operation_id, outcome_id=f"{operation_id}:{seat_id}")
        verdict = dict(record["verdict"])
        if finding:
            verdict[bucket] = [*(verdict.get(bucket) or []), {
                "item": "late_detail", "verdict": "FAIL", "severity": "advisory", "reason": finding, "seat_id": seat_id}]
        return {**record, "rows": rows, "verdict": verdict}

    return review_ledger.revise_record(root, record_id, mutate)


def test_the_same_record_hands_back_each_new_seat_outcome_but_not_a_cost_only_revision(tmp_path, monkeypatch):
    """Identity is each seat's actual outcome with its status: a revision that only records cost is the outcome
    already received; under the same record id a late answer of the same operation (its custody was lost) and
    a new operation (an explicit reviewer continuation's shape) are new outcomes, handed back once each, never
    suppressed because that record was shown before."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([]))
    from ouroboros.tools.review_change import run_review_change

    root = review_ledger.ledger_root(ctx)
    record_id = run_review_change(ctx, subject="result", answer=ANSWER)["record_id"]  # Main's own call, unread
    _revise_seat(root, record_id, row={"status": "error", "operation_state": "custody_lost"}, outcome={"status": "error"})
    trace, shown = {"tool_calls": []}, []
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is True, "the unreceived outcome"
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is False

    priced = review_ledger.revise_record(root, record_id, lambda r: {**r, "cost": {**(r.get("cost") or {}), "usd": 0.42}})
    assert priced["revision"] == 3
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is False and len(shown) == 1

    operation = review_ledger.load_record(root, record_id)["rows"][0]["operation_id"]
    _revise_seat(root, record_id, row={"status": "completed", "operation_state": "settled"},
             outcome={"status": "responded"}, finding="LATE-ANSWER-REMARK")
    assert review_ledger.load_record(root, record_id)["rows"][0]["operation_id"] == operation, "same operation"
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is True and "LATE-ANSWER-REMARK" in shown[-1]
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is False

    _revise_seat(root, record_id, operation_id=operation + "-continued", finding="CONTINUED-REMARK")
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is True and "CONTINUED-REMARK" in shown[-1]
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is False, "received once is enough"
    assert len(trace["review_runs"]) == 1 and len(shown) == 3


def test_an_added_reviewers_remarks_reach_the_author_and_its_outcome_counts(tmp_path, monkeypatch):
    """A row the mind added beside the counted seats is heard, never counted: its remarks are handed back with
    the rest, and a new outcome of that seat alone is handed back too."""
    from ouroboros.tools.review_change import run_review_change

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    set_review_pool(monkeypatch, pool_roster(  # the golden seats, with s1 enabled but not marked for the pool
        pool_seat("t1", "openai/gpt-5.6-sol", effort="high"),
        pool_seat("t2", "codex=gpt-5.6-sol", kind="agent_session", profile_id="pinned", effort="high"),
        pool_seat("s1", "openai/gpt-5.6-sol", delivery="native", effort="xhigh", marked=False)))
    adviser = "ADDED-REVIEWER-REMARK: the unit is missing."
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([], {"s1": json.dumps([
        {"item": "result_detail", "verdict": "FAIL", "severity": "advisory", "reason": adviser}])}))
    reviewed = run_review_change(ctx, subject="result", answer=ANSWER, reviewers=["t1", "s1"],
                                 reason="one counted seat and an added eye")
    assert reviewed["panel"]["additional"] == ["s1"] and reviewed["aggregate"] == "PASS"
    assert [item["reason"] for item in reviewed["findings"]["additional_findings"]] == [adviser]
    trace, shown = {"tool_calls": []}, []

    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is True
    assert adviser in shown[0] and "heard, not counted" in shown[0]
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is False

    def added_seat_changes(record):
        rows = copy.deepcopy(record["rows"])
        seat = next(item for item in rows if item["seat_id"] == "s1")
        seat["outcome"] = {**seat["outcome"], "status": "interrupted"}
        return {**record, "rows": rows}

    review_ledger.revise_record(review_ledger.ledger_root(ctx), reviewed["record_id"], added_seat_changes)
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is True, "the added seat's new outcome"

def test_only_a_complete_tool_result_of_that_record_counts_as_received(tmp_path, monkeypatch):
    """The receipt the loop recorded beside the tool's own result is the exposure: the same receipt over a
    shortened view, or another record's receipt, is not; the unreceived outcome is handed back once."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([], {"t1": ADVISORY}))
    _text, trace, _requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER}, "review"), _finish()])
    [call] = [row for row in trace["tool_calls"] if (row.get("tool_result_meta") or {}).get("review_delivery")]
    record_id = call["tool_result_meta"]["review_delivery"]["record_id"]

    for variant in ({**call, "result_partial": True},
                    {**call, "tool_result_meta": {"review_delivery": {
                        **call["tool_result_meta"]["review_delivery"], "record_id": "rl-another-record"}}}):
        fresh, shown = {"tool_calls": [variant]}, []
        ctx._result_review_shown = set()
        assert result_review_at_delivery(ctx, ANSWER, fresh, shown.append) is True, variant.keys()
        assert REMARK in shown[0] and record_id in shown[0]


def test_selected_results_of_every_seat_are_not_the_review_reaching_its_author(tmp_path, monkeypatch):
    """A wait that selected every seat receives each one's outcome as [Selected results], and that notice
    advances the wait's delivery cursor; it is still not the review's own result, so its outcome is handed
    back whole before delivery."""
    from ouroboros.tools.review_change import run_review_change
    from ouroboros.working_checkpoint import flush_content_acks

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([], {"t1": ADVISORY}))
    record_id = run_review_change(ctx, subject="result", answer=ANSWER)["record_id"]
    root = review_ledger.ledger_root(ctx)
    record = review_ledger.load_record(root, record_id)
    pins = [{"wave_id": record["review_wave_id"], "operation_id": row["operation_id"], "slot_id": row["seat_id"]}
            for row in record["rows"]]
    ctx._left_review_wait = {"record_id": record_id, "selection": {"reviewers": pins, "record_id": record_id},
                             "seen_helpers": [], "seen_outcomes": []}
    messages = []
    waiting.append_helper_notice(ctx, messages)
    flush_content_acks(ctx)
    [notice] = messages
    assert notice["content"].startswith("[Selected results]") and len(json.loads(notice["content"][19:])) == 3
    assert len(review_ledger.load_record(root, record_id)["waiting"]["seen_outcomes"]) == 3, "cursor advanced"
    shown = []

    assert result_review_at_delivery(ctx, ANSWER, {"tool_calls": []}, shown.append) is True
    assert REMARK in shown[0] and ": PASS." in shown[0]


def test_a_receipt_carries_another_waves_readiness_without_counting_that_review_as_received(tmp_path, monkeypatch):
    """The result of one review may also carry the selected readiness of a wave left earlier. Its cursor advances
    for them, so they are not offered again; its own record's outcomes stay the only ones it delivered whole, so
    that earlier review is still handed back."""
    from ouroboros.tools.review_change import run_review_change
    from ouroboros.tools.tool_result import (
        _install_tool_result_sidecar,
        _published_tool_result,
        _restore_tool_result_sidecar,
    )

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([], {"t1": ADVISORY}))
    root = review_ledger.ledger_root(ctx)
    left = review_ledger.load_record(root, run_review_change(ctx, subject="result", answer=ANSWER)["record_id"])
    other = review_ledger.load_record(root, run_review_change(ctx, subject="result", answer="Another answer")["record_id"])
    pins = [{"wave_id": left["review_wave_id"], "operation_id": row["operation_id"], "slot_id": row["seat_id"]}
            for row in left["rows"]]
    ctx._left_review_wait = {"record_id": left["record_id"], "selection": {"record_id": left["record_id"], "reviewers": pins},
                             "seen_helpers": [], "seen_outcomes": []}
    ready = [row["id"] for row in waiting.delivery_ready(ctx, ctx._left_review_wait)]
    assert len(ready) == len(pins)
    sidecar = _install_tool_result_sidecar(ctx, object())
    try:
        text = waiting.publish_delivery(ctx, "The other review's result.", other, rendered_rows=other["rows"])
        published = _published_tool_result(ctx, None)
    finally:
        _restore_tool_result_sidecar(sidecar)
    receipt = published.meta["review_delivery"]
    assert "[Selected results]" in text and receipt["record_id"] == other["record_id"]
    assert set(ready) <= set(receipt["outcomes"]) and len(receipt["outcomes"]) == len(ready) + len(other["rows"])
    assert len(receipt["outcome_states"]) == len(other["rows"]) and not {state.split("#")[0] for state in receipt["outcome_states"]} & set(ready)
    waiting.acknowledge_delivery(ctx, receipt)
    assert waiting.ready_helpers(ctx, ctx._review_wait_selection | {"selection": ctx._left_review_wait["selection"]}) == []
    trace, shown = {"tool_calls": [{"tool": "review_change", "tool_result_meta": dict(published.meta)}]}, []

    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is True and REMARK in shown[0], "left: unreceived"
    assert result_review_at_delivery(ctx, "Another answer", trace, shown.append) is False, "its own record: received"


def test_a_received_outcome_survives_the_checkpoint_and_shown_keeps_its_string_shape(tmp_path, monkeypatch):
    """The working checkpoint carries the trace as JSON: a tool-returned outcome stays received after a
    round-trip. ``_result_review_shown`` stays a set of strings (an outcome identity each; a $0 refusal's is its
    record id), so a restored JSON list still reads; a bare record id of a record with seat outcomes is not its
    identity and hands that outcome back once, never suppresses it."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([], {"t1": ADVISORY}))
    from ouroboros.owner_wait import continuation_state

    _text, trace, _requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER}, "review"), _finish()])
    saved = json.dumps(continuation_state(ctx, [], trace, {}, 2, [], set()), default=lambda value: {"opaque": 1})
    restored = {}
    restored.update(json.loads(saved)["trace"])  # ``restore_continuation_state``: ``trace.update(state["trace"])``
    ctx._result_review_shown = set()
    assert result_review_at_delivery(ctx, ANSWER, restored, lambda _text: None) is False

    host = _floor_ctx(tmp_path / "host", monkeypatch)
    trace, shown = {"tool_calls": []}, []
    assert result_review_at_delivery(host, ANSWER, trace, shown.append) is True
    keys = host._result_review_shown
    assert keys and all(isinstance(key, str) for key in keys)
    host._result_review_shown = sorted(keys)  # a JSON-restored list
    assert result_review_at_delivery(host, ANSWER, trace, shown.append) is False and len(shown) == 1
    host._result_review_shown = [trace["review_runs"][0]["record_id"]]  # a record id alone
    assert result_review_at_delivery(host, ANSWER, trace, shown.append) is True and len(shown) == 2


@pytest.mark.parametrize("handback", ["whole", "request_only", "insertion_failed"])
def test_host_handback_acknowledges_only_inserted_outcome_text_at_its_checkpoint(tmp_path, handback):
    from ouroboros import review_ledger
    from ouroboros.review_pause import owe_unwaited_review
    from ouroboros.tools.review_result import _findings_text, _show
    from ouroboros.working_checkpoint import flush_content_acks
    from tests.test_review_waiting import _author, _duties, _result_record

    ctx = _author(tmp_path)
    record = _result_record(tmp_path, settled=("a", "b"))
    assert owe_unwaited_review(ctx, surface="result", retry_key="same-wave", record_id=record["record_id"], left="unread")
    shown = []

    def insert(text):
        if handback == "insertion_failed":
            raise RuntimeError("transcript insertion failed")
        shown.append(text)

    text = "New owner words only" if handback == "request_only" else _findings_text(record, blocking=False)
    if handback == "insertion_failed":
        with pytest.raises(RuntimeError, match="transcript insertion failed"):
            _show(ctx, insert, text, record, "findings")
        assert not getattr(ctx, "_result_review_shown", None)
    else:
        assert _show(ctx, insert, text, record, "findings", outcome=handback == "whole")
        assert shown == [text]
    assert _duties(tmp_path) == [(record["record_id"], "unread")], "insertion alone is not a saved receipt"
    flush_content_acks(ctx)
    saved = review_ledger.load_record(tmp_path, record["record_id"])
    assert bool((saved.get("waiting") or {}).get("seen_outcome_states")) is (handback == "whole")
    assert _duties(tmp_path) == ([] if handback == "whole" else [(record["record_id"], "unread")])


def test_presence_speech_and_an_explicit_pending_finish_are_never_held_for_a_pass(tmp_path, monkeypatch):
    import ouroboros.tools.review_change as review_change_mod

    presence = _floor_ctx(tmp_path / "presence", monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    # The host stamps a Presence turn before any review (``presence_runner._build_task``).
    presence.task_metadata = {**dict(getattr(presence, "task_metadata", None) or {}),
                              "presence_event_identity": "event-1"}
    trace, shown = {"tool_calls": []}, []
    assert result_review_at_delivery(presence, ANSWER, trace, shown.append) is False and shown == []
    assert len(sends) == 3 and trace["review_runs"][0]["aggregate_signal"] == "PASS"

    pending = _floor_ctx(tmp_path / "pending", monkeypatch)
    monkeypatch.setattr(review_change_mod, "run_review_change",
                        lambda ctx, **kwargs: {"state": "pending", "record_id": "rl-running"})
    assert result_review_at_delivery(pending, ANSWER, {"tool_calls": []}, shown.append) is True
    assert "still running" in shown[-1]
    pending._acceptance_pending_review_choice = "finish"
    assert result_review_at_delivery(pending, ANSWER, {"tool_calls": []}, shown.append) is False


@pytest.mark.parametrize("seat, received", [("[]", True), (CRITICAL, True), ("[]", False)])
def test_presence_speech_revised_after_its_review_buys_no_second_panel_once_its_author_received_it(
        tmp_path, monkeypatch, seat, received):
    """Astra, 9 Nov 2026 (owner, 8 Nov 2026: after received feedback Advisory re-review is Ouroboros's choice): a
    Presence turn that produced a registered file, called the result review and received it whole, then revised its
    answer, speaks at once with that review bound as the earlier version's and its own acceptance beside it: an
    earlier PASS reads Done with that label, a FAIL Done with warnings, and no second panel is bought. A review its
    author never received gives no such choice, so the Required floor still reviews the revision, holding nothing."""
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.outcomes import _review_axis
    from ouroboros.review_waiting import outcome_states
    from ouroboros.tools.review_change import run_review_change

    ctx = _floor_ctx(tmp_path, monkeypatch)
    ctx.task_metadata = {**dict(getattr(ctx, "task_metadata", None) or {}), "presence_event_identity": "event-1"}
    store_task_artifact_bytes(ctx.drive_root, ctx.task_id, "report.txt", b"The measured length is 2 units.\n")
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": seat}))
    first = run_review_change(ctx, subject="result", answer="The report gives the length as 2.")
    record = review_ledger.load_record(ctx.drive_root, first["record_id"])
    receipt = {"record_id": first["record_id"], "outcome_states": outcome_states(record)}  # the tool's whole return
    trace = {"tool_calls": [{"tool": "review_change", "tool_result_meta": {"review_delivery": receipt}}] if received else []}
    ctx._completion_selected = {"action": "finish", "answer": "The report gives the length as 2 units.",
                                "rationale": "Added the unit; the report and its conclusion are unchanged."}
    shown = []
    assert result_review_at_delivery(ctx, "The report gives the length as 2 units.", trace, shown.append) is False
    assert shown == [] and len(sends) == (3 if received else 6), "Presence speech is never held"
    review = _review_axis(trace)
    if not received:
        assert review.get("author_finish") is None, "an unreceived review is no author's acceptance"
        return
    [run] = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    assert run["earlier_revision"] is True and run["reviewed_subject_sha"] == record["subject"]["diff_sha"]
    assert run["informed_finish"]["outcomes"] == sorted(outcome_states(record))
    assert review["author_finish"] == ("previous_revision_accepted" if seat == "[]" else "author_finish")
    assert review["acceptance_decision"]["author_disposition"]["rationale"].startswith("Added the unit")


# --- The catalog's whole roster: a receipt travels in a result's bounded metadata --------------------------

def _any_seat_seam(sends, answers=None, *, hold=None):
    """One seat's physical send for any catalog id (the golden seam serves t1, t2 and s1 only): the real paid
    stamp, producer trail and finalized outcome; the seat answers ``answers[slot_id]``, else clean. ``hold``:
    a ``(seat, event)`` whose seat stays in flight until the event is set."""
    from dataclasses import asdict

    from ouroboros.observability import persist_call
    from ouroboros.review_custody import finalize_review_actor
    from ouroboros.review_dispatch import invoke_review_paid_stamp, review_operation_binding

    def run_slot(self, request, slot, *, operation_id="", retry_state=None, logical_deadline_monotonic=None,
                 pending_invocation_checkpoint=None):
        invoke_review_paid_stamp(self._review_paid_stamp)
        call_id, call_type = str(operation_id), request.call_type or f"{request.surface}_review"
        binding = review_operation_binding(request, slot, call_id)
        sends.append(slot.slot_id)
        if hold and slot.slot_id == hold[0]:
            assert hold[1].wait(60), "the held seat was never released"
        actor = self._error_actor(request, slot, "unused", operation_id=call_id)
        actor.status, actor.error, actor.raw_text = "ok", "", (answers or {}).get(slot.slot_id, "[]")
        actor.usage = {"cost": 0.0, "prompt_tokens": 2, "completion_tokens": 1}
        persist_call(self._custody_drive_root(), task_id=request.task_id or "review", call_id=f"{call_id}_prompt",
                     call_type=f"{call_type}_prompt", payload={"request": asdict(request), "slot": asdict(slot)},
                     manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model,
                               "review_operation_binding": binding})
        actor.recovery_binding = {**binding, "pending_invocation_id": ""}
        finalize_review_actor(actor, operation_id=call_id)
        actor.response_ref = self._persist_producer_outcome(
            request, actor, call_id, call_type, {"message": {"content": actor.raw_text}, "usage": actor.usage})
        return actor

    return run_slot


def _roster(monkeypatch, count, *, added=0):
    """``count`` enabled rows with the longest ids the catalog admits, each answering one remark of its own; the
    last ``added`` are not marked for the pool. Returns the ids and the seats' answers."""
    ids = [f"seat-{index:02d}-".ljust(64, "x") for index in range(count)]
    set_review_pool(monkeypatch, pool_roster(*(
        pool_seat(seat, "openai/gpt-5.6-sol", effort="high", marked=index < count - added)
        for index, seat in enumerate(ids))))
    return ids, {seat: json.dumps([{"item": "result_detail", "verdict": "FAIL", "severity": "advisory",
                                    "reason": f"REMARK-{index:02d}"}]) for index, seat in enumerate(ids)}


def _names_every_seat(text, ids):
    return [seat for index, seat in enumerate(ids) if seat not in text or f"REMARK-{index:02d}" not in text]


@pytest.mark.parametrize("count", [10, MAX_CONFIGURED_SUBAGENTS])
def test_a_supported_roster_the_tool_returned_whole_delivers_without_a_second_turn(tmp_path, monkeypatch, count):
    """Real publication, transcript and finish at a marked roster the catalog admits, ids at their longest: the
    receipt stays inside the result's metadata, every seat's remark reaches Main in the tool's own result, the
    delivery cursor advances for every seat, and nothing is bought twice."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    ids, answers = _roster(monkeypatch, count)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _any_seat_seam(sends, answers))

    text, trace, requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER}, "review"), _finish()])

    [call] = [row for row in trace["tool_calls"] if row["tool"] == "review_change"]
    assert not call["is_error"], call["result"]
    [delivered] = [row["content"] for row in requests[1] if row.get("tool_call_id") == "review"]
    assert _names_every_seat(delivered, ids) == []
    assert text == ANSWER and len(requests) == 2 and _holds(ctx) == [], "received whole: no second turn"
    assert sorted(sends) == ids, "one paid send per seat"
    receipt = call["tool_result_meta"]["review_delivery"]
    assert len(receipt["outcomes"]) == len(receipt["outcome_states"]) == count
    state = ctx._review_wait_selection
    assert sorted(state["seen_outcomes"]) == receipt["outcomes"]
    # The cursor names these outcomes as a wait's readiness does: a wait selecting them finds nothing new.
    record = review_ledger.load_record(review_ledger.ledger_root(ctx), receipt["record_id"])
    state["selection"] = {"record_id": record["record_id"], "reviewers": [
        {"wave_id": record["review_wave_id"], "operation_id": row["operation_id"], "slot_id": row["seat_id"]}
        for row in record["rows"]]}
    assert waiting.ready_helpers(ctx, state) == []


@pytest.mark.parametrize("count", [10, MAX_CONFIGURED_SUBAGENTS])
def test_a_supported_roster_no_tool_returned_is_handed_back_with_every_seats_remark(tmp_path, monkeypatch, count):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    ids, answers = _roster(monkeypatch, count)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _any_seat_seam(sends, answers))

    text, _trace, requests = _main(ctx, monkeypatch, [{"content": ANSWER}, _finish()])

    [handback] = [notice for notice in _host_notices(requests[1]) if notice.startswith("[Result review]")]
    assert _names_every_seat(handback, ids) == [] and ": PASS." in handback
    assert text == ANSWER and len(requests) == 2 and sorted(sends) == ids
    assert [row[2:4] for row in _holds(ctx)] == [("result_review", "pass")]


def test_one_added_seats_new_outcome_in_a_full_roster_is_handed_back_after_the_tool_returned_it(tmp_path, monkeypatch):
    """A full catalog with one added reviewer, returned whole by the real tool: received, a cost-only revision
    stays received, and one later answer of the added seat's same operation is a new outcome, handed back."""
    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    ids, answers = _roster(monkeypatch, MAX_CONFIGURED_SUBAGENTS, added=1)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _any_seat_seam(sends, answers))
    _text, trace, requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER, "reviewers": ids,
                                "reason": "every marked seat and an added eye"}, "review"), _finish()])
    [call] = [row for row in trace["tool_calls"] if row["tool"] == "review_change"]
    assert not call["is_error"] and len(requests) == 2 and sorted(sends) == ids, call["result"]
    record_id, root = call["tool_result_meta"]["review_delivery"]["record_id"], review_ledger.ledger_root(ctx)
    record = review_ledger.load_record(root, record_id)
    assert record["panel"]["additional"] == [ids[-1]]
    assert [item["seat_id"] for item in record["verdict"]["additional_findings"]] == [ids[-1]]
    ctx._result_review_shown = set()
    shown = []

    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is False, "received through the tool"
    review_ledger.revise_record(root, record_id, lambda r: {**r, "cost": {**(r.get("cost") or {}), "usd": 0.42}})
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is False and shown == []

    _revise_seat(root, record_id, ids[-1], outcome={"status": "interrupted"}, finding="ADDED-LATE-REMARK",
                 bucket="additional_findings")
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is True
    assert "ADDED-LATE-REMARK" in shown[0] and "heard, not counted" in shown[0]
    assert result_review_at_delivery(ctx, ANSWER, trace, shown.append) is False and len(sends) == len(ids)


def test_a_wait_that_selected_a_whole_roster_never_loses_a_remark_or_buys_twice(tmp_path, monkeypatch):
    """Selected readiness beside the review's own result: publication holds with both in one receipt, every
    seat's remark reaches Main before its answer is delivered, and each seat was paid once."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    ids, answers = _roster(monkeypatch, MAX_CONFIGURED_SUBAGENTS)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _any_seat_seam(sends, answers))

    text, trace, requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER, "wake_on": {"reviewers": ids}}, "review"),
        *[_finish() for _ in range(8)]])

    assert not [row["result"] for row in trace["tool_calls"] if row["tool"] == "review_change" and row["is_error"]]
    assert text == ANSWER and sorted(sends) == ids
    assert _names_every_seat(json.dumps(requests[-1]), ids) == [], "all of it before the delivered decision"
    state = ctx._review_wait_selection
    assert len(state["selection"]["reviewers"]) == len(ids) and waiting.ready_helpers(ctx, state) == [], \
        "what the result carried is never offered to the wait again"


def _interactive_author(tmp_path, monkeypatch):
    """A Required root author as the agent binds one: a continuation owner and a task frame, so it can leave a
    review wait (a selected outcome, a wake time, an addressed message)."""
    from ouroboros import review_pause
    from ouroboros.owner_wait import direct_owner_wait
    from ouroboros.task_results import write_task_result

    ctx = _floor_ctx(tmp_path, monkeypatch)
    ctx.owner_wait_callback, ctx.task_lifecycle_bound, ctx.task_attempt = direct_owner_wait, True, 1
    ctx.current_task_type, ctx._loop_mailbox_seen_ids = "task", set()
    write_task_result(ctx.drive_root, ctx.task_id, "running", chat_id=7)
    monkeypatch.setattr(review_pause, "DETACH_POLL_SEC", 0.05)
    _produced_answer_protocol(monkeypatch)
    return ctx


def _main_in_task_frame(ctx, monkeypatch, responses, release):
    """``_main`` under the task's real model-wait owner; every held seat is released and finished at the end."""
    from ouroboros import model_wait, review_operation

    try:
        with model_wait.task_model_wait_scope(task={"id": ctx.task_id, "chat_id": 7, "_attempt": 1},
                                              drive_root=ctx.drive_root, event_queue=queue.Queue(),
                                              worker_slot_held=False) as owner:
            ctx.model_wait_context, owner.tool_context = owner, ctx
            return _main(ctx, monkeypatch, responses)
    finally:
        release.set()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and [op for op in review_operation._LIVE.values() if op.task_id == ctx.task_id]:
            time.sleep(0.05)  # the seats finish under this test's fakes


def test_a_selected_seat_ending_the_wait_returns_part_of_a_roster_and_the_rest_is_handed_back(tmp_path, monkeypatch):
    """One seat still runs when a selected seat ends the tool's wait: the tool returns the roster so far, its
    receipt names only those outcomes, and the settled outcome as a whole has not reached Main. The running seat
    finishes unpaid-for again, and Main receives every remark before its answer is delivered."""
    ctx = _interactive_author(tmp_path, monkeypatch)
    ids, answers = _roster(monkeypatch, MAX_CONFIGURED_SUBAGENTS)
    release, sends = threading.Event(), []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _any_seat_seam(sends, answers, hold=(ids[-1], release)))

    def released_finish():
        release.set()
        return _finish()

    text, trace, requests = _main_in_task_frame(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER, "wake_on": {"reviewers": [ids[0]]}}, "review"),
        released_finish, *[_finish() for _ in range(8)]], release)

    [call, *_later] = [row for row in trace["tool_calls"] if row["tool"] == "review_change"]
    assert not call["is_error"], call["result"]
    receipt = call["tool_result_meta"]["review_delivery"]
    assert len(receipt["outcome_states"]) < len(ids), "the tool returned before the wave settled"
    [returned] = [row["content"] for row in requests[1] if row.get("tool_call_id") == "review"]
    assert "[Selected results]" in returned and f"REMARK-{len(ids) - 1:02d}" not in returned
    assert text == ANSWER and sorted(sends) == ids, "the running seat was collected, never sent again"
    marker = "[Result review] Your result was reviewed before delivery"
    [handback] = [str(row["content"]) for row in requests[-1] if row.get("role") == "user" and marker in str(row["content"])]
    assert _names_every_seat(handback, ids) == [], "the whole outcome, not only the seats still unseen"
    assert ": PASS." in handback


def test_a_change_review_and_the_commit_door_publish_a_whole_roster_through_the_same_receipt(tmp_path, monkeypatch):
    """The receipt's owner serves every subject: a change review of a whole roster returns to its author, and the
    commit door publishes that record's receipt the same way."""
    from ouroboros.tools.review_collection import publish_commit_delivery
    from ouroboros.tools.tool_result import (
        _install_tool_result_sidecar,
        _published_tool_result,
        _restore_tool_result_sidecar,
    )
    from tests import _contributor_packet_shared as shared

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    ids, _answers = _roster(monkeypatch, MAX_CONFIGURED_SUBAGENTS)
    (ctx.workspace_root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    shared.git(ctx.workspace_root, "add", "app.py")
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _any_seat_seam(sends))

    text, trace, requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "index", "goal": "Bump the value", "scope": "app.py"}, "review"),
        _finish("Reviewed.")])

    [call] = [row for row in trace["tool_calls"] if row["tool"] == "review_change"]
    assert not call["is_error"], call["result"]
    assert text == "Reviewed." and len(requests) == 2 and sorted(sends) == ids
    receipt = call["tool_result_meta"]["review_delivery"]
    assert len(receipt["outcomes"]) == len(ids)
    ctx._current_review_record_id = receipt["record_id"]
    sidecar = _install_tool_result_sidecar(ctx, object())
    try:
        assert publish_commit_delivery(ctx, "Committed.").startswith("Committed.")
        door = _published_tool_result(ctx, None).meta["review_delivery"]
    finally:
        _restore_tool_result_sidecar(sidecar)
    assert door["outcomes"] == receipt["outcomes"], "the wait's cursor as before"
    assert door["outcome_states"] == [], "a commit's own report renders no ledger rows: it certifies none"


# --- A receipt names only what its own response carries ------------------------------------------------

LATE = "LATE-SEAT-FINDING: the second figure is not checked."


def _late_answer(kind):
    return json.dumps([{"item": "late_detail", "verdict": "FAIL", "severity": kind, "reason": LATE}])


def _handed_back(request):
    """The result review's own handback in one model request, never another context row that quotes findings."""
    marker = "[Result review] Your result was reviewed before delivery"
    return [str(row["content"]) for row in request if row.get("role") == "user" and marker in str(row.get("content"))]


def _settled(ctx, record_id, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        record = review_ledger.load_record(review_ledger.ledger_root(ctx), record_id) or {}
        if record.get("state") == "settled":
            return record
        time.sleep(0.02)
    raise AssertionError(f"record {record_id} never settled")


@pytest.mark.parametrize("kind, aggregate", [("advisory", "PASS"), ("critical", "FAIL")])
def test_a_seat_settling_after_the_tool_rendered_its_result_is_handed_back_before_delivery(
        tmp_path, monkeypatch, kind, aggregate):
    """The tool's wait ends when the seat it selected is ready, while the last seat still runs; that seat settles
    after the tool rendered its result and before the receipt is published (the controlled boundary). The receipt
    names exactly the rows the text rendered, so under Advisory the late finding reaches Main in the handback
    before its answer is delivered: an advisory remark on the PASS, and a critical FAIL before any informed finish."""
    import ouroboros.tools.review_change as review_change_mod

    ctx = _interactive_author(tmp_path, monkeypatch)
    ids, answers = _roster(monkeypatch, 3)
    answers[ids[-1]] = _late_answer(kind)
    release, sends, rendered = threading.Event(), [], []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _any_seat_seam(sends, answers, hold=(ids[-1], release)))
    render = review_change_mod._handle_new_review_change

    def settle_after_render(ctx_, **kwargs):
        text = render(ctx_, **kwargs)  # the result as the tool returns it, rendered while the last seat runs
        rendered.append(text)
        release.set()
        _settled(ctx_, json.loads(text[text.index("{"):])["record_id"])
        return text  # publication follows, after the settlement

    monkeypatch.setattr(review_change_mod, "_handle_new_review_change", settle_after_render)

    text, trace, requests = _main_in_task_frame(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER, "wake_on": {"reviewers": [ids[0]]}}, "review"),
        *[_finish() for _ in range(6)]], release)

    [returned] = rendered
    payload = json.loads(returned[returned.index("{"):])
    assert payload["state"] == "pending" and LATE not in returned, "rendered while the late seat ran"
    assert text == ANSWER and sorted(sends) == ids, "no seat was sent twice"
    [run] = trace["review_runs"]
    assert run["aggregate"] == aggregate
    assert [row[2:4] for row in _holds(ctx)] == [("result_review", "pass" if aggregate == "PASS" else "findings")], \
        "the late finding must be handed back before the delivered decision"
    [handback] = _handed_back(requests[-1])
    assert LATE in handback and "REMARK-00" in handback, "the whole outcome, the late finding with it"
    assert bool(run.get("informed_finish")) is (aggregate == "FAIL"), "an informed finish only after it was shown"
    [call] = [row for row in trace["tool_calls"] if row["tool"] == "review_change"]
    record = review_ledger.load_record(review_ledger.ledger_root(ctx), payload["record_id"])
    carried = waiting.outcome_states({**record, "rows": payload["rows"]})
    assert call["tool_result_meta"]["review_delivery"]["outcome_states"] == carried, "exactly the rows it rendered"
    late = waiting.outcome_states(record)[[row["seat_id"] for row in record["rows"]].index(ids[-1])]
    assert late not in carried, "the late seat's settled state is not certified"


def test_a_collected_review_that_changed_after_its_collection_rendered_is_handed_back(tmp_path, monkeypatch):
    """Reference collection: the record gains a later answer of the same operation after the collector rendered
    it and before the receipt is published. That answer was not in the text, so it is not received."""
    import ouroboros.tools.review_collection as review_collection

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    collect, rendered = review_collection.collect_change, []

    def change_after_render(ctx_, reference, continue_seats=None):
        assert continue_seats is None  # this is an ordinary free collection, not a continuation choice
        payload = collect(ctx_, reference, continue_seats)
        rendered.append(json.dumps(payload))
        _revise_seat(review_ledger.ledger_root(ctx_), payload["record_id"], row={"status": "completed"},
                     outcome={"status": "interrupted"}, finding=LATE)  # a later answer of the same operation
        return payload

    monkeypatch.setattr(review_collection, "collect_change", change_after_render)

    def collect_the_reference():
        root = review_ledger.ledger_root(ctx)
        [row] = review_ledger.recent_records(root, ctx.task_id, limit=5, surface="result")
        reference = review_collection.reference_for(review_ledger.load_record(root, row["record_id"]))
        return _call("review_change", {"review_reference": reference}, "collect")

    text, trace, requests = _main(ctx, monkeypatch, [
        _call("review_change", {"subject": "result", "answer": ANSWER}, "review"),
        collect_the_reference, *[_finish() for _ in range(4)]])

    [returned] = rendered
    assert LATE not in returned and '"state": "settled"' in returned, "the collection rendered the record before"
    [collected] = [row["content"] for row in requests[2] if row.get("tool_call_id") == "collect"]
    assert LATE not in collected and len(sends) == 3, "collected at $0; the later answer is not in its text"
    [handback] = _handed_back(requests[-1])
    assert text == ANSWER and LATE in handback, "the later answer was handed back before delivery"
    assert [row[2:4] for row in _holds(ctx)] == [("result_review", "pass")]
