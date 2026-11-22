"""Reported amounts retain their knowledge independently from reviewer lifecycle."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger
from ouroboros.tools.review_change import review_result
from ouroboros.tools.review_response import parse_model_response
from ouroboros.triad_review import _actor_record
from tests.test_review_ledger import _facts


def _through_actor(status, operation_state, usage, *, headers=None):
    result = {
        "slot_id": "seat", "operation_state": operation_state, "usage": usage,
        "choices": [{"message": {"content": "[]\nNO_FINDINGS"}}],
    }
    if status == "error":
        result["error"] = "The answer could not be observed."
    envelope = parse_model_response("example/model", result, headers)
    return _actor_record(
        envelope, idx=0, model_label="example/model", status=status,
        raw_text=envelope["text"],
    ).to_dict()


@pytest.mark.parametrize(("amount", "status", "state"), [
    (None, "error", "in_flight"),
    (None, "responded", "late_settled"),
    (0.0, "error", "custody_lost"),
    (0.0, "pending", "in_flight"),
    (1.25, "responded", "settled"),
])
def test_amount_survives_response_actor_record_and_tool(status, state, amount):
    raw = _through_actor(status, state, {"cost": amount})
    assert raw["cost_usd"] == amount
    record = review_ledger.build_commit_gate_record(_facts([raw])).to_dict()
    assert record["rows"][0]["usd"] == amount
    expected = {"usd": amount if amount is not None else 0.0, "unknown": amount is None}
    assert record["cost"] == expected
    assert review_result(record, reused=False)["cost"] == expected
    assert record["rows"][0]["operation_state"] == state


@pytest.mark.parametrize("status", ["error", "responded"])
def test_estimated_or_missing_amount_is_not_promoted_to_reported_money(status):
    for usage in ({}, {"cost": None, "cost_disclosed_usd": 3.0, "cost_estimated": True},
                  {"cost": 3.0, "cost_estimated": True}):
        raw = _through_actor(status, "settled", usage)
        assert raw["cost_usd"] is None
    assert _through_actor(status, "settled", {"total_cost": 0.75})["cost_usd"] == 0.75
    assert _through_actor(status, "settled", {}, headers={"X-OpenRouter-Cost": "0"})["cost_usd"] == 0.0


def test_missing_amount_changes_cost_knowledge_not_verdict_or_free_reuse(tmp_path):
    known = _through_actor("responded", "settled", {"cost": 1.25})
    absent = _through_actor("responded", "settled", {"cost": None})
    absent["slot_id"] = "other-seat"
    record = review_ledger.build_commit_gate_record(_facts([known, absent])).to_dict()
    assert record["cost"] == {"usd": 1.25, "unknown": True}
    absent["cost_usd"] = 0.0
    priced = review_ledger.build_commit_gate_record(_facts([known, absent])).to_dict()
    assert record["verdict"] == priced["verdict"]
    assert priced["cost"] == {"usd": 1.25, "unknown": False}
    written = review_ledger.write_record(tmp_path, review_ledger.ReviewLedgerRecord.from_dict(record))
    assert review_ledger.load_record(tmp_path, record["record_id"])["cost"] == record["cost"]
    assert review_ledger.recent_records(tmp_path, "task-1", 1)[0]["cost"] == record["cost"]
    assert review_result(written, reused=True)["cost"] == {"usd": 0.0, "unknown": False}


@pytest.mark.parametrize("first_final", [True, False])
def test_packet_retry_books_the_empty_reply_and_the_answer_once(tmp_path, monkeypatch, first_final):
    from ouroboros import usage_accounting as ua, usage_store
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    sent, read_back = [], []
    attempt = usage_store.Txn.attempt

    def read_attempt(txn, attempt_id):
        if not txn.writable:
            read_back.append(attempt_id)
        return attempt(txn, attempt_id)

    monkeypatch.setattr(usage_store.Txn, "attempt", read_attempt)

    class LedgerChat:
        def chat(self, **kwargs):
            first = not sent
            text, cash, final = ("", 0.125, first_final) if first else ("[]\nNO_FINDINGS", 0.0, True)
            message = {"role": "assistant", "content": text}
            usage = {"cost": cash, "cost_final": final, "cost_estimated": not final}
            with ua.capture_attempt_ids() as ids:
                ua.execute_physical_attempt(ua.AttemptRequest(model=kwargs["model"], provider="openai-compatible"),
                    lambda: message, extractor=lambda _: (usage, cash, final))
            # A later carrier can name an earlier attempt again; it is still one bill.
            usage["ledger_attempt_ids"] = [*sent, *ids]
            sent.extend(ids)
            return message, usage

    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, non_task_operation=True)):
        run = run_review_request(ReviewRequest(surface="multi_model_review", goal="Review the fixture"),
                                 slots=[ReviewSlot("seat", "example/model")], drive_root=tmp_path, llm=LedgerChat())
    actor, = run.actors
    assert actor["status"] == "ok" and len(sent) == 2
    assert read_back == ([] if first_final else sent[:1])
    usage = actor["usage"]
    assert usage["ledger_attempt_ids"] == sent == actor["outcome"]["usage_ref"]["ledger_attempt_ids"]
    assert (usage["cost"], usage["cost_partial"], usage["cost_estimated"], usage["cost_final"]) == (
        0.125, False, not first_final, first_final)
    raw = _through_actor("responded", "settled", usage)
    record = review_ledger.build_commit_gate_record(_facts([raw])).to_dict()
    assert record["rows"][0]["usd"] == (0.125 if first_final else None)
    assert record["cost"]["unknown"] is (not first_final)


@pytest.mark.parametrize("prior,release_read", [
    ("", True), ("", False), ("ledger", True), ("capture", True), ("reply", True),
])
def test_released_review_hold_keeps_its_id_without_inventing_a_send(tmp_path, monkeypatch, prior, release_read):
    from ouroboros import delegate_custody_usage as usage, subagent_metric_events as metrics
    from ouroboros.review_execution_projection import review_executions_from_actor_usage
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    receipts = {}
    if prior == "ledger":
        receipts["earlier"] = {"state": "settled", "cost_usd": 0.0, "cost_final": True}
    elif prior == "capture":
        receipts["earlier"] = None
    receipts["hold"] = {"state": "released"} if release_read else None
    monkeypatch.setattr(usage, "_review_attempt_rows", lambda ids: {key: receipts[key] for key in ids})
    refusal = RuntimeError("not sent")
    refusal.physical_attempt_capture = SimpleNamespace(attempt_id="hold", state="released")
    refusal.ledger_attempt_ids = list(receipts) if prior == "ledger" else ["hold"]
    events = []
    costs = usage.ReviewAttemptCosts()
    if prior == "capture":
        earlier = RuntimeError("send failed")
        earlier.physical_attempt_capture = SimpleNamespace(attempt_id="earlier", state="unresolved")
        costs.failed(usage.observe_failed_review_send(events.append, earlier))
    elif prior == "reply":
        costs.observe({}, None)  # a legacy adapter can return before a later proven-unsent hold
    costs.failed(usage.observe_failed_review_send(events.append, refusal))
    fact = costs.project({"physical_attempt_state": "released"})
    assert fact["ledger_attempt_ids"] == list(receipts)
    priced = prior == "ledger"
    assert fact["cost"] == (0.0 if priced else None) and fact["cost_final"] is priced
    assert [row["ledger_attempt_ids"] for row in events] == ([["earlier"]] if "earlier" in receipts else [])

    assert review_executions_from_actor_usage([{"usage": fact}]) == ([{"kind": "api"}] if prior else [])
    native = {**fact, "delivery": "native_tool_rounds", "native_rounds": 0}
    assert review_executions_from_actor_usage([{"usage": native}]) == ([{"kind": "native"}] if prior else [])

    emitted = []
    monkeypatch.setattr(metrics, "publish", lambda _root, row: emitted.append(row))
    actor = SimpleNamespace(usage=fact, operation_id="operation", raw_text="", duration_sec=0,
                            failure_code="", awaiting_since="", outcome={"outcome_id": "outcome",
                            "operation_id": "operation", "status": "error",
                            "usage_ref": {"ledger_attempt_ids": list(receipts)}})
    metrics.review_execution(ReviewRequest(surface="plan_review", goal="Review the fixture"),
                             ReviewSlot("seat", "example/model"), actor, None, fallback_root=tmp_path)
    assert emitted[0]["dispatched"] is (True if prior else None)
    assert emitted[0]["actor"]["usage_ref"]["ledger_attempt_ids"] == list(receipts)


@pytest.mark.parametrize("cash,final,partial,estimated", [
    (0.0, True, False, False), (0.125, False, False, True),
    (0.125, False, True, False), (None, False, False, False),
])
def test_read_failure_amount_and_finality_reach_the_usage_event(tmp_path, monkeypatch, cash, final, partial, estimated):
    from ouroboros import delegate_custody_usage as usage
    from ouroboros.tools.review_helpers import emit_review_usage
    from supervisor.events_budget import _handle_llm_usage

    reads = []

    def read(ids):
        reads.extend(ids)
        return {key: {"state": "settled", "cost_usd": cash, "cost_final": final, "cost_partial": partial,
                      "cost_estimated": estimated, "model": "example/model", "provider": "example"} for key in ids}

    monkeypatch.setattr(usage, "_review_attempt_rows", read)
    error = RuntimeError("no review answer")
    error.ledger_attempt_ids = ["failed"]
    ctx = SimpleNamespace(task_id="fixture-review", pending_events=[])
    costs = usage.ReviewAttemptCosts()
    costs.failed(usage.observe_failed_review_send(
        lambda row: emit_review_usage(ctx, model=row["resolved_model"], usage=row, source="review"), error))
    seat = costs.project({})
    assert seat["cost"] == cash and seat["cost_final"] is final and seat["cost_estimated"] is estimated
    assert reads == ["failed"]

    event, = ctx.pending_events
    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=tmp_path))
    persisted = json.loads((tmp_path / "logs" / "events.jsonl").read_text())
    assert persisted["ledger_attempt_ids"] == ["failed"]
    assert (persisted["cost"], persisted["cost_known"], persisted["cost_final"], persisted["cost_estimated"]) == (
        cash, cash is not None, final, estimated)


@pytest.mark.parametrize("read", [True, False])
def test_a_failed_send_names_each_attempt_by_its_own_receipt_or_capture(monkeypatch, read):
    """The capture speaks for the attempt it captured, never for another id the failure names."""
    from ouroboros import delegate_custody_usage as usage

    rows = {"earlier": {"state": "settled", "cost_usd": 0.0, "cost_final": True, "model": "earlier/model",
                        "provider": "earlier"}, "failed": {"state": "unresolved"}}
    monkeypatch.setattr(usage, "_review_attempt_rows", lambda ids: {key: rows[key] if read else None for key in ids})
    error = RuntimeError("no review answer")
    error.ledger_attempt_ids = ["earlier", "failed"]
    error.physical_attempt_capture = SimpleNamespace(attempt_id="failed", state="unresolved", model="failed/model",
                                                     provider="failed")
    events = []
    usage.observe_failed_review_send(events.append, error)
    unknown = {"cost": None, "cost_final": False, "cost_estimated": False}
    assert events == [
        {"resolved_model": "earlier/model", "provider": "earlier", "ledger_attempt_ids": ["earlier"],
         "cost": 0.0, "cost_final": True, "cost_estimated": False} if read else
        {"resolved_model": None, "provider": None, "ledger_attempt_ids": ["earlier"]},  # unread: neither price nor route
        {"resolved_model": "failed/model", "provider": "failed", "ledger_attempt_ids": ["failed"],
         **(unknown if read else {})},
    ]


@pytest.mark.parametrize("sent_provider", ["compatible-example", None], ids=["known-route", "unknown-route"])
def test_failed_physical_route_and_reported_zero_reach_error_row_and_card(monkeypatch, sent_provider):
    from ouroboros import delegate_custody_usage as usage, reviewer_slot_config as cfg
    from ouroboros.review_projection import ledger_record_panel
    from ouroboros.review_substrate import ReviewActorRecord, ReviewSlot

    receipts = {
        "earlier": {"state": "settled", "model": "earlier/model", "provider": "earlier", "cost_usd": 0.0, "cost_final": True},
        "sent": {"state": "settled", "cost_usd": 0.0, "cost_final": True},
        "hold": {"state": "released"},
    }
    reads, events = [], []

    def read(ids):
        reads.extend(ids)
        return {key: receipts[key] for key in ids}

    monkeypatch.setattr(usage, "_review_attempt_rows", read)
    error = RuntimeError("synthetic failed review call")
    error.ledger_attempt_ids = ["earlier", "sent", "hold"]
    error.physical_attempt_capture = SimpleNamespace(attempt_id="sent", state="settled", model="sent/model",
                                                     provider=sent_provider)
    costs = usage.ReviewAttemptCosts()
    costs.failed(usage.observe_failed_review_send(events.append, error))
    fact = costs.project({"resolved_model": "stale/model", "provider": "stale"}, failed=True)
    assert fact == {"resolved_model": "sent/model", "provider": sent_provider, "physical_dispatch_observed": True,
                    "ledger_attempt_ids": ["earlier", "sent", "hold"], "cost": 0.0, "cost_final": True,
                    "cost_partial": False, "cost_estimated": False}
    assert [(e["resolved_model"], e["provider"], e["cost"]) for e in events] == [
        ("earlier/model", "earlier", 0.0), ("sent/model", sent_provider, 0.0)]
    assert reads == ["earlier", "sent", "hold"]

    configured = "example/model"
    requested = {"slot_id": "seat", "model": configured, "route": "api_chat", "parts": ["change"]}
    envelope = parse_model_response(configured, {"slot_id": "seat", "usage": fact, "error": "unavailable"}, None)
    assert (envelope["request_model"], envelope["model"], envelope["provider"], envelope["cost_estimate"]) == (
        configured, "sent/model", sent_provider or "unknown", 0.0)
    actor = ReviewActorRecord("seat", configured, "error", usage=fact)
    executions = cfg.reviewer_slot_execution_rows("commit_gate", [actor], {"seat": ReviewSlot("seat", configured)})
    raw = _through_actor("error", "settled", fact)
    record = review_ledger.build_commit_gate_record(
        _facts([raw], rows=[requested], slot_executions=executions)).to_dict()
    [card] = ledger_record_panel(record)["actors"]
    assert card["requested_model"] == configured and card["executions"] == [{"kind": "api", "model": "sent/model"}]
    assert card["provider"] == (sent_provider or "unknown") and record["rows"][0]["usd"] == 0.0


_SWITCHED = {"resolved_model": "anthropic/claude-opus-5.5", "provider": "openrouter"}


@pytest.mark.parametrize(("served", "cash", "kind", "card_cost"), [
    (_SWITCHED, 0.42, "measured", {"kind": "usd", "usd": 0.42}),
    (_SWITCHED, 0.0, "measured", {"kind": "usd", "usd": 0.0}),  # a measured zero is a price
    (_SWITCHED, None, "unknown", {"kind": "unknown", "usd": None}),  # an unconfirmed one is neither $0 nor a subscription
    ({}, 0.42, "subscription", {"kind": "subscription", "usd": None}),  # the reply names no route: the requested row decides
], ids=["charged", "measured-zero", "unconfirmed", "route-not-named"])
def test_a_reply_served_on_another_route_keeps_its_charge_beside_the_requested_row(tmp_path, monkeypatch, served, cash, kind, card_cost):
    """A subscription row whose answer the model wait served on an API route: Settings, the review
    record and the card name the route that answered and its money; the configured row stays the
    request. The route comes from the reply's own usage, never from the amount."""
    from ouroboros import reviewer_slot_config as cfg
    from ouroboros.review_projection import ledger_record_panel
    from ouroboros.review_substrate import ReviewActorRecord, ReviewSlot

    monkeypatch.setattr(cfg, "_last_execution_path", lambda: tmp_path / "last.json")
    configured = "claudexor::codex=gpt-6.2-sol"
    usage = {**served, "cost": cash, "cost_final": cash is not None}
    actor = ReviewActorRecord("seat", configured, "ok", usage=usage)
    executions = cfg.record_reviewer_slot_executions("commit_gate", [actor], {"seat": ReviewSlot("seat", configured)})
    row = executions["seat"]
    assert row["requested"]["model"] == configured and row["requested"]["route_kind"] == "api_chat"
    assert row["effective"]["model"] == (served.get("resolved_model") or configured)
    assert row["effective"].get("provider") == served.get("provider")
    assert (row["cost_kind"], row["cost_usd"]) == (kind, cash)
    # Settings reads the same row (GET /api/review-pool ``last_executions``).
    shown = cfg.public_slot_executions(cfg.reviewer_slot_last_executions())["seat"]
    assert shown["cost"] == card_cost and shown["effective"]["model"] == row["effective"]["model"]
    assert shown["requested"]["model"] == configured

    requested = {"slot_id": "seat", "model": configured, "route": "api_chat", "parts": ["change"]}
    raw = _through_actor("responded", "settled", usage)
    record = review_ledger.build_commit_gate_record(
        _facts([raw], rows=[requested], slot_executions=executions, panel={"pool": ["seat"]})).to_dict()
    [seat] = record["rows"]
    assert seat["requested"]["model"] == configured and seat["effective"]["model"] == row["effective"]["model"]
    assert (seat["cost_kind"], seat["cost_usd"], seat["usd"]) == (kind, cash, cash)
    [card] = ledger_record_panel(record)["actors"]
    assert card["requested_model"] == configured and card["cost"] == card_cost
    assert card["executions"] == [{"kind": "api", "model": row["effective"]["model"]}]
    if served:
        assert card["provider"] == served["provider"]


def test_an_unsent_seat_stays_unexecuted_and_keeps_its_requested_row():
    """Nothing left the host: no route, model or executor is claimed, whatever a reservation recorded."""
    from ouroboros import reviewer_slot_config as cfg
    from ouroboros.review_projection import ledger_record_panel
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRouteKind, ReviewSlot

    configured = "claudexor::codex=gpt-6.2-sol"
    actor = ReviewActorRecord("seat", configured, "not_dispatched", operation_state="not_dispatched", usage={})
    session = ReviewActorRecord("held", "claude", "not_dispatched", operation_state="not_dispatched", usage={})
    slots = {"seat": ReviewSlot("seat", configured),
             "held": ReviewSlot("held", "claude", route=ReviewRouteKind.AGENT_SESSION, session_target="claude=opus")}
    executions = cfg.reviewer_slot_execution_rows("commit_gate", [actor, session], slots)
    assert executions["seat"]["effective"] == {"route": "", "model": ""}
    # Settings reads the same rows: an explicitly unsent session shows no applied route either.
    assert executions["held"]["effective"] == {"route": "", "model": ""}
    plans = [{"slot_id": "seat", "model": configured, "route": "api_chat", "parts": ["change"]},
             {"slot_id": "held", "model": "claude", "route": "agent_session", "session_target": "claude=opus", "parts": ["change"]}]
    raws = [{**_through_actor("error", "not_dispatched", {}), "slot_id": slot, "status": "not_dispatched"} for slot in ("seat", "held")]
    record = review_ledger.build_commit_gate_record(_facts(raws, rows=plans, slot_executions=executions)).to_dict()
    cards = {card["slot_id"]: card for card in ledger_record_panel(record)["actors"]}
    for slot, requested in (("seat", configured), ("held", "claude")):
        # A session start nobody observed is no executed actor either, and never an API run.
        assert cards[slot]["executions"] == [] and cards[slot]["requested_model"] == requested, slot
        assert cards[slot]["model"] == "" and cards[slot]["transport_status"] == "not_dispatched", slot


def test_single_exact_review_reply_keeps_the_no_read_fast_path(monkeypatch):
    from ouroboros import delegate_custody_usage as usage, usage_accounting as ua

    def read(ids):
        assert ids == []  # no attempt-store read back for the exact receipt
        return {}

    monkeypatch.setattr(usage, "_review_attempt_rows", read)
    monkeypatch.setattr(ua, "last_physical_attempt_capture",
                        lambda: SimpleNamespace(attempt_id="ordinary", state="settled"))
    emitted = []
    legacy = {"resolved_model": "example/model", "cost": 0.0}
    costs = usage.ReviewAttemptCosts()
    assert costs.project(legacy) is legacy
    reply = {**legacy, "ledger_attempt_ids": ["ordinary"], "cost_final": True}
    costs.observe(reply, emitted.append)
    assert emitted == [reply]
    assert costs.project(reply) == {**reply, "cost_partial": False, "cost_estimated": False}


@pytest.mark.parametrize(("receipt", "money"), [
    ({"state": "settled", "cost_usd": 0.125, "cost_final": True},
     {"cost": 0.125, "cost_final": True, "cost_estimated": False}),
    ({"state": "settled", "cost_usd": 0.0, "cost_final": True},  # a reported zero is exact money
     {"cost": 0.0, "cost_final": True, "cost_estimated": False}),
    ({"state": "settled", "cost_usd": 0.25, "cost_estimated": True},
     {"cost": 0.25, "cost_final": False, "cost_estimated": True}),
    ({"state": "settled", "cost_usd": 0.25, "cost_partial": True, "cost_estimated": False},
     {"cost": 0.25, "cost_final": False, "cost_estimated": False}),  # an exact subtotal, not final
    ({"state": "unresolved"},  # sent, its price unsettled: an explicit unknown, never a fabricated zero
     {"cost": None, "cost_final": False, "cost_estimated": False}),
    ({"state": "released"}, None),  # read as a hold that never left the host: no send is presented
    ({"state": "reserved"}, None),
])
def test_prior_attempt_receipt_price_and_route_reach_the_usage_event(monkeypatch, receipt, money):
    """A reply naming an earlier physical attempt reports that attempt's own price and route.

    One event per physical send: the earlier attempt states its own receipt's route, amount and
    finality, never the reply's, and the answer keeps the reply's. A hold that never left the host
    presents no send and keeps its id. The seat ledger reads that receipt once and the money
    projection reuses the same read instead of repeating it.
    """
    from ouroboros import delegate_custody_usage as usage, usage_accounting as ua

    reads = []

    def read(ids):
        reads.extend(ids)
        return {key: {**receipt, "model": "earlier/model", "provider": "earlier"} for key in ids}

    monkeypatch.setattr(usage, "_review_attempt_rows", read)
    monkeypatch.setattr(ua, "last_physical_attempt_capture",
                        lambda: SimpleNamespace(attempt_id="answer", state="settled"))
    emitted = []
    reply = {"resolved_model": "example/model", "provider": "example", "cost": 0.5,
             "cost_final": True, "ledger_attempt_ids": ["prior", "answer"]}
    costs = usage.ReviewAttemptCosts()
    costs.observe(reply, emitted.append)

    answer = emitted[-1]
    assert (answer["ledger_attempt_ids"], answer["resolved_model"], answer["cost"]) == (
        ["answer"], "example/model", 0.5)
    assert emitted[:-1] == ([] if money is None else [{
        "resolved_model": "earlier/model", "provider": "earlier", "ledger_attempt_ids": ["prior"], **money}])
    assert reads == ["prior"], "the earlier receipt is read exactly once"
    fact = costs.project({})
    assert fact["ledger_attempt_ids"] == ["prior", "answer"], "custody keeps every id, sent or not"
    assert fact["cost"] == sum(row["cost"] for row in emitted if row["cost"] is not None), (
        "the seat's cumulative total says the same money its events state")
    assert reads == ["prior"], "the projection reuses the event's read instead of repeating it"


def test_an_unread_prior_receipt_stays_unpriced_unrouted_and_nonfinal(monkeypatch):
    """A read failure is a gap, never a zero or a borrowed route: the earlier event states neither."""
    from ouroboros import delegate_custody_usage as usage, usage_accounting as ua

    monkeypatch.setattr(usage, "_review_attempt_rows", lambda ids: dict.fromkeys(ids))
    monkeypatch.setattr(ua, "last_physical_attempt_capture",
                        lambda: SimpleNamespace(attempt_id="answer", state="settled"))
    emitted = []
    reply = {"resolved_model": "example/model", "provider": "example", "cost": 0.5,
             "cost_final": True, "ledger_attempt_ids": ["prior", "answer"]}
    costs = usage.ReviewAttemptCosts()
    costs.observe(reply, emitted.append)

    prior, _answer = emitted
    assert prior == {"resolved_model": None, "provider": None, "ledger_attempt_ids": ["prior"]}
    fact = costs.project({})
    assert (fact["cost"], fact["cost_partial"], fact["cost_final"]) == (0.5, True, False)


def test_an_accounted_reply_keeps_its_own_route_for_a_later_reply_that_names_it(monkeypatch):
    """The no-read fact of an ordinary exact reply names that reply's route, not the later carrier's."""
    from ouroboros import delegate_custody_usage as usage, usage_accounting as ua

    monkeypatch.setattr(usage, "_review_attempt_rows",
                        lambda ids: pytest.fail(f"no read back for {ids}") if ids else {})
    last = {}
    monkeypatch.setattr(ua, "last_physical_attempt_capture", lambda: SimpleNamespace(state="settled", **last))
    emitted = []
    costs = usage.ReviewAttemptCosts()
    for attempt_id, model, cash in (("first", "first/model", 0.0), ("second", "second/model", 0.5)):
        last["attempt_id"] = attempt_id
        costs.observe({"resolved_model": model, "provider": model.partition("/")[0], "cost": cash, "cost_final": True,
                       "ledger_attempt_ids": [*(["first"] if attempt_id == "second" else []), attempt_id]},
                      emitted.append)

    assert emitted[1] == {"resolved_model": "first/model", "provider": "first", "ledger_attempt_ids": ["first"],
                          "cost": 0.0, "cost_final": True, "cost_estimated": False}
    assert (emitted[2]["resolved_model"], emitted[2]["ledger_attempt_ids"]) == ("second/model", ["second"])


@pytest.mark.parametrize("unread", [False, True])
def test_one_reply_reports_each_issued_call_on_its_own_route_and_keeps_every_id(tmp_path, monkeypatch, unread):
    """One reply names a hold that never left the host and two issued calls on two routes.

    Through the seat's own observer and usage emitter: custody keeps every id; the events are the issued
    calls, each with its own route and price (an exact reported zero included); a receipt this store does
    not hold stays unknown instead of borrowing the seat's or the reply's route; and only the earlier
    ids are read, once, by address.
    """
    from ouroboros import usage_accounting as ua, usage_store
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    read_back, scans, sent = [], [], []

    def spy(name, seen):
        original = getattr(usage_store.Txn, name)

        def reader(txn, *args, **kwargs):
            if not txn.writable:
                seen.append(args[0] if seen is read_back else name)
            return original(txn, *args, **kwargs)

        monkeypatch.setattr(usage_store.Txn, name, reader)

    spy("attempt", read_back)
    for wide in ("attempts", "attempt_rows", "open_attempt_rows"):
        spy(wide, scans)

    class Journey:
        def chat(self, **kwargs):
            def not_sent(_reservation):
                raise RuntimeError("this candidate is withdrawn before its send")

            answer = {"role": "assistant", "content": "[]\nNO_FINDINGS"}
            usage = {"cost": 0.25, "cost_final": True, "cost_estimated": False,
                     "resolved_model": "second/model", "provider": "second"}
            with ua.capture_attempt_ids() as ids:
                try:  # reserved, then released before any send
                    ua.execute_physical_attempt(ua.AttemptRequest(model="first/model", provider="first"),
                                                lambda: answer, before_dispatch=not_sent)
                except ua.PhysicalAttemptPreparationFailed:
                    pass
                ua.execute_physical_attempt(  # issued on the first route; its receipt is an exact reported zero
                    ua.AttemptRequest(model="first/model", provider="first"),
                    lambda: {"role": "assistant", "content": ""},
                    extractor=lambda _: ({"cost": 0.0, "cost_final": True}, 0.0, True))
                ua.execute_physical_attempt(ua.AttemptRequest(model="second/model", provider="second"),
                                            lambda: answer, extractor=lambda _: (usage, 0.25, True))
            sent.extend([*ids[:2], *(["unrecorded"] if unread else []), ids[2]])
            usage["ledger_attempt_ids"] = list(sent)
            return answer, usage

    ctx = SimpleNamespace(task_id="fixture-review", pending_events=[])
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, non_task_operation=True)):
        run = run_review_request(ReviewRequest(surface="multi_model_review", goal="Review the fixture"),
                                 slots=[ReviewSlot("seat", "configured/model")], drive_root=tmp_path,
                                 llm=Journey(), usage_ctx=ctx)
        reads = list(read_back)
        with usage_store.read(tmp_path) as txn:
            states = [(txn.attempt(key) or {}).get("state") for key in sent]
    actor, = run.actors
    hold, first, last = sent[0], sent[1], sent[-1]
    assert states == ["released", "settled", *([None] if unread else []), "settled"]
    assert reads == sent[:-1] and scans == [], "the earlier ids, once, by address; the exact answer is not read back"

    rows = {tuple(row["ledger_attempt_ids"]): row for row in ctx.pending_events if row["type"] == "llm_usage"}
    assert [(ids, row["model"], row["provider"], row["usage"]["cost"], row["usage"]["cost_known"],
             row["usage"].get("cost_final")) for ids, row in rows.items()] == [
        ((first,), "first/model", "first", 0.0, True, True),
        *([(("unrecorded",), "", "", None, False, None)] if unread else []),
        ((last,), "second/model", "second", 0.25, True, True),
    ], "the hold presents no send; no row borrows the configured seat model or the reply's route"
    if unread:
        assert rows[("unrecorded",)]["api_key_type"] == "", "no key is guessed for a route nobody observed"

    usage = actor["usage"]
    assert usage["ledger_attempt_ids"] == sent == actor["outcome"]["usage_ref"]["ledger_attempt_ids"]
    assert hold in usage["ledger_attempt_ids"] and usage["physical_dispatch_observed"] is True
    assert (usage["cost"], usage["cost_partial"], usage["cost_estimated"], usage["cost_final"]) == (
        0.25, unread, False, not unread)
