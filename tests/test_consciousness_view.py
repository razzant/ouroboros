"""Facts use typed place/index seams and keep missing evidence distinct from zero."""

import datetime as dt
import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import consciousness_view as view, places, usage_accounting as ua, usage_store
from ouroboros.consciousness_wake import PLACEHOLDERS, render_wake_message
from tests._store_shared import seed_result
from tests.fixtures_usage_store import _request, _settle


FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "wake_document_214a31b6.json"
NOW = dt.datetime(2026, 10, 24, 8, tzinfo=dt.timezone.utc).timestamp()


def _call(root, monkeypatch, task_id, *, at=NOW - 60, category="consciousness", **fields):
    stamp = dt.datetime.fromtimestamp(at, dt.timezone.utc).isoformat()
    cause = fields.pop("consciousness_cause", "")
    with ua.usage_scope(ua.UsageScope(drive_root=root, root_task_id=task_id, consciousness_cause=cause)), monkeypatch.context() as clock:
        clock.setattr(usage_store, "utc_now_iso", lambda: stamp)
        return _settle(root, cost=0, cost_final=True, task_id=task_id, root_task_id=task_id,
                       category=category, **fields)


@pytest.fixture
def wake(tmp_path, monkeypatch):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    seed_result(tmp_path, document["task_id"], document)
    _call(tmp_path, monkeypatch, document["task_id"])
    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: NOW))
    return document


def test_places_context_has_every_place_and_room_without_reading_history(monkeypatch, tmp_path):
    configured = [places.connect_place(tmp_path, place_id=pid, connector="bridge", provider="chat",
                                       conversation_id=pid, label=label, state=state)
                  for pid, label, state in [("mail", "Inbox", "active"), ("paused", "Paused room", "paused"),
                                            ("quiet", "Quiet room", "silent")]]
    places.edit_place(tmp_path, "mail", "add_note", author="owner", text="Watch releases.\nKeep the exact second line.")
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    monkeypatch.setattr("ouroboros.place_projection.place_activity", lambda _root, place: {
        "unanswered": {"mail": 3, "quiet": 0, "paused": None}[place.place_id], "last_activity_at": ""})
    calls = []
    def rooms(root, place_id):
        calls.append((root, place_id))
        return {"mail": [{"room_chat_id": -1, "unanswered_count": 2},
                          {"room_chat_id": -2, "unanswered_count": 1}], "quiet": [],
                "paused": [{"room_chat_id": -3, "unanswered_count": None}]}[place_id]
    monkeypatch.setattr(places, "place_room_index", rooms, raising=False)
    rows = view.place_views(tmp_path)
    assert [row["unanswered"] for row in rows] == [3, None, 0]
    text = view.places_text(tmp_path, rows)
    assert "Inbox [mail]: active; open turns: 3; chat_history(chat_id=-1); chat_history(chat_id=-2)" in text
    assert "Quiet room [quiet]: silent; open turns: 0; no captured room" in text
    assert "Paused room [paused]: paused; open turns: unknown" in text
    assert "note (owner): Watch releases." in text and "note (owner): Keep the exact second line." in text
    assert calls == [(tmp_path, place.place_id) for place in configured]
    for place in configured:
        places.edit_place(tmp_path, place.place_id, "disconnect", author="owner")
    assert view.places_text(tmp_path, view.place_views(tmp_path)) == "none"


def test_unavailable_index_does_not_claim_zero(monkeypatch, tmp_path):
    places.connect_place(tmp_path, connector="bridge", provider="chat")
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    def unavailable(*_args):
        raise OSError("room index unavailable")
    monkeypatch.setattr(places, "place_room_index", unavailable)
    assert view.place_views(tmp_path)[0]["unanswered"] is None
    assert "unknown" in view.places_text(tmp_path, view.place_views(tmp_path))
    assert "unknown" in view.places_text(tmp_path, None)


def test_prompt_carries_labels_counts_and_one_allowance(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    configured = [{"place_id": "mail", "label": "Inbox", "state": "active", "unanswered": 0, "rooms": []}]
    causes = view.cause_views([{"cause": "place:mail", "usd": 2.5, "unmetered_calls": 1}], configured)
    roots = view.root_views([{"task_id": "root", "title": "Study queue", "consciousness_cause": "started"}], configured)
    text = render_wake_message(repo, drive_root=tmp_path, reason="heartbeat", last_wake_at=0, now=0, level="act", disabled_tools=[],
                               spent_usd=2.5, daily_usd=20, running=1, max_tasks=2, interval=900, events="nothing new",
                               spent_is_floor=True, spend_by_cause=causes, counted_roots=roots, places=configured)
    assert all("{" + placeholder + "}" not in text for placeholder in PLACEHOLDERS)
    assert "Inbox: at least $2.50 (1 unmetered calls)" in text
    assert "root: Study queue (started work)" in text
    assert "places: Inbox [mail]: active; open turns: 0" in text
    assert "allowance known spend (last 24 h): at least 2.50 / 20.00 USD" in text


def test_real_terminal_wake_uses_terminal_fact_and_card_phase(tmp_path, wake):
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE

    row, = view.recent_wakes(tmp_path, [])
    assert row["task_id"] == wake["task_id"] and row["chat_id"] == 1
    assert row["live"] is False and row["quiet"] is True
    assert row["ended_at"] == wake["terminal_time"]["occurred_at"]
    assert row["ended_at"] not in (wake["ts"], wake["updated_at"])
    assert row["phase"] == "done" and OUTCOME_PHASE_HEADLINE[row["phase"]] == "Done"
    assert row["note"] == wake["result"] and "omitted" not in row["note"]
    assert "summary" not in row and "said_owner" not in row
    assert not {"finished_at", "outcome", "unmetered", "cost_usd"} & row.keys()
    assert row["cause"] == "own" and row["conversation"] == ""
    assert "first_reply_at" not in row and "place_id" not in row
    assert row["rounds"] == 9  # accepted solve responses, not 13 physical calls
    assert "tool_calls" not in row  # the old document has no typed count


@pytest.mark.parametrize("summary", ["", "Проверил состояние; решения не нужно."])
def test_admitted_wake_running_reset_never_publishes_null_summary(tmp_path, monkeypatch, summary):
    from ouroboros.gateway.schema import contract_typed_dict, validate_ingress
    from ouroboros.task_finalization import terminal_result_fields
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.worker_chat_lane import name_wake_on_admission

    task_id = "admitted-wake"
    metadata = {"initiator": "consciousness", "wake_cause": "own_clock"}
    name_wake_on_admission(tmp_path, {"id": task_id, "metadata": metadata})
    write_task_result(tmp_path, task_id, "running", task_attempt=0, metadata=metadata,
                      started_at="2026-10-24T07:00:00+00:00", chat_id=1, _is_direct_chat=True)
    assert load_task_result(tmp_path, task_id)["wake_summary"] is None
    usage = {"wake_silent": True, "terminal_origin": "model_final", **({"wake_summary": summary} if summary else {})}
    write_task_result(tmp_path, task_id, "completed", result="Whole note.",
                      _terminal_observed=True, **terminal_result_fields(usage))
    _call(tmp_path, monkeypatch, task_id)
    row, = view.recent_wakes(tmp_path, [], now=NOW)
    assert ("summary" in row) is bool(summary)
    if summary:
        assert row["summary"] == summary
    assert not validate_ingress(row, contract_typed_dict("RecentWake"))


def test_clock_identity_alone_can_keep_a_terminal_wake_live(tmp_path, wake):
    for live_id, expected in ((wake["task_id"], True), ("another-wake", False), ("", False)):
        row, = view.recent_wakes(tmp_path, [], live_wake_task_id=live_id)
        assert row["live"] is expected
        assert row["ended_at"] == wake["terminal_time"]["occurred_at"]
        assert row["phase"] == "done"  # clock identity never overwrites a known outcome


@pytest.mark.parametrize("status,phase,live", [
    ("running", "running", True), ("interrupted", "unknown", True), ("paused", "paused", True),
    ("scheduled", "queued", True), ("requested", "queued", True), ("queued", "queued", True),
    ("completed", "done", False), ("failed", "error", False), ("cancelled", "cancelled", False),
    ("rejected_duplicate", "warn", False),
])
def test_status_is_independent_of_missing_terminal_time(tmp_path, wake, status, phase, live):
    wake.update(status=status, outcome_axes={}, wake_silent=False)
    wake.pop("terminal_time")
    # Enrichment timestamps and the retired field cannot stand in for occurrence.
    wake["finished_at"] = wake["updated_at"]
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [])
    assert row["live"] is live and row["ended_at"] is None
    assert row["phase"] == phase and row["quiet"] is False


@pytest.mark.parametrize("clock_live", [False, True])
@pytest.mark.parametrize("status,owner_wait,phase", [
    ("running", {}, "running"),
    ("running", {"state": "waiting", "quiz_id": "q1"}, "waiting"),
    ("running", {"state": "resumed", "quiz_id": "q1"}, "running"),
    ("running", {"state": "answered", "wait_for_answer": True}, "running"),
    ("running", {"state": "open", "quiz_id": "q1"}, "unknown"),
    ("scheduled", {}, "queued"),
    ("scheduled", {"state": "waiting", "quiz_id": "q1"}, "waiting"),
    ("paused", {}, "paused"),
])
def test_live_wakes_keep_the_task_activity_word(tmp_path, wake, status, owner_wait, phase, clock_live):
    wake.update(status=status, owner_wait=owner_wait, outcome_axes={}, wake_silent=False)
    wake.pop("terminal_time")
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [], live_wake_task_id=wake["task_id"] if clock_live else "")
    assert row["live"] is True and row["ended_at"] is None
    assert row["phase"] == phase


@pytest.mark.parametrize("clock_live", [False, True])
def test_unconfirmed_status_needs_clock_evidence_to_say_running(tmp_path, wake, clock_live):
    wake.update(status="interrupted", outcome_axes={}, wake_silent=False)
    wake.pop("terminal_time")
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [], live_wake_task_id=wake["task_id"] if clock_live else "other")
    assert row["phase"] == ("running" if clock_live else "unknown")


@pytest.mark.parametrize("quiz_state,phase", [("open", "waiting"), ("answered", "running"), ("closed", "running")])
def test_question_projection_keeps_answered_waits_from_looking_blocked(tmp_path, wake, quiz_state, phase):
    wake.update(status="running", outcome_axes={}, owner_wait={"state": "waiting", "quiz_id": "q1"},
                owner_quiz={"q1": {"state": quiz_state, "wait_for_answer": True}})
    wake.pop("terminal_time")
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [])
    assert row["phase"] == phase


@pytest.mark.parametrize("pause_state,phase", [("paused", "paused"), ("released", "queued")])
def test_scheduled_wake_keeps_the_durable_pause_word(tmp_path, wake, pause_state, phase):
    wake.update(status="scheduled", outcome_axes={}, budget_pause={"state": pause_state})
    wake.pop("terminal_time")
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [], live_wake_task_id=wake["task_id"])
    assert row["phase"] == phase and row["live"] is True


def test_terminal_time_for_another_attempt_is_unknown(tmp_path, wake):
    wake["terminal_time"]["attempt"]["task_attempt"] += 1
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [])
    assert row["ended_at"] is None and row["live"] is False


@pytest.mark.parametrize("summary", [None, "Проверил состояние; нового решения не требуется.", ""])
@pytest.mark.parametrize("note", ["Short note", "First line\n" + "several words\t" * 400])
def test_summary_is_only_authored_summary_and_note_is_whole(tmp_path, wake, summary, note):
    wake["result"] = note
    if summary is not None:
        wake["wake_summary"] = summary
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [])
    assert row["note"] == note and "omitted" not in row["note"]
    assert ("summary" in row) is bool(summary)
    if summary:
        assert row["summary"] == summary


@pytest.mark.parametrize("silent,owner,notice,quiet", [
    (True, None, "", True), (True, False, "", True), (True, True, "", False),
    (True, None, "A tool failed.", False), (False, None, "", False), ("true", None, "", False),
])
def test_quiet_uses_delivery_facts(tmp_path, wake, silent, owner, notice, quiet):
    wake.update(wake_silent=silent, terminal_host_notice=notice)
    if owner is not None:
        wake["owner_message_sent"] = owner
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [])
    assert row["quiet"] is quiet
    assert ("said_owner" in row) is (owner is not None)
    if owner is not None:
        assert row["said_owner"] is owner


@pytest.mark.parametrize("money", [{}, {"accounted_upper_bound_usd": 2.5, "reserved_usd": 1.0},
                                  {"cost_presentation": None, "cost_accounting_status": "unavailable",
                                   "cost_accounting_error": "ledger unavailable"}])
def test_cost_and_checks_pass_through_without_invented_facts(tmp_path, wake, money):
    from ouroboros.task_results import TASK_COST_META_FIELDS

    wake.update(money)
    for present in (True, False):
        if not present:
            for key in (*TASK_COST_META_FIELDS, "safety_checks", "loop_outcome"):
                wake.pop(key, None)
        seed_result(tmp_path, wake["task_id"], wake)
        row, = view.recent_wakes(tmp_path, [])
        assert {k: row[k] for k in TASK_COST_META_FIELDS if k in row} == {
            k: wake[k] for k in TASK_COST_META_FIELDS if k in wake}
        assert row["checks"] == (wake["safety_checks"] if present else [])
        assert row.get("rounds") == (9 if present else None)
        assert ("rounds" in row) is present


@pytest.mark.parametrize("checks", [[], [
    {"chip": "not_checked", "tool": "read_file"}, {"chip": "not_checked", "tool": "run_command"},
    {"chip": "warned", "tool": "send_message"}, {"chip": "held", "tool": "send_message"},
]])
def test_check_chips_and_tools_are_the_stored_facts(tmp_path, wake, checks):
    wake["safety_checks"] = checks
    seed_result(tmp_path, wake["task_id"], wake)
    row, = view.recent_wakes(tmp_path, [])
    assert row["checks"] == checks


def test_typed_counts_keep_zero_and_known_values(tmp_path, wake):
    for calls, rounds in ((0, 0), (16, 9)):
        wake["tool_calls"] = calls
        wake["loop_outcome"]["usage"]["total_rounds"] = rounds
        seed_result(tmp_path, wake["task_id"], wake)
        row, = view.recent_wakes(tmp_path, [])
        assert row["tool_calls"] == calls and row["rounds"] == rounds


def test_unrelated_newer_tasks_and_unstarted_wakes_do_not_hide_a_wake(tmp_path, monkeypatch, wake):
    for n in range(60):
        seed_result(tmp_path, f"owner-{n}", {**wake, "task_id": f"owner-{n}", "metadata": {},
                                          "ts": "2026-10-24T08:00:00Z"})
    seed_result(tmp_path, "no-call", {**wake, "task_id": "no-call"})
    seed_result(tmp_path, "started-work", {**wake, "task_id": "started-work"})
    _call(tmp_path, monkeypatch, "started-work", category="consciousness_task")
    # The category is authority even when old task metadata cannot name the cause.
    seed_result(tmp_path, "legacy", {**wake, "task_id": "legacy", "metadata": {}})
    _call(tmp_path, monkeypatch, "legacy", at=NOW)
    rows = view.recent_wakes(tmp_path, [])
    assert [r["task_id"] for r in rows] == ["legacy", wake["task_id"]]
    assert rows[0]["cause"] == "own"  # the allowance's legacy category fallback


def test_selection_uses_the_category_time_index(tmp_path, monkeypatch, wake):
    from contextlib import contextmanager

    queries = []
    real_read = usage_store.read
    @contextmanager
    def traced(*args, **kwargs):
        assert kwargs["allow_stale"] is True
        with real_read(*args, **kwargs) as txn:
            txn.conn.set_trace_callback(queries.append)
            try:
                yield txn
            finally:
                txn.conn.set_trace_callback(None)
    monkeypatch.setattr(usage_store, "read", traced)
    assert [row["task_id"] for row in view.recent_wakes(tmp_path, [])] == [wake["task_id"]]
    query, = [sql for sql in queries if sql.startswith("SELECT root_task_id, place FROM")]
    with real_read(tmp_path, allow_stale=True) as txn:
        plan = [row[3] for row in txn.conn.execute("EXPLAIN QUERY PLAN " + query)]
    assert any("SEARCH attempts USING INDEX attempts_category_time (category=? AND ts_last_epoch>?)" in step
               for step in plan), plan
    assert not any("SCAN attempts" in step for step in plan), plan


def test_selection_is_distinct_bounded_and_ordered_by_ledger_time(tmp_path, monkeypatch, wake):
    ids = [f"wake-{n:02}" for n in range(55)]
    for n, task_id in enumerate(ids):
        seed_result(tmp_path, task_id, {**wake, "task_id": task_id})
        _call(tmp_path, monkeypatch, task_id, at=NOW - n)
    # Many newer calls of one root still occupy exactly one wake row.
    for _ in range(55):
        _call(tmp_path, monkeypatch, ids[0], at=NOW)
    assert [r["task_id"] for r in view.recent_wakes(tmp_path, [])] == ids[:50]


@pytest.mark.parametrize("own,places", [(50, 0), (0, 50), (49, 1), (25, 25), (75, 75), (75, 1), (1, 75)])
def test_wake_selection_reserves_each_source_and_tops_up_unused_share(tmp_path, monkeypatch, own, places):
    ids = {}
    for cause, count in (("own", own), ("place:mail", places)):
        ids[cause] = [f"{cause.replace(':', '-')}-{n:02}" for n in range(count)]
        for n, task_id in enumerate(ids[cause]):
            _call(tmp_path, monkeypatch, task_id, at=NOW - n - (100 if cause == "own" else 0),
                  consciousness_cause=cause)
    selected = view.wake_ids(tmp_path, NOW)
    assert len(selected) == min(50, own + places)
    for cause, other in (("own", places), ("place:mail", own)):
        assert [task_id for task_id in selected if task_id.startswith(cause.replace(':', '-'))] == ids[cause][:max(25, 50 - other)]
    assert selected == sorted(selected, key=lambda task_id: (
        int(task_id.rsplit("-", 1)[1]) + (100 if task_id.startswith("own") else 0), task_id))


@pytest.mark.parametrize("cause,child_cause", [("own", "place:mail"), ("place:mail", "own")])
def test_child_call_cannot_reclassify_its_root_in_selection(tmp_path, monkeypatch, cause, child_cause):
    from ouroboros.usage_origin import root_cause

    _call(tmp_path, monkeypatch, "root", consciousness_cause=cause)
    for n in range(50):
        _call(tmp_path, monkeypatch, f"flood-{n:02}", at=NOW - n, consciousness_cause=child_cause)
    monkeypatch.setattr(usage_store, "utc_now_iso", lambda: dt.datetime.fromtimestamp(NOW - 60, dt.timezone.utc).isoformat())
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, root_task_id="root", consciousness_cause=child_cause)):
        _settle(tmp_path, cost=0, cost_final=True, task_id="child", root_task_id="root", category="consciousness")
    with usage_store.read(tmp_path, allow_stale=True) as txn:
        assert root_cause(txn, "root") == cause
        assert {row[0] for row in txn.conn.execute(
            "SELECT json_extract(extra, '$.consciousness_cause') FROM attempts WHERE root_task_id = 'root'")} == {cause}
    assert "root" in view.wake_ids(tmp_path, NOW)
    assert "child" not in view.wake_ids(tmp_path, NOW)


def test_ledger_child_attempt_selects_its_root_not_the_child(tmp_path, wake):
    seed_result(tmp_path, "child", {**wake, "task_id": "child", "parent_task_id": wake["task_id"]})
    _settle(tmp_path, cost=0, cost_final=True, task_id="child", root_task_id=wake["task_id"],
            category="consciousness")
    assert [row["task_id"] for row in view.recent_wakes(tmp_path, [])] == [wake["task_id"]]


def test_the_allowance_horizon_is_reapplied_without_a_new_write(tmp_path, monkeypatch, wake):
    from ouroboros.consciousness_allowance import ROOT_HORIZON_SEC

    boundary = NOW - 60 + ROOT_HORIZON_SEC
    assert len(view.recent_wakes(tmp_path, [], now=boundary)) == 1
    assert view.recent_wakes(tmp_path, [], now=boundary + 1) == []
    # Fresh calls in another category cannot refresh the wake selection.
    _call(tmp_path, monkeypatch, wake["task_id"], at=boundary + 1, category="task")
    assert view.recent_wakes(tmp_path, [], now=boundary + 1) == []


@pytest.mark.parametrize("state,dispatched,listed", [
    ("open", False, False), ("open", True, True), ("released", False, False),
    ("released", True, False), ("unresolved", True, True),
])
def test_only_dispatched_attempts_not_proven_unsent_are_calls(tmp_path, wake, state, dispatched, listed):
    task_id = "attempted-wake"
    seed_result(tmp_path, task_id, {**wake, "task_id": task_id})
    reservation = ua.reserve_attempt(_request(tmp_path, task_id=task_id, root_task_id=task_id,
                                              category="consciousness"))
    if dispatched:
        ua.mark_dispatched(reservation)
    if state == "released":
        ua.release_attempt(reservation, "before_dispatch_failed:ConnectError", proven_unsent=True)
    elif state == "unresolved":
        ua.mark_unresolved(reservation, "provider went dark")
    ids = [r["task_id"] for r in view.recent_wakes(tmp_path, [])]
    assert (task_id in ids) is listed


@pytest.mark.parametrize("kind", ["subscription_session", "external_unmetered"])
def test_subscription_and_external_wakes_are_calls(tmp_path, wake, kind):
    task_id = "subscription-wake"
    seed_result(tmp_path, task_id, {**wake, "task_id": task_id})
    fields = dict(drive_root=tmp_path, task_id=task_id, root_task_id=task_id, category="consciousness")
    if kind == "subscription_session":
        ua.record_subscription_session(task_id, route="claudexor:codex", **fields)
    else:
        ua.record_unmetered_external_dispatch(task_id, **fields)
    assert task_id in [r["task_id"] for r in view.recent_wakes(tmp_path, [])]


def test_display_read_and_one_result_query_keep_the_note_out_of_full_body_reads(tmp_path, monkeypatch, wake):
    from ouroboros import task_store
    from ouroboros import store
    from contextlib import contextmanager

    def forbidden(*args, **kwargs):
        pytest.fail("recent wakes must not materialize full task documents")
    monkeypatch.setattr(task_store, "documents_by_ids", forbidden)
    monkeypatch.setattr(task_store, "read_document", forbidden)
    queries = []
    real_open = store.open_store
    @contextmanager
    def traced(*args, **kwargs):
        with real_open(*args, **kwargs) as txn:
            txn.conn.set_trace_callback(queries.append)
            yield txn
    monkeypatch.setattr(store, "open_store", traced)
    row, = view.recent_wakes(tmp_path, [])
    assert row["note"] == wake["result"]
    assert len([q for q in queries if "json_extract(body, '$.result')" in q]) == 1


def test_pruned_task_document_does_not_break_other_wake_rows(tmp_path, monkeypatch, wake):
    from ouroboros.task_store import delete_document

    seed_result(tmp_path, "retained", {**wake, "task_id": "retained"})
    _call(tmp_path, monkeypatch, "retained", at=NOW)
    assert len(view.recent_wakes(tmp_path, [])) == 2
    assert delete_document(tmp_path, wake["task_id"])
    assert [row["task_id"] for row in view.recent_wakes(tmp_path, [])] == ["retained"]


def test_reply_time_reads_only_exact_task_rows_and_uses_earliest_instant(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from ouroboros import dialogue_store

    def receipt(task, at, state):
        return SimpleNamespace(row={"task_id": task, "ts": at, "type": "presence_delivery",
                                   "transport": {"delivery": {"state": state}}})

    rows = [receipt("wake", "2026-10-21T10:00:05Z", "failed"),
            receipt("wake", "2026-10-21T10:00:10Z", "uncertain"),
            receipt("child", "2026-10-21T10:00:11Z", "delivered")]
    calls = []
    def probes(root, stream, keys):
        calls.append((root, stream, keys))
        return iter(rows)
    monkeypatch.setattr(dialogue_store, "iter_probes", probes)
    assert view.first_reply_at(tmp_path, "wake") is None
    rows += [receipt("wake", "2026-10-21T10:00:30Z", "delivered"),
             receipt("wake", "2026-10-21T13:00:20+03:00", "accepted")]
    assert view.first_reply_at(tmp_path, "wake") == "2026-10-21T13:00:20+03:00"
    assert calls == [(tmp_path, "chat", [("task_id", "wake")])] * 2


def test_durable_automatic_release_and_receipts_survive_result_updates(tmp_path, wake):
    from ouroboros import dialogue_store
    from ouroboros.task_results import write_task_result

    wake.update(status="running", outcome_axes={}, wake_silent=False, chat_id=-7,
                metadata={"wake_cause": "focus", "focus": {"place_id": "mail", "conversation_key": "chat:a:r"}})
    wake.pop("terminal_time")
    seed_result(tmp_path, wake["task_id"], wake)
    task_id = wake["task_id"]
    outgoing = {"task_id": task_id, "ts": "2026-10-21T10:00:15Z", "direction": "out", "chat_id": -7,
                "source": "presence:mail", "text": "answer", "transport": {"delivery": {"state": "authored"}}}
    # A result/draft is not a released automatic reply.
    dialogue_store.append(tmp_path, "chat", {**outgoing, "source": "task", "type": "task_summary"})
    assert view.recent_wakes(tmp_path, [])[0]["first_reply_at"] is None
    dialogue_store.append(tmp_path, "chat", outgoing)
    assert view.recent_wakes(tmp_path, [])[0]["first_reply_at"] == outgoing["ts"]
    # A tool receipt may arrive later with an earlier host-reported instant.
    dialogue_store.append(tmp_path, "chat", {**outgoing, "type": "presence_delivery", "source": "skill:mail",
        "ts": "2026-10-21T10:00:30Z", "transport": {"delivery": {
            "state": "accepted", "reported_at": "2026-10-21T10:00:10Z"}}})
    write_task_result(tmp_path, task_id, "completed", metadata={**wake["metadata"], "presence_outcome": "message"})
    row = view.recent_wakes(tmp_path, [])[0]
    assert row["first_reply_at"] == "2026-10-21T10:00:10Z"
    assert row["phase"] == "done" and row["live"] is False
    assert row["conversation"] == "chat:a:r" and row["chat_id"] == -7
    labelled, = view.recent_wakes(tmp_path, [{"place_id": "mail", "label": "Inbox"}])
    assert labelled["cause"] == "focus" and labelled["place_label"] == "Inbox"


def test_own_clock_text_survives_an_unreadable_room(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from ouroboros import place_arrivals, store
    place = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="account")
    from tests.test_place_arrivals import seed_identity_arrival
    seed_identity_arrival(tmp_path, place, {"sends_as": {"id": "writer", "label": "Mailbox", "kind": "user"}})
    brief = view.place_views(tmp_path, {})
    assert "writes as Mailbox" in view.places_text(tmp_path, brief)

    @contextmanager
    def busy(*args, **kwargs):
        raise store.StoreBusy("held")
        yield  # pragma: no cover
    monkeypatch.setattr(place_arrivals.dialogue_store, "transaction", busy)
    text = view.places_text(tmp_path, brief)
    assert place.label in text and "writes as" not in text


@pytest.mark.parametrize("kind,owner,clause", [("user", True, "writes as Mailbox (your human's own account)"),
    ("user", False, "writes as Mailbox"), ("bot", False, "as Ouroboros (bot Mailbox)"), ("bot", True, "as Ouroboros (bot Mailbox)"), (None, False, "")])
def test_own_clock_reads_captured_identity_without_changing_brief_view(tmp_path, kind, owner, clause):
    place = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="account")
    if owner:
        places.edit_place(tmp_path, place.place_id, "confirm_owner_account", author="owner", platform_actor_id="writer")
    from tests.test_place_arrivals import seed_identity_arrival
    if kind:
        seed_identity_arrival(tmp_path, place, {"sends_as": {"id": "writer", "label": "Mailbox", "kind": kind}})
    brief = view.place_views(tmp_path, {})
    assert "identity" not in brief[0]
    text = view.places_text(tmp_path, brief)
    assert ("your human's own account" in text) is (owner and kind == "user")
    if clause:
        assert clause in text
    else:
        assert "writes as" not in text and "as Ouroboros" not in text


@pytest.mark.parametrize("cause,identity,owner", [("own_clock", True, True), ("focus", False, False),
    ("focus", True, False), ("focus", True, True)])
def test_recent_focus_as_comes_from_stored_metadata(tmp_path, wake, cause, identity, owner):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    focus = {"place_id": "mail", "conversation_key": "mail:account:inbox:0"}
    if identity:
        focus.update(identity={"sends_as": {"id": "writer", "label": "At the event", "kind": "user"}}, identity_owner=owner)
    document["metadata"].update(wake_cause=cause, focus=focus)
    seed_result(tmp_path, document["task_id"], document)
    row, = view.recent_wakes(tmp_path, [])
    if cause == "focus" and identity:
        assert row["as"] == {"label": "At the event", "owner": owner}
    else:
        assert "as" not in row
