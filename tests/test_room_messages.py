"""Projects ordered by their newest message (DESIGN Shell "Navigation"; ``ouroboros/room_messages.py``).

The rows come from their real producers (``message_bus``, the terminal delivery's event, the terminal
projection, a binding, the archive import) through the one writer seam into ``state/dialog.sqlite``; the
order is read where the sidebar reads it (``projects_registry.projects_summary``)."""
import contextlib
import json
import types

import pytest

from ouroboros import dialogue_store as ds, projects_registry as registry, room_messages
from supervisor import message_bus

PAST, T = "2020-11-07T{}:00+00:00".format, "2099-11-07T{}:00+00:00".format  # before and after the rooms' creation


@pytest.fixture
def bus(tmp_path, monkeypatch):
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = lambda _frame: None
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    ds.ensure(tmp_path)
    return types.SimpleNamespace(root=tmp_path, bridge=bridge)


def dated(root):
    """The history pass, as the first maintenance look after an update runs it; then the job rests."""
    from ouroboros import maintenance_jobs

    row = lambda: maintenance_jobs.job(root, room_messages.JOB) or {}  # noqa: E731
    assert room_messages.due(root, row(), {}, True)
    assert room_messages.run(root, lambda: False)["state"] == "done"
    assert not room_messages.due(root, row(), {}, True)


def time_of(root, room):
    return (room_messages.read(root)[0] or {}).get(room, "")


def forget_dating(root):
    """The store as a body without ``room_messages`` left it: its rows, revisions and cursors, no times."""
    with ds.transaction(root, write=True) as conn:
        conn.execute("DROP TABLE IF EXISTS room_messages")
        conn.execute("DELETE FROM meta WHERE key IN (?, ?)", (room_messages.GENERATION_KEY, room_messages.REVISION_KEY))


def room_facts(root, room):
    return {key: value for key, value in (ds.room(root, room) or {}).items()}


def test_messages_results_and_questions_date_a_project_and_host_rows_do_not(bus):
    root, room = bus.root, registry.create_project(bus.root, "alpha", name="Alpha")["chat_id"]
    dated(root)
    message_bus.log_chat("in", room, 7, "Owner asks", ts=PAST("10:00"))
    assert time_of(root, room) == PAST("10:00")  # the owner's own words, before any answer
    message_bus.send_with_budget(room, "The reply", ts=PAST("10:01"), task_id="t1")
    assert time_of(root, room) == PAST("10:01")
    revision = room_messages.read(root)[1]
    # Housekeeping and the work's own motion: a closure, a pause, progress, a card's line, a child's words.
    message_bus.send_with_budget(room, "Ouroboros closed this project.", ts=PAST("11:00"), role="system",
                                 system_type="project_closure")
    message_bus.send_with_budget(room, "Paused for resources", ts=PAST("11:01"), task_id="t1", role="system",
                                 system_type="task_pause_notice")
    message_bus.send_with_budget(room, "Working on it", ts=PAST("11:02"), task_id="t1", is_progress=True)
    message_bus.send_with_budget(room, "Custody kept", ts=PAST("11:03"), task_id="t1", role="system",
                                 system_type="custody_notice", progress_meta={"card_row": "timeline", "card_row_id": "k"})
    message_bus.send_with_budget(room, "A helper's words", ts=PAST("11:04"), task_id="c1",
                                 progress_meta={"delegation_role": "subagent", "parent_task_id": "t1", "root_task_id": "t1"})
    assert time_of(root, room) == PAST("10:01") and room_messages.read(root)[1] == revision
    facts = room_facts(root, room)
    assert facts["activity_at"] == PAST("11:04") and facts["revision"] >= 2  # activity and unread move as before
    # A question is a message (``send_quiz``'s canonical row); closing an older one is not; the owner's answer is.
    quiz = {"quiz_id": "q1", "options": [{"label": "A"}, {"label": "B"}], "state": "open"}
    message_bus.log_chat("out", room, 7, "Which one?", ts=PAST("12:00"), task_id="t1", record_type="quiz", feed=True,
                         client_message_id="quiz_publication:t1:q1", require_write=True, quiz=quiz)
    assert time_of(root, room) == PAST("12:00") and room_messages.read(root)[1] != revision
    message_bus.log_chat("system", room, 0, "no longer needed", ts=PAST("12:30"), task_id="t0", record_type="quiz_closed",
                         quiz={**quiz, "quiz_id": "q0", "state": "closed"}, client_message_id="quiz_closed:t0:q0",
                         source="model_question_close", require_write=True)  # events_chat_delivery's row
    assert time_of(root, room) == PAST("12:00")
    message_bus.log_chat("system", room, 0, "A", ts=PAST("12:40"), source="owner_quiz_answer", task_id="t1",
                         record_type="quiz_answer", quiz={**quiz, "state": "answered"}, client_message_id="answer:q1",
                         require_write=True)  # gateway.task_decision's row
    assert time_of(root, room) == PAST("12:40")
    message_bus.send_with_budget(room, "Remember the release", ts="2099-01-01T00:00:00+00:00", role="system",
                                 system_type="reminder")
    assert time_of(root, room) == "2099-01-01T00:00:00+00:00"


def test_a_photo_without_words_and_a_late_answer(bus):
    root, room = bus.root, registry.create_project(bus.root, "beta", name="Beta")["chat_id"]
    dated(root)
    bus.bridge.send_photo(room, b"\x89PNG\r\n\x1a\n" + b"0" * 32, task_id="t1")
    photo = time_of(root, room)
    assert photo, "a delivered photo dates its room"
    message_bus.send_with_budget(room, "Late answer", ts=PAST("00:01"), task_id="t1")  # delivered late, written earlier
    assert time_of(root, room) == photo  # times only rise: an older event changes nothing


def test_a_roots_final_dates_its_project_and_its_done_card_does_not(bus):
    from ouroboros.project_dialogue import append_canonical_task_summary
    from ouroboros.terminal_projection import _project_row
    from supervisor.terminal_delivery import project_terminal_result_event

    root, room = bus.root, registry.create_project(bus.root, "gamma", name="Gamma")["chat_id"]
    dated(root)

    def deliver(task_id, origin, text, ts):
        """``events_chat_delivery._send_message``: the event's meta and its final id into the send seam."""
        event = project_terminal_result_event(root, {"id": task_id, "chat_id": room}, task_id, result_text=text,
                                              terminal_origin=origin, base_event={"chat_id": room, "text": text})
        meta = {**(event.get("progress_meta") or {}), "delivery_id": event["delivery_id"]}
        message_bus.send_with_budget(room, event["text"], ts=ts, task_id=task_id, role=str(event.get("role") or ""),
                                     system_type=str(event.get("system_type") or ""), progress_meta=meta,
                                     require_write=True)
        return event

    assert deliver("r1", "model_final", "Shipped", T("12:00"))["delivery_id"].startswith("final:r1:")
    assert time_of(root, room) == T("12:00")
    salvage = deliver("r2", "host_salvage", "partial output", T("12:05"))
    assert (salvage["role"], salvage["system_type"]) == ("system", "terminal_incident")
    assert time_of(root, room) == T("12:05")
    notice = deliver("r3", "host_notice", "The host ended this work", T("12:10"))
    assert notice["role"] == "system" and not notice.get("system_type")
    assert time_of(root, room) == T("12:10")
    # The Done card that follows a final is the work's summary row, never a second message.
    summary = _project_row("r1", {"task_id": "r1", "status": "completed", "chat_id": room, "result": "Shipped"},
                           {"ts": T("13:00"), "chat_id": room}, {"chat_id": room})
    assert summary["type"] == "task_summary" and summary["summary_kind"] == "terminal_root_projection"
    assert append_canonical_task_summary(root, {**summary, "ts": T("13:00")})
    assert time_of(root, room) == T("12:10")


def test_a_message_shown_in_a_project_dates_it_there_live_and_in_history(bus):
    from ouroboros.chat_references import reference_from_row

    root = bus.root
    alpha = registry.create_project(root, "alpha", name="Alpha")["chat_id"]
    registry.bind_task_to_project(root, "work", "alpha", alpha, origin={"absent": "system"})
    dated(root)
    target = message_bus.send_with_budget(alpha, "A finding", ts=T("09:00"), task_id="work")
    assert target and time_of(root, alpha) == T("09:00")
    stored = next(entry.row for entry in ds.iter_probes(root, "chat", [("chat_id", alpha)], newest_first=True))
    # The owner's direct Reply in Main to that message, and the work's own notice to Main: both shown in Alpha.
    message_bus.log_chat("in", 1, 7, "Good, go on", ts=T("09:30"), message_meta={"reply_to": reference_from_row(stored)})
    assert time_of(root, alpha) == T("09:30") and time_of(root, 1) == T("09:30")
    message_bus.send_with_budget(1, "Heads-up from Alpha", ts=T("09:45"), task_id="work", system_type="main_notice")
    assert time_of(root, alpha) == T("09:45")
    live = room_messages.read(root)[0]
    forget_dating(root)
    dated(root)
    assert room_messages.read(root)[0][alpha] == live[alpha]  # the history pass reads the same membership


def test_a_binding_dates_its_project_from_its_origin_and_earlier_rows(bus):
    from ouroboros.project_dialogue import build_owner_message_ref

    root = bus.root
    room = registry.create_project(root, "delta", name="Delta")["chat_id"]
    dated(root)
    asked = message_bus.log_chat("in", 1, 7, "Do it in Delta", ts=T("08:00"), client_message_id="m1")
    message_bus.send_with_budget(1, "On it", ts=T("08:05"), task_id="late")  # its task spoke in Main first
    assert time_of(root, room) == ""
    ref = build_owner_message_ref(chat_id=1, client_message_id="m1", text=asked["text"], ts=asked["ts"])
    registry.bind_task_to_project(root, "late", "delta", room, origin={"ref": ref, "text": asked["text"]})
    assert time_of(root, room) == T("08:05")
    registry.bind_task_to_project(root, "fresh", "delta", room, origin={"ref": ref, "text": asked["text"]})
    assert time_of(root, room) == T("08:05")  # the origin alone is older than what the room already shows


def test_history_is_dated_once_and_a_closure_no_longer_lifts_a_room(bus):
    root = bus.root
    jeff = registry.create_project(root, "jeff", name="Jeff")["chat_id"]
    other = registry.create_project(root, "other", name="Other")["chat_id"]
    message_bus.send_with_budget(jeff, "Interview notes", ts=T("06:00"), task_id="j")
    message_bus.send_with_budget(other, "Fresh result", ts=T("15:00"), task_id="o")
    message_bus.send_with_budget(jeff, "Ouroboros closed this project.", ts=T("18:00"), role="system",
                                 system_type="project_closure")  # today's memory cleanup
    before = (room_facts(root, jeff), ds.counts(root))
    forget_dating(root)  # an older body wrote all of it
    assert room_messages.read(root) == (None, None)  # pending, said so, never guessed
    pending = registry.projects_summary(root, by_message=True, limit=None)
    assert [row["id"] for row in pending] == ["jeff", "other"] and "last_message_at" not in pending[0]
    dated(root)
    assert time_of(root, jeff) == T("06:00") and time_of(root, other) == T("15:00")
    assert (room_facts(root, jeff), ds.counts(root)) == before  # rows, revision, cursor and activity kept
    listed = registry.projects_summary(root, by_message=True, limit=None)
    assert [row["id"] for row in listed] == ["other", "jeff"]
    assert all(row["last_active_at"] for row in listed) and listed[1]["last_message_at"] >= T("06:00")


def test_an_archive_import_dates_its_rooms_once(tmp_path):
    from ouroboros import dialogue_import
    from tests._dialogue import seed

    room = registry.create_project(tmp_path, "zeta", name="Zeta")["chat_id"]
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [
        {"ts": T("05:00"), "direction": "in", "chat_id": room, "text": "archived question"},
        {"ts": T("05:30"), "direction": "system", "role": "system", "type": "project_closure", "chat_id": room,
         "text": "closed"}]})
    dialogue_import.cutover(tmp_path)
    dialogue_import.run_dialogue_import(tmp_path)
    dated(tmp_path)
    assert time_of(tmp_path, room) == T("05:00")
    with ds.transaction(tmp_path, write=True) as conn:  # a replayed import adds nothing
        room_messages.note(conn, [(room, T("05:00"))])
    assert time_of(tmp_path, room) == T("05:00")


def test_the_feed_and_the_complete_list_take_the_newest_message_beyond_fifty_technical_rooms(bus):
    """Opus o1: fifty-five rooms made technically newer after a room's real message never push it out."""
    root = bus.root
    noisy = [registry.create_project(root, f"noisy-{index:02d}", name=f"Noisy {index}") for index in range(55)]
    useful = registry.create_project(root, "useful", name="Useful")["chat_id"]
    message_bus.send_with_budget(useful, "A real answer", task_id="u")
    for index, project in enumerate(noisy):  # then housekeeping, progress and registry touches everywhere else
        if index % 3 == 0:
            message_bus.send_with_budget(project["chat_id"], "closed", role="system", system_type="project_closure")
        elif index % 3 == 1:
            message_bus.send_with_budget(project["chat_id"], "tick", task_id=f"n{index}", is_progress=True)
        registry.touch_project(root, project["id"])  # journal, digest, binding
    technical = registry.projects_summary(root, include_pinned=True)  # the agent's list keeps its activity order
    assert len(technical) == 50 and "useful" not in [row["id"] for row in technical]
    assert "last_message_at" not in technical[0]
    forget_dating(root)
    dated(root)
    feed = registry.projects_summary(root, include_pinned=True, by_message=True)
    assert feed[0]["id"] == "useful" and len(feed) == 50
    everything = registry.projects_summary(root, by_message=True, limit=None)
    assert everything[0]["id"] == "useful" and everything[0]["last_message_at"] > max(
        str(project["created_at"]) for project in noisy)
    assert [row["last_message_at"] for row in everything[1:]] == sorted(
        (str(project["created_at"]) for project in noisy), reverse=True)  # no message: their creation


def test_the_state_says_when_the_order_moved(bus):
    from ouroboros.gateway.state_publication import _dialogue_revision

    root = bus.root
    room = registry.create_project(root, "eta", name="Eta")["chat_id"]
    message_bus.send_with_budget(room, "Answer", ts=T("10:00"), task_id="e")
    forget_dating(root)
    before = _dialogue_revision(str(root))
    dated(root)  # the pass changes no room fact but the times: the publication key still moves
    assert _dialogue_revision(str(root)) != before
    assert json.dumps(room_messages.read(root)[1]) != "null"


# ---- the parent's counterexamples of 2026-11-07 (each failed on the first candidate) ----------------------------

def test_a_late_written_older_notice_never_hides_a_newer_one_on_binding_or_in_history(bus):
    """H1: the last row written is not the newest event: a binding and the history pass take the newest time."""
    root = bus.root
    room = registry.create_project(root, "theta", name="Theta")["chat_id"]
    dated(root)
    message_bus.send_with_budget(1, "Newer notice", ts=PAST("10:00"), task_id="w", system_type="main_notice")
    message_bus.send_with_budget(1, "Older notice, delivered late", ts=PAST("09:00"), task_id="w",
                                 system_type="main_notice")
    registry.bind_task_to_project(root, "w", "theta", room, origin={"absent": "system"})
    assert time_of(root, room) == PAST("10:00")
    forget_dating(root)
    dated(root)
    assert time_of(root, room) == PAST("10:00")


def test_a_reply_whose_rooms_the_appearance_job_derives_later_dates_its_project(bus, monkeypatch):
    """H2: a row written while the room rule could not be read joins its Project through the appearance job,
    which dates that Project too (the history pass already ran)."""
    from ouroboros import room_appearances
    from ouroboros.chat_references import reference_from_row

    root = bus.root
    alpha = registry.create_project(root, "alpha", name="Alpha")["chat_id"]
    registry.bind_task_to_project(root, "work", "alpha", alpha, origin={"absent": "system"})
    dated(root)
    message_bus.send_with_budget(alpha, "A finding", ts=PAST("09:00"), task_id="work")
    stored = next(entry.row for entry in ds.iter_probes(root, "chat", [("chat_id", alpha)], newest_first=True))
    rule = room_appearances._rule

    def unreadable(_root):
        raise OSError("the registry is being replaced")

    monkeypatch.setattr(room_appearances, "_rule", unreadable)
    message_bus.log_chat("in", 1, 7, "Go on", ts=PAST("09:30"), message_meta={"reply_to": reference_from_row(stored)})
    monkeypatch.setattr(room_appearances, "_rule", rule)
    assert time_of(root, alpha) == PAST("09:00")  # shown nowhere yet: its keys are the job's
    room_appearances.run(root, lambda: False)
    assert time_of(root, alpha) == PAST("09:30")


def test_a_room_registered_after_its_messages_keeps_its_last_message_time(bus):
    """H4: a registry row (re)made today does not make its old room new: creation stands in only without a message."""
    root = bus.root
    room = registry.project_chat_id("old")
    message_bus.send_with_budget(room, "Long ago", ts=PAST("08:00"), task_id="o")
    registry.create_project(root, "old", name="Old")  # as reconcile rebuilds a lost row: created now
    fresh = registry.create_project(root, "fresh", name="Fresh")
    dated(root)
    listed = {row["id"]: row for row in registry.projects_summary(root, by_message=True, limit=None)}
    assert listed["old"]["last_message_at"] == PAST("08:00")
    assert listed["fresh"]["last_message_at"] == fresh["created_at"]
    assert [row["id"] for row in registry.projects_summary(root, by_message=True, limit=None)][:2] == ["fresh", "old"]


@contextlib.contextmanager
def busy_store_and_job_row(monkeypatch):
    """Neither the dialogue store (without waiting) nor the job row can be written meanwhile."""
    from ouroboros import maintenance_jobs

    real_transaction, real_record = ds.transaction, maintenance_jobs.record

    def busy_store(root_, *, write, wait_sec=ds.WAIT_SEC):
        if write and wait_sec == 0:
            raise ds.StoreBusy("held")
        return real_transaction(root_, write=write, wait_sec=wait_sec)

    def busy_job(*_args, **_kwargs):
        raise ds.StoreBusy("held")

    monkeypatch.setattr(ds, "transaction", busy_store)
    monkeypatch.setattr(maintenance_jobs, "record", busy_job)
    try:
        yield
    finally:
        monkeypatch.setattr(ds, "transaction", real_transaction)
        monkeypatch.setattr(maintenance_jobs, "record", real_record)


def test_a_binding_neither_store_could_take_is_dated_by_the_job(bus, monkeypatch):
    """H5 through the real admission chain: on a quiet store the server's own look (``maintenance.due``, not the
    child's) finds the job owed after a busy binding, and the job dates the room with no new
    message; the debt is the bindings file the store has not covered, no journal."""
    from ouroboros import maintenance

    root = bus.root
    room = registry.create_project(root, "eps", name="Eps")["chat_id"]
    dated(root)
    message_bus.send_with_budget(1, "Earlier words", ts=PAST("07:00"), task_id="busy")
    assert room_messages.JOB not in maintenance.due(root)
    with busy_store_and_job_row(monkeypatch):
        registry.bind_task_to_project(root, "busy", "eps", room, origin={"absent": "system"})
    assert time_of(root, room) == ""
    assert room_messages.JOB in maintenance.due(root)  # what MaintenanceChild._run spawns on
    maintenance.run_job(root, room_messages.JOB)
    assert time_of(root, room) == PAST("07:00")
    assert room_messages.JOB not in maintenance.due(root)


def refless_binding_with_a_newer_origin(root, name):
    """A Project dated by its own 06:00 words, holding a ref-less binding; the owner's 08:00 message in Main started
    that task, its ref as ingress captures it (``build_owner_message_ref``). Returns the room, the origin, the time."""
    from ouroboros.project_dialogue import build_owner_message_ref

    room = registry.create_project(root, name, name=name.title())["chat_id"]
    message_bus.send_with_budget(room, "Older words here", ts=PAST("06:00"), task_id="old")
    registry.bind_task_to_project(root, "late", name, room, origin={"absent": "system"})
    asked = message_bus.log_chat("in", 1, 7, f"Do it in {name}", ts=PAST("08:00"), client_message_id=f"m-{name}")
    dated(root)
    ref = build_owner_message_ref(chat_id=1, client_message_id=f"m-{name}", text=asked["text"], ts=asked["ts"])
    return room, {"ref": ref, "text": asked["text"]}, registry.project_task_bindings(root)["late"]["bound_at"]


def test_an_origin_a_ref_less_binding_gains_dates_its_project_at_once(bus):
    """The supported enrichment (a ref-less binding re-bound with its owner ref) makes a newer origin the room's:
    that save dates the room at once and moves the witness from exactly its own start; the binding keeps its time."""
    from ouroboros import maintenance

    root = bus.root
    room, origin, bound_at = refless_binding_with_a_newer_origin(root, "iota")
    assert time_of(root, room) == PAST("06:00")
    registry.bind_task_to_project(root, "late", "iota", room, origin=origin)
    binding = registry.project_task_bindings(root)["late"]
    assert binding["source_ref"] == origin["ref"] and binding["bound_at"] == bound_at
    assert time_of(root, room) == PAST("08:00")
    assert room_messages.JOB not in maintenance.due(root)


def test_an_origin_gained_while_the_store_is_busy_is_dated_by_the_job(bus, monkeypatch):
    """The same enrichment while neither the store nor the job row can be written at once: the server's own look
    finds the job owed and its entry point dates the room from that origin, with no other message."""
    from ouroboros import maintenance

    root = bus.root
    room, origin, bound_at = refless_binding_with_a_newer_origin(root, "kappa")
    with busy_store_and_job_row(monkeypatch):
        registry.bind_task_to_project(root, "late", "kappa", room, origin=origin)
    assert time_of(root, room) == PAST("06:00")
    assert room_messages.JOB in maintenance.due(root)
    maintenance.run_job(root, room_messages.JOB)
    assert time_of(root, room) == PAST("08:00")
    assert registry.project_task_bindings(root)["late"]["bound_at"] == bound_at
    assert room_messages.JOB not in maintenance.due(root)


def test_a_binding_that_dated_its_room_leaves_the_look_at_rest(bus):
    """The happy path spawns nothing: the binding moved the witness from exactly its own start."""
    from ouroboros import maintenance

    root = bus.root
    room = registry.create_project(root, "zeta2", name="Zeta")["chat_id"]
    dated(root)
    message_bus.send_with_budget(1, "Words before the binding", ts=PAST("06:30"), task_id="calm")
    registry.bind_task_to_project(root, "calm", "zeta2", room, origin={"absent": "system"})
    assert time_of(root, room) == PAST("06:30")
    assert room_messages.JOB not in maintenance.due(root)


@pytest.mark.parametrize("nested", [False, True], ids=["typed-lineage", "legacy-metadata"])
def test_legacy_child_words_are_card_content_but_deliveries_and_root_finals_count(tmp_path, nested):
    from ouroboros import dialogue_import
    from tests._dialogue import seed, rows
    from tests._store_shared import seed_result

    room = registry.create_project(tmp_path, "legacy", name="Legacy")["chat_id"]
    lineage = {"delegation_role": "subagent", "parent_task_id": "parent", "root_task_id": "parent"}
    seed_result(tmp_path, "child", {"chat_id": room, "status": "completed",
                                   **({"metadata": lineage} if nested else lineage)})
    seed(tmp_path, archives={"chat_20200101T000000.jsonl": [
        {"ts": PAST("06:00"), "chat_id": room, "direction": "out", "text": "Root answer"},
        {"ts": PAST("07:00"), "chat_id": room, "direction": "out", "task_id": "child", "type": "photo"},
        {"ts": PAST("08:00"), "chat_id": room, "direction": "out", "task_id": "child", "text": "Child final"}]})
    dialogue_import.run_dialogue_import(tmp_path)
    with ds.transaction(tmp_path, write=False) as conn:
        assert conn.execute("SELECT at FROM room_messages WHERE chat_id=?", (room,)).fetchone()[0] == PAST("07:00")
    original = rows(tmp_path), room_facts(tmp_path, room), ds.counts(tmp_path)
    # An installed earlier candidate may already have the bad maximum: update must lower it.
    with ds.transaction(tmp_path, write=True) as conn:
        room_messages.note(conn, [(room, PAST("08:00"))])
        ds._set_meta(conn, room_messages.GENERATION_KEY, 1)
    dated(tmp_path)
    assert time_of(tmp_path, room) == PAST("07:00")
    assert (rows(tmp_path), room_facts(tmp_path, room), ds.counts(tmp_path)) == original
    ds.append(tmp_path, "chat", {"ts": PAST("09:00"), "chat_id": room, "direction": "out",
                                 "task_id": "child", "text": "Another old-format child line"}, appearances=[])
    assert time_of(tmp_path, room) == PAST("07:00")
    ds.append(tmp_path, "chat", {"ts": PAST("10:00"), "chat_id": room, "direction": "system",
                                 "task_id": "parent", "delivery_id": "final:parent:done", "text": "Root end"}, appearances=[])
    assert time_of(tmp_path, room) == PAST("10:00")
    assert not room_messages.due(tmp_path, {}, {}, False)


def test_legacy_role_classification_decodes_after_releasing_the_store(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from ouroboros import store, store_capture
    from tests._store_shared import seed_result

    seed_result(tmp_path, "legacy-speaker", {"chat_id": 7, "metadata": {"delegation_role": "subagent"}})
    real_open, real_decode = store.open_store, store_capture.Captured.tuples
    active, at_decode = [], []

    @contextmanager
    def observed_store(*args, **kwargs):
        with real_open(*args, **kwargs) as txn:
            active.append(txn)
            try:
                yield txn
            finally:
                active.pop()

    def decoded(captured):
        at_decode.append(len(active))
        return real_decode(captured)

    monkeypatch.setattr(store, "open_store", observed_store)
    monkeypatch.setattr(store_capture.Captured, "tuples", decoded)
    answer = {"direction": "out", "text": "Root answer", "chat_id": 7}
    child = {**answer, "text": "Child card", "task_id": "legacy-speaker"}
    assert room_messages.classified([answer, child], registry.RoomResolver(tmp_path, strict=True)) == [answer]
    assert at_decode and not any(at_decode)


def test_retained_origins_date_binding_and_history_without_a_dialogue_copy(bus):
    from ouroboros.gateway.history import _origin_fallback_rows
    from ouroboros.project_dialogue import build_owner_message_ref

    root, room = bus.root, registry.create_project(bus.root, "retained", name="Retained")["chat_id"]
    message_bus.send_with_budget(room, "Old reply", ts=PAST("06:00"))
    dated(root)
    text = "The retained request"
    ref = build_owner_message_ref(chat_id=1, client_message_id="gone", text=text, ts=PAST("08:00"))
    registry.bind_task_to_project(root, "retained", "retained", room, origin={"ref": ref, "text": text})
    assert _origin_fallback_rows(root, room, [])[0]["ts"] == PAST("08:00")
    assert time_of(root, room) == PAST("08:00")
    forget_dating(root)
    dated(root)
    assert time_of(root, room) == PAST("08:00")


def test_reupdate_dates_old_body_appends_without_another_message(bus, monkeypatch):
    from ouroboros import maintenance, room_appearances
    from tests._dialogue import rows

    root, room = bus.root, registry.create_project(bus.root, "rollback", name="Rollback")["chat_id"]
    message_bus.send_with_budget(room, "Before rollback", ts=PAST("06:00"))
    dated(root)
    covered = ds.meta(root, room_messages.THROUGH_KEY)
    # The old store-aware writer still moves the appearance marks, but knows no dating coverage.
    with monkeypatch.context() as old:
        old.setattr(room_messages, "note", lambda *_: None)
        old.setattr(room_messages, "THROUGH_KEY", "unused-old-body-key")
        message_bus.send_with_budget(room, "During rollback", ts=PAST("08:00"))
        message_bus.send_with_budget(room, "Closed", ts=PAST("09:00"), role="system", system_type="project_closure")
    assert ds.meta(root, room_messages.THROUGH_KEY) == covered
    assert ds.meta(root, ds.APPEARANCES_MARK) == ds.stream_end(root, "chat")
    assert not room_appearances.owed(root)
    original = rows(root), room_facts(root, room), ds.counts(root)
    assert room_messages.JOB in maintenance.due(root)
    maintenance.run_job(root, room_messages.JOB)
    assert time_of(root, room) == PAST("08:00")
    assert (rows(root), room_facts(root, room), ds.counts(root)) == original
    assert room_messages.JOB not in maintenance.due(root)


def test_a_busy_lineage_read_defers_dating_without_losing_the_message(bus, monkeypatch):
    from tests._store_shared import seed_result

    root, room = bus.root, registry.create_project(bus.root, "busy-lineage", name="Busy")["chat_id"]
    dated(root)
    seed_result(root, "legacy", {"chat_id": room, "delegation_role": "subagent"})
    with monkeypatch.context() as held:
        held.setattr(registry.RoomResolver, "warm", lambda *_: False)
        ds.append(root, "chat", {"ts": PAST("08:00"), "chat_id": room, "direction": "out",
                                 "task_id": "legacy", "text": "Child words"}, appearances=[])
    assert time_of(root, room) == ""
    dated(root)
    assert time_of(root, room) == ""


def test_binding_delta_includes_descendant_deliveries_and_replies_to_their_work(bus):
    from ouroboros.room_appearances import write_facts
    from tests._store_shared import seed_result

    root, room = bus.root, registry.create_project(bus.root, "tree", name="Tree")["chat_id"]
    seed_result(root, "child", {"delegation_role": "subagent", "parent_task_id": "tree", "root_task_id": "tree"})
    ds.append(root, "chat", {"chat_id": 1, "ts": PAST("07:00"), "direction": "out", "type": "photo",
                             "task_id": "child"}, appearances=[])
    reply = {"chat_id": 1, "ts": PAST("08:00"), "direction": "in", "text": "Owner reply",
             "reply_to": {"address": "task:child", "task_id": "child", "target_home": "work"}}
    keys, shown = write_facts(root, reply)
    ds.append(root, "chat", reply, appearances=keys, shown_in=shown)
    ds.append(root, "chat", {"chat_id": 1, "ts": PAST("09:00"), "direction": "out",
                             "task_id": "child", "text": "Child final"}, appearances=[])
    dated(root)
    assert time_of(root, room) == ""
    registry.bind_task_to_project(root, "tree", "tree", room, origin={"absent": "system"})
    assert time_of(root, room) == PAST("08:00")
    assert room_messages.newest(root, room) == PAST("08:00")


@pytest.mark.parametrize("collisions", [1, 2])
def test_history_pass_retries_one_colliding_room_and_keeps_its_progress(bus, monkeypatch, collisions):
    from ouroboros import maintenance_jobs

    root = bus.root
    chats = sorted(registry.create_project(root, f"resume-{n}", name=f"Resume {n}")["chat_id"] for n in range(3))
    for chat in chats:
        message_bus.send_with_budget(chat, "Room words", ts=PAST("06:00"))
    real, visits, left = room_messages.newest, [], collisions

    def newest(root, chat, **kwargs):
        nonlocal left
        visits.append(chat)
        answer = real(root, chat, **kwargs)
        if chat == chats[1] and left:
            left -= 1
            message_bus.send_with_budget(1, "Concurrent Main message", ts=PAST("09:00"))
        return answer

    monkeypatch.setattr(room_messages, "newest", newest)
    first = room_messages.run(root, lambda: False)
    if collisions == 1:
        assert first["state"] == "done"
    else:
        assert first["state"] == "pending"
        assert maintenance_jobs.job(root, room_messages.JOB)["cursor"] == chats[0]
        visits.clear()
        assert room_messages.run(root, lambda: False)["state"] == "done"
        assert chats[0] not in visits
    assert all(time_of(root, chat) == PAST("06:00") for chat in chats)


@pytest.mark.parametrize("new_debt", ["binding", "old-body"])
def test_resuming_a_history_pass_does_not_cover_new_debt_behind_its_cursor(bus, monkeypatch, new_debt):
    root = bus.root
    projects = sorted((registry.create_project(root, f"debt-{n}", name=f"Debt {n}") for n in range(2)),
                      key=lambda project: project["chat_id"])
    earlier = projects[0]
    for project in projects:
        message_bus.send_with_budget(project["chat_id"], "Room words", ts=PAST("06:00"))
    assert room_messages.run(root, lambda: True)["state"] == "pending"
    if new_debt == "binding":
        message_bus.send_with_budget(1, "Work before late binding", task_id="late-debt", ts=PAST("08:00"))
        with monkeypatch.context() as busy:
            busy.setattr(room_messages, "bound", lambda *_: None)
            registry.bind_task_to_project(root, "late-debt", earlier["id"], earlier["chat_id"], origin={"absent": "system"})
    else:
        with monkeypatch.context() as old:
            old.setattr(room_messages, "note", lambda *_: None)
            old.setattr(room_messages, "THROUGH_KEY", "old-body-unused-key")
            message_bus.send_with_budget(earlier["chat_id"], "Message while rolled back", ts=PAST("08:00"))
    assert room_messages.run(root, lambda: False)["state"] == "done"
    # Either this pass includes the new words or it leaves exactly their ordinary recovery owed.
    if time_of(root, earlier["chat_id"]) != PAST("08:00"):
        assert room_messages.due(root, {}, {}, False)
        assert room_messages.run(root, lambda: False)["state"] == "done"
    assert time_of(root, earlier["chat_id"]) == PAST("08:00")
    assert not room_messages.due(root, {}, {}, False)


def test_import_keeps_legacy_nonnumeric_chat_rows_and_dates_the_normalized_room(tmp_path):
    from ouroboros import dialogue_import
    from tests._dialogue import seed, rows

    original = [{"chat_id": "main", "ts": PAST("06:00"), "direction": "in", "text": "Legacy main"},
                {"chat_id": 1, "ts": PAST("07:00"), "direction": "in", "text": "Main"}]
    seed(tmp_path, archives={"chat_20200101T000000.jsonl": original})
    dialogue_import.run_dialogue_import(tmp_path)
    ds.require_imported(tmp_path, "chat")
    assert rows(tmp_path) == original
