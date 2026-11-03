"""Current Main decisions survive history horizons and share canonical controls."""
import json
import pytest

from ouroboros import obligations, store
from ouroboros.gateway.main_decisions import main_decisions_snapshot
from ouroboros.owner_quiz import quiz_states, record_asked, record_publication, reconcile_terminal
from ouroboros.projects_registry import create_project
from ouroboros.startup_migrations import prepare_startup_state
from ouroboros.store_import import run_legacy_reconcile
from ouroboros.task_results import write_task_result


def corrupt_body(root, task_id="task"):
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("UPDATE task_bodies SET body='unreadable' WHERE task_id=?", (task_id,))


def project_task(root, task_id="task"):
    project = create_project(root, "project", name="Current Project")
    write_task_result(root, task_id, "running", project_id=project["id"], chat_id=project["chat_id"])
    return project


def older_body_ask(root, task_id, quiz_id, **fields):
    """An ask as an older body left it: no delivery record and no publication address
    (this body's ``record_asked`` owes its own publication, so it is not legacy)."""
    from ouroboros.owner_quiz import _mutate_projection

    record_asked(root, task_id, quiz_id=quiz_id, question=fields.pop("question", "Still needed?"),
                 options=fields.pop("options", ["Yes", "No"]), **fields)

    def legacy(quizzes):
        for key in ("publication_chat_id", "publication_evidence", "publication_bound_at", "published_at", "delivery"):
            quizzes[quiz_id].pop(key, None)
        return True
    _mutate_projection(root, task_id, legacy)


def test_all_accepted_optional_questions_survive_terminal_and_initial_history(tmp_path, monkeypatch):
    project = project_task(tmp_path)
    # Each wave obeys the existing 16 OPEN admission. Expiry must not erase the
    # previously accepted questions to make room for subsequent task attempts.
    for wave in range(3):
        for index in range(16):
            record_asked(tmp_path, "task", quiz_id=f"q-{wave}-{index}", question="Choose a format",
                         options=["PDF", "HTML"], assumption="HTML meanwhile")
            record_publication(tmp_path, "task", f"q-{wave}-{index}", project["chat_id"])
        reconcile_terminal(tmp_path, "task")
    # No biography enumeration is allowed on this ordinary state read.
    from ouroboros.memory import Memory
    monkeypatch.setattr(Memory, "read_chat_generations", lambda *a, **k: pytest.fail("history scan"))
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is True
    assert len(snapshot["current_questions"]) == 48
    assert {row["quiz_state"] for row in snapshot["current_questions"]} == {"expired_terminal"}
    assert all(row["assumption"] == "HTML meanwhile" for row in snapshot["current_questions"])


def test_lifecycle_import_recovers_pre_index_questions_once(tmp_path, monkeypatch):
    project = project_task(tmp_path)
    older_body_ask(tmp_path, "task", "older")
    from ouroboros.utils import append_jsonl
    append_jsonl(tmp_path / "logs/chat.jsonl", {"direction": "out", "type": "quiz",
                 "task_id": "task", "chat_id": project["chat_id"], "quiz": {"quiz_id": "older"}})
    assert main_decisions_snapshot(tmp_path, [], {})["current_questions"] == []
    prepare_startup_state(tmp_path)
    assert len(main_decisions_snapshot(tmp_path, [], {})["current_questions"]) == 1
    from ouroboros import startup_migrations
    monkeypatch.setattr(startup_migrations, "_legacy_question_publications", lambda *a: pytest.fail("repeat import"))
    report = prepare_startup_state(tmp_path)
    assert report["imported"] is False and report["question_publications"] == {}


def test_store_published_before_the_question_flag_gains_it_once_at_boot(tmp_path):
    from ouroboros.store_import import ensure_current_store
    project = project_task(tmp_path)
    record_asked(tmp_path, "task", quiz_id="q", question="?", options=["a", "b"])
    record_publication(tmp_path, "task", "q", project["chat_id"])
    with store.open_store(tmp_path, write=True) as txn:  # the column set before ``owner_questions``
        txn.conn.execute("DROP INDEX tasks_has_owner_questions")
        txn.conn.execute("DROP INDEX tasks_underived_flags")
        txn.conn.execute("ALTER TABLE tasks DROP COLUMN has_owner_questions")
        txn.set_meta("schema_version", 1)
    assert ensure_current_store(tmp_path, with_children=False)["flag_columns_added"] == ["owner_questions"]
    assert set(obligations.members(tmp_path, "owner_questions")) == {"task"}
    assert len(main_decisions_snapshot(tmp_path, [], {})["current_questions"]) == 1
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION
    assert "flag_columns_added" not in ensure_current_store(tmp_path, with_children=False)


def schema_one_write(root, document):
    """A schema-1 body's ``Txn.store_task`` on an upgraded store (a rollback's write): its own
    column list, which predates the nullable additions; it never touches ``schema_version``."""
    task_id = str(document["task_id"])
    with store.open_store(root, write=True) as txn:
        columns = {key: value for key, value in store.typed_columns(document).items()
                   if key not in {"has_owner_questions", "review_collection_pending", "plan_collection_pending"}}
        columns["write_seq"] = (txn.row(task_id, "write_seq") or 0) + 1
        txn.conn.execute("INSERT OR REPLACE INTO task_bodies (task_id, body) VALUES (?, ?)",
                         (task_id, store.encode_document(document)))
        names = ("task_id", *columns)
        txn.conn.execute(f"INSERT OR REPLACE INTO tasks ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                         (task_id, *columns.values()))


def test_questions_an_older_body_rewrote_or_asked_return_to_main_after_legacy_job(tmp_path, monkeypatch):
    from ouroboros import startup_migrations
    from ouroboros.owner_quiz import mark_discharged, quiz_delivery_due, record_answered
    from ouroboros.task_store import read_document
    from ouroboros.utils import append_jsonl
    project = project_task(tmp_path)
    for quiz_id in ("kept", "owed", "settled"):
        record_asked(tmp_path, "task", quiz_id=quiz_id, question=f"{quiz_id}?", options=["a", "b"])
        record_publication(tmp_path, "task", quiz_id, project["chat_id"])
    record_answered(tmp_path, "task", quiz_id="owed", option_index=0, request_id="r-owed")
    record_answered(tmp_path, "task", quiz_id="settled", option_index=1, request_id="r-settled")
    mark_discharged(tmp_path, "task", "settled")
    prepare_startup_state(tmp_path)  # this body's generation watermarks exist before the rollback
    # The rolled-back body finishes the asking task and asks on another one: its quiz block has
    # neither a delivery record nor a publication address; its own send put the card in chat.
    schema_one_write(tmp_path, {**read_document(tmp_path, "task"), "status": "completed"})
    write_task_result(tmp_path, "later", "running", project_id=project["id"], chat_id=project["chat_id"])
    schema_one_write(tmp_path, {**read_document(tmp_path, "later"), "owner_quiz": {"older": {
        "quiz_id": "older", "question": "Which?", "options": ["x", "y"], "stake": "", "assumption": "",
        "state": "open", "asked_at": "2026-10-12T00:00:00+00:00", "chat_id": project["chat_id"]}}})
    append_jsonl(tmp_path / "logs/chat.jsonl", {"direction": "out", "type": "quiz", "task_id": "later",
                 "chat_id": project["chat_id"], "quiz": {"quiz_id": "older"}})
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION
        assert [txn.row(tid, "has_owner_questions") for tid in ("task", "later")] == [None, None]
    report = prepare_startup_state(tmp_path)  # the return to this body
    assert report["flag_rows_derived"] == 2 and report["question_publications"] == {}
    assert run_legacy_reconcile(tmp_path)["question_publications"] == {"bound": 1, "unknown": 0}
    assert set(obligations.members(tmp_path, "owner_questions")) == {"task", "later"}
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is True and snapshot["current_questions_gaps"] == []
    assert {(row["task_id"], row["quiz_id"]) for row in snapshot["current_questions"]} == {
        ("task", "kept"), ("task", "owed"), ("later", "older")}  # a settled answer stays settled
    assert quiz_delivery_due(read_document(tmp_path, "task")["owner_quiz"]["owed"])  # recovery owes it again
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION
    monkeypatch.setattr(startup_migrations, "_legacy_question_publications", lambda *a: pytest.fail("repeat import"))
    assert "flag_rows_derived" not in prepare_startup_state(tmp_path)


def test_the_explicit_rebuild_derives_every_question_flag_from_its_body(tmp_path):
    project = project_task(tmp_path)
    record_asked(tmp_path, "task", quiz_id="q", question="?", options=["a", "b"])
    record_publication(tmp_path, "task", "q", project["chat_id"])
    write_task_result(tmp_path, "quiet", "completed")
    with store.open_store(tmp_path, write=True) as txn:  # a derived fact gone wrong by any means
        txn.conn.execute("UPDATE tasks SET has_owner_questions = 1 - has_owner_questions")
    assert set(obligations.members(tmp_path, "owner_questions")) == {"quiet"}
    assert "flag_rows_derived" not in prepare_startup_state(tmp_path)  # an ordinary boot trusts derived rows
    assert prepare_startup_state(tmp_path, rebuild=True)["flag_rows_derived"] == 2
    assert set(obligations.members(tmp_path, "owner_questions")) == {"task"}
    assert len(main_decisions_snapshot(tmp_path, [], {})["current_questions"]) == 1


def test_flag_upgrade_keeps_an_undecodable_body_owed_and_disclosed(tmp_path):
    from ouroboros.store_import import ensure_current_store
    project_task(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("DROP INDEX tasks_has_owner_questions")
        txn.conn.execute("DROP INDEX tasks_underived_flags")
        txn.conn.execute("ALTER TABLE tasks DROP COLUMN has_owner_questions")
    corrupt_body(tmp_path)
    assert ensure_current_store(tmp_path, with_children=False)["flag_columns_added"] == ["owner_questions"]
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is False
    assert snapshot["current_questions_gaps"] == [{"reason": "question_source_unavailable", "task_id": "task"}]


@pytest.mark.parametrize("evidence", ["none", "ambiguous"])
def test_legacy_import_does_not_guess_from_current_routing_or_partial_history(tmp_path, evidence):
    project = project_task(tmp_path)
    older_body_ask(tmp_path, "task", "older", question="?", options=["a", "b"], chat_id=project["chat_id"])
    from ouroboros.utils import append_jsonl
    if evidence != "none":
        for chat_id in [project["chat_id"], 1] if evidence == "ambiguous" else [project["chat_id"]]:
            append_jsonl(tmp_path / "logs/chat.jsonl", {"direction": "out", "type": "quiz",
                         "task_id": "task", "chat_id": chat_id, "quiz": {"quiz_id": "older"}})
    prepare_startup_state(tmp_path)
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions"] == []
    assert snapshot["current_questions_complete"] is False
    assert snapshot["current_questions_gaps"] == [
        {"reason": "publication_address_unknown", "task_id": "task", "quiz_id": "older"}]
    assert "task" in obligations.members(tmp_path, "owner_questions")


def test_question_read_is_readonly(tmp_path):
    project = project_task(tmp_path)
    record_asked(tmp_path, "task", quiz_id="q", question="?", options=["a", "b"])
    record_publication(tmp_path, "task", "q", project["chat_id"])
    with store.open_store(tmp_path, write=False) as txn:
        revision = txn.revision()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert len(snapshot["current_questions"]) == 1
    assert snapshot["current_questions_complete"] is True and snapshot["current_questions_gaps"] == []
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.revision() == revision


@pytest.mark.parametrize("room,chat_id", [("main", 1), ("agent-to-agent", -1001)])
@pytest.mark.parametrize("second,order", [(1, ("waited", "calm")), (0, ("calm", "waited"))],
                         ids=["distinct-times", "tied-times"])
def test_mains_own_unanswered_questions_are_named_by_identity_and_other_rooms_stay_out(
        tmp_path, monkeypatch, room, chat_id, second, order):
    """Main's waiting count reaches Main's own unanswered questions, whose cards are already its native
    history: the census names each by identity and never builds a copy. Another known room that is
    neither Main nor a Project stays out of Main, without a gap."""
    from ouroboros.owner_quiz import mark_discharged, record_answered
    from ouroboros.owner_wait import set_owner_wait

    write_task_result(tmp_path, "author", "running", chat_id=chat_id)
    asked = []
    # Coarse clocks can give both questions the same timestamp; identity breaks that tie.
    for quiz_id, tick in (("waited", 0), ("calm", second)):
        monkeypatch.setattr("ouroboros.owner_quiz.utc_now_iso", lambda tick=tick: f"2026-01-01T00:00:0{tick}+00:00")
        asked.append(record_asked(tmp_path, "author", quiz_id=quiz_id, question=f"{quiz_id}?", options=["Yes", "No"],
                                  wait_for_answer=quiz_id == "waited", chat_id=chat_id))
    for block in asked:
        record_publication(tmp_path, "author", block["quiz_id"], chat_id)
    set_owner_wait(tmp_path, "author", {"quiz_id": "waited", "wait_id": "w", "state": "waiting"})
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is True and snapshot["current_questions_gaps"] == []
    if room != "main":
        assert snapshot["current_questions"] == []
        return
    waited, calm = asked
    expected = {
        "waited": {"task_id": "author", "quiz_id": "waited", "chat_id": 1, "native": True, "quiz_state": "open",
         "ts": waited["asked_at"], "owner_wait_state": "waiting"},
        "calm": {"task_id": "author", "quiz_id": "calm", "chat_id": 1, "native": True, "quiz_state": "open",
         "ts": calm["asked_at"]}}
    assert snapshot["current_questions"] == [expected[quiz_id] for quiz_id in order]
    # Unanswered stays current after its author finishes; an answered one leaves once delivered.
    reconcile_terminal(tmp_path, "author")
    rows = main_decisions_snapshot(tmp_path, [], {})["current_questions"]
    assert [(row["quiz_id"], row["quiz_state"], row["native"]) for row in rows] == [
        (quiz_id, "expired_terminal", True) for quiz_id in order]
    record_answered(tmp_path, "author", quiz_id="calm", option_index=0, request_id="owner", allow_expired=True)
    mark_discharged(tmp_path, "author", "calm")
    assert [row["quiz_id"] for row in main_decisions_snapshot(tmp_path, [], {})["current_questions"]] == ["waited"]


def test_unavailable_index_or_member_is_not_complete_empty(tmp_path, monkeypatch):
    project_task(tmp_path)
    assert main_decisions_snapshot(tmp_path, [], {})["current_questions_complete"] is True
    record_asked(tmp_path, "task", quiz_id="q", question="?", options=["a", "b"])
    with monkeypatch.context() as patched:
        patched.setattr(obligations, "members", lambda *a: (_ for _ in ()).throw(store.StoreBusy("busy")))
        snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is False
    assert snapshot["current_questions_gaps"] == [{"reason": "question_index_unavailable"}]
    corrupt_body(tmp_path)
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is False
    assert snapshot["current_questions"] == []
    assert snapshot["current_questions_gaps"] == [{"reason": "question_source_unavailable", "task_id": "task"}]


@pytest.mark.parametrize("state", ["answered", "closed"])
def test_settled_question_remains_current_until_delivery_is_discharged(tmp_path, state):
    from ouroboros.owner_quiz import mark_discharged, record_answered, record_closed
    project = project_task(tmp_path)
    record_asked(tmp_path, "task", quiz_id="q", question="?", options=["a", "b"])
    record_publication(tmp_path, "task", "q", project["chat_id"])
    if state == "closed":
        record_closed(tmp_path, "task", quiz_id="q", actor_task_id="task", reason="The requested format is obsolete")
    else:
        record_answered(tmp_path, "task", quiz_id="q", option_index=0, request_id="accepted")
    [row] = main_decisions_snapshot(tmp_path, [], {})["current_questions"]
    assert row["quiz_state"] == state
    assert row["question_delivery"] == {"kind": "closure" if state == "closed" else "answer", "state": "owed"}
    mark_discharged(tmp_path, "task", "q")
    assert main_decisions_snapshot(tmp_path, [], {})["current_questions"] == []


def test_unpublished_question_is_not_exposed_and_rebuild_leaves_it_its_own_publication(tmp_path):
    from ouroboros.owner_quiz import quiz_states

    project_task(tmp_path)
    record_asked(tmp_path, "task", quiz_id="q", question="?", options=["a", "b"])
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions"] == [] and snapshot["current_questions_complete"] is False
    prepare_startup_state(tmp_path, rebuild=True)
    assert "task" in obligations.members(tmp_path, "owner_questions")
    # Not legacy: this body owes and binds its own publication, so the import records no unknown.
    assert "question:task:q" not in obligations.members(tmp_path, "unknowns")
    delivery = quiz_states(tmp_path, "task")["q"]["delivery"]
    assert (delivery["kind"], delivery["state"]) == ("publication", "owed")
    assert main_decisions_snapshot(tmp_path, [], {})["current_questions_gaps"] == [
        {"reason": "publication_address_unknown", "task_id": "task", "quiz_id": "q"}]


def test_an_older_bodys_ask_imports_without_turning_a_current_owed_publication_unknown(tmp_path, monkeypatch):
    """F2: after a rollback the one-time import runs for the older body's ask, while a current ask
    still owes its own publication. The import must not record that ask as a ``question:`` unknown:
    once the body publishes it, later boots return early and nothing would clear the row."""
    from ouroboros import startup_migrations
    from ouroboros.owner_quiz import mark_published
    from ouroboros.utils import append_jsonl

    project = project_task(tmp_path)
    record_asked(tmp_path, "task", quiz_id="current", question="Current?", options=["a", "b"])
    prepare_startup_state(tmp_path)  # this body's watermarks exist before the rollback
    write_task_result(tmp_path, "later", "running", project_id=project["id"], chat_id=project["chat_id"])
    older_body_ask(tmp_path, "later", "older", chat_id=project["chat_id"])
    append_jsonl(tmp_path / "logs/chat.jsonl", {"direction": "out", "type": "quiz", "task_id": "later",
                 "chat_id": project["chat_id"], "quiz": {"quiz_id": "older"}})
    assert prepare_startup_state(tmp_path)["question_publications"] == {}
    assert run_legacy_reconcile(tmp_path)["question_publications"] == {"bound": 1, "unknown": 0}
    assert not [key for key in obligations.members(tmp_path, "unknowns") if key.startswith("question:")]
    record_publication(tmp_path, "task", "current", project["chat_id"])  # the body publishes its own ask
    mark_published(tmp_path, "task", "current")
    monkeypatch.setattr(startup_migrations, "_legacy_question_publications", lambda *a: pytest.fail("repeat import"))
    assert prepare_startup_state(tmp_path)["question_publications"] == {}
    assert not [key for key in obligations.members(tmp_path, "unknowns") if key.startswith("question:")]
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is True and snapshot["current_questions_gaps"] == []
    assert {(row["task_id"], row["quiz_id"]) for row in snapshot["current_questions"]} == {
        ("task", "current"), ("later", "older")}


def test_corrupt_registry_and_unconfirmed_binding_are_not_complete_empty(tmp_path):
    project_task(tmp_path)
    obligations.add(tmp_path, "owner_questions", "task")
    assert main_decisions_snapshot(tmp_path, [], {}, bindings_complete=False)["current_questions_complete"] is False
    from ouroboros import projects_registry
    # Registry's strict read reports corruption; its display-only default would
    # silently return [], falsely withdrawing current questions and controls.
    projects_registry._registry_path(tmp_path).write_text("unreadable")
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is False and snapshot["main_attention_complete"] is False


@pytest.mark.parametrize("reason,automatic,source,expected", [
    ("auth", True, "codex", True), ("quota", True, "codex", False),
    ("auth_quota", True, "codex", False), ("unavailable", True, "codex", False),
    ("quota", False, "codex", True), ("auth_quota", False, "codex", True),
    ("unavailable", False, "codex", True), ("unavailable", True, "", True),
])
def test_human_required_waits_use_owner_capability_not_blanket_reason_exclusion(
        tmp_path, reason, automatic, source, expected):
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.task_results import load_task_result
    project = project_task(tmp_path)
    obligations.add(tmp_path, "owner_questions", "task")
    wait = {"wait_id": "wait", "task_attempt": 2, "state": "waiting", "reason": reason,
            "auto_continue": automatic, "source": source}
    owner = TaskModelWait(task={"id": "task", "_attempt": 2}, drive_root=tmp_path,
                          event_queue=None, worker_slot_held=True)
    owner._publish(wait)
    assert wait["requires_owner_action"] is expected
    assert load_task_result(tmp_path, "task")["model_waits"]["wait"] == wait
    activity = {"activity_id": "task", "project_id": project["id"], "phase": "working",
                "task_attempt": 2, "model_waits": {"wait": wait}}
    snapshot = main_decisions_snapshot(tmp_path, [activity], {})
    assert snapshot["main_attention_complete"] is True
    assert bool(snapshot["main_attention"]) is expected
    if expected:
        assert snapshot["main_attention"][0]["model_waits"] == {"wait": wait}
    activity["task_attempt"] = 3
    assert main_decisions_snapshot(tmp_path, [activity], {})["main_attention"] == []
    owner._publish({**wait, "state": "resolved"})
    assert load_task_result(tmp_path, "task")["model_waits"]["wait"]["requires_owner_action"] is False


def test_wait_owner_recomputes_human_fact_on_each_control_revision(tmp_path):
    from ouroboros.model_wait import TaskModelWait
    project_task(tmp_path)
    owner = TaskModelWait(task={"id": "task"}, drive_root=tmp_path,
                          event_queue=None, worker_slot_held=True)
    row = {"wait_id": "wait", "state": "waiting", "reason": "auth_quota",
           "source": "codex", "auto_continue": True}
    owner._publish(row)
    first = row["revision"]
    assert row["requires_owner_action"] is False
    row["auto_continue"] = False
    owner._publish(row, applied_request_id="pause-auto")
    assert row["requires_owner_action"] is True and row["revision"] > first
    row["auto_continue"] = True
    owner._publish(row, applied_request_id="resume-auto")
    assert row["requires_owner_action"] is False


@pytest.mark.parametrize("missing", ["requires_owner_action", "revision"])
def test_model_wait_without_current_owner_action_evidence_discloses_gap(tmp_path, missing):
    project = project_task(tmp_path)
    row = {"wait_id": "wait", "task_attempt": 1, "revision": 2,
           "state": "waiting", "reason": "auth", "requires_owner_action": True}
    row.pop(missing)
    activity = {"activity_id": "task", "project_id": project["id"], "phase": "working",
                "model_waits": {"wait": row}}
    snapshot = main_decisions_snapshot(tmp_path, [activity], {})
    assert snapshot["main_attention"] == []
    assert snapshot["main_attention_complete"] is False
    assert snapshot["main_attention_gaps"] == [{"reason": "model_wait_action_unknown",
                                               "task_id": "task", "wait_id": "wait"}]


@pytest.mark.parametrize("cause,can_resume,reason", [
    ("budget", True, "budget"), ("restart", True, "restart"), ("sleep", True, "resume"), ("unknown", True, "resume"),
    ("owner", False, None), ("sleep", False, None), ("unknown", False, None)])
def test_resume_attention_follows_the_parked_rows_release_rule(tmp_path, cause, can_resume, reason):
    """The census owner derives can_resume from the row's own release rule
    (``events_budget.explicit_resume_required``); the projector never
    re-classifies a pause by its display cause. A held saved sleep needs Resume;
    own Pause and a sleep that wakes itself stay quiet without an incompleteness gap."""
    project = project_task(tmp_path)
    activity = {"activity_id": "task", "project_id": project["id"], "phase": "budget_paused",
                "pause_cause": cause, "can_resume": can_resume, "cancelable": True}
    snapshot = main_decisions_snapshot(tmp_path, [activity], {})
    rows = snapshot["main_attention"]
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []
    assert bool(rows) is can_resume
    if rows:
        assert rows[0]["can_resume"] is True and rows[0]["cancelable"] is True
        assert rows[0]["attention_reason"] == reason
    # Pausing is not yet parked: no Resume is offered before the boundary.
    activity.update(phase="budget_pausing", live=True)
    assert main_decisions_snapshot(tmp_path, [activity], {})["main_attention"] == []
    activity["live"] = False
    assert bool(main_decisions_snapshot(tmp_path, [activity], {})["main_attention"]) is can_resume


def test_required_quiz_wait_is_one_identity_and_does_not_hide_other_task_actions(tmp_path):
    project = project_task(tmp_path)
    record_asked(tmp_path, "task", quiz_id="q", question="Which format?", options=["PDF", "HTML"],
                 wait_for_answer=True)
    record_publication(tmp_path, "task", "q", project["chat_id"])
    activity = {"activity_id": "task", "project_id": project["id"], "phase": "working",
                "owner_wait": {"quiz_id": "q", "owner_wait_state": "waiting"}}
    snapshot = main_decisions_snapshot(tmp_path, [activity], {})
    assert len(snapshot["current_questions"]) == 1 and snapshot["main_attention"] == []
    # A second parallel call needs model access. Same task does not mean same
    # question: keep its existing model-wait controls alongside the quiz.
    wait = {"wait_id": "review-model", "state": "waiting", "task_attempt": 1,
            "revision": 3, "requires_owner_action": True}
    activity["model_waits"] = {"review-model": wait}
    [row] = main_decisions_snapshot(tmp_path, [activity], {})["main_attention"]
    assert row["model_waits"] == {"review-model": wait}
    activity.update(phase="budget_paused", pause_cause="budget", can_resume=True)
    [row] = main_decisions_snapshot(tmp_path, [activity], {})["main_attention"]
    # A parked row runs no wait loop, so its recorded wait is no separate
    # action beside Resume.
    assert row["can_resume"] is True and row["model_waits"] == {}
    activity.update(phase="budget_pausing", live=True)
    assert main_decisions_snapshot(tmp_path, [activity], {})["main_attention"] == []


def test_complete_registry_not_sidebar_and_immutable_conversion_binding(tmp_path):
    project = project_task(tmp_path)
    record_asked(tmp_path, "task", quiz_id="q", question="?", options=["a", "b"], chat_id=1)
    record_publication(tmp_path, "task", "q", 1)
    # Canonical task metadata may predate conversion; its immutable binding wins.
    from ouroboros.task_store import patch_document
    from ouroboros.projects_registry import bind_task_to_project
    bind_task_to_project(tmp_path, "task", project["id"], origin={"absent": "system"})
    patch_document(tmp_path, "task", lambda result: {**{k: v for k, v in result.items() if k != "project_id"},
                                                     "chat_id": 1})
    snapshot = main_decisions_snapshot(tmp_path, [], {"task": {"project_id": project["id"],
                                                              "chat_id": project["chat_id"]}})
    assert snapshot["current_questions_complete"] is True
    assert snapshot["current_questions"][0]["project_chat_id"] == project["chat_id"]


@pytest.mark.parametrize("stop,fail", [(False, False), (False, True), (True, False)])
def test_existing_maintenance_duty_recovers_questions_without_blocking_other_obligations(tmp_path, monkeypatch, stop, fail):
    """Question recovery is its own single-flight duty on the existing cadence; the other
    duties never wait on it (tests/test_terminal_maintenance_independence.py)."""
    import threading
    from ouroboros import server_maintenance
    from ouroboros.gateway import task_decision

    calls = []
    async def recover(root):
        assert root == tmp_path
        calls.append("questions")
        if fail:
            raise OSError("source temporarily unavailable")
    monkeypatch.setattr(task_decision, "recover_question_deliveries", recover)
    monkeypatch.setattr(server_maintenance, "_stop_requested", lambda event: event.is_set())
    monkeypatch.setattr(server_maintenance, "_STEP_FAILURES", {"terminal_delivery_replay": 2})
    lock, event = threading.Lock(), threading.Event()
    if stop:
        event.set()
    lock.acquire()
    server_maintenance._run_terminal_duty(tmp_path, "question_delivery_replay", lock, event)
    assert calls == ([] if stop else ["questions"])
    assert not lock.locked()
    # A failed pass keeps its own streak; another duty's record is untouched.
    assert server_maintenance._STEP_FAILURES == {"terminal_delivery_replay": 2,
                                                  **({"question_delivery_replay": 1} if fail else {})}


@pytest.mark.parametrize("depth,converted", [(1, "root"), (3, "middle")])
@pytest.mark.parametrize("action", ["answer", "close"])
def test_retained_descendant_question_converges_after_ancestor_conversion(tmp_path, monkeypatch, depth, converted, action):
    """An already accepted retained ask follows current display membership, while its reply
    keeps the publication address. This does not grant live subagents new quiz authority."""
    from ouroboros.gateway.history import _assemble_history_response
    from ouroboros.projects_registry import bind_task_to_project
    from tests.test_quiz_answer import _decision_app, _post
    from tests.test_quiz_late_answer import _late_bridge, _inbox

    project = create_project(tmp_path, "converted-room", name="Converted room")
    chain = ["root", "middle", "deeper", "leaf"][:depth + 1]
    for index, tid in enumerate(chain):
        write_task_result(tmp_path, tid, "completed", chat_id=1, root_task_id="root",
                          parent_task_id=chain[index - 1] if index else "",
                          delegation_role="subagent" if index else "root")
    tid = chain[-1]
    record_asked(tmp_path, tid, quiz_id="retained", question="Choose the output", options=["PDF", "HTML"], chat_id=1)
    record_publication(tmp_path, tid, "retained", 1)
    from ouroboros import dialogue_store as ds
    ds.append(tmp_path, "chat", {"type": "quiz", "direction": "out", "chat_id": 1,
        "task_id": tid, "ts": "2026-10-15T00:00:00Z", "text": "Choose the output",
        "quiz": {"quiz_id": "retained", "options": ["PDF", "HTML"]}})
    bind_task_to_project(tmp_path, converted, project["id"], origin={"absent": "system"})
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"], snapshot
    [pointer] = snapshot["current_questions"]
    assert (pointer["task_id"], pointer["quiz_id"], pointer["project_chat_id"]) == (tid, "retained", project["chat_id"])
    main = json.loads(_assemble_history_response(tmp_path, 1, 20, 0))["messages"]
    assert len([r for r in main if r.get("quiz_id") == "retained"]) == 1
    room = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 20, 0))["messages"]
    assert len([r for r in room if r.get("msg_type") == "quiz"]) == 1
    bridge, _ = _late_bridge(tmp_path, monkeypatch)
    body = {"request_id": "owner-choice", "decision_id": f"quiz:{tid}:retained"}
    body.update({"option_index": 1} if action == "answer" else {"action": "close", "reason": "No longer needed"})
    response = _post(_decision_app(tmp_path, monkeypatch), body)
    assert response.status_code == 200, response.text
    block = quiz_states(tmp_path, tid)["retained"]
    assert block["publication_chat_id"] == 1
    assert block["state"] == ("answered" if action == "answer" else "closed")
    if action == "answer":
        assert response.json()["forwarded"] is True
        [message] = _inbox(bridge)
        assert message["chat_id"] == 1
    assert main_decisions_snapshot(tmp_path, [], {})["current_questions"] == []
    main = json.loads(_assemble_history_response(tmp_path, 1, 20, 0))["messages"]
    assert next(r for r in main if r.get("quiz_id") == "retained")["quiz_state"] == block["state"]


def test_unreadable_ancestor_never_hides_a_retained_main_ask_as_unbound(tmp_path, monkeypatch):
    from ouroboros import projects_registry

    write_task_result(tmp_path, "child", "completed", chat_id=1, parent_task_id="unreadable", root_task_id="root")
    record_asked(tmp_path, "child", quiz_id="q", question="Still pending?", options=["Yes"], chat_id=1)
    record_publication(tmp_path, "child", "q", 1)
    monkeypatch.setattr(projects_registry, "_read_lineage", lambda *a, **_kw: None)
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is False
    assert snapshot["current_questions_gaps"] == [{"reason": "question_source_unavailable", "task_id": "child"}]
