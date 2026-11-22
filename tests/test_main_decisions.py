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
from tests._queue_state import gateway, read_state  # noqa: F401


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


@pytest.mark.parametrize(("second", "order", "room", "chat_id"), [
    pytest.param(1, ("waited", "calm"), "main", 1, id='distinct-times-main-1'),
    pytest.param(0, ("calm", "waited"), "agent-to-agent", -1001, id='tied-times-agent-to-agent--1001'),
])
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
         "ts": waited["asked_at"], "owner_wait_state": "waiting", "wait_for_answer": True},
        "calm": {"task_id": "author", "quiz_id": "calm", "chat_id": 1, "native": True, "quiz_state": "open",
         "ts": calm["asked_at"]}}
    assert snapshot["current_questions"] == [expected[quiz_id] for quiz_id in order]
    # A bound that closed is said on the row too, as a Project's pointer says it: never a wait nobody holds.
    from ouroboros.owner_quiz import mark_wait_ended

    assert mark_wait_ended(tmp_path, "author", "waited") is True
    set_owner_wait(tmp_path, "author", {"quiz_id": "calm", "wait_id": "w2", "state": "waiting"})
    ended = {row["quiz_id"]: row for row in main_decisions_snapshot(tmp_path, [], {})["current_questions"]}
    assert ended["waited"]["wait_ended_at"] and ended["waited"]["owner_wait_state"] == "resumed"
    assert ended["calm"]["owner_wait_state"] == "waiting" and "wait_ended_at" not in ended["calm"]
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


def test_main_actions_require_a_known_room_even_in_a_partial_census(tmp_path):
    from ouroboros import projects_registry

    project = project_task(tmp_path)
    # One binding that cannot be read, and one to a Project the registry does not hold.
    projects_registry._save_bindings(tmp_path, {"bindings": {"bad-parent": None, "moved-away": {
        "task_id": "moved-away", "project_id": "no-such-project", "project_chat_id": 4242}}})
    parked = {"phase": "budget_paused", "pause_cause": "budget", "can_resume": True}
    rows = [dict(parked, activity_id=task, chat_id=chat) for task, chat in (
        ("native", 1), ("hidden", 0), ("internal", -7), ("elsewhere", 9), ("missing", None))]
    rows.extend([{**parked, "activity_id": "project", "project_id": project["id"], "chat_id": project["chat_id"]},
                 # A Project ID no room has (a workspace lease lane) is no room and no proof of absence: the
                 # address and the bindings place the row, and so does a registered room's own ID when this
                 # Main work only leases that room's folder. A bound ancestor whose Project is gone, or whose
                 # binding cannot be read, leaves the room unknown.
                 {**parked, "activity_id": "leased", "chat_id": 1, "project_id": "proj_0f1e2d3c4b5a"},
                 {**parked, "activity_id": "leases-a-room-folder", "chat_id": 1, "project_id": project["id"]},
                 {**parked, "activity_id": "unresolved", "chat_id": 1, "parent_task_id": "moved-away"},
                 {**parked, "activity_id": "descendant", "chat_id": 1, "parent_task_id": "bad-parent"}])
    rows.extend({**parked, "activity_id": f"context-{chat}", "chat_id": chat, "project_id": project["id"]}
                for chat in (0, -7, 9))
    snapshot = main_decisions_snapshot(tmp_path, rows, {}, activities_complete=False)
    assert {row["task_id"]: row.get("project_id") for row in snapshot["main_attention"]} == {
        "native": None, "project": project["id"], "leased": "", "leases-a-room-folder": ""}
    assert snapshot["main_attention_complete"] is False
    assert snapshot["main_attention_gaps"] == [
        {"reason": "action_room_unknown", "task_id": "unresolved"},
        {"reason": "action_room_unknown", "task_id": "descendant"}]

    unavailable = main_decisions_snapshot(tmp_path, rows, {}, bindings_complete=False)
    assert [row["task_id"] for row in unavailable["main_attention"]] == ["project"]
    assert not unavailable["main_attention_complete"]
    assert unavailable["main_attention_gaps"] == [{"reason": "task_bindings_unavailable"}]


# --- Which room shows an owner action of Main-addressed work ---------------------------------------
# A census row carries its queue row's Project ID, and that ID is a scope: a room, a workspace lease
# lane — the room's own ID when the leased folder belongs to a registered room — or a scope a helper
# inherited (``worker_promotion._admit_promoted_workspace`` → ``project_scope_admission``). The row's
# address and the strict room rule place its action; the ID alone neither shows it in Main nor proves
# it left.

def _parked(task_id, **fields):
    """A parked census row that only Resume continues, addressed to Main unless ``fields`` say otherwise."""
    return {"activity_id": task_id, "chat_id": 1, "phase": "budget_paused", "pause_cause": "budget",
            "can_resume": True, "cancelable": True, **fields}


def _lease_lane(root):
    """The lane ID a bare workspace start gets for a folder no Project owns: no registry row, no binding."""
    from ouroboros.projects_registry import project_scope_admission

    folder = root / "outside-checkout"
    folder.mkdir()
    basis = project_scope_admission(root, workspace_root=str(folder))
    assert basis["project"] is None and basis["project_id"].startswith("proj_")
    return basis["project_id"]


def _converted_branch(root, project):
    """Main work whose helper the owner turned into a Project: ``root → middle (bound) → parent → helper``.
    The census re-homes a row by its own, its parent's and its root's binding, so ``helper`` still says Main."""
    from ouroboros.projects_registry import bind_task_to_project

    chain = ["root", "middle", "parent", "helper"]
    for index, task_id in enumerate(chain):
        write_task_result(root, task_id, "running", chat_id=1, root_task_id="root",
                          parent_task_id=chain[index - 1] if index else "",
                          delegation_role="subagent" if index else "root")
    bind_task_to_project(root, "middle", project["id"], origin={"absent": "system"})
    return {"parent_task_id": "parent", "root_task_id": "root", "delegation_role": "subagent", "cancelable": False}


def _registered_lease(root, project):
    """The scope ID ``project_scope_admission`` derives for a folder a REGISTERED Project owns: that room's
    own ID with no durable binding, which is what a Main start aimed at the room's folder gets."""
    from ouroboros.projects_registry import project_scope_admission, update_project

    folder = root / "room-checkout"
    folder.mkdir()
    update_project(root, project["id"], working_dir=str(folder))
    basis = project_scope_admission(root, workspace_root=str(folder))
    assert basis["project_id"] == project["id"] and (basis["project"] or {}).get("id") == project["id"]
    return project["id"]


@pytest.mark.parametrize("other_work_converted", [False, True])
def test_a_workspace_lease_id_neither_hides_a_main_action_nor_names_a_project(tmp_path, other_work_converted):
    """A Main start in a folder no Project owns keeps Main as its address and gets a lease lane ID no room has.
    Read as a missing Project, that ID left a permanent ``action_room_unknown`` gap. The task's question was
    always Main's own; its Resume now stands beside it, once, with no Project reference."""
    from ouroboros.projects_registry import bind_task_to_project

    project = project_task(tmp_path)
    lane = _lease_lane(tmp_path)
    write_task_result(tmp_path, "survey", "running", chat_id=1, root_task_id="survey", project_id=lane)
    record_asked(tmp_path, "survey", quiz_id="scope", question="The whole folder?", options=["Yes", "No"], chat_id=1)
    record_publication(tmp_path, "survey", "scope", 1)
    if other_work_converted:  # bindings exist: the strict room rule reads this task's own lineage and finds none
        bind_task_to_project(tmp_path, "task", project["id"], origin={"absent": "system"})
    snapshot = main_decisions_snapshot(tmp_path, [_parked("survey", project_id=lane)], {})
    assert snapshot["current_questions_complete"] is True and snapshot["current_questions_gaps"] == []
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []
    [question] = snapshot["current_questions"]
    assert (question["task_id"], question["quiz_id"], question["chat_id"], question["native"]) == (
        "survey", "scope", 1, True)
    [action] = snapshot["main_attention"]
    assert (action["task_id"], action["chat_id"], action["project_id"], action["project_name"]) == ("survey", 1, "", "")
    assert (action["can_resume"], action["attention_reason"], action["cancelable"]) == (True, "budget", True)


@pytest.mark.parametrize("inherited_scope", [False, True])
def test_a_held_helper_of_converted_work_is_offered_under_that_project(tmp_path, inherited_scope):
    """The owner turned a helper of Main work into a Project; a helper two levels below it holds a Resume and its
    census row still says Main. The strict room rule reaching that Project used to drop the row without a gap. It
    is that Project's action: one row with the Project's reference, whatever scope ID the helper inherited."""
    project = create_project(tmp_path, "atlas-room", name="Atlas room")
    lineage = _converted_branch(tmp_path, project)
    scope = {"project_id": _lease_lane(tmp_path)} if inherited_scope else {}
    snapshot = main_decisions_snapshot(tmp_path, [_parked("helper", pause_cause="restart", **lineage, **scope)], {})
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []
    [action] = snapshot["main_attention"]
    assert (action["task_id"], action["project_id"], action["project_name"], action["chat_id"]) == (
        "helper", project["id"], "Atlas room", project["chat_id"])
    assert (action["can_resume"], action["attention_reason"], action["cancelable"]) == (True, "restart", False)
    assert (action["parent_task_id"], action["root_task_id"]) == ("parent", "root")


def test_placed_rows_stay_single_and_follow_every_existing_owner_action_rule(tmp_path):
    """Placement decides only where a row is shown. The census's release rule still decides whether it is: the
    owner's own Pause, a sleep that wakes itself, an authority pause that reconciles itself, a Pause still
    settling and a wait that continues by itself stay quiet, without a gap, in Main and under a Project alike.
    Nothing is offered twice, a Project ID a room really has keeps its reference, and the same ID a Main row
    only leases through that room's folder stays Main's own until a durable binding says otherwise."""
    project = project_task(tmp_path)
    lane = _lease_lane(tmp_path)
    lineage = _converted_branch(tmp_path, project)
    automatic = {"wait_id": "quota", "state": "waiting", "task_attempt": 1, "revision": 2, "requires_owner_action": False}
    rows = [
        _parked("survey", project_id=lane),
        _parked("helper", **lineage),
        _parked("room-work", project_id=project["id"], chat_id=project["chat_id"]),
        _parked("main-start-in-room-folder", project_id=project["id"]),  # only that room's folder is leased
        _parked("main-lease-bound", project_id=project["id"], parent_task_id="middle"),  # the ID, bound
        _parked("own-pause", project_id=lane, pause_cause="owner", can_resume=False),
        _parked("sleeping", project_id=lane, pause_cause="sleep", can_resume=False),
        _parked("reconciling", project_id=lane, pause_cause="authority"),
        _parked("settling", project_id=lane, phase="budget_pausing", live=True),
        _parked("helper-own-pause", **lineage, pause_cause="owner", can_resume=False),
        _parked("helper-reconciling", **lineage, pause_cause="authority"),
        {"activity_id": "continues-itself", "chat_id": 1, "project_id": lane, "phase": "working",
         "task_attempt": 1, "model_waits": {"quota": automatic}},
    ]
    snapshot = main_decisions_snapshot(tmp_path, rows, {})
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []
    assert [(row["task_id"], row["project_id"], row["project_name"]) for row in snapshot["main_attention"]] == [
        ("survey", "", ""), ("helper", project["id"], "Current Project"),
        ("room-work", project["id"], "Current Project"), ("main-start-in-room-folder", "", ""),
        ("main-lease-bound", project["id"], "Current Project")]
    assert all(row["can_resume"] is True and row["model_waits"] == {} for row in snapshot["main_attention"])


def test_a_leased_rooms_scope_keeps_main_work_and_its_resume_in_main(tmp_path):
    """Main work admitted with a registered room's folder carries that room's own ID as its scope, exactly as
    an unregistered lease lane does. The ID names no room for this row: the task's question stays Main's own
    native card and its Resume stays Main's own row, with no Project reference. Where a durable binding does
    exist, the same ID goes to the bound room, even a different room than the one its own ID names."""
    from ouroboros.projects_registry import bind_task_to_project

    room = project_task(tmp_path)
    lease = _registered_lease(tmp_path, room)
    other = create_project(tmp_path, "atlas-room", name="Atlas room")
    _converted_branch(tmp_path, room)  # root → middle (bound to the room) → parent → helper
    bind_task_to_project(tmp_path, "leaf", other["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "survey", "running", chat_id=1, root_task_id="survey", project_id=lease)
    record_asked(tmp_path, "survey", quiz_id="scope", question="The whole folder?", options=["Yes", "No"],
                 chat_id=1)
    record_publication(tmp_path, "survey", "scope", 1)
    snapshot = main_decisions_snapshot(tmp_path, [
        _parked("survey", project_id=lease),
        _parked("bound-elsewhere", project_id=lease, parent_task_id="leaf"),
        _parked("bound-here", project_id=other["id"], parent_task_id="middle")], {})
    assert snapshot["current_questions_complete"] is True and snapshot["current_questions_gaps"] == []
    [question] = snapshot["current_questions"]
    assert (question["task_id"], question["quiz_id"], question["chat_id"], question["native"]) == (
        "survey", "scope", 1, True)
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []
    assert [(row["task_id"], row["chat_id"], row["project_id"], row["project_name"])
            for row in snapshot["main_attention"]] == [
        ("survey", 1, "", ""),
        ("bound-elsewhere", other["chat_id"], other["id"], "Atlas room"),
        ("bound-here", room["chat_id"], room["id"], "Current Project")]
    assert all(row["can_resume"] is True and row["model_waits"] == {} for row in snapshot["main_attention"])


def test_a_room_addressed_nested_row_lands_at_its_nearest_confirmed_project(tmp_path):
    """A nested row's room address is provisional: the census re-homes it from the nearest LIVE owner row
    (its root, once the ended intermediate ancestor left the census), skipping that ancestor's durable
    binding. The strict room rule finds the NEAREST confirmed Project and the row lands there; a chain
    without a nearer binding keeps the address its census row brought, without a gap."""
    from ouroboros.projects_registry import bind_task_to_project

    outer = project_task(tmp_path)
    nearer = create_project(tmp_path, "nearer-room", name="Nearer room")
    for chain, bound, room in ((["root", "middle", "helper"], "middle", nearer),
                               (["root2", "helper2"], "root2", outer)):
        for index, task_id in enumerate(chain):
            write_task_result(tmp_path, task_id, "running", chat_id=1, root_task_id=chain[0],
                              parent_task_id=chain[index - 1] if index else "",
                              delegation_role="subagent" if index else "root")
        bind_task_to_project(tmp_path, bound, room["id"], origin={"absent": "system"})
    rows = [_parked("helper", project_id=outer["id"], chat_id=outer["chat_id"], parent_task_id="middle",
                    root_task_id="root", delegation_role="subagent", cancelable=False),
            _parked("helper2", project_id=outer["id"], chat_id=outer["chat_id"], parent_task_id="root2",
                    root_task_id="root2", delegation_role="subagent", cancelable=False)]
    snapshot = main_decisions_snapshot(tmp_path, rows, {})
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []
    assert [(row["task_id"], row["project_id"], row["project_name"], row["chat_id"])
            for row in snapshot["main_attention"]] == [
        ("helper", nearer["id"], "Nearer room", nearer["chat_id"]),
        ("helper2", outer["id"], "Current Project", outer["chat_id"])]
    assert all(row["can_resume"] is True and row["cancelable"] is False for row in snapshot["main_attention"])


def test_a_nested_rows_stale_room_is_never_offered_while_the_bindings_cannot_be_read(tmp_path):
    """Without the bindings a nested row keeps the room its source record retained: an ended ancestor's later
    conversion re-homes it only in the strict rule. That address is no proof, so no Resume is offered under it
    and the partial census says why. A row addressed to its own Project has no ancestry the rule could consult;
    Main work leasing a room's folder is no Project's. The restored read places each where the rule proves it."""
    from ouroboros.projects_registry import bind_task_to_project

    outer = project_task(tmp_path)
    nearer = create_project(tmp_path, "nearer-room", name="Nearer room")
    for task_id, parent in (("root", ""), ("middle", "root"), ("helper", "middle")):
        write_task_result(tmp_path, task_id, "running", chat_id=1, root_task_id="root", parent_task_id=parent,
                          delegation_role="subagent" if parent else "root")
    bind_task_to_project(tmp_path, "middle", nearer["id"], origin={"absent": "system"})
    rows = [_parked("helper", project_id=outer["id"], chat_id=outer["chat_id"], parent_task_id="middle",
                    root_task_id="root", delegation_role="subagent", cancelable=False),
            _parked("addressed", project_id=outer["id"], chat_id=outer["chat_id"]),
            _parked("leased", project_id=_registered_lease(tmp_path, outer))]
    unread = main_decisions_snapshot(tmp_path, rows, {}, bindings_complete=False)
    assert [(row["task_id"], row["project_id"]) for row in unread["main_attention"]] == [("addressed", outer["id"])]
    assert unread["main_attention_complete"] is False
    assert unread["main_attention_gaps"] == [{"reason": "task_bindings_unavailable"}]
    restored = main_decisions_snapshot(tmp_path, rows, {})
    assert restored["main_attention_complete"] is True and restored["main_attention_gaps"] == []
    assert [(row["task_id"], row["project_id"], row["chat_id"]) for row in restored["main_attention"]] == [
        ("helper", nearer["id"], nearer["chat_id"]), ("addressed", outer["id"], outer["chat_id"]), ("leased", "", 1)]


def test_unreadable_bindings_keep_a_row_at_its_own_registered_project_address(tmp_path):
    """The branch's independent positive control, apart from any nested row: a row whose own census address IS
    a registered Project's (its chat is that room's, with that Project's ID, no ancestry) is still that
    Project's action in the partial census. The same ID addressed to another registered room, or to a chat no
    room reserves, is offered nowhere. (A row with no chat ID at all, a shape the census never emits, keeps its
    scope's room here exactly as in the complete census.)"""
    outer = project_task(tmp_path)
    other = create_project(tmp_path, "other-room", name="Other room")
    rows = [_parked("own", project_id=outer["id"], chat_id=outer["chat_id"]),
            _parked("elsewhere", project_id=outer["id"], chat_id=other["chat_id"]),
            _parked("unreserved", project_id=outer["id"], chat_id=424242)]
    unread = main_decisions_snapshot(tmp_path, rows, {}, bindings_complete=False)
    assert [(row["task_id"], row["project_id"], row["chat_id"]) for row in unread["main_attention"]] == [
        ("own", outer["id"], outer["chat_id"])]
    assert unread["main_attention_complete"] is False
    assert unread["main_attention_gaps"] == [{"reason": "task_bindings_unavailable"}]


def test_hidden_and_unreadable_neighbours_never_borrow_a_placed_rows_room(tmp_path, monkeypatch):
    """The same lane or the same converted branch at another address is another chat's work and stays out,
    without a gap. A row whose bindings or ancestry cannot be read is a gap of its own beside the rows that
    could be placed; it is never shown in Main for want of proof and never hides its neighbours."""
    from ouroboros import projects_registry

    project = project_task(tmp_path)
    lane = _lease_lane(tmp_path)
    lineage = _converted_branch(tmp_path, project)
    rows = [_parked("survey", project_id=lane), _parked("helper", **lineage)]
    rows.extend(_parked(f"lane-at-{chat}", project_id=lane, chat_id=chat) for chat in (0, -7, 9, None))
    rows.extend(_parked(f"helper-at-{chat}", **lineage, chat_id=chat) for chat in (0, -7, 9, None))
    snapshot = main_decisions_snapshot(tmp_path, rows, {})
    assert [row["task_id"] for row in snapshot["main_attention"]] == ["survey", "helper"]
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []

    # The store stops answering lineage: neither row's room can be proven, so neither is shown.
    monkeypatch.setattr(projects_registry, "_TASK_LINEAGE", {})
    with monkeypatch.context() as unreadable:
        unreadable.setattr(projects_registry, "_read_lineage", lambda *_a, **_kw: None)
        blind = main_decisions_snapshot(tmp_path, rows, {})
    assert blind["main_attention"] == [] and blind["main_attention_complete"] is False
    assert blind["main_attention_gaps"] == [{"reason": "action_room_unknown", "task_id": "survey"},
                                            {"reason": "action_room_unknown", "task_id": "helper"}]
    # One helper's ancestor binding is unreadable: that helper is the gap, its neighbours keep their rooms.
    bindings = dict(projects_registry._load_bindings(tmp_path, fresh=True)["bindings"])
    projects_registry._save_bindings(tmp_path, {"bindings": {**bindings, "torn": None}})
    torn = main_decisions_snapshot(tmp_path, [*rows, _parked("orphan", project_id=lane, parent_task_id="torn")], {})
    assert [row["task_id"] for row in torn["main_attention"]] == ["survey", "helper"]
    assert torn["main_attention_complete"] is False
    assert torn["main_attention_gaps"] == [{"reason": "action_room_unknown", "task_id": "orphan"}]


def _budget_parked(root, task_id, **fields):
    """A queue root parked before dispatch at the spending limit, with the result its admission wrote."""
    write_task_result(root, task_id, "running", root_task_id=task_id, **fields)
    return {"id": task_id, "root_task_id": task_id, "budget_drive_root": str(root), "_attempt": 3, **fields,
            "_budget_pause": {"status": "paused_before_dispatch", "reason": "budget", "replay_safe": True,
                              "auto_resume": False}}


def test_the_census_row_of_a_leased_main_start_reaches_main_with_its_resume(gateway):  # noqa: F811
    """Producer to census to placement: the queue row of a Main start in a folder no Project owns carries the
    lane ID ``project_scope_admission`` derives and the address Main. The census keeps both; Main offers the
    task's existing Resume as its own row, and reading it resumes nothing."""
    lane = _lease_lane(gateway.root)
    parked = _budget_parked(gateway.root, "survey", chat_id=1, project_id=lane)
    gateway.pending.append(parked)
    payload = read_state(gateway)
    [census] = payload["active_chat_activities"]
    assert (census["activity_id"], census["chat_id"], census["project_id"]) == ("survey", 1, lane)
    assert (census["phase"], census["pause_cause"], census["can_resume"]) == ("budget_paused", "budget", True)
    assert payload["main_attention_complete"] is True and payload["main_attention_gaps"] == []
    [action] = payload["main_attention"]
    assert (action["task_id"], action["chat_id"], action["project_id"], action["project_name"]) == ("survey", 1, "", "")
    assert (action["can_resume"], action["attention_reason"], action["cancelable"]) == (True, "budget", True)
    assert gateway.pending == [parked]


def test_the_census_row_of_a_main_start_in_a_registered_rooms_folder_reaches_main(gateway):  # noqa: F811
    """Producer to census to placement: a Main start admitted with a registered room's folder carries that
    room's OWN ID in its queue row. Reading the ID as the room made Main's pill offer that Project instead of
    Main's own System row, beside a question that is Main's own card. The census keeps the scope ID; placement
    keeps both of this task's decisions in Main, with no Project reference, and reading it resumes nothing."""
    room = project_task(gateway.root)
    lease = _registered_lease(gateway.root, room)
    parked = _budget_parked(gateway.root, "survey", chat_id=1, project_id=lease)
    record_asked(gateway.root, "survey", quiz_id="scope", question="The whole folder?", options=["Yes", "No"],
                 chat_id=1)
    record_publication(gateway.root, "survey", "scope", 1)
    gateway.pending.append(parked)
    payload = read_state(gateway)
    [census] = payload["active_chat_activities"]
    assert (census["activity_id"], census["chat_id"], census["project_id"]) == ("survey", 1, lease)
    assert payload["current_questions_complete"] is True and payload["current_questions_gaps"] == []
    [question] = payload["current_questions"]
    assert (question["task_id"], question["quiz_id"], question["chat_id"], question["native"]) == (
        "survey", "scope", 1, True)
    assert payload["main_attention_complete"] is True and payload["main_attention_gaps"] == []
    [action] = payload["main_attention"]
    assert (action["task_id"], action["chat_id"], action["project_id"], action["project_name"]) == ("survey", 1, "", "")
    assert (action["can_resume"], action["attention_reason"], action["cancelable"]) == (True, "budget", True)
    assert gateway.pending == [parked]


def test_the_census_row_of_a_converted_works_held_helper_reaches_main_under_its_project(gateway):  # noqa: F811
    """Producer to census to placement: the census leaves a helper two levels below a converted helper in Main
    (it reads the row's own, its parent's and its root's binding). The strict room rule finds the Project, and
    Main offers the helper's own Resume under that Project, once, while its root keeps running untouched."""
    project = create_project(gateway.root, "atlas-room", name="Atlas room")
    lineage = _converted_branch(gateway.root, project)
    root = {"id": "root", "root_task_id": "root", "chat_id": 1, "budget_drive_root": str(gateway.root), "_attempt": 3}
    helper = {"id": "helper", "chat_id": 1, "budget_drive_root": str(gateway.root),
              **{key: lineage[key] for key in ("parent_task_id", "root_task_id", "delegation_role")},
              "_budget_pause_hold": {"reason": "owner_restart_hold", "selected": False, "dispatchable": False}}
    gateway.running["root"] = {"task": root, "attempt": 3, "started_at": 1.0}
    gateway.pending.append(helper)
    payload = read_state(gateway)
    assert payload["active_chat_activities_complete"] is True
    assert {row["activity_id"] for row in payload["active_chat_activities"]} == {"root"}
    assert payload["main_attention_complete"] is True and payload["main_attention_gaps"] == []
    [action] = payload["main_attention"]
    assert (action["task_id"], action["project_id"], action["project_name"], action["chat_id"]) == (
        "helper", project["id"], "Atlas room", project["chat_id"])
    assert (action["phase"], action["can_resume"], action["cancelable"]) == ("budget_paused", True, False)
    assert (action["parent_task_id"], action["root_task_id"]) == ("parent", "root")
    assert gateway.pending == [helper] and set(gateway.running) == {"root"}


@pytest.mark.parametrize("readable", [False, True])
def test_the_census_row_of_a_nested_helper_follows_its_nearest_confirmed_project(gateway, monkeypatch, readable):  # noqa: F811
    """Producer to census to placement: the live root leases the OUTER room and the census re-homes its
    nested held helper from that root's row (the ended bound middle is no census row of its own). Main
    offers the helper's Resume under the middle's NEARER Project, and reading the census resumes nothing.
    While the census cannot read the bindings that room is the census's alone: nothing is offered under it."""
    from ouroboros import projects_registry
    from ouroboros.projects_registry import bind_task_to_project

    outer = project_task(gateway.root)
    nearer = create_project(gateway.root, "nearer-room", name="Nearer room")
    lineage = _converted_branch(gateway.root, nearer)  # root → middle (bound to nearer) → parent → helper
    bind_task_to_project(gateway.root, "root", outer["id"], origin={"absent": "system"})
    root = {"id": "root", "root_task_id": "root", "chat_id": outer["chat_id"], "project_id": outer["id"],
            "budget_drive_root": str(gateway.root), "_attempt": 3}
    helper = {"id": "helper", "chat_id": 1, "budget_drive_root": str(gateway.root),
              **{key: lineage[key] for key in ("parent_task_id", "root_task_id", "delegation_role")},
              "_budget_pause_hold": {"reason": "owner_restart_hold", "selected": False, "dispatchable": False}}
    gateway.running["root"] = {"task": root, "attempt": 3, "started_at": 1.0}
    gateway.pending.append(helper)
    if not readable:  # the census's own strict read of the durable bindings fails; the rest of the registry reads
        census_read = projects_registry.all_task_project_bindings

        def unreadable(*args, strict=False, **kwargs):
            if strict:
                raise ValueError("Project bindings are unavailable")
            return census_read(*args, **kwargs)
        monkeypatch.setattr(projects_registry, "all_task_project_bindings", unreadable)
    payload = read_state(gateway)
    census = {row["activity_id"]: row for row in payload["active_chat_activities"]}
    assert set(census) == {"root"}, "the helper is no census row of its own"
    assert (census["root"]["chat_id"], census["root"]["project_id"]) == (outer["chat_id"], outer["id"])
    if not readable:
        assert payload["active_chat_activities_complete"] is False and payload["main_attention_complete"] is False
        assert payload["main_attention"] == [], "the census's room for the helper is no proof: no Resume under it"
        assert payload["main_attention_gaps"] == [{"reason": "task_bindings_unavailable"}]
        assert gateway.pending == [helper] and set(gateway.running) == {"root"}
        return
    assert payload["active_chat_activities_complete"] is True
    assert payload["main_attention_complete"] is True and payload["main_attention_gaps"] == []
    [action] = payload["main_attention"]
    assert (action["task_id"], action["project_id"], action["project_name"], action["chat_id"]) == (
        "helper", nearer["id"], "Nearer room", nearer["chat_id"])
    assert (action["phase"], action["can_resume"], action["cancelable"]) == ("budget_paused", True, False)
    assert gateway.pending == [helper] and set(gateway.running) == {"root"}


@pytest.mark.parametrize("cause,can_resume,reason", [
    ("budget", True, "budget"), ("restart", True, "restart"), ("sleep", True, "resume"), ("owner", False, None), ("sleep", False, None), ("unknown", False, None)])
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


@pytest.mark.parametrize(("action", "depth", "converted"), [("answer", 1, "root"), ("close", 3, "middle")])
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
