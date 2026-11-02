"""#1589: a place sees and withdraws the promises (follow-ups) it made, and nothing else.

Real Place admission and focus task construction (``admit_presence_turn``,
``presence_runner._build_task``), the production tool registry with its dispatch guards,
the real ``schedule_followup`` producer, the real schedule table, scheduler and audited
withdrawal lifecycle. One ownership rule (``queue_schedules.presence_promise``) decides the
scoped ``manage_schedules`` read, the withdrawal's in-transaction check, the focus listing
and the runtime schedule digest. Only Safety's verdict is a fixture.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import replace

import pytest

from ouroboros import places as p
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_runner import PresenceTurnEvent, _build_task
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.serial

RUN_AT = "2030-01-01T09:00:00+00:00"


@pytest.fixture
def table(tmp_path, monkeypatch):
    from supervisor import queue

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    queue.init(tmp_path)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    p.connect_place(tmp_path, connector="bridge", provider="chat", account_id="team", wakes=["mention"])
    p.connect_place(tmp_path, connector="bridge", provider="chat", account_id="other", wakes=["mention"])
    return tmp_path, pending


def _turn(root, account, conversation, number=1):
    """A Place focus exactly as the Host builds it, and the registry its loop would use."""
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id=account, conversation_id=conversation)
    event = PresenceTurnEvent(f"evt-{conversation}-{number}", "chat", account, conversation, "",
                              f"chat:{account}:{conversation}", {"platform_actor_id": "person"}, {}, {},
                              "hello", facts=("mention",))
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}), event,
                       drive_root=root, staged_files=())
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=task["id"], task_metadata=dict(task["metadata"]),
                      task_contract=task["task_contract"], is_direct_chat=True, current_chat_id=task["chat_id"])
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry.set_context(ctx)
    return registry, task, admission.binding_id


def _promise(registry, objective):
    reply = registry.execute("schedule_followup", {"objective": objective, "run_at": RUN_AT, "relation": "independent"})
    assert reply.startswith("FOLLOWUP_SCHEDULED"), reply
    return reply.split("follow-up ", 1)[1].split(" ", 1)[0]


def _rows(root):
    from supervisor.queue import list_scheduled_tasks

    return {row["id"]: row for row in list_scheduled_tasks(root)["tasks"]}


def test_a_place_lists_reads_and_withdraws_only_its_own_promises_across_its_conversations(table):
    from supervisor.schedule_lifecycle import upsert_scheduled_task

    root, _pending = table
    first, _task, binding = _turn(root, "team", "k1")
    asked = _promise(first, "Check whether Ann still needs the backup report")
    second, _task2, same = _turn(root, "team", "k2")
    later = _promise(second, "Send Bob the meeting notes if he has not answered")
    stranger, _task3, other = _turn(root, "other", "z1")
    foreign = _promise(stranger, "Another place's own promise")
    assert same == binding != other
    upsert_scheduled_task({"id": "owner-row", "name": "Owner's weekly digest", "timezone": "UTC",
                           "trigger": {"type": "cron", "expr": "0 9 * * 1"},
                           "task": {"type": "task", "text": "Owner work"}}, drive_root=root)
    upsert_scheduled_task({"id": "empty-carrier", "name": "Lost carrier", "source": "task_followup",
                           "trigger": {"type": "once", "run_at": RUN_AT},
                           "task": {"type": "task", "text": "A carrier without a binding",
                                    "metadata": {"presence": {"binding_id": ""}}}}, drive_root=root)

    # A third conversation of the same place reads both of its promises and nothing else,
    # filtered before the page is cut.
    third, _task4, _ = _turn(root, "team", "k3")
    page = json.loads(third.execute("manage_schedules", {"action": "list", "limit": 1}))
    assert page["total"] == 2 and page["next_offset"] == 1 and page["tasks"][0]["id"] == asked
    assert [row["id"] for row in json.loads(third.execute("manage_schedules", {"action": "list"}))["tasks"]] == [asked, later]
    assert json.loads(third.execute("manage_schedules", {"action": "get", "schedule_id": later}))["ok"] is True
    for hidden in (foreign, "owner-row", "empty-carrier"):
        assert json.loads(third.execute("manage_schedules", {"action": "get", "schedule_id": hidden}))["status"] == "not_found"
        refused = json.loads(third.execute("manage_schedules", {"action": "delete", "schedule_id": hidden,
                                                                "reason": "not mine"}))
        assert refused["status"] == "not_found" and refused["audit"] == "not_written"
        assert not _rows(root)[hidden].get("delete_requested_at")
    for action in ("update", "disable", "restore"):
        assert "RESOURCE_CONSTRAINT_BLOCKED" in third.execute("manage_schedules", {
            "action": action, "schedule_id": asked, "reason": "not offered", "objective": "changed"})

    # The withdrawal is the owner's audited lifecycle: future dispatch only, restorable by the owner.
    receipt = json.loads(third.execute("manage_schedules", {"action": "delete", "schedule_id": asked,
                                                            "reason": "Ann withdrew the question"}))
    assert receipt["status"] == "withdrawn" and receipt["audit"] == "recorded" and receipt["changed"] is True
    assert receipt["running_or_queued"] is False and receipt["schedule"]["delete_pending"] is True
    stored = _rows(root)
    assert stored[asked]["delete_requested_at"] and stored[asked]["enabled"] is False
    assert stored[later]["enabled"] is True and not stored[foreign].get("delete_requested_at")
    events = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["phase"] for row in events if row.get("type") == "schedule_mutation"
            and row.get("schedule_id") == asked and row.get("action") == "delete"] == ["intent", "outcome"]

    # An empty carrier reaches nothing, not even a list.
    lost = ToolRegistry(repo_dir=root, drive_root=root)
    lost.set_context(ToolContext(repo_dir=root, drive_root=root, task_id="lost-carrier",
                                 task_metadata={"presence": {"binding_id": ""}}))
    assert "RESOURCE_CONSTRAINT_BLOCKED" in lost.execute("manage_schedules", {"action": "list"})


def test_the_place_sees_its_pending_promises_in_its_focus_and_digest(table):
    from ouroboros import context as context_module
    from types import SimpleNamespace

    root, _pending = table
    first, _task, binding = _turn(root, "team", "k1")
    asked = _promise(first, "Check whether Ann still needs the backup report")
    withdrawn = _promise(first, "A promise this place already took back")
    assert json.loads(first.execute("manage_schedules", {"action": "delete", "schedule_id": withdrawn,
                                                         "reason": "no longer needed"}))["status"] == "withdrawn"
    stranger, _task2, other = _turn(root, "other", "z1")
    foreign = _promise(stranger, "Another place's own promise")

    # The next turn of the place, in another of its conversations, is told what it still owes.
    _registry, task, _ = _turn(root, "team", "k2", number=2)
    work_here = task["metadata"]["focus_work_here"]
    assert "Promises this place scheduled" in work_here and asked in work_here
    assert "Check whether Ann still needs the backup report" in work_here and "from conversation chat:team:k1" in work_here
    assert withdrawn not in work_here and foreign not in work_here
    assert "manage_schedules" in work_here and "cancel_task" in work_here
    # SYSTEM's in-place update is the owner's; here a change is a withdrawal plus a new promise.
    assert "withdraw it and schedule it anew" in work_here and "action=\"update\" is refused" in work_here
    from ouroboros.presence_context import frame_focus_content
    assert asked in frame_focus_content(task, "hello")

    # Its runtime schedule digest is the same scoped view; the owner keeps the whole table.
    env = SimpleNamespace(drive_root=root)
    scoped = context_module._scheduled_tasks_digest(env, task)
    names = [entry["id"] for group in ("standing", "attention", "one_shots") for entry in scoped[group]]
    assert set(names) == {asked, withdrawn} and foreign not in names and binding
    owner = context_module._scheduled_tasks_digest(env, {"id": "owner-turn", "metadata": {}})
    assert foreign in [entry["id"] for group in ("standing", "attention", "one_shots") for entry in owner[group]]
    assert context_module._scheduled_tasks_digest(env, {"metadata": {"presence": {"binding_id": ""}}}) == {
        "standing": [], "attention": [], "one_shots": [], "omitted_count": 0, "pointer": "manage_schedules(action='list')"}
    from ouroboros.agent import Env

    runtime = context_module.build_runtime_section(Env(repo_dir=pathlib.Path(__file__).resolve().parents[1],
                                                       drive_root=root), task)
    assert asked in runtime and foreign not in runtime



@pytest.mark.parametrize("fault", ["unreadable_table", "policy_read_failure"])
def test_an_unreadable_listing_never_fails_the_turn_and_never_reads_as_no_promises(table, monkeypatch, fault):
    root, _pending = table
    if fault == "unreadable_table":
        (root / "state" / "scheduled_tasks.json").write_text("{not json", encoding="utf-8")
    else:
        first, _task, _binding = _turn(root, "team", "k1")
        _promise(first, "A promise whose current controls cannot be read now")

        def unreadable_controls(*_a, **_kw):
            raise RuntimeError("origin controls unreadable")

        monkeypatch.setattr("supervisor.followup_policy.observed_store", unreadable_controls)
    _registry, task, _binding = _turn(root, "team", "k2")
    work_here = task["metadata"]["focus_work_here"]
    assert "Promises this place scheduled" in work_here and "Unavailable now" in work_here


DUE = "2000-01-01T00:00:00+00:00"


def _due_promise(registry, objective):
    reply = registry.execute("schedule_followup", {"objective": objective, "run_at": DUE, "relation": "independent"})
    assert reply.startswith("FOLLOWUP_SCHEDULED"), reply
    return reply.split("follow-up ", 1)[1].split(" ", 1)[0]


@pytest.fixture
def scheduler(table, monkeypatch):
    """The supervisor's scheduler pass over the same table, with consciousness work enabled."""
    from supervisor import queue

    monkeypatch.setattr("supervisor.consciousness_carriers.autonomous_work_enabled", lambda *_a, **_kw: True,
                        raising=False)
    return table, queue.check_scheduled_tasks


def test_withdrawal_before_acceptance_prevents_the_run_and_after_it_names_the_owed_task(scheduler):
    from ouroboros.task_results import load_task_result

    (root, pending), fire = scheduler
    turn, _task, _binding = _turn(root, "team", "k1")
    before = _due_promise(turn, "Withdrawn before the scheduler reaches it")
    receipt = json.loads(turn.execute("manage_schedules", {"action": "delete", "schedule_id": before,
                                                           "reason": "the question is closed"}))
    assert receipt["status"] == "withdrawn" and receipt["running_or_queued"] is False
    fire()
    assert pending == [] and _rows(root)[before]["delete_requested_at"]

    after = _due_promise(turn, "Accepted before the place changes its mind")
    fire()
    [accepted] = pending
    assert accepted["metadata"]["schedule_occurrence"]["schedule_id"] == after
    receipt = json.loads(turn.execute("manage_schedules", {"action": "delete", "schedule_id": after,
                                                           "reason": "too late to matter"}))
    # Future dispatch is withdrawn; the accepted occurrence is named and still owed, never "stopped".
    assert receipt["status"] == "withdrawn" and receipt["running_or_queued"] is True
    assert receipt["schedule"]["last_task_id"] == accepted["id"]
    assert pending == [accepted]
    admission = load_task_result(root, accepted["id"])["schedule_admission"]
    assert admission["status"] == "accepted" and admission["dispatch"] == "none"
    fire()
    assert pending == [accepted]  # nothing new is admitted from the withdrawn row


@pytest.fixture
def fired_context(tmp_path, monkeypatch):
    """A drive with a minimal body to render the first request of a fired promise."""
    from supervisor import queue
    from tests.test_cache_optimization import _make_env_and_memory

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr("supervisor.consciousness_carriers.autonomous_work_enabled", lambda *_a, **_kw: True,
                        raising=False)
    env, memory = _make_env_and_memory(tmp_path)
    root = env.drive_root
    queue.init(root)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    p.connect_place(root, connector="bridge", provider="chat", account_id="team", wakes=["mention"])

    def first_request(task):
        from ouroboros.context import build_llm_messages

        messages, _ = build_llm_messages(env=env, memory=memory, task=task)
        return json.dumps(messages, ensure_ascii=False)

    return root, pending, first_request


def _pointer(root, task_id, message):
    from ouroboros.presence_runner import _write_previous_turn

    _write_previous_turn(root, "chat:team:k1", task_id, outcome="message", message=message, sends=[message],
                         work_ref="", finished_at=f"2026-10-0{len(task_id)}T00:00:00+00:00", delivery="confirmed")


@pytest.mark.parametrize("later_turn", [True, False], ids=["later-turn", "no-later-turn"])
def test_a_fired_promise_sees_its_conversation_as_it_is_now(fired_context, later_turn):
    from supervisor.queue import check_scheduled_tasks

    root, pending, first_request = fired_context
    _pointer(root, "z", "Z: Ann asked for the backup report.")
    turn, task, _binding = _turn(root, "team", "k1")
    # The promise's carrier is copied by value: it still holds the pointer this turn read at its start.
    assert task["metadata"]["presence"]["previous_turn"]["message"] == "Z: Ann asked for the backup report."
    promised = _due_promise(turn, "Check whether Ann still needs the backup report")
    _pointer(root, "aa", "A: I will check back tomorrow.")  # the runner's own write when turn A ends
    if later_turn:
        _pointer(root, "bbb", "B: Ann withdrew the question; nothing is needed.")
    check_scheduled_tasks()
    [fired] = pending
    assert fired["metadata"]["schedule_occurrence"]["schedule_id"] == promised
    assert fired["metadata"]["presence"]["previous_turn"]["message"].startswith("Z:")  # the stale copy is still there
    text = first_request(fired)
    current = "B: Ann withdrew the question" if later_turn else "A: I will check back tomorrow."
    assert current in text and "Z: Ann asked for the backup report." not in text
    assert "This run started from a schedule, not from a new message" in text
    assert "People receive only what a connector tool sends" in text
    if later_turn:
        assert "A: I will check back tomorrow." not in text


def test_a_retained_legacy_promise_renders_the_current_pointer_and_the_scheduled_delivery_rule(fired_context):
    """Saved rows from before Places carry a legacy carrier whose section rendered the copied pointer."""
    from supervisor.queue import check_scheduled_tasks
    from supervisor.schedule_lifecycle import upsert_scheduled_task

    root, pending, first_request = fired_context
    _pointer(root, "z", "Z: Ann asked for the backup report.")
    _pointer(root, "bbb", "B: Ann withdrew the question; nothing is needed.")
    legacy = {"binding_id": "legacy-binding", "transport_skill": "bridge", "instructions": "Reviewed behavior.",
              "event": {"conversation_key": "chat:team:k1", "provider": "chat", "account_id": "team",
                        "conversation_id": "k1", "thread_id": "", "source_event_id": "evt-legacy"},
              "previous_turn": {"conversation_key": "chat:team:k1", "task_id": "z", "outcome": "message",
                                "message": "Z: Ann asked for the backup report.", "transport_sends": [],
                                "finished_at": "2026-10-01T00:00:00+00:00", "delivery": "confirmed"}}
    upsert_scheduled_task({"id": "legacy-promise", "name": "Legacy follow-up", "source": "task_followup",
                           "trigger": {"type": "once", "run_at": DUE},
                           "task": {"type": "task", "text": "Check on Ann's report", "chat_id": 7,
                                    "metadata": {"presence": legacy, "origin_task_id": "legacy-origin"}}},
                          drive_root=root, host_followup={"followup_origin": {"task_id": "legacy-origin"},
                                                          "followup_relation": {"kind": "independent"}})
    check_scheduled_tasks()
    [fired] = pending
    text = first_request(fired)
    assert "B: Ann withdrew the question" in text and "Z: Ann asked for the backup report." not in text
    assert "This run started from a schedule" in text and "Use finish_task for the reply" not in text
