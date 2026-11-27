"""One web Reply on a work's card: which physical attempt answers, and which live readers hear it as context.

Invented works on a temporary data root, the real store, mailboxes, intake facts and Project bindings; only the
accepted owner row's quote is a fixed fake (``_host_quote`` is the host door's own concern).
"""
from __future__ import annotations

import json
import types

import pytest

from ouroboros import work_reply
from ouroboros.conversation_lease import follow_loop_admission, open_author_intake, root_intake
from ouroboros.owner_mailbox import _mailbox_path
from ouroboros.owner_relay import ROUTED_BY_HEAD
from ouroboros.server_routing_context import RootSnapshot
from ouroboros.task_results import write_task_result

QUOTE = {"source": {"address": "row:1@2026-11-23T10:00:00Z#aaaa", "chat_id": 1, "ts": "2026-11-23T10:00:00Z",
                    "label": "Main"}, "text": "Use the smaller set.", "attachments": [],
         "owner_row": {"text": "Use the smaller set.", "chat_id": 1, "ts": "2026-11-23T10:00:00Z"}}
SOURCE = {"chat_id": 1, "row_sha256": "f" * 64}


@pytest.fixture
def world(tmp_path, monkeypatch):
    from supervisor import queue

    monkeypatch.setattr(work_reply, "_host_quote", lambda *_a: dict(QUOTE))
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_k: None)
    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    return types.SimpleNamespace(root=tmp_path, ctx=types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[]))


def _root(tid, **extra):
    return {"id": tid, "chat_id": 1, "delegation_role": "root", "root_task_id": tid, **extra}


def _running(world, task, attempt=1, status="running", handoff=""):
    """The dispatcher's RUNNING row; ``handoff``: the ``phase_generation`` its put stamped on the row and the copy."""
    write_task_result(world.root, task["id"], status, chat_id=task["chat_id"], title=task["id"],
                      **{k: v for k, v in task.items() if k in ("delegation_role", "root_task_id", "original_task_id",
                                                                "timeout_retry_from", "supersedes_task_id")})
    if handoff:
        task = {**task, "_attempt": attempt, "_phase_generation": handoff}
    world.ctx.RUNNING[task["id"]] = {"task": task, "attempt": attempt, **({"phase_generation": handoff} if handoff else {})}
    return task


def _route(world, quoted, row="f"):
    """One accepted Reply; ``row`` names its source row, so two Replies are two relays, never one replayed."""
    return work_reply.route_work_reply(world.ctx, {"work_task_id": quoted, "address": f"task:{quoted}"},
                                       {**SOURCE, "row_sha256": row * 64}, None)


def _park(world, task, **controls):
    """The supervisor's exact park: the same attempt leaves RUNNING for PENDING with its controls."""
    world.ctx.RUNNING.pop(task["id"], None)
    parked = {key: value for key, value in task.items() if key != "_phase_generation"}
    world.ctx.PENDING.append({**parked, **controls})
    write_task_result(world.root, task["id"], "scheduled", chat_id=task["chat_id"], title=task["id"],
                      delegation_role="root", root_task_id=task["id"])
    return world.ctx.PENDING[-1]


SLEEP = {"_budget_pause": {"reason": "sleep", "sleep": {"sleep_id": "s"}}}


def _mail(world, tid):
    path = _mailbox_path(world.root, tid)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _retry(world):
    """``first`` hard-timed out; ``second`` is its retry: a fresh physical id under the same logical root."""
    write_task_result(world.root, "first", "interrupted", chat_id=1, title="first", delegation_role="root",
                      root_task_id="first", superseded_by="second", retry_task_id="second")
    return _running(world, _root("second", root_task_id="first", original_task_id="first",
                                 timeout_retry_from="first", supersedes_task_id="first"))


@pytest.mark.parametrize("quoted", ["first", "second", "child"])
def test_the_live_retry_answers_its_old_card_its_own_card_and_its_childs_and_counts_once(world, monkeypatch, quoted):
    from supervisor import queue

    _retry(world)
    if quoted == "child":
        write_task_result(world.root, "child", "running", chat_id=1, title="child", parent_task_id="first",
                          root_task_id="first", delegation_role="subagent")
    fence = {"status": "active", "owner_message_generation": 0, "task_id": "second"}
    monkeypatch.setitem(queue.ACCEPTANCE_FENCES, "first", fence)
    assert open_author_intake(world.root, "second", "work:second", attempt=1)
    delivery = _route(world, quoted)
    assert (delivery.outcome, delivery.target, delivery.logical, delivery.room) == ("delivered", "second", "first", 1)
    assert [row["work_reply"] for row in _mail(world, "second")] == [True]
    assert not _mail(world, "first") and not _mail(world, "child"), "never the finished predecessor or the child"
    assert fence["owner_message_generation"] == 1, "the logical root's acceptance fence revises its seal"
    assert _route(world, quoted).outcome == "delivered" and len(_mail(world, "second")) == 1
    assert fence["owner_message_generation"] == 1, "a replay neither writes nor counts again"


def test_the_logical_roots_owner_pause_holds_its_retry_and_the_room_answers(world, monkeypatch):
    from ouroboros import owner_pause

    _retry(world)
    asked = []
    monkeypatch.setattr(owner_pause, "read_fence", lambda _root, key: asked.append(key) or (
        {"state": owner_pause.FENCE_PAUSED} if key == "first" else {}))
    delivery = _route(world, "second")
    assert (delivery.outcome, delivery.reason, delivery.target) == ("fallback", "paused", "second")
    assert set(asked) == {"first"}, "Pause is read on the logical root, never on the physical id"
    assert [row["host_carried"]["handled_by_source_room"] for row in _mail(world, "second")] == [True]


# --- Old nested rows name only a parent: the root is proved through them, never guessed --------------------------------


def _old_child(world, tid, parent, **extra):
    """An older body's stored child row: it names its parent, never its tree's root."""
    write_task_result(world.root, tid, "completed", chat_id=1, title=tid, parent_task_id=parent,
                      delegation_role="subagent", **extra)


@pytest.mark.parametrize("shape", ["stored", "live_middle", "root_recorded_above", "retried_root", "under_the_retry"])
def test_an_old_grandchilds_card_reaches_the_attempt_of_the_root_its_parents_prove(world, shape):
    if shape in ("retried_root", "under_the_retry"):
        _retry(world)
        root, attempt = "first", "second"
    else:
        _running(world, _root("root"))
        root = attempt = "root"
    above = attempt if shape == "under_the_retry" else root
    recorded = {"root_task_id": root} if shape == "root_recorded_above" else {}
    _old_child(world, "mid", above, **recorded)
    if shape == "live_middle":
        world.ctx.RUNNING["mid"] = {"task": {"id": "mid", "chat_id": 1, "parent_task_id": root,
                                             "delegation_role": "subagent"}, "attempt": 1}
    _old_child(world, "leaf", "mid")
    assert open_author_intake(world.root, attempt, f"work:{attempt}", attempt=1)
    delivery = _route(world, "leaf")
    assert (delivery.outcome, delivery.target, delivery.logical) == ("delivered", attempt, root)
    assert [row.get("work_reply") for row in _mail(world, attempt)] == [True]
    assert not _mail(world, "mid") and not _mail(world, "leaf"), "no ancestor on the way is an addressee"


@pytest.mark.parametrize("broken", [
    "missing_parent", "cycle", "own_parent", "top_is_no_root", "recorded_root_missing", "recorded_root_is_a_child",
    "recorded_root_names_another", "unreadable_parent", "past_the_bound", "a_places_root"])
def test_an_old_nested_card_whose_ancestry_proves_no_owner_root_names_no_addressee(world, monkeypatch, broken):
    """An eligible-looking root runs beside each chain; a chain that does not prove it writes nothing anywhere."""
    from ouroboros import task_results

    _running(world, _root("root", **(SHAPES["place_delegated"][0] if broken == "a_places_root" else {})))
    assert open_author_intake(world.root, "root", "work:root", attempt=1)
    # a parentless running row that records another root and is no retry of it: no root itself
    _running(world, _root("stray", root_task_id="elsewhere"))
    assert open_author_intake(world.root, "stray", "work:stray", attempt=1)
    above = {"missing_parent": "gone", "cycle": "leaf", "top_is_no_root": "top"}.get(broken, "root")
    recorded = {"recorded_root_missing": "gone", "recorded_root_is_a_child": "other",
                "recorded_root_names_another": "stray"}.get(broken)
    write_task_result(world.root, "top", "completed", chat_id=1, title="top", delegation_role="subagent")
    _old_child(world, "other", "root")
    _old_child(world, "mid", above, **({"root_task_id": recorded} if recorded else {}))
    _old_child(world, "leaf", "leaf" if broken == "own_parent" else "mid")
    if broken == "past_the_bound":
        monkeypatch.setattr(work_reply, "_ANCESTOR_HOPS", 1)
    if broken == "unreadable_parent":
        stored = task_results.load_task_result

        def read(root, tid, *args, **kwargs):
            if tid == "mid" and kwargs.get("strict"):
                raise ValueError("unreadable task authority")
            return stored(root, tid, *args, **kwargs)

        monkeypatch.setattr(task_results, "load_task_result", read)
    delivery = _route(world, "leaf")
    assert (delivery.outcome, delivery.reason, delivery.target, delivery.logical) == ("fallback", "unresolved", "", "")
    assert delivery.quote == QUOTE, "the source room still answers the accepted words"
    assert not any(_mail(world, tid) for tid in ("root", "stray", "top", "other", "mid", "leaf", "gone"))


@pytest.mark.parametrize("control, reason", [("stop", "not_reading"), ("owner_pause", "paused"), ("hold", "paused")])
def test_the_proved_roots_own_controls_judge_an_old_grandchilds_reply(world, monkeypatch, control, reason):
    from ouroboros import owner_pause
    from ouroboros.cancel_intents import request_cancel

    _running(world, _root("root", **({"_owner_hold": True} if control == "hold" else {})))
    _old_child(world, "mid", "root")
    _old_child(world, "leaf", "mid")
    assert open_author_intake(world.root, "root", "work:root", attempt=1)
    asked = []
    if control == "stop":
        request_cancel(world.root, "root", reason="owner stop")
    if control == "owner_pause":
        monkeypatch.setattr(owner_pause, "read_fence", lambda _root, key: asked.append(key) or (
            {"state": owner_pause.FENCE_PAUSED} if key == "root" else {}))
    delivery = _route(world, "leaf")
    assert (delivery.outcome, delivery.reason, delivery.target, delivery.logical) == ("fallback", reason, "root", "root")
    assert not any(row.get("work_reply") for row in _mail(world, "root")), "no active Reply past a control"
    assert control != "owner_pause" or set(asked) == {"root"}, "Pause is read on the proved root"


@pytest.mark.parametrize("state, expected", [
    ("queued", "delivered"),     # just started or woken back to PENDING: its first drain reads the mailbox
    ("preparing", "delivered"),  # RUNNING, no intake fact of this attempt yet
    ("closed", "not_reading"),   # this attempt's final drain closed its intake while the row still runs
    ("resumed", "delivered"),    # attempt 2: attempt 1's closed fact proves nothing about it
    ("settled", "not_reading"),  # its result settled; only post-work remains
])
def test_a_live_work_without_open_intake_takes_the_reply_unless_its_last_drain_closed(world, state, expected):
    task = _root("work")
    if state == "queued":
        write_task_result(world.root, "work", "requested", chat_id=1, title="work", delegation_role="root")
        world.ctx.PENDING.append(task)
    else:
        _running(world, task, attempt=2 if state == "resumed" else 1,
                 status="completed" if state == "settled" else "running")
    if state in {"closed", "resumed"}:
        assert open_author_intake(world.root, "work", "work:work", attempt=1)
        follow_loop_admission(types.SimpleNamespace(drive_root=world.root, task_id="work", task_attempt=1,
                                                    task_metadata={}), False)
    delivery = _route(world, "work")
    assert (delivery.outcome if expected == "delivered" else delivery.reason) == expected
    assert bool(_mail(world, "work")) is (expected == "delivered"), "a non-reader's mailbox gets no active Reply"


def test_a_resumed_owner_turn_reads_under_a_work_fact_whose_final_close_ends_the_reply(world, monkeypatch):
    from ouroboros import conversation_lease

    monkeypatch.setattr(conversation_lease, "owner_conversation_key", lambda _task: "owner:1")
    turn = _running(world, {**_root("turn"), "_attempt": 1})
    assert open_author_intake(world.root, "turn", "owner:1", attempt=1)  # attempt 1: its carrier's own fact
    with root_intake(world.root, turn):
        fact = json.loads(conversation_lease._intake_path(world.root, "turn").read_text())
        assert fact["conversation_key"] == "owner:1", "the carrier's fact is never replaced"
    conversation_lease.close_author_intake(world.root, "turn")  # CarrierTurn.ended()
    resumed = _running(world, {**_root("turn"), "_attempt": 2}, attempt=2)
    ctx = types.SimpleNamespace(drive_root=world.root, task_id="turn", task_attempt=2, task_metadata={})
    with root_intake(world.root, resumed):
        assert _route(world, "turn").outcome == "delivered"
        follow_loop_admission(ctx, False)  # the resumed loop's final drain
        assert _route(world, "turn").reason == "not_reading", "RUNNING alone never reopens a closed last drain"


def test_a_vanished_work_still_leaves_passive_context_with_the_source_rooms_readers(world):
    watcher = _running(world, _root("watcher"))
    delivery = _route(world, "gone")
    assert (delivery.outcome, delivery.reason, delivery.target) == ("fallback", "unresolved", "")
    assert delivery.quote == QUOTE, "the accepted words are quoted whatever happened to their destination"
    roots = RootSnapshot(pairs=(({"task_id": watcher["id"], "status": "running", "room_reader": True}, frozenset({1})),))
    work_reply.carry_reply_context(world.ctx, roots, 1, delivery)
    (entry,) = _mail(world, "watcher")
    assert entry["host_carried"] == {"went_to": ROUTED_BY_HEAD, "handled_by_source_room": True, "context_only": True}


def test_a_reader_moved_to_another_project_is_no_reader_of_its_old_chat_at_choice_or_at_delivery(world, monkeypatch):
    from ouroboros import owner_relay
    from ouroboros.projects_registry import bind_task_to_project, create_project

    nova = int(create_project(world.root, "nova", name="Nova")["chat_id"])
    target, moved, late, stays = (_running(world, _root(tid)) for tid in ("target", "moved", "late", "stays"))
    bind_task_to_project(world.root, "moved", "nova", nova, origin={"absent": "system"})
    assert open_author_intake(world.root, "target", "work:target", attempt=1)
    delivery = _route(world, "target")
    assert delivery.outcome == "delivered" and delivery.room == 1
    real = owner_relay.write_relay

    def move_late_first(drive, tid, **kwargs):
        if tid == "late":  # its Project changes while the context waits for its mail lock
            bind_task_to_project(world.root, "late", "nova", nova, origin={"absent": "system"})
        return real(drive, tid, **kwargs)

    monkeypatch.setattr(owner_relay, "write_relay", move_late_first)
    roots = RootSnapshot(pairs=tuple(({"task_id": t["id"], "status": "running", "room_reader": True},
                                      frozenset({1, nova} if t["id"] == "moved" else {1}))
                                     for t in (target, moved, late, stays)))
    work_reply.carry_reply_context(world.ctx, roots, 1, delivery)
    assert [row["host_carried"]["went_to"] for row in _mail(world, "stays")] == ["target"]
    assert not _mail(world, "moved") and not _mail(world, "late"), "the old chat_id alone makes no reader"
    assert [row.get("work_reply") for row in _mail(world, "target")] == [True], "the answering work gets no copy"


def test_the_answering_work_moved_while_its_reply_waits_names_its_new_room(world, monkeypatch):
    from ouroboros import owner_relay
    from ouroboros.projects_registry import bind_task_to_project, create_project

    nova = int(create_project(world.root, "nova", name="Nova")["chat_id"])
    _running(world, _root("target"))
    assert open_author_intake(world.root, "target", "work:target", attempt=1)
    real = owner_relay.write_relay

    def move_first(drive, tid, **kwargs):
        bind_task_to_project(world.root, "target", "nova", nova, origin={"absent": "system"})
        return real(drive, tid, **kwargs)

    monkeypatch.setattr(owner_relay, "write_relay", move_first)
    delivery = _route(world, "target")
    assert (delivery.outcome, delivery.room) == ("delivered", nova), "the room readers hear is the one it answers in"


@pytest.mark.parametrize("fence, moved", [({"status": "active"}, True), ({"status": "sealed"}, False), (None, False)])
def test_owner_words_count_only_on_an_active_acceptance_fence(fence, moved):
    assert work_reply.owner_message_landed(fence) is moved
    assert (fence or {}).get("owner_message_generation", 0) == int(moved)


# --- The two edges of the contract: the loop's end, and the tree's passive copy -------------------------------------


def test_the_reply_after_the_loops_last_drain_and_after_its_end_reaches_the_source_room(world):
    """The distinguishing flow on the real intake: preparing reads, the final drain closes, and the loop's end (the
    ``with`` exited, the result not stored yet, the row still RUNNING) still reads nothing — the source room answers,
    not a mailbox no loop drains again."""
    task = _running(world, _root("work"))
    ctx = types.SimpleNamespace(drive_root=world.root, task_id="work", task_attempt=1, task_metadata={})
    with root_intake(world.root, task):
        assert _route(world, "work").outcome == "delivered"  # preparing/reading: its first drain takes it
        follow_loop_admission(ctx, False)  # the final drain is about to read
        assert _route(world, "work").reason == "not_reading"  # the last drain closed it
    assert _route(world, "work").reason == "not_reading", "after the loop, before the stored result, the room answers"
    assert [row.get("work_reply") for row in _mail(world, "work")] == [True], "no active Reply after the last drain"


def test_a_crashed_writers_open_intake_is_no_preparation(world, monkeypatch):
    """A vanished/dead writer's standing open fact reads nothing: it is not the preparing branch that takes a Reply."""
    from ouroboros import conversation_lease

    _running(world, _root("work"))
    assert open_author_intake(world.root, "work", "work:work", attempt=1)
    monkeypatch.setattr(conversation_lease, "_writer_alive", lambda _pid: False)  # the writer is gone
    assert _route(world, "work").reason == "not_reading", "a dead writer's open fact is not preparation"
    assert not _mail(world, "work")


def test_an_addressed_reply_carries_passive_copies_to_its_own_live_helpers_at_receipt(world, monkeypatch):
    from ouroboros import delegate_interactions

    placed = []  # the shared receipt carriage frames ordinary rows into open sessions; a passive copy takes none
    monkeypatch.setattr(delegate_interactions, "place_at_receipt", lambda *args: placed.append(args))
    _running(world, _root("target"))
    write_task_result(world.root, "helper", "running", chat_id=1, title="helper",
                      parent_task_id="target", root_task_id="target", delegation_role="subagent")
    assert open_author_intake(world.root, "target", "work:target", attempt=1)
    delivery = _route(world, "target")
    assert (delivery.outcome, delivery.logical, delivery.target) == ("delivered", "target", "target")
    work_reply.carry_reply_to_members(world.root, delivery)
    (entry,) = _mail(world, "helper")
    assert entry["host_carried"] == {"went_to": "target", "context_only": True}
    assert not entry.get("work_reply"), "a passive copy owes no answer"
    assert [row.get("work_reply") for row in _mail(world, "target")] == [True], "the answering work gets no extra copy"
    work_reply.carry_reply_to_members(world.root, delivery)  # a retried carriage writes nothing more
    assert len(_mail(world, "helper")) == 1
    assert not placed, "a passive copy is not a delivery: no session frame"


def test_a_physical_retrys_helpers_under_the_logical_root_get_the_passive_copy(world):
    _retry(world)  # ``first`` interrupted; ``second`` runs under the logical root ``first``
    write_task_result(world.root, "helper", "running", chat_id=1, title="helper",
                      parent_task_id="first", root_task_id="first", delegation_role="subagent")
    assert open_author_intake(world.root, "second", "work:second", attempt=1)
    delivery = _route(world, "first")  # the old card names the logical root
    assert (delivery.outcome, delivery.target, delivery.logical) == ("delivered", "second", "first")
    work_reply.carry_reply_to_members(world.root, delivery)
    (entry,) = _mail(world, "helper")
    assert entry["host_carried"] == {"went_to": "second", "context_only": True}
    assert not _mail(world, "first"), "the settled predecessor gets nothing"
    assert [row.get("work_reply") for row in _mail(world, "second")] == [True], "the retry answers; no copy of its own Reply"


def test_the_recipients_drain_catch_up_writes_the_same_passive_copy_once(world, monkeypatch):
    """The receipt copy and the recipient's drain catch-up (``carry_to_members`` over the held ``work_reply`` row) name
    the same passive entry, so the catch-up writes nothing new — no repeat, no conflicting bytes."""
    from ouroboros.owner_relay import carry_to_members, owner_quote, owner_row_address

    row = {"direction": "in", "source": "web", "chat_id": 1, "text": "Use the smaller set.",
           "ts": "2026-11-23T10:00:00Z", "client_message_id": "owner-reply"}
    quote = owner_quote(world.root, {**row, "_owner_source_root": str(world.root)}, address=owner_row_address(row))
    monkeypatch.setattr(work_reply, "_host_quote", lambda *_a: dict(quote))
    _running(world, _root("target"))
    write_task_result(world.root, "helper", "running", chat_id=1, title="helper",
                      parent_task_id="target", root_task_id="target", delegation_role="subagent")
    assert open_author_intake(world.root, "target", "work:target", attempt=1)
    delivery = _route(world, "target")
    assert delivery.outcome == "delivered"
    work_reply.carry_reply_to_members(world.root, delivery)
    assert len(_mail(world, "helper")) == 1
    # the recipient holds its addressed Reply as inject_relay leaves it, then its drain reconciles the tree
    held = {**quote["owner_row"], "_owner_source_root": str(world.root), "carried": {"went_to": "", "work_reply": True}}
    ctx = types.SimpleNamespace(drive_root=world.root, task_id="target", task_attempt=1,
                                task_metadata={"root_task_id": "target", "delegation_role": "root"},
                                _held_owner_messages={quote["source"]["address"]: held},
                                _carriage_memo=None, _last_scheduled_subagents=[])
    carry_to_members(ctx, delivered=True)
    assert len(_mail(world, "helper")) == 1, "the catch-up recognizes the receipt copy and writes nothing new"


# --- One attempt, several handoffs: a loop's close or dead writer speaks for its own handoff alone --------------------


def _closed_by_loop(world, task):
    """The loop's own scope ran and ended (``agent.handle_task``): its positive close stands."""
    with root_intake(world.root, task):
        pass


def test_two_replies_across_one_same_attempt_redispatch_both_reach_the_work(world):
    """A cold sleep leaves the loop without a terminal (``BudgetPauseRequested``), the supervisor parks the SAME attempt
    (``install_exact_budget_pause`` keeps ``_attempt``) and its wake hands it out again under a fresh
    ``phase_generation``. Parking, asleep and preparing again it takes the words; only the final close ends them."""
    from ouroboros.budget_pause import BudgetPauseRequested
    from ouroboros.conversation_lease import _intake_path

    first = _running(world, _root("work"), handoff="g1")
    with pytest.raises(BudgetPauseRequested), root_intake(world.root, first):
        assert _route(world, "work", row="a").outcome == "delivered"  # handoff g1 reads
        raise BudgetPauseRequested({"pause_id": "p", "rail": "model_sleep"})  # its loop leaves for a cold sleep
    assert not _intake_path(world.root, "work").exists(), "a park ends no reading: no close is left for it"
    assert _route(world, "work", row="b").outcome == "delivered", "parking: still RUNNING, the words wait for its wake"
    _park(world, first, **SLEEP)  # the supervisor's park: the same attempt waits in PENDING
    assert _route(world, "work", row="c").outcome == "delivered", "an unheld cold sleep wakes for it"
    world.ctx.PENDING.clear()
    second = _running(world, _root("work"), handoff="g2")  # the wake: attempt 1 again, a fresh handoff
    assert _route(world, "work", row="d").outcome == "delivered", "preparing again"
    with root_intake(world.root, second):
        assert _route(world, "work", row="e").outcome == "delivered"
        follow_loop_admission(types.SimpleNamespace(drive_root=world.root, task_id="work", task_attempt=1,
                                                    task_metadata={}), False)  # g2's final drain
        assert _route(world, "work", row="7").reason == "not_reading", "its own final close still ends the Reply"
    assert _route(world, "work", row="9").reason == "not_reading", "and so does its loop's end before the result"
    assert json.loads(_intake_path(world.root, "work").read_text())["phase_generation"] == "g2"
    assert [row["work_reply"] for row in _mail(world, "work")] == [True] * 5, "a to e: each once, none lost"


def test_a_close_naming_an_earlier_handoff_of_the_attempt_does_not_close_this_one(world):
    """Defence in depth for the close a final end leaves: should the attempt be handed out again, the earlier
    handoff's close is not this handoff's (``conversation_lease._same_handoff``)."""
    import os

    from ouroboros import conversation_lease

    _running(world, _root("work"), handoff="g2")
    conversation_lease._write_intake(world.root, "work", {"accepting": False, "conversation_key": "work:work",
                                                          "pid": os.getpid(), "attempt": 1, "phase_generation": "g1"})
    assert _route(world, "work").outcome == "delivered"
    conversation_lease._write_intake(world.root, "work", {"accepting": False, "conversation_key": "work:work",
                                                          "pid": os.getpid(), "attempt": 1, "phase_generation": "g2"})
    assert _route(world, "work", row="a").reason == "not_reading", "its own handoff's close ends it"


def test_a_dead_writers_open_fact_ends_only_its_own_handoff(world, monkeypatch):
    """A planned restart keeps a warm wait's attempt (``workers``' ``continuing_wait``) and its process dies: the open
    fact it left reads nothing while that handoff stands, and the same attempt handed out again prepares."""
    from ouroboros import conversation_lease

    _running(world, _root("work"), handoff="g1")
    conversation_lease._write_intake(world.root, "work", {"accepting": True, "conversation_key": "work:work",
                                                          "pid": 1, "attempt": 1, "phase_generation": "g1"})
    monkeypatch.setattr(conversation_lease, "_writer_alive", lambda _pid: False)  # its writer is gone
    assert _route(world, "work", row="a").reason == "not_reading", "the dead writer's own handoff reads nothing"
    _running(world, _root("work"), handoff="g2")  # the preserved successor's handoff: attempt 1 again
    assert _route(world, "work", row="b").outcome == "delivered", "preparing: g1's dead writer is not g2's"
    assert [row["work_reply"] for row in _mail(world, "work")] == [True]


def test_a_fact_naming_no_handoff_is_judged_by_its_attempt_as_before(world):
    """An older writer's close (written before facts named handoffs) still ends this attempt: no handoff on one side
    leaves the attempt to decide, so an upgrade never turns a closed loop into a reader."""
    import os

    from ouroboros import conversation_lease

    _running(world, _root("work"), handoff="g2")
    conversation_lease._write_intake(world.root, "work", {"accepting": False, "conversation_key": "work:work",
                                                          "pid": os.getpid(), "attempt": 1})
    assert _route(world, "work").reason == "not_reading"
    assert not _mail(world, "work")


def test_a_loops_intake_fact_names_its_handoff_through_its_drain_and_end(world, monkeypatch):
    from ouroboros import conversation_lease
    from ouroboros.conversation_lease import _intake_path, author_intake_closed
    from supervisor.worker_chat_lane import open_owner_intake

    task = _running(world, _root("work"), handoff="g1")
    with root_intake(world.root, task):
        follow_loop_admission(types.SimpleNamespace(drive_root=world.root, task_id="work", task_attempt=1,
                                                    task_metadata={}), False)
    fact = json.loads(_intake_path(world.root, "work").read_text())
    assert (fact["accepting"], fact["attempt"], fact["phase_generation"]) == (False, 1, "g1")
    assert author_intake_closed(world.root, "work", "", attempt=1, handoff="g1")
    assert not author_intake_closed(world.root, "work", "", attempt=1, handoff="g2")
    assert author_intake_closed(world.root, "work", "", attempt=1), "a reader naming no handoff: the attempt decides"
    monkeypatch.setattr(conversation_lease, "owner_conversation_key", lambda _task: "owner:1")
    turn = {**_root("turn"), "_attempt": 1}
    assert open_owner_intake(turn, world.root) == "owner:1"  # the host's fact, before any put
    assert "phase_generation" not in json.loads(_intake_path(world.root, "turn").read_text())
    assert open_owner_intake({**turn, "_phase_generation": "g1"}, world.root) == "owner:1"  # its carrier's take
    assert json.loads(_intake_path(world.root, "turn").read_text())["phase_generation"] == "g1"


# --- A work no first drain will reach cannot take the source room's Reply -------------------------------------------


@pytest.mark.parametrize("hold", ["_owner_hold", "_project_admission_restore_hold", "_cancel_intent_authority_hold"])
def test_a_held_queued_work_cannot_take_the_reply_and_keeps_it_as_context(world, hold):
    """``worker_assignment`` dispatches none of these rows (a restore-time unknown cancel authority is resolved before
    any dispatch): the source room answers and the work finds the words as passive context at a later drain."""
    write_task_result(world.root, "work", "requested", chat_id=1, title="work", delegation_role="root")
    world.ctx.PENDING.append({**_root("work"), hold: {"reason": "test"}})
    delivery = _route(world, "work")
    assert (delivery.outcome, delivery.reason, delivery.target) == ("fallback", "paused", "work")
    assert [(row.get("work_reply"), row["host_carried"].get("handled_by_source_room"))
            for row in _mail(world, "work")] == [(None, True)], "context only: no active Reply, no answer owed"


BINDING = "binding-1"
DOOR = {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-late-answer", "ts": "2026-10-29T13:00:00Z"}}
AUTHORITY = {"presence_binding_authority": {"binding_id": BINDING}}
SHAPES = {  # the records tests/test_presence_related_owner_work.py illustrates: how an owner's work, old and new, reads
    "plain": ({}, True),
    "related": ({"metadata": {"presence_binding_relation": {"binding_id": BINDING}}}, True),
    "old_owner_root": ({"metadata": {"work_kind": "owner", **AUTHORITY}, "source": "presence_promote", **DOOR}, True),
    "place_delegated": ({"metadata": AUTHORITY}, False),
    "ceiling_only": ({"task_contract": {"capability_ceiling": {}}}, False),
}


@pytest.mark.parametrize("shape", SHAPES)
def test_an_addressed_reply_reaches_the_owners_work_by_its_whole_record_never_a_places(world, shape):
    extra, answers = SHAPES[shape]
    _running(world, _root("work", **extra))
    assert open_author_intake(world.root, "work", "work:work", attempt=1)
    delivery = _route(world, "work")
    assert (delivery.outcome, delivery.reason if not answers else "") == (
        ("delivered", "") if answers else ("fallback", "unresolved")), "a place's own work is no recipient"
    assert [row.get("work_reply") for row in _mail(world, "work")] == ([True] if answers else [])


@pytest.mark.parametrize("carrier", ["task", "metadata", "contract"])
def test_a_work_that_withholds_its_return_tool_cannot_take_the_reply_and_keeps_it_as_context(world, carrier):
    """Active delivery owes an answer in the source room through ``send_user_message``; with no way to give it the room
    answers (the work is named, its saved result readable) and the work still finds the words as context."""
    off = ["send_user_message"]
    extra = {"task": {"disabled_tools": off}, "metadata": {"metadata": {"disabled_tools": off}},
             "contract": {"task_contract": {"disabled_tools": off}}}[carrier]
    _running(world, _root("work", **extra))
    assert open_author_intake(world.root, "work", "work:work", attempt=1)
    delivery = _route(world, "work")
    assert (delivery.outcome, delivery.reason, delivery.target) == ("fallback", "cannot_answer", "work")
    assert work_reply.work_reply_fallback(delivery)["reader"]["arguments"]["task_id"] == "work"
    assert [(row.get("work_reply"), row["host_carried"].get("handled_by_source_room"))
            for row in _mail(world, "work")] == [(None, True)], "context only: no active Reply, no answer owed"


def test_withholding_any_other_tool_leaves_the_reply_with_its_work(world):
    _running(world, _root("work", disabled_tools=["send_photo", "show_card", "web_search"]))
    assert open_author_intake(world.root, "work", "work:work", attempt=1)
    assert _route(world, "work").outcome == "delivered"
    assert [row.get("work_reply") for row in _mail(world, "work")] == [True]


@pytest.mark.parametrize("standing, outcome, reason, mail", [
    ("reading", "fallback", "cannot_answer", [(None, True)]),
    ("queued", "fallback", "cannot_answer", [(None, True)]),  # its first drain reads, but it could not answer
    ("held", "fallback", "paused", [(None, True)]),  # the stronger state names itself
    ("closed", "fallback", "not_reading", []),  # no drain ever: nothing is written
])
def test_the_missing_return_tool_only_changes_what_would_have_been_a_delivery(world, standing, outcome, reason, mail):
    task = _root("work", disabled_tools=["send_user_message"])
    if standing == "reading":
        _running(world, task)
        assert open_author_intake(world.root, "work", "work:work", attempt=1)
    elif standing == "closed":
        _running(world, task)
        assert open_author_intake(world.root, "work", "work:work", attempt=1)
        follow_loop_admission(types.SimpleNamespace(drive_root=world.root, task_id="work", task_attempt=1,
                                                    task_metadata={}), False)
    else:
        write_task_result(world.root, "work", "requested", chat_id=1, title="work", delegation_role="root")
        world.ctx.PENDING.append({**task, **({"_owner_hold": {"reason": "test"}} if standing == "held" else {})})
    delivery = _route(world, "work")
    assert (delivery.outcome, delivery.reason) == (outcome, reason)
    assert [(row.get("work_reply"), row["host_carried"].get("handled_by_source_room"))
            for row in _mail(world, "work")] == mail


def test_a_return_tool_withheld_while_the_reply_waits_for_its_mail_lock_gives_the_words_to_the_room(world, monkeypatch):
    """The recipient is judged again under its mail lock, return capability included: never an active Reply a work
    that has since lost the means to answer."""
    from ouroboros import owner_relay

    task = _running(world, _root("work"))
    assert open_author_intake(world.root, "work", "work:work", attempt=1)
    real, withheld = owner_relay.write_relay, []

    def withhold_first(drive, tid, **kwargs):
        if kwargs.get("work_reply") and not withheld:  # chosen as a reader that answers; the tool goes before its lock
            withheld.append(tid)
            world.ctx.RUNNING[tid]["task"] = {**task, "disabled_tools": ["send_user_message"]}
        return real(drive, tid, **kwargs)

    monkeypatch.setattr(owner_relay, "write_relay", withhold_first)
    delivery = _route(world, "work")
    assert withheld == ["work"] and (delivery.outcome, delivery.reason) == ("fallback", "cannot_answer")
    assert [row.get("work_reply") for row in _mail(world, "work")] == [None], "no active Reply"
    assert [row["host_carried"]["handled_by_source_room"] for row in _mail(world, "work")] == [True]


def test_only_real_mailbox_readers_hear_a_reply_as_context_and_none_needs_a_return_tool(world):
    """The roster's reader flag is a first look; a deep review runs no loop, a place's work or a ceiling-bound task is
    no owner's reader, an old owner root is, and a work without its return tool may still observe."""
    _running(world, _root("target"))
    assert open_author_intake(world.root, "target", "work:target", attempt=1)
    delivery = _route(world, "target")
    assert delivery.outcome == "delivered"
    readers = {"old_owner_root": SHAPES["old_owner_root"][0], "plain": {}, "observer": {"disabled_tools": ["send_user_message"]},
               "review": {"type": "deep_self_review"}, "place": SHAPES["place_delegated"][0],
               "ceiling": SHAPES["ceiling_only"][0]}
    tasks = {tid: _running(world, _root(tid, **extra)) for tid, extra in readers.items()}
    roots = RootSnapshot(pairs=tuple(({"task_id": tid, "status": "running", "room_reader": True}, frozenset({1}))
                                     for tid in tasks))
    work_reply.carry_reply_context(world.ctx, roots, 1, delivery)
    heard = {tid for tid in tasks if _mail(world, tid)}
    assert heard == {"old_owner_root", "plain", "observer"}
    for tid in heard:
        (entry,) = _mail(world, tid)
        assert entry["host_carried"] == {"went_to": "target", "context_only": True} and not entry.get("work_reply")


@pytest.mark.parametrize("when", ["after_the_census", "at_the_lock"])
@pytest.mark.parametrize("state, hears", [
    ("reading", True),    # control: its intake is open
    ("preparing", True),  # control: no fact of this handoff yet, so its first drain observes
    ("closed", False),    # its last drain closed while its row still runs
    ("held", False),      # a dispatch hold on its running row
])
def test_an_observer_that_stops_reading_after_the_census_gets_no_passive_copy(world, monkeypatch, when, state, hears):
    """The roster's reader flag is a census: it says a running root may read. The same reader rule as the addressed
    work's (no return tool) is read before the copy is written and again inside its locked admit, so a root that closed
    its last drain or took a hold since hears nothing, and a reading or preparing one still does."""
    from ouroboros import owner_relay

    _running(world, _root("target"))
    assert open_author_intake(world.root, "target", "work:target", attempt=1)
    delivery = _route(world, "target")
    assert delivery.outcome == "delivered"
    _running(world, _root("observer"))
    if state != "preparing":
        assert open_author_intake(world.root, "observer", "work:observer", attempt=1)
    roots = RootSnapshot(pairs=(({"task_id": "observer", "status": "running", "room_reader": True}, frozenset({1})),))

    def stop_reading():
        if state == "closed":
            follow_loop_admission(types.SimpleNamespace(drive_root=world.root, task_id="observer", task_attempt=1,
                                                        task_metadata={}), False)
        elif state == "held":
            world.ctx.RUNNING["observer"]["task"]["_owner_hold"] = {"reason": "test"}

    real, writes = owner_relay.write_relay, []

    def write_relay(drive, tid, **kwargs):
        writes.append(tid)
        if when == "at_the_lock":
            stop_reading()  # after the choice, before the mail lock's admit
        return real(drive, tid, **kwargs)

    if when == "after_the_census":
        stop_reading()
    monkeypatch.setattr(owner_relay, "write_relay", write_relay)
    work_reply.carry_reply_context(world.ctx, roots, 1, delivery)
    assert writes == ([] if when == "after_the_census" and not hears else ["observer"]), "refused before or at the write"
    assert [row["host_carried"] for row in _mail(world, "observer")] == (
        [{"went_to": "target", "context_only": True}] if hears else [])


@pytest.mark.parametrize("queued", [False, True], ids=["running", "queued"])
def test_a_deep_self_review_runs_no_mail_loop_so_the_source_room_answers(world, queued):
    review = _root("review", type="deep_self_review")
    if queued:
        write_task_result(world.root, "review", "requested", chat_id=1, title="review", delegation_role="root")
        world.ctx.PENDING.append(review)
    else:
        _running(world, review, handoff="g1")  # running, no intake fact: not "preparing", it never opens one
    delivery = _route(world, "review")
    assert (delivery.outcome, delivery.reason, delivery.target) == ("fallback", "not_reading", "review")
    assert not _mail(world, "review"), "nothing would ever drain it"


# --- A transition while the Reply waits for the work's mail lock ------------------------------------------------------


@pytest.mark.parametrize("parked_as, outcome, reason, mail", [
    ("cold_sleep", "delivered", "reading", [True]),   # its sleep wakes for the words
    ("owner_pause", "fallback", "paused", [None]),    # the room answers; the work keeps them as context
    ("other_drive", "fallback", "not_reading", []),   # never another mailbox than the one verified
])
def test_a_reading_work_parked_while_the_reply_waits_for_its_mail_lock_is_judged_as_it_is_now(
        world, monkeypatch, parked_as, outcome, reason, mail):
    from ouroboros import owner_relay
    from ouroboros.conversation_lease import close_author_intake

    task = _running(world, _root("work"), handoff="g1")
    assert open_author_intake(world.root, "work", "work:work", attempt=1)
    controls = {"cold_sleep": SLEEP, "owner_pause": {"_budget_pause": {"reason": "owner"}},
                "other_drive": {**SLEEP, "drive_root": str(world.root / "elsewhere")}}[parked_as]
    real, moved = owner_relay.write_relay, []

    def park_first(drive, tid, **kwargs):
        if kwargs.get("work_reply") and not moved:  # chosen while RUNNING; parked before its mail lock
            moved.append(tid)
            close_author_intake(world.root, "work", own=True, attempt=1)  # its loop left for the park ...
            _park(world, task, **controls)  # ... and the supervisor parked the same attempt
        return real(drive, tid, **kwargs)

    monkeypatch.setattr(owner_relay, "write_relay", park_first)
    delivery = _route(world, "work")
    assert moved == ["work"] and (delivery.outcome, delivery.reason) == (outcome, reason)
    assert [row.get("work_reply") for row in _mail(world, "work")] == mail
    assert not (world.root / "elsewhere").exists(), "nothing written beside a drive it was not chosen on"


# --- Settlement retires the close a loop leaves ----------------------------------------------------------------------


def test_settlement_retires_the_loops_closed_intake_also_of_a_work_that_never_had_mail(world):
    """The task-done seam and the off-loop sweep share ``task_custody.settle_task_mailbox``: a settled work's closed
    intake fact goes with its mailbox, and a work with no mailbox at all is still listed by its fact."""
    from ouroboros.conversation_lease import _intake_path
    from ouroboros.owner_mailbox import cleanup_task_mailbox, sweep_settled_owner_mailboxes

    for tid in ("done", "late", "live"):
        _closed_by_loop(world, _running(world, _root(tid), handoff="g1"))  # no mail ever written for any of them
        assert json.loads(_intake_path(world.root, tid).read_text())["accepting"] is False
    for tid in ("done", "late"):
        write_task_result(world.root, tid, "completed", chat_id=1, title=tid, result="done")
    assert cleanup_task_mailbox(world.root, "done") and not _intake_path(world.root, "done").exists(), "task-done seam"
    report = sweep_settled_owner_mailboxes(world.root)
    assert report["removed"] == ["late"] and not _intake_path(world.root, "late").exists(), "the sweep lists its fact"
    assert _intake_path(world.root, "live").exists() and report["kept"] == 1, "an unsettled work keeps its close"


# --- A park is durable before its loop leaves: until the row is parked, the park itself decides -----------------------


def _saved_park(world, tid, reason, *, attempt=1, state="paused"):
    """The loop's own durable write before it raises ``BudgetPauseRequested`` (``budget_pause.set_budget_pause``)."""
    from ouroboros.budget_pause import RAIL_MODEL_SLEEP, RAIL_OWNER_PAUSE, set_budget_pause

    rail = {"sleep": RAIL_MODEL_SLEEP, "owner": RAIL_OWNER_PAUSE}.get(reason, "global_exhausted")
    return set_budget_pause(world.root, tid, {
        "pause_id": f"p-{reason}-{attempt}", "state": state, "task_attempt": attempt, "reason": reason, "rail": rail,
        "scope": "root", "root_task_id": tid, "paused_at": 1.0, "source_ref": f"src-{tid}", "physical_calls": 1,
        **({"sleep": {"sleep_id": "s", "until": "2099-01-01T00:00:00Z"}} if reason == "sleep" else {})})


@pytest.mark.parametrize("reason, outcome", [
    ("budget", ("fallback", "paused")),     # money: it waits for a Resume, so the room answers
    ("owner", ("fallback", "paused")),      # a Pause
    ("authority", ("fallback", "paused")),  # a technical park
    ("sleep", ("delivered", "reading")),    # a voluntary unheld sleep wakes for the words
])
def test_a_park_saved_before_its_loop_leaves_decides_the_reply_until_the_row_is_parked(world, reason, outcome):
    from ouroboros.budget_pause import BudgetPauseRequested

    task = _running(world, _root("work"), handoff="g1")
    with pytest.raises(BudgetPauseRequested), root_intake(world.root, task):
        assert _route(world, "work", row="a").outcome == "delivered"  # reading
        pause = _saved_park(world, "work", reason)
        inside = _route(world, "work", row="b")  # its fact still open, the park already durable
        raise BudgetPauseRequested(pause)
    after = _route(world, "work", row="c")  # the loop left; the row is RUNNING until the supervisor parks it
    assert (inside.outcome, inside.reason) == (after.outcome, after.reason) == outcome
    mail = _mail(world, "work")
    assert [row.get("work_reply") for row in mail].count(True) == (3 if reason == "sleep" else 1), "a park holds none"
    assert sum(1 for row in mail if (row.get("host_carried") or {}).get("handled_by_source_room")) == (
        0 if reason == "sleep" else 2), "a held park keeps the words as context"


@pytest.mark.parametrize("pause", ["older_attempt", "granted", "consumed", "abandoned"])
def test_a_pause_that_is_no_park_of_this_handoff_leaves_the_reply_to_the_work(world, pause):
    """Only this attempt's saved, ungranted park (``pausing``/``paused``) decides; an older attempt's pause, a granted
    or consumed one (the handoff that runs it now) and an abandoned one leave the preparing work its Reply."""
    attempt = 2 if pause == "older_attempt" else 1
    _running(world, _root("work"), attempt=attempt, handoff="g2")
    state = {"older_attempt": "paused", "granted": "resume_granted", "consumed": "resumed", "abandoned": "abandoned"}
    _saved_park(world, "work", "budget", attempt=1, state=state[pause])
    assert _route(world, "work").outcome == "delivered"
    assert [row["work_reply"] for row in _mail(world, "work")] == [True]


@pytest.mark.parametrize("end", ["terminal", "money_park", "sleep_park"])
def test_a_direct_owner_turns_carrier_route_through_its_terminal_or_its_park(world, monkeypatch, end):
    """The real carrier (``worker_chat_lane.CarrierTurn``) owns the turn's ``owner:`` fact, never ``root_intake``. Its
    terminal is stored before ``ended()`` removes the fact (``emit_task_results`` inside the body, ``ended`` in
    ``run_worker_task``'s ``finally``); its park is durable before the body leaves."""
    import queue as stdqueue
    from contextlib import nullcontext

    from ouroboros import conversation_lease
    from ouroboros.budget_pause import BudgetPauseRequested
    from supervisor.worker_chat_lane import CarrierTurn

    monkeypatch.setattr(conversation_lease, "owner_conversation_key", lambda _task: "owner:1")
    turn = _running(world, {**_root("turn"), "_is_direct_chat": True, "metadata": {}}, handoff="g1")
    carrier = CarrierTurn(turn, world.root, stdqueue.Queue())
    try:
        assert carrier.intake == "owner:1"
        with pytest.raises(BudgetPauseRequested) if end != "terminal" else nullcontext(), root_intake(world.root, turn):
            assert _route(world, "turn", row="a").outcome == "delivered"  # reading under its carrier's fact
            if end == "terminal":
                follow_loop_admission(types.SimpleNamespace(drive_root=world.root, task_id="turn", task_attempt=1,
                                                            task_metadata={}), False)  # its final drain
                inside = _route(world, "turn", row="b")
            else:
                pause = _saved_park(world, "turn", "budget" if end == "money_park" else "sleep")
                inside = _route(world, "turn", row="b")  # the carrier's fact still open
                raise BudgetPauseRequested(pause)
        if end == "terminal":
            write_task_result(world.root, "turn", "completed", chat_id=1, title="turn", result="done")  # stored first
        carrier.ended()  # then the carrier lets go of its fact
        assert not conversation_lease._intake_path(world.root, "turn").exists()
        after = _route(world, "turn", row="c")
    finally:
        carrier.ended()
    expected = {"terminal": ("fallback", "not_reading"), "money_park": ("fallback", "paused"),
                "sleep_park": ("delivered", "reading")}[end]
    assert (inside.outcome, inside.reason) == (after.outcome, after.reason) == expected
    assert [row.get("work_reply") for row in _mail(world, "turn")].count(True) == (3 if end == "sleep_park" else 1)


@pytest.mark.parametrize("mail", [False, True], ids=["no_mail", "with_mail"])
def test_settlement_never_reports_gone_an_intake_fact_it_could_not_remove(world, monkeypatch, mail):
    """A refused intake unlink keeps the id ``kept`` in the sweep even when its mailbox did go (the sweep's own
    ``cleanup_task_mailbox`` answer, never the vanished ``.jsonl``), and the next pass removes it and says so."""
    import pathlib

    from ouroboros.conversation_lease import _intake_path
    from ouroboros.owner_mailbox import cleanup_task_mailbox, sweep_settled_owner_mailboxes, write_task_message

    _closed_by_loop(world, _running(world, _root("done"), handoff="g1"))
    if mail:  # words after its last drain: unread, carried into its row by the same settlement
        write_task_message(world.root, "Late words.", "done", source_task_id="parent")
    write_task_result(world.root, "done", "completed", chat_id=1, title="done", result="done")
    real, refusing = pathlib.Path.unlink, [True]

    def unlink(self, *args, **kwargs):
        if refusing[0] and self.name == "done.intake.json":
            raise PermissionError("a reader holds it open")  # as Windows refuses
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    assert _mailbox_path(world.root, "done").exists() is mail
    assert sweep_settled_owner_mailboxes(world.root) == {"removed": [], "kept": 1}, "the sweep first: the fact stayed"
    assert _intake_path(world.root, "done").exists() and not _mailbox_path(world.root, "done").exists()
    assert cleanup_task_mailbox(world.root, "done") is False, "a single cleanup says the same"
    refusing[0] = False  # the reader let go: the next pass removes it and says so
    assert sweep_settled_owner_mailboxes(world.root) == {"removed": ["done"], "kept": 0}
    assert not _intake_path(world.root, "done").exists()


# --- Autonomous work held by Consciousness Off or Panic cannot consume the source room's Reply -----------------------


@pytest.mark.parametrize("kind", ["own", "focus", "started"])
@pytest.mark.parametrize("standing", ["queued", "cold_sleep", "running"])
@pytest.mark.parametrize("gate", ["off", "panic"])
def test_autonomous_work_held_by_consciousness_off_or_panic_falls_back_to_source_room(
        world, monkeypatch, kind, standing, gate):
    from ouroboros import consciousness as clock_module
    from ouroboros.consciousness_wake import wake_task_metadata
    from supervisor import state

    own = wake_task_metadata("act", "heartbeat")
    meta = {
        "own": own,
        "focus": {**own, "wake_cause": "focus", "consciousness_cause": "place:mail"},
        "started": {"initiator": "consciousness", "work_kind": "consciousness", "consciousness_cause": "started"},
    }[kind]
    enabled, panic = {"value": gate != "off"}, {"value": gate == "panic"}
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": enabled["value"]})
    monkeypatch.setattr(clock_module, "panic_blocks_wake", lambda _root: panic["value"])

    task = _root("auto", metadata=meta)
    if standing == "running":
        _running(world, task, handoff="g1")
        assert open_author_intake(world.root, "auto", "work:auto", attempt=1)
    else:
        write_task_result(world.root, "auto", "scheduled", chat_id=1, title="auto", delegation_role="root")
        world.ctx.PENDING.append({**task, **(SLEEP if standing == "cold_sleep" else {})})

    held = _route(world, "auto", row="a")
    assert (held.outcome, held.reason, held.target) == ("fallback", "paused", "auto")
    assert [(r.get("work_reply"), (r.get("host_carried") or {}).get("handled_by_source_room"))
            for r in _mail(world, "auto")] == [(None, True)]

    enabled["value"], panic["value"] = True, False
    live = _route(world, "auto", row="b")
    assert (live.outcome, live.target) == ("delivered", "auto")
    assert [r.get("work_reply") for r in _mail(world, "auto")] == [None, True]


def test_consciousness_off_preserves_owner_work_stops_closed_and_passive_observer_contracts(world, monkeypatch):
    from ouroboros import consciousness as clock_module
    from ouroboros import owner_relay
    from supervisor import state

    enabled = {"value": False}
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": enabled["value"]})
    monkeypatch.setattr(clock_module, "panic_blocks_wake", lambda _root: False)
    auto_meta = {"initiator": "consciousness", "work_kind": "consciousness", "consciousness_cause": "started"}

    # Ordinary owner work (unmarked, owner-marked, and owner-commissioned) still receives Replies while Off.
    for tid, meta in (("plain", {}), ("owner", {"work_kind": "owner"}),
                      ("commissioned", {"initiator": "consciousness", "work_kind": "owner"})):
        write_task_result(world.root, tid, "requested", chat_id=1, title=tid, delegation_role="root")
        world.ctx.PENDING.append(_root(tid, metadata=meta))
        delivery = _route(world, tid, row=tid[0])
        assert (delivery.outcome, delivery.target) == ("delivered", tid)
        assert [r.get("work_reply") for r in _mail(world, tid)] == [True]

    # A closed autonomous work stays not_reading (no passive context written into a dead mailbox).
    closed = _running(world, _root("closed-auto", metadata=auto_meta), handoff="g1")
    _closed_by_loop(world, closed)
    assert _route(world, "closed-auto", row="d").reason == "not_reading"
    assert not _mail(world, "closed-auto")

    # Turning Consciousness Off while waiting for the mail lock rechecks inside admit and falls back.
    enabled["value"] = True
    _running(world, _root("race-auto", metadata=auto_meta), handoff="g1")
    assert open_author_intake(world.root, "race-auto", "work:race-auto", attempt=1)
    real, flipped = owner_relay.write_relay, []

    def flip_off(drive, tid, **kwargs):
        if kwargs.get("work_reply") and not flipped:
            flipped.append(tid)
            enabled["value"] = False
        return real(drive, tid, **kwargs)

    monkeypatch.setattr(owner_relay, "write_relay", flip_off)
    raced = _route(world, "race-auto", row="e")
    assert flipped == ["race-auto"] and (raced.outcome, raced.reason) == ("fallback", "paused")
    assert [(r.get("work_reply"), (r.get("host_carried") or {}).get("handled_by_source_room"))
            for r in _mail(world, "race-auto")] == [(None, True)]

    # Passive room observer: autonomous work held by Off gets no passive copy; ordinary owner work still does.
    monkeypatch.setattr(owner_relay, "write_relay", real)
    _running(world, _root("target", metadata={"work_kind": "owner"}), handoff="g1")
    _running(world, _root("obs-owner", metadata={"work_kind": "owner"}), handoff="g1")
    _running(world, _root("obs-auto", metadata=auto_meta), handoff="g1")
    delivery = _route(world, "target", row="9")
    assert delivery.outcome == "delivered"
    roots = RootSnapshot(pairs=tuple(({"task_id": tid, "status": "running", "room_reader": True}, frozenset({1}))
                                     for tid in ("target", "obs-owner", "obs-auto")))
    work_reply.carry_reply_context(world.ctx, roots, 1, delivery)
    assert [r["host_carried"] for r in _mail(world, "obs-owner")] == [{"went_to": "target", "context_only": True}]
    assert not _mail(world, "obs-auto")
