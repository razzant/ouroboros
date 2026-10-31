"""A settled body adoption is its authorizing task's fact (#2242; DESIGN "Chat authorship and System rows").

Real Git, the real switch helper, boot settlement, the terminal outbox, the supervisor's delivery
handler and the history endpoint: the task result keeps ``body_adoptions``, the row reaches the
task's own room as a ``reviews``-placed card row, a replayed or repeated settlement adds nothing,
and only a handoff that names no task stays an owner-wide runtime notice.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros import body_adoption, body_switch
from ouroboros.task_results import load_task_result
from supervisor import message_bus
from tests.test_body_adoption_consumers import _switch, _task_rows, scene  # noqa: F401

SEND = message_bus.send_with_budget  # the real sender: a boot double never stands in for delivery

SELECTED = {"tier": "selected", "source": "proof", "head": "h"}  # no automatic push follows such an adoption
LOADED = body_adoption._loaded_generation  # what this process attested of the switched tree


def _stored_task(scene, chat_id=1):
    """The authorizing task's stored result and its host facts row, written by the task pipeline's own writers."""
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.post_task_synthesis import _record_task_facts
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

    env = SimpleNamespace(drive_root=scene.data, repo_dir=scene.serving)
    task = {"id": scene.ctx.task_id, "type": "task", "chat_id": chat_id}
    # The queue's admission receipt names the task's chat (supervisor.task_admission), as for any admitted task.
    write_task_result(scene.data, task["id"], STATUS_SCHEDULED, chat_id=chat_id, root_task_id=task["id"],
                      result="Task accepted and scheduled.")
    _store_task_result(env, task, "Fixed the composer focus.", {"rounds": 1, "cost": 0}, {"tool_calls": []})
    _record_task_facts(env, task, {"rounds": 1}, {"tool_calls": []}, scene.data / "logs")


def _quiet_boot(monkeypatch, *, owner_chat=7):
    from supervisor import state

    sent = []
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": owner_chat})
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat, text, **kw: sent.append((chat, text, kw)))
    monkeypatch.setattr("ouroboros.preflight_qualification.adoption_test_evidence", lambda *a, **kw: dict(SELECTED))
    return sent


def _deliver(monkeypatch, root, events):
    """The supervisor's own send handler over the real local bridge; returns the live frames it broadcast."""
    from supervisor import events_chat_delivery as delivery

    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=1000))
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "publish_event", lambda *a, **k: None)
    bridge = message_bus.LocalChatBridge({})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    frames, errors = [], []
    bridge._broadcast_fn = frames.append
    ctx = SimpleNamespace(DRIVE_ROOT=root, task_registry={}, send_with_budget=SEND,
                          append_jsonl=lambda *a: errors.append(a))
    for event in events:
        delivery._handle_send_message(dict(event), ctx)
    assert not errors
    return [frame for frame in frames if frame.get("type") == "chat"]


def _history(root, chat_id=1):
    from ouroboros.gateway.history import make_chat_history_endpoint

    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": str(chat_id)})))
    return json.loads(response.body)["messages"]


def _lose_task_writes(monkeypatch):
    """Every task-result write of the boots that follow fails before it reads the record; returns the restore."""
    from ouroboros import task_store

    real = task_store.patch_document

    def lost(*_a, **_kw):
        raise OSError("task store write lost before mutation")

    monkeypatch.setattr(task_store, "patch_document", lost)
    return lambda: monkeypatch.setattr(task_store, "patch_document", real)


def _wait_words(monkeypatch, words):
    """The next boots settle ``unconfirmed`` in words A (the native host did not confirm) or B (the loaded tree was
    not attested), or ``adopted`` once both are confirmed."""
    from supervisor import update_merge

    monkeypatch.setattr(update_merge, "external_host_restart_smoke", lambda: {"ok": words == "adopted"})
    monkeypatch.setattr(body_adoption, "_loaded_generation", (lambda _handoff: {}) if words == "B" else LOADED)


def _said(text):
    """Which settlement a row's words are: A, B or the adoption."""
    for words, lead in (("A", "Body "), ("B", "The checkout is at "), ("adopted", "Adopted candidate commit ")):
        if str(text).startswith(lead):
            return words
    return text


def _card(scene):
    """The task's adoption rows in history, each as (revision, words)."""
    return [(message["card_row_revision"], _said(message["text"])) for message in _history(scene.data)
            if message.get("system_type") == "body_adoption"]


def test_an_adoption_reaches_its_task_card_history_and_reviews_through_the_real_outbox(scene, monkeypatch):
    from supervisor import terminal_delivery

    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    _stored_task(scene)
    _switch(scene)

    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert settled["outcome"] == "adopted" and settled["task_id"] == scene.ctx.task_id
    assert sent == [] and len(rows) == 1  # no owner-wide bubble: the task's own row only
    row_id = f"body-adoption:{settled['adoption_id']}"
    [fact] = load_task_result(scene.data, scene.ctx.task_id)["body_adoptions"]
    # The first row of an adoption is revision 1, whatever it says; its handoff orders every later one.
    assert {key: fact[key] for key in ("adoption_id", "outcome", "commit_sha", "previous_sha", "branch", "revision")} == {
        "adoption_id": settled["adoption_id"], "outcome": "adopted", "commit_sha": scene.cand,
        "previous_sha": scene.old, "branch": "ouroboros", "revision": 1}
    assert fact["commits"] == [scene.cand] and fact["commits_complete"] is True and fact["note"] == rows[0]["text"]
    assert "commits_omitted" not in fact  # nothing was cut
    event = rows[0]
    assert (event["chat_id"], event["task_id"], event["role"], event["system_type"]) == (
        1, scene.ctx.task_id, "system", "body_adoption")  # the task's own room, not Main's owner chat 7
    assert event["progress_meta"] == {"card_row": "reviews", "card_row_id": row_id, "card_row_revision": 1,
                                      "narration": False}
    assert event["delivery_id"] == f"{row_id}:1"
    assert [owed["delivery_id"] for owed in terminal_delivery.pending_deliveries(scene.data)] == [event["delivery_id"]]

    frames = _deliver(monkeypatch, scene.data, [event, event])  # a replayed copy is the same delivery

    assert terminal_delivery.pending_deliveries(scene.data) == []
    live = [frame for frame in frames if frame.get("card_row_id") == row_id]
    assert len(live) == 1 and live[0]["card_row"] == "reviews" and live[0]["task_id"] == scene.ctx.task_id
    history = _history(scene.data)
    placed = [message for message in history if message.get("card_row_id") == row_id]
    assert len(placed) == 1 and placed[0]["card_row"] == "reviews" and placed[0]["card_row_revision"] == 1
    assert placed[0]["system_type"] == "body_adoption" and placed[0]["text"] == event["text"]
    assert placed[0]["text"].split("\n")[0].startswith(f"Adopted candidate commit {scene.cand[:12]} on ouroboros")
    # Cold history reads the adoption beside the task's Reviews on its own summary row.
    summary = [message for message in history if message.get("system_type") == "task_summary"
               and message.get("task_id") == scene.ctx.task_id]
    assert summary and summary[-1]["body_adoptions"][0]["commit_sha"] == scene.cand
    assert not [message for message in history if "Body adoption" in str(message.get("text") or "")]
    assert body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True) == {}
    assert len(rows) == 1 and sent == []


def test_a_repeated_unconfirmed_boot_keeps_its_revision_and_the_adoption_raises_it(scene, monkeypatch):
    from supervisor import update_merge

    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    smoke = {"ok": False, "reason": "native host artifact unread"}
    monkeypatch.setattr(update_merge, "external_host_restart_smoke", lambda: dict(smoke))
    _stored_task(scene)
    _switch(scene)

    first = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    again = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    smoke["ok"] = True
    adopted = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert (first["outcome"], again["outcome"], adopted["outcome"]) == ("unconfirmed", "unconfirmed", "adopted")
    assert first["adoption_id"] == again["adoption_id"] == adopted["adoption_id"]
    assert [row["delivery_id"].rsplit(":", 1)[1] for row in rows] == ["1", "1", "2"]
    assert [row["progress_meta"]["card_row_revision"] for row in rows] == [1, 1, 2]
    assert len({row["progress_meta"]["card_row_id"] for row in rows}) == 1  # one row, its text revised in place
    assert rows[0]["text"] == rows[1]["text"] and "stays unconfirmed" in rows[0]["text"]
    [fact] = load_task_result(scene.data, scene.ctx.task_id)["body_adoptions"]
    assert (fact["outcome"], fact["revision"], fact["commits"]) == ("adopted", 2, [scene.cand])
    assert sent == []


def test_a_project_bound_task_gets_its_row_in_its_project_room(scene, monkeypatch):
    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    monkeypatch.setattr("ouroboros.projects_registry.project_chat_for_task",
                        lambda _root, task_id: 4242 if task_id == scene.ctx.task_id else 0)
    _stored_task(scene)
    _switch(scene)

    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert sent == [] and [(row["chat_id"], row["task_id"]) for row in rows] == [(4242, scene.ctx.task_id)]


def test_a_handoff_that_names_no_task_stays_an_owner_wide_runtime_notice(scene, monkeypatch):
    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    _stored_task(scene)
    body_adoption.authorize(scene.ctx, scene.cand, reason="adopt it")
    handoff = {key: value for key, value in body_adoption.read(scene.data).items() if key != "task_id"}
    body_switch.write_handoff(str(body_adoption.helper_dir(scene.data)), handoff)  # an earlier handoff schema

    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert settled["outcome"] == "not_applied" and settled["task_id"] == ""
    assert rows == [] and sent == [(7, f"🧬 Body adoption: {settled['note']}",
                                    {"role": "system", "system_type": "restart_notice"})]
    assert "body_adoptions" not in load_task_result(scene.data, scene.ctx.task_id)


@pytest.mark.parametrize("room", [0, "project"])
def test_a_hidden_task_keeps_its_row_hidden_and_a_project_task_keeps_its_room(scene, monkeypatch, room):
    """Owner chat 7 and Main (1) exist; the row goes only where the task's own record or Project puts it."""
    from supervisor import terminal_delivery

    sent, rows = _quiet_boot(monkeypatch, owner_chat=7), _task_rows(monkeypatch)
    if room == "project":
        monkeypatch.setattr("ouroboros.projects_registry.project_chat_for_task",
                            lambda _root, task_id: 4242 if task_id == scene.ctx.task_id else 0)
    _stored_task(scene, chat_id=0 if room == 0 else 1)  # a headless/API task is admitted into the hidden partition
    _switch(scene)

    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    expected = 0 if room == 0 else 4242
    assert settled["outcome"] == "adopted" and sent == []
    assert [(row["chat_id"], row["task_id"]) for row in rows] == [(expected, scene.ctx.task_id)]
    frames = _deliver(monkeypatch, scene.data, rows)
    row_id = f"body-adoption:{settled['adoption_id']}"
    assert [frame["chat_id"] for frame in frames if frame.get("card_row_id") == row_id] == [expected]
    assert terminal_delivery.pending_deliveries(scene.data) == []
    if room == 0:  # the stored row stays in the hidden partition, which Main never reads (row_rooms)
        assert not [message for message in _history(scene.data) if message.get("system_type") == "body_adoption"]
    assert load_task_result(scene.data, scene.ctx.task_id)["body_adoptions"][0]["commit_sha"] == scene.cand


@pytest.mark.parametrize("record", ["absent", "names_no_room", "unreadable"])
def test_a_task_record_that_names_no_room_gets_no_row_anywhere(scene, monkeypatch, record):
    from ouroboros import task_results
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

    sent, rows = _quiet_boot(monkeypatch, owner_chat=7), _task_rows(monkeypatch)
    if record == "names_no_room":  # a record written without the room its admission would name
        write_task_result(scene.data, scene.ctx.task_id, STATUS_SCHEDULED, root_task_id=scene.ctx.task_id,
                          result="Task accepted and scheduled.")
    if record == "unreadable":
        _stored_task(scene)
    body_adoption.authorize(scene.ctx, scene.cand, reason="adopt it")
    real = task_results.load_task_result

    def unreadable(*_a, **_kw):
        raise OSError("task store unreadable")

    if record == "unreadable":
        monkeypatch.setattr(task_results, "load_task_result", unreadable)
    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    monkeypatch.setattr(task_results, "load_task_result", real)

    # With no room it can name, nothing is sent anywhere: never Main or the owner chat in its place.
    assert settled["outcome"] == "not_applied" and sent == [] and rows == []
    assert _deliver(monkeypatch, scene.data, rows) == []
    assert not [message for message in _history(scene.data) if message.get("system_type") == "body_adoption"]


def test_a_lost_task_fact_still_reaches_the_card_and_leaves_the_reviews_without_it(scene, monkeypatch):
    """The task-fact write fails, the outbox publication succeeds: the row says the outcome; the task result
    and its summary row (what the Reviews read) never claim an adoption nothing recorded."""
    from ouroboros import task_store

    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    _stored_task(scene)
    _switch(scene)
    real = task_store.patch_document

    def lost(*_a, **_kw):
        raise OSError("task store write lost")

    monkeypatch.setattr(task_store, "patch_document", lost)
    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    monkeypatch.setattr(task_store, "patch_document", real)  # the write fault was this boot's only

    assert settled["outcome"] == "adopted" and sent == []
    [event] = rows
    assert (event["chat_id"], event["progress_meta"]["card_row_revision"]) == (1, 1)
    frames = _deliver(monkeypatch, scene.data, [event])
    assert [frame["card_row_id"] for frame in frames if frame.get("card_row_id")] == [f"body-adoption:{settled['adoption_id']}"]
    assert "body_adoptions" not in load_task_result(scene.data, scene.ctx.task_id)
    history = _history(scene.data)
    assert [message["text"] for message in history if message.get("system_type") == "body_adoption"] == [event["text"]]
    summary = [message for message in history if message.get("system_type") == "task_summary"
               and message.get("task_id") == scene.ctx.task_id]
    assert summary and "body_adoptions" not in summary[-1]  # the Reviews have no adoption to name


def test_a_lost_wait_does_not_hide_the_adoption_that_follows_it(scene, monkeypatch):
    """Boot 1 cannot record its `unconfirmed` row but sends it (revision 1); boot 2 records `adopted`. The record
    never saw revision 1; the handoff kept it, so the adoption's first recorded row is 2, not a spent delivery id."""
    from ouroboros import task_store
    from supervisor import update_merge

    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    smoke = {"ok": False, "reason": "native host artifact unread"}
    monkeypatch.setattr(update_merge, "external_host_restart_smoke", lambda: dict(smoke))
    _stored_task(scene)
    _switch(scene)
    real = task_store.patch_document

    def lost(*_a, **_kw):
        raise OSError("task store write lost")

    monkeypatch.setattr(task_store, "patch_document", lost)
    first = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    monkeypatch.setattr(task_store, "patch_document", real)
    _deliver(monkeypatch, scene.data, list(rows))
    smoke["ok"] = True
    adopted = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert (first["outcome"], adopted["outcome"]) == ("unconfirmed", "adopted") and sent == []
    assert [row["progress_meta"]["card_row_revision"] for row in rows] == [1, 2]
    frames = _deliver(monkeypatch, scene.data, rows[1:])
    assert [frame.get("card_row_revision") for frame in frames if frame.get("card_row_id")] == [2]
    placed = [message for message in _history(scene.data) if message.get("system_type") == "body_adoption"]
    assert placed[-1]["card_row_revision"] == 2 and placed[-1]["text"].startswith("Adopted candidate commit")
    assert load_task_result(scene.data, scene.ctx.task_id)["body_adoptions"][0]["revision"] == 2


def test_a_lost_final_write_follows_each_already_delivered_wait(scene, monkeypatch):
    """Different unconfirmed notes can already have spent revision 2 before the final write fails."""
    from ouroboros import task_store
    from supervisor import update_merge

    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    smoke = {"ok": False}
    monkeypatch.setattr(update_merge, "external_host_restart_smoke", lambda: dict(smoke))
    _stored_task(scene)
    _switch(scene)
    loaded = body_adoption._loaded_generation
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    monkeypatch.setattr(body_adoption, "_loaded_generation", lambda _handoff: {})
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert [row["progress_meta"]["card_row_revision"] for row in rows] == [1, 2]
    _deliver(monkeypatch, scene.data, list(rows))

    def lost(*_a, **_kw):
        raise OSError("task store write lost before mutation")

    monkeypatch.setattr(body_adoption, "_loaded_generation", loaded)
    monkeypatch.setattr(task_store, "patch_document", lost)
    smoke["ok"] = True
    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert settled["outcome"] == "adopted" and sent == []
    final = rows[-1]
    assert final["progress_meta"]["card_row_revision"] == 3
    frames = _deliver(monkeypatch, scene.data, [final])
    assert [frame["content"] for frame in frames if frame.get("card_row_id")] == [final["text"]]
    placed = [message for message in _history(scene.data) if message.get("system_type") == "body_adoption"]
    assert placed[-1]["card_row_revision"] == 3
    assert placed[-1]["text"].startswith("Adopted candidate commit")
    # A sent fact is not a successful task-result write; its old record stays honest.
    assert load_task_result(scene.data, scene.ctx.task_id)["body_adoptions"][0]["outcome"] == "unconfirmed"


def test_a_lost_intermediate_write_cannot_spend_the_settled_end(scene, monkeypatch):
    """Three boots: wait A is recorded and sent; wait B, in other words, is sent while its task-result write fails
    before any mutation; the adoption is then recorded. Its row must follow both waits, never reuse B's identity."""
    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    _stored_task(scene)
    _switch(scene)
    _wait_words(monkeypatch, "A")
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    _deliver(monkeypatch, scene.data, list(rows))
    _wait_words(monkeypatch, "B")
    restore = _lose_task_writes(monkeypatch)
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    restore()
    _deliver(monkeypatch, scene.data, rows[1:])
    _wait_words(monkeypatch, "adopted")

    adopted = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert adopted["outcome"] == "adopted" and sent == []
    frames = _deliver(monkeypatch, scene.data, rows[2:])
    [fact] = load_task_result(scene.data, scene.ctx.task_id)["body_adoptions"]
    assert {"queued": [(row["progress_meta"]["card_row_revision"], row["delivery_id"].rsplit(":", 1)[1],
                        _said(row["text"])) for row in rows],
            "delivered": [_said(frame["content"]) for frame in frames if frame.get("card_row_id")],
            "card": _card(scene), "fact": (fact["outcome"], fact["revision"])} == {
        "queued": [(1, "1", "A"), (2, "2", "B"), (3, "3", "adopted")], "delivered": ["adopted"],
        "card": [(1, "A"), (2, "B"), (3, "adopted")], "fact": ("adopted", 3)}


def test_waits_after_a_lost_write_stay_distinct_and_a_repeated_one_adds_nothing(scene, monkeypatch):
    """A is recorded and sent; B is sent while task-result writes fail, then repeated under the same fault; A comes
    back with writes working, then the adoption. The card says each boot's words in order, B only once."""
    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    _stored_task(scene)
    _switch(scene)
    delivered = []

    def boot(words):
        _wait_words(monkeypatch, words)
        before = len(rows)
        body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
        delivered.append([_said(frame["content"]) for frame in _deliver(monkeypatch, scene.data, rows[before:])
                          if frame.get("card_row_id")])

    boot("A")
    restore = _lose_task_writes(monkeypatch)
    boot("B")
    boot("B")  # the same words again: the same row, nothing new on the card
    restore()
    boot("A")
    boot("adopted")

    [fact] = load_task_result(scene.data, scene.ctx.task_id)["body_adoptions"]
    assert sent == []
    assert {"delivered": delivered, "queued": [row["progress_meta"]["card_row_revision"] for row in rows],
            "card": _card(scene), "fact": (fact["outcome"], fact["revision"])} == {
        "delivered": [["A"], ["B"], [], ["A"], ["adopted"]], "queued": [1, 2, 3, 4],
        "card": [(1, "A"), (2, "B"), (3, "A"), (4, "adopted")], "fact": ("adopted", 4)}


def test_a_wait_its_handoff_cannot_keep_is_not_sent_and_the_next_boot_follows(scene, monkeypatch):
    """The handoff, not the task result, keeps the card row's revision: a boot that cannot keep its wait there sends
    nothing and leaves the handoff for the next boot, whose rows follow every row already sent."""
    sent, rows = _quiet_boot(monkeypatch), _task_rows(monkeypatch)
    _stored_task(scene)
    _switch(scene)
    _wait_words(monkeypatch, "A")
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    _deliver(monkeypatch, scene.data, list(rows))
    real = body_switch.write_handoff

    def unwritable(*_a, **_kw):
        raise OSError("handoff unwritable")

    _wait_words(monkeypatch, "B")
    monkeypatch.setattr(body_switch, "write_handoff", unwritable)
    unkept = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    monkeypatch.setattr(body_switch, "write_handoff", real)
    assert (unkept, len(rows), body_adoption.read(scene.data)["phase"]) == ({}, 1, "switched")
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    _wait_words(monkeypatch, "adopted")
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    _deliver(monkeypatch, scene.data, rows[1:])

    assert sent == [] and [row["progress_meta"]["card_row_revision"] for row in rows] == [1, 2, 3]
    assert _card(scene) == [(1, "A"), (2, "B"), (3, "adopted")]


def test_adopted_ranges_count_only_the_commits_the_bounded_list_omits(scene):
    """Real Git intervals at both sides of the 64-entry view; canonical range endpoints stay resolvable."""
    tree = body_adoption._git(scene.serving, "rev-parse", "HEAD^{tree}").stdout.strip()
    parent, commits = scene.old, []
    for number in range(1, 66):
        parent = body_adoption._git(scene.serving, "commit-tree", tree, "-p", parent,
                                    "-m", f"range test {number}").stdout.strip()
        commits.append(parent)
        if number < 63:
            continue
        fact = body_adoption._adopted_commits(scene.serving, {
            "outcome": "adopted", "previous": scene.old, "candidate": parent})
        assert fact["commits"] == list(reversed(commits))[:64]
        assert fact["commits_complete"] is (number <= 64)
        assert fact.get("commits_omitted", 0) == max(0, number - 64)
