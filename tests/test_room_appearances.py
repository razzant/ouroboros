"""A Project keeps its whole conversation: work speech written elsewhere and the owner's direct Reply.

One row, one address. ``main_notice`` and ``relay_reply`` stay where they were written (Main,
a relay's source room) and are members of their work's Project too; the owner's direct Reply
to a Project message is a member of that Project. Readers find such a row through the dialogue
store's derived appearance index (``room_appearances``), never by walking another room, live,
on reload, around an exact row and in memory. Ordinary conversion still moves a work's own
speech; host refusals and Presence private contact stay where they were issued.
"""
from __future__ import annotations

import asyncio
import json
import queue
import types

import pytest

from ouroboros import dialogue_store as ds
from ouroboros.chat_chain import format_address, row_address
from ouroboros.tools.control_runtime import _send_user_message
from tests.test_dialogue_addressed_readers import metered
from tests.test_main_notice import host as host_fixture

host = host_fixture  # the real supervisor handler, bus and store


def _project(root, project_id):
    from ouroboros.projects_registry import create_project

    return int(create_project(root, project_id, name=project_id.title())["chat_id"])


def _born_in(root, task_id, chat_id):
    """A root admitted in a Project room: no binding, its own admitted chat is its room."""
    from ouroboros.task_results import write_task_result

    write_task_result(root, task_id, "running", chat_id=chat_id, title=task_id)


def _bound(root, task_id, project_id, chat_id):
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result

    write_task_result(root, task_id, "running", chat_id=1, title=task_id)
    bind_task_to_project(root, task_id, project_id, chat_id, origin={"absent": "system"})


def _ctx(root, task_id, chat_id, q=None):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(current_chat_id=chat_id, task_id=task_id, task_metadata={}, event_queue=q,
                                 pending_events=[], drive_root=root, drive_logs=lambda: root / "logs")


def _notice(root, host, task_id, chat_id, text):
    q = queue.Queue()
    assert _send_user_message(_ctx(root, task_id, chat_id, q), text, destination="main").startswith("OK:")
    host.handle(q.get_nowait())
    return next(frame for frame in reversed(host.frames) if frame.get("system_type") == "main_notice")


def _read(root, chat_id, **params):
    from ouroboros.gateway.history import make_chat_history_endpoint

    query = {"chat_id": str(chat_id), **{key: str(value) for key, value in params.items()}}
    response = asyncio.run(make_chat_history_endpoint(root)(types.SimpleNamespace(query_params=query)))
    return response.status_code, json.loads(response.body)


def _texts(root, chat_id):
    return [row.get("text") for row in _read(root, chat_id)[1]["messages"] if row.get("role") in {"assistant", "user"}]


def _room_rows(root, chat_id):
    from ouroboros.chat_chain import room_rows

    return [entry.row.get("text") for entry in room_rows(root, chat_id)]


def _owner(bridge, root, text, reply_to, *, chat_id=1, message_id):
    """The web owner door: the canonical row, its echo and the dispatch it hands over."""
    dispatched = []
    bridge.handle_web_message(text, client_message_id=message_id, chat_id=chat_id,
                              task_metadata={"reply_to": reply_to}, dispatch=lambda *a, **k: dispatched.append(k))
    row = ds.accepted_row(root, chat_id, message_id).row
    return row, dispatched


class TestWorkSpeechWrittenElsewhere:
    def test_a_born_in_project_roots_notice_is_one_message_in_main_and_its_project_everywhere(self, tmp_path, host):
        nova, other = _project(tmp_path, "nova"), _project(tmp_path, "other")
        _born_in(tmp_path, "nova-root", nova)
        frame = _notice(tmp_path, host, "nova-root", nova, "All scheduled series have finished.")
        address = frame["row_address"]
        assert frame["chat_id"] == 1 and frame["also_rooms"] == [nova]
        # Reload: one row, the same address, in both rooms; never in an unrelated Project.
        assert "All scheduled series have finished." in _texts(tmp_path, 1)
        assert "All scheduled series have finished." in _texts(tmp_path, nova)
        assert "All scheduled series have finished." not in _texts(tmp_path, other)
        for chat in (1, nova):
            status, page = _read(tmp_path, chat, around=address)
            assert status == 200 and page["anchor"]["state"] == "found"
            assert address in {row.get("row_address") for row in page["messages"]}
        assert _read(tmp_path, other, around=address)[1]["anchor"]["state"] == "moved"
        # Memory readers: the room's indexed rows and the whole-mind inventory use the same rule.
        assert "All scheduled series have finished." in _room_rows(tmp_path, nova)
        assert "All scheduled series have finished." not in _room_rows(tmp_path, other)
        from ouroboros import memory_inventory as mi
        row = ds.rows_by_identity(tmp_path, 1, frame["ts"])[0].row
        assert mi.rooms_of_row(mi.row_meta(row), mi.membership_facts(tmp_path)) == {"1", str(nova)}

    def test_a_relay_answer_returns_to_its_source_and_stays_in_its_works_project(self, tmp_path, host):
        from ouroboros.owner_relay import relay_envelope

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        q = queue.Queue()
        ctx = _ctx(tmp_path, "nova-root", nova, q)
        ctx._open_relays = {"r1": relay_envelope("r1", [], chat_id=1)}
        assert _send_user_message(ctx, "Here is the result you asked for.", reply_to_relay="r1").startswith("OK:")
        host.handle(q.get_nowait())
        frame = next(f for f in host.frames if f.get("system_type") == "relay_reply")
        assert frame["chat_id"] == 1 and frame["also_rooms"] == [nova]
        assert "Here is the result you asked for." in _texts(tmp_path, 1)
        assert "Here is the result you asked for." in _texts(tmp_path, nova)

    def test_a_relay_answer_replayed_after_a_restart_is_still_its_works(self, tmp_path, host, monkeypatch):
        """F4's debt names its producer: the relay answer owed before the queue sees it carries its work's
        identity in the durable row itself. A restart that loses the live copy replays, through the outbox, the
        real bus and the supervisor handler, an answer still of its work: it returns to its source room, stands
        in the work's Project and retires the debt."""
        from ouroboros.owner_relay import relay_envelope
        from supervisor import terminal_delivery
        from supervisor.event_observation import ObservedEventQueue

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        lost = queue.Queue()  # the live copy a restart drops before the supervisor consumed it
        ctx = _ctx(tmp_path, "nova-root", nova, lost)
        ctx.task_metadata = {"root_task_id": "nova-root", "parent_task_id": ""}
        ctx._open_relays = {"r1": relay_envelope("r1", [], chat_id=1)}
        assert _send_user_message(ctx, "Here is the result you asked for.", reply_to_relay="r1").startswith("OK:")
        live = lost.get_nowait()
        lineage = ("task_id", "parent_task_id", "root_task_id")
        (owed,) = [row for row in terminal_delivery.pending_deliveries(tmp_path) if row.get("delivery_id") == live["delivery_id"]]
        assert owed["task_id"] == "nova-root" and {key: owed.get(key) for key in lineage} == {key: live[key] for key in lineage}
        monkeypatch.setattr(terminal_delivery, "_REPLAY_MIN_AGE_SEC", 0)
        bus = ObservedEventQueue()
        assert terminal_delivery.replay_pending_deliveries(tmp_path, event_queue=bus) == [live["delivery_id"]]
        host.handle(bus.get_nowait())
        frame = next(f for f in host.frames if f.get("system_type") == "relay_reply")
        assert frame["chat_id"] == 1 and frame.get("also_rooms") == [nova]
        assert "Here is the result you asked for." in _texts(tmp_path, nova)
        assert not [row for row in terminal_delivery.pending_deliveries(tmp_path)
                    if row.get("delivery_id") == live["delivery_id"]]

    def test_ordinary_speech_still_moves_and_host_rows_stay_where_they_were_issued(self, tmp_path, host):
        nova = _project(tmp_path, "nova")
        _bound(tmp_path, "moved-root", "nova", nova)
        q = queue.Queue()
        assert _send_user_message(_ctx(tmp_path, "moved-root", 1, q), "A reply while I work.").startswith("OK:")
        host.handle(q.get_nowait())
        ds.append(tmp_path, "chat", {"ts": "2026-10-28T05:30:00Z", "direction": "system", "chat_id": 1,
                                     "type": "task_not_started", "task_id": "moved-root", "text": "Not started"})
        ds.append(tmp_path, "chat", {"ts": "2026-10-28T05:31:00Z", "direction": "system", "chat_id": 1,
                                     "type": "project_started", "task_id": "moved-root", "text": "Moved to Nova"})
        main, project = _texts(tmp_path, 1), _texts(tmp_path, nova)
        assert "A reply while I work." in project and "A reply while I work." not in main
        for text in ("Not started", "Moved to Nova"):
            assert text not in project
        assert {row.get("text") for row in _read(tmp_path, 1)[1]["messages"]} >= {"Not started", "Moved to Nova"}

    def test_presence_private_contact_never_spills_into_a_project(self, tmp_path, host):
        nova = _project(tmp_path, "nova")
        _bound(tmp_path, "presence-root", "nova", nova)
        contact = {"task_id": "presence-root", "binding_id": "place-1", "source_event_id": "evt-1"}
        q = queue.Queue()
        ctx = _ctx(tmp_path, "presence-root", nova, q)
        from ouroboros.tools.owner_delivery import deliver_owner_event
        deliver_owner_event(ctx, {"type": "send_message", "chat_id": 1, "text": "A private question.",
                                  "format": "markdown", "is_progress": False, "system_type": "main_notice",
                                  "progress_meta": {"presence_contact": contact}})
        host.handle(q.get_nowait())
        frame = next(f for f in host.frames if f.get("system_type") == "main_notice")
        assert "also_rooms" not in frame and set(frame.get("room_revisions") or {"1": 0}) == {"1"}
        assert "A private question." in _texts(tmp_path, 1)
        assert "A private question." not in _texts(tmp_path, nova)
        assert "A private question." not in _room_rows(tmp_path, nova)


class TestOwnerReplyLink:
    @pytest.fixture
    def room(self, tmp_path, host):
        from supervisor import message_bus

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        notice = _notice(tmp_path, host, "nova-root", nova, "Titan series are done; no new requests.")
        return types.SimpleNamespace(root=tmp_path, nova=nova, notice=notice, host=host,
                                     bridge=message_bus.get_bridge())

    def test_a_direct_reply_in_main_to_a_project_message_is_in_that_project_and_dispatches_only_in_main(self, room):
        row, dispatched = _owner(room.bridge, room.root, "Why is this not in Nova?", room.notice["row_address"],
                                 message_id="owner-1")
        echo = next(f for f in room.host.frames if f.get("client_message_id") == "owner-1" and f.get("role") == "user")
        assert echo["chat_id"] == 1 and echo["also_rooms"] == [room.nova]
        assert row["reply_to"]["target_home"] == "work" and row["reply_to"]["task_id"] == "nova-root"
        # Shown in both rooms at its one Main address; handed only to Main's ordinary routing.
        assert "Why is this not in Nova?" in _texts(room.root, 1)
        assert "Why is this not in Nova?" in _texts(room.root, room.nova)
        address = format_address(row_address(row))
        assert _read(room.root, room.nova, around=address)[1]["anchor"]["state"] == "found"
        assert [call["chat_id"] for call in dispatched] == [1]
        from ouroboros.owner_mailbox import _mailbox_path
        assert not _mailbox_path(room.root, "nova-root").exists()  # appearance is not a tell, wake or Resume

    def test_unrelated_main_speech_and_the_reply_chain_stay_in_main(self, room):
        # A Main direct turn's answer: not a Project message.
        ds.append(room.root, "chat", {"ts": "2026-10-28T06:00:00Z", "direction": "out", "chat_id": 1,
                                      "task_id": "main-turn", "text": "A Main answer."})
        main_answer = format_address(row_address(ds.rows_by_identity(room.root, 1, "2026-10-28T06:00:00Z")[0].row))
        _owner(room.bridge, room.root, "About the Main answer.", main_answer, message_id="owner-2")
        first, _ = _owner(room.bridge, room.root, "About the notice.", room.notice["row_address"], message_id="owner-3")
        # A Reply to that Reply: one hop only, never the chain.
        _owner(room.bridge, room.root, "And about my own reply.", format_address(row_address(first)), message_id="owner-4")
        project = _texts(room.root, room.nova)
        assert "About the notice." in project
        assert "About the Main answer." not in project and "And about my own reply." not in project
        assert {"About the Main answer.", "And about my own reply."} <= set(_texts(room.root, 1))

    def test_a_reply_to_an_owner_row_written_in_the_project_is_in_that_project(self, room):
        ds.append(room.root, "chat", {"ts": "2026-10-28T06:05:00Z", "direction": "in", "chat_id": room.nova,
                                      "client_message_id": "p-own", "text": "Project note from me."})
        target = format_address(row_address(ds.rows_by_identity(room.root, room.nova, "2026-10-28T06:05:00Z")[0].row))
        row, _ = _owner(room.bridge, room.root, "Following up from Main.", target, message_id="owner-5")
        assert row["reply_to"]["target_home"] == "room"
        assert "Following up from Main." in _texts(room.root, room.nova)

    def test_a_reply_to_main_work_moved_later_follows_the_binding(self, room):
        ds.append(room.root, "chat", {"ts": "2026-10-28T06:10:00Z", "direction": "out", "chat_id": 1,
                                      "task_id": "later", "text": "Draft plan in Main."})
        from ouroboros.task_results import write_task_result
        write_task_result(room.root, "later", "running", chat_id=1, title="later")
        target = format_address(row_address(ds.rows_by_identity(room.root, 1, "2026-10-28T06:10:00Z")[0].row))
        _owner(room.bridge, room.root, "Please widen the plan.", target, message_id="owner-6")
        assert "Please widen the plan." not in _texts(room.root, room.nova)
        from ouroboros.projects_registry import bind_task_to_project
        bind_task_to_project(room.root, "later", "nova", room.nova, origin={"absent": "system"})
        assert {"Draft plan in Main.", "Please widen the plan."} <= set(_texts(room.root, room.nova))


class TestDerivedIndex:
    def _older_body_notice(self, root, nova):
        """A notice an older body wrote: no index keys, no revision stamp, the mark left where it was."""
        row = {"ts": "2026-10-28T05:29:14.599616+00:00", "direction": "out", "chat_id": 1, "task_id": "nova-root",
               "type": "main_notice", "text": "Old Nova notice.", "format": "markdown"}
        receipt = ds.append(root, "chat", row, feed=True, appearances=None)
        return receipt

    def test_an_older_bodys_rows_are_an_honest_gap_until_the_job_derives_them_unchanged(self, tmp_path):
        from ouroboros import room_appearances

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        receipt = self._older_body_notice(tmp_path, nova)
        before = ds.row_at(tmp_path, "chat", receipt.seq)
        revisions = {room["chat_id"]: room["revision"] for room in ds.rooms(tmp_path)}
        status, page = _read(tmp_path, nova)
        assert "chat_appearances_deriving" in page["window"]["truncated_by"]
        assert "Old Nova notice." not in [row.get("text") for row in page["messages"]]
        assert room_appearances.owed(tmp_path)
        assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        assert room_appearances.run(tmp_path, lambda: False) == {"state": "done", "items": 0}  # idempotent, nothing left
        status, page = _read(tmp_path, nova)
        assert "Old Nova notice." in [row.get("text") for row in page["messages"]]
        assert "chat_appearances_deriving" not in page["window"]["truncated_by"]
        after = ds.row_at(tmp_path, "chat", receipt.seq)
        assert after.row == before.row and after.length == before.length  # body, address and coordinate unchanged
        assert format_address(row_address(after.row)) == format_address(row_address(receipt.row))
        assert {room["chat_id"]: room["revision"] for room in ds.rooms(tmp_path)} == revisions  # no re-feed

    def test_a_new_rows_append_keeps_the_mark_current(self, tmp_path, host):
        from ouroboros import room_appearances

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        _notice(tmp_path, host, "nova-root", nova, "A fresh notice.")
        assert not room_appearances.owed(tmp_path)
        assert "chat_appearances_deriving" not in _read(tmp_path, nova)[1]["window"]["truncated_by"]

    def test_a_project_page_reads_only_its_own_appearances(self, tmp_path, monkeypatch):
        def steps(root, foreign):
            nova, other = _project(root, "nova"), _project(root, "other")
            _born_in(root, "nova-root", nova)
            _born_in(root, "other-root", other)
            for index in range(foreign):  # another Project's notices, written to Main
                ds.append(root, "chat", {"ts": f"2026-10-27T{index // 3600:02d}:{index // 60 % 60:02d}:{index % 60:02d}Z",
                                         "direction": "out", "chat_id": 1, "task_id": "other-root",
                                         "type": "main_notice", "text": f"other {index}"},
                          feed=True, appearances=[f"room:{other}"])
            ds.append(root, "chat", {"ts": "2026-10-28T05:00:00Z", "direction": "out", "chat_id": 1,
                                     "task_id": "nova-root", "type": "main_notice", "text": "nova notice"},
                      feed=True, appearances=[f"room:{nova}"])
            with metered(monkeypatch) as measured:
                status, page = _read(root, nova)
            assert [row.get("text") for row in page["messages"] if row.get("role") == "assistant"] == ["nova notice"]
            return measured["steps"]

        small, large = steps(tmp_path / "small", 20), steps(tmp_path / "large", 2000)
        # Reader work does not grow with another room's appearances: one seek of this room's keys.
        assert large < small * 1.25, (small, large)


class TestSavedReplyTargets:
    """Replies saved before ``reply_to.target_home`` are decided by their actual target row, read at its
    exact address: never by the reference's role, never by rewriting the owner's row."""

    def _older_reply(self, root, target, ts, client_id):
        from ouroboros.chat_references import reference_from_row

        legacy = {key: value for key, value in reference_from_row(target).items() if key != "target_home"}
        row = {"ts": ts, "direction": "in", "chat_id": 1, "client_message_id": client_id,
               "text": f"Owner words {client_id}", "reply_to": legacy}
        return ds.append(root, "chat", row, appearances=None)  # an older body: no keys, the mark left behind

    def _target(self, root, ts, **fields):
        row = {"ts": ts, "chat_id": 1, "task_id": "nova-root", "text": f"target {ts}", **fields}
        ds.append(root, "chat", row, feed=True, appearances=None)
        return row

    def test_a_saved_reply_to_a_project_system_row_joins_it_and_refusals_or_private_rows_do_not(self, tmp_path):
        from ouroboros import room_appearances

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        summary = self._target(tmp_path, "2026-10-28T05:00:00Z", direction="system", type="project_completion_summary")
        refusal = self._target(tmp_path, "2026-10-28T05:01:00Z", direction="system", type="task_not_started")
        private = self._target(tmp_path, "2026-10-28T05:02:00Z", direction="out", type="main_notice",
                               presence_contact={"task_id": "nova-root", "binding_id": "b", "source_event_id": "e"})
        replies = {name: self._older_reply(tmp_path, target, f"2026-10-28T06:0{index}:00Z", name)
                   for index, (name, target) in enumerate((("summary", summary), ("refusal", refusal),
                                                           ("private", private)))}
        absent = {**summary, "text": "never stored"}
        replies["absent"] = self._older_reply(tmp_path, absent, "2026-10-28T06:09:00Z", "absent")
        assert all(receipt.row["reply_to"].get("role") == "system" or name in {"private", "absent"}
                   for name, receipt in replies.items())  # the old reading would have dropped the System target
        bodies = {name: ds.row_at(tmp_path, "chat", receipt.seq).row for name, receipt in replies.items()}
        assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        project = _texts(tmp_path, nova)
        assert "Owner words summary" in project
        assert not {"Owner words refusal", "Owner words private", "Owner words absent"} & set(project)
        assert {f"Owner words {name}" for name in replies} <= set(_texts(tmp_path, 1))  # all stay in Main
        assert {name: ds.row_at(tmp_path, "chat", receipt.seq).row for name, receipt in replies.items()} == bodies

    def test_an_unreadable_saved_target_is_the_readers_typed_gap_never_a_guess(self, tmp_path, monkeypatch):
        from ouroboros import chat_chain, room_appearances, room_membership
        from ouroboros.store import StoreError

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        summary = self._target(tmp_path, "2026-10-28T05:00:00Z", direction="system", type="project_completion_summary")
        self._older_reply(tmp_path, summary, "2026-10-28T06:00:00Z", "summary")
        assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        room_membership._SAVED_TARGETS.clear()

        def unreadable(*_args, **_kwargs):
            raise StoreError("dialogue history source unavailable")
        monkeypatch.setattr(chat_chain, "resolve_row", unreadable)
        status, page = _read(tmp_path, nova)
        assert status == 503 and page["reason_code"] == "history_source_unavailable"

    def test_a_saved_target_whose_bytes_are_unreadable_is_a_gap_where_its_reply_could_belong(self, tmp_path):
        """A recorded unreadable interval is not proof of absence: the Reply stays Main's and the Project
        its saved reference names discloses ``reply_target_unreadable`` over the pages that reach it."""
        from ouroboros import room_appearances
        from ouroboros.chat_references import reference_from_row
        from ouroboros.room_membership import TARGET_UNKNOWN, _saved_target
        from ouroboros.projects_registry import RoomResolver
        from tests._dialogue import import_all, seed

        nova, other = _project(tmp_path, "nova"), _project(tmp_path, "other")
        _born_in(tmp_path, "nova-root", nova)
        target = {"ts": "2026-10-01T05:00:00Z", "direction": "system", "chat_id": 1, "task_id": "nova-root",
                  "type": "project_completion_summary", "text": "Nova finished its series."}
        legacy = {key: value for key, value in reference_from_row(target).items() if key != "target_home"}
        reply = {"ts": "2026-10-01T06:00:00Z", "direction": "in", "chat_id": 1, "client_message_id": "torn",
                 "text": "Owner words torn", "reply_to": legacy}
        seed(tmp_path, archives={"chat_20261001T000000.jsonl": [json.dumps(target)[:48] + "\n", reply]})
        import_all(tmp_path)
        assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        stored = ds.rows_by_identity(tmp_path, 1, reply["ts"])[0]
        assert _saved_target(RoomResolver(tmp_path), legacy)[0] == TARGET_UNKNOWN
        page = _read(tmp_path, nova)[1]
        assert "Owner words torn" not in _texts(tmp_path, nova) and "Owner words torn" in _texts(tmp_path, 1)
        assert "chat_reply_target_unreadable" in page["window"]["truncated_by"]
        assert "chat_reply_target_unreadable" not in _read(tmp_path, other)[1]["window"]["truncated_by"]
        keys = ["room:%d" % nova, "task:nova-root"]  # only the pages that reach the Reply say so
        assert room_appearances.unknown_targets(tmp_path, keys, stored.seq, stored.seq + 1)
        assert not room_appearances.unknown_targets(tmp_path, keys, stored.seq + 1, ds.stream_end(tmp_path, "chat") + 1)
        assert not room_appearances.unknown_targets(tmp_path, ["room:%d" % other], 0, ds.stream_end(tmp_path, "chat"))
        assert ds.row_at(tmp_path, "chat", stored.seq).row == stored.row  # the owner's row is unchanged


def test_an_older_bodys_archive_import_below_the_mark_is_derived_by_its_segment(tmp_path, monkeypatch):
    """The supported downgrade window: this body cut over (its live rows derived, the mark past them) with
    an archive segment still pending; an older body imported that segment, deriving nothing. The segment
    stays awaiting, a Project page discloses the gap, and the job derives exactly its rows, changing none."""
    from ouroboros import dialogue_import, room_appearances
    from tests._dialogue import seed

    nova = _project(tmp_path, "nova")
    _born_in(tmp_path, "nova-root", nova)
    archived = {"ts": "2026-10-01T00:00:00Z", "direction": "out", "chat_id": 1, "task_id": "nova-root",
                "type": "main_notice", "text": "An archived Nova notice."}
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [archived]},
         live={"chat": [{"ts": "2026-10-02T00:00:00Z", "direction": "in", "chat_id": 1, "text": "live"}]})
    dialogue_import.cutover(tmp_path)  # this body: the live tail imported and derived, the archive planned
    assert room_appearances.mark(tmp_path) == ds.stream_end(tmp_path, "chat") and not room_appearances.owed(tmp_path)
    with monkeypatch.context() as older:
        older.setattr(dialogue_import, "import_keys", lambda *_args, **_kwargs: None)  # derives nothing, moves nothing
        dialogue_import.run_dialogue_import(tmp_path)
    stored = ds.rows_by_identity(tmp_path, 1, archived["ts"])[0]
    revisions = {room["chat_id"]: room["revision"] for room in ds.rooms(tmp_path)}
    page = _read(tmp_path, nova)[1]
    assert "An archived Nova notice." not in _texts(tmp_path, nova)
    assert "chat_appearances_deriving" in page["window"]["truncated_by"] and room_appearances.owed(tmp_path)
    assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
    assert "An archived Nova notice." in _texts(tmp_path, nova) and not room_appearances.owed(tmp_path)
    assert "chat_appearances_deriving" not in _read(tmp_path, nova)[1]["window"]["truncated_by"]
    after = ds.row_at(tmp_path, "chat", stored.seq)
    assert after.row == stored.row and after.length == stored.length
    assert {room["chat_id"]: room["revision"] for room in ds.rooms(tmp_path)} == revisions


def _legacy_reply(target, ts, client_id, *, chat_id=1):
    """A Reply an older body saved: its reference resolved then, without ``target_home``."""
    from ouroboros.chat_references import reference_from_row

    legacy = {key: value for key, value in reference_from_row(target).items() if key != "target_home"}
    return legacy, {"ts": ts, "direction": "in", "chat_id": chat_id, "client_message_id": client_id,
                    "text": f"Owner words {client_id}", "reply_to": legacy}


def _gap_kinds(coverage):
    return {gap.get("kind") for gap in coverage.get("gaps") or ()}


def _index_keys(root, seq):
    with ds.transaction(root, write=False) as conn:
        return {record["key"] for record in conn.execute("SELECT key FROM room_appearances WHERE stream='chat' AND seq=?",
                                                         (seq,))}


_SUMMARY = {"ts": "2026-10-01T05:00:00Z", "direction": "system", "chat_id": 1, "task_id": "nova-root",
            "type": "project_completion_summary", "text": "Nova finished its series."}


class TestSavedTargetRecovery:
    """A saved target not found is a fact of the store it was read under, never of the process: once the
    store gains rows (an import, a repair) the next read decides it again, in every reader, and a gap it
    disclosed closes. A found target row is immutable and read once."""

    def test_an_unreadable_target_made_readable_joins_its_project_in_the_same_process(self, tmp_path):
        from ouroboros import room_appearances
        from ouroboros.dialogue_evidence import read_room_source
        from tests._dialogue import import_all, seed

        nova, other = _project(tmp_path, "nova"), _project(tmp_path, "other")
        _born_in(tmp_path, "nova-root", nova)
        _legacy, reply = _legacy_reply(_SUMMARY, "2026-10-01T06:00:00Z", "torn")
        seed(tmp_path, archives={"chat_20261001T000000.jsonl": [json.dumps(_SUMMARY)[:48] + "\n", reply]})
        import_all(tmp_path)
        assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        stored = ds.rows_by_identity(tmp_path, 1, reply["ts"])[0]
        assert "Owner words torn" not in _texts(tmp_path, nova)
        assert "chat_reply_target_unreadable" in _read(tmp_path, nova)[1]["window"]["truncated_by"]
        # The repair: the target's exact row is stored again, as an older body or a restore writes it. No cache is
        # cleared and nothing restarts: the next reads decide the Reply again.
        ds.append(tmp_path, "chat", dict(_SUMMARY), appearances=None)
        assert "Owner words torn" in _texts(tmp_path, nova) and "Owner words torn" in _texts(tmp_path, 1)
        assert "chat_reply_target_unreadable" not in _read(tmp_path, nova)[1]["window"]["truncated_by"]
        assert "Owner words torn" not in _texts(tmp_path, other)
        assert "Owner words torn" in _room_rows(tmp_path, nova)
        source = read_room_source(tmp_path, nova)
        assert "Owner words torn" in [row.get("text") for row in source["rows"]]
        assert "reply_target_unreadable" not in _gap_kinds(source["coverage"]["chat"])
        # The existing catch-up derives the row again: its gap keys go, the keys of its found home stay.
        assert "gap:task:nova-root" in _index_keys(tmp_path, stored.seq)
        assert room_appearances.owed(tmp_path) and room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        keys = _index_keys(tmp_path, stored.seq)
        assert not [key for key in keys if key.startswith("gap:")] and "task:nova-root" in keys
        assert "Owner words torn" in _texts(tmp_path, nova)
        assert ds.row_at(tmp_path, "chat", stored.seq).row == stored.row  # the owner's row is unchanged

    def test_an_absent_target_imported_later_joins_its_project_in_the_same_process(self, tmp_path):
        """Proven absent is no gap; a backup segment restored into the archive and imported brings the target,
        and the same process shows the Reply in its Project, in history, room reads and memory."""
        from ouroboros import memory_inventory as mi, room_appearances
        from tests._dialogue import import_all, seed

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        _legacy, reply = _legacy_reply(_SUMMARY, "2026-10-02T06:00:00Z", "restored")
        seed(tmp_path, archives={"chat_20261002T000000.jsonl": [reply]})
        import_all(tmp_path)
        assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        stored = ds.rows_by_identity(tmp_path, 1, reply["ts"])[0]
        assert "Owner words restored" not in _texts(tmp_path, nova)
        assert "chat_reply_target_unreadable" not in _read(tmp_path, nova)[1]["window"]["truncated_by"]
        assert mi.rooms_of_row(mi.row_meta(stored.row), mi.membership_facts(tmp_path)) == {"1"}
        seed(tmp_path, archives={"chat_20261001T000000.jsonl": [dict(_SUMMARY)]})
        import_all(tmp_path)  # the replan finds the restored segment and imports it
        assert ds.rows_by_identity(tmp_path, 1, _SUMMARY["ts"])
        assert "Owner words restored" in _texts(tmp_path, nova)
        assert "Owner words restored" in _room_rows(tmp_path, nova)
        assert mi.rooms_of_row(mi.row_meta(stored.row), mi.membership_facts(tmp_path)) == {"1", str(nova)}
        assert ds.row_at(tmp_path, "chat", stored.seq).row == stored.row

    def test_a_target_in_a_planned_segment_is_never_cached_absent_and_is_found_once_imported(self, tmp_path):
        """The stream ends are the cache's change counter, and the archive import fills intervals planned before it
        without moving them. While such a segment waits no read can decide its rows absent (the store answers
        importing and nothing is kept), so its import finds no stale decision. A segment restored later is placed past
        the ends and imported by the replan itself: the ends move then."""
        from ouroboros import dialogue_import, room_membership
        from ouroboros.projects_registry import RoomResolver
        from ouroboros.store import StoreMigrating
        from tests._dialogue import import_all, seed

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        legacy, reply = _legacy_reply(_SUMMARY, "2026-10-02T06:00:00Z", "planned")
        older, newer = "chat_20261001T000000.jsonl", "chat_20261002T000000.jsonl"
        seed(tmp_path, archives={older: [dict(_SUMMARY)], newer: [reply]})
        dialogue_import.cutover(tmp_path)  # both planned: the stream's end already lies past both
        planned = room_membership._stream_ends(tmp_path)
        assert dialogue_import.import_segment(tmp_path, "chat", newer)["status"] == "done"  # newest first, as the job
        with pytest.raises(StoreMigrating):
            room_membership._saved_target(RoomResolver(tmp_path), legacy)
        assert (str(tmp_path), legacy["address"]) not in room_membership._SAVED_TARGETS
        import_all(tmp_path)
        assert room_membership._stream_ends(tmp_path) == planned  # the import filled the planned interval in place
        assert room_membership._saved_target(RoomResolver(tmp_path), legacy)[0] == room_membership.TARGET_WORK

    def test_a_same_time_row_beside_unreadable_lines_is_no_proof_the_target_is_absent(self, tmp_path):
        """``chat_references.resolve_history_row``'s rule: another row at the target's chat and time, with
        unreadable lines in the stream, proves nothing; the Reply's Projects disclose the gap."""
        from ouroboros import room_appearances
        from ouroboros.room_membership import TARGET_UNKNOWN, _saved_target
        from ouroboros.projects_registry import RoomResolver
        from tests._dialogue import import_all, seed

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        sibling = {**_SUMMARY, "type": "task_summary", "text": "Another row of the same second."}
        legacy, reply = _legacy_reply(_SUMMARY, "2026-10-01T06:00:00Z", "beside")
        seed(tmp_path, archives={"chat_20261001T000000.jsonl": [json.dumps(_SUMMARY)[:48] + "\n", sibling, reply]})
        import_all(tmp_path)
        assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        assert _saved_target(RoomResolver(tmp_path), legacy)[0] == TARGET_UNKNOWN
        assert "chat_reply_target_unreadable" in _read(tmp_path, nova)[1]["window"]["truncated_by"]
        assert "Owner words beside" not in _texts(tmp_path, nova)

    def test_a_gap_is_disclosed_only_where_the_reply_could_be_but_is_not_shown(self, tmp_path):
        """A Reply written in a Project to an unreadable target of that Project's work is shown where it was
        written: no gap there. The other Project its saved reference names is where it could be missing."""
        from ouroboros import room_appearances
        from tests._dialogue import import_all, seed

        nova, other = _project(tmp_path, "nova"), _project(tmp_path, "other")
        _bound(tmp_path, "nova-work", "nova", nova)
        target = {"ts": "2026-10-01T05:00:00Z", "direction": "out", "chat_id": other, "task_id": "nova-work",
                  "text": "Spoken in Other."}
        _legacy, reply = _legacy_reply(target, "2026-10-01T06:00:00Z", "inside", chat_id=nova)
        seed(tmp_path, archives={"chat_20261001T000000.jsonl": [json.dumps(target)[:40] + "\n", reply]})
        import_all(tmp_path)
        assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
        assert "Owner words inside" in _texts(tmp_path, nova)
        assert "chat_reply_target_unreadable" not in _read(tmp_path, nova)[1]["window"]["truncated_by"]
        assert "chat_reply_target_unreadable" in _read(tmp_path, other)[1]["window"]["truncated_by"]
        assert "Owner words inside" not in _texts(tmp_path, other)

    def test_a_target_not_found_is_read_again_only_after_the_store_moved(self, tmp_path, monkeypatch):
        """The bound of the recovery: a found target is read once; one not found is read again only when a
        stream's end moved since (an append, a planned segment), never on every read."""
        from ouroboros import chat_chain, room_membership
        from ouroboros.projects_registry import RoomResolver

        nova = _project(tmp_path, "nova")
        _born_in(tmp_path, "nova-root", nova)
        ds.append(tmp_path, "chat", dict(_SUMMARY))
        found, _ = _legacy_reply(_SUMMARY, "2026-10-01T06:00:00Z", "found")
        missing, _ = _legacy_reply({**_SUMMARY, "text": "never stored"}, "2026-10-01T06:01:00Z", "missing")
        reads = []
        real = chat_chain.resolve_row
        monkeypatch.setattr(chat_chain, "resolve_row", lambda *a, **k: (reads.append(a[1]), real(*a, **k))[1])
        rooms = RoomResolver(tmp_path)
        for _ in range(3):
            assert room_membership._saved_target(rooms, found)[0] == room_membership.TARGET_WORK
            assert room_membership._saved_target(rooms, missing)[0] == room_membership.TARGET_ABSENT
        assert reads.count(found["address"]) == 1 and reads.count(missing["address"]) == 2  # chat, then progress
        ds.append(tmp_path, "chat", {"ts": "2026-10-01T07:00:00Z", "direction": "out", "chat_id": 1, "text": "later"})
        for _ in range(3):
            room_membership._saved_target(rooms, found)
            room_membership._saved_target(rooms, missing)
        assert reads.count(found["address"]) == 1 and reads.count(missing["address"]) == 4


def test_a_saved_reply_in_memory_agrees_with_history_and_evidence(tmp_path):
    """The memory inventory projects the reference fields the room rule reads, the saved target's address
    included: a readable saved target puts the Reply in its Project for history, room reads, evidence and
    the whole-mind inventory alike; a refusal's Reply stays Main's in every one of them."""
    from ouroboros import memory_inventory as mi, room_appearances
    from ouroboros.dialogue_evidence import read_room_source

    nova = _project(tmp_path, "nova")
    _born_in(tmp_path, "nova-root", nova)
    refusal = {**_SUMMARY, "ts": "2026-10-01T05:01:00Z", "type": "task_not_started", "text": "Not started"}
    for row in (_SUMMARY, refusal):
        ds.append(tmp_path, "chat", dict(row), appearances=None)
    receipts = {}
    for name, target, ts in (("summary", _SUMMARY, "2026-10-01T06:00:00Z"), ("refusal", refusal, "2026-10-01T06:01:00Z")):
        _legacy, reply = _legacy_reply(target, ts, name)
        receipts[name] = ds.append(tmp_path, "chat", reply, appearances=None)
    assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
    facts = mi.membership_facts(tmp_path)
    memory = {name: mi.rooms_of_row(mi.row_meta(ds.row_at(tmp_path, "chat", receipt.seq).row), facts)
              for name, receipt in receipts.items()}
    assert memory == {"summary": {"1", str(nova)}, "refusal": {"1"}}
    evidence = [row.get("text") for row in read_room_source(tmp_path, nova)["rows"]]
    for readers in (_texts(tmp_path, nova), _room_rows(tmp_path, nova), evidence):
        assert "Owner words summary" in readers and "Owner words refusal" not in readers


def test_an_unreadable_target_is_a_gap_of_its_projects_memory_and_evidence_reads(tmp_path):
    """History, room reads, evidence and the memory view name the same gap for the Projects the Reply could
    join, and only for them; rows not derived yet are a gap of the indexed reads too."""
    from ouroboros import chat_chain, room_appearances
    from ouroboros.dialogue_evidence import read_room_source
    from tests._dialogue import import_all, seed

    nova, other = _project(tmp_path, "nova"), _project(tmp_path, "other")
    _born_in(tmp_path, "nova-root", nova)
    _legacy, reply = _legacy_reply(_SUMMARY, "2026-10-01T06:00:00Z", "torn")
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [json.dumps(_SUMMARY)[:48] + "\n", reply]})
    import_all(tmp_path)
    assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
    for chat, expected in ((nova, {"reply_target_unreadable"}), (other, set())):
        gaps: set = set()
        list(chat_chain.room_rows(tmp_path, chat, gaps=gaps))
        assert gaps == expected
        assert _gap_kinds(read_room_source(tmp_path, chat)["coverage"]["chat"]) & {"reply_target_unreadable"} == expected
    # An older body's notice above the mark: the indexed reads say so until the job derives it.
    ds.append(tmp_path, "chat", {"ts": "2026-10-02T00:00:00Z", "direction": "out", "chat_id": 1, "task_id": "nova-root",
                                 "type": "main_notice", "text": "Old Nova notice."}, feed=True, appearances=None)
    gaps = set()
    list(chat_chain.room_rows(tmp_path, nova, gaps=gaps))
    assert gaps == {"reply_target_unreadable", "appearances_deriving"}
    assert "appearances_deriving" in _gap_kinds(read_room_source(tmp_path, nova)["coverage"]["chat"])
    assert room_appearances.run(tmp_path, lambda: False)["state"] == "done"
    gaps = set()
    assert "Old Nova notice." in [entry.row.get("text") for entry in chat_chain.room_rows(tmp_path, nova, gaps=gaps)]
    assert gaps == {"reply_target_unreadable"}


def test_the_memory_view_names_the_gap_of_a_room_a_reply_could_belong_to(tmp_path):
    """The whole-mind inventory reads every open row: a Reply whose saved target cannot be read is Main's
    there, and the Project it could join says its open rows may lack it, in the room's standing facts."""
    from ouroboros import dialogue_store, memory_inventory as mi
    from tests import _memory_inventory_shared as shared
    from tests._memory_view_context import room_text

    rooms = shared.world(tmp_path)  # its stream holds a torn line: an absent target is unknown there
    alpha, beta = rooms["alpha"], rooms["beta"]
    target = {"ts": "2026-09-04T00:00:00+00:00", "direction": "system", "chat_id": 1, "task_id": "bound",
              "type": "project_completion_summary", "text": "Bound work finished."}
    _legacy, reply = _legacy_reply(target, "2026-09-04T00:01:00+00:00", "unknown-target")
    dialogue_store.append(tmp_path, "chat", reply)
    gaps: dict = {}
    by_room = mi.open_rows_by_room(tmp_path, gaps=gaps)
    assert gaps == {str(alpha): {"reply_target_unreadable"}}
    assert "unknown-target" not in [meta.get("client_message_id") for _a, meta, _p in by_room.get(str(alpha), [])]
    assert "unknown-target" in [meta.get("client_message_id") for _a, meta, _p in by_room["1"]]
    def head(chat):
        return next(line for line in room_text(tmp_path, {"id": f"turn-{chat}", "chat_id": chat}).splitlines()
                    if line.startswith("## This room"))

    assert "incomplete: a Reply whose target cannot be read may belong here" in head(alpha), head(alpha)
    assert "cannot be read" not in head(beta)
    # The target stored again: the same process puts the Reply in its Project and the gap closes.
    dialogue_store.append(tmp_path, "chat", target)
    gaps = {}
    assert str(alpha) in {room for room, entries in mi.open_rows_by_room(tmp_path, gaps=gaps).items()
                          if any(meta.get("client_message_id") == "unknown-target" for _a, meta, _p in entries)}
    assert gaps == {}
    assert "cannot be read" not in head(alpha)
