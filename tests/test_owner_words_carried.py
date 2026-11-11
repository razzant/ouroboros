"""My human's words reach the work they caused, by value: from a place turn that cites them, through a follow-up
(once and cron) to its root and that root's helpers, with the message they replied to under its own author.

Every guard is checked both ways: the words arrive where my human's cited words started the work, and stay out
where they did not (uncited provenance, a participant, a forward, a run with no owner words).
"""
from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store
from ouroboros.chat_chain import format_address, row_address
from ouroboros.loop_messages import _initialize_owner_directives, _record_owner_directive
from ouroboros.owner_relay import words_unseen_in
from ouroboros.owner_words import (FIELD, OWNER_SOURCES, REPLY_TARGET, _deduped, carried_owner_corpus,
                                   governing_words_for_schedule, render_owner_words)
from ouroboros.tools.registry import ToolContext

LONG = ("Sasha's estimate is fine, but before you start: keep the old export path working for the finance team, "
        "write the migration as a reversible step, and only then switch the default. ") * 3 + "Do NOT ship on Friday."


def _place_turn(tmp_path, *, confirmed=True, reply_to_message_id="", forwarded=False, message=None):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_task_type="presence")
    ctx.task_id = "place-turn"
    message = {**(message or {}), **({"reply_to_message_id": reply_to_message_id} if reply_to_message_id else {})}
    message.update({"forward_origin": {"sender_user_name": "Andrew"}} if forwarded else {})
    ctx.task_metadata = {
        "focus": {"place_id": "p1", "room_chat_id": 2 ** 41 + 3,
                  "actor": {"platform_actor_id": "anton", "owner_confirmed": confirmed}},
        "presence": {"observed_text": LONG, "event": {"source_event_id": "ev-9", "message": message}}}
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[FOCUS] an event in Team chat …"}])
    return ctx


def test_a_confirmed_place_turn_records_my_humans_exact_words_as_provenance_that_a_cite_carries(tmp_path):
    """Their words from a confirmed account are provenance (no directive, no verdict about whom they are for): a
    helper or schedule gets no governing words from them; only the cite the mind makes carries them whole."""
    from ouroboros.owner_words import PLACE_CONFIRMED, cited_owner_basis
    ctx = _place_turn(tmp_path)
    place, frame = ctx._owner_directives
    assert place == {"source": PLACE_CONFIRMED, "content": LONG, "msg_id": f"chat {2 ** 41 + 3} / ev-9", "carried": True}
    assert frame["source"] == "initial_text", "the event frame stays the first text Safety reads"
    assert FIELD not in governing_words_for_schedule(ctx), "provenance alone governs nothing"
    basis, error = cited_owner_basis(ctx, LONG)
    assert error == "" and basis["content"] == LONG and basis["basis"] == "cited"
    seeded = SimpleNamespace(task_metadata={"objective_author": {"kind": "task", "task_id": "place-turn"}, "owner_corpus": [basis]})
    _initialize_owner_directives(seeded, [{"role": "user", "content": "drafted"}])
    assert [row["text"] for row in governing_words_for_schedule(seeded)[FIELD]] == [LONG], "a helper gets the whole words, not a cut"


@pytest.mark.parametrize("confirmed,forwarded", [(False, False), (True, True)])
def test_a_participant_or_a_forward_is_not_my_humans_words(tmp_path, confirmed, forwarded):
    ctx = _place_turn(tmp_path, confirmed=confirmed, forwarded=forwarded)  # a forward: another author
    assert [row["source"] for row in ctx._owner_directives] == ["initial_text"]
    assert FIELD not in governing_words_for_schedule(ctx)
    from ouroboros.owner_words import cited_owner_basis
    assert cited_owner_basis(ctx, LONG)[0] is None, "their bytes cannot be cited as my human's"


def test_a_place_reply_names_its_target_by_address_when_the_connector_gave_no_words(tmp_path):
    from ouroboros.owner_words import cited_owner_basis
    basis, _error = cited_owner_basis(_place_turn(tmp_path, reply_to_message_id="m-41"), LONG)
    seeded = SimpleNamespace(task_metadata={"objective_author": {"kind": "task", "task_id": "place-turn"}, "owner_corpus": [basis]})
    _initialize_owner_directives(seeded, [{"role": "user", "content": "drafted"}])
    seeded.task_id = "place-turn"
    rows = governing_words_for_schedule(seeded)[FIELD]
    assert rows[1] == {"text": "", "source": REPLY_TARGET, "carrier": "ctx", "task_id": "place-turn", "ts": "",
                       "ref": "platform message m-41 in this room", "author": "author not recorded"}
    text = render_owner_words(rows, audience="child")
    assert "author not recorded: the message my human replied to, not my human's words" in text
    assert "(its words are not carried: the connector named only its platform message id, no host address)" in text
    assert "this address reads them" not in text, "no host address reads a platform message id"


def test_a_place_reply_carries_its_known_words_and_their_author_as_context_not_an_order(tmp_path):
    from ouroboros.owner_words import cited_owner_basis, owner_words_contain

    words = "  A proposal from someone else.\n"
    ctx = _place_turn(tmp_path, message={"reply_to_message": {"message_id": 41,
        "from": {"id": 8, "first_name": "Sasha"}, "text": words}, "quote": {"text": "proposal"}})
    basis, error = cited_owner_basis(ctx, LONG)
    assert not error and basis["reply_to"]["text"] == words
    assert "Sasha" in basis["reply_to"]["author"] and "41" in basis["reply_to"]["address"]
    seeded = SimpleNamespace(task_id="new-work", task_metadata={"owner_corpus": [basis],
        "objective_author": {"kind": "task", "task_id": "place-turn"}})
    _initialize_owner_directives(seeded, [{"role": "user", "content": "drafted"}])
    assert "not my human's words" in render_owner_words(governing_words_for_schedule(seeded)[FIELD], audience="child")
    assert not owner_words_contain(seeded, words), "a reply target is context, never the owner's order"


def _owner_root(tmp_path, metadata=None):
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data", task_id="root-1",
                      task_metadata={"root_task_id": "root-1", "delegation_role": "root", **(metadata or {})},
                      task_contract={"objective": "x", "delegation_role": "root"})
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    return ctx


@pytest.mark.parametrize("trigger", [{"run_at": "2030-01-01T00:00:00+00:00"}, {"cron": "0 9 * * 1"}])
def test_a_follow_up_carries_the_words_to_its_root_and_the_roots_helpers(tmp_path, trigger):
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules

    ctx = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-7"}})
    _initialize_owner_directives(ctx, [{"role": "user", "content": LONG}])
    _record_owner_directive(ctx, source="owner_mailbox", content="And keep the finance export.", msg_id="mail-1")
    out = _handle_schedule_followup(ctx, relation="independent", objective="Check the migration.", **trigger)
    assert out.startswith("FOLLOWUP_SCHEDULED"), out
    [record] = schedules.list_scheduled_tasks(tmp_path / "data")["tasks"]
    carried = record["task"]["metadata"]["owner_corpus"]
    assert [row["content"] for row in carried] == [LONG, "And keep the finance export."]
    successor = schedules._task_from_schedule(record)
    live = SimpleNamespace(task_id="fu-1", task_metadata=successor["metadata"], current_task_type="task")
    _initialize_owner_directives(live, [{"role": "user", "content": successor["text"]}])
    helper = governing_words_for_schedule(live)[FIELD]
    assert [row["text"] for row in helper] == [LONG, "And keep the finance export."], "the second generation too"
    shown = project_main_task_authority(successor)["task_owner_words"]
    assert shown["carried_by"] == "root-1" and [row["content"] for row in shown["rows"]] == [
        LONG, "And keep the finance export."], "the follow-up's own mind sees the words, not only its helpers"


def test_a_follow_up_of_a_run_without_owner_words_claims_none(tmp_path):
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules

    ctx = _owner_root(tmp_path, {"initiator": "consciousness"})
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[Wake-up] my own time"}])
    assert _handle_schedule_followup(ctx, relation="independent", objective="Look again.",
                                     run_at="2030-01-01T00:00:00+00:00").startswith("FOLLOWUP_SCHEDULED")
    [record] = schedules.list_scheduled_tasks(tmp_path / "data")["tasks"]
    assert "owner_corpus" not in record["task"]["metadata"]
    assert "task_owner_words" not in project_main_task_authority(schedules._task_from_schedule(record))


def test_one_list_names_my_humans_corpus_labels_for_every_reader(tmp_path):
    from ouroboros.owner_continue import CORPUS_INITIAL_SOURCES, CORPUS_OWNER_SOURCES
    from ouroboros.tools.control_routing import _attach_drafted_objective

    from ouroboros.owner_words import cited_owner_basis
    assert "place_owner" in OWNER_SOURCES and CORPUS_OWNER_SOURCES == OWNER_SOURCES - CORPUS_INITIAL_SOURCES
    ctx = _place_turn(tmp_path)
    evt: dict = {}
    _attach_drafted_objective(ctx, evt)
    assert evt["owner_corpus"] == [] == carried_owner_corpus(ctx), "provenance rows are not the corpus a start carries"
    corpus = [*evt["owner_corpus"], cited_owner_basis(ctx, LONG)[0]]  # what start_task adds on a verified cite
    seeded = SimpleNamespace(task_metadata={"objective_author": {"kind": "task", "task_id": "place-turn"}, "owner_corpus": corpus})
    _initialize_owner_directives(seeded, [{"role": "user", "content": "drafted"}])
    assert [(row["source"], row["content"]) for row in seeded._owner_directives] == [("place_owner", LONG)]


def test_words_repeat_only_by_identity_never_by_likeness(tmp_path):
    def row(text, ref, ts=""):
        return {"text": text, "source": "owner_mailbox", "carrier": "ctx", "task_id": "t", "ts": ts, "ref": ref}

    assert len(_deduped([row("да", "msg a"), row("да", "msg b")])) == 2, "two yeses are two messages"
    assert len(_deduped([row("да", "msg a"), row("да", "msg a")])) == 1
    assert len(_deduped([row("да", ""), row("да", "")])) == 1, "without an identity the text is the only key"
    # A row with no reference is told by its words and its time: two id-less «да» at two times are two rows,
    # and the first of one identity keeps its carrier.
    two = _deduped([row("да", "", "received 2026-11-05T10:00:00+00:00"), row("да", "", "received 2026-11-05T10:05:00+00:00")])
    assert [r["ts"] for r in two] == ["received 2026-11-05T10:00:00+00:00", "received 2026-11-05T10:05:00+00:00"]
    assert len(_deduped([row("да", "", "2026-11-05T10:00:00+00:00"), row("да", "", "2026-11-05T10:00:00+00:00")])) == 1
    # In an earlier subject a row with no reference is told by its words whole under a stamp with its time, never
    # by its words anywhere in the text: «да» heading another row's words does not hide an undated «да», nor does
    # a dated «да»; its own undated block does, whatever section follows it.
    ctx = _owner_root(tmp_path)
    ctx._owner_directives = [{"source": "owner_mailbox", "content": "да"}]
    dated, undated = "[2026-11-05T10:00:00+00:00 · owner · owner_mailbox of task t]", "[owner · owner_mailbox of task t]"
    assert [r["text"] for r in words_unseen_in(ctx, f"{undated}\nда, но позже")] == ["да"]
    assert [r["text"] for r in words_unseen_in(ctx, f"{dated}\nда")] == ["да"]
    assert words_unseen_in(ctx, f"{undated}\nда\n\n## The reply\n\nда") == []


def test_the_message_my_human_replied_to_rides_whole_under_its_author(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    spec = {"ts": "2026-11-04T09:00:00+00:00", "direction": "in", "chat_id": 1, "text": LONG + " — Sasha",
            "sender_label": "Sasha", "client_message_id": "sasha-1"}
    dialogue_store.append(root, "chat", spec)
    stored = next(entry.row for entry in dialogue_store.iter_rows(root, "chat"))
    address = format_address(row_address(stored))
    ctx = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-8"},
                                 "reply_reference": {"address": address, "author_label": "Sasha"}})
    ctx.event_queue = queue.Queue()
    _initialize_owner_directives(ctx, [{"role": "user", "content": "Do this."}])
    [owner] = ctx._owner_directives
    assert owner["reply_to"] == {"address": address, "author": "Sasha", "text": LONG + " — Sasha"}
    rows = governing_words_for_schedule(ctx)[FIELD]
    assert [(row["source"], row["text"]) for row in rows] == [("initial_user", "Do this."),
                                                             (REPLY_TARGET, LONG + " — Sasha")]
    text = render_owner_words(rows, audience="child")
    assert "Sasha: the message my human replied to, not my human's words · " + address in text
    from ouroboros.loop_messages import owner_input_revision
    assert owner_input_revision(ctx) == 1, "the target is context, never a second owner message"
    lost = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-9"},
                                  "reply_reference": {"address": "row:1@2026-11-04T09:00:00+00:00#000000000000"}})
    _initialize_owner_directives(lost, [{"role": "user", "content": "Do that."}])
    assert lost._owner_directives[0]["reply_to"]["text"] == "", "an unreadable target keeps its address alone"


def test_a_root_hands_on_the_words_it_carries_beside_its_own(tmp_path):
    carried = {"text": "The original ask.", "source": "origin_message", "carrier": "ctx", "task_id": "first",
               "ts": "", "ref": "chat 1 / owner-1"}
    ctx = _owner_root(tmp_path, {FIELD: [carried], "origin_message_ref": {"chat_id": 1, "client_message_id": "o2"}})
    _initialize_owner_directives(ctx, [{"role": "user", "content": "A newer word."}])
    assert [row["text"] for row in governing_words_for_schedule(ctx)[FIELD]] == ["The original ask.", "A newer word."]


@pytest.fixture(params=["live_call", "initial_user", "confirmed_owner_place"])
def cited_door(tmp_path, monkeypatch, request):
    from ouroboros.owner_words import PLACE_CONFIRMED
    from ouroboros.tools import control_models, control_routing
    from ouroboros.tools.tool_result import _install_tool_result_sidecar, _restore_tool_result_sidecar

    ctx = _owner_root(tmp_path)
    source = PLACE_CONFIRMED if request.param == "confirmed_owner_place" else request.param
    ctx._owner_directives = [{"source": source, "content": LONG, "msg_id": "cited-item"}]
    events = []
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing",
                        lambda _ctx, event: (events.append(event) or "live", {"status": "scheduled"}))
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda ctx: "")
    monkeypatch.setattr(control_models, "known_models", lambda: [{"model": "vendor/example", "use_local": False}])
    token = _install_tool_result_sidecar(ctx, object())
    try:
        yield ctx, events, source == "live_call"
    finally:
        _restore_tool_result_sidecar(token)


@pytest.mark.parametrize("door", ["start", "schedule", "update"])
def test_only_typed_or_confirmed_place_words_authorise_a_plain_start_or_schedule(cited_door, door):
    from ouroboros.tools import control_routing, followup
    from ouroboros.tools.tool_result import _published_tool_result
    from supervisor import queue as schedules

    ctx, events, spoken = cited_door
    if door == "update":
        assert followup._handle_schedule_followup(ctx, relation="independent", objective="Original work",
            run_at="2030-01-01T00:00:00+00:00").startswith("FOLLOWUP_SCHEDULED")
    before = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    if door == "start":
        text = control_routing._start_task(ctx, "Prepare the work", title="Report", owner_words="Do NOT ship on Friday.")
    elif door == "schedule":
        text = followup._handle_schedule_followup(ctx, relation="independent", objective="Prepare the work",
            run_at="2030-01-01T00:00:00+00:00", owner_words="Do NOT ship on Friday.")
    else:
        text = followup._manage_schedules(ctx, action="update", schedule_id=before[0]["id"],
            objective="Prepare the work", reason="The assignment changed", owner_words="Do NOT ship on Friday.")
    after = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    if spoken:
        assert "OWNER_WORDS_UNVERIFIED" in text and not events and after == before
        assert _published_tool_result(ctx, None).status == "error"
        return
    assert "OWNER_WORDS_UNVERIFIED" not in text
    corpus = events[-1]["owner_corpus"] if door == "start" else after[0]["task"]["metadata"]["owner_corpus"]
    assert any(row.get("cited") == "Do NOT ship on Friday." and row["content"] == LONG for row in corpus)
    if door == "update":
        assert after[0]["task"]["text"] == "Prepare the work"


def test_a_model_start_carries_spoken_words_only_on_the_lead_and_keeps_typed_owner_words_whole(cited_door):
    from ouroboros.owner_words import owner_words_text
    from ouroboros.tools.control_routing import _start_task

    ctx, events, spoken = cited_door
    text = _start_task(ctx, "Prepare the work", title="Report", model="vendor/example", owner_words="Do NOT ship on Friday.")
    assert text.startswith("OK: task")
    event = events[-1]
    assert event["metadata"]["lead"]["cited"] == "Do NOT ship on Friday."
    assert event["objective_author"]["kind"] == "task"
    assert bool(event["owner_corpus"]) is not spoken
    seeded = SimpleNamespace(task_metadata={"objective_author": event["objective_author"], "owner_corpus": event["owner_corpus"]})
    _initialize_owner_directives(seeded, [{"role": "user", "content": "drafted"}])
    assert bool(governing_words_for_schedule(seeded).get(FIELD)) is not spoken
    assert (LONG in owner_words_text(seeded, audience="reviewer")) is not spoken
    if not spoken:
        assert all(row["content"] == LONG for row in event["owner_corpus"])
