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


_SEEDS, _BEDS = "c" * 64, "a" * 64


def _drafted(corpus, *, text="Check the seed totals.", ref=None):
    """A drafted start's task as its first request projects it: the origin ref and its recorded words at the top
    level, the drafter's carried corpus under ``metadata``. ``ref``: the origin the door stamped (the selected
    message by default, with its row key)."""
    ref = {"chat_id": 1, "client_message_id": "owner-seeds", "ts": "2026-11-20T10:02:00+00:00",
           "text_sha256": "e" * 64, "row_sha256": _SEEDS} if ref is None else ref
    return {"id": "totals", "text": "Check the totals", "origin_message_ref": ref,
            **({"origin_message_text": text} if text else {}),
            "metadata": {"objective_author": {"kind": "task", "task_id": "root-1"}, "owner_corpus": corpus}}


def _shown(task):
    from ouroboros.main_context_authority import project_main_task_authority

    return project_main_task_authority(task).get("task_owner_words", {}).get("rows", [])


def test_the_first_request_folds_the_opening_by_identity_and_keeps_every_neighbour():
    """The selected message (seed totals, with its file) opens the work; the garden-beds message, a later one that
    says the same sentence again, and one that carried only a photo ride beside it. The projection shows the
    opening once, as ``task_authority_origin``; the three neighbours stay whole in ``task_owner_words``, each under
    its own source, time and identity, their files named and not given. The stored corpus is not rewritten."""
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.owner_relay import NOT_GIVEN

    seeds = {"source": "origin_message", "content": "Check the seed totals.", "chat_id": "1", "ts": "2026-11-20T10:02:00+00:00",
             "client_message_id": "owner-seeds", "row_sha256": _SEEDS, "attachments": [{"name": "seed-counts.csv", "size": 41}]}
    beds = {"source": "origin_message", "content": "Plan the garden beds.", "chat_id": "1", "ts": "2026-11-20T10:01:00+00:00",
            "client_message_id": "owner-beds", "row_sha256": _BEDS, "attachments": [{"name": "garden-beds.txt", "size": 12}]}
    again = {"source": "owner_mailbox", "content": "Check the seed totals.", "chat_id": "1", "carried": True,
             "ts": "2026-11-20T10:07:00+00:00", "client_message_id": "owner-again", "row_sha256": "b" * 64}
    photo = {"source": "owner_mailbox", "content": "", "chat_id": "1", "ts": "2026-11-20T10:09:00+00:00", "carried": True,
             "client_message_id": "owner-photo", "row_sha256": "d" * 64, "attachments": [{"name": "plot.jpg", "size": 7}]}
    task = _drafted([seeds, beds, again, photo])
    projection = project_main_task_authority(task)
    assert projection["task_authority_origin"]["text"] == "Check the seed totals.", "the opening's words, once"
    rows = projection["task_owner_words"]["rows"]
    assert [(row["client_message_id"], row["source"], row["ts"], row["content"]) for row in rows] == [
        ("owner-beds", "origin_message", "2026-11-20T10:01:00+00:00", "Plan the garden beds."),
        ("owner-again", "owner_mailbox", "2026-11-20T10:07:00+00:00", "Check the seed totals."),
        ("owner-photo", "owner_mailbox", "2026-11-20T10:09:00+00:00", "")], "equal words are not one message"
    assert all(ref["staged"].startswith(NOT_GIVEN) for row in rows for ref in row.get("attachments") or [])
    assert [ref["name"] for ref in rows[0]["attachments"]] == ["garden-beds.txt"]
    assert [ref["name"] for ref in rows[2]["attachments"]] == ["plot.jpg"], "a row of files alone keeps them named"
    assert all("staged" not in ref for row in task["metadata"]["owner_corpus"] for ref in row.get("attachments") or [])
    # The same opening carried twice is one row: its repeat folds with it, and nothing else does.
    assert [row["client_message_id"] for row in _shown(_drafted([seeds, dict(seeds), beds]))] == ["owner-beds"]


def test_an_opening_that_carried_only_a_file_folds_and_one_whose_words_are_not_shown_stays():
    """A file-only opening has no words to show beside its ref, and its file is the work's own: it folds. An opening
    the task records no words for is not shown by ``task_authority_origin``, so its carried row keeps them."""
    only_file = {"source": "origin_message", "content": "", "chat_id": "1", "client_message_id": "owner-seeds",
                 "row_sha256": _SEEDS, "attachments": [{"name": "seed-counts.csv", "size": 41}]}
    assert _shown(_drafted([only_file], text="[file: seed-counts.csv]")) == []
    worded = {**only_file, "content": "Check the seed totals."}
    assert [row["content"] for row in _shown(_drafted([worded], text=""))] == ["Check the seed totals."]
    assert all("staged" not in ref for row in _shown(_drafted([worded], text="")) for ref in row["attachments"]), (
        "the opening's own file is this work's: never marked as not given")


def test_an_older_record_keeps_its_known_opening_folded_and_an_untold_relation_whole():
    """Records written before row keys: the opening named by its room and client id folds as before, and so does
    the first row a record kept with no identity at all. A carried row with no identity that only says the same
    words has no relation the record can tell: it stays, under its own source."""
    old_ref = {"chat_id": 1, "client_message_id": "owner-seeds", "ts": "2026-10-01T09:00:00+00:00", "text_sha256": "e" * 64}
    by_client = {"source": "origin_message", "content": "Check the seed totals.", "chat_id": "1", "client_message_id": "owner-seeds"}
    follow = {"source": "owner_mailbox", "content": "And keep the finance export.", "msg_id": "mail-1"}
    assert [row["content"] for row in _shown(_drafted([by_client, follow], ref=old_ref))] == ["And keep the finance export."]
    for first in ({"source": "origin_message", "content": "Check the seed totals."},
                  {"source": "initial_user", "content": "Check the seed totals."}):
        assert [row["content"] for row in _shown(_drafted([first, follow], ref=old_ref))] == ["And keep the finance export."]
    untold = {"source": "initial_user", "content": "Check the seed totals.", "carried": True}
    assert [(row["source"], row["content"]) for row in _shown(_drafted([by_client, untold], ref=old_ref))] == [
        ("initial_user", "Check the seed totals.")]
    # The older stamp names a time too: the same sentence sent again under the same client id is another message.
    later = {**by_client, "ts": "2026-10-01T09:05:00+00:00"}
    assert [row["ts"] for row in _shown(_drafted([by_client, later], ref=old_ref))] == ["2026-10-01T09:05:00+00:00"]


def test_a_row_key_only_one_side_names_tells_no_opening():
    """The fold needs the identity the record holds, never one it could guess: when the origin names its physical row
    and a carried row names only the room and client id, or the row names a key the older stamp lacks, the record
    does not tell they are one row. The row stays whole beside the origin, with its own source and words."""
    by_client = {"source": "owner_mailbox", "content": "Check the seed totals.", "chat_id": "1", "client_message_id": "owner-seeds"}
    assert [(row["source"], row["content"]) for row in _shown(_drafted([by_client]))] == [
        ("owner_mailbox", "Check the seed totals.")]
    old_ref = {"chat_id": 1, "client_message_id": "owner-seeds", "ts": "2026-10-01T09:00:00+00:00", "text_sha256": "e" * 64}
    keyed = {**by_client, "row_sha256": _SEEDS}
    assert [row["row_sha256"] for row in _shown(_drafted([keyed], ref=old_ref))] == [_SEEDS]
    assert _shown(_drafted([keyed])) == [], "the same key on both sides is the opening, shown once"


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


@pytest.mark.parametrize("cited_door", ["owner_press"], indirect=True)
@pytest.mark.parametrize("door", ["start", "model", "schedule", "update"])
def test_a_wake_can_cite_the_whole_pressed_choice_for_owner_work(cited_door, door):
    from ouroboros.consciousness_wake import wake_task_metadata
    from ouroboros.owner_words import answered_card_reply, owner_words_contain
    from ouroboros.tools import control_routing, followup
    from supervisor import queue as schedules

    ctx, events, _ = cited_door
    ctx.task_metadata.update(wake_task_metadata("act", "heartbeat"))
    label = "Catalogue the blue reeds"
    card = {"quiz_id": "marsh-card", "question": "Which work?", "options": [label], "answered_index": 0}
    ctx._owner_directives = [{"source": "owner_press", "content": "1. " + label, "quiz_id": card["quiz_id"],
                              "source_task_id": "marsh-keeper", "msg_id": "marsh-answer",
                              "reply_to": answered_card_reply(card, "marsh-keeper")}]
    when = "2040-05-01T00:00:00+00:00"
    if door == "update":
        assert followup._handle_schedule_followup(ctx, relation="independent", objective="Read the marsh log",
                                                  run_at=when).startswith("FOLLOWUP_SCHEDULED")
    before = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    def choose(cite):
        if door in ("start", "model"):
            return control_routing._start_task(ctx, "Catalogue the reeds", title="Marsh catalogue",
                                               model="vendor/example" if door == "model" else "", owner_words=cite)
        if door == "schedule":
            return followup._handle_schedule_followup(ctx, relation="independent", objective="Catalogue the reeds",
                                                      run_at=when, owner_words=cite)
        return followup._manage_schedules(ctx, action="update", schedule_id=before[0]["id"],
                                          objective="Catalogue the reeds", reason="The owner chose a card", owner_words=cite)

    assert "OWNER_WORDS_UNVERIFIED" in choose("blue reeds") and not events
    assert schedules.list_scheduled_tasks(ctx.drive_root)["tasks"] == before
    assert "OWNER_WORDS_UNVERIFIED" not in choose(label)
    if events:
        event = events[0]
        kind, meta, corpus = event["work_kind"], event.get("metadata", {}), event["owner_corpus"]
    else:
        [saved] = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
        kind, meta = saved["writer_kind"], saved["task"]["metadata"]
        corpus = meta["owner_corpus"]
    assert any(row.get("source") == "owner_press" and row.get("cited") == label and row.get("msg_id") == "marsh-answer"
               for row in corpus)
    if door == "model":
        assert meta["lead"]["source"] == "owner_press" and meta["lead"]["cited"] == label
    assert kind == "owner"
    assert not owner_words_contain(ctx, label)


def test_the_model_and_work_tools_describe_both_allowed_owner_bases():
    from ouroboros.tools import control, followup

    entries = {entry.name: entry.schema for entry in [*control.get_tools(), *followup.get_tools()]}
    for name in ("start_task", "switch_model", "schedule_followup", "manage_schedules"):
        description = entries[name]["parameters"]["properties"]["owner_words"]["description"]
        assert "verbatim" in description and "complete label" in description and "press" in description
