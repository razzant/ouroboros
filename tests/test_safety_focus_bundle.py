"""A place reply's Supervisor bundle, built by the real producer and consumer.

The focus task comes from the presence producer, its first row from the real FOCUS frame, the reply
check from ``prepare_focus_reply``; only the physical model call is recorded instead of sent.
Each source projection appears once, a changed note keeps both versions, and the reply's route
names the thread it answers.
"""
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import places as p
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_bindings import conversation_key
from ouroboros.presence_runner import PresenceTurnEvent, _build_task
from tests._safety_fixtures import reset_safety_stores
from tests.test_place_focus import focus as _focus

focus = _focus
RULES = "Rule one: keep private work private.\nRule two: hold a disputed send and ask me in Main."
OPENING = "host or Ouroboros text that opened this run; not the owner's words"
ORIGIN_AUTHOR = "the owner: a message recorded as a work's origin"
LATER_OWNER_AUTHOR = "the owner: a later message to this work"
CARRIED_AUTHOR = "a record of owner input carried into this run; any host framing is not the owner's wording"
PRESS_AUTHOR = ("the owner's answer by a press of an offered option on the card in reply_to, with no words of their own "
                "(the option's label is the card's text, written by Ouroboros)")
PLACE_AUTHOR = "a message from the owner's confirmed account in the source conversation"
CLOSED_AUTHOR = ("the host's notice that the owner closed the question it names without an answer: no option was "
                 "chosen and nothing was approved. The notice's words are the host's; the reason it quotes is the "
                 "owner's own words or the host's standard sentence")
MAIL_AUTHOR = ("input that reached this work under the mail label without a retained kind: the owner's own words or the "
               "host's record of an act of theirs, such as closing a question; the row does not show which")
CARD_NOTE = ("reference text kept at this card's address: it may come from a different revision of the card, and a label "
             "in it from another press; this submission's own fields are recorded separately (pressed_label, "
             "submitted_values)")


def _whole(rows):
    """Recorded opening rows as the Supervisor reads them: whole, each naming who wrote it."""
    return [{**row, "author": OPENING} for row in rows]


def _sources(ctx, messages=None):
    from ouroboros import safety
    prompt = safety._build_check_prompt("run_command", {"cmd": ["echo", "hello"]}, messages, ctx=ctx)
    return json.loads(prompt.split("Host evidence (provenance is not consent):\n", 1)[1]
                      .split("\nProposed tool call:", 1)[0])


@pytest.fixture
def checker(monkeypatch):
    """The real check path; the physical model call records the prompt and answers SAFE."""
    from ouroboros import safety

    prompts = []
    reset_safety_stores()
    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 0)
    monkeypatch.setattr(safety, "_resolve_safety_routing", lambda: (False, False, None))
    monkeypatch.setattr(safety, "_emit_durable_safety_event", lambda ctx, event: None)
    monkeypatch.setattr(safety, "get_safety_model", lambda: "fixture/checker")
    monkeypatch.setattr(safety, "get_safety_mode", lambda kind="owner": "full")
    monkeypatch.setenv("OUROBOROS_SAFETY_OWNER_RULES", RULES)

    def call(**kwargs):
        prompts.append(kwargs["user_prompt"])
        return {"content": json.dumps({"status": "SAFE", "reason": "fine"})}, {}
    monkeypatch.setattr(safety, "_safety_model_call", call)
    yield prompts
    reset_safety_stores()


def _focus_task(root, *, text="hello there", thread="", staged=(), conversation=None, message=None):
    """One admitted event's focus task from the presence producer, and the first user content it frames."""
    from ouroboros.context import build_user_content

    key = conversation_key("chat", "a", "r", thread)
    event = PresenceTurnEvent("event-" + hashlib.sha256(f"{thread}:{text}".encode()).hexdigest()[:8], "chat", "a",
                              "r", thread, key, {"platform_actor_id": "person", "display_name": "Person"},
                              conversation or {}, message or {}, text, facts=("mention",))
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r", thread_id=thread)
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2,
                                                     "roots": ["root-a", "root-b"]}),
                       event, drive_root=root, staged_files=tuple(staged))
    task["drive_root"] = str(root)  # the worker lane binds it before the context reads staged images
    return task, build_user_content(task)


def _reply_bundle(root, monkeypatch, prompts, *, thread="", before_reply=None, typed=True, staged=(), extra_rows=(),
                  text="hello there", conversation=None, message=None):
    """Admit an event, build its focus task, record its first row, then check the reply."""
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.place_reply import prepare_focus_reply

    task, content = _focus_task(root, text=text, thread=thread, staged=staged, conversation=conversation, message=message)
    if not typed:  # a task retained before the typed event existed
        task["metadata"]["presence"].pop("observed_text")
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=root)
    _initialize_owner_directives(ctx, [{"role": "user", "content": content}])
    ctx._owner_directives.extend(extra_rows)
    if before_reply:
        before_reply()
    monkeypatch.setattr("ouroboros.presence_context.confirmed_delivery_receipts", lambda _ctx: frozenset())
    assert prepare_focus_reply(ctx, "A short reply")[0]
    prompt = prompts[-1]
    bundle = json.loads(prompt.split("Host evidence (provenance is not consent):\n", 1)[1]
                        .split("\nProposed tool call:", 1)[0])
    return ctx, bundle, prompt


@pytest.mark.parametrize("status", ["SUSPICIOUS", "SAFE"])
def test_a_flagged_reply_returns_once_and_its_later_send_leaves_a_note(focus, monkeypatch, checker, status):
    """A SUSPICIOUS reply returns unsent; the mind's finish from a later model call posts it, and that decision with
    the check's reason stays in the room's record beside it, where the next round in this place reads it. A SAFE
    reply is posted at once and leaves no note."""
    from ouroboros import dialogue_store, safety
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.place_reply import _reply_digest, prepare_focus_reply, reply_carrier
    from ouroboros.places import conversation_chat_id

    def call(**kwargs):
        checker.append(kwargs["user_prompt"])
        return {"content": json.dumps({"status": status, "reason": "names a colleague the note keeps out"})}, {}
    monkeypatch.setattr(safety, "_safety_model_call", call)
    root, _place, _admission, _event = focus
    task, content = _focus_task(root)
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=root,
                          _current_llm_call_meta={"llm_call_id": "round-1"})
    _initialize_owner_directives(ctx, [{"role": "user", "content": content}])
    monkeypatch.setattr("ouroboros.presence_context.confirmed_delivery_receipts", lambda _ctx: frozenset())
    ok, message = prepare_focus_reply(ctx, "A short reply")
    assert ok is (status == "SAFE") and message.startswith("⚠️ SAFETY_RETURNED: not run;") is (status == "SUSPICIOUS")
    if status == "SUSPICIOUS":
        ctx._current_llm_call_meta = {"llm_call_id": "round-2"}
        assert prepare_focus_reply(ctx, "A short reply")[0] and len(checker) == 1
    carrier = reply_carrier(ctx.task_metadata)
    client_id = f"place-notice:{ctx.task_id}:{_reply_digest(carrier, 'A short reply')}:safety-note"
    rows = dialogue_store.rows_by_client_id(root, client_id, chat_id=conversation_chat_id(carrier["conversation_key"]))
    assert [(r.row["direction"], "names a colleague the note keeps out" in r.row["text"]) for r in rows] == (
        [("system", True)] if status == "SUSPICIOUS" else [])


def test_an_owner_answer_reaches_the_supervisor_with_the_card_it_answers(focus, monkeypatch, checker, tmp_path):
    """The Supervisor reads an owner's answer together with what it answers (audit 05.11 item 37): the row the drain
    records for a delivered answer reaches the bundle with the card beside my human's words, never inside them."""
    import queue

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message

    card = {"quiz_id": "q7", "question": "Publish exactly this reply?\n> Thanks, taking it up.", "state": "answered",
            "options": ["Publish it", "Not now"], "answered_index": 0}
    assert write_owner_message(tmp_path, "[Owner quiz answer] quiz q7", task_id="root", msg_id="qa-7",
                               kind=KIND_QUIZ_ANSWER, quiz_delivery={"task_id": "asker", "quiz_id": "q7",
                                                                     "answer_snapshot": card})
    drained = SimpleNamespace(_owner_directives=[])
    _drain_incoming_messages([], queue.Queue(), tmp_path, "root", None, set(), owner_ctx=drained)
    [answer] = drained._owner_directives
    root, _place, _admission, _event = focus
    _ctx, bundle, _prompt = _reply_bundle(root, monkeypatch, checker, extra_rows=tuple(drained._owner_directives))
    [shown] = [source for source in bundle["occasion"]["sources"] if source.get("quiz_id") == "q7"]
    assert shown == {**answer, "author": PRESS_AUTHOR}  # the press, an act of my human's, and its card
    assert answer["source"] == "owner_press"
    assert shown["content"] == "1. Publish it" and shown["reply_to"]["text"].startswith(card["question"])


def test_every_source_names_its_author_by_its_label_or_path(tmp_path):
    """Each row the Supervisor reads names its ``author`` by the label it was recorded under (``SOURCE_AUTHORS``),
    apart from the run's opening text, which is the host's or Ouroboros's (a helper's brief: its parent's) and never
    the owner's, a carried copy of a row the table would call addressed here, and a mail row without its writer's
    retained kind, which reads neutrally. A row under no known label, and a caller's raw messages, are unattributed.
    Mail rows by their kind have their own test."""
    from ouroboros.safety_evidence import SOURCE_AUTHORS, UNATTRIBUTED

    rows = [*({"source": label, "content": f"words {n}", "msg_id": f"m{n}"} for n, label in enumerate(SOURCE_AUTHORS)),
            {"source": "initial_text", "content": "Opening frame"},
            {"source": "made_up", "content": "x", "msg_id": "q"}, {"content": "no label", "msg_id": "r"}]
    ctx = SimpleNamespace(task_id="t", task_metadata={}, drive_root=tmp_path, _owner_directives=rows)
    shown = _sources(ctx)["occasion"]["sources"]
    # Assert the address-sensitive labels independently of the table that renders them.
    direct = {**SOURCE_AUTHORS, "initial_user": "the owner: the message that opened this task",
              "origin_message": ORIGIN_AUTHOR, "owner_mailbox": MAIL_AUTHOR,
              "direct_incoming": LATER_OWNER_AUTHOR, "place_confirmed": PLACE_AUTHOR, "place_owner": PLACE_AUTHOR}
    assert [row["author"] for row in shown] == [*direct.values(), OPENING, UNATTRIBUTED, UNATTRIBUTED]
    helper = SimpleNamespace(task_id="t", drive_root=tmp_path,
                             _owner_directives=[{"source": "initial_text", "content": "Do X"}],
                             task_metadata={"delegation_role": "subagent", "root_task_id": "root",
                                            "parent_task_id": "root"})
    assert _sources(helper)["occasion"]["sources"][0]["author"] == (
        "this task's brief: written by its parent task (Ouroboros) with blocks added by the host; "
        "not the owner's words")
    raw = SimpleNamespace(task_id="t", task_metadata={}, drive_root=tmp_path)
    assert _sources(raw, [{"role": "user", "content": "raw words"}, {"role": "assistant", "content": "mine"}])[
        "occasion"]["sources"] == [{"role": "user", "content": "raw words", "author": UNATTRIBUTED}]
    # Only a label naming this run as addressee or opening yields. Origin, place, card and schedule labels already
    # describe the source, not delivery to this run.
    carried = SimpleNamespace(task_id="t", task_metadata={}, drive_root=tmp_path, _owner_directives=[
        {**row, "msg_id": f"c{row['msg_id']}", "carried": True} for row in rows[:len(SOURCE_AUTHORS)]])
    assert [row["author"] for row in _sources(carried)["occasion"]["sources"]] == [
        CARRIED_AUTHOR if label in {"initial_user", "owner_mailbox", "direct_incoming"} else author
        for label, author in direct.items()]


def test_confirmed_place_words_name_the_source_conversation_not_the_receiving_chat(tmp_path):
    """A direct place event and its carried copies attest the account a message was observed from at its source, not
    the receiving room and not who wrote each of its words."""
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_words import FIELD

    reply_to = {"address": "message club-question", "author": "Club member", "text": "What colour is the poster?"}
    direct = {"source": "place_confirmed", "content": "The chess poster is blue.", "msg_id": "chat 42 / club-event",
              "conversation_id": "chess-club", "message_id": "club-event", "reply_to": reply_to, "carried": True}
    origin = SimpleNamespace(task_id="source", current_chat_id=42, drive_root=tmp_path, task_metadata={},
                             _owner_directives=[direct])
    inherited = {"source": "place_owner", "text": "The chess table is round.", "ref": "chat 42 / table-event",
                 "carrier": "ctx", "task_id": "source", "ts": "2026-10-02T09:00:00+00:00"}
    continued = SimpleNamespace(task_id="continued", current_chat_id=7, drive_root=tmp_path, task_metadata={
        "objective_author": {"kind": "continuation", "task_id": "source"}, "owner_corpus": [direct], FIELD: [inherited]})
    _initialize_owner_directives(continued, [{"role": "user", "content": "Continue the club notes."}])

    [here] = _sources(origin)["occasion"]["sources"]
    carried, helper_copy = _sources(continued)["occasion"]["sources"]
    assert carried == here  # same source and reply, even though the work is now in chat 7
    assert {key: value for key, value in here.items() if key != "author"} == direct
    assert {key: value for key, value in helper_copy.items() if key != "author"} == {
        "source": "place_owner", "content": inherited["text"], "msg_id": inherited["ref"],
        "ts": inherited["ts"], "carried": True}
    for row in (here, carried, helper_copy):
        assert row["author"] == PLACE_AUTHOR and "this chat" not in row["author"]


def test_owner_words_relayed_from_another_room_name_their_room_and_time_and_no_new_message(tmp_path):
    """A teller relays my human's permission in a club room, its later narrowing and a press there to a work that has a
    later message of its own (the real producer: quote, mail, drain). The Supervisor reads each relayed row whole, at
    the room and time it was written, with a label describing the transfer; the work's own message keeps its label,
    and the press keeps its card: the option is Ouroboros's text, the press my human's act."""
    import queue

    from ouroboros import dialogue_store
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_relay import owner_quote, write_relay

    card = {"quiz_id": "q-board", "question": "Pin the chess poster on the hall board?", "state": "answered",
            "answered_index": 0, "options": ["Pin it", "Not now"]}
    room = [{"chat_id": 42, "direction": "in", "ts": f"2026-10-0{day}T09:00:00+00:00", "client_message_id": f"club-{day}",
             "text": text} for day, text in ((1, "Both club posters may go on the hall board."),
                                             (2, "Only the chess club poster for now."))]
    room.append({**room[1], "ts": "2026-10-02T09:05:00+00:00", "client_message_id": "club-3", "text": "1. Pin it",
                 "type": "quiz_answer", "task_id": "asker", "quiz": card})
    for row in room:
        dialogue_store.append(tmp_path, "chat", row)
    write_relay(tmp_path, "receiver", relay_id="posters", note="The club posters.", chat_id=42, teller_task_id="teller",
                quotes=[owner_quote(tmp_path, row, address=format_address(row_address(row))) for row in room])
    own = {"source": "owner_mailbox", "content": "Print the chess poster in A3.", "msg_id": "row:7@2026-10-03",
           "chat_id": "7", "ts": "2026-10-03T10:00:00+00:00", "mail_kind": "owner_text"}  # as the drain keeps it
    receiver = SimpleNamespace(task_id="receiver", drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=7,
                               task_metadata={}, pending_events=[], event_queue=None, drive_logs=lambda: tmp_path / "logs",
                               _owner_directives=[dict(own)])
    _drain_incoming_messages([], queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)
    shown = _sources(receiver)["occasion"]["sources"]
    assert [(row["content"], row["author"], row["chat_id"], row["ts"]) for row in shown] == [
        (own["content"], LATER_OWNER_AUTHOR, "7", own["ts"]),
        *((row["text"], CARRIED_AUTHOR, "42", row["ts"]) for row in room[:2]), ("1. Pin it", PRESS_AUTHOR, "42", room[2]["ts"])]
    assert [row.get("carried") for row in shown] == [None, True, True, True]
    assert shown[3]["reply_to"]["author"] == "Ouroboros, task asker" and "Pin it" in shown[3]["reply_to"]["text"]


@pytest.mark.parametrize("kind", ["task", "continuation"])
def test_a_seeded_corpus_keeps_its_sources_without_claiming_delivery_to_this_run(tmp_path, kind):
    """A drafted start's or a Continue's corpus is seeded by value (``_initialize_owner_directives``): each row keeps
    its room and time. The label names a retained origin or reports the transfer, not an address to this run."""
    from ouroboros.loop_messages import _initialize_owner_directives

    corpus = [{"source": source, "content": text, "chat_id": chat, "client_message_id": client, "ts": ts}
              for source, text, chat, client, ts in (
                  ("origin_message", "Make a board for the club posters.", "1", "open", "2026-10-01T08:00:00+00:00"),
                  ("initial_user", "Plan the hall decorations.", "1", "door", "2026-09-30T08:00:00+00:00"),
                  ("owner_mailbox", "Both club posters may go up.", "42", "club-1", "2026-10-01T09:00:00+00:00"))]
    ctx = SimpleNamespace(task_id="seeded", drive_root=tmp_path, task_metadata={
        "objective_author": {"kind": kind, "task_id": "starter"}, "owner_corpus": corpus})
    _initialize_owner_directives(ctx, [{"role": "user", "content": "The work order."}])
    assert [(row["content"], row["author"], row["chat_id"], row["ts"]) for row in _sources(ctx)["occasion"]["sources"]] == [
        (corpus[0]["content"], ORIGIN_AUTHOR, "1", corpus[0]["ts"]),
        *((row["content"], CARRIED_AUTHOR, row["chat_id"], row["ts"]) for row in corpus[1:])]


def test_a_helpers_inherited_and_held_rows_read_as_carried_and_its_own_message_keeps_its_label(tmp_path):
    """An inherited or held row's label describes this copy's path without denying its original reply target.
    Direct words keep their label, cards stay whole, and identities gain no transfer marker from the projection."""
    from ouroboros.answer_facts import press_record
    from ouroboros.owner_words import governing_words_for_schedule, safety_sources

    comment_card = {"address": "quiz q-posters of task arranger", "author": "Ouroboros, task arranger",
                    "text": "Which club posters can go up?"}
    card = {"address": "quiz q-board of task asker", "author": "Ouroboros, task asker",
            "text": "Pin the chess poster?\n1. Pin it\n2. Not now\n(host record: my human pressed option 1)"}
    parent = SimpleNamespace(task_id="root", drive_root=tmp_path, task_metadata={}, _owner_directives=[
        {"source": "owner_mailbox", "content": "Both club posters may go up.", "msg_id": "relayed-1",
         "carried": True, "reply_to": comment_card},
        {"source": "owner_press", "content": "1. Pin it", "quiz_id": "q-board", "source_task_id": "asker",
         "msg_id": "press-1", "reply_to": card}])
    helper = SimpleNamespace(task_id="helper", drive_root=tmp_path, task_metadata={
        "delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root",
        **governing_words_for_schedule(parent)}, _owner_directives=[
        {"source": "initial_text", "content": "Pin the poster."},
        {"source": "owner_mailbox", "content": "Print it in A3.", "msg_id": "mail-7", "ts": "2026-10-03T10:00:00+00:00",
         "mail_kind": "owner_text"}],
        _held_owner_messages={"row:42@held": {"direction": "in", "chat_id": 42, "client_message_id": "club-2",
                                              "text": "Only the chess poster.", "carried": {"went_to": ""}}})
    shown = _sources(helper)["occasion"]["sources"][1:]
    assert [(row["content"], row["author"], row.get("carried"), "ts" in row) for row in shown] == [
        ("Print it in A3.", LATER_OWNER_AUTHOR, None, True),
        ("Both club posters may go up.", CARRIED_AUTHOR, True, False),
        (press_record("1. Pin it", "asker"), PRESS_AUTHOR, True, False),
        ("Only the chess poster.", CARRIED_AUTHOR, True, False)]
    assert shown[1]["reply_to"] == comment_card and shown[2]["reply_to"] == card
    assert not any(row.get("carried") for row in safety_sources(helper, _private=True))


def test_mail_rows_read_by_their_writers_kind_and_copies_as_copies(tmp_path):
    """The asker's mail holds a message written before kinds were kept, the host's notice that my human closed a card
    (its writer's typed closure) and an ordinary message of theirs; the real drain keeps each writer's kind beside its
    row (``mail_kind``) and never takes its own routing default for a writer's: the oldest row reads neutrally. The
    asker's Supervisor reads the notice as the host's, which chose nothing, and the message as my human's. A Continue's
    corpus copy and a scheduled helper's field copy keep the trusted writer kind, so the closure stays the host's notice
    while ordinary mail reads as a carried record, and a row that never kept a kind gains none."""
    import json
    import queue

    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_QUIZ_CLOSED, _mailbox_path, write_owner_message
    from ouroboros.owner_words import carried_owner_corpus, governing_words_for_schedule

    notice = ("[Owner question closed without answer] quiz q-ink — closed 2026-10-05T10:00:00+00:00. Reason: Closed by "
              "the owner without an answer.\nThis closes the question; no option was chosen and no approval was given.")
    words = ["Use the amber ink for the poster.", notice, "Use the teal ink for the flyer."]
    mailbox = _mailbox_path(tmp_path, "asker")
    mailbox.parent.mkdir(parents=True, exist_ok=True)
    mailbox.write_text(json.dumps({"msg_id": "mail-1", "text": words[0], "ts": "2026-10-04T09:00:00+00:00"}) + "\n",
                       encoding="utf-8")  # its writer kept no kind
    assert write_owner_message(tmp_path, words[1], "asker", msg_id="mail-2", kind=KIND_QUIZ_CLOSED)
    assert write_owner_message(tmp_path, words[2], "asker", msg_id="mail-3")
    asker = SimpleNamespace(task_id="asker", drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=1,
                            task_metadata={}, pending_events=[], event_queue=None, drive_logs=lambda: tmp_path / "logs",
                            _owner_directives=[])
    _drain_incoming_messages([], queue.Queue(), tmp_path, "asker", None, set(), owner_ctx=asker)
    continued = SimpleNamespace(task_id="continued", drive_root=tmp_path, task_metadata={
        "objective_author": {"kind": "continuation", "task_id": "asker"}, "owner_corpus": carried_owner_corpus(asker)})
    _initialize_owner_directives(continued, [{"role": "user", "content": "Continue the ink order."}])
    helper = SimpleNamespace(task_id="helper", drive_root=tmp_path, task_metadata={
        "delegation_role": "subagent", "root_task_id": "asker", "parent_task_id": "asker",
        **governing_words_for_schedule(asker)}, _owner_directives=[{"source": "initial_text", "content": "Order it."}])
    assert [(row["content"], row.get("mail_kind"), row["author"]) for row in _sources(asker)["occasion"]["sources"]] == [
        (words[0], None, MAIL_AUTHOR), (words[1], "quiz_closed", CLOSED_AUTHOR),
        (words[2], "owner_text", LATER_OWNER_AUTHOR)]
    copies = [(words[0], None, CARRIED_AUTHOR), (words[1], "quiz_closed", CLOSED_AUTHOR),
              (words[2], "owner_text", CARRIED_AUTHOR)]
    assert [(row["content"], row.get("mail_kind"), row["author"])
            for row in _sources(continued)["occasion"]["sources"]] == copies
    assert [(row["content"], row.get("mail_kind"), row["author"])
            for row in _sources(helper)["occasion"]["sources"][1:]] == copies

    # The helper's field: each row under its mailbox id and receipt, its writer's kind beside it. A field written
    # before rows carried that kind holds the same row without it, so the identity a member is found by and the bytes
    # its quote is rebuilt from stay the same; the closure is stamped as the host's notice, found by its reference.
    from ouroboros.owner_mailbox import mailbox_lines
    from ouroboros.owner_relay import _member_identity, _pristine_form, _words_identity, stamped_refs
    from ouroboros.owner_words import FIELD, words_stamp
    from ouroboros.safety_evidence import CLOSED_STAMP

    mailed = {row["msg_id"]: row["ts"] for row in map(json.loads, mailbox_lines(mailbox.read_text(encoding="utf-8")))}
    field = helper.task_metadata[FIELD]
    assert [(row["ref"], row["ts"], row.get("mail_kind")) for row in field] == [
        ("msg mail-1", "received 2026-10-04T09:00:00+00:00", None), ("msg mail-2", f"received {mailed['mail-2']}", "quiz_closed"),
        ("msg mail-3", f"received {mailed['mail-3']}", "owner_text")]
    for row in field:
        legacy = {key: value for key, value in row.items() if key != "mail_kind"}
        assert _member_identity(row) == _member_identity(legacy) and _words_identity(row) == _words_identity(legacy)
        assert _pristine_form(helper, row) == _pristine_form(helper, legacy)
    stamps = [words_stamp(row) for row in field]
    assert CLOSED_STAMP in stamps[1] and " · owner · " not in stamps[1]
    assert [CLOSED_STAMP in stamp for stamp in stamps] == [False, True, False]
    assert stamped_refs("\n".join(stamps)) == {"msg mail-1", "msg mail-2", "msg mail-3"}


def test_each_inherited_submission_of_one_card_keeps_that_card_and_its_own_values(tmp_path):
    """My human submits one form and presses one button. The inherited field keeps the card's body once, beside the
    first submission (``_deduped``); a helper's Supervisor reads each submission with its own fields and the card body
    as a reference snapshot (a living card's address does not pin its revision); the direct rows keep the body as-is."""
    from ouroboros.owner_relay import owner_row_identity
    from ouroboros.owner_words import FIELD, REPLY_TARGET, governing_words_for_schedule

    body = {"address": "card c-buttons of task designer", "author": "Ouroboros, task designer",
            "text": "Button colours for the stationery form\nSave", "pressed_label": "Save", "card_id": "c-buttons"}
    body_two = {**body, "text": "Button colours for the large-print form\nSave"}
    submissions = [{"source": "owner_mailbox", "content": "blue", "msg_id": "card_input:c-buttons:1",
                    "received": "2026-10-01T09:00:00+00:00", "source_task_id": "designer",
                    "reply_to": {**body, "submitted_values": {"colour": "blue"}}},
                   {"source": "owner_mailbox", "content": "Save", "msg_id": "card_input:c-buttons:2",
                    "received": "2026-10-02T09:00:00+00:00", "source_task_id": "designer", "reply_to": dict(body_two)}]
    parent = SimpleNamespace(task_id="root", drive_root=tmp_path, task_metadata={},
                             _owner_directives=[dict(row) for row in submissions])
    helper = SimpleNamespace(task_id="helper", drive_root=tmp_path, task_metadata={
        "delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root",
        **governing_words_for_schedule(parent)}, _owner_directives=[{"source": "initial_text", "content": "Draw."}])
    assert [row["source"] for row in helper.task_metadata[FIELD]] == ["owner_mailbox", REPLY_TARGET, "owner_mailbox"]
    direct, inherited = _sources(parent)["occasion"]["sources"], _sources(helper)["occasion"]["sources"][1:]
    assert [row["reply_to"] for row in direct] == [row["reply_to"] for row in submissions]
    assert [row["reply_to"] for row in inherited] == [{**submissions[0]["reply_to"], "note": CARD_NOTE},
                                                      {**body, "note": CARD_NOTE}]
    assert [(row["content"], row["ts"], row["author"]) for row in inherited] == [
        (row["content"], f"received {row['received']}", CARRIED_AUTHOR) for row in submissions]
    edge = SimpleNamespace(task_id="edge", drive_root=tmp_path, task_metadata={FIELD: [
        {"text": "Save", "source": "owner_mailbox", "ref": "card_input:c-buttons:9", "carrier": "field",
         "answer_input": {"reply_address": body["address"], "card_id": "c-buttons", "pressed_label": "Save",
                          "submitted_values": {"colour": "blue"}}}]}, _owner_directives=[])
    [shown] = _sources(edge)["occasion"]["sources"]
    assert shown["reply_to"] == {"address": body["address"], "author": "", "text": "", "card_id": "c-buttons",
                                 "pressed_label": "Save", "submitted_values": {"colour": "blue"}}
    held = {"direction": "in", "chat_id": 42, "client_message_id": "club-9", "ts": "2026-10-01T09:00:00+00:00",
            "text": "Save", "carried": {"went_to": ""},
            "corpus_answer": {"source": "owner_mailbox", "reply_to": dict(body)}}
    edge.task_metadata[FIELD][0]["row_sha256"] = owner_row_identity(held)["row_sha256"]
    edge._held_owner_messages = {"held club-9": held}
    [shown] = _sources(edge)["occasion"]["sources"]
    assert shown["reply_to"] == {**body, "submitted_values": {"colour": "blue"}, "note": CARD_NOTE}


def test_an_answer_carried_after_a_helpers_start_keeps_its_card(tmp_path):
    """My human presses an option after a helper started. The host carries the press down as a quote rebuilt from
    the corpus row (``owner_relay._pristine_form``), with the card it answers; the helper's Supervisor reads the press
    beside that card, as its parent's does."""
    import queue

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_relay import _carry_to_member, _rows_to_carry
    from ouroboros.tools.registry import ToolContext

    card = {"address": "quiz q-diagram of task drafter", "author": "Ouroboros, task drafter",
            "text": "Which diagram goes on the cover?\n1. The flow chart\n2. The timeline\n"
                    "(host record: my human pressed option 1)"}
    parent = SimpleNamespace(task_id="root", drive_root=tmp_path, task_metadata={}, current_chat_id=1,
                             _held_owner_messages={}, _owner_directives=[{
                                 "source": "owner_press", "content": "1. The flow chart", "msg_id": "qa-diagram",
                                 "source_task_id": "drafter", "quiz_id": "q-diagram",
                                 "received": "2026-10-06T08:00:00+00:00", "reply_to": card}])
    helper = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_task_type="task")
    helper.task_id, helper.event_queue = "helper", queue.Queue()
    helper._owner_directives = [{"source": "initial_text", "content": "Draw the cover."}]
    helper.task_metadata = {"delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root"}
    _carry_to_member(tmp_path, tmp_path, "helper", {"metadata": helper.task_metadata}, _rows_to_carry(parent, tmp_path),
                     "root")
    _drain_incoming_messages([], queue.Queue(), tmp_path, "helper", None, set(), owner_ctx=helper)
    [(address, held)] = helper._held_owner_messages.items()
    assert held["corpus_answer"]["reply_to"] == card and "reply_to" not in held  # the card rides in the rebuilt answer
    [shown] = _sources(helper)["occasion"]["sources"][1:]
    assert (shown["source"], shown["content"], shown["msg_id"], shown["author"]) == (
        "owner_press", "1. The flow chart", address, PRESS_AUTHOR)
    assert shown["reply_to"] == card == _sources(parent)["occasion"]["sources"][0]["reply_to"]


def test_a_rebuilt_card_answer_names_its_card_a_reference_in_every_record_it_reaches(tmp_path):
    """My human submits a living card's form, then presses another of its buttons. A helper's field keeps the card's
    body once, at its address, beside the form, so the body the press is joined to ends with the form's label. The host
    rebuilds each answer from that field and a relay carries it on (``owner_relay._pristine_form``; the real producer:
    quote, mail, drain). Each keeps its own label and values as submitted, an empty value too, and names that body a
    reference to the card's address in every record it reaches: the helper's Supervisor, the receiver's corpus, a start
    drafted from it and a Continue that reads the relay from its mail."""
    import queue

    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.owner_relay import _rows_to_carry, write_relay
    from ouroboros.owner_words import carried_owner_corpus, governing_words_for_schedule

    body = {"address": "card c-ink of task designer", "author": "Ouroboros, task designer", "card_id": "c-ink",
            "text": "Ink for the club flyer\nApply"}
    values = {"ink": "teal", "size": ""}
    parent = SimpleNamespace(task_id="root", drive_root=tmp_path, task_metadata={}, _owner_directives=[
        {"source": "owner_mailbox", "content": "teal", "msg_id": "mail-ink-1", "source_task_id": "designer",
         "received": "2026-10-07T09:00:00+00:00", "reply_to": {**body, "pressed_label": "Apply", "submitted_values": values}},
        {"source": "owner_press", "content": "Reset", "msg_id": "mail-ink-2", "source_task_id": "designer",
         "received": "2026-10-07T09:05:00+00:00",
         "reply_to": {**body, "text": "Ink for the club flyer\nReset", "pressed_label": "Reset"}}])
    tree = {"delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root"}
    helper = SimpleNamespace(task_id="helper", drive_root=tmp_path, current_chat_id=1, _held_owner_messages={},
                             _owner_directives=[], task_metadata={**tree, **governing_words_for_schedule(parent)})
    quotes = [quote for _ids, _address, quote, _went, _before in _rows_to_carry(helper, tmp_path)]
    write_relay(tmp_path, "receiver", relay_id="inks", note="The flyer inks.", chat_id=1, teller_task_id="helper",
                quotes=quotes)
    receiver = SimpleNamespace(task_id="receiver", drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=1,
                               task_metadata={}, pending_events=[], event_queue=None, drive_logs=lambda: tmp_path / "logs",
                               _owner_directives=[])
    frames = []
    _drain_incoming_messages(frames, queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)
    # Where the quote shows the card's body, it is framed as a reference to the card's address, not as the card that
    # stood when my human pressed.
    shown = [row["content"] for row in frames if row.get("_host_context_kind") == "owner_words"]
    choices = [
        "(host record: my human pressed «Apply»; the label is the text of task designer)",
        "(host record: my human pressed «Reset» and wrote no words; the label is the text of task designer)",
    ]
    assert [(text.count("[Host question frame]"), text.count(f"[Host note: {CARD_NOTE}]")) for text in shown] == [(1, 1)] * 2
    assert [quote["question_frame"].endswith(f"\n[Host note: {CARD_NOTE}]\n{choice}")
            for quote, choice in zip(quotes, choices)] == [True, True]
    assert [text.endswith(f"\n[Host note: {CARD_NOTE}]\n{choice}")
            for text, choice in zip(shown, choices)] == [True, True]
    seeded = [("task", carried_owner_corpus(receiver)),
              ("continuation", owner_corpus_rows(owner_sources(tmp_path, {"origin_message_text": "Ink it"}, "receiver")))]
    works = [helper, receiver]
    for kind, corpus in seeded:
        works.append(SimpleNamespace(task_id=f"{kind}-of-receiver", drive_root=tmp_path, task_metadata={
            "objective_author": {"kind": kind, "task_id": "receiver"}, "owner_corpus": corpus}))
        _initialize_owner_directives(works[-1], [{"role": "user", "content": "Ink the flyer."}])
    cards = [{**body, "pressed_label": "Apply", "submitted_values": values, "note": CARD_NOTE},
             {**body, "pressed_label": "Reset", "note": CARD_NOTE}]  # the press's own label beside the form's body
    for work in works:
        assert [row["reply_to"] for row in _sources(work)["occasion"]["sources"] if "reply_to" in row] == cards
    # Direct snapshots stay unannotated; a pre-update saved carried directive is qualified only by a matching
    # already-delivered reconstructed row in _held_owner_messages (never a went_to context row or missing source).
    assert all("note" not in row["reply_to"] for row in _sources(parent)["occasion"]["sources"])
    pre_update = [{**row, "reply_to": {k: v for k, v in row["reply_to"].items() if k != "note"}}
                  for row in receiver._owner_directives]
    saved = SimpleNamespace(task_id="saved", drive_root=tmp_path, task_metadata={},
                            _owner_directives=[dict(r) for r in pre_update],
                            _held_owner_messages=dict(receiver._held_owner_messages))
    assert [row["reply_to"] for row in _sources(saved)["occasion"]["sources"]] == cards
    # Work the saved task drafts keeps that proof by value (``carried_owner_corpus``); the saved rows stay as they are.
    drafted = SimpleNamespace(task_id="drafted", drive_root=tmp_path, task_metadata={
        "objective_author": {"kind": "task", "task_id": "saved"}, "owner_corpus": carried_owner_corpus(saved)})
    _initialize_owner_directives(drafted, [{"role": "user", "content": "Ink the flyer."}])
    assert [row["reply_to"] for row in _sources(drafted)["occasion"]["sources"]] == cards
    assert all("note" not in row["reply_to"] for row in saved._owner_directives)
    context_only = SimpleNamespace(task_id="ctx-only", drive_root=tmp_path, task_metadata={},
                                   _owner_directives=[dict(r) for r in pre_update],
                                   _held_owner_messages={k: {**v, "carried": {"went_to": "other"}}
                                                         for k, v in receiver._held_owner_messages.items()})
    assert all("note" not in row["reply_to"] for row in _sources(context_only)["occasion"]["sources"])
    assert all("note" not in row["reply_to"] for row in [*carried_owner_corpus(context_only), *carried_owner_corpus(parent)])


def test_a_rebuilt_answer_keeps_its_earlier_form_and_address_for_a_card_with_no_body_or_empty_fields(tmp_path):
    """A rebuilt quote's address is the hash of the row the host builds, so earlier work holds each answer under what it
    built before. The answers to a paper card recorded with its body, without it (never kept) and with empty fields
    rebuild to exactly those rows: the card rides only where the field kept its body, and a field that was empty is not
    carried. Each reader then shows what its own record holds: the helper's field, the card's address and the fields it
    kept; the receiver's quote, its body. Where the quote holds no card the gap stays; it is never rebuilt."""
    import queue

    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_relay import _rows_to_carry, write_relay
    from ouroboros.owner_words import FIELD, REPLY_TARGET, governing_words_for_schedule

    def card(name, **recorded):  # one paper card's reference as the corpus recorded it
        return {"address": f"card c-{name} of task designer", "author": "Ouroboros, task designer",
                "card_id": f"c-{name}", **recorded}

    body = "Paper for the club flyer\nApply"
    full = card("full", text=body, pressed_label="Apply", submitted_values={"paper": "recycled"})
    bare = card("bare", text="", pressed_label="Apply", submitted_values={"paper": "kraft"})  # its body was never kept
    blank = card("blank", text=body, pressed_label="", submitted_values={})  # its fields were empty
    answers = {"recycled": full, "kraft": bare, "plain": blank}
    received = {words: f"2026-10-09T09:0{n}:00+00:00" for n, words in enumerate(answers)}
    parent = SimpleNamespace(task_id="root", drive_root=tmp_path, task_metadata={}, _owner_directives=[
        {"source": "owner_mailbox", "content": words, "msg_id": f"mail-paper-{n}", "source_task_id": "designer",
         "received": received[words], "reply_to": dict(reply)} for n, (words, reply) in enumerate(answers.items())])
    tree = {"delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root"}
    helper = SimpleNamespace(task_id="helper", drive_root=tmp_path, current_chat_id=1, _held_owner_messages={},
                             _owner_directives=[], task_metadata={**tree, **governing_words_for_schedule(parent)})
    # The exact earlier forms: the field's own record of each answer, and the card each rebuilt row carries.
    assert [row["answer_input"] for row in helper.task_metadata[FIELD] if row["source"] != REPLY_TARGET] == [
        {"reply_address": full["address"], "source_task_id": "designer", "card_id": "c-full", "pressed_label": "Apply",
         "submitted_values": {"paper": "recycled"}},
        {"reply_address": bare["address"], "source_task_id": "designer", "card_id": "c-bare", "pressed_label": "Apply",
         "submitted_values": {"paper": "kraft"}},
        {"reply_address": blank["address"], "source_task_id": "designer", "card_id": "c-blank"}]
    rebuilt = {"recycled": dict(full), "kraft": None,
               "plain": {"address": blank["address"], "author": blank["author"], "text": body, "card_id": "c-blank"}}
    carried = _rows_to_carry(helper, tmp_path)
    assert len(carried) == len(answers)
    for (_ids, address, quote, _went, _before), words in zip(carried, answers):
        answer = {"source": "owner_mailbox", "source_task_id": "designer",
                  **({"reply_to": rebuilt[words]} if rebuilt[words] else {})}
        earlier = {"direction": "in", "chat_id": 1, "ts": f"received {received[words]}", "text": words,
                   "corpus_answer": answer}
        assert quote["owner_row"]["corpus_answer"] == answer, words
        assert address == format_address(row_address(earlier)), words
    write_relay(tmp_path, "receiver", relay_id="paper", note="The paper cards.", chat_id=1, teller_task_id="helper",
                quotes=[quote for _ids, _address, quote, _went, _before in carried])
    receiver = SimpleNamespace(task_id="receiver", drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=1,
                               task_metadata={}, pending_events=[], event_queue=None, drive_logs=lambda: tmp_path / "logs",
                               _owner_directives=[])
    _drain_incoming_messages([], queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)

    def shown(work):  # the card each answer is read with, by its words (none: the row shows no card)
        return {row["content"]: row.get("reply_to") for row in _sources(work)["occasion"]["sources"]}

    note = {"note": CARD_NOTE}
    assert shown(parent) == answers  # the direct rows hold each card as recorded, empty fields too
    # The field kept `kraft`'s address and fields but never its body, which stays absent, not rebuilt; the fields `plain`
    # had empty were never carried, so none is shown.
    assert shown(helper) == {"recycled": {**full, **note}, "kraft": {**bare, "author": ""},
                             "plain": {**rebuilt["plain"], **note}}
    assert shown(receiver) == {"recycled": {**full, **note}, "kraft": None, "plain": {**rebuilt["plain"], **note}}


def test_a_quote_frames_a_card_by_what_it_holds_a_snapshot_a_quiz_a_reference_or_nothing():
    """A living submission's own card is the snapshot it was pressed on and a quiz card is one record: both stand as the
    question frame, unannotated. A card a rebuilt answer rejoined by its address is a reference and says so beside its
    body, which may carry another press's label, so this press's own choice stands apart from it; one the answer holds
    no body for has no frame at all. Whichever card it is read with, the fields the submission held stand once beside
    its words, by name: an empty one and a line break inside a value stay."""
    from ouroboros.answer_facts import CARD_REFERENCE_NOTE
    from ouroboros.owner_relay import _quote_body

    living = {"address": "card c-ink of task designer", "author": "Ouroboros, task designer", "text": "Ink\nApply",
              "card_id": "c-ink", "pressed_label": "Apply"}
    quiz = {"address": "quiz q-ink of task designer", "author": "Ouroboros, task designer", "text": "Which ink?\n1. Teal"}

    def rebuilt(card, source="owner_press", words="Apply"):  # a quote rebuilt from the corpus, its card by address or none
        return _quote_body({"direction": "in", "chat_id": 1, "ts": "2026-10-08T09:00:00+00:00", "text": words,
                            "corpus_answer": {"source": source, "source_task_id": "designer",
                                              **({"reply_to": card} if card else {})}}, "[Owner card press]")

    frame = "[Host question frame]\nOuroboros, task designer: "
    snapshot = _quote_body({}, "[Owner card press]", {"source": "owner_press", "content": "Apply",
                                                      "source_task_id": "designer", "reply_to": living})
    assert snapshot.endswith(frame + "Ink\nApply")  # the card as it stood when pressed: no note
    assert rebuilt(quiz).endswith(frame + quiz["text"])  # a quiz card is one record: no note
    assert rebuilt(living).endswith(
        frame + "Ink\nApply\n[Host note: " + CARD_REFERENCE_NOTE
        + "]\n(host record: my human pressed «Apply» and wrote no words; the label is the text of task designer)")
    assert rebuilt({**living, "pressed_label": "Reset"}).endswith(
        frame + "Ink\nApply\n[Host note: " + CARD_REFERENCE_NOTE
        + "]\n(host record: my human pressed «Reset» and wrote no words; the label is the text of task designer)")
    assert "[Host question frame]" not in rebuilt({**living, "author": "", "text": ""})  # its address and fields only
    assert "[Host question frame]" not in rebuilt(None)

    def held(body):  # the field records a model reads in a quote, parsed back from their JSON
        head = "[Host record: submitted fields (JSON): "
        return [json.loads(line[len(head):-1]) for line in body.split("\n") if line.startswith(head)]

    values, typed = {"ink": "teal", "size": "", "note": "line one\nline two"}, "teal\nline one\nline two"
    captured = _quote_body({}, "[Owner card input]", {"source": "owner_mailbox", "content": typed, "source_task_id": "designer",
                                                      "reply_to": {**living, "submitted_values": values}})
    assert held(captured) == [values] and "[Host note:" not in captured  # a snapshot: its label is this press's own
    assert captured.index(typed) < captured.index("[Host record: ") < captured.index("[Host question frame]")
    assert captured.endswith(frame + "Ink\nApply")
    # A reference's body carries another press's label (Save): the choice and the fields are this submission's own.
    reconstructed = rebuilt({**living, "text": "Ink\nSave", "submitted_values": values}, "owner_mailbox", typed)
    assert held(reconstructed) == [values]
    assert reconstructed.endswith(
        frame + "Ink\nSave\n[Host note: " + CARD_REFERENCE_NOTE
        + "]\n(host record: my human pressed «Apply»; the label is the text of task designer)")
    blank = {"ink": "", "size": ""}  # every field left empty: a press, still naming the fields it left
    press = {"source": "owner_press", "content": "Apply", "source_task_id": "designer"}
    assert held(_quote_body({}, "[Owner card press]", {**press, "reply_to": {**living, "submitted_values": blank}})) == [blank]
    assert held(rebuilt({**living, "submitted_values": blank})) == [blank]
    assert held(rebuilt({**living, "author": "", "text": "", "submitted_values": values}, "owner_mailbox", typed)) == [values]


def test_a_mail_kind_the_host_knows_stays_with_its_row_and_one_it_never_kept_is_never_guessed(tmp_path):
    """A closure notice reads as the host's wherever the row still holds its writer's kind, a copy of an ordinary message
    reads as a copy (never as a message addressed to this work), and a row recorded before kinds were kept reads
    neutrally even when its id and words look like a closure notice: the gap is named, the kind is not rebuilt."""
    notice = ("[Owner question closed without answer] quiz q-old — closed 2026-10-05T10:00:00+00:00. Reason: Closed by "
              "the owner without an answer.\nThis closes the question; no option was chosen and no approval was given.")
    legacy = {"source": "owner_mailbox", "content": notice, "msg_id": "quiz_closed:q-old"}
    closed = {**legacy, "msg_id": "quiz_closed:q-new", "mail_kind": "quiz_closed"}
    ordinary = {"source": "owner_mailbox", "content": "Use the teal ink.", "msg_id": "mail-9", "mail_kind": "owner_text"}
    direct = SimpleNamespace(task_id="t", task_metadata={}, drive_root=tmp_path,
                             _owner_directives=[dict(row) for row in (legacy, closed, ordinary)])
    copies = SimpleNamespace(task_id="c", task_metadata={}, drive_root=tmp_path, _owner_directives=[
        {**row, "msg_id": f"c-{row['msg_id']}", "carried": True} for row in (legacy, closed, ordinary)])
    assert [row["author"] for row in _sources(direct)["occasion"]["sources"]] == [
        MAIL_AUTHOR, CLOSED_AUTHOR, LATER_OWNER_AUTHOR]
    assert [row["author"] for row in _sources(copies)["occasion"]["sources"]] == [
        CARRIED_AUTHOR, CLOSED_AUTHOR, CARRIED_AUTHOR]


def test_one_source_carried_twice_is_sent_once_whole_with_its_time_beside_one_now(checker, tmp_path):
    """One message the corpus carries under two reference formats (its room and client id; the ``chat N / id`` a carrier
    printed as its msg_id) is sent once, its words never clipped; rows that name no message stay apart even in the
    same words. Owner rows carry the time recorded for them, and the bundle one ``now``."""
    from datetime import datetime

    long_words = "".join(f"Item {n}: split it into projects. " for n in range(900))
    rows = [{"source": "origin_message", "content": long_words, "chat_id": "c-1", "client_message_id": "msg-1",
             "carried": True},
            {"source": "initial_user", "content": long_words, "msg_id": "chat c-1 / msg-1", "carried": True},
            {"source": "owner_mailbox", "content": "yes", "msg_id": "row:1@a", "ts": "2026-11-09T12:49:09+00:00"},
            {"source": "initial_text", "content": "yes"}, {"source": "initial_text", "content": "yes"}]
    ctx = SimpleNamespace(task_id="t", task_metadata={}, drive_root=tmp_path, _owner_directives=rows)
    bundle = _sources(ctx)
    shown = bundle["occasion"]["sources"]
    assert [row["source"] for row in shown] == ["origin_message", "owner_mailbox", "initial_text", "initial_text"]
    assert shown[0]["content"] == long_words and json.dumps(bundle, ensure_ascii=False).count(long_words) == 1
    assert shown[1]["ts"] == "2026-11-09T12:49:09+00:00" and not any("now" in row for row in shown)
    now = bundle["work"]["now"]  # the rows' form: whole seconds and a +00:00 offset
    assert datetime.fromisoformat(now).microsecond == 0 and len(now) == len(shown[1]["ts"]) and now.endswith("+00:00")


def test_a_helper_reads_the_words_carried_to_its_work_and_never_a_siblings(checker, tmp_path):
    """A helper's Supervisor reads my human's later words the host carried to its work, shaped as the corpus rows it
    already reads and with the card they answer; a row held as context because it went to a sibling stays out. A
    helper scheduled after those words arrived holds them only in its inherited field, and its Supervisor reads
    them from there, with the same card; a quote its corpus already holds is listed once, with the card and time
    only its other copy carried."""
    import queue

    from ouroboros import dialogue_store, safety
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_relay import owner_quote, write_relay
    from ouroboros.owner_words import FIELD, governing_words_for_schedule
    from ouroboros.tools.registry import ToolContext

    def sources_of(ctx):
        prompt = safety._build_check_prompt("run_command", {"cmd": ["echo", "hello"]}, ctx=ctx)
        return prompt, json.loads(prompt.split("Host evidence (provenance is not consent):\n", 1)[1]
                                  .split("\nProposed tool call:", 1)[0])["occasion"]["sources"]

    card = {"quiz_id": "q7", "question": "Publish exactly this reply?", "state": "answered", "answered_index": 0,
            "options": ["Publish it", "Not now"], "answered_after_terminal": True}
    answer = {"chat_id": 1, "direction": "in", "ts": "2026-10-04T06:00:00+00:00", "client_message_id": "owner-answer",
              "text": "1. Publish it", "late_answer": {"task_id": "asker", "quiz_id": "q7",
                                                       "answer_snapshot": {"task_id": "asker", "quiz": card}}}
    elsewhere = {"chat_id": 1, "direction": "in", "ts": "2026-10-04T06:01:00+00:00",
                 "client_message_id": "for-sibling", "text": "The sibling should use the March figures."}
    helper = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_task_type="task")
    helper.task_id, helper.event_queue, helper._owner_directives = "helper", queue.Queue(), []
    helper.task_metadata = {"delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root"}
    addresses = {}
    for row, went_to in ((answer, ""), (elsewhere, "sibling")):
        dialogue_store.append(tmp_path, "chat", row)
        addresses[went_to] = address = format_address(row_address(row))
        write_relay(tmp_path, "helper", relay_id=f"{'carry-context' if went_to else 'carry'}:{address}", note="",
                    quotes=[owner_quote(tmp_path, row, address=address)], chat_id=1, teller_task_id="root",
                    canonical_root=tmp_path, host_carried={"went_to": went_to})
    _drain_incoming_messages([], queue.Queue(), tmp_path, "helper", None, set(), owner_ctx=helper)
    assert not helper._owner_directives  # carried words stay outside the helper's corpus
    assert {key: row["carried"]["went_to"] for key, row in helper._held_owner_messages.items()} == {
        addresses[""]: "", addresses["sibling"]: "sibling"}
    prompt, sources = sources_of(helper)
    card_reply = {"address": "quiz q7 of task asker", "author": "Ouroboros, task asker",
                  "text": "Publish exactly this reply?\n1. Publish it\n2. Not now\n(host record: my human pressed option 1)"}
    # A press: my human's act, no words; this copy came carried, whatever the row's first addressee.
    later_words = {"author": PRESS_AUTHOR, "ts": "2026-10-04T06:00:00+00:00", "carried": True}
    assert sources == [{"source": "owner_press", "content": "1. Publish it", "msg_id": addresses[""],
                        "reply_to": card_reply, **later_words}]
    assert addresses["sibling"] not in prompt and "March figures" not in prompt
    corpus = {"source": "owner_press", "content": "1. Publish it", "msg_id": addresses[""], "chat_id": "1",
              "client_message_id": "owner-answer", "carried": True}
    helper._owner_directives = [dict(corpus)]  # its corpus copy, without the card and time its held row has
    assert sources_of(helper)[1] == [{**corpus, "reply_to": card_reply, **later_words}]
    assert helper._owner_directives == [corpus]
    helper._owner_directives = []
    # The helper schedules a helper of its own: the answer rides in that helper's field, and its Supervisor reads it.
    later = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_task_type="task")
    later.task_id, later._owner_directives = "later", [{"source": "initial_text", "content": "Ship the release notes."}]
    later.task_metadata = {"delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "helper",
                           **governing_words_for_schedule(helper)}
    assert [row["ref"] for row in later.task_metadata[FIELD]] == ["chat 1 / owner-answer", "quiz q7 of task asker"]
    prompt, sources = sources_of(later)
    brief = ("this task's brief: written by its parent task (Ouroboros) with blocks added by the host; "
             "not the owner's words")
    record = ("(host record: my human pressed «1. Publish it» and wrote no words; the label is the text of task "
              "asker)")  # the field carries the host's record of the act, never the label as their words
    assert sources == [{"source": "initial_text", "content": "Ship the release notes.", "author": brief},
                       {"source": "owner_press", "content": record, "msg_id": "chat 1 / owner-answer",
                        "reply_to": card_reply, **later_words}]
    assert addresses["sibling"] not in prompt and "March figures" not in prompt
    later._owner_directives.append(dict(corpus))
    sources = sources_of(later)[1]
    assert sources[1:] == [{**corpus, "reply_to": card_reply, **later_words}], sources  # the corpus row, once, whole
    assert later._owner_directives[-1] == corpus  # the context's own row is read, never changed


def test_each_source_projection_appears_once_and_the_event_is_typed(focus, monkeypatch, checker):
    root, place, _admission, _event = focus
    p.edit_place(root, place.place_id, "add_note", author="owner", text="OWNER NOTE: reply in Russian here.")
    p.edit_place(root, place.place_id, "add_note", author="ouroboros", text="MIND NOTE: the person likes brevity.")
    ctx, bundle, prompt = _reply_bundle(root, monkeypatch, checker)

    retained = ctx._owner_directives[0]
    assert retained["source"] == "initial_text" and "counted_roots" in retained["content"]
    source = bundle["occasion"]["sources"][0]
    assert source["message_text"] == "hello there" and source["event"]["conversation_key"] == "chat:a:r:0"
    assert source["author"] == ("a chat event that woke this turn: event.actor wrote message_text; "
                                "the other keys are host facts")
    assert source["work_here"] == "Already started: task1" and "content" not in source
    assert "notes_at_event" not in source  # the same note versions are read now
    for text in ("OWNER NOTE", "MIND NOTE", RULES.splitlines()[0], "hello there"):
        assert prompt.count(text) == 1, text
    for frame_only in ("counted_roots", "root-a", "Place facts (context only)"):
        assert frame_only not in prompt
    notes = bundle["standing_delegation"]["place_notes"]
    assert [(n["author"], n["standing_delegation"], bool(n["at"])) for n in notes] == [
        ("owner", True, True), ("ouroboros", False, True)]
    row = bundle["subject"]["place"]["places"][0]
    assert row["place_id"] == place.place_id and "notes" not in row and "place_id" not in bundle["subject"]["place"]
    assert bundle["standing_delegation"]["owner_rules"] == RULES


def test_a_note_changed_after_the_event_keeps_both_versions(focus, monkeypatch, checker):
    root, place, _admission, _event = focus
    p.edit_place(root, place.place_id, "add_note", author="owner", text="EARLIER: you may discuss my projects here.")

    def narrow():  # the owner replaces the note between the event and the reply
        p.edit_place(root, place.place_id, "remove_note", author="owner", index=0)
        p.edit_place(root, place.place_id, "add_note", author="owner", text="NOW: keep my projects out of this chat.")
    _ctx, bundle, prompt = _reply_bundle(root, monkeypatch, checker, before_reply=narrow)
    source = bundle["occasion"]["sources"][0]
    assert [n["text"] for n in source["notes_at_event"]] == ["EARLIER: you may discuss my projects here."]
    assert [n["text"] for n in bundle["standing_delegation"]["place_notes"]] == ["NOW: keep my projects out of this chat."]
    assert prompt.count("EARLIER") == 1 and prompt.count("NOW: keep") == 1


def test_a_retained_task_without_typed_sources_keeps_its_whole_row(focus, monkeypatch, checker):
    root, place, _admission, _event = focus
    p.edit_place(root, place.place_id, "add_note", author="owner", text="OWNER NOTE")
    ctx, bundle, _prompt = _reply_bundle(root, monkeypatch, checker, typed=False)
    assert bundle["occasion"]["sources"] == _whole(ctx._owner_directives)
    assert "Place facts (context only)" in bundle["occasion"]["sources"][0]["content"]


@pytest.mark.parametrize("thread,expected", [("T", "chat:a:r:T"), ("", "chat:a:r:0")])
def test_reply_route_names_the_thread_it_answers(focus, monkeypatch, checker, thread, expected):
    """A room-wide place admits an event in topic T; the reply returns to T, and so does its evidence."""
    root, place, _admission, _event = focus
    assert place.thread_id == ""
    _ctx, bundle, _prompt = _reply_bundle(root, monkeypatch, checker, thread=thread)
    row = bundle["subject"]["place"]["places"][0]
    assert row["conversation_key"] == expected and row["conversation_keys"] == [expected]
    assert bundle["occasion"]["focus"]["conversation_key"] == expected
    assert bundle["subject"]["place"]["targets"]["thread_id" if thread else "conversation_id"] == [thread or "r"]


@pytest.mark.parametrize("kind", ["file", "unreadable_image", "image"])
def test_a_frame_with_attachments_rerenders_from_its_records(focus, monkeypatch, checker, tmp_path, kind):
    """The producer adds an attachment manifest and image blocks to the frame: re-rendered with them byte for byte,
    the frame becomes its typed records, the manifest kept and each image as facts whose pixels are not assessed.
    A frame the records cannot reproduce (a pixels-unavailable note) stays whole."""
    root, place, _admission, _event = focus
    if kind == "file":
        source = tmp_path / "brief.txt"
        source.write_text("attached words", encoding="utf-8")
    elif kind == "unreadable_image":  # an image signature with bytes no decoder reads
        source = tmp_path / "photo.png"
        source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not an image")
    else:
        image = pytest.importorskip("PIL.Image")
        source = tmp_path / "photo.png"
        image.new("RGB", (2, 2), "red").save(source)
    ctx, bundle, prompt = _reply_bundle(root, monkeypatch, checker, staged=[source])
    retained = ctx._owner_directives[0]
    frame = retained["content"] if isinstance(retained["content"], str) else retained["content"][0]["text"]
    assert "[ATTACHMENTS]" in frame and source.name in frame
    shown = bundle["occasion"]["sources"][0]
    if kind == "unreadable_image":
        assert bundle["occasion"]["sources"] == _whole(ctx._owner_directives)  # the recorded row, whole
        assert "Pixels unavailable" in frame and "Pixels unavailable" in prompt
        return
    assert "content" not in shown and shown["message_text"] == "hello there"
    assert shown["attachments"].startswith("[ATTACHMENTS]") and source.name in shown["attachments"]
    assert prompt.count("[ATTACHMENTS]") == 1 and "base64," not in prompt
    if kind == "image":
        [facts] = shown["images"]
        assert facts["pixels"] == "not assessed" and facts["media_type"].startswith("image/")
        assert len(facts["sha256"]) == 64 and facts["bytes"] > 0
    else:
        assert "images" not in shown


def test_a_large_image_raw_payload_and_thread_fit_with_the_text_once(focus, monkeypatch, checker, tmp_path):
    """A 2 MB screenshot, the provider's raw message and the thread history (the 04.11 TOO_LARGE class): the bundle
    carries no base64, the correspondent's text once (the thread points at it) and fits the budget."""
    import os
    from ouroboros import safety
    image = pytest.importorskip("PIL.Image")
    source = tmp_path / "screen.png"
    image.frombytes("RGB", (1000, 700), os.urandom(1000 * 700 * 3)).save(source)
    assert source.stat().st_size > 2_000_000
    root, _place, _admission, _event = focus
    text = "On the two screenshots the banner texts differ; which one ships? " * 30
    conversation = {"thread_history": {"status": "available", "data": {"complete": True, "messages": [
        {"user": "U2", "ts": "1700000000.0", "text": "Banner review thread"},
        {"user": "person", "ts": "1700000001.0", "text": text}]}}}
    message = {"message_id": "1700000001.0", "raw": {"type": "message", "text": text, "blocks": [{"type": "rich_text",
                                                                                                 "text": text}]}}
    ctx, bundle, prompt = _reply_bundle(root, monkeypatch, checker, staged=[source], text=text,
                                        conversation=conversation, message=message)
    assert len(json.dumps(ctx._owner_directives).encode()) > 2_000_000  # the mind saw the pixels
    assert "base64," not in prompt and len(prompt.encode("utf-8")) < safety._SAFETY_BUNDLE_BYTE_BUDGET
    assert prompt.count(text.strip()) == 1
    shown = bundle["occasion"]["sources"][0]
    assert "raw" not in shown["event"]["message"]
    assert shown["event"]["conversation"]["thread_history"]["lines"] == [
        'U2 at 2023-11-14T22:13:20Z: "Banner review thread"',
        "person at 2023-11-14T22:13:21Z: [this message: its text is message_text]"]
    assert shown["images"][0]["pixels"] == "not assessed" and shown["images"][0]["bytes"] > 0


def test_each_retained_row_is_matched_to_its_own_event(focus, monkeypatch, checker):
    """Current metadata replaces only the frame it renders exactly; another event's frame or an order stays."""
    root, place, _admission, _event = focus
    p.edit_place(root, place.place_id, "add_note", author="owner", text="OWNER NOTE")
    _other, other_frame = _focus_task(root, text="an earlier, different message")
    other = {"source": "initial_text", "content": other_frame}
    order = {"source": "initial_text", "content": "Continue the work this place started."}
    _ctx, bundle, prompt = _reply_bundle(root, monkeypatch, checker, extra_rows=[other, order])
    first, second, third = bundle["occasion"]["sources"]
    assert first["message_text"] == "hello there" and "content" not in first
    assert [second, third] == _whole([other, order])
    assert prompt.count("an earlier, different message") == 1


def test_an_empty_event_placeholder_keeps_the_row_whole(focus, monkeypatch, checker):
    """The producer frames empty text as a placeholder the typed records do not hold."""
    root, _place, _admission, _event = focus
    ctx, bundle, _prompt = _reply_bundle(root, monkeypatch, checker, text="")
    assert "(empty presence event)" in ctx._owner_directives[0]["content"]
    assert bundle["occasion"]["sources"] == _whole(ctx._owner_directives)


def test_source_place_and_wake_authorship_survive_a_nonplace_call(focus, monkeypatch, checker):
    """Source-scoped rules remain judgeable when the action has no destination place."""
    from ouroboros import safety
    from ouroboros.loop_messages import _initialize_owner_directives

    root, place, _admission, _event = focus
    task, content = _focus_task(root)
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=root)
    _initialize_owner_directives(ctx, [{"role": "user", "content": content}])
    prompt = safety._build_check_prompt("run_command", {"cmd": ["echo", "hello"]}, ctx=ctx)
    bundle = json.loads(prompt.split("Host evidence (provenance is not consent):\n", 1)[1]
                        .split("\nProposed tool call:", 1)[0])
    assert not bundle["subject"]["place"]
    source = bundle["occasion"]["sources"][0]["place_at_event"]
    captured = task["metadata"]["presence"]["place"]
    assert source["place_id"] == place.place_id
    assert source["label"] == captured["label"]
    assert source["state"] == captured["state"]
    assert source["matched_wakes"] == [w for w in captured["wakes"]
        if w["fact"] in task["metadata"]["focus"]["matched_wakes"]]
    assert source["matched_wakes"] and all(w["author"] for w in source["matched_wakes"])


def _in_circle(root, place, *, text="CIRCLE WORDS: keep my projects out of this chat.", placed_by="owner"):
    circle = p.create_circle(root, name="Friends", text=text, author="owner")
    p.edit_place(root, place.place_id, "set_circle", author=placed_by, circle_id=circle["circle_id"])
    return circle


def test_circle_words_reach_the_frame_replay_and_bundle_once_with_both_origins(focus, monkeypatch, checker):
    """A saved circle and placement reach the admitted frame (the event-time record), re-render byte for byte from the
    typed records, and reach the Supervisor once, as standing delegation only through my human's own placement."""
    from ouroboros.presence_context import focus_frame_sources, frame_focus_content
    root, place, _admission, _event = focus
    _in_circle(root, place)
    task, content = _focus_task(root)
    frame = content[0]["text"] if isinstance(content, list) else content
    circle = task["metadata"]["presence"]["circle"]
    assert (circle["name"], circle["text_author"], circle["placed_by"]) == ("Friends", "owner", "owner")
    assert '- its circle\'s words, shared by every place in it' in frame and frame.count("CIRCLE WORDS") == 1
    typed = focus_frame_sources(task["metadata"], frame_focus_content(task, task["metadata"]["presence"]["observed_text"]))
    assert typed["circle_at_event"]["text"].startswith("CIRCLE WORDS")
    _ctx, bundle, prompt = _reply_bundle(root, monkeypatch, checker)
    assert "circle_at_event" not in bundle["occasion"]["sources"][0]  # the same words are read now
    [entry] = [n for n in bundle["standing_delegation"]["place_notes"] if "circle" in n]
    assert (entry["circle"], entry["author"], entry["placed_by"], entry["standing_delegation"]) == (
        "Friends", "owner", "owner", True)
    assert prompt.count("CIRCLE WORDS") == 1


@pytest.mark.parametrize("change", ["reworded", "deleted"])
def test_circle_words_changed_after_the_event_keep_the_events_words_beside_todays(focus, monkeypatch, checker, change):
    root, place, _admission, _event = focus
    circle = _in_circle(root, place, text="EARLIER: you may discuss my projects here.")

    def later():
        if change == "reworded":
            p.edit_circle(root, circle["circle_id"], author="owner", text="NOW: keep my projects out of this chat.")
        else:
            p.delete_circle(root, circle["circle_id"], author="owner")
    _ctx, bundle, prompt = _reply_bundle(root, monkeypatch, checker, before_reply=later)
    source = bundle["occasion"]["sources"][0]
    assert source["circle_at_event"]["text"] == "EARLIER: you may discuss my projects here."
    now = [n["text"] for n in bundle["standing_delegation"]["place_notes"] if "circle" in n]
    assert now == (["NOW: keep my projects out of this chat."] if change == "reworded" else [])
    assert prompt.count("EARLIER") == 1


def test_a_frame_recorded_before_circles_renders_as_it_did(focus):
    """FOCUS.md has no circle field: the circle's words ride one more line after the place's own notes only when the
    event-time record has them, so a frame recorded before circles re-renders byte for byte and still replays."""
    from ouroboros.presence_context import PLACEHOLDERS, focus_frame_sources, frame_focus_content
    root, place, _admission, _event = focus
    task, content = _focus_task(root)
    frame = content[0]["text"] if isinstance(content, list) else content
    assert "circle" not in task["metadata"]["presence"] and "its circle" not in frame
    assert "circle_line" not in PLACEHOLDERS
    assert frame_focus_content(task, task["metadata"]["presence"]["observed_text"]) == frame
    assert focus_frame_sources(task["metadata"], frame) is not None
    _in_circle(root, place)
    with_circle, framed = _focus_task(root, text="another message")
    framed = framed[0]["text"] if isinstance(framed, list) else framed
    notes_line = next(line for line in framed.splitlines() if line.startswith("- notes with their authors:"))
    following = framed.splitlines()[framed.splitlines().index(notes_line) + 1]
    assert following.startswith("- its circle's words") and "CIRCLE WORDS" in following
    assert "(your human's words restrict this place, and consent here only when your human placed it)" in following


def test_a_placement_i_made_carries_my_humans_words_as_labelled_restriction_into_a_refusal(focus, monkeypatch, checker):
    from ouroboros import safety
    root, place, _admission, _event = focus
    _in_circle(root, place, placed_by="ouroboros")
    _ctx, bundle, _prompt = _reply_bundle(root, monkeypatch, checker)
    [entry] = [n for n in bundle["standing_delegation"]["place_notes"] if "circle" in n]
    assert entry["placed_by"] == "ouroboros" and entry["standing_delegation"] is False  # words kept, not filtered
    lines = safety._destination_lines({"place": bundle["subject"]["place"], "notes": bundle["standing_delegation"]["place_notes"]})
    assert len(lines) == 1 and "its circle \"Friends\"'s words, by your human, placed there by you" in lines[0]
    # Their restriction still binds there; only consent needs their own placement (prompts/SAFETY.md).
    assert "(they still restrict here; consent needs your human's placement)" in lines[0] and "CIRCLE WORDS" in lines[0]
    assert "context" not in lines[0]


@pytest.mark.parametrize("kind", ["press", "comment", "unavailable"])
def test_a_held_reference_only_answer_keeps_the_safety_source_shape(tmp_path, kind):
    from ouroboros.owner_quiz import record_answered, record_asked
    from ouroboros.owner_words import safety_sources

    asker, quiz, label = "birch-desk", "birch-choice", "Use birch paper"
    comment = "  keep the sample  " if kind == "comment" else ""
    if kind != "unavailable":
        record_asked(tmp_path, asker, quiz_id=quiz, question="Which paper?", options=[label])
        record_answered(tmp_path, asker, quiz_id=quiz, option_index=0, comment=comment, request_id="birch-answer")
    row = {"direction": "in", "text": comment or "1. " + label, "ts": "2040-03-01T00:00:00Z",
           "late_answer": {"task_id": asker, "quiz_id": quiz}, "carried": {"went_to": ""}}
    ctx = SimpleNamespace(task_id="birch-reader", drive_root=tmp_path, _held_owner_messages={"birch-address": row})
    [item] = safety_sources(ctx)
    assert item["source"] == {"press": "owner_press", "comment": "owner_mailbox", "unavailable": "owner_card_input"}[kind]
    assert (item["msg_id"], item["ts"]) == ("birch-address", row["ts"])
    if kind == "unavailable":
        assert set(item) == {"content", "source", "msg_id", "ts", "carried"} and label not in item["content"]
    else:
        assert set(item) == {"content", "source", "msg_id", "ts", "reply_to", "carried"}
        assert item["content"] == row["text"] and item["reply_to"]["address"] == "quiz birch-choice of task birch-desk"
