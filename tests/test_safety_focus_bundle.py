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
    assert shown == {**answer, "author": "the owner: the owner's answer to the card quoted in reply_to "
                                         "(the card's text was written by Ouroboros)"}  # the answer and its card
    assert shown["content"] == "1. Publish it" and shown["reply_to"]["text"].startswith(card["question"])


def test_every_source_names_its_author_from_the_one_table(checker, tmp_path):
    """Each row the Supervisor reads carries ``author`` from the host's one table, by the label it was recorded under.
    The run's opening text is the host's or Ouroboros's (a helper's brief: its parent's), never the owner's; a row
    under no known label, and a caller's raw messages, are unattributed."""
    from ouroboros.safety_evidence import SOURCE_AUTHORS, UNATTRIBUTED

    rows = [*({"source": label, "content": f"words {n}", "msg_id": f"m{n}"} for n, label in enumerate(SOURCE_AUTHORS)),
            {"source": "initial_text", "content": "Opening frame"},
            {"source": "made_up", "content": "x", "msg_id": "q"}, {"content": "no label", "msg_id": "r"}]
    ctx = SimpleNamespace(task_id="t", task_metadata={}, drive_root=tmp_path, _owner_directives=rows)
    shown = _sources(ctx)["occasion"]["sources"]
    assert [row["author"] for row in shown] == [*SOURCE_AUTHORS.values(), OPENING, UNATTRIBUTED, UNATTRIBUTED]
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
    later_words = {"author": "the owner: a later message to this work", "ts": "2026-10-04T06:00:00+00:00"}
    assert sources == [{"source": "owner_mailbox", "content": "1. Publish it", "msg_id": addresses[""],
                        "reply_to": card_reply, **later_words}]
    assert addresses["sibling"] not in prompt and "March figures" not in prompt
    corpus = {"source": "owner_mailbox", "content": "1. Publish it", "msg_id": addresses[""], "chat_id": "1",
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
    assert sources == [{"source": "initial_text", "content": "Ship the release notes.", "author": brief},
                       {"source": "owner_mailbox", "content": "1. Publish it", "msg_id": "chat 1 / owner-answer",
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
            p.delete_circle(root, circle["circle_id"])
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
