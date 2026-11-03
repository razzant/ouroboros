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
from tests.test_place_focus import focus as _focus

focus = _focus
RULES = "Rule one: keep private work private.\nRule two: hold a disputed send and ask me in Main."


@pytest.fixture
def checker(monkeypatch):
    """The real check path; the physical model call records the prompt and answers SAFE."""
    from ouroboros import safety

    prompts = []
    safety._PRIOR_VERDICTS.clear()
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
    safety._PRIOR_VERDICTS.clear()


def _focus_task(root, *, text="hello there", thread="", staged=()):
    """One admitted event's focus task from the presence producer, and the first user content it frames."""
    from ouroboros.context import build_user_content

    key = conversation_key("chat", "a", "r", thread)
    event = PresenceTurnEvent("event-" + hashlib.sha256(f"{thread}:{text}".encode()).hexdigest()[:8], "chat", "a",
                              "r", thread, key, {"platform_actor_id": "person", "display_name": "Person"}, {}, {},
                              text, facts=("mention",))
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r", thread_id=thread)
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2,
                                                     "roots": ["root-a", "root-b"]}),
                       event, drive_root=root, staged_files=tuple(staged))
    task["drive_root"] = str(root)  # the worker lane binds it before the context reads staged images
    return task, build_user_content(task)


def _reply_bundle(root, monkeypatch, prompts, *, thread="", before_reply=None, typed=True, staged=(), extra_rows=(),
                  text="hello there"):
    """Admit an event, build its focus task, record its first row, then check the reply."""
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.place_reply import prepare_focus_reply

    task, content = _focus_task(root, text=text, thread=thread, staged=staged)
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


def test_each_source_projection_appears_once_and_the_event_is_typed(focus, monkeypatch, checker):
    root, place, _admission, _event = focus
    p.edit_place(root, place.place_id, "add_note", author="owner", text="OWNER NOTE: reply in Russian here.")
    p.edit_place(root, place.place_id, "add_note", author="ouroboros", text="MIND NOTE: the person likes brevity.")
    ctx, bundle, prompt = _reply_bundle(root, monkeypatch, checker)

    retained = ctx._owner_directives[0]
    assert retained["source"] == "initial_text" and "allowance known spend" in retained["content"]
    source = bundle["occasion"]["sources"][0]
    assert source["message_text"] == "hello there" and source["event"]["conversation_key"] == "chat:a:r:0"
    assert source["work_here"] == "Already started: task1" and "content" not in source
    assert "notes_at_event" not in source  # the same note versions are read now
    for text in ("OWNER NOTE", "MIND NOTE", RULES.splitlines()[0], "hello there"):
        assert prompt.count(text) == 1, text
    for frame_only in ("allowance known spend", "root-a", "Place facts (context only)"):
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
    assert bundle["occasion"]["sources"] == ctx._owner_directives
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
def test_a_frame_the_records_cannot_reproduce_stays_whole(focus, monkeypatch, checker, tmp_path, kind):
    """The producer adds an attachment manifest, image notes or blocks; the typed records hold none of them."""
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
    assert bundle["occasion"]["sources"] == ctx._owner_directives  # the recorded row, whole
    assert "[ATTACHMENTS]" in prompt
    if kind == "unreadable_image":
        assert "Pixels unavailable" in frame and "Pixels unavailable" in prompt
    if kind == "image":
        assert [block["type"] for block in retained["content"]] == ["text", "image_url"]
        assert bundle["occasion"]["sources"][0]["content"][1]["image_url"] == retained["content"][1]["image_url"]


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
    assert second == other and third == order
    assert prompt.count("an earlier, different message") == 1


def test_an_empty_event_placeholder_keeps_the_row_whole(focus, monkeypatch, checker):
    """The producer frames empty text as a placeholder the typed records do not hold."""
    root, _place, _admission, _event = focus
    ctx, bundle, _prompt = _reply_bundle(root, monkeypatch, checker, text="")
    assert "(empty presence event)" in ctx._owner_directives[0]["content"]
    assert bundle["occasion"]["sources"] == ctx._owner_directives


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
