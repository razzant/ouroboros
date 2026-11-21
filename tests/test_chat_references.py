"""Reply facts and addresses name persisted rows, independently of their transport/view."""
import json

import pytest

from ouroboros import dialogue_store
from ouroboros.chat_chain import format_address, row_address
from ouroboros.chat_references import reference_from_row, resolve_reply
from supervisor import message_bus
from tests import _dialogue as dialogue_test
from tests.test_chat_attachments import _history, _web_bridge


def test_reply_is_host_resolved_and_replay_uses_the_same_address(tmp_path, monkeypatch):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    original = message_bus.log_chat("out", 1, 0, "  One\n\t" + "word " * 80, ts="2026-10-01T00:00:00Z")
    address = format_address(row_address(original))
    bridge.handle_web_message("this one", client_message_id="reply", task_metadata={"reply_to": address})
    row = dialogue_test.chat_rows(tmp_path)[-1]
    assert row["reply_to"]["snippet"] == ("One " + "word " * 80)[:159].rstrip() + "…"  # a cut preview says so
    assert row["reply_to"]["chars"] == len(original["text"]), "and the reference keeps how long the whole message is"
    assert row["reply_to"]["role"] == "assistant"
    assert echoes[-1]["reply_to"] == row["reply_to"]
    assert echoes[-1]["row_address"] == format_address(row_address(row))
    replay = next(r for r in _history(tmp_path) if r.get("client_message_id") == "reply")
    assert replay["row_address"] == echoes[-1]["row_address"]
    assert replay["reply_to"] == row["reply_to"]
    queued = bridge._inbox.get_nowait()
    assert queued["task_metadata"]["reply_reference"] == row["reply_to"]
    bridge.handle_web_message("this one", client_message_id="reply", task_metadata={"reply_to": address})
    assert bridge._inbox.empty()
    with pytest.raises(ValueError, match="different message"):
        bridge.handle_web_message("this one", client_message_id="reply", task_metadata={"reply_to": "task:other"})


def test_reply_survives_rotation_and_missing_reference_stays_explicit(tmp_path):
    row = {"chat_id": 1, "ts": "2026-10-01T00:00:00Z", "direction": "in", "text": "yes"}
    dialogue_test.seed(tmp_path, archives={"chat_20261002T000000.jsonl": [row]})
    address = format_address(row_address(row))
    assert resolve_reply(tmp_path, address)["unresolved"] is True  # still importing: unknown, never a failure
    dialogue_test.import_all(tmp_path)
    assert resolve_reply(tmp_path, address)["snippet"] == "yes"
    assert resolve_reply(tmp_path, "task:missing") == {"address": "task:missing", "unresolved": True}
    assert resolve_reply(tmp_path, address[:-12] + "000000000000")["unresolved"] is True
    with pytest.raises(ValueError, match="string"):
        resolve_reply(tmp_path, {"snippet": "forged"})


@pytest.mark.parametrize("extra", [{}, {"reply_excerpt": {"text": "this", "start_utf16": 5}}])
def test_skill_ingress_shares_reply_and_address_with_web(tmp_path, monkeypatch, extra):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    row = message_bus.log_chat("out", 1, 0, "Pick this", ts="2026-10-01T00:00:00Z")
    address = format_address(row_address(row))
    accepted, rejoined = message_bus.accept_local_message(
        bridge, tmp_path, "yes", chat_id=1, user_id=1, client_message_id="skill-reply",
        source="skill:telegram", task_metadata={"reply_to": address, **extra})
    assert not rejoined and accepted["reply_to"]["address"] == address
    assert accepted["reply_to"].get("excerpt") == extra.get("reply_excerpt")
    queued = bridge._inbox.get_nowait()
    message_bus.record_inbound_message(bridge, queued, chat_id=1, user_id=1,
        client_message_id="skill-reply", text="yes", ts=accepted["ts"])
    assert echoes[-1]["row_address"] == format_address(row_address(accepted))
    assert echoes[-1]["reply_to"] == accepted["reply_to"]


def test_progress_address_resolves_without_hashing_projection_fields(tmp_path):
    from ouroboros.chat_references import projection_fields
    row = {"chat_id": 1, "ts": "2026-10-01T01:00:00Z", "task_id": "worker", "text": "Testing"}
    dialogue_store.append(tmp_path, "progress", row)
    address = projection_fields(row)["row_address"]
    assert resolve_reply(tmp_path, address)["snippet"] == "Testing"
    assert resolve_reply(tmp_path, address + "0")["unresolved"] is True


def test_answer_row_uses_the_turns_host_origin(tmp_path, monkeypatch):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    origin = message_bus.log_chat("in", 1, 1, "Original request", client_message_id="origin")
    write_task_result(tmp_path, "work", "running", origin_message_ref=build_owner_message_ref(
        chat_id=1, client_message_id="origin", text=origin["text"], ts=origin["ts"]))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    message_bus.send_with_budget(1, "Answer", task_id="work")
    row = dialogue_test.chat_rows(tmp_path)[-1]
    assert row["reply_to"]["address"] == format_address(row_address(origin))
    assert echoes[-1]["row_address"] == format_address(row_address(row))
    assert echoes[-1]["reply_to"] == row["reply_to"]


def _reply_to(bridge, stored, cmid):
    address = format_address(row_address(stored))
    bridge.handle_web_message("About this", client_message_id=cmid, task_metadata={"reply_to": address})
    return address, bridge._inbox.get_nowait()["task_metadata"]["reply_reference"]


def test_reply_to_authored_completion_mirror_quotes_the_shown_answer(tmp_path, monkeypatch):
    from ouroboros.context import build_runtime_section
    from tests._context_shared import _make_health_env
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    answer = "ACTUAL AUTHORED ANSWER:\n\n the exact result the owner sees. " + "more " * 60
    stored = message_bus.log_chat("system", 1, 0, "Project / Done / open room", ts="2026-10-17T08:00:00Z",
                                  record_type="project_completion_summary",
                                  message_meta={"completion_answer": answer, "project_id": "project-a"})
    address, reference = _reply_to(bridge, stored, "mirror-reply")
    assert reference["address"] == address and reference["kind"] == "completion"
    assert reference["role"] == "assistant" and reference["author_label"] == "Ouroboros"
    assert reference["snippet"] == " ".join(answer.split())[:159].rstrip() + "…" and reference["chars"] == len(answer)
    assert echoes[-1]["reply_to"] == reference
    replay = next(r for r in _history(tmp_path) if r.get("client_message_id") == "mirror-reply")
    assert replay["reply_to"] == reference
    task = {"id": "reply", "type": "task", "metadata": {"reply_reference": reference}}
    model = json.loads(build_runtime_section(_make_health_env(tmp_path / "env"), task).split("\n\n", 1)[1])
    assert model["reply_to"]["snippet"] == reference["snippet"]
    raw = dialogue_test.chat_rows(tmp_path)[0]  # the stored row, as the dialogue store keeps it
    assert raw["text"] == "Project / Done / open room" and raw["direction"] == "system"
    assert raw["completion_answer"] == answer and format_address(row_address(raw)) == address


@pytest.mark.parametrize("direction,record_type,meta,role", [
    ("system", "task_error", {}, "system"),
    ("system", "project_completion_summary", {"completion_answer": "", "project_id": "project-a"}, "system"),
    ("out", "", {}, "assistant"),
])
def test_reply_to_other_rows_keeps_their_own_text_and_voice(tmp_path, monkeypatch, direction, record_type, meta, role):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    stored = message_bus.log_chat(direction, 1, 0, "Row  text\nas stored", ts="2026-10-17T08:00:00Z",
                                  record_type=record_type, message_meta=meta)
    address, reference = _reply_to(bridge, stored, "plain-reply")
    assert reference["address"] == address and reference["snippet"] == "Row text as stored"
    assert reference["role"] == role and echoes[-1]["reply_to"] == reference
    # Only a Project lifecycle row speaks with its authored answer, as in Main.
    other = reference_from_row({**stored, "direction": "system", "type": "task_error", "completion_answer": "Hidden"})
    assert other["role"] == "system" and other["snippet"] == "Row text as stored"


def test_an_outgoing_rows_own_name_travels_beside_the_owner_correlation(tmp_path, monkeypatch):
    """A relay is saved under its own name; its live frame carries that name, never the owner's id."""
    from types import SimpleNamespace

    from ouroboros.chat_references import projection_fields
    from ouroboros.owner_relay import publish_relay, write_relay
    from ouroboros.task_finalization import host_operation_reply_kwargs
    entry, _ = write_relay(tmp_path, "receiver", relay_id="r1", note="Same words", quotes=[], chat_id=1,
                           teller_task_id="teller")
    sent = []
    publish_relay(tmp_path, "receiver", entry, SimpleNamespace(send_message=lambda chat_id, text, **kw: sent.append(kw)))
    assert sent[0]["progress_meta"]["row_client_message_id"] == entry["msg_id"]
    assert "client_message_id" not in sent[0]["progress_meta"], "the owner's correlation field stays unused"
    stored = next(found.row for found in dialogue_store.rows_by_client_id(tmp_path, entry["msg_id"]))
    assert projection_fields(stored)["row_client_message_id"] == entry["msg_id"], "history projects the same name"
    # The owner's own row keeps its id in client_message_id alone; a row saved without a name gains nothing.
    assert "row_client_message_id" not in projection_fields({**stored, "direction": "in"})
    assert "row_client_message_id" not in projection_fields({**stored, "client_message_id": ""})
    assert "row_client_message_id" not in projection_fields({**stored, "is_progress": True}), "progress merges meta"
    # A host reply names the owner's submission on its live frame; its saved row has no name of its own.
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    message_bus.send_with_budget(1, "Restarting", role="system", system_type="command_reply",
                                 **host_operation_reply_kwargs("web", "", "owner-cmid"))
    frame = next(echo for echo in echoes if echo.get("content") == "Restarting")
    assert frame["client_message_id"] == "owner-cmid" and frame["row_address"]
    assert "row_client_message_id" not in frame, "the owner's correlation never becomes the reply's own name"


def test_a_call_opening_row_names_itself_on_its_live_echo():
    """The call-open row is ingress the host wrote: its live echo must carry its own name, or the chat cannot fold
    it into the call card and the owner sees the opening as a System bubble beside every call (27.10)."""
    from ouroboros.chat_references import projection_fields
    opening = {"direction": "in", "role": "system", "system_type": "call_open", "chat_id": 1, "ts": "t",
               "client_message_id": "call:c1:open", "host_fact": {"kind": "call_open", "call_id": "c1"}, "text": "Cue"}
    assert projection_fields(opening)["row_client_message_id"] == "call:c1:open"
    owner = {**opening, "role": "user", "system_type": ""}
    owner.pop("host_fact")
    assert "row_client_message_id" not in projection_fields(owner), "the owner's own row keeps its id alone"


def test_a_reply_to_a_system_row_names_system_on_every_surface_never_you(tmp_path, monkeypatch):
    """A call's opening enters by the owner's door (``direction: in``) yet is the host's row (``role: system``): a Reply
    to it signs System on the reference, on the row's live frame and on its replay, and nobody can quote a passage of it."""
    from ouroboros.chat_references import ReplyExcerptNotAuthored, ingress_reply, projection_fields
    from ouroboros.owner_words import reply_target

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    bridge.handle_web_message("Cue: greet the caller.", client_message_id="call:c1:open",
                              task_metadata={"host_fact": {"kind": "call_open", "call_id": "c1"}})
    opening = next(echo for echo in echoes if echo.get("client_message_id") == "call:c1:open")
    stored = dialogue_test.chat_rows(tmp_path)[-1]
    assert (stored["direction"], stored["role"]) == ("in", "system")
    assert opening["role"] == "system" and opening["reply_author_label"] == projection_fields(stored)["reply_author_label"] == "System"
    replay = next(row for row in _history(tmp_path) if row.get("client_message_id") == "call:c1:open")
    assert (replay["role"], replay["reply_author_label"]) == ("system", "System")
    reference = resolve_reply(tmp_path, opening["row_address"])
    assert (reference["role"], reference["author_label"]) == ("system", "System")
    bridge.handle_web_message("About the call.", client_message_id="about-call", task_metadata={"reply_to": opening["row_address"]})
    echo = next(echo for echo in echoes if echo.get("client_message_id") == "about-call")
    assert (echo["reply_to"]["role"], echo["reply_to"]["author_label"]) == ("system", "System")
    assert reply_target(tmp_path, echo["reply_to"])["author"] != "You", "the model reads the host's row, never my human's"
    with pytest.raises(ReplyExcerptNotAuthored):
        ingress_reply(tmp_path, {"reply_to": opening["row_address"], "reply_excerpt": {"text": "Cue", "start_utf16": 0}})
    # The owner's own row, beside it, still signs You: only the host's rows changed.
    own = message_bus.log_chat("in", 1, 1, "Meet at noon.", ts="2026-10-01T00:00:00Z", source="web")
    assert projection_fields(own)["reply_author_label"] == "You" and resolve_reply(
        tmp_path, format_address(row_address(own)))["role"] == "user"


@pytest.mark.parametrize("comment", ["", "  Follow the creek at dawn.  "])
def test_late_quiz_bubble_is_selectable_only_when_it_contains_entered_words(tmp_path, monkeypatch, comment):
    from ouroboros.chat_references import ReplyExcerptNotAuthored, ingress_reply
    from ouroboros.gateway.task_decision import deliver_as_owner_message
    from ouroboros.owner_quiz import late_answer_owner_text
    from ouroboros.owner_words import reply_target

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    block = {"quiz_id": "trail", "state": "answered", "answered_after_terminal": True,
             "answered_index": 0, "options": ["Take the ridge"], "comment": comment}
    answer = {"task_id": "trail-work", "quiz_id": "trail", "answer_snapshot": {"task_id": "trail-work", "quiz": block}}
    text = late_answer_owner_text(block)
    assert deliver_as_owner_message(tmp_path, 1, text, "quiz_late_answer:trail-work:trail",
                                    task_metadata={"late_answer": answer}) == (True, "")
    [stored] = dialogue_test.chat_rows(tmp_path)
    assert stored["late_answer"] == answer and stored["text"] == text
    [live] = echoes
    [history] = [r for r in _history(tmp_path) if r.get("client_message_id") == stored["client_message_id"]]
    assert live["content"] == history["text"] == text  # don't change what a card submission displayed
    assert live["role"] == history["role"] == "user"
    assert live.get("reply_excerpt_disabled", False) == history.get("reply_excerpt_disabled", False) == (not comment)
    address = live["row_address"]
    excerpt = {"text": "creek" if comment else "ridge", "start_utf16": text.index("creek" if comment else "ridge")}
    request = {"reply_to": address, "reply_excerpt": excerpt}
    if comment:
        reply = ingress_reply(tmp_path, request)
        assert reply["role"] == "user" and reply["excerpt"] == excerpt
        assert reply_target(tmp_path, reply)["text"] == comment
    else:
        with pytest.raises(ReplyExcerptNotAuthored):
            ingress_reply(tmp_path, request)
        reply = resolve_reply(tmp_path, address)
        # The bubble is still the owner's: a whole-row Reply names its author as it did; only the preview is the host's account.
        assert reply["role"] == "user" and reply["author_label"] == "You" and reply["kind"] == "message"
        assert "wrote no words" in reply["snippet"]
        target = reply_target(tmp_path, reply)
        assert "wrote no words" in target["text"] and "the text of task trail-work" in target["text"]
        assert target["author"] == "System", "a model reads the account as the host's"


def test_selected_reply_preserves_source_author_citation_and_replay_identity(tmp_path, monkeypatch):
    from ouroboros.chat_references import stored_reply
    from ouroboros.dialogue_provenance import quote_words, relation_text
    from ouroboros.owner_words import reply_target

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    source = "Amber **moon**. See the [map](atlas)."
    original = message_bus.log_chat("out", 1, 0, source, ts="2026-10-01T00:00:00Z")
    address = format_address(row_address(original))
    # A client can claim arbitrary rendered words. Only the addressed source can speak for its author.
    excerpt = {"text": "A different passage.\n  " + "Snow-covered hills. " * 12, "start_utf16": 7}
    meta = {"reply_to": address, "reply_excerpt": {**excerpt, "author": "Pretender", "role": "user"}}
    bridge.handle_web_message("Trace this path.", client_message_id="selected", task_metadata=meta)
    stored = dialogue_test.chat_rows(tmp_path)[-1]
    ref = stored["reply_to"]
    assert ref == {**reference_from_row(original), "excerpt": excerpt}
    delivered = bridge._inbox.get_nowait()["task_metadata"]["reply_reference"]
    assert ref == echoes[-1]["reply_to"] == stored_reply(delivered), "the row and its echo keep the compact reference"
    assert delivered["text"] == source, "delivery carries the source whole, captured before the owner row was saved"
    assert ref == next(r for r in _history(tmp_path) if r.get("client_message_id") == "selected")["reply_to"]
    assert json.dumps(excerpt["text"], ensure_ascii=False) in relation_text(stored)
    assert quote_words(stored) == stored["text"] == "Trace this path."
    assert reply_target(tmp_path, ref)["text"] == source
    monkeypatch.setattr("ouroboros.chat_references.resolve_row", lambda *args: (None, {}))
    bridge.handle_web_message("Trace this path.", client_message_id="selected", task_metadata=meta)
    for replacement in ({}, {"reply_excerpt": {**excerpt, "start_utf16": 2}}, {"reply_excerpt": {**excerpt, "text": "Amber"}}):
        with pytest.raises(ValueError, match="different message"):
            bridge.handle_web_message("Trace this path.", client_message_id="selected",
                                      task_metadata={"reply_to": address, **replacement})
    with pytest.raises(ValueError, match="unavailable"):
        bridge.handle_web_message("Trace this path.", client_message_id="fresh", task_metadata=meta)
    assert bridge._inbox.empty()


@pytest.mark.parametrize("words", ["Meet here", "Meet here\n\n[Attached file: map.pdf]"])
def test_selected_web_attachment_source_keeps_only_the_author_caption(tmp_path, monkeypatch, words):
    from ouroboros.chat_references import ingress_reply
    from ouroboros.owner_words import reply_target

    _web_bridge(tmp_path, monkeypatch)
    tail = "[Attached file: map.pdf]"
    file = {"name": "map.pdf", "mime": "application/pdf", "size": 55}
    source = message_bus.log_chat("in", 1, 1, f"{words}\n\n{tail}", source="web",
                                  message_meta={"attachments": [file]})
    address = format_address(row_address(source))
    selection = {"text": "Meet here", "start_utf16": 0}
    whole = ingress_reply(tmp_path, {"reply_to": address})
    ref = ingress_reply(tmp_path, {"reply_to": address, "reply_excerpt": selection})
    assert ref["snippet"] == " ".join(words.split())
    assert tail in reply_target(tmp_path, whole)["text"]  # whole-row Reply did not change
    target = reply_target(tmp_path, ref)
    assert target["text"] == words and target["attachments"][0]["name"] == "map.pdf"


@pytest.mark.parametrize("excerpt", [None, {}, {"text": " ", "start_utf16": 0},
                                     {"text": "a", "start_utf16": False}, {"text": "a", "start_utf16": -1},
                                     {"text": "a", "start_utf16": 2**53}, {"text": "\ud800", "start_utf16": 1}])
def test_invalid_excerpt_never_downgrades_to_an_ordinary_reply(excerpt):
    from ouroboros.chat_references import reply_identity

    with pytest.raises(ValueError, match="reply_excerpt"):
        reply_identity({"reply_to": "row:1@2026-10-01T00:00:00Z#0123456789ab", "reply_excerpt": excerpt})


def test_excerpt_of_project_answer_keeps_runtime_citation_and_whole_authored_source(tmp_path, monkeypatch):
    from ouroboros.context import build_runtime_section
    from ouroboros.owner_words import reply_target
    from tests._context_shared import _make_health_env

    _web_bridge(tmp_path, monkeypatch)
    selected = "The star chart is ready. " * 10 + "Finish."
    answer = f"**Atlas**\n\n{selected}"
    mirror = message_bus.log_chat("system", 1, 0, "Observatory update", record_type="project_completion_summary",
                                  message_meta={"completion_answer": answer, "project_id": "observatory"})
    from ouroboros.chat_references import ingress_reply
    excerpt = {"text": selected, "start_utf16": 6}
    ref = ingress_reply(tmp_path, {"reply_to": format_address(row_address(mirror)), "reply_excerpt": excerpt})
    task = {"id": "chart", "type": "task", "metadata": {"reply_reference": ref}}
    model = json.loads(build_runtime_section(_make_health_env(tmp_path / "env"), task).split("\n\n", 1)[1])
    assert model["reply_to"]["excerpt"] == excerpt and "client-reported" in model["reply_excerpt_note"]
    target = reply_target(tmp_path, ref)
    assert target == {"address": ref["address"], "author": "Ouroboros", "text": answer, "excerpt": excerpt,
                      "role": ref["role"]} and ref["role"] == "assistant"
    question = message_bus.log_chat("out", 1, 0, "Which chart?", record_type="quiz")
    with pytest.raises(ValueError, match="authored chat message"):
        ingress_reply(tmp_path, {"reply_to": format_address(row_address(question)), "reply_excerpt": excerpt})


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("initial_width", [12, 35, 64])
def test_reply_digest_acceptance_and_retry_verify_the_whole_supplied_prefix(tmp_path, monkeypatch, selected, initial_width):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    source = message_bus.log_chat("out", 1, 0, "A harbor.", record_type="proactive_message")
    row = row_address(source)
    prefix = format_address(row).rsplit("#", 1)[0] + "#"
    digest = row["row_sha256"]
    fields = {"reply_excerpt": {"text": "harbor", "start_utf16": 2}} if selected else {}
    def send(sha):
        bridge.handle_web_message("Explain", client_message_id="same-reply",
                                  task_metadata={"reply_to": prefix + sha, **fields})
    wrong = digest[:-1] + ("1" if digest[-1] == "0" else "0")
    if selected:
        with pytest.raises(ValueError, match="unavailable"):
            send(wrong)
    send(digest[:initial_width])
    assert echoes[-1]["reply_to"]["address"] == prefix + digest[:12]
    assert echoes[-1]["reply_to"]["snippet"] == "A harbor."
    monkeypatch.setattr("ouroboros.chat_references.resolve_row", lambda *_: pytest.fail("accepted digest lost"))
    for length in (12, 36, 64):
        send(digest[:length])
    with pytest.raises(ValueError, match="different message"):
        send(wrong)
    assert bridge._inbox.qsize() == 1


def test_older_reply_with_only_a_short_address_requires_the_source_for_more_digits(tmp_path, monkeypatch):
    from ouroboros.chat_references import same_reply

    _web_bridge(tmp_path, monkeypatch)
    source = message_bus.log_chat("out", 1, 0, "Earlier source.")
    row = row_address(source)
    short = format_address(row)
    prefix = short.rsplit("#", 1)[0] + "#"
    digest = row["row_sha256"]
    wrong = digest[:-1] + ("1" if digest[-1] == "0" else "0")
    saved = {"address": short}
    assert same_reply(tmp_path, saved, {"address": prefix + digest})
    assert not same_reply(tmp_path, saved, {"address": prefix + wrong})
    monkeypatch.setattr("ouroboros.chat_references.resolve_history_row", lambda *_: (None, {"state": "absent"}))
    assert same_reply(tmp_path, saved, {"address": short})
    assert not same_reply(tmp_path, saved, {"address": prefix + digest})


@pytest.mark.parametrize("fields,expected", [
    ({"transport": {"actor": {"display_name": "Mira", "platform_actor_id": "42"}}}, "Mira"),
    ({"transport": {"actor": {"platform_actor_id": "42"}}}, "42"),
    ({"sender_label": "Sasha", "transport": {"actor": {"display_name": "Mira"}}}, "Sasha"),
    ({"source": "web"}, "You"),
    ({"source": "presence:telegram", "transport": {"provider": "telegram"}}, "author not recorded"),
], ids=["actor-name", "actor-id", "sender-label-first", "owner-door", "unrecorded-place-author"])
def test_a_replied_row_is_signed_with_the_author_its_record_names(tmp_path, monkeypatch, fields, expected):
    """Without a ``sender_label`` the actor its transport recorded is the author; only a row of the owner's own door
    is You, and a place row whose author nothing records is not."""
    from ouroboros.chat_references import ingress_reply, projection_fields
    from ouroboros.gateway.history import _history_identity

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    source = message_bus.log_chat("in", 1, 1, "Meet at noon.", ts="2026-10-01T00:00:00Z", **{"source": "skill:telegram", **fields})
    assert projection_fields(source)["reply_author_label"] == expected
    assert _history_identity({**source, **projection_fields(source)})["reply_author_label"] == expected
    if not fields.get("source"):  # a skill's live echo; web is echoed at acceptance, places use their bound room
        message_bus.record_inbound_message(bridge, {"source": "skill:telegram", **fields}, chat_id=1, user_id=1,
                                           client_message_id="speaker-live", text="Meet at noon.", ts="2026-10-01T00:00:01Z")
        assert echoes[-1]["reply_author_label"] == expected
        replay = next(row for row in _history(tmp_path) if row.get("client_message_id") == "speaker-live")
        assert replay["reply_author_label"] == expected and replay["row_address"] == echoes[-1]["row_address"]
    whole = resolve_reply(tmp_path, format_address(row_address(source)))
    assert whole["role"] == "user" and whole["author_label"] == expected
    selected = ingress_reply(tmp_path, {"reply_to": whole["address"], "reply_excerpt": {"text": "noon", "start_utf16": 8}})
    assert selected["author_label"] == expected and selected["excerpt"] == {"text": "noon", "start_utf16": 8}


def test_a_late_answer_stored_without_its_snapshot_echoes_selectable_as_history_replays_it(tmp_path, monkeypatch):
    """The card's answer of an older late answer lives in the store: a skill's live echo reads it as history does, so
    its entered words are selectable at once, not only after a reload."""
    from ouroboros.owner_quiz import late_answer_owner_text

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    block = {"quiz_id": "trail", "state": "answered", "answered_after_terminal": True,
             "answered_index": 0, "options": ["Take the ridge"], "comment": "Follow the creek at dawn."}
    monkeypatch.setattr("ouroboros.owner_quiz.quiz_states", lambda root, task_id: {"trail": block})
    text = late_answer_owner_text(block)
    accepted, rejoined = message_bus.accept_local_message(
        bridge, tmp_path, text, chat_id=1, user_id=1, client_message_id="late-skill", source="skill:telegram",
        task_metadata={"late_answer": {"task_id": "trail-work", "quiz_id": "trail"}})
    assert not rejoined and "answer_snapshot" not in accepted["late_answer"]
    message_bus.record_inbound_message(bridge, bridge._inbox.get_nowait(), chat_id=1, user_id=1,
                                       client_message_id="late-skill", text=text, ts=accepted["ts"])
    [history] = [r for r in _history(tmp_path) if r.get("client_message_id") == "late-skill"]
    assert "reply_excerpt_disabled" not in history
    assert "reply_excerpt_disabled" not in echoes[-1], "the live echo reads the same store as the replay"
