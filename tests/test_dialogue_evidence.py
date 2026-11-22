"""Full room evidence retains conversations beyond consolidation and rotation."""
from __future__ import annotations

import pytest

from ouroboros import dialogue_store
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.memory_inventory import open_room_rows
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.utils import atomic_write_json
from tests import _dialogue as dialogue_test
from tests._store_shared import seed_result


def test_full_room_includes_rotated_consolidated_dialogue_and_child_lineage(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.projects_registry import project_binding_for_task

    project = create_project(tmp_path, "room", name="Our discussion")
    ref = build_owner_message_ref(chat_id=1, client_message_id="origin", ts="2026-09-01T00:00:00Z", text="Original choice")
    bind_task_to_project(tmp_path, "parent", project["id"], origin={"ref": ref, "text": "Original choice",
                         "snapshot": {"projectable": True}})
    unknown = "Check the legend"
    weak = build_owner_message_ref(chat_id=1, client_message_id="older", ts="2026-09-01T00:30:00Z", text=unknown)
    bind_task_to_project(tmp_path, "older", project["id"], origin={"ref": weak, "text": unknown})
    # Two rotated archive segments and the live chain of the older install, imported as the upgrade does.
    dialogue_test.seed(tmp_path, archives={
        f"chat_2026090{index + 1}.jsonl": [{"ts": f"2026-09-0{index + 1}T01:00:00Z", "chat_id": project["chat_id"],
                                           "direction": direction, "text": text}]
        for index, (direction, text) in enumerate([("in", "I need a plan"), ("out", "Option A saves time; option B keeps flexibility")])},
        live={"chat": [{"ts": "2026-09-03T01:00:00Z", "chat_id": 1, "direction": "out", "text": "Child explanation",
                        "task_id": "child", "root_task_id": "parent"}]})
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {"consolidated_chat_lines": 100000})
    # While the archive import is pending the reader names the gap; it never claims the rows are absent.
    importing = read_room_source(tmp_path, project["chat_id"])
    assert any(gap["kind"] == "history_importing" for gap in importing["coverage"]["chat"]["gaps"])
    dialogue_test.import_all(tmp_path)
    # The full reader ignores what the memory view counts as retold before the update (its open rows).
    ChronicleStore(tmp_path).ensure_activated()
    recent = open_room_rows(tmp_path, project["chat_id"])
    source = read_room_source(tmp_path, project["chat_id"])
    assert "Original choice" in source["text"] and "I need a plan" in source["text"]
    assert unknown not in source["text"] and project_binding_for_task(tmp_path, "older")["source_text"] == unknown
    assert {"kind": "origin_omission", "detail": "The visibility of a retained project opening could not be verified."} in (
        source["coverage"]["chat"]["gaps"])
    assert "Option A saves time; option B keeps flexibility" in source["text"]
    assert "Child explanation" in source["text"]
    assert source["coverage"]["chat"]["snapshot_stable"]
    assert len(source["rows"]) > len(recent)


def test_quiz_provenance_mailbox_and_attachment_names_are_complete(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.owner_mailbox import write_task_message

    quiz = {"quiz_id": "q", "question": "Which approach?", "options": [{"label": "Fast", "detail": "Less flexible", "recommended": True}, {"label": "Flexible"}], "state": "open"}
    dialogue_store.append(tmp_path, "chat", {"direction": "out", "chat_id": 1, "type": "quiz", "quiz": quiz, "task_id": "root", "ts": "2026-09-01T00:00:00Z"})
    winning = {"quiz_id": "q", "question": "Which approach?", "options": ["Fast", "Flexible"], "option_details": ["Less flexible", ""], "recommended_index": 0, "answered_index": 1, "comment": "  exact words  ", "asked_at": "2026-09-01T00:00:00Z", "answered_at": "2026-09-01T00:01:00Z", "state": "answered"}
    answer = {"direction": "system", "source": "owner_quiz_answer", "chat_id": 1, "type": "quiz_answer", "quiz": winning, "task_id": "root", "client_message_id": "quiz_answer:root:q", "ts": winning["answered_at"]}
    dialogue_store.append(tmp_path, "chat", answer)
    dialogue_store.append(tmp_path, "chat", answer)
    dialogue_store.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "text": "See attachment", "filename": "notes.pdf", "file_base64": "MUST_NOT_ATTACH_BINARY"})
    dialogue_store.append(tmp_path, "progress", {"chat_id": 1, "task_id": "root", "content": "Panel completed, collection pending", "status": "pending"})
    write_task_message(tmp_path, "Peer suggestion only", task_id="root", source_task_id="parent", provenance="peer_via_ancestor", relayed_from_task_id="peer")
    source = read_room_source(tmp_path, 1, task_id="root")
    assert source["text"].count('"type": "quiz_answer"') == 1
    assert '"author": "Owner"' in source["text"]
    for text in ("Less flexible", "recommended", "answered_index", "  exact words  ", "notes.pdf", "Panel completed, collection pending", "relayed by ancestor parent", "Peer suggestion only"):
        assert text in source["text"]
    assert "MUST_NOT_ATTACH_BINARY" not in source["text"]


def test_a_live_card_input_mailbox_delivery_reads_its_typed_input(tmp_path):
    from ouroboros.card_store import record_submission, write_card
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, write_owner_message

    asker, label, values = "map-desk", "Save choice", {"notes": "first line", "copies": 2}
    card_id = write_card(tmp_path, chat_id=1, task_id=asker, title="Map notes", fallback_text="Map notes",
                         kind="components", body={"components": []})["card_id"]
    record_submission(tmp_path, card_id, request_id="notes-1", action_id="save", label=label,
                      values=values, once=False, revision=1)
    record_submission(tmp_path, card_id, request_id="go-1", action_id="go", label="Just go",
                      values={}, once=False, revision=1)
    frame = f"[Card input] The owner pressed «{label}» on your card «Map notes» (card {card_id})."
    assert write_owner_message(tmp_path, frame, "root", msg_id=f"card_input:{card_id}:notes-1",
                               kind=KIND_CARD_INPUT, card_press={"task_id": asker, "card_id": card_id, "request_id": "notes-1"})
    assert write_owner_message(tmp_path, "[Card input] The owner pressed «Just go».", "root",
                               msg_id=f"card_input:{card_id}:go-1", kind=KIND_CARD_INPUT,
                               card_press={"task_id": asker, "card_id": card_id, "request_id": "go-1"})
    rows = {row.get("source_input", {}).get("source"): row for row in read_room_source(tmp_path, 1, task_id="root")["rows"]
            if row.get("stream") == "mailbox"}
    assert set(rows) == {"owner_mailbox", "owner_press"}
    form, press = rows["owner_mailbox"], rows["owner_press"]
    assert form["author"] == "Owner" and form["text"] == "first line\n2" and label not in form["text"]
    assert form["display_text"] == frame and form["source_input"]["submitted_values"] == values
    assert form["source_input"]["source_task_id"] == asker and form["source_input"]["card_id"] == card_id
    assert press["author"] == "host" and "pressed «Just go»" in press["text"] and "wrote no words" in press["text"]
    assert press["display_text"] == "[Card input] The owner pressed «Just go»."


def test_unknown_room_is_an_explicit_gap_and_chat_selectors_keep_the_room_text(tmp_path):
    import json
    from ouroboros.dialogue_evidence import chat_evidence_reader, read_room_source
    from ouroboros.tools.plan_evidence import resolve_evidence

    assert read_room_source(tmp_path, 999) is None
    dialogue_store.append(tmp_path, "chat", {"direction": "out", "chat_id": 1, "text": "First explanation"})
    dialogue_store.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "text": "OPENAI_API_KEY=sk-" + "x" * 48})
    manifest = resolve_evidence(["chat:999", "chat:1::lines=2-3"], active_root=tmp_path, allowed_roots=[], resolve_chat=chat_evidence_reader(tmp_path))
    assert {"locator": "chat:999", "reason": "chat_not_found"} in manifest["omissions"]
    [row] = manifest["attached"]
    assert row["kind"] == "chat" and row["selection_bytes"] > 0
    assert row["attached_bytes"] == len(row["text"].encode())
    assert json.loads(row["text"].split("\n")[1])["text"] == "OPENAI_API_KEY=sk-" + "x" * 48
    assert "secrets_redacted" not in row and "secrets_redacted" not in read_room_source(tmp_path, 1)


def test_explicit_hidden_room_keeps_its_address_and_never_becomes_main(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source
    dialogue_store.append(tmp_path, "chat", {"direction": "out", "chat_id": 0, "text": "Headless discussion"})
    dialogue_store.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "text": "Main conversation"})
    assert "Headless discussion" in read_room_source(tmp_path, 0)["text"]
    assert "Main conversation" not in read_room_source(tmp_path, 0)["text"]
    assert "Headless discussion" not in read_room_source(tmp_path, 1)["text"]


def test_unicode_separators_remain_inside_one_physical_jsonl_record(tmp_path):
    import json
    from ouroboros.dialogue_evidence import read_room_source, chat_evidence_reader
    from ouroboros.tools.plan_evidence import resolve_evidence

    original = 'A is cheaper\u2028B is flexible\u2029Keep both\u0085Owner explanation\nOrdinary newline'
    dialogue_store.append(tmp_path, "chat", {'direction': 'in', 'chat_id': 1, 'text': original})
    source = read_room_source(tmp_path, 1)
    assert source['rows'][0]['text'] == original and source['text'].count('\n') == 2
    manifest = resolve_evidence(['chat:1::lines=2-2'], active_root=tmp_path, allowed_roots=[],
                                resolve_chat=chat_evidence_reader(tmp_path))
    assert json.loads(manifest['attached'][0]['text'])['text'] == original


def test_steered_delivery_joins_exact_source_id_and_keeps_changed_text(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.owner_mailbox import write_owner_message

    dialogue_store.append(tmp_path, "chat", {'direction': 'in', 'chat_id': 1,
                 'text': 'Use B', 'client_message_id': 'cm1'})
    assert write_owner_message(tmp_path, 'Use B', 'root', msg_id='cm1:root')
    source = read_room_source(tmp_path, 1, task_id='root')
    assert source['text'].count('Use B') == 1
    delivery = source['rows'][0]['mailbox_deliveries'][0]
    assert delivery['task_id'] == 'root' and delivery['msg_id'] == 'cm1:root' and delivery['ts']
    # A transformed delivery of the same source remains exact, not deduplicated
    # by an interpretation of equivalent meaning.
    assert write_owner_message(tmp_path, 'Use B with the additional context', 'other', msg_id='cm1:other')
    changed = read_room_source(tmp_path, 1, task_id='other')
    assert changed['rows'][0]['mailbox_deliveries'][0]['text'] == 'Use B with the additional context'
    assert write_owner_message(tmp_path, 'Use B', 'root', msg_id='unrelated-id')
    distinct = read_room_source(tmp_path, 1, task_id='root')
    assert distinct['text'].count('Use B') == 2


def test_incoming_attachment_names_follow_the_existing_history_annotation(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace
    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.dialogue_evidence import read_room_source, _row_projection
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.project_dialogue import append_chat_annotation
    from supervisor.message_bus import log_chat

    upload = tmp_path / 'board-forecast.pdf'
    upload.write_bytes(b'%PDF-1.4\nBINARY_PAYLOAD_NOT_DIALOGUE\n')
    manifest = stage_task_attachments(tmp_path, 'prior-task', [str(upload)])
    assert manifest[0]['label'] == 'board-forecast.pdf'
    log_chat('in', 1, 0, 'Use the attached forecast', source='web',
             client_message_id='owner-upload', drive_root=tmp_path)
    append_chat_annotation(tmp_path, 'owner-upload', action='new_task', target='prior-task',
                           status='scheduled', attachment_manifest=manifest)
    history = asyncio.run(make_chat_history_endpoint(tmp_path)(
        SimpleNamespace(query_params={'n_human': '20', 'thread': '1'})))
    owner = next(row for row in json.loads(history.body)['messages'] if row.get('client_message_id') == 'owner-upload')
    source = read_room_source(tmp_path, 1)
    assert source['rows'][0]['attachments'][0]['label'] == owner['chat_annotation']['attachment_manifest'][0]['label']
    assert 'BINARY_PAYLOAD_NOT_DIALOGUE' not in source['text']
    direct = _row_projection({'direction': 'in', 'attachment_manifest': manifest}, 'mailbox', 1, tmp_path)
    assert direct['attachments'][0]['label'] == 'board-forecast.pdf'


@pytest.mark.parametrize("exact", [True, False])
def test_room_source_keeps_distinct_messages_and_links_only_identified_delivery(tmp_path, exact):
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.project_dialogue import append_chat_annotation

    first = {"direction": "in", "chat_id": 1, "client_message_id": "reused", "text": "Review these estimates",
             "ts": "2026-09-01T10:00:00+00:00"}
    second = {**first, "ts": "2026-09-01T10:01:00+00:00"}  # a separate message with the same client id and words
    dialogue_test.seed(tmp_path, live={"chat": [first, second, first]})
    dialogue_test.import_all(tmp_path)
    addresses = [format_address(row_address(row)) for row in (first, second)]
    assert append_chat_annotation(tmp_path, "reused", source_chat_id=1, source_row_address=addresses[0],
                                  action="mailbox_delivery", status="delivered",
                                  attachment_manifest=[{"label": "estimate.pdf", "abs_path": "/tmp/upload"}])
    assert write_owner_message(tmp_path, first["text"], "root", msg_id="reused:root", client_message_id="reused",
                               origin_message_ref=owner_row_source(first)[0] if exact else None)
    source = read_room_source(tmp_path, 1, task_id="root")
    a, b = [row for row in source["rows"] if row["stream"] == "chat"]
    assert [a["row_address"], b["row_address"]] == addresses
    assert a["attachments"] == [{"label": "estimate.pdf"}] and "attachments" not in b
    assert "mailbox_deliveries" not in b
    if exact:
        [delivery] = a["mailbox_deliveries"]
        assert delivery["msg_id"] == "reused:root" and "text" not in delivery
    else:
        assert "mailbox_deliveries" not in a
        [delivery] = [row for row in source["rows"] if row["stream"] == "mailbox"]
        assert delivery["text"] == first["text"]


def test_retained_file_only_origin_keeps_file_metadata_and_host_placeholder(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.owner_relay import owner_row_source

    project = create_project(tmp_path, "files")
    upload = "a" * 32 + "_sketch.pdf"
    media = {"name": "sketch.pdf", "mime": "application/pdf", "size": 123, "sha256": "b" * 64}
    ref, words = owner_row_source({"direction": "in", "chat_id": 1, "client_message_id": "file",
                                  "ts": "2026-09-01T10:00:00+00:00", "text": "(file attached)"})
    bind_task_to_project(tmp_path, "root", project["id"], origin={"ref": ref, "text": words, "snapshot": {
        "projectable": True, "placeholder_text": words, "source": "web",
        "attachments": [{**media, "upload": upload, "kind": "file"}],
    }})
    source = read_room_source(tmp_path, project["chat_id"])
    [opening] = source["rows"]
    assert (opening["stream"], opening["author"], opening["text_placeholder"]) == ("retained_origin", "host", True)
    assert opening["text"] == words and opening["attachments"] == [media]
    assert upload not in source["text"]


@pytest.mark.parametrize("kind", ["press", "form", "message"])
def test_opening_keeps_display_separate_from_accepted_input(tmp_path, kind):
    import asyncio
    import json
    from types import SimpleNamespace

    from ouroboros.card_store import owner_press_text, record_submission, write_card
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.owner_quiz import record_answered, record_asked
    from ouroboros.owner_relay import origin_snapshot, owner_row_address, owner_row_source
    from ouroboros.owner_words import task_owner_words, words_block

    source = tmp_path / "source"
    asker, label = "map-desk", "Save choice"
    values = {"notes": " first line\nsecond line ", "copies": 2}
    original = {"direction": "in", "chat_id": 1, "client_message_id": "accepted-opening",
                "ts": "2026-09-01T10:00:00+00:00", "text": "  Bring the map.  "}
    card_id = None
    if kind == "press":
        record_asked(source, asker, quiz_id="map-choice", question="Which map?", options=[label])
        record_answered(source, asker, quiz_id="map-choice", option_index=0, comment="", request_id="choice-1")
        original.update(text="1. " + label, late_answer={"task_id": asker, "quiz_id": "map-choice"})
    elif kind == "form":
        card_id = write_card(source, chat_id=1, task_id=asker, title="Map notes", fallback_text="Map notes",
                             kind="components", body={"components": []})["card_id"]
        record_submission(source, card_id, request_id="notes-1", action_id="save", label=label,
                          values=values, once=False, revision=1)
        original.update(text=owner_press_text(label, values),
                        card_press={"task_id": asker, "card_id": card_id, "request_id": "notes-1"})

    ref, display = owner_row_source(original)
    snapshot = origin_snapshot({**original, "_owner_source_root": str(source)}, tmp_path, include_reply=True)
    project = create_project(tmp_path, "map-room")
    binding = bind_task_to_project(tmp_path, "map-root", project["id"],
                                  origin={"ref": ref, "text": display, "snapshot": snapshot})
    history = asyncio.run(make_chat_history_endpoint(tmp_path)(
        SimpleNamespace(query_params={"n_human": "20", "chat_id": str(project["chat_id"])})))
    bubble = next(row for row in json.loads(history.body)["messages"] if row.get("origin_projected"))
    assert bubble["text"] == display and bubble["source_row_address"] == owner_row_address(original)
    assert bubble["ts"] == ref["ts"] and bubble["role"] == "user"
    [evidence] = read_room_source(tmp_path, project["chat_id"])["rows"]
    meta = {"origin_message_snapshot": snapshot}
    record = {"origin_message_ref": ref, "origin_message_text": display, "metadata": meta}
    authority = project_main_task_authority(record)["task_authority_origin"]
    words, _ = task_owner_words(tmp_path, "map-root", task=record)
    assert evidence["row_address"] == owner_row_address(original) == snapshot["reply"]["address"]
    assert authority["ref"] == ref and words[0]["row_sha256"] == ref["row_sha256"]

    if kind == "message":
        assert "input" not in snapshot and "source_input" not in binding and "opening_input_kind" not in bubble
        assert "source_input" not in evidence and evidence["text"] == authority["text"] == words[0]["text"] == display
        return

    typed = binding["source_input"]
    assert typed == snapshot["input"] == evidence["source_input"]
    assert typed["source_task_id"] == asker and bubble["opening_input_kind"] == typed["source"]
    assert authority["display_text"] == evidence["display_text"] == display
    assert authority["input"]["source"] == words[0]["source"] == typed["source"]
    if kind == "press":
        from ouroboros.answer_facts import retained_card_input

        assert typed["source"] == "owner_press" and typed["pressed"] == display
        assert evidence["author"] == "host" and "wrote no words" in evidence["text"]
        assert "wrote no words" in words[0]["text"] and f"text of task {asker}" in words[0]["text"]
        accepted = {**retained_card_input(original, source), **ref}
        draft = project_main_task_authority({**record, "metadata": {
            "objective_author": {"kind": "task", "task_id": "drafting-task"}, "owner_corpus": [accepted]}})
        # Ordinary dedup sees the very same raw string (the clicked label), not coverage of the owner's act.
        assert draft["task_owner_words"]["rows"] == [accepted]
    else:
        assert typed["source"] == "owner_mailbox" and typed["pressed"] == label
        assert typed["card_id"] == card_id and typed["submitted_values"] == values
        assert authority["input"]["reply_to"]["submitted_values"] == words[0]["answer_input"]["submitted_values"] == values
        assert evidence["text"] == words[0]["text"] == values["notes"] + "\n2"
        assert '"notes": " first line\\nsecond line ", "copies": 2' in words_block(words[0])
        from ouroboros.project_dialogue import opening_view, project_origin_rows

        [origin] = project_origin_rows(tmp_path, project["chat_id"])
        host_lines = opening_view(origin)["host_lines"]
        assert any("the card my human answered" in line for line in host_lines), host_lines
        assert any(line.startswith("[Host record: submitted fields (JSON): ") and '"copies": 2' in line
                   for line in host_lines), host_lines
        assert label not in evidence["text"] and label in words[1]["text"]
        # This same physical row can subsequently be read without its card store; the accepted binding still wins.
        dialogue_store.append(tmp_path, "chat", original)
        [physical] = read_room_source(tmp_path, project["chat_id"])["rows"]
        assert physical["stream"] == "chat" and physical["row_address"] == evidence["row_address"]
        assert physical["text"] == evidence["text"] and physical["source_input"] == typed


def test_room_source_signs_old_outgoing_rows_by_the_activation_lineage_epoch(tmp_path):
    """Before the chronicle is active every lineage-free outgoing row reads as mine, as it always
    did. Once the activation records the lineage epoch, a row before it is mine only by its task
    result; rows after it, and the reader's other rows, keep their signature."""

    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.task_result_schema import SCHEMA_VERSION_KEY, TASK_RESULT_SCHEMA_VERSION

    out = {"chat_id": 1, "direction": "out"}
    for row in (
        {"chat_id": 1, "direction": "in", "ts": "2026-08-01T00:00:01Z", "text": "Review the patch."},
        {**out, "ts": "2026-08-01T00:00:02Z", "task_id": "kid00001", "text": "## Summary Nobody recorded who wrote this."},
        {**out, "ts": "2026-08-01T00:00:03Z", "task_id": "kid00002", "text": "## Summary A child's report."},
        {**out, "ts": "2026-08-01T00:00:04Z", "task_id": "root0001", "text": "My own answer."},
        {**out, "ts": "2026-08-21T00:00:05Z", "task_id": "kid00003", "subagent_task_id": "kid00003",
         "parent_task_id": "root0002", "text": "The first row that carries lineage."},
        {**out, "ts": "2026-08-21T00:00:06Z", "task_id": "root0002", "text": "My answer after the epoch."},
    ):
        dialogue_store.append(tmp_path, "chat", row)
    for task_id, fields in (("root0001", {}), ("kid00002", {"parent_task_id": "root0001", "root_task_id": "root0001",
                                                             "delegation_role": "subagent"})):
        seed_result(tmp_path, task_id, {
            SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION, "task_id": task_id, "status": "completed", **fields})

    def authors():
        return [row["author"] for row in read_room_source(tmp_path, 1)["rows"]]

    human, child = authors()[0], "child kid00003 of root0002"
    assert authors() == [human, "Ouroboros (task kid00001)", "Ouroboros (task kid00002)", "Ouroboros (task root0001)",
                         child, "Ouroboros (task root0002)"]
    assert not (tmp_path / "memory" / "chronicle").exists()  # the reader never creates the chronicle
    assert ChronicleStore(tmp_path).ensure_activated()["metadata"]["lineage_epoch"]["pos"] == 4
    assert authors() == [human, "outgoing, author not recorded", "child kid00002 of root0001",
                         "Ouroboros (task root0001)", child, "Ouroboros (task root0002)"]
