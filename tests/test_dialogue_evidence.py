"""Full room evidence retains conversations beyond consolidation and rotation."""
from __future__ import annotations

from ouroboros.chronicle_store import ChronicleStore
from ouroboros.memory_inventory import open_room_rows
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.utils import append_jsonl, atomic_write_json


def test_full_room_includes_rotated_consolidated_dialogue_and_child_lineage(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source

    project = create_project(tmp_path, "room", name="Our discussion")
    ref = build_owner_message_ref(chat_id=1, client_message_id="origin", ts="2026-09-01T00:00:00Z", text="Original choice")
    bind_task_to_project(tmp_path, "parent", project["id"], origin={"ref": ref, "text": "Original choice"})
    for index, (direction, text) in enumerate([("in", "I need a plan"), ("out", "Option A saves time; option B keeps flexibility")]):
        append_jsonl(tmp_path / "archive" / f"chat_2026090{index + 1}.jsonl",
                     {"ts": f"2026-09-0{index + 1}T01:00:00Z", "chat_id": project["chat_id"], "direction": direction, "text": text})
    live = tmp_path / "logs" / "chat.jsonl"
    append_jsonl(live, {"ts": "2026-09-03T01:00:00Z", "chat_id": 1, "direction": "out", "text": "Child explanation", "task_id": "child", "root_task_id": "parent"})
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {"consolidated_chat_lines": 100000})
    # The full reader ignores what the memory view counts as retold before the update (its open rows).
    ChronicleStore(tmp_path).ensure_activated()
    recent = open_room_rows(tmp_path, project["chat_id"])
    source = read_room_source(tmp_path, project["chat_id"])
    assert "Original choice" in source["text"] and "I need a plan" in source["text"]
    assert "Option A saves time; option B keeps flexibility" in source["text"]
    assert "Child explanation" in source["text"]
    assert source["coverage"]["chat"]["snapshot_stable"]
    assert len(source["rows"]) > len(recent)


def test_quiz_provenance_mailbox_and_attachment_names_are_complete(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.owner_mailbox import write_task_message

    quiz = {"quiz_id": "q", "question": "Which approach?", "options": [{"label": "Fast", "detail": "Less flexible", "recommended": True}, {"label": "Flexible"}], "state": "open"}
    append_jsonl(tmp_path / "logs" / "chat.jsonl", {"direction": "out", "chat_id": 1, "type": "quiz", "quiz": quiz, "task_id": "root", "ts": "2026-09-01T00:00:00Z"})
    winning = {"quiz_id": "q", "question": "Which approach?", "options": ["Fast", "Flexible"], "option_details": ["Less flexible", ""], "recommended_index": 0, "answered_index": 1, "comment": "  exact words  ", "asked_at": "2026-09-01T00:00:00Z", "answered_at": "2026-09-01T00:01:00Z", "state": "answered"}
    answer = {"direction": "system", "source": "owner_quiz_answer", "chat_id": 1, "type": "quiz_answer", "quiz": winning, "task_id": "root", "client_message_id": "quiz_answer:root:q", "ts": winning["answered_at"]}
    append_jsonl(tmp_path / "logs" / "chat.jsonl", answer)
    append_jsonl(tmp_path / "logs" / "chat.jsonl", answer)
    append_jsonl(tmp_path / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1, "text": "See attachment", "filename": "notes.pdf", "file_base64": "MUST_NOT_ATTACH_BINARY"})
    append_jsonl(tmp_path / "logs" / "progress.jsonl", {"chat_id": 1, "task_id": "root", "content": "Panel completed, collection pending", "status": "pending"})
    write_task_message(tmp_path, "Peer suggestion only", task_id="root", source_task_id="parent", provenance="peer_via_ancestor", relayed_from_task_id="peer")
    source = read_room_source(tmp_path, 1, task_id="root")
    assert source["text"].count('"type": "quiz_answer"') == 1
    assert '"author": "Owner"' in source["text"]
    for text in ("Less flexible", "recommended", "answered_index", "  exact words  ", "notes.pdf", "Panel completed, collection pending", "relayed by ancestor parent", "Peer suggestion only"):
        assert text in source["text"]
    assert "MUST_NOT_ATTACH_BINARY" not in source["text"]


def test_unknown_room_is_an_explicit_gap_and_chat_selectors_use_redaction(tmp_path):
    from ouroboros.dialogue_evidence import chat_evidence_reader, read_room_source
    from ouroboros.tools.plan_evidence import resolve_evidence

    assert read_room_source(tmp_path, 999) is None
    append_jsonl(tmp_path / "logs" / "chat.jsonl", {"direction": "out", "chat_id": 1, "text": "First explanation"})
    append_jsonl(tmp_path / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1, "text": "OPENAI_API_KEY=sk-" + "x" * 48})
    manifest = resolve_evidence(["chat:999", "chat:1::lines=2-3"], active_root=tmp_path, allowed_roots=[], resolve_chat=chat_evidence_reader(tmp_path))
    assert {"locator": "chat:999", "reason": "chat_not_found"} in manifest["omissions"]
    [row] = manifest["attached"]
    assert row["kind"] == "chat" and row["selection_bytes"] > 0
    assert row["attached_bytes"] == len(row["text"].encode())
    assert "x" * 48 not in row["text"]


def test_explicit_hidden_room_keeps_its_address_and_never_becomes_main(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source
    append_jsonl(tmp_path / "logs" / "chat.jsonl", {"direction": "out", "chat_id": 0, "text": "Headless discussion"})
    append_jsonl(tmp_path / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1, "text": "Main conversation"})
    assert "Headless discussion" in read_room_source(tmp_path, 0)["text"]
    assert "Main conversation" not in read_room_source(tmp_path, 0)["text"]
    assert "Headless discussion" not in read_room_source(tmp_path, 1)["text"]


def test_unicode_separators_remain_inside_one_physical_jsonl_record(tmp_path):
    import json
    from ouroboros.dialogue_evidence import read_room_source, chat_evidence_reader
    from ouroboros.tools.plan_evidence import resolve_evidence

    original = 'A is cheaper\u2028B is flexible\u2029Keep both\u0085Owner explanation\nOrdinary newline'
    append_jsonl(tmp_path / 'logs/chat.jsonl', {'direction': 'in', 'chat_id': 1, 'text': original})
    source = read_room_source(tmp_path, 1)
    assert source['rows'][0]['text'] == original and source['text'].count('\n') == 2
    manifest = resolve_evidence(['chat:1::lines=2-2'], active_root=tmp_path, allowed_roots=[],
                                resolve_chat=chat_evidence_reader(tmp_path))
    assert json.loads(manifest['attached'][0]['text'])['text'] == original


def test_steered_delivery_joins_exact_source_id_and_keeps_changed_text(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.owner_mailbox import write_owner_message

    append_jsonl(tmp_path / 'logs/chat.jsonl', {'direction': 'in', 'chat_id': 1,
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


def test_room_source_signs_old_outgoing_rows_by_the_activation_lineage_epoch(tmp_path):
    """Before the chronicle is active every lineage-free outgoing row reads as mine, as it always
    did. Once the activation records the lineage epoch, a row before it is mine only by its task
    result; rows after it, and the reader's other rows, keep their signature."""
    import json

    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.task_result_schema import SCHEMA_VERSION_KEY, TASK_RESULT_SCHEMA_VERSION
    from ouroboros.task_results import task_result_path

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
        append_jsonl(tmp_path / "logs" / "chat.jsonl", row)
    for task_id, fields in (("root0001", {}), ("kid00002", {"parent_task_id": "root0001", "root_task_id": "root0001",
                                                             "delegation_role": "subagent"})):
        task_result_path(tmp_path, task_id).write_text(json.dumps({
            SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION, "task_id": task_id, "status": "completed", **fields}),
            encoding="utf-8")

    def authors():
        return [row["author"] for row in read_room_source(tmp_path, 1)["rows"]]

    human, child = authors()[0], "child kid00003 of root0002"
    assert authors() == [human, "Ouroboros", "Ouroboros", "Ouroboros", child, "Ouroboros"]
    assert not (tmp_path / "memory" / "chronicle").exists()  # the reader never creates the chronicle
    assert ChronicleStore(tmp_path).ensure_activated()["metadata"]["lineage_epoch"]["pos"] == 4
    assert authors() == [human, "outgoing, author not recorded", "child kid00002 of root0001", "Ouroboros", child,
                         "Ouroboros"]
