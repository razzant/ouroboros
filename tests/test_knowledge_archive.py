"""Reversible archival through the real tools, next context and source reader.

All shelves, task artifacts and model messages are synthetic. No model call or
live memory is needed to observe what the next ContextFit request would send.
"""

import json

import pytest

from ouroboros import consolidator, context, knowledge as store
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.memory import Memory
from ouroboros.presence_context import build_presence_context_section
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests import test_book_context_capture as book


TOPIC = "episodes/река"
SUMMARY = "UNIQUE OLD EPISODE SUMMARY"
RAW = (f"---\r\ntype: experience\r\nsummary: {SUMMARY}\r\n"
       "custom: {unknown: [α, β]}\r\n---\r\n\r\n# Река\r\n"
       "Старый берег.\r\n[Other](../other.md)\r\n").encode()


def body(note):
    return note.raw[note.source.body_span.start_byte:]


def receipt(reply):
    assert reply.startswith("✅"), reply
    return json.loads(reply.split("\n", 1)[1])


def history(address):
    path = address.shelf.parent / "knowledge_history.jsonl"
    return [json.loads(line) for line in path.read_bytes().decode().split("\n") if line]


def previous_bytes(root, task_id, ref):
    payload = json.loads(read_actor_source_bytes(root, task_id, ref))
    assert payload["revision"] == ref["revision"]
    return payload[ref["field"]].encode("utf-8")


def read_previous(registry, ref):
    reply = registry.execute(ref["read"]["tool"], ref["read"]["arguments"])
    return json.loads(reply.split("\n", 1)[1])[ref["field"]].encode("utf-8")


def setup(tmp_path, scope="global", fork=False):
    drive = tmp_path / "fork" if fork else tmp_path
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive, task_id="archive-test",
                      budget_drive_root=str(tmp_path), project_id=scope.partition(":")[2])
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=drive)
    registry.set_context(ctx)
    address = store.resolve_knowledge_address(tmp_path, TOPIC, scope)
    address.path.parent.mkdir(parents=True, exist_ok=True)
    address.path.write_bytes(RAW)
    return ctx, registry, address, store.read_knowledge_note(address)


def transition(registry, note, mode="archive", **extra):
    return registry.execute("knowledge_write", {"topic": note.address.topic,
        "scope": note.address.scope, "mode": mode, "expected_revision": note.revision,
        **({"reason": "Meaning retained in an ordinary overview."} if mode == "archive" else {}), **extra})


@pytest.mark.parametrize("scope", ["global", "project:demo"])
@pytest.mark.parametrize("overview", [False, True])
@pytest.mark.parametrize("index_exists", [False, True])
def test_archive_reaches_next_context_list_and_index_without_mutating_a_capture(
        tmp_path, monkeypatch, scope, overview, index_exists):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "data")
    project = scope.partition(":")[2]
    env, _, task = book._capture(tmp_path, {"project_id": project})
    ctx, registry, address, original = setup(env.drive_root, scope)
    # An ordinary semantic overview is itself an active note; its summary carries
    # meaning, and the source link remains valid independently of archival state.
    semantic = store.resolve_knowledge_address(env.drive_root, "experience", scope)
    assert store.write_knowledge_note(semantic,
        "---\nsummary: RETAINED MEANING WITH LIMITS\n---\n"
        "What the experience taught me: [source](episodes/река.md).\n").ok
    if overview:
        assert store.write_knowledge_note(store.resolve_knowledge_address(env.drive_root, "overview"),
                                         "Shared authored orientation.").ok
    if not index_exists:
        (address.shelf / store.INDEX_FILE).unlink()
    memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
    earlier = context._capture_context_core(env, memory, task, None, None)
    old_plan = book._plan(env, earlier, task, "max")
    old_messages = json.dumps(old_plan.messages_for("max"))
    assert SUMMARY in old_messages
    result = receipt(transition(registry, original))
    assert result["knowledge_state"] == "archived"
    assert result["knowledge_delta"]["body_changed"] is False
    archived = store.read_knowledge_note(address)
    assert body(archived) == body(original)
    archive = archived.metadata["archive"]
    for text in ((address.shelf / store.INDEX_FILE).read_text(),
                 registry.execute("knowledge_list", {"scope": scope})):
        assert SUMMARY not in text and f"**{TOPIC}**" not in text
        assert archive["at"] not in text and archive["reason"] not in text
        assert "RETAINED MEANING WITH LIMITS" in text and "Archived notes: 1" in text
        assert f"scope={scope!r}, view='archived'" in text
    for view in ("archived", "all"):
        listing = registry.execute("knowledge_list", {"scope": scope, "view": view})
        assert SUMMARY in listing and f"**{TOPIC}**" in listing and "(archived)" in listing
        assert f"Archived at: {archive['at']}" in listing and f"Reason: {archive['reason']}" in listing
        assert ("RETAINED MEANING WITH LIMITS" in listing) == (view == "all")
    fresh = context._capture_context_core(env, memory, task, None, None)
    plan = book._plan(env, fresh, task, "max")
    for mode in ("max", "low", "nano"):
        messages = json.dumps(plan.messages_for(mode))
        assert SUMMARY not in messages and "Archived notes: 1" in messages
        assert archive["at"] not in messages and archive["reason"] not in messages
        assert "RETAINED MEANING WITH LIMITS" in messages
    assert json.dumps(old_plan.messages_for("max")) == old_messages
    assert SUMMARY in json.dumps(book._plan(env, earlier, task, "max").messages_for("max"))
    link = store.knowledge_links(store.read_knowledge_note(semantic))[0]
    assert link["status"] == "present" and link["address"]["path"] == str(address.path)
    current_read = registry.execute(link["read"]["tool"], link["read"]["arguments"])
    assert '"state": "archived"' in current_read and "Старый берег." in current_read
    assert {row["topic"] for row in store.inventory_knowledge(address)} >= {TOPIC, "experience"}


@pytest.mark.parametrize("scope", ["global", "project:demo"])
def test_archive_edit_restore_preserves_current_bytes_and_previous_sources(tmp_path, scope):
    ctx, registry, address, original = setup(tmp_path, scope, fork=True)
    archived_receipt = receipt(transition(registry, original))
    archived = store.read_knowledge_note(address)
    assert archived.metadata == {**original.metadata, "archive": archived.metadata["archive"]}
    previous = archived_receipt["knowledge_previous_source"]
    assert previous["revision"] == original.revision
    assert previous_bytes(ctx.drive_root, ctx.task_id, previous) == RAW
    # Exercise the actor's actual reader, including a project shelf from a fork.
    assert read_previous(registry, previous) == RAW
    assert history(address)[-1]["old_content"].encode() == RAW
    assert history(address)[-1]["new_content"].encode() == archived.raw
    assert body(archived) == body(original)
    for args in ({"mode": "edit", "old_str": "Старый", "content": "Новый"},
                 {"mode": "append", "content": "Added later.\r\n"},
                 {"mode": "edit", "summary": "REVISED ARCHIVED SUMMARY"}):
        receipt(registry.execute("knowledge_write", {"topic": TOPIC, "scope": scope,
                    "expected_revision": archived.revision, **args}))
        archived = store.read_knowledge_note(address)
        assert archived.state == "archived"
        assert "REVISED ARCHIVED SUMMARY" not in registry.execute("knowledge_list", {"scope": scope})
    # Full-content overwrite carrying the unchanged lifecycle is legitimate.
    receipt(registry.execute("knowledge_write", {"topic": TOPIC, "scope": scope,
        "content": archived.text.replace("Added later.", "Changed later."), "expected_revision": archived.revision}))
    edited = store.read_knowledge_note(address)
    restored_receipt = receipt(transition(registry, edited, "restore"))
    restored = store.read_knowledge_note(address)
    assert restored.state == "active" and "archive" not in restored.metadata
    assert body(restored) == body(edited)
    assert restored.summary == "REVISED ARCHIVED SUMMARY" and "Новый берег." in restored.text
    assert "Changed later." in restored.text
    assert "REVISED ARCHIVED SUMMARY" in registry.execute("knowledge_list", {"scope": scope})
    assert previous_bytes(ctx.drive_root, ctx.task_id, restored_receipt["knowledge_previous_source"]) == edited.raw
    assert history(address)[-1]["old_content"].encode() == edited.raw
    assert history(address)[-1]["new_content"].encode() == restored.raw
    # The previous handle still names the original, not the newer current note.
    assert previous_bytes(ctx.drive_root, ctx.task_id, previous) == RAW


@pytest.mark.parametrize("mode", ["archive", "restore"])
def test_revision_and_input_refusals_leave_source_history_and_index_unchanged(tmp_path, mode):
    _, registry, address, original = setup(tmp_path)
    if mode == "restore":
        receipt(transition(registry, original))
        original = store.read_knowledge_note(address)
    before = {p: p.read_bytes() for p in address.shelf.parent.rglob("*") if p.is_file()}
    for extra, expected in [({"expected_revision": None}, "revision_required"),
                            ({"expected_revision": "stale"}, "revision_conflict"),
                            ({"expected_revision": ""}, "revision_conflict"),
                            ({"content": "Do not replace me"}, "take no content"),
                            ({"summary": "Surprise"}, "summary is used only"),
                            ({"old_str": "Old"}, "old_str is used only")]:
        reply = transition(registry, original, mode, **extra)
        assert expected in reply, reply
    for reason in ("", "  ", None):
        assert "reason is required" in transition(registry, original, "archive", reason=reason)
    after = {p: p.read_bytes() for p in address.shelf.parent.rglob("*") if p.is_file()}
    # A lock sidecar can be created on the first refusal; no source/history/index changes.
    assert {p: v for p, v in after.items() if not p.name.endswith(".lock")} == {
        p: v for p, v in before.items() if not p.name.endswith(".lock")}
    missing = registry.execute("knowledge_write", {"topic": "missing", "mode": mode,
        "expected_revision": "", **({"reason": "Absent"} if mode == "archive" else {})})
    assert "lifecycle_source_missing" in missing
    assert not (address.shelf / "missing.md").exists()


@pytest.mark.parametrize("topic", ["overview", "patterns", "improvement-backlog"])
def test_reserved_roots_cannot_be_hidden_even_by_imported_archive_metadata(tmp_path, topic):
    _, registry, _, _ = setup(tmp_path)
    address = store.resolve_knowledge_address(tmp_path, topic)
    raw = b"---\narchive: {at: yesterday, reason: old}\n---\nIndependent loader body.\n"
    address.path.write_bytes(raw)
    note = store.read_knowledge_note(address)
    assert "must stay active" in transition(registry, note)
    with pytest.raises(ValueError, match="must stay active"):
        store.write_knowledge_note(address, "", "archive", note.revision, reason="why")
    assert note.state == "active" and note.archive_error
    assert f"**{topic}**" in registry.execute("knowledge_list", {})
    assert address.path.read_bytes() == raw
    restored = receipt(transition(registry, note, "restore"))
    current = store.read_knowledge_note(address)
    assert current.state == "active" and not current.archive_error
    assert "archive" not in current.metadata and body(current) == body(note)
    assert read_previous(registry, restored["knowledge_previous_source"]) == raw


@pytest.mark.parametrize("metadata", ["archive: null", "archive: []", "archive: old",
                                     "archive: {at: yesterday, reason: 7}", "archive: {reason: old}"])
@pytest.mark.parametrize("scope", ["global", "project:demo"])
def test_malformed_archive_metadata_is_visible_and_revision_checked_restore_repairs_it(tmp_path, metadata, scope):
    _, registry, address, _ = setup(tmp_path, scope)
    raw = (f"---\r\n{metadata}\r\ncustom: {{unknown: [α, β]}}\r\n"
           "summary: Current understanding.\r\n---\r\n# Evidence\r\nStill available.\r\n").encode()
    address.path.write_bytes(raw)
    note = store.read_knowledge_note(address)
    assert note.state == "active" and note.archive_error and not note.parse_error
    assert TOPIC in registry.execute("knowledge_list", {"scope": scope})
    assert "Still available" in registry.execute("knowledge_read", {"topic": TOPIC, "scope": scope})
    assert "lifecycle requires readable, valid metadata" in transition(registry, note)
    assert "revision_required" in transition(registry, note, "restore", expected_revision=None)
    assert "revision_conflict" in transition(registry, note, "restore", expected_revision="stale")
    assert address.path.read_bytes() == raw
    # An ordinary edit remains legal, but cannot repair the lifecycle field.
    receipt(registry.execute("knowledge_write", {"topic": TOPIC, "scope": scope, "mode": "edit",
        "old_str": "Still available.", "content": "Later corrected facts.", "expected_revision": note.revision}))
    assert "revision_conflict" in transition(registry, note, "restore")
    edited = store.read_knowledge_note(address)
    repaired = receipt(transition(registry, edited, "restore"))
    current = store.read_knowledge_note(address)
    assert body(current) == body(edited) and "Later corrected facts." in current.text
    assert current.metadata == {key: value for key, value in edited.metadata.items() if key != "archive"}
    assert current.state == "active" and not current.archive_error
    assert repaired["knowledge_delta"]["body_changed"] is False
    assert read_previous(registry, repaired["knowledge_previous_source"]) == edited.raw
    assert history(address)[-1]["old_content"].encode() == edited.raw
    assert history(address)[-1]["new_content"].encode() == current.raw


def test_malformed_whole_yaml_remains_readable_visible_and_not_mutated_by_lifecycle(tmp_path):
    _, registry, address, _ = setup(tmp_path)
    raw = b"---\narchive: [unfinished\n---\n# Evidence\nStill available.\n"
    address.path.write_bytes(raw)
    note = store.read_knowledge_note(address)
    assert note.state == "active"
    assert TOPIC in registry.execute("knowledge_list", {})
    assert "Still available" in registry.execute("knowledge_read", {"topic": TOPIC})
    for mode in ("archive", "restore"):
        assert "lifecycle requires readable, valid metadata" in transition(registry, note, mode)
    assert address.path.read_bytes() == raw


def test_ordinary_overwrites_and_nominations_cannot_author_or_remove_lifecycle(tmp_path):
    ctx, registry, address, original = setup(tmp_path)
    for field in ("archive: null", "archive: {at: now, reason: old}"):
        reply = registry.execute("knowledge_write", {"topic": TOPIC,
            "content": f"---\n{field}\n---\nOops", "expected_revision": original.revision})
        assert "archive is owned" in reply
        reply = registry.execute("knowledge_write", {"topic": "new", "content": f"---\n{field}\n---\nOops"})
        assert "archive is owned" in reply
    receipt(transition(registry, original))
    archived = store.read_knowledge_note(address)
    assert "archive is owned" in registry.execute("knowledge_write", {"topic": TOPIC,
        "content": "---\narchive: null\n---\nOops", "expected_revision": archived.revision})
    # Omission merges retained metadata; it does not reactivate the source.
    receipt(registry.execute("knowledge_write", {"topic": TOPIC, "content": "New body.",
                                                   "expected_revision": archived.revision}))
    archived = store.read_knowledge_note(address)
    entries = [{"topic": TOPIC, "scope": "global", "expected_revision": archived.revision,
                "edits": [{"old_text": "New body.", "new_text": "Nominated correction.", "basis": "Read source."}],
                "summary": "NOMINATED SUMMARY"}]
    outcomes = consolidator._write_knowledge_entries(address.shelf, entries, context=ctx)
    assert outcomes[0]["ok"], outcomes
    assert store.read_knowledge_note(address).state == "archived"
    assert "NOMINATED SUMMARY" not in registry.execute("knowledge_list", {})
    assert "NOMINATED SUMMARY" in registry.execute("knowledge_list", {"view": "archived"})


@pytest.mark.parametrize("mode", ["archive", "restore"])
def test_history_capture_failure_precedes_publication(tmp_path, monkeypatch, mode):
    _, registry, address, note = setup(tmp_path)
    receipt(transition(registry, note))
    note = store.read_knowledge_note(address)
    if mode == "archive":
        receipt(transition(registry, note, "restore"))
        note = store.read_knowledge_note(address)
    index = (address.shelf / store.INDEX_FILE).read_bytes()
    previous = history(address)
    monkeypatch.setattr(store, "append_jsonl", lambda *a, **kw: False)
    assert "history_unavailable" in transition(registry, note, mode)
    assert address.path.read_bytes() == note.raw
    assert (address.shelf / store.INDEX_FILE).read_bytes() == index
    assert history(address) == previous


def test_actor_source_failure_does_not_publish_note_history_or_index(tmp_path, monkeypatch):
    _, registry, address, note = setup(tmp_path)
    monkeypatch.setattr("ouroboros.artifacts.persist_exact_text_source",
                        lambda *a, **kw: ("", {}, {"reason": "source disk failed"}))
    assert "source_capture_unavailable" in transition(registry, note)
    assert address.path.read_bytes() == RAW
    assert not (address.shelf / store.INDEX_FILE).exists()
    assert not (address.shelf.parent / "knowledge_history.jsonl").exists()


@pytest.mark.parametrize("mode", ["archive", "restore"])
def test_partial_index_publication_repeated_operation_recovers_truthfully(tmp_path, monkeypatch, mode):
    _, registry, address, original = setup(tmp_path)
    receipt(transition(registry, original))
    note = store.read_knowledge_note(address)
    if mode == "archive":
        receipt(transition(registry, note, "restore"))
        note = store.read_knowledge_note(address)
    rebuild = store.rebuild_knowledge_index

    def fail(_):
        raise OSError("index offline")

    monkeypatch.setattr(store, "rebuild_knowledge_index", fail)
    assert "publication_incomplete" in transition(registry, note, mode)
    published = store.read_knowledge_note(address)
    assert body(published) == body(note)
    assert published.state == ("archived" if mode == "archive" else "active")
    assert history(address)[-1]["old_content"].encode() == note.raw
    assert history(address)[-1]["new_content"].encode() == published.raw
    count = len(history(address))
    assert "revision_conflict" in transition(registry, note, mode)
    assert "publication_incomplete" in transition(registry, published, mode)
    assert len(history(address)) == count
    # Readers use current files even while the persisted generated index is stale.
    listing = registry.execute("knowledge_list", {})
    assert (SUMMARY in listing) == (mode == "restore")
    monkeypatch.setattr(store, "rebuild_knowledge_index", rebuild)
    assert receipt(transition(registry, published, mode))["knowledge_write_reason"] == "unchanged"
    assert (SUMMARY in (address.shelf / store.INDEX_FILE).read_text()) == (mode == "restore")
    assert len(history(address)) == count and address.path.read_bytes() == published.raw


def test_legacy_index_prose_is_disclosed_until_authored_overview_and_presence_still_reads(tmp_path):
    _, registry, address, note = setup(tmp_path)
    legacy = f"Older prose about {TOPIC}: {SUMMARY}\n"
    (address.shelf / store.INDEX_FILE).write_text(legacy, encoding="utf-8", newline="")
    receipt(transition(registry, note))
    text = registry.execute("knowledge_list", {})
    assert legacy in text and "may still mention archived notes" in text
    assert f"**{TOPIC}**" not in text
    assert any(row.get("old_content") == legacy for row in history(address))
    receipt(registry.execute("knowledge_write", {"topic": "overview", "content": "Authored orientation."}))
    assert SUMMARY not in registry.execute("knowledge_list", {})
    # Explicit profile topics are independent reads; archival changes no profile authority.
    presence = build_presence_context_section(tmp_path, {"instructions": "Stay aware.", "event": {"type": "message"},
                                                        "context_topics": [TOPIC]})
    assert "Старый берег." in presence and SUMMARY in presence and "archive:" in presence


def test_archive_lists_are_read_only_and_bad_view_does_not_fall_back(tmp_path):
    _, registry, address, note = setup(tmp_path)
    receipt(transition(registry, note))
    (address.shelf / store.INDEX_FILE).unlink()
    before = {p: p.read_bytes() for p in address.shelf.parent.rglob("*") if p.is_file()}
    for view in ("active", "archived", "all"):
        registry.execute("knowledge_list", {"view": view})
    assert "view must be" in registry.execute("knowledge_list", {"view": "unknown"})
    assert before == {p: p.read_bytes() for p in address.shelf.parent.rglob("*") if p.is_file()}


@pytest.mark.parametrize("raw", [b"# Plain\r\nLegacy body.\r\n",
    b"---\ntype: note\ncustom: &loop [*loop]\n---\n# Recursive metadata\r\nBody.\r\n"])
def test_plain_and_recursive_yaml_notes_keep_their_body_and_unknown_metadata(tmp_path, raw):
    _, registry, address, _ = setup(tmp_path)
    address.path.write_bytes(raw)
    original = store.read_knowledge_note(address)
    receipt(transition(registry, original))
    archived = store.read_knowledge_note(address)
    assert body(archived) == body(original)
    # Summary revisions re-render YAML while preserving recursive unknown values.
    receipt(registry.execute("knowledge_write", {"topic": TOPIC, "mode": "edit", "summary": "New view.",
                                                  "expected_revision": archived.revision}))
    edited = store.read_knowledge_note(address)
    receipt(transition(registry, edited, "restore"))
    restored = store.read_knowledge_note(address)
    assert body(restored) == body(original)
    assert restored.summary == "New view." and restored.state == "active"
    if "custom" in original.metadata:
        assert restored.metadata["custom"][0] is restored.metadata["custom"]


def test_failed_source_replacement_returns_actual_old_note_with_durable_capture(tmp_path, monkeypatch):
    _, registry, address, original = setup(tmp_path)
    write_bytes = store.write_bytes_atomic

    def fail_source(path, raw):
        if path == address.path:
            raise OSError("source is not writable")
        return write_bytes(path, raw)

    monkeypatch.setattr(store, "write_bytes_atomic", fail_source)
    reply = transition(registry, original)
    assert "publication_incomplete" in reply
    assert original.revision in reply and '"state": "active"' in reply
    assert address.path.read_bytes() == RAW
    assert history(address)[-1]["old_content"].encode() == RAW
    assert "archive:" in history(address)[-1]["new_content"]
    assert not (address.shelf / store.INDEX_FILE).exists()


def test_previous_source_handle_survives_existing_child_copy_and_cleanup(tmp_path):
    from ouroboros.headless import (copy_child_task_result, prepare_task_drive,
                                    remove_subagent_task_drive, retry_child_task_refs)
    from ouroboros.task_results import write_task_result
    from ouroboros.loop_tool_execution import _execute_single_tool

    ctx, registry, _, note = setup(tmp_path, "project:demo")
    child = prepare_task_drive(tmp_path, ctx.task_id, "empty")
    ctx.drive_root = child
    registry.set_context(ctx)
    call = {"id": "archive-call", "function": {"name": "knowledge_write", "arguments": json.dumps({
        "topic": TOPIC, "scope": note.address.scope, "mode": "archive", "expected_revision": note.revision,
        "reason": "Meaning retained."})}}
    executed = _execute_single_tool(registry, call, child / "logs", ctx.task_id)
    assert not executed["is_error"], executed
    result = receipt(executed["result"])
    previous = result["knowledge_previous_source"]
    write_task_result(child, ctx.task_id, "completed")
    copy_child_task_result(tmp_path, {"id": ctx.task_id, "drive_root": str(child)})
    copied = retry_child_task_refs(tmp_path, child, ctx.task_id)
    assert copied["child_ref_promotion"]["promoted_source_handle_count"] == 1
    assert remove_subagent_task_drive(tmp_path, ctx.task_id, live=lambda _: False)
    assert not child.exists()
    assert previous_bytes(tmp_path, ctx.task_id, previous) == RAW
    ctx.drive_root = tmp_path
    registry.set_context(ctx)
    assert read_previous(registry, previous) == RAW


def test_external_result_metadata_lookalikes_do_not_promote_knowledge_sources(tmp_path):
    from types import SimpleNamespace

    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.observability import read_call_payload
    from ouroboros.review_source_closure import retain_review_refs
    from ouroboros.tools.extension_dispatch import _extension_completion

    child, custody = tmp_path / "child", tmp_path / "custody"
    ctx, registry, _, note = setup(child)
    previous = receipt(transition(registry, note))["knowledge_previous_source"]
    lookalikes = {"knowledge_previous_source": previous, "source_ref": previous}
    payload = json.dumps({"result_meta": {**lookalikes, "tool_result_meta": lookalikes},
                          "tool_result_meta": lookalikes})
    typed = _extension_completion(payload, "Host annotation.")
    tools = SimpleNamespace(_ctx=ctx, CODE_TOOLS=frozenset(), execute_result=lambda *_: typed)
    executed = _execute_single_tool(tools, {
        "id": "external-data", "function": {"name": "ext_data", "arguments": "{}"}},
        child / "logs", ctx.task_id)
    assert not executed["is_error"], executed
    call_id = executed["trace_ref"]["call_id"]
    _, recorded, _ = read_call_payload(child, task_id=ctx.task_id, call_id=call_id)
    assert recorded["producer_result"] == payload
    assert recorded["result"] == typed.text
    # _typed_result_metadata copies the dispatcher's typed facts, not JSON keys
    # in either the external producer body or the annotated result body.
    assert recorded["result_meta"]["tool_result_meta"] == dict(typed.meta)
    assert previous["sha256"] not in json.dumps(recorded["result_meta"])
    retain_review_refs({"trace_refs": {"tool_call_refs": [executed["trace_ref"]]}},
                       child, custody, ctx.task_id, carrier="task_result")
    _, retained, _ = read_call_payload(custody, task_id=ctx.task_id, call_id=call_id)
    assert retained == recorded
    assert not (task_artifact_dir_path(custody, ctx.task_id) / previous["path"]).exists()
    with pytest.raises(FileNotFoundError):
        read_actor_source_bytes(custody, ctx.task_id, previous)
    assert previous_bytes(child, ctx.task_id, previous) == RAW


def test_presence_payload_metadata_lookalikes_stay_data_in_retained_context(tmp_path):
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path
    from ouroboros.review_source_closure import retain_review_refs

    child, custody = tmp_path / "child", tmp_path / "custody"
    ctx, registry, _, note = setup(child)
    previous = receipt(transition(registry, note))["knowledge_previous_source"]
    lookalikes = {"knowledge_previous_source": previous, "source_ref": previous}
    event = {"type": "message", "message": {
        "result_meta": {**lookalikes, "tool_result_meta": lookalikes},
        "tool_result_meta": lookalikes}}
    rendered = build_presence_context_section(child, {"instructions": "Stay aware.", "event": event})
    assert previous["sha256"] in rendered and '"tool_result_meta"' in rendered
    captured = json.dumps({"messages": [{"role": "system", "content": rendered}],
        "selection_fingerprint": "presence-data", "observed_view_revision": "view-1",
        "selected_unit_ids": []}).encode()
    checkpoint = store_actor_source_bytes(child, ctx.task_id, category="context_checkpoints",
        source_id="presence-data", data=captured, extension="json")
    # source_carrier follows the host checkpoint and its message roles, but the
    # rendered Presence event cannot nominate its own source closure.
    retained = retain_review_refs({"checkpoint_ref": checkpoint}, child, custody, ctx.task_id)
    assert read_actor_source_bytes(custody, ctx.task_id, retained["checkpoint_ref"]) == captured
    assert not (task_artifact_dir_path(custody, ctx.task_id) / previous["path"]).exists()
    with pytest.raises(FileNotFoundError):
        read_actor_source_bytes(custody, ctx.task_id, previous)
    assert previous_bytes(child, ctx.task_id, previous) == RAW


def test_tool_metadata_closure_retains_only_named_previous_note_source(tmp_path):
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path
    from ouroboros.review_source_closure import retain_review_refs

    child, custody, task = tmp_path / "child", tmp_path / "custody", "metadata-closure"
    previous = store_actor_source_bytes(child, task, category="tool_results",
        source_id="knowledge-previous", data=b"previous note", extension="txt")
    unrelated = store_actor_source_bytes(child, task, category="tool_results",
        source_id="unrelated", data=b"unrelated data", extension="txt")
    payload = {"result_meta": {"source_ref": unrelated, "refs": [unrelated],
        "tool_result_meta": {"knowledge_previous_source": previous,
                             "source_ref": unrelated, "sources": [unrelated]}}}
    retained = retain_review_refs(payload, child, custody, task)
    assert read_actor_source_bytes(custody, task,
        retained["result_meta"]["tool_result_meta"]["knowledge_previous_source"]) == b"previous note"
    assert not (task_artifact_dir_path(custody, task) / unrelated["path"]).exists()
    # Missing arbitrary metadata is data, not a reason to refuse review.
    (task_artifact_dir_path(child, task) / unrelated["path"]).unlink()
    assert retain_review_refs(payload, child, custody, task) == retained
