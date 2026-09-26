"""Light revises actual current knowledge in the same existing memory operation."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c, knowledge as k, reflection
from ouroboros.memory import Memory
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers
from tests.test_consolidator_context_fit import _paths, _write_chat

fit = fit_helpers.fit


def _address(root, topic="people/alex"):
    return k.resolve_knowledge_address(root, topic, "global")


def _call(topic="people/alex"):
    return {"id": "read-current", "type": "function", "function": {
        "name": "knowledge_read", "arguments": json.dumps({"topic": topic, "scope": "global"}),
    }}


class MemoryLLM:
    def __init__(self, answer, *, before_answer=None):
        self.answer, self.before_answer = answer, before_answer
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if kwargs["messages"][0]["content"].startswith("Compare this draft memory"):
            if kwargs["messages"][-1]["role"] != "tool":
                return {"content": "", "tool_calls": [_call()]}, {"cost": 0.01}
            # Corrected existing-note replacements require this operation's read.
            prompt = kwargs["messages"][0]["content"]
            block = prompt.split("## Draft memory", 1)[1].split("\n\n", 1)[0] if "## Draft memory" in prompt else ""
            nominations = block[block.index("KNOWLEDGE_ENTRIES_JSON:"):] if "KNOWLEDGE_ENTRIES_JSON:" in block else ""
            return {"content": "Checked interpretation." + ("\n" + nominations if nominations else "")}, {
                "prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.02}
        if len(self.calls) == 1:
            return {"content": "", "tool_calls": [_call()]}, {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}
        if self.before_answer:
            self.before_answer()
        return {"content": self.answer}, {
            "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.02}


def _initial(root):
    return k.write_knowledge_note(_address(root),
        "---\ntype: understanding\nsummary: Brevity may depend on context.\ncustom: retained\n---\n"
        "# Alex\n\nHe asked for brevity while hurried.\nDECISIVE ORIGINAL TAIL.\n").current


def test_same_light_operation_reads_complete_note_and_binds_actual_revision(tmp_path, fit, monkeypatch):
    original = _initial(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="memory-operation")
    reads = c.KnowledgeReadContext(ctx)
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("claudexor::codex=test", False))
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"light":"kept-account"}')
    llm = MemoryLLM("Completed interpretation.")
    content, usage = c._call_consolidation_llm(llm, "ORIGINAL EPISODE: today he asked for depth.",
                                              "test memory", knowledge=reads)
    assert content == "Completed interpretation."
    assert usage["cost"] == pytest.approx(0.03)
    assert usage["prompt_tokens"] == 30 and usage["completion_tokens"] == 15
    assert len(llm.calls) == 2
    assert llm.calls[0]["model_turn_state"] is llm.calls[1]["model_turn_state"]
    assert all(call["model_account_override"] == "kept-account" for call in llm.calls)
    assert all(call["model_role"] == "light" for call in llm.calls)
    second = llm.calls[1]["messages"]
    assert "ORIGINAL EPISODE" in second[0]["content"]
    assert second[-1]["content"].endswith(original.text)
    assert "DECISIVE ORIGINAL TAIL." in second[-1]["content"]
    entries = reads.bind_entries([{"topic": "people/alex", "scope": "global", "edits": [{
        "old_text": "He asked for brevity while hurried.", "new_text": "He asked for depth today.",
        "basis": "Today's request corrected the preference."}], "expected_revision": "invented"}])
    assert entries[0]["expected_revision"] == original.revision
    assert c._write_knowledge_entries(original.address.shelf, entries, context=ctx)[0]["ok"]
    assert k.read_knowledge_note(original.address).raw == original.raw.replace(
        b"He asked for brevity while hurried.", b"He asked for depth today.")
    # A complete read of the current note cannot authorize a whole-note replacement instead.
    reads.reads[("global", "people/alex")] = k.read_knowledge_note(original.address).revision
    whole = reads.bind_entries([{"topic": "people/alex", "scope": "global", "content": "Revised."}])
    before = k.read_knowledge_note(original.address).raw
    outcome = c._write_knowledge_entries(original.address.shelf, whole, context=ctx)[0]
    assert outcome == {"topic": "people/alex", "scope": "global", "ok": False, "reason": "existing_note_requires_edits"}
    assert k.read_knowledge_note(original.address).raw == before


def test_unread_existing_note_is_preserved_while_new_note_can_be_created(tmp_path, fit):
    original = _initial(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    reads = c.KnowledgeReadContext(ctx)
    entries = reads.bind_entries([
        {"topic": "people/alex", "content": "Unseen replacement.", "expected_revision": original.revision},
        {"topic": "new idea", "content": "A new authored observation."},
    ])
    results = c._write_knowledge_entries(original.address.shelf, entries, context=ctx)
    assert results[0]["reason"] == "revision_required" and not results[0]["ok"]
    assert results[1]["ok"]
    assert k.read_knowledge_note(original.address).raw == original.raw


def test_backlog_keeps_its_dedicated_merge_not_the_ordinary_edit_contract(tmp_path, fit):
    item = "### ibl-ordinary-notes\n- summary: Keep cumulative notes intact.\n- category: memory\n"
    first = c._write_knowledge_entries(tmp_path / "memory" / "knowledge", [
        {"topic": "improvement-backlog", "content": item}])
    second = c._write_knowledge_entries(tmp_path / "memory" / "knowledge", [
        {"topic": "improvement-backlog", "content": item}])
    assert first[0]["ok"] and second[0]["ok"]
    assert first[0]["reason"] == second[0]["reason"] == "backlog_merge"
    edits = c._write_knowledge_entries(tmp_path / "memory" / "knowledge", [
        {"topic": "improvement-backlog", "edits": [{"old_text": "a", "new_text": "b", "basis": "c"}]}])
    assert edits == [{"topic": "improvement-backlog", "scope": "global", "ok": False, "reason": "unparseable_backlog"}]


def test_every_nomination_keeps_one_positional_outcome_and_its_host_stamp(tmp_path, fit):
    original = _initial(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="positional")
    edit = {"old_text": "DECISIVE ORIGINAL TAIL.", "new_text": "DECISIVE ORIGINAL TAIL. Later: depth.",
            "basis": "The new episode adds a context-specific request."}
    read = {"topic": "people/alex", "expected_revision": original.revision}
    bad_edits = "invalid_nomination: edits must be a list of {old_text, new_text, basis}"
    bad_summary = "invalid_nomination: summary must be non-empty text"
    legacy = "invalid_nomination: frontmatter is not an automatic field; revise the summary with summary"
    ambiguous = "ambiguous_nomination: content creates a new note; edits and summary change a read one"
    entries = ["not a nomination", {"topic": "people/alex", "content": ""},
               {**read, "content": "Whole replacement."},
               {**read, "edits": [{**edit, "basis": " "}]},
               # A present key is judged by shape: falsey malformed values never become an empty list.
               *({**read, "edits": bad} for bad in (None, "", {}, 0, False)),
               {"topic": "fresh-null", "content": "A new note.", "edits": None},
               *({**read, "edits": [edit], "summary": bad} for bad in (None, "", " ", 5, ["x"])),
               # Legacy generic metadata is refused, never silently dropped.
               {**read, "edits": [edit], "frontmatter": {"summary": "Legacy generic metadata."}},
               {**read, "edits": [], "frontmatter": {}},
               {**read, "content": "Whole replacement.", "edits": [edit]},
               {**read, "content": "Whole replacement.", "summary": "Also a summary."},
               {"topic": "fresh-ambiguous", "content": "A new note.", "edits": [edit]},
               {**read, "edits": [edit], "summary": "Depth depends on context.", "content": "",
                "_nomination_route": {"provider": "p", "model": "m"}},
               {"topic": "fresh", "content": "A new note."},
               # An absent or empty edit list beside new-note content is the compatible create form.
               {"topic": "fresh-too", "content": "Another new note.", "edits": []}]
    outcomes = c._write_knowledge_entries(original.address.shelf, entries, context=ctx,
                                          stamp={"writer": "consolidation", "route": "block", "writer_input_ref": {"id": 1}})
    assert [row["reason"] for row in outcomes] == [
        "malformed_nomination", "empty_nomination", "existing_note_requires_edits",
        "invalid_note: edit 1 needs string old_text and new_text and a non-empty basis",
        *[bad_edits] * 6, *[bad_summary] * 5, legacy, legacy, ambiguous, ambiguous, ambiguous,
        "saved", "saved", "saved"]
    current = k.read_knowledge_note(original.address)
    assert current.metadata == {"type": "understanding", "summary": "Depth depends on context.", "custom": "retained"}
    assert current.text.endswith("# Alex\n\nHe asked for brevity while hurried.\nDECISIVE ORIGINAL TAIL. Later: depth.\n")
    assert not (original.address.shelf / "fresh-null.md").exists()
    assert not (original.address.shelf / "fresh-ambiguous.md").exists()
    rows = [json.loads(line) for line in (tmp_path / "memory" / "knowledge_history.jsonl").read_text().splitlines()]
    assert [row["mode"] for row in rows if row.get("topic") == "people/alex"] == ["overwrite", "edit"]
    change = rows[-3]
    assert change["edits"] == [edit] and change["summary"] == "Depth depends on context."
    assert (change["mode"], change["writer"], change["route"], change["writer_input_ref"]) == (
        "edit", "consolidation", {"provider": "p", "model": "m"}, {"id": 1})
    assert [("edits" in row, "summary" in row, row["route"]) for row in rows[-2:]] == [(False, False, "block")] * 2
    # Settlement is positional: only the three published entries retire their debt.
    from ouroboros.memory_nomination_receipts import prepare, settle
    meta = {}
    ids = prepare(meta, "source", [(None, entries)])
    settle(meta, ids, outcomes)
    assert {row["id"] for row in meta["pending_knowledge_nominations"]} == {
        f"source:0:{i}" for i in range(len(entries) - 3)}


def test_summary_revision_keeps_recursive_yaml_and_the_rest_of_its_batch(tmp_path):
    target = _address(tmp_path, "recursive")
    target.path.parent.mkdir(parents=True)
    # Legal YAML may alias a node inside itself; re-rendering the preamble for a
    # revised summary must keep that graph and every later nomination's outcome.
    target.path.write_bytes(b"---\ntype: note\nsummary: Old view.\ncustom: &loop [*loop]\n---\nOld.\nKept.\n")
    original = k.read_knowledge_note(target)
    edit = [{"old_text": "Old.", "new_text": "New.", "basis": "This episode revised it."}]
    entries = [{"topic": "recursive", "expected_revision": original.revision, "edits": edit, "summary": "New view."},
               {"topic": "fresh-ambiguous", "content": "A new note.", "edits": edit},
               {"topic": "fresh", "content": "A following note."}]
    outcomes = c._write_knowledge_entries(target.shelf, entries, stamp={"writer": "consolidation"})
    assert [row["reason"] for row in outcomes] == [
        "saved", "ambiguous_nomination: content creates a new note; edits and summary change a read one", "saved"]
    current = k.read_knowledge_note(target)
    loop = current.metadata["custom"]
    assert len(loop) == 1 and loop[0] is loop
    assert (current.metadata["type"], current.metadata["summary"]) == ("note", "New view.")
    assert current.text.endswith("---\nNew.\nKept.\n")
    assert (target.shelf / "fresh.md").exists() and not (target.shelf / "fresh-ambiguous.md").exists()
    rows = [json.loads(line) for line in (tmp_path / "memory" / "knowledge_history.jsonl").read_text().splitlines()]
    assert [(row["topic"], row["mode"]) for row in rows] == [("recursive", "edit"), ("fresh", "overwrite")]
    assert (rows[0]["old_content"], rows[0]["new_content"]) == (original.text, current.text)
    assert (rows[0]["edits"], rows[0]["summary"], rows[0]["writer"]) == (edit, "New view.", "consolidation")
    from ouroboros.memory_nomination_receipts import prepare, settle
    meta = {}
    settle(meta, prepare(meta, "source", [(None, entries)]), outcomes)
    assert [row["id"] for row in meta["pending_knowledge_nominations"]] == ["source:0:1"]
    # The same summary and a body-only edit keep the re-rendered preamble bytes.
    for extra, old, new in (({"summary": "New view."}, "New.", "Newer."), ({}, "Kept.", "Still kept.")):
        before = k.read_knowledge_note(target)
        again = c._write_knowledge_entries(target.shelf, [{"topic": "recursive", "expected_revision": before.revision,
            "edits": [{"old_text": old, "new_text": new, "basis": "Same episode."}], **extra}])
        assert again[0]["reason"] == "saved"
        assert k.read_knowledge_note(target).raw == before.raw.replace(old.encode(), new.encode())
    assert k.read_knowledge_note(target).raw.startswith(current.raw[:current.source.body_span.start_byte])


def _scratchpad(root):
    memory = Memory(root, root)
    memory.mutate_scratchpad_blocks(lambda _: [
        {"ts": f"2026-09-12T10:0{index}:00Z", "source": "dialogue",
         "content": f"ORIGINAL EPISODE {index}: asked for depth. " + "x" * 11000}
        for index in range(3)])
    return memory


@pytest.mark.parametrize("concurrent", [False, True])
def test_scratchpad_source_and_failed_revisions_remain_durable(tmp_path, fit, concurrent):
    original = _initial(tmp_path)
    memory = _scratchpad(tmp_path)
    update = "Two original episodes, context differs."
    edits = [{"old_text": "He asked for brevity while hurried.", "new_text": update,
              "basis": "The original episodes establish context-specific requests."}]
    answer = json.dumps({"knowledge_entries": [{"topic": "people/alex", "edits": edits,
                                                "summary": "Brevity while hurried; depth while exploring."}],
                         "compressed_block": "I learned that context matters; keep both episodes."})
    competing = lambda: k.write_knowledge_note(original.address, "A simultaneous newer observation.",
                                               expected_revision=original.revision)
    llm = MemoryLLM(answer, before_answer=competing if concurrent else None)
    usage = c.consolidate_scratchpad(memory, original.address.shelf, llm)
    assert usage["cost"] == pytest.approx(0.03)
    assert "ORIGINAL EPISODE 0" in llm.calls[1]["messages"][0]["content"]
    assert "DECISIVE ORIGINAL TAIL." in llm.calls[1]["messages"][-1]["content"]
    blocks = memory.load_scratchpad_blocks()
    assert len(blocks) == 2
    assert blocks[0]["metadata"]["knowledge_writes"][0]["ok"] is not concurrent
    source = memory.journal_path().read_text()
    assert "ORIGINAL EPISODE 0" in source and json.loads(next(
        line for line in source.splitlines() if json.loads(line).get("type") == "blocks_consolidated"))["knowledge_entries"][0]["edits"] == edits
    current = k.read_knowledge_note(original.address)
    assert ("simultaneous newer" if concurrent else "Two original episodes") in current.text
    if not concurrent:  # the summary changes through the ordinary merge; unknown fields and other bytes stay
        assert current.metadata == {**original.metadata, "summary": "Brevity while hurried; depth while exploring."}
        assert current.text.endswith("# Alex\n\nTwo original episodes, context differs.\nDECISIVE ORIGINAL TAIL.\n")
    if concurrent:
        assert "not published" in blocks[0]["content"]


def test_dialogue_consolidation_retains_nominations_and_commits_shared_note(tmp_path, fit):
    original = _initial(tmp_path)
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, text_size=0)
    answer = "### Block: episode\nI learned why the requested depth changes.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps([
        {"topic": "people/alex", "edits": [{"old_text": "He asked for brevity while hurried.",
            "new_text": "Current understanding with original episode evidence.",
            "basis": "The episode established a more precise preference."}]}])
    llm = MemoryLLM(answer)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="dialogue-memory")
    usage = c.consolidate(chat, blocks, meta, llm, knowledge_context=ctx)
    assert usage["cost"] == pytest.approx(0.06)  # draft read/answer, correction read/answer
    block = json.loads(blocks.read_text())[0]
    assert "KNOWLEDGE_ENTRIES_JSON" not in block["content"]
    assert block["rooms"][0]["content"] == "Checked interpretation."
    source_id = block["knowledge_source_ref"]["entry_id"]
    rows = [json.loads(line) for line in (tmp_path / "memory" / "knowledge_history.jsonl").read_text().splitlines()]
    source = next(row for row in rows if row.get("entry_id") == source_id)
    assert source["nominations"][0]["entries"][0]["expected_revision"] == original.revision
    assert block["knowledge_writes"][0]["ok"]
    assert "Current understanding" in k.read_knowledge_note(original.address).text
    assert "DECISIVE ORIGINAL TAIL." in k.read_knowledge_note(original.address).text
    assert json.loads(meta.read_text())["last_consolidated_offset"] == 100
    assert "pending_knowledge_nominations" not in json.loads(meta.read_text())


def test_reflection_reads_current_note_preserves_full_update_and_counts_only_actual_write(tmp_path, fit):
    original = _initial(tmp_path)
    content = "Full revised understanding. " * 80 + "PRESERVE LAST SENTENCE."
    edits = [{"old_text": "He asked for brevity while hurried.", "new_text": content,
              "basis": "The full original task establishes revised understanding."}]
    answer = "Reflection.\nMEMORY_ACTIONS_JSON: " + json.dumps([
        {"type": "knowledge_write", "topic": "people/alex", "edits": edits}])
    llm = MemoryLLM(answer)
    entry = reflection.generate_reflection(
        {"id": "reflection-task", "text": "FULL ORIGINAL EPISODE " * 100, "drive_root": str(tmp_path)},
        {}, "trace", llm, {"rounds": 2, "cost": 0.1})
    assert entry["memory_actions"][0]["edits"] == edits and entry["memory_actions"][0]["content"] == ""
    assert entry["memory_actions"][0]["expected_revision"] == original.revision
    assert "FULL ORIGINAL EPISODE " * 100 in llm.calls[1]["messages"][0]["content"]
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    assert reflection.apply_memory_actions(env, entry["memory_actions"]) == 1
    assert k.read_knowledge_note(original.address).raw == original.raw.replace(
        b"He asked for brevity while hurried.", content.encode())
    assert reflection.apply_memory_actions(env, entry["memory_actions"]) == 0


def test_reflection_forwards_authored_change_keys_to_the_one_publisher_contract(tmp_path):
    original = _initial(tmp_path)
    edit = {"old_text": "He asked for brevity while hurried.", "new_text": "Depth today.", "basis": "This task."}
    skipped = []

    def on_skip(action, reason):
        skipped.append((action["type"], reason))

    # Present keys pass verbatim, however malformed; only an action with no change at all is empty.
    malformed = reflection._validate_memory_actions([
        {"type": "knowledge_write", "topic": "people/alex", "edits": None},
        {"type": "knowledge_write", "topic": "people/alex", "content": "Whole.", "edits": [edit]},
        {"type": "knowledge_write", "topic": "people/alex", "edits": [edit], "frontmatter": {"summary": "x"}},
    ], "t1", on_skip=on_skip)
    assert malformed == [
        {"type": "knowledge_write", "content": "", "task_id": "t1", "edits": None, "topic": "people/alex"},
        {"type": "knowledge_write", "content": "Whole.", "task_id": "t1", "edits": [edit], "topic": "people/alex"},
        {"type": "knowledge_write", "content": "", "task_id": "t1", "edits": [edit], "frontmatter": {"summary": "x"},
         "topic": "people/alex"}]
    assert reflection._validate_memory_actions([
        {"type": "knowledge_write", "topic": "people/alex", "content": " "},
        {"type": "scratchpad_append", "content": "", "edits": [edit]},
    ], "t1", on_skip=on_skip) == []
    assert skipped == [("knowledge_write", "empty_content"), ("scratchpad_append", "empty_content")]
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    assert reflection.apply_memory_actions(env, [{"type": "scratchpad_append", "content": "", "edits": [edit]}]) == 0
    events = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    assert [(row["action_type"], row["reason"]) for row in events] == [("scratchpad_append", "empty_content")]
    # Bound to this operation's complete read, each malformed action is refused with a typed outcome.
    bound = [{**action, "expected_revision": original.revision} for action in malformed]
    assert reflection.apply_memory_actions(env, bound) == 0
    assert k.read_knowledge_note(original.address).raw == original.raw
    rows = [json.loads(line) for line in (tmp_path / "memory" / "knowledge_history.jsonl").read_text().splitlines()]
    assert [row["outcomes"][0]["reason"].split(":")[0] for row in rows
            if row.get("type") == "reflection_knowledge_write_incomplete"] == [
        "invalid_nomination", "ambiguous_nomination", "invalid_nomination"]
    summary = {"type": "knowledge_write", "content": "", "task_id": "t1", "topic": "people/alex",
               "edits": [edit], "summary": "Depth varies.", "expected_revision": original.revision}
    assert reflection.apply_memory_actions(env, [summary]) == 1
    current = k.read_knowledge_note(original.address)
    assert current.metadata == {**original.metadata, "summary": "Depth varies."}
    assert current.text.endswith("---\n# Alex\n\nDepth today.\nDECISIVE ORIGINAL TAIL.\n")


def test_oversized_requested_note_is_retained_and_only_delivered_prefix_is_credited(tmp_path, fit):
    from ouroboros.artifacts import read_actor_source_bytes

    original = _initial(tmp_path)
    current = k.write_knowledge_note(original.address, "Large source. " * 10000, expected_revision=original.revision).current
    fit.window = 24000
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    llm = MemoryLLM("The delivered source is partial; do not rewrite it yet.")
    reads = c.KnowledgeReadContext(ctx)
    content, usage = c._call_consolidation_llm(llm, "Original episode.", "test memory",
                                              knowledge=reads)
    assert content and len(llm.calls) == 2
    shown = llm.calls[-1]["messages"][-1]["content"]
    notice = json.loads(shown.split("\n[Tool result source view]\n", 1)[1])
    assert read_actor_source_bytes(tmp_path, "consolidation", notice["source_ref"]).decode().endswith(current.text)
    assert notice["delivered_range"][1] < notice["complete_chars"]
    entry = reads.bind_entries([{"topic": "people/alex", "content": "Unseen rewrite."}])[0]
    assert entry["expected_revision"] is None
    assert not c._write_knowledge_entries(current.address.shelf, [entry], context=ctx)[0]["ok"]
    assert k.read_knowledge_note(current.address).raw == current.raw
    assert usage["cost"] == pytest.approx(0.03)


def test_era_compression_cannot_erase_unpublished_knowledge_proposals(tmp_path, fit):
    original = _initial(tmp_path)
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=1100, text_size=0)

    class ManyBlocks:
        count = 0

        def chat(self, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            if prompt.startswith("Compress these older memory blocks"):
                return {"content": "The full historical span remains represented."}, {"cost": 0.01}
            if prompt.startswith("Compare this draft memory"):
                return {"content": f"Episode {self.count}, checked.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps([
                    {"topic": "people/alex", "content": f"Unpublished complete proposal {self.count}."}])}, {"cost": 0.01}
            self.count += 1
            return {"content": f"Episode {self.count}.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps([
                {"topic": "people/alex", "content": f"Unpublished complete proposal {self.count}."}])}, {"cost": 0.01}

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="many-blocks")
    c.consolidate(chat, blocks, meta, ManyBlocks(), knowledge_context=ctx)
    saved = json.loads(blocks.read_text())
    assert saved[0]["type"] == "era" and len(saved) == 8
    assert k.read_knowledge_note(original.address).raw == original.raw
    records = [json.loads(line) for line in (tmp_path / "memory" / "knowledge_history.jsonl").read_text().splitlines()]
    nominations = next(row for row in records if row.get("type") == "dialogue_knowledge_nominations")
    assert len(nominations["nominations"]) == 11
    assert nominations["nominations"][0]["entries"][0]["content"] == "Unpublished complete proposal 1."
    assert len([row for row in records if row.get("type") == "dialogue_knowledge_writes_incomplete"]) == 11
    # The era object carries no knowledge_writes, so the batch receipt lives in meta:
    # without it the incomplete publication would vanish from every resident surface.
    assert "knowledge_writes" not in saved[0]
    pending = json.loads(meta.read_text())["pending_knowledge_nominations"]
    assert len(pending) == 11
    assert all(row["id"].startswith(nominations["entry_id"] + ":") for row in pending)
    assert all(row["reason"] == "revision_required" for row in pending)
