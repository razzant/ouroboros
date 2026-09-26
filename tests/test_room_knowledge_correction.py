"""Cumulative note changes belong to the correction's own delivered sources."""

from copy import deepcopy
import json

import pytest

from ouroboros import consolidator as c, knowledge as k, room_consolidation as rc
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

fit = fit_helpers.fit
TOPIC = "shared-understanding"


def _answer(content, edits=None):
    change = edits if isinstance(edits, dict) else {"edits": edits}  # a dict names every authored field
    return "I recorded the episode.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps([
        {"topic": TOPIC, "scope": "global", **(change if edits is not None else {"content": content})}])


class CorrectionLLM:
    """Scripted responses; all read, delivery, binding and publication are real."""

    def __init__(self, draft, corrected, *, correction_read="complete", draft_read=True,
                 before_correction=None, after_correction_read=None, expected_note=None,
                 correction_edits=None):
        self.draft, self.corrected, self.correction_edits = draft, corrected, correction_edits
        self.correction_read, self.draft_read = correction_read, draft_read
        self.before_correction = before_correction
        self.after_correction_read = after_correction_read
        self.expected_note = expected_note
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        messages = kwargs["messages"]
        correction = messages[0]["content"].startswith("Compare this draft memory")
        if messages[-1]["role"] != "tool":
            if correction and self.before_correction:
                self.before_correction()
            mode = self.correction_read if correction else ("complete" if self.draft_read else "none")
            if mode != "none":
                args = {"topic": TOPIC, "scope": "global"}
                if mode == "partial":
                    args.update(start_char=0, end_char=12)
                return {"tool_calls": [{"id": "read-note", "type": "function", "function": {
                    "name": "knowledge_read", "arguments": json.dumps(args)}}]}, {"cost": 0.01}
        if correction and self.correction_read == "complete":
            # The scripted answer is permitted only after actual source delivery.
            assert self.expected_note in messages[-1]["content"]
            if self.after_correction_read:
                self.after_correction_read()
        return {"content": _answer(self.corrected if correction else self.draft,
                                   self.correction_edits if correction else None)}, {"cost": 0.01}


def _setup(tmp_path, initial):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="room-memory")
    address = k.resolve_knowledge_address(tmp_path, TOPIC, "global")
    original = k.write_knowledge_note(address, initial).current if initial else None
    return ctx, address, original


def _consolidate(ctx, llm, episode):
    room = rc.RoomSource("1", "Main", [{"text": episode}], episode)
    block, usage = rc.summarize_block(
        c._light_call(llm, ctx, {}), [room], first_ts="2026-09-01T10:00:00Z",
        last_ts="2026-09-01T10:01:00Z", knowledge_instruction=c.KNOWLEDGE_MAINTENANCE_PROMPT)
    assert block and block["rooms"][0]["content"] == "I recorded the episode."
    return usage["_knowledge_entries"]


CASES = [
    ("Alex maintains Alpha and prefers email.", "Alex sent a meeting invitation.",
     "Alex maintains Alpha and prefers email. Alex sent a meeting invitation; attendance is unknown."),
    ("Delivery D1 of report R1 was confirmed at 10:05. Reply is unknown.",
     "At 10:00 I was asked to send report R1.",
     "Delivery D1 of report R1 was confirmed at 10:05 after the 10:00 request. Reply is unknown."),
    ("Alpha owns ingestion; Beta owns storage; Gamma owns search.", "Alpha now owns validation too.",
     "Alpha owns ingestion and validation; Beta owns storage; Gamma owns search."),
]


@pytest.mark.parametrize("initial,episode,updated", CASES)
@pytest.mark.parametrize("correction_read", ["none", "partial", "complete"])
def test_correction_cannot_borrow_draft_read_and_can_preserve_cumulative_knowledge(
        tmp_path, fit, initial, episode, updated, correction_read):
    ctx, address, original = _setup(tmp_path, initial)
    # Without the full note the failure-shaped response reduces it to the episode.
    content = updated if correction_read == "complete" else episode
    edits = [{"old_text": initial, "new_text": updated, "basis": episode}] if correction_read == "complete" else None
    llm = CorrectionLLM(updated, content, correction_read=correction_read,
                        expected_note=original.text, correction_edits=edits)
    entries = _consolidate(ctx, llm, episode)
    outcome = c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]
    current = k.read_knowledge_note(address)
    if correction_read == "complete":
        assert entries[0]["expected_revision"] == original.revision
        assert outcome["ok"] and current.text.endswith(updated)
    else:
        assert entries[0]["expected_revision"] is None
        assert outcome["reason"] == "revision_required" and not outcome["ok"]
        assert current.raw == original.raw
    # Both authoring stages receive the cumulative scope and range-read support.
    for call in (llm.calls[0], next(call for call in llm.calls
                                  if call["messages"][0]["content"].startswith("Compare this draft memory"))):
        assert c.KNOWLEDGE_MAINTENANCE_PROMPT in call["messages"][0]["content"]


def test_own_complete_read_can_revise_and_remove_obsolete_facts_without_draft_read(tmp_path, fit):
    ctx, address, original = _setup(tmp_path, "Alex owns Alpha. Old office: Building 7. Contact by email.")
    updated = "Alex now owns Beta. Contact by email."
    basis = "The new episode moves ownership and asks to remove the obsolete office."
    llm = CorrectionLLM(original.text, updated, draft_read=False, expected_note=original.text, correction_edits=[
        {"old_text": "Alex owns Alpha.", "new_text": "Alex now owns Beta.", "basis": basis},
        {"old_text": "Old office: Building 7. ", "new_text": "", "basis": basis}])
    entries = _consolidate(ctx, llm, "Alex moved from Alpha to Beta and asked to remove the obsolete office address.")
    assert entries[0]["expected_revision"] == original.revision
    assert c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]["ok"]
    current = k.read_knowledge_note(address)
    assert current.text.endswith(updated)
    assert "Building 7" not in current.text and "owns Alpha" not in current.text
    history = (tmp_path / "memory" / "knowledge_history.jsonl").read_text(encoding="utf-8")
    assert "Building 7" in history and "now owns Beta" in history


@pytest.mark.parametrize("change_after_read", [False, True])
def test_correction_binds_its_current_revision_and_preserves_later_concurrent_changes(tmp_path, fit, change_after_read):
    ctx, address, original = _setup(tmp_path, "Draft-era understanding.")
    latest = "A newer established observation."

    def replace():
        assert k.write_knowledge_note(address, latest, expected_revision=original.revision).ok

    llm = CorrectionLLM(original.text, "A newer established observation with the new episode.",
                        before_correction=None if change_after_read else replace,
                        after_correction_read=replace if change_after_read else None,
                        expected_note=original.text if change_after_read else latest,
                        correction_edits=[{"old_text": "Draft-era understanding." if change_after_read else latest,
                                           "new_text": "A newer established observation with the new episode.",
                                           "basis": "The new episode establishes this change."}])
    entries = _consolidate(ctx, llm, "A new episode.")
    current = k.read_knowledge_note(address)
    assert entries[0]["expected_revision"] == (original.revision if change_after_read else current.revision)
    result = c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]
    if change_after_read:
        assert not result["ok"] and result["reason"] == "revision_conflict"
        assert k.read_knowledge_note(address).raw == current.raw
    else:
        assert result["ok"]
        assert k.read_knowledge_note(address).text.endswith("with the new episode.")


def test_correction_can_create_a_new_nominated_note_without_reading(tmp_path, fit):
    ctx, address, _ = _setup(tmp_path, None)
    llm = CorrectionLLM("New observation.", "Corrected new observation.",
                        draft_read=False, correction_read="none")
    entries = _consolidate(ctx, llm, "A new observation.")
    assert entries[0]["expected_revision"] is None
    assert c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]["ok"]
    assert k.read_knowledge_note(address).text.endswith("Corrected new observation.")


def test_unread_correction_failure_remains_visible_after_dialogue_publication(tmp_path, fit):
    ctx, address, original = _setup(tmp_path, "An established body of knowledge.")
    chat, blocks, meta = fit_helpers._paths(tmp_path)
    fit_helpers._write_chat(chat, text_size=0)
    llm = CorrectionLLM(original.text, "Only this episode.", correction_read="none")
    c.consolidate(chat, blocks, meta, llm, knowledge_context=ctx)
    stored = json.loads(blocks.read_text(encoding="utf-8"))[0]
    assert stored["knowledge_writes"][0]["reason"] == "revision_required"
    state = json.loads(meta.read_text(encoding="utf-8"))
    assert state["last_consolidated_offset"] == 100
    assert len(state["pending_knowledge_nominations"]) == 1
    assert state["pending_knowledge_nominations"][0]["reason"] == "revision_required"
    assert k.read_knowledge_note(address).raw == original.raw


def test_legacy_full_replacement_after_complete_reads_stays_visible_debt(tmp_path, fit):
    initial = ("---\ntype: person\nsummary: Established biography and delivery.\n---\n"
               "# Alex\n\nLong-standing role: maintains Alpha.\nReport R1 was delivered at 10:05.\n"
               "Private contact preference: email.\n")
    ctx, address, original = _setup(tmp_path, initial)
    chat, blocks, meta = fit_helpers._paths(tmp_path)
    fit_helpers._write_chat(chat, text_size=0)
    # The historical failure shape: both actors read the whole note, then
    # nominate a short replacement that omits its older facts.
    llm = CorrectionLLM("Alex wants shorter updates.", "Alex wants shorter updates.", expected_note=original.text)
    c.consolidate(chat, blocks, meta, llm, knowledge_context=ctx)
    stored = json.loads(blocks.read_text(encoding="utf-8"))[0]
    assert stored["knowledge_writes"][0]["reason"] == "existing_note_requires_edits"
    state = json.loads(meta.read_text(encoding="utf-8"))
    assert state["last_consolidated_offset"] == 100
    assert [row["reason"] for row in state["pending_knowledge_nominations"]] == ["existing_note_requires_edits"]
    assert k.read_knowledge_note(address).raw == original.raw


def test_narrow_source_grounded_edit_moves_only_its_span(tmp_path, fit):
    initial = ("---\ntype: person\nsummary: Established biography and delivery.\n---\n"
               "# Alex\n\nLong-standing role: maintains Alpha.\nPrivate contact preference: email.\n")
    ctx, address, original = _setup(tmp_path, initial)
    llm = CorrectionLLM("Alex wants shorter updates.", "Alex wants shorter updates.",
                        expected_note=original.text, correction_edits=[{
                            "old_text": "Private contact preference: email.",
                            "new_text": "Private contact preference: email. Today Alex asked for shorter updates.",
                            "basis": "Alex's new request in this episode."}])
    entries = _consolidate(ctx, llm, "Alex asked for a shorter update today.")
    assert entries[0]["expected_revision"] == original.revision
    assert c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]["ok"]
    assert k.read_knowledge_note(address).raw == original.raw.replace(
        b"Private contact preference: email.",
        b"Private contact preference: email. Today Alex asked for shorter updates.")


def test_explicit_removal_and_summary_revision_keep_unrelated_content_while_bad_edits_fail(tmp_path, fit):
    initial = ("---\ntype: note\nsummary: Alpha owns ingestion.\ncustom: [kept]\n---\n"
               "Alpha owns ingestion.\nOld address: Building 7.\nContact by email.\n")
    ctx, address, original = _setup(tmp_path, initial)
    episode = "Alex asked to remove the old address and moved ownership to Beta."
    edits = [{"old_text": "Alpha owns ingestion.", "new_text": "Beta owns ingestion.", "basis": episode},
             {"old_text": "Old address: Building 7.\n", "new_text": "", "basis": episode}]
    llm = CorrectionLLM(initial, "Updated note.", expected_note=original.text,
                        correction_edits={"edits": edits, "summary": "Beta owns ingestion."})
    entries = _consolidate(ctx, llm, episode)
    assert c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]["ok"]
    assert k.read_knowledge_note(address).raw == original.raw.replace(
        b"Alpha owns ingestion.", b"Beta owns ingestion.").replace(b"Old address: Building 7.\n", b"").replace(
        b"custom: [kept]", b"custom:\n- kept")  # a metadata change re-renders YAML through the ordinary merge

    for bad in ([{"old_text": "Contact", "new_text": "Phone", "basis": ""}],
                [{"old_text": "missing", "new_text": "anything", "basis": episode}],
                [{"old_text": "summary: Beta", "new_text": "summary: Gamma", "basis": episode}],  # frontmatter is not body
                [{"old_text": "Contact by email.", "new_text": "Phone", "basis": episode},
                 {"old_text": "email.", "new_text": "phone.", "basis": episode}],
                [{"old_text": "Contact by email.", "new_text": "Phone", "basis": episode},
                 {"old_text": "Beta owns", "new_text": "Gamma owns", "basis": ""}],
                {"old_text": "Contact", "new_text": "Phone", "basis": episode}):
        before = k.read_knowledge_note(address)
        bound = c.KnowledgeReadContext(ctx)
        bound.reads[(address.scope, address.topic)] = before.revision
        outcome = c._write_knowledge_entries(address.shelf, bound.bind_entries([{
            "topic": TOPIC, "scope": "global", "edits": bad}]), context=ctx)[0]
        # A non-list edit shape is refused before the source is read; the rest by the compiler.
        assert not outcome["ok"] and outcome["reason"].startswith(
            "invalid_nomination: " if isinstance(bad, dict) else "invalid_note: ")
        assert k.read_knowledge_note(address).raw == before.raw


def test_summary_only_change_after_its_own_complete_read_merges_and_keeps_unknown_yaml(tmp_path, fit):
    initial = ("---\ntype: note\nsummary: Old context.\nlegacy: obsolete  # authored comment\n"
               "nested: {a: [1, 2], b: {c: true}}\nwhen: 2026-09-01\n---\n# History\nKeep this.\n")
    ctx, address, original = _setup(tmp_path, initial)
    unread = c.KnowledgeReadContext(ctx).bind_entries([{"topic": TOPIC, "scope": "global", "edits": [],
                                                        "summary": "Unread context."}])
    assert c._write_knowledge_entries(address.shelf, unread, context=ctx)[0]["reason"] == "revision_required"
    llm = CorrectionLLM(initial, "unused", expected_note=original.text,
                        correction_edits={"edits": [], "summary": "New context."})
    entries = _consolidate(ctx, llm, "The context of this history changed.")
    assert entries[0]["expected_revision"] == original.revision
    assert c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]["ok"]
    current = k.read_knowledge_note(address)
    # Only the summary value changes; every other field keeps its value while the
    # YAML is re-rendered (the comment and flow style are not kept), body bytes exact.
    assert current.metadata == {**original.metadata, "summary": "New context."}
    assert current.raw != original.raw and current.raw.endswith(b"\n---\n# History\nKeep this.\n")
    row = json.loads((tmp_path / "memory" / "knowledge_history.jsonl").read_text().splitlines()[-1])
    assert (row["edits"], row["summary"], row["source_ref"]) == ([], "New context.", current.source_ref())
    # The same nomination is bound to the revision it read, so it cannot land on the changed note.
    assert c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]["reason"] == "revision_conflict"
    assert k.read_knowledge_note(address).raw == current.raw


def test_body_edit_without_a_changed_summary_keeps_preamble_bytes_and_bad_metadata_never_writes(tmp_path, fit):
    initial = "---\n# authored\ntype:   note\nsummary: 'Quoted.'\ncustom: [kept, 2]\n---\nBody one.\nBody two.\n"
    ctx, address, original = _setup(tmp_path, initial)
    assert original.raw == initial.encode()
    reads = c.KnowledgeReadContext(ctx)
    edit = {"old_text": "Body one.", "new_text": "Body 1.", "basis": "The episode renamed it."}
    history = tmp_path / "memory" / "knowledge_history.jsonl"
    rows = len(history.read_text().splitlines())
    for bad in ({"summary": None}, {"summary": ""}, {"summary": 7}, {"summary": ["x"]},
                {"frontmatter": {"summary": "Generic."}}, {"content": "Whole note."}):
        reads.reads[(address.scope, address.topic)] = original.revision
        outcome = c._write_knowledge_entries(address.shelf, reads.bind_entries([
            {"topic": TOPIC, "scope": "global", "edits": [edit], **bad}]), context=ctx)[0]
        assert not outcome["ok"] and outcome["reason"].split(":")[0] in {"invalid_nomination", "ambiguous_nomination"}
    assert k.read_knowledge_note(address).raw == original.raw
    assert len(history.read_text().splitlines()) == rows
    # Omitted, or equal to the current value, the summary leaves every preamble byte as authored.
    for extra, old, new in (({}, "Body one.", "Body 1."), ({"summary": "Quoted."}, "Body two.", "Body 2.")):
        before = k.read_knowledge_note(address)
        reads.reads[(address.scope, address.topic)] = before.revision
        assert c._write_knowledge_entries(address.shelf, reads.bind_entries([{"topic": TOPIC, "scope": "global", **extra,
            "edits": [{"old_text": old, "new_text": new, "basis": "The episode renamed it."}]}]), context=ctx)[0]["ok"]
        assert k.read_knowledge_note(address).raw == before.raw.replace(old.encode(), new.encode())
        assert ("summary" in json.loads(history.read_text().splitlines()[-1])) == bool(extra)
    assert k.read_knowledge_note(address).raw.startswith(initial.split("Body one.")[0].encode())


def test_list_compiler_is_atomic_and_overlap_is_never_a_unique_anchor():
    with pytest.raises(ValueError, match="exactly once"):
        k.compile_anchored_edits("aaa", [("aa", "b")], "old_text")
    with pytest.raises(ValueError, match="must not overlap"):
        k.compile_anchored_edits("one two three", [("one two", "1 2"), ("two three", "2 3")])
    with pytest.raises(ValueError, match="edit 2"):
        k.compile_anchored_edits("one two", [("one", "1"), ("absent", "x")])
    # Adjacent spans are distinct; every anchor is found in the ORIGINAL text.
    assert k.compile_anchored_edits("one two", [("two", "one"), ("one ", "")]) == "one"
