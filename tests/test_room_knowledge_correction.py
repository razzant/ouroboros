"""A knowledge nomination binds only the revision its own operation completely read.

Reflection and scratchpad consolidation send one Light operation (``_call_consolidation_llm``
with a ``KnowledgeReadContext``); the retired dialogue writer's draft/correction pair and the
retired pressure maintenance were more such operations. The host credits a read only for the
characters actually delivered, binds that revision to the nomination, and the common
writer then requires it for every existing note: no read, a partial read or a stale read
never lands a change, an existing note changes only by anchored edits, and a newer
concurrent change is preserved.
"""

from copy import deepcopy
import json

import pytest

from ouroboros import consolidator as c, knowledge as k
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

fit = fit_helpers.fit
TOPIC = "shared-understanding"


class ReadingLLM:
    """One scripted operation: an optional knowledge_read, then its nomination. Reads,
    delivery, binding and publication are real."""

    def __init__(self, change, *, read="complete", after_read=None, before_read=None, expected_note=None):
        self.change, self.read, self.after_read, self.before_read = change, read, after_read, before_read
        self.expected_note = expected_note
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        messages = kwargs["messages"]
        if messages[-1]["role"] != "tool" and self.read != "none":
            if self.before_read:
                self.before_read()
            args = {"topic": TOPIC, "scope": "global"}
            if self.read == "partial":
                args.update(start_char=0, end_char=12)
            return {"tool_calls": [{"id": "read-note", "type": "function", "function": {
                "name": "knowledge_read", "arguments": json.dumps(args)}}]}, {"cost": 0.01}
        if self.read == "complete" and self.expected_note is not None:
            # The scripted answer is permitted only after actual source delivery.
            assert self.expected_note in messages[-1]["content"]
        if self.after_read:
            self.after_read()
        return {"content": json.dumps({"knowledge_entries": [{"topic": TOPIC, "scope": "global", **self.change}]})}, {
            "cost": 0.01}


def _setup(tmp_path, initial):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="memory-operation")
    address = k.resolve_knowledge_address(tmp_path, TOPIC, "global")
    original = k.write_knowledge_note(address, initial).current if initial else None
    return ctx, address, original


def _operate(ctx, llm, episode="An episode."):
    knowledge = c.KnowledgeReadContext(ctx, "knowledge_probe")
    content, usage = c._call_consolidation_llm(
        llm, c.KNOWLEDGE_MAINTENANCE_PROMPT + "\n## Episode\n" + episode, "Knowledge probe", knowledge=knowledge)
    assert content, usage
    return knowledge.bind_entries(json.loads(content)["knowledge_entries"])


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
@pytest.mark.parametrize("read", ["none", "partial", "complete"])
def test_only_the_operations_own_complete_read_binds_a_revision(tmp_path, fit, initial, episode, updated, read):
    ctx, address, original = _setup(tmp_path, initial)
    change = ({"edits": [{"old_text": initial, "new_text": updated, "basis": episode}]} if read == "complete"
              else {"content": episode})  # without the full note the failure shape reduces it to the episode
    entries = _operate(ctx, ReadingLLM(change, read=read, expected_note=original.text), episode)
    outcome = c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]
    current = k.read_knowledge_note(address)
    if read == "complete":
        assert entries[0]["expected_revision"] == original.revision
        assert outcome["ok"] and current.text.endswith(updated)
    else:
        assert entries[0]["expected_revision"] is None
        assert outcome["reason"] == "revision_required" and not outcome["ok"]
        assert current.raw == original.raw


def test_own_complete_read_can_revise_and_remove_obsolete_facts(tmp_path, fit):
    ctx, address, original = _setup(tmp_path, "Alex owns Alpha. Old office: Building 7. Contact by email.")
    updated = "Alex now owns Beta. Contact by email."
    basis = "The new episode moves ownership and asks to remove the obsolete office."
    entries = _operate(ctx, ReadingLLM({"edits": [
        {"old_text": "Alex owns Alpha.", "new_text": "Alex now owns Beta.", "basis": basis},
        {"old_text": "Old office: Building 7. ", "new_text": "", "basis": basis}]}, expected_note=original.text))
    assert entries[0]["expected_revision"] == original.revision
    assert c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]["ok"]
    current = k.read_knowledge_note(address)
    assert current.text.endswith(updated)
    assert "Building 7" not in current.text and "owns Alpha" not in current.text
    history = (tmp_path / "memory" / "knowledge_history.jsonl").read_text(encoding="utf-8")
    assert "Building 7" in history and "now owns Beta" in history


@pytest.mark.parametrize("change_after_read", [False, True])
def test_binding_holds_the_read_revision_and_preserves_later_concurrent_changes(tmp_path, fit, change_after_read):
    ctx, address, original = _setup(tmp_path, "Draft-era understanding.")
    latest = "A newer established observation."

    def replace():
        assert k.write_knowledge_note(address, latest, expected_revision=original.revision).ok

    llm = ReadingLLM({"edits": [{"old_text": "Draft-era understanding." if change_after_read else latest,
                                 "new_text": "A newer established observation with the new episode.",
                                 "basis": "The new episode establishes this change."}]},
                     before_read=None if change_after_read else replace,
                     after_read=replace if change_after_read else None,
                     expected_note=original.text if change_after_read else latest)
    entries = _operate(ctx, llm)
    current = k.read_knowledge_note(address)
    assert entries[0]["expected_revision"] == (original.revision if change_after_read else current.revision)
    result = c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]
    if change_after_read:
        assert not result["ok"] and result["reason"] == "revision_conflict"
        assert k.read_knowledge_note(address).raw == current.raw
    else:
        assert result["ok"]
        assert k.read_knowledge_note(address).text.endswith("with the new episode.")


def test_an_unread_operation_can_create_a_new_note(tmp_path, fit):
    ctx, address, _ = _setup(tmp_path, None)
    entries = _operate(ctx, ReadingLLM({"content": "New observation."}, read="none"))
    assert entries[0]["expected_revision"] is None
    assert c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]["ok"]
    assert k.read_knowledge_note(address).text.endswith("New observation.")


def test_a_whole_replacement_after_a_complete_read_is_refused(tmp_path, fit):
    initial = ("---\ntype: person\nsummary: Established biography and delivery.\n---\n"
               "# Alex\n\nLong-standing role: maintains Alpha.\nReport R1 was delivered at 10:05.\n"
               "Private contact preference: email.\n")
    ctx, address, original = _setup(tmp_path, initial)
    # The historical failure shape: read the whole note, then nominate a short replacement.
    entries = _operate(ctx, ReadingLLM({"content": "Alex wants shorter updates."}, expected_note=original.text))
    assert entries[0]["expected_revision"] == original.revision
    outcome = c._write_knowledge_entries(address.shelf, entries, context=ctx)[0]
    assert not outcome["ok"] and outcome["reason"] == "existing_note_requires_edits"
    assert k.read_knowledge_note(address).raw == original.raw


def test_narrow_source_grounded_edit_moves_only_its_span(tmp_path, fit):
    initial = ("---\ntype: person\nsummary: Established biography and delivery.\n---\n"
               "# Alex\n\nLong-standing role: maintains Alpha.\nPrivate contact preference: email.\n")
    ctx, address, original = _setup(tmp_path, initial)
    entries = _operate(ctx, ReadingLLM({"edits": [{
        "old_text": "Private contact preference: email.",
        "new_text": "Private contact preference: email. Today Alex asked for shorter updates.",
        "basis": "Alex's new request in this episode."}]}, expected_note=original.text))
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
    entries = _operate(ctx, ReadingLLM({"edits": edits, "summary": "Beta owns ingestion."},
                                       expected_note=original.text), episode)
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
    entries = _operate(ctx, ReadingLLM({"edits": [], "summary": "New context."}, expected_note=original.text),
                       "The context of this history changed.")
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
