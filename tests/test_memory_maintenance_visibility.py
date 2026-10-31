"""Nothing in memory maintenance leaves silently (TZ-3 PR-1, invariant I4).

Every host decision that used to vanish is a typed fact, each proven in both
directions: a scratchpad fold and its outcome, and the ``writer``/``route``/
``writer_input_ref``/``old_chars``/``new_chars`` stamp on every ``source_capture``
history row. The reader-less ``knowledge_journal.jsonl`` writer is gone, and so are
the old dialogue writer, the automatic reflection and scratchpad knowledge writers
and the post-task scratchpad pass (#918) whose skips and events this file used to
pin: a fold is the mind's act, named by its own journal rows, and a fold text that
still nominates notes writes none.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import consolidator as c
from ouroboros import knowledge as store
from ouroboros.memory import Memory, ScratchpadFoldFailed, scratchpad_block_sha256
from ouroboros.tools import knowledge as knowledge_tools
from ouroboros.tools.registry import ToolContext


def _events(root, kind):
    path = root / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == kind]


def _history(root):
    path = root / "memory" / "knowledge_history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- every scratchpad fold names its outcome ------------------------------------------


def _scratchpad(tmp_path, count=4):
    memory = Memory(tmp_path)
    for index in range(count):
        memory.append_scratchpad_block(f"block-{index}-" + (chr(97 + index) * 8_000), source=f"source-{index}")
    return memory


def _journal(memory, kind=None):
    rows = [json.loads(line) for line in memory.journal_path().read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if kind is None or row.get("type") == kind]


def test_a_scratchpad_fold_reports_its_counts_and_source(tmp_path):
    memory = _scratchpad(tmp_path)
    oldest = [scratchpad_block_sha256(block) for block in memory.load_scratchpad_blocks()[:2]]
    memory.fold_scratchpad_blocks("compressed", oldest)
    [row] = _journal(memory, "blocks_folded")
    assert (row["blocks_before"], len(row["replaced_sha256"]), row["blocks_after"]) == (4, 2, 3)
    assert row["replaced_sha256"] == oldest
    assert row["content_len"] == sum(len(block["content"]) for block in memory.load_scratchpad_blocks())
    assert row["entry_id"] == memory.load_scratchpad_blocks()[0]["metadata"]["source_ref"]["entry_id"]
    assert [kind["type"] for kind in _journal(memory)][-2:] == ["fold_sources_preserved", "blocks_folded"]
    assert "knowledge_writes" not in row


@pytest.mark.parametrize("cause, kind, row_type", [
    ("stale", "not_active", "block_fold_refused"),
    ("journal", "journal_unavailable", "block_fold_failed"),
])
def test_a_refused_or_failed_fold_names_why_and_keeps_the_blocks(tmp_path, monkeypatch, cause, kind, row_type):
    from ouroboros import memory as memory_module

    memory = _scratchpad(tmp_path)
    before = memory.load_scratchpad_blocks()
    named = ["0" * 64] if cause == "stale" else [scratchpad_block_sha256(before[0])]
    if cause == "journal":
        real = memory_module.append_jsonl
        monkeypatch.setattr(memory_module, "append_jsonl", lambda path, row, *a, **k: (
            False if row.get("type") == "fold_sources_preserved" else real(path, row, *a, **k)))
    with pytest.raises(ScratchpadFoldFailed) as refused:
        memory.fold_scratchpad_blocks("compressed", named)
    assert refused.value.kind == kind
    assert memory.load_scratchpad_blocks() == before
    [row] = _journal(memory, row_type)
    assert row["kind"] == kind and row["fold_block"]["content"] == "compressed"
    assert _journal(memory, "blocks_folded") == []


def test_no_fold_means_no_fold_rows_and_no_pass_event(tmp_path):
    memory = Memory(tmp_path)
    memory.append_scratchpad_block("small", source="task")
    assert [row["type"] for row in _journal(memory)] == ["block_appended"]
    assert _events(tmp_path, "scratchpad_consolidation") == []


# --- the host stamp on every source_capture row --------------------------------------


def _address(root, topic="people/alex"):
    return store.resolve_knowledge_address(root, topic, "global")


def test_an_unnamed_writer_and_a_named_one_both_stamp_the_capture_row(tmp_path):
    target = _address(tmp_path)
    assert store.write_knowledge_note(target, "# Alex\n\nFirst.").ok
    legacy_shaped = _history(tmp_path)[-1]
    assert legacy_shaped["publication"] == "source_capture"
    assert (legacy_shaped["writer"], legacy_shaped["route"], legacy_shaped["writer_input_ref"]) == (
        store.UNKNOWN_STAMP, store.UNKNOWN_STAMP, store.UNKNOWN_STAMP)
    assert (legacy_shaped["old_chars"], legacy_shaped["new_chars"]) == (0, len(legacy_shaped["new_content"]))

    current = store.read_knowledge_note(target)
    result = store.write_knowledge_note(target, "# Alex\n\nFirst. Second.", expected_revision=current.revision,
                                        writer="turn", route={"model": "m"}, writer_input_ref={"chat": 1})
    assert result.ok
    row = _history(tmp_path)[-1]
    assert (row["writer"], row["route"], row["writer_input_ref"]) == ("turn", {"model": "m"}, {"chat": 1})
    assert row["old_chars"] == len(current.text) and row["new_chars"] == len(result.current.text)
    assert row["delta"]["old_chars"] == row["old_chars"] and row["source_ref"] == result.current.source_ref()
    assert "writer" not in result.current.text  # the stamp lives on the history row, never in the note


def test_the_route_stamp_is_what_answered_never_the_configuration():
    # Review F3: a model-wait override or account rotation changes what ANSWERED;
    # the configured Light route cannot say which model wrote the note.
    assert store.observed_route_stamp({"cost": 0.01}) == store.UNKNOWN_STAMP  # no physical fact at all
    assert store.observed_route_stamp(None) == store.UNKNOWN_STAMP
    assert store.observed_route_stamp({"provider": "openrouter", "resolved_model": "openai/gpt-x"}) == {
        "provider": "openrouter", "model": "openai/gpt-x"}
    # Round 2 (advisory 5): a partial physical fact leaves the missing field unknown,
    # never the caller's configured model; the local lane stamps its own resolved_model.
    assert store.observed_route_stamp({"provider": "openrouter"}) == {"provider": "openrouter", "model": "unknown"}
    assert store.observed_route_stamp({"resolved_model": "local-model"}) == {"provider": "unknown", "model": "local-model"}
    assert store.observed_route_stamp({"provider": "local", "resolved_model": "local-model"}) == {
        "provider": "local", "model": "local-model"}
    # The production-shaped served route names the account as credentialProfileId.
    served = {"provider": "claudexor", "resolved_model": "claude-fable", "claudexor": {"route": {
        "source": "claude", "model": "claude-fable", "credentialProfileId": "acct-B", "accountFingerprint": "fp-B"}}}
    assert store.observed_route_stamp(served) == {
        "provider": "claudexor", "model": "claude-fable", "source": "claude", "account": "acct-B",
        "account_fingerprint": "fp-B"}
    rotated = {**served, "claudexor": {"route": {**served["claudexor"]["route"], "credentialProfileId": "acct-C"}}}
    assert store.observed_route_stamp(rotated)["account"] == "acct-C"  # rotation is visible in the stamp
    # A merged consolidation usage forwards the LAST physical route of the unit.
    merged = c._merge_consolidation_usage({"cost": 0.01, "provider": "openrouter", "resolved_model": "a"},
                                          {"cost": 0.01, "provider": "openrouter", "resolved_model": "b"})
    assert merged["_observed_route"] == {"provider": "openrouter", "model": "b"}
    assert store.observed_route_stamp(merged) == {"provider": "openrouter", "model": "b"}
    assert "_observed_route" not in c._merge_consolidation_usage({"cost": 0.01})


def test_the_route_stamp_never_reads_a_configured_route_from_an_empty_usage():
    # Round 2 (advisory 5): the former ``model``/``use_local`` fallback turned an
    # empty usage into the configured route. Only physical facts are read now.
    import inspect

    assert list(inspect.signature(store.observed_route_stamp).parameters) == ["usage"]
    assert store.observed_route_stamp({}) == store.UNKNOWN_STAMP
    assert store.observed_route_stamp({"cost": 0.0, "prompt_tokens": 0}) == store.UNKNOWN_STAMP
    assert store.observed_route_stamp({"_observed_route": "unknown"}) == store.UNKNOWN_STAMP  # not a dict stamp


def test_a_merge_never_lets_an_earlier_stamp_masquerade_as_the_final_call():
    # Round 2 (advisory 5): the final call of a unit answered without a physical
    # fact (a released send, an exception's usage); an earlier known stamp must not
    # be forwarded as if that call had produced it.
    stamped = {"cost": 0.01, "provider": "openrouter", "resolved_model": "a"}
    merged = c._merge_consolidation_usage(stamped, {"cost": 0.01})
    assert "_observed_route" not in merged and store.observed_route_stamp(merged) == store.UNKNOWN_STAMP
    # A forwarded merged stamp counts as the (known) last call when it IS the last usage ...
    again = c._merge_consolidation_usage({"cost": 0.01}, c._merge_consolidation_usage(stamped))
    assert again["_observed_route"] == {"provider": "openrouter", "model": "a"}
    # ... and an unknown-last merged unit stays unknown through a further merge.
    nested = c._merge_consolidation_usage(stamped, c._merge_consolidation_usage(stamped, {"cost": 0.01}))
    assert store.observed_route_stamp(nested) == store.UNKNOWN_STAMP


def test_a_direct_turn_stamps_itself_and_its_observed_route_when_the_loop_recorded_one(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="turn-1")
    assert "✅" in knowledge_tools._knowledge_write(ctx, "notes/a", "Plain observation.")
    first = _history(tmp_path)[-1]
    assert (first["writer"], first["route"], first["task_id"]) == ("turn", store.UNKNOWN_STAMP, "turn-1")

    # The loop records what answered its last round on every lane, not only Claudexor.
    ctx._accumulated_usage = {"_observed_route": {"provider": "openrouter", "model": "openai/gpt-x"}}
    assert "✅" in knowledge_tools._knowledge_write(ctx, "notes/b", "Another observation.")
    second = _history(tmp_path)[-1]
    assert second["writer"] == "turn" and second["route"] == {"provider": "openrouter", "model": "openai/gpt-x"}


_FORGED = {"provider": "forged", "model": "forged/model", "account": "forged-acct"}


def test_a_fold_text_that_still_nominates_notes_writes_none(tmp_path):
    """The older prompt asked for knowledge_entries beside the block; a fold text of that
    shape, forged host keys included, is stored as text and writes no note."""
    memory = _scratchpad(tmp_path)
    text = json.dumps({
        "knowledge_entries": [{"topic": "lessons/one", "content": "A durable lesson.", "_nomination_route": _FORGED,
                               "_expected_revision": "forged", "_task_id": "forged"}],
        "compressed_block": "compressed"})
    memory.fold_scratchpad_blocks(text, [scratchpad_block_sha256(memory.load_scratchpad_blocks()[0])])
    [row] = _journal(memory, "blocks_folded")
    assert "knowledge_writes" not in row and "knowledge_entries" not in row
    assert not (tmp_path / "memory" / "knowledge").exists()
    assert not (tmp_path / "memory" / "knowledge_history.jsonl").exists()
    folded = memory.load_scratchpad_blocks()[0]
    assert folded["content"] == text
    assert set(folded["metadata"]) == {"fold", "source_ref"}


def test_the_reader_less_knowledge_journal_is_no_longer_written(tmp_path):
    assert store.write_knowledge_note(_address(tmp_path), "# Alex\n\nFirst.").ok
    assert not (tmp_path / "memory" / "knowledge_journal.jsonl").exists()
    assert (tmp_path / "memory" / "knowledge_history.jsonl").exists()
