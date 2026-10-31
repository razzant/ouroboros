"""A fold's text is the mind's own: stored verbatim, never parsed, never a note (#918).

This file pinned how the retired post-task Light pass recovered its JSON answer
(``{"compressed_block": …}``, fenced, prose-tailed, malformed) and ignored the older
prompt's ``knowledge_entries``. No model answer is parsed for the scratchpad any more:
the integrating mind passes the accepted text to ``update_scratchpad`` with ``replaces``.
The same corpus of answer shapes now proves that whatever the text looks like it lands
as written — a nomination-shaped text creates or revises no note, a JSON-looking text is
not unwrapped — and that an empty text changes nothing. The paid-stage accounting the
file also pinned belongs to the remaining fallback draft stage (``test_budget_tracking``).
"""
import json

import pytest

from ouroboros.memory import Memory, scratchpad_block_sha256
from ouroboros.tools.registry import ToolContext, ToolRegistry

FINAL = {"knowledge_entries": [], "compressed_block": "Actual consolidated memory."}  # older answer shape
NOMINATING = json.dumps({"knowledge_entries": [{"topic": "lessons/new", "content": "# Lesson\nNominated."},
                                              {"topic": "untouched", "scope": "global", "edits": [
                                                  {"old_text": "Keep unchanged.", "new_text": "Rewritten."}]}],
                         "compressed_block": "Actual consolidated memory."})
JSON = json.dumps(FINAL)
EXAMPLE = json.dumps({"knowledge_entries": [], "compressed_block": "Quoted prompt example."})
CASES = [
    ("pure", JSON),
    ("old_nominations", NOMINATING),
    ("fenced", "```json\n" + JSON + "\n```"),
    ("prose_tail", "Here is the consolidated memory.\n" + JSON),
    ("example_then_tail", "Example: " + EXAMPLE + "\nActual result:\n" + JSON),
    ("prose_fenced_tail", "Here is the result.\n```json\n" + JSON + "\n```"),
    ("invalid", "No JSON result."),
    ("malformed", '{"compressed_block":'),
    ("wrong_block_type", '{"knowledge_entries":[],"compressed_block":7}'),
    ("missing_block", '{"knowledge_entries":[]}'),
    ("empty_block", '{"knowledge_entries":[],"compressed_block":"   "}'),
    ("prose_after_object", "Example: " + JSON + "\nThis is only an example."),
    ("duplicate_tail", 'Result:\n{"compressed_block":"first","compressed_block":"second"}'),
]
EMPTY = [("empty", ""), ("blank", "   \n  "), ("array", "[]")]


def _memory_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in (root / "memory").rglob("*") if p.is_file()}


def _install(tmp_path):
    memory = Memory(tmp_path)
    for index in range(4):
        memory.append_scratchpad_block(f"block-{index}-" + (chr(97 + index) * 8_000), source=f"source-{index}")
    knowledge = tmp_path / "memory" / "knowledge"
    knowledge.mkdir(parents=True, exist_ok=True)
    (knowledge / "untouched.md").write_text("# Existing note\nKeep unchanged.\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="fold-shapes"))
    return memory, registry


def _knowledge(files):
    return {key: value for key, value in files.items() if key.startswith("memory/knowledge")}


@pytest.mark.parametrize("label,text", CASES, ids=[row[0] for row in CASES])
def test_a_fold_stores_any_text_verbatim_and_writes_no_note(tmp_path, label, text):
    memory, registry = _install(tmp_path)
    before = memory.load_scratchpad_blocks()
    files_before = _memory_bytes(tmp_path)

    result = registry.execute_result("update_scratchpad", {
        "content": text, "replaces": [scratchpad_block_sha256(block) for block in before[:2]]})

    assert result.status == "ok", result.text
    after = memory.load_scratchpad_blocks()
    assert after[0]["content"] == text and after[1:] == before[2:]
    assert _knowledge(_memory_bytes(tmp_path)) == _knowledge(files_before)
    assert not (tmp_path / "memory" / "knowledge_history.jsonl").exists()
    journal = [json.loads(row) for row in memory.journal_path().read_text(encoding="utf-8").splitlines()]
    [preserved] = [row for row in journal if row.get("type") == "fold_sources_preserved"]
    assert preserved["source_blocks"] == before[:2] and "knowledge_entries" not in preserved
    assert preserved["fold_block"]["content"] == text


@pytest.mark.parametrize("label,text", EMPTY, ids=[row[0] for row in EMPTY])
def test_an_empty_fold_text_changes_nothing(tmp_path, label, text):
    memory, registry = _install(tmp_path)
    before = memory.load_scratchpad_blocks()
    files_before = _memory_bytes(tmp_path)

    result = registry.execute_result("update_scratchpad", {
        "content": text, "replaces": [scratchpad_block_sha256(block) for block in before[:2]]})

    assert result.status != "ok" and result.code == "TOOL_ARG_ERROR"
    assert memory.load_scratchpad_blocks() == before and _memory_bytes(tmp_path) == files_before
