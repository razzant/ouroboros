"""Dialogue consolidation reports what it actually did.

Three honesty facts of this one stage: a run that advanced the cursor without failing
retires a stale error; every non-None return carries the block count it wrote; and a
nomination batch that was accepted but not fully published leaves a durable receipt
that era compression cannot erase, surfaced as a Health line.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c
from ouroboros import context_health
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers
from tests.test_consolidator_context_fit import _LLM, _Refusal, _paths, _write_chat

fit = fit_helpers.fit


def _health_env(tmp_path):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path,
                           repo_path=lambda p: tmp_path / p, drive_path=lambda p: tmp_path / p)


# --- stale error is retired only by a run that recorded none of its own -----------


def test_a_chronicle_only_pass_still_reports_zero_written_blocks(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=5, text_size=0)  # below the block size: nothing to consolidate
    monkeypatch.setattr(c, "_compact_chronicle", lambda *a, **k: {"prompt_tokens": 1, "completion_tokens": 1,
                                                                   "total_tokens": 2, "cost": 0.0})
    usage = c.consolidate(chat, blocks, meta, _LLM(), compact_chronicle=True, pressure_fits=lambda: False)
    assert usage["_blocks_written"] == 0


def test_a_chronicle_pass_after_a_real_run_keeps_the_written_count(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, text_size=0)
    monkeypatch.setattr(c, "_compact_chronicle", lambda *a, **k: {"prompt_tokens": 1, "completion_tokens": 1,
                                                                   "total_tokens": 2, "cost": 0.0})
    usage = c.consolidate(chat, blocks, meta, _LLM(), compact_chronicle=True, pressure_fits=lambda: False)
    assert usage["_blocks_written"] == 1


def test_light_is_told_to_author_summaries_and_keep_explicit_requests_explicit():
    # Light never sees the knowledge_write schema, so the maintenance prompt is its only carrier.
    assert "YAML summary" in c.KNOWLEDGE_MAINTENANCE_PROMPT
    assert "resident in the index" in c.KNOWLEDGE_MAINTENANCE_PROMPT
    assert "explicit standing request stays explicit" in c.KNOWLEDGE_MAINTENANCE_PROMPT


def test_a_nominated_note_with_a_summary_becomes_resident_in_the_index(tmp_path):
    from ouroboros.knowledge import inventory_knowledge, render_knowledge_index, resolve_knowledge_address
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=str(tmp_path), task_id="t")
    entry = {"topic": "people/alex", "scope": "global", "expected_revision": None,
             "content": "---\ntype: understanding\nsummary: Alex asks for brevity; an interpretation to test.\n---\nEvidence."}
    outcomes = c._write_knowledge_entries(tmp_path / "memory" / "knowledge", [entry], context=ctx)
    assert outcomes and outcomes[0]["ok"]
    rendered = render_knowledge_index(inventory_knowledge(resolve_knowledge_address(tmp_path, "overview", "global")))
    assert "Alex asks for brevity" in rendered


def test_clean_run_clears_a_stale_error_from_an_earlier_run(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    c.atomic_write_json(meta, {"last_consolidated_offset": 0,
                               "chat_log_signature": c._chat_log_signature(chat),
                               "last_consolidation_error": {"kind": "context_overflow", "cursor_offset": 0}})
    c.consolidate(chat, blocks, meta, _LLM())
    saved = json.loads(meta.read_text())
    assert saved["last_consolidated_offset"] == 100
    assert "last_consolidation_error" not in saved


def test_a_run_that_never_advances_keeps_the_stale_error(tmp_path, fit):
    """No cursor movement, no proof: the earlier failure is still the latest word."""
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    stale = {"kind": "provider_failed", "cursor_offset": 0}
    c.atomic_write_json(meta, {"last_consolidated_offset": 0,
                               "chat_log_signature": c._chat_log_signature(chat),
                               "last_consolidation_error": stale})

    def refuse(_llm, _prompt):
        raise _Refusal("auth failed", code="invalid_api_key")

    c.consolidate(chat, blocks, meta, _LLM(effect=refuse))
    assert json.loads(meta.read_text())["last_consolidation_error"]["kind"]


# --- every non-None return carries its own block count ---------------------------


def test_block_count_is_reported_on_a_successful_run(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=200, text_size=0)
    usage = c.consolidate(chat, blocks, meta, _LLM())
    assert usage["_blocks_written"] == 2


def test_block_count_is_zero_when_nothing_was_written(tmp_path, fit):
    """An empty summary writes no block; the receipt says 0 rather than going absent."""
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)

    def empty(_llm, _prompt):
        return {"content": "   "}, {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1, "cost": 0.0}

    usage = c.consolidate(chat, blocks, meta, _LLM(effect=empty))
    assert usage["_blocks_written"] == 0
    assert not blocks.exists()


def test_block_count_is_zero_when_nomination_retention_is_refused(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)

    class Nominating:
        def chat(self, **kwargs):
            if kwargs["messages"][0]["content"].startswith("Compare this draft memory"):
                return {"content": "Episode, checked against its source.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps(
                    [{"topic": "people/alex", "content": "A durable understanding."}])}, {"cost": 0.01}
            return {"content": "Episode.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps(
                [{"topic": "people/alex", "content": "A durable understanding."}])}, {"cost": 0.01}

    monkeypatch.setattr(c, "append_jsonl", lambda *a, **k: False)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="refused")
    usage = c.consolidate(chat, blocks, meta, Nominating(), knowledge_context=ctx)
    assert usage["_blocks_written"] == 0
    assert not blocks.exists()  # the cursor and the blocks are both preserved for retry


@pytest.mark.parametrize("usage, blocks_written, error_kind", [
    ({"cost": 0.25, "prompt_tokens": 10, "_blocks_written": 2,
      "_consolidation_errors": [{"kind": "context_overflow"}, {"kind": "provider_failed"}]}, 2, "provider_failed"),
    ({"cost": 0.0, "_blocks_written": 0, "_consolidation_errors": []}, 0, None),
])
def test_the_event_row_carries_the_block_count_and_the_last_error_kind(
    tmp_path, monkeypatch, usage, blocks_written, error_kind,
):
    """post_task_synthesis turns the usage receipt into the observable event row."""
    import supervisor.state as state

    from ouroboros import post_task_synthesis as pts

    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda p: tmp_path / p)
    memory = SimpleNamespace(load_identity=lambda: "identity")
    monkeypatch.setattr(c, "should_consolidate", lambda *_a, **_k: True)
    monkeypatch.setattr(c, "consolidate", lambda **_k: usage)
    monkeypatch.setattr(state, "update_budget_from_usage", lambda *_a, **_k: None)

    pts._run_chat_consolidation(env, memory, object(), {"id": "task-1"}, logs)

    row = json.loads((logs / "events.jsonl").read_text().splitlines()[-1])
    assert row["type"] == "chat_block_consolidation"
    assert row["blocks_written"] == blocks_written
    assert row["last_error_kind"] == error_kind


# --- unpublished nominations leave a receipt that survives era compression --------


class _Nominating:
    """One nomination per block; the caller decides whether publication succeeds."""

    def __init__(self, topic="people/alex"):
        self.topic, self.count = topic, 0

    def chat(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if prompt.startswith("Compress these older memory blocks"):
            return {"content": "The full historical span remains represented."}, {"cost": 0.01}
        if prompt.startswith("Compare this draft memory"):
            # The correction returns the checked text with the draft's nomination
            # block carried through the same source check; that block is released.
            return {"content": f"Episode {self.count}, checked against its source.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps(
                [{"topic": self.topic, "content": f"Understanding {self.count}."}])}, {"cost": 0.01}
        self.count += 1
        return {"content": f"Episode {self.count}.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps(
            [{"topic": self.topic, "content": f"Understanding {self.count}."}])}, {"cost": 0.01}


def test_partial_publication_records_the_batch_receipt_in_meta(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    monkeypatch.setattr(c, "_write_knowledge_entries",
                        lambda *_a, **_k: [{"topic": "people/alex", "ok": False, "reason": "revision_conflict"}])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="partial")
    c.consolidate(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    pending = json.loads(meta.read_text())["pending_knowledge_nominations"]
    assert len(pending) == 1
    assert pending[0]["topic"] == "people/alex" and pending[0]["reason"] == "revision_conflict"
    assert pending[0]["id"].endswith(":0:0")


def test_new_success_does_not_erase_an_old_failed_entry_or_legacy_receipt(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    legacy = {"entry_id": "old", "failed": 3, "total": 4}
    c.atomic_write_json(meta, {"last_unpublished_nominations": legacy})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="clean")
    original = c._write_knowledge_entries
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"topic": "people/alex", "scope": "global", "ok": False,
                     "reason": "revision_conflict"}]
        return original(*args, **kwargs)

    monkeypatch.setattr(c, "_write_knowledge_entries", fail_once)
    c.consolidate(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    older = json.loads(meta.read_text())["pending_knowledge_nominations"][0]
    _write_chat(chat, count=200, text_size=0)
    c.consolidate(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    saved = json.loads(meta.read_text())
    assert saved["last_unpublished_nominations"] == legacy
    assert saved["pending_knowledge_nominations"] == [older]
    assert calls == 2


def test_a_run_without_nominations_leaves_the_receipt_alone(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    standing = {"entry_id": "old", "failed": 2, "total": 5}
    c.atomic_write_json(meta, {"last_unpublished_nominations": standing})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="silent")
    c.consolidate(chat, blocks, meta, _LLM(), knowledge_context=ctx)
    assert json.loads(meta.read_text())["last_unpublished_nominations"] == standing


def test_the_receipt_survives_era_compression(tmp_path, fit, monkeypatch):
    """Era compression replaces blocks with an object carrying no knowledge_writes."""
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=1100, text_size=0)
    monkeypatch.setattr(c, "_write_knowledge_entries",
                        lambda _shelf, entries, **_k: [{"topic": "people/alex", "ok": False,
                                                        "reason": "revision_conflict"} for _ in entries])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="era")
    c.consolidate(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    saved_blocks = json.loads(blocks.read_text())
    assert saved_blocks[0]["type"] == "era"
    assert "knowledge_writes" not in saved_blocks[0]
    pending = json.loads(meta.read_text())["pending_knowledge_nominations"]
    assert len(pending) == 11
    assert len({row["id"] for row in pending}) == 11


# --- the Health block is where stale memory becomes visible -----------------------


def test_health_names_an_incomplete_publication_with_its_recovery_route(tmp_path):
    env = _health_env(tmp_path)
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
                        {"last_unpublished_nominations": {"entry_id": "abc123", "failed": 2, "total": 5}})
    lines = context_health._memory_health_lines(env)
    row = next(line for line in lines if "PUBLICATION INCOMPLETE" in line)
    assert "2 of 5 nominations" in row and "abc123" in row
    assert "memory/knowledge_history.jsonl" in row and "read_file(root='runtime_data'" in row


def test_health_names_the_last_consolidation_failure(tmp_path):
    env = _health_env(tmp_path)
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
                        {"last_consolidation_error": {"kind": "context_overflow", "cursor_offset": 400}})
    row = next(line for line in context_health._memory_health_lines(env)
               if "LAST DIALOGUE CONSOLIDATION FAILED" in line)
    assert "kind=context_overflow" in row and "at cursor 400" in row


def test_health_stays_silent_when_the_pipeline_is_healthy(tmp_path):
    env = _health_env(tmp_path)
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {"last_consolidated_offset": 100})
    lines = context_health._memory_health_lines(env)
    assert not any("DIALOGUE" in line for line in lines)


def test_health_lines_carry_no_timestamp(tmp_path):
    """These are latest-run STATE, not events: a clock in them would read as freshness."""
    env = _health_env(tmp_path)
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
                        {"last_unpublished_nominations": {"entry_id": "abc", "failed": 1, "total": 1},
                         "last_consolidation_error": {"kind": "provider_failed", "cursor_offset": 0,
                                                      "ts": "2026-09-14T00:00:00Z"}})
    dialogue = [line for line in context_health._memory_health_lines(env) if "DIALOGUE" in line]
    assert len(dialogue) == 2
    assert not any("2026-" in line for line in dialogue)


def test_memory_health_lines_still_carry_identity_and_scratchpad(tmp_path):
    """The extracted helper keeps the existing own-memory checks it was split from."""
    env = _health_env(tmp_path)
    (tmp_path / "memory" / "identity.md").write_text("thin", encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text("x", encoding="utf-8")
    lines = context_health._memory_health_lines(env)
    assert any("THIN IDENTITY" in line for line in lines)
    assert any("EMPTY SCRATCHPAD" in line for line in lines)


@pytest.mark.parametrize("payload", [{"last_unpublished_nominations": "corrupt"},
                                     {"last_unpublished_nominations": {"failed": 0, "total": 3}},
                                     {"last_consolidation_error": "corrupt"}])
def test_unreadable_receipts_do_not_raise_or_shout(tmp_path, payload):
    env = _health_env(tmp_path)
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", payload)
    assert not any("DIALOGUE" in line for line in context_health._memory_health_lines(env))


def test_invalid_legacy_receipt_does_not_impersonate_unreadable_meta(tmp_path):
    env = _health_env(tmp_path)
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
                        {"last_unpublished_nominations": {"failed": "many", "total": 2}})
    lines = context_health._memory_health_lines(env)
    assert any("LEGACY NOMINATION RECEIPT INVALID" in line for line in lines)
    assert not any("DIALOGUE META UNREADABLE" in line for line in lines)


def test_pending_receipt_precedes_the_note_writer_and_cannot_be_replaced_by_corrupt_meta(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="interrupted")

    def interrupted(*_args, **_kwargs):
        saved = json.loads(meta.read_text())
        assert len(saved["pending_knowledge_nominations"]) == 1
        assert saved["pending_knowledge_nominations"][0]["reason"] == "publication_pending"
        raise RuntimeError("simulated stop after pending publication")

    monkeypatch.setattr(c, "_write_knowledge_entries", interrupted)
    with pytest.raises(RuntimeError, match="simulated stop"):
        c.consolidate(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    saved = json.loads(meta.read_text())
    assert saved["pending_knowledge_nominations"][0]["reason"] == "publication_pending"
    assert saved.get("last_consolidated_offset", 0) == 0


def test_health_projects_three_owed_addresses_and_omission_count(tmp_path):
    env = _health_env(tmp_path)
    rows = [{"id": f"source{i}:0:0", "scope": "global", "topic": f"people/{i}",
             "reason": "revision_conflict"} for i in range(5)]
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
                        {"pending_knowledge_nominations": rows})
    lines = context_health._memory_health_lines(env)
    row = next(line for line in lines if "KNOWLEDGE PUBLICATION OPEN" in line)
    assert "5 source-addressed" in row and "first 3" in row and "omitted 2" in row
    assert "source0" in row and "source2" in row and "source3" not in row
    assert "memory/knowledge_history.jsonl" in row


def test_malformed_nomination_keeps_its_position_and_cannot_retire_another_entry(tmp_path):
    from ouroboros.memory_nomination_receipts import prepare, settle

    meta = {}
    ids = prepare(meta, "source", [({}, [None, {"topic": "people/alex", "content": "Valid"}])])
    outcomes = c._write_knowledge_entries(tmp_path / "memory" / "knowledge", [None,
        {"topic": "people/alex", "content": "Valid"}])
    assert len(outcomes) == 2 and outcomes[0]["reason"] == "malformed_nomination"
    assert outcomes[1]["ok"]
    settle(meta, ids, outcomes)
    assert [row["id"] for row in meta["pending_knowledge_nominations"]] == ["source:0:0"]


def test_corrupt_obligation_index_refuses_replacement(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    c.atomic_write_json(meta, {"pending_knowledge_nominations": {"not": "a list"}})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="corrupt")
    with pytest.raises(ValueError, match="refusing to replace"):
        c.consolidate(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    assert not blocks.exists()
    assert json.loads(meta.read_text())["pending_knowledge_nominations"] == {"not": "a list"}


@pytest.mark.parametrize("bad_bytes", [b'{"pending_knowledge_nominations":[{"id":"old"}]',
                                       b'["wrong top-level type"]',
                                       b'{"pending_knowledge_nominations":[],"pending_knowledge_nominations":[]}'])
def test_unreadable_existing_meta_cannot_erase_obligations(tmp_path, fit, bad_bytes):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_bytes(bad_bytes)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="corrupt")
    with pytest.raises(ValueError):
        c.should_consolidate(meta, chat)
    with pytest.raises(ValueError):
        c.consolidate(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    assert meta.read_bytes() == bad_bytes
    assert not blocks.exists()
    assert any("DIALOGUE META UNREADABLE" in line for line in
               context_health._memory_health_lines(_health_env(tmp_path)))


def test_pending_health_disambiguates_two_entries_from_one_source(tmp_path):
    env = _health_env(tmp_path)
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {
        "pending_knowledge_nominations": [
            {"id": "a" * 64 + f":{index}:0", "reason": "revision_conflict"}
            for index in (0, 1)]})
    row = next(line for line in context_health._memory_health_lines(env)
               if "KNOWLEDGE PUBLICATION OPEN" in line)
    assert "aaaaaaaaaaaa:0:0" in row and "aaaaaaaaaaaa:1:0" in row
