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
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.utils import atomic_write_json
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers
from tests.test_consolidator_context_fit import _LLM, _Refusal, _paths, _write_chat

fit = fit_helpers.fit


def consolidate_closed(*args, **kwargs):
    return c.consolidate(*args, completed_task={"id": "fixture"}, **kwargs)




def _health_env(tmp_path):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path,
                           repo_path=lambda p: tmp_path / p, drive_path=lambda p: tmp_path / p)


# --- stale error is retired only by a run that recorded none of its own -----------


def test_a_chronicle_only_pass_still_reports_zero_written_blocks(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=5, text_size=0)
    store = ChronicleStore(tmp_path)
    store.import_legacy()
    store.append_episode("1", "Earlier meaning. " * 3000, [], {"kind": "mind"})
    usage = c.consolidate(chat, blocks, meta, _LLM(), compact_chronicle=True, pressure_fits=lambda: False)
    assert usage["_blocks_written"] == 0  # raw room remains open; only a digest was made
    assert store.records(kinds=["digest"])



def test_a_chronicle_pass_after_a_real_run_keeps_the_written_count(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, text_size=0)
    usage = consolidate_closed(chat, blocks, meta, _LLM(), compact_chronicle=True, pressure_fits=lambda: False)
    assert usage["_blocks_written"] == 1
    assert len(ChronicleStore(tmp_path).records(kinds=["episode"])) == 1



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
    atomic_write_json(meta, {"last_consolidated_offset": 0,
                               "chat_log_signature": c._chat_log_signature(chat),
                               "last_consolidation_error": {"kind": "context_overflow", "cursor_offset": 0}})
    consolidate_closed(chat, blocks, meta, _LLM())
    saved = ChronicleStore(tmp_path).scan_state()
    assert saved["last_consolidated_offset"] == 100
    assert "last_consolidation_error" not in saved



def test_a_run_that_never_advances_keeps_the_stale_error(tmp_path, fit):
    """No cursor movement, no proof: the earlier failure is still the latest word."""
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    stale = {"kind": "provider_failed", "cursor_offset": 0}
    atomic_write_json(meta, {"last_consolidated_offset": 0,
                               "chat_log_signature": c._chat_log_signature(chat),
                               "last_consolidation_error": stale})

    def refuse(_llm, _prompt):
        raise _Refusal("auth failed", code="invalid_api_key")

    consolidate_closed(chat, blocks, meta, _LLM(effect=refuse))
    assert ChronicleStore(tmp_path).scan_state()["last_consolidation_error"]["kind"]



# --- every non-None return carries its own block count ---------------------------


def test_block_count_is_reported_on_a_successful_run(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=200, text_size=0)
    usage = consolidate_closed(chat, blocks, meta, _LLM())
    assert usage["_blocks_written"] == 1



def test_block_count_is_zero_when_nothing_was_written(tmp_path, fit):
    """An empty summary writes no block; the receipt says 0 rather than going absent."""
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)

    def empty(_llm, _prompt):
        return {"content": "   "}, {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1, "cost": 0.0}

    usage = consolidate_closed(chat, blocks, meta, _LLM(effect=empty))
    assert usage["_blocks_written"] == 0
    assert not blocks.exists()



def test_original_and_nomination_debt_survive_a_history_write_failure(tmp_path, fit, monkeypatch):
    from ouroboros import utils
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=1, text_size=0)
    real_append = utils.append_jsonl
    def fail_history(path, *args, **kwargs):
        return False if str(path).endswith("knowledge_history.jsonl") else real_append(path, *args, **kwargs)
    monkeypatch.setattr(utils, "append_jsonl", fail_history)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="refused")
    with pytest.raises(OSError, match="nominations could not be retained"):
        consolidate_closed(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    store = ChronicleStore(tmp_path)
    assert len(store.records(kinds=["episode"])) == 1
    assert len(store.records(kinds=["revision"])) == 1
    assert store.scan_state()["pending_knowledge_nominations"][0]["reason"] == "publication_pending"
    assert not blocks.exists()



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
    consolidate_closed(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    pending = ChronicleStore(tmp_path).scan_state()["pending_knowledge_nominations"]
    assert len(pending) == 1
    assert pending[0]["topic"] == "people/alex" and pending[0]["reason"] == "revision_conflict"
    assert pending[0]["id"].endswith(":0:0")



def test_new_success_does_not_erase_an_old_failed_entry_or_legacy_receipt(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    legacy = {"entry_id": "old", "failed": 3, "total": 4}
    atomic_write_json(meta, {"last_unpublished_nominations": legacy})
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
    consolidate_closed(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    older = ChronicleStore(tmp_path).scan_state()["pending_knowledge_nominations"][0]
    _write_chat(chat, count=200, text_size=0)
    consolidate_closed(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    saved = ChronicleStore(tmp_path).scan_state()
    assert saved["last_unpublished_nominations"] == legacy
    assert saved["pending_knowledge_nominations"] == [older]
    assert calls == 2



def test_a_run_without_nominations_leaves_the_receipt_alone(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    standing = {"entry_id": "old", "failed": 2, "total": 5}
    atomic_write_json(meta, {"last_unpublished_nominations": standing})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="silent")
    consolidate_closed(chat, blocks, meta, _LLM(), knowledge_context=ctx)
    assert ChronicleStore(tmp_path).scan_state()["last_unpublished_nominations"] == standing



def test_the_receipt_survives_room_digest_publication(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    monkeypatch.setattr(c, "_write_knowledge_entries", lambda _shelf, entries, **_k: [
        {"topic": "people/alex", "ok": False, "reason": "revision_conflict"} for _ in entries])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="digest")
    consolidate_closed(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    store = ChronicleStore(tmp_path)
    pending = store.scan_state()["pending_knowledge_nominations"]
    store.append_episode("1", "A larger coherent account. " * 100, [], {"kind": "mind"})
    consolidate_closed(chat, blocks, meta, _LLM(), knowledge_context=ctx,
                       compact_chronicle=True, pressure_fits=lambda: False)
    assert store.records(kinds=["digest"])
    assert store.scan_state()["pending_knowledge_nominations"] == pending and pending



# --- the Health block is where stale memory becomes visible -----------------------


def test_health_names_an_incomplete_publication_with_its_recovery_route(tmp_path):
    env = _health_env(tmp_path)
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
                        {"last_unpublished_nominations": {"entry_id": "abc123", "failed": 2, "total": 5}})
    lines = context_health._memory_health_lines(env)
    row = next(line for line in lines if "PUBLICATION INCOMPLETE" in line)
    assert "2 of 5 nominations" in row and "abc123" in row
    assert "memory/knowledge_history.jsonl" in row and "read_file(root='runtime_data'" in row


def test_health_names_the_last_consolidation_failure(tmp_path):
    env = _health_env(tmp_path)
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
                        {"last_consolidation_error": {"kind": "context_overflow", "cursor_offset": 400}})
    row = next(line for line in context_health._memory_health_lines(env)
               if "LAST DIALOGUE CONSOLIDATION FAILED" in line)
    assert "kind=context_overflow" in row and "at cursor 400" in row


def test_activated_health_distinguishes_old_era_receipt_from_current_failure(tmp_path):
    env = _health_env(tmp_path)
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {
        "era_retry": {"source_sha256": "old-source", "route": {"model": "past-route"}}})
    before = "\n".join(context_health._memory_health_lines(env))
    assert "ERA COMPRESSION WITHHELD" in before
    store = ChronicleStore(tmp_path)
    store.import_legacy()
    scan = store.scan_state()
    scan["last_consolidation_error"] = {"kind": "source_incomplete", "source_ref": {"task_id": "t"}}
    store.publish([], scan_state=scan)
    after = "\n".join(context_health._memory_health_lines(env))
    assert "LEGACY HISTORY" in after and "1 old era compression attempt" in after
    assert "ERA COMPRESSION WITHHELD" not in after and "cursor None" not in after
    assert "kind=source_incomplete" in after and "memory/chronicle/records.jsonl" in after


def test_health_stays_silent_when_the_pipeline_is_healthy(tmp_path):
    env = _health_env(tmp_path)
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {"last_consolidated_offset": 100})
    lines = context_health._memory_health_lines(env)
    assert not any("DIALOGUE" in line for line in lines)


def test_health_lines_carry_no_timestamp(tmp_path):
    """These are latest-run STATE, not events: a clock in them would read as freshness."""
    env = _health_env(tmp_path)
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
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
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", payload)
    assert not any("DIALOGUE" in line for line in context_health._memory_health_lines(env))


def test_invalid_legacy_receipt_does_not_impersonate_unreadable_meta(tmp_path):
    env = _health_env(tmp_path)
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
                        {"last_unpublished_nominations": {"failed": "many", "total": 2}})
    lines = context_health._memory_health_lines(env)
    assert any("LEGACY NOMINATION RECEIPT INVALID" in line for line in lines)
    assert not any("DIALOGUE META UNREADABLE" in line for line in lines)


def test_pending_receipt_precedes_the_note_writer_and_cannot_be_replaced_by_corrupt_meta(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="interrupted")

    def interrupted(*_args, **_kwargs):
        saved = ChronicleStore(tmp_path).scan_state()
        assert len(saved["pending_knowledge_nominations"]) == 1
        assert saved["pending_knowledge_nominations"][0]["reason"] == "publication_pending"
        raise RuntimeError("simulated stop after pending publication")

    monkeypatch.setattr(c, "_write_knowledge_entries", interrupted)
    with pytest.raises(RuntimeError, match="simulated stop"):
        consolidate_closed(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    saved = ChronicleStore(tmp_path).scan_state()
    assert saved["pending_knowledge_nominations"][0]["reason"] == "publication_pending"
    assert saved.get("last_consolidated_offset", 0) == 0



def test_health_projects_three_owed_addresses_and_omission_count(tmp_path):
    env = _health_env(tmp_path)
    rows = [{"id": f"source{i}:0:0", "scope": "global", "topic": f"people/{i}",
             "reason": "revision_conflict"} for i in range(5)]
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json",
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
    atomic_write_json(meta, {"pending_knowledge_nominations": {"not": "a list"}})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="corrupt")
    before = meta.read_bytes()
    usage = consolidate_closed(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    assert usage["_blocks_written"] == 0 and not blocks.exists()
    assert meta.read_bytes() == before
    store = ChronicleStore(tmp_path)
    assert store.activation() and not store.records(kinds=["episode"])
    gaps = [row for row in store.records() if row.get("metadata", {}).get("source_gap")]
    assert gaps
    from ouroboros.artifacts import read_actor_source_bytes
    assert any(read_actor_source_bytes(tmp_path, ref["task_id"], ref) == before
               for row in gaps for ref in row.get("source_refs", []))



@pytest.mark.parametrize("bad_bytes", [b'{"pending_knowledge_nominations":[{"id":"old"}]',
                                       b'["wrong top-level type"]',
                                       b'{"pending_knowledge_nominations":[],"pending_knowledge_nominations":[]}'])
def test_unreadable_existing_meta_cannot_erase_obligations(tmp_path, fit, bad_bytes):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=100, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_bytes(bad_bytes)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="corrupt")
    assert c.should_consolidate(meta, chat)
    usage = consolidate_closed(chat, blocks, meta, _Nominating(), knowledge_context=ctx)
    assert usage["_blocks_written"] == 0
    assert meta.read_bytes() == bad_bytes and not blocks.exists()
    store = ChronicleStore(tmp_path)
    assert store.activation() and not store.records(kinds=["episode"])
    gaps = [row for row in store.records() if row.get("metadata", {}).get("source_gap")]
    assert gaps and store.scan_state()["last_consolidated_offset"] == 100
    from ouroboros.artifacts import read_actor_source_bytes
    assert any(read_actor_source_bytes(tmp_path, ref["task_id"], ref) == bad_bytes
               for row in gaps for ref in row.get("source_refs", []))



def test_pending_health_disambiguates_two_entries_from_one_source(tmp_path):
    env = _health_env(tmp_path)
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {
        "pending_knowledge_nominations": [
            {"id": "a" * 64 + f":{index}:0", "reason": "revision_conflict"}
            for index in (0, 1)]})
    row = next(line for line in context_health._memory_health_lines(env)
               if "KNOWLEDGE PUBLICATION OPEN" in line)
    assert "aaaaaaaaaaaa:0:0" in row and "aaaaaaaaaaaa:1:0" in row


# --- the event row measures what the run covered (#1321) ------------------------


def _two_room_chat(chat):
    rows = _write_chat(chat, count=200, text_size=0)
    for row in rows[::2]:
        row["chat_id"] = 2
    chat.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _event_row(tmp_path, monkeypatch, usage):
    import supervisor.state as state
    from ouroboros import post_task_synthesis as pts

    logs = tmp_path / "event-logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda p: tmp_path / p)
    monkeypatch.setattr(c, "should_consolidate", lambda *_a, **_k: True)
    monkeypatch.setattr(c, "consolidate", lambda **_k: usage)
    monkeypatch.setattr(state, "update_budget_from_usage", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.chronicle_view.maintenance_projection", lambda *_a: (None, {}))
    pts._run_chat_consolidation(env, SimpleNamespace(load_identity=lambda: "identity"), object(), {"id": "t"}, logs)
    return json.loads((logs / "events.jsonl").read_text().splitlines()[-1])


def test_the_event_row_measures_published_and_withheld_room_sources(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _two_room_chat(chat)
    def refuse(llm, _prompt):
        if len(llm.calls) == 3:
            raise _Refusal("down", code="invalid_api_key")
    usage = consolidate_closed(chat, blocks, meta, _LLM(effect=refuse))
    assert usage["_blocks_written"] == 1
    accepted, withheld = usage["_coverage"]
    assert (accepted["status"], accepted["rooms"], accepted["messages"]) == ("accepted", 1, 100)
    assert (withheld["status"], withheld["rooms"], withheld["messages"], withheld["output_chars"]) == ("withheld", 1, 100, 0)
    row = _event_row(tmp_path, monkeypatch, usage)
    coverage = row["coverage"]
    assert coverage["attempted"]["count"] == 2 and coverage["attempted"]["messages"] == 200
    assert coverage["accepted"]["count"] == coverage["withheld"]["count"] == 1
    assert coverage["split_attempts"] == 0 and coverage["eras"]["count"] == 0
    ratio = coverage["accepted_output_to_source"]
    assert ratio["denominator"] == "source chars of accepted chunks"
    assert ratio["ratio"] == round(accepted["output_chars"] / accepted["source_chars"], 4)
    assert row["cost_usd"] is None



def test_a_clean_zero_cost_run_reports_zero_and_nothing_attempted_has_no_ratio(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _two_room_chat(chat)
    usage = consolidate_closed(chat, blocks, meta, _LLM(usage={"prompt_tokens": 1, "completion_tokens": 1,
                                                          "total_tokens": 2, "cost": 0.0}))
    row = _event_row(tmp_path, monkeypatch, usage)
    assert row["cost_usd"] == 0.0 and row["coverage"]["accepted"]["count"] == 2
    assert row["coverage"]["withheld"]["count"] == 0 and row["coverage"]["split_attempts"] == 0
    empty = _event_row(tmp_path, monkeypatch, {"cost": 0.0, "_blocks_written": 0, "_consolidation_errors": []})
    assert empty["coverage"]["attempted"]["count"] == 0
    assert empty["coverage"]["accepted_output_to_source"]["ratio"] is None



def test_context_refusal_and_a_later_failed_room_never_claim_recovery(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _two_room_chat(chat)
    fit.window = None
    def refuse(llm, _prompt):
        raise _Refusal() if len(llm.calls) == 1 else _Refusal("down", code="invalid_api_key")
    usage = consolidate_closed(chat, blocks, meta, _LLM(effect=refuse))
    assert usage["_blocks_written"] == 0 and not blocks.exists()
    assert len(usage["_coverage"]) == 2
    assert all(row["status"] == "withheld" and row["output_chars"] == 0 for row in usage["_coverage"])
    coverage = _event_row(tmp_path, monkeypatch, usage)["coverage"]
    assert coverage["accepted"]["count"] == 0 and coverage["withheld"]["count"] == 2
    assert coverage["split_attempts"] == 0 and coverage["accepted_output_to_source"]["ratio"] is None
    assert "recover" not in json.dumps(coverage)
