"""Nothing in memory maintenance leaves silently (TZ-3 PR-1, invariant I4).

Every host decision that used to vanish is a typed fact, each proven in both
directions: a consolidation skipped on the lock, an era withheld because it was
not shorter (with its ``era_retry`` record), a scratchpad pass and its outcome,
a reflection lesson the host declined, and the ``writer``/``route``/
``writer_input_ref``/``old_chars``/``new_chars`` stamp on every
``source_capture`` history row. An era is built from summary blocks, never from
an earlier era. The reader-less ``knowledge_journal.jsonl`` writer is gone.
"""

from __future__ import annotations

import inspect
import json
import os
from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c
from ouroboros import context_health
from ouroboros import knowledge as store
from ouroboros import reflection
from ouroboros.memory import Memory
from ouroboros.tools import knowledge as knowledge_tools
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers
from tests.test_consolidator_context_fit import _LLM, _paths, _write_chat

fit = fit_helpers.fit


def _events(root, kind):
    path = root / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == kind]


def _history(root):
    path = root / "memory" / "knowledge_history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary_blocks(count, start=0):
    return [{"ts": "2026-01-01T00:00:00Z", "type": "summary", "range": f"2026-01-01 {i:02d}:00 - {i:02d}:59",
             "message_count": 1, "content": f"block-{i} " + "x" * 40} for i in range(start, start + count)]


def _era_block(label="old"):
    return {"ts": "2025-12-31T00:00:00Z", "type": "era", "range": "2025-12-01 to 2025-12-31",
            "message_count": 4, "content": f"### Era: {label}\n" + "e" * 30}


# --- the consolidation lock ---------------------------------------------------------


def test_a_lock_skip_is_a_typed_event_and_a_free_run_is_not(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    holder = os.open(str(meta.parent / ".consolidation.lock"), os.O_CREAT | os.O_WRONLY, 0o644)
    c._lock_nb(holder)
    try:
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="held")
        assert c.consolidate(chat, blocks, meta, _LLM(), knowledge_context=ctx) is None
    finally:
        c._unlock(holder)
        os.close(holder)
    skipped = _events(tmp_path, "consolidation_skipped_locked")
    assert len(skipped) == 1 and skipped[0]["task_id"] == "held"
    assert skipped[0]["lock_path"].endswith(".consolidation.lock")
    assert not blocks.exists()  # the holder owned the run; nothing was consolidated twice

    assert c.consolidate(chat, blocks, meta, _LLM())["_blocks_written"] == 1
    assert len(_events(tmp_path, "consolidation_skipped_locked")) == 1


# --- an era is a compression of summary blocks, never of an era ---------------------


def _seed_run(tmp_path, old_blocks, *, chat_count=c.BLOCK_SIZE):
    chat, blocks_path, meta_path = _paths(tmp_path)
    _write_chat(chat, count=chat_count, text_size=2)
    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    blocks_path.write_text(json.dumps(old_blocks), encoding="utf-8")
    return chat, blocks_path, meta_path


def _fake_era(monkeypatch, *, shorter):
    seen = []

    def fake(run, *_args, **_kwargs):
        seen.append(list(run))
        source_len = sum(len(b["content"]) for b in run)
        content = "e" * (max(1, source_len // 4) if shorter else source_len + 10)
        return {"type": "era", "range": "era", "message_count": len(run), "content": content}, {
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.01}

    monkeypatch.setattr(c, "_compress_blocks_to_era", fake)
    return seen


def test_an_earlier_era_bounds_the_run_and_is_never_recompressed(tmp_path, fit, monkeypatch):
    old = [_era_block(), *_summary_blocks(9)]
    chat, blocks_path, meta_path = _seed_run(tmp_path, old)
    seen = _fake_era(monkeypatch, shorter=True)
    assert c._run_block_consolidation(chat, blocks_path, meta_path, _LLM(), "", force_tail=True)
    run = old[1:1 + c.ERA_COMPRESS_COUNT]
    assert seen == [run]  # the oldest run of summary blocks, the era ahead of it excluded
    stored = json.loads(blocks_path.read_text(encoding="utf-8"))
    assert stored[0] == old[0]  # the old era keeps its place and bytes
    assert stored[1]["type"] == "era" and stored[1]["content"] != old[0]["content"]
    assert stored[2:2 + len(old) - 1 - c.ERA_COMPRESS_COUNT] == old[1 + c.ERA_COMPRESS_COUNT:]  # the rest untouched


def test_eras_ahead_of_the_run_never_hide_the_later_summaries(tmp_path, fit, monkeypatch):
    # Once the oldest four blocks were eras, a fixed four-block window found no
    # summary to compress and the later summaries were never compressed (review F1).
    old = [_era_block(str(i)) for i in range(c.ERA_COMPRESS_COUNT)] + _summary_blocks(6)
    chat, blocks_path, meta_path = _seed_run(tmp_path, old)
    seen = _fake_era(monkeypatch, shorter=True)
    assert c._run_block_consolidation(chat, blocks_path, meta_path, _LLM(), "", force_tail=True)
    run = old[c.ERA_COMPRESS_COUNT:2 * c.ERA_COMPRESS_COUNT]
    assert seen == [run]
    stored = json.loads(blocks_path.read_text(encoding="utf-8"))
    assert stored[:c.ERA_COMPRESS_COUNT] == old[:c.ERA_COMPRESS_COUNT]  # the old eras keep their bytes
    assert stored[c.ERA_COMPRESS_COUNT]["type"] == "era" and stored[c.ERA_COMPRESS_COUNT] not in old
    assert stored[c.ERA_COMPRESS_COUNT + 1:-1] == old[2 * c.ERA_COMPRESS_COUNT:]
    assert len(stored) == len(old) + 1 - c.ERA_COMPRESS_COUNT + 1


def test_a_history_of_eras_alone_makes_no_call_and_keeps_every_block(tmp_path, fit, monkeypatch):
    old = [_era_block(str(i)) for i in range(c.MAX_SUMMARY_BLOCKS)]
    chat, blocks_path, meta_path = _seed_run(tmp_path, old)
    seen = _fake_era(monkeypatch, shorter=True)
    assert c._run_block_consolidation(chat, blocks_path, meta_path, _LLM(), "", force_tail=True)
    assert seen == []  # the newest summary block is never its own era; nothing else is compressible
    stored = json.loads(blocks_path.read_text(encoding="utf-8"))
    assert stored[:len(old)] == old and len(stored) == len(old) + 1


def test_the_chronicle_pass_treats_eras_and_gaps_as_boundaries(tmp_path, fit, monkeypatch):
    blocks_path = tmp_path / "memory" / "dialogue_blocks.json"
    blocks_path.parent.mkdir(parents=True)
    gap = {"gap_id": "g1", "type": "gap", "content": "[MEMORY GAP]"}
    summaries = _summary_blocks(3)
    blocks = [_era_block(), summaries[0], summaries[1], gap, summaries[2]]
    blocks_path.write_text(json.dumps(blocks), encoding="utf-8")
    seen = _fake_era(monkeypatch, shorter=True)
    c._compact_chronicle(blocks_path, _LLM(), "", None)
    assert seen == [summaries[:2], summaries[2:]]
    stored = json.loads(blocks_path.read_text(encoding="utf-8"))
    assert stored[0] == blocks[0] and stored[2] == gap
    assert [b["type"] for b in stored] == ["era", "era", "gap", "era"]


def test_the_chronicle_pass_consults_and_records_the_same_era_retry(tmp_path, fit, monkeypatch):
    # Review F2: a throwaway meta let every pressure pass pay again for a run that was not
    # shorter, and a recorded refusal (no call, no usage) would have crashed the pass.
    blocks_path = tmp_path / "memory" / "dialogue_blocks.json"
    meta_path = tmp_path / "memory" / "dialogue_meta.json"
    blocks_path.parent.mkdir(parents=True)
    blocks = _summary_blocks(3)
    blocks_path.write_text(json.dumps(blocks), encoding="utf-8")
    c.atomic_write_json(meta_path, {"last_consolidated_offset": 7})
    seen = _fake_era(monkeypatch, shorter=False)

    first = c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert len(seen) == 1 and first["cost"] == 0.01
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["last_consolidated_offset"] == 7  # the rest of meta survives the record
    (record,) = meta["era_retry"].values()
    assert record["route"] == {"model": "test/model", "use_local": False}
    assert json.loads(blocks_path.read_text(encoding="utf-8")) == blocks

    second = c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert len(seen) == 1  # the recorded refusal made no second paid call
    assert second["cost"] == 0 and second["_consolidation_errors"] == []  # no call was made
    assert [e["attempted"] for e in _events(tmp_path, "era_not_shorter")] == [True, False]
    assert json.loads(blocks_path.read_text(encoding="utf-8")) == blocks

    # A pass without a meta path still pays the attempt; it reads and records no era_retry.
    assert c._compact_chronicle(blocks_path, _LLM(), "", None)["cost"] == 0.01
    assert len(seen) == 2 and json.loads(meta_path.read_text(encoding="utf-8")) == meta


def test_each_run_keeps_its_own_refusal_and_a_success_erases_only_its_own(tmp_path, fit, monkeypatch):
    # Review N1: one refusal record for the whole chronicle paid again for run A on every
    # pass once run B's refusal overwrote it, and run B's success erased run A's record.
    blocks_path = tmp_path / "memory" / "dialogue_blocks.json"
    meta_path = tmp_path / "memory" / "dialogue_meta.json"
    blocks_path.parent.mkdir(parents=True)
    gap = {"gap_id": "g1", "type": "gap", "content": "[MEMORY GAP]"}
    run_a, run_b = _summary_blocks(2), [{**b, "range": "b-" + b["range"]} for b in _summary_blocks(2)]
    blocks_path.write_text(json.dumps([*run_a, gap, *run_b]), encoding="utf-8")
    c.atomic_write_json(meta_path, {})
    seen = _fake_era(monkeypatch, shorter=False)
    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert seen == [run_a, run_b]
    runs = c._era_retry_runs(json.loads(meta_path.read_text(encoding="utf-8")))
    assert len(runs) == 2  # both refusals remembered, keyed by source
    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert seen == [run_a, run_b]  # neither run is paid for again

    seen.clear()
    shorter = _fake_era(monkeypatch, shorter=True)
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("other/model", False))  # a new route re-attempts
    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert shorter == [run_a, run_b]
    assert "era_retry" not in json.loads(meta_path.read_text(encoding="utf-8"))  # each success cleared its own
    assert [b["type"] for b in json.loads(blocks_path.read_text(encoding="utf-8"))] == ["era", "gap", "era"]


def test_era_retry_is_keyed_on_the_effective_binding_dispatch_uses(tmp_path, fit, monkeypatch):
    # Round 2 (critical 2): dispatch applies the Light account pin and the live
    # model-wait override; a refusal recorded under one effective binding must not
    # suppress the paid retry under another, while an unchanged binding still does.
    from contextlib import nullcontext

    from ouroboros import model_slots, model_wait

    blocks_path = tmp_path / "memory" / "dialogue_blocks.json"
    meta_path = tmp_path / "memory" / "dialogue_meta.json"
    blocks_path.parent.mkdir(parents=True)
    blocks_path.write_text(json.dumps(_summary_blocks(3)), encoding="utf-8")
    c.atomic_write_json(meta_path, {})
    seen = _fake_era(monkeypatch, shorter=False)

    def record():
        (row,) = c._era_retry_runs(json.loads(meta_path.read_text(encoding="utf-8"))).values()
        return row["route"]

    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert len(seen) == 1 and record() == {"model": "test/model", "use_local": False}  # unchanged: suppressed

    # The Light account pin changes; the configured lane does not.
    monkeypatch.setattr(model_slots, "model_role_option", lambda key, role, **_kw: "acct-B" if role == "light" else "")
    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert len(seen) == 2 and record() == {"model": "test/model", "use_local": False, "model_account_override": "acct-B"}
    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert len(seen) == 2  # the same pin: suppressed again

    # A model-wait override rebinds the role; lane and pin are unchanged.
    override = {"model": "override/model", "use_local": False, "model_account_override": "acct-B"}
    waiter = SimpleNamespace(overrides={"light": override}, register_reprepare=lambda role, callback: nullcontext())
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert len(seen) == 3 and record() == override
    c._compact_chronicle(blocks_path, _LLM(), "", None, meta_path=meta_path)
    assert len(seen) == 3
    assert [e["attempted"] for e in _events(tmp_path, "era_not_shorter")] == [True, False, True, False, True, False]

    # One helper feeds both sites: the key IS the binding a real call dispatches on.
    assert c._light_dispatch_binding() == override
    llm = _LLM()
    assert c._call_consolidation_llm(llm, "prompt", "probe")[0] == "summary-1"
    sent = llm.calls[0]
    assert {key: sent[key] for key in override} == override


def test_era_retry_is_keyed_to_the_binding_the_era_call_executed_on(tmp_path, fit, monkeypatch):
    # Round 3 (critical F2): an owner ``switch`` during a model wait INSIDE the era call
    # rebinds the role's override before the paid send, so a key captured before the
    # call named the binding that never answered: the one that did paid again on the
    # next pass, and a fresh attempt on the captured one was suppressed by its outcome.
    from contextlib import nullcontext

    from ouroboros import model_wait

    blocks_path = tmp_path / "memory" / "dialogue_blocks.json"
    meta_path = tmp_path / "memory" / "dialogue_meta.json"
    blocks_path.parent.mkdir(parents=True)
    blocks = _summary_blocks(3)
    blocks_path.write_text(json.dumps(blocks), encoding="utf-8")
    c.atomic_write_json(meta_path, {})
    route_a = {"model": "test/model", "use_local": False}
    route_b = {"model": "switched/model", "use_local": False, "model_account_override": "acct-B"}
    waiter = SimpleNamespace(overrides={}, register_reprepare=lambda role, callback: nullcontext())
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    not_shorter = "e" * (sum(len(b["content"]) for b in blocks) + 10)

    def switch_inside_the_call(llm, prompt):
        # The real era path: the first send dispatched on A; the owner switches the
        # role while it is in flight, and the rest of the unit dispatches on B.
        if len(llm.calls) == 1:
            assert c._light_route() == route_a
            waiter.overrides["light"] = dict(route_b)
        return {"content": not_shorter}, dict(llm.usage)

    def record():
        (row,) = c._era_retry_runs(json.loads(meta_path.read_text(encoding="utf-8"))).values()
        return row["route"]

    llm = _LLM(effect=switch_inside_the_call)
    c._compact_chronicle(blocks_path, llm, "", None, meta_path=meta_path)
    assert llm.calls[0]["model"] == "test/model" and llm.calls[-1]["model"] == "switched/model"
    assert record() == route_b  # keyed to the executed binding, not the one captured before the call
    (event,) = _events(tmp_path, "era_not_shorter")
    assert event["attempted"] is True and event["route"] == route_b
    assert json.loads(blocks_path.read_text(encoding="utf-8")) == blocks

    # The override still binds B: B's own refusal suppresses the paid retry on B.
    still_b = _LLM(effect=lambda llm, prompt: ({"content": not_shorter}, dict(llm.usage)))
    c._compact_chronicle(blocks_path, still_b, "", None, meta_path=meta_path)
    assert still_b.calls == [] and record() == route_b

    # Back on A, which never answered for this run: A is paid for once, then keyed honestly.
    waiter.overrides.clear()
    on_a = _LLM(effect=lambda llm, prompt: ({"content": not_shorter}, dict(llm.usage)))
    c._compact_chronicle(blocks_path, on_a, "", None, meta_path=meta_path)
    assert on_a.calls and all(call["model"] == "test/model" for call in on_a.calls)
    assert record() == route_a
    assert [(e["attempted"], e["route"]) for e in _events(tmp_path, "era_not_shorter")] == [
        (True, route_b), (False, route_b), (True, route_a)]


def test_a_legacy_single_era_retry_record_is_read_as_one_run(tmp_path):
    legacy = {"era_retry": {"source_sha256": "abc", "route": {"model": "m", "use_local": False}}}
    assert c._era_retry_runs(legacy) == {"abc": {"route": {"model": "m", "use_local": False}}}
    assert c._era_retry_runs({"era_retry": "garbage"}) == {} and c._era_retry_runs({}) == {}


# --- a not-shorter era is recorded, visible, and not paid for twice -----------------


def test_a_not_shorter_era_records_era_retry_and_the_event(tmp_path, fit, monkeypatch):
    old = _summary_blocks(c.MAX_SUMMARY_BLOCKS)
    chat, blocks_path, meta_path = _seed_run(tmp_path, old)
    seen = _fake_era(monkeypatch, shorter=False)
    assert c._run_block_consolidation(chat, blocks_path, meta_path, _LLM(), "", force_tail=True)
    assert len(seen) == 1
    stored = json.loads(blocks_path.read_text(encoding="utf-8"))
    assert stored[:c.MAX_SUMMARY_BLOCKS] == old and not any(b["type"] == "era" for b in stored)
    retry = json.loads(meta_path.read_text(encoding="utf-8"))["era_retry"]
    (source_sha256, record), = retry.items()
    assert record == {"route": {"model": "test/model", "use_local": False},
                      "observed_route": store.UNKNOWN_STAMP}  # the fake era usage names no physical route
    events = _events(tmp_path, "era_not_shorter")
    assert len(events) == 1 and events[0]["attempted"] is True and events[0]["observed_route"] == store.UNKNOWN_STAMP
    assert events[0]["source_sha256"] == source_sha256 and events[0]["blocks"] == c.ERA_COMPRESS_COUNT
    assert events[0]["era_chars"] > events[0]["source_chars"]


def test_the_same_run_on_the_same_route_is_not_paid_again_until_either_changes(tmp_path, fit, monkeypatch):
    old = _summary_blocks(c.MAX_SUMMARY_BLOCKS)
    chat, blocks_path, meta_path = _seed_run(tmp_path, old)
    seen = _fake_era(monkeypatch, shorter=False)
    assert c._run_block_consolidation(chat, blocks_path, meta_path, _LLM(), "", force_tail=True)
    _write_chat(chat, count=2 * c.BLOCK_SIZE, text_size=2)
    assert c._run_block_consolidation(chat, blocks_path, meta_path, _LLM(), "", force_tail=True)
    assert len(seen) == 1  # the second run made no era call
    events = _events(tmp_path, "era_not_shorter")
    assert [event["attempted"] for event in events] == [True, False]
    assert events[1]["source_sha256"] == events[0]["source_sha256"]

    monkeypatch.setattr(c, "_consolidation_route", lambda: ("other/model", False))
    _write_chat(chat, count=3 * c.BLOCK_SIZE, text_size=2)
    assert c._run_block_consolidation(chat, blocks_path, meta_path, _LLM(), "", force_tail=True)
    assert len(seen) == 2  # a new route earns a new attempt
    (record,) = c._era_retry_runs(json.loads(meta_path.read_text(encoding="utf-8"))).values()
    assert record["route"] == {"model": "other/model", "use_local": False}


def test_a_shorter_era_replaces_the_run_and_clears_era_retry(tmp_path, fit, monkeypatch):
    old = _summary_blocks(c.MAX_SUMMARY_BLOCKS)
    chat, blocks_path, meta_path = _seed_run(tmp_path, old)
    meta_path.write_text(json.dumps({"era_retry": {"source_sha256": "stale", "route": "unknown"}}), encoding="utf-8")
    seen = _fake_era(monkeypatch, shorter=True)
    assert c._run_block_consolidation(chat, blocks_path, meta_path, _LLM(), "", force_tail=True)
    assert len(seen) == 1
    stored = json.loads(blocks_path.read_text(encoding="utf-8"))
    assert stored[0]["type"] == "era" and stored[1:c.MAX_SUMMARY_BLOCKS - c.ERA_COMPRESS_COUNT + 1] == old[c.ERA_COMPRESS_COUNT:]
    # Another run's (legacy-shaped) refusal survives this run's success; this run never had one.
    assert c._era_retry_runs(json.loads(meta_path.read_text(encoding="utf-8"))) == {"stale": {"route": "unknown"}}
    assert _events(tmp_path, "era_not_shorter") == []


def test_health_names_a_withheld_era_without_a_timestamp(tmp_path):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path,
                          repo_path=lambda p: tmp_path / p, drive_path=lambda p: tmp_path / p)
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {"last_consolidated_offset": 100})
    assert not any("ERA COMPRESSION" in line for line in context_health._memory_health_lines(env))
    c.atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {
        "era_retry": {"abcdef0123456789": {"route": {"model": "light/model", "use_local": False}},
                      "0123456789abcdef": {"route": "unknown"}}})
    rows = [line for line in context_health._memory_health_lines(env) if "ERA COMPRESSION WITHHELD" in line]
    assert len(rows) == 2 and "abcdef012345" in rows[0] and "light/model" in rows[0] and "2026-" not in rows[0]
    assert "0123456789ab" in rows[1]


# --- every scratchpad pass names its outcome ------------------------------------------


class _Scratch:
    def __init__(self, content):
        self.content = content

    def chat(self, **_kwargs):
        return {"content": self.content}, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.02,
                                           "provider": "openrouter", "resolved_model": "light/served"}


def _scratchpad(tmp_path, count=4):
    memory = Memory(tmp_path)
    for index in range(count):
        memory.append_scratchpad_block(f"block-{index}-" + (chr(97 + index) * 8_000), source=f"source-{index}")
    return memory


def test_a_scratchpad_replacement_reports_its_counts_and_source(tmp_path):
    memory = _scratchpad(tmp_path)
    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge",
                             _Scratch(json.dumps({"knowledge_entries": [], "compressed_block": "compressed"})))
    events = _events(tmp_path, "scratchpad_consolidation")
    assert len(events) == 1
    event = events[0]
    assert event["outcome"] == "replaced" and event["pressure"] is False
    assert (event["blocks_before"], event["compressed_blocks"], event["blocks_after"]) == (4, 2, 3)
    assert event["chars_before"] > event["chars_after"] > 0
    assert event["source_entry_id"] == memory.load_scratchpad_blocks()[0]["metadata"]["source_ref"]["entry_id"]
    assert event["knowledge_writes"] == {"ok": 0, "failed": 0}
    assert event["last_error_kind"] is None and event["accounted_upper_bound_usd"] == 0.02


@pytest.mark.parametrize("content, outcome", [
    ("not the requested JSON", "failed"),
    (json.dumps({"knowledge_entries": [], "compressed_block": "   "}), "empty_block"),
])
def test_a_refused_scratchpad_pass_names_why_and_keeps_the_blocks(tmp_path, content, outcome):
    memory = _scratchpad(tmp_path)
    before = memory.load_scratchpad_blocks()
    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", _Scratch(content))
    assert memory.load_scratchpad_blocks() == before
    events = _events(tmp_path, "scratchpad_consolidation")
    assert len(events) == 1 and events[0]["outcome"] == outcome
    assert events[0]["blocks_after"] == 4 and events[0]["source_entry_id"] == ""
    assert events[0]["last_error_kind"] == ("scratchpad_consolidation_failed" if outcome == "failed" else None)


def test_no_scratchpad_pass_means_no_event(tmp_path):
    memory = Memory(tmp_path)
    memory.append_scratchpad_block("small", source="task")
    assert c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", _Scratch("unused")) is None
    assert _events(tmp_path, "scratchpad_consolidation") == []


# --- a declined reflection lesson is a fact --------------------------------------------


def test_project_scoped_reflection_skips_are_typed_events(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    applied = reflection.apply_memory_actions(env, [
        {"type": "scratchpad_append", "content": "a lesson", "task_id": "t1"},
        {"type": "identity_update_candidate", "content": "a trait", "task_id": "t1"},
        {"type": "knowledge_write", "content": "topic-less", "task_id": "t1"},
    ], project_id="proj_x")
    assert applied == 0
    events = _events(tmp_path, "reflection_memory_action_skipped")
    assert [(e["action_type"], e["reason"]) for e in events] == [
        ("scratchpad_append", "project_scoped_task"), ("identity_update_candidate", "project_scoped_task"),
        ("knowledge_write", "missing_topic")]
    assert all(e["project_id"] == "proj_x" and e["task_id"] == "t1" and e["content_chars"] > 0 for e in events)
    assert events[0]["input_ref"] == {"status": "source_unavailable", "project_id": "proj_x"}
    assert not (tmp_path / "memory" / "scratchpad_blocks.json").exists()


def _reflect(tmp_path, llm, task_id="t-skip", project_id="proj_x"):
    return reflection.generate_reflection(
        {"id": task_id, "text": "the episode", "drive_root": str(tmp_path), "project_id": project_id},
        {}, "trace", llm, {"rounds": 1, "cost": 0.1})


_REJECTED_RAW = [
    {"type": "knowledge_write", "topic": "people/alex", "content": "   "},
    {"type": "knowledge_write", "content": "topic-less"},
    {"type": "delete_everything", "content": "nope"},
    {"type": "scratchpad_append", "content": "a kept lesson"},
]


def test_rejected_raw_reflection_actions_are_typed_events_on_the_real_path(tmp_path, fit):
    # Round 2 (critical 3): production drops empty and topic-less actions in the
    # validator, before apply_memory_actions ever runs; the event must fire there.
    from tests.test_knowledge_consolidation import MemoryLLM

    entry = _reflect(tmp_path, MemoryLLM("Reflection.\nMEMORY_ACTIONS_JSON: " + json.dumps(_REJECTED_RAW)))
    assert entry["reflection"] == "Reflection." and [a["type"] for a in entry["memory_actions"]] == ["scratchpad_append"]
    events = _events(tmp_path, "reflection_memory_action_skipped")
    assert [(e["action_type"], e["reason"], e["content_chars"]) for e in events] == [
        ("knowledge_write", "empty_content", 3), ("knowledge_write", "missing_topic", len("topic-less")),
        ("delete_everything", "unsupported_type", len("nope"))]
    assert all((e["task_id"], e["project_id"]) == ("t-skip", "proj_x") for e in events)
    # The retained exact task input the model answered is what the validator seam
    # kept; the rejected reflection text itself is not retained.
    assert entry["source_ref"]["kind"] == "task_source"
    assert all(e["input_ref"] == entry["source_ref"] for e in events)


def test_a_failed_validator_skip_event_cannot_abort_the_reflection(tmp_path, fit, monkeypatch, caplog):
    from tests.test_knowledge_consolidation import MemoryLLM

    original = reflection.append_jsonl

    def broken_event(path, row, **kwargs):
        if row.get("type") == "reflection_memory_action_skipped":
            raise OSError("event store unavailable")
        return original(path, row, **kwargs)

    monkeypatch.setattr(reflection, "append_jsonl", broken_event)
    entry = _reflect(tmp_path, MemoryLLM("Reflection.\nMEMORY_ACTIONS_JSON: " + json.dumps(_REJECTED_RAW)))
    assert entry["reflection"] == "Reflection."  # not "(reflection generation failed ...)"
    assert [a["content"] for a in entry["memory_actions"]] == ["a kept lesson"]
    assert caplog.text.count("Reflection memory skip event could not be written") == 3
    assert _events(tmp_path, "reflection_memory_action_skipped") == []


def test_failed_skip_event_cannot_discard_later_reflection_lessons(tmp_path, monkeypatch, caplog):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    original = reflection.append_jsonl

    def broken_event(path, row, **kwargs):
        if row.get("type") == "reflection_memory_action_skipped":
            raise OSError("event store unavailable")
        return original(path, row, **kwargs)

    monkeypatch.setattr(reflection, "append_jsonl", broken_event)
    assert reflection.apply_memory_actions(env, [
        {"type": "knowledge_write", "content": "topic-less", "task_id": "t1"},
        {"type": "scratchpad_append", "content": "a real lesson", "task_id": "t1"},
    ], project_id="") == 1
    assert "Reflection memory skip event could not be written" in caplog.text
    assert "a real lesson" in Memory(tmp_path).load_scratchpad()


def test_applied_reflection_actions_emit_no_skip(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    assert reflection.apply_memory_actions(env, [
        {"type": "scratchpad_append", "content": "a lesson", "task_id": "t1"},
        {"type": "identity_update_candidate", "content": "a trait", "task_id": "t1"},
        {"type": "scratchpad_append", "content": "   ", "task_id": "t1"},
    ]) == 2
    events = _events(tmp_path, "reflection_memory_action_skipped")
    assert [(e["action_type"], e["reason"], e["project_id"]) for e in events] == [
        ("scratchpad_append", "empty_content", "")]


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


class _ThreeRoomNominating:
    """One block of three rooms; each room's correction answers on its own route.

    Review N2 / round 2 (advisory 6): provenance is per nomination. Room A's
    correction has no physical route fact (unknown), rooms B and C answer on two
    different accounts, and every correction also tries to forge the host stamp.
    """
    topics = {"A": "people/alex", "B": "people/bob", "C": "people/cara"}
    accounts = {"A": None, "B": "acct-1", "C": "acct-2"}

    def __init__(self):
        self.corrections = []

    @staticmethod
    def _served(account):
        return {"provider": "claudexor", "resolved_model": "light/served",
                "claudexor": {"route": {"source": "codex", "credentialProfileId": account}}}

    def chat(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if prompt.startswith("Compare this draft memory"):
            room = next(name for name in "ABC" if f"## Draft memory\nEpisode {name}." in prompt)
            self.corrections.append(room)
            account = self.accounts[room]
            usage = {"cost": 0.01, **({} if account is None else self._served(account))}
            return {"content": f"Episode {room}, checked.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps([
                {"topic": self.topics[room], "content": f"Understanding {room}.", "_nomination_route": _FORGED}])}, usage
        room = "A" if "entry-0 " in prompt else "B" if "entry-34 " in prompt else "C"
        return {"content": f"Episode {room}.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps([
            {"topic": self.topics[room], "content": f"Understanding {room}."}])}, {"cost": 0.01, **self._served("acct-draft")}


def test_dialogue_consolidation_stamps_each_nomination_with_its_own_correction_route(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    chat.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"ts": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00Z", "direction": "in",
             "text": f"entry-{index} ", "chat_id": 1 if index < 34 else 2 if index < 67 else 3} for index in range(100)]
    chat.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="consolidate")
    llm = _ThreeRoomNominating()
    usage = c.consolidate(chat, blocks, meta, llm, knowledge_context=ctx)
    assert llm.corrections == ["A", "B", "C"] and usage["_blocks_written"] == 1
    # The block-wide stamp is the LAST call's known route (room C's correction) ...
    assert c._route_stamp(usage)["account"] == "acct-2"
    captures = {row["topic"]: row for row in _history(tmp_path) if row.get("publication") == "source_capture"}
    assert set(captures) == {"people/alex", "people/bob", "people/cara"}
    assert all(row["writer"] == "consolidation" for row in captures.values())
    # ... yet each nomination carries the route of the correction that released IT:
    # an explicit unknown outranks the known block stamp, two known routes stay
    # distinct within one block, and the forged model stamp reached none of them.
    assert captures["people/alex"]["route"] == store.UNKNOWN_STAMP
    assert captures["people/bob"]["route"] == {
        "provider": "claudexor", "model": "light/served", "source": "codex", "account": "acct-1"}
    assert captures["people/cara"]["route"] == {
        "provider": "claudexor", "model": "light/served", "source": "codex", "account": "acct-2"}
    block = json.loads(blocks.read_text(encoding="utf-8"))[0]
    assert all(row["writer_input_ref"] == block["knowledge_source_ref"] for row in captures.values())
    assert block["knowledge_source_ref"]["entry_id"] and len(block["knowledge_writes"]) == 3
    assert "_nomination_route" not in block  # a history stamp, never a persisted block field
    # The retained nominations row keeps the HOST stamp per entry, never the model's.
    nominations = _history(tmp_path)[0]
    assert nominations["type"] == "dialogue_knowledge_nominations"
    assert [(entry["topic"], entry["_nomination_route"]) for entry in nominations["nominations"][0]["entries"]] == [
        ("people/alex", store.UNKNOWN_STAMP), ("people/bob", captures["people/bob"]["route"]),
        ("people/cara", captures["people/cara"]["route"])]


def test_scratchpad_consolidation_stamps_its_journal_source(tmp_path):
    memory = _scratchpad(tmp_path)
    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", _Scratch(json.dumps({
        "knowledge_entries": [{"topic": "lessons/one", "content": "A durable lesson."}],
        "compressed_block": "compressed"})))
    capture = next(row for row in _history(tmp_path) if row.get("publication") == "source_capture")
    assert capture["writer"] == "scratchpad_consolidation"
    assert capture["route"] == {"provider": "openrouter", "model": "light/served"}
    assert capture["writer_input_ref"] == memory.load_scratchpad_blocks()[0]["metadata"]["source_ref"]
    assert _events(tmp_path, "scratchpad_consolidation")[0]["knowledge_writes"] == {"ok": 1, "failed": 0}


_FORGED = {"provider": "forged", "model": "forged/model", "account": "forged-acct"}


def test_a_model_supplied_nomination_route_never_reaches_history_from_scratchpad(tmp_path):
    # Round 2 (critical 1): the scratchpad producer binds every key the model wrote
    # before the writer ran; a forged host stamp must be dropped at that binding.
    memory = _scratchpad(tmp_path)
    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", _Scratch(json.dumps({
        "knowledge_entries": [{"topic": "lessons/one", "content": "A durable lesson.", "_nomination_route": _FORGED,
                               "_expected_revision": "forged", "_task_id": "forged"}],
        "compressed_block": "compressed"})))
    capture = next(row for row in _history(tmp_path) if row.get("publication") == "source_capture")
    assert capture["topic"] == "lessons/one" and capture["writer"] == "scratchpad_consolidation"
    assert capture["route"] == {"provider": "openrouter", "model": "light/served"}  # the host's observed stamp
    assert _events(tmp_path, "scratchpad_consolidation")[0]["knowledge_writes"] == {"ok": 1, "failed": 0}
    journal = [json.loads(line) for line in memory.journal_path().read_text(encoding="utf-8").splitlines() if line.strip()]
    (bound,) = next(row for row in journal if row.get("type") == "blocks_consolidated")["knowledge_entries"]
    assert not [key for key in bound if key.startswith("_")]  # nothing model-written survives as a host key


def test_a_model_supplied_nomination_route_never_reaches_history_from_knowledge_maintenance(tmp_path, fit):
    from tests.test_memory_pressure_maintenance import setup_memory

    memory, ctx = setup_memory(tmp_path)
    assert store.write_knowledge_note(store.resolve_knowledge_address(tmp_path, "overview", "global"),
                                      "---\nsummary: Orientation.\n---\n" + "Detailed understanding. " * 200).ok

    class Forging:
        def chat(self, **_kwargs):
            return {"content": json.dumps({"knowledge_entries": [
                {"topic": "lessons/forged", "scope": "global", "content": "A shorter detail note.",
                 "_nomination_route": _FORGED}]})}, {
                "cost": 0.02, "provider": "claudexor", "resolved_model": "light/served",
                "claudexor": {"route": {"source": "codex", "credentialProfileId": "acct-real"}}}

    result = c.maintain_memory_pressure(memory, Forging(), ctx, fits=lambda: False)
    action = next(row for row in result["actions"] if row["owner"] == "knowledge_maintenance")
    assert [(row["topic"], row["scope"], row["ok"]) for row in action["writes"]] == [("lessons/forged", "global", True)]
    capture = next(row for row in _history(tmp_path) if row.get("publication") == "source_capture"
                   and row["topic"] == "lessons/forged")
    assert capture["writer"] == "knowledge_maintenance"
    assert capture["route"] == {"provider": "claudexor", "model": "light/served", "source": "codex", "account": "acct-real"}


def test_bind_entries_keeps_model_fields_and_drops_every_host_key(tmp_path):
    reads = c.KnowledgeReadContext(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="op-1"))
    (bound,) = reads.bind_entries([{"topic": "people/alex", "content": "Observed.", "scope": "global",
                                    "_nomination_route": _FORGED, "_anything": 1, "extra": "kept"}])
    assert bound == {"topic": "people/alex", "content": "Observed.", "scope": "global", "extra": "kept",
                     "expected_revision": None, "canonical_root": str(tmp_path), "task_id": "op-1"}


def test_project_reflection_action_uses_actor_readable_exact_source(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    env = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path, repo_dir=tmp_path)
    entry = {"task_id": "t3", "ts": "2026-01-01T00:00:00Z", "route": {"provider": "claudexor", "model": "claude-fable"},
             "memory_actions": [
                 {"type": "knowledge_write", "topic": "lessons/project", "content": "Grounded.", "task_id": "t3"}]}
    reflection.append_reflection_routed(env, {"id": "t3", "project_id": "proj_x",
                                              "budget_drive_root": str(tmp_path)}, entry)
    action = entry["memory_actions"][0]
    source = action["_reflection_source_ref"]
    assert source["kind"] == "task_source"
    assert json.loads(read_actor_source_bytes(tmp_path, "t3", source))["memory_actions"][0]["content"] == "Grounded."
    assert reflection.apply_memory_actions(env, entry["memory_actions"], project_id="proj_x") == 1
    history = tmp_path / "projects" / "proj_x" / "knowledge_history.jsonl"
    row = json.loads(history.read_text(encoding="utf-8").splitlines()[-1])
    assert row["writer_input_ref"]["sha256"] == source["sha256"]
    assert row["writer_input_ref"]["task_id"] == "t3"
    assert row["route"] == {"provider": "claudexor", "model": "claude-fable"}  # the reflection's answering route


def test_reflection_stamps_the_reflection_row_it_came_from(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    assert reflection.apply_memory_actions(env, [
        {"type": "knowledge_write", "topic": "lessons/two", "content": "Reusable fact.", "task_id": "t9"}]) == 1
    capture = next(row for row in _history(tmp_path) if row.get("publication") == "source_capture")
    assert capture["writer"] == "reflection" and capture["route"] == store.UNKNOWN_STAMP
    assert capture["writer_input_ref"]["task_id"] == "t9"
    assert capture["writer_input_ref"]["read"]["arguments"]["path"] == "logs/task_reflections.jsonl"


def test_the_reader_less_knowledge_journal_is_no_longer_written(tmp_path):
    assert store.write_knowledge_note(_address(tmp_path), "# Alex\n\nFirst.").ok
    assert not (tmp_path / "memory" / "knowledge_journal.jsonl").exists()
    assert (tmp_path / "memory" / "knowledge_history.jsonl").exists()
    assert "knowledge_journal" not in inspect.getsource(store)
