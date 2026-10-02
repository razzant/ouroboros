"""Real store/maintenance consumers: publication, correction, sources and pressure."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c, room_consolidation as rooms
from ouroboros.chronicle_store import ChronicleStore, source_row_id
from ouroboros.context_health import _memory_health_lines
from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address, write_knowledge_note
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

fit = fit_helpers.fit


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("represented_only", [False, True])
def test_busy_memory_preparation_reports_availability_without_claiming_irreducibility(
    tmp_path, monkeypatch, locked, represented_only,
):
    _store, ctx, chat, blocks, meta = setup(tmp_path)
    if locked:
        def busy(_fd):
            raise BlockingIOError("another publication owns the lock")
        monkeypatch.setattr(c, "_lock_nb", busy)
    result = c.consolidate(chat, blocks, meta, None, knowledge_context=ctx,
        represented_only=represented_only, compact_chronicle=True, pressure_fits=lambda: True)
    if locked and not represented_only:
        assert result is None  # Existing ordinary maintenance skip is unchanged.
    else:
        errors = result.get("_consolidation_errors", [])
        assert bool(errors) is locked
        if locked:
            assert errors[0]["reason"] == "consolidation_lock_held"
            assert errors[0]["kind"] == "temporarily_unavailable"
            assert result["_blocks_written"] == 0


def consolidate_closed(*args, **kwargs):
    """These fixtures explicitly model a completed post-task producer."""
    return c.consolidate(*args, completed_task={"id": "memory-writer"}, **kwargs)


def setup(root, rows=()):
    for row in rows:
        row.setdefault("task_id", "memory-writer")
    memory = root / "memory"
    memory.mkdir()
    chat = root / "logs/chat.jsonl"
    chat.parent.mkdir()
    chat.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    blocks, meta = memory / "dialogue_blocks.json", memory / "dialogue_meta.json"
    blocks.write_text("[]", encoding="utf-8")
    meta.write_text("{}", encoding="utf-8")
    store = ChronicleStore(root)
    store.import_legacy()
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="memory-writer")
    return store, ctx, chat, blocks, meta


class Helper:
    def __init__(self):
        self.calls = []
        self.correction_error = None
        self.before_correction = None

    def __call__(self, prompt, label, **kwargs):
        self.calls.append((prompt, label, kwargs))
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": .01}
        if label == "Episode correction":
            if self.before_correction:
                self.before_correction()
            if self.correction_error:
                return "", {**usage, "ledger_attempt_ids": ["physical-unknown"] if self.correction_error == "provider_outcome_unknown" else [],
                            "_consolidation_errors": [{"kind": self.correction_error}]}, None
            return "The owner asked a question, not granted publication.", usage, None
        if label == "Room digest":
            return "Room history retains the open question and its cause.", usage, None
        return "Ouroboros considered the owner's question.", usage, None


def test_room_tail_publishes_without_global_message_threshold_and_freezes_legacy(tmp_path, monkeypatch):
    raw = [{"ts": "2026-09-30T01:00:00", "chat_id": 1, "direction": "in", "text": "An open question"},
           {"ts": "2026-09-30T01:01:00", "chat_id": 22, "direction": "in", "text": "A different room"}]
    store, ctx, chat, blocks, meta = setup(tmp_path, raw)
    before = blocks.read_bytes(), meta.read_bytes()
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    assert c.should_consolidate(meta, chat)
    usage = consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    episodes = store.records(kinds=["episode"])
    assert len(episodes) == 2 and usage["_blocks_written"] == 2
    assert {r["room_id"] for r in episodes} == {"1", "22"}
    assert [r["metadata"]["source_row_ids"] for r in episodes] == [[source_row_id(raw[0])], [source_row_id(raw[1])]]
    assert all(r["source_refs"][0]["task_id"] == "memory-writer" for r in episodes)
    assert [r["metadata"]["source_span"] for r in episodes] == [
        {"start": row["ts"] + "+00:00", "end": row["ts"] + "+00:00", "incomplete": False} for row in raw]
    assert all(r["ts"] != r["metadata"]["source_span"]["start"] for r in episodes)
    assert (blocks.read_bytes(), meta.read_bytes()) == before
    assert store.scan_state()["last_consolidated_offset"] == 2


def test_original_exists_during_correction_and_failed_correction_never_retracts_it(tmp_path, monkeypatch):
    store, ctx, chat, blocks, meta = setup(tmp_path, [
        {"ts": "2026-09-30T01:00:00", "chat_id": 1, "direction": "in", "text": "Question"}])
    helper = Helper()
    helper.correction_error = "transport_error"
    def retained():
        assert len(store.records(kinds=["episode"])) == 1
    helper.before_correction = retained
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    usage = consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert usage["_consolidation_errors"][0]["kind"] == "transport_error"
    assert store.records(kinds=["episode"])[0]["text"] == "Ouroboros considered the owner's question."
    assert not store.records(kinds=["revision"])
    helper.correction_error = None
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert len(store.records(kinds=["episode"])) == 1
    assert len(store.records(kinds=["revision"])) == 1
    assert sum(label == "Room episode" for _p, label, _kw in helper.calls) == 1


def test_unknown_paid_correction_preserves_custody_and_does_not_retry_on_next_task(tmp_path, monkeypatch):
    store, ctx, chat, blocks, meta = setup(tmp_path, [
        {"ts": "2026-09-30T01:00:00", "chat_id": 1, "direction": "in", "text": "Question"}])
    helper = Helper()
    helper.correction_error = "provider_outcome_unknown"
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    first = consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    calls = len(helper.calls)
    helper.correction_error = None
    second = consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert len(helper.calls) == calls
    assert first["_consolidation_errors"][0]["kind"] == "provider_outcome_unknown"
    assert not second["_consolidation_errors"]  # No fresh call failed on the later task.
    assert store.scan_state()["last_consolidation_error"]["source_ref"]
    assert store.records(kinds=["episode"])[0]["text"]


def test_legacy_unbound_unknown_does_not_veto_new_room_memory(tmp_path, monkeypatch):
    store, ctx, chat, blocks, meta = setup(tmp_path, [
        {"ts": "2026-09-30T01:00:00", "chat_id": 22, "direction": "in", "text": "A new room."}])
    store.publish([], scan_state={"last_consolidation_error": {"kind": "provider_outcome_unknown", "message": "old unbound fact"}})
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert len(store.records(kinds=["episode"])) == len(store.records(kinds=["revision"])) == 1
    assert len(helper.calls) == 2
    assert store.scan_state()["last_consolidation_error"]["message"] == "old unbound fact"


def test_bound_unknown_raw_region_stays_raw_while_other_room_progresses(tmp_path, monkeypatch):
    raw = [{"ts": "2026-09-30T01:00:00", "chat_id": 1, "direction": "in", "text": "Unknown region."},
           {"ts": "2026-09-30T01:01:00", "chat_id": 22, "direction": "in", "text": "Independent room."}]
    store, ctx, chat, blocks, meta = setup(tmp_path, raw)
    helper = Helper()
    def one_unknown(prompt, label, **kwargs):
        if label == "Room episode" and '"chat_id": 1' in prompt:
            helper.calls.append((prompt, label, kwargs))
            return "", {"cost": None, "ledger_attempt_ids": ["physical-raw-unknown"],
                "_consolidation_errors": [{"kind": "provider_outcome_unknown"}]}, None
        return helper(prompt, label, **kwargs)
    monkeypatch.setattr(c, "_light_call", lambda *_args: one_unknown)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert [row["room_id"] for row in store.records(kinds=["episode"])] == ["22"]
    assert not store.scan_state().get("last_consolidated_offset")
    calls = len(helper.calls)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert len(helper.calls) == calls  # neither the unknown nor the finished room repeats
    assert not store.scan_state().get("last_consolidated_offset")
    ref = c.retain_memory_source(ctx, "mind-recovery", json.dumps([raw[0]]).encode())
    store.append_episode("1", "I recovered the original source myself.", [ref], {"kind": "mind"},
        metadata={"source_row_ids": [source_row_id(raw[0])]})
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert store.scan_state()["last_consolidated_offset"] == 2
    assert store.scan_state()["pending_consolidation_outcomes"]  # physical custody is independent


@pytest.mark.parametrize("blob_manifest", [False, True])
def test_mind_episode_is_corrected_with_source_and_not_reconstructed_again(tmp_path, monkeypatch, blob_manifest):
    raw = {"ts": "2026-09-30T01:00:00", "chat_id": 1, "direction": "in", "text": "Please explain before deciding."}
    store, ctx, chat, blocks, meta = setup(tmp_path, [raw])
    if blob_manifest:
        from ouroboros.artifacts import read_actor_source_bytes
        from ouroboros.chronicle_view import capture_chronicle
        from ouroboros.memory import Memory
        snapshot = json.loads(capture_chronicle(Memory(tmp_path), {"id": ctx.task_id, "chat_id": 1}))
        ref = snapshot["source_ref"]
        manifest = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, ref))
        assert manifest["source_chunks"] and "rows" not in manifest
        assert raw["text"] not in json.dumps(manifest)
        chat.unlink()  # The immutable retained source alone must suffice.
    else:
        ref = c.retain_memory_source(ctx, "original-dialogue", json.dumps([raw]).encode())
    original = store.append_episode("1", "The owner approved publication.", [ref], {"kind": "mind"},
        metadata={"source_row_ids": [source_row_id(raw)]})
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    visible = store.room_records("1")[0]
    assert visible["text"] == original["text"]
    assert visible["current_text"] == "The owner asked a question, not granted publication."
    assert visible["current_author"]["kind"] == "helper"
    assert len(store.records(kinds=["episode"])) == 1
    assert len(helper.calls) == 1 and raw["text"] in helper.calls[0][0]
    if blob_manifest:
        assert '"source_chunks"' not in helper.calls[0][0]
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert len(helper.calls) == 1


def test_correction_can_nominate_knowledge_when_author_nominated_none(tmp_path, monkeypatch):
    store, ctx, chat, blocks, meta = setup(tmp_path)
    ref = c.retain_memory_source(ctx, "source", b"The owner distinguished questions from approval.")
    original = store.append_episode("1", "I asked the owner a question.", [ref], {"kind": "mind"})
    knowledge = c.KnowledgeReadContext(ctx)
    def helper(prompt, *_args, **_kwargs):
        assert "even when the author's episode nominated none" in prompt
        return ('Faithful interpretation.\nKNOWLEDGE_ENTRIES_JSON: '
                '[{"topic":"question-provenance","scope":"global","content":"A question is not an approval."}]'), {
                    "cost": 0, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, knowledge
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    note = read_knowledge_note(resolve_knowledge_address(tmp_path, "question-provenance", "global"))
    assert note.text.endswith("A question is not an approval.")
    assert store.room_records("1")[0]["correction"]["target_id"] == original["id"]
    assert not store.scan_state().get("pending_knowledge_nominations")


def test_nomination_history_failure_keeps_the_correction_full_proposal_and_open_debt(tmp_path, monkeypatch):
    from ouroboros import utils
    store, ctx, chat, blocks, meta = setup(tmp_path)
    ref = c.retain_memory_source(ctx, "source", b"The owner asked for clear writing.")
    store.append_episode("1", "I learned something about clear writing.", [ref], {"kind": "mind"})
    knowledge = c.KnowledgeReadContext(ctx)
    def helper(*_args, **_kwargs):
        return ('Interpretation.\nKNOWLEDGE_ENTRIES_JSON: '
                '[{"topic":"clear-writing","scope":"global","content":"Explain the concrete situation."}]'), {
                    "cost": 0, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, knowledge
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    append = utils.append_jsonl
    monkeypatch.setattr(utils, "append_jsonl", lambda path, *args, **kwargs:
        False if path.name == "knowledge_history.jsonl" else append(path, *args, **kwargs))
    with pytest.raises(OSError, match="nominations could not be retained"):
        consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    revision = store.records(kinds=["revision"])[0]
    assert revision["metadata"]["knowledge_entries"][0]["content"] == "Explain the concrete situation."
    assert store.scan_state()["pending_knowledge_nominations"][0]["topic"] == "clear-writing"
    assert store.room_records("1")[0]["current_text"] == "Interpretation."


def test_active_cursor_follows_rotation_without_rewriting_legacy_cursor(tmp_path, monkeypatch):
    raw = {"ts": "2026-09-30T01:00:00", "chat_id": 1, "direction": "in", "text": "Before rotation."}
    store, ctx, chat, blocks, meta = setup(tmp_path, [raw])
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    archive = tmp_path / "archive"
    archive.mkdir()
    chat.replace(archive / "chat_20260930T010001.jsonl")
    later = {**raw, "ts": "2026-09-30T02:00:00", "text": "After rotation."}
    chat.write_text(json.dumps(later) + "\n", encoding="utf-8")
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert len(store.records(kinds=["episode"])) == 2
    assert store.scan_state()["last_consolidated_offset"] == 1
    assert store.scan_state()["chat_log_signature"] == c._chat_log_signature(chat)
    assert meta.read_text(encoding="utf-8") == "{}"


def test_pressure_closes_new_tails_and_joins_stable_peers_without_self_paraphrase(tmp_path, monkeypatch):
    store, ctx, _chat, _blocks, _meta = setup(tmp_path)
    helper = Helper()
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    episodes = []
    for index in range(8):
        episode = store.append_episode("1", f"Open question {index}, owner has not approved. " * 200,
            [{"kind": "unavailable", "reason": f"original {index} missing"}], {"kind": "mind"},
            metadata={"source_row_ids": [str(index)], "task_ids": [f"task-{index}"], "source_gap": "Exact original unavailable."})
        episodes.append(episode)
        before = len(helper.calls)
        rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False)
        # The first call only consumes the new tail, never an old room digest.
        assert episode["text"] in helper.calls[before][0]
        assert '"kind": "helper"' not in helper.calls[before][0]
        assert "Exact original unavailable." in helper.calls[before][0]
        for prior in episodes[:-1]:
            assert prior["text"] not in helper.calls[before][0]
        calls = len(helper.calls)
        assert rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False) == []
        assert len(helper.calls) == calls
    digests = store.records(kinds=["digest"])
    assert any(all(store.get(cid)["kind"] == "digest" for cid in d["metadata"]["covers_record_ids"])
               for d in digests)  # genuine higher parents remain possible
    for digest in digests:
        children = [store.get(cid) for cid in digest["metadata"]["covers_record_ids"]]
        assert len(children) > 1 or children[0]["kind"] != "digest"
        assert digest["metadata"]["source_gap"]
        assert set(digest["metadata"]["source_span"]) == {"start", "end", "incomplete"}
        source_ref = digest["source_refs"][0]
        source = (tmp_path / source_ref["read"]["arguments"]["path"]).read_text(encoding="utf-8")
        assert all(child["text"] in source for child in children)
    cover = store.room_cover("1")
    assert len(cover) == 1 and set(cover[0]["metadata"]["source_row_ids"]) == {str(n) for n in range(8)}
    assert len(store.records(kinds=["episode"])) == 8


@pytest.mark.parametrize("purpose,boundary,tighter", [
    ("actual_context_refusal", None, 500000), ("owner_mode", 250000, 85000)])
def test_real_requirement_rebuilds_fixed_children_but_leftover_jitter_does_not(
        tmp_path, monkeypatch, purpose, boundary, tighter):
    store, ctx, _chat, _blocks, _meta = setup(tmp_path)
    original = store.append_episode("1", "Original source with the open obligation. " * 100, [], {"kind": "mind"})
    helper = Helper()
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    demand = {"memory_budget_tokens": 1000, "route_fingerprint": "main-route", "rendered_mode": "low",
              "purpose": purpose, "requirement_tokens": boundary}
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    first = store.records(kinds=["digest"])[0]
    assert len(helper.calls) == 1
    demand.update(rendered_memory_tokens=9999, task_id="later-task", unrelated_observation="changed")
    assert not rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 1  # Observation churn is not a new paid demand.
    for budget in (991, 982, 1037, 1210):
        demand["memory_budget_tokens"] = budget
        assert not rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 1  # A looser allowance cannot buy the same digest again.
    demand.update(memory_budget_tokens=500, requirement_tokens=tighter)
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 2 and original["text"] in helper.calls[-1][0]
    assert first["text"] not in helper.calls[-1][0]
    assert "500" in helper.calls[-1][0]
    assert all(d["metadata"]["covers_record_ids"] == [original["id"]] for d in store.records(kinds=["digest"]))
    assert not rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    demand.update(memory_budget_tokens=491, requirement_tokens=tighter + 9)
    assert not rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 2  # The tightest prior attempt remains authoritative.


def test_account_change_does_not_rebuy_a_completed_digest(tmp_path, monkeypatch):
    store, ctx, _chat, _blocks, _meta = setup(tmp_path)
    original = store.append_episode("1", "The owner's decision and its source reason. " * 100, [], {"kind": "mind"})
    helper = Helper()
    route = {"model": "same-model", "use_local": False, "model_account_override": "first"}
    monkeypatch.setattr(c, "_light_route", lambda: dict(route))
    demand = {"memory_budget_tokens": 1000, "route_fingerprint": "main-one", "rendered_mode": "max",
              "purpose": "actual_context_refusal", "requirement_tokens": 1000000}
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 1
    route["model_account_override"] = "healthy-other-account"
    demand["route_fingerprint"] = "main-another-account"
    for budget in (1000, 991, 1200):
        demand["memory_budget_tokens"] = budget
        assert not rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 1
    demand.update(memory_budget_tokens=500, requirement_tokens=500000)
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 2 and original["text"] in helper.calls[-1][0]



@pytest.mark.parametrize("change", ["guidance", "tighter_boundary"])
def test_changed_guidance_or_tighter_boundary_can_revise_fixed_sources_once(tmp_path, monkeypatch, change):
    store, ctx, *_ = setup(tmp_path)
    original = store.append_episode("1", "The owner left this unresolved. " * 100, [], {"kind": "mind"})
    helper = Helper()
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    demand = {"purpose": "owner_mode", "rendered_mode": "low", "memory_budget_tokens": 900,
              "requirement_tokens": 250000}
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    if change == "guidance":
        write_knowledge_note(resolve_knowledge_address(tmp_path, "remembering", "global"),
                             "Retain the unresolved owner's choice and distinguish my assumptions.")
    else:
        demand.update(rendered_mode="nano", requirement_tokens=85000)
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 2 and original["text"] in helper.calls[-1][0]
    assert all(d["metadata"]["covers_record_ids"] == [original["id"]] for d in store.records(kinds=["digest"]))
    demand.update(memory_budget_tokens=891, failed_candidate_sha256="different-request", round_id="another-round")
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand) == []
    assert len(helper.calls) == 2


def test_old_soft_budget_digest_remains_usable_until_first_real_refusal(tmp_path, monkeypatch):
    store, ctx, *_ = setup(tmp_path)
    original = store.append_episode("1", "Old meaning still carries its source. " * 100, [], {"kind": "mind"})
    keys = [[original["id"], original["id"]]]
    store.publish([
        {"id": "digest-attempt:old", "kind": "maintenance", "room_id": "1", "status": "compressed",
         "source_keys": keys, "fitting_demand": {"memory_budget_tokens": 1000, "rendered_mode": "max"}},
        {"id": "digest:old", "kind": "digest", "room_id": "1", "text": "Old published meaning.",
         "author": {"kind": "helper"}, "metadata": {"covers_record_ids": [original["id"]], "source_revisions": keys}}])
    helper = Helper()
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "other-account-model"})
    demand = {"memory_budget_tokens": 991, "rendered_mode": "max", "route_fingerprint": "another-account"}
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand) == []
    assert helper.calls == []
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False) == []
    demand.update(purpose="actual_context_refusal", requirement_tokens=None, refused_digest_ids=["digest:old"])
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(helper.calls) == 1 and original["text"] in helper.calls[0][0]
    demand.update(memory_budget_tokens=982, failed_candidate_sha256="next-refused-body")
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand) == []
    assert len(helper.calls) == 1 and store.get("digest:old")["text"] == "Old published meaning."
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False) == []


def test_digest_projection_preserves_meaning_and_retains_exact_host_metadata(tmp_path, monkeypatch):
    import hashlib
    store, ctx, _chat, _blocks, _meta = setup(tmp_path)
    ref = {**c.retain_memory_source(ctx, "original", b"Exact original source."),
           "availability": "partial", "reason": "Original owner reply is missing."}
    metadata = {"children": [f"host-child-{n}" for n in range(20)], "source_row_ids": ["row-one", "row-two"],
                "legacy_source_ref": ref, "range": "earlier to later", "coverage": "partial",
                "source_gap": "Do not infer approval from the missing reply.", "status": "owner_cancelled",
                "new_semantic_field": {"words": "Future metadata must remain visible."}}
    original = store.append_episode("1", "Original authored account. " * 40, [ref], {"kind": "mind"}, metadata=metadata)
    revision = store.revise(original["id"], "Corrected full account, publication was cancelled. " * 40,
                            {"kind": "helper", "model": "corrector"})
    rows = store.room_records("1")
    expected = "\n\n".join(json.dumps({"record_id": r["id"], "revision_id": (r.get("correction") or r)["id"],
        "author": r["current_author"], "text": r["current_text"], "source_refs": r.get("source_refs", []),
        "metadata": r.get("metadata", {})}, ensure_ascii=False) for r in rows).encode("utf-8")
    helper = Helper()
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False)
    prompt, _label, options = helper.calls[0]
    projected = json.loads(prompt.split("## Source group: complete text, projected host metadata\n", 1)[1])
    assert projected["text"] == revision["text"] and projected["author"] == revision["author"]
    assert projected["record_id"] == original["id"] and projected["revision_id"] == revision["id"]
    assert projected["metadata"]["children"] == {"retained_items": 20}
    assert "host-child-0" not in prompt and ref["sha256"] not in prompt
    assert "Their omission is not a read receipt" in prompt
    for key in ("range", "coverage", "source_gap", "status", "new_semantic_field"):
        assert projected["metadata"][key] == metadata[key]
    assert projected["source_refs"][0]["reason"] == ref["reason"]
    assert projected["metadata"]["legacy_source_ref"]["availability"] == "partial"
    digest = store.records(kinds=["digest"])[0]
    assert digest["metadata"]["source_span"] == {"start": None, "end": None, "incomplete": True}
    retained = digest["source_refs"][0]
    assert (tmp_path / retained["read"]["arguments"]["path"]).read_bytes() == expected
    assert retained["sha256"] == hashlib.sha256(expected).hexdigest()
    read = c.KnowledgeReadContext(ctx).read_call({"id": "inspect-full-group", "function": {
        "name": "read_file", "arguments": json.dumps(retained["read"]["arguments"])}})
    assert read["status"] == "ok" and "host-child-0" in read["result"] and ref["sha256"] in read["result"]
    assert options["memory_operation"]["source_revisions"] == [[original["id"], revision["id"]]]
    assert digest["metadata"]["covers_record_ids"] == [revision["id"]]
    assert {k: v for k, v in store.get(original["id"]).items() if k != "sequence"} == original
    assert store.scan_state() == {}


def test_digest_prompt_refreshes_whole_view_observations_after_publication(tmp_path, monkeypatch):
    store, ctx, _chat, _blocks, _meta = setup(tmp_path)
    for room in ("1", "2"):
        store.append_episode(room, "Complete source with an unresolved owner obligation. " * 100, [], {"kind": "mind"})
    helper, demand = Helper(), {"memory_budget_tokens": 1000, "route_fingerprint": "same", "rendered_mode": "max"}
    def fits():
        demand["rendered_memory_tokens"] = 4000 - 1000 * len(store.records(kinds=["digest"]))
        return False
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand)
    assert len(helper.calls) == 2
    for call, expected in zip(helper.calls, (4000, 3000)):
        line = next(line for line in call[0].splitlines() if line.startswith("Measured whole-memory"))
        facts = json.loads(line.split(": ", 1)[1])
        assert facts["rendered_memory_tokens"] == expected and facts["memory_deficit_tokens"] == expected - 1000
        assert facts["token_estimate_basis"] == "chars_div_4" and facts["source_text_chars"] > 0
        assert facts["source_text_estimated_tokens"] == (facts["source_text_chars"] + 3) // 4
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand) == []
    assert len(helper.calls) == 2


@pytest.mark.parametrize("failure", [None, "provider_outcome_unknown", "budget_exhausted"])
def test_represented_only_preparation_never_drafts_raw_or_corrects_episodes(tmp_path, monkeypatch, failure):
    store, ctx, chat, blocks, meta = setup(tmp_path, [{"ts": "2026-09-30T01:00:00", "chat_id": 1,
        "direction": "in", "text": "RAW HISTORY MUST NOT BE DRAFTED"}])
    small = store.append_episode("22", "Smaller room. " * 100, [], {"kind": "helper"})
    original = store.append_episode("1", "Already represented memory with source attribution. " * 100,
        [], {"kind": "mind"})
    helper = Helper()
    def transport(prompt, label, **options):
        assert label == "Room digest" and "RAW HISTORY MUST NOT BE DRAFTED" not in prompt
        if failure:
            helper.calls.append((prompt, label, options))
            return "", {"cost": None, "ledger_attempt_ids": ["attempt-bootstrap"],
                "_consolidation_errors": [{"kind": failure}]}, None
        return helper(prompt, label, **options)
    monkeypatch.setattr(c, "_light_call", lambda *_args: transport)
    before = chat.read_bytes(), blocks.read_bytes(), meta.read_bytes()
    usage = c.consolidate(chat, blocks, meta, None, knowledge_context=ctx,
        represented_only=True, compact_chronicle=True, pressure_fits=lambda: bool(store.records(kinds=["digest"])))
    assert len(helper.calls) == 1 and not store.records(kinds=["revision"])
    assert len(store.records(kinds=["episode"])) == 2 and store.get(original["id"])["text"] == original["text"]
    assert original["text"] in helper.calls[0][0] and small["text"] not in helper.calls[0][0]
    assert (chat.read_bytes(), blocks.read_bytes(), meta.read_bytes()) == before
    assert not store.scan_state().get("last_consolidated_offset")
    assert usage["_blocks_written"] == (0 if failure else 1)
    if failure == "provider_outcome_unknown":
        failure = None  # The next independent room can use a healthy transport.
        usage = c.consolidate(chat, blocks, meta, None, knowledge_context=ctx,
            represented_only=True, compact_chronicle=True, pressure_fits=lambda: False,
            fitting_demand={"memory_budget_tokens": 100})
        assert len(helper.calls) == 2 and not usage["_consolidation_errors"]
        assert original["text"] not in helper.calls[-1][0] and small["text"] in helper.calls[-1][0]
        assert usage["_blocks_written"] == 1 and len(store.records(kinds=["digest"])) == 1
        assert store.scan_state()["pending_consolidation_outcomes"][0]["source_ref"]
        store.append_episode("1", "A newly closed tail must not re-buy the held original. " * 20, [], {"kind": "mind"})
        c.consolidate(chat, blocks, meta, None, knowledge_context=ctx,
            represented_only=True, compact_chronicle=True, pressure_fits=lambda: False,
            fitting_demand={"memory_budget_tokens": 100})
        assert len(helper.calls) == 2  # Regrouping cannot evade unknown-attempt custody.
    elif not failure:
        assert usage["_coverage"][0]["record_id"] == store.records(kinds=["digest"])[0]["id"]


def test_pressure_never_folds_stale_digest_over_a_new_correction(tmp_path, monkeypatch):
    store, ctx, _chat, _blocks, _meta = setup(tmp_path)
    a = store.append_episode("1", "The owner approved an irreversible publication. " * 50, [], {"kind": "mind"})
    store.append_episode("1", "STALE approval digest.", [], {"kind": "helper"},
        kind="digest", metadata={"covers_record_ids": [a["id"]]})
    revision = store.revise(a["id"], "It was only a question; publication remains undecided. " * 50, {"kind": "mind"})
    helper = Helper()
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False)
    assert "STALE approval digest" not in helper.calls[0][0]
    assert "publication remains undecided" in helper.calls[0][0]
    assert store.records(kinds=["digest"])[-1]["metadata"]["covers_record_ids"] == [revision["id"]]


def test_health_reads_active_nomination_debt_not_frozen_legacy_meta(tmp_path):
    store, _ctx, _chat, _blocks, meta = setup(tmp_path)
    store.publish([], scan_state={"pending_knowledge_nominations": [{"id": "source:0:0", "topic": "pending"}]})
    env = SimpleNamespace(drive_path=lambda p: tmp_path / p)
    text = "\n".join(_memory_health_lines(env))
    assert "1 source-addressed nominations" in text and "memory/chronicle/records.jsonl" in text
    assert meta.read_text(encoding="utf-8") == "{}"


def test_guidance_revision_reaches_next_real_memory_request(tmp_path, fit):
    # Reuse the existing isolated route/fit fixture rather than a network model.
    from ouroboros.memory_guidance import remembering_guidance
    address = resolve_knowledge_address(tmp_path, "remembering", "global")
    first = write_knowledge_note(address, "Keep questions distinct from approval.").current
    assert first.text in remembering_guidance(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="guided-memory")
    model = fit_helpers._LLM()
    c._call_consolidation_llm(model, "A source episode.", "guided", knowledge=c.KnowledgeReadContext(ctx))
    second = write_knowledge_note(address, "Also retain who corrected the interpretation.", expected_revision=first.revision).current
    assert second.text in remembering_guidance(tmp_path)
    assert first.revision not in remembering_guidance(tmp_path)
    c._call_consolidation_llm(model, "Another source episode.", "guided", knowledge=c.KnowledgeReadContext(ctx))
    assert first.text in model.calls[0]["messages"][0]["content"]
    assert second.text in model.calls[1]["messages"][0]["content"]
    assert first.text not in model.calls[1]["messages"][0]["content"]


def test_large_guidance_keeps_the_existing_complete_source_reading_path(tmp_path, fit):
    from tests.test_memory_pressure_maintenance import SourceReader
    fit.window = 50000
    address = resolve_knowledge_address(tmp_path, "remembering", "global")
    note = write_knowledge_note(address, "Preserve who changed their understanding. " * 16000 + "GUIDANCE FINAL CONDITION.").current
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="large-guidance")
    original = c.retain_memory_source(ctx, "original-episode", b"A small source episode.")
    actor = SourceReader(tmp_path, fit.window, "Faithful result.")
    text, usage = c._call_consolidation_llm(actor, "A small source episode.", "guided",
        knowledge=c.KnowledgeReadContext(ctx), source_ref=original)
    assert text == "Faithful result." and not usage.get("_consolidation_errors")
    assert actor.received and note.text in actor.received[0]
    assert "GUIDANCE FINAL CONDITION." in actor.received[0]


@pytest.mark.parametrize("pressure", [False, True])
@pytest.mark.parametrize("owner,rendered,legacy", [
    ("max", "max", True), ("max", "low", True), ("max", "max", False),
    ("low", "low", True), ("nano", "nano", True)])
def test_post_task_pressure_keeps_owner_modes_with_legacy_and_new_sources(
        tmp_path, monkeypatch, pressure, owner, rendered, legacy):
    from ouroboros import chronicle_view, post_task_synthesis
    from ouroboros.memory import Memory

    store, _ctx, chat, _blocks, _meta = setup(tmp_path)
    store.append_episode("1", "A complete historical interpretation. " * 200, [], {"kind": "helper"})
    helper = Helper()
    def free_helper(*args, **kwargs):
        text, usage, knowledge = helper(*args, **kwargs)
        return text, {**usage, "cost": 0, "prompt_tokens": 0}, knowledge
    monkeypatch.setattr(c, "_light_call", lambda *_args: free_helper)
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    facts = {"basis": "post_task_evaluation", "memory_budget_tokens": 200,
             "rendered_memory_tokens": 2000, "route_fp": "post-route", "mode": rendered,
             "owner_context_mode": owner, "legacy_transition": legacy, "window_tokens": 1000000}
    monkeypatch.setattr(chronicle_view, "maintenance_projection", lambda *_args:
        (lambda: not pressure or bool(store.records(kinds=["digest"])), facts), raising=False)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda path: tmp_path / path)
    post_task_synthesis._run_chat_consolidation(env, Memory(tmp_path, tmp_path), None,
        {"id": "post-memory", "chat_id": 1}, chat.parent)
    purchased = pressure
    assert len(helper.calls) == int(purchased)
    assert len(store.records(kinds=["digest"])) == int(purchased)
    if purchased:
        attempt = next(row for row in store.records(kinds=["maintenance"]) if row.get("source_keys"))
        expected = ("owner_mode", 85000 if owner == "nano" else 250000) if owner != "max" else ("working_headroom", 1000000)
        assert (attempt["fitting_demand"]["purpose"], attempt["requirement"]["requirement_tokens"]) == expected


@pytest.mark.parametrize("missing", [False, True])
def test_authored_locator_source_corrects_only_complete_retrieved_claim(tmp_path, monkeypatch, missing):
    from ouroboros.chronicle_sources import capture_room, retain_room_source
    from ouroboros.memory import Memory
    from ouroboros.tools.chronicle import _chronicle_write, _memory_read

    raw = [{"chat_id": 1, "text": "First exact source."}, {"chat_id": 1, "text": "Second exact source."}]
    store, ctx, chat, blocks, meta = setup(tmp_path, raw)
    memory = Memory(tmp_path)
    captured, coverage = capture_room(memory, "1", rendered_chars_budget=1)
    ref = retain_room_source(memory, ctx.task_id, captured, coverage.pop("row_locators"), coverage)
    if missing:
        chat.unlink()
        selected = ref
    else:
        # Authored subset is legitimate, despite not representing the whole room.
        page = json.loads(_memory_read(ctx, source_ref=ref, start=1, end=2))
        assert not page["page_complete"] and page["range_complete"]
        selected = page["source_ref"]
        chat.unlink()  # Correction must read the exact retained page, not current chat.
    original = json.loads(_chronicle_write(ctx, room_id="1", text="My original account.", source_ref=selected))
    assert original["id"] and store.get(original["id"])["text"] == "My original account."
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx)
    assert len(helper.calls) == (0 if missing else 1)
    if missing:
        assert not store.records(kinds=["revision"])
        assert store.get("source-gap:" + original["id"])["status"] == "source_unavailable"
    else:
        assert "Second exact source." in helper.calls[0][0]
        assert "First exact source." not in helper.calls[0][0]
        assert store.records(kinds=["revision"])[0]["target_id"] == original["id"]


@pytest.mark.parametrize("corrected", [False, True])
def test_digest_keeps_bounded_source_span_and_unknowns_apart_from_publication(tmp_path, monkeypatch, corrected):
    from ouroboros.chronicle_store import source_time_span
    store, ctx, *_ = setup(tmp_path)
    # Normalize different offsets before comparing; unknown interior rows remain explicit.
    captured = ["2021-02-01T11:00:00+02:00", None, "2020-01-01T10:00:00Z", "invalid"]
    span = source_time_span(captured)
    assert span == {"start": "2020-01-01T10:00:00+00:00", "end": "2021-02-01T09:00:00+00:00", "incomplete": True}
    first = store.append_episode("1", "A long source-grounded account. " * 100, [], {"kind": "mind"},
                                 metadata={"source_span": span})
    if corrected:
        span = source_time_span(["2022-03-01T00:00:00Z", "2022-03-02T00:00:00Z"])
        store.revise(first["id"], "Corrected source-grounded account. " * 100, {"kind": "mind"},
                     metadata={"source_span": span})
    second = store.append_episode("1", "Another undated account. " * 100, [], {"kind": "mind"})
    helper = Helper()
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False)
    digest = store.records(kinds=["digest"])[0]
    assert digest["metadata"]["source_span"] == {**span, "incomplete": True}
    assert digest["ts"] != digest["metadata"]["source_span"]["start"]
    assert store.get(first["id"])["text"] == first["text"]
    assert second["text"] in helper.calls[0][0]
    if corrected:
        assert '"revision_source_span"' in helper.calls[0][0] and span["start"] in helper.calls[0][0]


@pytest.mark.parametrize("improves", [True, False])
def test_real_refusal_refines_exposed_digest_from_fixed_children_once(tmp_path, monkeypatch, improves):
    store, ctx, *_ = setup(tmp_path)
    original = store.append_episode("1", "Owner obligation and why it remains open. " * 100, [], {"kind": "mind"})
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    calls = []
    outputs = iter(["First account: the owner's unresolved obligation remains. " * 5,
                    "The obligation remains open." if improves else "Expanded account. " * 100,
                    "Obligation open."])
    def helper(prompt, *_args, **_kwargs):
        calls.append(prompt)
        return next(outputs), {}, None
    demand = {"purpose": "actual_context_refusal", "rendered_mode": "max"}
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    first = store.records(kinds=["digest"])[0]
    # Another request or profile refusing the same unaltered representation
    # is not a new semantic attempt; the representation itself is the need.
    demand.update(refused_digest_ids=[first["id"]], failed_candidate_sha256="physical-rejection")
    rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    assert len(calls) == 2 and original["text"] in calls[-1] and first["text"] not in calls[-1]
    assert len(store.records(kinds=["digest"])) == (2 if improves else 1)
    for jitter in ("another-account", "another-request"):
        demand.update(route_fingerprint=jitter, failed_candidate_sha256=jitter, memory_budget_tokens=991)
        assert rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand) == []
    attempts = [r for r in store.records(kinds=["maintenance"]) if r["id"].startswith("digest-fit:")]
    assert all(r["target_fits"] is None for r in attempts)
    assert attempts[-1]["published_progress"] is improves
    if improves:
        second = store.records(kinds=["digest"])[-1]
        demand["refused_digest_ids"] = [second["id"]]
        rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
        assert len(calls) == 3 and original["text"] in calls[-1] and second["text"] not in calls[-1]
        assert store.records(kinds=["digest"])[-1]["text"] == "Obligation open."


def test_ordinary_maintenance_advances_old_and_new_without_whole_corpus_sweep(tmp_path, monkeypatch):
    store, ctx, *_ = setup(tmp_path)
    store.publish([{"id": "legacy-history", "kind": "legacy", "room_id": "old",
                    "text": "Original old room meaning. " * 1000, "author": {"kind": "legacy"}}])
    store.append_episode("new", "Fresh completed source. " * 500, [], {"kind": "mind"})
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    helper = Helper()
    demand = {"purpose": "working_headroom", "requirement_tokens": 1000000,
              "ordinary_maintenance": True, "legacy_transition": True}
    def fits():
        demand["rendered_memory_tokens"] = sum(len(r["current_text"]) for room in store.room_ids()
                                               for r in store.room_cover(room)) // 4
        return demand["rendered_memory_tokens"] < 100
    rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand)
    assert [r["room_id"] for r in store.records(kinds=["digest"])] == ["old"]
    assert demand["published_progress"] and demand["target_fits"] is False
    rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand)
    assert [r["room_id"] for r in store.records(kinds=["digest"])] == ["old", "new"]
    assert demand["target_fits"] is True and len(helper.calls) == 2
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand) == []
    assert store.get("legacy-history")["text"] == "Original old room meaning. " * 1000


@pytest.mark.parametrize("saved_old_shape", [False, True])
@pytest.mark.parametrize("next_tokens", [250000, 500000])
def test_purpose_and_mode_labels_do_not_rebuy_fixed_sources(tmp_path, monkeypatch, saved_old_shape, next_tokens):
    store, ctx, *_ = setup(tmp_path)
    original = store.append_episode("1", "The owner left a decision unresolved. " * 100, [], {"kind": "mind"})
    keys = [[original["id"], original["id"]]]
    helper = Helper()
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    demand = {"purpose": "working_headroom", "rendered_mode": "max", "requirement_tokens": 250000}
    if saved_old_shape:
        store.publish([
            {"id": "digest-attempt:old-labels", "kind": "maintenance", "room_id": "1", "status": "compressed",
             "source_keys": keys, "requirement": dict(demand)},
            {"id": "digest:old-labels", "kind": "digest", "room_id": "1", "text": "Earlier whole account. " * 20,
             "author": {"kind": "helper"}, "metadata": {"covers_record_ids": [original["id"]], "source_revisions": keys}}])
    else:
        rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand)
    before_calls = len(helper.calls)
    before_records = store.records()
    demand.update(purpose="actual_context_refusal", rendered_mode="nano", requirement_tokens=next_tokens)
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", lambda: False, fitting_demand=demand) == []
    assert len(helper.calls) == before_calls
    assert store.records() == before_records  # Old authoritative records are never rewritten.


@pytest.mark.parametrize("failure,count", [("output_truncated", 4), ("output_truncated", 1), ("provider_outcome_unknown", 4)])
def test_terminal_output_cut_uses_source_halves_without_rebuy_or_unknown_overlap(tmp_path, monkeypatch, failure, count):
    store, ctx, chat, blocks, meta = setup(tmp_path)
    original = [store.append_episode("1", f"Original source {n}. " * 200, [], {"kind": "mind"},
                                    metadata={"source_row_ids": [str(n)]}) for n in range(count)]
    calls = []
    def helper(prompt, _label, **_kwargs):
        rows = [json.loads(row) for row in prompt.split("## Source group: complete text, projected host metadata\n", 1)[1].split("\n\n")]
        calls.append([row["record_id"] for row in rows])
        if len(calls) == 1:
            return "", {"ledger_attempt_ids": ["cut-or-unknown"], "_consolidation_errors": [{"kind": failure,
                         "preflight_only": False}]}, None
        return "The sources preserve an unresolved obligation.", {}, None
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    demand = {"ordinary_maintenance": True, "requirement_tokens": 250000}
    def fits():
        return all(r["kind"] == "digest" for r in store.room_cover("1"))
    def run():
        return c.consolidate(chat, blocks, meta, None, knowledge_context=ctx, represented_only=True,
                             compact_chronicle=True, pressure_fits=fits, fitting_demand=demand)
    first = run()
    assert first["_consolidation_errors"][0]["kind"] == failure
    for _ in range(3):
        run()
    assert calls.count([r["id"] for r in original]) == 1
    if failure == "output_truncated" and count > 1:
        assert calls == [[r["id"] for r in original], [r["id"] for r in original[:2]], [r["id"] for r in original[2:]]]
        assert fits() and {i for r in store.room_cover("1") for i in r["metadata"]["source_row_ids"]} == {str(n) for n in range(count)}
    else:
        assert len(calls) == 1 and not store.records(kinds=["digest"])
    assert all(store.get(r["id"])["text"] == r["text"] for r in original)
    failed = [r for r in store.records(kinds=["maintenance"]) if r.get("status") == "output_truncated"]
    if failure == "output_truncated":
        assert len(failed) == 1 and failed[0]["source_ref"]["read"]
    else:
        assert failed == [] and store.scan_state()["pending_consolidation_outcomes"]


def test_same_unfulfilled_goal_refines_fixed_children_until_fit_then_ignores_jitter(tmp_path, monkeypatch):
    store, ctx, *_ = setup(tmp_path)
    original = store.append_episode("1", "Immutable original meaning. " * 1000, [], {"kind": "mind"})
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    calls, answers = [], iter(["First useful account. " * 100, "Coarser account. " * 40, "Obligation open. " * 20])
    def helper(prompt, *_args, **_kwargs):
        calls.append(prompt)
        return next(answers), {}, None
    demand = {"ordinary_maintenance": True, "requirement_tokens": 500000, "memory_budget_tokens": 85}
    def fits():
        cover = store.room_cover("1")
        demand.update(selected_digest_ids=[r["id"] for r in cover if r["kind"] == "digest"],
                      rendered_memory_tokens=sum(len(r["current_text"]) for r in cover) // 4)
        demand["target_miss"] = demand["rendered_memory_tokens"] > demand["memory_budget_tokens"]
        return not demand["target_miss"]
    for expected in (1, 2, 3):
        rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand)
        assert len(calls) == expected and original["text"] in calls[-1]
    digests = store.records(kinds=["digest"])
    assert len(digests) == 3 and demand["target_fits"] is True
    assert all(d["text"] not in call for d, call in zip(digests, calls[1:]))
    demand.update(memory_budget_tokens=76, route_fingerprint="another-account")
    assert not fits()
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand) == []
    assert len(calls) == 3 and store.get(original["id"])["text"] == original["text"]


@pytest.mark.parametrize("settle_without_publication", [False, True])
def test_later_whole_view_success_settles_older_false_digest_need(tmp_path, monkeypatch, settle_without_publication):
    from ouroboros import chronicle_view, post_task_synthesis
    from ouroboros.memory import Memory
    store, ctx, chat, *_ = setup(tmp_path)
    store.append_episode("1", "Large old room. " * 1000, [], {"kind": "mind"})
    if not settle_without_publication:
        store.append_episode("2", "Second old room. " * 500, [], {"kind": "mind"})
    calls = []
    def helper(*_args, **_kwargs):
        calls.append(1)
        return ("A" * 1800 if len(calls) == 1 else "B" * 100), {}, None
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    demand = {"ordinary_maintenance": True, "requirement_tokens": 500000, "memory_budget_tokens": 100}
    def fits():
        cover = [r for room in store.room_ids() for r in store.room_cover(room)]
        demand.update(selected_digest_ids=[r["id"] for r in cover if r["kind"] == "digest"],
                      rendered_memory_tokens=sum(len(r["current_text"]) for r in cover) // 4)
        demand["target_miss"] = demand["rendered_memory_tokens"] > demand["memory_budget_tokens"]
        return not demand["target_miss"]
    rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand)
    first = store.records(kinds=["digest"])[0]
    assert demand["target_fits"] is False
    demand["memory_budget_tokens"] = 455 if settle_without_publication else 480
    if settle_without_publication:
        demand.update(owner_context_mode="max", mode="max", window_tokens=500000)
        monkeypatch.setattr(c, "should_consolidate", lambda *_args: False)
        monkeypatch.setattr(chronicle_view, "maintenance_projection", lambda *_args: (fits, demand))
        env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda path: tmp_path / path)
        post_task_synthesis._run_chat_consolidation(env, Memory(tmp_path, tmp_path), None, {"id": "already-fits"}, chat.parent)
    else:
        rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand)
    assert fits()
    assert any(r.get("target_id") == first["id"] and r.get("target_fits") is True
               for r in store.records(kinds=["maintenance"]) if r["id"].startswith("digest-fulfilled:"))
    before = len(calls)
    demand["memory_budget_tokens"] -= 9
    assert not fits()
    assert rooms.compact_chronicle_rooms(store, helper, ctx, "", fits, fitting_demand=demand) == []
    assert len(calls) == before


def test_fit_receipts_do_not_credit_uncaptured_append_or_budget_zero_stop(tmp_path):
    store, _, *_ = setup(tmp_path)
    first = store.publish([{"id": "digest:first", "kind": "digest", "room_id": "1", "text": "Captured"}])[0]
    observed = {"ordinary_maintenance": True, "requirement_tokens": 500000, "target_miss": False,
                "selected_digest_ids": [first["id"]], "memory_budget_tokens": 100}
    late = store.publish([{"id": "digest:later", "kind": "digest", "room_id": "2", "text": "Not in fit observation"}])[0]
    rooms.record_maintenance_fit(store, observed)
    receipts = store.records(kinds=["maintenance"])
    assert [r["target_id"] for r in receipts] == [first["id"]]
    rooms.record_maintenance_fit(store, observed)
    assert store.records(kinds=["maintenance"]) == receipts
    observed.update(selected_digest_ids=[late["id"]], target_miss=True, memory_budget_tokens=0)
    rooms._record_digest_fit(store, "zero-stop", "2", False, True, observed)
    assert store.get("digest-fit:zero-stop")["target_fits"] is False
    assert [r["target_id"] for r in store.records(kinds=["maintenance"])
            if r["id"].startswith("digest-fulfilled:")] == [first["id"]]


@pytest.mark.parametrize("ordinary", [False, True])
def test_raw_room_publication_unit_preserves_interleaved_frontier_and_no_rebuy(tmp_path, monkeypatch, ordinary):
    raw = [{"chat_id": room, "direction": "in", "ts": "2026-09-30T01:00:00Z", "text": text}
           for room, text in [(2, "First A"), (3, "B"), (2, "Later A"), (4, "C")]]
    store, ctx, chat, blocks, meta = setup(tmp_path, raw)
    original = chat.read_bytes(), blocks.read_bytes(), meta.read_bytes()
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_a: helper)
    demand = {"ordinary_maintenance": ordinary}
    expected_rooms = ["2", "3", "4"]
    for step, frontier in enumerate(([1, 3, 4] if ordinary else [4]), 1):
        usage = consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx,
            compact_chronicle=True, pressure_fits=lambda: True, fitting_demand=demand)
        count = step if ordinary else 3
        assert [r["room_id"] for r in store.records(kinds=["episode"])] == expected_rooms[:count]
        assert len(store.records(kinds=["revision"])) == count
        assert store.scan_state()["last_consolidated_offset"] == frontier
        assert usage["_blocks_written"] == (1 if ordinary else 3)
        assert [call[1] for call in helper.calls] == ["Room episode", "Episode correction"] * count
    first = store.records(kinds=["episode"])[0]
    assert first["metadata"]["source_row_ids"] == [source_row_id(raw[0]), source_row_id(raw[2])]
    assert (chat.read_bytes(), blocks.read_bytes(), meta.read_bytes()) == original
    prior_calls = len(helper.calls)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx,
        compact_chronicle=True, pressure_fits=lambda: True, fitting_demand=demand)
    assert len(helper.calls) == prior_calls


def test_ordinary_raw_unit_keeps_own_correction_and_independent_pressure_progress(tmp_path, monkeypatch):
    raw = [{"chat_id": n, "direction": "in", "ts": "2026-09-30T01:00:00Z", "text": f"Completed room {n}"}
           for n in (2, 3)]
    store, ctx, chat, blocks, meta = setup(tmp_path, raw)
    # Immediate mind-authored checking is separate from importing old raw rooms.
    ref = c.retain_memory_source(ctx, "own-source", json.dumps([{"text": "I learned this."}]).encode(), "json")
    own = store.append_episode("own", "My interpretation.", [ref], {"kind": "mind"})
    for n in (1, 2):
        store.append_episode(f"old-{n}", "Older meaningful causal account. " * 1000, [], {"kind": "legacy"}, kind="legacy")
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_a: helper)
    monkeypatch.setattr(c, "_light_route", lambda: {"model": "fake"})
    demand = {"ordinary_maintenance": True, "purpose": "working_headroom"}
    for step in (1, 2):
        if step == 2:
            with chat.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"chat_id": 4, "task_id": "memory-writer", "direction": "in",
                                         "ts": "2026-09-30T02:00:00Z", "text": "New completed experience."}) + "\n")
        before = len(helper.calls)
        consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx,
            compact_chronicle=True, pressure_fits=lambda: False, fitting_demand=demand)
        labels = [call[1] for call in helper.calls[before:]]
        assert labels == (["Episode correction"] if step == 1 else []) + ["Room episode", "Episode correction", "Room digest"]
        assert len(store.records(kinds=["digest"])) == step
    assert store.room_records("own")[0]["correction"]["target_id"] == own["id"]
    assert {r["room_id"] for r in store.records(kinds=["digest"])} == {"old-1", "old-2"}
    assert {r["room_id"] for r in store.records(kinds=["episode"])} == {"own", "2", "3"}


def test_ordinary_raw_correction_budget_failure_keeps_publication_without_rebuy(tmp_path, monkeypatch):
    raw = [{"chat_id": n, "direction": "in", "ts": "2026-09-30T01:00:00Z", "text": f"Completed room {n}"}
           for n in (2, 3)]
    store, ctx, chat, blocks, meta = setup(tmp_path, raw)
    helper = Helper()
    helper.correction_error = "budget_exhausted"
    monkeypatch.setattr(c, "_light_call", lambda *_a: helper)
    demand = {"ordinary_maintenance": True}
    result = consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx,
        compact_chronicle=True, pressure_fits=lambda: False, fitting_demand=demand)
    first = store.records(kinds=["episode"])[0]
    assert result["_consolidation_errors"][-1]["kind"] == "budget_exhausted"
    assert store.scan_state()["last_consolidated_offset"] == 1
    assert len(store.records(kinds=["episode"])) == 1 and not store.records(kinds=["revision"])
    helper.correction_error = None
    before = len(helper.calls)
    consolidate_closed(chat, blocks, meta, None, knowledge_context=ctx,
        compact_chronicle=True, pressure_fits=lambda: True, fitting_demand=demand)
    assert [call[1] for call in helper.calls[before:]] == ["Episode correction", "Room episode", "Episode correction"]
    assert store.records(kinds=["episode"])[0]["id"] == first["id"]
    assert len(store.records(kinds=["episode"])) == len(store.records(kinds=["revision"])) == 2
    assert store.scan_state()["last_consolidated_offset"] == 2
