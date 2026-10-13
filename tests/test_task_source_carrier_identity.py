"""A task_source reference is placement; retention verifies its bytes and never rewrites it.

Child copy-back promotion once let the first-registered shape of a file decide
every carrier of that file: a compact capsule handle (no ``read`` by contract)
failed a read-contract precondition, and its path-less placeholder replaced the
healthy full plan references too, in a frozenset field order that changed with
each process's hash seed (Continue of the Quotas task, 11.10.2026). These tests
pin the class: every carrier keeps its own reference in any registration order
and under any seed, a failure is a promotion fact, and a legacy path-less gap
regains only the one address its own row recorded.
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
import os
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, headless, observability
from ouroboros import context_compaction as cc
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_context_reclaim_materializer import _request, _unit

REPO = pathlib.Path(__file__).resolve().parents[1]
FINGERPRINT = "f" * 64
SOURCE_FIELDS = ("review_evidence", "completion_observations", "plan_review_state")


def _compact(ref: dict) -> dict:
    """The closed focus/capsule handle shape, which carries no ``read``."""
    return {key: ref[key] for key in ("kind", "root", "path", "size", "sha256")}


def _record_plan(root: pathlib.Path, task_id: str) -> tuple[dict, dict]:
    """The real plan-review writer: exact wave and spec sources plus the hot index."""
    from ouroboros.tools import plan_review_artifacts, plan_spec

    spec, errors = plan_spec.normalize_spec({"goal": "Keep the tower", "acceptance_claims": ["three levels"]})
    assert not errors
    wave = {"schema_version": 2, "cycle_index": 1, "request_fingerprint": FINGERPRINT, "goal": spec["goal"],
            "spec": spec, "spec_hash": plan_spec.spec_hash(spec), "aggregate": "GREEN", "closed": True,
            "findings": [], "dispositions": [], "paid": True}
    plan_review_artifacts.record_exact_wave(root, task_id, wave, copy.deepcopy(wave),
                                            need_evidence_seen=[], page_size=32)
    return spec, load_task_result(root, task_id)["plan_review_state"]


def _capsule_request(child: pathlib.Path, task_id: str, handles: list) -> dict:
    """An LLM request whose real context capsule names compact handles of the same files."""
    messages = _unit("read-1", "x")
    request = _request(messages, 100)
    unit = cc._atomic_units(messages)[-1]
    checkpoint = cc._persist_reclaim_checkpoint(messages, request, SimpleNamespace(fingerprint="c" * 64, units=[]),
                                                drive_root=child, task_id=task_id)
    unit = dataclasses.replace(unit, source_refs=(*unit.source_refs, *({"source_ref": ref} for ref in handles)))
    capsule, _ = cc._capsule_message(cc._SelectedUnit(unit, 0, ""), "summary",
                                     [cc._part(unit.unit_id, unit.source_text)], checkpoint, request)
    assert cc._capsule_metadata(capsule)[1] is not None  # the walk follows a valid capsule's sources
    return observability.persist_call(child, task_id=task_id, call_id="capsule-request", call_type="llm_request",
                                      payload={"messages": [capsule]})["manifest_ref"]


def _carrier_fixture(root: pathlib.Path):
    """Four files, each named by a full producer reference AND a compact capsule handle."""
    from ouroboros.chat_chain import retain_memory_source
    from ouroboros.tool_result_delivery import prepare_producer_sources, with_producer_source

    parent, task_id = root / "canonical", "carrier-identity"
    parent.mkdir(parents=True)
    child = headless.prepare_task_drive(parent, task_id, "empty")
    _spec, plan = _record_plan(child, task_id)
    wave = plan["waves"][-1]
    # The system's own producers: a runtime_data read envelope and a producer marker.
    memory = retain_memory_source(SimpleNamespace(drive_root=child, task_id=task_id), "memory-note", b"memory bytes")
    assert memory["read"]["arguments"]["root"] == "runtime_data"
    tool_row = prepare_producer_sources([{
        "tool_call_id": "call-1", "result": "annotated result",
        "tool_result": SimpleNamespace(producer_text="exact producer bytes", host_annotations=("host note",)),
    }], child, task_id)[0]
    marker = with_producer_source(tool_row, tool_row["result"])
    assert '"kind": "task_source"' in marker  # the producer writes JSON with spaces
    full = [wave["wave_artifact"], wave["spec_source_ref"], memory, tool_row["producer_source_ref"]]
    body = {
        "trace_refs": {"llm_call_refs": [{"request_ref": _capsule_request(child, task_id, [_compact(r) for r in full])}]},
        "review_evidence": {"source_refs": [memory]},
        # A host message carrier the walk follows outside captured JSON.
        "completion_observations": {"messages": [{"role": "tool", "tool_call_id": "call-1", "content": marker}]},
        "plan_review_state": plan,
    }
    return parent, child, task_id, body, full


def _run_seeded(seed: str, script: str, *args: str):
    completed = subprocess.run([sys.executable, "-c", script, *args], cwd=REPO, capture_output=True, text=True,
                               env={**os.environ, "PYTHONHASHSEED": seed}, timeout=300)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("order", ["capsule_first", "full_reference_first"])
def test_every_task_source_carrier_keeps_its_reference_in_either_registration_order(tmp_path, monkeypatch, order):
    """The live poisoning order (trace capsules before the plan) and its reverse."""
    parent, child, task_id, body, full = _carrier_fixture(tmp_path)
    fields = observability._PUBLISHED_CHILD_REF_FIELDS
    walked = fields if order == "capsule_first" else tuple(reversed(fields))
    assert (walked.index("trace_refs") < walked.index("plan_review_state")) == (order == "capsule_first")
    monkeypatch.setattr(observability, "_PUBLISHED_CHILD_REF_FIELDS", walked)
    before = {key: json.dumps(body[key], ensure_ascii=False) for key in SOURCE_FIELDS}
    originals = [artifacts.read_actor_source_bytes(child, task_id, ref) for ref in full]

    copied, state = observability.promote_child_task_refs(parent, child, task_id, copy.deepcopy(body))

    assert (state["status"], state["unavailable_refs"], state["pending_refs"]) == ("complete", [], [])
    # Byte-identical, key order and the producer's marker spacing included.
    assert {key: json.dumps(copied[key], ensure_ascii=False) for key in SOURCE_FIELDS} == before
    assert copied["completion_observations"]["messages"][0]["content"] == body["completion_observations"]["messages"][0]["content"]
    shutil.rmtree(child)
    for ref, raw in zip(full, originals):  # verified and copied, whatever shape reached the file first
        assert artifacts.read_actor_source_bytes(parent, task_id, ref) == raw
        assert artifacts.read_actor_source_bytes(parent, task_id, _compact(ref)) == raw


_SEEDED_PROMOTION = """
import json, sys
from ouroboros.observability import promote_child_task_refs
parent, child, task_id, body_path = sys.argv[1:5]
with open(body_path, encoding="utf-8") as stream:
    body = json.load(stream)
copied, state = promote_child_task_refs(parent, child, task_id, body)
fields = ("review_evidence", "completion_observations", "plan_review_state")
print(json.dumps({"same": [json.dumps(copied[key], ensure_ascii=False) == json.dumps(body[key], ensure_ascii=False)
                           for key in fields],
                  "status": state["status"], "unavailable": len(state["unavailable_refs"])}))
"""


@pytest.mark.serial  # real subprocesses
@pytest.mark.parametrize("seed", ["0", "1", "2", "3"])
def test_retention_outcome_does_not_depend_on_the_process_hash_seed(tmp_path, seed):
    """The production field constant, unpatched, in fresh processes with different seeds."""
    parent, child, task_id, body, _full = _carrier_fixture(tmp_path)
    body_path = tmp_path / "body.json"
    body_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")

    outcome = _run_seeded(seed, _SEEDED_PROMOTION, str(parent), str(child), task_id, str(body_path))

    assert outcome == {"same": [True, True, True], "status": "complete", "unavailable": 0}


_SEEDED_WALK = """
import json, pathlib, sys
from ouroboros import observability as obs
seen, real = [], obs._rewrite_child_ref_tree
def spy(value, *args, **kwargs):
    if isinstance(value, dict) and set(value) == {"walked_field"} and value["walked_field"] not in seen:
        seen.append(value["walked_field"])
    return real(value, *args, **kwargs)
obs._rewrite_child_ref_tree = spy
root = pathlib.Path(sys.argv[1])
(root / "parent").mkdir(parents=True)
(root / "child").mkdir()
fields = [key for key in obs._PUBLISHED_CHILD_REF_FIELDS
          if key not in {"task_contract", "attachment_manifest", "attachment_manifest_ref"}]
obs.promote_child_task_refs(root / "parent", root / "child", "walk-order", {key: {"walked_field": key} for key in fields})
print(json.dumps(seen))
"""


@pytest.mark.serial  # real subprocesses
def test_published_fields_are_walked_in_written_order_under_every_hash_seed(tmp_path):
    skipped = {"task_contract", "attachment_manifest", "attachment_manifest_ref"}
    assert isinstance(observability._PUBLISHED_CHILD_REF_FIELDS, tuple)
    written = [key for key in observability._PUBLISHED_CHILD_REF_FIELDS if key not in skipped]

    orders = {seed: _run_seeded(seed, _SEEDED_WALK, str(tmp_path / seed)) for seed in ("0", "1", "2", "3")}

    assert orders == {seed: written for seed in orders}


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_a_failed_source_keeps_its_carrier_and_a_retry_of_the_returned_row_heals(tmp_path, damage):
    parent, task_id = tmp_path / "canonical", f"failed-{damage}"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task_id, "empty")
    ref = artifacts.store_actor_source_bytes(child, task_id, category="tool_results", source_id="answer",
                                             data=b"exact answer", extension="txt")
    path = artifacts.task_artifact_dir_path(child, task_id) / ref["path"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"exact answeR")  # same size, different digest
    blob = observability.write_blob(child, {"result": "original"})
    with gzip.open(blob["path"], "wb") as handle:
        handle.write(b'{"result":"tampered"}')
    body = {"review_evidence": {"source_refs": [ref, _compact(ref)]},
            "trace_refs": {"tool_call_refs": [{"redacted_projection_ref": blob}]}}

    copied, state = observability.promote_child_task_refs(parent, child, task_id, copy.deepcopy(body))

    assert copied["review_evidence"] == body["review_evidence"]  # both shapes keep their own reference
    facts = [row for row in state["unavailable_refs"] if row["kind"] == "task_source"]
    assert [(row["path"], row["owner_task_id"], row["reason"]) for row in facts] == [
        (ref["path"], task_id, "source_missing" if damage == "missing" else "digest_mismatch")]
    gap = copied["trace_refs"]["tool_call_refs"][0]["redacted_projection_ref"]
    assert gap["availability"] == "unavailable" and "path" not in gap  # an absolute blob locator is still dropped

    path.write_bytes(b"exact answer")
    healed, healed_state = observability.promote_child_task_refs(parent, child, task_id, copied)

    assert healed["review_evidence"] == body["review_evidence"]
    assert not [row for row in healed_state["unavailable_refs"] if row["kind"] == "task_source"]
    shutil.rmtree(child)
    assert artifacts.read_actor_source_bytes(parent, task_id, ref) == b"exact answer"


def _legacy_damage(state: dict, task_id: str) -> tuple[dict, list]:
    """Exactly what the old walk wrote: a path-less placeholder per carrier, the path kept in a fact."""
    damaged, facts = copy.deepcopy(state), []
    for wave in damaged["waves"]:
        for key in ("wave_artifact", "spec_source_ref"):
            facts.append(observability._promotion_fact(
                {**wave[key], "owner_task_id": task_id, "source_error_type": "ValueError"}, "invalid_ref"))
            wave[key] = observability._typed_unavailable_ref(wave[key], "invalid_ref")
    return damaged, facts


def _promotion(status: str, facts: list, pending: list) -> dict:
    return {"schema_version": 1, "status": status, "promoted_ref_count": 0, "promoted_source_handle_count": 0,
            "pending_refs": pending, "unavailable_refs": facts}


def test_a_pending_legacy_row_heals_on_one_retry_and_leaves_the_ledger(tmp_path):
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.tools.plan_review_artifacts import PlanReviewSourceUnavailable

    parent, task_id = tmp_path / "canonical", "legacy-pending"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task_id, "empty")
    spec, plan = _record_plan(child, task_id)
    original = plan["waves"][-1]
    damaged, facts = _legacy_damage(plan, task_id)
    pending = [{"kind": "history_retention_deferred", "path": str(child), "reason": "call_inventory_unavailable"}]
    write_task_result(parent, task_id, "failed", result="provider unavailable", plan_review_state=damaged,
                      child_ref_promotion=_promotion("incomplete", facts, pending),
                      headless_child_drive_root=str(child), child_drive_root=str(child))
    with pytest.raises(PlanReviewSourceUnavailable):
        load_plan_review_state(parent, task_id)  # the strict decider sees the damage

    report = observability.retry_pending_child_ref_promotions(parent, generation=object())

    assert (report["scanned"], report["completed"], report["errors"]) == (1, [task_id], [])
    row = load_task_result(parent, task_id)
    wave = row["plan_review_state"]["waves"][-1]
    assert wave["wave_artifact"] == _compact(original["wave_artifact"])
    assert wave["spec_source_ref"] == _compact(original["spec_source_ref"])
    assert (row["child_ref_promotion"]["status"], row["child_ref_promotion"]["unavailable_refs"]) == ("complete", [])
    shutil.rmtree(child)
    assert load_plan_review_state(parent, task_id)["waves"][-1]["spec"] == spec  # verified canonical bytes
    assert observability.retry_pending_child_ref_promotions(parent, generation=object())["scanned"] == 0


def test_a_complete_legacy_row_is_readable_in_memory_and_never_rewritten(tmp_path):
    """No pending ledger row will retry it: the read projection restores, nothing writes."""
    from ouroboros.agent_startup_checks import task_result_authority_projection
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.task_store import read_document, signature
    from ouroboros.tools.plan_review_artifacts import PlanReviewSourceUnavailable

    task_id = "legacy-complete"
    spec, plan = _record_plan(tmp_path, task_id)  # the canonical store holds the bytes, like the live rows
    damaged, facts = _legacy_damage(plan, task_id)
    write_task_result(tmp_path, task_id, "completed", result="done", plan_review_state=damaged,
                      child_ref_promotion=_promotion("complete", facts, []))
    row_before = read_document(tmp_path, task_id)
    signature_before = signature(tmp_path, task_id)

    authority = task_result_authority_projection(load_task_result(tmp_path, task_id), drive_root=tmp_path)

    assert "plan_review_source_gap" not in authority
    assert authority["plan_review_state"]["waves"][-1]["spec"] == spec
    assert read_document(tmp_path, task_id) == row_before
    assert signature(tmp_path, task_id) == signature_before
    with pytest.raises(PlanReviewSourceUnavailable):
        load_plan_review_state(tmp_path, task_id)  # deciders stay strict


def test_a_restored_address_whose_bytes_are_gone_stays_addressed_with_a_fact(tmp_path):
    parent, task_id = tmp_path / "canonical", "legacy-missing"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task_id, "empty")
    _spec, plan = _record_plan(child, task_id)
    original = plan["waves"][-1]
    damaged, facts = _legacy_damage(plan, task_id)
    (artifacts.task_artifact_dir_path(child, task_id) / original["spec_source_ref"]["path"]).unlink()
    body = {"plan_review_state": damaged, "child_ref_promotion": _promotion("incomplete", facts, [])}

    copied, state = observability.promote_child_task_refs(parent, child, task_id, body, retry_only=True)

    wave = copied["plan_review_state"]["waves"][-1]
    assert wave["wave_artifact"] == _compact(original["wave_artifact"])  # verified and retained
    assert wave["spec_source_ref"] == _compact(original["spec_source_ref"])  # addressed, so the next retry can heal
    assert [(row["path"], row["reason"]) for row in state["unavailable_refs"]] == [
        (original["spec_source_ref"]["path"], "source_missing")]


@pytest.mark.parametrize("case", ["no_recorded_path", "two_recorded_paths", "other_owner"])
def test_a_gap_without_exactly_one_own_recorded_path_stays_a_gap(case):
    digest = "d" * 64
    ref = {"kind": "task_source", "root": "artifact_store", "size": 5959, "sha256": digest,
           "path": f"source_handles/context_checkpoints/plan-spec-{digest}.json"}
    gap = observability._typed_unavailable_ref(ref, "invalid_ref")
    fact = observability._promotion_fact({**ref, "owner_task_id": "owner", "source_error_type": "ValueError"},
                                         "invalid_ref")
    facts = {"no_recorded_path": [{key: value for key, value in fact.items() if key != "path"}],
             "two_recorded_paths": [fact, {**fact, "path": f"source_handles/context_checkpoints/plan-copy-{digest}.json"}],
             "other_owner": [{**fact, "owner_task_id": "another-task"}]}[case]
    value = {"waves": [{"spec_source_ref": gap}]}

    assert observability.restore_recorded_task_source_gaps(value, {"unavailable_refs": facts}, "owner") == value
    # Positive control: exactly one own recorded path restores the same gap.
    assert observability.restore_recorded_task_source_gaps(value, {"unavailable_refs": [fact]}, "owner") == {
        "waves": [{"spec_source_ref": _compact(ref)}]}


def test_a_successor_review_binds_every_addressed_source_past_an_unrestorable_gap(tmp_path):
    """A gap no own recorded path restores rides into a started successor's task contract;
    reader binding skips it (it names no address) and still binds every addressed source."""
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.review_source_closure import _reader_bindings
    from tests.test_doc_context import _make_env_and_memory

    env, _memory = _make_env_and_memory(tmp_path)
    root, pred = env.drive_root, "legacy-pred"
    _spec, plan = _record_plan(root, pred)
    wave = plan["waves"][-1]
    intact = wave["wave_artifact"]
    wave["spec_source_ref"] = observability._typed_unavailable_ref(wave["spec_source_ref"], "invalid_ref")
    write_task_result(root, pred, "failed", result="x", reason_code="provider_unavailable",
                      task_contract={"objective": "o"}, plan_review_state=plan,
                      child_ref_promotion=_promotion("complete", [], []))  # no fact: nothing restores it
    task = {"id": "succ", "type": "task", "text": "continue", "budget_drive_root": str(root),
            "predecessor_authority_source": {"kind": "task_result", "task_id": pred, "tool": "get_task_result",
                                             "arguments": {"task_id": pred, "include_authority": True}}}
    assert validate_task_authority_sources(env, task) == {}
    assert task["predecessor_authority"]["plan_review_source_gap"]["code"] == "PLAN_REVIEW_SOURCE_UNAVAILABLE"
    own = artifacts.store_actor_source_bytes(root, "succ", category="tool_results", source_id="own",
                                             data=b"own bytes", extension="txt")
    request = {"evidence": {"task_contract": build_task_contract(task), "source_refs": [own]}}

    bindings = _reader_bindings(request, root, "succ")

    assert [(row["owner_task_id"], row["source_path"]) for row in bindings] == [
        (pred, intact["path"]), ("succ", own["path"])]
