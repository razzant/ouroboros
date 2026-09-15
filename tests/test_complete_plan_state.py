"""Complete operative plans survive the bounded hot review-state round trip."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ouroboros import task_results
from ouroboros.tools import plan_review_artifacts as artifacts, plan_spec
from tests import test_plan_review_engine


plan_review_harness_fixture = pytest.fixture(name="_harness")(
    test_plan_review_engine.harness.__wrapped__
)


def _record(root, raw, *, closed=True, findings=None, evidence_manifest=None):
    spec, errors = plan_spec.normalize_spec(raw)
    assert not errors
    wave = {
        "schema_version": 2, "cycle_index": 1, "request_fingerprint": "a" * 64,
        "goal": spec["goal"], "spec": spec, "spec_hash": plan_spec.spec_hash(spec),
        "aggregate": "GREEN" if closed else "REVIEW_REQUIRED", "closed": closed,
        "findings": findings or [], "dispositions": [], "paid": True,
    }
    if evidence_manifest is not None:
        wave["evidence_manifest"] = evidence_manifest
    result = artifacts.record_exact_wave(
        root, "large-plan", wave, deepcopy(wave), need_evidence_seen=[], page_size=32,
    )
    return spec, result


def _raw_state(root, task_id="large-plan"):
    return json.loads(task_results.task_result_path(root, task_id).read_text(encoding="utf-8"))["plan_review_state"]


@pytest.mark.parametrize("field", ["goal", "acceptance_claims", "in_scope"])
def test_large_operative_values_persist_once_and_rehydrate_for_consumers(tmp_path, field):
    from ouroboros.agent_startup_checks import task_result_authority_projection
    from ouroboros.contracts.task_contract import effective_acceptance_claims
    from ouroboros.review_evidence_sections import _accept_effective_claims

    large = "complete chosen requirement\n" * 40_000 + "DECISIVE_TAIL"
    raw = {"goal": "Deliver", "acceptance_claims": ["a complete deliverable"]}
    raw[field] = (large if field == "goal" else [large] if field == "acceptance_claims"
                  else [f"requirement {i}: " + large[:1000] for i in range(1100)])
    spec, result = _record(tmp_path, raw, closed=False)
    assert result["spec"] == spec and result["goal"] == spec["goal"]
    hot = _raw_state(tmp_path)
    assert hot["cycles_paid"] == 1
    assert len(json.dumps(hot).encode()) < task_results._PLAN_REVIEW_STATE_MAX_BYTES
    assert hot["waves"][0]["spec"] == {} and hot["waves"][0]["spec_in_artifact"]
    assert "goal" not in hot["waves"][0] and not hot["waves"][0]["closed"]
    assert task_results.load_plan_review_state(tmp_path, "large-plan")["waves"][0]["spec"] == spec
    task_results.record_plan_review_dispositions(
        tmp_path, "large-plan", fingerprint="a" * 64,
        dispositions=[], closed=True, closure_notes=[],
    )
    state = task_results.load_plan_review_state(tmp_path, "large-plan")
    wave = task_results.closed_plan_review_wave(state)
    assert wave["spec"] == spec and wave["goal"] == spec["goal"]
    assert effective_acceptance_claims({}, wave)[0] == spec["acceptance_claims"]
    ctx = SimpleNamespace(task_metadata={}, task_contract={}, drive_root=tmp_path)
    claims, source, _ = _accept_effective_claims(ctx, {}, tmp_path, "large-plan")
    assert claims == spec["acceptance_claims"] and source == "plan_review"
    record = task_results.load_task_result(tmp_path, "large-plan")
    projection = task_result_authority_projection(record, drive_root=tmp_path)
    assert projection["plan_review_state"]["waves"][0]["spec"] == spec
    assert record["plan_review_state"]["waves"][0]["spec"] == {}
    assert _raw_state(tmp_path)["waves"][0]["spec"] == {}
    assert task_results.load_plan_review_state(tmp_path, "large-plan")["cycles_paid"] == 1


def test_full_plan_review_disposition_repeat_and_tail_delta(_harness):
    from tests.test_plan_review_engine import CLEAN, _call, _control, _finding, _state, _user_text
    from ouroboros.tools.plan_review import _apply_disposition

    note = json.dumps([_finding("n1", "note", summary="Consider another example")])
    substrate = _harness.install({"s1": note, "s2": CLEAN, "s3": CLEAN})
    ctx = _harness.make_ctx()
    goal = "full chosen goal\n" * 22_000 + "GOAL_TAIL"
    spec = {"in_scope": ["full requirement\n" * 22_000 + "SCOPE_TAIL_A"],
            "acceptance_claims": ["full criterion\n" * 22_000 + "CLAIM_TAIL"],
            "affected_paths": []}  # required on every submitted spec (owner 9=A)
    assert _control(_call(ctx, spec, goal=goal)) == {"outcome": "REVIEW_REQUIRED", "closed": True}
    state = _state(_harness)
    fingerprint = state["waves"][-1]["request_fingerprint"]
    finding = state["waves"][-1]["findings"][0]["finding_id"]
    disposed = _apply_disposition(ctx, {"review_fingerprint": fingerprint, "items": [
        {"finding_id": finding, "decision": "reject", "rationale": "Existing example suffices"},
    ]})
    assert _control(disposed)["closed"]
    replay = _call(ctx, spec, goal=goal)
    assert "cached exact review" in replay and len(substrate.calls) == 1
    assert _state(_harness)["cycles_paid"] == 1
    assert _raw_state(_harness.drive, "task-1")["waves"][-1]["spec"] == {}
    changed = deepcopy(spec)
    changed["in_scope"][0] = changed["in_scope"][0].replace("TAIL_A", "TAIL_B")
    assert _control(_call(ctx, changed, goal=goal))["outcome"] == "REVIEW_REQUIRED"
    assert len(substrate.calls) == 2 and _state(_harness)["cycles_paid"] == 2
    sent = _user_text(substrate.calls[-1]["request"].messages[-1]["content"])
    assert "SCOPE_TAIL_B" in sent and "previous frozen spec body truncated" not in sent


@pytest.mark.parametrize("source", ["wave_artifact", "spec_source_ref"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_unavailable_full_spec_preserves_paid_state_and_never_becomes_empty_claims(tmp_path, damage, source):
    from ouroboros.owner_hurry import force_plan_decision
    from ouroboros.review_evidence import build_task_acceptance_evidence

    _, wave = _record(tmp_path, {"goal": "Deliver", "acceptance_claims": ["required criterion"]})
    ref = wave[source]
    from ouroboros.artifacts import task_artifact_dir_path

    path = task_artifact_dir_path(tmp_path, "large-plan", create=False) / ref["path"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"{}")
    assert _raw_state(tmp_path)["cycles_paid"] == 1
    with pytest.raises(artifacts.PlanReviewSourceUnavailable):
        task_results.load_plan_review_state(tmp_path, "large-plan")
    ctx = SimpleNamespace(task_id="large-plan", task_metadata={},
                          task_contract={"objective": "Deliver"}, drive_root=tmp_path)
    with pytest.raises(artifacts.PlanReviewSourceUnavailable):
        build_task_acceptance_evidence(ctx, drive_root=tmp_path, task_id="large-plan")
    assert force_plan_decision(ctx, {}, enforcement="blocking")["allow"] is False


def test_legacy_cut_without_source_keeps_its_gap(tmp_path):
    legacy = {"spec": {"goal": "known goal", "in_scope": ["partial…"]},
              "spec_body_truncated": True}
    state = {"schema_version": 2, "waves": [legacy]}
    assert artifacts.authority_state(tmp_path, "legacy", state) == state
    with pytest.raises(artifacts.PlanReviewSourceUnavailable):
        artifacts.authority_wave(tmp_path, "legacy", legacy)


@pytest.mark.parametrize("example", [
    'Document JSON example {"api_key": "YOUR_API_KEY"}.',
    'The example includes OPENAI_API_KEY=your-key-here.',
])
def test_operative_example_text_keeps_its_reviewed_identity(tmp_path, example):
    spec, wave = _record(tmp_path, {
        "goal": example, "acceptance_claims": [example],
    })
    assert wave["spec"] == spec
    assert plan_spec.spec_hash(wave["spec"]) == wave["spec_hash"]
    evidence = artifacts.read_wave(tmp_path, "large-plan", wave["wave_artifact"])
    assert "***REDACTED***" in evidence["spec"]["goal"]
    assert evidence["spec_source_ref"] == wave["spec_source_ref"]


@pytest.mark.parametrize("compact", [False, True])
def test_spec_source_is_in_the_existing_child_promotion_closure(tmp_path, compact):
    import shutil
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.observability import promote_child_task_refs

    child, parent = tmp_path / "child", tmp_path / "parent"
    spec, _ = _record(child, {"goal": "Keep the complete goal", "in_scope": ["full requirement\n" * 100_000]})
    state = _raw_state(child)
    if compact:
        state["waves"] = [task_results._compact_plan_review_wave(state["waves"][0])]
    ref = state["waves"][0]["spec_source_ref"]
    copied, receipt = promote_child_task_refs(parent, child, "large-plan", {"plan_review_state": state})
    assert receipt["status"] == "complete"
    copied_wave = copied["plan_review_state"]["waves"][0]
    copied_ref = copied_wave["spec_source_ref"]
    assert copied_ref["sha256"] == ref["sha256"]
    shutil.rmtree(child)
    assert json.loads(read_actor_source_bytes(parent, "large-plan", copied_ref)) == spec
    assert artifacts.read_wave(parent, "large-plan", copied_wave["wave_artifact"])["spec"] == spec


def test_last_resort_hot_state_fit_preserves_exact_source_references(tmp_path):
    wide = "𝕏" * 1000
    findings = [{"finding_id": f"s{slot}:f{i}", "slot": str(slot), "class": "note",
                 "summary": wide, "recommendation": wide, "locator": wide, "detail": wide}
                for slot in range(10) for i in range(32)]
    spec, wave = _record(tmp_path, {"goal": "Preserve every source"}, closed=False, findings=findings)
    hot = _raw_state(tmp_path)["waves"][0]
    assert hot["findings_texts_truncated"]
    assert hot["spec_source_ref"] == wave["spec_source_ref"]
    assert hot["wave_artifact"] == wave["wave_artifact"]
    assert task_results.load_plan_review_state(tmp_path, "large-plan")["waves"][0]["spec"] == spec


def test_large_evidence_manifest_persists_through_the_exact_wave_source(tmp_path):
    from ouroboros.tools import plan_evidence

    urls = [f"https://example.test/source/{i:05d}" for i in range(10_000)]
    manifest = plan_evidence.resolve_evidence(urls, active_root=tmp_path, allowed_roots=[tmp_path])
    assert manifest["declared"] == urls
    assert len(manifest["omissions"]) == len(urls)
    assert all(row["reason"] == "url_not_fetched" for row in manifest["omissions"])
    spec, wave = _record(
        tmp_path, {"goal": "Audit supplied sources", "evidence": urls},
        evidence_manifest=manifest,
    )
    hot = _raw_state(tmp_path)
    assert hot["cycles_paid"] == 1 and hot["waves"][0]["closed"]
    assert "evidence_manifest" not in hot["waves"][0]
    assert len(json.dumps(hot).encode()) < task_results._PLAN_REVIEW_STATE_MAX_BYTES
    assert wave["evidence_manifest"] == manifest
    restored = task_results.load_plan_review_state(tmp_path, "large-plan")["waves"][0]
    assert restored["spec"] == spec and restored["evidence_manifest"] == manifest
    exact = artifacts.read_wave(tmp_path, "large-plan", restored["wave_artifact"])
    assert exact["evidence_manifest"] == manifest
