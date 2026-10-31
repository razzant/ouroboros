"""Complete operative plans of earlier-schema (v2) records survive the bounded hot review-state
round trip: the readers rehydrate the exact spec and evidence manifest from the retained
sources, a damaged source is a typed gap and never an empty plan, and a child's plan sources
promote into its parent. The records are planted as the retired engine's exact-wave writer
left them (``tests._legacy_plan_shared``)."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import task_results
from ouroboros.tools import plan_review_artifacts as artifacts, plan_spec
from tests import _legacy_plan_shared as legacy
from tests._store_shared import read_result, remove_store_tree


def _spec(**fields):
    """A spec as the retired parser stored it (every list present)."""
    return {"goal": "g", "in_scope": [], "non_goals": [], "acceptance_claims": [], "invariants": [],
            "decisions": [], "deferred": [], "affected_resources": [], "evidence": [],
            "normalization_omissions": [], **fields}


def _record(root, spec, *, closed=True, evidence_manifest=None, task_id="large-plan"):
    """One wave recorded as the exact-wave writer left it → ``(spec, hot wave)``."""
    wave = legacy.wave(1, fingerprint="a" * 64, spec=spec, goal=spec["goal"],
                       aggregate="GREEN" if closed else "REVIEW_REQUIRED", closed=closed)
    if evidence_manifest is not None:
        wave["evidence_manifest"] = evidence_manifest
    state = legacy.record_exact_wave(root, task_id, wave)
    return spec, state["waves"][0]


def _raw_state(root, task_id="large-plan"):
    return read_result(root, task_id)["plan_review_state"]


def _claim(text):
    return {"id": "claim_1", "claim": text, "surface": "", "support": "", "priority": "must"}


@pytest.mark.parametrize("field", ["goal", "acceptance_claims", "in_scope"])
def test_large_operative_values_persist_once_and_rehydrate_for_consumers(tmp_path, field):
    from ouroboros.agent_startup_checks import task_result_authority_projection
    from ouroboros.review_evidence_sections import _accept_effective_claims

    large = "complete chosen requirement\n" * 40_000 + "DECISIVE_TAIL"
    fields = {"goal": "Deliver", "acceptance_claims": [_claim("a complete deliverable")]}
    if field == "goal":
        fields["goal"] = large
    elif field == "acceptance_claims":
        fields["acceptance_claims"] = [_claim(large)]
    else:
        fields["in_scope"] = [f"requirement {i}: " + large[:1000] for i in range(1100)]
    spec, _hot = _record(tmp_path, _spec(**fields))
    hot = _raw_state(tmp_path)
    assert hot["waves"][0]["spec"] == {} and hot["waves"][0]["spec_in_artifact"]
    assert "goal" not in hot["waves"][0]
    state = task_results.load_plan_review_state(tmp_path, "large-plan")
    assert state["waves"][0]["spec"] == spec and state["cycles_paid"] == 1
    ctx = SimpleNamespace(task_metadata={}, task_contract={}, drive_root=tmp_path)
    claims, source, _exhibit = _accept_effective_claims(ctx, {}, tmp_path, "large-plan")
    assert claims == spec["acceptance_claims"] and source == "plan_review"
    record = task_results.load_task_result(tmp_path, "large-plan")
    projection = task_result_authority_projection(record, drive_root=tmp_path)
    assert projection["plan_review_state"]["waves"][0]["spec"] == spec
    assert record["plan_review_state"]["waves"][0]["spec"] == {}
    assert _raw_state(tmp_path)["waves"][0]["spec"] == {}


@pytest.mark.parametrize("source", ["wave_artifact", "spec_source_ref"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_unavailable_full_spec_preserves_paid_state_and_never_becomes_empty_claims(tmp_path, damage, source):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.owner_hurry import plan_gate_decision
    from ouroboros.review_evidence_sections import _accept_effective_claims

    _, wave = _record(tmp_path, _spec(goal="Deliver", acceptance_claims=[_claim("required criterion")]))
    path = task_artifact_dir_path(tmp_path, "large-plan", create=False) / wave[source]["path"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"{}")
    assert _raw_state(tmp_path)["cycles_paid"] == 1
    with pytest.raises(artifacts.PlanReviewSourceUnavailable):
        task_results.load_plan_review_state(tmp_path, "large-plan")
    ctx = SimpleNamespace(task_id="large-plan", task_metadata={},
                          task_contract={"objective": "Deliver"}, drive_root=tmp_path)
    # The claims binder the result review reads raises; it never answers with an empty bar.
    with pytest.raises(artifacts.PlanReviewSourceUnavailable):
        _accept_effective_claims(ctx, ctx.task_contract, tmp_path, "large-plan")
    assert plan_gate_decision(ctx, {}, enforcement="blocking")["allow"] is False


def test_legacy_cut_without_source_keeps_its_gap(tmp_path):
    legacy_wave = {"spec": {"goal": "known goal", "in_scope": ["partial…"]},
                   "spec_body_truncated": True}
    state = {"schema_version": 2, "waves": [legacy_wave]}
    assert artifacts.authority_state(tmp_path, "legacy", state) == state
    with pytest.raises(artifacts.PlanReviewSourceUnavailable):
        artifacts.authority_wave(tmp_path, "legacy", legacy_wave)


@pytest.mark.parametrize("example", [
    'Document JSON example {"api_key": "YOUR_API_KEY"}.',
    'The example includes OPENAI_API_KEY=your-key-here.',
])
def test_operative_example_text_keeps_its_reviewed_identity(tmp_path, example):
    """The wave artifact went through the redaction; the operative spec is read from its own
    exact source, so the reviewed text and its hash are what the readers return."""
    spec, hot = _record(tmp_path, _spec(goal=example, acceptance_claims=[_claim(example)]))
    wave = task_results.load_plan_review_state(tmp_path, "large-plan")["waves"][0]
    assert wave["spec"] == spec
    assert plan_spec.spec_hash(wave["spec"]) == wave["spec_hash"]
    evidence = artifacts.read_wave(tmp_path, "large-plan", hot["wave_artifact"])
    assert "***REDACTED***" in evidence["spec"]["goal"]
    assert evidence["spec_source_ref"] == hot["spec_source_ref"]


@pytest.mark.parametrize("compact", [False, True])
def test_spec_source_is_in_the_existing_child_promotion_closure(tmp_path, compact):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.observability import promote_child_task_refs

    child, parent = tmp_path / "child", tmp_path / "parent"
    spec, _ = _record(child, _spec(goal="Keep the complete goal", in_scope=["full requirement\n" * 100_000]))
    state = _raw_state(child)
    if compact:
        state["waves"] = [task_results._compact_plan_review_wave(state["waves"][0])]
    ref = state["waves"][0]["spec_source_ref"]
    copied, receipt = promote_child_task_refs(parent, child, "large-plan", {"plan_review_state": state})
    assert receipt["status"] == "complete"
    copied_wave = copied["plan_review_state"]["waves"][0]
    copied_ref = copied_wave["spec_source_ref"]
    assert copied_ref["sha256"] == ref["sha256"]
    remove_store_tree(child)
    assert json.loads(read_actor_source_bytes(parent, "large-plan", copied_ref)) == spec
    assert artifacts.read_wave(parent, "large-plan", copied_wave["wave_artifact"])["spec"] == spec


def test_large_evidence_manifest_persists_through_the_exact_wave_source(tmp_path):
    from ouroboros.tools import plan_evidence

    urls = [f"https://example.test/source/{i:05d}" for i in range(10_000)]
    manifest = plan_evidence.resolve_evidence(urls, active_root=tmp_path, allowed_roots=[tmp_path])
    assert manifest["declared"] == urls
    assert len(manifest["omissions"]) == len(urls)
    assert all(row["reason"] == "url_not_fetched" for row in manifest["omissions"])
    spec, hot = _record(tmp_path, _spec(goal="Audit supplied sources", evidence=urls), evidence_manifest=manifest)
    stored = _raw_state(tmp_path)
    assert stored["cycles_paid"] == 1 and stored["waves"][0]["closed"]
    assert "evidence_manifest" not in stored["waves"][0]
    restored = task_results.load_plan_review_state(tmp_path, "large-plan")["waves"][0]
    assert restored["spec"] == spec and restored["evidence_manifest"] == manifest
    exact = artifacts.read_wave(tmp_path, "large-plan", restored["wave_artifact"])
    assert exact["evidence_manifest"] == manifest


@pytest.mark.parametrize("damage", ["none", "missing", "corrupt"])
def test_submitted_plan_retains_input_without_finish_or_review_authority(tmp_path, damage):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.review_history_view import current_plan_history

    spec = {"goal": "Exact submitted goal", "decisions": [{"choice": "A", "why": "Owner reason"}]}
    state = legacy.seed_submitted_subject(tmp_path, "submitted", fingerprint="b" * 64, spec=spec,
                                          plan_prose="Exact proposed work")
    ref = state["current_attempt"]["submitted_subject"]
    unavailable = {**state, "current_attempt": {**state["current_attempt"], "status": "unavailable",
                                                "reason": "review_context_unavailable"}}
    legacy.seed_plan_review_state(tmp_path, "submitted", unavailable)
    if damage != "none":
        path = task_artifact_dir_path(tmp_path, "submitted") / ref["path"]
        path.unlink() if damage == "missing" else path.write_bytes(b"{}")
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="submitted")
    history, operative = current_plan_history(ctx)
    if damage == "none":
        assert operative["spec"] == spec and operative["plan_prose"] == "Exact proposed work"
        assert not operative.get("author_disposition") and not operative.get("aggregate")
    else:
        assert operative is None and history["status"] == "source_unavailable"
        assert history["gaps"][-1]["code"] == "PLAN_SUBMITTED_SOURCE_UNAVAILABLE"
    loaded = task_results.load_plan_review_state(tmp_path, "submitted")
    assert loaded["cycles_paid"] == 0 and loaded["waves"] == []
    assert task_results.current_plan_review_wave(loaded) is None
    assert "author_subject" not in loaded["current_attempt"]
    assert loaded["current_attempt"]["submitted_subject"] == ref
