"""Author reaction reaches execution and publication without inventing critic PASS."""
from __future__ import annotations

import base64
import json

import pytest

from ouroboros.skill_loader import (
    SkillReviewState, compute_content_hash, load_skill, load_review_state,
    save_review_state, save_enabled,
)
from ouroboros.skill_review import SkillReviewOutcome
from ouroboros.skill_review_runner import _write_review_job, review_job_state_path
from ouroboros.tool_access_types import ResolvedResourceBinding
from ouroboros.tools import skill_exec, skill_publish
from tests.test_skill_exec import _build_skill, _make_ctx


@pytest.fixture
def subject(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    ctx = _make_ctx(tmp_path)
    directory = _build_skill(ctx.drive_root / "skills" / "external", "demo")
    binding = ResolvedResourceBinding(
        profile="self_modification", root="skill_payload", operation="review",
        base_path=directory, target_path=directory, source="test",
        skill_name="demo", state_drive_root=ctx.drive_root,
    )
    monkeypatch.setattr(skill_exec, "_skill_tool_preflight", lambda *a, **kw: "")
    return ctx, directory, binding


def _finish(subject, *, reference=None):
    ctx, _directory, binding = subject
    return skill_exec._author_finish_existing_skill_review(
        ctx, binding, "demo", disposition="partial",
        rationale="Verified the current implementation; retain the original reviewer evidence.",
        review_reference=reference,
    )


def _unavailable(subject, *, status="completed", finished=True):
    ctx, directory, _binding = subject
    content_hash = compute_content_hash(directory)
    reference = {"job_id": "review-unavailable", "content_hash": content_hash}
    _write_review_job(review_job_state_path(ctx.drive_root, "demo"), {
        **reference, "skill": "demo", "status": status, "review_status": "pending",
        "finished_at": "2026-09-18T10:00:00Z" if finished else "",
    })
    return reference


def _critic_then_change(subject):
    ctx, directory, _binding = subject
    old_hash = compute_content_hash(directory)
    findings = [{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical", "reason": "Retained critic finding."}]
    save_review_state(ctx.drive_root, "demo", SkillReviewState(
        status="blockers", content_hash=old_hash, findings=findings,
        raw_actor_records=[{"slot_id": "critic", "status": "ok"}],
    ))
    (directory / "scripts/hello.py").write_text("print('author revised')\n", encoding="utf-8")
    assert "error" not in _finish(subject)
    loaded = load_skill(directory, ctx.drive_root)
    loaded.source = "external"
    return loaded, old_hash, findings


@pytest.mark.parametrize("partial", [False, True])
def test_main_receives_terminal_reference_then_finishes_without_second_panel(subject, monkeypatch, partial):
    ctx, directory, binding = subject
    calls = []

    def unavailable_review(*args, **kwargs):
        calls.append(True)
        findings = [{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical", "reason": "Partial critic finding."}] if partial else []
        actors = ([{"slot_id": "critic1", "status": "responded", "parsed_items": findings},
                   {"slot_id": "critic2", "operation_state": "in_flight", "late_result_pending": True}] if partial else [])
        content_hash = compute_content_hash(directory)
        # What review_skill projects for a settled wave that formed no verdict: the pending state with the
        # seats' raw records (the partial feedback an author finish may cite), never a verdict.
        save_review_state(ctx.drive_root, "demo", SkillReviewState(status="pending", content_hash=content_hash,
                                                                   findings=findings, raw_actor_records=actors))
        return SkillReviewOutcome(skill_name="demo", status="pending", content_hash=content_hash,
                                  error="No reviewer quorum: configured route unavailable.",
                                  findings=findings, raw_actor_records=actors)

    monkeypatch.setattr(skill_exec, "_review_skill_impl", unavailable_review)
    # Real lifecycle records the terminal job and returns it to Main's first call.
    first = skill_exec._handle_review_skill(ctx, skill="demo", _resolved_binding=binding)
    assert "unavailable" in first
    reference = json.loads(first.rsplit("Review reference: ", 1)[1])
    assert not load_skill(directory, ctx.drive_root).review.gate_for(reference["content_hash"])["executable_review"]
    job_before_finish = review_job_state_path(ctx.drive_root, "demo").read_bytes()
    second = skill_exec._handle_review_skill(ctx, skill="demo", _resolved_binding=binding,
        author_disposition="accepted", author_rationale="Local verification passed; independent review was unavailable.",
        review_reference=reference)
    assert "Author finish recorded" in second
    assert len(calls) == 1
    assert review_job_state_path(ctx.drive_root, "demo").read_bytes() == job_before_finish
    loaded = load_skill(directory, ctx.drive_root)
    assert loaded.review.status == "pending"
    reference = loaded.review.author_disposition["review_reference"]
    assert reference["basis"] == ("partial_feedback" if partial else "unavailable")
    if partial:
        assert "(critical): Partial critic finding." in skill_publish._advisory_findings_section(loaded.review)
        assert "partial_feedback" in skill_publish._author_checklist(loaded.review, loaded.content_hash)
    assert loaded.review.gate_for(loaded.content_hash)["executable_review"]
    save_enabled(ctx.drive_root, "demo", True)
    result = json.loads(skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py"))
    assert result["exit_code"] == 0 and "hello from skill" in result["stdout"]
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    assert "SKILL_EXEC_BLOCKED" in skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py")


def test_exec_schema_states_the_advisory_author_path_its_gate_permits(subject, monkeypatch):
    """skill_exec's description is the gate it obeys: outside Cyber Pro a qualified
    Advisory author acceptance runs the current bytes while the reviewer status is
    still pending, and enablement, grants and dependencies stay separate conditions."""
    ctx, directory, _ = subject
    assert "error" not in _finish(subject, reference=_unavailable(subject))
    loaded = load_skill(directory, ctx.drive_root)
    gate = loaded.review.gate_for(loaded.content_hash)
    assert (loaded.review.status, gate["executable_review"], gate["blocking_reason"]) == (
        "pending", True, "author_accepted_advisory")
    description = skill_exec._EXEC_SCHEMA["description"]
    assert "fresh executable review verdict" not in description
    assert "Advisory author acceptance" in description and "still be pending" in description
    assert all(term in description for term in ("must be enabled", "grants", "dependencies", "Cyber Pro"))
    save_enabled(ctx.drive_root, "demo", True)
    result = json.loads(skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py"))
    assert result["exit_code"] == 0
    save_enabled(ctx.drive_root, "demo", False)
    assert "is disabled" in skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py")


def test_toggle_schema_states_the_gate_its_enable_owner_obeys(subject, monkeypatch):
    """#1735: toggle_skill enables through the same author-aware gate as skill_exec, so its
    text names both authorities; Blocking enforcement still refuses the same bytes."""
    ctx, directory, _ = subject
    assert "error" not in _finish(subject, reference=_unavailable(subject))
    assert load_skill(directory, ctx.drive_root).review.status == "pending"
    description = skill_exec._TOGGLE_SCHEMA["description"]
    assert "fresh executable review" not in description
    assert all(term in description for term in (
        "Advisory author acceptance", "grants", "dependencies", "Cyber Pro"))
    enabled = json.loads(skill_exec._handle_toggle_skill(ctx, skill="demo", enabled=True))
    assert enabled["enabled"] is True and enabled["review_gate"]["blocking_reason"] == "author_accepted_advisory"
    assert json.loads(skill_exec._handle_toggle_skill(ctx, skill="demo", enabled=False))["enabled"] is False
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    refused = skill_exec._handle_toggle_skill(ctx, skill="demo", enabled=True)
    # Inverted pin: the refusal is typed (code + one owner sentence); the gate's need moved to ``detail``.
    assert refused.startswith("⚠️ SKILL_TOGGLE_ERROR") and '"code": "acceptance"' in refused and (
        '"detail": "a current executable reviewer verdict"') in refused


def test_refusals_name_the_authorities_their_gate_admits(subject, monkeypatch):
    """#1735: a refused gate names what it still needs. Advisory admits a current reviewer verdict
    or a qualified author acceptance of these bytes; Blocking admits the verdict only. skill_exec,
    toggle_skill and the grant owner say the same thing their gate decided."""
    from ouroboros.skill_lifecycle_actions import prepare_skill_grant

    ctx, directory, _ = subject
    # A review of these exact bytes that produced no executable verdict (not a stale one).
    save_review_state(ctx.drive_root, "demo", SkillReviewState(status="pending", content_hash=compute_content_hash(directory)))
    both = "a current executable reviewer verdict or a qualified Advisory author acceptance of these exact bytes"
    for enforcement, needed in (("advisory", both), ("blocking", "a current executable reviewer verdict")):
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
        save_enabled(ctx.drive_root, "demo", True)
        ran = skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py")
        assert ran.startswith("⚠️ SKILL_EXEC_BLOCKED") and f"Execution needs {needed}. " in ran, ran
        save_enabled(ctx.drive_root, "demo", False)
        toggled = skill_exec._handle_toggle_skill(ctx, skill="demo", enabled=True)
        assert '"error": "It turns on after a review passes."' in toggled and f'"detail": "{needed}"' in toggled, toggled
        granted = prepare_skill_grant(load_skill(directory, ctx.drive_root), ctx.drive_root, [])
        assert granted["error"] == f"key and permission grants need the review gate to permit the current bytes: {needed}"
        assert granted["review_gate"]["blocking_reason"] == "review_pending"
        assert "fresh executable review" not in ran + toggled + granted["error"]


def test_grant_prompt_does_not_ask_for_a_review_the_gate_already_permits(subject, monkeypatch):
    """#1735: the grant prompt is reached only past the executable gate; under a qualified
    Advisory author acceptance (reviewer still pending) it asks for the grant alone."""
    ctx, directory, _ = subject
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    manifest = directory / "SKILL.md"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace(
        "scripts:\n", "permissions: [inject_chat]\nscripts:\n"), encoding="utf-8")
    assert "error" not in _finish(subject, reference=_unavailable(subject))
    loaded = load_skill(directory, ctx.drive_root)
    gate = loaded.review.gate_for(loaded.content_hash)
    assert (loaded.review.status, gate["blocking_reason"]) == ("pending", "author_accepted_advisory")
    refused = skill_exec._handle_toggle_skill(ctx, skill="demo", enabled=True)
    assert "Grant it the keys and permissions" in refused, "the gate passed; grants are still required"
    save_enabled(ctx.drive_root, "demo", True)
    ran = skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py")
    assert ran.startswith("⚠️ SKILL_EXEC_GRANT_REQUIRED") and "inject_chat" in ran
    assert "fresh executable review" not in ran and "review gate already permits" in ran


@pytest.mark.parametrize("case", ["running", "no_reference", "wrong_reference", "no_job", "no_terminal_time"])
def test_pending_finish_needs_actual_terminal_source(subject, case):
    reference = None
    if case != "no_job":
        reference = _unavailable(subject, status="running" if case == "running" else "completed",
                                 finished=case != "no_terminal_time")
    if case == "no_reference":
        reference = None
    if case == "wrong_reference":
        reference = {**reference, "job_id": "another-job"}
    assert "error" in _finish(subject, reference=reference)
    ctx, directory, _ = subject
    assert not load_skill(directory, ctx.drive_root).review.author_disposition


def test_current_preflight_required_and_old_cyber_pending_is_not_authority(subject, monkeypatch):
    ctx, directory, _ = subject
    reference = _unavailable(subject)
    (directory / "scripts/hello.py").write_text("def broken(\n", encoding="utf-8")
    assert "deterministic preflight" in _finish(subject, reference=reference)["error"]
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    assert "error" not in _finish(subject)
    state = load_review_state(ctx.drive_root, "demo")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    gate = state.gate_for(compute_content_hash(directory))
    assert not gate["executable_review"] and gate["preflight_failed"]
    # Legacy Cyber records carry no current deterministic success proof.
    state.author_disposition.pop("review_reference")
    assert not state.gate_for(compute_content_hash(directory))["executable_review"]


def test_pending_author_acceptance_preserves_enable_grants_and_deps(subject, monkeypatch):
    from ouroboros import config
    from ouroboros.skill_loader import auto_grant_if_enabled, grant_status_for_skill
    from ouroboros.skill_readiness import skill_readiness_for_execution
    from ouroboros import skill_dependencies

    ctx, directory, _ = subject
    assert "error" not in _finish(subject, reference=_unavailable(subject))
    loaded = load_skill(directory, ctx.drive_root)
    loaded.manifest.env_from_settings = ["OPENAI_API_KEY"]
    monkeypatch.setattr(config, "get_auto_grant_enabled", lambda: False)
    assert not auto_grant_if_enabled(ctx.drive_root, loaded).granted
    assert not grant_status_for_skill(ctx.drive_root, loaded)["usable"]
    monkeypatch.setattr(config, "get_auto_grant_enabled", lambda: True)
    assert auto_grant_if_enabled(ctx.drive_root, loaded).granted
    assert grant_status_for_skill(ctx.drive_root, loaded)["usable"]
    readiness = skill_readiness_for_execution(ctx.drive_root, loaded, skills=[loaded])
    assert "skill_disabled" in readiness.blockers
    loaded.enabled = True
    assert skill_readiness_for_execution(ctx.drive_root, loaded, skills=[loaded]).ready
    monkeypatch.setattr(skill_dependencies, "auto_install_specs_for_skill", lambda *a: [{"kind": "pip", "package": "test-only"}])
    readiness = skill_readiness_for_execution(ctx.drive_root, loaded, skills=[loaded])
    assert any(item.startswith("deps_not_ready:") for item in readiness.blockers)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    assert not auto_grant_if_enabled(ctx.drive_root, loaded).granted


@pytest.mark.parametrize("basis", ["critic", "unavailable"])
def test_actual_publish_transaction_binds_author_bytes_and_preserves_provenance(subject, monkeypatch, tmp_path, basis):
    from ouroboros.skill_publish_snapshot import capture_skill_publish_snapshot
    from tests.test_skill_publish import _patch_validate
    from tests import test_skill_publish_transaction as transaction
    from tests.test_skill_publish_transaction import _install_transaction_fakes, _scan_result, _finding

    ctx, directory, _ = subject
    if basis == "critic":
        loaded, old_hash, findings = _critic_then_change(subject)
    else:
        assert "error" not in _finish(subject, reference=_unavailable(subject))
        loaded = load_skill(directory, ctx.drive_root)
        old_hash, findings = "", []
    loaded.source = "external"
    original_validate = skill_publish._validate_local_skill
    snapshot = capture_skill_publish_snapshot(loaded)
    monkeypatch.setattr(transaction, "SNAPSHOT_SHA", snapshot.content_hash)
    _unused_ctx, events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=snapshot)
    # Keep real local validation AND actual immutable byte capture on the mutation path.
    monkeypatch.setattr(skill_publish, "_validate_local_skill", original_validate)
    monkeypatch.setattr(skill_publish, "capture_skill_publish_snapshot", capture_skill_publish_snapshot)
    _patch_validate(monkeypatch, loaded)
    result = json.loads(skill_publish._submit_skill_to_hub(ctx, "demo", confirm_public_submission=True))
    assert result["ok"], json.dumps(result, indent=2)
    assert result["snapshot_hash"] == loaded.content_hash
    assert result["author_content_hash"] == loaded.content_hash
    assert result["review_status"] == loaded.review.status
    assert loaded.review.findings == findings
    body = captured["pr_body"]
    assert any(event[0] == "scan" for event in events)
    assert "Scanner status: not_run" not in body
    assert loaded.content_hash in body and "Author rationale:" in body
    assert "Fresh clean review verified" not in body and "immutable reviewed snapshot" not in body
    if basis == "critic":
        assert old_hash in body and "(critical): Retained critic finding." in body
        assert "Non-blocking FAIL" not in body
    else:
        assert "Independent review was unavailable" in body
    committed = {item["path"]: base64.b64decode(item["contents"]) for item in captured["additions"]}
    assert committed["skills/demo/scripts/hello.py"] == (directory / "scripts/hello.py").read_bytes()
    catalog = json.loads(committed["catalog.json"])
    assert not any("review" in key for key in catalog["skills"][0])
    assert any(row == ("mutation", "pr") for row in events)
    events.clear()
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    assert not json.loads(skill_publish._submit_skill_to_hub(ctx, "demo", confirm_public_submission=True))["ok"]
    assert not events
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    assert json.loads(skill_publish._submit_skill_to_hub(ctx, "demo"))["reason_code"] == "confirmation_required"
    monkeypatch.setattr(skill_publish, "scan_named_bytes", lambda *a, **kw: _scan_result(_finding("SKILL.md", confidence="high", disposition="blocker")))
    assert not json.loads(skill_publish._submit_skill_to_hub(ctx, "demo", confirm_public_submission=True))["ok"]
    assert not any(row[0] == "mutation" for row in events)


def test_selected_and_passive_preflight_follow_same_current_authority(subject, monkeypatch, tmp_path):
    from ouroboros.gateway.extensions import _passive_submit_hub
    from ouroboros.gateway import skill_publish as preflight
    from ouroboros.skill_publish_snapshot import capture_skill_publish_candidate
    from tests.test_skill_publish_preflight import _patch_domain, _scan, _build

    loaded, _old_hash, _findings = _critic_then_change(subject)
    passive = _passive_submit_hub(loaded, github_token_configured=True)
    assert "author-accepted" in passive["reason"] and not passive["publication_ready"]
    snapshot = capture_skill_publish_candidate(loaded)
    preflight._PREFLIGHT_SCAN_CACHE.clear()
    _patch_domain(monkeypatch, loaded, snapshot, _scan(), [])
    outcome = _build(tmp_path).payload
    assert outcome["publication_ready"] and outcome["state"] == "warnings"
    assert outcome["review"]["status"] == "blockers" and outcome["review"]["stale"]
    assert outcome["review"]["author_accepted"]
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    assert not _build(tmp_path).payload["publication_ready"]
    assert "fresh review" in _passive_submit_hub(loaded, github_token_configured=True)["reason"]


def test_completed_waiter_with_inflight_actor_is_not_unavailable(subject):
    """A terminal job whose projected record still names a seat in flight is a review still running."""
    reference = _unavailable(subject)
    ctx, _directory, _ = subject
    save_review_state(ctx.drive_root, "demo", SkillReviewState(
        status="pending", content_hash=reference["content_hash"],
        raw_actor_records=[{"slot_id": "critic", "operation_state": "in_flight", "late_result_pending": True}]))
    assert "error" in _finish(subject, reference=reference)


def test_owner_attestation_alone_is_not_public_authority(subject):
    from ouroboros.review_records import build_author_disposition
    from ouroboros.skill_publish_eligibility import publication_author_acceptance

    ctx, directory, _ = subject
    current_hash = compute_content_hash(directory)
    state = SkillReviewState(status="clean", content_hash=current_hash, review_profile="owner_attested",
        author_disposition=build_author_disposition(disposition="accepted", rationale="Legacy owner attestation.",
            subject_hash=current_hash, reviewer_signal="clean", enforcement="advisory"))
    assert not publication_author_acceptance(state, current_hash)
    marker = ctx.drive_root / "state/skills/demo/owner_attestation.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    save_review_state(ctx.drive_root, "demo", state)
    assert "error" in _finish(subject)
    assert "error" not in _finish(subject, reference=_unavailable(subject))
    state = load_review_state(ctx.drive_root, "demo")
    assert state.review_profile == "owner_attested"
    assert publication_author_acceptance(state, current_hash)


def test_host_service_preserves_token_enablement_and_permission_checks(subject, monkeypatch):
    from ouroboros.gateway.host_service import HostServiceContext, HostServiceAuthError
    from ouroboros.skill_loader import save_skill_grants
    from ouroboros import skill_review_runner
    from tests.test_host_service_api import _seed_token

    ctx, _directory, _ = subject
    _seed_token(ctx.drive_root, skill="demo", permissions=["inject_chat"])
    save_review_state(ctx.drive_root, "demo", SkillReviewState(status="pending"))
    monkeypatch.setattr(skill_review_runner, "_reconcile_extension_payload", lambda *a, **kw: {})
    assert "error" not in _finish(subject, reference=_unavailable(subject))
    host = HostServiceContext(ctx.drive_root)
    skill, token = host.authenticate_token_payload("token")
    assert skill == "demo"
    host.require_permission(skill, token, "inject_chat")
    with pytest.raises(HostServiceAuthError, match="lacks grant"):
        host.require_permission(skill, token, "presence")
    save_skill_grants(ctx.drive_root, "demo", [], content_hash=token["content_hash"], requested_keys=[], granted_permissions=[])
    with pytest.raises(HostServiceAuthError, match="lacks grant"):
        host.require_permission(skill, token, "inject_chat")
    save_enabled(ctx.drive_root, "demo", False)
    with pytest.raises(HostServiceAuthError, match="disabled"):
        host.authenticate_token_payload("token")
    save_enabled(ctx.drive_root, "demo", True)
    with pytest.raises(HostServiceAuthError, match="token is stale"):
        host._assert_active_token("demo", {**token, "content_hash": "old"})
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    with pytest.raises(HostServiceAuthError, match="executable review"):
        host.authenticate_token_payload("token")


def test_prior_substantive_feedback_still_allows_finish_with_another_actor_pending(subject):
    loaded, _old_hash, _findings = _critic_then_change(subject)
    _unavailable(subject, status="running", finished=False)
    assert "error" not in _finish(subject)
    assert loaded.review.gate_for(loaded.content_hash)["executable_review"]


def test_reserved_operation_without_terminal_actor_does_not_invent_unavailability(subject):
    """A seat still open in the terminal job's projected record (a reserved operation, a late result pending)
    is a review still running, not an unavailable one: the finish is refused until it settles."""
    reference = _unavailable(subject)
    ctx, _directory, _ = subject
    save_review_state(ctx.drive_root, "demo", SkillReviewState(
        status="pending", content_hash=reference["content_hash"],
        raw_actor_records=[{"slot_id": "critic", "operation_state": "in_flight", "late_result_pending": True}]))
    assert "A review still running is not unavailable" in _finish(subject, reference=reference)["error"]


def test_author_acceptance_does_not_follow_subsequent_payload_changes(subject, monkeypatch, tmp_path):
    from ouroboros.skill_publish_snapshot import (
        SkillPublishSnapshotError, capture_skill_publish_snapshot, capture_skill_publish_candidate,
    )
    from tests.test_skill_publish_preflight import _patch_domain, _scan, _build

    loaded, _old_hash, _findings = _critic_then_change(subject)
    _ctx, directory, _ = subject
    (directory / "scripts/hello.py").write_text("print('another revision')\n", encoding="utf-8")
    with pytest.raises(SkillPublishSnapshotError, match="snapshot_review_stale"):
        capture_skill_publish_snapshot(loaded)
    # The selected preflight binds captured bytes, not the loader's earlier hash.
    _patch_domain(monkeypatch, loaded, capture_skill_publish_candidate(loaded), _scan(), [])
    assert not _build(tmp_path).payload["publication_ready"]
