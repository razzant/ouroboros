"""Configured pool rights on the combined change/coupling brief consumer."""
from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.review_checking import frozen_checking_source
from ouroboros.reviewer_slot_config import commit_triad_delivery, review_pool_slots
from ouroboros.tools.review_brief_coupling import BriefInputs, build_retrieving_brief
from ouroboros.tools.review_multi_model import _multi_model_review_async
from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject
from ouroboros.triad_review import parse_seat_answers
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_review_session_rights import (  # noqa: F401 - shared offline fixtures
    FakeGateway, _owned_gateway_uses_each_test_transport, fake_route, repo, writable_engine,
)


@pytest.mark.parametrize("access", ["readonly", "workspace_write", "full"])
def test_combined_pool_brief_preserves_rights_profile_and_frozen_subject(
    monkeypatch, repo, tmp_path, writable_engine, access,  # noqa: F811
):
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_admission import seat_vectors
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    from tests._review_session_route_shared import _terminal_detail

    drive = tmp_path / "drive"
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    ctx.task_id = "combined-pool-rights"
    set_review_pool(monkeypatch, pool_roster(pool_seat(
        "combined", "fake-review=fake-small", kind="agent_session", access=access,
        profile_id="captured-profile", effort="high")))
    [slot] = review_pool_slots()
    frozen = freeze_subject(ctx, ReviewSubjectSpec(
        root_kind="active_workspace", root=str(repo), kind="index", governance_root=str(repo)))
    brief, manifest = build_retrieving_brief(repo, BriefInputs(
        commit_message="Check the same staged subject", delegated=True, model=slot.model,
        slot_id=slot.slot_id, session_profile=slot.session_profile, drive_root=drive,
        task_id=ctx.task_id, diff_text=frozen.diff_text))
    plan = seat_vectors(commit_triad_delivery())
    plan["session_tasks"] = [brief]
    answer = {"change": [], "change_clean": True, "coupling": [
        {"item": item, "verdict": "PASS", "severity": "advisory", "reason": "Verified the frozen input and its consumers."}
        for item in sorted(SCOPE_REQUIRED_ITEMS)
    ]}
    FakeGateway.detail = _terminal_detail(json.dumps(answer), conformance="passed")
    result = asyncio.run(_multi_model_review_async(
        "Review the original staged subject", "PACKET_MUST_NOT_REACH_SESSION", plan["models"], ctx,
        routes=plan["routes"], row_plan=plan, session_root=str(repo),
        checking_source=frozen_checking_source(frozen, data_root=drive) if access != "readonly" else {}))
    parsed = parse_seat_answers(result, dict(zip(plan["slot_ids"], plan["parts"])))
    [actor] = parsed.actor_records
    assert actor.status == "responded", result
    assert actor.answers["change"]["verdict"] == actor.answers["coupling"]["verdict"] == "PASS"
    [gateway] = FakeGateway.instances
    [body] = gateway.start_requests
    assert body["access"] == access
    assert body["credentialProfileId"] == "captured-profile" and body["effort"] == "high"
    assert brief in body["prompt"] and "Coupling questions" in body["prompt"]
    assert "PACKET_MUST_NOT_REACH_SESSION" not in body["prompt"]
    assert body["outputSchema"]["properties"]["coupling"]["minItems"] == 1
    assert manifest["parts"] == ["change", "coupling"]
    record = custody.invocation_record(drive, gateway.start_keys[0])
    assert record["request"] == body
    if access == "readonly":
        assert body["mode"] == "ask" and "execution" not in body
    else:
        assert body["mode"] == "agent" and record["authority_source"] == "investigation"
        from ouroboros.observability import read_call_payload

        response = read_call_payload(drive, task_id=ctx.task_id,
                                     call_id=result["results"][0]["operation_id"] + "_response")[1]
        evidence = response["usage"]["checking_evidence"]
        assert evidence["source"]["tree"] == frozen.tree_sha
        assert evidence["status"] == "retained" and Path(evidence["evidence_dir"], "summary.json").is_file()
    assert (repo / "f.txt").read_text() == "A staged\nB unstaged\n"


def test_historical_readonly_identity_survives_explicit_catalog_lowering(monkeypatch):
    from ouroboros.review_custody import _attempt_key
    from ouroboros.review_dispatch import review_reconciliation_identity
    from ouroboros.review_records import ReviewRequest

    set_review_pool(monkeypatch, pool_roster(pool_seat(
        "session", "fake-review=fake-small", kind="agent_session", access="readonly")))
    [slot] = review_pool_slots()
    request = ReviewRequest(surface="skill_review", goal="original goal", task_id="original-task", retry_key="original-wave")
    historical = dataclasses.replace(slot, session_access="")
    assert _attempt_key(request, slot) == _attempt_key(request, historical)
    assert review_reconciliation_identity(request, [slot], root_task_id="original-task") == (
        review_reconciliation_identity(request, [historical], root_task_id="original-task"))


def test_commit_contract_identity_carries_writable_seat_rights(monkeypatch):
    from ouroboros import reviewer_slot_config
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    def fingerprint(access):
        set_review_pool(monkeypatch, pool_roster(
            pool_seat("session", "fake-review=fake-small", kind="agent_session", access=access),
            pool_seat("packet", "m/packet", delivery="packet")))
        return commit_review_contract_fingerprint()

    writable = {access: fingerprint(access) for access in ("workspace_write", "full")}
    readonly = fingerprint("readonly")
    assert readonly and len({readonly, *writable.values()}) == 3
    assert fingerprint("full") == writable["full"]  # unchanged rights keep their identity
    # A row plan from before captured rights has no access vector: same exact bytes.
    plan = {key: value for key, value in commit_triad_delivery().items() if key != "session_accesses"}
    monkeypatch.setattr(reviewer_slot_config, "commit_triad_delivery", lambda: plan)
    assert commit_review_contract_fingerprint() == readonly
