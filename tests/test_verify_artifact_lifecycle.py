"""C: verify_and_record after-only artifact-lifecycle FLAG (a check that built then deleted a
declared deliverable). Flag-only: status stays pass; the structural fact reaches the ledger and
the advisory acceptance reviewer."""
from __future__ import annotations

from ouroboros.tools.registry import ToolContext
from ouroboros.tools.verify import _probe_artifact_lifecycle


def _ctx(tmp_path):
    work = tmp_path / "ws"
    work.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    return ToolContext(repo_dir=work, drive_root=drive, task_id="t"), work


def test_probe_flags_deleted_artifact_host(tmp_path):
    ctx, work = _ctx(tmp_path)
    (work / "kept.txt").write_text("x")  # present after the check
    # "out.so" was built then deleted by the check -> absent now
    lifecycle, missing = _probe_artifact_lifecycle(ctx, ["kept.txt", "out.so"], work, use_executor=False)
    by = {e["path"]: e for e in lifecycle}
    assert by["kept.txt"]["exists_after"] is True
    assert by["out.so"]["exists_after"] is False
    assert missing == ["out.so"]
    assert all(e["check_surface"] == "host" for e in lifecycle)


def test_probe_traversal_is_unavailable_not_probed(tmp_path):
    ctx, work = _ctx(tmp_path)
    lifecycle, missing = _probe_artifact_lifecycle(ctx, ["../../etc/passwd"], work, use_executor=False)
    assert lifecycle and lifecycle[0]["check_surface"] == "unavailable"
    assert lifecycle[0]["exists_after"] is None
    assert missing == []  # refused path is never reported as "missing" (no arbitrary host probe)


# --- v6.57.0 (1.2): refused_out_of_scope + observation on read-only roots -------


def test_observe_refused_out_of_scope_is_not_failure(tmp_path):
    """An artifact_observation on a path outside the observable roots is a POLICY
    refusal (refused_out_of_scope), NOT a fail — it must not raise has_failures."""
    from ouroboros.outcomes import build_verification_ledger
    from ouroboros.tools.verify import _observe_artifacts

    ctx, _work = _ctx(tmp_path)
    status, detail = _observe_artifacts(ctx, ["/etc/shadow"])
    assert status == "refused_out_of_scope"
    assert "refused" in detail

    led = build_verification_ledger(
        task={"id": "t", "task_contract": {}},
        loop_outcome={"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}},
        llm_trace={"tool_calls": [], "verification_receipts": [{
            "status": "refused_out_of_scope", "contract_kind": "artifact_observation",
            "paths": ["/etc/shadow"], "summary": "path refused",
        }]},
        artifact_bundle={},
    )
    assert led["summary"]["has_failures"] is False


def test_observe_allows_deliverables_root(tmp_path, monkeypatch):
    """A parent CAN observe a child's deliverable under the read-only deliverables
    root (existence only) — an artifact_observation there is OBSERVED, not refused."""
    from ouroboros.tools.verify import _observe_artifacts

    deliverables = tmp_path / "Deliverables"
    deliverables.mkdir()
    (deliverables / "site.html").write_text("<html></html>")
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))

    ctx, _work = _ctx(tmp_path)
    status, detail = _observe_artifacts(ctx, [str(deliverables / "site.html")])
    assert status == "observed"
    assert "1 artifact" in detail

    missing_status, _ = _observe_artifacts(ctx, [str(deliverables / "nope.html")])
    assert missing_status == "fail"  # inside an observable root but genuinely missing


def test_ledger_carries_artifact_lifecycle_flag_only():
    from ouroboros.outcomes import build_verification_ledger

    led = build_verification_ledger(
        task={"id": "t", "task_contract": {}},
        loop_outcome={"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}},
        llm_trace={"tool_calls": [], "verification_receipts": [{
            "status": "pass", "contract_kind": "explicit_command", "check": "build.sh",
            "artifact_lifecycle": [{"path": "out.so", "exists_after": False, "check_surface": "host"}],
            "artifacts_missing_after": ["out.so"],
        }]},
        artifact_bundle={},
    )
    entry = next(e for e in led["entries"] if e.get("kind") == "verification_receipt")
    assert entry["status"] == "pass"  # FLAG-ONLY: a missing artifact does NOT flip the status
    assert entry["artifacts_missing_after"] == ["out.so"]
    assert entry["artifact_lifecycle"][0]["exists_after"] is False


def test_ledger_preserves_verification_receipt_criterion_id():
    from ouroboros.outcomes import build_verification_ledger

    led = build_verification_ledger(
        task={"id": "t", "task_contract": {}},
        loop_outcome={"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}},
        llm_trace={"tool_calls": [], "verification_receipts": [{
            "status": "pass",
            "contract_kind": "explicit_command",
            "criterion_id": "claim_1",
        }]},
        artifact_bundle={},
    )
    entry = next(e for e in led["entries"] if e.get("kind") == "verification_receipt")
    assert entry["criterion_id"] == "claim_1"


def test_acceptance_summary_surfaces_and_redacts_missing_after():
    from ouroboros.review_evidence import _accept_verification_summary

    summary = _accept_verification_summary([
        {"status": "pass", "check": "a", "artifacts_missing_after": ["/work/out.so"]},
        {"status": "pass", "check": "b"},
    ])
    assert summary["artifacts_missing_after_any"] is True
    assert any("out.so" in p for p in summary["artifacts_missing_after"])
    assert summary["count"] == 2


def test_acceptance_evidence_links_claims_to_host_receipts(tmp_path):
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.review_evidence import build_task_acceptance_evidence

    ctx, _work = _ctx(tmp_path)
    ctx.task_contract = {
        "acceptance_claims": [{
            "id": "answer",
            "claim": "final answer is exact",
            "surface": "FINAL ANSWER",
            "support": "exact host check",
            "priority": "must",
        }]
    }
    append_verification_receipt(tmp_path / "drive", "t", {
        "tool": "verify_and_record",
        "criterion_id": "answer",
        "contract_kind": "explicit_command",
        "status": "pass",
        "matched": True,
        "ts": "2026-01-01T00:00:00+00:00",
    })

    evidence = build_task_acceptance_evidence(ctx, drive_root=tmp_path / "drive", task_id="t")
    refs = evidence["acceptance_support_refs"]
    assert refs[0]["criterion_id"] == "answer"
    assert refs[0]["support_status"] == "supported"
    assert refs[0]["support_refs"][0]["provenance"] == "host_attested"


def test_acceptance_support_refs_use_global_receipt_index(tmp_path):
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.review_evidence import build_task_acceptance_evidence

    ctx, _work = _ctx(tmp_path)
    ctx.task_contract = {"acceptance_claims": [{"id": "second", "claim": "second claim"}]}
    append_verification_receipt(tmp_path / "drive", "t", {"status": "pass", "criterion_id": "first"})
    append_verification_receipt(tmp_path / "drive", "t", {"status": "pass", "criterion_id": "second"})

    evidence = build_task_acceptance_evidence(ctx, drive_root=tmp_path / "drive", task_id="t")
    assert evidence["acceptance_support_refs"][0]["support_refs"][0]["ref"] == "verification_receipts[1]"


def test_acceptance_support_refs_do_not_mark_failed_receipt_supported(tmp_path):
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.review_evidence import build_task_acceptance_evidence

    ctx, _work = _ctx(tmp_path)
    ctx.task_contract = {"acceptance_claims": [{"id": "answer", "claim": "answer exact"}]}
    append_verification_receipt(tmp_path / "drive", "t", {
        "status": "fail",
        "criterion_id": "answer",
        "matched": False,
        "contract_kind": "explicit_command",
    })

    evidence = build_task_acceptance_evidence(ctx, drive_root=tmp_path / "drive", task_id="t")
    row = evidence["acceptance_support_refs"][0]
    assert row["support_refs"]
    assert row["support_status"] == "linked_failed"


def test_acceptance_support_refs_include_artifact_lifecycle(tmp_path):
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.review_evidence import build_task_acceptance_evidence

    ctx, _work = _ctx(tmp_path)
    ctx.task_contract = {"acceptance_claims": [{"id": "wheel", "claim": "wheel remains present"}]}
    append_verification_receipt(tmp_path / "drive", "t", {
        "status": "pass",
        "criterion_id": "wheel",
        "matched": True,
        "contract_kind": "explicit_command",
        "artifact_lifecycle": [{"path": "dist/pkg.whl", "exists_after": True, "check_surface": "host"}],
        "artifacts_missing_after": [],
    })

    evidence = build_task_acceptance_evidence(ctx, drive_root=tmp_path / "drive", task_id="t")
    ref = evidence["acceptance_support_refs"][0]["support_refs"][0]
    assert ref["artifact_lifecycle"][0]["path"] == "dist/pkg.whl"
    assert ref["artifact_lifecycle"][0]["exists_after"] is True


def test_acceptance_support_refs_do_not_mark_declared_receipt_supported(tmp_path):
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.review_evidence import build_task_acceptance_evidence

    ctx, _work = _ctx(tmp_path)
    ctx.task_contract = {"acceptance_claims": [{"id": "risk", "claim": "manual risk accepted"}]}
    append_verification_receipt(tmp_path / "drive", "t", {
        "status": "declared",
        "criterion_id": "risk",
        "contract_kind": "no_visible_machine_contract",
    })

    evidence = build_task_acceptance_evidence(ctx, drive_root=tmp_path / "drive", task_id="t")
    assert evidence["acceptance_support_refs"][0]["support_status"] == "declared_only"


def test_acceptance_support_refs_disclose_bounded_claim_text(tmp_path):
    from ouroboros.review_evidence import _accept_claim_support_refs

    contract = {"acceptance_claims": [{"id": "long", "claim": "x" * 500, "support": "y" * 600}]}
    refs = _accept_claim_support_refs(contract, [])
    assert "OMISSION NOTE" in refs[0]["claim"]
    assert "OMISSION NOTE" in refs[0]["support_expected"]
