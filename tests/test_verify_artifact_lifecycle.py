"""C: verify_and_record after-only artifact-lifecycle FLAG (a check that built then deleted a
declared deliverable). Flag-only: status stays pass; the structural fact reaches the ledger and
the verification summary a result review reads."""
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


def test_probe_executor_branch_records_present_artifact_as_present(tmp_path, monkeypatch):
    """capinv-447 D7: the executor probe coerced returncode with `or 1`, so rc=0
    (artifact EXISTS in-container) still read as 1 — every declared artifact on a
    healthy executor run was recorded exists_after=false and pushed into
    artifacts_missing_after, reaching the durable receipt and the acceptance
    reviewer. rc=0 must mean present; rc!=0 must still mean missing."""
    import ouroboros.workspace_executor as workspace_executor

    class _Res:
        def __init__(self, rc):
            self.returncode = rc

    def fake_execute(_ctx, cmd, _cwd, _timeout, **_kwargs):
        # `test -e "$1"` receives the probed path as the last argv element.
        return _Res(0 if cmd[-1] == "kept.txt" else 1)

    monkeypatch.setattr(workspace_executor, "execute", fake_execute)
    ctx, work = _ctx(tmp_path)
    lifecycle, missing = _probe_artifact_lifecycle(ctx, ["kept.txt", "out.so"], work, use_executor=True)
    by = {e["path"]: e for e in lifecycle}
    assert by["kept.txt"]["exists_after"] is True  # was inverted to False before the fix
    assert by["out.so"]["exists_after"] is False
    assert missing == ["out.so"]
    assert all(e["check_surface"] == "executor" for e in lifecycle)


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


def test_acceptance_summary_surfaces_missing_after():
    from ouroboros.review_evidence import _accept_verification_summary

    summary = _accept_verification_summary([
        {"status": "pass", "check": "a", "artifacts_missing_after": ["/work/out.so"]},
        {"status": "pass", "check": "b"},
    ])
    assert summary["artifacts_missing_after_any"] is True
    assert any("out.so" in p for p in summary["artifacts_missing_after"])
    assert summary["count"] == 2


# --- submarine unwind (2026-08-08): observable roots derive from the _POLICY matrix

def test_observe_artifacts_accepts_artifact_store_path(tmp_path):
    """wave3 r24: the agent registered a screenshot into artifact_store and
    verify_and_record REFUSED the very same path (private root list). The
    observable roots now derive from the Tool API matrix — anything the profile
    can already read_file it can also existence-observe."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tools.verify import _observe_artifacts

    ctx, _work = _ctx(tmp_path)
    art = task_artifact_dir_path(ctx.drive_root, "t", create=True)
    shot = art / "shot.png"
    shot.write_bytes(b"png-bytes")

    status, detail = _observe_artifacts(ctx, [str(shot)])
    assert status == "observed", detail


def test_observe_artifacts_still_refuses_control_plane(tmp_path):
    """The anti-cheat boundary is unchanged: runtime_data/system-repo state files
    outside the sanctioned observation roots stay refused."""
    from ouroboros.tools.verify import _observe_artifacts

    ctx, _work = _ctx(tmp_path)
    secret = ctx.drive_root / "state" / "state.json"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("{}", encoding="utf-8")

    status, detail = _observe_artifacts(ctx, [str(secret)])
    assert status == "refused_out_of_scope", (status, detail)
