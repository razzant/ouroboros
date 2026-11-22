"""C: verify_and_record after-only artifact-lifecycle FLAG (a check that built then deleted a
declared deliverable). Flag-only: status stays pass; the structural fact reaches the ledger and
the verification summary a result review reads."""
from __future__ import annotations

import sys

import pytest

from ouroboros.tools.registry import ToolContext
from ouroboros.tools.verify import _probe_artifact_lifecycle


def _ctx(tmp_path):
    work = tmp_path / "ws"
    work.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    return ToolContext(repo_dir=work, drive_root=drive, task_id="t"), work


def test_observation_reconciles_on_all_paths_before_the_review_display_limit(tmp_path):
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.review_evidence import _accept_verification_summary
    from ouroboros.tools.verify import _verify_and_record

    ctx, work = _ctx(tmp_path)
    prefix = [f"part-{index:02}.txt" for index in range(20)]
    paths = [*prefix, "tail.txt"]
    for name in [*prefix, "sibling.txt"]:
        (work / name).touch()

    def observe(names):
        return _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=names)

    assert "FAIL" in observe(paths)
    assert "OBSERVED" in observe(prefix)
    assert "OBSERVED" in observe([*prefix, "sibling.txt"])
    receipts = read_verification_receipts(ctx.drive_root, ctx.task_id)
    assert receipts[0]["paths"] == paths
    summary = _accept_verification_summary(receipts)
    assert summary["unreconciled_red_count"] == 1
    red, sibling = summary["unreconciled_red_identity"], summary["latest_identity"]
    assert red["paths"] == sibling["paths"] == prefix
    assert red["paths_omitted"] == sibling["paths_omitted"] == 1
    assert red["paths_identity_sha256"] != sibling["paths_identity_sha256"]

    (work / "tail.txt").touch()
    assert "OBSERVED" in observe([*reversed(paths), prefix[0]])
    summary = _accept_verification_summary(read_verification_receipts(ctx.drive_root, ctx.task_id))
    assert summary["unreconciled_red_count"] == 0
    assert summary["latest_identity"]["paths_identity_sha256"] == red["paths_identity_sha256"]


def test_observation_prose_delivers_byte_names_and_the_receipt_keeps_them_exact(tmp_path):
    """An artifact name whose POSIX bytes are not UTF-8 enters this call's result text as an exact JSON string
    literal, so the text survives every UTF-8 boundary, while the receipt keeps the declared name itself."""
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _observe_artifacts, _verify_and_record

    ctx, work = _ctx(tmp_path)
    byte_name = "report_\udcff.txt"  # POSIX `surrogateescape` of the name byte FF, spelled out for every platform
    created = sys.platform == "linux"  # the one platform relied on to store such a name; elsewhere it is missing
    if created:
        (work / byte_name).touch()

    text = _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=[byte_name])
    text.encode("utf-8")
    assert ("OBSERVED" if created else "FAIL") in text and '"report_\\udcff.txt"' in text
    [receipt] = read_verification_receipts(ctx.drive_root, ctx.task_id)
    assert receipt["paths"] == [byte_name] and receipt["status"] == ("observed" if created else "fail")

    # Ordinary Unicode and a typed backslash keep their own spelling, unquoted.
    for name in ("отчёт λ.md", "literal \\udcff.md"):
        assert f"missing: {name} (resolved as " in _observe_artifacts(ctx, [name])[1]
    # A policy refusal is prose too; this control-plane path is absolute on every platform.
    status, detail = _observe_artifacts(ctx, [str(ctx.drive_root / "state" / byte_name)])
    detail.encode("utf-8")
    assert status == "refused_out_of_scope" and "report_\\udcff.txt" in detail


@pytest.mark.parametrize("spaced", [
    " report-雪.txt",
    pytest.param("report-雪.txt ", marks=pytest.mark.skipif(sys.platform == "win32", reason="no trailing-space names")),
], ids=["leading-space", "trailing-space"])
def test_observation_looks_up_and_records_the_exact_spelling(tmp_path, spaced):
    """An edge space is a filename byte: lookup, prose and the persisted receipt all attest the exact
    declared spelling, whitespace included. No trimmed sibling ever stands in for the name asked for."""
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    ctx, work = _ctx(tmp_path)
    plain = spaced.strip()

    def observe(name):
        text = _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=[name])
        receipt = read_verification_receipts(ctx.drive_root, ctx.task_id)[-1]
        return text, receipt["status"], receipt["paths"]

    # Neither spelling exists: the missing path is the one that was asked for.
    assert observe(spaced)[1:] == ("fail", [spaced])

    # Only the trimmed sibling exists: it satisfies nothing. The receipt keeps naming the requested
    # spelling — the attestation is exactly what was asked for, edge whitespace included.
    (work / plain).touch()
    text, status, paths = observe(spaced)
    assert (status, paths) == ("fail", [spaced]) and f"missing: {spaced} (resolved as " in text
    assert observe(plain)[1:] == ("observed", [plain])

    # Both siblings exist: each request selects its own file, never the other's.
    (work / spaced).touch()
    assert observe(spaced)[1:] == ("observed", [spaced])
    assert observe(plain)[1:] == ("observed", [plain])
    (work / plain).unlink()  # only the padded name remains: the plain request fails, it does not fall back
    assert observe(spaced)[1:] == ("observed", [spaced])
    text, status, paths = observe(plain)
    assert (status, paths) == ("fail", [plain]) and f"missing: {plain} (resolved as " in text


@pytest.mark.skipif(sys.platform == "win32", reason="no spaces-only names")
def test_observation_keeps_a_spaces_only_name_as_one_exact_name(tmp_path):
    """Spaces are filename bytes: a file named by them is observed and recorded exactly beside an ordinary file, in the
    receipt and its reviewer identity. Absent, that exact name is the missing one; the empty string alone names nothing."""
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.review_evidence import _accept_verification_summary
    from ouroboros.tools.verify import _verify_and_record

    ctx, work = _ctx(tmp_path)
    blank = "   "
    (work / blank).touch()
    (work / "plain.txt").touch()

    def observe(names):
        text = _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=names)
        return text, read_verification_receipts(ctx.drive_root, ctx.task_id)

    text, receipts = observe([blank, "plain.txt", ""])
    assert (receipts[-1]["status"], receipts[-1]["paths"]) == ("observed", [blank, "plain.txt"])
    assert "observed 2 artifact(s)" in text
    assert _accept_verification_summary(receipts)["latest_identity"]["paths"] == [blank, "plain.txt"]

    (work / blank).unlink()
    text, receipts = observe([blank, "plain.txt"])
    assert (receipts[-1]["status"], receipts[-1]["paths"]) == ("fail", [blank, "plain.txt"])
    assert f"missing: {blank} (resolved as " in text

    assert observe([""])[1][-1]["paths"] == []


@pytest.mark.skipif(sys.platform == "win32", reason="a leading space before a drive letter is no absolute path")
def test_padded_spelling_never_reaches_a_file_outside_the_observable_roots(tmp_path):
    """The leading space belongs to the name: the padded spelling resolves as its own in-workspace path
    and nothing else stands in for it — no existence oracle through a leading space."""
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    ctx, _work = _ctx(tmp_path)
    secret = ctx.drive_root / "state" / "state.json"
    secret.parent.mkdir(parents=True)
    padded = f" {secret}"

    for present in (False, True):  # the same answer whether or not the refused file exists
        if present:
            secret.write_text("{}", encoding="utf-8")
        assert "OBSERVED" not in _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=[padded])
        receipt = read_verification_receipts(ctx.drive_root, ctx.task_id)[-1]
        assert (receipt["status"], receipt["paths"]) == ("fail", [padded])


def test_bytes_equal_operands_are_the_exact_declared_names(tmp_path, monkeypatch):
    """bytes_equal gets the two declared operands exactly, on both surfaces: a name's leading space is
    part of it, so a trimmed sibling is never the file that gets compared (the receipt attests the name
    that was asked for)."""
    import ouroboros.workspace_executor as workspace_executor
    from ouroboros.tools.verify import _compare_files_bytes_equal

    ctx, work = _ctx(tmp_path)
    a_name, b_name = " golden.bin", " mirror.bin"
    (work / a_name).write_bytes(b"payload")
    (work / b_name).write_bytes(b"payload")

    equal, detail = _compare_files_bytes_equal(ctx, [a_name, b_name], work, use_executor=False)
    assert equal is True and a_name in detail and b_name in detail

    seen: list = []

    class _Res:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    def fake_execute(_ctx, cmd, _cwd, _timeout, **_kwargs):
        seen.append(cmd)
        return _Res(0 if cmd[-2:] == [a_name, b_name] else 2)

    monkeypatch.setattr(workspace_executor, "execute", fake_execute)
    equal2, detail2 = _compare_files_bytes_equal(ctx, [a_name, b_name], work, use_executor=True)
    assert seen == [["cmp", "--", a_name, b_name]] and equal2 is True and "(executor cmp)" in detail2


@pytest.mark.skipif(sys.platform == "win32", reason="no spaces-only names")
def test_bytes_equal_receipt_uses_both_exact_names(tmp_path):
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    ctx, work = _ctx(tmp_path)
    names = ["   ", " mirror-雪.bin "]
    for name in names:
        (work / name).write_bytes(b"payload")

    text = _verify_and_record(ctx, contract_kind="explicit_command",
                              check=[sys.executable, "-c", "print('ready')"],
                              expected_match="bytes_equal", artifact_paths=[*names, ""])
    assert "PASS" in text and f"{names[0]} == {names[1]} (7 bytes)" in text
    text.encode("utf-8")
    [receipt] = read_verification_receipts(ctx.drive_root, ctx.task_id)
    assert (receipt["status"], receipt["matched"]) == ("pass", True)
    assert receipt["artifact_lifecycle"] == [
        {"path": name, "exists_after": True, "check_surface": "host"} for name in names]
    assert "artifacts_missing_after" not in receipt


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


def test_verify_description_names_the_task_and_the_human_not_a_benchmark():
    """Audit 08.11 item 11: the schema every task of every install sees names what a receipt rests on and
    no benchmark's files; the hidden-grader boundary stays a mechanism (the probe refusal above) and the
    contract rule keeps its home in prompts/SYSTEM.md."""
    from ouroboros.tools.verify import get_tools

    description = next(entry.schema for entry in get_tools() if entry.name == "verify_and_record")["description"]
    assert "the task's own statement and materials, your human's words" in description
    for benchmark_word in ("solution.sh", "hidden /tests/", "look up the answer online"):
        assert benchmark_word not in description, benchmark_word
