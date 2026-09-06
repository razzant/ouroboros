"""v6.47.0 mega-commit immune tests (TIER-5): verify-before-done (FR3), cooperative
subagents (FR2), workspace-aware code-intel (R1/R2/R5), skill-publish SSOT (FR1),
and the M2/M6 reliability invariants. Pure-logic where possible; a few use a tmp
git tree / user_files root."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
from tests._typed_guard_shared import _shell_guard_text

PY = sys.executable or "python3"  # portable interpreter for cross-platform check commands


# ── R2: PYTHONPATH repo-scrub env isolation ───────────────────────────────────
def test_scrub_repo_from_pythonpath_drops_only_repo_entry():
    from ouroboros.platform_layer import scrub_repo_from_pythonpath

    repo = "/obo/repo"
    sep = os.pathsep
    env = {"PYTHONPATH": sep.join([repo, "/app", "/usr/lib/py"]), "X": "1"}
    out = scrub_repo_from_pythonpath(env, repo)
    assert out["PYTHONPATH"] == sep.join(["/app", "/usr/lib/py"])
    assert out["X"] == "1"
    assert env["PYTHONPATH"].startswith(repo)  # original not mutated
    # only the repo entry -> PYTHONPATH removed entirely
    assert "PYTHONPATH" not in scrub_repo_from_pythonpath({"PYTHONPATH": repo}, repo)
    # no PYTHONPATH / no repo_dir -> no-op
    assert scrub_repo_from_pythonpath({"A": "b"}, repo) == {"A": "b"}
    assert scrub_repo_from_pythonpath({"PYTHONPATH": repo}, None) == {"PYTHONPATH": repo}
    # trailing-slash equivalence
    assert "PYTHONPATH" not in scrub_repo_from_pythonpath({"PYTHONPATH": repo + "/"}, repo)


def test_shell_env_for_cwd_scrubs_external_keeps_repo():
    from ouroboros.tools.shell import _shell_env_for_cwd

    repo = Path(tempfile.mkdtemp())
    (repo / "sub").mkdir()
    ext = Path(tempfile.mkdtemp())
    ctx = types.SimpleNamespace(repo_dir=str(repo))
    # a command inside the repo inherits os.environ (None -> no scrub)
    assert _shell_env_for_cwd(ctx, repo / "sub") is None
    # a command outside the repo gets a scrubbed env (dict, not None)
    env = _shell_env_for_cwd(ctx, ext)
    assert isinstance(env, dict)


# ── R5: effect-based artifact-audit gate ──────────────────────────────────────
def test_user_files_run_effect_gate():
    from ouroboros.tools.shell import _shallow_listing, _user_files_run_had_effect

    d = Path(tempfile.mkdtemp())
    (d / "a.txt").write_text("1")
    sig = _shallow_listing(d)
    assert _user_files_run_had_effect([], [], sig, d) is False  # read-only
    (d / "b.txt").write_text("2")
    assert _user_files_run_had_effect([], [], sig, d) is True   # new file
    assert _user_files_run_had_effect(["x"], ["x", "y"], None, d) is True  # git delta
    assert _user_files_run_had_effect(["x"], ["x"], None, d) is False


# ── R1: query_code root=user_files guards ─────────────────────────────────────
def test_query_code_user_files_empty_path_hard_error():
    from ouroboros.tools.query_code import _query_code

    ctx = types.SimpleNamespace(
        drive_root=tempfile.mkdtemp(), repo_dir=tempfile.mkdtemp(),
        workspace_root="", workspace_mode="", task_constraint=None,
    )
    out = _query_code(ctx, "symbols", root="user_files", path="")
    assert "requires an explicit path" in out


def test_query_code_user_files_blocked_for_subagent():
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.query_code import _query_code

    ctx = types.SimpleNamespace(
        drive_root=tempfile.mkdtemp(), repo_dir=tempfile.mkdtemp(),
        workspace_root="", workspace_mode="", task_constraint=TaskConstraint(mode="local_readonly_subagent"),
    )
    out = _query_code(ctx, "symbols", root="user_files", path="/whatever")
    assert "TOOL_ACCESS_BLOCKED" in out
    assert "profile=local_readonly_subagent cannot search root=user_files" in out


def test_query_code_structural_walk_is_bounded_and_symlink_safe():
    from ouroboros.tools.query_code import _walk_candidate_files

    d = Path(tempfile.mkdtemp())
    (d / "a.py").write_text("x=1")
    (d / "sub").mkdir()
    (d / "sub" / "b.py").write_text("y=2")
    outside = Path(tempfile.mkdtemp())
    (outside / "secret.py").write_text("S=1")
    try:
        os.symlink(outside, d / "escape")
    except OSError:
        pass
    files, note = _walk_candidate_files(d, d)
    names = {f.name for f in files}
    assert "a.py" in names and "b.py" in names
    assert "secret.py" not in names  # symlink escaping the root is dropped


# ── FR3: receipt store, grounding, flag, nudge ────────────────────────────────
def test_receipt_store_roundtrip_and_task_id_guard():
    from ouroboros import outcomes as O

    dr = tempfile.mkdtemp()
    O.append_verification_receipt(dr, "task-1", {"status": "pass", "check": "pytest"})
    rs = O.read_verification_receipts(dr, "task-1")
    assert rs and rs[0]["status"] == "pass"
    # an invalid task id must not escape the artifacts dir
    with pytest.raises(Exception):
        O.verification_receipts_path(dr, "../escape", create=True)


def test_merge_objective_warning_coexist():
    from ouroboros.outcomes import _merge_objective_warning

    obj = {"status": "not_evaluated"}
    _merge_objective_warning(obj, "residual_tool_errors_without_review")
    _merge_objective_warning(obj, "receipt_absent")
    assert obj["warning"] == "residual_tool_errors_without_review"  # primary unchanged
    assert obj["warnings"] == ["residual_tool_errors_without_review", "receipt_absent"]


def test_receipt_absent_flag_and_suppression():
    from ouroboros import outcomes as O

    def lo():
        return {"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}, "final_answer": ""}

    # effects + no grounding -> receipt_absent
    a = lo()
    O.apply_receipt_absent_flag(a, {"tool_calls": [{"tool": "commit_reviewed", "status": "ok"}]}, tempfile.mkdtemp(), "t1")
    assert a["outcome_axes"]["objective"].get("warning") == "receipt_absent"
    # a write/edit deliverable is its own grounding -> no flag
    b = lo()
    O.apply_receipt_absent_flag(b, {"tool_calls": [{"tool": "write_file", "status": "ok", "args": {"root": "user_files"}}]}, tempfile.mkdtemp(), "t2")
    assert "warning" not in b["outcome_axes"]["objective"]
    # a verify receipt -> no flag, and the receipt is injected into the trace for the ledger
    dr = tempfile.mkdtemp()
    O.append_verification_receipt(dr, "t3", {"status": "pass"})
    c = lo()
    tr = {"tool_calls": [{"tool": "commit_reviewed", "status": "ok"}]}
    O.apply_receipt_absent_flag(c, tr, dr, "t3")
    assert "warning" not in c["outcome_axes"]["objective"]
    assert tr.get("verification_receipts")


def test_receipt_absent_never_on_best_effort():
    from ouroboros import outcomes as O

    d = {"outcome_axes": {"execution": {"status": "best_effort"}, "objective": {"status": "not_evaluated"}}, "final_answer": ""}
    O.apply_receipt_absent_flag(d, {"tool_calls": [{"tool": "commit_reviewed", "status": "ok"}]}, tempfile.mkdtemp(), "t4")
    assert "warning" not in d["outcome_axes"]["objective"]


def test_m2_zero_grounding_flag():
    from ouroboros import outcomes as O

    # declared expected_output, no tool work, no structured answer -> M2
    a = {"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}, "final_answer": ""}
    O.apply_receipt_absent_flag(a, {"tool_calls": []}, tempfile.mkdtemp(), "m1", expected_output="report.html")
    assert a["outcome_axes"]["objective"].get("warning") == "expected_output_ungrounded"
    # a text-answer task (FINAL ANSWER present) is never M2-flagged
    b = {"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}, "final_answer": "42"}
    O.apply_receipt_absent_flag(b, {"tool_calls": []}, tempfile.mkdtemp(), "m2", expected_output="the number")
    assert "warning" not in b["outcome_axes"]["objective"]
    # no declared expected_output -> never M2
    c = {"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}, "final_answer": ""}
    O.apply_receipt_absent_flag(c, {"tool_calls": []}, tempfile.mkdtemp(), "m3", expected_output="")
    assert "warning" not in c["outcome_axes"]["objective"]


def test_nudge_gate_and_auto_equals_required():
    from ouroboros import outcomes as O

    tr = {"tool_calls": [{"tool": "commit_reviewed", "status": "ok"}]}
    # effects + no grounding -> nudge
    assert O.should_nudge_verification(tr, tempfile.mkdtemp(), "n1") is True
    # the durable flag is identical regardless of review mode (auto/required read the SAME store)
    dr = tempfile.mkdtemp()
    O.append_verification_receipt(dr, "n2", {"status": "pass"})
    assert O.should_nudge_verification(tr, dr, "n2") is False
    # no effects -> no nudge
    assert O.should_nudge_verification({"tool_calls": [{"tool": "read_file", "status": "ok"}]}, tempfile.mkdtemp(), "n3") is False


def test_latest_unreconciled_failed_verification_predicate():
    """v6.51.0 idea-3 + v6.78.0 (owner Q28=B): the red-verification predicate. The
    typed receipt status decides pass/fail (never prose), and a green reconciles a red
    ONLY when it grounds the SAME verification — same `criterion_id`, else the same
    whitespace-normalized `check` text. `declared` never reconciles a red."""
    from ouroboros import outcomes as O

    def dr_with(receipts):
        d = tempfile.mkdtemp()
        for i, r in enumerate(receipts):
            O.append_verification_receipt(d, "rt", r)
        return d

    # no receipts -> None
    assert O.latest_unreconciled_failed_verification(tempfile.mkdtemp(), "rt") is None
    # latest pass / observed -> no red
    assert O.latest_unreconciled_failed_verification(dr_with([{"status": "pass"}]), "rt") is None
    assert O.latest_unreconciled_failed_verification(dr_with([{"status": "observed"}]), "rt") is None
    # a lone fail -> returns it
    got = O.latest_unreconciled_failed_verification(dr_with([{"status": "fail", "check": "go test", "returncode": 1}]), "rt")
    assert got is not None and got.get("check") == "go test"
    # fail then later pass / observed OF THE SAME CHECK -> reconciled -> None
    assert O.latest_unreconciled_failed_verification(
        dr_with([{"status": "fail", "check": "go test"}, {"status": "pass", "check": "go test"}]), "rt") is None
    assert O.latest_unreconciled_failed_verification(
        dr_with([{"status": "fail", "check": "go test"}, {"status": "observed", "check": "go test"}]), "rt") is None
    # spacing BETWEEN tokens still reconciles (the canonical text is the same command)
    assert O.latest_unreconciled_failed_verification(
        dr_with([{"status": "fail", "check": "go  test"}, {"status": "pass", "check": "go test"}]), "rt") is None
    # same criterion_id reconciles even when the command text differs (id wins)
    assert O.latest_unreconciled_failed_verification(dr_with([
        {"status": "fail", "criterion_id": "c1", "check": "pytest tests/x.py"},
        {"status": "pass", "criterion_id": "c1", "check": "pytest tests/x.py -v"},
    ]), "rt") is None
    # v6.78.0: a green re-run of a DIFFERENT check no longer clears an unrelated red
    still_red = O.latest_unreconciled_failed_verification(dr_with([
        {"status": "fail", "check": "pytest tests/x.py"},
        {"status": "pass", "check": "pytest tests/y.py"},
    ]), "rt")
    assert still_red is not None and still_red.get("check") == "pytest tests/x.py"
    # ...including the cosmetic-reflag case (`-v`) with no criterion_id to bind them
    assert O.latest_unreconciled_failed_verification(dr_with([
        {"status": "fail", "check": "pytest tests/x.py"},
        {"status": "pass", "check": "pytest tests/x.py -v"},
    ]), "rt") is not None
    # an identity-LESS red (no criterion_id, no check, no paths) has no identity to
    # protect, so it keeps the pre-v6.78.0 rule: any later green clears it. The narrowing
    # would only mint an UNCLEARABLE flag — verify.py writes exactly such a red for a
    # malformed `artifact_observation` with no artifact_paths.
    assert O.latest_unreconciled_failed_verification(dr_with([{"status": "fail"}, {"status": "pass"}]), "rt") is None
    # fail then later DECLARED (escape hatch) -> NOT reconciled (codex #8) -> returns the fail
    assert O.latest_unreconciled_failed_verification(
        dr_with([{"status": "fail", "check": "c"}, {"status": "declared", "check": "c"}]), "rt") is not None
    # pass then fail -> latest red unreconciled -> returns the fail
    assert O.latest_unreconciled_failed_verification(dr_with([{"status": "pass"}, {"status": "fail"}]), "rt") is not None
    # content-independence of the STATUS: a PASS whose summary contains "FAIL" must NOT count as red...
    assert O.latest_unreconciled_failed_verification(dr_with([{"status": "pass", "summary": "1 FAIL earlier, now PASS"}]), "rt") is None
    # ...and a fail with a bland summary MUST count
    assert O.latest_unreconciled_failed_verification(dr_with([{"status": "fail", "summary": "all good"}]), "rt") is not None
    # malformed entries tolerated (non-dict / missing status)
    assert O.latest_unreconciled_failed_verification(dr_with([{"nostatus": 1}]), "rt") is None


def test_masked_pass_reconciliation_cannot_use_the_check_text_identity():
    """The masked path must NOT inherit the red path's check-text identity. A masked
    receipt's only text identity is its own MASKED command, and the sensor is a pure
    function of argv, so the byte-identical re-run is re-flagged masked and could never
    be the clean reconciler — text equality would make the flag unclearable by the very
    remediation the host prescribes ("drop the masking pipe"). Rule: equal criterion_id
    when the MASKED receipt carries one — a later clean receipt that omits its id does
    not clear it — else ANY later clean grounding."""
    from ouroboros import outcomes as O
    from ouroboros.tools.verify import _check_has_exit_masking

    # WHY text identity cannot bind here: remediation necessarily changes the text.
    assert _check_has_exit_masking(["sh", "-c", "make test | tail"])[0] is True
    assert _check_has_exit_masking(["sh", "-c", "make test"])[0] is False

    masked = {"status": "pass", "check": "sh -c make test | tail", "check_exit_masking": True}
    assert O.latest_unreconciled_masked_pass([masked]) is not None
    # the PRESCRIBED remediation (same run, masking pipe dropped) clears it
    assert O.latest_unreconciled_masked_pass(
        [masked, {"status": "pass", "check": "sh -c make test"}]) is None
    # so does any other later clean grounding, as before v6.78.0
    assert O.latest_unreconciled_masked_pass(
        [masked, {"status": "pass", "check": "make lint"}]) is None
    # a later receipt that is ITSELF masked never reconciles
    assert O.latest_unreconciled_masked_pass(
        [masked, {"status": "pass", "check": "make lint | tail", "check_exit_masking": True}]) is not None
    # criterion_id still binds when BOTH receipts carry one
    assert O.latest_unreconciled_masked_pass([
        dict(masked, criterion_id="c1"),
        {"status": "pass", "criterion_id": "c1", "check": "sh -c make test"},
    ]) is None
    assert O.latest_unreconciled_masked_pass([
        dict(masked, criterion_id="c1"),
        {"status": "pass", "criterion_id": "c2", "check": "sh -c make test"},
    ]) is not None


def test_artifact_observation_class_reconciles_on_its_observed_path_set():
    """Adversarial review (minor, accepted): the artifact-observation class runs NO
    command, so it has neither `check` nor (usually) `criterion_id` — under strict
    single-identity equality a red "report.md missing" could never be cleared by the
    byte-identical green observation after the file was written, leaving a permanent
    unreconciled red and a nudge on a task that DID verify itself. The observed path
    SET is that class's real identity."""
    from ouroboros import _outcome_receipts as R
    from ouroboros import outcomes as O

    red = {"status": "fail", "contract_kind": "artifact_observation", "paths": ["report.md"]}
    green = {"status": "observed", "contract_kind": "artifact_observation", "paths": ["report.md"]}
    other = {"status": "observed", "contract_kind": "artifact_observation", "paths": ["other.md"]}

    # The key VALUE is the injective serialization of the set, not a line-join: a path
    # may legally contain a newline, so `["a\nb"]` and `["a", "b"]` must not share a key.
    assert R.receipt_identity(red) == ("artifact_paths", '["report.md"]')
    assert R.receipt_identity({"paths": ["a\nb"]}) != R.receipt_identity({"paths": ["a", "b"]})
    # same path set -> reconciled (order-insensitive; the SET is the identity)
    assert O.latest_unreconciled_failed_receipt([red, green]) is None
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "paths": ["b.md", "a.md"]},
        {"status": "observed", "paths": ["a.md", "b.md"]},
    ]) is None
    # a DIFFERENT path does not reconcile it
    assert O.latest_unreconciled_failed_receipt([red, other]) is not None
    # a green COMMAND check does not reconcile an observation (different identity)
    assert O.latest_unreconciled_failed_receipt(
        [red, {"status": "pass", "check": "ls report.md"}]) is not None
    # criterion_id still wins when present
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "criterion_id": "c1", "paths": ["report.md"]},
        {"status": "observed", "criterion_id": "c1", "paths": ["elsewhere.md"]},
    ]) is None
    # an identity-LESS red keeps the pre-v6.78.0 rule (any later green clears it): it has
    # no identity to protect, and verify.py writes exactly this shape for a malformed
    # `artifact_observation` call with no artifact_paths — under the narrowing that one
    # mistake poisoned `unreconciled_red` for the rest of the task.
    assert R.receipt_identity({}) == ("none", "")
    assert R.receipt_identity({"paths": []}) == ("none", "")
    assert O.latest_unreconciled_failed_receipt([{"status": "fail"}, {"status": "observed"}]) is None
    malformed = {"status": "fail", "contract_kind": "artifact_observation",
                 "paths": [], "summary": "no artifact_paths given"}
    assert O.latest_unreconciled_failed_receipt(
        [malformed, {"status": "pass", "check": "pytest -q"}]) is None
    assert O.latest_unreconciled_failed_receipt([malformed]) is not None
    # a `declared` escape hatch still does not reconcile it
    assert O.latest_unreconciled_failed_receipt(
        [malformed, {"status": "declared", "check": "c"}]) is not None


def test_artifact_observation_reconciliation_clears_the_red_nudge(tmp_path, monkeypatch):
    """Loop-level consequence of the fix: the concrete flow (declare -> observe FAIL ->
    write the file -> observe again) must leave NO red nudge, while an observation of a
    different path must still nudge."""
    from types import SimpleNamespace

    import ouroboros.loop as loop_mod
    from ouroboros.outcomes import append_verification_receipt

    monkeypatch.setattr(loop_mod, "_skill_finalization_message", lambda *_a, **_k: "")

    def _fires(drive, receipts):
        for receipt in receipts:
            append_verification_receipt(drive, "t", receipt)
        return loop_mod._maybe_inject_finalization_nudges(
            SimpleNamespace(_ctx=SimpleNamespace()), drive, "t",
            {"reasoning_notes": [], "tool_calls": []}, "answer", [], lambda *_: None,
        )

    same = tmp_path / "same"
    same.mkdir()
    assert _fires(same, [
        {"status": "fail", "contract_kind": "artifact_observation", "paths": ["report.md"],
         "summary": "missing: report.md"},
        {"status": "observed", "contract_kind": "artifact_observation", "paths": ["report.md"],
         "summary": "observed 1 artifact(s): report.md"},
    ]) is False

    other = tmp_path / "other"
    other.mkdir()
    assert _fires(other, [
        {"status": "fail", "contract_kind": "artifact_observation", "paths": ["report.md"]},
        {"status": "observed", "contract_kind": "artifact_observation", "paths": ["notes.md"]},
    ]) is True


def test_receipt_identity_and_disclosed_whitespace_flag():
    """The identity rule and its DISCLOSED flag (Q28=B): criterion_id wins; without
    one the whitespace-normalized check text IS the verification's identity, and both
    reviewer-facing consumers say so."""
    from ouroboros import _outcome_receipts as R
    from ouroboros.outcomes import build_verification_ledger
    from ouroboros.review_evidence import _accept_verification_summary

    # The check key pairs the canonical text with the RENDERING that produced the stored
    # string (round 8) — `receipt_identity_parts` below still exposes the plain text.
    go_test = ("check", json.dumps(["unversioned", "go test"]))
    assert R.receipt_identity({"criterion_id": " c1 ", "check": "x"}) == ("criterion_id", "c1")
    assert R.receipt_identity({"check": "go   test"}) == go_test
    assert R.receipt_identity({"paths": ["a.md"], "check": "go test"}) == go_test
    # The three components are independent; `receipt_identity` selects the ONE typed key
    # sameness is decided by, and `receipt_identity_parts` discloses all three.
    assert R.receipt_identity_parts(
        {"criterion_id": " c1 ", "check": "go   test", "paths": ["b.md", "a.md"]}
    ) == ("c1", "go test", "a.md\nb.md")
    assert R.receipt_identity_parts({}) == ("", "", "")
    assert R.receipt_identity_parts({"paths": "not-a-list"}) == ("", "", "")
    # The flag is TRUE for the `check` kind and false for every other kind — see
    # test_the_whitespace_flag_is_derived_once_for_every_identity_kind.
    assert R.receipt_expected_whitespace_normalized({"check": "go test"}) is True
    assert R.receipt_expected_whitespace_normalized({"paths": ["a.md"]}) is False
    assert R.receipt_expected_whitespace_normalized({"criterion_id": "c1", "check": "go test"}) is False
    assert R.receipt_expected_whitespace_normalized({}) is False

    # Consumer 1: the acceptance reviewer's verification_summary.
    summary = _accept_verification_summary([{"status": "fail", "check": "pytest tests/x.py"}])
    assert summary["expected_whitespace_normalized"] is True
    assert summary["reconciliation_identity_kinds"] == ["check"]
    assert summary["latest_identity"]["paths"] == []  # a command check observes no path set
    id_summary = _accept_verification_summary([
        {"status": "pass", "criterion_id": "c1", "check": "pytest"},
        {"status": "observed", "paths": ["report.md"]},
    ])
    # Neither row is governed by command text: the `criterion_id` row is named, and the
    # observation row's path SET normalizes nothing (round 7).
    assert id_summary["expected_whitespace_normalized"] is False
    assert id_summary["reconciliation_identity_kinds"] == ["artifact_paths", "criterion_id"]
    # the observation class runs no command, so its identity reaches the reviewer as the
    # observed path set (latest_check is empty for it) — never silently dropped
    assert id_summary["latest_check"] == ""
    assert id_summary["latest_identity"]["paths"] == ["report.md"]
    assert _accept_verification_summary(
        [{"status": "pass", "criterion_id": "c1", "check": "pytest"}]
    )["expected_whitespace_normalized"] is False

    # Consumer 2: the FIXED verification-ledger receipt projection.
    ledger = build_verification_ledger(
        task={"id": "t1"},
        loop_outcome={"outcome_axes": {"execution": {"status": "ok"}}},
        llm_trace={"verification_receipts": [
            {"status": "fail", "check": "pytest tests/x.py"},
            {"status": "observed", "contract_kind": "artifact_observation", "paths": ["report.md"]},
        ]},
        artifact_bundle={"status": "not_applicable", "artifacts": [], "errors": []},
    )
    rows = [r for r in ledger["entries"] if r.get("kind") == "verification_receipt"]
    assert rows[0]["expected_whitespace_normalized"] is True
    assert rows[0]["reconciliation_identity"] == "check"
    assert rows[0]["paths"] == []
    assert rows[1]["reconciliation_identity"] == "artifact_paths"
    assert rows[1]["expected_whitespace_normalized"] is False
    # the FIXED projection carries the identity the reconciliation actually used
    assert rows[1]["paths"] == ["report.md"]


def test_receipt_reconciliation_matches_one_typed_identity_key():
    """Sameness is ONE typed key (kind AND value), never a fallback chain over the three
    components: an existing `criterion_id` is authoritative, so a green that carries only
    the check text cannot clear a criterion-keyed red — in either direction — and a
    command check never reconciles a bare observation."""
    from ouroboros import outcomes as O

    # Round-5 CRITICAL, stated as behaviour: the earlier red is keyed by `c1`, the later
    # green only by its check text. Different KINDS never match, so the red stays open.
    # This is the narrowing that made the relation transitive; it fails SAFE (a red that
    # the chain used to clear may now stay open — never the reverse).
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "criterion_id": "c1", "check": "pytest tests/x.py"},
        {"status": "pass", "check": "pytest  tests/x.py"},
    ]) is not None
    # ...and the reverse direction (the red lacked the id, the green added one)
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "check": "pytest tests/x.py"},
        {"status": "pass", "criterion_id": "c1", "check": "pytest tests/x.py"},
    ]) is not None
    # The same key on both sides still reconciles, by id or by check text alone.
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "criterion_id": "c1", "check": "pytest tests/x.py"},
        {"status": "pass", "criterion_id": "c1", "check": "pytest tests/x.py -v"},
    ]) is None
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "check": "pytest tests/x.py"},
        {"status": "pass", "check": "pytest  tests/x.py"},
    ]) is None
    # two DIFFERENT ids are two different criteria — identical command text does NOT
    # collapse them
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "criterion_id": "c1", "check": "pytest tests/x.py"},
        {"status": "pass", "criterion_id": "c2", "check": "pytest tests/x.py"},
    ]) is not None
    # a command check and a bare observation are different verifications, both ways
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "check": "ls report.md"},
        {"status": "observed", "paths": ["report.md"]},
    ]) is not None
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "paths": ["report.md"]},
        {"status": "pass", "check": "ls report.md", "paths": ["report.md"]},
    ]) is not None
    # the masked-green path keys on the criterion_id ALONE (its own text identity is the
    # masked command, which the remediation changes — see
    # test_masked_pass_reconciliation_cannot_use_the_check_text_identity), and it is the
    # same typed key: a later clean receipt that OMITS its id no longer clears an
    # identified masked criterion (round 5, the masked half of the same ambiguity).
    assert O.latest_unreconciled_masked_pass([
        {"status": "pass", "criterion_id": "c1", "check": "make test | tail", "check_exit_masking": True},
        {"status": "pass", "check": "make test | tail"},
    ]) is not None
    assert O.latest_unreconciled_masked_pass([
        {"status": "pass", "criterion_id": "c1", "check": "make test | tail", "check_exit_masking": True},
        {"status": "pass", "criterion_id": "c1", "check": "make test"},
    ]) is None
    # ...while a masked receipt with NO id keeps the any-later-clean fallback: it names
    # no criterion, so narrowing would only mint an unclearable flag.
    assert O.latest_unreconciled_masked_pass([
        {"status": "pass", "check": "make test | tail", "check_exit_masking": True},
        {"status": "pass", "check": "make test"},
    ]) is None
    assert O.latest_unreconciled_masked_pass([
        {"status": "pass", "criterion_id": "c1", "check": "make test | tail", "check_exit_masking": True},
        {"status": "pass", "criterion_id": "c2", "check": "make test | tail"},
    ]) is not None


def test_red_verification_nudge_one_shot_and_before_receipt_absent(monkeypatch):
    """v6.51.0 idea-3 loop wiring: the red nudge fires ONCE (latch) and BEFORE the FR3
    receipt-absent nudge. (Forced-finalization paths return before the injector is called;
    that is a property of the call site, not exercised here.)"""
    import types as _t
    from ouroboros import loop as L
    from ouroboros import outcomes as O

    monkeypatch.setattr(L, "_skill_finalization_message", lambda *a, **k: "")
    dr = Path(tempfile.mkdtemp())
    O.append_verification_receipt(dr, "redt", {"status": "fail", "check": "pytest", "returncode": 1})
    ctx = _t.SimpleNamespace(task_contract={}, task_metadata={})
    tools = _t.SimpleNamespace(_ctx=ctx)
    # a turn WITH reviewable effects + a red receipt: BOTH the red gate and the FR3 gate qualify.
    trace = {"reasoning_notes": [], "tool_calls": [{"tool": "commit_reviewed", "status": "ok"}]}
    msgs: list = []
    fired = L._maybe_inject_finalization_nudges(tools, dr, "redt", trace, "draft answer", msgs, lambda *_: None)
    assert fired is True
    assert getattr(ctx, "_verify_red_nudged", False) is True
    assert getattr(ctx, "_verify_nudged", False) is False  # red fired first, FR3 NOT yet
    assert "RED" in msgs[-1]["content"] and "pytest" in msgs[-1]["content"]
    # second call: red latch set -> red skipped -> FR3 receipt-absent nudge now fires
    fired2 = L._maybe_inject_finalization_nudges(tools, dr, "redt", trace, "draft answer", msgs, lambda *_: None)
    assert fired2 is True
    assert getattr(ctx, "_verify_nudged", False) is True
    # third call: both latched, no further effects-based nudge -> no re-loop
    assert L._maybe_inject_finalization_nudges(tools, dr, "redt", trace, "draft", msgs, lambda *_: None) is False


def test_verification_receipts_in_ledger():
    from ouroboros.outcomes import build_verification_ledger

    led = build_verification_ledger(
        task={"id": "t", "task_contract": {}},
        loop_outcome={"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}},
        llm_trace={"tool_calls": [], "verification_receipts": [{"status": "pass", "contract_kind": "explicit_command", "check": "pytest"}]},
        artifact_bundle={},
    )
    kinds = [e.get("kind") for e in led.get("entries", [])]
    assert "verification_receipt" in kinds


# ── M6: cosmetic/recovered tool errors never produce a terminal tool_failure ──
def test_m6_cosmetic_errors_no_terminal_tool_failure():
    from ouroboros.outcomes import REASON_TOOL_FAILURE, derive_loop_outcome

    # an unrecovered one-shot run_command non-zero exit is COSMETIC (T4) — the turn
    # finished with a real answer, so it must land execution ok / final_message, never
    # a terminal tool_failure.
    trace = {"tool_calls": [{"tool": "run_command", "status": "non_zero_exit", "is_error": True, "exit_code": 1, "result": "boom"}]}
    out = derive_loop_outcome("done", {}, trace)
    axes = out["outcome_axes"]
    assert axes["execution"]["status"] == "ok"
    assert out["reason_code"] != REASON_TOOL_FAILURE
    # the cosmetic residual is still surfaced as a warning (just never as tool_failure)
    assert axes["objective"].get("warning") == "residual_tool_errors_without_review"


# ── FR2: cooperative subagent shared tree + depth reservation ─────────────────
def test_depth_reservation_admits():
    from supervisor.events import _depth_reservation_admits

    def t(tid, parent, rt="R"):
        return {"id": tid, "parent_task_id": parent, "root_task_id": rt, "delegation_role": "subagent"}

    running = {"w1": {"task": t("P", "ROOT")}}
    pending = [t(f"c{i}", "ROOT") for i in range(6)]  # tree at cap=6
    # P is a running subagent with no direct child -> reservation admits ONE
    assert _depth_reservation_admits("R", "P", pending, running, 6) is True
    # once P has a direct child, no further reservation for P
    assert _depth_reservation_admits("R", "P", pending + [t("pc", "P")], running, 6) is False
    # parent not a running subagent -> no reservation
    assert _depth_reservation_admits("R", "ROOT", pending, running, 6) is False
    # hard ceiling (2*cap) bounds it
    assert _depth_reservation_admits("R", "P", [t(f"d{i}", "ROOT") for i in range(12)], running, 6) is False


def test_ensure_cooperative_shared_root_mints_git_tree(monkeypatch):
    from ouroboros.tools import control_delegation as CD

    projects_root = Path(tempfile.mkdtemp()) / "coop_projects"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    CD._COOP_SHARED_ROOTS.clear()
    ctx = types.SimpleNamespace(repo_dir=str(Path(tempfile.mkdtemp())), task_id="root-1")
    path = CD.ensure_cooperative_shared_root(ctx, "root-1")
    assert not path.startswith("⚠️"), path
    assert (Path(path) / ".git").exists()  # a real git tree was minted
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head  # has the seed commit
    # cached: a second call returns the SAME tree (one shared tree per task-tree)
    assert CD.ensure_cooperative_shared_root(ctx, "root-1") == path


def test_resolve_cooperative_write_root_routes_flat_parent(monkeypatch):
    from ouroboros.tools import control_delegation as CD

    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(Path(tempfile.mkdtemp()) / "p"))
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    CD._COOP_SHARED_ROOTS.clear()
    ctx = types.SimpleNamespace(repo_dir=str(Path(tempfile.mkdtemp())), task_id="r2", task_constraint=None,
                                is_workspace_mode=lambda: False, is_direct_chat=False)
    eff, profile, err = CD.resolve_cooperative_write_root(ctx, "external_workspace", "", "", {"root_task_id": "r2"})
    assert err == "" and eff and (Path(eff) / ".git").exists()
    # an explicit write_root is passed through untouched
    eff2, _p, err2 = CD.resolve_cooperative_write_root(ctx, "external_workspace", "/some/dir", "", {})
    assert err2 == "" and eff2 == "/some/dir"


# ── FR1: skill-publish SSOT predicate ─────────────────────────────────────────
def test_submit_hub_eligibility_warnings_now_enabled():
    from ouroboros.skill_publish_eligibility import submit_hub_eligibility as E

    # THE desync fix: advisory-only warnings are publishable (was UI-disabled before)
    assert E(source="external", review_status="warnings", github_token_configured=True)["disabled"] is False
    assert E(source="external", review_status="clean", github_token_configured=True)["disabled"] is False
    for row in (
        E(source="external", review_status="blockers", github_token_configured=True),
        E(source="external", review_status="pending", github_token_configured=True),
        E(
            source="external",
            review_status="clean",
            review_profile="owner_attested",
            github_token_configured=True,
        ),
        E(
            source="external",
            review_status="clean",
            review_stale=True,
            github_token_configured=True,
        ),
    ):
        assert row["publication_ready"] is False
        assert row["task_start_allowed"] is True
        assert row["disabled"] is False
    assert "GITHUB_TOKEN" in E(source="external", review_status="clean")["reason"]
    assert E(source="native", review_status="clean", github_token_configured=True)["visible"] is False


def test_publish_gate_and_predicate_share_statuses():
    # The backend publish gate uses the SAME SSOT status set as the UI predicate.
    from ouroboros.skill_publish_eligibility import PUBLISHABLE_STATUSES
    from ouroboros.skill_review_status import STATUS_CLEAN, STATUS_WARNINGS

    assert PUBLISHABLE_STATUSES == frozenset({STATUS_CLEAN, STATUS_WARNINGS})


# ── verify_and_record safety policy + dispatch guard ──────────────────────────
def test_verify_and_record_safety_policy_is_conditional():
    from ouroboros.safety import POLICY_CHECK_CONDITIONAL, TOOL_POLICY

    assert TOOL_POLICY.get("verify_and_record") == POLICY_CHECK_CONDITIONAL


def test_verify_and_record_check_is_shell_guarded_against_subagent_secret_read():
    # F1 (review #1): an acting subagent must NOT be able to read Ouroboros secrets
    # through verify_and_record's `check` — it routes through the same deterministic
    # shell guard as run_command.
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.shell_guards import process_shell_guard_args

    reg = ToolRegistry(repo_dir=".", drive_root=tempfile.mkdtemp())
    reg._ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=tempfile.mkdtemp())
    mapped = process_shell_guard_args("verify_and_record", {"check": "cat data/settings.json", "cwd": ""})
    # v6.51.0: normalized via the SSOT (non-login `sh -c`); the guard still inspects the inner command.
    assert mapped["cmd"] == ["sh", "-c", "cat data/settings.json"]
    block = _shell_guard_text(reg, mapped, "advanced")
    assert block and "SECRET" in block.upper()


def test_verify_string_check_no_safe_subject_bypass(monkeypatch):
    # triad round-3 #1: a STRING check runs via `sh -c`, so a safe-looking first word
    # cannot bypass the LLM safety review (a compound `cat x; rm` would be shell-run);
    # a LIST check (argv, no shell) stays safe-subject-eligible like run_command.
    import ouroboros.safety as S

    monkeypatch.setattr(S, "_run_llm_check", lambda *a, **k: (True, "LLM_REACHED"))
    _ok, msg_list = S.check_safety("verify_and_record", {"contract_kind": "explicit_command", "check": ["cat", "x"]}, messages=[], ctx=None)
    assert msg_list != "LLM_REACHED"  # safe-subject bypass for an argv list
    _ok, msg_str = S.check_safety("verify_and_record", {"contract_kind": "explicit_command", "check": "cat x; rm -rf y"}, messages=[], ctx=None)
    assert msg_str == "LLM_REACHED"  # string check forced through the LLM review


def test_verify_and_record_reachable_in_workspace_mode(tmp_path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system, workspace, data = tmp_path / "system", tmp_path / "workspace", tmp_path / "data"
    for path in (system, workspace, data):
        path.mkdir()
    registry = ToolRegistry(system, data)
    registry.set_context(ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    ))
    assert registry.get_schema_by_name("verify_and_record") is not None


def test_verify_and_record_is_shell_guarded_not_process_command():
    # triad round-5: verify_and_record clears the PRE-EXECUTION shell guards (the security
    # boundary, which blocks a forbidden mutation before the handler runs) but is NOT in
    # _PROCESS_COMMAND_TOOLS — those POST-execution checks run AFTER the handler already
    # wrote the receipt, so they would not gate the durable receipt (an ordering inversion).
    from ouroboros.tools.registry import _PROCESS_COMMAND_TOOLS, _SHELL_GUARDED_TOOLS

    assert "verify_and_record" in _SHELL_GUARDED_TOOLS
    assert "verify_and_record" not in _PROCESS_COMMAND_TOOLS


def test_v651_stringified_argv_recovery_and_check_normalization():
    """v6.51.0 idea-1: the SSOT stringified-argv recovery + check normalization."""
    from ouroboros.shell_parse import normalize_check_argv, recover_stringified_argv

    # recovery: JSON list, Python list literal, plain string, malformed
    assert recover_stringified_argv('["go","test"]') == ["go", "test"]
    assert recover_stringified_argv("['go','test']") == ["go", "test"]
    assert recover_stringified_argv("go test") is None      # plain string is NOT shell-split here
    assert recover_stringified_argv("[broken") is None
    assert recover_stringified_argv("[]") == []             # no-drift edge: matches the old inline _run_shell
    # normalization: stringified-argv recovers to argv; plain string -> NON-login sh -c (PATH parity)
    assert normalize_check_argv('["go","test"]') == ["go", "test"]
    assert normalize_check_argv("go test") == ["sh", "-c", "go test"]
    assert normalize_check_argv(["go", "test"]) == ["go", "test"]
    assert normalize_check_argv("   ") is None


def test_v651_verify_guard_inspects_exact_executed_argv():
    """v6.51.0 idea-1: the shell guard normalizes the check through the SAME SSOT as
    execution, so a stringified-argv check is guard-inspected as the recovered argv (not
    as a literal sh -c string) — guard == execution, no drift."""
    from ouroboros.tools.shell_guards import process_shell_guard_args
    from ouroboros.tools.verify import _normalize_check

    mapped = process_shell_guard_args("verify_and_record", {"check": '["cat", "data/settings.json"]', "cwd": ""})
    assert mapped["cmd"] == ["cat", "data/settings.json"]          # recovered argv, NOT ["sh","-c",'["cat",...]']
    assert mapped["cmd"] == _normalize_check('["cat", "data/settings.json"]')  # guard == execution


def test_v651_build_task_acceptance_evidence_process_aware():
    """v6.51.0 idea-2: process-aware acceptance evidence — typed sections, provenance tags,
    first-class verification_summary (RED surfaced), bounded+redacted trajectory, leak-safe
    artifacts (protected = manifest-only), and an agent diff demoted (never host)."""
    import types as _t
    from ouroboros import outcomes as O
    from ouroboros.review_evidence import build_task_acceptance_evidence

    dr = Path(tempfile.mkdtemp())
    # the receipt summary is raw host stdout — a secret in it must be REDACTED (review HIGH-1)
    O.append_verification_receipt(dr, "acc", {"status": "fail", "check": "pytest", "returncode": 1,
                                              "summary": "leaked sk-or-v1-SECRETTOKEN0123456789abcdef in output"})
    art = dr / "task_results" / "artifacts" / "acc"
    art.mkdir(parents=True, exist_ok=True)
    (art / "secret_oracle.txt").write_text("HIDDEN GOLD TESTS")
    (art / "out.txt").write_text("hello world")
    ctx = _t.SimpleNamespace(
        task_contract={"requirements": "do X", "interface": "def f()", "expected_output": "42",
                       "resource_policy": {"protected_artifacts": [{"path": "secret_oracle.txt"}]}},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )
    trace = {"reasoning_notes": ["thinking"], "tool_calls": [
        {"tool": "run_command", "status": "ok", "result": "x" * 20000},
        {"tool": "verify_and_record", "status": "ok", "result": "ghp_SECRETGHTOKEN0123456789abcdefABCD"},
    ]}
    ev = build_task_acceptance_evidence(ctx, llm_trace=trace, drive_root=dr, task_id="acc",
                                        agent_evidence={"repo_diff": "agent claims clean", "note": "n",
                                                        "leaked": "sk-or-v1-AGENTSUPPLIED0123456789abcdefXYZ"})
    # leak-safety: no secret (receipt summary, tool result, OR agent-supplied) survives serialization
    import json as _json
    _blob = _json.dumps(ev, ensure_ascii=False)
    assert "sk-or-v1-SECRETTOKEN0123456789abcdef" not in _blob   # HIGH-1: verification_summary redacted
    assert "ghp_SECRETGHTOKEN0123456789abcdefABCD" not in _blob  # tool-trajectory result redacted
    assert "sk-or-v1-AGENTSUPPLIED0123456789abcdef" not in _blob  # round-4: agent_supplied redacted
    assert ev["task_contract"]["requirements"] == "do X" and ev["task_contract"]["expected_output"] == "42"
    assert ev["verification_summary"]["unreconciled_red"] is True
    assert ev["verification_summary"]["failed_count"] == 1
    # v6.71.1 evidence-parity: the per-result cap now tracks the actor's own PER-TOOL
    # window (SSOT tool_capabilities.TOOL_RESULT_LIMITS / DEFAULT_TOOL_RESULT_LIMIT),
    # so a 20k run_command result (actor window 80k) reaches the reviewer whole
    # instead of the old hidden 700.
    from ouroboros.review_evidence import _ACCEPT_RESULT_CAP as _DEFAULT_CAP
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS as _LIMITS
    assert ev["tool_trajectory"] and all(
        len(c["result"]) <= _LIMITS.get(c["tool"], _DEFAULT_CAP) + 300
        for c in ev["tool_trajectory"]
    )  # per-result cap = actor's own window
    _rc = next(c for c in ev["tool_trajectory"] if c["tool"] == "run_command")
    assert len(_rc["result"]) > 6000  # the reviewer now sees far more than the old 700/4000
    p = ev["__provenance__"]
    assert p["task_contract"] == "host_attested" and p["verification_summary"] == "host_attested"
    assert p["tool_trajectory"] == "tool_result" and p.get("agent_supplied") == "agent_supplied"
    assert ev["agent_supplied"]["agent_supplied_repo_diff"] == "agent claims clean"  # demoted
    assert "repo_diff" not in ev["agent_supplied"]                                    # never host
    arts = {a["name"]: a for a in ev["artifacts"]}
    assert "HIDDEN GOLD" not in arts["secret_oracle.txt"].get("preview", "")          # protected = no bytes
    assert arts["secret_oracle.txt"]["provenance"] == "hidden_or_restricted"
    assert "hello world" in arts["out.txt"]["preview"]


def test_acceptance_evidence_unions_split_root_actor_receipts(tmp_path):
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.review_evidence import build_task_acceptance_evidence

    local = tmp_path / "child"
    canonical = tmp_path / "canonical"
    local.mkdir()
    canonical.mkdir()
    ctx = _verify_ctx(local, task_id="split-acceptance")
    ctx.drive_root = local
    ctx.budget_drive_root = str(canonical)
    append_verification_receipt(canonical, ctx.task_id, {
        "status": "declared", "contract_kind": "delegation_zero_run",
        "zero_run": True, "zero_run_decision": "complete",
    })
    append_verification_receipt(local, ctx.task_id, {
        "status": "pass", "check": "pytest tests/split.py",
        "criterion_id": "split-check",
    })

    packet = build_task_acceptance_evidence(
        ctx, drive_root=local, task_id=ctx.task_id,
    )

    assert packet["verification_summary"]["count"] == 2
    assert [row.get("contract_kind") for row in packet["verification_receipts"]] == [
        "", "delegation_zero_run",
    ]


def test_v651_acceptance_evidence_budget_disclosed():
    """Verbose evidence is reduced, but immutable owner intent is never truncated."""
    import types as _t
    from ouroboros.review_evidence import build_task_acceptance_evidence

    import json as _json
    from ouroboros.review_evidence import _ACCEPT_TOTAL_BUDGET

    dr = Path(tempfile.mkdtemp())
    # Pathological immutable intent cannot be made to fit without changing it.
    ctx = _t.SimpleNamespace(task_contract={"requirements": "Q" * 500_000}, task_metadata={}, repo_dir=str(dr))
    trace = {"tool_calls": [{"tool": "run_command", "status": "ok", "result": "y" * 3000} for _ in range(400)]}
    ev = build_task_acceptance_evidence(ctx, llm_trace=trace, drive_root=dr, task_id="b")
    assert "__budget_note__" in ev and "OMISSION NOTE" in ev["__budget_note__"]
    assert len(ev["tool_trajectory"]) == 20
    assert ev["tool_trajectory_omitted_leading"] >= 380
    serialized = _json.dumps(ev, ensure_ascii=False)
    assert len(serialized) > _ACCEPT_TOTAL_BUDGET
    overflow = ev["__immutable_core_overflow__"]
    assert overflow["budget_chars"] == _ACCEPT_TOTAL_BUDGET
    assert overflow["packet_chars"] > _ACCEPT_TOTAL_BUDGET
    assert ev["task_contract"]["requirements"] == "Q" * 500_000


def test_v651_protected_artifact_normalized_paths_shape():
    """v6.51.0 review round-2: protected artifacts in the NORMALIZED resource_policy shape store
    locations under a `paths` LIST (normalize_resource_policy) — they must be classified
    hidden_or_restricted (manifest-only), not previewed."""
    import types as _t
    from ouroboros.review_evidence import build_task_acceptance_evidence

    dr = Path(tempfile.mkdtemp())
    art = dr / "task_results" / "artifacts" / "pp"
    art.mkdir(parents=True, exist_ok=True)
    (art / "oracle.bin").write_text("GOLD ORACLE CONTENT")
    ctx = _t.SimpleNamespace(
        task_contract={"resource_policy": {"protected_artifacts": [
            {"id": "o", "role": "black_box_reference", "paths": ["oracle.bin"]}]}},
        task_metadata={}, repo_dir=str(dr),
    )
    ev = build_task_acceptance_evidence(ctx, drive_root=dr, task_id="pp")
    arts = {a["name"]: a for a in ev["artifacts"]}
    assert arts["oracle.bin"]["provenance"] == "hidden_or_restricted"   # matched via "paths" list
    assert "GOLD ORACLE" not in arts["oracle.bin"].get("preview", "")   # no bytes leaked


def test_v651_orchestrate_timeout_reaps_group_and_container(monkeypatch):
    """v6.51.0 review round-4: a host-timed-out run_pro is killed via platform_layer (cross-
    platform, NOT a raw os.killpg) and the worker's leaked obopro-w{N}-* container is removed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "orchp_t", "devtools/benchmarks/swe_bench_pro/e1v2/orchestrate_probe.py")
    orchp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchp)

    killed = {}
    monkeypatch.setattr(orchp, "kill_process_tree", lambda p: killed.setdefault("tree", p))
    calls = []

    class _R:
        stdout = "abc123\n"

    monkeypatch.setattr(orchp.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), _R())[1])

    class _Proc:
        def wait(self, timeout=None):
            return 0

    orchp.reap_timed_out_runpro(_Proc(), 2, {})
    assert killed.get("tree") is not None                                    # process TREE killed (cross-platform)
    assert ["docker", "ps", "-q", "--filter", "name=obopro-w2-"] in calls    # worker-scoped reap filter
    assert ["docker", "rm", "-f", "abc123"] in calls                         # leaked container removed


def _verify_ctx(tmp_path, *, task_id="vhandler"):
    from ouroboros.tools.registry import ToolContext

    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "drive").mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(repo_dir=str(tmp_path / "repo"), drive_root=str(tmp_path / "drive"))
    ctx.task_id = task_id
    return ctx


def test_verify_and_record_handler_run_kinds(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    from ouroboros.outcomes import read_verification_receipts, verification_grounding_present
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    assert "PASS" in _verify_and_record(ctx, contract_kind="explicit_command", check=[PY, "-c", "print('ok')"])
    rs = read_verification_receipts(ctx.drive_root, "vhandler")
    assert rs[-1]["status"] == "pass" and rs[-1]["returncode"] == 0
    assert "FAIL" in _verify_and_record(ctx, contract_kind="explicit_command", check=[PY, "-c", "import sys; sys.exit(1)"])
    assert read_verification_receipts(ctx.drive_root, "vhandler")[-1]["status"] == "fail"
    # expected-substring gates pass/fail even on exit 0
    assert "FAIL" in _verify_and_record(ctx, contract_kind="explicit_metric", check=[PY, "-c", "print('hello')"], expected="WORLD")
    assert "PASS" in _verify_and_record(ctx, contract_kind="explicit_metric", check=[PY, "-c", "print('hello')"], expected="hello")
    # the handler's pass status actually grounds the turn (handler<->grounding contract)
    assert verification_grounding_present({"tool_calls": []}, ctx.drive_root, "vhandler") is True


def test_verify_and_record_handler_fail_closed(tmp_path):
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    assert "TOOL_ARG_ERROR" in _verify_and_record(ctx, contract_kind="some_future_kind")
    assert "requires `check`" in _verify_and_record(ctx, contract_kind="explicit_command")


def test_verify_and_record_handler_artifact_and_declared(tmp_path):
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    # a relative artifact path resolves under the active workspace (the repo dir here)
    (Path(ctx.repo_dir) / "deliv.txt").write_text("x")
    assert "OBSERVED" in _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=["deliv.txt"])
    assert "FAIL" in _verify_and_record(ctx, contract_kind="artifact_observation", artifact_paths=["missing.txt"])
    # triad #A: a RELATIVE path that escapes the workspace cannot probe arbitrary host
    # files (no `../../../etc/passwd` existence oracle) — confined post-resolution. The
    # security invariant is that the traversal target is NEVER OBSERVED. v6.57.0: an
    # out-of-scope path is now a POLICY refusal (REFUSED_OUT_OF_SCOPE), not a FAIL — it
    # still never yields an existence oracle.
    # Deep enough to escape to the FILESYSTEM ROOT on every OS (Windows runners
    # nest tmp ~7 levels under the user home — six `..` landed INSIDE home there,
    # where the deliberate user_files read lane made the probe an honest miss
    # instead of a refusal).
    escaped = _verify_and_record(
        ctx, contract_kind="artifact_observation",
        artifact_paths=["../" * 15 + "etc/passwd"],
    )
    assert "OBSERVED" not in escaped and "REFUSED_OUT_OF_SCOPE" in escaped
    # no_visible_machine_contract -> honest declared receipt (grounding)
    assert "DECLARED" in _verify_and_record(ctx, contract_kind="no_visible_machine_contract", check="manual UI review")
    assert read_verification_receipts(ctx.drive_root, "vhandler")[-1]["status"] == "declared"


def test_configured_actor_can_record_typed_zero_run_only_before_leaf(tmp_path):
    from ouroboros.outcomes import (
        apply_receipt_absent_flag,
        read_verification_receipts,
        should_nudge_verification,
        verification_grounding_present,
    )
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    result = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="incomplete",
        zero_run_basis="The selected route was unavailable and no host child produced the requested artifact.",
    )
    assert "INCOMPLETE" in result
    receipt = read_verification_receipts(ctx.drive_root, ctx.task_id)[-1]
    assert receipt["contract_kind"] == "delegation_zero_run"
    assert receipt["status"] == "declared"
    assert receipt["zero_run_decision"] == "incomplete"
    assert receipt["physical_run_started"] is False
    assert ctx._configured_actor_bootstrap["zero_run_receipt_recorded"] is True

    # A no-leaf lifecycle decision is not proof that the requested deliverable
    # is correct.  It remains visible in the receipt packet without suppressing
    # receipt_absent or masquerading as host-attested grounding.  The actor also
    # must not become grounding. The pure receipt helper has no tool-profile
    # context, so it still reports that a reviewable effect needs verification;
    # the loop suppresses that impossible reminder only for the readonly actor
    # profile whose generic verify surface disappears after zero-run.
    trace = {"tool_calls": [{"tool": "integrate_subagent_patch", "status": "ok", "args": {}}]}
    assert verification_grounding_present(trace, ctx.drive_root, ctx.task_id) is False
    assert should_nudge_verification(trace, ctx.drive_root, ctx.task_id) is True
    loop_outcome = {
        "outcome_axes": {
            "execution": {"status": "ok"},
            "objective": {"status": "satisfied"},
        },
        "final_answer": "host-side result",
    }
    apply_receipt_absent_flag(
        loop_outcome, trace, ctx.drive_root, ctx.task_id,
        expected_output="verified result",
    )
    assert "receipt_absent" in loop_outcome["outcome_axes"]["objective"]["warnings"]

    ctx._configured_actor_bootstrap["physical_started"] = True
    refused = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="late decision",
    )
    assert "TOOL_ARG_ERROR" in refused

    duplicate = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="incomplete",
        zero_run_basis="second terminal decision",
    )
    assert "TOOL_ARG_ERROR" in duplicate


def test_zero_run_refuses_ambiguous_physical_start_custody(tmp_path):
    from ouroboros import delegate_custody as custody
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    custody_drive = custody.custody_root(ctx)
    assert custody.record_start_requested(
        custody_drive,
        task_id=ctx.task_id,
        invocation_id="inv-unknown",
        idempotency_key="inv-unknown",
        run_id="",
        request={"prompt": "canonical"},
        route="codex",
    )
    refused = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="the POST answer was lost",
    )
    assert "zero_run_requires_settlement" in refused
    assert "inv-unknown" in refused
    assert read_verification_receipts(custody_drive, ctx.task_id) == []
    assert ctx._configured_actor_bootstrap.get("zero_run_receipt_recorded") is not True


def test_zero_run_refuses_unreadable_custody_without_fail_soft_scan(
    tmp_path, monkeypatch,
):
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_recovery
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: True)
    monkeypatch.setattr(
        delegate_recovery,
        "unsettled_start_ids",
        lambda *_a, **_k: pytest.fail("an unreadable authority must stop before scan"),
    )
    refused = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="no visible run",
    )
    assert "zero_run_custody_unknown" in refused
    assert "custody_log_unreadable" in refused
    assert read_verification_receipts(tmp_path, ctx.task_id) == []
    assert ctx._configured_actor_bootstrap.get("zero_run_receipt_recorded") is not True


def test_zero_run_and_fresh_start_share_one_atomic_actor_decision(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from ouroboros import delegate_custody as custody
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.delegate_integration import claimed_start_request
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path, task_id="actor-claim")
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    drive = custody.custody_root(ctx)
    barrier = threading.Barrier(2)

    def claim_start():
        barrier.wait()
        return claimed_start_request(
            drive,
            claim_target="",
            actor_ctx=ctx,
            enforce_actor_idle=True,
            run_id="",
            task_id=ctx.task_id,
            idempotency_key="actor-claim-invocation",
            invocation_id="actor-claim-invocation",
            max_seconds=30,
            request={"prompt": "exact physical assignment"},
            project_id="project-1",
            project_owned=False,
            route="codex",
        )

    def claim_zero_run():
        barrier.wait()
        return _verify_and_record(
            ctx,
            contract_kind="delegation_zero_run",
            zero_run_decision="incomplete",
            zero_run_basis="host-visible work stopped without a physical leaf",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        start_future = pool.submit(claim_start)
        zero_future = pool.submit(claim_zero_run)
        start = start_future.result()
        zero = zero_future.result()

    start_won = start[0] is True
    zero_won = "typed host receipt recorded" in zero
    assert start_won is not zero_won
    receipts = read_verification_receipts(drive, ctx.task_id)
    pending = custody.pending_invocations(drive)
    if start_won:
        assert "zero_run_requires_settlement" in zero
        assert receipts == []
        assert [row["invocation_id"] for row in pending] == ["actor-claim-invocation"]
    else:
        assert start[1]["reason"] == "zero_run_already_recorded"
        assert len(receipts) == 1 and receipts[0]["zero_run"] is True
        assert pending == []


def test_failed_direct_child_does_not_make_actor_first_terminal_clean(tmp_path):
    from ouroboros.subagent_bootstrap import actor_first_unresolved_fact
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    ctx = types.SimpleNamespace(
        task_id="actor-with-failed-child",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "physical_started": False,
            "exact_start_pending": True,
            "route_available": True,
            "selected_subagent_id": "session-a",
            "work_order_fingerprint": "a" * 64,
        },
    )
    write_task_result(
        tmp_path,
        "failed-child",
        STATUS_FAILED,
        parent_task_id=ctx.task_id,
        root_task_id=ctx.task_id,
        delegation_role="subagent",
        result="provider failed before producing a result",
    )
    # Charter (owner 2026-08-28): children are evidence, never a completion
    # path — the typed reason names the missing leaf, the child statuses ride
    # beside it as evidence.
    fact = actor_first_unresolved_fact(ctx, drive_root=tmp_path)
    assert fact["status"] == "incomplete"
    assert fact["reason"] == "physical_leaf_not_started"
    assert fact["direct_child_statuses"] == [STATUS_FAILED]


def test_discarded_completed_child_does_not_make_actor_first_terminal_clean(tmp_path):
    from ouroboros.subagent_bootstrap import actor_first_unresolved_fact
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.join_ledger import _discard_child_result

    ctx = types.SimpleNamespace(
        task_id="actor-with-discarded-child",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        task_metadata={},
        role="orchestrator",
        _configured_actor_bootstrap={
            "physical_started": False,
            "exact_start_pending": True,
            "route_available": True,
            "selected_subagent_id": "session-a",
            "work_order_fingerprint": "a" * 64,
        },
    )
    write_task_result(
        tmp_path,
        "discarded-child",
        STATUS_COMPLETED,
        parent_task_id=ctx.task_id,
        root_task_id=ctx.task_id,
        delegation_role="subagent",
        result="superseded result",
    )
    assert "Discarded" in _discard_child_result(
        ctx, "discarded-child", "a better branch already produced the result",
    )

    fact = actor_first_unresolved_fact(ctx, drive_root=tmp_path)
    assert fact["status"] == "incomplete"
    assert fact["reason"] == "physical_leaf_not_started"


def test_zero_run_receipt_write_failure_does_not_claim_terminal_truth(tmp_path, monkeypatch):
    from ouroboros.tools import verify as verify_module
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    monkeypatch.setattr(verify_module, "append_verification_receipt", lambda *_a, **_k: False)
    result = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="receipt sink unavailable",
    )
    assert "could not be durably written" in result
    assert ctx._configured_actor_bootstrap.get("zero_run_receipt_recorded") is not True


def test_zero_run_receipt_uses_canonical_budget_root_for_split_drive(tmp_path, monkeypatch):
    from ouroboros import loop as loop_module
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    local = tmp_path / "child-drive"
    canonical = tmp_path / "budget-root"
    local.mkdir()
    canonical.mkdir()
    ctx = _verify_ctx(tmp_path, task_id="split-zero")
    ctx.drive_root = local
    ctx.budget_drive_root = str(canonical)
    ctx.task_constraint = TaskConstraint(
        mode="local_readonly_subagent", allow_enable=False,
    )
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    result = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="canonical root receipt test",
    )
    assert "UNKNOWN" in result
    assert read_verification_receipts(canonical, "split-zero")
    assert read_verification_receipts(local, "split-zero") == []

    # Every active finalization reader follows the same canonical root as the
    # writer. A forked actor must not receive an impossible generic verification
    # reminder after recording its terminal no-leaf decision.
    monkeypatch.setattr(loop_module, "_skill_finalization_message", lambda *_a, **_k: "")
    messages = []
    trace = {
        "reasoning_notes": [],
        "tool_calls": [{"tool": "integrate_subagent_patch", "status": "ok", "args": {}}],
    }
    fired = loop_module._maybe_inject_finalization_nudges(
        types.SimpleNamespace(_ctx=ctx), local, "split-zero",
        trace, "typed zero-run result", messages, lambda *_: None,
    )
    assert fired is False
    assert not any("verify_and_record" in item.get("content", "") for item in messages)


def test_acting_zero_run_keeps_generic_verification_nudge(tmp_path, monkeypatch):
    from ouroboros import loop as loop_module
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.verify import _verify_and_record

    local = tmp_path / "acting-child"
    canonical = tmp_path / "budget-root"
    workspace = tmp_path / "workspace"
    local.mkdir()
    canonical.mkdir()
    workspace.mkdir()
    ctx = _verify_ctx(tmp_path / "acting-ctx", task_id="acting-zero")
    ctx.drive_root = local
    ctx.budget_drive_root = str(canonical)
    ctx.task_constraint = TaskConstraint(
        mode="acting_subagent", surface="external_workspace",
        write_root=str(workspace), allow_enable=False,
    )
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    assert "INCOMPLETE" in _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="incomplete",
        zero_run_basis="host-side integration proceeded without a physical leaf",
    )
    monkeypatch.setattr(loop_module, "_skill_finalization_message", lambda *_a, **_k: "")
    messages = []
    fired = loop_module._maybe_inject_finalization_nudges(
        types.SimpleNamespace(_ctx=ctx), local, "acting-zero",
        {
            "reasoning_notes": [],
            "tool_calls": [{
                "tool": "integrate_subagent_patch", "status": "ok", "args": {},
            }],
        },
        "integrated host result", messages, lambda *_: None,
    )
    assert fired is True
    assert any("verify_and_record" in item.get("content", "") for item in messages)


def test_actor_first_unresolved_terminal_is_typed_degraded_not_clean():
    from ouroboros.outcomes import derive_loop_outcome

    outcome = derive_loop_outcome(
        "A plain answer without a physical start.",
        {
            "actor_first_terminal": {
                "status": "incomplete",
                "reason": "physical_leaf_not_started_and_no_direct_child",
            },
        },
        {"tool_calls": []},
    )
    assert outcome["reason_code"] == "configured_actor_incomplete"
    assert outcome["outcome_axes"]["execution"]["status"] == "degraded"
    assert outcome["outcome_axes"]["objective"]["status"] == "degraded"
    assert outcome["actor_first_terminal"]["status"] == "incomplete"


def test_recorded_zero_run_incomplete_or_unknown_stays_degraded():
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.subagent_bootstrap import actor_first_terminal_projection

    for decision in ("incomplete", "unknown"):
        ctx = types.SimpleNamespace(
            task_id=f"zero-{decision}",
            _configured_actor_bootstrap={
                "zero_run_receipt_recorded": True,
                "zero_run_decision": decision,
                "zero_run_basis": "typed route evidence",
            },
        )
        fact, usage, trace = actor_first_terminal_projection(
            ctx, {"id": ctx.task_id}, {}, {}, None,
        )
        assert fact["status"] == decision
        outcome = derive_loop_outcome("typed zero-run", usage, trace)
        assert outcome["reason_code"] == f"configured_actor_{decision}"
        assert outcome["outcome_axes"]["execution"]["status"] == "degraded"


def test_zero_run_requires_actor_marker_and_basis(tmp_path):
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    assert "TOOL_ARG_ERROR" in _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
        zero_run_basis="not an actor",
    )
    ctx._configured_actor_bootstrap = {"physical_started": False}
    assert "TOOL_ARG_ERROR" in _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="unknown",
    )


def test_verify_and_record_receipt_truncation_is_disclosed(tmp_path, monkeypatch):
    # triad #C (BIBLE P1): a large check output is bounded in the durable receipt but the
    # truncation is DISCLOSED, never silent.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _RECEIPT_OUTPUT_CAP, _verify_and_record

    ctx = _verify_ctx(tmp_path)
    _verify_and_record(ctx, contract_kind="explicit_command", check=[PY, "-c", f"print('x' * {_RECEIPT_OUTPUT_CAP + 5000})"])
    summary = read_verification_receipts(ctx.drive_root, "vhandler")[-1]["summary"]
    assert "truncated" in summary and "chars]" in summary
    assert len(summary) < _RECEIPT_OUTPUT_CAP + 200  # bounded


def test_verify_and_record_declared_receipt_truncation_is_disclosed(tmp_path):
    """Round-9 CRITICAL 2 (BIBLE P1, same invariant as the sibling above).

    The `no_visible_machine_contract` receipt is the honest escape hatch: no host run
    happened, so the agent's OWN stated proxy and residual risk IS the evidence the
    acceptance reviewer judges. It used to be stored with a bare `[:1000]` — a hard,
    silent clip of decision-shaping evidence, and the one durable receipt field in this
    module that bypassed `_bounded`. Bounding is fine; hiding the cut is not.
    """
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _RECEIPT_DECLARED_SUMMARY_CAP, _verify_and_record

    ctx = _verify_ctx(tmp_path)
    long_expected = "proxy: reviewed by hand. residual risk: " + "y" * (
        _RECEIPT_DECLARED_SUMMARY_CAP + 5000
    )
    assert "DECLARED" in _verify_and_record(
        ctx, contract_kind="no_visible_machine_contract",
        check="manual UI review", expected=long_expected,
    )
    summary = read_verification_receipts(ctx.drive_root, "vhandler")[-1]["summary"]
    assert "truncated" in summary and "chars]" in summary, summary[-200:]
    assert str(len(long_expected)) in summary  # the ORIGINAL length is stated
    assert len(summary) < _RECEIPT_DECLARED_SUMMARY_CAP + 200  # still bounded

    # ...and an in-bounds declaration passes through whole, with no marker noise.
    _verify_and_record(
        ctx, contract_kind="no_visible_machine_contract",
        check="manual UI review", expected="proxy: eyeballed the rendered layout",
    )
    short = read_verification_receipts(ctx.drive_root, "vhandler")[-1]["summary"]
    assert short == "proxy: eyeballed the rendered layout" and "truncated" not in short


def test_grounding_statuses_match_handler_vocabulary():
    # F7 (review #1): every grounding status is one the handler can actually emit (no
    # dangling 'recorded'); 'fail' is excluded.
    from ouroboros.outcomes import _RECEIPT_GROUNDING_STATUSES

    assert _RECEIPT_GROUNDING_STATUSES == frozenset({"pass", "observed", "declared"})
    assert "fail" not in _RECEIPT_GROUNDING_STATUSES
    assert "recorded" not in _RECEIPT_GROUNDING_STATUSES


def test_fr2_deep_inheritance_resolves_shared_tree(monkeypatch, tmp_path):
    # F6 (review #1): the deep-inheritance lynchpin — an external_workspace child with
    # an EMPTY write_root inherits the parent's workspace_root (the shared cooperative
    # tree), so a grandchild builds in the same tree.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    from ouroboros.subagent_worktrees import provision_genesis_project
    from supervisor.events import _resolve_subagent_constraint

    handle = provision_genesis_project(repo_dir=str(tmp_path / "repo"), task_id="root")
    shared = handle.path
    ctx = types.SimpleNamespace(repo_dir=str(tmp_path / "repo"))
    requested = {"mode": "acting_subagent", "surface": "external_workspace", "write_root": "", "base_sha": handle.base_sha}
    constraint, resolved_ws, ws_mode, reject = _resolve_subagent_constraint(
        ctx, tid="grandchild", requested_constraint=requested, workspace_root=shared,
        workspace_mode="", base_sha=handle.base_sha, parent_task_id="root",
    )
    assert reject == "", reject
    assert resolved_ws == shared and ws_mode == "external_workspace"
    assert constraint["write_root"] == shared


def test_finalize_schedule_emission_surfaces_coop_tree_path():
    """The host-minted shared coop tree is a SCHEDULE-TIME fact: the parent's
    tool result names the tree path + the root=subagent_projects read recipe,
    so continuation waves stop rediscovering their own tree by trial and error
    (submarine waves). The request-only doctrine (v6.87.28) is untouched: no
    lane/executor resolution rides along."""
    from ouroboros.tools.control import _finalize_schedule_emission

    ctx = types.SimpleNamespace(task_id="p1", drive_root=Path(tempfile.mkdtemp()))
    out = _finalize_schedule_emission(ctx, {
        "task_ids": ["c1"],
        "requested_model_lane": "light",
        "objective": "build",
        "role": "builder",
        "depth": 1,
        "parent_task_id": "p1",
        "root_task_id": "p1",
        "emitted_modes": ["live"],
        "write_surface": "external_workspace",
        "coop_shared_tree": "/tmp/projects/coop_abc123",
    })
    # Windows renders the tree path with os separators — pin label + tree name.
    assert "shared coop tree: " in out and "coop_abc123" in out
    assert "root=subagent_projects" in out
    assert "coop_abc123" in out
    assert "effective_lane=" not in out  # request-only doctrine intact
    # Without a minted tree the result is unchanged.
    out2 = _finalize_schedule_emission(ctx, {
        "task_ids": ["c2"], "requested_model_lane": "light", "objective": "probe",
        "role": "scout", "depth": 1, "parent_task_id": "p1", "root_task_id": "p1",
        "emitted_modes": ["live"],
    })
    assert "shared coop tree" not in out2


def test_zero_run_complete_is_no_longer_writable(tmp_path):
    # Owner N3=A (2026-08-28): a zero-run "complete" is unverifiable self-report
    # and stopped being writable; the write enum is incomplete|unknown.
    from ouroboros.tools.verify import _verify_and_record

    ctx = _verify_ctx(tmp_path)
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    refused = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="complete",
        zero_run_basis="claims to be done without a run",
    )
    assert "TOOL_ARG_ERROR" in refused
    assert "incomplete, unknown" in refused
    from ouroboros.outcomes import read_verification_receipts

    assert read_verification_receipts(tmp_path, ctx.task_id) == []


def test_historical_zero_run_complete_projects_unknown_with_disclosure():
    # Owner R4=A (2026-08-28): a historical "complete" receipt still fences a
    # second start on read, but the terminal projection degrades it to UNKNOWN
    # with disclosure — never a clean terminal.
    from ouroboros.subagent_bootstrap import actor_first_terminal_projection

    ctx = types.SimpleNamespace(
        task_id="actor-historical",
        _configured_actor_bootstrap={
            "zero_run_receipt_recorded": True,
            "zero_run_decision": "complete",
            "zero_run_basis": "pre-charter self-report",
            "route_available": True,
            "physical_started": False,
        },
    )
    fact, usage, trace = actor_first_terminal_projection(
        ctx, {"id": "actor-historical"}, {}, {}, None,
    )
    assert fact is not None
    assert fact["status"] == "unknown"
    assert fact["reason"] == "historical_zero_run_complete"
    assert fact["zero_run_decision"] == "complete"
    assert usage["actor_first_terminal"]["reason"] == "historical_zero_run_complete"
    assert trace["actor_first_terminal"]["status"] == "unknown"
