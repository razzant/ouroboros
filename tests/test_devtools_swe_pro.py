"""SWE-bench and SWE-bench Pro: what the capture takes and what the grade may claim.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
patch capture and the files it refuses to carry, the official evaluation the grade runs,
the tri-state verdicts it reports, the denominator ledger the prediction loop keeps, and
the instance ids it rejects before a path can escape.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys

import pytest

from devtools.benchmarks.swe_bench.presets import resolve_preset

from tests._devtools_benchmarks_shared import (
    REPO_ROOT,
    _git_repo,
)
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


_BASH_CAPTURE_AVAILABLE = sys.platform != "win32" and shutil.which("bash") is not None

def test_swe_pro_e1v2_port_has_csv_option_a_heal_and_no_secrets():
    e1v2 = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2"
    csv_path = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "task_order_pro_70.csv"

    assert csv_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 71
    entrypoint = (e1v2 / "entrypoint_pro.sh").read_text(encoding="utf-8")
    # NW-7 (nq10): the harness-side Option A heal is restored so a dangling
    # committed evolution transaction from the previous task does not poison
    # enqueue for all subsequent tasks (E1v2 -> E1) on agents whose core lacks
    # boot reconciliation. It must keep its merge-base reachability guard so a
    # rolled-back commit is ABANDONED, not falsely marked absorbed. With a
    # newer core's own boot reconciliation it is a harmless no-op.
    assert "Option A:" in entrypoint
    assert "merge-base" in entrypoint and "--is-ancestor" in entrypoint
    assert "boot reconciliation" in entrypoint  # documents the no-op interaction
    assert "/opt/ouroboros-ro/devtools/benchmarks/swe_bench_pro/capture_patch.sh" in entrypoint
    assert '"/opt/capture_patch.sh"' not in (e1v2 / "run_pro.py").read_text(encoding="utf-8")
    assert 'post-task evolution=disabled baseline' in entrypoint
    assert 'reason":"evolution_disabled' in entrypoint
    assert 'if [ "${OBO_SELFIMPROVE:-0}" = "1" ]' in entrypoint
    assert "view_image" in entrypoint
    # owner_chat_id must be seeded BEFORE the budget reset (else native
    # post-task evolution is dropped on fresh volumes -> E1v2 silently == E0).
    assert entrypoint.index('printf \'{"owner_chat_id": 1}\'') < entrypoint.index('reset_per_task_budget("/obo-data"')
    for name in ("settings_base.json", "_run_settings.example.json"):
        payload = json.loads((e1v2 / name).read_text(encoding="utf-8"))
        for key, value in payload.items():
            if any(token in key for token in ("API_KEY", "TOKEN", "PASSWORD", "CREDENTIAL")):
                assert value in ("", None, False), (name, key)
        if name == "settings_base.json":
            assert payload["OUROBOROS_TASK_REVIEW_MODE"] == "required"
            assert payload["OUROBOROS_POST_TASK_EVOLUTION"] == "false"

    from ouroboros.config import SETTINGS_DEFAULTS

    assert SETTINGS_DEFAULTS["OUROBOROS_TASK_REVIEW_MODE"] == "auto"
    run_pro = (e1v2 / "run_pro.py").read_text(encoding="utf-8")
    assert "default fixed-model baseline" in run_pro
    assert "default E1v2 (post-task evolution on)" not in run_pro

def test_swe_pro_e1v2_curve_rows(tmp_path):
    from devtools.benchmarks.swe_bench_pro.e1v2.plot_e1v2_curves import curve_rows, load_e0, load_e1v2_results

    csv_path = tmp_path / "order.csv"
    csv_path.write_text("idx,instance_id,verdict\n1,a,pass\n2,b,fail\n", encoding="utf-8")
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"instance_id":"a","resolved":false}\n{"instance_id":"b","resolved":true}\n', encoding="utf-8")

    rows = curve_rows(load_e0(csv_path), load_e1v2_results(results_path), window=2)

    assert rows[-1]["e0_window_rate"] == 0.5
    assert rows[-1]["e1v2_window_rate"] == 0.5

def test_swe_verified_preset_uses_official_dataset_name():
    assert resolve_preset("verified") == "princeton-nlp/SWE-bench_Verified"
    assert resolve_preset("SWE-bench/SWE-bench_Verified") == "princeton-nlp/SWE-bench_Verified"

@pytest.mark.skipif(not _BASH_CAPTURE_AVAILABLE, reason="capture_patch.sh is a POSIX shell helper; Python wrappers are covered separately")
def test_swe_pro_capture_keeps_untracked_text_and_drops_binary(tmp_path):
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    (repo / "new_file.py").write_text("print('new')\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[tool.example]\nvalue = true\n", encoding="utf-8")
    (repo / "setup.py").write_text("from setuptools import setup\nsetup()\n", encoding="utf-8")
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (repo / "poetry.lock").write_text("# lock\n", encoding="utf-8")
    (repo / "binary.bin").write_bytes(b"\x00\x01\x02\x03")
    (repo / "build").mkdir()
    (repo / "build" / "out.txt").write_text("junk\n", encoding="utf-8")
    (repo / "dist").mkdir()
    (repo / "dist" / "out.txt").write_text("junk\n", encoding="utf-8")
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    capture = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"
    out = tmp_path / "patch.diff"

    subprocess.run(["bash", str(capture), str(repo), base, str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    patch = out.read_text(encoding="utf-8")

    assert "new_file.py" in patch
    assert "pyproject.toml" in patch
    assert "setup.py" in patch
    assert "package-lock.json" not in patch
    assert "poetry.lock" in patch
    assert "app.py" in patch
    assert "binary.bin" not in patch
    assert "build/out.txt" not in patch
    assert "dist/out.txt" not in patch

@pytest.mark.skipif(not _BASH_CAPTURE_AVAILABLE, reason="capture_patch.sh is a POSIX shell helper; Python wrappers are covered separately")
def test_swe_pro_capture_excludes_base_untracked_snapshot(tmp_path):
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    (repo / "auth.yaml").write_text("pre-existing secret-ish fixture\n", encoding="utf-8")
    (repo / "new_agent_file.py").write_text("print('agent-created')\n", encoding="utf-8")
    snapshot = tmp_path / "base_untracked.snapshot"
    snapshot.write_bytes(b"auth.yaml\0")
    capture = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"
    out = tmp_path / "patch.diff"

    subprocess.run(
        ["bash", str(capture), str(repo), base, str(out), str(snapshot)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    patch = out.read_text(encoding="utf-8")
    post_status = (tmp_path / "patch.status.post.txt").read_text(encoding="utf-8")

    assert "auth.yaml" not in patch
    assert "new_agent_file.py" in patch
    assert "auth.yaml" not in post_status
    assert "new_agent_file.py" in post_status

@pytest.mark.skipif(not _BASH_CAPTURE_AVAILABLE, reason="capture_patch.sh is a POSIX shell helper; Python wrappers are covered separately")
def test_swe_pro_capture_preserves_pure_lockfile_patch(tmp_path):
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    capture = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"
    out = tmp_path / "patch.diff"

    subprocess.run(["bash", str(capture), str(repo), base, str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    patch = out.read_text(encoding="utf-8")

    assert "package-lock.json" in patch

@pytest.mark.skipif(not _BASH_CAPTURE_AVAILABLE, reason="capture_patch.sh is a POSIX shell helper; Python wrappers are covered separately")
def test_swe_pro_capture_requires_valid_base_and_external_output(tmp_path):
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    capture = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"

    missing_output = subprocess.run(["bash", str(capture), str(repo), base], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bad_base = subprocess.run(
        ["bash", str(capture), str(repo), "not-a-commit", str(tmp_path / "bad.diff")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    internal_output = REPO_ROOT / "devtools" / "should-not-write.diff"
    internal_dir = REPO_ROOT / "_test_rejected_capture_output_dir"
    nested_internal_output = internal_dir / "out.diff"
    shutil.rmtree(internal_dir, ignore_errors=True)
    try:
        repo_internal = subprocess.run(
            ["bash", str(capture), str(repo), base, str(internal_output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        nested_repo_internal = subprocess.run(
            ["bash", str(capture), str(repo), base, str(nested_internal_output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        internal_output.unlink(missing_ok=True)
        shutil.rmtree(internal_dir, ignore_errors=True)

    assert missing_output.returncode != 0
    assert bad_base.returncode != 0
    assert repo_internal.returncode != 0
    assert "outside the Ouroboros repo" in repo_internal.stderr
    assert nested_repo_internal.returncode != 0
    assert "outside the Ouroboros repo" in nested_repo_internal.stderr
    assert not internal_dir.exists()

def test_swe_pro_grade_runs_official_eval_with_raw_sample(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.grade_pro as grade_pro

    eval_repo = tmp_path / "SWE-bench_Pro-os"
    helper = eval_repo / "helper_code"
    helper.mkdir(parents=True)
    raw_sample = helper / "sweap_eval_full_v2.jsonl"
    raw_sample.write_text(json.dumps({"instance_id": "x", "FAIL_TO_PASS": [], "PASS_TO_PASS": []}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"instance_id": "x", "model_patch": "diff --git a/a b/a\n", "model_name_or_path": "m"}) + "\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grade_pro.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grade_pro.py",
            "--predictions",
            str(predictions),
            "--out-dir",
            str(tmp_path / "out"),
            "--eval-repo",
            str(eval_repo),
        ],
    )

    assert grade_pro.main() == 0
    assert "--raw_sample_path" in captured["cmd"]
    assert str(raw_sample) in captured["cmd"]
    assert captured["cwd"] == str(eval_repo)

def test_swe_pro_grade_rejects_repo_internal_output(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.grade_pro as grade_pro

    eval_repo = tmp_path / "SWE-bench_Pro-os"
    helper = eval_repo / "helper_code"
    helper.mkdir(parents=True)
    raw_sample = helper / "sweap_eval_full_v2.jsonl"
    raw_sample.write_text(json.dumps({"instance_id": "x", "FAIL_TO_PASS": [], "PASS_TO_PASS": []}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"instance_id": "x", "model_patch": "diff --git a/a b/a\n", "model_name_or_path": "m"}) + "\n", encoding="utf-8")
    internal_out = REPO_ROOT / "_test_rejected_grade_output_dir"
    shutil.rmtree(internal_out, ignore_errors=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grade_pro.py",
            "--predictions",
            str(predictions),
            "--out-dir",
            str(internal_out),
            "--eval-repo",
            str(eval_repo),
            "--skip-run",
        ],
    )
    try:
        with pytest.raises(ValueError, match="under repo"):
            grade_pro.main()
        assert not internal_out.exists()
    finally:
        shutil.rmtree(internal_out, ignore_errors=True)

def test_swe_pro_prediction_capture_rejects_empty_patch(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.pro_predictions as pro_predictions

    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "empty.diff"

    def fake_run(cmd, **kwargs):
        out.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(pro_predictions.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="empty patch"):
        pro_predictions._capture_patch(repo, "HEAD", out)

def test_swe_pro_predictions_continue_on_error_writes_denominator_ledger(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.pro_predictions as pro_predictions

    repo = tmp_path / "repo"
    repo.mkdir()
    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    input_jsonl.write_text(
        json.dumps({"instance_id": "case1", "repo_dir": str(repo), "base_commit": "HEAD"}) + "\n",
        encoding="utf-8",
    )

    def fake_capture(repo_dir, base_commit, out_path):
        raise RuntimeError(f"capture_patch.sh produced an empty patch for {repo_dir}")

    monkeypatch.setattr(pro_predictions, "_capture_patch", fake_capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pro_predictions.py",
            "--allow-dirty-seed",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
            "--continue-on-error",
        ],
    )

    assert pro_predictions.main() == 0
    assert output_jsonl.read_text(encoding="utf-8") == ""
    ledger = [json.loads(line) for line in (tmp_path / "predictions.jsonl.ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [json.loads(line) for line in (tmp_path / "predictions.jsonl.errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger[0]["instance_id"] == "case1"
    assert ledger[0]["status"] == "empty_patch"
    assert errors[0]["reason_code"] == "empty_patch"

def test_swe_pro_predictions_fail_fast_marks_remaining_requested_tasks(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.pro_predictions as pro_predictions

    repo = tmp_path / "repo"
    repo.mkdir()
    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    input_jsonl.write_text(
        json.dumps({"instance_id": "case1", "repo_dir": str(repo), "base_commit": "HEAD"})
        + "\n"
        + json.dumps({"instance_id": "case2", "repo_dir": str(repo), "base_commit": "HEAD"})
        + "\n",
        encoding="utf-8",
    )

    def fake_capture(repo_dir, base_commit, out_path):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(pro_predictions, "_capture_patch", fake_capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pro_predictions.py",
            "--allow-dirty-seed",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
        ],
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        pro_predictions.main()
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl.ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["case1", "case2"]
    assert rows[0]["status"] == "failed"
    assert rows[1]["status"] == "not_attempted"
    assert rows[1]["reason_code"] == "aborted_after_prior_error"

def test_swe_predictions_rejects_unsafe_instance_id_before_logs_escape(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench.swebench_predictions as swe_predictions

    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    logs_dir = tmp_path / "logs"
    input_jsonl.write_text(
        json.dumps({"instance_id": "../escape", "workspace_root": "/missing", "problem_statement": "fix"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swebench_predictions.py",
            "--allow-dirty-seed",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
            "--logs-dir",
            str(logs_dir),
            "--continue-on-error",
        ],
    )

    assert swe_predictions.main() == 0
    errors = json.loads((tmp_path / "predictions.jsonl.errors.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ledger = json.loads((tmp_path / "predictions.jsonl.ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert errors["reason_code"] == "invalid_instance_id"
    assert ledger["reason_code"] == "invalid_instance_id"
    assert ledger["status"] == "failed"
    assert not (tmp_path / "escape").exists()

def test_swe_predictions_fail_fast_still_writes_sidecars(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench.swebench_predictions as swe_predictions

    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    input_jsonl.write_text(
        json.dumps({"instance_id": "case1", "workspace_root": "/missing", "problem_statement": "fix"})
        + "\n"
        + json.dumps({"instance_id": "case2", "workspace_root": "/also-missing", "problem_statement": "fix"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swebench_predictions.py",
            "--allow-dirty-seed",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
        ],
    )

    with pytest.raises(RuntimeError, match="workspace_root is not a directory"):
        swe_predictions.main()
    assert output_jsonl.exists()
    assert (tmp_path / "predictions.jsonl.errors.jsonl").exists()
    assert (tmp_path / "predictions.jsonl.ledger.jsonl").exists()
    assert (tmp_path / "predictions.jsonl.run_manifest.json").exists()
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl.ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((tmp_path / "predictions.jsonl.run_manifest.json").read_text(encoding="utf-8"))
    assert [row["instance_id"] for row in ledger_rows] == ["case1", "case2"]
    assert ledger_rows[0]["reason_code"] == "invalid_workspace"
    assert ledger_rows[1]["status"] == "not_attempted"
    assert ledger_rows[1]["reason_code"] == "aborted_after_prior_error"
    assert manifest["requested_task_ids"] == ["case1", "case2"]

def test_swe_pro_predictions_rejects_unsafe_instance_id_before_patch_path(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench_pro.pro_predictions as pro_predictions

    repo = tmp_path / "repo"
    repo.mkdir()
    input_jsonl = tmp_path / "instances.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    patch_dir = tmp_path / "patches"
    input_jsonl.write_text(
        json.dumps({"instance_id": "../escape", "repo_dir": str(repo), "base_commit": "HEAD"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pro_predictions, "_capture_patch", lambda *a, **k: pytest.fail("unsafe id should fail before capture"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pro_predictions.py",
            "--allow-dirty-seed",
            "--input",
            str(input_jsonl),
            "--output",
            str(output_jsonl),
            "--patch-dir",
            str(patch_dir),
        ],
    )

    with pytest.raises(ValueError, match="single safe path component"):
        pro_predictions.main()
    assert not (tmp_path / "escape").exists()

def test_swe_pro_grade_reports_tri_state_verdicts(tmp_path, monkeypatch):
    """Owner Q17=B: an instance the official evaluator never scored is `ungraded`, not a FAIL.
    The official headline FORMULA is unchanged (pass over submitted); `ungraded=N/total` is
    printed next to it and the shrunken-denominator percentage is explicitly labelled
    diagnostic / not leaderboard-valid."""
    import devtools.benchmarks.swe_bench_pro.grade_pro as grade_pro

    eval_repo = tmp_path / "SWE-bench_Pro-os"
    helper = eval_repo / "helper_code"
    helper.mkdir(parents=True)
    (helper / "sweap_eval_full_v2.jsonl").write_text(
        json.dumps({"instance_id": "won", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []}) + "\n"
        + json.dumps({"instance_id": "lost", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []}) + "\n"
        + json.dumps({"instance_id": "crashed", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []}) + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"instance_id": iid, "model_patch": "diff --git a/a b/a\n", "model_name_or_path": "m"})
            for iid in ("won", "lost", "crashed", "not_in_dataset")
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    for iid, tests in (("won", [{"name": "t1", "status": "PASSED"}]), ("lost", [{"name": "t1", "status": "FAILED"}])):
        (out_dir / iid).mkdir(parents=True)
        (out_dir / iid / "ours_output.json").write_text(json.dumps({"tests": tests}), encoding="utf-8")
    # "crashed" has no official output at all -> ungraded, not a model failure.

    monkeypatch.setattr(
        sys, "argv",
        ["grade_pro.py", "--predictions", str(predictions), "--out-dir", str(out_dir),
         "--eval-repo", str(eval_repo), "--skip-run"],
    )
    assert grade_pro.main() == 0

    summary = json.loads((out_dir / "grade_summary.json").read_text(encoding="utf-8"))
    assert summary["submitted"] == 4
    assert summary["pass"] == 1
    assert summary["fail"] == 1
    assert summary["ungraded"] == 2
    assert summary["headline_raw_pass_at_1_pct"] == 25.0          # UNCHANGED formula: 1/4
    assert summary["diagnostic_pass_over_graded_pct"] == 50.0     # 1/2, labelled diagnostic
    assert summary["diagnostic_not_leaderboard_valid"] is True
    by_id = {row["instance_id"]: row for row in summary["verdicts"]}
    assert by_id["won"]["verdict"] == "pass"
    assert by_id["lost"]["verdict"] == "fail"
    assert by_id["crashed"]["verdict"] == "ungraded"
    assert by_id["crashed"]["reason"] == "no_official_output"
    assert by_id["not_in_dataset"]["reason"] == "instance_not_in_dataset"

def test_swe_pro_grade_ungraded_covers_unparseable_and_empty_requirements(tmp_path):
    """The other two ungraded classes: an official output we cannot parse, and a dataset row
    with no required tests (an empty `need` set used to silently read as FAIL)."""
    from devtools.benchmarks.swe_bench_pro.grade_pro import instance_verdict

    broken = tmp_path / "ours_output.json"
    broken.write_text("{not json", encoding="utf-8")
    verdict, reason, _ = instance_verdict(broken, {"FAIL_TO_PASS": ["t1"]})
    assert verdict == "ungraded" and reason.startswith("output_unparseable")

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"tests": [{"name": "t1", "status": "PASSED"}]}), encoding="utf-8")
    assert instance_verdict(empty, {"FAIL_TO_PASS": [], "PASS_TO_PASS": []})[:2] == ("ungraded", "no_required_tests")
    assert instance_verdict(empty, None)[:2] == ("ungraded", "instance_not_in_dataset")

    # Valid JSON with an UNEXPECTED SHAPE is also unparseable output, never a headline: the row
    # extraction has to sit inside the same guard as json.loads (a raised TypeError/KeyError here
    # would abort the whole grading pass).
    for payload in ({"tests": {"t1": "PASSED"}}, {"tests": [{"status": "PASSED"}]}, {"tests": [None]}):
        odd = tmp_path / f"odd_{abs(hash(str(payload)))}.json"
        odd.write_text(json.dumps(payload), encoding="utf-8")
        verdict, reason, column = instance_verdict(odd, {"FAIL_TO_PASS": ["t1"]})
        assert verdict == "ungraded" and reason.startswith("output_unparseable") and column == "-"

def test_swe_pro_manifest_records_the_derived_model_not_the_template(tmp_path, monkeypatch):
    """The manifest must name the model that RAN.

    A live SWE-Pro smoke found `run_manifest.json` reporting `anthropic/claude-sonnet-4.5`
    while `_run_settings.json`, the container environment and the in-container settings all
    agreed the run was on `openai/gpt-5.5`. `model_slot_snapshot` had been handed `--settings`
    — the TEMPLATE — while `derive_run_settings` applies `pin_single_model(--solve-model)` on
    top of it. Nothing in the artefact contradicted an auditor who believed the wrong name,
    which is precisely the failure this release exists to remove.

    Note what a weaker test would have done here: `model_slots["OUROBOROS_MODEL"]` is non-empty
    in the BUGGY case too. So this pins it to the DERIVED file and asserts the two disagree.
    """
    import importlib

    from devtools.benchmarks.common.manifests import model_slot_snapshot

    run_pro = importlib.import_module("devtools.benchmarks.swe_bench_pro.e1v2.run_pro")
    template = tmp_path / "settings_template.json"
    template.write_text(json.dumps({
        "OUROBOROS_MODEL": "anthropic/claude-sonnet-4.5",
        "OUROBOROS_MODEL_HEAVY": "anthropic/claude-sonnet-4.5",
        "TOTAL_BUDGET": 50.0,
    }), encoding="utf-8")
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    derived = run_pro.derive_run_settings(str(template), out_dir, "openai/gpt-5.5", 50.0, 5.0)
    assert derived == out_dir / "_run_settings.json"

    # The container is handed the FILE and a fresh environment, so the launcher's own env is
    # not part of that server's configuration and must not be reported as if it were.
    monkeypatch.setenv("OUROBOROS_MODEL", "some/host-env-model")
    assert model_slot_snapshot(derived, env_overrides=False)["OUROBOROS_MODEL"] == "openai/gpt-5.5"
    # ...and the template, which is what the manifest used to record, names a DIFFERENT model:
    # the exact disagreement the smoke observed.
    assert model_slot_snapshot(template, env_overrides=False)["OUROBOROS_MODEL"] == \
        "anthropic/claude-sonnet-4.5"

    # The call site takes the value `derive_run_settings` RETURNED, not `args.settings`.
    tree = ast.parse((REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2"
                      / "run_pro.py").read_text(encoding="utf-8"))
    snapshots = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", "") == "model_slot_snapshot"]
    assert len(snapshots) == 1
    assert getattr(snapshots[0].args[0], "id", "") == "seed"
    assert [(kw.arg, kw.value.value) for kw in snapshots[0].keywords] == [("env_overrides", False)]
