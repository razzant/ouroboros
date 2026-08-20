"""What a launcher must record about a run it could not finish.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
typed outcome every migrated launcher writes on its failure paths, the exit status that has
to match the recorded exit code, the refusal manifests published exactly once, and the
append-only ledger whose manifest is written first.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


from tests._devtools_benchmarks_shared import _git_repo
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


def test_programbench_e2e_ledger_is_append_only_and_manifest_is_written_first(tmp_path, monkeypatch):
    """P1.5 + P1.2 on the biggest spender: every row is appended the moment it exists (a crash
    used to discard the whole run's ledger, and a resume silently replaced the previous run's
    history), and the manifest — which carries the seed gate — is written BEFORE the first
    instance instead of after the official eval."""
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    run_root = tmp_path / "pb-run"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    instances = [{"instance_id": "inst-a", "image_name": "img-a"}, {"instance_id": "inst-b", "image_name": "img-b"}]
    monkeypatch.setattr(e2e, "_load_instances", lambda **_k: list(instances))
    monkeypatch.setattr(e2e, "runtime_attestation", lambda url, repo: {"ok": True, "runtime_version": "6.75.0"})
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: run_root)

    seen: list[str] = []

    def _fake_process(instance, cfg):
        seen.append(str(instance["instance_id"]))
        # The ledger must already hold the FIRST row while the SECOND instance is still running.
        if len(seen) == 2:
            lines = (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()
            assert [json.loads(line)["instance_id"] for line in lines] == ["inst-a"]
            # The manifest already exists mid-run and carries the seed gate. Assert the gate's
            # SHAPE, never its verdict: `ok` mirrors the ambient checkout, so pinning it to False
            # passes on a developer's dirty tree and fails on a clean CI checkout.
            gate = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))["seed_gate"]
            assert set(gate) >= {"ok", "reason", "require_clean", "allow_dirty_seed", "dirty", "git_available"}
            assert gate["require_clean"] is False and gate["allow_dirty_seed"] is True
            assert gate["ok"] is (not gate["reason"])
        return e2e.task_result_row(
            benchmark="programbench", instance_id=str(instance["instance_id"]),
            status="completed", reason_code="submission_prepared",
        )

    monkeypatch.setattr(e2e, "_process_instance", _fake_process)
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench_e2e.py", "--allow-dirty-seed", "--settings-path", str(settings),
         "--ouroboros-url", "http://127.0.0.1:9"],
    )

    assert e2e.main() == 0
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["inst-a", "inst-b"]

    # A resume APPENDS; readers dedup by instance_id with the last row winning.
    monkeypatch.setattr(e2e, "_load_instances", lambda **_k: [instances[1]])
    monkeypatch.setattr(e2e, "_process_instance", lambda instance, cfg: e2e.task_result_row(
        benchmark="programbench", instance_id="inst-b", status="failed", reason_code="task_not_completed"))
    assert e2e.main() == 1
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["inst-a", "inst-b", "inst-b"]
    latest = {row["instance_id"]: row for row in rows}
    assert latest["inst-b"]["status"] == "failed"
    assert latest["inst-a"]["status"] == "completed"

    # Every processed row reached BOTH ledgers, which is the contract programbench/README.md
    # states without qualification.
    for iid in ("inst-a", "inst-b"):
        per_instance = (run_root / iid / "result_index.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["instance_id"] for line in per_instance] == [iid] * len(per_instance)

    # ... and so does a SKIP row. A resume narrows the work, never the ledger: the instance that
    # is skipped because it already has a submission gets its skip event appended at the run root
    # AND in its own directory. Only the run root was written, so a resumed instance's own history
    # silently omitted the resume while the README claimed both locations.
    submission = run_root / "inst-a" / "submission.tar.gz"
    submission.write_bytes(b"tarball")
    root_before = len((run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines())
    instance_before = len((run_root / "inst-a" / "result_index.jsonl").read_text(encoding="utf-8").splitlines())
    monkeypatch.setattr(e2e, "_load_instances", lambda **_k: list(instances))
    monkeypatch.setattr(e2e, "_process_instance", lambda instance, cfg: e2e.task_result_row(
        benchmark="programbench", instance_id=str(instance["instance_id"]),
        status="completed", reason_code="submission_prepared"))
    assert e2e.main() == 0

    root_rows = [json.loads(line) for line in
                 (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    instance_rows = [json.loads(line) for line in
                     (run_root / "inst-a" / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(root_rows) == root_before + 2          # inst-a skipped + inst-b processed
    assert len(instance_rows) == instance_before + 1
    assert instance_rows[-1]["status"] == "skipped"
    assert instance_rows[-1]["reason_code"] == "skipped_existing_submission"
    assert instance_rows[-1] == next(r for r in root_rows if r["status"] == "skipped")

def test_harness_bench_fast_manifest_is_durable_and_records_the_final_outcome(tmp_path, monkeypatch):
    """The third P1 launcher's half of the manifest lifecycle. It wrote its manifest inline and
    never touched it again, so a run's own record never said how the run ENDED. It is now built
    once, on disk before the harness subprocess starts (asserted from inside the subprocess
    stand-in, i.e. before anything is spent), retained, and rewritten with the final outcome and
    exit code — including the harness's own non-zero exit."""
    from devtools.benchmarks.harness_bench_fast import run_harness_bench_fast as hbf

    out_root = tmp_path / "hbf-run"
    manifest_path = out_root / "run_manifest.json"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hbf, "_read_task_ids", lambda root, ids, task_file="": ["task_1"])

    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["manifest_before_spend"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="")

    monkeypatch.setattr(hbf.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["run_harness_bench_fast.py", "--run-root", str(out_root), "--allow-dirty-seed",
         "--settings-path", str(settings), "--bench-root", str(tmp_path / "bench")],
    )

    assert hbf.main() == 7
    # Durable BEFORE the harness ran, and the seed gate's SHAPE is in it (never its verdict: `ok`
    # mirrors the ambient checkout and would flip between a dirty tree and clean CI).
    early = seen["manifest_before_spend"]
    assert early["extra"]["outcome"] == "started"
    assert set(early["seed_gate"]) >= {"ok", "reason", "require_clean", "allow_dirty_seed"}
    assert early["seed_gate"]["require_clean"] is False

    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert final["extra"]["outcome"] == "harness_nonzero_exit"
    assert final["extra"]["exit_code"] == 7
    assert final["requested_task_ids"] == ["task_1"]

    # A dry run records that it was a dry run rather than leaving `started` behind forever.
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--dry-run"])
    assert hbf.main() == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["extra"]["outcome"] == "dry_run"

def test_benchmark_admission_persists_the_refusal_before_enforcement_raises(tmp_path):
    """The provenance lifecycle is now ENFORCED, not promised.

    `_seed_gate` used to raise from inside `benchmark_run_manifest`, i.e. before the dict reached
    any caller, so no launcher could persist the refusal the contract promises: a refused run left
    nothing but a stderr line that a shard launcher discards. Admission now builds the COMPLETE
    payload, `admit_benchmark_run` writes it, and only then does enforcement raise — and the typed
    exception carries the same payload so any other caller can persist it too.
    """
    from devtools.benchmarks.common.manifests import (
        BenchmarkAdmissionRefused,
        admit_benchmark_run,
    )

    repo = tmp_path / "repo"
    _git_repo(repo)

    admitted_path = tmp_path / "admitted" / "run_manifest.json"
    admitted = admit_benchmark_run(
        admitted_path, benchmark="unit", run_root=tmp_path / "run", repo_dir=repo,
        requested_task_ids=["t"],
    )
    assert admitted["seed_gate"]["ok"] is True
    assert json.loads(admitted_path.read_text(encoding="utf-8"))["seed_gate"]["ok"] is True
    assert "refusal" not in admitted["extra"]

    (repo / "app.py").write_text("print('dirty')\n", encoding="utf-8")
    refused_path = tmp_path / "refused" / "run_manifest.json"
    with pytest.raises(BenchmarkAdmissionRefused, match="reason=seed_dirty") as refused:
        admit_benchmark_run(
            refused_path, benchmark="unit", run_root=tmp_path / "run", repo_dir=repo,
            requested_task_ids=["t"],
        )
    # Still a RuntimeError for every pre-existing caller, and the payload rode on the exception.
    assert isinstance(refused.value, RuntimeError)
    assert refused.value.manifest["seed_gate"]["reason"] == "seed_dirty"

    persisted = json.loads(refused_path.read_text(encoding="utf-8"))
    assert persisted["seed_gate"]["reason"] == "seed_dirty"
    assert persisted["seed_gate"]["ok"] is False
    assert persisted["requested_task_ids"] == ["t"]
    # Same terminal vocabulary a completed run uses, so both read the same way in an audit.
    assert persisted["extra"]["outcome"] == "refused"
    assert persisted["extra"]["exit_code"] == 1
    assert persisted["extra"]["refusal"] == {
        "stage": "seed_gate", "reason": "seed_dirty", "exit_code": 1}

    # The `expect` pin is refused even WITH the dirty-seed escape, and is just as durable.
    pinned_path = tmp_path / "pinned" / "run_manifest.json"
    with pytest.raises(BenchmarkAdmissionRefused, match="reason=seed_mismatch"):
        admit_benchmark_run(
            pinned_path, benchmark="unit", run_root=tmp_path / "run", repo_dir=repo,
            requested_task_ids=["t"], require_clean=False, expect="0" * 40,
        )
    pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    assert pinned["extra"]["refusal"]["reason"] == "seed_mismatch"
    assert pinned["seed_gate"]["allow_dirty_seed"] is True

def test_finalize_run_manifest_records_a_typed_outcome_on_every_exit_path(tmp_path):
    """The ONE finalization seam. Its whole point is the paths a launcher does NOT think about:
    an early typed return and an escaping exception. Several migrated launchers only ever updated
    counts, so their own record still said `started` after they had finished or died."""
    from devtools.benchmarks.common.manifests import finalize_run_manifest

    target = tmp_path / "deep" / "run_manifest.json"

    def _extra():
        return json.loads(target.read_text(encoding="utf-8"))["extra"]

    manifest = {"extra": {"outcome": "started"}}
    with finalize_run_manifest(target, manifest) as final:
        assert final["outcome"] == "completed"
    assert _extra() == {"outcome": "completed", "exit_code": 0}

    manifest = {"extra": {"outcome": "started"}}
    with finalize_run_manifest(target, manifest) as final:
        final.update({"outcome": "refused", "exit_code": 3,
                      "refusal": {"stage": "seed_shape", "reason": "seed_is_not_a_git_directory"}})
    recorded = _extra()
    assert recorded["outcome"] == "refused" and recorded["exit_code"] == 3
    assert recorded["refusal"]["stage"] == "seed_shape"

    manifest = {"extra": {"outcome": "started"}}
    with pytest.raises(ZeroDivisionError):
        with finalize_run_manifest(target, manifest):
            raise ZeroDivisionError("boom")
    recorded = _extra()
    assert recorded["outcome"] == "crashed" and recorded["exit_code"] == 1
    assert recorded["error"] == {"type": "ZeroDivisionError", "message": "boom"}

    # A launcher that NAMED its outcome before re-raising keeps that name; the typed error is
    # recorded NEXT to it rather than replacing it.
    manifest = {"extra": {}}
    with pytest.raises(RuntimeError):
        with finalize_run_manifest(target, manifest) as final:
            final.update({"outcome": "stopped_instance_error", "exit_code": 1})
            raise RuntimeError("instance blew up")
    recorded = _extra()
    assert recorded["outcome"] == "stopped_instance_error"
    assert recorded["error"]["type"] == "RuntimeError"

    # BaseException (SIGINT / SystemExit) must not slip past the seam either.
    manifest = {"extra": {}}
    with pytest.raises(KeyboardInterrupt):
        with finalize_run_manifest(target, manifest):
            raise KeyboardInterrupt
    recorded = _extra()
    assert recorded["outcome"] == "crashed" and recorded["error"]["type"] == "KeyboardInterrupt"

    # ... and a SystemExit keeps its REAL status: flattening it to 1 made the record disagree
    # with the code the process exits with (auto_run's campaign-fatal refusal exits 2).
    manifest = {"extra": {}}
    with pytest.raises(SystemExit):
        with finalize_run_manifest(target, manifest):
            raise SystemExit(2)
    recorded = _extra()
    assert recorded["outcome"] == "crashed" and recorded["exit_code"] == 2
    # A non-integer status (SystemExit("message")) has no numeric meaning -> generic failure.
    manifest = {"extra": {}}
    with pytest.raises(SystemExit):
        with finalize_run_manifest(target, manifest):
            raise SystemExit("no numeric status")
    assert _extra()["exit_code"] == 1

def test_programbench_launcher_records_a_typed_outcome_on_its_failure_path(tmp_path, monkeypatch):
    """Failure path of the per-instance ProgramBench launcher: it only ever wrote
    `failure_reason_code`, so its manifest still claimed the run was `started` after it died."""
    from devtools.benchmarks.programbench import run_programbench as pb

    out_root = tmp_path / "pb"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    instruction = tmp_path / "task.txt"
    instruction.write_text("do it", encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pb, "run_root", lambda *_a, **_k: out_root)

    def _boom(container_name):
        raise RuntimeError("cleanroom container is not running")

    monkeypatch.setattr(pb, "preflight_cleanroom_container", _boom)
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench.py", "--workspace", str(workspace), "--instruction-file",
         str(instruction), "--container-name", "c", "--instance-id", "inst-a",
         "--settings-path", str(settings), "--allow-dirty-seed"],
    )
    with pytest.raises(RuntimeError, match="cleanroom container is not running"):
        pb.main()

    extra = json.loads((out_root / "inst-a" / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "blocked"
    assert extra["exit_code"] == 1
    assert extra["refusal"]["stage"] == "cleanroom_preflight_failed"
    assert extra["error"]["type"] == "RuntimeError"
    assert extra["failure_reason_code"] == "cleanroom_preflight_failed"

def test_programbench_e2e_records_a_typed_outcome_on_its_failure_paths(tmp_path, monkeypatch):
    """Failure paths of the biggest spender: a completed run whose instances failed gets a NAMED
    outcome (not just exit 1), and an instance that RAISES leaves `crashed`, never `started`."""
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    out_root = tmp_path / "pb-e2e"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(e2e, "_load_instances",
                        lambda **_k: [{"instance_id": "inst-a", "image_name": "img-a"}])
    monkeypatch.setattr(e2e, "runtime_attestation", lambda url, repo: {"ok": True})
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: out_root)
    monkeypatch.setattr(e2e, "_process_instance", lambda instance, cfg: e2e.task_result_row(
        benchmark="programbench", instance_id="inst-a", status="failed",
        reason_code="task_not_completed"))
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench_e2e.py", "--allow-dirty-seed", "--settings-path", str(settings),
         "--ouroboros-url", "http://127.0.0.1:9"],
    )
    assert e2e.main() == 1
    extra = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "instances_failed" and extra["exit_code"] == 1

    def _boom(instance, cfg):
        raise RuntimeError("docker exec died")

    monkeypatch.setattr(e2e, "_process_instance", _boom)
    with pytest.raises(RuntimeError, match="docker exec died"):
        e2e.main()
    extra = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "crashed" and extra["error"]["type"] == "RuntimeError"

def test_swebench_predictions_records_a_typed_outcome_when_it_stops_on_an_error(tmp_path, monkeypatch):
    """Failure path of the SWE-bench predictions launcher: it re-raises the first instance error,
    which used to escape with the manifest's `outcome` never written at all."""
    from devtools.benchmarks.swe_bench import swebench_predictions as sp

    input_path = tmp_path / "instances.jsonl"
    input_path.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    output = tmp_path / "preds.jsonl"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sp, "_run_prediction_rows",
                        lambda args, rows, **_k: ([], [], [], RuntimeError("agent never started")))
    monkeypatch.setattr(
        sys, "argv",
        ["swebench_predictions.py", "--input", str(input_path), "--output", str(output),
         "--settings-path", str(settings), "--allow-dirty-seed"],
    )
    with pytest.raises(RuntimeError, match="agent never started"):
        sp.main()

    extra = json.loads(Path(str(output) + ".run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "stopped_instance_error"
    assert extra["exit_code"] == 1
    assert extra["error"]["type"] == "RuntimeError"
    assert extra["prediction_count"] == 0

def test_pro_predictions_records_a_typed_outcome_when_it_stops_on_an_error(tmp_path, monkeypatch):
    """Failure path of the SWE-Pro prediction packer, driven by a REAL malformed input row."""
    from devtools.benchmarks.swe_bench_pro import pro_predictions as pp

    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    output = tmp_path / "preds.jsonl"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["pro_predictions.py", "--input", str(input_path), "--output", str(output),
         "--patch-dir", str(tmp_path / "patches"), "--settings-path", str(settings),
         "--allow-dirty-seed"],
    )
    with pytest.raises(RuntimeError, match="each row must include"):
        pp.main()

    extra = json.loads(Path(str(output) + ".run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "stopped_instance_error"
    assert extra["exit_code"] == 1
    assert extra["error"]["type"] == "RuntimeError"

def test_harness_bench_fast_records_a_crash_instead_of_leaving_started(tmp_path, monkeypatch):
    """The exceptional path of `harness_bench_fast`: its `_finish` helper covered every INTENDED
    exit, so an unhandled failure (missing harness runner) left `outcome: started` forever."""
    from devtools.benchmarks.harness_bench_fast import run_harness_bench_fast as hbf

    out_root = tmp_path / "hbf-run"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hbf, "_read_task_ids", lambda root, ids, task_file="": ["task_1"])

    def _boom(cmd, **kwargs):
        raise FileNotFoundError("harness runner is not installed")

    monkeypatch.setattr(hbf.subprocess, "run", _boom)
    monkeypatch.setattr(
        sys, "argv",
        ["run_harness_bench_fast.py", "--run-root", str(out_root), "--allow-dirty-seed",
         "--settings-path", str(settings), "--bench-root", str(tmp_path / "bench")],
    )
    with pytest.raises(FileNotFoundError):
        hbf.main()

    extra = json.loads((out_root / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "crashed"
    assert extra["exit_code"] == 1
    assert extra["error"]["type"] == "FileNotFoundError"

def _process_status_of(main) -> int:
    """The status a process would exit with, exactly as ``raise SystemExit(main())`` computes it."""
    try:
        return int(main() or 0)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException:
        return 1                      # any other escaping exception: CPython exits 1

def _refusal_case_programbench(tmp_path, monkeypatch):
    from devtools.benchmarks.programbench import run_programbench as pb

    out_root = tmp_path / "pb"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    instruction = tmp_path / "task.txt"
    instruction.write_text("do it", encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    def _boom(container_name):
        raise RuntimeError("cleanroom container is not running")

    monkeypatch.setattr(pb, "run_root", lambda *_a, **_k: out_root)
    monkeypatch.setattr(pb, "preflight_cleanroom_container", _boom)
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench.py", "--workspace", str(workspace), "--instruction-file",
         str(instruction), "--container-name", "c", "--instance-id", "inst-a",
         "--settings-path", str(settings), "--allow-dirty-seed"],
    )
    return pb.main, out_root / "inst-a" / "run_manifest.json"

def _refusal_case_programbench_e2e(tmp_path, monkeypatch):
    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused
    from devtools.benchmarks.programbench import run_programbench_e2e as e2e

    out_root = tmp_path / "pb-e2e"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    def _refuse(url, repo):
        raise RuntimeAttestationRefused("runtime attestation failed reason=runtime_unreachable",
                                        {"reason": "runtime_unreachable", "ok": False})

    monkeypatch.setattr(e2e, "_load_instances",
                        lambda **_k: [{"instance_id": "inst-a", "image_name": "img-a"}])
    monkeypatch.setattr(e2e, "run_root", lambda *_a, **_k: out_root)
    monkeypatch.setattr(e2e, "runtime_attestation", _refuse)
    monkeypatch.setattr(
        sys, "argv",
        ["run_programbench_e2e.py", "--allow-dirty-seed", "--settings-path", str(settings),
         "--ouroboros-url", "http://127.0.0.1:9"],
    )
    return e2e.main, out_root / "run_manifest.json"

def _refusal_case_swebench_predictions(tmp_path, monkeypatch):
    from devtools.benchmarks.swe_bench import swebench_predictions as sp

    input_path = tmp_path / "instances.jsonl"
    input_path.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    output = tmp_path / "preds.jsonl"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sp, "_run_prediction_rows",
                        lambda args, rows, **_k: ([], [], [], RuntimeError("agent never started")))
    monkeypatch.setattr(
        sys, "argv",
        ["swebench_predictions.py", "--input", str(input_path), "--output", str(output),
         "--settings-path", str(settings), "--allow-dirty-seed"],
    )
    return sp.main, Path(str(output) + ".run_manifest.json")

def _refusal_case_pro_predictions(tmp_path, monkeypatch):
    from devtools.benchmarks.swe_bench_pro import pro_predictions as pp

    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    output = tmp_path / "preds.jsonl"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["pro_predictions.py", "--input", str(input_path), "--output", str(output),
         "--patch-dir", str(tmp_path / "patches"), "--settings-path", str(settings),
         "--allow-dirty-seed"],
    )
    return pp.main, Path(str(output) + ".run_manifest.json")

def _refusal_case_harness_bench_fast(tmp_path, monkeypatch):
    from devtools.benchmarks.harness_bench_fast import run_harness_bench_fast as hbf

    out_root = tmp_path / "hbf-run"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hbf, "_read_task_ids", lambda root, ids, task_file="": ["task_1"])
    monkeypatch.setattr(hbf.subprocess, "run",
                        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 7, stdout="", stderr=""))
    monkeypatch.setattr(
        sys, "argv",
        ["run_harness_bench_fast.py", "--run-root", str(out_root), "--allow-dirty-seed",
         "--settings-path", str(settings), "--bench-root", str(tmp_path / "bench")],
    )
    return hbf.main, out_root / "run_manifest.json"

def _refusal_case_run_pro(tmp_path, monkeypatch):
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    out_dir = tmp_path / "out"
    seed = tmp_path / "worktree-seed"
    seed.mkdir()
    (seed / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    monkeypatch.setattr(run_pro, "SRC", seed)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(run_pro, "read_full_order", lambda: ["inst__a"])
    monkeypatch.setattr(run_pro, "load_pro_rows", lambda ids: {})
    monkeypatch.setattr(sys, "argv", ["run_pro.py", "--full-set", "--out-dir", str(out_dir),
                                      "--allow-dirty-seed"])
    return run_pro.main, out_dir / "run_manifest.json"

def _refusal_case_auto_run(tmp_path, monkeypatch):
    from devtools.benchmarks.common.manifests import SeedShapeRefused
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    out_dir = tmp_path / "auto"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    fake_run_pro = SimpleNamespace()

    def _refuse(path):
        raise SeedShapeRefused("seed_is_not_a_git_directory", "no real .git directory")

    fake_run_pro.assert_seed_is_git_directory = _refuse
    fake_run_pro.ensure_util_image = lambda: None
    monkeypatch.setitem(sys.modules, "devtools.benchmarks.swe_bench_pro.e1v2.run_pro", fake_run_pro)
    monkeypatch.setattr(sys, "argv", ["auto_run.py", "--start", "1", "--end", "1",
                                      "--out-dir", str(out_dir), "--allow-dirty-seed"])
    return auto_run.main, out_dir / "auto_run_manifest.json"

def _refusal_case_run_clb(tmp_path, monkeypatch):
    """CL-Bench refuses on the EXECUTION clone's provenance. The clone here is a bare
    non-git directory, so the verdict is a property of the fixture, never of the ambient
    checkout (and no `--allow-dirty-seed`, because the refusal IS what is under test)."""
    from devtools.benchmarks.continual_learning import run_clb

    clone = tmp_path / "execution-clone"
    (clone / "devtools" / "benchmarks" / "common").mkdir(parents=True)
    out = tmp_path / "clb-run"
    monkeypatch.setattr(
        sys, "argv",
        ["run_clb.py", "--ouroboros-clone", str(clone), "--out-dir", str(out), "--dry-run"],
    )
    return run_clb.main, out / "run_manifest.json"

def _refusal_case_run_step_agent(tmp_path, monkeypatch):
    from devtools.benchmarks.osworld import run_step_agent

    repo_dir = tmp_path / "repo"                 # bare dir: no git identity, ambient-free
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    task = tmp_path / "OSWorld" / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv",
        ["run_step_agent.py", "--osworld-root", str(tmp_path / "OSWorld"), "--task", str(task),
         "--result_dir", str(results), "--repo-dir", str(repo_dir),
         "--data-dir", str(tmp_path / "data"), "--settings-path", str(tmp_path / "settings.json"),
         "--ouroboros-url", "http://127.0.0.1:9", "--provider_name", "docker",
         "--model", "m"],
    )
    manifest = (results / "pyautogui" / "screenshot_a11y_tree" / "m" / "chrome" / "abc"
                / "task_run_manifest.json")
    return run_step_agent.main, manifest

def _refusal_case_run_cu_bridge_agent(tmp_path, monkeypatch):
    """Admitted, then refused by the runtime attestation (nothing listens on the URL), so the
    finalization seam — not the admission payload — has to record the real status."""
    from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb

    osworld = tmp_path / "OSWorld"
    (osworld / "evaluation_examples" / "examples" / "chrome").mkdir(parents=True)
    task = osworld / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv",
        ["run_cu_bridge_agent.py", "--osworld-root", str(osworld), "--provider_name", "docker",
         "--path_to_vm", "/vm/Ubuntu.qcow2", "--task", str(task), "--result_dir", str(results),
         "--repo-dir", str(repo_dir), "--data-dir", str(tmp_path / "data"),
         "--settings-path", str(tmp_path / "settings.json"),
         "--ouroboros-url", "http://127.0.0.1:9",
         "--target-file", str(tmp_path / "target.txt"), "--allow-dirty-seed"],
    )
    return rcb.main, results / "chrome" / "abc" / "task_run_manifest.json"

def _refusal_case_run_cu_bridge_agent_seed_gate(tmp_path, monkeypatch):
    """The SEED-GATE refusal, with NO `--claim-dir`, so this attempt OWNS the task.

    Owning it means the launcher keeps two copies of the record: the append-only
    `attempts/<id>/task_run_manifest.json` and the shared canonical
    `run_dir/task_run_manifest.json` a scorer reads — and the case above cannot reach this
    branch at all, because it passes `--allow-dirty-seed`. The seed is a REAL git repo left
    deliberately dirty, so the refusal is a property of the fixture and never of whatever
    checkout (or sandbox layout) the test itself happens to run inside.
    """
    from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb

    osworld = tmp_path / "OSWorld"
    (osworld / "evaluation_examples" / "examples" / "chrome").mkdir(parents=True)
    task = osworld / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    repo_dir = tmp_path / "repo"
    _git_repo(repo_dir)
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")   # uncommitted => seed_dirty
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv",
        ["run_cu_bridge_agent.py", "--osworld-root", str(osworld), "--provider_name", "docker",
         "--path_to_vm", "/vm/Ubuntu.qcow2", "--task", str(task), "--result_dir", str(results),
         "--repo-dir", str(repo_dir), "--data-dir", str(tmp_path / "data"),
         "--settings-path", str(tmp_path / "settings.json"),
         "--ouroboros-url", "http://127.0.0.1:9",
         "--target-file", str(tmp_path / "target.txt")],
    )
    return rcb.main, results / "chrome" / "abc" / "task_run_manifest.json"

def _refusal_case_osworld_adapter_skeleton(tmp_path, monkeypatch):
    from devtools.benchmarks.osworld import osworld_adapter_skeleton as skeleton

    repo_root = tmp_path / "repo"                # bare dir: no git identity, ambient-free
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    output_root = tmp_path / "runs" / "osworld"
    for path in (repo_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    monkeypatch.setattr(skeleton, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(skeleton, "DEFAULT_DATA_ROOT", tmp_path / "live-data")
    monkeypatch.setattr(
        sys, "argv",
        ["osworld_adapter_skeleton.py", "--osworld-root", str(osworld),
         "--ouroboros-url", "http://127.0.0.1:9", "--osworld-server-url", "http://127.0.0.1:9",
         "--unix-computer-use-payload", str(payload), "--output-root", str(output_root)],
    )
    return skeleton.main, output_root / "osworld_preflight.run_manifest.json"

_REFUSAL_CASES = (
    _refusal_case_programbench,
    _refusal_case_programbench_e2e,
    _refusal_case_swebench_predictions,
    _refusal_case_pro_predictions,
    _refusal_case_harness_bench_fast,
    _refusal_case_run_pro,
    _refusal_case_auto_run,
    _refusal_case_run_clb,
    _refusal_case_run_step_agent,
    _refusal_case_run_cu_bridge_agent,
    _refusal_case_run_cu_bridge_agent_seed_gate,
    _refusal_case_osworld_adapter_skeleton,
)

@pytest.mark.parametrize(
    "build_case", _REFUSAL_CASES,
    ids=[case.__name__[len("_refusal_case_"):] for case in _REFUSAL_CASES],
)
def test_migrated_launcher_exit_status_matches_the_recorded_exit_code(build_case, tmp_path, monkeypatch):
    """The manifest's `exit_code` must BE the status the process exits with, per launcher.

    Asserted as a PROPERTY rather than as syntax: each case drives the launcher into a failing
    path and compares the status `raise SystemExit(main())` would produce against the
    `extra.exit_code` the run's own record claims. Recording a code and then letting a plain
    exception escape silently reports 1 instead — which is how three separate review rounds each
    found a fresh instance of the record disagreeing with reality.
    """
    main, manifest_path = build_case(tmp_path, monkeypatch)
    status = _process_status_of(main)
    extra = json.loads(manifest_path.read_text(encoding="utf-8"))["extra"]
    assert status == extra["exit_code"], (
        f"process would exit {status} but the manifest records exit_code={extra['exit_code']} "
        f"(outcome={extra.get('outcome')!r})"
    )
    assert status != 0                      # every case here is a failure path
    assert extra["outcome"] not in ("started", "completed")

def test_cu_bridge_refusal_mirrors_the_terminal_record_to_the_canonical_manifest(
    tmp_path, monkeypatch, capsys
):
    """The SHARED canonical manifest must carry the SAME terminal record as the attempt's own.

    `run_cu_bridge_agent` is the one launcher whose record lives in two places: the attempt's
    append-only copy, which the finalization seam writes, and the canonical
    `run_dir/task_run_manifest.json`, which a separate mirror writes for whichever attempt owns
    the task. The mirror is only correct AFTER the seam's context manager has exited, because
    that exit is what merges the terminal `outcome`/`exit_code`/`refusal` into the manifest.
    The seed-gate branch used to mirror from INSIDE its seam and then `return` past the outer
    `finally`, so the artefact a scorer reads kept the admission seam's GENERIC refusal —
    `exit_code` 1 and no terminal outcome — while the process really exited 2. That is the
    "recorded status != real status" defect this release exists to eliminate, inside the
    machinery built to forbid it, on a path any operator hits with a dirty seed and no
    `--claim-dir`.
    """
    main, canonical = _refusal_case_run_cu_bridge_agent_seed_gate(tmp_path, monkeypatch)
    status = _process_status_of(main)
    capsys.readouterr()
    assert status == 2

    recorded = json.loads(canonical.read_text(encoding="utf-8"))
    extra = recorded["extra"]
    assert recorded["seed_gate"]["reason"] == "seed_dirty"      # the fixture's own verdict
    assert extra["outcome"] == "refused"
    assert extra["exit_code"] == status
    assert extra["refusal"] == {"stage": "seed_gate", "reason": "seed_dirty", "exit_code": status}
    assert extra["allow_dirty_seed"] is False
    # The canonical OUTCOME sidecar names the same refusal in this launcher's own vocabulary.
    outcome = json.loads((canonical.parent / "task_outcome.json").read_text(encoding="utf-8"))
    assert (outcome["status"], outcome["reason_code"]) == ("blocked", "seed_gate_failed")

    # ...and it is byte-for-byte the attempt's OWN record, not merely a plausible one: the two
    # copies of a single run's provenance may never tell different stories about how it ended.
    attempts = sorted((canonical.parent / "attempts").iterdir())
    assert len(attempts) == 1
    assert json.loads((attempts[0] / "task_run_manifest.json").read_text(encoding="utf-8")) == recorded

def test_step_agent_refusal_writes_its_manifest_only_on_admission_and_on_seam_exit(
    tmp_path, monkeypatch, capsys
):
    """Invariant C, behaviourally, on the launcher whose canonical path IS the seam's own.

    `run_step_agent` keeps ONE manifest, so round nine's "is there a second copy that can go
    stale?" sweep cleared it — wrongly, because the hazard is publishing before the merge, which
    a single-path launcher does just as readily. `_write_task_records` wrote the manifest from
    inside the seam, so a reader could observe `exit_code` 1 on a run that exits 2, and an
    interruption in that window left it durable.

    The property is the WRITE SEQUENCE at that path: the deliberate admission record, then the
    seam's terminal write on exit, and nothing in between. Asserting only the final content
    passes just as happily with an extra pre-merge publication.
    """
    from devtools.benchmarks.common import manifests
    from devtools.benchmarks.osworld import run_step_agent

    main, manifest_path = _refusal_case_run_step_agent(tmp_path, monkeypatch)
    target = manifest_path.resolve(strict=False)
    writes: list[dict] = []
    real_write_json = manifests.write_json

    def _recording_write_json(path, payload):
        if Path(path).resolve(strict=False) == target:
            writes.append(json.loads(json.dumps(payload)))      # snapshot exactly AS WRITTEN
        return real_write_json(path, payload)

    # Both bindings: the seam writes through `manifests`, the launcher through its own import,
    # so watching one name only would miss half the writes to the very path under test.
    monkeypatch.setattr(manifests, "write_json", _recording_write_json)
    monkeypatch.setattr(run_step_agent, "write_json", _recording_write_json)
    assert _process_status_of(main) == 2
    capsys.readouterr()

    states = [((w.get("extra") or {}).get("outcome"), (w.get("extra") or {}).get("exit_code"))
              for w in writes]
    # The admission record is written BEFORE any seam is open and is deliberately durable — that
    # is the whole point of `admit_benchmark_run`. Any write BETWEEN it and the seam's exit is
    # the forbidden pre-merge publication; before the fix there were three.
    assert states == [("refused", 1), ("refused", 2)], states

def test_cu_bridge_refusal_publishes_the_canonical_manifest_exactly_once(
    tmp_path, monkeypatch, capsys
):
    """The canonical manifest is published ONCE, after the seam — never in a pre-merge state.

    Asserting the FINAL content is not enough: it passes just as happily when the record is
    published TWICE — first from inside `finalize_run_manifest` carrying the admission seam's
    generic `exit_code` 1, then corrected on seam exit — which is exactly how this window
    survived the round that fixed the final artefact. The intermediate publish is observable:
    OSWorld ships multi-lane in this release, the canonical path is what a concurrent reader
    consumes, and an interruption inside the window leaves the wrong record durably. So the
    property under test is the WRITE SEQUENCE at that path, not its last element.
    """
    from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb

    main, canonical = _refusal_case_run_cu_bridge_agent_seed_gate(tmp_path, monkeypatch)
    target = canonical.resolve(strict=False)
    published: list[dict] = []
    real_write_json = rcb.write_json

    def _recording_write_json(path, payload):
        if Path(path).resolve(strict=False) == target:
            published.append(json.loads(json.dumps(payload)))      # snapshot exactly AS WRITTEN
        return real_write_json(path, payload)

    monkeypatch.setattr(rcb, "write_json", _recording_write_json)
    assert _process_status_of(main) == 2
    capsys.readouterr()

    states = [((p.get("extra") or {}).get("outcome"), (p.get("extra") or {}).get("exit_code"))
              for p in published]
    assert len(published) == 1, f"canonical manifest published {len(published)} times: {states}"
    # ...and the single published state is the real one, so no reader can ever observe a record
    # disagreeing with the status the process exits with.
    assert states == [("refused", 2)]
