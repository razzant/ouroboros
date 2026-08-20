"""The shared benchmark scaffolding every harness in devtools/benchmarks stands on.

This module owns the boundary that keeps the runtime from importing devtools, the official
command builders that may not replace scoring, the run manifest and its provenance, the
seed gate that fails closed, the atomic JSON write, the output helpers that refuse a
repo-internal destination, and the packaging of the devtools entrypoints themselves.

The per-harness suites were split verbatim into ``tests/test_devtools_gaia.py``,
``tests/test_devtools_programbench.py``, ``tests/test_devtools_swe_pro.py``,
``tests/test_devtools_osworld.py``, ``tests/test_devtools_terminal_bench.py`` and
``tests/test_devtools_harbor_jobs.py``; the launcher contracts into
``tests/test_devtools_launcher_gate.py``, ``tests/test_devtools_launcher_outcomes.py`` and
``tests/test_devtools_runtime_attestation.py``. The git helpers they share live in
``tests/_devtools_benchmarks_shared.py``.
"""

from __future__ import annotations

import inspect
import importlib.util
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from devtools.benchmarks.common.official_commands import programbench_eval_cmd, swebench_eval_cmd
from devtools.benchmarks.common.manifests import benchmark_run_manifest, repo_provenance

from tests._devtools_benchmarks_shared import (
    REPO_ROOT,
    _git_commit_all,
    _git_repo,
)
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


def test_runtime_core_does_not_import_devtools():
    runtime_paths = [REPO_ROOT / "ouroboros", REPO_ROOT / "server.py"]
    offenders: list[str] = []
    for root in runtime_paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "import devtools" in text or "from devtools" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders

def test_official_command_builders_do_not_replace_scoring(monkeypatch):
    from devtools.benchmarks.common import official_commands

    monkeypatch.setattr(official_commands, "resolve_programbench_cli", lambda: ["/opt/homebrew/bin/programbench"])
    monkeypatch.delenv("PROGRAMBENCH_DOCKER_CPUS", raising=False)
    # The builders stringify the Path via str(); compare against the platform
    # spelling so the argv-structure assertion stays valid on Windows too
    # (str(Path("/runs/pb")) == "\\runs\\pb" there).
    pb_run = str(Path("/runs/pb"))
    preds = str(Path("/runs/predictions.jsonl"))
    assert programbench_eval_cmd(Path("/runs/pb")) == [
        "/opt/homebrew/bin/programbench",
        "eval",
        pb_run,
        "--docker-cpus",
        "4",
    ]
    assert swebench_eval_cmd("princeton-nlp/SWE-bench_Verified", Path("/runs/predictions.jsonl"), "ouroboros", 2) == [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "princeton-nlp/SWE-bench_Verified",
        "--predictions_path",
        preds,
        "--max_workers",
        "2",
        "--run_id",
        "ouroboros",
    ]

def test_benchmark_manifest_records_provenance_without_diff_text(tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")

    provenance = repo_provenance(repo)
    # require_clean=False: this test asserts the provenance RECORD on a deliberately dirty
    # checkout. The gate itself (default require_clean=True) is asserted separately below.
    manifest = benchmark_run_manifest(
        benchmark="unit",
        run_root=tmp_path / "run",
        repo_dir=repo,
        requested_task_ids=["task-1"],
        require_clean=False,
        metadata={"argv": ["bench", "--task", "task-1"]},
    )

    assert provenance["dirty"] is True
    assert provenance["tracked_diff_sha256"]
    assert "print('changed')" not in json.dumps(provenance)
    assert manifest["requested_count"] == 1
    assert manifest["source"]["tracked_diff_sha256"]
    assert manifest["seed_gate"] == {
        "require_clean": False,
        "allow_dirty_seed": True,
        "expect": "",
        "git_available": True,
        "status_available": True,
        "dirty": True,
        "describe": manifest["source"]["describe"],
        "reason": "seed_dirty",
        "ok": False,
    }

def test_benchmark_common_helpers_keep_compact_api_surface():
    from devtools.benchmarks.common.result_index import task_result_row

    manifest_params = inspect.signature(benchmark_run_manifest).parameters
    row_params = inspect.signature(task_result_row).parameters

    assert len(manifest_params) <= 8
    assert len(row_params) <= 8

def test_benchmark_manifest_model_slots_cover_runtime_model_settings():
    from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS
    from ouroboros.config import SETTINGS_DEFAULTS

    # These match the OUROBOROS_MODEL* prefix but are a concurrency CAP / slot-wait
    # CEILING, not model-id slots, so they are not part of the model-slot manifest.
    _non_model_slot = {"OUROBOROS_MODEL_MAX_CONCURRENCY", "OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC"}
    relevant = {
        key
        for key in SETTINGS_DEFAULTS
        if key not in _non_model_slot
        and (
            key.startswith("OUROBOROS_MODEL")
            or key in {"CLAUDE_CODE_MODEL", "OUROBOROS_WEBSEARCH_MODEL", "OUROBOROS_REVIEW_MODELS"}
            or key.startswith("OUROBOROS_SCOPE_REVIEW_MODEL")
        )
    }

    assert relevant.issubset(set(MODEL_SLOT_KEYS))

def test_benchmark_default_paths_derive_from_workspace_root(monkeypatch):
    from devtools.benchmarks.common import run_roots
    from devtools.benchmarks.common import secrets

    monkeypatch.delenv("OUROBOROS_BENCH_RUNS_ROOT", raising=False)
    monkeypatch.delenv("OUROBOROS_SETTINGS_PATH", raising=False)

    workspace = REPO_ROOT.parent
    assert run_roots.DEFAULT_BENCH_RUNS_ROOT == workspace / "bench_runs"
    assert run_roots.default_settings_path() == workspace / "data" / "settings.json"
    assert secrets.settings_path() == workspace / "data" / "settings.json"

def test_benchmark_manifest_explicit_falsy_kwargs_override_metadata(tmp_path):
    repo = tmp_path / "repo"
    _git_repo(repo)

    manifest = benchmark_run_manifest(
        benchmark="unit",
        run_root=tmp_path / "run",
        repo_dir=repo,
        requested_task_ids=["task-1"],
        argv=[],
        dataset="",
        isolated_data_root="",
        metadata={"argv": ["stale"], "dataset": "stale-ds", "isolated_data_root": "/tmp/stale"},
    )

    assert manifest["argv"] == []
    assert manifest["dataset"] == ""
    assert manifest["isolated_data_root"] == ""

def test_task_result_row_explicit_falsy_kwargs_override_metadata():
    from devtools.benchmarks.common.result_index import task_result_row

    row = task_result_row(
        benchmark="unit",
        instance_id="task-1",
        status="failed",
        reason_code="",
        prediction_written=False,
        official_eval_status="not_run",
        error="",
        metadata={
            "reason_code": "stale_success",
            "prediction_written": True,
            "official_eval_status": "completed",
            "error": "stale",
        },
    )

    assert row["reason_code"] == ""
    assert row["prediction_written"] is False
    assert row["official_eval_status"] == "not_run"
    assert row["error"] == ""

def test_pyproject_does_not_package_devtools_runtime_assets():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"devtools*"' not in pyproject
    assert "devtools = [" not in pyproject
    assert '"benchmarks/**/*.sh"' not in pyproject
    assert '"benchmarks/**/*.md"' not in pyproject

def test_executable_devtools_entrypoints_support_direct_help():
    scripts = [
        "devtools/benchmarks/programbench/run_programbench.py",
        "devtools/benchmarks/programbench/run_programbench_e2e.py",
        "devtools/benchmarks/programbench/export_programbench_submissions.py",
        "devtools/benchmarks/harness_bench_fast/ouroboros_cli_wrapper.py",
        "devtools/benchmarks/terminal_bench/run_harbor_smoke.py",
        "devtools/benchmarks/terminal_bench/run_tb.py",
        "devtools/benchmarks/swe_bench/swebench_predictions.py",
        "devtools/benchmarks/swe_bench_pro/grade_pro.py",
        "devtools/benchmarks/swe_bench_pro/pro_predictions.py",
        "devtools/benchmarks/swe_bench_pro/e1v2/auto_run.py",
        "devtools/benchmarks/swe_bench_pro/e1v2/build_predictions.py",
        "devtools/benchmarks/swe_bench_pro/e1v2/plot_e1v2_curves.py",
        "devtools/benchmarks/swe_bench_pro/e1v2/run_pro.py",
        "devtools/benchmarks/gaia/run_gaia.py",
        "devtools/benchmarks/gaia/score_gaia.py",
        "devtools/benchmarks/osworld/normalize_logs.py",
        "devtools/benchmarks/osworld/osworld_adapter_skeleton.py",
        "devtools/benchmarks/osworld/run_step_agent.py",
    ]
    for rel in scripts:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / rel), "--help"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        assert proc.returncode == 0, f"{rel} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        assert "usage:" in proc.stdout.lower()

def test_harness_bench_fast_wrapper_builds_ouroboros_run_command():
    # The upgraded harness-bench-fast wrapper builds the `ouroboros run` command inline in
    # main() (per-task logs, retries, --result-json-out, --start). Verify the command shape
    # and the v6.39 Phase-2 slot rename (HEAVY/FALLBACKS, never the legacy CODE/FALLBACK).
    from devtools.benchmarks.harness_bench_fast import ouroboros_cli_wrapper as w

    assert hasattr(w, "main")
    src = (
        REPO_ROOT / "devtools" / "benchmarks" / "harness_bench_fast" / "ouroboros_cli_wrapper.py"
    ).read_text(encoding="utf-8")
    for token in ('"run",', '"--memory-mode",', '"--quiet",', '"--result-json-out",', '"--actor-id",'):
        assert token in src, token
    assert '"OUROBOROS_MODEL_HEAVY": args.model' in src
    assert "OUROBOROS_MODEL_CODE" not in src

def test_benchmark_output_helpers_reject_repo_internal_outputs(tmp_path, monkeypatch):
    import devtools.benchmarks.swe_bench.swebench_predictions as swe_predictions
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke
    from devtools.benchmarks.common.run_roots import ensure_file_output_outside_repo

    input_jsonl = tmp_path / "instances.jsonl"
    input_jsonl.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["swebench_predictions.py", "--allow-dirty-seed", "--input", str(input_jsonl), "--output", str(REPO_ROOT / "devtools" / "bad.jsonl")])
    with pytest.raises(ValueError, match="benchmark run output must not be under repo"):
        swe_predictions.main()

    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(REPO_ROOT / "devtools" / "bad_run")])
    with pytest.raises(ValueError, match="benchmark run output must not be under repo"):
        harbor_smoke.main()

    live_data = tmp_path / "live-data"
    live_data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(live_data))
    with pytest.raises(ValueError, match="live runtime data"):
        ensure_file_output_outside_repo(live_data / "bench" / "result_index.jsonl", REPO_ROOT)

    monkeypatch.setattr(sys, "argv", ["swebench_predictions.py", "--allow-dirty-seed", "--input", str(input_jsonl), "--output", str(live_data / "predictions.jsonl")])
    with pytest.raises(ValueError, match="live runtime data"):
        swe_predictions.main()

def test_epistemic_rule_stays_out_of_the_global_system_prompt():
    """Owner Q20/Q22 scoped the rule to the GAIA adapter ONLY: no global grounding duty in
    `prompts/SYSTEM.md` (it would push the runtime into searching for trivia) and no typed
    contract field. This is the invariant that keeps a future 'while we are here' edit honest."""
    system_md = (REPO_ROOT / "prompts" / "SYSTEM.md").read_text(encoding="utf-8").lower()
    for banned in (
        "epistemic honesty",
        "source your external claims",
        "source your claims",
        "cite a primary source",
        "check it against a primary source",
    ):
        assert banned not in system_md, f"SYSTEM.md must not carry the GAIA grounding rule: {banned}"

    contracts = (REPO_ROOT / "ouroboros" / "contracts" / "task_contract.py").read_text(encoding="utf-8")
    assert "epistemic" not in contracts.lower(), "Q20/Q22 explicitly rejected a typed contract field"

def test_benchmark_manifest_seed_gate_fails_closed_by_default(tmp_path):
    """Owner Q19=B: an unreproducible seed refuses the run BY DEFAULT, with a recorded escape.

    Three refusal classes, all before any paid task: a dirty working tree (the manifest would
    say `-dirty` and the run would not be submittable), a checkout with no git identity at all
    (the source cannot be named), and a seed that does not match an explicit `expect` pin. The
    `expect` mismatch is NOT waivable by --allow-dirty-seed: 'dirty' and 'wrong commit' are
    different facts.
    """
    repo = tmp_path / "repo"
    _git_repo(repo)
    clean = benchmark_run_manifest(
        benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
    )
    assert clean["seed_gate"]["ok"] is True
    assert clean["seed_gate"]["require_clean"] is True
    assert clean["seed_gate"]["allow_dirty_seed"] is False

    head = clean["source"]["head"]
    pinned = benchmark_run_manifest(
        benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        expect=head[:12],
    )
    assert pinned["seed_gate"]["expect"] == head[:12]

    (repo / "app.py").write_text("print('dirty')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reason=seed_dirty"):
        benchmark_run_manifest(
            benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        )
    waived = benchmark_run_manifest(
        benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        require_clean=False,
    )
    assert waived["seed_gate"]["reason"] == "seed_dirty"
    assert waived["seed_gate"]["allow_dirty_seed"] is True

    with pytest.raises(RuntimeError, match="reason=seed_mismatch"):
        benchmark_run_manifest(
            benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
            require_clean=False, expect="0" * 40,
        )

    not_git = tmp_path / "plain"
    not_git.mkdir()
    with pytest.raises(RuntimeError, match="reason=seed_identity_unavailable"):
        benchmark_run_manifest(
            benchmark="unit", run_root=tmp_path / "run", repo_dir=not_git, requested_task_ids=["t"],
        )

def test_benchmark_seed_gate_refuses_when_cleanliness_cannot_be_determined(tmp_path):
    """The fourth refusal class: the cleanliness probe itself did not answer.

    `git status` can fail for real (a corrupt `.git/index`, or the 10s timeout on a huge
    untracked tree / CephFS). Coercing that into `dirty: False` let a genuinely dirty seed pass
    the gate with `seed_gate.ok: true`, which is exactly the `-dirty`-provenance run owner
    Q19=B exists to prevent. Reproduced with a REAL corrupted index, not a mock: `rev-parse
    HEAD` still works (so the seed has an identity) while `status` fails.
    """
    repo = tmp_path / "repo"
    _git_repo(repo)
    (repo / "app.py").write_text("print('dirty and unreportable')\n", encoding="utf-8")
    (repo / ".git" / "index").write_bytes(b"DIRC\x00\x00\x00\xffnot-an-index")

    provenance = repo_provenance(repo)
    assert provenance["git_available"] is True          # the commit is still readable
    assert provenance["status_available"] is False      # the cleanliness probe is not
    assert provenance["dirty"] is False                 # ... and its value carries no information

    with pytest.raises(RuntimeError, match="reason=seed_status_unavailable"):
        benchmark_run_manifest(
            benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        )
    # The recorded escape keeps working and keeps saying WHY it was needed.
    waived = benchmark_run_manifest(
        benchmark="unit", run_root=tmp_path / "run", repo_dir=repo, requested_task_ids=["t"],
        require_clean=False,
    )
    assert waived["seed_gate"]["reason"] == "seed_status_unavailable"
    assert waived["seed_gate"]["ok"] is False
    assert waived["seed_gate"]["status_available"] is False

def test_benchmark_write_json_is_atomic_and_byte_identical(tmp_path):
    """write_json became atomic without changing a single byte of any existing sidecar.

    The atomic helper defaults to NO trailing newline, so the call must pass
    trailing_newline=True — otherwise every manifest/ledger sidecar silently changes shape.
    Also asserts no temp sibling survives a successful write.
    """
    from devtools.benchmarks.common.manifests import write_json

    payload = {"b": 1, "a": ["x", "ю"], "nested": {"k": None}}
    target = tmp_path / "deep" / "run_manifest.json"
    write_json(target, payload)
    legacy = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    assert target.read_text(encoding="utf-8") == legacy
    assert sorted(p.name for p in target.parent.iterdir()) == ["run_manifest.json"]

    write_json(target, {"replaced": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"replaced": True}

def test_benchmark_manifests_module_stays_stdlib_only_at_import():
    """`common/manifests.py` is imported by every launcher, including the container-side
    Terminal-Bench agent, so the atomic-write dependency on the runtime package must be a LAZY
    import inside write_json — a module-level `import ouroboros` would make the runtime a hard
    dependency of all benchmark families."""
    source = (REPO_ROOT / "devtools" / "benchmarks" / "common" / "manifests.py").read_text(encoding="utf-8")
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "ouroboros" in line
    ]
    assert module_level == []

    # Cross-launcher import smoke: every P1-owned launcher imports the shared module cleanly.
    for module in (
        "devtools.benchmarks.common.manifests",
        "devtools.benchmarks.programbench.run_programbench",
        "devtools.benchmarks.programbench.run_programbench_e2e",
        "devtools.benchmarks.swe_bench.swebench_predictions",
        "devtools.benchmarks.swe_bench_pro.pro_predictions",
        "devtools.benchmarks.harness_bench_fast.run_harness_bench_fast",
    ):
        importlib.import_module(module)

def test_openrouter_key_remaining_uses_authoritative_field(monkeypatch):
    """`limit_remaining` is the source of truth; `limit - usage` is only a FALLBACK, and an
    uncapped key is None (not 0.0, not 'plenty'). The credit-endpoint arithmetic this replaces
    lied on a nearly exhausted key and burned half a run."""
    from devtools.benchmarks.common.manifests import openrouter_key_remaining

    bodies: list[bytes] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return bodies.pop(0)

    def fake_urlopen(req, timeout=0):
        assert req.full_url == "https://openrouter.ai/api/v1/key"
        assert req.headers["Authorization"] == "Bearer or-key"
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    bodies.append(b'{"data":{"limit":100,"usage":97.5,"limit_remaining":0.23}}')
    assert openrouter_key_remaining("or-key") == 0.23
    bodies.append(b'{"data":{"limit":100,"usage":97.5}}')
    assert openrouter_key_remaining("or-key") == pytest.approx(2.5)
    bodies.append(b'{"data":{"limit":null,"usage":12.0}}')
    assert openrouter_key_remaining("or-key") is None

    with pytest.raises(RuntimeError, match="requires an API key"):
        openrouter_key_remaining("  ")

def test_gaia_and_tb_launchers_run_the_shared_seed_gate(tmp_path, monkeypatch):
    """P5.4: GAIA and TB dropped their v6.75.0 `require_clean=False` pins AND route both manifest
    seams, so the refusal is DURABLE: the record reaches disk and no other artefact does. Asserting
    only `pytest.raises` is what let an inert handler pass review, so every launcher's PERSISTED
    outcome is checked here. Deterministic — the gate runs against a PURPOSE-BUILT dirty repo,
    never the ambient checkout."""
    import devtools.benchmarks.gaia.run_gaia as run_gaia
    from devtools.benchmarks.common.manifests import BenchmarkAdmissionRefused
    from devtools.benchmarks.terminal_bench import run_harbor_smoke, run_tb

    seed = tmp_path / "seed"
    _git_repo(seed)
    (seed / "VERSION").write_text("6.79.0\n", encoding="utf-8")
    _git_commit_all(seed)
    monkeypatch.setattr(run_gaia, "REPO", seed)
    monkeypatch.setattr(run_tb, "repo_root_from_devtools", lambda: seed)
    monkeypatch.setattr(run_harbor_smoke, "repo_root_from_devtools", lambda: seed)
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    def _extra(run_dir):
        return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["extra"]

    # Clean seed: GAIA admits, records the gate verdict, augments the manifest with the
    # settings-derived slots, and the finalization seam names the terminal outcome.
    clean = tmp_path / "clean"
    assert run_gaia.main(["--out-dir", str(clean), "--solve-model", "m", "--dry-run"]) == 0
    manifest = json.loads((clean / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed_gate"]["ok"] is True and manifest["seed_gate"]["require_clean"] is True
    assert manifest["model_slots"]["OUROBOROS_MODEL"] == "m"
    assert manifest["extra"]["outcome"] == "dry_run" and manifest["extra"]["exit_code"] == 0

    # Dirty seed: refused before anything is spent -- and the REFUSAL is on disk, so a shard
    # wrapper reading run_manifest.json can tell "refused" from "never started" or "crashed".
    (seed / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")

    dirty = tmp_path / "dirty-gaia"
    with pytest.raises(BenchmarkAdmissionRefused, match="seed_dirty"):
        run_gaia.main(["--out-dir", str(dirty), "--solve-model", "m", "--dry-run"])
    extra = _extra(dirty)
    assert extra["outcome"] == "refused" and extra["exit_code"] == 1
    assert extra["refusal"] == {"stage": "seed_gate", "reason": "seed_dirty", "exit_code": 1}
    # The renderer that injects LIVE provider keys into the run dir never ran.
    assert not (dirty / "settings.json").exists()

    def _no_probe(_harbor_bin):
        raise AssertionError("harbor --version was probed before admission")

    monkeypatch.setattr(run_tb, "harbor_version", _no_probe)
    tb_root = tmp_path / "dirty-tb"
    with pytest.raises(BenchmarkAdmissionRefused, match="seed_dirty"):
        run_tb.main(["--model", "m", "--run-root", str(tb_root), "--settings-path", str(settings)])
    assert _extra(tb_root)["refusal"]["stage"] == "seed_gate"
    # No half-built submission tree (no job dir, no metadata.yaml).
    assert not (tb_root / "submission" / "submissions").exists()

    smoke_root = tmp_path / "dirty-smoke"
    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--run-root", str(smoke_root),
                                      "--settings-path", str(settings)])
    with pytest.raises(BenchmarkAdmissionRefused, match="seed_dirty"):
        run_harbor_smoke.main()
    assert _extra(smoke_root)["refusal"]["stage"] == "seed_gate"
    assert not (smoke_root / "harbor_command.json").exists()
    assert not (smoke_root / "result_index.jsonl").exists()

    # ...unless the escape is recorded.
    escaped = tmp_path / "escaped"
    assert run_gaia.main(["--out-dir", str(escaped), "--solve-model", "m", "--dry-run",
                          "--allow-dirty-seed"]) == 0
    recorded = json.loads((escaped / "run_manifest.json").read_text(encoding="utf-8"))
    assert recorded["seed_gate"]["allow_dirty_seed"] is True
    assert recorded["seed_gate"]["reason"] == "seed_dirty"
    assert recorded["extra"]["outcome"] == "dry_run"
