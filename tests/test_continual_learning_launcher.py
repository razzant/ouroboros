"""Schema/CLI-level tests for the CL-Bench launcher (no network, no docker)."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from devtools.benchmarks.continual_learning import run_clb

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_BASE = REPO_ROOT / "devtools" / "benchmarks" / "continual_learning" / "settings_base.json"


@pytest.fixture(autouse=True)
def _isolate_bench_runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_BENCH_RUNS_ROOT", str(tmp_path / "bench_runs"))
    monkeypatch.delenv("CLBENCH_RUNNER_PATH", raising=False)
    monkeypatch.delenv("OUROBOROS_BENCH_CLONE", raising=False)


def _fake_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "continual-learning-bench"
    adapter = runner / "src" / "systems" / "ouroboros"
    adapter.mkdir(parents=True)
    (runner / "run_benchmark.py").write_text("# stub\n", encoding="utf-8")
    (adapter / "system.py").write_text("# stub\n", encoding="utf-8")
    (adapter / "run_clbench_bridge_agent.py").write_text("# stub\n", encoding="utf-8")
    return runner


def _fake_clone(tmp_path: Path) -> Path:
    clone = tmp_path / "ouroboros-bench-src"
    common = clone / "devtools" / "benchmarks" / "common"
    common.mkdir(parents=True)
    (common / "server_runner.py").write_text("# stub\n", encoding="utf-8")
    return clone


def test_settings_template_contract():
    settings = json.loads(SETTINGS_BASE.read_text(encoding="utf-8"))
    # Sprint bench-template decisions.
    assert settings["OUROBOROS_MAX_WORKERS"] == 4
    assert settings["OUROBOROS_SAFETY_MODE"] == "light"
    assert settings["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
    assert settings["OUROBOROS_POST_TASK_EVOLUTION"] == "false"
    assert "claude_code_edit" in settings["CLBENCH_SOLVE_DISABLED_TOOLS"]
    # The declared solve denylist must cover the registry's REAL web-tool set
    # (cumulative review r2: youtube_transcript had drifted out) and must not
    # carry names that are not actual tools.
    from ouroboros.tools.registry import _WEB_TOOLS
    assert set(_WEB_TOOLS) <= set(settings["CLBENCH_SOLVE_DISABLED_TOOLS"])
    assert "screenshot" not in settings["CLBENCH_SOLVE_DISABLED_TOOLS"]
    # Faithful to the reference run: live agent in the adapter's advanced sandbox mode.
    assert settings["OUROBOROS_RUNTIME_MODE"] == "advanced"
    # Secrets ship blank.
    for key, value in settings.items():
        if any(token in key.upper() for token in ("API_KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIALS")):
            assert value == "", f"secret-shaped template key {key} must be blank"


def test_help_exits_zero():
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc, contextlib.redirect_stdout(buf):
        run_clb.main(["--help"])
    assert exc.value.code == 0
    assert "--runner-path" in buf.getvalue()


def test_dry_run_writes_manifest_and_blanked_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    runner = _fake_runner(tmp_path)
    clone = _fake_clone(tmp_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_clb.main([
            "--runner-path", str(runner),
            "--ouroboros-clone", str(clone),
            "--path", "standard",
            "--dry-run",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    run_dir = Path(payload["run_root"])
    assert run_dir.exists() and str(run_dir).startswith(str(tmp_path))

    rendered = json.loads((run_dir / "_run_settings.json").read_text(encoding="utf-8"))
    assert rendered["OPENROUTER_API_KEY"] == ""  # secrets blanked on disk
    assert rendered["OUROBOROS_MODEL"] == rendered["OUROBOROS_MODEL_LIGHT"]  # single-model pin

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "continual_learning"
    argv = payload["planned_invocations"][0]
    assert "--no-live-dashboard" in argv
    assert argv[argv.index("--max-workers") + 1] == "1"  # strict sequential default
    params = json.loads(argv[argv.index("--system-params") + 1])
    assert params["evolution"] is False
    assert params["max_workers"] == 4  # within-task subagent pool, not cross-task parallelism
    fidelity = manifest["extra"]["fidelity"]
    assert "OUROBOROS_SAFETY_MODE" in fidelity["declared_only_pinned_adapter_gap"]
    assert "OUROBOROS_REVIEW_ENFORCEMENT" in fidelity["declared_only_pinned_adapter_gap"]
    assert "test-not-a-real-key" not in (run_dir / "run_manifest.json").read_text(encoding="utf-8")


def test_bridge_dry_run_plans_one_invocation_per_phase(tmp_path):
    runner = _fake_runner(tmp_path)
    clone = _fake_clone(tmp_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_clb.main([
            "--runner-path", str(runner),
            "--ouroboros-clone", str(clone),
            "--path", "bridge",
            "--phases", "stateless,stateful_noevo",
            "--num-instances", "3",
            "--dry-run",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    plans = payload["planned_invocations"]
    assert len(plans) == 2
    assert plans[0][plans[0].index("--phases") + 1] == "stateless"
    assert plans[1][plans[1].index("--phases") + 1] == "stateful_noevo"
    assert all("--docker" in plan for plan in plans)


def test_sequential_guard_refuses_parallel_without_opt_in(tmp_path):
    runner = _fake_runner(tmp_path)
    clone = _fake_clone(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_clb.main([
            "--runner-path", str(runner),
            "--ouroboros-clone", str(clone),
            "--instance-workers", "3",
            "--dry-run",
        ])
    assert "STRICTLY SEQUENTIAL" in str(exc.value)


def test_sequential_guard_refuses_parallel_stateful(tmp_path):
    """--allow-parallel-baseline is stateless-only: the standard path (always
    mode=stateful) and any non-stateless bridge phase must still be rejected."""
    runner = _fake_runner(tmp_path)
    clone = _fake_clone(tmp_path)
    base = ["--runner-path", str(runner), "--ouroboros-clone", str(clone),
            "--instance-workers", "3", "--allow-parallel-baseline", "--dry-run"]
    with pytest.raises(SystemExit) as exc:
        run_clb.main(base + ["--path", "standard"])
    assert "stateless-baseline-only" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        run_clb.main(base + ["--path", "bridge", "--phases", "stateless,stateful_noevo"])
    assert "ONLY stateless phases" in str(exc.value)


def test_collect_results_standard_path_keeps_denominator(tmp_path):
    """A --path standard run has NO bridge trace conditions (artifacts live in the
    external runner's own tree): the ledger still gets one explicit pointer row
    per requested run instead of an empty file."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "requested_task_ids": ["database_exploration:default:run0",
                               "database_exploration:default:run1"],
        "extra": {"phases": ""},
    }), encoding="utf-8")
    run_clb.collect_results(run_dir)
    ledger = [json.loads(line) for line in (run_dir / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ledger) == 2
    assert all(row["condition"] == "standard" for row in ledger)
    assert all(row["ouroboros_status"] == "external_runner_sidecar_only" for row in ledger)
    assert [row["instance_id"] for row in ledger] == ["default:run0", "default:run1"]


def test_collect_results_covers_absent_condition_dirs(tmp_path):
    """A planned phase whose traces/<condition>/ dir never appeared (runner died
    early) must still yield missing_outcome rows for every requested id; found
    outcomes are keyed by (domain, qid) so a same-qid row in another domain
    cannot mask a miss."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "requested_task_ids": ["database_exploration:q000", "web_research:q000"],
        "extra": {"phases": "stateless,stateful_noevo"},
    }), encoding="utf-8")
    # only stateless/database_exploration/q000 exists; stateful_noevo dir is ABSENT
    qdir = run_dir / "traces" / "stateless" / "database_exploration" / "q000"
    qdir.mkdir(parents=True)
    (qdir / "task_outcome.json").write_text(json.dumps(
        {"domain": "database_exploration", "instance_index": 0, "reward": 1.0,
         "success": True, "ouroboros_status": "completed", "cost_usd": 0.1}), encoding="utf-8")
    run_clb.collect_results(run_dir)
    ledger = [json.loads(line) for line in (run_dir / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    missing = {(row["condition"], row["domain"], row["instance_id"])
               for row in ledger if row["ouroboros_status"] == "missing_outcome"}
    # (domain,qid) keying: web_research:q000 is missing in stateless even though
    # database_exploration produced a q000; the whole absent phase is covered too.
    assert missing == {
        ("stateless", "web_research", "q000"),
        ("stateful_noevo", "database_exploration", "q000"),
        ("stateful_noevo", "web_research", "q000"),
    }


def test_collect_results_synthesizes_missing_outcome_rows(tmp_path):
    """A requested question with no task_outcome.json must surface as an explicit
    missing row (denominator preservation), driven by the run's own manifest."""
    run_dir = tmp_path / "run"
    qdir = run_dir / "traces" / "stateless" / "database_exploration" / "q000"
    qdir.mkdir(parents=True)
    (qdir / "task_outcome.json").write_text(json.dumps(
        {"domain": "database_exploration", "instance_index": 0, "reward": 1.0,
         "success": True, "ouroboros_status": "completed", "cost_usd": 0.5}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(
        {"requested_task_ids": ["database_exploration:q000", "database_exploration:q001"]}),
        encoding="utf-8")
    run_clb.collect_results(run_dir)
    ledger = [json.loads(line) for line in (run_dir / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    missing = [row for row in ledger if row["ouroboros_status"] == "missing_outcome"]
    assert [row["instance_id"] for row in missing] == ["q001"]
    assert missing[0]["reward"] is None


def test_live_repo_clone_refused(tmp_path):
    runner = _fake_runner(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_clb.main([
            "--runner-path", str(runner),
            "--ouroboros-clone", str(REPO_ROOT),
            "--dry-run",
        ])
    assert "never the live repo" in str(exc.value)


def test_collect_results_normalizes_bridge_traces(tmp_path):
    run_dir = tmp_path / "run"
    for condition, rewards in {"stateless": [0.2, 0.4, None], "stateful_noevo": [0.8, 0.6, 1.0]}.items():
        for i, reward in enumerate(rewards):
            qdir = run_dir / "traces" / condition / "database_exploration" / f"q{i:03d}"
            qdir.mkdir(parents=True)
            row = {"domain": "database_exploration", "instance_index": i, "reward": reward,
                   "success": reward == 1.0, "ouroboros_status": "completed", "cost_usd": 0.5}
            (qdir / "task_outcome.json").write_text(json.dumps(row), encoding="utf-8")
    results = run_clb.collect_results(run_dir)
    assert results["conditions"]["stateless"]["mean_reward"] == 0.3
    assert results["conditions"]["stateless"]["n_scored"] == 2
    assert results["conditions"]["stateless"]["missing_reward_indices"] == [2]
    assert results["conditions"]["stateful_noevo"]["mean_reward"] == 0.8
    assert results["memory_effect"] == 0.5
    assert (run_dir / "results.json").exists()
    ledger = [json.loads(line) for line in (run_dir / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ledger) == 6  # denominator-preserving: the reward-less row is still recorded
    assert {row["condition"] for row in ledger} == {"stateless", "stateful_noevo"}
