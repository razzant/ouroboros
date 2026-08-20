"""Harbor jobs: the config the run writes, the values it scrubs and the result it believes.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
job config and its dataset identity, the task cache lookup and the timeouts it may borrow,
the agent/verifier environment scrubbing that must not leak a value, the submission subtree
confinement, and the classification of a job by its trials rather than its exit code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


from tests._devtools_benchmarks_shared import (
    REPO_ROOT,
    _git_repo,
)
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


def test_terminal_bench_smoke_writes_manifest_and_planned_ledger(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb-run"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_harbor_smoke.py",
            # State-independent: this asserts ledger/denominator behaviour, not the seed gate.
            "--allow-dirty-seed",
            "--run-root",
            str(run_root),
            "--model",
            "google/gemini-3.5-flash",
            "--settings-path",
            str(settings),
        ],
    )

    assert harbor_smoke.main() == 0
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["benchmark"] == "terminal_bench"
    assert manifest["requested_count"] == 5
    assert manifest["requested_task_ids"] == []
    assert manifest["extra"]["selection"]["mode"] == "deterministic_first_n"
    assert len(manifest["extra"]["selection"]["requested_slots"]) == 5
    assert "--jobs-dir" in manifest["official_command"]
    assert "--output-dir" not in manifest["official_command"]
    assert f"host_settings_path={settings}" in manifest["official_command"]
    assert rows and {row["status"] for row in rows} == {"planned"}
    assert {row["instance_id"] for row in rows} == {f"selection-slot-{idx}" for idx in range(1, 6)}
    assert all(row["official_eval_status"] == "not_run" for row in rows)

def test_terminal_bench_parses_harbor_task_outcomes(tmp_path):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "stats": {
                    "evals": {
                        "eval": {
                            "reward_stats": {
                                "reward": {
                                    "1.0": ["task-b"],
                                    "0.0": ["task-a"],
                                }
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert harbor_smoke._harbor_task_outcomes(result_path) == [
        {"instance_id": "task-a", "reward": 0.0},
        {"instance_id": "task-b", "reward": 1.0},
    ]

def test_terminal_bench_resolves_only_new_harbor_result(tmp_path):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    old = tmp_path / "old" / "result.json"
    old.parent.mkdir()
    old.write_text("{}", encoding="utf-8")
    before = set(harbor_smoke._harbor_results(tmp_path))
    new = tmp_path / "new" / "result.json"
    new.parent.mkdir()
    new.write_text("{}", encoding="utf-8")

    assert harbor_smoke._new_harbor_result(tmp_path, before) == new.resolve(strict=False)

def test_terminal_bench_ambiguous_harbor_result_fails_closed(tmp_path):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    before: set[Path] = set()
    for name in ("a", "b"):
        result = tmp_path / name / "result.json"
        result.parent.mkdir()
        result.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly one new Harbor result"):
        harbor_smoke._new_harbor_result(tmp_path, before)

def test_terminal_bench_explicit_execute_uses_requested_denominator(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"
    commands = []

    def fake_run(cmd, cwd=None, env=None):
        commands.append(cmd)
        assert env and str(REPO_ROOT) in env.get("PYTHONPATH", "")
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(
            json.dumps({"stats": {"evals": {"eval": {"reward_stats": {"reward": {"1.0": ["task-a", "task-b"]}}}}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
    )

    assert harbor_smoke.main() == 0
    assert commands[0][commands[0].index("--n-tasks") + 1] == "2"
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-a", "task-b"]
    assert {row["status"] for row in rows} == {"harness_completed"}

def test_terminal_bench_explicit_execute_rejects_unexpected_observed_task(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(
            json.dumps({"stats": {"evals": {"eval": {"reward_stats": {"reward": {"1.0": ["unexpected-task"]}}}}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--task", "task-a", "--execute"],
    )

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-a"]
    assert rows[0]["status"] == "harness_failed"
    assert rows[0]["reason_code"] == "harbor_result_unresolved"
    assert "unexpected-task" in rows[0]["error"]

def test_terminal_bench_explicit_execute_rejects_missing_requested_task(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(
            json.dumps({"stats": {"evals": {"eval": {"reward_stats": {"reward": {"1.0": ["task-a"]}}}}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
    )

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-a", "task-b"]
    assert {row["status"] for row in rows} == {"harness_failed"}
    assert all(row["reason_code"] == "harbor_result_unresolved" for row in rows)
    assert all("task-b" in row["error"] for row in rows)

def test_terminal_bench_execute_fails_closed_on_unparseable_harbor_result(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--execute"])

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert {row["status"] for row in rows} == {"harness_failed"}
    assert all(row["reason_code"] == "harbor_result_unresolved" for row in rows)

def test_terminal_bench_execute_fails_closed_on_partial_deterministic_result(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        result = run_root / "job" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(
            json.dumps({"stats": {"evals": {"eval": {"reward_stats": {"reward": {"1.0": ["task-a"]}}}}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--n-tasks", "2", "--execute"])

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"harness_failed"}
    assert all("expected 2" in row["error"] for row in rows)

def test_terminal_bench_execute_writes_ledger_when_harbor_invocation_fails(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.run_harbor_smoke as harbor_smoke

    run_root = tmp_path / "tb"

    def fake_run(cmd, cwd=None, env=None):
        raise FileNotFoundError("harbor missing")

    monkeypatch.setattr(harbor_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_harbor_smoke.py", "--allow-dirty-seed", "--run-root", str(run_root), "--task", "task-a", "--task", "task-b", "--execute"],
    )

    assert harbor_smoke.main() == 2
    rows = [json.loads(line) for line in (run_root / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-a", "task-b"]
    assert {row["status"] for row in rows} == {"harness_failed"}
    assert {row["reason_code"] for row in rows} == {"harbor_invocation_failed"}
    assert all("harbor missing" in row["error"] for row in rows)

def test_terminal_bench_run_tb_validates_leaderboard_methodology():
    from devtools.benchmarks.terminal_bench.run_tb import validate_methodology

    validate_methodology(k=5, timeout_multiplier=1.0, resource_overrides=[])
    with pytest.raises(ValueError, match="k >= 5"):
        validate_methodology(k=1, timeout_multiplier=1.0, resource_overrides=[])
    with pytest.raises(ValueError, match="timeout_multiplier"):
        validate_methodology(k=5, timeout_multiplier=2.0, resource_overrides=[])
    with pytest.raises(ValueError, match="forbids resource overrides"):
        validate_methodology(k=5, timeout_multiplier=1.0, resource_overrides=["cpus=8"])

def test_terminal_bench_run_tb_builds_required_agent_kwargs(tmp_path, monkeypatch):
    import json as _json

    from devtools.benchmarks.terminal_bench.run_harbor_smoke import AGENT_IMPORT
    from devtools.benchmarks.terminal_bench.run_tb import HarborCommandConfig, harbor_command

    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "medium")
    cmd = harbor_command(HarborCommandConfig(
        dataset="terminal-bench/terminal-bench-2-1",
        model="openai/gpt-5.5",
        k=5,
        jobs_dir=tmp_path / "jobs",
        harbor_bin="harbor",
        n_concurrent=1,
        task_filters=["pypi-server"],
        settings_path=tmp_path / "settings.json",
        execute=True,
        light_model="google/gemini-3.5-flash",
    ))

    joined = " ".join(cmd)
    assert "-k 5" in joined
    # The agent MUST go through a job config (-c): the bare --agent-import-path
    # flag records agents[0].name = null, which the TB2.1 leaderboard static
    # analysis can never match (terminal-bench-2-1#121).
    assert "--agent-import-path" not in cmd
    assert "--agent-kwarg" not in cmd
    assert "--config" in cmd
    cfg_path = cmd[cmd.index("--config") + 1]
    agent_cfg = _json.loads(open(cfg_path, encoding="utf-8").read())["agents"][0]
    assert agent_cfg["name"] == "Ouroboros Installed"
    assert agent_cfg["import_path"] == AGENT_IMPORT
    assert agent_cfg["model_name"] == "ouroboros-openai-gpt-5.5"
    kw = agent_cfg["kwargs"]
    assert kw["task_review_mode"] == "required"
    assert kw["ouroboros_light_model"] == "google/gemini-3.5-flash"
    assert kw["disable_agent_web"] is True
    # Effort labeling: OUROBOROS_EFFORT_TASK becomes the declared submission
    # effort; the adapter forwards it back into the container env.
    assert kw["reasoning_effort"] == "medium"
    assert "--include-task-name" in cmd
    assert "pypi-server" in cmd
    assert "--force-build" in cmd
    # 6a: leaderboard-faithful default — Harbor static_validation REJECTS the
    # setup/build timeout multipliers (static_validation.py
    # _trial_timeout_override_fields rejects agent_setup_timeout_multiplier +
    # environment_build_timeout_multiplier), so harbor_command omits them by default;
    # they appear only under the local --allow-setup-build-multipliers opt-in (covered
    # in test_run_tb_methodology.py). Task/verifier timeout multipliers stay 1.0 too.
    assert "--agent-setup-timeout-multiplier" not in cmd
    assert "--environment-build-timeout-multiplier" not in cmd
    assert "--agent-timeout-multiplier" not in cmd

def _write_cached_task(cache_root, org, name, digest, timeout_sec):
    task = cache_root / org / name / digest
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(
        f"[agent]\ntimeout_sec = {timeout_sec}\n", encoding="utf-8"
    )
    return task / "task.toml"

def test_harbor_task_cache_lookup_uses_dataset_org_not_a_hardcoded_one(tmp_path, monkeypatch):
    """The adapter used to hardcode org `terminal-bench`, so every non-TB dataset silently ran
    deadline-blind. The org now comes from the threaded dataset identity."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    cache = tmp_path / "packages"
    _write_cached_task(cache, "terminal-bench", "shared-name", "aaa", 600)
    _write_cached_task(cache, "harbor-index", "shared-name", "bbb", 1800)
    monkeypatch.setattr(tb_agent.OuroborosTerminalBenchAgent, "_PACKAGE_CACHE_DIR", cache)

    logs = tmp_path / "logs" / "shared-name__trialhash" / "agent"
    logs.mkdir(parents=True)
    for dataset, expected in (
        ("terminal-bench/terminal-bench-2-1", 600),
        ("harbor-index/harbor-index-1-0", 1800),
    ):
        agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=logs, dataset=dataset)
        assert agent._cached_task_toml("shared-name").parent.parent.parent.name == dataset.split("/")[0]
        assert agent._resolve_task_timeout_from_dataset(object()) == expected

def test_harbor_task_cache_lookup_refuses_an_ambiguous_task_name(tmp_path, monkeypatch):
    """Same-named tasks in two orgs and no dataset org at all: returning either one would hand
    the agent a FOREIGN wall-clock cap, so the name-only lookup refuses (deadline-blind is the
    honest degradation). A single owner is still resolved when the dataset names no org."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    cache = tmp_path / "packages"
    _write_cached_task(cache, "terminal-bench", "collide", "aaa", 600)
    _write_cached_task(cache, "scale-ai", "collide", "bbb", 1200)
    _write_cached_task(cache, "gaia", "only-here", "ccc", 900)
    monkeypatch.setattr(tb_agent.OuroborosTerminalBenchAgent, "_PACKAGE_CACHE_DIR", cache)

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, dataset="")
    assert agent._cached_task_toml("collide") is None
    assert agent._cached_task_toml("only-here") is not None
    assert agent._cached_task_toml("absent") is None

def test_harbor_task_cache_lookup_never_borrows_another_orgs_timeout(tmp_path, monkeypatch):
    """An EXPLICIT dataset org is authoritative: no cross-owner fallback, ever.

    This previously fell back from a missing configured org to "any unique cache owner", and an
    earlier revision of the ambiguity test asserted that borrow as intended behaviour. It is not
    a lenient fallback — the borrowed field is the wall-clock cap, so the trial silently runs
    under another benchmark's deadline. Frontier-Bench (600s verifier caps) next to
    Terminal-Bench 2.1 (3600s) is the live 6x case, and both are routinely cached side by side
    on the same host."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent
    from devtools.benchmarks.terminal_bench import run_tb

    cache = tmp_path / "packages"
    # Only terminal-bench has this task cached; frontier-bench does not.
    _write_cached_task(cache, "terminal-bench", "borrowed-task", "aaa", 3600)
    monkeypatch.setattr(tb_agent.OuroborosTerminalBenchAgent, "_PACKAGE_CACHE_DIR", cache)

    logs = tmp_path / "logs" / "borrowed-task__trialhash" / "agent"
    logs.mkdir(parents=True)
    fb = tb_agent.OuroborosTerminalBenchAgent(logs_dir=logs, dataset=run_tb.FRONTIER_BENCH_DATASET)
    assert fb._cached_task_toml("borrowed-task") is None
    assert fb._resolve_task_timeout_from_dataset(object()) is None   # deadline-blind, not 3600
    # ...while the org that really owns the task still resolves its own cap.
    tb = tb_agent.OuroborosTerminalBenchAgent(logs_dir=logs, dataset=run_tb.DEFAULT_DATASET)
    assert tb._resolve_task_timeout_from_dataset(object()) == 3600

def test_frontier_bench_wall_clock_cap_resolves_from_its_own_cache_org(tmp_path, monkeypatch):
    """Frontier-Bench needs NO adapter change: harbor caches its tasks under org `frontier-bench`
    (verified against harbor 0.18.0, which populates
    `~/.cache/harbor/tasks/packages/frontier-bench/<task>/<digest>/task.toml`), so the already
    dataset-parametric lookup resolves FB's own cap even while TB2.1 caches a same-named task.
    FB caps are an order above TB2.1's (median 7200s), so picking the wrong org is not cosmetic."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent
    from devtools.benchmarks.terminal_bench import run_tb

    cache = tmp_path / "packages"
    _write_cached_task(cache, "terminal-bench", "bun-sourcemap-leak", "aaa", 600)
    _write_cached_task(cache, "frontier-bench", "bun-sourcemap-leak", "bbb", 1800)
    monkeypatch.setattr(tb_agent.OuroborosTerminalBenchAgent, "_PACKAGE_CACHE_DIR", cache)

    logs = tmp_path / "logs" / "bun-sourcemap-leak__trialhash" / "agent"
    logs.mkdir(parents=True)
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=logs, dataset=run_tb.FRONTIER_BENCH_DATASET)
    assert agent._cached_task_toml("bun-sourcemap-leak").parent.parent.parent.name == "frontier-bench"
    assert agent._resolve_task_timeout_from_dataset(object()) == 1800

def test_run_tb_job_config_carries_dataset_and_deep_merges_a_base_config(tmp_path):
    from devtools.benchmarks.terminal_bench import run_tb

    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "environment": {"env": {"UPSTREAM": "keep"}, "type": "docker"},
        "agents": [{"name": "Upstream Agent", "kwargs": {"dropped": True}}],
        "verifier": {"timeout_multiplier": 1.0},
    }), encoding="utf-8")
    cfg = run_tb.HarborCommandConfig(
        dataset="harbor-index/harbor-index-1-0", model="m", k=5, jobs_dir=tmp_path / "jd",
        harbor_bin="harbor", n_concurrent=1, task_filters=[], settings_path=tmp_path / "s.json",
        execute=False, light_model="m", base_job_config=base,
    )
    path = run_tb._write_agent_job_config(cfg)
    written = json.loads(path.read_text(encoding="utf-8"))

    # Upstream keys survive untouched; our agents[] block wins whole (name must stay ours,
    # a null/foreign agents[0].name permanently invalidates a submission).
    assert written["environment"] == {"env": {"UPSTREAM": "keep"}, "type": "docker"}
    assert written["verifier"] == {"timeout_multiplier": 1.0}
    assert len(written["agents"]) == 1
    assert written["agents"][0]["name"] == "Ouroboros Installed"
    assert "dropped" not in written["agents"][0]["kwargs"]
    assert written["agents"][0]["kwargs"]["dataset"] == "harbor-index/harbor-index-1-0"

def test_run_tb_forwards_agent_and_verifier_env_without_leaking_values(tmp_path):
    from devtools.benchmarks.terminal_bench import run_tb

    cfg = run_tb.HarborCommandConfig(
        dataset=run_tb.DEFAULT_DATASET, model="m", k=5, jobs_dir=tmp_path, harbor_bin="harbor",
        n_concurrent=1, task_filters=["t1"], settings_path=tmp_path / "s.json", execute=False,
        light_model="m", agent_env=("AWS_REGION=us-east-1",), verifier_env=("OPENAI_API_KEY=sk-secret",),
    )
    cmd = run_tb.harbor_command(cfg)

    assert cmd[cmd.index("--ae") + 1] == "AWS_REGION=us-east-1"
    assert cmd[cmd.index("--ve") + 1] == "OPENAI_API_KEY=sk-secret"
    safe = run_tb.redacted_command(cmd)
    assert "sk-secret" not in " ".join(safe)
    assert "OPENAI_API_KEY=<redacted>" in safe and "AWS_REGION=<redacted>" in safe
    # Nothing else about the command changes.
    assert [tok for tok in safe if "=" not in tok] == [tok for tok in cmd if "=" not in tok]

def _harbor_job_tree(root, *, cleartext: str, partial: str) -> dict:
    """A synthetic harbor 0.18.0 job tree, using harbor's REAL filenames and layout.

    Ground truth (installed harbor 0.18.0, `harbor/job.py`): the job config is
    `<jobs-dir>/<job_name>/config.json` — one timestamp level below the `--jobs-dir` our
    launcher passes — written as `self.config.model_dump_json(indent=4, exclude_defaults=True)`,
    and the same env dicts are re-serialized into the job `lock.json` and every trial's
    `config.json` / `lock.json` / `result.json`. `--ae` lands in `agents[].env`, `--ve` in
    `verifier.env`. Harbor's own `templatize_sensitive_env` writes a value VERBATIM when the
    NAME does not match `KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH`, and only partially
    (`value[:4] + "****" + value[-3:]`) when it does — both forms are planted here."""
    job = root / "job" / "2026-07-25__12-00-00"
    trial = job / "some-task__abc123"
    (trial / "agent").mkdir(parents=True)
    written = {}
    env_block = {"JUDGE_API_KEY": partial, "MY_BEARER": cleartext}
    for name in ("config.json", "lock.json"):
        p = job / name
        p.write_text(json.dumps({"verifier": {"env": env_block}, "agents": [{"env": env_block}]},
                                indent=4), encoding="utf-8")
        written[str(p)] = True
    for name in ("config.json", "lock.json", "result.json"):
        p = trial / name
        p.write_text(json.dumps({"config": {"verifier": {"env": env_block}}}, indent=4),
                     encoding="utf-8")
        written[str(p)] = True
    # harbor's only un-redacted path (`trial.py` writes `traceback.format_exc()`), plus an
    # agent-written log: both can carry the resolved cleartext value.
    (trial / "exception.txt").write_text(f"RuntimeError: Command: docker -e MY_BEARER={cleartext}\n",
                                         encoding="utf-8")
    (trial / "agent" / "session.log").write_text(f"env MY_BEARER={cleartext}\n", encoding="utf-8")
    return written

def test_scrub_covers_the_harbor_written_job_config_for_ae_ve_values(tmp_path, monkeypatch):
    """The leak this closes: harbor persists its own JobConfig (and lock/result files) into the
    job dir that gets PUBLICLY uploaded, so a `--ve` value is on disk even though the launcher's
    own artifacts carry names only. The scrub must sweep the whole tree BY VALUE.

    Deterministic and self-contained: the tree is built here, the secret is obviously fake, and
    nothing depends on the ambient checkout or on a real key."""
    from devtools.benchmarks.terminal_bench import run_tb
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    fake = "FAKEfake-judge-key-0000000000deadbeef"   # obviously fake; never a real credential
    # 1. The launcher really does hand this value to harbor's `--ve`.
    cmd = run_tb.harbor_command(run_tb.HarborCommandConfig(
        dataset=run_tb.DEFAULT_DATASET, model="m", k=5, jobs_dir=tmp_path / "jobs",
        harbor_bin="harbor", n_concurrent=1, task_filters=["t1"],
        settings_path=tmp_path / "s.json", execute=False, light_model="m",
        verifier_env=(f"JUDGE_API_KEY={fake}",),
    ))
    assert cmd[cmd.index("--ve") + 1] == f"JUDGE_API_KEY={fake}"
    assert fake not in " ".join(run_tb.redacted_command(cmd))

    # 2. A submission copy of the job dir, in harbor's real shape.
    root = tmp_path / "job_copy"
    root.mkdir()
    partial = scrub.harbor_redacted_form(fake)
    assert partial and partial != fake                    # harbor leaks 7 chars, not zero
    _harbor_job_tree(root, cleartext=fake, partial=partial)
    sources = tmp_path / "fake_settings.json"
    sources.write_text(json.dumps({"OPENROUTER_API_KEY": "FAKEfake-other-value-1111"}),
                       encoding="utf-8")

    # 3. Scrub, then assert the value is gone from EVERY file in the tree.
    monkeypatch.setattr(sys, "argv", ["scrub", "--root", str(root), "--secrets-from", str(sources),
                                      "--env-passthrough", f"JUDGE_API_KEY={fake}"])
    assert scrub.main() == 0
    files = [p for p in root.rglob("*") if p.is_file()]
    assert len(files) >= 7
    for path in files:
        raw = path.read_bytes()
        assert fake.encode() not in raw, f"cleartext survived in {path}"
        assert partial.encode() not in raw, f"harbor's partial form survived in {path}"
    # Structure preserved: the config is still valid JSON with the key present, value redacted.
    cfg = json.loads((root / "job" / "2026-07-25__12-00-00" / "config.json").read_text(encoding="utf-8"))
    assert cfg["verifier"]["env"]["MY_BEARER"] == "<REDACTED:JUDGE_API_KEY>"

def test_scrub_keeps_every_passthrough_occurrence_of_a_repeated_env_name(tmp_path, monkeypatch):
    """One env NAME, two DIFFERENT values (agent phase vs verifier phase) — the real shape when a
    judge key and an agent key share a name, or when a flag is repeated. Keying the passthrough
    needles on the NAME alone dropped the earlier value, so a CORRECT scrub invocation published
    that credential in harbor's job tree. Every distinct occurrence must survive collection."""
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    agent_value = "FAKEfake-agent-value-000000000000aaaa"    # obviously fake; never a credential
    verifier_value = "FAKEfake-verifier-value-11111111bbbb"
    name = "SHARED_API_KEY"

    # Collection alone must retain both values (plus each one's harbor partial form).
    needles, refusals = scrub.collect_env_passthrough(
        [f"{name}={agent_value}", f"{name}={verifier_value}"]
    )
    assert refusals == []
    assert sorted(needles.values()) == sorted([
        agent_value, verifier_value,
        scrub.harbor_redacted_form(agent_value), scrub.harbor_redacted_form(verifier_value),
    ])
    # An exact repeat is the same secret, not a second one: it must not inflate the needle set.
    repeated, _ = scrub.collect_env_passthrough([f"{name}={agent_value}"] * 3)
    assert len(repeated) == 2   # the value plus its harbor partial form

    # End-to-end: both values are planted in harbor's own job tree and both must be gone.
    root = tmp_path / "job_copy"
    job = root / "job" / "2026-07-25__12-00-00"
    job.mkdir(parents=True)
    (job / "config.json").write_text(
        json.dumps({"agents": [{"env": {name: agent_value}}],
                    "verifier": {"env": {name: verifier_value}}}, indent=4),
        encoding="utf-8",
    )
    sources = tmp_path / "fake_settings.json"
    sources.write_text(json.dumps({"OPENROUTER_API_KEY": "FAKEfake-other-value-1111"}),
                       encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "scrub", "--root", str(root), "--secrets-from", str(sources),
        "--env-passthrough", f"{name}={agent_value}",
        "--env-passthrough", f"{name}={verifier_value}",
    ])
    assert scrub.main() == 0
    raw = (job / "config.json").read_bytes()
    assert agent_value.encode() not in raw and verifier_value.encode() not in raw
    # Structure preserved: both entries are still present, each redacted under its own label.
    cfg = json.loads((job / "config.json").read_text(encoding="utf-8"))
    assert cfg["agents"][0]["env"][name].startswith("<REDACTED:")
    assert cfg["verifier"]["env"][name].startswith("<REDACTED:")
    assert cfg["agents"][0]["env"][name] != cfg["verifier"]["env"][name]

def test_scrub_fails_closed_on_an_unsweepable_ae_ve_value_and_changes_nothing(tmp_path, monkeypatch):
    """A value we cannot sweep safely must ABORT before any write: a maybe-scrubbed tree that
    then gets uploaded is strictly worse than no submission at all."""
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    root = tmp_path / "job_copy"
    root.mkdir()
    target = root / "job" / "2026-07-25__12-00-00"
    target.mkdir(parents=True)
    before = json.dumps({"verifier": {"env": {"SHORT_TOKEN": "ab1"}}}, indent=4)
    (target / "config.json").write_text(before, encoding="utf-8")
    sources = tmp_path / "fake_settings.json"
    sources.write_text(json.dumps({"OPENROUTER_API_KEY": "FAKEfake-other-value-1111"}),
                       encoding="utf-8")

    for bad in ("SHORT_TOKEN=ab1",           # too short to sweep safely
                "WORDY_TOKEN=onlyletters",   # not credential-shaped
                "NOEQUALS"):                 # malformed pair
        monkeypatch.setattr(sys, "argv", ["scrub", "--root", str(root), "--secrets-from",
                                          str(sources), "--env-passthrough", bad])
        assert scrub.main() == 2, bad
        assert (target / "config.json").read_text(encoding="utf-8") == before, bad

def test_scrub_sweeps_and_verifies_json_escaped_forms_not_only_the_literal(tmp_path, monkeypatch):
    r"""The false all-clear this closes. Harbor persists env values through JSON serializers, so
    a value containing a quote, a backslash, a control character or a non-ASCII character is on
    disk ESCAPED (``abc"1234`` is stored as ``abc\"1234``). A literal-only sweep walked past it
    AND the literal-only verify then printed zero leftovers — a scrubber that misses a secret and
    reports success is worse than no scrubber, because it turns "check this by hand" into a tool
    verdict. Both passes must see every persisted form.

    Self-contained: values are obviously fake, the tree is built here, nothing reads the ambient
    checkout, the cwd or any real credential source."""
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    # One awkward value per escape class the reviewer named. Obviously fake, never credentials.
    awkward = {
        "QUOTE_TOKEN": 'FAKEfake-quote-"-0000000001',
        "BACKSLASH_TOKEN": "FAKEfake-backslash-\\-0000002",
        "CONTROL_TOKEN": "FAKEfake-control-\x01-0000003",
        "UNICODE_TOKEN": "FAKEfake-nonascii-é-000004",
    }

    # 1. The encoded forms are the SERIALIZER's output, not a hand-kept escape table.
    assert scrub.json_encoded_forms(awkward["QUOTE_TOKEN"]) == ['FAKEfake-quote-\\"-0000000001']
    assert scrub.json_encoded_forms(awkward["BACKSLASH_TOKEN"]) == ["FAKEfake-backslash-\\\\-0000002"]
    assert scrub.json_encoded_forms(awkward["CONTROL_TOKEN"]) == ["FAKEfake-control-\\u0001-0000003"]
    # ensure_ascii=True escapes the non-ASCII char; ensure_ascii=False leaves it == the literal,
    # which is already swept as its own needle, so exactly one extra form is produced.
    assert scrub.json_encoded_forms(awkward["UNICODE_TOKEN"]) == ["FAKEfake-nonascii-\\u00e9-000004"]
    # A value with nothing to escape adds no needle at all.
    assert scrub.json_encoded_forms("FAKEfake-plain-00000005") == []
    expanded = scrub.expand_encoded_forms({"QUOTE_TOKEN": awkward["QUOTE_TOKEN"]})
    assert expanded["QUOTE_TOKEN"] == awkward["QUOTE_TOKEN"]
    assert expanded["QUOTE_TOKEN:json"] == 'FAKEfake-quote-\\"-0000000001'

    # 2. A harbor-shaped tree holding each value in BOTH serializer configurations, plus the raw
    #    literals in an un-redacted log (harbor's traceback path).
    root = tmp_path / "job_copy"
    job = root / "job" / "2026-07-25__12-00-00"
    job.mkdir(parents=True)
    payload = {"verifier": {"env": dict(awkward)}, "agents": [{"env": dict(awkward)}]}
    (job / "config.json").write_text(       # pydantic/serde shape: raw UTF-8
        json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    (job / "lock.json").write_text(         # python json default shape: \uXXXX
        json.dumps(payload, indent=4, ensure_ascii=True), encoding="utf-8")
    (job / "exception.txt").write_text(
        "".join(f"RuntimeError: docker -e {name}={value}\n" for name, value in awkward.items()),
        encoding="utf-8")
    # A --secrets-from value needs the same treatment; expansion is not passthrough-only.
    from_source = 'FAKEfake-source-"-000000006'
    sources = tmp_path / "fake_settings.json"
    sources.write_text(json.dumps({"OPENROUTER_API_KEY": from_source}), encoding="utf-8")
    (job / "settings_echo.json").write_text(
        json.dumps({"OPENROUTER_API_KEY": from_source}, indent=4), encoding="utf-8")

    argv = ["scrub", "--root", str(root), "--secrets-from", str(sources)]
    for name, value in awkward.items():
        argv += ["--env-passthrough", f"{name}={value}"]
    monkeypatch.setattr(sys, "argv", list(argv))

    # 3. Every form of every value must be gone — this is what failed before the fix.
    assert scrub.main() == 0
    files = [p for p in root.rglob("*") if p.is_file()]
    for path in files:
        raw = path.read_bytes()
        for value in (*awkward.values(), from_source):
            assert value.encode() not in raw, f"literal survived in {path}"
            for form in scrub.json_encoded_forms(value):
                assert form.encode() not in raw, f"JSON-escaped form survived in {path}"
    # Structure preserved: still valid JSON, keys intact, values redacted.
    for name in ("config.json", "lock.json"):
        cfg = json.loads((job / name).read_text(encoding="utf-8"))
        assert sorted(cfg["verifier"]["env"]) == sorted(awkward)
        for redacted in cfg["verifier"]["env"].values():
            assert redacted.startswith("<REDACTED:")

    # 4. The VERIFY pass must refuse to declare success while an escaped form remains. Re-planting
    #    only the escaped form is precisely the case that used to exit 0 with the secret on disk.
    (job / "lock.json").write_text(
        json.dumps({"verifier": {"env": {"QUOTE_TOKEN": awkward["QUOTE_TOKEN"]}}},
                   indent=4, ensure_ascii=False),
        encoding="utf-8")
    planted = (job / "lock.json").read_text(encoding="utf-8")
    assert 'FAKEfake-quote-\\"-0000000001' in planted        # escaped form only
    assert awkward["QUOTE_TOKEN"] not in planted             # literal is NOT on disk
    monkeypatch.setattr(scrub, "_sweep_file", lambda path, secrets: (0, []))  # verify pass alone
    monkeypatch.setattr(sys, "argv", list(argv))
    assert scrub.main() == 1

def test_run_tb_submission_subtree_is_derived_from_the_dataset():
    from devtools.benchmarks.terminal_bench import run_tb

    # TB2.1 keeps its published layout byte-identical.
    assert run_tb.submission_subtree("terminal-bench/terminal-bench-2-1") == ("terminal-bench", "2.1")
    # Another dataset no longer lands in the TB2.1 tree.
    assert run_tb.submission_subtree("harbor-index/harbor-index-1-0") == ("harbor-index", "1.0")
    family, version = run_tb.submission_subtree("some-org/unversioned")
    assert (family, version) == ("unversioned", "")

def test_run_tb_submission_subtree_components_are_confined():
    """`submission_root` is validated, but the job dir is DERIVED from it — so the components that
    are about to be created are what must be checked, not their already-checked ancestor. Pure
    function: no env, no cwd, no repo path, nothing derived from `__file__`."""
    from devtools.benchmarks.terminal_bench import run_tb

    # Accepted shapes are unchanged, derived and explicit alike (trailing slash still tolerated).
    assert run_tb.confined_submission_subtree("", dataset="terminal-bench/terminal-bench-2-1") == [
        "terminal-bench", "2.1",
    ]
    assert run_tb.confined_submission_subtree("terminal-bench/2.1", dataset="ignored") == ["terminal-bench", "2.1"]
    assert run_tb.confined_submission_subtree("frontier-bench/", dataset="ignored") == ["frontier-bench"]

    for escape in ("..", "../..", "../../../etc", "terminal-bench/../../..", ".", "./x", "/abs/path", "/"):
        with pytest.raises(ValueError):
            run_tb.confined_submission_subtree(escape, dataset="terminal-bench/terminal-bench-2-1")

    # Windows forms: `\` is a separator and `C:` a drive qualifier there, and this repo's CI matrix
    # runs all three OSes — a value that is an inert directory name on POSIX escapes on Windows.
    for windows_form in ("..\\..\\evil", "sub\\dir", "C:\\evil", "C:/evil", "C:evil", "\\\\server\\share", "\\evil"):
        with pytest.raises(ValueError):
            run_tb.confined_submission_subtree(windows_form, dataset="terminal-bench/terminal-bench-2-1")

    # The DERIVED path is untrusted too: `--dataset` reaches it through submission_subtree(), which
    # splits the name without judging it.
    assert run_tb.submission_subtree("org/..-2-1") == ("..", "2.1")
    with pytest.raises(ValueError):
        run_tb.confined_submission_subtree("", dataset="org/..-2-1")

def test_run_tb_refuses_an_escaping_subtree_before_creating_anything(tmp_path, monkeypatch):
    """The refusal must land ahead of the first mkdir, so a rejected run leaves no directories at
    all. Hermetic: cwd is redirected into tmp_path, so a regression that reaches the run-root
    default writes there and is caught by the emptiness assertion instead of touching a checkout."""
    from devtools.benchmarks.terminal_bench import run_tb

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="single safe path component"):
        run_tb.main([
            "--model", "anthropic/claude-sonnet-5",
            "--submission-root", str(tmp_path / "submission"),
            "--run-root", str(tmp_path / "run"),
            "--submission-subtree", "../../../escaped",
        ])
    assert list(tmp_path.iterdir()) == []

def test_run_tb_classifies_a_harbor_job_by_its_trials_not_its_exit_code():
    """The sibling swallow: `harbor run` has no non-zero exit path for a job whose trials all
    ERRORED either (2026-07-04: a job wrote 444 trial `result.json` files and zero rewards while
    looking healthy), so run_tb decides from the disclosure ledger it already builds.

    Same three-way distinction as GAIA, and the same reason for it: an all-zero reward
    distribution over SCORED trials is a genuine result, while trials that never reached the
    verifier are not."""
    from devtools.benchmarks.terminal_bench.run_tb import classify_harbor_outcome

    # Scored trials, all zero -> a genuine result.
    assert classify_harbor_outcome({"n_trials": 4, "reward_distribution": {"0.0": 4}}, 0) == ("completed", 0)
    assert classify_harbor_outcome({"n_trials": 4, "reward_distribution": {"0.0": 3, "1.0": 1}}, 0) == ("completed", 0)
    # Nothing reached the verifier -> an infra zero, non-zero exit despite harbor's 0.
    assert classify_harbor_outcome({"n_trials": 444, "reward_distribution": {"null": 444}}, 0) == ("no_scored_trials", 1)
    assert classify_harbor_outcome({"n_trials": 0, "reward_distribution": {}}, 0) == ("no_scored_trials", 1)
    # Ledger unavailable -> no evidence of a result; fail closed rather than claim `completed`.
    assert classify_harbor_outcome(None, 0) == ("trials_unverified", 1)
    # A harness that DID fail keeps its own status.
    assert classify_harbor_outcome({"n_trials": 4, "reward_distribution": {"1.0": 4}}, 2) == ("harness_nonzero_exit", 2)

def test_run_tb_manifest_records_the_model_the_run_actually_resolved(tmp_path, monkeypatch):
    """TB's manifest must name the model that RAN, in the SAME field GAIA records it in.

    Presence is deliberately not the property under test. The sibling failure this guards
    against is SWE-Pro's manifest naming a model that did not run because it snapshotted the
    settings TEMPLATE instead of the derived settings, so a decoy model is planted in BOTH the
    host env and the host settings file: an implementation that copies either one still writes a
    perfectly non-empty `model_slots`, and still fails every equality assertion below. The
    `--all-model` leg additionally pins the post-override value, the one `--model` alone never
    sees.

    Hermetic by construction — purpose-built seed repo, tmp settings file, tmp run root, cwd
    redirected into tmp_path and the harbor probe stubbed — so nothing here depends on this
    machine's workspace layout or on a harbor binary being installed.
    """
    from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS
    from devtools.benchmarks.terminal_bench import run_tb

    seed = tmp_path / "seed"
    _git_repo(seed)
    monkeypatch.setattr(run_tb, "repo_root_from_devtools", lambda: seed)
    monkeypatch.setattr(run_tb, "harbor_version", lambda _harbor_bin: "")
    monkeypatch.chdir(tmp_path)
    for key in MODEL_SLOT_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"OUROBOROS_MODEL": "decoy/template-main",
                    "OUROBOROS_MODEL_LIGHT": "decoy/template-light"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_MODEL", "decoy/ambient-main")

    def _manifest(run_root):
        return json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))

    measured = tmp_path / "measured"
    assert run_tb.main([
        "--model", "anthropic/claude-fable-5",
        "--light-model", "google/gemini-3.5-flash",
        "--run-root", str(measured),
        "--submission-root", str(tmp_path / "submission"),
        "--settings-path", str(settings),
    ]) == 0
    manifest = _manifest(measured)
    slots = manifest["model_slots"]
    # The measured model, NOT the ambient env decoy and NOT the settings-template decoy.
    assert slots["OUROBOROS_MODEL"] == "anthropic/claude-fable-5"
    assert slots["OUROBOROS_MODEL_LIGHT"] == "google/gemini-3.5-flash"
    # The adapter drives HEAVY and the fallback chain off the same kwarg, so they must not
    # imply a second model.
    assert slots["OUROBOROS_MODEL_HEAVY"] == "anthropic/claude-fable-5"
    assert slots["OUROBOROS_MODEL_FALLBACKS"] == "anthropic/claude-fable-5"
    assert "decoy/ambient-main" not in slots.values()
    assert "decoy/template-main" not in slots.values()
    # `model_slots` means the same thing here as in GAIA's manifest: MODEL_SLOT_KEYS only.
    assert set(slots).issubset(set(MODEL_SLOT_KEYS))
    # ...and the same fact is on disk from admission onward, in TB's established `extra` shape.
    assert manifest["extra"]["model"] == "anthropic/claude-fable-5"
    assert manifest["extra"]["light_model"] == "google/gemini-3.5-flash"

    # --all-model rewrites --model AFTER parsing; the manifest must follow the override, not the
    # (here empty) --model it was parsed with.
    single = tmp_path / "single"
    assert run_tb.main([
        "--all-model", "openai/gpt-5.6-sol",
        "--run-root", str(single),
        "--submission-root", str(tmp_path / "submission"),
        "--settings-path", str(settings),
    ]) == 0
    single_manifest = _manifest(single)
    assert single_manifest["model_slots"]["OUROBOROS_MODEL"] == "openai/gpt-5.6-sol"
    assert single_manifest["model_slots"]["OUROBOROS_MODEL_LIGHT"] == "openai/gpt-5.6-sol"
    assert single_manifest["extra"]["model"] == "openai/gpt-5.6-sol"
    # Every forwarded slot the single-model run pinned is recorded as that one model.
    for key in run_tb._ALL_MODEL_SLOT_KEYS:
        assert single_manifest["model_slots"][key] == "openai/gpt-5.6-sol"
    # Slots the in-container adapter never forwards stay OUT: recording a model the container
    # cannot see would be as false as recording the wrong one.
    assert not set(single_manifest["model_slots"]) & set(run_tb._UNFORWARDED_MODEL_SLOT_KEYS)

def test_scrubber_refuses_symlinks_instead_of_writing_through_them(tmp_path, capsys, monkeypatch):
    """A symlink under --root must stop the scrub dead, before anything is written.

    Two independent failures, both proven against the pre-fix tool:

    * `p.is_file()` and `path.write_text()` BOTH follow a file symlink, so the sweep
      rewrites the link's TARGET outside --root. A pack linking to the live settings.json
      had its real keys replaced with `<REDACTED:...>` by the tool meant to protect them.
      `cp -a` preserves symlinks, so the procedural "run this on a COPY" rule does not
      help — the copy carries the same link.
    * `rglob` does not descend through a DIRECTORY symlink, yet the verify pass still
      printed `verify_leftovers=0` and exited 0. The tool certified a tree it had never
      read, for content reachable under --root and about to be uploaded publicly.

    The second is the one that matters most: silent non-coverage reported as cleanliness
    is precisely the class of false claim this release exists to remove, and here the
    consequence is a live API key on a public leaderboard."""
    from devtools.benchmarks.terminal_bench import scrub_submission_secrets as scrub

    fake = "FAKEfake-scrub-symlink-000000000000cccc"   # obviously fake; never a credential

    outside = tmp_path / "outside"
    outside.mkdir()
    live = outside / "live_settings.json"
    live.write_text(f'{{"OPENROUTER_API_KEY": "{fake}"}}\n', encoding="utf-8")

    behind_dir_link = tmp_path / "behind"
    behind_dir_link.mkdir()
    (behind_dir_link / "deep.txt").write_text(f"token={fake}\n", encoding="utf-8")

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "normal.txt").write_text(f"plain={fake}\n", encoding="utf-8")
    (pack / "linked_settings.json").symlink_to(live)
    (pack / "hidden_dir").symlink_to(behind_dir_link, target_is_directory=True)

    secrets_src = tmp_path / "secrets.txt"
    secrets_src.write_text(f"OPENROUTER_API_KEY: {fake}\n", encoding="utf-8")

    argv = ["scrub_submission_secrets.py", "--root", str(pack),
            "--secrets-from", str(secrets_src)]
    monkeypatch.setattr(sys, "argv", argv)
    rc = scrub.main()

    assert rc == 2, "a symlink under --root must be a hard refusal, not a warning"

    err = capsys.readouterr().err
    assert "REFUSING TO SCRUB" in err
    # BOTH kinds must be named, with their targets, so the operator can act.
    assert "linked_settings.json" in err and str(live) in err
    assert "hidden_dir" in err and str(behind_dir_link) in err

    # Fail CLOSED: not one byte written anywhere — not through the link, not even to the
    # ordinary file the tool could legitimately have swept.
    assert fake in live.read_text(encoding="utf-8"), "wrote through the symlink"
    assert fake in (behind_dir_link / "deep.txt").read_text(encoding="utf-8")
    assert fake in (pack / "normal.txt").read_text(encoding="utf-8"), (
        "a refusal must leave the tree untouched; a partially swept pack is worse than "
        "an unswept one"
    )
    assert (pack / "linked_settings.json").is_symlink(), "must not have been replaced"

    # ...and with the links gone the tool still does its job, so the guard is a refusal
    # of an unsafe shape rather than a loss of capability.
    (pack / "linked_settings.json").unlink()
    (pack / "hidden_dir").unlink()
    monkeypatch.setattr(sys, "argv", argv)
    assert scrub.main() == 0
    assert fake not in (pack / "normal.txt").read_text(encoding="utf-8")
