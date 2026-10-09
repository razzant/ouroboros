"""Schema/CLI-level tests for the CL-Bench launcher (no network, no docker)."""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import tempfile
from pathlib import Path

import pytest

from devtools.benchmarks.common import run_roots
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
    actors = json.loads(settings["OUROBOROS_SUBAGENTS"])
    assert actors["enabled"] is True
    # One actor row plus the three packet review seats, all on the solve model.
    rows = {marked: [row["route"]["target_id"] for row in actors["items"]
                     if bool(row.get("review_eligible")) is marked] for marked in (False, True)}
    assert rows == {False: [settings["OUROBOROS_MODEL"]], True: [settings["OUROBOROS_MODEL"]] * 3}
    assert all(row["delivery"] == "packet" for row in actors["items"] if row.get("review_eligible"))
    assert "OUROBOROS_REVIEWER_SLOTS" not in settings
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


def test_context_mode_template_and_child_env_carry_false_tombstone(tmp_path):
    import argparse

    base = tmp_path / "context-settings.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    for marker in (False, "false", "off", 0):
        base.write_text(json.dumps({
            "OUROBOROS_CONTEXT_MODE": "low",
            "OUROBOROS_CONTEXT_MODE_AUTO_LOW": marker,
        }), encoding="utf-8")
        rendered = run_clb.render_run_settings(
            base, run_dir, solve_model="openai/gpt-5.5",
            evolution=False, total_budget=1.0,
        )
        assert rendered["OUROBOROS_CONTEXT_MODE"] == "low"
        assert rendered["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"

    # A template mode is an explicit benchmark choice even when the tombstone was
    # omitted; the renderer adds it before the compatibility normalizer.
    base.write_text(json.dumps({"OUROBOROS_CONTEXT_MODE": "low"}), encoding="utf-8")
    rendered = run_clb.render_run_settings(
        base, run_dir, solve_model="openai/gpt-5.5",
        evolution=False, total_budget=1.0,
    )
    assert rendered["OUROBOROS_CONTEXT_MODE"] == "low"
    assert rendered["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"

    # A legacy true marker is never forwarded as true.
    base.write_text(json.dumps({
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true",
    }), encoding="utf-8")
    legacy = run_clb.render_run_settings(
        base, run_dir, solve_model="openai/gpt-5.5",
        evolution=False, total_budget=1.0,
    )
    assert legacy["OUROBOROS_CONTEXT_MODE"] == "max"
    assert legacy["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"

    args = argparse.Namespace(
        ouroboros_clone=str(tmp_path / "clone"),
        effort="low",
        or_provider="",
    )
    child_env = run_clb._sanitized_child_env(run_dir, {
        "TOTAL_BUDGET": 1.0,
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": False,
        "OUROBOROS_SUBAGENTS": run_clb.single_model_subagents_setting("openai/gpt-5.5"),
    }, args)
    assert child_env["OUROBOROS_CONTEXT_MODE"] == "low"
    assert child_env["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert json.loads(child_env["OUROBOROS_SUBAGENTS"])["items"][0]["route"][
        "target_id"
    ] == "openai/gpt-5.5"

    legacy_child_env = run_clb._sanitized_child_env(run_dir, {
        "TOTAL_BUDGET": 1.0,
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true",
    }, args)
    assert legacy_child_env["OUROBOROS_CONTEXT_MODE"] == "max"
    assert legacy_child_env["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"


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
            # This test pins manifest + settings RENDERING, not seed provenance. The gate
            # runs against the EXECUTION clone (v6.76.0), which here is a bare stub directory
            # with no git identity, so without the recorded escape it would refuse. The gate
            # itself is covered against a controlled repo by
            # test_benchmark_manifest_seed_gate_fails_closed_by_default.
            "--allow-dirty-seed",
            "--dry-run",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    run_dir = Path(payload["run_root"])
    assert run_dir.exists() and str(run_dir).startswith(str(tmp_path))

    rendered = json.loads((run_dir / "_run_settings.json").read_text(encoding="utf-8"))
    assert rendered["OPENROUTER_API_KEY"] == ""  # secrets blanked on disk
    assert rendered["OUROBOROS_MODEL"] == rendered["OUROBOROS_MODEL_LIGHT"]  # single-model pin
    assert json.loads(rendered["OUROBOROS_SUBAGENTS"])["items"][0]["route"][
        "target_id"
    ] == rendered["OUROBOROS_MODEL"]

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "continual_learning"
    # ...and the escape above really reached the gate instead of being ignored.
    assert manifest["seed_gate"]["allow_dirty_seed"] is True
    argv = payload["planned_invocations"][0]
    assert "--no-live-dashboard" in argv
    assert argv[argv.index("--max-workers") + 1] == "1"  # strict sequential default
    params = json.loads(argv[argv.index("--system-params") + 1])
    assert params["evolution"] is False
    assert params["max_workers"] == 4  # within-task subagent pool, not cross-task parallelism
    fidelity = manifest["extra"]["fidelity"]
    assert "OUROBOROS_SAFETY_MODE" in fidelity["declared_only_pinned_adapter_gap"]
    assert "OUROBOROS_REVIEW_ENFORCEMENT" in fidelity["declared_only_pinned_adapter_gap"]
    assert "OUROBOROS_SUBAGENTS" in fidelity["declared_only_pinned_adapter_gap"]
    assert manifest["available_subagents"]["items"][0]["route"]["target_id"] == \
        rendered["OUROBOROS_MODEL"]
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
            # Pins the per-phase invocation plan, not seed provenance: see the note in
            # test_dry_run_writes_manifest_and_blanked_settings.
            "--allow-dirty-seed",
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


def _pin_live_roots(tmp_path, monkeypatch, *, live_repo: Path) -> None:
    """Make `live_repo_roots()` return EXACTLY ``live_repo``, owning every contributor.

    `refuse_live_repo_clone` compares against the live runtime layout, and that layout has FOUR
    inputs: `$OUROBOROS_REPO_DIR`, `$OUROBOROS_DATA_DIR`, `$OUROBOROS_SETTINGS_PATH` (its parent
    is a live data root) and `run_roots._WORKSPACE_ROOT`, whose `repo` sibling counts too. The
    last one is a `__file__`-derived module constant, so NO amount of `setenv` reaches it — an
    earlier version of these tests set only the environment and therefore asserted a verdict it
    did not own. It passed in a worktree (where `_WORKSPACE_ROOT/repo` is a different tree) and
    failed in the advisory preflight sandbox, where the suite runs from
    `/tmp/ouroboros-preflight-*/repo` and that sibling IS the checkout under test. The
    production logic was right in both places; the test was reading ambient state.

    Tests here assert the gate's SHAPE, or a verdict every input of which they control. This is
    the second kind, and the callers assert the isolation actually took before relying on it.
    """
    monkeypatch.setattr(run_roots, "_WORKSPACE_ROOT", tmp_path / "workspace")
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(live_repo))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "workspace" / "data"))
    monkeypatch.delenv("OUROBOROS_SETTINGS_PATH", raising=False)


def test_live_repo_clone_refused(tmp_path, monkeypatch):
    live_repo = tmp_path / "workspace" / "repo"
    _pin_live_roots(tmp_path, monkeypatch, live_repo=live_repo)
    runner = _fake_runner(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_clb.main([
            "--runner-path", str(runner),
            "--ouroboros-clone", str(live_repo),
            "--dry-run",
        ])
    assert "never the LIVE repo" in str(exc.value)


def test_a_pinned_seed_may_be_the_launchers_own_checkout(tmp_path, monkeypatch):
    """The refusal is about the LIVE repo, not about "the tree I am executing from".

    Running a pinned seed's own `run_clb.py` and handing it that same seed is the recipe
    METHODOLOGY prescribes, and the field smoke had to clone the seed twice to get past a
    guard that compared against `Path(__file__).parents[3]`. Those two trees coincide only when
    the checkout IS the live one — which is a property of the runtime layout, never of where
    the launcher's source happens to sit.
    """
    _pin_live_roots(tmp_path, monkeypatch, live_repo=tmp_path / "elsewhere" / "repo")
    # The isolation HELD: this checkout is not the live repo in this test's world. Asserted
    # rather than assumed, so the case below can never pass vacuously.
    live = {root.expanduser().resolve(strict=False) for root in run_roots.live_repo_roots()}
    assert REPO_ROOT.resolve(strict=False) not in live

    assert run_clb.refuse_live_repo_clone(REPO_ROOT) == REPO_ROOT.resolve(strict=False)
    # ...and it really is the launcher's own checkout that was accepted.
    assert run_clb.REPO.resolve(strict=False) == REPO_ROOT.resolve(strict=False)


def test_the_launchers_own_checkout_is_refused_when_it_IS_the_live_repo(tmp_path, monkeypatch):
    """The mirror, in the same deterministic style: the two tests pin the real contract.

    Together they say the verdict tracks the LIVE RUNTIME LAYOUT and nothing else — the same
    seed path is accepted or refused purely according to whether it is the configured live
    repo. This is the case the advisory preflight sandbox hits for real (it runs the suite from
    a checkout that IS its `$OUROBOROS_REPO_DIR`), and the refusal there is CORRECT.
    """
    _pin_live_roots(tmp_path, monkeypatch, live_repo=REPO_ROOT)
    live = {root.expanduser().resolve(strict=False) for root in run_roots.live_repo_roots()}
    assert REPO_ROOT.resolve(strict=False) in live

    with pytest.raises(SystemExit) as exc:
        run_clb.refuse_live_repo_clone(REPO_ROOT)
    assert "never the LIVE repo" in str(exc.value)

    # The data-root SIBLING is an authority in its own right, not only the explicit env var —
    # a synthetic workspace, so this does not quietly depend on what the checkout is NAMED.
    seed = tmp_path / "workspace" / "repo"
    _pin_live_roots(tmp_path, monkeypatch, live_repo=tmp_path / "elsewhere" / "repo")
    with pytest.raises(SystemExit):
        run_clb.refuse_live_repo_clone(seed)


# ---------------------------------------------------------------- v6.76.0 (P2)

def test_collect_results_counts_auto_resolved_rows_without_hiding_them(tmp_path):
    """Instances closed by ANOTHER question's step are scored but agent-turn-less.

    They must be counted in the denominator AND disclosed separately, and the count
    is also why n_outcomes can exceed the requested --num-instances window.
    """
    run_dir = tmp_path / "run"
    rows = [
        (0, 1.0, "completed"),
        (1, 0.5, run_clb.AUTO_RESOLVED_STATUS),
        (2, 0.0, run_clb.AUTO_RESOLVED_STATUS),
    ]
    for index, reward, status in rows:
        qdir = run_dir / "traces" / "stateful_noevo" / "exploitable_poker" / f"q{index:03d}"
        qdir.mkdir(parents=True)
        (qdir / "task_outcome.json").write_text(json.dumps(
            {"domain": "exploitable_poker", "instance_index": index, "reward": reward,
             "success": None, "ouroboros_status": status,
             "cost_usd": 0.4 if status == "completed" else None}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(
        {"requested_task_ids": ["exploitable_poker:q000"], "extra": {"phases": "stateful_noevo"}}),
        encoding="utf-8")
    results = run_clb.collect_results(run_dir)
    cond = results["conditions"]["stateful_noevo"]
    assert cond["n_outcomes"] == 3 and cond["n_scored"] == 3       # nothing dropped
    assert cond["n_auto_resolved_no_agent_turn"] == 2              # honestly disclosed
    assert cond["auto_resolved_no_agent_turn_indices"] == [1, 2]
    assert cond["mean_reward"] == 0.5
    ledger = [json.loads(line) for line in (run_dir / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    statuses = sorted(row["ouroboros_status"] for row in ledger)
    assert statuses == ["auto_resolved_no_agent_turn", "auto_resolved_no_agent_turn", "completed"]


def test_dry_run_records_seed_escape_and_attestation_path(tmp_path):
    runner = _fake_runner(tmp_path)
    clone = _fake_clone(tmp_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_clb.main([
            "--runner-path", str(runner), "--ouroboros-clone", str(clone),
            "--path", "bridge", "--allow-dirty-seed", "--dry-run",
        ])
    assert rc == 0
    manifest = json.loads((Path(json.loads(buf.getvalue())["run_root"]) / "run_manifest.json")
                          .read_text(encoding="utf-8"))
    extra = manifest["extra"]
    assert extra["allow_dirty_seed"] is True
    # docker is the launcher default, and this clone is UNPATCHED: the disclosure must say the
    # run is unattested. Naming the patch because `--docker` was passed asserted a provenance
    # check that never ran — the same defect as a manifest naming a model that never ran. The
    # previous form of this assertion pinned that bug: it demanded the false claim.
    assert extra["runtime_attestation_available"] is False
    assert "UNATTESTED" in extra["runtime_attestation_path"]
    assert extra["adapter_operator_patches"]["patches"][
        "clb_docker_runtime_attestation.v6746"] is False


def test_attestation_field_claims_availability_not_that_the_check_ran(tmp_path):
    """The mirror — but it names the fact the probe actually has.

    Both directions matter: deriving the field from the tree is only correct if it still
    reports the patched path when the patch is there, otherwise the fix trades a false claim
    for a false denial. What this test no longer does is call it `runtime_attested`.

    INVERTED BUG-PINNING TEST. The previous form asserted `extra["runtime_attested"] is True`
    for a DRY RUN against a clone that merely contained an `_attest_runtime` DEFINITION — no
    container ever started, so nothing was attested. It encoded a false positive as the
    contract. The probe is a tree probe (owner-approved goal) and the field is now
    `runtime_attestation_available`, which is exactly what a source scan can establish.
    """
    runner = _fake_runner(tmp_path)
    (runner / "src" / "systems" / "ouroboros" / "_docker_launcher.py").write_text(
        "class DockerOuroborosEngine:\n"
        "    def __init__(self):\n        self.runtime_attestation: dict = {}\n"
        "    def _attest_runtime(self, clone):\n        pass\n",
        encoding="utf-8")
    clone = _fake_clone(tmp_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_clb.main([
            "--runner-path", str(runner), "--ouroboros-clone", str(clone),
            "--path", "bridge", "--allow-dirty-seed", "--dry-run",
        ])
    assert rc == 0
    extra = json.loads((Path(json.loads(buf.getvalue())["run_root"]) / "run_manifest.json")
                       .read_text(encoding="utf-8"))["extra"]
    assert extra["runtime_attestation_available"] is True
    assert "runtime_attested" not in extra, \
        "a dry run attested nothing; the manifest must not carry a field claiming it did"
    assert extra["runtime_attestation_evidence"] == (
        "static_source_scan of the --runner-path adapter checkout")
    assert "clb_docker_runtime_attestation" in extra["runtime_attestation_path"]
    assert "UNATTESTED" not in extra["runtime_attestation_path"]


def test_patch_probe_ignores_bare_env_name_mentions(tmp_path):
    """A marker must be PATCH-UNIQUE, not a bare env-var name.

    The pinned adapter may legitimately name `OUROBOROS_SAFETY_MODE` or
    `CLBENCH_SOLVE_DISABLED_TOOLS` in a comment or a docker `-e` passthrough list without the
    patch being applied. Keying detection on the bare name made the probe false-positive, and
    the direction is the dangerous one: the manifest would file the knob under
    `enforced_via_operator_patch` for a run that executed unenforced.
    """
    runner = _fake_runner(tmp_path)
    adapter = runner / "src" / "systems" / "ouroboros"
    (adapter / "_docker_launcher.py").write_text(
        "# forwards nothing; the -e list below is just a passthrough\n"
        'PASSTHROUGH = ["OUROBOROS_SAFETY_MODE", "OUROBOROS_RUNTIME_MODE"]\n'
        "# see also _attest_runtime in a newer adapter\n",
        encoding="utf-8")
    (adapter / "run_clbench_bridge_agent.py").write_text(
        "# CLBENCH_SOLVE_DISABLED_TOOLS is honored by an operator patch we did not apply\n"
        "DISABLED_TOOLS: list[str] = []\n",
        encoding="utf-8")
    probe = run_clb.adapter_patch_probe(runner)
    assert probe["patches"] == {
        "clb_docker_runtime_attestation.v6746": False,
        "clb_env_campaign_overrides.v6745": False,
        "clb_host_available_subagents.phase1d": False,
        "clb_disabled_tools_env.v6745": False,
    }
    assert probe["evidence"] == "static_source_scan"


def test_disabled_tools_gap_is_entrypoint_specific(tmp_path):
    """`--path standard` never executes the patched bridge module, so the knob is a GAP there.

    Verified against the live adapter: on the standard path the run goes run_benchmark.py ->
    system.py -> `_docker_launcher.submit()`, which hardcodes `"disabled_tools": []`; only
    `run_clbench_bridge_agent.py` (bridge) and `_live_bridge.py` pass DISABLED_TOOLS. Keying
    the verdict purely on the marker filed the knob as ENFORCED on the DEFAULT path against a
    patched checkout and claimed claude_code_edit was excluded from a task contract that
    excluded nothing.
    """
    runner = _fake_runner(tmp_path)
    adapter = runner / "src" / "systems" / "ouroboros"
    clone = _fake_clone(tmp_path)
    # A genuinely patched bridge module (patch-unique tokens, as the operator's patch writes).
    (adapter / "run_clbench_bridge_agent.py").write_text(
        "import os as _os\n"
        "# Operator patch 2026-07-23: honor CLBENCH_SOLVE_DISABLED_TOOLS from env\n"
        'DISABLED_TOOLS: list[str] = [t.strip() for t in '
        '_os.environ.get("CLBENCH_SOLVE_DISABLED_TOOLS", "").split(",") if t.strip()]\n',
        encoding="utf-8")

    def _fidelity_for(path: str) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_clb.main([
                "--runner-path", str(runner), "--ouroboros-clone", str(clone),
                "--path", path, "--allow-dirty-seed", "--dry-run",
            ])
        assert rc == 0
        return json.loads(buf.getvalue())["fidelity"]

    bridge = _fidelity_for("bridge")
    assert "CLBENCH_SOLVE_DISABLED_TOOLS" in bridge["enforced_via_operator_patch"]

    standard = _fidelity_for("standard")
    assert "CLBENCH_SOLVE_DISABLED_TOOLS" not in standard["enforced_via_operator_patch"], \
        "the standard entrypoint never imports the patched bridge module"
    gap = standard["declared_only_pinned_adapter_gap"]["CLBENCH_SOLVE_DISABLED_TOOLS"]
    assert "standard" in gap["status"] and "disabled_tools=[]" in gap["status"]
    # The patch IS present in the checkout — the probe must keep saying so, honestly.
    assert standard["adapter_operator_patches"]["clb_disabled_tools_env.v6745"] is True


def test_fidelity_follows_the_execution_clone_not_the_pinned_commit_constant(tmp_path):
    """The three "declared-only" knobs are enforced iff their operator patch is in the clone.

    The report described the PINNED adapter, so on a patched clone it announced a gap that the
    patches had closed: it claimed safety `full`, advisory enforcement and no `claude_code_edit`
    exclusion for a run that was really running `light`/`blocking` with all nine tools disabled.
    Understating enforcement is the safer direction but it is still a manifest that does not
    match its run, and a reader who trusts it discounts a stricter run than they were given.
    """
    runner = _fake_runner(tmp_path)
    adapter = runner / "src" / "systems" / "ouroboros"
    clone = _fake_clone(tmp_path)

    def _fidelity_of(*extra_args: str):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_clb.main([
                "--runner-path", str(runner), "--ouroboros-clone", str(clone),
                "--path", "bridge", "--allow-dirty-seed", "--dry-run",
                *extra_args,
            ])
        assert rc == 0
        return json.loads(buf.getvalue())["fidelity"]

    unpatched = _fidelity_of()
    for knob in ("OUROBOROS_SAFETY_MODE", "OUROBOROS_REVIEW_ENFORCEMENT", "OUROBOROS_SUBAGENTS",
                 "CLBENCH_SOLVE_DISABLED_TOOLS"):
        assert knob in unpatched["declared_only_pinned_adapter_gap"]
        assert knob not in unpatched["enforced_via_operator_patch"]

    host_unpatched = _fidelity_of("--no-docker")
    assert "OUROBOROS_SUBAGENTS" in host_unpatched["declared_only_pinned_adapter_gap"]
    assert "OUROBOROS_SAFETY_MODE" in host_unpatched["enforced_via_operator_patch"]

    # Now apply the two forwarding patches, as the operator does before a real run. The
    # fixtures carry the patches' OWN tokens (comment tag + the exact expression they
    # introduce), not bare env-var names — see
    # test_patch_probe_ignores_bare_env_name_mentions for why the difference is load-bearing.
    (adapter / "_docker_launcher.py").write_text(
        "        # Operator env overrides (campaign knobs). Parity defaults stay authoritative\n"
        '        for _k in ("OUROBOROS_RUNTIME_MODE", "OUROBOROS_REVIEW_ENFORCEMENT",\n'
        '                   "OUROBOROS_SAFETY_MODE", "OUROBOROS_SUBAGENTS"):\n'
        "            _v = os.environ.get(_k)\n"
        "            if _v:\n                ov[_k] = _v\n",
        encoding="utf-8")
    (adapter / "run_clbench_bridge_agent.py").write_text(
        "import os as _os\n"
        "# Operator patch 2026-07-23: honor CLBENCH_SOLVE_DISABLED_TOOLS from env\n"
        'DISABLED_TOOLS: list[str] = [t.strip() for t in '
        '_os.environ.get("CLBENCH_SOLVE_DISABLED_TOOLS", "").split(",") if t.strip()]\n',
        encoding="utf-8")

    patched = _fidelity_of()
    assert patched["declared_only_pinned_adapter_gap"] == {}
    for knob in ("OUROBOROS_SAFETY_MODE", "OUROBOROS_REVIEW_ENFORCEMENT", "OUROBOROS_SUBAGENTS",
                 "CLBENCH_SOLVE_DISABLED_TOOLS"):
        assert knob in patched["enforced_via_operator_patch"]
    # The declared values are the ones the patched clone really applies.
    assert patched["enforced_via_operator_patch"]["OUROBOROS_SAFETY_MODE"]["declared"] == "light"
    assert patched["enforced_via_operator_patch"][
        "OUROBOROS_REVIEW_ENFORCEMENT"]["declared"] == "blocking"
    assert json.loads(patched["enforced_via_operator_patch"][
        "OUROBOROS_SUBAGENTS"]["declared"])["items"][0]["route"]["target_id"] == \
        "anthropic/claude-sonnet-4.6"
    assert "claude_code_edit" in patched["enforced_via_operator_patch"][
        "CLBENCH_SOLVE_DISABLED_TOOLS"]["declared"]

    (adapter / "_launcher.py").write_text(
        "# Canonical benchmark actor bytes are authored by run_clb.py through Phase-1A\n"
        'OUROBOROS_SUBAGENTS=os.environ.get("OUROBOROS_SUBAGENTS", "")\n',
        encoding="utf-8",
    )
    host_patched = _fidelity_of("--no-docker")
    assert "OUROBOROS_SUBAGENTS" in host_patched["enforced_via_operator_patch"]
    assert host_patched["declared_only_pinned_adapter_gap"] == {}


def test_seed_gate_binds_to_the_execution_clone_and_records_the_launcher_separately(tmp_path):
    """The gate must judge the checkout the RUN executes from — `--ouroboros-clone`, which the
    external adapter boots its agent servers from — not this launcher's own tree. They are
    different trees by construction (the clone may never be the live repo), so gating REPO let
    a DIRTY execution seed pass whenever the launcher happened to be clean.

    Asserts the gate's SHAPE and its TARGET, never its verdict: the verdict would depend on
    the ambient checkout, which is exactly what this binding is about.
    """
    runner = _fake_runner(tmp_path)
    clone = _fake_clone(tmp_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_clb.main([
            "--runner-path", str(runner), "--ouroboros-clone", str(clone),
            "--allow-dirty-seed", "--dry-run",
        ])
    assert rc == 0
    manifest = json.loads((Path(json.loads(buf.getvalue())["run_root"]) / "run_manifest.json")
                          .read_text(encoding="utf-8"))
    # The gate's subject IS the execution clone...
    assert manifest["source"]["repo_dir"] == str(clone.resolve())
    assert manifest["extra"]["seed_gate_target"] == "execution_clone"
    assert manifest["extra"]["execution_clone"] == str(clone.resolve())
    # ...and the launcher's own provenance is recorded ALONGSIDE it, not instead of it.
    assert manifest["extra"]["launcher_provenance"]["repo_dir"] == str(REPO_ROOT)
    gate = manifest["seed_gate"]
    assert gate["require_clean"] is False and gate["allow_dirty_seed"] is True
    assert gate["ok"] is (not gate["reason"])
    # The finalization seam recorded how this run ENDED, on the dry-run path too.
    assert manifest["extra"]["outcome"] == "dry_run" and manifest["extra"]["exit_code"] == 0


def test_seed_refusal_leaves_a_durable_manifest_with_the_real_exit_code(tmp_path):
    """A refused run must be as auditable as an admitted one: the payload reaches disk BEFORE
    the refusal propagates. The clone is a bare directory, so the verdict is a property of the
    fixture rather than of the ambient checkout."""
    clone = tmp_path / "execution-clone"
    (clone / "devtools" / "benchmarks" / "common").mkdir(parents=True)
    out = tmp_path / "clb-refused"
    with pytest.raises(RuntimeError, match="seed provenance gate failed"):
        run_clb.main(["--ouroboros-clone", str(clone), "--out-dir", str(out), "--dry-run"])
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed_gate"]["reason"] == "seed_identity_unavailable"
    assert manifest["extra"]["outcome"] == "refused"
    # A plain exception escapes `main()`, so CPython exits 1 — and the record says 1.
    assert manifest["extra"]["exit_code"] == 1
    # Nothing was rendered or launched: the refusal precedes every mutation.
    assert not (out / "_run_settings.json").exists()


def test_out_dir_is_confined_against_the_execution_clone_not_only_the_launcher(tmp_path):
    """`--out-dir` was validated against REPO alone, so a run root INSIDE the execution clone
    passed — admission artefacts landing in the very seed whose cleanliness the gate is about
    to attest, which also dirties it. The clone here is an alternate one under tmp_path, never
    the ambient checkout, so the verdict is a property of the fixture."""
    runner = _fake_runner(tmp_path)
    clone = _fake_clone(tmp_path)

    with pytest.raises(ValueError) as refused:
        run_clb.main([
            "--runner-path", str(runner), "--ouroboros-clone", str(clone),
            "--out-dir", str(clone / "bench_runs" / "inside-the-seed"),
            "--allow-dirty-seed", "--dry-run",
        ])
    assert "must not be under" in str(refused.value)
    assert not (clone / "bench_runs").exists()          # refused before anything was created

    # The launcher's own checkout is still an authority (both are checked, not either/or)...
    with pytest.raises(ValueError):
        run_clb.main([
            "--runner-path", str(runner), "--ouroboros-clone", str(clone),
            "--out-dir", str(REPO_ROOT / "bench_runs" / "inside-the-launcher"),
            "--allow-dirty-seed", "--dry-run",
        ])
    assert not (REPO_ROOT / "bench_runs" / "inside-the-launcher").exists()

    # ...and a root outside BOTH is admitted, with no directory created before the manifest.
    out = tmp_path / "runs" / "clb"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_clb.main([
            "--runner-path", str(runner), "--ouroboros-clone", str(clone),
            "--out-dir", str(out), "--allow-dirty-seed", "--dry-run",
        ])
    assert rc == 0
    assert json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["extra"]["outcome"] == "dry_run"


def _bridge_helpers_from_patch():
    """Exec the helpers the outcomes patch ADDS, so their behaviour is testable in-tree.

    The plugin lives in the external CL-Bench checkout, so the patch file is the only copy of
    this code we own. Asserting on its TEXT (as the tracking test below does) cannot tell an
    atomic write from a torn one; running it can.
    """
    patch = (REPO_ROOT / "devtools" / "benchmarks" / "continual_learning" / "operator_patches"
             / "clb_multi_instance_outcomes.v6746.patch").read_text(encoding="utf-8")
    added = [line[1:] for line in patch.splitlines()
             if line.startswith("+") and not line.startswith("+++")]
    start = next(i for i, line in enumerate(added) if line.startswith("def _atomic_write_json"))
    wanted = {"_atomic_write_json", "_write_instance_outcome", "_refresh_instance_reward"}
    for end in range(len(added), start, -1):
        block = "from __future__ import annotations\n" + "\n".join(added[start:end])
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        if wanted <= {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}:
            namespace = {"json": json, "os": os, "tempfile": tempfile, "Path": Path}
            exec(compile(block, "<clb_multi_instance_outcomes.patch>", "exec"), namespace)
            return namespace
    raise AssertionError("the outcomes patch no longer adds the three outcome-row helpers")


def test_an_interrupted_late_reward_update_cannot_destroy_the_recorded_outcome_row(
        tmp_path, monkeypatch):
    """`task_outcome.json` is the ONLY record that an instance existed.

    `collect_results` reads the denominator and every row's provenance from it, so a torn write
    does not lose an update, it loses the ROW — the file parses as `{}` and the instance
    vanishes from the very denominator this change exists to preserve. The late-reward fill
    rewrites rows that are ALREADY VALID, which is exactly when that costs the most.

    Deliberately NOT "the file still exists": the assertion is that the PREVIOUS ROW'S CONTENT
    survives, byte for byte, an update interrupted at the moment of publication.
    """
    helpers = _bridge_helpers_from_patch()
    run_dir = tmp_path / "q000"
    helpers["_write_instance_outcome"](run_dir, "exploitable_poker", 0, {"reward": None,
                                                                        "success": None},
                                       ouroboros_status="completed", cost_usd=1.25)
    path = run_dir / "task_outcome.json"
    before = path.read_bytes()
    assert json.loads(before)["ouroboros_status"] == "completed"

    def _interrupted(src, dst):
        raise KeyboardInterrupt("killed between the temp write and the rename")

    monkeypatch.setattr(os, "replace", _interrupted)
    with pytest.raises(KeyboardInterrupt):
        helpers["_refresh_instance_reward"](run_dir, {"reward": 1.0, "success": True})

    assert path.read_bytes() == before
    assert [p.name for p in run_dir.iterdir()] == ["task_outcome.json"]

    # And the update itself still works, preserving the row's agent-turn provenance.
    monkeypatch.undo()
    assert helpers["_refresh_instance_reward"](run_dir, {"reward": 1.0, "success": True}) is True
    row = json.loads(path.read_text(encoding="utf-8"))
    assert (row["reward"], row["success"], row["reward_source"]) == (
        1.0, True, "late_instance_outcomes_fill")
    assert (row["ouroboros_status"], row["cost_usd"]) == ("completed", 1.25)


def test_clb_operator_patches_are_tracked_with_an_unambiguous_apply_order():
    patches = REPO_ROOT / "devtools" / "benchmarks" / "continual_learning" / "operator_patches"
    outcomes = (patches / "clb_multi_instance_outcomes.v6746.patch").read_text(encoding="utf-8")
    assert "src/systems/ouroboros/run_clbench_bridge_agent.py" in outcomes
    assert "auto_resolved_no_agent_turn" in outcomes
    assert "instance_outcomes" in outcomes
    # a late (evaluate-time) reward fills a null row without rewriting its provenance
    assert "late_instance_outcomes_fill" in outcomes
    # BOTH outcome-row writers go through the atomic seam, and neither truncates in place.
    # Checked on the ADDED lines only: the removed ones still show the torn-write original.
    added = "\n".join(line[1:] for line in outcomes.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    assert "_atomic_write_json" in added
    assert "os.replace(tmp, path)" in added
    assert ".write_text(" not in added
    attest = (patches / "clb_docker_runtime_attestation.v6746.patch").read_text(encoding="utf-8")
    assert "src/systems/ouroboros/_docker_launcher.py" in attest
    # ARITY, not just the name. The shared helper takes `repo_dir` as a REQUIRED POSITIONAL
    # (that is how the local HEAD is reported), so a name-only assertion happily passes a
    # patch that raises TypeError on every docker-engine run — the launcher default.
    from devtools.benchmarks.common.manifests import runtime_attestation
    required = list(inspect.signature(runtime_attestation).parameters.values())[:2]
    assert [p.name for p in required] == ["base_url", "repo_dir"]
    assert all(p.default is inspect.Parameter.empty for p in required)
    assert "runtime_attestation(self._server.base_url, pathlib.Path(clone)," in attest
    # ...and the signature the patch DOCUMENTS must name it too, or the next reader copies
    # the broken call back out of the docstring.
    assert "runtime_attestation(base_url, repo_dir, *, expected_version=" in attest
    assert "OBO_ALLOW_EVOLVED_VOLUME" in attest
    readme = (patches / "README.md").read_text(encoding="utf-8")
    # The stack order must name the three predecessor bridge patches explicitly.
    for name in ("run_clbench_bridge_agent.v6560.patch", "clb_acceptance_claims.v674.patch",
                 "clb_disabled_tools_env.v6745.patch"):
        assert name in readme
    order = readme.index("Apply order (unambiguous)")
    assert order < readme.index("clb_multi_instance_outcomes.v6746.patch")


def test_clb_methodology_states_poker_instance_count_and_outcome_semantics():
    text = (REPO_ROOT / "devtools" / "benchmarks" / "continual_learning"
            / "METHODOLOGY.md").read_text(encoding="utf-8")
    assert "`exploitable_poker` = 120" in text
    assert "--num-instances 120" in text
    assert "auto_resolved_no_agent_turn" in text
    assert "Accepted overshoot (owner decision)" in text
    assert "late_instance_outcomes_fill" in text
    assert "Disclosed residual" in text


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
