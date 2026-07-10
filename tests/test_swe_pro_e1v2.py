"""Focused tests for the SWE-bench-Pro e1v2 harness producer/consumer contracts.

Split out of tests/test_devtools_benchmarks.py to keep that module focused and
small. Covers the run_pro -> auto_run timeline handoff (infra-flag persistence
and stop/skip semantics), the `--cadence off` settings contract, and the
build_predictions leaderboard-shaped output schema.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_BASH_AVAILABLE = sys.platform != "win32" and shutil.which("bash") is not None


def test_e1v2_timeline_row_persists_infra_flags():
    """Producer side of the run_pro -> auto_run handoff: the timeline row MUST carry
    the infra non-execution markers, else auto_run cannot stop on a secret refusal."""
    from devtools.benchmarks.swe_bench_pro.e1v2.run_pro import build_timeline_row

    res = {"model_patch": "", "timed_out": False, "infra_suspect": True,
           "secret_opt_in_required": True, "libc_skip": "musl:vol", "health_rollback": False,
           "api_errors": 0, "api_ctx": 0, "refl_line": "", "quiet_line": "",
           "selfedit": {}, "evolution_degraded": False, "absorb_reason": ""}
    row = build_timeline_row(1, "inst", res, 0.0, ["INFRA"])
    assert row["infra_suspect"] is True
    assert row["secret_opt_in_required"] is True
    assert row["libc_skip"] == "musl:vol"


def test_e1v2_auto_run_one_stops_on_secret_and_skips_infra(tmp_path, monkeypatch):
    """Consumer side: a secret-opt-in refusal hard-stops; an infra skip is non-LEGIT
    (patch_bytes=None), never snapshotted as a completed last-good."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    args = _types.SimpleNamespace(
        total_budget=10.0, per_task_cost=5.0, task_wall_timeout=9000,
        volume_suffix="", full_set=False, csv="", settings="", solve_model="",
        model_name="", review_slots=None, review_effort="", solve_timeout=None,
        memory_mode="", baseline=False,
    )

    def _popen_writing(payload):
        # run_pro is launched via subprocess.Popen(..., start_new_session=True); the
        # fake writes the timeline (as run_pro would) and completes without timing out.
        class _P:
            def __init__(self, *a, **k):
                (tmp_path / "timeline.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
                self.pid = 1234

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        return _P

    # secret-injection refusal -> hard stop (config error, not a transient)
    monkeypatch.setattr(auto_run.subprocess, "Popen",
                        _popen_writing({"patch_bytes": 0, "api_errors": 0, "instance_id": "x",
                                        "secret_opt_in_required": True}))
    with pytest.raises(SystemExit):
        auto_run.run_one(1, tmp_path, args, attempt=1)

    # generic infra skip -> non-LEGIT (pb=None), so it is retried/stopped not counted ok
    monkeypatch.setattr(auto_run.subprocess, "Popen",
                        _popen_writing({"patch_bytes": 0, "api_errors": 0, "instance_id": "y",
                                        "infra_suspect": True}))
    r = auto_run.run_one(1, tmp_path, args, attempt=1)
    assert r["pb"] is None
    assert r["permanent_skip"] is False


def test_e1v2_cadence_off_disables_post_task_evolution(tmp_path):
    """`--cadence off` must disable evolution via the documented POST_TASK_EVOLUTION
    contract (false), not leave it 'true' relying on a downstream cadence guard."""
    from devtools.benchmarks.swe_bench_pro.e1v2.run_pro import derive_run_settings

    base = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2" / "settings_base.json"
    off_dir = tmp_path / "off"; off_dir.mkdir()
    on_dir = tmp_path / "on"; on_dir.mkdir()
    p_off = derive_run_settings(str(base), off_dir, "m", 10.0, 5.0,
                                post_task_evolution=True, cadence="off")
    p_on = derive_run_settings(str(base), on_dir, "m", 10.0, 5.0,
                               post_task_evolution=True, cadence="every_n:1")
    assert json.loads(p_off.read_text(encoding="utf-8"))["OUROBOROS_POST_TASK_EVOLUTION"] == "false"
    assert json.loads(p_on.read_text(encoding="utf-8"))["OUROBOROS_POST_TASK_EVOLUTION"] == "true"


def test_e1v2_build_predictions_emits_leaderboard_schema(tmp_path, monkeypatch):
    """build_predictions rows must carry the leaderboard-shaped model_name_or_path,
    not just {instance_id, model_patch}, or the artifact is harness-incompatible."""
    import importlib
    bp = importlib.import_module("devtools.benchmarks.swe_bench_pro.e1v2.build_predictions")

    # Point the consolidated run root at a temp tree with one patched instance.
    full = tmp_path / "pro_e1_full"
    (full / "inst__a").mkdir(parents=True)
    (full / "inst__a" / "patch.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")
    csv_path = tmp_path / "order.csv"
    csv_path.write_text("idx,instance_id\n1,inst__a\n", encoding="utf-8")
    out_path = tmp_path / "preds.jsonl"
    monkeypatch.setattr(bp, "FULL", full)
    monkeypatch.setattr(bp, "CSV", csv_path)
    monkeypatch.setattr(
        bp.sys, "argv",
        ["build_predictions.py", "--start", "1", "--end", "1",
         "--out", str(out_path), "--model-name", "ouroboros-e1-pro-test"],
    )
    assert bp.main() == 0
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and set(rows[0]) == {"instance_id", "model_name_or_path", "model_patch"}
    assert rows[0]["model_name_or_path"] == "ouroboros-e1-pro-test"


def test_e1v2_resume_result_no_docker(tmp_path, monkeypatch):
    """The RESUME path rebuilds the result from an existing patch.diff WITHOUT any
    Docker call (no image pull / no state read), else it reintroduces the image-pull
    stall this hardening removes."""
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    def _boom(*a, **k):
        raise AssertionError("resume_result must not touch Docker")

    monkeypatch.setattr(run_pro, "read_spent_usd", _boom)
    monkeypatch.setattr(run_pro, "docker_pull_if_missing", _boom)

    cid_dir = tmp_path / "inst__a"
    cid_dir.mkdir()
    assert run_pro.resume_result("inst__a", cid_dir, "m") is None          # no patch
    (cid_dir / "patch.diff").write_text("", encoding="utf-8")
    assert run_pro.resume_result("inst__a", cid_dir, "m") is None          # empty patch
    (cid_dir / "patch.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")
    res = run_pro.resume_result("inst__a", cid_dir, "ouroboros-x")
    assert res and res["model_patch"].startswith("diff --git")
    assert res["model_name_or_path"] == "ouroboros-x"


def test_e1v2_auto_run_one_timeout_cleans_up_and_continues(tmp_path, monkeypatch):
    """A run_pro wall-timeout must kill the process group, remove leftover obopro
    containers, and STILL return the LEGIT task from the timeline run_pro wrote
    BEFORE teardown — not a phantom failure that gets re-pulled/re-solved."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    args = _types.SimpleNamespace(
        total_budget=10.0, per_task_cost=5.0, task_wall_timeout=1,
        volume_suffix="", full_set=False, csv="", settings="", solve_model="",
        model_name="", review_slots=None, review_effort="", solve_timeout=None,
        memory_mode="", baseline=False,
    )
    tl = tmp_path / "timeline.jsonl"

    class FakeProc:
        def __init__(self, *a, **k):
            # run_pro writes the durable row before the teardown that then hangs.
            tl.write_text(json.dumps({"patch_bytes": 1234, "api_errors": 0,
                                      "instance_id": "z"}) + "\n", encoding="utf-8")
            self.pid = 4242424

        def wait(self, timeout=None):
            raise auto_run.subprocess.TimeoutExpired(cmd="run_pro", timeout=timeout)

        def kill(self):
            pass

    killed = {"tree": False}
    cleaned = {"rm": False}
    monkeypatch.setattr(auto_run.subprocess, "Popen", FakeProc)
    # cross-platform process-tree kill is routed through platform_layer; mock it.
    monkeypatch.setattr(auto_run, "kill_process_tree", lambda proc: killed.__setitem__("tree", True))
    monkeypatch.setattr(auto_run, "_rm_obopro_containers",
                        lambda *_a, **_k: cleaned.__setitem__("rm", True))

    r = auto_run.run_one(7, tmp_path, args, attempt=1)
    assert killed["tree"] is True and cleaned["rm"] is True
    assert r["pb"] == 1234 and r["iid"] == "z"
    assert r["permanent_skip"] is False


def test_e1v2_run_instance_runtime_mode_passthrough(tmp_path, monkeypatch):
    """run_instance forwards `-e OUROBOROS_RUNTIME_MODE` ONLY when --runtime-mode is
    explicit; when omitted the seed settings profile drives it (not a forced 'pro')."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    monkeypatch.setenv("OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS", "1")
    # bench1 port: docker_pull_if_missing returns PRESENCE (False = image_unavailable
    # infra skip before the docker run), so the fake must return True.
    monkeypatch.setattr(run_pro, "docker_pull_if_missing", lambda img: True)
    monkeypatch.setattr(run_pro, "image_libc", lambda img: "glibc")
    monkeypatch.setattr(run_pro, "volume_exists", lambda name: True)
    monkeypatch.setattr(run_pro, "kill_container", lambda name: None)

    captured = {}

    def fake_run(cmd, **kw):
        # Capture ONLY the solve `docker run` argv: the bench1 result collector also
        # issues `docker image inspect` afterwards, which must not clobber the capture.
        if list(cmd[:2]) == ["docker", "run"]:
            captured["cmd"] = list(cmd)
        for a in cmd:  # write the patch into the host dir mounted at /out
            if isinstance(a, str) and a.endswith(":/out"):
                Path(a[: -len(":/out")], "patch.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")
        return _types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_pro.subprocess, "run", fake_run)

    row = {"dockerhub_tag": "t", "base_commit": "b", "repo": "r/r", "repo_language": "python",
           "problem_statement": "p", "requirements": "", "interface": ""}
    base = dict(out_dir=str(tmp_path), self_improve=False, model_name="m", mem_limit="",
                solve_model="openai/gpt-5.5", per_task_cost=5.0, solve_timeout=10, absorb_max=10,
                reflect_min=1, reflect_max=1, quiet_stable=1, memory_mode="empty", disable_tools="x")

    run_pro.run_instance("inst__a", row, _types.SimpleNamespace(runtime_mode="light", **base),
                         "key", tmp_path / "seed.json", 5.0)
    assert "OUROBOROS_RUNTIME_MODE=light" in " ".join(captured["cmd"])

    run_pro.run_instance("inst__a", row, _types.SimpleNamespace(runtime_mode="", **base),
                         "key", tmp_path / "seed.json", 5.0)
    assert "OUROBOROS_RUNTIME_MODE" not in " ".join(captured["cmd"])


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash required for strip_gold_history.sh")
def test_e1v2_strip_gold_history_keeps_base_and_drops_future(tmp_path):
    """strip_gold_history.sh leaves base reachable (capture_patch.sh diffs against it)
    while making the future/gold commit unreachable and unprintable (issue #93)."""
    repo = tmp_path / "app"
    repo.mkdir()

    def g(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)

    g("init", "-q")
    g("config", "user.email", "t@t.t"); g("config", "user.name", "t")
    (repo / "f.txt").write_text("base\n"); g("add", "-A"); g("commit", "-qm", "base")
    base = g("rev-parse", "HEAD").stdout.strip()
    (repo / "f.txt").write_text("gold fix\n"); g("add", "-A"); g("commit", "-qm", "gold fix")
    future = g("rev-parse", "HEAD").stdout.strip()
    g("tag", "goldtag"); g("branch", "dev")

    script = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "strip_gold_history.sh"
    r = subprocess.run(["bash", str(script), str(repo), base], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # base still resolvable for capture_patch.sh
    assert g("rev-parse", "--verify", base + "^{commit}").returncode == 0
    # no surviving ref reaches beyond base, and the gold commit object is gone
    assert g("rev-list", "--all", "--not", base).stdout.strip() == ""
    assert g("cat-file", "-e", future).returncode != 0


def test_e1v2_entrypoint_solve_argv_pins_workspace_and_budget_metadata():
    """Static harness contract (v6.56.0): the container solve invocation runs /app
    as the ACTIVE EXTERNAL WORKSPACE by default (empty override = legacy mode),
    carries the uncapped-cost budget_profile via --task-metadata-json, and the
    per-task memory default is an EMPTY child drive."""
    entry = (
        REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2" / "entrypoint_pro.sh"
    ).read_text(encoding="utf-8")
    # /app default via ${VAR-default} (an EXPLICIT empty string keeps legacy mode).
    assert 'OBO_SOLVE_WORKSPACE_ROOT="${OBO_SOLVE_WORKSPACE_ROOT-/app}"' in entry
    assert '--workspace "$OBO_SOLVE_WORKSPACE_ROOT"' in entry
    # Budget metadata: until_deadline pacing + no in-task cost stop.
    assert "--task-metadata-json" in entry
    assert '"improvement_policy": "until_deadline"' in entry
    assert '"cost_hard_stop_pct": 0' in entry
    # Fresh child memory drive is the explicit entrypoint default.
    assert 'OBO_MEMORY_MODE="${OBO_MEMORY_MODE:-empty}"' in entry
    # The solve args are a bash array (quoting-safe), not an interpolated string.
    assert "SOLVE_ARGS=(" in entry and '"${SOLVE_ARGS[@]}"' in entry


def test_workspace_allowlist_includes_acting_integration_tools():
    """v6.56.0 owner-approved registry change: a workspace parent can absorb acting
    children's patches (integrate) and compare best-of-N candidates."""
    from ouroboros.tools.registry import _WORKSPACE_ALLOWED_TOOLS

    assert "integrate_subagent_patch" in _WORKSPACE_ALLOWED_TOOLS
    assert "compare_subagent_patches" in _WORKSPACE_ALLOWED_TOOLS


def test_e1v2_ensure_util_image_preflights_pull_never_dependency(monkeypatch):
    """v6.56.0 review r6: snapshot/restore/state-read all use `--pull=never alpine:3`.
    ensure_util_image must be a no-op when present, and FAIL LOUD (not silently
    leave the image absent → empty-volume restores) when it is missing and unpullable."""
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    calls = {"pull": 0}

    # (a) already present → no pull, no raise
    monkeypatch.setattr(run_pro, "_image_present", lambda img=run_pro.UTIL_IMAGE: True)
    run_pro.ensure_util_image()
    assert calls["pull"] == 0

    # (b) absent and pull fails → RuntimeError (fail closed)
    monkeypatch.setattr(run_pro, "_image_present", lambda img=run_pro.UTIL_IMAGE: False)

    def _fake_pull(cmd, **kw):
        calls["pull"] += 1
        import types as _t
        return _t.SimpleNamespace(returncode=1, stdout=b"", stderr=b"no network")

    monkeypatch.setattr(run_pro.subprocess, "run", _fake_pull)
    with pytest.raises(RuntimeError, match="utility image"):
        run_pro.ensure_util_image()
    assert calls["pull"] == 1


def test_e1v2_restore_skips_missing_snapshot_and_reports_failure(tmp_path, monkeypatch):
    """v6.56.0 review r6: restore must not recreate an EMPTY volume when its snapshot
    tgz is absent (that would silently blank the retry state); it skips and reports False."""
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    # No .tgz files exist in tmp_path → both volumes skipped, no docker calls, returns False.
    called = {"docker": 0}
    monkeypatch.setattr(auto_run.subprocess, "run",
                        lambda *a, **k: called.__setitem__("docker", called["docker"] + 1))
    ok = auto_run.restore(tmp_path, "-testsuf")
    assert ok is False
    assert called["docker"] == 0  # never wiped/recreated a volume with no snapshot to restore
