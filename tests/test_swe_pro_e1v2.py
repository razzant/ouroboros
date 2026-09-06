"""Focused tests for the SWE-bench-Pro e1v2 harness producer/consumer contracts.

Split out of tests/test_devtools_benchmarks.py to keep that module focused and
small. Covers the run_pro -> auto_run timeline handoff (infra-flag persistence
and stop/skip semantics), the `--cadence off` settings contract, and the
build_predictions leaderboard-shaped output schema.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_BASH_AVAILABLE = sys.platform != "win32" and shutil.which("bash") is not None


def _real_admit_without_clean_seed(module, monkeypatch, *, on_call=None):
    """Route a launcher through the REAL `admit_benchmark_run`, ambient-independently.

    These tests are about ORDERING and the run record, not about the seed gate's verdict (which
    has its own tests). Forcing `require_clean=False` — instead of replacing the admission seam
    with a stub — keeps the assertions independent of whether the ambient checkout happens to be
    clean while still exercising the real 'persist the manifest before enforcement' write.
    """
    real = module.admit_benchmark_run

    def _admit(manifest_path, **kwargs):
        if on_call is not None:
            on_call(kwargs)
        kwargs["require_clean"] = False
        return real(manifest_path, **kwargs)

    monkeypatch.setattr(module, "admit_benchmark_run", _admit)


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
    # Budget metadata: fixed policy + explicit pass cap, no in-task cost stop.
    assert "--task-metadata-json" in entry
    assert '"improvement_policy": "fixed"' in entry
    assert '"max_improvement_passes": 6' in entry
    assert '"cost_hard_stop_pct": 0' in entry
    # Fresh child memory drive is the explicit entrypoint default.
    assert 'OBO_MEMORY_MODE="${OBO_MEMORY_MODE:-empty}"' in entry
    # The solve args are a bash array (quoting-safe), not an interpolated string.
    assert "SOLVE_ARGS=(" in entry and '"${SOLVE_ARGS[@]}"' in entry


def test_all_live_swe_producers_share_the_methodology_tool_default():
    default = (
        "web_search,browse_page,browser_action,analyze_screenshot,vlm_query,"
        "view_image,youtube_transcript,claude_code_edit,switch_model"
    )
    base = REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2"
    run_pro = (base / "run_pro.py").read_text(encoding="utf-8")
    entrypoint = (base / "entrypoint_pro.sh").read_text(encoding="utf-8")
    probe = (base / "orchestrate_probe.py").read_text(encoding="utf-8")

    assert f'default="{default}"' in run_pro
    assert f'default="{default}"' in probe
    assert f'OBO_DISABLE_TOOLS="${{OBO_DISABLE_TOOLS:-{default}}}"' in entrypoint
    names = set(default.split(","))
    assert {"youtube_transcript", "switch_model", "claude_code_edit"} <= names
    assert "schedule_subagent" not in names  # same-model decomposition is part of the method


def test_workspace_parent_sees_acting_integration_tools(tmp_path):
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
    names = set(registry.available_tools())
    assert {"integrate_subagent_patch", "compare_subagent_patches"} <= names


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


# --------------------------------------------------------------------------------------
# v6.75.0 (P1) — seed provenance inside the evolving volume + the manifest lifecycle.
# --------------------------------------------------------------------------------------


def test_e1v2_entrypoint_stamps_the_seed_without_ever_re_seeding():
    """The seeding CONDITION must stay: an existing /obo-repo/.git is an EVOLVED volume and
    re-seeding it would erase the agent's own reviewed commits. The stamp is written INSIDE that
    same branch (and inside .git, so it never shows up as untracked working-tree dirt)."""
    entry = (
        REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2" / "entrypoint_pro.sh"
    ).read_text(encoding="utf-8")
    assert "[ -e /obo-repo/.git ] || {" in entry
    seed_branch = entry.split("[ -e /obo-repo/.git ] || {", 1)[1].split("\n}", 1)[0]
    assert "cp -a /opt/ouroboros-ro/. /obo-repo/" in seed_branch
    assert "/obo-repo/.git/ouroboros_seed" in seed_branch
    # exactly one writer of the stamp, and it is the seeding branch
    assert entry.count("> /obo-repo/.git/ouroboros_seed") == 1


def test_e1v2_entrypoint_seed_verification_is_one_shot_outside_ready_probe():
    """Fail-closed provenance must be a ONE-SHOT step, not part of the polled readiness probe:
    the loop treats any non-zero probe rc as 'not ready yet', so a refusal inside ready_probe
    would be swallowed and burn the whole 900s window instead of stopping the task."""
    entry = (
        REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2" / "entrypoint_pro.sh"
    ).read_text(encoding="utf-8")
    probe_body = entry.split("ready_probe() {", 1)[1].split("\n}\n", 1)[0]
    assert "exit 88" not in probe_body
    assert "ouroboros_seed" not in probe_body
    # typed reasons the auto_run retry classifier keys on, and the named override
    for reason in ("stamp_absent", "seed_mismatch", "lineage_broken", "runtime_skew",
                   "runtime_unreachable", "seed_head_unreadable"):
        assert f'SEED_REASON="{reason}"' in entry
    assert "SOLVE_INFRA_SUSPECT reason=$SEED_REASON" in entry
    assert "merge-base --is-ancestor" in entry
    assert 'OBO_ALLOW_EVOLVED_VOLUME:-0}" = "1"' in entry
    assert "/out/seed_attestation.json" in entry
    # the solve is reached only after the refusal point
    assert entry.index("SOLVE_INFRA_SUSPECT reason=$SEED_REASON") < entry.index("/opt/problem_statement.txt")


def test_e1v2_entrypoint_computes_the_seed_head_once_and_never_stamps_a_sentinel():
    """The mounted seed's HEAD is read by ONE `rev-parse` reused by the stamp write and the
    verification (two independent reads could disagree), and an unreadable HEAD gets its own
    typed reason instead of a persisted `unknown` that later reads as seed_mismatch /
    lineage_broken."""
    entry = (
        REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2" / "entrypoint_pro.sh"
    ).read_text(encoding="utf-8")
    assert entry.count("git -C /opt/ouroboros-ro -c safe.directory='*' rev-parse HEAD") == 1
    assert "|| echo unknown" not in entry          # no sentinel HEAD anywhere
    seed_branch = entry.split("[ -e /obo-repo/.git ] || {", 1)[1].split("\n}", 1)[0]
    # the stamp write is guarded by a readable HEAD
    assert seed_branch.index('if [ -n "$MOUNTED_HEAD" ]; then') < seed_branch.index(
        "> /obo-repo/.git/ouroboros_seed"
    )
    assert 'SEED_REASON="seed_head_unreadable"' in entry
    assert entry.index('SEED_REASON="seed_head_unreadable"') < entry.index(
        "[ -e /obo-repo/.git ] || {"
    )
    from devtools.benchmarks.common.manifests import CAMPAIGN_FATAL_PROVENANCE_REASONS

    assert "seed_head_unreadable" in CAMPAIGN_FATAL_PROVENANCE_REASONS


def test_e1v2_assert_seed_is_git_directory_rejects_worktree_pointer(tmp_path):
    """The container seeds /obo-repo with `cp -a`, so a `git worktree add --detach` seed (whose
    .git is a POINTER FILE) leaves the agent with no git identity: no stamp, no merge-base
    lineage, no self-edit accounting. This long-implicit invariant is now explicit.

    It raises the typed `SeedShapeRefused` (a `RuntimeError`) rather than `SystemExit`: the
    launchers record this refusal in their manifest, and `SystemExit` derives from
    `BaseException`, so their handlers were inert and the run recorded a generic crash instead."""
    from devtools.benchmarks.common.manifests import SeedShapeRefused
    from devtools.benchmarks.swe_bench_pro.e1v2.run_pro import assert_seed_is_git_directory

    # The property that made the previous mechanism silently wrong.
    assert issubclass(SeedShapeRefused, Exception)

    real = tmp_path / "clone"
    (real / ".git").mkdir(parents=True)
    assert_seed_is_git_directory(real)  # no raise

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    with pytest.raises(SeedShapeRefused, match="git worktree pointer") as pointer:
        assert_seed_is_git_directory(worktree)
    assert pointer.value.reason == "seed_is_not_a_git_directory"

    with pytest.raises(SeedShapeRefused, match="missing") as absent:
        assert_seed_is_git_directory(tmp_path / "nothing")
    assert absent.value.reason == "seed_is_not_a_git_directory"


def test_e1v2_timeline_row_carries_seed_provenance():
    """A patch is only attributable if the row says WHICH agent identity produced it."""
    from devtools.benchmarks.swe_bench_pro.e1v2.run_pro import build_timeline_row

    res = {"model_patch": "", "timed_out": False, "infra_suspect": False, "health_rollback": False,
           "api_errors": 0, "api_ctx": 0, "refl_line": "", "quiet_line": "", "selfedit": {},
           "evolution_degraded": False, "absorb_reason": "",
           "seed_attestation": {"seed": "6.75.0 abc", "ok": True}}
    row = build_timeline_row(1, "inst", res, 0.0, [])
    assert row["seed_attestation"]["seed"] == "6.75.0 abc"


def test_e1v2_auto_run_hard_stops_on_a_volume_wide_provenance_refusal(tmp_path, monkeypatch):
    """Owner Q8 = hard stop. `stamp_absent` / `seed_mismatch` / `lineage_broken` /
    `runtime_skew` are properties of the VOLUME plus the mounted seed and are therefore
    identical for every task in the shard: recording them as per-task skips restores the same
    broken volume N times, burns the whole wall clock and still exits 0 with a zero headline.
    Per-task refusals (a missing env volume) must keep skipping, not stop the shard."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    args = _types.SimpleNamespace(
        total_budget=10.0, per_task_cost=5.0, task_wall_timeout=9000,
        volume_suffix="", full_set=False, csv="", settings="", solve_model="",
        model_name="", review_slots=None, review_effort="", solve_timeout=None,
        memory_mode="", baseline=False, allow_dirty_seed=False,
    )

    def _popen_writing(payload):
        class _P:
            def __init__(self, *a, **k):
                (tmp_path / "timeline.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
                self.pid = 4321

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        return _P

    for reason in sorted(auto_run.CAMPAIGN_FATAL_PROVENANCE_REASONS):
        monkeypatch.setattr(auto_run.subprocess, "Popen",
                            _popen_writing({"patch_bytes": 0, "api_errors": 0, "instance_id": "p",
                                            "infra_suspect": True, "infra_reason": reason}))
        with pytest.raises(SystemExit) as exc:
            auto_run.run_one(3, tmp_path, args, attempt=0)
        assert exc.value.code == 2

    # ... while a genuinely per-task refusal is still recorded and the shard continues.
    assert not (auto_run.CAMPAIGN_FATAL_PROVENANCE_REASONS & auto_run.PERMANENT_INFRA)
    for reason in ("libc_skip", "pip_bootstrap_failed"):
        assert reason in auto_run.PERMANENT_INFRA
        monkeypatch.setattr(auto_run.subprocess, "Popen",
                            _popen_writing({"patch_bytes": 0, "api_errors": 0, "instance_id": "q",
                                            "infra_suspect": True, "infra_reason": reason}))
        r = auto_run.run_one(3, tmp_path, args, attempt=0)
        assert r["permanent_skip"] is True and r["pb"] == 0


def test_e1v2_run_pro_creates_nothing_when_admission_refuses(tmp_path, monkeypatch):
    """A run the SEED GATE refuses must not have touched the SHARED docker daemon at all: the
    admission seam is the outer boundary, so a refused run issues no docker command whatsoever."""
    import types as _types
    from devtools.benchmarks.common import manifests as run_pro_manifests
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(run_pro, "assert_seed_is_git_directory", lambda path: None)
    monkeypatch.setattr(run_pro, "ensure_util_image", lambda: True)
    monkeypatch.setattr(run_pro, "read_full_order", lambda: ["inst__a"])
    monkeypatch.setattr(run_pro, "load_pro_rows", lambda ids: {})     # nothing schedulable
    docker_calls: list[list] = []
    monkeypatch.setattr(run_pro.subprocess, "run",
                        lambda *a, **k: docker_calls.append(list(a[0]) if a else [])
                        or _types.SimpleNamespace(returncode=0, stdout="", stderr=""))

    def _refuse(manifest_path, **kwargs):
        # The typed refusal `admit_benchmark_run` raises after persisting its payload. Forced
        # here rather than provoked from the ambient checkout, which may be clean.
        raise run_pro_manifests.BenchmarkAdmissionRefused(
            "seed provenance gate failed reason=seed_dirty",
            {"schema": "ouroboros.benchmark.run_manifest.v1", "extra": {}},
        )

    monkeypatch.setattr(run_pro, "admit_benchmark_run", _refuse)
    monkeypatch.setattr(sys, "argv", ["run_pro.py", "--full-set", "--out-dir", str(tmp_path / "out2")])
    with pytest.raises(RuntimeError, match="reason=seed_dirty"):
        run_pro.main()
    assert docker_calls == []                 # a refused run issues NO docker command at all


def test_e1v2_probe_reads_the_typed_grade_artefact_round_trip(tmp_path, monkeypatch):
    """Lock against the exact defect found in review: `grade_pro` no longer prints
    DIAGNOSTIC_PASS/FAIL/NO_OUTPUT, so the orchestrator's log string-matching returned ("ERR","")
    for EVERY task and reported a healthy run as a broken harness. The verdict now comes from
    `grade_summary.json`, and this test runs the REAL grader over a synthetic official-output dir
    and asserts the probe's extraction of it."""
    import devtools.benchmarks.swe_bench_pro.grade_pro as grade_pro
    from devtools.benchmarks.swe_bench_pro.e1v2.orchestrate_probe import read_graded_verdict

    eval_repo = tmp_path / "SWE-bench_Pro-os"
    (eval_repo / "helper_code").mkdir(parents=True)
    (eval_repo / "helper_code" / "sweap_eval_full_v2.jsonl").write_text(
        "\n".join(
            json.dumps({"instance_id": iid, "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": ["t2"]})
            for iid in ("won", "lost", "crashed")
        ) + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"instance_id": iid, "model_patch": "diff --git a/a b/a\n",
                        "model_name_or_path": "m"})
            for iid in ("won", "lost", "crashed")
        ) + "\n",
        encoding="utf-8",
    )
    ev = tmp_path / "pro_eval"
    for iid, tests in (
        ("won", [{"name": "t1", "status": "PASSED"}, {"name": "t2", "status": "PASSED"}]),
        ("lost", [{"name": "t1", "status": "FAILED"}, {"name": "t2", "status": "PASSED"}]),
    ):
        (ev / iid).mkdir(parents=True)
        (ev / iid / "ours_output.json").write_text(json.dumps({"tests": tests}), encoding="utf-8")

    monkeypatch.setattr(sys, "argv",
                        ["grade_pro.py", "--predictions", str(predictions), "--out-dir", str(ev),
                         "--eval-repo", str(eval_repo), "--skip-run"])
    assert grade_pro.main() == 0

    assert read_graded_verdict(ev, "won") == ("PASS", "2/0/2")
    assert read_graded_verdict(ev, "lost") == ("FAIL", "1/1/2")
    assert read_graded_verdict(ev, "crashed") == ("NO_OUTPUT", "-")
    # a missing artefact is an ERR, never a silent PASS/FAIL
    assert read_graded_verdict(tmp_path / "nowhere", "won") == ("ERR", "")


def test_e1v2_run_schedule_stays_inside_the_parameter_budget():
    """`_run_schedule` carries the whole schedule, and it must not do so with a signature the
    review checklist forbids (`docs/DEVELOPMENT.md`: no function has more than 8 parameters).
    `missing` was the redundant one: a pure function of `ids` and `rows`, so it is derived inside
    instead of passed. Asserted statically, and together with the call site, so a re-added
    argument cannot pass unnoticed."""
    import ast
    import inspect

    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    params = inspect.signature(run_pro._run_schedule).parameters
    assert len(params) <= 8, f"_run_schedule grew back to {len(params)} parameters: {list(params)}"
    assert "missing" not in params

    # `main()` must still call it with exactly that arity (a stale call site would be a TypeError
    # only on the real path, i.e. mid-campaign).
    tree = ast.parse(inspect.getsource(run_pro.main))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_run_schedule"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) + len(calls[0].keywords) == len(params)


def test_e1v2_run_pro_writes_a_durable_manifest_before_any_refusal(tmp_path, monkeypatch):
    """The manifest is persisted the MOMENT it is built, not once the post-admission checks have
    passed: a refusal after admission previously returned with no run_manifest.json on disk at
    all, so a refused shard left nothing but a stderr line the launcher discards. Asserted
    MID-FLIGHT (inside the refusing check) as well as from the retained record afterwards."""
    import types as _types
    from devtools.benchmarks.common.manifests import SeedShapeRefused
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    out_dir = tmp_path / "out"

    def _refuse_seed_shape(src):
        # The durable record has to exist ALREADY, before this can refuse.
        assert (out_dir / "run_manifest.json").is_file()
        raise SeedShapeRefused("seed_is_not_a_git_directory",
                               "error: the mounted seed has no real .git directory")

    monkeypatch.setattr(run_pro, "assert_seed_is_git_directory", _refuse_seed_shape)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(run_pro, "ensure_util_image", lambda: True)
    monkeypatch.setattr(run_pro, "read_full_order", lambda: ["inst__a"])
    monkeypatch.setattr(run_pro, "load_pro_rows", lambda ids: {})
    monkeypatch.setattr(run_pro.subprocess, "run",
                        lambda *a, **k: _types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    # The REAL admission seam runs here — that is the claim under test. `--allow-dirty-seed`
    # keeps it independent of the ambient checkout without stubbing the write away.
    monkeypatch.setattr(sys, "argv", ["run_pro.py", "--full-set", "--out-dir", str(out_dir),
                                      "--allow-dirty-seed"])

    assert run_pro.main() == 2
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["outcome"] == "refused"
    assert manifest["extra"]["refusal"] == {
        "stage": "seed_shape", "exit_code": 2, "reason": "seed_is_not_a_git_directory"}


def test_e1v2_auto_run_gates_before_docker_and_records_the_final_outcome(tmp_path, monkeypatch):
    """Two halves of the same lifecycle claim for `auto_run`: (a) nothing that touches the SHARED
    docker daemon (util-image pull, volume reads, baseline snapshot) may run before the manifest
    is built and persisted, and (b) the retained manifest is rewritten with the final outcome —
    here a baseline-snapshot refusal, which returned 3 with no record of what was refused."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    out_dir = tmp_path / "auto"
    order: list[str] = []
    manifest_seen_by: dict[str, bool] = {}

    def _mark(name, value=None):
        order.append(name)
        manifest_seen_by[name] = (out_dir / "auto_run_manifest.json").is_file()
        return value

    monkeypatch.setattr(auto_run, "reflections", lambda vsuf: _mark("reflections", 0))
    monkeypatch.setattr(auto_run, "seed_stamp", lambda vsuf: _mark("seed_stamp", "6.75.0 abc"))
    monkeypatch.setattr(auto_run, "snapshot", lambda lg, vsuf: _mark("snapshot", False))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    fake_run_pro = _types.ModuleType("fake_run_pro")
    # A pure local check with no cost and no side effect; it legitimately precedes the manifest.
    fake_run_pro.assert_seed_is_git_directory = lambda path: _mark("assert_seed", None)
    fake_run_pro.ensure_util_image = lambda: _mark("ensure_util_image", None)
    monkeypatch.setitem(sys.modules, "devtools.benchmarks.swe_bench_pro.e1v2.run_pro", fake_run_pro)
    _real_admit_without_clean_seed(auto_run, monkeypatch,
                                   on_call=lambda _kwargs: _mark("manifest"))
    monkeypatch.setattr(sys, "argv", ["auto_run.py", "--start", "1", "--end", "2",
                                      "--out-dir", str(out_dir)])

    assert auto_run.main() == 3
    # `assert_seed` moved AFTER `manifest`: it is a FILESYSTEM assertion, so refusing on it before
    # admission left that refusal with no durable record (the earlier order asserted the opposite
    # and was wrong). Everything with a cost or a side effect still follows the manifest.
    assert order == ["manifest", "assert_seed", "ensure_util_image", "reflections", "seed_stamp",
                     "snapshot"]
    assert manifest_seen_by["assert_seed"] is True           # durable before the seed-shape check
    assert manifest_seen_by["ensure_util_image"] is True     # durable before the first docker call
    manifest = json.loads((out_dir / "auto_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["outcome"] == "refused"
    assert manifest["extra"]["exit_code"] == 3
    assert manifest["extra"]["refusal"]["reason"] == "lastgood_snapshot_incomplete"
    assert manifest["extra"]["volume_seed_stamp"] == "6.75.0 abc"
    assert manifest["extra"]["task_count"] == 0


def _entrypoint_shell_block(start: str) -> str:
    """Extract ONE top-level `if ... fi` block from the entrypoint so it can be RUN, not matched.

    Static string assertions cannot tell `waives only runtime_skew` from `waives anything`; this
    lets the tests below execute the real decision logic under bash with controlled inputs.
    """
    entry = (
        REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2" / "entrypoint_pro.sh"
    ).read_text(encoding="utf-8")
    assert entry.count(start) == 1, start
    begin = entry.index(start)
    end = entry.index("\nfi\n", begin) + len("\nfi\n")
    return entry[begin:end]


def _run_shell(block: str, env: dict, tail: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", block + tail],
        capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", ""), **env},
    )


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash is required to execute the entrypoint block")
def test_e1v2_entrypoint_classifies_an_unreachable_runtime_separately_from_a_skew():
    """An EMPTY runtime_version is the absence of a live identity, not a version skew.

    The health heredoc prints "" for any transport failure, non-200 or non-contract body. Folding
    that into `runtime_skew` made it OVERRIDABLE, so with `OBO_ALLOW_EVOLVED_VOLUME=1` exported the
    container solved on against a server it had never identified. Executed, not string-matched."""
    block = _entrypoint_shell_block('if [ -z "$SEED_REASON" ]; then\n  if [ -z "$RUNTIME_VERSION" ]; then')
    tail = '\nprintf "%s" "$SEED_REASON"'

    def _classify(runtime_version: str, live_version: str, seed_reason: str = "") -> str:
        return _run_shell(block, {"RUNTIME_VERSION": runtime_version,
                                  "LIVE_VERSION": live_version,
                                  "SEED_REASON": seed_reason}, tail).stdout

    assert _classify("", "6.75.0") == "runtime_unreachable"
    assert _classify("6.74.5", "6.75.0") == "runtime_skew"
    assert _classify("6.75.0", "6.75.0") == ""
    # An earlier, more specific reason is never overwritten by this step.
    assert _classify("", "6.75.0", seed_reason="stamp_absent") == "stamp_absent"


@pytest.mark.skipif(not _BASH_AVAILABLE, reason="bash is required to execute the entrypoint block")
def test_e1v2_entrypoint_override_waives_only_a_genuine_version_skew():
    """`OBO_ALLOW_EVOLVED_VOLUME=1` used to waive ANY non-empty `SEED_REASON` in the shell, so a
    volume with no stamp, a foreign seed, a broken lineage, an unreadable seed HEAD or an
    unidentified runtime all solved on with the override exported — the exact fail-open the
    Python helper had already been narrowed against. The shell allowlist is a single reason and it
    is the SAME one, asserted against `manifests.OVERRIDABLE_ATTESTATION_REASONS`."""
    from devtools.benchmarks.common.manifests import OVERRIDABLE_ATTESTATION_REASONS

    entry = (
        REPO_ROOT / "devtools" / "benchmarks" / "swe_bench_pro" / "e1v2" / "entrypoint_pro.sh"
    ).read_text(encoding="utf-8")
    waives = ",".join(OVERRIDABLE_ATTESTATION_REASONS)
    assert f'OBO_OVERRIDABLE_SEED_REASON="{waives}"' in entry

    block = _entrypoint_shell_block(
        'if [ -n "$SEED_REASON" ]; then\n  if [ "${OBO_ALLOW_EVOLVED_VOLUME:-0}" = "1" ]'
    )

    def _decide(seed_reason: str, override: str) -> subprocess.CompletedProcess:
        return _run_shell(block, {"SEED_REASON": seed_reason,
                                  "OBO_ALLOW_EVOLVED_VOLUME": override,
                                  "OBO_OVERRIDABLE_SEED_REASON": waives}, "\nexit 0")

    admitted = _decide("runtime_skew", "1")
    assert admitted.returncode == 0
    assert "OVERRIDDEN" in admitted.stderr

    for reason in ("stamp_absent", "seed_mismatch", "lineage_broken", "seed_head_unreadable",
                   "runtime_unreachable"):
        refused = _decide(reason, "1")
        assert refused.returncode == 88, f"{reason} was waived by the override"
        assert f"SOLVE_INFRA_SUSPECT reason={reason}" in refused.stderr
        assert "does NOT waive" in refused.stderr

    # Without the override even the waivable reason refuses, and a clean run is untouched.
    assert _decide("runtime_skew", "0").returncode == 88
    assert _decide("", "1").returncode == 0

    # The recorded attestation distinguishes "was the override exported" from "did it waive THIS
    # reason" — conflating them made an audit of a refused run read as if it had applied.
    assert '"override_set": override_set,' in entry
    assert '"overridden": bool(reason) and override_set and reason in waives,' in entry


def test_e1v2_probe_never_publishes_a_stale_grade(tmp_path, monkeypatch):
    """`grade_one` neither removed a previous `pro_eval/grade_summary.json` nor checked the
    grader's return code, so a rerun whose grader timed out or crashed silently attributed the
    PREVIOUS attempt's PASS/FAIL to the new one — a wrong headline with no error anywhere. It now
    unlinks the artefact first and demands BOTH a zero exit and a freshly written summary."""
    from devtools.benchmarks.swe_bench_pro.e1v2 import orchestrate_probe

    tdir = tmp_path / "w0" / "t01"
    summary = tdir / "pro_eval" / "grade_summary.json"
    summary.parent.mkdir(parents=True)
    stale = {"verdicts": [{"instance_id": "inst-a", "verdict": "pass", "tests": "2/0/2"}]}
    summary.write_text(json.dumps(stale), encoding="utf-8")
    pred = tdir / "predictions.jsonl"
    pred.write_text(json.dumps({"instance_id": "inst-a"}) + "\n", encoding="utf-8")

    def _grade():
        return orchestrate_probe.grade_one(
            tdir, pred, "inst-a", host_python=sys.executable,
            eval_repo=str(tmp_path / "eval-repo"), run_csv=tmp_path / "run.csv")

    # (a) the grader TIMES OUT over a directory that still holds a PASS -> ERR, never PASS.
    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 2400)

    monkeypatch.setattr(orchestrate_probe.subprocess, "run", _timeout)
    assert _grade() == ("ERR", "")
    assert not summary.exists()          # removed, not merely ignored
    assert "GRADE_TIMEOUT" in (tdir / "grade.log").read_text(encoding="utf-8")

    # (b) the grader FAILS but leaves a summary behind (partial write) -> still ERR.
    def _fail(cmd, **kwargs):
        summary.write_text(json.dumps(stale), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="eval repo missing")

    monkeypatch.setattr(orchestrate_probe.subprocess, "run", _fail)
    assert _grade() == ("ERR", "")

    # (c) the grader SUCCEEDS and writes a fresh summary -> the verdict is read normally.
    def _ok(cmd, **kwargs):
        summary.write_text(json.dumps(stale), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(orchestrate_probe.subprocess, "run", _ok)
    assert _grade() == ("PASS", "2/0/2")

    # (d) the grader exits 0 but writes NOTHING -> ERR, never a verdict out of thin air.
    monkeypatch.setattr(orchestrate_probe.subprocess, "run",
                        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    assert _grade() == ("ERR", "")


def test_e1v2_run_pro_records_a_crash_instead_of_leaving_started(tmp_path, monkeypatch):
    """`run_pro`'s manifest was rewritten only on the paths it anticipated, so any unhandled
    failure after admission left the run's own record claiming `started` forever."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    out_dir = tmp_path / "out"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(run_pro, "assert_seed_is_git_directory", lambda path: None)
    monkeypatch.setattr(run_pro, "read_full_order", lambda: ["inst__a"])
    monkeypatch.setattr(run_pro, "load_pro_rows", lambda ids: {})
    monkeypatch.setattr(run_pro.subprocess, "run",
                        lambda *a, **k: _types.SimpleNamespace(returncode=0, stdout="", stderr=""))

    def _boom():
        raise RuntimeError("utility image is unavailable")

    monkeypatch.setattr(run_pro, "ensure_util_image", _boom)
    monkeypatch.setattr(sys, "argv", ["run_pro.py", "--full-set", "--out-dir", str(out_dir),
                                      "--allow-dirty-seed"])
    with pytest.raises(RuntimeError, match="utility image is unavailable"):
        run_pro.main()

    extra = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "crashed"
    assert extra["exit_code"] == 1
    assert extra["error"]["type"] == "RuntimeError"


def test_e1v2_auto_run_records_a_crash_instead_of_leaving_started(tmp_path, monkeypatch):
    """Same claim for the campaign driver: `_finish` covered every INTENDED exit, so a docker
    transient between admission and the baseline snapshot left `outcome: started`."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    out_dir = tmp_path / "auto"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(auto_run, "reflections", lambda vsuf: 0)

    def _boom(vsuf):
        raise RuntimeError("docker volume inspect failed")

    monkeypatch.setattr(auto_run, "seed_stamp", _boom)
    fake_run_pro = _types.ModuleType("fake_run_pro")
    fake_run_pro.assert_seed_is_git_directory = lambda path: None
    fake_run_pro.ensure_util_image = lambda: None
    monkeypatch.setitem(sys.modules, "devtools.benchmarks.swe_bench_pro.e1v2.run_pro", fake_run_pro)
    monkeypatch.setattr(sys, "argv", ["auto_run.py", "--start", "1", "--end", "1",
                                      "--out-dir", str(out_dir), "--allow-dirty-seed"])
    with pytest.raises(RuntimeError, match="docker volume inspect failed"):
        auto_run.main()

    extra = json.loads((out_dir / "auto_run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "crashed"
    assert extra["exit_code"] == 1
    assert extra["error"]["type"] == "RuntimeError"


def test_e1v2_auto_run_records_a_campaign_fatal_refusal_with_its_real_exit_status(tmp_path, monkeypatch):
    """The manifest half of the campaign-fatal stop, through `main()`.

    `run_one` raises `SystemExit(2)` on a volume-wide provenance refusal. The record must say
    `refused` with exit code **2** — the generic `crashed`/`exit_code: 1` the seam produced for any
    escaping exception corrupted exactly the record this phase exists to make authoritative."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    out_dir = tmp_path / "auto"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(auto_run, "reflections", lambda vsuf: 0)
    monkeypatch.setattr(auto_run, "seed_stamp", lambda vsuf: "6.75.0 abc")
    monkeypatch.setattr(auto_run, "snapshot", lambda lg, vsuf: True)

    def _campaign_fatal(i, out, args, attempt=0):
        raise SystemExit(2)

    monkeypatch.setattr(auto_run, "run_one", _campaign_fatal)
    fake_run_pro = _types.ModuleType("fake_run_pro")
    fake_run_pro.assert_seed_is_git_directory = lambda path: None
    fake_run_pro.ensure_util_image = lambda: None
    monkeypatch.setitem(sys.modules, "devtools.benchmarks.swe_bench_pro.e1v2.run_pro", fake_run_pro)
    monkeypatch.setattr(sys, "argv", ["auto_run.py", "--start", "1", "--end", "2",
                                      "--out-dir", str(out_dir), "--allow-dirty-seed"])

    with pytest.raises(SystemExit) as exc:
        auto_run.main()
    assert exc.value.code == 2                     # process behaviour unchanged

    extra = json.loads((out_dir / "auto_run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "refused"           # a deliberate refusal, not a crash
    assert extra["exit_code"] == 2                 # ... with the status the process really uses
    assert extra["refusal"]["stage"] == "campaign_fatal_infra"
    assert extra["stopped_at"] == 1
    assert extra["error"]["type"] == "SystemExit"


def test_e1v2_run_pro_stops_the_schedule_on_a_volume_wide_provenance_refusal(tmp_path, monkeypatch):
    """A DIRECT `run_pro` invocation must honour the campaign-fatal contract too.

    `entrypoint_pro.sh` emits volume-wide provenance reasons, but only `auto_run.run_one` acted on
    them — so `run_pro` invoked directly (and `orchestrate_probe`, which shells out to it) appended
    one INFRA timeline row per task, refused the WHOLE schedule and still returned 0 with
    `outcome: completed`: a zero headline reported as success. Both drivers now consume ONE shared
    authority, `manifests.CAMPAIGN_FATAL_PROVENANCE_REASONS`, and the refused task's row is still
    written first because `auto_run.run_one` reads it."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    out_dir = tmp_path / "out"
    solved: list[str] = []
    row = {"dockerhub_tag": "t", "base_commit": "b", "repo": "r/r", "repo_language": "python",
           "problem_statement": "p", "requirements": "", "interface": ""}

    def fake_run_instance(cid, row_, args, api_key, seed, task_total):
        solved.append(cid)
        return {"instance_id": cid, "model_name_or_path": "m", "model_patch": "",
                "timed_out": False, "infra_suspect": True, "infra_reason": "seed_mismatch",
                "health_rollback": False, "refl_line": "", "solve_line": "", "quiet_line": ""}

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(run_pro, "assert_seed_is_git_directory", lambda path: None)
    monkeypatch.setattr(run_pro, "ensure_util_image", lambda: True)
    monkeypatch.setattr(run_pro, "read_full_order", lambda: ["inst__a", "inst__b", "inst__c"])
    monkeypatch.setattr(run_pro, "load_pro_rows", lambda ids: {i: dict(row) for i in ids})
    monkeypatch.setattr(run_pro, "docker_pull_if_missing", lambda img: True)
    monkeypatch.setattr(run_pro, "read_spent_usd", lambda vol: 0.0)
    monkeypatch.setattr(run_pro, "derive_run_settings", lambda *a, **k: tmp_path / "seed.json")
    dumped: list = []
    monkeypatch.setattr(run_pro, "dump_state", lambda *a, **k: dumped.append(a))
    monkeypatch.setattr(run_pro, "run_instance", fake_run_instance)
    monkeypatch.setattr(run_pro.subprocess, "run",
                        lambda *a, **k: _types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(sys, "argv", ["run_pro.py", "--full-set", "--out-dir", str(out_dir),
                                      "--allow-dirty-seed"])

    assert run_pro.main() == 2                      # nonzero: the schedule was refused
    assert solved == ["inst__a"]                    # the remaining tasks were never attempted

    extra = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "refused"
    assert extra["exit_code"] == 2
    assert extra["refusal"] == {"stage": "volume_provenance", "exit_code": 2,
                                "reason": "seed_mismatch", "task_index": 1,
                                "instance_id": "inst__a"}
    # The stop precedes the volume archival: `dump_state` can hang for hours on a loaded daemon
    # (its own comment says so), which would strand the refusal and the exit code instead of
    # recording them. The timeline row auto_run parses is still written first.
    assert dumped == []
    # The refused task itself IS recorded, with its exact reason — parity with what auto_run reads.
    timeline = [json.loads(line) for line
                in (out_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["instance_id"] for r in timeline] == ["inst__a"]
    assert timeline[0]["infra_reason"] == "seed_mismatch"


def test_e1v2_run_pro_records_the_typed_seed_shape_refusal(tmp_path, monkeypatch):
    """The RECORD, not just the raise. `assert_seed_is_git_directory` used to raise `SystemExit`
    (a `BaseException`), so the launcher's `except Exception` handler never ran and the manifest
    said `crashed` instead of the promised typed refusal — an inert fix that the earlier test
    could not catch because it asserted only that something was raised."""
    import types as _types
    from devtools.benchmarks.swe_bench_pro.e1v2 import run_pro

    out_dir = tmp_path / "out"
    seed = tmp_path / "worktree-seed"
    seed.mkdir()
    (seed / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    monkeypatch.setattr(run_pro, "SRC", seed)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(run_pro, "ensure_util_image",
                        lambda: pytest.fail("docker touched after a seed-shape refusal"))
    monkeypatch.setattr(run_pro, "read_full_order", lambda: ["inst__a"])
    monkeypatch.setattr(run_pro, "load_pro_rows", lambda ids: {})
    monkeypatch.setattr(run_pro.subprocess, "run",
                        lambda *a, **k: _types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(sys, "argv", ["run_pro.py", "--full-set", "--out-dir", str(out_dir),
                                      "--allow-dirty-seed"])

    # A nonzero RETURN, so the recorded exit_code equals the status the process really exits with.
    assert run_pro.main() == 2
    extra = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "refused"
    assert extra["exit_code"] == 2
    assert extra["refusal"] == {"stage": "seed_shape", "exit_code": 2,
                                "reason": "seed_is_not_a_git_directory"}


def test_e1v2_auto_run_records_the_typed_seed_shape_refusal(tmp_path, monkeypatch):
    """Same record on the campaign driver, whose handler was inert for the same reason."""
    import types as _types
    from devtools.benchmarks.common.manifests import SeedShapeRefused
    from devtools.benchmarks.swe_bench_pro.e1v2 import auto_run

    out_dir = tmp_path / "auto"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    fake_run_pro = _types.ModuleType("fake_run_pro")

    def _refuse(path):
        raise SeedShapeRefused("seed_is_not_a_git_directory",
                               "error: the mounted seed has no real .git directory")

    fake_run_pro.assert_seed_is_git_directory = _refuse
    fake_run_pro.ensure_util_image = lambda: pytest.fail("docker touched after the refusal")
    monkeypatch.setitem(sys.modules, "devtools.benchmarks.swe_bench_pro.e1v2.run_pro", fake_run_pro)
    monkeypatch.setattr(auto_run, "snapshot", lambda lg, vsuf: pytest.fail("snapshot after refusal"))
    monkeypatch.setattr(sys, "argv", ["auto_run.py", "--start", "1", "--end", "1",
                                      "--out-dir", str(out_dir), "--allow-dirty-seed"])

    assert auto_run.main() == 2
    extra = json.loads((out_dir / "auto_run_manifest.json").read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "refused"
    assert extra["exit_code"] == 2
    assert extra["refusal"] == {"stage": "seed_shape", "exit_code": 2,
                                "reason": "seed_is_not_a_git_directory"}
    assert extra["task_count"] == 0
