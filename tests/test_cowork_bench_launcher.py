"""Campaign continuity, resume and custody checks without a provider or Docker."""

from __future__ import annotations

import argparse
import json
import io
import os
import pathlib
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import pytest

from devtools.benchmarks.common.launcher_audit import audit_launcher, launcher_paths
from devtools.benchmarks.common.model_slots import runtime_actor_snapshot
from devtools.benchmarks.cowork_bench import campaign as budgets
from devtools.benchmarks.cowork_bench import eval_attempt as attempts
from devtools.benchmarks.cowork_bench import run_cowork_bench as launcher

IMAGE_ID = "sha256:" + "1" * 64


@pytest.fixture(autouse=True)
def forbid_live_services(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("offline launcher test tried to contact a provider or Docker")
    monkeypatch.setattr(budgets.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(launcher, "_docker", forbidden)


def write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_campaign_restarts_keep_one_baseline_and_prior_spend(tmp_path):
    path = tmp_path / "campaign.json"
    first = budgets.CampaignBudget(path, fingerprint="key-a", ceiling=1000, usage=500, prior_spend=30)
    first.start(tmp_path / "smoke")
    first.observe(570)
    first.finish(tmp_path / "smoke", outcome="done")
    second = budgets.CampaignBudget(path, fingerprint="key-a", ceiling=1000, usage=580)
    assert second.spent == 110
    assert second.remaining == 890
    second.start(tmp_path / "recovery")
    second.observe(640)
    second.finish(tmp_path / "recovery", outcome="done")
    third = budgets.CampaignBudget(path, fingerprint="key-a", ceiling=1000, usage=650)
    assert third.spent == 180
    assert third.remaining == 820
    assert [row["run_root"] for row in third.record["runs"]] == [str(tmp_path / "smoke"), str(tmp_path / "recovery")]
    assert third.record["usage_baseline"] == 500


@pytest.mark.parametrize("changed", [{"fingerprint": "different"}, {"ceiling": 2000}, {"prior_spend": 10}])
def test_existing_campaign_cannot_silently_reset_owner_budget(tmp_path, changed):
    path = tmp_path / "campaign.json"
    budgets.CampaignBudget(path, fingerprint="key-a", ceiling=1000, usage=100)
    original = path.read_bytes()
    kwargs = {"fingerprint": "key-a", "ceiling": 1000, "usage": 110, **changed}
    with pytest.raises(ValueError):
        budgets.CampaignBudget(path, **kwargs)
    assert path.read_bytes() == original


def test_unsettled_run_and_counter_regression_refuse_a_new_paid_run(tmp_path):
    path = tmp_path / "campaign.json"
    budget = budgets.CampaignBudget(path, fingerprint="key-a", ceiling=1000, usage=100)
    budget.start(tmp_path / "active")
    with pytest.raises(ValueError, match="unsettled custody"):
        budgets.CampaignBudget(path, fingerprint="key-a", ceiling=1000, usage=120)
    budget.finish(tmp_path / "active", outcome="stopped")
    with pytest.raises(ValueError, match="decreased"):
        budgets.CampaignBudget(path, fingerprint="key-a", ceiling=1000, usage=99)
    assert json.loads(path.read_text(encoding="utf-8"))["last_usage"] == 100


def test_campaign_lock_excludes_another_launcher_and_releases_on_exception(tmp_path):
    path = tmp_path / "campaign.json"
    with pytest.raises(LookupError):
        with budgets.campaign_lock(path):
            with pytest.raises(RuntimeError, match="another launcher"):
                with budgets.campaign_lock(path):
                    pytest.fail("second launcher acquired the active campaign")
            raise LookupError("caller failed")
    with budgets.campaign_lock(path):
        assert path.with_suffix(".json.lock").exists()
    assert not path.with_suffix(".json.lock").exists()


@pytest.fixture
def selection(tmp_path):
    bench = tmp_path / "bench"
    for task in ("passed", "wrong-answer", "infra", "new"):
        write_json(bench / "tasks" / "finalpool" / task / "task_config.json", {})
    previous = tmp_path / "previous"
    config = {"model": "test-model", "settings": {"effort": "high"}}
    write_json(previous / "run_manifest.json", {
        "harness": {"applied_config": config, "bench": {"head": "bench-sha"}, "image_id": IMAGE_ID},
        "source": {"head": "seed-sha"},
        "requested_task_ids": ["passed", "wrong-answer", "infra"],
    })
    rows = [{"instance_id": name, "status": status} for name, status in (
        ("passed", "passed"), ("wrong-answer", "failed"), ("infra", "infra_failed"))]
    (previous / "result_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    args = argparse.Namespace(task_file="", task=[], resume_from=[str(previous)],
                              bench_commit="bench-sha", image_id=IMAGE_ID)
    return bench, previous, args, config


def test_resume_retries_infrastructure_but_never_genuine_failures(selection):
    bench, _previous, args, config = selection
    # The original invocation selected three tasks, not the whole dataset.
    assert launcher.select_tasks(bench, args, config, "seed-sha") == ["infra"]
    assert set(args.selection_task_ids) == {"passed", "wrong-answer", "infra"}
    assert args.resume_generation == 1
    args.task = ["passed", "wrong-answer"]
    assert launcher.select_tasks(bench, args, config, "seed-sha") == []


@pytest.mark.parametrize("change", ["config", "seed", "benchmark", "image"])
def test_resume_refuses_incompatible_protocol_or_source(selection, change):
    bench, _previous, args, config = selection
    seed = "seed-sha"
    if change == "config":
        config = {**config, "model": "another-model"}
    elif change == "seed":
        seed = "new-seed"
    elif change == "benchmark":
        args.bench_commit = "new-benchmark"
    else:
        args.image_id = "sha256:" + "2" * 64
    with pytest.raises(ValueError, match="configuration|seed|image"):
        launcher.select_tasks(bench, args, config, seed)


@pytest.mark.parametrize("summary,evaluation,runner,expected", [
    ({"bench_status": "success"}, {"pass": True}, {"status": "success"}, "passed"),
    ({"bench_status": "success"}, {"pass": False}, {"status": "success"}, "failed"),
    ({"bench_status": "failed", "reason_code": "timeout"}, {}, {"status": "failed"}, "agent_failed"),
    ({"bench_status": "failed", "infra_failed": True, "reason_code": "llm_api_error"}, {}, {"status": "failed"}, "infra_failed"),
    ({"bench_status": "success"}, {}, {"status": "success"}, "infra_failed"),
    ({}, {}, {"status": "pg_fail"}, "infra_failed"),
    ({}, {}, {}, "not_attempted"),
])
def test_ledger_separates_real_failures_from_recoverable_infrastructure(tmp_path, summary, evaluation, runner, expected):
    if summary:
        write_json(tmp_path / "ouroboros_summary.json", summary)
    if evaluation:
        write_json(tmp_path / "eval_res.json", evaluation)
    row = launcher.ledger_row("task", tmp_path, runner, protocol="legacy", cause="runner_exited")
    assert row["status"] == expected
    assert row["instance_id"] == "task"


@pytest.fixture
def dry_launcher(tmp_path, monkeypatch):
    bench, out, repo = tmp_path / "source-bench", tmp_path / "run", tmp_path / "seed"
    write_json(bench / "tasks" / "finalpool" / "one" / "task_config.json", {})
    (bench / "configs").mkdir()
    repo.mkdir()
    def admit(path, **kwargs):
        path.parent.mkdir(parents=True)
        return {"source": {"head": "seed-sha"}, "harness": kwargs["harness"], "extra": kwargs["extra"]}
    def clone(command, **_kwargs):
        assert command[:4] == ["git", "clone", "--quiet", "--no-hardlinks"]
        shutil.copytree(command[-2], command[-1])
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(launcher, "admit_benchmark_run", admit)
    monkeypatch.setattr(launcher, "bench_provenance", lambda *_a: {"head": launcher.PINNED_BENCH_COMMIT})
    monkeypatch.setattr(launcher, "image_exists", lambda *_a: True)
    monkeypatch.setattr(launcher, "image_identity", lambda *_a: {
        "id": IMAGE_ID, "repo_digests": [], "labels": {
            "org.ouroboros.cowork.seed_sha": "seed-sha",
            "org.ouroboros.cowork.bench_sha": launcher.PINNED_BENCH_COMMIT,
        },
    })
    monkeypatch.setattr(launcher, "_git", lambda *_a: "")
    monkeypatch.setattr(launcher.subprocess, "run", clone)
    argv = ["--bench-root", str(bench), "--repo-dir", str(repo), "--run-root", str(out),
            "--docker-host", "unix:///fake.sock", "--min-free-gib", "0", "--dry-run"]
    return out, argv


def test_empty_resume_never_calls_official_runner_with_zero_task_arguments(dry_launcher, monkeypatch):
    out, argv = dry_launcher
    args = launcher.parse_args(argv)
    template = json.loads(launcher.SETTINGS_TEMPLATE.read_text(encoding="utf-8"))
    settings, _actor = launcher.render_settings(template, args)
    config = launcher.bench_config(args, settings)
    previous = out.parent / "previous-run"
    write_json(previous / "run_manifest.json", {
        "source": {"head": "seed-sha"}, "requested_task_ids": ["one"],
        "harness": {"applied_config": config, "image_id": IMAGE_ID,
                    "bench": {"head": launcher.PINNED_BENCH_COMMIT}},
    })
    (previous / "result_index.jsonl").write_text(
        json.dumps({"instance_id": "one", "status": "passed"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "spawn_supervised", lambda *_a, **_k: pytest.fail("empty remainder launched all tasks"))
    assert launcher.main([*argv, "--resume-from", str(previous)]) == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["outcome"] == "nothing_remaining"
    assert manifest["requested_count"] == 0
    assert (out / "result_index.jsonl").read_text(encoding="utf-8") == ""


def test_manifest_metadata_matches_config_received_by_container(dry_launcher):
    out, argv = dry_launcher
    assert launcher.main([*argv, "--effort", "xhigh", "--model", "provider/test-model"]) == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((out / "bench" / "configs" / launcher.CONFIG_NAME).read_text(encoding="utf-8"))
    assert manifest["harness"]["applied_config"] == config
    assert manifest["harness"]["image_id"] == IMAGE_ID
    assert manifest["harness"]["selection_task_ids"] == ["one"]
    assert manifest["harness"]["resume_generation"] == 0
    observed = runtime_actor_snapshot(config["settings"], expected_model=config["model"])
    assert not observed["mismatches"]
    assert manifest["model_slots"] == observed["model_slots"]
    assert manifest["available_subagents"] == observed["available_subagents"]
    assert manifest["harness"]["fixed_model_actor"] == observed
    assert config["settings"]["OUROBOROS_EFFORT_TASK"] == "xhigh"
    assert not config["settings"].get("OUROBOROS_OR_PROVIDER")


def test_meter_blindness_bound_is_validated_and_recorded_in_the_manifest(dry_launcher):
    out, argv = dry_launcher
    assert launcher.parse_args([]).meter_blindness_sec == 30.0
    for invalid in ("20", "-1", "nan", "inf"):  # Must leave room for one 15 s poll plus one read.
        with pytest.raises(SystemExit):
            launcher.parse_args(["--meter-blindness-sec", invalid])
    assert launcher.main([*argv, "--meter-blindness-sec", "45"]) == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["harness"]["meter"] == {"blindness_sec": 45.0, "poll_sec": 15.0,
                                            "retry_sec": 3.0, "read_slice_sec": 5.0}


@pytest.fixture
def supervised(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    bench.mkdir()
    args = launcher.parse_args([])
    args.resource_root = tmp_path
    args.docker_host = "unix:///owned.sock"
    args.selected_tasks = ["unstarted"]
    args.min_free_gib = args.min_root_free_gib = 0
    env = {"COWORK_STOP_FILE": str(tmp_path / "resource_stop"), "COWORK_RUN_LABEL": "owned-run"}
    budget = budgets.CampaignBudget(tmp_path / "campaign.json", fingerprint="key-a", ceiling=1000, usage=100)
    events = []
    proc = SimpleNamespace(pid=424242, returncode=None)
    proc.poll = lambda: proc.returncode
    proc.wait = lambda **_kwargs: proc.returncode
    handlers = {}
    def register(sig, handler):
        old = handlers.get(sig, "old-handler")
        handlers[sig] = handler
        return old
    def stop(owned):
        assert owned is proc
        events.append("stop-group")
        owned.returncode = -15
    def cleanup(host, label):
        assert (host, label) == (args.docker_host, env["COWORK_RUN_LABEL"])
        events.append("cleanup-owned")
    monkeypatch.setattr(launcher.signal, "signal", register)
    def launch(_command, **kwargs):
        assert kwargs["drive_root"] == bench.parent
        assert kwargs["purpose"] == "cowork-official-runner"
        assert kwargs["scope"] == "session"
        assert kwargs.get("new_process_group", True) is True
        return proc
    monkeypatch.setattr(launcher, "spawn_supervised", launch)
    monkeypatch.setattr(launcher, "stop_process_group", stop)
    monkeypatch.setattr(launcher, "remove_run_containers", cleanup)
    monkeypatch.setattr(launcher, "key_usage", lambda _key, **_kwargs: 100)
    # Retries wait on this simulated clock, so a meter outage reaches its bound instantly.
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(launcher, "time", SimpleNamespace(
        time=lambda: clock.now, monotonic=lambda: clock.now,
        sleep=lambda delay: setattr(clock, "now", clock.now + delay)))
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1024**4))
    return args, bench, env, budget, events, handlers, proc


@pytest.mark.parametrize("trigger,expected", [
    ("meter-loss", "budget_meter_unavailable"), ("disk", "disk_reserve"),
    ("campaign", "campaign_budget_reserve"), ("run", "run_budget"),
])
def test_supervisor_stops_owned_work_on_real_boundaries(supervised, monkeypatch, trigger, expected):
    args, bench, env, budget, events, handlers, _proc = supervised
    if trigger == "meter-loss":
        def offline(_key, **_kwargs):
            raise OSError("meter unavailable")
        monkeypatch.setattr(launcher, "key_usage", offline)
    elif trigger == "disk":
        args.min_free_gib = 200
        monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _path: SimpleNamespace(free=199 * 1024**3))
    else:
        monkeypatch.setattr(launcher, "key_usage", lambda _key, **_kwargs: 1000 if trigger == "campaign" else 250)
    result = launcher.supervise_run(args, ["fake-runner"], bench, env, "not-a-real-key", budget)
    assert result["stop_reason"] == expected
    assert events == ["stop-group", "cleanup-owned"]
    assert pathlib.Path(env["COWORK_STOP_FILE"]).read_text(encoding="utf-8").strip() == expected
    assert "active_run" not in budget.record
    assert budget.record["runs"][-1]["outcome"] == expected
    assert all(value == "old-handler" for value in handlers.values())
    if trigger == "meter-loss":
        assert result["meter_error"] == "OSError"


@pytest.mark.parametrize("reserve", [0.0, 100.0, 250.0])
def test_parallel_campaign_spends_past_lifetime_caps_then_stops_at_explicit_reserve(supervised, monkeypatch, reserve):
    args, bench, env, _old_budget, events, _handlers, proc = supervised
    args.concurrency = 32
    args.per_task_cost_usd = 25
    args.budget_usd = 2000
    args.budget_reserve_usd = reserve
    budget = budgets.CampaignBudget(bench.parent / "full-campaign.json", fingerprint="key-a",
                                   ceiling=2000, usage=100)
    # Meter values are cumulative key usage. Continue after $1200 (the old
    # 32*$25 reserve stopped here) and $1600; stop only at the selected margin.
    usage = iter([1300, 1700, 2100 - reserve, 2100 - reserve])
    monkeypatch.setattr(launcher, "key_usage", lambda _key, **_kwargs: next(usage))
    continued_at = []
    def still_running(**kwargs):
        continued_at.append(budget.spent)
        raise subprocess.TimeoutExpired("fake-runner", kwargs["timeout"])
    proc.wait = still_running
    result = launcher.supervise_run(args, ["fake-runner"], bench, env, "not-a-real-key", budget)
    assert continued_at == [1200, 1600]
    assert result["stop_reason"] == "campaign_budget_reserve"
    assert result["campaign_spent_usd"] == 2000 - reserve
    assert result["campaign_remaining_usd"] == reserve
    assert result["inflight_reserve_usd"] == reserve
    assert events == ["stop-group", "cleanup-owned"]
    assert "active_run" not in budget.record


def test_negative_explicit_billing_reserve_is_rejected():
    with pytest.raises(SystemExit) as exc:
        launcher.parse_args(["--budget-reserve-usd", "-1"])
    assert exc.value.code == 2


def test_supervisor_handles_sigterm_then_cleans_only_owned_resources(supervised, monkeypatch):
    args, bench, env, budget, events, handlers, proc = supervised
    def launch(*_args, **kwargs):
        assert kwargs["drive_root"] == bench.parent
        assert kwargs["scope"] == "session"
        assert kwargs["purpose"] == "cowork-official-runner"
        assert kwargs.get("new_process_group", True) is True
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return proc
    monkeypatch.setattr(launcher, "spawn_supervised", launch)
    result = launcher.supervise_run(args, ["fake-runner"], bench, env, "not-a-real-key", budget)
    assert result["stop_reason"] == f"signal_{signal.SIGTERM}"
    assert events == ["stop-group", "cleanup-owned"]
    assert all(value == "old-handler" for value in handlers.values())


def test_cleanup_failure_keeps_campaign_custody_unsettled(supervised, monkeypatch):
    args, bench, env, budget, _events, _handlers, _proc = supervised
    pathlib.Path(env["COWORK_STOP_FILE"]).write_text("operator_stop\n", encoding="utf-8")
    def fail_cleanup(*_args):
        raise RuntimeError("Docker unreachable")
    monkeypatch.setattr(launcher, "remove_run_containers", fail_cleanup)
    with pytest.raises(RuntimeError, match="Docker unreachable"):
        launcher.supervise_run(args, ["fake-runner"], bench, env, "not-a-real-key", budget)
    assert budget.record["active_run"] == str(bench.parent)
    with pytest.raises(ValueError, match="unsettled custody"):
        budgets.CampaignBudget(budget.path, fingerprint="key-a", ceiling=1000, usage=100)


def test_cleanup_selection_cannot_include_a_peer_run(monkeypatch):
    calls = []
    remaining = {"containers": "own-a\nown-b\n", "networks": "own-net\n"}
    def docker(host, *argv, **_kwargs):
        assert host == "unix:///owned.sock"
        calls.append(argv)
        if argv[:2] == ("ps", "-aq"):
            assert argv[-1] == f"label={launcher.LABEL_KEY}=owned-run"
            return subprocess.CompletedProcess(argv, 0, remaining["containers"])
        if argv[:3] == ("network", "ls", "-q"):
            assert argv[-1] == f"label={launcher.LABEL_KEY}=owned-run"
            return subprocess.CompletedProcess(argv, 0, remaining["networks"])
        if argv[:2] == ("rm", "-fv"):
            remaining["containers"] = ""
        if argv[:2] == ("network", "rm"):
            remaining["networks"] = ""
        return subprocess.CompletedProcess(argv, 0, "")
    monkeypatch.setattr(launcher, "_docker", docker)
    launcher.remove_run_containers("unix:///owned.sock", "owned-run")
    assert ("rm", "-fv", "own-a", "own-b") in calls
    assert ("network", "rm", "own-net") in calls


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt" or not shutil.which("sh"), reason="real POSIX process-group ownership check")
def test_stop_process_group_terminates_its_child_without_touching_peer(tmp_path):
    child_file = tmp_path / "child.pid"
    shell = shutil.which("sh")
    owned = subprocess.Popen([shell, "-c", 'trap \'kill "$child" 2>/dev/null; wait "$child"; exit 0\' TERM; '
                              'sleep 60 & child=$!; printf "%s" "$child" > "$1"; wait "$child"',
                              "cowork-test", str(child_file)], start_new_session=True)
    peer = subprocess.Popen([shell, "-c", "sleep 60"], start_new_session=True)
    try:
        deadline = time.monotonic() + 3
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_file.exists()
        child = int(child_file.read_text(encoding="utf-8"))
        launcher.stop_process_group(owned)
        assert owned.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        assert peer.poll() is None
    finally:
        for process in (owned, peer):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_cowork_launcher_is_registered_and_obeys_shared_manifest_contract():
    path = pathlib.Path(launcher.__file__).resolve()
    assert path in launcher_paths()
    assert audit_launcher(path) == []


def test_existing_run_is_preserved_and_refusal_gets_its_own_manifest(dry_launcher, monkeypatch):
    out, argv = dry_launcher
    write_json(out / "run_manifest.json", {"previous": "finished run", "score": 0.75})
    write_json(out / "dumps" / "task" / "eval_res.json", {"pass": True})
    original_manifest = (out / "run_manifest.json").read_bytes()
    original_result = (out / "dumps" / "task" / "eval_res.json").read_bytes()
    monkeypatch.setattr(launcher, "timestamp_run_id", lambda _prefix: "fresh-refusal")
    monkeypatch.setattr(launcher, "image_exists", lambda *_a: pytest.fail("reused run reached Docker preflight"))
    assert launcher.main(argv) == 2
    assert (out / "run_manifest.json").read_bytes() == original_manifest
    assert (out / "dumps" / "task" / "eval_res.json").read_bytes() == original_result
    assert not (out / "bench").exists()
    refused = json.loads((out.parent / "fresh-refusal" / "run_manifest.json").read_text(encoding="utf-8"))
    assert refused["extra"]["outcome"] == "refused"
    assert refused["extra"]["exit_code"] == 2
    assert refused["extra"]["refusal"]["requested_root"] == str(out)


def test_web_and_delegated_vision_mirrors_follow_the_actual_tool_catalog():
    from ouroboros.tools.registry import _WEB_TOOLS
    from ouroboros.tools.vision import get_tools

    assert set(launcher.WEB_TOOLS) == set(_WEB_TOOLS)
    # As in Terminal-Bench, local images stay available to the measured model;
    # the vision module's other entries call a separate model.
    assert {tool.name for tool in get_tools()} == set(launcher.DELEGATED_VISION_TOOLS) | {"view_image"}
    for allow_subagents in (False, True):
        disabled = set(launcher.disabled_tools(subagents=allow_subagents))
        assert set(_WEB_TOOLS) | set(launcher.DELEGATED_VISION_TOOLS) <= disabled
        assert "view_image" not in disabled


def recovery_manifest(root, previous, config, *, selected, generation, settled):
    write_json(root / "run_manifest.json", {
        "source": {"head": "seed-sha"},
        "requested_task_ids": selected,
        "harness": {"applied_config": config, "bench": {"head": "bench-sha"},
                    "image_id": IMAGE_ID, "selection_task_ids": selected,
                    "resume_generation": generation,
                    "resume_from": [str(path.resolve()) for path in previous]},
    })
    (root / "result_index.jsonl").write_text(
        "".join(json.dumps({"instance_id": name, "status": status}) + "\n"
                for name, status in settled.items()), encoding="utf-8")


def test_resume_explicit_selection_can_widen_without_repeating_settled_tasks(selection):
    bench, _previous, args, config = selection
    args.task = ["new", "passed", "infra", "wrong-answer"]
    assert launcher.select_tasks(bench, args, config, "seed-sha") == ["new", "infra"]
    assert args.selection_task_ids == args.task


def test_latest_parent_keeps_ancestral_settlements_and_recovery_depth(selection):
    bench, initial, args, config = selection
    first = initial.parent / "recovery-one"
    recovery_manifest(first, [initial], config, selected=["infra", "new"], generation=1,
                      settled={"infra": "passed", "new": "infra_failed"})
    args.resume_from = [str(first)]
    # Explicitly revisiting an old task still consults its ancestor's ledger.
    args.task = ["passed", "wrong-answer", "infra", "new"]
    assert launcher.select_tasks(bench, args, config, "seed-sha") == ["new"]
    assert args.resume_generation == 2
    assert set(args.resume_sources) == {str(initial.resolve()), str(first.resolve())}
    args.task = []
    assert launcher.select_tasks(bench, args, config, "seed-sha") == ["new"]
    assert set(args.selection_task_ids) == {"infra", "new"}


def test_third_recovery_refuses_work_but_empty_completion_needs_no_new_pass(selection):
    bench, initial, args, config = selection
    first, second = initial.parent / "recovery-one", initial.parent / "recovery-two"
    recovery_manifest(first, [initial], config, selected=["infra"], generation=1,
                      settled={"infra": "infra_failed"})
    recovery_manifest(second, [initial, first], config, selected=["infra"], generation=2,
                      settled={"infra": "infra_failed"})
    args.resume_from = [str(second)]
    with pytest.raises(ValueError, match="recovery|resume|two"):
        launcher.select_tasks(bench, args, config, "seed-sha")
    (second / "result_index.jsonl").write_text(
        json.dumps({"instance_id": "infra", "status": "passed"}) + "\n", encoding="utf-8")
    assert launcher.select_tasks(bench, args, config, "seed-sha") == []


def test_multiple_parent_scopes_union_without_expanding_to_full_dataset(selection):
    bench, initial, args, config = selection
    second = initial.parent / "other-initial"
    recovery_manifest(second, [], config, selected=["new"], generation=0,
                      settled={"new": "not_attempted"})
    args.resume_from = [str(initial), str(second)]
    assert set(launcher.select_tasks(bench, args, config, "seed-sha")) == {"infra", "new"}
    assert set(args.selection_task_ids) == {"passed", "wrong-answer", "infra", "new"}
    assert args.resume_generation == 1


def test_missing_resume_ledger_is_an_explicit_refusal(selection):
    bench, previous, args, config = selection
    (previous / "result_index.jsonl").unlink()
    with pytest.raises(ValueError, match="ledger"):
        launcher.select_tasks(bench, args, config, "seed-sha")


def test_resume_checks_immutable_image_even_in_an_older_ancestor(selection):
    bench, initial, args, config = selection
    first = initial.parent / "recovery-one"
    recovery_manifest(first, [initial], config, selected=["infra"], generation=1,
                      settled={"infra": "infra_failed"})
    old = json.loads((initial / "run_manifest.json").read_text(encoding="utf-8"))
    old["harness"]["image_id"] = "sha256:" + "2" * 64
    write_json(initial / "run_manifest.json", old)
    args.resume_from = [str(first)]
    with pytest.raises(ValueError, match="image|configuration"):
        launcher.select_tasks(bench, args, config, "seed-sha")


def test_resuming_a_noop_after_last_recovery_retains_the_original_settlements(selection):
    bench, initial, args, config = selection
    first, second = initial.parent / "recovery-one", initial.parent / "recovery-two"
    recovery_manifest(first, [initial], config, selected=["infra"], generation=1,
                      settled={"infra": "infra_failed"})
    recovery_manifest(second, [initial, first], config, selected=["infra"], generation=2,
                      settled={"infra": "passed"})
    args.resume_from = [str(second)]
    assert launcher.select_tasks(bench, args, config, "seed-sha") == []
    assert args.resume_generation == 2
    noop = initial.parent / "completed-noop"
    recovery_manifest(noop, list(map(pathlib.Path, args.resume_sources)), config,
                      selected=args.selection_task_ids, generation=args.resume_generation, settled={})
    record = json.loads((noop / "run_manifest.json").read_text(encoding="utf-8"))
    record["requested_task_ids"] = []
    record["extra"] = {"outcome": "nothing_remaining"}
    write_json(noop / "run_manifest.json", record)
    args.resume_from = [str(noop)]
    assert launcher.select_tasks(bench, args, config, "seed-sha") == []
    assert args.resume_generation == 2
    assert str(second.resolve()) in args.resume_sources
    assert str(initial.resolve()) in args.resume_sources


def test_image_identity_uses_local_content_id_without_requiring_registry_digest(monkeypatch):
    calls = []
    def docker(host, *argv, **_kwargs):
        calls.append((host, argv))
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            "Id": IMAGE_ID, "Config": {"Labels": {"seed": "fixed"}}, "RepoDigests": [],
        }))
    monkeypatch.setattr(launcher, "_docker", docker)
    identity = launcher.image_identity("unix:///owned.sock", "mutable-tag:latest")
    assert identity == {"id": IMAGE_ID, "labels": {"seed": "fixed"}, "repo_digests": []}
    assert len(calls) == 1
    assert calls[0][0] == "unix:///owned.sock"
    assert calls[0][1][-1] == "mutable-tag:latest"


def test_missing_immutable_image_identity_refuses_instead_of_trusting_tag(monkeypatch):
    monkeypatch.setattr(launcher, "_docker", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, '{"Config": {}}'))
    with pytest.raises(RuntimeError, match="immutable image identity"):
        launcher.image_identity("unix:///owned.sock", "mutable-tag:latest")


@pytest.mark.parametrize("low_disk", [False, True])
def test_image_preparation_obeys_disk_reserve_and_settles_its_process(tmp_path, monkeypatch, low_disk):
    args = argparse.Namespace(resource_root=tmp_path, min_free_gib=200, min_root_free_gib=40,
                              docker_host="unix:///owned.sock")
    events = []
    process = SimpleNamespace(pid=424242, returncode=None)
    process.poll = lambda: process.returncode
    def wait(**_kwargs):
        events.append("wait")
        process.returncode = 0
        return 0
    process.wait = wait
    def spawn(command, **kwargs):
        assert command == ["fake-docker", "build"]
        assert kwargs["drive_root"] == tmp_path
        assert kwargs["scope"] == "session"
        assert kwargs["purpose"] == "cowork-image-preparation"
        assert kwargs["env"]["DOCKER_HOST"] == args.docker_host
        events.append("spawn-owned")
        return process
    def stop(owned):
        assert owned is process
        events.append("settle-owned")
    monkeypatch.setattr(launcher, "spawn_supervised", spawn)
    monkeypatch.setattr(launcher, "stop_process_group", stop)
    monkeypatch.setattr(launcher.shutil, "disk_usage",
                        lambda _path: SimpleNamespace(free=(199 if low_disk else 300) * 1024**3))
    if low_disk:
        with pytest.raises(RuntimeError, match="disk reserve"):
            launcher.run_preparation(["fake-docker", "build"], args, tmp_path / "build.log", timeout=60)
        assert events == ["spawn-owned", "settle-owned"]
    else:
        launcher.run_preparation(["fake-docker", "build"], args, tmp_path / "build.log", timeout=60)
        assert events == ["spawn-owned", "wait", "settle-owned"]


def test_paid_runner_uses_immutable_image_and_scrubs_ambient_alternate_keys(dry_launcher, monkeypatch):
    out, argv = dry_launcher
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-selected-key")
    monkeypatch.setenv("LLM_API_KEY", "test-unrelated-llm-key")
    monkeypatch.setenv("MODEL_API_KEY", "test-unrelated-model-key")
    monkeypatch.setattr(launcher, "key_headroom", lambda _key, **_kwargs: {"effective": 1000})
    monkeypatch.setattr(launcher, "key_usage", lambda _key, **_kwargs: 100)
    monkeypatch.setattr(launcher, "prepare_resource_env", lambda env, **_kwargs: dict(env))
    def supervise(args, command, bench, env, api_key, campaign, *, confirmed_at):
        assert isinstance(confirmed_at, float)
        assert attempts.claim_protocol(bench.parent) == "current"  # read during the real admission path
        assert command[-1] == "one"
        assert env["IMAGE"] == IMAGE_ID
        assert not {"LLM_API_KEY", "MODEL_API_KEY", "OPENROUTER_API_KEY"} & env.keys()
        assert api_key == "test-selected-key"
        secret = json.loads((bench / "configs" / launcher.SECRET_NAME).read_text(encoding="utf-8"))
        assert secret["settings"][args.credential_setting] == api_key
        dump = bench / "dumps" / launcher.dump_dir_name(args.model) / "SingleUserTurn-one"
        write_json(dump / "ouroboros_summary.json", {"bench_status": "success"})
        write_json(dump / "eval_res.json", {"pass": True})
        # The paid-run fixture must supply the exact terminal evidence a real
        # current eval entrypoint publishes, not just a bare unclaimed file.
        attempt_id = "a" * 32
        attempts.publish_exclusive(dump / attempts.CLAIM_NAME,
                                   {"kind": "official", "attempt_id": attempt_id})
        attempts.publish_exclusive(dump / f"{attempts.RECEIPT_PREFIX}{attempt_id}.json", {
            "kind": "official", "attempt_id": attempt_id, "official_run": True,
            "returned": {"pass": True}, "raised": None,
            "result_file": attempts.file_facts(dump / "eval_res.json"),
        })
        counts = launcher.write_ledger(bench.parent / "result_index.jsonl", bench, args.model,
                                       args.selected_tasks, cause="runner_exited")
        return {"stop_reason": "", "meter_error": "", "runner_exit_code": 0,
                "interruption_cause": "runner_exited", "ledger_counts": counts}
    monkeypatch.setattr(launcher, "supervise_run", supervise)
    paid_argv = [item for item in argv if item != "--dry-run"]
    assert launcher.main([*paid_argv, "--campaign-file", str(out.parent / "campaign.json")]) == 0
    assert (out / "bench" / "configs" / launcher.SECRET_NAME).read_text(encoding="utf-8") == "{}"
    assert "test-selected-key" not in (out / "run_manifest.json").read_text(encoding="utf-8")


def test_meter_http_read_requests_uncached_data_with_remaining_timeout(monkeypatch):
    received = []
    def response(request, *, timeout):
        received.append((dict(request.header_items()), timeout))
        return io.BytesIO(b'{"data":{"usage":123.5}}')
    monkeypatch.setattr(budgets.urllib.request, "urlopen", response)
    assert budgets._read_usage_http("not-a-real-key", budgets._KEY_USAGE_URL, timeout=2.25) == 123.5
    assert received[0][0]["Cache-control"] == "no-cache"
    assert received[0][1] == 2.25


def test_backward_meter_read_is_confirmed_without_stopping_the_run(supervised, monkeypatch, capsys):
    args, bench, env, budget, events, _handlers, proc = supervised
    observations = iter([99.0, 101.0, 102.0])
    monkeypatch.setattr(launcher, "key_usage", lambda _key, **_kwargs: next(observations))
    def completed(**_kwargs):
        proc.returncode = 0
        return 0
    proc.wait = completed
    result = launcher.supervise_run(args, ["fake-runner"], bench, env, "not-a-real-key", budget)
    assert result["stop_reason"] == ""
    assert result["meter_error"] == ""
    assert budget.record["last_usage"] == 102.0
    assert budget.spent == 2.0
    diagnostics = result["meter_diagnostics"]
    assert [(row["observed_usage"], row["accepted"]) for row in diagnostics] == [(99.0, False), (101.0, True)]
    assert diagnostics[0]["previous_usage"] == 100
    final = json.loads((bench.parent / "monitor.json").read_text(encoding="utf-8"))
    assert final["finished"] is True and final["meter_diagnostics"] == diagnostics
    assert '"observed_usage": 99.0' in capsys.readouterr().out
    assert events == ["stop-group", "cleanup-owned"]


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_confirmation_never_accepts_an_invalid_numeric_usage(supervised, monkeypatch, invalid):
    _args, _bench, _env, budget, _events, _handlers, _proc = supervised
    values = iter([invalid, 101.0])
    monkeypatch.setattr(launcher, "key_usage", lambda _key, **_kwargs: next(values))
    diagnostics = []
    launcher.observe_campaign_usage("not-a-real-key", budget, diagnostics, phase="poll", deadline=30.0)
    assert budget.record["last_usage"] == 101
    assert diagnostics[0]["accepted"] is False
    assert diagnostics[0]["previous_usage"] == 100
    # Non-finite rejected observations remain printable JSON evidence, never NaN budget truth.
    json.dumps(diagnostics, allow_nan=False)


def test_persistent_bad_counter_stops_and_keeps_rejected_values_in_final_monitor(supervised, monkeypatch):
    args, bench, env, budget, _events, _handlers, _proc = supervised
    calls = []
    def stale(_key, *, timeout):
        calls.append(timeout)
        return 99.0
    monkeypatch.setattr(launcher, "key_usage", stale)
    result = launcher.supervise_run(args, ["fake-runner"], bench, env, "not-a-real-key", budget)
    assert result["stop_reason"] == "budget_meter_unavailable"
    assert budget.record["last_usage"] == 100
    # Reads 3 s apart until the 30 s bound, then one bounded final settlement of the same shape.
    assert calls == ([5.0] * 9 + [3.0]) * 2
    assert {row["phase"] for row in result["meter_diagnostics"]} == {"poll", "final"}
    assert all(row["observed_usage"] == 99.0 and not row["accepted"] for row in result["meter_diagnostics"])
    final = json.loads((bench.parent / "monitor.json").read_text(encoding="utf-8"))
    assert final["meter_diagnostics"] == result["meter_diagnostics"]
    assert final["meter_error"] == "UsageCounterError"


def test_each_read_gets_a_slice_so_one_slow_read_cannot_consume_the_bound(supervised, monkeypatch):
    _args, _bench, _env, budget, _events, _handlers, _proc = supervised
    clock = SimpleNamespace(now=0.0)
    def advance(delay):
        clock.now += delay
    monkeypatch.setattr(launcher, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: clock.now, sleep=advance))
    timeouts = []
    def unavailable(_key, *, timeout):
        timeouts.append(timeout)
        advance(min(7.0, timeout))
        raise OSError("meter unavailable")
    monkeypatch.setattr(launcher, "key_usage", unavailable)
    original = budget.path.read_bytes()
    diagnostics = []
    with pytest.raises(OSError, match="meter unavailable"):
        launcher.observe_campaign_usage("not-a-real-key", budget, diagnostics, phase="poll", deadline=30.0)
    assert timeouts == [5.0] * 4  # Reads at 0, 8, 16 and 24 s; the bound ends at exactly 30 s.
    assert clock.now == 30.0
    assert len(diagnostics) == 4
    assert budget.path.read_bytes() == original


def test_confirmed_exhausted_budget_stops_without_waiting_for_another_poll(supervised, monkeypatch):
    args, bench, env, budget, _events, _handlers, proc = supervised
    calls = []
    def exhausted(_key, *, timeout):
        calls.append(timeout)
        return 1000.0
    monkeypatch.setattr(launcher, "key_usage", exhausted)
    proc.wait = lambda **_kwargs: pytest.fail("known exhausted budget must not continue work")
    result = launcher.supervise_run(args, ["fake-runner"], bench, env, "not-a-real-key", budget)
    assert result["stop_reason"] == "campaign_budget_reserve"
    assert len(calls) == 2  # Poll plus final settlement, with no confirmation retry.
    assert result["meter_diagnostics"] == []


@pytest.mark.parametrize("kind", ["container", "network"])
def test_cleanup_racing_404_is_success_only_after_exact_label_is_empty(monkeypatch, kind):
    seen = []
    removed = False
    def docker(host, *argv, **_kwargs):
        nonlocal removed
        assert host == "unix:///owned.sock"
        seen.append(argv)
        selected = argv[:2] == ("ps", "-aq") if kind == "container" else argv[:3] == ("network", "ls", "-q")
        if selected:
            assert argv[-1] == f"label={launcher.LABEL_KEY}=owned-run"
            return subprocess.CompletedProcess(argv, 0, "" if removed else "own-id\n")
        if argv[:2] in (("rm", "-fv"), ("network", "rm")):
            removed = True  # A competing cleanup got there first.
            return subprocess.CompletedProcess(argv, 1, "", "No such resource")
        return subprocess.CompletedProcess(argv, 0, "")
    monkeypatch.setattr(launcher, "_docker", docker)
    launcher.remove_run_containers("unix:///owned.sock", "owned-run")
    assert removed
    assert sum(argv[-1] == f"label={launcher.LABEL_KEY}=owned-run" for argv in seen) == 3


@pytest.mark.parametrize("remove_exit", [0, 1])
def test_cleanup_never_claims_surviving_containers_are_gone(monkeypatch, remove_exit):
    def docker(_host, *argv, **_kwargs):
        if argv[:2] == ("ps", "-aq"):
            return subprocess.CompletedProcess(argv, 0, "still-owned\n")
        return subprocess.CompletedProcess(argv, remove_exit, "")
    monkeypatch.setattr(launcher, "_docker", docker)
    with pytest.raises(RuntimeError, match="could not remove all containers"):
        launcher.remove_run_containers("unix:///owned.sock", "owned-run")


def test_cleanup_failed_relist_is_unknown_even_when_stdout_is_empty(monkeypatch):
    lists = 0
    def docker(_host, *argv, **_kwargs):
        nonlocal lists
        if argv[:2] == ("ps", "-aq"):
            lists += 1
            return subprocess.CompletedProcess(argv, 0 if lists == 1 else 1, "own-id\n" if lists == 1 else "")
        return subprocess.CompletedProcess(argv, 1, "", "No such resource")
    monkeypatch.setattr(launcher, "_docker", docker)
    with pytest.raises(RuntimeError, match="cannot verify.*custody after removal"):
        launcher.remove_run_containers("unix:///owned.sock", "owned-run")


def test_lagging_counter_gets_time_to_catch_up_without_weakening_monotonicity(supervised, monkeypatch):
    _args, _bench, _env, budget, _events, _handlers, _proc = supervised
    clock = SimpleNamespace(now=0.0)
    def advance(delay):
        clock.now += delay
    monkeypatch.setattr(launcher, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: clock.now, sleep=advance))
    reads = []
    def provider(_key, *, timeout):
        reads.append((clock.now, timeout))
        # Neither the stale value nor the waiting period may lower durable spending.
        assert budget.record["last_usage"] == 100
        assert json.loads(budget.path.read_text(encoding="utf-8"))["last_usage"] == 100
        return 99.0 if clock.now < 3.0 else 101.0
    monkeypatch.setattr(launcher, "key_usage", provider)
    diagnostics = []
    anchor = launcher.observe_campaign_usage("not-a-real-key", budget, diagnostics, phase="poll", deadline=30.0)
    assert reads == [(0.0, 5.0), (3.0, 5.0)]
    assert anchor == 3.0  # The accepted read's request time, not the start of the confirmation.
    assert budget.record["last_usage"] == 101
    assert [(row["observed_usage"], row["accepted"]) for row in diagnostics] == [(99.0, False), (101.0, True)]
    assert clock.now == 3.0


@pytest.mark.parametrize("removal", ["stderr-with-zero", "nonzero", "exception"])
def test_cleanup_retains_actual_removal_diagnostics_without_requiring_nonzero(monkeypatch, capsys, removal):
    listed = 0
    def docker(_host, *argv, **_kwargs):
        nonlocal listed
        if argv[:2] == ("ps", "-aq"):
            listed += 1
            return subprocess.CompletedProcess(argv, 0, "own-id\n" if listed == 1 else "")
        if argv[:2] == ("rm", "-fv"):
            if removal == "exception":
                raise OSError("daemon connection reset")
            return subprocess.CompletedProcess(argv, int(removal == "nonzero"), "", "No such container: own-id")
        return subprocess.CompletedProcess(argv, 0, "")
    monkeypatch.setattr(launcher, "_docker", docker)
    launcher.remove_run_containers("unix:///owned.sock", "owned-run")
    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "cowork_cleanup_remove"
    assert record["run_label"] == "owned-run"
    assert record["selected_ids"] == ["own-id"]
    if removal == "exception":
        assert record["exception"] == {"type": "OSError", "message": "daemon connection reset"}
        assert record["returncode"] is None
    else:
        assert record["returncode"] == int(removal == "nonzero")
        assert record["stderr"] == "No such container: own-id"
        assert record["stdout"] == ""


def test_cleanup_logs_survivor_ids_and_still_refuses_custody_release(monkeypatch, capsys):
    def docker(_host, *argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, "owned-survivor\n" if argv[:2] == ("ps", "-aq") else "")
    monkeypatch.setattr(launcher, "_docker", docker)
    with pytest.raises(RuntimeError, match="could not remove all containers"):
        launcher.remove_run_containers("unix:///owned.sock", "owned-run")
    record = json.loads(capsys.readouterr().out)
    assert record == {"event": "cowork_cleanup_remaining", "resource": "container",
                      "run_label": "owned-run", "remaining_ids": ["owned-survivor"]}


def test_late_valid_confirmation_is_not_accepted_even_if_reader_returns_it(supervised, monkeypatch):
    _args, _bench, _env, budget, _events, _handlers, _proc = supervised
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(launcher, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds)))
    def late(_key, *, timeout):
        clock.now += timeout + 1
        return 101.0
    monkeypatch.setattr(launcher, "key_usage", late)
    diagnostics = []
    with pytest.raises(TimeoutError, match="after the blindness bound"):
        launcher.observe_campaign_usage("not-a-real-key", budget, diagnostics, phase="poll", deadline=4.0)
    assert budget.record["last_usage"] == 100
    assert diagnostics[0]["observed_usage"] == 101
    assert diagnostics[0]["accepted"] is False
