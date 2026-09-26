"""Meter blindness, failure domains and interrupted-task rows through the real supervisor.

A simulated clock drives every read, retry pause and runner wait. The meter is a
script, and the runner, its process-group stop and exact-label cleanup are recorded
fakes, so no provider, Docker daemon or paid benchmark is reached.
"""

from __future__ import annotations

from functools import partial
import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from devtools.benchmarks.common import manifests
from devtools.benchmarks.cowork_bench import campaign as budgets
from devtools.benchmarks.cowork_bench import eval_attempt as attempts
from devtools.benchmarks.cowork_bench import run_cowork_bench as launcher

legacy_ledger_row = partial(launcher.ledger_row, protocol="legacy")

@pytest.fixture(autouse=True)
def forbid_live_services(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("offline meter test tried to contact a provider or Docker")
    monkeypatch.setattr(budgets.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(launcher, "_docker", forbidden)


def slow(sim, timeout):
    sim.now += timeout  # The read consumes its whole slice, as in the #1259 incident.
    raise TimeoutError("meter HTTP read exceeded its wall-clock deadline")


def offline(_sim, _timeout):
    raise OSError("meter unavailable")


def regressed(_sim, _timeout):
    return 99.0


@pytest.fixture
def sim(tmp_path, monkeypatch):
    """One run whose meter answers ``sim.script`` in order, then ``sim.default``."""
    state = SimpleNamespace(now=0.0, events=[], reads=[], script=[], runner_ends=None, campaign_writes=0,
                            fail_campaign_write=lambda _index: False)
    state.default = lambda sim, _timeout: sim.budget.record["last_usage"]
    bench = tmp_path / "bench"
    bench.mkdir()
    args = launcher.parse_args([])
    args.resource_root = tmp_path
    args.docker_host = "unix:///owned.sock"
    args.selected_tasks = []
    args.min_free_gib = args.min_root_free_gib = 0
    env = {"COWORK_STOP_FILE": str(tmp_path / "resource_stop"), "COWORK_RUN_LABEL": "owned-run"}
    state.budget = budgets.CampaignBudget(tmp_path / "campaign.json", fingerprint="key-a", ceiling=1000, usage=100)
    state.args, state.bench, state.env = args, bench, env
    state.dumps = bench / "dumps" / launcher.dump_dir_name(args.model)
    state.stop_file = pathlib.Path(env["COWORK_STOP_FILE"])
    proc = SimpleNamespace(pid=424242, returncode=None)
    proc.poll = lambda: proc.returncode

    def wait(*, timeout):
        if state.now > 3600:
            pytest.fail("the simulated run was never stopped")
        if state.runner_ends is not None and state.now + timeout >= state.runner_ends:
            state.now = max(state.now, state.runner_ends)
            proc.returncode = 0
            return 0
        state.now += timeout
        raise subprocess.TimeoutExpired("fake-runner", timeout)

    def spawn(command, **kwargs):
        assert command == ["fake-runner"] and kwargs["purpose"] == "cowork-official-runner"
        state.events.append((state.now, "spawn"))
        return proc

    def stop(owned):
        assert owned is proc
        state.events.append((state.now, "stop-group"))
        owned.returncode = -15

    def cleanup(host, label):
        assert (host, label) == (args.docker_host, env["COWORK_RUN_LABEL"])
        state.events.append((state.now, "cleanup-owned"))

    def meter(_key, *, timeout):
        state.reads.append((state.now, timeout))
        step = state.script.pop(0) if state.script else state.default
        return step(state, timeout) if callable(step) else step

    real_write_json = manifests.write_json

    def campaign_write(path, payload):
        if pathlib.Path(path) == state.budget.path:
            state.campaign_writes += 1
            if state.fail_campaign_write(state.campaign_writes):
                raise OSError(28, "No space left on device")
        real_write_json(path, payload)

    proc.wait = wait
    monkeypatch.setattr(launcher, "spawn_supervised", spawn)
    monkeypatch.setattr(launcher, "stop_process_group", stop)
    monkeypatch.setattr(launcher, "remove_run_containers", cleanup)
    monkeypatch.setattr(launcher, "key_usage", meter)
    monkeypatch.setattr(manifests, "write_json", campaign_write)  # The campaign's own writer.
    monkeypatch.setattr(launcher.signal, "signal", lambda _sig, _handler: "old-handler")
    monkeypatch.setattr(launcher, "time", SimpleNamespace(
        time=lambda: state.now, monotonic=lambda: state.now,
        sleep=lambda delay: setattr(state, "now", state.now + delay)))
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1024**4))
    state.supervise = lambda: launcher.supervise_run(args, ["fake-runner"], bench, env, "not-a-real-key",
                                                     state.budget)
    return state


def lifecycle(sim) -> list[str]:
    """Every owned effect happens once: one runner, one group stop, one exact-label cleanup."""
    return [name for _at, name in sim.events]


def durable(sim) -> dict:
    return json.loads(sim.budget.path.read_text(encoding="utf-8"))


def reads_between(sim, start, end) -> list[tuple[float, float]]:
    return [read for read in sim.reads if start <= read[0] < end]


def test_healthy_meter_polls_every_fifteen_seconds_with_bounded_reads(sim):
    sim.runner_ends = 60.0
    sim.default = lambda s, _timeout: 100.0 + s.now / 10
    result = sim.supervise()
    assert result["stop_reason"] == "" and result["meter_error"] == ""
    assert sim.reads == [(0.0, 5.0), (15.0, 5.0), (30.0, 5.0), (45.0, 5.0), (60.0, 5.0)]
    assert durable(sim)["last_usage"] == 106.0
    assert durable(sim)["runs"][-1]["outcome"] == "runner_finished"
    assert lifecycle(sim) == ["spawn", "stop-group", "cleanup-owned"]


def test_one_slow_read_uses_only_its_slice_and_a_second_read_confirms(sim):
    sim.runner_ends = 50.0
    sim.script = [101.0, slow, 102.0]
    result = sim.supervise()
    assert result["stop_reason"] == ""
    # The slow read at 15 s gets 5 s, not the whole remaining bound; a retry follows at 23 s.
    assert sim.reads[:3] == [(0.0, 5.0), (15.0, 5.0), (23.0, 5.0)]
    assert [(row["error_type"] if not row["accepted"] else "accepted") for row in result["meter_diagnostics"]] == [
        "TimeoutError", "accepted"]
    # The accepted read at 23 s anchors the next bound, so the run keeps polling after 30 s.
    assert (38.0, 5.0) in sim.reads
    assert lifecycle(sim) == ["spawn", "stop-group", "cleanup-owned"]


def test_immediate_transient_errors_retry_every_three_seconds_inside_the_bound(sim):
    sim.runner_ends = 40.0
    sim.script = [101.0, offline, offline, 102.0]
    result = sim.supervise()
    assert result["stop_reason"] == ""
    assert [at for at, _timeout in sim.reads[:4]] == [0.0, 15.0, 18.0, 21.0]
    assert durable(sim)["last_usage"] == 102.0
    assert lifecycle(sim) == ["spawn", "stop-group", "cleanup-owned"]


@pytest.mark.parametrize("bound", [30.0, 45.0])
def test_persistent_outage_stops_admission_and_run_at_the_bound_then_settles_bounded(sim, bound):
    sim.args.meter_blindness_sec = bound
    sim.script = [101.0]
    sim.default = offline
    result = sim.supervise()
    assert result["stop_reason"] == "budget_meter_unavailable"
    assert result["meter_error"] == "OSError"
    assert sim.stop_file.read_text(encoding="utf-8").strip() == "budget_meter_unavailable"
    # Blindness counts from the last confirmed, saved reading at 0 s, including waits and retries.
    assert sim.events == [(0.0, "spawn"), (bound, "stop-group"), (bound, "cleanup-owned")]
    # The final settlement is accounting only, after cleanup, and bounded by the same B.
    assert reads_between(sim, bound, 2 * bound) and sim.now == 2 * bound
    record = durable(sim)
    assert "active_run" not in record and record["last_usage"] == 101.0
    assert record["runs"][-1]["outcome"] == "budget_meter_unavailable"
    assert record["runs"][-1]["meter_error"] == "OSError"


def test_repeated_counter_regression_never_resets_the_bound_or_lowers_spend(sim):
    sim.script = [101.0]
    sim.default = regressed
    result = sim.supervise()
    assert result["stop_reason"] == "budget_meter_unavailable"
    assert result["meter_error"] == "UsageCounterError"
    assert [at for at, _timeout in reads_between(sim, 1, 30)] == [15.0, 18.0, 21.0, 24.0, 27.0]
    assert (30.0, "stop-group") in sim.events
    rejected = [row for row in result["meter_diagnostics"] if not row["accepted"]]
    assert rejected and all(row["observed_usage"] == 99.0 and row["previous_usage"] == 101.0 for row in rejected)
    assert durable(sim)["last_usage"] == 101.0


def test_late_success_is_rejected_and_final_settlement_records_it_after_cleanup(sim):
    def late(s, _timeout):
        s.now = 31.0  # A reader that returns a valid value only after the bound.
        return 150.0

    sim.script = [101.0, late, 150.0]
    result = sim.supervise()
    assert result["stop_reason"] == "budget_meter_unavailable"
    late_row = result["meter_diagnostics"][0]
    assert late_row["observed_usage"] == 150.0 and late_row["accepted"] is False
    assert "after the blindness bound" in late_row["error"]
    assert sim.events == [(0.0, "spawn"), (31.0, "stop-group"), (31.0, "cleanup-owned")]
    # Only the post-cleanup settlement may record 150; it cannot authorize further work.
    assert sim.reads[-1] == (31.0, 5.0)
    assert result["campaign_spent_usd"] == 50.0
    assert durable(sim)["runs"][-1]["outcome"] == "budget_meter_unavailable"
    assert lifecycle(sim) == ["spawn", "stop-group", "cleanup-owned"]


def test_bound_counts_diagnostic_time_and_polls_early_enough_for_one_read(sim, monkeypatch):
    real_write_ledger = launcher.write_ledger

    def slow_ledger(*args, **kwargs):
        sim.now += 16.0
        return real_write_ledger(*args, **kwargs)

    monkeypatch.setattr(launcher, "write_ledger", slow_ledger)
    sim.script = [101.0]
    sim.default = offline
    result = sim.supervise()
    # A 16 s diagnostic write shortens the next wait so one full read starts at 25 s.
    assert reads_between(sim, 1, 30) == [(25.0, 5.0), (28.0, 2.0)]
    assert result["stop_reason"] == "budget_meter_unavailable"
    assert (30.0, "stop-group") in sim.events


def test_transient_campaign_write_failure_stops_with_its_own_reason(sim):
    sim.runner_ends = 100.0
    sim.script = [101.0, 105.0]
    sim.fail_campaign_write = lambda index: index == 3  # start, first poll, then the failed save
    result = sim.supervise()
    assert result["stop_reason"] == "campaign_persistence_failed"
    assert result["meter_error"] == "CampaignPersistenceError"
    assert sim.stop_file.read_text(encoding="utf-8").strip() == "campaign_persistence_failed"
    assert sim.events == [(0.0, "spawn"), (15.0, "stop-group"), (15.0, "cleanup-owned")]
    record = durable(sim)
    assert record["runs"][-1]["outcome"] == "campaign_persistence_failed"
    assert record["last_usage"] == 105.0 and "active_run" not in record


def test_persistent_campaign_write_failure_keeps_custody_unsettled_after_exact_cleanup(sim):
    sim.runner_ends = 100.0
    sim.script = [101.0, 105.0]
    sim.args.selected_tasks = ["cut"]
    write_task(sim.dumps, "cut", {"applied_settings.json": {}, "ouroboros/events.jsonl": [USAGE]})
    sim.fail_campaign_write = lambda index: index >= 3
    with pytest.raises(budgets.CampaignPersistenceError):
        sim.supervise()
    assert sim.stop_file.read_text(encoding="utf-8").strip() == "campaign_persistence_failed"
    assert lifecycle(sim) == ["spawn", "stop-group", "cleanup-owned"]
    record = durable(sim)
    assert record["active_run"] == str(sim.bench.parent) and record["last_usage"] == 101.0
    row = json.loads((sim.bench.parent / "result_index.jsonl").read_text(encoding="utf-8"))
    assert (row["status"], row["reason_code"], row["details"]["paid_activity"]) == (
        "infra_failed", "interrupted:campaign_persistence_failed", "observed")
    assert json.loads((sim.bench.parent / "monitor.json").read_text(encoding="utf-8"))["campaign_settlement"] == "unconfirmed"
    sim.fail_campaign_write = lambda _index: False
    with pytest.raises(ValueError, match="unsettled custody"):
        budgets.CampaignBudget(sim.budget.path, fingerprint="key-a", ceiling=1000, usage=110)


def test_unwritable_stop_marker_never_skips_owned_cleanup_or_interruption_evidence(sim, monkeypatch, capsys):
    sim.args.selected_tasks = ["cut"]
    write_task(sim.dumps, "cut", {"applied_settings.json": {}, "ouroboros/events.jsonl": [USAGE]})
    sim.script = [101.0]
    sim.default = offline

    def unwritable(_path, _reason):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(launcher, "mark_stop", unwritable)
    with pytest.raises(OSError, match="No space left on device"):
        sim.supervise()
    assert lifecycle(sim) == ["spawn", "stop-group", "cleanup-owned"]
    assert "cowork_stop_marker_failed" in capsys.readouterr().err
    row = json.loads((sim.bench.parent / "result_index.jsonl").read_text(encoding="utf-8"))
    assert (row["status"], row["reason_code"]) == ("infra_failed", "interrupted:budget_meter_unavailable")
    assert durable(sim)["runs"][-1]["outcome"] == "budget_meter_unavailable"


def test_diagnostic_write_failures_are_disclosed_and_valid_work_continues(sim, monkeypatch, capsys):
    def unavailable(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(launcher, "write_ledger", unavailable)
    monkeypatch.setattr(launcher, "write_json", unavailable)
    sim.runner_ends = 40.0
    result = sim.supervise()
    assert result["stop_reason"] == "" and result["runner_exit_code"] == 0
    assert {name: row["count"] for name, row in result["diagnostic_write_failures"].items()} == {
        "result_index.jsonl": 4, "monitor.json": 4}  # Polls at 0, 15 and 30 s plus the final record.
    stderr = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert {row["artifact"] for row in stderr if row["event"] == "cowork_diagnostic_write_failed"} == {
        "result_index.jsonl", "monitor.json"}
    assert not (sim.bench.parent / "monitor.json").exists()
    assert durable(sim)["runs"][-1]["outcome"] == "runner_finished"
    assert lifecycle(sim) == ["spawn", "stop-group", "cleanup-owned"]


def test_unexpected_supervisor_failure_is_not_reported_as_a_finished_runner(sim, monkeypatch):
    polls = []

    def disk(_path):
        polls.append(sim.now)
        if len(polls) > 2:
            raise PermissionError("disk probe denied")
        return SimpleNamespace(free=1024**4)

    monkeypatch.setattr(launcher.shutil, "disk_usage", disk)
    with pytest.raises(PermissionError):
        sim.supervise()
    assert sim.stop_file.read_text(encoding="utf-8").strip() == "supervisor_error"
    assert durable(sim)["runs"][-1]["outcome"] == "supervisor_error"
    assert lifecycle(sim) == ["spawn", "stop-group", "cleanup-owned"]


def test_empty_stop_request_is_named_rather_than_runner_finished(sim, monkeypatch):
    def spawn(_command, **_kwargs):
        sim.stop_file.write_text("", encoding="utf-8")
        sim.events.append((sim.now, "spawn"))
        return SimpleNamespace(pid=1, returncode=None, poll=lambda: None)

    monkeypatch.setattr(launcher, "stop_process_group", lambda _proc: sim.events.append((sim.now, "stop-group")))
    monkeypatch.setattr(launcher, "spawn_supervised", spawn)
    result = sim.supervise()
    assert result["stop_reason"] == "stop_requested"
    assert durable(sim)["runs"][-1]["outcome"] == "stop_requested"


def write_task(dumps: pathlib.Path, task: str, files: dict[str, object]) -> None:
    root = dumps / f"SingleUserTurn-{task}"
    (root / "workspace").mkdir(parents=True)  # The benchmark pre-creates both directories.
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, list):
            path.write_text("".join(json.dumps(row) + "\n" for row in value), encoding="utf-8")
        else:
            path.write_text(json.dumps(value), encoding="utf-8")


USAGE = {"type": "llm_usage", "prompt_tokens": 1200, "completion_tokens": 30, "cost": 0.01}


def test_meter_stop_marks_started_tasks_interrupted_with_honest_paid_activity(sim):
    write_task(sim.dumps, "paid", {"applied_settings.json": {}, "ouroboros/events.jsonl": [{"type": "task"}, USAGE]})
    write_task(sim.dumps, "admitted", {"applied_settings.json": {}, "ouroboros/events.jsonl": [{"type": "task"}]})
    write_task(sim.dumps, "precreated", {})
    write_task(sim.dumps, "finished", {"ouroboros_summary.json": {"bench_status": "success"},
                                       "eval_res.json": {"pass": True}})
    finished = sim.dumps / "SingleUserTurn-finished"
    attempt_id = "b" * 32
    attempts.publish_exclusive(finished / attempts.CLAIM_NAME,
                               {"kind": "official", "attempt_id": attempt_id})
    attempts.publish_exclusive(finished / f"{attempts.RECEIPT_PREFIX}{attempt_id}.json", {
        "kind": "official", "attempt_id": attempt_id, "official_run": True,
        "returned": {"pass": True}, "raised": None,
        "result_file": attempts.file_facts(finished / "eval_res.json"),
    })
    sim.args.selected_tasks = ["paid", "admitted", "precreated", "never", "finished"]
    sim.script = [101.0]
    sim.default = offline
    result = sim.supervise()
    assert result["interruption_cause"] == "budget_meter_unavailable"
    rows = {row["instance_id"]: row for row in map(json.loads, (sim.bench.parent / "result_index.jsonl")
                                                   .read_text(encoding="utf-8").splitlines())}
    assert {task: (row["status"], row["reason_code"], row["details"].get("paid_activity"))
            for task, row in rows.items()} == {
        "paid": ("infra_failed", "interrupted:budget_meter_unavailable", "observed"),
        "admitted": ("infra_failed", "interrupted:budget_meter_unavailable", "unknown"),
        "precreated": ("not_attempted", "missing_result", None),
        "never": ("not_attempted", "missing_result", None),
        "finished": ("passed", "passed", None),
    }
    assert rows["paid"]["details"]["start_evidence"] == ["ouroboros", "applied_settings.json"]
    assert not any("cost" in key for row in rows.values() for key in row["details"])
    # Interrupted rows stay unsettled for an explicit recovery; the passed task is never repeated.
    assert launcher.settled_tasks([sim.bench.parent]) == {"finished"}


def test_task_without_summary_after_the_runner_exited_names_that_cause(sim):
    write_task(sim.dumps, "cut", {"mcp_proxy.log": "", "ouroboros/events.jsonl": [USAGE]})
    sim.args.selected_tasks = ["cut"]
    sim.runner_ends = 20.0
    result = sim.supervise()
    assert result["stop_reason"] == "" and result["interruption_cause"] == "runner_exited"
    row = json.loads((sim.bench.parent / "result_index.jsonl").read_text(encoding="utf-8"))
    assert (row["status"], row["reason_code"], row["details"]["paid_activity"]) == (
        "infra_failed", "interrupted:runner_exited", "observed")


@pytest.mark.parametrize("events,expected", [
    ([{"type": "llm_usage", "prompt_tokens": 0, "completion_tokens": 0}], "unknown"),
    ([{"type": "llm_usage", "usage": {"prompt_tokens": 7, "completion_tokens": 1}}], "observed"),
    ([{"type": "llm_usage"}], "unknown"),
])
def test_only_token_bearing_usage_is_observed_paid_activity(tmp_path, events, expected):
    write_task(tmp_path, "task", {"ouroboros/events.jsonl": events})
    row = legacy_ledger_row("task", tmp_path / "SingleUserTurn-task", {}, cause="in_progress")
    assert (row["status"], row["reason_code"]) == ("infra_failed", "missing_adapter_summary")
    assert row["details"]["provisional"] is True
    assert row["details"]["paid_activity"] == expected


def test_runner_row_without_summary_names_interruption_and_discloses_activity(tmp_path):
    write_task(tmp_path, "task", {"ouroboros/events.jsonl": [USAGE]})
    row = legacy_ledger_row("task", tmp_path / "SingleUserTurn-task", {"status": "unknown"}, cause="signal_15")
    assert (row["status"], row["reason_code"], row["details"]["paid_activity"]) == (
        "infra_failed", "interrupted:signal_15", "observed")
    assert row["details"]["provisional"] is False


def test_interrupted_agent_keeps_independent_official_evaluator_receipt(tmp_path):
    write_task(tmp_path, "task", {"applied_settings.json": {}, "ouroboros/events.jsonl": [USAGE],
                                   "eval_res.json": {"pass": False, "failure": "private evaluator detail"}})
    row = legacy_ledger_row("task", tmp_path / "SingleUserTurn-task", {}, cause="budget_meter_unavailable")
    assert (row["status"], row["reason_code"], row["official_eval_status"]) == (
        "infra_failed", "interrupted:budget_meter_unavailable", "completed")
    assert row["details"]["paid_activity"] == "observed"
    assert row["details"]["official_receipt"]["pass"] is False
    assert "private evaluator detail" not in json.dumps(row)


def test_evaluator_log_alone_does_not_claim_agent_started(tmp_path):
    write_task(tmp_path, "task", {"traj_log.json": {"status": "failed"},
                                   "eval_res.json": {"pass": None}})
    row = legacy_ledger_row("task", tmp_path / "SingleUserTurn-task", {}, cause="budget_meter_unavailable")
    assert (row["status"], row["reason_code"], row["official_eval_status"]) == (
        "not_attempted", "missing_result", "unknown")


def test_slow_persistence_cannot_renew_an_expired_window(sim, monkeypatch):
    real_observe = sim.budget.observe

    def slow_save(usage):
        if sim.now == 15.0:
            sim.now = 31.0
        real_observe(usage)

    monkeypatch.setattr(sim.budget, "observe", slow_save)
    result = sim.supervise()
    assert result["stop_reason"] == "budget_meter_unavailable"
    assert sim.events == [(0.0, "spawn"), (31.0, "stop-group"), (31.0, "cleanup-owned")]
    assert [at for at, _ in sim.reads] == [0.0, 15.0, 31.0]  # Last is accounting-only.


def test_startup_persistence_cannot_spawn_with_an_expired_observation(sim, monkeypatch):
    real_start = sim.budget.start

    def slow_start(root):
        sim.now = 31.0
        real_start(root)

    monkeypatch.setattr(sim.budget, "start", slow_start)
    with pytest.raises(TimeoutError, match="before runner spawn"):
        sim.supervise()
    assert lifecycle(sim) == ["cleanup-owned"]
    assert sim.stop_file.read_text(encoding="utf-8").strip() == "budget_meter_unavailable"


def test_overlong_diagnostics_stop_on_return_even_if_runner_finished(sim, monkeypatch):
    def slow_ledger(*_args, **_kwargs):
        sim.now += 31.0
        return {}

    monkeypatch.setattr(launcher, "write_ledger", slow_ledger)
    result = sim.supervise()
    assert result["stop_reason"] == "budget_meter_unavailable"
    assert sim.events == [(0.0, "spawn"), (31.0, "stop-group"), (31.0, "cleanup-owned")]
    assert [at for at, _ in sim.reads] == [0.0, 31.0]
