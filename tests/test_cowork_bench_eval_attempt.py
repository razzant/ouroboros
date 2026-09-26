"""At-most-once official evaluation and the residual-state diagnostic, without Docker.

The benchmark side is a task-authored fake of the pinned upstream interfaces (``TaskConfig``,
``Evaluation``, ``TaskEvaluator`` and a checker); no real benchmark evaluator or task checker
is read or run. End-to-end cases execute the real entrypoint beside its copied helpers, as the
image lays them out, and parse the combined stdout/stderr with the runner's own grep rules.
"""
from __future__ import annotations

import argparse
import ast
from functools import partial
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import types

import pytest

from devtools.benchmarks.cowork_bench import eval_attempt as attempts
from devtools.benchmarks.cowork_bench import run_cowork_bench as launcher

current_ledger_row = partial(launcher.ledger_row, protocol="current")
pytestmark = pytest.mark.skipif(os.name == "nt", reason="the Cowork eval container is POSIX")

REPO = pathlib.Path(__file__).resolve().parents[1]
COWORK = REPO / "devtools" / "benchmarks" / "cowork_bench"
TASK = "fixture_task"
MODEL = "fixture-model"
DUMP_DIR = f"ouroboros_{MODEL}"
SAVED_LAUNCH = "2026-09-24 09:15:00 Wednesday"

FAKE_TASK_CONFIG = r'''
import os
import sys
from pathlib import Path

if "diagnose" in sys.argv:
    print("Pass: True <- import-time marker in the diagnostic child", flush=True)
    print("====== Status: success ======", file=sys.stderr, flush=True)


class Evaluation:
    def __init__(self, groundtruth_workspace=None, evaluation_command=None):
        self.groundtruth_workspace = groundtruth_workspace
        self.evaluation_command = evaluation_command

    @classmethod
    def build(cls, task_dir, cn_mode=False):
        if os.environ.get("FAKE_BUILD_FORBIDDEN"):
            raise AssertionError("Evaluation.build called although both saved values exist")
        root = Path("tasks/finalpool") / task_dir
        command = (f"{os.environ.get('PYTHON_BIN', 'python3')} -m tasks.finalpool.{task_dir}.evaluation.main"
                   if (root / "evaluation" / "main.py").exists() else None)
        groundtruth = str(root / "groundtruth_workspace") if (root / "groundtruth_workspace").exists() else None
        return cls(groundtruth, command)


class TaskConfig:
    def __init__(self, task_dir, task_root=None, log_file=None, agent_workspace=None, launch_time=None,
                 evaluation=None, single_turn_mode=False, cn_mode=False, global_task_config=None,
                 agent_short_name=None, **_ignored):
        if os.environ.get("FAKE_TASK_CONFIG_RAISES") and "diagnose" in sys.argv:
            raise RuntimeError("Pass: True inside a traceback")
        self.task_dir, self.launch_time, self.evaluation = task_dir, launch_time, evaluation
        self.single_turn_mode, self.cn_mode = single_turn_mode, cn_mode
        root = Path(task_root or task_dir)
        prefix = ("Chinese-" if cn_mode else "") + ("SingleUserTurn-" if single_turn_mode else "")
        root = root.with_name(prefix + root.name)
        if global_task_config and "dump_path" in global_task_config:
            root = Path(global_task_config["dump_path"]) / agent_short_name.replace("/", "_") / root
        self.task_root = os.path.abspath(root)
        self.log_file = os.path.abspath(log_file or os.path.join(self.task_root, "traj_log.json"))
        self.agent_workspace = os.path.abspath(agent_workspace or os.path.join(self.task_root, "workspace"))
        # Upstream __post_init__ removes <task_root>/eval_res.json.
        stale = Path(self.task_root) / "eval_res.json"
        if stale.exists():
            stale.unlink()

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["evaluation"] = Evaluation(**data["evaluation"])
        return cls(**data)

    @classmethod
    def build(cls, task_dir, agent_short_name=None, global_task_config=None, single_turn_mode=False, cn_mode=False):
        return cls(task_dir, evaluation=Evaluation.build(task_dir, cn_mode), single_turn_mode=single_turn_mode,
                   cn_mode=cn_mode, global_task_config=global_task_config, agent_short_name=agent_short_name,
                   launch_time="2026-09-25 10:00:00 Thursday")
'''

FAKE_EVALUATOR = r'''
import json
import os
import time

from utils.data_structures.task_config import TaskConfig


class TaskEvaluator:
    @staticmethod
    async def evaluate_one(dump_line):
        TaskConfig.from_dict(dump_line["config"])
        status = dump_line["status"]
        if status != "success":
            return {"pass": None, "details": f"Task status: {status}, only SUCCESS counts as pass; pass is null"}
        return {"pass": os.environ.get("FAKE_OFFICIAL_PASS", "true") == "true", "details": "fixture verdict"}

    @staticmethod
    async def evaluate_from_log_file(log_file_path, allow_resume=False):
        with open(os.environ["FAKE_EVALUATOR_CALLS"], "a", encoding="utf-8") as calls:
            calls.write(f"{os.getpid()}\n")
        time.sleep(float(os.environ.get("FAKE_EVALUATOR_DELAY", "0")))
        if os.environ.get("FAKE_EVALUATOR_CRASH"):
            os._exit(9)
        with open(log_file_path, encoding="utf-8") as handle:
            eval_res = await TaskEvaluator.evaluate_one(json.load(handle))
        with open(os.path.join(os.path.dirname(log_file_path), "eval_res.json"), "w", encoding="utf-8") as handle:
            json.dump(eval_res, handle)
        return eval_res
'''

FAKE_HELPER = r'''
import json


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
'''

FAKE_CHECKER = r'''
import json
import os
import subprocess
import sys
import time

print("Pass: True"); print("====== Status: success ======", flush=True)
print("Pass: True", file=sys.stderr); print("Status: success", file=sys.stderr, flush=True)
with open(os.environ["FAKE_CHECKER_ARGV"], "w", encoding="utf-8") as handle:
    json.dump(sys.argv[1:], handle)
ready = os.environ["FAKE_CHECKER_PIDS"] + ".ready"
grandchild = subprocess.Popen([sys.executable, "-c", (
    "import pathlib, sys, time; print('Pass: True from grandchild', flush=True); "
    "print('Status: success', file=sys.stderr, flush=True); pathlib.Path(sys.argv[2]).touch(); "
    "time.sleep(float(sys.argv[1]))"), os.environ.get("FAKE_GRANDCHILD_SLEEP", "30"), ready])
with open(os.environ["FAKE_CHECKER_PIDS"], "w", encoding="utf-8") as handle:
    json.dump([os.getpid(), grandchild.pid], handle)
deadline = time.monotonic() + 20
while not os.path.exists(ready) and time.monotonic() < deadline:
    time.sleep(0.01)
time.sleep(float(os.environ.get("FAKE_CHECKER_SLEEP", "0")))
sys.exit(int(os.environ.get("FAKE_CHECKER_EXIT", "0")))
'''


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json(path: pathlib.Path, value) -> None:
    _write(path, json.dumps(value))


def make_bench(tmp_path: pathlib.Path, *, flag: bool | None = True, status: str = "max_turns_reached",
               summary: dict | None = None) -> types.SimpleNamespace:
    """An image-shaped /workspace: entrypoint, helpers, fake upstream code, config and one dump."""
    work = tmp_path / "workspace"
    work.mkdir(parents=True)
    for source in (COWORK / "container" / "main_ouroboros.py", COWORK / "eval_attempt.py", COWORK / "official_receipt.py"):
        shutil.copy2(source, work / source.name)
    for package in ("utils", "utils/evaluation", "utils/data_structures", "utils/general"):
        _write(work / package / "__init__.py", "")
    _write(work / "utils/data_structures/task_config.py", FAKE_TASK_CONFIG)
    _write(work / "utils/evaluation/evaluator.py", FAKE_EVALUATOR)
    _write(work / "utils/general/helper.py", FAKE_HELPER)
    _json(work / "scripts/eval_config_strands.json", {"dump_path": "./dumps/"})
    task = work / "tasks/finalpool" / TASK
    _write(task / "evaluation/main.py", FAKE_CHECKER)
    (task / "groundtruth_workspace").mkdir(parents=True)
    config = {"model": MODEL, "settings": {}, "proxy_port": 8096, "server_port": 8765,
              "task_timeout_sec": 3600, "truncation_reason_codes": []}
    if flag is not None:
        config[attempts.FLAG] = flag
    _json(work / "configs/ouroboros_bench.json", config)
    dump = work / "dumps" / DUMP_DIR / f"SingleUserTurn-{TASK}"
    (dump / "workspace").mkdir(parents=True)
    saved = {"task_dir": TASK, "task_root": str(dump), "log_file": str(dump / "traj_log.json"),
             "agent_workspace": str(dump / "workspace"), "launch_time": SAVED_LAUNCH,
             "single_turn_mode": True, "cn_mode": False,
             "evaluation": {"groundtruth_workspace": None, "evaluation_command": None}}
    _json(dump / "traj_log.json", {"config": saved, "status": status,
                                   "start_time": "2026-09-24T09:15:00", "end_time": "2026-09-24T10:15:00"})
    _json(dump / "ouroboros_summary.json", summary or {
        "task": TASK, "bench_status": status, "reason_code": "deadline_local", "truncated": True,
        "infra_failed": False, "task_submission_started": True, "model_activity_observed": True})
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update({"OUROBOROS_COWORK_CONFIG": str(work / "configs/ouroboros_bench.json"),
                "PYTHON_BIN": sys.executable, attempts.AGENT_STATE_ENV: "absent",
                "FAKE_EVALUATOR_CALLS": str(tmp_path / "evaluator_calls"),
                "FAKE_CHECKER_ARGV": str(tmp_path / "checker_argv.json"),
                "FAKE_CHECKER_PIDS": str(tmp_path / "checker_pids.json"),
                "FAKE_GRANDCHILD_SLEEP": "30"})
    return types.SimpleNamespace(work=work, dump=dump, env=env, tmp=tmp_path, task_log=tmp_path / "task.log")


def run_eval(bench, **extra_env) -> subprocess.CompletedProcess:
    """One runner eval exec: stdout and stderr appended to the shared TASK_LOG."""
    with bench.task_log.open("a", encoding="utf-8") as log:
        return subprocess.run([sys.executable, "-u", "main_ouroboros.py", "--task_dir", TASK, "--phase", "eval"],
                              cwd=bench.work, env={**bench.env, **extra_env}, stdout=log,
                              stderr=subprocess.STDOUT, timeout=120)


def runner_grep(text: str) -> tuple[str, str]:
    """run_parallel.sh: `grep -q "Status: success"`, then `Pass:.*True` before `Pass:.*False`."""
    status = "success" if "Status: success" in text else "failed" if "Status: failed" in text else "unknown"
    verdict = "null"
    if status == "success":
        verdict = "True" if re.search(r"Pass:.*True", text) else "False" if re.search(r"Pass:.*False", text) else "null"
    return status, verdict


def evaluator_calls(bench) -> int:
    path = bench.tmp / "evaluator_calls"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def claim_of(dump: pathlib.Path) -> dict:
    return json.loads((dump / attempts.CLAIM_NAME).read_text(encoding="utf-8"))


def dead(pid: int) -> bool:
    """Gone or a zombie (orphans of a non-reaping container init stay zombies)."""
    if pathlib.Path("/proc/self").exists():
        try:
            text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return True
        return text[text.rfind(")") + 2:].split()[0] in {"Z", "X"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


# --------------------------------------------------------------------------- end to end


@pytest.mark.serial
def test_diagnostic_output_never_reaches_runner_log_and_command_matches_pinned_evaluator(tmp_path):
    bench = make_bench(tmp_path)
    (bench.task_log).write_text("====== Status: max_turns_reached ======\n", encoding="utf-8")
    log_before = (bench.dump / "traj_log.json").read_bytes()
    assert run_eval(bench).returncode == 1
    text = bench.task_log.read_text(encoding="utf-8")
    # The whole diagnostic path, including import-time output, the checker and its
    # grandchild, stays out of the combined runner log the summary grep reads.
    assert runner_grep(text) == ("unknown", "null")
    assert [line for line in text.splitlines() if "Pass:" in line] == ["Pass:    None"]
    assert "Status: success" not in text
    attempt = claim_of(bench.dump)["attempt_id"]
    diagnostic_log = (bench.dump / f"{attempts.DIAGNOSTIC_LOG_PREFIX}{attempt}.log").read_text(encoding="utf-8")
    for marker in ("import-time marker", "Pass: True from grandchild", "====== Status: success ======"):
        assert marker in diagnostic_log
    receipt = attempts.read_receipt(bench.dump, attempt, kind="diagnostic")
    assert receipt["outcome"] == "checks_passed" and receipt["custody"] == "group_dead"
    assert receipt["official_verdict_unchanged"] is True and "pass" not in receipt
    assert all(dead(pid) for pid in json.loads((tmp_path / "checker_pids.json").read_text()))
    # Same saved config, Evaluation.build fallback and two launch_time tokens as upstream.
    log_file, workspace = str(bench.dump / "traj_log.json"), str(bench.dump / "workspace")
    groundtruth = f"tasks/finalpool/{TASK}/groundtruth_workspace"
    expected = (f"{sys.executable} -m tasks.finalpool.{TASK}.evaluation.main --res_log_file {log_file} "
                f"--agent_workspace {workspace} --groundtruth_workspace {groundtruth} "
                f"--launch_time \"2026-09-24 09:15:00\"")
    assert receipt["command"] == expected
    assert receipt["command_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert json.loads((tmp_path / "checker_argv.json").read_text()) == [
        "--res_log_file", log_file, "--agent_workspace", workspace,
        "--groundtruth_workspace", groundtruth, "--launch_time", "2026-09-24 09:15:00"]
    # One official effect; its file, the status log and the ledger verdict stay untouched.
    assert evaluator_calls(bench) == 1
    official = attempts.read_receipt(bench.dump, attempt)
    assert official["returned"]["pass"] is None
    assert attempts.file_facts(bench.dump / "eval_res.json")["sha256"] == official["result_file"]["sha256"]
    assert (bench.dump / "traj_log.json").read_bytes() == log_before
    row = current_ledger_row(TASK, bench.dump, {"status": "unknown"})
    assert (row["status"], row["official_eval_status"]) == ("agent_failed", "declined")
    assert "checks_passed" not in json.dumps(row) and not any("diagnostic" in key for key in row["details"])
    # Reentry replays the official lines only: no evaluator call and no second diagnostic.
    diagnostic_receipt = (bench.dump / f"{attempts.DIAGNOSTIC_RECEIPT_PREFIX}{attempt}.json").read_bytes()
    assert run_eval(bench).returncode == 1
    assert evaluator_calls(bench) == 1
    assert bench.task_log.read_text(encoding="utf-8").count("Pass:    None") == 2
    assert (bench.dump / f"{attempts.DIAGNOSTIC_RECEIPT_PREFIX}{attempt}.json").read_bytes() == diagnostic_receipt


@pytest.mark.serial
@pytest.mark.parametrize("flag", [None, False, True])
def test_reentry_replays_exact_official_lines_without_second_effect(tmp_path, flag):
    bench = make_bench(tmp_path, flag=flag, status="success")
    first = run_eval(bench, FAKE_OFFICIAL_PASS="false")
    result = bench.dump / "eval_res.json"
    before = result.read_bytes()
    lines = bench.task_log.read_text(encoding="utf-8").splitlines()
    second = run_eval(bench, FAKE_OFFICIAL_PASS="true")
    assert first.returncode == second.returncode == 1
    replayed = bench.task_log.read_text(encoding="utf-8").splitlines()[len(lines):]
    assert [line for line in replayed if line.startswith(("Pass:", "Details:"))] == \
        [line for line in lines if line.startswith(("Pass:", "Details:"))] == ["Pass:    False", "Details: fixture verdict"]
    assert evaluator_calls(bench) == 1
    assert result.read_bytes() == before
    assert not (bench.dump / attempts.DIAGNOSTIC_CLAIM_NAME).exists()


@pytest.mark.serial
def test_concurrent_entries_make_exactly_one_official_effect(tmp_path):
    bench = make_bench(tmp_path, flag=False)
    env = {**bench.env, "FAKE_EVALUATOR_DELAY": "2"}
    command = [sys.executable, "-u", "main_ouroboros.py", "--task_dir", TASK, "--phase", "eval"]
    procs = [subprocess.Popen(command, cwd=bench.work, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outputs = [proc.communicate(timeout=60)[0] for proc in procs]
    assert evaluator_calls(bench) == 1
    assert sum("Pass:    None" in text for text in outputs) >= 1
    refused = [text for text in outputs if "Pass:" not in text]
    assert all("official evaluation not repeated: unfinished_claim" in text for text in refused)
    receipt = attempts.read_receipt(bench.dump, claim_of(bench.dump)["attempt_id"])
    assert receipt["official_run"] is True


@pytest.mark.serial
def test_crash_after_claim_blocks_every_automatic_reentry(tmp_path):
    bench = make_bench(tmp_path)
    assert run_eval(bench, FAKE_EVALUATOR_CRASH="1").returncode == 9
    attempt = claim_of(bench.dump)["attempt_id"]
    assert attempts.read_receipt(bench.dump, attempt) is None
    for _ in range(2):
        assert run_eval(bench).returncode == 1
    text = bench.task_log.read_text(encoding="utf-8")
    assert evaluator_calls(bench) == 1
    assert text.count(f"official evaluation not repeated: unfinished_claim; attempt {attempt}") == 2
    assert "Pass:" not in text
    assert not (bench.dump / attempts.DIAGNOSTIC_CLAIM_NAME).exists()
    facts = attempts.attempt_facts(bench.dump)
    assert facts["state"] == "claimed" and "official_run" not in facts


@pytest.mark.serial
def test_unclaimed_existing_result_is_preserved_never_overwritten_diagnosed_or_scored(tmp_path):
    bench = make_bench(tmp_path, status="success", summary={
        "task": TASK, "bench_status": "success", "infra_failed": False, "task_submission_started": True})
    forged = bench.dump / "eval_res.json"
    forged.write_text('{"pass": true, "details": "written before any claim"}', encoding="utf-8")
    before = forged.read_bytes()
    for _ in range(2):
        assert run_eval(bench).returncode == 1
    text = bench.task_log.read_text(encoding="utf-8")
    assert evaluator_calls(bench) == 0
    assert forged.read_bytes() == before
    assert text.count("official evaluator not run: unclaimed_prior_result") == 2 and "Pass:" not in text
    assert not (bench.dump / attempts.DIAGNOSTIC_CLAIM_NAME).exists()
    row = current_ledger_row(TASK, bench.dump, {"status": "success"})
    assert (row["status"], row["reason_code"], row["official_eval_status"]) == (
        "infra_failed", "official_eval_not_run", "not_run")
    assert row["details"]["official_attempt"]["cause"] == "unclaimed_prior_result"
    assert row["details"]["official_receipt"]["pass"] is True  # disclosed, never scored


@pytest.mark.parametrize("terminal", [False, True])
def test_unfinished_or_exceptional_claim_cannot_score_a_prior_pass(tmp_path, terminal):
    bench = make_bench(tmp_path, status="success", summary={
        "task": TASK, "bench_status": "success", "infra_failed": False, "task_submission_started": True})
    forged = bench.dump / "eval_res.json"
    forged.write_text('{"pass": true}', encoding="utf-8")
    attempt_id = "c" * 32
    attempts.publish_exclusive(bench.dump / attempts.CLAIM_NAME,
                               {"kind": "official", "attempt_id": attempt_id, "bindings": {"b": 1}})
    if terminal:
        attempts.publish_exclusive(bench.dump / f"{attempts.RECEIPT_PREFIX}{attempt_id}.json", {
            "kind": "official", "attempt_id": attempt_id, "official_run": True,
            "raised": "KeyboardInterrupt", "returned": None,
            "result_file": attempts.file_facts(forged),
        })
    row = current_ledger_row(TASK, bench.dump, {"status": "success"})
    assert (row["status"], row["official_eval_status"], row["details"]["official_receipt"]["pass"]) == (
        "infra_failed", "unknown", True)
    attempt = row["details"]["official_attempt"]
    if terminal:
        assert attempt["file_matches_returned"] is False
    else:
        assert attempt["state"] == "claimed"


@pytest.mark.serial
def test_changed_binding_refuses_reentry_without_touching_the_attempt(tmp_path):
    bench = make_bench(tmp_path, flag=False)
    assert run_eval(bench).returncode == 1
    attempt = claim_of(bench.dump)["attempt_id"]
    receipt_path = bench.dump / f"{attempts.RECEIPT_PREFIX}{attempt}.json"
    receipt = receipt_path.read_bytes()
    log = bench.dump / "traj_log.json"
    log.write_text(log.read_text(encoding="utf-8").replace("max_turns_reached", "success"), encoding="utf-8")
    assert run_eval(bench).returncode == 1
    assert evaluator_calls(bench) == 1
    assert f"not repeated: binding_mismatch; attempt {attempt}" in bench.task_log.read_text(encoding="utf-8")
    assert receipt_path.read_bytes() == receipt


@pytest.mark.serial
def test_diagnostic_traceback_stays_in_its_own_log(tmp_path):
    bench = make_bench(tmp_path)
    # Raises only inside the diagnostic child; the official evaluator is unaffected.
    assert run_eval(bench, FAKE_TASK_CONFIG_RAISES="1").returncode == 1
    attempt = claim_of(bench.dump)["attempt_id"]
    text = bench.task_log.read_text(encoding="utf-8")
    assert "Traceback" not in text and runner_grep(text) == ("unknown", "null")
    log = (bench.dump / f"{attempts.DIAGNOSTIC_LOG_PREFIX}{attempt}.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "Pass: True inside a traceback" in log
    receipt = attempts.read_receipt(bench.dump, attempt, kind="diagnostic")
    assert (receipt["outcome"], receipt["reason"]) == ("diagnostic_error", "exception:RuntimeError")


# --------------------------------------------------------------------------- in process


def settled_attempt(tmp_path: pathlib.Path, monkeypatch, *, agent_state="absent", **bench_kwargs):
    """A flag-off official run, leaving the claim and terminal receipt for direct diagnostic calls."""
    bench = make_bench(tmp_path, flag=False, **bench_kwargs)
    bench.env[attempts.AGENT_STATE_ENV] = agent_state
    run_eval(bench)
    monkeypatch.chdir(bench.work)
    for key, value in bench.env.items():
        if key.startswith(("FAKE_", "PYTHON_BIN")):
            monkeypatch.setenv(key, value)
    return bench, claim_of(bench.dump)


@pytest.mark.serial
def test_timeout_ends_the_owned_group_proves_death_and_prints_nothing(tmp_path, monkeypatch, capfd):
    bench, claim = settled_attempt(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CHECKER_SLEEP", "60")
    started = time.monotonic()
    receipt = attempts.run_residual_diagnostic(bench.dump, claim["attempt_id"], cap_sec=2)
    assert time.monotonic() - started < 30
    assert (receipt["outcome"], receipt["custody"], receipt["report_stage"]) == ("timeout", "group_dead", "checker_started")
    assert receipt["eligible"] is True
    pids = json.loads((tmp_path / "checker_pids.json").read_text())
    assert all(dead(pid) for pid in pids)
    assert capfd.readouterr() == ("", "")
    # The claim is exclusive: a second call spawns nothing and writes nothing.
    assert attempts.run_residual_diagnostic(bench.dump, claim["attempt_id"], cap_sec=2) is None
    assert attempts.diagnostic_facts(bench.dump)["outcome"] == "timeout"


@pytest.mark.serial
def test_unconfirmed_group_death_is_unknown_not_timeout(tmp_path, monkeypatch):
    bench, claim = settled_attempt(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CHECKER_SLEEP", "60")
    monkeypatch.setattr(attempts, "TEARDOWN_GRACE_SEC", 0.2)
    real_alive = attempts._group_alive
    monkeypatch.setattr(attempts, "_group_alive", lambda pgid: True)
    try:
        receipt = attempts.run_residual_diagnostic(bench.dump, claim["attempt_id"], cap_sec=1)
    finally:
        monkeypatch.setattr(attempts, "_group_alive", real_alive)
    assert (receipt["outcome"], receipt["custody"]) == ("unknown", "unconfirmed")
    pids = json.loads((tmp_path / "checker_pids.json").read_text())
    deadline = time.monotonic() + 10
    while not all(dead(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert all(dead(pid) for pid in pids)


@pytest.mark.serial
@pytest.mark.parametrize(("code", "outcome"), [(0, "checks_passed"), (3, "checks_failed")])
def test_checker_exit_maps_to_audit_outcome_only(tmp_path, monkeypatch, code, outcome):
    bench, claim = settled_attempt(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CHECKER_EXIT", str(code))
    monkeypatch.setenv("FAKE_GRANDCHILD_SLEEP", "0")
    receipt = attempts.run_residual_diagnostic(bench.dump, claim["attempt_id"], cap_sec=60)
    assert (receipt["outcome"], receipt["checker_exit_code"]) == (outcome, code)
    assert attempts.read_receipt(bench.dump, claim["attempt_id"])["returned"]["pass"] is None


@pytest.mark.serial
@pytest.mark.parametrize(("summary_patch", "agent_state", "reason"), [
    ({"reason_code": "round_limit"}, "absent", ""),
    ({"reason_code": "budget_exhausted"}, "absent", ""),
    ({"reason_code": "wall_clock_timeout", "bench_status": "failed"}, "absent", ""),
    ({"reason_code": "owner_requested_finalization"}, "absent", "stop_reason_not_eligible"),
    ({"reason_code": "children_unabsorbed"}, "absent", "stop_reason_not_eligible"),
    ({"reason_code": "finalization_grace"}, "absent", "stop_reason_not_eligible"),
    ({"reason_code": "llm_api_error", "infra_failed": True}, "absent", "infrastructure_outcome"),
    ({"model_activity_observed": False}, "absent", "no_agent_activity"),
    ({"model_activity_observed": None}, "absent", "agent_activity_unknown"),
    ({"task_submission_started": False}, "absent", "no_task_submission"),
    ({"task": "another_task"}, "absent", "adapter_summary_not_linked"),
    ({}, "present", "agent_termination_unproven"),
    ({}, "unknown", "agent_termination_unproven"),
])
def test_eligibility_is_narrow_and_distinguishes_unknown(tmp_path, monkeypatch, summary_patch, agent_state, reason):
    status = summary_patch.get("bench_status", "max_turns_reached")
    summary = {"task": TASK, "bench_status": status, "reason_code": "deadline_local", "truncated": True,
               "infra_failed": False, "task_submission_started": True, "model_activity_observed": True,
               **summary_patch}
    bench, claim = settled_attempt(tmp_path, monkeypatch, agent_state=agent_state, status=status, summary=summary)
    receipt = attempts.read_receipt(bench.dump, claim["attempt_id"])
    assert attempts.eligibility(bench.dump, claim, receipt)[0] == reason


@pytest.mark.serial
@pytest.mark.parametrize("change", ["boolean_verdict", "result_rewritten", "log_rewritten", "no_receipt"])
def test_only_a_proven_same_attempt_status_gate_decline_is_eligible(tmp_path, monkeypatch, change):
    status = "success" if change == "boolean_verdict" else "max_turns_reached"
    bench, claim = settled_attempt(tmp_path, monkeypatch, status=status, summary={
        "task": TASK, "bench_status": status, "reason_code": "deadline_local", "infra_failed": False,
        "task_submission_started": True, "model_activity_observed": True})
    receipt = attempts.read_receipt(bench.dump, claim["attempt_id"])
    if change == "result_rewritten":
        (bench.dump / "eval_res.json").write_text(json.dumps(receipt["returned"], indent=1), encoding="utf-8")
    elif change == "log_rewritten":
        log = bench.dump / "traj_log.json"
        log.write_text(log.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif change == "no_receipt":
        receipt = None
    assert attempts.eligibility(bench.dump, claim, receipt)[0] == {
        "boolean_verdict": "not_same_attempt_status_gate_decline",
        "result_rewritten": "not_same_attempt_status_gate_decline",
        "log_rewritten": "log_changed_or_unreadable", "no_receipt": "official_not_completed"}[change]


def install_fake_upstream(monkeypatch) -> None:
    for name in ("utils", "utils.data_structures"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("utils.data_structures.task_config")
    exec(compile(FAKE_TASK_CONFIG, "fake_task_config.py", "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, "utils.data_structures.task_config", module)


@pytest.mark.parametrize(("saved_eval", "expected"), [
    ({"groundtruth_workspace": "/saved/gt", "evaluation_command": "saved-cmd"}, ("saved-cmd", "/saved/gt")),
    ({"groundtruth_workspace": None, "evaluation_command": "saved-cmd"}, ("saved-cmd", "BUILT")),
    ({"groundtruth_workspace": "/saved/gt", "evaluation_command": None}, ("BUILT_CMD", "/saved/gt")),
    ({"groundtruth_workspace": None, "evaluation_command": None}, ("BUILT_CMD", "BUILT")),
])
def test_command_uses_saved_config_with_upstream_build_fallback(tmp_path, monkeypatch, saved_eval, expected):
    bench = make_bench(tmp_path)
    monkeypatch.chdir(bench.work)
    monkeypatch.setenv("PYTHON_BIN", "py")
    install_fake_upstream(monkeypatch)
    log = bench.dump / "traj_log.json"
    record = json.loads(log.read_text())
    record["config"]["evaluation"] = saved_eval
    _json(log, record)
    if expected == ("saved-cmd", "/saved/gt"):
        monkeypatch.setenv("FAKE_BUILD_FORBIDDEN", "1")
    command, groundtruth = expected
    command = command.replace("BUILT_CMD", f"py -m tasks.finalpool.{TASK}.evaluation.main")
    groundtruth = groundtruth.replace("BUILT", f"tasks/finalpool/{TASK}/groundtruth_workspace")
    assert attempts.evaluation_command(bench.dump, TASK) == (
        f"{command} --res_log_file {log} --agent_workspace {bench.dump / 'workspace'} "
        f"--groundtruth_workspace {groundtruth} --launch_time \"2026-09-24 09:15:00\"")


@pytest.mark.parametrize("change", ["no_command", "not_single_turn", "other_task"])
def test_unreconstructable_command_is_unavailable_and_never_touches_the_result(tmp_path, monkeypatch, change):
    bench = make_bench(tmp_path)
    monkeypatch.chdir(bench.work)
    install_fake_upstream(monkeypatch)
    result = bench.dump / "eval_res.json"
    result.write_text('{"pass": null}', encoding="utf-8")
    log = bench.dump / "traj_log.json"
    record = json.loads(log.read_text())
    if change == "no_command":
        shutil.rmtree(bench.work / "tasks/finalpool" / TASK / "evaluation")
    elif change == "not_single_turn":
        # Without the prefix, TaskConfig.__post_init__ would delete the official result.
        record["config"]["single_turn_mode"] = False
    else:
        record["config"]["task_dir"] = "other"
    _json(log, record)
    expected = {"no_command": "no_eval_command", "not_single_turn": "saved_config_unsupported",
                "other_task": "saved_config_not_linked"}[change]
    with pytest.raises(attempts.Unavailable, match=expected):
        attempts.evaluation_command(bench.dump, TASK)
    assert result.read_text(encoding="utf-8") == '{"pass": null}'


def test_receipt_publication_failure_keeps_the_verdict_and_blocks_reentry(tmp_path, monkeypatch, capsys):
    root = tmp_path / "dump"
    root.mkdir()
    calls = []
    real = attempts.publish_exclusive

    def publish(path, record):
        if path.name.startswith(attempts.RECEIPT_PREFIX):
            raise OSError("disk full")
        return real(path, record)

    monkeypatch.setattr(attempts, "publish_exclusive", publish)
    bindings = {"task_dir": TASK, "log": {"sha256": "a"}}
    record, ran = attempts.official_attempt(root, bindings, {}, lambda: "log", lambda log: calls.append(log) or {"pass": True})
    assert ran and attempts.official_verdict(record) == {"pass": True}
    assert "receipt not persisted: OSError" in capsys.readouterr().err
    record, ran = attempts.official_attempt(root, bindings, {}, lambda: "log", lambda log: calls.append(log))
    assert (ran, record["cause"], calls) == (False, "unfinished_claim", ["log"])


@pytest.mark.parametrize(("prepare_error", "cause", "propagates"), [
    (attempts.NotRun("log_path_mismatch"), "log_path_mismatch", False),
    (FileNotFoundError("task.md"), "preparation_error:FileNotFoundError", True),
])
def test_preparation_failure_settles_unevaluated_and_is_never_retried(tmp_path, prepare_error, cause, propagates):
    root = tmp_path / "dump"
    calls = []

    def prepare():
        raise prepare_error

    if propagates:
        with pytest.raises(type(prepare_error)):
            attempts.official_attempt(root, {"b": 1}, {}, prepare, calls.append)
    else:
        record, _ = attempts.official_attempt(root, {"b": 1}, {}, prepare, calls.append)
        assert record["official_run"] is False
    facts = attempts.attempt_facts(root)
    assert (facts["official_run"], facts["cause"], calls) == (False, cause, [])
    record, ran = attempts.official_attempt(root, {"b": 1}, {}, lambda: "log", calls.append)
    assert (ran, record["cause"], calls) == (False, cause, [])
    assert "Pass:" not in attempts.refusal_line(record) and "Status:" not in attempts.refusal_line(record)


def test_evaluator_exception_is_recorded_and_propagates_like_the_unwrapped_call(tmp_path):
    root = tmp_path / "dump"

    def evaluate(_log):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        attempts.official_attempt(root, {"b": 1}, {}, lambda: "log", evaluate)
    record, ran = attempts.official_attempt(root, {"b": 1}, {}, lambda: "log", evaluate)
    assert (ran, record["official_run"], record["raised"]) == (False, True, "KeyboardInterrupt")
    assert "(KeyboardInterrupt)" in attempts.refusal_line(record)


def test_foreign_receipt_names_and_invalid_claims_never_count(tmp_path):
    root = tmp_path / "dump"
    root.mkdir()
    forged = {"kind": "official", "attempt_id": "f" * 32, "official_run": True, "returned": {"pass": True}}
    _json(root / f"{attempts.RECEIPT_PREFIX}{'f' * 32}.json", forged)
    record, ran = attempts.official_attempt(root, {"b": 1}, {}, lambda: "log", lambda log: {"pass": False})
    assert ran and record["attempt_id"] != "f" * 32 and record["returned"] == {"pass": False}
    (root / attempts.CLAIM_NAME).write_text("{", encoding="utf-8")
    record, ran = attempts.official_attempt(root, {"b": 1}, {}, lambda: "log", lambda log: pytest.fail("rerun"))
    assert (ran, record["cause"]) == (False, "claim_unreadable")
    assert attempts.attempt_facts(root) == {"state": "invalid"}


# --------------------------------------------------------------------------- launcher and image


def test_flag_is_default_off_and_part_of_the_applied_config():
    args = launcher.parse_args([])
    assert launcher.bench_config(args, {})[attempts.FLAG] is False
    args = launcher.parse_args(["--diagnostic-eval-on-truncation"])
    assert launcher.bench_config(args, {})[attempts.FLAG] is True


@pytest.mark.parametrize(("previous", "current", "accepted"), [
    ("missing", False, True), (False, False, True), (True, True, True),
    ("missing", True, False), (False, True, False), (True, False, False), ("false", False, False),
])
def test_resume_normalizes_only_a_missing_flag_to_false(tmp_path, previous, current, accepted):
    bench = tmp_path / "bench"
    _json(bench / "tasks/finalpool/one/task_config.json", {})
    root = tmp_path / "previous"
    base = {"model": "m", "settings": {}}
    recorded = dict(base) if previous == "missing" else {**base, attempts.FLAG: previous}
    _json(root / "run_manifest.json", {"harness": {"applied_config": recorded, "bench": {"head": "b"},
                                                   "image_id": "sha256:1"}, "source": {"head": "s"},
                                       "requested_task_ids": ["one"]})
    (root / "result_index.jsonl").write_text("", encoding="utf-8")
    args = argparse.Namespace(task_file="", task=[], resume_from=[str(root)], bench_commit="b", image_id="sha256:1")
    config = {**base, attempts.FLAG: current}
    if accepted:
        assert launcher.select_tasks(bench, args, config, "s") == ["one"]
    else:
        with pytest.raises(ValueError, match="configuration"):
            launcher.select_tasks(bench, args, config, "s")


def test_image_carries_stdlib_helpers_beside_the_entrypoint():
    dockerfile = (COWORK / "container" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY main_ouroboros.py eval_attempt.py official_receipt.py /workspace/" in dockerfile
    assert [path.name for path in launcher.CONTAINER_HELPERS] == ["eval_attempt.py", "official_receipt.py"]
    for helper in launcher.CONTAINER_HELPERS:
        for node in ast.walk(ast.parse(helper.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                for name in names:
                    root = name.split(".")[0]
                    assert root in sys.stdlib_module_names or name in {
                        "devtools.benchmarks.cowork_bench.official_receipt", "official_receipt",
                        "utils.data_structures.task_config"}, name
