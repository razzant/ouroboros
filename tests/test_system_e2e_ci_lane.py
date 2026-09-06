"""The scheduled keyless `system-e2e-mock` CI job (owner 9A).

`tests/system_e2e/` is gated three ways — the `integration` and `serial`
markers plus the `OUROBOROS_E2E_DEEP` env var — precisely so that no existing
CI pytest pass can reach it. That is what makes a suite nobody executes: the
gates work, and then nothing opens them. This job is the one thing that does,
and the plan's §8 pull-request lane was replaced by a daily schedule (owner
9A) because the scenarios spawn real isolated servers and cost minutes.

Two properties are load-bearing enough to pin. The job must stay OFF push and
pull_request, or the lane it was made cheap for becomes the slowest thing in
every PR. And the daily schedule must not wake the PAID provider lane: three
of `integration-test`'s branch conditions match the default branch ref a
scheduled run carries, so without an explicit event guard adding `schedule:`
to this workflow would spend real provider credit every night.
"""

from __future__ import annotations

import pathlib
import re

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
JOB = "system-e2e-mock"
MIN_TIMEOUT_MINUTES = 30


def _workflow() -> dict:
    return yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # YAML 1.1 reads a bare `on:` key as the boolean True; PyYAML follows it.
    return workflow.get("on") or workflow.get(True) or {}


def _job_text(job: str) -> str:
    """The job's raw block — what a `secrets.` reference would have to be in."""
    ci = CI_PATH.read_text(encoding="utf-8")
    block = re.search(
        rf"^  {re.escape(job)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:$|\Z)",
        ci, re.MULTILINE | re.DOTALL,
    )
    assert block, f"ci.yml has no `{job}:` job"
    return block.group(1)


MOCK_CRON = "37 4 * * *"


def test_the_workflow_carries_daily_off_peak_schedules_each_owned_by_one_job():
    workflow = _workflow()
    schedule = _triggers(workflow).get("schedule") or []
    crons = [str(entry["cron"]) for entry in schedule]
    # Two crons: this keyless lane and the paid `e2e-live` stand
    # (tests/test_e2e_live_ci_lane.py). A cron nobody binds to is a second
    # nightly wake-up of every job gated on the bare event name.
    assert crons == [MOCK_CRON, "17 3 * * *"], schedule
    for entry in schedule:
        minute, hour, day, month, weekday = str(entry["cron"]).split()
        assert (day, month, weekday) == ("*", "*", "*"), entry
        assert minute.isdigit() and hour.isdigit(), "one fixed daily time, not a range"
        # On the hour is when everyone else's cron fires and GitHub's queue is
        # deepest; an off-peak minute is the documented way to avoid the backlog.
        assert int(minute) != 0, entry
    # Every job that fires on `schedule` names ITS cron string, so neither cron
    # wakes the other lane: a bare `github.event_name == 'schedule'` would.
    for name, job in workflow["jobs"].items():
        condition = " ".join(str(job.get("if", "")).split())
        if "github.event_name == 'schedule'" not in condition:
            continue  # `!= 'schedule'` guards (integration-test) keep a lane OFF both crons
        assert "github.event.schedule ==" in condition, (name, condition)
        assert "github.event_name == 'schedule' ||" not in condition, (name, condition)


def test_the_scheduled_lane_never_runs_on_a_push_or_a_pull_request():
    job = _workflow()["jobs"][JOB]
    condition = " ".join(str(job["if"]).split())
    assert condition == (
        f"(github.event_name == 'schedule' && github.event.schedule == '{MOCK_CRON}')"
        " || github.event_name == 'workflow_dispatch'"
        " || startsWith(github.ref, 'refs/tags/v')"
    )  # a release tag joins the lane to the release bar (batch №13 item 4); push/PR never, condition
    assert job["runs-on"] == "ubuntu-latest"
    # The budget must clear the suite, not merely exist. `> 0` accepted
    # `timeout-minutes: 1`, which cancels the job mid-scenario and reports the
    # same red as a real failure — the one thing a nightly lane nobody watches
    # must not do. The floor is the measured walltime (~17 minutes for
    # tests/system_e2e/ on the mock lane) plus room for a slow runner and for
    # the scenarios a later wave adds; production sits at 40.
    assert int(job["timeout-minutes"]) >= MIN_TIMEOUT_MINUTES


def test_the_scheduled_lane_runs_the_keyless_suite_on_a_throwaway_root():
    steps = _workflow()["jobs"][JOB]["steps"]
    assert [step.get("uses") for step in steps][:2] == [
        "actions/checkout@v4", "./.github/actions/setup-python-env",
    ]
    run_step = next(step for step in steps if "run" in step)
    assert run_step["run"].strip() == (
        'python -m pytest tests/system_e2e/ -o addopts="" -q'
    )
    env = run_step["env"]
    assert env["OUROBOROS_E2E_DEEP"] == "mock"
    # All four roots, all under the runner's temp: a scenario server that
    # escaped its isolation could otherwise write into the checkout.
    roots = ["OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR",
             "OUROBOROS_SETTINGS_PATH"]
    assert all("runner.temp" in str(env[name]) for name in roots), env


def test_the_scheduled_lane_asks_for_no_secret():
    """Keyless by construction: a job gets a secret only by naming it."""
    assert "secrets." not in _job_text(JOB), _job_text(JOB)


def test_the_daily_schedule_does_not_wake_the_paid_provider_lane():
    """`integration-test` fires on refs/heads/main|ouroboros|ouroboros-stable —
    one of which is whatever default branch a scheduled run reports. Without
    this guard the new cron would buy provider credit every night."""
    condition = " ".join(str(_workflow()["jobs"]["integration-test"]["if"]).split())
    assert condition.startswith("github.event_name != 'schedule'"), condition
