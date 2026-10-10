"""The keyless `system-e2e-mock` CI job (owner 9A).

`tests/system_e2e/` is gated three ways — the `integration` and `serial`
markers plus the `OUROBOROS_E2E_DEEP` env var — precisely so that no existing
CI pytest pass can reach it. That is what makes a suite nobody executes: the
gates work, and then nothing opens them. This job is the one thing that does,
on manual dispatch and on release tags; the plan's §8 pull-request lane gave
way to a daily schedule (owner 9A), and the schedule itself was removed on
2026-10-05 because nobody read the nightly results (owner).

Two properties are load-bearing enough to pin. The job must stay OFF push and
pull_request, or the lane it was made cheap for becomes the slowest thing in
every PR. And no schedule may wake the PAID provider lane: a scheduled run
carries the default branch in its ref, so `integration-test` leads with an
explicit event guard that holds whatever ref conditions follow it.
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


def test_pull_requests_and_ouroboros_pushes_share_one_full_browser_lane():
    """The PR lane is no longer the narrow Publish proof: it is the whole marker lane.

    `ci.yml` matches pull requests, manual runs and tags and delegates to the
    reusable lane; every `ouroboros` push reaches the SAME job through its own
    path-filter-free workflow. Nothing here may pull the costly system-e2e
    scenarios into a pull request.
    """
    shared_path = REPO_ROOT / ".github" / "workflows" / "ui-browser.yml"
    push_path = REPO_ROOT / ".github" / "workflows" / "ui-browser-push.yml"
    caller = _workflow()["jobs"]["ui-smoke"]
    assert " ".join(caller["if"].split()) == (
        "github.event_name == 'pull_request' || github.event_name == 'workflow_dispatch'"
        " || startsWith(github.ref, 'refs/tags/v')"
    )
    assert caller["uses"] == "./.github/workflows/ui-browser.yml"
    assert "steps" not in caller, "the browser steps belong to the shared lane"

    push = yaml.safe_load(push_path.read_text(encoding="utf-8"))
    assert set(_triggers(push)) == {"push", "workflow_dispatch"}
    assert _triggers(push)["push"] == {"branches": ["ouroboros"]}
    assert _triggers(push)["workflow_dispatch"]["inputs"]["diagnostic"]["options"] == ["full", "viewport", "inflight"]
    assert push["jobs"]["ui-smoke"]["uses"] == caller["uses"]

    shared = yaml.safe_load(shared_path.read_text(encoding="utf-8"))
    assert list(_triggers(shared)) == ["workflow_call"]
    # The complete lane runs in the shard job; `ui-smoke` is the aggregator that judges it.
    steps = {step.get("name"): step for step in shared["jobs"]["ui-shard"]["steps"] if step.get("name")}
    full = steps["Run complete host UI lane with collection and availability guards"]
    assert full["if"] == "${{ !cancelled() && steps.install_browsers.outcome == 'success' }}"
    assert "python -m pytest tests/ -m ui_browser --require-ui-browser -vv --tb=short" in full["run"]
    assert full["env"]["OUROBOROS_RUN_UI_SMOKE"] == "1"
    assert full["env"]["OUROBOROS_EXPECT_BROWSER_ENGINES"] == "chromium,webkit"
    assert steps["Run browser tools Chromium/WebKit smoke"]["if"] == (
        "${{ !cancelled() && steps.install_browsers.outcome == 'success'"
        " && (github.event_name == 'workflow_dispatch' || startsWith(github.ref, 'refs/tags/v'))"
        " && matrix.shard == 1 }}")
    for text in (_job_text("ui-smoke"), shared_path.read_text(encoding="utf-8"),
                 push_path.read_text(encoding="utf-8")):
        assert "secrets." not in text
        assert "system_e2e" not in text and "OUROBOROS_E2E_DEEP" not in text


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


def test_the_workflow_carries_no_schedule_and_no_job_admits_one():
    workflow = _workflow()
    # Owner, 2026-10-05: nobody read the nightly results, so the workflow runs
    # nothing on a timer. A scheduled run would carry the default branch in its
    # ref; the remaining `!= 'schedule'` guards (integration-test) keep paid
    # lanes off one if a schedule ever returns.
    assert "schedule" not in _triggers(workflow), _triggers(workflow)
    for name, job in workflow["jobs"].items():
        condition = " ".join(str(job.get("if", "")).split())
        assert "github.event_name == 'schedule'" not in condition, (name, condition)


def test_the_lane_runs_only_on_a_dispatch_or_a_release_tag():
    job = _workflow()["jobs"][JOB]
    condition = " ".join(str(job["if"]).split())
    assert condition == (
        "github.event_name == 'workflow_dispatch' || startsWith(github.ref, 'refs/tags/v')"
    )  # a release tag joins the lane to the release bar (batch №13 item 4); push/PR never, condition
    assert job["runs-on"] == "ubuntu-latest"
    # The budget must clear the suite, not merely exist. `> 0` accepted
    # `timeout-minutes: 1`, which cancels the job mid-scenario and reports the
    # same red as a real failure — the one thing a nightly lane nobody watches
    # must not do. The floor is the measured walltime (~17 minutes for
    # tests/system_e2e/ on the mock lane) plus room for a slow runner and for
    # the scenarios a later wave adds; production sits at 40.
    assert int(job["timeout-minutes"]) >= MIN_TIMEOUT_MINUTES


def test_the_lane_runs_the_keyless_suite_on_a_throwaway_root():
    steps = _workflow()["jobs"][JOB]["steps"]
    assert [step.get("uses") for step in steps][:2] == [
        "actions/checkout@v4", "./.github/actions/setup-python-env",
    ]
    run_steps = [step for step in steps if "run" in step]
    assert len(run_steps) == 2
    expected = [("tests/system_e2e/", "OUROBOROS_E2E_DEEP"),
                ("tests/test_e2e_cancellation_scenarios.py", "OUROBOROS_E2E_CANCEL")]
    # All four roots, all under the runner's temp: a scenario server that
    # escaped its isolation could otherwise write into the checkout.
    roots = ["OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR",
             "OUROBOROS_SETTINGS_PATH"]
    for run_step, (target, lane) in zip(run_steps, expected):
        assert run_step["run"].strip() == (
            f'python -m pytest {target} -o addopts="" -o faulthandler_timeout=540 -q'
        )
        env = run_step["env"]
        assert env[lane] == "mock"
        assert env["PYTHONUNBUFFERED"] == "1"
        assert all("runner.temp" in str(env[name]) for name in roots), env


def test_the_lane_uploads_its_servers_traces_and_never_a_settings_file():
    """Owner, 2026-10-04: the traces are the most useful part of a CI run. The
    scenario servers write their journals under pytest tmp_path trees — the
    OUROBOROS_* roots of the run steps are what tests/conftest.py isolates FROM —
    and bare pytest (no OUROBOROS_TEST_TEMP_ROOT, no TMPDIR on a hosted runner)
    creates its session root `ouroboros-pytest-*` in the default temp directory,
    /tmp. The upload names exactly that root, journals and task results only."""
    jobs = _workflow()["jobs"]
    steps = jobs[JOB]["steps"]
    uploads = [step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
    assert len(uploads) == 1, uploads
    upload = uploads[0]
    assert steps.index(upload) == len(steps) - 1      # after both scenario passes
    assert upload["if"] == "always()"
    # Diagnostics only: a failed upload never reddens the job (the release bar needs it);
    # both scenario passes still decide it.
    assert upload.get("continue-on-error") is True
    passes = [step for step in steps if "python -m pytest" in str(step.get("run", ""))]
    assert len(passes) == 2 and not any("continue-on-error" in step for step in passes), passes
    paid = next(step for step in jobs["e2e-live"]["steps"]
                if str(step.get("uses", "")).startswith("actions/upload-artifact@"))
    assert upload["uses"] == paid["uses"]             # one pinned action for both lanes
    assert upload["with"]["name"] == "system-e2e-traces" and upload["with"]["retention-days"] == 30
    assert upload["with"]["if-no-files-found"] == "warn"
    lines = [line.strip() for line in str(upload["with"]["path"]).splitlines() if line.strip()]
    root = "/tmp/ouroboros-pytest-*/"
    includes = [line for line in lines if not line.startswith("!")]
    # A keyless server carries no benchmark sentinel, so its journals past 800 KB rotate into
    # data/archive/<prefix>_<ts>.jsonl (supervisor/state.py): the head of a long journal lives there.
    assert includes == [f"{root}**/data/logs/**", f"{root}**/data/task_results/**",
                        f"{root}**/data/archive/*.jsonl"], includes
    assert [line for line in lines if line.startswith("!")] == [f"!{root}**/settings.json"], lines
    # The root is the one conftest creates for a bare run, in the runner's default temp dir.
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert 'prefix="p" if _SAFE_TEMP_ROOT else "ouroboros-pytest-"' in conftest
    for step in steps:
        env = step.get("env") or {}
        assert "TMPDIR" not in env and "OUROBOROS_TEST_TEMP_ROOT" not in env, step


def test_the_lane_asks_for_no_secret():
    """Keyless by construction: a job gets a secret only by naming it."""
    assert "secrets." not in _job_text(JOB), _job_text(JOB)


def test_a_schedule_would_not_wake_the_paid_provider_lane():
    """A scheduled run would report the default branch in github.ref. The leading
    event guard keeps any schedule off the paid lane whatever ref conditions
    follow it; the push workflow that serves branch pushes has no schedule."""
    condition = " ".join(str(_workflow()["jobs"]["integration-test"]["if"]).split())
    assert condition.startswith("github.event_name != 'schedule'"), condition
    push = yaml.safe_load((CI_PATH.parent / "provider-canary-push.yml").read_text(encoding="utf-8"))
    assert list(push.get("on", push.get(True))) == ["push"]
