"""Manual/release CI admission and the retained job/secret boundaries."""

from __future__ import annotations

import json
import re
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def _value(expression, *, event, ref, base="", schedule="", cancelled=False, attempt=1, repository="razzant/ouroboros", e2e_live="false"):
    """Evaluate the workflow's small expression vocabulary against real event shapes."""
    expression = expression.strip()
    if expression.startswith("$" + "{{"):
        expression = expression[3:-2]
    expression = " ".join(expression.split()).replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", "not ", expression)
    github = SimpleNamespace(
        event_name=event, ref=ref, base_ref=base, run_id=77, run_attempt=attempt, repository=repository,
        event=SimpleNamespace(
            schedule=schedule, inputs=SimpleNamespace(e2e_live=e2e_live), before="previous-tip",
            pull_request=SimpleNamespace(number=42, base=SimpleNamespace(sha="pr-base")),
        ),
    )
    return eval(expression, {"__builtins__": {}}, {
        "github": github, "fromJSON": json.loads,
        "startsWith": lambda value, prefix: value.startswith(prefix),
        "always": lambda: True, "cancelled": lambda: cancelled,
        "format": lambda template, *values: template.format(*values),
    })


def _triggered(workflow, event, ref):
    """Positive push filters used here, per GitHub workflow-syntax on.push docs.

    Tags-only excludes the undefined branch ref type; check this BEFORE jobs.
    This is not a general GitHub event/path/expression emulator.
    """
    triggers = workflow.get("on", workflow.get(True))
    if event not in triggers:
        return False
    if event != "push":
        return True
    filters = triggers[event] or {}
    for prefix, kind, other in (("refs/heads/", "branches", "tags"),
                                 ("refs/tags/", "tags", "branches")):
        if ref.startswith(prefix):
            if kind not in filters:
                return other not in filters
            return any(fnmatchcase(ref[len(prefix):], pattern) for pattern in filters[kind])
    return False


@pytest.mark.parametrize("name", ["ci.yml", "ui-browser-push.yml", "provider-canary-push.yml"])
@pytest.mark.parametrize("event,ref", [
    ("push", "refs/heads/main"), ("push", "refs/heads/ouroboros"),
    ("push", "refs/heads/ouroboros-stable"), ("push", "refs/heads/v7.7.0"),
    ("pull_request", "refs/pull/42/merge"), ("pull_request_target", "refs/heads/ouroboros"),
    ("schedule", "refs/heads/main"), ("push", "refs/tags/other"),
    ("push", "refs/tags/v7.7.0"), ("push", "refs/tags/v8.0.0-rc.1"),
    ("workflow_dispatch", "refs/heads/candidate"), ("workflow_dispatch", "refs/tags/v7.7.0"),
])
def test_ordinary_ci_triggers_only_on_request_or_release(name, event, ref):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))
    expected = event == "workflow_dispatch" or (
        name == "ci.yml" and event == "push" and ref.startswith("refs/tags/v"))
    assert _triggered(workflow, event, ref) is expected


@pytest.mark.parametrize("event,ref,base,quick,platforms", [
    ("pull_request", "refs/pull/42/merge", "ouroboros", False, []),
    ("pull_request", "refs/pull/42/merge", "main", False, []),
    # Retained job predicates cannot bypass the removed workflow triggers.
    ("push", "refs/heads/ouroboros", "", False, []),
    ("push", "refs/heads/main", "", False, []),
    ("push", "refs/heads/ouroboros-stable", "", False, []),
    ("push", "refs/tags/v7.0.0", "", False, ["ubuntu-latest", "windows-latest", "macos-latest"]),
    ("workflow_dispatch", "refs/heads/candidate", "", True, ["ubuntu-latest", "windows-latest", "macos-latest"]),
    ("schedule", "refs/heads/main", "", False, []),
])
def test_ordinary_matrix_requires_admitted_manual_or_tag_event(event, ref, base, quick, platforms):
    facts = {"event": event, "ref": ref, "base": base}
    triggered = _triggered(WORKFLOW, event, ref)
    assert (triggered and bool(_value(WORKFLOW["jobs"]["quick-test"]["if"], **facts))) is quick
    job = WORKFLOW["jobs"]["full-test-runs"]
    admitted = triggered and _value(job["if"], **facts)
    rows = job["strategy"]["matrix"]["include"]
    actual = [row["os"] for row in rows] if admitted else []
    assert sorted(set(actual)) == sorted(platforms)
    # Every OS runs whole-file duration shards 1..N; macOS stays within the free plan's five slots.
    shards = {"ubuntu-latest": 4, "macos-latest": 4, "windows-latest": 8}
    for os, count in shards.items():
        assert actual.count(os) == (count if os in platforms else 0)
        mine = [row for row in rows if row["os"] == os]
        assert sorted(row["shard"] for row in mine) == list(range(1, count + 1))
        assert {row["shards"] for row in mine} == {count}
    assert job["strategy"]["fail-fast"] is False
    assert job["runs-on"] == "$" + "{{ matrix.os }}"


@pytest.mark.parametrize("name", ["quick-test", "sqlite-floor", "betterleaks-platform-smoke"])
@pytest.mark.parametrize("event,ref,expected", [
    ("workflow_dispatch", "refs/heads/candidate", True),
    ("push", "refs/tags/v7.7.0", False),
    ("push", "refs/heads/ouroboros", False),
    ("pull_request", "refs/pull/42/merge", False),
])
def test_additional_pre_tag_checks_require_manual_dispatch(name, event, ref, expected):
    admitted = _triggered(WORKFLOW, event, ref) and bool(_value(
        WORKFLOW["jobs"][name]["if"], event=event, ref=ref, base="ouroboros"))
    assert admitted is expected


def test_dormant_pr_predicates_keep_checkout_and_base_evidence_secret_free():
    triggers = WORKFLOW.get("on") or WORKFLOW[True]  # PyYAML's YAML 1.1 spelling of "on".
    assert "pull_request" not in triggers
    assert "pull_request_target" not in triggers
    assert WORKFLOW["permissions"] == {"contents": "read"}
    job = WORKFLOW["jobs"]["quick-test"]
    assert "secrets." not in json.dumps(job)
    # The book-growth exception reads PR label events and the accepted-failure check reads
    # issue label events, without any write grant.
    assert job["permissions"] == {"contents": "read", "pull-requests": "read", "issues": "read"}
    checkout = job["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"] == {"fetch-depth": 0}  # Default PR checkout tests the merge ref.
    base = next(step["env"]["OURO_SIZE_RATCHET_BASE_REF"] for step in job["steps"]
                if "OURO_SIZE_RATCHET_BASE_REF" in step.get("env", {}))
    assert _value(base, event="pull_request", ref="refs/pull/42/merge", base="ouroboros") == "pr-base"
    assert _value(base, event="push", ref="refs/heads/ouroboros-stable") == "previous-tip"


# This workflow carries no schedule (owner, 2026-10-05): an event bearing any
# cron, current or stale, admits no ordinary job.
@pytest.mark.parametrize("cron", ["37 4 * * *", pytest.param("17 3 * * *", id="foreign-cron")])
def test_scheduled_main_runs_do_not_enter_the_ordinary_matrix(cron):
    for name in ("quick-test", "full-test-runs"):
        assert not _value(
            WORKFLOW["jobs"][name]["if"], event="schedule", ref="refs/heads/main", schedule=cron,
        )


@pytest.mark.parametrize("event,ref,schedule,e2e_live,admitted", [
    ("workflow_dispatch", "refs/heads/ouroboros", "", "true", True),
    ("workflow_dispatch", "refs/heads/candidate", "", "true", True),
    ("workflow_dispatch", "refs/heads/ouroboros", "", "false", False),
    ("schedule", "refs/heads/main", "37 4 * * *", "false", False),
    ("schedule", "refs/heads/main", "17 3 * * *", "false", False),
    ("push", "refs/heads/ouroboros", "", "false", False),
    ("push", "refs/heads/main", "", "false", False),
    ("push", "refs/tags/v7.0.0", "", "false", False),
    ("pull_request", "refs/pull/42/merge", "", "false", False),
])
def test_the_paid_live_stand_runs_only_on_a_dispatch_that_opts_in(event, ref, schedule, e2e_live, admitted):
    """The paid `e2e-live` job takes no schedule: only a dispatch with `e2e_live=true` admits it."""
    job = WORKFLOW["jobs"]["e2e-live"]
    facts = {"event": event, "ref": ref, "schedule": schedule, "e2e_live": e2e_live}
    assert bool(_value(job["if"], base="ouroboros" if event == "pull_request" else "", **facts)) is admitted


@pytest.mark.parametrize("name", [
    "integration-test", "integration-optional", "skill-smoke", "system-e2e-mock", "e2e-live",
    "marker-guards", "docker-ui-smoke", "docker-portable-test",
    "release-preflight", "build", "release", "vendor-package-smoke",
])
def test_desktop_pr_coverage_does_not_admit_provider_or_release_jobs(name):
    job = WORKFLOW["jobs"][name]
    assert not _value(job["if"], event="pull_request", ref="refs/pull/42/merge", base="ouroboros")


@pytest.mark.parametrize("event,ref,schedule,called", [
    ("push", "refs/heads/main", "", False),
    ("push", "refs/heads/ouroboros", "", False),
    ("push", "refs/heads/ouroboros-stable", "", False),
    ("push", "refs/tags/v7.0.0", "", True),
    ("workflow_dispatch", "refs/heads/candidate", "", True),
    ("pull_request", "refs/pull/42/merge", "", False),
    ("schedule", "refs/heads/main", "37 4 * * *", False),
    ("schedule", "refs/heads/main", "17 3 * * *", False),
])
def test_provider_canaries_join_this_workflow_only_for_manual_runs_and_tags(event, ref, schedule, called):
    """Provider jobs retain their required/optional manual/tag qualification split."""
    for name, scope in (("integration-test", "required"), ("integration-optional", "optional")):
        job = WORKFLOW["jobs"][name]
        assert job["uses"] == "./.github/workflows/provider-canary.yml"
        assert job["with"] == {"scope": scope}
        for repository in ("razzant/ouroboros", "someone/private-copy"):
            assert bool(_value(job["if"], event=event, ref=ref, base="ouroboros",
                               schedule=schedule, repository=repository)) is called


@pytest.mark.parametrize(("event", "cancelled", "expected"), [
    ("pull_request", False, True), ("pull_request", True, False),
    ("schedule", False, False), ("schedule", True, False),
])
def test_status_checks_and_inequality_keep_independent_meanings(event, cancelled, expected):
    assert _value(
        "${{ always() && !cancelled() && github.event_name != 'schedule' }}",
        event=event, ref="refs/heads/candidate", cancelled=cancelled,
    ) is expected


@pytest.mark.parametrize("event,attempt,group", [
    ("push", 1, "ci-run-77-1"),
    # GitHub keeps one pending run per group: every attempt owns its group, so none displaces another.
    ("push", 2, "ci-run-77-2"),
    ("workflow_dispatch", 1, "ci-run-77-1"),
])
def test_every_run_attempt_has_its_own_group_and_cancels_nothing(event, attempt, group):
    concurrency = WORKFLOW["concurrency"]
    facts = {"event": event, "ref": "refs/heads/candidate", "attempt": attempt}
    assert _value(concurrency["group"], **facts) == group
    assert concurrency["cancel-in-progress"] is False


def test_manual_and_tag_desktop_matrix_remains_available_to_forks():
    """No automatic branch tests; explicit qualification still covers three OSes."""
    job = WORKFLOW["jobs"]["full-test-runs"]
    assert not _triggered(WORKFLOW, "push", "refs/heads/ouroboros")
    for event, ref in (("workflow_dispatch", "refs/heads/candidate"), ("push", "refs/tags/v7.7.0")):
        assert _triggered(WORKFLOW, event, ref)
        assert _value(job["if"], event=event, ref=ref, repository="someone/private-copy")


@pytest.mark.parametrize("run_id,module", [
    ("attachment_browser", "tests/test_chat_attachments_browser.py"),
    ("composer_browser", "tests/test_chat_input_sizing_browser.py"),
])
def test_the_windows_leg_runs_its_browser_modules_and_refuses_a_skip(tmp_path, run_id, module):
    """Real Windows browser cases in THIS matrix (no second one): the owner-attachment route's confined
    reads and the message field's sizing, each one module's Chromium cases from one shared install,
    never the UI lane's guard, and a run with no case or a skip fails."""
    import subprocess
    import sys

    job = WORKFLOW["jobs"]["full-test-runs"]
    steps = {step.get("id"): step for step in job["steps"] if step.get("id")}
    install, run = steps["attachment_browser_install"], steps[run_id]
    for step in (install, run):
        assert "!cancelled()" in step["if"] and "runner.os == 'Windows'" in step["if"]
        assert isinstance(step["timeout-minutes"], int) and "continue-on-error" not in step
    assert "steps.attachment_browser_install.outcome == 'success'" in run["if"]
    # The fixture refuses an interpreter that can import an installed ouroboros, and this job's
    # shared env installs the checkout: both steps use their own dependency-only env, synced from
    # the same lock as the setup action's test profile (safe_test keeps PATH, so a bare `python`
    # would be the shared env's).
    shared = next(step for step in job["steps"] if step.get("uses") == "./.github/actions/setup-python-env")
    assert shared.get("with", {}).get("install-project", "true") == "true", "the shared env is left as it is"
    action = yaml.safe_load((ROOT / ".github/actions/setup-python-env/action.yml").read_text(encoding="utf-8"))
    sync_script = next(step["run"] for step in action["runs"]["steps"] if step.get("name") == "Sync locked dependencies")
    profile = re.search(r'^\s*test\) (uv sync [^;]+?) "\$\{PROJECT_ARGS\[@\]\}"', sync_script, re.M).group(1)
    env_root = install["env"]["UV_PROJECT_ENVIRONMENT"]
    assert env_root.startswith("${{ runner.temp }}/"), "outside the checkout the fixture copies"
    assert run["env"]["ATTACHMENT_PYTHON"] == f"{env_root}/Scripts/python.exe"
    base, sync, playwright = install["run"].strip().splitlines()
    # No trailing newline to capture: Git Bash keeps a Windows CR inside $(...).
    assert base == 'base="$(python -c ' + "'import sys; print(sys._base_executable, end=\"\")'" + ')"'
    assert sync == f'{profile} --no-install-project --python "$base"'
    assert playwright == '"$UV_PROJECT_ENVIRONMENT/Scripts/python.exe" -m playwright install chromium'
    assert run["env"]["OUROBOROS_RUN_UI_SMOKE"] == "1"
    assert install["env"]["PLAYWRIGHT_BROWSERS_PATH"] == run["env"]["PLAYWRIGHT_BROWSERS_PATH"]
    command, check, *evidence = run["run"].strip().splitlines()
    assert command.startswith('"$ATTACHMENT_PYTHON" -I -S scripts/safe_test.py -- "$ATTACHMENT_PYTHON" -m pytest '
                              + module)
    assert " -m ui_browser -k chromium " in command and "--require-ui-browser" not in command
    assert "matrix" not in run["run"]
    assert "matrix.shard == 1" in install["if"] and "matrix.shard == 1" in run["if"]
    assert job["strategy"]["matrix"].keys() == {"include"}
    # The skip guard, run as written against a junit report of each shape.
    for tests, skipped, ok in ((3, 0, True), (3, 1, False), (0, 0, False)):
        report = tmp_path / "attachment-browser.xml"
        report.write_text(f'<testsuites><testsuite tests="{tests}" skipped="{skipped}"/></testsuites>', encoding="utf-8")
        script = check.split('python -c "', 1)[1].rsplit('" "$RUNNER_TEMP', 1)[0]
        result = subprocess.run([sys.executable, "-c", script, str(report)], capture_output=True, text=True)
        assert (result.returncode == 0) is ok, (tests, skipped, result.stdout, result.stderr)
    if run_id != "composer_browser":
        assert not evidence
        return
    # The sizing evidence reaches the artifact: safe_test keeps the evidence root, the step fails
    # without the geometry it wrote, and the upload takes that root whatever the test's outcome.
    # Exercise the launcher rather than pinning a duplicated keep-list literal:
    # the shared diagnostic-key vocabulary may supply it through a splat.
    import os
    destination = tmp_path / "selected-composer-evidence"
    probe = ("import os,pathlib; p=pathlib.Path(os.environ['OUROBOROS_UI_EVIDENCE_DIR']); "
             "p.mkdir(parents=True); (p/'geometry.json').write_text('[]'); "
             "assert 'OPENAI_API_KEY' not in os.environ; print(p)")
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(ROOT / "scripts/safe_test.py"),
         "--temp-parent", str(tmp_path), "--evidence-dir", str(tmp_path / "launcher-evidence"),
         "--", sys.executable, "-c", probe], cwd=ROOT,
        env={**os.environ, "OUROBOROS_UI_EVIDENCE_DIR": str(destination),
             "OPENAI_API_KEY": "synthetic-must-be-scrubbed"},
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert str(destination) in result.stdout
    assert (destination / "geometry.json").read_text() == "[]"
    assert "SAFE_TEST_RELEASED " in result.stderr
    root = run["env"]["OUROBOROS_UI_EVIDENCE_DIR"]
    assert root == "${{ runner.temp }}/ci-evidence/composer"
    [written] = evidence
    script = written.split('python -c "', 1)[1].rsplit('" "$OUROBOROS_UI_EVIDENCE_DIR', 1)[0]
    assert written.endswith('"$OUROBOROS_UI_EVIDENCE_DIR/composer-sizing-chromium.json"')
    for content, ok in ((None, False), ("", False), ("[]", True)):
        geometry = tmp_path / "composer-sizing-chromium.json"
        geometry.unlink(missing_ok=True)
        if content is not None:
            geometry.write_text(content, encoding="utf-8")
        result = subprocess.run([sys.executable, "-c", script, str(geometry)], capture_output=True, text=True)
        assert (result.returncode == 0) is ok, (content, result.stdout, result.stderr)
    upload = next(step for step in job["steps"] if step.get("with", {}).get("name") == "composer-sizing-windows")
    assert job["steps"].index(upload) == job["steps"].index(run) + 1
    assert upload["uses"].startswith("actions/upload-artifact@") and upload["continue-on-error"] is True
    assert upload["with"]["path"].rstrip("/") == root
    assert "runner.os == 'Windows'" in upload["if"] and "steps.attachment_browser_install.outcome == 'success'" in upload["if"]
