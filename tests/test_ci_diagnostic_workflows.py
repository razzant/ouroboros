"""Public CI exports preserve producer exits, secret boundaries and full UI proof."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml
from tests._cases import chosen_rows
import hashlib
import importlib.util
import io
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ACTION = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
CANARY, CANARY_PUSH = "provider-canary.yml", "provider-canary-push.yml"
# OPENROUTER_PROVIDER_ROUTES_API_KEY: the OpenRouter routes to Google, Anthropic and OpenAI (tests/provider_contract_ci.py).
PROVIDER_SECRETS = (
    "OPENROUTER_API_KEY", "OPENROUTER_PROVIDER_ROUTES_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "CLOUDRU_FOUNDATION_MODELS_BASE_URL", "GIGACHAT_CREDENTIALS",
)
DOWNLOAD_ACTION = "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
UI_JOBS = ["scope", "ui-shard", "ui-manifest", "ui-smoke"]
# The complete lane takes no selection: its jobs run whenever browsers are needed.
FULL_LANE = "${{ needs.scope.outputs.run_browser == 'true' }}"
# The single-scenario PARTIAL DIAGNOSTIC is a job of the push wrapper, its only selector.
PARTIAL = ("ui-browser-push.yml", "ui-diagnostic")


def _workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def _steps(name, job):
    return {step["id"]: step for step in _workflow(name)["jobs"][job]["steps"] if "id" in step}


def test_provider_diagnostics_stay_outside_test_and_release_authority():
    workflow = _workflow("ci.yml")
    job = _workflow(CANARY)["jobs"]["integration-test"]
    steps = _steps(CANARY, "integration-test")
    producer = steps["provider_tests"]
    assert "continue-on-error" not in job and "continue-on-error" not in producer
    assert "continue-on-error" not in workflow["jobs"]["integration-test"]
    assert 'pytest tests/test_provider_integration.py -m "$selection" -q -rs --tb=short' in producer["run"]
    assert "--ci-evidence-dir=" in producer["run"] and "--junitxml=" in producer["run"]
    assert "ci-private/provider/results.xml" in producer["run"]
    assert "ci-evidence/provider" in producer["run"]
    assert "secrets." in str(producer["env"])
    for step in job["steps"]:
        if step.get("id") != "provider_tests":
            assert "secrets." not in str(step)
    assert "env" not in job
    upload, summary = steps["provider_evidence"], steps["provider_summary"]
    assert upload["uses"] == UPLOAD_ACTION
    assert upload["with"]["path"] == "${{ runner.temp }}/ci-evidence/provider/"
    assert upload["with"]["retention-days"] == 30
    assert "github.sha" in upload["with"]["name"] and "github.run_attempt" in upload["with"]["name"]
    assert "${{ inputs.scope }}" in upload["with"]["name"]
    assert summary["env"]["PRODUCER_OUTCOME"] == "${{ steps.provider_tests.outcome }}"
    assert summary["env"]["ARTIFACT_URL"] == "${{ steps.provider_evidence.outputs.artifact-url }}"
    assert "python -m tests.ci_evidence summarize" in summary["run"]
    assert list(steps).index("provider_evidence") < list(steps).index("provider_summary")
    # The preflight checks the tag alone (builds start beside the tests); the release requires the canaries.
    assert "needs" not in workflow["jobs"]["release-preflight"]
    release = workflow["jobs"]["release"]
    assert "integration-test" in release["needs"] and "needs.integration-test.result == 'success'" in release["if"]
    assert "needs.release-preflight.result == 'success'" in release["if"]
    optional = workflow["jobs"]["integration-optional"]
    assert "continue-on-error" not in optional
    assert not any("integration-optional" in job.get("needs", []) for job in workflow["jobs"].values())


def test_only_informational_steps_tolerate_errors_and_report_missing_uploads():
    for name, jobname in ((CANARY, "integration-test"), ("ui-browser.yml", "ui-shard"), PARTIAL):
        job = _workflow(name)["jobs"][jobname]
        tolerant = [step for step in job["steps"] if step.get("continue-on-error")]
        assert tolerant
        for step in tolerant:
            assert "!cancelled()" in step["if"]
            assert (step.get("uses") == UPLOAD_ACTION or
                    "tests.ci_evidence summarize" in step.get("run", "") or
                    "tests/ci_evidence.py annotate" in step.get("run", "") or
                    "ouroboros.test_evidence export-diagnostics" in step.get("run", "") or
                    "diagnostics_incomplete" in step.get("run", ""))
            assert "-m pytest" not in step.get("run", "")
        warning = next(step for step in job["steps"] if "Disclose incomplete" in step.get("name", ""))
        assert "outputs.artifact-url == ''" in warning["if"]
        assert "summary.outcome != 'success'" in warning["if"]
        assert "::warning::" in warning["run"] and "$GITHUB_STEP_SUMMARY" in warning["run"]
    # The coverage decision, the lane manifest and the lane verdict tolerate nothing:
    # there a failed step is the result.
    for jobname in ("scope", "ui-manifest", "ui-smoke"):
        job = _workflow("ui-browser.yml")["jobs"][jobname]
        assert "continue-on-error" not in job
        assert not any("continue-on-error" in step for step in job["steps"]), jobname


def _triggers(workflow):
    return workflow.get("on", workflow.get(True))  # PyYAML's YAML 1.1 spelling of "on".


def _canary_wiring_faults(ci, shared, push):
    """One body, two disjoint release groups and an explicit all-canary dispatch."""
    faults = []
    passed = {name: f"${{{{ secrets.{name} }}}}" for name in PROVIDER_SECRETS}
    for label, caller, scope in (("ci.yml", ci["jobs"]["integration-test"], {"scope": "required"}),
                                ("ci.yml optional", ci["jobs"]["integration-optional"], {"scope": "optional"}),
                                (CANARY_PUSH, push["jobs"]["integration-test"], {})):
        if caller.get("uses") != f"./.github/workflows/{CANARY}":
            faults.append(f"{label} does not call the shared job")
        if caller.get("secrets") != passed:
            faults.append(f"{label} does not pass exactly the nine provider secrets by name")
        if caller.get("permissions") != {"contents": "read"}:
            faults.append(f"{label} widens the called job's permissions")
        if caller.get("with", {}) != scope:
            faults.append(f"{label} does not select its disjoint canary scope")
    if ci["jobs"]["integration-test"].get("if") != ci["jobs"]["integration-optional"].get("if"):
        faults.append("required/optional callers do not share the manual/tag admission")
    # Exact key sets keep an added condition/group from silently changing paid admission.
    for label, job, keys in (("ci.yml", ci["jobs"]["integration-test"], {"if", "uses", "with", "secrets", "permissions"}),
                             ("ci.yml optional", ci["jobs"]["integration-optional"], {"if", "uses", "with", "secrets", "permissions"}),
                             (CANARY_PUSH, push["jobs"]["integration-test"], {"uses", "secrets", "permissions"}),
                             (CANARY, shared["jobs"]["integration-test"], {"runs-on", "steps"})):
        if set(job) != keys:
            faults.append(f"{label} canary job carries keys other than {sorted(keys)}")
    declared = {name: {"required": False} for name in PROVIDER_SECRETS}
    inputs = {"scope": {"description": "Canary subset: all, required or optional; failures remain visible",
                        "type": "string", "default": "all"}}
    if _triggers(shared) != {"workflow_call": {"secrets": declared, "inputs": inputs}}:
        faults.append("the shared job is not call-only with scoped input and nine optional secrets")
    producer = next((step for step in shared["jobs"]["integration-test"]["steps"] if step.get("id") == "provider_tests"), {})
    if {name: (producer.get("env") or {}).get(name) for name in PROVIDER_SECRETS} != passed:
        faults.append("the pytest step does not receive every declared secret under its own name")
    if _triggers(push) != {"workflow_dispatch": None}:
        faults.append("the standalone provider wrapper must be manual-only")
    # Tags-only is not an unfiltered push: GitHub excludes the undefined branch ref type.
    if set(_triggers(ci)) != {"push", "workflow_dispatch"} or _triggers(ci).get("push") != {"tags": ["v*"]}:
        faults.append("ci.yml must admit only manual dispatch and v* tags without a path filter")
    for label, workflow in ((CANARY, shared), (CANARY_PUSH, push)):
        if list(workflow["jobs"]) != ["integration-test"]:
            faults.append(f"{label} holds more than the canary job")
        if workflow.get("permissions") != {"contents": "read"}:
            faults.append(f"{label} does not default to read-only permissions")
        # A group would cancel or replace a paid run that belongs to another commit.
        if any("concurrency" in scope for scope in (workflow, *workflow["jobs"].values())):
            faults.append(f"{label} declares a concurrency group")
    return faults


def test_provider_canaries_share_one_body_between_release_and_manual_callers():
    assert _canary_wiring_faults(*map(_workflow, ("ci.yml", CANARY, CANARY_PUSH))) == []
    callers = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        for name, job in _workflow(path.name)["jobs"].items():
            assert job.get("secrets") != "inherit", path.name  # Would hand over signing and live-stand keys.
            if str(job.get("uses", "")).endswith(f"/{CANARY}"):
                callers.append((path.name, name))
    # Two release groups partition one matrix; a further caller could duplicate paid tests.
    assert callers == [("ci.yml", "integration-test"), ("ci.yml", "integration-optional"),
                       (CANARY_PUSH, "integration-test")]


def test_provider_canary_workflows_are_editable_while_ci_yml_stays_protected():
    """`release-preflight` requires the provider-canary job, but ci.yml, which
    wires that job into the release chain, is the protected release invariant;
    the canary body and its manual wrapper reach no install, so Advanced
    may edit them. The contributor review label still follows the body."""
    from ouroboros.runtime_mode_policy import RELEASE_INVARIANT_PATHS, protected_path_category
    from scripts.run_external_review import _RELEASE_MACHINERY_PATHS

    parent = ".github/workflows/ci.yml"
    assert parent in RELEASE_INVARIANT_PATHS
    assert protected_path_category(parent) == protected_path_category(f"./{parent}") == "release-invariant"
    for name in (CANARY, CANARY_PUSH):
        path = f".github/workflows/{name}"
        assert (ROOT / path).is_file(), path
        assert path not in RELEASE_INVARIANT_PATHS, path
        assert path in _RELEASE_MACHINERY_PATHS, path  # The contributor label follows the body too.
        assert protected_path_category(path) == protected_path_category(f"./{path}") == "", path


CANARY_DRIFTS = {
    "optional repeats required": lambda w: w.ci["jobs"]["integration-optional"]["with"].update(scope="required"),
    "required repeats all": lambda w: w.ci["jobs"]["integration-test"]["with"].update(scope="all"),
    "optional admission diverges": lambda w: w.ci["jobs"]["integration-optional"].update({"if": "true"}),
    "wrapper also fires on tags": lambda w: _triggers(w.push).update(push={"tags": ["v*"]}),
    "wrapper restores branch pushes": lambda w: _triggers(w.push).update(push={"branches": ["ouroboros"]}),
    "wrapper restores PRs": lambda w: _triggers(w.push).update(pull_request=None),
    "wrapper loses dispatch": lambda w: _triggers(w.push).pop("workflow_dispatch"),
    "ci.yml regains a path filter": lambda w: _triggers(w.ci)["push"].update(paths=["ouroboros/**"]),
    "ci.yml ignores a path": lambda w: _triggers(w.ci)["push"].update({"paths-ignore": ["prompts/**"]}),
    "ci.yml restores branches": lambda w: _triggers(w.ci)["push"].update(branches=["ouroboros"]),
    "ci.yml restores PRs": lambda w: _triggers(w.ci).update(pull_request={"branches": ["ouroboros"]}),
    "ci.yml drops tags": lambda w: _triggers(w.ci)["push"].pop("tags"),
    "ci.yml drops dispatch": lambda w: _triggers(w.ci).pop("workflow_dispatch"),
    "wrapper concurrency": lambda w: w.push.update(concurrency="canary"),
    "shared job concurrency": lambda w: w.shared["jobs"]["integration-test"].update(concurrency="canary"),
    "inherited secrets": lambda w: w.ci["jobs"]["integration-test"].update(secrets="inherit"),
    "caller drops a secret": lambda w: w.push["jobs"]["integration-test"]["secrets"].pop("OPENAI_API_KEY"),
    "required secret": lambda w: _triggers(w.shared)["workflow_call"]["secrets"]["OPENAI_API_KEY"].update(
        required=True),
    "undeclared tenth secret": lambda w: _triggers(w.shared)["workflow_call"]["secrets"].update(EXTRA={}),
    "pytest step drops the routes key": lambda w: next(step for step in w.shared["jobs"]["integration-test"]["steps"]
                                                   if step.get("id") == "provider_tests")["env"].pop(
                                                       "OPENROUTER_PROVIDER_ROUTES_API_KEY"),
    "caller drops the routes key": lambda w: w.ci["jobs"]["integration-test"]["secrets"].pop(
        "OPENROUTER_PROVIDER_ROUTES_API_KEY"),
    "shared job self-triggers": lambda w: _triggers(w.shared).update(push=None),
    "caller leaves the shared job": lambda w: w.ci["jobs"]["integration-test"].update(uses="./other.yml"),
    "shared job loses its read-only default": lambda w: w.shared.pop("permissions"),
    "caller widens permissions": lambda w: w.push["jobs"]["integration-test"].update(permissions="write-all"),
    "wrapper job condition": lambda w: w.push["jobs"]["integration-test"].update(
        {"if": "github.ref == 'refs/heads/ouroboros-stable'"}),
    "shared job condition": lambda w: w.shared["jobs"]["integration-test"].update(
        {"if": "github.event_name == 'workflow_dispatch'"}),
    "code-workflow caller concurrency": lambda w: w.ci["jobs"]["integration-test"].update(concurrency="canary"),
}


@pytest.mark.parametrize("drift", chosen_rows(CANARY_DRIFTS.values(), 0, 21, 27), ids=chosen_rows(list(CANARY_DRIFTS), 0, 21, 27))
def test_each_provider_canary_wiring_drift_is_named(drift):
    files = SimpleNamespace(ci=_workflow("ci.yml"), shared=_workflow(CANARY), push=_workflow(CANARY_PUSH))
    drift(files)
    assert _canary_wiring_faults(files.ci, files.shared, files.push)
def _selected(job, event, selection):
    """Whether a push-wrapper job runs for an event: its `if`, two comparisons joined once."""
    facts = {"github.event_name": event, "inputs.diagnostic": selection}
    comparison = r"(github\.event_name|inputs\.diagnostic) (==|!=) '([a-z_]+)'"
    match = re.fullmatch(r"\$\{\{ " + comparison + r" (&&|\|\|) " + comparison + r" \}\}", job["if"])
    assert match, job["if"]
    left, right = (((facts[name] == value) is (operator == "==")) for name, operator, value
                   in (match.group(1, 2, 3), match.group(5, 6, 7)))
    return (left and right) if match[4] == "&&" else (left or right)


def test_manual_ui_selection_is_fixed_partial_and_never_a_paid_or_full_check():
    caller = _workflow("ui-browser-push.yml")
    triggers = caller.get("on", caller.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    selection = triggers["workflow_dispatch"]["inputs"]["diagnostic"]
    assert selection["type"] == "choice" and selection["default"] == "full"
    assert selection["options"] == ["full", "viewport", "inflight"]
    assert "PARTIAL DIAGNOSTIC" in caller["run-name"]
    assert list(caller["jobs"]) == ["shard-plan", "ui-smoke", "ui-diagnostic"] and "secrets" not in str(caller)
    lane, partial_job = caller["jobs"]["ui-smoke"], caller["jobs"]["ui-diagnostic"]
    # Exactly one job runs for a dispatch: full or a clearly partial diagnostic.
    for event, choice, runs_lane in (("workflow_dispatch", "full", True),
                                     ("workflow_dispatch", "viewport", False),
                                     ("workflow_dispatch", "inflight", False),
                                     ("workflow_dispatch", "", False), ("workflow_dispatch", "tests", False)):
        assert _selected(lane, event, choice) is runs_lane, (event, choice)
        assert _selected(partial_job, event, choice) is not runs_lane, (event, choice)
    # The lane job is the shared lane and carries no selection into it.
    assert set(lane) == {"name", "if", "needs", "uses", "permissions"} and lane["name"] == "ui-smoke"
    assert lane["uses"] == "./.github/workflows/ui-browser.yml"
    # A partial run is named as one wherever it shows; the lane's names never say so.
    assert "format('PARTIAL DIAGNOSTIC UI - {0}', inputs.diagnostic)" in partial_job["name"]
    assert "format('PARTIAL DIAGNOSTIC UI - {0}', inputs.diagnostic)" in caller["run-name"]
    assert "uses" not in partial_job and "needs" not in partial_job and "strategy" not in partial_job

    shared_text = (ROOT / ".github/workflows/ui-browser.yml").read_text(encoding="utf-8")
    shared = _workflow("ui-browser.yml")
    # The shared lane is the complete lane only: no input, so no caller can select a part
    # of it, and no job of it is skipped whenever browsers run.
    assert shared.get("on", shared.get(True)) == {"workflow_call": None}
    assert "inputs." not in shared_text and "PARTIAL DIAGNOSTIC UI" not in shared_text
    assert list(shared["jobs"]) == UI_JOBS
    assert [shared["jobs"][name].get("if") for name in UI_JOBS] == [None, FULL_LANE, FULL_LANE, "${{ !cancelled() }}"]
    assert [shared["jobs"][name]["name"] for name in UI_JOBS] == [
        "ui-scope", "ui-shard (${{ matrix.shard }}/9)", "ui-manifest", "ui-smoke"]
    assert "with" not in _workflow("ci.yml")["jobs"]["ui-smoke"]
    shard, diagnostic = _steps("ui-browser.yml", "ui-shard"), _steps(*PARTIAL)
    full, partial = shard["ui_tests"], diagnostic["ui_diagnostic"]
    assert "--require-ui-browser" in full["run"] and "pytest tests/ -m ui_browser" in full["run"]
    # A partial scenario is neither guarded, sharded, reconciled nor followed by browser tools.
    for full_lane_only in ("--require-ui-browser", "--ui-browser-shard", "reconcile-shards",
                           "test_browser_tools_smoke", "matrix", "download-artifact"):
        assert full_lane_only not in str(partial_job), full_lane_only
        assert full_lane_only in shared_text, full_lane_only
    assert "-vv --tb=short" in _steps("ui-browser.yml", "ui-shard")["ui_tests"]["run"]
    assert '${{ inputs.diagnostic }}' not in partial["run"]
    assert partial["env"]["DIAGNOSTIC"] == "${{ inputs.diagnostic }}"
    assert "--scope \"$DIAGNOSTIC\"" in diagnostic["ui_summary"]["run"]
    assert diagnostic["ui_summary"]["env"]["DIAGNOSTIC"] == "${{ inputs.diagnostic }}"
    assert "--scope full " in shard["ui_summary"]["run"] and "DIAGNOSTIC" not in shard["ui_summary"]["env"]


@pytest.mark.parametrize("exported", [True, False], ids=["proof-present", "export-failed"])
@pytest.mark.parametrize("jobname, producer, proof_id, upload_id, directory", [
    ("ui-shard", "ui_tests", "ui_proof", "ui_evidence", "ui"),
    ("ui-manifest", "ui_manifest", "manifest_proof", "manifest_evidence", "ui-manifest"),
])
def test_an_attempt_without_a_published_proof_is_red_itself(tmp_path, jobname, producer, proof_id,
                                                              upload_id, directory, exported):
    """An attempt whose export or upload failed must not leave `ui-smoke` an earlier attempt's proof to accept."""
    job = _workflow("ui-browser.yml")["jobs"][jobname]
    steps = _steps("ui-browser.yml", jobname)
    proof, upload = steps[proof_id], steps[upload_id]
    assert "continue-on-error" not in proof and "continue-on-error" not in upload
    assert upload["with"]["if-no-files-found"] == "error"
    if jobname == "ui-shard":  # The shard's later steps run after a red lane; the manifest job stops at its first.
        assert proof["if"] == "${{ !cancelled() && steps.ui_tests.outcome != 'skipped' }}"
        assert "!cancelled()" in upload["if"]
    else:
        assert "if" not in proof and "if" not in upload
    order = [step.get("id") for step in job["steps"]]
    assert order.index(producer) < order.index(proof_id) < order.index(upload_id)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow shell unavailable on this host")
    if exported:
        target = tmp_path / "ci-evidence" / directory / "host" / "results.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}", encoding="utf-8")
    command = proof["run"].replace("${{ runner.temp }}", str(tmp_path))
    done = subprocess.run([bash, "-e", "-o", "pipefail", "-c", command], capture_output=True, timeout=10)
    assert (done.returncode == 0) is exported


def test_ui_producers_have_distinct_reports_and_only_the_public_parent_is_uploaded():
    shard, diagnostic = _steps("ui-browser.yml", "ui-shard"), _steps(*PARTIAL)
    run = "${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}"
    for steps in (shard, diagnostic):
        upload = steps["ui_evidence"]
        assert upload["uses"] == UPLOAD_ACTION
        assert upload["with"]["path"] == "${{ runner.temp }}/ci-evidence/ui/"
    # One artifact per shard and attempt: a matrix leg or a rerun never collides with another.
    assert shard["ui_evidence"]["with"]["name"] == "ui-ci-full-shard-${{ matrix.shard }}-" + run
    # A scenario's artifact carries its selection, which is never `full`: no lane proof pattern matches it.
    assert diagnostic["ui_evidence"]["with"]["name"] == "ui-ci-${{ inputs.diagnostic }}-" + run
    for steps, key, directory in ((shard, "ui_tests", "host"), (diagnostic, "ui_diagnostic", "host"),
                                  (shard, "browser_tools", "tools")):
        run = steps[key]["run"]
        assert f"ci-private/ui/{directory}.xml" in run
        assert f"ci-evidence/ui/{directory}" in run
        assert "continue-on-error" not in steps[key]
    assert shard["ui_summary"]["env"]["PRODUCER_OUTCOME"] == "${{ steps.ui_tests.outcome }}"
    assert diagnostic["ui_summary"]["env"]["PRODUCER_OUTCOME"] == "${{ steps.ui_diagnostic.outcome }}"
    assert shard["tools_summary"]["env"]["PRODUCER_OUTCOME"] == "${{ steps.browser_tools.outcome }}"
    assert "ci-evidence/ui/host" in shard["ui_summary"]["run"]
    assert "ci-evidence/ui/host" in diagnostic["ui_summary"]["run"]
    assert "ci-evidence/ui/tools" in shard["tools_summary"]["run"]
    assert '--label "shard ${{ matrix.shard }}/9"' in shard["ui_summary"]["run"]
    annotate = shard["ui_annotations"]["run"]
    assert "python -I -S tests/ci_evidence.py annotate" in annotate
    assert "ci-evidence/ui/host" in annotate and "ci-evidence/ui/tools" in annotate


@pytest.mark.parametrize("workflow, job, producer, lane", [
    ("ui-browser.yml", "ui-shard", "ui_tests", "host"), ("ui-browser.yml", "ui-shard", "browser_tools", "tools"),
    (*PARTIAL, "ui_diagnostic", "host")], ids=["host-lane", "browser-tools", "partial"])
def test_each_ui_invocation_keeps_its_raw_run_apart_and_publishes_only_its_diagnostics(tmp_path, workflow, job,
                                                                                      producer, lane):
    """#2763: custom frames, facts and startup logs live in each safe_test invocation's run directory, never the
    proof. The run gets a raw parent outside ci-evidence; its declared diagnostics leave in their own artifact."""
    from fnmatch import fnmatchcase

    from ouroboros.test_environment import DIAGNOSTIC_OUTPUT_KEYS

    steps, job_steps = _steps(workflow, job), _workflow(workflow)["jobs"][job]["steps"]
    # A step-level destination would be shared by every retry round, which then overwrites the first attempt.
    assert set(steps[producer].get("env", {})).isdisjoint({*DIAGNOSTIC_OUTPUT_KEYS, "OUROBOROS_TEST_EVIDENCE_ROOT"})
    completed, args = _run_shell(steps[producer], tmp_path)
    assert completed.returncode == 0, completed.stderr
    launcher = args.index("scripts/safe_test.py")
    assert args[launcher + 1:launcher + 4] == ["--evidence-dir", f"{tmp_path}/ci-raw/ui/{lane}", "--"]
    export, upload, proof = steps["ui_diagnostics_export"], steps["ui_diagnostics"], steps["ui_evidence"]
    order = [step.get("id") for step in job_steps]
    assert order.index(producer) < order.index("ui_evidence") < order.index("ui_diagnostics_export") < order.index(
        "ui_diagnostics")
    assert export["run"] == ('python -m ouroboros.test_evidence export-diagnostics "${{ runner.temp }}/ci-raw/ui" '
                             '"${{ runner.temp }}/ci-diagnostics/ui"')
    assert export["if"] == "${{ !cancelled() && steps.install_browsers.outcome == 'success' }}"
    assert export["continue-on-error"] is True and upload["continue-on-error"] is True
    assert upload["uses"] == UPLOAD_ACTION and upload["if"] == (
        "${{ !cancelled() && steps.ui_diagnostics_export.outcome != 'skipped' }}")
    assert upload["with"]["path"] == "${{ runner.temp }}/ci-diagnostics/ui/" and proof["with"]["path"].endswith(
        "/ci-evidence/ui/")
    assert upload["with"]["if-no-files-found"] == "ignore" and upload["with"]["retention-days"] == 30
    warning = next(step for step in job_steps if "Disclose incomplete" in step.get("name", ""))
    for failed in ("steps.ui_diagnostics_export.outcome == 'failure'", "steps.ui_diagnostics.outcome == 'failure'"):
        assert failed in warning["if"]
    # No proof download can select the diagnostics, whatever shard or attempt made them.
    def concrete(text, shard="*", attempt="*"):
        for expression, value in (("github.sha", "c" * 40), ("github.run_id", "31"), ("inputs.diagnostic", "viewport"),
                                  ("matrix.shard", shard), ("github.run_attempt", attempt)):
            text = text.replace(f"${{{{ {expression} }}}}", value)
        return text

    patterns = [concrete(step["with"]["pattern"]) for step in _workflow("ui-browser.yml")["jobs"]["ui-smoke"]["steps"]
                if step.get("uses") == DOWNLOAD_ACTION]
    assert len(patterns) == 2 and fnmatchcase(concrete(proof["with"]["name"], "3", "2"), patterns[1]) is (
        workflow == "ui-browser.yml")
    for shard in map(str, range(1, 10)):
        for attempt in ("1", "2"):
            name = concrete(upload["with"]["name"], shard, attempt)
            assert "${{" not in name and not any(fnmatchcase(name, pattern) for pattern in patterns), name


def _evidence_run(run, files, published=()):
    """A run directory as scripts/safe_test.py and its producers write it: ``files`` maps relative paths to bytes,
    and each ``published`` producer folder holds the declaration ``diagnostic_directory(publish=True)`` writes."""
    from ouroboros.test_evidence import EVIDENCE_RUN_MARKER, PUBLISHED_DIAGNOSTICS_MARKER

    run.mkdir(parents=True)
    (run / EVIDENCE_RUN_MARKER).write_text("test evidence output, not a source input\n", encoding="utf-8")
    for relative, data in {**files, **{f"{folder}/{PUBLISHED_DIAGNOSTICS_MARKER}": b"" for folder in published}}.items():
        (run / relative).parent.mkdir(parents=True, exist_ok=True)
        (run / relative).write_bytes(data)
    return run


KEY, OTHER_KEY = "0123456789abcdef", "fedcba9876543210"  # diagnostic_directory's producer folders


def _raw_runs(raw):
    """One shard's raw parent: the host lane's first session and retry round, and the browser-tools smoke."""
    from ouroboros.test_evidence import PUBLISHED_DIAGNOSTICS_MARKER

    first = _evidence_run(raw / "host" / "run-first", {
        "owner.json": b'{"command": ["python", "-m", "pytest"]}', "result.json": b'{"returncode": 1}',
        "source/source.json": b"{}", "source/candidate.diff": b"diff", "command-x1/stdout.bin": b"full stdout",
        "command-x1/stderr.bin": b"full stderr", "command-x1/result.json": b"{}",
        "pytest-nested/results.json": b'{"reports": []}', "pytest-nested/events.jsonl": b"{}\n",
        "ui/late-input-heads/receipt.png": b"first frame", "ui/late-input-heads/results.json": b'{"width": 1}',
        f"server-startup/{KEY}/server-stdio.log": b"traceback", f"server-startup/{KEY}/data/logs/events.jsonl": b"{}\n",
        f"receipt-facts/{OTHER_KEY}/facts.json": b'{"attempt": "first"}', "receipt-facts/not-a-key/stray.txt": b"?",
        # A retained neighbour of the published facts, in their family: its producer never declared it.
        f"receipt-facts/{KEY}/drive.json": b"a synthetic review drive",
        # Shaped like producer folders, declared by nobody: the owner's record and a stray output stay with the run,
        # as does everything under a declaration at a family's level (a producer's folder declares only itself).
        f"source/{KEY}/private.txt": b"the owner's record", f"grok-boundaries/{KEY}/out.txt": b"undeclared output",
        f"grok-boundaries/{PUBLISHED_DIAGNOSTICS_MARKER}": b"", "stray.txt": b"outside the contract"},
        published=(f"server-startup/{KEY}", f"receipt-facts/{OTHER_KEY}", f"launch/{KEY}"))
    # A nested launcher run inside a published folder is that run's own record, never this run's diagnostic.
    _evidence_run(first / "launch" / KEY / "run-nested", {"source/source.json": b"{}", "ui/nested.png": b"nested"})
    _evidence_run(raw / "host" / "retry-1" / "run-retry", {"owner.json": b"{}", "ui/late-input-heads/receipt.png":
                                                           b"retry frame", f"receipt-facts/{OTHER_KEY}/facts.json":
                                                           b'{"attempt": "retry"}'}, published=(f"receipt-facts/{OTHER_KEY}",))
    _evidence_run(raw / "tools" / "run-tools", {"result.json": b"{}", f"browser-tools/{KEY}/smoke.png": b"tools"},
                  published=(f"browser-tools/{KEY}",))
    return first


_EXPORTED = {  # what the producers of _raw_runs published, without their declarations, at each run's relative path
    "host/run-first/ui/late-input-heads/receipt.png": b"first frame",
    "host/run-first/ui/late-input-heads/results.json": b'{"width": 1}',
    f"host/run-first/server-startup/{KEY}/server-stdio.log": b"traceback",
    f"host/run-first/server-startup/{KEY}/data/logs/events.jsonl": b"{}\n",
    f"host/run-first/receipt-facts/{OTHER_KEY}/facts.json": b'{"attempt": "first"}',
    "host/retry-1/run-retry/ui/late-input-heads/receipt.png": b"retry frame",
    f"host/retry-1/run-retry/receipt-facts/{OTHER_KEY}/facts.json": b'{"attempt": "retry"}',
    f"tools/run-tools/browser-tools/{KEY}/smoke.png": b"tools"}


def test_the_export_takes_each_runs_declared_diagnostics_at_its_relative_path_and_leaves_the_rest(tmp_path):
    from ouroboros.test_evidence import export_diagnostics

    raw, out = tmp_path / "ci-raw" / "ui", tmp_path / "ci-diagnostics" / "ui"
    first = _raw_runs(raw)
    try:  # a link is never followed, whatever it names (where this host can make one)
        (first / "ui" / "late-input-heads" / "elsewhere.txt").symlink_to(first / "owner.json")
    except (OSError, NotImplementedError):
        pass
    facts = export_diagnostics(raw, out)
    assert facts["errors"] == [] and facts["files"] == 8
    assert {row["run"]: row["left"] for row in facts["runs"]} == {
        "host/retry-1/run-retry": ["OUROBOROS_TEST_EVIDENCE_RUN.txt", "owner.json"],
        "host/run-first": ["OUROBOROS_TEST_EVIDENCE_RUN.txt", "command-x1", "grok-boundaries", "owner.json",
                           "pytest-nested", "result.json", "source", "stray.txt"],
        "tools/run-tools": ["OUROBOROS_TEST_EVIDENCE_RUN.txt", "result.json"]}
    exported = {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    assert exported == _EXPORTED
    assert (first / "ui" / "late-input-heads" / "receipt.png").read_bytes() == b"first frame", "sources stay"


def test_a_producer_publishes_only_its_own_folder_and_retained_output_is_the_default(tmp_path, monkeypatch):
    """Through the real helper: a drive, raw JUnit or checkout beside frames, in their very family, is retained by
    default; only the frames' producer declares its folder, and a bare pytest's own fallback declares nothing."""
    from ouroboros.test_evidence import (
        EVIDENCE_ENV,
        PUBLISHED_DIAGNOSTICS_MARKER,
        diagnostic_directory,
        export_diagnostics,
    )

    run = _evidence_run(tmp_path / "ci-raw" / "host" / "run-first", {})
    monkeypatch.setenv(EVIDENCE_ENV, str(run))
    frames = diagnostic_directory(tmp_path / "case-a", "ui-smoke", publish=True)
    drive, junit = diagnostic_directory(tmp_path / "case-b", "ui-smoke"), diagnostic_directory(tmp_path / "case-c", "ctrl-c")
    for folder, name in ((frames, "page.png"), (drive, "state.json"), (junit, "raw.xml")):
        (folder / name).write_bytes(name.encode())
    assert frames.parent == drive.parent and diagnostic_directory(tmp_path / "case-a", "ui-smoke", publish=True) == frames
    assert sorted(path.name for path in frames.iterdir()) == [PUBLISHED_DIAGNOSTICS_MARKER, "page.png"]
    assert [path.name for path in drive.iterdir()] == ["state.json"], "a retained folder is byte for byte its output"
    assert sorted(path.name for path in frames.parent.iterdir()) == sorted([frames.name, drive.name])
    facts = export_diagnostics(tmp_path / "ci-raw", tmp_path / "out")
    assert facts["errors"] == [] and facts["runs"] == [{"run": "host/run-first", "files": 1, "left": [
        "OUROBOROS_TEST_EVIDENCE_RUN.txt", "ctrl-c"]}]
    assert [path.relative_to(tmp_path / "out").as_posix() for path in (tmp_path / "out").rglob("*") if path.is_file()] == [
        f"host/run-first/ui-smoke/{frames.name}/page.png"]
    monkeypatch.delenv(EVIDENCE_ENV)
    bare = diagnostic_directory(tmp_path / "bare", "ui-smoke", publish=True)
    assert bare == tmp_path / "bare" and list(bare.iterdir()) == []


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="mode 0 denies a listing only to a POSIX non-root")
@pytest.mark.parametrize("locked", ["host/retry-1", "host/run-first/ui/late-input-heads", "host/run-first/server-startup",
                                    f"host/run-first/receipt-facts/{KEY}"],
                         ids=["raw-discovery", "selected-subtree", "family", "unexaminable-producer"])
def test_an_unreadable_folder_is_a_named_gap_and_its_healthy_siblings_still_export(tmp_path, capsys, locked):
    """os.walk skips a folder it cannot list without a word. The export names it, exits 1 (the workflow's
    diagnostics_incomplete; the tests' outcome is their own) and copies every other published diagnostic. A producer
    folder whose declaration cannot be examined is such a gap too, never an absence, and its family's others leave."""
    from ouroboros.test_evidence import _export_main

    raw, out = tmp_path / "ci-raw" / "ui", tmp_path / "ci-diagnostics" / "ui"
    _raw_runs(raw)
    (raw / locked).chmod(0)
    try:
        assert _export_main(["export-diagnostics", str(raw), str(out)]) == 1
    finally:
        (raw / locked).chmod(0o700)
    assert [line.split(": ", 1)[0] for line in capsys.readouterr().out.splitlines()
            if line.startswith("CI_DIAGNOSTICS_INCOMPLETE ")] == [f"CI_DIAGNOSTICS_INCOMPLETE {locked}"]
    exported = {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    assert exported == {name: data for name, data in _EXPORTED.items() if not name.startswith(locked + "/")}


def test_an_unwritable_leaf_is_a_named_gap_and_the_rest_still_exports(tmp_path, capsys):
    from ouroboros.test_evidence import _export_main

    raw, out = tmp_path / "ci-raw" / "ui", tmp_path / "ci-diagnostics" / "ui"
    _raw_runs(raw)
    (out / "host" / "run-first").mkdir(parents=True)
    (out / "host" / "run-first" / "ui").write_text("a file where the run's frames go", encoding="utf-8")
    assert _export_main(["export-diagnostics", str(raw), str(out)]) == 1
    lines = capsys.readouterr().out.splitlines()
    gaps = [line for line in lines if line.startswith("CI_DIAGNOSTICS_INCOMPLETE ")]
    assert [line.split(":", 1)[0] for line in gaps] == [
        "CI_DIAGNOSTICS_INCOMPLETE host/run-first/ui/late-input-heads/receipt.png",
        "CI_DIAGNOSTICS_INCOMPLETE host/run-first/ui/late-input-heads/results.json"]
    assert lines[-1].startswith("CI_DIAGNOSTICS_EXPORTED 6 file(s), ") and "from 3 run(s)" in lines[-1]
    assert (out / "host" / "retry-1" / "run-retry" / "ui" / "late-input-heads" / "receipt.png").read_bytes() == b"retry frame"
    assert _export_main(["export-diagnostics", str(raw)]) == 2  # a malformed call exports nothing


def _run_shell(step, tmp_path, *, diagnostic="viewport", result=0, scope="all"):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow shell unavailable on this host")
    args = tmp_path / "args"
    command = step["run"].replace("${{ runner.temp }}", str(tmp_path)).replace("${{ matrix.shard }}", "2")
    command = command.replace("${{ matrix.shards }}", "4")
    command = command.replace("${{ runner.os == 'Windows' && format('--default-suite-shard={0}/8', matrix.shard) || '' }}",
                              "--default-suite-shard=2/8")
    stub = ('xvfb-run() { test "$1" = -a || return 97; shift; DISPLAY=:ci-test "$@"; }\n'
            'python() { [[ "$NEEDS_DISPLAY" != 1 || "$DISPLAY" == :ci-test ]] || return 97; '
            'printf "%s\\n" "$@" > "$ARGV_FILE"; return "$PRODUCER_EXIT"; }\n')
    env = {**os.environ, "ARGV_FILE": str(args), "PRODUCER_EXIT": str(result),
           "DIAGNOSTIC": diagnostic, "SCOPE": scope,
           "NEEDS_DISPLAY": "1" if step.get("id") == "ui_tests" else "0"}
    completed = subprocess.run([bash, "-e", "-o", "pipefail", "-c", stub + command],
                               env=env, capture_output=True, text=True, encoding="utf-8", timeout=10)
    return completed, args.read_text(encoding="utf-8").splitlines() if args.exists() else []


@pytest.mark.parametrize("result", [0, 17])
@pytest.mark.parametrize("workflow,job,step", [
    (CANARY, "integration-test", "provider_tests"),
    ("ui-browser.yml", "ui-shard", "ui_tests"),
    (*PARTIAL, "ui_diagnostic"),
    ("ui-browser.yml", "ui-shard", "browser_tools"),
    ("ui-browser.yml", "ui-manifest", "ui_manifest"),
    *[("ci.yml", job, f"tests_{label}") for job in ("quick-test", "full-test-runs")
      for label in ("parallel", "serial", "size")],
])
def test_actual_producer_shell_preserves_success_and_failure_exit(tmp_path, workflow, job, step, result):
    completed, args = _run_shell(_steps(workflow, job)[step], tmp_path, result=result)
    assert completed.returncode == result, completed.stderr
    # One producer: pytest itself, or the owned runner that forwards its pytest arguments.
    assert args.count("pytest") + args.count("scripts/run_tests.py") == 1 and "summarize" not in args


OFFICIAL, FORK, TAG = "razzant/ouroboros", "someone/private-copy", "refs/tags/v7.5.2"


@pytest.mark.parametrize("event,ref,repository,flag", [
    ("push", TAG, OFFICIAL, "--enforce"),
    ("workflow_dispatch", TAG, OFFICIAL, "--enforce"),  # A manual run on a tag ref can publish too.
    ("push", "refs/heads/ouroboros", OFFICIAL, ""),
    ("workflow_dispatch", "refs/heads/candidate", OFFICIAL, ""),
    ("schedule", "refs/heads/main", OFFICIAL, ""),
    # A fork without the ten provider keys keeps releasing its own tags.
    ("push", TAG, FORK, ""),
    ("workflow_dispatch", TAG, FORK, ""),
])
def test_release_floor_is_the_last_canary_step_and_enforces_only_on_an_official_release_tag(
        tmp_path, monkeypatch, event, ref, repository, flag):
    from tests.test_platform_ci_events import _value

    job = _workflow(CANARY)["jobs"]["integration-test"]
    floor = job["steps"][-1]
    assert floor["name"] == "Release floor for required provider canaries"
    # Its exit IS the floor for required/all; optional-only contains no required results.
    assert set(floor) == {"name", "if", "env", "run"}
    assert floor["if"] == "${{ !cancelled() && inputs.scope != 'optional' }}"
    assert "secrets." not in str(floor) and list(floor["env"]) == ["FLOOR_ENFORCE"]
    # It reads the report the producer wrote, by the same path string.
    produced = re.search(r'--junitxml=("[^"]+")', _steps(CANARY, "integration-test")["provider_tests"]["run"])
    assert floor["run"] == ("python -m tests.provider_release_floor "
                            f'--junit {produced.group(1)} --summary "$GITHUB_STEP_SUMMARY" $FLOOR_ENFORCE')
    assert _value(floor["env"]["FLOOR_ENFORCE"], event=event, ref=ref, repository=repository) == flag
    # The real shell passes the flag as its own argument, or no argument at all, and keeps the reader's exit.
    monkeypatch.setenv("FLOOR_ENFORCE", flag)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    completed, args = _run_shell(floor, tmp_path, result=1)
    assert completed.returncode == 1, completed.stderr
    assert args[:3] == ["-m", "tests.provider_release_floor", "--junit"] and args[4] == "--summary"
    assert args[3].endswith("/ci-private/provider/results.xml") and args[5].endswith("summary.md")
    assert args[6:] == ([flag] if flag else [])


def test_release_records_every_consumed_gate_and_waives_none():
    from tests.test_release_proof import release_proof

    release = _workflow("ci.yml")["jobs"]["release"]
    assert set(release["needs"]) - {"android-build"} == set(release_proof.RELEASE_GATES)
    assemble = next(step for step in release["steps"] if step.get("name") == "Assemble release proof capsule and notes")
    assert assemble["env"]["RELEASE_GATE_RESULTS"] == " ".join(
        f"{name}=${{{{ needs.{name}.result }}}}" for name in release_proof.RELEASE_GATES)
    assert 'ARGS+=(--gate-result "$GATE")' in assemble["run"]
    assert "--test-verdict release-verdict/test-verdict.json" in assemble["run"]
    # No tag is named anywhere: the v7.7.0 and v7.7.1 owner exceptions are gone with their constants.
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "refs/tags/v7" not in text and "OWNER_EXCEPTION" not in text
    assert not any(name.startswith("OWNER_") for name in vars(release_proof))


@pytest.mark.parametrize("ref,repository", [
    ("refs/tags/v7.7.1", OFFICIAL), ("refs/tags/v7.7.1", FORK), (TAG, OFFICIAL),
])
def test_no_tag_passes_a_red_or_skipped_required_lane(ref, repository):
    from tests.test_windows_signing_release import expression_value

    release = _workflow("ci.yml")["jobs"]["release"]
    green = {"github.ref": ref, "github.repository": repository,
             **{f"needs.{name}.result": "success" for name in release["needs"]}}

    def admitted(results, cancelled=False):
        context = {**green, "cancelled": cancelled, **{f"needs.{job}.result": r for job, r in results.items()}}
        return bool(expression_value(release["if"], context))

    assert admitted({}) and not admitted({}, cancelled=True)
    # The experimental Android pair alone may be red: its assets are then left out, never faked.
    assert admitted({"android-build": "failure"})
    for job in set(release["needs"]) - {"android-build"}:
        for result in ("failure", "cancelled", "skipped"):
            assert not admitted({job: result}), (job, result)
    names = [step.get("name") for step in release["steps"]]
    publish = release["steps"][names.index("Publish verified GitHub Release")]
    assert names.index("Verify remote release tag before publish") < names.index(publish["name"])
    assert "if" not in publish


@pytest.mark.parametrize("selection,target", [
    ("viewport", "tests/test_ui_smoke_playwright.py::test_ui_smoke_live_card_mutations_preserve_viewport"),
    ("inflight", "tests/test_ui_smoke_inflight_indicator.py::test_ui_smoke_chat_inflight_indicator_lifecycle"),
])
def test_partial_selection_passes_exact_target_as_argv(tmp_path, selection, target):
    completed, args = _run_shell(_steps(*PARTIAL)["ui_diagnostic"], tmp_path, diagnostic=selection)
    assert completed.returncode == 0 and target in args
    assert "--require-ui-browser" not in args and "tests/test_browser_tools_smoke.py" not in args
    assert "-k" not in args and "tests/" not in args
    assert not any(arg.startswith("--ui-browser-shard") for arg in args)


def test_unknown_or_shell_like_selection_cannot_execute_or_silently_run_full(tmp_path):
    marker = tmp_path / "injected"
    step = _steps(*PARTIAL)["ui_diagnostic"]
    completed, args = _run_shell(step, tmp_path, diagnostic=f"viewport; touch {marker}")
    assert completed.returncode == 2 and not args and not marker.exists()
    # `full` and an absent selection are no scenario either: nothing here runs the lane.
    for selection in ("full", "", "FULL", "tests/"):
        completed, args = _run_shell(step, tmp_path, diagnostic=selection)
        assert completed.returncode == 2 and not args, selection


def test_every_added_outcome_reference_names_an_already_declared_step():
    pattern = re.compile(r"steps\.([A-Za-z_][A-Za-z0-9_]*)\.")
    for workflow, job in ((CANARY, "integration-test"), *(("ui-browser.yml", job) for job in UI_JOBS),
                          PARTIAL):
        known = set()
        for step in _workflow(workflow)["jobs"][job]["steps"]:
            references = pattern.findall(str(step))
            assert set(references) <= known, (workflow, step.get("name"), references, known)
            if step.get("id"):
                known.add(step["id"])


def _ui_shard_counts(text):
    """Every statement of the UI lane's shard count in the workflow's text, by where it stands."""
    jobs = yaml.safe_load(text)["jobs"]
    shard = jobs["ui-shard"]
    steps = {step["id"]: step for step in shard["steps"] if "id" in step}
    reconcile = next(step for step in jobs["ui-smoke"]["steps"] if step.get("id") == "reconcile")
    matrix = shard["strategy"]["matrix"]["shard"]
    slash = r"\$\{\{ matrix\.shard \}\}/(\d+)"

    def stated(pattern, source):
        return [int(number) for number in re.findall(pattern, source)]

    return {"matrix": [len(matrix)] if matrix == list(range(1, len(matrix) + 1)) else [],
            "job name": stated(slash, shard["name"]),
            "shard argument": stated("--ui-browser-shard=" + slash, steps["ui_tests"]["run"]),
            "summary label": stated('--label "shard ' + slash, steps["ui_summary"]["run"]),
            "reconcile count": stated(r"--count (\d+)", reconcile["run"])}


def test_the_ui_shard_count_is_one_number_wherever_the_workflow_states_it():
    """Nine, by decision on run 42833510999: a partial bump would leave nodes unrun or reconcile the wrong lane."""
    counts = _ui_shard_counts((ROOT / ".github/workflows/ui-browser.yml").read_text(encoding="utf-8"))
    assert all(count == [9] for count in counts.values()), counts


def test_full_lane_runs_as_nine_guarded_shards_under_step_ceilings(tmp_path):
    job = _workflow("ui-browser.yml")["jobs"]["ui-shard"]
    steps = _steps("ui-browser.yml", "ui-shard")
    assert job["needs"] == "scope" and job["if"] == FULL_LANE
    assert job["strategy"] == {"fail-fast": False, "matrix": {"shard": list(range(1, 10))}}
    install, lane, tools = steps["install_browsers"], steps["ui_tests"], steps["browser_tools"]
    # Every producer step has its own ceiling, and their sum stays under the job's: a step
    # ceiling fails the step and leaves the evidence steps running, the job ceiling cancels them.
    ceilings = (install["timeout-minutes"], lane["timeout-minutes"], tools["timeout-minutes"])
    assert ceilings == (45, 90, 20) and sum(ceilings) < job["timeout-minutes"] == 170
    assert "chromium webkit" in install["run"]
    for producer in (lane, tools):
        assert producer["env"]["OUROBOROS_EXPECT_BROWSER_ENGINES"] == "chromium,webkit"
    completed, args = _run_shell(lane, tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert args[args.index("pytest") + 1:][:4] == ["tests/", "-m", "ui_browser", "--require-ui-browser"]
    assert "--ui-browser-shard=2/9" in args and "--session-timeout=4500" in args
    # A hung test fails alone at its per-test timeout (SIGALRM on this Linux runner); nothing narrows the lane.
    assert [arg for arg in args if arg.startswith("--timeout")] == ["--timeout=900"]
    assert not any(arg.startswith(("-k", "--deselect", "--ignore", "--lf", "-x", "-n")) for arg in args)
    # The retry wrapper owns the evidence directory the lane exports into.
    assert args[:6] == ["-I", "-S", "scripts/ci_retry.py", "--evidence-dir", f"{tmp_path}/ci-evidence/ui/host",
                        "--accepted"]
    assert "--ui-browser-shard=${{ matrix.shard }}/9" in lane["run"]
    # Browser tools keep their manual/tag condition, independent of the host lane's outcome, on one shard.
    assert "matrix.shard == 1" in tools["if"] and "steps.ui_tests" not in tools["if"]
    assert "github.event_name == 'workflow_dispatch' || startsWith(github.ref, 'refs/tags/v')" in tools["if"]
    # The real headed visibility fixture must reach an X server; preserving only
    # the pytest argv would miss the Linux setup failure this wrapper repairs.
    assert lane["run"].startswith("xvfb-run -a python ")
    without_display = {**lane, "run": lane["run"].removeprefix("xvfb-run -a ")}
    completed, _args = _run_shell(without_display, tmp_path)
    assert completed.returncode == 97


# The GitHub-published SHA-256 of caddy_2.11.7_linux_amd64.tar.gz; that archive's SHA-512 is the
# line in the release's sigstore-signed caddy_2.11.7_checksums.txt.
CADDY_SHA256 = "727b91701a392de6ebc5027509f548bf39979e5216340d0faed8fa5e69c84f8b"


@pytest.mark.parametrize("pinned", [True, False], ids=["pinned-bytes", "foreign-bytes"])
def test_every_shard_installs_the_digest_pinned_official_caddy_its_lane_needs(tmp_path, pinned):
    """tests/test_separate_ui_browser.py needs a real Caddy. Its skip without one is no reviewed
    platform gate (the required lane fails it), so every shard installs the official release asset
    by the digest GitHub publishes for it and hands that binary to the lane."""
    from tests.browser_lane import permitted_skip
    from tests.separate_ui_support import CADDY_ENV

    job = _workflow("ui-browser.yml")["jobs"]["ui-shard"]
    steps = _steps("ui-browser.yml", "ui-shard")
    install, lane = steps["install_caddy"], steps["ui_tests"]
    order = [step.get("id") for step in job["steps"]]
    assert order.index("setup_python") < order.index("install_caddy") < order.index("ui_tests")
    assert install["env"] == {"CADDY_VERSION": "2.11.7", "CADDY_SHA256": CADDY_SHA256}
    assert "if" not in install and "continue-on-error" not in install
    assert ("https://github.com/caddyserver/caddy/releases/download/v$CADDY_VERSION/"
            "caddy_${CADDY_VERSION}_linux_amd64.tar.gz") in install["run"]
    assert lane["env"][CADDY_ENV] == "${{ steps.install_caddy.outputs.path }}"
    assert sum(step.get("timeout-minutes", 0) for step in job["steps"]) < job["timeout-minutes"]
    assert not permitted_skip("tests/test_separate_ui_browser.py::test_one_tls_origin_routes_each_prefix_to_its_owner"
                              "_and_keeps_the_password_gate", f"set {CADDY_ENV} to an absolute Caddy v2 binary")

    # The step's own shell over a stand-in release: only the pinned bytes give the lane a binary.
    bash, tar = shutil.which("bash"), shutil.which("tar")
    if bash is None or tar is None:
        pytest.skip("workflow shell unavailable on this host")
    release = tmp_path / "release.tar.gz"
    with tarfile.open(release, "w:gz") as archive:
        for name, data, mode in (("caddy", b"#!/bin/sh\necho 'v2.11.7 h1:stand-in'\n", 0o755),
                                 ("LICENSE", b"stand-in", 0o644)):
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), mode
            archive.addfile(member, io.BytesIO(data))
    runner, output = tmp_path / "runner", tmp_path / "github-output"
    runner.mkdir()

    def shell_path(path):
        # The Ubuntu runner hands the step POSIX paths. A Windows (MSYS) bash stand-in gets the
        # same spelling from its own cygpath: GNU tar reads a native `C:\...` as a remote host.
        if os.name != "nt":
            return str(path)
        return subprocess.run([bash, "-c", 'cygpath -u "$1"', "_", str(path)], capture_output=True,
                              text=True, encoding="utf-8", timeout=10, check=True).stdout.strip()

    stub = 'curl() { cp "$STAND_IN_RELEASE" "${@: -1}"; }\npython() { "$PYTHON_BIN" "$@"; }\n'
    env = {**os.environ, "RUNNER_TEMP": shell_path(runner), "GITHUB_OUTPUT": shell_path(output),
           "CADDY_VERSION": "2.11.7",
           "CADDY_SHA256": hashlib.sha256(release.read_bytes()).hexdigest() if pinned else CADDY_SHA256,
           "STAND_IN_RELEASE": shell_path(release), "PYTHON_BIN": sys.executable}
    done = subprocess.run([bash, "-e", "-o", "pipefail", "-c", stub + install["run"]], env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=30)
    binary = runner / "caddy-2.11.7" / "caddy"
    if pinned:
        assert done.returncode == 0, done.stderr
        assert output.read_text(encoding="utf-8") == f"path={shell_path(runner)}/caddy-2.11.7/caddy\n"
        assert [path.name for path in binary.parent.iterdir()] == ["caddy"]
    else:
        assert done.returncode != 0 and "Caddy digest mismatch" in done.stderr, done.stderr
        assert not output.exists() and not binary.exists()


def test_lane_manifest_is_an_unsharded_collection_whose_upload_is_mandatory(tmp_path):
    jobs = _workflow("ui-browser.yml")["jobs"]
    job, steps = jobs["ui-manifest"], _steps("ui-browser.yml", "ui-manifest")
    assert job["needs"] == "scope" and job["if"] == jobs["ui-shard"]["if"] and "strategy" not in job
    for shard_only in ("--ui-browser-shard", "matrix", "playwright install", "--session-timeout"):
        assert shard_only not in str(job), shard_only
    collect, lane = steps["ui_manifest"], _steps("ui-browser.yml", "ui-shard")["ui_tests"]
    completed, args = _run_shell(collect, tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert args[args.index("pytest") + 1:][:5] == [
        "tests/", "-m", "ui_browser", "--require-ui-browser", "--collect-only"]
    assert "safe_test.py" in collect["run"] and "ci-evidence/ui-manifest/host" in collect["run"]
    # The witness collects under the environment the shards collect with.
    for name in ("OUROBOROS_RUN_UI_SMOKE", "OUROBOROS_EXPECT_BROWSER_ENGINES"):
        assert collect["env"][name] == lane["env"][name]
    upload = steps["manifest_evidence"]
    assert upload["uses"] == UPLOAD_ACTION and "if" not in upload
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["path"] == "${{ runner.temp }}/ci-evidence/ui-manifest/"
    assert upload["with"]["name"] == ("ui-ci-full-manifest-${{ github.sha }}-"
                                      "${{ github.run_id }}-${{ github.run_attempt }}")


def _pattern(name):
    """The download pattern that selects every shard and attempt of one upload name."""
    for expression, value in (("matrix.shard", "*"), ("github.run_attempt", "*")):
        name = name.replace(f"${{{{ {expression} }}}}", value)
    return name


def test_aggregator_reconciles_this_runs_shards_against_the_manifest_without_tolerance():
    jobs = _workflow("ui-browser.yml")["jobs"]
    job = jobs["ui-smoke"]
    # Every need is a job that runs whenever browsers do: the aggregator, and through it
    # the caller's `ui-smoke` result a release reads, never waits on an always-skipped job.
    assert job["needs"] == ["scope", "ui-shard", "ui-manifest"]
    assert job["if"] == "${{ !cancelled() }}" and job["timeout-minutes"] == 15
    verdict = _steps("ui-browser.yml", "ui-smoke")["verdict"]
    assert job["steps"][0] == verdict and "if" not in verdict
    assert "${{" not in verdict["run"], "every expression reaches the shell through env"
    assert verdict["env"] == {
        "SCOPE_RESULT": "${{ needs.scope.result }}",
        "RUN_BROWSER": "${{ needs.scope.outputs.run_browser }}",
        "SHARD_RESULT": "${{ needs.ui-shard.result }}",
        "MANIFEST_RESULT": "${{ needs.ui-manifest.result }}"}
    assert jobs["scope"]["outputs"] == {"run_browser": "${{ steps.scope.outputs.run_browser }}"}
    proof = job["steps"][1:]
    # Every proof step runs whenever browsers are needed, whatever the verdict step found:
    # a red shard's unexecuted nodes are named, and a red session is refused by its proof.
    for step in proof:
        assert step["if"] == FULL_LANE.replace("${{ ", "${{ !cancelled() && "), step
    checkout, manifest, shards, reconcile = proof
    assert checkout["uses"].startswith("actions/checkout@")
    assert manifest["uses"] == shards["uses"] == DOWNLOAD_ACTION
    assert manifest["with"]["pattern"] == _pattern(
        _steps("ui-browser.yml", "ui-manifest")["manifest_evidence"]["with"]["name"])
    assert shards["with"]["pattern"] == _pattern(
        _steps("ui-browser.yml", "ui-shard")["ui_evidence"]["with"]["name"])
    assert manifest["with"]["path"] != shards["with"]["path"]
    assert all("merge-multiple" not in step["with"] and "name" not in step["with"] for step in (manifest, shards))
    run = reconcile["run"]
    assert run.startswith("python3 -I -S tests/ci_evidence.py reconcile-shards ")
    assert f'--manifest "{manifest["with"]["path"]}"' in run and f'--shards "{shards["with"]["path"]}"' in run
    assert f'--count {len(jobs["ui-shard"]["strategy"]["matrix"]["shard"])} ' in run
    assert '--sha "${{ github.sha }}" --run-id "${{ github.run_id }}"' in run
    assert "continue-on-error" not in reconcile and "|| true" not in run


def _verdict(tmp_path, **results):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow shell unavailable on this host")
    summary = tmp_path / "summary.md"
    env = {**os.environ, "GITHUB_STEP_SUMMARY": str(summary), "SCOPE_RESULT": "success",
           "RUN_BROWSER": "true", "SHARD_RESULT": "success", "MANIFEST_RESULT": "success", **results}
    completed = subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
         _steps("ui-browser.yml", "ui-smoke")["verdict"]["run"]],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
    text = summary.read_text(encoding="utf-8", errors="replace") if summary.exists() else ""
    return completed.returncode, completed.stdout, text


SKIPPED_LANE = {"SHARD_RESULT": "skipped", "MANIFEST_RESULT": "skipped"}


@pytest.mark.parametrize("results, path, green, said", [
    ({}, "full", True, []),
    ({"SHARD_RESULT": "failure"}, "full", False, ["ui-shard: failure"]),
    ({"RUN_BROWSER": "false", **SKIPPED_LANE}, "documentation", True, []),
], ids=["full-green", "shard-failed", "documentation-only"])
def test_aggregator_verdict_shell_has_two_explicit_paths(tmp_path, results, path, green, said):
    code, stdout, summary = _verdict(tmp_path, **results)
    assert code == int(not green), stdout
    for text in said:
        assert text in summary, summary
    assert ("UI browser lane RED" in summary) is not green
    assert ("::error title=UI browser lane::" in stdout) is not green
    assert ("event changes documentation only." in summary) is (path == "documentation")
    assert "PARTIAL DIAGNOSTIC" not in summary
    if path == "full" and green:
        assert summary == "", "a green full lane is announced by the reconciliation, not by this step"


def test_real_reconciler_fails_the_aggregator_step_when_no_shard_reported(tmp_path):
    """The workflow's own reconcile command, run as written: an empty download is RED."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow shell unavailable on this host")
    summary = tmp_path / "summary.md"
    run = _steps("ui-browser.yml", "ui-smoke")["reconcile"]["run"]
    for expression, value in (("runner.temp", str(tmp_path)), ("github.sha", "c" * 40), ("github.run_id", "31")):
        run = run.replace(f"${{{{ {expression} }}}}", value)
    assert "${{" not in run
    completed = subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
         'python3() { "$REAL_PYTHON" "$@"; }\n' + run],
        cwd=ROOT, env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary), "REAL_PYTHON": sys.executable},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "manifest: no valid projection of the unsharded lane" in completed.stdout
    assert all(f"shard {index}/9: no proof" in completed.stdout for index in range(1, 10))
    assert "INCOMPLETE" in summary.read_text(encoding="utf-8")


EVIDENCE_ACTION = "./.github/actions/test-evidence"
PASSES = ("parallel", "serial", "size")


def _evidence_action():
    return yaml.safe_load((ROOT / EVIDENCE_ACTION / "action.yml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("jobname,artifact", [
    ("quick-test", "quick-test-${{ matrix.shard }}"), ("full-test-runs", "full-test-${{ matrix.os }}-shard-${{ matrix.shard }}"),
])
def test_ordinary_jobs_name_their_failures_without_owning_the_result(jobname, artifact):
    job = _workflow("ci.yml")["jobs"][jobname]
    steps = _steps("ci.yml", jobname)
    for label in PASSES:
        producer = steps[f"tests_{label}"]
        assert "continue-on-error" not in producer
        assert f'--ci-evidence-dir="${{{{ runner.temp }}}}/ci-evidence/{label}"' in producer["run"]
        # A hang fails the step, so the evidence step still runs; a job limit would cancel it.
        assert isinstance(producer["timeout-minutes"], int)
    assert "continue-on-error" not in job and "timeout-minutes" not in job
    evidence = next(step for step in job["steps"] if step.get("uses") == EVIDENCE_ACTION)
    assert evidence["continue-on-error"] is True and "!cancelled()" in evidence["if"]
    assert evidence["with"] == {"name": artifact, "passes": " ".join(
        f"{label}=${{{{ steps.tests_{label}.outcome }}}}" for label in PASSES)}
    assert job["steps"].index(evidence) > max(job["steps"].index(steps[f"tests_{label}"])
                                              for label in PASSES)


@pytest.mark.parametrize("job", ["quick-test", "full-test-runs"])
def test_every_landed_test_producer_runs_under_the_owned_runner(job):
    """One owner per test run: the full passes and the Node lane run through scripts/run_tests.py,
    each pass behind scripts/ci_retry.py, which exports into the pass's own evidence directory."""
    steps = _workflow("ci.yml")["jobs"][job]["steps"]
    producers = {step["id"]: step["run"] for step in steps if step.get("id") in {"tests_parallel", "tests_serial"}}
    node = [step["run"] for step in steps if step.get("name", "").startswith("Run browser-module tests")]
    assert len(producers) == 2
    for step_id, run in producers.items():
        label = step_id.removeprefix("tests_")
        assert run.startswith(f'python scripts/ci_retry.py --evidence-dir "${{{{ runner.temp }}}}/ci-evidence/{label}" '
                              "--accepted tests/accepted_failures.json -- python scripts/run_tests.py tests/ -m "), run
        assert f'--ci-evidence-dir="${{{{ runner.temp }}}}/ci-evidence/{label}"' in run
    assert node == ["python scripts/run_tests.py --node"]


def test_evidence_action_steps_are_independent_diagnostics():
    upload, report = _evidence_action()["runs"]["steps"]
    for step in (upload, report):  # Neither waits for the other's success nor fails the caller.
        assert step["if"] == "${{ !cancelled() }}" and step["continue-on-error"] is True
    assert upload["uses"] == UPLOAD_ACTION
    assert upload["with"]["path"] == "${{ runner.temp }}/ci-evidence/"
    for fact in ("inputs.name", "github.run_id", "github.run_attempt"):
        assert fact in upload["with"]["name"]
    assert report["shell"] == "bash"  # One shell on all three runner systems.
    assert report["env"]["ARTIFACT_OUTCOME"] == "${{ steps.upload.outcome }}"
    assert report["env"]["ARTIFACT_URL"] == "${{ steps.upload.outputs.artifact-url }}"
    assert "python -I -S tests/ci_evidence.py summarize" in report["run"]
    assert "python -I -S tests/ci_evidence.py annotate" in report["run"]
    assert "pytest" not in report["run"] and "${{" not in report["run"]


def _publish(tmp_path, passes):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow shell unavailable on this host")
    summary = tmp_path / "summary.md"
    env = {**os.environ, "PASSES": passes, "ARTIFACT_OUTCOME": "success",
           "ARTIFACT_URL": "https://github.com/example/project/actions/runs/1/artifacts/2",
           "EVIDENCE_ROOT": str(tmp_path / "ci-evidence"), "GITHUB_STEP_SUMMARY": str(summary),
           "REAL_PYTHON": sys.executable}
    script = 'python() { "$REAL_PYTHON" "$@"; }\n' + _evidence_action()["runs"]["steps"][1]["run"]
    completed = subprocess.run([bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
                               cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    errors = [line for line in completed.stdout.splitlines() if line.startswith("::error")]
    return errors, summary.read_text(encoding="utf-8") if summary.exists() else ""


def _final_projection(directory, outcome, error_type=""):
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps({
        "format": 1, "session_exit_code": int(outcome == "failed"), "tests_collected": 1,
        "collection_failures": [], "github": {},
        "reports": [{"nodeid": "t.py::test_a", "phase": "call", "outcome": outcome,
                     "error_type": error_type}]}), encoding="utf-8")


def test_published_evidence_names_failed_killed_and_silent_passes(tmp_path):
    root = tmp_path / "ci-evidence"
    _final_projection(root / "parallel", "failed", "KeyError")
    (root / "serial").mkdir()
    (root / "serial" / "events.jsonl").write_text(
        json.dumps({"event": "start", "nodeid": "t.py::test_b"}) + "\n", encoding="utf-8")
    errors, summary = _publish(tmp_path, "parallel=failure serial=failure size=failure idle=skipped")
    assert errors == [
        "::error title=No test evidence::The size pass failed before any test reported; "
        "read that step's log.",
        "::error title=Failed test::t.py::test_a (call, KeyError)",
        "::error title=Test in flight when the session was killed::t.py::test_b",
    ]
    for label in PASSES:
        assert f"## CI test evidence — {label} pass" in summary
    assert "idle pass" not in summary  # A skipped producer ran nothing and reports nothing.
    assert "| t.py::test_a | call | KeyError |" in summary
    assert "| t.py::test_b |" in summary


def test_published_evidence_is_quiet_for_green_passes(tmp_path):
    _final_projection(tmp_path / "ci-evidence" / "parallel", "passed")
    errors, summary = _publish(tmp_path, "parallel=success serial=skipped size=skipped")
    assert errors == []
    assert "## CI test evidence — parallel pass" in summary and "passed=1, failed=0" in summary
    assert "serial pass" not in summary and "diagnostics_incomplete" not in summary


# ---------------------------------------------------------------- bounded browser install (#2895)

INSTALLER = ROOT / "scripts" / "ci_browser_install.py"
_POSIX_RUNNER = pytest.mark.skipif(os.name == "nt", reason="the install helper runs on the Linux runner: "
                                                          "POSIX process groups, sudo and apt-get")
# One POSIX-shell stand-in for sudo, apt-get and the interpreter, shell builtins only so that it stays fast
# on a loaded host: it logs each call in $FAKE/calls and acts as line N of $FAKE/plan/<kind> says for its
# Nth call (ok by default). A hang keeps the shell and its `sleep` in the command's process group and
# records both pids in $FAKE/hung; a deaf one also ignores SIGTERM. Real sudo keeps root's command beyond
# this user's signals instead; test_a_stopped_download_beyond_this_users_signals_is_refused covers that.
_STAND_IN = r'''#!/bin/sh
role=${0##*/}
if [ "$role" = sudo ]; then
  [ "$1" = -n ] && [ "$2" = -- ] || exit 98
  shift 2
  echo "sudo $*" >> "$FAKE/calls"
  exec "$@"
fi
case "$role $*" in
  "apt-get update") kind=update ;;
  "apt-get install -y --no-install-recommends --download-only "*) kind=download ;;
  "apt-get install -y --no-install-recommends --no-download "*) kind=dpkg ;;
  "python -m playwright install-deps --dry-run "*) kind=resolve ;;
  "python -m playwright install "*) kind=browsers ;;
  "python -m ouroboros.chromium_runtime install") kind=cft ;;
  *) echo "unexpected call: $role $*" >&2; exit 97 ;;
esac
n=0
[ -f "$FAKE/count.$kind" ] && read -r n < "$FAKE/count.$kind"
echo $((n + 1)) > "$FAKE/count.$kind"
echo "$kind $role $*" >> "$FAKE/calls"
act=ok i=0
if [ -f "$FAKE/plan/$kind" ]; then
  while read -r line; do
    [ "$i" -eq "$n" ] && { act=$line; break; }
    i=$((i + 1))
  done < "$FAKE/plan/$kind"
fi
case "$act" in
  ok) [ "$kind" = resolve ] || exit 0
      printf 'Missing system dependencies (2):\n  libasound2t64\n  xvfb\n'; exit 1 ;;
  present) echo "All system dependencies are installed."; exit 0 ;;
  garbled) printf "Failed to install browser dependencies\nError: 'apt-get install -s' exited with code 100:\nE: Unable to locate package xvfb\n"; exit 1 ;;
  fail) exit 100 ;;
  hang|deaf) [ "$act" = deaf ] && trap '' TERM
      sleep 600 & echo "$$ $!" >> "$FAKE/hung"; wait ;;
esac
exit 96
'''
CLEAN = ["update", "resolve", "download", "dpkg", "browsers", "cft"]
METADATA = "packages: apt-get update and its dependency report (playwright install-deps --dry-run)"


@pytest.fixture
def browser_install():
    spec = importlib.util.spec_from_file_location("ci_browser_install_script", INSTALLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    """The stand-ins first on PATH, so no real sudo or apt-get can run; ``plan(kind=[acts])`` scripts them."""
    bin_dir, fake = tmp_path / "bin", tmp_path / "fake"
    bin_dir.mkdir()
    (fake / "plan").mkdir(parents=True)
    for role in ("sudo", "apt-get", "python"):
        (bin_dir / role).write_text(_STAND_IN, encoding="utf-8")
        (bin_dir / role).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE", str(fake))
    assert [shutil.which(role) for role in ("sudo", "apt-get")] == [str(bin_dir / "sudo"), str(bin_dir / "apt-get")]

    def plan(**kinds):
        for kind, acts in kinds.items():
            (fake / "plan" / kind).write_text("".join(f"{act}\n" for act in acts), encoding="utf-8")

    def lines(name):
        return (fake / name).read_text(encoding="utf-8").splitlines() if (fake / name).exists() else []

    return SimpleNamespace(python=str(bin_dir / "python"), dir=fake, plan=plan,
                           calls=lambda: [line for line in lines("calls") if not line.startswith("sudo ")],
                           sudo=lambda: [line for line in lines("calls") if line.startswith("sudo ")],
                           hung=lambda: [int(pid) for line in lines("hung") for pid in line.split()])


def _install(module, stand_ins, capsys, *flags, attempt="600"):
    code = module.main(["--attempt-seconds", attempt, "--python", stand_ins.python, *flags, "chromium", "webkit"])
    return code, capsys.readouterr().out, [line.split()[0] for line in stand_ins.calls()]


def _gone(pid, within=5.0):
    """The pid runs no longer: absent, someone else's again, or a zombie its new parent has yet to reap."""
    deadline = time.monotonic() + within
    while True:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
                               timeout=10).stdout.strip()
        if not state or state.startswith("Z"):
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def _until_hung(stand_ins):
    deadline = time.monotonic() + 60
    while not stand_ins.hung() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert stand_ins.hung(), "the planned hang never started"


@pytest.mark.parametrize("workflow, job, arguments", [
    ("ui-browser.yml", "ui-shard", "--pinned-chromium chromium webkit"), (*PARTIAL, "chromium webkit")],
    ids=["full-lane", "partial"])
def test_ui_browser_installs_bound_each_download_below_their_step_ceiling(browser_install, workflow, job, arguments):
    """#2895: an Ubuntu mirror serving 30 KB/s held `playwright install --with-deps` to the 45-minute step ceiling."""
    install = _steps(workflow, job)["install_browsers"]
    assert install["run"] == f"python -I -S scripts/ci_browser_install.py {arguments}"
    assert install["timeout-minutes"] == 45 and "continue-on-error" not in install
    assert install["env"] == {"PLAYWRIGHT_BROWSERS_PATH": "${{ runner.temp }}/ui-browsers"}
    text = (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
    assert "playwright install" not in text and "--with-deps" not in text and "apt-get" not in text
    # Two bounded stages of two attempts, each stop's three graces included, take at most 41 minutes and
    # leave the rest of the step's unchanged ceiling to dpkg's single run.
    assert browser_install.ATTEMPTS == 2 and browser_install.ATTEMPT_SEC == 600
    bounded = 2 * browser_install.ATTEMPTS * (browser_install.ATTEMPT_SEC + 3 * browser_install.STOP_GRACE_SEC)
    assert bounded <= 41 * 60 < install["timeout-minutes"] * 60


def test_every_ubuntu_browser_install_is_the_bounded_one():
    """The Docker UI lane and the live stand's probe wait on the same mirrors as #2895. The one direct Playwright
    install left is Windows': no `--with-deps`, so no packages to wait for, under its own step ceiling."""
    jobs = _workflow("ci.yml")["jobs"]
    for job, arguments in (("docker-ui-smoke", "chromium webkit"), ("e2e-live", "chromium")):
        install = next(step for step in jobs[job]["steps"] if "ci_browser_install.py" in step.get("run", ""))
        assert install["run"] == f"python -I -S scripts/ci_browser_install.py {arguments}", job
        assert install["timeout-minutes"] == 45 and "continue-on-error" not in install, job
    workflows = sorted((ROOT / ".github/workflows").glob("*.yml"))
    assert workflows and not [path.name for path in workflows if "--with-deps" in path.read_text(encoding="utf-8")]
    windows = _steps("ci.yml", "full-test-runs")["attachment_browser_install"]
    assert "runner.os == 'Windows'" in windows["if"] and windows["timeout-minutes"] == 20
    assert "-m playwright install chromium" in windows["run"]


@_POSIX_RUNNER
@pytest.mark.parametrize("flags, kinds, euid", [(["--pinned-chromium"], CLEAN, 1001), ([], CLEAN[:-1], 0)],
                         ids=["full-lane-through-sudo", "partial-as-root"])
def test_a_clean_install_fetches_every_package_before_one_dpkg_run_and_retries_nothing(
        browser_install, stand_ins, capsys, monkeypatch, flags, kinds, euid):
    monkeypatch.setattr(browser_install.os, "geteuid", lambda: euid)
    code, out, called = _install(browser_install, stand_ins, capsys, *flags)
    assert code == 0, out
    assert called == kinds and "::warning" not in out and "::error" not in out
    packages = "-y --no-install-recommends {} libasound2t64 xvfb"  # Playwright's own apt options and its report
    assert stand_ins.calls()[:5] == [
        "update apt-get update", "resolve python -m playwright install-deps --dry-run chromium webkit",
        "download apt-get install " + packages.format("--download-only"),
        "dpkg apt-get install " + packages.format("--no-download"),
        "browsers python -m playwright install chromium webkit"]
    assert stand_ins.sudo() == ([] if euid == 0 else [
        "sudo apt-get update", "sudo apt-get install " + packages.format("--download-only"),
        "sudo apt-get install " + packages.format("--no-download")])


@_POSIX_RUNNER
@pytest.mark.parametrize("plan, stopped, kinds", [
    ({"download": ["fail"]}, "packages: apt-get install --download-only",
     ["update", "resolve", "download", "download", "dpkg", "browsers", "cft"]),
    ({"update": ["fail"]}, METADATA, ["update", *CLEAN]),
    ({"resolve": ["garbled"]}, METADATA, ["update", "resolve", *CLEAN]),  # The update exited 0 with unusable indices.
    ({"browsers": ["fail"]}, "browsers: playwright install",
     ["update", "resolve", "download", "dpkg", "browsers", "browsers", "cft"]),
    ({"cft": ["fail"]}, "browsers: chromium_runtime install", [*CLEAN, "cft"]),
], ids=["apt-download", "apt-update", "unusable-indices", "playwright-cdn", "chrome-for-testing"])
def test_a_failed_download_gets_one_retry_that_starts_where_it_stopped(browser_install, stand_ins, capsys, plan,
                                                                       stopped, kinds):
    """Only the failed step and the rest of its stage run again: what succeeded keeps its result, and
    dpkg and each browser install still run once. The update and its report are one step, so a report
    the indices left unreadable repeats the update too."""
    stand_ins.plan(**plan)
    code, out, called = _install(browser_install, stand_ins, capsys, "--pinned-chromium")
    assert code == 0, out
    assert called == kinds
    assert [line for line in out.splitlines() if line.startswith("::")] == [
        f"::warning title=Browser install retried::{stopped} failed or outlived the attempt's 600 s; "
        "its one retry starts again there"]


@_POSIX_RUNNER
@pytest.mark.parametrize("plan, refused, kinds", [
    ({"download": ["fail", "fail"]}, "packages: apt-get install --download-only",
     ["update", "resolve", "download", "download"]),
    ({"browsers": ["fail", "fail"]}, "browsers: playwright install", [*CLEAN[:5], "browsers"]),
], ids=["apt-download", "playwright-cdn"])
def test_a_download_that_fails_its_one_retry_too_is_refused(browser_install, stand_ins, capsys, plan, refused, kinds):
    stand_ins.plan(**plan)
    code, out, called = _install(browser_install, stand_ins, capsys, "--pinned-chromium")
    assert code == 1 and called == kinds
    assert out.splitlines()[-1] == f"::error title=Browser install refused::{refused} failed in both attempts of 600 s"


@_POSIX_RUNNER
@pytest.mark.parametrize("act", ["hang", "deaf"], ids=["stops-on-sigterm", "ignores-sigterm"])
def test_a_stop_leaves_no_member_of_the_commands_group(browser_install, stand_ins, monkeypatch, act):
    """SIGTERM stops a hung command with its group; one deaf to SIGTERM is killed after the grace."""
    monkeypatch.setattr(browser_install, "STOP_GRACE_SEC", 0.5)
    stand_ins.plan(update=[act])
    process = subprocess.Popen(["apt-get", "update"], start_new_session=True)
    try:
        _until_hung(stand_ins)
        assert browser_install._stop(process)
        assert len(stand_ins.hung()) == 2 and all(_gone(pid) for pid in stand_ins.hung())
    finally:
        # The negative control can break _stop itself; assertion failure must still clean our group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


@_POSIX_RUNNER
def test_a_stopped_download_beyond_this_users_signals_is_refused(browser_install, stand_ins, capsys, monkeypatch):
    """Without a terminal, real sudo runs root's apt-get in its own process group: this user's SIGTERM
    reaches sudo, which relays it, and SIGKILL stops only sudo (the #2895 VM probe). A member that
    outlives both is named, and no retry starts beside it."""
    real_killpg = os.killpg

    def as_if_root(pgid, sig):  # Every member is root's: alive (EPERM) or gone (ESRCH), never signalled.
        real_killpg(pgid, 0)
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(browser_install, "STOP_GRACE_SEC", 0.5)
    monkeypatch.setattr(browser_install.os, "geteuid", lambda: 1001)  # sudo's case, on a root host too
    monkeypatch.setattr(browser_install.os, "killpg", as_if_root)
    stand_ins.plan(update=["hang"])  # The first command, so that no earlier one spends this attempt.
    try:
        code, out, _ = _install(browser_install, stand_ins, capsys, attempt="2")
        _until_hung(stand_ins)  # No signal reaches it, so it gets there even past the deadline.
    finally:
        for pid in stand_ins.hung():
            os.kill(pid, signal.SIGKILL)
    assert code == 1 and [line.split()[0] for line in stand_ins.calls()] == ["update"] and "::warning" not in out, out
    assert re.fullmatch(r"::error title=Browser install refused::`sudo -n -- apt-get update` outlived SIGTERM and "
                        r"SIGKILL: its process group \d+ still has members this user cannot signal, so no "
                        r"retry starts beside them", out.splitlines()[-1]), out
    assert all(_gone(pid) for pid in stand_ins.hung())


@_POSIX_RUNNER
def test_without_root_or_sudo_nothing_runs(browser_install, stand_ins, capsys, monkeypatch):
    monkeypatch.setattr(browser_install.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(browser_install.shutil, "which", lambda name, *args, **kwargs: None)
    code, out, called = _install(browser_install, stand_ins, capsys)
    assert (code, called) == (1, [])
    assert out.splitlines()[-1] == ("::error title=Browser install refused::packages: apt-get needs root, "
                                    "and this user has no sudo")


@_POSIX_RUNNER
def test_a_dpkg_failure_is_an_installation_failure_and_is_never_retried(browser_install, stand_ins, capsys):
    stand_ins.plan(dpkg=["fail"])
    code, out, called = _install(browser_install, stand_ins, capsys, "--pinned-chromium")
    assert code == 1 and called == ["update", "resolve", "download", "dpkg"] and "::warning" not in out
    assert out.splitlines()[-1] == (
        "::error title=Browser install refused::dpkg: apt-get install --no-download exited 100; "
        "an installation failure is not a download, so it is not retried")


@_POSIX_RUNNER
@pytest.mark.parametrize("reports, kinds, code", [
    (["present"], ["update", "resolve", "browsers"], 0),  # Nothing to fetch or install: browsers only.
    (["garbled", "garbled"], ["update", "resolve", "update", "resolve"], 1),  # Unreadable after its one retry too.
], ids=["all-present", "unreadable-twice"])
def test_the_package_stage_follows_playwrights_own_report(browser_install, stand_ins, capsys, reports, kinds, code):
    stand_ins.plan(resolve=reports)
    result, out, called = _install(browser_install, stand_ins, capsys)
    assert (result, called) == (code, kinds), out
    assert out.count("::warning") == len(reports) - 1
    if code:
        assert "without the report this helper reads (Playwright 1.62)" in out
        assert "Failed to install browser dependencies" in out
        assert "E: Unable to locate package xvfb" in out.splitlines()[-1]  # apt's own reason reaches the refusal


@pytest.mark.parametrize("report, status, expected", [
    ("Missing system dependencies (2):\n  libasound2t64\n  xvfb\n", 1, ["libasound2t64", "xvfb"]),
    ("BEWARE: your OS is not officially supported\nMissing system dependencies (1):\n  libicu74:amd64\n", 1,
     ["libicu74:amd64"]),
    ("All system dependencies are installed.\n", 0, []),
    ("Missing system dependencies (3):\n  libasound2t64\n  xvfb\n", 1, None),  # fewer rows than counted
    ("Missing system dependencies (1):\nxvfb\n", 1, None),  # not the indented list
    ("Missing system dependencies (1):\n  xvfb; reboot\n", 1, None),  # not a package name
    ("Missing system dependencies (0):\n", 1, None),
    ("Missing system dependencies (1):\n  xvfb\n", 0, None),  # Playwright exits 1 when it names packages
    ("All system dependencies are installed.\n", 1, None),
    ("", 0, None),
])
def test_the_dependency_report_is_read_exactly_or_refused(browser_install, report, status, expected):
    if expected is None:
        with pytest.raises(browser_install.Refused, match=r"Playwright 1\.62"):
            browser_install.missing_packages(report, status)
    else:
        assert browser_install.missing_packages(report, status) == expected


@_POSIX_RUNNER
def test_the_entry_point_stops_a_hung_download_group_and_a_sigterm_stops_it_too(stand_ins):
    """The process-boundary witness, under -I -S as the workflow runs it: the deadline stops the hung
    command's whole group and its one retry then installs everything; a SIGTERM to the helper leaves
    no download running."""
    command = [sys.executable, "-I", "-S", str(INSTALLER), "--python", stand_ins.python, "chromium", "webkit"]
    stand_ins.plan(update=["hang"])  # The first command: the attempt's deadline times the hang alone.
    done = subprocess.run([*command, "--attempt-seconds", "6"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [line.split()[0] for line in stand_ins.calls()] == ["update", *CLEAN[:5]]
    assert (f"::warning title=Browser install retried::{METADATA} failed or outlived the "
            "attempt's 6 s; its one retry starts again there") in done.stdout.splitlines()
    assert len(stand_ins.hung()) == 2 and all(_gone(pid) for pid in stand_ins.hung())

    for spent in (*stand_ins.dir.glob("count.*"), stand_ins.dir / "calls", stand_ins.dir / "hung"):
        spent.unlink()
    stand_ins.plan(update=["ok"], download=["hang"])
    helper = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 60
        while not stand_ins.hung() and helper.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stand_ins.hung(), "the planned hang never started"
        helper.send_signal(signal.SIGTERM)
        output, _ = helper.communicate(timeout=60)
        assert helper.returncode == 128 + signal.SIGTERM, output
        assert [line.split()[0] for line in stand_ins.calls()] == ["update", "resolve", "download"]
        assert all(_gone(pid) for pid in stand_ins.hung())
    finally:
        if helper.poll() is None:
            helper.kill()
        helper.wait(timeout=10)
        for pid in stand_ins.hung():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if helper.stdout is not None:
            helper.stdout.close()
