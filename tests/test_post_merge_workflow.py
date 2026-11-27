"""The post-merge workflow's boundaries (owner decisions 3A, 2026-10-21 and 2026-10-28).

It runs after pushes to `ouroboros` in the official repository only, coalesces merges instead of
cancelling a running check, holds exactly the permissions its hand-over needs, uses no secret but
the workflow token, pins every third-party action by commit and installs no project into the
environment its probe worktrees share. Two legs run the same affected tests: Linux on the shipped
interpreter (WAL) and Windows on setup-python's (PERSIST, as installs updated in place); only the
Linux leg runs what does not depend on the system (ruff F, the size check, the cost contract).
"""
from __future__ import annotations

from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / ".github/workflows/post-merge.yml").read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load(TEXT)
WINDOW = WORKFLOW["jobs"]["window"]
JOB = WORKFLOW["jobs"]["post-merge"]
STEPS = {step.get("id") or step.get("uses") or step["name"]: step for step in JOB["steps"]}
LEGS = {leg["leg"]: leg for leg in JOB["strategy"]["matrix"]["include"]}
BROWSER_TEXT = (ROOT / ".github/workflows/post-merge-browser.yml").read_text(encoding="utf-8")
BROWSER = yaml.safe_load(BROWSER_TEXT)


def test_it_runs_after_pushes_to_ouroboros_only_and_in_the_official_repository_only():
    triggers = WORKFLOW.get("on", WORKFLOW.get(True))  # PyYAML's YAML 1.1 spelling of "on"
    assert set(triggers) == {"push", "workflow_dispatch"} and triggers["push"] == {"branches": ["ouroboros"]}
    assert triggers["workflow_dispatch"]["inputs"]["base"]["required"] is True
    # The browser check is a workflow of its own (post-merge-browser.yml; tests/test_post_merge_browser.py), which
    # the `browser-dispatch` job only starts.
    assert set(WORKFLOW["jobs"]) == {"window", "post-merge", "browser-dispatch"}
    assert WINDOW["if"] == JOB["if"] == "github.repository == 'razzant/ouroboros' && github.ref == 'refs/heads/ouroboros'"
    assert JOB["runs-on"] == "${{ matrix.os }}" and JOB["timeout-minutes"] == 165


def test_a_linux_and_a_windows_leg_run_the_journal_modes_fresh_and_updated_installs_run():
    assert JOB["strategy"]["fail-fast"] is False and set(JOB["strategy"]["matrix"]) == {"include"}
    assert LEGS == {
        "linux": {"os": "ubuntu-latest", "leg": "linux", "interpreter": "shipped", "journal": "WAL",
                  "gates": "--lint --size"},
        "windows": {"os": "windows-latest", "leg": "windows", "interpreter": "setup-python", "journal": "PERSIST",
                    "gates": ""}}
    assert JOB["defaults"] == {"run": {"shell": "bash"}}
    journal = STEPS["The stores run ${{ matrix.journal }} here"]
    assert journal["run"].endswith("sys.exit(mode != sys.argv[1])\" ${{ matrix.journal }}")
    # A drifted runner is an annotation, not the run's conclusion, which every install's update letter
    # reads (ouroboros/update_qualification.py); a red test on either system still is that conclusion.
    assert journal["continue-on-error"] is True and "print(f'::error title=Post-merge journal mode::" in journal["run"]
    assert "continue-on-error" not in next(step for step in JOB["steps"] if step.get("name") == "Publish the result")
    # The tests still run after a wrong journal mode: nothing waits on that step.
    order = [step.get("id") or step.get("name") for step in JOB["steps"]]
    later = JOB["steps"][order.index(journal["name"]) + 1:]
    assert all(step.get("if", "").startswith("${{ !cancelled()") for step in later), later
    assert all(journal["name"] not in step.get("if", "") for step in later)


def test_the_windows_leg_is_set_up_as_ci_yml_sets_up_the_same_tests_on_windows():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    full = {step.get("id") or step.get("name"): step for step in ci["jobs"]["full-test-runs"]["steps"]}
    for name in ("posix_tools", "select_posix_tools"):
        assert {key: STEPS[name][key] for key in ("uses", "with", "run", "shell") if key in STEPS[name]} == {
            key: full[name][key] for key in ("uses", "with", "run", "shell") if key in full[name]}, name
    temp = STEPS["Put Windows test temp on the work disk"]
    assert temp["run"] == full["Put Windows test temp on the work disk"]["run"] and temp["if"] == "${{ runner.os == 'Windows' }}"
    assert JOB["env"]["OUROBOROS_PREFLIGHT_TEST_WORKERS"] == ci["jobs"]["full-test-runs"]["env"]["OUROBOROS_PREFLIGHT_TEST_WORKERS"]
    assert "runner.os == 'Windows'" in STEPS["posix_tools"]["if"]
    selected = STEPS["selected"]
    assert "steps.select_posix_tools.outcome == 'success'" in selected["if"] and "steps.setup_node.outcome" in selected["if"]
    # One Windows runner cannot finish a full escalation, which ci.yml runs in eight shards.
    assert selected["env"]["DEFER_FULL"] == "${{ runner.os == 'Windows' && '--defer-full' || '' }}"
    assert "--keep-going $DEFER_FULL --evidence-dir" in selected["run"]


def test_merges_coalesce_behind_a_running_check_that_is_never_cancelled():
    assert WORKFLOW["concurrency"] == {"group": "post-merge-${{ github.ref }}", "cancel-in-progress": False}
    assert "concurrency" not in JOB


def test_the_matrix_uses_one_shared_base_and_keeps_the_same_receipt_on_both_systems():
    assert JOB["needs"] == "window" and WINDOW["steps"][0]["with"] == {"fetch-depth": 0}
    [choose] = [step for step in WINDOW["steps"] if step.get("id") == "base"]
    assert choose["run"] == 'python3 scripts/post_merge_check.py base --evidence-dir "$RUNNER_TEMP/post-merge-window"'
    assert choose["env"]["BEFORE"] == "${{ github.event.before }}"
    assert choose["env"]["INITIAL_BASE"] == "${{ inputs.base }}"
    assert WINDOW["steps"][-1]["name"] == (
        "Post-merge ${{ fromJSON(steps.base.outputs.window).continuous && 'confirmed' || 'provisional' }} "
        "base ${{ steps.base.outputs.base }}")
    assert WINDOW["outputs"] == {"window": "${{ steps.base.outputs.window }}",
                                 "browser": "${{ steps.browser_scope.outputs.applicable }}",
                                 "browser_base": "${{ steps.base.outputs.base }}"}
    assert JOB["env"]["BASE"] == "${{ fromJSON(needs.window.outputs.window).base }}"
    assert STEPS["base"]["env"] == {"WINDOW": "${{ needs.window.outputs.window }}"}
    assert '"$WINDOW" > "$RUNNER_TEMP/$EVIDENCE_NAME/base.json"' in STEPS["base"]["run"]
    assert "post_merge_check.py base" not in str(JOB["steps"])
    for stage in ("growth", "selected", "classify", "attribute"):
        assert '--base "$BASE"' in STEPS[stage]["run"] and "BASE" not in STEPS[stage].get("env", {})


def test_reruns_use_fresh_evidence_and_carry_each_systems_hand_over():
    assert JOB["env"]["EVIDENCE_NAME"] == "post-merge-${{ github.run_id }}-${{ github.run_attempt }}"
    for name in ("base", "growth", "selected", "classify", "attribute", "route", "cost_contract", "publish"):
        assert "$RUNNER_TEMP/$EVIDENCE_NAME" in STEPS[name]["run"]
    assert any(step.get("name") == "Post-merge replay input ${{ steps.base.outputs.replay }}" for step in WINDOW["steps"])


def test_the_evidence_upload_leaves_the_raw_selection_payloads_on_the_runner():
    """Public diagnostic evidence leaves the runner: each pass's exports (results.json, events.jsonl) and the
    stages' records (base, growth, classification, attribution, routing, summary, the retry, recheck and
    cost-contract sessions). The raw selection payloads stay on it — selection.json, whose record still holds
    the complete inventory, the full inventory-*.jsonl, the probe plugins that carry the scope and the warm's
    file list — together with the `.previous-` copies run_tests.py parks from an earlier attempt of the same
    window: no reader of this artifact consumes them, and the hand-off's own dedicated upload carries the
    validated handoff.json."""
    upload = next(s for s in JOB["steps"] if s.get("name") == "Upload the evidence")
    [include, *excluded] = upload["with"]["path"].splitlines()
    assert include == "${{ runner.temp }}/${{ env.EVIDENCE_NAME }}/"
    assert excluded == [
        "!${{ runner.temp }}/${{ env.EVIDENCE_NAME }}/inventory-*.jsonl",
        "!${{ runner.temp }}/${{ env.EVIDENCE_NAME }}/probe/",
        "!${{ runner.temp }}/${{ env.EVIDENCE_NAME }}/selection.json",
        "!${{ runner.temp }}/${{ env.EVIDENCE_NAME }}/selection.previous-*.json",
        "!${{ runner.temp }}/${{ env.EVIDENCE_NAME }}/warm-files.txt",
    ]


def test_the_window_job_starts_the_check_on_the_runners_bare_interpreter():
    """The window job installs nothing before `python3 scripts/post_merge_check.py base`: every module that script
    imports must stay stdlib-only, or no leg runs. A real interpreter without site-packages (-S) and the environment
    (-I) starts it as that job does."""
    import subprocess
    import sys

    assert not any("setup-python" in str(step.get("uses", "")) for step in WINDOW["steps"])
    done = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "scripts" / "post_merge_check.py"), "base", "--help"],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0 and "--evidence-dir" in done.stdout, done.stdout + done.stderr


def test_it_holds_exactly_the_permissions_of_its_hand_over_and_no_secret():
    assert WORKFLOW["permissions"] == {"contents": "read", "actions": "read", "issues": "write",
                                       "pull-requests": "write"}
    assert "permissions" not in JOB
    assert WINDOW["permissions"] == {"contents": "read", "actions": "read"}
    # actions: write starts the browser check (post-merge-browser.yml), in a job that checks out and runs no code.
    dispatch_job = WORKFLOW["jobs"]["browser-dispatch"]
    assert dispatch_job["permissions"] == {"actions": "write"} and dispatch_job["continue-on-error"] is True
    assert not any("uses" in step for step in dispatch_job["steps"])
    assert "secrets." not in TEXT and "secrets:" not in TEXT
    steps = WINDOW["steps"] + JOB["steps"] + dispatch_job["steps"]
    tokens = {name: value for step in steps for name, value in (step.get("env") or {}).items()
              if "TOKEN" in name}
    assert tokens == {"GH_TOKEN": "${{ github.token }}"}  # base, growth and route read GitHub, nothing else does
    assert [step["id"] for step in steps if "GH_TOKEN" in (step.get("env") or {})] == [
        "base", "growth", "route", "browser_dispatch"]
    [dispatch] = dispatch_job["steps"]
    assert "\ngh workflow run post-merge-browser.yml " in dispatch["run"]


def test_every_action_is_pinned_by_commit_or_local():
    uses = [step["uses"] for step in WINDOW["steps"] + JOB["steps"] if "uses" in step]
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", use) or use.startswith("./.github/actions/")
                        for use in uses), uses
    assert sum(use.startswith("./.github/actions/") for use in uses) == 1
    browser = [step for job in BROWSER["jobs"].values() for step in job["steps"]]
    assert all(re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", step["uses"]) or step["uses"].startswith(
        "./.github/actions/") for step in browser if "uses" in step)
    # No checkout of the browser check keeps the token on disk for the tests it runs.
    checkouts = [step for step in browser if step.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == 3 and all(step["with"]["persist-credentials"] is False for step in checkouts)
    assert not any("uses" in step for step in WORKFLOW["jobs"]["browser-dispatch"]["steps"])


def test_the_browser_check_is_a_workflow_of_its_own_that_the_post_merge_readers_never_see():
    """post-merge-browser.yml (owner choice 1, 2026-11-25): started by the window job, it holds no post-merge group,
    and the update letter and the window read post-merge.yml's runs by that file, title and leg names alone."""
    from ouroboros import update_qualification

    triggers = BROWSER.get("on", BROWSER.get(True))
    assert set(triggers) == {"workflow_dispatch", "push"}
    assert {name: value["required"] for name, value in triggers["workflow_dispatch"]["inputs"].items()} == {
        "base": True, "head": True, "budget": False}
    # A push only where this file changes: GitHub then knows the workflow before it reaches the default branch.
    assert triggers["push"] == {"branches": ["ouroboros"], "paths": [".github/workflows/post-merge-browser.yml"]}
    assert update_qualification.POST_MERGE_RUNS == "actions/workflows/post-merge.yml/runs"
    assert not BROWSER["run-name"].startswith(update_qualification.RANGE_TITLE)
    run = {"display_title": BROWSER["run-name"], "head_branch": "ouroboros", "event": "workflow_dispatch",
           "status": "completed", "conclusion": "failure", "head_sha": "a" * 40, "id": 1, "run_attempt": 1}
    assert update_qualification.range_candidate(run) is False
    assert not any(update_qualification._LEG.fullmatch(job.get("name") or "") for job in BROWSER["jobs"].values())
    # One runner (decision 1): a branch's windows wait in one group, FIFO, up to 100 pending (`queue: max`), so a
    # window's pending run is never replaced; the one writer is the one running window's route job.
    assert BROWSER["concurrency"]["queue"] == "max" and BROWSER["concurrency"]["cancel-in-progress"] is False
    assert BROWSER["concurrency"]["group"] == (
        "${{ github.event_name == 'push' && format('post-merge-browser-register-{0}', github.sha) "
        "|| format('post-merge-browser-{0}', github.ref_name) }}")  # a register push never shares the windows' line
    assert not any("concurrency" in job for job in BROWSER["jobs"].values())


def test_the_browser_check_writes_issues_only_from_its_route_job_on_ouroboros():
    jobs = BROWSER["jobs"]
    assert BROWSER["permissions"] == {}
    assert jobs["check"]["permissions"] == {"contents": "read", "issues": "read", "actions": "read"}
    assert jobs["route"]["permissions"] == {"contents": "read", "issues": "write", "pull-requests": "write"}
    assert jobs["route-trial"]["permissions"] == {"contents": "read", "issues": "read", "pull-requests": "read"}
    assert "github.ref == 'refs/heads/ouroboros'" in jobs["route"]["if"]
    assert "github.ref != 'refs/heads/ouroboros'" in jobs["route-trial"]["if"]
    assert "secrets." not in BROWSER_TEXT
    tokens = {(name, step.get("id")) for name, job in jobs.items() for step in job["steps"]
              if "GH_TOKEN" in (step.get("env") or {})}
    assert tokens == {("check", "weights"), ("check", "run"), ("check", "recheck"), ("route", "route"),
                      ("route-trial", "route")}
    # A push only registers the workflow: no window is checked over it (finding 17).
    assert jobs["register"] == {**jobs["register"], "if": "github.event_name == 'push'", "permissions": {}}
    assert not any("uses" in step for step in jobs["register"]["steps"])
    assert "github.event_name == 'workflow_dispatch'" in jobs["check"]["if"]
    # The writer runs this workflow's own revision, never the head's code, and writes only after it verified that
    # base and head lie on ouroboros's first-parent chain (decision 2).
    route = jobs["route"]["steps"]
    assert "ref" not in route[0]["with"] and [step.get("id") for step in route[1:3]] == ["trusted", "route"]
    assert route[2]["if"] == "steps.trusted.outputs.trusted == 'true'"
    assert "git rev-list --first-parent origin/ouroboros" in route[1]["run"] and "git checkout" not in route[1]["run"]


def test_probes_import_their_own_tree_because_no_project_is_installed():
    setup = STEPS["setup_python"]
    assert setup["uses"] == "./.github/actions/setup-python-env"
    assert setup["with"] == {"interpreter": "${{ matrix.interpreter }}", "install-project": "false"}
    assert STEPS["actions/checkout@11d5960a326750d5838078e36cf38b85af677262"]["with"] == {"fetch-depth": 0}
    assert STEPS["setup_node"]["with"] == {"node-version": "22"}


def test_the_steps_check_retry_attribute_hand_over_and_report_in_order():
    order = [step.get("id") for step in JOB["steps"] if step.get("id")]
    assert order == ["setup_python", "setup_node", "posix_tools", "select_posix_tools", "base", "growth", "selected",
                     "classify", "attribute", "route", "cost_contract", "publish"]
    selected = STEPS["selected"]
    assert selected["continue-on-error"] is True  # the retry, not this step, decides FAILED vs FLAKY
    # Each first-parent commit of the window selects: a held window still runs what a later commit reverted.
    assert "scripts/run_tests.py --selected --base \"$BASE\" --each-commit --keep-going" in selected["run"]
    assert "--sequential" not in selected["run"]  # gone with the serial pass's file groups (test diet, 2026-11-05)
    assert "steps.base.outcome == 'success'" in selected["if"]
    for name in ("classify", "attribute", "route"):
        assert f"scripts/post_merge_check.py {name} " in STEPS[name]["run"]
        assert "continue-on-error" not in STEPS[name] and STEPS[name]["if"].startswith("${{ !cancelled() &&")
    # Bisection needs a finished retry; the hand-over also takes what a killed retry or bisection recorded.
    assert "steps.classify.outcome == 'success'" in STEPS["attribute"]["if"]
    assert STEPS["route"]["if"] == "${{ !cancelled() && steps.base.outcome == 'success' }}"
    publish = STEPS["publish"]
    assert publish["run"].startswith("python scripts/post_merge_check.py summary ")
    assert publish["if"] == "${{ !cancelled() }}" and "continue-on-error" not in publish
    # Every step, the job timeout included, ends before GitHub would cancel the run.
    assert sum(step.get("timeout-minutes", 0) for step in JOB["steps"]) < JOB["timeout-minutes"]


def test_the_system_independent_gates_and_the_cost_contract_run_on_the_linux_leg_only():
    classify = STEPS["classify"]["run"]
    assert classify.startswith('python scripts/post_merge_check.py classify --evidence-dir "$RUNNER_TEMP/$EVIDENCE_NAME" '
                               '${{ matrix.gates }} --base "$BASE"')
    assert '--selected-outcome "${{ steps.selected.outcome }}"' in classify
    growth = STEPS["growth"]
    assert "contains(matrix.gates, '--size')" in growth["if"] and "steps.base.outcome == 'success'" in growth["if"]
    assert growth["run"].startswith("python scripts/post_merge_check.py growth ") and growth["timeout-minutes"] == 5
    assert growth["env"] == {"GH_TOKEN": "${{ github.token }}", "REPO": "${{ github.repository }}",
                             "OWNER": "${{ github.repository_owner }}"}
    assert "runner.os == 'Linux'" in STEPS["cost_contract"]["if"]


def test_each_leg_owns_the_issues_and_the_evidence_of_its_system():
    from ouroboros.update_qualification import CONTINUOUS_STEP, MEASURED_STEP, RANGE_TITLE

    assert WORKFLOW["run-name"] == RANGE_TITLE
    assert JOB["name"] == "post-merge (${{ matrix.leg }}) from ${{ fromJSON(needs.window.outputs.window).base }}"
    debt = STEPS["Post-merge untracked ${{ steps.publish.outputs.untracked }}"]
    assert debt["if"] == "${{ !cancelled() && steps.publish.outputs.untracked != '' }}"
    handoff = next(step for step in JOB["steps"] if step.get("name") == "Upload the full hand-over result")
    assert JOB["steps"][-5:] == [STEPS["publish"], handoff, debt, STEPS[MEASURED_STEP], STEPS[CONTINUOUS_STEP]]
    assert handoff["if"] == "${{ !cancelled() && steps.publish.outputs.handoff_file == 'yes' }}"
    assert handoff["with"]["name"] == "post-merge-handoff-${{ matrix.leg }}-${{ github.run_id }}-${{ github.run_attempt }}"
    assert handoff["with"]["path"] == "${{ runner.temp }}/${{ env.EVIDENCE_NAME }}/handoff.json"
    shared = next(step for step in WINDOW["steps"] if step.get("name") == "Upload the full replay input")
    assert shared["if"] == "steps.base.outputs.handoff_file == 'yes'"
    assert shared["with"]["name"] == "post-merge-handoff-window-${{ github.run_id }}-${{ github.run_attempt }}"
    assert shared["with"]["path"] == "${{ runner.temp }}/post-merge-window/handoff.json"
    for step, scope in ((MEASURED_STEP, "push"), (CONTINUOUS_STEP, "chain")):
        assert STEPS[step]["if"] == "${{ !cancelled() && steps.publish.outputs.window_scope == '" + scope + "' }}"
    for name in ("route", "publish"):
        step = STEPS[name]
        assert step["run"].endswith("--attempt ${{ github.run_attempt }} --os ${{ matrix.leg }}"), name
    # The issue bodies name this exact attempt's page, where the red was measured — not the run's page, which
    # mixes attempts (post_merge_routing.py embeds the URL verbatim).
    expected_run_url = ("${{ github.server_url }}/${{ github.repository }}/actions/runs/"
                        "${{ github.run_id }}/attempts/${{ github.run_attempt }}")
    assert STEPS["route"]["env"]["RUN_URL"] == expected_run_url
    upload = next(step for step in JOB["steps"] if step.get("name") == "Upload the evidence")
    assert upload["with"]["name"] == "post-merge-evidence-${{ matrix.leg }}-${{ github.run_id }}-${{ github.run_attempt }}"


def test_nothing_reverts_pushes_or_schedules():
    lowered = TEXT.lower()
    for forbidden in ("git push", "git revert", "git reset", "schedule:", "pull_request"):
        assert forbidden not in lowered, forbidden


def test_cost_contract_runs_whole_module_even_when_selected_tests_fail():
    import shlex

    step = STEPS["cost_contract"]
    args = shlex.split(step["run"])
    assert args[:3] == ["python", "-m", "pytest"]
    assert args[args.index("-m", 3) + 1] == "scale and not integration"
    assert "tests/test_cost_contract.py" in args
    assert "-k" not in args and not any("::" in arg for arg in args)
    assert "--selected" not in args and "continue-on-error" not in step
    assert "!cancelled()" in step["if"] and "selected.outcome" not in step["if"]
    assert "--ci-evidence-dir" in args and step["timeout-minutes"] == 8
    publish = STEPS["publish"]
    assert publish["env"]["COST_CONTRACT_OUTCOME"] == "${{ steps.cost_contract.outcome }}"
    for stage in ("classify", "attribute", "route"):
        assert publish["env"][f"POST_MERGE_{stage.upper()}_OUTCOME"] == "${{ steps." + stage + ".outcome }}"
