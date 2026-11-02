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
JOB = WORKFLOW["jobs"]["post-merge"]
STEPS = {step.get("id") or step.get("uses") or step["name"]: step for step in JOB["steps"]}
LEGS = {leg["leg"]: leg for leg in JOB["strategy"]["matrix"]["include"]}


def test_it_runs_after_pushes_to_ouroboros_only_and_in_the_official_repository_only():
    triggers = WORKFLOW.get("on", WORKFLOW.get(True))  # PyYAML's YAML 1.1 spelling of "on"
    assert triggers == {"push": {"branches": ["ouroboros"]}}  # no tags, PRs, dispatch or schedule
    assert list(WORKFLOW["jobs"]) == ["post-merge"]
    assert JOB["if"] == "github.repository == 'razzant/ouroboros'"
    assert JOB["runs-on"] == "${{ matrix.os }}" and JOB["timeout-minutes"] == 165


def test_a_linux_and_a_windows_leg_run_the_journal_modes_fresh_and_updated_installs_run():
    assert JOB["strategy"]["fail-fast"] is False and set(JOB["strategy"]["matrix"]) == {"include"}
    assert LEGS == {
        "linux": {"os": "ubuntu-latest", "leg": "linux", "interpreter": "shipped", "journal": "WAL",
                  "gates": "--lint --size"},
        "windows": {"os": "windows-latest", "leg": "windows", "interpreter": "setup-python", "journal": "PERSIST",
                    "gates": ""}}
    assert JOB["name"] == "post-merge (${{ matrix.leg }})" and JOB["defaults"] == {"run": {"shell": "bash"}}
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


def test_it_holds_exactly_the_permissions_of_its_hand_over_and_no_secret():
    assert WORKFLOW["permissions"] == {"contents": "read", "actions": "read", "issues": "write",
                                       "pull-requests": "write"}
    assert "permissions" not in JOB
    assert "secrets." not in TEXT and "secrets:" not in TEXT
    tokens = {name: value for step in JOB["steps"] for name, value in (step.get("env") or {}).items()
              if "TOKEN" in name}
    assert tokens == {"GH_TOKEN": "${{ github.token }}"}  # base, growth and route read GitHub, nothing else does
    assert [step["id"] for step in JOB["steps"] if "GH_TOKEN" in (step.get("env") or {})] == ["base", "growth", "route"]


def test_every_action_is_pinned_by_commit_or_local():
    uses = [step["uses"] for step in JOB["steps"] if "uses" in step]
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", use) or use.startswith("./.github/actions/")
                        for use in uses), uses
    assert sum(use.startswith("./.github/actions/") for use in uses) == 1


def test_probes_import_their_own_tree_because_no_project_is_installed():
    setup = STEPS["setup_python"]
    assert setup["uses"] == "./.github/actions/setup-python-env"
    assert setup["with"] == {"interpreter": "${{ matrix.interpreter }}", "install-project": "false"}
    assert STEPS["actions/checkout@11d5960a326750d5838078e36cf38b85af677262"]["with"] == {"fetch-depth": 0}
    assert STEPS["setup_node"]["with"] == {"node-version": "22"}


def test_the_steps_check_retry_attribute_hand_over_and_report_in_order():
    order = [step.get("id") for step in JOB["steps"] if step.get("id")]
    assert order == ["setup_python", "setup_node", "posix_tools", "select_posix_tools", "base", "growth", "selected",
                     "classify", "attribute", "route", "cost_contract"]
    selected = STEPS["selected"]
    assert selected["continue-on-error"] is True  # the retry, not this step, decides FAILED vs FLAKY
    assert "scripts/run_tests.py --selected --base \"$BASE\" --sequential --keep-going" in selected["run"]
    assert selected["env"]["BASE"] == "${{ steps.base.outputs.base }}"
    assert "steps.base.outcome == 'success'" in selected["if"]
    for name in ("classify", "attribute", "route"):
        assert f"scripts/post_merge_check.py {name} " in STEPS[name]["run"]
        assert "continue-on-error" not in STEPS[name] and STEPS[name]["if"].startswith("${{ !cancelled() &&")
    # Bisection needs a finished retry; the hand-over also takes what a killed retry or bisection recorded.
    assert "steps.classify.outcome == 'success'" in STEPS["attribute"]["if"]
    assert STEPS["route"]["if"] == "${{ !cancelled() && steps.base.outcome == 'success' }}"
    publish = JOB["steps"][-1]
    assert publish["run"].startswith("python scripts/post_merge_check.py summary ")
    assert publish["if"] == "${{ !cancelled() }}" and "continue-on-error" not in publish
    # Every step, the job timeout included, ends before GitHub would cancel the run.
    assert sum(step.get("timeout-minutes", 0) for step in JOB["steps"]) < JOB["timeout-minutes"]


def test_the_system_independent_gates_and_the_cost_contract_run_on_the_linux_leg_only():
    classify = STEPS["classify"]["run"]
    assert classify.startswith('python scripts/post_merge_check.py classify --evidence-dir "$RUNNER_TEMP/post-merge" '
                               '${{ matrix.gates }} --base "$BASE"')
    assert STEPS["classify"]["env"]["BASE"] == "${{ steps.base.outputs.base }}"
    growth = STEPS["growth"]
    assert "contains(matrix.gates, '--size')" in growth["if"] and "steps.base.outcome == 'success'" in growth["if"]
    assert growth["run"].startswith("python scripts/post_merge_check.py growth ") and growth["timeout-minutes"] == 5
    assert growth["env"] == {"GH_TOKEN": "${{ github.token }}", "REPO": "${{ github.repository }}",
                             "OWNER": "${{ github.repository_owner }}", "BASE": "${{ steps.base.outputs.base }}"}
    assert "runner.os == 'Linux'" in STEPS["cost_contract"]["if"]


def test_each_leg_owns_the_issues_and_the_evidence_of_its_system():
    for name in ("route", "Publish the result"):
        step = STEPS[name] if name in STEPS else next(step for step in JOB["steps"] if step.get("name") == name)
        assert step["run"].endswith("--os ${{ matrix.leg }}"), name
    upload = next(step for step in JOB["steps"] if step.get("name") == "Upload the evidence")
    assert upload["with"]["name"] == "post-merge-evidence-${{ matrix.leg }}-${{ github.run_id }}-${{ github.run_attempt }}"


def test_nothing_reverts_pushes_or_schedules():
    lowered = TEXT.lower()
    for forbidden in ("git push", "git revert", "git reset", "schedule:", "workflow_dispatch", "pull_request"):
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
