"""The paid `e2e-live` CI job: the live E2E stand (devtools/e2e_live) on a nightly
cron or an OPTED-IN manual dispatch, sized to a $30 cap, skipped honestly without
its secret.

Pinned as a contract, not as text: the job fires only on its OWN cron string or a
dispatch whose `e2e_live` input is true (never a plain dispatch — the pre-tag
3-OS matrix must not spend money — nor push, pull_request, tag, or the keyless
lane's cron); the input changes no other job's gate; the nightly checks out and
seeds the `ouroboros` branch tip (a schedule fires on the default branch, the
promoted release line) while a dispatch seeds its own sha; it names
exactly one secret, `OUROBOROS_E2E_LIVE_OPENROUTER_KEY`, gated through a
non-secret job-level env (GitHub rejects `secrets.*` inside `if:`); a missing
secret is one step-summary line and a green exit, not a red run and not a
pretend run; the stand is invoked with the operator's flag set on a clean
detached seed of the checked-out sha; the run size is FEASIBLE under the cap by
the stand's own worst-case reservation rule computed from the code (a set that
can never be admitted would be a nightly red by construction); artifacts are
uploaded even on failure and never include a lane settings file (0600, carries
the key); and the step summary renders EVERY manifest shape — verdicts on
completion, the typed refusal or error otherwise — and never fails on its own.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import textwrap

import yaml

from devtools.e2e_live.run_live_lanes import RunBudget
from devtools.e2e_live.scenarios import SCENARIOS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
JOB = "e2e-live"
SECRET = "OUROBOROS_E2E_LIVE_OPENROUTER_KEY"
LIVE_CRON = "17 3 * * *"
DISPATCH_INPUT = "e2e_live"
NIGHTLY_REF = "ouroboros"
SKIP_LINE = f"skipped: secret {SECRET} not configured"
TOTAL_BUDGET_USD = 30.0


def _workflow() -> dict:
    return yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))


def _job() -> dict:
    return _workflow()["jobs"][JOB]


def _job_text() -> str:
    ci = CI_PATH.read_text(encoding="utf-8")
    block = re.search(rf"^  {re.escape(JOB)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:$|\Z)", ci, re.MULTILINE | re.DOTALL)
    assert block, f"ci.yml has no `{JOB}:` job"
    return block.group(1)


def _stand_step() -> dict:
    return next(step for step in _job()["steps"] if "run_live_lanes" in str(step.get("run", "")))


def _stand_args() -> dict[str, str | None]:
    """The stand's argv as `{flag: value}` (a bare flag maps to None)."""
    argv = shlex.split(_stand_step()["run"].replace("\\\n", " "))
    assert argv[:3] == ["python", "-m", "devtools.e2e_live.run_live_lanes"], argv[:3]
    args: dict[str, str | None] = {}
    rest = argv[3:]
    while rest:
        flag = rest.pop(0)
        assert flag.startswith("--"), flag
        args[flag] = rest.pop(0) if rest and not rest[0].startswith("--") else None
    return args


def test_the_paid_lane_fires_only_on_its_own_cron_or_an_opted_in_dispatch():
    workflow = _workflow()
    triggers = workflow.get("on") or workflow.get(True)
    crons = [str(entry["cron"]) for entry in triggers["schedule"]]
    assert LIVE_CRON in crons, crons
    # The opt-in: a boolean dispatch input, OFF by default, naming the cost and
    # the secret. A plain `gh workflow run CI --ref <branch>` (the pre-tag 3-OS
    # matrix) therefore never runs the paid lane.
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert list(inputs) == [DISPATCH_INPUT], inputs
    spec = inputs[DISPATCH_INPUT]
    assert spec["type"] == "boolean" and spec["default"] is False, spec
    assert "$30" in spec["description"] and SECRET in spec["description"], spec
    condition = " ".join(str(_job()["if"]).split())
    assert condition == (
        f"(github.event_name == 'workflow_dispatch' && github.event.inputs.{DISPATCH_INPUT} == 'true')"
        f" || (github.event_name == 'schedule' && github.event.schedule == '{LIVE_CRON}')"
    )
    # The input gates THIS job only: no other job reads dispatch inputs, so the
    # `github.event_name == 'workflow_dispatch'` gates elsewhere keep firing on
    # every dispatch exactly as before the block gained inputs.
    for name, job in workflow["jobs"].items():
        if name != JOB:
            assert "inputs" not in str(job.get("if", "")), (name, job.get("if"))
    assert _job()["runs-on"] == "ubuntu-latest"
    # One SM1 lane with --self-mod: the task, the evolution cycle, the absorb
    # wait and two hermetic preflight suites on a 4-vCPU runner.
    assert int(_job()["timeout-minutes"]) >= 120
    # No job downstream of the release chain may wait for a paid nightly lane.
    for name, job in workflow["jobs"].items():
        needs = job.get("needs") or []
        assert JOB not in ([needs] if isinstance(needs, str) else needs), name


def test_the_secret_is_gated_through_a_non_secret_env_and_named_once():
    job = _job()
    assert job["env"]["HAS_E2E_LIVE_KEY"] == f"${{{{ secrets.{SECRET} != '' && 'true' || 'false' }}}}"
    text = _job_text()
    assert re.findall(r"secrets\.([A-Z0-9_]+)", text) == [SECRET, SECRET], text
    for step in job["steps"]:
        assert "secrets." not in str(step.get("if", "")), step
    # The value reaches exactly the stand step, under the NAME the stand reads.
    stand = _stand_step()
    assert stand["env"] == {SECRET: f"${{{{ secrets.{SECRET} }}}}"}
    assert _stand_args()["--key-env"] == SECRET
    for step in job["steps"]:
        if "run_live_lanes" not in str(step.get("run", "")):
            assert SECRET not in str(step.get("env", {})), step


def test_a_missing_secret_is_one_summary_line_and_a_green_exit():
    steps = _job()["steps"]
    skip = next(step for step in steps if SKIP_LINE in str(step.get("run", "")))
    assert skip["if"] == "env.HAS_E2E_LIVE_KEY != 'true'"
    assert f'echo "{SKIP_LINE}" >> "$GITHUB_STEP_SUMMARY"' in skip["run"]
    assert "exit 1" not in skip["run"] and "false" not in skip["run"]
    assert not skip.get("continue-on-error")
    # Every other step (after checkout) is gated the other way: no dependency
    # install, no stand, no upload without the key.
    for step in steps:
        if step is skip or step.get("uses", "").startswith("actions/checkout"):
            continue
        assert "env.HAS_E2E_LIVE_KEY == 'true'" in str(step.get("if", "")), step


def test_the_stand_runs_with_the_operator_flag_set_on_a_clean_seed_of_the_checkout():
    args = _stand_args()
    assert args["--source-repo"] == "$GITHUB_WORKSPACE"
    # HEAD of the checkout, never $GITHUB_SHA: on a schedule GITHUB_SHA names
    # the DEFAULT branch (main) while the checkout below is the ouroboros tip.
    assert args["--seed"] == "HEAD"
    assert args["--out"].startswith("$RUNNER_TEMP/")
    assert args["--self-mod"] is None
    assert float(args["--total-budget"]) == TOTAL_BUDGET_USD
    assert int(args["--task-timeout"]) == 2400 and float(args["--watch-interval"]) == 60
    assert "--stub" not in args and "--profile" not in args and "--model" not in args
    # The seed's `git describe` and the release admission gate read history and tags.
    checkout = _job()["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    # Nightly = the development line's tip; dispatch = the dispatched sha.
    assert checkout["with"] == {
        "ref": f"${{{{ github.event_name == 'schedule' && '{NIGHTLY_REF}' || github.sha }}}}",
        "fetch-depth": 0, "persist-credentials": False,
    }
    # The gate's node lane and the UI probe need node 22 and Chromium, as in ui-smoke.
    steps = _job()["steps"]
    assert any(step.get("uses", "").startswith("actions/setup-node@") for step in steps)
    assert any("playwright install --with-deps chromium" in str(step.get("run", "")) for step in steps)


def test_the_run_size_is_feasible_under_the_cap_by_the_worst_case_reservation_rule():
    """Every requested attempt must be admissible when each earlier attempt spent
    its whole reservation — the stand records an inadmissible one as `not_run`
    and fails the verdict, which would make the nightly red by construction."""
    args = _stand_args()
    per_task, attempts = float(args["--per-task-usd"]), int(args["--attempts"])
    scenarios = [s for s in str(args["--scenarios"]).split(",") if s]
    assert scenarios and set(scenarios) <= set(SCENARIOS), scenarios
    assert 1 <= int(args["--pass-of"]) <= attempts
    assert 1 <= int(args["--lanes"]) <= len(scenarios) * attempts
    # The stand's own rule (per-task x (roots + the self-mod evolution root)), asked from
    # the real ledger with the job's --self-mod, so a rule change re-derives this pin itself.
    budget = RunBudget(TOTAL_BUDGET_USD, per_task, self_mod=args["--self-mod"] is None)
    reservations = [budget.reservation(SCENARIOS[sid].root_tasks, SCENARIOS[sid].expects_absorb)
                    for sid in scenarios for _ in range(attempts)]
    assert sum(reservations) <= TOTAL_BUDGET_USD, (reservations, per_task)
    # ...and the set is MAXIMAL at this fence: one more single-root attempt would
    # not fit. When the reservation factor changes this trips on purpose — the
    # subset (and the arithmetic comment in ci.yml) must be revisited, not left.
    assert sum(reservations) + budget.reservation(1) > TOTAL_BUDGET_USD, (reservations, budget.reservation(1))
    # The job title says which subset runs; the full operator set is SM1,SW1,SK1 x3.
    assert "SM1" in _job()["name"] and "feasible" in _job()["name"]


def test_the_summary_header_and_the_job_comment_state_the_current_reservation_arithmetic():
    """rc.15 review MINOR 4: the summary header carried the retired 2x rule ($360
    for the full set). Its numbers are re-derived here from the stand's own ledger
    (per-task x (roots + the self-mod evolution root)), so a rule change trips
    this pin instead of leaving stale arithmetic in the nightly report; the run's
    OWN reservations are rendered from the manifest's budget_preflight."""
    args = _stand_args()
    budget = RunBudget(TOTAL_BUDGET_USD, float(args["--per-task-usd"]), self_mod=args["--self-mod"] is None)
    chosen = budget.reservation(SCENARIOS["SM1"].root_tasks, SCENARIOS["SM1"].expects_absorb)
    full_set = sum(budget.reservation(SCENARIOS[sid].root_tasks, SCENARIOS[sid].expects_absorb)
                   for sid in ("SM1", "SW1", "SK1")) * 3
    run = _summary_step()["run"]
    assert f"SM1 x1 = ${chosen:.0f}" in run and f"x3 set needs ${full_set:.0f}" in run, run
    assert "budget_preflight" in run, run
    ci = CI_PATH.read_text(encoding="utf-8")
    for retired in ("HARD_STOP_INVERSE", "$360", "2 x per", "$315", "(2 + 2 + 3)"):
        assert retired not in ci, retired


def test_artifacts_upload_even_on_failure_and_never_a_lane_settings_file():
    steps = _job()["steps"]
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "always() && env.HAS_E2E_LIVE_KEY == 'true'"
    paths = [line.strip() for line in str(upload["with"]["path"]).splitlines() if line.strip()]
    root = "${{ runner.temp }}/e2e_live/"
    assert all(path.startswith(root) for path in paths), paths
    rel = [path[len(root):] for path in paths]
    assert "run_manifest.json" in rel and "lanes/*/result.json" in rel
    assert any(path.endswith(".png") for path in rel), rel
    for path in rel:
        assert "**" not in path and "settings" not in path and not path.endswith("/*"), path
    assert upload["with"]["if-no-files-found"] in ("warn", "ignore")
    summary = next(step for step in steps if "GITHUB_STEP_SUMMARY" in str(step.get("run", ""))
                   and SKIP_LINE not in str(step.get("run", "")))
    assert summary["if"] == "always() && env.HAS_E2E_LIVE_KEY == 'true'"
    assert "run_manifest.json" in summary["run"]


def _summary_step() -> dict:
    return next(step for step in _job()["steps"] if "GITHUB_STEP_SUMMARY" in str(step.get("run", ""))
                and SKIP_LINE not in str(step.get("run", "")))


def _run_summary(tmp_path: pathlib.Path, manifest: dict | str | None) -> tuple[int, str, str]:
    """Execute the summary step's inline Python against a run root holding ``manifest``
    (a dict is written as JSON, a str verbatim, None means no manifest at all)."""
    run = _summary_step()["run"]
    heredoc = re.search(r"python - <<'PY'\n(.*?)\n\s*PY\s*$", run, re.DOTALL)
    assert heredoc, run
    code = textwrap.dedent(heredoc.group(1))
    runner_temp = tmp_path / "runner_temp"
    (runner_temp / "e2e_live").mkdir(parents=True)
    if manifest is not None:
        body = json.dumps(manifest) if isinstance(manifest, dict) else manifest
        (runner_temp / "e2e_live" / "run_manifest.json").write_text(body, encoding="utf-8")
    summary = tmp_path / "step_summary.md"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
        env={**os.environ, "RUNNER_TEMP": str(runner_temp), "GITHUB_STEP_SUMMARY": str(summary)},
    )
    return proc.returncode, summary.read_text(encoding="utf-8") if summary.exists() else "", proc.stderr


def test_the_summary_step_renders_every_manifest_shape_and_never_fails(tmp_path):
    """`extra.scenarios` is the requested id LIST from admission until the verdict
    dict is written on completion: a typed refusal (credit_preflight, key_unusable,
    seed_materialize) or a crash leaves the list behind. The summary must render
    the refusal/error then, the verdicts only for the dict, and exit 0 in every
    case — the stand's own exit code is the job's verdict, not this report."""
    assert _summary_step()["if"] == "always() && env.HAS_E2E_LIVE_KEY == 'true'"
    refused = {"extra": {"outcome": "refused", "exit_code": 3, "scenarios": ["SM1"], "seed_ref": "HEAD",
                         "refusal": {"stage": "credit_preflight", "reason": "insufficient_remaining",
                                     "remaining_usd": 4.2, "floor_usd": 30.0}}}
    rc, text, err = _run_summary(tmp_path / "refused", refused)
    assert rc == 0, err
    assert "outcome: refused (exit 3)" in text and "insufficient_remaining" in text, text
    assert "no verdicts" in text and "SM1" in text, text
    assert "reservations:" not in text, text     # no budget_preflight in this manifest: no reservation line

    crashed = {"extra": {"outcome": "crashed", "exit_code": 1, "scenarios": ["SM1"],
                         "error": {"type": "RuntimeError", "message": "lane server never became ready"}},
               "seed": {"resolved_sha": "abc123"}}
    rc, text, err = _run_summary(tmp_path / "crashed", crashed)
    assert rc == 0, err
    assert "outcome: crashed (exit 1)" in text and "lane server never became ready" in text, text
    assert "seed abc123" in text, text

    completed = {"extra": {"outcome": "completed", "exit_code": 0, "seed_describe": "v7.0.0-rc.14-3-gabc",
                           "effective_model": "openrouter/some-model",
                           "scenarios": {"SM1": {"attempts": 1, "passed": 1, "infra_errors": 0, "not_run": 0,
                                                 "verdict": "pass"}},
                           "budget": {"spent_usd": 12.345, "cap_usd": 30.0, "refusals": []},
                           "budget_preflight": {"cap_usd": 30.0, "per_task_usd": 15.0, "self_mod": True,
                                                "scenarios": [{"scenario": "SM1", "root_tasks": 1, "reservation_usd": 30.0,
                                                               "attempts": 1, "worst_case_usd": 30.0, "unreachable": False}],
                                                "worst_case_usd": 30.0, "lanes": 1, "round_worst_case_usd": 30.0,
                                                "unreachable": []},
                           "self_mod": {"lanes": 1, "absorb_unconfirmed": []}}}
    rc, text, err = _run_summary(tmp_path / "completed", completed)
    assert rc == 0, err
    assert "outcome: completed (exit 0)" in text and "seed v7.0.0-rc.14-3-gabc" in text, text
    assert 'SM1: {"attempts": 1' in text and '"verdict": "pass"' in text, text
    assert "budget: spent $12.35 of cap $30.00; refusals 0" in text and "self_mod:" in text, text
    # The run's own reservations are rendered from the manifest's budget_preflight.
    assert "reservations: SM1 $30.00 x 1 (1 root + evolution); worst case $30.00 of cap $30.00 at per-task $15.00" in text, text
    assert "no verdicts" not in text

    rc, text, err = _run_summary(tmp_path / "absent", None)
    assert rc == 0, err
    assert "no run_manifest.json" in text, text

    rc, text, err = _run_summary(tmp_path / "corrupt", "{not json")
    assert rc == 0, err
    assert "summary could not read run_manifest.json" in text, text
