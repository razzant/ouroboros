"""The checks reader behind ``github_read(method="checks")``, driven through the registered tool over a fake ``gh``.

Its report states GitHub's recorded facts and the sources it could not read, never a verdict.
"""

import json
from types import SimpleNamespace

import pytest

from ouroboros.tools import github, github_checks
from tests.test_github_project_target import BASE, MOVED, SHA, _context, _job, _run


def _answer(payload):
    return github.GhResult(True, json.dumps(payload), 0, None, "")


_FORBIDDEN = github.GhResult(False, "⚠️ GH_ERROR: gh: Resource not accessible by personal access token (HTTP 403)",
                             1, 403, "exit")


class FakeChecks:
    """GitHub's answers about one commit; every gh call is recorded with its transport and timeout."""

    def __init__(self, polls=((),), jobs=None, annotations=None, rollup=(), heads=(SHA,), failing=None, fork=False,
                 state="OPEN"):
        self.polls, self.heads = [list(poll) for poll in polls], list(heads)
        self.jobs, self.annotations = jobs or {}, annotations or {}
        self.rollup = None if rollup is None else list(rollup)  # None: GitHub answers null for the rollup.
        self.failing, self.fork, self.calls = failing or {}, fork, []
        self.states = [state] if isinstance(state, str) else list(state)  # One per pull request read, like the heads.

    def __call__(self, args, ctx, timeout=30, input_data=None, *, repo=github._GENERIC_TRANSPORT):
        self.calls.append((list(args), repo, timeout))
        source = {"pr": "rollup" if "statusCheckRollup" in args[-1] else "pr", "run": "runs" if args[1] == "list" else "jobs",
                  "api": "annotations"}[args[0]]
        failure = self.failing.get(source)
        if isinstance(failure, list):  # One entry per call of that source; None lets the call through.
            failure = failure.pop(0) if failure else None
        if failure:
            return failure
        if args[0] == "pr":
            head = self.heads.pop(0) if len(self.heads) > 1 else self.heads[0]
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            pr = {"headRefOid": head, "url": f"{BASE}/pull/7", "isCrossRepository": self.fork, "state": state}
            return _answer({**pr, "statusCheckRollup": self.rollup} if source == "rollup" else pr)
        if source == "runs":
            return _answer(self.polls.pop(0) if len(self.polls) > 1 else self.polls[0])
        if source == "jobs":  # A list is the run's jobs; anything else is the whole answer of `gh run view`.
            jobs = self.jobs.get(int(args[2]), [])
            return _answer({"jobs": jobs} if isinstance(jobs, list) else jobs)
        return _answer(self.annotations.get(int(args[1].split("/")[4]), []))

    def sent(self, kind):
        return [args for args, _repo, _timeout in self.calls if args[0] == kind]


@pytest.fixture
def checks(tmp_path, monkeypatch):
    """Call ``github_read(method="checks")`` over a fake gh, a fixed observation time and a fake clock.

    ``number=`` and ``sha=`` of the retired reader both name the one ``target`` argument."""
    from ouroboros.tools.registry import ToolRegistry

    clock = SimpleNamespace(now=0.0, slept=[], request_cost=0.0)
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    monkeypatch.setattr(github_checks, "utc_now_iso", lambda: "2026-10-02T12:00:00+00:00")
    monkeypatch.setattr(github_checks, "time", SimpleNamespace(
        monotonic=lambda: clock.now,
        sleep=lambda seconds: (clock.slept.append(seconds), setattr(clock, "now", clock.now + seconds))))
    ctx, _ = _context(tmp_path, "queued")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)

    def call(fake, **args):
        def transport(*a, **kw):
            clock.now += clock.request_cost
            return fake(*a, **kw)

        monkeypatch.setattr(github_checks, "_gh_run", transport)
        call = {"method": "checks", **{k: v for k, v in args.items() if k not in ("number", "sha")}}
        if "number" in args:
            call["target"] = args["number"]
        elif "sha" in args:
            call["target"] = args["sha"]
        result = registry.execute_result("github_read", call)
        assert not any(word in result.text.lower() for word in ("passed", "failing")), result.text
        return result

    call.clock = clock
    return call


_RUNS = f"{BASE}/actions/runs"
_HEADER = [f"GitHub checks for commit {SHA}", "Repository: github.example/owner/selected"]
_PR_LINE = f"Pull request: #7 {BASE}/pull/7 (open)"
_SHA_TAIL = ["", "Other checks: not read — a commit SHA target reads GitHub Actions workflow runs only; "
             "third-party checks and commit statuses are read for a pull request number."]
_ACTIONS_ROLLUP = [{"__typename": "CheckRun", "name": "quick-test", "workflowName": "CI", "status": "COMPLETED",
                    "conclusion": "SUCCESS", "detailsUrl": f"{_RUNS}/11/job/54"}]
_RUNNING = {"status": "in_progress", "conclusion": ""}


def _failed_world(**kw):
    """One failed workflow run (a failed job with a failed step and published annotations) beside a skipped one."""
    return FakeChecks(
        polls=[[_run(12, "Mirror", conclusion="skipped", event="pull_request_target"), _run(11, conclusion="failure")]],
        jobs={11: [_job(54, "quick-test"),
                   _job(55, "full-test (windows-latest)", conclusion="failure", steps=[
                       (9, "Install", "completed", "success"), (10, "Run tests (parallel)", "completed", "failure"),
                       (11, "Run tests (serial)", "completed", "success")]),
                   _job(56, "build", conclusion="skipped")],
              12: [_job(60, "redirect", conclusion="skipped", run_id=12)]},
        annotations={55: [
            {"annotation_level": "warning", "title": "", "message": "Node.js 20 is deprecated."},
            {"annotation_level": "failure", "title": "Failed test", "message": "tests/test_x.py::test_y (call, AssertionError)"},
            {"annotation_level": "failure", "title": "", "message": "Process completed with exit code 1."}]}, **kw)


_FAILED_RUN = [
    "", "Runs not completed with success (2):",
    f"- CI (pull_request) run 11 attempt 1: failure {_RUNS}/11",
    "    log of the failed steps: gh run view 11 --log-failed --repo github.example/owner/selected",
    "    jobs: 3 — success 1, failure 1, skipped 1",
    f"    - job full-test (windows-latest) [55]: failure {_RUNS}/11/job/55",
    "        step 10 Run tests (parallel): failure"]
_SKIPPED_RUN = [f"- Mirror (pull_request_target) run 12 attempt 1: skipped {_RUNS}/12", "    jobs: 1 — skipped 1"]


def test_checks_report_states_run_facts_and_no_verdict(checks):
    # A run GitHub calls success can hold skipped jobs: the rollup's job counts state them at no extra call.
    rollup = _ACTIONS_ROLLUP + [{**_ACTIONS_ROLLUP[0], "name": name, "conclusion": "SKIPPED"}
                                for name in ("full-test (ubuntu-latest)", "full-test (windows-latest)", "build")]
    fake = FakeChecks(polls=[[_run(11), _run(12, "UI browser")]], rollup=rollup, fork=True)
    result = checks(fake, number=7)
    assert (result.status, result.code) == ("ok", "OK")
    assert result.text.splitlines() == [
        *_HEADER, f"Pull request: #7 {BASE}/pull/7 (open, head in a fork); head as read by this call",
        "Observed: 2026-10-02T12:00:00+00:00",
        "Sources read: workflow runs; jobs (runs: 0); annotations (jobs: 0); pull request check rollup",
        "Sources unavailable: none",
        "Workflow runs: 2 — success 2",
        "Pull request rollup, GitHub Actions jobs: 4 — success 1, skipped 3",
        "", "Runs completed with success (2):",
        f"- CI (pull_request) run 11 attempt 1: success {_RUNS}/11",
        f"- UI browser (pull_request) run 12 attempt 1: success {_RUNS}/12",
        "", "Other checks: none — every entry of the pull request rollup (4) is a job of a GitHub Actions workflow."]
    # One pull request read fixes the head and carries the rollup; a run completed with success costs no jobs read.
    assert [args[:2] for args, _repo, _timeout in fake.calls] == [["pr", "view"], ["run", "list"]]
    assert fake.calls[1][0][2:4] == ["--commit", SHA] and all(repo == "" for _args, repo, _timeout in fake.calls)


def test_checks_report_names_the_failed_job_its_step_and_annotations(checks):
    fake = _failed_world()
    result = checks(fake, sha=SHA.upper(), repo="github.example/owner/selected")
    assert (result.status, result.code) == ("ok", "OK")
    assert result.text.splitlines() == [
        *_HEADER, "Observed: 2026-10-02T12:00:00+00:00",
        "Sources read: workflow runs; jobs (runs: 2); annotations (jobs: 1)",
        "Sources unavailable: none",
        "Workflow runs: 2 — failure 1, skipped 1",  # Skipped is its own state, never added to success.
        *_FAILED_RUN,
        "        annotation failure [Failed test]: tests/test_x.py::test_y (call, AssertionError)",
        "        annotation failure: Process completed with exit code 1.",
        "        annotations: 2 at failure level, 2 shown; 3 of all levels read",
        *_SKIPPED_RUN, *_SHA_TAIL]
    # Every repository command carries the explicit target; the annotation read, sent once every run's jobs are
    # known, is the one literal API path, sent on the generic transport with the host GitHub returned.
    assert [(args[0], repo) for args, repo, _timeout in fake.calls] == [
        ("run", "github.example/owner/selected"), ("run", "github.example/owner/selected"),
        ("run", "github.example/owner/selected"), ("api", github._GENERIC_TRANSPORT)]
    assert fake.sent("api") == [["api", "repos/owner/selected/check-runs/55/annotations?per_page=100",
                                 "--hostname", "github.example"]]


def test_checks_report_keeps_unfinished_and_unregistered_apart_from_results(checks):
    running = FakeChecks(polls=[[_run(11, **_RUNNING)]], jobs={11: [
        _job(54, "quick-test"),
        _job(55, "full-test (macos-latest)", steps=[(10, "Run tests", "in_progress", ""), (11, "Serial", "pending", "")], **_RUNNING),
        _job(56, "ui-smoke", status="queued", conclusion="")]})
    assert checks(running, sha=SHA).text.splitlines()[2:] == [
        "Observed: 2026-10-02T12:00:00+00:00",
        "Sources read: workflow runs; jobs (runs: 1); annotations (jobs: 0)",
        "Sources unavailable: none",
        "Workflow runs: 1 — in_progress 1",
        "", "Runs not completed with success (1):",
        f"- CI (pull_request) run 11 attempt 1: in_progress {_RUNS}/11",
        "    jobs: 3 — success 1, queued 1, in_progress 1",
        f"    - job full-test (macos-latest) [55]: in_progress {_RUNS}/11/job/55",
        "        step 10 Run tests: in_progress",
        f"    - job ui-smoke [56]: queued {_RUNS}/11/job/56",
        *_SHA_TAIL]
    assert running.sent("api") == []  # An unfinished job has no annotations to read yet.

    nothing = checks(FakeChecks(), sha=SHA)
    assert nothing.status == "ok" and nothing.text.splitlines() == [
        f"GitHub checks for commit {SHA}", "Repository: resolved by the GitHub CLI from the Project directory",
        "Observed: 2026-10-02T12:00:00+00:00",
        "Sources read: workflow runs; jobs (runs: 0); annotations (jobs: 0)",
        "Sources unavailable: none",
        "Workflow runs: 0 — no workflow run is registered for this commit — this is not a test result",
        "A commit SHA the repository does not hold and a commit with no run yet read the same; the repository is the one "
        "the GitHub CLI resolves from the Project directory (pass repo='[HOST/]OWNER/REPO' to name it).",
        *_SHA_TAIL]
    named = checks(FakeChecks(), sha=SHA, repo="owner/selected").text.splitlines()
    assert named[1] == "Repository: owner/selected"
    assert named[6] == "A commit SHA the repository does not hold and a commit with no run yet read the same."
    # A pull request head is a commit GitHub holds, and a commit with a run is known: neither carries the line.
    assert "read the same" not in checks(FakeChecks(rollup=_ACTIONS_ROLLUP), number=7).text
    assert "read the same" not in checks(FakeChecks(polls=[[_run(11)]]), sha=SHA).text

    # A run that failed before it created a job is still a run with its own state.
    jobless = checks(FakeChecks(polls=[[_run(11, conclusion="startup_failure")]]), sha=SHA).text.splitlines()
    assert jobless[5:] == [
        "Workflow runs: 1 — startup_failure 1",
        "", "Runs not completed with success (1):",
        f"- CI (pull_request) run 11 attempt 1: startup_failure {_RUNS}/11",
        "    jobs: 0 — GitHub lists no job for this run",  # No step ran: there is no failed-step log to point at.
        *_SHA_TAIL]
    # An answer of `gh run view` without a jobs list is an unreadable source, never "0 jobs".
    odd = checks(FakeChecks(polls=[[_run(11, conclusion="failure")]], jobs={11: {}}), sha=SHA).text.splitlines()
    assert odd[3:5] == ["Sources read: workflow runs; jobs (runs: 0); annotations (jobs: 0)",
                        "Sources unavailable: jobs (runs: 1; the answer holds no jobs list)"]
    assert odd[-3] == "    jobs: unavailable (the answer holds no jobs list)" and "jobs: 0 —" not in "\n".join(odd)


_TIME_LIMIT = {"annotation_level": "failure", "title": "",
               "message": "The job has exceeded the maximum execution time of 2h0m0s"}


def test_checks_report_keeps_cancelled_and_skipped_under_their_own_names(checks):
    """GitHub concludes a job stopped at its time limit `cancelled`, its step too, and names the cause only in a
    failure annotation: the job keeps its line and that annotation, while a skipped job stays in the counts."""
    fake = FakeChecks(polls=[[_run(11, conclusion="cancelled"), _run(12, "Mirror", conclusion="skipped")]], jobs={
        11: [_job(54, "ui-smoke", conclusion="cancelled", steps=[(7, "Run", "completed", "cancelled")]),
             _job(55, "build", conclusion="skipped")],
        12: [_job(60, "redirect", conclusion="skipped", run_id=12)]}, annotations={54: [_TIME_LIMIT]})
    lines = checks(fake, sha=SHA).text.splitlines()
    assert lines[5:] == [
        "Workflow runs: 2 — cancelled 1, skipped 1",
        "", "Runs not completed with success (2):",
        f"- CI (pull_request) run 11 attempt 1: cancelled {_RUNS}/11",  # A cancelled step leaves no failed-step log.
        "    jobs: 2 — cancelled 1, skipped 1",
        f"    - job ui-smoke [54]: cancelled {_RUNS}/11/job/54",
        "        step 7 Run: cancelled",
        "        annotation failure: The job has exceeded the maximum execution time of 2h0m0s",
        "        annotations: 1 at failure level, 1 shown; 1 of all levels read",
        f"- Mirror (pull_request) run 12 attempt 1: skipped {_RUNS}/12",
        "    jobs: 1 — skipped 1",
        *_SHA_TAIL]
    assert "Runs completed with success" not in "\n".join(lines)
    assert [args[1].split("/")[4] for args in fake.sent("api")] == ["54"]  # Skipped jobs cost no annotation read.


def test_failed_jobs_of_every_run_are_annotated_before_cancelled_ones(checks):
    # Within a run a cancelled job follows the failed and the unfinished ones.
    jobs = [_job(54, "ui-smoke", conclusion="cancelled"), _job(55, "macos", **_RUNNING),
            _job(56, "lint", conclusion="failure")]
    lines = checks(FakeChecks(polls=[[_run(11, **_RUNNING)]], jobs={11: jobs}), sha=SHA).text.splitlines()
    assert [line.split()[2] for line in lines if line.startswith("    - job ")] == ["lint", "macos", "ui-smoke"]

    # Fifteen cancelled jobs of a run listed first leave a failed job of a later run its annotations.
    cancelled = [_job(100 + index, f"shard {index}", conclusion="cancelled", steps=[(7, "Run", "completed", "cancelled")])
                 for index in range(15)]
    fake = FakeChecks(polls=[[_run(11, conclusion="cancelled"), _run(12, "Lint")]],
                      jobs={11: cancelled, 12: [{**_CONTINUED[3], "url": f"{_RUNS}/12/job/57"}]},
                      annotations={**_LINT_ANNOTATION, **{100 + index: [_TIME_LIMIT] for index in range(15)}})
    lines = checks(fake, sha=SHA).text.splitlines()
    assert lines.index(f"- CI (pull_request) run 11 attempt 1: cancelled {_RUNS}/11") < lines.index(
        f"- Lint (pull_request) run 12 attempt 1: success {_RUNS}/12")
    read = [args[1].split("/")[4] for args in fake.sent("api")]
    assert read == ["57"] + [str(100 + index) for index in range(9)]
    lint = lines.index(f"    - job lint [57]: failure {_RUNS}/12/job/57")
    assert lines[lint + 2] == "        annotation failure: ruff: F821"
    assert lines.count("        annotation failure: The job has exceeded the maximum execution time of 2h0m0s") == 9
    assert lines[lines.index(f"    - job shard 9 [109]: cancelled {_RUNS}/11/job/109") + 2] == (
        "        annotations: not read (one call reads the annotations of 10 jobs)")
    assert "    5 more jobs not shown: cancelled 5" in lines


def test_checks_report_lists_third_party_checks_and_statuses_of_a_pull_request(checks):
    rollup = _ACTIONS_ROLLUP + [
        {"__typename": "CheckRun", "name": "Vercel Agent Review", "workflowName": "", "status": "COMPLETED",
         "conclusion": "NEUTRAL", "detailsUrl": "https://vercel.example/github"},
        {"__typename": "StatusContext", "context": "deploy/netlify", "state": "PENDING", "targetUrl": "https://netlify.example/7"}]
    lines = checks(FakeChecks(polls=[[_run(11)]], rollup=rollup), number=7).text.splitlines()
    assert lines[-4:] == [
        "", "Other checks, outside GitHub Actions (2) — neutral 1, pending 1:",
        "- commit status deploy/netlify: pending https://netlify.example/7",
        "- check run Vercel Agent Review: neutral https://vercel.example/github"]
    assert "Workflow runs: 1 — success 1" in lines  # Other checks never enter the workflow-run counts.
    # A bare commit names the gap instead of reporting an empty list.
    assert checks(FakeChecks(polls=[[_run(11)]], rollup=rollup), sha=SHA).text.splitlines()[-2:] == _SHA_TAIL


def test_an_unavailable_source_is_named_while_the_others_are_reported(checks):
    refused = "HTTP 403: gh: Resource not accessible by personal access token (HTTP 403)"
    fake = FakeChecks(polls=[[_run(11)]], rollup=_ACTIONS_ROLLUP, failing={"rollup": _FORBIDDEN},
                      jobs={11: [_job(54, "quick-test"), _job(56, "build", conclusion="skipped")]})
    result = checks(fake, number=7)
    lines = result.text.splitlines()
    assert result.status == "ok" and f"Sources unavailable: pull request check rollup ({refused})" in lines
    # Without the rollup's job counts the run GitHub calls success is read for its own.
    assert "Sources read: workflow runs; jobs (runs: 1); annotations (jobs: 0)" in lines
    assert lines[-6:] == ["", "Runs completed with success (1):", f"- CI (pull_request) run 11 attempt 1: success {_RUNS}/11",
                          "    jobs: 2 — success 1, skipped 1",
                          "", "Other checks: not read — the pull request check rollup is unavailable."]
    # The head is read again without the refused field: the rollup is the only loss.
    assert [args[-1] for args in fake.sent("pr")] == ["headRefOid,url,isCrossRepository,state,statusCheckRollup",
                                                       "headRefOid,url,isCrossRepository,state"]

    result = checks(_failed_world(failing={"annotations": _FORBIDDEN}), sha=SHA)
    assert result.status == "ok" and result.text.splitlines()[3:] == [
        "Sources read: workflow runs; jobs (runs: 2); annotations (jobs: 0)",
        f"Sources unavailable: annotations (jobs: 1; {refused})",
        "Workflow runs: 2 — failure 1, skipped 1",
        *_FAILED_RUN, f"        annotations: unavailable ({refused})", *_SKIPPED_RUN, *_SHA_TAIL]

    slow = github.GhResult(False, "⚠️ GH_TIMEOUT: exceeded 30s.", None, None, "timeout")
    result = checks(_failed_world(failing={"jobs": slow}), sha=SHA)
    assert result.status == "ok" and result.text.splitlines()[3:] == [
        "Sources read: workflow runs; jobs (runs: 0); annotations (jobs: 0)",
        "Sources unavailable: jobs (runs: 2; timeout)",
        "Workflow runs: 2 — failure 1, skipped 1",
        "", "Runs not completed with success (2):",
        f"- CI (pull_request) run 11 attempt 1: failure {_RUNS}/11",
        "    log of the failed steps: gh run view 11 --log-failed --repo github.example/owner/selected",
        "    jobs: unavailable (timeout)",
        _SKIPPED_RUN[0], "    jobs: unavailable (timeout)", *_SHA_TAIL]


def test_only_a_failing_runs_source_is_a_typed_error(checks):
    gone = github.GhResult(False, "⚠️ GH_ERROR: HTTP 404: Not Found (https://api.github.example/repos/owner/selected/actions/runs)",
                           1, 404, "exit")
    result = checks(FakeChecks(polls=[[_run(11)]], failing={"runs": gone}), sha=SHA)
    assert result.status == "error" and result.text == gone.text
    result = checks(FakeChecks(polls=[[_run(11)]], failing={"runs": github.GhResult(True, "not json", 0, None, "")}), sha=SHA)
    assert (result.status, result.code) == ("error", "TOOL_ERROR") and "workflow runs JSON" in result.text
    assert "operation_outcome" not in result.meta  # After a gh launch a refusal attests nothing about effects.
    # A pull request whose head cannot be read leaves no commit to report on.
    fake = FakeChecks(polls=[[_run(11)]], failing={"rollup": gone, "pr": gone})
    result = checks(fake, number=7)
    assert result.status == "error" and result.text == gone.text and fake.sent("run") == []
    # A pull request answer without a full head SHA is refused before any run is read.
    fake = FakeChecks(polls=[[_run(11)]], heads=["not-a-sha"])
    result = checks(fake, number=7)
    assert (result.status, result.code) == ("error", "TOOL_ERROR") and fake.sent("run") == []
    assert result.text == "⚠️ TOOL_ERROR: GitHub returned no head commit for pull request #7."
    # The same refusal on a secondary source is a named gap, not an error.
    assert checks(_failed_world(failing={"jobs": gone}), sha=SHA).status == "ok"


def test_wait_ends_when_every_run_completes_or_at_its_cap(checks):
    polls = [[_run(11, status="queued", conclusion="")], [_run(11, **_RUNNING)], [_run(11)]]
    fake = FakeChecks(polls=polls, rollup=_ACTIONS_ROLLUP)
    waited = checks(fake, number=7, wait_seconds=120).text.splitlines()
    assert checks.clock.slept == [15, 15] and len(fake.sent("run")) == 3
    assert waited[2:4] == [_PR_LINE + "; head unchanged after the wait",
                           "Observed: 2026-10-02T12:00:00+00:00; waited 30s of 120s for the runs to complete"]
    # A waited call reports what a later call without a wait reports.
    checks.clock.slept.clear()
    later = checks(FakeChecks(polls=polls[-1:], rollup=_ACTIONS_ROLLUP), number=7).text.splitlines()
    assert checks.clock.slept == [] and waited[4:] == later[4:]

    # The requested wait is capped; an unfinished run is reported as unfinished when the time is up.
    checks.clock.now, fake = 0.0, FakeChecks(polls=[[_run(11, **_RUNNING)]], jobs={11: [_job(55, "full-test", **_RUNNING)]})
    lines = checks(fake, sha=SHA, wait_seconds=999).text.splitlines()
    assert sum(checks.clock.slept) == 240 and max(checks.clock.slept) == 15
    assert "Observed: 2026-10-02T12:00:00+00:00; waited 240s of 240s for the runs to complete" in lines
    assert "Workflow runs: 1 — in_progress 1" in lines and f"    - job full-test [55]: in_progress {_RUNS}/11/job/55" in lines

    # Runs that are complete at the first read need no wait, and the pull request is read once.
    checks.clock.now, checks.clock.slept[:] = 0.0, []
    fake = FakeChecks(polls=polls[-1:], rollup=_ACTIONS_ROLLUP, state="MERGED")
    ready = checks(fake, number=7, wait_seconds=120).text.splitlines()
    assert checks.clock.slept == [] and len(fake.sent("pr")) == 1 and len(fake.sent("run")) == 1
    assert ready[2:4] == [f"Pull request: #7 {BASE}/pull/7 (merged); head as read by this call",
                          "Observed: 2026-10-02T12:00:00+00:00; waited 0s of 120s for the runs to complete"]
    assert ready[4:] == later[4:]
    checks.clock.now = 0.0
    fake = FakeChecks(polls=polls, rollup=_ACTIONS_ROLLUP)
    checks(fake, number=7, wait_seconds=120)
    assert len(fake.sent("pr")) == 2  # A wait that happened reads the head again.

    # A commit with no registered run keeps the wait: a run that registers late is still waited for.
    checks.clock.now, checks.clock.slept[:] = 0.0, []
    lines = checks(FakeChecks(polls=[[], [_run(11, **_RUNNING)], [_run(11)]]), sha=SHA, wait_seconds=60).text.splitlines()
    assert checks.clock.slept == [15, 15] and "Workflow runs: 1 — success 1" in lines

    # The last sleep is cut to what is left of the wait, and slow requests never report more waiting than the wait.
    checks.clock.now, checks.clock.slept[:] = 0.0, []
    assert "; waited 20s of 20s for" in checks(FakeChecks(polls=[[_run(11, **_RUNNING)]]), sha=SHA, wait_seconds=20).text
    assert checks.clock.slept == [15, 5]
    checks.clock.now, checks.clock.slept[:], checks.clock.request_cost = 0.0, [], 4.0
    text = checks(FakeChecks(polls=[[_run(11, **_RUNNING)]]), sha=SHA, wait_seconds=20).text
    assert "; waited 20s of 20s for" in text and checks.clock.slept == [15] and checks.clock.now > 20

    # A run-list read that fails during the wait ends the call with that error.
    checks.clock.now, checks.clock.request_cost = 0.0, 0.0
    slow = github.GhResult(False, "⚠️ GH_TIMEOUT: exceeded 30s.", None, None, "timeout")
    fake = FakeChecks(polls=[[_run(11, **_RUNNING)]], failing={"runs": [None, slow]})
    result = checks(fake, sha=SHA, wait_seconds=60)
    assert result.status == "timeout" and result.text == slow.text and len(fake.sent("run")) == 2


def test_one_deadline_bounds_every_request(checks):
    fake = _failed_world()
    checks.clock.request_cost = 40.0  # Each request spends 40 s of the 90 s a call without a wait has.
    lines = checks(fake, sha=SHA).text.splitlines()
    # The runs and the first jobs read get the per-request ceiling, the jobs read of the second run gets what is
    # left, and the annotation read, which waits for every run's jobs, is never sent.
    assert [(args[0], timeout) for args, _repo, timeout in fake.calls] == [("run", 30), ("run", 30), ("run", 10)]
    assert "Sources unavailable: annotations (jobs: 1; deadline)" in lines and lines[-4:-2] == _SKIPPED_RUN
    assert "        annotations: unavailable (deadline)" in lines
    assert lines[:2] == _HEADER and "Workflow runs: 2 — failure 1, skipped 1" in lines
    # With time left every request is sent under the per-request ceiling.
    checks.clock.now, checks.clock.request_cost, fake = 0.0, 0.0, _failed_world()
    assert "deadline" not in checks(fake, sha=SHA).text
    assert [timeout for _args, _repo, timeout in fake.calls] == [30, 30, 30, 30] and len(fake.sent("api")) == 1


def test_a_head_that_moves_during_the_wait_is_named(checks):
    fake = FakeChecks(polls=[[_run(11, **_RUNNING)], [_run(11, conclusion="cancelled")]], heads=[SHA, MOVED],
                      rollup=_ACTIONS_ROLLUP, jobs={11: [_job(55, "full-test", conclusion="cancelled")]},
                      state=["OPEN", "CLOSED"])
    lines = checks(fake, number=7, wait_seconds=60).text.splitlines()
    # The state is the one read after the wait.
    assert lines[:3] == [*_HEADER, f"Pull request: #7 {BASE}/pull/7 (closed); head moved to {MOVED} during the wait; "
                                   f"this report is for {SHA}"]
    assert "Sources unavailable: pull request check rollup (GitHub's rollup describes the moved head)" in lines
    assert lines[-1] == "Other checks: not read — the pull request check rollup is unavailable."
    assert all(args[3] == SHA for args in fake.sent("run") if args[1] == "list")  # The SHA is fixed at the start.
    # A head that stays put keeps its rollup.
    checks.clock.now = 0.0
    steady = FakeChecks(polls=[[_run(11, **_RUNNING)], [_run(11)]], rollup=_ACTIONS_ROLLUP)
    lines = checks(steady, number=7, wait_seconds=60).text.splitlines()
    assert lines[2] == _PR_LINE + "; head unchanged after the wait" and lines[-1].startswith("Other checks: none")
    # A head that cannot be read after the wait is named as not read, never as unchanged.
    checks.clock.now = 0.0
    refused = "HTTP 403: gh: Resource not accessible by personal access token (HTTP 403)"
    blind = FakeChecks(polls=[[_run(11, **_RUNNING)], [_run(11)]], rollup=_ACTIONS_ROLLUP,
                       failing={"rollup": _FORBIDDEN, "pr": [None, _FORBIDDEN]})
    lines = checks(blind, number=7, wait_seconds=60).text.splitlines()
    assert lines[2] == _PR_LINE + "; head after the wait not read" and "unchanged" not in "\n".join(lines)
    assert (f"Sources unavailable: pull request check rollup ({refused}); "
            f"pull request head after the wait ({refused})") in lines
    assert lines[-1] == "Other checks: not read — the pull request check rollup is unavailable."


def test_the_source_lines_name_every_source_and_the_other_header_lines_are_clipped(checks):
    words = "gh: " + "Resource not accessible by integration " * 6 + "(HTTP 403)"
    refused = github.GhResult(False, "⚠️ GH_ERROR: " + words, 1, 403, "exit")
    fake = FakeChecks(polls=[[_run(11, **_RUNNING)], [_run(11, conclusion="failure"), _run(12, "Docs", conclusion="failure"),
                                                      _run(13, "Odd", conclusion="x" * 600)]],
                      jobs={12: [_job(60, "build", conclusion="failure", run_id=12)]}, rollup=_ACTIONS_ROLLUP,
                      failing={"rollup": refused, "pr": [None, refused], "jobs": [refused, None], "annotations": refused})
    lines = checks(fake, number=7, wait_seconds=60).text.splitlines()
    gap = f"HTTP 403: {words[:159]}…"  # Each named failure is cut at 160 characters; the line is not cut again.
    assert lines[5] == (f"Sources unavailable: pull request check rollup ({gap}); pull request head after the wait ({gap}); "
                        f"jobs (runs: 1; {gap}); annotations (jobs: 1; {gap})")
    # A header line built from GitHub's free text is cut at 500 characters.
    counts = next(line for line in lines if line.startswith("Workflow runs: 3 — failure 2, xxx"))
    assert len(counts) == 500 and counts.endswith("…")


@pytest.mark.parametrize("args", [{"sha": "main"}, {"sha": SHA[:12]}, {"sha": "seven"}, {"sha": "7.5"}, {}, {"number": -1}])
def test_checks_target_is_one_pull_request_or_one_full_sha(checks, args):
    """One ``target``: a number is a pull request, any other text is a commit SHA the reader validates."""
    fake = FakeChecks(polls=[[_run(11)]])
    result = checks(fake, **args)
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR") and fake.calls == []
    assert result.meta.get("operation_outcome") == "completed_no_effect"  # Refused before any gh launch.
    assert ("git rev-parse" in result.text) == ("sha" in args and "number" not in args)
    assert checks(fake, sha=SHA).status == "ok" and checks(fake, number=7).status == "ok"


@pytest.mark.parametrize("args", [{"number": True}, {"number": 7.5}, {"number": [7]},
                                  {"sha": SHA, "wait_seconds": True}, {"sha": SHA, "wait_seconds": 0.4},
                                  {"sha": SHA, "wait_seconds": "7.5"}, {"sha": SHA, "wait_seconds": "soon"}])
def test_checks_number_and_wait_are_whole_numbers(checks, args):
    fake = FakeChecks(polls=[[_run(11)]])
    result = checks(fake, **args)
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR") and fake.calls == []
    assert "whole numbers" in result.text and result.meta.get("operation_outcome") == "completed_no_effect"
    # The same values in a whole-number form are read.
    for good in ({"number": 7.0}, {"number": "7"}, {"sha": SHA, "wait_seconds": "30"}, {"sha": SHA, "wait_seconds": None}):
        assert checks(fake, **good).status == "ok"


def test_bounds_keep_the_header_and_every_run_id(checks):
    from ouroboros.tool_capabilities import tool_result_limit

    steps = [(1, "Run tests " + "y" * 80, "completed", "failure")]
    fake = FakeChecks(
        polls=[[_run(100 + index, f"Workflow {index}", conclusion="failure") for index in range(40)]],
        jobs={100 + index: [_job(1000 * (index + 1) + job, f"job {job} " + "x" * 60, conclusion="failure",
                                 run_id=100 + index, steps=steps) for job in range(300)] for index in range(40)})
    text = checks(fake, sha=SHA).text
    lines = text.splitlines()
    assert len(text) <= tool_result_limit("github_read")
    assert lines[:2] == _HEADER and "Workflow runs: 40 — failure 40" in lines and lines[-2:] == _SHA_TAIL
    assert all(str(100 + index) in text for index in range(40))
    assert "- 20 more runs — failure; ids: " + ", ".join(str(120 + index) for index in range(20)) in lines
    assert ("Sources read: workflow runs; jobs (runs: 8; jobs not read for 32 runs: one call reads the jobs of 8 runs); "
            "annotations (jobs: 10)") in lines
    assert "    290 more jobs not shown: failure 290" in lines
    assert "    jobs: not read (one call expands 8 runs)" in lines
    assert any(line.endswith("more detail lines of this run are not shown: the result bound is reached") for line in lines)
    assert len(fake.sent("run")) == 1 + 8 and len(fake.sent("api")) == 10
    assert "        annotations: not read (one call reads the annotations of 10 jobs)" in lines
    # A small report carries none of the bound notices.
    small = checks(_failed_world(), sha=SHA).text
    assert not any(notice in small for notice in ("more runs", "more jobs", "not shown", "jobs: not read", "jobs not read",
                                                  "annotations: not read", "may hold more"))


_CONTINUED = [_job(54, "quick-test"), _job(55, "full-test (ubuntu-latest)", conclusion="skipped"),
              _job(56, "build", conclusion="skipped"),
              _job(57, "lint", conclusion="failure", steps=[(2, "Run", "completed", "failure")])]
_LINT_ANNOTATION = {57: [{"annotation_level": "failure", "title": "", "message": "ruff: F821"}]}
_LINT_DETAIL = [f"    - job lint [57]: failure {_RUNS}/11/job/57", "        step 2 Run: failure",
                "        annotation failure: ruff: F821",
                "        annotations: 1 at failure level, 1 shown; 1 of all levels read"]


def test_a_run_completed_with_success_states_its_jobs_and_a_failed_one_in_full(checks):
    """GitHub calls a run success while its jobs are skipped or one of them failed under `continue-on-error`:
    the counts say so, and the failed job carries its steps and annotations like a job of a failed run."""
    fake = FakeChecks(polls=[[_run(11), _run(12, "Docs")]], jobs={11: _CONTINUED, 12: [_job(60, "build", run_id=12)]},
                      annotations=_LINT_ANNOTATION)
    assert checks(fake, sha=SHA).text.splitlines()[3:] == [
        "Sources read: workflow runs; jobs (runs: 2); annotations (jobs: 1)",
        "Sources unavailable: none",
        "Workflow runs: 2 — success 2",
        "", "Runs completed with success (2):",
        f"- CI (pull_request) run 11 attempt 1: success {_RUNS}/11",
        "    log of the failed steps: gh run view 11 --log-failed --repo github.example/owner/selected",
        "    jobs: 4 — success 1, failure 1, skipped 2",
        *_LINT_DETAIL,
        f"- Docs (pull_request) run 12 attempt 1: success {_RUNS}/12",
        "    jobs: 1 — success 1",
        *_SHA_TAIL]
    # Only the failed job's annotations are read; the run with successful jobs alone costs no annotation call.
    assert [args[1].split("/")[4] for args in fake.sent("api")] == ["57"]


def test_a_pull_request_reads_a_success_run_whose_rollup_job_failed(checks):
    rollup = [{**_ACTIONS_ROLLUP[0], "name": "lint", "conclusion": "FAILURE", "detailsUrl": f"{_RUNS}/11/job/57"},
              {**_ACTIONS_ROLLUP[0], "name": "build", "detailsUrl": f"{_RUNS}/12/job/60"}]
    fake = FakeChecks(polls=[[_run(11), _run(12, "Docs")]], jobs={11: _CONTINUED, 12: [_job(60, "build", run_id=12)]},
                      annotations=_LINT_ANNOTATION, rollup=rollup)
    lines = checks(fake, number=7).text.splitlines()
    at = lines.index(f"- CI (pull_request) run 11 attempt 1: success {_RUNS}/11")
    assert lines[at + 1:at + 7] == [
        "    log of the failed steps: gh run view 11 --log-failed --repo github.example/owner/selected",
        "    jobs: 4 — success 1, failure 1, skipped 2", *_LINT_DETAIL]
    # The run whose rollup jobs all succeeded is left to the rollup's counts: no jobs read for it.
    assert [args[2] for args in fake.sent("run") if args[1] == "view"] == ["11"]
    assert lines[lines.index(f"- Docs (pull_request) run 12 attempt 1: success {_RUNS}/12") + 1] == ""

    # Twenty runs have a line of their own across both groups; the others keep their ids beside their state.
    mixed = [_run(100 + index, conclusion="failure") for index in range(15)] + [_run(200 + index) for index in range(9)]
    lines = checks(FakeChecks(polls=[mixed], rollup=_ACTIONS_ROLLUP), number=7).text.splitlines()
    assert sum(line.startswith("- CI ") for line in lines) == 20
    assert "- 4 more runs — success; ids: 205, 206, 207, 208" in lines

    # One call reads the jobs of eight runs; what is left is stated in the header and on each run.
    ten = {20 + index: [_job(1, "test", run_id=20 + index)] for index in range(10)}
    fake = FakeChecks(polls=[[_run(run_id) for run_id in ten]], jobs=ten)
    lines = checks(fake, sha=SHA).text.splitlines()
    assert ("Sources read: workflow runs; jobs (runs: 8; jobs not read for 2 runs: one call reads the jobs of 8 runs); "
            "annotations (jobs: 0)") in lines
    assert lines.count("    jobs: 1 — success 1") == 8 and lines.count("    jobs: not read (one call expands 8 runs)") == 2
    assert len(fake.sent("run")) == 1 + 8
    eight = checks(FakeChecks(polls=[[_run(run_id) for run_id in list(ten)[:8]]], jobs=ten), sha=SHA).text
    assert "jobs not read" not in eight and "jobs: not read" not in eight and eight.count("    jobs: 1 — success 1") == 8


def test_rollup_facts_reach_the_header_and_a_job_no_listed_run_explains_is_shown(checks):
    failed_job = {"__typename": "CheckRun", "name": "full-test", "workflowName": "CI", "status": "COMPLETED",
                  "conclusion": "FAILURE", "detailsUrl": f"{_RUNS}/99/job/5"}
    vercel = {"__typename": "CheckRun", "name": "Vercel", "workflowName": "", "status": "COMPLETED",
              "conclusion": "FAILURE", "detailsUrl": "https://vercel.example/github"}
    stray = "in a run absent from the run list or listed as success"
    lines = checks(FakeChecks(polls=[[_run(11)]], rollup=_ACTIONS_ROLLUP + [failed_job, vercel]), number=7).text.splitlines()
    # The header carries the failure of a third-party check and of a job whose run is not listed.
    assert lines[6:lines.index("")] == [
        "Workflow runs: 1 — success 1",
        f"Pull request rollup, GitHub Actions jobs: 2 — success 1, failure 1; 1 of them {stray}",
        "Other checks, outside GitHub Actions: 1 — failure 1"]
    assert lines[-4:] == [
        f"GitHub Actions jobs of the rollup {stray} (1) — failure 1:",
        f"- job CI / full-test: failure {_RUNS}/99/job/5",
        "Other checks, outside GitHub Actions (1) — failure 1:",
        "- check run Vercel: failure https://vercel.example/github"]
    # A failed job of a run GitHub lists as success is shown the same way ...
    inside = {**failed_job, "detailsUrl": f"{_RUNS}/11/job/5"}
    lines = checks(FakeChecks(polls=[[_run(11)]], rollup=[inside]), number=7).text.splitlines()
    assert lines[-3:] == [f"GitHub Actions jobs of the rollup {stray} (1) — failure 1:",
                          f"- job CI / full-test: failure {_RUNS}/11/job/5",
                          "Other checks: none — every entry of the pull request rollup (1) is a job of a GitHub Actions workflow."]
    # ... while a run listed as not success accounts for its job, and no other check adds no header line.
    text = checks(FakeChecks(polls=[[_run(11, conclusion="failure")]], rollup=[inside]), number=7).text
    assert "Pull request rollup, GitHub Actions jobs: 1 — failure 1\n" in text
    assert stray not in text and "Other checks, outside GitHub Actions" not in text
    # A job still running in a run no listed run explains is listed too, after the failed one GitHub returned later.
    running = {**failed_job, "name": "ui-smoke", "status": "IN_PROGRESS", "conclusion": "", "detailsUrl": f"{_RUNS}/98/job/6"}
    lines = checks(FakeChecks(polls=[[_run(11)]], rollup=[running, failed_job]), number=7).text.splitlines()
    assert lines[-4:-1] == [f"GitHub Actions jobs of the rollup {stray} (2) — failure 1, in_progress 1:",
                            f"- job CI / full-test: failure {_RUNS}/99/job/5",
                            f"- job CI / ui-smoke: in_progress {_RUNS}/98/job/6"]


def test_failure_class_runs_are_listed_and_expanded_first(checks):
    order = [_run(1, conclusion="skipped"), _run(2, **_RUNNING), _run(3, conclusion="cancelled"), _run(4),
             _run(5, conclusion="timed_out"), _run(6, conclusion="failure")]
    spines = [line for line in checks(FakeChecks(polls=[order]), sha=SHA).text.splitlines() if line.startswith("- CI ")]
    assert [line.split()[4] for line in spines] == ["6", "5", "3", "2", "1", "4"]

    # Twenty cancelled runs that GitHub lists first leave the five failed ones their spine, log hint and jobs.
    runs = [_run(100 + index, conclusion="cancelled") for index in range(20)] + [
        _run(200 + index, conclusion="failure") for index in range(5)]
    held = [_job(1, "build", conclusion="failure", run_id=100, steps=[(4, "Compile", "completed", "failure")]),
            _job(2, "test", conclusion="cancelled", run_id=100)]
    fake = FakeChecks(polls=[runs], jobs={100: held, 101: [_job(3, "test", conclusion="cancelled", run_id=101)]})
    lines = checks(fake, sha=SHA).text.splitlines()
    spines = [line for line in lines if line.startswith("- CI ")]
    assert [line.split()[4] for line in spines[:6]] == ["200", "201", "202", "203", "204", "100"]
    assert [args[2] for args in fake.sent("run")[1:]] == ["200", "201", "202", "203", "204", "100", "101", "102"]
    assert "- 5 more runs — cancelled; ids: 115, 116, 117, 118, 119" in lines
    hint = "    log of the failed steps: gh run view {} --log-failed --repo github.example/owner/selected"
    assert all(lines[lines.index(spine) + 1] == hint.format(spine.split()[4]) for spine in spines[:5])
    # A cancelled run that holds a failed job carries the log command; a cancelled run without one does not.
    at = lines.index(f"- CI (pull_request) run 100 attempt 1: cancelled {_RUNS}/100")
    assert lines[at + 1:at + 5] == [hint.format(100), "    jobs: 2 — failure 1, cancelled 1",
                                    f"    - job build [1]: failure {_RUNS}/100/job/1", "        step 4 Compile: failure"]
    assert lines[at + 6] == f"    - job test [2]: cancelled {_RUNS}/100/job/2"  # The cancelled sibling comes after.
    at = lines.index(f"- CI (pull_request) run 101 attempt 1: cancelled {_RUNS}/101")
    assert lines[at + 1] == "    jobs: 1 — cancelled 1" and hint.format(101) not in lines


def test_the_limits_of_the_run_list_are_stated(checks):
    note = "The run list is read up to 100 runs; GitHub may hold more for this commit."
    other_head = _run(999, conclusion="failure", headSha=MOVED)
    fake = FakeChecks(polls=[[_run(run_id) for run_id in range(1, 100)] + [other_head]], rollup=_ACTIONS_ROLLUP)
    text = checks(fake, number=7).text
    # A run of another head is left out of the report, and it still counts as a row GitHub returned.
    assert "Workflow runs: 99 — success 99" in text.splitlines() and f"{_RUNS}/999" not in text and "failure" not in text
    assert note in text.splitlines()[:9] and fake.sent("run")[0][4:6] == ["--limit", "100"]
    text = checks(FakeChecks(polls=[[_run(run_id) for run_id in range(1, 100)]], rollup=_ACTIONS_ROLLUP), number=7).text
    assert "Workflow runs: 99 — success 99" in text and "may hold more" not in text

    # A failed job GitHub returns without a URL leaves no literal annotation path to read.
    fake = FakeChecks(polls=[[_run(11, conclusion="failure")]], jobs={11: [{**_job(55, "test", conclusion="failure"), "url": ""}]})
    lines = checks(fake, sha=SHA).text.splitlines()
    assert lines[lines.index("    - job test [55]: failure") + 1] == "        annotations: not read (GitHub returned no job URL)"
    assert fake.sent("api") == [] and "no job URL" not in checks(_failed_world(), sha=SHA).text


def test_a_failed_step_is_shown_whatever_its_job_concluded(checks):
    """GitHub can conclude a job cancelled after one of its steps failed: the step, its annotations and the log
    command are facts of the report, and the job keeps the state GitHub gives it."""
    steps = [(2, "Run tests", "completed", "failure"), (9, "Cleanup", "completed", "cancelled")]
    fake = FakeChecks(
        polls=[[_run(11, conclusion="cancelled")]],
        jobs={11: [_job(55, "full-test", conclusion="cancelled", steps=steps),
                   _job(56, "build", conclusion="cancelled", steps=[(3, "Compile", "completed", "cancelled")])]},
        annotations={55: [{"annotation_level": "failure", "title": "Failed test", "message": "tests/test_x.py::test_y"}]})
    assert checks(fake, sha=SHA).text.splitlines()[3:] == [
        "Sources read: workflow runs; jobs (runs: 1); annotations (jobs: 2)",
        "Sources unavailable: none",
        "Workflow runs: 1 — cancelled 1",
        "", "Runs not completed with success (1):",
        f"- CI (pull_request) run 11 attempt 1: cancelled {_RUNS}/11",
        "    log of the failed steps: gh run view 11 --log-failed --repo github.example/owner/selected",
        "    jobs: 2 — cancelled 2",
        f"    - job full-test [55]: cancelled {_RUNS}/11/job/55",
        "        step 2 Run tests: failure",
        "        step 9 Cleanup: cancelled",
        "        annotation failure [Failed test]: tests/test_x.py::test_y",
        "        annotations: 1 at failure level, 1 shown; 1 of all levels read",
        f"    - job build [56]: cancelled {_RUNS}/11/job/56",
        "        step 3 Compile: cancelled",
        "        annotations: 0 at failure level, 0 shown; 0 of all levels read",
        *_SHA_TAIL]
    assert [args[1].split("/")[4] for args in fake.sent("api")] == ["55", "56"]  # The job with the failed step first.

    # A step that failed inside a job still in progress is shown with its annotations; the run has no log yet.
    steps = [(2, "Run tests", "completed", "failure"), (3, "Upload", "in_progress", "")]
    fake = FakeChecks(polls=[[_run(11, **_RUNNING)]], jobs={11: [_job(55, "full-test", steps=steps, **_RUNNING)]})
    lines = checks(fake, sha=SHA).text.splitlines()
    assert lines[-8:] == [f"- CI (pull_request) run 11 attempt 1: in_progress {_RUNS}/11",
                          "    jobs: 1 — in_progress 1",
                          f"    - job full-test [55]: in_progress {_RUNS}/11/job/55",
                          "        step 2 Run tests: failure", "        step 3 Upload: in_progress",
                          "        annotations: 0 at failure level, 0 shown; 0 of all levels read", *_SHA_TAIL]
    assert len(fake.sent("api")) == 1 and "--log-failed" not in "\n".join(lines)


def test_the_log_command_stands_only_beside_runs_that_have_step_logs(checks):
    states = ["failure", "timed_out", "startup_failure", "action_required", "stale", "cancelled"]
    fake = FakeChecks(polls=[[_run(11 + index, conclusion=state) for index, state in enumerate(states)]],
                      jobs={11 + index: [_job(50 + index, "build", conclusion="skipped", run_id=11 + index)]
                            for index in range(len(states))})
    lines = checks(fake, sha=SHA).text.splitlines()
    hinted = [line.split()[8] for line in lines if line.startswith("    log of the failed steps: gh run view ")]
    assert hinted == ["11", "12"]  # failure and timed_out; the other states leave no failed step to read.
    # A run in one of the other states that holds a failed job does carry the command.
    fake = FakeChecks(polls=[[_run(13, conclusion="action_required")]],
                      jobs={13: [_job(52, "build", conclusion="failure", run_id=13)]})
    assert "    log of the failed steps: gh run view 13 --log-failed --repo github.example/owner/selected" in checks(
        fake, sha=SHA).text.splitlines()


def test_earlier_attempts_of_a_run_are_named_as_not_read(checks):
    fake = FakeChecks(polls=[[_run(11, attempt=2), _run(12, "Docs", attempt=4), _run(13, "Lint")]], rollup=_ACTIONS_ROLLUP)
    lines = checks(fake, number=7).text.splitlines()
    assert lines[lines.index("Runs completed with success (3):") + 1:-2] == [
        f"- CI (pull_request) run 11 attempt 2: success {_RUNS}/11",
        "    attempt 1 is not read: gh run view 11 --attempt 1 --repo github.example/owner/selected",
        f"- Docs (pull_request) run 12 attempt 4: success {_RUNS}/12",
        "    attempts 1 to 3 are not read: gh run view 12 --attempt 1 --repo github.example/owner/selected",
        f"- Lint (pull_request) run 13 attempt 1: success {_RUNS}/13"]


def test_states_github_does_not_name_are_never_read_as_success(checks):
    """A completed item without a conclusion is `completed` and an item without a status is `unknown`."""
    fake = FakeChecks(polls=[[_run(11, conclusion=""), _run(12, "Docs", status="", conclusion="success"), _run(13, "Lint")]],
                      jobs={11: [_job(54, "quick-test", conclusion=""), _job(55, "build", status="", conclusion="success")]})
    lines = checks(fake, sha=SHA).text.splitlines()
    assert "Workflow runs: 3 — success 1, completed 1, unknown 1" in lines
    assert lines[lines.index("Runs not completed with success (2):") + 1:lines.index("Runs completed with success (1):") - 1] == [
        f"- CI (pull_request) run 11 attempt 1: completed {_RUNS}/11",
        "    jobs: 2 — completed 1, unknown 1",
        f"    - job quick-test [54]: completed {_RUNS}/11/job/54",
        "        annotations: 0 at failure level, 0 shown; 0 of all levels read",
        f"    - job build [55]: unknown {_RUNS}/11/job/55",
        f"- Docs (pull_request) run 12 attempt 1: unknown {_RUNS}/12",
        "    jobs: 0 — GitHub lists no job for this run"]


def test_step_annotation_and_rollup_lists_state_what_they_leave_out(checks):
    steps = [(number, f"Check {number}", "completed", "failure") for number in range(1, 6)]
    page = [{"annotation_level": "failure", "title": "", "message": f"failure {index}"} for index in range(7)]
    full = page + [{"annotation_level": "warning", "title": "", "message": "deprecated"}] * 93
    fake = FakeChecks(polls=[[_run(11, conclusion="failure")]],
                      jobs={11: [_job(55, "lint", conclusion="failure", steps=steps),
                                 _job(56, "test", conclusion="failure")]},
                      annotations={55: full, 56: full[:99]})
    lines = checks(fake, sha=SHA).text.splitlines()
    at = lines.index(f"    - job lint [55]: failure {_RUNS}/11/job/55")
    assert lines[at + 1:at + 5] == ["        step 1 Check 1: failure", "        step 2 Check 2: failure",
                                    "        step 3 Check 3: failure", "        2 more steps of this job: failure 2"]
    # Five failure annotations are shown, and a full page says that GitHub may hold more.
    assert lines[at + 5:at + 11] == [f"        annotation failure: failure {index}" for index in range(5)] + [
        "        annotations: 7 at failure level, 5 shown; the first 100 of all levels read — "
        "GitHub may hold more for this job"]
    assert lines[lines.index(f"    - job test [56]: failure {_RUNS}/11/job/56") + 6] == (
        "        annotations: 7 at failure level, 5 shown; 99 of all levels read")

    # Rollup lists are failure-first before the twelve-entry cut, and the cut is stated.
    quiet = [{"__typename": "StatusContext", "context": f"deploy/{index}", "state": "SUCCESS", "targetUrl": ""}
             for index in range(12)]
    red = {"__typename": "StatusContext", "context": "security/scan", "state": "FAILURE", "targetUrl": "https://scan.example/7"}
    lines = checks(FakeChecks(polls=[[_run(11)]], rollup=quiet + [red]), number=7).text.splitlines()
    at = lines.index("Other checks, outside GitHub Actions (13) — success 12, failure 1:")
    assert lines[at + 1] == "- commit status security/scan: failure https://scan.example/7"
    assert lines[at + 2] == "- commit status deploy/0: success" and lines[-1] == "- 1 more of these: success 1"
    assert "more of these" not in checks(FakeChecks(polls=[[_run(11)]], rollup=quiet[:11] + [red]), number=7).text
    # A commit status in `error` sorts with the failures, ahead of pending ones.
    pending = [{**entry, "state": "PENDING"} for entry in quiet]
    errored = {**red, "context": "ci/legacy", "state": "ERROR"}
    lines = checks(FakeChecks(polls=[[_run(11)]], rollup=pending + [errored]), number=7).text.splitlines()
    at = lines.index("Other checks, outside GitHub Actions (13) — pending 12, error 1:")
    assert lines[at + 1] == "- commit status ci/legacy: error https://scan.example/7"
    # A required commit status that has not reported yet (`expected`) sorts with the unfinished ones, ahead of neutral.
    neutral = [{"__typename": "CheckRun", "name": f"optional/{index}", "workflowName": "", "status": "COMPLETED",
                "conclusion": "NEUTRAL", "detailsUrl": ""} for index in range(12)]
    required = {**red, "context": "ci/required", "state": "EXPECTED", "targetUrl": ""}
    lines = checks(FakeChecks(polls=[[_run(11)]], rollup=neutral + [required]), number=7).text.splitlines()
    assert lines[lines.index("Other checks, outside GitHub Actions (13) — neutral 12, expected 1:") + 1] == (
        "- commit status ci/required: expected")


@pytest.mark.parametrize("rollup", [[], None])
def test_an_empty_rollup_is_called_empty(checks, rollup):
    lines = checks(FakeChecks(polls=[[_run(11)]], rollup=rollup), number=7).text.splitlines()
    assert lines[6:8] == ["Workflow runs: 1 — success 1", "Pull request rollup: empty"]
    assert lines[-1] == "Other checks: none — the pull request rollup is empty."
    assert "Sources read: workflow runs; jobs (runs: 0); annotations (jobs: 0); pull request check rollup" in lines
    # A rollup that holds entries keeps the sentence about them.
    assert checks(FakeChecks(polls=[[_run(11)]], rollup=_ACTIONS_ROLLUP), number=7).text.splitlines()[-1] == (
        "Other checks: none — every entry of the pull request rollup (1) is a job of a GitHub Actions workflow.")


def test_the_report_stays_inside_the_result_limit(checks, monkeypatch):
    """Long names and long GitHub Enterprise URLs: the header facts, the counts and every run id are reserved
    first, and the rollup entries, the run lines and the detail lines share what is left."""
    from ouroboros.tool_capabilities import tool_result_limit

    host = "https://github.enterprise.internal.example-corporation.com/platform-infrastructure-organization/integration-tests"
    ids = [36980000000 + index for index in range(100)]
    runs = [{**_run(run_id, f"Integration tests of the infrastructure platform, shard {run_id} " + "w" * 40,
                    conclusion="failure" if run_id % 2 else "cancelled", attempt=3),
             "url": f"{host}/actions/runs/{run_id}"} for run_id in ids]
    jobs = {run_id: [{**_job(job, f"job {job} " + "x" * 100, conclusion="failure",
                             steps=[(1, "Run " + "y" * 100, "completed", "failure")]),
                      "url": f"{host}/actions/runs/{run_id}/job/{job}"} for job in range(300)] for run_id in ids}
    rollup = [{"__typename": "CheckRun", "name": f"job {index} " + "n" * 90, "workflowName": "Integration " + "w" * 60,
               "status": "COMPLETED", "conclusion": "FAILURE",
               "detailsUrl": f"{host}/actions/runs/{5000 + index}/job/{index}?" + "q" * 150} for index in range(40)]
    rollup += [{"__typename": "StatusContext", "context": f"continuous-integration/legacy-status-{index}-" + "c" * 90,
                "state": "FAILURE", "targetUrl": "https://status.example-corporation.com/builds/" + "z" * 220}
               for index in range(40)]
    result = checks(FakeChecks(polls=[runs], jobs=jobs, rollup=rollup), number=7)
    text, lines = result.text, result.text.splitlines()
    assert result.status == "ok" and len(text) <= tool_result_limit("github_read")
    assert lines[:3] == [f"GitHub checks for commit {SHA}", "Repository: " + host[len("https://"):],
                         _PR_LINE + "; head as read by this call"]
    assert lines[5:10] == [
        "Sources unavailable: none",
        "Workflow runs: 100 — failure 50, cancelled 50",
        "The run list is read up to 100 runs; GitHub may hold more for this commit.",
        "Pull request rollup, GitHub Actions jobs: 40 — failure 40; 40 of them in a run absent from the run list "
        "or listed as success",
        "Other checks, outside GitHub Actions: 40 — failure 40"]
    # Every run id stays beside its state: on a line of its own while room lasts, then on one ids line per state.
    own = [int(line.split()[line.split().index("run") + 1]) for line in lines if line.startswith("- Integration tests")]
    packed = {state: [int(run_id) for run_id in line.partition("ids: ")[2].split(", ")]
              for line in lines for state in ("failure", "cancelled") if f" more runs — {state}; ids: " in line}
    assert 0 < len(own) < 20 and all(run_id % 2 for run_id in own)  # Fewer than the twenty lines a roomy report has.
    assert all(run_id % 2 for run_id in packed["failure"]) and not any(run_id % 2 for run_id in packed["cancelled"])
    assert sorted(own + packed["failure"] + packed["cancelled"]) == ids
    # Both rollup lists keep their title and counts, and what they leave out is counted.
    assert "Other checks, outside GitHub Actions (40) — failure 40:" in lines and lines[-1] == "- 28 more of these: failure 28"
    assert any(line.endswith("more detail lines of this run are not shown: the result bound is reached") for line in lines)

    # The order of admission holds under any limit: with a third of the room the header facts, the counts and
    # every run id stay, the rollup entries are cut by the room that is left, and no run has a line of its own.
    monkeypatch.setattr(github_checks, "tool_result_limit", lambda _name: 5000)
    tight = checks(FakeChecks(polls=[runs], jobs=jobs, rollup=rollup), number=7).text.splitlines()
    assert len("\n".join(tight)) <= 5000 and tight[:3] == lines[:3] and tight[5:10] == lines[5:10]
    packed = {state: [int(run_id) for run_id in line.partition("ids: ")[2].split(", ")]
              for line in tight for state in ("failure", "cancelled") if f" more runs — {state}; ids: " in line}
    assert sorted(packed["failure"] + packed["cancelled"]) == ids and not any(line.startswith("- Integration") for line in tight)
    shown = [sum(line.startswith(kind) for line in tight) for kind in ("- job ", "- commit status ")]
    assert 0 < shown[0] < 12 and shown[1] < 12
    assert [int(line.split()[1]) for line in tight if " more of these: failure " in line] == [40 - shown[0], 40 - shown[1]]


def test_checks_reader_is_registered_read_only():
    from ouroboros import tool_capabilities
    from ouroboros.consciousness_authority import OBSERVE_DISABLED
    from ouroboros.safety import POLICY_SKIP, TOOL_POLICY
    from ouroboros.tools.registry_guards import _GITHUB_TOKEN_TOOLS

    entry = next(item for item in github.get_tools() if item.name == "github_read")
    assert TOOL_POLICY["github_read"] == POLICY_SKIP and "github_read" in _GITHUB_TOKEN_TOOLS
    assert entry.schema["parameters"]["required"] == ["method"] and not entry.mutates_worktree
    assert "checks" in entry.schema["parameters"]["properties"]["method"]["enum"]
    assert {"target", "wait_seconds", "repo"} <= set(entry.schema["parameters"]["properties"])
    assert github_checks._CHECKS_WAIT_CAP_SEC + github_checks._CHECKS_READ_BUDGET_SEC < entry.timeout_sec
    assert "max 240" in entry.schema["parameters"]["properties"]["wait_seconds"]["description"]
    for listing in (OBSERVE_DISABLED, tool_capabilities.OBSERVE_WORLD_MUTATION_TOOLS, tool_capabilities.READ_ONLY_PARALLEL_TOOLS):
        assert "github_read" not in listing
    for children in (tool_capabilities.LOCAL_READONLY_SUBAGENT_TOOL_NAMES, tool_capabilities.ACTING_SUBAGENT_TOOL_NAMES):
        assert "github_read" in children and "github_write" not in children
    assert "github_write" in OBSERVE_DISABLED  # The same lists do carry the family's write tool.
    # The reader's module is a helper of the family: the family module registers the tool, in frozen builds too.
    from ouroboros.tools.registry import ToolRegistry

    assert not hasattr(github_checks, "get_tools") and hasattr(github, "get_tools")
    assert "github" in ToolRegistry._FROZEN_TOOL_MODULES and "github_checks" not in ToolRegistry._FROZEN_TOOL_MODULES
