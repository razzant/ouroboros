"""Reader of one commit's GitHub checks for ``github_read(method="checks")``: facts and unavailable sources, never a verdict.

Not a tool module: ``github.py`` registers the tool and owns the transport this reader calls.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Dict, List, Optional

from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.github import GhResult, _gh_run, _refuse
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import utc_now_iso

# GitHub's states in report order: the finished ones, then the unfinished ones (`expected`: a required commit status
# that has not reported yet).
_CHECKS_UNFINISHED = ("queued", "in_progress", "waiting", "requested", "pending", "expected")
_CHECK_STATES = ("success", "failure", "cancelled", "timed_out", "startup_failure", "action_required", "stale",
                 "skipped", "neutral", *_CHECKS_UNFINISHED)
_CHECKS_QUIET = ("success", "skipped", "neutral")  # These carry no failed step.
_CHECKS_FAILED = ("failure", "timed_out")  # These leave a step log to read.
# Runs and rollup entries are listed in this order: failure-class states, unfinished ones, the rest, success last.
_CHECKS_RANK = {state: rank for rank, state in enumerate(  # `error` is a commit status's failure word.
    ("failure", "error", "timed_out", "startup_failure", "action_required", "cancelled", *_CHECKS_UNFINISHED))}
# The wait cap plus the read budget stays under the ToolEntry default of 360 s.
_CHECKS_WAIT_CAP_SEC, _CHECKS_READ_BUDGET_SEC, _CHECKS_POLL_SEC = 240, 90, 15
_CHECKS_RUN_LIMIT, _CHECKS_RUN_LINES, _CHECKS_EXPANDED_RUNS = 100, 20, 8
_CHECKS_JOB_LINES, _CHECKS_STEP_LINES, _CHECKS_OTHER_LINES = 10, 3, 12
_CHECKS_ANNOTATED_JOBS, _CHECKS_ANNOTATION_LINES, _CHECKS_ANNOTATION_PAGE = 10, 5, 100
_CHECKS_HEAD_LINE, _CHECKS_MARGIN = 500, 200  # The longest header line; characters kept free under the result limit.
_GH_REPO_URL_RE = re.compile(r"^https://([^/]+)/([^/]+)/([^/]+)/")
_GH_RUN_ID_RE = re.compile(r"/actions/runs/(\d+)")
_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _check_state(item: dict) -> str:
    """GitHub's own word for a run, job, step or check: the conclusion once completed, else the status."""
    status = str(item.get("status") or item.get("state") or "").lower()
    if status != "completed":
        return status or "unknown"
    return str(item.get("conclusion") or "").lower() or "completed"


def _state_counts(items: List[dict]) -> str:
    counts = Counter(_check_state(item) for item in items)
    order = [state for state in _CHECK_STATES if state in counts] + sorted(set(counts) - set(_CHECK_STATES))
    return ", ".join(f"{state} {counts[state]}" for state in order)


def _rank(item: dict) -> int:
    state = _check_state(item)
    return _CHECKS_RANK.get(state, len(_CHECKS_RANK) + (state == "success"))


def _one_line(value: object, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _whole(value: object) -> Optional[int]:
    """An omitted argument is 0 and a whole number is itself; a boolean, a fraction or other text is None."""
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        return int(value) if re.fullmatch(r"\s*[+-]?\d+\s*", value) else None
    return value if isinstance(value, int) else None


def _gh_json(res: GhResult, kind: type):
    """The parsed answer of a successful gh call when it has the expected container type, else None."""
    try:
        data = json.loads(res.text) if res.ok else None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, kind) else None


def _gh_failure(res: GhResult) -> str:
    """Error class of an unavailable source: gh's HTTP status when it printed one, else how the call ended."""
    kind = f"HTTP {res.http_status}" if res.http_status else res.failure or "unparseable answer"
    return f"{kind}: {_one_line(res.text.partition(': ')[2], 160)}" if res.failure == "exit" else kind


def _job_failed(job: dict) -> bool:
    """A job that failed or timed out, or one that holds such a step under another conclusion of its own."""
    return any(_check_state(item) in _CHECKS_FAILED
               for item in (job, *(step for step in job.get("steps") or [] if isinstance(step, dict))))


def _loud_jobs(jobs: List[dict]) -> List[dict]:
    """The jobs with a line of their own, by their own states: failed ones first, cancelled ones last.

    A skipped job adds nothing to the counts line. A cancelled one keeps its line: GitHub concludes a job stopped at its
    time limit `cancelled`, and only a failure annotation of that job names the cause."""
    return sorted((job for job in jobs if _job_failed(job) or _check_state(job) not in _CHECKS_QUIET), key=lambda job: (
        not _job_failed(job), _check_state(job) == "cancelled", job.get("status") != "completed"))


def _annotated(job: dict) -> bool:
    """A job whose annotations are read: a completed one, or one that holds a failed step while it runs."""
    return _job_failed(job) or job.get("status") == "completed"


def _job_lines(jobs: List[dict], notes: Dict[int, List[str]]) -> List[str]:
    """The lines of one run: job counts first, then its jobs with a line of their own, their steps and annotations.

    Chosen by the jobs' own states: a `continue-on-error` job fails inside a run GitHub calls success."""
    if not jobs:
        return ["    jobs: 0 — GitHub lists no job for this run"]
    lines = [f"    jobs: {len(jobs)} — {_state_counts(jobs)}"]
    loud = _loud_jobs(jobs)
    for job in loud[:_CHECKS_JOB_LINES]:
        state = _check_state(job)
        lines.append(f"    - job {_one_line(job.get('name'))} [{job.get('databaseId')}]: {state} "
                     f"{_one_line(job.get('url'), 200)}".rstrip())
        steps = [step for step in job.get("steps") or [] if isinstance(step, dict)  # By the step's own state.
                 and _check_state(step) not in (*_CHECKS_QUIET, "pending", "queued", "unknown")]
        lines += [f"        step {step.get('number')} {_one_line(step.get('name'))}: {_check_state(step)}"
                  for step in steps[:_CHECKS_STEP_LINES]]
        if len(steps) > _CHECKS_STEP_LINES:
            lines.append(f"        {len(steps) - _CHECKS_STEP_LINES} more steps of this job: "
                         f"{_state_counts(steps[_CHECKS_STEP_LINES:])}")
        lines += ["        " + line for line in notes.get(id(job), [])]
    if len(loud) > _CHECKS_JOB_LINES:
        lines.append(f"    {len(loud) - _CHECKS_JOB_LINES} more jobs not shown: {_state_counts(loud[_CHECKS_JOB_LINES:])}")
    return lines


def _run_block(run: dict, slug: str, log_failed: bool, jobs_line: str) -> str:
    """The lines a listed run always carries: its id, state and URL, then the commands and job counts that go with it."""
    run_id, target = run.get("databaseId"), f" --repo {slug}" if slug else ""
    block = [(f"- {_one_line(run.get('workflowName'), 80)} ({_one_line(run.get('event'), 30)}) run {run_id} "
              f"attempt {run.get('attempt')}: {_check_state(run)} {_one_line(run.get('url'), 200)}").rstrip()]
    if log_failed:
        block.append(f"    log of the failed steps: gh run view {run_id} --log-failed{target}")
    attempt = _whole(run.get("attempt")) or 0
    if attempt > 1:  # The run list holds the latest attempt of a run only.
        earlier = "attempt 1 is" if attempt == 2 else f"attempts 1 to {attempt - 1} are"
        block.append(f"    {earlier} not read: gh run view {run_id} --attempt 1{target}")
    return "\n".join(block + ([jobs_line] if jobs_line else []))


def _compact_runs(runs: List[dict]) -> List[str]:
    """Runs without a line of their own: one ids line per state keeps every run id beside its state."""
    by_state: dict = {}
    for run in runs:
        by_state.setdefault(_check_state(run), []).append(str(run.get("databaseId")))
    return [f"- {len(ids)} more runs — {state}; ids: {', '.join(ids)}" for state, ids in by_state.items()]


def _rollup_report(checks: object, runs: List[dict]) -> tuple:
    """A pull request's check rollup: header facts, the lists to print (title, entries) and the closing lines."""
    rollup = [check for check in checks or [] if isinstance(check, dict)]
    if not rollup:
        return ["Pull request rollup: empty"], [], ["Other checks: none — the pull request rollup is empty."]
    actions = [check for check in rollup if check.get("__typename") == "CheckRun" and check.get("workflowName")]
    others = sorted((check for check in rollup
                     if not (check.get("__typename") == "CheckRun" and check.get("workflowName"))), key=_rank)
    listed = {str(run.get("databaseId")) for run in runs if _check_state(run) != "success"}
    # A job in a failure or unfinished state that no run listed as not success holds is shown by itself.
    stray = sorted((job for job in actions if _check_state(job) not in _CHECKS_QUIET
                    and "".join(_GH_RUN_ID_RE.findall(str(job.get("detailsUrl") or ""))[:1]) not in listed), key=_rank)
    where = "in a run absent from the run list or listed as success"
    facts = [f"Pull request rollup, GitHub Actions jobs: {len(actions)}" + (f" — {_state_counts(actions)}" if actions else "")
             + (f"; {len(stray)} of them {where}" if stray else "")]
    if others:
        facts.append(f"Other checks, outside GitHub Actions: {len(others)} — {_state_counts(others)}")
    lists = [(title, entries) for title, entries in ((f"GitHub Actions jobs of the rollup {where}", stray),
                                                     ("Other checks, outside GitHub Actions", others)) if entries]
    return facts, lists, [] if others else [
        f"Other checks: none — every entry of the pull request rollup ({len(rollup)}) is a job of a GitHub Actions workflow."]


def _cost(lines: List[str]) -> int:
    return sum(len(line) + 1 for line in lines)


def _render(head: List[str], groups: List[tuple], lists: List[tuple], closing: List[str]) -> str:
    """Lay the report out inside the result limit.

    Reserved first: the header facts, every group and list title with its counts, and every run id beside its
    state. What is left admits, in this order, the rollup entries, a line per run, then the detail lines."""
    more = "- {} more of these: {}"
    spare = [len(more.format(len(checks), _state_counts(checks))) + 1 for _title, checks in lists]
    room = (tool_result_limit("github_read") - _CHECKS_MARGIN - _cost(head + closing) - 1 - sum(spare)
            - sum(len(title) + 1 + _cost(_compact_runs([run for run, _block, _detail in runs])) for title, runs in groups)
            - sum(len(title) + 1 for title, _checks in lists))
    tail: List[str] = []
    for (title, checks), reserved in zip(lists, spare):
        tail.append(title)
        shown = 0
        for check in checks[:_CHECKS_OTHER_LINES]:
            kind = ("commit status" if check.get("__typename") == "StatusContext" else
                    "job" if check.get("workflowName") else "check run")
            name = _one_line(check.get("context"), 80) or " / ".join(filter(None, (
                _one_line(check.get("workflowName"), 40), _one_line(check.get("name"), 80))))
            line = (f"- {kind} {name}: {_check_state(check)} "
                    f"{_one_line(check.get('targetUrl') or check.get('detailsUrl'), 200)}").rstrip()
            if len(line) + 1 > room:
                break
            room, shown = room - len(line) - 1, shown + 1
            tail.append(line)
        if shown < len(checks):
            tail.append(more.format(len(checks) - shown, _state_counts(checks[shown:])))
        else:
            room += reserved
    cut = "    {} more detail lines of this run are not shown: the result bound is reached"
    body: List[tuple] = []
    own_lines = _CHECKS_RUN_LINES  # Runs with a line of their own, across both groups.
    for title, runs in groups:
        body.append((title, []))
        shown = 0
        for index, (_run, block, detail) in enumerate(runs[:own_lines]):
            rest = [run for run, _block, _detail in runs[index:]]
            cost = len(block) + 1 + (len(cut) + 3 if detail else 0) - _cost(_compact_runs(rest)) + _cost(_compact_runs(rest[1:]))
            if cost > room:
                break
            room, shown = room - cost, shown + 1
            body.append((block, detail))
        body += [(line, []) for line in _compact_runs([run for run, _block, _detail in runs[shown:]])]
        own_lines -= shown
    lines = list(head)
    for block, detail in body:
        lines.append(block)
        for index, line in enumerate(detail):
            if len(line) + 1 > room:
                lines.append(cut.format(len(detail) - index))
                break
            room -= len(line) + 1
            lines.append(line)
        else:
            room += len(cut) + 3 if detail else 0
    return "\n".join(lines + [""] + tail + closing)


class _Sources:
    """The gh reads of one call under its one deadline, and the sources that could not be read."""

    def __init__(self, ctx: ToolContext, repo: str, deadline: float) -> None:
        self.ctx, self.repo, self.deadline = ctx, repo, deadline  # The one bound of every request and every sleep.
        self.rollup_gap = self.head_gap = ""
        self.job_failures: List[str] = []
        self.annotation_failures: List[str] = []
        self.annotation_reads: List[str] = []

    def call(self, args: List[str], bound: bool = True) -> GhResult:
        left = int(self.deadline - time.monotonic())
        if left < 1:
            return GhResult(False, "⚠️ GH_TIMEOUT: the deadline of this call was reached before the request.",
                            None, None, "deadline")
        return _gh_run(args, self.ctx, timeout=min(30, left), **({"repo": self.repo} if bound else {}))

    def pull_request(self, number: int) -> GhResult:
        """Head, state and check rollup of the pull request; GitHub can refuse the rollup alone."""
        fields = "headRefOid,url,isCrossRepository,state"
        res = self.call(["pr", "view", str(number), "--json", fields + ",statusCheckRollup"])
        self.rollup_gap = "" if res.ok else f"pull request check rollup ({_gh_failure(res)})"
        if res.failure in ("exit", "timeout"):
            res = self.call(["pr", "view", str(number), "--json", fields])
        return res

    def jobs(self, run: dict) -> tuple:
        """The jobs of one run, or None and the line that says why they are unreadable."""
        res = self.call(["run", "view", str(run.get("databaseId")), "--json", "jobs"])
        data = _gh_json(res, dict)
        jobs = data.get("jobs") if data is not None else None
        if isinstance(jobs, list):
            return [job for job in jobs if isinstance(job, dict)], ""
        self.job_failures.append(_gh_failure(res) if data is None else "the answer holds no jobs list")
        return None, f"    jobs: unavailable ({self.job_failures[-1]})"

    def annotations(self, job: dict) -> List[str]:
        # The one literal API path: host, owner and repository come from the URL GitHub returned.
        where, job_id = _GH_REPO_URL_RE.match(str(job.get("url") or "")), str(job.get("databaseId"))
        if not where or not job_id.isdigit():
            return ["annotations: not read (GitHub returned no job URL)"]
        if len(self.annotation_reads) >= _CHECKS_ANNOTATED_JOBS:
            return [f"annotations: not read (one call reads the annotations of {_CHECKS_ANNOTATED_JOBS} jobs)"]
        self.annotation_reads.append(job_id)
        host, owner, name = where.groups()
        res = self.call(["api", f"repos/{owner}/{name}/check-runs/{job_id}/annotations?per_page={_CHECKS_ANNOTATION_PAGE}",
                         "--hostname", host], bound=False)
        rows = _gh_json(res, list)
        if rows is None:
            self.annotation_failures.append(_gh_failure(res))
            return [f"annotations: unavailable ({self.annotation_failures[-1]})"]
        failures = [row for row in rows if isinstance(row, dict) and row.get("annotation_level") == "failure"]
        read = (f"{len(rows)} of all levels read" if len(rows) < _CHECKS_ANNOTATION_PAGE else
                f"the first {_CHECKS_ANNOTATION_PAGE} of all levels read — GitHub may hold more for this job")
        return [f"annotation failure{' [' + _one_line(row.get('title'), 80) + ']' if row.get('title') else ''}: "
                f"{_one_line(row.get('message'), 300)}" for row in failures[:_CHECKS_ANNOTATION_LINES]] + [
            f"annotations: {len(failures)} at failure level, {min(len(failures), _CHECKS_ANNOTATION_LINES)} shown; {read}"]

    def unavailable(self) -> str:
        gaps = [self.rollup_gap, self.head_gap]
        if self.job_failures:
            gaps.append(f"jobs (runs: {len(self.job_failures)}; {self.job_failures[0]})")
        if self.annotation_failures:
            gaps.append(f"annotations (jobs: {len(self.annotation_failures)}; {self.annotation_failures[0]})")
        return "; ".join(filter(None, gaps)) or "none"


def get_checks(ctx: ToolContext, number: object = 0, sha: object = "", wait_seconds: object = 0, repo: str = "") -> str:
    """Report what GitHub records about one commit's checks: facts and unavailable sources, never a verdict."""
    number, wait, sha = _whole(number), _whole(wait_seconds), str(sha or "").strip().lower()
    if number is None or wait is None:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: number (a pull request number) and wait_seconds are whole numbers.",
                       no_effect=True)
    if number < 0 or (number > 0) == bool(sha):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: pass exactly one target: number (a pull request) or sha "
                            "(a full 40-hex commit SHA).", no_effect=True)
    if sha and not _FULL_SHA_RE.fullmatch(sha):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: sha must be a full 40-hex commit SHA; pass a pull request number, "
                            "or resolve a branch or tag with `git rev-parse <ref>` first.", no_effect=True)
    started = time.monotonic()
    wait = max(0, min(wait, _CHECKS_WAIT_CAP_SEC))
    src = _Sources(ctx, repo, started + wait + _CHECKS_READ_BUDGET_SEC)
    pr: dict = {}
    if number:
        res = src.pull_request(number)
        if not res.ok:
            return res.text
        pr = _gh_json(res, dict) or {}
        sha = str(pr.get("headRefOid") or "").lower()
        if not _FULL_SHA_RE.fullmatch(sha):
            return _refuse(ctx, f"⚠️ TOOL_ERROR: GitHub returned no head commit for pull request #{number}.", "TOOL_ERROR")
    slept = False
    while True:  # Completion is judged on the runs of the fixed SHA, not on the jobs listed so far.
        res = src.call(["run", "list", "--commit", sha, "--limit", str(_CHECKS_RUN_LIMIT), "--json",
                        "databaseId,workflowName,event,status,conclusion,attempt,url,headSha"])
        if not res.ok:
            return res.text
        rows = _gh_json(res, list)
        if rows is None:
            return _refuse(ctx, f"⚠️ TOOL_ERROR: failed to parse workflow runs JSON: {res.text[:500]}", "TOOL_ERROR")
        runs = sorted((run for run in rows if isinstance(run, dict) and str(run.get("headSha") or sha).lower() == sha),
                      key=_rank)
        left = started + wait - time.monotonic()
        if left <= 0 or (runs and all(run.get("status") == "completed" for run in runs)):
            break
        time.sleep(min(_CHECKS_POLL_SEC, left))
        slept = True
    waited = min(int(time.monotonic() - started), wait)  # The time of the last request is not waiting.
    head_note = "head as read by this call"
    if number and slept:  # Only a wait that happened reads the pull request again.
        pr.pop("statusCheckRollup", None)  # The rollup read before the wait describes that moment.
        res = src.pull_request(number)
        latest = _gh_json(res, dict) or {}
        head_now = str(latest.get("headRefOid") or "").lower()
        if not head_now:
            src.head_gap = f"pull request head after the wait ({_gh_failure(res)})"
            head_note = "head after the wait not read"
        elif head_now == sha:
            head_note = "head unchanged after the wait"
            pr.update(latest)
        else:
            head_note = f"head moved to {head_now} during the wait; this report is for {sha}"
            pr["state"] = latest.get("state") or pr.get("state")
            src.rollup_gap = src.rollup_gap or "pull request check rollup (GitHub's rollup describes the moved head)"

    where = _GH_REPO_URL_RE.match(str((runs[0].get("url") if runs else "") or pr.get("url") or ""))
    repository = "/".join(where.groups()) if where else (repo or "resolved by the GitHub CLI from the Project directory")
    slug = ("/".join(where.groups()[1:]) if where.group(1) == "github.com" else repository) if where else repo
    # Without the rollup's job counts a run GitHub calls success is read for its own job counts; with them, such a
    # run is read when the rollup holds a failed job of it (a `continue-on-error` job fails inside a success run).
    flagged = {"".join(_GH_RUN_ID_RE.findall(str(check.get("detailsUrl") or ""))[:1])
               for check in pr.get("statusCheckRollup") or [] if isinstance(check, dict)
               and check.get("__typename") == "CheckRun" and _check_state(check) in _CHECKS_FAILED}
    wanted = [run for run in runs if _check_state(run) != "success" or "statusCheckRollup" not in pr
              or str(run.get("databaseId")) in flagged]
    expanded = {id(run): src.jobs(run) for run in wanted[:_CHECKS_EXPANDED_RUNS]}
    unread = {id(run) for run in wanted[_CHECKS_EXPANDED_RUNS:]}
    # Annotations are read once every run's jobs are known, failed jobs first: the cancelled jobs of a run listed
    # earlier never use up the annotation reads a failed job of a later run needs.
    shown = [job for jobs, _line in expanded.values()
             for job in _loud_jobs(jobs or [])[:_CHECKS_JOB_LINES] if _annotated(job)]
    notes = {id(job): src.annotations(job) for job in sorted(shown, key=lambda job: not _job_failed(job))}
    blocks = []
    for run in runs:
        jobs, jobs_line = expanded.get(id(run), (None, ""))
        detail: List[str] = []
        if jobs is not None:
            jobs_line, *detail = _job_lines(jobs, notes)
        elif id(run) in unread:
            jobs_line = f"    jobs: not read (one call expands {_CHECKS_EXPANDED_RUNS} runs)"
        # A step log exists for a completed run that failed or timed out, or whose jobs read here hold a failed
        # job or step.
        log_failed = run.get("status") == "completed" and (
            _check_state(run) in _CHECKS_FAILED or any(_job_failed(job) for job in jobs or []))
        blocks.append((run, _run_block(run, slug, log_failed, jobs_line), detail))
    over = len(unread)
    read = ["workflow runs", f"jobs (runs: {len(expanded) - len(src.job_failures)}" + (
                f"; jobs not read for {over} runs: one call reads the jobs of {_CHECKS_EXPANDED_RUNS} runs" if over else "") + ")",
            f"annotations (jobs: {len(src.annotation_reads) - len(src.annotation_failures)})"]
    head = [f"GitHub checks for commit {sha}", f"Repository: {repository}"]
    if number:
        about = [str(pr.get("state") or "").lower() or "state not returned"] + ["head in a fork"] * bool(pr.get("isCrossRepository"))
        head.append(f"Pull request: #{number} {pr.get('url') or ''} ({', '.join(about)}); {head_note}")
    facts = [f"Workflow runs: {len(runs)} — {_state_counts(runs)}" if runs else
             "Workflow runs: 0 — no workflow run is registered for this commit — this is not a test result"]
    if not runs and not number:
        facts.append("A commit SHA the repository does not hold and a commit with no run yet read the same" + (
            "." if repo else "; the repository is the one the GitHub CLI resolves from the Project directory "
                             "(pass repo='[HOST/]OWNER/REPO' to name it)."))
    if len(rows) >= _CHECKS_RUN_LIMIT:
        facts.append(f"The run list is read up to {_CHECKS_RUN_LIMIT} runs; GitHub may hold more for this commit.")
    lists, closing = [], ["Other checks: not read — " + (
        "the pull request check rollup is unavailable." if number else
        "a commit SHA target reads GitHub Actions workflow runs only; "
        "third-party checks and commit statuses are read for a pull request number.")]
    if "statusCheckRollup" in pr:
        read.append("pull request check rollup")
        rollup_facts, lists, closing = _rollup_report(pr["statusCheckRollup"], runs)
        facts += rollup_facts
    head += [f"Observed: {utc_now_iso()}" + (f"; waited {waited}s of {wait}s for the runs to complete" if wait else "")]
    # The two source lines are bounded by construction (a named failure is cut at 160 characters), and a clip
    # would drop the last source named: they are kept whole.
    head = [_one_line(line, _CHECKS_HEAD_LINE) for line in head] + [
        "Sources read: " + "; ".join(read), "Sources unavailable: " + src.unavailable()] + [
        _one_line(line, _CHECKS_HEAD_LINE) for line in facts]
    done = [block for block in blocks if _check_state(block[0]) == "success"]
    open_ = [block for block in blocks if _check_state(block[0]) != "success"]
    return _render(head,
                   [(f"\n{title} ({len(group)}):", group) for title, group in (
                       ("Runs not completed with success", open_), ("Runs completed with success", done)) if group],
                   [(f"{title} ({len(checks)}) — {_state_counts(checks)}:", checks) for title, checks in lists], closing)
