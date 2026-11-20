"""The post-merge check's hand-over to GitHub: one comment per culprit pull request, one issue per red.

scripts/post_merge_check.py runs the steps and documents them; this is its `route` step, with the
REST calls it makes through the workflow token (tests pass a fake). Every leg of the check runs on
one operating system and owns the `post-merge-red` issues of that system: the system is part of an
issue's key, title and machine block, and a leg extends and closes only its own issues. A block
that names no system is a Linux one: the check had only its Linux leg before 2026-10-28. Stdlib only.

A red no run could hand over survives as a record, not just an ID. `route` receives the still-owed
failures of earlier attempts as `prior_failures` — one {head, run, attempt, base, ids} group per
measurement, read from the digest-verified receipt — with `origin` naming this run the same way,
and returns what is still owed as `record["awaiting_handoff"]`, in that same shape. A record is
re-filed under its own head and its own attempt link, never under this run's; only an open issue
that lists it (for a size red, one measuring the base its record names), a whole pass or a
confirmed disappearance on this system retires it, and a record whose failing head this run has
not reached is settled by nothing — a late rerun cannot close a newer failure.
A record is owed by its exact ID, a size red once per base it was measured against (`_owed_key`):
an issue or a pass for one base settles nothing about another.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
import zipfile

BRANCH = "ouroboros"
LABEL = "post-merge-red"
NODE_LANE = "node-lane"
LINT_PREFIX = "ruff "  # a lint finding's id: "ruff <code> <path> <name>"
SIZE_PREFIX = "size "  # a size check test's id: "size <nodeid>"
BOOK_BUDGETS = "tests/test_reference_book_budgets.py"  # the size check's module that measures the books and prompts
SYSTEMS = {"linux": "Linux", "windows": "Windows"}
_BLOCK = re.compile(r"<!-- post-merge-red (\{.*?\}) -->", re.S)
_AUTHOR = re.compile(r" by `([^`\n]+)` landed\. First bad commit")  # a culprit issue's opening (issue_body)
_OBSERVED = re.compile(r"^(?:Found failing|Still not passing) at `([0-9a-f]{40})`: ", re.M)
_FORWARD = "This check blocks nothing and reverts nothing; the fix goes forward in a new pull request."
_COMMENT_LIMIT = 60_000  # GitHub refuses a comment longer than 65,536 characters


class GitHub:
    """The REST calls of this check, authenticated with the workflow token; tests pass a fake."""

    def __init__(self, repo: str, token: str):
        self.repo, self.token = repo, token

    def _call(self, method: str, url: str, body=None):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "ouroboros-post-merge-check"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data, headers["Content-Type"] = json.dumps(body).encode("utf-8"), "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8")
            return (json.loads(text) if text.strip() else None), response.headers.get("Link") or ""

    def _url(self, path: str) -> str:
        return f"https://api.github.com/repos/{self.repo}/{path}"

    def get(self, path: str):
        return self._call("GET", self._url(path))[0]

    def pages(self, path: str) -> list:
        items, url = [], self._url(path) + ("&" if "?" in path else "?") + "per_page=100"
        while url:
            page, link = self._call("GET", url)
            items.extend(page or [])
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else ""
        return items

    def handoff_json(self, run: int, attempt: int, part: str, digest: str):
        """Read the exact overflow receipt, not a run's potentially large diagnostic archive.

        `FileNotFoundError` is a fact about the run, not a read failure: the run's own listing —
        asked for this exact name and answered whole — carries no such artifact, or only an
        expired one, so that named receipt was never delivered (or is gone for good). Everything
        else leaves the receipt unknown, never absent: an unreadable or ambiguous listing, a
        partial page, a standalone HTTP status, an unreadable archive, a member that is not the
        receipt, bytes that do not match the recorded digest."""
        name = f"post-merge-handoff-{part}-{run}-{attempt}"
        listing = self.get(f"actions/runs/{run}/artifacts?name={name}")
        if (not isinstance(listing, dict) or not isinstance(listing.get("artifacts"), list)
                or type(listing.get("total_count")) is not int):
            raise ValueError("the run's artifact listing is unreadable")
        artifacts = listing["artifacts"]
        if any(not isinstance(artifact, dict) for artifact in artifacts) or len(artifacts) != listing["total_count"]:
            raise ValueError("the run's artifact listing is incomplete")
        exact = [artifact for artifact in artifacts if artifact.get("name") == name]
        if len(exact) != 1:
            if not exact and not artifacts:
                raise FileNotFoundError(f"run {run} attempt {attempt} has no hand-over artifact named {name}")
            raise ValueError("the run's artifact listing is ambiguous")
        artifact = exact[0]
        if artifact.get("expired") is not False:
            if artifact.get("expired") is True:
                raise FileNotFoundError(f"the hand-over artifact {name} of run {run} attempt {attempt} has expired")
            raise ValueError("the exact hand-over artifact is not available")
        if type(artifact.get("id")) is not int or artifact["id"] < 1:
            raise ValueError("the exact hand-over artifact is not available")
        request = urllib.request.Request(self._url(f"actions/artifacts/{artifact['id']}/zip"))
        if self.token:
            # GitHub redirects to a signed storage URL; do not forward the workflow token.
            request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        limit = 16 * 1024 * 1024
        with urllib.request.urlopen(request, timeout=30) as response:
            archive = response.read(limit + 1)
        if len(archive) > limit:
            raise ValueError("hand-over archive exceeds the reader's byte limit")
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            [member] = bundle.infolist()
            if member.filename != "handoff.json" or member.file_size > limit:
                raise ValueError("unexpected hand-over archive member")
            data = bundle.read(member)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("hand-over artifact does not match the recorded digest")
        return json.loads(data)

    def post(self, path: str, body: dict):
        return self._call("POST", self._url(path), body)[0]

    def patch(self, path: str, body: dict):
        return self._call("PATCH", self._url(path), body)[0]


def _reason(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code} {error.reason}"
    return f"{type(error).__name__}: {error}"


def machine_block(block: dict) -> str:
    """An issue's machine-readable record; `>` is escaped, so its JSON can never close the comment."""
    return "<!-- post-merge-red " + json.dumps(block, sort_keys=True).replace(">", "\\u003e") + " -->"


def parse_machine_block(body: str) -> dict | None:
    match = _BLOCK.search(body or "")
    try:
        block = json.loads(match.group(1)) if match else None
    except ValueError:
        return None
    return block if isinstance(block, dict) and isinstance(block.get("ids"), list) else None


def system_of(block: dict) -> str:
    return block.get("os") or "linux"


def issue_key(block: dict) -> tuple:
    """One open issue per system and culprit pull request (or commit), one per system and kind without blame."""
    if block.get("kind") == "culprit":
        return (system_of(block), "culprit", block.get("pr") or block.get("culprit"))
    return (system_of(block), block.get("kind"))


def size_base(block: dict) -> str | None:
    """The commit a size test of this issue is measured against: before the culprit, else the check's base. An issue
    keeps its own; a size red another base measures joins an issue of that base (``_upsert_issue``)."""
    return block.get("last_good") if block.get("kind") == "culprit" else block.get("base")


def _sized(block: dict) -> bool:
    return any(nodeid.startswith(SIZE_PREFIX) for nodeid in block["ids"])


def _joins(issue: dict, block: dict) -> bool:
    """Size tests join only an issue that measures them against their base, or one without a size test or base yet,
    which takes theirs: an issue of their key that measures another base keeps it, and they get one of their own."""
    measured = size_base(issue)
    return (not _sized(block) or not size_base(block) or measured == size_base(block)
            or not (measured or _sized(issue)))


def _named(nodeid: str) -> str:
    if nodeid == NODE_LANE:
        return f"`{nodeid}` (the complete Node lane, `python scripts/run_tests.py --node`)"
    if nodeid.startswith(LINT_PREFIX):
        return f"`{nodeid}` (a finding of `ruff check --select F`)"
    if nodeid.startswith(SIZE_PREFIX):
        return f"`{nodeid}` (a test of the size check, `pytest -m size_ratchet`)"
    return f"`{nodeid}`"


def _size_hint(block: dict) -> str:
    """How to see what a size test found, and how it stops being tracked: the release's own lane against the measured base."""
    modules = sorted({nodeid[len(SIZE_PREFIX):].split("::", 1)[0] for nodeid in block["ids"]
                      if nodeid.startswith(SIZE_PREFIX)})
    if not modules:
        return ""
    later = (" Growth a later merge's `book-growth` label approved still counts against that base: the check stays red "
             "until the text is no larger than it was there." if BOOK_BUDGETS in modules else "")
    return (f" The size check measures against `{size_base(block)}`: `OURO_SIZE_RATCHET_BASE_REF="
            f"{size_base(block)} python -m pytest -m size_ratchet {' '.join(modules)}` shows what grew. A size test "
            f"stops being tracked only by passing against that base; a renamed one is taken off by hand.{later}")


def issue_title(block: dict, pr: dict | None) -> str:
    on = f"Post-merge red on {SYSTEMS[system_of(block)]}"
    if block["kind"] == "culprit":
        return (f"{on}: #{pr['number']} {pr.get('title', '')}".strip()[:120] if pr
                else f"{on}: {block['culprit'][:12]}")
    if block["kind"] == "pre-existing":
        return f"{on}: tests that already failed before the checked merges"
    return f"{on}: failing tests without a found culprit"


def issue_body(block: dict, *, head: str, run_url: str, author: str = "", seen: str = "Found failing") -> str:
    system = SYSTEMS[system_of(block)]
    if block["kind"] == "culprit":
        merge = f"#{block['pr']}" if block.get("pr") else f"commit `{block['culprit'][:12]}`"
        opening = (f"Tests on `ouroboros` fail on {system} since {merge}" + (f" by `{author}`" if author else "")
                   + f" landed. First bad commit: `{block['culprit']}`; last good commit: `{block.get('last_good')}`.")
    elif block["kind"] == "pre-existing":
        opening = (f"These tests already failed on {system} at the base of the post-merge check, so no merge "
                   "of its window is blamed.")
    else:
        opening = f"These tests fail on `ouroboros` on {system}, but the post-merge check could not find the merge that broke them."
    rows = "\n".join(f"- {_named(nodeid)}" for nodeid in block["ids"])
    sized = sum(nodeid.startswith(SIZE_PREFIX) for nodeid in block["ids"])  # a size test is never found gone: `_size_hint`
    gone = ("" if sized == len(block["ids"]) else ", or finds one outside the size check gone" if sized
            else ", or finds it gone")
    return (f"{opening}\n\n{rows}\n\n{seen} at `{head}`: {run_url}\n\n{_FORWARD}{_size_hint(block)} The "
            f"post-merge check takes a test off this list when a check on {system} runs it and it passes{gone}, "
            f"and closes this issue when none is left.\n\n{machine_block(block)}\n")


def find_merged_pr(github, sha: str) -> dict | None:
    """The pull request whose merge into `ouroboros` is the commit `sha`, if one is."""
    for pr in github.pages(f"commits/{sha}/pulls"):
        if pr.get("merge_commit_sha") == sha and pr.get("merged_at") and (pr.get("base") or {}).get("ref") == BRANCH:
            return pr
    return None


def open_issues(github) -> list[dict]:
    """Open `post-merge-red` issues that carry a machine block: [{"number", "block", "author"}]."""
    rows = []
    for item in github.pages(f"issues?state=open&labels={LABEL}"):
        block = parse_machine_block(item.get("body") or "")
        if "pull_request" not in item and block is not None:
            author = _AUTHOR.search(item.get("body") or "") if block.get("kind") == "culprit" else None
            observation = _OBSERVED.findall(item.get("body") or "")
            rows.append({"number": item["number"], "block": block, "author": author.group(1) if author else "",
                        "observed": observation[-1] if observation else None})
    return rows


def ensure_label(github) -> None:
    try:
        github.get(f"labels/{LABEL}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        try:
            github.post("labels", {"name": LABEL, "color": "b60205",
                                   "description": "A merge into ouroboros broke tests (post-merge check)"})
        except urllib.error.HTTPError as again:
            if again.code != 422:  # the other leg created it in between
                raise


def _at_or_before(older, newer, ancestor) -> bool:
    """Whether `older` is at or before `newer` on the first-parent line, as far as `ancestor` can prove it."""
    return older == newer or bool(older and newer and ancestor and ancestor(older, newer))


def _owed_key(nodeid: str, base) -> tuple:
    """What one recorded red is owed as. A size red is owed once per base it was measured against: an issue or
    a pass for one base says nothing about another. Any other red is owed by its exact ID."""
    return (nodeid, base if nodeid.startswith(SIZE_PREFIX) else None)


def _prior_facts(prior_failures, head, ancestor) -> dict:
    """The recorded failures of earlier attempts this run may still owe a hand-over, read defensively:
    `rows` keeps each entry as given (validated), `ids` every recorded ID, `latest` the most recent record
    per obligation (`_owed_key`), and `beyond` the obligations whose latest known failing head this run's
    head has not reached — or cannot be shown to have reached: this run settles nothing about those, so a
    late rerun of an old window cannot retire a failure a newer run recorded."""
    rows, latest, stuck = [], {}, set()
    for entry in prior_failures or ():
        if not isinstance(entry, dict):
            continue
        ids = sorted({nodeid for nodeid in entry.get("ids") or () if isinstance(nodeid, str) and nodeid})
        if not ids:
            continue
        run, attempt = entry.get("run"), entry.get("attempt")
        rows.append({"head": entry.get("head") if isinstance(entry.get("head"), str) and entry.get("head") else None,
                     "run": run if type(run) is int and run >= 1 else None,
                     "attempt": attempt if type(attempt) is int and attempt >= 1 else None,
                     "base": entry.get("base") if isinstance(entry.get("base"), str) and entry.get("base") else None,
                     "ids": ids})
        for key in (_owed_key(nodeid, rows[-1]["base"]) for nodeid in ids):
            if key in stuck:
                continue
            chosen = latest.get(key)
            if chosen is None or _at_or_before(chosen["head"], rows[-1]["head"], ancestor):
                latest[key] = rows[-1]
            elif not _at_or_before(rows[-1]["head"], chosen["head"], ancestor):
                stuck.add(key)  # two records this run cannot order: the guard stays closed for it
    beyond = {key for key, row in latest.items()
              if key in stuck or not _at_or_before(row["head"], head, ancestor)}
    return {"rows": rows, "ids": {nodeid for row in rows for nodeid in row["ids"]},
            "latest": latest, "beyond": beyond}


def _observation_link(repo: str, head, run, attempt) -> str:
    """The page a recorded failure points to, built from the repository's own name and the numbers a
    digest-verified receipt carries — never from a URL the receipt itself might carry. A run and
    attempt number name the exact attempt; a record without them falls back to its commit; without
    a head there is nothing trustworthy to point at."""
    if type(run) is int and run >= 1 and type(attempt) is int and attempt >= 1:
        return f"https://github.com/{repo}/actions/runs/{run}/attempts/{attempt}"
    return f"https://github.com/{repo}/commit/{head}" if head else ""


def route(github, *, classification: dict, attribution: dict, head: str, run_url: str, recheck,
          size_recheck=None, system: str = "linux", same_run=None, checkpoint=None, owed=(), retire=True,
          removed=None, ancestor=None, owed_same_run=None, prior_failures=(), origin=None) -> dict:
    """Hand each red over, then take off this system's issues what passes at the head.

    A culprit's pull request gets one comment and one open issue per culprit pull request is
    created or extended; reds without blame not tracked yet get one issue per kind (per base for
    size tests). `prior_failures` carries the failures earlier attempts could not hand over, one
    {head, run, attempt, base, ids} group per measurement, and `origin` this run's same metadata;
    what is still owed after this run returns as `record["awaiting_handoff"]` in that same shape,
    each record keeping its original head, run, attempt and base, while `record["untracked"]`
    stays the ID-only projection for present readers. A recorded red is owed by its exact ID, a
    size red once per base it was measured against (`_owed_key`). One this run still owes is
    re-filed under its own head and its own attempt link — never this run's, which did not see it
    fail — and only an open issue that lists it (for a size red, one measuring the base its
    record names), a whole pass (a Node lane that passed only its retry is none), or outside the
    size check a confirmed disappearance on this system and on or after its failing head retires
    it; a record whose failing head this run has not reached is settled by nothing, whatever an
    older issue lists or this run saw fail. `checkpoint(record)` then receives that hand-over
    before any recheck starts, so a
    step killed during the recheck still leaves it. `same_run(ids)` answers what this run already
    settles: "passed" at the head on this system, "kept" for an id it owes no recheck (the run ran
    nothing of its existing module, or its own report was no whole pass); `recheck(ids)` runs the
    others once and answers `outcomes` values; `size_recheck(base, ids)` does so for size ids,
    measured against `base` — a recorded size red is re-measured against the base its record
    names, and only a pass against that base takes it off. An issue loses each id that passes,
    named in a comment, and closes when none is left. A failed GitHub call becomes a warning in
    the record and the hand-over goes on. Issues of another system are never read further,
    extended or closed.
    """
    record = {"format": 1, "os": system, "comments": [], "created": [], "updated": [], "closed": [],
              "narrowed": [], "warnings": [], "same_run": {}, "rechecked": {}, "handed": [],
              "awaiting_handoff": [], "complete": False}

    def call(what, function, *args):
        try:
            return True, function(*args)
        except Exception as error:  # noqa: BLE001 -- reporting never fails the check
            record["warnings"].append(f"{what}: {_reason(error)}")
            print(f"::warning title=Post-merge hand-over::{what}: {_reason(error)}", flush=True)
            return False, None

    failed = {row["id"] for row in classification.get("failed", [])}
    known, issues = call("list the open post-merge-red issues", open_issues, github)
    issues = [issue for issue in issues or [] if system_of(issue["block"]) == system]
    verdicts = attribution.get("verdicts") or [{"id": nodeid, "verdict": "unattributed"} for nodeid in failed]
    base = (classification.get("size") or {}).get("base") or classification.get("base")
    last_good = {row["id"]: row["last_good"] for row in verdicts if row.get("verdict") == "culprit" and row.get("last_good")}
    standing = {row["id"] for row in verdicts if row.get("verdict") == "pre-existing"}
    prior = _prior_facts(prior_failures, head, ancestor)
    report = owed is not None or bool(prior["ids"])

    def carried(nodeid):
        """Whether an open issue of this system carries `nodeid` into the next window: one that lists it, for a size
        test one whose base sees this red: the window's base, the last good commit before its culprit, or any base
        once it already fails at the window's base. Grouping and the confirmation below ask the same."""
        for issue in issues:
            measured = size_base(issue["block"])
            if nodeid in issue["block"]["ids"] and (not nodeid.startswith(SIZE_PREFIX) or bool(measured) and (
                    nodeid in standing or measured in (base, last_good.get(nodeid)))):
                return True
        return False

    def tracked(key):
        """Whether an open issue already carries this obligation (`_owed_key`): for a recorded size red, only one
        that measures the exact base its record names can recheck it — a pass against another base retires
        nothing."""
        nodeid, wanted = key
        if key not in prior["latest"] or not nodeid.startswith(SIZE_PREFIX):
            return carried(nodeid)
        return any(nodeid in issue["block"]["ids"] and bool(size_base(issue["block"]))
                   and size_base(issue["block"]) == wanted for issue in issues)

    def current(issue):
        observed = issue.get("observed")
        return observed == head or bool(observed and ancestor and ancestor(observed, head))

    groups = defaultdict(list)
    for verdict in verdicts:
        if verdict["id"] in failed and verdict["verdict"] == "culprit":
            groups[("culprit", verdict["culprit"], verdict.get("last_good"))].append(verdict["id"])
        elif verdict["id"] in failed and not carried(verdict["id"]):
            groups[(verdict["verdict"],)].append(verdict["id"])
    if groups:
        call("create the post-merge-red label", ensure_label, github)
    for key in sorted(groups, key=str):
        block, pr = {"format": 1, "os": system, "kind": key[0], "ids": sorted(groups[key])}, None
        if base:
            block["base"] = base
        if key[0] == "culprit":
            pr = call(f"find the pull request of {key[1][:12]}", find_merged_pr, github, key[1])[1]
            block.update(culprit=key[1], last_good=key[2], pr=pr["number"] if pr else None)
        number = _upsert_issue(call, github, issues if known else None, block, pr, head, run_url, record, current)
        record["handed"].append({"ids": block["ids"], "pr": block.get("pr"), "issue": number})
        # The author hears when the tracking issue opens or gains tests, not on every run it stays
        # red; without a tracking issue (the API failed) the hand-over still reaches the author.
        if pr and (number is None or number in record["created"] + record["updated"]):
            login = (pr.get("user") or {}).get("login", "")
            text = ((f"@{login} " if login else "") + f"the post-merge check found tests on `ouroboros` that fail on "
                    f"{SYSTEMS[system]} since this pull request was merged:\n\n"
                    + "\n".join(f"- {_named(n)}" for n in block["ids"])
                    + f"\n\nFirst bad commit: `{block['culprit']}` (this merge); last good commit: "
                    f"`{block['last_good']}`. Checked at `{head}`: {run_url}\n\n{_FORWARD}{_size_hint(block)}"
                    + (f" Tracking issue: #{number}." if number else ""))
            if call(f"comment on #{pr['number']}", github.post, f"issues/{pr['number']}/comments", {"body": text})[0]:
                record["comments"].append(pr["number"])
    node_flaky = any(row["id"] == NODE_LANE for row in classification.get("flaky", []))
    node_whole = not node_flaky and (classification.get("selection") or {}).get("node") == "success"
    # A Node lane that failed and then passed its retry is green for an issue, as it always was. It is no whole
    # pass: while an earlier attempt's record of the lane is owed, it retires neither that record nor its issue.
    node_green = (NODE_LANE not in failed and (node_whole
                                               or node_flaky and NODE_LANE not in prior["ids"]))
    retry_passed = {row["id"] for row in classification.get("flaky", ()) if row.get("retry") == "passed"}
    # What earlier attempts left owed: each recorded red as its own obligation (`_owed_key`), whatever this
    # run saw of that ID, and each ID owed without a record under the meaning it had. A flaky pass settles
    # nothing about a recorded red: it is no whole pass, no issue and no disappearance.
    unrecorded = set(owed or ()) - prior["ids"] - failed - retry_passed
    pending = set(prior["latest"]) | {(nodeid, None) for nodeid in unrecorded}
    # A whole pass, a confirmed disappearance or an open issue can settle a historical red.
    # A setup skip, an unrelated green test or an unreadable result leaves the exact ID owed.
    checks = sorted(key[0] for key in pending - prior["beyond"]
                    if key[0] != NODE_LANE and not key[0].startswith((SIZE_PREFIX, LINT_PREFIX))
                    and key[0] not in failed and not tracked(key))
    observe_owed = owed_same_run or same_run
    observed = observe_owed(checks) if observe_owed and checks else {}
    size_measured = (classification.get("size") or {}).get("base")

    def sized_pass(key):
        """Whether this run's own size gate settles the obligation: it measured, and a recorded red was measured
        against the base it now measures — a pass against another base discharges nothing."""
        nodeid, wanted = key
        if not (nodeid.startswith(SIZE_PREFIX) and classification.get("size_checked")
                and nodeid in (classification.get("size_passed") or ())):
            return False
        if key not in prior["latest"]:
            return True  # an ID-only carry keeps its meaning: the driver pairs it with the window's base
        return bool(size_measured) and wanted == size_measured

    outstanding = set()  # what earlier attempts left owed and this run leaves owed too, by obligation
    for key in pending:
        nodeid = key[0]
        if key in prior["beyond"]:
            outstanding.add(key)  # its newest known failure is at a head this run has not reached
            continue
        if tracked(key):
            continue
        # A red this run saw fail again is settled by no pass of this run: only a tracker settles it.
        if nodeid not in failed and (
                observed.get(nodeid) == "passed" or nodeid == NODE_LANE and node_green
                or nodeid.startswith(LINT_PREFIX) and classification.get("lint_checked")
                or sized_pass(key) or nodeid in checks and removed and removed(nodeid)):
            continue
        outstanding.add(key)
    # What is still owed of the recorded failures is re-filed under each record's own head and
    # attempt link — never this run's, which did not see them fail — before any recheck starts.
    # Only what was re-filed can have gained a tracker by it: a record beyond this head stays owed even
    # when an older issue lists its ID.
    # If the current filing did not confirm, another POST for the same obligation could duplicate it.
    # Keep both observations for the next run, which first lists issues again. A size test on a
    # different comparison base is a different obligation.
    attempted = {_owed_key(nodeid, base) for nodeid in failed}
    refile = {key for key in outstanding - prior["beyond"] - attempted
              if key in prior["latest"] and prior["latest"][key]["head"]}
    if refile:
        _file_prior_reds(call, github, issues if known else None, record, system=system, ancestor=ancestor,
                         latest=prior["latest"], keys=refile)
        outstanding -= {key for key in refile if tracked(key)}
    # Only an open issue (including its successful creation/extension above) carries a
    # red into the next window. A comment or a refused PATCH does not.
    unfiled = {nodeid for nodeid in failed if not carried(nodeid)}

    def publish():
        """Record what is still owed, with its ID-only projection for present readers."""
        record["awaiting_handoff"] = _awaiting_records(prior["rows"], outstanding, unfiled, head, origin, base)
        owed_now = sorted({nodeid for group in record["awaiting_handoff"] for nodeid in group["ids"]})
        record["untracked"] = owed_now if report else None

    publish()
    if checkpoint is not None:
        call("record the hand-over", checkpoint, record)
    # Untracked reds need a targeted collection even in a module the selected run did not touch.
    # Share the existing session, after saving the conservative pre-recheck hand-over.
    extra = {nodeid for nodeid in checks if (nodeid, None) in outstanding} - observed.keys()
    beyond_ids = {nodeid for nodeid, _ in prior["beyond"]}  # by ID: no issue loses a test recorded failing later
    current_issues = [issue for issue in issues if current(issue)] if known and retire else []
    if current_issues or extra:
        status = _close_green(call, github, current_issues, failed, node_green, head, run_url, recheck, record,
                              lint_checked=bool(classification.get("lint_checked")), size_recheck=size_recheck,
                              same_run=same_run, extra=extra, beyond=beyond_ids)
        outstanding -= {(nodeid, None) for nodeid in extra if status.get(nodeid) in ("passed", "absent")}
    if known and issues and not retire:
        record["warnings"].append("An older rerun cannot retire this branch’s open failures.")
    record["complete"] = known  # unreadable open issues: no recheck ran, whatever this run handed over
    publish()
    return record


def _close_green(call, github, issues, failed, node_green, head, run_url, recheck, record, *,
                 lint_checked=False, size_recheck=None, same_run=None, extra=(), beyond=()) -> dict:
    ids = sorted({nodeid for issue in issues for nodeid in issue["block"]["ids"]
                  if not nodeid.startswith((LINT_PREFIX, SIZE_PREFIX))} - failed - {NODE_LANE} - set(beyond))
    record["same_run"] = known = dict(same_run(ids)) if same_run and ids else {}
    rest = sorted((set(ids) | set(extra)) - known.keys())
    record["rechecked"] = dict(recheck(rest)) if rest else {}
    status = {**record["rechecked"], **known}
    sized = defaultdict(set)  # base -> the size ids measured against it
    for issue in issues:
        sized[size_base(issue["block"])].update(nodeid for nodeid in issue["block"]["ids"]
                                                if nodeid.startswith(SIZE_PREFIX) and nodeid not in failed)
    size_status = {}
    for base in sorted(base for base, members in sized.items() if base and members and size_recheck):
        size_status[base] = size_recheck(base, sorted(sized[base]))
        record["rechecked"].update({f"{nodeid} against {base[:12]}": value for nodeid, value in size_status[base].items()})

    def green(block, nodeid):
        if nodeid in beyond:
            return False  # a red recorded at a head this run has not reached is not green here
        if nodeid == NODE_LANE:
            return node_green
        if nodeid.startswith(LINT_PREFIX):  # the head's whole tree was linted: a finding not seen is gone
            return lint_checked and nodeid not in failed
        if nodeid.startswith(SIZE_PREFIX):  # a growth inside a later base is invisible to that check, and a
            # missing size test proves nothing: only a pass against the base the issue measures takes it off
            return nodeid not in failed and (size_status.get(size_base(block)) or {}).get(nodeid) == "passed"
        return nodeid not in failed and status.get(nodeid) in ("passed", "absent")

    for issue in issues:
        block, number = issue["block"], issue["number"]
        gone = [nodeid for nodeid in block["ids"] if green(block, nodeid)]
        system = SYSTEMS[system_of(block)]
        if gone and len(gone) == len(block["ids"]):
            text = (f"Green at `{head}` on {system}: every test this issue tracks passes on `ouroboros` or no longer "
                    f"exists ({run_url}). Closing.")
            if (call(f"comment on #{number}", github.post, f"issues/{number}/comments", {"body": text})[0]
                    and call(f"close #{number}", github.patch, f"issues/{number}",
                             {"state": "closed", "state_reason": "completed"})[0]):
                record["closed"].append(number)
        elif gone:
            # The machine block keeps what is still not green; the comments are the record of what left it.
            rest = {**block, "ids": [nodeid for nodeid in block["ids"] if nodeid not in gone]}
            texts = _retired_comments(gone, head=head, run_url=run_url, system=system, left=len(rest["ids"]))
            if (all(call(f"comment on #{number}", github.post, f"issues/{number}/comments", {"body": text})[0]
                    for text in texts)
                    and call(f"narrow #{number}", github.patch, f"issues/{number}", {"body": issue_body(
                        rest, head=head, run_url=run_url, author=issue.get("author", ""), seen="Still not passing")})[0]):
                issue["block"] = rest
                record["narrowed"].append({"issue": number, "ids": gone})
    return status


def _retired_comments(ids, *, head: str, run_url: str, system: str, left: int) -> list[str]:
    """The comments that name the ids leaving an issue, each within GitHub's comment limit."""
    opening = (f"Green at `{head}` on {system} ({run_url}): these pass on `ouroboros` or no longer exist, so this "
               "issue no longer tracks them:\n\n")
    closing = f"\n\nStill tracked here: {left}."
    chunks, rows = [], []
    for row in (f"- {_named(nodeid)}" for nodeid in ids):
        if rows and len(opening) + len(closing) + sum(len(line) + 1 for line in rows) + len(row) > _COMMENT_LIMIT:
            chunks.append(rows)
            rows = []
        rows.append(row)
    return [opening + "\n".join(chunk) + closing for chunk in [*chunks, rows]]


def _tracks(issue, block) -> bool:
    return (isinstance(issue, dict) and issue.get("state") == "open" and parse_machine_block(issue.get("body")) == block
            and any(isinstance(label, dict) and label.get("name") == LABEL for label in issue.get("labels") or []))


def _upsert_issue(call, github, issues, block, pr, head, run_url, record, current) -> int | None:
    if issues is None:
        return None  # the open issues are unknown: a new one could duplicate one of them
    author = ((pr or {}).get("user") or {}).get("login", "")
    existing = next((issue for issue in issues if issue["number"] not in record["closed"]
                     and issue_key(issue["block"]) == issue_key(block) and _joins(issue["block"], block)), None)
    if existing is None:
        ok, issue = call("create an issue", github.post, "issues", {
            "title": issue_title(block, pr), "labels": [LABEL],
            "body": issue_body(block, head=head, run_url=run_url, author=author)})
        if not (ok and isinstance(issue, dict) and issue.get("number")):
            return None
        record["created"].append(issue["number"])
        if _tracks(issue, block):
            issues.append({"number": issue["number"], "block": block, "author": author, "observed": head})
        else:
            record["warnings"].append(f"issue #{issue['number']} did not confirm the open {LABEL} hand-over")
        return issue["number"]
    number, known = existing["number"], set(existing["block"]["ids"])
    if added := sorted(set(block["ids"]) - known):
        if not current(existing):
            record["warnings"].append(f"issue #{number} has no observation at or before this head; its record was kept")
            return None
        # What the issue recorded stays; a field its block predates (an issue opened before the
        # legs has no `base`) comes from this run, or a size id added to it could never be measured.
        merged = {**block, **existing["block"], "ids": sorted(known | set(added))}
        ok, updated = call(f"update #{number}", github.patch, f"issues/{number}", {"body": issue_body(
            merged, head=head, run_url=run_url, author=author or existing.get("author", ""))})
        if ok and _tracks(updated, merged):
            existing.update(block=merged, observed=head)
            record["updated"].append(number)
            call(f"comment on #{number}", github.post, f"issues/{number}/comments", {"body": (
                f"Also failing at `{head}`:\n\n" + "\n".join(f"- {_named(n)}" for n in added) + f"\n\n{run_url}")})
        elif ok:
            issues.remove(existing)
            record["warnings"].append(f"issue #{number} did not confirm the updated {LABEL} hand-over")
    return number


def _file_prior_reds(call, github, issues, record, *, system, ancestor, latest, keys) -> None:
    """Re-file the recorded obligations (`_owed_key`) this run still owes, each group under the head and attempt
    link of its own record: the issue that tracks a red names the commit that saw it fail, and measures a size
    red against the base that record names. The observation guard is measured at the record's head, so an
    issue holding a newer observation is never extended with an older one."""
    if issues is None:
        return  # the open issues are unknown: a new one could duplicate one of them
    call("create the post-merge-red label", ensure_label, github)
    repo, groups = getattr(github, "repo", ""), defaultdict(list)
    for key in keys:
        row = latest[key]
        groups[(row["head"], row["run"], row["attempt"], row["base"])].append(key[0])
    for head, run, attempt, measured in sorted(groups, key=str):
        block = {"format": 1, "os": system, "kind": "unattributed",
                 "ids": sorted(groups[(head, run, attempt, measured)])}
        if measured:
            block["base"] = measured
        number = _upsert_issue(call, github, issues, block, None, head,
                               _observation_link(repo, head, run, attempt), record,
                               lambda issue, at=head: issue.get("observed") == at
                               or bool(issue.get("observed") and ancestor and ancestor(issue["observed"], at)))
        record["handed"].append({"ids": block["ids"], "pr": None, "issue": number})


def _awaiting_records(rows, outstanding, unfiled, head, origin, base) -> list:
    """The hand-over this run still owes, as records in the shape `prior_failures` arrives in: each recorded
    group keeps its head, run, attempt and size base with the IDs still owed of it, this run's own unconfirmed
    reds (`unfiled`) follow under this run's identity, and an ID owed without any recorded source keeps an
    empty one rather than an invented head. `outstanding` holds obligations (`_owed_key`), so a size red
    settled against one base leaves the record that measured it against another."""
    records = []
    for row in rows:
        ids = [nodeid for nodeid in row["ids"] if _owed_key(nodeid, row["base"]) in outstanding]
        if ids:
            records.append({"head": row["head"], "run": row["run"], "attempt": row["attempt"],
                            "base": row["base"], "ids": ids})
    fresh_ids = sorted(unfiled)
    if fresh_ids:
        origin = origin if isinstance(origin, dict) else {}
        run, attempt, measured = origin.get("run"), origin.get("attempt"), origin.get("base")
        records.append({"head": head, "run": run if type(run) is int and run >= 1 else None,
                        "attempt": attempt if type(attempt) is int and attempt >= 1 else None,
                        "base": measured if isinstance(measured, str) and measured else base, "ids": fresh_ids})
    recorded = {_owed_key(nodeid, row["base"]) for row in rows for nodeid in row["ids"]}
    legacy = sorted(nodeid for nodeid, _ in outstanding - recorded)
    if legacy:
        records.append({"head": None, "run": None, "attempt": None, "base": None, "ids": legacy})
    return records
