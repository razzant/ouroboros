"""The post-merge check's hand-over to GitHub: one comment per culprit pull request, one issue per red.

scripts/post_merge_check.py runs the steps and documents them; this is its `route` step, with the
REST calls it makes through the workflow token (tests pass a fake). Every leg of the check runs on
one operating system and owns the `post-merge-red` issues of that system: the system is part of an
issue's key, title and machine block, and a leg extends and closes only its own issues. A block
that names no system is a Linux one: the check had only its Linux leg before 2026-10-28. Stdlib only.
"""
from __future__ import annotations

from collections import defaultdict
import json
import re
import urllib.error
import urllib.request

BRANCH = "ouroboros"
LABEL = "post-merge-red"
NODE_LANE = "node-lane"
LINT_PREFIX = "ruff "  # a lint finding's id: "ruff <code> <path> <name>"
SIZE_PREFIX = "size "  # a size check test's id: "size <nodeid>"
BOOK_BUDGETS = "tests/test_reference_book_budgets.py"  # the size check's module that measures the books and prompts
SYSTEMS = {"linux": "Linux", "windows": "Windows"}
_BLOCK = re.compile(r"<!-- post-merge-red (\{.*?\}) -->", re.S)
_AUTHOR = re.compile(r" by `([^`\n]+)` landed\. First bad commit")  # a culprit issue's opening (issue_body)
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
    """The commit a size test of this issue is measured against: before the culprit, else the check's base."""
    return block.get("last_good") if block.get("kind") == "culprit" else block.get("base")


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
            rows.append({"number": item["number"], "block": block, "author": author.group(1) if author else ""})
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


def route(github, *, classification: dict, attribution: dict, head: str, run_url: str, recheck,
          size_recheck=None, system: str = "linux", same_run=None, checkpoint=None) -> dict:
    """Hand each red over, then take off this system's issues what passes at the head.

    A culprit's pull request gets one comment and one open issue per culprit pull request is
    created or extended; reds without blame not tracked yet get one issue per kind. `checkpoint(record)`
    then receives that hand-over before any recheck starts, so a step killed during the recheck still
    leaves it. `same_run(ids)` answers what this run already settles: "passed" at the head on this system,
    "kept" for an id it owes no recheck (the run ran nothing of its existing module, or its own report was no
    whole pass);
    `recheck(ids)` runs the others once and answers `outcomes` values; `size_recheck(base, ids)` does
    so for size ids, measured against `base`. An issue loses each id that passes, named in a comment,
    and closes when none is left. A failed GitHub call becomes a warning in the record and the
    hand-over goes on. Issues of another system are never read further, extended or closed.
    """
    record = {"format": 1, "os": system, "comments": [], "created": [], "updated": [], "closed": [],
              "narrowed": [], "warnings": [], "same_run": {}, "rechecked": {}, "handed": [], "complete": False}

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
    tracked = {nodeid for issue in issues for nodeid in issue["block"]["ids"]}
    groups = defaultdict(list)
    for verdict in attribution.get("verdicts") or [{"id": nodeid, "verdict": "unattributed"} for nodeid in failed]:
        if verdict["id"] in failed and verdict["verdict"] == "culprit":
            groups[("culprit", verdict["culprit"], verdict.get("last_good"))].append(verdict["id"])
        elif verdict["id"] in failed and verdict["id"] not in tracked:
            groups[(verdict["verdict"],)].append(verdict["id"])
    if groups:
        call("create the post-merge-red label", ensure_label, github)
    base = (classification.get("size") or {}).get("base") or classification.get("base")
    for key in sorted(groups, key=str):
        block, pr = {"format": 1, "os": system, "kind": key[0], "ids": sorted(groups[key])}, None
        if base:
            block["base"] = base
        if key[0] == "culprit":
            pr = call(f"find the pull request of {key[1][:12]}", find_merged_pr, github, key[1])[1]
            block.update(culprit=key[1], last_good=key[2], pr=pr["number"] if pr else None)
        number = _upsert_issue(call, github, issues if known else None, block, pr, head, run_url, record)
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
    if checkpoint is not None:
        call("record the hand-over", checkpoint, record)
    if known and issues:
        node_green = (NODE_LANE not in failed and ((classification.get("selection") or {}).get("node") == "success"
                                                   or any(row["id"] == NODE_LANE for row in classification.get("flaky", []))))
        _close_green(call, github, issues, failed, node_green, head, run_url, recheck, record,
                     lint_checked=bool(classification.get("lint_checked")), size_recheck=size_recheck,
                     same_run=same_run)
    record["complete"] = known  # unreadable open issues: no recheck ran, whatever this run handed over
    return record


def _close_green(call, github, issues, failed, node_green, head, run_url, recheck, record, *,
                 lint_checked=False, size_recheck=None, same_run=None) -> None:
    ids = sorted({nodeid for issue in issues for nodeid in issue["block"]["ids"]
                  if not nodeid.startswith((LINT_PREFIX, SIZE_PREFIX))} - failed - {NODE_LANE})
    record["same_run"] = known = dict(same_run(ids)) if same_run and ids else {}
    rest = [nodeid for nodeid in ids if nodeid not in known]
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
        if nodeid == NODE_LANE:
            return node_green
        if nodeid.startswith(LINT_PREFIX):  # the head's whole tree was linted: a finding not seen is gone
            return lint_checked and nodeid not in failed
        if nodeid.startswith(SIZE_PREFIX):  # a growth inside a later base is invisible to that check
            return nodeid not in failed and (size_status.get(size_base(block)) or {}).get(nodeid) in ("passed", "absent")
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


def _upsert_issue(call, github, issues, block, pr, head, run_url, record) -> int | None:
    if issues is None:
        return None  # the open issues are unknown: a new one could duplicate one of them
    author = ((pr or {}).get("user") or {}).get("login", "")
    existing = next((issue for issue in issues if issue["number"] not in record["closed"]
                     and issue_key(issue["block"]) == issue_key(block)), None)
    if existing is None:
        ok, issue = call("create an issue", github.post, "issues", {
            "title": issue_title(block, pr), "labels": [LABEL],
            "body": issue_body(block, head=head, run_url=run_url, author=author)})
        if not (ok and isinstance(issue, dict) and issue.get("number")):
            return None
        record["created"].append(issue["number"])
        issues.append({"number": issue["number"], "block": block, "author": author})
        return issue["number"]
    number, known = existing["number"], set(existing["block"]["ids"])
    if added := sorted(set(block["ids"]) - known):
        # What the issue recorded stays; a field its block predates (an issue opened before the
        # legs has no `base`) comes from this run, or a size id added to it could never be measured.
        merged = {**block, **existing["block"], "ids": sorted(known | set(added))}
        if call(f"update #{number}", github.patch, f"issues/{number}", {"body": issue_body(
                merged, head=head, run_url=run_url, author=author or existing.get("author", ""))})[0]:
            existing["block"] = merged
            record["updated"].append(number)
            call(f"comment on #{number}", github.post, f"issues/{number}/comments", {"body": (
                f"Also failing at `{head}`:\n\n" + "\n".join(f"- {_named(n)}" for n in added) + f"\n\n{run_url}")})
    return number
