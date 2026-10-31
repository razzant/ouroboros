"""The post-merge check's hand-over (scripts/post_merge_routing.py): comments, issues, closing.

Deterministic: GitHub is a fake in memory and the rechecks are fakes; each leg of the check owns
the `post-merge-red` issues of its own system.
"""
from __future__ import annotations

import urllib.error

import pytest

from scripts import post_merge_routing as routing

HEAD = "f" * 40
LINTED = "ruff F811 tests/t.py fake_route"
SIZED = "size tests/test_smoke.py::test_no_oversized_modules"


def test_the_machine_block_round_trips_through_an_issue_body():
    block = {"format": 1, "kind": "culprit", "pr": 2067, "culprit": "c" * 40, "last_good": "b" * 40,
             "ids": ["tests/t.py::test_x[a-->b]", "tests/t.py::test_y[<x>]"]}
    body = routing.issue_body(block, head=HEAD, run_url="https://example.invalid/run/1", author="dev")
    assert "-->" not in routing.machine_block(block)[:-3]
    assert routing.parse_machine_block(body) == block
    assert routing.parse_machine_block("no block here") is None
    assert routing.parse_machine_block("<!-- post-merge-red {not json} -->") is None


class FakeGitHub:
    """Issues, pull requests and labels in memory, with failures on demand."""

    def __init__(self, *, issues=(), prs=None, label=False, fail=()):
        self.issues = {issue["number"]: dict(issue, state="open") for issue in issues}
        self.prs, self.label, self.fail = prs or {}, label, set(fail)
        self.comments, self.posts = [], []

    def _check(self, method, path):
        if any((method, path.split("?")[0]) == entry or (method, None) == entry for entry in self.fail):
            raise urllib.error.HTTPError(path, 502, "Bad Gateway", None, None)

    def get(self, path):
        self._check("GET", path)
        if path == f"labels/{routing.LABEL}" and not self.label:
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
        return {}

    def pages(self, path):
        self._check("GET", path)
        if path.startswith("issues?"):
            return [issue for issue in self.issues.values() if issue["state"] == "open"]
        return self.prs.get(path.split("/")[1], [])

    def post(self, path, body):
        self._check("POST", path)
        self.posts.append((path, body))
        if path == "labels":
            self.label = True
        elif path == "issues":
            number = 100 + len(self.issues)
            self.issues[number] = {"number": number, "body": body["body"], "title": body["title"], "state": "open"}
            return {"number": number}
        elif path.endswith("/comments"):
            self.comments.append((int(path.split("/")[1]), body["body"]))
        return {}

    def patch(self, path, body):
        self._check("PATCH", path)
        self.issues[int(path.split("/")[1])].update(body)
        return {}


def _issue(number, block):
    return {"number": number, "body": "text\n" + routing.machine_block({"format": 1, **block})}


def _pr(number, sha, login="dev"):
    return {"number": number, "merge_commit_sha": sha, "merged_at": "2026-10-21T00:00:00Z",
            "base": {"ref": "ouroboros"}, "user": {"login": login}, "title": f"Change {number}"}


def _route(github, failed, verdicts, *, recheck=None, flaky=(), node="success", classification=None, **options):
    return routing.route(github, classification={"failed": [{"id": nodeid} for nodeid in failed],
                                                 "flaky": [{"id": nodeid} for nodeid in flaky],
                                                 "selection": {"node": node}, **(classification or {})},
                         attribution={"verdicts": verdicts}, head=HEAD, run_url="https://example.invalid/run/7",
                         recheck=recheck or (lambda ids: pytest.fail(f"nothing to recheck: {ids}")), **options)


def test_a_culprit_gets_one_pr_comment_and_one_new_issue():
    culprit = "c" * 40
    github = FakeGitHub(prs={culprit: [_pr(2066, "d" * 40), _pr(2067, culprit, login="author")]})
    verdicts = [{"id": nodeid, "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40}
                for nodeid in ("tests/t.py::test_a", "tests/t.py::test_b")]
    record = _route(github, ["tests/t.py::test_a", "tests/t.py::test_b"], verdicts)
    assert record["created"] == [100] and record["comments"] == [2067] and record["warnings"] == []
    assert record["handed"] == [{"ids": ["tests/t.py::test_a", "tests/t.py::test_b"], "pr": 2067, "issue": 100}]
    assert github.posts[0] == ("labels", {"name": routing.LABEL, "color": "b60205",
                                          "description": "A merge into ouroboros broke tests (post-merge check)"})
    issue = github.issues[100]
    assert issue["title"] == "Post-merge red on Linux: #2067 Change 2067"
    assert routing.parse_machine_block(issue["body"]) == {
        "format": 1, "os": "linux", "kind": "culprit", "pr": 2067, "culprit": culprit, "last_good": "b" * 40,
        "ids": ["tests/t.py::test_a", "tests/t.py::test_b"]}
    [(number, text)] = github.comments
    assert number == 2067 and text.startswith("@author ")
    for fact in ("`tests/t.py::test_a`", "`tests/t.py::test_b`", f"First bad commit: `{culprit}`", "fail on Linux since",
                 f"last good commit: `{'b' * 40}`", "https://example.invalid/run/7", "Tracking issue: #100.",
                 "blocks nothing and reverts nothing", "fix goes forward in a new pull request"):
        assert fact in text


def test_an_open_issue_of_the_same_pull_request_is_extended_not_duplicated():
    culprit = "c" * 40
    github = FakeGitHub(label=True, prs={culprit: [_pr(2067, culprit)]}, issues=[
        _issue(5, {"kind": "culprit", "pr": 2067, "culprit": culprit, "last_good": "b" * 40,
                   "ids": ["tests/t.py::test_a"]})])
    verdicts = [{"id": "tests/t.py::test_new", "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40}]
    record = _route(github, ["tests/t.py::test_new", "tests/t.py::test_a"], verdicts)
    assert record["created"] == [] and record["updated"] == [5] and record["closed"] == []
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == ["tests/t.py::test_a", "tests/t.py::test_new"]
    assert [number for number, _ in github.comments] == [5, 2067]
    assert "labels" not in [path for path, _ in github.posts]


def test_a_red_that_stays_red_does_not_comment_on_its_pull_request_again():
    culprit = "c" * 40
    github = FakeGitHub(prs={culprit: [_pr(2067, culprit, login="author")]})
    verdicts = [{"id": "tests/t.py::test_a", "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40}]
    first = _route(github, ["tests/t.py::test_a"], verdicts)
    again = _route(github, ["tests/t.py::test_a"], verdicts)
    assert first["comments"] == [2067] and first["created"] == [100]
    assert again["comments"] == [] and again["created"] == again["updated"] == []
    assert again["handed"] == [{"ids": ["tests/t.py::test_a"], "pr": 2067, "issue": 100}]
    assert [number for number, _ in github.comments] == [2067]


def test_issues_whose_tests_pass_at_the_head_are_closed_and_others_stay_open():
    github = FakeGitHub(label=True, issues=[
        _issue(5, {"kind": "culprit", "pr": 1, "culprit": "c" * 40, "ids": ["tests/t.py::test_fixed",
                                                                            "tests/t.py::test_deleted"]}),
        _issue(6, {"kind": "pre-existing", "ids": ["tests/t.py::test_still_red"]}),
        _issue(7, {"kind": "unattributed", "ids": ["tests/t.py::test_red_this_run"]}),
        _issue(8, {"kind": "culprit", "pr": 2, "culprit": "d" * 40, "ids": [routing.NODE_LANE]}),
    ])
    asked = []

    def recheck(ids):
        asked.append(ids)
        return {"tests/t.py::test_fixed": "passed", "tests/t.py::test_deleted": "absent",
                "tests/t.py::test_still_red": "failed"}

    record = _route(github, ["tests/t.py::test_red_this_run"],
                    [{"id": "tests/t.py::test_red_this_run", "verdict": "pre-existing"}], recheck=recheck)
    # This run's own failures are not run again; the Node lane's result is this run's.
    assert asked == [["tests/t.py::test_deleted", "tests/t.py::test_fixed", "tests/t.py::test_still_red"]]
    assert record["closed"] == [5, 8] and record["created"] == [] and record["comments"] == []
    assert github.issues[5]["state"] == "closed" and github.issues[5]["state_reason"] == "completed"
    assert github.issues[6]["state"] == github.issues[7]["state"] == "open"
    assert [number for number, _ in github.comments] == [5, 8]
    assert github.comments[0][1].startswith(f"Green at `{HEAD}`")
    record = _route(FakeGitHub(issues=[_issue(8, {"kind": "culprit", "pr": 2, "ids": [routing.NODE_LANE]})]),
                    [], [], node="failure")
    assert record["closed"] == []  # a red Node lane in this run is not green


def test_a_lint_issue_closes_only_once_a_linted_head_no_longer_has_its_finding():
    def run(lint_checked, failed=()):
        github = FakeGitHub(label=True, issues=[_issue(5, {"kind": "culprit", "pr": 1, "culprit": "c" * 40,
                                                           "ids": [LINTED]})])
        record = routing.route(github, classification={"failed": [{"id": nodeid} for nodeid in failed], "flaky": [],
                                                       "selection": {"node": "success"}, "lint_checked": lint_checked},
                               attribution={"verdicts": [{"id": nodeid, "verdict": "unattributed"} for nodeid in failed]},
                               head=HEAD, run_url="https://example.invalid/run/9",
                               recheck=lambda ids: pytest.fail(f"a finding is not a test to rerun: {ids}") if ids else {})
        return record["closed"]

    assert run(True) == [5]
    assert run(True, failed=[LINTED]) == []  # still found at the head
    assert run(False) == []  # ruff did not run: nothing proves the finding gone


def test_reds_without_blame_get_one_issue_unless_already_tracked():
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": "pre-existing", "ids": ["tests/t.py::test_known"]})])
    recheck = lambda ids: {nodeid: "failed" for nodeid in ids}  # noqa: E731
    verdicts = [{"id": "tests/t.py::test_known", "verdict": "pre-existing"},
                {"id": "tests/t.py::test_new_old", "verdict": "pre-existing"},
                {"id": "tests/t.py::test_lost", "verdict": "unattributed", "why": "probe budget"}]
    record = _route(github, [row["id"] for row in verdicts], verdicts, recheck=recheck)
    assert record["updated"] == [5] and record["created"] == [101] and record["comments"] == []
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == ["tests/t.py::test_known",
                                                                            "tests/t.py::test_new_old"]
    assert routing.parse_machine_block(github.issues[101]["body"]) == {"format": 1, "os": "linux", "kind": "unattributed",
                                                                       "ids": ["tests/t.py::test_lost"]}
    assert "@" not in github.issues[101]["body"]


def test_api_errors_become_warnings_and_the_hand_over_goes_on():
    culprit = "c" * 40
    github = FakeGitHub(prs={culprit: [_pr(2067, culprit)]}, fail={("POST", "labels"), ("POST", "issues")})
    verdicts = [{"id": "tests/t.py::test_a", "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40}]
    record = _route(github, ["tests/t.py::test_a"], verdicts)
    assert record["created"] == [] and record["comments"] == [2067]
    assert record["warnings"] == ["create the post-merge-red label: HTTP 502 Bad Gateway",
                                  "create an issue: HTTP 502 Bad Gateway"]
    assert "Tracking issue" not in github.comments[0][1]
    # Unreadable open issues: comments still go out, but no issue that could duplicate one.
    github = FakeGitHub(prs={culprit: [_pr(2067, culprit)]}, fail={("GET", "issues")})
    record = _route(github, ["tests/t.py::test_a"], verdicts)
    assert record["created"] == [] and record["comments"] == [2067] and len(record["warnings"]) == 1


def test_only_the_merge_commit_of_a_merged_pull_request_is_blamed():
    culprit = "c" * 40
    open_pr = dict(_pr(1, culprit), merged_at=None)
    other_base = dict(_pr(2, culprit), base={"ref": "main"})
    github = FakeGitHub(prs={culprit: [open_pr, other_base, _pr(3, "d" * 40)]})
    assert routing.find_merged_pr(github, culprit) is None
    github.prs[culprit].append(_pr(4, culprit))
    assert routing.find_merged_pr(github, culprit)["number"] == 4
    verdicts = [{"id": "tests/t.py::test_a", "verdict": "culprit", "culprit": "e" * 40, "last_good": "b" * 40}]
    record = _route(FakeGitHub(), ["tests/t.py::test_a"], verdicts)  # a direct push: no pull request
    assert record["comments"] == [] and record["created"] == [100]


def test_each_leg_extends_and_closes_only_the_issues_of_its_own_system():
    culprit = "c" * 40
    github = FakeGitHub(label=True, prs={culprit: [_pr(2067, culprit, login="author")]}, issues=[
        # Opened before the Windows leg existed: no system in its block, so a Linux issue.
        _issue(5, {"kind": "culprit", "pr": 2067, "culprit": culprit, "last_good": "b" * 40, "ids": ["tests/t.py::test_a"]}),
        _issue(6, {"os": "windows", "kind": "pre-existing", "ids": ["tests/t.py::test_fixed"]})])
    verdicts = [{"id": "tests/t.py::test_b", "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40}]
    asked = []

    def recheck(ids):
        asked.append(ids)
        return {nodeid: "passed" for nodeid in ids}

    record = _route(github, ["tests/t.py::test_b"], verdicts, recheck=recheck, system="windows")
    # The Windows leg rechecks, closes and opens Windows issues only; the green Linux issue stays as it was.
    assert asked == [["tests/t.py::test_fixed"]] and record["closed"] == [6] and record["os"] == "windows"
    assert record["created"] == [102] and record["updated"] == [] and github.issues[5]["state"] == "open"
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == ["tests/t.py::test_a"]
    issue = github.issues[102]
    assert issue["title"] == "Post-merge red on Windows: #2067 Change 2067"
    assert routing.parse_machine_block(issue["body"])["os"] == "windows" and "fail on Windows since #2067" in issue["body"]
    assert [(number, "fail on Windows since" in text) for number, text in github.comments] == [(6, False), (2067, True)]
    record = _route(github, ["tests/t.py::test_b"], verdicts, recheck=lambda ids: asked.append(ids) or {})
    # The Linux leg extends the issue it opened before the legs and leaves the Windows one alone.
    assert asked[-1] == ["tests/t.py::test_a"] and record["updated"] == [5] and record["created"] == []
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == ["tests/t.py::test_a", "tests/t.py::test_b"]
    assert github.issues[102]["state"] == "open" and "Also failing" not in github.issues[102]["body"]
    assert routing.issue_key({"kind": "culprit", "pr": 1}) == routing.issue_key({"os": "linux", "kind": "culprit", "pr": 1})
    assert routing.issue_key({"os": "windows", "kind": "culprit", "pr": 1}) != routing.issue_key({"kind": "culprit", "pr": 1})


def test_a_size_issue_closes_only_once_its_own_base_measures_green():
    """A later check's base already holds the growth, so not seeing it again proves nothing."""
    pairwise = "size tests/test_smoke.py::test_size_ratchet_transition_against_explicit_base"
    issues = [_issue(5, {"kind": "culprit", "pr": 1, "culprit": "c" * 40, "last_good": "b" * 40, "base": "0" * 40,
                         "ids": [pairwise]}),
              _issue(6, {"kind": "unattributed", "base": "a" * 40, "ids": [SIZED]})]

    def run(status, failed=()):
        asked, github = [], FakeGitHub(label=True, issues=issues)
        record = _route(github, failed, [{"id": nodeid, "verdict": "unattributed"} for nodeid in failed],
                        size_recheck=lambda base, ids: asked.append((base, ids)) or dict.fromkeys(ids, status))
        return record["closed"], asked

    closed, asked = run("passed")
    assert closed == [5, 6] and asked == [("a" * 40, [SIZED]), ("b" * 40, [pairwise])]  # culprit: its last good
    assert run("absent")[0] == [5, 6]
    for status in ("failed", "skipped", "unknown"):
        assert run(status)[0] == [], status
    closed, asked = run("passed", failed=[SIZED])  # red in this run: not rechecked, not green
    assert closed == [5] and asked == [("b" * 40, [pairwise])]
    assert _route(FakeGitHub(label=True, issues=issues), [], [])["closed"] == []  # nothing measured them


@pytest.mark.parametrize("kind", ["pre-existing", "unattributed"])
def test_an_issue_opened_before_the_legs_gains_the_base_of_a_size_id_added_to_it(kind):
    """Such an issue has no `os` and no `base`; the size id joining it must still be measurable."""
    old, base = "tests/t.py::test_old", "a" * 40
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": kind, "ids": [old]})])
    first = _route(github, [old, SIZED], [{"id": SIZED, "verdict": kind}], classification={"base": base, "size": {"base": base}})
    assert first["updated"] == [5] and first["created"] == []
    stored = github.issues[5]["body"]
    assert routing.parse_machine_block(stored) == {"format": 1, "os": "linux", "kind": kind, "base": base, "ids": [SIZED, old]}
    assert f"OURO_SIZE_RATCHET_BASE_REF={base} " in stored and "OURO_SIZE_RATCHET_BASE_REF=None" not in stored
    asked = []
    second = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "passed"),
                    size_recheck=lambda measured, ids: asked.append(measured) or dict.fromkeys(ids, "passed"))
    assert second["closed"] == [5] and asked == [base]  # fixed, measured against its base, closed


def test_a_size_red_is_handed_over_with_the_base_it_was_measured_against():
    culprit, other = "c" * 40, "size tests/test_x.py::test_y"
    github = FakeGitHub(prs={culprit: [_pr(2067, culprit, login="author")]})
    verdicts = [{"id": SIZED, "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40},
                {"id": other, "verdict": "unattributed"}]
    record = _route(github, [SIZED, other], verdicts, classification={"size": {"base": "a" * 40}, "base": "f" * 40})
    assert record["created"] == [100, 101] and record["comments"] == [2067]
    assert routing.parse_machine_block(github.issues[100]["body"]) == {
        "format": 1, "os": "linux", "kind": "culprit", "pr": 2067, "culprit": culprit, "last_good": "b" * 40,
        "base": "a" * 40, "ids": [SIZED]}
    hint = f"`OURO_SIZE_RATCHET_BASE_REF={'b' * 40} python -m pytest -m size_ratchet tests/test_smoke.py`"
    assert hint in github.issues[100]["body"] and hint in github.comments[0][1]
    assert f"`OURO_SIZE_RATCHET_BASE_REF={'a' * 40} python -m pytest -m size_ratchet tests/test_x.py`" in (
        github.issues[101]["body"])
    assert "(a test of the size check, `pytest -m size_ratchet`)" in github.comments[0][1]
    assert "OURO_SIZE_RATCHET_BASE_REF" not in routing.issue_body(
        {"kind": "unattributed", "base": "a" * 40, "ids": ["tests/t.py::test_a"]}, head=HEAD, run_url="u")


def test_a_label_the_other_leg_created_meanwhile_is_no_warning():
    def refused(code):
        def post(path, body):
            raise urllib.error.HTTPError(path, code, "refused", None, None)
        github = FakeGitHub()
        github.post = post
        return github

    routing.ensure_label(refused(422))  # already exists: the other leg won the race
    with pytest.raises(urllib.error.HTTPError):
        routing.ensure_label(refused(403))


def test_the_update_reader_asks_for_the_label_the_hand_over_gives():
    """The update check reads the issues this hand-over labels; renaming one side alone would make
    every install read "none open" while reds stand."""
    from ouroboros import update_qualification

    assert update_qualification.RED_LABEL == routing.LABEL

