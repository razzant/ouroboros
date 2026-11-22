"""The post-merge check's hand-over (scripts/post_merge_routing.py): comments, issues, closing.

Deterministic: GitHub is a fake in memory and the rechecks are fakes; each leg of the check owns
the `post-merge-red` issues of its own system. A recorded failure of an earlier attempt keeps its
own head, run and attempt: this file fixes that a check never files a historical red under its own
head, never retires a newer failure, and returns what is still owed as records.
"""
from __future__ import annotations

import urllib.error

import pytest

from scripts import post_merge_routing as routing
from tests._post_merge_github import FakeGitHub, tracking_issue as _issue

HEAD = "f" * 40
LINTED = "ruff F811 tests/t.py fake_route"
SIZED = "size tests/test_smoke.py::test_no_oversized_modules"
OLD = "1" * 40  # a head an earlier attempt measured at
REPO = "razzant/ouroboros"


def test_the_machine_block_round_trips_through_an_issue_body():
    block = {"format": 1, "kind": "culprit", "pr": 2067, "culprit": "c" * 40, "last_good": "b" * 40,
             "ids": ["tests/t.py::test_x[a-->b]", "tests/t.py::test_y[<x>]"]}
    body = routing.issue_body(block, head=HEAD, run_url="https://example.invalid/run/1", author="dev")
    assert "-->" not in routing.machine_block(block)[:-3]
    assert routing.parse_machine_block(body) == block
    assert routing.parse_machine_block("no block here") is None
    assert routing.parse_machine_block("<!-- post-merge-red {not json} -->") is None


def _pr(number, sha, login="dev"):
    return {"number": number, "merge_commit_sha": sha, "merged_at": "2026-10-21T00:00:00Z",
            "base": {"ref": "ouroboros"}, "user": {"login": login}, "title": f"Change {number}"}


def _route(github, failed, verdicts, *, recheck=None, flaky=(), node="success", classification=None, **options):
    return routing.route(github, classification={"failed": [{"id": nodeid} for nodeid in failed],
                                                 "flaky": [{"id": nodeid} for nodeid in flaky],
                                                 "selection": {"node": node}, **(classification or {})},
                         attribution={"verdicts": verdicts}, head=HEAD, run_url="https://example.invalid/run/7",
                         recheck=recheck or (lambda ids: pytest.fail(f"nothing to recheck: {ids}")), **options)


def _recorded(ids, *, head=OLD, run=71, attempt=2, base="a" * 40):
    """One `prior_failures` entry: these IDs failed together at `head`, measured by that attempt against `base`."""
    return [{"head": head, "run": run, "attempt": attempt, "base": base, "ids": list(ids)}]


def _repo_github(*args, **kwargs):
    github = FakeGitHub(*args, **kwargs)
    github.repo = REPO  # the real reader names the repository; so do the links a record points at
    return github


def test_a_culprit_gets_one_pr_comment_and_one_new_issue():
    culprit = "c" * 40
    github = FakeGitHub(prs={culprit: [_pr(2066, "d" * 40), _pr(2067, culprit, login="author")]})
    verdicts = [{"id": nodeid, "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40}
                for nodeid in ("tests/t.py::test_a", "tests/t.py::test_b")]
    record = _route(github, ["tests/t.py::test_a", "tests/t.py::test_b"], verdicts)
    assert record["created"] == [100] and record["comments"] == [2067] and record["warnings"] == []
    assert record["untracked"] == []
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
    assert record["untracked"] == []
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


@pytest.mark.parametrize("status,closed", [("passed", True), ("absent", True), ("skipped", False),
                                          ("unknown", False), ("failed", False)])
@pytest.mark.parametrize("kind", ["culprit", "pre-existing", "unattributed"])
def test_a_recheck_closes_only_executed_passes_or_removed_tests(status, closed, kind):
    nodeid = "tests/t.py::test_previous_failure"
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": kind, "pr": 1, "ids": [nodeid]})])
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, status))
    assert record["rechecked"] == {nodeid: status}
    assert record["closed"] == ([5] if closed else [])
    assert github.issues[5]["state"] == ("closed" if closed else "open")
    assert bool(github.comments) is closed


def test_only_a_head_at_or_after_the_recorded_failure_can_retire_or_narrow_it():
    older, red, later = [str(i) * 40 for i in range(1, 4)]
    ids = ["tests/t.py::test_a", "tests/t.py::test_b"]
    github = FakeGitHub(issues=[_issue(5, {"kind": "unattributed", "ids": ids}, head=red)])
    for tip, passed in ((older, ids), (later, ids[:1]), (later, ids[1:])):
        record = routing.route(github, classification={}, attribution={}, head=tip, run_url="run",
                               recheck=lambda asked: {n: "passed" if n in passed else "skipped" for n in asked},
                               ancestor=lambda a, b: a < b)
        if tip == older:
            assert not record["closed"] and not record["narrowed"] and not github.comments
        elif passed == ids[:1]:
            assert record["narrowed"] == [{"issue": 5, "ids": ids[:1]}]
        else:
            assert record["closed"] == [5]


def test_an_issue_loses_each_test_that_passes_and_keeps_what_is_not_green():
    left = ["tests/t.py::test_red", "tests/t.py::test_skip", "tests/t.py::test_unknown", "tests/t.py::test_xfailed"]
    github = FakeGitHub(label=True, issues=[_issue(5, {"os": "windows", "kind": "pre-existing",
                                                       "ids": sorted(["tests/t.py::test_fixed", "tests/t.py::test_gone", *left])})])
    status = {"tests/t.py::test_fixed": "passed", "tests/t.py::test_gone": "absent", "tests/t.py::test_red": "failed",
              "tests/t.py::test_skip": "skipped", "tests/t.py::test_unknown": "unknown", "tests/t.py::test_xfailed": "xfailed"}
    record = _route(github, [], [], recheck=lambda ids: {nodeid: status[nodeid] for nodeid in ids}, system="windows")
    assert record["closed"] == [] and record["narrowed"] == [{"issue": 5, "ids": ["tests/t.py::test_fixed",
                                                                                  "tests/t.py::test_gone"]}]
    body = github.issues[5]["body"]
    assert routing.parse_machine_block(body)["ids"] == left and github.issues[5]["state"] == "open"
    assert f"Still not passing at `{HEAD}`" in body and "takes a test off this list when a check on Windows runs it and it passes, or finds it gone" in body
    [(number, text)] = github.comments  # the record of what left the issue
    assert number == 5 and text.startswith(f"Green at `{HEAD}` on Windows (https://example.invalid/run/7)")
    assert "`tests/t.py::test_fixed`" in text and "`tests/t.py::test_gone`" in text and "Still tracked here: 4." in text
    assert not any(nodeid in text for nodeid in left)
    asked = []  # the next run rechecks only what is left, and closes the issue once that passes
    record = _route(github, [], [], recheck=lambda ids: asked.append(ids) or dict.fromkeys(ids, "passed"), system="windows")
    assert asked == [left] and record["closed"] == [5] and github.issues[5]["state"] == "closed"


@pytest.mark.parametrize("refused", [("POST", "issues/5/comments"), ("PATCH", "issues/5")])
def test_an_issue_is_narrowed_only_after_the_comment_that_names_what_left(refused):
    ids = ["tests/t.py::test_fixed", "tests/t.py::test_red"]
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": "pre-existing", "ids": ids})], fail={refused})
    record = _route(github, [], [], recheck=lambda asked: {ids[0]: "passed", ids[1]: "failed"})
    assert record["narrowed"] == [] and routing.parse_machine_block(github.issues[5]["body"])["ids"] == ids
    assert record["warnings"] == [("comment on #5" if refused[0] == "POST" else "narrow #5") + ": HTTP 502 Bad Gateway"]
    assert len(github.comments) == (refused[0] == "PATCH")  # a later run names them again; nothing leaves unrecorded


def test_a_long_list_of_passing_tests_is_named_over_comments_within_the_limit(monkeypatch):
    monkeypatch.setattr(routing, "_COMMENT_LIMIT", 600)
    ids = [f"tests/t.py::test_case_{index:03d}" for index in range(30)]
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": "pre-existing", "ids": [*ids, "tests/t.py::test_red"]})])
    record = _route(github, [], [], recheck=lambda asked: {nodeid: "failed" if nodeid.endswith("red") else "passed"
                                                          for nodeid in asked})
    texts = [text for _, text in github.comments]
    assert len(texts) > 1 and all(len(text) <= 600 for text in texts)
    assert [line[3:-1] for text in texts for line in text.splitlines() if line.startswith("- `")] == ids
    assert record["narrowed"] == [{"issue": 5, "ids": ids}]
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == ["tests/t.py::test_red"]


def test_narrowing_a_culprit_issue_keeps_the_author_its_opening_names():
    block = {"format": 1, "kind": "culprit", "pr": 2067, "culprit": "c" * 40, "last_good": "b" * 40,
             "ids": ["tests/t.py::test_a", "tests/t.py::test_b"]}
    github = FakeGitHub(label=True, issues=[{"number": 5, "body": routing.issue_body(block, head=HEAD, run_url="u",
                                                                                    author="dev")}])
    _route(github, [], [], recheck=lambda ids: {"tests/t.py::test_a": "passed", "tests/t.py::test_b": "failed"})
    body = github.issues[5]["body"]
    assert "since #2067 by `dev` landed." in body and routing.parse_machine_block(body)["ids"] == ["tests/t.py::test_b"]


def test_the_hand_over_is_recorded_before_the_recheck_which_skips_what_this_run_passed():
    import copy

    culprit = "c" * 40
    github = FakeGitHub(label=True, prs={culprit: [_pr(2067, culprit, login="author")]}, issues=[
        _issue(6, {"kind": "pre-existing", "ids": ["tests/t.py::test_elsewhere", "tests/t.py::test_ran_here"]}),
        _issue(7, {"kind": "culprit", "culprit": "d" * 40, "pr": 2000, "ids": ["tests/quiet.py::test_not_run"]})])
    verdicts = [{"id": "tests/t.py::test_new", "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40}]
    saved, asked = [], []

    def recheck(ids):
        # This run's red already reached its author and an issue before the long session starts.
        [handed] = saved
        assert handed["created"] == [102] and handed["comments"] == [2067] and handed["complete"] is False
        asked.append(ids)
        return dict.fromkeys(ids, "passed")

    record = _route(github, ["tests/t.py::test_new"], verdicts, recheck=recheck,
                    same_run=lambda ids: {"tests/t.py::test_ran_here": "passed", "tests/quiet.py::test_not_run": "kept"},
                    checkpoint=lambda handed: saved.append(copy.deepcopy(handed)))
    # A record whose module this run did not run is not run again for it, and stays as it was.
    assert asked == [["tests/t.py::test_elsewhere"]]
    assert record["same_run"] == {"tests/t.py::test_ran_here": "passed", "tests/quiet.py::test_not_run": "kept"}
    assert record["rechecked"] == {"tests/t.py::test_elsewhere": "passed"}
    assert record["closed"] == [6] and record["complete"] is True and github.issues[7]["state"] == "open"
    assert routing.parse_machine_block(github.issues[7]["body"])["ids"] == ["tests/quiet.py::test_not_run"]
    assert [number for number, _ in github.comments] == [2067, 6]


def test_a_carried_untracked_red_shares_the_tracking_recheck_batch():
    renamed, tracked, skipped = [f"tests/t.py::test_{name}" for name in ("old_name", "tracked", "skipped")]
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": "unattributed", "ids": [tracked]})])
    saved, batches = [], []

    def recheck(ids):
        batches.append(ids)
        assert saved == [[renamed, skipped]]  # The checkpoint precedes any collection.
        return {renamed: "absent", tracked: "passed"}

    record = _route(github, [], [], owed=[renamed, skipped], recheck=recheck,
                    owed_same_run=lambda ids: {skipped: "kept"},
                    checkpoint=lambda row: saved.append(list(row["untracked"])))
    assert batches == [[renamed, tracked]] and record["untracked"] == [skipped]
    assert record["closed"] == [5]


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
    assert record["untracked"] == ["tests/t.py::test_a"]
    assert record["warnings"] == ["create the post-merge-red label: HTTP 502 Bad Gateway",
                                  "create an issue: HTTP 502 Bad Gateway"]
    assert "Tracking issue" not in github.comments[0][1]
    # Unreadable open issues: comments still go out, but no issue that could duplicate one, and no recheck ran.
    github = FakeGitHub(prs={culprit: [_pr(2067, culprit)]}, fail={("GET", "issues")})
    record = _route(github, ["tests/t.py::test_a"], verdicts)
    assert record["created"] == [] and record["comments"] == [2067] and len(record["warnings"]) == 1
    assert record["complete"] is False and record["rechecked"] == {}
    assert record["untracked"] == ["tests/t.py::test_a"]


@pytest.mark.parametrize("failure,untracked", [
    (("PATCH", "issues/5"), ["new"]), (("POST", "issues/5/comments"), []),
])
def test_the_issue_block_decides_whether_a_red_survives_the_window(failure, untracked):
    github = FakeGitHub(label=True, fail=[failure], issues=[_issue(5, {"kind": "unattributed", "ids": ["old"]})])
    record = _route(github, ["old", "new"], [])
    assert record["handed"][0]["issue"] == 5 and record["warnings"]
    assert record["untracked"] == untracked


def test_an_issue_number_without_the_tracking_label_cannot_carry_a_red():
    github = FakeGitHub(fail=[("POST", "labels")])
    record = _route(github, ["new"], [])
    assert record["created"] == [100] and record["untracked"] == ["new"]


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
    # This run's red reaches its author before the recheck of older issues starts.
    assert [(number, "fail on Windows since" in text) for number, text in github.comments] == [(2067, True), (6, False)]
    record = _route(github, ["tests/t.py::test_b"], verdicts, recheck=lambda ids: asked.append(ids) or {})
    # The Linux leg extends the issue it opened before the legs and leaves the Windows one alone.
    assert asked[-1] == ["tests/t.py::test_a"] and record["updated"] == [5] and record["created"] == []
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == ["tests/t.py::test_a", "tests/t.py::test_b"]
    assert github.issues[102]["state"] == "open" and "Also failing" not in github.issues[102]["body"]
    assert routing.issue_key({"kind": "culprit", "pr": 1}) == routing.issue_key({"os": "linux", "kind": "culprit", "pr": 1})
    assert routing.issue_key({"os": "windows", "kind": "culprit", "pr": 1}) != routing.issue_key({"kind": "culprit", "pr": 1})


def test_a_size_issue_closes_only_once_its_own_base_measures_green():
    """A later check's base already holds the growth, so not seeing it again proves nothing; a missing size
    test proves even less — only a pass against the base the issue measures takes a size red off it."""
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
    assert run("absent")[0] == []  # a size red is never found gone: only a pass against its base retires it
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
    assert record["untracked"] == []
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
    # a book's size id says what a later approved growth does to its base; a module's size id does not
    book = routing.issue_body({"kind": "unattributed", "base": "a" * 40, "ids": [f"size {routing.BOOK_BUDGETS}::test_x"]},
                              head=HEAD, run_url="u")
    assert "`book-growth` label approved still counts against that base" in book
    assert "book-growth" not in github.issues[100]["body"] and "only by passing against that base" in github.issues[100]["body"]
    # the closing sentence promises "found gone" only for a test outside the size check
    plain, mixed = (routing.issue_body({"kind": "unattributed", "base": "a" * 40, "ids": ids}, head=HEAD, run_url="u")
                    for ids in (["tests/t.py::test_a"], ["tests/t.py::test_a", SIZED]))
    assert "it passes, or finds it gone, and closes" in plain
    assert "it passes, and closes" in github.issues[100]["body"] and "it passes, and closes" in book
    assert "it passes, or finds one outside the size check gone, and closes" in mixed


@pytest.mark.parametrize("old_base", ["a" * 40, None])
def test_a_size_red_an_issue_of_another_base_lists_gets_an_issue_of_this_base(old_base):
    """An open issue keeps the base it measures its size tests against. The same test red against this window's base
    is a delta that issue cannot recheck: it gets an issue of its own base, which carries it into the next window."""
    old = {"kind": "unattributed", "ids": [SIZED], "base": old_base}
    github = FakeGitHub(label=True, issues=[_issue(5, old)])
    record = _route(github, [SIZED], [], classification={"base": "b" * 40})
    assert record["untracked"] == [] and record["created"] == [101] and record["updated"] == []
    assert routing.parse_machine_block(github.issues[101]["body"]) == {
        "format": 1, "os": "linux", "kind": "unattributed", "base": "b" * 40, "ids": [SIZED]}
    assert routing.parse_machine_block(github.issues[5]["body"]) == {"format": 1, **old}
    # Grouping and the confirmation agree: the next window finds it carried and hands nothing over again.
    again = _route(github, [SIZED], [], classification={"base": "b" * 40})
    assert again["untracked"] == [] and again["created"] == again["updated"] == [] and again["handed"] == []


@pytest.mark.parametrize("kind", ["culprit", "pre-existing", "unattributed"])
def test_a_size_red_that_already_fails_at_the_window_base_stays_with_the_issue_that_lists_it(kind):
    """Red at the window's base itself, a size test is red against any earlier base too: the issue that lists it,
    whatever base it measures, carries it, and no second issue opens."""
    old = {"kind": kind, "ids": [SIZED], **({"pr": 1, "culprit": "c" * 40, "last_good": "a" * 40} if kind == "culprit"
                                           else {"base": "a" * 40})}
    github = FakeGitHub(label=True, issues=[_issue(5, old)])
    record = _route(github, [SIZED], [{"id": SIZED, "verdict": "pre-existing"}],
                    classification={"base": "b" * 40, "size": {"base": "b" * 40}})
    assert record["untracked"] == [] and record["created"] == record["updated"] == [] and record["handed"] == []
    assert routing.parse_machine_block(github.issues[5]["body"]) == {"format": 1, **old}


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


# --- the receipt reader: proof of absence is not an unknown read --------------------------------


class _ListingGitHub(routing.GitHub):
    """The real reader over one crafted artifact listing; nothing here touches the network."""

    def __init__(self, listing):
        super().__init__(REPO, "")
        self.listing = listing

    def get(self, path):
        if "/artifacts?name=" in path:
            return self.listing
        return {}


def _receipt(payload: bytes) -> bytes:
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("handoff.json", payload)
    return buffer.getvalue()


def _archive_download(monkeypatch, archive: bytes) -> None:
    import urllib.request

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit=-1):
            return archive if limit < 0 else archive[:limit]

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: _Response())


def test_a_complete_listing_without_the_receipt_proves_that_run_has_none():
    with pytest.raises(FileNotFoundError):
        _ListingGitHub({"total_count": 0, "artifacts": []}).handoff_json(71, 2, "linux", "0" * 64)


def test_an_expired_receipt_in_a_complete_listing_is_gone_for_good():
    listing = {"total_count": 1,
               "artifacts": [{"id": 5, "name": "post-merge-handoff-linux-71-2", "expired": True}]}
    with pytest.raises(FileNotFoundError):
        _ListingGitHub(listing).handoff_json(71, 2, "linux", "0" * 64)


@pytest.mark.parametrize("listing", [
    {}, {"artifacts": []}, {"total_count": 1, "artifacts": "no"},
    {"total_count": 2, "artifacts": [{"id": 5, "name": "n", "expired": False}]},
    {"total_count": 1, "artifacts": [{"id": 5, "name": "another", "expired": False}]},
    {"total_count": 2, "artifacts": [{"id": 5, "name": "post-merge-handoff-linux-71-2", "expired": False},
                                     {"id": 6, "name": "post-merge-handoff-linux-71-2", "expired": False}]},
    {"total_count": 1, "artifacts": [{"id": 5, "name": "post-merge-handoff-linux-71-2"}]},
    {"total_count": 1, "artifacts": [{"name": "post-merge-handoff-linux-71-2", "expired": False}]},
])
def test_an_unreadable_or_ambiguous_listing_leaves_the_receipt_unknown(listing):
    with pytest.raises(ValueError):
        _ListingGitHub(listing).handoff_json(71, 2, "linux", "0" * 64)


def test_a_failed_listing_is_unknown_not_absent():
    class _Refused(_ListingGitHub):
        def get(self, path):
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)

    with pytest.raises(urllib.error.HTTPError):
        _Refused(None).handoff_json(71, 2, "linux", "0" * 64)


def test_a_well_formed_receipt_reads_back_through_its_archive(monkeypatch):
    import hashlib
    payload = b'{"untracked": {"linux": ["tests/t.py::test_x"]}}'
    listing = {"total_count": 1,
               "artifacts": [{"id": 5, "name": "post-merge-handoff-linux-71-2", "expired": False}]}
    _archive_download(monkeypatch, _receipt(payload))
    assert _ListingGitHub(listing).handoff_json(71, 2, "linux", hashlib.sha256(payload).hexdigest()) == {
        "untracked": {"linux": ["tests/t.py::test_x"]}}
    with pytest.raises(ValueError):  # bytes that do not match the digest: unknown, never absent
        _ListingGitHub(listing).handoff_json(71, 2, "linux", "0" * 64)


# --- recorded failures of earlier attempts: filed under their own source, never this run's head --


def test_a_recorded_failure_is_filed_under_its_own_head_and_attempt_link():
    github = _repo_github(label=True)
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                    ancestor=lambda a, b: a < b, prior_failures=_recorded(["tests/t.py::test_x", "tests/t.py::test_y"]),
                    origin={"head": HEAD, "run": 90, "attempt": 1, "base": "b" * 40})
    assert record["created"] == [100] and record["updated"] == []
    assert record["handed"] == [{"ids": ["tests/t.py::test_x", "tests/t.py::test_y"], "pr": None, "issue": 100}]
    assert record["untracked"] == [] and record["awaiting_handoff"] == []
    body = github.issues[100]["body"]
    assert f"Found failing at `{OLD}`: https://github.com/{REPO}/actions/runs/71/attempts/2" in body
    assert f"Found failing at `{HEAD}`" not in body and "https://example.invalid/run/7" not in body
    assert routing.parse_machine_block(body) == {"format": 1, "os": "linux", "kind": "unattributed",
                                                 "base": "a" * 40,
                                                 "ids": ["tests/t.py::test_x", "tests/t.py::test_y"]}
    # The next run finds the red carried by the open issue and files nothing again.
    again = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                   ancestor=lambda a, b: a < b,
                   prior_failures=_recorded(["tests/t.py::test_x", "tests/t.py::test_y"]))
    assert again["created"] == again["updated"] == [] and again["handed"] == []
    assert again["untracked"] == [] and again["awaiting_handoff"] == []


def test_a_recorded_failure_without_numbers_links_its_own_commit():
    github = _repo_github(label=True)
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                    ancestor=lambda a, b: a < b,
                    prior_failures=[{"head": OLD, "base": "a" * 40, "ids": ["tests/t.py::test_x"]}])
    assert record["created"] == [100]
    assert f"Found failing at `{OLD}`: https://github.com/{REPO}/commit/{OLD}" in github.issues[100]["body"]


def test_a_refused_write_keeps_the_recorded_source_and_attempt_link():
    github = _repo_github(label=True, fail={("POST", "issues")})
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "skipped"),
                    ancestor=lambda a, b: a < b, prior_failures=_recorded(["tests/t.py::test_x"]))
    assert record["created"] == [] and "create an issue: HTTP 502 Bad Gateway" in record["warnings"]
    assert record["untracked"] == ["tests/t.py::test_x"]
    assert record["awaiting_handoff"] == [{"head": OLD, "run": 71, "attempt": 2, "base": "a" * 40,
                                           "ids": ["tests/t.py::test_x"]}]
    assert record["rechecked"] == {"tests/t.py::test_x": "skipped"}  # a skip settles nothing either


def test_a_whole_pass_of_this_run_retires_a_recorded_red_without_filing_it():
    asked = []
    record = _route(_repo_github(label=True), [], [],
                    recheck=lambda ids: asked.append(ids) or pytest.fail("nothing to recheck"),
                    owed_same_run=lambda ids: {"tests/t.py::test_x": "passed"},
                    ancestor=lambda a, b: a < b, prior_failures=_recorded(["tests/t.py::test_x"]))
    assert record["created"] == [] and record["handed"] == [] and asked == []
    assert record["untracked"] == [] and record["awaiting_handoff"] == []


def test_a_late_rerun_cannot_retire_a_newer_recorded_failure():
    """The K/A/B control: an issue filed at K still lists the red, a newer run recorded it failing at B,
    and this rerun measures an older head between K and B. Its pass or absence retires nothing, and
    the rerun spends no recheck on a failure it cannot reach."""
    k, a, b = "1" * 40, "2" * 40, "3" * 40
    x = "tests/t.py::test_x"
    github = FakeGitHub(issues=[_issue(5, {"kind": "unattributed", "ids": [x]}, head=k)])
    record = routing.route(github, classification={}, attribution={}, head=a, run_url="run",
                           recheck=lambda asked: dict.fromkeys(asked, "passed"),
                           ancestor=lambda older, newer: older < newer, owed=[x],
                           prior_failures=_recorded([x], head=b))
    assert record["closed"] == [] and record["narrowed"] == [] and not github.comments
    assert record["rechecked"] == {}
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == [x]
    assert github.issues[5]["state"] == "open"
    assert record["untracked"] == [x]
    assert record["awaiting_handoff"] == [{"head": b, "run": 71, "attempt": 2, "base": "a" * 40, "ids": [x]}]
    # The same observation reaching the recorded head does retire it: the pass is on/after the failure.
    github = FakeGitHub(issues=[_issue(5, {"kind": "unattributed", "ids": [x]}, head=k)])
    record = routing.route(github, classification={}, attribution={}, head=b, run_url="run",
                           recheck=lambda asked: dict.fromkeys(asked, "passed"),
                           ancestor=lambda older, newer: older < newer, owed=[x],
                           prior_failures=_recorded([x], head=b))
    assert record["closed"] == [5] and record["untracked"] == [] and record["awaiting_handoff"] == []


def test_a_record_beyond_the_head_is_settled_by_nothing_here():
    b, x = "3" * 40, "tests/t.py::test_x"
    asked = []
    record = routing.route(_repo_github(label=True), classification={}, attribution={}, head="2" * 40,
                           run_url="run", recheck=lambda ids: asked.append(ids) or dict.fromkeys(ids, "passed"),
                           ancestor=lambda a, other: a < other,
                           owed_same_run=lambda ids: {x: "passed"},  # even this run's whole passes settle
                           prior_failures=_recorded([x], head=b))    # nothing about a newer failure
    assert asked == [] and record["created"] == [] and record["rechecked"] == {}
    assert record["untracked"] == [x]
    assert record["awaiting_handoff"] == [{"head": b, "run": 71, "attempt": 2, "base": "a" * 40, "ids": [x]}]


def test_a_new_failure_and_a_recorded_one_are_not_conflated():
    culprit = "c" * 40
    github = _repo_github(label=True, prs={culprit: [_pr(2067, culprit, login="author")]})
    verdicts = [{"id": "tests/t.py::test_new_red", "verdict": "culprit", "culprit": culprit, "last_good": "b" * 40}]
    record = _route(github, ["tests/t.py::test_new_red"], verdicts,
                    recheck=lambda ids: dict.fromkeys(ids, "failed"), ancestor=lambda a, b: a < b,
                    prior_failures=_recorded(["tests/t.py::test_old_red"]), origin={"head": HEAD, "run": 90})
    assert record["created"] == [100, 101] and record["comments"] == [2067]
    assert record["handed"] == [{"ids": ["tests/t.py::test_new_red"], "pr": 2067, "issue": 100},
                                {"ids": ["tests/t.py::test_old_red"], "pr": None, "issue": 101}]
    assert record["untracked"] == [] and record["awaiting_handoff"] == []
    fresh, recorded = github.issues[100]["body"], github.issues[101]["body"]
    assert f"Found failing at `{HEAD}`: https://example.invalid/run/7" in fresh and OLD not in fresh
    assert f"Found failing at `{OLD}`: https://github.com/{REPO}/actions/runs/71/attempts/2" in recorded
    assert HEAD not in recorded and "https://example.invalid/run/7" not in recorded
    assert routing.parse_machine_block(recorded)["kind"] == "unattributed"


def test_a_recorded_size_red_tracks_only_against_the_base_its_record_names():
    window_base, record_base = "b" * 40, "a" * 40
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": "unattributed", "base": window_base, "ids": [SIZED]})])
    asked = []
    record = _route(github, [], [],
                    size_recheck=lambda base, ids: asked.append((base, ids)) or dict.fromkeys(ids, "failed"),
                    ancestor=lambda a, b: a < b,
                    classification={"size_checked": True, "size": {"base": window_base}, "base": window_base,
                                    "size_passed": [SIZED]},
                    prior_failures=_recorded([SIZED], base=record_base))
    assert record["created"] == [101] and record["updated"] == []
    assert (record_base, [SIZED]) in asked  # re-measured against the base its record names
    assert record["untracked"] == [] and record["awaiting_handoff"] == []
    assert routing.parse_machine_block(github.issues[101]["body"]) == {"format": 1, "os": "linux",
                                                                       "kind": "unattributed", "base": record_base,
                                                                       "ids": [SIZED]}
    # An issue measuring the record's own base carries the recorded red, and a window-base pass
    # against another base discharges nothing: the red settles without any new issue.
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": "unattributed", "base": record_base, "ids": [SIZED]})])
    record = _route(github, [], [], size_recheck=lambda base, ids: {}, ancestor=lambda a, b: a < b,
                    classification={"size_checked": True, "size": {"base": window_base}, "base": window_base,
                                    "size_passed": [SIZED]},
                    prior_failures=_recorded([SIZED], base=record_base))
    assert record["created"] == [] and record["untracked"] == [] and record["awaiting_handoff"] == []


def test_records_of_one_red_file_from_the_latest_and_all_survive_a_refusal():
    b1, b2 = "2" * 40, "3" * 40
    x = "tests/t.py::test_x"
    records = [{"head": b1, "run": 60, "attempt": 1, "base": "a" * 40, "ids": [x]},
               {"head": b2, "run": 71, "attempt": 2, "base": "a" * 40, "ids": [x]}]
    github = _repo_github(label=True)
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                    ancestor=lambda a, b: a < b, prior_failures=records)
    assert record["created"] == [100]  # the newest record is the one a tracker names
    assert f"Found failing at `{b2}`: https://github.com/{REPO}/actions/runs/71/attempts/2" in github.issues[100]["body"]
    assert record["awaiting_handoff"] == [] and record["untracked"] == []
    github = _repo_github(label=True, fail={("POST", "issues")})
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "skipped"),
                    ancestor=lambda a, b: a < b, prior_failures=records)
    assert record["created"] == []  # both sources and attempt links survive the refused write
    assert record["awaiting_handoff"] == [{"head": b1, "run": 60, "attempt": 1, "base": "a" * 40, "ids": [x]},
                                          {"head": b2, "run": 71, "attempt": 2, "base": "a" * 40, "ids": [x]}]


def test_an_issue_holding_a_newer_observation_is_not_extended_with_an_older_one():
    newer, x = "5" * 40, "tests/t.py::test_x"
    github = FakeGitHub(label=True, issues=[_issue(5, {"kind": "unattributed", "ids": ["tests/t.py::test_kept"],
                                                       "base": "a" * 40}, head=newer)])
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                    ancestor=lambda a, b: a < b, prior_failures=_recorded([x]))
    assert record["updated"] == [] and record["created"] == []
    assert "issue #5 has no observation at or before this head; its record was kept" in record["warnings"]
    assert record["untracked"] == [x]
    assert record["awaiting_handoff"] == [{"head": OLD, "run": 71, "attempt": 2, "base": "a" * 40, "ids": [x]}]
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == ["tests/t.py::test_kept"]


def test_a_recorded_failure_extends_a_matching_issue_from_its_own_head():
    x, y = "tests/t.py::test_x", "tests/t.py::test_y"
    github = _repo_github(label=True, issues=[_issue(5, {"kind": "unattributed", "ids": [y]}, head=OLD)])
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                    ancestor=lambda a, b: a < b, prior_failures=_recorded([x]))
    assert record["created"] == [] and record["updated"] == [5]
    assert [number for number, _ in github.comments] == [5]
    assert github.comments[0][1].startswith(f"Also failing at `{OLD}`")
    assert f"https://github.com/{REPO}/actions/runs/71/attempts/2" in github.comments[0][1]
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == [x, y]
    assert record["untracked"] == [] and record["awaiting_handoff"] == []


def test_a_recorded_red_that_passes_the_recheck_is_tracked_then_closed_the_same_run():
    """The hand-over precedes the recheck, so the red gets its tracker first; the recheck then retires
    it with evidence — the issue is born and closed by one run, and nothing stays owed."""
    x = "tests/t.py::test_x"
    github = _repo_github(label=True)
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "passed"),
                    ancestor=lambda a, b: a < b, prior_failures=_recorded([x]))
    assert record["created"] == [100] and record["closed"] == [100]
    assert f"Found failing at `{OLD}`" in github.issues[100]["body"]
    assert record["untracked"] == [] and record["awaiting_handoff"] == []


def test_the_recorded_hand_over_reaches_the_checkpoint_before_the_recheck():
    import copy

    x = "tests/t.py::test_x"
    github = _repo_github(label=True)
    saved = []

    def recheck(ids):
        [handed] = saved
        assert handed["created"] == [100] and handed["awaiting_handoff"] == []
        return dict.fromkeys(ids, "failed")

    record = _route(github, [], [], recheck=recheck, ancestor=lambda a, b: a < b,
                    prior_failures=_recorded([x]), checkpoint=lambda handed: saved.append(copy.deepcopy(handed)))
    assert record["created"] == [100] and record["awaiting_handoff"] == []


def test_a_flaky_pass_does_not_drop_a_recorded_red():
    """A retry that passes is no whole pass, no issue and no disappearance: the recorded red keeps its
    obligation, which the re-filing turns into a tracker instead of letting it vanish."""
    x = "tests/t.py::test_x"
    github = _repo_github(label=True)
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                    flaky=[x], classification={"flaky": [{"id": x, "retry": "passed"}]},
                    ancestor=lambda a, b: a < b, prior_failures=_recorded([x]))
    assert record["created"] == [100] and record["untracked"] == [] and record["awaiting_handoff"] == []
    # An ID owed without a record keeps the meaning it had: a flaky pass still settles it.
    legacy = "tests/t.py::test_legacy"
    record = _route(_repo_github(label=True), [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                    flaky=[legacy], classification={"flaky": [{"id": legacy, "retry": "passed"}]},
                    owed=[legacy], ancestor=lambda a, b: a < b)
    assert record["created"] == [] and record["untracked"] == [] and record["awaiting_handoff"] == []


def test_unreadable_open_issues_leave_the_records_owed():
    x = "tests/t.py::test_x"
    github = _repo_github(fail={("GET", "issues")})
    record = _route(github, [], [], recheck=lambda ids: dict.fromkeys(ids, "failed"),
                    ancestor=lambda a, b: a < b, prior_failures=_recorded([x]))
    assert record["created"] == [] and record["handed"] == []
    assert record["complete"] is False
    assert record["untracked"] == [x]
    assert record["awaiting_handoff"] == [{"head": OLD, "run": 71, "attempt": 2, "base": "a" * 40, "ids": [x]}]


def test_awaiting_records_keep_each_source_and_invent_none():
    legacy, x, z = "tests/t.py::test_legacy", "tests/t.py::test_x", "tests/t.py::test_new_red"
    github = _repo_github(label=True, fail={("POST", "issues")})
    record = _route(github, [z], [], owed=[legacy], recheck=lambda ids: dict.fromkeys(ids, "skipped"),
                    ancestor=lambda a, b: a < b, prior_failures=_recorded([x]),
                    origin={"head": HEAD, "run": 90, "attempt": 1, "base": "b" * 40})
    assert record["created"] == []
    assert record["awaiting_handoff"] == [
        {"head": OLD, "run": 71, "attempt": 2, "base": "a" * 40, "ids": [x]},
        {"head": HEAD, "run": 90, "attempt": 1, "base": "b" * 40, "ids": [z]},
        {"head": None, "run": None, "attempt": None, "base": None, "ids": [legacy]},
    ]
    # A call without an origin still dates this run's own red honestly: its head, nothing invented.
    record = _route(_repo_github(label=True, fail={("POST", "issues")}), [z], [],
                    recheck=lambda ids: dict.fromkeys(ids, "skipped"))
    assert record["awaiting_handoff"] == [{"head": HEAD, "run": None, "attempt": None, "base": None, "ids": [z]}]


def test_one_size_red_recorded_against_two_bases_is_owed_once_per_base():
    """Two attempts measured the same size test against different bases: each record is its own obligation.
    An issue measuring one base tracks that record only, so the other is filed against its own base under
    its own attempt link; a refused write leaves exactly that one owed, and a pass against one base
    settles one."""
    listed, other = "a" * 40, "b" * 40
    records = _recorded([SIZED], run=60, attempt=1, base=other) + _recorded([SIZED], base=listed)
    tracker = _issue(5, {"kind": "unattributed", "base": listed, "ids": [SIZED]})
    github = _repo_github(label=True, issues=[tracker])
    record = _route(github, [], [], ancestor=lambda a, b: a < b, prior_failures=records)
    assert record["created"] == [101] and record["updated"] == []
    assert record["handed"] == [{"ids": [SIZED], "pr": None, "issue": 101}]
    filed = github.issues[101]["body"]
    assert routing.parse_machine_block(filed) == {
        "format": 1, "os": "linux", "kind": "unattributed", "base": other, "ids": [SIZED]}
    assert f"Found failing at `{OLD}`: https://github.com/{REPO}/actions/runs/60/attempts/1" in filed
    assert routing.parse_machine_block(github.issues[5]["body"])["base"] == listed
    assert record["untracked"] == [] and record["awaiting_handoff"] == []
    # A refused write releases only the record an issue of its own base carries.
    github = _repo_github(label=True, issues=[tracker], fail={("POST", "issues")})
    record = _route(github, [], [], ancestor=lambda a, b: a < b, prior_failures=records)
    assert record["created"] == [] and record["untracked"] == [SIZED]
    assert record["awaiting_handoff"] == [{"head": OLD, "run": 60, "attempt": 1, "base": other, "ids": [SIZED]}]
    # This run's own size gate passed against one of the two bases: that settles one record, never the other.
    github = _repo_github(label=True)
    record = _route(github, [], [], ancestor=lambda a, b: a < b, prior_failures=records,
                    classification={"size_checked": True, "size": {"base": listed}, "size_passed": [SIZED]})
    assert record["created"] == [100] and record["untracked"] == [] and record["awaiting_handoff"] == []
    assert routing.parse_machine_block(github.issues[100]["body"])["base"] == other


@pytest.mark.parametrize("fails_here", [False, True])
def test_a_newer_recorded_failure_survives_a_batch_that_files_another(fails_here):
    """An issue filed at K lists the red, a newer run recorded it failing at B, and this rerun at A, between
    them, re-files another recorded red in the same batch. Only what was re-filed can have gained a tracker:
    the record at B stays owed whatever the older issue lists, also when this rerun saw that very test fail
    itself, so a later run still knows the newer failure."""
    k, a, b = "1" * 40, "2" * 40, "3" * 40
    x, y = "tests/t.py::test_x", "tests/t.py::test_y"
    github = _repo_github(label=True, issues=[_issue(5, {"kind": "unattributed", "ids": [x]}, head=k)])
    record = routing.route(
        github, classification={"failed": [{"id": x}] if fails_here else []}, attribution={}, head=a,
        run_url="run", recheck=lambda asked: dict.fromkeys(asked, "failed"),
        ancestor=lambda older, newer: older < newer,
        prior_failures=_recorded([x], head=b) + _recorded([y], head=k, run=60, attempt=1))
    assert record["updated"] == [5] and record["created"] == []  # the batch files the other recorded red
    assert routing.parse_machine_block(github.issues[5]["body"])["ids"] == [x, y]
    assert record["closed"] == [] and record["narrowed"] == []
    assert record["untracked"] == [x]
    assert record["awaiting_handoff"] == [{"head": b, "run": 71, "attempt": 2, "base": "a" * 40, "ids": [x]}]


def test_a_recorded_node_lane_red_is_retired_by_a_whole_pass_never_by_a_passed_retry():
    """The Node lane failed, then passed its retry: that is no whole pass, so an earlier attempt's record of
    the lane gets its tracker and keeps it, and an issue that already tracks it stays open. A lane that
    passed outright retires the record; an ID owed without a record keeps the meaning it had."""
    lane, retried = routing.NODE_LANE, {"flaky": [{"id": routing.NODE_LANE, "retry": "passed"}]}
    github = _repo_github(label=True)
    record = _route(github, [], [], node="failure", classification=retried, ancestor=lambda a, b: a < b,
                    prior_failures=_recorded([lane]))
    assert record["created"] == [100] and record["closed"] == [] and github.issues[100]["state"] == "open"
    assert routing.parse_machine_block(github.issues[100]["body"])["ids"] == [lane]
    assert record["untracked"] == [] and record["awaiting_handoff"] == []
    github = _repo_github(label=True, issues=[_issue(5, {"kind": "unattributed", "ids": [lane]}, head=OLD)])
    record = _route(github, [], [], node="failure", classification=retried, ancestor=lambda a, b: a < b,
                    prior_failures=_recorded([lane]))
    assert record["created"] == [] and record["closed"] == [] and github.issues[5]["state"] == "open"
    assert record["untracked"] == [] and record["awaiting_handoff"] == []
    # A whole pass of the lane retires the record without filing it.
    record = _route(_repo_github(label=True), [], [], ancestor=lambda a, b: a < b,
                    prior_failures=_recorded([lane]))
    assert record["created"] == [] and record["untracked"] == [] and record["awaiting_handoff"] == []
    # Owed without a record, the lane keeps the meaning it had: the passed retry settles it.
    record = _route(_repo_github(label=True), [], [], node="failure", classification=retried, owed=[lane])
    assert record["created"] == [] and record["untracked"] == [] and record["awaiting_handoff"] == []


def test_current_and_prior_observations_do_not_retry_an_unconfirmed_issue_write():
    x = "tests/t.py::test_x"
    github = _repo_github(label=True, fail={("POST", "issues")})
    prior = _recorded([x])
    record = _route(github, [x], [], ancestor=lambda a, b: a < b, prior_failures=prior,
                    origin={"head": HEAD, "run": 90, "attempt": 1, "base": "c" * 40})
    assert record["warnings"].count("create an issue: HTTP 502 Bad Gateway") == 1
    assert record["untracked"] == [x]
    assert record["awaiting_handoff"] == prior + [
        {"head": HEAD, "run": 90, "attempt": 1, "base": "c" * 40, "ids": [x]}]
