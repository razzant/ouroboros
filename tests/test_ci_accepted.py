"""tests/ci_accepted.py: a failure is accepted only while its open issue carries the owner's label."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from tests import ci_accepted

REPO = Path(__file__).resolve().parents[1]
NODE, OTHER = "tests/test_x.py::test_flaky_platform", "tests/test_y.py::test_other"
REPO_NAME, OWNER, TOKEN = "owner/repo", "owner", "test-token"


def _labelled(*steps):
    """Issue events: (event, actor) pairs for the label, in order."""
    return [{"event": event, "label": {"name": ci_accepted.LABEL}, "actor": {"login": actor}} for event, actor in steps]


class _Issues:
    """Stands in for the GitHub API; records every read."""

    def __init__(self, states=None, events=None, fail=None):
        self.states, self.events, self.fail, self.reads = states or {}, events or {}, fail, []

    def issue(self, repo, number, token):
        self.reads.append(("issue", repo, number, token))
        if self.fail == "issue":
            raise OSError("network down")
        return self.states.get(number, {"state": "open"})

    def events_of(self, repo, number, token):
        self.reads.append(("events", repo, number, token))
        if self.fail == "events":
            raise ValueError("unparsable page")
        return self.events.get(number, _labelled(("labeled", OWNER)))


def _accept(entries, issues, *, failed=(NODE,), platform="windows", repo=REPO_NAME, owner=OWNER, token=TOKEN):
    return ci_accepted.accepted(list(failed), platform, entries, repo, owner, token,
                                read_issue=issues.issue, read_events=issues.events_of)


def _file(tmp_path, document):
    path = tmp_path / "accepted_failures.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_committed_file_is_a_valid_empty_list():
    assert ci_accepted.load(REPO / "tests" / "accepted_failures.json") == []
    document = json.loads((REPO / "tests" / "accepted_failures.json").read_text(encoding="utf-8"))
    assert document == {"format": 1, "label": "not-a-product-bug", "entries": []}


@pytest.mark.parametrize("document", [
    [], {"format": 2, "label": ci_accepted.LABEL, "entries": []},
    {"format": 1, "label": "wontfix", "entries": []}, {"format": 1, "label": ci_accepted.LABEL},
    {"format": 1, "label": ci_accepted.LABEL, "entries": [{"nodeid": NODE}]},
    {"format": 1, "label": ci_accepted.LABEL, "entries": [{"nodeid": NODE, "issue": 0}]},
    {"format": 1, "label": ci_accepted.LABEL, "entries": [{"nodeid": NODE, "issue": "2083"}]},
    {"format": 1, "label": ci_accepted.LABEL, "entries": [{"nodeid": "", "issue": 1}]},
    {"format": 1, "label": ci_accepted.LABEL, "entries": [{"nodeid": NODE, "issue": 1, "platforms": []}]},
    {"format": 1, "label": ci_accepted.LABEL, "entries": [{"nodeid": NODE, "issue": 1, "platforms": ["win32"]}]},
    # A misspelt key would otherwise widen the entry to every system.
    {"format": 1, "label": ci_accepted.LABEL, "entries": [{"nodeid": NODE, "issue": 1, "platform": ["windows"]}]},
], ids=["not-object", "format", "label", "no-entries", "no-issue", "issue-zero", "issue-text", "empty-node",
        "no-platform", "unknown-platform", "unknown-key"])
def test_a_malformed_file_is_refused_whole(tmp_path, document):
    with pytest.raises(ValueError):
        ci_accepted.load(_file(tmp_path, document))


def test_a_well_formed_file_loads_its_entries(tmp_path):
    entries = [{"nodeid": NODE, "issue": 2083, "platforms": ["windows"]}, {"nodeid": OTHER, "issue": 7}]
    assert ci_accepted.load(_file(tmp_path, {"format": 1, "label": ci_accepted.LABEL, "entries": entries})) == entries


@pytest.mark.parametrize("steps, accepted", [
    ([("labeled", OWNER)], True),
    ([("labeled", "contributor")], False),
    ([("labeled", OWNER), ("unlabeled", OWNER)], False),
    ([("labeled", OWNER), ("unlabeled", "contributor")], False),
    ([("unlabeled", OWNER), ("labeled", OWNER)], True),
    ([("labeled", OWNER), ("labeled", "contributor")], False),
    ([], False),
], ids=["owner", "not-owner", "removed", "removed-by-other", "relabelled", "relabelled-by-other", "never"])
def test_only_the_owner_applying_the_label_last_accepts(steps, accepted):
    issues = _Issues(events={2083: _labelled(*steps) + [
        {"event": "labeled", "label": {"name": "bug"}, "actor": {"login": OWNER}},  # Another label never counts.
        {"event": "closed", "actor": {"login": OWNER}}]})
    assert _accept([{"nodeid": NODE, "issue": 2083}], issues) == ([{"nodeid": NODE, "issue": 2083}] if accepted else [])


def test_a_closed_issue_accepts_nothing():
    issues = _Issues(states={2083: {"state": "closed"}})
    assert _accept([{"nodeid": NODE, "issue": 2083}], issues) == []
    assert ("issue", REPO_NAME, 2083, TOKEN) in issues.reads


@pytest.mark.parametrize("platform, accepted", [("windows", True), ("linux", False), ("darwin", False)])
def test_an_entry_holds_only_on_its_systems(platform, accepted):
    entries = [{"nodeid": NODE, "issue": 2083, "platforms": ["windows"]}, {"nodeid": OTHER, "issue": 7}]
    expected = ([{"nodeid": NODE, "issue": 2083}] if accepted else []) + [{"nodeid": OTHER, "issue": 7}]
    assert _accept(entries, _Issues(), failed=(NODE, OTHER), platform=platform) == expected


def test_only_failures_an_entry_names_are_read_and_accepted():
    issues = _Issues()
    entries = [{"nodeid": NODE, "issue": 2083}, {"nodeid": OTHER, "issue": 7}]
    assert _accept(entries, issues, failed=(NODE, "tests/test_z.py::test_unlisted")) == [{"nodeid": NODE, "issue": 2083}]
    assert {read[2] for read in issues.reads} == {2083}
    issues.reads.clear()
    assert _accept(entries, issues, failed=("tests/test_z.py::test_unlisted",)) == [] and issues.reads == []


def test_one_node_listed_twice_is_accepted_once_by_its_first_standing_issue():
    issues = _Issues(states={5: {"state": "closed"}})
    entries = [{"nodeid": NODE, "issue": 9}, {"nodeid": NODE, "issue": 5}, {"nodeid": NODE, "issue": 7}]
    assert _accept(entries, issues) == [{"nodeid": NODE, "issue": 7}]


@pytest.mark.parametrize("missing", ["repo", "owner", "token"])
def test_without_the_repository_identity_and_token_nothing_is_accepted_or_read(missing, capsys):
    issues = _Issues()
    assert _accept([{"nodeid": NODE, "issue": 2083}], issues, **{missing: ""}) == []
    assert issues.reads == [] and "none is accepted" in capsys.readouterr().out


@pytest.mark.parametrize("fail", ["issue", "events"])
def test_any_unreadable_issue_evidence_accepts_nothing_at_all(fail, capsys):
    """Fail closed: one unreadable issue withdraws every acceptance of the run, the readable ones too."""
    issues = _Issues(fail=fail)
    entries = [{"nodeid": NODE, "issue": 2083}, {"nodeid": OTHER, "issue": 7}]
    assert _accept(entries, issues, failed=(NODE, OTHER)) == []
    assert "not applied" in capsys.readouterr().out


@pytest.mark.parametrize("issue, events", [(["not", "an", "object"], []), ({"state": "open"}, {"not": "a list"})])
def test_an_unexpected_api_shape_accepts_nothing(issue, events):
    assert ci_accepted.accepted([NODE], "linux", [{"nodeid": NODE, "issue": 1}], REPO_NAME, OWNER, TOKEN,
                                read_issue=lambda *args: issue, read_events=lambda *args: events) == []


def test_the_issue_events_are_read_completely_with_the_token(monkeypatch):
    """Every page of the events list, in order, as scripts/book_growth_mode.py reads label history."""
    pages = {
        f"https://api.github.com/repos/{REPO_NAME}/issues/7/events?per_page=100": (
            _labelled(("labeled", OWNER)), '<https://api.github.com/page2>; rel="next"'),
        "https://api.github.com/page2": (_labelled(("unlabeled", OWNER)), ""),
        f"https://api.github.com/repos/{REPO_NAME}/issues/7": ({"state": "open"}, ""),
    }
    seen = []

    class Response(io.BytesIO):
        def __init__(self, url):
            body, link = pages[url]
            super().__init__(json.dumps(body).encode("utf-8"))
            self.headers = {"Link": link}

    def urlopen(request, timeout):
        seen.append((request.full_url, request.get_header("Authorization"), timeout))
        return Response(request.full_url)

    monkeypatch.setattr(ci_accepted.urllib.request, "urlopen", urlopen)
    events = ci_accepted.read_events(REPO_NAME, 7, TOKEN)
    assert [event["event"] for event in events] == ["labeled", "unlabeled"]
    assert not ci_accepted.labelled_by_owner(events, OWNER), "the later page's removal is the last word"
    assert ci_accepted.read_issue(REPO_NAME, 7, TOKEN) == {"state": "open"}
    assert [url for url, _, _ in seen] == list(pages)
    assert {(auth, timeout) for _, auth, timeout in seen} == {(f"Bearer {TOKEN}", 30)}
    pages[f"https://api.github.com/repos/{REPO_NAME}/issues/7/events?per_page=100"] = ({"message": "Not Found"}, "")
    with pytest.raises(ValueError):
        ci_accepted.read_events(REPO_NAME, 7, TOKEN)
