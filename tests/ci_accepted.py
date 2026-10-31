"""Known CI failures the repository owner accepted as no product bug (owner decision 1A, 2026-10-21).

tests/accepted_failures.json names test node ids, each with the GitHub issue that
explains it and, optionally, the systems (`linux`, `darwin`, `windows`) it fails on;
without `platforms` an entry holds on every system. A test that failed in its first
run and in its retry (scripts/ci_retry.py) is ACCEPTED, so reported but not blocking,
only while all of this holds: its node id and system match an entry exactly, the
issue is open, and the last time the `not-a-product-bug` label was added to or removed
from the issue, the repository owner added it. Closing the issue or removing the
label makes the failure block again; nothing in code counts, expires or waives.

Everything else fails closed: an unreadable or malformed file, a missing token and
any API or parse error accept nothing. Stdlib only, since scripts/ci_retry.py loads
it under `python -I -S`; the label history is read as scripts/book_growth_mode.py reads it.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import urllib.request

LABEL = "not-a-product-bug"
SYSTEMS = ("linux", "darwin", "windows")


def load(path) -> list[dict]:
    """The entries of an accepted-failures file; ValueError when it is anything else.

    An unknown key is refused rather than ignored: a misspelt `platforms` would
    silently widen an entry to every system.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not (isinstance(data, dict) and data.get("format") == 1 and data.get("label") == LABEL
            and isinstance(data.get("entries"), list)):
        raise ValueError(f"not a format-1 accepted-failures file for the label {LABEL}")
    for entry in data["entries"]:
        platforms = entry.get("platforms", list(SYSTEMS)) if isinstance(entry, dict) else None
        if not (isinstance(entry, dict) and set(entry) <= {"nodeid", "issue", "platforms"}
                and isinstance(entry.get("nodeid"), str) and entry["nodeid"]
                and type(entry.get("issue")) is int and entry["issue"] > 0
                and isinstance(platforms, list) and platforms and set(platforms) <= set(SYSTEMS)):
            raise ValueError(f"invalid accepted-failure entry: {entry!r}")
    return data["entries"]


def _get(url: str, token: str):
    """(one GitHub API response's JSON, the next page's URL or "")."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "Authorization": f"Bearer {token}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        link = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link") or "")
        return json.load(response), link.group(1) if link else ""


def read_issue(repo: str, number: int, token: str) -> dict:
    return _get(f"https://api.github.com/repos/{repo}/issues/{number}", token)[0]


def read_events(repo: str, number: int, token: str) -> list:
    """Every event of the issue, in the API's chronological order, across all pages."""
    events, url = [], f"https://api.github.com/repos/{repo}/issues/{number}/events?per_page=100"
    while url:
        page, url = _get(url, token)
        if not isinstance(page, list):
            raise ValueError("an issue events page is not a list")
        events.extend(page)
    return events


def labelled_by_owner(events: list, owner: str) -> bool:
    """The label's last `labeled`/`unlabeled` event is a `labeled` by the owner."""
    history = [event for event in events if isinstance(event, dict)
               and event.get("event") in ("labeled", "unlabeled")
               and (event.get("label") or {}).get("name") == LABEL]
    return bool(history) and history[-1]["event"] == "labeled" and (
        (history[-1].get("actor") or {}).get("login") == owner)


def accepted(failed_nodes, platform: str, entries: list[dict], repo: str, owner: str, token: str,
             read_issue=read_issue, read_events=read_events) -> list[dict]:
    """The accepted subset of `failed_nodes` on `platform`: `[{"nodeid", "issue"}]`, sorted.

    Issues are read only for failures an entry names. Any error reading any of
    them accepts nothing at all.
    """
    failed = set(failed_nodes)
    matches = sorted((entry for entry in entries
                      if entry["nodeid"] in failed and platform in entry.get("platforms", SYSTEMS)),
                     key=lambda entry: (entry["nodeid"], entry["issue"]))
    if not matches:
        return []
    if not (repo and owner and token):
        print(f"::warning::{len(matches)} failure(s) match accepted entries, but no GitHub repository, "
              "owner and token are available: none is accepted.")
        return []
    standing = {}
    try:
        for number in sorted({entry["issue"] for entry in matches}):
            issue, events = read_issue(repo, number, token), read_events(repo, number, token)
            if not (isinstance(issue, dict) and isinstance(events, list)):
                raise ValueError(f"unexpected API response for issue #{number}")
            standing[number] = issue.get("state") == "open" and labelled_by_owner(events, owner)
    except Exception as error:  # noqa: BLE001 -- unknown issue evidence accepts nothing
        print(f"::warning::accepted failures are not applied: issue evidence is unreadable ({type(error).__name__}).")
        return []
    chosen = {}
    for entry in matches:
        if standing[entry["issue"]]:
            chosen.setdefault(entry["nodeid"], entry["issue"])
    return [{"nodeid": node, "issue": issue} for node, issue in chosen.items()]
