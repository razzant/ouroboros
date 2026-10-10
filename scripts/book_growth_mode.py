#!/usr/bin/env python3
"""Decide whom the reference-book growth rule binds in this CI run.

Prints ``reference-book growth mode: <mode>`` and appends ``mode=<mode>`` to ``$GITHUB_OUTPUT``;
the size lane reads it as ``OURO_BOOK_GROWTH_MODE`` (tests/test_reference_book_budgets.py).

The official repository blocks a grown book on owner/member/collaborator pull requests and
pushes to ``ouroboros``. Outside contributor pull requests and fork CI only warn. The
``book-growth`` label approves a pull request when the repository owner applied it last. A
push inherits that exception only from a merged PR into this repository's ``ouroboros``
branch whose landed ``merge_commit_sha`` is the pushed SHA. Associated open or unrelated PRs
approve nothing. Label events are read live, so a re-run sees a later approval or removal;
unreadable evidence grants no exception.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

OFFICIAL_REPOSITORY = "razzant/ouroboros"
LABEL = "book-growth"
BINDING_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


def binds(repo: str, event: str, association: str, ref: str = "") -> bool:
    return repo == OFFICIAL_REPOSITORY and (
        (event == "pull_request" and association in BINDING_ASSOCIATIONS)
        or (event == "push" and ref == "refs/heads/ouroboros")
    )


def decide(repo: str, owner: str, event: str, association: str, label_events: list | None, ref: str = "") -> str:
    """``block``, ``warn`` or ``approved``; ``label_events`` is ``None`` when they could not be read."""
    if not binds(repo, event, association, ref):
        return "warn"
    history = [item for item in label_events or []
               if item.get("event") in ("labeled", "unlabeled") and (item.get("label") or {}).get("name") == LABEL]
    if history and history[-1]["event"] == "labeled" and (history[-1].get("actor") or {}).get("login") == owner:
        return "approved"
    return "block"


def read_pages(repo: str, path: str, token: str) -> list:
    """Read a GitHub list endpoint completely, preserving its event order."""
    items: list = []
    url = f"https://api.github.com/repos/{repo}/{path}?per_page=100"
    while url:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            items.extend(json.load(response))
            link = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link") or "")
        url = link.group(1) if link else ""
    return items


def read_label_events(repo: str, number: str, token: str) -> list:
    return read_pages(repo, f"issues/{number}/events", token)


def read_associated_pull_requests(repo: str, sha: str, token: str) -> list:
    return read_pages(repo, f"commits/{sha}/pulls", token)


def main(environ=os.environ, read=read_label_events, read_prs=read_associated_pull_requests) -> str:
    repo, event = environ.get("REPO", ""), environ.get("EVENT", "")
    association, ref = environ.get("ASSOCIATION", ""), environ.get("REF", "")
    owner, token = environ.get("OWNER", ""), environ.get("GH_TOKEN", "")
    mode = decide(repo, owner, event, association, None, ref)
    if binds(repo, event, association, ref):
        numbers = [environ["PR_NUMBER"]] if event == "pull_request" and environ.get("PR_NUMBER") else []
        if event == "push" and (sha := environ.get("SHA", "")):
            try:
                # Non-default branches can return both open and merged PRs, in any order.
                # merge_commit_sha is the landed merge/squash commit or the rebased tip.
                numbers = [str(pr["number"]) for pr in read_prs(repo, sha, token)
                           if pr.get("state") == "closed" and pr.get("merged_at")
                           and pr.get("merge_commit_sha") == sha
                           and (pr.get("base") or {}).get("ref") == "ouroboros"
                           and ((pr.get("base") or {}).get("repo") or {}).get("full_name") == repo]
            except Exception as exc:  # noqa: BLE001 -- unknown merge evidence approves nothing
                print(f"::warning::could not read merged PRs for {sha} ({exc}); no exception applied")
        for number in numbers:
            try:
                mode = decide(repo, owner, event, association, read(repo, number, token), ref)
            except Exception as exc:  # noqa: BLE001 -- any failure means the label is unknown
                print(f"::warning::could not read the {LABEL} label of #{number} ({exc}); no exception applied")
            if mode == "approved":
                break
    print(f"reference-book growth mode: {mode}")
    if environ.get("GITHUB_OUTPUT"):
        with open(environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"mode={mode}\n")
    return mode


if __name__ == "__main__":
    main()
    sys.exit(0)
