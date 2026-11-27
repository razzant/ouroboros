"""GitHub in memory for the post-merge hand-over (scripts/post_merge_routing.py), with failures on demand.

The routing tests and the post-merge check's end-to-end window test share it; neither imports the other.
"""
from __future__ import annotations

import urllib.error

from scripts import post_merge_routing as routing


class FakeGitHub:
    """Issues, pull requests and labels in memory, with failures on demand."""

    def __init__(self, *, issues=(), prs=None, label=False, fail=()):
        self.issues = {issue["number"]: dict(issue, state="open", labels=[{"name": routing.LABEL}]) for issue in issues}
        self.prs, self.label, self.fail = prs or {}, label, set(fail)
        self.comments, self.posts = [], []

    def _check(self, method, path):
        if any((method, path.split("?")[0]) == entry or (method, None) == entry for entry in self.fail):
            raise urllib.error.HTTPError(path, 502, "Bad Gateway", None, None)

    def get(self, path):
        self._check("GET", path)
        if path == f"labels/{routing.LABEL}" and not self.label:
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
        if path.startswith("issues/") and path[len("issues/"):].isdigit():
            return dict(self.issues[int(path[len("issues/"):])])
        return {}

    def pages(self, path):
        self._check("GET", path)
        if path.startswith("issues?"):
            return [issue for issue in self.issues.values() if issue["state"] == "open"
                    and {"name": routing.LABEL} in issue["labels"]]
        return self.prs.get(path.split("/")[1], [])

    def post(self, path, body):
        self._check("POST", path)
        self.posts.append((path, body))
        if path == "labels":
            self.label = True
        elif path == "issues":
            number = 100 + len(self.issues)
            self.issues[number] = {"number": number, "body": body["body"], "title": body["title"], "state": "open",
                                   "labels": [{"name": name} for name in body["labels"]] if self.label else []}
            return self.issues[number]
        elif path.endswith("/comments"):
            self.comments.append((int(path.split("/")[1]), body["body"]))
        return {}

    def patch(self, path, body):
        self._check("PATCH", path)
        issue = self.issues[int(path.split("/")[1])]
        issue.update(body)
        return issue


def tracking_issue(number, block, *, head="f" * 40):
    """An open `post-merge-red` issue carrying `block`, as `FakeGitHub(issues=...)` takes it."""
    return {"number": number, "body": f"Found failing at `{head}`: run\n\n" + routing.machine_block({"format": 1, **block})}
