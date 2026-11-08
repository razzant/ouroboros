#!/usr/bin/env python3
"""Refresh the archived joi-lab repositories from an official stable commit.

Before 2026-05-31 Ouroboros lived at ``joi-lab/ouroboros`` (a fork of this
repository) and ``joi-lab/ouroboros-desktop``. Both stay archived with a
"moved" notice, yet old posts, clones, Colab links and installed builds still
reach them. After every stable release (``main`` takes a release only once it is
published, as a promotion commit over its tag) this script reads the source
commit through to that tag, checks that it is the published, non-prerelease
release, then briefly unarchives each mirror and:

- fast-forwards its update branches to the stable commit (never a force push:
  a branch that is not an ancestor of the stable commit is reported and left);
- records a ``main`` commit whose tree is the stable tree plus the mirror's own
  README (and, for the fork, a notebook at the path the old Colab badge opens);
  its parents are the mirror's ``main`` and the stable commit, so no published
  ref is rewritten;
- keeps one "moved" release, latest, pointing at the official latest release;
- marks the outdated desktop releases at the top of their notes;
- turns GitHub Actions off, limits new pull requests to collaborators and
  archives the mirror again, also on error.

Owner decisions 2026-10-11 №1=B (refresh on every stable release), №2=A (keep
and mark old releases), №5=A (act as ouroboros-agent). ``MIRROR_TOKEN`` needs
admin on both mirrors (archive state) and the workflow scope (the stable tree
carries ``.github/workflows``). Standard library only.

Usage: ``python3 scripts/sync_legacy_mirrors.py --source-ref <sha> [--dry-run]
[--only joi-lab/ouroboros] [--skip-branch joi-lab/ouroboros-desktop:ouroboros]``;
``--archive-only`` just archives both mirrors (the workflow's cleanup step).
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

API = "https://api.github.com"
OFFICIAL_URL = "https://github.com/razzant/ouroboros"
LATEST_URL = f"{OFFICIAL_URL}/releases/latest"
POINTER_TAG = "moved-to-razzant"
OUTDATED_MARK = "<!-- ouroboros-outdated-release -->"
AUTHOR = ("Ouroboros", "311266734+ouroboros-agent@users.noreply.github.com")


@dataclasses.dataclass(frozen=True)
class Mirror:
    repo: str
    branches: tuple[str, ...]
    title: str
    intro: str
    colab_stub: bool = False
    mark_old_releases: bool = False


MIRRORS = (
    Mirror(
        repo="joi-lab/ouroboros",
        branches=("ouroboros", "ouroboros-stable"),
        title="Ouroboros moved to razzant/ouroboros",
        intro=("This repository was the first public home of Ouroboros (the Google Colab version) "
               "and later a mirror of the official repository."),
        colab_stub=True,
    ),
    Mirror(
        repo="joi-lab/ouroboros-desktop",
        branches=("ouroboros",),
        title="Ouroboros Desktop moved to razzant/ouroboros",
        intro=("This repository held Ouroboros Desktop from February to May 2026; "
               "desktop apps installed then still look for updates here."),
        mark_old_releases=True,
    ),
)


def render_readme(mirror: Mirror, version: str, source: str, branches: tuple[str, ...] = ()) -> str:
    return (
        f"# {mirror.title}\n\n"
        "> [!IMPORTANT]\n"
        f"> This is a read-only archive. Ouroboros is developed in the official repository: **{OFFICIAL_URL}**\n"
        ">\n"
        f"> - Download the current version: {LATEST_URL}\n"
        f"> - Issues, pull requests, discussions and stars: {OFFICIAL_URL}\n"
        "> - Website: https://ouroboros-agent.ai/\n\n"
        f"{mirror.intro} After each stable release these branches are refreshed automatically: "
        f"`{'`, `'.join(('main', *branches))}`. They hold Ouroboros v{version} (official commit {source[:9]}); "
        "other branches are historical.\n"
    )


def render_pointer(mirror: Mirror, version: str) -> str:
    body = (
        "## Ouroboros has moved\n\n"
        f"Ouroboros is developed at **{OFFICIAL_URL}**.\n\n"
        f"**Download the current version (v{version}):** {LATEST_URL}\n\n"
        "The *Source code* archives attached to this notice are a historical snapshot; "
        f"get the current source from {OFFICIAL_URL}.\n"
    )
    if mirror.mark_old_releases:
        body += ("\nThe releases listed below this one are outdated Ouroboros Desktop builds from "
                 "February to May 2026. They are kept for history and do not receive updates.\n")
    return body


def mark_outdated(body: str | None) -> str:
    """Prepend the outdated-build notice once; idempotent."""
    text = body or ""
    if OUTDATED_MARK in text:
        return text
    notice = (f"{OUTDATED_MARK}\n> [!WARNING]\n> **Outdated build.** This version no longer receives updates. "
              f"Download the current Ouroboros: {LATEST_URL}\n")
    return notice + ("\n" + text if text.strip() else "")


def colab_stub() -> str:
    """The notebook behind the first-generation 'Open in Colab' badge."""
    quickstart = "https://raw.githubusercontent.com/razzant/ouroboros/main/notebooks/colab_quickstart.py"
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"colab": {"name": "Ouroboros has moved"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Ouroboros has moved\n", "\n",
                f"This early Colab notebook is retired. Ouroboros is developed at {OFFICIAL_URL}.\n", "\n",
                "- Desktop app and other installs: https://ouroboros-agent.ai/install/\n",
                f"- The first-generation Colab version is preserved on the `legacy-google-colab` branch: "
                f"{OFFICIAL_URL}/tree/legacy-google-colab\n",
                "- To run the current version in Colab, run the cell below; it fetches the official "
                "`notebooks/colab_quickstart.py` (runtime data stays on your Google Drive).\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                f"!curl -fsSL {quickstart} -o /content/ouroboros_colab_quickstart.py\n",
                "%run /content/ouroboros_colab_quickstart.py\n",
            ]},
        ],
    }
    return json.dumps(notebook, indent=1) + "\n"


class GitHub:
    def __init__(self, token: str, dry_run: bool) -> None:
        self.token, self.dry_run = token, dry_run

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        if self.dry_run and method != "GET":
            print(f"DRY-RUN {method} {path} {json.dumps(body or {})[:200]}")
            return 0, {}
        data = json.dumps(body).encode() if body is not None else None
        # Only idempotent requests are retried: a repeated POST could create a second release.
        attempts = 1 if method == "POST" else 3
        for attempt in range(1, attempts + 1):
            req = urllib.request.Request(f"{API}{path}", data=data, method=method, headers={
                "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "ouroboros-legacy-mirrors"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                    return resp.status, json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError as exc:
                message = exc.read().decode("utf-8", "replace")[:400]
                limited = exc.code == 429 or (exc.code == 403 and "rate limit" in message.lower())
                if not (exc.code >= 500 or limited) or attempt == attempts:
                    return exc.code, {"message": message}
                if limited:  # wait as long as GitHub asks, a minute when it does not say
                    time.sleep(min(300, int(exc.headers.get("Retry-After") or 60)))
                    continue
            except (urllib.error.URLError, http.client.HTTPException, OSError):
                if attempt == attempts:
                    raise
            time.sleep(3 * attempt)
        raise AssertionError("unreachable")

    def must(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        status, data = self.call(method, path, body)
        if status and status >= 300:
            raise RuntimeError(f"{method} {path} -> HTTP {status}: {data.get('message', '')}")
        return data

    def releases(self, repo: str) -> list[dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}  # pages can repeat an item; one entry per release
        for page in range(1, 100):
            batch = self.must("GET", f"/repos/{repo}/releases?per_page=30&page={page}")
            if not batch:
                break
            out.update((release["id"], release) for release in batch)
        return list(out.values())


def git(*args: str, env: dict[str, str] | None = None, stdin: str | None = None) -> str:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, input=stdin,
                                env={**os.environ, **(env or {})}, timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"git {' '.join(args[:3])}: no answer in 15 minutes") from exc
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {result.stderr.strip()[:400]}")
    return result.stdout.strip()


def auth_args(token: str) -> list[str]:
    header = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", "credential.helper=", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {header}"]


def is_ancestor(old: str, new: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", old, new]).returncode == 0


def build_main_commit(mirror: Mirror, source: str, parent: str | None, version: str,
                      branches: tuple[str, ...] = ()) -> str | None:
    """Return a new main commit, or None when the mirror's main already has this tree."""
    files = {"README.md": render_readme(mirror, version, source, branches)}
    if mirror.colab_stub:
        files["notebooks/quickstart.ipynb"] = colab_stub()
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": os.path.join(tmp, "index")}
        git("read-tree", source, env=env)
        for path, text in files.items():
            blob = git("hash-object", "-w", "--stdin", stdin=text)
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
        tree = git("write-tree", env=env)
    if parent and git("rev-parse", f"{parent}^{{tree}}") == tree:
        return None
    parents = (["-p", parent] if parent else []) + ["-p", source]
    name, email = AUTHOR
    ident = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name,
             "GIT_COMMITTER_EMAIL": email}
    message = f"Refresh archive from official stable v{version}\n\nSource: {OFFICIAL_URL}/commit/{source}\n"
    return git("commit-tree", tree, *parents, "-m", message, env=ident)


def refresh(gh: GitHub, token: str, mirror: Mirror, source: str, version: str, skip: set[str],
            log: list[str]) -> None:
    url = f"https://github.com/{mirror.repo}.git"
    slug = mirror.repo.replace("/", "__")
    refspecs = [f"+refs/heads/{b}:refs/legacy-mirror/{slug}/{b}" for b in ("main", *mirror.branches)]
    git(*auth_args(token), "fetch", "--no-tags", "--quiet", url, *refspecs)
    pushes: list[str] = []
    refreshed = tuple(b for b in mirror.branches if f"{mirror.repo}:{b}" not in skip)
    for branch in mirror.branches:
        if branch not in refreshed:
            log.append(f"{mirror.repo}:{branch} skipped by request")
            continue
        old = git("rev-parse", f"refs/legacy-mirror/{slug}/{branch}")
        if old == source:
            continue
        if not is_ancestor(old, source):
            raise RuntimeError(f"{mirror.repo}:{branch} {old[:10]} is not an ancestor of {source[:10]}; "
                               "refusing a force push")
        pushes.append(f"{source}:refs/heads/{branch}")
    main_old = git("rev-parse", f"refs/legacy-mirror/{slug}/main")
    main_new = build_main_commit(mirror, source, main_old, version, refreshed)
    if main_new:
        pushes.append(f"{main_new}:refs/heads/main")
    if pushes and not gh.dry_run:
        git(*auth_args(token), "push", "--quiet", url, *pushes)
    verb = "would push" if gh.dry_run else "pushed"
    log.append(f"{mirror.repo}: {verb} {[p.split(':', 1)[1] for p in pushes] or 'nothing'}")

    pointer_body = render_pointer(mirror, version)
    status, existing = gh.call("GET", f"/repos/{mirror.repo}/releases/tags/{POINTER_TAG}")
    if status not in (200, 404) and not gh.dry_run:
        raise RuntimeError(f"pointer release lookup -> HTTP {status}: {existing.get('message', '')}")
    if status == 200:
        gh.must("PATCH", f"/repos/{mirror.repo}/releases/{existing['id']}",
                {"body": pointer_body, "make_latest": "true"})
    else:
        gh.must("POST", f"/repos/{mirror.repo}/releases", {
            "tag_name": POINTER_TAG, "target_commitish": main_new or main_old, "name": "Ouroboros has moved",
            "body": pointer_body, "draft": False, "prerelease": False, "make_latest": "true"})
    if mirror.mark_old_releases:
        marked = 0
        for release in gh.releases(mirror.repo):
            if release.get("tag_name") == POINTER_TAG or release.get("draft"):
                continue
            new_body = mark_outdated(release.get("body"))
            if new_body != (release.get("body") or ""):
                gh.must("PATCH", f"/repos/{mirror.repo}/releases/{release['id']}",
                        {"body": new_body, "make_latest": "false"})
                marked += 1
                time.sleep(0 if gh.dry_run else 1)  # GitHub asks for a pause between mutations
        log.append(f"{mirror.repo}: {'would mark' if gh.dry_run else 'marked'} {marked} outdated releases")


def released_commit(source: str) -> str:
    """The tagged commit a Stable promotion projected onto main (scripts/promote_stable.py).

    A promotion commit's tree is its tag's tree plus ``Casks/``; the tag is its last
    parent. Any other commit is returned as it is."""
    for _ in range(20):
        parents = git("rev-list", "--parents", "-n", "1", source).split()[1:]
        below = next((parent for parent in reversed(parents)
                      if (changed := git("diff", "--name-only", parent, source).splitlines())
                      and all(path.startswith("Casks/") for path in changed)), None)
        if below is None:
            return source
        source = below
    return source


def published_stable_problem(gh: GitHub, source: str, version: str) -> str:
    """Empty when ``source`` is the commit of the published, non-prerelease ``v<VERSION>``."""
    tag = f"v{version}"
    try:
        tagged = git("rev-parse", f"{tag}^{{commit}}")
    except RuntimeError:
        return f"no tag {tag}"
    if tagged != source:
        return f"{tag} is {tagged[:10]}"
    status, release = gh.call("GET", f"/repos/razzant/ouroboros/releases/tags/{tag}")
    if status == 404:
        return f"no GitHub release {tag}"
    if status != 200:  # a failed check is an error, never a quiet "nothing to refresh"
        raise RuntimeError(f"release {tag} lookup -> HTTP {status}: {release.get('message', '')}")
    if release.get("draft") or release.get("prerelease"):
        return f"release {tag} is a draft or a prerelease"
    return ""


def archive_all(gh: GitHub, mirrors: list[Mirror]) -> int:
    failures = 0
    for mirror in mirrors:
        try:
            if not gh.must("GET", f"/repos/{mirror.repo}").get("archived"):
                gh.must("PATCH", f"/repos/{mirror.repo}", {"archived": True})
            print(f"{mirror.repo}: archived")
        except Exception as exc:  # noqa: BLE001 - name the repository left open
            failures += 1
            print(f"{mirror.repo}: STILL UNARCHIVED: {exc}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--skip-branch", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--archive-only", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get("MIRROR_TOKEN", "").strip()
    if not token:
        print("MIRROR_TOKEN is not set", file=sys.stderr)
        return 2
    gh = GitHub(token, args.dry_run)
    if args.archive_only:
        return archive_all(gh, [m for m in MIRRORS if not args.only or m.repo in args.only])
    source = released_commit(git("rev-parse", f"{args.source_ref}^{{commit}}"))
    version = git("show", f"{source}:VERSION").strip()
    try:
        reason = published_stable_problem(gh, source, version)
    except RuntimeError as exc:
        print(f"cannot confirm that {source[:10]} is a published stable release: {exc}", file=sys.stderr)
        return 1
    if reason:
        print(f"{source[:10]} is not a published stable release ({reason}); nothing to refresh")
        return 0
    log: list[str] = [f"acting as {gh.must('GET', '/user').get('login', '?')}"]
    failures = 0
    for mirror in MIRRORS:
        if args.only and mirror.repo not in args.only:
            continue
        unarchived = False
        try:
            if gh.must("GET", f"/repos/{mirror.repo}").get("archived"):
                gh.must("PATCH", f"/repos/{mirror.repo}", {"archived": False})
                unarchived = True
            gh.must("PUT", f"/repos/{mirror.repo}/actions/permissions", {"enabled": False})
            # New pull requests only from collaborators; the old ones stay readable (owner 2026-05-31: keep them).
            gh.must("PATCH", f"/repos/{mirror.repo}", {"pull_request_creation_policy": "collaborators_only"})
            refresh(gh, token, mirror, source, version, set(args.skip_branch), log)
        except Exception as exc:  # noqa: BLE001 - every mirror is attempted; the archive is restored below
            failures += 1
            log.append(f"{mirror.repo}: FAILED {exc}")
        finally:
            try:
                gh.must("PATCH", f"/repos/{mirror.repo}", {"archived": True})
                log.append(f"{mirror.repo}: {'would archive' if gh.dry_run else 'archived'}"
                           f"{' again' if unarchived else ''}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log.append(f"{mirror.repo}: FAILED to archive: {exc}")
    print(f"source v{version} {source}")
    print("\n".join(log))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
