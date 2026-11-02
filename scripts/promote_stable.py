#!/usr/bin/env python3
"""Promote a published release to Stable: the one operator step after a release.

Stable promotion moves two official branches once a release has published all its
proof-bound installers (ARCHITECTURE §8 "Build scripts"):

- ``ouroboros-stable``, the QA feed and recovery fallback, fast-forwards to the
  release tag's commit unless it already contains it; it never moves back.
- ``main``, the Stable boundary behind the default README, Pages (``main:/docs``)
  and the Homebrew tap, gains one commit whose first parent is the previous
  ``main``, whose second parent is the tag's commit and whose tree is the tag's tree
  plus ``Casks/ouroboros.rb`` for that release: the projection
  ``scripts/sync_legacy_mirrors.py`` builds for the archived mirrors. A ``main``
  that already contains the tag (today's raw tag commit, or a re-rendered cask)
  gets a single-parent commit instead. ``main`` always fast-forwards, the
  immutable tag stays an ancestor the Stable resolver selects, and a release may
  be tagged before or after any promotion. The step never writes ``ouroboros``.

Before writing it reads only public data: the tag's GitHub release, its
``SHA256SUMS``, ``release-evidence.json`` and macOS smoke receipt, and with
``--verify-download`` the DMG bytes. Every digest must agree and the receipt must
prove an arm64, notarized build. It refuses anything but a plain ``vX.Y.Z`` tag, a
draft, prerelease or unproven release, a release older than the newest one on
``main`` and an ``ouroboros-stable`` that does not lead to the tag. The write is
one atomic push without force: if either branch moved meanwhile, both stay as
they were and a rerun plans again. Repeating a finished promotion changes nothing.

Rollout: a promoted ``main`` runs the release's own mirror refresh
(``.github/workflows/sync-legacy-mirrors.yml`` and ``scripts/sync_legacy_mirrors.py``).
A release tagged before that script could read a promotion commit through to its tag
(v7.8.0 included) takes it for an untagged commit and skips the refresh, automatic or
dispatched. The first promotion, of v7.8.0, loses nothing: the mirrors were already at
7.8.0 when this step was added. Promoting another such release needs one manual run of
the repaired script with ``--source-ref vX.Y.Z`` (and ``MIRROR_TOKEN``) until a later
Stable release carries it; this step itself never writes a mirror.

Usage: ``python3 scripts/promote_stable.py promote vX.Y.Z [--remote origin]
[--verify-download] [--push]`` from any checkout with push access; without
``--push`` it prints the plan. ``render`` prints a cask. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
# Loaded by path, like release_proof.py loads release_sync.py, so the step reuses the
# owners of asset naming, smoke-receipt requirements and the Stable rule.
_SYNC = runpy.run_path(str(ROOT / "ouroboros" / "tools" / "release_sync.py"))
_PROOF = runpy.run_path(str(ROOT / "scripts" / "release_proof.py"))
_UPDATE = runpy.run_path(str(ROOT / "supervisor" / "update_source.py"))

REPOSITORY = "razzant/ouroboros"
API = f"https://api.github.com/repos/{REPOSITORY}"
STABLE, QA = "main", "ouroboros-stable"
CASK_PATH = "Casks/ouroboros.rb"
PROOF_ID = "macos-arm64"
RECEIPT = f"release-smoke-{PROOF_ID}.json"
# Ouroboros.spec owns these (tests/test_promote_stable.py pins them to it). Homebrew reads a
# bare `depends_on macos:` symbol as that release or newer.
APP_BUNDLE = "Ouroboros.app"
BUNDLE_ID = "com.ouroboros.agent"
MACOS_MINIMUM = ("12.0", ":monterey")
AUTHOR = ("Ouroboros", "311266734+ouroboros-agent@users.noreply.github.com")
_PLAIN_TAG = _UPDATE["_RELEASE_TAG_RE"]
_TAGS = _UPDATE["MANAGED_TAG_NAMESPACE"]
_SHA256 = re.compile(r"[0-9a-f]{64}")

_TEMPLATE = """\
# Written by scripts/promote_stable.py into main's promotion commit for this
# Stable release, from its published proof. Do not edit by hand.
cask "ouroboros" do
  version "{version}"
  sha256 "{sha256}"

  url "{url}"
  name "Ouroboros"
  desc "Self-creating AI agent"
  homepage "https://ouroboros-agent.ai/"

  # A plain `brew upgrade` first compares the installed app's own version.
  auto_updates true
  depends_on arch: :arm64
  depends_on macos: {macos}

  app "{app}"

  # Asks a running app to quit before an upgrade or uninstall. No launchctl or
  # zap: launchctl also runs on upgrade and would drop the sign-in registration,
  # and ~/Ouroboros holds the owner's repository, memory and projects.
  uninstall quit: "{bundle_id}"
{caveat}
end
"""

_SETUP_780_CAVEAT = """\
  caveats do
    <<~EOS
      In 7.8.0, finish Setup before the first Quit (including a Homebrew upgrade).
      Quitting earlier can make a new data folder refuse initialization (#2412).
      If affected, back up the data folder and diagnose it before changing anything.
      Use Reset runtime data only for a confirmed empty new installation.
      Guidance: https://github.com/razzant/ouroboros/blob/ouroboros/README.md#homebrew-on-macos
    EOS
  end
"""


def render_cask(version: str, sha256: str) -> str:
    """The cask for one plain release version and its DMG digest."""
    if not _PLAIN_TAG.fullmatch(f"v{version}") or not _SHA256.fullmatch(str(sha256)):
        raise ValueError(f"a cask needs a plain X.Y.Z version and a sha256, not {version!r} / {sha256!r}")
    # The URL owner with Ruby's own interpolation as the version: one naming source.
    url = _SYNC["release_asset_download_url"](PROOF_ID, "#{version}", repository=REPOSITORY)
    # main retains the release's README. Put this measured old-release warning at the
    # install surface too, without changing the release tag or guessing other releases.
    return _TEMPLATE.format(version=version, sha256=sha256, url=url, macos=MACOS_MINIMUM[1],
                            app=APP_BUNDLE, bundle_id=BUNDLE_ID,
                            caveat=_SETUP_780_CAVEAT if version == "7.8.0" else "")


def _version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _PLAIN_TAG.fullmatch(tag).groups())


def _git(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None,
         stdin: str | None = None) -> str:
    # Bytes in and out: text-mode pipes would turn the cask's LF into CRLF on Windows.
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                            input=stdin.encode("utf-8") if stdin is not None else None,
                            env={**os.environ, **(env or {})}, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {result.stderr.decode('utf-8', 'replace').strip()[:400]}")
    return result.stdout.decode("utf-8").strip()


def _is_ancestor(old: str, new: str, cwd: Path | None) -> bool:
    code = subprocess.run(["git", "merge-base", "--is-ancestor", old, new], cwd=cwd,
                          capture_output=True, timeout=120).returncode
    if code not in (0, 1):
        raise RuntimeError(f"git merge-base --is-ancestor {old[:10]} {new[:10]} failed ({code})")
    return code == 0


# --------------------------------------------------------------------------- public facts

def https_get(url: str) -> bytes:
    headers = {"User-Agent": "ouroboros-promote-stable"}
    token = os.environ.get("GH_TOKEN", "").strip()
    if url.startswith("https://api.github.com/"):  # a read token never reaches asset hosts
        headers["Accept"] = "application/vnd.github+json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        return response.read()


def https_sha256(url: str) -> tuple[str, int]:
    """Digest and size of the bytes an unauthenticated download of *url* serves."""
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={"User-Agent": "ouroboros-promote-stable"})
    with urllib.request.urlopen(request, timeout=300) as response:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _expect(condition: bool, problem: str) -> None:
    if not condition:
        raise ValueError(problem)


def _read_assets(release: dict, tag: str, get: Callable[[str], bytes]) -> tuple[dict, dict[str, bytes]]:
    _expect(release.get("tag_name") == tag and not release.get("draft") and not release.get("prerelease"),
            f"{tag} is not a published, non-prerelease GitHub release")
    assets = {row.get("name"): row for row in release.get("assets") or [] if isinstance(row, dict)}
    dmg = _SYNC["release_asset_name"](PROOF_ID, tag[1:])
    missing = [name for name in (dmg, "SHA256SUMS", "release-evidence.json", RECEIPT) if name not in assets]
    _expect(not missing, f"release {tag} lacks {', '.join(missing)}")
    files = {name: get(assets[name]["browser_download_url"])
             for name in ("SHA256SUMS", "release-evidence.json", RECEIPT)}
    # Each byte read must be the upload GitHub recorded (the release job checked those digests).
    for name, data in files.items():
        recorded = assets[name].get("digest")
        _expect(not recorded or recorded == f"sha256:{hashlib.sha256(data).hexdigest()}",
                f"{name} does not match the digest GitHub recorded for it")
    return assets, files


def _sums(text: str) -> dict[str, str]:
    rows = (line.split("  ", 1) for line in text.splitlines() if line.strip())
    return {name: digest for digest, name in rows}


def _check_proof(files: dict[str, bytes], assets: dict, tag: str, commit: str) -> str:
    """The DMG sha256 after SHA256SUMS, evidence, receipt and GitHub all agree."""
    version, dmg = tag[1:], _SYNC["release_asset_name"](PROOF_ID, tag[1:])
    sums = _sums(files["SHA256SUMS"].decode("utf-8"))
    sha256 = sums.get(dmg, "")
    _expect(bool(_SHA256.fullmatch(sha256)), f"SHA256SUMS has no digest for {dmg}")
    _expect(sums.get(RECEIPT) == hashlib.sha256(files[RECEIPT]).hexdigest(), f"SHA256SUMS does not cover {RECEIPT}")
    evidence = json.loads(files["release-evidence.json"])
    _expect(evidence.get("kind") == "build_time_release_proof" and evidence.get("version") == version
            and evidence.get("source", {}).get("tag") == tag and evidence.get("source", {}).get("commit") == commit,
            f"release-evidence.json is not the proof of {tag} at {commit[:12]}")
    records = [row for row in evidence.get("artifacts") or []
               if isinstance(row, dict) and row.get("proofId") == PROOF_ID]
    _expect(len(records) == 1, f"release-evidence.json has no single {PROOF_ID} record")
    record = records[0]
    _expect(record.get("name") == dmg and record.get("sha256") == sha256 and record.get("smokeReceipt") == RECEIPT,
            f"release-evidence.json and SHA256SUMS disagree about {dmg}")
    _expect(record.get("size") == assets[dmg].get("size"), f"{dmg} size differs from its release evidence")
    _expect(record.get("notarization") == {"status": "notarized"}, f"{dmg} is not notarized")
    receipt = json.loads(files[RECEIPT])
    required = _PROOF["REQUIRED_SMOKE_CHECKS"][PROOF_ID] | _PROOF["NOTARIZATION_SMOKE_CHECKS"]
    _expect(receipt.get("status") == "passed" and receipt.get("proofId") == PROOF_ID
            and receipt.get("artifact") == dmg and receipt.get("sha256") == sha256
            and receipt.get("releaseTag") == tag and receipt.get("sourceCommit") == commit
            and receipt.get("notarization") == {"status": "notarized"}
            and required <= set(receipt.get("checks") or ()),
            f"{RECEIPT} does not prove a passed, arm64, notarized {dmg}")
    _expect(assets[dmg].get("digest") in (None, f"sha256:{sha256}"), f"GitHub records another digest for {dmg}")
    return sha256


def release_facts(tag: str, commit: str, *, get: Callable[[str], bytes] = https_get,
                  download: Callable[[str], tuple[str, int]] | None = None) -> dict[str, Any]:
    """Public, cross-checked inputs of the cask for one release tag and its commit."""
    assets, files = _read_assets(json.loads(get(f"{API}/releases/tags/{tag}")), tag, get)
    sha256 = _check_proof(files, assets, tag, commit)
    dmg = assets[_SYNC["release_asset_name"](PROOF_ID, tag[1:])]
    url = _SYNC["release_asset_download_url"](PROOF_ID, tag[1:], repository=REPOSITORY)
    _expect(dmg.get("browser_download_url") == url, f"the DMG is not published at {url}")
    if download is not None:
        _expect(download(url) == (sha256, dmg.get("size")), f"the bytes served at {url} are not the proven DMG")
    return {"tag": tag, "version": tag[1:], "commit": commit, "url": url, "sha256": sha256,
            "size": dmg.get("size"), "downloadVerified": download is not None}


# --------------------------------------------------------------------------- official refs

def fetch_official(remote: str, cwd: Path | None = None) -> dict[str, str]:
    """Fresh main and ouroboros-stable, with release tags where the updater keeps them."""
    _git("fetch", "--quiet", "--no-tags", remote, f"+refs/heads/{STABLE}:refs/remotes/{remote}/{STABLE}",
         f"+refs/heads/{QA}:refs/remotes/{remote}/{QA}", f"+refs/tags/v*:{_TAGS}/v*", cwd=cwd)
    return {name: _git("rev-parse", f"{remote}/{name}^{{commit}}", cwd=cwd) for name in (STABLE, QA)}


def tag_commit(tag: str, cwd: Path | None = None) -> str:
    _expect(bool(_PLAIN_TAG.fullmatch(tag)), f"Stable takes a plain vX.Y.Z release tag, not {tag!r}")
    return _git("rev-parse", "--verify", f"{_TAGS}/{tag}^{{commit}}", cwd=cwd)


def newest_release_on(ref: str, cwd: Path | None = None) -> str:
    """The newest plain release tag *ref* contains, or ''."""
    tags = _git("for-each-ref", f"--merged={ref}", "--format=%(refname:strip=3)", f"{_TAGS}/v*", cwd=cwd)
    plain = [tag for tag in tags.split() if _PLAIN_TAG.fullmatch(tag)]
    return max(plain, key=_version_key, default="")


def stable_selection(remote: str, cwd: Path | None = None) -> str:
    """The tag the in-app Stable channel selects from these refs (its own resolver)."""
    def capture(args: list[str]) -> tuple[int, str, str]:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120)
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    tag_ref, _commit, _error = _UPDATE["resolve_managed_update_target"](
        remote, STABLE, f"{remote}/{STABLE}", update_channel="stable", capture=capture)
    return tag_ref.rsplit("/", 1)[-1] if tag_ref else ""


# --------------------------------------------------------------------------- promotion

def _projected_tree(commit: str, cask: str, cwd: Path | None) -> str:
    """The tree of *commit* plus the cask, as one tree object."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": os.path.join(tmp, "index")}
        _git("read-tree", commit, cwd=cwd, env=env)
        blob = _git("hash-object", "-w", "--stdin", cwd=cwd, stdin=cask)
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{CASK_PATH}", cwd=cwd, env=env)
        return _git("write-tree", cwd=cwd, env=env)


def plan_promotion(tag: str, commit: str, cask: str, main: str, qa: str,
                   cwd: Path | None = None) -> dict[str, str] | None:
    """New tips for main and ouroboros-stable (None where a branch stays), or None when done."""
    newest = newest_release_on(main, cwd)
    if newest and _version_key(newest) > _version_key(tag):
        raise ValueError(f"main already carries {newest}; promoting {tag} would downgrade Stable")
    if _is_ancestor(commit, qa, cwd):
        new_qa = None  # QA already contains the release; never move it back
    else:
        _expect(_is_ancestor(qa, commit, cwd), f"{QA} {qa[:10]} does not lead to {tag}; reconcile it first")
        new_qa = commit
    tree = _projected_tree(commit, cask, cwd)
    on_main = _is_ancestor(commit, main, cwd)
    new_main = None
    if not (on_main and _git("rev-parse", f"{main}^{{tree}}", cwd=cwd) == tree):
        name, email = AUTHOR
        ident = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
                 "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}
        parents = [main] if on_main else [main, commit]
        message = (f"Promote Ouroboros {tag} to Stable\n\nmain is {tag} ({commit}) plus {CASK_PATH} "
                   "for its published DMG.\n")
        new_main = _git("commit-tree", tree, *(arg for sha in parents for arg in ("-p", sha)), "-m", message,
                        cwd=cwd, env=ident)
    return None if new_main is None and new_qa is None else {STABLE: new_main, QA: new_qa}


def push_promotion(plan: dict[str, str | None], remote: str, cwd: Path | None = None) -> None:
    refspecs = [f"{sha}:refs/heads/{branch}" for branch, sha in plan.items() if sha]
    # Atomic and never forced: a branch that moved since the plan rejects both.
    _git("push", "--atomic", "--quiet", remote, *refspecs, cwd=cwd)


def promoted_problems(tag: str, commit: str, cask: str, remote: str, cwd: Path | None = None) -> list[str]:
    """What a fresh read of the official refs finds wrong with this promotion."""
    refs = fetch_official(remote, cwd)
    problems = []
    if (selected := stable_selection(remote, cwd)) != tag:
        problems.append(f"the Stable resolver selects {selected or 'nothing'}, not {tag}")
    if not _is_ancestor(commit, refs[STABLE], cwd) or (
            _git("rev-parse", f"{refs[STABLE]}^{{tree}}", cwd=cwd) != _projected_tree(commit, cask, cwd)):
        problems.append(f"main is not {tag} plus its {CASK_PATH}")
    if not _is_ancestor(commit, refs[QA], cwd):
        problems.append(f"{QA} does not contain {tag}")
    return problems


def promote(tag: str, *, remote: str = "origin", cwd: Path | None = None, push: bool = False,
            read_release: Callable[[str, str], dict[str, Any]] = release_facts) -> int:
    refs = fetch_official(remote, cwd)
    commit = tag_commit(tag, cwd)
    facts = read_release(tag, commit)  # public reads only; nothing is written before this passes
    cask = render_cask(facts["version"], facts["sha256"])
    plan = plan_promotion(tag, commit, cask, refs[STABLE], refs[QA], cwd)
    if plan is None:
        print(f"{tag} is already Stable on {STABLE} and {QA}; nothing to push")
        return 0
    for branch, sha in plan.items():
        print(f"{branch}: {refs[branch][:10]} -> {sha[:10]}" if sha else f"{branch}: stays at {refs[branch][:10]}")
    if not push:
        print("plan only; rerun with --push to publish it")
        return 0
    push_promotion(plan, remote, cwd)
    problems = promoted_problems(tag, commit, cask, remote, cwd)
    for problem in problems:
        print(f"after the push: {problem}", file=sys.stderr)
    if not problems:
        print(f"promoted {tag} to Stable")
    return 1 if problems else 0


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("promote", help="promote a published release tag to Stable")
    step.add_argument("tag")
    step.add_argument("--remote", default="origin")
    step.add_argument("--verify-download", action="store_true", help="also hash the DMG the URL serves")
    step.add_argument("--push", action="store_true", help="publish the plan (default: print it)")
    render = commands.add_parser("render", help="print the cask for a version and DMG digest")
    render.add_argument("version")
    render.add_argument("sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            sys.stdout.write(render_cask(args.version, args.sha256))
            return 0
        download = https_sha256 if args.verify_download else None
        return promote(args.tag, remote=args.remote, push=args.push,
                       read_release=lambda tag, commit: release_facts(tag, commit, download=download))
    except (ValueError, KeyError, RuntimeError, OSError, urllib.error.URLError) as exc:
        print(f"promote stable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
