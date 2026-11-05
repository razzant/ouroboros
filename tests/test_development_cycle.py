"""BIBLE P9's development cycle on real Git: one marker, neutral landings, a plain cut, the next marker.

The real owners do the work over local bare repositories: release sync, commit admission, the review
binding, the tag writer, the carrier resolver, startup's version check, Stable promotion, the bundle
builder, CI's tag preflight and its tag-source step. GitHub's server, signing and a published build stay
outside this proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from ouroboros import agent_startup_checks as startup
from ouroboros import commit_admission as admission
from ouroboros.tools.git import (
    _auto_tag_on_version_bump,
    _fingerprint_staged_diff,
    _review_binding_precondition_error,
    _verify_reviewed_commit_binding,
)
from ouroboros.tools.release_sync import (
    carrier_spans_for,
    locate_carrier_span,
    published_download_version,
    sync_release_metadata,
)
from scripts import promote_stable as promotion
from supervisor import git_ops
from supervisor.update_carriers import resolve_carrier_conflicts
from tests.test_release_sync import _download_page, _download_references, _make_repo

ROOT = Path(__file__).resolve().parents[1]
IDENT = ("-c", "user.name=t", "-c", "user.email=t@example.invalid")


def _git(cwd, *args, check=True):
    result = subprocess.run(["git", *IDENT, *args], cwd=cwd, capture_output=True, text=True)
    if check and result.returncode:
        raise AssertionError(f"git {' '.join(args)}: {result.stderr}")
    return result.stdout.strip() if check else result


class Cycle:
    """origin (bare) and one integrator clone; the public 7.9.0 is on every official branch."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.origin, self.work, self.publisher = tmp_path / "origin.git", tmp_path / "work", tmp_path / "publisher"
        _git(tmp_path, "init", "-q", "--bare", str(self.origin))
        self.work.mkdir()
        _git(self.work, "init", "-q")
        _git(self.work, "config", "core.autocrlf", "false")
        # The product's commit and tag steps read the checkout's Git identity (a packaged install's is written by
        # ouroboros/launcher_bootstrap.py, a source clone has its owner's), so this one gets its own, never the host's.
        # Its text policy is the repository's .gitattributes: LF in the index, whatever the OS writes.
        for setting in IDENT[1::2]:
            _git(self.work, "config", *setting.split("=", 1))
        (self.work / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
        _make_repo(self.work, "7.9.0")
        sync_release_metadata(str(self.work))
        _git(self.work, "add", "-A")
        _git(self.work, "commit", "-qm", "release 7.9.0")
        _git(self.work, "tag", "-a", "v7.9.0", "-m", "v7.9.0")
        self.push("HEAD:refs/heads/main", "HEAD:refs/heads/ouroboros-stable", "HEAD:refs/heads/ouroboros",
                  "refs/tags/v7.9.0")
        _git(self.origin, "symbolic-ref", "HEAD", "refs/heads/main")
        _git(tmp_path, "clone", "-q", str(self.origin), str(self.publisher))
        _git(self.work, "checkout", "-q", "-b", "ouroboros")

    def push(self, *refspecs):
        _git(self.work, "push", "-q", str(self.origin), *refspecs)

    def read(self, rel):
        return (self.work / rel).read_text(encoding="utf-8")

    def set_version(self, version, *, row=False):
        """What the commit path does before admission: VERSION, the deliberate row, the sync, the index."""
        (self.work / "VERSION").write_text(f"{version}\n", encoding="utf-8", newline="\n")
        if row:
            readme = self.read("README.md")
            first = locate_carrier_span(readme, carrier_spans_for("README.md")[1])[1][0]
            (self.work / "README.md").write_text(
                f"{readme[:first]}| {version} | 2026-11-03 | The release. |\n{readme[first:]}",
                encoding="utf-8", newline="\n")
        sync_release_metadata(str(self.work))
        _git(self.work, "add", "-A")

    def admit(self, **kwargs):
        return admission.release_metadata_diagnostics(self.work, source="index", **kwargs)

    def land(self, message):
        """The reviewed commit path's Git half: bind, commit, tag what the binding names, verify."""
        fingerprint = _fingerprint_staged_diff(self.work)
        assert fingerprint["ok"], fingerprint
        assert _review_binding_precondition_error(fingerprint) == ""
        _git(self.work, "commit", "-qm", message)
        sha = _git(self.work, "rev-parse", "HEAD")
        binding = fingerprint["binding"]
        tagged = _auto_tag_on_version_bump(self.work, message, expected_commit_sha=sha,
                                           expected_tag=binding["expected_tag"])
        assert _verify_reviewed_commit_binding(self.work, sha, fingerprint, verify_expected_tag=True) == (True, "")
        return types.SimpleNamespace(sha=sha, binding=binding, tagged=tagged)

    def neutral_landing(self, name):
        (self.work / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8", newline="\n")
        _git(self.work, "add", "-A")
        report = self.admit()
        assert (report["form"], report["status"], report["findings"]) == ("neutral", "not_applicable", [])
        return self.land(f"landing {name}")

    def public_links(self):
        readme = self.read("README.md")
        span = next(item for item in carrier_spans_for("README.md") if item.carrier_id == "readme_download_refs")
        start, end = locate_carrier_span(readme, span)[1]
        return readme[start:end], self.read("site/install/index.html"), self.read("docs/install/index.html")

    def history(self):
        readme = self.read("README.md")
        span = next(item for item in carrier_spans_for("README.md") if item.carrier_id == "readme_history")
        start, end = locate_carrier_span(readme, span)[1]
        return readme[start:end]

    def tags(self):
        return _git(self.work, "tag", "--list").split()

    def refs(self):
        return {name: _git(self.origin, "rev-parse", f"refs/heads/{name}")
                for name in ("main", "ouroboros-stable", "ouroboros")}

    def promote(self, tag, **kwargs):
        def facts(name, _commit):
            return {"version": name[1:], "sha256": hashlib.sha256(name.encode()).hexdigest()}
        return promotion.promote(tag, cwd=self.publisher, read_release=facts, **kwargs)

    def stable(self):
        return promotion.stable_selection("origin", self.publisher)

    def startup_check(self):
        env = types.SimpleNamespace(repo_dir=self.work, repo_path=lambda rel: self.work / rel)
        return startup.check_version_sync(env)


@pytest.fixture()
def cycle(tmp_path, monkeypatch):
    # Git guesses no identity here, as on a runner whose account has no name: a tag must find the checkout's own.
    guess_free = tmp_path / "guess-free.gitconfig"
    guess_free.write_text("[user]\n\tuseConfigOnly = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(guess_free))
    return Cycle(tmp_path)


def test_opening_a_cycle_projects_every_carrier_but_the_links_and_creates_no_release(cycle):
    links, history = cycle.public_links(), cycle.history()
    cycle.set_version("7.10.0-alpha.0")

    for source in ("index", "worktree"):
        report = admission.release_metadata_diagnostics(cycle.work, source=source, neutral_allowed=True)
        assert (report["form"], report["status"], report["findings"]) == ("development_marker", "clean", []), source
    landed = cycle.land("open the 7.10.0 cycle")

    assert landed.binding["version_staged"] and landed.binding["staged_version"] == "7.10.0-alpha.0"
    assert landed.binding["expected_tag"] == "" and landed.tagged == "" and cycle.tags() == ["v7.9.0"]
    # The tag writer's own fallback, reading VERSION from the commit, agrees.
    assert _auto_tag_on_version_bump(cycle.work, "open", expected_commit_sha=landed.sha) == ""
    assert cycle.tags() == ["v7.9.0"]

    assert cycle.public_links() == links and cycle.history() == history
    assert published_download_version(*links) == "7.9.0"
    assert 'version = "7.10.0a0"' in cycle.read("pyproject.toml")
    assert 'name = "ouroboros"\nversion = "7.10.0a0"\n' in cycle.read("uv.lock")
    assert '"version": "7.10.0-alpha.0"' in cycle.read("web/package.json")
    assert cycle.read("web/package-lock.json").count('"version": "7.10.0-alpha.0"') == 2
    assert "GATEWAY_CONTRACT_VERSION = '7.10.0-alpha.0'" in cycle.read("web/modules/api_types.js")
    assert "[![Version 7.10.0-alpha.0](https://img.shields.io/badge/version-7.10.0--alpha.0-green.svg)]" \
        in cycle.read("README.md")
    assert cycle.read("docs/ARCHITECTURE.md").startswith("# Ouroboros v7.10.0-alpha.0 ")

    result, issues = cycle.startup_check()
    assert (issues, result["status"], result["tag_sync"]) == (0, "ok", "development_marker")


@pytest.mark.parametrize("damage,expected", [
    (lambda c: c.set_version("7.10.0-alpha.0", row=True), "never takes a README Version History row"),
    (lambda c: c.set_version("7.9.0-alpha.0"), "expected one published plain release below 7.9.0-alpha.0"),
])
def test_a_marker_refuses_a_history_row_and_a_base_it_has_not_outgrown(cycle, damage, expected):
    damage(cycle)
    report = cycle.admit()
    assert report["form"] == "development_marker" and report["status"] == "blocked"
    assert expected in "\n".join(report["findings"])


@pytest.mark.parametrize("source,allowed,admitted", [
    ("index", None, True),  # the prepared index lane
    ("worktree", True, True),  # a bound candidate
    ("index", False, False),  # a commit on the serving checkout
    ("worktree", False, False),
    ("worktree", None, False),
])
def test_a_marker_is_admitted_only_where_the_neutral_form_is(cycle, source, allowed, admitted):
    cycle.set_version("7.10.0-alpha.0")
    report = admission.release_metadata_diagnostics(cycle.work, source=source, neutral_allowed=allowed)
    error = admission.format_release_metadata_preflight(report)
    if admitted:
        assert (report["form"], report["status"], report["findings"]) == ("development_marker", "clean", [])
        assert error is None
    else:
        assert (report["form"], report["status"]) == ("numbered", "blocked")
        assert len(report["findings"]) == 1 and "development marker 7.10.0-alpha.0" in report["findings"][0]
        assert "PREFLIGHT_BLOCKED" in error


@pytest.mark.parametrize("version", ["7.10.0", "7.10.0-alpha.1"])
def test_a_serving_commit_keeps_its_numbered_forms(cycle, version):
    cycle.set_version(version, row=True)
    report = admission.release_metadata_diagnostics(cycle.work, source="index", neutral_allowed=False)
    assert (report["form"], report["status"], report["findings"]) == ("numbered", "clean", [])


def test_a_landing_cannot_move_a_carrier_under_the_marker(cycle):
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    readme = cycle.read("README.md")
    (cycle.work / "README.md").write_text(readme.replace("v7.9.0/", "v7.9.1/", 1), encoding="utf-8", newline="\n")
    _git(cycle.work, "add", "-A")
    report = cycle.admit()
    assert (report["form"], report["status"]) == ("neutral", "blocked")
    assert "readme_download_refs" in report["findings"][0]


def test_one_whole_cycle_from_marker_through_promotion_to_the_next_marker(cycle):
    # The marker, and landings that neither bump nor tag while other authors keep going.
    cycle.set_version("7.10.0-alpha.0")
    opened = cycle.land("open the 7.10.0 cycle")
    links = cycle.public_links()
    for name in ("first", "second"):
        landed = cycle.neutral_landing(name)
        assert landed.binding["expected_tag"] == "" and not landed.binding["version_staged"]
    candidate = landed.sha
    assert cycle.tags() == ["v7.9.0"] and cycle.public_links() == links
    cycle.push("HEAD:refs/heads/ouroboros")

    # The selected candidate becomes the plain release beside development, which moved on.
    _git(cycle.work, "checkout", "-q", "-b", "release", candidate)
    cycle.set_version("7.10.0", row=True)
    report = cycle.admit()
    assert (report["form"], report["status"], report["findings"]) == ("numbered", "clean", [])
    release = cycle.land("release 7.10.0")
    assert release.binding["expected_tag"] == "v7.10.0" and release.tagged == " [tagged: v7.10.0]"
    assert _git(cycle.work, "cat-file", "-t", "refs/tags/v7.10.0") == "tag"
    assert _git(cycle.work, "rev-parse", "v7.10.0^{commit}") == release.sha
    assert published_download_version(*cycle.public_links()) == "7.10.0"
    _git(cycle.work, "checkout", "-q", "ouroboros")
    cycle.neutral_landing("third")
    cycle.push("HEAD:refs/heads/ouroboros")
    assert published_download_version(*cycle.public_links()) == "7.9.0"  # nothing is cut on this line yet

    # The cut reaches ouroboros first, then its tag goes out. For this one build window only development
    # links to the unpublished release: the branches behind the public surfaces have not moved.
    before = cycle.refs()
    _git(cycle.work, "merge", "-q", "--no-edit", "v7.10.0")
    assert cycle.read("VERSION").strip() == "7.10.0"
    assert _git(cycle.work, "log", "-1", "--format=%P").count(" ") == 1
    cycle.push("HEAD:refs/heads/ouroboros")
    cycle.push("refs/tags/v7.10.0")
    assert published_download_version(*cycle.public_links()) == "7.10.0"
    for branch in ("main", "ouroboros-stable"):
        assert "v7.9.0/" in _git(cycle.origin, "show", f"refs/heads/{branch}:README.md")
        assert "v7.10.0/" not in _git(cycle.origin, "show", f"refs/heads/{branch}:README.md")

    # Every way to forget the next cycle refuses and writes nothing.
    with pytest.raises(ValueError, match="7.10.0 is not a development marker"):
        cycle.promote("v7.10.0", push=True)
    assert {name: sha for name, sha in cycle.refs().items() if name != "ouroboros"} == \
        {name: sha for name, sha in before.items() if name != "ouroboros"}

    # A same-base marker is a different string, not a new cycle: admission and promotion both say so.
    cycle.set_version("7.10.0-alpha.0")
    assert cycle.admit()["status"] == "blocked"
    _git(cycle.work, "commit", "-qm", "the same base")
    cycle.push("HEAD:refs/heads/ouroboros")
    assert cycle.startup_check()[1] == 1
    with pytest.raises(ValueError, match="7.10.0-alpha.0 does not sort above the release 7.10.0"):
        cycle.promote("v7.10.0", push=True)
    assert cycle.refs()["main"] == before["main"] and cycle.refs()["ouroboros-stable"] == before["ouroboros-stable"]

    # The next greater marker keeps the release's links and row; promotion now succeeds.
    released = cycle.history(), cycle.public_links()
    cycle.set_version("7.11.0-alpha.0")
    report = cycle.admit()
    assert (report["form"], report["status"]) == ("development_marker", "clean")
    next_cycle = cycle.land("open the 7.11.0 cycle")
    assert next_cycle.binding["expected_tag"] == "" and "v7.11.0-alpha.0" not in cycle.tags()
    assert (cycle.history(), cycle.public_links()) == released
    cycle.push("HEAD:refs/heads/ouroboros")
    assert cycle.startup_check()[1] == 0

    development = cycle.refs()["ouroboros"]
    assert cycle.promote("v7.10.0", push=True) == 0
    refs = cycle.refs()
    assert refs["ouroboros"] == development == next_cycle.sha  # promotion never writes development
    assert refs["ouroboros-stable"] == release.sha and cycle.stable() == "v7.10.0"
    assert _git(cycle.origin, "diff", "--name-status", "v7.10.0", refs["main"]) == f"A\t{promotion.CASK_PATH}"
    assert _git(cycle.origin, "show", f"{refs['main']}:VERSION") == "7.10.0"  # main is the release, never the marker
    assert _git(cycle.origin, "show", f"{refs['ouroboros']}:VERSION") == "7.11.0-alpha.0"
    assert opened.sha != candidate
    assert cycle.promote("v7.10.0", push=True) == 0 and cycle.refs() == refs  # repeating it changes nothing


def _cut_beside(cycle, candidate="HEAD", *, version="7.10.0", branch="release"):
    """The plain cut on a side branch of the integrator's clone; its tag stays local until it is pushed."""
    _git(cycle.work, "checkout", "-q", "-b", branch, candidate)
    cycle.set_version(version, row=True)
    return cycle.land(f"release {version}")


def _cut_repair(cycle, release_sha):
    """The repair: plain 7.10.1 with its fix, cut from the 7.10.0 release commit and tagged locally."""
    _git(cycle.work, "checkout", "-q", "-b", "repair", release_sha)
    (cycle.work / "fix.txt").write_text("the fix\n", encoding="utf-8", newline="\n")
    cycle.set_version("7.10.1", row=True)
    repair = cycle.land("repair 7.10.1")
    assert repair.tagged == " [tagged: v7.10.1]"
    return repair


def _merge_cut_under_marker(cycle, tag, marker):
    """The merge's resolution through the operator's own helper: the release's links and row, the marker again."""
    assert _git(cycle.work, "merge", "--no-commit", "--no-ff", tag, check=False).returncode
    conflicted = _git(cycle.work, "diff", "--name-only", "--diff-filter=U").split()
    assert {"VERSION", "README.md", "pyproject.toml", "uv.lock"} <= set(conflicted)
    helper = subprocess.run([sys.executable, str(ROOT / "scripts" / "carrier_rebase_helper.py"),
                             "--worktree", str(cycle.work), "--prefer", "theirs"], capture_output=True, text=True)
    assert helper.returncode == 0, helper.stdout + helper.stderr
    assert _git(cycle.work, "diff", "--name-only", "--diff-filter=U") == ""
    cycle.set_version(marker)
    report = cycle.admit()
    assert (report["form"], report["status"], report["findings"]) == ("development_marker", "clean", [])
    return cycle.land(f"bring {tag} into development")


def test_a_next_marker_already_open_resolves_the_cut_merge_and_keeps_its_links_and_row(cycle):
    """The marker moved first, so the cut comes in as a merge whose resolution writes the marker back."""
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    candidate = cycle.neutral_landing("first").sha
    _cut_beside(cycle, candidate)
    _git(cycle.work, "checkout", "-q", "ouroboros")
    cycle.set_version("7.11.0-alpha.0")
    cycle.land("open the 7.11.0 cycle")
    assert published_download_version(*cycle.public_links()) == "7.9.0"
    cycle.push("HEAD:refs/heads/ouroboros")

    merged = _merge_cut_under_marker(cycle, "v7.10.0", "7.11.0-alpha.0")
    assert merged.binding["expected_tag"] == "" and len(merged.binding["parents"]) == 2
    assert published_download_version(*cycle.public_links()) == "7.10.0" and "| 7.10.0 |" in cycle.history()
    cycle.push("HEAD:refs/heads/ouroboros")
    cycle.push("refs/tags/v7.10.0")
    assert cycle.promote("v7.10.0", push=True) == 0 and cycle.stable() == "v7.10.0"
    assert _git(cycle.origin, "show", "refs/heads/ouroboros:VERSION") == "7.11.0-alpha.0"


def test_a_repair_of_the_published_release_and_the_next_release_both_lead_on(cycle):
    """The next cycle opened at a greater minor, so 7.10.1 stays free: cut from the release, merged like any cut."""
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    release = _cut_beside(cycle)
    _git(cycle.work, "checkout", "-q", "ouroboros")
    _git(cycle.work, "merge", "-q", "--no-edit", "v7.10.0")
    cycle.set_version("7.11.0-alpha.0")
    cycle.land("open the 7.11.0 cycle")
    cycle.push("HEAD:refs/heads/ouroboros")
    cycle.push("refs/tags/v7.10.0")
    assert cycle.promote("v7.10.0", push=True) == 0

    # The repair is a plain release cut from the release, so it reaches QA; development's marker never moved.
    repair = _cut_repair(cycle, release.sha)
    _git(cycle.work, "checkout", "-q", "ouroboros")
    assert cycle.read("VERSION").strip() == "7.11.0-alpha.0"
    assert _git(cycle.work, "merge", "--no-commit", "--no-ff", "v7.10.1", check=False).returncode
    conflicted = _git(cycle.work, "diff", "--name-only", "--diff-filter=U").split()
    assert resolve_carrier_conflicts(str(cycle.work), conflicted, prefer="theirs")["kept"] == {}
    cycle.set_version("7.11.0-alpha.0")
    report = cycle.admit()
    assert (report["form"], report["status"], report["findings"]) == ("development_marker", "clean", [])
    cycle.land("bring the repair into development")
    cycle.push("HEAD:refs/heads/ouroboros")
    cycle.push("refs/tags/v7.10.1")
    assert cycle.promote("v7.10.1", push=True) == 0
    assert cycle.refs()["ouroboros-stable"] == repair.sha and cycle.stable() == "v7.10.1"

    # The next minor leads on from both. Cut from a line that dropped them, QA's own refusal still stands.
    _git(cycle.work, "checkout", "-q", "-b", "stray", release.sha)
    cycle.set_version("7.11.0", row=True)
    stray = cycle.land("a release that skipped the repair").sha
    _git(cycle.work, "tag", "-d", "v7.11.0")
    cycle.push("HEAD:refs/heads/stray")
    refs = promotion.fetch_official("origin", cycle.publisher)
    _git(cycle.publisher, "fetch", "-q", "origin", "stray")
    with pytest.raises(ValueError, match="does not lead to v7.11.0"):
        promotion.plan_promotion("v7.11.0", stray, promotion.render_cask("7.11.0", "0" * 64),
                                 refs[promotion.STABLE], refs[promotion.QA], cycle.publisher)

    _git(cycle.work, "checkout", "-q", "ouroboros")
    cycle.set_version("7.11.0", row=True)
    release2 = cycle.land("release 7.11.0")
    cycle.push("HEAD:refs/heads/ouroboros", "refs/tags/v7.11.0")
    cycle.set_version("7.12.0-alpha.0")
    cycle.land("open the 7.12.0 cycle")
    cycle.push("HEAD:refs/heads/ouroboros")
    assert cycle.promote("v7.11.0", push=True) == 0
    assert cycle.refs()["ouroboros-stable"] == release2.sha and cycle.stable() == "v7.11.0"


def test_promotion_refuses_a_merge_that_dropped_the_release_links(cycle):
    """Keeping development's side of the carrier spans holds the ancestry, but not the links."""
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    _git(cycle.work, "checkout", "-q", "-b", "release")
    cycle.set_version("7.10.0", row=True)
    cycle.land("release 7.10.0")
    _git(cycle.work, "checkout", "-q", "ouroboros")
    cycle.set_version("7.11.0-alpha.0")
    cycle.land("open the next cycle")
    _git(cycle.work, "merge", "--no-commit", "--no-ff", "v7.10.0", check=False)
    conflicted = _git(cycle.work, "diff", "--name-only", "--diff-filter=U").split()
    assert resolve_carrier_conflicts(str(cycle.work), conflicted, prefer="ours")["kept"] == {}
    _git(cycle.work, "add", "-A")
    _git(cycle.work, "commit", "-qm", "merge keeping development's spans")
    links = cycle.public_links()  # the pages merged cleanly to the release; README's spans stayed behind
    assert (published_download_version(links[0]), published_download_version(links[1])) == ("7.9.0", "7.10.0")
    assert cycle.admit()["form"] == "none"
    cycle.push("HEAD:refs/heads/ouroboros")
    cycle.push("refs/tags/v7.10.0")
    before = cycle.refs()
    with pytest.raises(ValueError, match="download links still name 7.9.0: move them to the published v7.10.0"):
        cycle.promote("v7.10.0", push=True)
    assert cycle.refs() == before

    # The explicit repair: the existing projection, told which release was really published, plus the deliberate row.
    assert sync_release_metadata(str(cycle.work), published_version="7.10.0") == ["README.md"]
    readme = cycle.read("README.md")
    (cycle.work / "README.md").write_text(
        readme.replace("| 7.9.0 |", "| 7.10.0 | 2026-11-03 | The release. |\n| 7.9.0 |", 1), encoding="utf-8", newline="\n")
    _git(cycle.work, "add", "-A")
    report = cycle.admit()
    assert (report["form"], report["status"], report["findings"]) == ("development_marker", "clean", [])
    repaired = cycle.land("the published links and row")
    assert repaired.binding["expected_tag"] == "" and "v7.11.0-alpha.0" not in cycle.tags()
    cycle.push("HEAD:refs/heads/ouroboros")
    assert cycle.promote("v7.10.0", push=True) == 0


def _publish_under_marker(cycle, *, links_to="7.9.1", row=True, pages=True):
    """A commit under a marker that moves README's links, optionally its row and the install pages."""
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    readme = cycle.read("README.md").replace(_download_references("7.9.0"), _download_references(links_to))
    assert links_to == "7.9.0" or readme != cycle.read("README.md")
    if row:
        readme = readme.replace("| 7.9.0 |", f"| {links_to} | 2026-11-03 | The release. |\n| 7.9.0 |", 1)
    (cycle.work / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    if pages:
        for rel in ("site/install/index.html", "docs/install/index.html"):
            (cycle.work / rel).write_text(_download_page(links_to), encoding="utf-8", newline="\n")
    _git(cycle.work, "add", "-A")
    return cycle.admit()


def test_only_a_whole_forward_move_of_the_published_release_passes_without_a_version_change(cycle, tmp_path):
    report = _publish_under_marker(cycle)
    assert (report["form"], report["status"], report["findings"]) == ("development_marker", "clean", [])


@pytest.mark.parametrize("kwargs,expected", [
    (dict(links_to="7.8.0", row=False), "readme_download_refs"),  # backward
    (dict(links_to="7.10.0"), "expected one published plain release below 7.10.0-alpha.0"),  # the cycle's own base
    (dict(pages=False), "expected one published plain release below 7.10.0-alpha.0"),  # the pages stay behind
])
def test_any_partial_or_backward_move_of_the_published_release_is_refused(cycle, kwargs, expected):
    report = _publish_under_marker(cycle, **kwargs)
    assert report["status"] == "blocked"
    assert expected in "\n".join(report["findings"])


def test_outside_a_marker_a_move_of_the_links_stays_a_neutral_refusal(cycle):
    readme = cycle.read("README.md").replace(_download_references("7.9.0"), _download_references("7.9.1"))
    (cycle.work / "README.md").write_text(readme.replace("| 7.9.0 |", "| 7.9.1 | 2026-11-03 | x |\n| 7.9.0 |", 1),
                                          encoding="utf-8", newline="\n")
    _git(cycle.work, "add", "-A")
    report = cycle.admit()
    assert (report["form"], report["status"]) == ("neutral", "blocked")
    assert "readme_download_refs" in report["findings"][0]


def test_a_badge_moved_with_the_links_is_not_a_publication(cycle):
    _publish_under_marker(cycle)
    readme = cycle.read("README.md")
    (cycle.work / "README.md").write_text(readme.replace("Version 7.10.0-alpha.0", "Version 7.10.1-alpha.0"),
                                          encoding="utf-8", newline="\n")
    _git(cycle.work, "add", "-A")
    report = cycle.admit()
    assert (report["form"], report["status"]) == ("neutral", "blocked")
    assert "readme_badge" in report["findings"][0]


def test_an_accidental_marker_tag_neither_blocks_the_marker_nor_counts_as_a_release(cycle):
    _git(cycle.work, "tag", "-a", "v7.10.0-alpha.0", "-m", "an accident")
    cycle.set_version("7.10.0-alpha.0")
    fingerprint = _fingerprint_staged_diff(cycle.work)
    assert fingerprint["binding"]["existing_tag_target"] == ""
    assert _review_binding_precondition_error(fingerprint) == ""
    landed = cycle.land("open")
    assert landed.tagged == ""
    assert _git(cycle.work, "rev-parse", "v7.10.0-alpha.0^{commit}") != landed.sha  # never retargeted
    result, issues = cycle.startup_check()
    assert issues == 0 and result["tag_sync"] == "development_marker"


def test_a_marker_is_never_packaged_by_any_build_script(tmp_path):
    bundle = runpy.run_path(str(ROOT / "scripts" / "build_repo_bundle.py"))
    (tmp_path / "VERSION").write_text("7.10.0-alpha.0\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="development-cycle marker"):
        bundle["_read_version"](tmp_path)
    for version in ("7.10.0", "7.10.0-alpha.1", "7.10.0-rc.1"):
        (tmp_path / "VERSION").write_text(f"{version}\n", encoding="utf-8")
        assert bundle["_read_version"](tmp_path) == version


def _ci_checkout(cycle, tag):
    """What a tag-push build starts from: every official branch as origin/*, the tag object, HEAD detached at it."""
    checkout = cycle.root / f"ci-{tag}-{len(list(cycle.root.glob('ci-*')))}"
    _git(cycle.root, "clone", "-q", str(cycle.origin), str(checkout))
    _git(checkout, "checkout", "-q", "--detach", f"refs/tags/{tag}")
    return checkout


def _ci_tag_source_run(checkout, tag):
    """CI's real step, run from ci.yml as written; returns the result and the branch it exports."""
    jobs = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))["jobs"]
    scripts = {step["run"] for job in jobs.values() for step in job.get("steps", [])
               if step.get("name") == "Resolve the managed source branch of this tag"}
    assert len(scripts) == 1
    output = checkout.parent / f"{checkout.name}.env"
    output.write_text("", encoding="utf-8")
    result = subprocess.run([shutil.which("bash"), "-c", scripts.pop()], cwd=checkout, capture_output=True, text=True,
                            env={**os.environ, "GITHUB_REF_NAME": tag, "GITHUB_ENV": str(output)})
    exported = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    return result, exported.get("OUROBOROS_MANAGED_SOURCE_BRANCH", "")


def _bundle_manifest(checkout, tag, branch, monkeypatch):
    """The real bundle build at that checkout; the manifest is what a packaged install reads."""
    # A tag-push runner sets these for its own tag; the build would read it as a second tag
    # (tests/test_launcher_sync.py _BUILD_BUNDLE_ENV_SCRUB_KEYS).
    for key in ("GITHUB_REF_TYPE", "GITHUB_REF", "GITHUB_REF_NAME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OUROBOROS_RELEASE_TAG", tag)
    bundle = runpy.run_path(str(ROOT / "scripts" / "build_repo_bundle.py"))
    manifest = checkout.parent / f"{checkout.name}.json"
    bundle["build_bundle"](checkout, checkout.parent / f"{checkout.name}.bundle", manifest, source_branch=branch,
                           local_branch="ouroboros", local_stable_branch="ouroboros-stable",
                           remote_stable_branch="ouroboros-stable", managed_remote_name="managed")
    return json.loads(manifest.read_text(encoding="utf-8"))


def _assert_ci_builds_for_ouroboros(cycle, tag, sha, monkeypatch):
    """The tag-push build: CI's step names `ouroboros`, the bundle packs the tag's commit, installs track it."""
    checkout = _ci_checkout(cycle, tag)
    assert _git(checkout, "rev-parse", "HEAD") == sha
    result, branch = _ci_tag_source_run(checkout, tag)
    assert (result.returncode, branch) == (0, "ouroboros"), result.stderr
    manifest = _bundle_manifest(checkout, tag, branch, monkeypatch)
    assert (manifest["source_sha"], manifest["app_version"]) == (sha, tag[1:])
    assert git_ops._managed_remote_branch_for(git_ops.BRANCH_DEV, manifest) == "ouroboros"


def _cut_on_the_tip(cycle):
    cycle.set_version("7.10.0", row=True)
    return cycle.land("release 7.10.0").sha, "7.10.0"


def _cut_beside_a_moved_tip(cycle):
    candidate = _git(cycle.work, "rev-parse", "HEAD")
    release = _cut_beside(cycle, candidate)
    _git(cycle.work, "checkout", "-q", "ouroboros")
    cycle.neutral_landing("moved")
    cycle.push("HEAD:refs/heads/ouroboros")
    _git(cycle.work, "merge", "-q", "--no-edit", "v7.10.0")
    return release.sha, "7.10.0"


def _cut_merged_under_the_next_marker(cycle):
    candidate = _git(cycle.work, "rev-parse", "HEAD")
    release = _cut_beside(cycle, candidate)
    _git(cycle.work, "checkout", "-q", "ouroboros")
    cycle.set_version("7.11.0-alpha.0")
    cycle.land("open the 7.11.0 cycle")
    _merge_cut_under_marker(cycle, "v7.10.0", "7.11.0-alpha.0")
    return release.sha, "7.11.0-alpha.0"


@pytest.mark.skipif(os.name == "nt", reason="the step is a Bash script on the runners")
@pytest.mark.parametrize("cut", [_cut_on_the_tip, _cut_beside_a_moved_tip, _cut_merged_under_the_next_marker])
def test_every_documented_cut_order_builds_and_installs_track_ouroboros(cycle, monkeypatch, cut):
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    cycle.push("HEAD:refs/heads/ouroboros")
    cut_sha, development_version = cut(cycle)
    assert cycle.read("VERSION").strip() == development_version
    assert published_download_version(*cycle.public_links()) == "7.10.0"
    cycle.push("HEAD:refs/heads/ouroboros")
    cycle.push("refs/tags/v7.10.0")
    _assert_ci_builds_for_ouroboros(cycle, "v7.10.0", cut_sha, monkeypatch)


@pytest.mark.skipif(os.name == "nt", reason="the step is a Bash script on the runners")
def test_a_repair_cut_from_the_release_and_merged_in_builds_and_installs_track_ouroboros(cycle, monkeypatch):
    """CONTRIBUTING's repair: 7.10.1 cut from the 7.10.0 commit, merged under 7.11.0-alpha.0, branch before tag."""
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    cycle.push("HEAD:refs/heads/ouroboros")
    release = _cut_beside(cycle)
    _git(cycle.work, "checkout", "-q", "ouroboros")
    cycle.set_version("7.11.0-alpha.0")
    cycle.land("open the 7.11.0 cycle")
    _merge_cut_under_marker(cycle, "v7.10.0", "7.11.0-alpha.0")
    cycle.push("HEAD:refs/heads/ouroboros")
    cycle.push("refs/tags/v7.10.0")

    repair = _cut_repair(cycle, release.sha)
    _git(cycle.work, "checkout", "-q", "ouroboros")
    _merge_cut_under_marker(cycle, "v7.10.1", "7.11.0-alpha.0")
    assert cycle.read("VERSION").strip() == "7.11.0-alpha.0"
    assert published_download_version(*cycle.public_links()) == "7.10.1" and "| 7.10.1 |" in cycle.history()
    cycle.push("HEAD:refs/heads/ouroboros")
    cycle.push("refs/tags/v7.10.1")
    _assert_ci_builds_for_ouroboros(cycle, "v7.10.1", repair.sha, monkeypatch)


def _tag_no_branch_contains(cycle):
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    cycle.push("HEAD:refs/heads/ouroboros")
    cut = _cut_beside(cycle)
    cycle.push("refs/tags/v7.10.0")
    return cut


def test_a_tag_no_remote_branch_contains_is_refused_by_the_bundle_build_and_by_promotion(cycle, monkeypatch):
    cut = _tag_no_branch_contains(cycle)
    checkout = _ci_checkout(cycle, "v7.10.0")
    bundle = runpy.run_path(str(ROOT / "scripts" / "build_repo_bundle.py"))
    with pytest.raises(SystemExit, match="is not reachable from the configured managed source branch 'ouroboros'"):
        bundle["_ensure_source_sha_tracks_branch"](checkout, cut.sha, "ouroboros")
    with pytest.raises(SystemExit, match="is not reachable from the configured managed source branch 'ouroboros'"):
        _bundle_manifest(checkout, "v7.10.0", "ouroboros", monkeypatch)
    with pytest.raises(ValueError, match="does not contain v7.10.0"):
        cycle.promote("v7.10.0", push=True)

    # The same tag becomes buildable once the cut lands on the branch, the order CONTRIBUTING documents.
    _git(cycle.work, "checkout", "-q", "ouroboros")
    _git(cycle.work, "merge", "-q", "--no-edit", "v7.10.0")
    cycle.push("HEAD:refs/heads/ouroboros")
    checkout = _ci_checkout(cycle, "v7.10.0")
    bundle["_ensure_source_sha_tracks_branch"](checkout, cut.sha, "ouroboros")
    assert _bundle_manifest(checkout, "v7.10.0", "ouroboros", monkeypatch)["managed_remote_branch"] == "ouroboros"


@pytest.mark.skipif(os.name == "nt", reason="the step is a Bash script on the runners")
def test_ci_refuses_a_tag_no_remote_branch_contains_and_builds_it_once_ouroboros_does(cycle):
    _tag_no_branch_contains(cycle)
    result, branch = _ci_tag_source_run(_ci_checkout(cycle, "v7.10.0"), "v7.10.0")
    assert (result.returncode, branch) == (1, "")
    assert "expected exactly one remote branch containing HEAD, got: none" in result.stderr

    _git(cycle.work, "checkout", "-q", "ouroboros")
    _git(cycle.work, "merge", "-q", "--no-edit", "v7.10.0")
    cycle.push("HEAD:refs/heads/ouroboros")
    result, branch = _ci_tag_source_run(_ci_checkout(cycle, "v7.10.0"), "v7.10.0")
    assert (result.returncode, branch) == (0, "ouroboros"), result.stderr


@pytest.mark.skipif(os.name == "nt", reason="the step is a Bash script on the runners")
def test_a_cut_on_its_own_remote_branch_would_become_the_branch_installs_track(cycle, monkeypatch):
    """Why the cut lands on ouroboros: the build accepts this, and the manifest then misdirects Development installs."""
    cycle.set_version("7.10.0-alpha.0")
    cycle.land("open")
    cycle.push("HEAD:refs/heads/ouroboros")
    _cut_beside(cycle)
    cycle.push("HEAD:refs/heads/release", "refs/tags/v7.10.0")

    checkout = _ci_checkout(cycle, "v7.10.0")
    result, branch = _ci_tag_source_run(checkout, "v7.10.0")
    assert (result.returncode, branch) == (0, "release"), result.stderr
    manifest = _bundle_manifest(checkout, "v7.10.0", branch, monkeypatch)
    assert git_ops._managed_remote_branch_for(git_ops.BRANCH_DEV, manifest) == "release"

    # Two such branches leave no one source: refused.
    cycle.push("HEAD:refs/heads/release-copy")
    result, branch = _ci_tag_source_run(_ci_checkout(cycle, "v7.10.0"), "v7.10.0")
    assert (result.returncode, branch) == (1, "") and "expected exactly one remote branch" in result.stderr


def _release_meta_run(tmp_path, version, tag):
    """CI's real tag preflight step, on one VERSION and pushed tag; bash and a `python` on PATH, as on the runner."""
    jobs = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))["jobs"]
    script = next(step for step in jobs["release-preflight"]["steps"] if step.get("id") == "release_meta")["run"]
    (tmp_path / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    (binaries / "python").chmod(0o755)
    output = tmp_path / "output"
    env = {**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ.get('PATH', '')}", "PYTHONPATH": str(ROOT),
           "GITHUB_REF_NAME": tag, "GITHUB_OUTPUT": str(output)}
    result = subprocess.run([shutil.which("bash"), "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    return result, dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").split())


@pytest.mark.skipif(os.name == "nt", reason="the step is a Bash heredoc on Ubuntu runners")
@pytest.mark.parametrize("version,tag,code,valid,prerelease", [
    ("7.10.0-alpha.0", "v7.10.0-alpha.0", 1, "false", "false"),  # tag == VERSION, the one check it used to pass
    ("7.10.0", "v7.10.0", 0, "true", "false"),
    ("7.10.0-rc.1", "v7.10.0-rc.1", 0, "true", "true"),
])
def test_the_tag_preflight_never_passes_a_marker_to_the_builds(tmp_path, version, tag, code, valid, prerelease):
    result, outputs = _release_meta_run(tmp_path, version, tag)
    assert result.returncode == code, result.stderr
    assert outputs == {"tag_valid": valid, "is_prerelease": prerelease}
    if tag == f"v{version}" and code:
        assert "development-cycle marker" in result.stderr
