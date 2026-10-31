"""The archived joi-lab mirrors refresh (scripts/sync_legacy_mirrors.py).

Owner decisions 2026-10-11: №1=B refresh the archives on every stable release,
№2=A keep and mark the outdated desktop releases.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from scripts import sync_legacy_mirrors as mirrors

ROOT = Path(__file__).resolve().parents[1]


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo, files, message):
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def test_mirrors_are_only_the_two_archived_joi_lab_repositories():
    assert [m.repo for m in mirrors.MIRRORS] == ["joi-lab/ouroboros", "joi-lab/ouroboros-desktop"]
    # The desktop branch old apps update from, and the fork's two update branches.
    assert {m.repo: m.branches for m in mirrors.MIRRORS} == {
        "joi-lab/ouroboros": ("ouroboros", "ouroboros-stable"),
        "joi-lab/ouroboros-desktop": ("ouroboros",),
    }


def test_readme_and_pointer_send_visitors_to_the_official_repository():
    fork, desktop = mirrors.MIRRORS
    readme = mirrors.render_readme(desktop, "7.6.0", "4b37f96223a1")
    assert mirrors.OFFICIAL_URL in readme and mirrors.LATEST_URL in readme
    assert "v7.6.0 (official commit 4b37f9622)" in readme
    assert "refreshed automatically: `main`." in readme  # only the branches this run refreshed
    assert "`main`, `ouroboros`." in mirrors.render_readme(desktop, "7.6.0", "4b37f96223a1", ("ouroboros",))
    assert "historical snapshot" in mirrors.render_pointer(fork, "7.6.0")
    assert mirrors.LATEST_URL in mirrors.render_pointer(fork, "7.6.0")
    assert "outdated Ouroboros Desktop builds" in mirrors.render_pointer(desktop, "7.6.0")
    assert "outdated Ouroboros Desktop builds" not in mirrors.render_pointer(fork, "7.6.0")


def test_outdated_notice_is_prepended_once_and_keeps_the_notes():
    once = mirrors.mark_outdated("Original notes")
    assert once.startswith(mirrors.OUTDATED_MARK) and once.endswith("Original notes")
    assert mirrors.mark_outdated(once) == once
    assert mirrors.mark_outdated(None).startswith(mirrors.OUTDATED_MARK)


def test_colab_stub_is_a_notebook_that_points_at_the_official_quickstart():
    notebook = json.loads(mirrors.colab_stub())
    assert notebook["nbformat"] == 4
    text = json.dumps(notebook)
    assert "razzant/ouroboros/main/notebooks/colab_quickstart.py" in text
    assert "joi-lab" not in text


def test_main_commit_keeps_both_histories_and_is_idempotent(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    base = _commit(repo, {"VERSION": "6.7.3\n", "README.md": "old\n"}, "base")
    mirror_main = _commit(repo, {"README.md": "moved\n"}, "tombstone")
    _git(repo, "checkout", "-q", base)
    source = _commit(repo, {"VERSION": "7.6.0\n", "README.md": "official\n", "app.py": "x = 1\n"}, "stable")
    monkeypatch.chdir(repo)
    fork = mirrors.MIRRORS[0]

    new = mirrors.build_main_commit(fork, source, mirror_main, "7.6.0")

    assert _git(repo, "rev-parse", f"{new}^1") == mirror_main  # no rewrite: a fast-forward of the mirror
    assert _git(repo, "rev-parse", f"{new}^2") == source
    assert _git(repo, "show", f"{new}:app.py") == "x = 1"
    assert _git(repo, "show", f"{new}:README.md").startswith("# Ouroboros moved to razzant/ouroboros")
    assert json.loads(_git(repo, "show", f"{new}:notebooks/quickstart.ipynb"))["nbformat"] == 4
    assert mirrors.build_main_commit(fork, source, new, "7.6.0") is None
    desktop = mirrors.build_main_commit(mirrors.MIRRORS[1], source, mirror_main, "7.6.0")
    assert "notebooks/quickstart.ipynb" not in _git(repo, "ls-tree", "-r", "--name-only", desktop)


def _fake_github(monkeypatch, release, *, lookup_status=None):
    calls = []

    def call(self, method, path, body=None):
        calls.append((method, path))
        if path == "/user":
            return 200, {"login": "ouroboros-agent"}
        if path.startswith("/repos/razzant/ouroboros/releases/tags/"):
            if lookup_status:
                return lookup_status, {"message": "Service Unavailable"}
            return (200, release) if release is not None else (404, {"message": "Not Found"})
        if method == "GET":
            return 200, {"archived": False}
        return 200, {}

    monkeypatch.setattr(mirrors.GitHub, "call", call)
    monkeypatch.setenv("MIRROR_TOKEN", "test-token")
    return calls


def _tagged_repo(tmp_path, monkeypatch, *, tag):
    repo = tmp_path / "official"
    repo.mkdir()
    _git(repo, "init", "-q")
    head = _commit(repo, {"VERSION": "7.7.0\n"}, "release")
    if tag:
        _git(repo, "tag", tag)
    monkeypatch.chdir(repo)
    return head


def _mutations(calls):
    return [c for c in calls if c[0] != "GET"]


def test_a_commit_without_the_published_release_tag_changes_nothing(tmp_path, monkeypatch):
    head = _tagged_repo(tmp_path, monkeypatch, tag="")
    calls = _fake_github(monkeypatch, release={"draft": False, "prerelease": False})
    assert mirrors.main(["--source-ref", head]) == 0
    assert _mutations(calls) == []


def test_a_prerelease_changes_nothing(tmp_path, monkeypatch):
    head = _tagged_repo(tmp_path, monkeypatch, tag="v7.7.0")
    calls = _fake_github(monkeypatch, release={"draft": False, "prerelease": True})
    assert mirrors.main(["--source-ref", head]) == 0
    assert _mutations(calls) == []


def test_a_failed_release_check_is_an_error_not_a_quiet_skip(tmp_path, monkeypatch):
    head = _tagged_repo(tmp_path, monkeypatch, tag="v7.7.0")
    calls = _fake_github(monkeypatch, release=None, lookup_status=503)
    assert mirrors.main(["--source-ref", head]) == 1
    assert _mutations(calls) == []


def test_the_published_stable_release_passes_the_gate(tmp_path, monkeypatch):
    head = _tagged_repo(tmp_path, monkeypatch, tag="v7.7.0")
    calls = _fake_github(monkeypatch, release={"draft": False, "prerelease": False})
    # No mirror selected: the run stops right after the gate, without network or mutations.
    assert mirrors.main(["--source-ref", head, "--only", "joi-lab/none"]) == 0
    assert ("GET", "/user") in calls and _mutations(calls) == []


def test_main_promotion_commits_resolve_to_the_tag_they_project(tmp_path, monkeypatch):
    from scripts import promote_stable as promotion

    first = _tagged_repo(tmp_path, monkeypatch, tag="v7.7.0")  # a main that is still the raw tag commit
    repo = tmp_path / "official"
    p1 = promotion.plan_promotion("v7.7.0", first, promotion.render_cask("7.7.0", "a" * 64), first, first)["main"]
    assert mirrors.released_commit(p1) == first
    _git(repo, "checkout", "-q", first)
    second = _commit(repo, {"VERSION": "7.8.0\n"}, "release 7.8.0")
    _git(repo, "tag", "v7.8.0")
    p2 = promotion.plan_promotion("v7.8.0", second, promotion.render_cask("7.8.0", "b" * 64), p1, first)["main"]
    assert _git(repo, "rev-parse", f"{p2}^1", f"{p2}^2").split() == [p1, second]
    assert mirrors.released_commit(p2) == second
    p3 = promotion.plan_promotion("v7.8.0", second, promotion.render_cask("7.8.0", "c" * 64), p2, second)["main"]
    assert mirrors.released_commit(p3) == second  # a re-rendered cask on top reads through as well
    _git(repo, "checkout", "-q", p3)
    code = _commit(repo, {"app.py": "x = 1\n"}, "code")
    assert mirrors.released_commit(code) == code  # anything else is not skipped
    calls = _fake_github(monkeypatch, release={"draft": False, "prerelease": False})
    # A run on main's promotion commit refreshes the tagged release instead of reporting "not a release".
    assert mirrors.main(["--source-ref", p2, "--only", "joi-lab/none"]) == 0
    assert ("GET", "/user") in calls and _mutations(calls) == []


def test_the_refresh_ignores_pushes_that_only_rerender_the_cask():
    workflow = yaml.safe_load((ROOT / ".github/workflows/sync-legacy-mirrors.yml").read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))  # PyYAML's YAML 1.1 spelling of "on"
    assert triggers["push"] == {"branches": ["main"], "paths-ignore": ["Casks/**"]}


def test_archive_only_archives_both_mirrors(monkeypatch):
    calls = _fake_github(monkeypatch, release=None)
    assert mirrors.main(["--archive-only"]) == 0
    assert _mutations(calls) == [("PATCH", "/repos/joi-lab/ouroboros"), ("PATCH", "/repos/joi-lab/ouroboros-desktop")]
