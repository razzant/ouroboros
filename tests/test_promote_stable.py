"""Stable promotion and the Homebrew cask it puts on main (scripts/promote_stable.py).

Static checks and real Git promotion against local bare repositories. They prove
neither a native ``brew`` install nor GitHub's server: that is the separate macOS
consumer probe.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ouroboros.tools.release_sync import RELEASE_ASSET_TEMPLATES, release_asset_download_url
from scripts import promote_stable as promotion
from scripts import release_proof
from scripts import sync_legacy_mirrors as mirrors
from tests.test_release_sync import _download_page

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = b"notarized dmg bytes"
DMG_SHA = hashlib.sha256(PAYLOAD).hexdigest()
COMMIT = "6f3ddff05ff3e6c6f8d007ff96358df6af872e83"
# LSMinimumSystemVersion -> Homebrew's macOS symbol, for the floor the spec may name.
MACOS_SYMBOLS = {"12.0": ":monterey", "13.0": ":ventura", "14.0": ":sonoma", "15.0": ":sequoia"}
TAP = "brew tap razzant/ouroboros https://github.com/razzant/ouroboros.git"
INSTALL = "brew install --cask razzant/ouroboros/ouroboros"
CLI = "/Applications/Ouroboros.app/Contents/Resources/bin/install-ouroboros-cli"


def test_the_cask_names_the_app_and_platform_the_release_builds():
    spec = (ROOT / "Ouroboros.spec").read_text(encoding="utf-8")
    assert f"name='{promotion.APP_BUNDLE}'" in spec
    assert f"bundle_identifier='{promotion.BUNDLE_ID}'" in spec
    minimum = re.search(r"'LSMinimumSystemVersion': '([\d.]+)'", spec).group(1)
    assert promotion.MACOS_MINIMUM == (minimum, MACOS_SYMBOLS[minimum])
    # The one macOS installer is the arm64 DMG, and its receipt must prove arm64.
    assert [key for key in RELEASE_ASSET_TEMPLATES if key.startswith("macos")] == [promotion.PROOF_ID] == ["macos-arm64"]
    assert "arm64_main_executable" in release_proof.REQUIRED_SMOKE_CHECKS[promotion.PROOF_ID]
    cask = promotion.render_cask("7.8.0", DMG_SHA)
    url = re.search(r'^  url "([^"]+)"$', cask, re.MULTILINE).group(1).replace("#{version}", "7.8.0")
    assert url == release_asset_download_url("macos-arm64", "7.8.0")
    assert f"  depends_on macos: {MACOS_SYMBOLS[minimum]}\n" in cask
    # Current Homebrew deprecates `url ..., verified:` and the ">= :release" string form.
    assert "verified:" not in cask and '">=' not in cask


def test_the_cask_declares_a_self_updating_app_and_never_touches_owner_data():
    cask = promotion.render_cask("7.8.0", DMG_SHA)
    # auto_updates: the app's own bundle update (Homebrew then compares the bundle on a plain
    # `brew upgrade`). No zap/delete/trash (owner data in ~/Ouroboros or custom roots), no
    # launchctl (it also runs on upgrade and deletes ~/Library/LaunchAgents/com.ouroboros.agent.plist).
    stanzas = re.findall(r"^  ([a-z0-9_]+)\b", cask, re.MULTILINE)
    assert stanzas == ["version", "sha256", "url", "name", "desc", "homepage",
                       "auto_updates", "depends_on", "depends_on", "app", "uninstall", "caveats", "end"]
    assert "  auto_updates true\n" in cask
    assert re.findall(r"^  uninstall .*$", cask, re.MULTILINE) == ['  uninstall quit: "com.ouroboros.agent"']


def test_first_cask_exposes_the_known_setup_limitation_without_rewriting_the_release():
    cask = promotion.render_cask("7.8.0", DMG_SHA)
    assert "finish Setup before the first Quit" in cask
    assert "confirmed empty new installation" in cask
    assert "blob/ouroboros/README.md#homebrew-on-macos" in cask
    assert '#2412' not in promotion.render_cask("7.9.0", DMG_SHA)


@pytest.mark.parametrize("version,sha256", [("7.9.0-rc.1", DMG_SHA), ("7.9", DMG_SHA), ("7.8.0", "ABC")])
def test_render_refuses_a_prerelease_or_a_malformed_digest(version, sha256):
    with pytest.raises(ValueError, match="plain X.Y.Z"):
        promotion.render_cask(version, sha256)


def test_every_install_surface_shows_the_same_homebrew_commands():
    for path in ("README.md", "site/install/index.html", "docs/install/index.html"):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert TAP in text and INSTALL in text and CLI in text, path


# --------------------------------------------------------------------------- public facts

def _asset(name, data):
    return {"name": name, "size": len(data), "digest": f"sha256:{hashlib.sha256(data).hexdigest()}",
            "browser_download_url": f"https://github.com/razzant/ouroboros/releases/download/v7.8.0/{name}"}


def _world(*, release=None, evidence=None, record=None, receipt=None, dmg_sum=DMG_SHA, sums=None, drop=(),
           digest=None):
    """A published v7.8.0 whose proof files can be altered one at a time."""
    dmg = "Ouroboros-7.8.0.dmg"
    receipt_doc = {"artifact": dmg, "kind": "packaged_artifact_smoke", "notarization": {"status": "notarized"},
                   "proofId": "macos-arm64", "releaseTag": "v7.8.0", "schemaVersion": 1, "sha256": DMG_SHA,
                   "sourceCommit": COMMIT, "status": "passed",
                   "checks": sorted(release_proof.REQUIRED_SMOKE_CHECKS["macos-arm64"]
                                    | release_proof.NOTARIZATION_SMOKE_CHECKS)}
    receipt_doc.update(receipt or {})
    receipt_bytes = json.dumps(receipt_doc).encode()
    record_doc = {"name": dmg, "notarization": {"status": "notarized"}, "proofId": "macos-arm64",
                  "sbom": "sbom-macos-arm64.cdx.json", "sha256": DMG_SHA, "size": len(PAYLOAD),
                  "smokeReceipt": promotion.RECEIPT}
    record_doc.update(record or {})
    evidence_doc = {"kind": "build_time_release_proof", "version": "7.8.0", "artifacts": [record_doc],
                    "source": {"commit": COMMIT, "repository": "https://github.com/razzant/ouroboros",
                               "tag": "v7.8.0"}}
    evidence_doc.update(evidence or {})
    evidence_bytes = json.dumps(evidence_doc).encode()
    sums_bytes = (sums or f"{dmg_sum}  {dmg}\n{hashlib.sha256(receipt_bytes).hexdigest()}  {promotion.RECEIPT}\n"
                  ).encode()
    files = {dmg: PAYLOAD, "SHA256SUMS": sums_bytes, "release-evidence.json": evidence_bytes,
             promotion.RECEIPT: receipt_bytes}
    assets = [_asset(name, data) for name, data in files.items() if name not in drop]
    for row in assets:
        if digest and row["name"] in digest:
            row["digest"] = digest[row["name"]]
    release_doc = {"tag_name": "v7.8.0", "draft": False, "prerelease": False, "assets": assets}
    release_doc.update(release or {})
    pages = {f"{promotion.API}/releases/tags/v7.8.0": json.dumps(release_doc).encode(),
             **{row["browser_download_url"]: files[row["name"]] for row in assets}}
    return pages.__getitem__


def test_facts_bind_the_dmg_digest_across_every_published_proof():
    facts = promotion.release_facts("v7.8.0", COMMIT, get=_world(), download=lambda url: (DMG_SHA, len(PAYLOAD)))
    assert facts == {"tag": "v7.8.0", "version": "7.8.0", "commit": COMMIT,
                     "url": "https://github.com/razzant/ouroboros/releases/download/v7.8.0/Ouroboros-7.8.0.dmg",
                     "sha256": DMG_SHA, "size": len(PAYLOAD), "downloadVerified": True}


OTHER = "0" * 64
REFUSALS = {
    "checksum disagrees": (dict(dmg_sum=OTHER), "release-evidence.json and SHA256SUMS disagree"),
    "receipt outside SHA256SUMS": (dict(sums=f"{DMG_SHA}  Ouroboros-7.8.0.dmg\n"), "does not cover"),
    "evidence of an unnotarized DMG": (dict(record={"notarization": {"status": "not_notarized"}}),
                                       "is not notarized"),
    "evidence of another size": (dict(record={"size": 1}), "size differs"),
    "dmg missing": (dict(drop=("Ouroboros-7.8.0.dmg",)), "lacks Ouroboros-7.8.0.dmg"),
    "receipt missing": (dict(drop=(promotion.RECEIPT,)), "lacks release-smoke"),
    "prerelease": (dict(release={"prerelease": True}), "not a published"),
    "draft": (dict(release={"draft": True}), "not a published"),
    "stale evidence of an older release": (dict(evidence={"version": "7.7.1"}), "not the proof of v7.8.0"),
    "evidence of another commit": (dict(evidence={"source": {"tag": "v7.8.0", "commit": "1" * 40}}),
                                   "not the proof of v7.8.0"),
    "unnotarized": (dict(receipt={"notarization": {"status": "not_notarized"}}), "notarized"),
    "no arm64 proof": (dict(receipt={"checks": ["stapled_app", "stapled_dmg", "gatekeeper_app", "gatekeeper_dmg"]}),
                       "arm64"),
    "bytes GitHub did not record": (dict(digest={"SHA256SUMS": f"sha256:{OTHER}"}), "digest GitHub recorded"),
    "GitHub records another DMG": (dict(digest={"Ouroboros-7.8.0.dmg": f"sha256:{OTHER}"}), "another digest"),
}


def test_facts_refuse_an_unproven_release(subtests):
    for change, problem in REFUSALS.values():
        with subtests.test(change=change, problem=problem):
            with pytest.raises(ValueError, match=problem):
                promotion.release_facts("v7.8.0", COMMIT, get=_world(**change))


def test_facts_refuse_bytes_the_url_does_not_serve():
    with pytest.raises(ValueError, match="not the proven DMG"):
        promotion.release_facts("v7.8.0", COMMIT, get=_world(), download=lambda url: (OTHER, len(PAYLOAD)))


# --------------------------------------------------------------------------- promotion topology

def _git(cwd, *args, check=True):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip() if check else result


def _commit(repo, files, message):
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _digest(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


class Official:
    """origin (bare) on today's shape: main and ouroboros-stable are the raw v7.8.0 tag commit,
    and each release leaves ouroboros on its next development marker (BIBLE P9)."""

    def __init__(self, tmp_path):
        self.origin, self.work, self.publisher = tmp_path / "origin.git", tmp_path / "work", tmp_path / "publisher"
        _git(tmp_path, "init", "-q", "--bare", str(self.origin))
        _git(tmp_path, "init", "-q", str(self.work))
        self.tags = {"v7.8.0": self._tag("7.8.0")}
        first = self.tags["v7.8.0"]
        self.push(f"{first}:refs/heads/main", f"{first}:refs/heads/ouroboros-stable",
                  f"{self.develop('7.8.1-alpha.0')}:refs/heads/ouroboros", "refs/tags/v7.8.0")
        _git(self.origin, "symbolic-ref", "HEAD", "refs/heads/main")
        _git(tmp_path, "clone", "-q", str(self.origin), str(self.publisher))

    def _tag(self, version):
        sha = _commit(self.work, {"VERSION": f"{version}\n", "README.md": f"readme {version}\n",
                                  "docs/index.html": f"pages {version}\n"}, f"release {version}")
        _git(self.work, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
             "tag", "-a", f"v{version}", "-m", f"v{version}")
        return sha

    def develop(self, version):
        """One commit that moves VERSION on the checked-out development tip."""
        return _commit(self.work, {"VERSION": f"{version}\n"}, f"development at {version}")

    def push(self, *refspecs):
        _git(self.work, "push", "-q", str(self.origin), *refspecs)

    def release(self, version):
        """Tag the next release on ouroboros; no promotion happens."""
        _git(self.work, "fetch", "-q", str(self.origin), "ouroboros")
        _git(self.work, "checkout", "-q", "FETCH_HEAD")
        self.tags[f"v{version}"] = self._tag(version)
        major, minor, patch = version.split(".")
        next_cycle = self.develop(f"{major}.{minor}.{int(patch) + 1}-alpha.0")
        self.push(f"{next_cycle}:refs/heads/ouroboros", f"refs/tags/v{version}")
        return self.tags[f"v{version}"]

    def refs(self):
        return {name: _git(self.origin, "rev-parse", f"refs/heads/{name}")
                for name in ("main", "ouroboros-stable", "ouroboros")}

    def promote(self, tag, **kwargs):
        def facts(name, commit):
            return {"version": name[1:], "sha256": _digest(name)}
        return promotion.promote(tag, cwd=self.publisher, read_release=facts, **kwargs)

    def stable(self):
        return promotion.stable_selection("origin", self.publisher)


@pytest.fixture()
def official(tmp_path):
    return Official(tmp_path)


def _exact_public_main(official, tag):
    """main is the tag's tree plus the cask, so Pages, README and every file read the release."""
    main = official.refs()["main"]
    assert _git(official.origin, "diff", "--name-status", tag, main) == f"A\t{promotion.CASK_PATH}"
    assert _git(official.origin, "show", f"{main}:{promotion.CASK_PATH}") + "\n" == \
        promotion.render_cask(tag[1:], _digest(tag))
    assert _git(official.origin, "merge-base", "--is-ancestor", official.tags[tag], main) == ""
    return main


def test_the_first_promotion_turns_todays_raw_tag_main_into_the_tag_plus_its_cask(official):
    before = official.refs()
    assert official.promote("v7.8.0", push=True) == 0
    main = _exact_public_main(official, "v7.8.0")
    assert _git(official.origin, "rev-parse", f"{main}^") == before["main"]  # a fast-forward, one parent
    assert official.refs()["ouroboros-stable"] == before["ouroboros-stable"]  # QA already held the tag
    assert official.refs()["ouroboros"] == before["ouroboros"]  # development is never written
    assert official.stable() == "v7.8.0"


def test_releases_tagged_before_any_promotion_promote_in_turn_by_fast_forward(official, monkeypatch):
    v781, v790 = official.release("7.8.1"), official.release("7.9.0")  # both tagged before promotion
    development = official.refs()["ouroboros"]
    assert official.promote("v7.8.1", push=True) == 0
    first = _exact_public_main(official, "v7.8.1")
    assert official.refs()["ouroboros-stable"] == v781 and official.stable() == "v7.8.1"
    assert official.promote("v7.9.0", push=True) == 0
    second = _exact_public_main(official, "v7.9.0")
    assert _git(official.origin, "rev-parse", f"{second}^1", f"{second}^2").split() == [first, v790]
    assert official.refs()["ouroboros-stable"] == v790 and official.stable() == "v7.9.0"
    assert official.refs()["ouroboros"] == development
    monkeypatch.chdir(official.publisher)
    _git(official.publisher, "fetch", "-q", "origin", "--tags", "main")
    assert mirrors.released_commit(second) == v790  # the mirror refresh reads through to the tag
    # The old raw-tag push can no longer land on main by accident: it is not a fast-forward.
    assert _git(official.work, "push", str(official.origin), f"{v790}:refs/heads/main", check=False).returncode


def test_a_repeated_promotion_changes_nothing(official, capsys):
    assert official.promote("v7.8.0", push=True) == 0
    promoted = official.refs()
    assert official.promote("v7.8.0", push=True) == 0
    assert "already Stable" in capsys.readouterr().out
    assert official.refs() == promoted


def test_an_older_release_never_replaces_a_newer_stable(official):
    official.release("7.8.1")
    official.release("7.9.0")
    assert official.promote("v7.9.0", push=True) == 0
    promoted = official.refs()
    with pytest.raises(ValueError, match="downgrade"):
        official.promote("v7.8.1", push=True)
    assert official.refs() == promoted and official.stable() == "v7.9.0"


def test_qa_ahead_of_the_release_stays_where_it_is(official):
    official.release("7.8.1")
    v790 = official.release("7.9.0")
    official.push(f"{v790}:refs/heads/ouroboros-stable")  # QA took v7.9.0 before Stable promotion
    assert official.promote("v7.8.1", push=True) == 0
    _exact_public_main(official, "v7.8.1")
    assert official.refs()["ouroboros-stable"] == v790 and official.stable() == "v7.8.1"
    assert official.promote("v7.9.0", push=True) == 0
    assert official.refs()["ouroboros-stable"] == v790 and official.stable() == "v7.9.0"


def test_a_qa_branch_that_does_not_lead_to_the_release_refuses_the_promotion(official):
    official.release("7.8.1")
    _git(official.work, "checkout", "-q", official.tags["v7.8.0"])
    side = _commit(official.work, {"qa.txt": "side\n"}, "a QA-only commit")
    official.push(f"{side}:refs/heads/ouroboros-stable")
    before = official.refs()
    with pytest.raises(ValueError, match="does not lead to v7.8.1"):
        official.promote("v7.8.1", push=True)
    assert official.refs() == before


def test_a_branch_moved_after_the_plan_rejects_both_and_a_rerun_recovers(official):
    official.release("7.8.1")
    v790 = official.release("7.9.0")
    refs = promotion.fetch_official("origin", official.publisher)
    commit = promotion.tag_commit("v7.8.1", official.publisher)
    plan = promotion.plan_promotion("v7.8.1", commit, promotion.render_cask("7.8.1", _digest("v7.8.1")),
                                    refs["main"], refs["ouroboros-stable"], official.publisher)
    assert plan["main"] and plan["ouroboros-stable"] == commit
    official.push(f"{v790}:refs/heads/ouroboros-stable")  # QA moves on before the push
    before = official.refs()
    with pytest.raises(RuntimeError, match="push"):
        promotion.push_promotion(plan, "origin", official.publisher)
    assert official.refs() == before  # atomic: main did not move either
    assert official.promote("v7.8.1", push=True) == 0  # a rerun plans from the new tips
    _exact_public_main(official, "v7.8.1")
    assert official.refs()["ouroboros-stable"] == v790


@pytest.mark.skipif(sys.platform == "win32", reason="the refusing update hook is a POSIX shell script")
def test_one_refused_branch_leaves_both_untouched(official):
    official.release("7.8.1")
    hook = official.origin / "hooks" / "update"
    hook.write_text('#!/bin/sh\n[ "$1" != refs/heads/main ]\n', encoding="utf-8")
    hook.chmod(0o755)
    before = official.refs()
    with pytest.raises(RuntimeError, match="push"):
        official.promote("v7.8.1", push=True)
    assert official.refs() == before


def test_a_plan_and_an_unproven_release_write_nothing(official, capsys):
    official.release("7.8.1")
    before = official.refs()
    assert official.promote("v7.8.1") == 0  # no --push: the plan is printed
    assert "rerun with --push" in capsys.readouterr().out
    assert official.refs() == before

    def unproven(tag, commit):
        raise ValueError(f"{tag} is not a published, non-prerelease GitHub release")
    with pytest.raises(ValueError, match="not a published"):
        promotion.promote("v7.8.1", cwd=official.publisher, push=True, read_release=unproven)
    with pytest.raises(ValueError, match="plain vX.Y.Z"):
        official.promote("v7.8.1-rc.1", push=True)
    assert official.refs() == before


# --------------------------------------------------------------------------- the next development cycle

def _links(version):
    return "".join(f"[download-{proof_id}]: {release_asset_download_url(proof_id, version)}\n"
                   for proof_id in RELEASE_ASSET_TEMPLATES)


def _development_at(official, files):
    """ouroboros gets one more commit on its own tip; its links are what *files* write into README.md."""
    _git(official.work, "fetch", "-q", str(official.origin), "ouroboros")
    _git(official.work, "checkout", "-q", "FETCH_HEAD")
    official.push(f"{_commit(official.work, files, 'development moves')}:refs/heads/ouroboros")


@pytest.mark.parametrize("development,problem", [
    ("7.8.1\n", "7.8.1 is not a development marker"),  # the release cut, and the cycle never reopened
    ("7.8.1-alpha.0\n", "does not sort above the release 7.8.1"),  # the same base, a different string
    ("7.8.1-rc.1\n", "is not a development marker"),
])
def test_promotion_refuses_a_development_line_that_has_not_opened_a_greater_cycle(official, development, problem):
    official.release("7.8.1")
    _development_at(official, {"VERSION": development})
    before = official.refs()
    for push in (False, True):  # the printed plan refuses too, before anyone reaches for --push
        with pytest.raises(ValueError, match=problem):
            official.promote("v7.8.1", push=push)
    assert official.refs() == before
    _development_at(official, {"VERSION": "7.9.0-alpha.0\n"})  # any numerically greater base opens it
    assert official.promote("v7.8.1", push=True) == 0
    _exact_public_main(official, "v7.8.1")
    assert _git(official.origin, "show", "refs/heads/ouroboros:VERSION") == "7.9.0-alpha.0"


def test_promotion_compares_the_marker_numerically(official):
    official.release("7.9.0")
    _development_at(official, {"VERSION": "7.10.0-alpha.0\n"})  # as text "7.10.0" sorts below "7.9.0"
    assert official.promote("v7.9.0", push=True) == 0
    assert promotion.next_cycle_refusal("v7.11.0", official.tags["v7.9.0"], official.refs()["ouroboros"],
                                        official.publisher).endswith(
        "7.10.0-alpha.0 does not sort above the release 7.11.0")


def test_promotion_refuses_a_release_development_does_not_contain(official):
    official.release("7.8.1")
    _git(official.work, "checkout", "-q", official.tags["v7.8.1"])
    side = _commit(official.work, {"VERSION": "7.8.2\n"}, "a release cut beside development")
    _git(official.work, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "tag", "-a", "v7.8.2", "-m", "v7.8.2")
    official.tags["v7.8.2"] = side
    official.push("refs/tags/v7.8.2")
    _development_at(official, {"VERSION": "7.9.0-alpha.0\n"})
    before = official.refs()
    with pytest.raises(ValueError, match="does not contain v7.8.2"):
        official.promote("v7.8.2", push=True)
    assert official.refs() == before
    _git(official.work, "fetch", "-q", str(official.origin), "ouroboros")
    _git(official.work, "checkout", "-q", "FETCH_HEAD")
    _git(official.work, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "merge", "-q",
         "-s", "ours", "-m", "bring the release onto development", side)
    official.push("HEAD:refs/heads/ouroboros")
    assert official.promote("v7.8.2", push=True) == 0
    _exact_public_main(official, "v7.8.2")


def test_promotion_refuses_development_links_still_on_an_older_release(official):
    """A marker never moves the links, so nothing but this explicit update would ever reach them."""
    official.release("7.8.1")
    _development_at(official, {"VERSION": "7.9.0-alpha.0\n", "README.md": _links("7.8.0")})
    before = official.refs()
    with pytest.raises(ValueError, match="download links still name 7.8.0: move them to the published v7.8.1"):
        official.promote("v7.8.1", push=True)
    assert official.refs() == before
    _development_at(official, {"README.md": _links("7.8.1")})
    assert official.promote("v7.8.1", push=True) == 0


def test_a_finished_promotion_stays_finished_whatever_development_does_next(official, capsys):
    official.release("7.8.1")
    assert official.promote("v7.8.1", push=True) == 0
    promoted = official.refs()["main"]
    _development_at(official, {"VERSION": "7.9.0\n"})  # the next release is being cut
    assert official.promote("v7.8.1", push=True) == 0
    assert "already Stable" in capsys.readouterr().out
    assert official.refs()["main"] == promoted


def test_a_marker_tag_is_never_a_stable_candidate(official):
    official.release("7.8.1")
    assert official.promote("v7.8.1", push=True) == 0
    _git(official.work, "fetch", "-q", str(official.origin), "ouroboros")
    _git(official.work, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "tag", "-a", "v7.8.2-alpha.0", "-m", "an accident", "FETCH_HEAD")
    official.push("refs/tags/v7.8.2-alpha.0")
    before = official.refs()
    promotion.fetch_official("origin", official.publisher)
    with pytest.raises(ValueError, match="plain vX.Y.Z"):
        official.promote("v7.8.2-alpha.0", push=True)
    assert official.refs() == before and official.stable() == "v7.8.1"


@pytest.mark.parametrize("files", [
    {"README.md": _links("7.8.1").replace("/v7.8.1/", "/v7.8.0/", 1)},  # one link left behind
    {"README.md": _links("7.8.1-rc.1")},  # a prerelease is not a published release the links keep
    {"README.md": _links("7.8.1"), "site/install/index.html": _download_page("7.8.0")},  # a page left behind
], ids=["mixed", "prerelease", "page-behind"])
def test_promotion_refuses_development_links_that_name_no_one_published_release(official, files):
    official.release("7.8.1")
    _development_at(official, {"VERSION": "7.9.0-alpha.0\n", **files})
    before = official.refs()
    with pytest.raises(ValueError, match="do not all name one published plain release: move them to the published v7.8.1"):
        official.promote("v7.8.1", push=True)
    assert official.refs() == before
    _development_at(official, {"README.md": _links("7.8.1"), "site/install/index.html": _download_page("7.8.1")})
    assert official.promote("v7.8.1", push=True) == 0
