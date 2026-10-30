"""Official macOS package facts with a fake runner; codesign, spctl, hdiutil, ditto and open never run.

``FakeMac`` answers the module's absolute tool paths from one signing identity (a bundle's
code-directory hash can differ per path) and fails the test on any other command; an attach
mounts the payload of the version the image is named for. Protocol declarations, tags and
receipts are real files in real Git under ``tmp_path``. Nothing here qualifies native signing,
LaunchServices, TCC or an old app's shutdown (``ouroboros/native_package.py``).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import plistlib
import re
import shutil
import subprocess
import sys

import pytest

from ouroboros import native_package as native
from ouroboros.launcher_bootstrap import BUNDLE_MANIFEST_NAME

TEAM = "OFFICIAL01"
CDHASH = "c0de" * 10
INSTALLED, TARGET = "7.8.0", "7.9.0"
URL = f"https://github.com/razzant/ouroboros/releases/download/v{TARGET}/Ouroboros-{TARGET}.dmg"
IMAGE = b"official disk image bytes"
RUNTIME = f"{sys.version_info.major}.{sys.version_info.minor}"


def make_app(path, *, version, bundle_id=native.BUNDLE_ID, tag=None, entry=True, runtime=RUNTIME):
    """A bundle as the launcher reads it; its embedded interpreter is this test's Python, really run."""
    contents = path / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Frameworks" / "python-standalone" / "lib" / f"python{runtime}").mkdir(parents=True)
    (contents / "Frameworks" / "python-standalone" / "bin").mkdir()
    (contents / "Frameworks" / "python-standalone" / "bin" / "python3").symlink_to(sys.executable)
    with open(contents / "Info.plist", "wb") as stream:
        plistlib.dump({"CFBundleIdentifier": bundle_id, "CFBundleShortVersionString": version}, stream)
    manifest = {"app_version": version, "release_tag": f"v{version}" if tag is None else tag}
    (contents / "Frameworks" / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (contents / "MacOS" / "Ouroboros").write_bytes(b"\0" * 64)
    if entry:
        shell = contents / "MacOS" / "ouroboros-shell"
        shell.write_bytes(b"\0" * 16)
        shell.chmod(0o755)
    return path


def done(code=0, err=""):
    return subprocess.CompletedProcess([], code, "", err)


def capsule(*versions):
    return "".join(f"{hashlib.sha256(IMAGE).hexdigest()}  Ouroboros-{version}.dmg\n" for version in versions) + (
        f"{'0' * 64}  Ouroboros-{TARGET}-windows-x64.zip\n")


class FakeMac:
    """One signing identity for every bundle; ``cdhashes`` gives a path its own code-directory hash."""

    def __init__(self, **identity):
        self.calls, self.missing, self.cdhashes, self.payload = [], set(), {}, {}
        self.unnotarized = set()  # app versions Gatekeeper reports as Unnotarized Developer ID (official 7.7.0)
        self.identity = {"verify": 0, "team": TEAM, "identifier": native.BUNDLE_ID, "spctl": 0,
                         "source": "Notarized Developer ID", "cdhash": CDHASH, **identity}

    def __call__(self, argv):
        self.calls.append(list(argv))
        tool, ident = argv[0], self.identity
        if tool in self.missing:
            raise FileNotFoundError(tool)
        if argv[:2] == [native.CODESIGN, "--verify"]:
            return done(ident["verify"], "" if ident["verify"] == 0 else f"{argv[-1]}: invalid signature")
        if argv[:2] == [native.CODESIGN, "-dv"]:
            if ident["team"] is None and ident["verify"] != 0:
                return done(1, f"{argv[-1]}: code object is not signed at all")
            cdhash = next((value for path, value in self.cdhashes.items() if argv[-1].startswith(path)),
                          ident["cdhash"])
            # As the real tool: the code-directory hash only from verbosity 3 on (2 omits it).
            verbosity = max([int(arg.split("=", 1)[1]) for arg in argv if arg.startswith("--verbose=")] or [1])
            return done(0, f"Executable={argv[-1]}/Contents/MacOS/Ouroboros\nIdentifier={ident['identifier']}\n"
                           + (f"CDHash={cdhash}\n" if verbosity >= 3 else "")
                           + f"TeamIdentifier={ident['team'] or 'not set'}\n")
        if tool == native.SPCTL:
            code, source = ident["spctl"], ident["source"]
            try:
                with open(pathlib.Path(argv[-1]) / "Contents" / "Info.plist", "rb") as stream:
                    if plistlib.load(stream).get("CFBundleShortVersionString") in self.unnotarized:
                        code, source = 3, "Unnotarized Developer ID"
            except OSError:
                pass
            verdict = "accepted" if code == 0 else "rejected"
            return done(code, f"{argv[-1]}: {verdict}\nsource={source}\n")
        if argv[:2] == [native.HDIUTIL, "attach"]:
            mount = pathlib.Path(argv[argv.index("-mountpoint") + 1])
            version = re.search(r"Ouroboros-([0-9.]+)\.dmg$", argv[-1]).group(1)
            make_app(mount / native.PAYLOAD, version=version, **self.payload)
            return done()
        if argv[:2] == [native.HDIUTIL, "detach"]:
            shutil.rmtree(pathlib.Path(argv[-1]) / native.PAYLOAD, ignore_errors=True)
            return done()
        if tool == native.DITTO:
            shutil.copytree(argv[1], argv[2], symlinks=True)
            return done()
        raise AssertionError(f"unexpected command {argv}")


def labels(fake):
    return [" ".join([os.path.basename(call[0]), *call[1:2]]) if call[0] in (native.CODESIGN, native.HDIUTIL)
            else os.path.basename(call[0]) for call in fake.calls]


def installed_official(tmp_path, fake=None, **options):
    app = make_app(tmp_path / "Applications" / "Ouroboros.app", version=INSTALLED)
    fake = fake or FakeMac()
    facts = native.installed_app(app, runner=fake, official_team_id=TEAM,
                                 **{"published_tags": [f"v{INSTALLED}", f"v{TARGET}"], **options})
    return app, fake, facts


def _image(tmp_path, version=TARGET):
    image = tmp_path / f"Ouroboros-{version}.dmg"
    image.write_bytes(IMAGE)
    return image, hashlib.sha256(IMAGE).hexdigest()


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def releases(tmp_path):
    """Real Git: releases v7.7.0 (no protocol), v7.8.0 (protocol 1) and v7.9.0 (protocol 2)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    shas = {}
    for version, body in (("7.7.0", "print('launcher')\n"), ("7.8.0", "MACOS_SOURCE_ENTRY_PROTOCOL = 1\n"),
                          ("7.9.0", "import os\nMACOS_SOURCE_ENTRY_PROTOCOL = 2  # the entry protocol\n")):
        (repo / "launcher.py").write_text(body)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"release {version}")
        _git(repo, "tag", f"v{version}")
        shas[version] = _git(repo, "rev-parse", "HEAD")
    return repo, shas


def test_the_protocol_is_read_from_its_one_owner_without_importing_it(releases, tmp_path):
    repo, shas = releases
    assert [native.protocol_at(repo, f"v{version}") for version in ("7.7.0", "7.8.0", "7.9.0")] == [None, 1, 2]
    assert native.protocol_at(repo, "v9.9.9") is None  # a tag that is not here
    (repo / "launcher.py").write_text("MACOS_SOURCE_ENTRY_PROTOCOL = int('2')\n")  # not a literal: unread
    _git(repo, "commit", "-qam", "computed")
    assert native.protocol_at(repo, "HEAD") is None


def test_a_new_app_is_needed_when_it_cannot_host_the_targets_source_launcher():
    entry, no_entry = {"entry": True}, {"entry": False}
    need = native.requirement
    assert need(entry, None, serving=1, hosted=1)["reason"] == "nothing_declared"
    assert need(no_entry, None, serving=None, hosted=None)["reason"] == "nothing_declared"  # plain Python suffices
    assert need(entry, 1, serving=1, hosted=None)["reason"] == "protocol_unchanged"
    for serving in (None, 1, 2):  # 7.7.0 checks nothing: no protocol answer stands in for the missing entry
        assert (need(no_entry, 1, serving=serving, hosted=None)["status"],
                need(no_entry, 1, serving=serving, hosted=None)["reason"]) == ("needed", "signed_entry_missing")
    assert need({"entry": None}, 2, serving=1, hosted=1)["status"] == "unknown"
    assert need(entry, 2, serving=1, hosted=None)["reason"] == "protocol_unread"
    assert need(entry, 2, serving=1, hosted=2)["reason"] == "protocol_hosted"
    assert (need(entry, 2, serving=1, hosted=1)["status"], need(entry, 2, serving=1, hosted=1)["reason"]) == (
        "needed", "protocol_differs")


def test_the_published_7_7_0_app_has_no_signed_entry_and_7_8_0_has_it():
    """The real releases, where their tags are in this clone: 7.7.0 declares no protocol and builds no
    entry; 7.8.0 declares protocol 1 and builds ``Contents/MacOS/ouroboros-shell`` (``build.sh``)."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    builds = {}
    for tag in ("v7.7.0", "v7.8.0"):
        shown = subprocess.run(["git", "cat-file", "blob", f"{tag}:build.sh"], cwd=repo, capture_output=True,
                               text=True)
        if shown.returncode != 0:
            pytest.skip(f"{tag} is not in this clone")
        builds[tag] = "ouroboros-shell" in shown.stdout
    assert (native.protocol_at(repo, "v7.7.0"), builds["v7.7.0"]) == (None, False)
    assert (native.protocol_at(repo, "v7.8.0"), builds["v7.8.0"]) == (1, True)


def test_the_package_is_the_newest_published_release_declaring_that_protocol(releases):
    repo, shas = releases
    published = [{"tag": f"v{v}", "sha": shas[v]} for v in ("7.9.0", "7.8.0", "7.7.0")]
    read = []
    sums = lambda version: read.append(version) or capsule(version)
    chosen = native.select_package(repo, 1, published, sums)
    digest = hashlib.sha256(IMAGE).hexdigest()
    assert (chosen["tag"], chosen["version"], chosen["sha"], chosen["sha256"]) == ("v7.8.0", "7.8.0", shas["7.8.0"],
                                                                                   digest)
    assert read == ["7.8.0"]  # only a release whose tree hosts the protocol is asked for its assets
    assert native.select_package(repo, 2, published, sums)["version"] == "7.9.0"
    assert native.select_package(repo, 3, published, sums) is None  # no published app hosts it yet
    assert native.select_package(repo, 2, [{"tag": "v8.0.0", "sha": "f" * 40}], sums) is None  # not fetched here


def test_a_tag_whose_release_carries_no_macos_image_never_shadows_the_one_that_does(releases):
    """v7.9.0 hosts protocol 1 too here, but its release published no assets (or none for macOS): v7.8.0 is chosen."""
    repo, shas = releases
    (repo / "launcher.py").write_text("MACOS_SOURCE_ENTRY_PROTOCOL = 1\n")
    _git(repo, "commit", "-qam", "release 7.9.1")
    _git(repo, "tag", "v7.9.1")
    published = [{"tag": "v7.9.1", "sha": _git(repo, "rev-parse", "HEAD")}, {"tag": "v7.8.0", "sha": shas["7.8.0"]}]
    unpublished = {"7.9.1": None}
    assert native.select_package(repo, 1, published, lambda v: unpublished.get(v, capsule(v)))["version"] == "7.8.0"
    windows_only = {"7.9.1": f"{'0' * 64}  Ouroboros-7.9.1-windows-x64.zip\n"}
    assert native.select_package(repo, 1, published, lambda v: windows_only.get(v, capsule(v)))["version"] == "7.8.0"

    def unreadable(version):
        raise OSError("offline")

    with pytest.raises(OSError):  # unknown is never a choice: the caller makes it a manual step
        native.select_package(repo, 1, published, unreadable)


def _sealed(path, *, resource="one\n", nested="/usr/bin/true"):
    """A minimal bundle around real Mach-O code, ad-hoc signed by the real codesign (no team, no network)."""
    (path / "Contents" / "MacOS").mkdir(parents=True)
    (path / "Contents" / "Resources").mkdir()
    for source, name in (("/usr/bin/true", "Ouroboros"), (nested, "ouroboros-shell")):  # nested code beside the main
        shutil.copyfile(source, path / "Contents" / "MacOS" / name)  # bytes only: system file flags stay behind
        (path / "Contents" / "MacOS" / name).chmod(0o755)
    with open(path / "Contents" / "Info.plist", "wb") as stream:
        plistlib.dump({"CFBundleExecutable": "Ouroboros", "CFBundleIdentifier": native.BUNDLE_ID,
                       "CFBundleShortVersionString": INSTALLED}, stream)
    (path / "Contents" / "Resources" / "style.css").write_text(resource)
    for target in (path / "Contents" / "MacOS" / "ouroboros-shell", path):
        signed = subprocess.run([native.CODESIGN, "--force", "-s", "-", "-i", native.BUNDLE_ID, str(target)],
                                capture_output=True, text=True)
        if signed.returncode != 0:
            pytest.skip(f"codesign cannot ad-hoc sign here: {signed.stderr.strip()}")
    return path


@pytest.mark.skipif(sys.platform != "darwin" or not os.path.exists(native.CODESIGN), reason="the real codesign")
def test_the_code_directory_hash_covers_resources_and_nested_code_with_the_real_codesign(tmp_path):
    """Why the origin proof compares one hash: it seals Info.plist, every resource and each nested executable.

    Real ``codesign`` on ad-hoc bundles: the hash ``installed_app``/``verify`` compare is read at all
    (verbosity 2 prints none), identical bytes give the same hash, a resource or a nested executable
    that differs gives another, and an edit after signing fails the ``--deep --strict`` check."""
    real = lambda argv: subprocess.run([str(part) for part in argv], capture_output=True, text=True, timeout=120)
    original = native._signed(_sealed(tmp_path / "a" / "Ouroboros.app"), real, None)
    assert original["signature"] is True and re.fullmatch(r"[0-9a-f]{40}", original["cdhash"] or "")
    assert original["bundle_id"] is True
    copy = tmp_path / "copy" / "Ouroboros.app"
    copy.parent.mkdir()
    shutil.copytree(tmp_path / "a" / "Ouroboros.app", copy, symlinks=True)
    assert native._signed(copy, real, None)["cdhash"] == original["cdhash"]  # the same bytes, the same seal
    other_resource = native._signed(_sealed(tmp_path / "b" / "Ouroboros.app", resource="two\n"), real, None)
    other_nested = native._signed(_sealed(tmp_path / "c" / "Ouroboros.app", nested="/usr/bin/false"), real, None)
    assert other_resource["signature"] is True and other_nested["signature"] is True
    assert len({original["cdhash"], other_resource["cdhash"], other_nested["cdhash"]}) == 3
    (copy / "Contents" / "Resources" / "style.css").write_text("edited after signing\n")
    edited = native._signed(copy, real, None)
    # The seal itself is unchanged, so its hash alone would pass: the deep check proves the bytes on disk are it.
    assert edited["cdhash"] == original["cdhash"] and edited["signature"] is False


def test_official_origin_reads_every_fact(tmp_path):
    app, fake, facts = installed_official(tmp_path)
    assert facts["origin"] == "official" and facts["reasons"] == []
    assert facts["checks"] == dict.fromkeys(native.CHECKS, True)
    assert (facts["team"], facts["cdhash"], facts["version"], facts["release_tag"]) == (
        TEAM, CDHASH, INSTALLED, f"v{INSTALLED}")
    assert facts["entry"] is True and facts["bytes"] > 0 and facts["device"] == os.stat(app).st_dev
    assert labels(fake) == ["codesign --verify", "codesign -dv", "spctl"]


@pytest.mark.parametrize("identity", [{"verify": 1, "team": None}, {"team": "SOMEONE999"}],
                         ids=["unsigned-self-build", "other-team"])
def test_custom_origin_is_never_replaced_automatically(tmp_path, identity):
    _, _, facts = installed_official(tmp_path, FakeMac(**identity))
    assert facts["origin"] == "custom" and "team_id" in facts["reasons"]
    planned = native.plan(TARGET, facts, release_metadata={"sha256sums": capsule(TARGET)})
    assert (planned["status"], planned["reason"]) == ("refused", "custom_origin")
    assert planned["url"] == URL and URL in planned["instruction"] and TARGET in planned["instruction"]
    assert planned["staged_app"] is None and planned["sha256"] is None


def test_without_a_proven_team_or_published_tags_nothing_is_official(tmp_path):
    fake = FakeMac()
    fake.missing = {native.CODESIGN, native.SPCTL}
    app, _, facts = installed_official(tmp_path, fake)
    assert facts["origin"] == "unknown" and set(facts["reasons"]) == {"signature", "team_id", "bundle_id"}
    unset = native.installed_app(app, runner=FakeMac())  # every local fact holds; the team is unproven
    assert unset["origin"] == "unknown" and unset["reasons"] == ["team_id", "release"]
    (app / "Contents" / "Info.plist").write_bytes(b"not a property list")
    silent = FakeMac()
    assert native.installed_app(app, runner=silent, official_team_id=TEAM,
                                published_tags=[f"v{INSTALLED}"])["origin"] == "unknown" and silent.calls == []


@pytest.mark.parametrize("broken, identity, app_options, options", [
    ("signature", {"verify": 1}, {}, {}),
    ("bundle_id", {"identifier": "com.example.fork"}, {}, {}),
    ("release", {}, {}, {"published_tags": ["v7.7.0"]}),
    ("release", {}, {"tag": "v7.7.0"}, {}),
], ids=["signature", "signing-id", "unpublished-tag", "foreign-manifest"])
def test_team_id_alone_never_makes_an_app_official(tmp_path, broken, identity, app_options, options):
    app = make_app(tmp_path / "Ouroboros.app", version=INSTALLED, **app_options)
    facts = native.installed_app(app, runner=FakeMac(**identity), official_team_id=TEAM,
                                 **{"published_tags": [f"v{INSTALLED}"], **options})
    assert facts["checks"]["team_id"] is True and facts["origin"] == "custom" and broken in facts["reasons"]


def test_an_installed_official_app_gatekeeper_calls_unnotarized_keeps_its_provenance(tmp_path):
    """The official 7.7.0 app: valid strict Developer ID signature, which offline ``spctl`` reports as
    ``Unnotarized Developer ID``. That is a reported fact, never its origin; a package to install
    is still refused without notarization (``verify``)."""
    app = make_app(tmp_path / "Ouroboros.app", version=INSTALLED)
    facts = native.installed_app(app, runner=FakeMac(spctl=3, source="Unnotarized Developer ID"),
                                 official_team_id=TEAM, published_tags=[f"v{INSTALLED}"])
    assert (facts["origin"], facts["notarized"], facts["gatekeeper_source"]) == (
        "official", False, "Unnotarized Developer ID")
    image, digest = _image(tmp_path, INSTALLED)
    unnotarized = FakeMac(spctl=3, source="Unnotarized Developer ID")
    proof = native.verify(image, digest, runner=unnotarized, version=INSTALLED, notarized=False)
    assert (proof["status"], proof["cdhash"]) == ("verified", CDHASH)  # its own release, read for provenance
    refused = native.verify(image, digest, runner=unnotarized, version=INSTALLED, notarized=False,
                            stage_to=tmp_path / "staging" / "Ouroboros.app")
    assert refused["reason"] == "gatekeeper_rejected"  # what would be installed is checked in full


def test_a_homebrew_receipt_names_who_installed_the_app_and_never_decides_its_origin(tmp_path):
    """The official image a Homebrew Cask installed is official and is planned like any other app. The receipt is
    read only: who installed the bundle and which version it last installed (its newest record, as Homebrew reads
    it), which an in-app Update leaves behind the bundle's own version. Only an app not proven official is left to
    its manager's own upgrade."""
    folder = tmp_path / "Applications"
    app = make_app(folder / "Ouroboros.app", version=TARGET)  # an in-app Update already moved it past Homebrew's
    caskroom = tmp_path / "Caskroom"
    newest = caskroom / "ouroboros" / ".metadata" / INSTALLED / "20261027120000.000" / "Casks" / "ouroboros.json"
    newest.parent.mkdir(parents=True)
    newest.write_text(json.dumps({"token": "ouroboros", "artifacts": [{"app": ["Ouroboros.app"]}, {"zap": []}]}))
    older = caskroom / "ouroboros" / ".metadata" / "7.7.0" / "20261001090000.000" / "Casks" / "ouroboros.rb"
    older.parent.mkdir(parents=True)
    older.write_text('cask "ouroboros" do\n  version "7.7.0"\n  app "Ouroboros.app"\nend\n')
    other = caskroom / "other" / ".metadata" / "1.0" / "1" / "Casks" / "other.rb"
    other.parent.mkdir(parents=True)
    other.write_text('cask "other" do\n  app "Other.app"\nend\n')
    written = {path: path.read_bytes() for path in caskroom.rglob("*") if path.is_file()}

    receipts = native.package_manager_receipts(app, caskrooms=[caskroom], appdirs=[folder])

    upgrade = "brew upgrade --cask ouroboros"
    assert [(r["manager"], r["version"], r["upgrade"]) for r in receipts] == [
        ("Homebrew Cask", INSTALLED, upgrade), ("Homebrew Cask", "7.7.0", upgrade)]
    assert native.package_manager_receipts(app, caskrooms=[caskroom], appdirs=[tmp_path / "Elsewhere"]) == []
    facts = native.installed_app(app, runner=FakeMac(), official_team_id=TEAM, published_tags=[f"v{TARGET}"],
                                 owners=receipts)
    assert facts["checks"] == dict.fromkeys(native.CHECKS, True)
    assert (facts["origin"], facts["manager"], facts["manager_version"], facts["version"]) == (
        "official", "Homebrew Cask", INSTALLED, TARGET)  # Homebrew's record and the bundle's own version, apart
    planned = native.plan("7.9.5", facts, release_metadata={"sha256sums": capsule("7.9.5")})
    assert planned["status"] == "ready" and planned["instruction"].startswith("Download Ouroboros 7.9.5 from https://")
    custom = native.installed_app(app, runner=FakeMac(team="SOMEONE01"), official_team_id=TEAM,
                                  published_tags=[f"v{TARGET}"], owners=receipts)  # another cask's own build
    assert (custom["origin"], custom["manager"]) == ("custom", "Homebrew Cask")
    refused = native.plan("7.9.5", custom, release_metadata={"sha256sums": capsule("7.9.5")})
    assert (refused["status"], refused["reason"]) == ("refused", "custom_origin")
    assert "brew upgrade --cask ouroboros" in refused["instruction"] and "7.9.5" in refused["instruction"]
    assert {path: path.read_bytes() for path in caskroom.rglob("*") if path.is_file()} == written  # read only


def test_staging_on_another_volume_is_refused(tmp_path):
    app, _, facts = installed_official(tmp_path)
    mounted = {**facts, "device": os.stat(app.parent).st_dev + 1}  # an app run from an image or translocated
    planned = native.plan(TARGET, mounted, release_metadata={"sha256sums": capsule(TARGET)})
    assert (planned["status"], planned["reason"]) == ("refused", "other_volume")
    assert not pathlib.Path(planned["staging_dir"]).exists()


def test_missing_checksum_refuses_the_plan_and_the_verification(tmp_path):
    _, _, facts = installed_official(tmp_path)
    conflicting = f"{'a' * 64}  Ouroboros-{TARGET}.dmg\n{'b' * 64}  Ouroboros-{TARGET}.dmg\n"
    for sums in (None, "", f"{'a' * 64}  Ouroboros-{TARGET}-windows-x64.zip\n", conflicting):
        planned = native.plan(TARGET, facts, release_metadata={"sha256sums": sums})
        assert (planned["reason"], planned["sha256"]) == ("checksum_missing", None)
    image, _digest = _image(tmp_path)
    fake = FakeMac()
    assert native.verify(image, "", runner=fake, version=TARGET)["reason"] == "checksum_missing" and not fake.calls


def test_wrong_sha256_is_refused_before_anything_mounts(tmp_path):
    image = tmp_path / f"Ouroboros-{TARGET}.dmg"
    image.write_bytes(b"tampered bytes")
    fake = FakeMac()
    checked = native.verify(image, hashlib.sha256(IMAGE).hexdigest(), runner=fake, version=TARGET,
                            official_team_id=TEAM)
    assert (checked["status"], checked["reason"]) == ("failed", "sha256_mismatch") and fake.calls == []


def test_verify_reads_the_payload_identity_and_detaches_on_every_path(tmp_path):
    image, digest = _image(tmp_path)
    good = FakeMac()
    checked = native.verify(image, digest, runner=good, version=TARGET)
    assert (checked["status"], checked["detached"], checked["team"], checked["cdhash"]) == (
        "verified", True, TEAM, CDHASH)  # the payload's own identity, reported for comparison
    assert good.calls[0][:6] == [native.HDIUTIL, "attach", "-readonly", "-nobrowse", "-noautoopen", "-mountpoint"]
    assert labels(good)[-1] == "hdiutil detach"
    for fake, reason in ((FakeMac(verify=1), "signature_invalid"),
                         (FakeMac(spctl=3, source="Unnotarized Developer ID"), "gatekeeper_rejected"),
                         (FakeMac(cdhash=""), "identity_mismatch")):
        checked = native.verify(image, digest, runner=fake, version=TARGET, official_team_id=TEAM)
        assert (checked["status"], checked["reason"], checked["detached"]) == ("failed", reason, True)
    other_team = native.verify(image, digest, runner=FakeMac(team="SOMEONE999"), version=TARGET,
                               official_team_id=TEAM)
    assert other_team["reason"] == "identity_mismatch"
    older, older_digest = _image(tmp_path, "7.8.9")
    assert native.verify(older, older_digest, runner=FakeMac(), version=TARGET)["reason"] == "identity_mismatch"


def test_staging_measures_the_payload_and_needs_its_entry_and_runtime(tmp_path, monkeypatch):
    image, digest = _image(tmp_path)
    staged = tmp_path / "Apps" / f".Ouroboros-update-{TARGET}" / "Ouroboros.app"
    done_ = native.verify(image, digest, runner=FakeMac(), version=TARGET, official_team_id=TEAM, stage_to=staged,
                          runtime=RUNTIME)
    assert done_["status"] == "verified" and done_["staged"] == str(staged) and (staged / "Contents").is_dir()
    for payload, reason in (({"entry": False}, "entry_missing"), ({"runtime": "3.99"}, "runtime_differs")):
        fake = FakeMac()
        fake.payload = payload
        target = tmp_path / reason / "Ouroboros.app"
        checked = native.verify(image, digest, runner=fake, version=TARGET, official_team_id=TEAM, stage_to=target,
                                runtime=RUNTIME)
        assert (checked["status"], checked["reason"]) == ("failed", reason)
    import collections

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(native.shutil, "disk_usage", lambda _path: usage(10, 10, 5))
    short = native.verify(image, digest, runner=FakeMac(), version=TARGET, official_team_id=TEAM,
                          stage_to=tmp_path / "short" / "Ouroboros.app", runtime=RUNTIME)
    assert short["reason"] == "insufficient_space" and "5 are free" in short["detail"]
    assert not (tmp_path / "short" / "Ouroboros.app").exists()
