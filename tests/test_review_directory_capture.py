"""A skill seat retains its copy's typed changes before release; an ordinary folder is read in place."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import review_checking as rc
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.review_substrate import run_review_request
from ouroboros.skill_loader import compute_content_hash
from ouroboros.skill_review_packs import skill_checking_source
from tests._review_session_route_shared import (  # noqa: F401 - transport fixtures
    FakeGateway,
    FakeLLM,
    _agent_request,
    _agent_slot,
    _owned_gateway_uses_each_test_transport,
    fake_route,
)
from tests.test_review_session_rights import writable_engine  # noqa: F401


COPY_SURFACES = ["skill_review"]  # the one non-Git subject still copied: ordinary folders are read in place


def _subject(tmp_path, surface, *, links=False):
    root = tmp_path / "external" / "demo" if surface == "skill_review" else tmp_path / "folder"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: demo\ndescription: Fixture.\n---\nBody.\n")
    (root / "old-target").write_bytes(b"old\n")
    (root / "new-target").write_bytes(b"new\n")
    if links:
        try:
            (root / "retarget").symlink_to("old-target")
            (root / "removed-link").symlink_to("missing-target")
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"this host cannot create symlinks: {exc}")
    for name in ("mode-dir", "removed-dir", "to-link"):
        (root / name).mkdir()
        (root / name).chmod(0o755)  # mkdir(mode=...) is filtered by the umask
    (root / "to-link" / "old-child").write_bytes(b"pristine child\n")
    return root


def _require_posix_modes(tmp_path):
    probe = tmp_path / "mode-probe"
    probe.mkdir()
    (probe / "file").write_bytes(b"")
    for path, mode in ((probe, 0o750), (probe / "file", 0o700)):
        path.chmod(mode)
        if stat.S_IMODE(path.stat().st_mode) != mode:
            pytest.skip("this filesystem does not preserve POSIX permission bits")


def _source(root, drive, surface, *, expected_hash=None):
    if surface != "skill_review":
        return rc.freeze_checking_source(root, data_root=drive)
    skill = SimpleNamespace(skill_dir=root, manifest=SimpleNamespace(entry="", scripts=[]))
    return skill_checking_source(
        drive, skill, {"routes": [ReviewRouteKind.AGENT_SESSION], "session_accesses": ["full"]},
        expected_hash if expected_hash is not None else compute_content_hash(root))


def _run(root, drive, surface, source, monkeypatch, edit):
    original_start = FakeGateway.start_run

    def start(self, request, **kwargs):
        assert request["execution"]["workspaceKind"] == "directory"
        assert request["execution"]["isolation"] == "live"
        edit(Path(request["execution"]["workspaceRoot"]))
        return original_start(self, request, **kwargs)

    monkeypatch.setattr(FakeGateway, "start_run", start)
    result = run_review_request(
        _agent_request(surface=surface, call_type=surface, task_id="directory-task",
                       session_root=str(root), checking_source=source),
        slots=[_agent_slot(slot_id="session", session_access="full")],
        drive_root=drive, llm=FakeLLM())
    return result.actors[0]


def _released(actor, area):
    assert actor["status"] == "ok", actor
    receipt = actor["usage"]["checking_evidence"]
    assert receipt["status"] == "retained" and receipt["gaps"] == []
    evidence = Path(receipt["evidence_dir"])
    assert json.loads((evidence / "summary.json").read_text()) == receipt
    assert "entries" in receipt, "released directory evidence loses typed symlink and mode changes"
    assert not area.exists()
    assert len(FakeGateway.instances[-1].start_requests) == 1
    return receipt["entries"], evidence


@pytest.mark.parametrize("surface", COPY_SURFACES)
def test_public_directory_consumer_retains_content_deletions_and_evidence_before_release(
    tmp_path, writable_engine, monkeypatch, surface,  # noqa: F811
):
    root, drive = _subject(tmp_path, surface), tmp_path / "drive"
    observed = {}

    def edit(area):
        observed["area"] = area
        (area / "new-target").write_bytes(b"rewritten\n")
        (area / "old-target").unlink()
        (area / "new-dir").mkdir()
        (area / "removed-dir").rmdir()
        (area / "to-link" / "old-child").unlink()
        (area / ".review-evidence-extra.txt").write_bytes(b"ordinary changed file\n")
        declared = area / rc.EVIDENCE_DIRNAME
        declared.mkdir()
        (declared / "proof.log").write_bytes(b"all checks complete\n")

    actor = _run(root, drive, surface, _source(root, drive, surface), monkeypatch, edit)
    entries, evidence = _released(actor, observed["area"])
    assert entries["new-target"]["after"]["kind"] == "file"
    assert entries["old-target"]["before"]["kind"] == "file" and entries["old-target"]["after"] is None
    assert entries["new-dir"]["before"] is None and entries["new-dir"]["after"]["kind"] == "directory"
    assert entries["removed-dir"]["before"]["kind"] == "directory" and entries["removed-dir"]["after"] is None
    assert entries["to-link/old-child"]["after"] is None
    assert (evidence / "files" / "new-target").read_bytes() == b"rewritten\n"
    assert (evidence / "files" / ".review-evidence-extra.txt").read_bytes() == b"ordinary changed file\n"
    assert (evidence / "files" / rc.EVIDENCE_DIRNAME / "proof.log").read_bytes() == b"all checks complete\n"
    assert (root / "new-target").read_bytes() == b"new\n" and (root / "old-target").read_bytes() == b"old\n"
    assert (root / "removed-dir").is_dir() and not (root / "new-dir").exists()
    assert (root / "to-link" / "old-child").read_bytes() == b"pristine child\n"


@pytest.mark.parametrize("surface", COPY_SURFACES)
def test_public_directory_consumer_retains_links_before_release(
    tmp_path, writable_engine, monkeypatch, surface,  # noqa: F811
):
    root, drive = _subject(tmp_path, surface, links=True), tmp_path / "drive"
    source = _source(root, drive, surface)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private").write_bytes(b"must not follow directory link\n")
    observed = {}

    def edit(area):
        observed["area"] = area
        (area / "retarget").unlink()
        (area / "retarget").symlink_to("new-target")
        (area / "new-link").symlink_to("missing-new-target")
        (area / "removed-link").unlink()
        (area / "to-link" / "old-child").unlink()
        (area / "to-link").rmdir()
        (area / "to-link").symlink_to(outside, target_is_directory=True)
        # The host's own link text: Windows may report an absolute target with a \\?\ prefix.
        observed["outside-link"] = os.readlink(area / "to-link")

    actor = _run(root, drive, surface, source, monkeypatch, edit)
    entries, _evidence = _released(actor, observed["area"])
    assert entries["retarget"]["before"]["target"] == "old-target"
    assert entries["retarget"]["after"]["kind"] == "symlink"
    assert entries["retarget"]["after"]["target"] == "new-target"
    assert entries["new-link"]["before"] is None
    assert entries["new-link"]["after"]["target"] == "missing-new-target"
    assert entries["removed-link"]["before"]["target"] == "missing-target"
    assert entries["removed-link"]["after"] is None
    assert entries["to-link"]["before"]["kind"] == "directory"
    assert entries["to-link"]["after"]["target"] == observed["outside-link"]
    assert entries["to-link/old-child"]["after"] is None
    assert not any("private" in path for path in entries)
    assert os.readlink(root / "retarget") == "old-target"
    assert (root / "removed-link").is_symlink()
    assert (root / "to-link" / "old-child").read_bytes() == b"pristine child\n"
    assert (outside / "private").read_bytes() == b"must not follow directory link\n"


@pytest.mark.parametrize("surface", COPY_SURFACES)
def test_public_directory_consumer_retains_posix_modes_before_release(
    tmp_path, writable_engine, monkeypatch, surface,  # noqa: F811
):
    _require_posix_modes(tmp_path)
    root, drive = _subject(tmp_path, surface), tmp_path / "drive"
    observed = {}

    def edit(area):
        observed["area"] = area
        (area / "mode-dir").chmod(0o700)
        (area / "new-dir").mkdir()
        (area / "new-dir").chmod(0o750)
        (area / "removed-dir").rmdir()
        (area / "new-target").chmod(0o700)

    actor = _run(root, drive, surface, _source(root, drive, surface), monkeypatch, edit)
    entries, evidence = _released(actor, observed["area"])
    assert entries["mode-dir"]["before"] == {"kind": "directory", "mode": 0o755}
    assert entries["mode-dir"]["after"] == {"kind": "directory", "mode": 0o700}
    assert entries["new-dir"] == {"before": None, "after": {"kind": "directory", "mode": 0o750}}
    assert entries["removed-dir"] == {"before": {"kind": "directory", "mode": 0o755}, "after": None}
    assert entries["new-target"]["after"]["mode"] == 0o700
    assert (evidence / "files" / "new-target").read_bytes() == b"new\n"
    assert stat.S_IMODE((evidence / "files" / "new-target").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "mode-dir").stat().st_mode) == 0o755


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO output requires POSIX")
@pytest.mark.parametrize("surface", COPY_SURFACES)
def test_public_directory_consumer_names_unretainable_output_and_releases_the_copy(
    tmp_path, writable_engine, monkeypatch, surface,  # noqa: F811
):
    root, drive = _subject(tmp_path, surface, links=True), tmp_path / "drive"
    observed = []
    actor = _run(root, drive, surface, _source(root, drive, surface), monkeypatch,
                 lambda area: observed.append(area) or os.mkfifo(area / "pipe"))
    assert actor["status"] == "ok", actor
    receipt = actor["usage"]["checking_evidence"]
    assert receipt["status"] == "partial", "special output must not disappear behind a complete receipt"
    assert receipt["gaps"] == [{"path": "pipe", "reason": "unsupported_entry_type"}]
    assert receipt["entries"]["pipe"]["after"]["kind"] == "special"
    assert not observed[0].exists(), "a receipt that names its gap releases the copy"


def test_skill_consumer_rejects_foreign_payload_hash_before_start(tmp_path, writable_engine):  # noqa: F811
    root, drive = _subject(tmp_path, "skill_review"), tmp_path / "drive"
    source = _source(root, drive, "skill_review", expected_hash="unrelated-payload-hash")
    result = run_review_request(
        _agent_request(surface="skill_review", call_type="skill_review", task_id="directory-task",
                       session_root=str(root), checking_source=source),
        slots=[_agent_slot(slot_id="session", session_access="full")],
        drive_root=drive, llm=FakeLLM())
    assert result.actors[0]["status"] != "ok"
    assert not any(gateway.start_requests for gateway in FakeGateway.instances)
    assert not list(drive.glob("state/review_checkouts/seats/*/area"))


def test_directory_link_metadata_is_retained_whole_beside_the_paid_result(
    tmp_path, writable_engine, monkeypatch,  # noqa: F811
):
    root, drive = _subject(tmp_path, "task_acceptance", links=True), tmp_path / "drive"
    observed = {}

    def edit(area):
        observed["area"] = area
        (area / "long-link").symlink_to("x" * 512)
        (area / rc.EVIDENCE_DIRNAME).mkdir()
        (area / rc.EVIDENCE_DIRNAME / "proof.log").write_bytes(b"proof\n")

    actor = _run(root, drive, "skill_review", _source(root, drive, "skill_review"),
                 monkeypatch, edit)
    entries, evidence = _released(actor, observed["area"])
    assert entries["long-link"]["before"] is None and entries["long-link"]["after"]["target"] == "x" * 512
    assert (evidence / "files" / rc.EVIDENCE_DIRNAME / "proof.log").read_bytes() == b"proof\n"


def _same_stat_rewrite(path, payload):
    """A timestamp-preserving rewrite (``cp -p``, ``rsync -t``, an archive tool):
    new bytes of the same length, with the file's own times put back. The file is
    regular, never a link, so the times go back through its path: Windows has no
    no-follow ``utime`` (it raises NotImplementedError there)."""
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and len(payload) == before.st_size
    path.write_bytes(payload)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def _same_stat(left, right):
    one, two = left.lstat(), right.lstat()
    return (one.st_mode, one.st_size, one.st_mtime_ns) == (two.st_mode, two.st_size, two.st_mtime_ns)


def _acl_read_denial(path, *, clear=False):
    """macOS ACL denial: mode, size and mtime stay; reading the bytes fails."""
    import getpass
    import subprocess

    argv = ["-N"] if clear else ["+a", f"user:{getpass.getuser()} deny read"]
    return subprocess.run(["/bin/chmod", *argv, str(path)], capture_output=True).returncode == 0


def _require_same_stat_read_denial(tmp_path):
    import sys

    probe = tmp_path / "acl-probe"
    probe.write_bytes(b"probe")
    before = probe.lstat()
    if sys.platform != "darwin" or not _acl_read_denial(probe):
        pytest.skip("a read denial that keeps the POSIX stat needs macOS ACLs")
    try:
        probe.read_bytes()
        denied = False
    except PermissionError:
        denied = True
    _acl_read_denial(probe, clear=True)
    if not denied or stat.S_IMODE(probe.lstat().st_mode) != stat.S_IMODE(before.st_mode):
        pytest.skip("this host's ACL denial does not leave an unreadable same-mode file")


@pytest.mark.parametrize("surface", COPY_SURFACES)
def test_public_directory_consumer_retains_same_stat_content_edit_before_release(
    tmp_path, writable_engine, monkeypatch, surface,  # noqa: F811
):
    root, drive = _subject(tmp_path, surface), tmp_path / "drive"
    source = _source(root, drive, surface)
    observed = {}

    def edit(area):
        observed["area"] = area
        _same_stat_rewrite(area / "new-target", b"NEW\n")
        observed["same_stat"] = _same_stat(Path(source["view"]) / "new-target", area / "new-target")

    actor = _run(root, drive, surface, source, monkeypatch, edit)
    if not observed["same_stat"]:
        pytest.skip("this filesystem did not keep the rewritten file's size, mode and mtime")
    entries, evidence = _released(actor, observed["area"])
    assert "new-target" in actor["usage"]["checking_evidence"]["changed"], (
        "a same-size, same-mtime edit read as unchanged and was released with the copy")
    assert (evidence / "files" / "new-target").read_bytes() == b"NEW\n"
    assert entries["new-target"]["before"] == entries["new-target"]["after"]  # only the bytes moved
    assert (root / "new-target").read_bytes() == b"new\n"


@pytest.mark.parametrize("surface", COPY_SURFACES)
def test_public_directory_consumer_keeps_copy_when_same_stat_bytes_are_unreadable(
    tmp_path, writable_engine, monkeypatch, surface,  # noqa: F811
):
    _require_same_stat_read_denial(tmp_path)
    root, drive = _subject(tmp_path, surface), tmp_path / "drive"
    source = _source(root, drive, surface)
    observed = {}

    def edit(area):
        observed["area"] = area
        _same_stat_rewrite(area / "new-target", b"NEW\n")
        observed["denied"] = _acl_read_denial(area / "new-target")
        observed["same_stat"] = _same_stat(Path(source["view"]) / "new-target", area / "new-target")

    try:
        actor = _run(root, drive, surface, source, monkeypatch, edit)
    finally:
        if (observed["area"] / "new-target").exists():
            _acl_read_denial(observed["area"] / "new-target", clear=True)
    assert observed["denied"] and observed["same_stat"]
    assert actor["status"] == "ok", actor
    receipt = actor["usage"]["checking_evidence"]
    assert receipt["status"] == "capture_failed", f"uncompared bytes were released as unchanged: {receipt}"
    assert "new-target" in json.dumps([receipt.get("gaps"), receipt.get("error")]), receipt
    assert (observed["area"] / "new-target").read_bytes() == b"NEW\n"


@pytest.mark.skipif(os.name == "nt", reason="symlink privileges vary on Windows")
@pytest.mark.parametrize("surface", COPY_SURFACES)
def test_public_directory_consumer_compares_content_without_reading_through_links(
    tmp_path, writable_engine, monkeypatch, surface,  # noqa: F811
):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret"
    secret.write_bytes(b"private\n")
    root, drive = _subject(tmp_path, surface), tmp_path / "drive"
    (root / "file-link").symlink_to(secret)
    (root / "dir-link").symlink_to(outside, target_is_directory=True)
    source = _source(root, drive, surface)
    observed = {}

    def edit(area):
        observed["area"] = area
        _same_stat_rewrite(secret, b"PRIVATE\n")
        secret.chmod(0)  # a comparison that follows either link cannot even read the target

    try:
        actor = _run(root, drive, surface, source, monkeypatch, edit)
    finally:
        secret.chmod(0o600)
    entries, _evidence = _released(actor, observed["area"])
    assert "file-link" not in entries and "dir-link" not in entries
    assert os.readlink(root / "file-link") == str(secret)
    assert secret.read_bytes() == b"PRIVATE\n"


@pytest.mark.skipif(os.name == "nt" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="needs POSIX permission bits that bind the test user")
@pytest.mark.parametrize("surface", COPY_SURFACES)
def test_public_directory_consumer_keeps_copy_when_a_subtree_cannot_be_listed(
    tmp_path, writable_engine, monkeypatch, surface,  # noqa: F811
):
    root, drive = _subject(tmp_path, surface), tmp_path / "drive"
    observed = {}

    def edit(area):
        observed["area"] = area
        locked = area / "locked"
        locked.mkdir()
        (locked / "only-copy.log").write_bytes(b"written before the folder was locked\n")
        locked.chmod(0)

    try:
        actor = _run(root, drive, surface, _source(root, drive, surface), monkeypatch, edit)
    finally:
        if (observed["area"] / "locked").exists():
            (observed["area"] / "locked").chmod(0o755)
    assert actor["status"] == "ok", actor
    receipt = actor["usage"]["checking_evidence"]
    assert receipt["status"] == "capture_failed"
    assert (observed["area"] / "locked" / "only-copy.log").read_bytes() == b"written before the folder was locked\n"
    assert (observed["area"] / "SKILL.md").is_file(), "an incomplete capture must not partially release the copy"


def test_an_ordinary_folder_is_read_in_place_and_keeps_only_published_evidence(
    tmp_path, writable_engine, monkeypatch,  # noqa: F811
):
    root, drive = _subject(tmp_path, "task_acceptance"), tmp_path / "drive"
    observed = {}

    def edit(area):
        observed.update(area=area, started_empty=not any(area.iterdir()))
        (area / "copied-to-run.txt").write_bytes((root / "new-target").read_bytes())
        (area / rc.EVIDENCE_DIRNAME).mkdir()
        (area / rc.EVIDENCE_DIRNAME / "proof.log").write_bytes(b"checked the live folder\n")

    source = _source(root, drive, "task_acceptance")
    assert source == {"kind": "workspace", "root": str(root)}
    actor = _run(root, drive, "task_acceptance", source, monkeypatch, edit)
    assert actor["status"] == "ok", actor
    receipt = actor["usage"]["checking_evidence"]
    assert observed["started_empty"] and receipt["status"] == "retained" and not observed["area"].exists()
    kept = Path(receipt["evidence_dir"]) / "files"
    assert (kept / rc.EVIDENCE_DIRNAME / "proof.log").read_bytes() == b"checked the live folder\n"
    assert not (kept / "copied-to-run.txt").exists()  # scratch, not published evidence
    assert not (drive / "state" / "review_checkouts" / "views").exists()  # nothing of the folder was copied
