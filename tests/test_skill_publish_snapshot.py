"""Focused immutable-byte tests for the skill publish snapshot."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib

import pytest

from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    load_skill,
    reduce_skill_content_hash,
)
from ouroboros.skill_publish_snapshot import (
    MAX_PUBLIC_PAYLOAD_BYTES,
    SkillPublishSnapshotError,
    capture_skill_publish_snapshot,
)


class _ObservedBinaryReader:
    def __init__(self, stream, record):
        self._stream = stream
        self._record = record

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._stream.__exit__(exc_type, exc_value, traceback)

    def read(self, size=-1):
        data = self._stream.read(size)
        self._record(size, len(data))
        return data


def _write_skill(root: pathlib.Path, *, manifest: str | None = None) -> pathlib.Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        manifest
        or (
            "---\n"
            "name: demo\n"
            "description: Snapshot fixture.\n"
            "version: 1.2.3\n"
            "type: instruction\n"
            "when_to_use: Use for snapshot tests.\n"
            "install_specs:\n"
            "  - kind: pip\n"
            "    package: example-package\n"
            "---\n"
            "# Demo\n"
        ),
        encoding="utf-8",
    )
    return root


def _reviewed_skill(skill_dir: pathlib.Path, drive_root: pathlib.Path):
    loaded = load_skill(skill_dir, drive_root)
    assert loaded is not None and not loaded.load_error
    loaded.source = "external"
    loaded.review = SkillReviewState(
        status="clean",
        content_hash=loaded.content_hash,
    )
    return loaded


def test_reducer_matches_loader_canonical_inventory_order(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills" / "demo")
    (skill_dir / "payload.bin").write_bytes(b"payload")
    manual = []
    for path in sorted(skill_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(skill_dir).as_posix()
            manual.append((rel, hashlib.sha256(path.read_bytes()).digest()))

    historical = hashlib.sha256()
    for rel, file_digest in manual:
        historical.update(rel.encode("utf-8"))
        historical.update(b"\0")
        historical.update(file_digest)
    assert reduce_skill_content_hash(manual) == historical.hexdigest()
    assert reduce_skill_content_hash(manual) == compute_content_hash(skill_dir)


def test_snapshot_hash_parity_and_full_public_control_views(tmp_path):
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(drive_root / "skills" / "external" / "demo")
    (skill_dir / "payload.txt").write_bytes(b"public payload\n")
    (skill_dir / ".gitleaksignore").write_bytes(b"public scanner input\n")
    (skill_dir / ".ouroboroshub.json").write_bytes(b'{"slug":"origin"}\n')
    loaded = _reviewed_skill(skill_dir, drive_root)

    snapshot = capture_skill_publish_snapshot(loaded)

    assert snapshot.content_hash == compute_content_hash(
        skill_dir,
        manifest_entry=loaded.manifest.entry,
        manifest_scripts=loaded.manifest.scripts,
    )
    assert {item.path for item in snapshot.full_files} == {
        ".gitleaksignore",
        ".ouroboroshub.json",
        "SKILL.md",
        "payload.txt",
    }
    assert {item.path for item in snapshot.public_files} == {
        ".gitleaksignore",
        "SKILL.md",
        "payload.txt",
    }
    assert [item.path for item in snapshot.control_files] == [".ouroboroshub.json"]
    assert snapshot.manifest_file is snapshot.file("SKILL.md")
    assert snapshot.manifest_bytes == (skill_dir / "SKILL.md").read_bytes()
    assert snapshot.public_byte_count == sum(item.byte_count for item in snapshot.public_files)
    assert snapshot.manifest.install_specs() == [{"kind": "pip", "package": "example-package"}]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.skill = "other"  # type: ignore[misc]


def test_manifest_precedence_and_each_inventory_file_read_once(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(drive_root / "skills" / "external" / "demo")
    (skill_dir / "skill.json").write_text(
        '{"name":"wrong","version":"9.9.9","type":"instruction"}',
        encoding="utf-8",
    )
    (skill_dir / "payload.txt").write_bytes(b"once")
    loaded = _reviewed_skill(skill_dir, drive_root)
    loaded.manifest.version = "stale-loaded-projection"
    real_open = pathlib.Path.open
    opens: dict[pathlib.Path, int] = {}
    reads: dict[pathlib.Path, int] = {}

    def counted_open(path: pathlib.Path, *args, **kwargs):
        resolved = path.resolve()
        opens[resolved] = opens.get(resolved, 0) + 1

        def record(_requested: int, _returned: int) -> None:
            reads[resolved] = reads.get(resolved, 0) + 1

        return _ObservedBinaryReader(real_open(path, *args, **kwargs), record)

    monkeypatch.setattr(pathlib.Path, "open", counted_open)
    snapshot = capture_skill_publish_snapshot(loaded)

    assert snapshot.manifest.path == "SKILL.md"
    assert snapshot.manifest.version == "1.2.3"
    expected = {path.resolve() for path in skill_dir.iterdir() if path.is_file()}
    assert set(opens) == expected
    assert set(reads) == expected
    assert set(opens.values()) == {1}
    assert set(reads.values()) == {1}


def test_live_mutation_after_capture_cannot_change_snapshot(tmp_path):
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(drive_root / "skills" / "external" / "demo")
    payload = skill_dir / "payload.txt"
    payload.write_bytes(b"reviewed bytes")
    loaded = _reviewed_skill(skill_dir, drive_root)

    snapshot = capture_skill_publish_snapshot(loaded)
    captured_hash = snapshot.content_hash
    payload.write_bytes(b"later live edit")

    assert snapshot.file("payload.txt").content == b"reviewed bytes"
    assert snapshot.content_hash == captured_hash
    assert compute_content_hash(skill_dir) != captured_hash


def test_snapshot_rejects_review_mismatch_without_live_rehash(tmp_path):
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(drive_root / "skills" / "external" / "demo")
    loaded = _reviewed_skill(skill_dir, drive_root)
    loaded.review.content_hash = "0" * 64

    with pytest.raises(SkillPublishSnapshotError) as caught:
        capture_skill_publish_snapshot(loaded)
    assert caught.value.reason_code == "snapshot_review_stale"


def test_snapshot_enforces_existing_public_limit_only(tmp_path, monkeypatch):
    assert MAX_PUBLIC_PAYLOAD_BYTES == 5 * 1024 * 1024
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(drive_root / "skills" / "external" / "demo")
    (skill_dir / ".ouroboroshub.json").write_bytes(b"x" * 512)
    (skill_dir / "payload.bin").write_bytes(b"y" * 64)
    loaded = _reviewed_skill(skill_dir, drive_root)
    exact_public_size = sum(path.stat().st_size for path in (skill_dir / "SKILL.md", skill_dir / "payload.bin"))
    monkeypatch.setattr(
        "ouroboros.skill_publish_snapshot.MAX_PUBLIC_PAYLOAD_BYTES",
        exact_public_size,
    )

    snapshot = capture_skill_publish_snapshot(loaded)
    assert snapshot.public_byte_count == exact_public_size
    assert snapshot.control_files[0].byte_count == 512

    monkeypatch.setattr(
        "ouroboros.skill_publish_snapshot.MAX_PUBLIC_PAYLOAD_BYTES",
        exact_public_size - 1,
    )

    with pytest.raises(SkillPublishSnapshotError) as caught:
        capture_skill_publish_snapshot(loaded)
    assert caught.value.reason_code == "snapshot_payload_too_large"


def test_snapshot_oversized_public_stream_reads_only_remaining_plus_one(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(drive_root / "skills" / "external" / "demo")
    payload = skill_dir / "payload.bin"
    payload_size = 1024 * 1024 + 1
    with payload.open("wb") as stream:
        stream.seek(payload_size - 1)
        stream.write(b"x")
    loaded = _reviewed_skill(skill_dir, drive_root)
    manifest_size = len((skill_dir / "SKILL.md").read_bytes())
    remaining = 8
    monkeypatch.setattr(
        "ouroboros.skill_publish_snapshot.MAX_PUBLIC_PAYLOAD_BYTES",
        manifest_size + remaining,
    )

    real_open = pathlib.Path.open
    opened = 0
    requested: list[int] = []
    returned: list[int] = []

    def tracked_open(path: pathlib.Path, *args, **kwargs):
        nonlocal opened
        stream = real_open(path, *args, **kwargs)
        if path.resolve() != payload.resolve():
            return stream
        opened += 1

        def record(read_size: int, byte_count: int) -> None:
            requested.append(read_size)
            returned.append(byte_count)

        return _ObservedBinaryReader(stream, record)

    monkeypatch.setattr(pathlib.Path, "open", tracked_open)
    with pytest.raises(SkillPublishSnapshotError) as caught:
        capture_skill_publish_snapshot(loaded)

    assert caught.value.reason_code == "snapshot_payload_too_large"
    assert opened == 1
    assert requested == [remaining + 1]
    assert returned == [remaining + 1]
    assert returned[0] < payload_size


def test_snapshot_fails_closed_on_sensitive_or_unreadable_payload(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    sensitive = _write_skill(drive_root / "skills" / "external" / "sensitive")
    (sensitive / ".env").write_bytes(b"fixture")
    loaded = load_skill(sensitive, drive_root)
    assert loaded is not None and loaded.load_error
    loaded.load_error = ""
    loaded.review = SkillReviewState(status="clean", content_hash="irrelevant")
    with pytest.raises(SkillPublishSnapshotError) as caught:
        capture_skill_publish_snapshot(loaded)
    assert caught.value.reason_code == "snapshot_payload_unreadable"

    unreadable = _write_skill(drive_root / "skills" / "external" / "unreadable")
    payload = unreadable / "payload.txt"
    payload.write_bytes(b"fixture")
    loaded = _reviewed_skill(unreadable, drive_root)
    real_open = pathlib.Path.open

    def fail_selected(path: pathlib.Path, *args, **kwargs):
        if path.resolve() == payload.resolve():
            raise OSError("fixture read failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", fail_selected)
    with pytest.raises(SkillPublishSnapshotError) as caught:
        capture_skill_publish_snapshot(loaded)
    assert caught.value.reason_code == "snapshot_payload_unreadable"
    assert str(caught.value) == "snapshot_payload_unreadable"


@pytest.mark.skipif(os.name == "nt", reason="symlink creation is not portable on Windows")
def test_snapshot_excludes_symlink_escape(tmp_path):
    drive_root = tmp_path / "drive"
    skill_dir = _write_skill(drive_root / "skills" / "external" / "demo")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    os.symlink(outside, skill_dir / "escape.txt")
    loaded = _reviewed_skill(skill_dir, drive_root)

    snapshot = capture_skill_publish_snapshot(loaded)

    assert "escape.txt" not in {item.path for item in snapshot.full_files}
