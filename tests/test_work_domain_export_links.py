"""``work_domain.export_tree`` receives work-controlled tar bytes into a service folder: a received link, composed
links or a name swapped beside the intake never redirects a write; legitimate links and hard-linked files arrive."""

from __future__ import annotations

import io
import stat
import subprocess
import tarfile

import pytest

from ouroboros import work_domain


def _stream(monkeypatch, members, *, mode=0o644):
    """The work domain's ``tar -c`` answer replaced by exactly these crafted members (the bytes W controls)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, kind, value in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            if kind == "file":
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
                continue
            info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            info.linkname = value
            archive.addfile(info)

    def fake_run(argv, *, env=None, stdout=None, timeout_sec=None, **_kw):
        stdout.write(buffer.getvalue())
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(work_domain, "run_in_work", fake_run)
    monkeypatch.setattr(work_domain, "container_path", str)


def test_composed_links_never_carry_a_write_outside_the_intake(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    members = [("a", "symlink", "."), ("a/x", "symlink", "../outside"), ("a/x/file", "file", b"ESCAPED")]
    _stream(monkeypatch, members)
    with pytest.raises(work_domain.WorkDomainRefused):
        work_domain.export_tree(tmp_path / "work", [m[0] for m in members], tmp_path / "intake", links=True)
    assert list(outside.iterdir()) == []


def test_a_name_swapped_for_a_link_beside_the_intake_is_never_followed(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    intake = tmp_path / "intake"
    intake.mkdir()
    (intake / "sub").symlink_to(outside)  # an existing name that is not a folder of its own
    _stream(monkeypatch, [("sub/file", "file", b"ESCAPED")])
    with pytest.raises(work_domain.WorkDomainRefused):
        work_domain.export_tree(tmp_path / "work", ["sub/file"], intake)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("mode", [0o644, 0o755, 0o7755])
def test_valid_links_and_hard_linked_files_still_arrive(tmp_path, monkeypatch, mode):
    members = [("d/a.bin", "file", b"A" * 3000), ("d/b.bin", "hardlink", "d/a.bin"), ("d/l", "symlink", "a.bin"),
               ("top", "symlink", "d/a.bin")]
    _stream(monkeypatch, members, mode=mode)
    received = work_domain.export_tree(tmp_path / "work", [m[0] for m in members], tmp_path / "intake", links=True)
    intake = tmp_path / "intake"
    assert {row["path"] for row in received} == {"d/a.bin", "d/b.bin", "d/l", "top"}
    assert (intake / "d" / "b.bin").read_bytes() == b"A" * 3000 and not (intake / "d" / "b.bin").is_symlink()
    assert (intake / "d" / "l").readlink().as_posix() == "a.bin" and (intake / "top").read_bytes() == b"A" * 3000
    assert stat.S_IMODE((intake / "d" / "a.bin").stat().st_mode) == mode & 0o777
    assert stat.S_IMODE((intake / "d" / "b.bin").stat().st_mode) == mode & 0o777
