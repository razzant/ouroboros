"""Status discloses the last verified installed host APK, and opens without a receipt."""
import os
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "bootstrap" / "core-control"
ANCHOR = "base=/data/local/ouroboros-phone\n"
CAVEAT = "not Git HEAD"
HISTORICAL = "historical; not proof of a live process"


@pytest.fixture
def phone(tmp_path):
    """Execute the real status branch against a temporary phone tree.

    Only the single root anchor line is rewritten, so the status logic under
    test stays byte-identical; a moved anchor fails here instead of letting the
    check silently pass against nothing.
    """
    text = SOURCE.read_text(encoding="utf-8")
    assert text.count(ANCHOR) == 1, "the core-control root anchor moved"
    base = tmp_path / "phone"
    script = tmp_path / "core-control"
    script.write_text(text.replace(ANCHOR, "base=%s\n" % base), encoding="utf-8")
    state = base / "rootfs/opt/ouroboros/data/state"
    state.mkdir(parents=True)
    (state / "server_port").write_text("8765", encoding="utf-8")

    def status():
        env = os.environ.copy()
        fake_bin = base / "fake-bin"
        if fake_bin.exists():
            env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
        return subprocess.run(["sh", str(script), "status"], env=env,
                              capture_output=True, text=True, timeout=30)

    return state, status


def test_status_reports_the_receipt_version_as_the_installed_apk(phone):
    state, status = phone
    (state / "android_host.json").write_text(
        '{\n  "schema_version": 1,\n  "source_commit": "579049da",\n'
        '  "version_code": 6,\n  "version_name": "7.0.0-alpha.2026091202"\n}\n',
        encoding="utf-8")
    result = status()
    assert result.returncode == 0, result.stderr
    assert "Android host APK: 7.0.0-alpha.2026091202 (versionCode 6)" in result.stdout
    assert CAVEAT in result.stdout
    assert "Live core probe: unavailable (Linux entry is not installed)." in result.stdout


def _entry(state, success):
    base = state.parents[4]
    entry = base / "bin/enter-linux"
    entry.parent.mkdir(parents=True)
    entry.write_text("#!/bin/sh\nexit %d\n" % (0 if success else 1), encoding="utf-8")
    entry.chmod(0o755)
    fake_bin = base / "fake-bin"
    fake_bin.mkdir()
    unshare = fake_bin / "unshare"
    unshare.write_text("#!/bin/sh\nshift 2\nexec \"$@\"\n", encoding="utf-8")
    unshare.chmod(0o755)


def test_status_labels_a_saved_receipt_historical_when_probe_is_unhealthy(phone):
    state, status = phone
    _entry(state, success=False)
    (state / "server_process.json").write_text('{"pid": 4933}\n', encoding="utf-8")
    result = status()
    assert result.returncode == 0, result.stderr
    assert "Live core probe: no healthy core at the recorded endpoint." in result.stdout
    assert HISTORICAL in result.stdout
    assert '{"pid": 4933}' in result.stdout


def test_status_separates_healthy_probe_from_saved_receipt(phone):
    state, status = phone
    _entry(state, success=True)
    (state / "server_process.json").write_text('{"pid": 4933}\n', encoding="utf-8")
    result = status()
    assert result.returncode == 0, result.stderr
    assert "Live core probe: healthy at http://127.0.0.1:8765" in result.stdout
    assert HISTORICAL in result.stdout


def test_status_opens_without_a_receipt(phone):
    state, status = phone
    assert not (state / "android_host.json").exists()
    result = status()
    assert result.returncode == 0, result.stderr
    assert "Recorded runtime endpoint: http://127.0.0.1:8765" in result.stdout
    assert "Android host APK" not in result.stdout
    assert CAVEAT not in result.stdout


def test_status_marks_an_unreadable_receipt_unknown(phone):
    state, status = phone
    (state / "android_host.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    result = status()
    assert result.returncode == 0, result.stderr
    assert "Android host APK: unknown (versionCode unknown)" in result.stdout
    assert CAVEAT in result.stdout
