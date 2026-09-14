"""Execute the same localhost setup used by package provisioning and repairs."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


PACKAGES = Path(__file__).resolve().parents[1] / "provision/packages.sh"


def prepare(path):
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    return subprocess.run(["/bin/sh", str(PACKAGES), "--ensure-localhost", str(path)],
                          capture_output=True, text=True, check=True, env=env)


@pytest.mark.parametrize("initial", [None, b"", b"# owner hosts\n192.0.2.10 owner-machine"])
def test_missing_localhost_mappings_are_added_without_replacing_owner_entries(tmp_path, initial):
    hosts = tmp_path / "hosts"
    if initial is not None:
        hosts.write_bytes(initial)
        hosts.chmod(0o640)
    result = prepare(hosts)
    assert "added: 2" in result.stdout
    assert hosts.read_bytes().startswith(initial or b"")
    assert b"127.0.0.1 localhost\n" in hosts.read_bytes()
    assert b"::1 localhost ip6-localhost ip6-loopback\n" in hosts.read_bytes()
    assert hosts.stat().st_mode & 0o777 == (0o644 if initial is None else 0o640)
    before = hosts.stat()
    content = hosts.read_bytes()
    assert "added: 0" in prepare(hosts).stdout
    assert hosts.read_bytes() == content
    assert hosts.stat().st_mtime_ns == before.st_mtime_ns
    assert hosts.stat().st_ino == before.st_ino


@pytest.mark.parametrize("initial,added", [
    (b"127.0.0.1 my-host localhost # personal alias\n", b"::1 localhost"),
    (b"0:0:0:0:0:0:0:1 LOCALHOST ip6-localhost\n", b"127.0.0.1 localhost"),
    (b"# localhost is missing\n192.0.2.1 not-localhost\n", b"127.0.0.1 localhost"),
])
def test_existing_aliases_and_comments_are_interpreted_as_hosts_fields(tmp_path, initial, added):
    hosts = tmp_path / "hosts"
    hosts.write_bytes(initial)
    prepare(hosts)
    after = hosts.read_bytes()
    assert after.startswith(initial)
    assert added in after[len(initial):]
    if b"127.0.0.1" in initial:
        assert after.count(b"127.0.0.1") == 1
