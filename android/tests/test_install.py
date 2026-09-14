"""USB installer boundaries with temporary archives and mocked device transport."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import shlex
import urllib.request

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "install.py"
SPEC = importlib.util.spec_from_file_location("ouroboros_android_install_tests", SOURCE)
assert SPEC and SPEC.loader
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


@pytest.mark.parametrize(
    "listing,requested,expected",
    [
        ("one\tdevice\n", None, "one"),
        ("one\tdevice\ntwo\tdevice\n", "two", "two"),
        ("one\tunauthorized\ntwo\tdevice\n", None, "two"),
    ],
)
def test_select_device_uses_only_authorized_exact_device(monkeypatch, listing, requested, expected):
    calls = []

    def run(argv):
        calls.append(argv)
        return ("List of devices attached\n" + listing).encode()

    monkeypatch.setattr(installer, "run", run)
    assert installer.select_device("custom-adb", requested) == expected
    assert calls == [["custom-adb", "devices"]]


@pytest.mark.parametrize(
    "listing,requested",
    [("one\tdevice\ntwo\tdevice\n", None), ("one\tunauthorized\n", None),
     ("one\tunauthorized\n", "one"), ("one\tdevice\n", "missing"), ("", None)],
)
def test_select_device_refuses_ambiguity_or_missing_authorization(monkeypatch, listing, requested):
    monkeypatch.setattr(installer, "run", lambda _argv: ("List of devices attached\n" + listing).encode())
    with pytest.raises(RuntimeError):
        installer.select_device("adb", requested)


def test_dns_uses_the_current_default_network_and_preserves_order():
    connectivity = """Active default network: 101
NetworkAgentInfo{network{100} DnsAddresses: [ /192.0.2.1 ]}
NetworkAgentInfo{network{101} DnsAddresses: [ /192.0.2.2,/2001:db8::53,/192.0.2.2 ]}
"""
    assert installer.discover_dns(connectivity, []) == ["192.0.2.2", "2001:db8::53"]


def test_explicit_dns_is_validated_and_does_not_depend_on_android_output():
    assert installer.discover_dns("unavailable", ["192.0.2.3", "2001:db8::1"]) == ["192.0.2.3", "2001:db8::1"]
    with pytest.raises(ValueError):
        installer.discover_dns("", ["not-an-address"])


def test_missing_active_dns_is_a_visible_failure():
    with pytest.raises(RuntimeError, match="Supply --dns"):
        installer.discover_dns("Active default network: 101\nNetworkAgentInfo{network{100} DnsAddresses: [ /192.0.2.1 ]}", [])


def source_archive(tmp_path, *, mismatch=None, traversal=False):
    payload = tmp_path / "source" / "Ouroboros-Android"
    payload.mkdir(parents=True)
    (payload / "repo.bundle").write_bytes(b"small bundle fixture, never executed")
    bundle = {"source_sha": "a" * 40, "release_tag": "v7.1.0", "app_version": "7.1.0",
              "bundle_sha256": installer.digest(payload / "repo.bundle")}
    if mismatch == "bundle":
        bundle["bundle_sha256"] = "0" * 64
    (payload / "repo_bundle_manifest.json").write_text(json.dumps(bundle), encoding="utf-8")
    manifest = {
        "schemaVersion": 1, "kind": "android_release_source", "version": "7.1.0",
        "sourceCommit": "a" * 40, "releaseTag": "v7.1.0",
        "files": {path.name: {"sha256": installer.digest(path), "size": path.stat().st_size}
                  for path in payload.iterdir()},
    }
    if mismatch == "inventory":
        (payload / "repo.bundle").write_bytes(b"changed after hashing")
    (payload / "android_release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname=payload.name)
        if traversal:
            member = tarfile.TarInfo("../../escape")
            member.size = 1
            handle.addfile(member, io.BytesIO(b"x"))
    return archive


def test_explicit_owner_digest_verifies_complete_source_without_network(tmp_path, monkeypatch):
    archive = source_archive(tmp_path)
    monkeypatch.setattr(installer, "run", lambda *_args, **_kwargs: pytest.fail("owner-digest mode must not contact GitHub"))
    payload, identity = installer.verify_payload(archive, tmp_path / "out", installer.digest(archive), "razzant/ouroboros")
    assert (payload / "repo.bundle").read_bytes() == b"small bundle fixture, never executed"
    assert identity["source_trust"] == "owner_digest"
    assert identity["source_sha"] == "a" * 40
    assert identity["archive_sha256"] == installer.digest(archive)


def test_wrong_owner_digest_refuses_before_extraction_or_transport(tmp_path, monkeypatch):
    archive = source_archive(tmp_path)
    monkeypatch.setattr(installer, "run", lambda *_args, **_kwargs: pytest.fail("no remote fallback for a digest mismatch"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        installer.verify_payload(archive, tmp_path / "out", "0" * 64, "razzant/ouroboros")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mismatch,message", [("inventory", "complete release inventory"), ("bundle", "Embedded repository")])
def test_source_digest_does_not_hide_inventory_or_bundle_mismatch(tmp_path, mismatch, message):
    archive = source_archive(tmp_path, mismatch=mismatch)
    with pytest.raises(RuntimeError, match=message):
        installer.verify_payload(archive, tmp_path / "out", installer.digest(archive), "razzant/ouroboros")


def test_source_archive_cannot_extract_outside_its_destination(tmp_path):
    archive = source_archive(tmp_path, traversal=True)
    with pytest.raises(RuntimeError, match="Unexpected source archive member"):
        installer.verify_payload(archive, tmp_path / "out", installer.digest(archive), "razzant/ouroboros")
    assert not (tmp_path / "escape").exists()


def test_completed_install_rerun_preserves_personal_source_data_and_key(tmp_path, monkeypatch):
    marker = {"status": "installed", "source_sha": "a" * 40, "version": "7.1.0"}
    reads = []

    class ExistingPhone:
        def shell(self, command):
            reads.append(command)
            if command.startswith("if test -f ") and "installation.json" in command:
                return json.dumps(marker).encode()
            assert command.startswith("test -s ")
            assert "/signing/host.keystore" in command and "/signing/host-password" in command
            return b""

        def write(self, *_args):
            pytest.fail("a successful rerun cannot overwrite installation state")

        def push(self, *_args):
            pytest.fail("a successful rerun cannot replace personal source or keys")

        def linux(self, *_args):
            pytest.fail("a successful rerun must use ordinary Updates")

    monkeypatch.setattr(installer, "download", lambda *_args: pytest.fail("a completed install needs no provision download"))
    result = installer.install(ExistingPhone(), tmp_path, {"source_sha": "b" * 40}, [], tmp_path, {})
    assert result is not None and result["source_sha"] == marker["source_sha"]
    assert len(reads) == 2


def test_help_is_offline_and_does_not_touch_a_phone(monkeypatch, capsys):
    monkeypatch.setattr(installer, "run", lambda *_args, **_kwargs: pytest.fail("--help must not run processes"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("--help must remain offline"))
    with pytest.raises(SystemExit) as exc:
        installer.main(["--help"])
    assert exc.value.code == 0
    assert "--archive" in capsys.readouterr().out


def test_phone_shell_preserves_remote_exit_and_binary_input(monkeypatch):
    calls = []
    monkeypatch.setattr(installer, "run", lambda argv, **kwargs: calls.append((argv, kwargs)) or b"ok")
    phone = installer.Phone("adb", "selected")
    assert phone.shell("cat > '/private file'", data=b"\x00\xff") == b"ok"
    argv, options = calls[0]
    assert argv[:5] == ["adb", "-s", "selected", "shell", "-T"]
    assert "exec-out" not in argv
    assert "set -e" in argv[-1]
    assert options["data"] == b"\x00\xff"


def test_proxy_is_explicit_after_chroot_and_loopback_bypasses_it(monkeypatch):
    phone = installer.Phone("adb", "selected", "http://127.0.0.1:8080")
    calls = []
    monkeypatch.setattr(phone, "shell", lambda command, **kwargs: calls.append((command, kwargs)) or b"")
    phone.linux("/bin/sh", "/opt/ouroboros/provision/packages.sh", log_path=Path("packages.log"))
    words = shlex.split(calls[0][0])
    assert words[:5] == ["unshare", "-m", "sh", installer.BASE + "/bin/enter-linux", "/usr/bin/env"]
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        assert name + "=http://127.0.0.1:8080" in words
    assert "no_proxy=127.0.0.1,localhost,::1" in words
    assert "NO_PROXY=127.0.0.1,localhost,::1" in words
    assert words[-2:] == ["/bin/sh", "/opt/ouroboros/provision/packages.sh"]
    assert calls[0][1] == {"log_path": Path("packages.log")}


def test_no_proxy_keeps_the_plain_linux_entry(monkeypatch):
    phone = installer.Phone("adb", "selected")
    calls = []
    monkeypatch.setattr(phone, "shell", lambda command, **kwargs: calls.append(command) or b"")
    phone.linux("python3", "script.py")
    assert shlex.split(calls[0]) == ["unshare", "-m", "sh", installer.BASE + "/bin/enter-linux", "python3", "script.py"]


def test_absent_default_network_warns_without_blocking_a_proxy_route(monkeypatch, capsys):
    phone = installer.Phone("adb", "selected", "http://127.0.0.1:8080")
    facts = ("0\narm64-v8a\n36\naarch64\nMAGISK_BUSYBOX\n/system/bin/unshare\n"
             "/system/bin/chroot\n/system/bin/setsid\nFilesystem 1024-blocks Used Available Capacity Mounted\n"
             "/data 100000 1000 99000 1% /data\n")
    connectivity = "Active default network: none\nTransports: VPN UnderlyingNetworks: []\n"
    monkeypatch.setattr(phone, "shell", lambda command: (connectivity if command == "dumpsys connectivity" else facts).encode())
    result = installer.preflight(phone, [])
    assert result["dns"] == []
    assert result["network_warnings"]
    assert "no active default network" in capsys.readouterr().err
    assert "proxy" not in result


def test_valid_default_with_vpn_does_not_gain_a_new_network_gate():
    assert installer.network_warnings("Active default network: 101\nTransports: VPN UnderlyingNetworks: []") == []


@pytest.mark.parametrize("code", [0, 7])
def test_streamed_output_is_visible_saved_and_keeps_the_producer_exit(monkeypatch, tmp_path, capsys, code):
    output = "apt progress\nОшибка DNS\n".encode()
    observed = []

    class Process:
        stdout = io.BytesIO(output)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def wait(self):
            # Output must already be visible before the producer finishes.
            assert capsys.readouterr().out.encode() == output
            return code

    def popen(argv, **kwargs):
        observed.append((argv, kwargs))
        return Process()

    monkeypatch.setattr(installer.subprocess, "Popen", popen)
    log = tmp_path / "packages.log"
    if code:
        with pytest.raises(RuntimeError, match=r"failed \(7\)"):
            installer.run_stream(["adb", "shell", "-T", "remote"], log)
    else:
        assert installer.run_stream(["adb", "shell", "-T", "remote"], log) == b""
    assert log.read_bytes() == output
    assert Path(str(log) + ".exit-code").read_text() == str(code) + "\n"
    assert observed[0][1]["stderr"] == installer.subprocess.STDOUT


@pytest.mark.parametrize("existing_ca", [False, True])
def test_apt_initialization_pins_the_source_without_snapshot_autodiscovery(existing_ca):
    certificates = b"-----BEGIN CERTIFICATE-----\nfixture\n-----END CERTIFICATE-----\n"
    writes, commands = {}, []

    class Phone:
        def shell(self, command):
            commands.append(command)
            if command.startswith("if test -s "):
                return b"PRESENT" if existing_ca else b""
            if command.startswith("for store in "):
                return certificates
            assert command.startswith("mkdir -p ")
            return b""

        def write(self, path, data, mode):
            assert mode == "644"
            writes[path] = data

    installer.configure_apt(Phone())
    sources = writes[installer.ROOTFS + "/etc/apt/sources.list.d/ubuntu.sources"].decode()
    assert "URIs: https://snapshot.ubuntu.com/ubuntu/" + installer.APT_SNAPSHOT + "/" in sources
    assert "Snapshot: no" in sources
    assert "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg" in sources
    config = writes[installer.ROOTFS + "/etc/apt/apt.conf.d/80ouroboros-cache"]
    assert b"APT::Snapshot" not in config
    assert b"Keep-Downloaded-Packages" in config
    bundle = installer.ROOTFS + "/etc/ssl/certs/ca-certificates.crt"
    assert (bundle not in writes) if existing_ca else (writes[bundle] == certificates)
    assert bool(any(command.startswith("for store in ") for command in commands)) != existing_ca
    assert "Verify-Peer" not in str(writes)
    assert "trusted=yes" not in sources
