#!/usr/bin/env python3
"""Install the common Ouroboros release on an already Magisk-rooted ARM64 phone.

Run from the verified Android source release. Requires Python 3.10+, adb, and
GitHub CLI for public build attestation verification. --help is entirely offline.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shlex
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlsplit
import uuid

_archive_tools = runpy.run_path(str(Path(__file__).parent / "provision/archive_identity.py"))
verify_artifact = _archive_tools["verify_artifact"]
download = _archive_tools["download"]

BASE = "/data/local/ouroboros-phone"
ROOTFS = BASE + "/rootfs"
APP = "/opt/ouroboros"
APT_SOURCES = (Path(__file__).parent / "provision/ubuntu.sources").read_text()
APT_SNAPSHOT = re.search(r"snapshot\.ubuntu\.com/ubuntu/([^/]+)/", APT_SOURCES)[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(argv, *, data=None, capture=True):
    result = subprocess.run([str(item) for item in argv], input=data,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        detail = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{Path(str(argv[0])).name} failed ({result.returncode}): {detail}")
    return result.stdout or b""


def run_stream(argv, log_path):
    """Relay provisioning output immediately and keep the same bytes on disk."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    with Path(log_path).open("wb") as log:
        with subprocess.Popen([str(item) for item in argv], stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            while True:
                block = process.stdout.read1(65536)
                if not block:
                    break
                log.write(block)
                log.flush()
                sys.stdout.write(decoder.decode(block))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b"", final=True))
            sys.stdout.flush()
            code = process.wait()
    Path(str(log_path) + ".exit-code").write_text(str(code) + "\n")
    if code:
        raise RuntimeError(f"{Path(str(argv[0])).name} failed ({code}); complete output: {log_path}")
    return b""


def select_device(adb, requested=None):
    rows = run([adb, "devices"]).decode().splitlines()[1:]
    devices = [line.split() for line in rows if len(line.split()) >= 2]
    if requested:
        if [requested, "device"] not in [row[:2] for row in devices]:
            raise RuntimeError("Selected phone is unavailable or has not authorized this computer.")
        return requested
    ready = [row[0] for row in devices if row[1] == "device"]
    if len(ready) != 1:
        raise RuntimeError("Connect and authorize one phone, or select it explicitly with --serial.")
    return ready[0]


class Phone:
    def __init__(self, adb, serial, proxy=None):
        self.argv = [adb, "-s", serial]
        self.proxy = proxy
        if proxy:
            parsed = urlsplit(proxy)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("--proxy must be an http:// or https:// proxy URL reachable from the phone")

    def shell(self, command, *, data=None, log_path=None):
        # shell-v2 with no PTY preserves remote exit status and separate stderr.
        # exec-out masks su's exit code on real devices and cannot certify install.
        argv = [*self.argv, "shell", "-T", "su -c " + shlex.quote("set -e\n" + command)]
        if log_path is not None:
            return run_stream(argv, log_path)
        return run(argv, data=data)

    def write(self, path, data, mode="600"):
        path = shlex.quote(path)
        self.shell(f"cat > {path}.tmp && chmod {mode} {path}.tmp && mv {path}.tmp {path}", data=data)

    def push(self, local, target):
        temporary = "/data/local/tmp/ouroboros-install-" + uuid.uuid4().hex
        run([*self.argv, "push", local, temporary])
        self.shell("mv " + shlex.quote(temporary) + " " + shlex.quote(target))

    def linux(self, *argv, log_path=None):
        if self.proxy:
            # enter-linux clears the environment; pass the setup-only proxy after
            # chroot. Nothing persists it into core startup or settings.
            argv = ("/usr/bin/env", *[name + "=" + self.proxy for name in
                    ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")],
                    "no_proxy=127.0.0.1,localhost,::1", "NO_PROXY=127.0.0.1,localhost,::1", *argv)
        return self.shell("unshare -m sh " + shlex.quote(BASE + "/bin/enter-linux")
                          + " " + shlex.join([str(item) for item in argv]), log_path=log_path)


def network_warnings(connectivity):
    if not re.search(r"Active default network:\s*(?:none|null)\b", connectivity, re.IGNORECASE):
        return []
    return ["Android reports no active default network. A VPN or its listed DNS servers do not prove "
            "internet access. Connect Wi-Fi/mobile data, or use a reachable --proxy for this setup. "
            "An explicit USB/proxy route can still work; installation is not blocked by this warning."]


def discover_dns(connectivity, overrides):
    if overrides:
        return [str(ipaddress.ip_address(value)) for value in overrides]
    default = re.search(r"Active default network:\s*(\d+)", connectivity)
    lines = connectivity.splitlines()
    if default:
        network = default.group(1)
        lines = [line for line in lines if re.search(r"(?:network\{|netId=|network\()" + network + r"\b", line)]
    addresses = []
    for line in lines:
        match = re.search(r"DnsAddresses:\s*\[([^]]*)\]", line)
        if not match:
            continue
        for value in re.split(r"[,\s]+", match.group(1)):
            try:
                address = str(ipaddress.ip_address(value.lstrip("/")))
            except ValueError:
                continue
            if address not in addresses:
                addresses.append(address)
    if not addresses:
        raise RuntimeError("Android did not report usable DNS servers for its active network. Supply --dns ADDRESS.")
    return addresses


def preflight(phone, dns):
    facts = phone.shell("id -u; getprop ro.product.cpu.abilist; getprop ro.build.version.sdk; "
                        "uname -m; test -x /data/adb/magisk/busybox && echo MAGISK_BUSYBOX; "
                        "command -v unshare; command -v chroot; command -v setsid; df -Pk /data/local").decode()
    lines = facts.splitlines()
    if not lines or lines[0].strip() != "0":
        raise RuntimeError("Allow root for the USB shell in Magisk, then run the installer again.")
    if len(lines) < 9 or "arm64-v8a" not in lines[1] or lines[3].strip() != "aarch64":
        raise RuntimeError("This runtime requires an ARM64 Android kernel and userspace.")
    if int(lines[2]) < 26 or "MAGISK_BUSYBOX" not in lines:
        raise RuntimeError("Android API 26+ and the existing Magisk BusyBox are required; other roots are unqualified.")
    if not all(lines[index].strip() for index in (5, 6, 7)):
        raise RuntimeError("Android must provide unshare, chroot and setsid.")
    available = int(lines[-1].split()[3]) * 1024
    connectivity = phone.shell("dumpsys connectivity").decode()
    warnings = network_warnings(connectivity)
    for warning in warnings:
        print("Network note: " + warning, file=sys.stderr, flush=True)
    try:
        servers = discover_dns(connectivity, dns)
    except RuntimeError:
        if not phone.proxy:
            raise
        servers = []
        print("Network note: no phone DNS was discovered; setup will use the explicit proxy. "
              "Standalone operation still needs the phone's own network.", file=sys.stderr, flush=True)
    return {"uid": 0, "abi": lines[1], "api": int(lines[2]), "kernel_arch": lines[3],
            "free_bytes": available, "dns": servers, "network_warnings": warnings,
            "qualification": "experimental; physically tested on Pixel 10a only"}


def verify_payload(archive, destination, expected_sha, repository):
    actual_sha = digest(archive)
    if expected_sha:
        if actual_sha != expected_sha.lower():
            raise RuntimeError("Archive does not match the explicit owner-supplied SHA-256.")
        trust = "owner_digest"
    else:
        run(["gh", "attestation", "verify", archive, "--repo", repository])
        trust = "github_attestation"
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                raise RuntimeError("Unexpected source archive member: " + member.name)
        handle.extractall(destination)
    payload = destination / "Ouroboros-Android"
    manifest = json.loads((payload / "android_release_manifest.json").read_text())
    if manifest.get("schemaVersion") != 1 or manifest.get("kind") != "android_release_source":
        raise RuntimeError("Not a supported Ouroboros Android source release.")
    actual = {p.relative_to(payload).as_posix(): {"sha256": digest(p), "size": p.stat().st_size}
              for p in payload.rglob("*") if p.is_file() and p != payload / "android_release_manifest.json"}
    if manifest["files"] != actual:
        raise RuntimeError("Source payload does not match the complete release inventory.")
    bundle = json.loads((payload / "repo_bundle_manifest.json").read_text())
    if (bundle["bundle_sha256"] != digest(payload / "repo.bundle")
            or bundle["source_sha"] != manifest["sourceCommit"]
            or bundle["release_tag"] != manifest["releaseTag"]):
        raise RuntimeError("Embedded repository differs from the verified release source.")
    if not expected_sha:
        # Match the same workflow/source binding printed by the common release proof.
        run(["gh", "attestation", "verify", archive, "--repo", repository,
             "--signer-workflow", repository + "/.github/workflows/ci.yml",
             "--source-digest", manifest["sourceCommit"], "--source-ref", "refs/tags/" + manifest["releaseTag"]])
    return payload, {"source_sha": bundle["source_sha"], "version": manifest["version"],
                     "archive_sha256": actual_sha, "source_trust": trust}


def sync_download_cache(phone, cache, restore=False):
    """Keep expensive public apt/pip/browser inputs on the computer between installs."""
    archive = cache / "linux-download-caches.tar"
    paths = ("var/cache/apt/archives root/.cache/pip root/.cache/ms-playwright "
             "opt/ouroboros/data/state/cx/cache")
    if restore and archive.exists():
        phone.push(archive, BASE + "/install-cache/linux-download-caches.tar")
        phone.shell("tar -xf " + BASE + "/install-cache/linux-download-caches.tar -C " + ROOTFS)
    elif not restore:
        temporary = archive.with_name(archive.name + ".partial")
        with temporary.open("wb") as output:
            result = subprocess.run([*phone.argv, "shell", "-T", "su -c " + shlex.quote(
                "set -e; cd " + ROOTFS + "; mkdir -p " + paths + "; tar -cf - " + paths)],
                stdout=output, stderr=subprocess.PIPE)
        if result.returncode == 0:
            os.replace(temporary, archive)
        else:
            print("Download caches remain on the phone; the computer cache copy did not complete.", file=sys.stderr)


def configure_apt(phone):
    """Use one dated source from the first metadata read, including on an empty base."""
    bundle = ROOTFS + "/etc/ssl/certs/ca-certificates.crt"
    if not phone.shell("if test -s " + bundle + "; then echo PRESENT; fi").strip():
        # Ubuntu Base has no CA package yet. Use the trust anchors already on
        # this phone; the signed Ubuntu ca-certificates package replaces this
        # bootstrap bundle normally. Neither TLS nor repository signing is bypassed.
        certificates = phone.shell(
            'for store in /apex/com.android.conscrypt/cacerts /system/etc/security/cacerts; do '
            'if test -d "$store"; then cat "$store"/*; exit 0; fi; done; '
            'echo "Android system CA store is unavailable" >&2; exit 1')
        if b"-----BEGIN CERTIFICATE-----" not in certificates:
            raise RuntimeError("Android system trust store did not provide PEM certificates for initial Ubuntu HTTPS.")
        phone.shell("mkdir -p " + ROOTFS + "/etc/ssl/certs")
        phone.write(bundle, certificates, "644")
        print("Initial Ubuntu HTTPS uses this phone's existing system CA certificates.", flush=True)
    sources = APT_SOURCES
    phone.write(ROOTFS + "/etc/apt/sources.list.d/ubuntu.sources", sources.encode(), "644")
    # The date is pinned in the URI. An additional implicit APT::Snapshot lookup
    # changes index identity between cold update and install on Ubuntu ports.
    phone.write(ROOTFS + "/etc/apt/apt.conf.d/80ouroboros-cache",
                b'APT::Keep-Downloaded-Packages "true";\n', "644")


def install(phone, payload, identity, artifacts, cache, facts):
    marker_path = BASE + "/installation.json"
    existing = phone.shell("if test -f " + marker_path + "; then cat " + marker_path + "; fi")
    marker = json.loads(existing) if existing.strip() else None
    if marker and marker.get("status") == "installed":
        phone.shell("test -s " + ROOTFS + APP + "/signing/host.keystore && test -s "
                    + ROOTFS + APP + "/signing/host-password")
        print("This installation already exists. Its source, data and signing key are preserved; use Ouroboros Updates.")
        return marker
    if marker and marker.get("source_sha") != identity["source_sha"]:
        raise RuntimeError("An interrupted installation belongs to another source. Resume with its original archive.")
    if not marker and phone.shell("test ! -e " + ROOTFS + " || echo EXISTING").strip():
        raise RuntimeError("An existing unmanaged rootfs was found. Preserve it before a separate clean installation.")
    print("Downloading verified dependencies into", cache)
    files = {item["name"]: download(item, cache) for item in artifacts}
    root_archive = files["ubuntu-base"]
    with tarfile.open(root_archive) as handle:
        expanded_root = sum(item.size for item in handle.getmembers() if item.isfile())
    known_bytes = expanded_root + sum(item.stat().st_size for item in files.values())
    if facts["free_bytes"] < known_bytes:
        raise RuntimeError("Insufficient phone storage even for the base and downloaded inputs. Free space and retry.")
    print("Phone free space:", facts["free_bytes"], "bytes. Additional apt/Python/browser space is checked by their installers.")
    phone.shell("mkdir -p " + BASE + "/bin " + BASE + "/install-cache")
    phone.write(marker_path, (json.dumps({**identity, "status": "provisioning"}) + "\n").encode())
    for name, local in files.items():
        remote = BASE + "/install-cache/" + local.name
        checksum = phone.shell("if test -f " + shlex.quote(remote) + "; then sha256sum " + shlex.quote(remote) + "; fi").decode()
        if not checksum.startswith(digest(local) + " "):
            print("Copying", name)
            phone.push(local, remote)
    if not phone.shell("test ! -f " + ROOTFS + "/etc/os-release || echo READY").strip():
        stage = BASE + "/rootfs.extracting"
        phone.shell("mkdir -p " + stage + "; tar -xpf " + shlex.quote(BASE + "/install-cache/" + root_archive.name)
                    + " -C " + stage + "; mv " + stage + " " + ROOTFS)
    for local in (payload / "android/bootstrap").iterdir():
        if local.is_file():
            phone.push(local, BASE + "/bin/" + local.name)
    phone.shell("chmod 755 " + BASE + "/bin/*; mkdir -p " + ROOTFS + APP + "/provision "
                + ROOTFS + "/root/.cache " + ROOTFS + "/var/cache/apt/archives")
    phone.write(ROOTFS + "/etc/resolv.conf", ("".join("nameserver " + x + "\n" for x in facts["dns"])).encode(), "644")
    sync_download_cache(phone, cache, restore=True)
    # A chroot runs no init system: package postinst must not start host services.
    phone.write(ROOTFS + "/usr/sbin/policy-rc.d", b"#!/bin/sh\nexit 101\n", "755")
    configure_apt(phone)
    for name in ("repo.bundle", "repo_bundle_manifest.json"):
        phone.push(payload / name, ROOTFS + APP + "/provision/" + name)
    for local in (payload / "android/provision").rglob("*"):
        if local.is_file():
            relative = local.relative_to(payload / "android/provision").as_posix()
            target = ROOTFS + APP + "/provision/" + relative
            phone.shell("mkdir -p " + shlex.quote(str(PurePosixPath(target).parent)))
            phone.push(local, target)
    pins = [{**item, "cached_file": "/ouroboros-install-cache/" + files[item["name"]].name} for item in artifacts]
    phone.write(ROOTFS + APP + "/provision/downloads.json", json.dumps(pins).encode())
    phone.shell("mkdir -p " + ROOTFS + "/ouroboros-install-cache")
    for local in files.values():
        phone.shell("ln -f " + shlex.quote(BASE + "/install-cache/" + local.name) + " "
                    + shlex.quote(ROOTFS + "/ouroboros-install-cache/" + local.name))
    try:
        logs = cache / "install-logs" / uuid.uuid4().hex
        logs.mkdir(parents=True)
        print("Installing the pinned Ubuntu build/runtime packages. Live logs:", logs, flush=True)
        try:
            phone.linux("/bin/sh", APP + "/provision/packages.sh", log_path=logs / "packages.log")
        except RuntimeError:
            print("Package setup failed; see apt's output above. For DNS/network errors, connect the phone "
                  "to Wi-Fi/mobile data or supply a reachable --proxy, then resume with the same archive.",
                  file=sys.stderr, flush=True)
            raise
        print("Provisioning the common source, compiler, browser and personal signing identity…", flush=True)
        phone.linux("/usr/bin/python3", "-u", APP + "/provision/runtime.py", log_path=logs / "provision.log")
        result = {**identity, "status": "installed", "apt_snapshot": APT_SNAPSHOT,
                  "abi": facts["abi"], "api": facts["api"], "dependencies": artifacts}
        phone.write(marker_path, (json.dumps(result, indent=2) + "\n").encode())
        phone.shell("am start -n ai.ouroboros.android/.MainActivity")
        return result
    finally:
        sync_download_cache(phone, cache)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Downloaded common Android source release archive.")
    parser.add_argument("--serial", help="Exact adb device serial when more than one device is connected.")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "ouroboros" / "android")
    parser.add_argument("--repository", default="razzant/ouroboros", help="Expected GitHub build provenance repository.")
    parser.add_argument("--expected-sha256", help="Explicit owner-trusted unpublished archive digest; records owner_digest, not attestation.")
    parser.add_argument("--accept-sdk-license", action="store_true", help="Accept Google's Android SDK license for upstream SDK downloads.")
    parser.add_argument("--dns", action="append", default=[], metavar="ADDRESS", help="DNS address if active Android DNS discovery is unavailable.")
    parser.add_argument("--proxy", metavar="URL", help="Optional HTTP(S) proxy reachable from the phone, used only during setup; not saved.")
    parser.add_argument("--check", action="store_true", help="Verify source and inspect prerequisites only; do not install.")
    args = parser.parse_args(argv)
    try:
        if not args.check and not args.accept_sdk_license:
            raise RuntimeError("Read https://developer.android.com/studio/terms then pass --accept-sdk-license if you accept.")
        with tempfile.TemporaryDirectory(prefix="ouroboros-android-source-") as directory:
            payload, identity = verify_payload(args.archive, Path(directory), args.expected_sha256, args.repository)
            serial = select_device(args.adb, args.serial)
            phone = Phone(args.adb, serial, args.proxy)
            facts = preflight(phone, args.dns)
            if args.check:
                print(json.dumps({**identity, "device": serial, "preflight": facts}, indent=2))
                return 0
            artifacts = json.loads((payload / "android/provision/artifacts.json").read_text())
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            result = install(phone, payload, identity, artifacts, args.cache_dir, facts)
            print(json.dumps(result, indent=2))
            return 0
    except (OSError, ValueError, KeyError, RuntimeError, tarfile.TarError) as error:
        print("Installation did not complete: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
