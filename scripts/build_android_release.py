#!/usr/bin/env python3
"""Build and inspect Android assets through the common release proof pipeline.

The archive carries first-party source and the existing managed repository seed.
Rootfs, SDKs, caches and installation signing keys are provisioned separately.
Artifact inspection does not claim physical Android/root runtime verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = runpy.run_path(ROOT / "scripts" / "release_proof.py")
BUNDLE = runpy.run_path(ROOT / "scripts" / "build_repo_bundle.py")
SOURCE_PATHS = ("android", "docs/ANDROID_INSTALL.md", "docs/ANDROID_RECOVERY.md", "LICENSE", "assets/icon_1024.png")
REQUIRED_FILES = (
    "android/install.py", "android/host/build.py", "docs/ANDROID_INSTALL.md", "docs/ANDROID_RECOVERY.md",
)


def run(argv: list[object], *, cwd: Path = ROOT, binary: bool = False):
    return subprocess.run(
        [str(item) for item in argv], cwd=cwd, check=True,
        stdout=subprocess.PIPE, text=not binary,
    ).stdout


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict:
    return {"sha256": RELEASE["sha256_file"](path), "size": path.stat().st_size}


def apk_identity(args: argparse.Namespace, apk: Path) -> dict:
    tools = args.sdk / "build-tools" / "36.0.0"
    signature = run([
        args.java_home / "bin" / "java", "-jar", tools / "lib" / "apksigner.jar",
        "verify", "--verbose", "--print-certs", apk,
    ])
    fingerprints = re.findall(r"Signer #\d+ certificate SHA-256 digest: ([0-9a-fA-F]+)", signature)
    if len(fingerprints) != 1 or len(fingerprints[0]) != 64:
        raise ValueError("APK must expose exactly one verified signing certificate")
    badging = run([tools / "aapt2", "dump", "badging", apk])
    package_line = next((line for line in badging.splitlines() if line.startswith("package: ")), "")
    values = dict(item.split("=", 1) for item in shlex.split(package_line)[1:] if "=" in item)
    return {
        "packageName": values.get("name"), "versionName": values.get("versionName"),
        "versionCode": int(values.get("versionCode", "0")),
        "signerSha256": fingerprints[0].lower(),
    }


def create_archive(args: argparse.Namespace, stage: Path, version: str, identity: dict) -> Path:
    BUNDLE["build_bundle"](
        ROOT, stage / "repo.bundle", stage / "repo_bundle_manifest.json",
        source_branch=args.source_branch, local_branch="ouroboros",
        local_stable_branch="ouroboros-stable", remote_stable_branch="ouroboros-stable",
        managed_remote_name="managed",
    )
    paths = run(["git", "ls-files", "-z", "--", *SOURCE_PATHS]).split("\0")
    for name in filter(None, paths):
        source = ROOT / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Android release source is not a regular file: {name}")
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in REQUIRED_FILES:
        if not (stage / name).is_file():
            raise ValueError(f"Android release source is missing: {name}")
    bundle_manifest = json.loads((stage / "repo_bundle_manifest.json").read_text())
    manifest = {
        "schemaVersion": 1, "kind": "android_release_source", "version": version,
        "sourceCommit": bundle_manifest["source_sha"],
        "releaseTag": bundle_manifest["release_tag"],
        "referenceApk": {"name": args.apk.name, **file_record(args.apk), **identity},
        "files": {
            path.relative_to(stage).as_posix(): file_record(path)
            for path in sorted(stage.rglob("*")) if path.is_file()
        },
    }
    write_json(stage / "android_release_manifest.json", manifest)
    archive = args.out / RELEASE["release_asset_name"]("android-arm64", version)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(stage, arcname="Ouroboros-Android")
    return archive


def inspect_archive(args: argparse.Namespace, archive: Path, destination: Path) -> dict:
    with tarfile.open(archive, "r:gz") as handle:
        # This is a local build output, but read back the archive's actual member
        # set before extraction so a packaging bug cannot escape the smoke root.
        for member in handle.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unexpected Android archive member: {member.name}")
        handle.extractall(destination)
    payload = destination / "Ouroboros-Android"
    manifest = json.loads((payload / "android_release_manifest.json").read_text())
    expected = manifest["files"]
    actual = {
        path.relative_to(payload).as_posix(): file_record(path)
        for path in sorted(payload.rglob("*"))
        if path.is_file() and path.name != "android_release_manifest.json"
    }
    if actual != expected:
        raise ValueError("Android archive source inventory does not match its manifest")
    bundle_manifest = json.loads((payload / "repo_bundle_manifest.json").read_text())
    if (
        bundle_manifest["source_sha"] != args.commit
        or bundle_manifest["release_tag"] != args.tag
        or manifest["sourceCommit"] != args.commit
        or manifest["releaseTag"] != args.tag
        or bundle_manifest["bundle_sha256"] != RELEASE["sha256_file"](payload / "repo.bundle")
    ):
        raise ValueError("Android archive does not match the exact release source")
    # Clone the final embedded bytes and compare HEAD, rather than merely testing
    # that git can list a bundle header.
    checkout = destination / "bundle-checkout"
    run(["git", "clone", "--no-checkout", payload / "repo.bundle", checkout])
    if run(["git", "rev-parse", "HEAD"], cwd=checkout).strip() != args.commit:
        raise ValueError("Android repository bundle HEAD differs from release source")
    run(["git", "checkout", "--detach", args.commit], cwd=checkout)
    run([sys.executable, payload / "android" / "install.py", "--help"], cwd=payload)
    reference = manifest["referenceApk"]
    identity = apk_identity(args, args.apk)
    if reference != {"name": args.apk.name, **file_record(args.apk), **identity}:
        raise ValueError("Final Android APK differs from the manifest-bound build")
    if identity["versionName"] != manifest["version"] or identity["versionCode"] != args.version_code:
        raise ValueError("Final APK version does not match the release")
    # The source SBOM inventories the actual embedded Git tree, not an opaque
    # repo.bundle file or the external SDK/rootfs provisioned at installation.
    args.sbom_root = checkout
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--keystore", type=Path, required=True)
    parser.add_argument("--keystore-pass-file", type=Path, required=True)
    parser.add_argument("--key-alias", default="ouroboros-host")
    parser.add_argument("--version-code", type=int, required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    if not args.keystore.is_file() or not args.keystore_pass_file.is_file():
        parser.error("release signing key and password file must already exist; no development fallback")
    if not 1 <= args.version_code <= 2_100_000_000:
        parser.error("version code must be between 1 and 2100000000")
    # Verify source/tag before a compiler or output directory can create dirt.
    BUNDLE["_ensure_clean_worktree"](ROOT)
    version = BUNDLE["_read_version"](ROOT)
    args.tag = BUNDLE["_resolve_release_tag"](ROOT, version)
    args.commit = BUNDLE["_git_output"](ROOT, "rev-parse", "HEAD")
    for key in ("out", "work", "sdk", "java_home", "keystore", "keystore_pass_file"):
        setattr(args, key, getattr(args, key).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="android-build-", dir=args.work) as temporary:
        work = Path(temporary)
        run([
            sys.executable, ROOT / "android" / "host" / "build.py",
            "--sdk", args.sdk, "--java-home", args.java_home, "--out", work / "apk",
            "--keystore", args.keystore, "--key-alias", args.key_alias,
            "--keystore-pass-file", args.keystore_pass_file,
            "--version-code", args.version_code, "--version-name", version,
        ])
        args.apk = args.out / RELEASE["release_asset_name"]("android-apk", version)
        shutil.copyfile(work / "apk" / "Ouroboros.apk", args.apk)
        identity = apk_identity(args, args.apk)
        package = ET.parse(ROOT / "android" / "host" / "AndroidManifest.xml").getroot().get("package")
        if identity["packageName"] != package:
            raise ValueError("APK package name differs from the Android host source")
        certificate = run([
            args.java_home / "bin" / "keytool", "-exportcert", "-keystore", args.keystore,
            "-alias", args.key_alias, "-storepass:file", args.keystore_pass_file,
        ], binary=True)
        if identity["signerSha256"] != hashlib.sha256(certificate).hexdigest():
            raise ValueError("APK was not signed by the supplied publisher certificate")
        stage = work / "payload"
        stage.mkdir()
        archive = create_archive(args, stage, version, identity)
    # Keep only the final extracted payload for the subsequent SBOM step, and
    # use a unique directory so a retry cannot blend old and new payload bytes.
    smoke_root = Path(tempfile.mkdtemp(prefix="android-smoke-", dir=args.work))
    inspect_archive(args, archive, smoke_root)
    for proof_id, artifact in (("android-arm64", archive), ("android-apk", args.apk)):
        RELEASE["command_record_smoke"](argparse.Namespace(
            proof_id=proof_id, artifact=artifact,
            output=args.out / f"release-smoke-{proof_id}.json",
            commit=args.commit, tag=args.tag, check=RELEASE["REQUIRED_SMOKE_CHECKS"][proof_id],
        ))
    values = {"archive": str(archive), "apk": str(args.apk), "sbom_path": str(args.sbom_root)}
    if args.github_output:
        RELEASE["_append_github_output"](args.github_output, values)
    print(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
