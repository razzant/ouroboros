#!/usr/bin/env python3
"""Build and verify the public proof capsule for an Ouroboros release."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import runpy
import zipfile
from functools import partial
from pathlib import Path
from typing import Iterable


# This script runs in release jobs that intentionally install only Python, not
# the Ouroboros package. Load the shared release metadata by path so it remains
# standalone while still using the same naming source as the review gates.
_RELEASE_SYNC = runpy.run_path(
    Path(__file__).resolve().parents[1] / "ouroboros" / "tools" / "release_sync.py"
)
RELEASE_ASSET_TEMPLATES = _RELEASE_SYNC["RELEASE_ASSET_TEMPLATES"]
DESKTOP_DOWNLOAD_IDS = _RELEASE_SYNC["DESKTOP_DOWNLOAD_IDS"]
ANDROID_DOWNLOAD_IDS = _RELEASE_SYNC["ANDROID_DOWNLOAD_IDS"]
release_asset_download_url = _RELEASE_SYNC["release_asset_download_url"]
release_asset_name = _RELEASE_SYNC["release_asset_name"]


# One build job produces exactly one platform archive, so ``locate`` only ever
# looks at these suffixes. The AppImage and native Linux packages are produced
# after the tarball is located and are discovered separately.
ARCHIVE_SUFFIXES = (".dmg", ".tar.gz", ".zip")
RELEASE_ASSET_SUFFIXES = ARCHIVE_SUFFIXES + (".AppImage", ".deb", ".rpm", ".apk")
PROOF_IDS = {
    proof_id: partial(release_asset_name, proof_id)
    for proof_id in RELEASE_ASSET_TEMPLATES
}
DOWNLOAD_LABELS = {
    "macos-arm64": "macOS 12+ on Apple silicon (.dmg)",
    "windows-x64": "Windows x64 (.zip)",
    "linux-deb-amd64": "Debian, Ubuntu, or Astra Linux x86_64 (.deb)",
    "linux-rpm-x86_64": "Fedora or RHEL x86_64 (.rpm)",
    "linux-rpm-red80-x86_64": "RED OS 8 x86_64 (.rpm)",
    "linux-appimage-x86_64": "Other Linux x86_64 (AppImage)",
    "linux-x86_64": "Linux x86_64 archive (.tar.gz)",
    "android-arm64": "Experimental Android ARM64 USB setup (.tar.gz)",
    "android-apk": "Android publisher-signed reference APK (see setup guide)",
}
# The release job's needs apart from the optional Android build. The proof
# records each job's actual GitHub conclusion; a missing one is an error,
# never a pass.
RELEASE_GATES = (
    "full-test",
    "integration-test",
    "system-e2e-mock",
    "release-preflight",
    "build",
    "windows-proof",
    "marker-guards",
    "ui-smoke",
    "docker-ui-smoke",
    "docker-portable-test",
    "skill-smoke",
    "lint",
    "test-verdict",
)
JOB_RESULTS = ("success", "failure", "cancelled", "skipped")
# Owner decision 1A of 2026-10-21: every gate must succeed; nothing is waived by tag.
# A test that failed once and passed on its retry in the same job (FLAKY) and a
# failure the owner accepted as not a product bug (ACCEPTED) are already green
# inside their lanes; the run's `test-verdict` table names them and the notes
# disclose them. A FAILED or never-run test in that table refuses the release.
NOTES_FLAKY_SHOWN = 30
COMMON_SMOKE_CHECKS = frozenset(
    {
        "embedded_repo_bundle",
        "embedded_claudexor_runtime",
        "embedded_betterleaks_runtime",
        "packaged_cli_help",
    }
)
# The native packages are proven by a real install in a stock distro container,
# not by unpacking an archive, so they carry their own check set. Only the
# release-gating lane counts here: the Astra Linux and RED OS runs are
# informational and cannot be required of a receipt.
PACKAGE_SMOKE_CHECKS = frozenset(
    {
        "package_install",
        "runtime_dependency",
        "embedded_betterleaks_runtime",
        "packaged_cli_help",
        "desktop_entry",
        "systemd_user_unit",
        "desktop_launcher_start",
    }
)
REQUIRED_SMOKE_CHECKS = {
    "macos-arm64": COMMON_SMOKE_CHECKS
    | frozenset(
        {"applications_shortcut", "install_cli_command", "arm64_main_executable"}
    ),
    "linux-x86_64": COMMON_SMOKE_CHECKS,
    "linux-appimage-x86_64": COMMON_SMOKE_CHECKS
    | frozenset(
        {
            "appimage_extract_and_run",
            "appimage_metadata",
            "browser_fallback_start",
            "clean_shutdown",
            "gateway_readiness",
            "product_version",
            "shared_libraries",
        }
    ),
    "linux-deb-amd64": PACKAGE_SMOKE_CHECKS,
    "linux-rpm-x86_64": PACKAGE_SMOKE_CHECKS,
    "linux-rpm-red80-x86_64": PACKAGE_SMOKE_CHECKS,
    "windows-x64": COMMON_SMOKE_CHECKS,
    "android-arm64": frozenset({
        "embedded_repo_bundle", "android_source_manifest", "usb_installer_help",
    }),
    "android-apk": frozenset({"apk_signature", "apk_package_version"}),
}
# Windows signing is selected by configuration: without a signer thumbprint the
# ZIP ships explicitly unsigned; with one, the receipt must name that verified
# signer, carry these checks and hash exactly the catalogued executables before
# and after signing, or the whole release stops.
AUTHENTICODE_PROOF_ID = "windows-x64"
AUTHENTICODE_SMOKE_CHECKS = frozenset(
    {"authenticode_signer", "timestamp", "signed_payload_archive_match"}
)
WINDOWS_SIGNING_CATALOG = json.loads(
    (Path(__file__).resolve().parent / "windows_signing_catalog.json").read_text(encoding="utf-8")
)
# Like Windows signing, configuration selects the macOS state: a signed stable
# build must prove both stapled tickets and Gatekeeper acceptance of the
# copied-out app or the release stops; every other build is not_notarized.
NOTARIZATION_PROOF_ID = "macos-arm64"
NOTARIZATION_SMOKE_CHECKS = frozenset(
    {"stapled_app", "stapled_dmg", "gatekeeper_app", "gatekeeper_dmg"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_assets(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    # The .deb and .rpm follow distro naming, which is lowercase.
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.name.lower().startswith("ouroboros")
        and path.name.endswith(suffixes)
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_github_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"GitHub output {key!r} contains a newline")
            handle.write(f"{key}={value}\n")


def locate_artifact(directory: Path) -> Path:
    archives = _release_assets(directory, ARCHIVE_SUFFIXES)
    if len(archives) != 1:
        names = ", ".join(path.name for path in archives) or "none"
        raise ValueError(f"expected exactly one release archive in {directory}, found: {names}")
    return archives[0]


def command_locate(args: argparse.Namespace) -> None:
    artifact = locate_artifact(args.directory)
    values = {
        "path": artifact.as_posix(),
        "name": artifact.name,
        "sha256": sha256_file(artifact),
    }
    # A handed-over archive is checked against its producer job's output
    # before anything extracts or executes it.
    if args.expect_sha256 is not None and values["sha256"] != args.expect_sha256:
        raise ValueError(
            f"{artifact.name} has SHA-256 {values['sha256']}, not the "
            f"{args.expect_sha256 or '(missing)'} digest its producer job recorded"
        )
    if args.github_output:
        _append_github_output(args.github_output, values)
    print(json.dumps(values, sort_keys=True))


def _signed_files(records: object) -> list[dict]:
    """Each catalogued executable exactly once, with distinct hashes before and
    after signing; an upstream-pinned file must have been signed from its pin."""
    catalog = {entry["path"]: entry for entry in WINDOWS_SIGNING_CATALOG["files"]}
    if not isinstance(records, list) or not all(
        isinstance(row, dict) and set(row) == {"path", "unsignedSha256", "sha256"}
        and all(isinstance(value, str) for value in row.values())
        for row in records
    ):
        raise ValueError("Windows signing record must give path, unsignedSha256 and sha256 per file")
    paths = [row["path"] for row in records]
    if sorted(paths) != sorted(catalog):
        raise ValueError(f"Windows signing record covers {paths}, not exactly the catalogued {list(catalog)}")
    for row in records:
        pin = catalog[row["path"]].get("upstreamSha256")
        if not all(re.fullmatch(r"[0-9a-f]{64}", row[key]) for key in ("unsignedSha256", "sha256")) or (
            row["unsignedSha256"] == row["sha256"]
        ):
            raise ValueError(f"Windows signing record for {row['path']} lacks distinct SHA-256 digests")
        if pin and row["unsignedSha256"] != pin:
            raise ValueError(
                f"{row['path']} was signed from SHA-256 {row['unsignedSha256']}, not the catalogued upstream {pin}"
            )
    return sorted(records, key=lambda row: list(catalog).index(row["path"]))


def _archived_signed_files(artifact: Path, record: str) -> list[dict]:
    """The signing job's record, bound to the bytes the final ZIP holds."""
    files = _signed_files(json.loads(record or "null"))
    try:
        with zipfile.ZipFile(artifact) as archive:
            for row in files:
                digest = hashlib.sha256()
                with archive.open(row["path"]) as member:
                    for chunk in iter(lambda: member.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError(
                        f"{row['path']} in {artifact.name} has SHA-256 {digest.hexdigest()}, "
                        f"not the signed {row['sha256']}"
                    )
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{artifact.name} does not hold every catalogued executable: {exc}") from exc
    return files


def command_record_smoke(args: argparse.Namespace) -> None:
    if args.proof_id not in PROOF_IDS:
        raise ValueError(f"unsupported proof id: {args.proof_id}")
    if not args.artifact.is_file():
        raise ValueError(f"artifact does not exist: {args.artifact}")
    if not args.check:
        raise ValueError("at least one completed smoke check is required")
    receipt = {
        "schemaVersion": 1,
        "kind": "packaged_artifact_smoke",
        "status": "passed",
        "proofId": args.proof_id,
        "artifact": args.artifact.name,
        "sha256": sha256_file(args.artifact),
        "sourceCommit": args.commit,
        "releaseTag": args.tag,
        "checks": sorted(set(args.check)),
    }
    if args.proof_id == AUTHENTICODE_PROOF_ID:
        receipt["authenticode"] = {"status": "unsigned"}
        if args.authenticode_thumbprint:
            receipt["authenticode"] = {
                "status": "signed",
                "signerThumbprint": args.authenticode_thumbprint.upper(),
                "publisher": args.authenticode_publisher,
                "files": _archived_signed_files(args.artifact, args.authenticode_files),
                "pythonUpstream": WINDOWS_SIGNING_CATALOG["pythonUpstream"],
            }
        elif args.authenticode_files:
            raise ValueError("signed Windows file hashes need the verified signer")
    if args.proof_id == NOTARIZATION_PROOF_ID:
        receipt["notarization"] = {"status": args.macos_notarization or "not_notarized"}
    _write_json(args.output, receipt)


def _read_release_description(readme: Path, version: str) -> tuple[str, str]:
    prefix = f"| {version} |"
    for line in readme.read_text(encoding="utf-8").splitlines():
        if not line.startswith(prefix):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            break
        description = re.sub(r"\*\*", "", cells[2]).strip()
        return cells[1], description
    raise ValueError(f"README Version History has no exact row for {version}")


def _gate_results(pairs: list[str]) -> dict[str, str]:
    """Return every gate's actual conclusion; each must have succeeded."""
    results: dict[str, str] = {}
    for pair in pairs:
        name, _, result = pair.partition("=")
        if name not in RELEASE_GATES or name in results or result not in JOB_RESULTS:
            raise ValueError(f"invalid release gate result: {pair!r}")
        results[name] = result
    missing = [name for name in RELEASE_GATES if name not in results]
    if missing:
        raise ValueError(f"missing release gate results: {missing}")
    ordered = {name: results[name] for name in RELEASE_GATES}
    for name, result in ordered.items():
        if result != "success":
            raise ValueError(f"release gate {name} concluded {result}, not success")
    return ordered


def _test_verdict(path: Path) -> dict[str, list[dict]]:
    """The run's FLAKY and ACCEPTED tests from `tests/ci_evidence.py run-verdict`.

    Green gates and a FAILED or never-run row cannot both be true: the table is refused.
    """
    table = _load_json(path)
    rows = {}
    for key in ("flaky", "accepted", "failed", "not_run"):
        value = table.get(key)
        if not isinstance(value, list) or not all(isinstance(row, dict) and isinstance(row.get("nodeid"), str)
                                                  for row in value):
            raise ValueError(f"test verdict {path} has no valid {key!r} list")
        rows[key] = value
    if rows["failed"] or rows["not_run"]:
        raise ValueError(f"test verdict lists {len(rows['failed'])} failed and "
                         f"{len(rows['not_run'])} never-run tests; the release is refused")
    return {"flaky": rows["flaky"], "accepted": rows["accepted"]}


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _authenticode_state(smoke: dict, checks: list, signer_thumbprint: str) -> dict:
    state = smoke.get("authenticode")
    claimed = AUTHENTICODE_SMOKE_CHECKS & set(checks)
    if not signer_thumbprint:
        if state != {"status": "unsigned"} or claimed:
            raise ValueError(
                "Windows signing is not configured, but the windows-x64 receipt "
                "does not record an unsigned archive"
            )
        return state
    expected = signer_thumbprint.upper()
    state = state if isinstance(state, dict) else {}
    publisher = state.get("publisher")
    if (
        state.get("status") != "signed"
        or state.get("signerThumbprint") != expected
        or not isinstance(publisher, str)
        or not publisher.strip()
    ):
        raise ValueError(
            f"windows-x64 receipt does not record a signature by the configured certificate {expected}"
        )
    missing = AUTHENTICODE_SMOKE_CHECKS - claimed
    if missing:
        raise ValueError(f"windows-x64 receipt is missing signing checks: {sorted(missing)}")
    if state.get("pythonUpstream") != WINDOWS_SIGNING_CATALOG["pythonUpstream"]:
        raise ValueError("windows-x64 receipt does not name the catalogued upstream Python")
    return {"status": "signed", "signerThumbprint": expected, "publisher": publisher,
            "files": _signed_files(state.get("files")), "pythonUpstream": state["pythonUpstream"]}


def _notarization_state(smoke: dict, checks: list) -> dict:
    state = smoke.get("notarization")
    claimed = NOTARIZATION_SMOKE_CHECKS & set(checks)
    if state == {"status": "notarized"} and claimed == NOTARIZATION_SMOKE_CHECKS:
        return state
    if state == {"status": "not_notarized"} and not claimed:
        return state
    raise ValueError(
        "macos-arm64 receipt does not record a consistent notarization state: a notarized "
        f"DMG needs the checks {sorted(NOTARIZATION_SMOKE_CHECKS)}, an unnotarized one claims none"
    )


def _proof_files(
    directory: Path,
    version: str,
    *,
    commit: str,
    tag: str,
    android_build_result: str,
    android_attestation_result: str,
    windows_signer_thumbprint: str,
) -> tuple[list[dict], dict]:
    archives = {
        path.name: path for path in _release_assets(directory, RELEASE_ASSET_SUFFIXES)
    }
    expected = {proof_id: factory(version) for proof_id, factory in PROOF_IDS.items()}
    required = {expected[proof_id] for proof_id in DESKTOP_DOWNLOAD_IDS}
    android_suffixes = tuple(RELEASE_ASSET_TEMPLATES[key].split("{version}", 1)[1]
                             for key in ANDROID_DOWNLOAD_IDS)
    android_names = {name for name in archives if name.endswith(android_suffixes)}
    if not required <= archives.keys() or set(archives) - required - android_names:
        raise ValueError(
            "release asset set does not match the expected platform assets: "
            f"required {sorted(required)}, found {sorted(archives)}"
        )

    android = {"status": "unavailable", "buildResult": android_build_result,
               "attestationResult": android_attestation_result, "reason": ""}
    selected = list(DESKTOP_DOWNLOAD_IDS)
    if android_build_result != "success":
        android["reason"] = f"Android build result: {android_build_result}"
    elif android_names != {expected[key] for key in ANDROID_DOWNLOAD_IDS}:
        android["reason"] = "Android source archive and reference APK are not a complete matching pair"
    elif android_attestation_result != "success":
        android["reason"] = f"Android artifact attestation result: {android_attestation_result}"
    else:
        selected.extend(ANDROID_DOWNLOAD_IDS)
    records: list[dict] = []
    for proof_id in selected:
        artifact_name = expected[proof_id]
        try:
            artifact = archives[artifact_name]
            digest = sha256_file(artifact)
            smoke_path = directory / f"release-smoke-{proof_id}.json"
            sbom_path = directory / f"sbom-{proof_id}.cdx.json"
            if not smoke_path.is_file() or not sbom_path.is_file():
                raise ValueError(f"missing smoke receipt or SBOM for {proof_id}")
            smoke = _load_json(smoke_path)
            sbom = _load_json(sbom_path)
            if smoke.get("status") != "passed":
                raise ValueError(f"smoke receipt is not passed: {smoke_path}")
            expected_identity = {
                "schemaVersion": 1,
                "kind": "packaged_artifact_smoke",
                "proofId": proof_id,
                "sourceCommit": commit,
                "releaseTag": tag,
            }
            if any(smoke.get(key) != value for key, value in expected_identity.items()):
                raise ValueError(f"smoke receipt identity does not match {proof_id}")
            if smoke.get("artifact") != artifact.name or smoke.get("sha256") != digest:
                raise ValueError(f"smoke receipt is not bound to {artifact.name}")
            checks = smoke.get("checks")
            if not isinstance(checks, list) or not all(isinstance(item, str) for item in checks):
                raise ValueError(f"smoke receipt checks are invalid: {smoke_path}")
            missing_checks = REQUIRED_SMOKE_CHECKS[proof_id] - set(checks)
            if missing_checks:
                raise ValueError(
                    f"smoke receipt is missing required checks for {proof_id}: "
                    f"{sorted(missing_checks)}"
                )
            if (
                sbom.get("bomFormat") != "CycloneDX"
                or not isinstance(sbom.get("specVersion"), str)
                or not isinstance(sbom.get("serialNumber"), str)
            ):
                raise ValueError(f"SBOM is not CycloneDX JSON: {sbom_path}")
            record = {
                "proofId": proof_id,
                "name": artifact.name,
                "size": artifact.stat().st_size,
                "sha256": digest,
                "smokeReceipt": smoke_path.name,
                "sbom": sbom_path.name,
            }
            if proof_id == AUTHENTICODE_PROOF_ID:
                record["authenticode"] = _authenticode_state(
                    smoke, checks, windows_signer_thumbprint
                )
            if proof_id == NOTARIZATION_PROOF_ID:
                record["notarization"] = _notarization_state(smoke, checks)
            records.append(record)
        except (OSError, ValueError) as exc:
            if proof_id not in ANDROID_DOWNLOAD_IDS:
                raise
            # Android is one optional pair: never retain just its successful half.
            records = [row for row in records if row["proofId"] not in ANDROID_DOWNLOAD_IDS]
            android["reason"] = str(exc)
            break
    if sum(row["proofId"] in ANDROID_DOWNLOAD_IDS for row in records) == len(ANDROID_DOWNLOAD_IDS):
        android["status"] = "verified"
    return records, android


def _checksum_targets(directory: Path, records: Iterable[dict]) -> list[Path]:
    paths: list[Path] = []
    for record in records:
        paths.extend(
            directory / name
            for name in (record["name"], record["smokeReceipt"], record["sbom"])
        )
    return sorted(paths, key=lambda path: path.name)


def _verdict_lines(verdict: dict[str, list[dict]], repository: str) -> list[str]:
    """Release-note lines for the run's FLAKY and ACCEPTED tests (none: no lines)."""
    lines = []
    flaky, accepted = verdict["flaky"], verdict["accepted"]
    if accepted:
        lines.extend(["", "Known failures the project owner accepted as not product bugs "
                      "(each issue stays open until it is fixed):"])
        lines.extend(f"- `{row['nodeid']}` — [#{row.get('issue')}](https://github.com/{repository}/issues/{row.get('issue')})"
                     for row in accepted)
    if flaky:
        lines.extend(["", f"{len(flaky)} test(s) failed once and passed on their retry in the same job (flaky):"])
        lines.extend(f"- `{row['nodeid']}`" + (f" ({row['label']})" if row.get("label") else "")
                     for row in flaky[:NOTES_FLAKY_SHOWN])
        if len(flaky) > NOTES_FLAKY_SHOWN:
            lines.append(f"- … and {len(flaky) - NOTES_FLAKY_SHOWN} more, listed in `release-evidence.json`.")
    return lines


def _release_notes(
    *,
    version: str,
    description: str,
    repository: str,
    commit: str,
    tag: str,
    previous_tag: str | None,
    records: Iterable[dict],
    android: dict,
    run_url: str,
    verdict: dict[str, list[dict]],
) -> str:
    short_commit = commit[:12]
    verify_base = (
        f"gh attestation verify <file> --repo {repository} "
        f"--signer-workflow {repository}/.github/workflows/ci.yml "
        f"--source-digest {commit} --source-ref refs/tags/{tag}"
    )
    lines = [
        f"# Ouroboros {tag}",
        "",
        "## Download",
        "",
        "Choose your platform below. Desktop installers do not require Python or uv. "
        "Experimental Android uses the USB setup guide on an already Magisk-rooted ARM64 device.",
        "",
    ]
    records_by_id = {
        str(record.get("proofId") or ""): record
        for record in records
    }
    for proof_id, label in DOWNLOAD_LABELS.items():
        record = records_by_id.get(proof_id)
        if not record:
            if proof_id in ANDROID_DOWNLOAD_IDS:
                continue
            raise ValueError(f"release notes missing verified asset: {proof_id}")
        name = str(record.get("name") or "")
        expected_name = release_asset_name(proof_id, version)
        if name != expected_name:
            raise ValueError(
                f"release notes asset mismatch for {proof_id}: {name} != {expected_name}"
            )
        url = release_asset_download_url(
            proof_id,
            version,
            repository=repository,
        )
        lines.append(f"- **{label}:** [{name}]({url})")
    authenticode = records_by_id[AUTHENTICODE_PROOF_ID]["authenticode"]
    if authenticode["status"] == "signed":
        python = authenticode["pythonUpstream"]
        lines.extend([
            "",
            "Windows: `Ouroboros.exe` and the bundled interpreter's `python.exe` and `pythonw.exe` "
            f"(CPython {python['pythonVersion']} from python-build-standalone {python['release']}) "
            "carry timestamped Authenticode signatures "
            f"from **{authenticode['publisher']}** (certificate SHA-1 "
            f"`{authenticode['signerThumbprint']}`); `release-evidence.json` lists each file's "
            "SHA-256 before and after signing. This release signs no other file in the ZIP, "
            "no Python environment later created from that interpreter and no code that later "
            "Git updates bring, and SmartScreen or antivirus may still warn while the "
            "publisher's reputation builds.",
        ])
    else:
        lines.extend([
            "",
            "Windows: this ZIP is unsigned. The release was built without a configured "
            "code-signing certificate, so Windows reports an unknown publisher; check the "
            "download against `SHA256SUMS` and its attestations below.",
        ])
    if records_by_id[NOTARIZATION_PROOF_ID]["notarization"]["status"] == "notarized":
        lines.extend([
            "",
            "macOS: Apple notarized both the app and the DMG, each carries its stapled "
            "ticket, and Gatekeeper accepted the app copied out of the DMG before publication.",
        ])
    else:
        lines.extend([
            "",
            "macOS: this DMG is not notarized by Apple, so on macOS 15 and later open the "
            "app once, then choose Open Anyway in System Settings → Privacy & Security.",
        ])
    if android["status"] != "verified":
        lines.extend([
            "",
            "Experimental Android artifacts are unavailable for this release: "
            f"{android['reason']}. [CI run]({run_url}). "
            "The independently verified desktop installers remain available above.",
        ])
    lines.extend([
        "",
        f"[Android setup guide](https://github.com/{repository}/blob/{tag}/docs/ANDROID_INSTALL.md): "
        "the installer builds the installed host with a persistent personal signing key. "
        "Installing the reference APK alone does not provision the Linux runtime. "
        "CI artifact checks do not certify root, boot, hardware, or phone runtime behavior.",
        "",
        "Files named `SHA256SUMS`, `release-evidence.json`, `release-smoke-*.json`, "
        "and `sbom-*.cdx.json` are verification evidence, not additional installers.",
        "",
        "## What's new",
        "",
        description,
        "",
        "## Release proof",
        "",
        f"This release was built from [`{short_commit}`](https://github.com/{repository}/commit/{commit}) "
        f"by [this workflow run]({run_url}).",
        "The release workflow passed the full test matrix, required provider integration, "
        "system E2E, UI and Docker smoke tests, skill smoke tests, and packaged artifact "
        "smoke tests before publication.",
        *_verdict_lines(verdict, repository),
        "",
        "- `SHA256SUMS` covers every installable platform artifact, its SBOM, and its smoke receipt.",
        "- `release-evidence.json` binds the tag, commit, workflow run, each release job's actual conclusion, artifact hashes, SBOMs, and smoke receipts.",
        "- Each installable platform artifact has GitHub build provenance and a CycloneDX SBOM attestation.",
        f"- Verify build provenance with `{verify_base}`.",
        f"- Verify the CycloneDX attestation with `{verify_base} --predicate-type https://cyclonedx.org/bom`.",
    ])
    if previous_tag:
        lines.extend(
            [
                "",
                f"[Compare {previous_tag}...{tag}](https://github.com/{repository}/compare/{previous_tag}...{tag})",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def command_assemble(args: argparse.Namespace) -> None:
    version = args.version_file.read_text(encoding="utf-8").strip()
    if args.tag != f"v{version}":
        raise ValueError(f"tag/version mismatch: {args.tag} != v{version}")
    release_date, description = _read_release_description(args.readme, version)
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("commit must be a full lowercase Git SHA")
    # Set but malformed is a broken configuration, never an unsigned release.
    if args.windows_signer_thumbprint and not re.fullmatch(
        r"[0-9A-Fa-f]{40}", args.windows_signer_thumbprint
    ):
        raise ValueError("ESIGNER_CERT_SHA1 is set but is not a 40-hex SHA-1 thumbprint")
    gates = _gate_results(args.gate_result)
    verdict = _test_verdict(args.test_verdict)
    records, android = _proof_files(
        args.directory,
        version,
        commit=args.commit,
        tag=args.tag,
        android_build_result=args.android_build_result,
        android_attestation_result=args.android_attestation_result,
        windows_signer_thumbprint=args.windows_signer_thumbprint,
    )
    checksum_targets = _checksum_targets(args.directory, records)
    checksums = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in checksum_targets
    )
    (args.directory / "SHA256SUMS").write_text(checksums, encoding="utf-8")

    generated_at = args.generated_at or dt.datetime.now(dt.timezone.utc).isoformat()
    evidence = {
        # 3: gates carry actual job conclusions; the run's FLAKY and ACCEPTED tests are recorded.
        "schemaVersion": 3,
        "kind": "build_time_release_proof",
        "product": "Ouroboros",
        "version": version,
        "releaseDate": release_date,
        "source": {
            "repository": f"https://github.com/{args.repository}",
            "tag": args.tag,
            "commit": args.commit,
        },
        "workflow": {
            "runUrl": args.run_url,
            "gates": [
                *({"name": name, "status": status} for name, status in gates.items()),
                # Verified above from every selected asset's bound receipt.
                {"name": "packaged-artifact-smoke", "status": "passed"},
                {"name": "android-build", "status": args.android_build_result},
            ],
            "testVerdict": verdict,
        },
        "experimentalAndroid": android,
        "generatedAt": generated_at,
        "artifacts": records,
        "verification": {
            "checksums": "SHA256SUMS",
            "attestationCommands": [
                (
                    f"gh attestation verify <file> --repo {args.repository} "
                    f"--signer-workflow {args.repository}/.github/workflows/ci.yml "
                    f"--source-digest {args.commit} --source-ref refs/tags/{args.tag}"
                ),
                (
                    f"gh attestation verify <file> --repo {args.repository} "
                    f"--signer-workflow {args.repository}/.github/workflows/ci.yml "
                    f"--source-digest {args.commit} --source-ref refs/tags/{args.tag} "
                    "--predicate-type https://cyclonedx.org/bom"
                ),
            ],
        },
    }
    _write_json(args.directory / "release-evidence.json", evidence)
    args.notes_output.write_text(
        _release_notes(
            version=version,
            description=description,
            repository=args.repository,
            commit=args.commit,
            tag=args.tag,
            previous_tag=args.previous_tag,
            records=records,
            android=android,
            run_url=args.run_url,
            verdict=verdict,
        ),
        encoding="utf-8",
    )
    if args.github_output:
        files = [*_checksum_targets(args.directory, records),
                 args.directory / "SHA256SUMS", args.directory / "release-evidence.json"]
        _append_github_output(args.github_output, {
            "files_json": json.dumps("\n".join(path.as_posix() for path in files)),
        })


def command_verify_uploaded(args: argparse.Namespace) -> None:
    metadata = _load_json(args.metadata)
    remote_rows = metadata.get("assets")
    if not isinstance(remote_rows, list):
        raise ValueError("release metadata has no assets list")
    remote = {
        row.get("name"): row
        for row in remote_rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    evidence = _load_json(args.directory / "release-evidence.json")
    records = evidence.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("release evidence has no proof-accepted artifact list")
    local_names = {path.name for path in _checksum_targets(args.directory, records)}
    local_names.update({"SHA256SUMS", "release-evidence.json"})
    if set(remote) != local_names:
        raise ValueError(
            "uploaded asset set differs from the local allowlist: "
            f"local={sorted(local_names)}, remote={sorted(remote)}"
        )
    for name in sorted(local_names):
        path = args.directory / name
        row = remote[name]
        if row.get("size") != path.stat().st_size:
            raise ValueError(f"uploaded size mismatch for {name}")
        if row.get("digest") != f"sha256:{sha256_file(path)}":
            raise ValueError(f"uploaded digest mismatch for {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    locate = commands.add_parser("locate", help="locate and hash one built archive")
    locate.add_argument("--directory", type=Path, default=Path("dist"))
    locate.add_argument("--github-output", type=Path)
    locate.add_argument("--expect-sha256", help="fail unless the archive has this digest")
    locate.set_defaults(func=command_locate)

    smoke = commands.add_parser("record-smoke", help="write a passed smoke receipt")
    smoke.add_argument("--proof-id", required=True)
    smoke.add_argument("--artifact", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--commit", required=True)
    smoke.add_argument("--tag", required=True)
    smoke.add_argument("--check", action="append", default=[])
    smoke.add_argument("--authenticode-thumbprint", default="")
    smoke.add_argument("--authenticode-publisher", default="")
    # The windows-sign job's signed_files output: JSON rows of path, unsignedSha256, sha256.
    smoke.add_argument("--authenticode-files", default="")
    smoke.add_argument("--macos-notarization", choices=("notarized", "not_notarized"))
    smoke.set_defaults(func=command_record_smoke)

    assemble = commands.add_parser("assemble", help="assemble the release proof capsule")
    assemble.add_argument("--directory", type=Path, required=True)
    assemble.add_argument("--version-file", type=Path, default=Path("VERSION"))
    assemble.add_argument("--readme", type=Path, default=Path("README.md"))
    assemble.add_argument("--repository", required=True)
    assemble.add_argument("--tag", required=True)
    assemble.add_argument("--commit", required=True)
    assemble.add_argument("--run-url", required=True)
    assemble.add_argument("--previous-tag")
    assemble.add_argument("--generated-at")
    assemble.add_argument("--notes-output", type=Path, required=True)
    assemble.add_argument("--android-build-result", choices=("success", "failure", "cancelled", "skipped", "not_run"), default="not_run")
    assemble.add_argument("--android-attestation-result", choices=("success", "failure", "not_run"), default="not_run")
    # Required, possibly empty: the release states which Windows mode it expects.
    assemble.add_argument("--windows-signer-thumbprint", required=True)
    # One per RELEASE_GATES job, its actual result: ``--gate-result full-test=failure``.
    assemble.add_argument("--gate-result", action="append", default=[], metavar="JOB=RESULT")
    # The `test-verdict` job's table (tests/ci_evidence.py run-verdict): FLAKY and ACCEPTED rows go to the notes.
    assemble.add_argument("--test-verdict", type=Path, required=True)
    assemble.add_argument("--github-output", type=Path)
    assemble.set_defaults(func=command_assemble)

    verify = commands.add_parser(
        "verify-uploaded", help="verify a draft release against the local allowlist"
    )
    verify.add_argument("--directory", type=Path, required=True)
    verify.add_argument("--metadata", type=Path, required=True)
    verify.set_defaults(func=command_verify_uploaded)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"release proof error: {exc}") from exc


if __name__ == "__main__":
    main()
