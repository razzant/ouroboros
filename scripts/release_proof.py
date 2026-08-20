#!/usr/bin/env python3
"""Build and verify the public proof capsule for an Ouroboros release."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import runpy
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
release_asset_download_url = _RELEASE_SYNC["release_asset_download_url"]
release_asset_name = _RELEASE_SYNC["release_asset_name"]


# One build job produces exactly one platform archive, so ``locate`` only ever
# looks at these suffixes. The AppImage and native Linux packages are produced
# after the tarball is located and are discovered separately.
ARCHIVE_SUFFIXES = (".dmg", ".tar.gz", ".zip")
RELEASE_ASSET_SUFFIXES = ARCHIVE_SUFFIXES + (".AppImage", ".deb", ".rpm")
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
}
RELEASE_GATES = (
    "full-test",
    "marker-guards",
    "ui-smoke",
    "docker-ui-smoke",
    "docker-portable-test",
    "skill-smoke",
    "packaged-artifact-smoke",
)
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
}


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
    if args.github_output:
        _append_github_output(args.github_output, values)
    print(json.dumps(values, sort_keys=True))


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


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _proof_files(
    directory: Path,
    version: str,
    *,
    commit: str,
    tag: str,
) -> list[dict]:
    archives = {
        path.name: path for path in _release_assets(directory, RELEASE_ASSET_SUFFIXES)
    }
    expected = {proof_id: factory(version) for proof_id, factory in PROOF_IDS.items()}
    if set(archives) != set(expected.values()):
        raise ValueError(
            "release asset set does not match the expected platform assets: "
            f"expected {sorted(expected.values())}, found {sorted(archives)}"
        )

    records: list[dict] = []
    for proof_id, artifact_name in expected.items():
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
        records.append(
            {
                "proofId": proof_id,
                "name": artifact.name,
                "size": artifact.stat().st_size,
                "sha256": digest,
                "smokeReceipt": smoke_path.name,
                "sbom": sbom_path.name,
            }
        )
    return records


def _checksum_targets(directory: Path, records: Iterable[dict]) -> list[Path]:
    paths: list[Path] = []
    for record in records:
        paths.extend(
            directory / name
            for name in (record["name"], record["smokeReceipt"], record["sbom"])
        )
    return sorted(paths, key=lambda path: path.name)


def _release_notes(
    *,
    version: str,
    description: str,
    repository: str,
    commit: str,
    tag: str,
    previous_tag: str | None,
    records: Iterable[dict],
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
        "Choose your platform below. You do not need to clone the repository or install Python or uv.",
        "",
    ]
    records_by_id = {
        str(record.get("proofId") or ""): record
        for record in records
    }
    for proof_id, label in DOWNLOAD_LABELS.items():
        record = records_by_id.get(proof_id)
        if not record:
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
    lines.extend([
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
        f"This release was built from [`{short_commit}`](https://github.com/{repository}/commit/{commit}).",
        "The release workflow passed the full test matrix, UI and Docker smoke tests, skill smoke tests, and packaged artifact smoke tests before publication.",
        "",
        "- `SHA256SUMS` covers every installable platform artifact, its SBOM, and its smoke receipt.",
        "- `release-evidence.json` binds the tag, commit, workflow run, artifact hashes, SBOMs, and smoke receipts.",
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
    records = _proof_files(
        args.directory,
        version,
        commit=args.commit,
        tag=args.tag,
    )
    checksum_targets = _checksum_targets(args.directory, records)
    checksums = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in checksum_targets
    )
    (args.directory / "SHA256SUMS").write_text(checksums, encoding="utf-8")

    generated_at = args.generated_at or dt.datetime.now(dt.timezone.utc).isoformat()
    evidence = {
        "schemaVersion": 1,
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
            "gates": [{"name": name, "status": "passed"} for name in RELEASE_GATES],
        },
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
        ),
        encoding="utf-8",
    )


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
    local_names = {
        path.name
        for path in args.directory.iterdir()
        if path.is_file() and path.name != args.metadata.name
    }
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
    locate.set_defaults(func=command_locate)

    smoke = commands.add_parser("record-smoke", help="write a passed smoke receipt")
    smoke.add_argument("--proof-id", required=True)
    smoke.add_argument("--artifact", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--commit", required=True)
    smoke.add_argument("--tag", required=True)
    smoke.add_argument("--check", action="append", default=[])
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
