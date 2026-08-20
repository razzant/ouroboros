from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_proof", REPO / "scripts" / "release_proof.py"
)
assert SPEC and SPEC.loader
release_proof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_proof)


def test_release_proof_remains_runnable_without_installed_package():
    result = subprocess.run(
        [sys.executable, "-S", str(REPO / "scripts" / "release_proof.py"), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_release(tmp_path: Path, version: str = "6.87.5") -> tuple[Path, Path, Path]:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    version_file = tmp_path / "VERSION"
    version_file.write_text(f"{version}\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Version History\n\n"
        "| Version | Date | Description |\n"
        "|---|---|---|\n"
        f"| {version} | 2026-08-01 | **A clear release note.** |\n",
        encoding="utf-8",
    )
    for proof_id, name_factory in release_proof.PROOF_IDS.items():
        artifact = release_dir / name_factory(version)
        artifact.write_bytes(f"archive:{proof_id}".encode())
        receipt = {
            "schemaVersion": 1,
            "kind": "packaged_artifact_smoke",
            "status": "passed",
            "proofId": proof_id,
            "artifact": artifact.name,
            "sha256": _digest(artifact),
            "sourceCommit": "a" * 40,
            "releaseTag": f"v{version}",
            "checks": sorted(release_proof.REQUIRED_SMOKE_CHECKS[proof_id]),
        }
        (release_dir / f"release-smoke-{proof_id}.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        (release_dir / f"sbom-{proof_id}.cdx.json").write_text(
            json.dumps(
                {
                    "bomFormat": "CycloneDX",
                    "specVersion": "1.6",
                    "serialNumber": f"urn:uuid:{proof_id}",
                }
            ),
            encoding="utf-8",
        )
    return release_dir, version_file, readme


def test_locate_artifact_requires_exactly_one_archive(tmp_path: Path):
    one = tmp_path / "Ouroboros-1.0.0.dmg"
    one.write_bytes(b"one")
    assert release_proof.locate_artifact(tmp_path) == one
    (tmp_path / "Ouroboros-1.0.0.zip").write_bytes(b"two")
    with pytest.raises(ValueError, match="exactly one"):
        release_proof.locate_artifact(tmp_path)


def test_locate_artifact_ignores_companion_linux_assets(tmp_path: Path):
    archive = tmp_path / "Ouroboros-1.0.0-linux-x86_64.tar.gz"
    archive.write_bytes(b"archive")
    (tmp_path / "ouroboros_1.0.0_amd64.deb").write_bytes(b"deb")
    (tmp_path / "ouroboros-1.0.0-1.x86_64.rpm").write_bytes(b"rpm")
    (tmp_path / "ouroboros-1.0.0-1.red80.x86_64.rpm").write_bytes(b"rpm")
    (tmp_path / "Ouroboros-1.0.0-linux-x86_64.AppImage").write_bytes(b"appimage")
    assert release_proof.locate_artifact(tmp_path) == archive


def test_assemble_binds_every_asset_smoke_and_sbom(tmp_path: Path):
    release_dir, version_file, readme = _fixture_release(tmp_path)
    notes = tmp_path / "notes.md"
    args = argparse.Namespace(
        directory=release_dir,
        version_file=version_file,
        readme=readme,
        repository="razzant/ouroboros",
        tag="v6.87.5",
        commit="a" * 40,
        run_url="https://github.com/razzant/ouroboros/actions/runs/1",
        previous_tag="v6.87.4",
        generated_at="2026-08-02T00:00:00+00:00",
        notes_output=notes,
    )
    release_proof.command_assemble(args)

    evidence = json.loads((release_dir / "release-evidence.json").read_text())
    assert evidence["source"]["commit"] == "a" * 40
    assert len(evidence["artifacts"]) == 7
    assert {row["proofId"] for row in evidence["artifacts"]} == set(
        release_proof.PROOF_IDS
    )
    assert {row["name"] for row in evidence["artifacts"]} >= {
        "ouroboros_6.87.5_amd64.deb",
        "ouroboros-6.87.5-1.x86_64.rpm",
        "ouroboros-6.87.5-1.red80.x86_64.rpm",
        "Ouroboros-6.87.5-linux-x86_64.AppImage",
    }
    # One archive + one smoke receipt + one SBOM per proof id.
    checksum_lines = (release_dir / "SHA256SUMS").read_text().splitlines()
    assert len(checksum_lines) == 3 * len(release_proof.PROOF_IDS)
    assert checksum_lines == sorted(checksum_lines, key=lambda line: line.split("  ", 1)[1])
    notes_text = notes.read_text()
    assert "A clear release note." in notes_text
    assert "## Download" in notes_text
    assert "verification evidence, not additional installers" in notes_text
    assert "every installable platform artifact, its SBOM, and its smoke receipt" in notes_text
    assert "Each installable platform artifact has GitHub build provenance" in notes_text
    assert "/releases/latest" not in notes_text
    for proof_id in release_proof.PROOF_IDS:
        assert release_proof.release_asset_download_url(
            proof_id,
            "6.87.5",
            repository="razzant/ouroboros",
        ) in notes_text
    assert "v6.87.4...v6.87.5" in notes_text
    commands = evidence["verification"]["attestationCommands"]
    assert len(commands) == 2
    assert all("--source-digest " + "a" * 40 in command for command in commands)
    assert all("--source-ref refs/tags/v6.87.5" in command for command in commands)
    assert "--predicate-type https://cyclonedx.org/bom" in commands[1]


def test_prerelease_notes_link_to_the_exact_prerelease_assets(tmp_path: Path):
    version = "6.87.5-rc.1"
    release_dir, version_file, readme = _fixture_release(tmp_path, version=version)
    notes = tmp_path / "notes.md"
    args = argparse.Namespace(
        directory=release_dir,
        version_file=version_file,
        readme=readme,
        repository="razzant/ouroboros",
        tag=f"v{version}",
        commit="a" * 40,
        run_url="https://github.com/razzant/ouroboros/actions/runs/1",
        previous_tag="v6.87.4",
        generated_at="2026-08-02T00:00:00+00:00",
        notes_output=notes,
    )

    release_proof.command_assemble(args)

    text = notes.read_text(encoding="utf-8")
    assert f"/releases/download/v{version}/" in text
    assert "/releases/latest" not in text


def test_assemble_rejects_smoke_digest_drift(tmp_path: Path):
    release_dir, version_file, readme = _fixture_release(tmp_path)
    receipt_path = release_dir / "release-smoke-macos-arm64.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    args = argparse.Namespace(
        directory=release_dir,
        version_file=version_file,
        readme=readme,
        repository="razzant/ouroboros",
        tag="v6.87.5",
        commit="a" * 40,
        run_url="https://example.test/run",
        previous_tag=None,
        generated_at="2026-08-02T00:00:00+00:00",
        notes_output=tmp_path / "notes.md",
    )
    with pytest.raises(ValueError, match="not bound"):
        release_proof.command_assemble(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("proofId", "linux-x86_64", "identity"),
        ("sourceCommit", "b" * 40, "identity"),
        ("releaseTag", "v6.87.4", "identity"),
        ("checks", ["packaged_cli_help"], "missing required checks"),
    ],
)
def test_assemble_rejects_unbound_or_incomplete_smoke_receipt(
    tmp_path: Path, field: str, value: object, message: str
):
    release_dir, version_file, readme = _fixture_release(tmp_path)
    receipt_path = release_dir / "release-smoke-macos-arm64.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    args = argparse.Namespace(
        directory=release_dir,
        version_file=version_file,
        readme=readme,
        repository="razzant/ouroboros",
        tag="v6.87.5",
        commit="a" * 40,
        run_url="https://example.test/run",
        previous_tag=None,
        generated_at="2026-08-02T00:00:00+00:00",
        notes_output=tmp_path / "notes.md",
    )
    with pytest.raises(ValueError, match=message):
        release_proof.command_assemble(args)


def test_assemble_rejects_tag_version_mismatch(tmp_path: Path):
    release_dir, version_file, readme = _fixture_release(tmp_path)
    args = argparse.Namespace(
        directory=release_dir,
        version_file=version_file,
        readme=readme,
        repository="razzant/ouboros",
        tag="v6.87.6",
        commit="a" * 40,
        run_url="https://example.test/run",
        previous_tag=None,
        generated_at=None,
        notes_output=tmp_path / "notes.md",
    )
    with pytest.raises(ValueError, match="tag/version mismatch"):
        release_proof.command_assemble(args)


def test_verify_uploaded_requires_exact_names_sizes_and_digests(tmp_path: Path):
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    asset = release_dir / "Ouroboros-1.0.0.dmg"
    asset.write_bytes(b"artifact")
    metadata = tmp_path / "remote.json"
    metadata.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "name": asset.name,
                        "size": asset.stat().st_size,
                        "digest": f"sha256:{_digest(asset)}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    release_proof.command_verify_uploaded(
        argparse.Namespace(directory=release_dir, metadata=metadata)
    )
    asset.write_bytes(b"ARTIFACT")
    with pytest.raises(ValueError, match="digest mismatch"):
        release_proof.command_verify_uploaded(
            argparse.Namespace(directory=release_dir, metadata=metadata)
        )


def test_linux_package_smoke_pins_third_party_vendor_images_by_digest():
    script = (REPO / "scripts" / "smoke_linux_packages.sh").read_text(encoding="utf-8")
    for repository in (
        "registry.red-soft.ru/ubi8/ubi",
        "registry.astralinux.ru/library/astra/ubi18",
    ):
        assert f"{repository}@sha256:" in script
        assert f"{repository}:" not in script


def test_linux_packages_declare_and_resolve_the_git_runtime_dependency():
    builder = (REPO / "scripts" / "build_linux_packages.sh").read_text(encoding="utf-8")
    smoke = (REPO / "scripts" / "smoke_linux_packages.sh").read_text(encoding="utf-8")

    assert "Depends: git" in builder
    assert "Requires:       git" in builder
    assert "normalize_linux_package_version" in builder
    assert "apt-get install -y -qq" in smoke
    assert "dnf install -y -q" in smoke
    assert "command -v git" in smoke
    assert "dpkg --install" not in smoke
    assert "rpm --install" not in smoke


def test_every_future_release_receipt_requires_real_embedded_betterleaks():
    assert set(release_proof.REQUIRED_SMOKE_CHECKS) == set(release_proof.PROOF_IDS)
    for proof_id, checks in release_proof.REQUIRED_SMOKE_CHECKS.items():
        assert "embedded_betterleaks_runtime" in checks, proof_id


def test_future_final_artifact_lanes_smoke_betterleaks_from_the_artifact():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    build_job = workflow[
        workflow.index("  build:") : workflow.index("  vendor-package-smoke:")
    ]
    assert build_job.count("scripts/betterleaks_platform_smoke.py") == 4
    assert '--bundle-root "$MOUNT/Ouroboros.app/Contents/Resources"' in build_job
    assert '--bundle-root "$SMOKE_ROOT/Ouroboros/_internal"' in build_job
    assert '--bundle-root "$APPDIR/usr/lib/ouroboros/_internal"' in build_job
    assert '--bundle-root "$SmokeRoot\\Ouroboros\\_internal"' in build_job
    assert "betterleaks-standalone/bin/betterleaks" in build_job
    assert "codesign --verify --strict" in build_job
    assert "--check embedded_betterleaks_runtime" in build_job

    package_smoke = (REPO / "scripts" / "smoke_linux_packages.sh").read_text(
        encoding="utf-8"
    )
    assert "betterleaks_platform_smoke.py:/tmp/betterleaks_platform_smoke.py:ro" in package_smoke
    assert "PYTHONPATH=/opt/ouroboros/_internal" in package_smoke
    assert "--bundle-root /opt/ouroboros/_internal" in package_smoke


def test_linux_rpm_stage_recreates_the_absolute_cli_symlink():
    builder = (REPO / "scripts" / "build_linux_packages.sh").read_text(encoding="utf-8")

    assert 'cp -al "$ROOT"/. %{buildroot}/' not in builder
    assert 'cp -al "$ROOT/opt/ouroboros" "%{buildroot}/opt/ouroboros"' in builder
    assert (
        'ln -s /opt/ouroboros/bin/ouroboros '
        '"%{buildroot}/usr/bin/ouroboros"'
    ) in builder


def test_linux_packages_ship_the_systemd_user_unit():
    """Both packages must carry the inert launcher unit and prove it after install.

    Without the unit a packaged install has no stable name to stop: the desktop
    launcher lands in a transient scope whose name changes every start, and
    killing only the parent leaves workers holding port 8765.  Shipping it must
    stay inert, though — enabling or starting a desktop agent from a package
    postinst would be wrong.
    """
    builder = (REPO / "scripts" / "build_linux_packages.sh").read_text(encoding="utf-8")
    smoke = (REPO / "scripts" / "smoke_linux_packages.sh").read_text(encoding="utf-8")
    unit = (REPO / "packaging" / "systemd" / "ouroboros.service").read_text(encoding="utf-8")

    # deb stage
    assert (
        'install -m 644 packaging/systemd/ouroboros.service'
    ) in builder
    assert '"$ROOT/usr/lib/systemd/user"' in builder
    # rpm stage
    assert '"%{buildroot}/usr/lib/systemd/user"' in builder
    assert '/usr/lib/systemd/user/ouroboros.service' in builder

    # A user unit, not a system one: state lives in $HOME.
    assert "WantedBy=default.target" in unit
    # The native launcher remains the only bootstrap/restart/panic owner.
    assert "ExecStart=/opt/ouroboros/Ouroboros" in unit
    assert not any(line.startswith("Restart=") for line in unit.splitlines())
    # Stopping must reach the worker pool, not just the launcher.
    assert "KillMode=control-group" in unit

    # The digest-bound package smoke verifies the unit from the installed
    # .deb/.rpm, not only the source staging tree.
    assert "test -s /usr/lib/systemd/user/ouroboros.service" in smoke
    assert "grep -Fqx 'ExecStart=/opt/ouroboros/Ouroboros'" in smoke
    assert "grep -Fqx 'KillMode=control-group'" in smoke
    assert "! grep -q '^Restart='" in smoke

    for proof_id in (
        "linux-deb-amd64",
        "linux-rpm-x86_64",
        "linux-rpm-red80-x86_64",
    ):
        assert "systemd_user_unit" in release_proof.REQUIRED_SMOKE_CHECKS[proof_id]

    # Nothing may activate it on install.
    for forbidden in (
        "systemctl enable",
        "systemctl --user enable",
        "systemctl start",
        "systemctl --user start",
    ):
        assert forbidden not in builder, (
            f"packaging must not run {forbidden!r}: enabling a desktop agent "
            "from a package is the user's decision"
        )


def test_linux_package_smoke_starts_the_desktop_launcher_on_ubuntu_22_04():
    smoke = (REPO / "scripts" / "smoke_linux_packages.sh").read_text(encoding="utf-8")

    assert "ubuntu:22.04" in smoke
    assert "test -x /opt/ouroboros/Ouroboros" in smoke
    assert "timeout --signal=TERM --kill-after=5s 5s /opt/ouroboros/Ouroboros" in smoke
    assert "desktop launcher exited before the smoke deadline" in smoke
    assert "ouroboros-smoke-data/logs/launcher.log" in smoke


def test_vendor_distro_smoke_is_informational_and_never_gates_a_release():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    vendor_job = workflow[
        workflow.index("  vendor-package-smoke:") : workflow.index("  release:")
    ]
    assert "continue-on-error: true" in vendor_job
    assert "smoke_linux_packages.sh vendor" in vendor_job

    # The gating lane runs every package through Docker Hub images only, so a
    # vendor registry outage cannot stop a tagged release.
    build_job = workflow[
        workflow.index("  build:") : workflow.index("  vendor-package-smoke:")
    ]
    assert "smoke_linux_packages.sh official" in build_job
    assert "smoke_linux_packages.sh vendor" not in build_job

    release_needs = next(
        line
        for line in workflow[workflow.index("  release:") :].splitlines()
        if line.strip().startswith("needs:")
    )
    assert "vendor-package-smoke" not in release_needs


def test_release_workflow_orders_smoke_sbom_attestation_and_draft_verification():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    markers = [
        "- name: Locate final release archive",
        "- name: Smoke final macOS DMG",
        "- name: Record packaged artifact smoke",
        "- name: Install digest-pinned Syft",
        "- name: Generate CycloneDX SBOM from packaged payload",
        "- name: Attest build provenance",
        "- name: Attest SBOM",
        "- name: Build Linux .deb and .rpm packages",
        "- name: Smoke Linux packages in Ubuntu and Fedora containers",
        "- name: Record Linux package smoke and reuse payload SBOM",
        "- name: Attest Linux package provenance",
        "- name: Upload build artifact",
        "- name: Assemble release proof capsule and notes",
        "- name: Verify artifact attestations",
        "- name: Require an unpublished release slot",
        "- name: Verify remote release tag before draft",
        "- name: Create draft GitHub Release",
        "- name: Verify uploaded draft",
        "- name: Verify remote release tag before publish",
        "- name: Publish verified GitHub Release",
    ]
    positions = [workflow.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d" in workflow
    assert "anchore/sbom-action@" not in workflow
    assert "SYFT_VERSION: 1.50.0" in workflow
    assert "syft_1.50.0_darwin_arm64.tar.gz" in workflow
    assert "e32fdb9d47823fa633748a1efca2528fd77c37469ea93c9e40ab835da44e4cce" in workflow
    assert "if-no-files-found: error" in workflow
    assert "draft: true" in workflow
    assert "files: release-artifacts/*" not in workflow
    assert "matrix.sbom_path" not in workflow
    assert "steps.smoke_macos.outputs.sbom_path" in workflow
    assert "steps.smoke_appimage.outputs.sbom_path" in workflow
    assert "release-smoke-linux-appimage-x86_64.json" in workflow
    assert "sbom-linux-appimage-x86_64.cdx.json" in workflow
    assert "Ouroboros-*-linux-x86_64.AppImage" in workflow
    assert "--check appimage_extract_and_run" in workflow
    assert "--check appimage_metadata" in workflow
    assert "--check product_version" in workflow
    assert "--check browser_fallback_start" in workflow
    assert "--check gateway_readiness" in workflow
    assert "--check clean_shutdown" in workflow
    assert "--check shared_libraries" in workflow
    assert 'APP_ROOT="$HOME_DIR/Ouroboros"' in workflow
    assert 'APPIMAGE_CUSTODIAN_PID="$(ps -o ppid= -p "$LAUNCHER_PID"' in workflow
    assert 'APPIMAGE_RUNTIME_PID="$(ps -o ppid= -p "$APPIMAGE_CUSTODIAN_PID"' in workflow
    assert 'kill -0 "$APPIMAGE_RUNTIME_PID"' in workflow
    assert 'APPIMAGE_PRIVATE_BASE="${APPIMAGE_RUNTIME_ROOT%/*}"' in workflow
    assert 'if [ -e "$APPIMAGE_PRIVATE_BASE" ]; then' in workflow
    assert "runtime death orders the cleanup" in workflow
    assert 'OUROBOROS_APP_ROOT="$APP_ROOT"' not in workflow
    assert 'test -x "$MOUNT/Install CLI.command"' in workflow
    assert 'test -L "$MOUNT/Applications"' in workflow
    assert 'test "$(readlink "$MOUNT/Applications")" = "/Applications"' in workflow
    assert 'test -L "$SBOM_ROOT/Applications"' in workflow
    assert 'unlink "$SBOM_ROOT/Applications"' in workflow
    assert "--check applications_shortcut" in workflow
    # The native Linux packages are released alongside the tarball and go
    # through the same smoke → SBOM → attestation → upload chain.
    assert "bash scripts/build_linux_packages.sh" in workflow
    assert "bash scripts/smoke_linux_packages.sh" in workflow
    assert "--check package_install" in workflow
    assert "--check runtime_dependency" in workflow
    assert "--check systemd_user_unit" in workflow
    assert "--check desktop_launcher_start" in workflow
    assert "release-artifacts/ouroboros_*_amd64.deb" in workflow
    assert "release-artifacts/ouroboros-*-1.x86_64.rpm" in workflow
    assert "release-artifacts/ouroboros-*-1.red80.x86_64.rpm" in workflow
    assert "sbom-path: dist/sbom-linux-deb-amd64.cdx.json" in workflow
    assert "sbom-path: dist/sbom-linux-rpm-x86_64.cdx.json" in workflow
    assert "sbom-path: dist/sbom-linux-rpm-red80-x86_64.cdx.json" in workflow
    package_proof = workflow[
        workflow.index("- name: Record Linux package smoke and reuse payload SBOM") :
        workflow.index("- name: Attest Linux package provenance")
    ]
    assert 'PAYLOAD_SBOM="dist/sbom-linux-x86_64.cdx.json"' in package_proof
    assert 'cp "$PAYLOAD_SBOM" "dist/sbom-$1.cdx.json"' in package_proof
    assert "steps.syft.outputs.path" not in package_proof
    assert "lipo -archs" in workflow
    assert "Refusing to modify the published release" in workflow
    assert "group: release-${{ github.ref }}" in workflow
    assert workflow.count('git ls-remote --exit-code origin "$TAG_REF" "$PEELED_REF"') == 2
    assert workflow.count('test "$(git cat-file -t "$TAG_REF")" = "tag"') == 2
    assert workflow.count('[ "$PEELED_SHA" != "$GITHUB_SHA" ]') == 2
    assert 'target_commitish: ${{ github.sha }}' in workflow
    assert '--source-digest "$GITHUB_SHA"' in workflow
    assert '--source-ref "$GITHUB_REF"' in workflow
    assert '--signer-workflow "$GITHUB_REPOSITORY/.github/workflows/ci.yml"' in workflow
    assert "--predicate-type https://cyclonedx.org/bom" in workflow
    assert "$env:USERPROFILE = $HomeDir" in workflow
    assert "$env:LOCALAPPDATA = Join-Path $HomeDir" in workflow
    assert "$env:APPDATA = Join-Path $HomeDir" in workflow
    assert "$env:HOMEDRIVE = Split-Path -Qualifier $HomeDir" in workflow
    assert "$env:HOMEPATH = $HomeDir.Substring" in workflow
    build_job = workflow[
        workflow.index("  build:") : workflow.index("  vendor-package-smoke:")
    ]
    job_env = build_job[build_job.index("    env:") : build_job.index("    steps:")]
    assert "BUILD_CERTIFICATE_BASE64:" not in job_env
    assert "P12_PASSWORD:" not in job_env
    assert "KEYCHAIN_PASSWORD:" not in job_env
