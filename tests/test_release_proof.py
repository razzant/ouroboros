from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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


SIGNER = "0123456789ABCDEF0123456789ABCDEF01234567"  # synthetic SHA-1 thumbprint


def _fixture_release(
    tmp_path: Path, version: str = "6.87.5", signer: str = ""
) -> tuple[Path, Path, Path]:
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
        if proof_id == release_proof.AUTHENTICODE_PROOF_ID:
            receipt["authenticode"] = {"status": "unsigned"}
            if signer:
                receipt["checks"] = sorted(
                    {*receipt["checks"], *release_proof.AUTHENTICODE_SMOKE_CHECKS}
                )
                receipt["authenticode"] = {
                    "status": "signed", "signerThumbprint": signer, "publisher": "Example Publisher",
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
        android_build_result="success",
        android_attestation_result="success",
        windows_signer_thumbprint="",
        github_output=tmp_path / "github-output",
        notes_output=notes,
    )
    release_proof.command_assemble(args)

    evidence = json.loads((release_dir / "release-evidence.json").read_text())
    assert evidence["source"]["commit"] == "a" * 40
    assert len(evidence["artifacts"]) == 9
    assert evidence["experimentalAndroid"]["status"] == "verified"
    upload = json.loads(args.github_output.read_text(encoding="utf-8").split("=", 1)[1]).splitlines()
    assert len(upload) == 29
    assert {Path(path).name for path in upload} >= {row["name"] for row in evidence["artifacts"]}
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
        android_build_result="success",
        android_attestation_result="success",
        windows_signer_thumbprint="",
        github_output=None,
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
        android_build_result="success",
        android_attestation_result="success",
        windows_signer_thumbprint="",
        github_output=None,
        notes_output=tmp_path / "notes.md",
    )
    with pytest.raises(ValueError, match="not bound"):
        release_proof.command_assemble(args)


@pytest.mark.parametrize("failure", ["failure", "cancelled", "skipped", "missing", "partial", "digest", "smoke", "sbom", "attestation"])
def test_unavailable_android_pair_keeps_verified_desktop_release(tmp_path, failure):
    release_dir, version_file, readme = _fixture_release(tmp_path)
    build_result = failure if failure in {"failure", "cancelled", "skipped"} else "success"
    attestation_result = "failure" if failure == "attestation" else "success"
    apk = release_dir / release_proof.release_asset_name("android-apk", "6.87.5")
    if failure == "missing":
        for proof_id in release_proof.ANDROID_DOWNLOAD_IDS:
            for path in (release_dir / release_proof.release_asset_name(proof_id, "6.87.5"),
                         release_dir / f"release-smoke-{proof_id}.json", release_dir / f"sbom-{proof_id}.cdx.json"):
                path.unlink()
    elif failure == "partial":
        apk.unlink()
    elif failure == "digest":
        apk.write_bytes(b"different APK bytes")
    elif failure == "smoke":
        path = release_dir / "release-smoke-android-apk.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["checks"] = []
        path.write_text(json.dumps(receipt), encoding="utf-8")
    elif failure == "sbom":
        (release_dir / "sbom-android-apk.cdx.json").write_text("{}", encoding="utf-8")
    output, notes = tmp_path / "github-output", tmp_path / "notes.md"
    args = argparse.Namespace(directory=release_dir, version_file=version_file, readme=readme,
        repository="razzant/ouroboros", tag="v6.87.5", commit="a" * 40,
        run_url="https://example.test/failed-run", previous_tag=None, generated_at=None,
        notes_output=notes, android_build_result=build_result,
        android_attestation_result=attestation_result,
        windows_signer_thumbprint="", github_output=output)
    release_proof.command_assemble(args)
    evidence = json.loads((release_dir / "release-evidence.json").read_text(encoding="utf-8"))
    assert {row["proofId"] for row in evidence["artifacts"]} == set(release_proof.DESKTOP_DOWNLOAD_IDS)
    assert evidence["experimentalAndroid"]["status"] == "unavailable"
    assert evidence["experimentalAndroid"]["buildResult"] == build_result
    assert evidence["experimentalAndroid"]["reason"]
    assert next(row for row in evidence["workflow"]["gates"] if row["name"] == "android-build")["status"] == build_result
    assert len((release_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()) == 21
    upload = json.loads(output.read_text(encoding="utf-8").split("=", 1)[1]).splitlines()
    assert len(upload) == 23
    assert not any("android" in Path(path).name for path in upload)
    text = notes.read_text(encoding="utf-8")
    assert "Android artifacts are unavailable" in text and args.run_url in text
    assert "-android.apk]" not in text and "-android-arm64.tar.gz]" not in text
    assert "Ouroboros-6.87.5.dmg" in text
    remote = tmp_path / "remote.json"
    rows = [{"name": Path(path).name, "size": Path(path).stat().st_size,
             "digest": "sha256:" + _digest(Path(path))} for path in upload]
    remote.write_text(json.dumps({"assets": rows}), encoding="utf-8")
    release_proof.command_verify_uploaded(argparse.Namespace(directory=release_dir, metadata=remote))
    rows.append({"name": apk.name, "size": 1, "digest": "sha256:unverified"})
    remote.write_text(json.dumps({"assets": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="uploaded asset set"):
        release_proof.command_verify_uploaded(argparse.Namespace(directory=release_dir, metadata=remote))


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
        android_build_result="success",
        android_attestation_result="success",
        windows_signer_thumbprint="",
        github_output=None,
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
        android_build_result="success",
        android_attestation_result="success",
        windows_signer_thumbprint="",
        github_output=None,
        notes_output=tmp_path / "notes.md",
    )
    with pytest.raises(ValueError, match="tag/version mismatch"):
        release_proof.command_assemble(args)


def test_verify_uploaded_requires_exact_names_sizes_and_digests(tmp_path: Path):
    release_dir, version_file, readme = _fixture_release(tmp_path)
    release_proof.command_assemble(argparse.Namespace(
        directory=release_dir, version_file=version_file, readme=readme,
        repository="razzant/ouroboros", tag="v6.87.5", commit="a" * 40,
        run_url="https://example.test/run", previous_tag=None, generated_at=None,
        notes_output=tmp_path / "notes.md", android_build_result="success",
        android_attestation_result="success",
        windows_signer_thumbprint="", github_output=None))
    metadata = tmp_path / "remote.json"
    metadata.write_text(json.dumps({"assets": [
        {"name": path.name, "size": path.stat().st_size, "digest": "sha256:" + _digest(path)}
        for path in release_dir.iterdir()
    ]}), encoding="utf-8")
    release_proof.command_verify_uploaded(argparse.Namespace(directory=release_dir, metadata=metadata))
    asset = release_dir / release_proof.release_asset_name("macos-arm64", "6.87.5")
    asset.write_bytes(asset.read_bytes().upper())
    with pytest.raises(ValueError, match="digest mismatch"):
        release_proof.command_verify_uploaded(argparse.Namespace(directory=release_dir, metadata=metadata))


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


def test_release_receipts_match_the_runtime_the_artifact_distributes():
    assert set(release_proof.REQUIRED_SMOKE_CHECKS) == set(release_proof.PROOF_IDS)
    for proof_id, checks in release_proof.REQUIRED_SMOKE_CHECKS.items():
        if proof_id in {"android-arm64", "android-apk"}:
            # Android installs dependencies from upstream; no embedded scanner
            # or physical-device smoke is claimed for the source/setup archive.
            assert "embedded_betterleaks_runtime" not in checks, proof_id
        else:
            assert "embedded_betterleaks_runtime" in checks, proof_id


def test_future_final_artifact_lanes_smoke_betterleaks_from_the_artifact():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    build_job = workflow[
        workflow.index("\n  build:\n") : workflow.index("\n  vendor-package-smoke:\n")
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
    assert "ExecStart=/opt/ouroboros/Ouroboros --launch-intent automatic" in unit.splitlines()
    assert not any(line.startswith("Restart=") for line in unit.splitlines())
    # Stopping must reach the worker pool, not just the launcher.
    assert "KillMode=control-group" in unit

    # The digest-bound package smoke verifies the unit from the installed
    # .deb/.rpm, not only the source staging tree.
    assert "test -s /usr/lib/systemd/user/ouroboros.service" in smoke
    assert "grep -Fqx 'ExecStart=/opt/ouroboros/Ouroboros --launch-intent automatic'" in smoke
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
        workflow.index("\n  vendor-package-smoke:\n") : workflow.index("\n  release:\n")
    ]
    assert "continue-on-error: true" in vendor_job
    assert "smoke_linux_packages.sh vendor" in vendor_job

    # The gating lane runs every package through Docker Hub images only, so a
    # vendor registry outage cannot stop a tagged release.
    build_job = workflow[
        workflow.index("\n  build:\n") : workflow.index("\n  vendor-package-smoke:\n")
    ]
    assert "smoke_linux_packages.sh official" in build_job
    assert "smoke_linux_packages.sh vendor" not in build_job

    release_needs = next(
        line
        for line in workflow[workflow.index("\n  release:\n") :].splitlines()
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
        "- name: Verify artifact attestations",
        # Proof acceptance consumes the actual optional-Android verification outcome.
        "- name: Assemble release proof capsule and notes",
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
    assert 'ids = list(registry["DESKTOP_DOWNLOAD_IDS"])' in workflow
    assert 'registry["release_asset_name"](proof_id, version)' in workflow
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
    assert "files: ${{ fromJSON(steps.release_proof.outputs.files_json) }}" in workflow
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
    publication = workflow.split("\n  release:\n", 1)[1]
    assert publication.count('git ls-remote --exit-code origin "$TAG_REF" "$PEELED_REF"') == 2
    assert publication.count('test "$(git cat-file -t "$TAG_REF")" = "tag"') == 2
    assert publication.count('[ "$PEELED_SHA" != "$GITHUB_SHA" ]') == 2
    create_release_step = workflow[
        workflow.index("- name: Create draft GitHub Release") :
        workflow.index("- name: Verify uploaded draft")
    ]
    assert "target_commitish:" not in create_release_step
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
        workflow.index("\n  build:\n") : workflow.index("\n  vendor-package-smoke:\n")
    ]
    job_env = build_job[build_job.index("    env:") : build_job.index("    steps:")]
    assert "BUILD_CERTIFICATE_BASE64:" not in job_env
    assert "P12_PASSWORD:" not in job_env
    assert "KEYCHAIN_PASSWORD:" not in job_env


def test_optional_android_does_not_waive_missing_desktop_asset(tmp_path):
    release_dir, _version, _readme = _fixture_release(tmp_path)
    (release_dir / release_proof.release_asset_name("macos-arm64", "6.87.5")).unlink()
    with pytest.raises(ValueError, match="required"):
        release_proof._proof_files(release_dir, "6.87.5", commit="a" * 40, tag="v6.87.5",
                                  android_build_result="failure", android_attestation_result="not_run",
                                  windows_signer_thumbprint="")


def _assemble_windows(tmp_path: Path, release_dir: Path, version_file: Path, readme: Path,
                      signer: str) -> tuple[dict, str]:
    """Run the release consumer with one Windows signing configuration."""
    notes = tmp_path / "notes.md"
    release_proof.command_assemble(argparse.Namespace(
        directory=release_dir, version_file=version_file, readme=readme,
        repository="razzant/ouroboros", tag="v6.87.5", commit="a" * 40,
        run_url="https://example.test/run", previous_tag=None, generated_at=None,
        notes_output=notes, android_build_result="success",
        android_attestation_result="success", windows_signer_thumbprint=signer,
        github_output=None))
    evidence = json.loads((release_dir / "release-evidence.json").read_text(encoding="utf-8"))
    windows = next(row for row in evidence["artifacts"] if row["proofId"] == "windows-x64")
    return windows, notes.read_text(encoding="utf-8")


def test_unconfigured_signing_publishes_an_explicitly_unsigned_windows_zip(tmp_path: Path):
    windows, notes = _assemble_windows(tmp_path, *_fixture_release(tmp_path), signer="")

    assert windows["authenticode"] == {"status": "unsigned"}
    assert "Windows: this ZIP is unsigned." in notes
    assert "unknown publisher" in notes
    assert "Authenticode signature" not in notes


def test_configured_signing_publishes_the_verified_signer(tmp_path: Path):
    # Configuration case and receipt case may differ; the published value is canonical.
    windows, notes = _assemble_windows(
        tmp_path, *_fixture_release(tmp_path, signer=SIGNER), signer=SIGNER.lower()
    )

    assert windows["authenticode"] == {
        "status": "signed", "signerThumbprint": SIGNER, "publisher": "Example Publisher",
    }
    assert f"from **Example Publisher** (certificate SHA-1 `{SIGNER}`)" in notes
    assert "timestamped Authenticode signature" in notes
    assert "not signed individually" in notes and "SmartScreen may still warn" in notes
    assert "this ZIP is unsigned" not in notes


@pytest.mark.parametrize(
    ("receipt_signer", "configured", "mutate", "message"),
    [
        # Configured signing never degrades to an unsigned release.
        ("", SIGNER, None, "configured certificate"),
        ("F" * 40, SIGNER, None, "configured certificate"),
        (SIGNER, SIGNER, "drop-timestamp", "missing signing checks"),
        (SIGNER, SIGNER, "drop-publisher", "configured certificate"),
        (SIGNER, SIGNER, "drop-state", "configured certificate"),
        # Without configuration, a receipt may not claim a signature.
        (SIGNER, "", None, "does not record an unsigned archive"),
        ("", "", "claim-timestamp", "does not record an unsigned archive"),
        ("", "", "drop-state", "does not record an unsigned archive"),
        # A set but malformed thumbprint is a broken configuration.
        ("", " ", None, "not a 40-hex"),
        ("", SIGNER[:-1], None, "not a 40-hex"),
        (SIGNER, SIGNER + "0", None, "not a 40-hex"),
    ],
)
def test_windows_signing_state_must_match_the_configuration(
    tmp_path: Path, receipt_signer: str, configured: str, mutate: str | None, message: str
):
    release_dir, version_file, readme = _fixture_release(tmp_path, signer=receipt_signer)
    path = release_dir / "release-smoke-windows-x64.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if mutate == "drop-timestamp":
        receipt["checks"].remove("timestamp")
    elif mutate == "claim-timestamp":
        receipt["checks"].append("timestamp")
    elif mutate == "drop-publisher":
        receipt["authenticode"]["publisher"] = " "
    elif mutate == "drop-state":
        del receipt["authenticode"]
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _assemble_windows(tmp_path, release_dir, version_file, readme, signer=configured)
    assert not (release_dir / "release-evidence.json").exists()


def test_release_consumer_must_state_the_windows_signing_mode():
    parser = release_proof.build_parser()
    base = ["assemble", "--directory", "r", "--repository", "o/r", "--tag", "v1",
            "--commit", "a" * 40, "--run-url", "u", "--notes-output", "n"]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    assert parser.parse_args([*base, "--windows-signer-thumbprint", ""]).windows_signer_thumbprint == ""


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-S", str(REPO / "scripts" / "release_proof.py"), *args],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


def test_handoff_archive_digest_is_checked_before_extraction(tmp_path: Path):
    archive = tmp_path / "Ouroboros-6.87.5-windows-x64.zip"
    archive.write_bytes(b"final archive bytes")
    output = tmp_path / "github-output"

    ok = _run_cli("locate", "--directory", str(tmp_path), "--expect-sha256", _digest(archive),
                  "--github-output", str(output))
    assert ok.returncode == 0, ok.stderr
    assert f"sha256={_digest(archive)}" in output.read_text(encoding="utf-8")

    # A replaced archive, or a producer job whose digest output never arrived.
    for expected in ("0" * 64, ""):
        refused = _run_cli("locate", "--directory", str(tmp_path), "--expect-sha256", expected,
                           "--github-output", str(tmp_path / "refused-output"))
        assert refused.returncode != 0
        assert "digest its producer job recorded" in refused.stderr
        assert not (tmp_path / "refused-output").exists()


def test_windows_receipt_records_signing_state_and_android_callers_stay_compatible(tmp_path: Path):
    archive = tmp_path / "Ouroboros-6.87.5-windows-x64.zip"
    archive.write_bytes(b"zip")
    common = ["record-smoke", "--proof-id", "windows-x64", "--artifact", str(archive),
              "--commit", "a" * 40, "--tag", "v6.87.5", "--check", "packaged_cli_help"]
    unsigned, signed = tmp_path / "unsigned.json", tmp_path / "signed.json"
    assert _run_cli(*common, "--output", str(unsigned)).returncode == 0
    assert json.loads(unsigned.read_text())["authenticode"] == {"status": "unsigned"}
    assert _run_cli(*common, "--output", str(signed), "--authenticode-thumbprint", SIGNER.lower(),
                    "--authenticode-publisher", "Example Publisher").returncode == 0
    assert json.loads(signed.read_text())["authenticode"] == {
        "status": "signed", "signerThumbprint": SIGNER, "publisher": "Example Publisher",
    }

    # scripts/build_android_release.py builds its own Namespace without these fields.
    apk = tmp_path / "Ouroboros-6.87.5-android.apk"
    apk.write_bytes(b"apk")
    release_proof.command_record_smoke(argparse.Namespace(
        proof_id="android-apk", artifact=apk, output=tmp_path / "android.json",
        commit="a" * 40, tag="v6.87.5", check=["apk_signature"]))
    assert "authenticode" not in json.loads((tmp_path / "android.json").read_text())


@pytest.mark.parametrize("version", ["7.6.0", "7.6.1-rc.1"])
@pytest.mark.parametrize("signer", ["", SIGNER])
def test_workflow_shell_handoff_receipt_and_release_consumer(tmp_path: Path, version: str, signer: str):
    """Execute the YAML's digest selector, receipt writer and release assembler.

    Only the Windows signature observation and the other platforms' receipts
    are fixtures; no native app, attestation service or signing service runs.
    """
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("workflow shell execution requires bash")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    release_dir, _, _ = _fixture_release(tmp_path, version=version)
    release_dir = release_dir.rename(tmp_path / "release-artifacts")
    dist = tmp_path / "dist"
    dist.mkdir()
    name = release_proof.release_asset_name("windows-x64", version)
    archive = dist / name
    archive.write_bytes(b"final mocked signed ZIP" if signer else b"final unsigned ZIP")
    output = tmp_path / "output"
    env = dict(os.environ, GITHUB_OUTPUT=str(output), WINDOWS_SIGNER_THUMBPRINT=signer,
               SIGNED_SHA256=_digest(archive) if signer else "wrong-unused-signed-digest",
               UNSIGNED_SHA256="wrong-unused-unsigned-digest" if signer else _digest(archive),
               SIGNER_THUMBPRINT=signer, SIGNER_PUBLISHER="Example Publisher" if signer else "",
               GITHUB_REF_NAME=f"v{version}", GITHUB_SHA="a" * 40, GITHUB_REPOSITORY="example/project",
               GITHUB_SERVER_URL="https://example.test", GITHUB_RUN_ID="123",
               ANDROID_BUILD_RESULT="skipped", ANDROID_ATTESTATION_RESULT="not_run")

    def run_step(job_id, name):
        step = next(step for step in jobs[job_id]["steps"] if step.get("name") == name)
        script = step["run"].replace("python scripts/release_proof.py", " ".join(map(shlex.quote, [
            Path(sys.executable).as_posix(), (REPO / "scripts/release_proof.py").as_posix()])))
        script = script.replace("${{ steps.release_asset.outputs.path }}", archive.as_posix())
        # Supply a previous-tag fixture without discovering the enclosing checkout.
        return subprocess.run([bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
                               "git() { echo v0.0.0; }\n" + script], cwd=tmp_path, env=env,
                              capture_output=True, text=True)

    check_name = "Check final archive digest before extraction"
    checked = run_step("windows-proof", check_name)
    assert checked.returncode == 0, checked.stderr
    assert f"sha256={_digest(archive)}" in output.read_text()
    output.unlink()
    selected = "SIGNED_SHA256" if signer else "UNSIGNED_SHA256"
    env[selected] = ""
    refused = run_step("windows-proof", check_name)
    assert refused.returncode != 0 and "digest its producer job recorded" in refused.stderr
    assert not output.exists()
    env[selected] = _digest(archive)

    recorded = run_step("windows-proof", "Record packaged artifact smoke")
    assert recorded.returncode == 0, recorded.stderr
    for path in (archive, dist / "release-smoke-windows-x64.json"):
        shutil.copyfile(path, release_dir / path.name)
    assembled = run_step("release", "Assemble release proof capsule and notes")
    assert assembled.returncode == 0, assembled.stderr
    evidence = json.loads((release_dir / "release-evidence.json").read_text())
    assert {row["proofId"] for row in evidence["artifacts"]} == set(release_proof.DESKTOP_DOWNLOAD_IDS)
    windows = next(row for row in evidence["artifacts"] if row["proofId"] == "windows-x64")
    assert windows["sha256"] == _digest(archive)
    assert windows["authenticode"] == ({"status": "signed", "signerThumbprint": signer,
                                        "publisher": "Example Publisher"} if signer else {"status": "unsigned"})
    notes = (tmp_path / "release-notes.md").read_text()
    assert ("timestamped Authenticode signature" if signer else "this ZIP is unsigned") in notes
    # The assembly consumer also rejects bytes changed after proof generation.
    (release_dir / archive.name).write_bytes(b"replaced ZIP")
    refused = run_step("release", "Assemble release proof capsule and notes")
    assert refused.returncode != 0 and "receipt is not bound" in refused.stderr
