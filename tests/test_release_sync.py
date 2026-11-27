"""Tests for ouroboros/tools/release_sync.py (standalone library, no wire-up)."""

from pathlib import Path
import subprocess

import pytest

from ouroboros.tools.release_sync import (
    ANDROID_DOWNLOAD_IDS,
    DESKTOP_DOWNLOAD_IDS,
    RELEASE_ASSET_TEMPLATES,
    WINDOWS_SETUP_PROOF_ID,
    WINDOWS_ZIP_PROOF_ID,
    carrier_only_change,
    carrier_spans_for,
    check_history_limit,
    check_worktree_version_sync,
    desktop_download_ids,
    detect_numeric_claims,
    download_links_include_windows_installer,
    is_development_marker,
    is_release_version,
    locate_carrier_span,
    next_cycle_problem,
    normalize_linux_package_version,
    normalize_release_tag,
    published_download_version,
    release_asset_download_url,
    release_asset_name,
    resolve_release_windows_installer,
    run_release_preflight,
    substitute_carrier_spans,
    sync_release_metadata,
    version_carrier_desyncs,
)
from ouroboros.tools.release_sync import (  # noqa: E402 — private helpers under test
    _normalize_pep440,
    _render_html_windows_actions,
    _shields_escape,
    _sync_html_download_urls,
    _VERSION_RE,
    _VERSION_ROW_RE,
    _README_BADGE_RE,
    _ARCH_HEADER_RE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_references(version: str, *, windows_installer: bool = True) -> str:
    return "".join(
        f"[download-{proof_id}]: {release_asset_download_url(proof_id, version)}\n"
        for proof_id in (*desktop_download_ids(windows_installer=windows_installer), *ANDROID_DOWNLOAD_IDS)
    )


def _download_page(version: str, *, windows_installer: bool = True) -> str:
    parts = []
    for proof_id in (*desktop_download_ids(windows_installer=windows_installer), *ANDROID_DOWNLOAD_IDS):
        if proof_id == WINDOWS_SETUP_PROOF_ID:
            continue
        if proof_id == WINDOWS_ZIP_PROOF_ID:
            parts.append(
                _render_html_windows_actions(version, "", "\n", windows_installer=windows_installer) + "\n"
            )
        else:
            parts.append(
                f'<a data-release-download="{proof_id}" '
                f'href="{release_asset_download_url(proof_id, version)}">download</a>\n'
            )
    return "".join(parts)


def _make_repo(tmp_path: Path, version: str = "4.99.1") -> Path:
    """Create a minimal fake repo with all version-carrier files."""
    (tmp_path / "VERSION").write_text(version + "\n", encoding="utf-8")

    (tmp_path / "pyproject.toml").write_text(
        '[tool.poetry]\nname = "ouroboros"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        'version = 1\nrevision = 3\nrequires-python = ">=3.10"\n\n'
        '[[package]]\nname = "ouroboros"\nversion = "0.0.0"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    web = tmp_path / "web"
    web.mkdir()
    (web / "package.json").write_text(
        '{\n  "name": "ouroboros-web",\n  "version": "0.0.0"\n}\n',
        encoding="utf-8",
    )
    (web / "package-lock.json").write_text(
        '{\n  "name": "ouroboros-web",\n  "version": "0.0.0",\n  "lockfileVersion": 3,\n'
        '  "packages": {\n    "": {\n      "name": "ouroboros-web",\n      "version": "0.0.0",\n'
        '      "devDependencies": {"eslint": "10.10.0"}\n    },\n    "node_modules/eslint": {\n'
        '      "version": "10.10.0"\n    }\n  }\n}\n',
        encoding="utf-8",
    )
    modules = web / "modules"
    modules.mkdir()
    (modules / "api_types.js").write_text(
        "export const GATEWAY_CONTRACT_VERSION = '0.0.0';\n", encoding="utf-8"
    )

    badge_line = (
        '[![Version 0.0.0]'
        '(https://img.shields.io/badge/version-0.0.0-green.svg)]'
        '(VERSION)\n'
    )
    readme_content = (
        "# Ouroboros\n\n"
        + badge_line
        + _download_references("0.0.0")
        + "\n## Version History\n\n"
        "| Version | Date | Description |\n"
        "|---------|------|-------------|\n"
        f"| {version} | 2026-01-01 | New release |\n"
    )
    (tmp_path / "README.md").write_text(readme_content, encoding="utf-8")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ARCHITECTURE.md").write_text(
        "# Ouroboros v0.0.0 — Architecture\n\nContent here.\n",
        encoding="utf-8",
    )
    (docs / "install").mkdir()
    (docs / "install" / "index.html").write_text(
        _download_page("0.0.0"), encoding="utf-8"
    )

    site_install = tmp_path / "site" / "install"
    site_install.mkdir(parents=True)
    (site_install / "index.html").write_text(
        _download_page("0.0.0"), encoding="utf-8"
    )

    return tmp_path


# ---------------------------------------------------------------------------
# sync_release_metadata
# ---------------------------------------------------------------------------

class TestSyncReleaseMetadata:
    def test_syncs_pyproject_toml(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3")
        changed = sync_release_metadata(str(repo))
        assert "pyproject.toml" in changed
        text = (repo / "pyproject.toml").read_text()
        assert 'version = "1.2.3"' in text

    def test_syncs_editable_root_version_in_uv_lock(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3-rc.4")
        changed = sync_release_metadata(str(repo))

        assert "uv.lock" in changed
        lock_text = (repo / "uv.lock").read_text(encoding="utf-8")
        assert 'name = "ouroboros"\nversion = "1.2.3rc4"' in lock_text

    def test_stale_uv_lock_root_version_is_reported(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3-rc.4")
        sync_release_metadata(str(repo))
        lock_path = repo / "uv.lock"
        lock_path.write_text(
            lock_path.read_text(encoding="utf-8").replace(
                'version = "1.2.3rc4"',
                'version = "1.2.3rc3"',
            ),
            encoding="utf-8",
        )

        desync = version_carrier_desyncs(
            "1.2.3-rc.4",
            uv_lock_text=lock_path.read_text(encoding="utf-8"),
            detailed=True,
        )

        assert desync == ['uv.lock (expected editable root version = "1.2.3rc4")']

    def test_syncs_readme_badge(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3")
        changed = sync_release_metadata(str(repo))
        assert "README.md" in changed
        text = (repo / "README.md").read_text()
        assert "Version 1.2.3" in text
        assert "version-1.2.3-green" in text

    def test_syncs_web_package_json(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3-rc.4")
        changed = sync_release_metadata(str(repo))
        assert "web/package.json" in changed
        text = (repo / "web" / "package.json").read_text()
        assert '"version": "1.2.3-rc.4"' in text

    def test_syncs_web_package_lock_root_versions_only(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3-rc.4")
        changed = sync_release_metadata(str(repo))
        assert "web/package-lock.json" in changed
        text = (repo / "web" / "package-lock.json").read_text(encoding="utf-8")
        assert text.count('"version": "1.2.3-rc.4"') == 2 and '"version": "10.10.0"' in text

    def test_web_package_lock_participates_in_desync_checks(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3-rc.4")
        sync_release_metadata(str(repo))
        lock = repo / "web" / "package-lock.json"
        lock.write_text(lock.read_text(encoding="utf-8").replace('"version": "1.2.3-rc.4"', '"version": "1.2.3-rc.3"', 1),
                        encoding="utf-8")
        desync = version_carrier_desyncs(
            "1.2.3-rc.4", detailed=True,
            web_package_lock_text=lock.read_text(encoding="utf-8"),
        )
        assert desync == ['web/package-lock.json (expected both root "version" entries = "1.2.3-rc.4")']

    def test_web_package_json_participates_in_desync_checks(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3-rc.4")
        sync_release_metadata(str(repo))
        (repo / "web" / "package.json").write_text('{"version": "1.2.3-rc.3"}', encoding="utf-8")

        desync = version_carrier_desyncs(
            "1.2.3-rc.4",
            web_package_text=(repo / "web" / "package.json").read_text(encoding="utf-8"),
            detailed=True,
        )

        assert desync == ['web/package.json (expected "version": "1.2.3-rc.4")']

    def test_gateway_contract_version_is_a_release_carrier(self, tmp_path):
        """tests/test_gateway_parity.py pins GATEWAY_CONTRACT_VERSION to VERSION,
        so the release tooling must sync it. It did not, and every release had to
        rediscover the carrier through a red CI run."""
        repo = _make_repo(tmp_path, "1.2.3")

        changed = sync_release_metadata(str(repo))
        api_types = repo / "web" / "modules" / "api_types.js"

        assert "web/modules/api_types.js" in changed
        assert "GATEWAY_CONTRACT_VERSION = '1.2.3'" in api_types.read_text(encoding="utf-8")

    def test_stale_gateway_contract_version_is_reported(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3")
        sync_release_metadata(str(repo))
        api_types = repo / "web" / "modules" / "api_types.js"
        api_types.write_text("export const GATEWAY_CONTRACT_VERSION = '1.2.2';\n", encoding="utf-8")

        desync = version_carrier_desyncs(
            "1.2.3",
            api_types_text=api_types.read_text(encoding="utf-8"),
            detailed=True,
        )

        assert desync == ["web/modules/api_types.js (expected GATEWAY_CONTRACT_VERSION = '1.2.3')"]

    def test_syncs_architecture_header(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3")
        changed = sync_release_metadata(str(repo))
        assert "docs/ARCHITECTURE.md" in changed
        text = (repo / "docs" / "ARCHITECTURE.md").read_text()
        assert "v1.2.3" in text

    def test_syncs_direct_download_urls_from_version(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3")

        changed = sync_release_metadata(str(repo))

        assert "README.md" in changed
        assert "site/install/index.html" in changed
        assert "docs/install/index.html" in changed
        readme = (repo / "README.md").read_text(encoding="utf-8")
        site_install = (repo / "site" / "install" / "index.html").read_text(
            encoding="utf-8"
        )
        docs_install = (repo / "docs" / "install" / "index.html").read_text(
            encoding="utf-8"
        )
        for proof_id in RELEASE_ASSET_TEMPLATES:
            expected = release_asset_download_url(proof_id, "1.2.3")
            assert f"[download-{proof_id}]: {expected}" in readme
            assert expected in site_install
            assert expected in docs_install

    def test_prerelease_download_urls_use_exact_tag_not_latest(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3-rc.4")

        sync_release_metadata(str(repo))

        projected = "\n".join([
            (repo / "README.md").read_text(encoding="utf-8"),
            (repo / "site" / "install" / "index.html").read_text(encoding="utf-8"),
            (repo / "docs" / "install" / "index.html").read_text(encoding="utf-8"),
        ])
        assert "/releases/download/v1.2.3-rc.4/" in projected
        assert "/releases/latest/download/" not in projected

    def test_stale_direct_download_url_is_reported(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3")
        sync_release_metadata(str(repo))
        readme = (repo / "README.md").read_text(encoding="utf-8")
        stale_readme = readme.replace(
            release_asset_download_url("macos-arm64", "1.2.3"),
            release_asset_download_url("macos-arm64", "1.2.2"),
        )

        desync = version_carrier_desyncs(
            "1.2.3",
            download_readme_text=stale_readme,
            detailed=True,
        )

        assert desync == [
            "README.md download macos-arm64 "
            f"(expected {release_asset_download_url('macos-arm64', '1.2.3')})"
        ]

    def test_every_duplicate_readme_reference_must_match_version(self):
        expected = release_asset_download_url("macos-arm64", "1.2.3")
        stale = release_asset_download_url("macos-arm64", "1.2.2")
        readme = _download_references("1.2.3") + (
            f"[download-macos-arm64]: {stale}\n"
        )

        desync = version_carrier_desyncs(
            "1.2.3",
            download_readme_text=readme,
            detailed=True,
        )

        assert desync == [
            "README.md download macos-arm64 "
            f"(expected {expected})"
        ]

    def test_every_marked_html_download_must_match_version(self):
        expected = release_asset_download_url("macos-arm64", "1.2.3")
        stale = release_asset_download_url("macos-arm64", "1.2.2")
        html = _download_page("1.2.3") + (
            f'<a data-release-download="macos-arm64" href="{stale}">quick start</a>\n'
        )

        desync = version_carrier_desyncs(
            "1.2.3",
            site_install_text=html,
            detailed=True,
        )

        assert desync == [
            "site/install/index.html download macos-arm64 "
            f"(expected {expected})"
        ]

    def test_unmarked_legacy_documents_do_not_join_the_projection(self):
        desync = version_carrier_desyncs(
            "1.2.3",
            download_readme_text="# Minimal fixture\n",
            site_install_text="<h1>Install</h1>\n",
            docs_install_text="<h1>Install</h1>\n",
            detailed=True,
        )

        assert desync == []

    def test_partial_download_projection_reports_missing_members(self):
        expected = release_asset_download_url("macos-arm64", "1.2.3")
        readme = f"[download-macos-arm64]: {expected}\n"

        desync = version_carrier_desyncs(
            "1.2.3",
            download_readme_text=readme,
        )

        assert "README.md download macos-arm64" not in desync
        assert f"README.md download {WINDOWS_SETUP_PROOF_ID}" in desync
        assert len(desync) == len(DESKTOP_DOWNLOAD_IDS) - 1

        zip_only_desync = version_carrier_desyncs(
            "7.12.2",
            download_readme_text=f"[download-macos-arm64]: {release_asset_download_url('macos-arm64', '7.12.2')}\n",
            windows_installer=False,
        )
        assert f"README.md download {WINDOWS_SETUP_PROOF_ID}" not in zip_only_desync
        assert len(zip_only_desync) == len(desktop_download_ids(windows_installer=False)) - 1

    def test_idempotent_second_call_returns_no_changes(self, tmp_path):
        repo = _make_repo(tmp_path, "1.2.3")
        sync_release_metadata(str(repo))  # first call mutates
        changed2 = sync_release_metadata(str(repo))  # second call: already in sync
        assert changed2 == []

    def test_no_changes_when_already_in_sync(self, tmp_path):
        repo = _make_repo(tmp_path, "0.0.0")  # _make_repo already uses 0.0.0 in carriers
        # Overwrite VERSION to match the pre-set values
        (repo / "VERSION").write_text("0.0.0\n", encoding="utf-8")
        changed = sync_release_metadata(str(repo))
        assert changed == []

    def test_returns_empty_when_version_file_missing(self, tmp_path):
        changed = sync_release_metadata(str(tmp_path))
        assert changed == []

    def test_returns_empty_for_invalid_version_string(self, tmp_path):
        (tmp_path / "VERSION").write_text("not-a-version\n", encoding="utf-8")
        changed = sync_release_metadata(str(tmp_path))
        assert changed == []

    def test_missing_pyproject_skipped_gracefully(self, tmp_path):
        repo = _make_repo(tmp_path, "2.0.0")
        (repo / "pyproject.toml").unlink()
        changed = sync_release_metadata(str(repo))
        assert "pyproject.toml" not in changed
        # README and ARCHITECTURE still synced
        assert "README.md" in changed

    def test_missing_architecture_skipped_gracefully(self, tmp_path):
        repo = _make_repo(tmp_path, "2.0.0")
        (repo / "docs" / "ARCHITECTURE.md").unlink()
        changed = sync_release_metadata(str(repo))
        assert "docs/ARCHITECTURE.md" not in changed
        assert "pyproject.toml" in changed


# ---------------------------------------------------------------------------
# check_history_limit
# ---------------------------------------------------------------------------

def _build_readme_history(*versions: str) -> str:
    rows = "\n".join(f"| {v} | 2026-01-01 | desc |" for v in versions)
    return f"## Version History\n\n| Version | Date | Description |\n|---|---|---|\n{rows}\n"


class TestCheckHistoryLimit:
    def test_no_warnings_within_limits(self):
        readme = _build_readme_history(
            "5.0.0", "4.0.0",           # 2 major
            "4.5.0", "4.4.0", "4.3.0", "4.2.0", "4.1.0",  # 5 minor
            "4.5.5", "4.5.4", "4.5.3", "4.5.2", "4.5.1",  # 5 patch
        )
        warnings = check_history_limit(readme)
        assert warnings == []

    def test_too_many_major_rows(self):
        readme = _build_readme_history("5.0.0", "4.0.0", "3.0.0")  # 3 major
        warnings = check_history_limit(readme)
        assert any("major" in w for w in warnings)

    def test_too_many_minor_rows(self):
        readme = _build_readme_history(
            "4.6.0", "4.5.0", "4.4.0", "4.3.0", "4.2.0", "4.1.0"  # 6 minor
        )
        warnings = check_history_limit(readme)
        assert any("minor" in w for w in warnings)

    def test_too_many_patch_rows(self):
        readme = _build_readme_history(
            "4.5.6", "4.5.5", "4.5.4", "4.5.3", "4.5.2", "4.5.1"  # 6 patch
        )
        warnings = check_history_limit(readme)
        assert any("patch" in w for w in warnings)

    def test_multiple_violations_reported_separately(self):
        # 3 major + 6 minor → two separate warnings
        readme = _build_readme_history(
            "5.0.0", "4.0.0", "3.0.0",
            "4.6.0", "4.5.0", "4.4.0", "4.3.0", "4.2.0", "4.1.0",
        )
        warnings = check_history_limit(readme)
        assert len(warnings) >= 2

    def test_empty_readme_returns_no_warnings(self):
        assert check_history_limit("") == []

    def test_exact_limit_not_a_violation(self):
        readme = _build_readme_history(
            "4.0.0", "3.0.0",                          # exactly 2 major
            "4.5.0", "4.4.0", "4.3.0", "4.2.0", "4.1.0",  # exactly 5 minor
            "4.5.5", "4.5.4", "4.5.3", "4.5.2", "4.5.1",  # exactly 5 patch
        )
        warnings = check_history_limit(readme)
        assert warnings == []


# ---------------------------------------------------------------------------
# detect_numeric_claims
# ---------------------------------------------------------------------------

class TestDetectNumericClaims:
    def test_detects_N_tests(self):
        claims = detect_numeric_claims("Added 16 tests for the new module.")
        assert any("16" in c for c in claims)

    def test_detects_N_fixes(self):
        claims = detect_numeric_claims("Contains 3 fixes for edge cases.")
        assert any("3" in c for c in claims)

    def test_detects_N_new_tests(self):
        claims = detect_numeric_claims("Includes 42 new regression tests.")
        assert any("42" in c for c in claims)

    def test_no_false_positive_on_plain_numbers(self):
        claims = detect_numeric_claims("Version 4.36.3 released in 2026.")
        assert claims == []

    def test_no_false_positive_on_non_claim_nouns(self):
        claims = detect_numeric_claims("The 5 providers are all supported.")
        assert claims == []

    def test_returns_all_matches_in_text(self):
        text = "Added 5 tests and fixed 2 regressions plus 10 new assertions."
        claims = detect_numeric_claims(text)
        assert len(claims) == 3

    def test_empty_string_returns_empty(self):
        assert detect_numeric_claims("") == []

    def test_case_insensitive(self):
        claims = detect_numeric_claims("Ships 7 TESTS for reliability.")
        assert any("7" in c for c in claims)


# ---------------------------------------------------------------------------
# run_release_preflight (orchestrator)
# ---------------------------------------------------------------------------

class TestRunReleasePreflight:
    def test_returns_changed_and_warnings_tuple(self, tmp_path):
        repo = _make_repo(tmp_path, "4.99.1")
        changed, warnings = run_release_preflight(str(repo))
        assert isinstance(changed, list)
        assert isinstance(warnings, list)

    def test_syncs_carriers_and_returns_paths(self, tmp_path):
        repo = _make_repo(tmp_path, "4.99.1")
        changed, _ = run_release_preflight(str(repo))
        assert len(changed) >= 1  # at least one carrier was out of sync

    def test_second_call_idempotent_no_changes(self, tmp_path):
        repo = _make_repo(tmp_path, "4.99.1")
        run_release_preflight(str(repo))
        changed2, _ = run_release_preflight(str(repo))
        assert changed2 == []

    def test_warns_on_history_limit_breach(self, tmp_path):
        repo = _make_repo(tmp_path, "4.99.1")
        # Inject too many patch rows into README
        readme = repo / "README.md"
        extra_rows = "\n".join(
            f"| 4.99.{i} | 2026-01-01 | desc |" for i in range(10)
        )
        text = readme.read_text()
        readme.write_text(text + "\n" + extra_rows + "\n", encoding="utf-8")
        _, warnings = run_release_preflight(str(repo))
        assert any("patch" in w for w in warnings)

    def test_warns_on_numeric_claims_in_changelog_row(self, tmp_path):
        repo = _make_repo(tmp_path, "4.99.1")
        readme = repo / "README.md"
        text = readme.read_text()
        # Replace the changelog row description with a numeric claim
        text = text.replace(
            "| 4.99.1 | 2026-01-01 | New release |",
            "| 4.99.1 | 2026-01-01 | Ships 12 new tests for reliability. |",
        )
        readme.write_text(text, encoding="utf-8")
        _, warnings = run_release_preflight(str(repo))
        assert any("numeric claims" in w for w in warnings)

    def test_no_warnings_on_clean_repo(self, tmp_path):
        repo = _make_repo(tmp_path, "0.0.0")
        (repo / "VERSION").write_text("0.0.0\n", encoding="utf-8")
        _, warnings = run_release_preflight(str(repo))
        assert warnings == []

    def test_handles_missing_readme_gracefully(self, tmp_path):
        repo = _make_repo(tmp_path, "4.99.1")
        (repo / "README.md").unlink()
        changed, warnings = run_release_preflight(str(repo))
        # Should not raise; README-dependent output will be empty
        assert isinstance(changed, list)
        assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# PEP 440 pre-release (RC / alpha / beta) support
# ---------------------------------------------------------------------------


class TestNormalizePep440:
    """``_normalize_pep440`` turns author-facing RC spellings into pip-compatible form."""

    @pytest.mark.parametrize(
        "src,expected",
        [
            ("4.50.0-rc.1", "4.50.0rc1"),
            ("4.50.0rc1", "4.50.0rc1"),
            ("4.50.0-rc1", "4.50.0rc1"),
            ("4.50.0rc.1", "4.50.0rc1"),
            # PEP 440 §Pre-release spelling: ``alpha`` / ``beta`` collapse
            # to their canonical short forms ``a`` / ``b`` in the output
            # spelling (pip normalises on read anyway, but the helper's
            # docstring promises to return the canonical form).
            ("4.50.0-alpha.2", "4.50.0a2"),
            ("4.50.0-beta.3", "4.50.0b3"),
            ("4.50.0-ALPHA.2", "4.50.0a2"),  # case-insensitive alpha alias
            ("4.50.0-BETA.3", "4.50.0b3"),  # case-insensitive beta alias
            ("4.50.0-a.1", "4.50.0a1"),
            ("4.50.0-b.1", "4.50.0b1"),
            ("4.50.0-RC.1", "4.50.0rc1"),  # case-insensitive identifier
        ],
    )
    def test_rc_spellings_normalise(self, src, expected):
        assert _normalize_pep440(src) == expected

    def test_stable_version_passes_through_unchanged(self):
        assert _normalize_pep440("4.50.0") == "4.50.0"
        assert _normalize_pep440("1.2.3") == "1.2.3"


class TestNormalizeLinuxPackageVersion:
    @pytest.mark.parametrize(
        "src,expected",
        [
            ("4.50.0", "4.50.0"),
            ("4.50.0-rc.1", "4.50.0~rc1"),
            ("4.50.0rc1", "4.50.0~rc1"),
            ("4.50.0-alpha.2", "4.50.0~a2"),
            ("4.50.0-beta3", "4.50.0~b3"),
        ],
    )
    def test_normalizes_supported_release_spellings(self, src, expected):
        assert normalize_linux_package_version(src) == expected

    def test_rejects_a_non_release_version(self):
        with pytest.raises(ValueError, match="unsupported release version"):
            normalize_linux_package_version("dev")


class TestShieldsEscape:
    """shields.io badge URL path-segment escape."""

    def test_hyphen_doubled_for_rc(self):
        assert _shields_escape("4.50.0-rc.1") == "4.50.0--rc.1"

    def test_plain_version_unchanged(self):
        assert _shields_escape("4.50.0") == "4.50.0"

    def test_multiple_hyphens_all_doubled(self):
        assert _shields_escape("1.2.3-rc-1") == "1.2.3--rc--1"


class TestVersionRegexAcceptsRc:
    """The widened ``_VERSION_RE`` accepts both plain and RC spellings."""

    @pytest.mark.parametrize(
        "ver",
        [
            "4.50.0",
            "4.50.0-rc.1",
            "4.50.0rc1",
            "4.50.0-alpha.2",
            "4.50.0-beta.3",
            "4.50.0-a.1",
            "4.50.0-b.1",
        ],
    )
    def test_accepts_valid_versions(self, ver):
        assert _VERSION_RE.match(ver) is not None, ver

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "not-a-version",
            "4.50",  # too few segments
            "4.50.0.1",  # too many segments
            "4.50.0-foo.1",  # unknown identifier
            "v4.50.0",  # leading 'v' not allowed
        ],
    )
    def test_rejects_invalid_versions(self, bad):
        assert _VERSION_RE.match(bad) is None, bad


def _make_rc_repo(tmp_path: Path, version: str = "4.50.0-rc.1") -> Path:
    """Like _make_repo but with a realistic pre-existing plain carrier set.

    Mimics the real ``ouroboros`` tree state right before the VERSION
    bump: VERSION carries the new RC spelling, but badge / pyproject /
    ARCHITECTURE still reference the previous stable (``4.50.0``) —
    ``sync_release_metadata`` must migrate them to the RC spelling.
    """
    (tmp_path / "VERSION").write_text(version + "\n", encoding="utf-8")

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "ouroboros"\nversion = "4.50.0"\n',
        encoding="utf-8",
    )

    # Existing plain-version badge that was in-sync for ``4.50.0``.
    badge_line = (
        '[![Version 4.50.0]'
        '(https://img.shields.io/badge/version-4.50.0-green.svg)]'
        '(VERSION)\n'
    )
    readme_content = (
        "# Ouroboros\n\n"
        + badge_line
        + "\n## Version History\n\n"
        "| Version | Date | Description |\n"
        "|---------|------|-------------|\n"
        "| 4.50.0 | 2026-04-21 | Phase 6 landed |\n"
    )
    (tmp_path / "README.md").write_text(readme_content, encoding="utf-8")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ARCHITECTURE.md").write_text(
        "# Ouroboros v4.50.0 — Architecture\n\nContent here.\n",
        encoding="utf-8",
    )
    return tmp_path


class TestSyncReleaseMetadataRc:
    """``sync_release_metadata`` handles PEP 440 RC versions end-to-end."""

    def test_syncs_pyproject_to_canonical_pep440(self, tmp_path):
        repo = _make_rc_repo(tmp_path, "4.50.0-rc.1")
        changed = sync_release_metadata(str(repo))
        assert "pyproject.toml" in changed
        text = (repo / "pyproject.toml").read_text()
        # PEP 440 canonical: no separator between base and ``rc``, no dot.
        assert 'version = "4.50.0rc1"' in text
        # And NOT the author-facing spelling.
        assert 'version = "4.50.0-rc.1"' not in text

    def test_syncs_readme_badge_display_and_url_differ(self, tmp_path):
        repo = _make_rc_repo(tmp_path, "4.50.0-rc.1")
        changed = sync_release_metadata(str(repo))
        assert "README.md" in changed
        text = (repo / "README.md").read_text()
        # Display text keeps the author spelling (single hyphen).
        assert "[![Version 4.50.0-rc.1]" in text
        # URL path doubles the literal hyphen per shields.io escape.
        assert "badge/version-4.50.0--rc.1-green" in text

    def test_syncs_architecture_header(self, tmp_path):
        repo = _make_rc_repo(tmp_path, "4.50.0-rc.1")
        changed = sync_release_metadata(str(repo))
        assert "docs/ARCHITECTURE.md" in changed
        text = (repo / "docs" / "ARCHITECTURE.md").read_text()
        assert "v4.50.0-rc.1" in text
        # Must not leave the old stable spelling around.
        assert "v4.50.0 —" not in text

    def test_idempotent_on_rc_version(self, tmp_path):
        repo = _make_rc_repo(tmp_path, "4.50.0-rc.1")
        sync_release_metadata(str(repo))
        changed2 = sync_release_metadata(str(repo))
        assert changed2 == []

    def test_accepts_canonical_pep440_input(self, tmp_path):
        """VERSION=``4.50.0rc1`` (already canonical) works too."""
        repo = _make_rc_repo(tmp_path, "4.50.0rc1")
        changed = sync_release_metadata(str(repo))
        # At least pyproject should update (since its starting value was 4.50.0).
        assert "pyproject.toml" in changed
        pyproject_text = (repo / "pyproject.toml").read_text()
        assert 'version = "4.50.0rc1"' in pyproject_text

    def test_rejects_unparseable_version(self, tmp_path):
        repo = _make_rc_repo(tmp_path, "4.50.0-rc.1")
        (repo / "VERSION").write_text("not-a-version\n", encoding="utf-8")
        assert sync_release_metadata(str(repo)) == []


class TestVersionRowRegexBucketingWithRc:
    """A pre-release row buckets under its base semver slot."""

    def test_rc_row_counted_as_minor(self):
        readme = (
            "## Version History\n\n"
            "| Version | Date | Description |\n"
            "|---|---|---|\n"
            "| 4.50.0-rc.1 | 2026-04-21 | RC of phase 6 |\n"
            "| 4.50.0 | 2026-04-21 | phase 6 |\n"
        )
        rows = list(_VERSION_ROW_RE.finditer(readme))
        assert len(rows) == 2
        # Both rows parse as X=4, Y=50, Z=0 (RC and its base share the slot).
        x0, y0, z0 = (int(g) for g in rows[0].groups())
        x1, y1, z1 = (int(g) for g in rows[1].groups())
        assert (x0, y0, z0) == (4, 50, 0)
        assert (x1, y1, z1) == (4, 50, 0)

    def test_rc_rows_do_not_break_history_limit_check(self):
        # 2 majors + 5 minors + 1 RC (also a minor slot) + 4 patches:
        # within P9 caps overall.
        readme = (
            "## Version History\n\n"
            "| Version | Date | Description |\n"
            "|---|---|---|\n"
            "| 5.0.0 | 2026-01-01 | major |\n"
            "| 4.0.0 | 2026-01-01 | major |\n"
            "| 4.5.0 | 2026-01-01 | minor |\n"
            "| 4.4.0 | 2026-01-01 | minor |\n"
            "| 4.3.0 | 2026-01-01 | minor |\n"
            "| 4.2.0 | 2026-01-01 | minor |\n"
            "| 4.1.0 | 2026-01-01 | minor |\n"
            "| 4.1.0-rc.1 | 2026-01-01 | minor-rc |\n"
            "| 4.5.5 | 2026-01-01 | patch |\n"
            "| 4.5.4 | 2026-01-01 | patch |\n"
            "| 4.5.3 | 2026-01-01 | patch |\n"
            "| 4.5.2 | 2026-01-01 | patch |\n"
        )
        warnings = check_history_limit(readme)
        # 6 minor rows (5 + 1 RC sharing minor slot) — should warn.
        assert any("minor" in w for w in warnings)


class TestBadgeAndArchRegexAcceptRc:
    """Carrier-regex smoke: make sure the RC spellings match existing prose."""

    def test_badge_regex_matches_rc_badge(self):
        badge = (
            '[![Version 4.50.0-rc.1]'
            '(https://img.shields.io/badge/version-4.50.0--rc.1-green.svg)](VERSION)'
        )
        assert _README_BADGE_RE.search(badge) is not None

    def test_arch_header_regex_matches_rc_header(self):
        header = "# Ouroboros v4.50.0-rc.1 — Three-layer refactor\n"
        assert _ARCH_HEADER_RE.search(header) is not None

    def test_arch_header_regex_still_matches_stable(self):
        header = "# Ouroboros v4.50.0 — Three-layer refactor\n"
        assert _ARCH_HEADER_RE.search(header) is not None


# ---------------------------------------------------------------------------
# The development-cycle marker (BIBLE P9); the whole cycle: tests/test_development_cycle.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["7.10.0-alpha.0", "7.10.0a0", "7.10.0-ALPHA.0", "7.10.0alpha0", "7.10.0-a.0"])
def test_alpha_zero_is_the_marker_in_every_spelling_of_the_grammar(version):
    assert is_development_marker(version) and is_development_marker(f" {version}\n")
    assert is_release_version(version)  # its carriers keep the shared grammar...
    assert normalize_release_tag(f"v{version}") == ""  # ...but it never names a release tag


@pytest.mark.parametrize("version", ["7.10.0", "7.10.0-alpha.1", "7.10.0a1", "7.10.0-alpha.10", "7.10.0-beta.0",
                                     "7.10.0-rc.0", "7.10.0-test", "", None])
def test_nothing_else_is_the_marker(version):
    assert not is_development_marker(version)


def test_the_marker_keeps_its_package_spellings():
    assert _normalize_pep440("7.10.0-alpha.0") == "7.10.0a0"
    assert normalize_linux_package_version("7.10.0-alpha.0") == "7.10.0~a0"
    assert _shields_escape("7.10.0-alpha.0") == "7.10.0--alpha.0"


@pytest.mark.parametrize("released,development,problem", [
    ("7.9.0", "7.10.0-alpha.0", ""),  # numeric: as text 7.10 sorts below 7.9
    ("7.10.0", "7.10.1-alpha.0", ""),
    ("7.10.0", "7.11.0-alpha.0", ""),
    ("7.10.3", "8.0.0-alpha.0", ""),
    ("7.10.0-rc.1", "7.10.0-alpha.0", ""),  # a prerelease belongs to its own base's cycle
    ("7.10.0", "7.10.0-alpha.0", "does not sort above"),
    ("7.10.0", "7.9.9-alpha.0", "does not sort above"),
    ("7.10.0", "7.10.0", "is not a development marker"),
    ("7.10.0", "7.10.1", "is not a development marker"),
    ("7.10.0", "7.10.1-alpha.1", "is not a development marker"),
    ("7.10.0", "", "an empty VERSION"),
])
def test_next_cycle_problem(released, development, problem):
    answer = next_cycle_problem(released, development)
    assert (problem in answer and bool(answer)) if problem else answer == ""


def test_a_marker_syncs_every_carrier_but_leaves_links_and_history_alone(tmp_path):
    repo = _make_repo(tmp_path, "7.9.0")
    sync_release_metadata(str(repo))
    before = {rel: (repo / rel).read_text(encoding="utf-8")
              for rel in ("site/install/index.html", "docs/install/index.html")}
    readme_before = (repo / "README.md").read_text(encoding="utf-8")

    (repo / "VERSION").write_text("7.10.0-alpha.0\n", encoding="utf-8")
    would_write = sync_release_metadata(str(repo), write=False)
    assert sync_release_metadata(str(repo)) == would_write
    assert sync_release_metadata(str(repo)) == []  # idempotent
    assert not {"site/install/index.html", "docs/install/index.html"} & set(would_write)
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert _download_references("7.9.0") in readme and "| 7.9.0 | 2026-01-01 | New release |" in readme
    assert "7.10.0-alpha.0/" not in readme and "| 7.10.0-alpha.0 |" not in readme
    assert readme.replace("Version 7.10.0-alpha.0", "Version 7.9.0").replace("version-7.10.0--alpha.0-", "version-7.9.0-") \
        == readme_before
    for rel, text in before.items():
        assert (repo / rel).read_text(encoding="utf-8") == text
    assert check_worktree_version_sync(repo) == ""


def test_a_marker_projects_links_only_to_the_release_its_caller_names(tmp_path):
    repo = _make_repo(tmp_path, "7.9.0")
    sync_release_metadata(str(repo))
    (repo / "VERSION").write_text("7.10.0-alpha.0\n", encoding="utf-8")
    sync_release_metadata(str(repo))
    before = {rel: (repo / rel).read_text(encoding="utf-8")
              for rel in ("README.md", "site/install/index.html", "docs/install/index.html")}
    with pytest.raises(ValueError, match="Cannot determine whether published release 7.9.1"):
        sync_release_metadata(str(repo), published_version="7.9.1")
    for rel, text in before.items():
        assert (repo / rel).read_text(encoding="utf-8") == text
    assert sync_release_metadata(str(repo), published_version="7.9.1", windows_installer=True) == [
        "README.md", "site/install/index.html", "docs/install/index.html"]
    assert _download_references("7.9.1") in (repo / "README.md").read_text(encoding="utf-8")
    assert (repo / "docs/install/index.html").read_text(encoding="utf-8") == _download_page("7.9.1")
    # A release derives its links from VERSION; the argument cannot move them.
    (repo / "VERSION").write_text("7.10.0\n", encoding="utf-8")
    sync_release_metadata(str(repo), published_version="7.9.1")
    assert _download_references("7.10.0") in (repo / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("readme,site,docs,labelled", [
    ("7.9.0", "7.9.0", "7.9.0", False),
    ("7.10.0-alpha.0", "7.10.0-alpha.0", "7.10.0-alpha.0", True),  # the dead links an old sync writes
    ("7.10.0", "7.10.0", "7.10.0", True),  # the cycle's own, unpublished base
    ("7.11.0", "7.11.0", "7.11.0", True),
    ("7.9.0-rc.1", "7.9.0-rc.1", "7.9.0-rc.1", True),
    ("7.9.0", "7.8.0", "7.9.0", True),  # mixed
    ("7.9.0", "7.9.0", "7.10.0-alpha.0", True),
])
def test_a_marker_checks_one_published_plain_release_below_it(readme, site, docs, labelled):
    desync = version_carrier_desyncs(
        "7.10.0-alpha.0", download_readme_text=_download_references(readme),
        site_install_text=_download_page(site), docs_install_text=_download_page(docs), detailed=True)
    assert bool(desync) is labelled, desync
    if labelled:
        assert "expected one published plain release below 7.10.0-alpha.0" in desync[0]


def test_a_marker_catches_one_inexact_link_of_its_published_release():
    references = _download_references("7.9.0").replace("Ouroboros-7.9.0.dmg", "Other-7.9.0.dmg")
    desync = version_carrier_desyncs("7.10.0-alpha.0", download_readme_text=references)
    assert desync == ["README.md download macos-arm64"]


def test_a_marker_needs_no_projection_where_a_tree_has_none():
    assert version_carrier_desyncs("7.10.0-alpha.0", download_readme_text="# no links here\n") == []


def test_published_download_version_reads_only_the_projection():
    historic = "See https://github.com/razzant/ouroboros/releases/download/v6.0.0/old.dmg\n"
    assert published_download_version(historic + _download_references("7.9.0")) == "7.9.0"
    assert published_download_version(_download_page("7.9.0")) == "7.9.0"
    assert published_download_version(_download_references("7.9.0"), _download_page("7.8.0")) == ""
    assert published_download_version(historic) == "" and published_download_version() == ""


# ---------------------------------------------------------------------------
# Version-aware Windows installer (primary) + portable ZIP (secondary)
# ---------------------------------------------------------------------------

def _seeded_real_repo(
    tmp_path: Path,
    *,
    version: str = "7.12.2-alpha.0",
    published_version: str = "7.12.1",
    windows_installer: bool = False,
) -> Path:
    """Copy the real carrier templates and normalize them to an explicit starting state."""
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "VERSION", "pyproject.toml", "uv.lock", "web/package.json", "web/package-lock.json",
        "web/modules/api_types.js", "README.md", "docs/ARCHITECTURE.md",
        "site/install/index.html", "docs/install/index.html",
    ):
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text((root / rel).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    (tmp_path / "VERSION").write_text(f"{version}\n", encoding="utf-8", newline="\n")
    sync_release_metadata(str(tmp_path), published_version=published_version, windows_installer=windows_installer)
    return tmp_path


def _snapshot_carriers(repo: Path) -> dict[str, str]:
    return {
        rel: (repo / rel).read_text(encoding="utf-8")
        for rel in ("pyproject.toml", "README.md", "site/install/index.html", "docs/install/index.html")
        if (repo / rel).exists()
    }


def _git_local(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    )


class TestWindowsInstallerProjection:
    def test_windows_registry_and_projection_detection(self):
        assert release_asset_name(WINDOWS_SETUP_PROOF_ID, "7.12.2") == "Ouroboros-Setup-7.12.2-windows-x64.exe"
        assert release_asset_name(WINDOWS_ZIP_PROOF_ID, "7.12.2") == "Ouroboros-7.12.2-windows-x64.zip"
        with_setup, zip_only = desktop_download_ids(windows_installer=True), desktop_download_ids(windows_installer=False)
        assert with_setup == DESKTOP_DOWNLOAD_IDS
        assert with_setup.index(WINDOWS_SETUP_PROOF_ID) < with_setup.index(WINDOWS_ZIP_PROOF_ID)
        assert WINDOWS_SETUP_PROOF_ID not in zip_only and WINDOWS_ZIP_PROOF_ID in zip_only
        refs_setup = _download_references("7.12.2", windows_installer=True)
        refs_zip = _download_references("7.12.2", windows_installer=False)
        page_setup = _download_page("7.12.2", windows_installer=True)
        page_zip = _download_page("7.12.2", windows_installer=False)
        assert download_links_include_windows_installer(refs_setup) is True
        assert download_links_include_windows_installer(refs_zip) is False
        assert download_links_include_windows_installer(page_setup) is True
        assert download_links_include_windows_installer(page_zip) is False
        assert download_links_include_windows_installer(refs_setup, page_setup, version="7.12.2") is True
        assert download_links_include_windows_installer(refs_setup, version="7.12.1") is None
        # Plain text, Mac-only projections, foreign-project links, empty/malformed/mismatched Windows
        # markers, and conflicting surfaces must remain None (uncertain), never silently coerce.
        mac_only_md = f"[download-macos-arm64]: {release_asset_download_url('macos-arm64', '7.12.1')}\n"
        mac_only_html = f'<a data-release-download="macos-arm64" href="{release_asset_download_url("macos-arm64", "7.12.1")}">mac</a>\n'
        zip_7121 = f"[download-{WINDOWS_ZIP_PROOF_ID}]: {release_asset_download_url(WINDOWS_ZIP_PROOF_ID, '7.12.1')}\n"
        foreign_zip_url = release_asset_download_url(WINDOWS_ZIP_PROOF_ID, "7.12.1", repository="other/project")
        foreign_setup_url = release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.1", repository="other/project")
        for uncertain in (
            ("# Plain markdown\n", "<h1>Install</h1>\n"),
            (mac_only_md,),
            (mac_only_html,),
            (mac_only_md, mac_only_html),
            (f"[Other]({foreign_setup_url})\n", f'<a href="{foreign_zip_url}">Other</a>\n'),
            (f"[download-{WINDOWS_ZIP_PROOF_ID}]: {foreign_zip_url}\n",),
            (f"[download-{WINDOWS_SETUP_PROOF_ID}]: {foreign_setup_url}\n{zip_7121}",),
            (f'<a data-release-download="{WINDOWS_ZIP_PROOF_ID}" href="{foreign_zip_url}">ZIP</a>\n',),
            (
                f'<a data-release-download="{WINDOWS_SETUP_PROOF_ID}" href="{foreign_setup_url}">Setup</a>\n'
                f'<a data-release-download="{WINDOWS_ZIP_PROOF_ID}" href="{release_asset_download_url(WINDOWS_ZIP_PROOF_ID, "7.12.1")}">ZIP</a>\n',
            ),
            (f"[download-{WINDOWS_SETUP_PROOF_ID}]:\n",),
            (f"[download-{WINDOWS_SETUP_PROOF_ID}]:\n{zip_7121}",),
            (f"[download-{WINDOWS_SETUP_PROOF_ID}]: https://example.invalid/not-a-release.exe\n{zip_7121}",),
            (f"[download-{WINDOWS_SETUP_PROOF_ID}]: {release_asset_download_url(WINDOWS_ZIP_PROOF_ID, '7.12.1')}\n{zip_7121}",),
            (f"[download-{WINDOWS_SETUP_PROOF_ID}]: https://github.com/razzant/ouroboros/releases/download/v7.12.2/Ouroboros-Setup-7.12.1-windows-x64.exe\n{zip_7121}",),
            (f"[download-{WINDOWS_ZIP_PROOF_ID}]: {release_asset_download_url(WINDOWS_ZIP_PROOF_ID, '7.12.2-alpha.0')}\n",),
            (f'<a data-release-download="{WINDOWS_SETUP_PROOF_ID}" href="">Setup</a>\n',),
            (f'<a data-release-download="{WINDOWS_ZIP_PROOF_ID}" data-href="{release_asset_download_url(WINDOWS_ZIP_PROOF_ID, "7.12.1")}">ZIP</a>\n',),
            (f'<a data-release-download="{WINDOWS_SETUP_PROOF_ID}" href="{release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2")}">Setup</a>\n',),
            (refs_setup, page_zip),
            (refs_setup, _download_references("7.12.1", windows_installer=True)),
        ):
            assert download_links_include_windows_installer(*uncertain) is None, uncertain
        # Foreign-project Windows URLs are also rejected by version_carrier_desyncs (never accepted as ZIP-only).
        foreign_refs = _download_references("7.12.1", windows_installer=False).replace(
            release_asset_download_url(WINDOWS_ZIP_PROOF_ID, "7.12.1"), foreign_zip_url,
        )
        foreign_desync = version_carrier_desyncs("7.12.1", download_readme_text=foreign_refs)
        assert f"README.md download {WINDOWS_ZIP_PROOF_ID}" in foreign_desync
        assert f"README.md download {WINDOWS_SETUP_PROOF_ID}" in foreign_desync

    def test_marker_repair_refuses_ambiguous_target_and_uses_exact_evidence(self, tmp_path):
        def _repo(name: str, version: str) -> Path:
            path = tmp_path / name
            path.mkdir()
            return _make_repo(path, version)

        # Counterexample 1: from dual 88.77.66, repairing to ZIP-only 7.12.1 with NO flag/tag must refuse
        # BEFORE modifying any file (even if pyproject.toml is stale) instead of borrowing 88.77.66's Setup.
        ce1 = _repo("ce1", "88.77.66")
        sync_release_metadata(str(ce1))
        (ce1 / "VERSION").write_text("88.77.67-alpha.0\n", encoding="utf-8")
        before_ce1 = _snapshot_carriers(ce1)
        for write_mode in (False, True):
            with pytest.raises(ValueError, match="Cannot determine whether published release 7.12.1"):
                sync_release_metadata(str(ce1), write=write_mode, published_version="7.12.1")
        assert _snapshot_carriers(ce1) == before_ce1

        # Counterexample 2: merge dropped ALL 7.12.2 links back to stale 7.12.1 ZIP-only; repairing to
        # 7.12.2 with NO flag/tag must refuse BEFORE writing instead of borrowing 7.12.1's ZIP-only shape.
        ce2 = _repo("ce2", "7.12.1")
        sync_release_metadata(str(ce2), windows_installer=False)
        (ce2 / "VERSION").write_text("7.12.3-alpha.0\n", encoding="utf-8")
        before_ce2 = _snapshot_carriers(ce2)
        with pytest.raises(ValueError, match="Cannot determine whether published release 7.12.2"):
            sync_release_metadata(str(ce2), published_version="7.12.2")
        assert _snapshot_carriers(ce2) == before_ce2

        # Positive A (local git tags): local tag v7.12.1 (ZIP-only) and v7.12.2 (dual) resolve BOTH unflagged repairs.
        tagged = _repo("tagged", "7.12.1")
        _git_local(tagged, "init", "-q")
        sync_release_metadata(str(tagged), windows_installer=False)
        _git_local(tagged, "add", "-A")
        _git_local(tagged, "commit", "-qm", "release 7.12.1")
        _git_local(tagged, "tag", "v7.12.1")
        (tagged / "VERSION").write_text("7.12.2\n", encoding="utf-8")
        sync_release_metadata(str(tagged))
        _git_local(tagged, "add", "-A")
        _git_local(tagged, "commit", "-qm", "release 7.12.2")
        _git_local(tagged, "tag", "v7.12.2")
        # From dual 88.77.66 -> unflagged repair to tagged ZIP-only 7.12.1 stays ZIP-only!
        (tagged / "VERSION").write_text("88.77.66\n", encoding="utf-8")
        sync_release_metadata(str(tagged))
        (tagged / "VERSION").write_text("88.77.67-alpha.0\n", encoding="utf-8")
        sync_release_metadata(str(tagged), published_version="7.12.1")
        assert WINDOWS_SETUP_PROOF_ID not in (tagged / "README.md").read_text(encoding="utf-8")
        assert WINDOWS_SETUP_PROOF_ID not in (tagged / "site/install/index.html").read_text(encoding="utf-8")
        assert check_worktree_version_sync(tagged) == ""
        # From stale 7.12.1 ZIP-only -> unflagged repair to tagged dual 7.12.2 restores Setup + ZIP!
        (tagged / "VERSION").write_text("7.12.3-alpha.0\n", encoding="utf-8")
        sync_release_metadata(str(tagged), published_version="7.12.2")
        assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2") in (tagged / "README.md").read_text(encoding="utf-8")
        assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2") in (tagged / "site/install/index.html").read_text(encoding="utf-8")
        assert check_worktree_version_sync(tagged) == ""
        # A same-named local branch without the tag must NOT count as tag evidence when current files point elsewhere.
        _git_local(tagged, "branch", "v7.12.1", "refs/tags/v7.12.1")
        _git_local(tagged, "tag", "-d", "v7.12.1")
        before_branch_only = _snapshot_carriers(tagged)
        with pytest.raises(ValueError, match="Cannot determine whether published release 7.12.1"):
            sync_release_metadata(str(tagged), published_version="7.12.1")
        assert _snapshot_carriers(tagged) == before_branch_only

        # Positive B (surviving matching surface without git tag) & disagreement refusal.
        partial = _repo("partial", "7.12.1")
        sync_release_metadata(str(partial), windows_installer=False)
        (partial / "VERSION").write_text("7.12.3-alpha.0\n", encoding="utf-8")
        (partial / "site/install/index.html").write_text(_download_page("7.12.2", windows_installer=True), encoding="utf-8")
        changed = sync_release_metadata(str(partial), published_version="7.12.2")
        assert "README.md" in changed and "docs/install/index.html" in changed
        assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2") in (partial / "README.md").read_text(encoding="utf-8")
        # Conflicting surfaces for the same version without a tag must refuse without writing.
        (partial / "docs/install/index.html").write_text(_download_page("7.12.2", windows_installer=False), encoding="utf-8")
        before_conflict = _snapshot_carriers(partial)
        with pytest.raises(ValueError, match="Cannot determine whether published release 7.12.2"):
            sync_release_metadata(str(partial), published_version="7.12.2")
        assert _snapshot_carriers(partial) == before_conflict

    def test_saved_v7_12_1_tag_and_tagged_broken_cut_and_unreadable_build_probe(self, tmp_path):
        from ouroboros import commit_admission as admission
        from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS, release_metadata_findings

        repo_root = Path(__file__).resolve().parents[1]
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", "refs/tags/v7.12.1^{commit}"],
                          cwd=repo_root, capture_output=True, check=False).returncode != 0:
            pytest.skip("local tag refs/tags/v7.12.1 is unavailable in this checkout")

        saved = {}
        for rel in (*sorted(CARRIER_SPAN_PATHS), "build_windows.ps1", "scripts/pack_windows_setup.ps1"):
            proc = subprocess.run(["git", "show", f"refs/tags/v7.12.1:{rel}"], cwd=repo_root, capture_output=True, check=False)
            if proc.returncode == 0:
                saved[rel] = proc.stdout.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        assert saved.get("VERSION", "").strip() == "7.12.1" and "build_windows.ps1" in saved
        assert "scripts/pack_windows_setup.ps1" not in saved

        def _write_saved(dest_root: Path) -> None:
            for rel, text in saved.items():
                dest = dest_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf-8", newline="\n")

        # 1. Genuine historical v7.12.1 (tagless, tagged, and tagged even with newer worktree Setup script):
        # writer and checkers agree on ZIP-only and never invent Setup.
        hist = tmp_path / "genuine_v7_12_1"
        _write_saved(hist)
        assert resolve_release_windows_installer(hist, "7.12.1") is False
        assert sync_release_metadata(str(hist)) == [] and check_worktree_version_sync(hist) == ""
        assert release_metadata_findings(
            {rel: saved[rel] for rel in CARRIER_SPAN_PATHS if rel in saved},
            windows_installer=resolve_release_windows_installer(hist, "7.12.1"),
        ) == []
        _git_local(hist, "init", "-q")
        (hist / "VERSION").write_text("7.12.0\n", encoding="utf-8", newline="\n")
        _git_local(hist, "add", "VERSION")
        _git_local(hist, "commit", "-qm", "base")
        (hist / "VERSION").write_text(saved["VERSION"], encoding="utf-8", newline="\n")
        _git_local(hist, "add", "-A")
        for src in ("worktree", "index"):
            rep = admission.release_metadata_diagnostics(hist, source=src, neutral_allowed=False)
            assert (rep["form"], rep["status"], rep["findings"]) == ("numbered", "clean", []), (src, rep)
        _git_local(hist, "commit", "-qm", "v7.12.1")
        _git_local(hist, "tag", "v7.12.1")
        (hist / "scripts").mkdir(exist_ok=True)
        (hist / "scripts" / "pack_windows_setup.ps1").write_text("# future setup\n", encoding="utf-8", newline="\n")
        assert resolve_release_windows_installer(hist, "7.12.1") is False
        assert sync_release_metadata(str(hist)) == [] and check_worktree_version_sync(hist) == ""

        # 2. Current 7.12.2 cut (Setup build) with coherent wrong ZIP-only carriers — BOTH untagged AND tagged v7.12.2:
        # must resolve True (from build inventory, never validating a tagged README against itself) and reject ZIP-only.
        future = tmp_path / "cut_7_12_2"
        _write_saved(future)
        (future / "scripts").mkdir(exist_ok=True)
        (future / "scripts" / "pack_windows_setup.ps1").write_text("# setup build\n", encoding="utf-8", newline="\n")
        _git_local(future, "init", "-q")
        _git_local(future, "add", "-A")
        _git_local(future, "commit", "-qm", "base")
        (future / "VERSION").write_text("7.12.2\n", encoding="utf-8", newline="\n")
        (future / "README.md").write_text(saved["README.md"].replace("| 7.12.1 |", "| 7.12.2 |", 1), encoding="utf-8", newline="\n")
        sync_release_metadata(str(future), windows_installer=False)
        _git_local(future, "add", "-A")

        def _assert_broken_zip_only_rejected(label: str, *, staged_diff: bool) -> None:
            assert resolve_release_windows_installer(future, "7.12.2") is True, label
            assert set(sync_release_metadata(str(future), write=False)) >= {"README.md", "site/install/index.html", "docs/install/index.html"}, label
            assert f"README.md download {WINDOWS_SETUP_PROOF_ID}" in check_worktree_version_sync(future), label
            wt = admission.release_metadata_diagnostics(future, ["VERSION", "README.md"], source="worktree", neutral_allowed=False)
            assert wt["status"] == "blocked" and any(WINDOWS_SETUP_PROOF_ID in f for f in wt["findings"]), (label, wt)
            idx_kwargs = {} if staged_diff else {"read_text": lambda rel: admission.read_release_file(future, rel, source="index")}
            idx = admission.release_metadata_diagnostics(future, ["VERSION", "README.md"], source="index", neutral_allowed=False, **idx_kwargs)
            assert idx["status"] == "blocked" and any(WINDOWS_SETUP_PROOF_ID in f for f in idx["findings"]), (label, idx)

        _assert_broken_zip_only_rejected("untagged", staged_diff=True)

        # 3. Unreadable/unmerged build script in commit_admission surfaces as unavailable (never swallowed as False).
        blob = subprocess.check_output(["git", "rev-parse", ":scripts/pack_windows_setup.ps1"], cwd=future, text=True).strip()
        subprocess.run(
            ["git", "update-index", "--index-info"], cwd=future, check=True,
            input=f"0 {'0' * 40}\tscripts/pack_windows_setup.ps1\n100644 {blob} 1\tscripts/pack_windows_setup.ps1\n100644 {blob} 2\tscripts/pack_windows_setup.ps1\n".encode(),
        )
        unavail = admission.release_metadata_diagnostics(future, source="index", neutral_allowed=False)
        assert unavail["status"] == "unavailable"
        assert any("scripts/pack_windows_setup.ps1" in u for u in unavail["unavailable"]), unavail
        assert not any(WINDOWS_SETUP_PROOF_ID in f for f in unavail["findings"]), unavail
        _git_local(future, "add", "scripts/pack_windows_setup.ps1")

        # 4. Tagged broken current cut (Setup build + wrong ZIP-only README) still rejects ZIP-only.
        _git_local(future, "commit", "-qm", "broken tagged zip-only 7.12.2")
        _git_local(future, "tag", "v7.12.2")
        _assert_broken_zip_only_rejected("tagged", staged_diff=False)

        # 5. Running the regular writer projects Setup + ZIP cleanly.
        sync_release_metadata(str(future))
        _git_local(future, "add", "-A")
        assert check_worktree_version_sync(future) == ""
        ok_wt = admission.release_metadata_diagnostics(future, ["VERSION", "README.md"], source="worktree", neutral_allowed=False)
        ok_idx = admission.release_metadata_diagnostics(
            future, ["VERSION", "README.md"], source="index", neutral_allowed=False,
            read_text=lambda rel: admission.read_release_file(future, rel, source="index"),
        )
        for src, ok in (("worktree", ok_wt), ("index", ok_idx)):
            assert (ok["form"], ok["status"], ok["findings"]) == ("numbered", "clean", []), (src, ok)

    @pytest.mark.parametrize("start_has_setup", [False, True])
    def test_zip_only_hotfix_marker_and_numbered_release_transition(self, tmp_path, start_has_setup):
        from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS, release_metadata_findings

        # Already-released numbered v7.12.1 (ZIP-only) passes release_metadata_findings and worktree sync
        # cleanly when its build tree (build_windows.ps1 without pack_windows_setup.ps1) matches ZIP-only,
        # and regular sync does NOT invent Setup.
        hist = _seeded_real_repo(
            tmp_path / "hist_7121", version="7.12.1", published_version="7.12.1", windows_installer=False
        )
        (hist / "build_windows.ps1").write_text("# zip build\n", encoding="utf-8", newline="\n")
        assert sync_release_metadata(str(hist)) == []
        assert check_worktree_version_sync(hist) == ""
        hist_texts = {p: (hist / p).read_text(encoding="utf-8") for p in CARRIER_SPAN_PATHS if (hist / p).exists()}
        assert release_metadata_findings(
            hist_texts, windows_installer=resolve_release_windows_installer(hist, "7.12.1"),
        ) == []
        hist_card = hist_texts["site/install/index.html"].split('<article id="windows">', 1)[1].split("</article>", 1)[0]
        hist_qs = hist_texts["README.md"].split("### Windows quick start", 1)[1].split("### Linux quick start", 1)[0]
        assert "Extract the ZIP to a permanent folder" in hist_card and "installer" not in hist_card.lower()
        assert "Download for Windows (.zip)" in hist_qs and "(.exe)" not in hist_qs and "installer" not in hist_qs.lower()

        # Intervening ZIP-only hotfix 7.12.2 under 7.13.0-alpha.0 stays clean and rejects stray Setup.
        hotfix = _seeded_real_repo(
            tmp_path / "hotfix", version="7.13.0-alpha.0", published_version="7.12.2", windows_installer=False
        )
        readme = (hotfix / "README.md").read_text(encoding="utf-8")
        site = (hotfix / "site/install/index.html").read_text(encoding="utf-8")
        docs = (hotfix / "docs/install/index.html").read_text(encoding="utf-8")
        assert check_worktree_version_sync(hotfix) == ""
        setup_url_7122 = release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2")
        bad_site = site.replace(
            'data-release-download="windows-x64"',
            f'data-release-download="{WINDOWS_SETUP_PROOF_ID}" href="{setup_url_7122}">x</a>'
            '<a data-release-download="windows-x64"',
            1,
        )
        assert "Windows download projection" in version_carrier_desyncs(
            "7.13.0-alpha.0", download_readme_text=readme, site_install_text=bad_site, docs_install_text=docs,
        )
        assert f"site/install/index.html unexpected download {WINDOWS_SETUP_PROOF_ID}" in version_carrier_desyncs(
            "7.13.0-alpha.0", download_readme_text=readme, site_install_text=bad_site, docs_install_text=docs,
            windows_installer=False,
        )
        # Numbered 7.12.2 ZIP-only MUST NOT silently validate under default Setup-shipping expectation,
        # while explicit windows_installer=False validates a ZIP-only hotfix.
        assert f"README.md download {WINDOWS_SETUP_PROOF_ID}" in version_carrier_desyncs(
            "7.12.2", download_readme_text=readme, site_install_text=site, docs_install_text=docs,
        )
        assert version_carrier_desyncs(
            "7.12.2", download_readme_text=readme, site_install_text=site, docs_install_text=docs,
            windows_installer=False,
        ) == []

        # New numbered cut from either old ZIP-only or already-cut state projects Setup primary + ZIP secondary.
        repo = _seeded_real_repo(
            tmp_path / "cut", version="7.12.2-alpha.0", published_version="7.12.1", windows_installer=start_has_setup
        )
        (repo / "VERSION").write_text("7.12.2-rc.1\n", encoding="utf-8")
        changed = sync_release_metadata(str(repo))
        assert {"README.md", "site/install/index.html", "docs/install/index.html"} <= set(changed)
        assert sync_release_metadata(str(repo)) == []
        assert check_worktree_version_sync(repo) == ""
        setup_url = release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2-rc.1")
        zip_url = release_asset_download_url(WINDOWS_ZIP_PROOF_ID, "7.12.2-rc.1")
        cut_readme = (repo / "README.md").read_text(encoding="utf-8")
        assert cut_readme.index(f"[download-{WINDOWS_SETUP_PROOF_ID}]: {setup_url}") < cut_readme.index(
            f"[download-{WINDOWS_ZIP_PROOF_ID}]: {zip_url}"
        )
        assert "[![Windows](https://img.shields.io/badge/Windows-x64-blue.svg)][download-windows-setup-x64]" in cut_readme
        assert "[**Download for Windows (.exe)**][download-windows-setup-x64] or the [portable ZIP][download-windows-x64]" in cut_readme
        cut_qs = cut_readme.split("### Windows quick start", 1)[1].split("### Linux quick start", 1)[0]
        assert "Download for Windows (.exe)" in cut_qs and "portable ZIP" in cut_qs
        cut_site = (repo / "site/install/index.html").read_text(encoding="utf-8")
        cut_docs = (repo / "docs/install/index.html").read_text(encoding="utf-8")
        assert cut_site.split("<body", 1)[1] == cut_docs.split("<body", 1)[1]
        card = cut_site.split('<article id="windows">', 1)[1].split("</article>", 1)[0]
        assert f'data-release-download="{WINDOWS_SETUP_PROOF_ID}" class="btn btn-primary" href="{setup_url}"' in card
        assert f'data-release-download="{WINDOWS_ZIP_PROOF_ID}" class="btn btn-quiet" href="{zip_url}"' in card
        assert card.index(WINDOWS_SETUP_PROOF_ID) < card.index(WINDOWS_ZIP_PROOF_ID)
        assert "Run the signed per-user installer" in card

    def test_customized_button_captions_stay_outside_replaceable_span_and_survive_sync(self, tmp_path):
        win_span = next(s for s in carrier_spans_for("site/install/index.html") if s.carrier_id.endswith("windows-x64"))

        # ZIP-only (case-only AND reworded captions): stay outside canonical span, survive sync, pass checks.
        for slug, custom_zip_caption in (("case_only", "DOWNLOAD FOR WINDOWS (.zip)"), ("reworded", "Get Ouroboros ZIP for Windows")):
            zip_repo = _seeded_real_repo(tmp_path / f"zip_{slug}", version="7.12.1", published_version="7.12.1", windows_installer=False)
            (zip_repo / "build_windows.ps1").write_text("# zip build\n", encoding="utf-8", newline="\n")
            site_p = zip_repo / "site/install/index.html"
            canonical = site_p.read_text(encoding="utf-8")
            custom = canonical.replace(">Download for Windows (.zip)</a>", f">{custom_zip_caption}</a>", 1)
            assert locate_carrier_span(custom, win_span) == ("malformed_anchor", None)
            assert carrier_only_change(canonical, custom, "site/install/index.html") is False
            assert substitute_carrier_spans(custom, carrier_spans_for("site/install/index.html"), canonical) == (None, "malformed_anchor:site_install_download_windows-x64")
            site_p.write_text(custom, encoding="utf-8", newline="\n")
            assert sync_release_metadata(str(zip_repo)) == []
            assert f">{custom_zip_caption}</a>" in site_p.read_text(encoding="utf-8")
            assert check_worktree_version_sync(zip_repo) == ""

        # Cross-mode bump on Setup build preserves custom ZIP caption (including case-only) and flags missing Setup.
        (zip_repo / "scripts").mkdir(exist_ok=True)
        (zip_repo / "scripts" / "pack_windows_setup.ps1").write_text("# setup\n", encoding="utf-8", newline="\n")
        (zip_repo / "VERSION").write_text("7.12.2\n", encoding="utf-8", newline="\n")
        sync_release_metadata(str(zip_repo))
        assert f">{custom_zip_caption}</a>" in site_p.read_text(encoding="utf-8")
        assert release_asset_download_url(WINDOWS_ZIP_PROOF_ID, "7.12.2") in site_p.read_text(encoding="utf-8")
        assert f"site/install/index.html download {WINDOWS_SETUP_PROOF_ID}" in check_worktree_version_sync(zip_repo)

        # Setup+ZIP cut: custom HTML captions (including case-only) and custom README bullet survive version bumps.
        cut_repo = _seeded_real_repo(tmp_path / "cut_custom", version="7.12.2", published_version="7.12.2", windows_installer=True)
        (cut_repo / "build_windows.ps1").write_text("# zip\n", encoding="utf-8", newline="\n")
        (cut_repo / "scripts").mkdir(exist_ok=True)
        (cut_repo / "scripts" / "pack_windows_setup.ps1").write_text("# setup\n", encoding="utf-8", newline="\n")
        cut_site_p, cut_readme_p = cut_repo / "site/install/index.html", cut_repo / "README.md"
        canon_site, canon_readme = cut_site_p.read_text(encoding="utf-8"), cut_readme_p.read_text(encoding="utf-8")
        custom_site = canon_site.replace(">Download for Windows (.exe)</a>", ">DOWNLOAD FOR WINDOWS (.exe)</a>", 1).replace(
            ">Portable ZIP (.zip)</a>", ">PORTABLE ZIP (.zip)</a>", 1,
        )
        custom_readme = canon_readme.replace(
            "[**Download for Windows (.exe)**][download-windows-setup-x64] or the [portable ZIP][download-windows-x64]",
            "[**Custom Windows Setup**][download-windows-setup-x64] / [Custom ZIP][download-windows-x64]", 1,
        )
        assert locate_carrier_span(custom_site, win_span) == ("malformed_anchor", None)
        assert carrier_only_change(canon_site, custom_site, "site/install/index.html") is False
        assert carrier_only_change(canon_readme, custom_readme, "README.md") is False
        assert substitute_carrier_spans(custom_readme, carrier_spans_for("README.md"), canon_readme) == (None, "malformed_anchor:readme_windows_links")
        cut_site_p.write_text(custom_site, encoding="utf-8", newline="\n")
        cut_readme_p.write_text(custom_readme, encoding="utf-8", newline="\n")
        (cut_repo / "VERSION").write_text("7.12.3\n", encoding="utf-8", newline="\n")
        sync_release_metadata(str(cut_repo))
        bumped_site, bumped_readme = cut_site_p.read_text(encoding="utf-8"), cut_readme_p.read_text(encoding="utf-8")
        assert ">DOWNLOAD FOR WINDOWS (.exe)</a>" in bumped_site and ">PORTABLE ZIP (.zip)</a>" in bumped_site
        assert "[**Custom Windows Setup**][download-windows-setup-x64] / [Custom ZIP][download-windows-x64]" in bumped_readme
        assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.3") in bumped_site
        assert release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.3") in bumped_readme
        assert check_worktree_version_sync(cut_repo) == ""

    def test_outside_span_prose_and_images_survive_sync_and_span_substitution(self, tmp_path):
        repo = _seeded_real_repo(
            tmp_path, version="7.12.2-alpha.0", published_version="7.12.1", windows_installer=False
        )
        card_add = (
            '        <p class="local-note">UNIQUE_WINDOWS_CARD_NOTE</p>\n'
            '        <figure><img src="/assets/local-win.png" alt="UNIQUE_WINDOWS_FIG"></figure>\n'
        )
        bullet_suffix = " <!-- UNIQUE_BULLET_SUFFIX -->"
        site_path, docs_path, readme_path = repo / "site/install/index.html", repo / "docs/install/index.html", repo / "README.md"
        base_site, base_readme = site_path.read_text(encoding="utf-8"), readme_path.read_text(encoding="utf-8")
        edited_site = base_site.replace('      </article>\n      <article id="linux">', f'{card_add}      </article>\n      <article id="linux">', 1)
        edited_readme = base_readme.replace("### Linux quick start", "\nUNIQUE_WINDOWS_QUICKSTART_NOTE\n\n### Linux quick start", 1).replace(
            "- **Windows x64:** [**Download for Windows (.zip)**][download-windows-x64]",
            f"- **Windows x64:** [**Download for Windows (.zip)**][download-windows-x64]{bullet_suffix}", 1,
        )
        assert carrier_only_change(base_site, edited_site, "site/install/index.html") is False
        assert carrier_only_change(base_readme, edited_readme, "README.md") is False
        site_path.write_text(edited_site, encoding="utf-8", newline="\n")
        docs_path.write_text(
            docs_path.read_text(encoding="utf-8").replace('      </article>\n      <article id="linux">', f'{card_add}      </article>\n      <article id="linux">', 1),
            encoding="utf-8", newline="\n",
        )
        readme_path.write_text(edited_readme, encoding="utf-8", newline="\n")

        assert sync_release_metadata(str(repo), published_version="7.12.1", windows_installer=False) == []
        (repo / "VERSION").write_text("7.12.2\n", encoding="utf-8", newline="\n")
        sync_release_metadata(str(repo))
        site_7122, readme_7122 = site_path.read_text(encoding="utf-8"), readme_path.read_text(encoding="utf-8")
        assert "UNIQUE_WINDOWS_CARD_NOTE" in site_7122 and "UNIQUE_WINDOWS_FIG" in site_7122
        assert "UNIQUE_WINDOWS_QUICKSTART_NOTE" in readme_7122 and bullet_suffix in readme_7122
        assert carrier_only_change(edited_site, site_7122, "site/install/index.html") is True
        assert carrier_only_change(edited_readme, readme_7122, "README.md") is True
        assert check_worktree_version_sync(repo) == ""

        clean_repo = _seeded_real_repo(tmp_path / "clean", version="7.12.2", published_version="7.12.2", windows_installer=True)
        clean_site = (clean_repo / "site/install/index.html").read_text(encoding="utf-8")
        clean_readme = (clean_repo / "README.md").read_text(encoding="utf-8")
        sub_site, reason = substitute_carrier_spans(edited_site, carrier_spans_for("site/install/index.html"), clean_site)
        assert reason == "" and sub_site is not None and "UNIQUE_WINDOWS_CARD_NOTE" in sub_site and "UNIQUE_WINDOWS_FIG" in sub_site
        assert version_carrier_desyncs("7.12.2", site_install_text=sub_site) == []
        sub_readme, reason = substitute_carrier_spans(edited_readme, carrier_spans_for("README.md"), clean_readme)
        assert reason == "" and sub_readme is not None and "UNIQUE_WINDOWS_QUICKSTART_NOTE" in sub_readme and bullet_suffix in sub_readme
        assert version_carrier_desyncs("7.12.2", readme_text=sub_readme, download_readme_text=sub_readme) == []

        # Reverse span substitution (Setup+ZIP -> ZIP-only) must leave ZERO dangling [download-windows-setup-x64]
        # in the badge, bullet, or quick start, and keep outside-span edits intact.
        sub_back_readme, reason = substitute_carrier_spans(readme_7122, carrier_spans_for("README.md"), base_readme)
        assert reason == "" and sub_back_readme is not None
        assert WINDOWS_SETUP_PROOF_ID not in sub_back_readme
        assert "UNIQUE_WINDOWS_QUICKSTART_NOTE" in sub_back_readme and bullet_suffix in sub_back_readme
        assert version_carrier_desyncs("7.12.2-alpha.0", readme_text=sub_back_readme, download_readme_text=sub_back_readme) == []
        sub_back_site, reason = substitute_carrier_spans(site_7122, carrier_spans_for("site/install/index.html"), base_site)
        assert reason == "" and sub_back_site is not None and WINDOWS_SETUP_PROOF_ID not in sub_back_site
        assert "Extract the ZIP to a permanent folder" in sub_back_site and "UNIQUE_WINDOWS_CARD_NOTE" in sub_back_site
        assert version_carrier_desyncs("7.12.2-alpha.0", site_install_text=sub_back_site) == []

        # Fork-customized Windows quick start degrades cleanly instead of silently overwriting fork prose.
        custom_qs = edited_readme.replace("2. Extract the ZIP to a permanent folder.", "2. Custom fork step.")
        degraded, degrade_reason = substitute_carrier_spans(custom_qs, carrier_spans_for("README.md"), clean_readme)
        assert degraded is None and degrade_reason == "malformed_anchor:readme_windows_quickstart"

        (repo / "VERSION").write_text("7.12.2-alpha.0\n", encoding="utf-8", newline="\n")
        sync_release_metadata(str(repo), published_version="7.12.1", windows_installer=False)
        assert site_path.read_text(encoding="utf-8") == edited_site
        assert readme_path.read_text(encoding="utf-8") == edited_readme

    def test_strict_missing_and_duplicate_html_windows_anchors(self, tmp_path):
        repo = _seeded_real_repo(tmp_path, version="7.12.2", published_version="7.12.2", windows_installer=True)
        site_7122 = (repo / "site/install/index.html").read_text(encoding="utf-8")
        readme_7122 = (repo / "README.md").read_text(encoding="utf-8")
        win_span = next(s for s in carrier_spans_for("site/install/index.html") if s.carrier_id.endswith("windows-x64"))
        setup_url = release_asset_download_url(WINDOWS_SETUP_PROOF_ID, "7.12.2")
        zip_url = release_asset_download_url(WINDOWS_ZIP_PROOF_ID, "7.12.2")

        no_windows = f'<a data-release-download="macos-arm64" href="{release_asset_download_url("macos-arm64", "7.12.1")}">mac</a>\n'
        assert locate_carrier_span(no_windows, win_span) == ("malformed_anchor", None)
        synced_missing = _sync_html_download_urls(no_windows, "7.12.2", windows_installer=True)
        assert WINDOWS_SETUP_PROOF_ID not in synced_missing and ">download</a>" not in synced_missing

        setup_only = f'<a data-release-download="{WINDOWS_SETUP_PROOF_ID}" class="btn btn-primary" href="{setup_url}">Download for Windows (.exe)</a>\n'
        assert locate_carrier_span(setup_only, win_span) == ("malformed_anchor", None)
        assert _sync_html_download_urls(setup_only, "7.12.2", windows_installer=True) == setup_only

        dup_zip = site_7122.replace("</div>\n        <p>Run the signed per-user installer", f'</div>\n        <a data-release-download="{WINDOWS_ZIP_PROOF_ID}" href="{zip_url}">dup</a>\n        <p>Run the signed per-user installer', 1)
        dup_setup = site_7122.replace("</div>\n        <p>Run the signed per-user installer", f'</div>\n        <a data-release-download="{WINDOWS_SETUP_PROOF_ID}" href="{setup_url}">dup</a>\n        <p>Run the signed per-user installer', 1)
        wedged = site_7122.replace(f'<a data-release-download="{WINDOWS_ZIP_PROOF_ID}"', f'<p>KEEP_WEDGED</p>\n          <a data-release-download="{WINDOWS_ZIP_PROOF_ID}"', 1)
        swapped = site_7122.replace(
            _render_html_windows_actions("7.12.2", "        ", "\n", windows_installer=True),
            '\n'.join((
                '<div class="download-actions">',
                f'          <a data-release-download="{WINDOWS_ZIP_PROOF_ID}" class="btn btn-quiet" href="{zip_url}">Portable ZIP (.zip)</a>',
                f'          <a data-release-download="{WINDOWS_SETUP_PROOF_ID}" class="btn btn-primary" href="{setup_url}">Download for Windows (.exe)</a>',
                '        </div>',
            )), 1,
        )
        for bad_html, expected_status in (
            (dup_zip, "duplicate_anchor"),
            (dup_setup, "duplicate_anchor"),
            (wedged, "malformed_anchor"),
            (swapped, "malformed_anchor"),
        ):
            assert locate_carrier_span(bad_html, win_span) == (expected_status, None)
            assert _sync_html_download_urls(bad_html, "7.12.2", windows_installer=True) == bad_html
            assert "site/install/index.html Windows download choice" in version_carrier_desyncs("7.12.2", site_install_text=bad_html)
        assert "KEEP_WEDGED" in _sync_html_download_urls(wedged, "7.12.3", windows_installer=True)

        swapped_style = site_7122.replace(f'class="btn btn-primary" href="{setup_url}"', f'class="btn btn-quiet" href="{setup_url}"')
        assert "site/install/index.html Windows download choice" in version_carrier_desyncs("7.12.2", site_install_text=swapped_style)
        # Mismatched help prose next to buttons or quick start must be flagged in both forms.
        stale_html_help = site_7122.replace(
            "<p>Run the signed per-user installer (<code>.exe</code>) and open Ouroboros from the Start menu, or extract the portable ZIP (<code>.zip</code>) to a permanent folder and run <code>Ouroboros.exe</code>.</p>",
            "<p>Extract the ZIP to a permanent folder, open the <code>Ouroboros</code> folder, and run <code>Ouroboros.exe</code>.</p>",
        )
        assert "site/install/index.html Windows download choice" in version_carrier_desyncs("7.12.2", site_install_text=stale_html_help)
        zip_site = _sync_html_download_urls(site_7122, "7.12.1", windows_installer=False)
        bad_zip_help = zip_site.replace(
            "<p>Extract the ZIP to a permanent folder, open the <code>Ouroboros</code> folder, and run <code>Ouroboros.exe</code>.</p>",
            "<p>Run the signed per-user installer (<code>.exe</code>) and open Ouroboros from the Start menu, or extract the portable ZIP (<code>.zip</code>) to a permanent folder and run <code>Ouroboros.exe</code>.</p>",
        )
        assert "site/install/index.html Windows download choice" in version_carrier_desyncs(
            "7.12.1", site_install_text=bad_zip_help, windows_installer=False,
        )
        stale_readme = readme_7122.replace(
            "[**Download for Windows (.exe)**][download-windows-setup-x64] or the [portable ZIP][download-windows-x64]",
            "[**Download for Windows (.zip)**][download-windows-x64]",
        )
        assert "README.md Windows download choice" in version_carrier_desyncs("7.12.2", download_readme_text=stale_readme)
        stale_qs = readme_7122.replace(
            "1. Click [**Download for Windows (.exe)**][download-windows-setup-x64] and run the signed per-user installer (no administrator rights required), then open Ouroboros from the Start menu.\n"
            "2. Or use the [portable ZIP][download-windows-x64]: extract it to a permanent folder, open the extracted `Ouroboros` folder, and run `Ouroboros.exe`.",
            "1. Click [**Download for Windows (.zip)**][download-windows-x64].\n"
            "2. Extract the ZIP to a permanent folder.\n"
            "3. Open the extracted `Ouroboros` folder and run `Ouroboros.exe`.",
        )
        assert "README.md Windows download choice" in version_carrier_desyncs("7.12.2", download_readme_text=stale_qs)
