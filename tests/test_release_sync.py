"""Tests for ouroboros/tools/release_sync.py (standalone library, no wire-up)."""

from pathlib import Path

import pytest

from ouroboros.tools.release_sync import (
    RELEASE_ASSET_TEMPLATES,
    check_history_limit,
    detect_numeric_claims,
    normalize_linux_package_version,
    release_asset_download_url,
    run_release_preflight,
    sync_release_metadata,
    version_carrier_desyncs,
)
from ouroboros.tools.release_sync import (  # noqa: E402 — private helpers under test
    _normalize_pep440,
    _shields_escape,
    _VERSION_RE,
    _VERSION_ROW_RE,
    _README_BADGE_RE,
    _ARCH_HEADER_RE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_references(version: str) -> str:
    return "".join(
        f"[download-{proof_id}]: {release_asset_download_url(proof_id, version)}\n"
        for proof_id in RELEASE_ASSET_TEMPLATES
    )


def _download_page(version: str) -> str:
    return "".join(
        f'<a data-release-download="{proof_id}" '
        f'href="{release_asset_download_url(proof_id, version)}">download</a>\n'
        for proof_id in RELEASE_ASSET_TEMPLATES
    )


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
        assert len(desync) == len(RELEASE_ASSET_TEMPLATES) - 1

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
