import pathlib
import re

import pytest

from ouroboros.runtime_mode_policy import SAFETY_CRITICAL_PATHS
from ouroboros.tools.release_sync import (
    RELEASE_ASSET_TEMPLATES,
    _normalize_pep440,
    check_history_limit,
    release_asset_download_url,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_version_file_and_pyproject_are_synced():
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    package_json = (REPO / "web" / "package.json").read_text(encoding="utf-8")
    uv_lock = (REPO / "uv.lock").read_text(encoding="utf-8")

    # ``VERSION`` holds the author-facing spelling (``4.50.0-rc.1`` /
    # ``4.50.0``); ``pyproject.toml`` must carry the PEP 440-canonical
    # form (``4.50.0rc1`` / ``4.50.0``) so pip / build / twine accept
    # the project metadata. For stable versions the two forms are
    # identical; for pre-releases ``_normalize_pep440`` collapses the
    # separators.
    pyproject_version = _normalize_pep440(version)
    assert f'version = "{pyproject_version}"' in pyproject
    assert f'"version": "{version}"' in package_json
    assert f'name = "ouroboros"\nversion = "{pyproject_version}"' in uv_lock


def test_push_to_remote_push_tags_compatibility(monkeypatch):
    from supervisor import git_ops

    commands = []
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name: True)
    monkeypatch.setattr(
        git_ops,
        "git_capture",
        lambda command, **_kwargs: commands.append(list(command)) or (0, "", ""),
    )

    ok, _ = git_ops.push_to_remote("feature", push_tags=False)
    assert ok is True
    assert commands == [["git", "push", "-u", "origin", "feature"]]

    commands.clear()
    ok, _ = git_ops.push_to_remote("feature", push_tags=True)
    assert ok is True
    assert commands == [
        ["git", "push", "-u", "origin", "feature"],
        ["git", "push", "origin", "--tags"],
    ]


def test_readme_version_history_contains_current_version_row():
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    assert f"| {version} |" in readme


def test_release_guidance_accepts_author_facing_and_pep440_forms():
    bible = (REPO / "BIBLE.md").read_text(encoding="utf-8")
    system = (REPO / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")

    assert "PEP 440 canonical form" in bible
    assert "PEP 440 canonical form" in system
    assert "`VERSION` == `pyproject.toml` version == latest git tag" not in bible
    assert "VERSION == pyproject.toml version == latest git tag" not in system


def test_architecture_docs_describe_bundle_bootstrap_not_per_launch_core_sync():
    architecture = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "scripts/build_repo_bundle.py" in architecture
    assert "repo.bundle" in architecture
    assert "launcher.py" in architecture
    assert "repo_bundle_manifest.json" in architecture
    assert "overwritten from bundle on every launch" not in architecture
    assert "copies workspace to `~/Ouroboros/repo/` on first run" not in architecture


def test_readme_version_history_stays_within_minor_row_limit():
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    warnings = check_history_limit(readme)
    assert not [w for w in warnings if "minor rows" in w]


def test_readme_version_history_rows_render_as_three_cells():
    """GFM splits a table row on EVERY unescaped `|` — including one inside a code
    span. A row that inlines a pipe-separated vocabulary (`a | b`, `PASS|FAIL`)
    unescaped is parsed as 5+ cells against a 3-column table, and GitHub silently
    DROPS everything after the third cell: the v6.78.0 row lost its whole disclosed-
    narrowing tail that way. Every Version History row must carry exactly the 4
    unescaped pipes of a 3-cell row."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    rows = [
        line for line in readme.splitlines()
        if re.match(r"^\| \d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)? \|", line)
    ]
    assert len(rows) >= 2, "Version History rows not found"
    offenders = {
        line.split("|")[1].strip(): len(re.findall(r"(?<!\\)\|", line))
        for line in rows
        if len(re.findall(r"(?<!\\)\|", line)) != 4
    }
    assert not offenders, f"escape the inner pipes in these rows: {offenders}"


def test_readme_documents_release_tag_prerequisite_for_build_scripts():
    """The platform build scripts now hard-fail if HEAD is not tagged with
    ``v$(cat VERSION)`` (see tests/test_build_scripts.py). The README must
    document that prerequisite alongside the macOS/Linux/Windows build
    sections so users are not surprised by the new failure mode."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    assert "Release tag prerequisite" in readme
    assert "git tag -a" in readme


def test_readme_documents_packaged_cli_installer_and_locked_source_env():
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    assert "Install CLI.command" in readme
    assert "./Ouroboros/bin/install-ouroboros-cli" in readme
    assert r"Ouroboros\bin\install-ouroboros-cli.cmd" in readme
    assert "uv sync --locked --extra browser --group dev" in readme
    assert "https://astral.sh/uv/0.12.1/install.sh" in readme
    assert "https://astral.sh/uv/0.12.1/install.ps1" in readme
    assert "ouroboros run --start \"2+2?\"" in readme


def test_legacy_requirements_file_is_only_a_runtime_lock_pointer():
    lines = [
        line.strip()
        for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines == ["-r requirements-runtime.lock"]


def test_readme_prioritizes_macos_dmg_install_and_model_access():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()

    download_pos = readme.index("## Download Ouroboros")
    claudexor_pos = readme.index("Ouroboros bundles [Claudexor]")
    paper_pos = readme.index("The technical report")
    benchmark_chart_pos = readme.index('src="assets/bench-terminal-bench.svg"')
    origin_pos = readme.index("Ouroboros first booted")
    evolution_image_pos = readme.index('src="assets/evolution.png"')
    advanced_pos = readme.index("## Advanced installation")
    assert download_pos < claudexor_pos < paper_pos < benchmark_chart_pos
    assert download_pos < origin_pos < evolution_image_pos < advanced_pos
    assert "You do not need to clone this repository or install Python or uv" in readme
    assert "Ouroboros-<version>.dmg" in readme
    assert "drag `Ouroboros.app` onto the **Applications** shortcut" in readme
    assert 'src="assets/install-macos.png"' in readme
    assert "at least one supported remote provider API key or a local GGUF model" in readme
    assert "https://ouroboros-agent.ai/install/#linux" in readme
    assert (
        "badges%2Fdownloads.json)](https://ouroboros-agent.ai/install/)"
        in readme
    )
    assert "/releases/latest/download/" not in readme
    for proof_id in RELEASE_ASSET_TEMPLATES:
        assert (
            f"[download-{proof_id}]: {release_asset_download_url(proof_id, version)}"
            in readme
        )


def test_install_page_matches_macos_quick_start_and_model_prerequisite():
    install_page = (REPO / "site" / "install" / "index.html").read_text(encoding="utf-8")
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()

    assert 'id="macos-quick-start"' in install_page
    macos_section = install_page.split('id="macos-quick-start"', 1)[1].split(
        "</section>", 1
    )[0]
    assert macos_section.count("<li>") == 3
    assert "Applications</strong> shortcut" in install_page
    assert 'src="/assets/install-macos.png?v=' in install_page
    assert "at least one supported remote provider API key or a local GGUF model" in install_page
    assert "Normal installation does not require cloning the repository, Python, or uv" in install_page
    assert "verification evidence, not additional installers" in install_page
    assert install_page.index("platform-downloads") < install_page.index(
        "Advanced: headless CLI with uv"
    ) < install_page.index("Develop or run from source")
    for proof_id in RELEASE_ASSET_TEMPLATES:
        assert f'data-release-download="{proof_id}"' in install_page
        assert release_asset_download_url(proof_id, version) in install_page


def test_architecture_doc_describes_build_script_release_tag_check():
    architecture = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "Release tag prerequisite" in architecture
    assert "scripts/build_repo_bundle.py" in architecture
    assert "release-tag SSOT" in architecture
    assert "annotated `v$(cat VERSION)` tag points at `HEAD`" in architecture


def test_system_prompt_lists_bible_in_safety_critical_set():
    """Both LLM-facing safety inventories must match the runtime SSOT."""
    system_md = (REPO / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    safety_md = (REPO / "prompts" / "SAFETY.md").read_text(encoding="utf-8")

    safety_section_start = system_md.find("## Immutable Safety Files")
    assert safety_section_start != -1
    safety_section_end = system_md.find("##", safety_section_start + 1)
    safety_section = system_md[safety_section_start:safety_section_end]
    system_paths = {
        match.group(1)
        for line in safety_section.splitlines()
        if (match := re.match(r"^- `([^`]+)`", line))
    }
    assert system_paths == SAFETY_CRITICAL_PATHS

    safety_inventory = re.search(
        r"safety-critical files \((.*?)\), frozen contracts",
        safety_md,
    )
    assert safety_inventory is not None
    safety_paths = set(re.findall(r"`([^`]+)`", safety_inventory.group(1)))
    assert safety_paths == SAFETY_CRITICAL_PATHS


def test_architecture_doc_does_not_claim_ensure_managed_repo_fetches():
    """ensure_managed_repo only validates + ensures the managed remote is
    configured; the actual fetch lives in supervisor.git_ops.checkout_and_reset.
    The ARCHITECTURE.md startup flow must not conflate the two."""
    architecture = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "ensure_managed_repo()" in architecture
    assert "supervisor/git_ops.checkout_and_reset" in architecture


def test_checklists_describe_pep440_spelling_rule():
    checklists = (REPO / "docs" / "CHECKLISTS.md").read_text(encoding="utf-8")

    assert "PEP 440 canonical form" in checklists
    assert "_normalize_pep440" in checklists


def test_server_workers_init_reads_manifest_branches_not_hardcoded_strings():
    """server.py::_run_supervisor must feed ``workers.init`` the
    manifest-driven branch names from ``_runtime_branch_defaults()`` —
    not literal ``"ouroboros"`` / ``"ouroboros-stable"`` strings. A
    packaged bundle built with non-default
    ``--managed-local-branch`` / ``--managed-local-stable-branch`` would
    otherwise bootstrap one branch set and run workers against the old
    hardcoded names."""
    server_py = (REPO / "server.py").read_text(encoding="utf-8")

    assert "workers_init(" in server_py
    assert 'branch_dev="ouroboros", branch_stable="ouroboros-stable"' not in server_py
    assert 'branch_dev=_workers_branch_dev' in server_py
    assert 'branch_stable=_workers_branch_stable' in server_py


def test_architecture_module_tree_lists_all_live_extension_http_endpoints():
    """The high-level module map entry for ``ouroboros/gateway/extensions.py``
    must list every HTTP path the module actually registers, so the
    architecture map does not contradict the endpoint table later in the
    same document. Specifically the Phase 5 review surface
    ``POST /api/skills/<skill>/review`` is exported via
    ``server.py`` and must appear in both places."""
    architecture = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    # Module map entry lives on the ``gateway/extensions.py`` tree line.
    tree_idx = architecture.find("├── extensions.py")
    assert tree_idx != -1
    tree_line = architecture[tree_idx : architecture.find("\n", tree_idx)]
    assert "POST /api/skills/<skill>/toggle" in tree_line
    assert "POST /api/skills/<skill>/delete" in tree_line
    assert "POST /api/skills/<skill>/review" in tree_line


def test_architecture_doc_lists_valid_extension_route_methods_in_frozen_contracts():
    """Phase 4 ``PluginAPI`` exposes ``VALID_EXTENSION_ROUTE_METHODS`` as
    part of the frozen contract (see ``ouroboros/contracts/plugin_api.py``
    ``__all__`` + ``tests/test_contracts.py``). The ARCHITECTURE §11.1
    frozen-contract table must list it alongside the other Phase 4
    plugin_api exports so the doc/code mirror is accurate."""
    architecture = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "VALID_EXTENSION_ROUTE_METHODS" in architecture
    assert "test_extension_route_methods_contract_matches_server_dispatch" in architecture


def test_architecture_doc_describes_extension_staging_surface():
    """Phase 4's ``_stage_extension_import_tree`` creates a new durable
    runtime subdirectory under ``data/state/skills/<name>/``. The
    architecture doc's skills data-layout section must describe it so the
    doc/code mirror is accurate."""
    architecture = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "__extension_imports/" in architecture
    assert "_stage_extension_import_tree" in architecture


def test_pyproject_includes_provider_svgs():
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")

    assert '"providers/*.svg"' in pyproject


@pytest.mark.skipif(not (REPO / "Dockerfile").exists(), reason="Dockerfile not present in repo (bundle-only)")
def test_dockerfile_sets_default_file_browser_root():
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")

    assert "OUROBOROS_FILE_BROWSER_DEFAULT=${APP_HOME}" in dockerfile
