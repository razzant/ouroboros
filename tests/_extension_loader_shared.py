"""Extension-skill builders and the loader-state fixture shared by the extension_loader suites.

Split out of ``tests/test_extension_loader.py`` when that module was divided by
theme; every definition is verbatim, so each sibling suite (and the pre-existing
importers ``test_extension_surfaces.py``, ``test_extension_isolated_deps.py``,
``test_extension_process_runner.py``, ``test_tool_catalog.py`` and
``test_tool_capabilities_readonly_subagent.py``, which keep reaching these
helpers through the parent module's re-export) keeps the exact skill payloads
and review state it was written against. ``_clear_loader_state`` is autouse, so
importing it into a test module re-applies it there — every sibling suite
imports it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from ouroboros.skill_loader import SkillReviewState, save_enabled, save_review_state

from tests._shared import clean_extension_runtime_state


@pytest.fixture(autouse=True)
def _clear_loader_state(monkeypatch):
    """Reset the module-level registries between tests."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def _write_ext_skill(
    repo_root: pathlib.Path,
    name: str,
    *,
    plugin_body: str,
    permissions: list[str],
    env_from_settings: list[str] | None = None,
    entry: str = "plugin.py",
    extra_frontmatter: str = "",
) -> pathlib.Path:
    skill_dir = repo_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    perms_yaml = json.dumps(permissions)
    env_yaml = json.dumps(env_from_settings or [])
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Phase 4 extension.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            f"entry: {entry}\n"
            f"permissions: {perms_yaml}\n"
            f"env_from_settings: {env_yaml}\n"
            f"{extra_frontmatter}"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    entry_path = skill_dir / entry
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(plugin_body, encoding="utf-8")
    return skill_dir


def _prepare_extension(
    tmp_path: pathlib.Path,
    name: str,
    plugin_body: str,
    permissions: list[str],
    env_from_settings: list[str] | None = None,
    extra_frontmatter: str = "",
):
    """Write + enable + PASS-review an extension so the loader accepts it."""
    from ouroboros.skill_loader import find_skill
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir(exist_ok=True)
    _write_ext_skill(
        repo_root,
        name,
        plugin_body=plugin_body,
        permissions=permissions,
        env_from_settings=env_from_settings,
        extra_frontmatter=extra_frontmatter,
    )
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, name, True)
    save_review_state(
        drive_root,
        name,
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    # Refetch with fresh state on the loaded struct.
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    return loaded, repo_root, drive_root


def _mark_isolated_deps_installed(drive_root: pathlib.Path, loaded) -> None:
    from ouroboros.marketplace.install_specs import install_specs_hash
    from ouroboros.marketplace.isolated_deps import FINGERPRINT_FILENAME, isolated_env_dir
    from ouroboros.skill_dependencies import auto_install_specs_for_skill
    from ouroboros.skill_loader import skill_state_dir

    auto_specs = auto_install_specs_for_skill(drive_root, loaded)
    assert auto_specs
    payload = {
        "status": "installed",
        "specs_hash": install_specs_hash(auto_specs),
        "installed": auto_specs,
    }
    state_dir = skill_state_dir(drive_root, loaded.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "deps.json").write_text(json.dumps(payload), encoding="utf-8")
    env_dir = isolated_env_dir(loaded.skill_dir)
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / FINGERPRINT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _isolated_site_packages_dir(loaded) -> pathlib.Path:
    return (
        loaded.skill_dir
        / ".ouroboros_env"
        / "python"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _add_fake_native_dep(loaded, package_name: str = "dummy_pkg") -> pathlib.Path:
    site_dir = _isolated_site_packages_dir(loaded)
    pkg_dir = site_dir / package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("VALUE = 'isolated-native-risk'\n", encoding="utf-8")
    (site_dir / "fake_native.so").write_bytes(b"not a real shared object; scan marker only")
    return site_dir
