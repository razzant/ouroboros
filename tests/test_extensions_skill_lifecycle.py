"""Enabling, disabling and deleting an extension skill over the HTTP surface.

Split verbatim out of ``tests/test_extensions_api.py`` by theme. This module owns the
toggle that enables and loads an extension, the peer conflict it refuses to resolve on
its own, the review verdicts it accepts and the missing isolated deps it blocks, the
collision disable that must not write shared state, and the delete that removes an
external payload's state and unloads it — with the symlink, collision and unsanitized
leaf cases around it.
"""

from __future__ import annotations

import json

import pytest



from tests._extensions_api_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extensions,
    _make_client,
    _stop_patches,
    _write_ext,
)


@pytest.fixture
def client_env(tmp_path, monkeypatch):
    """Yield ``(client, drive_root)`` and stop lifecycle patches at teardown."""
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        yield client, drive_root
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_enables_and_loads_extension(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, save_review_state
    from ouroboros.skill_loader import compute_content_hash

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_toggle", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    broadcasts = []
    client.app.app.state.broadcast_ws_sync = lambda payload: broadcasts.append(payload)  # type: ignore[attr-defined]
    try:
        # Pre-mark review PASS so enable actually loads.
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_toggle",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        resp = client.post(
            "/api/skills/ext_toggle/toggle",
            json={"enabled": True},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["enabled"] is True
        assert data["extension_action"] == "extension_loaded"
        assert broadcasts[-1]["type"] == "extension_lifecycle"
        assert broadcasts[-1]["skill"] == "ext_toggle"
        assert broadcasts[-1]["action"] == "extension_loaded"
        assert "ext_toggle" in extension_loader.snapshot()["extensions"]

        # Disable → unload.
        resp = client.post(
            "/api/skills/ext_toggle/toggle",
            json={"enabled": False},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["enabled"] is False
        assert data["extension_action"] == "extension_unloaded"
        assert broadcasts[-1]["action"] == "extension_unloaded"
        assert "ext_toggle" not in extension_loader.snapshot()["extensions"]
    finally:
        _stop_patches(patches)


def test_api_projects_conflict_and_refuses_enable_until_peer_is_disabled(
    tmp_path, monkeypatch
):
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    pass\n"
    telegram_dir = _write_ext(
        skills_root,
        "telegram",
        permissions=[],
        plugin=plugin,
        conflicts=["telegram-bridge"],
    )
    _write_ext(
        skills_root,
        "telegram-bridge",
        permissions=[],
        plugin=plugin,
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        save_review_state(
            drive_root,
            "telegram",
            SkillReviewState(
                status="pass",
                content_hash=compute_content_hash(telegram_dir, manifest_entry="plugin.py"),
            ),
        )
        save_enabled(drive_root, "telegram-bridge", True)

        index = client.get("/api/extensions")
        assert index.status_code == 200, index.text
        row = next(item for item in index.json()["skills"] if item["name"] == "telegram")
        assert row["conflicts"] == ["telegram-bridge"]
        assert row["conflict"] == {
            "code": "skill_conflict",
            "skills": ["telegram-bridge"],
            "omitted": 0,
        }

        blocked = client.post("/api/skills/telegram/toggle", json={"enabled": True})
        assert blocked.status_code == 409, blocked.text
        assert blocked.json()["conflict"] == row["conflict"]

        disabled = client.post(
            "/api/skills/telegram-bridge/toggle",
            json={"enabled": False},
        )
        assert disabled.status_code == 200, disabled.text
        enabled = client.post("/api/skills/telegram/toggle", json={"enabled": True})
        assert enabled.status_code == 200, enabled.text
        assert enabled.json()["enabled"] is True
    finally:
        _stop_patches(patches)


def test_api_skill_delete_removes_external_payload_state_and_unloads(client_env):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    client, drive_root = client_env
    skill_dir = _write_ext(
        drive_root / "skills" / "external",
        "local_delete",
        permissions=["tool"],
        plugin="def register(api):\n    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(drive_root, "local_delete", SkillReviewState(status="pass", content_hash=content_hash))

    enabled = client.post("/api/skills/local_delete/toggle", json={"enabled": True})
    assert enabled.status_code == 200, enabled.text
    assert "local_delete" in extension_loader.snapshot()["extensions"]
    assert (drive_root / "state" / "skills" / "local_delete").is_dir()

    resp = client.post("/api/skills/local_delete/delete", json={"payload_root": "skills/external/local_delete"})

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] is True
    assert data["deleted_payload_root"] == "skills/external/local_delete"
    assert not skill_dir.exists()
    assert not (drive_root / "state" / "skills" / "local_delete").exists()
    assert "local_delete" not in extension_loader.snapshot()["extensions"]

    hub_skill_dir = _write_ext(
        drive_root / "skills" / "clawhub",
        "hub_delete",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    (hub_skill_dir / ".clawhub.json").write_text("{}", encoding="utf-8")

    resp = client.post("/api/skills/hub_delete/delete", json={"payload_root": "skills/clawhub/hub_delete"})

    assert resp.status_code == 403
    assert hub_skill_dir.exists()


def test_api_skill_delete_rejects_external_symlink_bucket(client_env, tmp_path):
    client, drive_root = client_env
    external_target = tmp_path / "outside-external"
    _write_ext(
        external_target,
        "symlink_delete",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    skills_root = drive_root / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    try:
        (skills_root / "external").symlink_to(external_target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable in this environment: {exc}")

    resp = client.post(
        "/api/skills/symlink_delete/delete",
        json={"payload_root": "skills/external/symlink_delete"},
    )

    assert resp.status_code == 403
    assert (external_target / "symlink_delete").exists()


def test_api_skill_delete_rejects_name_collision_before_state_delete(client_env):
    client, drive_root = client_env
    external_dir = _write_ext(
        drive_root / "skills" / "external",
        "collide_delete",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    native_dir = _write_ext(
        drive_root / "skills" / "native",
        "collide_delete",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    state_dir = drive_root / "state" / "skills" / "collide_delete"
    state_dir.mkdir(parents=True)
    (state_dir / "enabled.json").write_text('{"enabled": true}', encoding="utf-8")

    resp = client.post(
        "/api/skills/collide_delete/delete",
        json={"payload_root": "skills/external/collide_delete"},
    )

    assert resp.status_code == 409
    assert external_dir.exists()
    assert native_dir.exists()
    assert state_dir.exists()


def test_api_skill_delete_accepts_unsanitized_external_directory_leaf(client_env):
    client, drive_root = client_env
    skill_dir = _write_ext(
        drive_root / "skills" / "external",
        "hello world",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    state_dir = drive_root / "state" / "skills" / "hello_world"
    state_dir.mkdir(parents=True)

    resp = client.post(
        "/api/skills/hello_world/delete",
        json={"payload_root": "skills/external/hello world"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted_payload_root"] == "skills/external/hello world"
    assert not skill_dir.exists()
    assert not state_dir.exists()


def test_api_skill_toggle_allows_warnings_review(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, save_review_state
    from ouroboros.skill_loader import compute_content_hash

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_advisory", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_advisory",
            SkillReviewState(status="warnings", content_hash=content_hash),
        )
        resp = client.post("/api/skills/ext_advisory/toggle", json={"enabled": True})

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["enabled"] is True
        assert data["review_status"] == "warnings"
        assert data["extension_action"] == "extension_loaded"
        assert "ext_advisory" in extension_loader.snapshot()["extensions"]
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_allows_warnings_under_blocking(tmp_path, monkeypatch):
    from ouroboros.skill_loader import SkillReviewState, save_review_state, compute_content_hash

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_blocked", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_blocked",
            SkillReviewState(status="warnings", content_hash=content_hash),
        )
        resp = client.post("/api/skills/ext_blocked/toggle", json={"enabled": True})

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["executable_review"] is True
        assert data["review_gate"]["blocking_reason"] == "warnings_do_not_block_execution"
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_blocks_missing_isolated_deps_env(tmp_path, monkeypatch):
    from ouroboros.marketplace.install_specs import install_specs_hash
    from ouroboros.marketplace.isolated_deps import DEPS_STATE_FILENAME
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_review_state,
        skill_state_dir,
    )

    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    skill_dir = _write_ext(skills_root, "ext_deps", permissions=["tool"], plugin=plugin)
    manifest = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        manifest.replace(
            "permissions: [\"tool\"]\n",
            "permissions: [\"tool\"]\n"
            "install_specs:\n"
            "  - kind: pip\n"
            "    package: wheel\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_deps",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        state_dir = skill_state_dir(drive_root, "ext_deps")
        state_dir.mkdir(parents=True, exist_ok=True)
        specs = [{"kind": "pip", "package": "wheel"}]
        (state_dir / DEPS_STATE_FILENAME).write_text(
            json.dumps({"status": "installed", "specs_hash": install_specs_hash(specs)}),
            encoding="utf-8",
        )

        resp = client.post("/api/skills/ext_deps/toggle", json={"enabled": True})

        assert resp.status_code == 409, resp.text
        data = resp.json()
        assert data["deps_status"] == "missing"
        assert not (state_dir / "enabled.json").exists()
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_collision_disable_does_not_write_shared_state(
    tmp_path, monkeypatch
):
    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    return None\n"
    _write_ext(skills_root, "hello world", permissions=[], plugin=plugin)
    _write_ext(skills_root, "hello_world", permissions=[], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.post("/api/skills/hello_world/toggle", json={"enabled": False})
        assert resp.status_code == 400, resp.text
        data = resp.json()
        assert data["extension_reason"] == "name_collision"
        state_file = drive_root / "state" / "skills" / "hello_world" / "enabled.json"
        assert not state_file.exists()
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_rejects_non_boolean_enabled(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    pass\n"
    _write_ext(skills_root, "ext_toggle_bad", permissions=[], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, _, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.post("/api/skills/ext_toggle_bad/toggle", json={"enabled": "definitely"})
        assert resp.status_code == 400
        assert "boolean" in resp.text
    finally:
        _stop_patches(patches)
