import os

import pytest

from ouroboros.contracts.skill_payload_policy import (
    SKILL_PAYLOAD_CONTROL_FILENAMES,
    is_skill_control_plane_path,
    is_skill_owner_state_alias,
    is_skill_owner_state_target,
    resolve_constrained_payload_path,
)
from ouroboros.contracts.task_constraint import TaskConstraint, resolve_payload_path
from tests._typed_guard_shared import _shell_guard_text




def test_resolve_payload_path_legacy_wrapper_matches_policy(tmp_path):
    drive_root = tmp_path / "data"
    skill_root = drive_root / "skills" / "external" / "alpha"
    skill_root.mkdir(parents=True)
    constraint = TaskConstraint(
        mode="skill_repair",
        skill_name="alpha",
        payload_root="skills/external/alpha",
    )

    for path_text in (
        "plugin.py",
        "skills/external/alpha/plugin.py",
        "data/skills/external/alpha/plugin.py",
    ):
        assert resolve_payload_path(drive_root, constraint, path_text) == resolve_constrained_payload_path(
            drive_root,
            constraint,
            path_text,
        )


def test_owner_state_policy_matches_legacy_wrappers(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.gateway import files as gateway_files
    from ouroboros.tools import core

    data_root = tmp_path / "data"
    monkeypatch.setattr(cfg, "DATA_DIR", data_root, raising=True)

    for filename in ("deps.json", "review_job.json"):
        target = data_root / "State" / "Skills" / "weather" / filename
        assert is_skill_owner_state_target(target, data_root) is True
        assert core._is_skill_owner_state_target(target, data_root) is True
        assert gateway_files._is_skill_owner_state_target(target) is True
        assert gateway_files._is_owner_only_file(target) is True


def test_publication_receipt_is_owner_state_across_wrappers(tmp_path, monkeypatch):
    """The OuroborosHub publication receipt is owner state: filename, stem,
    case-variant paths, and every legacy wrapper must all agree."""
    from ouroboros import config as cfg
    from ouroboros.contracts.skill_payload_policy import (
        SKILL_OWNER_STATE_FILENAMES,
        SKILL_OWNER_STATE_STEMS,
    )
    from ouroboros.gateway import files as gateway_files
    from ouroboros.tools import core

    assert "ouroboroshub.json" in SKILL_OWNER_STATE_FILENAMES
    assert "ouroboroshub" in SKILL_OWNER_STATE_STEMS

    data_root = tmp_path / "data"
    monkeypatch.setattr(cfg, "DATA_DIR", data_root, raising=True)

    for parts in (("state", "skills"), ("State", "Skills")):
        target = data_root.joinpath(*parts) / "weather" / "ouroboroshub.json"
        assert is_skill_owner_state_target(target, data_root) is True
        assert core._is_skill_owner_state_target(target, data_root) is True
        assert gateway_files._is_skill_owner_state_target(target) is True
        assert gateway_files._is_owner_only_file(target) is True


def test_publication_receipt_hardlink_alias_is_owner_state(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.gateway import files as gateway_files

    data_root = tmp_path / "data"
    state_dir = data_root / "state" / "skills" / "weather"
    state_dir.mkdir(parents=True)
    record = state_dir / "ouroboroshub.json"
    record.write_text("{}", encoding="utf-8")
    alias = data_root / "memory" / "receipt-copy.json"
    alias.parent.mkdir(parents=True)
    try:
        os.link(record, alias)
    except OSError:
        pytest.skip("Hardlinks unavailable on this filesystem")
    monkeypatch.setattr(cfg, "DATA_DIR", data_root, raising=True)

    assert is_skill_owner_state_alias(alias, data_root) is True
    assert gateway_files._is_owner_only_file(alias) is True


def test_publication_receipt_stem_reaches_shell_guard(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    blocked = _shell_guard_text(reg,
        {"cmd": "rm data/state/skills/weather/ouroboroshub.json"},
        "advanced",
    )
    assert blocked is not None
    assert "SKILL_STATE_WRITE_BLOCKED" in blocked


def test_owner_state_alias_policy_matches_files_api(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.gateway import files as gateway_files

    data_root = tmp_path / "data"
    state_dir = data_root / "state" / "skills" / "weather"
    state_dir.mkdir(parents=True)
    review = state_dir / "review.json"
    review.write_text("{}", encoding="utf-8")
    alias = data_root / "memory" / "review-copy.json"
    alias.parent.mkdir(parents=True)
    try:
        os.link(review, alias)
    except OSError:
        pytest.skip("Hardlinks unavailable on this filesystem")
    monkeypatch.setattr(cfg, "DATA_DIR", data_root, raising=True)

    assert is_skill_owner_state_alias(alias, data_root) is True
    assert gateway_files._is_owner_only_file(alias) is True


def test_control_plane_policy_matches_core_wrapper(tmp_path):
    from ouroboros.tools.core import is_skill_control_plane_path as core_control_plane_path

    data_root = tmp_path / "data"
    payload = data_root / "skills" / "external" / "alpha"
    payload.mkdir(parents=True)
    sidecar = payload / ".self_authored.json"
    user_file = payload / "plugin.py"

    assert is_skill_control_plane_path(sidecar, data_root) is True
    assert core_control_plane_path(sidecar, data_root) is True
    assert is_skill_control_plane_path(payload / "node_modules" / "pkg.json", data_root) is True
    assert is_skill_control_plane_path(user_file, data_root) is False
    assert core_control_plane_path(user_file, data_root) is False


def test_control_plane_policy_catches_hardlink_alias_outside_payload(tmp_path):
    data_root = tmp_path / "data"
    payload = data_root / "skills" / "external" / "alpha"
    payload.mkdir(parents=True)
    sidecar = payload / ".self_authored.json"
    sidecar.write_text("{}", encoding="utf-8")
    alias = data_root / "memory" / "payload-sidecar-copy.json"
    alias.parent.mkdir(parents=True)
    try:
        os.link(sidecar, alias)
    except OSError:
        pytest.skip("Hardlinks unavailable on this filesystem")

    assert is_skill_control_plane_path(alias, data_root) is True


def test_control_plane_policy_catches_hardlink_alias_inside_payload(tmp_path):
    data_root = tmp_path / "data"
    payload = data_root / "skills" / "external" / "alpha"
    payload.mkdir(parents=True)
    sidecar = payload / ".self_authored.json"
    sidecar.write_text("{}", encoding="utf-8")
    alias = payload / "payload-sidecar-copy.json"
    try:
        os.link(sidecar, alias)
    except OSError:
        pytest.skip("Hardlinks unavailable on this filesystem")

    assert is_skill_control_plane_path(alias, data_root) is True


def test_registry_heal_sidecar_wrapper_uses_shared_control_filenames():
    from ouroboros.tools.registry import _heal_protected_payload_sidecar

    for filename in SKILL_PAYLOAD_CONTROL_FILENAMES:
        assert _heal_protected_payload_sidecar(f"nested/{filename}") is True
    assert _heal_protected_payload_sidecar("nested/plugin.py") is False


def test_registry_shell_guard_keeps_legacy_control_dir_subset(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    blocked = _shell_guard_text(reg,
        {"cmd": "rm data/skills/external/alpha/.self_authored.json"},
        "advanced",
    )
    assert blocked is not None
    assert "SAFETY_VIOLATION" in blocked

    allowed = _shell_guard_text(reg,
        {"cmd": "rm data/skills/external/alpha/__pycache__/plugin.pyc"},
        "advanced",
    )
    assert allowed is None
