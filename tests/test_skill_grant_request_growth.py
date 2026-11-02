"""A skill's key request grown by saved Settings keeps the owner's earlier approvals of the same bytes.

``requested_core_setting_keys`` asks a grant for a custom key only once that key is present in
Settings, so saving a new key a skill declares (from the Settings page or the protected secret form)
grows every declaring skill's request. The grown request must add the new key as missing and keep
every approval the owner already gave for those bytes: the reader (``grant_status_for_skill``), the
partial-approval writer (``save_skill_grants``) and a real consumer (a script's environment) agree.
New bytes still reset approvals, and nothing here grants a key or permission the owner did not.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.contracts.skill_manifest import SkillManifest
from ouroboros.skill_loader import (
    LoadedSkill,
    SkillReviewState,
    grant_status_for_skill,
    load_skill_grants,
    requested_core_setting_keys,
    save_skill_grants,
)

HASH = "bytes-1"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from ouroboros import config

    path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    def write(**values):
        path.write_text(json.dumps(values), encoding="utf-8")

    write(MAILER_HOST_TOKEN="host-synthetic")
    return write


def _skill(tmp_path: pathlib.Path, name: str, env, *, permissions=None, content_hash=HASH) -> LoadedSkill:
    (tmp_path / name).mkdir(exist_ok=True)
    manifest = SkillManifest(name=name, description="grant growth", version="0.1", type="script",
                             env_from_settings=list(env), permissions=list(permissions or []))
    return LoadedSkill(name=name, skill_dir=tmp_path / name, manifest=manifest, content_hash=content_hash,
                       review=SkillReviewState(status="pass", content_hash=content_hash))


def _grant(drive: pathlib.Path, skill: LoadedSkill, keys, permissions=()):
    from ouroboros.skill_loader import requested_skill_permissions

    save_skill_grants(drive, skill.name, list(keys), content_hash=skill.content_hash,
                      requested_keys=requested_core_setting_keys(list(skill.manifest.env_from_settings)),
                      granted_permissions=list(permissions),
                      requested_permissions=requested_skill_permissions(list(skill.manifest.permissions), []))


def test_a_new_shared_key_joins_the_request_missing_and_keeps_other_approvals(tmp_path, settings):
    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"])
    other = _skill(tmp_path, "other", ["SHARED_API_KEY"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN"])
    before = grant_status_for_skill(drive, mailer)
    assert before["requested_keys"] == ["MAILER_HOST_TOKEN"] and before["granted_keys"] == ["MAILER_HOST_TOKEN"]
    assert grant_status_for_skill(drive, other)["requested_keys"] == []  # an absent custom key is not requested

    settings(MAILER_HOST_TOKEN="host-synthetic", SHARED_API_KEY="shared-synthetic")

    after = grant_status_for_skill(drive, mailer)
    assert after["requested_keys"] == ["MAILER_HOST_TOKEN", "SHARED_API_KEY"]
    assert after["granted_keys"] == ["MAILER_HOST_TOKEN"], "the earlier approval of the same bytes survives"
    assert after["missing_keys"] == ["SHARED_API_KEY"], "the newly present key is never granted by itself"
    assert after["all_granted"] is False and after["usable"] is False
    other_after = grant_status_for_skill(drive, other)
    assert other_after["granted_keys"] == [] and other_after["missing_keys"] == ["SHARED_API_KEY"]
    assert load_skill_grants(drive, "other")["granted_keys"] == []  # reading wrote nothing


def test_a_partial_approval_after_growth_adds_the_key_without_revoking_keys_or_permissions(tmp_path, settings):
    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"], permissions=["inject_chat"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN"], ["inject_chat"])
    settings(MAILER_HOST_TOKEN="host-synthetic", SHARED_API_KEY="shared-synthetic")

    _grant(drive, mailer, ["SHARED_API_KEY"])  # the owner approves only the new key
    stored = load_skill_grants(drive, "mailer")
    assert sorted(stored["granted_keys"]) == ["MAILER_HOST_TOKEN", "SHARED_API_KEY"]
    assert stored["granted_permissions"] == ["inject_chat"]
    status = grant_status_for_skill(drive, mailer)
    assert status["granted_keys"] == ["MAILER_HOST_TOKEN", "SHARED_API_KEY"] and status["all_granted"] is True


def test_the_writer_never_grants_beyond_the_current_request(tmp_path, settings):
    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN", "SHARED_API_KEY", "UNDECLARED_KEY"])
    # SHARED_API_KEY is absent, so it is not requested yet: approving it now records nothing.
    assert load_skill_grants(drive, "mailer")["granted_keys"] == ["MAILER_HOST_TOKEN"]
    settings(MAILER_HOST_TOKEN="host-synthetic", SHARED_API_KEY="shared-synthetic")
    assert grant_status_for_skill(drive, mailer)["missing_keys"] == ["SHARED_API_KEY"]


def test_new_bytes_still_reset_every_approval(tmp_path, settings):
    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"], permissions=["inject_chat"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN"], ["inject_chat"])
    changed = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"], permissions=["inject_chat"],
                     content_hash="bytes-2")
    status = grant_status_for_skill(drive, changed)
    assert status["granted_keys"] == [] and status["granted_permissions"] == []
    _grant(drive, changed, [])  # a write under the new bytes carries nothing from the old ones
    assert load_skill_grants(drive, "mailer")["granted_keys"] == []
    assert load_skill_grants(drive, "mailer")["granted_permissions"] == []


def test_a_script_environment_follows_the_kept_approval(tmp_path, settings):
    from ouroboros.tools.skill_exec import _scrub_env

    drive = tmp_path / "drive"
    mailer = _skill(tmp_path, "mailer", ["MAILER_HOST_TOKEN", "SHARED_API_KEY"])
    _grant(drive, mailer, ["MAILER_HOST_TOKEN"])
    settings(MAILER_HOST_TOKEN="host-synthetic", SHARED_API_KEY="shared-synthetic")
    granted = grant_status_for_skill(drive, mailer)["granted_keys"]
    env = _scrub_env(list(mailer.manifest.env_from_settings), tmp_path / "state", "mailer", granted_keys=granted)
    assert env.get("MAILER_HOST_TOKEN") == "host-synthetic"
    assert "SHARED_API_KEY" not in env  # requested now, not granted: the script does not receive it
