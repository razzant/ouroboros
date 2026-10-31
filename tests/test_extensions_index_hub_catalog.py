"""GET /api/extensions is a local read: it never fetches a skill source.

A source's trust is an install basis read by the review gate, so the listing needs
no catalog to state review facts: ``owner_attestable`` (Skip review) is offered for
every origin with readable bytes, and no source repository is fetched, whatever the
number of installed hub skills.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from ouroboros.contracts.skill_manifest import SkillManifest
from ouroboros.skill_loader import LoadedSkill, SkillReviewState


def _skill(drive_root: pathlib.Path, bucket: str, slug: str) -> LoadedSkill:
    skill_dir = drive_root / "skills" / bucket / slug
    skill_dir.mkdir(parents=True)
    body = f"---\nname: {slug}\ndescription: demo\nversion: 1.0.0\ntype: instruction\n---\n".encode("utf-8")
    (skill_dir / "SKILL.md").write_bytes(body)
    return LoadedSkill(
        name=slug,
        skill_dir=skill_dir,
        manifest=SkillManifest(name=slug, description="demo", version="1.0.0", type="instruction"),
        content_hash=hashlib.sha256(slug.encode("utf-8")).hexdigest(),
        enabled=False,
        review=SkillReviewState(status="pending"),
        load_error="",
        source=bucket,
    )


@pytest.fixture
def index(monkeypatch, tmp_path):
    """Build the real index over ``count`` hub skills; any source fetch fails the test."""
    from ouroboros import skill_source_catalog, skill_source_git
    from ouroboros.gateway import extensions as extensions_api

    drive_root = tmp_path / "data"
    monkeypatch.setattr(
        extensions_api, "snapshot",
        lambda: {"tools": [], "routes": [], "ws_handlers": [], "ui_tabs": []},
    )
    monkeypatch.setattr("ouroboros.tools.github.github_token_from_env_or_settings", lambda: "")

    def _no_fetch(*_a, **_kw):
        raise AssertionError("the index must never fetch a skill source")

    monkeypatch.setattr(skill_source_git, "update_clone", _no_fetch)
    monkeypatch.setattr(skill_source_catalog, "refresh_in_background", _no_fetch)

    def build(count: int, *, extra=()):
        skills = [_skill(drive_root, "ouroboroshub", f"hub-skill-{i}") for i in range(count)] + list(extra)
        monkeypatch.setattr(extensions_api, "discover_skills", lambda *_a, **_kw: list(skills))
        payload = extensions_api._build_extensions_index(drive_root, repo_path="")
        return {row["name"]: row for row in payload["skills"]}, skills

    return build, drive_root


@pytest.mark.parametrize("count", [1, 18])
def test_hub_rows_need_no_catalog_and_offer_skip_review(index, count):
    build, drive_root = index
    own = _skill(drive_root, "external", "my-own")
    rows, _skills = build(count, extra=[own])
    assert len(rows) == count + 1
    assert all(row["owner_attestable"] is True for row in rows.values())
    assert all("official_hub_verified" not in row for row in rows.values())
    assert all(row["install"] is None for row in rows.values())  # no install record: no source facts


def test_the_manifest_read_never_fetches(index, monkeypatch):
    """GET /api/extensions/{skill}/manifest carries the same review facts and no source read."""
    import asyncio

    from ouroboros.gateway import extensions as extensions_api

    build, drive_root = index
    _rows, skills = build(1)
    monkeypatch.setattr(extensions_api, "find_skill", lambda *_a, **_kw: skills[0])
    monkeypatch.setattr("ouroboros.extension_loader.runtime_state_for_skill_name", lambda *_a, **_kw: {})
    monkeypatch.setattr(extensions_api, "_request_drive_root", lambda _request: drive_root)

    class _Request:
        path_params = {"skill": "hub-skill-0"}

    body = json.loads(asyncio.run(extensions_api.api_extension_manifest(_Request())).body)
    assert body["owner_attestable"] is True and "official_hub_verified" not in body
