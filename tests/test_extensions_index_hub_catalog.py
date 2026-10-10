"""GET /api/extensions is a local read: it never waits for the OuroborosHub catalog.

``official_hub_verified`` (and the hub side of ``owner_attestable``) is a display
hint — the review profile and the owner-attestation endpoint re-verify against a
fresh catalog. The index therefore only PEEKS at the display-plane catalog memo
(§7.1a): ``True``/``False`` when a fresh view exists, ``None`` when it does not,
and no download either way, whatever the number of installed hub skills.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from ouroboros.contracts.skill_manifest import SkillManifest
from ouroboros.marketplace import ouroboroshub
from ouroboros.skill_loader import LoadedSkill, SkillReviewState

_RAW_BASE = "https://raw.githubusercontent.com/razzant/OuroborosHub/main"


def _skill(drive_root: pathlib.Path, bucket: str, slug: str) -> tuple[LoadedSkill, dict]:
    skill_dir = drive_root / "skills" / bucket / slug
    skill_dir.mkdir(parents=True)
    body = f"---\nname: {slug}\ndescription: demo\nversion: 1.0.0\ntype: instruction\n---\n".encode("utf-8")
    (skill_dir / "SKILL.md").write_bytes(body)
    files = [{"path": "SKILL.md", "sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}]
    if bucket == "ouroboroshub":
        sidecar = {"source": "ouroboroshub", "slug": slug, "sanitized_name": slug, "files": files}
        (skill_dir / ".ouroboroshub.json").write_text(json.dumps(sidecar), encoding="utf-8")
    loaded = LoadedSkill(
        name=slug,
        skill_dir=skill_dir,
        manifest=SkillManifest(name=slug, description="demo", version="1.0.0", type="instruction"),
        content_hash=hashlib.sha256(slug.encode("utf-8")).hexdigest(),
        enabled=False,
        review=SkillReviewState(status="pending"),
        load_error="",
        source=bucket,
    )
    return loaded, {"slug": slug, "version": "1.0.0", "files": files}


@pytest.fixture
def index(monkeypatch, tmp_path):
    """Build the real index over ``count`` hub skills; any catalog download fails the test."""
    from ouroboros.gateway import extensions as extensions_api

    drive_root = tmp_path / "data"
    monkeypatch.setattr(
        extensions_api, "snapshot",
        lambda: {"tools": [], "routes": [], "ws_handlers": [], "ui_tabs": []},
    )
    monkeypatch.setattr("ouroboros.tools.github.github_token_from_env_or_settings", lambda: "")
    monkeypatch.setattr(
        ouroboroshub, "_fetch_bytes",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("the index must never download the catalog")),
    )
    ouroboroshub._catalog_cache_clear()

    def build(count: int, *, memo_age_sec: float | None = 0.0, extra=()):
        built = [_skill(drive_root, "ouroboroshub", f"hub-skill-{i}") for i in range(count)]
        skills = [loaded for loaded, _row in built] + list(extra)
        monkeypatch.setattr(extensions_api, "discover_skills", lambda *_a, **_kw: list(skills))
        if memo_age_sec is not None:
            catalog = {"raw_base_url": _RAW_BASE, "skills": [row for _loaded, row in built]}
            ouroboroshub._catalog_cache_inject(catalog, age_sec=memo_age_sec)
        payload = extensions_api._build_extensions_index(drive_root, repo_path="")
        return {row["name"]: row for row in payload["skills"]}, skills

    yield build, drive_root
    ouroboroshub._catalog_cache_clear()


@pytest.mark.parametrize("count", [1, 18])
def test_fresh_display_view_verifies_every_hub_skill_without_a_download(index, count):
    build, _drive_root = index
    rows, _skills = build(count)
    assert len(rows) == count
    assert all(row["official_hub_verified"] is True for row in rows.values())
    assert all(row["owner_attestable"] is True for row in rows.values())


@pytest.mark.parametrize("memo_age_sec", [None, ouroboroshub._CATALOG_CACHE_TTL_SEC + 1])
def test_without_a_fresh_view_the_hub_facts_are_unknown_not_negative(index, memo_age_sec):
    build, drive_root = index
    own, _row = _skill(drive_root, "external", "my-own")
    rows, _skills = build(18, memo_age_sec=memo_age_sec, extra=[own])
    hub_rows = [row for name, row in rows.items() if name != "my-own"]
    assert len(hub_rows) == 18
    assert all(row["official_hub_verified"] is None for row in hub_rows)
    assert all(row["owner_attestable"] is None for row in hub_rows)
    # Local facts never depend on the catalog.
    assert rows["my-own"]["official_hub_verified"] is False
    assert rows["my-own"]["owner_attestable"] is True


def test_a_locally_edited_hub_payload_is_a_definitive_negative(index):
    build, _drive_root = index
    rows, skills = build(2)
    (skills[0].skill_dir / "SKILL.md").write_text("edited locally\n", encoding="utf-8")
    rows, _skills = build(0, memo_age_sec=None, extra=skills)
    assert rows["hub-skill-0"]["official_hub_verified"] is False
    assert rows["hub-skill-0"]["owner_attestable"] is False
    assert rows["hub-skill-1"]["official_hub_verified"] is True


def test_display_catalog_files_peeks_and_never_fetches(monkeypatch):
    monkeypatch.setattr(
        ouroboroshub, "_fetch_bytes",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("peek must not fetch")),
    )
    ouroboroshub._catalog_cache_clear()
    try:
        assert ouroboroshub.display_catalog_files() is None
        files = [{"path": "SKILL.md", "sha256": "ab", "size": 1}]
        catalog = {"raw_base_url": _RAW_BASE, "skills": [{"slug": "demo", "files": files}]}
        ouroboroshub._catalog_cache_inject(catalog)
        assert ouroboroshub.display_catalog_files() == {"demo": files}
        ouroboroshub._catalog_cache_inject(catalog, age_sec=ouroboroshub._CATALOG_CACHE_TTL_SEC + 1)
        assert ouroboroshub.display_catalog_files() is None
    finally:
        ouroboroshub._catalog_cache_clear()


def test_an_unreadable_hub_payload_fails_only_its_own_row(index, monkeypatch):
    """One payload the matcher cannot read is False for that row; the listing survives."""
    build, _drive_root = index
    _rows, skills = build(2)
    victim = skills[0].skill_dir / "SKILL.md"
    real_read_bytes = pathlib.Path.read_bytes

    def boom(self):
        if self == victim.resolve():
            raise OSError("payload vanished between is_file() and read_bytes()")
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", boom)
    rows, _skills = build(0, memo_age_sec=None, extra=skills)
    assert rows["hub-skill-0"]["official_hub_verified"] is False
    assert rows["hub-skill-0"]["owner_attestable"] is False
    assert rows["hub-skill-1"]["official_hub_verified"] is True


@pytest.mark.parametrize("skills_value", [{"not": "a list"}, [{"slug": "hub-skill-0", "files": 7}]])
def test_a_malformed_display_memo_is_no_view_not_a_failed_listing(index, skills_value):
    build, _drive_root = index
    _rows, skills = build(1)
    ouroboroshub._catalog_cache_inject({"raw_base_url": _RAW_BASE, "skills": skills_value})
    assert ouroboroshub.display_catalog_files() is None
    rows, _skills = build(0, memo_age_sec=None, extra=skills)
    assert rows["hub-skill-0"]["official_hub_verified"] is None


def test_the_manifest_read_peeks_the_same_view_and_never_downloads(index, monkeypatch):
    """GET /api/extensions/{skill}/manifest carries the same hub hint: a display
    peek, never a catalog fetch, unknown without a view."""
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
    assert body["official_hub_verified"] is True
    ouroboroshub._catalog_cache_clear()
    body = json.loads(asyncio.run(extensions_api.api_extension_manifest(_Request())).body)
    assert body["official_hub_verified"] is None
