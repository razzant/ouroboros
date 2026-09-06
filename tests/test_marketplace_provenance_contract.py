"""v5 Opus critic finding O-5 — pin /api/extensions provenance shape
against /api/marketplace/clawhub/installed so the two endpoints
cannot drift silently.

Both endpoints exist for legitimate reasons (the Skills pane consumes
the catalog endpoint; the Marketplace pane consumes the marketplace
endpoint), but they MUST return the same provenance shape for the
same clawhub skill — adding a field to one without the other would
silently break either tab. This test mocks the underlying skill_loader
+ provenance reader and asserts the projection matches.
"""

from __future__ import annotations

import asyncio
import pathlib
from unittest import mock



# The minimal canonical provenance record any v5 marketplace install
# writes via ouroboros.marketplace.provenance.write_provenance. We
# don't care about every field — we care that both endpoints surface
# the same SUBSET to the operator.
_CANONICAL_PROVENANCE = {
    "schema_version": 1,
    "source": "clawhub",
    "slug": "demo/hello",
    "sanitized_name": "demo__hello",
    "version": "1.0.0",
    "sha256": "a" * 64,
    "is_plugin": False,
    "installed_at": "2026-04-25T12:00:00+00:00",
    "updated_at": "2026-04-25T12:00:00+00:00",
    "homepage": "https://example.com/hello",
    "license": "MIT",
    "primary_env": "DEMO_API_KEY",
    "original_manifest_sha256": "b" * 64,
    "translated_manifest_sha256": "c" * 64,
    "adapter_version": "5.4.0",
    "openclaw_compat": {
        "adapter_version": "5.4.0",
        "requires": {"bins": [], "anyBins": [], "env": [], "config": ["browser.enabled"]},
    },
    "adapter_warnings": ["adapter could not derive permissions"],
    "registry_url": "https://clawhub.ai/api/v1",
}


# Fields the operator-facing UI cards need to be able to depend on.
# Both endpoints MUST expose these in their provenance projection.
# Adding a new field here without updating both endpoints is the bug
# this test exists to catch.
_REQUIRED_PROJECTION_FIELDS = frozenset(
    {
        "slug",
        "version",
        "sha256",
        "homepage",
        "license",
        "adapter_warnings",
        "adapter_version",
        "openclaw_compat",
    }
)


def test_required_projection_fields_match_between_endpoints(monkeypatch):
    """Cycle 1 Opus O-5 — pin the two provenance projections.

    Drives both endpoints with the same mock skill catalogue + the
    same provenance record, and asserts that the projected fields
    overlap on the required-set. A future commit that adds, say,
    ``installed_size_bytes`` to one endpoint without the other will
    fail this test.
    """
    # Cycle 2 GPT critic Finding 1: pre-import the API modules
    # BEFORE any monkeypatch.setattr() so the lazy ``from
    # ouroboros.skill_loader import discover_skills`` line inside
    # ``ouroboros.gateway.extensions`` resolves the REAL function and
    # binds it to the module namespace. monkeypatch then captures
    # the real function as oldval and correctly restores it on
    # teardown. Without this pre-import, a previous monkeypatch on
    # ``ouroboros.skill_loader.discover_skills`` would leak into the
    # later monkeypatch's saved oldval, polluting subsequent tests
    # that import these modules in default order.
    import ouroboros.gateway.extensions  # noqa: F401
    import ouroboros.gateway.marketplace  # noqa: F401
    import ouroboros.marketplace.provenance  # noqa: F401
    # Cycle 3 Gemini follow-up: pre-import ``skill_exec`` too, since
    # it is the one remaining module-level consumer of
    # ``ouroboros.skill_loader.discover_skills`` outside the
    # extension-API path. Without this pre-import a future test that
    # imports ``skill_exec`` after our monkeypatch would freeze its
    # binding on the stub.
    import ouroboros.tools.skill_exec  # noqa: F401

    from ouroboros.skill_loader import LoadedSkill, SkillReviewState
    from ouroboros.contracts.skill_manifest import SkillManifest

    fake_manifest = SkillManifest(
        name="demo__hello",
        description="x",
        version="1.0.0",
        type="instruction",
        permissions=[],
    )
    fake_skill = LoadedSkill(
        name="demo__hello",
        skill_dir=pathlib.Path("/tmp/notreal"),
        manifest=fake_manifest,
        content_hash="0" * 64,
        enabled=False,
        review=SkillReviewState(status="pending"),
        load_error="",
        source="clawhub",
    )

    def _stub_discover_skills(*_a, **_kw):
        return [fake_skill]

    def _stub_runtime_state_for_skill_name(*_a, **_kw):
        return {"desired_live": False, "live_loaded": False, "reason": "not_extension"}

    def _stub_snapshot():
        return {
            "extensions": [],
            "tools": [],
            "routes": [],
            "ws_handlers": [],
            "ui_tabs": [],
        }

    def _stub_read_provenance(*_a, **_kw):
        return dict(_CANONICAL_PROVENANCE)

    # Both endpoints lazy-import ``discover_skills`` from
    # ``ouroboros.skill_loader``, so we patch that single source so
    # both code paths see the same stubbed catalogue.
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", _stub_discover_skills)
    monkeypatch.setattr("ouroboros.gateway.extensions.discover_skills", _stub_discover_skills)
    monkeypatch.setattr("ouroboros.gateway.extensions.snapshot", _stub_snapshot)
    monkeypatch.setattr(
        "ouroboros.extension_loader.runtime_state_for_skill_name",
        _stub_runtime_state_for_skill_name,
    )
    # ``read_provenance`` is imported eagerly at module load time by
    # ``ouroboros.gateway.marketplace`` (so a monkeypatch on the original
    # module would never reach the marketplace endpoint's local
    # reference). ``ouroboros.gateway.extensions`` does a lazy import
    # inside ``api_extensions_index``, so the original-module patch
    # IS sufficient there. Patching both source locations covers both
    # styles.
    monkeypatch.setattr(
        "ouroboros.marketplace.provenance.read_provenance",
        _stub_read_provenance,
    )
    monkeypatch.setattr(
        "ouroboros.gateway.marketplace.read_provenance",
        _stub_read_provenance,
        raising=False,
    )

    # Drive /api/extensions
    from ouroboros.gateway.extensions import api_extensions_index
    from ouroboros.gateway.marketplace import api_marketplace_installed

    fake_request = mock.MagicMock()
    fake_request.app.state.drive_root = pathlib.Path("/tmp/notreal_drive")
    fake_request.app.state.repo_dir = pathlib.Path("/tmp/notreal_repo")
    monkeypatch.setattr(
        "ouroboros.gateway.extensions._request_drive_root",
        lambda req: pathlib.Path("/tmp/notreal_drive"),
    )
    monkeypatch.setattr(
        "ouroboros.gateway.marketplace._request_drive_root",
        lambda req: pathlib.Path("/tmp/notreal_drive"),
    )
    # v5.15.0: marketplace is always-on; the legacy ``get_clawhub_enabled`` /
    # ``OUROBOROS_CLAWHUB_ENABLED`` switch is retired. No setup needed here.

    loop = asyncio.new_event_loop()
    try:
        catalog_resp = loop.run_until_complete(api_extensions_index(fake_request))
        marketplace_resp = loop.run_until_complete(api_marketplace_installed(fake_request))
    finally:
        loop.close()

    import json
    catalog_body = json.loads(catalog_resp.body.decode("utf-8"))
    marketplace_body = json.loads(marketplace_resp.body.decode("utf-8"))

    catalog_clawhub = [s for s in catalog_body["skills"] if s.get("source") == "clawhub"]
    marketplace_skills = marketplace_body["skills"]
    assert len(catalog_clawhub) == 1, catalog_body
    assert len(marketplace_skills) == 1, marketplace_body

    catalog_prov = catalog_clawhub[0].get("provenance") or {}
    marketplace_prov = marketplace_skills[0].get("provenance") or {}
    assert catalog_prov, "Catalog endpoint must embed provenance for clawhub skills"
    assert marketplace_prov, "Marketplace installed endpoint must embed provenance"

    missing_in_catalog = _REQUIRED_PROJECTION_FIELDS - set(catalog_prov.keys())
    missing_in_marketplace = _REQUIRED_PROJECTION_FIELDS - set(marketplace_prov.keys())

    assert not missing_in_catalog, (
        f"/api/extensions clawhub provenance is missing required fields: "
        f"{sorted(missing_in_catalog)}. Both endpoints must surface the "
        f"same provenance subset to the UI."
    )
    assert not missing_in_marketplace, (
        f"/api/marketplace/clawhub/installed provenance is missing required fields: "
        f"{sorted(missing_in_marketplace)}."
    )

    # Pin the values too — the projection must stringify the same values.
    for field in _REQUIRED_PROJECTION_FIELDS:
        assert catalog_prov.get(field) == marketplace_prov.get(field), (
            f"Provenance field {field!r} differs between endpoints: "
            f"catalog={catalog_prov.get(field)!r}, marketplace={marketplace_prov.get(field)!r}"
        )
