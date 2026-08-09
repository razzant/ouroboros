"""Targeted coverage for the v6.12.0 QA fixes that are not behavioural:

  - marketplace cross-bucket name-collision auto-rename;
  - ClawHub registry error mapping (404 -> 404, 429 -> 429);
  - ClawHub preview labelled metadata-only (not archive-validated);
  - Skills "Make runnable" affordance + repair-prompt conversion hint;
  - Files header no longer duplicates the directory description;
  - mcp_settings imports apiFetch so the MCP status refresh works.
"""

import pathlib
from types import SimpleNamespace

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"


def _web(rel: str) -> str:
    return (WEB / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Marketplace: cross-bucket collision auto-rename
# --------------------------------------------------------------------------- #

def test_dedupe_marketplace_skill_name_avoids_cross_bucket_collision(tmp_path, monkeypatch):
    from ouroboros.marketplace import install as inst

    native_dir = tmp_path / "data" / "skills" / "native" / "weather"
    clawhub_root = tmp_path / "data" / "skills" / "clawhub"
    native_dir.mkdir(parents=True)
    clawhub_root.mkdir(parents=True)

    native_skill = SimpleNamespace(name="weather", skill_dir=native_dir)
    monkeypatch.setattr(
        "ouroboros.skill_loader.discover_skills",
        lambda drive_root, **kw: [native_skill],
    )
    # Collides with native/weather in another bucket -> suffixed.
    assert inst.dedupe_marketplace_skill_name(
        tmp_path / "data", clawhub_root, "weather", suffix="clawhub"
    ) == "weather-clawhub"

    # No collision -> name is preserved unchanged (no behaviour change).
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda drive_root, **kw: [])
    assert inst.dedupe_marketplace_skill_name(
        tmp_path / "data", clawhub_root, "weather", suffix="clawhub"
    ) == "weather"


def test_update_skill_pins_existing_dir_via_target_override(tmp_path, monkeypatch):
    """Update must reinstall into the SAME (possibly auto-renamed) directory, not
    re-derive the name via dedupe — otherwise a renamed skill could drift to a new
    dir and orphan the old one if the original collision is gone."""
    from ouroboros.marketplace import install as inst
    from ouroboros.marketplace.provenance import write_provenance

    write_provenance(
        tmp_path, "weather-clawhub",
        {"schema_version": 1, "source": "clawhub", "slug": "weather", "sanitized_name": "weather-clawhub"},
    )
    captured = {}

    def _fake_install(drive_root, repo_dir, **kw):
        captured.update(kw)
        return inst.InstallResult(ok=True, sanitized_name=kw.get("target_name_override") or "")

    monkeypatch.setattr(inst, "install_skill", _fake_install)
    inst.update_skill(tmp_path, tmp_path, sanitized_name="weather-clawhub")

    assert captured.get("target_name_override") == "weather-clawhub"
    assert captured.get("slug") == "weather"
    assert captured.get("overwrite") is True


def test_rewrite_staged_identity_syncs_sidecar_manifest_and_digest(tmp_path):
    """After a collision/override rename, the staged sidecar + SKILL.md frontmatter
    name agree with the directory AND the translated_manifest_sha256 audit digest
    is recomputed to match the rewritten SKILL.md (no stale hash)."""
    import hashlib
    import json
    from ouroboros.marketplace.install import _rewrite_staged_identity

    (tmp_path / ".clawhub.json").write_text(
        json.dumps({
            "sanitized_name": "weather",
            "slug": "weather",
            "translated_manifest_sha256": "stale-old-hash",
        }),
        encoding="utf-8",
    )
    (tmp_path / "SKILL.md").write_text(
        "---\nname: weather\ndescription: x\n---\n\nbody text\n", encoding="utf-8"
    )

    new_hash = _rewrite_staged_identity(tmp_path, "weather-clawhub")

    md = (tmp_path / "SKILL.md").read_text(encoding="utf-8")
    assert "name: weather-clawhub" in md
    assert "name: weather\n" not in md
    assert "body text" in md  # body preserved
    # Digest recomputed from the FINAL SKILL.md and propagated to the sidecar.
    assert new_hash == hashlib.sha256(md.encode("utf-8")).hexdigest()
    side = json.loads((tmp_path / ".clawhub.json").read_text(encoding="utf-8"))
    assert side["sanitized_name"] == "weather-clawhub"
    assert side["translated_manifest_sha256"] == new_hash
    assert side["translated_manifest_sha256"] != "stale-old-hash"


def test_dedupe_reuses_own_bucket_dir_on_reinstall(tmp_path, monkeypatch):
    from ouroboros.marketplace import install as inst

    clawhub_root = tmp_path / "data" / "skills" / "clawhub"
    existing = clawhub_root / "weather-clawhub"
    existing.mkdir(parents=True)
    # An in-bucket skill with the deduped name is NOT a collision (overwrite path).
    own = SimpleNamespace(name="weather-clawhub", skill_dir=existing)
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda drive_root, **kw: [own])
    assert inst.dedupe_marketplace_skill_name(
        tmp_path / "data", clawhub_root, "weather-clawhub", suffix="clawhub"
    ) == "weather-clawhub"


# --------------------------------------------------------------------------- #
# Marketplace: registry error mapping
# --------------------------------------------------------------------------- #

def test_client_error_response_status_mapping():
    from ouroboros.gateway.marketplace import _client_error_response
    from ouroboros.marketplace.clawhub import (
        ClawHubClientError,
        ClawHubClientHostBlocked,
        ClawHubNotFoundError,
        ClawHubRateLimitError,
    )

    assert _client_error_response(ClawHubNotFoundError("missing")).status_code == 404
    assert _client_error_response(ClawHubRateLimitError("https://x")).status_code == 429
    assert _client_error_response(ClawHubClientHostBlocked("blocked")).status_code == 400
    assert _client_error_response(ClawHubClientError("upstream")).status_code == 502
    assert _client_error_response(ValueError("other")).status_code == 500


def test_install_skill_preserves_registry_404_and_429_status(tmp_path, monkeypatch):
    """install/update routes must surface upstream 404/429 instead of a blanket
    400: install_skill records the status on InstallResult.error_status."""
    from ouroboros.marketplace import install as inst
    from ouroboros.marketplace.clawhub import ClawHubNotFoundError, ClawHubRateLimitError

    monkeypatch.setattr(inst, "_registry_info", lambda slug: (_ for _ in ()).throw(ClawHubNotFoundError("nope")))
    nf = inst.install_skill(tmp_path / "data", tmp_path / "repo", slug="ghost", auto_review=False)
    assert not nf.ok and nf.error_status == 404

    monkeypatch.setattr(inst, "_registry_info", lambda slug: (_ for _ in ()).throw(ClawHubRateLimitError("https://clawhub", 30)))
    rl = inst.install_skill(tmp_path / "data", tmp_path / "repo", slug="busy", auto_review=False)
    assert not rl.ok and rl.error_status == 429


def test_http_get_raises_not_found_on_404(monkeypatch):
    """A 404 from the registry surfaces as ClawHubNotFoundError, not the generic
    base (which the gateway would otherwise map to 502)."""
    import urllib.error
    from ouroboros.marketplace import clawhub

    def _raise_404(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(clawhub, "_active_opener", lambda: SimpleNamespace(open=_raise_404))
    try:
        clawhub._http_get("https://clawhub.ai/api/v1/skills/nope")
    except clawhub.ClawHubNotFoundError:
        pass
    else:
        raise AssertionError("expected ClawHubNotFoundError for a 404 response")


# --------------------------------------------------------------------------- #
# Marketplace: preview is metadata-only
# --------------------------------------------------------------------------- #

def test_preview_pipeline_flags_metadata_only(monkeypatch):
    from ouroboros.gateway import marketplace as mp

    fake = SimpleNamespace(is_plugin=False, latest_version="1.0.0", to_dict=lambda: {"slug": "weather"})
    monkeypatch.setattr(mp, "_registry_info", lambda slug: fake)

    out = mp._preview_pipeline("weather", None)
    assert out["metadata_only"] is True
    assert out["adapter"]["metadata_only"] is True
    assert out["adapter"]["ok"] is True
    assert any("metadata only" in w.lower() for w in out["adapter"]["warnings"])


# --------------------------------------------------------------------------- #
# Web UI fixes (static source contracts)
# --------------------------------------------------------------------------- #

def test_skills_offer_make_runnable_for_instruction_skills():
    src = _web("modules/skill_card_renderer.js")
    assert "skills-make-runnable" in src
    assert "skill.type === 'instruction'" in src
    # Reuses the existing repair action rather than a bespoke path.
    assert 'data-skill-action="repair"' in src


def test_repair_prompt_has_make_runnable_conversion_hint():
    src = _web("modules/utils.js")
    assert "Make-runnable" in src
    assert "type=script" in src


def test_files_header_does_not_duplicate_directory_description():
    src = _web("modules/files.js")
    # v6.12 bug: the directory blurb was rendered TWICE — once in the page header
    # and once in the preview pane. The read-only Files page has no directory meta
    # blurb at all (`defaultDirectoryMeta()` went with the file manager), so the
    # duplication is now structurally impossible rather than merely avoided. The
    # negative assertion stays as the standing guarantee; the old positive pin on
    # `defaultDirectoryMeta()` is dropped because the function no longer exists —
    # requiring a deleted helper would only re-fail the day someone deletes it
    # again. The whole name is asserted absent so a reintroduced blurb is caught.
    assert "description: defaultDirectoryMeta()" not in src
    assert "defaultDirectoryMeta" not in src


def test_mcp_settings_imports_apifetch_for_status_refresh():
    src = _web("modules/mcp_settings.js")
    assert "apiFetch" in src
    assert "} from './api_client.js';" in src
