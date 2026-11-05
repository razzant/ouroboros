"""Targeted coverage for the v6.12.0 QA fixes that are not behavioural:

  - marketplace cross-bucket name-collision auto-rename;
  - ClawHub registry error mapping (404 -> 404, 429 -> 429);
  - ClawHub preview labelled metadata-only (not archive-validated);
  - Skills "Make runnable" affordance + repair-prompt conversion hint;
  - Files header no longer duplicates the directory description;
  - mcp_settings imports apiFetch so the MCP status refresh works.
"""

from types import SimpleNamespace


# --------------------------------------------------------------------------- #
# Marketplace: cross-bucket collision auto-rename
# --------------------------------------------------------------------------- #


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
