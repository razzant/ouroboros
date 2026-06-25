"""Security-focused tests for ouroboros.marketplace.clawhub.

Trimmed in v5.16.0-rc.2 to keep only the host-allowlist, redirect,
response-size-cap, traversal-slug, and registry-URL invariants. The
broader search/info/enrichment matrix (mocked HTTP envelope variants)
was removed because it pinned implementation envelopes more than real
security boundaries. ``test_marketplace_fetcher.py`` continues to cover
archive validation; this file now covers registry-client boundaries.
"""

from __future__ import annotations

from unittest import mock

import pytest

from ouroboros.marketplace import clawhub as clawhub_mod
from ouroboros.marketplace.clawhub import ClawHubClientError, info


def _mock_response(body: bytes, *, status: int = 200, headers: dict | None = None):
    response = mock.MagicMock()
    response.status = status
    response.getcode.return_value = status
    response.headers = headers or {"Content-Type": "application/json"}
    chunks = [body[: max(1, len(body) // 2)], body[max(1, len(body) // 2) :], b""]
    iter_chunks = iter(chunks)
    response.read = lambda _n=64 * 1024: next(iter_chunks, b"")
    cm = mock.MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return cm


def _patch_opener(body, *, status=200, headers=None):
    # Openers are now built lazily and selected by process role via
    # _active_opener(); patch that selector to return a fake opener so the test
    # does not depend on a module-level opener being pre-built.
    fake = mock.Mock()
    fake.open.return_value = _mock_response(body, status=status, headers=headers)
    return mock.patch.object(clawhub_mod, "_active_opener", return_value=fake)


# ---------------------------------------------------------------------------
# Registry host allowlist
# ---------------------------------------------------------------------------


def test_evil_host_is_blocked():
    with pytest.raises(clawhub_mod.ClawHubClientHostBlocked):
        clawhub_mod._registry_base_url("https://evil.example.com/api")


def test_http_only_blocked_for_non_localhost():
    with pytest.raises(clawhub_mod.ClawHubClientHostBlocked):
        clawhub_mod._registry_base_url("http://clawhub.ai/api")


def test_localhost_http_allowed_for_dev():
    assert clawhub_mod._registry_base_url(
        "http://localhost:8081/api/v1"
    ).startswith("http://localhost")


def test_clawhub_ai_default():
    assert clawhub_mod._registry_base_url(None) == "https://clawhub.ai/api/v1"


def test_clawhub_com_host_no_longer_allowed():
    """v4.50 removed clawhub.com from the host allowlist (ownership unverified)."""
    with pytest.raises(clawhub_mod.ClawHubClientHostBlocked):
        clawhub_mod._registry_base_url("https://clawhub.com/api/v1")


def test_registry_url_strips_query_strings(monkeypatch):
    """Query strings on the registry URL must be discarded."""
    from ouroboros.config import get_clawhub_registry_url

    monkeypatch.setenv("OUROBOROS_CLAWHUB_REGISTRY_URL", "https://clawhub.ai/api/v1?key=foo")
    assert get_clawhub_registry_url() == "https://clawhub.ai/api/v1"


# ---------------------------------------------------------------------------
# Redirect handler allowlist
# ---------------------------------------------------------------------------


def test_redirect_to_evil_host_is_blocked():
    """A 30x Location pointing outside the allowlist must raise."""
    handler = clawhub_mod._redirect_handler()
    fake_req = mock.MagicMock()
    fake_fp = mock.MagicMock()
    fake_headers = {"Location": "https://evil.example.com/x"}
    with pytest.raises(clawhub_mod.ClawHubClientHostBlocked):
        handler.redirect_request(
            fake_req, fake_fp, 302, "Found", fake_headers, "https://evil.example.com/x"
        )


def test_redirect_to_allowed_host_is_followed():
    """A redirect within the allowlist must not be blocked at the host check."""
    handler = clawhub_mod._redirect_handler()
    fake_req = mock.MagicMock()
    fake_fp = mock.MagicMock()
    fake_headers = {}
    try:
        handler.redirect_request(
            fake_req, fake_fp, 302, "Found", fake_headers, "https://www.clawhub.ai/x"
        )
    except clawhub_mod.ClawHubClientHostBlocked:
        pytest.fail("Allowed host was incorrectly blocked")
    except Exception:
        # Other exceptions from the parent class for missing args are fine —
        # what matters is that it's NOT ClawHubClientHostBlocked.
        pass


# ---------------------------------------------------------------------------
# Response-size cap and slug-traversal guards
# ---------------------------------------------------------------------------


def test_response_size_cap_enforced():
    """Massive payload should raise rather than allocate without bound."""
    huge = b"x" * (clawhub_mod._MAX_JSON_RESPONSE_BYTES + 100)
    with _patch_opener(huge):
        with pytest.raises(ClawHubClientError):
            clawhub_mod.search("foo")


def test_info_blank_slug_rejected():
    with pytest.raises(ClawHubClientError):
        info("")


@pytest.mark.parametrize("bad_slug", ["../etc", "../../etc", "foo/../bar", "./x"])
def test_info_rejects_traversal_slugs(bad_slug):
    with pytest.raises(ClawHubClientError, match="must not contain"):
        info(bad_slug)
