"""Both wizard entry points disclose fresh defaults without writing settings."""

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient



@pytest.mark.parametrize(("existing", "surface"), [(False, "page"), (True, "readiness")])
def test_wizard_freshness_comes_from_settings_file(surface, existing, monkeypatch, tmp_path):
    from ouroboros import config
    from ouroboros.gateway import onboarding_host, settings

    path = tmp_path / "settings.json"
    content = '{"OUROBOROS_MODEL":"google/gemini-3.8-flash"}'
    if existing:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    # Both surfaces read the one live Setup derivation in onboarding_host.
    module, handler = (onboarding_host, onboarding_host.onboarding_page) if surface == "page" else (
        onboarding_host, settings.api_onboarding)
    monkeypatch.setattr(module, "load_settings", lambda: {
        "OUROBOROS_MODEL": "google/gemini-3.8-flash",
    })
    with TestClient(Starlette(routes=[Route("/wizard", handler)])) as client:
        response = client.get("/wizard")
    assert response.status_code == 200
    expected = "false" if existing else "true"
    assert f'"freshInstall": {expected}' in response.text
    assert path.exists() is existing
    if existing:
        assert path.read_text(encoding="utf-8") == content
