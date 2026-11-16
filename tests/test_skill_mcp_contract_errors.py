"""Public Connect refusals and the first Node-only MCP declaration contract."""
import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.contracts.skill_manifest import SkillManifestError, parse_skill_manifest_text


@pytest.mark.parametrize("runtime", ["python", "python3", "go", "bash", "deno", "ruby"])
def test_mcp_declaration_does_not_promise_other_script_runtime_semantics(runtime):
    text = ('---\nname: demo\ndescription: d\nversion: 1.0.0\ntype: script\nruntime: '
            + runtime + '\nscripts: [server]\nmcp_servers: [{id: demo, entry: server}]\n---\n')
    with pytest.raises(SkillManifestError, match="only the node runtime"):
        parse_skill_manifest_text(text)
    # The existing script contract is unaffected by this new optional declaration.
    assert parse_skill_manifest_text(text.replace('mcp_servers: [{id: demo, entry: server}]\n', '')).runtime == runtime


@pytest.mark.parametrize("present,status", [(False, 404), (True, 409)])
def test_connect_http_early_lifecycle_refusal_reports_not_saved(tmp_path, monkeypatch, present, status):
    from ouroboros import config, skill_lifecycle_actions
    from ouroboros.gateway.extensions import api_skill_mcp_connect
    from ouroboros.gateway.schema import validate_ingress
    from ouroboros.gateway.skill_mcp_contracts import SkillMcpConnectResponse

    data = tmp_path / 'data'
    data.mkdir()
    settings = data / 'settings.json'
    before = json.dumps({'MCP_ENABLED': False, 'custom': 'unchanged'})
    settings.write_text(before)
    monkeypatch.setattr(config, 'DATA_DIR', data)
    monkeypatch.setattr(config, 'SETTINGS_PATH', settings)
    # Real lifecycle selection, with only installation lookup controlled: deletion
    # and revision drift occur before Connect's settings mutation owner is entered.
    monkeypatch.setattr(skill_lifecycle_actions, 'find_skill', lambda *_a, **_kw:
                        SimpleNamespace(content_hash='new') if present else None)
    app = Starlette(routes=[Route('/api/skills/{skill}/mcp-connect', api_skill_mcp_connect, methods=['POST'])])
    app.state.drive_root = data
    app.state.repo_dir = tmp_path
    with TestClient(app) as client:
        response = client.post('/api/skills/demo/mcp-connect', json={
            'expected_content_hash': 'old', 'settings_digest': 'preview'})
    assert response.status_code == status, response.text
    result = response.json()
    assert result['ok'] is False and result['saved'] is False
    assert 'status_code' not in result  # internal routing metadata is not a response field
    assert validate_ingress(result, SkillMcpConnectResponse) == []
    assert settings.read_text() == before
