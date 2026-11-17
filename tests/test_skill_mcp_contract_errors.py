"""Public Connect refusals and the skill MCP runtime declaration contract."""
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
    with pytest.raises(SkillManifestError, match="only node or agent_python runtimes"):
        parse_skill_manifest_text(text)
    # The existing script contract is unaffected by this new optional declaration.
    assert parse_skill_manifest_text(text.replace('mcp_servers: [{id: demo, entry: server}]\n', '')).runtime == runtime


def test_agent_python_is_only_an_explicit_mcp_runtime_and_node_still_inherits():
    text = ('---\nname: demo\ndescription: d\nversion: 1.0.0\ntype: script\n'
            'runtime: python3\nscripts: [setup.py]\n'
            'mcp_servers: [{id: demo, runtime: agent_python, entry: server.py}]\n---\n')
    manifest = parse_skill_manifest_text(text)
    assert manifest.runtime == "python3"
    assert manifest.mcp_servers[0]["runtime"] == "agent_python"
    top_level = parse_skill_manifest_text(text.replace("runtime: python3", "runtime: agent_python"))
    assert any("unknown runtime 'agent_python'" in warning for warning in top_level.validate())
    inherited = parse_skill_manifest_text(text.replace("runtime: python3", "runtime: node")
                                           .replace("runtime: agent_python, ", ""))
    assert inherited.mcp_servers[0]["runtime"] == "node"


def test_optional_servers_append_to_legacy_servers_using_the_same_contract():
    node = {"id": "existing", "entry": "server.mjs"}
    python = {"id": "extra", "runtime": "agent_python", "entry": "server.py", "task_session": True}
    manifest = parse_skill_manifest_text(json.dumps({
        "runtime": "node", "optional_mcp_servers": [python], "mcp_servers": [node],
    }))
    single_list = parse_skill_manifest_text(json.dumps({"runtime": "node", "mcp_servers": [node, python]}))
    assert manifest.mcp_servers == single_list.mcp_servers
    assert [(row["id"], row["runtime"]) for row in manifest.mcp_servers] == [
        ("existing", "node"), ("extra", "agent_python"),
    ]
    assert "optional_mcp_servers" not in manifest.raw_extra


@pytest.mark.parametrize("empty", [None, "", []])
def test_legacy_empty_declarations_still_allow_an_optional_server(empty):
    manifest = parse_skill_manifest_text(json.dumps({
        "mcp_servers": empty, "optional_mcp_servers": [{"id": "extra", "entry": "server.py", "runtime": "agent_python"}],
    }))
    assert [row["id"] for row in manifest.mcp_servers] == ["extra"]


@pytest.mark.parametrize("extra,problem", [
    (None, "must be a list"), ("", "must be a list"), ({}, "must be a list"),
    (["other"], "must be a mapping"),
    ([{"id": "other", "entry": "../server.py"}], "relative path"),
    ([{"id": "other", "entry": "server.py", "command": "python"}], "unknown keys"),
    ([{"id": "other", "entry": "server.py", "runtime": "python3"}], "supports only"),
    ([{"id": "other", "entry": "server.mjs", "browser_policy": True}], "without task_session"),
    ([{"id": "other", "entry": "one.mjs"}, {"id": "other", "entry": "two.mjs"}], "declared twice"),
    ([{"id": "existing", "entry": "replacement.py", "runtime": "agent_python"}], "declared twice"),
])
def test_invalid_optional_declaration_rejects_the_manifest(extra, problem):
    text = json.dumps({"runtime": "node", "mcp_servers": [{"id": "existing", "entry": "server.mjs"}],
                       "optional_mcp_servers": extra})
    with pytest.raises(SkillManifestError, match=f"optional_mcp_servers.*{problem}"):
        parse_skill_manifest_text(text)


@pytest.mark.parametrize("runtime", ["[python]", "{name: python}"])
def test_mcp_runtime_wrong_shape_has_typed_manifest_error(runtime):
    text = ('---\nname: demo\ndescription: d\nversion: 1.0.0\ntype: script\n'
            'runtime: node\nscripts: [setup.mjs]\n'
            f'mcp_servers: [{{id: demo, runtime: {runtime}, entry: server.mjs}}]\n---\n')
    with pytest.raises(SkillManifestError, match="supports only node or agent_python runtimes"):
        parse_skill_manifest_text(text)


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
