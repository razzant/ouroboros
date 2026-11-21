"""One-way MCP client-file import preview: translation, matching, problems and the route.

Values are synthetic. The preview is pure: it never connects, starts, saves or
changes the draft list it is given. The draft list is matching metadata only
(``MCPImportDraftServer``): identities, transports and header names. The journey at
the end carries one document on through Apply, Enable and Save to a real listing.
"""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import time

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import mcp_client
from ouroboros.mcp_import import preview_import
from tests.test_mcp_headers_loopback import start_asgi
from tests.test_mcp_task_session import SERVER, rows
from tests.test_mcp_ui_static import _node_bin
from tests.test_settings_secret_mask import settings_client  # noqa: F401

BEARER = "Bearer synthetic-import-0001"
CUSTOM = "synthetic-import-key-0002"
ENV_SECRET = "synthetic-import-env-0003"


def preview(document, draft=()):
    draft = list(draft)
    before = copy.deepcopy(draft)
    result = preview_import(document if isinstance(document, str) else json.dumps(document), draft)
    assert draft == before
    return result


def entries(document, draft=()):
    result = preview(document, draft)
    assert result["ok"], result
    return {entry["source_name"]: entry for entry in result["entries"]}


def displayed(entry):
    return json.dumps({key: value for key, value in entry.items() if key != "patch"})


def test_new_http_and_stdio_servers_are_disabled_adds_with_literal_values():
    found = entries({"mcpServers": {
        "Docs": {"type": "http", "url": "https://docs.example/mcp",
                 "headers": {"Authorization": BEARER, "X-Api-Key": CUSTOM}},
        "files": {"command": "npx", "args": ["-y", "server", "--token", ENV_SECRET],
                  "env": {"API_TOKEN": ENV_SECRET}, "cwd": "/srv/files"},
    }})
    docs, files = found["Docs"], found["files"]
    assert docs["action"] == files["action"] == "add"
    assert docs["patch"] == {"id": "docs", "name": "Docs", "enabled": False, "transport": "streamable_http",
                             "url": "https://docs.example/mcp",
                             "headers": {"Authorization": BEARER, "X-Api-Key": CUSTOM}}
    assert "auth_token" not in docs["patch"] and "auth_header" not in docs["patch"]
    assert docs["header_names"] == ["Authorization", "X-Api-Key"]
    assert files["patch"]["args"] == ["-y", "server", "--token", ENV_SECRET] and files["arg_count"] == 4
    assert files["env_names"] == ["API_TOKEN"] and "ordinary visible settings" in files["warnings"][0]
    for entry in (docs, files):
        for secret in (BEARER, CUSTOM, ENV_SECRET):
            assert secret not in displayed(entry)


@pytest.mark.parametrize("declared, transport", [
    ({"type": "http"}, "streamable_http"), ({"transport": "sse"}, "sse"), ({}, "streamable_http"),
])
def test_http_transport_names_map_explicitly(declared, transport):
    entry = entries({"mcpServers": {"a": {**declared, "url": "https://a.example/mcp"}}})["a"]
    assert entry["transport"] == entry["patch"]["transport"] == transport


@pytest.mark.parametrize("raw, problem", [
    ({"type": "websocket", "url": "https://a.example"}, "type must be one of"),
    ({"type": "http", "transport": "sse", "url": "https://a.example"}, "name different transports"),
    ({"url": "https://a.example", "command": "npx"}, "both url and command"),
    ({"type": "stdio", "url": "https://a.example"}, "url cannot be used with transport stdio"),
    ({"url": "https://a.example", "headers": None}, "null is not a value for: headers"),
    ({"command": ["npx"]}, "command must be a string"),
])
def test_malformed_entries_are_problems_not_partial_imports(raw, problem):
    entry = entries({"mcpServers": {"a": raw}})["a"]
    assert not entry["action"] and entry["patch"] == {}
    assert any(problem in text for text in entry["problems"]), entry["problems"]
    assert CUSTOM not in displayed(entry)


@pytest.mark.parametrize("text, error", [
    ('{"mcpServers": {"a": {"url": "https://a.example"}, "a": {}}}', "duplicate JSON object key"),
    ('[]', "top-level mcpServers or servers object"),
    ('{"mcpServers": {}, "servers": {}}', "both mcpServers and servers"),
])
def test_unreadable_documents_are_refused_whole(text, error):
    result = preview(text)
    assert result["ok"] is False and error in result["error"] and result["entries"] == []


def test_unsupported_keys_are_named_without_values_and_other_top_level_keys_are_ignored():
    result = preview({"mcpServers": {"a": {"url": "https://a.example/mcp", "disabled": False,
                                           "headersHelper": "print-" + CUSTOM, "oauth": {"secret": CUSTOM}}},
                      "globalShortcut": "Ctrl+Space"})
    entry = result["entries"][0]
    assert entry["action"] == "add" and entry["unsupported_keys"] == ["disabled", "headersHelper", "oauth"]
    assert set(entry["patch"]) == {"id", "name", "enabled", "transport", "url"}
    assert result["ignored_keys"] == ["globalShortcut"] and CUSTOM not in json.dumps(result)


def test_url_credentials_are_never_displayed(monkeypatch):
    url = f"https://user:{CUSTOM}@docs.example/mcp"
    refused = entries({"mcpServers": {"a": {"url": url}}})["a"]
    assert "username/password" in refused["problems"][0] and CUSTOM not in json.dumps(refused)
    monkeypatch.setattr(mcp_client, "mode_has_unrestricted_agency", lambda _mode: True)
    accepted = entries({"mcpServers": {"a": {"url": url}}})["a"]
    assert "url" not in accepted and CUSTOM not in displayed(accepted)
    assert accepted["patch"]["url"] == url


DRAFT = [
    {"id": "docs", "name": "Docs (renamed)", "transport": "streamable_http",
     "header_names": ["Authorization", "X-Tenant"], "legacy_header": "Authorization"},
    {"id": "files", "transport": "stdio", "header_names": [], "legacy_header": None},
]


def test_update_touches_only_named_fields_and_keeps_every_header_when_headers_is_absent():
    found = entries({"mcpServers": {
        "Docs": {"url": "https://docs-v2.example/mcp"},
        "files": {"command": "uvx", "cwd": ""},
    }}, DRAFT)
    assert found["Docs"]["action"] == "update" and found["Docs"]["index"] == 0
    assert found["Docs"]["patch"] == {"transport": "streamable_http", "url": "https://docs-v2.example/mcp"}
    assert found["Docs"]["kept_header_names"] == ["Authorization", "X-Tenant"]
    assert "header_changes" not in found["Docs"] and found["Docs"]["legacy_header"] == "Authorization"
    assert found["files"]["patch"] == {"transport": "stdio", "command": "uvx", "cwd": ""}
    assert "kept_header_names" not in found["files"] and not found["Docs"]["warnings"]


def test_present_headers_replace_the_whole_list_and_clear_the_legacy_pair_in_the_same_patch():
    replaced = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp",
                                                "headers": {"X-Api-Key": CUSTOM, "x-tenant": "t2"}}}}, DRAFT)["docs"]
    assert replaced["patch"] == {"transport": "streamable_http", "url": "https://docs.example/mcp",
                                 "headers": {"X-Api-Key": CUSTOM, "x-tenant": "t2"},
                                 "auth_header": "Authorization", "auth_token": ""}
    assert replaced["header_changes"] == {"added": ["X-Api-Key"], "replaced": ["X-Tenant"],
                                          "removed": ["Authorization"]}
    assert replaced["legacy_header"] == "Authorization" and not replaced["warnings"]
    cleared = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp", "headers": {}}}}, DRAFT)["docs"]
    assert cleared["patch"]["headers"] == {} and cleared["patch"]["auth_token"] == ""
    assert cleared["header_changes"] == {"added": [], "replaced": [], "removed": ["Authorization", "X-Tenant"]}


def test_imported_authorization_replaces_the_saved_single_token_instead_of_colliding():
    entry = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp",
                                             "headers": {"authorization": BEARER}}}}, DRAFT)["docs"]
    assert entry["action"] == "update" and entry["patch"]["headers"] == {"authorization": BEARER}
    assert entry["patch"]["auth_token"] == "" and entry["header_changes"]["replaced"] == ["Authorization"]
    assert not entry["warnings"] and BEARER not in displayed(entry)
    plain = [dict(DRAFT[0], header_names=["X-Tenant"], legacy_header=None), DRAFT[1]]
    entry = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp",
                                             "headers": {"Authorization": BEARER}}}}, plain)["docs"]
    assert entry["patch"]["auth_token"] == "" and "legacy_header" not in entry


@pytest.mark.parametrize("headers", [{}, {"X-Api-Key": CUSTOM}])
def test_present_headers_clear_the_legacy_pair_even_when_the_draft_no_longer_shows_it(headers):
    """The draft row was removed unsaved: its server object still holds the saved token, so the patch clears it."""
    removed = [dict(DRAFT[0], header_names=["X-Tenant"], legacy_header=None), DRAFT[1]]
    entry = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp", "headers": headers}}}, removed)["docs"]
    assert entry["patch"]["headers"] == headers
    assert (entry["patch"]["auth_header"], entry["patch"]["auth_token"]) == ("Authorization", "")
    assert entry["header_changes"]["removed"] == ["X-Tenant"] and "legacy_header" not in entry
    kept = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp"}}}, removed)["docs"]
    assert "auth_token" not in kept["patch"] and kept["kept_header_names"] == ["X-Tenant"]


def test_an_unreadable_saved_map_is_named_when_a_file_replaces_it():
    unreadable = [dict(DRAFT[0], header_names=[], legacy_header=None, unreadable_headers=True)]
    entry = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp",
                                             "headers": {"X-Api-Key": CUSTOM}}}}, unreadable)["docs"]
    assert entry["unreadable_headers"] is True and entry["header_changes"]["added"] == ["X-Api-Key"]


def test_a_transport_change_is_refused_with_remove_and_add_guidance():
    entry = entries({"mcpServers": {"docs": {"type": "sse", "url": "https://docs.example/sse"}}}, DRAFT)["docs"]
    assert not entry["action"] and "remove that server first" in entry["problems"][0]


def test_stdio_call_lifetime_flags_are_imported_as_written_and_the_loader_applies_them():
    found = entries({"mcpServers": {
        "held": {"type": "stdio", "command": "node", "args": ["server.mjs"], "task_session": True,
                 "browser_policy": True, "autoApprove": ["x"]},
        "session": {"command": "node", "task_session": True},
        "off": {"command": "node", "task_session": False, "browser_policy": False},
        "plain": {"command": "node"},
        "remote": {"type": "http", "url": "https://docs.example/mcp"},
    }})
    assert found["held"]["action"] == "add" and found["held"]["patch"] == {
        "id": "held", "name": "held", "enabled": False, "transport": "stdio", "command": "node",
        "args": ["server.mjs"], "task_session": True, "browser_policy": True}
    assert found["held"]["unsupported_keys"] == ["autoApprove"]  # an unknown key is still named, not imported
    for untouched in ("plain", "remote"):  # no document that names neither flag changes
        assert found[untouched]["action"] == "add" and not {"task_session", "browser_policy"} & set(found[untouched]["patch"])
    errors = []
    parsed = {cfg.id: cfg for cfg in mcp_client.parse_servers([row["patch"] for row in found.values()], errors=errors)}
    assert not errors and not any(cfg.configuration_warnings for cfg in parsed.values())
    assert {name: (cfg.task_session, cfg.browser_policy, cfg.enabled) for name, cfg in parsed.items()} == {
        "held": (True, True, False), "session": (True, False, False), "off": (False, False, False),
        "plain": (False, False, False), "remote": (False, False, False)}


def test_a_title_as_the_key_changes_the_server_id_that_names_every_tool():
    """Why a setup document keys its entry by the server ID: the key is the identity, and the title is not it."""
    keyed = entries({"mcpServers": {"chrome_browser": {"command": "node"}}})["chrome_browser"]
    titled = entries({"mcpServers": {"Chrome (shared tab)": {"command": "node"}}})["Chrome (shared tab)"]
    assert (keyed["server_id"], titled["server_id"]) == ("chrome_browser", "chrome_shared_tab")


@pytest.mark.parametrize("transport", ["http", "sse"])
def test_explicit_false_lifetime_flags_preserve_the_remote_default(transport):
    raw = {"type": transport, "url": "https://example.test/mcp", "task_session": False, "browser_policy": False}
    row = entries({"mcpServers": {"remote": raw}})["remote"]
    assert row["action"] == "add" and not row["problems"] and not row["unsupported_keys"]
    errors = []
    cfg = mcp_client.normalize_server_config(row["patch"], errors=errors)
    assert cfg is not None and not errors and not cfg.task_session and not cfg.browser_policy


def test_an_update_changes_only_the_named_flags_and_leaves_no_pair_the_loader_refuses():
    def update(**flags):
        return entries({"mcpServers": {"files": {"command": "uvx", **flags}}}, DRAFT)["files"]

    assert update()["patch"] == {"transport": "stdio", "command": "uvx"}
    assert update(task_session=True)["patch"] == {"transport": "stdio", "command": "uvx", "task_session": True}
    assert update(browser_policy=False)["patch"] == {"transport": "stdio", "command": "uvx", "browser_policy": False}
    # The preview does not see the draft's flags: a lone true policy is refused even over a saved session, and a
    # session turned off takes the policy with it in the same patch (as a file's headers take the legacy token).
    assert "browser_policy requires task_session" in update(browser_policy=True)["problems"][0]
    assert update(task_session=False)["patch"] == {"transport": "stdio", "command": "uvx", "task_session": False,
                                                   "browser_policy": False}
    for named in ({}, {"task_session": True}, {"task_session": False}, {"browser_policy": False},
                  {"task_session": True, "browser_policy": True}, {"task_session": True, "browser_policy": False},
                  {"task_session": False, "browser_policy": False}):
        patch = update(**named)["patch"]
        for saved in ((False, False), (True, False), (True, True)):
            draft = {"id": "files", "transport": "stdio", "command": "uv", "task_session": saved[0], "browser_policy": saved[1]}
            errors = []
            assert mcp_client.normalize_server_config({**draft, **patch}, errors=errors) is not None, (named, saved, errors)


def test_canonical_identity_collisions_are_ambiguous_on_both_sides():
    found = entries({"mcpServers": {"GitHub": {"url": "https://a.example"}, "github": {"url": "https://b.example"},
                                    "!!!": {"url": "https://c.example"}}})
    for name in ("GitHub", "github"):
        assert "several imported names" in found[name]["problems"][0] and not found[name]["action"]
    assert "no usable server ID" in found["!!!"]["problems"][0]
    twice = [{"id": "docs"}, {"name": "Docs"}]
    entry = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp"}}}, twice)["docs"]
    assert "several draft servers" in entry["problems"][0] and not entry["action"]


def test_the_route_is_no_store_and_refuses_a_malformed_request():
    from ouroboros.gateway import mcp

    app = Starlette(routes=[Route("/preview", mcp.api_mcp_import_preview, methods=["POST"])])
    with TestClient(app) as client:
        document = {"mcpServers": {"docs": {"url": "https://docs.example/mcp", "headers": {"X-Api-Key": CUSTOM}}}}
        response = client.post("/preview", json={"text": json.dumps(document), "servers": [*DRAFT, None]})
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert response.json()["entries"][0]["action"] == "update"
        for body, reason in (({"text": 1, "servers": []}, "text must be a JSON string"),
                             ({"text": "{}", "servers": {}}, "servers must be a JSON array"),
                             (["text"], "payload must be a JSON object"),
                             ({"text": "{}", "servers": [{"id": "a", "header_names": "X-A"}]},
                              "servers[0] matches no allowed shape"),
                             ({"servers": []}, "text is required")):
            refused = client.post("/preview", json=body)
            assert refused.status_code == 400 and refused.headers["cache-control"] == "no-store"
            assert reason in refused.json()["error"], (body, refused.json())


@pytest.mark.parametrize("text", [
    '{"mcpServers":{"a":{"headers":{"synthetic-private-key":"one","synthetic-private-key":"two"}}}}',
    '{"mcpServers": synthetic-private-key}',
    '{"mcpServers":{"a":{"command":"npx","task_session":"synthetic-private-key"}}}',
])
def test_parse_and_validation_errors_do_not_echo_input_values(text, caplog):
    result = preview(text)
    assert "synthetic-private-key" not in json.dumps(result)
    assert "synthetic-private-key" not in caplog.text


def test_preview_neither_contacts_services_nor_reads_settings_or_saves(monkeypatch):
    from ouroboros import config
    from ouroboros.gateway import mcp

    def unexpected(*args, **kwargs):
        raise AssertionError("Preview crossed an I/O boundary")

    monkeypatch.setattr(mcp, "_ensure_configured", unexpected)
    monkeypatch.setattr(mcp, "get_manager", unexpected)
    monkeypatch.setattr(mcp, "load_settings", unexpected)
    monkeypatch.setattr(config, "load_settings", unexpected)
    monkeypatch.setattr(mcp_client, "_transport_factory", unexpected)
    app = Starlette(routes=[Route("/preview", mcp.api_mcp_import_preview, methods=["POST"])])
    with TestClient(app) as client:
        response = client.post("/preview", json={"text": json.dumps({"mcpServers": {
            "a": {"url": "https://example.test/mcp", "headers": {"X-Key": CUSTOM}},
            "b": {"command": "npx", "args": ["--sample"], "env": {"PORT": "8080"},
                  "task_session": True, "browser_policy": True},
        }}), "servers": [None, {"id": "unrelated", "headers": ["legacy"]}]})
    assert response.status_code == 200 and response.json()["ok"]
    assert all(row["action"] == "add" for row in response.json()["entries"])


def test_literals_in_url_and_command_stay_only_in_the_patch():
    found = entries({"mcpServers": {
        "a": {"url": "https://example.test/" + CUSTOM + "?token=" + CUSTOM},
        "b": {"command": "synthetic-command-" + CUSTOM},
    }})
    assert all(CUSTOM not in displayed(row) for row in found.values())
    assert CUSTOM in found["a"]["patch"]["url"] and CUSTOM in found["b"]["patch"]["command"]


def test_vs_code_servers_wrapper_reads_the_same_entries():
    result = preview({"servers": {"Docs": {"type": "http", "url": "https://docs.example/mcp",
                                           "headers": {"X-Api-Key": CUSTOM}}},
                      "inputs": [{"id": "key", "type": "promptString", "password": True}]})
    assert result["ok"] and result["wrapper"] == "servers" and result["ignored_keys"] == ["inputs"]
    assert result["entries"][0]["action"] == "add" and result["entries"][0]["patch"]["headers"] == {"X-Api-Key": CUSTOM}


def test_substitution_text_is_kept_literally_with_a_warning_that_names_fields_not_values():
    entry = entries({"mcpServers": {
        "a": {"url": "https://example.test/${env:PATH_PART}", "headers": {"Authorization": "Bearer ${API_KEY}",
                                                                          "X-Plain": CUSTOM}},
        "b": {"command": "npx", "args": ["--token", "${input:token}"], "env": {"TOKEN": "${TOKEN:-x}", "PORT": "1"}},
    }})
    a, b = entry["a"], entry["b"]
    assert a["patch"]["headers"]["Authorization"] == "Bearer ${API_KEY}" and a["action"] == "add"
    assert a["warnings"][0].startswith("url, header Authorization: ${...} is kept as written")
    assert b["warnings"][0].startswith("args, environment TOKEN: ${...} is kept as written")
    for row in (a, b):
        assert "API_KEY" not in displayed(row) and "input:token" not in displayed(row) and CUSTOM not in displayed(row)


def test_the_settings_keep_marker_is_not_imported_as_a_header_value():
    entry = entries({"mcpServers": {"docs": {"url": "https://docs.example/mcp",
                                             "headers": {"X-Api-Key": "***set***", "X-Other": CUSTOM}}}}, DRAFT)["docs"]
    assert not entry["action"] and entry["patch"] == {}
    assert "header X-Api-Key has the value ***set***" in entry["problems"][0] and CUSTOM not in displayed(entry)


def test_matching_reads_only_identity_transport_and_header_names():
    # A full draft object with values gives the same answer as its metadata.
    full = [{"id": "docs", "url": "https://secret.example/?k=" + CUSTOM, "auth_header": "Authorization",
             "auth_token": "Bearer t...", "headers": {"X-Tenant": "***set***"}, "transport": "streamable_http",
             "header_names": ["Authorization", "X-Tenant"], "legacy_header": "Authorization"}]
    document = {"mcpServers": {"docs": {"url": "https://docs.example/mcp", "headers": {"X-Api-Key": "k"}}}}
    assert preview(document, full) == preview(document, DRAFT[:1])


def test_preview_contract_twin_and_the_producer_agree():
    import typing

    from ouroboros.gateway import mcp_import_contracts as contracts
    from tests.test_gateway_parity import _js_typedef_fields

    twin = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "mcp_import_types.js").read_text(
        encoding="utf-8")
    for name in contracts.__all__:
        cls = getattr(contracts, name)
        hints = typing.get_type_hints(cls, include_extras=True)
        assert _js_typedef_fields(twin, name) == set(hints), name
        optional = {field for field, hint in hints.items() if typing.get_origin(hint) is contracts.NotRequired}
        if not cls.__total__:
            optional = set(hints)
        declared = {line.split("}")[-1].split()[0] for line in twin.split(f"@typedef {{Object}} {name}\n")[1]
                    .split("\n */")[0].splitlines() if "@property {" in line and "=}" in line}
        assert declared == optional, name
    result = preview({"mcpServers": {"docs": {"url": "https://docs.example/mcp", "headers": {"X-A": "1"}},
                                     "files": {"command": "uvx"}, "bad": 7}}, DRAFT)
    entry_hints = typing.get_type_hints(contracts.MCPImportPreviewEntry, include_extras=True)
    entry_fields = set(entry_hints)
    # Postponed annotations hide NotRequired from __optional_keys__; the resolved hints keep it.
    required = {field for field, hint in entry_hints.items() if typing.get_origin(hint) is not contracts.NotRequired}
    for entry in result["entries"]:
        assert required <= set(entry) <= entry_fields, sorted(set(entry) ^ entry_fields)
    assert set(result) <= set(typing.get_type_hints(contracts.MCPImportPreviewResponse))
    api_types = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js").read_text(
        encoding="utf-8")
    assert "mcp_import_types.js" in api_types.splitlines()[0]
    from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS
    assert "POST /api/mcp/import/preview" in HTTP_ENDPOINTS
    # The gateway's derived ingress schema reads the twin like any contract TypedDict.
    from ouroboros.gateway.schema import json_schema_for, validate_ingress
    schema = json_schema_for(contracts.MCPImportPreviewRequest)
    assert schema["required"] == ["servers", "text"]
    item = schema["properties"]["servers"]["items"]
    assert {"type": "null"} in item["anyOf"] and any(option.get("properties", {}).get("legacy_header")
                                                    for option in item["anyOf"])
    assert validate_ingress({"text": "{}", "servers": [None, {"id": "a", "header_names": ["X"],
                                                              "legacy_header": None}]},
                            contracts.MCPImportPreviewRequest) == []


# The real Settings › Tools page (mcp_settings.js, mounted the way the shell mounts it) over the repo's DOM double,
# which parses nothing: this mounts the page's own markup, so a renamed control fails here. It stops short of an
# engine, layout and CSS; tests/test_settings_tools_models_browser.py is the rendered lane.
IMPORT_JOURNEY = r'''
import assert from 'node:assert/strict';
import { installDomDouble } from './web/tests/dom_double.js';
import { collectMcpSettings } from './web/modules/mcp_settings.js';
import { readSettings } from './web/modules/settings_controls.js';
import { settingsPageFor } from './web/modules/settings_pages.js';

const [base, documentText] = process.argv.slice(1);
const { doc } = installDomDouble();
doc.querySelector = (selector) => doc.body.querySelector(selector);
doc.querySelectorAll = (selector) => doc.body.querySelectorAll(selector);
const ENTITIES = { '&quot;': '"', '&amp;': '&', '&lt;': '<', '&gt;': '>', '&#39;': "'", '&#10;': '\n' };
const decode = (text) => text.replace(/&(?:quot|amp|lt|gt|#39|#10);/g, (entity) => ENTITIES[entity]);
const ATTRIBUTE = /([^\s=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g;
const TAG = /<!--[\s\S]*?-->|<\/([\w-]+)\s*>|<([\w-]+)((?:\s+[^\s=>]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?)*)\s*(\/?)>/g;
function mount(into, html) {
    const open = [into];
    let at = 0;
    for (const [whole, closing, tag, attributes, selfClosing] of html.matchAll(TAG)) {
        const index = html.indexOf(whole, at);
        open.at(-1)._text += decode(html.slice(at, index));
        at = index + whole.length;
        if (closing) {
            const node = open.pop();
            assert.equal(node.tagName.toLowerCase(), closing.toLowerCase(), 'the markup nests');
            if (node.tagName === 'TEXTAREA') node.value = node._text;
        } else if (tag) {
            const node = doc.createElement(tag);
            for (const [, key, double, single, bare] of attributes.matchAll(ATTRIBUTE)) {
                const value = decode(double ?? single ?? bare ?? '');
                if (key === 'class') node.className = value;
                else if (key.startsWith('data-')) node.dataset[key.slice(5).replace(/-([a-z])/g, (_all, c) => c.toUpperCase())] = value;
                else if (['hidden', 'disabled', 'checked', 'selected'].includes(key)) node[key] = true;
                else { node.setAttribute(key, value); if (key === 'value') node.value = value; }
            }
            open.at(-1).appendChild(node);
            if (!selfClosing && !['input', 'br', 'hr', 'img'].includes(tag.toLowerCase())) open.push(node);
        }
    }
    open.at(-1)._text += decode(html.slice(at));
    assert.equal(open.length, 1, 'the markup closes');
}
const nodes = Object.getPrototypeOf(doc.body);
Object.defineProperty(nodes, 'innerHTML', {
    get() { return this.html || ''; },
    set(value) { this.replaceChildren(); this.html = String(value); mount(this, this.html); },
});
// The double's grammar has no #id, and reads one as a compound that matches every node.
const matches = nodes.matches;
nodes.matches = function matchesWithIds(selector, scope) {
    return matches.call(this, selector.replace(/#([\w-]+)/g, '[id="$1"]'), scope);
};
const page = doc.createElement('div');
doc.body.appendChild(page);

const previews = [];
const saves = [];  // every write of the settings the page made: its body and the server's answer
const wire = globalThis.fetch;
globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/api/mcp/import/preview')) previews.push(JSON.parse(init.body));
    const answer = await wire(new URL(url, base), init);
    if (String(url).endsWith('/api/settings') && init?.method === 'POST') saves.push({ sent: JSON.parse(init.body), status: answer.status, body: await answer.clone().json() });
    return answer;
};
const settle = async (ready) => {
    const deadline = Date.now() + 20_000;
    while (!ready()) {
        assert.ok(Date.now() < deadline, 'the module settled');
        await new Promise((resolve) => setTimeout(resolve, 10));
    }
};

settingsPageFor('tools').mount(page, { path: 'tools', row: '', ctx: { readSettings, openSettings: () => true, state: { activePage: 'settings', settingsPath: 'tools' } } });
await settle(() => page.querySelector('[data-tools-list]').children.length > 0);
page.querySelector('[data-tools-action="tools:import"]').click();
const dialog = doc.body.querySelector('.mcp-import-dialog');
const field = dialog.querySelector('#mcp-import-text');
field.value = documentText;
field.emit('input');
dialog.querySelector('[data-mcp-import-preview-btn]').click();
const status = dialog.querySelector('[data-mcp-import-status]');
await settle(() => status.dataset.tone && status.dataset.tone !== 'muted');
const apply = dialog.querySelector('[data-mcp-import-apply]');
const facts = { preview_request: previews[0], status: status.textContent, apply_label: apply.textContent, apply_disabled: apply.disabled,
    entries: dialog.querySelectorAll('.mcp-import-entry').map((entry) => entry.textContent) };
if (!apply.disabled) {
    // Apply saves the chosen servers: one write of the list, then the list shows the new head, off.
    apply.click();
    await settle(() => !doc.body.querySelector('.mcp-import-dialog') && page.querySelector('[data-entry-id] [data-entry-switch]'));
    const use = () => page.querySelector('[data-entry-id] [data-entry-switch]');
    facts.enabled_after_apply = Boolean(use().checked);
    facts.draft = structuredClone(collectMcpSettings().MCP_SERVERS);
    facts.import_write = Object.keys(saves[0].sent).sort();
    // The owner's two switches: the tool client's, then this server's. Each writes at once, its own key alone.
    const client = page.querySelector('[data-setting-row="MCP_ENABLED"] [data-setting-control]');
    client.checked = true;
    client.emit('change');
    await settle(() => saves.length === 2);
    const head = use();
    head.checked = true;
    head.emit('change');
    await settle(() => saves.length === 3 && use() && !use().disabled && use().checked);
    facts.switch_writes = saves.slice(1).map((save) => Object.keys(save.sent).filter((key) => key !== 'if_unchanged'));
    facts.guarded = saves.map((save) => Object.keys(save.sent.if_unchanged || {}));
    facts.save = { status: saves.at(-1).status, body: saves.at(-1).body };
    await readSettings();
    facts.reloaded = structuredClone(collectMcpSettings().MCP_SERVERS);
}
process.stdout.write(JSON.stringify(facts), () => process.exit(0));
'''


def import_journey(document, tmp_path, settings_client, monkeypatch):  # noqa: F811
    """Paste, Preview, Apply and the two switches on the real Settings › Tools page, against the real routes on a real port."""
    from ouroboros import config
    from ouroboros.gateway import mcp as mcp_api

    client, on_disk, _saved = settings_client
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mcp_api, "load_settings", lambda: copy.deepcopy(on_disk))
    app = Starlette(routes=[*client.app.routes,
                            Route("/api/mcp/import/preview", mcp_api.api_mcp_import_preview, methods=["POST"]),
                            Route("/api/mcp/status", mcp_api.api_mcp_status, methods=["GET"])])
    app.state.drive_root, app.state.repo_dir = client.app.state.drive_root, client.app.state.repo_dir
    running = []
    mcp_client.reset_manager_for_tests()
    try:
        port = start_asgi(app, running, "mcp-import-journey")
        # The journey writes UTF-8 JSON (the Settings copy says "task’s"); decode it as UTF-8, not with the locale's codec.
        answer = subprocess.run([_node_bin(), "--input-type=module", "-e", IMPORT_JOURNEY, f"http://127.0.0.1:{port}", document],
                                cwd=pathlib.Path(__file__).resolve().parents[1], capture_output=True, text=True,
                                encoding="utf-8", timeout=60)
        assert answer.returncode == 0, answer.stderr
        return json.loads(answer.stdout), on_disk
    finally:
        for server, thread in running:
            server.should_exit = True
            thread.join(10)


def listed_status(server_id, timeout=30.0):
    """The Save's own background pass, as the status route reports it once it has listed."""
    manager, deadline = mcp_client.get_manager(), time.monotonic() + timeout
    while True:
        status = {row["id"]: row for row in manager.status_payload()["servers"]}.get(server_id) or {}
        if status.get("listed_at") or status.get("last_error"):
            return status
        assert time.monotonic() < deadline, "the Save's listing pass did not finish"
        time.sleep(0.05)


def test_setup_document_becomes_a_listed_held_server_through_import_enable_and_save(tmp_path, settings_client, monkeypatch):  # noqa: F811
    script = tmp_path / "inert_server.py"
    script.write_text(SERVER, encoding="utf-8")
    log = tmp_path / "inert.jsonl"
    command, args = sys.executable, [str(script), str(log)]
    document = json.dumps({"mcpServers": {"chrome_browser": {"type": "stdio", "command": command, "args": args,
                                                             "task_session": True, "browser_policy": True}}}, indent=2)
    try:
        facts, on_disk = import_journey(document, tmp_path, settings_client, monkeypatch)
        assert facts["preview_request"] == {"text": document, "servers": []}  # the empty editor's own request
        assert facts["status"] == "1 of 1 server can be applied." and facts["apply_label"] == "Apply 1"
        shown = facts["entries"][0]
        for fact in ("chrome_browser", "Add (off)", "Server ID chrome_browser", "Local process (stdio)", "2 arguments",
                     "task_session: a task’s calls share one running server process",
                     "browser_policy: that process gets a private local address and token"):
            assert fact in shown, (fact, shown)
        assert "Not imported" not in shown and command not in shown and str(script) not in shown
        assert facts["enabled_after_apply"] is False  # a new server starts off: its switch is the owner's step
        draft = facts["draft"][0]
        assert (draft["enabled"], draft["task_session"], draft["browser_policy"]) == (False, True, True)
        # Apply wrote the list; each switch then wrote its own key alone. A list that has a saved value is written
        # with the token of the read it changed (the first write of a list never saved has none to carry).
        assert [key for key in facts["import_write"] if key != "if_unchanged"] == ["MCP_SERVERS"]
        assert facts["switch_writes"] == [["MCP_ENABLED"], ["MCP_SERVERS"]]
        assert facts["guarded"][1] == [] and facts["guarded"][2] == ["MCP_SERVERS"], facts["guarded"]
        assert facts["save"]["status"] == 200 and facts["save"]["body"]["status"] == "saved", facts["save"]

        row = on_disk["MCP_SERVERS"][0]
        assert (row["id"], row["name"], row["enabled"], row["transport"], row["command"], row["args"]) == (
            "chrome_browser", "chrome_browser", True, "stdio", command, args)
        assert (row["task_session"], row["browser_policy"]) == (True, True) and on_disk["MCP_ENABLED"] is True
        reloaded = facts["reloaded"][0]
        assert (reloaded["task_session"], reloaded["browser_policy"]) == (True, True)  # the served row saves back whole

        errors = []
        cfg = mcp_client.parse_servers(on_disk["MCP_SERVERS"], settings=on_disk, errors=errors)[0]
        assert not errors and not cfg.configuration_warnings
        assert (cfg.id, cfg.enabled, cfg.task_session, cfg.browser_policy) == ("chrome_browser", True, True, True)

        status = listed_status("chrome_browser")
        assert status["listed_at"] and not status["last_error"] and status["tool_count"]
        assert (status["task_session"], status["browser_policy"]) == (True, True)
        names = {tool["name"] for tool in mcp_client.get_manager().list_tools_for_registry()}
        assert "mcp_chrome_browser__bump" in names
        # Listing is passive whatever the flags: its own process, no task scope, no guard, no call.
        assert rows(log, "list") and not rows(log, "call")
        assert all(start["guard"] == "" and start["token"] is False and start["task_scope"] is False for start in rows(log, "start"))
    finally:
        mcp_client.reset_manager_for_tests()


def test_a_refused_flag_pair_cannot_be_applied_and_nothing_reaches_settings(tmp_path, settings_client, monkeypatch):  # noqa: F811
    document = json.dumps({"mcpServers": {"chrome_browser": {"command": "node", "browser_policy": True}}})
    facts, on_disk = import_journey(document, tmp_path, settings_client, monkeypatch)
    assert facts["apply_disabled"] is True and "save" not in facts and "MCP_SERVERS" not in on_disk
    assert "Not applied" in facts["entries"][0] and "Browser_policy requires task_session" in facts["entries"][0]
