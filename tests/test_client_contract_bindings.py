"""The shapes ``GET /api/contract`` binds to each route, checked against the real handlers.

Every request below goes through the real router (``collect_routes``) on a disposable data root.
Each answer must use a status its contract entry lists, and its body must match one of the
shapes bound to that status. A success body must also carry no key its shape does not declare,
so a shape can never claim to be complete while the producer says more. Every published shape
has exactly one field-identical JSDoc twin among the browser modules.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import threading
import typing

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.gateway.router import collect_routes
from ouroboros.gateway.schema import CONTRACT_MODULES, contract_document, contract_typed_dict, validate_ingress
from tests.test_gateway_parity import _js_typedef_fields

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCUMENT = contract_document()
BY_TOKEN = {f"{entry['method']} {entry['path']}": entry for entry in DOCUMENT["http"]}


def _declared(name: str) -> set:
    return set(typing.get_type_hints(contract_typed_dict(name), include_extras=True))


def assert_bound(token: str, response, *, closed: bool = True) -> str:
    """The shape name the answer matched; fails when the status or the body is not the contract's."""
    entry = BY_TOKEN[token]
    status = str(response.status_code)
    assert status in entry["responses"], f"{token} answered {status}; its contract lists {sorted(entry['responses'])}"
    names = entry["responses"][status]["schemas"]
    if not names:
        return entry["responses"][status]["content_type"]
    payload = response.json()
    failures = []
    for name in names:
        errors = validate_ingress(payload, contract_typed_dict(name))
        extra = (set(payload) - _declared(name)) if closed and status.startswith("2") else set()
        if not errors and not extra:
            return name
        failures.append((name, errors[:4], sorted(extra)))
    raise AssertionError(f"{token} {status} matches none of its shapes: {failures}\n{payload}")


@pytest.fixture
def client(tmp_path, monkeypatch):
    import ouroboros.config as cfg
    from ouroboros.gateway import ui_i18n

    files = tmp_path / "files"
    files.mkdir()
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(files))
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(ui_i18n, "_LANGUAGE_HOOKS", [])
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    app.state.repo_dir = REPO
    with TestClient(app) as test_client:
        test_client.files_root = files
        yield test_client


def test_the_file_browser_journey_answers_its_published_shapes(client):
    (client.files_root / "notes.txt").write_text("hello", encoding="utf-8")
    assert assert_bound("GET /api/files/list", client.get("/api/files/list")) == "FileBrowserListResponse"
    assert assert_bound("GET /api/files/read", client.get("/api/files/read?path=notes.txt")) == "FileBrowserReadResponse"
    assert assert_bound("POST /api/files/mkdir", client.post("/api/files/mkdir", json={"name": "docs"})) == "FileChangeResponse"
    assert_bound("POST /api/files/mkdir", client.post("/api/files/mkdir", json={"name": "docs"}))  # 409: it exists
    written = client.post("/api/files/write", json={"path": "docs/a.txt", "content": "x", "create": True})
    assert assert_bound("POST /api/files/write", written) == "FileChangeResponse" and written.json()["created"] is True
    uploaded = client.post("/api/files/upload", data={"path": "docs"}, files={"file": ("b.txt", b"bytes")})
    assert assert_bound("POST /api/files/upload", uploaded) == "FileChangeResponse"
    copied = client.post("/api/files/transfer", json={"source_path": "notes.txt", "destination_dir": "docs"})
    assert assert_bound("POST /api/files/transfer", copied) == "FileChangeResponse" and copied.json()["mode"] == "copy"
    assert copied.json()["path"] == "docs/notes.txt"
    assert client.post("/api/files/transfer", json={"source_path": "notes.txt", "destination_dir": "docs"}).status_code == 409
    deleted = client.post("/api/files/delete", json={"path": "docs/a.txt"})
    assert assert_bound("POST /api/files/delete", deleted) == "FileChangeResponse"
    again = client.post("/api/files/delete", json={"path": "docs/a.txt"})
    assert again.status_code == 404 and assert_bound("POST /api/files/delete", again) == "ErrorResponse"
    escape = client.get("/api/files/read?path=../outside")
    assert escape.status_code == 400 and assert_bound("GET /api/files/read", escape) == "ErrorResponse"


def test_the_project_journey_answers_its_published_shapes(client):
    created = client.post("/api/projects", json={"name": "Trip planning"})
    assert assert_bound("POST /api/projects", created) == "ProjectCreateResponse"
    project = created.json()["project"]
    assert project.get("created") is not False
    repeat = client.post("/api/projects", json={"id": project["id"]})
    assert assert_bound("POST /api/projects", repeat) == "ProjectCreateResponse" and repeat.json()["project"]["created"] is False
    listed = client.get("/api/projects")
    assert assert_bound("GET /api/projects", listed) == "ProjectListResponse"
    assert [row["id"] for row in listed.json()["projects"]] == [project["id"]]
    renamed = client.post(f"/api/projects/{project['id']}/update", json={"name": "Trip"})
    assert assert_bound("POST /api/projects/{project_id}/update", renamed) == "ProjectUpdateResponse"
    missing = client.post("/api/projects/nope/update", json={"name": "x"})
    assert missing.status_code == 404 and assert_bound("POST /api/projects/{project_id}/update", missing)
    assert assert_bound("POST /api/projects/from-task", client.post("/api/projects/from-task", json={})) == "ErrorResponse"
    deleted = client.post(f"/api/projects/{project['id']}/delete", json={})
    assert assert_bound("POST /api/projects/{project_id}/delete", deleted) == "ProjectDeleteResponse"
    # The answer precedes the Project's cancellation worker; the journey that started it joins it.
    for worker in [thread for thread in threading.enumerate() if thread.name == f"project-delete-{project['id']}"]:
        worker.join(timeout=60)
        assert not worker.is_alive(), "the Project deletion worker never finished"


def test_settings_language_and_desktop_routes_answer_their_published_shapes(client, monkeypatch):
    from ouroboros import i18n_memory as memory

    secret = client.post("/api/settings/secret", json={"key": "OPENROUTER_API_KEY"})
    assert assert_bound("POST /api/settings/secret", secret) == "SettingsSecretResponse"
    assert secret.headers["cache-control"] == "no-store"
    assert client.post("/api/settings/secret", json={"key": "A", "mcp_server_id": "b"}).status_code == 400
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    root = client.app.state.drive_root
    memory.update_memory(root, "ru", lambda doc: (memory.apply_generated(doc, {"Files": {"text": "Файлы"}}, model="m"), doc)[1],
                         create=True)
    imported = client.post("/api/ui/i18n/import", json={"schema": 1, "language": "ru", "entries": {"Skills": {"text": "Навыки"}}})
    assert assert_bound("POST /api/ui/i18n/import", imported) == "UiI18nImportResponse"
    refused = client.post("/api/ui/i18n/import", json={"language": "en", "entries": {}})
    assert refused.status_code == 400 and refused.json()["code"] == "memory_invalid"
    exported = client.get("/api/ui/i18n/export")
    assert exported.status_code == 200 and not validate_ingress(exported.json(), contract_typed_dict("UiI18nMemoryDocument"))
    assert assert_bound("GET /api/ui/i18n/export", client.get("/api/ui/i18n/export?language=en")) == "ErrorResponse"
    regenerated = client.post("/api/ui/i18n/regenerate", json={})
    assert assert_bound("POST /api/ui/i18n/regenerate", regenerated) == "UiI18nRegenerateResponse"
    assert assert_bound("GET /api/desktop/autostart", client.get("/api/desktop/autostart")) == "DesktopAutostartResponse"
    assert assert_bound("GET /api/desktop/background", client.get("/api/desktop/background")) == "DesktopAutostartResponse"
    bad = client.post("/api/desktop/autostart", json={"enabled": "yes"})
    assert bad.status_code == 400 and assert_bound("POST /api/desktop/autostart", bad)
    ack = client.post("/api/owner/capability-ack", json={"provider": "openrouter"})
    assert ack.status_code == 400 and assert_bound("POST /api/owner/capability-ack", ack)


def test_settings_ignored_keys_are_an_optional_writer_map(client):
    response = client.post('/api/settings', json={'OUROBOROS_UI_LANGUAGE': 'en'})
    assert assert_bound('POST /api/settings', response) == 'SettingsSaveResponse'
    ignored = response.json()['ignored_keys']
    assert set(ignored) == {'OUROBOROS_UI_LANGUAGE'}
    assert isinstance(ignored['OUROBOROS_UI_LANGUAGE'], str)
    assert ignored['OUROBOROS_UI_LANGUAGE']


def test_skill_and_marketplace_reads_answer_their_published_shapes(client):
    assert assert_bound("GET /api/extensions", client.get("/api/extensions")) == "ExtensionsIndexResponse"
    assert assert_bound("GET /api/skills/lifecycle-queue", client.get("/api/skills/lifecycle-queue")) == "SkillLifecycleQueueResponse"
    assert assert_bound("GET /api/skills/daemons", client.get("/api/skills/daemons")) == "SkillDaemonsResponse"
    section = client.get("/api/extensions/nothing/settings_section")
    assert assert_bound("GET /api/extensions/{skill}/settings_section", section) == "ExtensionSettingsSectionResponse"
    manifest = client.get("/api/extensions/nothing/manifest")
    assert manifest.status_code == 404 and assert_bound("GET /api/extensions/{skill}/manifest", manifest)
    toggle = client.post("/api/skills/nothing/toggle", json={"enabled": True})
    assert toggle.status_code == 404 and assert_bound("POST /api/skills/{skill}/toggle", toggle) == "SkillActionRefusal"
    bad_toggle = client.post("/api/skills/nothing/toggle", json={"enabled": "maybe"})
    assert bad_toggle.status_code == 400 and assert_bound("POST /api/skills/{skill}/toggle", bad_toggle)
    history = client.get("/api/skills/nothing/review-history/job-1")
    assert history.status_code == 404 and assert_bound("GET /api/skills/{skill}/review-history/{job_id}", history)
    token = "GET /api/marketplace/clawhub/installed"
    assert assert_bound(token, client.get("/api/marketplace/clawhub/installed")) == "MarketplaceInstalledResponse"
    missing = client.post("/api/marketplace/clawhub/install", json={})
    assert missing.status_code == 400 and assert_bound("POST /api/marketplace/clawhub/install", missing)


def test_account_routes_refuse_malformed_input_in_their_published_shapes(client):
    status = client.get("/api/claudexor/status")
    assert assert_bound("GET /api/claudexor/status", status) == "ClaudexorStatusResponse"
    assert set(status.json()["reads"]) == {"catalog", "accounts", "quota"}
    quota = client.get("/api/claudexor/status?view=quota")
    assert assert_bound("GET /api/claudexor/status", quota) == "ClaudexorQuotaResponse"
    refresh = client.post("/api/claudexor/quota/refresh", json={"target": {"harness": ""}})
    assert refresh.status_code == 400 and assert_bound("POST /api/claudexor/quota/refresh", refresh) == "ErrorResponse"
    keyless = client.post("/api/claudexor/account-resets",
                          json={"target": {"harness": "codex", "profile_id": "p"}, "offer_id": "o"})
    assert keyless.status_code == 400 and "Idempotency-Key" in keyless.json()["error"]
    maintenance = client.post("/api/claudexor/maintenance/operations", json={"harness": "codex"})
    assert maintenance.status_code == 400
    assert assert_bound("POST /api/claudexor/maintenance/operations", maintenance) == "ClaudexorMaintenanceProblem"
    login = client.post("/api/claudexor/login", json={})
    assert login.status_code == 400 and assert_bound("POST /api/claudexor/login", login)
    answer = client.post("/api/claudexor/login/job-1/input", json={"value": ""})
    assert answer.status_code == 400 and assert_bound("POST /api/claudexor/login/{job_id}/input", answer)
    enabled = client.patch("/api/claudexor/credential-profiles/codex/p", json={"enabled": "yes"})
    assert enabled.status_code == 400 and assert_bound("PATCH /api/claudexor/credential-profiles/{harness}/{profile_id}", enabled)


def test_model_mcp_and_control_routes_answer_their_published_shapes(client, monkeypatch):
    from ouroboros import local_model

    manager = local_model.get_manager()
    monkeypatch.setattr(manager, "check_runtime", lambda: False)
    local = client.get("/api/local-model/status")
    assert assert_bound("GET /api/local-model/status", local) == "LocalModelStatusResponse"
    not_running = client.post("/api/local-model/test")
    assert not_running.status_code == 400 and assert_bound("POST /api/local-model/test", not_running)
    assert client.post("/api/local-model/start", json={}).status_code == 400
    assert assert_bound("GET /api/mcp/status", client.get("/api/mcp/status")) == "McpStatusResponse"
    assert assert_bound("POST /api/mcp/refresh", client.post("/api/mcp/refresh", json={})) == "McpRefreshResponse"
    models = client.post("/api/openai-compatible/models", json={})
    assert models.status_code == 400 and assert_bound("POST /api/openai-compatible/models", models)
    catalog = client.get("/api/model-catalog?credential_profile_id=p")
    assert catalog.status_code == 400 and assert_bound("GET /api/model-catalog", catalog)
    queued = client.post("/api/command", json={"cmd": ""})
    assert assert_bound("POST /api/command", queued) == "CommandResponse" and queued.json() == {"status": "ok"}
    resumed = client.post("/api/tasks/nothing-here/resume")
    assert resumed.status_code in {404, 409} and assert_bound("POST /api/tasks/{task_id}/resume", resumed)
    rollback = client.post("/api/git/rollback", json={})
    assert rollback.status_code == 400 and assert_bound("POST /api/git/rollback", rollback)
    preflight = client.post("/api/skills/nothing/publish-preflight", json={})
    assert assert_bound("POST /api/skills/{skill}/publish-preflight", preflight) == "SkillPublishPreflightResponse"
    assert preflight.json()["state"] == "hard_block" and preflight.json()["ok"] is False


def test_the_update_status_review_pool_and_spend_answer_their_published_shapes(client, monkeypatch):
    # A row's price is measured from the process data root's usage ledger, which this disposable
    # root does not own; the shape does not need it, so every row reads as unpriced.
    monkeypatch.setattr("ouroboros.review_price.review_price", lambda *_a, **_kw: None)
    status = client.get("/api/update/status")
    assert assert_bound("GET /api/update/status", status) == "UpdateStatusResponse"
    assert status.json()["check_ok"] is None and status.json()["update_tx"]["active"] in {True, False}
    assert assert_bound("GET /api/review-pool", client.get("/api/review-pool")) == "ReviewPoolResponse"
    spend = client.get("/api/cost-breakdown")
    assert assert_bound("GET /api/cost-breakdown", spend) == "CostBreakdownResponse"
    assert spend.json()["accounting"]["authority"] == "physical_attempt_ledger"


# --- the browser mirror ----------------------------------------------------------------------

def _owned_names() -> dict:
    names = {name: contract_typed_dict(name) for name in DOCUMENT["schemas"]}
    for module_name in ("ouroboros.gateway.client_contracts", "ouroboros.gateway.management_contracts",
                        "ouroboros.gateway.history_contracts"):
        module = importlib.import_module(module_name)
        names.update({name: getattr(module, name) for name in module.__all__})
    return names


@pytest.mark.parametrize("outcome", ["unlisted", "listed", "failed", "unpublished"])
def test_mcp_listing_evidence_matches_published_contracts(client, monkeypatch, tmp_path, outcome):
    from ouroboros import config, mcp_catalog, mcp_client
    from ouroboros.gateway import mcp as api

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    value = {"MCP_ENABLED": True, "MCP_SERVERS": [{"id": "demo", "enabled": True, "url": "https://example.test/mcp"}]}
    monkeypatch.setattr(api, "load_settings", lambda: value)
    manager = mcp_client.MCPManager()
    monkeypatch.setattr(mcp_client, "_manager", manager)
    async def listing(cfg, timeout):
        if outcome == "failed":
            raise TimeoutError("offline")
        return [{"name": "ping", "description": "Ping", "input_schema": {}}]
    manager._async_list_tools = listing
    if outcome == "unpublished":
        def cannot_publish(*args):
            raise PermissionError("catalog is locked")
        monkeypatch.setattr(mcp_catalog, "publish", cannot_publish)
    if outcome != "unlisted":
        response = client.post("/api/mcp/refresh", json={"server_id": "demo"})
        assert assert_bound("POST /api/mcp/refresh", response) == "McpRefreshResponse"
        assert response.json()["published"] is (outcome != "unpublished")
        assert bool(response.json()["publication_error"]) is (outcome == "unpublished")
    response = client.get("/api/mcp/status")
    assert assert_bound("GET /api/mcp/status", response) == "McpStatusResponse"
    row = response.json()["servers"][0]
    assert set(row) <= _declared("McpServerStatus")
    attempt = row["latest_attempt"]
    if outcome == "unlisted":
        assert attempt is None and row["listed_at"] is None and not row["complete"]
    else:
        assert set(attempt) == _declared("McpListingAttempt")
        assert attempt["ok"] is (outcome != "failed")
        invalid = {**row, "latest_attempt": {**attempt, "partial": "not a boolean"}}
        assert validate_ingress(invalid, contract_typed_dict("McpServerStatus"))


def test_every_published_and_owned_shape_has_one_field_identical_browser_twin():
    from ouroboros.review import is_gated_js_module

    texts = {path.name: path.read_text(encoding="utf-8") for path in (REPO / "web" / "modules").glob("*.js")
             if is_gated_js_module(path.relative_to(REPO).as_posix())}
    assert len(DOCUMENT["schemas"]) > 150, "the contract lost its published shapes"
    for name, cls in sorted(_owned_names().items()):
        homes = [file for file, text in texts.items() if re.search(rf"@typedef \{{Object\}} {name}\b", text)]
        assert len(homes) == 1, f"{name} needs exactly one JSDoc twin, found {homes}"
        assert _js_typedef_fields(texts[homes[0]], name) == set(typing.get_type_hints(cls, include_extras=True)), name
    assert all(module in CONTRACT_MODULES for module in (
        "ouroboros.gateway.management_contracts", "ouroboros.gateway.history_contracts"))


def test_every_request_and_header_fact_is_published_for_the_journeys():
    requests = {token: entry.get("request", {}).get("schemas") for token, entry in BY_TOKEN.items()}
    for token in ("POST /api/files/write", "POST /api/projects/from-task", "POST /api/skills/{skill}/toggle",
                  "POST /api/claudexor/login", "POST /api/skill-catalog/install", "POST /api/command",
                  "POST /api/schedules", "POST /api/owner/context-mode", "DELETE /api/chat/upload",
                  "POST /api/places", "POST /api/places/{place_id}"):
        assert requests[token] and requests[token] != [], token
    assert BY_TOKEN["POST /api/claudexor/maintenance/operations"]["headers"] == ["Idempotency-Key!"]
    assert BY_TOKEN["GET /api/tasks/{task_id}/artifacts/{name}"]["headers"] == ["Range"]
    plain = [token for token, entry in BY_TOKEN.items()
             if entry.get("request", {}).get("content_type") == "application/json" and not entry["request"]["schemas"]]
    # The only bodies still published as plain JSON are documents of setting keys.
    assert sorted(plain) == ["POST /api/settings", "POST /api/ui/preferences"], plain


def test_place_request_shapes_cover_the_owner_journey(client, monkeypatch):
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [{
        "skill": "bridge", "id": "chat", "destination": "vendor:chat", "label": "Test chat", "wakes": ["mention"],
    }])
    connect = {"connector": "bridge", "account_id": "account", "conversation_id": "room"}
    assert BY_TOKEN["POST /api/places"]["request"]["schemas"] == ["PlaceConnectRequest"]
    assert _declared("PlaceConnectRequest") == set(connect)
    connect_type = contract_typed_dict("PlaceConnectRequest")
    assert not validate_ingress(connect, connect_type)
    assert not validate_ingress({"connector": "bridge"}, connect_type)
    assert validate_ingress({}, connect_type)  # connector is required
    connected = client.post("/api/places", json=connect)
    assert connected.status_code == 200, connected.text
    place = connected.json()["place"]
    path = f"/api/places/{place['place_id']}"
    change = {
        "state": "silent", "wakes": ["direct_message"], "add_note": "Owner choice",
        "attention_names": ["Vic", "Ouroboros"],
        "working_folder": str(client.files_root), "model": "test-model",
        "confirm_owner_account": {"platform_actor_id": "person", "label": "Me", "account_id": "account"},
        "expected_revision": place["revision"],
    }
    assert BY_TOKEN["POST /api/places/{place_id}"]["request"]["schemas"] == ["PlaceUpdateRequest"]
    update_type = contract_typed_dict("PlaceUpdateRequest")
    assert not validate_ingress(change, update_type)
    assert _declared("PlaceOwnerAccountRequest") == set(change["confirm_owner_account"])
    assert validate_ingress({"confirm_owner_account": {}}, update_type)
    changed = client.post(path, json=change)
    assert changed.status_code == 200, changed.text
    place = changed.json()["place"]
    assert place["state"] == "silent" and place["wakes"] == [{"fact": "direct_message", "author": "owner"}]
    assert place["attention_names"] == change["attention_names"] and place["attention_names_author"] == "owner"
    assert place["notes"][0]["text"] == "Owner choice" and place["notes"][0]["author"] == "owner"
    assert place["working_folder"] == change["working_folder"] and place["model"] == change["model"]
    assert place["owner_accounts"][0]["platform_actor_id"] == "person"
    remove = {"remove_note": place["notes"][0]["note_id"], "revoke_owner_account": "person",
              "attention_names": [],
              "expected_revision": place["revision"]}
    assert _declared("PlaceUpdateRequest") == set(change) | set(remove)
    assert not validate_ingress(remove, update_type)
    removed = client.post(path, json=remove)
    assert removed.status_code == 200, removed.text
    assert removed.json()["place"]["notes"] == removed.json()["place"]["owner_accounts"] == []
    assert removed.json()["place"]["attention_names"] == []
    conflict = client.post(path, json=change)
    assert conflict.status_code == 409 and conflict.json()["code"] == "place_revision_conflict"
