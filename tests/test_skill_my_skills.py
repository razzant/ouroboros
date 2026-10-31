"""Owner My skills creation/connection, contract and publication, with no GitHub traffic."""

import ast
import base64
import json
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import skill_publish_client, skill_publish_destination, skill_publish_github, skill_sources
from ouroboros.gateway import skill_my_skills
from ouroboros.gateway.router import collect_routes
from ouroboros.skill_publish_client import GitHubError
from ouroboros.tools import skill_publish
from tests.test_client_contract_bindings import assert_bound
from tests.test_skill_publish_transaction import BASE_SHA, COMMIT_SHA, _install_transaction_fakes, _snapshot, _submit

ENDPOINT = "/api/skill-sources/my-skills"
REPOSITORY = "alice/ouroboros-skills"
URL = "https://github.com/" + REPOSITORY
INITIAL = [{"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True}]


def test_my_skills_is_declared_in_the_endpoint_tables():
    from ouroboros.gateway import endpoint_index

    tree = ast.parse(Path(endpoint_index.__file__).read_text(encoding="utf-8"))
    tables = {node.target.id: node.value for node in tree.body if isinstance(node, ast.AnnAssign)}
    token = "POST " + ENDPOINT
    assert token in ast.literal_eval(tables["HTTP_ENDPOINTS"])
    assert token in [ast.literal_eval(key) for key in tables["HTTP_SHAPES"].keys]


@pytest.fixture
def env(tmp_path, monkeypatch):
    import ouroboros.config as config

    settings = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(INITIAL))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SYNTHETIC1234567890")
    monkeypatch.setattr(skill_my_skills, "github_request", lambda *_a, **_k: pytest.fail("unexpected GitHub request"))
    monkeypatch.setattr(skill_publish_github, "request", lambda *_a, **_k: pytest.fail("unexpected publication request"))
    monkeypatch.setattr("ouroboros.tools.github._gh_run", lambda *_a, **_k: pytest.fail("My skills used gh"))
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    app.state.repo_dir = tmp_path
    with TestClient(app) as client:
        yield client, settings


def _fake(monkeypatch, *, exists=False, private=True, failure=None, error_stage="create"):
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == "/user":
            if failure and error_stage == "user":
                raise failure
            return {"login": "alice"}
        if method == "GET" and path == f"/repos/{REPOSITORY}":
            if failure and error_stage == "read":
                raise failure
            if not exists:
                raise GitHubError("Not Found", http_status=404)
            return {"name": "ouroboros-skills", "private": private}
        assert (method, path) == ("POST", "/user/repos")
        if failure:
            raise failure
        return {"name": "ouroboros-skills", "private": private}

    monkeypatch.setattr(skill_my_skills, "github_request", request)
    return calls


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("private", [False, True])
def test_create_or_connect_repository_reports_visibility_and_saves_new_source(env, monkeypatch, exists, private):
    client, settings = env
    calls = _fake(monkeypatch, exists=exists, private=private)
    response = client.post(ENDPOINT, json={})
    assert response.status_code == 200, response.text
    assert assert_bound("POST " + ENDPOINT, response) == "MySkillsResponse"
    body = response.json()
    assert (body["ok"], body["created"], body["repository"]) == (True, not exists, REPOSITORY)
    assert body["private"] is private
    source = body["source"]
    assert source == {"id": "my-skills", "name": "My skills", "kind": "git", "url": URL,
                      "trust": "trusted", "confirmed": True}
    saved = json.loads(json.loads(settings.read_text(encoding="utf-8"))["OUROBOROS_SKILL_SOURCES"])
    assert saved == [*INITIAL, source] == skill_sources.load_sources()
    assert calls[:2] == [("GET", "/user", {}), ("GET", f"/repos/{REPOSITORY}", {})]
    if exists:
        assert len(calls) == 2
    else:
        assert calls[2] == ("POST", "/user/repos", {"timeout": 60, "body": {
            "name": "ouroboros-skills", "private": True, "auto_init": True,
            "description": "My Ouroboros skills, shared between my installations."}})
    audit = [json.loads(line) for line in (settings.parent / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert audit[-1]["repository"] == REPOSITORY and audit[-1]["change"] == "my_skills"


def test_repeat_connect_keeps_one_source_and_its_id(env, monkeypatch):
    client, _settings = env
    _fake(monkeypatch, exists=True)
    first = client.post(ENDPOINT, json={}).json()
    second = client.post(ENDPOINT, json={}).json()
    assert second == first and second["created"] is False
    assert [r["id"] for r in skill_sources.load_sources()] == ["builtin", "my-skills"]


def test_existing_source_keeps_owner_name_trust_and_confirmation(env, monkeypatch):
    client, _settings = env
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([*INITIAL, {
        "id": "existing", "name": "Other name", "url": URL, "kind": "git", "trust": "review", "confirmed": False}]))
    _fake(monkeypatch, exists=True)
    result = client.post(ENDPOINT, json={}).json()
    assert result["source"] == {"id": "existing", "name": "Other name", "url": URL, "kind": "git",
                                "trust": "review", "confirmed": False}
    assert skill_sources.source_by_id(skill_sources.load_sources(), "existing") == result["source"]


def test_missing_token_never_touches_github_or_sources(env, monkeypatch):
    client, settings = env
    monkeypatch.setattr(skill_publish_client, "github_token_from_env_or_settings", lambda: "")
    monkeypatch.setattr(skill_publish_client.httpx, "request", lambda *_a, **_k: pytest.fail("network without token"))
    monkeypatch.setattr(skill_my_skills, "github_request", skill_publish_client.request)
    response = client.post(ENDPOINT, json={})
    assert response.status_code == 400 and response.json() == {
        "ok": False, "code": "no_token", "error": "Add a GitHub token in Settings to publish"}
    assert assert_bound("POST " + ENDPOINT, response)
    assert not settings.exists() and skill_sources.load_sources() == INITIAL


@pytest.mark.parametrize("stage,status,code", [("create", 403, "cannot_create"), ("create", 404, "cannot_create"),
    ("create", 422, "failed"), ("create", 401, "failed"), ("read", 403, "failed"), ("user", 401, "failed"),
    ("create", None, "failed")])
def test_failures_have_typed_codes_and_never_save_a_source(env, monkeypatch, stage, status, code):
    client, settings = env
    error = GitHubError("synthetic refusal ghp_SYNTHETIC1234567890", http_status=status,
                        failure="timeout" if status is None else "http")
    calls = _fake(monkeypatch, failure=error, error_stage=stage)
    response = client.post(ENDPOINT, json={})
    assert response.status_code == (403 if code == "cannot_create" else 502)
    assert response.json()["ok"] is False and response.json()["code"] == code
    assert assert_bound("POST " + ENDPOINT, response)
    if code == "cannot_create":
        assert response.json()["error"] == "The GitHub token needs permission to create repositories."
    assert "ghp_SYNTHETIC1234567890" not in response.text
    assert not settings.exists() and skill_sources.load_sources() == INITIAL
    assert len(calls) == {"user": 1, "read": 2, "create": 3}[stage]


def test_local_save_failure_is_failed_and_retry_can_connect_existing_repository(env, monkeypatch):
    client, _settings = env
    _fake(monkeypatch, exists=True)
    with monkeypatch.context() as local:
        local.setattr(skill_my_skills, "save_sources", lambda rows: (_ for _ in ()).throw(OSError("disk full")))
        failed = client.post(ENDPOINT, json={})
    assert failed.status_code == 502 and failed.json()["code"] == "failed"
    assert client.post(ENDPOINT, json={}).json()["ok"] is True


def test_new_my_skills_source_publishes_and_merges_through_the_existing_transaction(env, monkeypatch, tmp_path):
    client, _settings = env
    ctx, events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    monkeypatch.setattr(skill_publish, "resolve_publish_destination", skill_publish_destination.resolve_publish_destination)
    for name in ("github_login", "fetch_repository", "fetch_branch_head", "fetch_tree_paths",
                 "prepare_publish_repository", "ensure_branch", "commit_payload", "create_pr_receipt", "merge_pull_request"):
        monkeypatch.setattr(skill_publish, name, getattr(skill_publish_github, name))
    calls, commits = [], []
    exists = False

    def request(method, path, **kwargs):
        nonlocal exists
        calls.append((method, path, kwargs))
        if path == "/user":
            return {"login": "alice"}
        if path == f"/repos/{REPOSITORY}":
            if not exists:
                raise GitHubError("Not Found", http_status=404)
            return {"default_branch": "trunk", "permissions": {"push": True}, "private": True}
        if path == "/user/repos":
            assert kwargs["body"]["private"] and kwargs["body"]["auto_init"]
            exists = True
            return {"name": "ouroboros-skills", "private": True}
        if "/git/refs/heads/trunk" in path:
            return {"object": {"sha": BASE_SHA}}
        if "/git/trees/" in path:
            return {"tree": []}
        if "/git/ref/heads/submit/" in path:
            raise GitHubError("Not Found", http_status=404)
        if path.endswith("/git/refs"):
            return {"object": {"sha": BASE_SHA}}
        if path == "/graphql":
            assert len(events) >= 4 and all(event[0] == "scan" for event in events)
            payload = kwargs["body"]["variables"]["input"]
            assert payload["branch"]["repositoryNameWithOwner"] == REPOSITORY
            commits.append(payload)
            return {"data": {"createCommitOnBranch": {"commit": {"oid": COMMIT_SHA, "url": URL + "/commit/" + COMMIT_SHA}}}}
        if path.endswith("/pulls"):
            return {"html_url": URL + "/pull/7", "number": 7}
        assert method == "PUT" and path.endswith("/pulls/7/merge")
        assert kwargs["body"] == {"sha": COMMIT_SHA, "merge_method": "squash"}
        return {"merged": True, "sha": "9" * 40}

    monkeypatch.setattr(skill_my_skills, "github_request", request)
    monkeypatch.setattr(skill_publish_github, "request", request)
    connected = client.post(ENDPOINT, json={}).json()
    assert connected["ok"] and connected["created"]
    result = _submit(ctx, destination=connected["source"]["id"])
    assert result["ok"], result
    assert (result["files_published"], result["listed"], result["merged_commit"]) == ("merged", "listed", "9" * 40)
    assert result["receipt"]["repository"] == REPOSITORY and result["receipt"]["url"] == URL + "/pull/7"
    committed = {f["path"]: base64.b64decode(f["contents"]) for f in commits[0]["fileChanges"]["additions"]}
    assert committed == {"demo/" + f.path: f.content for f in _snapshot().public_files}
    assert result["publication_recorded"] and captured["llm_calls"] == 1
    assert not any("forks" in path or "merge-upstream" in path for _, path, _ in calls)
    record = json.loads((tmp_path / "state/skills/demo/ouroboroshub.json").read_text(encoding="utf-8"))
    assert record["published"]["repository"] == REPOSITORY
