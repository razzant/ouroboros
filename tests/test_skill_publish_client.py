"""GitHub wire behavior without a network or installed/authenticated gh."""

import json
import logging
import traceback

import httpx
import pytest

from ouroboros import skill_publish_client as client

TOKEN = "ghp_SYNTHETIC1234567890"


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    monkeypatch.setattr(client, "github_token_from_env_or_settings", lambda: TOKEN)
    monkeypatch.setattr(client.httpx, "request", lambda *_a, **_k: pytest.fail("unexpected HTTPS request"))


def test_request_uses_settings_token_json_headers_and_bounded_timeout(monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(201, json={"name": "ouroboros-skills"})

    monkeypatch.setattr(client.httpx, "request", request)
    body = {"name": "ouroboros-skills", "private": True, "auto_init": True}
    assert client.request("POST", "/user/repos", body=body) == {"name": "ouroboros-skills"}
    method, url, kwargs = calls[0]
    assert method == "POST" and url == "https://api.github.com/user/repos"
    assert kwargs["json"] == body and kwargs["timeout"] == 30
    assert kwargs["headers"] == {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
                                 "User-Agent": "Ouroboros", "X-GitHub-Api-Version": "2022-11-28"}
    client.request("GET", "/user", timeout=60)
    assert calls[-1][2]["json"] is None and calls[-1][2]["timeout"] == 60


def test_missing_token_is_one_plain_sentence_before_io(monkeypatch):
    monkeypatch.setattr(client, "github_token_from_env_or_settings", lambda: "")
    with pytest.raises(client.GitHubError) as caught:
        client.request("GET", "/user")
    assert str(caught.value) == "Add a GitHub token in Settings to publish"
    assert caught.value.failure == "no_token" and caught.value.http_status is None


@pytest.mark.parametrize("status,body", [(401, {"message": "Bad credentials"}), (403, {"message": "rate limited"}),
                                        (404, {"message": "Not Found"}),
                                        (422, {"message": "Validation Failed", "errors": [{"field": "name"}]})])
def test_http_status_and_github_message_are_preserved(monkeypatch, status, body):
    monkeypatch.setattr(client.httpx, "request", lambda *_a, **_k: httpx.Response(status, json=body))
    with pytest.raises(client.GitHubError) as caught:
        client.request("POST", "/user/repos", body={"name": "test"})
    error = caught.value
    assert error.http_status == status and error.failure == "http"
    assert body["message"] in error.message and str(status) in str(error)
    if body.get("errors"):
        assert "name" in error.message


@pytest.mark.parametrize("kind", ["http", "invalid_json", "graphql", "connection", "timeout"])
def test_token_never_appears_in_errors_tracebacks_or_logs(monkeypatch, caplog, kind):
    caplog.set_level(logging.DEBUG)

    def request(*args, **kwargs):
        if kind == "connection":
            raise httpx.ConnectError("failed " + TOKEN)
        if kind == "timeout":
            raise httpx.ReadTimeout("failed " + TOKEN)
        if kind == "graphql":
            return httpx.Response(200, json={"errors": [{"message": TOKEN + " rejected"}]})
        if kind == "invalid_json":
            return httpx.Response(200, text="invalid " + TOKEN + "x" * 1000)
        return httpx.Response(403, json={"message": TOKEN + " refused", "errors": [TOKEN]})

    monkeypatch.setattr(client.httpx, "request", request)
    with pytest.raises(client.GitHubError) as caught:
        client.request("POST", "/graphql", body={"query": "mutation"})
    error = caught.value
    assert TOKEN not in str(error) + repr(error) + error.message + caplog.text
    assert TOKEN not in "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert len(error.message) <= 600
    assert error.failure == {"invalid_json": "invalid_response", "connection": "transport"}.get(kind, kind)
    if kind not in {"timeout", "connection"}:
        assert "***" in error.message


def test_plain_http_failure_and_empty_success(monkeypatch):
    monkeypatch.setattr(client.httpx, "request", lambda *_a, **_k: httpx.Response(502, text="Gateway unavailable"))
    with pytest.raises(client.GitHubError, match="Gateway unavailable") as caught:
        client.request("GET", "/user")
    assert caught.value.http_status == 502
    monkeypatch.setattr(client.httpx, "request", lambda *_a, **_k: httpx.Response(204))
    assert client.request("PUT", "/resource") == {}


def test_transport_never_changes_tool_result_sidecars(monkeypatch):
    # No ToolContext crosses this client seam; a successful JSON body is returned verbatim.
    answer = {"data": {"viewer": {"login": "alice"}}}
    monkeypatch.setattr(client.httpx, "request", lambda *_a, **_k: httpx.Response(200, content=json.dumps(answer)))
    assert client.request("POST", "/graphql", body={"query": "query { viewer { login } }"}) == answer
