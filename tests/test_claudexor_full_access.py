"""The existing trust HTTP surface grants only an absent caller-authorized scope."""
import json
from urllib.parse import quote

import httpx
import pytest

from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable, DaemonEndpoint


ROOT = "/owner/project with spaces & unicode-ю"
PATH = "/owner/claudexor/trust/scoped.yaml"


def state(allow=False, **changes):
    return {"repoRoot": ROOT, "path": PATH, "allowFullAccess": allow,
            "accessDefault": "workspace_write", "testCommandGrantCount": 0, **changes}


@pytest.fixture
def gateway(monkeypatch):
    replies, requests = [], []
    client = httpx.Client

    def handler(request):
        requests.append(request)
        assert request.url.path == "/v2/trust"
        assert request.headers["Authorization"] == "Bearer fixture-control-token"
        assert request.headers["X-Claudexor-Protocol-Major"] == "3"
        assert replies, "Unexpected extra trust request"
        reply = replies.pop(0)
        return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(
        **kwargs, transport=httpx.MockTransport(handler)))
    with ClaudexorGateway(DaemonEndpoint("127.0.0.1", 1, "fixture-control-token")) as instance:
        yield instance, replies, requests
    assert not replies, "Expected trust request was not made"


def test_existing_full_grant_is_read_only_and_preserves_exact_root(gateway):
    instance, replies, requests = gateway
    expected = state(True)
    replies.append({"entries": [expected]})
    assert instance.ensure_full_access(ROOT) == expected
    assert len(requests) == 1 and requests[0].method == "GET"
    assert requests[0].url.params["repoRoot"] == ROOT
    assert requests[0].url.query.decode() == "repoRoot=" + quote(ROOT, safe="")


def test_missing_record_creates_only_scoped_full_grant_and_reuse_never_posts(gateway):
    instance, replies, requests = gateway
    granted = state(True)
    replies.extend([{"entries": [state()]}, {"entries": [state(False, path=PATH + ".other", repoRoot="/other")]},
                    granted, {"entries": [granted]}])
    assert instance.ensure_full_access(ROOT) == granted
    assert instance.ensure_full_access(ROOT) == granted
    assert [request.method for request in requests] == ["GET", "GET", "POST", "GET"]
    assert not requests[1].url.query
    assert json.loads(requests[2].content) == {"repoRoot": ROOT, "allowFullAccess": True}


@pytest.mark.parametrize("record_root", [ROOT, None])
def test_existing_denial_including_legacy_path_is_never_overwritten(gateway, record_root):
    instance, replies, requests = gateway
    replies.extend([{"entries": [state()]}, {"entries": [state(repoRoot=record_root)]}])
    with pytest.raises(ClaudexorUnavailable) as refused:
        instance.ensure_full_access(ROOT)
    assert refused.value.code == "trust_full_access_required"
    assert refused.value.status_code == 403
    assert [request.method for request in requests] == ["GET", "GET"]


def test_grant_observed_in_actual_file_list_needs_no_write(gateway):
    instance, replies, requests = gateway
    granted = state(True, repoRoot=None)
    replies.extend([{"entries": [state()]}, {"entries": [granted]}])
    assert instance.ensure_full_access(ROOT) == granted
    assert [request.method for request in requests] == ["GET", "GET"]


@pytest.mark.parametrize("response", [
    {}, {"entries": []}, {"entries": [state(), state()]},
    {"entries": [state(repoRoot="/different")]},
    {"entries": [state(True, path="")]},
    {"entries": [state(allowFullAccess="true")]},
    {"entries": [state(True, accessDefault=None)]},
])
def test_unknown_scoped_state_never_grants_or_claims_success(gateway, response):
    instance, replies, requests = gateway
    replies.append(response)
    with pytest.raises(ClaudexorUnavailable) as refused:
        instance.ensure_full_access(ROOT)
    assert refused.value.code == "malformed_response"
    assert len(requests) == 1


@pytest.mark.parametrize("response", [{}, {"entries": None}, {"entries": [None]},
                                      {"entries": [{"repoRoot": ROOT}]}])
def test_unknown_actual_file_list_does_not_mean_absence(gateway, response):
    instance, replies, requests = gateway
    replies.extend([{"entries": [state()]}, response])
    with pytest.raises(ClaudexorUnavailable) as refused:
        instance.ensure_full_access(ROOT)
    assert refused.value.code == "malformed_response"
    assert [request.method for request in requests] == ["GET", "GET"]


@pytest.mark.parametrize("response", [state(), state(True, repoRoot="/different"),
                                      state(True, path="/different/trust.yaml"), {}])
def test_post_requires_confirmed_root_path_and_full_access(gateway, response):
    instance, replies, requests = gateway
    replies.extend([{"entries": [state()]}, {"entries": []}, response])
    with pytest.raises(ClaudexorUnavailable) as refused:
        instance.ensure_full_access(ROOT)
    assert refused.value.code == "malformed_response"
    assert [request.method for request in requests] == ["GET", "GET", "POST"]


def test_daemon_refusal_survives_without_a_retry_or_grant(gateway):
    instance, replies, requests = gateway
    replies.append(httpx.Response(403, json={
        "code": "trust_full_access_required", "message": "Denied by trust owner"}))
    with pytest.raises(ClaudexorUnavailable) as refused:
        instance.ensure_full_access(ROOT)
    assert refused.value.code == "trust_full_access_required"
    assert refused.value.status_code == 403
    assert len(requests) == 1
