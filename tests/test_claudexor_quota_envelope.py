"""One-read quota envelope regressions for route health and Accounts status."""

from __future__ import annotations

from concurrent.futures import Future

import httpx

from ouroboros.gateways.claudexor import ClaudexorGateway, DaemonEndpoint
from ouroboros.subagents import _exhausted_window


def _spent(subject_id: str) -> dict:
    return {
        "subject": {"harness": "claude", "subject_id": subject_id},
        "freshness": "fresh",
        "constraints": [{"used_ratio": 1.0, "resets_at": "2099-01-01T00:00:00Z"}],
    }


def test_route_health_contradictory_same_subject_absence_fails_open():
    class Gateway:
        reads = 0

        def quota_state(self):
            self.reads += 1
            return {
                "snapshots": [_spent("proton4")],
                "absences": [{
                    "subject": {"harness": "claude", "subject_id": "proton4"},
                    "reason": "refresh_failed",
                }],
                "refreshed_at": "2026-08-30T00:00:00Z",
            }

        def quota_snapshots(self):  # pragma: no cover - forbidden production path
            raise AssertionError("route health must not perform a second quota read")

        def quota_absences(self):  # pragma: no cover - forbidden production path
            raise AssertionError("route health must not perform a second quota read")

    gateway = Gateway()
    assert _exhausted_window(gateway, "claude", pinned_profile="proton4") == (False, "")
    assert gateway.reads == 1


def test_route_health_gap_for_another_subject_keeps_unpinned_route_unknown():
    class Gateway:
        def quota_state(self):
            return {
                "snapshots": [_spent("proton4")],
                "absences": [{
                    "subject": {"harness": "claude", "subject_id": "proton3"},
                    "reason": "refresh_failed",
                }],
            }

    assert _exhausted_window(Gateway(), "claude") == (False, "")


def test_real_gateway_route_health_performs_one_physical_quota_get():
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, json={
            "snapshots": [_spent("proton4")],
            "absences": [{
                "subject": {"harness": "claude", "subject_id": "proton4"},
                "reason": "refresh_failed",
            }],
            "refreshed_at": "2026-08-30T00:00:00Z",
        })

    with ClaudexorGateway(DaemonEndpoint("127.0.0.1", 1, "test-token")) as gateway:
        gateway._client.close()
        gateway._client = httpx.Client(
            base_url="http://127.0.0.1:1",
            transport=httpx.MockTransport(handler),
        )
        assert _exhausted_window(
            gateway, "claude", pinned_profile="proton4") == (False, "")

    assert requests == ["GET /v2/quota"]


def test_same_envelope_sentinel_cannot_mix_with_a_later_quota_epoch():
    class Gateway:
        reads = 0

        def quota_state(self):
            self.reads += 1
            if self.reads > 1:
                raise AssertionError("a second epoch was read")
            return {"snapshots": [_spent("proton4")], "absences": []}

    gateway = Gateway()
    assert _exhausted_window(gateway, "claude", pinned_profile="proton4") == (
        True, "2099-01-01T00:00:00Z")
    assert gateway.reads == 1


def test_route_health_normalizes_malformed_quota_envelope_collections():
    class Gateway:
        def __init__(self, envelope):
            self.envelope = envelope

        def quota_state(self):
            return self.envelope

    for absences in (None, 42, "bad", {}):
        assert _exhausted_window(
            Gateway({"snapshots": [_spent("proton4")], "absences": absences}),
            "claude",
            pinned_profile="proton4",
        ) == (True, "2099-01-01T00:00:00Z")

    assert _exhausted_window(
        Gateway({"snapshots": [_spent("proton4")]}),
        "claude",
        pinned_profile="proton4",
    ) == (True, "2099-01-01T00:00:00Z")

    for snapshots in (None, 42, "bad", {}):
        assert _exhausted_window(
            Gateway({"snapshots": snapshots, "absences": []}),
            "claude",
            pinned_profile="proton4",
        ) == (False, "")

    assert _exhausted_window(
        Gateway({"absences": []}), "claude", pinned_profile="proton4"
    ) == (False, "")


def test_status_projects_snapshots_and_absences_from_its_single_read(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gateway_module

    class Daemon:
        def status_dict(self):
            return {"state": "running"}

    class Gateway:
        engine_version = "3.9.0"
        quota_reads = 0

        def __init__(self, _endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def handshake(self):
            return {}

        def agent_capabilities(self):
            return {"harnesses": []}

        def harnesses(self):
            return []

        def credential_profiles(self):
            return {"profiles": [], "harnessAccounts": []}

        def operations(self):
            return {"operations": []}

        def quota_state(self):
            type(self).quota_reads += 1
            return {
                "snapshots": [_spent("proton4")],
                "absences": [{
                    "subject": {"harness": "claude", "subject_id": "proton3"},
                    "reason": "poll_paced",
                }],
                # POST-refresh response metadata is not part of the ordinary
                # GET projection and must not grow a dead status contract.
                "refreshed_at": "2026-08-30T00:00:00Z",
                "refresh_skipped": [{"vendor": "claude", "not_before": "2026-08-30T01:00:00Z"}],
            }

        def quota_snapshots(self):  # pragma: no cover - forbidden status reader
            raise AssertionError("status must not perform a second quota read")

        def refresh_quota(self):  # pragma: no cover - passive GET must not mutate
            raise AssertionError("status must not perform a foreground quota refresh")

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: Daemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gateway_module, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", Gateway)

    payload = _status_payload(False)
    assert Gateway.quota_reads == 1
    assert payload["quota"][0]["subject"]["subject_id"] == "proton4"
    assert payload["quota_absences"][0]["reason"] == "poll_paced"
    assert "quota_refreshed_at" not in payload
    assert "quota_refresh_skipped" not in payload


def test_status_treats_absent_or_malformed_additive_quota_fields_as_empty(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gateway_module

    class Daemon:
        def status_dict(self):
            return {"state": "running"}

    class Gateway:
        engine_version = "3.8.3"

        def __init__(self, _endpoint): pass
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def handshake(self): return {}
        def agent_capabilities(self): return {"harnesses": []}
        def harnesses(self): return []
        def credential_profiles(self): return {"profiles": [], "harnessAccounts": []}
        def operations(self): return {"operations": []}
        def quota_state(self):
            return {"snapshots": [], "absences": 42}

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: Daemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gateway_module, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", Gateway)
    payload = _status_payload(False)
    assert payload["reads"]["quota"] == "ok"
    assert payload["quota_absences"] == []


def test_quota_facet_rejects_missing_or_non_list_snapshots():
    from ouroboros.gateway.claudexor_accounts import _facet_outcome

    for body in ({"absences": []}, {"snapshots": "bad", "absences": []}):
        call = Future()
        call.set_result(body)
        state, value, error = _facet_outcome(
            call,
            envelope=("snapshots",),
            list_fields=("snapshots",),
        )
        assert (state, value, error) == ("failed", None, None)
