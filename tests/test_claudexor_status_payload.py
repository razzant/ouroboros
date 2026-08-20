"""The status payload fan-out and the wake endpoint.

Split verbatim out of ``tests/test_claudexor_owned_daemon.py`` by theme. This module
owns the concurrent catalog/manifest/profile/quota reads, the typed classification of
each facet independently — including a normalized empty or half-filled envelope as a
failed read — the declared gateway contract they must match, and the wake endpoint
that starts the daemon and answers with a fresh reading or a typed refusal.

Everything here is offline: no daemon is spawned, no network is touched.
"""

import json

import pytest

from ouroboros import claudexor_daemon as owned


def test_status_payload_fans_out_the_independent_daemon_reads(monkeypatch, tmp_path):
    """The four catalog/manifest/profile/quota GETs run CONCURRENTLY.

    Each costs seconds daemon-side (it re-probes the coding-agent CLIs on every
    read), so serialized they made the Providers panel wait for their SUM — ~23s
    on a warm daemon, with nothing on screen (owner report, 2026-08-08).

    The pin is a rendezvous, not a stopwatch: all four reads must meet at one
    barrier before any returns, which only a genuine fan-out can do. Serialized
    code times out at the barrier instead of failing on a wall-clock margin no
    loaded CI machine can honor (review lens, 2026-08-08).
    """
    import threading

    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw

    rendezvous = threading.Barrier(4, timeout=10)

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "3.3.13"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            rendezvous.wait()
            return {"harnesses": [{"id": "codex", "displayName": "Codex CLI",
                                   "status": "ok", "enabled": True}]}

        def harnesses(self):
            rendezvous.wait()
            return [{"id": "codex", "manifest": {"capability_profile": {"auth": {
                "supported_sources": ["native_session"]}}}}]

        def credential_profiles(self):
            rendezvous.wait()
            return {"profiles": [], "harnessAccounts": []}

        def quota_snapshots(self):
            rendezvous.wait()
            return [{"subject": {"harness": "codex"}}]

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    # Serialized, the first read blocks forever waiting for siblings that never
    # start: the barrier breaks and the call raises instead of quietly passing.
    payload = _status_payload(include_models=False)

    assert [h["id"] for h in payload["harnesses"]] == ["codex"]
    assert payload["quota"] and payload["profiles"] == {"profiles": [], "harnessAccounts": []}


def test_status_payload_keeps_typed_unreachable_when_a_fanned_out_read_refuses(monkeypatch, tmp_path):
    """Concurrency must not change WHAT a refusal means: a catalog read that
    raises still lands as the typed unreachable daemon state, not a half-filled
    panel that looks healthy."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "3.3.13"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            raise ClaudexorUnavailable("daemon_unreachable", "gone mid-read")

        def harnesses(self):
            return []

        def credential_profiles(self):
            return {}

        def quota_snapshots(self):
            return []

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    payload = _status_payload(include_models=False)
    assert payload["daemon"]["state"] == "unreachable"
    assert "daemon_unreachable" in payload["daemon"]["last_error"]
    assert payload["harnesses"] == []


def _reads_probe(monkeypatch, tmp_path, daemon_state, failing_facet="", malformed=None):
    """Drive _status_payload with one facet optionally refusing, or answering
    with a body that does not carry the envelope it promised — either one the
    transport already collapsed to ``{}``, or an object that kept only half of
    the keys it owes."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    class FakeDaemon:
        def status_dict(self):
            return {"state": daemon_state}

    def refuse_if(name):
        if failing_facet == name:
            raise ClaudexorUnavailable("daemon_unreachable", f"{name} refused")

    class FakeGateway:
        engine_version = "3.3.13"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            refuse_if("catalog")
            if malformed == "catalog":
                return {}
            return {"harnesses": [{"id": "codex", "displayName": "Codex CLI",
                                   "status": "ok", "enabled": True}]}

        def harnesses(self):
            return [{"id": "codex", "manifest": {"capability_profile": {"auth": {
                "supported_sources": ["native_session"]}}}}]

        def credential_profiles(self):
            refuse_if("accounts")
            if malformed == "accounts":
                return {}
            if malformed == "accounts_half_named":
                return {"profiles": []}          # native rows key missing
            if malformed == "accounts_half_native":
                return {"harnessAccounts": []}   # named profiles key missing
            return {"profiles": [], "harnessAccounts": []}

        def quota_snapshots(self):
            refuse_if("quota")
            return []

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)
    return _status_payload(include_models=False)


@pytest.mark.parametrize("facet", ["catalog", "accounts"])
def test_status_payload_calls_a_normalized_empty_envelope_a_failed_read(
    monkeypatch, tmp_path, facet
):
    """A NON-OBJECT 2xx body — null, a list, a string — is collapsed by the
    transport into an empty ``{}`` (``ClaudexorGateway.agent_capabilities`` /
    ``credential_profiles`` both end in
    ``return body if isinstance(body, dict) else {}``), so it arrives looking
    like a legitimate empty answer. Without the envelope check it is published
    as `ok` — an AUTHORITATIVE nothing — and one daemon-side schema drift
    silently restores the owner-visible lie. The type check alone cannot see
    this: `{}` IS a dict. (A drifted OBJECT is not normalized at all; it reaches
    the same verdict through the same check — see the half-envelope test.)"""
    payload = _reads_probe(monkeypatch, tmp_path, "running", malformed=facet)

    assert payload["reads"][facet] == "failed", "a body that answered nothing read as ok"
    # The SIBLING facets are untouched — the envelope check must not become a
    # second way for one read to speak for another.
    for other in ("catalog", "accounts", "quota"):
        if other != facet:
            assert payload["reads"][other] == "ok"


@pytest.mark.parametrize("daemon_state", ["not_provisioned", "stale", "foreign_daemon"])
def test_status_payload_marks_every_facet_unread_when_the_daemon_is_not_running(
    monkeypatch, tmp_path, daemon_state
):
    """A lazily-started daemon is the ORDINARY idle state, so this is the path the
    owner actually sees. Empty collections here mean "never asked" — the panel
    printed "no account connected" for three harnesses while two claude profiles,
    a cursor profile and two native sessions sat on disk (owner report,
    2026-08-08). The payload must SAY it was not read (BIBLE P1: a gap is a gap)."""
    payload = _reads_probe(monkeypatch, tmp_path, daemon_state)

    assert payload["reads"] == {
        "catalog": "not_read", "accounts": "not_read", "quota": "not_read",
    }
    assert payload["harnesses"] == [] and payload["profiles"] == {}


def test_status_payload_marks_facets_ok_when_the_daemon_answered(monkeypatch, tmp_path):
    """The other half of the contract: after a successful read an EMPTY collection
    is authoritative — it really does mean "no account" — otherwise the honest
    hedge would never step aside and the panel could never say anything."""
    payload = _reads_probe(monkeypatch, tmp_path, "running")

    assert payload["reads"] == {"catalog": "ok", "accounts": "ok", "quota": "ok"}
    assert [h["id"] for h in payload["harnesses"]] == ["codex"]


def test_status_payload_discloses_a_refused_per_harness_model_read(monkeypatch, tmp_path):
    """`include=models` asks the daemon for each harness's model list separately,
    and one of those reads can refuse while the catalog itself landed. The row
    carries `models_error` so the UI can say "not checked" instead of calling a
    saved model undiscovered — a verdict about a search nobody ran. Nothing
    pinned the field's emission, so dropping it left every suite green."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "3.3.13"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            return {"harnesses": [{"id": "codex", "displayName": "Codex CLI",
                                   "status": "ok", "enabled": True}]}

        def harnesses(self):
            return [{"id": "codex", "manifest": {"capability_profile": {"auth": {
                "supported_sources": ["native_session"]}}}}]

        def harness_models(self, harness_id):
            raise ClaudexorUnavailable("daemon_unreachable", "model read refused")

        def credential_profiles(self):
            return {"profiles": [], "harnessAccounts": []}

        def quota_snapshots(self):
            return []

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    payload = _status_payload(include_models=True)

    row = payload["harnesses"][0]
    assert row["models"] == []
    assert row["models_error"] == "daemon_unreachable", (
        "a refused model read is indistinguishable from a harness with no models"
    )
    # The CATALOG itself answered, so its facet stays authoritative.
    assert payload["reads"]["catalog"] == "ok"


@pytest.mark.parametrize("half", ["accounts_half_named", "accounts_half_native"])
def test_status_payload_calls_half_an_account_envelope_a_failed_read(monkeypatch, tmp_path, half):
    """The accounts envelope carries TWO collections — named credential profiles
    and the daemon's native per-harness rows — and the owner's machine has both
    kinds. Accepting an envelope that brought only one made the missing half an
    authoritative empty: exactly the reported bug ("no account connected" beside
    accounts that exist), reached through a half-answer instead of a lazy daemon.
    The engine schema declares both inside a strict object
    (`ControlCredentialProfilesResponse`): `profiles` is REQUIRED outright and
    `harnessAccounts` carries `.default([])`, so a validating daemon always
    materializes the pair and a body missing either one is a read that did not
    answer."""
    payload = _reads_probe(monkeypatch, tmp_path, "running", malformed=half)

    assert payload["reads"]["accounts"] == "failed", (
        "half an account envelope was published as an authoritative empty"
    )
    assert payload["reads"]["catalog"] == "ok" and payload["reads"]["quota"] == "ok"


@pytest.mark.parametrize("failing", ["catalog", "accounts", "quota"])
def test_status_payload_classifies_each_fanned_out_facet_independently(
    monkeypatch, tmp_path, failing
):
    """ORDER-INDEPENDENCE. The facets are read concurrently, so a sibling's refusal
    must never downgrade a facet whose own read landed — and the verdict must not
    depend on which `.result()` the code happened to touch first. Consuming them
    in sequence used to report `accounts` as unread whenever `catalog` raised."""
    payload = _reads_probe(monkeypatch, tmp_path, "running", failing_facet=failing)

    expected = {"catalog": "ok", "accounts": "ok", "quota": "ok"}
    expected[failing] = "failed"
    assert payload["reads"] == expected
    # The refusal is still disclosed on the daemon, and the surviving facets keep
    # their payload instead of blanking the whole panel.
    assert payload["daemon"]["state"] == "unreachable"
    assert "daemon_unreachable" in payload["daemon"]["last_error"]
    if failing != "catalog":
        assert [h["id"] for h in payload["harnesses"]] == ["codex"]


def test_status_payload_reads_block_matches_the_declared_gateway_contract(monkeypatch, tmp_path):
    """PRODUCER pin: the wire always carries the full `reads` block with the exact
    keys the frozen gateway contract declares, so a consumer may key on it without
    defensive guessing."""
    from typing import get_type_hints

    from ouroboros.gateway.contracts import ClaudexorStatusReads

    payload = _reads_probe(monkeypatch, tmp_path, "running")

    assert set(payload["reads"]) == set(get_type_hints(ClaudexorStatusReads))
    assert set(payload["reads"].values()) <= {"ok", "not_read", "failed"}


def test_wake_endpoint_starts_the_daemon_and_returns_the_fresh_reading(monkeypatch, tmp_path):
    """The owner-initiated start behind the panel's Refresh button.

    The status GET is side-effect-free by contract, which leaves Refresh unable
    to do anything about a sleeping daemon — an owner who just wants to SEE
    their accounts had to start a login job or a delegated run. This endpoint is
    that missing action: it ensures the daemon, then answers with the reading it
    just made possible.
    """
    import asyncio

    from ouroboros.gateway import claudexor_accounts as accounts

    started = {"n": 0}

    class FakeGateway:
        def close(self):
            pass

    def fake_ensure():
        started["n"] += 1
        return FakeGateway()

    order = []

    def fake_ensure_ordered():
        order.append("ensure")
        return fake_ensure()

    def fake_status(include_models):
        order.append("read")
        return {"daemon": {"state": "running"},
                "reads": {"catalog": "ok", "accounts": "ok", "quota": "ok"}}

    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", fake_ensure_ordered)
    monkeypatch.setattr(accounts, "_status_payload", fake_status)

    response = asyncio.run(accounts.api_claudexor_wake(object()))

    # ORDER is the whole promise of this endpoint: a read taken BEFORE the
    # daemon exists answers with the same nothing Refresh already had, which is
    # the state the owner pressed the button to leave. The docstring said
    # "ensures the daemon, then answers with the reading it just made possible"
    # while a mocked constant payload made the sequence unobservable.
    assert order == ["ensure", "read"], f"the wake read the status before starting anything: {order}"


    assert started["n"] == 1, "the wake must actually ensure the daemon"
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["daemon"]["state"] == "running"
    assert body["reads"]["accounts"] == "ok", "the answer is the POST-wake reading"


def test_wake_endpoint_discloses_a_typed_refusal_instead_of_a_generic_error(monkeypatch, tmp_path):
    """A cold machine can refuse for reasons the owner can act on (no binary, a
    foreign daemon home). That reason must reach the panel as a typed 503 — the
    button says why it could not start, rather than returning silently to idle."""
    import asyncio

    from ouroboros.gateway import claudexor_accounts as accounts
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def refuse():
        raise ClaudexorUnavailable("claudexord_not_installed", "no managed binary")

    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", refuse)

    response = asyncio.run(accounts.api_claudexor_wake(object()))

    assert response.status_code == 503
    assert "claudexord_not_installed" in json.loads(response.body)["error"]


def test_status_payload_stamps_the_unified_accounts_fact(monkeypatch, tmp_path):
    """`unified_accounts` rides every status answer: True only when the
    operations catalog was READ and carries the account-pools marker; an old
    engine reads False, and a catalog read failure fails CLOSED to False (the
    legacy rendering is correct on every engine; a guessed True is not)."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw
    import ouroboros.claudexor_daemon as owned

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "9.9.9"
        operations_answer: list = [{"id": "get:account-pools"}]

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            return {"harnesses": []}

        def harnesses(self):
            return []

        def credential_profiles(self):
            return {"profiles": [], "harnessAccounts": [], "accountPools": []}

        def quota_snapshots(self):
            return []

        def operations(self):
            return type(self).operations_answer

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    assert _status_payload(include_models=False)["unified_accounts"] is True

    FakeGateway.operations_answer = [{"id": "get:quota"}]
    assert _status_payload(include_models=False)["unified_accounts"] is False

    class BrokenCatalog(FakeGateway):
        def operations(self):
            raise gw.ClaudexorUnavailable("daemon_unreachable", "catalog read died")

    monkeypatch.setattr(gw, "ClaudexorGateway", BrokenCatalog)
    payload = _status_payload(include_models=False)
    assert payload["unified_accounts"] is False, "an unreadable catalog fails closed to the old model"
    # …and the absorbed catalog read never downgrades the real facets.
    assert payload["reads"] == {"catalog": "ok", "accounts": "ok", "quota": "ok"}

    class NoCatalogMethod(FakeGateway):
        operations = None

    monkeypatch.setattr(gw, "ClaudexorGateway", NoCatalogMethod)
    assert _status_payload(include_models=False)["unified_accounts"] is False

    class UnifiedWire(FakeGateway):
        # The unified engine's full accounts body (frozen contract §L.1):
        # every account a named registry row, the legacy key an empty
        # compatibility list, the routing verdict in the ADDITIVE pool key.
        operations_answer = [{"id": "get:account-pools"}]

        def harnesses(self):
            # A populated manifest read turns the visibility filters ON —
            # exactly the path that rewrites the profiles body's other keys.
            return [{"id": "codex", "manifest": {"capability_profile": {
                "auth": {"supported_sources": ["native_session"]}}}}]

        def credential_profiles(self):
            return {
                "profiles": [{"profile": {"harness_id": "codex",
                                          "profile_id": "codex-default"}}],
                "harnessAccounts": [],
                "accountPools": [{"harness_id": "codex",
                                  "next_up": {"kind": "profile",
                                              "profileId": "codex-default"}}],
            }

    monkeypatch.setattr(gw, "ClaudexorGateway", UnifiedWire)
    served = _status_payload(include_models=False)
    # The ADDITIVE pool key rides the accounts facet through the visibility
    # filters untouched: the store's dual-wire nextUpAccount reader and the
    # onboarding dual-read both consume it from this one served payload.
    assert served["unified_accounts"] is True
    assert served["profiles"]["accountPools"] == [
        {"harness_id": "codex",
         "next_up": {"kind": "profile", "profileId": "codex-default"}}]
    assert served["profiles"]["harnessAccounts"] == []
    assert [w["profile"]["profile_id"] for w in served["profiles"]["profiles"]] == ["codex-default"]

    class StoppedDaemon:
        def status_dict(self):
            return {"state": "stale"}

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: StoppedDaemon())
    assert _status_payload(include_models=False)["unified_accounts"] is False
