"""The OpenRouter catalogue is observed off the turn, recorded once and adopted by every process.

Each process (a server restart, every new worker) used to GET the 790 KB ``/models`` catalogue
on its first OpenRouter request, with a five-second timeout, before the request left.
"""

from __future__ import annotations

import datetime
import threading

import pytest

from tests._shared import _PROVIDER_CATALOG_PRISTINE, _restore_provider_catalog

CATALOGUE = {"data": [{"id": "vendor/known", "context_length": 200_000,
                       "supported_parameters": ["max_tokens", "tools"]}]}


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    """A fresh process on its own data root; ``gets`` counts the catalogue's network reads."""
    import requests

    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    gets = []

    class Answer:
        status_code = 200

        def json(self):
            return CATALOGUE

    monkeypatch.setattr(requests, "get", lambda url, **_kwargs: gets.append(url) or Answer())
    _restore_provider_catalog(LLMClient, _PROVIDER_CATALOG_PRISTINE)
    return LLMClient, gets, tmp_path / "state" / "openrouter_models.json"


def _another_process(client) -> None:
    _restore_provider_catalog(client, _PROVIDER_CATALOG_PRISTINE)


def test_a_second_process_adopts_the_recorded_catalogue_without_a_get(catalogue):
    client, gets, record = catalogue
    client.prefetch_openrouter_catalog()  # the first process's boot
    assert len(gets) == 1 and record.is_file()

    _another_process(client)
    assert client._get_supported_parameters("vendor/known") == {"max_tokens", "tools"}
    assert client.openrouter_context_length("vendor/known", allow_fetch=True) == 200_000
    assert not client.metadata_fetch_attempted_and_failed()
    _another_process(client)
    client.prefetch_openrouter_catalog()  # a restart within the evidence TTL
    assert len(gets) == 1


def test_an_expired_record_costs_one_get_and_is_replaced(catalogue):
    from ouroboros.utils import atomic_write_json

    client, gets, record = catalogue
    stale = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)).isoformat()
    atomic_write_json(record, {"observed_at": stale, "models": {
        "vendor/known": {"context_length": 1, "supported_parameters": ["max_tokens"]}}})

    assert client._get_supported_parameters("vendor/known") is None  # expired: unknown, never fetched here
    assert client.openrouter_context_length("vendor/known", allow_fetch=True) == 200_000 and len(gets) == 1
    _another_process(client)
    assert client.openrouter_context_length("vendor/known", allow_fetch=True) == 200_000 and len(gets) == 1


def test_a_model_the_record_lacks_is_unknown_until_a_fetch_is_allowed(catalogue):
    client, gets, _record = catalogue
    client.prefetch_openrouter_catalog()
    _another_process(client)

    assert client.openrouter_context_length("vendor/new", allow_fetch=False) == 0 and len(gets) == 1
    assert client._get_supported_parameters("vendor/new") is None  # unknown, not denied: nothing stripped
    client.openrouter_context_length("vendor/new", allow_fetch=True)  # Settings' check observes it now
    assert len(gets) == 2


@pytest.mark.parametrize("settings,live,started", [
    ({"OPENROUTER_API_KEY": "key"}, True, True),
    ({}, True, False),  # no OpenRouter route on this install: nothing to observe
    ({"OPENROUTER_API_KEY": "key"}, False, False),  # not a provisioned home: no real-data side effect
    ({"OUROBOROS_CONNECTIONS": [{"kind": "openrouter", "name": "credits", "api_key": "key"}]}, True, True),  # added key
    ({"OPENROUTER_API_KEY": "key", "OUROBOROS_CONNECTIONS": [{"kind": "openrouter", "name": "default", "enabled": False}]},
     True, False),  # the only OpenRouter key turned off
])
def test_boot_observes_the_catalogue_off_the_startup_path(monkeypatch, settings, live, started):
    from ouroboros import claudexor_daemon, server_boot
    from ouroboros.llm import LLMClient

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(claudexor_daemon, "warm_owned_daemon", lambda: None)
    observed = threading.Event()
    monkeypatch.setattr(LLMClient, "prefetch_openrouter_catalog", classmethod(lambda cls: observed.set()))

    server_boot.start_background_services(settings, local_model=False, boot_managed_update=lambda: None,
                                          warm_daemon=live)
    assert observed.wait(5 if started else 0.2) is started


def _turn_probe(root):
    """What a turn's context fit asks of its route (``capability_evidence.probe``), fetching allowed as for every
    task but a subagent's (``context_fit.build_context_fit_plan``, ``loop_model_call``)."""
    from ouroboros.capability_evidence import probe

    return probe(root, provider="openrouter", model="vendor/known", base_url="https://openrouter.ai/api/v1",
                 allow_fetch=True)


def test_with_no_record_an_install_pays_one_get_and_a_pooled_worker_none(catalogue):
    """No record yet (a boot without an OpenRouter key, a new install): the first turn's probe observes the catalogue
    once and records it; the serving process's next turn and a pooled worker's (another process: the stores shared,
    no catalogue in memory) read what is held, its requests' parameters included."""
    client, gets, record = catalogue
    root = record.parent.parent
    for _turn in range(2):
        assert _turn_probe(root).window_tokens == 200_000
    assert len(gets) == 1 and record.is_file()

    _another_process(client)  # a pooled worker
    assert _turn_probe(root).window_tokens == 200_000
    assert client._get_supported_parameters("vendor/known") == {"max_tokens", "tools"}
    assert client.openrouter_context_length("vendor/known", allow_fetch=True) == 200_000
    assert len(gets) == 1


def test_with_no_record_and_no_network_a_turn_tries_the_catalogue_once(catalogue, monkeypatch):
    """With no record and no network the window reader's failed read (``openrouter_context_length``) answers the
    output limit's reader too (``resolve_response_limit``), each of which would wait out the 5 s timeout; the
    failures held for ``_FAILED_TTL_SEC`` spare the next turn and a pooled worker another read. The output limit
    names the failed read that answered it, never a maximum the catalogue was seen to omit."""
    import requests

    client, gets, record = catalogue

    def unreachable(url, **_kwargs):
        gets.append(url)
        raise requests.ConnectionError("no route to openrouter.ai")

    monkeypatch.setattr(requests, "get", unreachable)
    root = record.parent.parent
    limits = [_turn_probe(root).response_limit for _turn in range(2)]
    _another_process(client)  # a pooled worker within the failure window
    limits.append(_turn_probe(root).response_limit)
    assert len(gets) == 1 and not record.exists()
    assert {(limit["status"], limit["source"]) for limit in limits} == {
        ("unprobeable", "openrouter metadata: metadata read failed")}


@pytest.mark.parametrize("answers", [True, False], ids=["read-answers", "read-fails"])
def test_a_reader_overlapping_a_catalogue_read_in_flight_takes_its_answer(catalogue, monkeypatch, answers):
    """The boot's prefetch still waits for /models when a turn's probe asks: the probe waits for that read and takes
    its answer, so it is never told of a failure that has not happened and pays no second GET; when the read in
    flight fails, that finished failure answers the probe too (one attempt, as with no network)."""
    import requests

    client, gets, record = catalogue
    answer, in_flight, release = requests.get, threading.Event(), threading.Event()

    def the_read_in_flight(url, **kwargs):
        in_flight.set()
        assert release.wait(10)
        if not answers:
            gets.append(url)
            raise requests.ConnectionError("reset by peer")
        return answer(url, **kwargs)

    monkeypatch.setattr(requests, "get", the_read_in_flight)
    boot = threading.Thread(target=client.prefetch_openrouter_catalog, daemon=True)
    boot.start()
    assert in_flight.wait(5)
    threading.Timer(0.3, release.set).start()  # the boot's read answers while the probe waits for it
    probe = _turn_probe(record.parent.parent)
    boot.join(10)
    assert len(gets) == 1 and client.metadata_fetch_attempted_and_failed() is not answers
    if answers:
        assert probe.window_tokens == 200_000 and "failed" not in probe.response_limit["source"]
    else:
        assert probe.response_limit["source"] == "openrouter metadata: metadata read failed"
