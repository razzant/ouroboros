"""Readers of ``state/capability_evidence.json`` share one parse; its writers keep their fresh copy.

A context build asks the store for densities, owner acknowledgements and the probe cache
several times per turn; writes come about twice per model round. Model-global rows older
releases learned still load, and every writer carries them forward.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import capability_evidence as ce
from ouroboros import utils
from tests._shared import write_legacy_capability_rows


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_RACY_NS", 0)  # trust files as written (the clock-tick rule has its own case)
    write_legacy_capability_rows(tmp_path, effort_ceilings={"openai/model": "high"},
                                 rejected_params={"openai/model": ["temperature"]})
    ce.record_owner_ack(tmp_path, provider="openrouter", model="openai/model", window_tokens=100000)
    return tmp_path


def _parses_of(monkeypatch, path) -> list:
    seen, real = [], json.loads
    monkeypatch.setattr(utils.json, "loads", lambda raw, *args, **kwargs: (
        seen.append(raw) if raw == path.read_text(encoding="utf-8") else None) or real(raw, *args, **kwargs))
    return seen


def test_a_turns_readers_share_one_parse_and_see_each_write(store, monkeypatch):
    parses = _parses_of(monkeypatch, ce._store_path(store))

    assert ce._read(store)["effort_ceilings"]["openai/model"]["ceiling"] == "high"
    assert ce.get_token_density(store, "openai/model") == 0.0
    assert [ack["route"]["model"] for ack in ce.list_owner_acks(store)] == ["openai/model"]
    assert ce.resolve_review_token_density(store, "openai/model")[1] == "cold_conservative"
    assert len(parses) == 1

    ce.record_owner_ack(store, provider="openrouter", model="openai/other", window_tokens=50000)  # the writer's rename moves the identity
    assert len(ce.list_owner_acks(store)) == 2
    assert ce._read(store)["rejected_params"]["openai/model"]["params"] == ["temperature"]


def test_a_writers_copy_stays_invisible_until_it_saves(store, monkeypatch):
    assert len(ce.list_owner_acks(store)) == 1  # the shared parse is warm

    def refused(_root, _data):
        return False  # a failed save: _save reports it and the writer goes on

    with monkeypatch.context() as patch:
        patch.setattr(ce, "_save", refused)
        assert ce._store_evidence(store, "probes", "fp", {"status": "confirmed"}) is False
        with pytest.raises(OSError):
            ce.record_owner_ack(store, provider="openrouter", model="openai/other", window_tokens=50000)

    assert ce._read(store)["probes"] == {}
    assert len(ce.list_owner_acks(store)) == 1
