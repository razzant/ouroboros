"""Missing discovery metadata never fabricates a model or an upstream recovery."""

from types import SimpleNamespace

import pytest

from ouroboros import llm_claudexor as transport
from ouroboros import llm_capability_policy as metadata
from ouroboros.gateways.claudexor import ClaudexorUnavailable


@pytest.mark.parametrize("admission,allowed", [
    (None, False),
    ({"requestedModel": "next-model", "inventoryAbsence": "authoritative"}, False),
    ({"requestedModel": "another-model", "inventoryAbsence": "advisory"}, False),
    ({"inventoryAbsence": "advisory"}, False),
    ({"requestedModel": "next-model", "inventoryAbsence": "advisory"}, True),
])
def test_admission_is_exact_and_legacy_membership_still_works(admission, allowed):
    catalog = {"models": [], "admission": admission}
    assert transport.catalog_admits_model(catalog, "next-model") is allowed
    assert catalog["models"] == []
    catalog["models"] = [{"id": "next-model"}]
    assert transport.catalog_admits_model(catalog, "next-model") is True


def test_catalog_negotiation_and_read_share_transport_budget(monkeypatch):
    clock, seen = [0.0], []

    def operations(**kwargs):
        assert kwargs == {"timeout_sec": 3}
        clock[0] = 2
        return [{"method": "GET", "path": "/v2/model-sources/:id/models",
                 "parameters": [{"name": "includeAdmission", "location": "query", "enum": ["true"]}]}]

    def read(*args, **kwargs):
        assert args == ("source", "account")
        assert kwargs == {"requested_model": "model", "include_admission": True, "timeout_sec": 1}
        raise ClaudexorUnavailable("catalog_unavailable", "Upstream metadata unavailable")

    def connect(**kwargs):
        assert kwargs == {"timeout_sec": 3}  # the handshake spends the same budget
        return SimpleNamespace(operations=operations, list_source_models=read, close=lambda: seen.append("closed"))

    monkeypatch.setattr(metadata, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(metadata, "read_owned_gateway", connect)
    with pytest.raises(ClaudexorUnavailable, match="Upstream metadata unavailable"):
        transport.model_catalog("source", "account", requested_model="model", timeout_sec=3)
    assert seen == ["closed"]
