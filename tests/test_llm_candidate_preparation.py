"""The dispatch gate checks identity without repeating candidate measurement."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from ouroboros import llm_attempt as attempts
from ouroboros import usage_accounting as ua


def _prepared(tmp_path):
    candidate = attempts._physical_candidate({"model": "fixture", "messages": [{"role": "user", "content": "hello 🐍"}]})
    request = attempts._attempt_request({"provider": "claudexor", "usage_model": "fixture"}, candidate)
    reservation = SimpleNamespace(attempt_id="candidate-one", drive_root=tmp_path)
    return candidate, request, reservation


def test_identity_gate_does_not_repeat_measurement(tmp_path, monkeypatch):
    candidate, request, reservation = _prepared(tmp_path)

    def measure_again(*_args, **_kwargs):
        pytest.fail("Dispatch repeated token, clock or image measurement")

    monkeypatch.setattr(attempts, "_attempt_request", measure_again)
    ref = attempts._candidate_before_dispatch(candidate, request)(reservation)
    with open(ref["path"]) as stream:
        assert json.load(stream)["model_send_seal"]["pre_redaction_sha256"] == request.candidate_raw_sha256


@pytest.mark.parametrize("drift", ["raw", "context", "context_identity"])
def test_identity_drift_refuses_dispatch(tmp_path, drift):
    candidate, request, reservation = _prepared(tmp_path)
    if drift == "raw":
        candidate["model"] = "another"
    elif drift == "context":
        candidate["messages"][0]["content"] = "changed after measurement"
    else:
        request = replace(request, candidate_context_sha256="0" * 64)
    with pytest.raises(ua.PhysicalAttemptPreparationFailed, match="candidate changed before dispatch"):
        attempts._candidate_before_dispatch(candidate, request,
            prepare=lambda *_args, **_kwargs: pytest.fail("Changed candidate reached preparation"))(reservation)
    assert not (tmp_path / "observability").exists()


def test_preparation_receives_exact_sealed_bytes_after_identity_gate(tmp_path):
    candidate, request, reservation = _prepared(tmp_path)
    prepared = []

    def prepare(actual_reservation, **facts):
        assert actual_reservation is reservation
        prepared.append(facts)

    ref = attempts._candidate_before_dispatch(candidate, request, prepare=prepare)(reservation)
    [facts] = prepared
    assert facts["canonical_bytes"] == attempts._canonical_candidate_bytes(candidate)
    assert facts["sha256"] == hashlib.sha256(facts["canonical_bytes"]).hexdigest()
    assert facts["size_bytes"] == len(facts["canonical_bytes"])
    with open(ref["path"]) as stream:
        seal = json.load(stream)["model_send_seal"]
    assert (seal["pre_redaction_sha256"], seal["size_bytes"]) == (facts["sha256"], facts["size_bytes"])


def test_physical_candidate_keeps_its_defensive_copy():
    source = {"messages": [{"role": "user", "content": ["original"]}]}
    candidate = attempts._physical_candidate(source)
    source["messages"][0]["content"].append("later mutation")
    assert candidate["messages"][0]["content"] == ["original"]
