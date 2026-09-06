"""CPL-5 pins: the ``model-visible ⟺ logged`` invariant at the model_send seam.

Design contract: ``docs/v7next/DESIGN_MODEL_VISIBLE_LOGGED.md`` (F15-narrowed).
Forward — every physical attempt seals its exact send copy before dispatch and
the seam reconstructs that durable record and byte-compares it ON THE CALL;
a mismatch is a typed durable fact, never a second dispatch gate. Exclusions
form a CLOSED enum; delegated lanes disclose ``unobserved`` instead of sealing.
Reverse — the bounded reconciliation sweep joins seals and attempts both ways.
"""

from __future__ import annotations

import copy
import gzip
import json
import pathlib
from typing import Any, Dict

import pytest

from ouroboros import model_send_seal as seal_mod
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient, _canonical_candidate_bytes


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.delenv("OUROBOROS_OBSERVABILITY_KEEP_RAW", raising=False)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *args, **kwargs: 0.01)
    (root / "state").mkdir(parents=True)
    yield root


class _Response:
    _payload = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 2, "cost": 0.0},
    }

    def model_dump(self):
        return copy.deepcopy(self._payload)


def _scope(root: pathlib.Path, task_id: str) -> ua.UsageScope:
    return ua.UsageScope(
        drive_root=root, task_id=task_id, root_task_id=task_id,
        category="task", source="test.model_send",
    )


def _target() -> Dict[str, Any]:
    return {"provider": "openai", "usage_model": "openai/gpt-x", "resolved_model": "gpt-x"}


def _payload() -> Dict[str, Any]:
    return {
        "model": "gpt-x",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "OPENAI_API_KEY=sk-secret-model-send-seal summarize this",
            }],
        }],
        "tools": [{
            "type": "function",
            "function": {"name": "inspect", "parameters": {"type": "object"}},
        }],
        "max_tokens": 32,
    }


def _rows(root: pathlib.Path):
    path = root / ua.LEDGER_REL
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _dispatch(root: pathlib.Path, task_id: str, payload=None, target=None):
    client = LLMClient(api_key="unused")
    with ua.usage_scope(_scope(root, task_id)):
        client._create_chat_completion_with_retries(
            lambda **_candidate: _Response(),
            payload if payload is not None else _payload(),
            target if target is not None else _target(),
        )
    return _rows(root)[-1]


def _manifest(final: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(pathlib.Path(final["candidate_manifest_ref"]["path"]).read_text())


def _violation_files(root: pathlib.Path):
    calls = root / "observability" / "calls"
    return sorted(calls.glob("*/*.model_send_violation.json")) if calls.is_dir() else []


def _violation_events(root: pathlib.Path):
    path = root / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("type") == seal_mod.VIOLATION_EVENT_TYPE
    ]


# ---------------------------------------------------------------------------
# Forward: seal + reconstruction on the ordinary call
# ---------------------------------------------------------------------------


def test_ordinary_call_seals_the_send_copy_and_reconstructs_clean(data_root):
    final = _dispatch(data_root, "task-seal-clean")
    ref = final["candidate_manifest_ref"]
    assert ref["model_send_seal_version"] == seal_mod.SEAL_VERSION

    seal = _manifest(final)["model_send_seal"]
    assert seal["seal_version"] == seal_mod.SEAL_VERSION
    assert seal["canonical_basis"] == seal_mod.CANONICAL_BASIS
    assert seal["pre_redaction_sha256"] == final["candidate_raw_sha256"]
    assert seal["size_bytes"] == final["candidate_raw_size_bytes"]
    assert seal["attempt_id"] == final["attempt_id"]

    classes = [row["class"] for row in seal["exclusions"]]
    # Every disclosed instance uses the closed vocabulary.
    assert set(classes) <= seal_mod.EXCLUSION_CLASSES
    # The secret in the payload fired a per-instance redaction row with a path.
    redaction_rows = [r for r in seal["exclusions"] if r["class"] == "secret_redaction"]
    assert redaction_rows and all(r.get("path") for r in redaction_rows)
    # Class-level structural rows are always disclosed (note §4): the transport
    # adds envelope below the seam and provider-side transforms are unobservable.
    assert "transport_envelope" in classes
    assert "provider_side_transform" in classes

    # The invariant held: no typed fact anywhere, dispatch settled.
    assert final["state"] == "settled"
    assert not _violation_files(data_root)
    assert not _violation_events(data_root)


def test_the_closed_exclusion_enum_is_exactly_the_four_note_classes():
    assert seal_mod.EXCLUSION_CLASSES == {
        "secret_redaction", "provider_native_custody",
        "transport_envelope", "provider_side_transform",
    }
    assert seal_mod.MODEL_SEND_SEAL_UNOBSERVED == "unobserved"


def test_provider_native_custody_exclusion_is_disclosed_per_item(data_root):
    from ouroboros.anthropic_native_custody import ANTHROPIC_NATIVE_RECEIPT_KEY

    payload = _payload()
    payload["messages"].insert(0, {
        "role": "assistant",
        "content": [{"type": "text", "text": "prior turn"}],
        ANTHROPIC_NATIVE_RECEIPT_KEY: {
            "content_json": "[]", "content_sha256": "a" * 64, "tool_use_ids": [],
        },
    })
    final = _dispatch(data_root, "task-seal-custody", payload=payload)
    seal = _manifest(final)["model_send_seal"]
    custody_rows = [
        row for row in seal["exclusions"] if row["class"] == "provider_native_custody"
    ]
    assert custody_rows, seal["exclusions"]
    assert any(row.get("opaque_sha256") == "a" * 64 for row in custody_rows)
    assert all(row.get("path", "").startswith("$.messages") for row in custody_rows)
    # Both sides of the compare apply the same custody projection: still clean.
    assert not _violation_files(data_root)


def test_delegated_lane_carries_the_disclosed_unobserved_limit(data_root):
    with ua.usage_scope(_scope(data_root, "task-delegated")):
        ua.record_subscription_session(
            "session-cpl5", route="some-route", model="model-x",
        )
    row = _rows(data_root)[-1]
    assert row["kind"] == "subscription_session"
    assert row["model_send_seal"] == seal_mod.MODEL_SEND_SEAL_UNOBSERVED


def test_retry_rungs_carry_their_own_seals(data_root, monkeypatch):
    """§5.3: the invariant is per PHYSICAL attempt — each ladder rung's product
    is a new candidate with its own seal and attempt id."""
    client = LLMClient(api_key="unused")
    sends = []

    class _Rejected(RuntimeError):
        status_code = 400
        body = {"error": {"code": "unsupported_parameter", "type": "invalid_request_error"}}

    def create(**candidate):
        sends.append(copy.deepcopy(candidate))
        if len(sends) == 1:
            raise _Rejected("temperature unsupported")
        return _Response()

    def retry_without_temperature(candidate, model, exc):
        retry = copy.deepcopy(candidate)
        retry.pop("temperature", None)
        return retry

    monkeypatch.setattr(client, "_retry_without_optional_sampling", retry_without_temperature)
    monkeypatch.setattr(client, "_openrouter_signature_retry_kwargs", lambda *args: None)
    payload = {**_payload(), "temperature": 0.7}
    with ua.usage_scope(_scope(data_root, "task-rungs")):
        client._create_chat_completion_with_retries(create, payload, _target())

    finals: Dict[str, Dict[str, Any]] = {}
    for row in _rows(data_root):
        finals[row["attempt_id"]] = row
    seals = [_manifest(row)["model_send_seal"] for row in finals.values()]
    assert len(seals) == 2
    assert seals[0]["attempt_id"] != seals[1]["attempt_id"]
    assert seals[0]["pre_redaction_sha256"] != seals[1]["pre_redaction_sha256"]
    for attempt_id, row in finals.items():
        seal = _manifest(row)["model_send_seal"]
        assert seal["attempt_id"] == attempt_id
        assert seal["pre_redaction_sha256"] == row["candidate_raw_sha256"]
    assert not _violation_files(data_root)


# ---------------------------------------------------------------------------
# Forward: deliberately damaged durable records → typed facts, never blocks
# ---------------------------------------------------------------------------


def _tampering_persist(monkeypatch, tamper):
    """Wrap the real persist so ``tamper(manifest_path)`` corrupts the durable
    record between the write and the seam's read-back verification."""
    from ouroboros import observability

    real = observability.persist_physical_candidate

    def wrapped(*args, **kwargs):
        persisted = real(*args, **kwargs)
        tamper(pathlib.Path(persisted["manifest_ref"]["path"]))
        return persisted

    monkeypatch.setattr(observability, "persist_physical_candidate", wrapped)


def test_corrupted_blob_yields_reconstruction_fact_and_does_not_block(data_root, monkeypatch):
    def corrupt_blob(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        blob = pathlib.Path(manifest["full_payload_ref"]["path"])
        with gzip.open(blob, "wb") as handle:
            handle.write(b'{"tampered": true}')

    _tampering_persist(monkeypatch, corrupt_blob)
    final = _dispatch(data_root, "task-corrupt-blob")

    # Observability invariant: the call itself was NOT blocked.
    assert final["state"] == "settled"
    files = _violation_files(data_root)
    assert len(files) == 1
    fact = json.loads(files[0].read_text())
    assert fact["type"] == seal_mod.VIOLATION_EVENT_TYPE
    assert fact["kind"] == "reconstruction_divergence"
    assert fact["divergence_class"] == "record_corrupt"
    assert fact["attempt_id"] == final["attempt_id"]
    assert isinstance(fact["first_divergent_offset"], int)
    # Digests and offsets only — no payload/secret bytes in the fact.
    assert "sk-secret" not in json.dumps(fact)
    events = _violation_events(data_root)
    assert len(events) == 1 and events[0]["kind"] == "reconstruction_divergence"


def test_tampered_seal_digest_yields_content_divergence_fact(data_root, monkeypatch):
    def rewrite_seal(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        manifest["model_send_seal"]["pre_redaction_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _tampering_persist(monkeypatch, rewrite_seal)
    final = _dispatch(data_root, "task-tampered-digest")
    assert final["state"] == "settled"
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert fact["kind"] == "content_divergence"
    assert fact["divergence_class"] == "sdk_mutation"
    assert fact["expected"]["sha256"] == "0" * 64
    assert fact["observed"]["sha256"] == final["candidate_raw_sha256"]


def test_missing_seal_block_yields_reconstruction_fact(data_root, monkeypatch):
    def drop_seal(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("model_send_seal", None)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _tampering_persist(monkeypatch, drop_seal)
    final = _dispatch(data_root, "task-missing-seal")
    assert final["state"] == "settled"
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert (fact["kind"], fact["divergence_class"]) == (
        "reconstruction_divergence", "seal_unreadable",
    )


def test_undisclosed_exclusion_class_is_a_violation_of_the_closed_enum(data_root, monkeypatch):
    """§4: anything outside the enum is IN the byte domain — an exclusion row
    claiming a class the enum does not know is undisclosed by definition."""
    def smuggle_class(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        manifest["model_send_seal"]["exclusions"].append({"class": "quiet_new_transform"})
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _tampering_persist(monkeypatch, smuggle_class)
    final = _dispatch(data_root, "task-undisclosed")
    assert final["state"] == "settled"
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert (fact["kind"], fact["divergence_class"]) == (
        "reconstruction_divergence", "undisclosed_exclusion",
    )


def test_foreign_serializer_basis_is_never_reinterpreted(data_root, monkeypatch):
    """§5.4: ``canonical_json_v1`` is the only equality basis; a seal claiming a
    different basis is reported, not re-read under this one."""
    def rewrite_basis(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        manifest["model_send_seal"]["canonical_basis"] = "canonical_json_v2"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _tampering_persist(monkeypatch, rewrite_basis)
    _dispatch(data_root, "task-basis")
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert (fact["kind"], fact["divergence_class"]) == (
        "reconstruction_divergence", "serializer_basis",
    )
    assert fact["expected"]["basis"] == "canonical_json_v2"
    assert fact["observed"]["basis"] == "canonical_json_v1"


# ---------------------------------------------------------------------------
# Cost shape: the compare must not re-serialize without need
# ---------------------------------------------------------------------------


def test_verification_adds_one_projection_and_no_extra_raw_serialization(data_root, monkeypatch):
    """The note budgets ONE read-back and ONE projection per attempt: the raw
    candidate is never serialized again for the compare (the seam digests are
    reused), and the CAS-basis projection pipeline runs exactly once more than
    persist alone (persist: 1, verify: 1)."""
    from ouroboros import llm_attempt, observability

    canonical_calls = []
    real_canonical = llm_attempt._canonical_candidate_bytes
    monkeypatch.setattr(
        llm_attempt, "_canonical_candidate_bytes",
        lambda payload: canonical_calls.append(1) or real_canonical(payload),
    )
    redact_calls = []
    real_redact = observability.redact_projection
    monkeypatch.setattr(
        observability, "redact_projection",
        lambda value: redact_calls.append(1) or real_redact(value),
    )
    _dispatch(data_root, "task-cost-shape")
    # 2 per _attempt_request (raw + context) × 2 builds (request + seam
    # identity re-check) — verification adds ZERO raw serializations.
    assert len(canonical_calls) == 4, len(canonical_calls)
    # persist_call redacts once for the CAS write; verification reconstructs
    # once. Nothing else may redact this candidate.
    assert len(redact_calls) == 2, len(redact_calls)
    assert not _violation_files(data_root)


# ---------------------------------------------------------------------------
# Reverse: the bounded reconciliation sweep
# ---------------------------------------------------------------------------


def test_sweep_reports_clean_join_after_ordinary_calls(data_root):
    _dispatch(data_root, "task-sweep-clean")
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["seals"] == 1
    assert report["sealed_attempts"] == 1
    assert report["orphan_seals"] == report["unlogged_attempts"] == 0
    assert report["facts_written"] == 0


def test_sweep_flags_a_synthetic_orphan_seal(data_root):
    from ouroboros.observability import persist_physical_candidate

    raw = _canonical_candidate_bytes({"model": "gpt-x", "messages": []})
    import hashlib

    persist_physical_candidate(
        data_root,
        task_id="task-orphan",
        attempt_id="f" * 32,
        candidate={"model": "gpt-x", "messages": []},
        candidate_facts={
            "candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_raw_size_bytes": len(raw),
            "candidate_measurement_kind": "canonical_json_v1",
        },
    )
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["orphan_seals"] == 1 and report["facts_written"] == 1
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert fact["kind"] == "orphan_seal"
    assert fact["attempt_id"] == "f" * 32
    assert fact["divergence_class"] == "missing_attempt_row"
    # Facts, never repairs: the seal manifest is still there.
    assert list((data_root / "observability" / "calls" / "task-orphan").glob("*.json"))


def test_sweep_flags_a_dispatched_attempt_whose_seal_vanished(data_root):
    final = _dispatch(data_root, "task-sweep-unlogged")
    pathlib.Path(final["candidate_manifest_ref"]["path"]).unlink()
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["unlogged_attempts"] == 1 and report["facts_written"] == 1
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert fact["kind"] == "unlogged_attempt"
    assert fact["attempt_id"] == final["attempt_id"]
    assert fact["expected"]["sha256"] == final["candidate_raw_sha256"]


def test_sweep_is_idempotent_and_never_floods_the_events_plane(data_root):
    final = _dispatch(data_root, "task-sweep-dedup")
    pathlib.Path(final["candidate_manifest_ref"]["path"]).unlink()
    first = seal_mod.reconcile_model_send_seals(data_root)
    second = seal_mod.reconcile_model_send_seals(data_root)
    assert first["facts_written"] == 1
    assert second["facts_written"] == 0
    assert len(_violation_events(data_root)) == 1


def test_sweep_skips_promoted_child_manifests(data_root):
    """A manifest promoted from a child drive joins the CHILD ledger — reading
    it as an orphan here would be a false accusation."""
    import hashlib

    from ouroboros.observability import persist_physical_candidate

    raw = _canonical_candidate_bytes({"model": "gpt-x", "messages": []})
    persisted = persist_physical_candidate(
        data_root,
        task_id="task-promoted",
        attempt_id="e" * 32,
        candidate={"model": "gpt-x", "messages": []},
        candidate_facts={
            "candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_raw_size_bytes": len(raw),
            "candidate_measurement_kind": "canonical_json_v1",
        },
    )
    path = pathlib.Path(persisted["manifest_ref"]["path"])
    manifest = json.loads(path.read_text())
    manifest["promoted_call_manifest"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["seals"] == 0
    assert report["orphan_seals"] == 0 and report["facts_written"] == 0


def test_promotion_marks_the_canonical_copy(data_root, tmp_path):
    """The promotion path itself stamps the provenance marker the sweep trusts."""
    import hashlib

    from ouroboros.observability import persist_physical_candidate, promote_call_manifest_ref

    child = tmp_path / "child-drive"
    child.mkdir()
    raw = _canonical_candidate_bytes({"model": "gpt-x", "messages": []})
    persisted = persist_physical_candidate(
        child,
        task_id="task-child",
        attempt_id="d" * 32,
        candidate={"model": "gpt-x", "messages": []},
        candidate_facts={
            "candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_raw_size_bytes": len(raw),
            "candidate_measurement_kind": "canonical_json_v1",
        },
    )
    source_ref = {
        key: value for key, value in persisted["manifest_ref"].items()
        if key in {"path", "call_id", "sha256"}
    }
    promoted = promote_call_manifest_ref(
        child, data_root, source_ref, task_id="task-child",
    )
    manifest = json.loads(pathlib.Path(promoted["path"]).read_text())
    assert manifest["promoted_call_manifest"] is True
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["orphan_seals"] == 0 and report["facts_written"] == 0


def test_sweep_skips_every_conclusion_on_an_unreadable_ledger(data_root):
    _dispatch(data_root, "task-sweep-corrupt-ledger")
    (data_root / ua.LEDGER_REL).write_text("not-json\n{}\n", encoding="utf-8")
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report == {
        "seals": 0, "sealed_attempts": 0,
        "orphan_seals": 0, "unlogged_attempts": 0,
        "facts_written": 0, "truncated": False,
    }


def test_sweep_rides_the_startup_family(data_root, monkeypatch):
    import ouroboros.server_maintenance as maintenance

    monkeypatch.setattr(maintenance, "DATA_DIR", data_root)
    calls = []
    monkeypatch.setattr(
        seal_mod, "reconcile_model_send_seals",
        lambda root, **kwargs: calls.append(pathlib.Path(root)) or {},
    )
    # Neighbour sweep steps degrade fail-soft on this synthetic root.
    maintenance._startup_custody_sweep()
    assert calls == [data_root]
