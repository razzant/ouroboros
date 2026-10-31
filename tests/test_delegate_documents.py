"""Selected task documents reach delegated runs as engine-owned immutable refs (C505).

The engine double follows the keyed-import wire of the C505 engine contract
(``POST /v2/uploads/imports`` + ``PUT /v2/uploads/:id/bytes``, ``expectedStoreId`` /
``storeId``) behind the REAL gateway methods; the producer is the real
``_delegate_start`` / ``exact_start`` / scheduling path, and every custody fact is read
back from the durable store and chain.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from types import SimpleNamespace
from urllib.parse import unquote

import httpx
import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros.config import CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION
from ouroboros.delegate_documents import (
    DOCUMENT_IMPORT_REQUESTED, carried_receipts, material_id, task_document_imports,
)
from ouroboros.delegate_pending import request_body
from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable
from tests._delegated_transport_shared import (  # noqa: F401
    _delegating_ctx,
    _gateway,
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests import _delegate_plan_shared

plan_setup = _delegate_plan_shared.plan_setup

STORE = "11111111-2222-4333-8444-555555555555"
IMPORT_DESCRIPTOR = {"id": "post:uploads.imports", "method": "POST", "path": "/v2/uploads/imports",
                     "requestSchema": "ControlUploadImportRequest"}
BRIEF = b"# Brief\nThe attached plan is version one.\n"


class DocumentEngine:
    """The C505 keyed import over one store, with explicit fault points."""

    engine_version = CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION
    start_run = ClaudexorGateway.start_run
    import_upload = ClaudexorGateway.import_upload
    put_upload_bytes = ClaudexorGateway.put_upload_bytes

    def __init__(self, store=STORE, *, catalog=True, mimes=("text/plain", "text/markdown", "application/json")):
        self.store_identity = store
        self.catalog = [dict(IMPORT_DESCRIPTOR)] if catalog else []
        self.inputs = [{"kind": "file", "mime_types": list(mimes), "max_bytes": 65_536, "max_count": 4,
                        "transport": "file_path"}]
        self.bindings, self.uploads, self.resources, self.runs = {}, {}, {}, {}
        self.calls, self.faults = [], []

    def handshake(self, **_kw): return {}
    def operations(self, **_kw):
        self.calls.append(("operations",))
        return copy.deepcopy(self.catalog)
    def agent_capabilities(self, **_kw):
        return {"runControlKeys": ["continueFrom"], "harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                               "accessProfilesSupported": ["readonly", "workspace_write"],
                               "attachmentInputs": copy.deepcopy(self.inputs)}]}
    def quota_snapshots(self): return []
    def find_project_id(self, root): return "prj-existing"
    def register_project(self, root): return "prj-new"
    def close(self): pass

    def of(self, kind):
        return [call for call in self.calls if call[0] == kind]

    def _take(self, kind):
        if self.faults and self.faults[0][0] == kind:
            return self.faults.pop(0)[1]
        return ""

    def _request(self, method, path, *, json_body=None, headers=None, content_body=None, **_kw):
        key = (headers or {}).get("Idempotency-Key", "")
        if (method, path) == ("POST", "/v2/uploads/imports"):
            return self._import(copy.deepcopy(json_body), key)
        if method == "PUT" and path.startswith("/v2/uploads/") and path.endswith("/bytes"):
            assert headers == {"Content-Type": "application/octet-stream"}
            return self._bytes(unquote(path.split("/")[3]), content_body)
        if (method, path) == ("POST", "/v2/runs"):
            return self._run(copy.deepcopy(json_body), key)
        raise AssertionError(f"unexpected engine call {method} {path}")

    def _import(self, body, key):
        self.calls.append(("import", key, body))
        fault = self._take("import")
        if fault == "drop":
            raise ClaudexorUnavailable("daemon_unreachable", "the request never reached the daemon")
        if body.get("expectedStoreId") != self.store_identity:
            raise ClaudexorUnavailable("store_mismatch", "another store", status_code=409)
        semantic = {name: body.get(name) for name in ("kind", "mime", "name", "source")}
        binding = self.bindings.get(key)
        if binding is None:
            if "content" not in body:
                raise ClaudexorUnavailable("upload_import_not_found", "no retained binding", status_code=404)
            upload = f"upl_{len(self.uploads) + 1}"
            binding = self.bindings[key] = {"semantic": semantic, "content": dict(body["content"]),
                                            "upload": upload, "state": "open", "resource": None}
            self.uploads[upload] = key
        elif binding["semantic"] != semantic:
            raise ClaudexorUnavailable("idempotency_conflict", "other arguments", status_code=409)
        elif "content" in body and body["content"] != binding["content"]:
            raise ClaudexorUnavailable("upload_import_content_conflict", "other content", status_code=409)
        if binding["state"] == "uploaded":  # complete verified bytes publish inside the replay
            rid = f"res_{len(self.resources) + 1}"
            self.resources[rid] = dict(binding["content"])
            binding.update(state="ready", resource={"resourceId": rid, **{k: semantic[k] for k in ("kind", "mime", "name")},
                                                    "sha256": binding["content"]["sha256"],
                                                    "sizeBytes": binding["content"]["sizeBytes"],
                                                    "createdAt": "2026-10-27T00:00:00Z", "deduplicated": False})
        answer = {"storeId": self.store_identity, "state": binding["state"], "uploadId": binding["upload"],
                  "content": dict(binding["content"]), "receivedBytes": 0,
                  **({"resource": dict(binding["resource"])} if binding["resource"] else {})}
        if fault == "lose":
            raise ClaudexorUnavailable("daemon_unreachable", "the answer was lost")
        return answer

    def _bytes(self, upload, data):
        self.calls.append(("put", upload, bytes(data)))
        fault = self._take("put")
        if fault == "drop":
            raise ClaudexorUnavailable("daemon_unreachable", "the bytes never reached the daemon")
        binding = self.bindings[self.uploads[upload]]
        if binding["state"] != "open":
            raise ClaudexorUnavailable("upload_not_open", binding["state"], status_code=409)
        if ("sha256:" + hashlib.sha256(data).hexdigest(), len(data)) != (
                binding["content"]["sha256"], binding["content"]["sizeBytes"]):
            raise ClaudexorUnavailable("digest_mismatch", "bytes differ", status_code=409)
        binding["state"] = "uploading" if fault == "stall" else "uploaded"  # stall: still verifying the bytes
        if fault == "lose":
            raise ClaudexorUnavailable("daemon_unreachable", "the answer was lost")
        return {"uploadId": upload, "state": "uploaded", "receivedBytes": len(data), "expectedBytes": len(data)}

    def _run(self, body, key):
        self.calls.append(("run", key, body))
        for ref in body.get("attachments") or []:
            if set(ref) != {"resourceId"} or ref["resourceId"] not in self.resources:
                raise ClaudexorUnavailable("resource_not_found", json.dumps(ref), status_code=404)
        if key in self.runs and self.runs[key]["body"] != body:
            raise ClaudexorUnavailable("idempotency_conflict", "a retry changed the run body", status_code=409)
        run = self.runs.setdefault(key, {"run": f"run-{len(self.runs) + 1}", "body": body})["run"]
        if self._take("run") == "lose":
            raise ClaudexorUnavailable("daemon_unreachable", "the answer was lost")
        return {"runId": run, "runDir": f"/tmp/{run}"}


@pytest.fixture
def docs(tmp_path, monkeypatch):
    """A delegating task with two staged documents and one staged image, and the engine double."""
    from ouroboros.artifacts import attachment_manifest_projection, stage_task_attachments
    from ouroboros.gateways import claudexor as gateway_module

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    _select_transport_access(monkeypatch, "readonly")  # a read-only reader of its task inputs
    ctx = _delegating_ctx(tmp_path, acting=False, task_id="t-docs")
    sources = tmp_path / "owner-files"
    sources.mkdir()
    (sources / "brief.md").write_bytes(BRIEF)
    (sources / "notes.txt").write_bytes(b"plain notes\n")
    (sources / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    rows = stage_task_attachments(tmp_path, "t-docs", [
        {"path": str(sources / name), "label": name} for name in ("brief.md", "notes.txt", "photo.png")])
    assert [row["status"] for row in rows] == ["staged"] * 3, rows
    ctx.task_metadata = {**ctx.task_metadata,
                         "task_contract": attachment_manifest_projection(tmp_path, "t-docs", rows)}
    engine = DocumentEngine()
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: engine)
    return SimpleNamespace(ctx=ctx, engine=engine, root=tmp_path, rows={row["relpath"]: row for row in rows},
                           sources=sources)


def _start(ctx, *, memo=True, **kwargs):
    from ouroboros.tools import delegate

    if memo:  # concurrent callers keep the shared process memo (the durable rows decide)
        delegate._CUSTODY.clear()
    return json.loads(delegate._delegate_start(ctx, "Summarize the attached brief.", **kwargs).text)


def _chain(root, kind):
    return [row for row in custody._iter_rows(custody.event_log_path(root)) if row.get("type") == kind]


def _record(docs, path="attachments/brief.md", task_id="t-docs"):
    return task_document_imports(docs.root, task_id)[material_id(docs.rows[path])]


def _entry(docs, monkeypatch, ctx=None, **params):
    """The registered delegate_start handler, as a model's tool call reaches it."""
    from ouroboros import subagent_runtime
    from tests._shared import configure_test_subagent

    selected = configure_test_subagent(monkeypatch, subagent_id="docs-session", kind="agent_session",
                                       target="some-route=weak-model")
    custody._CUSTODY.clear()
    result = subagent_runtime.delegate_start_entry(ctx or docs.ctx, "Summarize the attached brief.",
                                                   subagent_id=selected, **params)
    return json.loads(result.text)


def _settle(docs, run_id):
    """The predecessor's observed terminal, as the continuation suites record it."""
    assert custody.emit(docs.root, custody.SETTLED, {"run_id": run_id, "task_id": "t-docs", "route": "some-route",
                                                     "state": "failed", "outcome_reason": "subscription_window_exhausted"})
    custody._CUSTODY.clear()


def _child(docs):
    """A declared child copy of brief.md made by the real scheduling materializer."""
    from ouroboros.tools.control_scheduling import _materialize_child_attachment_manifest

    authority, error = _materialize_child_attachment_manifest(
        docs.ctx.task_metadata["task_contract"], docs.root, "t-kid", owner_drive=docs.root, owner_task_id="t-docs",
        custody_drive=docs.root, selected=["attachments/brief.md"], declared=True)
    assert not error, error
    child = _delegating_ctx(docs.root, acting=False, task_id="t-kid")
    child.task_metadata = {**child.task_metadata, "task_contract": authority}
    return child, authority


def _retained(docs, path="attachments/brief.md"):
    from pathlib import Path

    return Path(docs.rows[path]["abs_path"])


# -- the ordinary path ------------------------------------------------------------------

def test_omitted_and_empty_selection_keep_the_canonical_body_and_touch_no_import(docs):
    omitted = _start(docs.ctx)
    empty = _start(docs.ctx, attachment_paths=[])
    assert omitted["status"] == empty["status"] == "started", (omitted, empty)
    first, second = (call[2] for call in docs.engine.of("run"))
    assert "attachments" not in first and first == second
    assert docs.engine.of("import") == docs.engine.of("operations") == []
    assert "documents" not in omitted and not _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)


def test_selected_document_is_imported_once_and_rides_the_recorded_run_body(docs):
    payload = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert payload["status"] == "started", payload
    (_, key, declaration), (_, _, replay) = docs.engine.of("import")
    row = docs.rows["attachments/brief.md"]
    assert declaration == {"expectedStoreId": STORE, "kind": "file", "mime": "text/markdown", "name": "brief.md",
                           "source": "ouroboros-task-input:t-docs/attachments/brief.md",
                           "content": {"sizeBytes": len(BRIEF), "sha256": f"sha256:{row['sha256']}"}}
    assert replay == declaration and key.startswith("ouroboros-document-")
    assert [call[2] for call in docs.engine.of("put")] == [BRIEF]
    (_, _, body), = docs.engine.of("run")
    assert body["attachments"] == [{"resourceId": "res_1"}]
    started, = _chain(docs.root, custody.START_REQUESTED)
    assert request_body(docs.root, started)["attachments"] == [{"resourceId": "res_1"}]
    fact, = started["document_attachments"]
    assert (fact["path"], fact["resource_id"], fact["store_id"], fact["import_key"], fact["origin"]) == (
        "attachments/brief.md", "res_1", STORE, key, "imported")
    assert payload["documents"] == [{"path": "attachments/brief.md", "resource_id": "res_1", "origin": "imported"}]
    record = _record(docs)
    assert (record["state"], record["import_key"], record["resource"]["resourceId"]) == ("ready", key, "res_1")
    # The intent row landed before the first import call: the chain orders them.
    kinds = [row["type"] for row in custody._iter_rows(custody.event_log_path(docs.root))]
    assert kinds.index("delegate_run_document_import_requested") < kinds.index("delegate_run_document_import_sending")


def test_a_start_refused_for_another_reason_imports_nothing(docs):
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"], continue_from="run-that-never-existed")
    assert refused["status"] == "refused" and refused["definitely_unrun"] is True, refused
    assert docs.engine.of("import") == [] and not _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)


def test_a_second_run_reuses_the_resource_by_lookup_without_reading_any_source(docs):
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    key = _record(docs)["import_key"]
    _retained(docs).unlink()
    (docs.sources / "brief.md").write_bytes(b"# Brief\nEdited after the import.\n")
    docs.engine.calls.clear()
    payload = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert payload["status"] == "started", payload
    (_, replay_key, lookup), = docs.engine.of("import")
    assert replay_key == key and "content" not in lookup and lookup["expectedStoreId"] == STORE
    assert docs.engine.of("put") == [] and len(docs.engine.resources) == 1
    assert docs.engine.of("run")[0][2]["attachments"] == [{"resourceId": "res_1"}]
    assert payload["documents"][0]["origin"] == "replayed"
    assert len(_chain(docs.root, DOCUMENT_IMPORT_REQUESTED)) == 1


# -- lost responses and source stability --------------------------------------------------

def test_a_lost_declaration_recovers_under_the_same_key_and_never_sends_changed_bytes(docs):
    docs.engine.faults = [("import", "lose")]
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_outcome_unknown" and refused["definitely_unrun"] is True
    assert docs.engine.of("run") == [] and "pending_invocation_id" not in refused
    key = _record(docs)["import_key"]
    assert _record(docs)["state"] == "send_possible"
    retained = _retained(docs)
    retained.write_bytes(b"# Brief\nA different edition.\n")  # the retained copy no longer matches
    changed = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert changed["reason"] == "document_source_unavailable", changed
    lookups = [call for call in docs.engine.of("import")[1:]]
    assert all(call[1] == key and "content" not in call[2] for call in lookups)
    assert docs.engine.of("put") == [] and docs.engine.of("run") == []
    retained.write_bytes(BRIEF)
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    assert {call[1] for call in docs.engine.of("import")} == {key}
    assert [call[2] for call in docs.engine.of("put")] == [BRIEF] and len(docs.engine.bindings) == 1


def test_lost_bytes_answer_publishes_on_replay_after_the_source_is_gone(docs):
    docs.engine.faults = [("put", "lose")]
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["reason"] == "document_import_outcome_unknown"
    _retained(docs).unlink()
    payload = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert payload["status"] == "started", payload
    assert len(docs.engine.of("put")) == 1 and len(docs.engine.resources) == 1


def test_retry_of_replays_the_recorded_body_without_reimporting(docs):
    docs.engine.faults = [("run", "lose")]
    unknown = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    pending = unknown["pending_invocation_id"]
    imports = len(docs.engine.of("import"))
    retried = _start(docs.ctx, retry_of=pending)
    assert retried["status"] == "started" and retried["idempotent_recovery"] is True
    first, second = docs.engine.of("run")
    assert first[1] == second[1] == pending and first[2] == second[2]
    assert second[2]["attachments"] == [{"resourceId": "res_1"}]
    assert len(docs.engine.of("import")) == imports and "documents" not in retried


def _parallel_starts(docs, monkeypatch, *, wait_for_lookup):
    """Pause the first start before its first declaration, then let a second start run."""
    import threading
    import time

    from ouroboros import delegate_documents

    entered, release, results = threading.Event(), threading.Event(), []
    real_post = delegate_documents._post

    def paused_post(gateway, drive, facts, record, fresh):
        if fresh and not entered.is_set():
            entered.set()
            release.wait(20)
        return real_post(gateway, drive, facts, record, fresh)

    def start():
        results.append(_start(docs.ctx, memo=False, attachment_paths=["attachments/brief.md"]))

    monkeypatch.setattr(delegate_documents, "_post", paused_post)
    first = threading.Thread(target=start)
    first.start()
    assert entered.wait(20)
    second = threading.Thread(target=start)
    second.start()
    deadline = time.monotonic() + (20 if wait_for_lookup else 0.5)
    while time.monotonic() < deadline and not (wait_for_lookup and docs.engine.of("import")):
        time.sleep(0.02)
    release.set()
    first.join(30)
    second.join(30)
    return results


def test_a_parallel_start_waits_for_the_first_declaration_instead_of_reading_a_gap(docs, monkeypatch):
    results = _parallel_starts(docs, monkeypatch, wait_for_lookup=False)
    assert [payload["status"] for payload in results] == ["started", "started"], results
    assert len(docs.engine.bindings) == 1 and len(docs.engine.resources) == 1
    assert "content" in docs.engine.of("import")[0][2]  # the first engine call is the declaration
    assert [call[2].get("attachments") for call in docs.engine.of("run")] == [[{"resourceId": "res_1"}]] * 2


def test_without_the_input_lock_the_parallel_lookup_would_record_a_false_gap(docs, monkeypatch):
    """Negative control for the lock above: the same interleaving, lock removed."""
    from contextlib import nullcontext

    from ouroboros import delegate_documents

    monkeypatch.setattr(delegate_documents, "_import_lock", lambda *a: nullcontext())
    results = _parallel_starts(docs, monkeypatch, wait_for_lookup=True)
    assert sorted(payload.get("reason", payload["status"]) for payload in results) == [
        "document_import_binding_gap", "started"], results


# -- finished keys, gaps and the explicit new intent ------------------------------------------------

def test_cancelled_import_refuses_by_default_and_an_explicit_new_intent_succeeds(docs):
    docs.engine.faults = [("put", "drop")]
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["reason"] == "document_import_outcome_unknown"
    old_key = _record(docs)["import_key"]
    docs.engine.bindings[old_key]["state"] = "cancelled"  # the operator cancelled the upload
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_finished" and refused["definitely_unrun"] is True
    assert _record(docs)["state"] == "cancelled"
    calls = len(docs.engine.calls)
    again = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert again["reason"] == "document_import_finished"
    assert [call[0] for call in docs.engine.calls[calls:]] == ["operations"]  # no import call under a finished key
    fresh = _start(docs.ctx, attachment_paths=["attachments/brief.md"], _document_intent="explicit-new-1")
    assert fresh["status"] == "started", fresh
    record = _record(docs)
    assert record["import_key"] != old_key and record["state"] == "ready" and record["intent"] == "explicit-new-1"
    assert [item["import_key"] for item in record["previous"]] == [old_key]
    assert record["previous"][0]["state"] == "cancelled"
    assert fresh["documents"] == [{"path": "attachments/brief.md", "resource_id": "res_1", "origin": "imported",
                                   "superseded": old_key}]
    assert [row["import_key"] for row in _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)] == [old_key, record["import_key"]]
    started = _chain(docs.root, custody.START_REQUESTED)[-1]
    assert started["document_import"] == "new"


def test_lost_answer_on_a_new_intent_recovers_the_same_key_without_a_duplicate(docs):
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    docs.engine.faults = [("import", "lose")]
    unknown = _start(docs.ctx, attachment_paths=["attachments/brief.md"], _document_intent="explicit-new-2")
    assert unknown["reason"] == "document_import_outcome_unknown"
    assert "reuse" in unknown["detail"] and "another resource" in unknown["detail"]
    new_key = _record(docs)["import_key"]
    # The same intent (a restarted host bootstrap) and a plain reuse start both replay it.
    for kwargs in ({"_document_intent": "explicit-new-2"}, {}):
        payload = _start(docs.ctx, attachment_paths=["attachments/brief.md"], **kwargs)
        assert payload["status"] == "started", payload
    assert _record(docs)["import_key"] == new_key and len(docs.engine.bindings) == 2
    assert len(docs.engine.resources) == 2 and len(_chain(docs.root, DOCUMENT_IMPORT_REQUESTED)) == 2
    assert _record(docs)["previous"][0]["state"] == "ready"  # the former receipt stays visible


def test_a_retained_binding_gap_refuses_and_mints_nothing_on_its_own(docs):
    docs.engine.faults = [("import", "drop")]
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["reason"] == "document_import_outcome_unknown"
    key = _record(docs)["import_key"]
    gap = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert gap["reason"] == "document_import_binding_gap" and "not proof" in gap["detail"]
    assert docs.engine.of("import")[-1][1] == key and "content" not in docs.engine.of("import")[-1][2]
    assert _record(docs)["state"] == "gap" and docs.engine.bindings == {} and docs.engine.of("put") == []
    assert len(_chain(docs.root, DOCUMENT_IMPORT_REQUESTED)) == 1


def test_an_answer_naming_other_bytes_is_refused_and_finishes_the_key(docs):
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    key = _record(docs)["import_key"]
    docs.engine.bindings[key]["content"] = {"sizeBytes": 7, "sha256": "sha256:" + "0" * 64}  # bound elsewhere
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_conflict" and refused["definitely_unrun"] is True
    assert (_record(docs)["state"], _record(docs)["code"]) == ("conflict", "content_mismatch")
    assert len(docs.engine.of("run")) == 1  # the second run never started with a foreign resource
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["reason"] == "document_import_finished"


@pytest.mark.parametrize("status,code,refusal,state", [
    (404, "resource_not_found", "document_import_binding_gap", "gap"),
    (500, "upload_import_binding_invalid", "document_import_binding_invalid", "damaged"),
])
def test_missing_resources_and_store_damage_are_typed_and_mint_nothing(docs, status, code, refusal, state):
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    key = _record(docs)["import_key"]

    def damaged(body, idempotency_key):
        docs.engine.calls.append(("import", idempotency_key, body))
        raise ClaudexorUnavailable(code, "engine store fact", status_code=status)

    docs.engine._import = damaged
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == refusal and refused["definitely_unrun"] is True, refused
    assert "content" not in docs.engine.of("import")[-1][2] and docs.engine.of("import")[-1][1] == key
    assert (_record(docs)["state"], _record(docs)["code"], _record(docs)["import_key"]) == (state, code, key)
    assert len(docs.engine.of("run")) == 1 and len(_chain(docs.root, DOCUMENT_IMPORT_REQUESTED)) == 1


def test_a_definite_no_effect_answer_leaves_a_fresh_intent_unsent(docs):
    real = docs.engine._import

    def store_changed_once(body, key):  # answered before any binding, as INTERFACE "Errors" requires
        docs.engine._import = real
        docs.engine.calls.append(("import", key, body))
        raise ClaudexorUnavailable("store_mismatch", "the daemon now serves another store", status_code=409)

    docs.engine._import = store_changed_once
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_refused" and "store_mismatch" in refused["detail"]
    key = _record(docs)["import_key"]
    assert _record(docs)["state"] == "minted"
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    assert list(docs.engine.bindings) == [key] and "content" in docs.engine.of("import")[1][2]


# -- capability and selection ---------------------------------------------------------------

def test_an_engine_without_keyed_import_refuses_an_explicit_selection_and_keeps_old_starts(docs):
    docs.engine.catalog = [{**IMPORT_DESCRIPTOR, "requestSchema": "ControlUploadCreateRequest"}]
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_unsupported" and refused["definitely_unrun"] is True
    assert docs.engine.of("import") == docs.engine.of("run") == []
    assert [row["reason"] for row in _chain(docs.root, "delegate_run_start_blocked")] == ["document_import_unsupported"]
    assert _start(docs.ctx)["status"] == "started"
    assert "attachments" not in docs.engine.of("run")[0][2]


def test_route_admission_and_store_identity_come_from_the_engine(docs):
    docs.engine.inputs = [{"kind": "file", "mime_types": ["application/pdf"], "max_bytes": 10, "max_count": 1}]
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_route_unsupported" and "text/markdown" in refused["detail"]
    docs.engine.inputs = [{"kind": "file", "mime_types": ["text/markdown", "text/plain"], "max_bytes": 65_536,
                           "max_count": 1}]
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md", "attachments/notes.txt"])["reason"] == \
        "document_route_unsupported"
    docs.engine.store_identity = ""
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["reason"] == "document_store_identity_unknown"
    assert docs.engine.of("import") == [] and not _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)


def test_a_recorded_octet_stream_markdown_row_still_refuses(docs):
    contract = docs.ctx.task_metadata["task_contract"]
    for row in contract["attachment_manifest"]:
        if row["relpath"] == "attachments/brief.md":
            row["mime"] = "application/octet-stream"  # a row staged before the label fix
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_route_unsupported" and "application/octet-stream" in refused["detail"]


@pytest.mark.parametrize("first", ["ready", "unknown"])
def test_a_store_change_never_rebinds_a_default_start_and_only_new_does(docs, first):
    """Default replay cannot change store. The A-bound intent refuses before any source read or
    import call (its retained copy is gone to prove it); attachment_import='new' rebinds to B."""
    if first == "unknown":
        docs.engine.faults = [("import", "lose")]
    outcome = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert outcome.get("reason", outcome["status"]) == {
        "ready": "started", "unknown": "document_import_outcome_unknown"}[first], outcome
    bound = _record(docs)
    retained = _retained(docs)
    retained.unlink()
    docs.engine.store_identity = "store-b"
    calls = len(docs.engine.calls)
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_store_mismatch" and refused["definitely_unrun"] is True, refused
    assert STORE in refused["detail"] and "store-b" in refused["detail"]
    assert [call[0] for call in docs.engine.calls[calls:]] == ["operations"]  # no lookup, begin, PUT or run
    assert _record(docs) == bound  # the A-bound key and store stay exactly as they were
    retained.write_bytes(BRIEF)
    fresh = _start(docs.ctx, attachment_paths=["attachments/brief.md"], _document_intent="explicit-store-b")
    assert fresh["status"] == "started", fresh
    record = _record(docs)
    assert (record["store_id"], record["request"]["expectedStoreId"], record["state"]) == ("store-b", "store-b", "ready")
    assert record["import_key"] != bound["import_key"]
    assert [(item["import_key"], item["store_id"]) for item in record["previous"]] == [(bound["import_key"], STORE)]
    declaration = next(call[2] for call in docs.engine.of("import") if call[1] == record["import_key"])
    assert declaration["content"]["sha256"] == f"sha256:{docs.rows['attachments/brief.md']['sha256']}"
    assert fresh["documents"][0]["superseded"] == bound["import_key"]


def test_an_inherited_receipt_from_another_store_refuses_until_an_explicit_new_import(docs):
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    parent_key = _record(docs)["import_key"]
    child, authority = _child(docs)
    copy_path = pathlib.Path(authority["attachment_manifest"][0]["abs_path"])
    copy_path.unlink()  # a refusal that read the source would answer document_source_unavailable instead
    docs.engine.store_identity = "store-b"
    calls = len(docs.engine.calls)
    refused = _start(child, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_store_mismatch", refused
    assert [call[0] for call in docs.engine.calls[calls:]] == ["operations"]
    inherited = _record(docs, task_id="t-kid")
    assert (inherited["import_key"], inherited["store_id"], inherited["origin"]) == (parent_key, STORE, "inherited")
    copy_path.write_bytes(BRIEF)
    fresh = _start(child, attachment_paths=["attachments/brief.md"], _document_intent="child-new-b")
    assert fresh["status"] == "started" and fresh["documents"][0]["origin"] == "imported", fresh
    assert _record(docs, task_id="t-kid")["previous"][-1]["import_key"] == parent_key


@pytest.mark.parametrize("field,value", [("mime", "application/pdf"), ("name", "other.md"), ("kind", "image"),
                                         ("purpose", "model")])
def test_a_ready_answer_with_other_bound_semantics_is_refused(docs, field, value):
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    docs.engine.bindings[_record(docs)["import_key"]]["resource"][field] = value
    refused = _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_conflict", refused
    assert _record(docs)["state"] == "conflict" and len(docs.engine.of("run")) == 1


def test_unreadable_parent_import_records_refuse_the_child_copy_before_it_exists(docs, monkeypatch):
    from ouroboros import delegate_documents
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.store import StoreError
    from ouroboros.tools import control
    from ouroboros.tools.control_scheduling import _materialize_child_attachment_manifest
    from tests._shared import configure_test_subagent

    def unreadable(drive, task_id):
        raise StoreError("custody store unreadable")

    monkeypatch.setattr(delegate_documents, "task_document_imports", unreadable)
    authority, error = _materialize_child_attachment_manifest(
        docs.ctx.task_metadata["task_contract"], docs.root, "t-kid", owner_drive=docs.root, owner_task_id="t-docs",
        custody_drive=docs.root, selected=["attachments/brief.md"], declared=True)
    assert authority == {} and error.startswith("document_import_custody_unavailable"), error
    assert not (task_artifact_dir_path(docs.root, "t-kid") / "attachments").exists()
    selected = configure_test_subagent(monkeypatch, kind="agent_session", target="some-route=model")
    ctx = _scheduler_ctx(docs)
    result = control._schedule_task(ctx, subagent_id=selected, objective="Read the brief", expected_output="Summary")
    assert "SUBTASK_ATTACHMENT_ERROR: document_import_custody_unavailable" in result
    assert not [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]


def test_a_continuation_through_the_tool_entry_reattaches_only_what_it_selects(docs, monkeypatch):
    first = _entry(docs, monkeypatch, attachment_paths=["attachments/brief.md"])
    assert first["status"] == "started", first
    _settle(docs, first["run_id"])
    imports, puts = len(docs.engine.of("import")), len(docs.engine.of("put"))
    kept = _entry(docs, monkeypatch, attachment_paths=["attachments/brief.md"], continue_from=first["run_id"])
    assert kept["status"] == "started" and kept["continuation"]["continuation_of"] == first["run_id"], kept
    body = docs.engine.of("run")[-1][2]
    assert body["continueFrom"] == first["run_id"] and body["attachments"] == [{"resourceId": "res_1"}]
    assert ["content" in call[2] for call in docs.engine.of("import")[imports:]] == [False]  # lookup-only reuse
    assert len(docs.engine.of("put")) == puts and len(docs.engine.resources) == 1
    started = _chain(docs.root, custody.START_REQUESTED)[-1]
    assert request_body(docs.root, started)["attachments"] == [{"resourceId": "res_1"}]
    assert kept["documents"] == [{"path": "attachments/brief.md", "resource_id": "res_1", "origin": "replayed"}]
    _settle(docs, kept["run_id"])
    imports = len(docs.engine.of("import"))
    omitted = _entry(docs, monkeypatch, continue_from=kept["run_id"])
    assert omitted["status"] == "started", omitted
    body = docs.engine.of("run")[-1][2]
    assert body["continueFrom"] == kept["run_id"] and "attachments" not in body
    assert len(docs.engine.of("import")) == imports and "documents" not in omitted


def test_each_explicit_new_invocation_mints_one_identity_and_reuse_recovers_it(docs, monkeypatch):
    assert _entry(docs, monkeypatch, attachment_paths=["attachments/brief.md"])["status"] == "started"
    first_key = _record(docs)["import_key"]
    docs.engine.faults = [("import", "lose")]
    lost = _entry(docs, monkeypatch, attachment_paths=["attachments/brief.md"], attachment_import="new")
    assert lost["reason"] == "document_import_outcome_unknown", lost
    lost_key = _record(docs)["import_key"]
    recovered = _entry(docs, monkeypatch, attachment_paths=["attachments/brief.md"])  # reuse: the same new key
    assert recovered["status"] == "started" and _record(docs)["import_key"] == lost_key, recovered
    again = _entry(docs, monkeypatch, attachment_paths=["attachments/brief.md"], attachment_import="new")
    assert again["status"] == "started", again
    keys = [row["import_key"] for row in _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)]
    assert keys == [first_key, lost_key, _record(docs)["import_key"]] and len(set(keys)) == 3
    assert len(docs.engine.bindings) == 3 and [item["import_key"] for item in _record(docs)["previous"]] == keys[:2]


def test_a_cold_restored_bootstrap_replays_its_saved_new_intent_without_minting(docs):
    """The host's first start dies after the engine accepted its declaration; a fresh worker
    reruns the real bootstrap and replays the intent saved with the task, minting nothing."""
    from ouroboros import subagent_bootstrap
    from ouroboros.subagents import DelegationRoute
    from tests._delegated_transport_shared import _transport_snapshot

    class WorkerDied(BaseException):
        pass

    accepted = docs.engine._import

    def dies_after_acceptance(body, key):
        accepted(body, key)
        docs.engine._import = accepted
        raise WorkerDied("the worker died before reading the answer")

    contract = {**docs.ctx.task_metadata["task_contract"], "objective": "Summarize the brief"}
    task = {"id": "t-docs", "objective": "Summarize the brief", "task_contract": contract,
            "configured_subagent": _transport_snapshot(DelegationRoute("some-route", "weak-model"), "readonly"),
            "attachment_paths": ["attachments/brief.md"], "attachment_import": "new",
            "document_import_intent": "intent-saved-at-schedule"}
    dispatch = SimpleNamespace(executor="harness", blocked=False, availability={})
    docs.engine._import = dies_after_acceptance
    with pytest.raises(WorkerDied):
        subagent_bootstrap.bootstrap_before_context(docs.ctx, task, dispatch)
    saved = _record(docs)
    assert (saved["state"], saved["intent"]) == ("send_possible", "intent-saved-at-schedule")
    assert docs.engine.of("run") == []
    custody._CUSTODY.clear()
    restored = _delegating_ctx(docs.root, acting=False, task_id="t-docs")  # a fresh worker's context
    restored.task_metadata = {**restored.task_metadata, "task_contract": contract}
    receipt = json.loads(subagent_bootstrap.bootstrap_before_context(restored, task, dispatch))
    assert receipt["status"] == "configured_session_started", receipt
    assert _record(docs)["import_key"] == saved["import_key"] and len(docs.engine.bindings) == 1
    assert ["content" in call[2] for call in docs.engine.of("import")] == [True, False, False]
    assert docs.engine.of("run")[0][2]["attachments"] == [{"resourceId": "res_1"}]
    assert len(_chain(docs.root, DOCUMENT_IMPORT_REQUESTED)) == 1


@pytest.mark.parametrize("spec,code", [
    ({"attachment_paths": ["attachments/photo.png"]}, "attachment_image_unsupported"),
    ({"attachment_paths": ["attachments/missing.md"]}, "attachment_path_unknown"),
    ({"attachment_paths": ["./attachments/brief.md"]}, "attachment_path_invalid"),
    ({"attachment_paths": ["/abs/attachments/brief.md"]}, "attachment_path_invalid"),
    ({"attachment_paths": "attachments/brief.md"}, "attachment_paths_invalid"),
    ({"attachment_paths": ["attachments/brief.md", "attachments/brief.md"]}, "attachment_paths_invalid"),
    ({"attachment_import": "new"}, "attachment_import_requires_selection"),
    ({"attachment_paths": ["attachments/brief.md"], "attachment_import": "fresh"}, "attachment_import_invalid"),
    ({"attachment_paths": ["attachments/brief.md"], "retry_of": "abc"}, "retry_selector_conflict"),
    ({"attachment_import": "reuse", "retry_of": "abc"}, "retry_selector_conflict"),
])
def test_exact_start_refuses_bad_selection_before_the_daemon(docs, monkeypatch, spec, code):
    from ouroboros import subagent_runtime
    from ouroboros.tools import delegate
    from tests._shared import configure_test_subagent

    selected = configure_test_subagent(monkeypatch, subagent_id="docs-session", kind="agent_session",
                                       target="some-route=weak-model")
    monkeypatch.setattr(delegate, "_delegate_start", lambda *a, **k: pytest.fail("the daemon path ran"))
    spec = {**spec, **({} if "retry_of" in spec else {"subagent_id": selected})}
    payload = json.loads(subagent_runtime.exact_start(docs.ctx, "Read the brief.", spec).text)
    assert payload["status"] == "refused" and payload["reason"] == code, payload


def test_exact_start_mints_one_new_intent_per_call_and_keeps_a_saved_one(docs, monkeypatch):
    from ouroboros import subagent_runtime
    from ouroboros.tools import delegate
    from tests._shared import configure_test_subagent

    selected = configure_test_subagent(monkeypatch, subagent_id="docs-session", kind="agent_session",
                                       target="some-route=weak-model")
    seen = []
    monkeypatch.setattr(delegate, "_delegate_start", lambda ctx, *a, **k: seen.append(k) or
                        delegate.delegate_result({"status": "started", "run_id": "r", "custody_durable": True}))
    for spec in ({"attachment_import": "new"}, {"attachment_import": "new"},
                 {"attachment_import": "new", "_document_intent": "saved-intent"},
                 {"attachment_import": "reuse", "_document_intent": "stray"}, {}):
        subagent_runtime.exact_start(docs.ctx, "Read.", {"subagent_id": selected,
                                                         "attachment_paths": ["attachments/brief.md"], **spec})
    intents = [call.get("_document_intent") for call in seen]
    assert intents[0] and intents[1] and intents[0] != intents[1]
    assert intents[2:] == ["saved-intent", None, None]
    assert [call.get("_document_first_start") for call in seen] == [False, False, True, None, None]
    assert all(call["attachment_paths"] == ["attachments/brief.md"] and "attachment_import" not in call for call in seen)


# -- children: scheduling, bootstrap and inherited receipts ------------------------------------

def _scheduler_ctx(docs):
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=docs.root / "repo", drive_root=docs.root)
    ctx.task_id, ctx.pending_events = "t-docs", []
    ctx.task_metadata = dict(docs.ctx.task_metadata)
    return ctx


def test_api_children_refuse_document_selection(docs, monkeypatch):
    from ouroboros.tools import control
    from tests._shared import configure_test_subagent

    selected = configure_test_subagent(monkeypatch, kind="api_model", target="openai/test-model")
    ctx = _scheduler_ctx(docs)
    for extra in ({"attachment_paths": ["attachments/brief.md"]}, {"attachment_import": "reuse"}):
        result = control._schedule_task(ctx, subagent_id=selected, objective="Read the brief",
                                        expected_output="A summary", **extra)
        assert "attachment_paths_unsupported" in result
    assert not [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]


@pytest.mark.parametrize("sources", ["shared", "declared"])
def test_scheduled_session_child_receives_selection_inputs_and_the_parents_receipt(docs, monkeypatch, sources):
    from ouroboros.subagent_work_order import compile_external_work_order
    from ouroboros.tools import control
    from tests._shared import configure_test_subagent
    from tests._store_shared import read_result

    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    parent_key = _record(docs)["import_key"]
    selected = configure_test_subagent(monkeypatch, kind="agent_session", target="some-route=model")
    ctx = _scheduler_ctx(docs)
    result = control._schedule_task(ctx, subagent_id=selected, objective="Read the brief",
                                    expected_output="A summary", attachment_paths=["attachments/brief.md"],
                                    attachment_import="new", input_sources=sources)
    events = [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]
    assert len(events) == 1, result
    saved = read_result(ctx.drive_root, events[0]["task_id"])
    assert saved["attachment_paths"] == ["attachments/brief.md"]
    assert saved["attachment_import"] == "new" and len(saved["document_import_intent"]) == 32
    rows = saved["task_contract"]["attachment_manifest"]
    expected = {"attachments/brief.md"} if sources == "declared" else set(docs.rows)
    assert {row["relpath"] for row in rows} == expected
    brief = next(row for row in rows if row["relpath"] == "attachments/brief.md")
    receipt, = brief["engine_imports"]
    assert (receipt["store_id"], receipt["import_key"], receipt["resource_id"]) == (STORE, parent_key, "res_1")
    order = compile_external_work_order(saved)
    assert "SELECTED DOCUMENTS" in order and "- attachments/brief.md" in order
    assert ("explicitly selected task documents" in order) is (sources == "declared")


def test_schedule_refuses_a_selection_the_parent_does_not_hold(docs, monkeypatch):
    from ouroboros.tools import control
    from tests._shared import configure_test_subagent

    selected = configure_test_subagent(monkeypatch, kind="agent_session", target="some-route=model")
    ctx = _scheduler_ctx(docs)
    result = control._schedule_task(ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                    attachment_paths=["attachments/photo.png"], input_sources="declared")
    assert "attachment_image_unsupported" in result
    result = control._schedule_task(ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                    attachment_import="new")
    assert "attachment_import='new' needs a nonempty attachment_paths" in result


def test_bootstrap_carries_only_the_saved_first_start_selection(plan_setup, monkeypatch):
    from ouroboros import subagent_bootstrap, subagent_runtime
    from ouroboros.delegate_shared import delegate_result

    p, calls = plan_setup, []
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda ctx, prompt, spec:
                        calls.append(spec) or delegate_result({"status": "started", "run_id": "initial",
                                                                "custody_durable": True}))
    task = {"id": p.ctx.task_id, "objective": "Read the brief", "configured_subagent": p.snapshot,
            "task_contract": {"objective": "Read the brief"}, "attachment_paths": ["attachments/brief.md"],
            "attachment_import": "new", "document_import_intent": "intent-saved-at-schedule"}
    result = json.loads(subagent_bootstrap.bootstrap_before_context(
        p.ctx, task, SimpleNamespace(executor="harness", blocked=False)))
    assert result["status"] == "configured_session_started", result
    spec, = calls
    assert (spec["attachment_paths"], spec["attachment_import"], spec["_document_intent"]) == (
        ["attachments/brief.md"], "new", "intent-saved-at-schedule")


def test_a_child_copy_reuses_the_parents_resource_through_its_receipt(docs):
    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    parent_key = _record(docs)["import_key"]
    child, _authority = _child(docs)
    docs.engine.calls.clear()
    payload = _start(child, attachment_paths=["attachments/brief.md"])
    assert payload["status"] == "started", payload
    (_, key, lookup), = docs.engine.of("import")
    assert key == parent_key and "content" not in lookup and docs.engine.of("put") == []
    assert payload["documents"] == [{"path": "attachments/brief.md", "resource_id": "res_1", "origin": "inherited"}]
    assert len(docs.engine.resources) == 1


def test_copying_only_images_reads_no_import_records_and_a_document_copy_does(docs, monkeypatch):
    """Images never become documents, so their copy reads no custody marks; a staged document's
    copy still reads them once even when nothing is selected (its receipts serve a later selection)."""
    from ouroboros import delegate_documents
    from ouroboros.store import StoreError
    from ouroboros.tools import control
    from tests._shared import configure_test_subagent
    from tests._store_shared import read_result

    reads = []

    def unreadable(drive, task_id):
        reads.append(task_id)
        raise StoreError("custody store unreadable")

    monkeypatch.setattr(delegate_documents, "task_document_imports", unreadable)
    selected = configure_test_subagent(monkeypatch, kind="api_model", target="openai/test-model")
    ctx = _scheduler_ctx(docs)
    contract = ctx.task_metadata["task_contract"]
    images = [row for row in contract["attachment_manifest"] if row["relpath"] == "attachments/photo.png"]
    ctx.task_metadata = {**ctx.task_metadata, "task_contract": {**contract, "attachment_manifest": images}}
    queued = control._schedule_task(ctx, subagent_id=selected, objective="Describe the photo",
                                    expected_output="A caption")
    events = [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]
    assert len(events) == 1 and reads == [], queued
    copied, = read_result(ctx.drive_root, events[0]["task_id"])["task_contract"]["attachment_manifest"]
    assert copied["relpath"] == "attachments/photo.png" and copied["is_image"] and "engine_imports" not in copied
    ctx.task_metadata = {**ctx.task_metadata, "task_contract": contract}
    refused = control._schedule_task(ctx, subagent_id=selected, objective="Read the notes", expected_output="A summary")
    assert "SUBTASK_ATTACHMENT_ERROR: document_import_custody_unavailable" in refused and reads == ["t-docs"]
    assert len([row for row in ctx.pending_events if row["type"] == "schedule_subagent"]) == 1


# -- a scheduled child's first start refused before any run ------------------------------------

_HARNESS = SimpleNamespace(executor="harness", blocked=False, availability={})


def _first_start_task(docs, mode):
    """The saved first-start selection, as admission hands it to the child's bootstrap."""
    from ouroboros.subagents import DelegationRoute
    from tests._delegated_transport_shared import _transport_snapshot

    contract = {**docs.ctx.task_metadata["task_contract"], "objective": "Summarize the brief"}
    return {"id": "t-docs", "objective": "Summarize the brief", "task_contract": contract,
            "configured_subagent": _transport_snapshot(DelegationRoute("some-route", "weak-model"), "readonly"),
            "attachment_paths": ["attachments/brief.md"], "attachment_import": mode,
            **({"document_import_intent": "intent-saved-at-schedule"} if mode == "new" else {})}


@pytest.mark.parametrize("fault,reason", [(("import", "lose"), "document_import_outcome_unknown"),
                                          (("put", "stall"), "document_import_in_flight")], ids=["lost", "stalled"])
@pytest.mark.parametrize("mode", ["reuse", "new"])
def test_an_unsettled_first_start_import_wakes_the_child_whose_next_start_keeps_the_key(docs, mode, fault, reason):
    """The host's first start is refused unrun with an import a later start settles. The child is
    woken (no $0 terminal, its route still available) and its next start, naming no selection of
    its own, is still the work order's first start: the saved selection and intent, the same key."""
    from ouroboros import subagent_bootstrap, subagent_runtime

    task = _first_start_task(docs, mode)
    docs.engine.faults = [fault]
    wake = subagent_bootstrap.bootstrap_before_context(docs.ctx, task, _HARNESS)
    assert wake, ("the host ended the child unrun", getattr(docs.ctx, "_configured_startup_refusal", None))
    receipt = json.loads(wake)
    assert receipt["status"] == "configured_session_startup_fault", receipt
    assert (receipt["startup"]["reason"], receipt["startup"]["definitely_unrun"]) == (reason, True)
    assert "subagent_availability" not in task and docs.engine.of("run") == []
    saved = _record(docs)
    assert saved["state"] == "send_possible" and saved.get("intent") == task.get("document_import_intent")
    if fault[0] == "put":
        docs.engine.bindings[saved["import_key"]]["state"] = "uploaded"  # the engine finished verifying
    custody._CUSTODY.clear()
    started = json.loads(subagent_runtime.delegate_start_entry(docs.ctx, "").text)
    assert started["status"] == "started", started
    (_, _, body), = docs.engine.of("run")
    assert body.get("attachments") == [{"resourceId": "res_1"}] and "- attachments/brief.md" in body["prompt"]
    assert {call[1] for call in docs.engine.of("import")} == {saved["import_key"]} and len(docs.engine.resources) == 1
    assert [call[2] for call in docs.engine.of("put")] == [BRIEF]
    assert _record(docs)["state"] == "ready" and len(_chain(docs.root, DOCUMENT_IMPORT_REQUESTED)) == 1


def test_a_busy_first_start_import_wakes_the_child_with_guidance_and_its_retry_mints_the_saved_intent(
        docs, monkeypatch):
    """Busy refuses before any custody read: nothing is minted, the child is woken and told the run is
    absent, and its retry mints exactly the intent saved at schedule time."""
    from ouroboros import delegate_documents, delegate_supervision, subagent_bootstrap, subagent_runtime

    real_lock = delegate_documents._import_lock

    def busy_once(*args):
        monkeypatch.setattr(delegate_documents, "_import_lock", real_lock)
        raise delegate_documents._refusal("document_import_busy", "Another start is importing this document now.")

    monkeypatch.setattr(delegate_documents, "_import_lock", busy_once)
    wake = subagent_bootstrap.bootstrap_before_context(docs.ctx, _first_start_task(docs, "new"), _HARNESS)
    assert wake, ("the host ended the child unrun", getattr(docs.ctx, "_configured_startup_refusal", None))
    assert json.loads(wake)["startup"]["reason"] == "document_import_busy" and docs.engine.of("import") == []
    monkeypatch.setattr(delegate_supervision, "acknowledge_pending_wake", lambda *_a, **_k: True)
    messages = []
    subagent_bootstrap.append_startup_receipt(SimpleNamespace(), messages, wake)
    assert "no run of this work order exists" in messages[0]["content"]
    assert "Physical custody is unresolved" not in messages[0]["content"]
    custody._CUSTODY.clear()
    started = json.loads(subagent_runtime.delegate_start_entry(docs.ctx, "").text)
    assert started["status"] == "started", started
    assert docs.engine.of("run")[0][2].get("attachments") == [{"resourceId": "res_1"}]
    (minted,) = _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)
    assert minted["intent"] == "intent-saved-at-schedule" and _record(docs)["state"] == "ready"


@pytest.mark.parametrize("own,attached", [(["attachments/brief.md"], True), ([], False)], ids=["reselected", "none"])
def test_a_woken_childs_own_selection_replaces_the_saved_one(docs, own, attached):
    """Naming a selection is the child's own decision: the same paths replay the saved key even
    after a saved new intent (no attachment_import means reuse), and [] starts without documents."""
    from ouroboros import subagent_bootstrap, subagent_runtime

    docs.engine.faults = [("import", "lose")]
    wake = subagent_bootstrap.bootstrap_before_context(docs.ctx, _first_start_task(docs, "new"), _HARNESS)
    assert wake, ("the host ended the child unrun", getattr(docs.ctx, "_configured_startup_refusal", None))
    key, imports = _record(docs)["import_key"], len(docs.engine.of("import"))
    custody._CUSTODY.clear()
    started = json.loads(subagent_runtime.delegate_start_entry(docs.ctx, "", attachment_paths=own).text)
    assert started["status"] == "started", started
    (_, _, body), = docs.engine.of("run")
    assert body.get("attachments") == ([{"resourceId": "res_1"}] if attached else None)
    assert {call[1] for call in docs.engine.of("import")} == {key}
    assert (len(docs.engine.of("import")) > imports) is attached
    assert len(_chain(docs.root, DOCUMENT_IMPORT_REQUESTED)) == 1


@pytest.mark.parametrize("choice", ["reuse", "new"])
def test_a_woken_childs_import_choice_alone_still_attaches_the_saved_documents(docs, choice):
    """Following the refusal detail, attachment_import without attachment_paths keeps the saved
    documents: reuse replays the saved key, while an explicit new begins one more import."""
    from ouroboros import subagent_bootstrap, subagent_runtime

    docs.engine.faults = [("import", "lose")]
    wake = subagent_bootstrap.bootstrap_before_context(docs.ctx, _first_start_task(docs, "new"), _HARNESS)
    assert wake, ("the host ended the child unrun", getattr(docs.ctx, "_configured_startup_refusal", None))
    saved = _record(docs)
    custody._CUSTODY.clear()
    started = json.loads(subagent_runtime.delegate_start_entry(docs.ctx, "", attachment_import=choice).text)
    assert started["status"] == "started", started
    (_, _, body), = docs.engine.of("run")
    assert body.get("attachments") == [{"resourceId": "res_1"}]
    minted = [row["import_key"] for row in _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)]
    assert minted[0] == saved["import_key"] and len(minted) == {"reuse": 1, "new": 2}[choice]
    assert _record(docs)["import_key"] == minted[-1] and len(docs.engine.bindings) == len(minted)


def test_a_restored_episode_that_started_nothing_still_carries_the_saved_documents(docs):
    """A pause or restart between the wake and the child's own start restores its episode, and the
    host starts nothing then; the child's next start is still the work order's first start."""
    from ouroboros import subagent_bootstrap, subagent_runtime

    task = _first_start_task(docs, "new")
    docs.engine.faults = [("import", "lose")]
    assert subagent_bootstrap.bootstrap_before_context(docs.ctx, task, _HARNESS)
    key = _record(docs)["import_key"]
    custody._CUSTODY.clear()
    restored = _delegating_ctx(docs.root, acting=False, task_id="t-docs")  # a fresh worker's context
    restored.task_metadata = {**restored.task_metadata, "task_contract": task["task_contract"]}
    receipt = json.loads(subagent_bootstrap.bootstrap_before_context(
        restored, {**task, "_working_recovery": {}}, _HARNESS))
    assert receipt["status"] == "configured_session_continuation", receipt
    assert receipt["continuation"]["delegated_runs_started"] == 0 and docs.engine.of("run") == []
    started = json.loads(subagent_runtime.delegate_start_entry(restored, "").text)
    assert started["status"] == "started", started
    assert docs.engine.of("run")[0][2].get("attachments") == [{"resourceId": "res_1"}]
    assert {call[1] for call in docs.engine.of("import")} == {key}
    assert len(_chain(docs.root, DOCUMENT_IMPORT_REQUESTED)) == 1


def _own_new_publication_lost(docs, ctx, paths):
    """The woken child's own attachment_import='new' start: each import is declared, sent and published,
    but the answer to the last publication is lost, so the start is refused with an unknown outcome."""
    from ouroboros import subagent_runtime

    docs.engine.faults = [*[("import", ""), ("put", ""), ("import", "")] * (len(paths) - 1),
                          ("import", ""), ("put", ""), ("import", "lose")]
    custody._CUSTODY.clear()
    lost = json.loads(subagent_runtime.delegate_start_entry(
        ctx, "", attachment_paths=paths, attachment_import="new").text)
    assert lost["reason"] == "document_import_outcome_unknown" and docs.engine.faults == [], lost


@pytest.mark.parametrize("episode", ["same", "restored"])
def test_an_omitted_retry_after_the_childs_own_new_import_keeps_its_key(docs, episode):
    """The saved new intent begins only an input's first import. After the child's own new import of
    it was published with the answer lost, an omitted retry restores the saved intent yet replays that
    current key: no third key and no second resource."""
    from ouroboros import subagent_bootstrap, subagent_runtime

    task = _first_start_task(docs, "new")
    docs.engine.faults = [("import", "lose")]
    assert subagent_bootstrap.bootstrap_before_context(docs.ctx, task, _HARNESS)
    first = _record(docs)["import_key"]
    _own_new_publication_lost(docs, docs.ctx, ["attachments/brief.md"])
    own = _record(docs)
    assert own["import_key"] != first and own["state"] == "send_possible" and len(docs.engine.resources) == 1
    ctx = docs.ctx
    if episode == "restored":
        custody._CUSTODY.clear()
        ctx = _delegating_ctx(docs.root, acting=False, task_id="t-docs")  # a fresh worker's context
        ctx.task_metadata = {**ctx.task_metadata, "task_contract": task["task_contract"]}
        receipt = json.loads(subagent_bootstrap.bootstrap_before_context(
            ctx, {**task, "_working_recovery": {}}, _HARNESS))
        assert receipt["status"] == "configured_session_continuation", receipt
    custody._CUSTODY.clear()
    started = json.loads(subagent_runtime.delegate_start_entry(ctx, "").text)
    assert started["status"] == "started", started
    assert started["documents"] == [{"path": "attachments/brief.md", "resource_id": "res_1", "origin": "replayed"}]
    assert [row["import_key"] for row in _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)] == [first, own["import_key"]]
    assert docs.engine.of("run")[0][2]["attachments"] == [{"resourceId": "res_1"}] and len(docs.engine.resources) == 1


@pytest.mark.parametrize("own", [["attachments/notes.txt", "attachments/brief.md"], ["attachments/brief.md"]],
                         ids=["all-reimported", "one-reimported"])
def test_an_omitted_retry_keeps_each_inputs_current_key_and_begins_only_a_missing_import(docs, own):
    """Mixed inputs: the omitted retry replays each input's current key, whether the saved intent or
    the child's own new import made it, and begins under the saved intent only an input never imported."""
    from ouroboros import subagent_bootstrap, subagent_runtime

    saved = ["attachments/brief.md", "attachments/notes.txt"]
    task = {**_first_start_task(docs, "new"), "attachment_paths": saved}
    docs.engine.faults = [("import", "lose")]  # brief's first declaration is lost; notes is never reached
    assert subagent_bootstrap.bootstrap_before_context(docs.ctx, task, _HARNESS)
    _own_new_publication_lost(docs, docs.ctx, own)
    keys = {path: _record(docs, path)["import_key"] for path in own}
    custody._CUSTODY.clear()
    started = json.loads(subagent_runtime.delegate_start_entry(docs.ctx, "").text)
    assert started["status"] == "started", started
    assert [(fact["path"], fact["origin"]) for fact in started["documents"]] == [
        (path, "replayed" if path in own else "imported") for path in saved]
    assert all(_record(docs, path)["import_key"] == key for path, key in keys.items())
    minted = _chain(docs.root, DOCUMENT_IMPORT_REQUESTED)
    assert len(minted) == 1 + len(own) + len(set(saved) - set(own)) and len(docs.engine.resources) == 2
    if "attachments/notes.txt" not in own:
        assert minted[-1]["intent"] == task["document_import_intent"]
    assert docs.engine.of("run")[0][2]["attachments"] == [
        {"resourceId": _record(docs, path)["resource"]["resourceId"]} for path in saved]


@pytest.mark.parametrize("answer,reason", [
    ({"state": "published"}, "document_import_outcome_unknown"),
    ({"state": []}, "document_import_outcome_unknown"),
    ({"state": {}}, "document_import_outcome_unknown"),
    ({"uploadId": 123}, "document_import_outcome_unknown"),
    ({"uploadId": ["upload"]}, "document_import_outcome_unknown"),
    ({"state": "ready", "resource": {"resourceId": 123}}, "document_import_outcome_unknown"),
    ({"state": "ready", "resource": {"resourceId": ["resource"]}}, "document_import_outcome_unknown"),
    (None, "document_import_outcome_unknown"),
    ([], "document_import_outcome_unknown"),
    ({"uploadId": ""}, "document_import_outcome_unknown"),
    ({"storeId": None}, "document_import_outcome_unknown"),
    ({"content": None}, "document_import_outcome_unknown"),
    ({"state": "ready"}, "document_import_outcome_unknown"),  # ready, naming no resource
    ({"storeId": "store-b"}, "document_import_store_mismatch"),
    ({"content": {"sizeBytes": 7, "sha256": "sha256:" + "0" * 64}}, "document_import_conflict"),
], ids=["state", "list-state", "object-state", "number-upload-id", "list-upload-id",
        "number-resource-id", "list-resource-id", "null-answer", "list-answer",
        "upload-id", "store-id", "content", "resource", "other-store", "other-bytes"])
def test_an_unusable_import_answer_is_an_unknown_outcome_the_childs_next_start_recovers(docs, answer, reason):
    """An accepted declaration answered by an object too malformed to compare leaves the outcome unknown:
    the child is woken and its next start replays the same key. An answer naming another store or other
    bytes is evidence and still ends the child unrun."""
    from ouroboros import subagent_bootstrap, subagent_runtime

    accepted = docs.engine._import

    def unusable_once(body, key):
        docs.engine._import = accepted
        result = accepted(body, key)
        return {**result, **answer} if isinstance(answer, dict) else answer

    docs.engine._import = unusable_once
    task = _first_start_task(docs, "reuse")
    wake = subagent_bootstrap.bootstrap_before_context(docs.ctx, task, _HARNESS)
    if reason != "document_import_outcome_unknown":
        assert wake == "" and docs.ctx._configured_startup_refusal["reason"] == reason
        return
    assert wake, ("the host ended the child unrun", getattr(docs.ctx, "_configured_startup_refusal", None))
    receipt = json.loads(wake)
    assert (receipt["startup"]["reason"], receipt["startup"]["definitely_unrun"]) == (reason, True), receipt
    saved = _record(docs)
    assert "subagent_availability" not in task and saved["state"] == "send_possible"
    custody._CUSTODY.clear()
    started = json.loads(subagent_runtime.delegate_start_entry(docs.ctx, "").text)
    assert started["status"] == "started", started
    assert {call[1] for call in docs.engine.of("import")} == {saved["import_key"]}
    assert docs.engine.of("run")[0][2]["attachments"] == [{"resourceId": "res_1"}] and len(docs.engine.resources) == 1


def test_a_definite_document_refusal_still_ends_the_child_unrun(docs):
    """Only an import a later start settles wakes the child: a route that admits no such document
    ends it unrun and typed, as before."""
    from ouroboros import subagent_bootstrap

    docs.engine.inputs = [{"kind": "file", "mime_types": ["application/pdf"], "max_bytes": 10, "max_count": 1}]
    task = _first_start_task(docs, "reuse")
    assert subagent_bootstrap.bootstrap_before_context(docs.ctx, task, _HARNESS) == ""
    assert docs.ctx._configured_startup_refusal["reason"] == "document_route_unsupported"
    assert task["subagent_availability"]["status"] == "unavailable" and docs.engine.of("import") == []


# -- the record vocabulary --------------------------------------------------------------------

def test_receipts_survive_normalization_only_for_their_own_bytes(docs):
    from ouroboros.contracts.task_contract import normalize_attachment_manifest

    assert _start(docs.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    row = dict(docs.rows["attachments/brief.md"])
    receipts = carried_receipts(row, task_document_imports(docs.root, "t-docs"))
    plain = normalize_attachment_manifest([row])
    assert "engine_imports" not in plain[0]
    kept = normalize_attachment_manifest([{**row, "engine_imports": receipts}])[0]["engine_imports"]
    assert kept == receipts and kept[0]["request"]["expectedStoreId"] == STORE
    for field, value in (("content", {"sizeBytes": row["size"], "sha256": "sha256:" + "0" * 64}),
                         ("mime", "application/pdf"), ("name", "other.md"), ("kind", "image"), ("source", 7)):
        forged = copy.deepcopy(receipts)
        forged[0]["request"][field] = value  # other bytes, or a resource admitted under other semantics
        assert "engine_imports" not in normalize_attachment_manifest([{**row, "engine_imports": forged}])[0], field


def test_the_chain_fold_rebuilds_the_same_import_records(docs):
    docs.engine.faults = [("put", "drop")]
    _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    docs.engine.bindings[_record(docs)["import_key"]]["state"] = "cancelled"
    _start(docs.ctx, attachment_paths=["attachments/brief.md"])
    _start(docs.ctx, attachment_paths=["attachments/brief.md"], _document_intent="explicit")
    live = task_document_imports(docs.root, "t-docs")
    current.rebuild(docs.root, replace=True)
    assert task_document_imports(docs.root, "t-docs") == live


@pytest.mark.parametrize("name", ["brief.md", "BRIEF.markdown"])
def test_markdown_label_does_not_depend_on_the_interpreter(monkeypatch, name):
    import mimetypes

    from ouroboros.chat_uploads import detect_media

    monkeypatch.setattr(mimetypes, "guess_type", lambda *a, **k: (None, None))  # Python 3.10 has no .md
    assert detect_media(b"# Title\n", name) == ("text/markdown", "file")
    assert detect_media(b"plain\n", "notes.txt") == ("application/octet-stream", "file")


def test_a_new_docx_staging_is_labelled_without_a_system_mime_table(tmp_path, monkeypatch):
    """The interpreter's own table (no system mime.types; 3.10 to 3.12 have no .docx) still labels
    a new DOCX staging with the DOCX MIME a route can declare; other suffixes follow the table."""
    import mimetypes

    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.chat_uploads import detect_media

    builtin = mimetypes.MimeTypes(filenames=())
    monkeypatch.setattr(mimetypes, "guess_type", lambda url, strict=True: builtin.guess_type(url, strict))
    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    zipped = b"PK\x03\x04" + b"\0" * 26
    assert detect_media(zipped, "Memo.DOCX") == (docx, "file")
    assert detect_media(zipped, "sheet.xlsx") == (builtin.guess_type("x.xlsx")[0] or "application/octet-stream", "file")
    (tmp_path / "memo.docx").write_bytes(zipped)
    row, = stage_task_attachments(tmp_path, "t-docx", [{"path": str(tmp_path / "memo.docx"), "label": "memo.docx"}])
    assert (row["status"], row["mime"]) == ("staged", docx), row


# -- the transport --------------------------------------------------------------------------

def test_gateway_sends_the_saved_key_and_exact_body_and_types_engine_refusals():
    from ouroboros.gateways.claudexor_uploads import upload_import_supported

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("Idempotency-Key"),
                     request.headers.get("Content-Type"), request.content))
        if request.url.path == "/v2/uploads/imports" and json.loads(request.content).get("expectedStoreId") != STORE:
            return httpx.Response(409, json={"code": "store_mismatch", "message": "another store"})
        if request.url.path == "/v2/uploads/imports":
            return httpx.Response(200, json={"storeId": STORE, "state": "open", "uploadId": "upl_1"})
        return httpx.Response(200, json={"uploadId": "upl_1", "state": "uploaded"})

    gateway = _gateway(handler)
    body = {"expectedStoreId": STORE, "kind": "file", "mime": "text/markdown", "name": "brief.md",
            "source": "ouroboros-task-input:t/attachments/brief.md", "content": {"sizeBytes": 3, "sha256": "sha256:x"}}
    assert gateway.import_upload(body, idempotency_key="key-1")["state"] == "open"
    assert gateway.put_upload_bytes("upl_1", b"abc")["state"] == "uploaded"
    with pytest.raises(ClaudexorUnavailable) as refused:
        gateway.import_upload({**body, "expectedStoreId": "other"}, idempotency_key="key-1")
    assert (refused.value.code, refused.value.status_code) == ("store_mismatch", 409)
    (m1, p1, k1, _, raw), (m2, p2, _, ctype, data), _ = seen
    assert (m1, p1, k1, json.loads(raw)) == ("POST", "/v2/uploads/imports", "key-1", body)
    assert (m2, p2, ctype, data) == ("PUT", "/v2/uploads/upl_1/bytes", "application/octet-stream", b"abc")
    assert upload_import_supported([IMPORT_DESCRIPTOR])
    assert not upload_import_supported([{**IMPORT_DESCRIPTOR, "path": "/v2/uploads"}])
    assert not upload_import_supported([{k: v for k, v in IMPORT_DESCRIPTOR.items() if k != "requestSchema"}])
