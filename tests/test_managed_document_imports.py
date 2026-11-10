"""Managed document imports: the opt-in at import, the release each one owes and the pass that releases it.

The engine doubles are MOCKS of the engine's managed-lifetime interface, never the engine: ``lifetime: "managed"`` is
bound into the import key, an import replay of a released resource answers ``released``, the same START key and body
of an accepted run is answered as that operation while a new operation naming a released resource (a new key, or that
body once its command was pruned) refuses ``410``, and ``POST /v2/resources/:id/release`` is naturally idempotent
(its catalog row ``ControlResourceReleaseRequest`` is the only capability signal). Import scenarios run
the real ``_delegate_start`` (or the real import layer) over the C505 document fixture through the real gateway
methods; release scenarios run the real ``runtime_retirement`` pass with its real liveness probe, queue census and
continuation reader over task rows the host's own writers produce, the owed imports recorded through the real
custody reducer exactly as ``_import`` records them.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import owner_keep
from ouroboros import review_thread_retention as rt
from ouroboros.delegate_documents import (
    DOCUMENT_IMPORT_OBSERVED, DOCUMENT_IMPORT_REQUESTED, DOCUMENT_IMPORT_SENDING, MAX_PREVIOUS_RECORDS,
    attach_selected_documents, carried_receipts, task_document_imports,
)
from ouroboros.delegate_pending import store_facts
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401
from tests.test_delegate_documents import (  # noqa: F401
    IMPORT_DESCRIPTOR, STORE, DocumentEngine, _chain, _child, _record, _start, docs,
)
from tests.test_review_thread_retention import KEEP, SEAT, C, R, S, _own_thread, _publish_queue, _row

RELEASE_DESCRIPTOR = {"id": "post:resources.release", "method": "POST", "path": "/v2/resources/:id/release",
                      "requestSchema": "ControlResourceReleaseRequest"}
RELEASED_AT = "2026-11-04T12:00:00Z"


def _owed(root, task_id):
    return current.marks(root, task_id).get("managed_imports") or {}


# -- the import: opt-in, replay and carried receipts -------------------------------------------------

class ManagedEngine(DocumentEngine):
    """The C505 double plus managed lifetime as the interface states it; ``released`` holds the resources its owner
    released (the release route itself is the ``ReleaseEngine``'s, below)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.catalog.append(dict(RELEASE_DESCRIPTOR))
        self.released = {}

    def _import(self, body, key):
        binding, lifetime = self.bindings.get(key), body.get("lifetime")
        if binding is not None and binding.get("lifetime") != lifetime:  # bound into the key like name or source
            self.calls.append(("import", key, body))
            raise ClaudexorUnavailable("idempotency_conflict", "another lifetime under this key", status_code=409)
        index = len(self.calls)
        try:
            answer = super()._import({name: value for name, value in body.items() if name != "lifetime"}, key)
        finally:
            self.calls[index] = ("import", key, body)  # the exact body the host sent
            if key in self.bindings:
                self.bindings[key].setdefault("lifetime", lifetime)
                if self.bindings[key]["resource"] and lifetime:
                    self.bindings[key]["resource"]["lifetime"] = lifetime
        resource = answer.get("resource")
        if resource:
            resource.update({"lifetime": lifetime} if lifetime else {})
            if resource["resourceId"] in self.released:
                answer.update(state="released", releasedAt=self.released[resource["resourceId"]])
        return answer

    def _run(self, body, key):
        named = {ref.get("resourceId") for ref in body.get("attachments") or []}
        if key not in self.runs and named & set(self.released):
            self.calls.append(("run", key, body))
            raise ClaudexorUnavailable("resource_released", "a new reference to a released resource", status_code=410)
        return super()._run(body, key)


@pytest.fixture
def managed(docs, monkeypatch):  # noqa: F811
    """The C505 document task, served by an engine whose catalog lists the release row."""
    from ouroboros.gateways import claudexor as gateway_module

    engine = ManagedEngine()
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: engine)
    docs.engine = engine
    return docs


@pytest.mark.parametrize("listed", [True, False])
def test_only_an_engine_listing_the_release_row_gets_a_managed_declaration_and_its_minter_owes_it(managed, listed):
    if not listed:
        managed.engine.catalog = [dict(IMPORT_DESCRIPTOR)]
    assert _start(managed.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    (_, key, declaration), (_, _, replay) = managed.engine.of("import")
    assert replay == {k: v for k, v in declaration.items() if k != "content"}
    assert declaration.get("lifetime") == ("managed" if listed else None)
    assert _record(managed).get("owner_task_id") == ("t-docs" if listed else None)
    owed = _owed(managed.root, "t-docs")
    assert list(owed) == ([key] if listed else [])  # a legacy body stays byte-identical and owes nothing
    if listed:
        assert (owed[key]["state"], owed[key]["resource"]["resourceId"], owed[key]["request"]) == (
            "ready", "res_1", declaration)
    managed.engine.calls.clear()
    assert _start(managed.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    (_, replayed, lookup), = managed.engine.of("import")  # lookup-only resends the same lifetime under the same key
    assert replayed == key and lookup == {name: value for name, value in declaration.items() if name != "content"}


def test_receipt_limit_counts_stores_after_validation_and_newest_state_selection(managed):
    from ouroboros.contracts.task_contract import normalize_engine_import_receipts

    attach_selected_documents(managed.ctx, managed.engine, "some-route", ["attachments/brief.md"])
    row = managed.rows["attachments/brief.md"]
    receipt, = carried_receipts(row, task_document_imports(managed.root, "t-docs"))
    latest = {**receipt, "state": "released"}
    invalid = {**receipt, "request": {**receipt["request"], "mime": "wrong/type"}}
    others = [{**receipt, "store_id": f"store-{i}", "request": {
        **receipt["request"], "expectedStoreId": f"store-{i}"}} for i in range(8)]
    result = normalize_engine_import_receipts([invalid, latest, *([receipt] * 8), *others], row)
    assert result == [latest, *others[:7]]


def test_a_child_reuses_an_older_store_after_many_imports_and_a_lost_answer_in_another_store(managed):
    paths = ["attachments/brief.md"]
    attach_selected_documents(managed.ctx, managed.engine, "some-route", paths)
    first = _record(managed)
    other = ManagedEngine(store="22222222-3333-4444-8555-666666666666")
    for i in range(MAX_PREVIOUS_RECORDS - 1):
        attach_selected_documents(managed.ctx, other, "some-route", paths, f"new-other-{i}")
    other.faults = [("import", "lose")]
    with pytest.raises(ClaudexorUnavailable):
        attach_selected_documents(managed.ctx, other, "some-route", paths, "lost-other")
    last = _record(managed)
    assert last["state"] == "send_possible"
    child, authority = _child(managed)
    receipts = authority["attachment_manifest"][0]["engine_imports"]
    assert {r["store_id"] for r in receipts} == {STORE, other.store_identity}
    assert next(r for r in receipts if r["store_id"] == other.store_identity)["import_key"] == last["import_key"]
    assert next(r for r in receipts if r["store_id"] == other.store_identity)["state"] == "send_possible"
    managed.engine.calls.clear()
    answer = _start(child, attachment_paths=paths)
    assert answer["status"] == "started", answer
    assert managed.engine.of("import") == [(
        "import", first["import_key"], {k: v for k, v in first["request"].items() if k != "content"})]
    assert managed.engine.of("put") == []


def test_a_childs_carried_managed_receipt_replays_its_lifetime_and_opens_no_release_of_its_own(managed):
    assert _start(managed.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    key = _record(managed)["import_key"]
    child, authority = _child(managed)
    receipt, = next(row for row in authority["attachment_manifest"]
                    if row["relpath"] == "attachments/brief.md")["engine_imports"]
    assert (receipt["import_key"], receipt["owner_task_id"], receipt["request"]["lifetime"]) == (
        key, "t-docs", "managed")
    managed.engine.calls.clear()
    payload = _start(child, attachment_paths=["attachments/brief.md"])
    assert payload["status"] == "started", payload
    (_, replayed, lookup), = managed.engine.of("import")
    assert replayed == key and lookup["lifetime"] == "managed" and "content" not in lookup
    record = _record(managed, task_id="t-kid")
    assert (record["origin"], record["owner_task_id"]) == ("inherited", "t-docs")
    assert _owed(managed.root, "t-kid") == {} and list(_owed(managed.root, "t-docs")) == [key]


@pytest.mark.parametrize("lost", ["import", "put"])
def test_a_child_recovers_its_owners_uncertain_import_by_the_same_key_and_body(managed, lost):
    managed.engine.faults = [(lost, "lose")]
    refused = _start(managed.ctx, attachment_paths=["attachments/brief.md"])
    assert refused["status"] != "started", refused
    record = _record(managed)
    assert record["state"] == "send_possible" and not managed.engine.of("run")
    child, authority = _child(managed)
    receipt, = next(row for row in authority["attachment_manifest"]
                    if row["relpath"] == "attachments/brief.md")["engine_imports"]
    assert receipt == {"store_id": STORE, "import_key": record["import_key"], "request": record["request"],
                       "owner_task_id": "t-docs", "state": "send_possible"}
    managed.engine.calls.clear()
    answer = _start(child, attachment_paths=["attachments/brief.md"])
    assert answer["status"] == "started", answer
    lookup = {name: value for name, value in record["request"].items() if name != "content"}
    assert managed.engine.of("import")[0] == ("import", record["import_key"], lookup)
    assert len(managed.engine.bindings) == len(managed.engine.resources) == 1
    assert len(managed.engine.of("put")) == (1 if lost == "import" else 0)
    assert _owed(managed.root, "t-kid") == {} and list(_owed(managed.root, "t-docs")) == [record["import_key"]]


@pytest.mark.parametrize("state", ["gap", "rejected", "released"])
def test_a_child_keeps_the_latest_terminal_or_gapped_import_instead_of_minting_another(managed, state):
    from ouroboros.delegate_documents import FINISHED_STATES

    attach_selected_documents(managed.ctx, managed.engine, "some-route", ["attachments/brief.md"])
    original = _record(managed)
    assert custody.emit(managed.root, DOCUMENT_IMPORT_OBSERVED, {
        "task_id": "t-docs", "material": next(iter(task_document_imports(managed.root, "t-docs"))),
        "import_key": original["import_key"], "state": state})
    child, authority = _child(managed)
    managed.engine.calls.clear()
    answer = _start(child, attachment_paths=["attachments/brief.md"])
    assert managed.engine.of("put") == [], answer
    assert set(managed.engine.bindings) == {original["import_key"]}
    if state in FINISHED_STATES:
        assert answer["reason"] == "document_import_finished", answer
        assert managed.engine.of("import") == []
    else:  # the same binding may become readable again; only lookup under that key can establish it
        assert answer["status"] == "started", answer
        assert managed.engine.of("import") == [(
            "import", original["import_key"], {k: v for k, v in original["request"].items() if k != "content"})]
    receipt, = authority["attachment_manifest"][0]["engine_imports"]
    assert receipt["state"] == state and receipt["request"] == original["request"]


def test_every_explicit_new_import_stays_owed_past_the_record_cap_through_the_fold_and_a_copy(managed, tmp_path):
    """Earlier records keep only the newest intents; the release obligations keep every managed key, the chain fold
    rebuilds them, and a canonical copy of the row keeps the receipt's lifetime and owner (any other lifetime drops)."""
    from ouroboros.artifacts import attachment_manifest_projection, materialize_inherited_attachment_manifest
    from ouroboros.contracts.task_contract import normalize_attachment_manifest

    for index in range(MAX_PREVIOUS_RECORDS + 2):  # the real import layer, one explicit new intent each
        attach_selected_documents(managed.ctx, managed.engine, "some-route", ["attachments/brief.md"],
                                  f"explicit-new-{index}")
    keys = [row["import_key"] for row in _chain(managed.root, DOCUMENT_IMPORT_REQUESTED)]
    assert len(set(keys)) == MAX_PREVIOUS_RECORDS + 2 and len(_record(managed)["previous"]) == MAX_PREVIOUS_RECORDS
    assert sorted(_owed(managed.root, "t-docs")) == sorted(keys)
    live = current.marks(managed.root, "t-docs")
    current.rebuild(managed.root, replace=True)
    assert current.marks(managed.root, "t-docs") == live
    row = dict(managed.rows["attachments/brief.md"])
    receipts = carried_receipts(row, task_document_imports(managed.root, "t-docs"))
    copied, error = materialize_inherited_attachment_manifest([{**row, "engine_imports": receipts}],
                                                              tmp_path / "canonical", "t-docs")
    assert not error, error
    kept = attachment_manifest_projection(tmp_path / "canonical", "t-docs", copied)["attachment_manifest"][0]
    assert kept["engine_imports"] == receipts and receipts[0]["owner_task_id"] == "t-docs"
    forged = copy.deepcopy(receipts)
    forged[0]["request"]["lifetime"] = "indefinite"
    assert "engine_imports" not in normalize_attachment_manifest([{**row, "engine_imports": forged}])[0]


def test_a_released_key_answers_finished_and_only_an_explicit_new_import_begins_another(managed):
    """A holder outside the owner's own task meets the owner's release typed: nothing re-uploads and nothing
    starts; its explicit new import is its own, and so is that import's release."""
    assert _start(managed.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    key = _record(managed)["import_key"]
    child, _authority = _child(managed)
    managed.engine.released["res_1"] = RELEASED_AT  # the owner's release, as the engine keeps it
    managed.engine.calls.clear()
    refused = _start(child, attachment_paths=["attachments/brief.md"])
    assert refused["reason"] == "document_import_finished" and refused["definitely_unrun"] is True, refused
    assert "released" in refused["detail"] and "t-docs" in refused["detail"]
    assert managed.engine.of("put") == [] and managed.engine.of("run") == []
    assert (_record(managed, task_id="t-kid")["state"], _record(managed, task_id="t-kid")["released_at"]) == (
        "released", RELEASED_AT)
    calls = len(managed.engine.calls)
    assert _start(child, attachment_paths=["attachments/brief.md"])["reason"] == "document_import_finished"
    assert [call[0] for call in managed.engine.calls[calls:]] == ["operations"]  # no import under a finished key
    fresh = _start(child, attachment_paths=["attachments/brief.md"], _document_intent="explicit-new")
    assert fresh["status"] == "started", fresh
    own = _record(managed, task_id="t-kid")["import_key"]
    assert own != key and list(_owed(managed.root, "t-kid")) == [own]


def _lost_start(managed, monkeypatch, *, before_store_identity):
    """A real start the engine accepted while its reply was lost. A row from before store identity lacks its facts,
    which the real writer omits here for this one start; only such a row may still be re-sent to a named store."""
    managed.engine.faults = [("run", "lose")]
    write = custody.record_start_requested
    with monkeypatch.context() as patch:
        if before_store_identity:
            patch.setattr(custody, "record_start_requested", lambda drive, **row: write(
                drive, **{name: value for name, value in row.items() if name not in store_facts({})}))
        return _start(managed.ctx, attachment_paths=["attachments/brief.md"])["pending_invocation_id"]


def test_an_accepted_start_replayed_by_retry_of_is_served_after_the_release_without_any_import(managed, monkeypatch):
    pending = _lost_start(managed, monkeypatch, before_store_identity=True)
    managed.engine.released["res_1"] = RELEASED_AT
    imports = len(managed.engine.of("import"))
    replayed = _start(managed.ctx, retry_of=pending)  # the same START key and body: the original operation
    assert replayed["status"] == "started" and replayed["idempotent_recovery"] is True, replayed
    first, second = managed.engine.of("run")
    assert first[1:] == second[1:], "the recorded key and bytes"
    assert len(managed.engine.of("import")) == imports


def test_a_retry_whose_command_the_engine_pruned_meets_the_release_typed_and_imports_nothing(managed, monkeypatch):
    pending = _lost_start(managed, monkeypatch, before_store_identity=True)
    managed.engine.released["res_1"] = RELEASED_AT
    managed.engine.runs.clear()  # command retention pruned it: the same body is a new reference now
    imports = len(managed.engine.of("import"))
    puts = copy.deepcopy(managed.engine.of("put"))
    refused = _start(managed.ctx, retry_of=pending)
    assert refused["reason"] == "resource_released" and refused["status"] == "refused", refused
    assert len(managed.engine.of("import")) == imports and managed.engine.of("put") == puts
    assert not managed.engine.runs, "the released reference started nothing"


@pytest.mark.parametrize("pruned", [False, True])
def test_a_lost_start_whose_store_is_unproven_is_not_resent_after_the_release(managed, monkeypatch, pruned):
    """Today's row: no receipt proves which store admitted it, so the engine is asked neither for the run nor for
    the release's refusal, and the outcome stays unknown."""
    pending = _lost_start(managed, monkeypatch, before_store_identity=False)
    managed.engine.released["res_1"] = RELEASED_AT
    if pruned:
        managed.engine.runs.clear()
    sent = copy.deepcopy([managed.engine.of(kind) for kind in ("run", "import", "put")])
    refused = _start(managed.ctx, retry_of=pending)
    assert (refused["status"], refused["reason"]) == ("refused", "invocation_store_unproven"), refused
    assert not refused.get("definitely_unrun"), refused
    assert [managed.engine.of(kind) for kind in ("run", "import", "put")] == sent
    assert custody.invocation_record(managed.ctx.drive_root, pending)["state"] == "pending"


def test_the_release_transport_sends_the_exact_body_and_the_catalog_row_is_matched_exactly():
    from ouroboros.gateways.claudexor_uploads import resource_release_supported
    from tests._delegated_transport_shared import _gateway

    seen = []
    sha = "sha256:" + "b" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.method, request.url.path, request.headers.get("Idempotency-Key"), body))
        if body["expectedStoreId"] != STORE:
            return httpx.Response(409, json={"code": "store_mismatch", "message": "another store"})
        return httpx.Response(200, json={"storeId": STORE, "resourceId": "res_1", "sha256": sha,
                                         "state": "released", "releasedAt": RELEASED_AT})

    gateway = _gateway(handler)
    body = {"expectedStoreId": STORE, "sha256": sha}
    assert gateway.release_resource("res_1", body)["releasedAt"] == RELEASED_AT
    with pytest.raises(ClaudexorUnavailable) as refused:
        gateway.release_resource("res_1", {**body, "expectedStoreId": "another-store"})
    assert (refused.value.code, refused.value.status_code) == ("store_mismatch", 409)
    assert seen[0] == ("POST", "/v2/resources/res_1/release", None, body)
    assert resource_release_supported([IMPORT_DESCRIPTOR, RELEASE_DESCRIPTOR])
    assert not resource_release_supported([{**RELEASE_DESCRIPTOR, "requestSchema": "ControlResourceRelease"}])
    assert not resource_release_supported([IMPORT_DESCRIPTOR])


def test_the_existing_upload_cancel_route_receives_only_the_exact_upload_id():
    from tests._delegated_transport_shared import _gateway

    seen = []

    def handler(request):
        seen.append((request.method, request.url.raw_path, request.content))
        return httpx.Response(200, json={"uploadId": "upl_1", "state": "cancelled"})

    assert _gateway(handler).cancel_upload("upl_1")["state"] == "cancelled"
    assert seen == [("DELETE", b"/v2/uploads/upl_1", b"")]


# -- the release: the runtime_retirement pass over finished tasks ----------------------------------------------

SHA = "sha256:" + "a" * 64
REQUEST = {"expectedStoreId": STORE, "kind": "file", "mime": "text/markdown", "name": "brief.md",
           "source": f"ouroboros-task-input:{R}/attachments/brief.md", "lifetime": "managed",
           "content": {"sizeBytes": 3, "sha256": SHA}}
BODY = {"expectedStoreId": STORE, "sha256": SHA}


class ReleaseEngine:
    """What the release step asks of the engine: its catalog and store, a lookup-only replay of a key (``lookups``:
    an answer or a typed refusal; none is the interface's 404) and the release route, whose next answers ``faults``
    may replace; a repeated release answers the identical body."""

    def __init__(self):
        self.store_identity = STORE
        self.catalog = [dict(IMPORT_DESCRIPTOR), dict(RELEASE_DESCRIPTOR)]
        self.lookups, self.faults, self.released, self.calls = {}, [], {}, []
        self.threads = False  # the thread step's route (only a test with a reviewer thread lists it)

    def operations(self, **_kw):
        return copy.deepcopy(self.catalog)

    def thread_retirement_supported(self):
        return self.threads

    def close(self):
        pass

    def import_upload(self, request, *, idempotency_key):
        self.calls.append(("import", idempotency_key, copy.deepcopy(request)))
        answer = self.lookups.get(idempotency_key) or ClaudexorUnavailable(
            "upload_import_not_found", "no retained binding", status_code=404)
        if isinstance(answer, Exception):
            raise answer
        return copy.deepcopy(answer)

    def release_resource(self, resource_id, request):
        self.calls.append(("release", resource_id, copy.deepcopy(request)))
        if self.faults:
            raise self.faults.pop(0)
        return {"storeId": self.store_identity, "resourceId": resource_id, "sha256": request["sha256"],
                "state": "released", "releasedAt": self.released.setdefault(resource_id, RELEASED_AT)}

    def cancel_upload(self, upload_id):
        self.calls.append(("cancel", upload_id))
        for answer in self.lookups.values():
            if isinstance(answer, dict) and answer.get("uploadId") == upload_id:
                answer["state"] = "cancelled"
        if self.faults:
            raise self.faults.pop(0)  # the cancellation happened; its answer did not arrive
        return {"uploadId": upload_id, "state": "cancelled"}


def _resource(key):
    return {"resourceId": f"res-{key}", "sha256": SHA, "sizeBytes": 3, "mime": "text/markdown", "name": "brief.md"}


def _owe(root, key, *, task=R, material="brief", sent=True, ready=True):
    """One managed import TASK minted, recorded as ``delegate_documents._import`` records it."""
    fact = {"task_id": task, "material": material, "import_key": key}
    assert custody.emit(root, DOCUMENT_IMPORT_REQUESTED, {**fact, "store_id": STORE, "request": REQUEST,
                                                          "owner_task_id": task, "origin": "imported",
                                                          "input": {"relpath": "attachments/brief.md"}})
    if sent:
        assert custody.emit(root, DOCUMENT_IMPORT_SENDING, fact)
    if ready:
        assert custody.emit(root, DOCUMENT_IMPORT_OBSERVED, {**fact, "state": "ready", "resource": _resource(key)})


@pytest.fixture
def work(tmp_path):
    """A running root on a real store and queue; the pass activated before any of its terminal writes."""
    root = tmp_path / "data"
    root.mkdir()
    _publish_queue(root)
    _row(root, R, "running")
    engine = ReleaseEngine()
    assert rt.retire_finished_review_threads(root, engine=engine) == {"activated": True}

    def run(pending=(), running=()):
        _publish_queue(root, pending, running)
        return rt.retire_finished_review_threads(root, engine=engine)  # the real live probe and queue census

    return SimpleNamespace(root=root, engine=engine, run=run, owed=lambda task=R: _owed(root, task))


@pytest.mark.parametrize("past_history", [False, True])
@pytest.mark.parametrize("outcome", ["bound", "released", "gap", "damaged", "conflict", "rejected", "refused",
                                    "cancelled", "unsent"])
def test_release_facts_for_an_older_key_do_not_erase_the_current_import(work, past_history, outcome):
    from ouroboros.delegate_documents import DOCUMENT_RELEASE_BOUND, DOCUMENT_RELEASE_SETTLED

    keys = [f"k-{index}" for index in range(MAX_PREVIOUS_RECORDS + 2 if past_history else 2)]
    for key in keys:
        _owe(work.root, key)
    before = task_document_imports(work.root, R)["brief"]
    event = {"task_id": R, "material": "brief", "import_key": keys[0], "outcome": outcome}
    kind = DOCUMENT_RELEASE_BOUND if outcome == "bound" else DOCUMENT_RELEASE_SETTLED
    assert custody.emit(work.root, kind, event)
    expected = copy.deepcopy(before)
    if outcome == "released" and not past_history:
        expected["previous"][0]["state"] = "released"
    assert task_document_imports(work.root, R)["brief"] == expected
    current.rebuild(work.root, replace=True)
    assert task_document_imports(work.root, R)["brief"] == expected


def test_a_finished_work_releases_each_owed_import_once_by_its_recorded_store_and_digest(work):
    _owe(work.root, "k-1")
    _row(work.root, R, "completed")  # finished by its author: Continue is not offered
    report = work.run()
    assert work.engine.calls == [("release", "res-k-1", BODY)] and work.owed() == {}, report
    assert report["waiting"] == 0 and not report["holds"], report
    fact, = report["observed"]
    assert (fact["outcome"], fact["import_key"], fact["released_at"]) == ("released", "k-1", RELEASED_AT)
    record = task_document_imports(work.root, R)["brief"]
    assert (record["state"], record["released_at"]) == ("released", RELEASED_AT)
    assert work.run()["tasks"] == 0 and len(work.engine.calls) == 1  # settled: never asked again


def test_a_queued_child_that_may_still_select_the_carried_receipt_holds_its_finished_owner(work):
    """The owner ended first; its queued child holds the owner's receipt on an input it has not selected yet."""
    _owe(work.root, "k-1")
    _row(work.root, C, "scheduled", parent=R)
    _row(work.root, R, "completed")
    report = work.run(pending=[{"id": C, "root_task_id": R, "parent_task_id": R}])
    assert report["holds"] == {"document_tree_live": 1} and report["waiting"] == 1, report
    assert work.engine.calls == [] and list(work.owed()) == ["k-1"]
    _row(work.root, C, "running", parent=R)
    _row(work.root, C, "completed", parent=R)
    assert work.run()["observed"][0]["outcome"] == "released"
    assert work.engine.calls == [("release", "res-k-1", BODY)] and work.owed() == {}


@pytest.mark.parametrize("end", ["author_stop", "technical_break"])
def test_any_continue_still_offered_holds_the_release_where_native_retention_lets_go(work, end):
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_store import patch_document, read_document

    _owe(work.root, "k-1")
    if end == "author_stop":
        _row(work.root, R, "completed")
        stop = {"execution": {"status": "ok", "task_completion": {"action": "stop", "rationale": "Not ready."}}}
        patch_document(work.root, R, lambda row: {**row, "reason_code": "final_message", "outcome_axes": stop})
    else:
        _row(work.root, R, "cancelled", cancel_origin={"source": "", "reason": "server_shutdown"})
    assert continuation_offer(read_document(work.root, R), R)["eligible"] is True
    report = work.run()
    assert report["holds"] == {"document_continuable": 1} and report["waiting"] == 1, report
    assert work.engine.calls == [] and list(work.owed()) == ["k-1"]


def test_an_unrecorded_interruption_holds_until_a_later_technical_fact_offers_continue_and_that_work_ends(work):
    """A root that failed with no recorded cause has no decided end: the orphan reconciler may still record a
    technical cause and so offer Continue. The registration reading lets such a root go (``test_continue_author_stop``);
    the document's release waits through that later offer and the continued work, and only then goes."""
    from ouroboros.owner_continue import claim_on_predecessor, continuation_offer
    from ouroboros.task_store import patch_document, read_document

    _owe(work.root, "k-1")
    _row(work.root, R, "failed")  # no cause recorded
    assert continuation_offer(read_document(work.root, R), R)["refusal"] == "interruption_cause_unrecorded"
    report = work.run()
    assert report["holds"] == {"document_continuable": 1} and work.engine.calls == [], report
    patch_document(work.root, R, lambda row: {**row, "reason_code": "orphaned_running_after_worker_restart",
                                              "status_reconciled_from": "running"})  # the later technical fact
    assert continuation_offer(read_document(work.root, R), R)["eligible"] is True
    assert work.run()["holds"] == {"document_continuable": 1} and work.engine.calls == []
    claim_on_predecessor(work.root, R, "nonce-1", {"successor_task_id": S})
    _row(work.root, S, "running", tree=S)
    _row(work.root, S, "completed", tree=S)
    report = work.run()
    assert work.engine.calls == [("release", "res-k-1", BODY)] and work.owed() == {}, report


def test_an_import_owed_before_the_first_pass_reaches_its_release_while_older_threads_keep_their_cutover(tmp_path):
    """A fresh installation: the import opens its obligation and its author-finished owner, which also owns a
    reviewer thread, ends before the retirement step ever ran. The import's own custody transaction left a floor,
    so the first pass reads the owner's end without needing another unscheduled pass: the document is
    released. The first pass's own time still bounds threads: the thread, started before it, is never a candidate."""
    from ouroboros import maintenance_jobs, review_ledger
    from ouroboros.maintenance_retirement import JOB

    root = tmp_path / "data"
    root.mkdir()
    _publish_queue(root)
    _row(root, R, "running")
    record_id = review_ledger.new_record_id()
    review_ledger.write_record(root, {"record_id": record_id, "task_id": R, "state": "settled", "surface": "intention",
                                      "verdict": {"aggregate": "PASS"}, "rows": [{"seat_id": SEAT, "source_refs": []}]})
    _own_thread(root, R, R, record_id)
    _owe(root, "k-1")
    floor = maintenance_jobs.job(root, JOB)[rt.JOB_FIELD]
    assert set(floor) == {"floor"}, floor  # the step never ran: only the import's position is kept
    _row(root, R, "completed")  # author-finished before the step's first pass
    engine = ReleaseEngine()
    engine.threads = True
    report = rt.retire_finished_review_threads(root, engine=engine)
    state = maintenance_jobs.job(root, JOB)[rt.JOB_FIELD]
    assert state["counter"] > floor["floor"] and state["waiting"] == [] and "floor" not in state, state
    assert engine.calls == [("release", "res-k-1", BODY)] and _owed(root, R) == {}, report
    assert report["holds"] == {"predates_activation": 1} and report["waiting"] == 0, report


def test_an_import_and_its_owners_end_between_the_first_pass_read_and_its_pin_lose_nothing(tmp_path, monkeypatch):
    """The activation race: the import's custody transaction and its owner's terminal write both commit after the
    first pass read the job row and before it wrote its pin. That pass drops the floor, but the floor and the owner's
    end both stand above the counter it read, so the next pass reads the end."""
    from ouroboros import maintenance_jobs, maintenance_retirement as retirement

    root = tmp_path / "data"
    root.mkdir()
    _publish_queue(root)
    _row(root, R, "running")
    real = maintenance_jobs.record

    def activation_after_the_import_and_its_end(*args, **kwargs):
        monkeypatch.setattr(maintenance_jobs, "record", real)
        _owe(root, "k-1")
        _row(root, R, "completed")
        return real(*args, **kwargs)

    monkeypatch.setattr(maintenance_jobs, "record", activation_after_the_import_and_its_end)
    engine = ReleaseEngine()
    assert rt.retire_finished_review_threads(root, engine=engine) == {"activated": True}
    # There are no task files or START. The real terminal writer must schedule the missing pass;
    # invoking the reader a second time without a dispatch reason would hide a stranded obligation.
    job = maintenance_jobs.job(root, retirement.JOB)
    assert job["next_due"] is not None
    assert retirement.due(root, {**job, "state": "done", "inputs": retirement._inputs(root)}, {}, child=False)
    report = rt.retire_finished_review_threads(root, engine=engine)
    assert engine.calls == [("release", "res-k-1", BODY)] and _owed(root, R) == {}, report


def test_a_bound_successor_holds_and_its_ended_tree_lets_the_origin_owner_release(work):
    from ouroboros.owner_continue import claim_on_predecessor

    _owe(work.root, "k-1")
    _row(work.root, R, "cancelled", cancel_origin={"source": "", "reason": "server_shutdown"})
    claim_on_predecessor(work.root, R, "nonce-1", {"successor_task_id": S})  # bound, its admission unconfirmed
    _row(work.root, S, "scheduled", tree=S)
    report = work.run(pending=[{"id": S, "root_task_id": S}])
    assert report["holds"] == {"document_continuable": 1} and work.engine.calls == [], report
    _row(work.root, S, "running", tree=S)
    _row(work.root, S, "completed", tree=S)  # the successor's own end offers no Continue
    report = work.run()
    assert work.engine.calls == [("release", "res-k-1", BODY)], report
    assert work.owed() == {} and work.owed(S) == {}  # the origin owner's obligation, never moved to the successor


def test_an_unknown_import_is_asked_lookup_only_with_its_exact_body_and_an_unsent_one_settles_unasked(work):
    _owe(work.root, "k-lost", ready=False)  # sent, its answer lost: no resource observed, no run ever started
    _owe(work.root, "k-unsent", material="notes", sent=False, ready=False)  # minted, never sent
    work.engine.lookups["k-lost"] = {"storeId": STORE, "state": "ready", "uploadId": "upl-1",
                                     "content": dict(REQUEST["content"]), "receivedBytes": 3,
                                     "resource": {**_resource("k-lost"), "kind": "file", "lifetime": "managed"}}
    _row(work.root, R, "completed")
    report = work.run()
    lookup = {name: value for name, value in REQUEST.items() if name != "content"}
    assert work.engine.calls == [("import", "k-lost", lookup), ("release", "res-k-lost", BODY)], report
    assert {fact["import_key"]: fact["outcome"] for fact in report["observed"]} == {
        "k-lost": "released", "k-unsent": "unsent"}
    assert work.owed() == {} and report["waiting"] == 0


def test_an_unknown_release_is_sent_again_unchanged_and_another_store_or_engine_waits_without_a_call(work):
    _owe(work.root, "k-1")
    _row(work.root, R, "completed")
    work.engine.faults = [ClaudexorUnavailable("daemon_unreachable", "the answer was lost")]
    report = work.run()
    assert report["holds"] == {"document_daemon_unreachable": 1} and report["waiting"] == 1, report
    work.engine.store_identity = "another-store"
    assert work.run()["holds"] == {"document_store_mismatch": 1}
    work.engine.store_identity, work.engine.catalog = STORE, [dict(IMPORT_DESCRIPTOR)]
    assert work.run()["holds"] == {"document_engine_release_unsupported": 1}
    work.engine.catalog.append(dict(RELEASE_DESCRIPTOR))
    report = work.run()
    assert work.engine.calls == [("release", "res-k-1", BODY)] * 2 and work.owed() == {}, report
    assert report["waiting"] == 0


def test_a_typed_release_refusal_settles_without_effect_and_is_not_asked_again(work):
    _owe(work.root, "k-1")
    _row(work.root, R, "completed")
    work.engine.faults = [ClaudexorUnavailable("resource_lifetime_indefinite", "indefinite", status_code=409)]
    report = work.run()
    fact, = report["observed"]
    assert (fact["outcome"], fact["code"]) == ("refused", "resource_lifetime_indefinite") and work.owed() == {}
    assert report["waiting"] == 0 and work.run()["tasks"] == 0 and len(work.engine.calls) == 1


def test_the_public_keep_setter_keeps_a_managed_import_even_without_local_material(work):
    _owe(work.root, "k-1")
    _row(work.root, R, "completed")
    receipt = owner_keep.set_owner_keep(work.root, R, keep=True, source=KEEP)
    assert receipt["keep"] and "managed_import:k-1" in receipt["retained"], receipt
    assert receipt["releases"] == {"managed_import:k-1": "held"}
    report = work.run()
    assert report["holds"] == {"document_owner_keep": 1} and report["waiting"] == 1, report
    assert work.engine.calls == []
    assert owner_keep.set_owner_keep(work.root, R, keep=False, source=KEEP)["status"] == "released"
    assert work.run()["observed"][0]["outcome"] == "released" and work.owed() == {}


@pytest.mark.parametrize("boundary", ["catalog", "lookup", "between_releases"])
def test_keep_arriving_after_the_initial_check_prevents_every_not_yet_bound_release(work, monkeypatch, boundary):
    _owe(work.root, "k-1", ready=boundary != "lookup")
    if boundary == "between_releases":
        _owe(work.root, "k-2", material="second")
    if boundary == "lookup":
        work.engine.lookups["k-1"] = {"storeId": STORE, "state": "ready", "uploadId": "upl-1",
                                      "content": dict(REQUEST["content"]), "receivedBytes": 3,
                                      "resource": {**_resource("k-1"), "kind": "file", "lifetime": "managed"}}
    _row(work.root, R, "completed")
    method = {"catalog": "operations", "lookup": "import_upload", "between_releases": "release_resource"}[boundary]
    real = getattr(work.engine, method)
    keeps = []

    def keep_after_wait(*args, **kwargs):
        result = real(*args, **kwargs)
        keeps.append(owner_keep.set_owner_keep(work.root, R, keep=True, source=KEEP))
        return result

    monkeypatch.setattr(work.engine, method, keep_after_wait)
    report = work.run()
    assert keeps and all(row["keep"] for row in keeps)
    assert report["holds"] == {"document_owner_keep": 1}, report
    assert [call for call in work.engine.calls if call[0] == "release"] == (
        [("release", "res-k-1", BODY)] if boundary == "between_releases" else [])
    assert list(work.owed()) == (["k-2"] if boundary == "between_releases" else ["k-1"])


def test_keep_after_a_bound_unknown_release_reports_uncertainty_and_does_not_replay_while_kept(work, monkeypatch):
    _owe(work.root, "k-1")
    _row(work.root, R, "completed")
    real = work.engine.release_resource
    keeps = []

    def keep_after_binding(*args):
        keeps.append(owner_keep.set_owner_keep(work.root, R, keep=True, source=KEEP))
        return real(*args)

    work.engine.faults = [ClaudexorUnavailable("daemon_unreachable", "reply lost")]
    monkeypatch.setattr(work.engine, "release_resource", keep_after_binding)
    assert work.run()["waiting"] == 1
    assert keeps[0]["status"] == "release_unconfirmed" and keeps[0]["keep"], keeps
    assert keeps[0]["releases"] == {"managed_import:k-1": "release_unconfirmed"}
    assert work.owed()["k-1"]["release_state"] == "bound"
    before = current.marks(work.root, R)
    current.rebuild(work.root, replace=True)
    assert current.marks(work.root, R) == before
    assert work.run()["holds"] == {"document_owner_keep": 1} and len(work.engine.calls) == 1
    owner_keep.set_owner_keep(work.root, R, keep=False, source=KEEP)
    monkeypatch.setattr(work.engine, "release_resource", real)
    assert work.run()["waiting"] == 0 and work.engine.calls == [("release", "res-k-1", BODY)] * 2


def test_a_new_child_during_engine_discovery_prevents_a_release(work, monkeypatch):
    _owe(work.root, "k-1")
    _row(work.root, R, "completed")
    real = work.engine.operations

    def child_after_wait(**kwargs):
        _row(work.root, C, "scheduled", parent=R)
        return real(**kwargs)

    monkeypatch.setattr(work.engine, "operations", child_after_wait)
    assert work.run()["waiting"] == 1 and work.engine.calls == []
    assert "release_state" not in work.owed()["k-1"]


@pytest.mark.parametrize("lost", [False, True])
def test_an_open_unfinished_upload_is_cancelled_and_a_lost_cancel_is_reconciled_without_reupload(work, lost):
    _owe(work.root, "k-1", ready=False)
    work.engine.lookups["k-1"] = {"storeId": STORE, "state": "open", "uploadId": "upl-1",
                                  "content": dict(REQUEST["content"]), "receivedBytes": 0}
    _row(work.root, R, "completed")
    if lost:
        work.engine.faults = [ClaudexorUnavailable("daemon_unreachable", "cancel reply lost")]
    report = work.run()
    assert (report["waiting"] == 1) is lost
    if lost:
        assert work.owed()["k-1"]["release_state"] == "bound"
        report = work.run()
    assert work.owed() == {} and report["observed"][0]["outcome"] == "cancelled"
    lookup = ("import", "k-1", {name: value for name, value in REQUEST.items() if name != "content"})
    assert work.engine.calls == [lookup, ("cancel", "upl-1"), *([lookup] if lost else [])]


@pytest.mark.parametrize("local", [False, True])
def test_public_keep_after_settlement_reports_closed_reuse_even_after_import_history_rotates(work, local):
    """Keep cannot restore a released resource ID, even when other material survives.
    Keep's current receipt must not depend on the bounded import replay history."""
    _owe(work.root, "k-released")
    _row(work.root, R, "completed")
    assert work.run()["observed"][0]["outcome"] == "released"
    assert work.owed() == {}
    _row(work.root, R, "running")
    for index in range(MAX_PREVIOUS_RECORDS + 2):
        _owe(work.root, f"k-unsent-{index}", sent=False, ready=False)
    _row(work.root, R, "completed")
    settlements = work.run()["observed"]
    assert len(settlements) == MAX_PREVIOUS_RECORDS + 2
    assert all(fact["outcome"] == "unsent" for fact in settlements)
    assert work.owed() == {}
    assert "k-released" not in json.dumps(task_document_imports(work.root, R))
    before = current.marks(work.root, R)
    current.rebuild(work.root, replace=True)
    assert current.marks(work.root, R) == before
    if local:
        material = work.root / "task_drives" / R
        material.mkdir(parents=True)
        (material / "answer.txt").write_text("retained answer", encoding="utf-8")
    receipt = owner_keep.set_owner_keep(work.root, R, keep=True, source=KEEP)
    assert receipt["status"] == ("partly_retired" if local else "already_retired"), receipt
    assert receipt["keep"] is local, receipt
    assert receipt["releases"] == {"managed_import:k-released": "future_reuse_released"}, receipt
    assert receipt["retired"] == ["managed_import:k-released"], receipt
    assert receipt["unknown"] == [], receipt
    if local:
        assert str(material) in receipt["retained"]
        assert (material / "answer.txt").read_text(encoding="utf-8") == "retained answer"


@pytest.mark.parametrize("code,status", [("store_mismatch", 409), ("upload_import_unavailable", 503), ("bad_request", 400)])
def test_refusal_after_accepted_upload_preserves_its_release_debt(managed, monkeypatch, code, status):
    from ouroboros import delegate_documents as dd

    real = managed.engine.import_upload
    calls = 0

    def refuse_final_post(body, *, idempotency_key):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ClaudexorUnavailable(code, "refused only the final POST", status_code=status)
        return real(body, idempotency_key=idempotency_key)

    monkeypatch.setattr(managed.engine, "import_upload", refuse_final_post)
    row = managed.rows["attachments/brief.md"]
    with pytest.raises(ClaudexorUnavailable):
        dd._import(managed.engine, managed.root, "t-docs", STORE, row, "", managed=True)
    record = _record(managed)
    assert record["state"] == "send_possible"
    assert _owed(managed.root, "t-docs")[record["import_key"]]["state"] == "send_possible"
    puts = list(managed.engine.of("put"))
    replayed = dd._import(managed.engine, managed.root, "t-docs", STORE, row, "", managed=True)
    assert replayed["import_key"] == record["import_key"] and managed.engine.of("put") == puts


@pytest.mark.parametrize("inherited", [False, True])
def test_capability_loss_refuses_managed_replay_without_changing_identity(managed, inherited):
    assert _start(managed.ctx, attachment_paths=["attachments/brief.md"])["status"] == "started"
    original = copy.deepcopy(_record(managed))
    ctx = _child(managed)[0] if inherited else managed.ctx
    managed.engine.catalog = [dict(IMPORT_DESCRIPTOR)]
    managed.engine.calls.clear()
    with pytest.raises(ClaudexorUnavailable) as refusal:
        attach_selected_documents(ctx, managed.engine, "some-route", ["attachments/brief.md"])
    assert refusal.value.code == "document_import_unsupported"
    assert not managed.engine.of("import") and not managed.engine.of("put")
    saved = task_document_imports(managed.root, ctx.task_id)
    record = next(iter(saved.values()))
    assert (record["import_key"], record["request"], record["store_id"]) == (
        original["import_key"], original["request"], original["store_id"])
    managed.engine.catalog.append(dict(RELEASE_DESCRIPTOR))
    result = attach_selected_documents(ctx, managed.engine, "some-route", ["attachments/brief.md"])
    assert result.facts[0]["resource_id"] == original["resource"]["resourceId"]
    assert not managed.engine.of("put")


@pytest.mark.parametrize("outcome,code", [("gap", "upload_import_not_found"),
                                         ("damaged", "upload_import_binding_invalid"),
                                         ("refused", "resource_lifetime_indefinite")])
def test_public_keep_preserves_settled_import_uncertainty(work, outcome, code):
    _owe(work.root, "k-1", ready=outcome == "refused")
    _row(work.root, R, "completed")
    if outcome == "refused":
        work.engine.faults = [ClaudexorUnavailable(code, code, status_code=409)]
    else:
        work.engine.lookups["k-1"] = ClaudexorUnavailable(code, code, status_code=404)
    assert work.run()["observed"][0]["outcome"] == outcome
    assert not work.owed()
    current.rebuild(work.root, replace=True)
    receipt = owner_keep.set_owner_keep(work.root, R, keep=True, source=KEEP)
    assert receipt["keep"] and receipt["unknown"] == ["managed_imports"], receipt
    expected = "release_refused:" + code if outcome == "refused" else "binding_unavailable:" + outcome
    assert receipt["releases"] == {"managed_import:k-1": expected}
    assert "managed_import:k-1" not in receipt["retained"] + receipt["retired"]
