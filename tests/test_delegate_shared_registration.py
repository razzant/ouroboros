"""Shared registration custody across starts, sweeps and restarts on a stateful fixture engine."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_custody_usage as usage
from ouroboros import delegate_registration_policy as policy
from ouroboros import delegate_workspace_hold as hold
from ouroboros import execution_domain, owner_pause
from ouroboros.gateways import claudexor as gateway_module
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.task_results import write_task_result
from ouroboros.tools import delegate
from tests._delegated_transport_shared import (
    _LiveRunStub, _nanny_ctx, _select_transport_access,
    _owned_gateway_uses_each_test_transport as _owned_gateway_fixture,
)
from tests._bound_registration import BoundRegistry, STORE, binding, owner_facts
from tests._store_shared import seed_result

_owned_gateway_uses_each_test_transport = _owned_gateway_fixture

WORK_REF = {"kind": execution_domain.WORK_ENGINE, "descriptor": "/admin/work-engine.json", "store": "store-w"}
WAIT = 30  # seconds an event may take: a hung interleaving fails instead of blocking the run


def _fire(engine, name):
    hook = getattr(engine, name)
    if hook is not None:
        setattr(engine, name, None)  # one-shot: the race happens once, at this exact point
        hook()


class Engine(BoundRegistry, _LiveRunStub):
    engine_version, engine_build_sha = "6.0.0", "build-shared-registration"

    def __init__(self, *, engine_ref=None, store_identity=STORE, preexisting=""):
        super().__init__()
        self.engine_ref, self.store_identity, self.preexisting = engine_ref, store_identity, preexisting
        self.projects, self.minted, self.refuse = {}, 0, {}
        self.posts, self.lookups, self.deletes, self.run_posts, self.started = [], [], [], [], []
        self.start_error = self.on_register = self.on_lookup = self.on_start = self.on_delete = None

    def _register(self, root):
        self.posts.append(root)
        created = root not in self.projects
        if created and self.preexisting:  # registered outside Ouroboros before this start
            self.projects[root], created, self.preexisting = self.preexisting, False, ""
        elif created:
            self.minted += 1
            self.projects[root] = f"prj-{self.minted}"
        return self.projects[root], created

    def project_registration(self, root):
        answer = self._register(root)
        _fire(self, "on_register")  # after the engine decided its answer, before the start reads it
        return answer

    def project_registration_receipt(self, root):
        store_id = self.store_identity
        project_id, created = self.project_registration(root)
        return {"id": project_id, "created": created, "registration_root": root, "registration_store_id": store_id}

    def registered_root(self, project_id):
        return next((root for root, pid in self.projects.items() if pid == project_id), None)

    def register_project(self, root):
        return self.project_registration(root)[0]

    def find_project_id(self, root):
        self.lookups.append("find")
        answer = self.projects.get(root, "")
        _fire(self, "on_lookup")
        return answer

    def project_registrations(self):
        self.lookups.append("list")
        _fire(self, "on_lookup")
        return [{"id": pid, "root": root} for root, pid in sorted(self.projects.items())]

    def release_registered(self, project_id):
        self.deletes.append(project_id)
        _fire(self, "on_delete")
        if project_id in self.refuse:
            raise ClaudexorUnavailable(self.refuse[project_id], "refused", status_code=409)
        for root, known in list(self.projects.items()):
            if known == project_id:
                del self.projects[root]
                return "released"
        return "absent"

    def start_run(self, request, *, idempotency_key=""):
        self.run_posts.append(idempotency_key)
        _fire(self, "on_start")  # the POST is in flight: its row is recorded, its answer not yet back
        if self.start_error is not None:
            error, self.start_error = self.start_error, None
            raise error
        scope = request.get("scope") or {}
        if scope.get("root") not in self.projects and not scope.get("ephemeral"):
            raise ClaudexorUnavailable("project_not_registered", "register the root first", status_code=404)
        self.started.append(idempotency_key)
        return {"runId": f"run-{len(self.started)}", "storeId": self.store_identity}


class LegacyEngine(Engine):
    """An engine whose registration answers carry no ``created``: the start lists, then registers what is missing,
    and ``find_project_id`` is its client's only lookup."""

    project_registration_receipt = operations = None
    project_registration = None
    project_registrations = None

    def register_project(self, root):
        return self._register(root)[0]


class LegacyRegistrationReply(Engine):
    operations = None

    def project_registration(self, root):
        project_id, _ = super().project_registration(root)
        return project_id, None


@pytest.fixture
def w(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "_CREATED_REPORTED", {})
    for task in ("t-a", "t-b", "t-c"):
        write_task_result(tmp_path, task, "running", root_task_id=task)
    w = SimpleNamespace(root=tmp_path, fenced={}, engine=None)

    def use(engine):
        w.engine = engine
        monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: engine)
        return engine

    w.use = use
    use(Engine())
    _select_transport_access(monkeypatch, "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    real = delegate.run_operation

    def run_operation(source, function, *args, **kwargs):
        task = str(getattr(source, "task_id", "") or "")
        if task in w.fenced:  # what runs at this task's launch fence before the fence refuses it: a definite nonstart
            w.fenced.pop(task)()
            raise owner_pause.OwnerPauseRefused("owner_pause")
        return real(source, function, *args, **kwargs)

    monkeypatch.setattr(delegate, "run_operation", run_operation)
    return w


def _start(root, task, **kwargs):
    return json.loads(delegate._delegate_start(_nanny_ctx(root, task), "inspect the source", **kwargs).text)


def _rows(root, kind, task=None):
    return [row for row in custody.custody_rows(root) if row.get("type") == kind and task in (None, row.get("task_id"))]


def _owed(root):
    return sorted(row["project_id"] for row in current.owed_registrations(root))


def _sweep(root, gateway, live=()):
    return custody.reconcile_orphaned_runs(root, set(live), gateway_factory=lambda: gateway)


def _finish(root, *tasks):
    for task in tasks:
        seed_result(root, task, {"status": "completed", "root_task_id": task})


def _settle(root, run_id, task):
    assert custody.emit(root, custody.SETTLED, {"run_id": run_id, "task_id": task, "route": "some-route"})


def _after_the_fence(monkeypatch, action):
    """Run ``action`` once in a start's window after its fence let go: its POST answered (an adopted registration has
    no preparation of this start yet) and nothing is claimed; its confirmation under the project lock comes next. A
    POST itself runs under the route's workspace fence, which a retirer also takes, so a race reaches it here."""
    real, pending = policy.checked_registration, [action]

    @contextmanager
    def window(*args, **kwargs):
        if pending:
            pending.pop()()
        with real(*args, **kwargs) as held:
            yield held

    monkeypatch.setattr(policy, "checked_registration", window)


def _a_never_runs_while_b_is_pending(w):
    """A creates the registration; at A's launch fence B adopts it and POSTs, and B's answer is lost; then A's
    fence refuses A (it definitely never ran). Returns (A's result, B's result)."""
    b = {}

    def b_starts_and_loses_its_answer():
        w.engine.start_error = ClaudexorUnavailable("daemon_unreachable", "POST answer lost", status_code=0)
        b.update(_start(w.root, "t-b"))

    w.fenced["t-a"] = b_starts_and_loses_its_answer
    return _start(w.root, "t-a"), b


def test_a_creator_that_never_ran_keeps_what_a_pending_start_still_needs(w):
    a, b = _a_never_runs_while_b_is_pending(w)
    assert a["reason"] == "owner_pause" and a["definitely_unrun"] is True and a["project_id"] == "prj-1"
    [failed] = _rows(w.root, custody.START_FAILED, "t-a")
    assert failed["definite"] is True
    token = b["pending_invocation_id"]
    assert custody.invocation_record(w.root, token)["state"] == "pending", "B's POST answer is UNKNOWN"
    assert w.engine.run_posts == [token] and w.engine.started == []
    assert w.engine.deletes == [], "a POST whose answer was lost may have a run on this registration"
    assert (a["project_retired"], a["project_retention_reason"]) == (False, policy.REGISTRATION_IN_USE)
    assert (failed["project_retired"], failed["project_retention_reason"]) == (False, "retirement_owed")
    [owed] = current.owed_registrations(w.root)
    assert (owed["project_id"], owed["task_id"], owed["invocation_id"]) == ("prj-1", "t-a", failed["invocation_id"])
    assert len(w.engine.posts) == 2 and "list" not in w.engine.lookups, (
        "B adopted while A's request was pending, a use every retirer reads first: nothing to ask the engine")


@pytest.mark.parametrize("answer", ["removed", "absent", "project_has_threads"])
def test_the_owed_duty_survives_a_restart_and_is_discharged_once_after_the_last_use(w, answer):
    _a_never_runs_while_b_is_pending(w)
    token = _rows(w.root, custody.START_REQUESTED, "t-b")[0]["invocation_id"]
    # A restart: nothing of this process is kept; the store and the chain are.
    custody._CUSTODY.clear()
    policy._CREATED_REPORTED.clear()
    replayed = current.fold(custody.custody_rows(w.root))["registrations"]
    assert sorted(row["project_id"] for row in replayed.values()) == _owed(w.root) == ["prj-1"]
    _sweep(w.root, w.engine, live={"t-a", "t-b"})
    assert w.engine.deletes == [], "B's start is still pending"
    # The original POST was accepted. Bind its late receipt to B without replay into a named store.
    from ouroboros.delegate_custody_reconcile import recovered_custody
    request = custody.invocation_record(w.root, token)
    entry, shape = recovered_custody(w.root, w.engine, {**request, "invocation_id": token}, "late-run",
                                      daemon_store_id=w.engine.store_identity)
    assert custody.bind_started(w.root, entry, shape=shape, exclusive=True).durable
    _sweep(w.root, w.engine, live={"t-b"})
    assert w.engine.deletes == [], "B's run is live"
    _settle(w.root, entry.run_id, "t-b")
    _finish(w.root, "t-a", "t-b")
    if answer == "absent":
        w.engine.projects.clear()  # removed outside custody meanwhile: the absence discharges the duty
    elif answer == "project_has_threads":
        w.engine.refuse["prj-1"] = answer
    _sweep(w.root, w.engine)
    assert w.engine.deletes == ["prj-1"]
    [retired] = _rows(w.root, custody.PROJECT_RETIRED)
    assert retired["project_id"] == "prj-1" and retired["run_id"] == ""
    assert bool(retired.get("project_kept")) is (answer == "project_has_threads")
    assert _owed(w.root) == [] and current.fold(custody.custody_rows(w.root))["registrations"] == {}
    assert _rows(w.root, custody.PROJECT_RETIRE_FAILED) == [], "a typed permanent refusal is discharged once"
    for _ in range(2):
        _sweep(w.root, w.engine)
    assert w.engine.deletes == ["prj-1"], "exactly once"


def _creator_run_ended(w):
    created = _start(w.root, "t-c")
    assert created["status"] == "started" and w.engine.projects
    _settle(w.root, created["run_id"], "t-c")
    _finish(w.root, "t-c")
    assert [row.run_id for row in custody.owned_project_registrations(w.root)] == [created["run_id"]]


def _b_refused_with_nothing_sent(w, b, reason="registration_replaced"):
    assert (b["reason"], b["definitely_unrun"]) == (reason, True)
    assert _rows(w.root, custody.START_REQUESTED, "t-b") == [], "no row names the removed id"
    assert _rows(w.root, custody.START_FAILED, "t-b")[-1]["definite"] is True
    assert "t-b" not in {row.get("task_id") for row in _rows(w.root, custody.STARTED)}
    assert not any(row.get("task_id") == "t-b" for row in current.owed_registrations(w.root))


def test_a_retirement_that_reads_an_adopters_recorded_request_waits_for_it(w):
    _creator_run_ended(w)
    w.engine.on_start = lambda: _sweep(w.root, w.engine, live={"t-b"})
    b = _start(w.root, "t-b")
    assert b["status"] == "started" and w.engine.deletes == []
    [request] = _rows(w.root, custody.START_REQUESTED, "t-b")
    assert (request["project_id"], request["project_owned"]) == ("prj-1", False), "B adopted C's registration"
    _sweep(w.root, w.engine, live={"t-b"})
    assert w.engine.deletes == [], "B's run is live"
    _settle(w.root, b["run_id"], "t-b")
    _finish(w.root, "t-b")
    _sweep(w.root, w.engine)
    _sweep(w.root, w.engine)
    assert w.engine.deletes == ["prj-1"] and _owed(w.root) == []


@pytest.mark.parametrize("existing", [False, True], ids=["creator", "adopter"])
def test_two_preparations_can_share_a_registration_until_their_starts_land(w, monkeypatch, existing):
    if existing:
        _creator_run_ended(w)
    b = {}

    def import_documents(ctx, *_args):
        assert not _rows(w.root, custody.START_REQUESTED, ctx.task_id)
        if ctx.task_id == "t-a":
            b.update(_start(w.root, "t-b"))  # Both preparations overlap; no project-wide import lock.
        else:
            custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
            assert not w.engine.deletes and list(w.engine.projects.values()) == ["prj-1"]
        return delegate.NO_DOCUMENTS

    monkeypatch.setattr(delegate, "attach_selected_documents", import_documents)
    a = _start(w.root, "t-a")
    assert a["status"] == b["status"] == "started"
    assert w.engine.run_posts[-2:] == [b["invocation_id"], a["invocation_id"]]
    for task, start in (("t-a", a), ("t-b", b)):
        _settle(w.root, start["run_id"], task)
    _finish(w.root, "t-a", "t-b")
    for _ in range(2):
        custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == ["prj-1"] and not _owed(w.root)


def test_a_definite_preparation_refusal_releases_its_use(w, monkeypatch):
    def refuse(*_args):
        custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
        assert not w.engine.deletes
        raise ClaudexorUnavailable("document_import_finished", "fixture preparation refused")

    monkeypatch.setattr(delegate, "attach_selected_documents", refuse)
    a = _start(w.root, "t-a")
    assert a["definitely_unrun"] and not w.engine.run_posts
    assert not _rows(w.root, custody.START_REQUESTED, "t-a")
    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == ["prj-1"] and not _owed(w.root)


def test_a_retirement_inside_an_adopters_window_refuses_the_start_before_anything_is_sent(w, monkeypatch):
    _creator_run_ended(w)
    _after_the_fence(monkeypatch, lambda: _sweep(w.root, w.engine, live={"t-b"}))
    b = _start(w.root, "t-b")
    assert w.engine.deletes == ["prj-1"] and w.engine.projects == {}
    _b_refused_with_nothing_sent(w, b)
    assert w.engine.run_posts == [w.engine.started[0]], "only C's run was ever POSTed"
    again = _start(w.root, "t-b")
    assert again["status"] == "started" and _rows(w.root, custody.START_REQUESTED, "t-b")[-1]["project_owned"]


@pytest.mark.parametrize("generation", ["modern", "legacy"])
def test_an_adopter_never_claims_a_registration_its_creator_removed_meanwhile(w, monkeypatch, generation):
    engine = w.use(Engine() if generation == "modern" else LegacyEngine())
    fenced, adopted = threading.Event(), threading.Event()
    w.fenced["t-a"] = lambda: (fenced.set(), adopted.wait(WAIT))
    with ThreadPoolExecutor(max_workers=1) as worker:
        first = worker.submit(_start, w.root, "t-a")
        try:
            assert fenced.wait(WAIT)
            # B adopted (its lookup or POST answered); A's refusal and cleanup run before B's confirmation.
            _after_the_fence(monkeypatch, lambda: (adopted.set(), first.result(timeout=WAIT)))
            b = _start(w.root, "t-b")
        finally:
            adopted.set()
        a = first.result(timeout=WAIT)
    if generation == "legacy":
        assert a["definitely_unrun"] and not a["project_retired"]
        assert b["status"] == "started" and not engine.deletes
        return
    assert a["definitely_unrun"] and a["project_retired"]
    assert engine.deletes == ["prj-1"] and engine.run_posts == []
    _b_refused_with_nothing_sent(w, b)
    assert len(engine.posts) == 2 and "list" not in engine.lookups, "the refused recheck read the binding's address"


def test_an_adopters_request_waits_for_a_removal_in_progress_and_never_lands_on_a_removed_registration(
        w, monkeypatch):
    """B's request waits at the route's fence while the release of the folder's registration is in flight; nothing of
    B is asked, claimed or sent meanwhile. Then the folder is unregistered: B registers it anew and starts there, never
    on the removed id."""
    _creator_run_ended(w)
    deleting, waiting, release = threading.Event(), threading.Event(), threading.Event()
    w.engine.on_delete = lambda: (deleting.set(), release.wait(WAIT))
    real = usage.project_retirement_lock

    @contextmanager
    def observed(root, project_id, **kw):
        if deleting.is_set() and project_id in ("prj-1", hold.FOLDER):
            waiting.set()
        with real(root, project_id, **kw):
            yield

    monkeypatch.setattr(usage, "project_retirement_lock", observed)
    with ThreadPoolExecutor(max_workers=2) as worker:
        sweep = worker.submit(_sweep, w.root, w.engine, {"t-b"})
        try:
            assert deleting.wait(WAIT)
            posts = list(w.engine.posts)
            start = worker.submit(_start, w.root, "t-b")
            assert waiting.wait(WAIT)
            assert _rows(w.root, custody.START_REQUESTED, "t-b") == [] and w.engine.posts == posts
        finally:
            release.set()
        sweep.result(timeout=WAIT)
        b = start.result(timeout=WAIT)
    [request] = _rows(w.root, custody.START_REQUESTED, "t-b")
    assert w.engine.deletes == ["prj-1"] and b["status"] == "started"
    assert request["project_id"] not in ("", "prj-1") and request["project_owned"], "a fresh registration of its own"


def test_one_project_id_on_two_engines_is_two_registrations(tmp_path, monkeypatch):
    owned, work = Engine(), Engine(engine_ref=WORK_REF, store_identity="store-w")
    for engine in (owned, work):
        engine.projects["/shared"] = "prj-same"
    monkeypatch.setattr(execution_domain, "work_gateway", lambda ref, **_kw: work)

    def run(run_id, task, *, owner, engine_ref=None, settled=True):
        assert custody.record_started(tmp_path, custody.RunCustody(
            run_id=run_id, task_id=task, root_task_id=task, route_id="r", model="m", project_id="prj-same",
            project_owned=owner, ledger_root=str(tmp_path),
            **owner_facts(engine_ref["store"] if engine_ref else STORE, "/shared", engine_ref=dict(engine_ref or {}))))
        seed_result(tmp_path, task, {"status": "completed" if settled else "running", "root_task_id": task})
        if settled:
            _settle(tmp_path, run_id, task)

    run("run-o", "t-o", owner=True)
    run("run-w", "t-w", owner=True, engine_ref=WORK_REF)
    run("run-wl", "t-wl", owner=False, engine_ref=WORK_REF, settled=False)  # live on the work engine only
    custody._CUSTODY.clear()
    _sweep(tmp_path, owned, live={"t-wl"})
    assert (owned.deletes, work.deletes) == (["prj-same"], []), "the work engine's live run holds only its own"
    assert owned.projects == {} and work.projects == {"/shared": "prj-same"}
    replay = custody.replay(tmp_path)
    assert (replay["run-o"].project_owned, replay["run-w"].project_owned) == (False, True)
    folded = current.fold(custody.custody_rows(tmp_path))["runs"]
    assert (folded["run-o"][0]["project_owned"], folded["run-w"][0]["project_owned"]) == (False, True)
    _settle(tmp_path, "run-wl", "t-wl")
    seed_result(tmp_path, "t-wl", {"status": "completed", "root_task_id": "t-wl"})
    custody._CUSTODY.clear()
    for _ in range(2):
        _sweep(tmp_path, owned)
    assert (owned.deletes, work.deletes) == (["prj-same"], ["prj-same"])
    assert owned.projects == work.projects == {}
    assert custody.owned_project_registrations(tmp_path) == []
    [_, work_retired] = _rows(tmp_path, custody.PROJECT_RETIRED)
    assert work_retired["run_id"] == "run-w" and work_retired["engine_ref"] == WORK_REF


@pytest.mark.parametrize("lost", [custody.START_UNSENT, custody.START_FAILED], ids=["unsent-close", "cleanup-row"])
def test_failed_definite_publication_keeps_the_request_and_prevents_delete(w, monkeypatch, lost):
    """A lost unsent close keeps the request pending and unknown; a cleanup row lost after that close keeps the
    creation's duty for the sweep. Neither deletes nor sends anything."""
    emit = custody.emit
    monkeypatch.setattr(custody, "emit", lambda root, kind, row, **kw:
                        False if kind == lost else emit(root, kind, row, **kw))
    w.fenced["t-a"] = lambda: None
    result = _start(w.root, "t-a")
    [requested] = _rows(w.root, custody.START_REQUESTED, "t-a")
    state = custody.invocation_record(w.root, requested["invocation_id"])["state"]
    if lost == custody.START_UNSENT:
        assert not result.get("definitely_unrun") and result["pending_invocation_id"] == requested["invocation_id"]
        [request] = custody.pending_invocations(w.root)
        assert request["project_id"] == "prj-1" and state == "pending"
    else:
        assert result["definitely_unrun"] and custody.pending_invocations(w.root) == [] and state == "failed_definite"
    assert not result["project_retired"] and _owed(w.root) == ["prj-1"]
    assert not w.engine.deletes and not w.engine.run_posts


@pytest.mark.parametrize("owner", ["tool", "review"])
@pytest.mark.parametrize("failure_lands", [True, False], ids=["recorded", "write-lost"])
def test_unsent_request_closes_before_delete_even_if_discharge_publication_fails(w, monkeypatch, owner, failure_lands):
    """B has its engine answer. A's DELETE lands after its refusal; losing the discharge must never reopen A."""
    from ouroboros import review_execution

    w.engine.store_identity = "s"
    pid, owned, _, _, resolution = policy.resolve_for_start(w.root, w.engine, str(w.root), "", "readonly")
    assert pid == owned == "prj-1"
    assert custody.record_start_requested(w.root, gateway=w.engine, registration_resolution=resolution,
        task_id="t-a", invocation_id="inv-a", idempotency_key="inv-a", run_id="", project_id=pid, project_owned=True,
        route="r", request={"prompt": "a", "scope": {"kind": "project", "root": str(w.root)}})
    removed, claiming = threading.Event(), threading.Event()
    real_emit, real_lock = custody.emit, usage.project_retirement_lock

    def publish(root, kind, row, **kwargs):
        if kind == custody.PROJECT_RETIRED and row.get("project_id") == pid:
            assert not w.engine.projects and w.engine.deletes == ["prj-1"]
            assert custody.invocation_record(root, "inv-a")["state"] == "failed_definite"
            removed.set()
            assert claiming.wait(WAIT)
            if not failure_lands:
                return False
        return real_emit(root, kind, row, **kwargs)

    @contextmanager
    def observe_claim(root, project_id, **kw):
        if project_id == "prj-1" and removed.is_set():
            claiming.set()
        with real_lock(root, project_id, **kw):
            yield

    monkeypatch.setattr(custody, "emit", publish)
    monkeypatch.setattr(usage, "project_retirement_lock", observe_claim)

    def refuse():
        facts = dict(reason="owner_pause", definite_refusal=True, invocation_id="inv-a", resolution=resolution)
        if owner == "review":
            return review_execution._retire_orphaned_review_registration(
                custody, w.engine, w.root, "prj-1", task_id="t-a", **facts)
        return delegate._retire_orphaned_registration(_nanny_ctx(w.root, "t-a"), w.engine, "prj-1",
                                                       project_persistent=False, **facts)["project_retired"]

    with ThreadPoolExecutor(max_workers=1) as worker:
        cleanups = []

        def after_adoption():  # B's POST answered under its fence; A's cleanup runs before B's confirmation
            cleanups.append(worker.submit(refuse))
            assert removed.wait(WAIT)

        _after_the_fence(monkeypatch, after_adoption)
        result = _start(w.root, "t-b")
        assert cleanups[0].result(timeout=WAIT) is failure_lands
    assert claiming.is_set() and w.engine.run_posts == []
    _b_refused_with_nothing_sent(w, result, "registration_replaced" if failure_lands else "registration_unconfirmed")
    assert custody.invocation_record(w.root, "inv-a")["state"] == "failed_definite"
    if not failure_lands:
        assert _owed(w.root) == [pid]
        monkeypatch.setattr(custody, "emit", real_emit)
        current.rebuild(w.root, replace=True)
        _sweep(w.root, w.engine)  # the same release again: absent, so nothing more is removed
        [retired] = _rows(w.root, custody.PROJECT_RETIRED)
        assert retired["removal_outcome"] == "gone_after_unknown" and not w.engine.projects
    assert not _owed(w.root) and w.engine.deletes == [pid] * (1 if failure_lands else 2)


@pytest.mark.parametrize("first_refusal", ["t-a", "t-b"], ids=["creator-first", "adopter-first"])
@pytest.mark.parametrize("legacy", [LegacyEngine, LegacyRegistrationReply], ids=["old-client", "no-created-field"])
def test_legacy_parallel_creators_have_one_owner_and_keep_the_registration(w, monkeypatch, first_refusal, legacy):
    engine = w.use(legacy())
    negative, competing = threading.Event(), threading.Event()
    tasks = ("t-a", "t-b")
    ready = {task: threading.Event() for task in tasks}
    release = {task: threading.Event() for task in tasks}
    lock = usage.project_retirement_lock

    @contextmanager
    def observe_resolution(root, key, **kw):
        if key.startswith("root:") and negative.is_set():
            competing.set()
        with lock(root, key, **kw):
            yield

    def first_lookup():
        negative.set()
        assert competing.wait(WAIT)

    def fence(task):
        ready[task].set()
        assert release[task].wait(WAIT)

    monkeypatch.setattr(usage, "project_retirement_lock", observe_resolution)
    engine.on_lookup = first_lookup
    for task in tasks:
        w.fenced[task] = lambda task=task: fence(task)
    with ThreadPoolExecutor(max_workers=2) as worker:
        try:
            starts = {"t-a": worker.submit(_start, w.root, "t-a")}
            assert negative.wait(WAIT)
            starts["t-b"] = worker.submit(_start, w.root, "t-b")
            assert all(event.wait(WAIT) for event in ready.values())
            assert {row["task_id"]: row["project_owned"] for row in _rows(w.root, custody.START_REQUESTED)} == {
                "t-a": True, "t-b": False}
            release[first_refusal].set()
            assert starts[first_refusal].result(timeout=WAIT)["definitely_unrun"]
            assert engine.deletes == [] and len(engine.projects) == 1
        finally:
            for event in release.values():
                event.set()
        assert all(start.result(timeout=WAIT)["definitely_unrun"] for start in starts.values())
    for _ in range(2):
        _sweep(w.root, engine)
    assert len(engine.posts) == 1 and not engine.deletes
    assert list(engine.projects.values()) == ["prj-1"] and _owed(w.root) == []


def test_an_unbound_use_holds_named_stores_without_discharging_their_duties(tmp_path):
    write_task_result(tmp_path, "task", "running", root_task_id="task")
    a, b = Engine(store_identity="a"), Engine(store_identity="b")
    for engine in (a, b):
        engine.projects["/shared"] = "p"
        assert policy.owe_registration(tmp_path, "p", binding(engine.store_identity, "/shared"))
    assert policy.owe_registration(tmp_path, "p", {})  # historical owned-engine duty without a bound store
    assert custody.record_start_requested(tmp_path, run_id="", invocation_id="unbound", idempotency_key="unbound",
        task_id="task", project_id="p", project_persistent=True, route="r", request={"prompt": "p"},
        store_identity_unproven=True, handshake_store_id="a")
    for engine in (a, b):
        custody.retire_settled_registrations(tmp_path, engine)
    assert a.deletes == b.deletes == [] and len(current.owed_registrations(tmp_path)) == 3
    assert custody.emit(tmp_path, custody.START_FAILED, {"run_id": "", "task_id": "task",
                        "invocation_id": "unbound", "definite": True})
    for engine, expected_remaining in [(a, {"", "b"}), (b, {""})]:
        custody._CUSTODY.clear()
        custody.retire_settled_registrations(tmp_path, engine)
        assert engine.deletes == ["p"] and not engine.projects
        assert {row.get("daemon_store_id", "") for row in current.owed_registrations(tmp_path)} == expected_remaining
        folded = current.fold(custody.custody_rows(tmp_path))["registrations"]
        assert {row.get("daemon_store_id", "") for row in folded.values()} == expected_remaining
        if engine is a:
            assert b.projects == {"/shared": "p"} and b.deletes == []


@pytest.mark.parametrize("first", ["t-a", "t-b"])
def test_two_legacy_owner_rows_discharge_the_same_duty_once(w, first):
    from ouroboros import review_execution

    engine = w.use(LegacyEngine())
    engine.projects["/shared"] = "p"
    second = "t-b" if first == "t-a" else "t-a"
    for task in (first, second):  # both pre-upgrade requests believed their lookup created the project
        assert custody.record_start_requested(w.root, task_id=task, invocation_id=task, idempotency_key=task,
            run_id="", project_id="p", project_owned=True, project_persistent=False, route="r", request={"prompt": task})
    def refuse(task):
        return review_execution._retire_orphaned_review_registration(custody, engine, w.root, "p",
            task_id=task, invocation_id=task, reason="owner_pause", definite_refusal=True)

    assert not refuse(first) and engine.deletes == [] and _owed(w.root) == ["p"]
    custody._CUSTODY.clear()
    assert not refuse(second) and engine.projects == {"/shared": "p"} and _owed(w.root) == []
    engine.projects = {"/reused": "p"}
    assert not refuse(second)
    _sweep(w.root, engine)
    assert engine.projects == {"/reused": "p"} and not engine.deletes
    assert current.fold(custody.custody_rows(w.root))["registrations"] == {}


def test_an_unsent_review_successor_hands_its_registration_to_the_sweep(w):
    from ouroboros.review_continuation import START_UNSENT, _retire_start

    w.engine.store_identity = "review-store"
    pid, owned, _, _, resolution = policy.resolve_for_start(w.root, w.engine, "/shared", "", "readonly")
    assert owned == pid
    assert custody.record_start_requested(w.root, gateway=w.engine, registration_resolution=resolution,
        task_id="t-a", invocation_id="review", idempotency_key="review", project_id=pid, project_owned=True,
        request={"scope": {"kind": "project", "root": "/shared"}}, source="review_substrate")
    decision = {"invocation_id": "review", "surface": "multi_model_review"}
    _retire_start(w.root, "t-a", "seat", decision, START_UNSENT)
    _retire_start(w.root, "t-a", "seat", decision, START_UNSENT)
    assert custody.invocation_record(w.root, "review")["state"] == "failed_definite" and w.engine.deletes == []
    [owed] = current.owed_registrations(w.root)
    assert (owed["project_id"], owed["daemon_store_id"]) == (pid, "review-store")
    assert list(current.fold(custody.custody_rows(w.root))["registrations"].values()) == [owed]
    custody.retire_settled_registrations(w.root, w.engine)
    assert w.engine.deletes == [pid] and not w.engine.projects and not _owed(w.root)


@pytest.mark.parametrize("path", ["new-claim", "explicit-retry", "automatic-recovery"])
def test_pending_row_cannot_prove_the_engine_still_has_the_registration(w, path):
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation

    w.engine.store_identity = ""
    body = {"scope": {"kind": "project", "root": "/shared"}, "prompt": "old request"}
    facts = dict(run_id="", task_id="t-a", idempotency_key="a", invocation_id="a", project_id="p", route="r",
                 request=body, project_owned=False)
    if path == "automatic-recovery":  # the store permits exact replay, but the registration is gone
        w.engine.store_identity = facts["daemon_store_id"] = "s"
    assert custody.record_start_requested(w.root, **facts)  # old row, no registry observation
    current.mark_removal(w.root, "p", {}, "custody")
    current.mark_removal(w.root, "p", {}, "")
    if path == "automatic-recovery":
        result = _recover_pending_invocation(w.root, w.engine, custody.pending_invocations(w.root)[0])
        assert (result["action"], result["reason"]) == ("invocation_retained", "registration_replaced")
    else:
        if path == "new-claim":
            facts.update(task_id="t-b", idempotency_key="b", invocation_id="b")
        with pytest.raises(ClaudexorUnavailable) as rejected:
            custody.record_start_requested(w.root, gateway=w.engine, **facts)
        assert rejected.value.code == "registration_replaced"
        if path == "explicit-retry":
            assert not getattr(rejected.value, "delegated_start_unrequested", False)
    assert w.engine.lookups == ["list"] and w.engine.posts == w.engine.run_posts == []
    assert len(_rows(w.root, custody.START_REQUESTED)) == 1
    assert custody.invocation_record(w.root, "a")["state"] == "pending"


def test_malformed_inventory_cannot_prove_a_stale_registration_absent(w, monkeypatch):
    pid, _, _, _, observed = policy.resolve_for_start(w.root, w.engine, "/shared", "", "readonly")
    current.mark_removal(w.root, pid, {}, "custody")
    current.mark_removal(w.root, pid, {}, "")
    monkeypatch.setattr(w.engine, "operations", lambda: [])  # a legacy engine: the list is its only confirmation

    def malformed():
        raise ClaudexorUnavailable("malformed_response", "invalid project row")

    monkeypatch.setattr(w.engine, "project_registrations", malformed)
    with pytest.raises(ClaudexorUnavailable) as refused:
        custody.record_start_requested(w.root, gateway=w.engine, registration_resolution=observed,
            invocation_id="attempt", project_id=pid, request={"scope": {"root": "/shared"}})
    assert refused.value.code == "malformed_response"
    assert not w.engine.run_posts and _owed(w.root) == [pid]


def test_an_unrelated_removals_stamp_change_confirms_through_one_addressed_read(w, monkeypatch):
    """A concurrent removal elsewhere bumps the global stamp; the fresh claim this creation just made is
    re-verified by the creation receipt's own address — one addressed GET — and a malformed inventory (which
    once refused such a start and owed a duty for it) is never listed."""
    def removal_elsewhere():  # inside the receipt: after the engine decided its answer
        current.mark_removal(w.root, "prj-elsewhere", policy.engine_facts(w.engine), "custody")

    def malformed():
        raise ClaudexorUnavailable("malformed_response", "invalid project row")

    w.engine.on_register = removal_elsewhere
    monkeypatch.setattr(w.engine, "project_registrations", malformed)
    pid, owned, _, _, resolution = policy.resolve_for_start(w.root, w.engine, "/shared", "", "readonly",
                                                            task_id="t-a", invocation_id="inv-a")
    assert (pid, owned) == ("prj-1", "prj-1")
    assert (resolution["registration_root"], resolution["registration_store_id"]) == ("/shared", STORE)
    assert resolution["creator_invocation"] == "inv-a"
    assert w.engine.lookups == ["find"], "the inventory was never listed"
    [duty] = current.owed_registrations(w.root)
    assert duty["reason"] == "start_preparation", "the created registration keeps its ordinary owed duty"


def test_a_created_registration_whose_addressed_confirmation_fails_stays_owed_to_the_sweep(w, monkeypatch):
    """The creation answer already made the registration ours, and its ordinary owed duty stands from the creation's
    own facts before any confirming read. An addressed read that then fails — the daemon went away between the
    creation and the check — refuses the start before the caller ever holds the id; the sweep discharges the duty
    once nothing uses it."""
    def removal_elsewhere():  # inside the receipt: after the engine decided its answer
        current.mark_removal(w.root, "prj-elsewhere", policy.engine_facts(w.engine), "custody")

    w.engine.on_register = removal_elsewhere
    bound = w.engine.bound_registration

    def refused_read(project_id, store_id, root, *, release=False):
        if not release:  # the confirming read fails; the sweep's own release still goes through
            raise ClaudexorUnavailable("daemon_recovery_only", "the addressed confirmation failed", status_code=503)
        return bound(project_id, store_id, root, release=release)

    monkeypatch.setattr(w.engine, "bound_registration", refused_read)
    with pytest.raises(ClaudexorUnavailable) as refused:
        policy.resolve_for_start(w.root, w.engine, "/shared", "", "readonly", task_id="t-a", invocation_id="inv-a")
    assert refused.value.code == "daemon_recovery_only"
    assert getattr(refused.value, "delegated_start_unrequested", False), "the typed refusal still travels"
    assert list(w.engine.projects.values()) == ["prj-1"] and not w.engine.deletes, "nothing was sent"
    [duty] = current.owed_registrations(w.root)
    assert (duty["project_id"], duty["task_id"], duty["invocation_id"], duty["reason"]) == (
        "prj-1", "t-a", "inv-a", "start_preparation"), "recorded with the creation, before the read that refused"
    assert (duty["registration_root"], duty["registration_store_id"], duty["daemon_store_id"]) == (
        "/shared", STORE, STORE)
    custody.retire_settled_registrations(w.root, w.engine)
    assert w.engine.deletes == ["prj-1"] and not _owed(w.root)


def test_a_created_registration_whose_legacy_confirmation_list_fails_stays_owed_to_the_sweep(w, monkeypatch):
    """The same refusal on a legacy engine, whose only confirmation is the full inventory: a malformed list
    cannot prove the fresh registration absent, so the start is refused and the ordinary owed duty stays for
    the sweep — one removed registration once nothing uses it."""
    def removal_elsewhere():  # inside the receipt: after the engine decided its answer
        current.mark_removal(w.root, "prj-elsewhere", policy.engine_facts(w.engine), "custody")

    w.engine.on_register = removal_elsewhere

    def malformed():
        raise ClaudexorUnavailable("malformed_response", "invalid project row")

    with monkeypatch.context() as legacy:  # a legacy engine without the bound route, for this claim only
        legacy.setattr(w.engine, "operations", lambda: [])
        legacy.setattr(w.engine, "project_registrations", malformed)
        with pytest.raises(ClaudexorUnavailable) as refused:
            policy.resolve_for_start(w.root, w.engine, "/shared", "", "readonly", task_id="t-a", invocation_id="inv-a")
    assert refused.value.code == "malformed_response"
    assert getattr(refused.value, "delegated_start_unrequested", False), "the typed refusal still travels"
    assert list(w.engine.projects.values()) == ["prj-1"] and not w.engine.deletes, "nothing was sent"
    [duty] = current.owed_registrations(w.root)
    assert (duty["project_id"], duty["task_id"], duty["invocation_id"], duty["reason"]) == (
        "prj-1", "t-a", "inv-a", "start_preparation"), "recorded with the creation, before the read that refused"
    assert (duty["registration_root"], duty["registration_store_id"], duty["daemon_store_id"]) == (
        "/shared", STORE, STORE)
    custody.retire_settled_registrations(w.root, w.engine)
    assert w.engine.deletes == ["prj-1"] and not _owed(w.root)


def test_refusing_an_explicit_retry_keeps_the_first_post_unknown(w):
    w.engine.store_identity = ""  # the legacy replay rule lets this retry reach the registration gate
    w.engine.start_error = ClaudexorUnavailable("daemon_unreachable", "lost first POST")
    token = _start(w.root, "t-a")["pending_invocation_id"]
    w.engine.projects.clear()
    current.mark_removal(w.root, "prj-1", {}, "historical")
    current.mark_removal(w.root, "prj-1", {}, "")
    refused = _start(w.root, "t-a", retry_of=token)
    assert (refused["reason"], refused["pending_invocation_id"]) == ("registration_replaced", token)
    assert not refused.get("definitely_unrun") and custody.invocation_record(w.root, token)["state"] == "pending"
    assert w.engine.run_posts == [token] and not w.engine.deletes


def test_another_stores_refusal_of_a_repeat_does_not_close_the_original_stores_pending(w, monkeypatch):
    """Store B answering a repeated release with 409 ``store_mismatch`` proves nothing about whether store A
    applied the first one: the intent and the folder fence stay with A's key, a start under B's own store key
    proceeds, and A's return answers the same release once (the sweep once cleared A's mark on B's word)."""
    import httpx

    facts = policy.engine_facts(w.engine)
    w.engine.projects["/shared"] = "prj-1"
    answers = ["lost"]
    bound = w.engine.bound_registration

    def answering(project_id, store_id, root, *, release=False):
        if answers[0] == "lost":  # the first release leaves; its answer never arrives
            raise ClaudexorUnavailable("daemon_unreachable", "answer lost",
                                       status_code=0) from httpx.ReadTimeout("mid-answer")
        if answers[0] == "other-store":  # the daemon restored another base between the checks and the request
            raise ClaudexorUnavailable("store_mismatch", "different store", status_code=409)
        return bound(project_id, store_id, root, release=release)

    monkeypatch.setattr(w.engine, "bound_registration", answering)
    with pytest.raises(ClaudexorUnavailable):
        policy.remove_registration(w.root, w.engine, "prj-1", facts, registration_root="/shared")
    assert current.removal_status(w.root, "prj-1", facts)[0] == "custody", "the unanswered first request stays"
    answers[0] = "other-store"
    with pytest.raises(ClaudexorUnavailable) as refused:
        policy.remove_registration(w.root, w.engine, "prj-1", facts, registration_root="/shared")
    assert refused.value.code == "store_mismatch"
    assert current.removal_status(w.root, "prj-1", facts)[0] == "custody", "another store's word closes nothing"
    assert hold.folder_release(w.root, "/shared", facts) == "prj-1", "the folder fence waits with the intent"
    assert not hold.folder_release(w.root, "/shared", {"daemon_store_id": "store-b"}), "B's own starts proceed"
    answers[0] = "serving"
    assert policy.remove_registration(w.root, w.engine, "prj-1", facts,
                                       registration_root="/shared")[0] == "removed"
    assert w.engine.deletes == ["prj-1"] and not hold.folder_release(w.root, "/shared", facts), "A's answer ends it"


def test_a_store_change_after_creation_leaves_cleanup_with_the_original_store(w):
    original = w.engine.projects
    w.engine.store_identity = "first"

    def replace():
        w.engine.store_identity = "second"
        w.engine.projects = original.copy()  # the same id denotes a real registration in each store

    w.engine.on_register = replace
    refused = _start(w.root, "t-a")
    assert refused["reason"] == "registration_store_changed" and refused["definitely_unrun"]
    assert w.engine.deletes == w.engine.run_posts == [] and w.engine.projects == original
    custody.retire_settled_registrations(w.root, w.engine)
    assert [row["daemon_store_id"] for row in current.owed_registrations(w.root)] == ["first"]
    second = w.engine.projects
    w.engine.store_identity, w.engine.projects = "first", original
    custody.retire_settled_registrations(w.root, w.engine)
    assert not original and list(second.values()) == ["prj-1"] and _owed(w.root) == []


def test_registration_receipt_identifies_a_creator_despite_a_different_cached_store(w, monkeypatch):
    w.engine.store_identity = "cache"

    def receipt(root):
        pid, created = w.engine.project_registration(root)
        return {"id": pid, "created": created, "registration_root": "/actual/root", "registration_store_id": "origin"}

    monkeypatch.setattr(w.engine, "project_registration_receipt", receipt, raising=False)
    assert _start(w.root, "t-a")["reason"] == "registration_store_changed"
    [duty] = _rows(w.root, custody.REGISTRATION_OWED)
    assert (duty["daemon_store_id"], duty["registration_store_id"], duty["registration_root"]) == (
        "origin", "origin", "/actual/root")
    assert not w.engine.run_posts and not w.engine.deletes


def test_a_work_engines_duty_names_its_registry_receipts_store_in_its_own_reference(w, monkeypatch):
    """A work engine whose POST receipt names another store than its handshake: the created registration's duty names
    the receipt's store in its own copy of the engine reference as well (the sweep finds its engine by that copy); the
    serving client's reference is left as it was. Once that store serves again, the duty is retired exactly once."""
    from contextlib import ExitStack

    engine = w.use(Engine(engine_ref=dict(WORK_REF), store_identity="store-w"))
    real = engine.project_registration_receipt
    monkeypatch.setattr(engine, "project_registration_receipt",
                        lambda root: {**real(root), "registration_store_id": "origin"}, raising=False)
    with ExitStack() as creating, pytest.raises(ClaudexorUnavailable) as refused:
        policy.resolve_for_start(w.root, engine, "/fixture/read", "", "readonly", task_id="t-a", invocation_id="a",
                                 preparation=creating)
    assert refused.value.code == "registration_store_changed" and engine.engine_ref == WORK_REF
    [duty] = current.owed_registrations(w.root)
    assert duty["daemon_store_id"] == duty["registration_store_id"] == duty["engine_ref"]["store"] == "origin"
    assert execution_domain.engine_key(duty) == execution_domain.registration_engine(duty) == (
        WORK_REF["descriptor"], "origin")
    engine.store_identity, engine.engine_ref = "origin", {**WORK_REF, "store": "origin"}
    for _ in range(2):
        policy.retire_owed(w.root, engine, live_task_ids=set())
    assert engine.deletes == ["prj-1"] and not current.owed_registrations(w.root) and not engine.run_posts


def test_a_busy_fence_before_anything_is_sent_is_a_typed_definite_refusal(w, monkeypatch):
    """A first start whose route fence stays busy sends nothing: typed, definitely unrun, no registration, no
    request row and no POST."""
    @contextmanager
    def busy(*_args, **_kwargs):
        raise TimeoutError("the workspace lock is busy")
        yield  # pragma: no cover

    monkeypatch.setattr(hold, "workspace_lock", busy)
    refused = _start(w.root, "t-a")
    assert (refused["reason"], refused["definitely_unrun"]) == ("registration_busy", True), refused
    assert not w.engine.posts and not w.engine.run_posts and not _rows(w.root, custody.START_REQUESTED, "t-a")


@pytest.mark.parametrize("admit", ["start", "lookup", "late-binding"])
def test_a_run_admitted_in_another_store_does_not_own_its_namesake(w, admit):
    from ouroboros.delegate_custody_reconcile import recovered_custody

    first, second = w.engine, Engine(store_identity="B")
    first.store_identity = "A"
    second.projects["/unrelated"] = "prj-1"
    if admit == "start":
        first.on_start = lambda: setattr(first, "store_identity", "B")
        started = _start(w.root, "t-a")
        run_id = started["run_id"]
    else:
        first.start_error = ClaudexorUnavailable("daemon_unreachable", "unknown start")
        token = _start(w.root, "t-a")["pending_invocation_id"]
        request = custody.invocation_record(w.root, token)
        run_id = "found"
        entry, shape = recovered_custody(w.root, second, {**request, "invocation_id": token}, run_id,
                                         daemon_store_id="B" if admit == "lookup" else "")
        assert custody.bind_started(w.root, entry, shape=shape, exclusive=True).durable
        if admit == "late-binding":
            custody.bind_daemon_store(w.root, second, entry)
    row = current.run(w.root, run_id)
    assert row.daemon_store_id == "B" and not row.project_owned
    assert [duty["daemon_store_id"] for duty in current.owed_registrations(w.root)] == ["A"]
    _settle(w.root, run_id, "t-a")
    _finish(w.root, "t-a")
    custody.retire_settled_registrations(w.root, second, live_task_ids=set())
    first.store_identity = "A"
    custody.retire_settled_registrations(w.root, first, live_task_ids=set())
    assert first.deletes == ["prj-1"] and not _owed(w.root)
    assert not second.deletes and second.projects == {"/unrelated": "prj-1"}


@pytest.mark.parametrize("with_receipt", [True, False], ids=["attested", "tuple-only"])
def test_interrupted_preparation_leaves_its_created_registration_owed_after_rebuild(w, monkeypatch, with_receipt):
    from contextlib import ExitStack

    w.engine.store_identity = "observed"
    proof = {}
    monkeypatch.setattr(w.engine, "project_registration_receipt", None)
    if with_receipt:
        proof = {"registration_root": "/canonical/shared", "registration_store_id": "answering"}

        def receipt(root):
            pid, created = w.engine.project_registration(root)
            w.engine.store_identity = "answering"
            w.engine.projects = {proof["registration_root"]: pid}
            return {"id": pid, "created": created, **proof}

        monkeypatch.setattr(w.engine, "project_registration_receipt", receipt, raising=False)
    with ExitStack() as preparing:
        pid, owned, _, _, resolution = policy.resolve_for_start(w.root, w.engine, "/shared", "", "readonly",
            task_id="t-a", invocation_id="prepare", preparation=preparing)
        assert pid == owned and not _rows(w.root, custody.START_REQUESTED)
        assert _owed(w.root) == [pid]
        for row in (resolution, _rows(w.root, custody.REGISTRATION_OWED)[0]):
            assert {key: row[key] for key in ("registration_root", "registration_store_id") if key in row} == proof
            assert row["daemon_store_id"] == w.engine.store_identity
    current.rebuild(w.root, replace=True)
    custody.retire_settled_registrations(w.root, w.engine)
    assert w.engine.deletes == ([pid] if with_receipt else []) and not _owed(w.root)


def test_project_lock_separates_engine_routes_but_not_stores_on_one_route(tmp_path, monkeypatch):
    from ouroboros import platform_layer

    paths = []
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", lambda path, **kw: paths.append(path) or 7)
    monkeypatch.setattr(platform_layer, "release_exclusive_file_lock", lambda *_: None)
    for engine in ({}, {"daemon_store_id": "A"}, {"engine_ref": WORK_REF},
                   {"engine_ref": {**WORK_REF, "store": "other"}}):
        with usage.project_retirement_lock(tmp_path, "same", engine=engine):
            pass
    assert paths[0] == paths[1] and paths[2] == paths[3] and paths[0] != paths[2]


def test_a_late_store_binding_cannot_discharge_another_stores_owner(w, monkeypatch):
    other = Engine(store_identity="other")
    for engine in (w.engine, other):
        engine.projects["/shared"] = "p"
    for run, task, store in [("r-a", "t-a", ""), ("r-b", "t-b", "other")]:
        assert custody.record_started(w.root, custody.RunCustody(run_id=run, task_id=task, project_id="p",
            project_owned=True, daemon_store_id=store, route_id="r", ledger_root=str(w.root)))
        _settle(w.root, run, task)
        _finish(w.root, task)
    w.engine.store_identity = ""
    custody.retire_settled_registrations(w.root, w.engine)
    assert not w.engine.deletes  # no census: a non-review owner's availability for Continue is unknown
    publish = custody._release_registration
    def bind_before_discharge(root, row, **facts):
        custody.bind_daemon_store(root, other, current.run(root, row.run_id))
        return publish(root, row, **facts)
    with monkeypatch.context() as racing:
        racing.setattr(custody, "_release_registration", bind_before_discharge)
        custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.projects == other.projects == {"/shared": "p"}
    assert {r.run_id for r in custody.owned_project_registrations(w.root)} == {"r-a", "r-b"}
    custody._CUSTODY.clear()
    custody.retire_settled_registrations(w.root, other, live_task_ids=set())
    assert not other.deletes and not custody.owned_project_registrations(w.root)


@pytest.mark.parametrize("observer,retirer", [("s", ""), ("", "s")])
def test_removal_invalidates_a_resolution_even_if_one_client_lacks_the_store(w, observer, retirer):
    reader, cleaner = Engine(store_identity=observer), Engine(store_identity=retirer)
    reader.projects = cleaner.projects
    _, created, _, _, creation = policy.resolve_for_start(w.root, cleaner, "/shared", "", "readonly")
    pid, owned, _, _, observed = policy.resolve_for_start(w.root, reader, "/shared", "", "readonly")
    assert pid == created and not owned
    if not retirer:
        assert policy.retire_unstarted(w.root, cleaner, pid, resolution=creation) == (False, "registration_unproven")
        assert reader.projects == {"/shared": pid}
        return
    assert policy.retire_unstarted(w.root, cleaner, pid, resolution=creation) == (True, "")
    with pytest.raises(ClaudexorUnavailable) as refused:
        custody.record_start_requested(w.root, gateway=reader, registration_resolution=observed,
            task_id="t-b", invocation_id="b", project_id=pid, request={"scope": {"root": "/shared"}})
    assert refused.value.code == "registration_replaced"
    assert reader.projects == {} and custody.invocation_record(w.root, "b") is None


@pytest.mark.parametrize("status,code", [
    (0, "daemon_unreachable"), (403, "forbidden"), (426, "handshake_required"),
    (503, "daemon_stopping"), (503, "daemon_recovery_only"), (503, "store_identity_unavailable"),
    (409, "active_run_scope_unresolved"), (409, "journal_recovery_required"),
])
def test_definite_delete_refusal_leaves_the_duty_retryable(w, monkeypatch, status, code):
    import httpx

    w.engine.store_identity = "s"
    w.fenced["t-a"] = lambda: None

    def reject(pid):
        if not status:
            raise ClaudexorUnavailable(code, "connect failed") from httpx.ConnectTimeout("connect")
        exc = ClaudexorUnavailable(code, "not applied", status_code=status)
        exc.problem = {"code": code}
        raise exc

    with monkeypatch.context() as temporary:
        temporary.setattr(w.engine, "release_registered", reject)
        refused = _start(w.root, "t-a")
    assert not refused["project_retired"] and _owed(w.root) == ["prj-1"]
    assert not current.removal_status(w.root, "prj-1", policy.engine_facts(w.engine))[0]
    custody.retire_settled_registrations(w.root, w.engine)
    assert w.engine.deletes == ["prj-1"] and not _owed(w.root)


@pytest.mark.parametrize("reply", ["timeout", None, {"registryRemoved": True, "projectId": "other"},
                                   {"registryRemoved": False, "projectId": "prj-1"},
                                   (503, "daemon_busy"), (503, "journal_recovery_required"),
                                   (500, "store_identity_unavailable"), (422, "unexpected_failure")])
def test_uncertain_removal_holds_across_lanes_and_restart_until_its_repeat_is_answered(w, monkeypatch, reply):
    from ouroboros import historical_retirement

    w.engine.store_identity = "s"
    w.fenced["t-a"] = lambda: None
    healthy = w.engine.release_registered

    def no_receipt(pid):
        w.engine.deletes.append(pid)
        if reply == "timeout":
            raise TimeoutError("lost reply")
        if isinstance(reply, tuple):
            status, code = reply
            exc = ClaudexorUnavailable(code, "not confirmed", status_code=status)
            exc.problem = {"code": code}
            raise exc
        return reply

    monkeypatch.setattr(w.engine, "release_registered", no_receipt)
    creator = _start(w.root, "t-a")
    assert creator["project_retention_reason"] == "retirement_owed" and _owed(w.root) == ["prj-1"]
    custody._CUSTODY.clear()
    current.rebuild(w.root, replace=True)
    assert _start(w.root, "t-b")["reason"] == "registration_unconfirmed" and not w.engine.run_posts
    assert historical_retirement._release(w.root, w.engine, "prj-1", {"daemon_store_id": "s"}, "/shared")[0] == "unknown"
    assert w.engine.deletes == ["prj-1"], "another lane neither reads nor repeats it"
    _sweep(w.root, w.engine)  # the same release again, still without a definite answer: the mark stays
    assert w.engine.projects and _owed(w.root) == ["prj-1"] and w.engine.deletes == ["prj-1"] * 2
    assert _start(w.root, "t-b")["reason"] == "registration_unconfirmed" and not w.engine.run_posts
    monkeypatch.setattr(w.engine, "release_registered", healthy)
    _sweep(w.root, w.engine)
    [release] = _rows(w.root, custody.PROJECT_RETIRED)
    assert release["removal_outcome"] == "removed" and not _owed(w.root) and not w.engine.projects
    assert w.engine.deletes == ["prj-1"] * 3 and _start(w.root, "t-b")["status"] == "started"


def test_failed_discharge_cannot_delete_a_reused_id_even_with_two_owners(w, monkeypatch):
    w.engine.store_identity = "s"
    _creator_run_ended(w)
    facts = policy.engine_facts(w.engine)
    assert policy.owe_registration(w.root, "prj-1", facts)
    emit = custody.emit
    with monkeypatch.context() as lost:
        lost.setattr(custody, "emit", lambda root, kind, row, **kw:
                     False if kind == custody.PROJECT_RETIRED else emit(root, kind, row, **kw))
        custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert not w.engine.projects and w.engine.deletes == ["prj-1"]
    assert _owed(w.root) == ["prj-1"] and custody.owned_project_registrations(w.root)
    w.engine.projects["/new-owner"] = "prj-1"
    custody._CUSTODY.clear()
    current.rebuild(w.root, replace=True)
    _sweep(w.root, w.engine)  # the repeat finds the id bound to another folder: a dead binding ends the mark
    assert w.engine.projects == {"/new-owner": "prj-1"} and w.engine.deletes == ["prj-1"]
    assert not _rows(w.root, custody.PROJECT_RETIRED) and _owed(w.root) == ["prj-1"]
    assert not current.removal_status(w.root, "prj-1", facts)[0]
    w.engine.projects.clear()
    _sweep(w.root, w.engine)
    [release] = _rows(w.root, custody.PROJECT_RETIRED)
    assert release["removal_outcome"] == "absent" and not _owed(w.root)
    assert current.run(w.root, "run-1").registration_settlement()["removal_outcome"] == "absent"
    assert w.engine.deletes == ["prj-1"] * 2, "the reused id was never released"


def test_an_ordinary_modern_adoption_records_one_post_and_no_registry_or_history_scan(w, monkeypatch):
    policy._CREATED_REPORTED[(w.engine.engine_version, w.engine.engine_build_sha)] = True
    w.engine.preexisting = "user-project"
    monkeypatch.setattr(current, "registration_snapshot", lambda *a, **kw: pytest.fail("unrelated history read"))
    assert _start(w.root, "t-a")["status"] == "started"
    assert len(w.engine.posts) == 1 and w.engine.lookups == [] and len(w.engine.run_posts) == 1


def test_engine_confirmation_precedes_the_actor_lock_and_the_claim_holds_both(w, monkeypatch):
    from ouroboros.delegate_start_claims import claimed_start_request

    held, observations = [], []
    w.engine.projects["/shared"] = "p"

    def watch(module, name):
        real = getattr(module, name)

        @contextmanager
        def lock(*args, **kwargs):
            with real(*args, **kwargs):
                held.append(name)
                try:
                    yield
                finally:
                    held.pop()

        monkeypatch.setattr(module, name, lock)

    watch(usage, "project_retirement_lock")
    watch(custody, "actor_decision_lock")
    write = custody.record_start_requested

    def record(*args, **kwargs):
        observations.append(tuple(held))
        return write(*args, **kwargs)

    monkeypatch.setattr(custody, "record_start_requested", record)
    w.engine.on_lookup = lambda: observations.append(tuple(held))
    assert claimed_start_request(w.root, gateway=w.engine, task_id="t-a", run_id="", invocation_id="a",
        idempotency_key="a", route="r", project_id="p",
        request={"prompt": "review", "scope": {"kind": "project", "root": "/shared"}}) == (True, {})
    assert observations == [("project_retirement_lock",), ("project_retirement_lock", "actor_decision_lock")]


def test_a_pending_use_on_another_engine_does_not_hold_this_engines_registration(tmp_path):
    engine = Engine()
    engine.projects["/repo"] = "prj-1"
    assert custody.record_start_requested(  # the same id, pending on a work engine
        tmp_path, run_id="", task_id="t-w", invocation_id="inv-w", idempotency_key="inv-w", request={"prompt": "p"},
        project_id="prj-1", route="r", engine_ref=dict(WORK_REF))
    assert policy.retire_unstarted(tmp_path, engine, "prj-1", task_id="t-a", invocation_id="inv-a",
                                   resolution=binding(root="/repo")) == (True, "")
    assert engine.deletes == ["prj-1"] and _owed(tmp_path) == []


@pytest.mark.parametrize("role", ["creator", "adopter", "preparation-only", "unreadable-creator"])
def test_keep_on_a_registration_user_holds_release_until_withdrawn(w, monkeypatch, role):
    if role in ("preparation-only", "unreadable-creator"):
        w.engine.projects["/shared"] = "prj-1"
        assert policy.owe_registration(w.root, "prj-1", binding(root="/shared"), task_id="t-a")
    else:
        _creator_run_ended(w)
        if role == "adopter":
            b = _start(w.root, "t-b")
            _settle(w.root, b["run_id"], "t-b")
    kept = {"creator": "t-c", "adopter": "t-b"}.get(role, "t-a")
    if role == "unreadable-creator":
        from tests._store_shared import unreadable_result

        _finish(w.root, kept)
        unreadable = unreadable_result(monkeypatch, kept, root=w.root)
    else:
        seed_result(w.root, kept, {"status": "completed", "root_task_id": kept, "owner_keep": {"scope": "task"}})
    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert not w.engine.deletes and not _rows(w.root, custody.PROJECT_RETIRED)
    if role == "unreadable-creator":
        unreadable["on"] = False
    _finish(w.root, kept)
    for _ in range(2):
        custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == ["prj-1"] and len(_rows(w.root, custody.PROJECT_RETIRED)) == 1


def test_a_fresh_scope_root_is_resolved_on_this_host_only_for_the_owned_engine(monkeypatch):
    """The legacy list comparison resolves a fresh scope root as the owned engine would, through this host's
    symlinks only; a work engine's folder is foreign to this host and compares as given. No directory is listed."""
    import ntpath

    calls = []

    def resolve(path):
        calls.append(path)
        return r"C:\Registry\Workspace"

    monkeypatch.setattr(policy, "os", SimpleNamespace(path=SimpleNamespace(realpath=resolve, normcase=ntpath.normcase)))
    selected = r"C:\Selected\Workspace"
    assert policy.fresh_root_key(selected, {}) == policy.recorded_workspace_key(r"C:\Registry\Workspace")
    assert calls == [selected]
    assert policy.fresh_root_key(selected, {"engine_ref": WORK_REF}) == r"c:\selected\workspace"
    assert calls == [selected], "a work engine's folder is never resolved on this host"


@pytest.mark.parametrize("carrier", ["projected", "legacy-inline", "legacy-root", "legacy-ref"])
def test_workspace_use_holds_only_its_registration_through_preparation_start_and_settlement(w, carrier):
    from contextlib import ExitStack

    from ouroboros.delegate_custody_reconcile import recovered_custody
    from ouroboros.delegate_pending import request_shape
    from ouroboros.observability import write_blob
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result
    from tests._owner_continue_helpers import _interrupted

    paths = {name: str(w.root / name) for name in ("p", "q", "r")}
    w.engine.projects.update({folder: name for name, folder in paths.items()})
    for reg in ("q", "r"):
        assert policy.owe_registration(w.root, reg, binding(root=paths[reg]), task_id="t-a")
    request = {"prompt": "check fixture", "scope": {"root": paths["p"]},
               "execution": {"workspaceRoot": paths["q"], "isolation": "live"}}
    facts = dict(run_id="", task_id="t-b", project_id="p", project_owned=False,
                 idempotency_key="b", invocation_id="b", route="r", request=request)
    with ExitStack() as preparing:
        pid, owned, _, _, resolution = policy.resolve_for_start(w.root, w.engine, paths["p"], paths["q"],
            "workspace_write", task_id="t-b", invocation_id="b", preparation=preparing)
        assert (pid, owned) == ("p", "") and w.engine.posts == [paths["p"]]
        custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
        assert w.engine.deletes == ["r"] and _owed(w.root) == ["q"], "only Q is held before START"
        if carrier == "projected":
            assert custody.record_start_requested(w.root, gateway=w.engine, registration_resolution=resolution, **facts)
        else:
            legacy_facts = dict(facts)
            if carrier == "legacy-root":
                legacy_facts.update(execution_root=paths["q"], request={"scope": request["scope"]})
            elif carrier == "legacy-ref":
                del legacy_facts["request"]
                legacy_facts.update(request_ref=write_blob(w.root, request, kind="json"),
                                    request_shape=request_shape(request))
            assert custody.emit(w.root, custody.START_REQUESTED, legacy_facts)
    assert custody.record_start_requested(w.root, **{**facts, "task_id": "t-c", "invocation_id": "foreign",
        "idempotency_key": "foreign", "engine_ref": WORK_REF})

    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == ["r"] and _owed(w.root) == ["q"]

    record = custody.invocation_record(w.root, "b")
    entry, shape = recovered_custody(w.root, w.engine, {**record, "invocation_id": "b"}, "bound-b", daemon_store_id=STORE)
    if carrier == "legacy-ref":
        entry.execution_root = paths["q"]  # The legacy STARTED row also records its execution geometry.
    assert custody.bind_started(w.root, entry, shape=shape, exclusive=True).durable
    current.rebuild(w.root, replace=True)
    assert {row["invocation_id"] for row in custody.pending_invocations(w.root)} == {"foreign"}
    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == ["r"] and not current.run(w.root, "bound-b").project_owned
    _settle(w.root, "bound-b", "t-b")
    _interrupted(w.root, "t-b")
    assert continuation_offer(load_task_result(w.root, "t-b"), "t-b")["eligible"]
    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == ["r"], "B's Continue offer holds Q"
    seed_result(w.root, "t-b", {"status": "completed", "root_task_id": "t-b", "owner_keep": {"scope": "task"}})
    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == ["r"], "Keep holds Q even after B finishes"
    _finish(w.root, "t-b")
    for _ in range(2):
        custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == ["r", "q"] and not _owed(w.root)
    assert w.engine.projects == {paths["p"]: "p"}
    assert set(w.engine.asked) == {("r", paths["q"])}, "R was asked about Q exactly as sent; Q's own root needs no read"


# -- controls ---------------------------------------------------------------------------------------------------------


def test_a_registration_that_existed_before_the_start_is_never_removed_by_it(w):
    w.engine.preexisting = "prj-user"
    w.fenced["t-a"] = lambda: None
    a = _start(w.root, "t-a")
    assert a["definitely_unrun"] is True and a["project_retired"] is False
    assert w.engine.deletes == [] and _owed(w.root) == [], "an adopted registration owes this start nothing"


def test_a_persistent_sharer_keeps_the_registration_without_an_owed_duty(w):
    assert custody.record_start_requested(  # the user's own project, named by a start still pending
        w.root, run_id="", task_id="t-b", root_task_id="t-b", invocation_id="inv-b", idempotency_key="inv-b",
        request={"prompt": "p"}, project_id="prj-1", project_persistent=True, daemon_store_id=STORE, route="r")
    w.fenced["t-a"] = lambda: None
    a = _start(w.root, "t-a")
    assert (a["project_retired"], a["project_retention_reason"]) == (False, "persistent_registration")
    assert w.engine.deletes == [] and _owed(w.root) == []


def test_an_adopter_of_a_persistent_project_asks_the_engine_nothing_more(w):
    _creator_run_ended(w)
    [owner] = custody.owned_project_registrations(w.root)
    assert custody.record_started(w.root, custody.RunCustody(
        run_id="run-user", task_id="t-a", root_task_id="t-a", route_id="r", model="m", project_id=owner.project_id,
        project_persistent=True, ledger_root=str(w.root)))
    _settle(w.root, "run-user", "t-a")
    before = (list(w.engine.posts), list(w.engine.lookups))
    b = _start(w.root, "t-b")
    assert b["status"] == "started"
    assert (w.engine.posts, w.engine.lookups) == (before[0] + [w.engine.posts[-1]], before[1])


def test_the_review_twin_keeps_and_owes_a_registration_a_pending_start_names(tmp_path):
    from ouroboros import review_execution

    engine = Engine()
    engine.projects["/repo"] = "prj-1"
    write_task_result(tmp_path, "t-b", "running", root_task_id="t-b")
    assert custody.record_start_requested(
        tmp_path, run_id="", task_id="t-b", root_task_id="t-b", invocation_id="inv-b", idempotency_key="inv-b",
        request={"prompt": "p"}, project_id="prj-1", route="r")
    retired = review_execution._retire_orphaned_review_registration(
        custody, engine, tmp_path, "prj-1", definite_refusal=True, reason="owner_pause", invocation_id="inv-a",
        task_id="t-a", resolution=binding(root="/repo"))
    assert retired is False and engine.deletes == [] and _owed(tmp_path) == ["prj-1"]
    assert _rows(tmp_path, custody.START_FAILED, "t-a")[-1]["project_retention_reason"] == "retirement_owed"
    custody.emit(tmp_path, custody.START_FAILED, {"run_id": "", "task_id": "t-b", "invocation_id": "inv-b",
                                                  "project_id": "prj-1", "definite": True, "reason": "refused"})
    custody.retire_settled_registrations(tmp_path, engine, live_task_ids=set())
    assert engine.deletes == ["prj-1"] and _owed(tmp_path) == [], "its last use ended: retired once"


# -- the engine's proven keep is not an unconfirmed removal (admission) -------------------


@pytest.mark.parametrize("lane", ["custody_kept", "custody"])
def test_a_proven_keep_admits_a_new_consumer_an_unanswered_removal_refuses(w, lane):
    """``custody_kept`` is the engine's typed refusal, not an unanswered removal: the registration is
    alive and in use, so an ordinary consumer adopts it. A release whose answer never came still
    refuses the next start before anything is sent."""
    _creator_run_ended(w)
    current.mark_removal(w.root, "prj-1", policy.engine_facts(w.engine), lane)
    sent = list(w.engine.run_posts)  # the creator's own POST; a refused start must add nothing
    started = _start(w.root, "t-b")
    if lane == "custody_kept":
        assert started["status"] == "started" and started["run_id"]
        [request] = _rows(w.root, custody.START_REQUESTED, "t-b")
        assert (request["project_id"], request["project_owned"]) == ("prj-1", False), "B adopted it"
    else:
        assert (started["reason"], started["definitely_unrun"]) == ("registration_unconfirmed", True)
        assert _rows(w.root, custody.START_REQUESTED, "t-b") == [] and w.engine.run_posts == sent


def test_a_bound_thread_successor_keeps_its_predecessors_binding_without_creator_rights(w):
    from contextlib import ExitStack

    created = _start(w.root, "t-c")
    _settle(w.root, created["run_id"], "t-c")
    _finish(w.root, "t-c")
    source = custody.invocation_record(w.root, created["invocation_id"])
    binding = policy.continued_binding(source, "prj-1")
    assert (binding["registration_root"], binding["registration_store_id"]) == (str(w.root), STORE)
    assert "creator_invocation" not in binding and "registration_created" not in binding, "no creator right"
    assert policy.continued_binding(source, "prj-other") is None
    assert policy.continued_binding({"project_id": "prj-1"}, "prj-1") is None, "no recorded binding"
    posts, lookups = list(w.engine.posts), list(w.engine.lookups)
    with ExitStack() as preparing:
        project_id, owned, _, _, resolution = policy.resolve_for_start(
            w.root, w.engine, str(w.root) + "/", "", "readonly", task_id="t-b", invocation_id="succ",
            preparation=preparing, bound_project_id="prj-1", bound_resolution=binding)
        assert (project_id, owned) == ("prj-1", "")
        assert resolution["registration_root"] == str(w.root), "the proven address, kept"
        assert (w.engine.posts, w.engine.lookups) == (posts, lookups), "a bound scope asks the engine nothing"
        assert custody.record_start_requested(w.root, gateway=w.engine, registration_resolution=resolution,
            task_id="t-b", invocation_id="succ", idempotency_key="succ", run_id="", project_id="prj-1",
            project_owned=False, route="r",
            request={"prompt": "s", "scope": {"kind": "project", "root": str(w.root) + "/"}})
    [row] = _rows(w.root, custody.START_REQUESTED, "t-b")
    assert row["project_resolution"]["registration_root"] == str(w.root)
    assert "creator_invocation" not in row["project_resolution"], "the successor owes no retirement"
    current.mark_removal(w.root, "prj-1", policy.engine_facts(w.engine), "")
    with ExitStack() as preparing:  # a change re-verifies the recorded address through one addressed read
        policy.resolve_for_start(w.root, w.engine, str(w.root) + "/", "", "readonly", task_id="t-b",
                                 invocation_id="succ2", preparation=preparing, bound_project_id="prj-1",
                                 bound_resolution=binding)
    assert w.engine.lookups == lookups, "the recheck read the binding's own address, never the list"
    w.engine.projects.clear()
    with ExitStack() as preparing, pytest.raises(ClaudexorUnavailable) as refused:
        policy.resolve_for_start(w.root, w.engine, str(w.root) + "/", "", "readonly", task_id="t-b",
                                 invocation_id="succ3", preparation=preparing, bound_project_id="prj-1",
                                 bound_resolution=binding)
    assert refused.value.code == "registration_replaced", "a changed registration is not weakened"


def test_a_fresh_scope_root_resolves_before_the_engine_compares_it(w):
    """The request's raw spelling is user input, not the registry's address: a fresh path resolves at
    admission; only a recorded registration root compares as recorded."""
    _creator_run_ended(w)
    current.mark_removal(w.root, "prj-1", policy.engine_facts(w.engine), "")
    row = {"project_id": "prj-1", "invocation_id": "fresh",
           "request": {"scope": {"kind": "project", "root": str(w.root) + "/"}}}
    with policy.registration_held(w.root, w.engine, row, None) as refusal:
        assert refusal == {}
    assert w.engine.lookups[-1] == "list", "verified by the resolved spelling, not the raw one"


def test_a_registry_row_spelling_the_folder_in_another_case_still_starts(w, monkeypatch):
    """A Windows registry row spells the drive in its canonical case while the resolved address is
    normcased; the same folder still starts, and a genuinely different folder is still refused."""
    monkeypatch.setattr(policy, "os", SimpleNamespace(path=SimpleNamespace(
        realpath=lambda path: path, normcase=lambda path: path.lower())))
    w.engine.projects.update({r"C:\Work\Repo": "prj-1"})
    same = {"project_id": "prj-1", "invocation_id": "fresh",
            "request": {"scope": {"kind": "project", "root": r"c:\work\repo"}}}
    with policy.registration_held(w.root, w.engine, dict(same), None) as refusal:
        assert refusal == {}, "the same folder in another case still starts"
    assert w.engine.lookups[-1] == "list"
    other = {**same, "invocation_id": "fresh-other",
             "request": {"scope": {"kind": "project", "root": r"c:\other"}}}
    with policy.registration_held(w.root, w.engine, other, None) as refusal:
        assert refusal["reason"] == "registration_replaced", "a genuinely different folder is still refused"


def test_a_start_naming_the_registered_folder_through_a_symlink_alias_still_starts(w):
    """The request's alias spelling resolves to the folder the registry row names before the engine compares it."""
    real = w.root / "registered"
    real.mkdir()
    alias = w.root / "alias"
    alias.symlink_to(real, target_is_directory=True)
    w.engine.projects[str(real)] = "prj-1"
    row = {"project_id": "prj-1", "invocation_id": "fresh",
           "request": {"scope": {"kind": "project", "root": str(alias)}}}
    with policy.registration_held(w.root, w.engine, row, None) as refusal:
        assert refusal == {}, "the alias resolves to the registered folder"
    assert w.engine.lookups[-1] == "list" and str(alias) != str(real)


def test_confirm_registration_checks_under_the_project_lock_and_releases_it(w, monkeypatch):
    """The sticky start's confirmation runs ``registration_held``'s checks under the project lock and
    releases it before the caller's slow thread creation; only the per-call preparation outlives it."""
    w.engine.projects[str(w.root)] = "prj-1"
    held = []
    real = usage.project_retirement_lock

    @contextmanager
    def observed(root, project_id, **kw):
        held.append(project_id)
        try:
            with real(root, project_id, **kw):
                yield
        finally:
            held.append(f"released:{project_id}")

    monkeypatch.setattr(usage, "project_retirement_lock", observed)
    claim = {"project_id": "prj-1", "invocation_id": "c",
             "request": {"scope": {"kind": "project", "root": str(w.root)}}}
    policy.confirm_registration(w.root, w.engine, claim, None)
    assert held == ["prj-1", "released:prj-1"], "the lock is gone before the caller proceeds"
    current.mark_removal(w.root, "prj-1", policy.engine_facts(w.engine), "custody")
    with pytest.raises(ClaudexorUnavailable) as refused:
        policy.confirm_registration(w.root, w.engine, dict(claim), None)
    assert refused.value.code == "registration_unconfirmed"
    assert held[-1] == "released:prj-1", "a typed refusal also releases the lock"
