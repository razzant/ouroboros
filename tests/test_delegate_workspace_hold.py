"""A start of another project in a temporary registration's folder holds that registration's release, and only the
engine rules a folder out (``candidateMatch``). A stateful in-memory fixture engine stands for Claudexor
(``tests.test_delegate_shared_registration.Engine``). In the protocol tests its folder identity is an alias table set
by hand: it exercises the host's protocol, never proves that two spellings name one folder. The real-folder control
at the end lets the engine's domain decide from the folders themselves (two temporary directories, each reached
through other spellings); the host's own code never resolves a folder in either."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_registration_policy as policy
from ouroboros import delegate_workspace_hold as hold
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._bound_registration import STORE, binding
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401 -- autouse
from tests._store_shared import seed_result
from tests.test_delegate_shared_registration import Engine, _finish, _owed, _rows, _settle, w  # noqa: F401 -- fixture

X = "prj-x"  # a temporary registration another project's start created
OWN = "/fixture/own"  # project P's own folder
KEY = current.owed_key(X, {"daemon_store_id": STORE})  # X's owed duty, where every use it meets is kept


@pytest.fixture
def x(w):  # noqa: F811
    """X names ``<tmp>/Repo`` in the catalog; the engine also answers ``<tmp>/repo`` for that folder, while
    ``<tmp>/Other`` is another folder. A file in X's folder stands for the work it holds."""
    w.root_x, w.alias, w.other = str(w.root / "Repo"), str(w.root / "repo"), str(w.root / "Other")
    (w.root / "Repo").mkdir()
    (w.root / "Repo" / "work.txt").write_text("kept", encoding="utf-8")
    w.engine.projects[w.root_x] = X
    w.engine.aliases = {w.alias: w.root_x}
    assert policy.owe_registration(w.root, X, binding(root=w.root_x), task_id="t-a")
    return w


def _pending(env, task, folder, invocation):
    """A START the host recorded for P in ``folder`` whose answer never came: no STARTED row, no refusal."""
    project_id = env.engine.projects.setdefault(OWN, "prj-p")
    body = {"prompt": "work", "scope": {"kind": "project", "root": OWN},
            "execution": {"isolation": "live", "delegated": True, "workspaceRoot": folder}}
    assert custody.record_start_requested(env.root, gateway=env.engine, run_id="", task_id=task, root_task_id=task,
        invocation_id=invocation, idempotency_key=invocation, project_id=project_id, route="r", request=body)
    return body


def _bound(env, task, folder, invocation, run_id, **facts):
    """The same START answered later: its run is bound to the recorded request (STARTED copies it)."""
    from ouroboros.delegate_custody_reconcile import recovered_custody

    _pending(env, task, folder, invocation)
    record = custody.invocation_record(env.root, invocation)
    entry, shape = recovered_custody(env.root, env.engine, {**record, "invocation_id": invocation}, run_id,
                                     daemon_store_id=STORE)
    for key, value in facts.items():
        setattr(entry, key, value)
    assert custody.bind_started(env.root, entry, shape=shape, exclusive=True).durable


def _prepare(env, folder, invocation, preparing):
    return policy.resolve_for_start(env.root, env.engine, OWN, folder, "workspace_write", task_id="t-b",
                                    invocation_id=invocation, preparation=preparing)[0]


def _sweep(env, live=()):
    custody.retire_settled_registrations(env.root, env.engine, live_task_ids=set(live))


def test_a_start_through_another_spelling_holds_the_registration_until_its_last_use(x):
    """P's start in X's folder, spelled otherwise, holds X from its preparation through a START whose answer never
    came to its live run; after its last use X is released exactly once, its folder's files untouched, and the
    host never sent a START of its own. The engine was asked about the folder exactly as the start names it."""
    with ExitStack() as preparing:
        _prepare(x, x.alias, "b", preparing)
        _sweep(x)
        assert x.engine.deletes == [] and _owed(x.root) == [X], "the preparation holds X before any START"
        body = _pending(x, "t-b", x.alias, "b")  # the START row takes the hold over before the preparation ends
    _sweep(x)
    assert x.engine.deletes == [] and _owed(x.root) == [X], "an unknown START holds X"
    [request] = _rows(x.root, custody.START_REQUESTED, "t-b")
    assert request["registration_workspace"] == x.alias, "recorded exactly as sent, never respelled"
    from ouroboros.delegate_pending import request_body

    assert request_body(x.root, request)["execution"] == body["execution"], "nothing of the check rides the START"
    record = custody.invocation_record(x.root, "b")
    from ouroboros.delegate_custody_reconcile import recovered_custody

    entry, shape = recovered_custody(x.root, x.engine, {**record, "invocation_id": "b"}, "run-b", daemon_store_id=STORE)
    assert custody.bind_started(x.root, entry, shape=shape, exclusive=True).durable
    _sweep(x)
    assert x.engine.deletes == [], "the live run holds X"
    _settle(x.root, "run-b", "t-b")
    _finish(x.root, "t-b")
    for _ in range(2):
        _sweep(x)
    assert x.engine.deletes == [X] and not _owed(x.root) and x.engine.run_posts == []
    assert (x.root / "Repo" / "work.txt").read_text(encoding="utf-8") == "kept"
    assert set(x.engine.asked) == {(X, x.alias)}


def test_a_start_in_another_folder_holds_nothing(x):
    """P's start in another folder R (pending, its answer unknown) leaves X free. Its claim named X's duty without
    asking the engine (the duty keeps the use); the release asked and the engine said ``different``."""
    _pending(x, "t-b", x.other, "b")
    [request] = _rows(x.root, custody.START_REQUESTED, "t-b")
    assert request["registration_links"] == [KEY] and _uses(x) == [{"invocation_id": "b"}]
    _sweep(x)
    assert x.engine.deletes == [X] and not _owed(x.root) and x.engine.asked == [(X, x.other)]


def test_two_preparations_of_one_start_each_hold_and_release_only_their_own_mark(x):
    """Two preparations of one logical start (a retry racing its original) each hold their own mark, on the folder and
    on the registration: the first one ending releases only its own, the other still holds X, and only when both are
    over does X go."""
    facts = policy.engine_facts(x.engine)

    def marks():
        with hold.workspace_lock(x.root, facts):
            return [folder for folder, _start in hold._prepared(x.root, facts)]

    with ExitStack() as first, ExitStack() as second:
        own = _prepare(x, x.alias, "b", first)
        assert _prepare(x, x.alias, "b", second) == own
        assert marks() == [x.alias, x.alias]
        first.close()
        assert marks() == [x.alias], "the other preparation keeps its own mark"
        from ouroboros.delegate_custody_usage import project_preparing, project_retirement_lock

        with project_retirement_lock(x.root, own, engine=facts), hold.workspace_lock(x.root, facts):
            assert project_preparing(x.root, own, facts), "its mark on its registration is its own too"
        _sweep(x)
        assert x.engine.deletes == [] and _owed(x.root) == [X], "the remaining preparation still holds X"
    assert marks() == []
    _sweep(x)
    assert x.engine.deletes == [X] and not _owed(x.root)


@pytest.mark.parametrize("answer", ["unknown", "silent", "another_store", "lost"])
def test_an_answer_that_rules_nothing_out_keeps_the_registration(x, monkeypatch, answer):
    """Only ``different`` (or X absent) frees a folder. The engine's ``unknown``, no field from an engine that
    ignores ``candidate``, a 409 from another store and a read whose answer never came all keep X, even for a
    folder the engine would rule out; nothing is released and nothing is sent again."""
    _pending(x, "t-b", x.other, "b")
    real = x.engine.bound_registration

    def reading(project_id, store_id, root, *, release=False, candidate=""):
        if candidate and answer == "another_store":
            raise ClaudexorUnavailable("store_mismatch", "another base serves", status_code=409)
        if candidate and answer == "lost":
            raise ClaudexorUnavailable("daemon_unreachable", "the answer never came", status_code=0)
        reply = real(project_id, store_id, root, release=release, candidate=candidate)
        if candidate and answer == "unknown":
            reply["candidateMatch"] = "unknown"
        if candidate and answer == "silent":
            reply.pop("candidateMatch", None)
        return reply

    monkeypatch.setattr(x.engine, "bound_registration", reading)
    _sweep(x)
    assert x.engine.deletes == [] and _owed(x.root) == [X] and x.engine.run_posts == []


@pytest.mark.parametrize("effect", ["lost", "applied"])
def test_an_unanswered_release_holds_back_a_start_in_its_folder_until_answered(x, monkeypatch, effect):
    """X's release left and its answer was lost. A start in X's folder, however spelled, waits before anything is
    sent while the engine still has X; once X is absent the release can no longer land, and the folder serves.
    A start in another folder proceeds. The next pass sends the same bound release, under the same proofs."""
    real, plan = x.engine.release_registered, [effect]

    def release(project_id):
        step = plan.pop(0) if plan else "answer"
        if step == "lost":
            x.engine.deletes.append(project_id)
            raise TimeoutError("the answer was lost")
        state = real(project_id)
        if step == "applied":
            raise TimeoutError("the answer was lost after the effect")
        return state

    monkeypatch.setattr(x.engine, "release_registered", release)
    _sweep(x)
    facts = policy.engine_facts(x.engine)
    assert hold.folder_release(x.root, x.root_x, facts) == X, "the folder mark waits for the answer"
    posts = list(x.engine.posts)
    with ExitStack() as preparing:
        if effect == "lost":
            with pytest.raises(ClaudexorUnavailable) as refused:
                _prepare(x, x.alias, "b-1", preparing)
            assert refused.value.code == "registration_unconfirmed" and x.engine.posts == posts, "nothing was sent"
        else:
            _prepare(x, x.alias, "b-1", preparing)
    with ExitStack() as preparing:
        _prepare(x, x.other, "b-2", preparing)
    row = {"project_id": x.engine.projects[OWN], "invocation_id": "b-3",
           "request": {"scope": {"kind": "project", "root": OWN}, "execution": {"workspaceRoot": x.alias}}}
    with policy.registration_held(x.root, x.engine, row) as refusal:
        assert refusal.get("reason") == ("registration_unconfirmed" if effect == "lost" else None)
    _sweep(x)
    [retired] = _rows(x.root, custody.PROJECT_RETIRED)
    assert retired["removal_outcome"] == ("removed" if effect == "lost" else "gone_after_unknown")
    assert not hold.folder_release(x.root, x.root_x, facts) and not _owed(x.root) and x.engine.run_posts == []


@pytest.mark.parametrize("why", ["live_census", "continue_offer", "owner_keep"])
def test_a_settled_run_in_another_spelling_keeps_holding_while_it_may_go_on(x, why):
    """After settlement P's run in X's folder, spelled otherwise, still holds X while its task line may continue it
    (the task is in the live census, or the owner may Continue it) or the owner kept its task. It is found by a
    bounded read (the live census, or custody still holding its private copy's changes), and the engine relates its
    folder to X. The same run in another folder holds nothing."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result
    from tests._owner_continue_helpers import _interrupted

    live = {"t-b", "t-c"} if why == "live_census" else set()
    for folder, task, run_id in ((x.alias, "t-b", "run-b"), (x.other, "t-c", "run-c")):
        # In the census the closed run is found through its live task; otherwise custody still holds its copy.
        _bound(x, task, folder, f"inv-{task}", run_id, **({} if live else {"snapshot_id": f"snap-{task}"}))
        _settle(x.root, run_id, task)
    if why == "continue_offer":
        _interrupted(x.root, "t-b")
        assert continuation_offer(load_task_result(x.root, "t-b"), "t-b")["eligible"]
    if why == "owner_keep":
        seed_result(x.root, "t-b", {"status": "completed", "root_task_id": "t-b", "owner_keep": {"scope": "task"}})
    _sweep(x, live)
    assert x.engine.deletes == [] and _owed(x.root) == [X]
    assert (X, x.alias) in x.engine.asked, "the engine, not the spelling, related the settled run's folder to X"
    _finish(x.root, "t-b", "t-c")
    for _ in range(2):
        _sweep(x)
    assert x.engine.deletes == [X] and not _owed(x.root)


# -- a run that ended: found by the link its START or the registration's creation kept ------------------------------


def _ended(env, task, folder, run_id, end=_finish):
    """P's run in ``folder``: started, answered, settled; its task ended (``end``). Custody no longer holds the row
    (no private copy, no collection debt) and the task is in no live census: only the kept link can find it."""
    _bound(env, task, folder, f"inv-{task}", run_id)
    _settle(env.root, run_id, task)
    end(env.root, task)
    entry = current.run(env.root, run_id)
    assert entry.settled and not entry.snapshot_id and not current.collection_pending(entry)
    from ouroboros import store

    with store.open_store(env.root, write=False) as txn:
        assert txn.conn.execute("SELECT state FROM custody_runs WHERE run_id=?", (run_id,)).fetchone()[0] == "closed"


def _links(env, task):
    return [row["registration_links"] for row in _rows(env.root, custody.START_REQUESTED, task)]


def _uses(env, key=KEY):
    """The uses X's duty keeps: those its creation met, then each linked START's own row (None: none, as after X)."""
    from ouroboros import store

    with store.open_store(env.root, write=False) as txn:
        return current.linked_uses(txn, key) or None


@pytest.mark.parametrize("hold_by", ["late_keep", "technical_continue"])
def test_an_ended_run_in_another_spelling_holds_through_its_tasks_keep_or_continue(x, hold_by):
    """P's run in X's folder spelled otherwise ended long ago: its START linked it to X (the engine did not rule the
    folder out), and the link finds it again. A Keep its task gets later, or a Continue its task line may still take
    after a technical break, holds X; once that hold ends X goes exactly once. The same ended run in another folder R
    was never linked and holds nothing, even when its own task is kept."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result
    from tests._owner_continue_helpers import _interrupted

    _ended(x, "t-b", x.alias, "run-b", end=_finish if hold_by == "late_keep" else _interrupted)
    _ended(x, "t-c", x.other, "run-c")
    assert _links(x, "t-b") == _links(x, "t-c") == [[KEY]], "each claim named the duty alive on its engine"
    assert _uses(x) == [{"invocation_id": "inv-t-b"}, {"invocation_id": "inv-t-c"}], "kept in X's own duty"
    seed_result(x.root, "t-c", {"status": "completed", "root_task_id": "t-c", "owner_keep": {"scope": "task"}})
    if hold_by == "late_keep":
        seed_result(x.root, "t-b", {"status": "completed", "root_task_id": "t-b", "owner_keep": {"scope": "task"}})
    else:  # a technical break: the owner may still Continue the line, the author's own stop would add no hold
        assert continuation_offer(load_task_result(x.root, "t-b"), "t-b")["eligible"]
    for _ in range(2):
        _sweep(x)
    assert x.engine.deletes == [] and _owed(x.root) == [X], f"{hold_by} holds X through the ended run"
    _finish(x.root, "t-b")  # the Keep withdrawn, or the line done: R's kept task still holds nothing
    for _ in range(2):
        _sweep(x)
    assert x.engine.deletes == [X] and not _owed(x.root) and x.engine.run_posts == []
    assert (x.root / "Repo" / "work.txt").read_text(encoding="utf-8") == "kept"


def test_a_keep_landing_after_the_uses_are_read_still_holds_through_the_related_users_gate(x, monkeypatch):
    """An ended run in X's folder (spelled otherwise), its task neither kept nor able to go on when the release reads
    the folder's uses, is still a related user: its task's Keep gate is taken before the release, so a Keep that lands
    between that read and the release holds X. The same ended run in another folder is never gated."""
    _ended(x, "t-b", x.alias, "run-b")
    _ended(x, "t-c", x.other, "run-c")
    facts = policy.engine_facts(x.engine)
    with hold.workspace_lock(x.root, facts):
        assert hold.workspace_holds(x.root, x.engine, X, x.root_x, facts, live_task_ids=set()) == (False, {"t-b"})
    read = hold.workspace_holds

    def keep_lands(*args, **kwargs):
        found = read(*args, **kwargs)
        seed_result(x.root, "t-b", {"status": "completed", "root_task_id": "t-b", "owner_keep": {"scope": "task"}})
        return found

    with monkeypatch.context() as late:
        late.setattr(hold, "workspace_holds", keep_lands)
        _sweep(x)
    assert x.engine.deletes == [] and _owed(x.root) == [X], "the related user's gate read the late Keep"
    _finish(x.root, "t-b")  # the Keep withdrawn: nothing else holds X
    _sweep(x)
    assert x.engine.deletes == [X] and not _owed(x.root)


def test_a_registration_created_while_a_use_runs_keeps_that_use_in_its_duty(w):  # noqa: F811
    """P's run in a folder started before any temporary registration named it, so its START linked nothing. A start
    of another task then creates the temporary registration X of that folder: its owed duty keeps the live use the
    engine did not rule out, and after P's run ended a Keep on P's task still holds X until it is withdrawn."""
    root_x, alias = str(w.root / "Repo"), str(w.root / "repo")
    (w.root / "Repo").mkdir()
    w.engine.aliases = {alias: root_x}
    _bound(w, "t-b", alias, "inv-b", "run-b")
    assert _links(w, "t-b") == [[]], "no temporary registration was alive at its START"
    with ExitStack() as creating:
        project_id = policy.resolve_for_start(w.root, w.engine, root_x, "", "readonly", task_id="t-a",
                                              invocation_id="a", preparation=creating)[0]
    [owed] = current.owed_registrations(w.root)
    assert (owed["project_id"], owed["workspace_uses"]) == (project_id, [{"run_id": "run-b"}])
    _settle(w.root, "run-b", "t-b")
    seed_result(w.root, "t-b", {"status": "completed", "root_task_id": "t-b", "owner_keep": {"scope": "task"}})
    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == [] and _owed(w.root) == [project_id]
    _finish(w.root, "t-b")
    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == [project_id] and not _owed(w.root)


# -- one fence per side: what is observed is recorded before the lock goes; busy is never empty -------------------


@contextmanager
def _busy(*_args, **_kwargs):
    raise TimeoutError("the workspace lock is busy")
    yield  # pragma: no cover


def _unreadable(*_args, **_kwargs):
    raise OSError("the custody store cannot be read")


@pytest.mark.parametrize("broken", ["busy", "unreadable"])
def test_a_creation_that_cannot_observe_its_folder_s_uses_sends_nothing(w, monkeypatch, broken):  # noqa: F811
    """A start that may create a temporary registration observes the live uses before its request. Its fence busy, or
    its observation unreadable, it sends nothing that could create one: refused typed, no registration and no duty.
    A registration its lookup already found is adopted all the same; one created with the fence free, no use alive,
    goes as soon as nothing uses it."""
    root_x, root_y, root_z = str(w.root / "Repo"), str(w.root / "Other"), str(w.root / "Found")
    w.engine.projects[root_z] = "prj-found"
    with monkeypatch.context() as patch:
        patch.setattr(*((hold, "workspace_lock", _busy) if broken == "busy" else (hold, "_live_uses", _unreadable)))
        with ExitStack() as creating, pytest.raises(ClaudexorUnavailable) as refused:
            policy.resolve_for_start(w.root, w.engine, root_x, "", "readonly", task_id="t-a", invocation_id="a",
                                     preparation=creating)
        with ExitStack() as adopting:
            found = policy.resolve_for_start(w.root, w.engine, root_z, "", "readonly", task_id="t-b",
                                             invocation_id="b", preparation=adopting)
    assert refused.value.code == ("registration_busy" if broken == "busy" else "registration_unconfirmed")
    assert refused.value.registration_claim_refused is True, "typed before any request: a first start never ran"
    assert root_x not in w.engine.posts and root_x not in w.engine.projects and not _owed(w.root)
    assert found[:2] == ("prj-found", ""), "the registration the lookup found is adopted, never lost or owned"
    with ExitStack() as creating:
        free_id = policy.resolve_for_start(w.root, w.engine, root_y, "", "readonly", task_id="t-c",
                                           invocation_id="c", preparation=creating)[0]
    duties = {owed["project_id"]: owed.get("workspace_uses") for owed in current.owed_registrations(w.root)}
    assert duties == {free_id: None}, "observed before the request, under the hold that wrote the duty"
    custody.retire_settled_registrations(w.root, w.engine, live_task_ids=set())
    assert w.engine.deletes == [free_id] and not _owed(w.root)


def test_a_busy_fence_refuses_the_start_claim_and_records_nothing(x, monkeypatch):
    """A START claim whose fence is busy fails typed before its START_REQUESTED (nothing was sent: a first start
    definitely never ran): no row, no use added to X's duty, nothing sent and nothing retried."""
    monkeypatch.setattr(hold, "workspace_lock", _busy)
    with pytest.raises(ClaudexorUnavailable) as refused:
        _pending(x, "t-b", x.alias, "b")
    assert refused.value.code == "registration_busy" and refused.value.registration_claim_refused is True
    assert _rows(x.root, custody.START_REQUESTED, "t-b") == [] and _uses(x) is None and x.engine.run_posts == []


@pytest.mark.parametrize("replace", [True, False])
def test_a_restart_fold_rebuilds_the_uses_a_duty_keeps_and_they_end_with_it(x, replace):
    """The uses X's duty keeps come back exactly from the retained chain after a restart's fold, whole or merged,
    and still hold X through a late Keep; once X is released they are gone with its duty, and a fold names none."""
    _ended(x, "t-b", x.alias, "run-b")
    seed_result(x.root, "t-b", {"status": "completed", "root_task_id": "t-b", "owner_keep": {"scope": "task"}})
    kept = _uses(x)
    assert kept == [{"invocation_id": "inv-t-b"}]
    custody._CUSTODY.clear()
    current.rebuild(x.root, replace=replace)
    assert _uses(x) == kept
    _sweep(x)
    assert x.engine.deletes == [], "the rebuilt use holds X through the Keep"
    _finish(x.root, "t-b")
    _sweep(x)
    current.rebuild(x.root, replace=replace)
    assert x.engine.deletes == [X] and _uses(x) is None and not _owed(x.root)


def test_a_review_continuation_without_a_registration_of_its_own_is_kept_by_x(x):
    """A continued review starts in its already admitted folder with no registration of its own (its claim carries
    no gateway). Its claim still names X's duty under the fence, so its run, ended, holds X through a late Keep;
    nothing registers a folder for it."""
    posts = list(x.engine.posts)
    body = {"prompt": "continue", "scope": {"kind": "project", "root": "/fixture/review", "ephemeral": True},
            "execution": {"isolation": "live", "delegated": True, "workspaceRoot": x.alias}}
    assert custody.record_start_requested(x.root, gateway=None, run_id="", task_id="t-r", root_task_id="t-r",
        invocation_id="rev-1", idempotency_key="rev-1", project_id="", route="r", request=body)
    assert _links(x, "t-r") == [[KEY]] and _uses(x) == [{"invocation_id": "rev-1"}] and x.engine.posts == posts
    from ouroboros.delegate_custody_reconcile import recovered_custody

    record = custody.invocation_record(x.root, "rev-1")
    entry, shape = recovered_custody(x.root, x.engine, {**record, "invocation_id": "rev-1"}, "run-r", daemon_store_id=STORE)
    assert custody.bind_started(x.root, entry, shape=shape, exclusive=True).durable
    _settle(x.root, "run-r", "t-r")
    seed_result(x.root, "t-r", {"status": "completed", "root_task_id": "t-r", "owner_keep": {"scope": "task"}})
    _sweep(x)
    assert x.engine.deletes == []
    _finish(x.root, "t-r")
    _sweep(x)
    assert x.engine.deletes == [X] and x.engine.posts == posts


# -- the real-folder control -------------------------------------------------------------------------------------


class _FolderEngine(Engine):
    """The engine's domain decided by the folders themselves: a spelling names the directory ``os.stat`` reaches
    through it (device, inode), so another spelling of one folder is that folder and two folders stay two."""

    def folder(self, path):
        try:
            found = os.stat(path)
        except OSError:
            return path
        return found.st_dev, found.st_ino


@pytest.mark.parametrize("spelling", ["symlink", "parent_step"])
def test_two_real_folders_are_told_apart_by_the_engine_not_by_the_spelling(w, spelling):  # noqa: F811
    """Two temporary directories: A, the folder X names, and B. Each is also reached through another spelling, a
    directory symlink or a ``..`` step through the other folder. The engine's domain compares what each spelling
    reaches; nothing names the answer in advance. P's ended run through A's other spelling holds X by its task's
    late Keep until it is withdrawn; the run through B's holds nothing though its task is kept too.

    OS boundaries: a directory symlink needs symlink rights (Windows without them: that case is skipped); the
    ``..`` spelling needs none. Device and inode identify a directory on POSIX filesystems and NTFS; a FAT or network
    volume may not give stable inodes. Case aliases exist only on case-insensitive volumes and are not used here."""
    base = w.root / "neutral"
    a, b = base / "A", base / "B"
    for folder in (a, b):
        folder.mkdir(parents=True)
    (a / "work.txt").write_text("kept", encoding="utf-8")
    if spelling == "symlink":
        try:
            (base / "qa").symlink_to(a, target_is_directory=True)
            (base / "rb").symlink_to(b, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"no directory symlink here: {exc}")
        q, r = str(base / "qa"), str(base / "rb")
    else:
        q, r = str(b / ".." / "A"), str(a / ".." / "B")
    assert q != str(a) and r != str(b), "neither run names its folder the way the catalog does"
    engine = w.use(_FolderEngine())
    engine.projects[str(a)] = X
    assert policy.owe_registration(w.root, X, binding(root=str(a)), task_id="t-a")
    _ended(w, "t-b", q, "run-b")
    _ended(w, "t-c", r, "run-c")
    assert _links(w, "t-b") == _links(w, "t-c") == [[KEY]], "the engine, at the release, tells the two apart"
    for task in ("t-b", "t-c"):
        seed_result(w.root, task, {"status": "completed", "root_task_id": task, "owner_keep": {"scope": "task"}})
    custody.retire_settled_registrations(w.root, engine, live_task_ids=set())
    assert engine.deletes == [] and _owed(w.root) == [X]
    _finish(w.root, "t-b")
    custody.retire_settled_registrations(w.root, engine, live_task_ids=set())
    assert engine.deletes == [X] and (a / "work.txt").read_text(encoding="utf-8") == "kept"
    assert {(X, q), (X, r)} <= set(engine.asked), "each run's folder exactly as sent"
