"""Recovery after a lost answer to a registration release: the same bound release goes again after the same proofs,
and the registration and its folder stay held until a definite answer. A stateful in-memory fixture engine stands
for Claudexor (``tests.test_delegate_shared_registration.Engine``); folders are neutral temporary paths."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_registration_policy as policy
from ouroboros import delegate_workspace_hold as hold
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._bound_registration import binding
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401 -- autouse
from tests.test_delegate_shared_registration import WAIT, _owed, _rows, _start, w  # noqa: F401 -- shared fixture


def _answers(env, monkeypatch, *plan):
    """The engine's next releases follow ``plan``: "lost" (nothing done, answer lost), "applied" (removed, answer
    lost) or a refusal to raise; later ones answer normally. Returns the ids the engine really removed."""
    real, plan, removed = env.engine.release_registered, list(plan), []

    def release(project_id):
        step = plan.pop(0) if plan else "answer"
        if step == "lost" or isinstance(step, ClaudexorUnavailable):
            env.engine.deletes.append(project_id)
            raise TimeoutError("the answer was lost") if step == "lost" else step
        state = real(project_id)
        removed.extend([project_id] if state == "released" else [])
        if step == "applied":
            raise TimeoutError("the answer was lost after the effect")
        return state

    monkeypatch.setattr(env.engine, "release_registered", release)
    return removed


def _held(env, project_id="prj-1", folder=None):
    """(the registration's own removal mark, the registration whose unanswered release holds the folder)."""
    facts = policy.engine_facts(env.engine)
    return current.removal_status(env.root, project_id, facts)[0], hold.folder_release(env.root, folder or str(env.root), facts)


def _a_creator_loses_its_release_answer(env):
    """A created prj-1 and definitely never ran, so it releases it at once; that answer never comes."""
    env.fenced["t-a"] = lambda: None
    a = _start(env.root, "t-a")
    assert (a["project_retired"], a["project_retention_reason"]) == (False, "retirement_owed")
    assert _owed(env.root) == ["prj-1"] and _held(env) == ("custody", "prj-1")


def _sweep(env, live=()):
    custody.retire_settled_registrations(env.root, env.engine, live_task_ids=set(live))


def test_a_lost_release_that_removed_nothing_is_repeated_once_and_never_reaches_a_later_registration(w, monkeypatch):  # noqa: F811
    removed = _answers(w, monkeypatch, "lost")
    _a_creator_loses_its_release_answer(w)
    b = _start(w.root, "t-b")  # the folder's registration is still prj-1: adopting it waits for the answer
    assert (b["reason"], b["definitely_unrun"]) == ("registration_unconfirmed", True) and not w.engine.run_posts
    _sweep(w)
    [retired] = _rows(w.root, custody.PROJECT_RETIRED)
    assert retired["removal_outcome"] == "removed", "the repeat itself archived it"
    assert removed == ["prj-1"] and w.engine.deletes == ["prj-1"] * 2 and _held(w) == ("", "")
    assert not _owed(w.root) and _start(w.root, "t-b")["status"] == "started"
    assert w.engine.projects == {str(w.root): "prj-2"}, "the folder registers again under a fresh id"
    late = w.engine.bound_registration("prj-1", w.engine.store_identity, str(w.root), release=True)
    assert late["state"] == "absent" and w.engine.projects == {str(w.root): "prj-2"}, "a late first request"


def test_a_release_applied_before_its_answer_was_lost_repeats_as_absent_without_a_second_removal(w, monkeypatch):  # noqa: F811
    removed = _answers(w, monkeypatch, "applied")
    _a_creator_loses_its_release_answer(w)
    assert removed == ["prj-1"] and not w.engine.projects
    assert _start(w.root, "t-b")["status"] == "started", "a new registration of the folder is another id"
    _sweep(w, live={"t-b"})
    [retired] = _rows(w.root, custody.PROJECT_RETIRED)
    assert (retired["project_id"], retired["removal_outcome"]) == ("prj-1", "gone_after_unknown")
    assert removed == ["prj-1"] and w.engine.deletes == ["prj-1"] * 2 and _held(w) == ("", "")
    assert w.engine.projects == {str(w.root): "prj-2"} and _owed(w.root) == ["prj-2"], "only B's live one stays owed"


def test_a_repeat_refused_while_the_engine_works_keeps_the_mark_and_later_releases_once(w, monkeypatch):  # noqa: F811
    working = ClaudexorUnavailable("project_has_active_run", "a run works there", status_code=409)
    removed = _answers(w, monkeypatch, "lost", working)
    _a_creator_loses_its_release_answer(w)
    _sweep(w)
    assert removed == [] and _held(w) == ("custody", "prj-1") and _owed(w.root) == ["prj-1"]
    assert _rows(w.root, custody.PROJECT_RETIRE_FAILED)[-1]["code"] == "project_has_active_run"
    assert _start(w.root, "t-b")["reason"] == "registration_unconfirmed" and not w.engine.run_posts
    _sweep(w)
    [retired] = _rows(w.root, custody.PROJECT_RETIRED)
    assert retired["removal_outcome"] == "removed" and removed == ["prj-1"] and w.engine.deletes == ["prj-1"] * 3
    assert _held(w) == ("", "") and not _owed(w.root)


def test_a_crash_between_the_mark_and_the_request_sends_the_release_once_after_restart(w, monkeypatch):  # noqa: F811
    def dies_before_sending(*_args, release=False):
        raise RuntimeError("the host died after the mark, before the request left")

    with monkeypatch.context() as crash:
        crash.setattr(w.engine, "bound_registration", dies_before_sending)
        _a_creator_loses_its_release_answer(w)
    assert w.engine.deletes == [] and w.engine.projects, "marked, nothing sent"
    custody._CUSTODY.clear()
    current.rebuild(w.root, replace=True)
    _sweep(w)
    [retired] = _rows(w.root, custody.PROJECT_RETIRED)
    assert retired["removal_outcome"] == "removed" and w.engine.deletes == ["prj-1"] and _held(w) == ("", "")


def test_two_concurrent_passes_over_one_unanswered_release_remove_it_once(w, monkeypatch):  # noqa: F811
    removed = _answers(w, monkeypatch, "lost")
    _a_creator_loses_its_release_answer(w)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(_sweep, w) for _ in range(2)]:
            future.result(timeout=WAIT)
    assert removed == ["prj-1"] and w.engine.deletes == ["prj-1"] * 2, "the project lock admits one repeat"
    assert len(_rows(w.root, custody.PROJECT_RETIRED)) == 1 and not _owed(w.root) and _held(w) == ("", "")


def test_an_unanswered_release_fences_its_folder_from_another_projects_workspace_until_answered(w, monkeypatch):  # noqa: F811
    paths = {name: str(w.root / name) for name in ("p", "q", "v")}
    w.engine.projects[paths["q"]] = "q"
    assert policy.owe_registration(w.root, "q", binding(root=paths["q"]), task_id="t-a")
    removed = _answers(w, monkeypatch, "lost")
    _sweep(w)
    assert removed == [] and _held(w, "q", paths["q"]) == ("custody", "q")

    def resolve(workspace, invocation_id):  # P registers its own folder and works in WORKSPACE
        with ExitStack() as preparing:
            return policy.resolve_for_start(w.root, w.engine, paths["p"], workspace, "workspace_write",
                                            task_id="t-b", invocation_id=invocation_id, preparation=preparing)[0]

    with pytest.raises(ClaudexorUnavailable) as refused:
        resolve(paths["q"], "b-1")
    assert refused.value.code == "registration_unconfirmed" and w.engine.posts == [], "P sent nothing"
    assert resolve(paths["v"], "b-2") == "prj-1", "an independent folder still serves P"
    row = {"project_id": "prj-1", "invocation_id": "b-3", "request": {
        "scope": {"kind": "project", "root": paths["p"]}, "execution": {"workspaceRoot": paths["q"]}}}
    with policy.registration_held(w.root, w.engine, row) as refusal:
        assert refusal["reason"] == "registration_unconfirmed", "a recorded start naming Q's folder waits too"
    _sweep(w)  # the same release, answered this time
    assert removed == ["q"] and _held(w, "q", paths["q"]) == ("", "")
    assert resolve(paths["q"], "b-4") == "prj-1", "answered: the folder serves P again"


def test_a_later_registration_of_the_folder_waits_for_the_earlier_ones_unanswered_release(w, monkeypatch):  # noqa: F811
    folder, facts = str(w.root / "q"), policy.engine_facts(w.engine)
    w.engine.projects[folder] = "q0"
    removed = _answers(w, monkeypatch, "applied")
    with pytest.raises(TimeoutError):
        policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=folder)
    w.engine.projects[folder] = "q1"  # the folder registered again, under a new id
    with pytest.raises(ClaudexorUnavailable) as held:
        policy.remove_registration(w.root, w.engine, "q1", facts, registration_root=folder)
    assert held.value.code == "registration_unconfirmed" and w.engine.deletes == ["q0"], "nothing sent for q1"
    assert policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=folder)[0] == "gone_after_unknown"
    assert policy.remove_registration(w.root, w.engine, "q1", facts, registration_root=folder)[0] == "removed"
    assert removed == ["q0", "q1"] and hold.folder_release(w.root, folder, facts) == ""


def test_custody_repeats_a_lost_historical_release_only_at_the_folder_it_left_for(w, monkeypatch):  # noqa: F811
    """A bound release the walk sent leaves its folder marked beside the ``historical`` mark: custody may send that same
    release again, and only for that folder, the original address."""
    folder, facts = str(w.root / "q"), policy.engine_facts(w.engine)
    w.engine.projects[folder] = "q0"
    removed = _answers(w, monkeypatch, "lost")
    with pytest.raises(TimeoutError):
        policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=folder, lane="historical")
    assert _held(w, "q0", folder) == ("historical", "q0") and w.engine.deletes == ["q0"]
    with pytest.raises(ClaudexorUnavailable) as moved:  # another folder: no bound release of it ever left
        policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=str(w.root / "elsewhere"))
    assert moved.value.code == "registration_unconfirmed" and w.engine.deletes == ["q0"]
    assert policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=folder)[0] == "removed"
    assert removed == ["q0"] and w.engine.deletes == ["q0", "q0"] and _held(w, "q0", folder) == ("historical", "")


@pytest.mark.parametrize("lane", ["historical", "historical_walk"])
def test_custody_only_holds_for_an_old_row_no_bound_release_left_under(w, lane):  # noqa: F811
    """A bare ``historical`` mark (an earlier walk's carry whose saved intent is gone) and a ``historical_walk`` one (an
    older id-only DELETE) name no bound release of this store, folder and id: custody holds and sends nothing."""
    folder, facts = str(w.root / "q"), policy.engine_facts(w.engine)
    w.engine.projects[folder] = "q0"
    current.mark_removal(w.root, "q0", facts, lane)
    with pytest.raises(ClaudexorUnavailable) as held:
        policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=folder)
    assert held.value.code == "registration_unconfirmed" and not w.engine.deletes
    assert _held(w, "q0", folder) == (lane, "") and w.engine.projects[folder] == "q0"


def test_a_proof_that_cannot_be_read_is_no_reason_to_send(w, monkeypatch):  # noqa: F811
    folder, facts = str(w.root / "q"), policy.engine_facts(w.engine)
    w.engine.projects[folder] = "q0"
    removed = _answers(w, monkeypatch, "lost")
    with pytest.raises(TimeoutError):
        policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=folder, lane="historical")

    def unreadable(*_args):
        raise OSError("the removal marks cannot be read")

    with monkeypatch.context() as broken:
        broken.setattr(hold, "folder_release", unreadable)
        with pytest.raises(OSError):
            policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=folder)
    assert w.engine.deletes == ["q0"] and removed == [], "the failure delayed the repeat and sent nothing"
    assert policy.remove_registration(w.root, w.engine, "q0", facts, registration_root=folder)[0] == "removed"
    assert removed == ["q0"]
