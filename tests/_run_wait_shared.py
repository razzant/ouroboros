"""Shared fixtures of the suites that drive ``delegate_wait`` as a sleep on runs.

The body records a run's terminal and questions in custody (``delegate_collection.observe_run``); a wake then
makes ONE addressed read of the run it speaks for (``tools.delegate._read_run``). Suites that pin the
delivery and acknowledgement of that wake script only that read: ``scripted_reads`` keeps the retired tick's
observer call shape ``(ctx, run_id, wait_sec, since_seq, **kw)``, so an observer written for the tick answers
the one read a wake makes, and ``held_run`` gives the run the custody facts the wake's readiness reads.
"""
from __future__ import annotations


def held_run(root, run_id, task_id, *, ended="", asks="", **facts):
    """A run ``task_id`` started (its STARTED custody row), optionally ended (``ended``: the terminal state the
    body observed) or asking (``asks``: the interaction id of its question)."""
    from ouroboros import delegate_custody as custody, delegate_custody_current as current
    from ouroboros.delegate_collection import observe_run

    if current.run(root, run_id) is None:
        assert custody.emit(root, custody.STARTED, {
            "run_id": run_id, "invocation_id": f"invoke-{run_id}", "task_id": task_id, "root_task_id": task_id,
            "route": "route", "source": "delegate_start", "access": "readonly", **facts})
    entry = current.run(root, run_id)
    if asks and asks not in entry.pending_interaction_ids:
        assert observe_run(root, entry, {"summary": {"runId": run_id, "state": "running"}, "lastSeq": 2,
                                         "pendingInteractions": [{"interactionId": asks, "questions": [
                                             {"id": "choice", "question": "Which path?"}]}]})
    if ended and not current.run(root, run_id).terminal_state:
        assert observe_run(root, current.run(root, run_id), {"summary": {"runId": run_id, "state": ended},
                                                             "lastSeq": 9})
    return current.run(root, run_id)


def scripted_reads(monkeypatch, observe):
    """Script the one addressed read a wake makes with a tick-shaped observer."""
    from ouroboros import delegate_supervision as ds
    from ouroboros.tools import delegate

    def read(ctx, run_id, since_seq, gateway=None):
        return observe(ctx, run_id, ds._TICK_SEC, since_seq, **({"gateway": gateway} if gateway is not None else {}))

    monkeypatch.setattr(delegate, "_read_run", read)
    return read


def publish(ctx, run_id, *, woke_by=""):
    """The wake a ready ``delegate_wait`` publishes now (its replay first), as the tool and the owed delivery do."""
    from ouroboros import delegate_supervision as ds
    from ouroboros.delegate_recovery import restore_bound_handoff

    runs = list(run_id) if isinstance(run_id, list) else [str(run_id)]
    for selected in runs:
        restore_bound_handoff(ctx, selected)
    return ds._publish(ctx, {"tool": "delegate_wait", "runs": runs}, woke_by)


def wake_now(ctx, run_id, wait_once=None, *, woke_by="", since_seq=None, checkpoint_reason=""):
    """What the retired tick's first meaningful beat returned, on today's path: the wake a READY ``delegate_wait``
    publishes now (its unacknowledged replay first), with only the one addressed read scripted by ``wait_once``
    (the tick's observer shape; ``None`` replays only). ``woke_by`` is what made it ready (``timeout`` with a
    ``checkpoint_reason``: the inspection time)."""
    from ouroboros import delegate_supervision as ds
    from ouroboros.tools import delegate

    import time

    from ouroboros.utils import utc_now_iso

    runs = list(run_id) if isinstance(run_id, list) else [str(run_id)]
    descriptor = {"tool": "delegate_wait", "runs": runs, "armed_at": utc_now_iso(), "armed_epoch": time.time(),
                  **({"since_seq": since_seq} if since_seq is not None else {}),
                  **({"checkpoint": {"reason": checkpoint_reason}} if checkpoint_reason else {})}
    saved = delegate._read_run

    def read(ctx_, rid, seq, gateway=None):
        if wait_once is None:
            raise AssertionError("an unacknowledged wake replays before the run is read")
        return wait_once(ctx_, rid, ds._TICK_SEC, seq)

    delegate._read_run = read
    try:
        from ouroboros.delegate_recovery import restore_bound_handoff

        for selected in runs:
            restore_bound_handoff(ctx, selected)
        return ds._publish(ctx, descriptor, woke_by)
    finally:
        delegate._read_run = saved
