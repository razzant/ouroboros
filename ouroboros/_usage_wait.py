"""Caller-owned pre-send accounting waits; no provider retry or task policy below.

Only positive lock contention can keep the same stack alive. Controls come
from the existing model-operation owner. Cleanup and post-response accounting
retain bounded acquisitions. The async bridge joins its mutator on cancellation
so a returned reservation cannot race a successor or lose its local owner.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import pathlib
import sys
import threading
import time
import uuid

from ouroboros.usage_ledger import UsageLockUnavailable, USAGE_LOCK_TIMEOUT_SEC

log = logging.getLogger(__name__)
_CANCEL = contextvars.ContextVar("usage_presend_cancel", default=None)
_CLEANUP = contextvars.ContextVar("usage_cleanup", default=False)
# Acquisition slicing controls responsiveness, not lifetime or quota policy.
ACQUISITION_SLICE_SEC = 0.25


def _check() -> None:
    cancel = _CANCEL.get()
    if cancel is not None and cancel.is_set():
        raise asyncio.CancelledError()
    from ouroboros.llm_attempt import require_physical_dispatch_window

    require_physical_dispatch_window()


def _hold(owner, phase: str, started: float, episode_id: str) -> None:
    if owner is None:
        return
    try:
        from ouroboros.loop_messages import _emit_checkpoint_event

        detail = "Waiting for accounting access" if phase == "entered" else "Accounting wait ended"
        _emit_checkpoint_event(owner.event_queue, owner.task_id, pathlib.Path(owner.drive_root) / "logs", {
            "checkpoint_kind": "usage_lock_wait", "owner_visible": True,
            "phase": phase, "episode_id": episode_id, "elapsed_sec": time.monotonic() - started,
            "detail": detail, "content": detail, "text": detail, "role": "system", "system_type": "task_checkpoint",
            **({"chat_id": owner.task["chat_id"]} if "chat_id" in owner.task else {}),
        })
    except Exception:
        log.debug("Could not disclose accounting wait", exc_info=True)


def send_acquisition(check=None):
    """Build one acquisition episode; critical-section exceptions never retry."""
    from ouroboros.model_wait import current_model_wait, dispatch_deadline_remaining_sec
    from ouroboros.config import get_task_idle_timeout_sec

    owner = current_model_wait()
    owned = owner is not None or dispatch_deadline_remaining_sec() is not None
    sliced = owned or _CANCEL.get() is not None
    started = time.monotonic()
    interactive_bound = (float(get_task_idle_timeout_sec()) if owner is not None
                         and owner.task.get("_is_direct_chat") else None)

    @contextlib.contextmanager
    def acquire(root):
        from ouroboros import usage_accounting as ua
        if not sliced or _CLEANUP.get():
            with ua._locked(root) as heartbeat:
                yield heartbeat
            return
        entered = False
        episode_id = uuid.uuid4().hex
        try:
            while True:
                _check()
                if check:
                    check()
                if interactive_bound is not None and time.monotonic() - started >= interactive_bound:
                    from ouroboros.llm_attempt import PhysicalDispatchInterrupted

                    raise PhysicalDispatchInterrupted("accounting_wait_expired")
                stack = contextlib.ExitStack()
                try:
                    heartbeat = stack.enter_context(ua._locked(root, timeout_sec=ACQUISITION_SLICE_SEC))
                except UsageLockUnavailable as exc:
                    if exc.reason != "contention" or (not owned and time.monotonic() - started >= USAGE_LOCK_TIMEOUT_SEC):
                        raise
                    if not entered:
                        entered = True
                        _hold(owner, "entered", started, episode_id)
                    continue
                with stack:
                    _check()
                    # The reserve/dispatch body checks its authoritative scope's
                    # fence once, after preparation and under this same hold.
                    yield heartbeat
                    return
        finally:
            if entered:
                _hold(owner, "ended", started, episode_id)
    return acquire


@contextlib.contextmanager
def bounded_cleanup():
    token = _CLEANUP.set(True)
    try:
        yield
    finally:
        _CLEANUP.reset(token)


def transition_acquisition(state, check=None):
    return cleanup_acquisition if _CLEANUP.get() else send_acquisition(check) if state == "dispatched" else None


def cleanup_acquisition(root):
    from ouroboros import usage_accounting as ua

    return ua._locked(root, timeout_sec=ACQUISITION_SLICE_SEC)


async def presend_off_loop(function, *args, on_cancel=None, **kwargs):
    """Copy ContextVars; cancel cooperatively and JOIN before unwinding custody."""
    from ouroboros import usage_accounting as ua

    def invoke():
        ua.adopt_physical_attempt_capture(None)
        result = function(*args, **kwargs)
        return result, ua.last_physical_attempt_capture()

    cancelled = threading.Event()
    token = _CANCEL.set(cancelled)
    try:
        future = asyncio.create_task(asyncio.to_thread(invoke))
    finally:
        _CANCEL.reset(token)
    try:
        result, capture = await asyncio.shield(future)
        if capture is not None:
            ua.adopt_physical_attempt_capture(capture)
        return result
    except asyncio.CancelledError:
        cancelled.set()
        # A second cancellation cannot abandon a mutator already holding money.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not future.cancelled() and future.exception() is None and on_cancel:
            # A reservation may have committed just as cancellation arrived.
            # Cleanup itself is bounded and joined under the same rule.
            cleanup = asyncio.create_task(asyncio.to_thread(on_cancel, future.result()[0]))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            failure = cleanup.result()
            capture = getattr(failure, "physical_attempt_capture", None)
            if capture is not None:
                ua.adopt_physical_attempt_capture(capture)
        raise


async def postresponse_off_loop(function, *args, retain_on_cancel):
    """Join accounting, retain the received answer, then propagate cancellation.

    The response already exists: cancellation cannot cancel accounting or
    skip its unresolved fallback. It still forbids caller continuation. The
    existing private response store owns retention, not the cancelled stack.
    """
    from ouroboros import usage_accounting as ua

    def invoke():
        result = function(*args)
        return result, ua.last_physical_attempt_capture()

    future = asyncio.create_task(asyncio.to_thread(invoke))
    cancelled = None
    while True:
        try:
            response, capture = await asyncio.shield(future)
            break
        except asyncio.CancelledError as exc:
            if future.cancelled():
                raise  # Not a caller cancellation; no outcome to invent.
            cancelled = exc
    ua.adopt_physical_attempt_capture(capture)
    if cancelled is None:
        return response
    cancelled.physical_attempt_capture = capture
    cancelled.response = response
    retention = asyncio.create_task(asyncio.to_thread(retain_on_cancel, response, capture))
    while not retention.done():
        try:
            await asyncio.shield(retention)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    try:
        cancelled.response_manifest_ref = retention.result()["manifest_ref"]
    except Exception as exc:
        # Keep the exact object on the exception too; do not turn a failed
        # retention into success or authorize execution after Stop.
        cancelled.response_retention_error = type(exc).__name__
        log.exception("Failed to retain cancelled paid response")
    raise cancelled


# The main round's own answer wait (``receiver_abandonable``) re-reads its
# owner Pause fence at this interval: responsiveness, not a lifetime or bound.
ABANDON_POLL_SEC = 0.5
_ABANDONABLE = contextvars.ContextVar("model_receiver_abandonable", default=False)
_ABANDONED_SENDS: set[asyncio.Task] = set()


@contextlib.contextmanager
def receiver_abandonable():
    """Mark an author model wait as one an owner Pause interrupts.

    Owner 2026-10-08 (quiz 9312a119, option 1): Pause interrupts the LOCAL wait
    for an already-sent answer and keeps the previous ready point. The loop's
    round, fallback, forced-final and compaction calls opt in; an unmarked
    tool or review-episode call keeps waiting as before.
    """
    token = _ABANDONABLE.set(True)
    try:
        yield
    finally:
        _ABANDONABLE.reset(token)


def _receiver_paused(reservation) -> bool:
    """The send's own task is under a closed owner fence it is not selected out of.

    Unreadable authority keeps waiting (the pre-existing behaviour); it never
    abandons a paid answer on a guess.
    """
    from ouroboros.owner_pause import _member_coordinates, fence_closed, read_fence

    root_drive, root_task_id, task_id = _member_coordinates(reservation.scope)
    if not root_drive or not root_task_id:
        return False
    try:
        fence = read_fence(root_drive, root_task_id)
    except Exception:
        return False
    return bool(fence_closed(fence) and task_id not in (fence.get("selected_members") or {}))


def model_send(reservation, send, *, settle_late=None):
    """One physical sender; its receiver may leave only through the handover below.

    The sender and the receiver race on ONE handover: either the receiver takes
    the answer (or error) and the caller settles it as before, or the receiver
    abandons first (owner Pause) and the SENDER settles whatever later arrives
    through ``settle_late`` — one accounting settlement and one retention
    disposition either way. Abandonment never claims the request unsent or
    cancelled remotely; the executor is not joined, so the caller unwinds at
    once while the sender keeps its reservation.
    """
    from concurrent.futures import ThreadPoolExecutor, wait
    from ouroboros.owner_pause import submit_model

    abandonable = bool(_ABANDONABLE.get() and settle_late is not None)
    lock, handover = threading.Lock(), {"state": "waiting"}

    def sender_keeps() -> bool:
        with lock:
            if handover["state"] == "abandoned":
                return True
            handover["state"] = "taken"
            return False

    def owned():
        try:
            result = send()
        except BaseException as exc:
            if sender_keeps():
                settle_late(error=exc)
            raise
        if sender_keeps():
            settle_late(result=result)
        return result

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-send")
    abandoned = False
    try:
        future = submit_model(reservation, executor.submit, owned if abandonable else send)
        if not abandonable:
            return future.result()
        while True:
            # Readiness is separate from retrieving the outcome: a provider's
            # own TimeoutError is an outcome, never another polling tick.
            if future.done() or wait((future,), timeout=ABANDON_POLL_SEC).done:
                return future.result()
            if not _receiver_paused(reservation):
                continue
            with lock:
                if handover["state"] != "waiting" or future.done():
                    continue  # completion/cancellation won: retrieve its outcome
                handover["state"] = "abandoned"
            abandoned = True
            raise _abandoned(reservation, future)
    finally:
        executor.shutdown(wait=not abandoned)


def retain_unadopted(request, reservation, **facts):
    """The retention callback for a paid answer its receiver will not adopt (cancelled or abandoned)."""
    from dataclasses import replace
    from ouroboros.llm_observability import retain_cancelled_response
    from ouroboros.usage_accounting import UsageScope

    target = replace(request, drive_root=reservation.drive_root,
                     task_id=(reservation.scope or UsageScope()).task_id or request.task_id)
    return lambda response, capture: retain_cancelled_response(target, response, capture, **facts)


def late_settler(reservation, request, extractor, manifest_ref, capture, late_owner):
    """``model_send``'s ``settle_late``: the SENDER's settlement of an answer its receiver abandoned.

    The late answer or error goes through the SAME accounting as a taken one; a received
    answer is retained as evidence (never adopted); the transport's ``late_owner`` then
    acknowledges or closes it. The receiver recorded the attempt as still dispatched.
    """
    from ouroboros import usage_accounting as ua

    retain = retain_unadopted(request, reservation, control_reason="owner_pause_abandoned")

    def settle_late(*, result=None, error=None) -> None:
        try:
            if error is not None:
                ua._terminalize_failed_attempt(reservation, error)
            else:
                ua._account_response(reservation, request, result, extractor, manifest_ref)
                retain(result, ua.last_physical_attempt_capture() or capture)
        except Exception:
            log.exception("Late settlement or retention of abandoned attempt %s failed; inspect durable attempt state",
                          reservation.attempt_id)
        if late_owner is not None:
            try:
                late_owner(result, error)
            except Exception:
                log.exception("Late transport custody of abandoned attempt %s failed", reservation.attempt_id)
    return settle_late


def _abandoned(reservation, sender):
    """Retire the exact receiver identity, then describe the interruption.

    The retired consumer can never accept this answer (``conflicting_writers``
    stops counting the in-flight attempt as a writer); the next send binds a
    fresh one. A failed retirement write keeps its positive witness (#1554).
    """
    from ouroboros.model_wait import ModelWaitInterrupted, current_model_wait

    owner = current_model_wait()
    if owner is not None:
        owner.rotate_answer_consumer()
    error = ModelWaitInterrupted("owner_pause")
    error.receiver_abandoned = True
    error.model_sender_future = sender
    error.ledger_attempt_ids = [reservation.attempt_id]
    return error


async def model_send_async(reservation, complete, *, retain_on_cancel, retain_on_abandon=None, late_owner=None):
    """Owner Pause hands settlement to the sender; caller cancellation joins it."""
    from ouroboros import usage_accounting as ua
    from ouroboros.owner_pause import submit_model

    outcome = {}
    async def invoke():
        outcome["entered"] = True
        try:
            outcome["response"] = await complete()
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            outcome["capture"] = ua.last_physical_attempt_capture()
        # No await separates completion from its handover. Only the event loop
        # changes this state, so Pause cannot also take a completed outcome.
        if outcome.get("abandoned"):
            try:
                if "error" not in outcome:
                    await asyncio.to_thread(retain_on_abandon, outcome["response"], outcome["capture"])
            except Exception:
                log.exception("Failed to retain abandoned paid response")
            finally:
                if late_owner is not None:
                    try:
                        await asyncio.to_thread(late_owner, outcome.get("response"), outcome.get("error"))
                    except Exception:
                        log.exception("Failed to close abandoned model transport")
        else:
            outcome["taken"] = True
        return outcome.get("response")  # Preserve exception identity in outcome.

    try:
        future = submit_model(reservation, lambda run, fn: run(asyncio.create_task, fn()), invoke)
    except BaseException as exc:
        exc.model_sender_not_started = True
        raise
    try:
        if _ABANDONABLE.get() and retain_on_abandon is not None:
            while not future.done():
                done, _pending = await asyncio.wait((future,), timeout=ABANDON_POLL_SEC)
                if done:
                    break
                if not outcome.get("taken") and _receiver_paused(reservation):
                    outcome["abandoned"] = True
                    # asyncio keeps weak references to tasks. The sender owns
                    # its paid receipt even after this receiver has unwound.
                    _ABANDONED_SENDS.add(future)
                    future.add_done_callback(_ABANDONED_SENDS.discard)
                    raise _abandoned(reservation, future)
        response = await asyncio.shield(future)
        if "error" in outcome:
            raise outcome["error"]
        return response
    except asyncio.CancelledError as cancelled:
        future.cancel()
        while not future.done():
            try:
                await asyncio.shield(future)
            except BaseException:
                if future.done():
                    break
        if not outcome.get("entered"):
            cancelled.model_sender_cancelled_before_entry = True
            raise cancelled
        try:
            response = future.result()
            if "error" in outcome:
                raise outcome["error"]
        except BaseException:
            # Task boundaries may synthesize a fresh CancelledError. Keep the
            # caller's cancellation, carrying the exact joined paid outcome.
            error = outcome.get("error", cancelled)
            for field in ("response", "response_manifest_ref", "response_retention_error", "physical_attempt_capture"):
                if hasattr(error, field):
                    setattr(cancelled, field, getattr(error, field))
            raise cancelled from error
        capture = outcome.get("capture")
        # Completion won the cancellation race; its value is still a paid answer.
        cancelled.response, cancelled.physical_attempt_capture = response, capture
        retention = asyncio.create_task(asyncio.to_thread(retain_on_cancel, response, capture))
        while not retention.done():
            try:
                await asyncio.shield(retention)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            cancelled.response_manifest_ref = retention.result()["manifest_ref"]
        except Exception as exc:
            cancelled.response_retention_error = type(exc).__name__
            log.exception("Failed to retain cancelled paid response")
        raise cancelled
    finally:
        capture = outcome.get("capture")
        if capture is not None:
            ua.adopt_physical_attempt_capture(capture)


def late_transport_custody(transport):
    """``late_owner`` for a transport whose receiver may leave (owner Pause): its SENDER
    acknowledges a retained answer exactly as an adopted one would be (the answer itself
    is never adopted), then closes the transport."""
    def settle(_result, error):
        try:
            if error is None and getattr(transport, "response_ref", None):
                transport.acknowledge()
        finally:
            transport.close()
    return settle


def close_unless_abandoned(transport) -> None:
    """The receiver's ``finally``: close its transport unless THIS exit is its own
    abandonment, whose sender still polls the transport and closes it after settling."""
    if not getattr(sys.exc_info()[1], "receiver_abandoned", False):
        transport.close()


def close_after_model_send(transport) -> None:
    """A temporary HTTP client stays with its abandoned sender until settlement."""
    sender = getattr(sys.exc_info()[1], "model_sender_future", None)
    if sender is None:
        transport.close()
    else:
        sender.add_done_callback(lambda _done: transport.close())


async def aclose_after_model_send(transport) -> None:
    """Async temporary-client custody follows the same exact sender future."""
    sender = getattr(sys.exc_info()[1], "model_sender_future", None)
    if sender is None:
        await transport.aclose()
        return

    def close(_done):
        task = asyncio.create_task(transport.aclose())
        _ABANDONED_SENDS.add(task)
        def finished(done):
            _ABANDONED_SENDS.discard(done)
            if not done.cancelled() and done.exception() is not None:
                log.error("Failed to close abandoned HTTP client", exc_info=done.exception())
        task.add_done_callback(finished)
    sender.add_done_callback(close)
