"""Model-call evidence and event transport, independent of admission and retry decisions.

The Main call publishes lifecycle and optional context through this existing observation
owner; the provider loop chooses what happened, never how the public event is shaped. A
round's ``llm_round`` also carries what the host did to prepare it (``round_window``).
"""

from __future__ import annotations

import contextlib
import pathlib
import logging
import queue
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple

from ouroboros.context_timing import STEPS as _STEPS, carry_spans, host_step, measured_sections, section_spans, timed
from ouroboros.observability import new_call_id, persist_call
from ouroboros.anthropic_native_custody import public_custody_projection
from ouroboros.utils import emit_cognitive_operation_event, sanitize_tool_result_for_log, utc_now_iso
from ouroboros.physical_attempt_context import attempt_fact


def emit_main_call_state(event_queue: Optional[queue.Queue], identity: Tuple[Any, ...],
                        phase: str, activity: Any = None, capture: Any = None) -> None:
    """Exact-call lifecycle and display observations; only terminal facts close the idle lease. Once this round's
    attempt is dispatched, its frames carry its context (``physical_attempt_context.attempt_fact`` of ``capture``).
    Event delivery failure cannot change the provider loop's send or retry decision."""
    task_id, task_attempt, llm_call_id, execution_id, round_id, call_attempt = identity
    if phase in {"finished", "failed"}:
        timed("event_put", emit_cognitive_operation_event, event_queue, task_id=task_id, operation_id=llm_call_id,
              phase=phase, kind="llm", task_attempt=task_attempt, execution_id=execution_id, round_id=round_id)
    fact = attempt_fact(capture, str(round_id or ""))  # ``dispatched``: the ledger's own dispatch, while it waits
    if event_queue is None or (phase == "observed" and not isinstance(activity, dict)) or (phase == "dispatched" and not fact):
        return
    try:
        timed("event_put", event_queue.put, {
            "type": "main_llm_call_state", "task_id": str(task_id or ""), "task_attempt": task_attempt,
            "llm_call_id": str(llm_call_id or ""), "execution_id": str(execution_id or ""),
            "round_id": str(round_id or ""), "call_attempt": int(call_attempt), "phase": phase,
            **({"model_activity": activity} if phase == "observed" else {}),
            **({"context_attempt": fact} if fact else {}),
        })
    except Exception:
        logging.getLogger(__name__).debug("Failed to emit main LLM call state event", exc_info=True)


_RUSAGE = ("ru_nvcsw", "ru_nivcsw", "ru_majflt", "ru_oublock")


def _rusage() -> Dict[str, int]:
    """The process's voluntary and involuntary context switches, major page faults and block writes so far: waiting,
    being preempted, paging, writing. Nothing where the platform keeps no such counters (Windows has no ``resource``)."""
    try:
        import resource
    except ImportError:
        return {}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {name[3:]: int(getattr(usage, name)) for name in _RUSAGE}


def round_starts(slot: Any) -> None:
    """A round's preparation starts now, at a batch's end: the next ``round_window`` on ``slot`` times its first step,
    ``round_top`` (the loop's own work up to the model call), and its ``rusage`` from here."""
    slot._round_prep_since = (time.perf_counter(), time.thread_time(), threading.get_ident(), _rusage())


@contextlib.contextmanager
def round_window(slot: Any = None) -> Iterator[None]:
    """One round's model call as a ``context_timing`` window, receipted on its ``llm_round`` (``round_receipt``): the
    counted work, each marked section's wall time, the steps between ``host_step`` marks (``round_top`` from
    ``round_starts`` on ``slot``, used once) and the process's ``_rusage`` since the round's start. When that call
    fails the loop opens another, without ``slot``, around the recovery (a configured fallback, an owner-wait resend),
    whose answer carries the recovery's own facts. Facts for the log only: nothing reads them to decide."""
    since, prologue = getattr(slot, "_round_prep_since", None), getattr(slot, "_round_prologue", None)
    if since is not None:
        slot._round_prep_since = None
    if prologue is not None:  # measured before this window: a start (``loop``), a post-batch read (``loop_budget``)
        slot._round_prologue = None
    now = (time.perf_counter(), time.thread_time(), threading.get_ident())
    with measured_sections():
        carry_spans("prologue", prologue)
        token = _STEPS.set({"steps": {}, "mark": since[:3] if since else now, "rusage": since[3] if since else _rusage()})
        try:
            if since is not None:
                host_step("round_top")
            yield
        finally:
            _STEPS.reset(token)


def round_receipt() -> Dict[str, Any]:
    """``work`` and ``host_steps`` of the open round window for its ``llm_round``, or nothing outside one:
    ``steps`` as ``[wall ms, thread CPU ms or None]``, the sections' wall ms (``total``: the window so far) and ``rusage``.
    The line lands in an archive kept for good, so it holds about a kilobyte, one entry per name."""
    spans, state = section_spans(), _STEPS.get()
    if not spans or state is None:
        return {}
    rusage = _rusage()
    return {"work": spans["work"], "host_steps": {
        "steps": {name: [round(span[0] * 1000), None if span[1] is None else round(span[1] * 1000)]
                  for name, span in state["steps"].items()},
        "sections_ms": spans["sections_ms"],
        **({"rusage": {name: count - state["rusage"].get(name, count) for name, count in rusage.items()}}
           if rusage else {})}}


def retain_cancelled_response(request, response, capture, *, control_reason: str = "caller_cancelled") -> dict:
    """Keep a received physical response before cancellation unwinds its caller.

    Reuse private CAS and the ordinary call reader. This records the provider
    object (model_dump for SDK responses), not a completed task/review verdict.
    Native transports continue to own any original wire-byte receipt separately.
    """
    from dataclasses import asdict

    body = response.model_dump() if hasattr(response, "model_dump") else response
    return persist_call(
        request.drive_root, task_id=request.task_id or "llm",
        call_id=f"physical_{capture.attempt_id}_response", call_type="physical_response",
        payload={"response": body, "physical_attempt_capture": asdict(capture)}, keep_raw=True,
        manifest={"attempt_id": capture.attempt_id, "model": capture.model,
                  "provider": capture.provider, "state": capture.state,
                  "status": "received", "control_reason": control_reason},
    )


class SendRefusals:
    """Observe each physical refusal before an inner retry can replace its capture. The caller's claim and end
    observer keep their existing contract. The receiver logs and queues the observed pairs at logical exit, handing
    the pair for its own error or returned body to that existing event. An abandoned send can only log. A new
    candidate consumes only the pairs carried by its prepared host rows and only at physical dispatch."""

    def __init__(self, claim: Any, drive_logs: Any, event_queue: Optional[queue.Queue], identity: Dict[str, Any],
                 usage: Optional[Dict[str, Any]] = None):
        self.claim, self.drive_logs, self.event_queue, self.identity = claim, drive_logs, event_queue, identity
        self.usage = usage
        self._lock = threading.Lock()
        self._returned: Dict[str, bool] = {}
        self._refused: Dict[str, Dict[str, Any]] = {}
        self._seen: set = set()
        self._abandoned: set = set()
        self._prepared: set = set()
        self._delivered: Dict[str, list] = {}
        self._closed = False

    def __call__(self, reservation: Any) -> None:
        if self.claim is not None:
            self.claim(reservation)

    def prepared_messages(self, messages) -> None:
        """Capture only private identities from this send's prepared rows; the wire carries the usual text."""
        from ouroboros.context_view_facts import refused_sends_in_messages

        self._prepared = set()
        try:
            if self.usage:
                self._prepared = refused_sends_in_messages(self.usage, messages)
        except Exception:
            logging.getLogger(__name__).debug("Could not read prepared refused sends", exc_info=True)

    def dispatched(self, capture: Any) -> None:
        """Called by the physical observer after the ledger's mark, never by the pre-send claim."""
        from ouroboros.context_view_facts import consume_refused_sends

        if capture.state == "dispatched" and self._prepared and self._bound(capture):
            self._delivered[capture.attempt_id] = consume_refused_sends(self.usage, self._prepared)

    def _bound(self, capture: Any) -> str:
        from ouroboros.physical_attempt_context import OBSERVED

        own_round = getattr(getattr(capture, "physical_context", None), "round_id", "")
        if getattr(capture, "state", "") in OBSERVED and (not own_round or own_round == self.identity.get("round_id")):
            return str(capture.attempt_id or "")
        return ""

    def attempt_ended(self, capture: Any, error: Any) -> None:
        ended = getattr(self.claim, "attempt_ended", None)
        if ended is not None:
            ended(capture, error)
        try:
            self._observe(capture, error)
        except Exception:
            logging.getLogger(__name__).warning("Could not read a refused send's counts", exc_info=True)

    def _observe(self, capture: Any, error: Any) -> None:
        from ouroboros.context_view_facts import REFUSED_SENDS_KEY
        from ouroboros.size_refusal import send_counts

        attempt_id = capture.attempt_id
        abandoned = getattr(error, "receiver_abandoned", False)
        with self._lock:
            delivered = self._delivered.pop(attempt_id, None)
            if delivered and capture.state == "released" and not self._closed and attempt_id not in self._abandoned:
                self.usage.setdefault(REFUSED_SENDS_KEY, [])[:0] = delivered  # commitment was proven unsent
            if abandoned:
                self._abandoned.add(attempt_id)
                late = self._refused.pop(attempt_id, None)
            else:
                if not self._closed and attempt_id not in self._abandoned and (self._bound(capture) or capture.state == "released"):
                    self._returned[attempt_id] = error is None and capture.state != "released"
                late = None
        if late:
            self._log(late, late=True)
        if not abandoned and isinstance(error, Exception) and self._bound(capture):
            self._record(send_counts(capture, error))

    def returned_body_refusal(self, capture: Any, payload: dict) -> None:
        """The wire dict paired by ``_account_response`` with its own capture, for both taken and abandoned sends."""
        from ouroboros.size_refusal import response_counts

        try:
            if self._bound(capture):
                self._record(response_counts(capture, payload))
        except Exception:
            logging.getLogger(__name__).warning("Could not read a refused response's counts", exc_info=True)

    def _record(self, counts: Optional[Dict[str, Any]]) -> None:
        if not counts:
            return
        attempt_id = counts["physical_attempt_id"]
        with self._lock:
            if attempt_id in self._seen:
                return
            self._seen.add(attempt_id)
            late = self._closed or attempt_id in self._abandoned
            if not late:
                self._refused[attempt_id] = counts
        if late:
            self._log(counts, late=True)

    def _returned_id(self, ledger_ids) -> Optional[str]:
        # The wire ladder can return a previous body after a later send raises. An id missing from this tracker
        # may have returned that body: do not assign it to the most recent ambient capture instead.
        for attempt_id in reversed(ledger_ids):
            if attempt_id in self._abandoned or attempt_id not in self._returned:
                break
            if self._returned[attempt_id]:
                return attempt_id
        return None

    def close(self, error: Any, usage: Dict[str, Any], *, refused_body_ids=()) -> Optional[Dict[str, Any]]:
        """Close the receiver; yield only the comparison its existing event can own, and tell every observed one."""
        from ouroboros.context_view_facts import REFUSED_SENDS_KEY
        from ouroboros.transport_custody import _capture_on_chain

        with self._lock:
            own = (self._returned_id(refused_body_ids) if error is None else
                   getattr(_capture_on_chain(error), "attempt_id", None))
            if hasattr(error, "deadline_attempt_capture"):  # a prior send's custody is not this deadline's answer
                own = None
            refused, self._refused, self._closed = self._refused, {}, True
        final = refused.pop(own, None) if own else None
        for counts in refused.values():
            self._log(counts)
        if told := [*refused.values(), *([final] if final else [])]:
            usage.setdefault(REFUSED_SENDS_KEY, []).extend(told)
        return final

    def _log(self, counts: Dict[str, Any], *, late: bool = False) -> None:
        """One refused send's own Logs row under its call's identity; the call's error row is never repeated here."""
        from ouroboros.utils import append_jsonl, emit_log_event, has_log_sink, utc_now_iso

        row = {"ts": utc_now_iso(), "type": "local_context_overflow" if counts["provider"] == "local" or counts["route_is_loopback"]
               else "remote_context_overflow", **self.identity, "physical_attempt_id": counts.get("physical_attempt_id"),
               "refused_send": counts, **({"settled_after_its_call": True} if late else {})}
        try:
            if not append_jsonl(pathlib.Path(self.drive_logs) / "events.jsonl", row) or not has_log_sink():
                emit_log_event(self.event_queue, row, log_label="Refused send")
        except Exception:
            logging.getLogger(__name__).warning("Could not log a refused send", exc_info=True)


def persist_observed_call(root: Any, *, payload: Any, writer: Any = None, **identity: Any) -> dict:
    """Best-effort public trace: opaque continuation never enters this projection.

    Failure to write observability cannot discard an already useful response or
    authorize another provider call. Private exact-byte transport custody uses
    persist_call directly and retains its own stronger acknowledgement contract.
    """
    try:
        return (writer or persist_call)(root, payload=public_custody_projection(payload), **identity)
    except Exception:
        logging.getLogger(__name__).warning("Failed to persist LLM observability payload", exc_info=True)
        return {}


def _root(drive_root: Any) -> pathlib.Path:
    try:
        return pathlib.Path(drive_root)
    except TypeError:
        return pathlib.Path("../data")


def _base_manifest(call_type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "call_type": call_type,
        "model": kwargs.get("model"),
        "reasoning_effort": kwargs.get("reasoning_effort"),
        "max_tokens": kwargs.get("max_tokens"),
        "use_local": kwargs.get("use_local"),
    }


def chat_observed(
    llm: Any,
    *,
    drive_root: Any,
    task_id: str = "",
    call_type: str = "llm_call",
    **kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run ``llm.chat`` while preserving request/response/error payloads."""

    root = _root(drive_root)
    call_id = new_call_id(call_type)
    traced_kwargs = dict(kwargs)
    persist_observed_call(
        root,
        task_id=task_id or call_type,
        call_id=f"{call_id}_request",
        call_type=f"{call_type}_request",
        payload={"kwargs": traced_kwargs},
        manifest=_base_manifest(call_type, kwargs),
    )
    try:
        msg, usage = llm.chat(**kwargs, chat_conversation_id=call_id)
    except Exception as exc:
        safe = sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}")
        reply = getattr(exc, "reply_message", None)  # a non-result reply keeps its output and usage on record
        persist_observed_call(
            root,
            task_id=task_id or call_type,
            call_id=f"{call_id}_error",
            call_type=f"{call_type}_error",
            payload={"error": f"{type(exc).__name__}: {exc}", "kwargs": traced_kwargs,
                     **({"message": reply, "usage": getattr(exc, "usage", {})} if isinstance(reply, dict) else {})},
            manifest={**_base_manifest(call_type, kwargs), "status": "error", "error": safe},
        )
        raise
    try:
        from ouroboros.openai_chat_dispatch import CUSTOM_RECEIPTS_USAGE_KEY

        public_usage = dict(usage)
        public_usage.pop(CUSTOM_RECEIPTS_USAGE_KEY, None)
        persist_call(
            root,
            task_id=task_id or call_type,
            call_id=f"{call_id}_response",
            call_type=f"{call_type}_response",
            payload={
                "message": public_custody_projection(msg),
                "usage": public_usage,
            },
            manifest={**_base_manifest(call_type, kwargs), "status": "ok"},
        )
    except Exception:
        pass
    return msg, usage


def _remember_llm_call(
    usage: Dict[str, Any], *, call_ids: Dict[str, Any], model: str, display_model: str, provider: str,
    request_ref: Dict[str, Any], response_ref: Dict[str, Any],
    reported_model: Any = None, use_local: Optional[bool] = None,
    requested_profile: Optional[str] = None, observed_route: Optional[Dict[str, Any]] = None,
    now: Any = utc_now_iso,
) -> None:
    call_meta = {
        "ts": now(), **call_ids,
        "model": model, "resolved_model": display_model, "provider": provider,
        "reported_model": reported_model.strip() if isinstance(reported_model, str) and reported_model.strip() else None,
        "use_local": use_local,
        "request_ref": request_ref.get("manifest_ref") if request_ref else None,
        "response_ref": response_ref.get("manifest_ref") if response_ref else None,
        "requested_profile": requested_profile, "observed_route": dict(observed_route or {}),
    }
    usage["_last_llm_call_meta"] = call_meta
    usage.setdefault("llm_call_refs", []).append(call_meta)


def record_main_request(
    drive_root: pathlib.Path, *, task_id: str, llm_call_id: str,
    messages: Any, send_messages: Any, tools: Any, model: str, effort: str,
    max_tokens: int, use_local: bool, allow_server_web_search: bool,
    response_cache_bypass_requested: bool, accumulated_usage: Dict[str, Any],
    call_ids: Dict[str, Any], context_fit_event_fields: Dict[str, Any],
    persist: Any = persist_observed_call, writer: Any = persist_call, now: Any = utc_now_iso,
) -> Dict[str, Any]:
    """Persist the Main call's request projection and return its manifest reference."""
    return persist(
        drive_root, task_id=task_id, call_id=f"{llm_call_id}_request", call_type="llm_request",
        payload={
            "messages": messages,
            "send_messages": send_messages,  # shares its equal elements with ``messages`` (stored once)
            "tools": tools or [],
            "model": model,
            "reasoning_effort": effort,
            "max_tokens": max_tokens,  # the logical caller ceiling; the sent allowance is the physical candidate's
            "use_local": bool(use_local),
            "allow_server_web_search": bool(allow_server_web_search),
            "response_cache_bypass_requested": response_cache_bypass_requested,
        },
        manifest={
            "first_request_at": accumulated_usage.setdefault("first_request_at", now()),
            **call_ids,
            "model": model,
            "reasoning_effort": effort,
            "response_cache_bypass_requested": response_cache_bypass_requested,
            **context_fit_event_fields,
        },
        writer=writer,
    )


def record_main_response(
    drive_root: pathlib.Path, *, task_id: str, llm_call_id: str,
    msg: Dict[str, Any], usage: Dict[str, Any], accumulated_usage: Dict[str, Any],
    call_ids: Dict[str, Any], model: str, display_model: str, provider: str,
    use_local: bool, request_ref: Dict[str, Any], physical: Any,
    host_route: Dict[str, Any], model_account_override: Optional[str],
    persist: Any = persist_observed_call, writer: Any = persist_call, now: Any = utc_now_iso,
) -> Dict[str, Any]:
    """Persist the Main call's response projection and append its call metadata to ``accumulated_usage``."""
    response_ref = persist(
        drive_root,
        task_id=task_id,
        call_id=f"{llm_call_id}_response",
        call_type="llm_response",
        payload={
            "message": msg,
            "usage": usage,
        },
        manifest={
            "first_answer_at": accumulated_usage.setdefault("first_answer_at", now()),
            **call_ids,
            "model": model,
            "resolved_model": display_model,
            "provider": provider,
        },
        writer=writer,
    )
    attempt_id = (usage.get("ledger_attempt_ids") or [None])[-1]
    _remember_llm_call(
        accumulated_usage,
        call_ids={
            **call_ids,
            "physical_attempt_id": attempt_id,
            **({
                "physical_model": physical.model,
                "physical_provider": physical.provider,
                "dispatch_account": getattr(physical.physical_context, "dispatch_account", ""),
                **({"processing_preference": physical.processing_preference}
                   if hasattr(physical, "processing_preference") else {}),
            } if physical is not None and physical.state == "settled" and physical.attempt_id == attempt_id else {}),
            **({"provider_body_error": True} if usage.get("provider_error") else {}),
        },
        model=model, display_model=display_model, provider=provider,
        request_ref=request_ref, response_ref=response_ref,
        reported_model=usage.get("reported_model", usage.get("resolved_model")), use_local=bool(use_local),
        requested_profile=(usage.get("claudexor") or {}).get(
            "requested_profile", host_route.get("credential_profile_id", model_account_override)
        ),
        observed_route=(usage.get("claudexor") or {}).get("route"),
        now=now,
    )
    return response_ref
