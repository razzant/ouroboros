"""The host clock line sealed into each Main physical model send (#1320, owner 6A/7A).

A task's context is captured once per run, so its ``context_captured_at`` stops
being "now" after the first round. Main instead reads the time of the request it
is answering: every NEW physical candidate a Main call prepares ends with one
host-authored clock line, sampled at that preparation — after any host wait
(backoff, quota wait, account rotation), before the candidate is measured,
priced, sealed and sent — in UTC and, when the sending client reported a valid
IANA zone, in that zone. An absent or unloadable zone is named unknown; the
server's own zone is never presented as the owner's.

Scope: only a Main send binds a policy (``loop_llm_call._send_main_candidate``);
delegated children, Presence, reviewers and helper calls made inside a Main call
(vision, reclaim summaries) bind none and their bytes are unchanged. Every loop that
sends through that seam still keeps a recovery send's omission of replayed reasoning
(below); the native review episode calls ``chat`` directly and does not.

Identity: every new physical send samples anew, including compatibility and
processing retries. Request-wire rebinding preserves the canonical tool source
and its applied actions. Rejoining one idempotent invocation retains its bytes.
The line of the send whose response came back becomes canonical history,
appended just before the answer (``main_send_scope``), so the next request
extends the last one byte for byte (OpenAI-family caches reuse only an exact
prefix, #906) and a replay shows what the model actually read; for the same
reason the history keeps a recovery send's omission of replayed reasoning. A line
no response consumed is never kept. A priced lookahead (the forced final) carries
a line of its own sample, so it is advice only: equal width is not equal
tokens. The final admission reads the FRESH request — its clock-free digest
(``AttemptRequest.candidate_clock_free_sha256``) must match the lookahead's,
and its OWN price must fit (``loop_forced_finalization._forced_admission_predicate``).
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

log = logging.getLogger(__name__)

# Task types whose loop is not an acting Main mind (the ingress responder and
# host summarizer/review passes); delegated children are excluded by role.
_NON_MAIN_TASK_TYPES = frozenset({"presence", "review", "summarize"})
CLOCK_NOTE_PREFIX = "[Host clock at this request: "
# The kept line is typed host state: a later line supersedes it (``context_source_view``).
CLOCK_ROW_NAME = "ouroboros_send_clock"


@dataclass(frozen=True)
class SendClockPolicy:
    """Main's opt-in; ``zone`` is a validated IANA name or ``""`` (unknown)."""

    zone: str = ""


def main_clock_policy(task_metadata: Any, *, task_type: str = "") -> Optional[SendClockPolicy]:
    """The policy of a Main loop, or None for a child, Presence or helper loop.

    The zone is the one the task's origin client reported
    (``metadata.client_surface.timezone``), re-validated here.
    """
    from ouroboros.client_surface import normalize_timezone_name

    meta = task_metadata if isinstance(task_metadata, dict) else {}
    if (str(meta.get("delegation_role") or "").strip().lower() == "subagent"
            or str(task_type or "").strip().lower() in _NON_MAIN_TASK_TYPES):
        return None
    surface = meta.get("client_surface") if isinstance(meta.get("client_surface"), dict) else {}
    return SendClockPolicy(zone=normalize_timezone_name(surface.get("timezone")))


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def render_clock_note(policy: SendClockPolicy, when: _dt.datetime) -> str:
    """Fixed width per policy: two samples differ in digits only, never in length."""
    utc = when.astimezone(_dt.timezone.utc).replace(microsecond=0)
    text = f"{CLOCK_NOTE_PREFIX}{utc.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC"
    zone = None
    if policy.zone:
        try:
            from zoneinfo import ZoneInfo

            zone = ZoneInfo(policy.zone)
        except Exception:
            zone = None
    if zone is None:
        return text + "; sender's time zone unknown]"
    local = utc.astimezone(zone)
    offset = local.strftime("%z")
    return (f"{text}; sender's zone {policy.zone}: {local.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(UTC{offset[:3]}:{offset[3:5]})]")


@dataclass
class MainSendClock:
    """One Main model call's clock custody: its policy and the lines it stamped.

    ``MainSendClock(None).bound()`` suspends an outer clock: helper calls made inside
    a Main call (vision, reclaim summaries) carry no line.
    """

    policy: Optional[SendClockPolicy]
    notes: List[str] = field(default_factory=list)
    by_candidate: Dict[str, str] = field(default_factory=dict)

    @contextlib.contextmanager
    def bound(self) -> Iterator["MainSendClock"]:
        token = _SCOPE.set(self if self.policy is not None else None)
        try:
            yield self
        finally:
            _SCOPE.reset(token)

    def consumed_note(self) -> Optional[str]:
        """The line of the candidate that produced this call's response, if known.

        Read from the ledger's own capture of the last physical attempt, never
        guessed from the order of stamping.
        """
        from ouroboros.usage_accounting import last_physical_attempt_capture

        capture = last_physical_attempt_capture()
        if capture is None or capture.state not in {"settled", "dispatched"}:
            return None
        return self.by_candidate.get(str(capture.candidate_raw_sha256 or ""))


_SCOPE: contextvars.ContextVar[Optional[MainSendClock]] = contextvars.ContextVar(
    "ouroboros_main_send_clock", default=None,
)


@contextlib.contextmanager
def main_send_scope(policy: Optional[SendClockPolicy],
                    canonical: Optional[List[Dict[str, Any]]] = None) -> Iterator[MainSendClock]:
    """Bind one Main send's clock; when the provider returned, replay its line.

    The consumed line is appended to ``canonical`` (the loop's transcript) before
    the caller appends the answer, so the canonical history is exactly what the
    model read and a prefix of the next request. For the same reason the ``canonical``
    of every loop sending through this scope adopts the omission of replayed reasoning
    by a recovery send this scope's call got its answer from (``reasoning_artifacts.pop_replay_omission``):
    a provider may bind a thinking block to the prefix it was produced on, so
    restoring the omitted blocks can get every later request refused. A raised send
    replays nothing. Without a policy the scope suspends any outer clock and adds
    no line.
    """
    from ouroboros.reasoning_artifacts import pop_replay_omission

    pop_replay_omission()  # only an omission by this scope's own send counts
    clock = MainSendClock(policy)
    with clock.bound():
        yield clock
    if pop_replay_omission() and isinstance(canonical, list):
        from ouroboros.llm import LLMClient

        canonical[:] = LLMClient._strip_openrouter_roundtrip_metadata(canonical, keep_reasoning_content=True)
        log.warning("The answered recovery send omitted the replayed reasoning; the transcript adopts the omission")
    note = clock.consumed_note() if policy is not None else None
    if note and isinstance(canonical, list):
        from ouroboros.context_budget import bound_host_row

        canonical.append(bound_host_row(CLOCK_ROW_NAME, note))  # typed host state; the wire copy drops the keys


def _tail_note(messages: List[Any], notes: List[str]) -> Tuple[Optional[str], List[Any]]:
    """``(this call's line at the tail or None, a copy of messages without it)``."""
    rest = list(messages or [])
    last = rest[-1] if rest and notes else None
    if not isinstance(last, dict) or str(last.get("role") or "") != "user":
        return None, rest
    content = last.get("content")
    if isinstance(content, str) and content in notes:
        return content, rest[:-1]
    block = content[-1] if isinstance(content, list) and content else None
    if (isinstance(block, dict) and set(block) == {"type", "text"}
            and block.get("type") == "text" and block.get("text") in notes):
        kept = content[:-1]
        return str(block["text"]), (rest[:-1] + [{**last, "content": kept}] if kept else rest[:-1])
    return None, rest


def split_clock_note(payload: Any) -> Tuple[Optional[str], Any]:
    """``(line, payload without it)`` for a line this call stamped, else ``(None, payload)``."""
    scope = _SCOPE.get()
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if scope is None or not isinstance(messages, list):
        return None, payload
    note, rest = _tail_note(messages, scope.notes)
    return (None, payload) if note is None else (note, {**payload, "messages": rest})


def record_candidate(raw_sha256: Optional[str], note: Optional[str]) -> None:
    """Remember which line a final candidate carries (read by ``consumed_note``)."""
    scope = _SCOPE.get()
    if scope is not None and note and raw_sha256:
        scope.by_candidate[str(raw_sha256)] = note


def stamp_clock_note(payload: Dict[str, Any], *, blocks: bool = False) -> Dict[str, Any]:
    """Return ``payload`` ending with a fresh clock line; no policy, no change.

    ``blocks`` is the Anthropic Messages shape: the line joins a trailing user
    turn as a text block, exactly as that provider's builder coalesces the
    canonical line on the next request. A pending line is replaced, never stacked;
    previously consumed lines belong to earlier scopes and remain history. The
    input payload is not mutated.
    """
    scope = _SCOPE.get()
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if scope is None or scope.policy is None or not isinstance(messages, list):
        return payload
    _found, rest = _tail_note(messages, scope.notes)
    note = render_clock_note(scope.policy, _now())
    scope.notes.append(note)
    last = rest[-1] if rest else None
    if (blocks and isinstance(last, dict) and str(last.get("role") or "") == "user"
            and isinstance(last.get("content"), list)):
        rest[-1] = {**last, "content": [*last["content"], {"type": "text", "text": note}]}
    else:
        rest.append({"role": "user", "content": note})
    return {**payload, "messages": rest}
