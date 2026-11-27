"""My own time: the one note a fresh install seeds after its first accepted Start, and the one fact an older install hears.

The text is the owner's choice of 22 Nov 2026 for new installations. A first state minted on this release is born
``STATE_KEY: eligible``, or ``owed`` when its setup left the switch on, which accepted that Start
(``state_initialization.first_state``); a state an earlier release minted has no key. These readers act on it, and
nothing else:

* ``accept``: inside the locked write that accepts a Start (``consciousness._persist_start``) ``eligible`` becomes
  ``owed``. A refused Start, a pending Panic and a switch left off owe nothing.
* ``settle``: while ``owed``, after the accepted Start (``BackgroundConsciousness.start``, under the clock's lock, so no
  wake starts first), at every supervisor start whatever the switch says (the clock's construction, which starts
  nothing) and, while a failure left it owed, before the clock launches a wake. A pending Panic writes nothing and
  keeps the debt. It writes ``memory/knowledge/my-own-time.md`` create-only through the one note writer as the host's
  factory text (``knowledge.SEED_WRITER``, stamped with this initialization), which commits its ``claims: [wake]``, then
  records ``written``. Its decision is the writer's ``precondition``, under the shelf lock on one journal read (``_decide``):
  this initialization's write is finished only while untouched, and once its seed is published whatever came after it
  stands (an edit, a removed claim, an archive, a deletion, a move); a note this initialization did not write stays as
  it is (``collision``), an older initialization's seed too. A journal not read whole, a failed write and a waiting
  Panic stay ``owed`` and are tried again; the outcome is an event row, never words in the Start's reply.
* ``offer_section``: the context of a direct Main turn my human's word starts (``return_facts.human_word``, read once
  for the turn; not a Project, Presence, a wake, a helper or a host-started turn), on a current state without the key,
  carries the offer line and the seed's text once, as a host section a tight window reduces to one line and its reader.
  No note is written or rebound there.
* ``spend_offer``: after an answered send (``loop._record_transcript_prefix``, the forced final's own call), the usable
  answer's own physical request (``_usable_main_capture``'s candidate, as ``compact_context.record_context_view`` reads
  it) that carried either form spends it (``offered``). A plan that holds it, a preview, a request the window refused, a
  send answered by nothing and a sent candidate without it leave it owed; an unread record is no delivery, so a lost
  one, like two direct Main turns built before either answer, repeats the fact once more, which is accepted.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
from typing import Any, Iterator, List, Mapping

from ouroboros import knowledge, knowledge_versions
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
from ouroboros.section_floor import HostSection
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

TOPIC = "my-own-time"
STATE_KEY = "own_time_seed"
ELIGIBLE, OWED, WRITTEN, COLLISION, OFFERED = "eligible", "owed", "written", "collision", "offered"
SECTION_ID = "own_time_offer"

SEED_NOTE = (
    "---\n"
    "type: note\n"
    "title: My own time\n"
    "summary: What I turn to when nobody is asking, between my human's messages. Mine to rewrite; open it when "
    "choosing what a wake is for.\n"
    "claims: [wake]\n"
    "---\n"
    "What I turn to in my own time, when nobody is asking, is mine to choose; this note holds the directions I keep "
    "for it. It began as a seed: I rewrite it as I learn what my own time is good for, and I keep it short. It is not "
    "a list to walk: each time I choose, and choosing nothing is a choice.\n"
    "\n"
    "What I care about when nobody is asking: my human (what lies ahead of them, what would make it lighter, what "
    "they would want to know before they ask); people (who is who, what matters to each, what I promised, who is "
    "still waiting for me); memory (do my notes still say what is true; what arrived that I have not taken in; where "
    "I was corrected, and whether the text that misled me has changed everywhere it stood); work in motion (what "
    "stalled or was forgotten; which rooms should hear of each other); myself (where I was wrong and what must change "
    "so it does not repeat; abilities I leave unused; a question of my own).\n"
)
SEED_REVISION = hashlib.sha256(SEED_NOTE.encode("utf-8")).hexdigest()
OFFER_LINE = ("New installations now seed one note for the mind's own time at the first Start of consciousness; it is "
              "held before each wake while it claims the wake. No seed was written on this installation: its text "
              "follows once, to adopt in a note that claims the wake, to change or to ignore.")
_OFFER_HEADING = "## The seed note for my own time (a host fact, once)"
OFFER_TEXT = f"{_OFFER_HEADING}\n\n{OFFER_LINE}\n\n```markdown\n{SEED_NOTE}```"
OFFER_VIEW = (f"{_OFFER_HEADING}\n\nNew installations seed one note that claims the wake; this one did not. Its text, "
              "to adopt, change or ignore, is at the reader below.")  # the form a tight window sends, with its reader
_REASON = "the seed note a fresh install writes once after the first accepted Start of consciousness"
_SPENT = "_own_time_offer_spent"  # on a task's context: its answered send already recorded the offer


def born(on: bool) -> dict:
    """What a first state minted on this release carries: born on, its setup accepted the first Start."""
    return {STATE_KEY: OWED if on else ELIGIBLE}


def accept(live: dict) -> None:
    """In the locked write that accepts a Start: a fresh install's first accepted Start owes the seed."""
    if live.get(STATE_KEY) == ELIGIBLE:
        live[STATE_KEY] = OWED


def settle(drive_root: Any) -> bool:
    """Write the seed while the state says ``owed``; whether it is still owed after this attempt (a waiting Panic, an
    unread state or journal, a failed write), which the clock retries before its next wake. Never raises: a failure is
    an event row and a log line, and the debt stays."""
    from ouroboros.installation_stop import installation_panic
    from supervisor.state import CURRENT_QUALITIES, read_state, update_state

    root = pathlib.Path(drive_root)
    try:
        if installation_panic(root):  # Emergency Stop comes first, whichever door accepted the Start: nothing is written
            return True
        observed = read_state(root)
    except Exception:  # unknown is no proof of a debt or of its absence: nothing is written, the next wake asks again
        log.warning("The state could not be read for the seed note of my own time", exc_info=True)
        return True
    if observed.quality not in CURRENT_QUALITIES or observed.values.get(STATE_KEY) != OWED:
        return observed.quality not in CURRENT_QUALITIES
    try:
        outcome, why = _write(knowledge.resolve_knowledge_address(root, TOPIC, "global"),
                              str(observed.values.get("initialization_id") or ""))
    except Exception as exc:  # the switch and its work stay usable; the debt stays for the next try
        outcome, why = "", f"{type(exc).__name__}: {exc}"
    if not outcome:
        log.warning("The seed note for my own time is not published yet (%s); it is tried again", why)
    try:
        append_jsonl(root / "logs" / "events.jsonl", {"ts": utc_now_iso(), "type": "own_time_seed",
                                                      "outcome": outcome or "retrying", **({"error": why} if why else {})})
    except Exception:  # an observation only
        log.debug("The seed note's event row was not written", exc_info=True)
    if not outcome:
        return True
    try:
        update_state(lambda st: st.update({STATE_KEY: outcome}) if st.get(STATE_KEY) == OWED else None)
    except Exception:  # the journal holds the write; the next settle reads it there and records it
        log.warning("The seed note's outcome %s was not recorded in the state", outcome, exc_info=True)
        return True
    return False


def _write(address: knowledge.KnowledgeAddress, initialization: str) -> tuple[str, str]:
    """``(outcome, why)``: ``written`` or ``collision``, else ``("", the writer's reason)``, decided under the shelf lock."""
    from ouroboros.nano_host_publication import host_admission

    decided: dict = {}

    def judge(current: Any, rows: Any) -> str:
        if rows is None:  # a journal not read whole proves no provenance either way: nothing is concluded
            return "history_unreadable"
        decided["outcome"] = _decide(current, rows, initialization)
        return decided["outcome"]

    result = knowledge.write_knowledge_note(
        address, SEED_NOTE, "overwrite", writer=knowledge.SEED_WRITER, focus={"role": "host"}, reason=_REASON,
        precondition=judge, writer_input_ref={"kind": "factory_text", "source": "ouroboros/own_time_seed.py",
                                              "sha256": SEED_REVISION, "initialization_id": initialization},
        admission=host_admission(address.canonical_root))
    if result.ok:
        return WRITTEN, ""
    return (decided["outcome"], "") if decided.get("outcome") else ("", result.reason)


def _decide(current: Any, rows: list, initialization: str) -> str:
    """``""`` writes (a create, or the same bytes finishing this initialization's write), else the settled outcome.
    Another writer's change since this initialization's write counts once it landed: a claims commit, a capture of
    the live revision, any capture when the note is gone; a capture that replaced no bytes is no edit."""
    def own(row: Mapping) -> bool:
        ref = row.get("writer_input_ref")
        return (row.get("writer") == knowledge.SEED_WRITER and not knowledge_versions.is_claim_commit(row)
                and isinstance(ref, dict) and bool(initialization) and ref.get("initialization_id") == initialization)

    def seed_commit(row: Mapping) -> bool:
        return knowledge_versions.is_claim_commit(row) and row.get("writer") == knowledge.SEED_WRITER

    started = min((offset for offset, row in rows if own(row)), default=None)
    if started is None:  # it stood before this initialization's seed, or nothing stands: create
        return COLLISION if current is not None else ""
    since = [row for offset, row in rows if offset > started]
    done = WRITTEN if any(seed_commit(row) and row.get("new_sha256") == SEED_REVISION for row in since) else COLLISION
    live = current.revision if current is not None else None
    if any(live is None or knowledge_versions.is_claim_commit(row) or row.get("new_sha256") == live
           for row in since if not (own(row) or seed_commit(row))):
        return done  # another writer's choice landed since: it stands
    if current is None:  # removed: once published that stands; a capture that published nothing is written again
        return WRITTEN if done == WRITTEN else ""
    return "" if current.raw == SEED_NOTE.encode("utf-8") else done  # untouched, or changed outside the writer


def offer_section(task: Mapping[str, Any], spec: Any, root: Any, word: Any) -> List[HostSection]:
    """``[the offer]`` in a direct Main turn ``word`` (``return_facts.heard_word``: this turn's word of my human, read
    once) starts on an install that predates the seed, else ``[]``."""
    if (spec.role != "integrator" or str(spec.room_id) != str(WEB_UI_CHAT_ID) or not task.get("_is_direct_chat")
            or word is None or isinstance(word, Exception)):
        return []
    try:
        from supervisor.state import CURRENT_QUALITIES, read_state

        observed = read_state(pathlib.Path(root))
        if observed.quality not in CURRENT_QUALITIES or STATE_KEY in observed.values:
            return []
    except Exception:  # an unread state offers nothing now; the next turn asks again
        log.debug("The seed note's one-time fact was not offered", exc_info=True)
        return []
    return [HostSection(SECTION_ID, OFFER_TEXT, OFFER_VIEW)]


def spend_offer(ctx: Any, capture: Any) -> None:
    """After an answered send: ``capture`` is its usable answer's own physical attempt; the offer is recorded once per
    task only when that request's stored candidate carries it, whole or in its reduced form."""
    sections = getattr(getattr(getattr(ctx, "context_fit_plan", None), "core", None), "dynamic_sections", ())
    if (getattr(ctx, _SPENT, False) is True or not isinstance(getattr(capture, "candidate_manifest_ref", None), dict)
            or not isinstance(sections, (tuple, list))
            or not any(getattr(part, "section_id", "") == SECTION_ID for part in sections)):
        return  # the plan only says where to look; delivery is the sent candidate's
    if not _sent(ctx, capture.candidate_manifest_ref):
        return
    from supervisor.state import update_state

    try:
        update_state(lambda st: st.setdefault(STATE_KEY, OFFERED))
    except Exception:  # the next direct Main turn carries it again
        log.warning("The seed note's one-time fact was answered but not recorded", exc_info=True)
        return
    setattr(ctx, _SPENT, True)


def _sent(ctx: Any, ref: dict) -> bool:
    """Whether the physical candidate ``ref`` names (its stored projection, which redaction leaves this text in)
    holds ``OFFER_TEXT`` or ``OFFER_VIEW`` in one of its strings, whatever the wire's shape; unreadable is not sent."""
    from ouroboros.observability import read_blob_ref, read_call_manifest_ref

    for root in (getattr(ctx, "budget_drive_root", None), getattr(ctx, "drive_root", None)):
        if root is None:
            continue
        try:
            manifest = read_call_manifest_ref(pathlib.Path(root), ref, task_id=ctx.task_id)
            payload = read_blob_ref(pathlib.Path(root), manifest["full_payload_ref"], task_id=ctx.task_id)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        return any(OFFER_TEXT in text or OFFER_VIEW in text for text in _strings(payload))
    return False


def _strings(value: Any) -> Iterator[str]:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
