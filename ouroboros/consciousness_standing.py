"""What stands in every own-clock wake while it holds: my human's answerable cards and the standing sections.

A wake's observation is a delta (``consciousness_wake.observe_wake``): each transition reaches one accepted wake.
After it every wake shows two inventories instead, in this order: my human's answerable cards (``outstanding_cards``;
the card line renderers live here, also for the card transitions the observation prints), then the standing sections
(``STANDING_PRODUCERS``): host facts, a work's line with its reader, never a waiting class, an instruction or a
ranking. A section with nothing to say is not printed. They stand whole in the full text and in the projection the
wake takes when its observation does not fit; only the bare pointer folds each to its count and the kind of its rows.

Works that ended without success (``fallen_works``): in each open project (lifecycle active, no ``closed`` record;
a root's project is its binding when it has one), its newest-started work root when that root ended failed or
cancelled and ``ended_without_success`` holds. A root that starts later in the project (a Continue successor is one)
or closing the project takes it off the list; nothing ages it out. Presence's own rows (a speaker turn, work acting
for a binding, a carrier that lost its id) neither stand nor retire; the owner's work related to a binding does both.
Not covered: works with no project, and a root that fails while a later-started root of its project exists. Cost per
wake: one index probe per open project, one primary-key read of the roots bound to open projects and the bodies of
the candidates, all in the same snapshot; nothing is stored. An unreadable binding store or a damaged row, or an
empty registry not proved empty (lost beside its commit witness, or no row readable), is the section's named gap.

Own clock (``_clock_section``): ``consciousness_clock.own_clock_lines`` over the state the alarm read when it admitted
the wake, while a named time or my human's reported zone stands; the appointment and the bounds alone make none.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from ouroboros.consciousness_authority import is_consciousness_origin, wake_cause
from ouroboros.deadline_utils import ago_text


def _parse_iso(value: Any) -> Optional[float]:
    try:
        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _clip_preview(value: Any, limit: int = 100) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 18)].rstrip() + f" …[{len(text) - (limit - 18)} chars omitted]"


# --- my human's cards --------------------------------------------------------------------


def _card_line(task_id: str, quiz_id: str, block: Dict[str, Any], *, now: float, owner_wait: Any = None,
               closed: Optional[str] = None) -> str:
    """One card's status line; ``closed`` (a transition's own stamp, or none) replaces its asked age and wait."""
    from ouroboros.project_dialogue import owner_wait_projection, question_status

    facts = owner_wait_projection(quiz_id, owner_wait, block)
    waits = closed is None and bool(block.get("wait_for_answer"))
    label = question_status(str(block.get("state") or "open"), facts, waits)
    age = closed
    if age is None:  # still open: its asked age
        asked_ts = _parse_iso(str(block.get("asked_at") or ""))
        age = f", {ago_text(now - asked_ts)}" if asked_ts is not None else ""
    preview = _clip_preview(block.get("question") or "question text unavailable")
    return f"- owner card {quiz_id} on task {task_id}: {label}{age}; {preview}"


def _answered_card_line(task_id: str, quiz_id: str, block: Dict[str, Any], *, now: float) -> str:
    """One answered card of the window: stamps, the recorded answer whole (my human's own words verbatim, so in the
    fallback form too; a press as their act on the asking task's label, ``answer_facts.option_answer``), a named
    question preview and no verdict on what the answer meant. A secret request's answer is the
    host's typed outcome: its event, plugin and field ids, never a value."""
    from ouroboros.answer_facts import option_answer
    from ouroboros.owner_quiz import secret_outcome_block, secret_outcome_summary

    parts = []
    for key, word in (("answered_at", "answered"), ("asked_at", "asked")):
        stamp = _parse_iso(block.get(key))
        parts.append(f"{word} {ago_text(now - stamp)}" if stamp is not None else f"{word} at an unknown time")
    index = block.get("answered_index")
    comment = str(block.get("comment") or "")
    if secret_outcome_block(block) is block:
        answer = f"host outcome {secret_outcome_summary(block)}"
    elif isinstance(index, int) and not isinstance(index, bool):
        answer = option_answer(block, index, comment, asker=task_id)
    elif comment.strip():
        answer = f"answered in own words: {comment}"
    else:
        answer = "answer text unavailable"
    preview = _clip_preview(block.get("question") or "question text unavailable")
    return f"- owner card {quiz_id} on task {task_id}: {'; '.join(parts)}; {answer}; question: {preview}"


def _closed_card_line(task_id, quiz_id, block, *, now):
    """A card closed without an answer (``quiz_closed``), dated by its own ``closed_at``, which a time-bootstrapped
    window reads too; its actor and reason are never read as an answer."""
    stamp = _parse_iso(block.get("closed_at"))
    when = ago_text(now - stamp) if stamp is not None else "at an unknown time"
    actor = str(block.get("closed_actor") or "unknown actor")
    if actor == "model" and block.get("closed_by_task_id"):
        actor += f" task {block['closed_by_task_id']}"
    return (f"- owner card {quiz_id} on task {task_id}: closed {when} by {actor}, without an owner answer; "
            f"reason: {block.get('closed_reason') or 'unavailable'}; "
            f"question: {_clip_preview(block.get('question') or 'question text unavailable')}")


def outstanding_cards(results: Mapping[str, Dict[str, Any]], *, now: float) -> Tuple[str, ...]:
    """Every answerable card (``open``/``expired_terminal`` — В17a) of the rows read, newest asked first."""
    from ouroboros.owner_quiz import STATE_EXPIRED_TERMINAL, STATE_OPEN

    cards = []
    for task_id, row in results.items():
        quizzes = row.get("owner_quiz") if isinstance(row.get("owner_quiz"), dict) else {}
        for quiz_id, block in quizzes.items():
            if isinstance(block, dict) and not block.get("answered_at") and block.get("state") in (
                    STATE_OPEN, STATE_EXPIRED_TERMINAL):
                cards.append((str(block.get("asked_at") or ""), _card_line(
                    task_id, str(quiz_id), block, now=now, owner_wait=row.get("owner_wait"))))
    return tuple(line for _stamp, line in sorted(cards, reverse=True))


# --- standing sections ---------------------------------------------------------------------


@dataclass(frozen=True)
class StandingSection:
    """One section: its row ``kind`` (the observation's composition and source rows), title, composition words
    ``(one, many)`` and lines."""

    kind: str
    title: str
    words: Tuple[str, str]
    lines: Tuple[str, ...]

    def heading(self) -> str:
        return f"{self.title} ({len(self.lines)}):"


@dataclass(frozen=True)
class StandingFacts:
    """What a section is computed from in the wake's carrier: the data root, the wake's instant, the project
    registry its observation read (``None``: unreadable) and the own clock's facts the alarm admitted it with
    (``consciousness._launch``: this wake's appointment, the named time and my human's last reported zone under their
    state keys, the owner's MIN and MAX as ``floor`` and ``ceiling``; ``None`` when none travelled)."""

    root: Any
    now: float
    registry: Optional[List[Dict[str, Any]]]
    clock: Optional[Mapping[str, Any]] = None


FALLEN_KIND = "fallen_work"
FALLEN_TITLE = "Works that ended without success; nothing has started in their project since"
FALLEN_WORDS = ("work that ended without success", "works that ended without success")
FALLEN_STATUSES = ("failed", "cancelled")
AGENT_TOOL_SOURCE = "agent_tool"  # ``join_ledger`` cancel_task: a focus's own tool stopped it


class FallenWork(NamedTuple):
    task_id: str
    project_id: str
    line: str


def ended_without_success(row: Any) -> bool:
    """Whether a task row is a work root that ended without success and was taken up by nobody: failed or cancelled,
    and its outcome word says so (``outcome_phase``: a stored failure whose own final answer reached the chat before
    its finalization was cut reads Done everywhere, owner 15.11); not stopped by my human (an
    ``owner_continue.OWNER_STOP_SOURCES`` origin, Panic included, read here: a stopped root's retained budget cause can
    answer ``hard_limit_reached`` before the offer reads the stop) nor wrapped up at their word
    (``owner_requested_finalization``); not cancelled by a focus's tool; not a wake (work a wake started is not one);
    no successor: a Continue past ``bound``, a supersession or a retry under another id. A Continue bound and not yet
    admitted keeps it (its line names the successor). Presence's own rows are not judged here: ``fallen_works`` skips
    them by their typed column."""
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION, recorded_cause
    from ouroboros.owner_continue import OWNER_STOP_SOURCES
    from ouroboros.project_dialogue import outcome_phase

    if not isinstance(row, dict) or str(row.get("status") or "") not in FALLEN_STATUSES:
        return False
    if outcome_phase(row, {}) not in ("error", "cancelled"):  # the word every owner surface says: Failed or Stopped
        return False
    task_id = str(row.get("task_id") or "")
    if str(row.get("parent_task_id") or "") or row.get("delegation_role") == "subagent":
        return False  # a child's work is its root's (``store.ROOT_RESULT_WHERE``)
    origin = row.get("cancel_origin") if isinstance(row.get("cancel_origin"), dict) else {}
    if str(origin.get("source") or "") in OWNER_STOP_SOURCES | {AGENT_TOOL_SOURCE}:
        return False
    if recorded_cause(row) == REASON_OWNER_REQUESTED_FINALIZATION:
        return False
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if is_consciousness_origin(metadata) and wake_cause(metadata):
        return False
    if any(str(row.get(key) or "") not in ("", task_id) for key in ("superseded_by", "retry_task_id")):
        return False
    claim = row.get("continued_by") if isinstance(row.get("continued_by"), dict) else {}
    return not (claim.get("successor_task_id") and claim.get("state") != "bound")


def open_projects(projects: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """``{project id: name}`` of the open projects among registry rows: lifecycle active and no ``closed`` record."""
    from ouroboros.project_closure import select_projects
    from ouroboros.projects_registry import PROJECT_ACTIVE

    active = [row for row in projects if isinstance(row, Mapping) and row.get("lifecycle") == PROJECT_ACTIVE]
    return {str(row["id"]): str(row.get("name") or row["id"]) for row in select_projects(active, closed=False)
            if str(row.get("id") or "")}


def fallen_line(row: Dict[str, Any], *, project: str, now: float) -> str:
    """One work, card-like: id, its outcome word and when it ended, its title (else its description, else its request),
    its recorded cause, the Continue offer as ``owner_continue.offer_view`` reads it now, the work it continues when its
    own body records that, its reader; never its summary again (its terminal line carried that once)."""
    from ouroboros.outcomes import recorded_cause
    from ouroboros.owner_continue import offer_view
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, outcome_phase
    from ouroboros.terminal_time import terminal_time_fact

    task_id = str(row.get("task_id") or "")
    stamp = _parse_iso(terminal_time_fact(row).get("occurred_at"))
    ended = f"ended {ago_text(now - stamp)}" if stamp is not None else "end time not recorded"
    title = _clip_preview(row.get("title") or row.get("description") or row.get("text") or "title not recorded", 100)
    origin = row.get("cancel_origin") if isinstance(row.get("cancel_origin"), dict) else {}
    cause = recorded_cause(row) or "not recorded"
    if origin.get("source"):  # a cancel records its cause as its origin
        cause += f"; cancel source {origin['source']}"
    offer = offer_view(row, task_id) or {}
    if offer.get("eligible"):
        continuing = f"Continue open ({offer.get('cause') or 'cause not recorded'})"
    elif offer.get("refusal") == "continuation_unconfirmed":
        continuing = f"Continue admission unconfirmed (successor {offer.get('successor_task_id')})"
    else:
        continuing = f"Continue refused ({offer.get('refusal') or 'not offered'})"
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    of = metadata.get("continuation") if isinstance(metadata.get("continuation"), dict) else {}
    lineage = ""
    if of.get("predecessor_task_id"):
        lineage = f"; continuation of {of['predecessor_task_id']}" + (f", work {of['work_id']}" if of.get("work_id") else "")
    outcome = OUTCOME_PHASE_HEADLINE[outcome_phase(row, {})]
    return (f"- task {task_id} {outcome} in project {project}, {ended}: {title}; cause {cause}; {continuing}{lineage}; "
            f"get_task_result(task_id='{task_id}')")


def fallen_works(root: Any, *, now: float, projects: Iterable[Mapping[str, Any]]) -> List[FallenWork]:
    """The works that stand, newest-started first: in each open project its newest-started work root when it ended
    failed or cancelled and ``ended_without_success`` holds. A root's project is its binding when it has one (owner
    B4=A, ``projects_registry.project_id_for_task``; read strictly: a broken store or row raises): the stored
    ``project_id`` the partial index ``tasks_project_root_ts`` walks is a copy a mid-run "Turn into project" never
    reaches, so a root bound elsewhere is skipped there and the roots bound to an open project are read beside it by
    primary key. Rows Presence's authority governs (``presence_authority_recorded``) are skipped by that typed column.
    Bodies are read only for the candidates, by primary key in the probes' snapshot, and decoded after it closes."""
    from ouroboros import task_store
    from ouroboros.projects_registry import project_task_bindings

    names = open_projects(projects)
    if not names:
        return []
    bindings = project_task_bindings(root, strict=True)
    bound = {task_id: str(row.get("project_id") or "") for task_id, row in bindings.items()}
    moved = sorted(task_id for task_id, project_id in bound.items() if project_id in names)
    own = f"{task_store.ROOT_RESULT_WHERE} AND presence_authority_recorded=0"
    newest = (f"SELECT task_id, status, ts FROM tasks INDEXED BY tasks_project_root_ts WHERE project_id=? AND {own} "
              f"{task_store._NEWEST_ORDER}")
    best: Dict[str, Tuple[bool, str, str, str]] = {}  # project -> (has a start, start, id, status) of its newest root

    def keep(project_id: str, task_id: Any, status: Any, ts: Any) -> None:
        key = (ts is not None, ts or "", str(task_id), str(status or ""))
        if project_id not in best or key[:3] > best[project_id][:3]:
            best[project_id] = key

    fetched = []
    with task_store.open_store(root, write=False) as txn:
        for project_id in names:
            for task_id, status, ts in txn.conn.execute(newest, (project_id,)):
                if (bound.get(task_id) or project_id) == project_id:  # one bound elsewhere is that project's root
                    keep(project_id, task_id, status, ts)
                    break
        if moved:  # by primary key: the ids lead (CROSS JOIN); the parent index, null for every root, is never walked
            bound_roots = ("SELECT tasks.task_id, status, ts FROM json_each(?) CROSS JOIN tasks "
                           f"ON tasks.task_id = json_each.value WHERE {own}")
            for task_id, status, ts in txn.conn.execute(bound_roots, (json.dumps(moved),)):
                keep(bound[task_id], task_id, status, ts)
        candidates = [(known, ts, task_id, project_id) for project_id, (known, ts, task_id, status) in best.items()
                      if status in FALLEN_STATUSES]
        if candidates:
            fetched = task_store._fetch(txn, "t.task_id IN (SELECT value FROM json_each(?))",
                                        (json.dumps(sorted(task_id for _known, _ts, task_id, _p in candidates)),))
    bodies = dict(task_store._documents(fetched))
    return [FallenWork(task_id, project_id, fallen_line(bodies[task_id], project=names[project_id], now=now))
            for _known, _ts, task_id, project_id in sorted(candidates, reverse=True)
            if ended_without_success(bodies.get(task_id))]


def _fallen_section(facts: StandingFacts) -> Optional[StandingSection]:
    from ouroboros.projects_registry import list_reserved_projects

    if facts.registry is None:
        raise LookupError("project registry unreadable")
    # empty is "no projects" only where the registry proves it (``readable_rooms``' rule), and the rows that read
    # validated are the ones listed: a registry repaired after the wake's lenient read still names its projects
    projects = facts.registry or list_reserved_projects(facts.root, require_readable=True)
    works = fallen_works(facts.root, now=facts.now, projects=projects)
    return StandingSection(FALLEN_KIND, FALLEN_TITLE, FALLEN_WORDS, tuple(work.line for work in works)) if works else None


CLOCK_KIND = "own_clock"
CLOCK_TITLE = "Own clock"
CLOCK_WORDS = ("own clock fact", "own clock facts")


def _clock_section(facts: StandingFacts) -> Optional[StandingSection]:
    """Every line of ``consciousness_clock.own_clock_lines`` while it has a fact of its own (a named time, or my
    human's reported zone: ``has_own_clock_fact``); none for the appointment and the bounds alone. Whether this wake
    came on time stays in the reason line, never repeated here."""
    from ouroboros.consciousness_clock import has_own_clock_fact, own_clock_lines

    clock = facts.clock or {}
    if not has_own_clock_fact(clock):
        return None
    lines = own_clock_lines(clock, now=facts.now, floor=int(clock["floor"]), ceiling=int(clock["ceiling"]))
    return StandingSection(CLOCK_KIND, CLOCK_TITLE, CLOCK_WORDS, tuple(lines))


# The standing sections in their order after the outstanding cards: ``(kind, producer)``, each producer returning its
# section or ``None``.
STANDING_PRODUCERS: Tuple[Tuple[str, Callable[[StandingFacts], Optional[StandingSection]]], ...] = (
    (FALLEN_KIND, _fallen_section),
    (CLOCK_KIND, _clock_section),
)


def standing_sections(facts: StandingFacts, gaps: set) -> Tuple[StandingSection, ...]:
    """Every producer's section, in order; one that cannot be read is the named gap ``<kind> unreadable: <error>``,
    never a missing wake."""
    sections = []
    for kind, produce in STANDING_PRODUCERS:
        try:
            section = produce(facts)
        except Exception as exc:
            gaps.add(f"{kind} unreadable: {type(exc).__name__}")
            continue
        if section is not None and section.lines:
            sections.append(section)
    return tuple(sections)
