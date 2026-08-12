"""One-writer-per-WORKING-FOLDER lease helpers (multi-project, v6.32.0).

Pure functions consumed by ``supervisor/workers.py::assign_tasks`` under the
queue lock: a PENDING task whose LANE is already RUNNING is skipped this
assignment pass (a folder serializes internally; parallelism happens between
lanes and via subagent swarms WITHIN a task).

The lane key is the FOLDER whenever one is known: a task whose folder resolves
keys on ``("", root)``, so every writer in that folder serializes REGARDLESS of
which project or thread it belongs to. Two threads of one project in the same
folder share the key; two different PROJECTS attached to the same folder share
it too — which the earlier ``(project_id, workspace_root)`` key did not deliver,
because folder exclusivity held only within one project while the docstring
claimed "one folder is one writer lane" (T0R2-5). A thread branched off into its
own git worktree names a different root and therefore runs concurrently, which is
the whole point of branching off.

**An absent workspace_root is not a second folder.** Only the promote/room path
stamps ``workspace_root`` on the task record; a task scoped POST-HOC through
:func:`mark_task_project` (the SSOT behind both the supervisor's in-task
``ensure_project_scope`` and the UI's ``api_project_from_task``) carries the
project id and nothing else. Reading the raw field alone would split ONE project
folder into two lanes — ``("", "/w/alpha")`` and ``(pid, "")`` — and let two
top-level writers into it concurrently. So an absent ``workspace_root`` resolves
to the project's REGISTERED ``working_dir`` through a caller-supplied
``project_workspaces`` map (``project_id -> working_dir``); the supervisor reads
that map from the registry once per assignment pass, because this module may not
touch the registry or the filesystem itself. An explicitly workspace-less task
(``workspace="none"``) therefore still serializes against its project's folder —
deliberate; "I write nowhere" is not a claim this module can verify.

Only when NO folder can be determined at all — the project has no registered
``working_dir``, or the caller passed no map — does the lane fall back to
``(project_id, "")``: honestly narrower, serializing within the project alone.
There is no wildcard lane. Under folder-keyed lanes a project id has no folder
lanes to be conservative against (a folder lane does not carry the project it
belongs to), so a "conflicts with everything in this project" key would have had
nothing to match; the resolution map is what closes that hole instead.

**"No project has a folder" and "the folders are unknown" are different facts,
and the narrow key is only honest for the first.** ``project_workspaces={}`` is
an answer: every project is file-less, so keying on the project alone serializes
exactly the tasks that need it. ``project_workspaces=None`` is the ABSENCE of an
answer — an unreadable registry — and it reaches here without any exception
being raised, because the parse simply yields no ``projects`` list. Narrowing the
key for that admitted a second writer in BOTH directions: a folder-bearing
candidate compared against a narrow held lane matched nothing and entered the
folder, and a placeless RUNNING holder stopped blocking a folder-bearing
candidate. So a project-scoped candidate whose folder is UNRESOLVABLE
(:func:`_folder_unresolvable`) is not leasable while any lane is held, and while
the map is missing a candidate that names a folder queues behind ANY narrow lane
that is held — not only one belonging to its OWN project. Comparing the
candidate's own project key alone left the hole in the covering branch itself: a
narrow lane says "this running writer's folder is unknown", so a candidate naming
``/w/shared`` under a DIFFERENT project id, and a projectless one naming it,
matched nothing and were admitted, while the same-project candidate beside them
queued. Projects may share a folder, which is why the lane is folder-keyed at
all. An unknown folder queues; it never runs parallel by accident (I3).
Everything else is unchanged: a real map — even an empty one — keeps the honest
narrow key for the genuinely file-less project, and a RESOLVED folder lane never
blocks a candidate writing in a different folder.

**Purity.** These functions run under the supervisor queue lock on every
assignment pass, so they NEVER touch the filesystem: normalization here is
``normpath`` + ``platform_layer.casefold_path`` (``normcase`` plus a ``casefold``
wherever ``PATH_CASE_INSENSITIVE`` holds — the platform layer owns that fact).
SYMLINK resolution happens at RECORD-WRITE time instead —
``workspace_admission.validate_workspace_root`` resolves the path before it is
stored on a task record, and ``projects_registry.create_project``/
``update_project`` resolve ``working_dir`` before storing it. Both carriers
therefore arrive here already realpath'd, and the caller's project->folder map
needs no FS access to build.

A task that names NEITHER a project NOR a folder has no lane: ordinary unscoped
tasks never serialize against each other. But naming a FOLDER is enough on its
own — a projectless task carrying a ``workspace_root`` writes in that folder, so
it occupies that folder's lane (:func:`_is_lane_occupant`). Requiring a
``project_id`` made ``reserved_folder_lane`` half a reservation: with the folder
held by a merge-back or a removal, a project-scoped candidate was refused and a
PROJECTLESS candidate naming the SAME folder was admitted into it. Subagents carry
their parent's stored ``project_id`` and workspace root but hold no lease of their
own — the parent task IS the folder's writer and its swarm must not deadlock
against itself, so only top-level (non-subagent) tasks count as lane occupants.

A task's lane is PINNED onto its record when it enters RUNNING
(:func:`pin_task_lane`) and read back from there afterwards. Deriving it on
demand meant a mid-run mutation of the task record — the post-hoc project
conversion is the live one — silently moved a running writer to a different
lane: it released the lane it actually held and admitted a second writer into
the same folder (T0R2-7). The pin is WRITE-ONCE: acquiring a lane a task never
had is a deliberate act, drifting out of one it holds is not. The pin MUST be
taken with the same ``project_workspaces`` map the admission check used, or the
task freezes a lane the scheduler never compared it against and the folder reads
as free to the very next candidate.

Only OCCUPANCY reads the pin. A pending CANDIDATE holds nothing, so it is always
compared by what its fields describe — which is also why a crash retry that
carries a stale pin back into the queue can never be mis-compared.

And the pin is STRIPPED on the way back into PENDING, in the enqueue SSOT
(``supervisor.queue.enqueue_task``), so every requeue path is covered at once. A
task in PENDING holds no lane, so a pin from a previous attempt is not a claim —
it is a stale fact about a run that is over. The in-process crash retry
(``ensure_workers_healthy``) re-enqueues the very dict ``assign_tasks`` stamped,
and ``enqueue_task``'s field-stripping allowlist did not include the pin: the next
assignment pass found one, the write-once pin returned it unchanged, and attempt 2
held a lane it does not write in. With the folder attached between attempts it
froze ``(pid, "")`` while writing the registered folder, and the very next
candidate for that folder read it as free (I4). The queue-SNAPSHOT path was
already safe — ``persist_queue_snapshot`` is an allowlist that omits the pin —
which is why only the in-process retry carried it.

``running_project_ids`` remains as the project-WIDE ACTIVITY query (is anything
happening anywhere in this project?), which merge/remove preconditions need. It
is deliberately NOT the lease key — do not reintroduce it as one — and it counts
what the lane deliberately ignores: subagents, and queued work.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Set, Tuple

from ouroboros.platform_layer import casefold_path

# A lane key. Either ("", normalized workspace_root) — a FOLDER lane, held by
# every writer in that folder — or (project_id, "") for a project-scoped task
# whose folder could not be determined at all.
LaneKey = Tuple[str, str]

#: The task field carrying the lane pinned at the RUNNING transition. Stored as
#: a plain list so it survives the JSON queue snapshot unchanged.
LANE_PIN_FIELD = "_lane_key"

# Whether a path comparison here may ignore case is a PLATFORM fact, so it is
# stated in the platform layer (`platform_layer.PATH_CASE_INSENSITIVE`, consumed
# through `casefold_path`) and NOT by reading `sys.platform` against a tuple in
# this module. `os.path.normcase` lowercases on win32 only, so on macOS
# `/Users/x/Repo` and `/Users/x/repo` — the SAME folder — produced two lanes and
# admitted two writers onto it, which is precisely what the lane exists to
# prevent (T0R2-4).


def _as_task(item: Any) -> Any:
    """Unwrap the supervisor RUNNING meta shape ({"task": {...}, ...}) to the
    task dict; pass a bare task dict through unchanged."""
    if isinstance(item, dict) and isinstance(item.get("task"), dict):
        return item["task"]
    return item


def _task_project_id(task: Any) -> str:
    task = _as_task(task)
    if not isinstance(task, dict):
        return ""
    return str(task.get("project_id") or "").strip()


def _task_workspace_root(task: Any) -> str:
    """The folder a task RECORD claims, normalized — ``""`` when it claims none.

    Read from the task record first and then from its ``metadata`` mirror —
    both carriers exist in the queue (``_is_workspace_task_record`` reads the
    same pair). This is the raw claim; :func:`_computed_lane` is what turns an
    absent claim into the project's registered folder.
    """
    task = _as_task(task)
    if not isinstance(task, dict):
        return ""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return normalize_workspace_root(
        task.get("workspace_root") or metadata.get("workspace_root") or ""
    )


def normalize_workspace_root(raw: Any) -> str:
    """The comparison form of a folder path — the ONE spelling a lane key uses.

    PURE (``normpath`` plus the platform layer's ``casefold_path``): this runs
    under the queue lock on every assignment pass, so it must never touch the
    filesystem to resolve symlinks or ask the OS what a path really is. Public
    because anything comparing a folder to the lane set — a merge-back holding a
    folder, an activity query asking whether something is writing in one — has to
    spell it the same way or the comparison is theatre.

    The case question is asked of ``platform_layer``, never of ``sys.platform``
    here: which platforms fold case is a cross-platform FACT and belongs in the
    one module that owns those facts.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    return casefold_path(os.path.normpath(text))


def _computed_lane(
    task: Any, project_workspaces: Optional[Mapping[str, str]] = None
) -> LaneKey:
    """The lane a task's CURRENT fields describe (before any pin is consulted).

    ``project_workspaces`` maps ``project_id -> registered working_dir`` and is
    supplied by the CALLER (the supervisor reads it from the registry; this
    module stays FS-free). A task whose record carries no ``workspace_root``
    resolves to its project's folder through that map, so a post-hoc-scoped task
    takes the same FOLDER lane as the room task already writing there. ``None``
    means the map could not be read at all — see :func:`_folder_unresolvable`,
    which is what stops that narrowing from admitting a second writer.
    """
    root = _task_workspace_root(task)
    pid = _task_project_id(task)
    if not root and pid and project_workspaces:
        root = normalize_workspace_root(project_workspaces.get(pid))
    # A known folder is the lane, across projects and threads alike; a task whose
    # folder is unknown can only be serialized against its own project.
    return ("", root) if root else (pid, "")


def _folder_unresolvable(
    task: Any, project_workspaces: Optional[Mapping[str, str]] = None
) -> bool:
    """Is this task's FOLDER unknowable right now, as opposed to absent?

    True only when the task names no folder of its own AND the caller could not
    supply the project->folder map at all (``None``). With a real map — even an
    empty one — a task that names no folder has an ANSWER: its project has no
    registered folder, and ``(project_id, "")`` is the honest lane for it.
    """
    return project_workspaces is None and not _task_workspace_root(task)


def _pinned_lane(task: Any) -> Optional[LaneKey]:
    task = _as_task(task)
    if not isinstance(task, dict):
        return None
    raw = task.get(LANE_PIN_FIELD)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (str(raw[0]), str(raw[1]))
    return None


def pin_task_lane(
    task: Any, project_workspaces: Optional[Mapping[str, str]] = None
) -> Optional[LaneKey]:
    """Freeze a task's lane onto its record at the RUNNING transition (T0R2-7).

    WRITE-ONCE, and only for a task that actually occupies a lane. A running
    writer must never drift out of the lane it holds because its record was
    edited underneath it — that releases the folder while the writer is still in
    it. A task that was NOT a lane occupant and later becomes one (the post-hoc
    project conversion) has nothing to drift out of, so it pins then instead.

    ``project_workspaces`` MUST be the same map the admission check
    (:func:`candidate_is_leasable`) was given. Pinning without it would freeze
    ``(project_id, "")`` for a task the scheduler had just compared as
    ``("", registered_folder)``, so the folder would read as unheld to the very
    next candidate — two writers, admitted by the pin itself.

    A lane that cannot be RESOLVED is not pinned at all
    (:func:`_folder_unresolvable`), and that is the point of the write-once rule
    rather than an exception to it: freezing ``(project_id, "")`` because the
    registry happened to be unreadable at this instant would OUTLIVE the outage.
    Once the map is readable again the candidate check has a real answer and stops
    applying the conservative rule, while the holder still carries the narrow key —
    so a folder-bearing candidate would match nothing and become a second writer.
    Left unpinned, occupancy falls back to what the record describes and resolves
    to the correct folder the moment the registry can be read. Nothing is lost: the
    pin exists to stop a live writer DRIFTING out of a lane it holds, and a task
    whose lane was never determined has no such lane.

    The caller MUST hold the queue lock. Returns the pinned key, or ``None`` when
    the task holds no lane, or when its lane cannot be resolved.
    """
    task = _as_task(task)
    if not isinstance(task, dict):
        return None
    existing = _pinned_lane(task)
    if existing is not None:
        return existing
    if not _is_lane_occupant(task):
        return None
    if _folder_unresolvable(task, project_workspaces):
        return None
    lane = _computed_lane(task, project_workspaces)
    task[LANE_PIN_FIELD] = list(lane)
    return lane


def _held_lane(
    task: Any, project_workspaces: Optional[Mapping[str, str]] = None
) -> LaneKey:
    """The lane a RUNNING task HOLDS: its pin if it has one, else the computed key.

    Only occupancy reads the pin. A CANDIDATE is by definition not running and
    holds nothing, so it is always compared by what it describes
    (:func:`_computed_lane`) — which is also why a crash retry or a snapshot
    restore carrying a stale pin back into PENDING cannot mis-compare it.
    """
    return _pinned_lane(task) or _computed_lane(task, project_workspaces)


def _is_lane_occupant(task: Any) -> bool:
    """Top-level tasks that NAME A FOLDER or a project occupy a lane; subagents do not.

    Occupancy used to require a ``project_id``, which contradicted the folder-keyed
    lane this module now describes: a task carrying a ``workspace_root`` and NO
    project id held no lane at all, so with ``reserved_folder_lane`` HELD over a
    folder — a merge-back rewriting it, a removal deleting it — a project-scoped
    candidate for that folder was correctly refused while a PROJECTLESS candidate
    naming the same folder was admitted straight into it. The reservation was
    therefore only half a reservation, and the hole was in the one place a
    reservation exists to cover.

    ``running_workspace_roots``' own docstring already named this ("a task carrying
    a ``workspace_root`` with NO ``project_id`` holds no lane at all yet still
    writes there"), and the promote path derives a project id specifically so the
    folder lane would bind — belt-and-braces now rather than the only defence. The
    DIRECT task API needs no such derivation to be safe.

    Subagents stay exempt, and that check stays FIRST: a swarm carries its parent's
    workspace root, and the parent task IS the folder's writer, so counting a
    subagent would deadlock a task against its own children.
    """
    task = _as_task(task)
    if not isinstance(task, dict):
        return False
    if str(task.get("delegation_role") or "") == "subagent":
        return False
    return bool(_task_project_id(task) or _task_workspace_root(task))


#: Folder lanes held by something that is NOT a task — today, a merge-back
#: rewriting the project folder. Refcounted so nested/concurrent holders cannot
#: release each other's claim, and in-process only: a restart clears it, which is
#: correct, because the operation holding it did not survive either.
_RESERVED_LANES: Dict[LaneKey, int] = {}
_RESERVED_LOCK = threading.Lock()

#: The reservations THIS thread is holding, so an operation can ask whether a
#: folder is held by anyone ELSE. Without it the holder's own precondition check
#: sees its own claim and refuses itself: merge-back takes the reservation and
#: then asks ``project_is_busy``, which after I5 consults this set.
_OWN_RESERVED = threading.local()


def _own_reserved() -> Dict[LaneKey, int]:
    counts = getattr(_OWN_RESERVED, "counts", None)
    if counts is None:
        counts = {}
        _OWN_RESERVED.counts = counts
    return counts


@contextlib.contextmanager
def reserved_folder_lane(workspace_root: Any) -> Iterator[Optional[LaneKey]]:
    """Hold a FOLDER's writer lane for the duration of a non-task operation.

    Merge-back rewrites the project folder and held nothing: its
    ``project_is_busy`` check is a bare read, so a task arriving one instant later
    was admitted into the folder mid-merge — the exact two-writer state the lane
    exists to prevent, reached through the gap between a check and the work it was
    checking for. A reservation closes the gap, and it is released in a ``finally``
    so a failed merge cannot strand a folder nobody can schedule into.

    Yields the lane key it took, or ``None`` when there is no folder to hold.
    """
    key: LaneKey = ("", normalize_workspace_root(workspace_root))
    if not key[1]:
        yield None
        return
    with _RESERVED_LOCK:
        _RESERVED_LANES[key] = _RESERVED_LANES.get(key, 0) + 1
    own = _own_reserved()
    own[key] = own.get(key, 0) + 1
    try:
        yield key
    finally:
        mine = own.get(key, 0) - 1
        if mine > 0:
            own[key] = mine
        else:
            own.pop(key, None)
        with _RESERVED_LOCK:
            remaining = _RESERVED_LANES.get(key, 0) - 1
            if remaining > 0:
                _RESERVED_LANES[key] = remaining
            else:
                _RESERVED_LANES.pop(key, None)


def reserved_folder_lanes(*, include_own: bool = True) -> Set[LaneKey]:
    """Folder lanes currently reserved by a non-task operation.

    ``include_own=False`` drops the reservations THIS thread is holding, which is
    what an operation's own precondition needs: a merge-back takes the folder and
    then asks whether the folder is busy, and it must not be refused by its own
    claim. Two genuine concurrent holders each see the other's and both stop,
    which is the fail-closed direction and the one the lane exists for.
    """
    with _RESERVED_LOCK:
        counts = dict(_RESERVED_LANES)
    if include_own:
        return set(counts)
    own = _own_reserved()
    return {key for key, held in counts.items() if held > own.get(key, 0)}


def running_project_lanes(
    running: Iterable[Any], project_workspaces: Optional[Mapping[str, str]] = None
) -> Set[LaneKey]:
    """Writer lanes currently held: ``{("", workspace_root) | (project_id, ""), ...}``.

    ``running`` is the supervisor's RUNNING mapping values (or any iterable of
    task dicts); read under the queue lock by the caller.
    ``project_workspaces`` (``project_id -> registered working_dir``) resolves
    tasks whose record carries no workspace of its own — see
    :func:`_computed_lane`. Lanes RESERVED by a non-task operation are unioned in
    — a merge-back holds the project folder for as long as it is rewriting it,
    and to the scheduler that is the same fact as a task holding it.
    """
    out: Set[LaneKey] = set(reserved_folder_lanes())
    for task in running or ():
        if _is_lane_occupant(task):
            out.add(_held_lane(task, project_workspaces))
    return out


def running_workspace_roots(running: Iterable[Any], pending: Iterable[Any] = ()) -> Set[str]:
    """Every FOLDER a live task names — the folder half of the activity query.

    Sibling of :func:`running_project_ids`, and needed for the same reason the
    lane key stopped being ``(project_id, workspace_root)``: "is anything writing
    in this folder" is not answerable from project ids. Project *alpha* merging
    into a folder while project *beta*'s task writes in it reduced to two
    different ids and read as idle, and a task carrying a ``workspace_root`` with
    NO ``project_id`` holds no lane at all yet still writes there.

    Reads the RAW claim only, deliberately: a task that named no folder is
    covered by :func:`running_project_ids`, which every caller of this function
    also consults, and resolving it here would need the registry map this
    function's callers (gateway-side, not scheduler-side) do not hold.

    Same activity semantics as :func:`running_project_ids` — subagents count,
    startable pending work counts, work parked for the owner does not — and
    deliberately NOT filtered by lane occupancy: occupancy is a scheduling rule
    about who may be ASSIGNED, and this asks who is actually in the folder.
    """
    startable = [task for task in (pending or ()) if _can_still_start(task)]
    roots = {_task_workspace_root(task) for task in (*(running or ()), *startable)}
    return {root for root in roots if root}


def running_project_ids(running: Iterable[Any], pending: Iterable[Any] = ()) -> Set[str]:
    """Project ids with ANY task alive in them — the project-WIDE ACTIVITY query.

    NOT the lease key (that is :func:`running_project_lanes`). This answers "is
    anything happening anywhere in this project?", which is the precondition a
    merge-back or a worktree removal needs: those touch the project as a whole,
    not one folder.

    Two deliberate differences from the lane query, both because the question is
    different (T3R-14):

    * SUBAGENTS COUNT here. The lane exempts them so a swarm cannot deadlock
      against its own parent — a SCHEDULING rule, about who may be assigned. A
      subagent still runs commands and writes files in the project, so for "is
      anything happening here" it is as real as any other task, and exempting it
      let a merge rewrite a folder while a swarm member was mid-write.
    * PENDING COUNTS, stated plainly rather than left to be inferred: a queued
      task for this project can be assigned at ANY instant, including the one
      right after this returns, and the callers hold no lock against the
      scheduler. Counting it costs the owner a wait; not counting it costs a
      folder rewritten under a task that has just started. ``pending`` is a
      separate argument because the supervisor keeps the two collections apart
      and a caller that can only see one must still get a true answer about it.

      EXCEPT a pending task that cannot start on its own. A budget-exhausted task
      is parked in PENDING with ``auto_resume: False`` and waits for the owner —
      possibly forever, and across a queue-snapshot restore. Counting those made
      "this project is busy" permanently true, so a single paused task locked the
      owner out of merging their own work back, with nothing on screen to explain
      why. Work only the owner can release is not activity to wait behind; it is
      the owner's own decision, already taken.
    """
    out: Set[str] = set()
    startable = [task for task in (pending or ()) if _can_still_start(task)]
    for task in (*(running or ()), *startable):
        pid = _task_project_id(task)
        if pid:
            out.add(pid)
    return out


def _can_still_start(task: Any) -> bool:
    """Could the SCHEDULER still pick this pending task up without the owner?

    Parked-for-the-owner is not queued: a budget-exhausted task carries a
    ``_budget_pause`` whose ``auto_resume`` is false and will not move again until
    the owner resumes or cancels it.
    """
    unwrapped = _as_task(task)
    if not isinstance(unwrapped, dict):
        return True
    pause = unwrapped.get("_budget_pause")
    return not (isinstance(pause, dict) and not pause.get("auto_resume"))


def candidate_is_leasable(
    candidate: Dict[str, Any],
    running_lanes: Set[LaneKey],
    project_workspaces: Optional[Mapping[str, str]] = None,
) -> bool:
    """True when ``candidate`` may be assigned now under the one-writer rule.

    ``running_lanes`` MUST come from :func:`running_project_lanes`. A set of
    bare project ids would never match a lane tuple, so every candidate would
    read as leasable and TWO writers could enter one folder — a silent
    data-corruption path. Misuse raises instead, and the shape check runs
    BEFORE the unscoped-candidate short-circuit: otherwise a caller passing
    bare project ids would be told nothing at all as long as the first
    candidates happened to be unscoped, and would learn about it only when a
    project task finally slipped through.

    ``project_workspaces`` must be the SAME map given to
    :func:`running_project_lanes`, or the two sides spell the same folder
    differently and the comparison decides nothing.

    ``project_workspaces=None`` means the folders could not be READ (an
    unreadable registry), which is not the same as "no project has one" and must
    not narrow the key: see the module docstring. While the map is missing, a
    project-scoped candidate that names no folder is not leasable at all if
    anything holds a lane, and one that DOES name a folder queues behind ANY
    narrow lane that is held — not merely a narrow lane of its own project.
    "Cannot tell" queues; it never runs in parallel by accident.

    That last rule used to compare only the candidate's OWN project key, which
    left the hole in the one place the fail-closed branch exists to cover. A
    narrow ``(project_id, "")`` lane means "this running task's folder is
    unknown", and the answer to "is it the folder this candidate names?" is
    exactly as unknown: the holder's registered ``working_dir`` may BE that
    folder, and two projects may even share one (the lane is folder-keyed for
    precisely that reason). So a candidate naming ``/w/shared`` under another
    project id, and a PROJECTLESS candidate naming it, both matched nothing and
    were admitted straight into a folder a live writer may hold — while the
    same-project candidate beside them was correctly queued. Under an unreadable
    map every unresolved project lane conflicts with every folder-bearing
    candidate, because nothing available here can prove they are disjoint.
    """
    for lane in running_lanes or ():
        if not (isinstance(lane, tuple) and len(lane) == 2):
            raise TypeError(
                "candidate_is_leasable expects lane keys from running_project_lanes "
                f"((project_id, workspace_root) tuples), got {lane!r}"
            )
    if not _is_lane_occupant(candidate):
        return True
    if project_workspaces is None:
        if _folder_unresolvable(candidate, project_workspaces):
            return not bool(running_lanes)
        # The candidate names a folder; a NARROW held lane names a project whose
        # folder this module cannot read. Disjointness is unprovable in both
        # directions, so it queues — whatever project the narrow lane belongs to.
        if any(pid for pid, _root in (running_lanes or ())):
            return False
    return _computed_lane(candidate, project_workspaces) not in running_lanes


def mark_task_project(
    running: Any,
    pending: Any,
    tid: Any,
    pid: Any,
    project_workspaces: Optional[Mapping[str, str]] = None,
) -> bool:
    """Set a task's ``project_id`` wherever it currently lives in the supervisor queue
    state — the live RUNNING map (``{tid: {"task": {...}}}``) AND the PENDING list (bare
    task dicts) — so a POST-HOC project conversion/scope makes it a one-writer lane
    occupant whether it has started yet or not. The lease + assignment read
    ``task['project_id']`` from these IN-MEMORY structures (assign_tasks checks the
    pending candidate's own dict, then copies it into RUNNING), NOT the durable bindings —
    so a converted PENDING task that is only bound durably would still start unscoped and
    miss its lane. This is the SSOT for both post-hoc convert paths — the supervisor
    in-task ``ensure_project_scope`` and the UI ``api_project_from_task`` — so they cannot
    drift apart again. The caller MUST hold the queue lock. Returns True if any in-memory
    task dict was updated; a no-op (False) when the task is neither running nor pending
    (then the durable bind alone is correct — there is no live lane to occupy).

    It deliberately stamps the project id ALONE and never guesses a folder onto the
    record: this module has no filesystem access and the task may not write in the
    project's folder at all. Resolving that task's LANE is :func:`_computed_lane`'s
    job — an absent ``workspace_root`` becomes the project's registered
    ``working_dir`` through ``project_workspaces``.

    A RUNNING task converted here ACQUIRES a lane it did not hold, so its lane is
    pinned on the spot. That is the one case the write-once pin admits: it takes a
    lane rather than drifting out of one, and leaving it unpinned would let a
    registry edit later move a live writer off the folder it is writing in."""
    key = str(tid or "")
    project = str(pid or "").strip()
    if not key or not project:
        return False
    updated = False
    meta = running.get(key) if hasattr(running, "get") else None
    rtask = _as_task(meta) if isinstance(meta, dict) else None
    if isinstance(rtask, dict):
        rtask["project_id"] = project
        pin_task_lane(rtask, project_workspaces)
        updated = True
    for item in (pending or ()):
        ptask = _as_task(item)
        if isinstance(ptask, dict) and str(ptask.get("id") or "") == key:
            ptask["project_id"] = project
            updated = True
    return updated


__all__ = [
    "LANE_PIN_FIELD",
    "LaneKey",
    "candidate_is_leasable",
    "mark_task_project",
    "normalize_workspace_root",
    "pin_task_lane",
    "reserved_folder_lane",
    "reserved_folder_lanes",
    "running_project_ids",
    "running_project_lanes",
    "running_workspace_roots",
]
