"""ONE thread-ancestry lens shared by the UI history and the agent's context.

A forked thread stores only a CURSOR — ``{fork_of_chat_id, fork_before_ts}`` —
and NEVER a copy of its parent's rows. Reading a fork therefore means reading
its own chat PLUS a bounded slice of every ancestor chat. Two independent
readers need that answer: ``gateway/history.py`` (what the owner sees) and
``ouroboros/context.py`` (what the agent sees). A cursor written into only one
of them would hand the UI and the agent DIFFERENT histories of the same thread,
so both consume the lens built here and nothing else.

Semantics pinned by this module:

* **Inclusive boundary.** An ancestor row is in scope when
  ``row_ts <= fork_before_ts``. ``fork_before_ts`` is stamped at the fork
  moment, so a row bearing exactly that timestamp existed BEFORE the fork.
  Comparison is lexicographic over ISO-8601 UTC strings — the same convention
  the history window already uses for its recency floor. A row with no
  timestamp sorts as oldest and is therefore admitted.
* **Intersected cutoffs for a fork of a fork.** Following the chain, each
  ancestor's effective cutoff is the MOST RESTRICTIVE bound on the path to it:
  a grandchild can never see more of a grandparent than its own parent could.
* **Lifecycle-blind ancestry.** Ancestors resolve whether they are active,
  archived, deleting or tombstoned. Filtering the chain by liveness would
  silently orphan every fork of a deleted thread.
* **Same-project ancestors only.** A cursor is followed ONLY to a chat that
  ``resolve_chat_binding`` recognises as a thread of the SAME project as the
  chat being read. A hand-written ``fork_of_chat_id: 1`` (or any unbound chat)
  would otherwise pour the WHOLE Main conversation into a project thread's
  history and into the agent's focused context; a cursor naming another
  project's thread would pour THAT project's conversation across the boundary
  just as silently, on both surfaces. The foreign ancestor is refused BEFORE it
  enters the cutoffs, and the refusal is disclosed.
* **The requesting chat's own present is never bounded.** A cycle that closes
  back on the chat being read (a self-parent, or A→B→A) must not tighten that
  chat's own cutoff: a thread would start rejecting the messages it just sent.
  The cycle is disclosed instead.
* **Bounded and disclosed.** The walk stops at ``MAX_ANCESTRY_DEPTH``, on a
  cycle, or at an unbound ancestor and sets ``truncated`` — the caller
  discloses it (``/api/chat/history`` reports it as the ``ancestry_depth``
  window cause) rather than quietly serving a short history.
* **Unreadable is its own state.** "The registry could not be read" is neither
  "this chat is genuinely Main" nor "the walk lost an ancestor it could see". It
  sets ``lens_unavailable`` (and ``truncated`` with it, so every existing
  disclosure consumer already reacts), reported as the ``lens_unavailable``
  window cause. Folding it into the ancestor-less answer is how a fork lost its
  whole shared past while the window still called itself complete.

The module also owns the ONE row-classification pair both readers share —
:func:`bound_chat_for_row` (a post-hoc bound task's owning project chat, by task
LINEAGE) and :func:`admits_row`. Keeping the lens but re-implementing "does this
row belong to the thread" per caller is exactly how the UI and the agent drifted
apart before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# A fork chain deeper than this is pathological (each link is an owner action).
# Hitting it is disclosed, never silently truncated.
MAX_ANCESTRY_DEPTH = 32


def _min_cutoff(left: str, right: str) -> str:
    """Intersect two cutoffs; ``""`` means "unbounded" and always loses."""
    if not left:
        return right
    if not right:
        return left
    return left if left <= right else right


@dataclass(frozen=True)
class ThreadLens:
    """Every chat one thread reads, with each chat's effective cutoff."""

    chat_id: int
    project_id: str = ""
    thread_id: int = 0
    # chat_id -> "" (the whole chat) or the INCLUSIVE ts upper bound.
    cutoffs: Dict[int, str] = field(default_factory=dict)
    # Self first, then ancestors nearest-first (render/disclosure order).
    order: List[int] = field(default_factory=list)
    # Binding-held canonical owner-row refs, bucketed by the LENS chat that owns
    # them: a converted project's start message lives in Main, so a fork of that
    # project's thread would lose it without carrying the ancestor's refs too.
    source_refs: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    truncated: bool = False
    #: The lens could not be BUILT — the registry could not be read at all, so
    #: whether this chat has ancestors is unknown. A THIRD state, distinct from
    #: both "this is genuinely Main / an external transport" (where an
    #: ancestor-less lens with ``truncated=False`` is the correct answer) and
    #: "the walk lost an ancestor it could see" (``truncated`` alone).
    #:
    #: Collapsing the first two is how a fork could lose its ENTIRE ancestry and
    #: still be reported as a complete window: ``_chat_binding`` swallowed the
    #: registry failure to ``{}``, the empty binding took the degenerate
    #: own-thread early return, and no exception was ever raised for the outer
    #: handlers to catch. Whenever this is true, ``truncated`` is true as well, so
    #: every existing disclosure consumer already reacts; this only says WHY.
    lens_unavailable: bool = False

    @property
    def chat_ids(self) -> set:
        return set(self.cutoffs)

    @property
    def is_project_thread(self) -> bool:
        return bool(self.project_id)

    @property
    def has_ancestors(self) -> bool:
        return len(self.cutoffs) > 1

    def admits(self, entry_chat: Any, ts: Any = "") -> bool:
        """True when a row of ``entry_chat`` stamped ``ts`` belongs to this thread."""
        try:
            chat = int(entry_chat or 0)
        except (TypeError, ValueError):
            return False
        if chat not in self.cutoffs:
            return False
        cutoff = self.cutoffs[chat]
        return not cutoff or str(ts or "") <= cutoff

    def admits_source_ref(self, entry: Dict[str, Any]) -> bool:
        """True when ``entry`` IS a canonical owner row referenced by a binding
        of this thread or of an in-scope ancestor (and is within that
        ancestor's cutoff)."""
        if not self.source_refs or not isinstance(entry, dict):
            return False
        try:
            from ouroboros.project_dialogue import entry_matches_source_ref
        except Exception:  # pragma: no cover - import guard
            return False
        ts = str(entry.get("ts") or "")
        for owner_chat, refs in self.source_refs.items():
            cutoff = self.cutoffs.get(owner_chat, "")
            if cutoff and ts > cutoff:
                continue
            try:
                if entry_matches_source_ref(entry, refs):
                    return True
            except Exception:
                log.debug("Thread source-ref classification failed", exc_info=True)
        return False


def _entry_chat(entry: Any) -> int:
    """Best-effort ``chat_id`` of a chat.jsonl row.

    A missing/blank id reads as the MAIN chat (1), the same default
    ``gateway/history.py`` has always applied when it decoded the raw row — the
    two readers must not disagree about what an unstamped row belongs to.
    """
    try:
        return int((entry or {}).get("chat_id", 1) or 1)
    except (TypeError, ValueError, AttributeError):
        return 1


def bound_chat_for_row(entry: Any, bindings_by_task: Dict[str, int]) -> int:
    """The project chat a post-hoc bound task's row belongs to, by LINEAGE.

    A task converted into a project AFTER it started keeps its original (main)
    ``chat_id`` on every row, so the durable binding is the only truth about
    ownership — and a subagent's rows carry only the ROOT's binding, hence the
    own -> parent -> root walk (same semantics as
    ``projects_registry.project_chat_for_task_tree``, served from ONE preloaded
    bindings map so no per-row file read happens).

    Both readers MUST use this: resolving the binding differently on the two
    surfaces is how the agent's context and the owner's history disagreed about
    a bound task's rows in the first place.
    """
    if not isinstance(entry, dict) or not bindings_by_task:
        return 0
    for field in ("task_id", "parent_task_id", "root_task_id"):
        tid = str(entry.get(field) or "").strip()
        if tid and bindings_by_task.get(tid):
            try:
                return int(bindings_by_task[tid])
            except (TypeError, ValueError):
                return 0
    return 0


def admits_row(lens: "ThreadLens", entry: Any, bound_chat: int = 0) -> bool:
    """Whether one chat.jsonl row belongs to the thread ``lens`` describes.

    THE shared predicate: a row is in scope when its own chat is admitted
    within its cutoff, when the project chat its task is BOUND to is admitted
    (post-hoc conversion — the row keeps a main ``chat_id``), or when the row IS
    a canonical owner row an in-scope binding references. ``bound_chat`` comes
    from :func:`bound_chat_for_row`.
    """
    ts = str((entry or {}).get("ts") or "") if isinstance(entry, dict) else ""
    if bound_chat and lens.admits(bound_chat, ts):
        return True
    if isinstance(entry, dict) and lens.admits_source_ref(entry):
        return True
    return lens.admits(_entry_chat(entry), ts)


def _source_refs_by_chat(drive_root: Any, chats: set) -> Dict[int, List[Dict[str, Any]]]:
    """Bucket binding-held source refs by project chat, in ONE bindings read.

    ``project_dialogue.source_refs_for_project`` re-reads the bindings file per
    chat; a fork chain would pay that once per ancestor.
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    if not chats:
        return out
    try:
        from ouroboros.projects_registry import project_task_bindings

        for row in project_task_bindings(drive_root).values():
            ref = row.get("source_ref")
            if not isinstance(ref, dict) or not ref:
                continue
            try:
                owner = int(row.get("project_chat_id") or 0)
            except (TypeError, ValueError):
                continue
            if owner in chats:
                out.setdefault(owner, []).append(dict(ref))
    except Exception:
        log.debug("Failed to bucket project source refs", exc_info=True)
    return out


def thread_ancestry_lens(
    drive_root: Any,
    chat_id: Any,
    *,
    with_source_refs: bool = True,
) -> ThreadLens:
    """Build the lens for ``chat_id`` (see the module docstring for semantics).

    A non-project chat (Main, an external transport) yields a degenerate lens
    over itself alone, so callers can use one code path.

    "Could not READ the binding" is a THIRD state and never wears that answer's
    clothes. ``_chat_binding`` fails closed to ``None`` (as opposed to ``{}``,
    which means the registry answered and this chat has no binding), and a
    ``None`` for the REQUESTED chat produces the same degenerate lens with
    ``truncated`` AND ``lens_unavailable`` set. Without that distinction a fork
    whose registry had just become unreadable lost its whole ancestry with
    ``truncated=False``, so ``_window_metadata`` answered ``complete: True`` and
    the agent's context silently narrowed to its own rows — no exception, no
    disclosure, BIBLE P1's no-silent-truncation broken on both surfaces.
    """
    try:
        cid = int(chat_id or 0)
    except (TypeError, ValueError):
        cid = 0
    binding = _chat_binding(drive_root, cid)
    if not binding:
        return ThreadLens(
            chat_id=cid,
            cutoffs={cid: ""} if cid else {},
            order=[cid] if cid else [],
            truncated=binding is None,
            lens_unavailable=binding is None,
        )

    cutoffs: Dict[int, str] = {cid: ""}
    order: List[int] = [cid]
    own_project = str(binding.get("project_id") or "")
    truncated = False
    unavailable = False
    current: Optional[Dict[str, Any]] = binding
    effective = ""
    depth = 0
    while current is not None:
        thread = _thread_row(current)
        parent_chat, fork_before = _fork_cursor(thread)
        if not parent_chat or not fork_before:
            break
        depth += 1
        if depth > MAX_ANCESTRY_DEPTH:
            truncated = True
            log.warning(
                "Thread ancestry for chat %s exceeds depth %s — older ancestors omitted",
                cid, MAX_ANCESTRY_DEPTH,
            )
            break
        # Intersection: a descendant can never see more of an ancestor than the
        # link it inherited the view through.
        effective = _min_cutoff(effective, fork_before)
        if parent_chat == cid:
            # The cycle closes on the chat being READ. Its own rows are its own
            # present — bounding them would make the thread reject the messages
            # it just sent (and the agent working in it lose its newest turn).
            # Leave the requesting chat unbounded and disclose the cycle.
            truncated = True
            log.warning(
                "Thread ancestry cycles back to the requesting chat %s — walk "
                "stopped; its own cutoff is left open",
                cid,
            )
            break
        if parent_chat in cutoffs:
            # A cycle among ANCESTORS (only reachable through hand-edited
            # state): tighten the existing bound and stop rather than loop.
            cutoffs[parent_chat] = _min_cutoff(cutoffs[parent_chat], effective)
            truncated = True
            log.warning("Thread ancestry cycle at chat %s — walk stopped", parent_chat)
            break
        # Lifecycle-blind by construction: _chat_binding answers for
        # deleting/tombstoned rows too, so a fork of a deleted thread keeps
        # reading its shared past (A3a). But an ancestor with NO binding at all
        # is the Main chat or an external transport, and an ancestor bound to
        # ANOTHER project is just as foreign — admitting either would pour a
        # whole foreign conversation into this thread. Refuse BEFORE it enters
        # the cutoffs, and disclose the refusal.
        parent_binding = _chat_binding(drive_root, parent_chat)
        if parent_binding is None:
            # The registry could not be READ for this ancestor. Distinct from the
            # ancestor genuinely having no binding: one is a refusal, the other is
            # ignorance, and only the second may be recoverable later.
            truncated = True
            unavailable = True
            log.warning(
                "Thread ancestry of chat %s could not read the binding of its "
                "parent chat %s — ancestry incomplete",
                cid, parent_chat,
            )
            break
        if not parent_binding:
            truncated = True
            log.warning(
                "Thread ancestry of chat %s names chat %s as a parent, but that "
                "chat has no project binding — ancestor refused",
                cid, parent_chat,
            )
            break
        if str(parent_binding.get("project_id") or "") != own_project:
            truncated = True
            log.warning(
                "Thread ancestry of chat %s (project %s) names chat %s as a "
                "parent, but that chat belongs to project %s — ancestor refused",
                cid, own_project, parent_chat,
                str(parent_binding.get("project_id") or ""),
            )
            break
        cutoffs[parent_chat] = effective
        order.append(parent_chat)
        current = parent_binding

    source_refs = (
        _source_refs_by_chat(drive_root, set(cutoffs)) if with_source_refs else {}
    )
    return ThreadLens(
        chat_id=cid,
        project_id=str(binding.get("project_id") or ""),
        thread_id=int(binding.get("thread_id") or 0),
        cutoffs=cutoffs,
        order=order,
        source_refs=source_refs,
        truncated=truncated,
        lens_unavailable=unavailable,
    )


def _chat_binding(drive_root: Any, chat_id: int) -> Optional[Dict[str, Any]]:
    """``resolve_chat_binding`` with the walk's fail-closed error handling.

    THREE answers, and the third is the point: a binding dict, ``{}`` when the
    registry answered and this chat owns no thread (Main, an external transport),
    and ``None`` when the registry could not be READ at all. Returning ``{}`` for
    the failure made an unreadable registry indistinguishable from a non-project
    chat, so a fork silently became ancestor-less with nothing marked truncated.

    ``strict=True`` is what makes the third answer reachable: ``resolve_chat_binding``
    fails closed to ``{}`` on its own account, which is right for ROUTING (an
    unplaceable message belongs in Main) and wrong here, so this asks the same seam
    for the honest error instead of building a second lookup that could drift.
    """
    if not chat_id:
        return {}
    try:
        from ouroboros.projects_registry import resolve_chat_binding

        return resolve_chat_binding(drive_root, chat_id, strict=True) or {}
    except Exception:
        log.warning(
            "thread_ancestry_lens could not read the binding for chat %s — the "
            "lens is degraded and will be disclosed as unavailable",
            chat_id, exc_info=True,
        )
        return None


def _thread_row(binding: Dict[str, Any]) -> Dict[str, Any]:
    """The stored thread row behind a binding (``{}`` for thread #0)."""
    project = binding.get("project")
    if not isinstance(project, dict):
        return {}
    try:
        want = int(binding.get("thread_id") or 0)
    except (TypeError, ValueError):
        return {}
    for row in project.get("threads") or ():
        if isinstance(row, dict):
            try:
                if int(row.get("id")) == want:
                    return row
            except (TypeError, ValueError):
                continue
    return {}


def _fork_cursor(thread: Dict[str, Any]) -> tuple:
    try:
        parent = int(thread.get("fork_of_chat_id") or 0)
    except (TypeError, ValueError):
        parent = 0
    return parent, str(thread.get("fork_before_ts") or "")


__all__ = [
    "MAX_ANCESTRY_DEPTH",
    "ThreadLens",
    "admits_row",
    "bound_chat_for_row",
    "thread_ancestry_lens",
]
