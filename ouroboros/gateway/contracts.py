"""Descriptive HTTP + WebSocket Gateway Boundary contracts (v1).

TypedDicts document payloads, not runtime validation. Keep discriminating
``type`` keys required; mark genuinely optional fields with ``NotRequired``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    from typing import Literal, NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, Required, TypedDict  # type: ignore[assignment]


class ChatAttachmentInbound(TypedDict, total=False):
    """One uploaded chat attachment reference (file already stored by
    /api/chat/upload under data/uploads/; ``filename`` is the stored
    basename). Image attachments are delivered to vision models as native
    image blocks (v6.26.0)."""

    filename: str
    display_name: str
    mime: str


class ChatInbound(TypedDict):
    """Inbound WS chat message. ``type`` and ``content`` are required."""

    type: Literal["chat"]
    content: str
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    force_plan: NotRequired[bool]
    attachments: NotRequired[list]  # list[ChatAttachmentInbound] (additive, v6.26.0)
    # Multi-project (additive, v6.32.0): per-project chat routing. The owner
    # stays user_id 1; chat_id selects the thread, project_id scopes memory.
    chat_id: NotRequired[int]
    project_id: NotRequired[str]


class TaskConstraintInbound(TypedDict, total=False):
    mode: str
    skill_name: str
    payload_root: str
    allow_enable: bool
    allow_review: bool
    extra_allowlist: list[str]


class CommandInbound(TypedDict):
    """Inbound WS command message."""

    type: Literal["command"]
    cmd: str


class ExtensionInbound(TypedDict, total=False):
    """Inbound extension-owned WS message.

    The concrete ``type`` value is provider-safe and namespaced as
    ``ext_<len>_<token>_<message>`` by ``extension_loader``.
    """

    type: str
    data: Any


class TransportMetadata(TypedDict, total=False):
    """Generic external transport provenance for bridge skills."""

    kind: str
    conversation_id: str
    sender_label: str


class ChatOutbound(TypedDict):
    """Outbound WS chat frame."""

    type: Literal["chat"]
    role: Literal["user", "assistant", "system"]
    content: str
    ts: str
    markdown: NotRequired[bool]
    is_progress: NotRequired[bool]
    task_id: NotRequired[str]
    ephemeral_decision: NotRequired[bool]
    task_incident: NotRequired[str]
    toast_once: NotRequired[str]
    lifecycle: NotRequired[Dict[str, Any]]
    subagent_event: NotRequired[str]
    subagent_task_id: NotRequired[str]
    root_task_id: NotRequired[str]
    parent_task_id: NotRequired[str]
    delegation_role: NotRequired[str]
    subagent_role: NotRequired[str]
    accepted: NotRequired[bool]
    active_subagent_count: NotRequired[int]
    max_active_subagents: NotRequired[int]
    queued_behind_active_cap: NotRequired[bool]
    required_capabilities: NotRequired[list[str]]
    write_surface: NotRequired[str]
    model_lane: NotRequired[str]
    requested_model_lane: NotRequired[str]
    effective_model_lane: NotRequired[str]
    # Phase 6: the OPAQUE harness route RESOLVED AT DISPATCH for this
    # bubble/subagent (`resolve_subagent_dispatch`, stamped once) — a delegated
    # route only; absent/empty means the ordinary native path and the UI draws
    # no chip. It is the route the run was sent to, not a receipt from the
    # engine saying where it landed: a landing below the ask is disclosed on
    # `capability_delta`, not by rewriting this field.
    executor_route: NotRequired[str]
    # The completion-seam EVIDENCE the route decision is reconciled against
    # (subagents.envelope_from_task): delegated runs started/settled, disclosed
    # subscription spend, engine-reported models. Terminal frames only; its
    # absence means "no evidence yet", never "ran natively".
    execution_evidence: NotRequired[Dict[str, Any]]
    model: NotRequired[str]
    task_group_id: NotRequired[str]
    task_event: NotRequired[str]
    status: NotRequired[str]
    # v6.82 (P5): host-attested marker — this frame's task is a supervisor-queue
    # task that POST /api/tasks/{id}/cancel can force-cancel (never set for
    # in-process direct-chat turns). Gates the UI "Cancel run" card action.
    cancelable: NotRequired[bool]
    # Monetary projections are nullable when the physical-attempt ledger cannot
    # be read.  ``None`` is deliberately distinct from a confirmed $0 result.
    cost_usd: NotRequired[Optional[float]]
    cost_accounting_status: NotRequired[Literal["available", "unavailable"]]
    cost_accounting_error: NotRequired[str]
    cost_final: NotRequired[bool]
    cost_usd_with_children: NotRequired[Optional[float]]
    cost_with_children_partial: NotRequired[bool]
    reserved_usd: NotRequired[Optional[float]]
    unresolved_upper_bound_usd: NotRequired[Optional[float]]
    unknown_unmetered: NotRequired[Optional[int]]
    # v6.87.48: count of OPEN ledger rows — the disclosed cause of
    # ``cost_final: false``, which can hold with every dollar bucket at zero.
    non_final_rows: NotRequired[Optional[int]]
    result: NotRequired[str]
    result_truncated: NotRequired[bool]  # P3: WS preview was capped; fetch full via task id
    trace_summary: NotRequired[str]
    trace_summary_truncated: NotRequired[bool]  # P3: WS preview capped
    error: NotRequired[str]
    artifact_status: NotRequired[str]
    artifact_bundle: NotRequired[Dict[str, Any]]
    outcome_axes: NotRequired[Dict[str, Any]]
    task_contract: NotRequired[Dict[str, Any]]
    reason_code: NotRequired[str]
    review_status: NotRequired[Dict[str, Any]]
    review_projection: NotRequired[Dict[str, Any]]
    worker_saturation_warning: NotRequired[bool]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    # Deprecated compatibility field: runtime emits ``transport`` instead.
    telegram_chat_id: NotRequired[int]
    # UI-only system annotation emitted by skill-repair visible commands.
    system_type: NotRequired[str]
    # Present on some transport re-broadcast paths.
    chat_id: NotRequired[int]


class PhotoOutbound(TypedDict):
    """Outbound WS photo frame."""

    type: Literal["photo"]
    role: Literal["user", "assistant"]
    image_base64: str
    mime: str
    ts: str
    caption: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    # Deprecated compatibility field: runtime emits ``transport`` instead.
    telegram_chat_id: NotRequired[int]


class VideoOutbound(TypedDict):
    """Outbound WS video frame."""

    type: Literal["video"]
    role: Literal["user", "assistant"]
    video_base64: str
    mime: str
    ts: str
    caption: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    # Deprecated compatibility field: runtime emits ``transport`` instead.
    telegram_chat_id: NotRequired[int]


class DocumentOutbound(TypedDict):
    """Outbound WS document/file frame."""

    type: Literal["document"]
    role: Literal["user", "assistant"]
    file_base64: str
    mime: str
    filename: str
    ts: str
    caption: NotRequired[str]
    # Loopback /api/files/download?path=<root-relative> URL for the durable
    # artifact copy, used by the desktop host-bridge download (WKWebView-safe)
    # and to rebuild the bubble on reload without persisting base64.
    download_url: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    # Deprecated compatibility field: runtime emits ``transport`` instead.
    telegram_chat_id: NotRequired[int]


class TypingOutbound(TypedDict):
    """Outbound WS typing indicator."""

    type: Literal["typing"]
    action: str
    # Multi-project: stamps the thread so the client fan-out routes a project
    # task's typing indicator to its panel instead of defaulting to main.
    chat_id: NotRequired[int]


class LogOutbound(TypedDict):
    """Outbound WS log event."""

    type: Literal["log"]
    data: Dict[str, Any]
    # Multi-project: surfaced at top level so live task progress routes to the
    # owning project panel (and mirrors into main) by thread.
    chat_id: NotRequired[int]


class HeartbeatOutbound(TypedDict):
    """Outbound heartbeat emitted by ``server_runtime.ws_heartbeat_loop``."""

    type: Literal["heartbeat"]
    ts: NotRequired[str]


class ExtensionLifecycleOutbound(TypedDict):
    """Outbound extension lifecycle notification."""

    type: Literal["extension_lifecycle"]
    skill: NotRequired[str]
    action: NotRequired[str]
    status: NotRequired[str]
    reason: NotRequired[str]
    data: NotRequired[Dict[str, Any]]


class ProjectsChangedOutbound(TypedDict):
    """Outbound notice that the project registry changed server-side (e.g. the
    agent's ``promote_chat_to_task`` created/bound a project). The client refreshes
    its project nav + WS-fan-out ``projectChatIds`` on receipt; ``chat_id`` lets it
    learn the new project thread immediately, before the /api/state round-trip."""

    type: Literal["projects_changed"]
    project_id: NotRequired[str]
    chat_id: NotRequired[int]


class MessageAnnotationOutbound(TypedDict):
    """Bubble-free presentation update for one canonical owner message."""

    type: Literal["message_annotation"]
    annotation_type: Literal["routing_ack"]
    client_message_id: str
    action: str
    status: str
    suppress_bubble: bool
    chat_id: NotRequired[int]
    target: NotRequired[str]
    options: NotRequired[List[Dict[str, Any]]]
    ts: NotRequired[str]


class UpdateMergePlan(TypedDict, total=False):
    """Exact, channel-bound merge plan shared by preflight and apply responses."""

    available: bool
    auto_mergeable: bool
    kind: Literal["clean", "conflicting", "current", "unavailable", "unknown"]
    error: str
    remote: str
    remote_branch: str
    target_ref: str
    update_channel: str
    current_branch: str
    base_sha: str
    target_sha: str
    local_dirty_count: int
    local_snapshot: str
    merge_commit: str
    code_conflict_paths: List[str]
    doc_conflict_paths: List[str]
    hot_code_paths: List[str]
    recommended_strategy: Literal["auto_merge", "assisted"]


class UpdatePreflightRequest(TypedDict, total=False):
    """POST /api/update/preflight has an intentionally empty JSON body."""


class UpdatePreflightResponse(TypedDict):
    merge_plan: UpdateMergePlan


class UpdateApplyRequest(TypedDict):
    strategy: Literal["auto_merge", "assisted", "manual", "replace"]
    expected_base_sha: NotRequired[str]
    expected_target_sha: NotRequired[str]
    confirm_recovery: NotRequired[bool]


class UpdateApplySuccessResponse(TypedDict):
    status: Literal["ok", "restart_required", "assisted_started", "manual"]
    restarting: NotRequired[bool]
    strategy: NotRequired[Literal["auto_merge", "assisted", "manual", "replace"]]
    task_id: NotRequired[str]
    merge_plan: NotRequired[UpdateMergePlan]
    error: NotRequired[str]


class UpdateApplyErrorResponse(TypedDict):
    error: str
    reason: NotRequired[str]
    blockers: NotRequired[List[str]]
    rolled_back: NotRequired[bool]
    rollback: NotRequired[str]
    restart_required: NotRequired[bool]
    merge_plan: NotRequired[UpdateMergePlan]
    smoke: NotRequired[Dict[str, Any]]


class UpdateStatusReadyOutbound(TypedDict):
    type: Literal["update_status_ready"]
    available: bool
    check_ok: Optional[bool]


class ProjectCreateRequest(TypedDict, total=False):
    """POST /api/projects body (v6.59.0). ONE source: ``path`` (attach an existing
    owner folder; optional ``init_git`` attach-snapshot commit — never auto-init),
    ``git_url`` (server-side clone, typed ``auth_required`` on credential failure),
    ``with_workspace`` (genesis provision), or none (file-less project).
    ``name``-only creation derives a filesystem id."""

    id: str
    name: str
    path: str
    init_git: bool
    git_url: str
    with_workspace: bool


class WorkspaceGitInitDecision(TypedDict, total=False):
    """The typed ``git_init_required`` OFFER (A12), not an error report.

    Raised by ``workspace_admission`` BEFORE a file task is queued in a folder that
    is safe and valid but not tracked by git. It reaches a CLIENT as an object on
    exactly ONE surface: the ``POST /api/tasks`` 400 body (``error_code`` +
    ``decision``). The project-room promote path carries the same object in its
    supervisor-internal outcome dict, but the halted task's chat message and
    persisted result carry the reason code and this object's ``message``, not its
    fields. ``enables`` is the plain-language answer to "what does saying yes buy me"
    — diff, rollback, branching — and ``offer`` names the operation the owner's yes
    calls (``POST /api/projects/{project_id}/init-git``). Nothing is ever
    initialised without that answer.
    """

    decision: Literal["git_init_required"]
    workspace_root: str
    project_id: str
    offer: Literal["init_git"]
    enables: List[str]
    message: str


class ThreadEntry(TypedDict, total=False):
    """One THREAD of a project — an empty chat sharing the project's folder.

    Thread ``0`` is the project's own chat (its ``chat_id`` equals the
    project's) and is SYNTHESIZED at read time from the project row, never
    stored; the project's top-level ``chat_id`` stays its compatibility alias.
    ``fork_of_chat_id`` + ``fork_before_ts`` are a CURSOR into the source
    thread's rows (rows are never copied) and appear together or not at all.

    ``lifecycle``/``archived_at``/``delete_error`` are D4's thread lifecycle, and
    they were SHIPPED on ``/api/state``, ``GET /api/projects`` and every
    ``ThreadResponse`` before they were declared here: the canonical projection
    normalises all three onto every row, and the client already reads
    ``thread.lifecycle`` to decide what a thread menu may offer. Field-level
    parity passed because both sides were equally wrong, which is the one failure
    mode a mirror cannot catch on its own — so the projection's own key set is
    pinned against this class as well. Thread #0 mirrors the PROJECT's lifecycle
    (it IS the project) and never carries an ``archived_at`` of its own.
    """

    id: int
    chat_id: int
    name: str
    created_at: str
    visible_revision: int
    fork_of_chat_id: int
    fork_before_ts: str
    lifecycle: Literal["active", "archived", "deleting", "tombstoned"]
    archived_at: str
    delete_error: str


class ProjectEntry(TypedDict, total=False):
    """A registry project row as returned by the projects endpoints. ``provenance``
    (attached|cloned|genesis|none) and ``clone_url`` are historical facts;
    operational git data is always read live from ``.git``. ``threads`` is the
    canonical projection (thread #0 first); a client that ignores it keeps
    working off ``chat_id`` exactly as before."""

    id: str
    name: str
    chat_id: int
    working_dir: str
    provenance: str
    clone_url: str
    trusted_at: str
    origin: str
    created_at: str
    last_active_at: str
    lifecycle: Literal["active", "deleting", "tombstoned"]
    routing_generation: int
    visible_revision: int
    delete_error: str
    threads: List[ThreadEntry]


class ProjectInitGitResponse(TypedDict, total=False):
    """``POST /api/projects/{project_id}/init-git`` — the owner's YES to the typed
    ``git_init_required`` offer, and the ONLY caller of the attach snapshot besides
    the create dialog's ``init_git``. ``init_git_skipped`` names credential-shaped
    files deliberately left OUT of the snapshot and still untracked (disclosed
    omission, P1); it is absent when nothing was skipped.
    """

    project: ProjectEntry
    working_dir: str
    init_git_skipped: List[str]


class ProjectFromTaskResponse(TypedDict, total=False):
    """``POST /api/projects/from-task`` — the "turn this task into a project" reply.

    ``working_dir`` is the folder the new project ADOPTED from the converted task
    (A11: a project born from work already happening somewhere inherits that place),
    and ``working_dir_error`` is the disclosure when it could not: the folder has
    moved, overlaps the Ouroboros roots, sits inside another repository, or is one
    of Ouroboros's own ephemeral checkouts. The conversion still succeeds — making
    the project is its job — which is exactly why the disclosure has to be typed.
    Untyped it was free text no contract knew about and no client read, so a
    conversion that quietly produced a PLACELESS project looked identical to one
    that worked, and the next task in that project provisioned a different empty
    tree somewhere else.
    """

    project: ProjectEntry
    binding: Dict[str, Any]
    working_dir: str
    working_dir_error: str


class ThreadCreateRequest(TypedDict, total=False):
    """POST /api/projects/{project_id}/threads body. ``name`` is optional — an
    unnamed thread gets a neutral default, with no model call."""

    name: str


class ThreadUpdateRequest(TypedDict):
    """POST /api/projects/{project_id}/threads/{thread_id}/update body."""

    name: str


class ThreadResponse(TypedDict):
    """Envelope of every thread lifecycle route (create / update / fork).

    These are OWNER surfaces reached through the gateway, deliberately not
    LLM-callable tools. The affected thread's ``chat_id`` also rides the
    `projects_changed` broadcast, so an open client adds it to its known-chat
    set before any live frame for it can arrive.
    """

    project_id: str
    thread: ThreadEntry


class ThreadLocation(TypedDict, total=False):
    """WHERE a thread works — DERIVED, never stored (A7).

    ``where`` is ``project_folder`` or ``worktree``, and it is the answer to one
    question: does a durable worktree exist for this thread? There is no toggle
    to read, so no client can be shown a location the filesystem disagrees with.
    The remaining fields are present only in the ``worktree`` case.
    """

    where: str
    path: str
    branch: str
    base_sha: str
    created_at: str


class ThreadBranchBase(TypedDict, total=False):
    """One base the owner may branch off from (A8).

    ``kind`` is ``branch``, ``tag`` or ``snapshot``. The snapshot entry — "exactly
    as it is now" — is not a git ref: ``creates_commit`` discloses whether
    choosing it would make a snapshot commit (a dirty tree) or simply reuse HEAD
    (a clean one). A commit-ish the owner types is accepted by the branch-off
    route and is deliberately not enumerated here; listing every commit is not an
    offer.
    """

    ref: str
    kind: str
    label: str
    dirty: bool
    creates_commit: bool


class ThreadQueueNotice(TypedDict, total=False):
    """Would a task started in this thread WAIT, and what should be said (A14)?

    ``queued`` is the fact; ``message`` is the ONE sentence every surface uses,
    and it says the TRUE thing — the task is queued behind the running one and
    will run when it finishes. It is not rejected. Earlier copy claimed rejection,
    which is the kind of wrong that teaches an owner to stop trusting the queue.
    ``remedy`` is ``branch_off`` only where branching would actually help: a
    thread already working in its own checkout is waiting on ITSELF, and offering
    to branch again there would be advice that does not work.
    """

    queued: bool
    reason: str
    message: str
    remedy: str


class ThreadBranchBasesResponse(TypedDict, total=False):
    """``GET /api/projects/{project_id}/threads/{thread_id}/branch-bases``."""

    project_id: str
    thread_id: int
    current_branch: str
    bases: List[ThreadBranchBase]
    snapshot: ThreadBranchBase
    location: ThreadLocation
    queue_notice: ThreadQueueNotice
    ok: bool
    reason: str
    message: str


class ThreadBranchOffRequest(TypedDict, total=False):
    """Branch-off body. ``base_ref`` is a branch, a tag, any commit-ish, or the
    ``@snapshot`` sentinel meaning "exactly as it is now"; empty means HEAD."""

    base_ref: str


class ThreadMergeBackRequest(TypedDict, total=False):
    """Merge-back body, entirely optional — a bare POST is the ordinary call.

    ``acknowledge_checkout_dirty`` IS the owner's consent to merge while the
    thread's checkout still holds uncommitted work: that work does not travel
    with the merge, so the default refuses (``checkout_dirty``) and the files are
    named again on the success. The same shape as the removal's
    ``acknowledge_unmerged`` — one consent idiom, not two."""

    acknowledge_checkout_dirty: bool


class ThreadWorktreeRemoveRequest(TypedDict, total=False):
    """Worktree-removal body. ``acknowledge_unmerged`` IS the owner's consent
    (A10): a checkout holding unmerged commits or uncommitted edits refuses
    without it, and there is no other path into the removal."""

    acknowledge_unmerged: bool


class ThreadDeleteRequest(TypedDict, total=False):
    """Thread-delete body, entirely optional — a bare POST is the ordinary call.

    ``acknowledge_unmerged`` IS the owner's consent to delete a thread whose
    checkout still holds ignored or untracked files (a ``node_modules/``, a
    ``build.log``): the default answers ``checkout_holds_rebuildable_files``,
    naming exactly what is there, and this flag is the yes. The SAME name the
    removal route uses, deliberately — one consent idiom, not three.

    It is NOT an override for work at risk. Unmerged commits, changes to tracked
    files and an unreadable checkout refuse with ``checkout_holds_work`` whatever
    this says, and are answered through the explicit removal route."""

    acknowledge_unmerged: bool


class ThreadWorktreeResponse(TypedDict, total=False):
    """ONE envelope for every branch/merge/remove answer, success or refusal.

    ``ok`` is the only field a client must read first. A refusal carries a TYPED
    ``reason`` plus owner-facing ``message`` copy and whatever evidence that
    reason has: ``conflicts`` for a stopped merge, ``dirty_files`` for a local
    tree that must be settled first, ``inspection`` for a removal that would
    destroy work, ``decision`` for T2's ``git_init_required`` offer. Sharing one
    envelope is deliberate — three near-identical shapes would drift, and the
    client renders refusals the same way whichever operation produced them.

    ``worktree_kept`` is stated explicitly on a successful merge because A10 turns
    on it: merging back never removes the checkout.

    ``dirty_files_total`` is the TRUE size of whichever bounded file listing this
    envelope carries — ``dirty_files`` on a refusal, ``checkout_left_behind`` on
    a merge that acknowledged work staying put, and the same number the
    ``inspection`` sub-object states. The lists are capped so an envelope cannot
    grow without bound; the count never is, because every owner-facing sentence
    that names a magnitude names this one. Counting the slice told an owner "200
    uncommitted file changes" about 800 of them, in the sentence immediately
    before an irreversible removal.
    """

    ok: bool
    reason: str
    message: str
    project_id: str
    thread_id: int
    location: ThreadLocation
    branch: str
    path: str
    base_ref: str
    base_sha: str
    working_dir: str
    decision: WorkspaceGitInitDecision
    snapshot_commit: Dict[str, Any]
    conflicts: List[str]
    dirty_files: List[str]
    #: Present whenever a bounded file listing is — the count the copy states.
    dirty_files_total: int
    merged: bool
    head_before: str
    head_after: str
    worktree_kept: bool
    removed: bool
    #: A CLEAN removal deletes the ``thread/<name>`` branch too, so the same
    #: thread can branch off again; a branch that SURVIVED reports why, because it
    #: is exactly what the next branch-off would refuse on.
    branch_removed: bool
    branch_kept_reason: str
    #: ``checkout_head_off_branch``: the branch the thread's checkout is actually
    #: standing on, which is not the one being merged.
    checkout_branch: str
    #: ``merge_abort_failed``: a merge that could neither complete NOR be undone
    #: left the project folder mid-merge, and says so rather than claiming the
    #: folder was left as it was.
    folder_left_mid_merge: bool
    abort_detail: str
    #: This refusal has an owner-answerable flag, so the owner is never stuck with
    #: only "no": ``checkout_dirty`` is answered by ``acknowledge_checkout_dirty``
    #: in the merge-back body, and ``unmerged_work`` by ``acknowledge_unmerged`` in
    #: the removal body. Documenting it for ``checkout_dirty`` alone was how the
    #: removal route came to build its refusal without the field at all, while its
    #: own sentence ended "or confirm you want it gone" and the client had no way
    #: to (I9). ``checkout_head_off_branch`` deliberately does NOT set it: that is
    #: not work left behind, it is a merge that would do nothing while reporting
    #: success.
    acknowledgeable: bool
    #: Named on a SUCCESSFUL merge: what the checkout still holds and the merge
    #: did not bring. Only ever non-empty when the owner acknowledged it —
    #: acknowledging is not the same as forgetting.
    checkout_left_behind: List[str]
    inspection: Dict[str, Any]
    error: str


class ThreadDiffResponse(TypedDict, total=False):
    """``GET /api/projects/{project_id}/threads/{thread_id}/diff`` (A13/X9).

    The SAME envelope as ``TaskDiffResponse`` — same statuses, same no-clipping
    rule, same ``patch``/``patch_sha256`` contract — plus the thread identity,
    because Changes is otherwise task-centric and its per-task route structurally
    cannot answer for a persistent checkout that has no task. ``source`` is always
    ``thread_checkout``; a thread that is not branched off answers ``blocked``
    with the typed ``thread_not_branched`` blocker rather than an empty diff,
    because "works in the project folder" is not "changed nothing".
    """

    project_id: str
    thread_id: int
    status: str
    source: str
    base_commit: str
    head_advanced: bool
    blockers: list[str]
    patch: str
    patch_sha256: str
    #: The checkout's branch, on EVERY answer including the refusals: the Changes
    #: header shows "thread · branch", and it learns the branch from the diff
    #: rather than requiring whoever opened the screen to already know it.
    branch: str
    error: str


class ThreadLifecycleResponse(TypedDict, total=False):
    """Archive / restore / delete answer (D4 with X10's admission fencing).

    ``lifecycle`` is ``active | archived | deleting | tombstoned``. Delete answers
    ``deleting``, not ``tombstoned``: the fence is up and routing into the thread
    is already closed, but its tasks are still being cancelled and the thread
    stays VISIBLE until they quiesce — the same shape a deleting PROJECT has.

    The three disclosures ride the response rather than living in a docstring no
    owner reads. ``journal_rows_retained`` is always true and says so: the chat
    journal is shared by every chat and nothing here rewrites it, so a deleted
    thread's rows physically remain and claiming erasure would be a lie.
    ``worktree_kept`` says the thread still has a checkout after the operation.
    ``worktree_removed`` (delete) says the checkout went with the thread, naming
    the ``branch`` and whether it went too: a tombstoned thread is invisible on
    every surface and branch/merge refuse it, so a checkout left behind is a
    folder and a branch that A10's explicit removal can no longer reach.

    Two refusals guard that, and they are NOT the same answer. Work at risk —
    unmerged commits, changes to TRACKED files, an unreadable checkout — refuses
    with ``checkout_holds_work`` and names the removal route; there is no flag
    that overrides it. A checkout holding only ignored or untracked content
    answers ``checkout_holds_rebuildable_files`` with ``acknowledgeable`` true,
    which is a question the owner answers by re-sending with
    ``acknowledge_unmerged`` (``ThreadDeleteRequest``). Both carry the
    ``inspection``: nothing here may destroy anything the owner has not been shown.
    ``visible_until_terminal`` (archive) says the thread was archived while a
    task was still running, so it stays on screen until that task finishes rather
    than hiding live output.
    """

    ok: bool
    reason: str
    message: str
    project_id: str
    thread_id: int
    chat_id: int
    lifecycle: str
    archived_at: str
    visible_until_terminal: bool
    journal_rows_retained: bool
    worktree_kept: bool
    worktree_removed: bool
    branch: str
    branch_removed: bool
    #: A refusal the owner can ANSWER (``checkout_holds_rebuildable_files``), in
    #: the same field name the merge-back envelope uses for ``checkout_dirty``.
    #: ``checkout_holds_work`` deliberately never sets it: that is not a question.
    acknowledgeable: bool
    inspection: Dict[str, Any]
    location: ThreadLocation


class ProjectDeleteResponse(TypedDict, total=False):
    """POST /api/projects/{project_id}/delete — fence, quiesce, and tombstone;
    the immutable binding, working folder, history, and memory are preserved.

    Its threads' CHECKOUTS are not (I1). A tombstoned project is invisible on
    every surface and branch/merge refuse a thread that is not live, so a checkout
    left behind is a folder and a ``thread/*`` branch nothing can reach — it goes
    WITH the project and is disclosed here rather than removed silently.
    ``worktrees_pending`` names the ones a task was still writing in, which the
    cancellation worker takes once the project quiesces; ``ok`` stays true because
    the deletion did start.

    A checkout holding work that cannot be REBUILT refuses instead: ``ok: false``,
    ``reason: "threads_hold_checkouts"`` under a 409, carrying the sentence a
    single thread's deletion gives for the same fact and ``threads`` naming each
    one. Same envelope shape as every other typed refusal, so a client reads
    ``ok`` first here as everywhere else.
    """

    ok: bool
    project_id: str
    folder_untouched: bool
    #: Thread ids whose checkout was removed with the project.
    worktrees_removed: List[int]
    #: ``thread/<name>`` branches deleted along with those checkouts.
    branches_removed: List[str]
    #: ``[{thread_id, path, branch, reason}]`` — a checkout that could not be
    #: taken yet. ``path``/``branch`` are named because the cancellation worker
    #: may not manage to take it either, and a tombstoned project has no surface
    #: left that could point the owner at the folder.
    worktrees_pending: List[Dict[str, Any]]
    reason: str
    message: str
    #: On ``threads_hold_checkouts``: ``[{thread_id, path, branch, inspection}]``.
    threads: List[Dict[str, Any]]


class FsDirsEntry(TypedDict):
    name: str
    path: str
    is_git: bool


class FsDirsResponse(TypedDict):
    """GET /api/fs/dirs — owner-facing directory browser for the New Project attach
    picker (server-side; works in web/Docker). Directories only, confined to the
    home tree, never file contents. ``truncated`` is True when the directory holds
    more children than the 500-entry cap (no silent truncation)."""

    path: str
    parent: str
    home: str
    dirs: List[FsDirsEntry]
    truncated: bool


class TaskNamedOutbound(TypedDict):
    """Outbound notice that the proactive card namer coined a project name for a fresh
    main-chat task (v6.40). The client sets the live card's title to ``suggested_name``;
    turn-into-project later reuses the same name. Not chat-scoped — carries only
    ``task_id`` and is a no-op unless a thread already holds that card."""

    type: Literal["task_named"]
    task_id: str
    suggested_name: str


class ErrorResponse(TypedDict):
    error: str


class StatusResponse(TypedDict):
    status: str


class HealthResponse(TypedDict):
    """Shape of ``GET /api/health``."""

    status: Literal["ok"]
    version: str
    runtime_version: str
    app_version: str


class EvolutionStateSnapshot(TypedDict):
    """Nested ``evolution_state`` block inside ``StateResponse``."""

    enabled: bool
    status: str
    detail: str
    cycle: int
    owner_chat_bound: bool
    last_task_at: str
    consecutive_failures: int
    # None when the remaining budget is infinite (unbudgeted): inf is not JSON
    # compliant, so the snapshot serializes it as null.
    budget_remaining_usd: Optional[float]
    budget_reserve_usd: float
    pending_count: int
    running_count: int
    queued_task_id: str
    running_task_id: str
    campaign: NotRequired[Dict[str, Any]]


class StateResponse(TypedDict):
    """Shape of ``GET /api/state`` (happy path)."""

    uptime: int
    workers_alive: int
    workers_total: int
    pending_count: int
    running_count: int
    spent_usd: Optional[float]
    budget_limit: float
    budget_pct: Optional[float]
    branch: str
    sha: str
    evolution_enabled: bool
    bg_consciousness_enabled: bool
    evolution_cycle: int
    evolution_state: EvolutionStateSnapshot
    bg_consciousness_state: Dict[str, Any]
    spent_calls: Optional[int]
    supervisor_ready: bool
    supervisor_error: Optional[str]
    runtime_mode: str
    context_mode: str
    # True when the EFFECTIVE `low` is a system auto-downgrade rather than an owner
    # selection: the owner control needs it to offer "confirm Low" on a no-op click.
    context_mode_auto_low: bool
    safety_mode: str
    skills_repo_configured: bool
    github_token_configured: bool
    accounting: Dict[str, Any]
    # Multi-project sidebar feed (additive, v6.32.0): compact registered
    # projects [{id, name, chat_id, working_dir, last_active_at, has_thread_activity}].
    projects: list
    # COMPLETE (uncapped) registered project chat_ids — the live WS fan-out
    # isolation SSOT, distinct from the capped/filtered `projects` list.
    project_chat_ids: list
    # Task->project bindings ({task_id: {project_id, chat_id}}) so the frontend
    # can recognise a project-scoped task card: suppress the stray "turn into
    # project" button (v6.33.0 P2) and render a pointer that opens the bound
    # project's panel (v6.33.0 F4).
    task_bindings: dict


class SettingsNetworkMeta(TypedDict):
    """Network fields inside the ``GET /api/settings`` ``_meta`` block."""

    bind_host: str
    bind_port: int
    lan_ip: str
    reachability: Literal["loopback_only", "lan_reachable", "host_ip_unknown"]
    recommended_url: str
    warning: str


class SettingsMeta(SettingsNetworkMeta, total=False):
    """Complete ``GET /api/settings`` ``_meta`` block."""

    custom_secret_keys: list[str]
    setup_contract: Dict[str, Any]


class SettingsSaveResponse(TypedDict, total=False):
    status: str
    no_changes: bool
    restart_required: bool
    restart_keys: list[str]
    immediate_changed: bool
    next_task_changed: bool
    warnings: list[str]
    # True when the save landed while an agent task was already STARTED: that
    # task keeps its start-time config (snapshot boundary); changes apply from
    # the next task. Queued-but-unstarted tasks re-read settings at start.
    agent_task_running: bool


class OwnerRuntimeModeResponse(TypedDict):
    ok: bool
    runtime_mode: str
    restart_required: bool


class OwnerAutoGrantResponse(TypedDict):
    ok: bool
    enabled: bool


class OwnerContextModeResponse(TypedDict):
    ok: bool
    context_mode: str


class OwnerScopeReviewFloorResponse(TypedDict):
    ok: bool
    scope_review_floor: str  # blocking_1m | advisory (v6.34.0, CW1)
    # v6.80.0: the value is STORED but enforcement-inert — scope-review applicability
    # follows the owner-only context mode. The notice says so on every write.
    deprecation_notice: str


class OwnerSafetyModeResponse(TypedDict):
    ok: bool
    safety_mode: str  # full | light | off (v6.54.3)


class SkillGrantResponse(TypedDict, total=False):
    ok: bool
    skill: str
    granted_keys: list[str]
    granted_permissions: list[str]
    extension_action: str
    extension_reason: str
    load_error: str
    grants: Dict[str, Any]
    error: str


class SkillDeleteResponse(TypedDict, total=False):
    ok: bool
    skill: str
    source: str
    deleted_payload_root: str
    deleted_state: bool
    extension_action: str
    extension_reason: str
    error: str


class UiPreferencesResponse(TypedDict):
    ok: NotRequired[bool]
    widget_order: list[str]
    nested_subagents_expanded: bool
    sidebar_width: int  # px; 0 = CSS default (resizable side sections, v6.33.0)
    project_panel_width: int  # px; 0 = CSS default
    # BREAKING since project threads (T1): the monotonic paint ACK is NESTED per
    # thread — {project_id: {thread_id: revision}}. A stored or posted FLAT
    # {project_id: revision} is accepted for one minor and normalized to
    # {project_id: {"0": revision}} (thread #0 IS the project's original chat).
    project_seen_revision: dict[str, dict[str, int]]
    project_order: list[str]  # owner drag-and-drop order; unlisted projects keep the default order
    project_thread_order: dict[str, list[str]]  # owner drag-and-drop thread order per project
    project_last_viewed: dict[str, str]  # deprecated one-minor accepted no-op
    project_hidden: dict[str, bool]  # deprecated one-minor accepted no-op


class GitLogResponse(TypedDict):
    commits: list[Dict[str, Any]]
    tags: list[str]
    branch: str
    sha: str


class EvolutionDataResponse(TypedDict):
    points: list[Dict[str, Any]]
    checkpoints: NotRequired[list[Dict[str, Any]]]
    generated_at: str
    cached: bool


class ScheduledTasksResponse(TypedDict):
    schema_version: int
    tasks: list[Dict[str, Any]]


class ScheduleUpsertResponse(TypedDict):
    ok: bool
    schedule: Dict[str, Any]


class ScheduleDeleteResponse(TypedDict):
    ok: bool


class UploadResponse(TypedDict):
    ok: bool
    filename: str
    display_name: str
    path: str
    size: int
    mime: str


class ExtensionsIndexResponse(TypedDict, total=False):
    extensions: list[Dict[str, Any]]
    skills: list[Dict[str, Any]]
    lifecycle: Dict[str, Any]
    error: str


class SkillLifecycleQueueResponse(TypedDict, total=False):
    active: Dict[str, Any]
    events: list[Dict[str, Any]]


class MarketplaceSearchResponse(TypedDict, total=False):
    items: list[Dict[str, Any]]
    results: list[Dict[str, Any]]
    installed: list[Dict[str, Any]]
    error: str


class MarketplaceInstalledResponse(TypedDict, total=False):
    installed: list[Dict[str, Any]]
    skills: list[Dict[str, Any]]
    error: str


class LocalModelStatusResponse(TypedDict, total=False):
    status: str
    running: bool
    ready: bool
    port: int
    message: str
    error: str


class McpStatusResponse(TypedDict, total=False):
    enabled: bool
    servers: list[Dict[str, Any]]
    tools: list[Dict[str, Any]]
    error: str


class ModelCatalogResponse(TypedDict, total=False):
    providers: list[Dict[str, Any]]
    models: list[Dict[str, Any]]
    error: str


class FileBrowserListResponse(TypedDict, total=False):
    root: str
    path: str
    entries: list[Dict[str, Any]]
    error: str


class ChatHistoryResponse(TypedDict, total=False):
    messages: list[Dict[str, Any]]
    has_more: bool
    next_before_ts: str
    error: str


class ExecutorRef(TypedDict, total=False):
    type: Required[Literal["local", "docker_exec"]]
    id: NotRequired[str]
    network: NotRequired[Literal["host", "none"]]
    workspace_host_path: NotRequired[str]
    workspace_backend_path: NotRequired[str]
    # Required at runtime when type == "docker_exec".
    container_name: NotRequired[str]
    path_mappings: NotRequired[list[Dict[str, str]]]


class _TaskCreateRequestRequired(TypedDict):
    description: str


class TaskCreateRequest(_TaskCreateRequestRequired, total=False):
    task_id: str
    type: str
    chat_id: int
    depth: int
    session_id: str
    workspace_root: str
    workspace_mode: str
    memory_mode: str
    project_id: str
    attachments: list[Dict[str, Any]]
    acceptance_claims: list[Dict[str, Any]]
    # v6.60.0: "" | "final_answer_line" — adapter-declared machine-extractable answer
    # protocol; flows into task_contract.answer_protocol and inherits to subagents.
    answer_protocol: str
    allowed_resources: Dict[str, Any]
    resource_policy: Dict[str, Any]
    disabled_tools: list[str]
    executor_ref: ExecutorRef
    service_teardown: Literal["stop", "keep"]
    deadline_at: str
    timeout_sec: float
    timeout: float
    context: str
    expected_output: str
    constraints: str
    context_requires_self_body_docs: bool
    actor_id: str
    source: str
    metadata: Dict[str, Any]


class TaskCreateResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    status: str
    error: str


class TaskListResponse(TypedDict, total=False):
    tasks: list[Dict[str, Any]]
    queue: Dict[str, Any]
    error: str


class TaskCostBreakdown(TypedDict):
    """Read-time "where did the money go" projection on ``GET /api/tasks/{task_id}``
    (ROOT tasks only). Computed from the physical-attempt ledger at read time and
    never persisted; ``own + children + unattributed == subtree``. When the object
    is present every key is present; the WHOLE object is absent — never a confident
    $0 — when accounting is unavailable or holds no attributable row for the
    subtree, and on non-root task details."""

    own_usd: float
    children_usd: float
    unattributed_usd: float
    delegated_disclosed_usd: float
    subscription_sessions: int
    unknown_unmetered: int
    non_final_rows: int
    cost_final: bool
    authority: Literal["physical_attempt_ledger"]


class TaskDetailResponse(TypedDict, total=False):
    """``GET /api/tasks/{task_id}`` — the public task-result envelope (open shape;
    stored task-result keys pass through) plus additive typed projections."""

    cost_breakdown: TaskCostBreakdown
    error: str


ClaudexorReadState = Literal["ok", "not_read", "failed"]


class ClaudexorStatusReads(TypedDict):
    """PROVENANCE for each independent facet of ``GET /api/claudexor/status``.

    An empty collection cannot say whether the daemon was ASKED: the owned
    Claudexor daemon starts lazily, so an idle machine used to serve empty
    lists that every consumer read as "no account connected" while real
    accounts sat in the agent home. Each facet answers only for itself, since
    one fanned-out read can fail while its siblings land:

    - ``ok`` — read; the matching collection is AUTHORITATIVE (empty means empty)
    - ``not_read`` — never asked (no live daemon)
    - ``failed`` — asked, and the answer did not arrive

    Facets map to ``harnesses`` (catalog), ``profiles`` (accounts) and ``quota``.
    The manifest read behind the login-capability filter is deliberately NOT a
    facet: its failure is absorbed (fail-open), never reported."""

    catalog: ClaudexorReadState
    accounts: ClaudexorReadState
    quota: ClaudexorReadState


class ClaudexorStatusResponse(TypedDict, total=False):
    """``GET /api/claudexor/status`` — owned-daemon lifecycle plus the daemon's
    own catalog/account/quota truth, each stamped with its read state. Read-only;
    never spawns the daemon (waking it is the owner-initiated POST)."""

    daemon: Dict[str, Any]
    config_dir: str
    harnesses: List[Dict[str, Any]]
    profiles: Dict[str, Any]
    quota: List[Dict[str, Any]]
    reads: ClaudexorStatusReads
    subagent_last_delegation: Dict[str, Any]
    error: str


class TaskEvent(TypedDict, total=False):
    seq: int
    source: str
    line: int
    ts: str
    type: str
    task_id: str
    root: str
    data: Dict[str, Any]


class TaskCancelResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    # v6.82 (P5): echoed when the optional request body {"cascade": true} asked
    # for the subtree cancel, which is COMPLETE by the time this answer is sent;
    # the plain envelope is unchanged.
    cascade: bool
    error: str


class TaskDiffResponse(TypedDict, total=False):
    """One task's owner-facing diff projection (GET /api/tasks/{id}/diff).

    ``status`` is the typed lifecycle: ``pending`` (artifacts are not finalized
    yet), ``ready`` (``patch`` carries the full unified diff), ``empty`` (the
    task changed nothing), ``blocked`` (``blockers`` names why no trustworthy
    patch can be shown). ``source`` is ``workspace_patch`` (durable artifact
    bytes) or ``mutation_baseline`` (a LIVE self-repo projection over the paths
    attributed to the task window). ``head_advanced`` discloses baseline drift
    as a boolean only — never commit counts, never an ownership claim. The patch
    is never truncated and carries no server-side file stats: the client parses
    the same bytes it renders.

    ``total=False`` is deliberate and pinned (§11.1): this envelope is an ADDITIVE
    frozen surface, so no field is ever declared required and the shape can never
    become a hard break for an older client. The endpoint's one response builder
    does in practice emit all seven on every 200 — a blocked or empty answer still
    carries ``patch: ""`` and ``blockers: []`` — but consumers are held to the
    weaker promise, and the JSDoc mirror in ``web/modules/api_types.js`` marks the
    same optionality so the two never disagree about what may be assumed.
    """

    status: str
    source: str
    base_commit: str
    head_advanced: bool
    blockers: list[str]
    patch: str
    patch_sha256: str
    error: str


class LogTailResponse(TypedDict, total=False):
    name: str
    entries: list[Dict[str, Any]]
    error: str


# Human/test-visible contract index; routers own executable Route objects.
HTTP_ENDPOINTS: tuple[str, ...] = (
    "GET /api/health",
    "GET /api/state",
    "GET /api/settings",
    "POST /api/settings",
    "GET /api/ui/preferences",
    "POST /api/ui/preferences",
    "POST /api/owner/runtime-mode",
    "POST /api/owner/auto-grant",
    "POST /api/owner/context-mode",
    "POST /api/owner/scope-review-floor",
    "POST /api/owner/safety-mode",
    "POST /api/owner/capability-ack",
    "POST /api/owner/skills/{skill}/attest-review",
    "GET /api/model-catalog",
    "POST /api/tasks",
    "GET /api/tasks",
    "GET /api/tasks/{task_id}",
    "GET /api/tasks/{task_id}/artifacts/{name}",
    "GET /api/tasks/{task_id}/diff",
    "GET /api/tasks/{task_id}/events",
    "POST /api/tasks/{task_id}/cancel",
    "POST /api/tasks/{task_id}/resume",
    "GET /api/schedules",
    "POST /api/schedules",
    "DELETE /api/schedules/{schedule_id}",
    "POST /api/command",
    "POST /api/reset",
    "GET /api/git/log",
    "POST /api/git/rollback",
    "POST /api/git/promote",
    "GET /api/update/status",
    "POST /api/update/check",
    "POST /api/update/preflight",
    "POST /api/update/apply",
    "GET /api/cost-breakdown",
    "GET /api/evolution-data",
    "GET /api/projects",
    "POST /api/projects",
    "POST /api/projects/from-task",
    "POST /api/projects/{project_id}/update",
    "POST /api/projects/{project_id}/delete",
    "POST /api/projects/{project_id}/init-git",
    "POST /api/projects/{project_id}/threads",
    "POST /api/projects/{project_id}/threads/{thread_id}/update",
    "POST /api/projects/{project_id}/threads/{thread_id}/fork",
    "GET /api/projects/{project_id}/threads/{thread_id}/branch-bases",
    "POST /api/projects/{project_id}/threads/{thread_id}/branch-off",
    "POST /api/projects/{project_id}/threads/{thread_id}/merge-back",
    "GET /api/projects/{project_id}/threads/{thread_id}/worktree",
    "POST /api/projects/{project_id}/threads/{thread_id}/worktree/remove",
    "GET /api/projects/{project_id}/threads/{thread_id}/diff",
    "POST /api/projects/{project_id}/threads/{thread_id}/archive",
    "POST /api/projects/{project_id}/threads/{thread_id}/restore",
    "POST /api/projects/{project_id}/threads/{thread_id}/delete",
    "GET /api/fs/dirs",
    "GET /api/chat/history",
    "GET /api/logs/{name}",
    "POST /api/chat/upload",
    "DELETE /api/chat/upload",
    "POST /api/openai-compatible/models",
    "GET /api/local-model/status",
    "POST /api/local-model/start",
    "POST /api/local-model/stop",
    "POST /api/local-model/test",
    "POST /api/local-model/install-runtime",
    "GET /api/mcp/status",
    "POST /api/mcp/refresh",
    "POST /api/mcp/test",
    "GET /api/reviewer-slots",
    "GET /api/claudexor/status",
    "POST /api/claudexor/wake",
    "POST /api/claudexor/login",
    "GET /api/claudexor/login/{job_id}",
    "DELETE /api/claudexor/login/{job_id}",
    "POST /api/claudexor/login/{job_id}/input",
    "GET /api/extensions",
    "GET /api/extensions/{skill}/manifest",
    "GET /api/extensions/{skill}/module/{entry}",
    "GET /api/extensions/{skill}/settings_section",
    "ANY /api/extensions/{skill}/{rest:path}",
    "GET /api/skills/daemons",
    "POST /api/skills/{skill}/toggle",
    "POST /api/skills/{skill}/delete",
    "GET /api/skills/lifecycle-queue",
    "POST /api/skills/{skill}/review",
    "POST /api/skills/{skill}/grants",
    "POST /api/skills/{skill}/reconcile",
    "GET /api/marketplace/clawhub/search",
    "GET /api/marketplace/clawhub/installed",
    "GET /api/marketplace/clawhub/info/{slug:path}",
    "GET /api/marketplace/clawhub/preview/{slug:path}",
    "POST /api/marketplace/clawhub/install",
    "POST /api/marketplace/clawhub/update/{name}",
    "POST /api/marketplace/clawhub/uninstall/{name}",
    "GET /api/marketplace/ouroboroshub/catalog",
    "GET /api/marketplace/ouroboroshub/installed",
    "GET /api/marketplace/ouroboroshub/preview/{slug:path}",
    "POST /api/marketplace/ouroboroshub/install",
    "POST /api/marketplace/ouroboroshub/update/{name}",
    "POST /api/marketplace/ouroboroshub/uninstall/{name}",
    "GET /api/onboarding",
    "GET /api/claude-code/status",
    "POST /api/claude-code/install",
    "GET /api/files/list",
    "GET /api/files/read",
    "GET /api/files/content",
    "GET /api/files/download",
    "POST /api/files/upload",
    "POST /api/files/mkdir",
    "POST /api/files/write",
    "POST /api/files/delete",
    "POST /api/files/transfer",
    "WS /ws",
)

WS_MESSAGE_TYPES: tuple[str, ...] = (
    "chat",
    "command",
    "photo",
    "video",
    "document",
    "typing",
    "log",
    "heartbeat",
    "extension_lifecycle",
    "message_annotation",
    "projects_changed",
    "task_named",
    "update_status_ready",
)


__all__ = [
    "ChatInbound",
    "TaskConstraintInbound",
    "CommandInbound",
    "ExtensionInbound",
    "TransportMetadata",
    "ChatOutbound",
    "PhotoOutbound",
    "VideoOutbound",
    "DocumentOutbound",
    "TypingOutbound",
    "LogOutbound",
    "HeartbeatOutbound",
    "ExtensionLifecycleOutbound",
    "ProjectsChangedOutbound",
    "MessageAnnotationOutbound",
    "UpdateMergePlan",
    "UpdatePreflightRequest",
    "UpdatePreflightResponse",
    "UpdateApplyRequest",
    "UpdateApplySuccessResponse",
    "UpdateApplyErrorResponse",
    "UpdateStatusReadyOutbound",
    "ProjectCreateRequest",
    "ProjectEntry",
    "ProjectFromTaskResponse",
    "ProjectInitGitResponse",
    "WorkspaceGitInitDecision",
    "ThreadCreateRequest",
    "ThreadBranchBase",
    "ThreadBranchBasesResponse",
    "ThreadBranchOffRequest",
    "ThreadDiffResponse",
    "ThreadEntry",
    "ThreadLifecycleResponse",
    "ThreadLocation",
    "ThreadQueueNotice",
    "ThreadMergeBackRequest",
    "ThreadWorktreeRemoveRequest",
    "ThreadWorktreeResponse",
    "ThreadResponse",
    "ThreadUpdateRequest",
    "ProjectDeleteResponse",
    "FsDirsEntry",
    "FsDirsResponse",
    "TaskNamedOutbound",
    "ErrorResponse",
    "StatusResponse",
    "HealthResponse",
    "StateResponse",
    "EvolutionStateSnapshot",
    "SettingsNetworkMeta",
    "SettingsMeta",
    "SettingsSaveResponse",
    "OwnerRuntimeModeResponse",
    "OwnerAutoGrantResponse",
    "OwnerContextModeResponse",
    "OwnerScopeReviewFloorResponse",
    "OwnerSafetyModeResponse",
    "SkillGrantResponse",
    "SkillDeleteResponse",
    "UiPreferencesResponse",
    "GitLogResponse",
    "EvolutionDataResponse",
    "ScheduledTasksResponse",
    "ScheduleUpsertResponse",
    "ScheduleDeleteResponse",
    "UploadResponse",
    "ExtensionsIndexResponse",
    "SkillLifecycleQueueResponse",
    "MarketplaceSearchResponse",
    "MarketplaceInstalledResponse",
    "LocalModelStatusResponse",
    "McpStatusResponse",
    "ModelCatalogResponse",
    "FileBrowserListResponse",
    "ChatHistoryResponse",
    "ExecutorRef",
    "TaskCreateRequest",
    "TaskCreateResponse",
    "TaskListResponse",
    "TaskCostBreakdown",
    "TaskDetailResponse",
    "ClaudexorReadState",
    "ClaudexorStatusReads",
    "ClaudexorStatusResponse",
    "TaskEvent",
    "TaskCancelResponse",
    "TaskDiffResponse",
    "LogTailResponse",
    "HTTP_ENDPOINTS",
    "WS_MESSAGE_TYPES",
]
