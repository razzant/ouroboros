/** Dependency-free JSDoc mirror of `ouroboros.gateway.contracts`. */

/**
 * @typedef {Object} StateResponse
 * @property {number} uptime
 * @property {number} workers_alive
 * @property {number} workers_total
 * @property {number} pending_count
 * @property {number} running_count
 * @property {?number} spent_usd
 * @property {number} budget_limit
 * @property {?number} budget_pct
 * @property {string} branch
 * @property {string} sha
 * @property {boolean} evolution_enabled
 * @property {boolean} bg_consciousness_enabled
 * @property {number} evolution_cycle
 * @property {Object} evolution_state
 * @property {Object} bg_consciousness_state
 * @property {?number} spent_calls
 * @property {boolean} supervisor_ready
 * @property {?string} supervisor_error
 * @property {string} runtime_mode
 * @property {string} context_mode
 * @property {boolean} context_mode_auto_low  // effective low is a system auto-downgrade, not an owner selection
 * @property {string} safety_mode
 * @property {boolean} skills_repo_configured
 * @property {boolean} github_token_configured
 * @property {Object} accounting  // physical-attempt ledger projection
 * @property {Array<Object>} projects  // active/deleting ProjectEntry sidebar projection
 * @property {Array<number>} project_chat_ids  // complete (uncapped) project chat_ids — WS fan-out isolation SSOT (v6.32.0)
 * @property {Object<string, {project_id: string, chat_id: number}>} task_bindings  // bound task -> its project: suppress the stray "turn into project" button (v6.33.0 P2) + render a pointer that opens the project panel (v6.33.0 F4)
 */

/**
 * @typedef {Object} EvolutionDataResponse
 * @property {Object[]} points
 * @property {Object[]=} checkpoints
 * @property {string} generated_at
 * @property {boolean} cached
 */

/**
 * @typedef {Object} HealthResponse
 * @property {"ok"} status
 * @property {string} version
 * @property {string} runtime_version
 * @property {string} app_version
 */

/**
 * @typedef {Object} OpenAICompatibleModelsResponse
 * @property {string[]} models
 * @property {string=} error
 */

/**
 * @typedef {Object} SettingsMeta
 * @property {string[]=} custom_secret_keys
 * @property {Object=} setup_contract
 */

/**
 * @typedef {Object} ChatInbound
 * @property {"chat"} type
 * @property {string} content
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {boolean=} force_plan
 * @property {Array<Object>=} attachments  // [{filename, display_name, mime}] — image uploads become native blocks (v6.26.0)
 * @property {number=} chat_id     // multi-project thread routing (v6.32.0); main chat = 1
 * @property {string=} project_id  // per-project memory scope (v6.32.0)
 */

/**
 * @typedef {Object} CommandInbound
 * @property {"command"} type
 * @property {string} cmd
 */

/**
 * @typedef {Object} ChatOutbound
 * @property {"chat"} type
 * @property {"user"|"assistant"|"system"} role
 * @property {string} content
 * @property {string} ts
 * @property {boolean=} markdown
 * @property {boolean=} is_progress
 * @property {string=} task_id
 * @property {boolean=} ephemeral_decision
 * @property {string=} task_incident
 * @property {string=} toast_once
 * @property {Object=} lifecycle
 * @property {string=} subagent_event
 * @property {string=} subagent_task_id
 * @property {string=} root_task_id
 * @property {string=} parent_task_id
 * @property {string=} delegation_role
 * @property {string=} subagent_role
 * @property {boolean=} accepted
 * @property {number=} active_subagent_count
 * @property {number=} max_active_subagents
 * @property {boolean=} queued_behind_active_cap
 * @property {string[]=} required_capabilities
 * @property {string=} write_surface
 * @property {string=} model_lane
 * @property {string=} requested_model_lane
 * @property {string=} effective_model_lane
 * @property {string=} executor_route
 *   Phase 6: the OPAQUE harness route RESOLVED AT DISPATCH for this bubble /
 *   subagent (delegated routes only) — the route it was sent to, not a receipt
 *   from the engine saying where it landed. Absent/empty = the ordinary native
 *   path; no chip is drawn.
 * @property {Object=} execution_evidence
 *   The completion-seam EVIDENCE the route decision is reconciled against:
 *   {delegated_runs_started, delegated_runs_settled, subscription_cost_usd,
 *   harness_models}. Terminal frames only; absent = "no evidence yet",
 *   never "ran natively".
 * @property {string=} model
 * @property {string=} task_group_id
 * @property {string=} task_event
 * @property {string=} status
 * @property {boolean=} cancelable
 *   v6.82 (P5): host-attested — this frame's task is a supervisor-queue task that
 *   POST /api/tasks/{id}/cancel can force-cancel (never set for direct-chat turns).
 * @property {?number=} cost_usd
 * @property {"available"|"unavailable"=} cost_accounting_status
 * @property {string=} cost_accounting_error
 * @property {boolean=} cost_final
 * @property {?number=} cost_usd_with_children
 * @property {boolean=} cost_with_children_partial
 * @property {?number=} reserved_usd
 * @property {?number=} unresolved_upper_bound_usd
 * @property {?number=} unknown_unmetered
 * @property {?number=} non_final_rows
 *   v6.87.48: the count of OPEN ledger rows — the disclosed cause of `cost_final: false`,
 *   which can hold with every dollar bucket at zero (an estimated $0.00, or a dispatched
 *   row whose reservation is exactly zero).
 * @property {string=} result
 * @property {boolean=} result_truncated
 * @property {string=} trace_summary
 * @property {boolean=} trace_summary_truncated
 * @property {string=} error
 * @property {string=} artifact_status
 * @property {Object=} artifact_bundle
 * @property {Object=} outcome_axes
 * @property {Object=} task_contract
 * @property {string=} reason_code
 * @property {Object=} review_status
 * @property {Object=} review_projection
 *   v6.74.0 additive keys: panels[].dialogue ({status, votes} — the reviewer-authored
 *   dialogue-status reduction), panels[].single_reviewer_no_diversity (boolean label),
 *   and actors[].dialogue_status ("continue_actionable"|"unreachable_here"|"stable_disagreement"|"").
 * @property {boolean=} worker_saturation_warning
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {Object=} transport
 * @property {number=} telegram_chat_id
 * @property {string=} system_type
 * @property {number=} chat_id
 */

/**
 * @typedef {Object} PhotoOutbound
 * @property {"photo"} type
 * @property {"user"|"assistant"} role
 * @property {string} image_base64
 * @property {string} mime
 * @property {string} ts
 * @property {string=} caption
 * @property {string=} content
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {Object=} transport
 * @property {number=} chat_id
 * @property {number=} telegram_chat_id
 */

/**
 * @typedef {Object} VideoOutbound
 * @property {"video"} type
 * @property {"user"|"assistant"} role
 * @property {string} video_base64
 * @property {string} mime
 * @property {string} ts
 * @property {string=} caption
 * @property {string=} content
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {Object=} transport
 * @property {number=} chat_id
 * @property {number=} telegram_chat_id
 */

/**
 * @typedef {Object} DocumentOutbound
 * @property {"document"} type
 * @property {"user"|"assistant"} role
 * @property {string} file_base64
 * @property {string} mime
 * @property {string} filename
 * @property {string} ts
 * @property {string=} caption
 * @property {string=} download_url
 * @property {string=} content
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {Object=} transport
 * @property {number=} chat_id
 * @property {number=} telegram_chat_id
 */

/**
 * @typedef {Object} LogOutbound
 * @property {"log"} type
 * @property {Object} data
 * @property {number=} chat_id  // multi-project thread routing (v6.32.0); main chat = 1
 */

/**
 * @typedef {Object} ProjectsChangedOutbound
 * @property {"projects_changed"} type
 * @property {string=} project_id
 * @property {number=} chat_id  // new project thread; client learns it before /api/state (v6.32.0)
 */

/**
 * Bubble-free presentation update for an existing owner message.
 * @typedef {Object} MessageAnnotationOutbound
 * @property {"message_annotation"} type
 * @property {"routing_ack"} annotation_type
 * @property {number=} chat_id
 * @property {string} client_message_id
 * @property {string} action
 * @property {string=} target
 * @property {string} status
 * @property {Array<Object>=} options
 * @property {boolean} suppress_bubble
 * @property {string=} ts
 */

/**
 * Additive /api/chat/history row fields (v6.73.0 Project origin projection).
 * A Project thread may synthesize its start message from the binding's own
 * durable copy when the canonical row left the bounded read window:
 * `origin_projected: true` marks such a synthesized user row (normal history
 * shape otherwise), and a `system_type: "origin_omission"` system row discloses
 * origins omitted past the synthesis cap. Both fields are additive and safely
 * ignorable by renderers.
 * @typedef {Object} ProjectOriginHistoryFields
 * @property {boolean=} origin_projected
 * @property {"origin_omission"=} system_type
 */

/**
 * POST /api/projects body (v6.59.0). ONE source: path (attach; optional init_git
 * attach-snapshot commit — never auto-init), git_url (server-side clone; typed
 * auth_required), with_workspace (genesis), or none (file-less).
 * @typedef {Object} ProjectCreateRequest
 * @property {string=} id
 * @property {string=} name
 * @property {string=} path
 * @property {boolean=} init_git
 * @property {string=} git_url
 * @property {boolean=} with_workspace
 */

/**
 * The typed git_init_required OFFER (A12) — not an error report. Admission raises
 * it BEFORE queueing a file task in a folder that is safe and valid but not tracked
 * by git. It reaches a client as an OBJECT on exactly one surface: the POST
 * /api/tasks 400 body (alongside error_code: 'git_init_required'). The project-room
 * promote path carries the same object in its supervisor-internal outcome, but the
 * halted task's chat message and result carry the reason code plus this object's
 * message, not its fields. `enables`
 * is the plain-language answer to "what does saying yes buy me" and `offer` names
 * the operation the yes calls (POST /api/projects/{project_id}/init-git). Nothing
 * is initialised in the owner's folder without that answer, and Ouroboros never
 * runs `git init` there itself.
 * @typedef {Object} WorkspaceGitInitDecision
 * @property {'git_init_required'=} decision
 * @property {string=} workspace_root
 * @property {string=} project_id
 * @property {'init_git'=} offer
 * @property {string[]=} enables
 * @property {string=} message
 */

/**
 * One THREAD of a project — an empty chat sharing the project's working folder.
 * Thread 0 is the project's OWN chat (its chat_id equals the project's) and is
 * synthesized from the project row rather than stored; the project's top-level
 * chat_id stays its compatibility alias. fork_of_chat_id + fork_before_ts are a
 * CURSOR into the source thread's rows (rows are never copied) and appear
 * together or not at all.
 *
 * lifecycle/archived_at/delete_error are D4's thread lifecycle. They were SHIPPED
 * on /api/state, GET /api/projects and every ThreadResponse before either side
 * declared them — the canonical projection normalises all three onto every row,
 * and project_thread_actions.js already reads thread.lifecycle — so field parity
 * passed with both sides equally wrong. Thread #0 mirrors the PROJECT's lifecycle
 * (it IS the project) and never carries an archived_at of its own.
 * @typedef {Object} ThreadEntry
 * @property {number=} id
 * @property {number=} chat_id
 * @property {string=} name
 * @property {string=} created_at
 * @property {number=} visible_revision
 * @property {number=} fork_of_chat_id
 * @property {string=} fork_before_ts
 * @property {"active"|"archived"|"deleting"|"tombstoned"=} lifecycle
 * @property {string=} archived_at
 * @property {string=} delete_error
 */

/**
 * @typedef {Object} ProjectEntry
 * @property {string} id
 * @property {string=} name
 * @property {number=} chat_id
 * @property {string=} working_dir
 * @property {string=} provenance   // attached | cloned | genesis | none (historical fact)
 * @property {string=} clone_url
 * @property {string=} trusted_at
 * @property {string=} origin
 * @property {string=} created_at
 * @property {string=} last_active_at
 * @property {"active"|"deleting"|"tombstoned"=} lifecycle
 * @property {number=} routing_generation
 * @property {number=} visible_revision
 * @property {string=} delete_error
 * @property {ThreadEntry[]=} threads  // canonical projection, thread 0 first
 */

/**
 * POST /api/projects/{project_id}/init-git response — the owner's YES to the
 * git_init_required offer, and the only caller of the attach snapshot besides the
 * create dialog's init_git. init_git_skipped names credential-shaped files
 * deliberately left OUT of the snapshot and still untracked; it is absent when
 * nothing was skipped.
 * @typedef {Object} ProjectInitGitResponse
 * @property {ProjectEntry=} project
 * @property {string=} working_dir
 * @property {string[]=} init_git_skipped
 */

/**
 * POST /api/projects/from-task response — "turn this task into a project".
 * working_dir is the folder the new project ADOPTED from the converted task (A11:
 * work already happening somewhere brings its place with it); working_dir_error is
 * the disclosure when it could not — the folder moved, overlaps the Ouroboros
 * roots, sits inside another repository, or is one of Ouroboros's own ephemeral
 * checkouts. The conversion still succeeds, so a client that ignores this field
 * shows a project that silently has no place.
 * @typedef {Object} ProjectFromTaskResponse
 * @property {ProjectEntry=} project
 * @property {Object=} binding
 * @property {string=} working_dir
 * @property {string=} working_dir_error
 */

/**
 * POST /api/projects/{project_id}/threads body. name is optional — an unnamed
 * thread gets a neutral default (no model call).
 * @typedef {Object} ThreadCreateRequest
 * @property {string=} name
 */

/**
 * POST /api/projects/{project_id}/threads/{thread_id}/update body.
 * @typedef {Object} ThreadUpdateRequest
 * @property {string} name
 */

/**
 * Envelope of every thread lifecycle route (create / update / fork). The
 * affected thread's chat_id also rides the projects_changed broadcast, so the
 * client adds it to its known-chat set before any live frame for it arrives.
 * @typedef {Object} ThreadResponse
 * @property {string} project_id
 * @property {ThreadEntry} thread
 */

/**
 * WHERE a thread works — DERIVED, never stored (A7). where is 'project_folder'
 * or 'worktree', and it answers exactly one question: does a durable worktree
 * exist for this thread? There is no toggle to read, so no client can be shown a
 * location the filesystem disagrees with. The other fields appear only for a
 * worktree.
 * @typedef {Object} ThreadLocation
 * @property {string=} where
 * @property {string=} path
 * @property {string=} branch
 * @property {string=} base_sha
 * @property {string=} created_at
 */

/**
 * One base the owner may branch off from (A8). kind is 'branch', 'tag' or
 * 'snapshot'. The snapshot entry — "exactly as it is now" — is not a git ref:
 * creates_commit discloses whether choosing it would make a snapshot commit (a
 * dirty tree) or simply reuse HEAD (a clean one). A commit-ish the owner types is
 * accepted by the branch-off route and is deliberately not enumerated here.
 * @typedef {Object} ThreadBranchBase
 * @property {string=} ref
 * @property {string=} kind
 * @property {string=} label
 * @property {boolean=} dirty
 * @property {boolean=} creates_commit
 */

/**
 * Would a task started in this thread WAIT, and what should be said (A14)?
 * queued is the fact; message is the ONE sentence every surface uses, and it says
 * the TRUE thing — the task is queued behind the running one and will run when it
 * finishes. It is not rejected. remedy is 'branch_off' only where branching would
 * actually help: a thread already working in its own checkout is waiting on
 * ITSELF, and offering to branch again there would be advice that does not work.
 * @typedef {Object} ThreadQueueNotice
 * @property {boolean=} queued
 * @property {string=} reason
 * @property {string=} message
 * @property {string=} remedy
 */

/**
 * GET /api/projects/{project_id}/threads/{thread_id}/branch-bases.
 * @typedef {Object} ThreadBranchBasesResponse
 * @property {string=} project_id
 * @property {number=} thread_id
 * @property {string=} current_branch
 * @property {ThreadBranchBase[]=} bases
 * @property {ThreadBranchBase=} snapshot
 * @property {ThreadLocation=} location
 * @property {ThreadQueueNotice=} queue_notice
 * @property {boolean=} ok
 * @property {string=} reason
 * @property {string=} message
 */

/**
 * Branch-off body. base_ref is a branch, a tag, any commit-ish, or the
 * '@snapshot' sentinel meaning "exactly as it is now"; empty means HEAD.
 * @typedef {Object} ThreadBranchOffRequest
 * @property {string=} base_ref
 */

/**
 * Worktree-removal body. acknowledge_unmerged IS the owner's consent (A10): a
 * checkout holding unmerged commits or uncommitted edits refuses without it, and
 * there is no other path into the removal.
 * @typedef {Object} ThreadWorktreeRemoveRequest
 * @property {boolean=} acknowledge_unmerged
 */

/**
 * Thread-delete body, entirely optional — a bare POST is the ordinary call.
 * acknowledge_unmerged IS the owner's consent to delete a thread whose checkout
 * still holds ignored or untracked files (a node_modules/, a build.log): the
 * default answers checkout_holds_rebuildable_files naming exactly what is there,
 * and this flag is the yes. The SAME name the removal route uses — one consent
 * idiom, not three. It is NOT an override for work at risk: unmerged commits,
 * changes to tracked files and an unreadable checkout refuse with
 * checkout_holds_work whatever this says.
 * @typedef {Object} ThreadDeleteRequest
 * @property {boolean=} acknowledge_unmerged
 */

/**
 * ONE envelope for every branch/merge/remove answer, success or refusal. ok is
 * the only field to read first. A refusal carries a typed reason, owner-facing
 * message copy, and whatever evidence that reason has: conflicts for a stopped
 * merge, dirty_files for a local tree that must be settled, inspection for a
 * removal that would destroy work, decision for the git_init_required offer.
 * worktree_kept is stated explicitly on a successful merge because A10 turns on
 * it: merging back never removes the checkout.
 * dirty_files_total is the TRUE size of whichever bounded listing rides along
 * (dirty_files, checkout_left_behind, and the inspection's own dirty_files):
 * the lists are capped so the envelope cannot grow without bound, the count
 * never is. Render the count from it, never from list.length — counting the
 * slice told an owner "200 uncommitted file changes" about 800 of them, in the
 * sentence immediately before an irreversible removal.
 * @typedef {Object} ThreadWorktreeResponse
 * @property {boolean=} ok
 * @property {string=} reason
 * @property {string=} message
 * @property {string=} project_id
 * @property {number=} thread_id
 * @property {ThreadLocation=} location
 * @property {string=} branch
 * @property {string=} path
 * @property {string=} base_ref
 * @property {string=} base_sha
 * @property {string=} working_dir
 * @property {WorkspaceGitInitDecision=} decision
 * @property {Object=} snapshot_commit
 * @property {string[]=} conflicts
 * @property {string[]=} dirty_files
 * @property {number=} dirty_files_total present whenever a bounded file listing
 *   is — the number every owner-facing sentence states
 * @property {boolean=} merged
 * @property {string=} head_before
 * @property {string=} head_after
 * @property {boolean=} worktree_kept
 * @property {boolean=} removed
 * @property {boolean=} branch_removed a CLEAN removal deletes the thread/<name>
 *   branch too, so the same thread can branch off again
 * @property {string=} branch_kept_reason why a branch SURVIVED a removal — it is
 *   exactly what the next branch-off would refuse on
 * @property {string=} checkout_branch on `checkout_head_off_branch`: the branch
 *   the thread's checkout is actually standing on, which is not the one merged
 * @property {boolean=} folder_left_mid_merge on `merge_abort_failed`: the merge
 *   could neither complete NOR be undone, so the project folder is stopped
 *   part-way through it and says so rather than claiming it was untouched
 * @property {string=} abort_detail
 * @property {boolean=} acknowledgeable this refusal carries an owner-answerable
 *   flag, so the owner is never stuck with only "no": checkout_dirty is answered
 *   by acknowledge_checkout_dirty (merge-back body), unmerged_work by
 *   acknowledge_unmerged (removal body). checkout_head_off_branch deliberately
 *   does not set it — that is not work left behind, it is a merge that would do
 *   nothing while reporting success
 * @property {string[]=} checkout_left_behind named on a SUCCESSFUL merge: what the
 *   checkout still holds and the merge did not bring — non-empty only when the
 *   owner acknowledged it, because acknowledging is not forgetting
 * @property {Object=} inspection
 * @property {string=} error
 */

/**
 * POST /api/projects/{project_id}/threads/{thread_id}/merge-back body. Entirely
 * optional — a bare POST is the ordinary call. acknowledge_checkout_dirty is the
 * owner's consent to merge while the checkout still holds uncommitted work, in
 * the same shape as the removal's acknowledge_unmerged.
 * @typedef {Object} ThreadMergeBackRequest
 * @property {boolean=} acknowledge_checkout_dirty
 */

/**
 * GET /api/projects/{project_id}/threads/{thread_id}/diff (A13). The SAME
 * envelope as TaskDiffResponse — same statuses, same no-clipping rule, same
 * patch/patch_sha256 contract — plus the thread identity, because Changes is
 * otherwise task-centric and its per-task route structurally cannot answer for a
 * persistent checkout that has no task. source is always 'thread_checkout'; a
 * thread that is not branched off answers blocked with the typed
 * thread_not_branched blocker, because "works in the project folder" is not
 * "changed nothing".
 * @typedef {Object} ThreadDiffResponse
 * @property {string=} project_id
 * @property {number=} thread_id
 * @property {string=} status
 * @property {string=} source
 * @property {string=} base_commit
 * @property {boolean=} head_advanced
 * @property {string[]=} blockers
 * @property {string=} patch
 * @property {string=} patch_sha256
 * @property {string=} branch the checkout's branch, on EVERY answer including
 *   the refusals — the Changes header shows "thread · branch" and learns the
 *   branch here rather than requiring whoever opened the screen to know it
 * @property {string=} error
 */

/**
 * Archive / restore / delete answer (D4 with X10's admission fencing). lifecycle
 * is 'active' | 'archived' | 'deleting' | 'tombstoned'. Delete answers
 * 'deleting', not 'tombstoned': the fence is up and routing into the thread is
 * already closed, but its tasks are still being cancelled and the thread stays
 * VISIBLE until they quiesce — the same shape a deleting project has.
 *
 * The three disclosures ride the response rather than living in a docstring no
 * owner reads. journal_rows_retained is always true and says so: the chat journal
 * is shared by every chat and nothing rewrites it, so a deleted thread's rows
 * physically remain and claiming erasure would be a lie. worktree_kept says the
 * thread still has a checkout afterwards; worktree_removed (delete) says a CLEAN
 * one went with the thread, naming its branch and whether that went too — a
 * tombstoned thread is invisible on every surface and branch/merge refuse it, so
 * a checkout left behind is a folder and a branch nothing can reach any more.
 *
 * Two refusals guard that and they are NOT the same answer. Work at risk —
 * unmerged commits, changes to TRACKED files, an unreadable checkout — refuses
 * with checkout_holds_work and names the removal route; no flag overrides it. A
 * checkout holding only ignored or untracked content answers
 * checkout_holds_rebuildable_files with acknowledgeable true, which is a question
 * the owner answers by re-sending with acknowledge_unmerged. Both carry the
 * inspection.
 * visible_until_terminal (archive) says the thread was archived while a task was
 * still running, so it stays on screen until that task finishes.
 * @typedef {Object} ThreadLifecycleResponse
 * @property {boolean=} ok
 * @property {string=} reason
 * @property {string=} message
 * @property {string=} project_id
 * @property {number=} thread_id
 * @property {number=} chat_id
 * @property {string=} lifecycle
 * @property {string=} archived_at
 * @property {boolean=} visible_until_terminal
 * @property {boolean=} journal_rows_retained
 * @property {boolean=} worktree_kept
 * @property {boolean=} worktree_removed
 * @property {string=} branch
 * @property {boolean=} branch_removed
 * @property {boolean=} acknowledgeable a refusal the owner can ANSWER
 *   (checkout_holds_rebuildable_files), in the same field name the merge-back
 *   envelope uses for checkout_dirty; checkout_holds_work never sets it
 * @property {Object=} inspection
 * @property {ThreadLocation=} location
 */

/**
 * POST /api/projects/{project_id}/delete. The project's own folder, history,
 * bindings, memory and id are preserved; its threads' CHECKOUTS are not. A
 * tombstoned project is invisible on every surface and branch/merge refuse a
 * thread that is not live, so a checkout left behind is a folder and a thread/…
 * branch nothing can reach — it goes WITH the project and is disclosed here
 * rather than removed silently. worktrees_pending names the ones a task was still
 * writing in, which the cancellation worker takes once the project quiesces; ok
 * stays true because the deletion did start.
 *
 * A checkout holding work that cannot be REBUILT refuses instead: ok false,
 * reason 'threads_hold_checkouts' under a 409, carrying the sentence a single
 * thread's deletion gives for the same fact and threads naming each one.
 * @typedef {Object} ProjectDeleteResponse
 * @property {boolean} ok
 * @property {string} project_id
 * @property {boolean} folder_untouched
 * @property {number[]=} worktrees_removed thread ids whose checkout went with it
 * @property {string[]=} branches_removed thread/<name> branches deleted with them
 * @property {Object[]=} worktrees_pending [{thread_id, path, branch, reason}] — not
 *   takeable yet; the folder is named because a tombstoned project has no surface
 *   left that could point at it
 * @property {string=} reason
 * @property {string=} message
 * @property {Object[]=} threads on threads_hold_checkouts: [{thread_id, path,
 *   branch, inspection}]
 */

/**
 * GET /api/fs/dirs — server-side directory browser (New Project attach picker).
 * @typedef {Object} FsDirsEntry
 * @property {string} name
 * @property {string} path
 * @property {boolean} is_git
 */

/**
 * @typedef {Object} FsDirsResponse
 * @property {string} path
 * @property {string} parent
 * @property {string} home
 * @property {FsDirsEntry[]} dirs
 * @property {boolean} truncated  // true when the dir holds more children than the 500-entry cap
 */

/**
 * @typedef {Object} TaskNamedOutbound
 * @property {"task_named"} type
 * @property {string} task_id
 * @property {string} suggested_name  // proactively-coined project name; client sets the live card title (v6.40.0)
 */

/**
 * @typedef {Object} UploadResponse
 * @property {boolean} ok
 * @property {string} filename
 * @property {string} display_name
 * @property {string} path
 * @property {number} size
 * @property {string} mime
 */

/**
 * @typedef {Object} OwnerRuntimeModeResponse
 * @property {boolean} ok
 * @property {string} runtime_mode
 * @property {boolean} restart_required
 */

/**
 * @typedef {Object} OwnerAutoGrantResponse
 * @property {boolean} ok
 * @property {boolean} enabled
 */

/**
 * @typedef {Object} OwnerContextModeResponse
 * @property {boolean} ok
 * @property {string} context_mode
 */

/**
 * @typedef {Object} OwnerScopeReviewFloorResponse
 * @property {boolean} ok
 * @property {string} scope_review_floor  // blocking_1m | advisory (v6.34.0, CW1)
 * @property {string} deprecation_notice  // v6.80.0: stored, but enforcement-inert
 */

/**
 * @typedef {Object} OwnerSafetyModeResponse
 * @property {boolean} ok
 * @property {string} safety_mode  // full | light | off (v6.54.3)
 */

/**
 * @typedef {Object} InstalledSkill
 * @property {string} name
 * @property {string} type
 * @property {string=} version
 * @property {string=} description
 * @property {boolean=} enabled
 * @property {string=} source
 * @property {string=} payload_root
 * @property {string=} review_status
 * @property {boolean=} review_stale
 * @property {Object=} review_gate
 * @property {boolean=} executable_review
 * @property {string=} review_profile
 * @property {boolean=} official_hub_verified
 * @property {boolean=} owner_attestable
 * @property {{visible: boolean, disabled: boolean, reason: string}=} submit_hub
 * @property {{current: Object, history: Object[]}=} skill_review
 * @property {boolean=} is_self_authored
 * @property {Object=} grants
 * @property {string[]=} permissions
 * @property {string[]=} conflicts
 * @property {{code: "skill_conflict", skills: string[], omitted: number}=} conflict
 */

/**
 * @typedef {Object} SkillGrantResponse
 * @property {boolean} ok
 * @property {string} skill
 * @property {string[]=} granted_keys
 * @property {string[]=} granted_permissions
 * @property {string=} extension_action
 * @property {string=} extension_reason
 * @property {string=} load_error
 * @property {Object=} grants
 */

/**
 * @typedef {Object} ExecutorRef
 * @property {"local"|"docker_exec"} type
 * @property {string=} id
 * @property {"host"|"none"=} network
 * @property {string=} workspace_host_path
 * @property {string=} workspace_backend_path
 * @property {string=} container_name Required when type is "docker_exec".
 * @property {Object[]=} path_mappings
 */

/**
 * @typedef {Object} TaskCreateRequest
 * @property {string} description
 * @property {string=} task_id
 * @property {string=} type
 * @property {number=} chat_id
 * @property {number=} depth
 * @property {string=} session_id
 * @property {string=} workspace_root
 * @property {"external"=} workspace_mode
 * @property {"forked"|"empty"|"shared"=} memory_mode
 * @property {string=} project_id Per-project facts scope id (else derived from the workspace path).
 * @property {Object[]=} attachments
 * @property {Object[]=} acceptance_claims Advisory Observable Acceptance Claims (`claim`/`surface`/`support`/`priority`).
 * @property {string=} answer_protocol  // "" | "final_answer_line" — machine-extractable answer line (v6.60.0)
 * @property {Object=} allowed_resources
 * @property {Object=} resource_policy
 * @property {string[]=} disabled_tools Declarative tool-policy denylist: tool names withheld from the agent (independent of allowed_resources).
 * @property {ExecutorRef=} executor_ref
 * @property {"stop"|"keep"=} service_teardown Task service finalization policy; `keep` is for external verifiers/owners that need live services after task completion. POSIX-only: on Windows a cancel/hard-timeout tree-kills all task processes, so `keep` is not preserved there.
 * @property {string=} deadline_at
 * @property {number=} timeout_sec
 * @property {number=} timeout
 * @property {string=} context
 * @property {string=} expected_output
 * @property {string=} constraints
 * @property {boolean=} context_requires_self_body_docs
 * @property {string=} actor_id Top-level task actor/provenance id; metadata.actor_id is reserved.
 * @property {string=} source Top-level task source/provenance label.
 * @property {Object=} metadata Arbitrary task metadata; executor_ref/workspace_executor keys are reserved.
 */

/**
 * @typedef {Object} TaskCreateResponse
 * @property {boolean} ok
 * @property {string} task_id
 * @property {string} status
 */

/**
 * @typedef {Object} TaskEvent
 * @property {number} seq
 * @property {string=} source
 * @property {number=} line
 * @property {string} type
 * @property {string} task_id
 * @property {string=} ts
 * @property {string=} root
 * @property {Object=} data
 */

/**
 * @typedef {Object} TaskListResponse
 * @property {Object[]} tasks
 * @property {Object=} queue
 */

/**
 * Read-time "where did the money go" projection on GET /api/tasks/{task_id}
 * (ROOT tasks only; computed from the physical-attempt ledger at read time,
 * never persisted). own + children + unattributed == subtree. When the object
 * is present every key is present; the WHOLE object is absent — never a
 * confident $0 — when accounting is unavailable or holds no attributable row
 * for the subtree, and on non-root task details.
 * @typedef {Object} TaskCostBreakdown
 * @property {number} own_usd
 * @property {number} children_usd
 * @property {number} unattributed_usd
 * @property {number} delegated_disclosed_usd
 * @property {number} subscription_sessions
 * @property {number} unknown_unmetered
 * @property {number} non_final_rows
 * @property {boolean} cost_final
 * @property {"physical_attempt_ledger"} authority
 */

/**
 * GET /api/tasks/{task_id} — the public task-result envelope (open shape;
 * stored task-result keys pass through) plus additive typed projections.
 * @typedef {Object} TaskDetailResponse
 * @property {TaskCostBreakdown=} cost_breakdown
 * @property {string=} error
 */

/**
 * PROVENANCE for each independent facet of GET /api/claudexor/status. An empty
 * collection cannot say whether the daemon was ASKED: the owned Claudexor daemon
 * starts lazily, so an idle machine served empty lists that every consumer read
 * as "no account connected" while real accounts sat in the agent home.
 * "ok" — read, the matching collection is AUTHORITATIVE (empty means empty);
 * "not_read" — never asked (no live daemon); "failed" — asked, no answer.
 * Facets are independent: one fanned-out read can fail while its siblings land.
 * @typedef {"ok"|"not_read"|"failed"} ClaudexorReadState
 */

/**
 * @typedef {Object} ClaudexorStatusReads
 * @property {ClaudexorReadState} catalog
 * @property {ClaudexorReadState} accounts
 * @property {ClaudexorReadState} quota
 */

/**
 * @typedef {Object} ClaudexorStatusResponse
 * @property {Object=} daemon
 * @property {string=} config_dir
 * @property {Array<Object>=} harnesses
 * @property {Object=} profiles
 * @property {Array<Object>=} quota
 * @property {ClaudexorStatusReads=} reads
 * @property {Object=} subagent_last_delegation
 * @property {string=} error
 */

/**
 * @typedef {Object} ScheduledTasksResponse
 * @property {number} schema_version
 * @property {Object[]} tasks
 */

/**
 * @typedef {Object} ScheduleUpsertResponse
 * @property {boolean} ok
 * @property {Object} schedule
 */

/**
 * @typedef {Object} ScheduleDeleteResponse
 * @property {boolean} ok
 */

/**
 * @typedef {Object} TaskCancelResponse
 * @property {boolean} ok
 * @property {string} task_id
 * @property {boolean=} cascade
 *   v6.82 (P5): echoed only when the request body {"cascade": true} asked for the
 *   subtree cancel, which is complete by the time this answer is sent; the plain
 *   single-task envelope is unchanged.
 */

/**
 * One task's owner-facing diff projection (GET /api/tasks/{task_id}/diff).
 * The client parses `patch` itself — the server sends no file stats and never
 * truncates the patch, so the file list, per-file status and +/- counts come
 * from the exact bytes the owner is shown (one snapshot = one truth).
 * EVERY field is optional here, mirroring `TaskDiffResponse(total=False)` in
 * contracts.py FIELD for field: the envelope is an additive frozen surface (§11.1)
 * that must never become a hard break for an older client, so a consumer is promised
 * only what it existence-checks. The live endpoint does emit all seven on every 200 —
 * a blocked or empty answer still carries `patch: ''` and `blockers: []` — but that
 * is behaviour, not the contract, and the two sides state the same weaker promise so
 * neither can drift into assuming more than the other.
 * @typedef {Object} TaskDiffResponse
 * @property {("pending"|"ready"|"empty"|"blocked")=} status
 *   pending = artifacts are not finalized yet; ready = `patch` holds the full
 *   unified diff; empty = the task changed nothing; blocked = `blockers` names
 *   why no trustworthy patch can be shown.
 * @property {("workspace_patch"|"mutation_baseline")=} source
 *   workspace_patch = durable artifact bytes; mutation_baseline = a LIVE
 *   self-repo projection over the paths attributed to the task window.
 * @property {string=} base_commit  the baseline commit the patch is computed against
 * @property {boolean=} head_advanced
 *   HEAD differs from the task baseline. A boolean disclosure only — no commit
 *   counts and no exclusive-ownership claim (attribution is evidence, not exclusion).
 * @property {string[]=} blockers  typed attribution/artifact blockers, always disclosed
 * @property {string=} patch  the full unified diff ('' unless status is ready)
 * @property {string=} patch_sha256  digest of the exact patch bytes served
 * @property {string=} error
 */

/**
 * @typedef {Object} LogTailResponse
 * @property {string} name
 * @property {Object[]} entries
 */

/**
 * @typedef {Object} SkillDeleteResponse
 * @property {boolean} ok
 * @property {string} skill
 * @property {string} source
 * @property {string} deleted_payload_root
 * @property {boolean} deleted_state
 * @property {string} extension_action
 * @property {string} extension_reason
 * @property {string=} error
 */

/**
 * @typedef {Object} UiPreferencesResponse
 * @property {string[]} widget_order
 * @property {boolean} nested_subagents_expanded
 * @property {number} sidebar_width  // px; 0 = CSS default (v6.33.0)
 * @property {number} project_panel_width  // px; 0 = CSS default
 * @property {Object.<string,Object.<string,number>>} project_seen_revision  // monotonic paint ACK, NESTED per thread since T1: {project_id: {thread_id: revision}}. A flat {project_id: revision} is accepted for one minor and reads back as {project_id: {"0": revision}}.
 * @property {string[]} project_order  // owner drag-and-drop order; unlisted projects keep the default order
 * @property {Object.<string,string[]>} project_thread_order  // owner drag-and-drop thread order per project
 * @property {Object.<string,string>} project_last_viewed  // deprecated accepted no-op
 * @property {Object.<string,boolean>} project_hidden  // deprecated accepted no-op
 * @property {boolean=} ok
 */

/**
 * @typedef {Object} UpdateMergePlan
 * @property {boolean=} available
 * @property {boolean=} auto_mergeable
 * @property {'clean'|'conflicting'|'current'|'unavailable'|'unknown'=} kind
 * @property {string=} error
 * @property {string=} remote
 * @property {string=} remote_branch
 * @property {string=} target_ref
 * @property {string=} update_channel
 * @property {string=} current_branch
 * @property {string=} base_sha
 * @property {string=} target_sha
 * @property {number=} local_dirty_count
 * @property {string=} local_snapshot
 * @property {string=} merge_commit
 * @property {string[]=} code_conflict_paths
 * @property {string[]=} doc_conflict_paths
 * @property {string[]=} hot_code_paths
 * @property {'auto_merge'|'assisted'=} recommended_strategy
 */

/**
 * @typedef {Object} UpdatePreflightRequest
 */

/**
 * @typedef {Object} UpdatePreflightResponse
 * @property {UpdateMergePlan} merge_plan
 */

/**
 * @typedef {Object} UpdateApplyRequest
 * @property {'auto_merge'|'assisted'|'manual'|'replace'} strategy
 * @property {string=} expected_base_sha
 * @property {string=} expected_target_sha
 * @property {boolean=} confirm_recovery
 */

/**
 * @typedef {Object} UpdateApplySuccessResponse
 * @property {'ok'|'restart_required'|'assisted_started'|'manual'} status
 * @property {boolean=} restarting
 * @property {'auto_merge'|'assisted'|'manual'|'replace'=} strategy
 * @property {string=} task_id
 * @property {UpdateMergePlan=} merge_plan
 * @property {string=} error
 */

/**
 * @typedef {Object} UpdateApplyErrorResponse
 * @property {string} error
 * @property {string=} reason
 * @property {string[]=} blockers
 * @property {boolean=} rolled_back
 * @property {string=} rollback
 * @property {boolean=} restart_required
 * @property {UpdateMergePlan=} merge_plan
 * @property {Object=} smoke
 */

/**
 * @typedef {Object} UpdateStatusReadyOutbound
 * @property {'update_status_ready'} type
 * @property {boolean} available
 * @property {?boolean} check_ok
 */

export const GATEWAY_CONTRACT_VERSION = '6.92.1';
