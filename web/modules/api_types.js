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
 * @property {boolean} context_mode_auto_low  // frozen compatibility field; always false
 * @property {string} safety_mode
 * @property {boolean} skills_repo_configured
 * @property {boolean} github_token_configured
 * @property {Object} accounting  // physical-attempt ledger projection
 * @property {Array<Object>} projects  // active/deleting ProjectEntry sidebar projection
 * @property {Array<number>} project_chat_ids  // complete (uncapped) project chat_ids — WS fan-out isolation SSOT (v6.32.0)
 * @property {Object<string, {project_id: string, chat_id: number}>} task_bindings  // bound task -> its project: suppress the stray "turn into project" button (v6.33.0 P2) + render a pointer that opens the project panel (v6.33.0 F4)
 * @property {ActiveDirectTurn[]=} active_direct_turns  // active direct/ephemeral chat turns snapshot
 * @property {ActiveChatActivity[]=} active_chat_activities  // combined snapshot: direct/ephemeral turns + root managed queue tasks
 */

/**
 * @typedef {Object} ActiveDirectTurn
 * @property {string} activity_id
 * @property {number} chat_id
 * @property {string} project_id
 * @property {string} client_message_id
 * @property {string} kind
 * @property {string} phase
 * @property {number} started_at
 */

/**
 * @typedef {Object} ActiveChatActivity
 * @property {string} activity_id
 * @property {number} chat_id
 * @property {string} project_id
 * @property {string} client_message_id  // empty for managed queue rows
 * @property {string} kind  // direct_chat | ephemeral_decision | managed_task
 * @property {string} phase  // managed rows: queued | working | finalizing
 * @property {number} started_at
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
 * @typedef {Object} ProviderTestRequest
 * @property {string} provider_id
 * @property {Object<string, string>=} overrides
 */

/**
 * @typedef {Object} ProviderTestResponse
 * @property {boolean} ok
 * @property {string=} error
 */

/**
 * @typedef {Object} SettingsMeta
 * @property {string[]=} custom_secret_keys
 * @property {Object=} setup_contract
 */

/**
 * The wizard payload plus two DECLARATIONS about the onboarding run. Settings
 * keys ride through unchanged (open shape); neither flag is authority — the
 * server re-proves fresh-install status and re-reads the live account state.
 * @typedef {Object} OnboardingCompleteRequest
 * @property {boolean=} subscriptionsConnected
 * @property {boolean=} skipSubscriptionPresets
 */

/**
 * @typedef {Object} OnboardingPresetProjection
 * @property {boolean} applied
 * @property {string} reason  // not_requested | not_install_time | skipped_by_owner | applied
 * @property {string[]} harnesses
 * @property {Object} receipt  // per-seat resolution record; {} when nothing was applied
 */

/**
 * Settings, runtime mode, the fresh-install safety default and the durable
 * completion fact land atomically on every success. Preset keys and the one-shot
 * preset marker land only when `preset.applied` is true.
 * @typedef {Object} OnboardingCompleteResponse
 * @property {boolean} ok
 * @property {string} status
 * @property {string} runtime_mode
 * @property {boolean} restart_required
 * @property {OnboardingPresetProjection} preset
 */

/**
 * 500 from an owner settings write whose BYTES ALREADY LANDED: `saved` is true
 * and `post_commit_failed` names the step that failed afterwards (environment
 * projection, supervisor start, hot-reload…). Never re-save on this — the
 * settings are on disk. Shared by POST /api/settings and the onboarding finish.
 * @typedef {Object} SettingsPostCommitFailureResponse
 * @property {string} error
 * @property {string} status  // saved_with_post_commit_error
 * @property {boolean} saved  // always true
 * @property {string} post_commit_failed
 */

/**
 * 503 from POST /api/onboarding/complete: NOTHING was persisted, the wizard
 * stays open, and `can_skip` means "finish without agent defaults" will work.
 * @typedef {Object} OnboardingPresetFailureResponse
 * @property {string} error
 * @property {string} code
 * @property {string} detail
 * @property {boolean} can_skip
 * @property {boolean} saved
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
 * @property {Object=} client_surface  // raw sending-surface observables measured at send time (pywebview/ua/viewport/matchMedia/captured_at)
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
 * @property {string=} task_phase
 *   "finalizing" on a root's early final answer: post-task synthesis still
 *   runs, so the frame is not the task's terminal conclusion.
 * @property {string=} task_incident
 * @property {string=} toast_once
 * @property {boolean=} task_id_pending
 *   X3: a repair receipt whose managed task id does not exist yet (minted at
 *   promotion) — typed truth instead of an invented id.
 * @property {Object=} lifecycle
 * @property {Object=} lifecycle_pointer
 *   C4 multi-chat dedupe: a duplicate lifecycle initiator's typed pointer to the
 *   job that already owns the routing ({job_id, kind, target, status, chat_id}).
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
 *   {delegated_runs_started, delegated_runs_settled, delegated_runs_succeeded,
 *   delegated_runs_failed, delegated_run_failure_states, evidence_read_failed,
 *   subscription_cost_usd, subscription_cost_estimated, harness_models,
 *   nanny_nudge_recorded, delegate_start_attempted}.
 *   Terminal frames only; absent = "no evidence yet", never "ran natively".
 *   `evidence_read_failed: true` = the custody log exists but could not be
 *   read — zero counts are then UNKNOWN, never a "no run" receipt.
 * @property {string=} actual_substrate
 *   The FACT beside the executor_route plan, derived from custody evidence
 *   ONLY (never usage/rounds): "harness_used" (>=1 delegated run succeeded) |
 *   "harness_attempted" (>=1 started, none succeeded) | "native_only" (none
 *   started). Always rides beside the raw execution_evidence counts. Terminal
 *   frames only; absent = no substrate claim (running, no evidence recorded,
 *   or unreadable evidence — unknown is never classified).
 * @property {string=} model
 * @property {string=} task_group_id
 * @property {string=} task_event
 * @property {string=} status
 * @property {boolean=} cancelable
 *   v6.82 (P5): host-attested — this frame's task is a supervisor-queue task that
 *   POST /api/tasks/{id}/cancel can force-cancel (never set for direct-chat turns).
 * @property {?number=} cost_usd
 * @property {?number=} accounted_upper_bound_usd
 *   C2: the additive HONEST name for cost_usd — an accounted upper bound, not a
 *   settled receipt. Same value as the deprecated alias, null when unknown.
 * @property {?number=} accounted_upper_bound_usd_with_children
 *   C2: honest name for cost_usd_with_children (same value, null when unknown).
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
 * @property {?boolean=} ledger_integrity_degraded
 *   C12: the ledger's INTEGRITY marker, produced by the cost authority all along but
 *   named in no carry list — an amount computed over a degraded ledger used to reach the
 *   surface indistinguishable from one computed over a sound ledger.
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
 * @typedef {Object} TypingOutbound
 * @property {"typing"} type
 * @property {string} action
 * @property {number=} chat_id  // multi-project: routes the indicator to the owning panel
 * @property {string=} activity_id
 * @property {string=} client_message_id
 * @property {string=} phase
 * @property {string=} kind  // stamped only for direct-registry-tracked turns; absent for queued managed tasks (snapshot has no deletion authority over them)
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
 * Additive /api/chat/history row fields on `system_type: "skill_review"` rows:
 * the exact-job reference the producer already writes into chat.jsonl. A row
 * carrying a non-empty `job_id` lets the Chat card lazily fetch the rendered
 * review via GET /api/skills/{skill}/review-history/{job_id}; rows without it
 * (legacy full-text rows) keep local expansion. All fields are additive and
 * safely ignorable by renderers.
 * @typedef {Object} SkillReviewHistoryRowFields
 * @property {string=} skill
 * @property {string=} status
 * @property {string=} content_hash
 * @property {string=} job_id
 * @property {number=} review_round
 * @property {number=} snapshot_attempt
 */

/**
 * GET /api/skills/{skill}/review-history/{job_id} response: the
 * server-rendered normalized review block for ONE terminal review record
 * (raw reviewer text stays in review_history.jsonl; degraded reviewers are
 * disclosed by model + status). Errors are `{error}` with a typed 404 for
 * unknown skill/job or unreadable history.
 * @typedef {Object} SkillReviewHistoryDetailResponse
 * @property {string} markdown
 * @property {string} status
 * @property {string} content_hash
 * @property {string} job_status
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
 * @typedef {Object} ProjectEntry
 * @property {string} id
 * @property {string=} name
 * @property {number=} chat_id
 * @property {string=} working_dir
 * @property {string=} provenance   // attached | cloned | genesis | none (historical fact)
 * @property {string=} clone_url
 * @property {string=} trusted_at
 * @property {string=} last_active_at
 * @property {"active"|"deleting"|"tombstoned"=} lifecycle
 * @property {number=} routing_generation
 * @property {number=} visible_revision
 * @property {string=} delete_error
 */

/**
 * @typedef {Object} ProjectDeleteResponse
 * @property {boolean} ok
 * @property {string} project_id
 * @property {boolean} folder_untouched
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
 * @property {number} accounted_upper_bound_usd
 *   C2: the explicit subtree total under its honest name — an accounted UPPER
 *   BOUND (own + children + unattributed), not a settled receipt.
 * @property {number} subscription_sessions
 * @property {number} unknown_unmetered
 * @property {number} non_final_rows
 * @property {boolean} cost_final
 * @property {"physical_attempt_ledger"} authority
 */

/**
 * GET /api/tasks/{task_id} — the public task-result envelope (open shape;
 * stored task-result keys pass through) plus additive typed projections.
 * cancel_state is the phase-A cancel projection: "pending" while a durable
 * cancel intent is open and the supervisor teardown has not settled (status
 * itself honestly stays running/scheduled); absent otherwise. The UI renders
 * the interim "Cancelling…" from this field, never from a status value.
 * cancel_reason rides beside it when the intent carries a reason (the WHY of
 * the pending cancellation); absent when no reason was recorded.
 * owner_hurry / owner_hurry_history (S3, HQ1): the typed owner-hurry
 * observability — the current block plus archived prior-attempt rows.
 * Absent on tasks nobody hurried. Task-detail data only, never chat.
 * stop_policy (S3, Q1) rides beside a pending cancel_state when the open
 * intent is the SOFT stop ("finalize_then_cancel") — the UI shows
 * "Finalizing…" and offers the hard escalation; absent on immediate intents.
 * @typedef {Object} TaskDetailResponse
 * @property {TaskCostBreakdown=} cost_breakdown
 * @property {string=} cancel_state
 * @property {string=} cancel_reason
 * @property {string=} stop_policy
 * @property {OwnerHurryProjection=} owner_hurry
 * @property {OwnerHurryProjection[]=} owner_hurry_history
 * @property {string=} error
 */

/**
 * PROVENANCE for each independent facet of GET /api/claudexor/status. An empty
 * collection cannot say whether the daemon was ASKED: the owned Claudexor daemon
 * starts lazily, so an idle machine served empty lists that every consumer read
 * as "no account connected" while real accounts sat in the agent home.
 * "ok" — read, the matching collection is AUTHORITATIVE (empty means empty);
 * "not_read" — never asked: no daemon, or discovery/handshake died before the
 * fan-out (which leaves every facet untouched); "failed" — asked, and no
 * usable answer came back (refused, or a body in the wrong shape).
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
 * @property {boolean=} unified_accounts
 * @property {Object=} subagent_last_delegation
 * @property {string=} error
 */

/**
 * One canonical login-job success envelope. Every operation carries one bare
 * daemon job at the top level; operation-specific metadata stays beside it.
 * Snapshot-only deviceCode is envelope-level, never nested inside job.
 * @typedef {Object} ClaudexorLoginJobResponse
 * @property {Object} job
 * @property {string=} cursor
 * @property {number=} sequence
 * @property {Object=} deviceCode
 * @property {string=} job_id
 * @property {boolean=} disclosure_native
 * @property {string=} attach_command
 * @property {boolean=} ok
 */

/**
 * Narrow typed problem envelope for login-job operations. required_actions is
 * the daemon's bounded top-level continuation list, not a client-side action
 * framework.
 * @typedef {Object} ClaudexorLoginJobProblem
 * @property {string} error
 * @property {string=} code
 * @property {Array<string>=} required_actions
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
 * @property {string=} cancel_state
 *   S3 (Q1/Q2): "pending" on the 202 acknowledgement of a
 *   {"stop_policy": "finalize_then_cancel"} request — the durable intent is
 *   open while the bounded finalization attempt runs. Absent on the legacy
 *   immediate path.
 * @property {string=} stop_policy
 *   The EFFECTIVE policy of the durable intent ("immediate" |
 *   "finalize_then_cancel"): a graceful request over an already-hard intent
 *   never softens it, and the answer says so.
 * @property {string=} error
 */

/**
 * POST /api/tasks/{task_id}/hurry — the text-free owner hurry control (HQ1:
 * no chat message, ever). The body carries ONLY a client-generated stable
 * request_id (reused on retry); any other field is refused.
 * @typedef {Object} TaskHurryRequest
 * @property {string} request_id
 */

/**
 * The owner_hurry block on the task result — task-detail observability,
 * never a chat message. state is the closed vocabulary
 * requested | applied | not_applied_before_terminal; effects maps each
 * host-rail effect to its recorded status. History rows carry the same shape
 * plus archived_at/archived_reason (rolled over on every same-id requeue).
 * @typedef {Object} OwnerHurryProjection
 * @property {number=} attempt_key
 * @property {string=} request_id
 * @property {string=} requested_by
 * @property {string=} requested_at
 * @property {string=} reason
 * @property {string=} state
 * @property {Object<string, string>=} effects
 * @property {string=} applied_at
 * @property {string=} reconciled_at
 * @property {string=} archived_at
 * @property {string=} archived_reason
 */

/**
 * Acknowledgement of the typed task-local acceleration control.
 * duplicate=true is the idempotent shape: the same request_id on the live
 * attempt (or a different id collapsing onto the one armed latch) returns
 * the existing acknowledgement without a second control.
 * @typedef {Object} TaskHurryResponse
 * @property {boolean} ok
 * @property {string} task_id
 * @property {string} request_id
 * @property {string=} state
 * @property {number=} attempt_key
 * @property {boolean=} duplicate
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
 * @property {Object.<string,number>} project_seen_revision  // monotonic paint ACK
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

export const GATEWAY_CONTRACT_VERSION = '7.0.0';
