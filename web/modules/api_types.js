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
 *   subscription_cost_usd, subscription_cost_estimated, harness_models}.
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
 * auth_required), with_workspace (genesis), connection_id + remote_root (a folder on
 * a remote host — RWS v2), or none (file-less).
 * The remote source is the TWO HALVES the owner can know and deliberately NOT a
 * serialized workspace_ref: the placement's workspace identity is allocated by the
 * TARGET at admission, so a client that could name it could claim a workspace it
 * never opened.
 * @typedef {Object} ProjectCreateRequest
 * @property {string=} id
 * @property {string=} name
 * @property {string=} path
 * @property {boolean=} init_git
 * @property {string=} git_url
 * @property {boolean=} with_workspace
 * @property {string=} connection_id An owner connection the store already trusts.
 * @property {string=} remote_root Target-native git worktree root; never a Home path.
 */

/**
 * POST /api/projects/{project_id}/update body. Two optional, combinable mutations:
 * `name` renames, and `connection_id` + `remote_root` REBIND the remote placement
 * (the same two halves create takes). A rebind advances routing_generation, so work
 * already resolved against the previous target is refused at queue insertion.
 * @typedef {Object} ProjectUpdateRequest
 * @property {string=} name
 * @property {string=} connection_id
 * @property {string=} remote_root
 */

/**
 * @typedef {Object} ProjectEntry
 * @property {string} id
 * @property {string=} name
 * @property {number=} chat_id
 * @property {string=} working_dir
 * @property {?ProjectWorkspaceRef=} placement  // present ONLY for a remote project; working_dir is then empty
 * @property {string=} provenance   // attached | cloned | genesis | remote | none (historical fact)
 * @property {string=} clone_url
 * @property {string=} trusted_at
 * @property {string=} origin
 * @property {string=} created_at
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
 * The executor projection. "ssh" is DERIVED from the persisted ProjectWorkspaceRef
 * (RWS v2) — nothing stores an ssh executor_ref independently, and Home path
 * mappings are forbidden on that arm.
 * @typedef {Object} ExecutorRef
 * @property {"local"|"docker_exec"|"ssh"} type
 * @property {string=} id
 * @property {"host"|"none"=} network
 * @property {string=} workspace_host_path
 * @property {string=} workspace_backend_path
 * @property {string=} container_name Required when type is "docker_exec".
 * @property {Object[]=} path_mappings
 * @property {string=} connection_id Required when type is "ssh".
 * @property {string=} remote_root Target-native root; never a Home path.
 * @property {string=} workspace_id
 */

/**
 * Wire mirror of the persisted placement descriptor (ouroboros/workspace_ref.py).
 * An "ssh" ref has NO Home path: `remote_root` is target-native.
 * @typedef {Object} ProjectWorkspaceRef
 * @property {"local"|"ssh"} kind
 * @property {string=} local_root
 * @property {string=} connection_id
 * @property {string=} remote_root
 * @property {string=} workspace_id
 */

/**
 * One owner connection row: the durable half comes from ouroboros/connection_store.py
 * and NEVER carries a secret (the OS account is the documented trust boundary, D6);
 * the status/evidence half is a bounded process-local live projection that is never
 * persisted back. `bootstrap_compatible`/`health_fresh` are Home admission evidence —
 * a transport status projection cannot manufacture them. `bootstrap_compatible` is
 * DERIVED from the durable `bootstrapped_at` (the executor stays installed on that
 * host across a Home restart), while `health_fresh` is inherently process-local.
 * @typedef {Object} ConnectionEntry
 * @property {string} id
 * @property {string} name
 * @property {string} ssh_alias
 * @property {string=} expected_host_id Pinned remote host identity (set by bootstrap only).
 * @property {Object[]=} host_id_history
 * @property {"active"|"retired"=} lifecycle Retire is SOFT; trust history survives.
 * @property {?string=} retired_at
 * @property {?string=} bootstrapped_at Durable: when a bootstrap last succeeded (cleared by retrust/retire).
 * @property {string=} bootstrap_build Durable: the executor build that bootstrap installed.
 * @property {number=} bootstrap_contract_set Durable: the Home<->execd contract set that bootstrap installed.
 * @property {string=} created_at
 * @property {string=} updated_at
 * @property {"connecting"|"ready"|"degraded"|"disconnected"|"unknown"=} status
 * @property {string=} phase
 * @property {string=} project_id Project whose live session the broker reported for this connection.
 * @property {string=} platform
 * @property {string=} architecture
 * @property {string=} build
 * @property {boolean=} bootstrap_compatible
 * @property {boolean=} health_fresh
 * @property {boolean=} execd_outdated Installed executor predates a shared-contract change; Bootstrap is the action.
 * @property {number=} required_contract_set The contract set this Home build requires.
 * @property {number=} bootstrap_contract_set The contract set the installed executor was built from.
 * @property {string=} blocked_by THE one blocker in front of this connection; ABSENT exactly when it is selectable.
 * @property {string=} blocker_action The single owner action that removes `blocked_by` — never a menu of maybes.
 * @property {string=} blocker_hint The owner sentence for `blocked_by`, rendered verbatim; no surface composes its own.
 * @property {number=} blocker_rank Position in the removal ladder; HIGHER means fewer remaining steps.
 * @property {string=} completion
 * @property {string=} error_code
 * @property {string=} action
 * @property {Object=} diagnostic
 * @property {Object[]=} log_refs
 * @property {Object[]=} warnings Bounded non-fatal transport observations.
 * @property {number=} log_refs_count TOTAL before the cap; truncation is count > log_refs.length.
 * @property {number=} warnings_count TOTAL before the cap; truncation is count > warnings.length.
 */

/**
 * @typedef {Object} ConnectionAddRequest
 * @property {string} name
 * @property {string} ssh_alias
 */

/**
 * @typedef {Object} ConnectionListResponse
 * @property {ConnectionEntry[]=} connections
 * @property {string=} error
 * @property {string=} error_code
 * @property {string=} action
 */

/**
 * Response of every transport-dependent owner connection action. A build without
 * the ssh transport answers a typed 503 `remote_transport_unavailable` here — the
 * UI must render that as an honest state, never as a pending spinner.
 *
 * `host_id`/`handshake` are what retrust rests on: they are the only way this page
 * learns the CURRENTLY observed host identity (see `observedHostId` in
 * connections_ui.js, and its twins in cli_connections.py and gateway/connections.py —
 * all three read the same three places, or a confirmation pair gets assembled from a
 * field one of them cannot see).
 * @typedef {Object} ConnectionActionResponse
 * @property {boolean=} ok
 * @property {ConnectionEntry=} connection
 * @property {string=} connection_id
 * @property {string=} status
 * @property {string=} phase
 * @property {string=} completion
 * @property {string=} error
 * @property {string=} error_code
 * @property {string=} action
 * @property {string=} host_id Host identity this live answer observed.
 * @property {Object=} handshake Target handshake block; also carries `host_id`.
 * @property {string=} platform
 * @property {string=} architecture
 * @property {string=} build
 * @property {boolean=} bootstrap_compatible
 * @property {boolean=} health_fresh
 * @property {boolean=} execd_outdated Installed executor predates a shared-contract change; Bootstrap is the action.
 * @property {number=} required_contract_set The contract set this Home build requires.
 * @property {number=} bootstrap_contract_set The contract set the installed executor was built from.
 * @property {string=} blocked_by THE one blocker in front of this connection; ABSENT exactly when it is selectable.
 * @property {string=} blocker_action The single owner action that removes `blocked_by` — never a menu of maybes.
 * @property {string=} blocker_hint The owner sentence for `blocked_by`, rendered verbatim; no surface composes its own.
 * @property {number=} blocker_rank Position in the removal ladder; HIGHER means fewer remaining steps.
 * @property {Object=} diagnostic
 * @property {Object[]=} log_refs
 * @property {Object[]=} warnings
 * @property {number=} log_refs_count TOTAL before the cap; truncation is count > log_refs.length.
 * @property {number=} warnings_count TOTAL before the cap; truncation is count > warnings.length.
 */

/**
 * @typedef {Object} ConnectionDirsResponse
 * @property {string=} connection_id
 * @property {string=} path
 * @property {string=} parent
 * @property {FsDirsEntry[]=} dirs
 * @property {boolean=} truncated
 * @property {string=} error
 * @property {string=} error_code
 * @property {string=} action
 */

/**
 * Live connection/admission WS frame. Durable secrets are absent by construction.
 * Every field comes out of the SAME projection that fills ConnectionEntry's live half
 * (gateway/connections.py::_public_live_fields), so the frame carries the target and
 * evidence fields too. `task_id`/`project_id` are present on the task-SCOPED frames
 * the gateway fans out per live task on the connection.
 * @typedef {Object} ConnectionStateOutbound
 * @property {"connection_state"} type
 * @property {string} connection_id
 * @property {string=} task_id
 * @property {string=} project_id
 * @property {"connecting"|"ready"|"degraded"|"disconnected"|"unknown"=} status
 * @property {string=} phase
 * @property {string=} completion
 * @property {string=} error_code
 * @property {string=} action
 * @property {string=} platform
 * @property {string=} architecture
 * @property {string=} build
 * @property {boolean=} bootstrap_compatible
 * @property {boolean=} health_fresh
 * @property {boolean=} execd_outdated Installed executor predates a shared-contract change; Bootstrap is the action.
 * @property {number=} required_contract_set The contract set this Home build requires.
 * @property {number=} bootstrap_contract_set The contract set the installed executor was built from.
 * @property {string=} blocked_by THE one blocker in front of this connection; ABSENT exactly when it is selectable.
 * @property {string=} blocker_action The single owner action that removes `blocked_by` — never a menu of maybes.
 * @property {string=} blocker_hint The owner sentence for `blocked_by`, rendered verbatim; no surface composes its own.
 * @property {number=} blocker_rank Position in the removal ladder; HIGHER means fewer remaining steps.
 * @property {Object=} diagnostic
 * @property {Object[]=} log_refs
 * @property {Object[]=} warnings
 * @property {number=} log_refs_count TOTAL before the cap; truncation is count > log_refs.length.
 * @property {number=} warnings_count TOTAL before the cap; truncation is count > warnings.length.
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
 * @typedef {Object} TaskDetailResponse
 * @property {TaskCostBreakdown=} cost_breakdown
 * @property {string=} cancel_state
 * @property {string=} cancel_reason
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

export const GATEWAY_CONTRACT_VERSION = '6.101.1';
