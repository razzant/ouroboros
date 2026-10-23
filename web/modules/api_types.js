/** Dependency-free JSDoc mirror of `ouroboros.gateway.contracts`; the interface-language envelopes (`UiI18n*`) sit in ./ui_i18n_types.js, the model-route previews in ./model_route_types.js, the Dashboard storage envelopes (`Storage*`) in ./storage_types.js, the Dashboard Evolution history (`Evolution*`) in ./evolution_types.js, the MCP import preview in ./mcp_import_types.js, the replaceable-client envelopes (`ServedUiState`, setup bootstrap, `/api/contract`, chat history rows, journey request bodies) in ./client_contract_types.js, the product-management envelopes (skills, marketplaces, accounts, local model, MCP, update status) in ./management_contract_types.js, the owner-decision ingress in ./decision_types.js. */
/**
 * @typedef {Object} CostPresentation
 * @property {'own'|'root_tree'} scope
 * @property {?number} tracked_amount
 * @property {boolean} has_unpriced
 * @property {boolean} tracked_final
 * @property {boolean} accounting_open
 * @property {boolean} has_rows
 */
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
 * @property {?string} branch  // null for detached HEAD or unavailable checkout identity
 * @property {string} sha
 * @property {?boolean} evolution_enabled  // null: the control is unknown (state unavailable/recovering)
 * @property {?boolean} bg_consciousness_enabled  // null: unknown, never "off"
 * @property {{quality: string, source: string, unconfirmed: Array<string>}} state_quality  // #1307 read quality of state.json
 * @property {number} evolution_cycle
 * @property {Object} evolution_state
 * @property {BgConsciousnessState} bg_consciousness_state  // the alarm clock's snapshot + server projection (status/detail)
 * @property {?number} spent_calls
 * @property {boolean} supervisor_ready
 * @property {?string} supervisor_error
 * @property {BootDocument=} boot  // the boot phases (same object as /api/health.boot); the Starting… header paints them
 * @property {string} runtime_mode
 * @property {string} context_mode
 * @property {boolean} context_mode_auto_low  // frozen compatibility field; always false
 * @property {EffortRange} effort_range  // the owner's effort range (POST /api/owner/effort-range), the tolerant read
 * @property {string} safety_mode
 * @property {boolean} skills_repo_configured
 * @property {boolean} github_token_configured
 * @property {Object} accounting  // physical-attempt ledger projection
 * @property {Array<ProjectEntry>} projects  // active/deleting sidebar projection
 * @property {Array<number>} project_chat_ids  // complete (uncapped) project chat_ids — WS fan-out isolation SSOT (v6.32.0)
 * @property {?number=} main_visible_revision  // Main's room revision (its standalone messages, as a Project's visible_revision counts); null while unreadable
 * @property {Object<string, {project_id: string, chat_id: number, origin_bound?: boolean}>} task_bindings  // bound task -> its project: suppress the stray "turn into project" button (v6.33.0 P2) + render a pointer that opens the project panel (v6.33.0 F4). origin_bound marks a task the host included because its OWNER MESSAGE already has a project (#902), so one message cannot keep a second convertible card
 * @property {ActiveDirectTurn[]=} active_direct_turns  // active direct/ephemeral chat turns snapshot
 * @property {boolean=} active_chat_activities_complete
 * @property {ActiveChatActivity[]=} active_chat_activities  // combined snapshot: direct/ephemeral turns + root managed queue tasks
 * @property {Array<Object>=} current_questions Canonical unanswered questions (Project forms and Main's own, by identity), independent of bounded history.
 * @property {boolean=} current_questions_complete False retains previously observed questions its gaps name.
 * @property {Array<{reason:string,task_id?:string,quiz_id?:string}>=} current_questions_gaps Canonical source availability diagnostics.
 * @property {Array<Object>=} main_attention Live human recovery owners, including children; the activity census stays root-only.
 * @property {boolean=} main_attention_complete False retains previously observed actions.
 * @property {Array<Object>=} main_attention_gaps Typed gaps in current action evidence.
 * @property {import('./work_types.js').Work[]} works  // one row per work: a request with its Continue episodes and live members
 * @property {Object<string, import('./work_types.js').RoomStatus>} rooms  // one status per room (chat_id), from its works
 * @property {import('./work_types.js').WorkSummary} work_summary  // the tray's one line
 * @property {boolean} works_complete  // false: a source was unreadable; keep what was seen, never read it as empty
 * @property {Array<Object>} works_gaps  // typed gaps of the work projection
 * @property {import('./client_contract_types.js').ServedUiState=} served_ui  // this generation's main interface; absent on older cores
 */
/**
 * @typedef {Object} BootDocument  // GET /api/health.boot and GET /api/state.boot: the serving generation's phases
 * @property {'starting'|'serving'|'ready'|'failed'} phase
 * @property {string} since
 * @property {boolean} dialogue_ready  // an owner message is accepted durably
 * @property {boolean} tasks_ready  // pooled task starts are admitted
 * @property {Array<{name: string, state: 'running'|'done'|'failed', started_at: string, ended_at: ?string, error: ?string}>} steps
 * @property {Object<string, {state: ?string, done: ?number, total: ?number}>} jobs  // the store's named background jobs
 * @property {?string} error
 * @property {?{ts: string, max_event_lag_sec: ?number}=} supervisor_load  // latest supervisor_load row's headline; null before the first
 */
/**
 * Background Consciousness alarm clock (server._describe_bg_consciousness_state over consciousness.status_snapshot). A wake-up is an ordinary Main turn; the direct-activity census owns its liveness.
 * @typedef {Object} BgConsciousnessState
 * @property {boolean} enabled
 * @property {string} status  // disabled | stopped | thinking | sleeping | waiting_for_first_conversation | allowance_exhausted | allowance_unknown | wake_rejected | wake_failed | wake_paused | wake_outcome_unknown
 * @property {string} detail  // one honest owner-readable line (e.g. "Sleeping until 14:05.")
 * @property {string} level  // observe | act | full
 * @property {string} next_wake_at  // ISO instant; "" when unknown
 * @property {string} last_wake_at  // ISO instant; "" before the first wake of this process
 * @property {string} last_wake_task_id
 * @property {string} last_wake_outcome  // running | done | paused | pausing | unknown | failed | rejected:<reason> | skipped:<reason>
 * @property {string} last_error
 * @property {?number} spent_24h_usd  // null when the ledger could not be read
 * @property {?number} daily_usd
 * @property {string} allowance_resets_at  // ISO instant the oldest counted spend leaves the 24 h window
 * @property {number} tasks_running  // live roots consciousness started
 * @property {number} max_tasks
 * @property {string} live_wake_task_id  // "" when no wake-up is running
 * @property {number} unknown_unmetered  // window rows without a price: spent_24h_usd is then a floor ("at least")
 * @property {boolean} integrity_degraded  // the ledger was quarantined/repaired; the numbers are best-effort
 */
/** @typedef {import('./task_activity_types.js').ActiveDirectTurn} ActiveDirectTurn */
/** @typedef {import('./task_activity_types.js').ActiveChatActivity} ActiveChatActivity */
/** @typedef {import('./task_activity_types.js').ToolVerificationFact} ToolVerificationFact */
/** @typedef {import('./evolution_types.js').EvolutionRefresh} EvolutionRefresh */
/** @typedef {import('./evolution_types.js').EvolutionDataResponse} EvolutionDataResponse */
/**
 * @typedef {Object} HealthResponse
 * @property {"ok"} status
 * @property {string} version
 * @property {string} runtime_version
 * @property {string} app_version
 * @property {string=} abi_version  // gateway ABI carrier; absent on cores older than GET /api/contract
 * @property {BootDocument=} boot  // the serving generation's phases; answered while the boot runs
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
 * @typedef {Object} AvailableSubagentRoute
 * @property {'api_model'|'agent_session'} kind
 * @property {string} target_id
 * @property {string=} credential_profile_id
 */
/**
 * @typedef {Object} AvailableSubagentItem
 * @property {string} subagent_id
 * @property {boolean=} enabled - Omitted means true; false withdraws new selections.
 * @property {string} recommended_use
 * @property {AvailableSubagentRoute} route
 * @property {string=} effort
 */
/**
 * @typedef {Object} AvailableSubagentsSetting
 * @property {boolean} enabled
 * @property {AvailableSubagentItem[]} items
 */
/**
 * @typedef {Object} AvailableSubagentsSettingsMeta
 * @property {string=} source
 * @property {string=} diagnostic
 * @property {Object[]=} diagnostics
 * @property {Object|null=} candidate
 * @property {boolean=} automatic The read's own unsaved catalog, following Main and access.
 */
/**
 * @typedef {Object} SettingsMeta
 * @property {string[]=} custom_secret_keys
 * @property {Object=} setup_contract
 * @property {AvailableSubagentsSettingsMeta=} available_subagents
 * @property {SettingsPolicyState=} policy_state
 * @property {{restart_required:boolean,restart_keys:string[],restart_source_unknown_keys:string[],unknown_keys:string[],local_model:Object,summary:string}=} restart_state Component application and source uncertainty.
 */
/**
 * @typedef {Object} SettingsPolicyState
 * @property {{configured:string,effective:string,current_process:string,next_task:string,restart_required:boolean,applies:string}} access
 * @property {{configured:string,effective:string,current_process:string,next_task:string,pending:boolean,applies:string,active_task_snapshot:boolean}} supervisor
 * @property {{configured:string,effective:string,current_process:string,next_task:string,pending:boolean,applies:string,active_task_snapshot:boolean}} review
 * @property {boolean} running_task_snapshot
 */
/**
 * The wizard payload plus two DECLARATIONS about the onboarding run. Settings
 * keys ride through unchanged (open shape); neither flag is authority — the
 * server re-proves fresh-install status and re-reads the live account state.
 * @typedef {Object} OnboardingCompleteRequest
 * @property {boolean=} subscriptionsConnected
 * @property {boolean=} skipSubscriptionPresets
 * @property {Object=} OUROBOROS_SUBAGENTS
 * @property {boolean=} allow_empty_review_pool  // confirms saving agents that leave no review row
 * @property {string[]=} starter_skills  // catalog ids installed once after completion; absent: the default list
 * @property {Object<string, string>=} skill_source_trust  // {hub row id: trusted|review}; marks them confirmed
 */
/**
 * POST /api/onboarding/subagents/preview accepts the same open provider/local draft and subscription
 * declarations as onboarding completion. It returns a canonical editable actor list without persisting anything.
 * @typedef {OnboardingCompleteRequest} OnboardingSubagentsPreviewRequest
 */
/**
 * @typedef {Object} OnboardingSubagentsPreviewResponse
 * @property {boolean} ok
 * @property {Object} model_settings  // the model setting keys the draft would save
 * @property {AvailableSubagentsSetting} available_subagents
 * @property {string} source
 * @property {Object[]} diagnostics
 */
/**
 * @typedef {Object} OnboardingPresetProjection
 * @property {boolean} applied
 * @property {string} reason  // not_requested | not_install_time | skipped_by_owner | configured_by_owner | applied
 * @property {string[]} harnesses
 * @property {Object} receipt  // per-seat resolution record; {} when nothing was applied
 */
/**
 * Settings, mode, safety and completion persist atomically; only applied presets persist a one-shot marker.
 * @typedef {Object} OnboardingCompleteResponse
 * @property {boolean} ok
 * @property {string} status
 * @property {string} runtime_mode
 * @property {boolean} restart_required
 * @property {OnboardingPresetProjection} preset
 */
/**
 * 500 AFTER settings landed: retry the named later step, never re-save (settings/onboarding writes).
 * @typedef {Object} SettingsPostCommitFailureResponse
 * @property {string} error
 * @property {string} status  // saved_with_post_commit_error
 * @property {boolean} saved  // always true
 * @property {string} post_commit_failed
 */
/**
 * 503 settings_save_timeout: the server writer still runs; saved=null is unknown. Reload to check status before retrying.
 * @typedef {Object} SettingsSaveTimeoutResponse
 * @property {string} error
 * @property {string} code  // settings_save_timeout
 * @property {null} saved
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
/** @typedef {import('./client_contract_types.js').ChatInbound} ChatInbound */
/** @typedef {import('./client_contract_types.js').CommandInbound} CommandInbound */
/** @typedef {import('./client_contract_types.js').PanicAcceptedOutbound} PanicAcceptedOutbound */
/** @typedef {import('./client_contract_types.js').CallOpenRequest} CallOpenRequest */
/** @typedef {import('./client_contract_types.js').CallOpenResponse} CallOpenResponse */
/** @typedef {import('./client_contract_types.js').CallRefusal} CallRefusal */
/** @typedef {import('./client_contract_types.js').CallSocketContract} CallSocketContract */
/** @typedef {import('./client_contract_types.js').ConversationFinishedOutbound} ConversationFinishedOutbound */
/**
 * @typedef {Object} ChatOutbound
 * @property {import('./client_contract_types.js').PresenceContact=} presence_contact
 * @property {{kind:'publication'|'answer'|'closure',state:'owed'|'delivered',reason_code?:string,dismissed?:boolean}=} question_delivery Canonical settlement delivery fact.
 * @property {string=} closed_reason
 * @property {string=} closed_at
 * @property {string=} closed_by_task_id
 * @property {string=} closed_actor
 * @property {string=} closed_request_id
 * @property {Array<Object>=} materials Validated existing artifact references with task_id, name, sha256, size and download_url.
 * @property {string=} quiz_id
 * @property {string=} quiz_state
 * @property {number=} project_chat_id
 * @property {string=} source_status
 * @property {string=} owner_wait_state
 * @property {string=} owner_wait_resume_reason
 * @property {boolean=} wait_for_answer
 * @property {string=} wait_ended_at
 * @property {string=} question
 * @property {string[]=} options
 * @property {string[]=} option_details
 * @property {string=} stake
 * @property {string=} assumption
 * @property {number=} recommended_index
 * @property {number=} answered_index
 * @property {string=} comment
 * @property {string=} host_facts
 * @property {"chat"} type
 * @property {"user"|"assistant"|"system"} role
 * @property {string} content
 * @property {string} ts
 * @property {boolean=} ingress_accepted Canonical inbound row saved; not proof of task start or model delivery.
 * @property {boolean=} ingress_dispatched This live host process accepted the row and entered its dispatch; absent after a host restart (unknown).
 * @property {boolean=} ingress_pending This live host process accepted the row and has entered or refused neither yet: a later echo or history read says which.
 * @property {Array<NonNullable<UploadResponse['view']>>=} attachments The owner message's ChatAttachmentView list: the same views history replays.
 * @property {boolean=} text_placeholder The owner row's text is the host's placeholder (no words were sent): no caption is shown.
 * @property {boolean=} markdown
 * @property {boolean=} is_progress
 * @property {string=} task_id
 * @property {Object=} origin_message_ref Host-captured inbound identity for a correlated operation's terminal reply.
 * @property {boolean=} ephemeral_decision
 * @property {number=} tool_calls
 * @property {number=} rounds
 * @property {string=} suggested_name
 * @property {Object=} model_execution
 * @property {string=} task_phase "finalizing" on a root's early final answer: post-task synthesis still runs, so the frame is not the task's terminal conclusion.
 * @property {string=} task_terminal_status Typed terminal fact on a frame that IS the turn's conclusion (stamped on direct/ephemeral finals and the direct error branch).
 * @property {string=} task_incident
 * @property {string=} cancel_physical_task_id
 *   Historical rows only: the retired cancellation-fault notice named the
 *   physical task it could not settle when that differed from the displayed id.
 * @property {string=} toast_once
 * @property {string=} toast_tone
 *   The incident's valence for the one-shot toast (warn/ok/error); absent = the alarm tone.
 * @property {boolean=} task_id_pending
 *   X3: a repair receipt whose managed task id does not exist yet (minted at promotion); typed truth, not an invented id.
 * @property {Object=} lifecycle
 * @property {Object=} lifecycle_pointer
 *   C4 multi-chat dedupe: a duplicate initiator's typed pointer to the job that owns the routing ({job_id, kind, target, status, chat_id}).
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
 * @property {string=} effort_level  The effort decided at dispatch (a session row: its leaf's); empty = unknown, no chip.
 * @property {string=} effort_requested  The parent's request when it made one.
 * @property {string=} effort_source  auto | pin | model_name | cyber.
 * @property {string=} executor_route  Opaque delegated harness route resolved at dispatch, not an engine receipt; absent = ordinary native presentation, no chip.
 * @property {Object=} parent_cognitive_route  Recorded parent snapshot; never inferred from the child's model lane.
 * @property {string=} coordinator_route
 * @property {boolean=} use_local_model
 * @property {string=} coordinator_billing  Recorded subscription | measured | unknown; never inferred on replay.
 * @property {string=} coordinator_provenance
 * @property {Object=} executor_observation
 *   Latest observed progress actor, bound to own task/attempt/run/revision.
 *   model_source distinguishes requested and observed; not a terminal receipt.
 * @property {Object=} delegated_activity
 *   One host progress observation: {v, task_id, run_id, after_seq,
 *   through_seq, source{kind: run_events|timeline_window, read_through?, ref?,
 *   provisional?}, parts[{kind: message|thinking|problem, actor, text, seq?,
 *   last_seq?, cuts?[[seq, Unicode code-point offset]], cuts_truncated?, chars?,
 *   truncated?}], technical?{count, labels, recent, seqs?, seqs_truncated?},
 *   gaps?[{after_seq, through_seq, reason, final?}],
 *   omitted?, latest_message?}. Identity is (run_id, seq); host progress about
 *   the executor, never narration or execution evidence; `source.ref` names
 *   retained redacted JSONL (confined task-file `?source=`, 503 where unsupported).
 *   Incomplete preview identity is disclosed; a preview is not the whole journal.
 * @property {Object=} execution_evidence
 *   The completion-seam EVIDENCE the route decision is reconciled against:
 *   {delegated_runs_started, delegated_runs_settled, delegated_runs_succeeded,
 *   delegated_runs_failed, delegated_run_failure_states, evidence_read_failed,
 *   subscription_cost_usd, subscription_cost_estimated, harness_models,
 *   nanny_nudge_recorded, delegate_start_attempted,
 *   applied_access_profiles}.
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
 * @property {boolean=} _is_direct_chat
 *   The lane fact of a direct conversation turn, stamped by the host on the
 *   turn's own progress/tool frames and on every task_done; the chat block
 *   reads it before any census lists the turn.
 * @property {boolean=} narration
 *   The VOICE of a progress frame, stamped by the worker on every note it
 *   emits: true only for the model's own round narration, false for every
 *   host-authored note (checkpoints, fallback, plan, acceptance, nudge,
 *   transport, density). Both stay visible rows; only narration may claim the
 *   card title and the collapsed activity line. Absent = a frame that predates
 *   the fact (an older worker, a supervisor note, a stored row), which keeps
 *   the legacy reading that promoted every progress frame.
 * @property {{call_id:string,item_id:string,played_ms:number,status:'played'|'generated'|'interrupted'}=} spoken
 *   Generated speech and actual playback evidence; retained on history replay.
 * @property {string=} initiator
 *   The turn's origin label: "consciousness" on every frame and row of a self-initiated wake-up (and the roots it starts); absent on an owner's turn.
 * @property {boolean=} cancelable
 *   v6.82 (P5): host-attested — this frame's task is a supervisor-queue task that
 *   POST /api/tasks/{id}/cancel can force-cancel: a lineage-resolved pooled root or
 *   the live in-process direct-chat turn (stopped cooperatively through the same
 *   ownership seam); never a subagent frame or an ephemeral decision turn.
 * @property {?number=} accounted_upper_bound_usd
 *   C2: an accounted upper bound, not a settled receipt; null when unknown (ABI-3 removed the `cost_usd` alias).
 * @property {?number=} accounted_upper_bound_usd_with_children
 *   C2: subtree upper bound (formerly aliased `cost_usd_with_children`); null when unknown.
 * @property {"available"|"unavailable"=} cost_accounting_status
 * @property {string=} cost_accounting_error
 * @property {boolean=} cost_final
 * @property {boolean=} cost_with_children_partial
 * @property {?number=} reserved_usd
 * @property {?number=} unresolved_upper_bound_usd
 * @property {?number=} unknown_unmetered
 * @property {?number=} non_final_rows
 *   v6.87.48: the count of OPEN ledger rows — the disclosed cause of `cost_final: false`,
 *   which can hold with every dollar bucket at zero (an estimated $0.00, or a dispatched
 *   row whose reservation is exactly zero).
 * @property {?CostPresentation=} cost_presentation
 *   #498: the facts that EXPLAIN the amount beside it, bound to the scope whose ledger
 *   rows produced them (`own` or `root_tree`). `tracked_amount` is null unless a priced
 *   or bounded row actually evidenced it, so an empty ledger and an all-unpriced one
 *   never read as a measured zero. Null when the ledger could not be read.
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
 *   Additive bounded-findings keys: actors[].findings (disclosed rows
 *   {id?, severity?, verdict?, item?, summary?, evidence?, reason?,
 *   recommendation?} — redacted, each string
 *   bounded with an explicit omission marker, at most 8 rows per actor) and
 *   actors[].findings_omitted (exact count, 0 included). Both are emitted only
 *   when that reviewer produced a parsed response; their absence is a
 *   transport/parse hole, never "zero findings". panels[].late_settlement
 *   ({note, settled_after_terminal, settled_at, reviewed_subject, reviewed_superseded,
 *   reviewed_revision: "delivered"|"different"|"unknown", reviewed_is_emitted: true|false|null,
 *   reviewer_outputs, emitted_answer}) binds original critique and subject to emitted bytes.
 *   reviewer_outputs[].response_ref and emitted receipts' source_ref retain full sources;
 *   emitted_answer.state is "delivered"|"owed"|"unknown", with delivered receipts,
 *   unverified ids and owed ids. Receipt basis "send_handler_returned" proves producer
 *   return, not human receipt. The Reviews group prints the note verbatim. `acceptance_incident`
 *   ({incident_id, status: "failed"|"resolved", stage, attempts, source_known,
 *   feedback_delivered, failure_kind?, failure_detail?, retry?, prior_incidents?})
 *   is the host's own LOCAL acceptance-preparation failure — published even when
 *   there is no panel at all, keyed by its stable incident id, with the REAL host
 *   attempt count; absent when no preparation ever failed. The Reviews group is
 *   its only carrier (no card row, no toast); a `resolved` status clears the
 *   active warning and keeps the row as history.
 * @property {boolean=} worker_saturation_warning
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {Object=} transport
 * @property {string=} system_type
 * @property {"timeline"|"reviews"=} card_row
 *   A host-stamped placement fact for a task-keyed System row: "timeline" = a timeline item of the task's card, "reviews" = the card's Reviews group carries the fact (the row is still attached to the card); absent = an ordinary row.
 * @property {string=} card_row_id  // the row's stable identity across live delivery, outbox replay and history
 * @property {number=} card_row_revision  // canonical source order, independent of delivery timestamp
 * @property {"retained_work"=} terminal_receipt  // a placed row the host types as a terminal receipt (Wrap up's saved work), not progress
 * @property {Object=} late_evidence
 *   Late-review identity, reviewed revision and exact applied source_ref served by
 *   taskSourceDownloadUrl; not an original reviewer transcript or a copy of the row.
 * @property {string=} target_label
 * @property {string=} project_id
 * @property {string=} task_name  // structured work title on Project creation/transfer entries
 * @property {string=} project_name
 * @property {string=} handoff_id  // immutable origin/destination receipt identity
 * @property {Object=} terminal_time  // host-owned occurrence; ts remains publication time
 * @property {string=} completion_answer  // a Project root's model-authored final answer, mirrored into Main
 * @property {string=} completion_reason One recorded nonclean-outcome fact beside the answer.
 * @property {Object=} completion_notice Transport view: neutral/detail/phase/salvage; Main adds cause and undescribed codes.
 * @property {string=} delivery_id Stable logical publication identity across transport retries.
 * @property {string=} set_at  // a `reminder` row: when its words were written (live frame only; the text carries the signature)
 * @property {string=} scheduled_for  // a `reminder` row: the due point it was written for
 * @property {string=} delivered_at  // a `reminder` row: when the host showed it (later than due after downtime)
 * @property {number=} chat_id
 * @property {boolean=} project_thread  // server-stamped: chat_id is a reserved Project thread; Main never adopts it even before projectChatIds learns the project
 * @property {string=} row_address
 * @property {Object=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {string=} kind
 */
/**
 * @typedef {Object} TypingOutbound
 * @property {"typing"} type
 * @property {string} action
 * @property {number=} chat_id  // multi-project: routes the indicator to the owning panel
 * @property {boolean=} project_thread  // server-stamped: chat_id is a reserved Project thread; Main never adopts it even before projectChatIds learns the project
 * @property {string=} activity_id
 * @property {string=} client_message_id
 * @property {string=} phase
 * @property {string=} kind  // direct_chat | managed_task, empty for children and untracked tasks; kept for wire compatibility, no in-repo client reads it. A typing frame is a submission receipt, never liveness: only the /api/state census inserts into the header live-set
 */
/**
 * @typedef {Object} PhotoOutbound
 * @property {"photo"} type
 * @property {"user"|"assistant"} role
 * @property {string} image_base64
 * @property {string} mime
 * @property {string} ts
 * @property {string=} caption
 * @property {string=} download_url  // durable task-artifact URL, replayed by chat history
 * @property {string=} download_url_compat  // same bytes on /api/files/download; host-bridge form for launchers whose gate predates the artifact route
 * @property {string=} content
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {Object=} transport
 * @property {number=} chat_id
 * @property {string=} task_id
 * @property {boolean=} project_thread  // server-stamped: chat_id is a reserved Project thread; Main never adopts it even before projectChatIds learns the project
 * @property {string=} row_address
 * @property {Object=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {string=} kind
 */
/**
 * @typedef {Object} VideoOutbound
 * @property {"video"} type
 * @property {"user"|"assistant"} role
 * @property {string} video_base64
 * @property {string} mime
 * @property {string} ts
 * @property {string=} caption
 * @property {string=} download_url  // durable task-artifact URL, replayed by chat history
 * @property {string=} download_url_compat  // same bytes on /api/files/download; host-bridge form for launchers whose gate predates the artifact route
 * @property {string=} content
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {Object=} transport
 * @property {number=} chat_id
 * @property {string=} task_id
 * @property {boolean=} project_thread  // server-stamped: chat_id is a reserved Project thread; Main never adopts it even before projectChatIds learns the project
 * @property {string=} row_address
 * @property {Object=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {string=} kind
 */
/**
 * @typedef {Object} LinkAction
 * @property {string} label
 * @property {string} url
 */

/**
 * @typedef {Object} LinksOutbound
 * @property {"links"} type
 * @property {"assistant"} role
 * @property {LinkAction[]} actions
 * @property {string} ts
 * @property {string=} title
 * @property {number=} chat_id
 * @property {string=} task_id
 * @property {boolean=} project_thread
 * @property {Object=} transport
 * @property {string=} row_address
 * @property {Object=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {string=} kind
 */

/**
 * @typedef {Object} QuizOption
 * @property {string} label
 * @property {string=} detail
 * @property {boolean=} recommended
 */

/**
 * @typedef {Object} QuizOutbound
 * @property {import('./client_contract_types.js').PresenceContact=} presence_contact
 * @property {string=} wait_ended_at
 * @property {Array<Object>=} materials Validated existing artifact references.
 * @property {{kind:'publication'|'answer'|'closure',state:'owed'|'delivered',reason_code?:string,dismissed?:boolean}=} question_delivery Canonical settlement delivery fact.
 * @property {string=} closed_reason
 * @property {string=} closed_at
 * @property {string=} closed_by_task_id
 * @property {string=} closed_actor
 * @property {string=} closed_request_id
 * @property {"quiz"} type
 * @property {"assistant"} role
 * @property {string} quiz_id
 * @property {string} question
 * @property {QuizOption[]} options
 * @property {string} stake
 * @property {string} assumption
 * @property {boolean=} wait_for_answer
 * @property {string} state
 * @property {string} ts
 * @property {number=} answered_index
 * @property {string=} comment
 * @property {string=} host_facts
 * @property {number=} chat_id
 * @property {string=} task_id
 * @property {boolean=} project_thread
 * @property {Object=} transport
 * @property {string=} row_address
 * @property {Object=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {string=} kind
 */

/**
 * Lifecycle update for an already-rendered quiz card (WS "quiz_state") —
 * a separate discriminator so a state change never dedupes as (or spawns)
 * a second card. answered_index rides only with state "answered".
 * @typedef {Object} QuizStateOutbound
 * @property {{kind:'publication'|'answer'|'closure',state:'owed'|'delivered',reason_code?:string,dismissed?:boolean}=} question_delivery Canonical settlement delivery fact.
 * @property {string=} closed_reason
 * @property {string=} closed_at
 * @property {string=} closed_by_task_id
 * @property {string=} closed_actor
 * @property {string=} closed_request_id
 * @property {"quiz_state"} type
 * @property {string} quiz_id
 * @property {string} task_id
 * @property {string} state
 * @property {string} ts
 * @property {number=} answered_index
 * @property {string=} comment
 *   The owner's recorded free-text answer, when one was recorded.
 * @property {boolean=} wait_for_answer
 *   False once a bounded wait closed and the task resumed; the card stays open and answerable.
 * @property {number=} chat_id
 */

/** @typedef {import('./decision_types.js').DecisionRequest} DecisionRequest */
/** @typedef {import('./decision_types.js').DecisionResponse} DecisionResponse */

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
 * @property {string=} download_url_compat
 * @property {Object=} file_ref
 * @property {string=} content
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {Object=} transport
 * @property {number=} chat_id
 * @property {string=} task_id
 * @property {number=} size_bytes
 * @property {boolean=} project_thread  // server-stamped: chat_id is a reserved Project thread; Main never adopts it even before projectChatIds learns the project
 * @property {string=} row_address
 * @property {Object=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {string=} kind
 */

/**
 * @typedef {Object} LogOutbound
 * @property {"log"} type
 * @property {Object} data
 * @property {number=} chat_id  // multi-project thread routing (v6.32.0); main chat = 1
 * @property {boolean=} project_thread  // server-stamped: chat_id is a reserved Project thread; Main never adopts it even before projectChatIds learns the project
 */

/**
 * @typedef {Object} ProjectsChangedOutbound
 * @property {"projects_changed"} type
 * @property {string=} project_id
 * @property {number=} chat_id  // new project thread; client learns it before /api/state (v6.32.0)
 */

/**
 * @typedef {Object} MessageAnnotationOutbound Bubble-free update for an existing owner message.
 * @property {"message_annotation"} type
 * @property {"routing_ack"} annotation_type
 * @property {number=} chat_id
 * @property {string} client_message_id
 * @property {string} action
 * @property {string=} target
 * @property {string=} target_label
 * @property {string=} project_id
 * @property {number=} project_chat_id
 * @property {string} status
 * @property {Array<Object>=} options
 * @property {AttachmentManifestEntry[]=} attachment_manifest
 * @property {string=} routing_token
 * @property {string=} cause  // host-authored owner sentence for a REFUSED act; absent on scheduled/delivered/pending and on the picker frame
 * @property {string=} reasoning_effort  // the explicit start a New task picked from this picker card requests
 * @property {boolean} suppress_bubble
 * @property {string=} ts
 * @property {string=} project_name
 * @property {string[]=} target_task_ids
 * @property {string[]=} target_titles
 */

/** Additive history fact: a project-owned Main mirror cannot grant cancel authority.
 * @typedef {Object} ProjectMirrorHistoryFields
 * @property {boolean=} project_mirror
 */

/** @typedef {import('./management_contract_types.js').SkillReviewHistoryDetailResponse} SkillReviewHistoryDetailResponse */

/** Additive /api/chat/history terminality projection.
 * @typedef {Object} TaskOutcomeHistoryFields
 * @property {"working"|"done"|"warn"|"error"|"cancelled"=} outcome_phase  // canonical display phase; "working" is not terminal
 * @property {boolean=} outcome_final  // true only after the canonical task outcome settles; false marks a pre-finalization narrative
 * @property {{status: string, phase: string, ts: string, provenance: string, model_execution?: Object}=} historical_terminal
 * @property {Object=} model_execution
 * @property {{v: 1, occurred_at: ?string, source: "executor_terminal"|"unknown", attempt: Object}=} terminal_time  // a task_summary row's host end fact; `ts` stays its publication time
 */

/**
 * POST /api/projects body (v6.59.0). ONE source: path (attach; optional init_git
 * attach-snapshot commit — never auto-init), git_url (server-side clone; typed
 * auth_required), with_workspace (genesis), or none (file-less).
 * @typedef {Object} ProjectCreateRequest
 * @property {string=} id  // `project_id` is accepted as an alias
 * @property {string=} name
 * @property {string=} path
 * @property {boolean=} init_git
 * @property {string=} git_url
 * @property {boolean=} with_workspace
 */

/**
 * A registry project row: list rows are the compact projection (with `has_thread_activity`);
 * a create/update answer is the full row (`created: false`: an idempotent repeat create).
 * @typedef {Object} ProjectEntry
 * @property {string} id
 * @property {string=} name
 * @property {boolean=} pinned  // absent on legacy rows means unpinned
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
 * @property {string=} routing_incarnation
 * @property {number=} visible_revision
 * @property {string=} delete_error
 * @property {boolean=} has_thread_activity
 * @property {boolean=} thread_activity_seen
 * @property {string=} last_task_result_id
 * @property {boolean=} created
 * @property {ProjectCognition} cognition
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
 * @property {string} suggested_name  // admission-coined name of a managed task; client sets the live card title
 */

/**
 * @typedef {Object} UploadResponse
 * @property {boolean} ok
 * @property {string} filename
 * @property {string} display_name
 * @property {string} path
 * @property {number} size
 * @property {string=} sha256
 * @property {string} mime  // the extension's type, as the model-input rail reads it
 * @property {{name: string, kind: ('image'|'video'|'audio'|'file'), mime: string, size: (?number|undefined), available: boolean, url: (string|undefined)}=} view  // ChatAttachmentView (chat_uploads.attachment_view): the sender's own bubble renders exactly this; `kind` proven from bytes, `url` only while available
 */

/**
 * @typedef {Object} StatusResponse
 * @property {string} status
 */

/**
 * A refusal: `error` is a sentence; `code` or `reason_code` types it when present.
 * @typedef {Object} ErrorResponse
 * @property {string} error
 * @property {string=} code
 * @property {string=} reason_code
 */

/**
 * @typedef {Object} HeartbeatOutbound
 * @property {'heartbeat'} type
 * @property {string=} ts
 */

/**
 * `extension_lifecycle`: re-read GET /api/widgets and GET /api/extensions.
 * @typedef {Object} ExtensionLifecycleOutbound
 * @property {'extension_lifecycle'} type
 * @property {string=} skill
 * @property {string=} action
 * @property {string=} status
 * @property {string=} reason
 * @property {Object=} data
 */

/**
 * @typedef {Object} GitLogResponse
 * @property {Object[]} commits
 * @property {Object[]} tags  // {tag, date, sha, message}
 * @property {string} branch
 * @property {string} sha
 */

/**
 * `POST /api/settings` success; `restart_keys` apply only after a restart.
 * @typedef {Object} SettingsSaveResponse
 * @property {string=} status
 * @property {boolean=} no_changes
 * @property {boolean=} restart_required
 * @property {string[]=} restart_keys
 * @property {Object=} restart_state
 * @property {boolean=} immediate_changed
 * @property {boolean=} next_task_changed
 * @property {string[]=} warnings
 * @property {Object<string, string>=} ignored_keys
 * @property {boolean=} agent_task_running
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
 * @property {string} project_id
 * @property {ProjectCognition} cognition
 */

/**
 * @typedef {Object} EffortRange  helper min ≤ max; primary recommended independent; each an EFFORT_SCALE tier
 * @property {string} min
 * @property {string} recommended
 * @property {string} max
 */

/**
 * @typedef {Object} OwnerEffortRangeResponse
 * @property {boolean} ok
 * @property {EffortRange} effort_range
 * @property {string} project_id
 * @property {ProjectCognition} cognition
 */

/**
 * @typedef {Object} ProjectCognition
 * @property {string} context_mode
 * @property {EffortRange} effort_range
 */

/**
 * @typedef {Object} OwnerSafetyModeResponse
 * @property {boolean} ok
 * @property {string} safety_mode  // full | light | off (v6.54.3)
 */

/** @typedef {import('./management_contract_types.js').InstalledSkill} InstalledSkill */

/** @typedef {import('./management_contract_types.js').SkillToggleResponse} SkillToggleResponse */

/** @typedef {import('./management_contract_types.js').SkillReconcileResponse} SkillReconcileResponse */

/**
 * One Widgets card from `GET /api/widgets` (`gateway/widgets.py::WidgetTab`).
 * `revision` is the owning skill's live payload content hash — a change
 * signature for the page, not an ETag or cache token. Frame geometry stays
 * inside `render`.
 * @typedef {Object} WidgetTab
 * @property {string} key
 * @property {string} skill
 * @property {string} tab_id
 * @property {string} title
 * @property {string} icon
 * @property {string} ws_prefix
 * @property {Object} render
 * @property {number} span
 * @property {number} grid_span
 * @property {string} revision
 * @property {boolean=} pinned_card  // a kept living card (skill "_cards"); presses use the card: decision family
 */

/**
 * @typedef {Object} WidgetsResponse
 * @property {WidgetTab[]} ui_tabs
 */

/**
 * @typedef {Object} SkillPublishFinding
 * @property {string} path
 * @property {number} line
 * @property {string} detector
 * @property {"low"|"medium"|"high"|"unknown"} confidence
 * @property {string} reason
 * @property {"not_attempted"} verification
 * @property {"blocker"|"warning"|"audited_false_positive"} disposition
 */

/**
 * @typedef {Object} SkillPublishPreflightResponse
 * @property {boolean} ok
 * @property {string} skill Canonical selected-skill name.
 * @property {string} repository Canonical case-preserving owner/repo from the configured catalog.
 * @property {"ready"|"warnings"|"needs_attention"|"repairable"|"hard_block"} state
 * @property {boolean} publication_ready
 * @property {boolean} task_start_allowed
 * @property {string} snapshot_hash
 * @property {{status?: string, stale?: boolean, profile?: string}} review
 * @property {{status?: string, engine?: string, version?: string, ruleset_sha256?: string}} scanner
 * @property {SkillPublishFinding[]} findings
 * @property {number} omitted_count
 * @property {number} blocker_count
 * @property {number} warning_count
 * @property {number} audited_false_positive_count
 * @property {string} reason_code
 * @property {string} summary
 * @property {string} repair_hint
 */

/** @typedef {import('./management_contract_types.js').SkillReviewResponse} SkillReviewResponse */

/**
 * `POST /api/skills/{skill}/grants`; refusals use `SkillActionRefusal`, slow setup {ok, pending: true, job_id, message}.
 * @typedef {Object} SkillGrantResponse
 * @property {boolean} ok
 * @property {boolean=} pending
 * @property {string=} job_id
 * @property {string=} message
 * @property {string=} skill
 * @property {boolean=} enabled
 * @property {string=} content_hash
 * @property {string[]=} granted_keys
 * @property {string[]=} granted_permissions
 * @property {string=} turn_on_when_ready
 * @property {string=} setup_error
 * @property {string=} extension_action
 * @property {string=} extension_reason
 * @property {string=} load_error
 * @property {Object=} grants
 * @property {string=} process
 * @property {string=} server_reconcile
 * @property {boolean=} live_loaded
 * @property {string=} error
 */

/**
 * @typedef {Object} OwnerSkillPresenceRuntimeRequest
 * @property {string} expected_state_fingerprint
 * @property {{model_slot: ("main"|"light"|null), inline_max_rounds: (number|null)}} runtime_overrides
 * @property {string=} workspace_root
 */

/**
 * @typedef {Object} OwnerSkillPresenceRuntimeResponse
 * @property {boolean} ok
 * @property {string} skill
 * @property {Object} presence_runtime
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
 * @property {string=} title Owner-facing run name; omitted, admission derives one from the description's first line.
 * @property {number=} chat_id
 * @property {number=} depth
 * @property {string=} session_id
 * @property {string=} workspace_root
 * @property {"external"=} workspace_mode
 * @property {"forked"|"empty"|"shared"=} memory_mode
 * @property {string=} project_id Per-project facts scope id (else derived from the workspace path).
 * @property {Object[]=} attachments
 * @property {boolean=} allow_partial_attachments Explicit raw-API opt-in; browser/UI task admission remains atomic.
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
 * @property {number=} execution_timeout_sec Execution seconds excluding queue and confirmed pauses; 0 adds no limit. Calendar timeout/deadline stays independent.
 * @property {string=} context
 * @property {string=} expected_output
 * @property {string=} constraints
 * @property {boolean=} context_requires_self_body_docs
 * @property {string=} reasoning_effort Optional explicit starting effort of this root (a server-validated effort tier); omitted = the Task default; metadata.reasoning_effort is refused.
 * @property {string=} actor_id Top-level task actor/provenance id; metadata.actor_id is reserved.
 * @property {string=} source Top-level task source/provenance label.
 * @property {Object=} metadata Arbitrary task metadata; executor_ref/workspace_executor keys are reserved.
 */

/**
 * @typedef {Object} TaskCreateResponse
 * @property {boolean} ok
 * @property {string} task_id
 * @property {string} status
 * @property {string=} reason_code
 * @property {string=} error
 * @property {AttachmentManifestEntry[]=} attachment_manifest
 * @property {Object=} attachment_manifest_ref
 */

/**
 * @typedef {Object} AttachmentManifestEntry
 * @property {number} ordinal
 * @property {"staged"|"rejected"} status
 * @property {string} reason
 * @property {string} label
 * @property {string=} root
 * @property {string=} relpath
 * @property {string=} abs_path
 * @property {string=} mime
 * @property {boolean=} is_image
 * @property {number=} size
 * @property {string=} sha256
 * @property {string=} rule
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
 * @property {string=} event_id
 * @property {TaskEventCursor=} cursor
 * @property {string=} reason
 * @property {string=} error
 */
/**
 * @typedef {Object} TaskEventCursor
 * @property {number} v
 * @property {number} seq
 * @property {string} view
 * @property {Object<string, Object<string, number>>} positions
 */
/**
 * @typedef {Object} TaskEventsRequest
 * @property {number} v
 * @property {number=} wait
 * @property {?TaskEventCursor=} cursor
 */
/**
 * @typedef {Object} TaskListResponse
 * @property {Object[]} tasks
 * @property {Object=} queue
 * @property {string=} error
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
 * @property {Object.<string,QuestionRoom>=} question_rooms Current display proof keyed by exact owner_quiz id.
 * @property {Array<{name:string, path?:string, relpath?:string, size?:number, measured?:boolean, status?:string, errors?:string[]}>=} artifacts
 *   Recorded result rows; a nested file keeps its store-relative `relpath`, and `measured: false`
 *   marks a stat-only listing, not a verified capture.
 * @property {Object.<string, {name:string, files:number, size:number, excluded:number, available:boolean}>=} artifact_archives
 *   Per top-level result directory, what `?archive=<dir>` would stream now (members from one
 *   confined stat each, rows left out counted); a folder offers its `.zip` only when `available`.
 * @property {{name:string, request_id:string}=} wrap_up_document  Wrap up's saved-work document, written on read by `?wrap_up=<request_id>` (`taskWrapUpDocumentUrl`); no size or digest before download.
 * @property {Object.<string,Object>=} model_waits
 * @property {TaskCostBreakdown=} cost_breakdown
 * @property {{status: 'pending'|'problem'|'complete', pending_count: number, problem_count: number, promoted_ref_count: number, promoted_source_handle_count: number, problem_reasons?: Array<{reason:string,count:number}>}=} history_retention
 *   Background history placement, separate from task outcome. Routine progress is detail-only;
 *   only a problem appears on the collapsed card.
 * @property {string=} cancel_state
 * @property {string=} cancel_reason
 * @property {string=} stop_policy
 * @property {OwnerHurryProjection=} owner_hurry
 * @property {OwnerHurryProjection[]=} owner_hurry_history
 * @property {{reason?:string, detail?:string, label?:string}=} project_admission_hold  // while the queue snapshot lists the row: its wait for original Project/scope evidence, or {}
 * @property {string=} error
 * @property {ContinuationOffer=} continuation_offer
 */
/**
 * Current question display proof; never a publication address or contextual Project association.
 * @typedef {Object} QuestionRoom
 * @property {'known'|'unknown'} state
 * @property {number=} chat_id
 * @property {string=} project_id
 */
/**
 * PROVENANCE for each independent facet of GET /api/claudexor/status. An empty collection
 * cannot say whether the lazily started daemon was ASKED: an idle machine's empty lists
 * read as "no account connected" while real accounts sat in the agent home.
 * "ok" — read, the matching collection is AUTHORITATIVE (empty means empty);
 * "not_read" — never asked: no daemon, or discovery/handshake died before the
 * fan-out (which leaves every facet untouched); "failed" — asked, and no
 * usable answer came back (refused, or a body in the wrong shape).
 * Facets are independent: one fanned-out read can fail while its siblings land.
 * @typedef {"ok"|"not_read"|"failed"} ClaudexorReadState
 */
/**
 * Independent facets: ok=authoritative, not_read=never asked, failed=no usable answer. Empty without ok proves nothing.
 * @typedef {Object} ClaudexorStatusReads
 * @property {ClaudexorReadState} catalog
 * @property {ClaudexorReadState} accounts
 * @property {ClaudexorReadState} quota
 */
/**
 * Last settled external leaf projected for the Available-subagents editor.
 * `selected_subagent_id` is optional only for pre-migration receipts, which
 * cannot be truthfully attached to a current row.
 * @typedef {Object} SubagentLastDelegation
 * @property {string=} selected_subagent_id
 * @property {string=} route
 * @property {string=} requested_model
 * @property {string=} applied_model
 * @property {string=} requested_profile
 * @property {string=} applied_profile
 * @property {Object=} observed_route Actual API attempt route; never the task's mutable last route.
 * @property {string=} run_id
 * @property {string=} ts
 * @property {string=} occurred_at
 * @property {string=} observed_at
 * @property {string=} outcome
 * @property {string=} failure_code
 * @property {string=} reset_at
 * @property {Object=} identity
 * @property {Object<string, SubagentLastDelegation>=} latest_by_subagent
 * @property {string=} task_id
 * @property {string=} invocation_id
 * @property {string=} attempt_id
 * @property {Object=} fallback
 */
/**
 * @typedef {Object} ClaudexorStatusResponse
 * @property {Object=} daemon
 * @property {string=} config_dir
 * @property {Array<Object>=} harnesses
 * @property {Object=} profiles
 * @property {Array<Object>=} quota
 * @property {Array<Object>=} quota_absences
 * @property {Array<Object>=} resources Engine resource facets retain decimal strings, units and independent observation times.
 * @property {Object=} availability_scope Engine binding: model (string|null), requestKind (text|image), basis quota_block.
 * @property {Object<string, boolean>=} resource_capabilities Catalog-negotiated read, refresh, reset and inspect_reset operations.
 * @property {ClaudexorReadState=} resource_capabilities_read Operations-catalog evidence, independent of reads.catalog (agent capabilities).
 * @property {ClaudexorStatusReads=} reads
 * @property {boolean=} unified_accounts
 * @property {Object<string, {observed_at: ?string, stale: boolean, error: ?string}>=} facets Per facet; a failed or unasked facet serves its last read, stale.
 * @property {SubagentLastDelegation=} subagent_last_delegation
 * @property {string=} error
 */
/**
 * @typedef {Object} ClaudexorPassiveReadError
 * @property {string} code
 * @property {number=} status_code
 */
/**
 * @typedef {Object} ClaudexorQuotaResponse
 * @property {'quota'} view
 * @property {Object} profiles
 * @property {Array<Object>} quota
 * @property {Array<Object>} quota_absences
 * @property {Array<Object>=} resources
 * @property {Object=} availability_scope
 * Engine echo: model (string|null), requestKind (text|image), basis quota_block.
 * @property {boolean} unified_accounts
 * @property {ClaudexorStatusReads} reads
 * @property {Object<'discovery'|'accounts'|'quota', ClaudexorPassiveReadError>} read_errors
 * @property {Object<string, number>} timings_ms
 */
/**
 * One required bare daemon job per operation. Create/input/snapshot metadata and deviceCode stay beside it; attach commands need the proven packaged role.
 * @typedef {Object} ClaudexorLoginJobResponse
 * @property {Object} job
 * @property {string=} cursor
 * @property {number=} sequence
 * @property {Object=} deviceCode
 * @property {string=} job_id
 * @property {boolean=} disclosure_native
 * @property {('per_harness'|'setup_job_admission'|'legacy_global_operation')=} setup_login_source
 * Present only after the exact serving package advertises setup_attach.
 * @property {string=} attach_command
 * @property {('posix'|'powershell')=} attach_shell
 * @property {boolean=} ok
 */
/**
 * Typed engine job error; marked retryable probe 503 and bounded engine actions pass through.
 * @typedef {Object} ClaudexorLoginJobProblem
 * @property {string} error
 * @property {string=} code
 * @property {Array<string>=} required_actions
 */
/**
 * @typedef {Object} ClaudexorVendorCredentialDisposition
 * @property {'vendor'} owner
 * @property {'left_unchanged'} state
 * @property {'os_user'} scope
 */
/**
 * Exact daemon receipt from deleting one credential-profile binding.
 * @typedef {Object} ClaudexorCredentialProfileDeleteResponse
 * @property {Object} profile
 * @property {boolean} removed
 * @property {('config_dir_removed'|'secret_deleted'|'none')} credentialCleanup
 * @property {string=} cleanupWarning
 * @property {ClaudexorVendorCredentialDisposition=} vendorCredentialDisposition
 */
/**
 * Mirrors `gateway/schedule_contracts.py`, which states what each field means.
 * @typedef {Object} ScheduledTasksResponse
 * @property {number} schema_version
 * @property {Object[]} tasks  // each row carries status/retained/restorable
 */
/**
 * @typedef {Object} ScheduleUpsertResponse
 * @property {boolean} ok  // follows schedule.audit: an incomplete audit is not ok
 * @property {Object} schedule
 */
/**
 * @typedef {Object} ScheduleActionResponse
 * @property {boolean} ok  // the requested state was ACHIEVED and both audit records landed
 * @property {boolean} changed  // the durable fact, whatever the audit did
 * @property {string} status
 * @property {string} schedule_id
 * @property {string=} operation_id
 * @property {?boolean=} running_or_queued  // already admitted; null = unknown
 * @property {('recorded'|'incomplete'|'not_written')} audit
 * @property {string=} detail
 * @property {Object=} schedule
 * @property {string[]=} allowed
 */
/**
 * Legacy DELETE response: the ScheduleActionResponse subset read by previous callers.
 * @typedef {Object} ScheduleDeleteResponse
 * @property {boolean} ok
 */
/** @typedef {import('./task_activity_types.js').TaskPauseRequest} TaskPauseRequest */
/** @typedef {import('./task_activity_types.js').TaskPauseResponse} TaskPauseResponse */
/**
 * @typedef {Object} TaskContinueRequest
 * @property {string} action_nonce
 */
/** @typedef {import('./task_activity_types.js').TaskContinueResponse} TaskContinueResponse */
/** @typedef {import('./task_activity_types.js').ContinuationOffer} ContinuationOffer */
/**
 * Reuse stop_action_id (at most 200 characters) for this exact action; a later
 * Stop needs a new ID, distinct from server request_id. No ID means no exact retry.
 * @typedef {Object} TaskCancelRequest
 * @property {boolean=} cascade
 * @property {string=} stop_policy
 * @property {string=} stop_action_id
 */
/**
 * @typedef {Object} TaskCancelResponse
 * @property {boolean} ok
 * @property {string} task_id
 * @property {boolean=} cascade
 *   Echoed when the Stop ran as a cascade (requested, or an already recorded cascade scope); single-task envelope unchanged.
 * @property {string=} cancel_state
 *   "pending" on every 202: the durable intent stays open — bounded graceful finalization, or an immediate
 *   teardown not confirmed yet that the supervisor finishes; absent on 200.
 * @property {string=} stop_policy
 *   On 202, the effective durable "immediate" | "finalize_then_cancel"; a graceful request never softens a hard intent.
 * @property {string=} request_id
 *   On an immediate 202, the open intent's custody id (distinct from the client's stop_action_id).
 * @property {string=} reason_code
 *   On an immediate 202, "teardown_unconfirmed".
 * @property {string=} error
 */
/**
 * POST /api/tasks/{task_id}/hurry: no chat message. Only client request_id is
 * accepted, stable across retries; every other body field is refused.
 * @typedef {Object} TaskHurryRequest
 * @property {string} request_id
 */
/**
 * Task-detail owner_hurry; same-ID requeues archive this shape with archived_at/archived_reason.
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
 * Same request/live-attempt or already armed latch returns one acknowledgement with duplicate=true.
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
 * @property {string=} error
 */
/**
 * `POST /api/skills/{skill}/delete`; a refusal carries only ok, error and status_code.
 * @typedef {Object} SkillDeleteResponse
 * @property {boolean} ok
 * @property {boolean=} pending  // queued behind a long job past the answer bound: {ok, pending, job_id, message}
 * @property {string=} job_id
 * @property {string=} message
 * @property {string=} skill
 * @property {string=} source
 * @property {string=} content_hash
 * @property {string=} deleted_payload_root
 * @property {boolean=} deleted_state
 * @property {string=} extension_action
 * @property {string=} extension_reason
 * @property {string=} error
 */
/**
 * @typedef {Object} UiPreferencesResponse
 * @property {{stt:string,tts:string,tts_voice:string}} voice
 * @property {{hidden:string[],order:string[]}} contributions
 * @property {string[]} widget_order
 * @property {Object.<string,'auto'|'manual'|'retain'>} widget_start_mode  // owner per-card launch-policy override, keyed "<skill>:<tab_id>"
 * @property {Object.<string,{w:number,h:number}>} widget_size  // owner Widgets card width: w masonry columns the card spans (12 = full width), h 0 (reserved)
 * @property {boolean} nested_subagents_expanded
 * @property {number} sidebar_width  // px; 0 = CSS default (v6.33.0)
 * @property {number} project_panel_width  // px; 0 = CSS default
 * @property {Object.<string,number>} project_seen_revision  // monotonic paint ACK
 * @property {?number} main_seen_revision  // Main's monotonic paint ACK; null until a client first reads Main
 * @property {{mode:'default'|'hidden'|'custom',text:string}} welcome  // install-wide empty-Main UI copy, not chat history
 * @property {boolean=} ok
 */
/**
 * Host sign-in registration as its OS reports it, or (/api/desktop/background) the keep-running choice; `reason` only when unavailable.
 * @typedef {Object} DesktopAutostartResponse
 * @property {'unavailable'|'off'|'on'|'other_copy'|'disabled_by_os'} state
 * @property {string=} reason
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
 * @property {string=} keep_branch
 */

/**
 * @typedef {Object} UpdateApplyErrorResponse
 * @property {string} error
 * @property {string=} reason
 * @property {boolean=} abandoned
 * @property {Object=} status Preparation observations, not the success discriminator.
 * @property {Object=} update_intent
 * @property {Object=} rescue
 * @property {string=} keep_branch
 * @property {string[]=} blockers
 * @property {boolean=} rolled_back
 * @property {string=} rollback
 * @property {boolean=} restart_required
 * @property {UpdateMergePlan=} merge_plan
 * @property {Object=} smoke
 * @property {string=} stash_note Stash-first prologue disclosure: how stashed local work was unwound.
 * @property {?number=} estimated_wave_usd Wave-floor admission estimate (worst-case review-pack caps).
 * @property {?number=} remaining_usd Remaining model budget the floor compared against.
 */

/** @typedef {import('./management_contract_types.js').UpdateProgress} UpdateProgress */

/**
 * @typedef {Object} UpdateProgressChangedOutbound
 * @property {'update_progress_changed'} type
 */

/**
 * @typedef {Object} UpdateStatusReadyOutbound
 * @property {'update_status_ready'} type
 * @property {boolean} available
 * @property {?boolean} check_ok
 */

/**
 * LLM-written update letter in `/api/update/status` and `/api/update/check`.
 * The additive `letter` is absent/null without a stored letter. It outlives its update;
 * `relation` compares startup source to target so the panel can relabel it instead of deleting it.
 *
 * @typedef {Object} UpdateLetter
 * @property {'ready'|'failed'} state
 * @property {'pending'|'applied'|'superseded'|'other'} relation  offered now / included in running source / newer target appeared / source moved elsewhere
 * @property {string} text  markdown; may be empty when a failed write has no previous good letter
 * @property {string} author_version  the Ouroboros version that wrote it
 * @property {string} target_version  the version it describes
 * @property {string} written_at  ISO 8601
 * @property {''|'no_credentials'|'budget_exhausted'|'context_overflow'|'timeout'|'material_unavailable'|'output_truncated'|'provider_unavailable'|'empty_response'|'runtime_source_unavailable'} error_kind
 * @property {string} error_text  short, secret-free; empty when state is ready
 * @property {{base_sha: string, target_sha: string, update_channel: string, target_ref: string}} key  the exact range the letter was written for
 * @property {boolean} has_last_good  `text` is the previous good letter kept through a failed rewrite; `relation`, `key` and the provenance describe THAT letter, not the range that failed
 * @property {boolean} description_current  successful shown text covers the exact running-source to checked-target range
 * @property {string} failed_at  ISO 8601 time of the failed attempt, independent of the shown text's written_at
 * @property {({base_sha: string, target_sha: string, update_channel: string, target_ref: string}|null)} latest_failed_key  the failed attempt's range, never the provenance of retained text
 */

export const MAX_LINK_ACTIONS = 12;
export const MAX_QUIZ_OPTIONS = 6;
// Mirror task_decision._COMMENT_MAX: ingress refuses longer comments, never
// truncates; cards must offer only comments the ingress can deliver verbatim.
export const MAX_DECISION_COMMENT = 2000;
export const GATEWAY_CONTRACT_VERSION = '7.8.0';
/** @typedef {import('./client_contract_types.js').ChatHistoryPosition} ChatHistoryPosition */
/** @typedef {import('./client_contract_types.js').ChatHistoryResponse} ChatHistoryResponse */

/** @typedef {import('./management_contract_types.js').UiContribution} UiContribution */
/** @typedef {import('./management_contract_types.js').UiContributionsResponse} UiContributionsResponse */
