/** Dependency-free JSDoc mirror of `ouroboros.gateway.management_contracts`: skills and extensions,
 *  the marketplaces, agent accounts (relayed Claudexor engine objects), local model, MCP, the model
 *  catalog, updates and core control. `UpdateLetter` sits in ./api_types.js. */

/**
 * One `GET /api/extensions` `skills` row; a name collision gives one row per occupant (`identity_collision`).
 * @typedef {Object} InstalledSkill
 * @property {string} name
 * @property {string} type
 * @property {string} version
 * @property {string} description
 * @property {boolean} enabled
 * @property {string[]} permissions
 * @property {string[]} conflicts
 * @property {?string} load_error
 * @property {boolean} is_self_authored
 * @property {string} source  // native | clawhub | ouroboroshub | external | self_authored | user_repo
 * @property {string} content_hash
 * @property {string} payload_root
 * @property {string} installed_at
 * @property {string=} removal  // uninstall | delete | ''
 * @property {string=} review_status
 * @property {boolean=} review_stale
 * @property {{status: string, stale: boolean, executable_review: boolean, blocking_reason: string, review_enforcement: string, summary: string, author_accepted: (boolean|undefined), reviewed_content_hash: (string|undefined), author_disposition: (Object|undefined), preflight_failed: (boolean|undefined), preflight_failed_stale: (boolean|undefined)}=} review_gate
 * @property {Object=} author_disposition
 * @property {string=} reviewed_content_hash
 * @property {boolean=} executable_review
 * @property {string=} review_profile
 * @property {?boolean=} owner_attestable
 * @property {{visible: boolean, publication_ready: boolean, task_start_allowed: boolean, disabled: boolean, state: string, reason: string}=} submit_hub
 * @property {?{code: "skill_conflict", skills: string[], omitted: number}=} conflict
 * @property {boolean=} desired_live
 * @property {boolean=} live_loaded
 * @property {string=} live_reason
 * @property {string=} process
 * @property {string=} server_reconcile
 * @property {boolean=} health_regressed
 * @property {?Object=} last_known_good
 * @property {Object=} health_observations
 * @property {boolean=} dispatch_live
 * @property {Object[]=} review_findings
 * @property {{current: Object, history: Object[], history_omitted: (number|undefined)}|Object=} skill_review
 * @property {Object=} grants
 * @property {?{slug: string, version: string, content_hash: string, repository: string, pr_number: number, pr_url: string, published_at: string}=} published
 * @property {boolean=} published_malformed
 * @property {Object=} presence_runtime
 * @property {Object=} provenance
 * @property {boolean=} identity_collision
 * @property {?Object=} install  // source, official (exact endorsed bytes), release, basis, base_known, modified, update_available, withdrawn ({reason, version} for a withdrawn release), left_source, turn_on_when_ready, installed_by, installed_at, reason?, task_id?, check?: {by, notes, at}, pending_update?: {version, digest}, update_rejected?: {version, digest, reason: review_blockers|review_unfinished|preflight_failed, at}
 * @property {string[]=} category
 */

/**
 * `GET /api/extensions/{skill}/manifest`.
 * @typedef {Object} ExtensionManifestResponse
 * @property {string} name
 * @property {Object} manifest
 * @property {boolean} enabled
 * @property {string} content_hash
 * @property {?string} load_error
 * @property {string=} review_status
 * @property {boolean=} review_stale
 * @property {Object=} review_gate
 * @property {Object=} author_disposition
 * @property {string=} reviewed_content_hash
 * @property {boolean=} executable_review
 * @property {string=} review_profile
 * @property {?boolean=} owner_attestable
 * @property {Object=} submit_hub
 */

/**
 * @typedef {Object} ExtensionSettingsSectionResponse
 * @property {string} skill
 * @property {Object[]} sections  // {skill, section_id, title, render, key}
 */

/**
 * `POST /api/skills/{skill}/toggle`.
 * @typedef {Object} SkillToggleRequest
 * @property {boolean|number|string} enabled
 * @property {string=} expected_content_hash
 */

/**
 * @typedef {Object} SkillRevisionRequest
 * @property {string=} expected_content_hash
 */

/**
 * @typedef {Object} SkillGrantRequest
 * @property {Array<string|Object>=} items
 * @property {Array<string|Object>=} keys
 * @property {Array<string|Object>=} granted_keys
 * @property {string=} expected_content_hash
 * @property {boolean=} enable_after_grant
 */

/**
 * @typedef {Object} SkillDeleteRequest
 * @property {string=} payload_root
 * @property {string=} expected_content_hash
 */

/**
 * @typedef {Object} SkillToggleResponse
 * @property {string} skill
 * @property {boolean=} ok
 * @property {boolean=} pending  // enabling still installs packages: follow job_id on the lifecycle queue; no enabled yet
 * @property {string=} job_id
 * @property {string=} message
 * @property {boolean=} enabled
 * @property {string=} review_status
 * @property {boolean=} review_stale
 * @property {Object=} review_gate
 * @property {boolean=} executable_review
 * @property {Object=} grants
 * @property {?string=} extension_action
 * @property {?string=} extension_reason
 * @property {string=} process
 * @property {string=} server_reconcile
 */

/**
 * A refused owner action on one skill; the HTTP status repeats `status_code`.
 * @typedef {Object} SkillActionRefusal
 * @property {boolean=} ok
 * @property {string} error
 * @property {string=} code
 * @property {string=} detail
 * @property {number=} status_code
 * @property {string=} skill
 * @property {string=} source
 * @property {string=} content_hash
 * @property {string=} review_status
 * @property {boolean=} review_stale
 * @property {Object=} review_gate
 * @property {boolean=} executable_review
 * @property {Object=} grants
 * @property {Object=} conflict
 * @property {string=} deps_status
 * @property {boolean=} enabled
 * @property {?string=} extension_action
 * @property {?string=} extension_reason
 * @property {string=} process
 * @property {string=} server_reconcile
 * @property {?string=} load_error
 * @property {string[]=} allowed_keys
 * @property {string[]=} allowed_permissions
 * @property {string[]=} rejected_items
 */

/**
 * @typedef {Object} SkillDaemonsResponse
 * @property {Object.<string,Object>} companions
 */

/**
 * One job of the skill lifecycle lane.
 * @typedef {Object} SkillLifecycleJob
 * @property {string} id
 * @property {string} kind
 * @property {string} target
 * @property {string=} source
 * @property {string} status  // queued | running | succeeded | failed | cancelled
 * @property {string=} message
 * @property {string=} error
 * @property {string=} dedupe_key
 * @property {number=} chat_id
 * @property {string=} group_id
 * @property {string=} task_id
 * @property {string=} root_task_id
 * @property {string=} origin_task_id
 * @property {string=} origin_root_task_id
 * @property {string=} presentation_owner_task_id
 * @property {string=} chat_task_id
 * @property {string=} queued_at
 * @property {string=} started_at
 * @property {string=} finished_at
 * @property {number=} age_seconds
 * @property {boolean=} stale
 * @property {string=} stale_reason
 * @property {string=} recovery_hint
 */

/**
 * GET /api/skills/{skill}/review-history/{job_id}: one terminal review rendered server-side.
 * @typedef {Object} SkillReviewHistoryDetailResponse
 * @property {string} markdown
 * @property {string} status
 * @property {string} content_hash
 * @property {string} job_status
 */

/**
 * @typedef {Object} SkillReconcileResponse
 * @property {string} skill
 * @property {?string} extension_action
 * @property {?string} extension_reason
 * @property {boolean} live_loaded
 * @property {?string} load_error
 * @property {string} process
 * @property {string} server_reconcile
 */

/**
 * Skill review and owner attestation outcome (`gateway/management_contracts.py::SkillReviewResponse`).
 * @typedef {Object} SkillReviewResponse
 * @property {string} skill
 * @property {string} status  // clean | warnings | blockers | pending
 * @property {string=} content_hash
 * @property {string=} error
 * @property {boolean=} ok  // attest only
 * @property {number=} status_code  // attest refusal only
 * @property {boolean=} pending  // attest: the job outlasted the answer bound; review: the wave is still open (its author left on an addressed message)
 * @property {string=} message  // attest only, with pending
 * @property {string[]=} reviewer_models
 * @property {string=} review_profile
 * @property {Object[]=} findings
 * @property {Object[]=} raw_actor_records
 * @property {*=} advisory_result
 * @property {Object=} review_gate
 * @property {boolean=} executable_review
 * @property {*=} auto_flow
 * @property {string[]=} auto_granted_keys
 * @property {string[]=} requested_keys
 * @property {string[]=} auto_granted_permissions
 * @property {string[]=} requested_permissions
 * @property {string=} deps_status
 * @property {string=} deps_error
 * @property {string=} extension_action
 * @property {string=} extension_reason
 * @property {string=} extension_process
 * @property {string=} extension_server_reconcile
 * @property {string|null=} extension_load_error
 * @property {boolean|null=} extension_live_loaded
 * @property {string=} record_id  // the review's ledger record
 * @property {boolean=} reused  // a settled record of the same subject answered at $0
 * @property {boolean=} paid
 * @property {Object[]=} seats
 * @property {Object=} governance
 * @property {string=} job_id
 * @property {string=} job_status
 * @property {string=} group_id
 * @property {number=} review_round
 * @property {number=} snapshot_attempt
 * @property {boolean=} snapshot_revised
 * @property {string=} task_id
 * @property {string=} root_task_id
 * @property {string=} origin_task_id
 * @property {string=} origin_root_task_id
 * @property {string=} presentation_owner_task_id
 * @property {number=} chat_id
 * @property {string=} source
 * @property {string=} terminal_reason
 * @property {Object[]=} executions
 */

/**
 * A catalog row (ClawHub registry fields; OuroborosHub adds `source`, `sanitized_name`, `identity_conflict`).
 * @typedef {Object} MarketplaceSkillSummary
 * @property {string} slug
 * @property {string=} display_name
 * @property {string=} summary
 * @property {string=} description
 * @property {string=} latest_version
 * @property {Array=} versions
 * @property {string=} license
 * @property {string=} homepage
 * @property {string[]=} os
 * @property {string[]=} requires_env
 * @property {string[]=} requires_bins
 * @property {string=} primary_env
 * @property {Array=} install_specs
 * @property {Object=} badges
 * @property {Object=} stats
 * @property {boolean=} is_plugin
 * @property {string=} source
 * @property {string=} sanitized_name
 * @property {boolean=} identity_conflict
 */

/**
 * @typedef {Object} MarketplaceInstalledSkill
 * @property {string} name
 * @property {string=} type
 * @property {string=} version
 * @property {string=} review_status
 * @property {boolean=} review_stale
 * @property {Object=} review_gate
 * @property {boolean=} executable_review
 * @property {Object[]=} review_findings
 * @property {boolean=} enabled
 * @property {?string=} load_error
 * @property {Object=} grants
 * @property {string=} payload_root
 * @property {Object=} provenance
 */

/**
 * @typedef {Object} ClawHubPreviewResponse
 * @property {string} slug
 * @property {string} version
 * @property {MarketplaceSkillSummary} summary
 * @property {boolean} metadata_only
 * @property {Object} adapter
 * @property {Object} staging
 */

/**
 * @typedef {Object} MarketplaceInstallRequest
 * @property {string} slug
 * @property {string=} version
 * @property {boolean=} auto_review
 * @property {boolean=} overwrite
 */

/**
 * @typedef {Object} MarketplaceUpdateRequest
 * @property {string=} version
 */

/**
 * ClawHub install or update result; `ok` stays true when review finds blockers.
 * @typedef {Object} MarketplaceInstallResponse
 * @property {boolean} ok
 * @property {string=} sanitized_name
 * @property {?string=} error
 * @property {string=} target_dir
 * @property {?MarketplaceSkillSummary=} summary
 * @property {Object=} archive
 * @property {Object=} adapter
 * @property {string=} review_status
 * @property {Object[]=} review_findings
 * @property {string=} review_error
 * @property {string=} deps_status
 * @property {string=} deps_error
 * @property {string=} deps_fingerprint
 * @property {Object=} provenance
 */

/**
 * @typedef {Object} MarketplaceUninstallResponse
 * @property {boolean} ok
 * @property {string} sanitized_name
 * @property {?string} error
 */

/**
 * `GET /api/marketplace/clawhub/search`.
 * @typedef {Object} MarketplaceSearchResponse
 * @property {string=} query
 * @property {string=} sort
 * @property {*=} limit
 * @property {number=} offset
 * @property {?string=} cursor
 * @property {?string=} next_cursor
 * @property {boolean=} official
 * @property {string=} registry_path
 * @property {Object[]=} registry_attempts
 * @property {string[]=} registry_warnings
 * @property {boolean=} registry_empty
 * @property {number} count
 * @property {MarketplaceSkillSummary[]} results
 */

/**
 * `GET /api/marketplace/{clawhub|ouroboroshub}/installed`; the hub route remains frozen.
 * @typedef {Object} MarketplaceInstalledResponse
 * @property {number} count
 * @property {MarketplaceInstalledSkill[]} skills
 */

/**
 * @typedef {Object} HubPublicationClearRequest
 * @property {Object} expected_published
 */

/**
 * @typedef {Object} HubPublicationClearResponse
 * @property {boolean} ok
 * @property {string} sanitized_name
 * @property {boolean} publication_cleared
 */

/**
 * An engine refusal relayed with its status.
 * @typedef {Object} ClaudexorProblem
 * @property {string} error
 * @property {string} code
 * @property {string[]=} required_actions
 */

/**
 * Engine or host preparation problem. POST may prepare Node/npm:
 * 409 maintenance_restart_required asks for chat-header Restart;
 * 409/503 maintenance_preparation_failed reports failure. Both carry
 * context.acceptance=not_submitted, scope=this_attempt. Optional inspection failure still
 * permits POST. After 503 acceptance=unknown, keep the same body and Idempotency-Key,
 * even after a later preparation-only reply, then resend both.
 * @typedef {Object} ClaudexorMaintenanceProblem
 * @property {Object} error  // {code, message, retryable?, requiredActions?, context?}
 */

/**
 * @typedef {Object} ClaudexorAccountTarget
 * @property {string} harness
 * @property {string} profile_id
 */

/**
 * Sent with a required Idempotency-Key header.
 * @typedef {Object} ClaudexorMaintenanceRequest
 * @property {string} harness
 * @property {Object} target
 */

/**
 * Engine ControlHarnessMaintenanceOperation, passed through.
 * @typedef {Object} ClaudexorMaintenanceOperation
 * @property {string} id
 * @property {string} state
 * @property {string=} harness
 * @property {string=} phase
 * @property {string=} mechanism
 * @property {Object=} target
 * @property {*=} before
 * @property {*=} after
 * @property {string=} mutation
 * @property {string=} termination
 * @property {*=} limitations
 * @property {*=} progress
 * @property {*=} problem
 * @property {string=} createdAt
 * @property {?string=} startedAt
 * @property {?string=} finishedAt
 */

/**
 * Engine ControlHarnessMaintenanceInventory, passed through.
 * @typedef {Object} ClaudexorMaintenanceInventory
 * @property {string} observedAt
 * @property {Object[]} harnesses
 */

/**
 * @typedef {Object} ClaudexorQuotaRefreshRequest
 * @property {ClaudexorAccountTarget=} target
 */

/**
 * Engine ControlAccountResourcesResponse (or ControlQuotaResponse), passed through.
 * @typedef {Object} ClaudexorAccountResources
 * @property {Object[]} snapshots
 * @property {Object[]=} absences
 * @property {string=} refreshed_at
 * @property {*=} refresh_skipped
 * @property {Object[]=} resources
 */

/**
 * Sent with a required Idempotency-Key header.
 * @typedef {Object} ClaudexorAccountResetRequest
 * @property {ClaudexorAccountTarget} target
 * @property {string} offer_id
 * @property {string=} grant_id
 */

/**
 * Engine ControlAccountResetResponse, passed through.
 * @typedef {Object} ClaudexorAccountResetReceipt
 * @property {string} id
 * @property {Object=} request
 * @property {string} state
 * @property {string=} created_at
 * @property {?string=} completed_at
 * @property {string=} outcome
 * @property {string=} detail
 * @property {Object=} readback
 * @property {*=} resources
 */

/**
 * @typedef {Object} ClaudexorLoginRequest
 * @property {string} harness
 * @property {string=} profile_id
 * @property {'client_pty'=} transport
 * @property {string=} login_flow
 */

/**
 * @typedef {Object} ClaudexorLoginInputRequest
 * @property {string} value
 */

/**
 * @typedef {Object} ClaudexorProfileEnabledResponse
 * @property {boolean} ok
 * @property {string} harness
 * @property {string} profile_id
 * @property {boolean} enabled
 */

/**
 * `GET /api/review-pool`.
 * @typedef {Object} ReviewPoolResponse
 * @property {Object} limits
 * @property {Object} catalog
 * @property {Object[]} pool
 * @property {Object[]} excluded
 * @property {Object} last_executions
 * @property {Object} row_costs
 * @property {string} config_error
 * @property {?Object} migration
 * @property {Array} pool_without_credentials
 */

/**
 * One selectable model; `value` is what a model setting stores.
 * @typedef {Object} ModelCatalogItem
 * @property {string} provider_id
 * @property {string} provider
 * @property {string=} provider_label
 * @property {string=} source
 * @property {string} id
 * @property {string=} name
 * @property {string} value
 * @property {string=} label
 * @property {string=} source_id
 * @property {string=} credential_profile_id
 * @property {string=} account_fingerprint
 * @property {*=} context_window
 * @property {*=} max_context_window
 * @property {*=} max_output_tokens
 * @property {boolean=} is_default
 * @property {*=} supported_options
 * @property {string[]=} reasoning_efforts
 * @property {(string|null)=} default_reasoning_effort
 * @property {string[]=} input_modalities
 * @property {*=} provenance
 * @property {(string|null)=} observed_at
 * @property {*=} availability
 * @property {*=} problem
 * @property {*=} processing
 */

/**
 * @typedef {Object} OpenAICompatibleModelsRequest
 * @property {string=} connection Named compatible row; omitted uses the built-in pair.
 * @property {string} baseUrl
 * @property {string=} apiKey
 */

/**
 * @typedef {Object} LocalModelStartRequest
 * @property {string} source
 * @property {string=} filename
 * @property {number=} port
 * @property {number=} n_gpu_layers
 * @property {number=} n_ctx
 * @property {string=} chat_format
 */

/**
 * @typedef {Object} LocalModelStartResponse
 * @property {'starting'} status
 * @property {string} model_path
 */

/**
 * @typedef {Object} LocalModelRuntimeMissingResponse
 * @property {'runtime_missing'} error
 * @property {string} message
 * @property {string} hint
 */

/**
 * @typedef {Object} LocalModelTestResponse
 * @property {boolean} success
 * @property {boolean} chat_ok
 * @property {boolean} tool_call_ok
 * @property {*} details
 * @property {*} tokens_per_sec
 */

/**
 * @typedef {Object} McpListingAttempt
 * @property {string} at
 * @property {boolean} ok
 * @property {string} error
 * @property {boolean} partial
 */

/**
 * One configured MCP server (a configuration error row carries only id, enabled, tool_count, last_error, code).
 * @typedef {Object} McpServerStatus
 * @property {string} id
 * @property {string=} name
 * @property {boolean} enabled
 * @property {string=} transport
 * @property {string=} url
 * @property {*=} auth_header
 * @property {boolean=} auth_configured
 * @property {string[]=} header_names
 * @property {string[]=} allowed_tools
 * @property {string=} cwd
 * @property {*=} env_from_settings
 * @property {string[]=} configuration_warnings
 * @property {boolean=} schemas_at_start
 * @property {string=} instructions
 * @property {number=} map_tokens
 * @property {number=} schema_tokens
 * @property {number} tool_count
 * @property {Object[]=} tools
 * @property {Object[]=} tool_name_collisions
 * @property {?string} last_error
 * @property {*=} last_refreshed
 * @property {*=} last_attempted
 * @property {?string=} listed_at
 * @property {?McpListingAttempt=} latest_attempt
 * @property {boolean=} complete
 * @property {string=} publication_error
 * @property {string=} code
 */

/**
 * @typedef {Object} McpRefreshRequest
 * @property {string=} server_id
 */

/**
 * @typedef {Object} McpRefreshResponse
 * @property {string=} server_id
 * @property {boolean=} ok
 * @property {boolean=} published
 * @property {string=} publication_error
 * @property {number=} tool_count
 * @property {string[]=} configuration_warnings
 * @property {Object[]=} tool_name_collisions
 * @property {Object[]=} tools
 * @property {string=} error
 * @property {string=} code
 * @property {boolean=} not_started
 * @property {Object=} refreshed
 */

/**
 * @typedef {Object} McpTestRequest
 * @property {string=} server_id
 * @property {Object=} server
 */

/**
 * @typedef {Object} McpTestResponse
 * @property {boolean} ok
 * @property {string=} server_id
 * @property {number=} tool_count
 * @property {string[]=} configuration_warnings
 * @property {boolean=} pagination_truncated
 * @property {Object[]=} tools
 * @property {string=} code
 * @property {string=} error
 */

/**
 * The last apply attempt on /api/update/status (`{}` when none), kept across restarts; never authority.
 * `outcome` is empty until concluded; `applied` only after a later server's startup checks; `unknown` names missing evidence.
 * @typedef {Object} UpdateProgress
 * @property {string=} operation_id
 * @property {string=} generation
 * @property {boolean=} current_generation
 * @property {string=} stage
 * @property {string=} started_at
 * @property {string=} stage_started_at
 * @property {boolean=} active
 * @property {string=} result
 * @property {string=} error
 * @property {boolean=} restart_required
 * @property {(''|'applied'|'not_applied'|'rolled_back'|'failed'|'unknown')=} outcome
 * @property {string=} outcome_at
 * @property {('executor'|'boot')=} concluded_by
 * @property {string=} reason
 * @property {string[]=} blockers
 * @property {(''|'none'|'invoked')=} work_stop
 * @property {string=} base_sha
 * @property {string=} target_sha
 * @property {string=} tx_attempt_id
 * @property {string=} keep_branch
 * @property {string=} stash_note
 * @property {boolean=} retention_unconfirmed - Live observation could not be saved across restart.
 * @property {?Object=} generation_facts - Before/after observations, never an update verdict.
 */

/**
 * @typedef {Object} UpdateTransactionState
 * @property {boolean} active
 * @property {string=} phase
 * @property {string=} task_id
 * @property {boolean=} restart_required
 * @property {boolean=} local_work_recovery
 */

/**
 * `GET /api/update/status` and `POST /api/update/check`. `local_changes`: the checkout's own content
 * since the shared base (not authorship or an apply plan), omitted without a comparison.
 * @typedef {Object} UpdateStatusResponse
 * @property {string} running_sha
 * @property {string=} current_version
 * @property {string=} latest_version
 * @property {Object[]=} official_tags
 * @property {UpdateTransactionState} update_tx
 * @property {UpdateProgress} update_progress
 * @property {?import('./api_types.js').UpdateLetter=} letter
 * @property {?Object=} after_update - Retained before/after process samples; missing observations stay null.
 * @property {boolean=} managed
 * @property {string=} remote
 * @property {string=} remote_branch
 * @property {string=} target_ref
 * @property {string=} official_repo_url
 * @property {string=} update_channel
 * @property {string=} current_branch
 * @property {string=} current_sha
 * @property {string=} current_short_sha
 * @property {string=} latest_sha
 * @property {string=} latest_short_sha
 * @property {string=} latest_message
 * @property {number=} ahead
 * @property {number=} behind
 * @property {('absent'|'present'|'unknown')=} local_changes
 * @property {string=} local_changes_base
 * @property {('unresolved_sha'|'shallow_history'|'git_error'|'several_bases'|'no_shared_base')=} local_changes_reason
 * @property {boolean=} dirty
 * @property {number=} dirty_count
 * @property {string[]=} dirty_preview
 * @property {string[]=} warnings
 * @property {?boolean=} check_ok
 * @property {boolean=} available
 * @property {boolean=} safe_to_apply
 * @property {string=} checked_at
 * @property {string=} checked_target_sha
 * @property {boolean=} from_cache
 */

/**
 * @typedef {Object} CommandRequest
 * @property {string} cmd
 * @property {Object=} client_surface
 * @property {string=} visible_text
 * @property {string=} visible_task_id
 * @property {Object=} task_constraint
 */

/**
 * @typedef {Object} CommandResponse
 * @property {string=} status
 * @property {boolean=} ok
 * @property {boolean=} deduped
 * @property {string=} task_id
 */

/**
 * @typedef {Object} GitRollbackRequest
 * @property {string} target
 */

/**
 * @typedef {Object} GitRollbackResponse
 * @property {string} status
 * @property {string} message
 * @property {boolean} restarting
 * @property {boolean} switch_deferred
 */

/**
 * @typedef {Object} GitPromoteResponse
 * @property {string} status
 * @property {string} sha
 * @property {string} message
 */

/**
 * @typedef {Object} ResetResponse
 * @property {string} status
 * @property {string[]} deleted
 * @property {boolean} restarting
 */

/**
 * `GET /api/cost-breakdown`; a 503 carries `accounting.available: false` (unknown, never zero).
 * @typedef {Object} CostBreakdownResponse
 * @property {number=} total_cost
 * @property {number=} total_calls
 * @property {number=} total_prompt_tokens
 * @property {number=} total_completion_tokens
 * @property {number=} total_cached_tokens
 * @property {number=} total_cache_write_tokens
 * @property {Object=} prompt_cache_ttls
 * @property {Object=} by_model
 * @property {Object=} by_api_key
 * @property {Object=} by_model_category
 * @property {Object=} by_task_category
 * @property {Object} accounting
 * @property {Object=} unattributed
 */

/**
 * `GET /api/extensions`: every installed skill and what the extension loader has live.
 * @typedef {Object} ExtensionsIndexResponse
 * @property {UiContribution[]} contributions
 * @property {Object.<string,Object[]>} offers
 * @property {Object} preferences
 * @property {Object.<string,{key:string,available:boolean,reason:string}>} voice
 * @property {string[]} icons
 * @property {Object.<string,string[]>} problems
 * @property {InstalledSkill[]} skills
 * @property {ExtensionLiveSnapshot} live
 */

/**
 * `GET /api/skills/lifecycle-queue`: the running job and the recent ones (at most 80, oldest first,
 * kept in memory only).
 * @typedef {Object} SkillLifecycleQueueResponse
 * @property {?SkillLifecycleJob} active
 * @property {SkillLifecycleJob[]} events
 */

/**
 * `GET /api/model-catalog`: with `source_id` (and optionally `credential_profile_id`) one agent
 * account's models only; otherwise every configured provider's, a failed one as an `errors` row.
 * @typedef {Object} ModelCatalogResponse
 * @property {ModelCatalogItem[]} items
 * @property {Object[]} errors
 * @property {Object[]} model_sources
 * @property {Object[]=} account_catalogs
 * @property {boolean=} partial
 */

/**
 * `GET /api/local-model/status`.
 * @typedef {Object} LocalModelStatusResponse
 * @property {string} status
 * @property {?string=} error
 * @property {?string=} model_path
 * @property {*=} model_name
 * @property {*=} context_length
 * @property {*=} port
 * @property {*=} download_progress
 * @property {string=} runtime_status
 * @property {string=} runtime_install_log
 * @property {Object=} settings_application
 */

/**
 * `GET /api/mcp/status`: configuration and the last listed tools (no network call).
 * @typedef {Object} McpStatusResponse
 * @property {boolean} enabled
 * @property {boolean} sdk_available
 * @property {*} sdk_error
 * @property {*} tool_timeout_sec
 * @property {McpServerStatus[]} servers
 * @property {Object<string, number>} weights
 */

/**
 * `extension_loader.snapshot()` — the `live` block of `GET /api/extensions`.
 * @typedef {Object} ExtensionLiveSnapshot
 * @property {UiContribution[]} contributions
 * @property {string[]} extensions
 * @property {string[]} tools
 * @property {string[]} routes
 * @property {string[]} ws_handlers
 * @property {Object[]} ui_tabs
 * @property {Object[]} settings_sections
 */

/**
 * Manifest contribution, projected from one live revision. Optional provider fields stay absent.
 * @typedef {Object} UiContribution
 * @property {string} key
 * @property {string} skill
 * @property {string} skill_label
 * @property {string} id
 * @property {string=} placement
 * @property {string=} provides
 * @property {string} label
 * @property {string} title
 * @property {string} icon
 * @property {Object<string,string>=} icon_spec
 * @property {number} order
 * @property {Object=} action
 * @property {{url:string,refresh_s:number}=} source
 * @property {string=} runtime
 * @property {string=} route_url
 * @property {string=} destination
 * @property {string=} scope
 * @property {string[]=} wakes
 * @property {string=} settings
 * @property {string=} read_only
 * @property {Object[]=} tools
 * @property {string[]=} languages
 * @property {string[]=} formats
 * @property {string=} install_size
 * @property {number=} max_bytes
 * @property {number=} max_seconds
 * @property {{kind:string,url:string}} render
 * @property {string=} revision
 */
/**
 * GET /api/ui/contributions (no-store). A missing chosen engine remains selected.
 * @typedef {Object} UiContributionsResponse
 * @property {UiContribution[]} contributions
 * @property {Object.<string,Object[]>} offers
 * @property {Object} preferences
 * @property {Object.<string,{key:string,available:boolean,reason:string}>} voice
 * @property {string[]} icons
 * @property {Object.<string,string[]>} problems
 */
