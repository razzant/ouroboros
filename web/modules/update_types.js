/** Dependency-free JSDoc mirror of the managed-update envelopes in `ouroboros/gateway/update_contracts.py`
 *  (`/api/update/preflight`, `/api/update/apply` and the rollback refusal sharing its error shape,
 *  `/api/update/description`, the `update_progress_changed` and `update_status_ready` frames),
 *  re-exported by ./api_types.js. */
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
 * @property {Object=} effects
 * @property {?Object=} letter
 * @property {?Object=} qualification
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
 * @property {'not_applied'|'rolled_back'=} outcome
 * @property {Object=} index_lock
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

/**
 * @typedef {Object} UpdateDescriptionRequest
 * @property {'write'|'rewrite'} mode
 * @property {string} base_sha
 * @property {string} target_sha
 */

/**
 * @typedef {Object} UpdateDescriptionResponse
 * @property {'queued'} status
 * @property {string} task_id
 * @property {Object} key {base_sha, target_sha, update_channel, target_ref}
 */

/**
 * @typedef {Object} UpdateDescriptionErrorResponse
 * @property {string} error
 * @property {string=} task_id
 * @property {string=} task_status
 * @property {Object[]=} legal_ranges
 * @property {string=} reason
 */

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
