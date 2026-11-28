/** Runtime UI limits and the release-version carrier kept at this path for older updaters.
 * The remaining typedefs cover browser views, open server payloads and legacy replies without a matching named wire shape. */

/**
 * Background Consciousness alarm clock (server._describe_bg_consciousness_state over consciousness.status_snapshot). A wake-up is an ordinary Main turn; the direct-activity census owns its liveness.
 * @typedef {Object} BgConsciousnessState
 * @property {boolean} enabled
 * @property {string} status  // disabled | starting | stopped | thinking | due | sleeping | owner_chat_unknown | allowance_exhausted | allowance_unknown | wake_rejected | wake_failed | wake_paused | wake_outcome_unknown
 * @property {string} detail  // one honest owner-readable line (e.g. "Sleeping until 14:05 (UTC+03:00).", in the server's zone)
 * @property {string=} detail_note  // what follows from that state, said beneath it; only a proven allowance hold has one
 * @property {string} level  // observe | act | full
 * @property {string} next_wake_at  // next attempt, moved by skips/refusals; ISO instant or ""
 * @property {string} appointment_at  // scheduled time, retained through retries; "" on legacy state
 * @property {string} last_wake_at  // ISO instant; "" before the first wake of this process
 * @property {string} last_wake_task_id
 * @property {string} last_wake_outcome  // running | done | paused | pausing | unknown | failed | rejected:<reason> | skipped:<reason>
 * @property {string} last_error
 * @property {?number} spent_24h_usd  // null when the ledger could not be read
 * @property {?number} daily_usd
 * @property {string} allowance_resets_at  // ISO instant the oldest counted spend leaves the 24 h window
 * @property {number} tasks_running  // live roots consciousness started
 * @property {number} max_tasks
 * @property {string} live_wake_task_id  // own wake through direct custody, queue and settlement; "" when absent
 * @property {boolean} live_wake_held  // own wake in PENDING: paused, held or awaiting dispatch
 * @property {?{cause: string, ceiling_usd: ?number, resume_additional_usd?: ?number, resume_available_usd?: ?number}} live_wake_hold  // pause cause and, when attributable, the allowance's additional room and what the total budget leaves of it now (null: unreadable)
 * @property {number} unknown_unmetered  // window rows without a price: spent_24h_usd is then a floor ("at least")
 * @property {boolean} integrity_degraded  // the ledger was quarantined/repaired; the numbers are best-effort
 * @property {string} allowance_status  // the current window's verdict: available | exhausted | allowance_unknown
 * @property {?number} allowance_open_rows  // window rows that may still settle; null unknown
 * @property {string} allowance_room_after  // ISO instant after which known spend falls below the limit if nothing more is charged; "" withheld
 */

/** @typedef {Object} AvailableSubagentRoute
 * @property {'api_model'|'agent_session'} kind
 * @property {string} target_id
 * @property {string=} credential_profile_id
 */

/** @typedef {Object} AvailableSubagentItem
 * @property {string} subagent_id
 * @property {boolean=} enabled - Omitted means true; false withdraws new selections.
 * @property {string} recommended_use
 * @property {AvailableSubagentRoute} route
 * @property {string=} effort
 */

/** @typedef {Object} AvailableSubagentsSetting
 * @property {boolean} enabled
 * @property {AvailableSubagentItem[]} items
 */

/**
 * 503 settings_save_timeout: the server writer still runs; saved=null is unknown. Reload to check status before retrying.
 * @typedef {Object} SettingsSaveTimeoutResponse
 * @property {string} error
 * @property {string} code  // settings_save_timeout
 * @property {null} saved
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

export const MAX_LINK_ACTIONS = 12;
export const MAX_QUIZ_OPTIONS = 6;
// Mirror task_decision._COMMENT_MAX: ingress refuses longer comments, never
// truncates; cards must offer only comments the ingress can deliver verbatim.
export const MAX_DECISION_COMMENT = 2000;
export const GATEWAY_CONTRACT_VERSION = '7.13.1-alpha.0';
