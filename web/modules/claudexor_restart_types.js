/** Dependency-free JSDoc mirror of the Restart Claudexor envelopes (`GET`/`POST /api/claudexor/restart`).
 *  Python twin: `ouroboros/gateway/management_contracts.py` (field parity:
 *  tests/test_client_contract_bindings.py). Consumer: ./claudexor_restart.js. */
/**
 * One live engine run a restart ends. owner is ouroboros when this installation's own run
 * records hold it, else other; task_id and kind are that record's, "" for other.
 * @typedef {Object} ClaudexorRestartRun
 * @property {string} run_id
 * @property {'queued'|'running'} state
 * @property {'ouroboros'|'other'} owner
 * @property {string} task_id
 * @property {string} kind
 * @property {string} harness
 * @property {string} project
 * @property {string} started_at
 */

/**
 * known: exact counts. unattributed: total listed, own records unreadable. unknown: the engine
 * did not list its runs; ouroboros is what the own records hold open. Null is a gap, never zero.
 * @typedef {Object} ClaudexorRestartRuns
 * @property {'known'|'unattributed'|'unknown'} status
 * @property {?number} total
 * @property {?number} ouroboros
 * @property {?number} other
 * @property {?string} error
 * @property {ClaudexorRestartRun[]} items  // at most 50, oldest first
 * @property {number} items_omitted
 */

/**
 * engine_pin_below_serving: the pinned engine is older than the serving one and may not start.
 * @typedef {Object} ClaudexorRestartWarning
 * @property {string} code
 * @property {string} message
 * @property {string} pin_version
 * @property {string} serving_version
 */

/**
 * GET /api/claudexor/restart: what Restart Claudexor would end and start. token names the
 * engine, the pin and exactly which runs end and whose; model calls are not part of it.
 * @typedef {Object} ClaudexorRestartPreview
 * @property {Object} engine  // {state: serving|not_answering|not_started, serving, pin, direction}
 * @property {ClaudexorRestartRuns} runs
 * @property {Object} model_calls  // {status: known|unknown, active, queued}
 * @property {ClaudexorRestartWarning[]} warnings
 * @property {string} token
 */

/**
 * @typedef {Object} ClaudexorRestartRequest
 * @property {string} token
 */

/**
 * stop unconfirmed: the old process may still run. serving false carries the start's error.
 * @typedef {Object} ClaudexorRestartReceipt
 * @property {'stopped'|'nothing_to_stop'|'unconfirmed'} stop
 * @property {boolean} serving
 * @property {?string} error
 * @property {Object} runs  // the confirmed counts {status, total, ouroboros, other}
 */

/**
 * @typedef {Object} ClaudexorRestartResponse
 * @property {import('./api_types.js').ClaudexorStatusResponse} status
 * @property {ClaudexorRestartReceipt} restart
 */

/**
 * Nothing was stopped. 409 restart_preview_stale carries the fresh preview; 409
 * restart_in_progress; 503 carries the engine's typed code.
 * @typedef {Object} ClaudexorRestartProblem
 * @property {string} error
 * @property {string} code
 * @property {ClaudexorRestartPreview=} preview
 */
