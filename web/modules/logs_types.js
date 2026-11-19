/** Dependency-free JSDoc mirror of the Dashboard › Logs envelopes in `ouroboros.gateway.logs`
 *  (`GET /api/logs/{name}?view=compact`, `GET /api/logs/{name}/raw`), re-exported by ./api_types.js:
 *  a compact row's admitted file source, its verified Raw page and the compact tail. */

/**
 * Stored JSONL byte identity, never a public projection digest or window row number.
 * @typedef {Object} LogRowSource
 * @property {1} v
 * @property {'log_row'} kind
 * @property {string} placement  // canonical, or server-admitted path relative to the canonical drive
 * @property {string} task_id  // the original query's task scope, empty for canonical unfiltered logs
 * @property {string} log
 * @property {number} start  // chain byte coordinate; inclusive
 * @property {number} length  // includes the original line terminator
 * @property {string} sha256
 */

/**
 * `reconstructed`: a context view checkpoint plus the absent structures its own `at_source` names.
 * @typedef {Object} LogRawReference
 * @property {string} ref  // base64url JSON of source; selector, never authorization
 * @property {LogRowSource} source
 * @property {string[]} omitted_fields  // at most 64 names, each at most 128 characters
 * @property {number} omitted_field_count
 * @property {number|null} public_bytes  // null when reconstructed: Raw reads the receipt, the preview never does
 * @property {'stored'|'public_task_result'|'reconstructed'} representation
 */

/**
 * A receipt Raw could not use leaves the stored event whole and names the declared fields it lacks.
 * @typedef {Object} LogRawResponse
 * @property {LogRowSource} source
 * @property {'stored'|'public_task_result'|'reconstructed'} representation
 * @property {string} sha256  // bytes of the served representation, not source.sha256
 * @property {number} total_bytes
 * @property {number} offset
 * @property {number} next_offset
 * @property {boolean} has_more
 * @property {string} text
 * @property {string[]} unavailable_fields
 * @property {''|'receipt_unavailable'|'receipt_mismatch'} unavailable_reason
 */

/**
 * @typedef {Object} CompactLogEntry
 * @property {LogRawReference=} _raw  // absent for unchanged dialogue streams; other public fields are open
 */

/**
 * @typedef {Object} CompactLogTailResponse
 * @property {string} name
 * @property {CompactLogEntry[]} entries
 * @property {'compact'} view
 * @property {string[]} gaps  // damage observed in the selected read window, not all history
 * @property {number} skipped_rows  // measured nonblank unaddressable records before task/limit filtering
 */
