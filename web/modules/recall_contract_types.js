/** Dependency-free JSDoc mirror of the recall door's envelopes in ouroboros.gateway.history_contracts
 *  (GET /api/recall, gateway/recall.py). The door's callers are other programs, not the builtin client, so the
 *  facade re-exports none of them. */

/**
 * A read the caller can run with the recall door alone: `path` (GET on this server) or `cli` (after `ouroboros`).
 * @typedef {Object} RecallReader
 * @property {string} path
 * @property {string[]} cli
 */

/**
 * One typed field of a dialogue row with its own speaker (`owner`, `host`, `task <id>`, "the asking task", "the closing
 * task", or the row's author label); an excerpt on a search answer, whole on a reopened row.
 * @typedef {Object} RecallField
 * @property {string} name
 * @property {string} speaker
 * @property {string} text
 */

/**
 * One original dialogue row as the recall door shows it: its full content address, rooms, speaker, fields.
 * @typedef {Object} RecallRow
 * @property {string} address
 * @property {string} room_id
 * @property {string} room
 * @property {string[]} rooms
 * @property {string} ts
 * @property {string} speaker
 * @property {string[]} matched
 * @property {RecallField[]} fields
 * @property {string=} reply_to
 * @property {boolean=} reply_unresolved
 * @property {RecallReader} reopen
 */

/**
 * `GET /api/recall?words=…`: one page of matching dialogue rows in the caller's order of time, never ranked.
 * @typedef {Object} RecallSearchResponse
 * @property {'search'} mode
 * @property {string[]} words
 * @property {string} room_id
 * @property {string} voice
 * @property {'newest'|'oldest'} order
 * @property {?number} total
 * @property {number} observed
 * @property {number} offset
 * @property {number} more
 * @property {?string} snapshot
 * @property {?RecallReader} next
 * @property {string[]} gaps
 * @property {string[]} notes
 * @property {RecallRow[]} rows
 */

/**
 * `GET /api/recall?row=<address>`: one dialogue row reopened whole.
 * @typedef {Object} RecallRowResponse
 * @property {'row'} mode
 * @property {RecallRow} row
 * @property {string[]} notes
 */
