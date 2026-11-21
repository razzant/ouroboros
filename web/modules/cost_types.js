/** Dependency-free JSDoc mirror of the cost envelopes a task card and its detail carry: `CostPresentation`
 *  (`ouroboros.cost_projection`) and `CostCustody` (`ouroboros.delegate_custody_usage`); api_types.js re-exports both. */
/** @typedef {Object} CostPresentation
 * @property {'own'|'root_tree'} scope
 * @property {?number} tracked_amount
 * @property {boolean} has_unpriced
 * @property {boolean} tracked_final
 * @property {boolean} accounting_open
 * @property {boolean} has_rows
 */
/** @typedef {Object} CostCustody
 * @property {'available'|'unavailable'} status
 * @property {Array<string>} open_run_ids
 * @property {Array<string>} unrecorded_run_ids
 * @property {Array<string>=} closed_unrecorded_run_ids
 * @property {Array<string>} pending_invocation_ids
 */
