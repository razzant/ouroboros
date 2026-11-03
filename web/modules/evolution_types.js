/** Dependency-free JSDoc mirror of the Dashboard Evolution envelope in `ouroboros.gateway.contracts`
 *  (`GET /api/evolution-data`, served from `ouroboros/evolution_history.py`), re-exported by ./api_types.js. */
/**
 * @typedef {Object} EvolutionRefresh
 * @property {'idle'|'running'|'failed'} state
 * @property {string} started_at
 * @property {string} finished_at
 * @property {string} error  // "" unless failed; the saved copy stays
 * @property {number} unmeasured  // versions of the saved copy whose commit could not be read (retried by force)
 * @property {string} unmeasured_error  // the first such reason, "" when every version was measured
 */
/**
 * @typedef {Object} EvolutionDataResponse  // answered at once from the saved copy
 * @property {Object[]} versions  // {tag, aliases, version, date, commit, level, milestone, code_lines, bible_bytes, system_bytes, release}
 * @property {string} saved_at  // "" = no saved copy yet
 * @property {EvolutionRefresh} refresh
 * @property {Object} install  // identity / scratchpad: {at_versions, first_at, latest}
 * @property {Object[]} points  // the earlier shape, kept for older clients: one per tag name; null = unmeasured; memory fields null
 * @property {boolean} cached  // true when the answer comes from a saved copy
 * @property {Object[]=} checkpoints
 * @property {string} generated_at
 */
