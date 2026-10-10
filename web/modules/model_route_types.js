/** Dependency-free JSDoc mirror of the `GET /api/settings` previews and the maximum-response acknowledgement:
 *  the built-in web_search route (`ouroboros/search_routes.py`) and the maximum response (`ouroboros/response_limits.py`).
 *  Python twin: `ouroboros/gateway/model_route_contracts.py` (field parity: tests/test_model_route_contract_parity.py). */
/**
 * One built-in search transport the resolver may try, in order.
 * @typedef {Object} WebSearchLeg
 * @property {string} source `openai`, `openrouter`, `anthropic` or `ddgs`.
 * @property {string} backend
 * @property {string} model The wire model; empty for ddgs.
 * @property {'selection'|'provider_default'} model_source
 */

/**
 * One built-in web_search Source/Model (`search_routes.resolve_web_search_route`). Skills,
 * MCP and browser tools are not part of it.
 * @typedef {Object} WebSearchRoute
 * @property {'builtin_web_search'} scope
 * @property {string} source `auto` or one strict source.
 * @property {string} model The selection as saved or drafted; empty means provider defaults.
 * @property {WebSearchLeg[]} legs Eligible, in order.
 * @property {WebSearchLeg[]} unavailable Credential or package missing.
 * @property {string} note
 * @property {string=} error The selection has no built-in transport.
 * @property {string=} ignored_legacy_model A saved bare model the Anthropic source does not apply.
 * @property {string=} unapplied_model Under Auto, a saved model no built-in transport serves.
 */

/**
 * The separate maximum-response fact of one exact route (`response_limits.ResponseLimit`).
 * @typedef {Object} ResponseLimit
 * @property {number} max_output_tokens 0 = unknown, never a cap.
 * @property {string} status
 * @property {string} source `owner_ack`, a metadata source, or `none`.
 * @property {string} observed_at
 * @property {boolean} stale
 * @property {string} route_fp
 */

/**
 * The endpoint and account the previewed model's send reaches.
 * @typedef {Object} ResponseLimitRoute
 * @property {string} provider
 * @property {string} model
 * @property {string} base_url
 * @property {boolean} use_local
 * @property {Object|null} options A subscription's exact account binding, else null.
 */

/**
 * `GET /api/settings?response_limit_preview=1`: the endpoint the send reaches and its maximum; stores nothing.
 * @typedef {Object} ResponseLimitPreview
 * @property {ResponseLimitRoute} route
 * @property {ResponseLimit} response_limit
 */

/**
 * `POST /api/owner/capability-ack` with `max_output_tokens` (0 removes the owner's maximum).
 * @typedef {Object} ResponseLimitAckResponse
 * @property {true} ok
 * @property {ResponseLimit} ack
 */
