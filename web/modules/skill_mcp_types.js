/** Dependency-free JSDoc mirror of a skill's Connect tools (`GET|POST /api/skills/{skill}/mcp-connect`).
 *  Python twin: `ouroboros/gateway/skill_mcp_contracts.py` (field parity: tests/test_skill_mcp.py).
 *  The preview writes nothing; the confirm carries its settings digest, the payload revision and the collateral consent. */
/**
 * One declared server and what Connect does to its Settings row.
 * @typedef {Object} SkillMcpConnectServer
 * @property {string} id
 * @property {string} title
 * @property {'add'|'enable'|'repair'|'unchanged'|'conflict'} change
 * @property {string} conflict Why a same-id row blocks it; empty otherwise.
 * @property {?Object<string, *>} row The exact MCP_SERVERS row after the confirm; null for a conflict.
 * @property {boolean} task_session
 * @property {boolean} browser_policy
 * @property {Object<string, *>=} executable Preview: {command, args, cwd} derived now, never stored.
 */

/**
 * A row already on that turning MCP on would start too.
 * @typedef {Object} SkillMcpConnectOtherServer
 * @property {string} id
 * @property {string} name
 */

/**
 * @typedef {Object} SkillMcpConnectPreviewResponse
 * @property {boolean} ok
 * @property {string=} skill
 * @property {string=} content_hash
 * @property {boolean=} ready
 * @property {string[]=} blockers
 * @property {string=} next_action
 * @property {SkillMcpConnectServer[]=} servers
 * @property {boolean=} mcp_enabled
 * @property {boolean=} enables_mcp
 * @property {SkillMcpConnectOtherServer[]=} other_servers
 * @property {boolean=} requires_other_servers_confirmation
 * @property {boolean=} changed
 * @property {boolean=} conflict
 * @property {string=} settings_digest
 * @property {string=} code
 * @property {string=} error
 * @property {boolean=} saved
 */

/**
 * @typedef {Object} SkillMcpConnectRequest
 * @property {string} expected_content_hash
 * @property {string} settings_digest
 * @property {boolean=} confirm_other_servers
 */

/**
 * @typedef {Object} SkillMcpConnectResponse
 * @property {boolean} ok
 * @property {boolean} saved
 * @property {boolean=} changed
 * @property {string=} skill
 * @property {boolean=} mcp_enabled
 * @property {SkillMcpConnectServer[]=} servers
 * @property {SkillMcpConnectOtherServer[]=} other_servers
 * @property {string[]=} changed_keys
 * @property {string=} code
 * @property {string=} error
 */
