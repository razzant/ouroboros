/** Dependency-free JSDoc mirror of the MCP import preview (`POST /api/mcp/import/preview`).
 *  Python twin: `ouroboros/gateway/mcp_import_contracts.py` (field parity: tests/test_mcp_import.py).
 *  The request carries matching metadata only; only an entry's `patch` carries pasted values. */
/**
 * What import matching reads from one draft server: no header value or address.
 * @typedef {Object} MCPImportDraftServer
 * @property {string=} id
 * @property {string=} slug
 * @property {string=} name
 * @property {string=} transport
 * @property {string[]=} header_names The editor's current header list, the legacy row included.
 * @property {?string=} legacy_header Which of those names is the stored legacy pair.
 * @property {boolean=} unreadable_headers A malformed saved map that a file's headers would replace.
 * @property {string=} skill A row Connect tools bound to a skill: import never rewrites it.
 */

/**
 * An update whose entry names `headers`: the file's list replaces the current one.
 * @typedef {Object} MCPImportHeaderChanges
 * @property {string[]} added
 * @property {string[]} replaced
 * @property {string[]} removed
 */

/**
 * @typedef {Object} MCPImportPreviewEntry
 * @property {string} source_name
 * @property {string} server_id
 * @property {''|'add'|'update'} action Empty while problems keep the entry out.
 * @property {string} transport
 * @property {number} arg_count
 * @property {string[]} header_names
 * @property {string[]} env_names
 * @property {string[]} unsupported_keys
 * @property {string[]} problems
 * @property {string[]} warnings
 * @property {Object<string, *>} patch The only field carrying pasted values; empty while problems exist.
 * @property {number=} index The updated draft position.
 * @property {MCPImportHeaderChanges=} header_changes
 * @property {string[]=} kept_header_names An update without headers keeps these.
 * @property {string=} legacy_header
 * @property {boolean=} unreadable_headers
 */

/**
 * @typedef {Object} MCPImportPreviewRequest
 * @property {string} text
 * @property {Array<?MCPImportDraftServer>} servers Null keeps a malformed draft row's position.
 */

/**
 * @typedef {Object} MCPImportPreviewResponse
 * @property {boolean} ok
 * @property {MCPImportPreviewEntry[]} entries
 * @property {('mcpServers'|'servers')=} wrapper
 * @property {string[]=} ignored_keys
 * @property {string=} error
 */
