# MCP Client Integration

MCP is client-only; remote data is not policy. Owners select Settings references.

`mcp_catalog.py`: atomic/stat-cached `_schema_version: 1` envelopes at `config.DATA_DIR/state/mcp/catalogs/server-<canonical_id>.json` (call-time root). `server_id`, `identity` (SHA-256: config/credentials/env; excludes enablement/allowlists/pin/mode/lifetime), `listed_at`, `complete`, ordered `items`, `instructions`, `schema_omissions`, `latest_attempt {at,ok,error,partial}`. Items: name, description, input_schema; metadata: title, annotations, outputSchema. Drop secret-bearing defaults/examples/outputSchema with notice; withhold only irreducible input contracts. `normalize_listing`/`_adopt_catalog` disclose live/old omissions; no disk wrapping/cuts.

List at boot, Save/mtime, Refresh. Configure seeds; reads adopt files without mutating resident request arrays. Serialize per server without a manager lock over network waits; config events coalesce one background pass. Complete success (empty too) replaces evidence; partial/failed attempts update only `latest_attempt` (first failure is attempt-only; stale runs write nothing). Failed publication keeps local tools (`publication_error`, never `published`). Only config events prune (`test_mcp_catalog.py`).

HTTP/SSE validate URLs/headers and refuse credential redirects across origins. Settings masks literal headers, not echoes. SDK logs mask tokens/selected secrets, not header values. Transport tests: loopback `tests/test_mcp_*.py`.

Stdio keeps argv/cwd/selected env exact; `task_session` adds the task marker, `browser_policy` reserved `OUROBOROS_BROWSER_GUARD`/`_TOKEN` (ARCHITECTURE §6 MCP). Admitted env references use `workspace_executor.resolve_process_env` under `tools/process_facts.py::settings_environment_allowed` (root/Cyber acting, not Presence; MCP/skill grants confer no authority). Mask secret diagnostics/logs; no secrets in host argv, shell source or process ledgers; Docker: inert aliases. Services overlay a minimal baseline; commands/scripts/verification inherit theirs (`test_process_environment.py`, `test_workspace_executor_services.py`).
