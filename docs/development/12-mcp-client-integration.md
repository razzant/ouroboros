# MCP Client Integration

This chapter owns the boundary for configured MCP servers: the base runtime is a client and never a server, descriptions and results are untrusted data rather than policy, and the owner alone selects which settings references a server may receive. It also fixes how referenced settings values reach a process environment, because a value's secret classification decides whether it is masked in diagnostics and logs.

The base runtime is an optional CLIENT for trusted HTTP/SSE and local stdio MCP
servers — never an MCP server (structure, transport validation, the stdio
`command`/`args`/`cwd`/`env`/`env_from_settings` contract and the masking rule:
ARCHITECTURE "MCP and browser-facing external tools";
`ouroboros/mcp_client.py`). MCP descriptions and results are untrusted data,
not policy: configuration trust must not turn remote prose into policy. Enabled
tools join the initial capability envelope, still pass runtime safety and the
caller's ordinary capability ceiling; discovery failure becomes a visible
capability omission. The owner chooses MCP references in Settings; a caller's
existing MCP tool grant does not authorize new references. Executable schema
properties, required fields, enum and default values stay intact. Resources,
prompts, and MCP server behavior remain separate architecture changes.
Enforcement: `tests/test_mcp_client.py`, `tests/test_process_environment.py`.

`start_service` overlays ordinary literal `env` on the existing minimal host
baseline; root tasks can additionally choose `env_from_settings` through their
host-resolved process authority. Restricted and Presence tasks receive no new
Settings-selection authority, while their previous literal env and configured MCP
access remain available. Skill grants do not authorize an unrelated service.
Both service backends and MCP reuse `workspace_executor.resolve_process_env`;
referenced Settings fields retain the same secret/ordinary classification.
Cwd still uses the host-owned resource binding; local import scrubbing and
interpreter defaults remain. Docker forwards values through inert CLI environment
aliases and restores the selected names inside the container, preserving host CLI
configuration. Values do not enter host argv or generated shell source.
Service diagnostics and finalized log blobs mask referenced secrets; ordinary
PORT/PATH/DEBUG values, protocol identity/state and executable schema stay intact.
The executor's record-stop owner finalizes local logs through the existing service
log owner before forgetting the in-memory selections, including task/global cleanup
and replacement after exit. Unconfirmed termination retains the existing record;
a later cleanup can settle it. Live child logs and logs surviving worker loss keep
the existing private raw-log contract until successful finalization; oversized or
uncapturable logs retain their existing explicit omission/error report. No secret
values are added to the durable process ledger. Enforcement:
`tests/test_process_environment.py`, `tests/test_workspace_executor_services.py`.

