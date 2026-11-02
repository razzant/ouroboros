# DEVELOPMENT.md — Development Principles & Module Guide

Rules by change class name their enforcement (or its absence). The first chapter fixes authority beside BIBLE, ARCHITECTURE, DESIGN and CHECKLISTS. Update the affected node, not a changelog. On the official line, leave each reference book no larger ("Documentation contract").

## Chapters

- [Role and authority](development/01-role-and-authority.md)
- [Naming and boundaries](development/02-naming-and-boundaries.md)
- [Module Size & Complexity](development/03-module-size-and-complexity.md)
- [Core Governance Artifacts](development/04-core-governance-artifacts.md)
- [Review & Commit Protocol](development/05-review-and-commit-protocol.md)
- [Rules by change class](development/06-rules-by-change-class.md)
- [Managed Update Rule](development/07-managed-update-rule.md)
- [Mutation Attribution Rule](development/08-mutation-attribution-rule.md)
- [Process Custody Rule](development/09-process-custody-rule.md)
- [Platform Abstraction Rule](development/10-platform-abstraction-rule.md)
- [Design System](development/11-design-system.md)
- [MCP Client Integration](development/12-mcp-client-integration.md)
- [Gateway Boundary Pattern](development/13-gateway-boundary-pattern.md)
- [Build & CI](development/14-build-and-ci.md)

`python devtools/context_fit_report.py --data ROOT --room ID --window TOKENS --reserve TOKENS [--caller direct|root]` measures a caller's fresh-task projection at density 1.0 (chars/4 estimates, not provider tokens): `root` is a queued task (two start work rooms), tools are a standalone registry's `initial_tool_schemas` (no live extension skills: may be smaller or larger than a real task), and `--reserve` is the route's reply ceiling (a call's working reserve under `--session-provider`). `--session-provider gemini_live|grok_realtime|openai_realtime` (direct caller only) also runs the real conversation host's tool selection and session-candidate seam with bundled default model/options and cached output limit (call tools, driver instructions, provider wire), never connecting. Output names missing inputs; `devtools._context_fit_session.measure_session` measures a retained host. Other options: `--mode max|low|nano`, `--json`. SQLite uses `mode=ro&immutable=1`; ROOT writes are forbidden, while content-addressed blobs go to scratch so the measured context keeps the server's pointers. A live or unknown PID/lock, or WAL, needs `--snapshot`: no liveness probe, WAL recovered in a private copy. Changing inputs, stale indexes and external local skill sources are refused. Live process scanning is unavailable and adds no text to the measured input; this estimates a fit, not provider acceptance.
