# DEVELOPMENT.md — Development Principles & Module Guide

This is Ouroboros's engineering handbook: imperative rules for changing the body, grouped by change class, each naming the surface that enforces it — a test, a gate, a CI lane — or stating honestly that none does. The first chapter fixes this book's authority beside the constitution, the architecture map, the design semantics and the reviewer checklists; the rest are the rules themselves, from naming and size discipline through the governance-artifact contract, the commit protocol, the per-change-class rules, and the build and CI topology. Read the chapter for the class of change in hand: this book is not a changelog, and a change replaces the description of the node it touched.

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
