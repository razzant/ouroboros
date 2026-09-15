# Ouroboros v7.0.0 — Architecture & Reference

This is the present-tense operational map of Ouroboros (BIBLE P6), in three layers: structure (what exists and where), operation (files, env keys, state paths, endpoints, flows), and rationale; it is NOT a changelog, and version history lives in README.md, git tags, and the commit log. Every important WHY stays in this book at least briefly, while mechanism detail lives in the module docstring the map points to by name, and rationale must be self-contained — future maintainers should not need old commits to understand why a guard, review gate, or lifecycle exists. The chapters below are the book: each owns one section of the map, and a change replaces the description of the node it touched.

## Chapters

- [1. High-Level Architecture](architecture/01-high-level-architecture.md)
- [2. Startup / Onboarding Flow](architecture/02-startup-onboarding-flow.md)
- [3. Web UI Pages & Buttons](architecture/03-web-ui-pages-and-buttons.md)
- [4. Server API Endpoints](architecture/04-server-api-endpoints.md)
- [5. Supervisor Loop](architecture/05-supervisor-loop.md)
- [6. Agent Core](architecture/06-agent-core.md)
- [7. Configuration (ouroboros/config.py)](architecture/07-configuration.md)
- [8. Git Branching, CI, and Build](architecture/08-git-branching-ci-and-build.md)
- [9. Shutdown & Process Cleanup](architecture/09-shutdown-and-process-cleanup.md)
- [10. Key Invariants](architecture/10-key-invariants.md)
- [11. Frozen Contracts v1 (`ouroboros/contracts/`)](architecture/11-frozen-contracts-v1.md)
- [12. Host Service, Companion Processes, and Chat IDs](architecture/12-host-service-companions-and-chat-ids.md)
- [13. External Skills Layer](architecture/13-external-skills-layer.md)
