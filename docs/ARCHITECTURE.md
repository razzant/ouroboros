# Ouroboros v7.12.2-alpha.0 — Architecture & Reference

This is the map of my body (BIBLE P6): what I am made of, what each part lets me do, how I reach it and which limits change my choices. Each chapter opens with that answer for its part; the rest of the chapter is the engineering manual I read before I change that part, §10 holds the invariants every part keeps and the roads already closed, and `ouroboros/domains.toml` names the manuals that own each module domain. This present-tense map explains structure and rationale; module docstrings own mechanisms. Its three mandatory layers: **Structural**, components/modules/endpoints/flows/boundaries; **Operational**, the complete registry of logs/state/configs/paths/environment variables/tools, their locations and use; **Rationale**, why non-trivial decisions hold. Replace affected descriptions. Guards, review and lifecycles need no historical search. History: README.md, tags and Git.

## Chapters

- [1. High-Level Architecture](architecture/01-high-level-architecture.md)
- [1.1 Module Directory](architecture/01-module-directory.md)
- [2. Startup / Onboarding Flow](architecture/02-startup-onboarding-flow.md)
- [3. Web UI Pages & Buttons](architecture/03-web-ui-pages-and-buttons.md)
- [4. Server API Endpoints](architecture/04-server-api-endpoints.md)
- [5. Supervisor Loop](architecture/05-supervisor-loop.md)
- [6.1 Agent Core — Loop, Tools & Safety](architecture/06-1-loop-tools-and-safety.md)
- [6.2 Agent Core — Model Calls, Usage & Budget](architecture/06-2-model-calls-and-budget.md)
- [6.3 Agent Core — Delegated Subagents & Vendor Maintenance](architecture/06-3-delegation-and-subagents.md)
- [6.4 Agent Core — Self-Modification & Review Stack](architecture/06-4-self-change-and-review.md)
- [6.5 Agent Core — Planning, Memory, Skills & External Tools](architecture/06-5-planning-memory-and-skills.md)
- [7. Configuration (ouroboros/config.py)](architecture/07-configuration.md)
- [8. Git Branching, CI, and Build](architecture/08-git-branching-ci-and-build.md)
- [9. Shutdown & Process Cleanup](architecture/09-shutdown-and-process-cleanup.md)
- [10. Key Invariants](architecture/10-key-invariants.md)
- [11. Frozen Contracts v1 (`ouroboros/contracts/`)](architecture/11-frozen-contracts-v1.md)
- [12. Host Service, Companion Processes, and Chat IDs](architecture/12-host-service-companions-and-chat-ids.md)
- [13. External Skills Layer](architecture/13-external-skills-layer.md)
