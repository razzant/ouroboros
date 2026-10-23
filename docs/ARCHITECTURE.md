# Ouroboros v7.8.0 — Architecture & Reference

This present-tense map (BIBLE P6) has three mandatory layers: **Structural**, components/modules/endpoints/flows/boundaries; **Operational**, the complete registry of logs/state/configs/paths/environment variables/tools, their locations and use; **Rationale**, why non-trivial decisions hold. Replace affected descriptions; mechanisms live in named docstrings. Guards, review and lifecycles need no historical search. History: README.md, tags and Git.

Two helpers support [Agent Core](architecture/06-agent-core.md): `ouroboros/consciousness_view.py` projects places, counted roots and bounded recent wakes; `ouroboros/usage_origin.py` owns task usage scopes and immutable billing/cause attribution in the usage ledger.

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
