# Ouroboros v7.12.2-alpha.0 — Architecture & Reference

This is the map of my body (BIBLE P6): what I am made of, what each part lets me do, how I reach it and which limits change my choices. Each chapter opens with that answer for its part; the rest of the chapter is the engineering manual I read before I change that part, §10 holds the invariants every part keeps and the roads already closed, `ouroboros/domains.toml` names the manuals that own each module domain, and the Reference Atlas below lists the other books and specifications. This present-tense map explains structure and rationale; module docstrings own mechanisms. Its three mandatory layers: **Structural**, components/modules/endpoints/flows/boundaries; **Operational**, the complete registry of logs/state/configs/paths/environment variables/tools, their locations and use; **Rationale**, why non-trivial decisions hold. Replace affected descriptions. Guards, review and lifecycles need no historical search. History: README.md, tags and Git.

## Chapters

- [1. High-Level Architecture](architecture/01-high-level-architecture.md)
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

## Reference Atlas

- [DEVELOPMENT.md](DEVELOPMENT.md) — how a change is made: authority, naming, module size, governance delivery, review and commit protocol, rules by change class, build and CI (chapters under `docs/development/`).
- [DESIGN.md](DESIGN.md) — the visual and interaction semantics of the UI.
- [PERSISTENCE.md](PERSISTENCE.md) — every durable data-root entity: schema, writer, retention and deletion effect.
- [CLIENT_CONTRACT.md](CLIENT_CONTRACT.md) — what a client needs beyond `GET /api/contract`: ordering, outcomes, identities and cards.
- [WIDGET_HOST_PROTOCOL.md](WIDGET_HOST_PROTOCOL.md) — the host side of reviewed module widgets.
- [CREATING_SKILLS.md](CREATING_SKILLS.md) — writing, reviewing, granting and enabling a skill.
- [CHECKLISTS.md](CHECKLISTS.md) and [CHECKLISTS_ARCHIVE.md](CHECKLISTS_ARCHIVE.md) — the questions my reviewers answer, and their standing disclosures.
- [code-navigation.md](code-navigation.md) — `query_code` operations, architecture facts and coverage limits.
- [DELEGATED_ADMISSION.md](DELEGATED_ADMISSION.md) — what a delegated run's admission enforces and what it only discloses.
- [PASSIVE_QUOTA_READ.md](PASSIVE_QUOTA_READ.md) — passive subscription quota and resource reads.
- [USAGE_STORE.md](USAGE_STORE.md) — `state/usage.sqlite`, the monetary authority.
- [MODEL_SEND_OBSERVABILITY.md](MODEL_SEND_OBSERVABILITY.md) — every model-visible send is logged.
- [IMAGE_GENERATION.md](IMAGE_GENERATION.md) — image creation and editing routes.
- [DEPLOYMENT.md](DEPLOYMENT.md) — roots and ports, first run without the wizard, containers, separately hosted interfaces and the deployment policy file.
- [ANDROID_INSTALL.md](ANDROID_INSTALL.md) and [ANDROID_RECOVERY.md](ANDROID_RECOVERY.md) — the experimental Android host.
- [inventories/UI_CONTROL_TEXT_INVENTORY.md](inventories/UI_CONTROL_TEXT_INVENTORY.md) — generated inventory of UI control text.
- `ouroboros/domains.toml` — module-to-domain assignment, dependency baselines and owning manuals.
- `README.md` and `CONTRIBUTING.md` — the project overview and release history; contribution and version policy.
