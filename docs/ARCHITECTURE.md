# Ouroboros — Body Map & Architecture Reference

This resident **Body Map** and **Reference Atlas** (BIBLE P1, P6) rides with me in every mode and focus so I know what my body can do, where its limits matter, and where every deeper manual lives. Before non-trivial work on my own body, I read the owning manual chapters, `docs/architecture/10-key-invariants.md` (invariants and *Closed roads*), and relevant `docs/DEVELOPMENT.md` rules with `read_file(root="system_repo", path=...)`; for live Python/`web/` module lookup I call `query_code(op="architecture", query="module_map <D01..D20|prefix|domains>")` or inspect `ouroboros/domains.toml`. Each manual stays present-tense across **Structural** flows, **Operational** state/logs/configs/paths/env/tools (code registries project from `ouroboros/domains.toml` and docstrings), and **Rationale** beside each rule. History lives in `README.md`, tags, and Git.

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

Deep manuals and standalone specifications, read on demand via `read_file(root="system_repo", path=...)`:

- `docs/DEVELOPMENT.md` — engineering discipline, capability placement, module complexity, governance delivery, review/commit protocol, change-class rules, managed updates, mutation attribution, process custody, platform abstraction, design system, MCP, gateway boundary, build/CI (chapters under `docs/development/01`–`14`).
- `docs/DESIGN.md` — UI/UX design bible, layout, composer controls, cards, widgets, voice controls, replies, and interaction semantics.
- `docs/PERSISTENCE.md` — canonical registry of durable files/SQLite stores, writers, schemas, lifetime classes, and reset rules.
- `docs/CLIENT_CONTRACT.md` — HTTP/WS client contract, living cards, sandboxed frames, and skill UI contributions.
- `docs/CREATING_SKILLS.md` — skill manifest format, companions, connectors, hooks, review, grants, and publication protocol.
- `docs/CHECKLISTS.md` — change review, plan review, skill review, and release verification checklists.
- `docs/code-navigation.md` — `query_code` operations, syntax/architecture facts, and coverage limits.
- `docs/DELEGATED_ADMISSION.md` — delegated run admission, workspace holds, directory mutability, and release rules.
- `docs/DEPLOYMENT.md` — packaging, Docker/containers, systemd, headless, and production deployment.
- `docs/IMAGE_GENERATION.md` — direct provider image-generation contracts and media handling.
- `docs/MODEL_SEND_OBSERVABILITY.md` — physical send seals, call manifests, redaction, and context assembly traces.
- `docs/PASSIVE_QUOTA_READ.md` — passive subscription quota reads, census freshness, and rotation evidence.
- `docs/USAGE_STORE.md` — `state/usage.sqlite` schema, attempt settlement, and monetary accounting boundary.
- `docs/WIDGET_HOST_PROTOCOL.md` — sandboxed widget iframe bridge, permissions, and lifecycle protocol.
- `docs/ANDROID_INSTALL.md` & `docs/ANDROID_RECOVERY.md` — experimental ARM64 Android host provisioning, APK bridge, and recovery.
- `ouroboros/domains.toml` — SSOT mapping Python modules to domains `D01`–`D20`, domain dependency baselines, and `[references]` manual ownership.
- `README.md` — user-facing project overview and release highlights.
