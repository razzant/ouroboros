# 7. Configuration (ouroboros/config.py)

Settings have one home and default; the table projects `config.SETTINGS_DEFAULTS`, never a second authority.

`ouroboros/config.py` re-exports paths, settings/env I/O, normalization, runtime mode, skills location and PID lock while leaves own defaults and vocabularies (§10.3): `settings_scales.setting_effect` owns the six presentation timings (§3, outside actor snapshots) and `update_channels.py` owns update defaults.

Paths use `OUROBOROS_*` overrides or `launch_roots`' pre-import `APP_ROOT` (`~/Ouroboros`). A direct server defaults `REPO_DIR` to its checkout; with only `APP_ROOT`, it refuses (`AmbiguousDataRoot`) if `~/Ouroboros/data` may contain memory. `resolve_port_file` relocates only the hosting root's port file. Unset work roots follow the initialization witness basis (`home` = `~/Ouroboros`, `app_root` = `APP_ROOT`); an `unknown` basis refuses (`WorkRootUnavailable`) while the two candidates differ. `settings.json` under the data root requires its lock for writes. `secret_masking.py` repairs only disk-recognized placeholders before env precedence, never real env credentials. Under `OUROBOROS_SETTINGS_SHA256`, `settings_integrity.py` treats the isolated child's seeded snapshot as an owner-authored trust root: reads verify every byte without a sibling lock and all writers refuse.

`openrouter_attribution.py` owns app identity (URL, title, categories; forks own theirs), sent only to `https://openrouter.ai`:443 without userinfo on a no-redirect client, else unlabelled.

### Reading and writing the settings document

`TaskSettingsSnapshot` holds per-task entry Context/Effort, review settings and catalog; cognition stays private through Resume (fallback uses pre-room env; `RoomResolver` tracks lineage and conversion). Before Main changes, `composer_cognition.py` freezes legacy rooms while new projects copy Main's Effort and Subagents and start in Max; Main effort is independent of helper bounds (`_effort_range_format=1`), and Access stays boot-pinned. Route keys (`route_choice.ROUTE_SETTING_KEYS`) read the current saved document (none saved: as every other key; absent ≠ empty; route overrides win; deployment policy reads last). Subagents stays live: Main's `OUROBOROS_SUBAGENTS_MODE` or project `cognition.subagents_mode`, default Auto (legacy rooms included); it bypasses checkpoints, and Settings only masks it (`metadata.lead`: CLIENT_CONTRACT §9, `route_choice.py`). Immediate keys, writers, granted skill keys and process `env_from_settings` read current disk, clears included (credential-free skill children reread disk; payloads carry operator env; reload failures disclose prior env and missing values).

`normalize_settings_raw()` runs pure and idempotent before defaults in `load_settings`, `_owner_read_settings_raw` and `build_colab_settings`. It runs `safety_config.normalize_consciousness_safety` (seeding absent coverage from saved owner coverage; explicit choices stay and guards compare the disk-derived value) and `review_pool_migration.apply_at_read_seam` (mapping run seats to roster rows while retaining unconvertible keys, and minting factory rows on comma-only or lane-free installs), then drops retired keys and comma-lists—including `OUROBOROS_DELEGATE_WAIT_SEC`/`OUROBOROS_DELEGATE_WAIT_MAX_SEC` (no wait-window setting remains)—with a once-per-set notice; dropped keys stay dropped. Review env keys are inert (one startup notice). Colab budget/mode: launch > saved Drive > fresh `200`/`advanced` (no default worker cap). Unauthored slots are proposed on read (§2), respelled unpersisted by `apply_runtime_provider_defaults`, and echoed in `OUROBOROS_PROJECTED_PROPOSALS` by `apply_settings_to_env` (inherited by children, distinct from operator env). Provider Test (minus search Model) and credential plans share `provider_models.ACTIVE_MODEL_SETTING_KEYS`, Safety pins included (blank inherits). Unmarked saved `OUROBOROS_MAX_WORKERS=10` (Colab first-launch `1`; explicit launch wins), `OUROBOROS_MODEL_MAX_CONCURRENCY=3` and `OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT=6` read empty until marked (`_UNCHOSEN_SAVED`; others stay; typos: `10`/`3`/`6`). `OUROBOROS_MAX_ROUNDS`, `OUROBOROS_TASK_ABS_CEILING_SEC` and `OUROBOROS_REVIEW_MAX_CYCLES` share `parse_positive_or_unlimited` (`unlimited` → `None`; money, deadlines, Stop/Panic still bind; silence triggers attention, §5): missing legacy keys and blank/non-positive/malformed values for the two task bounds (`OUROBOROS_MAX_ROUNDS` and `OUROBOROS_TASK_ABS_CEILING_SEC`) take `OPTIONAL_BOUND_LEGACY` (200 rounds / 21600 s, announced once by `upgrade_notices.py`), whereas missing or malformed `OUROBOROS_REVIEW_MAX_CYCLES` fails closed to its shipped default `2` (`review_cycles.py`). Operations take only true bounds (window, deadline, finite execution), else none. Code constant `HOST_MEMORY_RESERVE_FRACTION = 0.10` (not a setting) sets shared memory reserve (DEVELOPMENT §9).

Five writers (`tests/_shared.py::SETTINGS_WRITERS`; `settings_writers` rejects a sixth, startup `safety_config._migrate_consciousness_safety` exempt) write `serialize_settings()` bytes under the file lock (in-process: `settings_document_mutation` through projection). `config.save_settings`, `owner_settings._owner_update_settings` and `packaged_cli._save_settings` run `prepare_settings_for_persist`: unreadable disk refuses → deployment policy → `review_pool_receipts.persist_write_receipts` (the review-pool receipts this write owes, before the pre-image goes) → disk-authored silence → context/Safety ratchets. Exempt from that prologue by design: `context_mode_compat.normalize_and_persist_context_mode_compat` rewrites only the raw context pair byte-exact without receipts, and `colab_bootstrap.write_colab_settings` runs `persist_write_receipts` itself for a foreign (Drive) root before writing its document (own document uses `_owner_write_settings`).

`_owner_update_settings` checks preconditions (conditional writes are checked per key under the lock), commits canonical bytes atomically under the file lock, and then derives the change receipt and per-key tokens under that same lock; a receipt failure after commit cannot undo the save. Endpoints record sanitized audit receipts afterwards, best-effort (`record_settings_change`, declared origin, Main row per DESIGN; neither step can undo or fail the save). A context-mode switch dates its commit under `settings_document_mutation` and then sends one `context_mode_changed` System row to the changed room (`record_context_mode_change`). GET computes and pairs values with `_meta.tokens` (canonical value + presence hash, scalar secrets excluded; never persisted). Generic saves bind read digest (rereading and remerging once on miss, then 409); ordinary scalars use last-writer-wins, and incidental saves (launcher, `set_tool_timeout`, GitHub repo) preserve other fields via `fields_on_saved_document` (wire, codes, absent keys and saved/failed/unknown outcomes: CLIENT_CONTRACT §9).

### LLM output token budgets

OpenRouter/Anthropic-compatible calls send `max_tokens`; direct OpenAI sends `max_completion_tokens` (Responses: `max_output_tokens`), never by model name (§10 invariant 13). Budgets are floors lowered by known route maxima:

| Surface | Output-token budget |
|---------|---------------------|
| `LLMClient.chat()` / `chat_async()` defaults | 65,536 |
| Main task loop (`loop_llm_call.MAIN_LOOP_MAX_TOKENS`) | 65,536 |
| `LLMClient.vision_query()` and VLM tools (`analyze_screenshot`, `vlm_query`) | 32,768 |
| Light memory drafts (`consolidator.LIGHT_ANSWER_CEILING_TOKENS`) | 16,384 |
| Post-task evolution promotion decision (`post_task_evolution`) | 8,192 |
| Skill publish PR body generation | 8,192 |
| Project naming LIGHT one-shot (`project_naming.llm_project_name`) | 256 |
| Provider Test (`llm_probe`; direct OpenAI's tool call 4,096) | 16 |

Naming and translation share one pre-schedule deadline across preparation, slots and recovery (earliest wins; detached naming accounts cost without publishing).

### Default settings

One row per key (`settings_defaults.py`). `unset`-default rows (plus `Env-only` rows) have no `settings.json` carrier; `(migrated)`/`(none)` rows convert retired disk keys at load (env inert); `ENDPOINT_AUTHORED_SETTINGS` are endpoint-authored and disk-only.

| Key | Default | Description |
|-----|---------|-------------|
| OPENROUTER_API_KEY | "" | OpenRouter key |
| OPENAI_API_KEY | "" | Direct-OpenAI key |
| OPENAI_API_SURFACE | "" | `chat_completions`/empty or `responses` (new keys; rows: `api_surface`) |
| OPENAI_BASE_URL | "" | Legacy compatible URL (with `OPENAI_API_KEY`) |
| OPENAI_COMPATIBLE_API_KEY | "" | OpenAI-compatible key |
| OPENAI_COMPATIBLE_BASE_URL | "" | OpenAI-compatible URL (own key; blank = no auth) |
| CLOUDRU_FOUNDATION_MODELS_API_KEY | "" | Cloud.ru key |
| CLOUDRU_FOUNDATION_MODELS_BASE_URL | `https://foundation-models.api.cloud.ru/v1` | Cloud.ru base URL |
| GIGACHAT_CREDENTIALS | "" | Auth key |
| GIGACHAT_USER | "" | User login |
| GIGACHAT_PASSWORD | "" | Password |
| GIGACHAT_SCOPE | `GIGACHAT_API_PERS` | API scope |
| GIGACHAT_BASE_URL | `https://api.giga.chat/v1` | Base URL |
| GIGACHAT_VERIFY_SSL_CERTS | `true` | TLS check |
| GIGACHAT_PROFANITY_CHECK | "" | Profanity passthrough |
| OUROBOROS_EXTRA_CA_BUNDLE | "" | Extra CA PEM for first-party provider trust bundles |
| ANTHROPIC_API_KEY | "" | Direct-Anthropic key |
| MINIMAX_API_KEY | "" | MiniMax key |
| MINIMAX_REGION | "" | Region (empty = `global_en`) |
| DEEPSEEK_API_KEY | "" | DeepSeek key (`deepseek::`) |
| ZAI_API_KEY | "" | Z.ai key (`zai::`) |
| ZAI_PLAN | "" | Endpoint: empty/`payg` or `coding` |
| OUROBOROS_CONNECTIONS | [] | Extra direct-provider keys/URLs, never exported |
| OUROBOROS_NETWORK_PASSWORD | "" | Non-loopback HTTP password; unset warns (§8 Docker) |
| OUROBOROS_SERVER_HOST | 127.0.0.1 | HTTP bind host (`0.0.0.0` Docker/non-loopback) |
| OUROBOROS_UI_DIR | "" | UI source (§4, setup/login included): empty builtin, `none`, or trusted dir with `index.html`; restart-required; saved value (blank too) beats env, never exported; merged save never authors absent (`SAVED_CHOICE_SETTINGS`) |
| OUROBOROS_UPDATE_CHANNEL | `stable` | Feed `stable`/`qa`/`development` (§8) |
| OUROBOROS_MANAGED_UPDATE_FETCH_TIMEOUT_SEC | 300 | Managed-update fetch ceiling |
| OUROBOROS_RESCUE_GIT_TIMEOUT_SEC | 300 | Per-process rescue Git ceiling |
| OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD | unset | `1` allows saving non-loopback bind without password |
| OUROBOROS_MODEL | google/gemini-4-hydrogen | Main model |
| OUROBOROS_MODEL_HEAVY | "" | Migration input only, outside `ACTIVE_MODEL_SETTING_KEYS` |
| OUROBOROS_MODEL_LIGHT | openai/gpt-6-luna | Light model |
| OUROBOROS_MODEL_SAFETY | (empty) | Safety checker (empty → Light → Main); outside Cyber Pro only owner writes (Safety endpoints, model-wait card) change it; provider re-proposal of unauthored slot saves (audited); deployment checker replaces |
| OUROBOROS_MODEL_ACCOUNTS | {} | Role managed-account pins (empty = Auto) |
| OUROBOROS_PROCESSING_PREFERENCE | "" | Global `standard`/`fast`/`economy` (empty = adapter default) |
| OUROBOROS_MODEL_PROCESSING_PREFERENCES | {} | Role tiers (inherit global; only `main` accepts `ultrafast`) |
| OUROBOROS_MODEL_CONTEXT_WINDOWS | {} | Role context-window assertions (0 = Auto; no reviewer authority) |
| OUROBOROS_MODEL_VISION | "" | Vision model (empty inherits) |
| OUROBOROS_IMAGE_INPUT_MODE | auto | Send-time image routing (`vision_routing.py`) |
| OUROBOROS_VISION_CAPTION_TIMEOUT_SEC | 90 | Caption ceiling |
| OUROBOROS_MODEL_CONSCIOUSNESS | "" | Consciousness model (empty inherits) |
| OUROBOROS_MODEL_FALLBACKS | openai/gpt-6.1-terra | Fallback chain, Safety too on outage (not owner-guarded; `fallback_cooldown.py`; blank = none) |
| OUROBOROS_SERVED_MODEL_REDOS | 2 | Redos when another model answers (0-5) |
| OUROBOROS_MODEL_MAX_CONCURRENCY | "" | Install-wide fair queue per model/credential/endpoint (awaited turns first; blank: APIs unlimited, built-in send/count/Test share 1; ≤0 off; live in Settings › Models › More; Claudexor owns its queue) |
| OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC | 60 | Project-naming ceiling |
| OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC | 8 | Inline naming bound on card→project (`gateway/projects.py`) |
| OUROBOROS_UI_TRANSLATION_TIMEOUT_SEC | 120 | Translation batch ceiling, slot wait + call (`ui_translation.py`) |
| OUROBOROS_UI_LANGUAGE | "" | BCP-47 UI language (`""` = not chosen; live; `POST /api/ui/i18n/language` only, `ENDPOINT_WRITTEN_SETTINGS`; §3) |
| OUROBOROS_FALLBACK_COOLDOWN_ENABLED | true | 429-aware per-process cooldown |
| OUROBOROS_FALLBACK_COOLDOWN_SEC | 120 | Cooldown window |
| OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL | 1 | Attempts per model in fallback walk |
| OUROBOROS_MODEL_DEEP_SELF_REVIEW | (migrated) | Deep-review helper row in `OUROBOROS_SUBAGENTS`, no Reviewer mark (§6) |
| OUROBOROS_MAX_WORKERS | "" | Acting-project cap (blank: memory admits) |
| OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT | "" | Acting-child cap per root (`runtime_limits.py`) |
| OUROBOROS_MAX_SUBAGENT_DEPTH | 3 | Subagent tree depth |
| OUROBOROS_DISABLE_MANAGED_UPDATES | unset | `1` disables managed updates (`git_ops.py`) |
| OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS | "" | Mutative-subagent Auto override |
| OUROBOROS_SUBAGENT_WORKTREE_ROOT | "" | Acting-worktree root (empty = basis) |
| OUROBOROS_SUBAGENT_PROJECTS_ROOT | "" | Projects root (empty = basis) |
| OUROBOROS_SUBAGENTS_MODE | auto | Live delegation `off`/`auto`/`swarm` (§6) |
| OUROBOROS_SUBAGENTS | "" | Catalog (§6, `configured_subagents.py`), saved rows unexported |
| OUROBOROS_SUBAGENT_HARNESS | "" | Legacy harness migration input |
| OUROBOROS_SUBAGENT_PROFILE | "" | Legacy profile migration input |
| OUROBOROS_DELIVERABLES_ROOT | "" | Deliverables root (empty = basis; `tool_access.py`) |
| OUROBOROS_GC_RETENTION_DAYS | 7 | Unified GC retention (`retention.py`) |
| OUROBOROS_RESTART_DRAIN_MAX_SEC | 120 | Restart drain bound |
| TOTAL_BUDGET | 200.0 | Global USD budget over known spend (≤0 unlimited; live per call, `resolve_total_budget_usd`) |
| OUROBOROS_PER_TASK_COST_USD | 50.0 | Hard cap per task tree, confirmed + estimated (§6) |
| OUROBOROS_RUB_USD_RATE | "" | Manual RUB→USD rate |
| OUROBOROS_PRICING_TTL_SEC | 21600 | Pricing cache TTL |
| OUROBOROS_TOOL_TIMEOUT_SEC | 600 | Default tool timeout |
| OUROBOROS_FINALIZATION_GRACE_SEC | 120 | Finalization grace |
| OUROBOROS_WEBSEARCH_MODEL | "" | `web_search` Model (§6) |
| OUROBOROS_WEBSEARCH_BACKEND | auto | `web_search` Source |
| OUROBOROS_MAIN_WEB_SEARCH | off | Inline web search (§6) |
| OUROBOROS_MAIN_WEB_SEARCH_ENGINE | auto | Inline-search engine |
| OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS | 10 | Inline-search result cap |
| OUROBOROS_OR_PROVIDER | "" | OpenRouter provider-routing preference |
| OUROBOROS_SEARCH_CODE_WALL_SEC | 45 | `search_code` wall budget |
| OUROBOROS_PRESENTATION | unset | `desktop_window`/`browser_fallback`/`android_app` (absent = `web`) |
| OUROBOROS_DESKTOP_BACKGROUND | unset | `1` when desktop launcher can run hidden |
| OUROBOROS_EXTERNAL_HOST_UPDATE | unset | External-host installer supporting `--check` |
| OUROBOROS_EXTERNAL_HOST_RESULT | unset | Launcher-verified artifact facts for one generation, not a persisted PASS |
| OUROBOROS_USER_FILES_ROOT | "" | Env-only: `user_files` jail root (empty = `$HOME`) |
| OUROBOROS_CHROMIUM_SANDBOX | unset | `off` = admin Chromium sandbox opt-out (§6) |
| OUROBOROS_OBSERVABILITY_KEEP_RAW | unset | Truthy persists raw observability payloads |
| OUROBOROS_REVIEWER_SLOTS | (migrated) | Read once into `OUROBOROS_SUBAGENTS` reviewer rows; unreadable stays for owner (rollback: `before` in `state/review_migrations/<ts>-slots-to-pool.json`) |
| OUROBOROS_SUBSCRIPTION_PRESET_VERSION | "" | One-shot install-preset marker (`ENDPOINT_AUTHORED_SETTINGS`, §2); absence authorizes nothing |
| OUROBOROS_SUBAGENT_PRESET_RECEIPT | "" | Install-preset receipt; likewise |
| OUROBOROS_ONBOARDING_COMPLETED_AT | "" | Onboarding timestamp; likewise |
| OUROBOROS_ONBOARDING_ATTEMPT_ID | "" | Onboarding attempt (§2); likewise |
| OUROBOROS_ONBOARDING_CONSCIOUSNESS | "" | Setup Consciousness switch (PERSISTENCE); likewise |
| OUROBOROS_TASK_REVIEW_MODE | auto | Result review `auto`/`required`/`off` (§6) |
| OUROBOROS_SAFETY_MODE | full | Safety mode (fresh desktop wizard may set `light`; lowering owner-guarded, §6) |
| OUROBOROS_CONSCIOUSNESS_SAFETY_MODE | full | Consciousness Safety: own wakes, places and what they start |
| OUROBOROS_SAFETY_OWNER_RULES | (empty) | Owner standing rules for every mind/check; owner-only |
| OUROBOROS_SAFETY_MAX_TOKENS | 2000 | Safety output budget |
| OUROBOROS_SAFETY_CALL_TIMEOUT_SEC | 60 | Safety call ceiling |
| OUROBOROS_WEBSEARCH_TIMEOUT_SEC | 480 | `web_search` ceiling |
| OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC | 2 | Stopped direct-turn round wait before custody retries (0-10) |
| OUROBOROS_ONBOARDING_SNAPSHOT_TIMEOUT_SEC | 45 | Onboarding Claudexor snapshot bound (wedged daemon refuses, never hangs) |
| OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC | 30 | Owner-write lock bound; contention refuses before transform runs (`owner_settings.py`) |
| OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC | 2700 | LLM read timeout |
| OUROBOROS_REVIEW_MAX_CYCLES | "2" | Shared result/commit/skill review cap (`unlimited` = no count cap; `review_cycles.py`, §6) |
| OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES | (none) | Alias → `OUROBOROS_REVIEW_MAX_CYCLES` (passes + 1) |
| OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC | unset | Logical review window (late results stay in custody) |
| OUROBOROS_REVIEW_MAX_TOKENS | 65536 | Env-only: reviewer output budget (floor 8192) |
| OUROBOROS_PREFLIGHT_TIMEOUT_SEC | unset | Bounded preflight budget (`preflight_runner.py`; unset 1800, Android 3600, full none) |
| OUROBOROS_PREFLIGHT_SERIAL | unset | `1` serial pytest; scrubbed from candidate |
| OUROBOROS_PREFLIGHT_TEST_WORKERS | unset | Env-only: requested xdist workers, capped by memory and other runs (`test_parallelism.py`); scrubbed from candidate |
| OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS | true | Auto-grant manifest requests to hash-bound accepted bytes (§13) |
| OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS | true | Derived Built-in trust (`false` = review) until sources saved |
| OUROBOROS_CONTEXT_MODE | max | Main `nano`/`low`/`max` (owner-only outside Cyber; §6) |
| OUROBOROS_CONTEXT_MODE_AUTO_LOW | false | Tombstone: normalization writes `false`; persisted `low` honoured only beside it (`context_mode_compat.py`) |
| OUROBOROS_RUNTIME_MODE | advanced | Access `light`/`advanced`/`pro`/`cyber_pro`, next boot (§6) |
| OUROBOROS_SKILLS_REPO_PATH | "" | My folder (`my-folder`) in place even with saved sources; removing row clears it, other folders stay |
| MCP_ENABLED | false | MCP toggle (§6) |
| MCP_SERVERS | [] | Saved servers, never exported; changes re-list |
| MCP_TOOL_TIMEOUT_SEC | 60 | Per listing/tool call |
| OUROBOROS_HUB_CATALOG_URL | `https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json` | Derives two hub rows (non-GitHub raw URL needs `OUROBOROS_SKILL_SOURCES`) |
| OUROBOROS_CLAWHUB_REGISTRY_URL | `https://clawhub.ai/api/v1` | ClawHub URL |
| OUROBOROS_SKILL_SOURCES | "" | Skill sources (§13; empty = derived, unwritten) |
| OUROBOROS_SKILL_SOURCE_SUGGESTIONS | "" | Build Add-source suggestions (never auto-connected) |
| OUROBOROS_STARTER_SKILLS | "" | Starter ids (absent/`""` = recommended; `[]`/`"[]"` = none; built-ins before git) |
| OUROBOROS_STARTER_SKILLS_TURN_ON | true | Enable starter installs when ready (owner Off persists) |
| OUROBOROS_PROMPT_CACHE_TTL | 1h | Prompt-cache tier `default`/`5m`/`1h` stamped at send |
| OUROBOROS_EFFORT_MIN | low | Main helper floor (`settings_scales.effort_range`) |
| OUROBOROS_EFFORT_TASK | medium | Main/project primary effort (Auto helpers clamp) |
| OUROBOROS_EFFORT_MAX | high | Main helper ceiling; Auto reviewers, evolution and wakes |
| OUROBOROS_EFFORT_REVIEW | (migrated) | Fills missing triad-seat effort |
| OUROBOROS_EFFORT_SCOPE_REVIEW | (migrated) | Consumed into scope reviewer rows |
| OUROBOROS_EFFORT_DEEP_SELF_REVIEW | (migrated) | Consumed into deep-review helper row |
| OUROBOROS_RETURN_REASONING | true | Request OpenRouter reasoning (stripped on direct/local) |
| OUROBOROS_REASONING_SUMMARY | auto | Display-only reasoning summary (never in history/provider) |
| OUROBOROS_TASK_IDLE_TIMEOUT_SEC | 900 | Silence-attention threshold (§5) and direct-chat outage/wait bound (§6) |
| OUROBOROS_TASK_ABS_CEILING_SEC | unlimited | Activity-independent task lifetime (s) |
| OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC | 90 | Liveness watchdog (alerts `/restart`, never frees wedged locks) |
| LOCAL_MODEL_SOURCE | "" | HF repo or path |
| LOCAL_MODEL_FILENAME | "" | GGUF filename (split first shard expanded) |
| LOCAL_MODEL_CONTEXT_LENGTH | 65536 | Context window |
| LOCAL_MODEL_N_GPU_LAYERS | 0 | GPU offload layers |
| USE_LOCAL_MAIN | false | Route Main locally |
| USE_LOCAL_HEAVY | false | Migration input only, unrouted |
| USE_LOCAL_LIGHT | false | Route Light locally |
| USE_LOCAL_CONSCIOUSNESS | false | Route Consciousness locally |
| USE_LOCAL_FALLBACK | false | Route fallback locally |
| OUROBOROS_MAX_ROUNDS | unlimited | Round limit (hot-reloadable) |
| OUROBOROS_TRANSIENT_RETRY_MAX | 6 | Same-model transient retries |
| OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC | 1800 | Skill lifecycle timeout |
| OUROBOROS_CLAUDEXOR_HARNESS_INSTALL_TIMEOUT_SEC | 300 | Connect wait (expiry leaves install running) |
| OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC | 90 | Quota-refresh POST ceiling (1-90) |
| OUROBOROS_BUNDLE_DIR | unset | Env-only: bundle root for child Node/ripgrep discovery and autostart |
| OUROBOROS_GIT | unset | Env-only: absolute path of selected Git (`git_selection.py`) |
| OUROBOROS_BG_WAKEUP_MIN | 900 | Min `set_next_wakeup` step (s), re-read per alarm |
| OUROBOROS_BG_WAKEUP_MAX | 14400 | Max wake step (s; named time beyond kept; `runtime_limits.WAKE_DEFAULT_SEC` warms prefix cache) |
| OUROBOROS_CONSCIOUSNESS_AUTONOMY | act | `observe`/`act`/`full`: `act` excludes own code/prompts, evolution, restart and settings; `full` adds evolution (`consciousness_authority.py`) |
| OUROBOROS_CONSCIOUSNESS_DAILY_USD | 15.0 | Rolling-24h ceiling on wakes + started tasks (§6; `0` = no spend) |
| OUROBOROS_CONSCIOUSNESS_MAX_TASKS | 2 | Concurrent consciousness-started tasks (`0` = none) |
| OUROBOROS_POST_TASK_EVOLUTION | false | Post-task evolution toggle; guards block self-enablement (§6) |
| OUROBOROS_POST_TASK_EVOLUTION_CADENCE | llm | `llm` or `every_n:k` (malformed → `llm`) |
| OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD | 0.0 | Remaining-global-budget start floor, not cycle cap |
| OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE | "" | Owner-only campaign bias (still reviewed) |
| LOCAL_MODEL_PORT | 8766 | Local server port |
| OUROBOROS_HOST_SERVICE_PORT | 8767 | Host Service loopback port (§12) |
| OUROBOROS_DESKTOP_KEEP_RUNNING | false | Closing desktop window keeps running (`launcher_background.py`; disk-authored consent) |
| OUROBOROS_PRESENCE_MAX_ACTIVE | 2 | Cross-process Presence turn cap (UI 1-20) |
| LOCAL_MODEL_CHAT_FORMAT | "" | Chat template override |
| GITHUB_TOKEN | "" | GitHub token (push/PR/issues) |
| GITHUB_REPO | "" | Personal `origin` repository |
| OUROBOROS_FILE_BROWSER_DEFAULT | "" | File Browser root (required on Docker/non-localhost) |

#### Review pool

Enabled `OUROBOROS_SUBAGENTS` rows with `review_eligible: true` (Settings → Agents Reviewer mark) form one helper/reviewer list (`reviewer_slot_config.review_pool_slots` in task settings scope; waves: §6 Review stack). Rows own `effort` (empty = Auto; `reviewer_slot_config.row_effort`: model name → pin → clamped wave order → range top), API `delivery` (`native` default two-part brief with tools; `packet` only for models positively known tool-less—never inferred from quota, timeouts, unknown capability or fallbacks; sessions always retrieve) and optional `route.credential_profile_id` (sessions/managed-model APIs only; empty = rotation). Changing `delivery` updates review-contract identity (a later review may be paid anew) without launching a review. A marked row switched off leaves the pool silently, whereas naming an off/unknown row (`review_change(reviewers=[…])`, `preflight_reviewer`, `/review`) refuses (`catalog_review_row`), never rerouted. A pool with no reading row asks no seat the coupling question: body commit reviews end `NOT_PERFORMED` until a reading row is marked, and Save only warns (`subagent_runtime.review_pool_save_warning`).

Save surfaces share `reviewer_slot_config.review_pool_save_error` / `review_pool_save_judgement`: malformed catalogs refuse Save and review; unmarked non-empty catalogs require `allow_empty_review_pool`. Never-configured installs (no catalog or lanes key: wizardless Docker/Colab) re-mint factory rows over env-merged reads until saved. `subscription_install_presets.factory_review_rows` follows the reachable Main: OpenRouter Main (or no access yet) → three `OPENROUTER_REVIEW_DEFAULTS`; MiniMax Main → Main, Light, Light; any other reachable Main → 1-of-1 row of its route; unreachable Main → none (naming `OUROBOROS_MODEL`). Separately, only this never-configured cell adds Auto helpers (`subscription_install_presets.factory_actor_rows`): one Auto generalist on any reachable Main, a scout when reachable Light differs in its whole binding, and `OPENROUTER_HELPER_DEFAULTS` alongside any reachable OpenRouter (none without a reachable Main). Operator env catalogs (never our echo) beat minted rows (`review_pool_migration.environment_overridable_keys`); saved catalogs or lane-minted rows shadow env.

#### Direct-provider routes

`parse_model_route`/`llm_routing.resolve_target` map `kind@name::model` onto `OUROBOROS_CONNECTIONS` (missing/disabled → `connection_unavailable`, no fallback key). `compute_direct_review_seats` covers each direct provider; `exclusive_direct_kind` requires one direct kind without OpenRouter, a compatible URL or `OPENAI_BASE_URL`. Proposals (`proposal_connection`: built-in, sole named, else `main_connection_choice`) respell saved ids only for built-in; off-provider Main mints none (DEVELOPMENT "Provider Independence"). Only the official Go Chat URL (ASCII case-insensitive scheme/host, exact path) gets Ouroboros's User-Agent and model-scoped conversation ID (allowance ≠ cash); Claudexor's `opencode` Agent runs the external CLI.

Direct prefixes hit official APIs while proxies/mirrors (and Z.ai's China host) use `openai-compatible::` (`deepseek/...` and `zai/...` stay OpenRouter); DeepSeek, MiniMax and Z.ai map effort via provider aliases and disclose tier changes as `reasoning_effort_clamped` (probes: DEVELOPMENT "External facts", `provider_models.py`, `llm_openai_compatible.py`):
- **DeepSeek (`deepseek::`)**: `DEEPSEEK_BASE_URL`, `DEEPSEEK_REASONING_EFFORT_ALIASES`; forced tools disable thinking.
- **MiniMax**: `MINIMAX_DIRECT_DEFAULTS` (saved choices stay); exact direct `MiniMax-M3.1-Flash` maps none/minimal→low, ultra→max (`MINIMAX_FLASH_EFFORT_ALIASES`); `reasoning_split=true` replays verbatim same-route (display-only details; live wire unverified). Move PAYG M2.7 pins before [18 Nov 2026](https://platform.minimax.io/docs/faq/models.md).
- **Z.ai (`zai::`)**: GLM via `ZAI_PLAN`/`resolve_zai_base_url` and `ZAI_REASONING_EFFORT_ALIASES` (absent `reasoning_effort` runs at max); forced tool choice keeps its tier; 429 code 1113 is billing (Coding Plan keys on PAYG included), so Test reports `No credits`, not `Rate limited`.
- **GigaChat (`gigachat::`)**: native `gigachat` library (`llm_gigachat.py`, not compatible lane), one tool call/turn, omits `reasoning_effort` (hidden reasoning can exhaust the output budget and return empty content); with no live cost source, cost stays unknown, never a static tariff; scope rows retrieve on their own window (§6 Review stack).

---
