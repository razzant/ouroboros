# 7. Configuration (ouroboros/config.py)

One key, one home, one default: the table below is a test-mirrored projection of `config.SETTINGS_DEFAULTS`, never a second authority.

`ouroboros/config.py` is the import surface: the path constants (`APP_ROOT`, `REPO_DIR`, `DATA_DIR`, `SETTINGS_PATH`, `PID_FILE`, `PORT_FILE`), `RESTART_EXIT_CODE`, `AGENT_SERVER_PORT`, `load_settings()`/`save_settings()`, `apply_settings_to_env()`, `normalize_runtime_mode()` (one clamp for save, read coercion and onboarding), `get_runtime_mode()`, `get_skills_repo_path()`, the PID lock. Vocabularies, new keys and defaults belong to its re-exported leaves (§10 invariant 3), never the facade. `settings_scales.IMMEDIATE_SETTINGS` and `RESTART_REQUIRED_SETTINGS` name immediate and restart-required keys; other keys apply to the next task. `update_channels.py` owns the update channel and update-network defaults.

Path constants take their `OUROBOROS_*` overrides or derive from `APP_ROOT` (`~/Ouroboros`); a direct server's implicit `REPO_DIR` is its executing checkout, and `resolve_port_file` relocates only the hosting root's port file. Exception: a server given only `APP_ROOT` keeps `~/Ouroboros/data` (`server_process.DATA_DIR`); deriving it would silently start a fresh brain. The document is `data/settings.json` under the data root. Writes require the settings file lock. Secret placeholders are repaired only when recognized on disk, before environment precedence resolves, so a real environment credential is never read as a mask (`secret_masking.py`). With `OUROBOROS_SETTINGS_SHA256` set (`settings_integrity.py`), the seeded snapshot is an isolated child's owner-authored trust root: every read verifies all bytes without creating a sibling lock file, and every writer refuses.

`openrouter_attribution.py` is the application-identity SSOT for every first-party paid OpenRouter request (canonical URL + `X-OpenRouter-Title`); a fork uses its own URL rather than competing to rename one app record.

### Reading and writing the settings document

Each task keeps its entry settings (`settings_integrity.TaskSettingsSnapshot`): models, keys and review settings stay private; absent/empty differ and explicit route overrides win. Room Context/Effort overlays never enter process env. `RoomResolver` follows conversion/lineage for new participants; Resume restores the actor's choice before dispatch/context assembly. `composer_cognition.py` copies Main once on project creation/recovery and freezes old rows before Main changes. Primary effort is independent of ordered helper bounds; old documents retain their effective range; internal `_effort_range_format=1` marks independent Main triples. Immediate keys, writers and granted skill keys read current disk, including cleared keys (`skill_settings`: credential-free children reread it; payloads carry operator variables). Access stays boot-pinned; reload failure discloses prior-env fallback and unavailable document-only values.

`normalize_settings_raw()` is the only translation before every reader merges defaults (`load_settings()`, `_owner_read_settings_raw()`, `build_colab_settings()`); a dropped spelling is never promoted. Review-lane keys (`REVIEW_POOL_MIGRATED_SETTING_KEYS`) migrate, not drop: `review_pool_migration.apply_at_read_seam` maps every effectively run lane seat to an `OUROBOROS_SUBAGENTS` reviewer row (Review pool); refused values stay untouched in the document. Retired comma-lists and keys (`settings_defaults.RETIRED_SETTING_KEYS`/`RETIRED_COMMA_LIST_SETTING_KEYS`) drop with an owner notice; comma-list-only and lane-free installs get factory rows; no path reads review keys from the process environment; the boot says so once (`server_maintenance._startup_environment_review_notice`). The pure, idempotent function keeps reads read-only; every in-process reader uses `settings_integrity.read_settings_json_verified` and refuses a changed pinned snapshot. The loader proposes unauthored slots (§2); `server_runtime.apply_runtime_provider_defaults` only respells, unpersisted. `apply_settings_to_env` echoes projected slot/catalog values in internal `OUROBOROS_PROJECTED_PROPOSALS` (no setting or lever), inherited by children and distinguished from operator variables on read.

Optional bounds. `OUROBOROS_MAX_ROUNDS` and `OUROBOROS_TASK_ABS_CEILING_SEC` accept a positive integer or `unlimited` (`settings_scales.parse_positive_or_unlimited`, also used by the review-cycle cap) and ship `unlimited`: fresh installs are bounded by money, deadlines, Stop/Panic and the idle rail, not age or rounds. Older documents missing the key retain their prior finite value (`settings_scales.OPTIONAL_BOUND_LEGACY`: 200 rounds / 21600 s); blank, zero, negative or malformed values also use it, with a log line, never "no bound"; `upgrade_notices.py` names the applicable finite limit to the owner once. Readers (`runtime_limits.get_max_rounds`, `get_task_abs_ceiling_sec`) return `None` for no bound. Operations inheriting task lifetime use `operation_window_sec()`: never unbounded, never a transport bound.

Five functions persist settings via `serialize_settings()` and byte-exact writes, never text mode (LF becomes CRLF on Windows). Three write this process's document via `prepare_settings_for_persist()`, the sole point applying the disk-authored silence rule and ordinary-mode context/safety ratchets against disk values: `config.save_settings()`, `gateway/owner_settings._owner_update_settings()`, `packaged_cli._save_settings()`. Two exemptions: `context_mode_compat.normalize_and_persist_context_mode_compat()` (rewrites only an existing raw mapping's compat pair) and `colab_bootstrap.write_colab_settings()` (generates a foreign data root's document). Every writer, Colab's included, persists review-pool migration receipts (`review_pool_receipts.persist_write_receipts`) before replacing the pre-image, which only that process still holds. `tests/_shared.py::settings_writers` closes the inventory; a sixth writer fails the tripwire.

Owner endpoints change one decision without owning the document: `_owner_update_settings(transform, expected_digest)` runs the whole read-change-write under one settings lock; no-change never rewrites the file, and a digest mismatch (`settings_document_digest()`) refuses before the transform, never reverting a concurrent owner change key by key while claiming "saved". Incidental writers (launcher, `set_tool_timeout`) land one field (`fields_on_saved_document`).

### LLM output token budgets

OpenRouter and Anthropic-compatible calls send `max_tokens`; official direct OpenAI Chat sends `max_completion_tokens`. Model-name prefixes are not capability authority; only exact-route, success-confirmed wire evidence adapts a request. Budgets are floors, lowered only by a route's known maximum response; constants are the numeric SSOT:

| Surface | Output-token budget |
|---------|---------------------|
| `LLMClient.chat()` / `chat_async()` defaults | 65,536 |
| Main task loop (`loop_llm_call.MAIN_LOOP_MAX_TOKENS`) | 65,536 |
| `LLMClient.vision_query()` and VLM tools (`analyze_screenshot`, `vlm_query`) | 32,768 |
| Context compaction round summaries | 32,768 |
| Review synthesis dedup | 16,384 |
| Scratchpad consolidation, Light memory drafts (`consolidator.LIGHT_ANSWER_CEILING_TOKENS`) | 16,384 |
| Post-task evolution promotion decision (`post_task_evolution`) | 8,192 |
| Skill publish PR body generation | 8,192 |
| Update letter LIGHT one-shot (`update_letter.UPDATE_LETTER_MAX_TOKENS`) | 1,024 |
| Project naming LIGHT one-shot (`project_naming.llm_project_name`) | 256 |
| Provider Test (`llm_probe.PROVIDER_TEST_MAX_TOKENS`) | 16 |

### Default settings

One row per `config.SETTINGS_DEFAULTS` key (defaults canonical in `settings_defaults.py`). Env-only rows are operator environment levers without settings.json carriers; alias and `(migrated)` rows name retired keys converted at read.

| Key | Default | Description |
|-----|---------|-------------|
| OPENROUTER_API_KEY | "" | OpenRouter credential |
| OPENAI_API_KEY | "" | Official direct-OpenAI credential |
| OPENAI_BASE_URL | "" | OpenAI base-URL override; legacy compatible endpoint (only with `OPENAI_API_KEY`) |
| OPENAI_COMPATIBLE_API_KEY | "" | OpenAI-compatible endpoint credential |
| OPENAI_COMPATIBLE_BASE_URL | "" | OpenAI-compatible endpoint base URL; only with its own key (blank = no auth) |
| GOOGLE_API_KEY | "" | Google AI (Gemini API) key; calls need it |
| CLOUDRU_FOUNDATION_MODELS_API_KEY | "" | Cloud.ru credential |
| CLOUDRU_FOUNDATION_MODELS_BASE_URL | `https://foundation-models.api.cloud.ru/v1` | Cloud.ru base URL |
| GIGACHAT_CREDENTIALS | "" | GigaChat auth key |
| GIGACHAT_USER | "" | GigaChat user login |
| GIGACHAT_PASSWORD | "" | GigaChat password |
| GIGACHAT_SCOPE | `GIGACHAT_API_PERS` | GigaChat API scope |
| GIGACHAT_BASE_URL | `https://api.giga.chat/v1` | GigaChat base URL |
| GIGACHAT_VERIFY_SSL_CERTS | `true` | GigaChat TLS verification |
| GIGACHAT_PROFANITY_CHECK | "" | GigaChat profanity filter passthrough |
| OUROBOROS_EXTRA_CA_BUNDLE | "" | PEM file added to the trust bundle of every first-party provider call |
| ANTHROPIC_API_KEY | "" | Official direct-Anthropic credential |
| MINIMAX_API_KEY | "" | MiniMax credential |
| MINIMAX_REGION | "" | MiniMax region (empty = `global_en`) |
| DEEPSEEK_API_KEY | "" | DeepSeek direct key (`deepseek::` values; Direct-provider routes) |
| ZAI_API_KEY | "" | Z.ai direct key (`zai::` values; Direct-provider routes) |
| ZAI_PLAN | "" | Z.ai endpoint: empty/`payg` = pay-as-you-go, `coding` = Coding Plan |
| OUROBOROS_NETWORK_PASSWORD | "" | Non-loopback HTTP gate password; unset only warns (§8 Docker) |
| OUROBOROS_SERVER_HOST | 127.0.0.1 | HTTP bind host (`0.0.0.0` for Docker/non-loopback) |
| OUROBOROS_UI_DIR | "" | Interface (§4, optional setup/login included): empty builtin, `none`, or a trusted directory with `index.html`; restart-required; a saved value, blank too, beats the environment, is never exported, and a merged save never authors an absent one (`SAVED_CHOICE_SETTINGS`) |
| OUROBOROS_UPDATE_CHANNEL | `stable` | Update feed stable/qa/development (§8) |
| OUROBOROS_MANAGED_UPDATE_FETCH_TIMEOUT_SEC | 300 | Managed-update fetch ceiling |
| OUROBOROS_RESCUE_GIT_TIMEOUT_SEC | 300 | Per-process ceiling on rescue Git commands |
| OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD | unset | Env-only: `1` permits saving a non-loopback bind without a password |
| OUROBOROS_MODEL | google/gemini-3.8-flash | Main model |
| OUROBOROS_MODEL_HEAVY | "" | Outside `ACTIVE_MODEL_SETTING_KEYS`: read for migration, never routed |
| OUROBOROS_MODEL_LIGHT | openai/gpt-6-luna | Light model |
| OUROBOROS_MODEL_ACCOUNTS | {} | Role-owned managed-account pins; empty = Auto |
| OUROBOROS_PROCESSING_PREFERENCE | "" | Global provider-neutral preference standard/fast/economy; empty keeps the adapter default |
| OUROBOROS_MODEL_PROCESSING_PREFERENCES | {} | Role-owned preferences; empty roles inherit the global one |
| OUROBOROS_MODEL_CONTEXT_WINDOWS | {} | Role-owned context sizing assertions; zero = Auto; grants no reviewer authority |
| OUROBOROS_MODEL_VISION | "" | Vision model (empty inherits) |
| OUROBOROS_IMAGE_INPUT_MODE | auto | Send-time image routing (`vision_routing.py`) |
| OUROBOROS_VISION_CAPTION_TIMEOUT_SEC | 90 | Caption-generation ceiling |
| OUROBOROS_MODEL_CONSCIOUSNESS | "" | Background-consciousness model (empty inherits) |
| OUROBOROS_MODEL_FALLBACKS | openai/gpt-5.6-luna | Cross-model fallback chain (`fallback_cooldown.py`); blank = none |
| OUROBOROS_SERVED_MODEL_REDOS | 2 | Redos of a round another model answered, each a new operation (clamped 0-5) |
| OUROBOROS_MODEL_MAX_CONCURRENCY | 3 | Per-(model, route) concurrent provider-call cap (`model_concurrency.py`) |
| OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC | 180 | Concurrency-slot wait bound |
| OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC | 60 | Project-naming call ceiling |
| OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC | 8 | Inline naming bound when a card becomes a project (`gateway/projects.py`) |
| OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC | 120 | Update-letter one-shot ceiling, slot wait and call together (`update_letter.py`) |
| OUROBOROS_UI_TRANSLATION_TIMEOUT_SEC | 120 | One translation batch, slot wait and call together (`ui_translation.py`) |
| OUROBOROS_UI_LANGUAGE | "" | Owner interface language (BCP-47 tag; `""` = not chosen); written only by `POST /api/ui/i18n/language` (`ENDPOINT_WRITTEN_SETTINGS`), read live (§3 Settings and onboarding) |
| OUROBOROS_FALLBACK_COOLDOWN_ENABLED | true | 429-aware per-process model cooldown |
| OUROBOROS_FALLBACK_COOLDOWN_SEC | 120 | Cooldown window |
| OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL | 1 | Attempts per model in the fallback walk |
| OUROBOROS_MODEL_DEEP_SELF_REVIEW | (migrated) | Retired review-lane key: stored value loads as the deep-review helper row of `OUROBOROS_SUBAGENTS` (no Reviewer mark); env inert (`/review` reviewer: §6 Deep self-review) |
| OUROBOROS_MAX_WORKERS | 10 | Active worker dispatch capacity |
| OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT | 6 | Live-subagent cap per root (hard caps in `runtime_limits.py`) |
| OUROBOROS_MAX_SUBAGENT_DEPTH | 3 | Subagent tree depth |
| OUROBOROS_DISABLE_MANAGED_UPDATES | unset | Env-only: `1` disables managed updates (`git_ops.py`) |
| OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS | "" | Mutative-subagent Auto override |
| OUROBOROS_SUBAGENT_WORKTREE_ROOT | "" | Acting-worktree root (empty derives `~/Ouroboros/subagent_worktrees`) |
| OUROBOROS_SUBAGENT_PROJECTS_ROOT | "" | Projects root (empty derives `~/Ouroboros/projects`) |
| OUROBOROS_SUBAGENTS | "" | Configured-subagent roster (`configured_subagents.py`; §6 Delegated subagents) |
| OUROBOROS_SUBAGENT_HARNESS | "" | Narrow harness input, read for migration |
| OUROBOROS_SUBAGENT_PROFILE | "" | Narrow profile input, read for migration |
| OUROBOROS_DELEGATE_WAIT_SEC | 120 | Default `delegate_wait` window |
| OUROBOROS_DELEGATE_WAIT_MAX_SEC | 1800 | `delegate_wait` ceiling |
| OUROBOROS_DELIVERABLES_ROOT | "" | Deliverables root (empty derives `~/Ouroboros/Deliverables`; `tool_access.py`) |
| OUROBOROS_GC_RETENTION_DAYS | 7 | Unified GC retention (`retention.py`) |
| OUROBOROS_RESTART_DRAIN_MAX_SEC | 120 | Restart drain bound |
| TOTAL_BUDGET | 200.0 | Global USD budget, reached by known spend; non-positive = no finite limit; resolved live at every model call (`resolve_total_budget_usd`) |
| OUROBOROS_PER_TASK_COST_USD | 50.0 | Hard cap over one task's whole tree: new paid calls are refused once its known spend (confirmed + estimated; holds are not spending) reaches it; no default early stop precedes it, only an explicit `cost_hard_stop_pct` profile does (`task_pacing.py`; §6 Budget tracking) |
| OUROBOROS_RUB_USD_RATE | "" | Manual RUB→USD rate for RUB-priced providers |
| OUROBOROS_PRICING_TTL_SEC | 21600 | Provider-catalog pricing cache TTL |
| OUROBOROS_TOOL_TIMEOUT_SEC | 600 | Default tool timeout |
| OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC | 1800 | Per-call timeout clamp |
| OUROBOROS_FINALIZATION_GRACE_SEC | 120 | Finalization grace window |
| OUROBOROS_WEBSEARCH_MODEL | gpt-5.2 | `web_search` backing model |
| OUROBOROS_WEBSEARCH_BACKEND | auto | `web_search` backend selection |
| OUROBOROS_MAIN_WEB_SEARCH | off | Main-loop inline web search (§6 Web access mechanisms) |
| OUROBOROS_MAIN_WEB_SEARCH_ENGINE | auto | Inline-search engine |
| OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS | 10 | Inline-search result cap |
| OUROBOROS_OR_PROVIDER | "" | OpenRouter provider-routing preference merged into requests |
| OUROBOROS_SEARCH_CODE_WALL_SEC | 45 | `search_code` wall-clock budget |
| OUROBOROS_PRESENTATION | unset | Env-only: launcher-exported presentation (`desktop_window`/`browser_fallback`/`android_app`; absent = `web`) |
| OUROBOROS_DESKTOP_BACKGROUND | unset | Env-only: `1` when the desktop launcher can keep running hidden (Windows, macOS) |
| OUROBOROS_EXTERNAL_HOST_UPDATE | unset | Env-only: external-host installer command supporting read-only `--check` |
| OUROBOROS_EXTERNAL_HOST_RESULT | unset | Env-only: launcher-verified installed-artifact facts for one core generation, not a persisted PASS |
| OUROBOROS_USER_FILES_ROOT | "" | Env-only: `user_files` jail root (empty = `$HOME`) |
| OUROBOROS_OBSERVABILITY_KEEP_RAW | unset | Env-only: truthy persists raw observability payloads |
| OUROBOROS_GENERATIVE_PROBE | 1 | Generative-write probe toggle |
| OUROBOROS_GENERATIVE_PROBE_CHARS | 5000000 | Generative-probe size |
| OUROBOROS_REVIEWER_SLOTS | (migrated) | Retired review-lane key: stored values become `OUROBOROS_SUBAGENTS` reviewer rows on one read; unreadable values stay for the owner; never read from the environment. Rollback: restore keys from `before` in `state/review_migrations/<ts>-slots-to-pool.json`, named by the owner message (Review pool) |
| OUROBOROS_SUBSCRIPTION_PRESET_VERSION | "" | One-shot install-preset marker; endpoint-authored, disk-only (`ENDPOINT_AUTHORED_SETTINGS`); absence authorizes nothing (§2) |
| OUROBOROS_SUBAGENT_PRESET_RECEIPT | "" | Install-preset receipt; endpoint-authored, disk-only |
| OUROBOROS_ONBOARDING_COMPLETED_AT | "" | Onboarding completion fact; endpoint-authored, disk-only |
| OUROBOROS_TASK_REVIEW_MODE | auto | Task acceptance-review mode (§6 Task acceptance) |
| OUROBOROS_SAFETY_MODE | full | Safety supervision mode; a fresh desktop wizard may author `light`; lowering is owner-guarded (§6 Safety and runtime mode) |
| OUROBOROS_SAFETY_MAX_TOKENS | 2000 | Safety-check output budget |
| OUROBOROS_SAFETY_CALL_TIMEOUT_SEC | 60 | Safety-check call ceiling |
| OUROBOROS_WEBSEARCH_TIMEOUT_SEC | 480 | `web_search` ceiling |
| OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC | 2 | Custody retry wait for a stopped direct turn's round boundary (clamped 0-10) |
| OUROBOROS_ONBOARDING_SNAPSHOT_TIMEOUT_SEC | 45 | Onboarding Claudexor snapshot-read bound; a wedged daemon refuses without hanging |
| OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC | 30 | In-process owner-write document-lock bound; contention refuses before the transform runs (`owner_settings.py`) |
| OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC | 2700 | LLM transport read timeout |
| OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC | 300 | `plan_task` deadline floor |
| OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC | 200 | Spendable seconds above the finalization reserve needed to start a critic panel, else `review_skipped_deadline_reserve` (§6 Task lifecycle) |
| OUROBOROS_REVIEW_MAX_CYCLES | "2" | Paid review-cycle cap shared by acceptance/commit/skill gates (excludes plan asks); `unlimited` = no local count cap (`review_cycles.py`; §6 Review stack) |
| OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES | (none) | Alias only: stored values migrate into `OUROBOROS_REVIEW_MAX_CYCLES` (passes + 1) at load; environment values are inert |
| OUROBOROS_ACCEPTANCE_RESERVE_PCT | 5 | Acceptance budget reserve percentage |
| OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC | unset | Env-only: logical review timeout (absent = route-owned; late results stay in custody) |
| OUROBOROS_REVIEW_MAX_TOKENS | 65536 | Env-only: reviewer output budget, clamped to the 8192 floor |
| OUROBOROS_REVIEW_ENFORCEMENT | advisory | Review enforcement advisory/blocking; anything else coerces to the default (§6 Review stack) |
| OUROBOROS_PREFLIGHT_TIMEOUT_SEC | 1800 | Env-only: total wall-clock budget of the hermetic test preflight (`preflight_runner.py`); the Android entry defaults to 3600 |
| OUROBOROS_PREFLIGHT_SERIAL | unset | Env-only: `1` selects one serial pytest pass; scrubbed from the candidate environment |
| OUROBOROS_PREFLIGHT_TEST_WORKERS | unset | Env-only: xdist workers for the parallel pass (floor 2, else `os.cpu_count()`); scrubbed from the candidate |
| OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS | true | Auto-grant manifest-declared permissions to cleanly reviewed skills (hash-bound; blocking findings never grant) |
| OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS | true | Launcher seed/resync writes hash-pinned `native_seed` verdicts; no runtime grant endpoint |
| OUROBOROS_CONTEXT_MODE | max | Main's `nano`/`low`/`max` and initial project copy; owner-selected outside Cyber. Review remains independent (§6 Context fitting) |
| OUROBOROS_CONTEXT_MODE_AUTO_LOW | false | Provenance tombstone, not a toggle: normalization writes `false`; a persisted `low` is honoured only beside it (`context_mode_compat.py`) |
| OUROBOROS_RUNTIME_MODE | advanced | Effective Access light/advanced/pro/cyber_pro, persisted as the next-boot value (§6 Safety and runtime mode) |
| OUROBOROS_SKILLS_REPO_PATH | "" | Extra skills checkout path (expanded at read time, never cloned or pulled) |
| MCP_ENABLED | false | MCP client toggle (§6 MCP and browser-facing external tools) |
| MCP_SERVERS | [] | MCP server list; persisted in settings, never env-exported |
| MCP_TOOL_TIMEOUT_SEC | 60 | Per-MCP-tool timeout |
| OUROBOROS_HUB_CATALOG_URL | `https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json` | OuroborosHub catalog URL (automatic fetch limited to catalog JSON; installs verify SHA-256) |
| OUROBOROS_CLAWHUB_REGISTRY_URL | `https://clawhub.ai/api/v1` | ClawHub registry URL |
| OUROBOROS_PROMPT_CACHE_TTL | 1h | Prompt-cache tier default/5m/1h for Anthropic-family cache markers, applied at the send boundary and recorded in usage |
| OUROBOROS_EFFORT_MIN | low | Main helper floor; project composers save their own copy (`settings_scales.effort_range`) |
| OUROBOROS_EFFORT_TASK | medium | Main primary; project copy sets its roots' primary; Auto helpers clamp it into their bounds |
| OUROBOROS_EFFORT_MAX | high | Main helper ceiling; Auto reviewers, evolution and wakes start here; project copies are independent |
| OUROBOROS_EFFORT_REVIEW | (migrated) | Retired surface default: fills missing triad-seat reviewer effort at load; afterwards row `effort` is the only effort; env inert |
| OUROBOROS_EFFORT_SCOPE_REVIEW | (migrated) | Retired surface default: consumed into the scope seats' reviewer rows at load; env inert |
| OUROBOROS_EFFORT_DEEP_SELF_REVIEW | (migrated) | Retired surface default: consumed into the deep-review helper row at load; env inert |
| OUROBOROS_RETURN_REASONING | true | Ask OpenRouter to return reasoning; direct/local request copies strip OpenRouter-only fields |
| OUROBOROS_REASONING_SUMMARY | auto | Readable reasoning-summary rendering; presentation-only, never added to history or returned to providers |
| OUROBOROS_TASK_IDLE_TIMEOUT_SEC | 900 | Idle timeout; requires absence of real task/subtree progress (§6 Task lifecycle) |
| OUROBOROS_TASK_ABS_CEILING_SEC | unlimited | Optional absolute task lifetime in seconds (floor 300), activity-independent; deadline and budget stay separate axes |
| OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC | 90 | Liveness watchdog; alerts and recommends `/restart`, never frees a lock held by a wedged turn |
| LOCAL_MODEL_SOURCE | "" | Local-model source (HF repo or path) |
| LOCAL_MODEL_FILENAME | "" | Local-model GGUF filename (split first shard expanded) |
| LOCAL_MODEL_CONTEXT_LENGTH | 16384 | Local-model context window |
| LOCAL_MODEL_N_GPU_LAYERS | 0 | GPU offload layers |
| USE_LOCAL_MAIN | false | Route Main locally |
| USE_LOCAL_HEAVY | false | Migration input only; excluded from active routing |
| USE_LOCAL_LIGHT | false | Route Light locally |
| USE_LOCAL_CONSCIOUSNESS | false | Route Consciousness locally |
| USE_LOCAL_FALLBACK | false | Route fallback locally |
| OUROBOROS_MAX_ROUNDS | unlimited | Optional total task round limit (hot-reloadable; a Presence turn keeps its own finite inline cap) |
| OUROBOROS_TRANSIENT_RETRY_MAX | 6 | Same-model transient retry budget; pre-dispatch `transport_unavailable` is a separate task-bounded outer wait |
| OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC | 1800 | Skill lifecycle-lane timeout |
| OUROBOROS_CLAUDEXOR_HARNESS_INSTALL_TIMEOUT_SEC | 300 | Harness install ceiling (kills the tracked group, typed refusal) |
| OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC | 90 | Quota-refresh POST ceiling (clamped 1-90) |
| OUROBOROS_BUNDLE_DIR | unset | Env-only: launcher-owned bundle root propagated to embedded children for Node/ripgrep discovery |
| OUROBOROS_BG_WAKEUP_MIN | 900 | Lower bound (s) of the model-chosen wake interval (`set_next_wakeup`), re-read at each alarm |
| OUROBOROS_BG_WAKEUP_MAX | 14400 | Upper bound (s); with no chosen interval the alarm uses `runtime_limits.WAKE_DEFAULT_SEC`, below the default cache TTL to keep the shared prefix warm |
| OUROBOROS_CONSCIOUSNESS_AUTONOMY | act | Closed enum `observe`/`act`/`full` of what a wake may do; `act` excludes own code/prompts, evolution, restart and settings, `full` adds evolution (`consciousness_authority.py`) |
| OUROBOROS_CONSCIOUSNESS_DAILY_USD | 20.0 | Rolling-24h ceiling over wakes and the tasks they start, on known spend; exhausted blocks new wakes; `0` is a real choice |
| OUROBOROS_CONSCIOUSNESS_MAX_TASKS | 2 | Concurrent consciousness-started tasks; `0` = it never starts tasks |
| OUROBOROS_POST_TASK_EVOLUTION | false | Post-task evolution promotion toggle; self-enablement is blocked at every write guard (§6 Background consciousness and Evolution) |
| OUROBOROS_POST_TASK_EVOLUTION_CADENCE | llm | Promotion cadence `llm` or `every_n:k` (malformed normalizes to `llm`) |
| OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD | 0.0 | Remaining-global-budget start floor, not a cycle cap |
| OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE | "" | Owner-only persistent campaign bias; still passes review gates |
| LOCAL_MODEL_PORT | 8766 | Local-model server port |
| OUROBOROS_HOST_SERVICE_PORT | 8767 | Host Service port (loopback-only; §12) |
| OUROBOROS_DESKTOP_KEEP_RUNNING | false | Closing the desktop window keeps Ouroboros running (Windows, macOS; `launcher_background.py`); disk-authored consent, absent until the owner chooses |
| OUROBOROS_PRESENCE_MAX_ACTIVE | 2 | Cross-process Presence turn cap (UI-bounded 1-20) |
| OUROBOROS_CALL_MODEL_GOOGLE | "gemini-3.8-live" | Gemini Live model for calls |
| OUROBOROS_CALL_VOICE_GOOGLE | "" | Gemini call voice (blank = provider default) |
| OUROBOROS_CALL_NATIVE_SEARCH | true | Gemini's own Google Search in calls |
| OUROBOROS_CALL_MAX_MINUTES | "" | Optional call duration ceiling (blank = none) |
| OUROBOROS_PRESENCE_CALL_MAX_MINUTES | "" | Optional ceiling for a Presence call, which holds a Presence slot (blank = none) |
| LOCAL_MODEL_CHAT_FORMAT | "" | Local-model chat template override |
| GITHUB_TOKEN | "" | GitHub token (push/PR/issues) |
| GITHUB_REPO | "" | Personal `origin` repository |
| OUROBOROS_FILE_BROWSER_DEFAULT | "" | File Browser default root (explicit root required for Docker/non-localhost) |

#### Review pool

The review pool is the enabled `OUROBOROS_SUBAGENTS` rows marked `review_eligible: true` (Settings → Agents, Reviewer mark; `reviewer_slot_config.review_pool_slots`, read in task settings scope): one helper/reviewer list, no separate lane. Outside Cyber Pro every pool row sits on the commit panel. Effort and delivery are row properties: `effort` is the owner's pin, empty = Auto (`reviewer_slot_config.row_effort`: model name, else pin, else a wave's order clamped into the range, else its top; read once per pool build); an API row's `delivery` is `native` (the default: the two-part brief with inspection tools) or `packet`, for models proven to lack tool support (quota, timeout, unknown capability or an automatic fallback never imply that); a session row always retrieves. Only session and managed-model API routes may pin `route.credential_profile_id` (empty = rotation). Disabled marked rows silently leave the pool; calls (`review_change(reviewers=[…])`, `preflight_reviewer`, `/review`) naming disabled or unknown rows refuse (`catalog_review_row`), never reroute. Without a reading row, no seat gets the coupling question: under Blocking outside Cyber Pro every body commit ends `NOT_PERFORMED` until a reading row is marked or review is Advisory; Save only warns (`subagent_runtime.review_pool_save_warning`).

Save, wizard and migration share one judge (`reviewer_slot_config.review_pool_save_error`): a malformed catalog refuses Save and every review surface; a catalog with rows but no mark saves only with the owner's explicit `allow_empty_review_pool`; an empty pool reviews on Main alone. Never-configured installs (no catalog or lanes key: Docker/Colab without the wizard) mint a factory catalog over each environment-merged read until saved, following changed factory defaults. Reviewers (`subscription_install_presets.factory_review_rows`) follow the reachable Main: OpenRouter Main (or no access yet) gets the three `OPENROUTER_REVIEW_DEFAULTS`, a MiniMax Main its Main, Light, Light (Direct-provider routes), any other Main one row in its whole route (label, pin, processing; quorum 1 of 1); a Main the access cannot run mints nothing (lane keys wait), naming `OUROBOROS_MODEL`. The wizard's Auto helpers (`factory_actor_rows`) join only this cell. An operator's environment catalog (never our echo) wins over minted rows (`review_pool_migration.environment_overridable_keys`); saved catalogs or rows minted from the owner's lanes shadow it like other disk-authored keys. Changing delivery changes review-contract identity (later reviews may be paid anew) but launches nothing.

#### Direct-provider routes

For an official direct-provider Main, factory review rows are its declarative reviewer roles, each in its binding (`provider_models.compute_direct_review_seats`). Scope: official OpenAI, Anthropic, MiniMax, DeepSeek, Z.ai, Cloud.ru and GigaChat; the exclusive-provider detector (`server_runtime._exclusive_direct_remote_provider`; `_exclusive_direct_remote_provider_env` for the environment) is empty with OpenRouter, `OPENAI_BASE_URL`, `OPENAI_COMPATIBLE_BASE_URL` or several direct providers. Rows are provider-prefixed (`provider_models.migrate_model_value`; a Main off the provider mints none), so free text never enters a single-provider route; one official provider must suffice (DEVELOPMENT "Provider Independence").

DeepSeek (`deepseek::`) uses the fixed `provider_models.DEEPSEEK_BASE_URL`; proxies and mirrors belong to `openai-compatible::`, and `deepseek/...` stays OpenRouter. `DEEPSEEK_REASONING_EFFORT_ALIASES` maps send-time effort, every tier change reporting `reasoning_effort_clamped`; a forced tool choice disables thinking. Wire details: `llm_openai_compatible.py`.

MiniMax (`minimax::`) sends `reasoning_split=true`; raw reasoning details survive same-route turns verbatim and render display-only. The live wire is unverified.

Z.ai (`zai::`) is GLM through Z.ai's OpenAI-compatible API. `ZAI_PLAN` selects the endpoint (`provider_models.resolve_zai_base_url`); a proxy or the China host belongs to `openai-compatible::`, and `zai/...` stays OpenRouter. Z.ai serves an absent `reasoning_effort` at its maximum, so the canonical scale is projected onto its own enum (`ZAI_REASONING_EFFORT_ALIASES`), every tier change disclosed as `reasoning_effort_clamped`; a forced tool choice keeps its tier. HTTP 429 code 1113 is a billing refusal (also a Coding Plan key on the pay-as-you-go endpoint), so the provider Test reports `No credits`, not `Rate limited`.

GigaChat (`gigachat::`) runs on the native `gigachat` library, not the OpenAI-compatible lane: one function call per turn (message vocabulary: `llm_gigachat.py`). `reasoning_effort` is deliberately omitted, because hidden reasoning can consume the whole output budget and return empty content. No live cost source exists, so cost stays unknown, never a hand-maintained tariff. A GigaChat scope row retrieves on its own window (§6 Review stack).

---
