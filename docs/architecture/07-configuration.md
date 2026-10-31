# 7. Configuration (ouroboros/config.py)

Settings have one home and default; the table projects `config.SETTINGS_DEFAULTS`, never a second authority.

`ouroboros/config.py` re-exports paths, loaders/writers, environment projection, normalization, runtime mode, skills location and PID lock. Vocabularies/defaults belong to its leaves (§10 invariant 3). `settings_scales.IMMEDIATE_SETTINGS` applies live; `RESTART_REQUIRED_SETTINGS` awaits restart; other keys apply from the next task, a route key from the next model call. `update_channels.py` owns update defaults.

Each path constant takes its `OUROBOROS_*` override, else derives from `APP_ROOT` (`~/Ouroboros`) in `launch_roots` before imports; a direct server's implicit `REPO_DIR` is its executing checkout, and given only `APP_ROOT` it refuses while `~/Ouroboros/data` may hold memory; `resolve_port_file` relocates only the hosting root's port file. Unset work roots follow the witness basis: `~/Ouroboros` for older data, else `APP_ROOT`; unknown refuses. The document is `settings.json` under the data root. Writes need the settings file lock. Secret placeholders are repaired only when recognized on disk, before environment precedence resolves, so a real environment credential never reads as a mask (`secret_masking.py`). While `OUROBOROS_SETTINGS_SHA256` is set (`settings_integrity.py`) the seeded snapshot is an owner-authored trust root for an isolated child: every read verifies the whole byte stream without creating a sibling lock file, and every writer refuses.

`openrouter_attribution.py` is the app-identity SSOT of every first-party paid OpenRouter request (canonical URL + `X-OpenRouter-Title`); a fork uses its own URL, never renaming one app record.

### Reading and writing the settings document

Tasks privately keep their entry Context/Effort, review settings and catalog (`TaskSettingsSnapshot`); route keys (`route_choice.ROUTE_SETTING_KEYS`: models, local flags, accounts, provider keys) read the saved document as it is now (none saved: as every other key). Absent/empty differ, route overrides win, Resume restores cognition, and overlays never enter env. `RoomResolver` follows lineage/conversion. `composer_cognition.py` copies Main once on project creation/recovery and seeds legacy cognition before Main changes. Subagents stays live: Main's `OUROBOROS_SUBAGENTS_MODE` or project `cognition.subagents_mode`, `off|auto|swarm`, default Auto, including legacy rooms. It bypasses checkpoints; Settings only masks it. Main effort is independent of helper bounds (`_effort_range_format=1`). Immediate keys, writers and granted skill keys read current disk, including clears; credential-free skill children reread it, payloads carry operator variables. Access stays boot-pinned; reload failures disclose prior env and missing values.

`normalize_settings_raw()` runs before defaults in `load_settings`, `_owner_read_settings_raw` and `build_colab_settings`; dropped keys stay dropped. `safety_config.normalize_consciousness_safety` seeds absent coverage from saved owner coverage; explicit choices stay and guards compare the disk-derived value. `review_pool_migration.apply_at_read_seam` maps run seats to roster rows, retaining refused values. Retired keys/comma-lists drop with notice; comma-only/lane-free installs get factory rows. Review env keys are inert (one startup notice). Reads are pure/idempotent; `read_settings_json_verified` refuses changed pins. The loader proposes unauthored slots (§2); `apply_runtime_provider_defaults` only respells them, unpersisted. `apply_settings_to_env` marks slot/catalog echoes in internal `OUROBOROS_PROJECTED_PROPOSALS`, inherited by children, distinct from operator variables. Provider Test and credential plans share `provider_models.ACTIVE_MODEL_SETTING_KEYS`, including Safety pins; blank inherits.

Optional bounds: `OUROBOROS_MAX_ROUNDS`/`OUROBOROS_TASK_ABS_CEILING_SEC` use `parse_positive_or_unlimited` (also review cycles) and ship `unlimited`. Money, deadlines and Stop/Panic still bind; silence triggers attention (§5). Legacy missing keys retain 200 rounds/21600 s (`OPTIONAL_BOUND_LEGACY`); blank, zero, negative or malformed values also take those finite defaults, logged and announced once by `upgrade_notices.py`. Readers return `None` for unlimited. `operation_window_sec()` inherits task lifetime, never a transport bound or an unbounded operation.

Five writers use `serialize_settings()` and exact bytes (no CRLF conversion). `config.save_settings`, `owner_settings._owner_update_settings` and `packaged_cli._save_settings` use `prepare_settings_for_persist` for disk-authored silence and context/safety ratchets. Exempt: `context_mode_compat.normalize_and_persist_context_mode_compat` updates only the raw compat pair; `colab_bootstrap.write_colab_settings` generates a foreign-root document. All persist review migration receipts before replacing the pre-image (`review_pool_receipts.persist_write_receipts`). `tests/_shared.py::settings_writers` rejects an unlisted sixth writer.

An owner endpoint changes one decision in a document it does not own, so `_owner_update_settings(transform, expected_digest)` does the whole read-change-write under one settings lock: a no-change decision never rewrites the file, and a digest mismatch (`settings_document_digest()`) refuses before the transform runs, so a concurrent owner change is never reverted key by key while the request answers "saved". Incidental writers (launcher, `set_tool_timeout`) land one field (`fields_on_saved_document`).

### LLM output token budgets

OpenRouter/Anthropic-compatible output budgets use `max_tokens`; direct OpenAI Chat uses `max_completion_tokens`. Only successful exact-route wire evidence adapts requests, never model-name prefixes. Budgets are floors, lowered by a known route maximum; constants own the numbers:

| Surface | Output-token budget |
|---------|---------------------|
| `LLMClient.chat()` / `chat_async()` defaults | 65,536 |
| Main task loop (`loop_llm_call.MAIN_LOOP_MAX_TOKENS`) | 65,536 |
| `LLMClient.vision_query()` and VLM tools (`analyze_screenshot`, `vlm_query`) | 32,768 |
| Review synthesis dedup | 16,384 |
| Light memory drafts (`consolidator.LIGHT_ANSWER_CEILING_TOKENS`) | 16,384 |
| Post-task evolution promotion decision (`post_task_evolution`) | 8,192 |
| Skill publish PR body generation | 8,192 |
| Project naming LIGHT one-shot (`project_naming.llm_project_name`) | 256 |
| Provider Test (`llm_probe.PROVIDER_TEST_MAX_TOKENS`) | 16 |

Naming/translation share one deadline across preparation, slots and recovery; naming captures it before scheduling so workers cannot renew it. Earlier bounds win; detached naming refreshes cost without publishing a title.

### Default settings

One row per key (`settings_defaults.py`). Env-only rows have no settings.json carrier; alias and `(migrated)` rows name retired keys converted at read.

| Key | Default | Description |
|-----|---------|-------------|
| OPENROUTER_API_KEY | "" | OpenRouter credential |
| OPENAI_API_KEY | "" | Official direct-OpenAI credential |
| OPENAI_BASE_URL | "" | Legacy compatible endpoint (only with `OPENAI_API_KEY`) |
| OPENAI_COMPATIBLE_API_KEY | "" | OpenAI-compatible endpoint credential |
| OPENAI_COMPATIBLE_BASE_URL | "" | OpenAI-compatible endpoint; only with its own key (blank = no auth) |
| CLOUDRU_FOUNDATION_MODELS_API_KEY | "" | Cloud.ru credential |
| CLOUDRU_FOUNDATION_MODELS_BASE_URL | `https://foundation-models.api.cloud.ru/v1` | Cloud.ru base URL |
| GIGACHAT_CREDENTIALS | "" | GigaChat auth key |
| GIGACHAT_USER | "" | GigaChat user login |
| GIGACHAT_PASSWORD | "" | GigaChat password |
| GIGACHAT_SCOPE | `GIGACHAT_API_PERS` | GigaChat API scope |
| GIGACHAT_BASE_URL | `https://api.giga.chat/v1` | GigaChat base URL |
| GIGACHAT_VERIFY_SSL_CERTS | `true` | GigaChat TLS verification |
| GIGACHAT_PROFANITY_CHECK | "" | GigaChat profanity filter passthrough |
| OUROBOROS_EXTRA_CA_BUNDLE | "" | PEM file added to every first-party provider call's trust bundle |
| ANTHROPIC_API_KEY | "" | Official direct-Anthropic credential |
| MINIMAX_API_KEY | "" | MiniMax credential |
| MINIMAX_REGION | "" | MiniMax region (empty = `global_en`) |
| DEEPSEEK_API_KEY | "" | DeepSeek credential (`deepseek::`) |
| ZAI_API_KEY | "" | Z.ai credential (`zai::`) |
| ZAI_PLAN | "" | Z.ai endpoint: empty/`payg` = pay-as-you-go, `coding` = Coding Plan |
| OUROBOROS_CONNECTIONS | [] | Extra keys/endpoints, a row each (Direct-provider routes), never env-exported |
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
| OUROBOROS_MODEL_SAFETY | (empty) | Safety Supervisor; empty uses Light, then Main. Outside Cyber Pro only owner writes change the resolved checker (the owner Safety endpoints, the model-wait card); a provider change that re-proposes an unauthored slot saves and is audited |
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
| OUROBOROS_MODEL_MAX_CONCURRENCY | 3 | Concurrent physical calls of one model on one provider route (credential, endpoint), shared by every process of the installation with a fair queue, a turn the owner awaits first; routes the managed Claudexor engine serves are left to the engine (`model_concurrency.py`) |
| OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC | 180 | Concurrency-slot wait bound |
| OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC | 60 | Project-naming operation ceiling |
| OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC | 8 | Inline naming bound when a card becomes a project (`gateway/projects.py`) |
| OUROBOROS_UI_TRANSLATION_TIMEOUT_SEC | 120 | One translation batch, slot wait and call together (`ui_translation.py`) |
| OUROBOROS_UI_LANGUAGE | "" | Owner interface language (BCP-47 tag; `""` = not chosen); written only by `POST /api/ui/i18n/language` (`ENDPOINT_WRITTEN_SETTINGS`), read live (§3 Settings) |
| OUROBOROS_FALLBACK_COOLDOWN_ENABLED | true | 429-aware per-process model cooldown |
| OUROBOROS_FALLBACK_COOLDOWN_SEC | 120 | Cooldown window |
| OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL | 1 | Attempts per model in the fallback walk |
| OUROBOROS_MODEL_DEEP_SELF_REVIEW | (migrated) | Retired with the review lanes: a stored value becomes the deep-review helper row of `OUROBOROS_SUBAGENTS` (no Reviewer mark) at load; env inert (`/review` reviewer: §6 Deep self-review) |
| OUROBOROS_MAX_WORKERS | 10 | Active worker dispatch capacity |
| OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT | 6 | Live-subagent cap per root (hard caps in `runtime_limits.py`) |
| OUROBOROS_MAX_SUBAGENT_DEPTH | 3 | Subagent tree depth |
| OUROBOROS_DISABLE_MANAGED_UPDATES | unset | Env-only: `1` disables managed updates (`git_ops.py`) |
| OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS | "" | Mutative-subagent Auto override |
| OUROBOROS_SUBAGENT_WORKTREE_ROOT | "" | Acting-worktree root (empty: basis default) |
| OUROBOROS_SUBAGENT_PROJECTS_ROOT | "" | Projects root (empty: basis default) |
| OUROBOROS_SUBAGENTS_MODE | auto | Live room delegation choice: `off`, `auto`, `swarm`; project cognition is independent (§6) |
| OUROBOROS_SUBAGENTS | "" | Configured-subagent roster (`configured_subagents.py`; §6 Delegated subagents) |
| OUROBOROS_SUBAGENT_HARNESS | "" | Narrow harness input, read for migration |
| OUROBOROS_SUBAGENT_PROFILE | "" | Narrow profile input, read for migration |
| OUROBOROS_DELEGATE_WAIT_SEC | 120 | Default `delegate_wait` window |
| OUROBOROS_DELEGATE_WAIT_MAX_SEC | 1800 | `delegate_wait` ceiling |
| OUROBOROS_DELIVERABLES_ROOT | "" | Deliverables root (empty: basis default; `tool_access.py`) |
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
| OUROBOROS_REVIEWER_SLOTS | (migrated) | Retired review-lane key: a stored value is read once into reviewer rows of `OUROBOROS_SUBAGENTS`; an unreadable value stays for the owner; never read from the environment. Rollback: restore the keys from the `before` of `state/review_migrations/<ts>-slots-to-pool.json`, which the owner message names (Review pool) |
| OUROBOROS_SUBSCRIPTION_PRESET_VERSION | "" | One-shot install-preset marker; endpoint-authored, disk-only (`ENDPOINT_AUTHORED_SETTINGS`); absence authorizes nothing (§2) |
| OUROBOROS_SUBAGENT_PRESET_RECEIPT | "" | Install-preset receipt; likewise |
| OUROBOROS_ONBOARDING_COMPLETED_AT | "" | Onboarding completion fact; likewise |
| OUROBOROS_ONBOARDING_ATTEMPT_ID | "" | Client attempt that wrote it (§2); likewise |
| OUROBOROS_TASK_REVIEW_MODE | auto | Result review mode: `auto` forces nothing, `required` runs the host's result review at a root delivery with produced material, `off` refuses `review_change(subject="result")` (§6 Result review) |
| OUROBOROS_SAFETY_MODE | full | Safety supervision mode; a fresh desktop wizard may author `light`; lowering is owner-guarded (§6 Safety and runtime mode) |
| OUROBOROS_CONSCIOUSNESS_SAFETY_MODE | full | Coverage for consciousness work (own wakes, places, what they start); existing installs copy the saved Safety level once |
| OUROBOROS_SAFETY_OWNER_RULES | (empty) | The owner's standing rules for every mind and check; owner-only |
| OUROBOROS_SAFETY_MAX_TOKENS | 2000 | Safety-check output budget |
| OUROBOROS_SAFETY_CALL_TIMEOUT_SEC | 60 | Safety-check call ceiling |
| OUROBOROS_WEBSEARCH_TIMEOUT_SEC | 480 | `web_search` ceiling |
| OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC | 2 | Wait for a stopped direct turn to reach its round boundary before custody retries (clamped 0-10) |
| OUROBOROS_ONBOARDING_SNAPSHOT_TIMEOUT_SEC | 45 | Bound on the onboarding Claudexor snapshot read; a wedged daemon refuses rather than hangs |
| OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC | 30 | In-process document-lock bound for owner writes; lock contention refuses before the transform runs (`owner_settings.py`) |
| OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC | 2700 | LLM transport read timeout |
| OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC | 300 | `plan_task` deadline floor |
| OUROBOROS_REVIEW_MAX_CYCLES | "2" | Shared paid review-cycle cap over the result-review/commit/skill gates (plan asks stand outside it); `unlimited` = no local count cap (`review_cycles.py`; §6 Review stack) |
| OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES | (none) | Alias only: a stored value is MIGRATED into `OUROBOROS_REVIEW_MAX_CYCLES` (passes + 1) at load; an environment value is inert |
| OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC | unset | Env-only: logical review timeout (absent = route-owned; late results stay in custody) |
| OUROBOROS_REVIEW_MAX_TOKENS | 65536 | Env-only: reviewer output budget, clamped to the 8192 floor |
| OUROBOROS_REVIEW_ENFORCEMENT | advisory | Review enforcement advisory/blocking; anything else coerces to the default (§6 Review stack) |
| OUROBOROS_PREFLIGHT_TIMEOUT_SEC | 1800 | Env-only: total wall-clock budget of the hermetic test preflight (`preflight_runner.py`); the Android entry defaults to 3600 |
| OUROBOROS_PREFLIGHT_SERIAL | unset | Env-only: `1` selects one serial pytest pass; scrubbed from the candidate environment |
| OUROBOROS_PREFLIGHT_TEST_WORKERS | unset | Env-only: xdist workers for the parallel pass (floor 2, else `os.cpu_count()`); scrubbed from the candidate |
| OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS | true | Auto-grant manifest requests to accepted bytes (clean review or trusted install; hash-bound) |
| OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS | true | Derived Built in trust (`false`: review); once a source list is saved its Built in row decides |
| OUROBOROS_CONTEXT_MODE | max | Main's `nano`/`low`/`max` and initial project copy; owner-selected outside Cyber. Review stays independent (§6 Context fitting) |
| OUROBOROS_CONTEXT_MODE_AUTO_LOW | false | Provenance tombstone, not a toggle: normalization writes `false`; a persisted `low` is honoured only beside it (`context_mode_compat.py`) |
| OUROBOROS_RUNTIME_MODE | advanced | Effective Access light/advanced/pro/cyber_pro, persisted as the next-boot value (§6 Safety and runtime mode) |
| OUROBOROS_SKILLS_REPO_PATH | "" | Skills Repo Path: read in place as My folder (`my-folder`), even with a saved source list; removing that row clears this path. Other folders stay connected. |
| MCP_ENABLED | false | MCP client toggle (§6 MCP and browser-facing external tools) |
| MCP_SERVERS | [] | Saved, never env-exported; a change re-lists in the server |
| MCP_TOOL_TIMEOUT_SEC | 60 | Per listing and tool call |
| OUROBOROS_HUB_CATALOG_URL | `https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json` | Derives the two hub rows; a non-GitHub raw URL needs `OUROBOROS_SKILL_SOURCES` with the git address |
| OUROBOROS_CLAWHUB_REGISTRY_URL | `https://clawhub.ai/api/v1` | ClawHub registry URL |
| OUROBOROS_SKILL_SOURCES | "" | Skill source list (§13); empty = derived from the keys above, unwritten |
| OUROBOROS_STARTER_SKILLS | "" | Starter catalog ids from any source; absent/"" = recommended, `[]` or "[]" = none; built-ins install before git reads |
| OUROBOROS_STARTER_SKILLS_TURN_ON | true | Turn starter installs on when ready; an owner's Off persists |
| OUROBOROS_PROMPT_CACHE_TTL | 1h | Prompt-cache tier default/5m/1h of each normalized cache marker, stamped at send, recorded in usage |
| OUROBOROS_EFFORT_MIN | low | Main helper floor; project composers save their own copy (`settings_scales.effort_range`) |
| OUROBOROS_EFFORT_TASK | medium | Main primary; project copy sets its roots' primary; Auto helpers clamp it into their bounds |
| OUROBOROS_EFFORT_MAX | high | Main helper ceiling; Auto reviewers, evolution and wakes start here; project copies are independent |
| OUROBOROS_EFFORT_REVIEW | (migrated) | Retired surface default: fills missing triad-seat reviewer effort at load; afterwards row `effort` is the only effort; env inert |
| OUROBOROS_EFFORT_SCOPE_REVIEW | (migrated) | Retired surface default: consumed into the scope seats' reviewer rows at load; env inert |
| OUROBOROS_EFFORT_DEEP_SELF_REVIEW | (migrated) | Retired surface default: consumed into the deep-review helper row at load; env inert |
| OUROBOROS_RETURN_REASONING | true | Ask OpenRouter to return reasoning; direct/local request copies strip OpenRouter-only fields |
| OUROBOROS_REASONING_SUMMARY | auto | Readable reasoning-summary rendering; presentation-only, never added to history or returned to providers |
| OUROBOROS_TASK_IDLE_TIMEOUT_SEC | 900 | Silence-attention threshold: no real own/subtree progress (§5); direct-chat outage/accounting-wait bound (§6) |
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
| OUROBOROS_TRANSIENT_RETRY_MAX | 6 | Same-model transient retry budget |
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
| LOCAL_MODEL_CHAT_FORMAT | "" | Local-model chat template override |
| GITHUB_TOKEN | "" | GitHub token (push/PR/issues) |
| GITHUB_REPO | "" | Personal `origin` repository |
| OUROBOROS_FILE_BROWSER_DEFAULT | "" | File Browser default root (explicit root required for Docker/non-localhost) |

#### Review pool

The review pool is the enabled rows of `OUROBOROS_SUBAGENTS` marked `review_eligible: true` (Settings → Agents, the Reviewer mark; `reviewer_slot_config.review_pool_slots`, in the task's settings scope): one list for helpers and reviewers. Blocking seats every pool row on the commit panel, Advisory the named rows (with a reason); unmarked additions advise, never count. Effort and delivery are row properties: `effort` is the owner's pin, empty = Auto (`reviewer_slot_config.row_effort`: model name, else pin, else a wave's order clamped into the range, else its top); an API row's `delivery` is `native` (the default, the two-part brief with inspection tools) or `packet`, for a model positively known not to support tools (quota, timeouts, unknown capability or fallbacks never imply it); a session row always retrieves. Only sessions and managed-model API routes may pin `route.credential_profile_id` (empty = rotation). A marked row switched off leaves the pool silently; a row a call names (`review_change(reviewers=[…])`, `preflight_reviewer`, `/review`) that is off or unknown is refused (`catalog_review_row`), never rerouted. A pool with no reading row asks no seat the coupling question: under Blocking outside Cyber Pro every body commit ends `NOT_PERFORMED` until a reading row is marked or review is Advisory; Save only warns (`subagent_runtime.review_pool_save_warning`).

The generic Settings save, the owner Safety model endpoint, the wizard and the migration share one judge (`reviewer_slot_config.review_pool_save_error`; both settings endpoints use `review_pool_save_judgement` for changed catalogs): a malformed catalog refuses Save and every review surface; a catalog with rows but no mark saves only with the owner's explicit `allow_empty_review_pool`. A never-configured install (no catalog or lanes key: Docker or Colab without the wizard) mints its factory catalog over the environment-merged read at each read until saved, following a changed factory default. Reviewers (`subscription_install_presets.factory_review_rows`) follow the reachable Main: an OpenRouter Main (or no access yet) gets the three `OPENROUTER_REVIEW_DEFAULTS`, a MiniMax Main its Main, Light, Light (Direct-provider routes), any other Main one row in its whole route (quorum 1 of 1); a Main the access cannot run mints nothing, naming `OUROBOROS_MODEL`. The wizard's Auto helpers (`factory_actor_rows`) join only this cell. An operator's environment catalog (never our echo) wins over minted rows (`review_pool_migration.environment_overridable_keys`); a saved catalog, or rows minted from the owner's lanes, shadow it like any disk-authored key. A changed delivery changes the review-contract identity (a later review may be paid anew) but launches nothing.

#### Direct-provider routes

Routes carry their connection (`parse_model_route`): `kind@name::model` reads that `OUROBOROS_CONNECTIONS` row via the built-in kind's branch (`llm_routing.resolve_target`); a missing or disabled one refuses (`connection_unavailable`), never on another key. Factory review roles (`compute_direct_review_seats`): OpenAI, Anthropic, MiniMax, DeepSeek, Z.ai, Cloud.ru, GigaChat. Exclusivity (`exclusive_direct_kind`, named connections too) needs one provider without OpenRouter, a compatible endpoint or `OPENAI_BASE_URL`. `migrate_model_value` prefixes rows so free text never enters a single-provider route; off-provider Main mints none (DEVELOPMENT "Provider Independence").

DeepSeek (`deepseek::`) uses the fixed `DEEPSEEK_BASE_URL`; proxies and mirrors belong to `openai-compatible::`, `deepseek/...` stays OpenRouter. `DEEPSEEK_REASONING_EFFORT_ALIASES` maps send-time effort, each tier change reporting `reasoning_effort_clamped`; a forced tool choice disables thinking (wire: `llm_openai_compatible.py`). MiniMax (`minimax::`) uses `MINIMAX_DIRECT_DEFAULTS`, scouts/reviewers its slots. Saved choices stay; switch PAYG M2.7 pins explicitly before [18 Nov 2026](https://platform.minimax.io/docs/faq/models.md). `reasoning_split=true` requests reasoning, replayed verbatim on same-route turns; details stay display-only. Live wire unverified.

Z.ai (`zai::`) is GLM on Z.ai's OpenAI-compatible API; `ZAI_PLAN` selects the endpoint (`resolve_zai_base_url`), a proxy or the China host belongs to `openai-compatible::`, `zai/...` stays OpenRouter. Z.ai serves an absent `reasoning_effort` at maximum, so the scale maps onto its enum (`ZAI_REASONING_EFFORT_ALIASES`), each tier change disclosed as `reasoning_effort_clamped`; a forced tool choice keeps its tier. HTTP 429 code 1113 is billing (also a Coding Plan key on the PAYG endpoint): Test reports `No credits`, not `Rate limited`.

GigaChat (`gigachat::`) runs on the native `gigachat` library, not the OpenAI-compatible lane, one function call per turn (`llm_gigachat.py`). `reasoning_effort` is omitted: hidden reasoning can consume the whole output budget and return empty content. With no live cost source, cost stays unknown, never a hand-maintained tariff. A GigaChat scope row retrieves on its own window (§6 Review stack).

---
