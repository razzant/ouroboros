# 7. Configuration (ouroboros/config.py)

This chapter owns the settings surface: where the document lives, which functions may persist it, how an older release's document is translated into today's vocabulary before defaults merge, the per-surface output budgets, and the registry of shipped defaults. One key, one home, one default — and this table is a test-mirrored projection of `config.SETTINGS_DEFAULTS`, never a second authority.

`ouroboros/config.py` is the one IMPORT surface: paths (HOME, APP_ROOT, REPO_DIR, DATA_DIR, SETTINGS_PATH, PID_FILE, PORT_FILE), constants (RESTART_EXIT_CODE 42, AGENT_SERVER_PORT 8765; from `runtime_limits.py` the supervisor loop's events bound SUPERVISOR_EVENT_BATCH_MAX_EVENTS 100 / SUPERVISOR_EVENT_BATCH_MAX_SEC 2.0 and the budget-projection retry interval BUDGET_PROJECTION_RETRY_SEC 30, structural constants rather than settings keys), `load_settings()`/`save_settings()`, `apply_settings_to_env()` (hot-reloadable keys into `os.environ`), `normalize_runtime_mode()` (one clamp for the save path, the read coercion and onboarding validation), `get_runtime_mode()`/`get_skills_repo_path()`, `acquire_pid_lock()`/`release_pid_lock()`. The vocabularies live in leaves it re-exports — `settings_defaults.py`, `settings_scales.py` (whose `IMMEDIATE_SETTINGS` and `RESTART_REQUIRED_SETTINGS` name the keys that bite the running process at once or wait for a restart; a key in neither applies to the next task), `model_slots.py`, `review_model_routes.py`, `runtime_limits.py`, `settings_integrity.py` — so a new key and default belong to a leaf, never the facade (§10 invariant 3). `update_channels.py` owns `get_update_channel()`/`get_update_branch()` and the update-network defaults.

Settings file: `data/settings.json` under the data root (`~/Ouroboros/data/settings.json` by default; `APP_ROOT`, `DATA_DIR` and `SETTINGS_PATH` independently env-overridable), accessed under a file lock. `secret_masking.py` owns the Settings/MCP placeholder emitters and recognizers for known and owner-defined top-level secrets and the token/PEM patterns used by stored diagnostic redaction: `load_settings()` repairs only recognized DISK placeholders before environment precedence resolves, so a real environment credential is never read as a mask, and `prepare_settings_for_persist()` repeats that repair at the writer boundary; nested MCP values are never silently migrated. While `OUROBOROS_SETTINGS_SHA256` is set (`settings_integrity.py`) the seeded snapshot is an owner-authored trust root: every read verifies the whole byte stream and every writer refuses.

`ouroboros/openrouter_attribution.py` is the application-identity SSOT for every first-party paid OpenRouter request (canonical URL + `X-OpenRouter-Title`); a fork must use its own URL rather than competing to rename one app record.

The update letter's material is `base..target` (every commit reachable from the official target but not the running base, merged branches included) plus the README history rows added along the target's first-parent line; one accounted LIGHT-slot call writes it into `state/update_letter.json`, whose single projection the Updates panel and Runtime context share (`update_letter.py`, §3, §6).

### Reading and writing the settings document

`agent.handle_task` binds the admitted task's `settings_integrity.TaskSettingsSnapshot` for the whole task entry: document and environment projection are separate private in-memory views, so absent and empty stay distinct, concurrent tasks keep their own models/keys/Supervisor/Review settings, and explicit task route overrides still win. `SETTINGS_ENV_LOCK` serializes capture and publication only, never execution; `runtime_setting`/`runtime_settings`/`runtime_environ` read that view, writers still read the current document, immediate effects stay live, Access keeps its boot pin, and `model_wait.copy_wait_context` plus task-owned helper threads carry only the existing binding. Out-of-process extensions get only permitted typed next-task values through the existing per-call payload — no whole snapshot, no extra credential environment — and still validate grants and read immediate values live. A failed reload loudly keeps the prior environment, disclosing the document-only values it cannot answer.

A document on disk was written by whatever release the owner last used, so a read begins by translating it. `normalize_settings_raw()` is that translation and its only copy: type coercion, deprecated per-subsystem retention keys folded into the unified one, the retired acceptance-pass count consumed into the shared review-cycle cap, retired keys dropped, renamed slots promoted, secret placeholders repaired. Supported legacy customizations survive those translations; declared retired keys (`settings_defaults.RETIRED_SETTING_KEYS`, `RETIRED_COMMA_LIST_SETTING_KEYS`) are DROPPED, so a reviewer comma-list config must move to `OUROBOROS_REVIEWER_SLOTS` before the upgrade or that install gets the shipped default panel. The ORDER is load-bearing (pass count before the purge, purge before the slot rename, so a retired spelling is never promoted), and every reader applies it BEFORE defaults merge: `load_settings()`, `_owner_read_settings_raw()`, `build_colab_settings()`. "Raw" names the runtime-mode ratchets it skips, never the migrations. Pure and idempotent — no file, no environment — it keeps a read a read while a read-modify-write applies it on every save, and every in-process reader shares `settings_integrity.read_settings_json_verified` — the loader, the owner reader, and the single-key `TOTAL_BUDGET` money-path read (`settings_setup_contract._saved_total_budget`, which needs no vocabulary translation) — so a changed pinned snapshot refuses them all exactly as the loader. The seam is VOCABULARY normalization only; provider normalization (`server_runtime.apply_runtime_provider_defaults`) is a separate, never-persisted derivation every route consumer makes, `context_fit.resolve_context_fit_route()` included.

Optional bounds. `OUROBOROS_MAX_ROUNDS` and `OUROBOROS_TASK_ABS_CEILING_SEC` take a positive integer or `unlimited` (`settings_scales.parse_positive_or_unlimited`, the review-cycle cap's vocabulary too) and ship `unlimited`: a fresh install bounds a task by money, deadlines, Stop/Panic and the idle rail, not by age or round count. Every creating writer persists a defaults-merged document, so a document WITHOUT such a key predates it: the three readers' defaults merge (`defaults_for_settings_document`) gives it the finite value it ran under (200 rounds / 21600 s; an unreadable document too) without rewriting it. A blank, null, zero, negative or malformed value on disk, in the environment or at a reader takes that finite value with one log line, never "no bound"; the generic Settings save refuses it. Readers (`get_max_rounds`, `get_task_abs_ceiling_sec`) return `None` for no bound; a finite lifetime has a 300-second floor. `upgrade_notices` distinguishes absent keys from saved invalid values and names the effective finite limits using those same runtime getters, without inferring a manual choice. Only a known owner binding receives it; its durable chat row recovers a failed `update_state` marker. Marker writes preserve unconfirmed controls; failed writes leave delivery or persistence unconfirmed. An operation that inherited the task lifetime (delegated or review session, VLM child, plan/preflight envelope, active-operation lease) takes `operation_window_sec()`: the finite lifetime, else `OPERATION_WINDOW_FALLBACK_SEC` (21600 s) — never unbounded, never a transport bound; deadlines still narrow it.

Five functions persist a settings document, each through `serialize_settings()` and a byte-exact helper (`utils.write_text_atomic()`, or `Path.write_bytes` on the config saver's rename-less `OSError` fallback) — never a text-mode write, which would turn LF into CRLF on Windows. Three persist THIS process's document through `prepare_settings_for_persist()`, the single point applying the disk-authored silence rule and ordinary-mode context/safety ratchets against the value ON DISK, with Cyber configuration authority from effective Access: `config.save_settings()`, `gateway/owner_settings._owner_update_settings()` (`_owner_write_settings()` is one of its callers), `packaged_cli._save_settings()`. Two are exempt by design and pinned as such: `context_mode_compat.normalize_and_persist_context_mode_compat()` (under the load lock; the raw mapping with only the pair changed, never a defaults-merged document) and `colab_bootstrap.write_colab_settings()` (a generated document for a foreign data root the prologue's on-disk proofs would answer wrongly for). One scan over `ouroboros/**`, `supervisor/**`, `server.py` and the repo-root `launcher.py` closes the inventory, reading routing from a function's calls and never its text; the predicate and the declared non-writer matches live in `tests._shared.settings_writers`, so a sixth writer in those roots fails the tripwire, routed or not.

An owner endpoint changes one decision inside a document it does not own, so it writes the whole document back. `_owner_update_settings(transform, expected_digest)` does that read-change-write inside ONE settings lock: the transform sees the document under the lock and returns what to persist, or nothing, so a no-change decision never rewrites the file. Acting on an earlier read, it passes that read's digest (`settings_document_digest()`, the onboarding transaction's staleness question); a mismatch refuses before the transform runs, so a concurrent owner change is never reverted key by key while the request answers "saved".

### LLM output token budgets

Providers name the same output budget differently: OpenRouter/Anthropic-compatible calls send `max_tokens`, every official direct OpenAI Chat route `max_completion_tokens` — a provider-wire boundary, not naming style. Direct OpenAI also sends the requested `reasoning_effort` provider-wide; model-name prefixes are not capability authority, and only exact-route success-confirmed wire evidence may adapt a request. Runtime floors (numeric SSOT: the constants, pinned by `tests/test_max_tokens_constants.py`):

| Surface | Output-token budget |
|---------|---------------------|
| `LLMClient.chat()` / `chat_async()` defaults | 65,536 |
| Main task loop (`loop_llm_call.MAIN_LOOP_MAX_TOKENS`) | 65,536 |
| `LLMClient.vision_query()` and VLM tools (`analyze_screenshot`, `vlm_query`) | 32,768 |
| Review synthesis dedup | 16,384 |
| Chat block consolidation, era compression, scratchpad consolidation | 16,384 |
| Execution reflection and pattern-register update | 16,384 |
| Post-task summary (`agent_task_pipeline`) | 16,384 |
| Improvement-backlog grooming (`improvement_backlog.groom_backlog`) | 8,192 |
| Post-task evolution promotion decision (`post_task_evolution`) | 8,192 |
| Context compaction round summaries | 32,768 |
| Skill publish PR body generation | 8,192 |
| Project naming LIGHT one-shot (`project_naming.llm_project_name`) | 256 |
| Update letter LIGHT one-shot (`update_letter.write_letter`) | 1,024 |
| Provider Test (`llm_probe.PROVIDER_TEST_MAX_TOKENS`) | 16 |

### Default settings

A registry of `config.SETTINGS_DEFAULTS` (exact defaults canonical in `settings_defaults.py`). Env-only rows are operator environment levers with no settings.json carrier.

| Key | Default | Description |
|-----|---------|-------------|
| OPENROUTER_API_KEY | "" | OpenRouter credential |
| OPENAI_API_KEY | "" | Official direct-OpenAI credential |
| OPENAI_BASE_URL | "" | Legacy OpenAI base-URL override |
| OPENAI_COMPATIBLE_API_KEY | "" | OpenAI-compatible endpoint credential |
| OPENAI_COMPATIBLE_BASE_URL | "" | OpenAI-compatible endpoint base URL |
| CLOUDRU_FOUNDATION_MODELS_API_KEY | "" | Cloud.ru credential |
| CLOUDRU_FOUNDATION_MODELS_BASE_URL | `https://foundation-models.api.cloud.ru/v1` | Cloud.ru base URL |
| GIGACHAT_CREDENTIALS | "" | GigaChat auth key |
| GIGACHAT_USER | "" | GigaChat user login |
| GIGACHAT_PASSWORD | "" | GigaChat password |
| GIGACHAT_SCOPE | `GIGACHAT_API_PERS` | GigaChat API scope |
| GIGACHAT_BASE_URL | `https://api.giga.chat/v1` | GigaChat base URL |
| GIGACHAT_VERIFY_SSL_CERTS | `true` | GigaChat TLS verification |
| GIGACHAT_PROFANITY_CHECK | "" | GigaChat profanity filter passthrough |
| OUROBOROS_EXTRA_CA_BUNDLE | "" | PEM file whose CA certificates are added to the default trust bundle for every first-party provider call (merged copy under `state/`; empty = defaults only) |
| ANTHROPIC_API_KEY | "" | Official direct-Anthropic credential |
| MINIMAX_API_KEY | "" | MiniMax credential |
| MINIMAX_REGION | "" | MiniMax region (empty resolves `global_en`) |
| DEEPSEEK_API_KEY | "" | Optional DeepSeek direct key (`deepseek::...` values; route below) |
| ZAI_API_KEY | "" | Optional Z.ai (GLM) direct key (`zai::...` values; route below) |
| ZAI_PLAN | "" | Z.ai endpoint plan: empty/`payg` = pay-as-you-go, `coding` = Coding Plan |
| OUROBOROS_NETWORK_PASSWORD | "" | Non-localhost HTTP gate password (`server_auth.py`; unset only warns — §8) |
| OUROBOROS_SERVER_HOST | 127.0.0.1 | HTTP bind host (`0.0.0.0` for Docker/non-loopback) |
| OUROBOROS_UPDATE_CHANNEL | `stable` | Update channel: stable/qa/development (§8) |
| OUROBOROS_MANAGED_UPDATE_FETCH_TIMEOUT_SEC | 300 | Managed-update fetch ceiling |
| OUROBOROS_RESCUE_GIT_TIMEOUT_SEC | 300 | Per-process ceiling on rescue Git commands |
| OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD | unset | Env-only: `1` permits saving a non-loopback bind without a password |
| OUROBOROS_MODEL | google/gemini-3.8-flash | Main model |
| OUROBOROS_MODEL_HEAVY | "" | Legacy slot outside `ACTIVE_MODEL_SETTING_KEYS`: read for migration/history, never routed |
| OUROBOROS_MODEL_LIGHT | openai/gpt-5.6-luna | Light model |
| OUROBOROS_MODEL_ACCOUNTS | "{}" | Role-owned managed account pins; empty means Auto, fallback entries retain order |
| OUROBOROS_PROCESSING_PREFERENCE | "" | One global provider-neutral preference: standard/fast/economy; empty preserves adapter default |
| OUROBOROS_MODEL_PROCESSING_PREFERENCES | "{}" | Optional role-owned preferences; empty roles inherit the global one |
| OUROBOROS_MODEL_CONTEXT_WINDOWS | "{}" | Role-owned context sizing assertions; zero means Auto, not a provider limit, and no value grants reviewer authority |
| OUROBOROS_MODEL_VISION | "" | Vision model (empty inherits) |
| OUROBOROS_IMAGE_INPUT_MODE | auto | Send-time image routing (`vision_routing.py`) |
| OUROBOROS_VISION_CAPTION_TIMEOUT_SEC | 90 | Caption-generation ceiling |
| OUROBOROS_MODEL_CONSCIOUSNESS | "" | Background-consciousness model (empty inherits) |
| OUROBOROS_MODEL_FALLBACKS | openai/gpt-5.6-luna | Cross-model fallback chain (`fallback_cooldown.py`) |
| OUROBOROS_SERVED_MODEL_REDOS | 2 | How many times a round another model answered may be asked again, each on a new operation (`runtime_limits.get_model_substitution_redos`, clamped 0-5) |
| OUROBOROS_MODEL_MAX_CONCURRENCY | 3 | Per-(model,route) concurrent provider-call cap (`model_concurrency.py`) |
| OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC | 180 | Concurrency-slot wait bound |
| OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC | 60 | Project-naming call ceiling |
| OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC | 8 | Inline naming bound when a card becomes a project (`gateway/projects.py`); a direct Main turn is named in the background once it starts working (`spawn_turn_namer`, bounded by `OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC` + 30 s) |
| OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC | 120 | Update-letter LIGHT one-shot ceiling, slot wait and provider call together (`update_letter.py`) |
| OUROBOROS_UI_TRANSLATION_TIMEOUT_SEC | 120 | One translation-generator batch call, slot wait and provider call together (`ui_translation.py`); the output budget per batch is the module constant `UI_TRANSLATION_MAX_TOKENS` |
| OUROBOROS_UI_LANGUAGE | "" | The owner's interface language for this install: a BCP-47 tag (`ru`, `pt-BR`, `art-x-<slug>` for an invented language), an open set; `""` = not chosen (the English source renders), `en` = chosen English. Written only by `POST /api/ui/i18n/language` through the locked owner writer (`ENDPOINT_WRITTEN_SETTINGS`: the generic settings save skips it and names the writer in `ignored_keys`), yet exported to the environment like any other setting, so a restart, the worker and the Telegram skill read the choice; read live by the gateway and the Telegram skill, by the mind's runtime block at its next attempt (`ui_language.py`, `i18n_memory.py`, §3 Settings and onboarding) |
| OUROBOROS_FALLBACK_COOLDOWN_ENABLED | true | 429-aware per-process model cooldown |
| OUROBOROS_FALLBACK_COOLDOWN_SEC | 120 | Cooldown window |
| OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL | 1 | Attempts per model in the fallback walk |
| OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS | 900000 | Owner ceiling (chars) on ONE working view of the native review episode; effective bound = min(this, calibrated route capacity); mandatory reading uses successive views and never raises it (shortfall disclosed as `native_multiple_windows_required`); no round cap (the retired round-cap key: §11.4) — exhaustion fails closed for verdicts and discloses an incomplete report, never a silent truncation (§6 Review delivery) |
| OUROBOROS_MODEL_DEEP_SELF_REVIEW | (empty) | Legacy source for an absent `deep_review` row; native retrieval. Nonempty choices stay pinned. Empty uses Main on a fresh compatible-only panel, otherwise the OpenRouter default; a saved panel retains its legacy default. The row in Agents → Review lanes overrides this key. |
| OUROBOROS_MAX_WORKERS | 10 | Active worker dispatch capacity; required owner waits retain additional sleeping processes |
| OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT | 6 | Live-subagent cap per root (hard cap 500 ids; depth hard cap 10) |
| OUROBOROS_MAX_SUBAGENT_DEPTH | 3 | Subagent tree depth |
| OUROBOROS_DISABLE_MANAGED_UPDATES | (unset) | Env-only: `1` disables managed updates (`git_ops.py`) |
| OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS | (empty) | Mutative-subagent Auto override |
| OUROBOROS_SUBAGENT_WORKTREE_ROOT | (empty) | Acting-worktree root (empty derives `~/Ouroboros/subagent_worktrees`) |
| OUROBOROS_SUBAGENT_PROJECTS_ROOT | (empty) | Projects root (empty derives `~/Ouroboros/projects`) |
| OUROBOROS_SUBAGENTS | (empty) | Canonical configured-subagent roster (`configured_subagents.py`; §6) |
| OUROBOROS_SUBAGENT_HARNESS | (empty) | Legacy narrow harness input |
| OUROBOROS_SUBAGENT_PROFILE | (empty) | Legacy narrow profile input |
| OUROBOROS_DELEGATE_WAIT_SEC | 120 | Default delegate_wait window |
| OUROBOROS_DELEGATE_WAIT_MAX_SEC | 1800 | delegate_wait ceiling |
| OUROBOROS_DELIVERABLES_ROOT | (empty) | Deliverables root (empty derives the `~/Ouroboros/Deliverables` sibling; `tool_access.py`) |
| OUROBOROS_GC_RETENTION_DAYS | 7 | Unified GC retention (`retention.py`) |
| OUROBOROS_RESTART_DRAIN_MAX_SEC | 120 | Restart drain bound |
| TOTAL_BUDGET | 200.0 | Global budget (USD); an absent key resolves to this product default, a non-positive value means no finite limit; resolved live from the saved document, so a change binds running tasks at their next model call (`resolve_total_budget_usd`) |
| OUROBOROS_PER_TASK_COST_USD | 50.0 | Per-task cost cap and tree ceiling basis: the root resolves min(global share, cap minus margin), descendants retain it, admission stays independent, and the wrap-up affordability rail soft-lands under it (`task_pacing.py`, §6 Budget tracking) |
| OUROBOROS_RUB_USD_RATE | (empty) | Manual RUB→USD rate for RUB-priced providers |
| OUROBOROS_PRICING_TTL_SEC | 21600 | Provider-catalog pricing cache TTL |
| OUROBOROS_TOOL_TIMEOUT_SEC | 600 | Default tool timeout |
| OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC | 1800 | Per-call timeout clamp |
| OUROBOROS_FINALIZATION_GRACE_SEC | 120 | Finalization grace window |
| OUROBOROS_WEBSEARCH_MODEL | gpt-5.2 | web_search backing model |
| OUROBOROS_WEBSEARCH_BACKEND | auto | web_search backend selection |
| OUROBOROS_MAIN_WEB_SEARCH | off | Main-loop inline web search |
| OUROBOROS_MAIN_WEB_SEARCH_ENGINE | auto | Inline-search engine |
| OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS | 10 | Inline-search result cap |
| OUROBOROS_OR_PROVIDER | "" | OpenRouter provider-routing preference merged into requests |
| OUROBOROS_SEARCH_CODE_WALL_SEC | 45 | search_code wall-clock budget |
| OUROBOROS_PRESENTATION | (unset) | Env-only: launcher-exported presentation (`desktop_window`/`browser_fallback`/external `android_app`; absent renders `web`) |
| OUROBOROS_EXTERNAL_HOST_UPDATE | (unset) | Env-only: selected external-host installer supporting read-only `--check`; no second Git updater |
| OUROBOROS_EXTERNAL_HOST_RESULT | (unset) | Env-only: launcher-verified installed-artifact/input/source facts for one core generation; not a reusable persisted PASS |
| OUROBOROS_USER_FILES_ROOT | "" (home) | Env-only: user_files jail root (empty = `$HOME`) |
| OUROBOROS_OBSERVABILITY_KEEP_RAW | unset | Env-only: truthy enables raw observability payload persistence |
| OUROBOROS_GENERATIVE_PROBE | 1 (on) | Generative-write probe toggle |
| OUROBOROS_GENERATIVE_PROBE_CHARS | 5000000 | Generative-probe size companion |
| OUROBOROS_REVIEWER_SLOTS | (empty) | Structured reviewer-slot SSOT (`reviewer_slot_config.py`); empty = the shipped default panel; contract: #### Reviewer slots |
| OUROBOROS_SUBSCRIPTION_PRESET_VERSION | (empty) | One-shot install-preset marker; endpoint-authored, DISK-ONLY (`ENDPOINT_AUTHORED_SETTINGS`); its absence authorizes nothing (§2) |
| OUROBOROS_SUBAGENT_PRESET_RECEIPT | (empty) | Install-preset receipt; endpoint-authored, disk-only |
| OUROBOROS_ONBOARDING_COMPLETED_AT | (empty) | Durable completion fact; endpoint-authored, disk-only |
| OUROBOROS_TASK_REVIEW_MODE | auto | Task acceptance-review mode |
| OUROBOROS_SAFETY_MODE | full | Safety supervision mode (shipped `full`; a fresh desktop wizard may author `light`); lowering is owner-guarded (§6 Safety and runtime mode) |
| OUROBOROS_SAFETY_MAX_TOKENS | 2000 | Safety-check output budget |
| OUROBOROS_SAFETY_CALL_TIMEOUT_SEC | 60 | Safety-check call ceiling |
| OUROBOROS_WEBSEARCH_TIMEOUT_SEC | 480 | web_search ceiling |
| OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC | 2 | Seconds the chat lane waits for a stopped direct turn to reach its next round boundary after `finalize_now`; past it the outcome is `live` and the sweep retries custody rather than publishing a terminal (clamped 0-10) |
| OUROBOROS_ONBOARDING_SNAPSHOT_TIMEOUT_SEC | 45 | Bound on the onboarding transaction's settings snapshot read, so a wedged filesystem refuses the step instead of hanging the wizard |
| OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC | 30 | Lock bound for an owner read-modify-write and `_run_settings_writer`; the lock is a precondition of the write, never a hint — one lock wait plus one held episode for the generic save, the owner endpoints and onboarding completion alike; a timeout REFUSES before the transform runs and Save answers 503 `settings_save_timeout` with `saved: null` (residual: a body that outlives the bound is left to its thread) |
| OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC | 2700 | LLM transport read timeout |
| OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC | 300 | plan_task deadline floor |
| OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC | 200 | Floor (s) of spendable time above the finalization reserve to START a critic panel; below it `review_skipped_deadline_reserve`. Author corrections use ordinary remaining task time and explicit task-local limits, without this reviewer floor (§6 Task lifecycle) |
| OUROBOROS_REVIEW_MAX_CYCLES | "2" | Shared paid review-cycle cap over the plan/acceptance/commit/skill gates; `unlimited` = no local count cap (`review_cycles.py`, §6 Review stack) |
| OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES | (retired) | Retired alias: a stored value is MIGRATED into `OUROBOROS_REVIEW_MAX_CYCLES` (passes + 1) at load; a leftover env value is inert |
| OUROBOROS_ACCEPTANCE_RESERVE_PCT | 5 | Acceptance budget reserve percentage |
| OUROBOROS_OBSERVABILITY_RETENTION_DAYS | (retired) | Retired (`RETIRED_SETTING_KEYS`): rows are kept indefinitely and the reader never deletes, so the knob had no reader; stored value stripped at load, env inert |
| OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC | (unset) | Env-only: logical review timeout (absent = route-owned; late in-flight results stay in custody) |
| OUROBOROS_REVIEW_MAX_TOKENS | 65536 | Env-only: reviewer output budget, clamped to the 8192 floor |
| OUROBOROS_REVIEW_ENFORCEMENT | advisory | Review enforcement: advisory/blocking (closed enum; anything else coerces to the default); plan-review closure of a below-quorum blocking finding by a reasoned reject reads this configured value, never the hurry-projected advisory the finalization gate uses |
| OUROBOROS_PREFLIGHT_TIMEOUT_SEC | 1800 | Env-only: TOTAL wall-clock budget for the hermetic test preflight (node lane + both passes); Android entry defaults to 3600, explicit override wins; teardown/containment remain in `preflight_runner.py`/`process_containment.py` |
| OUROBOROS_PREFLIGHT_SERIAL | unset | Env-only: `1` selects one serial pytest pass; scrubbed from the candidate environment |
| OUROBOROS_PREFLIGHT_TEST_WORKERS | (unset) | Env-only: xdist workers for the hermetic parallel pass (floor 2, else `os.cpu_count()`); read from the OPERATOR environment, scrubbed from the candidate |
| OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS | true | Auto-grant manifest-declared permissions to cleanly reviewed skills (hash-bound; blocking findings never grant) |
| OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS | true | Launcher seed/resync writes hash-pinned `native_seed` verdicts; acts only at seed/resync, no runtime grant endpoint |
| OUROBOROS_CONTEXT_MODE | max | Context mode `nano`/`low`/`max`, owner-selected outside Cyber Pro; `nano` records `owner_nano`/`rendered_mode=nano`; sizes Ouroboros's own working window, while scope review runs in every mode (BIBLE P1/P3); Cyber may configure it through the same audited writer (§6 Context fitting, retry, and compaction) |
| OUROBOROS_CONTEXT_MODE_AUTO_LOW | false | Provenance tombstone of the RETIRED persistent auto-Low, not a toggle: normalization writes `false` and `get_owner_context_mode()` honours a persisted `low` only beside it (`context_mode_compat.py`); task-local overflow retry is separate (§6 Context fitting, retry, and compaction) |
| OUROBOROS_RUNTIME_MODE | advanced | Effective Access light/advanced/pro/cyber_pro, persisted as the next-boot value; boundaries and Cyber agency: §6 Safety and runtime mode; review enforcement stays independent |
| OUROBOROS_SKILLS_REPO_PATH | "" | Extra skills checkout path (expanded at read time, never cloned/pulled) |
| MCP_ENABLED | false | MCP client toggle (§6 MCP) |
| MCP_SERVERS | [] | MCP server list (HTTP/SSE via URL/auth, stdio via command+args and optional cwd/literal/settings-backed env; a stdio `browser_bridge: true` makes it a task-owned Playwright MCP bridge, §6); persisted in settings, never env-exported |
| MCP_TOOL_TIMEOUT_SEC | 60 | Per-MCP-tool timeout |
| OUROBOROS_HUB_CATALOG_URL | `https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json` | OuroborosHub catalog URL (automatic fetch limited to catalog JSON; installs verify SHA-256) |
| OUROBOROS_CLAWHUB_REGISTRY_URL | `https://clawhub.ai/api/v1` | ClawHub registry URL |
| OUROBOROS_PROMPT_CACHE_TTL | 1h | Prompt-cache tier default/5m/1h for cache markers on compatible Anthropic-family wire payloads; the final send boundary legalizes ordering, so prompt builders own no provider TTL policy; `review_helpers.cached_prompt_blocks` and `usage_accounting._reservation_cost` also consult it; usage records the applied tier |
| OUROBOROS_EFFORT_TASK | medium | Task reasoning effort (none/minimal/low/medium/high/xhigh/max/ultra; Settings hides `minimal`); preferred tier; exact-route, success-confirmed adaptation, original/sent/reported facts stay in usage/Logs. Controls Light post-task synthesis (reflection, Pattern Register update, episodic summary), which has no separate level |
| OUROBOROS_EFFORT_EVOLUTION | high | Evolution effort |
| OUROBOROS_EFFORT_REVIEW | high | Review effort for rows that pin none; a plan envelope's `reviewer_effort` outranks it and a row's pinned effort for that plan (a compound route slug keeps its encoded effort); the effective per-seat effort is recorded and a panel ordered weaker than the owner's setting is named |
| OUROBOROS_EFFORT_SCOPE_REVIEW | high | Scope-review effort |
| OUROBOROS_EFFORT_DEEP_SELF_REVIEW | high | Deep-self-review surface default; a saved `deep_review` row's own effort outranks it |
| OUROBOROS_EFFORT_CONSCIOUSNESS | (empty) | Consciousness effort; empty = the Task / Chat effort (a wake is an ordinary Main turn), a set value is honored |
| OUROBOROS_RETURN_REASONING | true | Ask OpenRouter to return reasoning; direct/local request copies strip OpenRouter-only fields |
| OUROBOROS_REASONING_SUMMARY | auto | Readable reasoning-summary rendering; presentation-only, never added to history or returned to providers |
| OUROBOROS_TASK_IDLE_TIMEOUT_SEC | 900 | Idle timeout; needs absence of real task/subtree progress — a typed in-flight main-LLM row spares only this rail, a settled child result stamps parent progress, because delivery creates immediate integration work and must not coincide with idle termination |
| OUROBOROS_TASK_ABS_CEILING_SEC | unlimited | Optional absolute task lifetime in seconds (floor 300), activity-independent; deadline and budget stay separate hard axes |
| OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC | 90 | Supervisor/direct-turn liveness watchdog; alerts and recommends `/restart`, never frees an in-process lock held by a wedged turn |
| OUROBOROS_PACING_INTERVAL_SEC | 600 | Pacing reminder interval |
| LOCAL_MODEL_SOURCE | "" | Local-model source (HF repo or path) |
| LOCAL_MODEL_FILENAME | "" | Local-model GGUF filename (split first-shard expanded) |
| LOCAL_MODEL_CONTEXT_LENGTH | 16384 | Local-model context window |
| LOCAL_MODEL_N_GPU_LAYERS | 0 | GPU offload layers |
| USE_LOCAL_MAIN | false | Route Main locally |
| USE_LOCAL_HEAVY | false | Legacy migration input only; excluded from active routing |
| USE_LOCAL_LIGHT | false | Route Light locally |
| USE_LOCAL_CONSCIOUSNESS | false | Route Consciousness locally |
| USE_LOCAL_FALLBACK | false | Route fallback locally |
| OUROBOROS_MAX_ROUNDS | unlimited | Optional total task round limit (hot-reloadable; a Presence turn keeps its own finite inline cap) |
| OUROBOROS_TRANSIENT_RETRY_MAX | 6 | Same-model transient retry budget; pre-dispatch `transport_unavailable` is a separate task-bounded outer wait, since no provider attempt was admitted |
| OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC | 1800 | Skill lifecycle-lane timeout |
| OUROBOROS_CLAUDEXOR_HARNESS_INSTALL_TIMEOUT_SEC | 300 | Harness install ceiling (kills the tracked group, typed refusal) |
| OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC | 90 | Quota-refresh POST ceiling (clamped 1–90) |
| OUROBOROS_BUNDLE_DIR | (unset) | Env-only: launcher-owned bundle root propagated to embedded children for Node/ripgrep discovery |
| OUROBOROS_BG_WAKEUP_MIN | 900 | Lower bound (s) of the model-chosen wake interval (`set_next_wakeup`), clamped into [min, max] and re-read at each alarm, so a change needs no restart |
| OUROBOROS_BG_WAKEUP_MAX | 14400 | Upper bound (s), never below the minimum; with no chosen interval the alarm uses `runtime_limits.WAKE_DEFAULT_SEC` (3300 s, just under the 1 h cache TTL, keeping the shared prefix warm) — a constant rather than a third knob |
| OUROBOROS_CONSCIOUSNESS_AUTONOMY | act | Closed enum for a wake (else the default): `observe` researches and performs internal work, can start and stop its own read-only children and use owner-governed schedule controls, and cannot write shell/user files/source/skills/settings or publish; `act` adds all the runtime mode allows except own code/prompts, evolution, restart, settings; `full` includes evolution |
| OUROBOROS_CONSCIOUSNESS_DAILY_USD | 20.0 | Rolling-24h ceiling over consciousness wakes and the tasks they start; exhausted blocks NEW wakes and tasks until spend leaves the window; `0` is a real choice |
| OUROBOROS_CONSCIOUSNESS_MAX_TASKS | 2 | How many consciousness-started tasks may run at once; `0` = it never starts tasks |
| OUROBOROS_POST_TASK_EVOLUTION | false | Post-task evolution promotion toggle; self-enablement is blocked at the shell/browser/settings/data-write guards, choosing an objective runs on Main because it is a high-leverage decision, execution stays behind review and owner gates |
| OUROBOROS_POST_TASK_EVOLUTION_CADENCE | llm | Promotion cadence `llm` or `every_n:k` (malformed normalizes to `llm`) |
| OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD | 0.0 | Remaining-global-budget start floor, not a cycle cap |
| OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE | "" | Owner-only persistent campaign bias; still passes review gates |
| LOCAL_MODEL_PORT | 8766 | Local-model server port |
| OUROBOROS_HOST_SERVICE_PORT | 8767 | Host Service port (loopback-only; §12) |
| OUROBOROS_PRESENCE_MAX_ACTIVE | 2 | Cross-process Presence turn cap (UI-bounded 1–20) |
| LOCAL_MODEL_CHAT_FORMAT | "" | Local-model chat template override |
| GITHUB_TOKEN | "" | GitHub token (push/PR/issues) |
| GITHUB_REPO | "" | Personal `origin` repository |
| OUROBOROS_FILE_BROWSER_DEFAULT | "" | File Browser default root (explicit root required for Docker/non-localhost) |

#### Reviewer slots

`OUROBOROS_REVIEWER_SLOTS` owns `{triad[], scope[], advisory, deep_review?}` (`reviewer_slot_config.py`). Each stable `slot_id` names either an inline route (`api_chat`|`agent_session`, `target_id`) or `subagent_id`, never both. Roster references resolve route and effort at load; row effort outranks roster effort, a plan caller's effort outranks the row, and a compound model slug keeps its own effort. Disabled or missing references refuse rather than reroute. Save validates against the roster that transaction produces, allowing disable-and-reassign together. Only sessions and managed-model API routes may pin `route.profile_id` (empty means rotation).

Scope and deep review retrieve on every route. The optional `deep_review` singleton has fixed id `deep_review_slot_1`; absent, it is synthesized from `OUROBOROS_MODEL_DEEP_SELF_REVIEW`. Direct API triad rows may save `delivery="native"` or `"packet"`; a legacy bare saved row remains packet, without a migration offer. Delivery elsewhere refuses typed. Consumers use the common `retrieves` fact. Saved panels of unknown provenance keep their inherited deep default; a compatible-only install may need an ordinary manual edit. Fresh defaults and onboarding provide three native readers plus scope; an OpenAI-compatible-only install uses Main's route. Unrelated Save omits defaults; first saving a default panel pins its displayed deep row. A model positively known not to support tools may be switched to Packet; quota, timeout and unknown capability never imply such support evidence or automatic fallback.

Empty means shipped defaults; malformed rows refuse Save/review. Retired comma/route envs are stripped. Window sizes grant no authority. Packet→retrieving Save returns `acceptance_delivery_disclosure`: rows/routes, measured packet costs (`_ACCEPTANCE_API_PANEL_MEASURED`), native round costs and subscription time. Keeping retrieval or returning to packet emits no migration notice. Startup describes defaults only without a saved panel; saved rows stay pinned. Default native delivery changes the review-contract identity, so a later requested review may buy a new paid review; the change itself launches nothing and promises no exact review count.

#### Direct-provider routes

Direct-provider review fallback (legacy name: OpenAI-only review fallback): with exactly one official direct provider configured, `config.get_review_models()` compiles that provider's declarative reviewer-role sequence from provider-prefixed model IDs. Scope covers official OpenAI, Anthropic, MiniMax, DeepSeek, Z.ai, Cloud.ru, and GigaChat; OpenRouter, legacy-base and mixed configurations stay outside; fresh compatible-only defaults use `compatible_only_main_model` for every reviewer, including deep review; saved choices/panels stay pinned. Per-provider role coverage differs — three independent Main slots down to one role model for every slot (`provider_models.compute_direct_review_models_fallback`). `_exclusive_direct_remote_provider_env` returns empty when OpenRouter, legacy `OPENAI_BASE_URL`, OpenAI-compatible keys or several direct providers are present, and the fallback requires `provider_models.migrate_model_value` to make the main model already start with the exclusive provider prefix, so free text cannot silently enter a single-provider route (DEVELOPMENT "Provider Independence").

DeepSeek (`deepseek::`) uses fixed `provider_models.DEEPSEEK_BASE_URL`; proxies/mirrors use `openai-compatible::`, and `deepseek/...` stays OpenRouter. `provider_models.DEEPSEEK_REASONING_EFFORT_ALIASES` maps send-time effort; tier changes report `reasoning_effort_clamped`. Forced tool choice disables thinking, which permits only `auto`/`none`. Canonical `reasoning_content` replays for v4; a turn without reasoning carries an empty string. Strict non-echo lanes/OpenRouter strip it on send, and cross-family switches scrub it. Only send copies flatten system/assistant/tool arrays (`llm_openai_compatible.py`); user arrays remain valid. Caching is automatic; unknown catalog cost stays null. 1M requires route-fingerprinted evidence or owner acknowledgement.

MiniMax (`minimax::`) sends `reasoning_split=true`. Raw `reasoning_details`/`reasoning_content` survive same-route tool and held no-tool turns verbatim; host cleanup preserves opaque payloads. Final `content` stays intact; no think-tag parsing. Details narration is display-only. Live wire and cross-route portability are unverified.

Z.ai (`zai::`): GLM through Z.ai's OpenAI-compatible API. `ZAI_PLAN` selects the endpoint (`provider_models.resolve_zai_base_url`: empty/`payg` = `api.z.ai/api/paas/v4`, `coding` = the Coding Plan endpoint); a proxy or the China host belongs to `openai-compatible::`, and slash-form `zai/...` stays OpenRouter. An absent `reasoning_effort` is served at the provider's MAX, so the canonical scale is always projected onto Z.ai's own `low/high/max` enum at the send boundary (`provider_models.ZAI_REASONING_EFFORT_ALIASES`: none/minimal→low, medium→high, xhigh/ultra→max — GLM-5.3 rejects every other value and cannot disable thinking, HTTP 400 code 1210) and every tier change is disclosed as `reasoning_effort_clamped`; a forced tool choice keeps its tier (no DeepSeek-style suppression). HTTP 429 code 1113 "Insufficient balance" is billing (also a Coding Plan key on the pay-as-you-go endpoint), so the provider Test reports it as `No credits`, not "Rate limited".

GigaChat (`gigachat::`): the native `gigachat` library, not OpenAI-compatible — OpenAI `tools` map to GigaChat `functions`, one `function_call` per turn (parallel `tool_calls` collapse to the first), `tool` results become role `function` and must be valid JSON (plain text is wrapped as `{"result": ...}`), and `system` must come first, so later system-reminders demote to `user` (`llm.py::_chat_gigachat`). `reasoning_effort` is deliberately omitted: hidden reasoning can consume the whole output budget and return empty content. No live cost source exists, so cost stays nullable/unknown, never a hand-maintained tariff. A GigaChat scope row runs native retrieval on its own window, with at most one function call per turn. Reading gaps are diagnostic and never remove its response from quorum; the agent decides whether more reading is needed. Missing inspection tools still produce `native_inspection_unavailable`, not a completed review, and the blocking triad continues to review the full staged diff (§6 Review stack).

---
