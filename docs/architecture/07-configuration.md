# 7. Configuration (ouroboros/config.py)

This chapter owns the settings surface: where the document lives, which functions may persist it, how an older release's document is translated into today's vocabulary before defaults merge, the per-surface output budgets, and the registry of shipped defaults. One key, one home, one default — and this table is a test-mirrored projection of `config.SETTINGS_DEFAULTS`, never a second authority.

`ouroboros/config.py` is the one IMPORT surface: paths (HOME, APP_ROOT, REPO_DIR, DATA_DIR, SETTINGS_PATH, PID_FILE, PORT_FILE), constants (RESTART_EXIT_CODE 42, AGENT_SERVER_PORT 8765), `load_settings()`/`save_settings()`, `apply_settings_to_env()` (hot-reloadable keys into `os.environ`), `normalize_runtime_mode()` (one clamp for the save path, the read coercion and onboarding validation), `get_runtime_mode()`/`get_skills_repo_path()`, `acquire_pid_lock()`/`release_pid_lock()`. The vocabularies live in leaves it re-exports — `settings_defaults.py`, `settings_scales.py` (whose `IMMEDIATE_SETTINGS` and `RESTART_REQUIRED_SETTINGS` name the keys that bite the running process at once or wait for a restart; a key in neither applies to the next task), `model_slots.py`, `review_model_routes.py`, `runtime_limits.py`, `settings_integrity.py` — so a new key and default belong to a leaf, never the facade (§10 invariant 3). `update_channels.py` owns `get_update_channel()`/`get_update_branch()` and the update-network defaults.

Settings file: `data/settings.json` under the data root (`~/Ouroboros/data/settings.json` by default; `APP_ROOT`, `DATA_DIR` and `SETTINGS_PATH` independently env-overridable), accessed under a file lock. `secret_masking.py` owns the Settings/MCP placeholder emitters and recognizers for known and owner-defined top-level secrets and the secret BYTE shapes masked on tool-output egress: `load_settings()` repairs only recognized DISK placeholders before environment precedence resolves, so a real environment credential is never read as a mask, and `prepare_settings_for_persist()` repeats that repair at the writer boundary; nested MCP values are never silently migrated. While `OUROBOROS_SETTINGS_SHA256` is set (`settings_integrity.py`) the seeded snapshot is an owner-authored trust root: every read verifies the whole byte stream and every writer refuses.

`ouroboros/openrouter_attribution.py` is the application-identity SSOT for every first-party paid OpenRouter request (canonical URL + `X-OpenRouter-Title`); a fork must use its own URL rather than competing to rename one app record.

The update letter's material is `base..target` (every commit reachable from the official target but not the running base, merged branches included) plus the README history rows added along the target's first-parent line; one accounted LIGHT-slot call writes it into `state/update_letter.json`, whose single projection the Updates panel and Runtime context share (`update_letter.py`, §3, §6).

### Reading and writing the settings document

`agent.handle_task` binds the admitted task's `settings_integrity.TaskSettingsSnapshot` for the whole task entry: document and environment projection are separate private in-memory views, so absent and empty stay distinct, concurrent tasks keep their own models/keys/Supervisor/Review settings, and explicit task route overrides still win. `SETTINGS_ENV_LOCK` serializes capture and publication only, never execution; `runtime_setting`/`runtime_settings`/`runtime_environ` read that view, writers still read the current document, immediate effects stay live, Access keeps its boot pin, and `model_wait.copy_wait_context` plus task-owned helper threads carry only the existing binding. Out-of-process extensions get only permitted typed next-task values through the existing per-call payload — no whole snapshot, no extra credential environment — and still validate grants and read immediate values live. A failed reload loudly keeps the prior environment, disclosing the document-only values it cannot answer.

A document on disk was written by whatever release the owner last used, so a read begins by translating it. `normalize_settings_raw()` is that translation and its only copy: type coercion, deprecated per-subsystem retention keys folded into the unified one, the retired acceptance-pass count consumed into the shared review-cycle cap, retired keys dropped, renamed slots promoted, secret placeholders repaired. Supported legacy customizations survive those translations; declared retired keys (`settings_defaults.RETIRED_SETTING_KEYS`, `RETIRED_COMMA_LIST_SETTING_KEYS`) are DROPPED, so a reviewer comma-list config must move to `OUROBOROS_REVIEWER_SLOTS` before the upgrade or that install gets the shipped default panel. The ORDER is load-bearing (pass count before the purge, purge before the slot rename, so a retired spelling is never promoted), and every reader applies it BEFORE defaults merge: `load_settings()`, `_owner_read_settings_raw()`, `build_colab_settings()`. "Raw" names the runtime-mode ratchets it skips, never the migrations. Pure and idempotent — no file, no environment — it keeps a read a read while a read-modify-write applies it on every save, and both in-process readers share `settings_integrity.read_settings_json_verified`, so a changed pinned snapshot refuses the owner reader exactly as the loader. The seam is VOCABULARY normalization only; provider normalization (`server_runtime.apply_runtime_provider_defaults`) is a separate, never-persisted derivation every route consumer makes, `context_fit.resolve_context_fit_route()` included.

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
| ANTHROPIC_API_KEY | "" | Official direct-Anthropic credential |
| MINIMAX_API_KEY | "" | MiniMax credential |
| MINIMAX_REGION | "" | MiniMax region (empty resolves `global_en`) |
| DEEPSEEK_API_KEY | "" | Optional DeepSeek direct key (`deepseek::...` values; route below) |
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
| OUROBOROS_MODEL_MAX_CONCURRENCY | 3 | Per-(model,route) concurrent provider-call cap (`model_concurrency.py`) |
| OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC | 180 | Concurrency-slot wait bound |
| OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC | 60 | Project-naming call ceiling |
| OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC | 8 | Inline naming bound when a card becomes a project (`gateway/projects.py`); a direct Main turn is named in the background once it starts working (`spawn_turn_namer`, bounded by `OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC` + 30 s) |
| OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC | 120 | Update-letter LIGHT one-shot ceiling, slot wait and provider call together (`update_letter.py`) |
| OUROBOROS_FALLBACK_COOLDOWN_ENABLED | true | 429-aware per-process model cooldown |
| OUROBOROS_FALLBACK_COOLDOWN_SEC | 120 | Cooldown window |
| OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL | 1 | Attempts per model in the fallback walk |
| OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS | 900000 | Owner ceiling (chars) on ONE working view of the native review episode; effective bound = min(this, calibrated route capacity); mandatory reading uses successive views and never raises it (shortfall disclosed as `native_multiple_windows_required`); no round cap (the retired round-cap key: §11.4) — exhaustion fails closed for verdicts and discloses an incomplete report, never a silent truncation (§6 Review delivery) |
| OUROBOROS_MODEL_DEEP_SELF_REVIEW | openai/gpt-5.6-sol | Deep self-review model; source for an absent API `deep_review` row (`deep_review_slot()`), which runs native retrieval; a saved row overrides it — the key is then not read, so its provider-default migrations (`server_runtime.py`) reach only installs that still synthesize; not a Settings field: the row lives in Agents → Review lanes (#### Reviewer slots) |
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
| TOTAL_BUDGET | 200.0 | Global budget (USD); an absent key resolves to this product default, a non-positive value means no finite limit (`resolve_total_budget_usd`) |
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
| OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC | 200 | Floor (s) of spendable time above the finalization reserve to START an acceptance panel, never lowered; improvement window ×2 adaptive, ×1 otherwise; below it `review_skipped_deadline_reserve`/`improvement_window_inside_reserve` (§6 Task lifecycle) |
| OUROBOROS_REVIEW_MAX_CYCLES | "2" | Shared paid review-cycle cap over the plan/acceptance/commit/skill gates; `unlimited` = no local count cap (`review_cycles.py`, §6 Review stack) |
| OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES | (retired) | Retired alias: a stored value is MIGRATED into `OUROBOROS_REVIEW_MAX_CYCLES` (passes + 1) at load; a leftover env value is inert |
| OUROBOROS_ACCEPTANCE_RESERVE_PCT | 5 | Acceptance budget reserve percentage |
| OUROBOROS_OBSERVABILITY_RETENTION_DAYS | (retired) | Retired (`RETIRED_SETTING_KEYS`): rows are kept indefinitely and the reader never deletes, so the knob had no reader; stored value stripped at load, env inert |
| OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC | (unset) | Env-only: logical review timeout (absent = route-owned; late in-flight results stay in custody) |
| OUROBOROS_REVIEW_MAX_TOKENS | 65536 | Env-only: reviewer output budget, clamped to the 8192 floor |
| OUROBOROS_REVIEW_ENFORCEMENT | advisory | Review enforcement: advisory/blocking (closed enum; anything else coerces to the default) |
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
| MCP_SERVERS | [] | MCP server list (HTTP/SSE via URL/auth, stdio via command+args and optional cwd/literal/settings-backed env); persisted in settings, never env-exported |
| MCP_TOOL_TIMEOUT_SEC | 60 | Per-MCP-tool timeout |
| OUROBOROS_HUB_CATALOG_URL | `https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json` | OuroborosHub catalog URL (automatic fetch limited to catalog JSON; installs verify SHA-256) |
| OUROBOROS_CLAWHUB_REGISTRY_URL | `https://clawhub.ai/api/v1` | ClawHub registry URL |
| OUROBOROS_PROMPT_CACHE_TTL | 1h | Prompt-cache tier default/5m/1h for cache markers on compatible Anthropic-family wire payloads; the final send boundary legalizes ordering, so prompt builders own no provider TTL policy; `review_helpers.cached_prompt_blocks` and `usage_accounting._reservation_cost` also consult it; usage records the applied tier |
| OUROBOROS_EFFORT_TASK | medium | Task reasoning effort (none/minimal/low/medium/high/xhigh/max/ultra; Settings hides `minimal`); adaptation is exact-route, success-confirmed, disclosed in `request_wire` |
| OUROBOROS_EFFORT_EVOLUTION | high | Evolution effort |
| OUROBOROS_EFFORT_REVIEW | high | Review effort; reaches plan review as every row's default rung unless the envelope declares `reviewer_effort` |
| OUROBOROS_EFFORT_SCOPE_REVIEW | high | Scope-review effort |
| OUROBOROS_EFFORT_DEEP_SELF_REVIEW | high | Deep-self-review surface default; a saved `deep_review` row's own effort outranks it |
| OUROBOROS_EFFORT_CONSCIOUSNESS | (empty) | Consciousness effort; empty = the Task / Chat effort (a wake is an ordinary Main turn), a set value is honored |
| OUROBOROS_RETURN_REASONING | true | Ask OpenRouter to return reasoning; direct/local request copies strip OpenRouter-only fields |
| OUROBOROS_REASONING_SUMMARY | auto | Readable reasoning-summary rendering; display-only — the stamped progress row is durable and replays through history (`_PROGRESS_META_FIELDS`), but reasoning never re-enters the prompt digest or is returned to providers |
| OUROBOROS_TASK_IDLE_TIMEOUT_SEC | 900 | Idle timeout; needs absence of real task/subtree progress — a typed in-flight main-LLM row spares only this rail, a settled child result stamps parent progress, because delivery creates immediate integration work and must not coincide with idle termination |
| OUROBOROS_TASK_ABS_CEILING_SEC | 21600 | Absolute task ceiling, activity-independent; deadline and budget stay separate hard axes |
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
| OUROBOROS_MAX_ROUNDS | 200 | Max task rounds (hot-reloadable) |
| OUROBOROS_TRANSIENT_RETRY_MAX | 6 | Same-model transient retry budget; pre-dispatch `transport_unavailable` is a separate task-bounded outer wait, since no provider attempt was admitted |
| OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC | 1800 | Skill lifecycle-lane timeout |
| OUROBOROS_CLAUDEXOR_HARNESS_INSTALL_TIMEOUT_SEC | 300 | Harness install ceiling (kills the tracked group, typed refusal) |
| OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC | 90 | Quota-refresh POST ceiling (clamped 1–90) |
| OUROBOROS_BUNDLE_DIR | (unset) | Env-only: launcher-owned bundle root propagated to embedded children for Node/ripgrep discovery |
| OUROBOROS_BG_WAKEUP_MIN | 900 | Lower bound (s) of the model-chosen wake interval (`set_next_wakeup`), clamped into [min, max] and re-read at each alarm, so a change needs no restart |
| OUROBOROS_BG_WAKEUP_MAX | 14400 | Upper bound (s), never below the minimum; with no chosen interval the alarm uses `runtime_limits.WAKE_DEFAULT_SEC` (3300 s, just under the 1 h cache TTL, keeping the shared prefix warm) — a constant rather than a third knob |
| OUROBOROS_CONSCIOUSNESS_AUTONOMY | act | Closed enum for a wake (else the default): `observe` thinks, remembers and writes but starts nothing; `act` adds all the runtime mode allows except own code/prompts, evolution, restart, settings; `full` includes evolution |
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

`OUROBOROS_REVIEWER_SLOTS` is the one reviewer configuration surface: `{triad[], scope[], advisory, deep_review?}`, row JSON owned by the `reviewer_slot_config.py` docstring. A row is EITHER an inline route (`api_chat`\|`agent_session` + `target_id`) OR a roster reference (`subagent_id`), never both — naming both refuses typed, and a reference materializes route and effort from the Available-subagents roster at load, an explicit row `effort` outranking the roster's. `slot_id` is a STABLE owner-assigned identity, never an array index. Only an `agent_session` or managed-model `api_chat` route may carry `route.profile_id` (empty = rotation; direct API-key routes reject account pins). The `deep_review` singleton takes those keys minus `slot_id` (fixed `deep_review_slot_1`); absent, it is synthesized as an API row from `OUROBOROS_MODEL_DEEP_SELF_REVIEW`. Every scope and deep-review API row runs native inspection and every session row delegates retrieval, with or without a roster reference; triad API rows retain packet delivery unless bound to a configured subagent. Windows size requests and grant no authority. Empty = the shipped default panel; retired comma and route envs are stripped at load and never read as configuration (§11.4). Malformed refuses typed at save AND at review time on every surface, task acceptance included; env-apply logs and leaves legacy keys unprojected. The save that FIRST gives the triad a retrieving row (agent session or configured-subagent native inspection) returns a one-time migration disclosure in the save response's `warnings`: the rows by id and target, and the measured API packet-panel cost they replace (`reviewer_slot_config._ACCEPTANCE_API_PANEL_MEASURED`, which the onboarding ladder footnote states independently) against API charges per round for native inspection or subscription-window minutes per task for an agent session. Keeping a retrieving triad discloses nothing again, nor does the reverse transition to a packet-only triad — a migration notice, not a routing monitor.

#### Direct-provider routes

Direct-provider review fallback (legacy name: OpenAI-only review fallback): with exactly one official direct provider configured, `config.get_review_models()` compiles that provider's declarative reviewer-role sequence from provider-prefixed model IDs. Scope covers official OpenAI, Anthropic, MiniMax, DeepSeek, Cloud.ru, and GigaChat; OpenRouter, legacy-base, OpenAI-compatible and mixed configurations stay outside. Per-provider role coverage differs — three independent Main slots down to one role model for every slot (`provider_models.compute_direct_review_models_fallback`). `_exclusive_direct_remote_provider_env` returns empty when OpenRouter, legacy `OPENAI_BASE_URL`, OpenAI-compatible keys or several direct providers are present, and the fallback requires `provider_models.migrate_model_value` to make the main model already start with the exclusive provider prefix, so free text cannot silently enter a single-provider route (DEVELOPMENT "Provider Independence").

DeepSeek (`deepseek::`): the OpenAI-compatible endpoint is the fixed constant `provider_models.DEEPSEEK_BASE_URL`; a proxy or mirror belongs to `openai-compatible::`, and slash-form `deepseek/...` stays OpenRouter. The canonical effort scale is projected onto the provider's wire dialect at the send boundary, a forced tool choice is served with thinking disabled (thinking accepts only `auto`/`none`), and every tier change is disclosed as `reasoning_effort_clamped` (projection table: `provider_models.DEEPSEEK_REASONING_EFFORT_ALIASES`). `reasoning_content` stays on CANONICAL assistant turns for strict v4 replay (an explicit empty string marks a turn produced without provider reasoning); other lanes strip the field from their physical send copy and cross-family switches scrub it. On the send copy only, system/assistant/tool content arrays are flattened to strings — the API accepts arrays on user turns alone (`llm_openai_compatible.py`). Caching is automatic, cost stays nullable without an exact catalog, and the 1M context claim needs route-fingerprinted evidence or owner acknowledgement.

GigaChat (`gigachat::`): the native `gigachat` library, not OpenAI-compatible — OpenAI `tools` map to GigaChat `functions`, one `function_call` per turn (parallel `tool_calls` collapse to the first), `tool` results become role `function` and must be valid JSON (plain text is wrapped as `{"result": ...}`), and `system` must come first, so later system-reminders demote to `user` (`llm.py::_chat_gigachat`). `reasoning_effort` is deliberately omitted: hidden reasoning can consume the whole output budget and return empty content. No live cost source exists, so cost stays nullable/unknown, never a hand-maintained tariff. A GigaChat scope row runs native retrieval on its own window, with at most one function call per turn. Reading gaps are diagnostic and never remove its response from quorum; the agent decides whether more reading is needed. Missing inspection tools still produce `native_inspection_unavailable`, not a completed review, and the blocking triad continues to review the full staged diff (§6 Review stack).

---

