# 7. Configuration (ouroboros/config.py)

This chapter owns the settings surface: where the document lives, which functions may persist it, how a document written by an older release is translated into today's vocabulary before any default is merged, the per-surface output-token budgets, and the full registry of shipped defaults with their meanings. It exists so a new key has exactly one home and one default, and so this table stays a test-mirrored projection of `config.SETTINGS_DEFAULTS` rather than a second authority.

`ouroboros/config.py` is the SSOT for paths (HOME, APP_ROOT, REPO_DIR, DATA_DIR, SETTINGS_PATH, PID_FILE, PORT_FILE), process constants (RESTART_EXIT_CODE 42, AGENT_SERVER_PORT 8765), every settings default below, and the load/save/env machinery: `load_settings()`, `save_settings()`, `apply_settings_to_env()` (copies hot-reloadable runtime keys into `os.environ`), `normalize_runtime_mode()` (one clamp shared by the save path, the read coercion, and onboarding validation), `get_runtime_mode()`/`get_skills_repo_path()`, and `acquire_pid_lock()`/`release_pid_lock()`; `ouroboros/update_channels.py` owns `get_update_channel()`/`get_update_branch()`.

Settings file: `data/settings.json` under the data root — `~/Ouroboros/data/settings.json` by default, with `APP_ROOT`, `DATA_DIR`, and `SETTINGS_PATH` independently env-overridable. Access is file-locked. `secret_masking.py` is the wire-placeholder authority for known and owner-defined top-level secrets: `load_settings()` repairs only recognized disk placeholders BEFORE environment precedence is resolved, so a real environment credential is never classified as a mask, and `prepare_settings_for_persist()` applies the same top-level repair at the common writer boundary; nested MCP values are never silently migrated.

`ouroboros/openrouter_attribution.py` is the application-identity SSOT for every first-party paid OpenRouter request (canonical URL + `X-OpenRouter-Title`); a fork must use its own URL rather than competing to rename one app record.

### Reading and writing the settings document

`agent.handle_task` binds the admitted task's `settings_integrity.TaskSettingsSnapshot` for the complete task entry. Its normalized document and exact environment projection are separate private in-memory views: absent and empty values remain distinct, concurrent tasks keep their own models/keys/Supervisor/Review settings, and explicit task route overrides still win. A short `SETTINGS_ENV_LOCK` serializes capture/publication only, never task execution. `runtime_setting`, `runtime_settings` and `runtime_environ` reuse that view; settings writers keep reading the current document, immediate effects stay live, and Access retains its boot pin. `model_wait.copy_wait_context` and task-owned helper threads carry only the existing settings binding alongside their established owners. Out-of-process extensions receive only their permitted typed next-task values through the existing private per-call payload; the child still validates current grants, reads immediate values live and receives no whole settings snapshot or extra credential environment. Failed reload loudly retains the prior environment while disclosing unavailable document-only values.

A settings document on disk was written by whatever release the owner last used, so reading one starts by translating it into today's vocabulary. `normalize_settings_raw()` is that translation and the only copy of it: type coercion against the declared defaults, the deprecated per-subsystem retention keys folded into the unified one, the retired acceptance-pass count consumed into the shared review-cycle cap, the keys a release retired dropped, the renamed model slots promoted, and secret placeholders repaired. Every step preserves an owner customization written under a former key, and the ORDER is load-bearing (the pass count is consumed before the retired purge would drop it; the purge runs before the slot rename, so a retired spelling is never promoted), so every reader applies it BEFORE the shipped defaults merge — `load_settings()`, the owner endpoints' `_owner_read_settings_raw()`, and the Colab re-run's `build_colab_settings()` alike. "Raw" in that name is about the runtime-mode ratchets it deliberately skips, never about the migrations. It is pure and idempotent — it touches no file and no environment — which is what lets a read stay a read and lets a read-modify-write apply it on every save; both in-process readers share one read primitive (`settings_integrity.read_settings_json_verified`), so a pinned snapshot that changed refuses the owner reader exactly as it refuses the loader. This seam carries the VOCABULARY normalization only: the provider normalization (`server_runtime.apply_runtime_provider_defaults`) is a separate, never-persisted derivation every route consumer makes over the effective document, `context_fit.resolve_context_fit_route()` included.

Five functions persist a settings document, and every one calls `serialize_settings()` and commits its output through a byte-exact helper (`utils.write_text_atomic()`, or `Path.write_bytes` on the config saver's rename-less `OSError` fallback) — never a text-mode write, which would turn LF into CRLF on Windows. Three persist THIS process's document through `prepare_settings_for_persist()`, the single point where the disk-authored silence rule and ordinary-mode context/safety ratchets are applied against the value ON DISK, with Cyber configuration authority from effective Access: `config.save_settings()`, `gateway/owner_settings._owner_update_settings()` (which `_owner_write_settings()` is one caller of), and the packaged bootstrap's `packaged_cli._save_settings()`. Two are exempt by design and pinned as such: the one-window raw context-pair migration (`context_mode_compat.normalize_and_persist_context_mode_compat()`, written under the load lock — the raw mapping with only the pair changed, never a defaults-merged document) and `colab_bootstrap.write_colab_settings()`, which writes a generated document for a foreign data root the prologue's on-disk proofs would answer wrongly for. One scan closes the inventory over `ouroboros/**`, `supervisor/**`, `server.py` and the repo-root `launcher.py` (`tests._shared.settings_writers`): a function is a settings writer when it CALLS `serialize_settings()`, or when it names the settings path or file AND does a write-shaped thing; it counts as ROUTED only when it CALLS the prologue, and naming either in prose is neither. The flagged set must equal the five writers plus the scan's declared non-writer matches, so a sixth writer in those roots fails the tripwire whether or not it is routed.

An owner endpoint changes one decision inside a document it does not otherwise own, so it must write the whole document back. `_owner_update_settings(transform, expected_digest)` does that read, change and write inside ONE settings lock: the transform receives the document as it is under the lock and returns what to persist, or nothing at all, which is how a no-change decision avoids rewriting the file. An endpoint acting on an earlier read passes the digest that read saw (`settings_document_digest()`, the same staleness question the onboarding transaction asks), and a mismatch refuses before the transform runs, so a concurrent owner change can never be reverted key by key while the request answers "saved".

### LLM output token budgets

Providers name the same output-token budget differently: OpenRouter/Anthropic-compatible calls send `max_tokens`, while every official direct OpenAI Chat route sends `max_completion_tokens` — a real provider-wire boundary, not naming style. Direct OpenAI also sends the requested `reasoning_effort` provider-wide; model-name prefixes are not capability authority, and only exact-route success-confirmed wire evidence may adapt a request. Runtime floors (numeric SSOT: the constants in code, pinned by `tests/test_max_tokens_constants.py`):

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

A registry of `config.SETTINGS_DEFAULTS` (exact defaults stay canonical in `config.py`; this table is test-mirrored against it). Rows marked env-only are operator environment levers with no settings.json carrier.

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
| DEEPSEEK_API_KEY | "" | Optional. DeepSeek direct provider key (`deepseek::...` model values, OpenAI-compatible API at the fixed official endpoint) |
| OUROBOROS_NETWORK_PASSWORD | "" | Non-localhost HTTP gate password (`server_auth.py`; unset only warns — see §8 packaging note) |
| OUROBOROS_SERVER_HOST | 127.0.0.1 | HTTP bind host (`0.0.0.0` for Docker/non-loopback) |
| OUROBOROS_UPDATE_CHANNEL | `stable` | Update channel: stable/qa/development (§8) |
| OUROBOROS_MANAGED_UPDATE_FETCH_TIMEOUT_SEC | 300 | Managed-update fetch ceiling |
| OUROBOROS_RESCUE_GIT_TIMEOUT_SEC | 300 | Per-process ceiling on rescue Git commands |
| OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD | unset | Env-only: `1` permits saving a non-loopback bind without a password |
| OUROBOROS_MODEL | google/gemini-3.8-flash | Main model |
| OUROBOROS_MODEL_HEAVY | "" | Legacy slot: readable for migration/history only, out of active routing |
| OUROBOROS_MODEL_LIGHT | openai/gpt-5.6-luna | Light model |
| OUROBOROS_MODEL_ACCOUNTS | "{}" | Role-owned managed account pins; empty means Auto, fallback entries retain order |
| OUROBOROS_PROCESSING_PREFERENCE | "" | One global provider-neutral processing preference: standard/fast/economy; empty preserves adapter default |
| OUROBOROS_MODEL_PROCESSING_PREFERENCES | "{}" | Optional role-owned processing preferences; empty roles inherit the global preference |
| OUROBOROS_MODEL_CONTEXT_WINDOWS | "{}" | Role-owned context sizing assertions; zero means Auto, not a provider limit or scope acknowledgement |
| OUROBOROS_MODEL_VISION | "" | Vision model (empty inherits) |
| OUROBOROS_IMAGE_INPUT_MODE | auto | Send-time image routing (`vision_routing.py`) |
| OUROBOROS_VISION_CAPTION_TIMEOUT_SEC | 90 | Caption-generation ceiling |
| OUROBOROS_MODEL_CONSCIOUSNESS | "" | Background-consciousness model (empty inherits) |
| OUROBOROS_MODEL_FALLBACKS | openai/gpt-5.6-luna | Cross-model fallback chain (`fallback_cooldown.py`) |
| OUROBOROS_MODEL_MAX_CONCURRENCY | 3 | Per-(model,route) concurrent provider-call cap (`model_concurrency.py`) |
| OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC | 180 | Concurrency-slot wait bound |
| OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC | 60 | Project-naming call ceiling |
| OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC | 8 | Bound of the inline naming call when a card is turned into a project (`gateway/projects.py`); a direct Main turn is named in the background once it starts working (`spawn_turn_namer`, bounded by `OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC` + 30 s) |
| OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC | 120 | Update-letter LIGHT one-shot ceiling, slot wait and provider call together (`update_letter.py`) |
| OUROBOROS_FALLBACK_COOLDOWN_ENABLED | true | 429-aware per-process model cooldown |
| OUROBOROS_FALLBACK_COOLDOWN_SEC | 120 | Cooldown window |
| OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL | 1 | Attempts per model in the fallback walk |
| OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS | 900000 | Owner CEILING (chars) on the native review inspection episode transcript; the effective bound is the reviewer window's calibrated capacity, never above this — except that a surface's declared mandatory reading (the advisory's five governance documents) lifts it up to the window, disclosed as `native_mandatory_read_exceeds_bound` when even the window cannot hold it. No round cap exists (the retired round-cap key: §11.4): exhaustion is a typed fail-closed refusal for verdict shapes and a disclosed incomplete product for the report shape, never a silent truncation |
| OUROBOROS_MODEL_DEEP_SELF_REVIEW | openai/gpt-5.6-sol | Deep self-review model key — the invisible migration source and fallback for the optional `deep_review` reviewer row: with no row saved, `deep_review_slot()` synthesizes the packed api row from it (the historical delivery); a saved row wins and the key is not read — so the provider-default migrations of this key (`server_runtime.py`) reach only installs that still synthesize from it; a row-configured install keeps its row, by design. Not a Settings UI field any more — the row lives in Agents → Review lanes |
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
| TOTAL_BUDGET | 200.0 | Global budget (USD) |
| OUROBOROS_PER_TASK_COST_USD | 50.0 | Per-task cost cap; also the tree ceiling basis (`task_pacing.py`): the root resolves min(global share, cap minus margin), enabled descendants retain that original threshold while actual monetary admission remains independent, and the wrap-up affordability rail soft-lands under it |
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
| OUROBOROS_PRESENTATION | (unset) | Env-only: launcher-exported presentation (`desktop_window`/`browser_fallback`; absent renders `web`) |
| OUROBOROS_USER_FILES_ROOT | "" (home) | Env-only: user_files jail root (empty = `$HOME`) |
| OUROBOROS_OBSERVABILITY_KEEP_RAW | unset | Env-only: truthy enables raw observability payload persistence |
| OUROBOROS_GENERATIVE_PROBE | 1 (on) | Generative-write probe toggle |
| OUROBOROS_GENERATIVE_PROBE_CHARS | 5000000 | Generative-probe size companion |
| OUROBOROS_REVIEWER_SLOTS | (empty) | Structured reviewer-slot SSOT (`reviewer_slot_config.py`): JSON `{triad[], scope[], advisory, deep_review?}`; each row is EITHER an inline route `{slot_id, route:{kind: api_chat\|agent_session, target_id}, effort}` OR a roster reference `{slot_id, subagent_id, effort}` — mutually exclusive (a row naming both refuses typed; the reference materializes route/effort from the Available-subagents roster at load time, an explicit row `effort` winning over the roster row's) — with a STABLE owner-assigned slot_id (never an array index); an `agent_session` or managed-model `api_chat` route may add the optional `route.profile_id` credential pin (empty = account rotation; direct API-key routes reject account pins); the optional `deep_review` singleton carries the same row keys minus `slot_id` (fixed `deep_review_slot_1`) and, absent, is synthesized as the packed api row from `OUROBOROS_MODEL_DEEP_SELF_REVIEW`. Empty = the shipped default panel; the retired comma keys and route envs are stripped at load and never read as configuration (§11.4). Malformed value refuses typed at save AND at review time on every surface, task acceptance included (owner R3); env-apply logs and leaves legacy keys unprojected. The save that FIRST gives the triad a retrieving row (agent session or configured-subagent native inspection) returns the one-time R12 migration disclosure in the save response's `warnings` — the rows by id and target, and the measured API packet-panel cost it replaces (≈12 s / ≈$0.07 per model row per task, median of the 2026-09-01 OSWorld traces; ≈75 s / ≈$0.82 for a three-row panel on ProgramBench) against minutes of subscription window per task for a session row; a later save that keeps a retrieving triad is silent, and so — by design — is the reverse transition back to a packet-only triad: R12 is a one-time migration notice, not a routing monitor. The onboarding ladder footnote states the same numbers. |
| OUROBOROS_SUBSCRIPTION_PRESET_VERSION | (empty) | One-shot install-preset marker; endpoint-authored, DISK-ONLY (`ENDPOINT_AUTHORED_SETTINGS`) — its absence authorizes nothing, which is why install time is proved by three facts (§2) |
| OUROBOROS_SUBAGENT_PRESET_RECEIPT | (empty) | Install-preset receipt; endpoint-authored, disk-only |
| OUROBOROS_ONBOARDING_COMPLETED_AT | (empty) | Durable completion fact; endpoint-authored, disk-only |
| OUROBOROS_TASK_REVIEW_MODE | auto | Task acceptance-review mode |
| OUROBOROS_SAFETY_MODE | full | Safety supervision mode (shipped default `full`; a fresh desktop wizard may author `light`); lowering is owner-guarded (the runtime-mode boundary: §6 Safety and runtime mode) |
| OUROBOROS_SAFETY_MAX_TOKENS | 2000 | Safety-check output budget |
| OUROBOROS_SAFETY_CALL_TIMEOUT_SEC | 60 | Safety-check call ceiling |
| OUROBOROS_WEBSEARCH_TIMEOUT_SEC | 480 | web_search ceiling |
| OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC | 2 | Seconds the chat lane waits for a stopped direct-chat turn to reach its next round boundary after the typed `finalize_now` control is armed; past it the outcome is `live` and the supervisor sweep retries custody rather than publishing a terminal (clamped 0-10) |
| OUROBOROS_ONBOARDING_SNAPSHOT_TIMEOUT_SEC | 45 | Bound on the onboarding transaction's settings snapshot read, so a wedged filesystem refuses the step instead of hanging the wizard |
| OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC | 30 | Bound on acquiring the settings-document lock for an owner read-modify-write; a timeout REFUSES before the transform runs — the lock is a precondition of the write, never a hint. The same bound also caps the initiating writer (`_run_settings_writer`: one lock wait plus one held episode; the generic save, the owner endpoints and onboarding completion alike): past it the Save answers 503 `settings_save_timeout` with `saved: null` and the body is left to its thread |
| OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC | 2700 | LLM transport read timeout |
| OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC | 300 | plan_task deadline floor |
| OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC | 200 | The floor: the minimum spendable seconds above the finalization reserve required to START an acceptance panel; never below 200 s (a smaller value is raised to it, a larger one wins). The improvement window is this floor ×2 under the adaptive improvement policy and ×1 otherwise. A spendable window at or below the applicable line is refused `review_skipped_deadline_reserve` / `improvement_window_inside_reserve`; the pacing semantics (owner R52/R23) are §6 Task lifecycle. |
| OUROBOROS_REVIEW_MAX_CYCLES | "2" | Shared paid review-cycle cap across the plan/acceptance/commit/skill gates (`unlimited` = no local count cap; per-gate semantics: `review_cycles.py` docstring, §6 Review stack) |
| OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES | (retired) | Retired alias: a stored value is MIGRATED into `OUROBOROS_REVIEW_MAX_CYCLES` (passes + 1) at settings load; a leftover env value is inert |
| OUROBOROS_ACCEPTANCE_RESERVE_PCT | 5 | Acceptance budget reserve percentage |
| OUROBOROS_OBSERVABILITY_RETENTION_DAYS | (retired) | Retired in 7.0 (`RETIRED_SETTING_KEYS`): observability rows are preserved indefinitely and the reader never deletes, so a retention knob had no reader; a stored value is stripped at settings load and a leftover env value is inert |
| OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC | (unset) | Env-only: logical review timeout (absent = route-owned behavior; late in-flight results stay in custody) |
| OUROBOROS_REVIEW_MAX_TOKENS | 65536 | Env-only: reviewer output budget, clamped to the 8192 floor |
| OUROBOROS_REVIEW_ENFORCEMENT | advisory | Review enforcement: advisory/blocking (closed enum; anything else coerces to the default) |
| OUROBOROS_PREFLIGHT_TIMEOUT_SEC | 1800 | Env-only: TOTAL wall-clock budget for the hermetic pre-commit pytest preflight (node lane + both passes; teardown + containment semantics in `preflight_runner.py`/`process_containment.py`) |
| OUROBOROS_PREFLIGHT_SERIAL | unset | Env-only: `1` selects one serial pytest pass; scrubbed from the candidate environment |
| OUROBOROS_PREFLIGHT_TEST_WORKERS | (unset) | Env-only: xdist worker count for the hermetic parallel pass; floor 2, otherwise `os.cpu_count()`. Read from the OPERATOR environment and scrubbed from the candidate; concurrent-lane sizing rule in `docs/DEVELOPMENT.md` |
| OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS | true | Auto-grant manifest-declared permissions to cleanly reviewed skills (hash-bound; blocking findings never grant) |
| OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS | true | Launcher seed/resync writes hash-pinned `native_seed` verdicts; acts only at seed/resync, no runtime grant endpoint |
| OUROBOROS_CONTEXT_MODE | max | Context mode (`nano`/`low`/`max`), owner-selected outside Cyber Pro; `nano` is the compact owner projection and is recorded as `owner_nano`/`rendered_mode=nano` in physical usage facts. Current scope applicability follows BIBLE P1/P3. Cyber may configure it through the same audited writer. |
| OUROBOROS_CONTEXT_MODE_AUTO_LOW | false | Task-local low-mode overflow retry toggle |
| OUROBOROS_RUNTIME_MODE | advanced | Effective Access light/advanced/pro/cyber_pro; ordinary self-modification boundaries and Cyber agency are defined in §6 Safety and runtime mode. Settings persist the next-boot value; configured review enforcement remains independent evidence. |
| OUROBOROS_SKILLS_REPO_PATH | "" | Extra skills checkout path (expanded at read time, never cloned/pulled) |
| MCP_ENABLED | false | MCP client toggle (§6 MCP) |
| MCP_SERVERS | [] | MCP server list (HTTP/SSE via URL/auth, stdio via command+args and optional cwd/literal/settings-backed env); persisted in settings, never env-exported |
| MCP_TOOL_TIMEOUT_SEC | 60 | Per-MCP-tool timeout |
| OUROBOROS_HUB_CATALOG_URL | `https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json` | OuroborosHub catalog URL (automatic fetch limited to catalog JSON; installs verify SHA-256) |
| OUROBOROS_CLAWHUB_REGISTRY_URL | `https://clawhub.ai/api/v1` | ClawHub registry URL |
| OUROBOROS_PROMPT_CACHE_TTL | 1h | Prompt-cache tier (default/5m/1h). The policy acts at the final send-time wire boundary so it can legalize provider ordering without prompt builders creating provider-specific TTL policy; `review_helpers.cached_prompt_blocks` and `usage_accounting._reservation_cost` also consult it; usage records the applied tier |
| OUROBOROS_EFFORT_TASK | medium | Task reasoning effort (scale none/minimal/low/medium/high/xhigh/max/ultra; Settings exposes all but `minimal`); provider adaptation is exact-route, success-confirmed, disclosed in `request_wire` |
| OUROBOROS_EFFORT_EVOLUTION | high | Evolution effort |
| OUROBOROS_EFFORT_REVIEW | high | Review effort; reaches plan review as every row's default rung unless the envelope declares `reviewer_effort` |
| OUROBOROS_EFFORT_SCOPE_REVIEW | high | Scope-review effort |
| OUROBOROS_EFFORT_DEEP_SELF_REVIEW | high | Deep-self-review effort — the surface default; a saved `deep_review` row's own effort outranks it |
| OUROBOROS_EFFORT_CONSCIOUSNESS | (empty) | Consciousness effort; empty = the Task / Chat effort (a wake-up is an ordinary Main turn), a set value is honored |
| OUROBOROS_RETURN_REASONING | true | Ask OpenRouter to return reasoning; direct/local request copies strip OpenRouter-only fields |
| OUROBOROS_REASONING_SUMMARY | auto | Readable reasoning-summary rendering; presentation-only, never added to history or returned to providers |
| OUROBOROS_TASK_IDLE_TIMEOUT_SEC | 900 | Idle timeout — requires absence of real task/subtree progress; the typed in-flight main-LLM row spares only this rail; a settled child result stamps parent progress, because delivery creates immediate integration work and must not coincide with idle termination |
| OUROBOROS_TASK_ABS_CEILING_SEC | 21600 | Absolute task ceiling, activity-independent; deadline and budget stay separate hard axes |
| OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC | 90 | Supervisor/direct-turn liveness watchdog — alerts and recommends `/restart` but never frees an in-process lock held by a genuinely wedged turn |
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
| OUROBOROS_TRANSIENT_RETRY_MAX | 6 | Same-model transient retry budget; pre-dispatch `transport_unavailable` is a separate task-bounded outer wait episode, because no provider attempt was admitted on that route |
| OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC | 1800 | Skill lifecycle-lane timeout |
| OUROBOROS_CLAUDEXOR_HARNESS_INSTALL_TIMEOUT_SEC | 300 | Harness install ceiling (kills the tracked group, typed refusal) |
| OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC | 90 | Quota-refresh POST ceiling (clamped 1–90) |
| OUROBOROS_BUNDLE_DIR | (unset) | Env-only: launcher-owned bundle root propagated to embedded children for Node/ripgrep discovery |
| OUROBOROS_BG_WAKEUP_MIN | 900 | Lower bound (s) of the interval between consciousness wake-ups; the model picks the interval itself (`set_next_wakeup`) and it is clamped into [min, max]. Read at each alarm decision, so a change applies without a restart |
| OUROBOROS_BG_WAKEUP_MAX | 14400 | Upper bound (s) of that same model-chosen interval; never below the lower bound. When the model chooses no interval the alarm uses `runtime_limits.WAKE_DEFAULT_SEC` (3300 s = 55 min, just under the default 1 h `OUROBOROS_PROMPT_CACHE_TTL` so the shared prefix stays warm on TTL-metered routes), a constant rather than a third knob |
| OUROBOROS_CONSCIOUSNESS_AUTONOMY | act | What a consciousness wake may do (closed enum, anything else falls back to the default): `observe` thinks, keeps memory and knowledge and writes to the owner but starts nothing; `act` adds everything the runtime mode allows except editing Ouroboros's own code and prompts, evolution, restart and settings; `full` is everything the runtime mode allows, evolution included |
| OUROBOROS_CONSCIOUSNESS_DAILY_USD | 20.0 | Rolling-24h spend ceiling for consciousness — its wakes plus the tasks they start. Exhausted means no NEW wake or task until spend leaves the window; `0` is a real choice (consciousness may not spend) |
| OUROBOROS_CONSCIOUSNESS_MAX_TASKS | 2 | How many consciousness-started tasks may run at once; `0` = it never starts tasks |
| OUROBOROS_POST_TASK_EVOLUTION | false | Post-task evolution promotion toggle; agent self-enablement is blocked at the shell/browser/settings/data-write guards; choosing an objective routes through the Main slot because it is a high-leverage decision, while execution stays behind ordinary review and owner gates |
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

Direct-provider review fallback (legacy name: OpenAI-only review fallback): when exactly one official direct provider is configured, `config.get_review_models()` compiles that provider's declarative reviewer-role sequence using provider-prefixed model IDs. Current scope covers official OpenAI, Anthropic, MiniMax, DeepSeek, Cloud.ru, and GigaChat; OpenRouter, legacy-base, OpenAI-compatible, and mixed-provider configurations stay outside it. OpenAI, Anthropic, and DeepSeek run three independent Main-model slots; MiniMax mixes Main/Light; Cloud.ru and GigaChat use their one role model for every slot. `_exclusive_direct_remote_provider_env` returns empty when OpenRouter, legacy `OPENAI_BASE_URL`, OpenAI-compatible keys, or multiple official direct providers are present, and the fallback requires `provider_models.migrate_model_value` to make the main model already start with the exclusive provider prefix — exact prefix checking prevents an arbitrary free-text model from silently entering a single-provider route. This is part of the single-provider independence invariant (docs/DEVELOPMENT.md "Provider Independence").

DeepSeek provider specifics (`deepseek::`): the official OpenAI-compatible endpoint is a fixed module constant (`provider_models.DEEPSEEK_BASE_URL`); a proxy or mirror belongs to the generic `openai-compatible::` route. The canonical reasoning scale is projected onto the provider's wire dialect at the send boundary (`minimal`→`low`, `medium`/`xhigh`→`high`, `ultra`→`max`, `none`→`extra_body.thinking.type=disabled`; native tiers pass through), a forced tool choice (`required` or a named tool) is served with thinking disabled because thinking mode accepts only `auto`/`none` (probed 2026-09-03), and every tier-changing projection is disclosed on usage as `reasoning_effort_clamped`; `reasoning_content` stays on canonical assistant turns for strict v4 tool replay with an explicit empty string for turns produced without provider reasoning, while other lanes strip the field and cross-family switches scrub it. System/assistant/tool content arrays are flattened to strings in the send copy only (the API accepts arrays on user turns alone); canonical block history is untouched. Prompt caching is automatic and cost remains nullable when no exact provider catalog is available. The 1M context claim is admitted only through route-fingerprinted capability evidence or owner acknowledgement. Slash-form `deepseek/...` remains OpenRouter; only `deepseek::...` selects the direct route. MiniMax (`minimax::`) is the mirror-image reasoning contract: the direct lane sends `extra_body.reasoning_split=true` so thinking comes back as `reasoning_details` instead of a `<think>` block inside `content`, and those records are replayed unchanged on the same lane for interleaved thinking while cross-family switches scrub them like any other provider-private artifact.

GigaChat provider specifics (`gigachat::`): routed through the native `gigachat` library, not OpenAI-compatible (`llm.py::_chat_gigachat`). OpenAI `tools` map to GigaChat `functions`; at most ONE `function_call` returns per turn, so parallel `tool_calls` collapse to the first; role `tool` results become role `function` and must be valid JSON (plain text wrapped as `{"result": ...}`); the `system` message must be first, so later system-reminders demote to `user`. `reasoning_effort` is deliberately omitted — hidden reasoning can consume the whole output budget and return empty content. GigaChat exposes no automatic live cost source, so cost stays nullable/unknown rather than a hand-maintained tariff. GigaChat models sit below the 1M scope-review floor; when no ≥1M reviewer is configured, the declared alternatives are the owner-selected `low` context mode (whole-repo scope review declaredly not performed; each commit records a typed `skipped_low_context_mode` evidence row) or an owner-selected retrieving scope slot at ≥200K sourced evidence (BIBLE P3); the blocking triad still reviews the full staged diff in both modes.

---

