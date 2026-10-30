/** Dependency-free JSDoc mirror of `ouroboros.gateway.client_contracts` — the replaceable-client envelopes
 *  `api_types.js` indexes: the served main interface, the setup bootstrap, the settings writers' refusal,
 *  completion readback, the auth gate's JSON login and the `GET /api/contract` document. */
/**
 * One completion attempt read back: saved is true while the settings document holds its write, false when
 * refused before writing, null when unconfirmed. Only `receipt` (its own success envelope) completes.
 * @typedef {Object} OnboardingAttemptResponse
 * @property {?boolean} saved
 * @property {import('./api_types.js').OnboardingCompleteResponse=} receipt
 * @property {string=} error
 * @property {string=} code
 * @property {string=} post_commit_failed
 */
/**
 * @typedef {Object} ChatInbound
 * @property {string=} reply_to
 * @property {"chat"} type
 * @property {string} content
 * @property {string=} sender_session_id
 * @property {string=} client_message_id
 * @property {boolean=} force_plan
 * @property {Array<Object>=} attachments  // [{filename, display_name, mime}] — image uploads become native blocks (v6.26.0)
 * @property {number=} chat_id     // multi-project thread routing (v6.32.0); main chat = 1
 * @property {string=} project_id  // per-project memory scope (v6.32.0)
 * @property {Object=} client_surface  // raw sending-surface observables measured at send time (pywebview/ua/viewport/matchMedia/captured_at; optional IANA timezone)
 */
/**
 * @typedef {Object} CommandInbound
 * @property {Object=} client_surface
 * @property {"command"} type
 * @property {string} cmd
 */
/**
 * Emergency request accepted on this connection; does not prove process exit.
 * @typedef {Object} PanicAcceptedOutbound
 * @property {"panic_accepted"} type
 */
/**
 * The main interface THIS server generation serves at `/` and `/static`: the applied selection (fixed until
 * the core restarts), not the requested value and not an asset digest.
 * @typedef {Object} ServedUiState
 * @property {'builtin'|'directory'|'none'} source
 * @property {string=} error  // why a requested directory was not applied (the builtin interface serves)
 */

/**
 * @typedef {Object} SetupStep
 * @property {string} id
 * @property {string} title
 * @property {string} railCopy
 * @property {string} copy
 * @property {string} footer
 */

/**
 * One credential or endpoint input; `settingKey` is the settings key it saves into.
 * @typedef {Object} SetupProviderField
 * @property {string} id
 * @property {string} stateKey
 * @property {string} settingKey
 * @property {string} settingsInputId
 * @property {string} label
 * @property {string} placeholder
 * @property {string} note
 * @property {string} inputType
 * @property {'primary'|'more'} group
 */

/**
 * @typedef {Object} SetupProviderProfile
 * @property {string} label
 * @property {string} providerCopy
 * @property {string} modelCopy
 */

/**
 * @typedef {Object} SetupModelSlot
 * @property {string} slot
 * @property {string} stateKey
 * @property {string} settingKey
 * @property {string} inputId
 * @property {string} label
 * @property {string} note
 * @property {string} settingsInputId
 * @property {string} settingsToggleId
 */

/**
 * A `reviewModes` or `runtimeModes` row; `value` is what the setting stores.
 * @typedef {Object} SetupModeChoice
 * @property {string} value
 * @property {string} label
 * @property {string} tone
 * @property {string} className
 * @property {string} copy
 */

/**
 * @typedef {Object} SetupLocalRoutingMode
 * @property {string} value
 * @property {string} buttonLabel
 * @property {string} label
 * @property {boolean[]} flags  // USE_LOCAL_MAIN, USE_LOCAL_LIGHT, USE_LOCAL_CONSCIOUSNESS, USE_LOCAL_FALLBACK
 */

/**
 * @typedef {Object} SetupBudgetField
 * @property {string} stateKey
 * @property {string} settingKey
 * @property {string} inputId
 * @property {string} settingsInputId
 * @property {string} title
 * @property {string} label
 * @property {string} note
 * @property {number} default
 * @property {string} min
 * @property {string} step
 */

/**
 * A wizard declaration (not a setting) sent with `POST /api/onboarding/complete`.
 * @typedef {Object} SetupSubscriptionField
 * @property {string} id
 * @property {string} payloadKey
 * @property {string} label
 * @property {string} note
 */

/**
 * @typedef {Object} SetupContract
 * @property {number} version
 * @property {'web'|'desktop'} hostMode
 * @property {SetupStep[]} steps
 * @property {SetupProviderField[]} providerFields
 * @property {Object.<string,SetupProviderProfile>} providerProfiles
 * @property {SetupModelSlot[]} modelSlots
 * @property {SetupModeChoice[]} reviewModes
 * @property {SetupModeChoice[]} runtimeModes
 * @property {SetupLocalRoutingMode[]} localRoutingModes
 * @property {SetupBudgetField[]} budgetFields
 * @property {SetupSubscriptionField[]} subscriptionFields
 */

/**
 * @typedef {Object} SetupLocalPreset
 * @property {string} label
 * @property {string} source
 * @property {string} filename
 * @property {number} contextLength
 * @property {string} chatFormat
 */

/**
 * The wizard's starting values; besides these keys it carries one key per `contract.providerFields[].stateKey`
 * (a configured secret reads as `secretPlaceholder`), `contract.budgetFields[].stateKey` and `contract.modelSlots[].stateKey`.
 * @typedef {Object} SetupInitialState
 * @property {Object[]} connections Masked OUROBOROS_CONNECTIONS rows, keyed by kind/name.
 * @property {string} safetyModel
 * @property {string} providerProfile
 * @property {string} reviewEnforcement
 * @property {string} runtimeMode
 * @property {string} skillsRepoPath
 * @property {string} localPreset
 * @property {string} localSource
 * @property {string} localFilename
 * @property {number} localContextLength
 * @property {number} localGpuLayers
 * @property {string} localChatFormat
 * @property {string} localRoutingMode
 * @property {Object} modelAccounts
 * @property {Object} modelContextWindows
 * @property {string} processingPreference
 * @property {Object} modelProcessingPreferences
 */

/**
 * `settings_setup_contract.build_setup_bootstrap`: what the builtin wizard page embeds.
 * @typedef {Object} SetupBootstrap
 * @property {'web'|'desktop'} hostMode
 * @property {boolean} freshInstall  // display provenance only; completion re-proves install eligibility
 * @property {boolean} supportsLocalRuntimeControls
 * @property {string[]} stepOrder
 * @property {Object.<string,Object.<string,string>>} modelDefaults  // provider profile -> slot -> model id
 * @property {Object.<string,SetupLocalPreset>} localPresets
 * @property {string[]} modelSuggestions
 * @property {string} secretPlaceholder  // a prefilled credential that is configured holds this marker
 * @property {SetupContract} contract
 * @property {SetupInitialState} initialState
 */

/**
 * `GET /api/onboarding/bootstrap`: a read that writes nothing.
 * @typedef {Object} OnboardingBootstrapResponse
 * @property {boolean} ready
 * @property {SetupBootstrap} bootstrap
 */

/**
 * `GET /api/settings`: every effective setting with its (masked) value, plus `_meta`.
 * @typedef {Object} SettingsReadResponse
 * @property {import('./api_types.js').SettingsMeta=} _meta
 */

/**
 * A settings-document writer's failure: `saved` false = nothing written, true = bytes landed and
 * `post_commit_failed` names the later step, null = unknown (re-read before deciding).
 * @typedef {Object} SettingsWriteRefusal
 * @property {string} error
 * @property {?boolean} saved
 * @property {string=} code
 * @property {string=} post_commit_failed
 */

/**
 * `POST /auth/login` as JSON.
 * @typedef {Object} AuthLoginRequest
 * @property {string} password
 * @property {string=} next  // a same-origin path to return to; anything else becomes "/"
 */

/**
 * @typedef {Object} AuthLoginResponse
 * @property {boolean} ok
 * @property {string} next
 */

/**
 * @typedef {Object} ContractBody
 * @property {string} content_type
 * @property {string[]} schemas
 */

/**
 * @typedef {Object} ContractHttpEntry
 * @property {string} method
 * @property {string} path
 * @property {string[]} path_params
 * @property {'gate'|'public'|'asset_lease'} auth
 * @property {boolean} typed  // false: published without shapes (an explicit gap)
 * @property {string[]=} query
 * @property {string[]=} headers  // request headers the handler reads; a trailing "!" marks a required one
 * @property {ContractBody=} request
 * @property {Object.<string,ContractBody>=} responses  // by HTTP status
 * @property {string=} note
 */

/**
 * @typedef {Object} ContractWsEntry
 * @property {string} type
 * @property {string=} inbound
 * @property {string=} outbound
 */

/**
 * @typedef {Object} ContractWs
 * @property {string} path
 * @property {ContractWsEntry[]} messages
 * @property {string} extension_namespace
 */

/**
 * `GET /api/contract`: the gateway ABI derived from the endpoint and envelope indexes.
 * @typedef {Object} GatewayContractResponse
 * @property {string} abi_version
 * @property {string} runtime_version
 * @property {string[]} core_paths
 * @property {string[]} interface_paths
 * @property {ContractHttpEntry[]} http
 * @property {ContractWs} ws
 * @property {CallSocketContract=} call_socket
 * @property {Object.<string,Object>} schemas
 */

/**
 * Where a stored history row lies: its stream and the row's byte offset in that stream's retained
 * chain. It names a stored record, never current task or review authority.
 * @typedef {Object} ChatHistoryPosition
 * @property {'chat'|'progress'} source
 * @property {number} offset
 */

/**
 * Host-retained source of a private Main contact; external context, never owner authority.
 * @typedef {Object} PresenceContact
 * @property {string} task_id
 * @property {string} binding_id
 * @property {string} source_event_id
 * @property {string} label
 * @property {string} conversation_key
 * @property {string=} transport_skill
 * @property {string=} behavior_skill
 * @property {string=} profile_fingerprint
 * @property {string=} state_fingerprint
 * @property {string=} selection_fingerprint
 * @property {string=} provider
 * @property {string=} account_id
 * @property {string=} conversation_id
 * @property {string=} thread_id
 * @property {string=} actor_id
 * @property {string=} place_id
 * @property {string=} url
 */

/**
 * The question card of a `msg_type: "quiz"` history row (the question itself is the row's `text`).
 * `state`: open | answered | closed | expired_terminal | superseded, merged at read time.
 * @typedef {Object} ChatHistoryQuiz
 * @property {PresenceContact=} presence_contact
 * @property {string} quiz_id
 * @property {Object[]=} options  // QuizOption rows {label, detail?, recommended?}
 * @property {string=} stake
 * @property {string=} assumption
 * @property {string=} state
 * @property {boolean=} wait_for_answer
 * @property {string=} wait_ended_at
 * @property {string=} host_facts
 * @property {number=} answered_index
 * @property {string=} comment
 * @property {string=} owner_wait_state
 * @property {string=} owner_wait_resume_reason
 */

/**
 * The living card of a `msg_type: "card"` history row (its plain `fallback_text` is the row's `text`): its
 * identity as first published. Read the current body, revision and presses from `GET /api/cards/{card_id}`.
 * @typedef {Object} ChatHistoryCard
 * @property {string} card_id
 * @property {'components'|'widget'|'app'} kind
 * @property {string} title
 * @property {number} revision
 */

/**
 * A `reply_to` reference as the host resolved it; an unresolved one is `{address, unresolved: true}`.
 * @typedef {Object} ChatReplyReference
 * @property {string=} address
 * @property {number=} chat_id
 * @property {'user'|'assistant'|'system'=} role
 * @property {string=} author_label
 * @property {string=} kind
 * @property {string=} snippet
 * @property {string=} ts
 * @property {string=} task_id
 * @property {'work'|'room'|'none'=} target_home  // what decides the quoted row's room; an older reference lacks it
 * @property {boolean=} unresolved
 */

/**
 * One `GET /api/chat/history` row (live frames call `text` `content`). A row may also carry the
 * task-progress and outcome fields of a live `chat` frame (`ChatOutbound`). Replay-evidence rows
 * (`quiz_answer`, `quiz_closed`, `task_evidence`, or `summary_kind` + `historical_terminal`) are never shown.
 * @typedef {Object} ChatHistoryMessage
 * @property {PlaceHistoryFacts=} place_history
 * @property {PresenceContact=} presence_contact
 * @property {string=} wait_ended_at
 * @property {string=} row_address
 * @property {string=} row_client_message_id  // an outgoing row's own saved name, the same as its client_message_id
 * @property {ChatReplyReference=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {Object.<string,number>=} room_revisions  // a row shown in several rooms: the revision it took in each room it counted in
 * @property {string} text
 * @property {'user'|'assistant'|'system'} role
 * @property {string} ts  // the stored row's time: the same string its live frame carried
 * @property {boolean} is_progress
 * @property {string} system_type
 * @property {string} task_id
 * @property {string=} history_id  // "<source>:<offset>"
 * @property {ChatHistoryPosition=} history_position
 * @property {boolean=} markdown
 * @property {string=} source
 * @property {string=} sender_label
 * @property {string=} sender_session_id
 * @property {string=} client_message_id  // correlation id, not proof of owner authorship; use role
 * @property {"swarm"=} subagents_mode  // effective mode at acceptance; preserved on replay
 * @property {boolean=} ingress_accepted
 * @property {boolean=} ingress_dispatched
 * @property {boolean=} ingress_pending
 * @property {Object[]=} attachments  // ChatAttachmentView
 * @property {boolean=} text_placeholder
 * @property {Object=} chat_annotation
 * @property {string=} origin_id
 * @property {boolean=} origin_projected
 * @property {('document'|'photo'|'video'|'links'|'quiz'|'card')=} msg_type
 * @property {string=} filename
 * @property {string=} mime
 * @property {string=} download_url
 * @property {string=} download_url_compat
 * @property {string=} caption
 * @property {number=} size_bytes
 * @property {Object[]=} actions  // LinkAction rows {label, url}
 * @property {string=} title
 * @property {ChatHistoryQuiz=} quiz
 * @property {ChatHistoryCard=} card
 * @property {string=} summary_kind
 * @property {Object=} historical_terminal
 * @property {Object=} model_waits
 * @property {boolean=} lifecycle_virtual
 * @property {Object=} review_group
 * @property {Object[]=} executions
 * @property {string=} surface
 * @property {string=} review_fingerprint
 * @property {string=} state_revision
 * @property {string=} presentation_owner_task_id
 * @property {string=} group_id
 * @property {string=} skill
 * @property {string=} content_hash
 * @property {string=} job_id
 * @property {string=} job_status
 * @property {string=} terminal_reason
 * @property {string=} origin_task_id
 * @property {string=} origin_root_task_id
 * @property {string=} replayed_from_ts
 * @property {number=} review_round
 * @property {number=} snapshot_attempt
 * @property {boolean=} snapshot_revised
 * @property {Object=} history_retention
 * @property {?number=} tool_errors
 * @property {?number=} routing_tool_calls
 * @property {?number=} completion_tool_calls
 * @property {?Object=} tool_call_counts
 * @property {string=} addressing_only
 * @property {string=} outcome_phase
 * @property {boolean=} outcome_final
 * @property {string=} checkpoint_kind
 * @property {string=} episode_id
 * @property {string=} phase
 * @property {*=} elapsed_sec
 * @property {Object=} tool_evidence
 * @property {Object=} cancel_origin
 * @property {Object=} continuation_offer
 */

/**
 * Additive /api/chat/history row fields (v6.73.0 Project origin projection).
 * A Project thread may synthesize its start message from the binding's own
 * durable copy when the canonical row left the bounded read window:
 * `origin_projected: true` marks such a synthesized user row (normal history
 * shape otherwise), and a `system_type: "origin_omission"` system row discloses
 * origins omitted past the synthesis cap. Both fields are additive and safely
 * ignorable by renderers.
 * @typedef {Object} ProjectOriginHistoryFields
 * @property {boolean=} origin_projected
 * @property {"origin_omission"=} system_type
 */

/** Additive history fact: a project-owned Main mirror cannot grant cancel authority.
 * @typedef {Object} ProjectMirrorHistoryFields
 * @property {boolean=} project_mirror
 */

/**
 * Additive /api/chat/history row fields on `system_type: "skill_review"` rows:
 * the exact-job reference the producer already writes into chat.jsonl. A row
 * carrying a non-empty `job_id` lets the Chat card lazily fetch the rendered
 * review via GET /api/skills/{skill}/review-history/{record_id or job_id} (the
 * review record when the row names one, else the job); rows without either
 * (legacy full-text rows) keep local expansion. All fields are additive and
 * safely ignorable by renderers.
 * @typedef {Object} SkillReviewHistoryRowFields
 * @property {string=} skill
 * @property {string=} status
 * @property {string=} content_hash
 * @property {string=} job_id
 * @property {string=} record_id
 * @property {number=} review_round
 * @property {number=} snapshot_attempt
 */

/**
 * `coverage` v1: delivered coordinates per stream; a span is `{from, to, chain, gaps}` or null. Stored
 * order counts bytes under chain `dialog`; a room with imported history counts chat rows in time order under
 * `dialog-chrono-<end>` (`-at-<position>` around a row) and adds `keys` (CLIENT_CONTRACT §5). Never proof of
 * the conversation's start.
 * @typedef {Object} ChatHistoryCoverage
 * @property {number} v
 * @property {string} view
 * @property {Object.<string,number>} upper
 * @property {Object.<string,?{from: number, to: number, chain: string, gaps: string[], keys: ({from: ?Array, to: ?Array, frozen: number}|undefined)}>} spans
 */

/**
 * Bounds of one history response. `latest_message` (Project rooms) names the latest eligible
 * non-owner standalone message by stored arrival, excluding card content; null means unknown.
 * `out_of_order`: it sorts above earlier arrivals or before the recent read. With a null message,
 * `latest_before` says the bounded search found none at or after that chat offset; older pages
 * continue the search. `latest_absent`: the search reached the chat's start without a gap and
 * found no eligible message below the frozen upper bound, not that the chat has no stored rows.
 * @typedef {Object} ChatHistoryWindow
 * @property {boolean} complete
 * @property {string[]} truncated_by
 * @property {?{history_id: string, out_of_order: boolean}=} latest_message
 * @property {number=} latest_before
 * @property {boolean=} latest_absent
 */

/**
 * `around`: a task card or an exact stored row. `found` binds `history_id`; a moved row names its
 * current `chat_id`. Only complete source misses prove absent; failures/imports answer 503.
 * @typedef {Object} ChatHistoryAnchor
 * @property {'found'|'absent'|'moved'|'ambiguous'|'unavailable'} state
 * @property {string=} task_id
 * @property {string=} row_address
 * @property {number=} chat_id
 * @property {string=} history_id
 */

/**
 * `GET /api/chat/history`: conversation rows oldest first; replay evidence may follow them
 * regardless of timestamp and recur across pages. An error answer keeps this shape: no rows,
 * `has_more` true and the cursor that was sent as `next_cursor`.
 * @typedef {Object} ChatHistoryResponse
 * @property {ChatHistoryMessage[]} messages
 * @property {boolean} has_more
 * @property {string=} next_before_ts  // legacy, no longer sent
 * @property {?string} next_cursor
 * @property {?string=} page_cursor
 * @property {string=} after_cursor  // the newest page and every `after` read: the rows stored since
 * @property {ChatHistoryWindow} window
 * @property {ChatHistoryCoverage=} coverage
 * @property {ChatHistoryAnchor=} anchor  // an `around` read only
 * @property {string=} error
 * @property {string=} reason_code
 */

// --- Journey bodies of routes whose handlers check fields by hand (ProjectEntry: ./api_types.js) ---

/**
 * `POST /api/owner/runtime-mode`; the mode applies when the core next starts.
 * @typedef {Object} OwnerRuntimeModeRequest
 * @property {'light'|'advanced'|'pro'|'cyber_pro'} mode
 */

/**
 * `POST /api/owner/context-mode`.
 * @typedef {Object} OwnerContextModeRequest
 * @property {'nano'|'low'|'max'} mode
 * @property {string=} project_id
 */

/**
 * @typedef {Object} OwnerEffortRangeRequest
 * @property {string} min
 * @property {string} recommended
 * @property {string} max
 * @property {string=} project_id
 */

/**
 * @typedef {Object} OwnerSubagentsModeRequest
 * @property {"off"|"auto"|"swarm"} mode
 * @property {string=} project_id
 */

/**
 * The room's model for new owner work: `main` or `fallback` (the configured reference) or `model`.
 * @typedef {Object} OwnerChatModelRequest
 * @property {"main"|"fallback"|"model"} kind
 * @property {string=} model  // kind "model" only: the raw route, its source included
 * @property {boolean=} use_local  // kind "model" only
 * @property {string=} project_id
 */
/**
 * A room's model for new owner work: the configured Main's or first Fallback's reference, or one raw route.
 * @typedef {Object} ChatModelChoice
 * @property {"main"|"fallback"|"model"} kind
 * @property {string=} model  // kind "model" only: the raw route, its source included
 * @property {boolean=} use_local  // kind "model" only
 */

/** @typedef {Object} ChatModelRoute
 * @property {string} model
 * @property {boolean} use_local
 */

/** What a room's two references resolve to now.
 * @typedef {Object} ChatModelRoutes
 * @property {ChatModelRoute} main
 * @property {?ChatModelRoute} fallback  // null: no Fallback configured, so its reference admits work on Main
 */

/**
 * @typedef {Object} OwnerChatModelResponse
 * @property {boolean} ok
 * @property {string} project_id
 * @property {ChatModelChoice} model_choice
 * @property {ChatModelChoice} previous_model_choice
 */

/**
 * @typedef {Object} OwnerSafetyModeRequest
 * @property {'your_work'|'consciousness'=} kind
 * @property {'full'|'light'|'off'} mode
 */

/**
 * `{enabled}` as a JSON boolean: `POST /api/owner/auto-grant`, `/api/desktop/autostart`,
 * `/api/desktop/background` and `PATCH /api/claudexor/credential-profiles/...`.
 * @typedef {Object} OwnerEnabledRequest
 * @property {boolean} enabled
 */

/**
 * One stored safety check, retaining both its chip and the tool it checked.
 * @typedef {Object} RecentWakeCheck
 * @property {string} chip
 * @property {string} tool
 */

/**
 * `GET /api/consciousness` recent_wakes: ledger-selected facts, never authored by the view.
 * CostPresentation is the task card's existing carrier (api_types.js).
 * @typedef {Object} RecentWake
 * @property {{label: string, owner: boolean}=} as
 * @property {string} task_id
 * @property {?number} chat_id
 * @property {'own'|'focus'} cause
 * @property {string} place_label
 * @property {string} conversation
 * @property {?string} started_at
 * @property {?string=} first_reply_at
 * @property {?string} ended_at
 * @property {boolean} live
 * @property {'running'|'waiting'|'queued'|'paused'|'unknown'|'working'|'done'|'warn'|'error'|'cancelled'} phase
 * @property {boolean} quiet
 * @property {string=} summary
 * @property {?string} note
 * @property {?number=} tool_calls
 * @property {?number=} rounds
 * @property {Array<RecentWakeCheck>} checks
 * @property {boolean=} said_owner
 * @property {?number=} accounted_upper_bound_usd
 * @property {?number=} accounted_upper_bound_usd_with_children
 * @property {?CostPresentation=} cost_presentation
 * @property {'available'|'unavailable'=} cost_accounting_status
 * @property {string=} cost_accounting_error
 * @property {boolean=} cost_final
 * @property {boolean=} cost_with_children_partial
 * @property {?CostCustody=} cost_custody
 * @property {?number=} unknown_unmetered
 * @property {?number=} non_final_rows
 * @property {?number=} reserved_usd
 * @property {?number=} unresolved_upper_bound_usd
 * @property {?boolean=} ledger_integrity_degraded
 */

/** Recent-wakes fields of the open consciousness panel.
 * @typedef {Object} RecentWakesView
 * @property {Array<RecentWake>|null} recent_wakes
 * @property {string=} recent_wakes_coverage
 */

/** Places declarations live in a sibling to keep the client envelope module within its size band. */
/** @typedef {import('./place_contract_types.js').PlaceConnectRequest} PlaceConnectRequest */
/** @typedef {import('./place_contract_types.js').PlaceIcon} PlaceIcon */
/** @typedef {import('./place_contract_types.js').PlaceReadiness} PlaceReadiness */
/** @typedef {import('./place_contract_types.js').PlaceSummary} PlaceSummary */
/** @typedef {import('./place_contract_types.js').PlaceIdentityAccount} PlaceIdentityAccount */
/** @typedef {import('./place_contract_types.js').PlaceIdentity} PlaceIdentity */
/** @typedef {import('./place_contract_types.js').PlaceStatusPayload} PlaceStatusPayload */
/** @typedef {import('./place_contract_types.js').PlaceHistoryAttention} PlaceHistoryAttention */
/** @typedef {import('./place_contract_types.js').PlaceHistoryImport} PlaceHistoryImport */
/** @typedef {import('./place_contract_types.js').PlaceHistoryFacts} PlaceHistoryFacts */
/** @typedef {import('./place_contract_types.js').PlaceDisconnectRequest} PlaceDisconnectRequest */
/** @typedef {import('./place_contract_types.js').PlaceOwnerAccountRequest} PlaceOwnerAccountRequest */
/** @typedef {import('./place_contract_types.js').PlaceRoomSummary} PlaceRoomSummary */
/** @typedef {import('./place_contract_types.js').PlaceRoomsCoverage} PlaceRoomsCoverage */
/** @typedef {import('./place_contract_types.js').PlaceRoomsResponse} PlaceRoomsResponse */
/** @typedef {import('./place_contract_types.js').PlaceUpdateRequest} PlaceUpdateRequest */

/**
 * `POST /api/settings/secret`: exactly one selector, `{key}`, `{mcp_server_id}`, `{mcp_server_id,
 * header_name}` or `{connection, field}` (an added API key `kind@name` and one of its secret fields).
 * @typedef {Object} SettingsSecretRequest
 * @property {string=} key
 * @property {string=} mcp_server_id
 * @property {string=} header_name
 * @property {string=} connection
 * @property {string=} field
 */

/**
 * @typedef {Object} SettingsSecretResponse
 * @property {string} value
 */

/**
 * `DELETE /api/chat/upload`: only a pending upload (one no message accepted yet) is deleted.
 * @typedef {Object} ChatUploadDeleteRequest
 * @property {string} filename
 */

/**
 * @typedef {Object} ChatUploadDeleteResponse
 * @property {boolean} ok
 * @property {string} filename
 */

/**
 * `GET /api/projects`: active and deleting projects, at most 200, most recently active first.
 * @typedef {Object} ProjectListResponse
 * @property {import('./api_types.js').ProjectEntry[]} projects
 */

/**
 * `POST /api/projects`: `project` is the full registry row.
 * @typedef {Object} ProjectCreateResponse
 * @property {import('./api_types.js').ProjectEntry} project
 * @property {Array=} init_git_warnings
 * @property {string[]=} init_git_skipped
 */

/**
 * `POST /api/projects/{project_id}/update`: name, pin or closure; at least one.
 * `closed`: true closes as owner; false reopens.
 * @typedef {Object} ProjectUpdateRequest
 * @property {string=} name
 * @property {boolean=} pinned
 * @property {boolean=} closed
 */

/**
 * @typedef {Object} ProjectUpdateResponse
 * @property {?import('./api_types.js').ProjectEntry} project
 */

/**
 * `POST /api/projects/from-task`.
 * @typedef {Object} ProjectFromTaskRequest
 * @property {string} task_id
 * @property {string=} id
 * @property {string=} project_id
 * @property {string=} name
 * @property {string=} objective_hint
 */

/**
 * The task's immutable binding to its project.
 * @typedef {Object} ProjectTaskBinding
 * @property {string=} task_id
 * @property {string=} project_id
 * @property {number=} project_chat_id
 * @property {string=} bound_at
 * @property {Object=} source_ref
 * @property {*=} origin_absent
 * @property {string=} source_text
 */

/**
 * `handoff_id` is derived by the core from the project, the task and its origin message, so a repeat
 * for the same task answers the same id; it is not a caller key.
 * @typedef {Object} ProjectFromTaskResponse
 * @property {import('./api_types.js').ProjectEntry} project
 * @property {ProjectTaskBinding} binding
 * @property {boolean} adopted
 * @property {string} handoff_receipt
 * @property {string} handoff_id
 */

/**
 * One `GET /api/files/list` row; directories come first.
 * @typedef {Object} FileBrowserEntry
 * @property {string} name
 * @property {string} path
 * @property {'dir'|'file'} type
 * @property {boolean} is_symlink
 * @property {number=} size
 */

/**
 * `GET /api/files/list`: one directory of the file-browser root, at most 500 entries (`truncated`).
 * @typedef {Object} FileBrowserListResponse
 * @property {string} root_path
 * @property {string} path
 * @property {string} display_path
 * @property {?string} parent_path
 * @property {Array<Object.<string,string>>} breadcrumb
 * @property {FileBrowserEntry[]} entries
 * @property {boolean} truncated
 * @property {string} default_path
 * @property {string} default_display_path
 */

/**
 * `GET /api/files/read`: text is capped (256 KiB or 120,000 characters, `truncated`); an image
 * or PDF carries `media_type` and `content_url` instead.
 * @typedef {Object} FileBrowserReadResponse
 * @property {string} root_path
 * @property {string} path
 * @property {string} display_path
 * @property {string} name
 * @property {number} size
 * @property {boolean} is_text
 * @property {boolean} is_image
 * @property {boolean} is_pdf
 * @property {string} content
 * @property {boolean} truncated
 * @property {string=} media_type
 * @property {string=} content_url
 */

/**
 * @typedef {Object} FileMkdirRequest
 * @property {string} name
 * @property {string=} path
 */

/**
 * Writes unconditionally, so repeating it is safe.
 * @typedef {Object} FileWriteRequest
 * @property {string} path
 * @property {string} content
 * @property {boolean=} create
 */

/**
 * `POST /api/files/delete`: a directory is deleted recursively; a repeat answers 404.
 * @typedef {Object} FilePathRequest
 * @property {string} path
 */

/**
 * `POST /api/files/transfer`: a repeated copy answers 409 (the target exists), a repeated move 404
 * (the source is gone).
 * @typedef {Object} FileTransferRequest
 * @property {string} source_path
 * @property {string=} destination_dir
 * @property {('copy'|'move')=} mode
 */

/**
 * Answer of the file-browser writers: upload and write `{ok, path, display_path, name, size}` (write
 * adds `created` for a new file), mkdir `{..., name, type: "dir"}`, transfer `{ok, mode, path,
 * display_path, name, type}`, delete `{ok, path, type}`.
 * @typedef {Object} FileChangeResponse
 * @property {boolean} ok
 * @property {string} path
 * @property {string=} display_path
 * @property {string=} name
 * @property {number=} size
 * @property {string=} type
 * @property {('copy'|'move')=} mode
 * @property {boolean=} created
 */

/**
 * @typedef {Object} ScheduleTriggerRequest
 * @property {'cron'|'once'} type
 * @property {string=} expr
 * @property {string=} run_at
 */

/**
 * What a schedule starts: an ordinary task, never a workspace or drive field.
 * @typedef {Object} ScheduleTaskTemplate
 * @property {'task'=} type
 * @property {string=} text
 * @property {number=} priority
 * @property {Object=} metadata
 * @property {string=} project_id
 * @property {string=} reasoning_effort
 */

/**
 * `POST /api/schedules`. A new `id` creates, an existing one edits; without an `id` a repeat creates
 * another schedule. `cron` is accepted as an alias of `trigger.expr`.
 * @typedef {Object} ScheduleUpsertRequest
 * @property {string=} id
 * @property {string=} name
 * @property {string=} description
 * @property {boolean=} enabled
 * @property {string=} timezone
 * @property {ScheduleTriggerRequest} trigger
 * @property {ScheduleTaskTemplate=} task
 * @property {string=} reason
 */

/**
 * `POST /api/schedules/{schedule_id}/action`: a lifecycle action governs future runs and never stops a
 * run already admitted.
 * @typedef {Object} ScheduleActionRequest
 * @property {'disable'|'delete'|'restore'} action
 * @property {string} reason
 * @property {string=} expected_hold_id
 * @property {string=} relation
 */

/**
 * The optional body of `POST /api/tasks/{task_id}/resume`: held calls whose outcome is unknown
 * (a refusal's `acceptable_unknown`) the owner goes on past, and why. Recorded, never re-run.
 * @typedef {Object} TaskResumeRequest
 * @property {string[]=} accept_unknown
 * @property {string=} reason
 */

/**
 * `POST /api/tasks/{task_id}/resume` after budget, owner, sleep or technical authority pause.
 * @typedef {Object} TaskResumeResponse
 * @property {boolean} ok
 * @property {string} task_id
 * @property {string=} root_task_id
 * @property {('resumed'|'settled')=} late_phase
 * @property {boolean=} tree_only
 * @property {boolean=} owner_pause_released
 * @property {boolean=} snapshot_persisted
 * @property {boolean=} exact_continuation
 * @property {string=} grant_id
 * @property {number=} grant_generation
 * @property {number=} paused_duration_sec
 * @property {string[]=} eligible_descendants
 * @property {string[]=} held_siblings
 * @property {string[]=} rebound_held_siblings
 * @property {string=} released_hold
 * @property {{rows: Array<{task_id: string, reason: string}>, count: number}=} membership_gaps
 * @property {string=} message
 */

/* Living-message cards (`ouroboros.gateway.card_contracts`): the `card` / `card_state` frames, the read API
 * and the pin receipt. Presses go to `POST /api/decisions` with `card:{card_id}:{action_id}`. */

/**
 * One recorded owner press: `delivery` is `delivered` (the authoring task's mailbox), `forwarded` (the owner's
 * message after that task ended) or `owed`; `request_id` retries an owed press, after a reload too.
 * @typedef {Object} CardSubmissionSummary
 * @property {string} action_id
 * @property {string} label
 * @property {string} at
 * @property {'owed'|'delivered'|'forwarded'} delivery
 * @property {string} request_id
 * @property {string=} reason_code  // 'awaiting_recipient_read' until a paused, held or queued author reads it
 */

/**
 * A living card in Ouroboros's message; its history row is `msg_type: "card"` with the same `card`.
 * @typedef {Object} CardOutbound
 * @property {string=} row_address
 * @property {string=} row_client_message_id  // the saved row's own name, when it has one
 * @property {ChatReplyReference=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {string=} kind  // ChatReferenceFields' row kind; the card's own kind is card.kind
 * @property {Object.<string,number>=} room_revisions
 * @property {'card'} type
 * @property {'assistant'} role
 * @property {ChatHistoryCard} card
 * @property {string} fallback_text
 * @property {string} ts
 * @property {number=} chat_id
 * @property {string=} task_id
 * @property {boolean=} project_thread
 */

/**
 * A shown card changed (a new body: `card.revision` grew; or a recorded press): read `GET /api/cards/{card_id}`.
 * @typedef {Object} CardStateOutbound
 * @property {'card_state'} type
 * @property {ChatHistoryCard} card
 * @property {CardSubmissionSummary[]} submissions_summary
 * @property {number} submissions_dropped
 * @property {string} ts
 * @property {number=} chat_id
 */

/**
 * @typedef {Object} CardWidgetRef
 * @property {string} skill
 * @property {string} tab_id
 * @property {Object=} input
 */

/**
 * `url` serves the app's HTML as `text/plain` (nosniff, no-store): build a sandboxed `srcdoc`, never navigate to it.
 * @typedef {Object} CardAppRef
 * @property {string} sha256
 * @property {number} size
 * @property {string} url
 */

/**
 * `GET /api/cards/{card_id}`: exactly one of `components` (with optional `data`), `widget` or `app`.
 * @typedef {Object} CardResponse
 * @property {string} card_id
 * @property {'components'|'widget'|'app'} kind
 * @property {string} title
 * @property {string} fallback_text
 * @property {number} chat_id
 * @property {string} task_id
 * @property {number} revision
 * @property {string} created_at
 * @property {string} updated_at
 * @property {boolean} pinned
 * @property {Object[]=} components
 * @property {Object=} data
 * @property {CardWidgetRef=} widget
 * @property {CardAppRef=} app
 * @property {CardSubmissionSummary[]} submissions_summary
 * @property {number} submissions_dropped
 */

/**
 * One start the page did not outlive (web/modules/holds.js).
 * @typedef {Object} UiHold
 * @property {string} key
 * @property {string} rev
 */

/**
 * @typedef {Object} UiHoldsSurface
 * @property {boolean=} pywebview
 * @property {boolean=} telegram
 */

/**
 * `POST /api/ui/holds`.
 * @typedef {Object} UiHoldsRequest
 * @property {Array<UiHold>} holds
 * @property {UiHoldsSurface=} surface
 */

/**
 * @typedef {Object} UiHoldsResponse
 * @property {boolean} ok
 * @property {number} named
 */

/**
 * `POST /api/widgets/fault` (`gateway/widgets.py`): one fault a mounted module widget's card shows; identity is
 * the page's mount, the rest the frame's clipped report.
 * @typedef {Object} WidgetFaultRequest
 * @property {string} skill
 * @property {string} tab_id
 * @property {string} revision
 * @property {'board'|'message'} where
 * @property {'error'|'rejection'|'csp'} kind
 * @property {string} message
 * @property {string} source
 * @property {number} line
 */

/**
 * @typedef {Object} WidgetFaultResponse
 * @property {boolean} ok
 * @property {boolean} logged
 */

/**
 * `POST`/`DELETE /api/cards/{card_id}/pin`.
 * @typedef {Object} CardPinResponse
 * @property {boolean} ok
 * @property {string} card_id
 * @property {boolean} pinned
 * @property {string=} pinned_at
 */

/**
 * @typedef {Object} CallOpenRequest
 * @property {number} chat_id
 * @property {string} client_message_id
 * @property {string=} feature
 */
/**
 * @typedef {Object} CallOpenResponse
 * @property {string} call_id
 * @property {string} ws_path
 * @property {boolean} vision
 * @property {boolean=} joined
 */
/**
 * @typedef {Object} CallRefusal
 * @property {false} ok
 * @property {string} code
 * @property {string} error
 */
/**
 * @typedef {Object} ConversationFinishedOutbound
 * @property {"conversation_finished"} type
 * @property {number} chat_id
 * @property {string} call_id
 * @property {string} task_id
 * @property {string} reason
 */
/**
 * Separate media socket; control schemas are in GET /api/contract.schemas.
 * @typedef {Object} CallSocketContract
 * @property {string} path
 * @property {string} host_open_path
 * @property {string} host_peer_path
 * @property {string} host_auth
 * @property {number} protocol
 * @property {string} audio_header
 * @property {number} audio_header_bytes
 * @property {number} max_audio_payload_bytes
 * @property {number} max_video_payload_bytes
 * @property {number} max_control_bytes
 * @property {Object.<string,string>} peer_messages
 * @property {Object.<string,string>} server_messages
 */
/**
 * @typedef {Object} HostCallOpenRequest
 * @property {'meeting'} role
 * @property {string=} call_id
 * @property {string=} binding_id
 * @property {Object=} event
 * @property {string=} label
 * @property {boolean=} mic
 * @property {boolean=} speaker
 * @property {boolean=} video
 */
/**
 * @typedef {Object} HostCallOpenResponse
 * @property {string} call_id
 * @property {string} peer_path
 */
/**
 * @typedef {Object} CallHello
 * @property {'hello'} type
 * @property {1} protocol
 * @property {'owner'|'meeting'} role
 * @property {boolean} mic
 * @property {boolean} speaker
 * @property {boolean=} video
 * @property {number} rate
 * @property {string=} label
 */
/**
 * @typedef {Object} CallMute
 * @property {'mute'} type
 * @property {boolean} muted
 */
/**
 * @typedef {Object} CallPlayed
 * @property {'played'} type
 * @property {string} item_id
 * @property {number} played_ms
 */
/**
 * @typedef {Object} CallContext
 * @property {'context'} type
 * @property {string} text
 * @property {string} source
 */
/**
 * @typedef {Object} CallParticipant
 * @property {'participant'} type
 * @property {string} label
 * @property {boolean} speaking
 */
/**
 * @typedef {Object} CallShare
 * @property {'share'} type
 * @property {boolean} sharing
 * @property {'screen'|'camera'} kind
 */
/**
 * @typedef {Object} CallBye
 * @property {'bye'} type
 * @property {string=} reason
 */
/**
 * @typedef {Object} CallState
 * @property {'state'} type
 * @property {'preparing'|'connecting'|'live'|'reconnecting'|'closing'|'closed'} state
 * @property {string} reason
 * @property {string=} engine
 * @property {string=} phase
 * @property {string=} detail
 * @property {'settings'|'retry'=} action
 */
/**
 * @typedef {Object} CallActivity
 * @property {'activity'} type
 * @property {'working'|'idle'} activity
 * @property {string} label
 */
/**
 * @typedef {Object} CallTurn
 * @property {'turn'} type
 * @property {'awaiting'|'answering'|'idle'} turn
 */
/**
 * @typedef {Object} CallCaption
 * @property {'caption'} type
 * @property {'owner'|'ouroboros'} speaker
 * @property {string} text
 * @property {boolean} final
 * @property {string} item_id
 * @property {boolean=} replace
 */
/**
 * @typedef {Object} CallFlush
 * @property {'flush'} type
 * @property {number} epoch
 */

/**
 * @typedef {Object} OwnerSafetyRulesRequest
 * @property {string} rules
 */

/**
 * @typedef {Object} OwnerSafetyRulesResponse
 * @property {boolean} ok
 * @property {string} rules
 */

/**
 * @typedef {Object} OwnerSafetyModelRequest
 * @property {Object} settings
 */

/**
 * @typedef {Object} OwnerSafetyModelResponse
 * @property {boolean} ok
 */
