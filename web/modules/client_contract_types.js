/** Dependency-free JSDoc mirror of `ouroboros.gateway.client_contracts` — the replaceable-client envelopes
 *  `api_types.js` indexes: the served main interface, the setup bootstrap, the settings writers' refusal,
 *  the auth gate's JSON login and the `GET /api/contract` document. */
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
 * @property {'gate'|'public'} auth
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
 * One `GET /api/chat/history` row (live frames call `text` `content`). A row may also carry the
 * task-progress and outcome fields of a live `chat` frame (`ChatOutbound`). Replay-evidence rows
 * (`quiz_answer`, `quiz_closed`, `task_evidence`, or `summary_kind` + `historical_terminal`) are never shown.
 * @typedef {Object} ChatHistoryMessage
 * @property {PresenceContact=} presence_contact
 * @property {string=} wait_ended_at
 * @property {string=} row_address
 * @property {Object=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
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
 * `coverage` v1: delivered stored bytes per stream; a span is `{from, to, chain, gaps}` or null.
 * Byte delivery, never chronology or proof of the conversation's start.
 * @typedef {Object} ChatHistoryCoverage
 * @property {number} v
 * @property {string} view
 * @property {Object.<string,number>} upper
 * @property {Object.<string,?{from: number, to: number, chain: string, gaps: string[]}>} spans
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
 * Connect an installed connector; omitted account and conversation ids default to empty.
 * @typedef {Object} PlaceConnectRequest
 * @property {string} connector
 * @property {string=} account_id
 * @property {string=} conversation_id
 */

/**
 * A captured local conversation, not a complete remote-provider history.
 * @typedef {Object} PlaceRoomSummary
 * @property {number} room_chat_id
 * @property {number|null} unanswered_count
 * @property {string} address
 */

/**
 * @typedef {Object} PlaceRoomsResponse
 * @property {string} place_id
 * @property {Array<PlaceRoomSummary>} rooms
 */

/**
 * Confirm a person; omitted account_id uses the place's account.
 * @typedef {Object} PlaceOwnerAccountRequest
 * @property {string} platform_actor_id
 * @property {string=} label
 * @property {string=} account_id
 */

/**
 * Atomic owner choices. Send a nonempty body; expected_revision compares before writing.
 * @typedef {Object} PlaceUpdateRequest
 * @property {'active'|'silent'|'paused'=} state
 * @property {Array<string>=} wakes
 * @property {string=} add_note
 * @property {string=} remove_note
 * @property {string=} working_folder
 * @property {string=} model
 * @property {PlaceOwnerAccountRequest=} confirm_owner_account
 * @property {string=} revoke_owner_account
 * @property {number=} expected_revision
 */

/**
 * `POST /api/settings/secret`: exactly one selector, `{key}`, `{mcp_server_id}` or `{mcp_server_id,
 * header_name}`.
 * @typedef {Object} SettingsSecretRequest
 * @property {string=} key
 * @property {string=} mcp_server_id
 * @property {string=} header_name
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
 * `POST /api/projects/{project_id}/update`: name and/or pin; at least one.
 * @typedef {Object} ProjectUpdateRequest
 * @property {string=} name
 * @property {boolean=} pinned
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
 * @property {Object=} reply_to
 * @property {Object=} relay
 * @property {string=} reply_to_relay
 * @property {string=} kind  // ChatReferenceFields' row kind; the card's own kind is card.kind
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
