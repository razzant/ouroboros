/** Dependency-free Places JSDoc mirror of ouroboros.gateway.client_contracts.
 *  Re-exported by client_contract_types.js; one PlaceSummary for every Places surface. */

/**
 * Connect Silent; ids default to empty. An instance label is prefixed with the service. An Add's note,
 * description and circle_id land with a new place only; a connected one answers 409 place_exists.
 * @typedef {Object} PlaceConnectRequest
 * @property {string} connector
 * @property {string=} account_id
 * @property {string=} conversation_id
 * @property {string=} label
 * @property {string=} note
 * @property {string=} description
 * @property {string=} circle_id
 */

/**
 * The circle a place is in and who placed it there; the words and their author are the circle's.
 * @typedef {Object} PlaceCircleChoice
 * @property {string} circle_id
 * @property {string} placed_by
 * @property {string} placed_at
 */

/**
 * A short purpose line under the place's name: never a rule or permission.
 * @typedef {Object} PlaceDescription
 * @property {string} text
 * @property {string} author
 * @property {string} at
 */

/**
 * One circle, named once per list: its whole words, who wrote them, and its model (empty inherits consciousness).
 * @typedef {Object} CircleSummary
 * @property {string} circle_id
 * @property {string} name
 * @property {string} text
 * @property {string} text_author
 * @property {string} text_at
 * @property {number} revision
 * @property {string} model
 */

/**
 * A new circle last in the order; its words are the owner's.
 * @typedef {Object} CircleCreateRequest
 * @property {string} name
 * @property {string=} text
 * @property {string=} model
 */

/**
 * Rename, reword or change the model; expected_revision compares the circle's revision before writing.
 * @typedef {Object} CircleUpdateRequest
 * @property {string=} name
 * @property {string=} text
 * @property {string=} model
 * @property {number=} expected_revision
 */

/**
 * @typedef {Object} CircleDeleteRequest
 * @property {number=} expected_revision
 */

/**
 * Sanitized host mark or SVG geometry; an unavailable mark is {}.
 * @typedef {Object} PlaceIcon
 * @property {'host'|'svg'=} kind
 * @property {string=} name
 * @property {string=} svg
 */

/**
 * @typedef {Object} PlaceReadiness
 * @property {string} text
 * @property {string} tone
 */

/**
 * The connector's declared account; the client derives owner from confirmed user accounts.
 * @typedef {Object} PlaceIdentityAccount
 * @property {string} id
 * @property {string} label
 * @property {'user'|'bot'} kind
 * @property {boolean=} owner
 */

/**
 * Optional accounts used to write and read; absent means undeclared.
 * @typedef {Object} PlaceIdentity
 * @property {PlaceIdentityAccount=} sends_as
 * @property {Array<PlaceIdentityAccount>=} reads_as
 */

/**
 * The declared source.url response; identity and labels are optional on older connectors.
 * @typedef {Object} PlaceStatusPayload
 * @property {string} text
 * @property {string} tone
 * @property {PlaceIdentity=} identity
 * @property {Object<string,string>=} labels
 */

/**
 * The model a place thinks with now, and the scope that chose it: its own, its circle's, or the consciousness
 * model. A stored alias (`main`, `light`) reads here as the model it resolves to now.
 * @typedef {Object} PlaceModelRoute
 * @property {string} value
 * @property {'place'|'circle'|'consciousness'} layer
 */

/**
 * One projection for list, write/conflict responses and the model's place map.
 * @typedef {Object} PlaceSummary
 * @property {string} place_id
 * @property {string} connector
 * @property {string} provider
 * @property {string} account_id
 * @property {string} conversation_id
 * @property {string} label
 * @property {'active'|'silent'|'paused'} state
 * @property {boolean=} only_hears Derived per read: true while consciousness is off (or Panic) and the place is not Paused.
 * @property {Array<Object<string,string>>} wakes
 * @property {Array<Object<string,string>>} notes
 * @property {Array<Object<string,string>>} owner_accounts
 * @property {string} working_folder
 * @property {string} model
 * @property {PlaceModelRoute} model_route
 * @property {Array<string>} context_topics
 * @property {Array<string>} legacy_binding_ids
 * @property {number} revision
 * @property {string} thread_id
 * @property {string} destination
 * @property {string} scope
 * @property {string} contribution_id
 * @property {Object<string,string>|null} reply_destination
 * @property {PlaceIcon} icon
 * @property {PlaceReadiness} readiness
 * @property {'off'|'light'|'full'=} safety_mode
 * @property {Object<string,string>=} labels
 * @property {Array<string>} wake_suggestions
 * @property {Array<*>=} facts
 * @property {Object<string,*>=} source
 * @property {string=} settings
 * @property {Array<Object>|null} rooms
 * @property {number|null} unanswered Open focus turns, not owed replies.
 * @property {string} last_activity_at
 * @property {string} activity_coverage
 * @property {number|null} spent_24h_usd
 * @property {Array<Object>} account_candidates
 * @property {Array<Object<string,string>>} shared_owner_accounts Confirmed in another place of this bot or app; they count here.
 * @property {PlaceCircleChoice|null} circle
 * @property {PlaceDescription|null} description
 */

/**
 * @typedef {Object} PlaceDisconnectRequest
 * @property {number=} expected_revision
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
 * @property {string=} circle_id  // "" takes it out of its circle
 * @property {string=} description  // "" clears it
 * @property {number=} expected_revision
 */

/**
 * A captured local conversation, not a complete remote-provider history.
 * @typedef {Object} PlaceRoomSummary
 * @property {number} room_chat_id
 * @property {number|null} unanswered_count
 * @property {string} address
 * @property {string|null} label
 * @property {string} last_at  // latest captured arrival
 * @property {string} conversation_id
 * @property {string} thread_id
 */

/**
 * Store-wide chat import/gaps, not provider completeness.
 * @typedef {Object} PlaceRoomsCoverage
 * @property {'arrivals'} discovery
 * @property {boolean} importing
 * @property {boolean} gaps
 */

/**
 * @typedef {Object} PlaceRoomsResponse
 * @property {string} place_id
 * @property {Array<PlaceRoomSummary>} rooms
 * @property {PlaceRoomsCoverage} coverage
 */

/**
 * Retained attention only; no exact wake match is inferred.
 * @typedef {Object} PlaceHistoryAttention
 * @property {'heard'|'focus_started'} decision
 * @property {string=} reason
 * @property {string=} recipient  // the work or author holding a heard message; its model read is unknown
 */

/**
 * An earlier message its connector imported: platform authorship and Host receipt time; `ts` is its send time.
 * @typedef {Object} PlaceHistoryImport
 * @property {'participant'|'platform_account'} authorship
 * @property {string} received_at
 */

/**
 * A retained room message, delivery receipt or Place notice. Absent facts stay unknown.
 * @typedef {Object} PlaceHistoryFacts
 * @property {'message'|'notice'|'delivery'} kind
 * @property {('authored'|'accepted'|'delivered'|'failed'|'uncertain')=} delivery_state
 * @property {PlaceHistoryAttention=} attention
 * @property {boolean=} owner_confirmed  // current stored account confirmation
 * @property {Array<string>=} facts
 * @property {PlaceIdentityAccount=} sends_as
 * @property {boolean=} sends_as_owner
 * @property {string=} held_text
 * @property {boolean=} held
 * @property {boolean=} checked
 * @property {PlaceIdentity=} identity
 * @property {PlaceHistoryImport=} historical_import  // never owner confirmation or attention
 * @property {string=} sent_at  // the provider send time its arrival recorded; absent: unknown, never inferred
 * @property {PlaceHistoryReaction=} reaction  // a typed reaction arrival or receipt; absent: not one
 */

/**
 * The host's English words for a row's typed reaction, never speech; words the row sent stay its text.
 * @typedef {Object} PlaceHistoryReaction
 * @property {string} summary
 */
