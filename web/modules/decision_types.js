/** Dependency-free JSDoc mirror of `ouroboros.gateway.decision_contracts`: the one owner-decision
 * ingress body and reply, re-exported by ./api_types.js beside the frames they answer, and the
 * integration-secret form's request, reply and metadata. */

/**
 * POST /api/decisions body — the ONE answer ingress for owner decision cards
 * (decision families quiz:/routing:/interaction:/model_wait:/card:). request_id is the
 * idempotency key; a replay returns the recorded confirmation. option_index is
 * optional for a quiz free answer and for typed model_wait actions. A quiz
 * free answer sends a non-empty comment; model_wait sends revision and action. A quiz
 * action=dismiss removes an accepted late answer of unknown delivery from current
 * questions, carrying no answer or reason (never delivered, never resent). A living card's press
 * (card:{card_id}:{action_id}) carries the card `revision` it was drawn from (a newer card answers 409
 * card_stale) and `values` only for a form action.
 * @typedef {Object} DecisionRequest
 * @property {string} request_id
 * @property {string} decision_id
 * @property {number=} option_index
 * @property {string=} comment
 * @property {number=} revision
 * @property {string=} action
 * @property {string=} reason Explicit quiz closure explanation.
 * @property {boolean=} auto_continue
 * @property {string=} model
 * @property {string=} credential_profile_id
 * @property {boolean=} use_local
 * @property {boolean=} persist_role
 * @property {Object=} values  // card: form field values (JSON scalars)
 */

/**
 * Answer-ingress reply; 409 carries the card's truthful lifecycle state so a
 * late click settles the card instead of inviting retries.
 * @typedef {Object} DecisionResponse
 * @property {string=} closed_actor
 * @property {string=} closed_at
 * @property {string=} closed_reason
 * @property {string=} closed_request_id
 * @property {Object=} question_delivery
 * @property {string=} request_id
 * @property {boolean=} applied
 * @property {(boolean|null)=} saved
 * @property {Object=} wait
 * @property {string=} reason_code
 * @property {boolean=} ok
 * @property {string=} decision_id
 * @property {string=} state
 * @property {number=} answered_index
 * @property {string=} comment
 * @property {boolean=} duplicate
 * @property {boolean=} answered_after_terminal
 * @property {boolean=} forwarded
 * @property {string=} error
 * @property {string=} dispatched
 * @property {string=} task_id
 * @property {string=} latest_status
 * @property {string=} reason
 * @property {string=} detail
 * @property {string=} cause  // the owner-facing sentence for a refused routing act (409 dispatch_rejected)
 * @property {string=} reasoning_effort  // a New task picked from the card: the start its admitted row requests; never on a steer
 * @property {('delivered'|'forwarded'|'owed')=} delivery  // card: where the press went (owed answers 202)
 * @property {string=} card_id
 * @property {number=} revision
 * @property {Object=} winner  // card 409 card_action_once: {request_id, at, delivery} of the press that counted
 */

/**
 * One asked field of an integration-secret request: its public id and the Settings key it writes.
 * @typedef {Object} SecretRequestField
 * @property {string} id
 * @property {string} setting_key
 */

/**
 * An integration-secret request (showSecretHitl) on its question card; metadata only, never a value.
 * A save writes only while the skill still maps these field ids to these keys.
 * @typedef {Object} SecretRequestMeta
 * @property {string} plugin_name
 * @property {SecretRequestField[]} fields
 */

/**
 * Whether an asked key holds a value in Settings now: a boolean, never a value or prefix.
 * @typedef {Object} SecretKeyState
 * @property {string} id
 * @property {string} setting_key
 * @property {boolean} configured
 */

/**
 * An installed skill that reads an asked key, and what its current grant covers (facts, never a grant).
 * @typedef {Object} SecretConsumer
 * @property {string} name
 * @property {boolean} enabled
 * @property {string[]} reads
 * @property {string[]=} granted
 * @property {boolean=} unsupported
 * @property {boolean=} grant_unknown
 */

/**
 * The host's typed completion of a secret request: secret_filled or secret_replaced after Settings
 * kept the values (replaced_fields held a value just before), or secret_cancelled. consumers is what
 * the readers' grants covered after the save (saving grants nothing), or consumers_unknown when they
 * could not be read, never an empty list; late marks a save after the asking task ended, which
 * resumed nothing.
 * @typedef {Object} SecretOutcome
 * @property {"secret_filled"|"secret_replaced"|"secret_cancelled"} event
 * @property {string} pluginName
 * @property {string[]} fields
 * @property {string} at
 * @property {string[]} settings_keys
 * @property {string[]=} replaced_fields
 * @property {SecretConsumer[]=} consumers
 * @property {boolean=} consumers_unknown
 * @property {boolean=} late
 * @property {boolean=} interrupted_save
 */

/**
 * The outcome's public event payload: plugin and field ids, never a value.
 * @typedef {Object} SecretEvent
 * @property {string} pluginName
 * @property {string[]} fields
 */

/**
 * POST /api/decisions/secret body — saveSecretToken. A save carries values for exactly the asked
 * field ids and observed (per field, whether the form showed its key as set); it writes only while
 * Settings still says that, so Replace is explicit. status and cancel carry neither; status needs no
 * request_id. Reuse request_id only for unchanged values.
 * @typedef {Object} SecretDecisionRequest
 * @property {string=} request_id
 * @property {string} decision_id
 * @property {("save"|"cancel"|"status")=} action
 * @property {Object<string, string>=} values
 * @property {Object<string, boolean>=} observed
 */

/**
 * Secret form reply; every save and cancel reply and every refusal carries saved on both sides of the
 * Settings commit, a status read only the lifecycle and form facts. A status read and a refusal on a
 * request that still takes a save add keys and consumers (or keys_unknown/consumers_unknown). saved
 * true with secret_save_unverified, secret_not_retained or secret_completion_unrecorded wrote Settings
 * without reporting a save; saved null with secret_save_unconfirmed is an attempt of unknown outcome.
 * Read status first; only a new request saves again.
 * @typedef {Object} SecretDecisionResponse
 * @property {boolean=} ok
 * @property {(boolean|null)=} saved
 * @property {string=} decision_id
 * @property {string=} state
 * @property {boolean=} duplicate
 * @property {SecretRequestMeta=} secret_request
 * @property {SecretEvent=} secret_filled
 * @property {SecretEvent=} secret_replaced
 * @property {SecretEvent=} secret_cancelled
 * @property {SecretOutcome=} secret_outcome
 * @property {SecretKeyState[]=} keys
 * @property {boolean=} keys_unknown
 * @property {SecretConsumer[]=} consumers
 * @property {boolean=} consumers_unknown
 * @property {boolean=} save_unconfirmed
 * @property {boolean=} answered_after_terminal
 * @property {string=} closed_reason
 * @property {string=} closed_at
 * @property {string=} closed_actor
 * @property {string=} closed_request_id
 * @property {string=} closed_by_task_id
 * @property {Object=} question_delivery
 * @property {string=} error
 * @property {string=} reason_code
 */
