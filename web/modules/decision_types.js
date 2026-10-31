/** Dependency-free JSDoc mirror of `ouroboros.gateway.decision_contracts`: the one owner-decision
 * ingress body and reply, re-exported by ./api_types.js beside the frames they answer. */

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
