/** Dependency-free JSDoc mirror of `ouroboros.gateway.sign_in_contracts`: a skill's sign-in to a service it
 *  declares (`POST /api/sign-in`, `GET /api/sign-in/status`). No shape carries a token, secret or verifier. */

/**
 * One step: begin | complete | cancel | disconnect | revoke | client_save | client_remove.
 * @typedef {Object} SignInRequest
 * @property {string} action
 * @property {string=} skill
 * @property {string=} service
 * @property {string=} account
 * @property {(string|Object)=} client  // begin: a client id or label; client_save: {id?, label, client_id, client_secret?, return}
 * @property {string[]=} optional_scopes
 * @property {string[]=} scope_words  // begin: the exact literal set shown for the selected permissions
 * @property {string=} registration  // begin, with a saved client: its registration as shown (SignInClientRow.registration)
 * @property {string=} consent  // disconnect, revoke: the account's consent as shown (SignInAccountRow.consent)
 * @property {string=} login_hint
 * @property {string=} state
 * @property {string=} code
 * @property {string=} iss
 * @property {string=} error
 * @property {string=} error_description
 * @property {string=} returned_url
 * @property {string=} attempt_id  // Cancel and an editor-bound paste; never OAuth state
 * @property {boolean=} register_client  // begin: explicit registration when no eligible client is saved, with the issuer shown
 * @property {string=} issuer  // client_save: the approved provider; begin with register_client: the provider shown (SignInServiceStatus.issuer)
 * @property {?Object<string, string>=} endpoints  // client_save: the full shown explicit map; absent or null for discovery
 */

/**
 * `ok: false` carries `code` (SIGN_IN_REQUIRED or SIGN_IN_UNAVAILABLE), `reason` and `error`. A Revoke is final only with
 * `revoked: true`; otherwise the account is kept and the step refused (`revocation_not_confirmed`, `revocation_not_asked`). A missing provider answer is typed: `registration_unconfirmed` (begin), `answer_saved` (complete, with `attempt_id`).
 * @typedef {Object} SignInResponse
 * @property {boolean} ok
 * @property {string=} url
 * @property {string=} redirect_uri
 * @property {Object=} client
 * @property {string=} scope
 * @property {string=} skill
 * @property {string=} service
 * @property {string=} label
 * @property {Object=} account
 * @property {string[]=} missing_required
 * @property {string=} attempt_id
 * @property {string=} state
 * @property {boolean=} cancelled
 * @property {string=} revocation_reason
 * @property {?boolean=} revoked
 * @property {boolean=} removed
 * @property {string=} code
 * @property {string=} reason
 * @property {string=} error
 */

/**
 * One of the owner's clients of an issuer; `return` is `loopback` or `exact:<address>`.
 * @typedef {Object} SignInClientRow
 * @property {string} id
 * @property {string=} label
 * @property {string=} issuer
 * @property {string=} client_id_tail
 * @property {string=} return
 * @property {string=} source
 * @property {boolean=} secret_set
 * @property {string=} registration  // names the exact registration shown, never a secret
 * @property {Object[]=} used_by
 */

/**
 * One account of one skill's service; `siblings`: other skills' grants of the same issuer, client and account.
 * @typedef {Object} SignInAccountRow
 * @property {string} key
 * @property {string=} label
 * @property {string=} subject
 * @property {string=} consent  // names the consent shown: a token refresh keeps it, a new sign-in changes it
 * @property {boolean=} verified
 * @property {string=} state
 * @property {string=} reason
 * @property {string=} required_scope  // the whole open requirement
 * @property {?string=} required_scope_since
 * @property {string[]=} missing_step_up  // the unmet words, computed by the host
 * @property {string=} attempt_id
 * @property {?string=} since
 * @property {string[]=} scope
 * @property {string[]=} missing_required
 * @property {string[]=} not_allowed
 * @property {Object=} client
 * @property {string=} via
 * @property {Object[]=} siblings
 * @property {Object=} last_error
 */

/**
 * A declared service of an installed skill with its own words (label, scope_labels, notes).
 * @typedef {Object} SignInServiceStatus
 * @property {string} skill
 * @property {string} service
 * @property {string=} label
 * @property {string=} issuer
 * @property {Object<string, string>=} endpoints
 * @property {string[]=} scopes
 * @property {Object=} scope_labels
 * @property {Object<string, string[]>=} scope_spellings  // other spellings a provider may return for one declared scope
 * @property {Object[]=} optional_scopes
 * @property {string=} accounts_mode
 * @property {boolean=} optional
 * @property {string=} admission  // "", or why the skill's current bytes may not start a sign-in: not_reviewed, disabled, not_trusted
 * @property {Object=} notes
 * @property {SignInAccountRow[]=} accounts
 * @property {?Object=} pending  // {started, client, attempt_id}; the ID is not a credential
 * @property {SignInClientRow[]=} clients
 */

/**
 * @typedef {Object} SignInStatusResponse
 * @property {boolean} ok
 * @property {SignInServiceStatus[]=} services
 * @property {string=} return_address
 */

/**
 * 503 of GET /api/sign-in/status: a saved sign-in, client or account list could not be read; never an empty list.
 * @typedef {Object} SignInStatusProblem
 * @property {boolean} ok
 * @property {string} code  // SIGN_IN_UNAVAILABLE
 * @property {string} reason  // sign_in_storage_unavailable
 * @property {string=} error
 */
