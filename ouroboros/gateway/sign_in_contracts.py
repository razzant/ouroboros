"""Owner contracts of skills' service sign-ins (``ouroboros/gateway/sign_in.py``); JSDoc twin
``web/modules/sign_in_contract_types.js``. No shape carries a token, a client secret or a PKCE verifier."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:
    from typing import Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python < 3.11
    from typing_extensions import Required, TypedDict  # type: ignore[assignment]


class SignInRequest(TypedDict, total=False):
    """``POST /api/sign-in``: ``begin`` selects a saved ``client`` or explicitly asks to ``register_client``
    when none is saved, naming the ``issuer`` shown; ``account`` names the row to sign in again. ``complete`` accepts the provider's
    return or ``returned_url``, optionally tied to the editor's ``attempt_id``; ``cancel`` retires that attempt.
    ``disconnect``/``revoke`` name the account. ``client_save`` takes the approved ``issuer`` and, for
    explicit mode, the whole ``endpoints`` map plus ``client``; a blank secret or client ID keeps the saved
    value. ``client_remove`` names the saved client."""

    action: Required[str]
    skill: str
    service: str
    account: str
    client: Union[str, Dict[str, Any]]
    register_client: bool
    attempt_id: str
    optional_scopes: List[str]
    scope_words: List[str]  # begin: the exact literal set shown for the selected permissions
    registration: str  # begin, with a saved client: its registration as shown (SignInClientRow.registration)
    consent: str  # disconnect, revoke: the account's consent as shown (SignInAccountRow.consent)
    login_hint: str
    state: str
    code: str
    iss: str
    error: str
    error_description: str
    returned_url: str
    issuer: str
    endpoints: Dict[str, str]


class SignInResponse(TypedDict, total=False):
    """A refused step carries a code and reason. A completed return can file an account that still needs sign-in. A Revoke is final
    only with ``revoked: true`` (the account is forgotten); otherwise the account is kept and the step refused: ``revocation_not_confirmed``
    (asked; a refusal, a lost answer or only part confirmed) or ``revocation_not_asked`` (nothing was sent; ``revocation_reason`` is
    no_endpoint, no_tokens or provider_approval_required). A missing provider answer is typed where the host knows what it left:
    ``registration_unconfirmed`` (``begin``) and ``answer_saved`` (``complete``, with ``attempt_id``: the answer is kept, only the account read waits)."""

    ok: Required[bool]
    url: str
    redirect_uri: str
    client: Dict[str, Any]
    scope: str
    skill: str
    service: str
    label: str
    account: Dict[str, Any]
    missing_required: List[str]
    revoked: Optional[bool]
    revocation_reason: str
    cancelled: bool
    attempt_id: str
    state: str
    removed: bool
    code: str
    reason: str
    error: str


# One of the owner's clients of an issuer: ``return`` is ``loopback`` or ``exact:<address>`` (a keyword, hence
# the functional form); ``secret_set`` says a secret is saved, never what it is; ``registration`` names the exact registration shown
# (client ID, provider addresses, return) without revealing it.
SignInClientRow = TypedDict("SignInClientRow", {
    "id": Required[str], "label": str, "issuer": str, "client_id_tail": str, "return": str, "source": str,
    "secret_set": bool, "registration": str, "used_by": List[Dict[str, Any]]}, total=False)


class SignInAccountRow(TypedDict, total=False):
    """One signed-in account of one skill's service; ``siblings`` are other skills' grants of the same
    issuer, client and account (a revocation may end them too)."""

    key: Required[str]
    label: str
    subject: str
    consent: str  # names the consent shown: a token refresh keeps it, a new sign-in changes it
    verified: bool
    state: str
    reason: str
    required_scope: str  # the whole open requirement
    required_scope_since: Optional[str]
    missing_step_up: List[str]  # unmet words, compared using the grant's saved spellings
    attempt_id: str
    since: Optional[str]
    scope: List[str]
    missing_required: List[str]
    not_allowed: List[str]
    client: Dict[str, Any]
    via: str
    siblings: List[Dict[str, Any]]
    last_error: Dict[str, Any]


class SignInServiceStatus(TypedDict, total=False):
    """A declared service of an installed skill, with its words (``label``, ``scope_labels``, ``notes``) and the
    other spellings a provider may return for one declared scope (``scope_spellings``, never a new right)."""

    skill: Required[str]
    service: Required[str]
    label: str
    issuer: str
    endpoints: Dict[str, str]
    scopes: List[str]
    scope_labels: Dict[str, str]
    scope_spellings: Dict[str, List[str]]
    optional_scopes: List[Dict[str, Any]]
    accounts_mode: str
    optional: bool
    admission: str  # "", or why the skill's current bytes may not start a sign-in: not_reviewed, disabled, not_trusted
    notes: Dict[str, str]
    accounts: List[SignInAccountRow]
    pending: Optional[Dict[str, Any]]
    clients: List[SignInClientRow]


class SignInStatusResponse(TypedDict, total=False):
    """``GET /api/sign-in/status``; ``return_address``: the builtin return page a desktop-kind client lists."""

    ok: Required[bool]
    services: List[SignInServiceStatus]
    return_address: str


class SignInStatusProblem(TypedDict, total=False):
    """``GET /api/sign-in/status`` 503: a saved sign-in, client or account list could not be read, so nothing is painted as empty.
    ``code`` is ``SIGN_IN_UNAVAILABLE``, ``reason`` ``sign_in_storage_unavailable``; ``error`` a masked sentence."""

    ok: Required[bool]
    code: Required[str]
    reason: Required[str]
    error: str


__all__ = ["SignInAccountRow", "SignInClientRow", "SignInRequest", "SignInResponse", "SignInServiceStatus",
           "SignInStatusProblem", "SignInStatusResponse"]
