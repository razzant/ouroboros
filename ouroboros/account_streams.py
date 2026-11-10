"""Which account a route's credential belongs to, and how many streams one account allows.

``model_concurrency`` admits physical calls per route. For most providers that route is
the credential, and two models on one key are two independent calls. A provider that
meters one concurrent stream per ACCOUNT is the exception: two models of one account
draw on the same single quota, so the account — not the model — is the unit, and a
second model admitted behind the first comes back as the provider's 429 rather than as
this host's own queue.

``account_identity`` answers "which account" only where the credential ENCODES one; the
answer is hashed, so no secret and no account id ever reaches a queue path. ``stream_ceiling``
is the provider's declared per-account stream count, and a provider absent from that table
declares none: its unit stays the credential and the configured cap applies as before.
A route whose account cannot be read keeps that same credential key — its ceiling is not
applied, because a ceiling without an account would cut one model's own concurrency.
"""

from __future__ import annotations

import base64
import hashlib

# Concurrent streams one ACCOUNT of a provider allows, as it documents and as measured.
# A provider absent here declares no per-account ceiling.
#
# gigachat: one stream for an individual-tier account. Measured 2026-11-10 on such a key:
# two DIFFERENT models on one account, two rounds each, produced exactly one 429 in 4 of 4
# rounds with the loser alternating (two accounts ran 6/6 in parallel). The same provider
# documents ten streams for a legal entity, so a higher tier is UNDER-admitted here until
# this row changes — the safe direction: a host cap only makes this host wait, while
# over-admission is refused by the provider as an outage-shaped 429.
PROVIDER_ACCOUNT_STREAM_CEILINGS: dict[str, int] = {"gigachat": 1}


def _b64_text(value: str) -> str:
    try:
        return base64.b64decode(str(value or "").strip(), validate=False).decode("utf-8", "replace")
    except (ValueError, TypeError):
        return ""


def account_identity(provider: str, route: dict) -> str:
    """A stable, non-secret label of the account behind ``route``'s credential, or ``''``.

    Only a credential that ENCODES its account is read: GigaChat's authorization key is
    ``base64("<client_id>:<secret>")`` and the client id is the quota subject (two keys of
    one client id share the one stream, keys of different client ids do not). The decoded
    text must have the ``id:secret`` shape — an opaque or JWT-shaped token decodes to
    anything, and hashing that would invent an account. Every other provider's credential
    is opaque, so the answer is ``''`` and the caller keeps its credential-based key
    instead of guessing an account from a secret's bytes.
    """
    if provider == "gigachat":
        client_id, separator, _secret = _b64_text(route.get("api_key") or "").partition(":")
        client_id = client_id.strip()
        if separator and client_id and len(client_id) <= 128 and not any(char.isspace() for char in client_id):
            return hashlib.sha256(f"gigachat\0{client_id}".encode("utf-8")).hexdigest()[:16]
    return ""


def stream_ceiling(provider: str) -> int | None:
    """Concurrent streams one account of ``provider`` allows, or ``None`` when it declares none."""
    ceiling = PROVIDER_ACCOUNT_STREAM_CEILINGS.get(str(provider or ""))
    return ceiling if isinstance(ceiling, int) and ceiling > 0 else None
