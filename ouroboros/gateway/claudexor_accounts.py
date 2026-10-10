"""Agent accounts HTTP surface (D30): six THIN proxies, zero auth logic.

Ouroboros's own Claudexor daemon (``claudexor_daemon.py``) owns every account
fact — profiles, login jobs, device-code custody, the two honest verification
statuses, quota windows. The browser cannot talk to the daemon directly (its
control plane is loopback-Origin-guarded and bearer-token'd; the token must
never reach a page), so these handlers translate: status aggregation, the
owner-initiated daemon wake behind Refresh or explicit Agents activation, login job
create, login job read/cancel/input, login job termination reconcile, and
the credential-profile row actions — removal and the Enabled toggle,
DELETE/PATCH on one route. Nothing here interprets a credential and nothing
here stores one.

Login-job wire contract (frozen, ``ClaudexorLoginJobResponse`` /
``ClaudexorLoginJobProblem`` in ``gateway/contracts.py``): every operation
answers ONE envelope with ONE top-level bare ``job`` — create adds its
create-only metadata beside it, input keeps its ``ok`` bit, and the snapshot
poll passes the daemon's own ``{job, cursor, sequence, deviceCode?}`` envelope
through VERBATIM (it already has that shape; wrapping it again was issue #124's
double ``job.job``). Daemon 404/410 job-absence verdicts pass through for
poll/cancel/reconcile; typed 409 passes through only for input and reconcile,
with stable ``code`` plus ``required_actions`` when the engine names the
continuation. A create-time daemon 400/409 — or the frozen retryable 503
terminal-transport probe verdict — passes through with its status, code,
actions and the engine's own sentence. Transport/discovery failure before
setup creation, other daemon 5xx, and untyped poll/cancel conflicts collapse
to this proxy's honest 503.

Login shapes ("красота-сначала", D30): a structural link/device-code card
wherever the engine can host the flow itself — codex device-code today, and
other harnesses according to the engine's per-harness, host-effective
``setupLogin`` fact: the job snapshot's transient overlay carries the sign-in
link and supported input. Only a legacy row that truly omits that field may
consult the old engine-global operations catalog. Explicit null delegates
support to the typed setup-create admission; malformed current evidence never
falls back. ``external_terminal`` produces a labelled, platform-correct
copy-paste command for the USER'S OWN terminal. Before any
profile/job mutation, the serving handshake's version/build/entry selects the
preserved packaged Node + runtime entry and that entry's fresh probe must
advertise ``setup_attach``; old probes without the additive role are honestly
unsupported. The exact argv is retained through job creation and rendered only
after its id exists, demoted card-side to a collapsed Advanced affordance.
There is no PATH-CLI or in-app terminal surface, and none may be added.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error, request_json_or

log = logging.getLogger(__name__)

# The three read states of one status facet (see the `reads` block below).
READ_OK = "ok"
READ_NOT_READ = "not_read"
READ_FAILED = "failed"

# An EXPLICIT "daemon" transport is DELIBERATELY not accepted from the
# browser: on a pre-disclosure engine it is the macOS Terminal.app handoff,
# and D30 forbids that mechanism outright. Whether an OMITTED transport may
# default to the engine-hosted flow is decided per-engine from the
# /v2/operations catalog (see _login_disclosure_native), never assumed.
_LOGIN_TRANSPORTS = ("", "client_pty")
_TERMINAL_TRANSPORT_ERROR_CODES = {
    "terminal_transport_unavailable",
    "terminal_transport_unsupported",
    "terminal_transport_probe_failed",
    "terminal_transport_failed",
}

# The engine advertises its disclosure-driven login modes (3.3.7 contract) by
# implementing the setup-job input route: `POST /v2/setup/jobs/:id/input`
# ships in the same engine change that makes claude/cursor login jobs
# disclosure-driven (oauth_url in the snapshot overlay, no Terminal). One
# structural marker for the one feature, and it is the catalog ID — the
# engine's own stable identifier for the operation, not the path spelling.
_LOGIN_INPUT_OPERATION_ID = "post:setup.jobs.id.input"

# Current engines publish the EFFECTIVE setup-login choice on each harness
# row. Keep the four wire states separate: omission is the one legacy signal;
# explicit null requires the authoritative setup-create admission; a valid
# object selects a transport; and a malformed/current row is a capability gap,
# never permission to guess from the old engine-global operation catalog.
_SETUP_LOGIN_ABSENT = "absent"
_SETUP_LOGIN_NULL = "null"
_SETUP_LOGIN_OBJECT = "object"
_SETUP_LOGIN_MALFORMED = "malformed"
_SETUP_LOGIN_TRANSPORTS = {
    "in_app": "",
    "external_terminal": "client_pty",
}

# Claudexor 3.6.0 assigned the generic ``internal_error`` code to a duplicate
# profile's HTTP 409. Neither spelling is sufficient on its own: only the
# canonical exact-row read below can turn one of these legacy conflicts into
# an idempotent retry.
_LEGACY_PROFILE_CONFLICT_CODES = {"http_409", "internal_error"}


def _harness_setup_login(capabilities: Dict[str, Any], harness: str) -> Tuple[str, str]:
    """Return ``(wire_state, mode)`` for exactly one harness catalog row.

    The wire deliberately makes ``setupLogin`` optional *and* nullable. An old
    3.6.0-shaped row omits it; a current producer owns the key and emits either
    null or ``{mode}``. Missing/duplicate harness rows and invalid values are
    malformed current evidence, not legacy omission.
    """
    rows = capabilities.get("harnesses") if isinstance(capabilities, dict) else None
    if not isinstance(rows, list):
        return _SETUP_LOGIN_MALFORMED, ""
    matches = [row for row in rows
               if isinstance(row, dict) and str(row.get("id") or "") == harness]
    if len(matches) != 1:
        return _SETUP_LOGIN_MALFORMED, ""
    row = matches[0]
    if "setupLogin" not in row:
        return _SETUP_LOGIN_ABSENT, ""
    setup = row.get("setupLogin")
    if setup is None:
        return _SETUP_LOGIN_NULL, ""
    if not isinstance(setup, dict):
        return _SETUP_LOGIN_MALFORMED, ""
    mode = setup.get("mode")
    if not isinstance(mode, str) or mode not in _SETUP_LOGIN_TRANSPORTS:
        return _SETUP_LOGIN_MALFORMED, ""
    return _SETUP_LOGIN_OBJECT, mode


def _credential_profile_registered(payload: Dict[str, Any], harness: str,
                                   profile_id: str) -> bool:
    """Whether an exact named profile exists in the daemon's canonical read."""
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, list):
        return False
    for wrapper in profiles:
        profile = wrapper.get("profile") if isinstance(wrapper, dict) else None
        if not isinstance(profile, dict):
            continue
        if (str(profile.get("harness_id") or "") == harness
                and str(profile.get("profile_id") or "") == profile_id):
            return True
    return False


def _login_disclosure_native(operations: List[Dict[str, Any]]) -> bool:
    """Legacy compatibility: did the old engine publish setup-job input?

    True iff the ``/v2/operations`` catalog advertises the setup-job input
    route under its EXACT catalog id. The path spelling is deliberately NOT a
    second, independent way to answer yes: a route template is a much weaker
    identifier (any engine that ever mounted that shape, under any id, would
    pass), and a false positive here is the expensive direction — it sends an
    OLD engine down the transportless path, whose daemon-side default is the
    macOS Terminal.app handoff D30 forbids. A false NEGATIVE only costs the
    attach fallback, which works on every engine. Pure for unit tests; a
    caller that could not READ the catalog must pass [] and get False. Current
    per-harness rows never use this engine-global signal."""
    for op in operations:
        if not isinstance(op, dict):
            continue
        if str(op.get("id") or "") == _LOGIN_INPUT_OPERATION_ID:
            return True
    return False


# The UNIFIED ACCOUNT MODEL's feature marker (frozen contract, sprint plan §L.2):
# `GET /v2/account-pools` ships in the same engine change that migrates every
# default CLI login into a named registry row, empties `harnessAccounts` and
# carries the pool routing verdict in the additive `accountPools` key. One
# structural marker for the one feature, and it is the catalog ID — the
# engine's own stable identifier for the operation, not the path spelling.
# Absent on every 3.5.0 engine.
_ACCOUNT_POOLS_OPERATION_ID = "get:account-pools"


def _unified_accounts_native(operations: List[Dict[str, Any]]) -> bool:
    """Does this engine serve the UNIFIED account model (every account a named
    registry row, routing facts in ``accountPools``)?

    True iff the ``/v2/operations`` catalog advertises the account-pools read
    under its EXACT catalog id (the ``_login_disclosure_native`` pattern). The
    expensive direction is a false POSITIVE: it would make the client treat a
    legacy engine's native pseudo-rows as named profiles (Remove buttons on
    rows the engine cannot delete, a verify-race that never resolves). A false
    negative only costs the old rendering, which is correct on every engine
    that lacks the route. Pure for unit tests; a caller that could not READ
    the catalog must pass [] and get False — fail closed to the old behavior.
    """
    for op in operations:
        if not isinstance(op, dict):
            continue
        if str(op.get("id") or "") == _ACCOUNT_POOLS_OPERATION_ID:
            return True
    return False


def _login_capable_harness_ids(rows: List[Dict[str, Any]]) -> "set | None":
    """Harnesses whose manifest declares a ``native_session`` auth source.

    That is the engine's own discriminator for "an account you LOG INTO"
    (codex, claude, cursor). The raw-api/openrouter adapters authenticate with
    an API key only — projecting them into the accounts panel gave them
    "Log in" buttons no flow can honor, and leaked them into the reviewer
    slots' subscriptions group. Read from ``/v2/harnesses`` because the
    agent-capability catalog is a derived projection that drops the
    manifest's auth block.

    Returns ``None`` when the response carried ZERO readable manifests overall
    — the answer succeeded but says nothing about auth, so callers must fail
    open exactly like an unreachable read (a blip must not blank the panel).
    Pure for unit tests."""
    capable = set()
    readable = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        manifest = row.get("manifest")
        if not isinstance(manifest, dict):
            continue
        readable += 1
        profile = manifest.get("capability_profile")
        auth = profile.get("auth") if isinstance(profile, dict) else None
        sources = auth.get("supported_sources") if isinstance(auth, dict) else None
        if any(str(s) == "native_session" for s in sources or []):
            capable.add(str(row.get("id") or ""))
    return capable if readable else None


def _account_row_visible(harness_id: str, row: Dict[str, Any], capable) -> bool:
    """Account-surface row predicate: manifest-declared login capability OR the
    daemon vouching an actual login on the row itself (``native_login_detected``)
    — a harness with a null/unreadable manifest must not lose the account the
    owner is really logged into. Deliberately NOT "keep all unknown-manifest
    rows": that would resurrect raw-api (manifest-null on the live engine),
    which is the exact leak this filter closes."""
    if capable is None:
        return True
    return harness_id in capable or bool(row.get("native_login_detected"))


# One status read per variant (with or without per-harness models) at a time: concurrent
# requests join the read in flight instead of fanning the daemon out again. Each facet
# keeps its own last answered read per daemon home (the catalog per variant: its rows carry
# models or not), so a failed or unasked facet serves that value again, stale, under its
# own observation time and this read's error.
_STATUS_LOCK = threading.Lock()
_STATUS_IN_FLIGHT: Dict[bool, Future] = {}
_FACET_MEMORY: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}
_FACET_FIELDS = {"catalog": ("harnesses",), "accounts": ("profiles",),
                 "quota": ("quota", "quota_absences", "resources")}


def _status_payload(include_models: bool, *, join: bool = True) -> Dict[str, Any]:
    """The status answer: joins a read of the same variant already in flight (an owner
    wake passes ``join=False`` and reads anew); every caller gets its own copy."""
    with _STATUS_LOCK:
        shared = _STATUS_IN_FLIGHT.get(include_models) if join else None
        owner = shared is None
        if owner:
            shared = Future()
            if join:
                _STATUS_IN_FLIGHT[include_models] = shared
    if owner:
        try:
            shared.set_result(_with_facet_memory(_read_status_payload(include_models), include_models))
        except BaseException as exc:
            shared.set_exception(exc)
        finally:
            with _STATUS_LOCK:
                if _STATUS_IN_FLIGHT.get(include_models) is shared:
                    del _STATUS_IN_FLIGHT[include_models]
    return copy.deepcopy(shared.result())


def _with_facet_memory(payload: Dict[str, Any], include_models: bool = False) -> Dict[str, Any]:
    """Stamp ``facets`` and serve each unanswered facet's last answered value as stale.

    An ``ok`` read replaces the facet's memory and observation time; a ``failed`` or
    ``not_read`` facet keeps both (a stale value never refreshes its timestamp) and carries
    this read's typed error. ``reads`` still states what THIS read established."""
    from ouroboros.utils import utc_now_iso

    now, home = utc_now_iso(), str(payload.get("config_dir") or "")
    errors = payload.pop("_facet_errors", {})
    facets: Dict[str, Dict[str, Any]] = {}
    with _STATUS_LOCK:
        for facet, fields in _FACET_FIELDS.items():
            key = (home, facet, include_models and facet == "catalog")
            if payload["reads"][facet] == READ_OK:
                _FACET_MEMORY[key] = held = {
                    "observed_at": now, "value": copy.deepcopy({name: payload[name] for name in fields if name in payload})}
            else:
                held = _FACET_MEMORY.get(key)
                payload.update(copy.deepcopy(held["value"]) if held else {})
            facets[facet] = {"observed_at": held["observed_at"] if held else None,
                             "stale": bool(held) and payload["reads"][facet] != READ_OK,
                             "error": errors.get(facet)}
    payload["facets"] = facets
    return payload


def _read_status_payload(include_models: bool) -> Dict[str, Any]:
    """One uncached status fan-out; a facet's refusal rides ``_facet_errors``, never the daemon state."""
    from ouroboros.claudexor_daemon import get_owned_daemon, owned_config_dir
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway,
        ClaudexorUnavailable,
        discover_daemon_at,
    )
    from ouroboros.subagents import subagent_last_delegation

    daemon = get_owned_daemon().status_dict()
    payload: Dict[str, Any] = {
        "daemon": daemon,
        "config_dir": str(owned_config_dir()),
        "harnesses": [],
        "profiles": {},
        "quota": [],
        "quota_absences": [],
        # /v2/operations is distinct from agent_capabilities (reads.catalog).
        "resource_capabilities_read": READ_NOT_READ,
        # PROVENANCE, per independent facet (BIBLE P1: missing data is a GAP,
        # never a value). `[]`/`{}` alone cannot say WHETHER the daemon was
        # asked: the owner's panel printed "no account connected" for three
        # harnesses while two claude profiles, a cursor profile and two native
        # sessions sat on disk, simply because a lazily-started daemon had not
        # been asked. `ok` — this facet was read and its collection is
        # AUTHORITATIVE (empty means empty); `not_read` — never asked, either
        # because no daemon was running or because discovery/handshake died
        # BEFORE the fan-out and left every facet untouched; `failed` — asked, and
        # no usable answer came back: the read refused, or the body arrived in a
        # shape the facet does not promise. Facets are
        # independent because one fan-out read can fail while its siblings land.
        "reads": {
            "catalog": READ_NOT_READ,
            "accounts": READ_NOT_READ,
            "quota": READ_NOT_READ,
        },
        # UNIFIED ACCOUNT MODEL feature fact (sprint plan §L.2): True only when
        # the engine's own /v2/operations catalog was READ and advertises
        # `GET /v2/account-pools`. Anything else — old engine, unreadable
        # catalog, no daemon — is False: the client falls closed to the legacy
        # native-pseudo-row rendering, which is correct on every engine that
        # lacks the route. Deliberately NOT a facet: like the login-capability
        # manifest read it is a rendering input whose failure is absorbed, not
        # reported.
        "unified_accounts": False,
        # The Subagents section's «last delegated run» receipt — Ouroboros's
        # own projection, not daemon truth, so it is served even with the
        # daemon down. {} = no delegated run recorded (absence, not a default).
        "subagent_last_delegation": subagent_last_delegation(),
    }
    if daemon.get("state") != "running":
        return payload
    try:
        endpoint = discover_daemon_at(owned_config_dir())
        with ClaudexorGateway(endpoint) as gateway:
            gateway.handshake()
            payload["daemon"]["engine_version"] = gateway.engine_version

            def _quota_state() -> Dict[str, Any]:
                from ouroboros.gateways.claudexor import account_resource_capabilities

                try:
                    capabilities = account_resource_capabilities(operations_call.result())
                    payload["resource_capabilities_read"] = READ_OK
                except Exception:
                    payload["resource_capabilities_read"] = READ_FAILED
                    capabilities = account_resource_capabilities([])
                payload["resource_capabilities"] = capabilities
                reader = getattr(gateway, "quota_state", None)
                if callable(reader):
                    if capabilities["read"]:
                        result = reader(view="resources")
                        if not isinstance(result, dict) or not isinstance(result.get("resources"), list):
                            raise ClaudexorUnavailable("malformed_response", "Account resources were not returned")
                        return result
                    return reader()
                # Compatibility for old embedded gateway doubles. The shipped
                # gateway has quota_state, so the live status path always uses
                # one physical GET and one evidence epoch.
                return {"snapshots": gateway.quota_snapshots(), "absences": []}
            # The catalog, manifest, profile and quota reads are INDEPENDENT GETs
            # over one thread-safe httpx client, and each costs SECONDS daemon-side
            # (it probes the real coding-agent CLIs on every read: binary, version,
            # login state). Serialized, the panel waited for their SUM — ~23s on a
            # warm daemon with nothing on screen; fanned out it waits for the
            # slowest. Failure semantics are per-facet: each result is classified
            # on its own (see `_facet_outcome`), a refusal surfaces as that facet's
            # own typed error below WITHOUT downgrading the siblings that
            # landed, and a manifest refusal still fails OPEN.
            with ThreadPoolExecutor(max_workers=5) as pool:
                operations_call = pool.submit(lambda: gateway.operations())
                catalog_call = pool.submit(gateway.agent_capabilities)
                manifests_call = pool.submit(gateway.harnesses)
                profiles_call = pool.submit(gateway.credential_profiles)
                quota_call = pool.submit(_quota_state)
                # Deferred lookup on purpose: the failure of THIS read (or a
                # transport double that lacks the method) must land inside the
                # future, where the absorbed fail-closed handling below owns it.
            # Classify every submitted future INDEPENDENTLY, before consuming
            # any of them. Reading `.result()` in sequence would make a facet's
            # verdict depend on which sibling raised first — a catalog failure
            # would leave `accounts` reported as unread even though its own read
            # succeeded. Each facet answers only for itself.
            catalog_outcome = _facet_outcome(catalog_call, envelope=("harnesses",))
            profiles_outcome = _facet_outcome(
                profiles_call, envelope=("profiles", "harnessAccounts"))
            quota_outcome = _facet_outcome(
                quota_call,
                envelope=("snapshots",),
                list_fields=("snapshots",),
            )
            payload["reads"] = {
                "catalog": catalog_outcome[0],
                "accounts": profiles_outcome[0],
                "quota": quota_outcome[0],
            }
            # A refused facet keeps its own typed error; the daemon that answered the
            # handshake and its other facets stay as read (no whole-payload `unreachable`).
            payload["_facet_errors"] = {
                facet: outcome[2].code if outcome[2] is not None else "malformed_response"
                for facet, outcome in (("catalog", catalog_outcome), ("accounts", profiles_outcome),
                                       ("quota", quota_outcome)) if outcome[0] == READ_FAILED}
            catalog = catalog_outcome[1] if catalog_outcome[0] == READ_OK else {}
            # Account surfaces show only harnesses with a login concept. On a
            # transient manifest-read failure — or a successful read with zero
            # readable manifests (the helper answers None) — fail OPEN
            # (no filter): a blip must not blank the panel. The manifest read is
            # deliberately NOT a facet: it is a filter input, and its failure is
            # already absorbed rather than reported.
            try:
                capable = _login_capable_harness_ids(manifests_call.result())
            except ClaudexorUnavailable:
                capable = None
            # The unified-accounts feature fact, from the engine's own route
            # catalog. An unreadable catalog fails CLOSED to the old behavior
            # (False) — the legacy rendering is correct on every engine, while
            # a guessed True would draw named-row affordances over pseudo-rows
            # the engine cannot honor. Absorbed like the manifest read: a
            # rendering input, never a reported facet.
            operations = []
            try:
                operations = operations_call.result()
                payload["unified_accounts"] = _unified_accounts_native(operations)
            except Exception:
                log.debug("operations catalog read failed; assuming legacy account model",
                          exc_info=True)
            rows: List[Dict[str, Any]] = []
            for row in catalog.get("harnesses") or []:
                if not isinstance(row, dict):
                    continue
                if not _account_row_visible(str(row.get("id") or ""), row, capable):
                    continue
                projected = {
                    "id": str(row.get("id") or ""),
                    "display_name": str(row.get("displayName") or row.get("id") or ""),
                    "status": str(row.get("status") or ""),
                    "enabled": bool(row.get("enabled")),
                    "provider_family": str(row.get("providerFamily") or ""),
                    "access_profiles_supported": [
                        str(v) for v in row.get("accessProfilesSupported") or []
                    ],
                }
                if include_models and projected["id"]:
                    try:
                        from ouroboros.gateway.models import account_catalog_models, account_catalog_supported

                        if account_catalog_supported(operations, "/v2/harnesses/:id/models"):
                            envelope = gateway.harness_model_catalog(projected["id"], view="accounts")
                            projected["model_catalog"] = envelope
                            projected["models"] = [
                                {**model, "credential_profile_id": account.get("credentialProfileId"),
                                 "availability": account.get("availability"), "problem": account.get("problem"),
                                 "provenance": source.get("provenance"), "observed_at": source.get("observedAt"),
                                 "catalog_source": source.get("source")}
                                for account, source, model in account_catalog_models(envelope)
                            ]
                        else:
                            projected["models"] = gateway.harness_models(projected["id"])
                    except ClaudexorUnavailable as exc:
                        projected["models"] = []
                        projected["models_error"] = exc.code
                rows.append(projected)
            payload["harnesses"] = rows
            profiles = profiles_outcome[1] if profiles_outcome[0] == READ_OK else {}
            accounts = profiles.get("harnessAccounts") if isinstance(profiles, dict) else None
            if capable is not None and isinstance(accounts, list):
                # The daemon emits a native pseudo-row for EVERY adapter,
                # including the API-key-only ones; same filter, same reason —
                # but a vouched login (native_login_detected) keeps its row
                # even when the manifest is unreadable.
                profiles["harnessAccounts"] = [
                    r for r in accounts
                    if isinstance(r, dict)
                    and _account_row_visible(str(r.get("harness_id") or ""), r, capable)
                ]
            wrappers = profiles.get("profiles") if isinstance(profiles, dict) else None
            if capable is not None and isinstance(wrappers, list):
                # Named credential-profile wrappers leak the same way: an
                # api_key-kind profile registered for a non-loginable harness
                # would render a fake-loginable account row. Keep a wrapper
                # only for a login-capable harness OR one the daemon vouches a
                # native login for among the pseudo-rows above.
                vouched = {
                    str(r.get("harness_id") or "")
                    for r in (accounts if isinstance(accounts, list) else [])
                    if isinstance(r, dict) and r.get("native_login_detected")
                }
                profiles["profiles"] = [
                    w for w in wrappers
                    if isinstance(w, dict)
                    and isinstance(w.get("profile"), dict)
                    and str(w["profile"].get("harness_id") or "") in (capable | vouched)
                ]
            payload["profiles"] = profiles
            quota = quota_outcome[1] if quota_outcome[0] == READ_OK else {}
            raw_snapshots = quota.get("snapshots") if isinstance(quota, dict) else None
            raw_absences = quota.get("absences") if isinstance(quota, dict) else None
            payload["quota"] = [
                row for row in (raw_snapshots if isinstance(raw_snapshots, list) else [])
                if isinstance(row, dict)
            ]
            payload["quota_absences"] = [
                row for row in (raw_absences if isinstance(raw_absences, list) else [])
                if isinstance(row, dict)
            ]
            if isinstance(quota.get("resources"), list):
                payload["resources"] = quota["resources"]
    except ClaudexorUnavailable as exc:
        payload["daemon"]["state"] = "unreachable"
        payload["daemon"]["last_error"] = f"{exc.code}: {exc}"
        # A failure BEFORE the fan-out (discovery, handshake) leaves every facet
        # at its `not_read` default, each carrying the error that kept it unasked.
        payload["_facet_errors"] = {facet: exc.code for facet in payload["reads"]}
    return payload


def _facet_outcome(
    call: "Future", *, envelope: tuple = (), list_fields: tuple = (),
) -> tuple:
    """Classify ONE fanned-out read: (state, value, error).

    Independent by construction — a sibling's exception can never downgrade a
    facet whose own read landed, and the verdict does not depend on completion
    or consumption order (the pool has already joined when this runs).

    ``envelope`` names EVERY key the facet's own reader promises to deliver;
    all of them must be present for the body to count as the shape we asked
    for. Requiring only one was its own version of the bug: the accounts
    envelope carries named profiles AND native rows, and half an envelope made
    the missing half an authoritative empty. This is not schema validation; it
    closes ONE reachable case, and TWO different bodies reach it. A NON-OBJECT
    body — null, a list, a string — is collapsed by the transport into an empty
    ``{}`` (both ``ClaudexorGateway.agent_capabilities`` and
    ``credential_profiles`` end in ``return body if isinstance(body, dict) else
    {}``), so it arrives here already looking like a legitimate empty answer. A
    body that IS an object but has drifted its keys is NOT touched by the
    transport and arrives intact. Either would otherwise be published as an
    AUTHORITATIVE empty — exactly the lie the read block exists to stop — and
    both land on the same verdict here. An envelope carrying none of its keys is
    a read that did not answer, not an account store that is empty. Quota requires
    its canonical ``snapshots`` member as an array while retaining the typed
    absences from that same evidence envelope.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        value = call.result()
    except ClaudexorUnavailable as exc:
        return (READ_FAILED, None, exc)
    if not isinstance(value, (dict, list)):
        return (READ_FAILED, None, None)
    if envelope and not (isinstance(value, dict) and all(key in value for key in envelope)):
        return (READ_FAILED, None, None)
    if list_fields and not all(isinstance(value.get(key), list) for key in list_fields):
        return (READ_FAILED, None, None)
    return (READ_OK, value, None)


async def api_claudexor_status(request: Request) -> JSONResponse:
    """GET /api/claudexor/status[?include=models] — owned-daemon state plus the
    daemon's own account/quota/catalog truth. ``?view=quota`` reads only the
    passive quota and account roster. Read-only; never spawns."""
    include_models = "models" in str(request.query_params.get("include") or "")
    try:
        if request.query_params.get("view") == "quota":
            from ouroboros.gateway.claudexor_passive import _quota_payload

            return JSONResponse(await asyncio.to_thread(_quota_payload))
        return JSONResponse(await asyncio.to_thread(_status_payload, include_models))
    except Exception as exc:
        log.exception("api_claudexor_status failed")
        return json_error(f"{type(exc).__name__}: Claudexor status failed")


async def api_claudexor_wake(request: Request) -> JSONResponse:
    """POST /api/claudexor/wake — OWNER-initiated: start the owned daemon, then
    answer with the freshly read status.

    The status GET stays side-effect-free by contract (and by test), which is
    right for a 5s poll but leaves the panel's Refresh POWERLESS: the daemon is
    lazy, so an owner who just wants to SEE their accounts had to start a login
    job or a delegated run to wake it. Refresh and explicit Agents activation
    call this owner action; background polling and the status GET never do.

    Provisioning cost rides here honestly: a cold runtime install happens inside
    this request rather than behind a silent GET. Owner decision D8 (#844): the
    owner's explicit Refresh also releases the start-failure spawn latch, then
    makes its ONE ordinary ensure through the same funnel — a still-crashing
    engine re-latches, so each press costs one spawn, an explicit owner action.
    """
    from ouroboros.claudexor_daemon import ensure_owned_gateway, get_owned_daemon
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def _wake() -> Dict[str, Any]:
        get_owned_daemon().clear_start_failure_latch(cleared_by="owner_wake")
        gateway = ensure_owned_gateway()
        gateway.close()
        return _status_payload(include_models=False, join=False)

    try:
        return JSONResponse(await asyncio.to_thread(_wake))
    except ClaudexorUnavailable as exc:
        # Typed refusal (missing binary, foreign home, a daemon that never
        # published a descriptor): the panel keeps its rows and says why.
        return json_error(f"{exc.code}: {exc}", 503)
    except Exception as exc:
        log.exception("api_claudexor_wake failed")
        return json_error(f"{type(exc).__name__}: Claudexor wake failed")


def _build_login_request(harness: str, profile_id: str, transport: str,
                         login_flow: str, *, disclosure_native: bool = False) -> Dict[str, Any]:
    """The setup-job request body, honoring the engine's wire contract.

    Pure so the harness-specific transport rule is unit-testable without a
    daemon. The rule mirrors Claudexor's own setup-transport refinement, not
    Ouroboros policy: codex client_pty (the terminal-attach fallback) REQUIRES
    loginFlow=browser_redirect — device/app-server flows are daemon-owned and a
    client_pty job without it is a hard 400 — and loginFlow exists ONLY for
    codex, so it is never sent for another harness (that too is a 400).

    ``disclosure_native`` is the resolved request behavior: normally selected
    by per-harness ``setupLogin.mode``; only a legacy row with that key absent
    may derive it from _login_disclosure_native. True omits the transport so
    the engine hosts the flow and discloses the sign-in link through the
    snapshot overlay. False forces client_pty for non-codex legacy/external
    flows, whose copy-paste attach command is the demoted Advanced fallback.
    """
    request: Dict[str, Any] = {"harness": harness, "action": "login", "authRequest": "subscription"}
    if profile_id:
        request["profileId"] = profile_id
    if harness != "codex" and not transport and not disclosure_native:
        transport = "client_pty"
    if transport:
        request["transport"] = transport
    if harness == "codex" and transport == "client_pty":
        # This is a transport invariant, not a caller preference: device_auth
        # is daemon-owned and Claudexor rejects it on a client_pty job.
        login_flow = "browser_redirect"
    if login_flow and harness == "codex":
        request["loginFlow"] = login_flow
    return request


def _login_job_response(job: Dict[str, Any], **metadata: Any) -> Dict[str, Any]:
    """The ONE producer of the browser login-job success envelope
    (``ClaudexorLoginJobResponse``): exactly one top-level bare ``job``,
    operation metadata beside it, never another envelope nested under ``job``.

    Every operation that receives a BARE ``ControlSetupJob`` from the daemon
    (create, cancel, input, reconcile) wraps it here, once. The snapshot poll
    deliberately does NOT pass through this producer: the daemon's snapshot
    answer is ALREADY the canonical ``{job, cursor, sequence, deviceCode?}``
    envelope, and re-wrapping it produced issue #124's double ``job.job``.
    """
    out: Dict[str, Any] = {"job": job}
    out.update(metadata)
    return out






def _login_create(body: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.claudexor_daemon import (
        attach_login_command,
        attach_login_shell,
        ensure_owned_gateway,
        resolve_attach_login_argv,
    )
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    harness = str(body.get("harness") or "").strip()
    if not harness:
        raise ValueError("harness is required")
    profile_id = str(body.get("profile_id") or "").strip()
    transport = str(body.get("transport") or "").strip()
    if transport not in _LOGIN_TRANSPORTS:
        raise ValueError("transport must be omitted or 'client_pty' (the Terminal.app handoff transport is not offered)")
    login_flow = str(body.get("login_flow") or "").strip()
    explicit_external_transport = transport == "client_pty"
    # Provisioning moment: the FIRST login action is what spawns the owned
    # daemon (and thereby flips default discovery to it) — an owner action,
    # never a boot-time side effect.
    with ensure_owned_gateway() as gateway:
        # The current contract is per HARNESS and per HOST. Only an exact row
        # whose setupLogin key is genuinely absent may use the old global
        # operation signal. Explicit null delegates support to the typed
        # setup/profile admission; malformed current evidence authorizes no
        # profile/setup mutation.
        capability_state, setup_mode = _harness_setup_login(
            gateway.agent_capabilities(), harness)
        setup_login_source = "per_harness"
        if capability_state == _SETUP_LOGIN_NULL:
            # A current null is ambiguous while the vendor CLI is absent: the
            # authoritative setup-create boundary distinguishes an admitted
            # managed-login family from an unsupported harness before job
            # mutation. Keep an omitted transport omitted so that boundary,
            # and the one exact post-install retry, can recompute the real mode.
            setup_login_source = "setup_job_admission"
            disclosure_native = not explicit_external_transport
        elif capability_state == _SETUP_LOGIN_MALFORMED:
            raise ClaudexorUnavailable(
                "setup_login_capability_malformed",
                "the agent service did not publish a valid setup-login capability "
                f"for {harness}",
                status_code=503,
            )
        elif capability_state == _SETUP_LOGIN_OBJECT:
            if explicit_external_transport:
                # The card's explicit recovery action is allowed to select the
                # existing client_pty transport even when the host normally
                # supports in-app login. An omitted request still follows the
                # engine's exact per-harness mode below.
                disclosure_native = False
            else:
                transport = _SETUP_LOGIN_TRANSPORTS[setup_mode]
                disclosure_native = setup_mode == "in_app"
        else:
            setup_login_source = "legacy_global_operation"
            if explicit_external_transport:
                disclosure_native = False
            else:
                try:
                    disclosure_native = _login_disclosure_native(gateway.operations())
                except Exception:
                    log.debug("legacy operations catalog read failed; using attach fallback", exc_info=True)
                    disclosure_native = False
        request_body = _build_login_request(
            harness, profile_id, transport, login_flow,
            disclosure_native=disclosure_native)
        attach_argv: list[str] = []
        if request_body.get("transport") == "client_pty":
            # Preflight BEFORE profile registration or setup-job creation. An
            # old serving probe has no setup_attach role; returning a missing
            # command after creating its client_pty job would strand that job.
            # Preserve this one exact argv through creation so a staged pin or
            # concurrent next-spawn selection can never change the command.
            handshake = gateway.handshake()
            engine = handshake.get("engine") if isinstance(handshake, dict) else None
            try:
                attach_argv = resolve_attach_login_argv(engine)
            except ClaudexorUnavailable as exc:
                exc.login_create_verdict = True
                raise
        if profile_id:
            try:
                gateway.create_credential_profile(harness, profile_id)
            except ClaudexorUnavailable as exc:
                if exc.code == "credential_profile_exists":
                    pass
                elif (int(getattr(exc, "status_code", 0) or 0) == 409
                      and exc.code in _LEGACY_PROFILE_CONFLICT_CODES):
                    # Claudexor 3.6.0 did not have the typed duplicate code.
                    # Its generic conflict is idempotent only after the daemon's
                    # exact read proves this same harness/profile row exists.
                    try:
                        exists = _credential_profile_registered(
                            gateway.credential_profiles(), harness, profile_id)
                    except Exception:
                        exists = False
                    if not exists:
                        exc.login_profile_verdict = True
                        raise
                else:
                    # Validation/conflict failures are daemon verdicts and keep
                    # their status/code/actions at the browser boundary. A 5xx
                    # or transport failure still becomes the proxy's honest 503.
                    exc.login_profile_verdict = True
                    raise

        def _create() -> Dict[str, Any]:
            try:
                return gateway.setup_job_create(request_body)
            except ClaudexorUnavailable as exc:
                # Mark WHERE the daemon answered: only a 400 from the job CREATE
                # is a verdict about the requested login shape (the pass-through
                # the endpoint forwards). A handshake or discovery 400 earlier in
                # this block is engine/protocol trouble and stays the honest 503.
                exc.login_create_verdict = True
                raise

        job = _create()
        # Connect is the owner's consent. If the exact pinned engine answers
        # this FIRST create with its structural pre-command missing-binary
        # terminal job, install that vendor CLI once and create the job once
        # more; a second refusal is returned, never looped.
        from ouroboros.claudexor_daemon import (
            install_missing_harness_cli,
            is_immediate_missing_cli_job,
        )

        if is_immediate_missing_cli_job(job, harness, gateway):
            install_missing_harness_cli(harness)
            job = _create()
    job_id = str(job.get("id") or job.get("jobId") or "")
    metadata: Dict[str, Any] = {
        "job_id": job_id,
        "disclosure_native": disclosure_native,
        "setup_login_source": setup_login_source,
    }
    if request_body.get("transport") == "client_pty" and job_id:
        # The fallback card's copy-paste command, run OUTSIDE this UI. Read
        # from the REQUEST actually sent (the non-codex default is forced
        # inside the builder, not in the caller's local variable).
        metadata["attach_command"] = attach_login_command(job_id, argv=attach_argv)
        metadata["attach_shell"] = attach_login_shell()
    return _login_job_response(job, **metadata)


async def api_claudexor_login(request: Request) -> JSONResponse:
    """POST /api/claudexor/login — create one login job on the owned daemon."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    body: Dict[str, Any] = await request_json_or(request, {})
    try:
        return JSONResponse(await asyncio.to_thread(_login_create, dict(body or {})))
    except ValueError as exc:
        return json_error(str(exc), 400)
    except ClaudexorUnavailable as exc:
        status = int(getattr(exc, "status_code", 0) or 0)
        profile_verdict = bool(getattr(exc, "login_profile_verdict", False))
        create_verdict = bool(getattr(exc, "login_create_verdict", False))
        create_problem = create_verdict and (
            status in (400, 409)
            or (status == 503 and exc.code in _TERMINAL_TRANSPORT_ERROR_CODES)
        )
        if profile_verdict or create_problem:
            # The daemon ANSWERED and refused either profile registration or
            # setup creation. Preserve profile-create 4xx verdicts — including
            # a typed required-profile action and non-idempotent conflicts —
            # instead of collapsing them to daemon unavailability. Setup-job
            # create keeps its frozen 400/409 plus retryable 503 boundary. Pass
            # status, stable code and the engine's own message through in the
            # frozen problem envelope
            # (``ClaudexorLoginJobProblem``). Transport/discovery failure
            # before this marked create stage and other daemon 5xx — unproven,
            # not verdicts — stay this proxy's honest generic 503.
            extra: Dict[str, Any] = {"code": exc.code}
            actions = tuple(getattr(exc, "required_actions", ()) or ())
            if actions:
                extra["required_actions"] = list(actions)
            browser_status = status if 400 <= status < 500 else 503
            return json_error(str(exc), browser_status, **extra)
        return json_error(f"{exc.code}: {exc}", 503)
    except Exception as exc:
        log.exception("api_claudexor_login failed")
        return json_error(f"{type(exc).__name__}: Claudexor login failed")


def _login_job_call(job_id: str, op: str, value: str = "") -> Dict[str, Any]:
    """The one internal job-operation dispatcher: EXPLICIT ``op``, never the
    HTTP method or path spelling, decides what runs. Both job-scoped handlers
    call it, so the browser envelope has one producer per shape: snapshot is
    the daemon's canonical envelope VERBATIM (no re-wrap — issue #124 — and no
    poll-time ``attach_command``: the card deliberately trusts attach metadata
    only from create); every bare-job answer is wrapped exactly once by
    ``_login_job_response``."""
    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at

    endpoint = discover_daemon_at(owned_config_dir())
    with ClaudexorGateway(endpoint) as gateway:
        gateway.handshake()
        answer = gateway.setup_job_call(job_id, op, value=value)
    if op == "snapshot":
        return answer
    if op == "input":
        return _login_job_response(answer, ok=True)
    return _login_job_response(answer)


def _login_job_problem(exc: Any, op: str) -> JSONResponse:
    """Translate one typed daemon refusal into the browser problem envelope
    (``ClaudexorLoginJobProblem``): required ``error``, optional stable
    ``code``, optional bounded ``required_actions``.

    The daemon's own job VERDICTS pass through with their status instead of
    being rewritten as 503: 404/410 mean the job is no longer available —
    an answer about the JOB record, never about the login outcome or the old
    process (#151 collapsed these to 503, making the client's already-gone
    branch unreachable). 409 is OPERATION-SCOPED, never blanket: input keeps
    its existing typed conflicts, and reconcile passes the engine's typed
    refusal through — ``setup_termination_unconfirmed`` plus its
    ``requiredActions`` continuation (``retry_setup_reconciliation``). A 409
    on poll or cancel has no typed client branch, so it stays the proxy's
    503 rather than acquiring a new passthrough. Input also keeps its
    operation-scoped 404 downgrade: ITS 404 means "this engine does not
    accept sign-in codes for this job", a capability fact, not job absence,
    so it stays the distinct ``input_not_supported`` code. Everything else —
    transport failure, daemon 5xx — is this proxy's honest 503: unproven,
    not a verdict.
    """
    status = int(getattr(exc, "status_code", 0) or 0)
    if op == "input":
        if status == 404:
            return json_error(
                "This engine does not accept sign-in codes for this job "
                "(input route not available)", 404, code="input_not_supported")
        if status == 409:
            # The engine's TYPED input conflicts ride through verbatim —
            # setup_input_not_applicable (the callback already completed; no
            # code needed) and setup_input_already_submitted (a repeat the
            # server refused). Answers, not failures: the card maps the code
            # to friendly copy.
            return json_error(str(exc), 409, code=exc.code)
    elif status in (404, 410) or (status == 409 and op == "reconcile"):
        extra: Dict[str, Any] = {"code": exc.code}
        actions = tuple(getattr(exc, "required_actions", ()) or ())
        if actions:
            extra["required_actions"] = list(actions)
        return json_error(str(exc), status, **extra)
    return json_error(f"{exc.code}: {exc}", 503)


async def api_claudexor_login_job(request: Request) -> JSONResponse:
    """The job-scoped login proxy — one endpoint, three verbs:

    - ``GET /api/claudexor/login/{job_id}`` — poll: the daemon's snapshot
      envelope verbatim; it carries the transient disclosure (device code /
      oauth_url) at the ENVELOPE level when the flow has one.
    - ``DELETE /api/claudexor/login/{job_id}`` — the card's cancel action;
      the answer body carries the resulting job so the client can tell a
      confirmed terminal state from ``termination_unconfirmed`` custody.
    - ``POST /api/claudexor/login/{job_id}/input`` — forward ONE line of user
      input (the claude OAuth paste-code) to the engine's setup-job input
      route. The value rides through and is never logged or stored here.
      Capability degradation is TYPED: an engine that predates the input
      route (or no longer knows the job) answers 404, which comes back as
      ``code=input_not_supported`` so the card can fall back to the Advanced
      attach affordance instead of showing a raw error.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    job_id = str(request.path_params.get("job_id") or "").strip()
    if not job_id:
        return json_error("job_id is required", 400)
    is_input = request.method == "POST"
    value = ""
    if is_input:
        # STRICT, and strict in the proxy sense: a JSON OBJECT carrying a
        # `value` that is ALREADY a string of the length the engine accepts.
        # Nothing is coerced (`str(123)` is not a sign-in code) and nothing is
        # rewritten — an edge that trims decides for the engine what the user
        # typed, and a length check on the trimmed form is not the check the
        # engine performs. A non-object body is refused here rather than
        # raising an AttributeError one line later.
        body: Any = await request_json_or(request, {})
        raw = body.get("value") if isinstance(body, dict) else None
        if not isinstance(raw, str) or not raw:
            return json_error("value is required (a non-empty JSON string)", 400)
        if len(raw) > 1024:
            # The engine's ControlSetupJobInputRequest caps the value at 1024
            # chars; refuse at this edge with the same bar (a paste-code is
            # one short line — an oversized body is a caller bug).
            return json_error("value is implausibly long for a sign-in code", 400)
        value = raw
    op = "input" if is_input else ("cancel" if request.method == "DELETE" else "snapshot")
    try:
        return JSONResponse(await asyncio.to_thread(_login_job_call, job_id, op, value))
    except ClaudexorUnavailable as exc:
        return _login_job_problem(exc, op)
    except Exception as exc:
        log.exception("api_claudexor_login_job failed (%s)", op)
        return json_error(f"{type(exc).__name__}: Claudexor login job {op} failed")


async def api_claudexor_login_job_reconcile(request: Request) -> JSONResponse:
    """POST /api/claudexor/login/{job_id}/reconcile — the SIXTH thin proxy:
    ask the daemon to prove an unconfirmed termination's process group empty
    (``POST /v2/setup/jobs/:id/reconcile``).

    Its own handler because its body contract differs from the sibling POST
    (/input requires a value; reconcile takes none), but the operation runs
    through the same explicit-op dispatcher — the path spelling is routing,
    never the state authority. Success returns the reconciled job in the one
    browser envelope; an unprovable termination comes back as the daemon's
    409 ``setup_termination_unconfirmed`` with ``required_actions`` naming
    the retry continuation. The check itself never creates a login (owner
    decision 1A: the reconciled face offers a separate explicit retry).
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    job_id = str(request.path_params.get("job_id") or "").strip()
    if not job_id:
        return json_error("job_id is required", 400)
    try:
        return JSONResponse(await asyncio.to_thread(_login_job_call, job_id, "reconcile"))
    except ClaudexorUnavailable as exc:
        return _login_job_problem(exc, "reconcile")
    except Exception as exc:
        log.exception("api_claudexor_login_job_reconcile failed")
        return json_error(f"{type(exc).__name__}: Claudexor login job reconcile failed")


def _remove_credential_profile(harness: str, profile_id: str) -> Dict[str, Any]:
    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at

    endpoint = discover_daemon_at(owned_config_dir())
    with ClaudexorGateway(endpoint) as gateway:
        gateway.handshake()
        return gateway.delete_credential_profile(harness, profile_id)


def _update_credential_profile(harness: str, profile_id: str, enabled: bool) -> Dict[str, Any]:
    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at

    endpoint = discover_daemon_at(owned_config_dir())
    with ClaudexorGateway(endpoint) as gateway:
        gateway.handshake()
        gateway.update_credential_profile(harness, profile_id, enabled=enabled)
    return {"ok": True, "harness": harness, "profile_id": profile_id, "enabled": bool(enabled)}


async def api_claudexor_credential_profile(request: Request) -> JSONResponse:
    """DELETE|PATCH /api/claudexor/credential-profiles/{harness}/{profile_id}.

    Two thin proxies on one route, same rule as their siblings: the daemon
    owns the account record. DELETE asks the engine to forget one named
    account (``DELETE /v2/credential-profiles/:harness/:profileId``) — nothing
    here touches a vendor credential file, and a legacy engine's native CLI
    login has no route because this process cannot honestly sign it out.
    PATCH is the Enabled toggle
    (``PATCH /v2/credential-profiles/:harness/:profileId`` with the engine's
    own strict ``{enabled}`` body): whether this account participates in the
    engine's rotation pool. The refusal, in both cases, is the answer.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    harness = str(request.path_params.get("harness") or "").strip()
    profile_id = str(request.path_params.get("profile_id") or "").strip()
    if not harness or not profile_id:
        return json_error("harness and profile_id are required", 400)
    is_update = request.method == "PATCH"
    if is_update:
        # STRICT in the proxy sense, mirroring the engine's own
        # ControlCredentialProfileUpdateRequest: one JSON object with one
        # BOOLEAN `enabled`. Nothing is coerced — a "true" string is a caller
        # bug the engine would refuse anyway, and rewriting it here would
        # decide for the engine what the caller meant.
        body: Any = await request_json_or(request, {})
        enabled = body.get("enabled") if isinstance(body, dict) else None
        if not isinstance(enabled, bool):
            return json_error("enabled is required (a JSON boolean)", 400)
    try:
        if is_update:
            return JSONResponse(
                await asyncio.to_thread(_update_credential_profile, harness, profile_id, enabled))
        return JSONResponse(
            await asyncio.to_thread(_remove_credential_profile, harness, profile_id))
    except ClaudexorUnavailable as exc:
        return json_error(f"{exc.code}: {exc}", 503)
    except Exception as exc:
        log.exception("api_claudexor_credential_profile failed (%s)", request.method)
        return json_error(f"{type(exc).__name__}: Claudexor account "
                          f"{'update' if is_update else 'removal'} failed")


__all__ = [
    "api_claudexor_credential_profile",
    "api_claudexor_login",
    "api_claudexor_login_job",
    "api_claudexor_login_job_reconcile",
    "api_claudexor_status",
    "api_claudexor_wake",
]
