"""``ouroboros connections …`` — a THIN owner client for the gateway (RWS v2, D6).

This module holds no runtime logic: every command parses flags, calls one
``/api/owner/connections`` endpoint, and renders the typed answer. The store,
the trust rules, the lease checks, and the transport all live behind the gateway
(``ouroboros/connection_store.py`` + ``ouroboros/gateway/connections.py``);
duplicating any of it here would create a second authority (docs/DEVELOPMENT.md).

It lives beside ``cli.py`` rather than inside it to keep that module under the
1000-line target; the parser wiring stays in ``cli.build_parser``.

The owner password is read ONLY from a controlling terminal — never from argv,
stdin, or the environment — because the connections namespace is owner-gated
even on loopback and an owner credential must not be reachable by a process that
merely inherited the agent's environment.

Exit codes are STABLE and meaningful for scripts:
  0 success · 2 generic refusal/decline · 3 owner action or authentication
  required · 4 the remote or its transport cannot serve this build · 5 conflict
  (retired connection, active task/lease, busy state).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.parse
from typing import Any

from ouroboros.remote_refusal_actions import (
    ACTION_LIST_CONNECTIONS,
    ACTION_NONE,
    ACTION_RUN_FROM_TERMINAL,
    ACTION_TEST,
)

# Owner/authentication class: the operator has to do something, no retry helps.
_OWNER_ACTION_CODES = frozenset({
    "auth_required",
    "confirmation_required",
    "host_identity_changed",
    "host_identity_confirmation_stale",
    "host_identity_missing",
    "owner_action_required",
    "owner_auth_not_configured",
    "owner_auth_required",
})

# Environment/compatibility class: this build, this host, or this remote cannot
# serve the request at all. The distinction a SCRIPT cares about is not which
# layer said no — it is "retrying will not help and no owner action will fix it
# either", which is exit 4 rather than 2 or 3. Everything the transport, the
# broker and the bootstrap raise for that reason belongs here; anything missing
# silently degrades to a generic refusal, which is why the set is enumerated
# rather than pattern-matched.
_UNSERVABLE_CODES = frozenset({
    # gateway
    "remote_transport_unavailable",
    "remote_service_unavailable",
    "remote_service_invalid_response",
    # broker (`remote_workspace`)
    "remote_workspace_unavailable",
    "bootstrap_unsupported",
    "directory_listing_unsupported",
    "reconnect_unsupported",
    "broker_closed",
    # Reconnect with nothing admitted to reconnect: both spellings of "this
    # connection currently has no project session" (the broker's own
    # `remote_session_absent` and the gateway's `remote_session_disconnected`
    # when a reconnect did not reach `ready`). Retrying is pointless — the fix
    # is to readmit a project — so a script must not read this as a generic
    # refusal it could loop over.
    "remote_session_absent",
    "remote_session_disconnected",
    # OpenSSH client / target platform (`remote_ssh`, `remote_ssh_bootstrap`)
    "unsupported_ssh_client",
    "remote_platform_unsupported",
    "remote_libc_unsupported",
    "remote_glibc_too_old",
    # execd bundle / release / protocol identity
    "execd_preamble_invalid",
    "execd_release_unselected",
    "execd_bundle_unavailable",
    "execd_bundle_invalid",
    "execd_artifact_mismatch",
    "capability_mismatch",
    "incompatible_protocol",
    # Home↔execd CONTRACT-SET skew. Refusing a build pair is not something a retry
    # can change, and an owner script that read these as a generic refusal would loop
    # over a condition only Bootstrap (or a rebuilt bundle) resolves.
    "remote_execd_outdated",
    "remote_execd_incompatible",
    "execd_bundle_contract_outdated",
})
# `broker_overloaded` is deliberately NOT in the set above, and used to be. The broker
# raises it with `retryable=True` for a full request queue or too many in-flight
# requests — a capacity state a retry genuinely clears — while exit 4 promises a script
# "retrying will not help and no owner action will fix it either". Two authorities said
# opposite things about one condition, and the mechanical cross-check in
# `tests/test_remote_refusal_action_proofs.py` is what found it: the refusal register
# names `retry` as the action while this set forbade one. Exit 2 is the honest answer
# for a transient refusal until the taxonomy grows a class of its own for one.


#: How many warnings the HUMAN rendering prints in full before disclosing a count.
#: `--json` is never abbreviated here — it carries whatever the gateway sent.
_PRINTED_WARNING_LIMIT = 4


def _read_owner_password() -> str:
    """Read only from a controlling terminal; never stdin, argv, or environment."""

    from ouroboros.cli import CLIError
    from ouroboros.platform_layer import POSIX_CONTROLLING_TERMINAL

    if os.name == "nt":
        if not sys.stdin.isatty():
            raise CLIError("owner password requires an interactive controlling terminal")
        return getpass.getpass("Ouroboros owner password: ")
    try:
        fd = os.open(POSIX_CONTROLLING_TERMINAL, os.O_RDWR)
    except OSError as exc:
        raise CLIError(
            "owner password requires an interactive controlling terminal"
        ) from exc
    with os.fdopen(fd, "r+", encoding="utf-8", closefd=True) as terminal:
        return getpass.getpass("Ouroboros owner password: ", stream=terminal)


def _confirm_host_retrust(old_host_id: str, new_host_id: str) -> bool:
    """Ask the owner, on the terminal, to accept a REPLACEMENT host identity."""

    from ouroboros.cli import CLIError
    from ouroboros.platform_layer import POSIX_CONTROLLING_TERMINAL

    prompt = (
        "Remote host identity changed.\n"
        f"Old: {old_host_id}\n"
        f"New: {new_host_id}\n"
        "Trust this replacement host? Type 'yes' to continue: "
    )
    if os.name == "nt":
        if not sys.stdin.isatty():
            raise CLIError(
                "host retrust confirmation requires an interactive controlling terminal"
            )
        return input(prompt).strip().lower() == "yes"
    try:
        fd = os.open(POSIX_CONTROLLING_TERMINAL, os.O_RDWR)
    except OSError as exc:
        raise CLIError(
            "host retrust confirmation requires an interactive controlling terminal"
        ) from exc
    with os.fdopen(fd, "r+", encoding="utf-8", closefd=True) as terminal:
        terminal.write(prompt)
        terminal.flush()
        return terminal.readline().strip().lower() == "yes"


def _connection_error_exit(error: Any, *, as_json: bool) -> int:
    """Render a typed gateway refusal and map it to a stable exit code."""

    from ouroboros.cli import _print_json

    payload = dict(getattr(error, "payload", {}) or {})
    payload.setdefault("status_code", getattr(error, "status_code", 0))
    if as_json:
        _print_json(payload)
    else:
        phase = str(payload.get("phase") or "").strip()
        action = str(payload.get("action") or "").strip()
        suffix = " ".join(
            part for part in (
                f"phase={phase}" if phase else "",
                f"action={action}" if action else "",
            )
            if part
        )
        print(
            f"error: {payload.get('error') or payload.get('error_code') or error}"
            + (f" ({suffix})" if suffix else ""),
            file=sys.stderr,
        )
    code = str(payload.get("error_code") or "")
    if code in _OWNER_ACTION_CODES:
        return 3
    if int(getattr(error, "status_code", 0)) == 409:
        return 5
    if code in _UNSERVABLE_CODES:
        return 4
    return 2


def _connection_local_error(
    message: str,
    *,
    error_code: str,
    action: str,
    as_json: bool,
    exit_code: int,
) -> int:
    """Report a refusal decided LOCALLY, in the same shape as a gateway refusal."""

    from ouroboros.cli import _print_json

    payload = {
        "ok": False,
        "error": message,
        "error_code": error_code,
        "action": action,
    }
    if as_json:
        _print_json(payload)
    else:
        print(f"error: {message} (action={action})", file=sys.stderr)
    return exit_code


def _observed_cli_host_id(payload: Any) -> str:
    row = payload if isinstance(payload, dict) else {}
    for source in (
        row,
        row.get("handshake") if isinstance(row.get("handshake"), dict) else {},
        row.get("diagnostic") if isinstance(row.get("diagnostic"), dict) else {},
    ):
        value = str(source.get("host_id") or source.get("observed_host_id") or "").strip()
        if value:
            return value
    return ""


def _print_connection_result(data: Any, *, as_json: bool) -> None:
    from ouroboros.cli import _print_json

    if as_json:
        _print_json(data)
        return
    payload = data if isinstance(data, dict) else {}
    row = (
        payload.get("connection")
        if isinstance(payload.get("connection"), dict)
        else payload
    )
    connection_id = str(row.get("id") or payload.get("connection_id") or "")
    name = str(row.get("name") or "")
    status = str(payload.get("status") or row.get("status") or "unknown")
    phase = str(payload.get("phase") or "")
    completion = str(payload.get("completion") or "")
    print(
        " ".join(
            value
            for value in (
                connection_id,
                name,
                f"status={status}",
                f"phase={phase}" if phase else "",
                f"completion={completion}" if completion else "",
            )
            if value
        )
    )
    warnings = [item for item in (payload.get("warnings") or []) if isinstance(item, dict)]
    for warning in warnings[:_PRINTED_WARNING_LIMIT]:
        print(
            "warning: " + json.dumps(warning, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
    # The gateway's own TOTAL if it capped before we saw the list, else what arrived.
    # A cap that leaves no trace is why a real transport warning could vanish between
    # the target and the operator's terminal with nothing to notice.
    total = max(int(payload.get("warnings_count") or 0), len(warnings))
    omitted = total - min(len(warnings), _PRINTED_WARNING_LIMIT)
    if omitted > 0:
        print(
            f"warning: {omitted} of {total} warnings omitted "
            "(use --json for the full bounded list)",
            file=sys.stderr,
        )


def _print_connection_list(data: Any, *, as_json: bool) -> None:
    from ouroboros.cli import _print_json

    if as_json:
        _print_json(data)
        return
    rows = data.get("connections") if isinstance(data, dict) else []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            # A SIXTH column, and only when there is something to say. A connection
            # whose installed execd predates a shared-contract change is `ready` by
            # every other measure and fails every remote tool call, so a status line
            # that omitted the fact described a working connection. The column names
            # the action rather than the number: the number is in `--json`.
            #
            # Both halves come from the SERVER's ladder (`blocked_by`/`blocker_action`
            # — `remote_refusal_actions.connection_blocker`), not from an
            # `execd_outdated` arm written here. Two reasons: this column used to be
            # blind to every other way a connection can be unselectable — a stale
            # health probe printed `ready` with nothing beside it — and a second copy
            # of "which action fixes which state" is a second copy that can drift from
            # what the browser shows for the same row.
            blocked = str(row.get("blocked_by") or "")
            note = (
                f"\tblocked={blocked}\taction={row.get('blocker_action') or 'unknown'}"
                if blocked
                else ""
            )
            print(
                f"{row.get('id', '')}\t{row.get('name', '')}\t"
                f"{row.get('ssh_alias', '')}\t"
                f"{row.get('lifecycle', 'active')}\t"
                f"{row.get('status', 'unknown')}{note}"
            )


def _retrust_request(client: Any, args: argparse.Namespace, *, as_json: bool) -> Any:
    """Collect the identity pair the gateway demands, then ask the owner.

    The gateway is the authority: it re-probes and refuses a stale pair. The CLI
    only makes the change LEGIBLE — the owner sees the old and new identity on
    their terminal before any mutation is sent.
    """

    from ouroboros.cli import CLIError, GatewayHTTPError

    quoted = urllib.parse.quote(args.connection_id)
    listing = client.request("GET", "/api/owner/connections")
    rows = listing.get("connections") if isinstance(listing, dict) else []
    current = next(
        (
            row for row in rows
            if isinstance(row, dict) and str(row.get("id") or "") == args.connection_id
        ),
        None,
    )
    if current is None:
        return _connection_local_error(
            f"unknown connection: {args.connection_id}",
            error_code="connection_not_found",
            action=ACTION_LIST_CONNECTIONS,
            as_json=as_json,
            exit_code=2,
        )
    old_host_id = str(current.get("expected_host_id") or "")
    try:
        probe = client.request("POST", f"/api/owner/connections/{quoted}/test", {})
    except GatewayHTTPError as exc:
        # A FAILED probe still carries the observed identity — that is precisely
        # the case retrust exists for (the pin no longer matches).
        probe = exc.payload
    new_host_id = _observed_cli_host_id(probe)
    if not old_host_id or not new_host_id:
        return _connection_local_error(
            "retrust requires both the pinned and currently observed host identities",
            error_code="host_identity_missing",
            action=ACTION_TEST,
            as_json=as_json,
            exit_code=3,
        )
    try:
        confirmed = _confirm_host_retrust(old_host_id, new_host_id)
    except CLIError as exc:
        return _connection_local_error(
            str(exc),
            error_code="confirmation_required",
            action=ACTION_RUN_FROM_TERMINAL,
            as_json=as_json,
            exit_code=3,
        )
    if not confirmed:
        return _connection_local_error(
            "host retrust declined; no connection metadata changed",
            error_code="confirmation_declined",
            action=ACTION_NONE,
            as_json=as_json,
            exit_code=2,
        )
    return client.request(
        "POST",
        f"/api/owner/connections/{quoted}/retrust",
        {
            "confirm": True,
            "old_host_id": old_host_id,
            "new_host_id": new_host_id,
        },
    )


def connections_command(args: argparse.Namespace) -> int:
    from ouroboros.cli import CLIError, GatewayHTTPError, OuroborosHTTPClient

    command = str(getattr(args, "connections_command", "") or "")
    as_json = bool(getattr(args, "json", False))
    try:
        password = _read_owner_password()
    except CLIError as exc:
        # No terminal ⇒ no request at all: an owner-gated call must never be
        # attempted unauthenticated just to see what the server says.
        return _connection_local_error(
            str(exc),
            error_code="owner_auth_required",
            action=ACTION_RUN_FROM_TERMINAL,
            as_json=as_json,
            exit_code=3,
        )
    client = OuroborosHTTPClient(
        getattr(args, "url", "") or "",
        owner_password=password,
    )
    quoted = urllib.parse.quote(str(getattr(args, "connection_id", "") or ""))
    try:
        if command == "list":
            _print_connection_list(
                client.request("GET", "/api/owner/connections"), as_json=as_json
            )
            return 0
        if command == "add":
            data = client.request(
                "POST",
                "/api/owner/connections",
                {"name": args.name, "ssh_alias": args.ssh_alias},
            )
        elif command in {"test", "bootstrap", "reconnect"}:
            data = client.request(
                "POST",
                f"/api/owner/connections/{quoted}/{command}",
                {},
                timeout=float(
                    _connection_timeouts().get(command, 30.0)
                ),
            )
        elif command == "retire":
            data = client.request("DELETE", f"/api/owner/connections/{quoted}")
        elif command == "retrust":
            data = _retrust_request(client, args, as_json=as_json)
            if isinstance(data, int):
                return data
        else:
            raise CLIError(f"unknown connections command: {command}")
    except GatewayHTTPError as exc:
        return _connection_error_exit(exc, as_json=as_json)
    _print_connection_result(data, as_json=as_json)
    return 0


def _connection_timeouts() -> dict[str, float]:
    """Client-side waits, generous enough to outlast the SERVER's own bounds.

    The authoritative bounds are the owner settings the gateway reads
    (``config.get_ssh_timeout_sec``); a CLI that gave up first would report a
    transport failure for an operation that actually succeeded.
    """

    try:
        from ouroboros.config import get_ssh_timeout_sec

        return {
            "test": get_ssh_timeout_sec("connect") + 10.0,
            "bootstrap": get_ssh_timeout_sec("bootstrap") + 30.0,
            "reconnect": get_ssh_timeout_sec("reconcile") + 10.0,
        }
    except Exception:
        return {"test": 30.0, "bootstrap": 150.0, "reconnect": 30.0}


def add_connections_parser(subparsers: argparse._SubParsersAction) -> None:
    connections = subparsers.add_parser(
        "connections",
        help="manage owner-authorized SSH connection metadata",
    )
    sub = connections.add_subparsers(dest="connections_command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=connections_command)
    add = sub.add_parser("add")
    add.add_argument("--name", required=True)
    add.add_argument("--ssh-alias", required=True, dest="ssh_alias")
    add.add_argument("--json", action="store_true")
    add.set_defaults(func=connections_command)
    for command in ("test", "bootstrap", "reconnect", "retrust", "retire"):
        action = sub.add_parser(command)
        action.add_argument("connection_id")
        action.add_argument("--json", action="store_true")
        action.set_defaults(func=connections_command)


#: THE exit-code taxonomy for every typed gateway refusal a thin owner CLI renders.
#: ``cli_projects`` shares it rather than restating it: two tables for one set of
#: refusal codes would drift, and the codes are the contract a script depends on.
render_typed_error_exit = _connection_error_exit

__all__ = [
    "add_connections_parser",
    "connections_command",
    "render_typed_error_exit",
]
