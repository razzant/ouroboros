"""CLI commands for managing linked Ouroboroses over the local owner HTTP API.

Commands:
- ``ouroboros link invite [--remote] [--own-url URL] [--label LABEL] [--own-label LABEL] [--note NOTE]
  [--may-start/--no-may-start] [--json]``
- ``ouroboros link confirm <link_id> [--json]``
- ``ouroboros link show [<link_id>] [--json]``
- ``ouroboros link revoke <link_id> [--json]``

Every command is non-interactive, supports ``--json``, calls only the installation's existing owner
HTTP doors, and returns a non-zero exit code on refusal or transport error.
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
from typing import Any

from ouroboros.cli import CLIError, ConnectionCLIError, StreamInterruptedCLIError, _client, _print_json


def _emit_error(args: argparse.Namespace, *, code: str, detail: str, status: int = 0,
                payload: dict[str, Any] | None = None, exit_code: int = 1) -> int:
    if getattr(args, "json", False):
        body: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {}
        body.setdefault("error", code)
        body.setdefault("refusal_code", str(body.get("refusal_code") or code))
        if detail and not body.get("detail"):
            body["detail"] = detail
        if status:
            body["status"] = status
        body["ok"] = False
        _print_json(body)
    else:
        suffix = f": {detail}" if detail and detail != code else ""
        print(f"error: {code}{suffix}", file=sys.stderr)
    return exit_code


def _call_json(args: argparse.Namespace, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[Any, int]:
    sent_request = False
    try:
        client = _client(args)
        sent_request = True
        return client.request(method, path, body), 0
    except StreamInterruptedCLIError as exc:
        status = int(getattr(exc, "status_code", 0) or 0)
        mutating = sent_request and method.upper() != "GET"
        detail = (f"{exc}; whether the server completed this {method.upper()} is unknown" if mutating else str(exc))
        payload = {"outcome": "unknown"} if mutating else None
        return None, _emit_error(args, code="response_interrupted", detail=detail, status=status, payload=payload, exit_code=2)
    except ConnectionCLIError as exc:
        return None, _emit_error(args, code="server_unreachable", detail=str(exc), exit_code=2)
    except CLIError as exc:
        payload = getattr(exc, "payload", None)
        status = int(getattr(exc, "status_code", 0) or 0)
        if isinstance(payload, dict):
            code = str(payload.get("refusal_code") or payload.get("error") or "request_failed")
            detail = str(payload.get("detail") or payload.get("error") or str(exc))
            return None, _emit_error(args, code=code, detail=detail, status=status, payload=payload, exit_code=1)
        return None, _emit_error(args, code="request_failed", detail=str(exc), status=status, exit_code=1)


def _link_invite_command(args: argparse.Namespace) -> int:
    body: dict[str, Any] = {"may_start": bool(args.may_start)}
    if args.remote:
        body["remote"] = True
    elif args.own_url:
        body["own_url"] = args.own_url
    if args.label:
        body["label"] = args.label
    if args.own_label:
        body["own_label"] = args.own_label
    if args.note:
        body["note"] = args.note
    data, err = _call_json(args, "POST", "/api/peer-links/invite", body)
    if err:
        return err
    if args.json:
        _print_json(data)
    else:
        print(f"link_id={data.get('link_id', '')}")
        if data.get("door_dir"):
            print(f"door_dir={data.get('door_dir', '')}")
        print(f"expires_at={data.get('expires_at', '')}")
        print(data.get("invite", ""))
    return 0


def _link_confirm_command(args: argparse.Namespace) -> int:
    quoted = urllib.parse.quote(str(args.link_id), safe="")
    data, err = _call_json(args, "POST", f"/api/peer-links/{quoted}/confirm", {})
    if err:
        return err
    if args.json:
        _print_json(data)
    else:
        print(f"link_id={data.get('link_id', '')} code={data.get('code', '')} "
              f"code_check={data.get('code_check', '')} state={data.get('state', '')}")
    return 0


def _link_revoke_command(args: argparse.Namespace) -> int:
    quoted = urllib.parse.quote(str(args.link_id), safe="")
    data, err = _call_json(args, "POST", f"/api/peer-links/{quoted}/revoke", {})
    if err:
        return err
    if args.json:
        _print_json(data)
    else:
        print(f"link_id={data.get('link_id', '')} state={data.get('state', '')} "
              f"revoked_at={data.get('revoked_at', '')}")
    return 0


def _place_link_entry(place: dict[str, Any]) -> dict[str, Any] | None:
    link = place.get("link") if isinstance(place.get("link"), dict) else None
    link_id = str(link.get("link_id") or "") if link is not None else ""
    if not link_id:
        return None
    entry: dict[str, Any] = {
        "link_id": link_id,
        "place_id": str(place.get("place_id") or ""),
        "peer_label": str(place.get("label") or ""),
        "code": str(link.get("code") or ""),
        "code_check": str(link.get("code_check") or ""),
        "state": "removed" if link.get("revoked_at") else str(place.get("state") or ""),
        "peer_install_id": str(link.get("peer_install_id") or ""),
        "created_at": str(link.get("created_at") or ""),
        "revoked_at": str(link.get("revoked_at") or ""),
        "remote_may_start": link.get("remote_may_start"),
        "remote_unknown_at": str(link.get("remote_unknown_at") or ""),
        "peer": place.get("peer"),
        "peer_effective": place.get("peer_effective"),
    }
    if link.get("transport"):
        entry["transport"] = str(link.get("transport") or "")
        entry["channel_role"] = str(link.get("channel_role") or "")
        entry["channel_session"] = str(link.get("channel_session") or "")
    return entry


def _link_show_command(args: argparse.Namespace) -> int:
    link_id = str(getattr(args, "link_id", "") or "").strip()
    if link_id:
        quoted = urllib.parse.quote(link_id, safe="")
        read_data, err = _call_json(args, "GET", f"/api/peer-links/{quoted}")
        if err:
            return err
        link_row = read_data.get("link") if isinstance(read_data, dict) else None
        if not isinstance(link_row, dict):
            return _emit_error(args, code="link_unknown", detail=f"no such link: {link_id}", status=404, exit_code=1)
        merged = dict(link_row)
        places_data, places_err = _call_json(args, "GET", "/api/places")
        if places_err:
            return places_err
        if isinstance(places_data, dict):
            for place in places_data.get("places") or ():
                if isinstance(place, dict) and place.get("connector") == "ouroboros":
                    candidate = _place_link_entry(place)
                    if candidate is not None and (candidate["link_id"] == link_id or candidate["place_id"] == merged.get("place_id")):
                        merged = {**candidate, **merged, "code": merged.get("code") or candidate.get("code") or "",
                                  "code_check": merged.get("code_check") or candidate.get("code_check") or ""}
                        break
        if args.json:
            _print_json({"link": merged})
        else:
            print(f"link_id={merged.get('link_id', '')} label={merged.get('peer_label', '')} "
                  f"code={merged.get('code', '')} code_check={merged.get('code_check', '')} "
                  f"state={merged.get('state', '')}")
        return 0

    places_data, err = _call_json(args, "GET", "/api/places")
    if err:
        return err
    raw_places = [
        place for place in (places_data.get("places") or () if isinstance(places_data, dict) else ())
        if isinstance(place, dict) and place.get("connector") == "ouroboros"
    ]
    if any((place.get("peer_effective") or {}).get("reason") == "unknown" for place in raw_places):
        return _emit_error(
            args,
            code="peer_links_unreadable",
            detail="the link registry cannot be read now",
            status=503,
            exit_code=1,
        )
    links = [entry for place in raw_places if (entry := _place_link_entry(place)) is not None]
    if args.json:
        _print_json({"links": links})
    else:
        if not links:
            print("(no linked Ouroboros)")
        for item in links:
            print(f"{item['link_id']}  {item['peer_label']}  code={item['code']}  "
                  f"code_check={item['code_check']}  state={item['state']}")
    return 0


def add_link_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``ouroboros link ...`` subcommands on the root CLI parser."""
    link = subparsers.add_parser("link", help="manage links to another Ouroboros")
    sub = link.add_subparsers(dest="link_command", required=True)

    invite = sub.add_parser("invite", help="create a one-time invitation for another Ouroboros")
    invite.add_argument("--remote", action="store_true",
                        help="create a cross-machine (v2) invitation using this installation's letters door")
    invite.add_argument("--own-url", default="", help="explicit local loopback URL for same-computer (v1) invite")
    invite.add_argument("--label", default="", help="name for the coming Ouroboros here")
    invite.add_argument("--own-label", default="", help="name this Ouroboros presents in the invitation")
    invite.add_argument("--note", default="", help="owner rule/description for the linked place")
    invite.add_argument("--may-start", action=argparse.BooleanOptionalAction, default=False,
                        help="whether the linked Ouroboros may start work here (default: --no-may-start)")
    invite.add_argument("--json", action="store_true", help="output JSON")
    invite.set_defaults(func=_link_invite_command)

    confirm = sub.add_parser("confirm", help="confirm the six-digit code for a link")
    confirm.add_argument("link_id", help="link identifier")
    confirm.add_argument("--json", action="store_true", help="output JSON")
    confirm.set_defaults(func=_link_confirm_command)

    show = sub.add_parser("show", help="show one link (including pending code) or list linked Ouroboroses")
    show.add_argument("link_id", nargs="?", default="", help="optional link identifier")
    show.add_argument("--json", action="store_true", help="output JSON")
    show.set_defaults(func=_link_show_command)

    revoke = sub.add_parser("revoke", help="revoke a link on this installation")
    revoke.add_argument("link_id", help="link identifier")
    revoke.add_argument("--json", action="store_true", help="output JSON")
    revoke.set_defaults(func=_link_revoke_command)
