"""``ouroboros projects list/rebind`` — the caller the rebind path did not have.

A remote Project's placement is REBINDABLE by design: `POST /api/projects/{id}/update`
takes the same two halves a create takes, admits them the same way, and advances
``routing_generation`` so work already resolved against the previous target is refused
at queue insertion instead of run there (``projects_registry.set_project_placement``,
``supervisor/queue.py``). Every layer of that existed — endpoint, compare-and-set,
contract, browser client method — and nothing called it: the only call site passed a
name. So after retiring a connection there was no way to move its Projects anywhere,
while Settings → Connections told the owner they "keep their binding … until they are
rebound".

This is the narrow honest path. A CLI command rather than a second dialog, because the
New Project picker's connection chooser would have to be duplicated to put a rebind in
the sidebar menu, and a duplicated picker is a second place that decides which
connections are selectable.

Like ``cli_connections``, this module holds no logic: flags in, one endpoint call, the
typed answer rendered out. It shares that module's exit-code taxonomy rather than
restating it — two tables for one set of gateway refusals would drift.
"""

from __future__ import annotations

import argparse
import urllib.parse
from typing import Any


def _print_project_result(data: Any, *, as_json: bool) -> None:
    from ouroboros.cli import _print_json

    if as_json:
        _print_json(data)
        return
    payload = data if isinstance(data, dict) else {}
    rows = payload.get("projects") if isinstance(payload.get("projects"), list) else None
    if rows is None:
        entry = payload.get("project") if isinstance(payload.get("project"), dict) else payload
        rows = [entry]
    for row in rows:
        if not isinstance(row, dict):
            continue
        placement = row.get("placement") if isinstance(row.get("placement"), dict) else {}
        # Where the project lives, in ONE column, because "local dir" and "remote
        # root on a connection" are the two answers and a row that shows neither is
        # exactly the row an owner is trying to repair.
        where = (
            f"ssh:{placement.get('connection_id', '')}:{placement.get('remote_root', '')}"
            if placement.get("kind") == "ssh"
            else str(row.get("working_dir") or "")
        )
        print(
            "\t".join(str(value) for value in (
                row.get("id", ""),
                row.get("name", ""),
                row.get("lifecycle", "active"),
                f"gen={row.get('routing_generation', 0)}",
                where,
            ))
        )


def projects_command(args: argparse.Namespace) -> int:
    from ouroboros.cli import CLIError, GatewayHTTPError
    from ouroboros.cli_connections import render_typed_error_exit

    command = str(getattr(args, "projects_command", "") or "")
    as_json = bool(getattr(args, "json", False))
    from ouroboros.cli import OuroborosHTTPClient

    client = OuroborosHTTPClient(getattr(args, "url", "") or "")
    try:
        if command == "list":
            data = client.request("GET", "/api/projects")
        elif command == "rebind":
            # safe="": a project id is ONE path segment, so a slash in it must be
            # escaped rather than silently becoming a different route.
            quoted = urllib.parse.quote(str(args.project_id), safe="")
            data = client.request(
                "POST",
                f"/api/projects/{quoted}/update",
                # BOTH halves, always: the gateway refuses half a placement rather
                # than guessing the other half, and the target allocates the third
                # field (the workspace identity) at admission.
                {"connection_id": args.connection, "remote_root": args.remote_root},
            )
        else:
            raise CLIError(f"unknown projects command: {command}")
    except GatewayHTTPError as exc:
        return render_typed_error_exit(exc, as_json=as_json)
    _print_project_result(data, as_json=as_json)
    return 0


def add_projects_parser(subparsers: argparse._SubParsersAction) -> None:
    projects = subparsers.add_parser(
        "projects",
        help="list projects and rebind a remote placement",
    )
    sub = projects.add_subparsers(dest="projects_command", required=True)
    listing = sub.add_parser("list", help="list projects with their placement")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=projects_command)
    rebind = sub.add_parser(
        "rebind",
        help="move a project's remote placement to another connection/root",
    )
    rebind.add_argument("project_id")
    rebind.add_argument("--connection", required=True, dest="connection")
    rebind.add_argument("--remote-root", required=True, dest="remote_root")
    rebind.add_argument("--json", action="store_true")
    rebind.set_defaults(func=projects_command)


__all__ = [
    "add_projects_parser",
    "projects_command",
]
