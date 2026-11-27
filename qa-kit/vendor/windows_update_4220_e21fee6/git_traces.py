"""Which Git ran and what it did, read only from the trace sinks Git itself writes into the app's environment.

Stdlib only and free of Windows calls, so every reading below can be checked anywhere on synthetic text.
What the pinned Git writes (git-for-windows v2.56.1.windows.1, read at source):

- Trace1 (``GIT_TRACE``: one absolute file every process appends to). At start each Git process writes one
  ``trace: resolved executable dir: <folder>`` line, the folder of its own image as Windows reports it
  (``_wpgmptr``). When that lookup fails Git takes ``argv[0]`` and says so first
  (``trace: resolved executable path from argv0:``), or writes ``trace: could not determine executable path
  from:``. The line names no process.
- Trace2 (``GIT_TRACE2_EVENT``: a folder, one file per process). Events keyed by ``sid``, each with its UTC
  ``time``: ``start`` with ``argv``, ``cmd_name``, ``exit``/``atexit`` with ``code``, ``signal``. ``cmd_path`` never
  arrives: Git computes it before Trace2 is initialised.

So identity is a bounded claim over the whole run, never one session's path: as many identity lines as started
sessions (each process writes one of each), none taken from ``argv[0]``, every line inside the bundled folder and
every executable there byte-equal to the package. Commands are judged per time window: a command counts when its
session starts inside the window and every terminal event has code 0 and its own instant inside it too; the apply
pair counts only with its exact arguments. A window whose bounds were not recorded is not evaluated, never widened,
and an instant without a zone is unreadable, never taken for UTC.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Iterable

# trace.c prepare_trace_line: "HH:MM:SS.uuuuuu <file>:<line>" padded to 40 columns, then the message.
_TRACE1 = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{6} \S+?:\d+ *trace: (?P<what>resolved executable dir|resolved executable path "
    r"from argv0|could not determine executable path from): ?(?P<value>.*)$"
)
_SAFE_CMD = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
# What each window must hold, by Trace2 command name; apply's pair by its exact argument tail (``apply_argv``).
REQUIRED = {"bootstrap": ("clone",), "check": ("fetch",), "apply": ("checkout", "reset")}


def apply_argv(target: str) -> dict[str, list[str]]:
    """How each apply step must end (supervisor/update_merge.py ``apply_managed_merge_update``, branch
    ``ouroboros``): ``checkout -B ouroboros <B>`` and ``reset --hard <B>``. Its ``reset --hard HEAD`` and any other
    target do not count."""
    return {"checkout": ["checkout", "-B", "ouroboros", target], "reset": ["reset", "--hard", target]}


def when(text: Any) -> datetime | None:
    """An ISO-8601 instant with its zone (Trace2's ``...Z`` or Python's ``+00:00``) as an aware datetime; None when
    unreadable or without a zone: none is assumed."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        moment = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", text.strip().replace("Z", "+00:00")))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def relative(path: str, root: str) -> str | None:
    """``path`` below ``root`` as a POSIX relative path (slashes and case of Windows paths ignored), else None."""
    clean = [re.sub(r"[\\/]+", "/", str(x or "").strip().strip('"').removeprefix("\\\\?\\")).rstrip("/")
             for x in (path, root)]
    if not all(clean) or not clean[0].lower().startswith(clean[1].lower() + "/"):
        return None
    parts = [part for part in clean[0][len(clean[1]) + 1:].split("/") if part and part != "."]
    return "/".join(parts) if parts and ".." not in parts else None


def identity_lines(text: str) -> dict[str, Any]:
    """The folders the Trace1 identity lines name, with how many processes named each; no other line is kept."""
    folders: Counter[str] = Counter()
    counts = {"argv0_fallback": 0, "unresolved": 0, "unreadable": 0}
    for line in text.splitlines():
        if not (found := _TRACE1.match(line)):
            continue
        what, value = found["what"], found["value"].strip()
        if what == "resolved executable dir" and value and "�" not in value:
            folders[value] += 1
        else:
            counts["unreadable" if what == "resolved executable dir" else
                   "argv0_fallback" if "argv0" in what else "unresolved"] += 1
    return {"folders": folders, **counts}


def sessions(texts: Iterable[str], *, discarded: bool = False) -> dict[str, Any]:
    """Trace2 sessions by ``sid`` (start instant and argv, command name, terminal codes with their instants, signal)
    and what could not be read. The argv stays in memory for ``window``; nothing publishes it."""
    found: dict[str, dict[str, Any]] = {}
    malformed = 0
    for text in texts:
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                event = None
            kind = event.get("event") if isinstance(event, dict) else None
            if kind == "too_many_files":
                discarded = True
                continue
            if not isinstance(kind, str) or not isinstance(event.get("sid"), str) or not event["sid"]:
                malformed += 1
                continue
            row = found.setdefault(event["sid"], {"start": None, "argv": [], "cmd": "", "ends": [], "signal": False})
            if kind == "start":
                row["start"], argv = when(event.get("time")), event.get("argv")
                row["argv"] = [str(part) for part in argv] if isinstance(argv, list) else []
            elif kind == "cmd_name" and not row["cmd"]:
                name = event.get("name")
                row["cmd"] = name if isinstance(name, str) and _SAFE_CMD.match(name) else "_other_"
            elif kind in ("exit", "atexit") and type(event.get("code")) is int:
                row["ends"].append((event["code"], when(event.get("time"))))
            elif kind == "signal":
                row["signal"] = True
    return {"sessions": found, "malformed": malformed, "discarded": discarded}


def window(found: dict[str, Any], start: datetime | None, end: datetime | None, *,
           target: str = "") -> dict[str, Any] | None:
    """Commands of the sessions that started inside ``[start, end]``; None when a bound was not recorded. A session
    succeeded when it has terminal events, each with code 0 and its instant inside the window, and no signal; one
    that ended outside it, unread or not at all is incomplete. With ``target``, ``to_target`` counts the successes
    whose argv after argv[0] ends as ``apply_argv`` (argv[0], the program, is never read)."""
    if start is None or end is None or end < start:
        return None
    done: dict[str, Counter[str]] = {"succeeded": Counter(), "nonzero": Counter(), "to_target": Counter()}
    incomplete = 0
    tails = apply_argv(target) if target else {}
    for row in found["sessions"].values():
        if row["start"] is None or not start <= row["start"] <= end:
            continue
        name = row["cmd"] or "_unnamed_"
        if row["signal"] or not row["ends"] or any(at is None or not start <= at <= end for _code, at in row["ends"]):
            incomplete += 1
        elif any(code != 0 for code, _at in row["ends"]):
            done["nonzero"][name] += 1
        else:
            done["succeeded"][name] += 1
            tail = tails.get(row["cmd"])
            if tail and row["argv"][1:][-len(tail):] == tail:
                done["to_target"][name] += 1
    return {"from": start.isoformat(), "to": end.isoformat(), "incomplete": incomplete,
            **{key: dict(sorted(counts.items())) for key, counts in done.items()}}


def identity(lines: dict[str, Any], found: dict[str, Any], *, package_root: str, git_root: str,
             package_exes: dict[str, str], folder_exes: Callable[[str], dict[str, str]]) -> dict[str, Any]:
    """The bounded identity claim. ``package_exes``: package-relative POSIX path -> SHA-256 from the ZIP;
    ``folder_exes(rel_folder)``: name -> SHA-256 of every ``.exe`` now on disk in that package folder.

    ``state``, first match wins: ``foreign`` (a line names a folder outside the bundled Git); ``not_the_package``
    (an executable in a named folder is missing from, or differs from, the ZIP); ``not_evaluated`` (no line or no
    session: an empty set proves nothing); ``unconfirmed`` (counts differ, an ``argv[0]`` or unresolved line, a
    session without a readable start, a malformed or discarded trace); else ``verified``."""
    started = sum(1 for row in found["sessions"].values() if row["start"] is not None)
    total = sum(lines["folders"].values())
    inside: Counter[str] = Counter()
    foreign = 0
    for folder, count in lines["folders"].items():
        rel = relative(folder, package_root)
        if rel is None or relative(folder, git_root) is None:
            foreign += count
        else:
            inside[rel] += count
    expected = {path.lower(): digest.lower() for path, digest in package_exes.items()}
    matching = {}
    for rel in sorted(inside):
        on_disk = folder_exes(rel)
        matching[rel] = bool(on_disk) and all(expected.get(f"{rel}/{name}".lower()) == digest.lower()
                                              for name, digest in on_disk.items())
    gaps = {"argv0_fallback_lines": lines["argv0_fallback"], "unresolved_lines": lines["unresolved"],
            "unreadable_lines": lines["unreadable"], "sessions_without_start": len(found["sessions"]) - started,
            "malformed_trace2_lines": found["malformed"]}
    state = ("foreign" if foreign else "not_the_package" if not all(matching.values()) else
             "not_evaluated" if not total or not started else
             "unconfirmed" if any(gaps.values()) or found["discarded"] or total != started else "verified")
    return {"state": state, "identity_lines": total, "started_sessions": started, "foreign_lines": foreign,
            "bundled_folders": dict(inside), "folders_match_package": matching,
            "trace2_discarded": found["discarded"], **gaps}


def control_passed(claim: dict[str, Any], found: dict[str, Any]) -> bool:
    """The package's own ``git version`` left exactly one identity line and one complete ``version`` session."""
    rows = list(found["sessions"].values())
    return (claim["state"] == "verified" and claim["identity_lines"] == 1 and len(rows) == 1
            and rows[0]["cmd"] == "version" and bool(rows[0]["ends"]) and {code for code, _at in rows[0]["ends"]} == {0}
            and not rows[0]["signal"])


def bounds(marks: dict[str, str], attempt: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """The three windows: launch to Setup (first-run clone), then either the plain check's click to "Update to"
    or (for an already offered B) "Update to" to "Update now" (preflight), then the carried apply from ``begin`` to the
    ``restart_requested`` stage its executor recorded after checkout, reset and smoke (gateway/control.py
    ``_restart_response``), and only when that attempt began after the recorded "Update now" click. Anything else
    leaves apply unbounded: it is never widened to the next boot."""
    clicked, begun = when(marks.get("update_now_clicked")), when(attempt.get("started_at"))
    applied = (attempt.get("stage") == attempt.get("result") == "restart_requested"
               and clicked is not None and begun is not None and begun >= clicked)
    already_offered = "development_offer_observed" in marks and "plain_check_clicked" not in marks
    fetch_window = ({"preflight": (marks.get("update_to_clicked"), marks.get("update_now_clicked"))} if already_offered
                    else {"check": (marks.get("plain_check_clicked"), marks.get("update_to_clicked"))})
    return {"bootstrap": (marks.get("launched"), marks.get("setup_window_ready")), **fetch_window,
            "apply": (attempt.get("started_at"), attempt.get("stage_started_at")) if applied else (None, None)}


def phase_requirements(windows: dict[str, dict[str, Any] | None]) -> dict[str, bool | None]:
    """Each required command succeeded at least once in its window; None for a window not evaluated."""
    commands_by_phase = {("preflight" if phase == "check" and "preflight" in windows and "check" not in windows else phase): commands
                         for phase, commands in REQUIRED.items()}
    return {phase: None if windows.get(phase) is None
            else all(windows[phase]["to_target" if phase == "apply" else "succeeded"].get(cmd, 0) >= 1
                     for cmd in commands)
            for phase, commands in commands_by_phase.items()}


def completed_check(status: Any, *, target: str, development: bool, after: str = "") -> datetime | None:
    """When the passive update status shows a completed check of ``target`` on that channel, later than ``after``
    (an instant, "" for none); None otherwise. The status keeps ``checked_at`` and ``checked_target_sha`` even for
    "no update" (supervisor/git_ops_updates.py); an unreadable or zone-less time proves nothing."""
    body = status if isinstance(status, dict) else {}
    moment, floor = when(body.get("checked_at")), when(after)
    if moment is None or body.get("checked_target_sha") != target or (after and floor is None):
        return None
    on_channel = body.get("update_channel") == ("development" if development else "stable")
    return moment if on_channel and (floor is None or moment > floor) else None


def judge(sinks: tuple[str, list[str], bool], bounds: dict[str, tuple[Any, Any]], *, control_ok: bool, target: str,
          **where: Any) -> tuple[dict[str, Any], tuple[str, str, dict[str, Any]] | None]:
    """The published summary of one read of the app's sinks (Trace1 text, Trace2 texts, discard sentinel) and the
    outcome it settles for a completed run: ``(verdict, code, fields)``, None when everything is confirmed.
    ``bounds``: window -> (first, last) instants as text, a missing one leaving that window not evaluated;
    ``target``: B, the apply pair's argument; ``where``: the package arguments of ``identity``. A foreign or altered
    Git is a failure; anything unproven, the instrument's own failed control included, is unconfirmed."""
    found = sessions(sinks[1], discarded=sinks[2])
    claim = identity(identity_lines(sinks[0]), found, **where)
    windows = {phase: window(found, when(first), when(last), target=target if phase == "apply" else "")
               for phase, (first, last) in bounds.items()}
    reqs = phase_requirements(windows)
    summary = {"scope": "git_processes_inheriting_the_app_environment", "system_wide_process_exclusion_claimed": False,
               "identity": claim, "control_passed": control_ok, "requirements": reqs,
               "windows": {phase: found_window or "not_evaluated" for phase, found_window in windows.items()}}
    if claim["state"] in ("foreign", "not_the_package"):
        return summary, ("failed", f"git_identity_{claim['state']}", {"foreign_lines": claim["foreign_lines"]})
    if not control_ok:
        return summary, ("unconfirmed", "git_identity_control_failed", {})
    if claim["state"] != "verified":
        return summary, ("unconfirmed", f"git_identity_{claim['state']}",
                         {"lines": claim["identity_lines"], "sessions": claim["started_sessions"]})
    if not all(reqs.values()):
        return summary, ("unconfirmed", "git_phase_operations_unconfirmed", reqs)
    return summary, None
