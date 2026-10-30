"""Place: who is in front, bringing an application forward, window rows, bounded element reads.

Mixed into the plugin's ``_ComputerUse``. The plugin supplies the seams (``_plat``, ``_session``,
``_which_tool``, ``_json_text``, ``_try``, ``_helper_python``, ``_backend_unavailable``,
``_disabled_connection_error``, ``_is_remote``, ``_active_connection``) and the remote mixin the
remote readers (``_remote_activate_dict``, ``_remote_window_list``, ``_remote_ax_tree``).

Contract (unix_computer_use SKILL.md, "Input receipts"): an input aimed at an application is sent
only after that application was brought forward and read as frontmost; the receipt states who was
in front when the input was dispatched (``delivered_to``), who was there before (``front_before``,
when it differs), who is there afterwards (``front_after``, when it differs or the dispatch front was
unknown), and what the tool did not observe. macOS facts come from a short-lived helper process
(``macos_native.py``); X11 facts from ``wmctrl``/``xdotool``; Wayland is an honest refusal. The same
receipt composer (``_finish``) serves the local, X11, OSWorld and SSH transports.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import time
from typing import Any

from .macos_native import parse_info

_UNREAD = object()  # "read the front yourself" marker for _finish


def parse_wmctrl_rows(text: str) -> list[dict[str, Any]]:
    """``wmctrl -lpxG`` rows: window_id, desktop, pid, bounds, app (WM_CLASS ``instance.Class``), title."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split(None, 9)
        if len(parts) < 9:
            continue
        try:
            rows.append({"window_id": parts[0], "desktop": int(parts[1]), "pid": int(parts[2]),
                         "bounds": [int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])],
                         "app": parts[7], "title": parts[9] if len(parts) > 9 else ""})
        except ValueError:
            continue
    return rows


def window_id_int(value: Any) -> int | None:
    """An X11 window id as an int: ``0x0320000a`` (wmctrl) or ``52428810`` (xdotool)."""
    text = str(value or "").strip().lower()
    try:
        return int(text, 16) if text.startswith("0x") else int(text)
    except ValueError:
        return None


def parse_x11_front(text: str, rows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """``<window id>|<pid>|<title>`` (what the guest/X11 front readers print) -> {window_id, pid, title, app}.

    ``app`` (the WM_CLASS) comes from the matching ``wmctrl`` row when rows are given. None = unread."""
    raw = str(text or "").strip()
    wid_text, _sep, rest = raw.partition("|")
    wid = window_id_int(wid_text)
    if wid is None:
        return None
    front: dict[str, Any] = {"window_id": f"0x{wid:08x}"}
    pid_text, _sep, title = rest.partition("|")
    if pid_text.strip().isdigit():
        front["pid"] = int(pid_text.strip())
    if title.startswith("b64:"):
        try:
            title = base64.b64decode(title[4:].strip() + "=" * (-len(title[4:].strip()) % 4)).decode("utf-8", "replace")
        except (ValueError, UnicodeError):
            title = ""
    if title:
        front["title"] = title
    for row in rows or []:
        if window_id_int(row.get("window_id")) == wid:
            front["app"] = row.get("app")
            front.setdefault("pid", row.get("pid"))
            front.setdefault("title", row.get("title"))
            break
    return front


def class_matches(wm_class: str, query: str) -> bool:
    """Whether ``query`` names this ``wmctrl`` WM_CLASS (``instance.Class``): the whole string, its instance (the
    part before the first dot) or its class (the part after a dot, so ``Firefox`` names ``navigator.Firefox`` and
    ``Gimp-2.10`` names ``gimp-2.10.Gimp-2.10``). Never a bare fragment between dots."""
    whole, low = wm_class.lower(), query.lower()
    return bool(low) and (low == whole or whole.startswith(low + ".") or whole.endswith("." + low))


def x11_resolve(query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which window(s) a name means, from ``wmctrl`` rows: ``pid:<n>``, a window id, a WM_CLASS (whole, instance or
    class part) or an exact title. Substring hits are never a target; they are returned as ``candidates`` so the
    caller can name one exactly. The target carries ``window_ids``: the windows that ARE the target (one for a title
    or id, every window of that class for a class, every window of the process for ``pid:``), which is what every
    later "is it in front" check compares against."""
    q = str(query or "").strip()
    low = q.lower()

    def candidate(r: dict[str, Any]) -> dict[str, Any]:
        return {"app": r["app"], "pid": r["pid"], "window_id": r["window_id"], "title": r["title"]}

    running = sorted({r["app"] for r in rows})
    if low.startswith("pid:"):
        digits = q[4:].strip()
        by, matches = "pid", [r for r in rows if digits.isdigit() and r["pid"] == int(digits)]
    elif low.startswith("0x"):
        wid = window_id_int(low)
        by, matches = "window_id", [r for r in rows if wid is not None and window_id_int(r["window_id"]) == wid]
    else:
        by, matches = "class", [r for r in rows if class_matches(r["app"], low)]
        if not matches:
            by, matches = "title", [r for r in rows if low == r["title"].lower()]
        if not matches:
            hints = [r for r in rows if low in r["app"].lower() or low in r["title"].lower()]
            out: dict[str, Any] = {"ok": False, "reason": "not_running", "running": running[:40]}
            if hints:
                out["candidates"] = [candidate(r) for r in hints[:10]]
            return out
    if not matches:
        return {"ok": False, "reason": "not_running", "running": running[:40]}
    # A window named by its title or id is that window; a class is every window of that class (a process may host
    # several classes: LibreOffice's Writer and Calc); ``pid:`` is every window of the process. Two different
    # applications (classes) or two same-titled windows are an ambiguity the caller resolves.
    if by == "class":
        identities = {r["app"].lower() for r in matches}
    elif by == "title":
        identities = {r["window_id"] for r in matches}
    else:
        identities = {r["pid"] or r["window_id"] for r in matches}
    if len(identities) > 1:
        return {"ok": False, "reason": "ambiguous", "candidates": [candidate(r) for r in matches[:10]]}
    return {"ok": True, "target": {**candidate(matches[0]), "matched_by": by,
                                   "window_ids": [r["window_id"] for r in matches]}}


def x11_same(front: dict[str, Any] | None, target: dict[str, Any]) -> bool:
    """Whether the front window is the target: its window id is one of the target's ``window_ids`` (the windows the
    resolver matched). A front read without a window id falls back to the process id, and a missing pid is unknown,
    never equal."""
    if not front:
        return False
    ids = {window_id_int(w) for w in (target.get("window_ids") or [target.get("window_id")]) if w is not None}
    if front.get("window_id") is not None and ids:
        return window_id_int(front.get("window_id")) in ids
    return bool(front.get("pid")) and front.get("pid") == target.get("pid")


def same_place(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Whether two front reads name the same place: the same window when both carry a window id, else the same pid."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    if a.get("window_id") is not None and b.get("window_id") is not None:
        return window_id_int(a.get("window_id")) == window_id_int(b.get("window_id"))
    return a.get("pid") is not None and a.get("pid") == b.get("pid")


class PlaceMixin:
    """Target resolution, activation, place receipts, window rows and bounded element reads."""

    def _native(self, op: str, *args: Any, timeout: int = 20) -> dict[str, Any]:
        """One fresh macOS native fact from the helper process (``macos_native.py``).

        A separate short-lived process so that a stuck or crashing native call can be killed
        and never takes the host down, and so that every answer is read anew (AppKit caches
        refresh only on a main run loop the server does not turn)."""
        cmd = [self._helper_python(), "-I", str(pathlib.Path(__file__).with_name("macos_native.py")), op, *[str(a) for a in args]]
        rc, out, err = self._try(cmd, timeout=timeout)
        for line in reversed(out.strip().splitlines()):
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                except ValueError:
                    break
                if isinstance(parsed, dict):
                    return parsed
                break
        return {"ok": False, "error": f"native helper gave no answer ({(err or out).strip()[:200] or f'exit {rc}'})", "op": op,
                "no_answer": True}

    def _front_local(self, rows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Who is in front NOW, read fresh each call: macOS via ``lsappinfo``, X11 via ``xdotool getactivewindow``
        plus the ``wmctrl`` rows (or ``xdotool`` itself when wmctrl is absent); None = unknown."""
        plat = self._plat()
        if plat == "macos" and self._which_tool("lsappinfo"):
            rc, out, _err = self._try(["lsappinfo", "front"], timeout=5)
            asn = out.strip()
            if rc == 0 and asn:
                rc, out, _err = self._try(["lsappinfo", "info", "-only", "pid", "-only", "bundleid", "-only", "name", asn], timeout=5)
                info = parse_info(out) if rc == 0 else {}
                if info.get("pid"):
                    return info
            return None
        if plat == "linux" and self._session() != "wayland" and self._which_tool("xdotool"):
            rc, out, _err = self._try(["xdotool", "getactivewindow"], timeout=5)
            wid = out.strip()
            if rc != 0 or not wid.isdigit():
                return None
            if rows is None and self._which_tool("wmctrl"):
                rows, _err = self._x11_windows()
            front = parse_x11_front(wid, rows)
            if front and "pid" in front:
                return front
            rc, out, _err = self._try(["xdotool", "getwindowpid", wid], timeout=5)
            if rc == 0 and out.strip().isdigit():
                front["pid"] = int(out.strip())
            rc, out, _err = self._try(["xdotool", "getwindowname", wid], timeout=5)
            if rc == 0 and out.strip():
                front["title"] = out.strip()
            return front
        return None

    def _x11_windows(self) -> tuple[list[dict[str, Any]], str]:
        """``wmctrl -lpxG`` rows (see ``parse_wmctrl_rows``) and an error when the listing could not be read."""
        if not self._which_tool("wmctrl"):
            return [], "wmctrl is not installed"
        rc, out, err = self._try(["wmctrl", "-lpxG"], timeout=10)
        if rc != 0:
            return [], err.strip() or out.strip() or f"wmctrl exit {rc}"
        return parse_wmctrl_rows(out), ""

    def _x11_activate(self, query: str, wait_ms: int) -> dict[str, Any]:
        rows, err = self._x11_windows()
        if err:
            return {"ok": False, "reason": "unsupported", "error": err}
        res = x11_resolve(query, rows)
        if not res.get("ok"):
            return res
        target = res["target"]
        before = self._front_local(rows)
        if x11_same(before, target):
            return {"ok": True, "target": target, "front_before": before, "front": before, "activated": False,
                    "already_front": True, "verified": True, "waited_ms": 0}
        if not self._which_tool("xdotool"):
            return {"ok": False, "reason": "unsupported", "target": target, "front_before": before,
                    "error": "xdotool is not installed, so a switch cannot be verified"}
        rc, _out, err2 = self._try(["wmctrl", "-i", "-a", target["window_id"]], timeout=10)
        if rc != 0:
            return {"ok": False, "reason": "activate_failed", "error": err2.strip() or f"wmctrl exit {rc}", "target": target, "front_before": before}
        t0 = time.monotonic()
        front = self._front_local(rows)
        while not x11_same(front, target) and (time.monotonic() - t0) * 1000 < max(200, int(wait_ms)):
            time.sleep(0.05)
            front = self._front_local(rows)
        waited = int((time.monotonic() - t0) * 1000)
        out: dict[str, Any] = {"ok": True, "target": target, "front_before": before, "front": front, "method": "wmctrl -i -a",
                               "waited_ms": waited}
        if front is None:
            out.update({"ok": False, "activated": None, "verified": False, "reason": "front_unread",
                        "error": "the active window could not be read after the switch (xdotool getactivewindow gave nothing)"})
        elif x11_same(front, target):
            out.update({"activated": True if before else None, "verified": True})
        else:
            out.update({"ok": False, "activated": False, "verified": False, "reason": "not_frontmost_after_activate"})
        return out

    @staticmethod
    def _target_error(app: str, res: dict[str, Any]) -> str:
        reason = str(res.get("reason") or "")
        front = res.get("front") or {}
        where = f" ({front.get('app') or front.get('title')} is in front)" if isinstance(front, dict) and (front.get("app") or front.get("title")) else ""
        # No earlier step ran and no switch was asked for: raise_policy='never', or a target that was already in front.
        first = res.get("activated") is False and not res.get("before_step")
        if reason == "not_running":
            text = f"{app!r} is not a running application; nothing was sent (running: {', '.join(res.get('running') or [])[:300]})"
            if res.get("candidates"):
                names = "; ".join(f"{c.get('app')} {c.get('title')!r} pid:{c.get('pid')}" for c in res["candidates"][:6])
                text += f". Windows whose class or title merely contain it: {names}. Name the exact WM_CLASS, title or pid:<n>"
            return text
        if reason == "ambiguous":
            names = ", ".join(f"{c.get('app')} pid:{c.get('pid')}" for c in (res.get("candidates") or [])[:6])
            return f"{app!r} matches several applications ({names}); name one, e.g. pid:<n>; nothing was sent"
        if reason == "not_frontmost_after_activate":
            return (f"{app!r} did not come to the front{where}; nothing was sent. The OS can refuse activation "
                    "from a background process: ask the owner to click it, or use the app's own commands (run_command)")
        if reason == "front_unread":
            if res.get("before_step"):
                return f"{app!r}: who is in front could not be read before the next step ({res.get('error')}); nothing more was sent"
            if first:
                return f"{app!r}: who is in front could not be read; no switch was attempted and nothing was sent"
            return f"{app!r}: the switch was requested but who is in front could not be read ({res.get('error')}); nothing was sent"
        if reason == "target_lost":
            resolved = res.get("resolved") or {}
            state, sent = ("not", "no switch was attempted and nothing was sent") if first else ("no longer", "nothing more was sent")
            if resolved.get("window_ids") and front.get("app") and front.get("app") == resolved.get("app"):
                # The same class is in front through a window that is not the target's (a sibling, or a dialog a step opened).
                return (f"{app!r} is {state} in front: another window of its class, {front.get('title')!r} ({front.get('window_id')}), "
                        f"is in front and is not one of the target's windows ({', '.join(resolved.get('window_ids') or [])}); {sent}")
            return f"{app!r} is {state} in front{where}; {sent}"
        if reason == "unsupported":
            return f"application targeting is unavailable here: {res.get('error') or 'no window manager access'}; nothing was sent"
        if reason == "apps_unread":
            return f"the running applications could not be listed ({res.get('error')}); nothing was sent"
        return str(res.get("error") or f"{app!r} could not be targeted{where}; nothing was sent")

    def _ctx_from_activation(self, app: str, res: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """An activation result (helper, X11 or remote) -> (refusal, context). ``delivered_to`` is only ever the
        verified front read, never the requested target."""
        target: dict[str, Any] = {"requested": app, "resolved": res.get("target"), "activated": res.get("activated"),
                                  "verified": bool(res.get("verified"))}
        if res.get("already_front"):
            target["already_front"] = True
        if res.get("method"):
            target["method"] = res["method"]
        if res.get("waited_ms") is not None:
            target["waited_ms"] = res["waited_ms"]
        ctx: dict[str, Any] = {"target": target, "front_before": res.get("front_before")}
        if not res.get("ok"):
            refusal: dict[str, Any] = {"ok": False, "reason": str(res.get("reason") or "target_unavailable"),
                                       "error": self._target_error(app, res), "target": target}
            for key in ("candidates", "running", "running_more", "backend"):
                if res.get(key):
                    refusal[key] = res[key]
            if res.get("front") is not None:
                refusal["front"] = res["front"]
            return refusal, ctx
        ctx["delivered_to"] = res.get("front")
        return None, ctx

    def _begin(self, app: str, wait_ms: int = 2000, *, raise_policy: str = "allow",
               connection: tuple[str, dict[str, Any]] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Before an input: resolve and bring ``app`` forward (launch-free) and verify it is in front.

        Returns ``(refusal, context)``: a refusal sends nothing. With no ``app`` the context only records who
        is in front now (local backends), which is who will receive the input; remote backends read it in the
        same call as the input."""
        app = str(app or "").strip()
        if raise_policy not in ("allow", "never"):
            return {"ok": False, "reason": "invalid_raise_policy", "error": "raise_policy must be 'allow' or 'never'", "dispatched": False}, {}
        if raise_policy == "never" and not app:
            return {"ok": False, "reason": "app_required", "error": "raise_policy='never' requires app", "dispatched": False}, {}
        name, conn = self._connection_snapshot(connection)
        if conn.get("disabled"):
            return json.loads(self._disabled_connection_error(name, conn)), {}
        if name != "local":
            if not app:
                return None, {}
            return self._ctx_from_activation(app, self._remote_activate_dict(conn, app, int(wait_ms), raise_policy=raise_policy))
        ctx: dict[str, Any] = {}
        if not app:
            ctx["front_before"] = ctx["delivered_to"] = self._front_local()
            return None, ctx
        plat = self._plat()
        if plat == "macos":
            if raise_policy == "never":
                resolved = self._native("resolve", app, timeout=15)
                if not resolved.get("ok"):
                    res = resolved
                else:
                    target = (resolved.get("candidates") or [{}])[0]
                    front = self._front_local()
                    same = bool(front and front.get("pid") == target.get("pid"))
                    res = {"ok": same, "target": target, "front_before": front, "front": front,
                           "activated": False, "already_front": same, "verified": same,
                           "reason": "target_lost" if front else "front_unread", "resolved": target}
            else:
                res = self._native("activate", app, int(wait_ms), timeout=max(12, 3 * int(wait_ms) // 1000 + 10))
        elif plat == "linux" and self._session() != "wayland":
            if raise_policy == "never":
                rows, err = self._x11_windows()
                res = {"ok": False, "reason": "unsupported", "error": err} if err else x11_resolve(app, rows)
                if res.get("ok"):
                    target = res["target"]
                    front = self._front_local(rows)
                    same = x11_same(front, target)
                    res = {"ok": same, "target": target, "front_before": front, "front": front,
                           "activated": False, "already_front": same, "verified": same,
                           "reason": "target_lost" if front else "front_unread", "resolved": target}
            else:
                res = self._x11_activate(app, int(wait_ms))
        else:
            res = {"ok": False, "reason": "unsupported", "error": f"no application targeting on {plat}/{self._session()}"}
        return self._ctx_from_activation(app, res)

    def _recheck(self, ctx: dict[str, Any], *, later: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Before a later step of one call (a key sequence, ``later``) or a never-policy input's first step: read the
        front again; a targeted call stops, sending nothing more, when its application is not in front. Returns
        (refusal, the step's context)."""
        front = self._front_local()
        step = dict(ctx)
        step["delivered_to"] = front
        target = (ctx.get("target") or {}).get("resolved") if ctx.get("target") else None
        if target:
            app = ctx["target"]["requested"]
            ctx["front_last"] = front  # the newest read, for the aggregate even when the step is refused
            facts = {"before_step": later, "activated": ctx["target"].get("activated")}
            if front is None:
                res = {"reason": "front_unread", "error": "no front reader answered before the next step", **facts}
                return {"ok": False, "reason": "front_unread", "error": self._target_error(app, res), "target": ctx["target"]}, step
            # An X11/OSWorld target is its matched windows; a macOS/SSH application is its pid.
            still = x11_same(front, target) if target.get("window_ids") else front.get("pid") == target.get("pid")
            if not still:
                res = {"reason": "target_lost", "front": front, "resolved": target, **facts}
                return {"ok": False, "reason": "target_lost", "error": self._target_error(app, res), "target": ctx["target"], "front": front}, step
        return None, step

    def _finish(self, payload: dict[str, Any], ctx: dict[str, Any], stderr: str = "", *, after: Any = _UNREAD) -> dict[str, Any]:
        """Compose the place facts of an input receipt (any transport) and demote a dispatch the helper itself
        reported as dropped (cliclick warns on stderr and exits 0). ``after`` is the post-input front; the local
        backends read it here, the remote ones pass what their probe printed (None = unread)."""
        if after is _UNREAD:
            after = self._front_local()
        ctx["front_last"] = after
        target = ctx.get("target") or {}
        if target:
            payload["target"] = target
        delivered = ctx.get("delivered_to")
        before = ctx.get("front_before")
        text = stderr.strip()
        if text:
            payload["helper_stderr"] = text[:300]
            if len(text) > 300:
                payload["helper_stderr_truncated"] = True
        known = isinstance(delivered, dict)
        where = (delivered.get("app") or delivered.get("title")) if known else None
        if after is not None and not same_place(after, delivered):
            payload["front_after"] = after
        steps_done = int(payload.get("steps_done") or 0)
        # The dispatch fact is explicit, never inferred: True = the helper ran; False = nothing was sent (a known
        # fact); None = unknown (the call that carried the input did not answer). Unknown is never read as "not sent".
        dispatched = payload.get("dispatched")
        switched = f"; the switch to {target.get('requested')} ran before it" if payload.get("activation_done") else ""
        if not payload.get("ok") and not steps_done:
            if isinstance(before, dict) and (not known or not same_place(before, delivered)):
                payload["front_before"] = before
            if dispatched is False:
                # Nothing was dispatched (a refused combo, a failed clipboard write): a known fact, not an unknown outcome.
                if known:
                    payload["front"] = delivered
                payload["note"] = "the input was not sent" + ("; the clipboard write before it completed" if payload.get("clipboard_done") else "") + switched
                return payload
            if known:
                payload["front_at_dispatch"] = delivered
            if dispatched is None:
                payload["note"] = ("the call that carried the input did not answer, so whether it ran and whether anything "
                                   f"reached {where or 'the front application'} is unknown (do not assume it did not)" + switched)
                return payload
            # The input helper ran and did not report success: whether anything reached the screen is unknown.
            payload["note"] = ("the input helper did not report success, so whether anything reached "
                               f"{where or 'the front application'} is unknown" + switched)
            return payload
        payload["delivered_to"] = delivered if known else "unknown"
        if isinstance(before, dict) and known and not same_place(before, delivered):
            payload["front_before"] = before
        if not payload.get("ok"):
            # Some steps of one call completed before it stopped (``error`` says why); the failing step may have been
            # dispatched (its outcome is unknown) or withheld.
            if dispatched:
                next_step = "the next step was dispatched and did not report success, "
            elif dispatched is False:
                next_step = "the next step was not sent, "
            else:
                next_step = "the call that carried the next step did not answer (whether it ran is unknown), "
            payload["note"] = (f"{steps_done} of {payload.get('steps_total')} step(s) completed while {where or 'an unknown application'} "
                               "was in front; " + next_step + "the rest were not sent; this tool observed no UI change")
            return payload
        dropped = "accessibility" in text.lower() and "not enabled" in text.lower()
        untyped = re.findall(r"Unable to get key code for (\S+)", text)
        if untyped or dropped:
            payload.pop("observed_effect", None)
            payload["ok"] = False
            if untyped:
                # cliclick types through key codes of the ACTIVE keyboard layout and exits 0 when it cannot find
                # one (measured 27.10.2026: Latin text under a Cyrillic layout typed nothing). Partial typing is unknown.
                if payload.get("method") == "paste":
                    hint = "the paste chord's key has no code in the active keyboard layout: switch the layout, or use the app's own commands"
                elif "keys" in payload:
                    hint = "switch the keyboard layout (a Latin one for Latin keys) or use the app's own commands"
                else:
                    hint = "use type_text with via=paste"
                payload.update({"reason": "untypable_characters", "untyped": untyped[:20],
                                "error": f"{len(untyped)} character(s) have no key code in the active keyboard layout and were "
                                         f"not typed (the rest may have been); {hint}"})
            else:
                payload.update({"reason": "permission_denied",
                                "permission": {"accessibility": "untrusted", "source": "helper stderr"},
                                "error": "macOS drops synthetic input from a process without the Accessibility grant: "
                                         "the helper exited 0 but nothing reached the screen"})
            effects = []
            if target.get("activated"):
                payload["observed_effect"] = True  # the switch before it was observed
                payload["activation_done"] = True
                effects.append(f"the switch to {target.get('requested')}")
            if payload.get("clipboard_replaced"):
                payload["observed_effect"] = True  # the clipboard write before it completed
                payload["clipboard_done"] = True
                effects.append("the clipboard write")
            if untyped and payload.get("method") == "paste":
                dropped_text = "the paste chord was not typed (see error)"
            elif untyped:
                dropped_text = "the characters without a key code were dropped (see error); those with one may have been typed"
            else:
                dropped_text = "the input was dropped (see error)"
            payload["note"] = ((" and ".join(effects) + " ran; the input after it: " + dropped_text) if effects
                               else dropped_text + "; this tool observed no UI change")
            return payload
        if payload.get("observed_effect"):
            payload["note"] = ((f"dispatched while {where} was in front" if where else "dispatched; who was in front is unknown")
                               + "; this tool observed no UI change: verify with ax_tree, a screenshot or the app's own commands")
        return payload

    def activate_app(self, *, app: str, wait_ms: int = 2000) -> str:
        """Bring a running application to the front and report who is in front afterwards (never launches)."""
        app = str(app or "").strip()
        if not app:
            return self._json_text({"ok": False, "error": "app is required (a running application's name, bundle id or pid:<n>)"})
        wait_ms = max(200, min(10_000, int(wait_ms or 2000)))
        refusal, ctx = self._begin(app, wait_ms)
        if refusal:
            return self._json_text(refusal)
        target = ctx["target"]
        payload: dict[str, Any] = {"ok": True, "target": target, "front": ctx.get("delivered_to"), "activated": target.get("activated")}
        if ctx.get("front_before") is not None:
            payload["front_before"] = ctx["front_before"]
        if target.get("activated") is True:
            payload["observed_effect"] = True  # the frontmost application changed: an observed external effect
            payload["note"] = f"{app} is now in front; inputs without app= go there until the owner switches away"
        elif target.get("activated") is None:
            payload["note"] = f"{app} is verified in front; whether it was already there is unknown (the previous front could not be read)"
        else:
            payload["note"] = f"{app} was already in front; nothing changed"
        return self._json_text(payload)

    def window_list(self, *, app: str = "", limit: int = 50) -> str:
        """Running applications and their windows with the frontmost named (titles need the OS grant)."""
        plat = self._plat()
        limit = max(1, min(500, int(limit or 50)))
        query = str(app or "").strip().lower()
        if self._is_remote():
            _name, conn = self._active_connection()
            if conn.get("disabled"):
                return self._disabled_connection_error(_name, conn)
            return self._remote_window_list(conn, query, limit)
        if plat == "macos":
            res = self._native("apps", limit, timeout=25)
            if not res.get("ok"):
                return self._json_text({"ok": False, "platform": plat, "reason": res.get("reason") or "helper_failed",
                                        "error": res.get("error") or "no answer", "frontmost": self._front_local()})
            rows = res.get("apps") or []
            if query:
                rows = [r for r in rows if query in (r.get("app") or "").lower() or query == (r.get("app_id") or "").lower()
                        or query == f"pid:{r.get('pid')}"]
            payload: dict[str, Any] = {"ok": True, "platform": plat, "frontmost": res.get("frontmost"), "windows": rows,
                                       "titles_available": bool(res.get("titles_available")), "truncated": bool(res.get("truncated")),
                                       "limit": limit, "note": "rows are applications with their windows; pass app=<name|bundle id|pid:n> "
                                                                "to the input tools to act in one of them"}
            if res.get("frontmost") is None:
                payload["front_unread"] = "lsappinfo named no frontmost application"
            if res.get("warning"):
                payload["warning"] = res["warning"]
            return self._json_text(payload)
        if plat == "linux":
            if self._session() == "wayland" and not self._which_tool("wmctrl"):
                return self._json_text({"ok": False, "platform": plat, "session_type": "wayland", "reason": "unsupported",
                              "error": "this Wayland session exposes no window list to other programs "
                                       f"({os.environ.get('XDG_CURRENT_DESKTOP') or 'unknown desktop'}); ask the owner to bring the app forward"})
            rows, err = self._x11_windows()
            if err:
                return self._json_text({"ok": False, "platform": plat, "error": err})
            front = self._front_local(rows)
            if front:
                for row in rows:
                    row["frontmost"] = same_place(front, row)
            if query:
                rows = [r for r in rows if query in r["app"].lower() or query in r["title"].lower() or query == f"pid:{r['pid']}"]
            payload = {"ok": True, "platform": plat, "session_type": self._session(), "frontmost": front,
                       "windows": rows[:limit], "truncated": len(rows) > limit, "limit": limit}
            if front is None:
                payload["front_unread"] = ("xdotool is not installed" if not self._which_tool("xdotool")
                                           else "xdotool getactivewindow named no window")
            if self._session() == "wayland":
                payload["note"] = "XWayland windows only: native Wayland windows are invisible to wmctrl"
            return self._json_text(payload)
        return self._backend_unavailable("window listing")

    def ax_tree(self, *, app: str = "", max_nodes: int = 150, timeout_ms: int = 3000, root: str = "window",
                include_web: bool = False) -> str:
        """Bounded accessibility read of one window (the frontmost app's, or ``app``'s, without bringing it forward)."""
        plat = self._plat()
        app = str(app or "").strip()
        max_nodes = max(1, min(2000, int(max_nodes or 150)))
        timeout_ms = max(100, min(15_000, int(timeout_ms or 3000)))
        root = "focused" if str(root or "").lower() == "focused" else "window"
        if self._is_remote():
            _name, conn = self._active_connection()
            if conn.get("disabled"):
                return self._disabled_connection_error(_name, conn)
            return self._remote_ax_tree(conn)
        if plat == "macos":
            if app:
                res = self._native("resolve", app, timeout=15)
                if not res.get("ok"):
                    return self._json_text({"ok": False, "platform": plat, "reason": res.get("reason") or "unresolved",
                                  "error": self._target_error(app, res), "candidates": res.get("candidates") or []})
                target = res["candidates"][0]
            else:
                target = self._front_local()
                if not target:
                    return self._json_text({"ok": False, "platform": plat, "reason": "front_unknown", "error": "the frontmost application could not be read"})
            tree = self._native("tree", target["pid"], max_nodes, timeout_ms, root, "true" if include_web else "false",
                                timeout=timeout_ms // 1000 + 8)
            if not tree.get("ok"):
                failed = {"ok": False, "platform": plat, "app": target, "reason": tree.get("reason") or "ax_unavailable",
                          "error": tree.get("error") or f"the accessibility read failed ({tree.get('reason')})"}
                if tree.get("errors"):
                    failed["errors"] = tree["errors"]
                return self._json_text(failed)
            tree.update({"platform": plat, "app": target, "coordinate_note": "node and mark coordinates are INPUT-space "
                         "(macOS logical points): click them with raw=true; a mark is a position, not proof the element reacts"})
            return self._json_text(tree)
        if plat == "linux":
            rows, err = self._x11_windows()
            return self._json_text({"ok": False, "platform": plat, "reason": "unsupported", "session_type": self._session(),
                          "error": "no element reader on Linux in this skill (use the app's own commands or screenshot)",
                          "windows": rows, **({"windows_error": err} if err else {})})
        return self._backend_unavailable("accessibility tree")

    def element_act(self, *, app: str, do: str, select: dict[str, Any] | None = None, text: str | None = None,
                    window: Any = "", max_nodes: int = 600, timeout_ms: int = 3000, include_web: bool = False) -> str:
        """Act on ONE element of ``app`` through the element's own Accessibility action (macOS): ``press`` or
        ``set_value``. The element is named by identity (the window as ``window_list`` prints it, role, exact label,
        AXIdentifier, nth), never by a mark or node id of an earlier read; it is found and acted on inside one
        short-lived helper call (``macos_native.py act``), with no pointer and no switch: who is in front is
        reported, not changed. The helper composes the receipt; this method only resolves the application and
        says when the helper gave no answer (an unknown outcome, never "not sent")."""
        plat = self._plat()
        app = str(app or "").strip()
        do = str(do or "").strip().lower()
        asked: dict[str, Any] = {"app": app, "do": do, "select": select if isinstance(select, dict) else {}}
        if not app:
            return self._json_text({"ok": False, "reason": "app_required", "dispatched": False, **asked,
                                    "error": "app is required (a running application's name, bundle id or pid:<n>): "
                                             "an element action is never aimed at whatever is in front"})
        if do not in ("press", "set_value"):
            return self._json_text({"ok": False, "reason": "unknown_action", "dispatched": False, **asked,
                                    "error": f"do must be press or set_value, not {do!r}"})
        if self._is_remote():
            name, conn = self._active_connection()
            if conn.get("disabled"):
                return self._disabled_connection_error(name, conn)
            return self._json_text({"ok": False, "backend": str(conn.get("backend") or ""), "reason": "unsupported", "dispatched": False, **asked,
                                    "error": "no element actions on a remote (an OSWorld guest is Linux, an SSH Mac has no element reader "
                                             "here); use the input tools with app= or the app's own commands through remote_exec"})
        if plat != "macos":
            return self._json_text({"ok": False, "platform": plat, "reason": "unsupported", "dispatched": False, **asked,
                                    "error": "no element actions on Linux in this skill (use the app's own commands or the input tools)"})
        res = self._native("resolve", app, timeout=15)
        if not res.get("ok"):
            return self._json_text({"ok": False, "platform": plat, "reason": res.get("reason") or "unresolved", "dispatched": False, **asked,
                                    "error": self._target_error(app, res), "candidates": res.get("candidates") or []})
        target = res["candidates"][0]
        selector: dict[str, Any] = {**asked["select"], "include_web": bool(include_web)}
        if window is not None and window != "":
            selector["window"] = window
        if text is not None:
            selector["text"] = text
        max_nodes = max(1, min(2000, int(max_nodes or 600)))
        timeout_ms = max(100, min(15_000, int(timeout_ms or 3000)))
        res = self._native("act", target["pid"], do, json.dumps(selector, ensure_ascii=False), max_nodes, timeout_ms,
                           timeout=timeout_ms // 1000 + 10)
        out: dict[str, Any] = {"platform": plat, "app": target, "do": do, **res}
        if "dispatched" not in res:
            # Every answer of ``act`` says whether the action call was made; one without that fact is a helper that
            # gave no answer or stopped mid-way: whether it found the element and acted is unknown, never "not sent".
            out.update({"ok": False, "reason": "helper_no_answer" if res.get("no_answer") else "helper_failed", "dispatched": None,
                        "action_outcome": "unknown", "select": asked["select"],
                        "note": "the helper that carried the action did not answer, so whether it found the element and acted "
                                "is unknown (do not assume it did not): read the window with ax_tree before acting again"})
        return self._json_text(out)
