"""Decisions over observed facts, as pure functions that are checked on fictitious data.

Three values everywhere: True (established), False (a KNOWN contrary fact) and None (not established). ``tri`` turns them into
"pass", "fail" and "unknown". An image that was not read, a command line that was not read, a brief process that was not seen, a
taskbar button that did not match, a property that errored: all None, never False and never True. Only a fact that was actually
read and is the other thing is False. Nothing here reads a pixel, and nothing returned here carries a host path.
"""

from __future__ import annotations

import re
from typing import Any

import git_traces as gt  # vendored #4220 module: relative() compares Windows paths without regard to slashes or case

AUMID = "com.ouroboros.agent"
MAIN_TITLE = re.compile(r"^Ouroboros v\d+\.\d+\.\d+")
PYTHONS = ("python.exe", "pythonw.exe")


def tri(ok: bool | None) -> str:
    return "unknown" if ok is None else "pass" if ok else "fail"


def and3(*values: bool | None) -> bool | None:
    """False dominates (one known contrary fact decides), then None (something is not established), else True."""
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def image(path: Any, roots: dict[str, Any]) -> dict[str, Any]:
    """Where a process image lies: a copy's frozen Ouroboros.exe, that copy's bundled Python, or elsewhere. The classification
    is by path inside a copy, never by the file name alone."""
    for copy, root in roots.items():
        rel = gt.relative(str(path or ""), str(root))
        if rel is None:
            continue
        low = rel.lower()
        kind = ("frozen_exe" if low == "ouroboros.exe" else
                "bundled_python" if low.startswith("_internal/python-standalone/") and low.rsplit("/", 1)[-1] in PYTHONS else
                "copy_other")
        return {"copy": copy, "rel": low, "kind": kind}
    return {"copy": None, "rel": None, "kind": "outside_copies"}


def command(line: str, roots: dict[str, Any], repo_dir: Any) -> dict[str, Any]:
    """A command line as fixed words. Like the app's own ``other_copy_line``, the raw line is searched: the profile
    checkout's launcher.py, and the copy named after the last ``--seed-bundle`` (quoted, or up to the next ``--`` flag)."""
    norm = lambda p: str(p).replace("/", "\\").rstrip("\\").lower()  # noqa: E731
    text = norm(line)
    rest = text.rpartition("--seed-bundle ")[2].strip() if "--seed-bundle " in text else ""
    tail = rest[1:].split('"')[0] if rest.startswith('"') else rest.split(" --")[0].strip()  # a later flag is not part of the path
    seed = next((c for c, r in roots.items() if tail and (tail == norm(r) or tail.startswith(norm(r) + "\\"))), None)
    return {"readable": bool(text), "launcher_py_in_profile_repo": norm(repo_dir) + "\\launcher.py" in text, "seed_bundle_copy": seed}


def read(ident: dict[str, Any] | None) -> bool:
    """The process image was actually read: anything decided about it is a known fact."""
    return bool(ident and ident.get("pid_read"))


def frozen_of(ident: dict[str, Any] | None, copy: str) -> bool | None:
    if not read(ident):
        return None
    return ident["kind"] == "frozen_exe" and ident["copy"] == copy


def first_launch(ident: dict[str, Any] | None, copy: str) -> bool | None:
    """The first holder of an installed copy at the reviewed commit: that copy's own frozen exe, with the ZIP's bytes."""
    return and3(frozen_of(ident, copy), (ident or {}).get("bytes_equal_zip") if read(ident) else None)


def handed_off(ident: dict[str, Any] | None, copy: str) -> bool | None:
    """The process is ``copy``'s bundled Python (the ZIP's bytes) running the profile checkout's launcher.py for that copy. A known
    image that is anything else, the copy's frozen exe included, is False; an unread image or command line is None."""
    if not read(ident):
        return None
    if ident["kind"] != "bundled_python" or ident["copy"] != copy:
        return False
    cmd = ident.get("command") or {}
    if not cmd.get("readable"):
        return and3(None, ident.get("bytes_equal_zip"))
    return and3(bool(cmd["launcher_py_in_profile_repo"]), cmd["seed_bundle_copy"] == copy, ident.get("bytes_equal_zip"))


def installed_window(owner: dict[str, Any] | None) -> bool | None:
    """The process that owns a window of the installed program: its own frozen exe with the package's bytes, or (after a source
    handoff) its bundled Python running the profile's launcher.py for that install. Another copy is False, unread is None."""
    if read(owner) and owner["kind"] == "frozen_exe":
        return first_launch(owner, "installed")
    return handed_off(owner, "installed")


def after_update(ident: dict[str, Any] | None, copy: str, *, pid: int, first_pid: int, file_is_b: bool | None) -> bool | None:
    """After the first Update, once the B shell is ready: a handoff, to another process than the one we started, whose launcher.py
    in the profile checkout has fixture B's bytes. A launcher.py that is not B's is a known failure whatever else was read."""
    return and3(file_is_b, handed_off(ident, copy), pid != first_pid)


def cold_chain(first: dict[str, Any] | None, final: dict[str, Any] | None, copy: str) -> bool | None:
    """A cold start: the copy's frozen exe, then (after the handoff) its bundled Python, both with the package's bytes. A contrary fact
    that was actually read is a failure whatever else was read: an end that is not that Python, another copy's frozen exe first, or a
    first program of this copy with other bytes. True needs the frozen exe SEEN with the package's bytes, as another process before the
    handed-off Python; a launcher that was not caught (it can be gone before the first look) or bytes not compared are unknown."""
    end = handed_off(final, copy)
    other = read(first) and first["kind"] == "frozen_exe" and first["copy"] not in (None, copy)  # another copy's frozen exe started it
    altered = read(first) and first["copy"] == copy and first.get("bytes_equal_zip") is False  # this copy's program, not the shipped bytes
    if end is False or other or altered:
        return False
    seen = first_launch(first, copy) is True and first.get("pid") != (final or {}).get("pid")
    return True if end is True and seen else None


def window(obs: dict[str, Any]) -> dict[str, str]:
    """The window's OWN label (its title) and icons: what WM_GETICON returns and, only where it answered "none", the icon of its
    window class (what Windows then shows: WM_GETICON's documented fallback; Windows derives a missing small class icon from the
    class's big one). Never what its image or the taskbar might supply. A window that vanished during the read, a window that did
    not answer, or a class icon that was not read leaves that item "unknown": only a title, or "no icon" read from both the window
    and its class, can fail."""
    win, icons = obs.get("window") or {}, obs.get("icons") or {}
    if not win.get("found"):
        return {"label": "unknown", "icons": "unknown"}
    flag = lambda key, field: (icons.get(key) or {}).get(field)  # noqa: E731 - True, False or None (not read)

    def shows(own: tuple[str, ...], classes: tuple[str, ...]) -> bool | None:
        answers = [flag(key, "window_supplies_icon") for key in own]
        if True in answers:
            return True
        if None in answers:
            return None
        fallback = [flag(key, "class_has_icon") for key in classes]
        return True if True in fallback else None if None in fallback else False

    small, big = shows(("small", "small2"), ("class_small", "class_big")), shows(("big",), ("class_big",))
    title = win.get("title")
    return {"label": tri(bool(MAIN_TITLE.match(str(title)))) if title else "unknown", "icons": tri(and3(small, big))}


def appid_state(value: Any) -> str:
    """An AppUserModel.ID as one fixed word: the contract's, another one, or none readable (which may be a path: never returned)."""
    if not isinstance(value, str) or not value or value.startswith("<"):
        return "unset"
    return "contract" if value == AUMID else "other"


def taskbar(obs: dict[str, Any]) -> dict[str, Any]:
    """Our button, or nothing. A match is exactly one on-screen button whose Appid is the contract's or whose name holds the product
    name. Other buttons being present proves nothing, and no match is "unknown", never a pass. No raw id or name is returned."""
    bar, ours = obs.get("taskbar") or {}, []
    for b in bar.get("buttons") or []:
        if b.get("offscreen") or not (b.get("rect") or [0, 0, 0, 0])[2]:
            continue
        if b.get("appid") == AUMID or "ouroboros" in str(b.get("name") or "").lower():
            ours.append(b)
    if len(ours) != 1:
        return {"state": "unknown", "reason": "no_match" if not ours else "ambiguous", "buttons_seen": len(bar.get("buttons") or [])}
    return {"state": "matched", "appid_state": appid_state(ours[0].get("appid")), "name_has_product": "ouroboros" in str(ours[0].get("name") or "").lower()}


def aumid(obs: dict[str, Any], bar: dict[str, Any]) -> str:
    """The AppUserModel.ID the window or its matched taskbar button carries. The window's own property (documented) decides either way,
    whatever the taskbar shows. Only where it holds none (unset or unread) can the matched button's AutomationId ("Appid:...",
    undocumented) confirm the contract's ID; alone it proves no other one. Anything else is unknown; the matched button's name and
    grouping are recorded beside it."""
    own = appid_state((obs.get("props") or {}).get("aumid"))
    if own != "unset":
        return tri(own == "contract")
    return "pass" if bar.get("state") == "matched" and bar.get("appid_state") == "contract" else "unknown"


def shortcut(lnk: dict[str, Any], exe: Any, install_dir: Any) -> dict[str, str]:
    """Target, Icon and AppUserModel.ID of the real Start-menu .lnk, judged as SOURCE (what the file says), never as pixels.
    The installer's [Icons] sets Target and the ID but no IconFilename, so an empty Icon with the installed exe as Target is the
    normal inheritance of the exe's icon (pass, source "inherits_target_exe"); an explicit Icon must lie in the installed copy.
    An unset ID (null) is a fail; a read error is unknown."""
    norm = lambda p: str(p).replace("/", "\\").rstrip("\\").lower()  # noqa: E731
    target_ok = bool(lnk.get("target")) and norm(lnk.get("target")) == norm(exe)
    icon = str(lnk.get("icon_location") or "").rsplit(",", 1)[0].strip()
    value = lnk.get("aumid")
    if not icon:
        icon_state, source = tri(target_ok), "inherits_target_exe"
    else:
        inside = gt.relative(icon, str(install_dir)) is not None
        icon_state, source = tri(inside), "explicit_in_copy" if inside else "explicit_elsewhere"
    return {"target": tri(target_ok), "icon": icon_state, "icon_source": source,
            "aumid": "unknown" if isinstance(value, str) and value.startswith("<") else tri(value == AUMID)}


def same_profile(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Byte equality of two profile manifests (path -> [size, sha256]). Anything unreadable on either side, and a "before"
    that holds no file at all (nothing to lose proves nothing), is unknown; a profile that is gone afterwards is a fail."""
    if before["unreadable"] or after["unreadable"] or not before["files"]:
        return "unknown"
    return tri(before["files"] == after["files"])


def saved_rows(read_ok: bool, complete: bool | None, found: bool) -> bool | None:
    """The expected exchange in the app's durable history. Found rows settle it even in a partial window; absent rows are a
    known loss only when the app itself said the window it returned is complete."""
    if read_ok and found:
        return True
    return False if read_ok and complete is True else None


def equals(value: Any, want: Any) -> bool | None:
    """A value that was not read is not established; a read value that differs is a known contrary fact."""
    return None if not value or not want else value == want


def bundled_git(claim_state: str, control_ok: bool, requirements: dict[str, bool | None], windows: tuple[str, ...]) -> str:
    """The same meaning as #4220's ``git_traces.judge``: a foreign or altered Git is a failure; an unevaluated or
    unconfirmed identity, a failed control and a missing or unmet phase window are all "unknown", never a failure."""
    if claim_state in ("foreign", "not_the_package"):
        return "fail"
    proven = claim_state == "verified" and control_ok and bool(requirements) and set(windows) <= set(requirements) and all(
        v is True for v in requirements.values())
    return "pass" if proven else "unknown"


def revision(value: Any, known: dict[str, str]) -> str:
    """A commit as one fixed word of the ones this run knows ("old_release", "candidate", "update_target_b"), else "other"."""
    if not value:
        return "unread"
    return next((word for word, sha in known.items() if sha and sha == value), "other")


def remote_url(config_text: str, name: str = "managed") -> str:
    """The ``url`` of one remote in a checkout's ``.git/config`` (read as text: no Git is run); "" when not there."""
    section = re.search(r'(?ms)^\[remote "%s"\]\s*$(.*?)(?=^\[|\Z)' % re.escape(name), config_text or "")
    found = re.search(r"(?m)^\s*url\s*=\s*(.+?)\s*$", section.group(1)) if section else None
    return found.group(1).replace("\\\\", "/").replace("\\", "/") if found else ""
