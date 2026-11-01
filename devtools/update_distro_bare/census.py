"""Every byte the interpreter's global packages and pip policy consist of, before and after a phase. Stdlib only.

    python3 -I -B census.py snapshot OUT.json [--root DIR ...]
    python3 -I -B census.py diff BEFORE.json AFTER.json OUT.json   exit 1: any change but bytecode, or anything unread

A root's ``__pycache__`` is reported apart: a root process can write it without installing anything, which is a fact to
state and not a package mutation. A file this uid could not read, or a folder ``os.walk`` could not list, is kept as its
error and listed as ``unread``: its bytes are unknown, so the census is not clean, whatever the two rows say.
``--root`` replaces the computed roots (the equipment's own tests).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import site
import stat
import subprocess
import sys
import sysconfig


def roots() -> list:
    home = os.path.expanduser("~")
    found = {sysconfig.get_path(name) for name in ("stdlib", "platstdlib", "purelib", "platlib", "scripts")}
    found |= set(site.getsitepackages()) | {site.getusersitepackages(), site.getuserbase()}
    for pattern in ("/usr/lib/python3", "/usr/lib/python3.*", "/usr/local/lib/python3*", "/usr/local/bin",
                    "/usr/share/python-wheels", "/usr/bin/pip*", "/usr/bin/python3*", "/etc/pip.conf", "/etc/python3*",
                    home + "/.local", home + "/.config/pip", home + "/.pip"):
        found |= set(glob.glob(pattern)) or ({pattern} if "*" not in pattern else set())
    return sorted(found)


def _entry(path: str) -> dict:
    facts = os.lstat(path)
    row = {"mode": stat.S_IMODE(facts.st_mode), "uid": facts.st_uid}
    if stat.S_ISLNK(facts.st_mode):
        return {**row, "link": os.readlink(path)}
    if stat.S_ISDIR(facts.st_mode):
        return {**row, "dir": True}
    row["size"] = facts.st_size
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:  # a file this uid may not read still has its size, mode and owner compared
        return {**row, "unreadable": type(exc).__name__}
    return {**row, "sha256": digest.hexdigest()}


def tree(root_paths: list) -> dict:
    rows = {}

    def unlisted(exc: OSError) -> None:  # os.walk drops these by default: a folder whose content went unseen
        rows[exc.filename] = {**rows.get(exc.filename, {}), "unreadable": type(exc).__name__}

    for root in root_paths:
        if not os.path.lexists(root):
            rows[root] = {"absent": True}
            continue
        rows[root] = _entry(root)
        if os.path.isdir(root) and not os.path.islink(root):
            for folder, names, files in os.walk(root, onerror=unlisted):
                for name in sorted(names + files):
                    path = os.path.join(folder, name)
                    rows[path] = _entry(path)
    return rows


def _dpkg() -> dict:
    def run(*argv):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        except OSError as exc:
            return {"unavailable": str(exc)}
        return {"rc": done.returncode, "out": done.stdout}

    return {"packages": run("dpkg-query", "-W"), "verify": run("dpkg", "--verify")}


def snapshot(root_paths: "list | None") -> dict:
    chosen = root_paths or roots()
    return {"roots": chosen, "files": tree(chosen), "dpkg": _dpkg() if not root_paths else {},
            "uid": os.geteuid(), "hostname": os.uname().nodename}


def _bytecode(path: str) -> bool:
    return "/__pycache__" in path or path.endswith((".pyc", ".pyo"))


def diff(before: dict, after: dict) -> dict:
    was, now = before["files"], after["files"]
    added = sorted(set(now) - set(was))
    removed = sorted(set(was) - set(now))
    changed = sorted(path for path in set(was) & set(now) if was[path] != now[path])
    split = lambda paths: ([p for p in paths if not _bytecode(p)], [p for p in paths if _bytecode(p)])  # noqa: E731
    result = {"files_before": len(was), "files_after": len(now), "same_roots": before["roots"] == after["roots"],
              "same_host": before["hostname"] == after["hostname"], "same_dpkg": before["dpkg"] == after["dpkg"]}
    for name, paths in (("added", added), ("removed", removed), ("changed", changed)):
        result[name], result[f"bytecode_{name}"] = split(paths)
    result["unread"] = sorted({path for rows in (was, now) for path, row in rows.items() if "unreadable" in row})
    result["clean"] = bool(result["same_roots"] and result["same_host"] and result["same_dpkg"] and not result["unread"]
                           and not (result["added"] or result["removed"] or result["changed"]) and len(was) > 0)
    return result


def main(argv: list) -> int:
    if argv[:1] == ["snapshot"]:
        chosen = [argv[i + 1] for i, word in enumerate(argv) if word == "--root"]
        with open(argv[1], "w", encoding="utf-8") as fh:
            json.dump(snapshot(chosen), fh)
        return 0
    if argv[:1] == ["diff"]:
        with open(argv[1], encoding="utf-8") as one, open(argv[2], encoding="utf-8") as two:
            result = diff(json.load(one), json.load(two))
        with open(argv[3], "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        return 0 if result["clean"] else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
