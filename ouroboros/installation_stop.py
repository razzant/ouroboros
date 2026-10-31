"""Panic's installation input and its single durable receipt, not task authority.

Launchers keep writing ``panic``. Boot translates that input into work stops
before acknowledging it in the existing controls store. A failed unlink cannot
apply the same input twice; rewriting the file is a new input, regardless of age.
"""
from __future__ import annotations

import os
import errno
import json
import pathlib
import stat
from typing import Any
from pathlib import Path

PANIC_RECEIPT = "panic_ingress_receipt"


def confirm_absent(path: Any) -> None:
    """After ``FileNotFoundError`` on ``path``, return only when it is really absent.

    POSIX types a non-directory where a directory belongs as ENOTDIR; Windows reports
    ERROR_PATH_NOT_FOUND, indistinguishable from a missing parent. A present entry
    after the failed operation is unknown. Only missing entries below a directory
    prove absence; a dangling link or an unexaminable ancestor cannot prove it."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(errno.EEXIST, "path present after failed read", str(path))
    for ancestor in pathlib.Path(path).parents:
        try:
            mode = os.stat(ancestor).st_mode
        except FileNotFoundError:
            try:
                os.lstat(ancestor)  # stat alone also reports a dangling link as ENOENT
            except FileNotFoundError:
                continue
            raise
        if not stat.S_ISDIR(mode):
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(ancestor))
        return
    raise FileNotFoundError(errno.ENOENT, "no directory ancestor confirmed", str(path))


def panic_input(root) -> dict:
    path = Path(root) / "state" / "panic_stop.flag"
    try:
        os.lstat(path)
        with path.open("r", encoding="utf-8") as stream:
            content = stream.read().strip()
            stat = os.fstat(stream.fileno())
    except FileNotFoundError:
        confirm_absent(path)
        return {}
    if content != "panic":
        return {}
    return {"dev": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "size": stat.st_size}


def panic_received(root, identity: dict) -> bool:
    try:
        with (Path(root) / "state" / "state.json").open(encoding="utf-8") as stream:
            state = json.load(stream)
    except (OSError, ValueError):
        return False
    return bool(identity and isinstance(state, dict) and state.get(PANIC_RECEIPT) == identity)


def installation_panic(root, *, strict: bool = False) -> bool:
    """Only installation gates read this; unreadable input keeps their old veto."""
    try:
        identity = panic_input(root)
        return bool(identity and not panic_received(root, identity))
    except (OSError, UnicodeError):
        if strict:
            raise
        return True


def remove_received_panic(root, identity: dict) -> None:
    """Do not erase a newer Panic that arrived while boot was saving controls."""
    if panic_input(root) == identity:
        (Path(root) / "state" / "panic_stop.flag").unlink(missing_ok=True)


if __name__ == "__main__":  # Android's installation gate, before any product import
    import sys
    try:
        pending = installation_panic(sys.argv[1])
    except Exception:
        sys.exit(2)  # the shell falls back to the file; 1 means an explicit absence
    sys.exit(0 if pending else 1)
