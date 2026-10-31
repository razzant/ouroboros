"""The permission-granting recursive removal every retirer of a copied tree shares.

``shutil.rmtree(ignore_errors=True)`` silently FAILS on read-only files (Windows git pack and
object files under ``.git``, copies that keep read-only modes), leaving the tree behind forever.
``force_rmtree`` clears the owner bits and retries, only inside the tree and never through a
link, so worktree and genesis-project teardown, review evidence copies and the custody trash of
a settled drive are really removed. Best effort: the caller checks whether the path is gone.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
from typing import Any


def force_rmtree(path: Any) -> None:
    """Best-effort recursive delete that also removes read-only files (module docstring). It only
    adds owner bits, only inside the tree and never through a link (a target may live outside it):
    a removal its folder refuses makes that folder writable, and a folder that could not be
    listed is made listable and removed in turn."""
    root = os.path.abspath(os.fspath(path))

    def _linked(info):  # a link, or a Windows junction (reparse point)
        return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)

    def _grant(target, bits):
        info = os.lstat(target)  # one look: the mode written is this entry's own
        if os.path.commonpath([root, target]) == root and not _linked(info):
            os.chmod(target, stat.S_IMODE(info.st_mode) | bits)

    def _on_error(func, p, _exc):
        p = os.path.abspath(p)  # a relative tree reports relative paths
        try:
            if func in (os.unlink, os.remove, os.rmdir):
                if os.path.dirname(p) != os.path.dirname(root):
                    _grant(os.path.dirname(p), stat.S_IRWXU)
                _grant(p, stat.S_IRWXU if os.path.isdir(p) else stat.S_IWRITE | stat.S_IREAD)
                func(p)
            elif not _linked(os.lstat(p)) and os.path.isdir(p):
                _grant(p, stat.S_IRWXU)
                shutil.rmtree(p, onerror=_on_error)
        except Exception:
            pass

    try:
        shutil.rmtree(pathlib.Path(path), onerror=_on_error)
    except Exception:
        pass


__all__ = ["force_rmtree"]
