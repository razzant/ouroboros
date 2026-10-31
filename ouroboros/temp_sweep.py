"""The stale temp sweep: orphaned atomic-write temps and the script fallback a hard kill left.

``utils.atomic_write_json`` writes to a unique ``.{name}.tmp.<pid>.<tid>.<uuid>`` sibling and
``os.replace``s it into place; a SIGKILL between create and rename orphans the temp. The
maintenance child's ``temp_sweep`` job reaps those in the atomic writers' directories once per
installed release; the server sweeps the ``tmp_scripts`` fallback at startup.
"""
from __future__ import annotations

import os
import pathlib
import threading
import time
import uuid
from typing import Iterator

# Where the atomic writers place their temp siblings (the data root's own files aside): the
# temp sweep walks only these, pruning trees with an owner of their own (a child drive goes
# whole with its drive's GC; checkouts and runtimes are foreign). Never ``observability/``.
ATOMIC_TEMP_DIRS = ("state", "logs", "memory", "projects", "task_results")
_ATOMIC_TEMP_PRUNED = (("state", "headless_tasks"), ("state", "review_checkouts"), ("state", "cx"))


# A file name is at most 255 bytes on the filesystems Ouroboros runs on (a physical limit).
_NAME_MAX_BYTES = 255
_SIGNATURE = set("0123456789abcdef.")  # pid, tid and uuid after the final ``.tmp.``


def is_atomic_temp(name: str) -> bool:
    """The signature ``atomic_temp_path`` writes (hex digits and dots after the final ``.tmp.``), so a
    user dotfile such as ``.config.tmp.backup`` never matches."""
    suffix = name.rsplit(".tmp.", 1)
    return name.startswith(".") and len(suffix) == 2 and bool(suffix[1]) and not set(suffix[1]) - _SIGNATURE


def atomic_temp_path(path: pathlib.Path, *, tag: str = "") -> pathlib.Path:
    """The sibling temp name every atomic writer uses, ``.{name}.tmp.<pid>.<tid>.<uuid>``: the one
    signature this sweep reaps, so a hard kill's leftover is reclaimed instead of surviving beside
    the file it was meant to replace (and, in an artifact folder, being listed as a deliverable).
    A writer's own ``tag`` (``.{name}.{tag}.tmp…``) lets it know its own siblings; it is never cut.
    A name too long to fit beside that suffix keeps a prefix and a digest of the whole name."""
    path = pathlib.Path(path)
    suffix = f"{'.' + tag if tag else ''}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}"
    stem = path.name
    room = _NAME_MAX_BYTES - 1 - len(suffix.encode("utf-8"))
    if len(stem.encode("utf-8")) > room:
        import hashlib

        digest = "~" + hashlib.sha256(stem.encode("utf-8")).hexdigest()[:12]
        keep = stem.encode("utf-8")[:max(0, room - len(digest))].decode("utf-8", "ignore")
        stem = keep + digest
    return path.with_name(f".{stem}{suffix}")


def _atomic_temp_candidates(root: pathlib.Path) -> Iterator[pathlib.Path]:
    yield from root.glob(".*.tmp.*")
    pruned = {root.joinpath(*parts) for parts in _ATOMIC_TEMP_PRUNED}
    for rel in ATOMIC_TEMP_DIRS:
        for current, dirs, files in os.walk(root / rel):  # never follows a symlinked directory
            here = pathlib.Path(current)
            dirs[:] = [name for name in dirs if here / name not in pruned]
            yield from (here / name for name in files if name.startswith(".") and ".tmp." in name)


def sweep_stale_temp_files(root: pathlib.Path, *, min_age_sec: float = 3600.0,
                           atomic_temps: bool = True, scripts: bool = True) -> int:
    """Remove orphaned atomic-write temp files left behind by a hard kill.

    ``atomic_write_json`` writes to a unique ``.{name}.tmp.<pid>.<tid>.<uuid>``
    sibling then ``os.replace``s it into place; a SIGKILL between create and
    rename can orphan the temp file (its try/finally cleanup never runs). This
    sweeps the atomic writers' directories (``ATOMIC_TEMP_DIRS``, never the whole
    root) for such temp files older than ``min_age_sec`` — the age guard avoids
    deleting a temp file from an in-flight write in another process.
    Returns the number removed. Best-effort: never raises.

    Only files whose suffix after the final ``.tmp.`` is the atomic signature
    (pid/tid/uuid → hex digits and dots) are reaped, so a legitimate user dotfile
    such as ``.config.tmp.backup`` is never deleted.

    The data-root ``tmp_scripts/`` fallback is in scope too (CPL4-C20):
    ``tools/shell.py`` unlinks its ``script_<uuid>.<ext>`` files in a
    ``finally``, so one that survived is a hard-kill orphan. Only the
    TOP-LEVEL fallback dir is swept here — task-drive copies die with their
    drive's own GC prune — and only at startup, when no script can be live
    (``scripts``); the walk for atomic temps (``atomic_temps``) is the maintenance
    child's ``temp_sweep`` job, once per installed release (ouroboros/maintenance.py).
    """
    root = pathlib.Path(root)
    if not root.is_dir():
        return 0
    removed = 0
    now = time.time()
    try:
        candidates = list(_atomic_temp_candidates(root)) if atomic_temps else []
        if scripts:
            candidates.extend(root.glob("tmp_scripts/script_*"))
    except OSError:
        return 0
    fallback_scripts = root / "tmp_scripts"
    for tmp in candidates:
        try:
            if not tmp.is_file():
                continue
            if not (tmp.parent == fallback_scripts and tmp.name.startswith("script_")) and not is_atomic_temp(tmp.name):
                continue  # never an unrelated dotfile that happens to match
            if now - tmp.stat().st_mtime < min_age_sec:
                continue
            tmp.unlink()
            removed += 1
        except OSError:
            continue
    return removed
