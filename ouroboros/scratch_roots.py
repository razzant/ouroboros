"""The session scratch folders Ouroboros currently owns, from their records: the Storage place ``session_scratch``.

A writable coding session's short scratch folder lives outside the data folder (``workspace_preflight``:
``obs-<random>/`` under the session scratch parent) and was counted nowhere, so a pile the release pass
failed to let go was invisible to Storage and to the mind. Its ownership is recorded twice: the record
``session_scratch_owners/<key>/scratch.json`` in the owning drive (path + device/inode) and, for a running
or finished task, the note ``state/session_scratch/<task>/<key>.json`` in the canonical root naming that
record's folder. This module lists the folders the notes name, validated exactly as the release pass
validates them (``workspace_preflight._recorded_scratch``: the recorded parent, the ``obs-`` name, the
recorded inode): no temp-root scan, no link followed, a note or record that cannot be read names no root.
The count walks them like any other place (``storage.PLACES``); the release pass and Free remove them.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, List


def recorded_scratch_roots(data_root: Any) -> List[str]:
    """Every session scratch folder a readable note of ``state/session_scratch`` names, each once, in note order."""
    from ouroboros.workspace_preflight import OPEN_SCRATCH_DIR, _recorded_scratch

    notes = pathlib.Path(data_root) / OPEN_SCRATCH_DIR
    roots: List[str] = []
    if notes.is_symlink() or not notes.is_dir():
        return roots
    for folder in sorted(notes.iterdir()):
        if folder.is_symlink() or not folder.is_dir():
            continue
        for note in sorted(folder.glob("*.json")):
            if note.is_symlink() or not note.is_file():
                continue
            try:
                owner = pathlib.Path(str(json.loads(note.read_text(encoding="utf-8"))["owner"]))
                record = owner / "scratch.json"
                if not os.path.lexists(record):
                    continue  # an uncompleted preparation: no scratch yet
                path = os.path.realpath(os.fspath(_recorded_scratch(record)))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # unreadable, replaced or gone: names no root (the release pass reports it)
            if path not in roots:
                roots.append(path)
    return roots
