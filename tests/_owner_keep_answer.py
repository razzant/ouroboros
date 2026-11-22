"""Keep's answer apart from the task's canonical artifact files.

Since #3981 (owner-confirmed release of stored task files) Keep names every regular file of the task's canonical
artifact store (``task_results/artifacts/<task>/``) beside its other places: each file in ``retained`` or ``retired``
and its release state in ``releases`` (``held``, ``queued``, ``moved``, ``staged``, ``deleting``, ``deleted``,
``missing``, ``unknown``). Tests about the other places (engine threads, saved-state folders, drives, imports)
compare those alone through ``outside_store``; the files themselves are pinned by ``tests/test_payload_release*.py``.
A saved state for exact return is never among those files: Keep names it once, by its ``saved_states:`` folder.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def _inside(store: str, place: str) -> bool:
    return place.startswith(store + os.sep)


def outside_store(answer: Dict[str, Any], root: Any, task_id: str) -> Dict[str, Any]:
    """``answer`` without the entries for files inside ``task_id``'s canonical store. Every file it drops from
    ``retained``/``retired`` must carry a state that agrees with its list, and none may be a saved state for exact return."""
    from ouroboros.payload_release_state import store_of
    from ouroboros.saved_state_retirement import worked_copy

    store = str(store_of(root, task_id))
    releases = answer.get("releases") or {}
    named = [place for key in ("retained", "retired") for place in answer.get(key) or () if _inside(store, place)]
    assert all(place in releases for place in named), (named, releases)
    handles = os.path.join(store, "source_handles")
    per_file = [place for place in releases if os.path.dirname(os.path.dirname(place)) == handles
                and worked_copy(os.path.basename(os.path.dirname(place)), os.path.basename(place))]
    assert not per_file, f"saved states are named by their folder, never per file: {per_file}"
    for key, allowed in (("retained", {"held", "queued"}), ("retired", {"moved", "staged", "deleting", "deleted", "missing"})):
        wrong = {place: releases.get(place) for place in answer.get(key) or () if _inside(store, place)
                 and releases.get(place) not in allowed}
        assert not wrong, f"a stored file's list and its release state disagree ({key}): {wrong}"
    stray = [place for place, state in releases.items() if _inside(store, place) and state == "unknown"
             and place in [*(answer.get("retained") or ()), *(answer.get("retired") or ())]]
    assert not stray, f"an unknown stored file is claimed kept or gone: {stray}"
    return {**answer,
            "retained": [place for place in answer.get("retained") or () if not _inside(store, place)],
            "retired": [place for place in answer.get("retired") or () if not _inside(store, place)],
            "releases": {place: state for place, state in releases.items() if not _inside(store, place)}}
