"""Keep's answer apart from the task's canonical artifact files.

Since #3981 (owner-confirmed release of stored task files) Keep names every regular file of the task's canonical
artifact store (``task_results/artifacts/<task>/``) beside its other places: each file in ``retained`` or ``retired``
and its release state in ``releases`` (``held``, ``queued``, ``moved``, ``staged``, ``deleting``, ``deleted``,
``missing``, ``unknown``). Tests about the other places (engine threads, saved-state folders, drives, imports)
compare those alone through ``outside_store``; the files themselves are pinned by ``tests/test_payload_release*.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def _inside(store: str, place: str) -> bool:
    return place.startswith(store + os.sep)


def outside_store(answer: Dict[str, Any], root: Any, task_id: str) -> Dict[str, Any]:
    """``answer`` without the entries for files inside ``task_id``'s canonical store. Every file it drops from
    ``retained``/``retired`` must carry its state in ``releases``: Keep never names such a file without one."""
    from ouroboros.payload_release_state import store_of

    store = str(store_of(root, task_id))
    releases = answer.get("releases") or {}
    named = [place for key in ("retained", "retired") for place in answer.get(key) or () if _inside(store, place)]
    assert all(place in releases for place in named), (named, releases)
    return {**answer,
            "retained": [place for place in answer.get("retained") or () if not _inside(store, place)],
            "retired": [place for place in answer.get("retired") or () if not _inside(store, place)],
            "releases": {place: state for place, state in releases.items() if not _inside(store, place)}}
