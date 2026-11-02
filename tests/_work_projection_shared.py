"""The server's own work projection of controlled census rows.

Tests that answer ``/api/state`` themselves (browser transports, the web fixture) send the
``works``/``rooms`` fields ``gateway/work_projection.works_snapshot`` builds from the same
census rows, so the page never reads a shape the server does not send. The projection's
stored reads (task results, owner fences, retained delegated runs, answer upkeep) are
answered from the given maps instead of a data directory.
"""
from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from unittest import mock

from ouroboros.gateway import work_projection as wp


def _custody(root_id, run):
    """One open custody row as ``_open_runs`` yields it: a bare run id is an ordinary run of the root itself."""
    run = {"run_id": run} if isinstance(run, str) else dict(run)
    return SimpleNamespace(run_id=run["run_id"], task_id=run.get("task_id", root_id), root_task_id=root_id,
                           source=run.get("source", "delegated_subagent"), terminal_state=run.get("terminal_state", ""))


def work_fields(owners, raw=None, *, results=None, fences=None, held=None, stops=None, models=None, complete=True):
    """``works``, ``rooms``, ``work_summary``, ``works_complete`` and ``works_gaps`` for these rows, as JSON.

    ``held`` maps a tree root to its open runs: run ids, or ``{run_id, task_id, source}`` rows; ``stops`` maps a
    task to the runs whose stop its live Pause could not confirm."""
    results, fences, stops = dict(results or {}), dict(fences or {}), dict(stops or {})
    held = {root: [_custody(root, run) for run in runs] for root, runs in dict(held or {}).items()}
    with tempfile.TemporaryDirectory() as root, \
            mock.patch.object(wp._Observations, "result", lambda self, task_id: results.get(task_id)), \
            mock.patch.object(wp._Observations, "fence", lambda self, root_id, drive_root=None: fences.get(root_id, {})), \
            mock.patch.object(wp._Observations, "model_handoffs",
                              lambda self, root_id, drive_root=None: (models or {}).get(root_id, [])), \
            mock.patch.object(wp._Observations, "stop_unknown",
                              lambda self, task_id, drive_root=None, owner=None: frozenset(stops.get(task_id, ()))), \
            mock.patch.object(wp, "_open_runs", lambda drive_root: held), \
            mock.patch.object(wp, "_inflight_upkeep", lambda drive_root, known: []):
        snapshot = wp.works_snapshot(root, [dict(row) for row in owners], dict(raw or {}), complete=complete)
    return json.loads(json.dumps(snapshot))


def with_work_fields(body, raw=None, **reads):
    """A ``/api/state`` body a test answers, with the work fields the server sends for the census rows it carries.

    The page reads ``works``/``rooms``, never the census rows themselves, so a body that states census rows also
    states their projection; ``raw`` and ``reads`` are the projection's stored reads (see ``work_fields``). A body
    without its census (an unavailable census) is returned as it is.
    """
    if not isinstance(body, dict) or not isinstance(body.get("active_chat_activities"), list):
        return body
    complete = body.get("active_chat_activities_complete", True) is not False
    return {**body, **work_fields(body["active_chat_activities"], raw, complete=complete, **reads)}


def state_fields(drive_root, task_bindings=None, **census):
    """The census rows and the server's own work projection of them for a real data root, as ``gateway/state.py``
    builds both for ``/api/state`` (``active_chat_activities`` and the work fields)."""
    from ouroboros.gateway import state as gateway_state

    availability = census.pop("availability", None) or {"complete": True}
    owners, facts = [], {}
    rows = gateway_state._chat_activities_snapshot_safe(drive_root, task_bindings, availability=availability,
                                                        decision_owners=owners, work_facts=facts, **census)
    projection = wp.works_snapshot(drive_root, owners, facts, complete=availability["complete"])
    return json.loads(json.dumps({"active_chat_activities": rows, **projection}))
