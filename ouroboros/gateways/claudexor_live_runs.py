"""The runs an engine still holds live: a passive, negotiated read of its own run list.

The engine owns the list (``GET /v2/runs``: newest-first keyset pages of durable run
summaries, filtered by lifecycle ``state``). A live run is ``queued`` or ``running``; those
are exactly the runs a stop of the engine ends, whoever started them. This leaf negotiates
the ``state`` filter from the serving operation catalog, reads every page of both live
states and keeps the engine's rows. An engine that does not declare the filter is not asked
(``run_list_unsupported``): an unfiltered page of its newest runs is not a count of its
live ones. A refused, unfinished or malformed answer raises; it is never a shorter list.
No attribution or verdict lives here (docs/DEVELOPMENT.md "Gateway Boundary Pattern").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

RUN_LIST_OPERATION = ("GET", "/v2/runs")
# Queued first: a run only moves queued -> running, so reading in that order can list one
# twice (the later row wins) and can never miss it.
LIVE_RUN_STATES = ("queued", "running")
_PAGE_ROWS = 1000  # the engine's own page ceiling
_MAX_PAGES = 20  # per state: a cursor that never ends is a malformed answer, not a wait


def live_run_list_supported(operations: List[dict]) -> bool:
    """The serving catalog declares the run list's ``state`` filter for both live states."""
    from ouroboros.gateways.claudexor import operation_query_supported

    method, path = RUN_LIST_OPERATION
    return all(operation_query_supported(operations or [], method=method, path=path, name="state", value=state)
               for state in LIVE_RUN_STATES)


def read_live_runs(gateway: Any, *, timeout_sec: Optional[float] = None) -> List[Dict[str, Any]]:
    """Every queued and running run summary, one row per run id, through an authenticated gateway.

    ``timeout_sec`` bounds each request, not their sum. Raises ``ClaudexorUnavailable``:
    ``run_list_unsupported`` when the catalog does not declare the filter, the engine's own
    refusal, or ``malformed_response`` for a body that is not the declared page.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    bound = {} if timeout_sec is None else {"timeout_sec": timeout_sec}
    if not live_run_list_supported(gateway.operations(**bound)):
        raise ClaudexorUnavailable("run_list_unsupported", "this Claudexor engine does not list its live runs")
    runs: Dict[str, Dict[str, Any]] = {}
    for state in LIVE_RUN_STATES:
        cursor = ""
        for _page in range(_MAX_PAGES):
            query = {"state": state, "limit": _PAGE_ROWS, **({"cursor": cursor} if cursor else {})}
            body = gateway._request("GET", f"{RUN_LIST_OPERATION[1]}?{urlencode(query)}", **bound)
            rows = body.get("runs") if isinstance(body, dict) else None
            if not isinstance(rows, list) or any(
                    not isinstance(row, dict) or not isinstance(row.get("runId"), str) or not row["runId"]
                    or row.get("state") != state for row in rows):
                raise ClaudexorUnavailable("malformed_response", "Claudexor run list returned no page of runs")
            runs.update({row["runId"]: row for row in rows})
            if not body.get("hasMore"):
                break
            cursor = body.get("nextCursor")
            if not isinstance(cursor, str) or not cursor:
                raise ClaudexorUnavailable("malformed_response", "Claudexor run list promised a page it did not address")
        else:
            raise ClaudexorUnavailable("malformed_response", "Claudexor run list did not end")
    return list(runs.values())
