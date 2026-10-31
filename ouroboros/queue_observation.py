"""Read queue projections against the last successfully published generation."""

import pathlib
import time

from ouroboros.utils import read_json_dict

_sleep = time.sleep  # the in-flight re-read's pause, kept apart from frozen test clocks
_monotonic = time.monotonic


def queue_stamp(document: dict, *, boot: bool = False) -> dict:
    # Boot can recover an older release's snapshot; live readers need its publisher.
    if boot and "generation" not in document:
        return document
    stamp = document.get("_queue_alive") or {}
    return stamp if document.get("generation") and document["generation"] == stamp.get("generation") else {}


def load_queue_document(drive_root, name="queue_snapshot.json") -> dict:
    path = pathlib.Path(drive_root) / "state" / name
    document = read_json_dict(path)
    if document is None:
        return {"_snapshot_invalid" if path.exists() else "_snapshot_missing": True}
    document["_queue_alive"] = read_json_dict(path.parent / "queue.alive") or {}
    if name == "queue_snapshot.json":
        from ouroboros.model_wait import execution_elapsed_seconds, quota_waited_seconds

        heartbeats = queue_stamp(document).get("heartbeats") or {}
        now = time.time()
        for row in document.get("running") or []:
            if not isinstance(row, dict):
                continue
            started = row.get("started_at") or 0
            started = started if isinstance(started, (int, float)) else 0
            heartbeat = heartbeats.get(str(row.get("id") or ""))
            row.update(runtime_sec=round(max(0.0, now - started), 2) if started > 0 else 0.0,
                       quota_wait_sec=quota_waited_seconds(row, now), execution_sec=execution_elapsed_seconds(row, now),
                       heartbeat_lag_sec=round(max(0.0, now - heartbeat), 2) if isinstance(heartbeat, (int, float)) and heartbeat > 0 else None)
    return document


DISPATCH_PUBLICATION_WAIT_SEC = 30.0  # the wait budget for an in-flight queue publication on a loaded host
DISPATCH_PUBLICATION_POLL_SEC = 0.05


class DispatchPublicationPending(RuntimeError):
    """A generated dispatch document is not certified yet; never permissive authority."""

    def __init__(self, message: str, *, probe_elapsed_sec: float = 0.0):
        super().__init__(message)
        self.probe_elapsed_sec = probe_elapsed_sec


def dispatch_probe(drive_root):
    """One read, with no sleep: a publication gap is distinct from a malformed document."""
    started = _monotonic()
    document = load_queue_document(drive_root, "queue_dispatch.json")
    if not document.get("generation"):
        return None
    if not queue_stamp(document):
        raise DispatchPublicationPending(
            f"root budget fence authority unavailable: {drive_root}/state/queue_dispatch.json",
            probe_elapsed_sec=max(0.0, _monotonic() - started))
    return document


def published_dispatch(drive_root, *, attempts=None, pause_sec=DISPATCH_PUBLICATION_POLL_SEC):
    """The dispatch document of the last successful publication: its generation certified by
    queue.alive, whatever its age (a stalled tick never stops a send). A publication in flight
    (the stamp is unlinked first) is re-read for up to DISPATCH_PUBLICATION_WAIT_SEC; None when
    it stays uncertified (issue #1754: a one-second window failed sends under load)."""
    if attempts is None:
        attempts = max(1, int(DISPATCH_PUBLICATION_WAIT_SEC / pause_sec))
    for attempt in range(attempts):
        document = load_queue_document(drive_root, "queue_dispatch.json")
        if not document.get("generation"):
            return None  # Only an uncertified publication can be in flight.
        if queue_stamp(document):
            return document
        if attempt + 1 < attempts:
            _sleep(pause_sec)
    return None
