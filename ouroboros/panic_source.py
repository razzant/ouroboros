"""Best-effort Panic evidence; diagnostic failure never holds the emergency door.

Only receiving ingress chooses owner_surface. Client labels are observations,
never identity. Each observation is retained in the existing event log; the
atomic latest-source projection is enriched only while it still names that
observation. Unknown process identity remains null rather than a guessed owner.
"""
from __future__ import annotations

import logging
from itertools import islice
import os
from pathlib import Path
import threading
import uuid

log = logging.getLogger(__name__)


def _record_source(root, *, ingress, owner_surface="unknown", peer=None, client_surface=None, skill_name=None, receiver_is_server=True):
    from ouroboros.utils import append_jsonl, update_json_locked, utc_now_iso
    surface = ({str(k)[:64]: str(v)[:256] for k, v in islice(client_surface.items(), 16)}
               if isinstance(client_surface, dict) else None)
    record = {"record_id": uuid.uuid4().hex, "ingress": ingress, "owner_surface": owner_surface,
              "time": utc_now_iso(), "receiver_pid": os.getpid(),
              "server_pid": os.getpid() if receiver_is_server else None,
              "server_pid_basis": "receiver" if receiver_is_server else "unknown",
              "peer": {"address": None, "pid": None, "executable": None, "command": None, "ppid": None,
                       **(peer or {})}, "client_surface": surface, "skill_name": skill_name}

    def write():
        try:
            path = Path(root) / "state" / "panic_source.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            def publish(*, enrichment=False):
                # The event log retains causal sources even when another ingress
                # (e.g. launcher fallback) becomes the latest projection.
                if not append_jsonl(Path(root) / "logs" / "events.jsonl",
                                    {"type": "panic_source", "enrichment": enrichment, **record},
                                    lock_timeout_sec=0):
                    log.warning("Panic source event could not be recorded")
                # Initial writes can also arrive out of order. Capture time is
                # diagnostic ordering only; it never admits or rejects Panic.
                update_json_locked(path, lambda current: (record if (
                    current.get("record_id") == record["record_id"] if enrichment else
                    str(current.get("time") or "") <= record["time"]) else None),
                    timeout_sec=.1)
            publish()
            if not receiver_is_server:
                from ouroboros.utils import read_json_dict
                saved = read_json_dict(Path(root) / "state" / "server_process.json") or {}
                pid = saved.get("pid")
                if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
                    record.update(server_pid=pid, server_pid_basis="launcher_record")
                    publish(enrichment=True)
            if record["peer"].get("address"):
                from ouroboros.platform_layer import local_connection_peer
                measured = local_connection_peer(record["peer"]["address"])
                for key in ("pid", "ppid", "executable", "command"):
                    if key in measured:
                        record["peer"][key] = measured[key] if key in {"pid", "ppid"} else str(measured[key])[:2048]
                publish(enrichment=True)
        except Exception:
            log.exception("Panic source evidence could not be recorded")
    try:
        threading.Thread(target=write, name="panic-source", daemon=True).start()
    except Exception:
        log.exception("Panic source recorder could not start")


def record_source(root, **evidence):
    try:
        _record_source(root, **evidence)
    except Exception:
        log.exception("Panic source evidence could not be prepared")


def source_evidence(request, kind, *, skill_name=None, client_surface=None):
    client = getattr(request, "client", None)
    return {"ingress": kind, "owner_surface": "unknown",
            "peer": {"address": list(client) if client else None},
            "client_surface": client_surface, "skill_name": skill_name}


def ingress_record(request, kind, *, skill_name=None, client_surface=None):
    from ouroboros.config import resolve_data_dir
    record_source(resolve_data_dir(), **source_evidence(request, kind, skill_name=skill_name,
                                                       client_surface=client_surface))
