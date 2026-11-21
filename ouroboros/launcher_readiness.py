"""Probe the launcher's child for first display or a planned main-window replacement."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Optional

log = logging.getLogger("launcher")


def wait_for_server(port: int, timeout: Optional[float] = None, abort_event=None, proc=None, *, for_room=False) -> bool:
    """First display needs only HTTP 200. A replacement also waits out the child's room startup.

    Room startup is over once dialogue is ready and the ``projects`` step is done or failed, or on any terminal boot
    outcome. Both waits end on child exit or shutdown. A caller probing a server it does not own may set a timeout.
    """
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = None if timeout is None else time.time() + timeout
    unreadable = False
    while deadline is None or time.time() < deadline:
        if abort_event is not None and abort_event.is_set():
            return False
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    boot = _read_boot(response)
                    # Retry a cut body once. With an older 200-only probe the page proves its own room.
                    retry_body = boot is None and not unreadable
                    unreadable = boot is None
                    if not for_room or (not retry_body and not _room_starting(boot or {})):
                        if boot:
                            log.info("Agent server is serving: boot phase %s, dialogue_ready=%s, tasks_ready=%s",
                                     boot.get("phase"), boot.get("dialogue_ready"), boot.get("tasks_ready"))
                        return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def absent_rooms(port: int, rooms) -> set:
    """Rooms the child's registry positively has no active Project for, once its ``projects`` step is done.

    The one answer that counts is HTTP 200 with an empty list (CLIENT_CONTRACT §6: a deleting, tombstoned or unknown
    room). One failed read, another status or shape, a registry still reconciling or an older server keeps every room:
    temporary unavailability is never deletion. Only room numbers travel, never what was typed there."""
    def read(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
            return json.load(response) if response.status == 200 else None

    try:
        if not rooms or not any(step["name"] == "projects" and step["state"] == "done"
                                for step in read("/api/health")["boot"]["steps"]):
            return set()
        answers = {room: read(f"/api/projects?chat_id={int(room)}")["projects"] for room in rooms}
        if not all(isinstance(rows, list) for rows in answers.values()):
            return set()
        return {room for room, rows in answers.items() if not rows}
    except Exception:
        return set()


def _read_boot(response) -> Optional[dict]:
    try:
        boot = json.load(response).get("boot", {})
    except Exception:
        return None  # a cut-off response cannot confirm the end of room startup
    return boot if isinstance(boot, dict) else {}


def _room_starting(boot: dict) -> bool:
    if boot.get("phase") not in ("starting", "serving"):
        return False  # includes terminal boot outcomes and older servers without a boot document
    if boot.get("dialogue_ready") is not True:
        return True
    steps = boot.get("steps")
    return isinstance(steps, list) and not any(
        isinstance(step, dict) and step.get("name") == "projects" and step.get("state") in ("done", "failed")
        for step in steps
    )
