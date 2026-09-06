"""Facts one server process shares with every server leaf.

The drive root it was launched against, the ``server`` logger every server
module writes to, and the restart-request signals plus the setter that raises
them. These live below the composition root so a leaf can read them without
importing ``server`` back.
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading


DATA_DIR = pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR",
    pathlib.Path.home() / "Ouroboros" / "data"))


log = logging.getLogger("server")


_restart_requested = threading.Event()
# Set FIRST in the lifespan teardown: the supervisor loop reads it in its
# ``while`` and in its crash handler, so the bus/Manager being torn down by
# the shutdown itself never counts as a loop crash (no false "died after 3
# consecutive crashes" alarm on a graceful window close / SIGTERM).
_supervisor_stop = threading.Event()


# Set only when the OWNER asked for the restart (the chat Restart button, and the
# control endpoints that restart on the owner's behalf). The single fact the
# re-exec needs to decide whether the runtime-mode ratchet pin rides along.
_owner_restart_requested = threading.Event()


def _request_restart_exit(owner: bool = False) -> None:
    """Signal server shutdown with restart exit code.

    ``owner`` is the ONE fact the re-exec needs: an owner-initiated restart
    re-reads the runtime mode from settings, an agent- or supervisor-initiated
    one keeps inheriting the boot pin (see server_control.restart_current_process).
    """
    if owner:
        _owner_restart_requested.set()
    _restart_requested.set()
