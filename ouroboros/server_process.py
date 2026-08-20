"""Facts one server process shares with every server leaf.

The drive root it was launched against, the ``server`` logger every server
module writes to, and the restart-request signals plus the setter that raises
them. These live below the composition root so a leaf can read them without
importing ``server`` back.
"""

import logging
import os
import pathlib
import threading


DATA_DIR = pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR",
    pathlib.Path.home() / "Ouroboros" / "data"))


log = logging.getLogger("server")


_restart_requested = threading.Event()
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
