"""The two seams where a Home↔execd build pair is admitted, or refused.

`remote_contracts` decides WHAT compatibility means; this module is the pair of
places that ASK, one on each side of the boundary. They live together because
they are one decision taken twice, and a reader checking "is a mismatched pair
refused before it can do damage?" needs both answers in one place — not one
buried in a 1500-line transport and the other in a 1600-line executor.

The seams are deliberately EARLY:

* Home asks at the session PREAMBLE — the first bytes the target emits, before a
  single frame is written and before any tool call has borrowed the session — and
  again at the handshake response, so a target cannot announce one contract set in
  its preamble and act on another.
* execd asks at the first frame it receives, which covers the skew Home cannot:
  a target running a NEWER execd than the Home talking to it, where the Home in
  question may predate the check entirely.

Neither seam is where the old failure appeared. That was PREPARE, inside an
unrelated tool call, as a bare `ValueError` about a policy field — an accurate
statement of a fact discovered far too late to be useful.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ouroboros.remote_contracts import (
    contract_set_compatible,
    contract_skew_refusal,
)
from ouroboros.remote_ssh_config import transport_error
from ouroboros.version import get_version


def admit_home_contract_set(
    peer_contract_set: Any,
    *,
    release: str,
    artifact_sha256: str = "",
    connection_id: str = "",
) -> None:
    """Home's refusal of a target whose contract set is not this build's.

    A transport error like every other bootstrap-phase refusal, so it travels the
    route the Connections surface, the CLI and the task result already read. The
    two builds and the owner's ACTION ride in `details` — the only slot on
    `RemoteWorkspaceError` that reaches the browser and `--json` unchanged, and now
    also the slot its `action` attribute is derived from.
    """

    if contract_set_compatible(peer_contract_set):
        return
    code, message, details = contract_skew_refusal(
        peer_contract_set,
        peer_build=release,
        local_build=get_version(),
        extra={
            "connection_id": str(connection_id or ""),
            "artifact_sha256": str(artifact_sha256 or ""),
            "refused_by": "home",
        },
    )
    raise transport_error(code, message, phase="bootstrap", details=details)


def admit_execd_contract_set(peer_contract_set: Any, *, release: str) -> None:
    """execd's refusal of a Home whose contract set is not this build's.

    The answer goes to STDERR and not to a control frame, and that is the point. A
    handshake carries no `request_id`, so the serve loop's diagnostic answer cannot
    speak for it; and no control kind exists that an older Home would recognize,
    because an unknown kind fails the session by contract. Stderr is the one channel
    every Home build already reads — the transport attaches it to `details.stderr` of
    the session error — so even a Home built before any of this gets a recognizable
    sentence instead of a silent disconnect. The `ExecdError` after it ends the
    session, which is the correct outcome once the pair is known to disagree.
    """

    if contract_set_compatible(peer_contract_set):
        return
    # Function-local, and the one import in this module that is: `execd_state` is
    # execd's PRIVATE durable-state module, and the Home transport imports this file
    # for the other seam. A module-level import would put execd's journal, custody and
    # CAS machinery on the Home transport's import graph to borrow one exception class.
    # Below the check, so the ordinary handshake never reaches for it at all.
    from ouroboros.execd_state import ExecdError

    code, message, details = contract_skew_refusal(
        peer_contract_set,
        peer_build="ouroboros-home",
        local_build=release,
        extra={"refused_by": "execd"},
    )
    print(
        json.dumps(
            {"execd_refusal": code, "message": message, "details": details},
            sort_keys=True,
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )
    raise ExecdError(code, message, phase="bootstrap", details=details)
