"""The room history reader child (``ouroboros.gateway.history_reader``) under the cost meter.

Since #4320 a room read (``GET /api/chat/history``) is built in one warm reader child per serving generation; the
server keeps the live inputs it captures (``history_live.capture``) and the HTTP delivery. The cost contract holds the
whole work of a read, not the server's share of it (#4348), so tests/_cost_driver.py starts this module in the reader's
place through the product's own spawn (custody, environment, direct interpreter) and adds what it counts here to the
boundary that sent the request. The meter is installed before production code is imported and the driver's racy clock
is held (``CLOCKS["racy_ns"]``: the archive listing and the project registry this build reads key their reuse on it);
then the production reader runs unchanged. The requests' work items carry the tag ``reader`` (``_w_submit``). Before
each reply frame, the work since the previous frame is appended to ``COST_READER_OUT`` under the request's id: the
driver's room reads are sequential, so that is this reply's work (``start``: the work before the READY frame).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from tests._cost_meter import BOUNDARY, CostMeter, tagged

TAG = "reader"
KINDS = ("counts", "callers", "paths", "statements", "names")


def _since(now: Dict[str, Dict[str, int]], before: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    return {kind: {key: value - before[kind].get(key, 0) for key, value in now[kind].items()
                   if value != before[kind].get(key, 0)} for kind in KINDS}


def _append(path: str, line: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(line, sort_keys=True) + "\n")


def main() -> None:
    meter = CostMeter(os.environ["OUROBOROS_DATA_DIR"], execution_root=os.environ.get("COST_EXECUTION_ROOT") or None)
    meter.__enter__()
    from ouroboros import utils

    utils._RACY_NS = int(os.environ["COST_RACY_NS"])
    from ouroboros.gateway import history_reader

    out, frame = os.environ["COST_READER_OUT"], history_reader._frame
    seen = [meter.snapshot(TAG)]

    def counted(output: Any, metadata: Dict[str, Any], body: bytes = b"") -> None:
        if "id" in metadata or "ready" in metadata:  # on disk before the driver can read the frame
            now = meter.snapshot(TAG)
            line = {"id": metadata.get("id", "start"), **_since(now, seen[0])}
            seen[0] = now
            tagged("meter", lambda: _append(out, line))
        frame(output, metadata, body)

    history_reader._frame = counted
    BOUNDARY.set(TAG)
    history_reader.main()


if __name__ == "__main__":
    main()
