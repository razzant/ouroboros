"""Owner-run probe: does a meeting service accept Ouroboros's browser participant?

A vendor (Zoom web, Google Meet, Microsoft Teams) is offered by ``meeting_join``
only after this probe recorded PASS for it. The probe needs a real meeting that
somebody else is in: the running participant companion opens the link in
Ouroboros's visible browser window, plays a test tone into the replaced
microphone, and after ``--seconds`` records what actually happened: whether the
page asked for and sent the microphone, whether audio bytes left, and whether
remote meeting audio arrived. Join from the window during that time (sign in,
click Join, let the host admit Ouroboros) and say something in the meeting.

    python3 scripts/probe_vendor.py --state-dir DIR --url https://meet.google.com/abc-defg-hij [--seconds 90]
    python3 scripts/probe_vendor.py --state-dir DIR --report

``DIR`` is this skill's state directory (``data/state/skills/meeting_browser``).
Only the standard library is used, so any python3 runs it; the companion does
the browser work. Without a run a vendor stays NOT_RUN.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from meeting_control import ControlUnavailable, request  # noqa: E402
from vendors import probe_table  # noqa: E402

# Joining by hand takes a while: the control reply waits for the whole probe.
REPLY_MARGIN_SEC = 180.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state-dir", default=os.environ.get("OUROBOROS_SKILL_STATE_DIR", ""), required=False)
    parser.add_argument("--url", default="", help="a real meeting link of the vendor to probe")
    parser.add_argument("--seconds", type=float, default=90.0, help="how long to stay and listen")
    parser.add_argument("--report", action="store_true", help="print PASS / FAIL / NOT_RUN per vendor")
    args = parser.parse_args(argv)
    state_dir = pathlib.Path(args.state_dir) if args.state_dir else None
    if args.report or not args.url:
        print(json.dumps(probe_table(state_dir), indent=2))
        return 0
    if state_dir is None:
        parser.error("--state-dir is required to reach the participant companion")
    try:
        reply = request(state_dir, "probe", {"url": args.url, "seconds": args.seconds},
                        timeout_sec=args.seconds + REPLY_MARGIN_SEC)
    except ControlUnavailable as exc:
        print(json.dumps({"ok": False, "code": "MEETING_COMPANION_UNAVAILABLE", "message": str(exc)}))
        return 1
    print(json.dumps(reply, indent=2))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
