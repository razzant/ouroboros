"""Frame sampler for stand arms: where each thread is, every 15 seconds, paths and lines only.

Loaded through ``PYTHONPATH`` by every Python process of an arm (``devtools.real_data.stand``
sets it); inert unless ``OUROBOROS_STAND_PROBE_DIR`` names a directory. Writes one JSONL
file per process. Never reads locals, never touches the server's state.
"""

import json
import os
import sys
import threading
import time

_DIR = os.environ.get("OUROBOROS_STAND_PROBE_DIR", "")
_EVERY = 15.0


def _short(path: str) -> str:
    for marker in ("/ouroboros/", "/supervisor/", "/lib/python3"):
        if marker in path:
            return marker.strip("/") + "/" + path.split(marker, 1)[1]
    return os.path.basename(path)


def _sample(out: str) -> None:
    import traceback

    while True:
        time.sleep(_EVERY)
        try:
            names = {thread.ident: thread.name for thread in threading.enumerate()}
            rows = []
            for ident, frame in sys._current_frames().items():
                stack = traceback.extract_stack(frame)[-10:]
                rows.append({"thread": names.get(ident, str(ident)),
                             "stack": [f"{_short(item.filename)}:{item.lineno}:{item.name}" for item in stack]})
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": time.time(), "argv0": (sys.argv or [""])[0][-60:], "threads": rows}) + "\n")
        except Exception:  # noqa: BLE001 - a sampler must never disturb the process it samples
            pass


if _DIR and os.path.isdir(_DIR):
    threading.Thread(target=_sample, args=(os.path.join(_DIR, f"frames-{os.getpid()}.jsonl"),),
                     name="stand-frame-sampler", daemon=True).start()
