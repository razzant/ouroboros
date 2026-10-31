"""Seatbelt containment for one stand arm, proven by canaries before the arm boots.

An arm is a real server on a copy of the owner's data, so the containment is the whole
safety story. The profile (``sandbox-exec``, macOS) lets the arm read the machine,
because the copy legitimately reads the owner's delegated execution roots the same way
the live body does, and allows writes ONLY under the arm's own directory. It denies:
reads of the live stores, settings and engine data (a missed reference there would
take a lock on, or leak from, the live install); every outbound connection except
loopback, and loopback to the live app and its engine; signals to any process outside
the arm's own sandbox; and running ``node``, so no Claudexor daemon can start.

Every rule is checked before the arm runs, from inside the same profile: each denial
must fail and a write in the arm's directory must succeed. A canary that behaves
otherwise stops the arm (``CanaryFailure``) and the report names it.
"""

from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import sys
import textwrap
from typing import Dict, Iterable, List

DEFAULT_APP_PORT = 8765


class CanaryFailure(RuntimeError):
    """A containment rule did not hold; the arm must not boot."""


def live_engine_port(live_home: pathlib.Path) -> int:
    """The live install's engine port from its descriptor; 0 when it has none."""
    try:
        descriptor = pathlib.Path(live_home) / "data" / "claudexor" / "daemon" / "control-api.json"
        return int(json.loads(descriptor.read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def profile(arm_root: pathlib.Path, *, live_home: pathlib.Path, live_port: int = DEFAULT_APP_PORT,
            extra_write: Iterable[pathlib.Path] = ()) -> str:
    """The SBPL text for an arm copied from ``live_home`` (the folder holding the install's
    ``data/`` and ``repo/``). Later rules win, so each deny is narrowed by the allows after it."""
    arm = str(pathlib.Path(arm_root).resolve())
    live_data = pathlib.Path(live_home).resolve() / "data"
    engine_home = pathlib.Path.home() / ".claudexor"
    writes = "\n".join(f'    (subpath "{path}")' for path in (arm, *map(str, extra_write)))
    engine = live_engine_port(live_home)
    blocked_ports = [int(live_port)] + ([engine] if engine else [])
    blocked = "\n".join(f'    (remote ip "localhost:{port}")' for port in blocked_ports)
    return textwrap.dedent(f"""\
        (version 1)
        (allow default)
        (deny file-write*)
        (allow file-write*
        {writes}
            (literal "/dev/null") (literal "/dev/zero") (literal "/dev/dtracehelper")
            (regex #"^/dev/tty") (regex #"^/dev/fd/"))
        (deny file-read*
            (regex #"^{live_data}/state/[^/]*\\.sqlite")
            (regex #"^{live_data}/observability/packs/[^/]*\\.sqlite")
            (literal "{live_data}/settings.json")
            (subpath "{live_data}/claudexor")
            (subpath "{engine_home}"))
        (deny network-outbound)
        (allow network-outbound (remote ip "localhost:*") (remote unix-socket (subpath "{arm}")))
        (deny network-outbound
        {blocked})
        (deny signal)
        (allow signal (target same-sandbox))
        (deny process-exec (regex #"/node$") (regex #"claudexord"))
        """)


_CANARY = r'''
import json, os, signal, socket, subprocess, sys
arm, outside_pid, engine, live, app_port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
out = {}
def attempt(name, fn):
    try:
        fn(); out[name] = "allowed"
    except Exception as exc:
        out[name] = "denied:" + type(exc).__name__
def write(path):
    with open(path, "w") as fh: fh.write("canary")
    os.unlink(path)
def read(path):
    with open(path, "rb") as fh: fh.read(1)
def connect(port, host="127.0.0.1"):
    with socket.create_connection((host, port), timeout=3): pass
attempt("write_arm", lambda: write(os.path.join(arm, ".canary")))
attempt("write_live_data", lambda: write(live + "/data/.stand-canary"))
attempt("write_live_repo", lambda: write(live + "/repo/.stand-canary"))
attempt("write_home", lambda: write(os.path.expanduser("~/.stand-canary")))
attempt("write_tmp", lambda: write("/private/tmp/.stand-canary"))
attempt("read_live_store", lambda: read(live + "/data/state/current.sqlite"))
attempt("read_live_settings", lambda: read(live + "/data/settings.json"))
attempt("read_live_engine", lambda: read(live + "/data/claudexor/daemon/control-api.json"))
attempt("read_live_home", lambda: os.listdir(live))
attempt("connect_live_app", lambda: connect(app_port))
if engine:
    attempt("connect_live_engine", lambda: connect(engine))
attempt("connect_internet", lambda: connect(443, "1.1.1.1"))
attempt("signal_outside", lambda: os.kill(outside_pid, signal.SIGTERM))
attempt("exec_node", lambda: subprocess.run(["/usr/bin/env", "node", "--version"], check=True, capture_output=True, timeout=10))
print(json.dumps(out))
'''

# What each canary must show. ``read_live_home`` is ALLOWED on purpose: the copy reads the
# owner's execution roots as the live body does (reads cannot change them). A denial proves
# containment only as the sandbox's own refusal (EPERM), never as a missing file or a closed
# port, so a target that does not exist outside the sandbox is reported "absent", not passed.
EXPECTED = {"write_arm": "allowed", "read_live_home": "allowed", "exec_node": "denied"}
_READ_TARGETS = {"read_live_store": "data/state/current.sqlite", "read_live_settings": "data/settings.json",
                 "read_live_engine": "data/claudexor/daemon/control-api.json",
                 "write_live_data": "data", "write_live_repo": "repo"}


def run_canaries(arm_root: pathlib.Path, profile_path: pathlib.Path, python: str, *,
                 live_home: pathlib.Path, live_port: int = DEFAULT_APP_PORT) -> Dict[str, str]:
    """Prove every rule from inside the profile; raise ``CanaryFailure`` on any surprise."""
    live_home = pathlib.Path(live_home).resolve()
    target = subprocess.Popen(["/bin/sleep", "120"])  # an outside process the arm must not signal
    try:
        proc = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile_path), python, "-I", "-c", _CANARY,
                               str(pathlib.Path(arm_root).resolve()), str(target.pid), str(live_engine_port(live_home)),
                               str(live_home), str(int(live_port))],
                              capture_output=True, text=True, timeout=120)
        alive = target.poll() is None
    finally:
        target.kill()
        target.wait(timeout=10)
    if proc.returncode != 0:
        raise CanaryFailure(f"canary script failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    seen["outside_process_survived"] = "allowed" if alive else "denied"
    for name, rel in _READ_TARGETS.items():
        if not (live_home / rel).exists():
            seen[name] = "absent"
    wrong: List[str] = []
    for name, outcome in seen.items():
        if outcome == "absent":
            continue
        want = EXPECTED.get(name, "allowed" if name == "outside_process_survived" else "denied:PermissionError")
        if not outcome.startswith(want):
            wrong.append(f"{name}={outcome} (want {want})")
    if wrong:
        raise CanaryFailure("containment did not hold: " + "; ".join(wrong))
    return seen


def loopback_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":  # manual proof: python -m devtools.real_data.sandbox <arm_dir> <live_home>
    arm_dir, home = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    arm_dir.mkdir(parents=True, exist_ok=True)
    path = arm_dir / "arm.sb"
    path.write_text(profile(arm_dir, live_home=home), encoding="utf-8")
    print(json.dumps(run_canaries(arm_dir, path, sys.executable, live_home=home), indent=1))
