"""Serving witness of the O0 -> O1 readiness on Linux (qualification equipment, not product).

    <stdlib python> -I -S -B o0o1_witness_linux.py PYTHON PREFIX SERVER_PY ENV_FILE VERSION OUT_DIR SECONDS

One bounded observation of a real source server: SERVER_PY, an installation's own ``server.py``, started under PYTHON,
the installed runtime the readiness driver then runs on, with exactly ENV_FILE's environment (NAME=VALUE lines), in a
session of its own, with this process as its child subreaper, so nothing of its tree leaves unseen. In order:
  1. Refusals, then PREFIX's file listing (``__pycache__`` aside), then OUT_DIR, create-only.
  2. ``PYTHON SERVER_PY`` from the environment's HOME: its pid and kernel start time.
  3. Within SECONDS: the port its OUROBOROS_PORT_FILE names, then ``GET /api/health`` there answering 200; then the
     boot phase, followed until ``ready`` or ``failed`` or for at most SETTLE seconds more.
  4. While it serves, from the kernel (/proc): the process holding that port's listening socket, its executable, argv,
     start time and initial environment, the files it mapped inside PREFIX and its descendants; and the server's own
     binding record (``state/server_port.bindings.json`` of its data root: pid, port, start time), if it wrote one.
  5. A fixed helper on the serving argv[0], ``-I -B``: its executable, prefix and where its packages load from.
  6. The stop: SIGTERM, then SIGKILL to whatever is left of its tree after GRACE seconds; then every descendant this
     process inherited is ended and reaped. Exit status, lifetime and the port's release are recorded, and an end
     found before any stop signal is kept as such.
  7. PREFIX's listing again, and the server's own logs, copied.
OUT_DIR then holds the readiness driver's interpreter evidence (``check_interpreter``): verdict.txt
(``verdict=PASS|FAIL|UNKNOWN expected=served detail=...``, written last), interpreter.txt (the serving argv[0] as the
kernel reports it) and localfn.json (the helper's facts, ``prefix`` among them); beside them observation.json,
health.json, server.stdout, server.stderr and logs/.
PASS: that process answered 200 with status ``ok`` and runtime_version VERSION, and no boot phase read was ``failed``
(``ready`` is not required: a phase still ``serving`` when SETTLE ends passes, and the detail names the last one). It
ran on PYTHON (its kernel executable and argv[0]), held that port's listening socket alone, and had exactly the
environment given. Its own binding record, when present, names its pid, port and start time; when absent, PASS rests on
the HTTP answer and the kernel's facts, and the detail says so. The helper on it read PREFIX, PREFIX is unchanged, no
poll found it ended before the stop signalled it, and nothing of its tree is left. FAIL: one of those measured not to
hold (a fact read counts even when a later read failed; a fact not read is no contradiction). UNKNOWN: not established
(no answer within SECONDS while it ran, an unread fact, a stop not confirmed, an interruption).
A SIGTERM or SIGINT to this process is answered in its handler: every process below this one gets SIGKILL at once. No
server or helper starts once one has been observed: the check and the start are one step for the handler (both signals
blocked across them, the child given the original mask back before its exec), so one arriving meanwhile is handled as
the start returns and ends the process just started. Once one has been observed, an end by a signal (the server's or
the helper's) is not taken as its own, and an argv or environment read finished after it is no contradiction (an ending
process reads both empty). Every wait returns. A listing, copy or hash in progress stops at its next file or 1 MiB
block (a log cut short is kept and marked partial), and none starts afterwards. What remains is confirming that the
helper, the server and its tree ended (at most HURRY seconds per wait, one already under way too; the helper's output
read by then is kept) and the small record files, with the interruption in the verdict (UNKNOWN, or FAIL when a
contradiction was measured: an exit of its own, a wrong executable). Before OUT_DIR exists, it is a refusal.
Exit: 0 PASS, 1 FAIL, 3 UNKNOWN, 2 refused before anything started.
"""
import ctypes
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

SETTLE = 60   # seconds after the first answer for the boot phase to reach ready or failed (within SECONDS)
GRACE = 30    # seconds from SIGTERM to SIGKILL; the source launcher's own stop grace is 10 s
HURRY = 2     # seconds each confirmation of an end waits once interrupted, one under way too (30 and 10 otherwise)
HELPER_SECONDS = 120
LOOKS = 400   # health reads kept in the record
LOG_BYTES = 32 << 20   # a server log larger than this is listed, not copied
PR_SET_CHILD_SUBREAPER = 36
SIGNALS = (signal.SIGTERM, signal.SIGINT)
INTERRUPT = []   # "SIGTERM at <utc>" for each SIGTERM or SIGINT this process received
# The helper, run on the serving argv[0] with -I -B: what that interpreter is and where its packages load from (the
# lock's names, as the readiness driver reads them).
HELPER = """import json, os, sqlite3, sys
from importlib import metadata
modules = {}
for name, dist in (("yaml", "pyyaml"), ("starlette", "starlette"), ("uvicorn", "uvicorn")):
    try:
        module = __import__(name)
        modules[name] = {"file": os.path.realpath(module.__file__), "version": metadata.version(dist)}
    except Exception as exc:
        modules[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
print(json.dumps({"executable": sys.executable, "real": os.path.realpath(sys.executable),
                  "version": sys.version.split()[0], "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                  "sqlite": sqlite3.sqlite_version, "modules": modules}, sort_keys=True))
"""


class Refused(Exception):
    """Refused before anything started."""


class Cancelled(Exception):
    """An interruption observed inside optional work (a listing, a copy, a hash): that work stops there."""


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def real(path):
    return os.path.realpath(str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            cancelled()
            digest.update(block)
    return digest.hexdigest()


def write_new(path, text):
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)


def on_signal(signum, _frame):
    """An interruption: recorded, and every process below this one (the server's tree, its orphans, the helper) gets
    SIGKILL now, whatever the main flow is doing. It never raises into that flow: a failed scan is recorded instead."""
    INTERRUPT.append("%s at %s" % (signal.Signals(signum).name, utc()))
    try:
        signal_all(descendants(os.getpid()), signal.SIGKILL)
    except Exception as exc:   # e.g. /proc unreadable; the stop below still ends what it can, and says what is left
        INTERRUPT.append("the tree was not signalled at once (%s: %s)" % (type(exc).__name__, exc))


def cancelled():
    """Raise Cancelled once an interruption was observed: optional work checks it at every file or block."""
    if INTERRUPT:
        raise Cancelled(INTERRUPT[0])


def hurried(until):
    """UNTIL, or HURRY seconds from now if sooner once an interruption was observed. Applied at every turn of a wait,
    it ends a wait already under way at most HURRY seconds after the turn that first sees the interruption."""
    return min(until, time.monotonic() + HURRY) if INTERRUPT else until


def admitted(start):
    """``start(restore)``, a Popen whose preexec_fn is ``restore``, only while no interruption was observed; None when
    refused. SIGTERM and SIGINT stay blocked from the check through the start, so none is observed in between; the
    child gets this process's original mask back before its exec, and a signal that arrived meanwhile is handled as
    the block ends, ending the process just started (``on_signal``)."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    try:
        if INTERRUPT:
            return None
        return start(lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def read_env(path):
    env = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if not sep or not name or name in env:
            raise Refused("unreadable environment line %r in %s" % (line, path))
        env[name] = value
    return env


def listing(prefix):
    """PREFIX's files as an install would change them: path, size, mtime and mode; ``__pycache__`` aside, counted.
    An interruption stops it at the next folder or file (Cancelled)."""
    digest, files, cached = hashlib.sha256(), 0, 0
    for folder, dirs, names in os.walk(prefix):
        cancelled()
        dirs.sort()
        rel = os.path.relpath(folder, prefix)
        if "__pycache__" in rel.split(os.sep):
            cached += len(names)
            continue
        for name in sorted(names):
            cancelled()
            facts = os.lstat(os.path.join(folder, name))
            digest.update(("%s\0%d\0%d\0%o\n" % (os.path.join(rel, name), facts.st_size, facts.st_mtime_ns,
                                                  facts.st_mode)).encode("utf-8", "surrogateescape"))
            files += 1
    return {"sha256": digest.hexdigest(), "files": files, "pycache_files": cached}


# ------------------------------------------------------------------ kernel facts (/proc)
def stat_of(pid):
    """/proc/PID/stat: state, ppid, process group, session and start time (clock ticks after boot); None when gone."""
    try:
        raw = Path("/proc/%d/stat" % pid).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields = raw.rpartition(")")[2].split()   # the command name may hold spaces and parentheses
    return {"state": fields[0], "ppid": int(fields[1]), "pgrp": int(fields[2]), "session": int(fields[3]),
            "starttime": int(fields[19])}


def descendants(root):
    """Every visible process below ROOT, by parent links."""
    children = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            facts = stat_of(int(entry))
            if facts:
                children.setdefault(facts["ppid"], []).append(int(entry))
    found, todo = [], list(children.get(root, []))
    while todo:
        pid = todo.pop()
        found.append(pid)
        todo.extend(children.get(pid, []))
    return sorted(found)


def cmdline(pid):
    try:
        data = Path("/proc/%d/cmdline" % pid).read_bytes()
    except OSError:
        return None
    if data.endswith(b"\0"):
        data = data[:-1]
    return [part.decode("utf-8", "surrogateescape") for part in data.split(b"\0")] if data else []


def exe_of(pid):
    try:
        return os.readlink("/proc/%d/exe" % pid)
    except OSError:
        return None


def environ(pid):
    """The environment PID was started with (execve's), not later changes inside it."""
    pairs = {}
    for item in Path("/proc/%d/environ" % pid).read_bytes().split(b"\0"):
        if item:
            name, _, value = item.decode("utf-8", "surrogateescape").partition("=")
            pairs[name] = value
    return pairs


def listening(port, tables=("/proc/net/tcp", "/proc/net/tcp6")):
    """The inodes of the sockets listening (state 0A) on PORT, from the kernel's tables."""
    inodes = set()
    for table in tables:
        try:
            lines = Path(table).read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) > 9 and parts[3] == "0A" and int(parts[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(int(parts[9]))
    return inodes


def holders(inodes, proc="/proc"):
    """{pid: [inode, ...]}: the visible processes holding one of INODES."""
    found = {}
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        try:
            fds = os.listdir("%s/%s/fd" % (proc, entry))
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink("%s/%s/fd/%s" % (proc, entry, fd))
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]") and int(target[8:-1]) in inodes:
                found.setdefault(int(entry), set()).add(int(target[8:-1]))
    return {pid: sorted(owned) for pid, owned in found.items()}


def mapped(pid, prefix):
    """(the files PID mapped inside PREFIX, how many it mapped outside it)."""
    inside, outside = set(), set()
    for line in Path("/proc/%d/maps" % pid).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith("/"):
            path = parts[5].strip()
            (inside if path.startswith(prefix + os.sep) else outside).add(path)
    return sorted(inside), len(outside)


# ------------------------------------------------------------------ the server
def health(port):
    """(HTTP status or None, the JSON body or None, why not)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:%d/api/health" % port, timeout=5) as answer:
            status, body = answer.status, answer.read(1 << 20)
    except urllib.error.HTTPError as exc:
        return exc.code, None, "HTTP %d" % exc.code
    except (OSError, ValueError) as exc:
        return None, None, "%s: %s" % (type(exc).__name__, exc)
    try:
        parsed = json.loads(body.decode("utf-8"))
    except ValueError as exc:
        return status, None, "unreadable body (%s)" % exc
    return status, parsed, ""


def observe(proc, port_file, obs, deadline):
    """Until it serves and its boot settles, it exits, DEADLINE or an interruption: the port file, then its health."""
    port, first = None, None
    while not INTERRUPT:
        now = time.monotonic()
        code = proc.poll()
        if code is not None:
            if not (INTERRUPT and code < 0):   # an end by a signal once interrupted may be on_signal's SIGKILL
                obs["exited_after_serving" if obs["served"] else "exited_early"] = {"returncode": code, "at": utc()}
            return
        if now >= deadline or (first is not None and now >= first + SETTLE):
            return
        if port is None:
            try:
                text = port_file.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text.isdigit() and 0 < int(text) < 65536:
                port = int(text)
                obs["port_file"] = {"port": port, "at": utc()}
        if port is not None:
            status, body, error = health(port)
            boot = body.get("boot") if isinstance(body, dict) else None
            phase = boot.get("phase") if isinstance(boot, dict) else None
            if len(obs["looks"]) < LOOKS:
                obs["looks"].append({"at": utc(), "port": port, "status": status, "phase": phase, "error": error})
            if status == 200 and isinstance(body, dict):
                if obs["served"] is None:
                    obs["served"], first = {"at": utc(), "port": port, "health": body}, now
                obs["settled"] = {"at": utc(), "health": body}
                if phase in ("ready", "failed"):
                    return
        time.sleep(1)


def identify(pid, port, prefix, env):
    """The kernel's facts of the serving process, and the server's own binding record."""
    facts = {}
    try:
        inodes = listening(port)
        facts["listeners"] = sorted(inodes)
        facts["holders"] = {str(holder): owned for holder, owned in sorted(holders(inodes).items())}
        facts["stat"] = stat_of(pid)
        facts["exe"] = os.readlink("/proc/%d/exe" % pid)
        facts["exe_real"] = real(facts["exe"])
        facts["argv"] = cmdline(pid)
        if INTERRUPT:   # on_signal's SIGKILL may have preceded this read: an ending process reads it empty
            facts["argv_interrupted"] = INTERRUPT[0]
        live = environ(pid)
        if INTERRUPT:   # the same for this one
            facts["environ_interrupted"] = INTERRUPT[0]
        facts["environ_equal"] = live == env
        facts["environ_differs"] = sorted(name for name in set(live) | set(env) if live.get(name) != env.get(name))
        facts["mapped_inside"], facts["mapped_outside"] = mapped(pid, real(prefix))
        facts["descendants"] = [{"pid": child, "exe": exe_of(child), "argv": cmdline(child)}
                                for child in descendants(pid)]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip().replace("-", "")
        facts["start_token"] = "%d.%s" % (facts["stat"]["starttime"], boot_id)   # platform_layer.process_start_time
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        facts["error"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        bindings = json.loads((Path(env["OUROBOROS_DATA_DIR"]) / "state" / "server_port.bindings.json").read_text(
            encoding="utf-8"))
        facts["binding"] = bindings.get("main") if isinstance(bindings, dict) else None
    except (OSError, ValueError) as exc:
        facts["binding"], facts["binding_error"] = None, "%s: %s" % (type(exc).__name__, exc)
    return facts


def drain(proc, seconds):
    """PROC's (stdout, stderr) once it ended or was killed: read until both close and it is reaped, for at most SECONDS
    (hurried once interrupted). Retrying loses no output. Past that bound: what was read so far, PROC's returncode
    still None."""
    until, out, err = hurried(time.monotonic() + seconds), None, None
    while True:
        try:
            return proc.communicate(timeout=0.2)
        except subprocess.TimeoutExpired as exc:   # it carries everything read so far
            out, err = exc.stdout or out, exc.stderr or err
        until = hurried(until)
        if time.monotonic() >= until:
            return out, err


def run_helper(argv0, env):
    record = {"argv": [argv0, "-I", "-B", "-c", "HELPER"], "started": utc()}
    try:
        helper = admitted(lambda restore: subprocess.Popen(
            [argv0, "-I", "-B", "-c", HELPER], env={"PATH": "/usr/bin:/bin", "LC_CTYPE": "C.UTF-8",
                                                    "HOME": env["HOME"]},
            cwd=env["HOME"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=restore))
    except OSError as exc:
        record["error"] = "not started: %s: %s" % (type(exc).__name__, exc)
        return record
    if helper is None:
        record["error"] = "not started: interrupted before it (%s)" % INTERRUPT[0]
        return record
    until = time.monotonic() + HELPER_SECONDS   # its output is a few hundred bytes: no pipe fills while it runs
    while helper.poll() is None and time.monotonic() < until and not INTERRUPT:
        time.sleep(0.05)
    if helper.poll() is None:
        helper.kill()
        record["error"] = "interrupted" if INTERRUPT else "timed out after %ss" % HELPER_SECONDS
    elif INTERRUPT and helper.returncode < 0:   # an end by a signal once interrupted may be on_signal's SIGKILL, reaped
        record["error"] = "interrupted (it ended by signal %d)" % -helper.returncode
    stdout, stderr = drain(helper, 10)
    if helper.returncode is None:   # killed and not reaped within the drain: its end is not confirmed
        record["error"] += "; not reaped after its SIGKILL (its output as read by then)"
    record.update(returncode=helper.returncode, ended=utc(), stdout=(stdout or b"").decode("utf-8", "replace"),
                  stderr=(stderr or b"").decode("utf-8", "replace")[-4000:])
    try:
        facts = json.loads(record["stdout"])
    except ValueError:
        facts = None
    record["facts"] = facts if isinstance(facts, dict) and "error" not in record else None
    return record


# ------------------------------------------------------------------ the stop
def signal_all(pids, signum):
    for pid in pids:
        try:
            os.kill(pid, signum)
        except OSError:
            pass


def reap():
    """Reap every ended child of this process (the subreaper's orphans included)."""
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def wait_for(proc, seconds, interruptible):
    """PROC's exit within SECONDS, polled; an interruption cuts an interruptible wait short and hurries any other."""
    until = time.monotonic() + seconds
    while proc.poll() is None and time.monotonic() < until and not (interruptible and INTERRUPT):
        time.sleep(0.1)
        until = hurried(until)
    return proc.poll()


def stop(proc, record):
    """SIGTERM and its GRACE (none once interrupted), then SIGKILL to what is left of its tree; how it ended. An end
    its first poll finds came before any stop signal: kept as ``ended_before_stop``, its own unless it is an end by a
    signal once interrupted (that may be on_signal's SIGKILL), as in observe()."""
    record["requested"] = utc()
    code = proc.poll()
    if code is not None:
        record["ended_before_stop"] = {"returncode": code, "at": utc(), "own": not (INTERRUPT and code < 0)}
    else:
        if INTERRUPT:
            record["signal"] = "SIGKILL to its tree (interrupted)"
            signal_all([proc.pid] + descendants(proc.pid), signal.SIGKILL)
        else:
            record["signal"] = "SIGTERM"
            signal_all([proc.pid], signal.SIGTERM)
            if wait_for(proc, GRACE, True) is None:
                record["forced"] = "SIGKILL to its tree after %s" % (INTERRUPT and "an interruption" or "%ss" % GRACE)
                signal_all([proc.pid] + descendants(proc.pid), signal.SIGKILL)
        if wait_for(proc, HURRY if INTERRUPT else 30, False) is None:
            record["unreaped"] = proc.pid
    record["returncode"], record["exited"] = proc.returncode, utc()


def left_behind(record):
    """Every descendant of this process once the server is gone (as its subreaper, its orphans come here): each named,
    ended (SIGTERM, then SIGKILL; SIGKILL at once once interrupted) and reaped; whatever still remains is recorded."""
    me = os.getpid()
    record["left"] = [{"pid": pid, "exe": exe_of(pid), "argv": cmdline(pid)} for pid in descendants(me)]
    for signum in (signal.SIGTERM, signal.SIGKILL):
        if not descendants(me) or (signum == signal.SIGTERM and INTERRUPT):
            continue
        signal_all(descendants(me), signum)
        until = time.monotonic() + (HURRY if INTERRUPT else 10)
        while descendants(me) and time.monotonic() < until and not (signum == signal.SIGTERM and INTERRUPT):
            reap()
            time.sleep(0.2)
            until = hurried(until)
    reap()
    record["remaining"] = [{"pid": pid, "argv": cmdline(pid)} for pid in descendants(me)]


# ------------------------------------------------------------------ the verdict
def adjudicate(obs):
    """PASS, FAIL or UNKNOWN, and why: the decisive facts only (observation.json keeps every one)."""
    inputs, fails, unknown = obs["inputs"], [], []
    served, ident, helper = obs.get("served"), obs.get("identity"), obs.get("helper") or {}
    pid = (obs.get("server") or {}).get("pid")
    if obs.get("interrupted"):
        unknown.append("interrupted (%s)" % obs["interrupted"])
    if obs.get("error"):
        unknown.append("the witness failed: %s" % obs["error"].strip().splitlines()[-1][:300])
    if obs.get("spawn_error"):
        fails.append("the server did not start: %s" % obs["spawn_error"])
    if obs.get("exited_early"):
        fails.append("the server exited (status %s) before it served" % obs["exited_early"]["returncode"])
    if obs.get("exited_after_serving"):
        fails.append("the server exited by itself (status %s) after it served" % (
            obs["exited_after_serving"]["returncode"]))
    ended = (obs.get("stop") or {}).get("ended_before_stop") or {}
    if ended.get("own") and not (obs.get("exited_early") or obs.get("exited_after_serving")):   # the stop saw it first
        fails.append("the server exited by itself (status %s) %s, before any stop signal" % (
            ended["returncode"], "after it served" if served else "before it served"))
    if not served:
        if not (fails or unknown):
            unknown.append("no health answer within %ss" % inputs["seconds"])
    else:
        body = served["health"]
        if body.get("status") != "ok" or body.get("runtime_version") != inputs["version"]:
            fails.append("the health answer is %s %s, not ok %s" % (body.get("status"), body.get("runtime_version"),
                                                                    inputs["version"]))
        if any(look.get("phase") == "failed" for look in obs.get("looks") or []):
            boot = ((obs.get("settled") or {}).get("health") or {}).get("boot") or {}
            fails.append("the boot failed: %s" % str(boot.get("error"))[:300])
        if ident is None:
            if not (obs.get("exited_after_serving") or obs.get("interrupted") or obs.get("error")):
                unknown.append("the kernel facts were not read")
        else:   # a fact read is judged even when a later read failed; a fact not read is no contradiction
            if ident.get("error"):
                unknown.append("a kernel fact was not read: %s" % ident["error"])
            if "holders" in ident:
                owners = sorted(int(key) for key in ident["holders"])
                if not ident.get("listeners") or not owners:
                    unknown.append("no visible listener of port %s or its holder" % served["port"])
                elif owners != [pid]:
                    fails.append("port %s is held by %s, not by the server %s" % (served["port"], owners, pid))
            if ident.get("stat") is not None and ident["stat"].get("starttime") != (
                    (obs.get("server") or {}).get("birth") or {}).get("starttime"):
                fails.append("pid %s is not the process that was started" % pid)
            if "exe_real" in ident and ident["exe_real"] != inputs["python_real"]:
                fails.append("the kernel executable is %s, not %s" % (ident["exe_real"], inputs["python_real"]))
            argv_late = " (read once interrupted)" if ident.get("argv_interrupted") else ""   # maybe the ending
            environ_late = " (read once interrupted)" if ident.get("environ_interrupted") else ""   # process's
            if "argv" in ident and ident["argv"] != [inputs["python"], inputs["server_py"]]:   # None: its read failed
                (unknown if argv_late or ident["argv"] is None else fails).append("its argv is %s%s" % (
                    "not readable" if ident["argv"] is None else ident["argv"], argv_late))
            if "environ_equal" in ident and not ident["environ_equal"]:
                (unknown if environ_late else fails).append("its environment differs in %s%s" % (
                    ", ".join(ident.get("environ_differs") or []), environ_late))
            binding = ident.get("binding")
            if binding is not None and (binding.get("pid") != pid or binding.get("port") != served["port"] or (
                    "start_token" in ident
                    and (binding.get("fingerprint") or {}).get("start_time") != ident["start_token"])):
                fails.append("the server's own binding record names %s" % json.dumps(binding, sort_keys=True))
            facts = helper.get("facts")
            if not helper:
                if not obs.get("interrupted"):
                    unknown.append("the helper did not run")
            elif helper.get("error"):
                unknown.append("the helper: %s" % helper["error"])
            elif helper.get("returncode") != 0:
                fails.append("the helper exited %s on the serving interpreter" % helper.get("returncode"))
            elif facts is None:
                unknown.append("the helper's facts are unreadable")
            else:
                if real(facts.get("prefix") or "/nonexistent") != inputs["prefix_real"]:
                    fails.append("the serving prefix is %s, not %s" % (facts.get("prefix"), inputs["prefix_real"]))
                if facts.get("prefix") != facts.get("base_prefix"):
                    fails.append("the serving interpreter is a virtual environment")
                if facts.get("real") != inputs["python_real"]:
                    fails.append("the helper ran on %s" % facts.get("real"))
                for name, module in sorted((facts.get("modules") or {}).items()):
                    if "error" in module or not str(module.get("file") or "").startswith(inputs["prefix_real"] + "/"):
                        fails.append("%s: %s" % (name, module.get("error") or "loaded from %s" % module.get("file")))
    before, after = obs.get("prefix_before"), obs.get("prefix_after")
    if after is None:
        if not obs.get("interrupted"):
            unknown.append("the runtime was not listed again")
    elif after["sha256"] != before["sha256"]:
        fails.append("the runtime changed while the server ran (%d files before, %d after)" % (
            before["files"], after["files"]))
    stop_record = obs.get("stop")
    if obs.get("server"):
        if not stop_record:
            unknown.append("the stop was not recorded")
        else:
            if stop_record.get("unreaped"):
                unknown.append("the server %s was not reaped" % stop_record["unreaped"])
            if stop_record.get("remaining"):
                unknown.append("left after the stop: %s" % [row["pid"] for row in stop_record["remaining"]])
            if served and stop_record.get("port_released") is False:
                unknown.append("port %s still listens after the stop" % served["port"])
    verdict = "FAIL" if fails else "UNKNOWN" if unknown else "PASS"
    if verdict != "PASS":   # what was measured not to hold, then what was not established (an unconfirmed stop too)
        return verdict, "; ".join(fails + unknown)
    boot = ((obs.get("settled") or {}).get("health") or {}).get("boot")
    phase = boot.get("phase") if isinstance(boot, dict) else None
    return verdict, ("served by pid %s on port %s (kernel executable %s, argv[0] %s, %d files mapped inside the "
                     "prefix; last boot phase %s; own binding record %s) from %s to %s, exit status %s%s, %d left "
                     "after the stop; prefix %s unchanged (%d files)" % (
                         pid, served["port"], ident["exe_real"], ident["argv"][0], len(ident["mapped_inside"]),
                         phase or "not reported", "matched" if ident.get("binding") is not None else
                         "absent (not required: HTTP and kernel facts only)",
                         obs["spawned"], stop_record["exited"], stop_record["returncode"],
                         ", " + stop_record["forced"] if stop_record.get("forced") else "",
                         len(stop_record.get("left") or []), inputs["prefix_real"], after["files"]))


# ------------------------------------------------------------------ main
def copy_logs(env, out, copied):
    """The server's own logs into OUT/logs, each copied and hashed in one pass and entered in COPIED as it goes. An
    interruption stops it at the next file or 1 MiB block (Cancelled): a log cut short stays, marked partial."""
    source = Path(env["OUROBOROS_DATA_DIR"]) / "logs"
    if not source.is_dir():
        return
    (out / "logs").mkdir()
    for path in sorted(source.iterdir()):
        cancelled()
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        if size > LOG_BYTES:
            copied[path.name] = {"bytes": size, "copied": False}
            continue
        entry = copied[path.name] = {"bytes": size, "copied": "partial", "bytes_copied": 0}
        digest = hashlib.sha256()
        with open(path, "rb") as source_file, open(out / "logs" / path.name, "xb") as target:
            for block in iter(lambda: source_file.read(1 << 20), b""):
                cancelled()
                target.write(block)
                digest.update(block)
                entry["bytes_copied"] += len(block)
        copied[path.name] = {"bytes": entry["bytes_copied"], "sha256": digest.hexdigest()}


def after_stop(prefix, env, out, obs):
    """The optional evidence once the server is gone: PREFIX listed again, its logs copied, its binding record read
    back. None of it starts once interrupted; an interruption meanwhile stops it at the next file or block, and what
    was done stays, recorded as done."""
    try:
        cancelled()
        obs["prefix_after"] = listing(prefix)
        obs["logs"] = {}
        copy_logs(env, out, obs["logs"])
        obs["binding_after"] = (Path(env["OUROBOROS_DATA_DIR"]) / "state" / "server_port.bindings.json").read_text(
            encoding="utf-8")
    except Cancelled as exc:
        obs["evidence_cut"] = "interrupted (%s): the evidence after the stop is incomplete" % exc
    except OSError as exc:
        obs["after_error"] = "%s: %s" % (type(exc).__name__, exc)


def run(python, prefix, server_py, env_file, version, out_dir, seconds):
    for signum in SIGNALS:   # first: from here on an interruption is recorded and answered at once (on_signal)
        signal.signal(signum, on_signal)
    out, env = Path(out_dir), read_env(env_file)
    for name in ("HOME", "OUROBOROS_PORT_FILE", "OUROBOROS_DATA_DIR"):
        if not env.get(name, "").startswith("/"):
            raise Refused("the environment names no absolute %s" % name)
    if not (os.path.isabs(python) and os.path.isfile(python) and os.access(python, os.X_OK)):
        raise Refused("no executable interpreter %s" % python)
    if not (os.path.isabs(prefix) and os.path.isdir(prefix)) or not os.path.isfile(server_py):
        raise Refused("no prefix %s or no server %s" % (prefix, server_py))
    if not seconds.isdigit() or int(seconds) <= 0:
        raise Refused("SECONDS is a positive number of seconds")
    port_file, data = Path(env["OUROBOROS_PORT_FILE"]), Path(env["OUROBOROS_DATA_DIR"])
    if os.path.lexists(port_file):
        raise Refused("a port file is already there: %s" % port_file)
    if data.exists() and any(data.iterdir()):
        raise Refused("the data root %s is not fresh" % data)
    if os.path.lexists(out) or not out.parent.is_dir():
        raise Refused("%s exists or has no parent" % out)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise Refused("prctl(PR_SET_CHILD_SUBREAPER) failed: errno %d" % ctypes.get_errno())
    try:   # all hashing and listing before the start happen before OUT_DIR exists: an interruption is a refusal
        witness, env_sha256, before = sha256_file(__file__), sha256_file(env_file), listing(prefix)
    except Cancelled as exc:
        raise Refused("interrupted before anything started (%s)" % exc)
    except OSError as exc:
        raise Refused("an input is not readable (the runtime %s, the witness or the environment): %s" % (prefix, exc))
    out.mkdir()
    obs = {"witness": {"path": real(__file__), "sha256": witness, "python": sys.version, "pid": os.getpid(),
                       "uname": list(os.uname())},
           "inputs": {"python": python, "python_real": real(python), "prefix": prefix, "prefix_real": real(prefix),
                      "server_py": server_py, "env_file": env_file, "env_sha256": env_sha256,
                      "env_names": sorted(env), "version": version, "seconds": int(seconds), "settle": SETTLE,
                      "grace": GRACE, "hurry": HURRY},
           "started": utc(), "prefix_before": before, "prefix_after": None, "looks": [], "served": None,
           "settled": None, "identity": None, "helper": None, "stop": None, "interrupted": None}
    proc, began = None, time.monotonic()
    try:
        with open(out / "server.stdout", "xb") as stdout, open(out / "server.stderr", "xb") as stderr:
            obs["spawned"], began = utc(), time.monotonic()
            try:
                proc = admitted(lambda restore: subprocess.Popen(
                    [python, server_py], env=env, cwd=env["HOME"], stdin=subprocess.DEVNULL, stdout=stdout,
                    stderr=stderr, start_new_session=True, close_fds=True, preexec_fn=restore))
            except OSError as exc:
                obs["spawn_error"] = "%s: %s" % (type(exc).__name__, exc)
        if proc is None and "spawn_error" not in obs:
            obs["not_started"] = "interrupted before the server's start (%s)" % INTERRUPT[0]
        if proc is not None:
            obs["server"] = {"pid": proc.pid, "birth": stat_of(proc.pid)}
            observe(proc, port_file, obs, began + int(seconds))
            code = proc.poll() if obs["served"] else None   # an end after observe's last poll: recorded as observe does
            if code is not None and "exited_after_serving" not in obs and not (INTERRUPT and code < 0):
                obs["exited_after_serving"] = {"returncode": code, "at": utc(), "seen": "after the observation"}
            if obs["served"] and code is None and not INTERRUPT:
                obs["identity"] = identify(proc.pid, obs["served"]["port"], prefix, env)
                argv = obs["identity"].get("argv") or []
                if argv and argv[0].startswith("/"):
                    obs["helper"] = run_helper(argv[0], env)
    except Exception:   # the witness's own failure: kept, never a PASS; the stop below still runs
        obs["error"] = traceback.format_exc()
    if proc is not None:
        record = obs["stop"] = {}
        try:
            stop(proc, record)
            left_behind(record)
        except Exception:   # a stop that could not be followed to its end is not confirmed
            obs["error"] = obs.get("error") or traceback.format_exc()
            record["unreaped"] = record.get("unreaped") or (proc.pid if proc.poll() is None else None)
        record["lifetime_seconds"] = round(time.monotonic() - began, 1)
        if obs["served"]:
            record["port_released"] = not listening(obs["served"]["port"])
    after_stop(prefix, env, out, obs)   # optional evidence: never once interrupted, and cut at the next file or block
    obs["interrupted"] = "; ".join(INTERRUPT) or None
    verdict, detail = adjudicate(obs)
    obs.update(verdict=verdict, detail=detail, ended=utc())
    argv = (obs.get("identity") or {}).get("argv") or []
    write_new(out / "interpreter.txt", (argv[0] if argv else "") + "\n")
    if (obs.get("helper") or {}).get("facts") is not None:
        write_new(out / "localfn.json", json.dumps(obs["helper"]["facts"], indent=1, sort_keys=True) + "\n")
    write_new(out / "health.json", json.dumps({"served": obs["served"], "settled": obs["settled"]}, indent=1,
                                              sort_keys=True) + "\n")
    write_new(out / "observation.json", json.dumps(obs, indent=1, sort_keys=True, default=str) + "\n")
    write_new(out / "verdict.txt", "verdict=%s expected=served detail=%s\n" % (verdict, detail))
    print("%s %s" % (verdict, detail))
    return {"PASS": 0, "FAIL": 1}.get(verdict, 3)


def main(argv):
    if len(argv) != 7:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        return run(*argv)
    except Refused as exc:
        print("refused before anything started: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
