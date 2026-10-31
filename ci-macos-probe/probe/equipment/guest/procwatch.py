"""Guest-side process observer for a requalification of the unified Update (test equipment, not product code).

Copied from an earlier lab's guest/procwatch.py; changed only: OBSERVER_MARKS names this copy's
guest folder (this file's own folder, <namespace>/bin/), QB1_ names are printed in plain, and `watch` arms only
when the process table was read and every process already running holds the registration its claim needs
(--need; a refusal writes OUT_DIR/arm-failed and exits 3), records a later failed registration or unreadable
table and then exits 1. A process gone before its registration (ESRCH) is recorded as gone: its exit was not
observed. The earlier description follows.

Originally: guest-side process observer for an earlier native qualification.

v2 of an earlier lab's observer, changed only in three ways:
  - each snapshot also records `image`, the kernel's path of the executable the process runs (proc_pidpath:
    symlinks resolved), so a venv interpreter shows the Python it actually runs;
  - `watch` accepts extra marks after MAX_SECONDS (the run's TMPDIR, where the switch-time import probe runs);
  - the observer's own folder mark is /oulab/q7b4/bin/ and Q7B4_ names are printed in plain.

Runs inside the lab guest under a stable interpreter outside every run root (the verified 7.8.0 copy's
python-standalone, `-I -S -B`), as the same user as the app. It reads only what the kernel already exposes
to that user; it starts, signals and changes nothing in the app.

  selftest OUT_JSON
      Proves the exit-status method on this guest: a double-forked process (re-parented to launchd, so NOT
      this process's child) exits with status 7; kqueue EVFILT_PROC NOTE_EXIT|NOTE_EXITSTATUS must report it.
  snap OUT_JSON PID...
      One environment snapshot per PID (KERN_PROCARGS2: the environment the process received at its last exec).
  watch [--need exit+status|exit] OUT_DIR RUN_ROOT APP_PATH MAX_SECONDS [EXTRA_MARK...]
      Every process whose command names RUN_ROOT/, APP_PATH or an EXTRA_MARK: an environment snapshot when it is first seen
      and again when its command changes (an execve in place), and its exit status through kqueue
      (NOTE_EXIT|NOTE_EXITSTATUS). Writes OUT_DIR/armed once the processes already running are registered as
      --need requires (default exit+status; none running arms too), else OUT_DIR/arm-failed and exit 3;
      stops when OUT_DIR/stop exists or after MAX_SECONDS, exit 1 if a later registration failed. Events go to
      OUT_DIR/events.jsonl, snapshots to OUT_DIR/snapshots.jsonl.

Environment records keep every NAME, a sha256 prefix of every value (so equal values can be compared across
processes without recording them), and the plain value only for an allowlist of non-secret launch and
configuration names; a name that looks like a credential is never printed in plain.
"""
import ctypes
import datetime
import errno
import hashlib
import json
import os
import select
import subprocess
import sys
import time

NOTE_EXIT = 0x80000000
NOTE_EXITSTATUS = 0x04000000
SAFE_EXACT = {
    "HOME", "CFFIXED_USER_HOME", "TMPDIR", "PATH", "DEVELOPER_DIR", "LANG", "LC_ALL", "LC_CTYPE", "USER",
    "LOGNAME", "SHELL", "PWD", "OLDPWD", "SHLVL", "_", "__CFBundleIdentifier", "__CF_USER_TEXT_ENCODING",
    "PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONUSERBASE", "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX",
    "PYTHONUNBUFFERED", "PYTHONIOENCODING", "PYTHONSAFEPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__",
    "PIP_NO_INDEX", "PIP_DISABLE_PIP_VERSION_CHECK", "COMMAND_MODE", "MallocNanoZone",
}
SAFE_PREFIX = ("OUROBOROS_", "_PYI_", "OURO_", "CLAUDEXOR_RUN_SCOPE_", "FB1_", "Q7B4_", "QB1_", "XPC_", "LC_")
HASH_ONLY = {"OUROBOROS_PROJECTED_SETTINGS", "OUROBOROS_PROJECTED_PROPOSALS", "SSH_AUTH_SOCK"}
SECRETISH = ("KEY", "TOKEN", "SECRET", "PASSW", "CREDENTIAL", "AUTH", "COOKIE", "SESSION", "ACCOUNT", "BEARER")
OBSERVER_MARKS = ("procwatch.py", os.path.dirname(os.path.abspath(__file__)) + "/")  # this namespace's equipment


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def plain_allowed(key):
    if key in HASH_ONLY or any(mark in key.upper() for mark in SECRETISH):
        return False
    return key in SAFE_EXACT or key.startswith(SAFE_PREFIX)


def procargs(pid):
    """argv and environment process ``pid`` received at its last exec (KERN_PROCARGS2)."""
    libc = ctypes.CDLL(None, use_errno=True)
    argmax, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    if libc.sysctl((ctypes.c_int * 2)(1, 8), 2, ctypes.byref(argmax), ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), "KERN_ARGMAX")
    buf, size = ctypes.create_string_buffer(argmax.value), ctypes.c_size_t(argmax.value)
    if libc.sysctl((ctypes.c_int * 3)(1, 49, int(pid)), 3, buf, ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), "KERN_PROCARGS2")
    raw = buf.raw[:size.value]
    argc, rest = int.from_bytes(raw[:4], sys.byteorder), raw[4:]
    i = rest.index(b"\0")  # the exec path, then padding
    while i < len(rest) and rest[i] == 0:
        i += 1
    parts = rest[i:].split(b"\0")
    argv = [p.decode("utf-8", "replace") for p in parts[:argc]]
    env = {}
    for part in parts[argc:]:
        if not part:
            break
        key, _, value = part.decode("utf-8", "replace").partition("=")
        env[key] = value
    return argv, env


def image(pid):
    """The kernel's path of the executable ``pid`` runs (proc_pidpath), or an error text."""
    libc = ctypes.CDLL(None, use_errno=True)
    buf = ctypes.create_string_buffer(4096)  # PROC_PIDPATHINFO_MAXSIZE
    length = libc.proc_pidpath(int(pid), buf, ctypes.c_uint32(4096))
    if length <= 0:
        return "error: errno %d" % ctypes.get_errno()
    return buf.raw[:length].decode("utf-8", "replace")


def env_record(env):
    record = {"keys": sorted(env), "sha16": {}, "plain": {}}
    for key, value in env.items():
        record["sha16"][key] = hashlib.sha256(value.encode("utf-8", "surrogateescape")).hexdigest()[:16]
        if plain_allowed(key):
            record["plain"][key] = value if len(value) <= 600 else value[:600] + "...[%d chars]" % len(value)
    return record


def snapshot(pid, command="", ppid=None, pgid=None):
    row = {"utc": utc(), "pid": int(pid), "ppid": ppid, "pgid": pgid, "command": command[:800]}
    try:
        row["image"] = image(pid)
    except (OSError, AttributeError, ValueError) as exc:
        row["image"] = "error: %s: %s" % (type(exc).__name__, exc)
    try:
        argv, env = procargs(pid)
        row["argv"] = argv
        row["env"] = env_record(env)
    except (OSError, ValueError) as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
    return row


class Unreadable(Exception):
    """The process table could not be read: an observation not established, never an empty (cold) one."""


def table():
    try:
        done = subprocess.run(["/bin/ps", "-axww", "-o", "pid=,ppid=,pgid=,command="], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise Unreadable("%s: %s" % (type(exc).__name__, exc)) from exc
    if done.returncode != 0:
        raise Unreadable("ps exit %d: %s" % (done.returncode, (done.stderr or "").strip()[:300]))
    rows = []
    for line in done.stdout.splitlines():
        bits = line.split(None, 3)
        if len(bits) == 4 and bits[0].isdigit():
            rows.append((int(bits[0]), int(bits[1]), int(bits[2]), bits[3]))
    return rows


def decode(status):
    if os.WIFEXITED(status):
        return {"exited": True, "exit_code": os.WEXITSTATUS(status)}
    if os.WIFSIGNALED(status):
        return {"exited": False, "signal": os.WTERMSIG(status)}
    return {"exited": None, "raw_unrecognized": status}


def register(kq, pid):
    """NOTE_EXIT with the exit status; without the status only if the kernel refuses it (recorded)."""
    flags = select.KQ_EV_ADD | select.KQ_EV_ENABLE | select.KQ_EV_ONESHOT
    try:
        kq.control([select.kevent(pid, select.KQ_FILTER_PROC, flags, NOTE_EXIT | NOTE_EXITSTATUS)], 0, 0)
        return "exit+status"
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return "gone (%s)" % exc
        first = "%s" % exc
    try:
        kq.control([select.kevent(pid, select.KQ_FILTER_PROC, flags, NOTE_EXIT)], 0, 0)
        return "exit_only (status refused: %s)" % first
    except OSError as exc:
        return "unregistered (%s; %s)" % (first, exc)


def holds(how, need):
    """Whether a registration result gives what the claim needs (``exit+status`` or ``exit``)."""
    return how == "exit+status" or (need == "exit" and how.startswith("exit_only"))


def short_of(registered, need):
    """Registrations that fall short of ``need``; a process gone before its registration is not running."""
    return {pid: how for pid, how in registered.items() if not how.startswith("gone") and not holds(how, need)}


def exit_row(event):
    row = {"utc": utc(), "event": "exit", "pid": int(event.ident), "fflags": "0x%08x" % event.fflags,
           "data": int(event.data), "status_reported": bool(event.fflags & NOTE_EXITSTATUS)}
    if row["status_reported"]:
        row.update(decode(int(event.data)))
    return row


def selftest(out_json):
    read_end, write_end = os.pipe()
    child = os.fork()
    if child == 0:
        grandchild = os.fork()
        if grandchild == 0:
            os.close(read_end)
            os.setsid()
            time.sleep(2.0)
            os._exit(7)
        os.write(write_end, str(grandchild).encode())
        os._exit(0)
    os.close(write_end)
    grandchild = int(os.read(read_end, 64).decode())
    os.waitpid(child, 0)
    time.sleep(0.3)
    parent = [r[1] for r in table() if r[0] == grandchild]
    kq = select.kqueue()
    result = {"utc": utc(), "observer_pid": os.getpid(), "target_pid": grandchild,
              "target_ppid_at_registration": parent[0] if parent else None,
              "registration": register(kq, grandchild)}
    events = kq.control(None, 4, 10.0)
    result["events"] = [exit_row(event) for event in events]
    good = [e for e in result["events"] if e.get("status_reported") and e.get("exit_code") == 7]
    result["method_ok"] = bool(good) and result["target_ppid_at_registration"] == 1
    with open(out_json, "x") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    print(json.dumps({"method_ok": result["method_ok"], "registration": result["registration"]}))
    return 0 if result["method_ok"] else 1


def snap(out_json, pids):
    rows = [snapshot(pid) for pid in pids]
    with open(out_json, "x") as handle:
        json.dump(rows, handle, indent=1, sort_keys=True)
    return 0


def watch(out_dir, run_root, app_path, max_seconds, extra=(), need="exit+status"):
    if need not in ("exit+status", "exit"):
        raise SystemExit("--need is exit+status or exit")
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, "events.jsonl"), "a", buffering=1) as events, \
            open(os.path.join(out_dir, "snapshots.jsonl"), "a", buffering=1) as snaps:
        return _watch(out_dir, run_root, app_path, max_seconds, extra, need, events, snaps)


def _watch(out_dir, run_root, app_path, max_seconds, extra, need, events, snaps):
    marks = (run_root.rstrip("/") + "/", app_path) + tuple(mark for mark in extra if mark)
    me, kq, seen, registered, unreadable = os.getpid(), select.kqueue(), {}, {}, 0
    deadline, armed = time.monotonic() + float(max_seconds), False
    events.write(json.dumps({"utc": utc(), "event": "start", "observer_pid": me, "run_root": run_root,
                             "app": app_path, "max_seconds": max_seconds, "marks": list(marks),
                             "need": need}) + "\n")
    while time.monotonic() < deadline and not os.path.exists(os.path.join(out_dir, "stop")):
        try:
            rows = table()
        except Unreadable as exc:
            if not armed:  # unknown is not cold: nothing may proceed on an observation never established
                with open(os.path.join(out_dir, "arm-failed"), "x") as handle:
                    handle.write(json.dumps({"utc": utc(), "need": need, "unreadable": str(exc),
                                             "registered": registered}) + "\n")
                events.write(json.dumps({"utc": utc(), "event": "arm_failed", "unreadable": str(exc)}) + "\n")
                return 3
            unreadable += 1
            events.write(json.dumps({"utc": utc(), "event": "table_unreadable", "error": str(exc)}) + "\n")
            rows = []
        for pid, ppid, pgid, command in rows:
            if pid == me or not any(mark in command for mark in marks):
                continue
            if any(mark in command for mark in OBSERVER_MARKS):
                continue
            if seen.get(pid) != command:
                first = pid not in seen
                seen[pid] = command
                snaps.write(json.dumps(snapshot(pid, command, ppid, pgid), sort_keys=True) + "\n")
                events.write(json.dumps({"utc": utc(), "event": "seen" if first else "exec_changed", "pid": pid,
                                         "ppid": ppid, "pgid": pgid, "command": command[:400]}) + "\n")
            if pid not in registered:
                registered[pid] = register(kq, pid)
                events.write(json.dumps({"utc": utc(), "event": "registered", "pid": pid,
                                         "how": registered[pid]}) + "\n")
        if not armed:
            missing = short_of(registered, need)
            name = "arm-failed" if missing else "armed"
            with open(os.path.join(out_dir, name), "x") as handle:
                handle.write(json.dumps({"utc": utc(), "need": need, "registered": registered,
                                         "short": missing}) + "\n")
            if missing:  # a running identity the claim needs is not observed: nothing may proceed on it
                events.write(json.dumps({"utc": utc(), "event": "arm_failed", "short": missing}) + "\n")
                return 3
            armed = True
        for event in kq.control(None, 32, 0.2):
            events.write(json.dumps(exit_row(event), sort_keys=True) + "\n")
    failed = short_of(registered, need)
    events.write(json.dumps({"utc": utc(), "event": "end", "seen": len(seen), "registered": len(registered),
                             "registration_failed": failed, "table_unreadable": unreadable,
                             "gone_before_registration": sorted(pid for pid, how in registered.items()
                                                                if how.startswith("gone"))}) + "\n")
    return 1 if failed or unreadable else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["selftest"] and len(args) == 2:
        sys.exit(selftest(args[1]))
    if args[:1] == ["snap"] and len(args) >= 3:
        sys.exit(snap(args[1], [int(p) for p in args[2:]]))
    if args[:1] == ["watch"] and args[1:2] == ["--need"] and len(args) >= 7:
        sys.exit(watch(args[3], args[4], args[5], args[6], args[7:], need=args[2]))
    if args[:1] == ["watch"] and len(args) >= 5:
        sys.exit(watch(args[1], args[2], args[3], args[4], args[5:]))
    raise SystemExit(__doc__)
