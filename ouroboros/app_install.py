"""The unified Update's macOS app installer: captured stdlib, run by the staged official app's signed entry.

An Update that needs a new official app stages it beside the installed one. At activation
``body_adoption.authorize_update`` copies THIS file beside the switch helper it calls
(``state/body_adoption/``), and ``supervisor/update_native.hand_over`` starts it with the staged
app's signed source entry (``Contents/MacOS/ouroboros-shell``, Python with ``-I``), before the
activating restart is requested. No published app needs a new flag: the old app quits through its
own Quit, the new one opens through LaunchServices as a launch of this installation.

From its arming on, every step is journalled in the activation's own record (``handoff.json``:
``native.state`` and ``native.steps``, through the captured helper's ``native_journal``; before it
this process leaves that record and its lock to ``arm``, which never waits), and Panic, read before
each step and inside each wait, ends it with nothing more done:

1. ``custody``: it acts only once it runs detached from the activating server's process tree
   (``--detach`` forks it away and the spawner reads its pid; macOS reparents it to launchd) and
   the record names its exact pid, which custody (``INSTALLER_PURPOSE``, ``daemon`` scope)
   holds, so no launcher's cleanup of that server reaches it while Panic still does.
2. ``arm``: it waits until the restart's exit tail armed the transition on its stop owners'
   evidence (a refused, never requested or abandoned restart changes nothing here); that
   server then holds its exit until the old app's Quit (``body_adoption.native_hold``).
3. ``quit`` and ``wait``: the installed app is asked to quit by its exact path (its own Quit
   path; the answer proves nothing), then the recorded launcher and server must be provably
   gone; otherwise the transition is abandoned and nothing changes.
4. ``lock``: the launcher's single-instance lock is taken, so no launcher of this installation
   runs while the tree and the app change; a second launch meanwhile signals this process with
   SIGURG, which it ignores, and stays quiet.
5. ``switch``: files, index, branch and the generation environment switch, and the new tree's
   import probe runs on the interpreter the new generation runs, the new app's or a runtime venv's
   own (``native_switch``), BEFORE that app opens: its frozen bootstrap installs packages and
   imports the tree before any hooked entry.
6. ``exchange``: the staged app and the installed one swap in one call (``renamex_np`` with
   ``RENAME_SWAP``): the installed path never lacks an app; the previous one then moves to its
   backup beside. The new app's frozen code is recorded as an entry of the tree first. Just
   before the swap both bundles must still be the ones prepared (their identities: the folder
   and its ``Contents``), else the composition returns, so an app another installer put in
   place, a new folder or new contents in the old one (Homebrew with app-management
   permission), is not swapped away. That is a check, not a lock: a replacement between it and
   the swap, or after it, is found by the readback alone.
7. ``open``: the lock is released and the installed path opens through LaunchServices with
   automatic intent, this installation's configuration (``open --env``), its launch inputs and
   the ``--runtime-venv`` its body runs in, without the previous generation's process role
   (``update_native.successor_launch``).

A failed switch, probe or exchange writes the previous composition back and reopens the previous
app. An interruption is resumed by what the disk holds: an entry under the previous app finds
the tree switched while the previous app is still installed and returns it; after the exchange
the new app's next start goes on. UNQUALIFIED natively (plan §11): no run has proven an old app's
quit on that Apple Event, the exchange under LaunchServices, or kept TCC grants, notifications,
the WebKit profile, login item and CLI links.
"""

import argparse
import contextlib
import ctypes
import json
import os
import runpy
import subprocess
import sys
import time

OPEN, OSASCRIPT = "/usr/bin/open", "/usr/bin/osascript"
INSTALLER_PURPOSE = "update_app_installer"  # its process-custody purpose (``update_native.hand_over``)
_RENAME_SWAP = 0x2  # renamex_np(2): exchange two paths atomically (APFS)
# Test seam only (fault injection at a named step); production never sets it.
_FAIL_AT = os.environ.get("OUROBOROS_APP_INSTALL_FAIL_AT", "")


def exchange(first, second):
    """Swap two paths in one call: neither is ever missing. ``OSError`` where the volume cannot."""
    try:
        function = ctypes.CDLL(None, use_errno=True).renamex_np
    except AttributeError as exc:
        raise OSError(45, "renamex_np is unavailable here", first) from exc  # ENOTSUP
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    if function(os.fsencode(first), os.fsencode(second), _RENAME_SWAP) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), first)


def exchange_supported(folder):
    """``""`` when this volume swaps two directories in ``folder`` atomically, else why not (a measured fact)."""
    left, right = os.path.join(folder, ".exchange-probe-a"), os.path.join(folder, ".exchange-probe-b")
    try:
        for path, name in ((left, "a"), (right, "b")):
            os.makedirs(os.path.join(path, name))
        exchange(left, right)
        swapped = os.path.isdir(os.path.join(left, "b")) and os.path.isdir(os.path.join(right, "a"))
        return "" if swapped else "the exchange left both paths as they were"
    except OSError as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    finally:
        for path in (left, right):
            for name in ("a", "b"):
                try:
                    os.rmdir(os.path.join(path, name))
                except OSError:
                    pass
            try:
                os.rmdir(path)
            except OSError:
                pass


def _fail_point(name):
    if _FAIL_AT == name:
        sys.stderr.write("[app_install] fault injection at %s\n" % name)
        sys.stderr.flush()
        os._exit(137)


class _Installer:
    def __init__(self, data_dir, adoption_id, *, open_tool, osascript, wait_sec, arm_wait_sec, log):
        self.data_dir, self.adoption_id, self.log = os.path.abspath(data_dir), adoption_id, log
        self.helper = os.path.join(self.data_dir, "state", "body_adoption")
        self.open_tool, self.osascript, self.wait_sec, self.arm_wait_sec = open_tool, osascript, wait_sec, arm_wait_sec
        self.switch = runpy.run_path(os.path.join(self.helper, "switch.py"))  # the captured helper beside this file
        # Read once: an archived record removes the helpers beside it, and Panic must still be read after that.
        self.panic_reader = runpy.run_path(os.path.join(self.helper, "installation_stop.py"))["installation_panic"]
        self.native, self.repo, self.pid_lock, self.holding = {}, "", None, False

    def locked(self):
        """The helper's lock while its folder exists; once the record is archived there is nothing to guard."""
        return self.switch["_locked"](self.helper) if os.path.isdir(self.helper) else contextlib.nullcontext()

    # ---- the record ----------------------------------------------------------------------- #
    def handoff(self):
        """This transition's record; ``{}`` when it is gone or another one replaced it, ``None`` when unreadable."""
        try:
            handoff = self.switch["read_handoff"](self.helper)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            return None
        return handoff if isinstance(handoff, dict) and handoff.get("id") == self.adoption_id else {}

    def phase(self):
        handoff = self.handoff()
        return "unreadable" if handoff is None else str(handoff.get("phase") or "unknown") if handoff else "gone"

    def journal(self, state, entry=False, **facts):
        """Record one step; under the helper's lock, which this process may already hold (never twice)."""
        with contextlib.nullcontext() if self.holding else self.locked():
            if self.handoff():
                self.switch["native_journal"](self.helper, state, entry=entry, **facts)
        self.note(state, **facts)

    def note(self, step, **facts):
        if self.log:
            with open(self.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": step, "ts": time.time(), **facts}) + "\n")

    def panic(self):
        return bool(self.panic_reader(self.data_dir))

    # ---- waits ------------------------------------------------------------------------------ #
    def wait_until(self, done, seconds):
        """``"done"``, ``"panic"`` or ``"timeout"``; Panic is read before every look."""
        deadline = time.monotonic() + seconds
        while True:
            if self.panic():
                return "panic"
            if done():
                return "done"
            if time.monotonic() >= deadline:
                return "timeout"
            time.sleep(0.5)

    def take_launcher_lock(self):
        """The launcher's own single-instance lock (``platform_layer.pid_lock_acquire``'s file and flock)."""
        import fcntl  # stdlib-only and macOS-only: it cannot import the platform layer

        path = self.native.get("pid_file") or ""
        try:
            handle = open(path, "a")
        except OSError:
            return False
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        handle.truncate(0)
        handle.write(str(os.getpid()))
        handle.flush()
        self.pid_lock = handle
        return True

    def release_launcher_lock(self):
        if self.pid_lock is not None:
            path = self.native.get("pid_file") or ""
            try:
                os.unlink(path)  # as the launcher releases it: the next one writes its own pid
            except OSError:
                pass
            self.pid_lock.close()
            self.pid_lock = None

    def launch(self, step):
        """Open the installed path through LaunchServices with automatic intent and this installation's launch.

        ``open`` hands the app it starts its own environment (``--env`` only adds to it), and this process
        has the activating server's: it runs with the inherited inputs the record names and the configuration
        it carries (``update_native.successor_launch``), never that generation's process role.
        """
        if self.panic():
            self.journal("stopped", reason="panic_before_" + step)
            return False
        self.release_launcher_lock()
        launch = self.native.get("launch") or {}
        configured = dict(launch.get("env") or {})
        env = {key: os.environ[key] for key in launch.get("inherit") or () if key in os.environ}
        env.update(configured)
        argv = [self.open_tool, self.native["installed"]]
        for key, value in sorted(configured.items()):
            argv += ["--env", "%s=%s" % (key, value)]
        argv += ["--args", *(launch.get("args") or ["--launch-intent", "automatic"])]
        done = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        self.journal(step if done.returncode == 0 else step + "_failed", returncode=done.returncode)
        return done.returncode == 0

    # ---- the steps ---------------------------------------------------------------------------- #
    def run(self):
        handoff = self.handoff() or {}
        self.native, self.repo = dict(handoff.get("native") or {}), str(handoff.get("repo_dir") or "")
        if not self.native:
            return "refused", "not_native", "the record names no app transition for %s" % self.adoption_id
        # 1. custody: named in the record, and detached from the activating server's tree.
        named = lambda: (((self.handoff() or {}).get("native") or {}).get("installer") or {}).get("pid") == os.getpid()
        custody = self.wait_until(lambda: self.phase() == "gone" or (named() and os.getppid() == 1), 60.0)
        if custody == "panic":
            return "stopped", "panic", "Panic is recorded before custody; nothing ran"
        if self.phase() == "gone":
            return "refused", "not_armed", "the activation was closed before it armed; nothing was changed"
        if custody != "done":
            return "refused", "custody_unconfirmed", "this process is not the recorded, detached installer"
        self.note("custody", ppid=os.getppid())
        # 2. arm
        armed = self.wait_until(lambda: self.phase() not in ("authorized", "unreadable"), self.arm_wait_sec)
        if armed == "panic":
            return "stopped", "panic", "Panic is recorded while waiting for the activation; nothing ran"
        if self.phase() != "armed":
            return "refused", "not_armed", "the activation is %s, never armed; nothing was changed" % self.phase()
        self.journal("in_custody", ppid=os.getppid())
        # 3. quit and wait: its server holds its exit until this Quit reaches it through the launcher pipe
        if self.panic():
            return "stopped", "panic", "Panic is recorded before the quit; nothing ran"
        previous = self.native.get("previous") or {}
        self.journal("quitting")
        if not self.switch["process_gone"](previous.get("launcher")):  # never launch an app just to quit it
            subprocess.run([self.osascript, "-e", 'tell application "%s" to quit' % self.native["installed"]],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        gone = lambda: all(self.switch["process_gone"](previous.get(name)) for name in ("launcher", "server"))
        quit_wait = self.wait_until(gone, self.wait_sec)
        if quit_wait != "done":
            return self.abandon("panic" if quit_wait == "panic" else "old_app_running",
                                "the previous launcher and server did not exit")
        # 4. lock
        locked = self.wait_until(self.take_launcher_lock, 30.0)
        if locked != "done":
            return self.abandon("panic" if locked == "panic" else "launcher_running",
                                "another launcher of this installation holds its lock")
        self.journal("locked")
        # 5. switch, then 6. exchange, under the switch helper's lock throughout: no entry sees between
        outcome = None
        with self.locked():
            self.holding = True
            try:
                if self.panic():
                    self.switch["native_abandon"](self.repo, self.helper, "panic_stop")
                    return "stopped", "panic", "Panic is recorded before the switch; nothing was switched"
                phase, detail = self.switch["native_switch"](self.repo, self.helper)
                self.note("switch", phase=phase, detail=detail)
                if phase == "switched":
                    _fail_point("after_switch")
                    outcome = self.exchange_apps()
            finally:
                self.holding = False
        if phase in ("returned", "abandoned") or outcome == "returned":
            self.launch("reopened")
            return "not_applied", phase if outcome is None else "exchange_failed", detail or "the previous app reopened"
        if phase != "switched":
            return "stuck", phase or "unknown", detail
        if outcome == "panic":
            return "stopped", "panic", "Panic is recorded after the switch; the previous app stays installed"
        # 7. open
        _fail_point("before_open")
        if not self.launch("opened"):
            return "failed", "open_failed", "the new app is installed at %s; open it" % self.native["installed"]
        return "launched", None, "the new app was opened at %s" % self.native["installed"]

    def exchange_apps(self):
        """Under the helper's lock with the tree switched: ``"exchanged"``, ``"returned"`` or ``"panic"``."""
        if self.panic():
            self.journal("stopped", reason="panic_before_exchange")
            return "panic"
        identity = self.switch["bundle_identity"]
        changed = [side for side, path, wanted in (("installed", self.native["installed"], "previous_identity"),
                                                   ("staged", self.native["staged"], "new_identity"))
                   if identity(path) != self.native.get(wanted)]
        if changed:  # another installer moved a bundle folder (an app put there by hand, a brew upgrade): no swap
            self.switch["native_return"](self.repo, self.helper, "app_changed: %s" % ", ".join(changed))
            self.journal("exchange_refused", changed=changed)
            return "returned"
        self.journal("exchanging", entry=True)  # from here the new app may run: its frozen code is an entry
        _fail_point("after_exchange_intent")
        try:
            exchange(self.native["staged"], self.native["installed"])
        except OSError as exc:
            self.switch["native_return"](self.repo, self.helper, "exchange_failed: %s" % exc)
            self.journal("exchange_failed", error=str(exc))
            return "returned"
        self.journal("exchanged")
        _fail_point("after_exchange")
        try:  # the new app is in place either way; a backup left at the staged path is the GC's to finish
            settled = self.switch["native_settle_backup"](self.native)
        except OSError:
            settled = False
        self.journal("backed_up" if settled else "backup_owed")
        return "exchanged"

    def abandon(self, reason, detail):
        with self.locked():
            phase = self.switch["native_abandon"](self.repo, self.helper, "%s: %s" % (reason, detail)) \
                if self.handoff() else "gone"
        self.release_launcher_lock()
        return ("stopped" if reason == "panic" else "failed"), reason, "%s (the transition is %s)" % (detail, phase)


def install(data_dir, adoption_id, *, open_tool=OPEN, osascript=OSASCRIPT, wait_sec=180.0, arm_wait_sec=900.0,
            log=None):
    """custody, arm, quit, wait, lock, switch, exchange, open; ``{"status", "reason", "detail"}``."""
    installer = _Installer(data_dir, adoption_id, open_tool=open_tool, osascript=osascript, wait_sec=wait_sec,
                           arm_wait_sec=arm_wait_sec, log=log)
    try:
        status, reason, detail = installer.run()
    except Exception as exc:  # the record stays exactly as the last step left it: recovery reads it
        status, reason, detail = "failed", "crashed", "%s: %s" % (type(exc).__name__, exc)
    finally:
        installer.release_launcher_lock()
    result = {"status": status, "reason": reason, "detail": detail}
    installer.note("end", **result)
    return result


def _detach(log):
    """Fork away from the spawner's process tree; the spawner reads this process's pid from stdout."""
    child = os.fork()
    if child:
        sys.stdout.write("%d\n" % child)
        sys.stdout.flush()
        os._exit(0)
    os.setsid()  # its own session and group: no group signal of the activating server reaches it
    out = os.open(log or os.devnull, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    for fd in (1, 2):
        os.dup2(out, fd)
    os.close(out)


def main(argv):
    parser = argparse.ArgumentParser(prog="app_install.py")
    for name in ("--data-dir", "--adoption-id"):
        parser.add_argument(name, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--log", default="")
    parser.add_argument("--open", dest="open_tool", default=OPEN)
    parser.add_argument("--osascript", default=OSASCRIPT)
    parser.add_argument("--wait-sec", type=float, default=180.0)
    parser.add_argument("--arm-wait-sec", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.detach:
        _detach(args.log)
    done = install(args.data_dir, args.adoption_id, open_tool=args.open_tool, osascript=args.osascript,
                   wait_sec=args.wait_sec, arm_wait_sec=args.arm_wait_sec, log=args.log or None)
    return 0 if done["status"] == "launched" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
