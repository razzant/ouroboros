"""Process custody of one private root: a harness-owned kill-on-close Job created before the first start.

Every process the qualification starts (the released app, a source launcher, a direct entry, a holder,
an observer) is created suspended, assigned to the mode's Job and only then resumed, so nothing it
spawns can start outside the Job: the frozen app's hand-over successor, the launcher's server (in the
launcher's own nested Job), the venv redirector's child, the waited-on child of a direct entry. A stop is
``TerminateJobObject`` plus confirmation that the Job holds no process and that no member of its census still
reads alive; a survivor is never assumed gone.
A Job that has received ``TerminateJobObject`` never admits another start: assignment fails while a Job is
terminating, and that state can outlast the moment its count and process list read empty, with nothing to read its
end. After a confirmed stop the next start opens a fresh Job; the terminated one stays open, retired, until the
teardown re-reads and closes it. After a stop that was not confirmed nothing more is started.
A process whose image lies under a scope root but is NOT in the Job is an escape (breakaway, a detached
start): reported, never ignored. Nothing here signals a process by PID alone. A census that could not be
read completely is ``None``, never an empty one. The teardown joins every holder's output pump before any
evidence is copied.
"""
import json
import os
import queue
import subprocess
import threading
import time
from time import perf_counter_ns

from wq_verdict import under
from wq_win import CREATE_NO_WINDOW, CREATE_SUSPENDED

STOP_CODE = 0x5751  # "WQ": every process the harness stops reads this exit code
TEARDOWN_ALLOWANCE = 90  # seconds, shared by a teardown's stop and then its drain of the holders' output


class Interactive:
    """A started holder: JSON-line requests on stdin, replies read by a pump thread that also writes the
    holder's transcript. The pump runs until the pipe's last bytes are read, which can be after the process
    is gone; the teardown joins it (``Custody.drain``) before evidence is copied. Its error is kept."""

    def __init__(self, proc, transcript):
        self.proc, self.lines, self.serial, self.transcript = proc, queue.Queue(), 0, transcript
        self.error = None
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self):
        try:
            for line in iter(self.proc.stdout.readline, b""):
                with open(self.transcript, "ab") as fh:
                    fh.write(line)
                self.lines.put(line)
        except Exception as exc:  # the transcript stopped part-way: incomplete evidence, never a silent end
            self.error = repr(exc)
        finally:
            self.lines.put(None)

    def reply(self, timeout=60):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        return None if line is None else json.loads(line.decode("utf-8"))

    def request(self, op, timeout=60):
        self.serial += 1
        try:
            self.proc.stdin.write((json.dumps({"id": self.serial, "op": op}) + "\n").encode())
            self.proc.stdin.flush()
        except OSError:
            return None
        answer = self.reply(timeout)
        return answer if isinstance(answer, dict) and answer.get("id") == self.serial else None


class Custody:
    def __init__(self, win, raw, label, scope_roots):
        self.win, self.raw, self.label, self.scope_roots = win, raw, label, [str(root) for root in scope_roots]
        self.job, self.started, self.closed, self.pumps = win.job(), [], False, []
        self.last_stop_ns = None  # observation only: when the last stop_all finished (high-resolution clock)
        # ``terminated``: None while the current Job never received TerminateJobObject, else that stop's reason and
        # its ``confirmed`` (None until the stop finished). ``retired``: the terminated Jobs a confirmed stop replaced.
        self.generation, self.terminated, self.retired = 0, None, []
        raw.emit("custody_open", mode=label, scope_roots=self.scope_roots)

    def _admitting_job(self):
        """Make the current Job one that can admit a start (see the module docstring); raise, starting nothing,
        when it cannot. A fresh Job replaces a terminated one only after that Job's stop was confirmed; the
        terminated one is kept open (kill-on-close still holds whatever it might hold) for the teardown."""
        if self.terminated is None:
            return
        if self.terminated.get("confirmed") is not True:
            raise RuntimeError(f"job {self.generation} was terminated ({self.terminated.get('reason')!r}) and its stop"
                               f" is not confirmed ({self.terminated.get('confirmed')!r}): nothing more is started")
        fresh = self.win.job()  # raises: the current Job stays the terminated one, and nothing was started
        self.retired.append({"generation": self.generation, "job": self.job, "stop": self.terminated["reason"]})
        self.job, self.generation, self.terminated = fresh, self.generation + 1, None
        self.raw.emit("job_renewed", mode=self.label, retired=self.retired[-1]["generation"], job=self.generation,
                      after_stop=self.retired[-1]["stop"])

    def spawn(self, label, argv, *, env, cwd, logs, interactive=False):
        """Start ``argv`` suspended, put it in the Job, resume it; return ``(proc, identity, interactive)``."""
        os.makedirs(logs, exist_ok=True)
        self.raw.emit("spawn_intent", mode=self.label, label=label, argv=[str(a) for a in argv], cwd=str(cwd))
        since_stop_ms = None if self.last_stop_ns is None else (perf_counter_ns() - self.last_stop_ns) / 1e6
        try:
            self._admitting_job()
        except BaseException as exc:  # no process was created: nothing to read or clean up
            self.raw.emit("spawn_refused", mode=self.label, label=label, error=repr(exc), stage="job",
                          since_stop_ms=since_stop_ms, observed=None, pid=None, cleanup=None)
            raise
        out = open(os.path.join(logs, f"{label}.stdout"), "ab")
        err = open(os.path.join(logs, f"{label}.stderr"), "ab")
        try:
            proc = subprocess.Popen([str(a) for a in argv], env=env, cwd=str(cwd),
                                    stdin=subprocess.PIPE if interactive else subprocess.DEVNULL,
                                    stdout=subprocess.PIPE if interactive else out, stderr=err,
                                    creationflags=CREATE_SUSPENDED | CREATE_NO_WINDOW)
        finally:
            out.close()
            err.close()
        stage = "assign"
        try:
            self.win.assign(self.job, int(proc._handle))
            stage = "birth"
            identity = {"pid": proc.pid, "birth": self.win.birth(int(proc._handle))}
            stage = "resume"
            self.win.resume(int(proc._handle))
        except BaseException as exc:
            # ``stage`` is the call that refused. Only at "assign" is the child outside the Job; at "birth" it is a
            # member, and at "resume" it may already be running. It was created suspended, but nothing here
            # measures that, and the cleanup below is the only thing that ends it.
            observed = self._refusal_readings(proc)  # before the kill below changes what they would say
            cleanup = {"kill": "not_run", "wait": "not_run"}
            try:
                try:
                    proc.kill()
                except BaseException as kill_exc:
                    cleanup["kill"] = repr(kill_exc)
                    raise
                cleanup["kill"] = "ok"
                try:
                    proc.wait(timeout=30)
                except BaseException as wait_exc:
                    cleanup["wait"] = repr(wait_exc)
                    raise
                cleanup["wait"] = "ok"
            finally:  # the refusal is recorded after, never instead of, the cleanup, and whether or not it failed
                self.raw.emit("spawn_refused", mode=self.label, label=label, error=repr(exc), stage=stage,
                              since_stop_ms=since_stop_ms, observed=observed, pid=proc.pid, cleanup=cleanup)
            raise
        self.started.append({"label": label, **identity, "job": self.generation})
        self.raw.emit("spawned", mode=self.label, label=label, since_stop_ms=since_stop_ms, job=self.generation,
                      **identity)
        holder = Interactive(proc, os.path.join(logs, f"{label}.stdout")) if interactive else None
        if holder is not None:
            self.pumps.append(holder)
        return proc, identity, holder

    def _refusal_readings(self, proc):
        """What the handle and the Job answer at a refused start, one bounded read each, before any cleanup. A
        reader that fails is its own ``read_error``, never a clean answer, and nothing here can replace the
        original exception or skip the cleanup that follows. ``exit_code`` ``None`` reads still active;
        ``in_any_job`` passes a NULL Job to IsProcessInJob: whether this process is in any Job at all."""
        def read(reader):
            try:
                return reader()
            except BaseException as exc:
                return {"read_error": repr(exc)}
        handle = lambda: int(proc._handle)  # noqa: E731 - read inside each reader's own guard
        return {"exit_code": read(lambda: self.win.exit_code(handle())),
                "birth": read(lambda: self.win.birth(handle())),
                "in_this_job": read(lambda: self.win.in_job(handle(), self.job)),
                "in_any_job": read(lambda: self.win.in_job(handle(), None)),
                "job_active": read(lambda: self.win.active(self.job)),
                "job_pids": read(lambda: self.win.pids(self.job)[:32])}

    def members(self):
        """Every process in the Job, by one handle each (identity, image, command line, parent); ``None`` when
        the census cannot be taken completely (a failed walk, an incomplete Job list)."""
        try:
            parents = {pid: ppid for pid, (ppid, _exe) in self.win.snapshot().items()}
            return [self.win.identity(pid, self.job, parents) for pid in self.win.pids(self.job)]
        except (OSError, RuntimeError, ValueError) as exc:
            self.raw.emit("members_unreadable", mode=self.label, error=repr(exc))
            return None

    def alive(self, identity):
        """Whether the process named by ``identity`` (pid AND birth) still runs in this Job: ``False`` only when
        measured (its PID absent, its exit read, or the PID now another process's), ``None`` when unobserved.
        The reading behind any answer but ``True`` is recorded: a check's ``None`` alone cannot name its cause."""
        if not isinstance(identity, dict):
            return None
        try:
            row = self.win.identity(identity["pid"], self.job)
        except (OSError, RuntimeError, ValueError) as exc:
            row = {"alive": None, "read_error": repr(exc)}
        if row.get("alive") is None:
            state = None
        elif row.get("alive") is False or row.get("birth") != str(identity.get("birth")):
            state = False
        else:
            state = True if row.get("in_job") is True else None
        if state is not True:
            self.raw.emit("process_read", mode=self.label, pid=identity.get("pid"), birth=identity.get("birth"),
                          state=state, row=row)
        return state

    def not_gone(self, census, end):
        """The live members of ``census`` not measured gone by ``end`` (monotonic), each with its last ``alive``
        reading: a termination has only started when the call returns, and the Job's count can read 0 before it
        ended. An unread member is a birth-less row, nothing to wait for; it stays disclosed in the census."""
        waiting = [{"pid": row.get("pid"), "birth": row.get("birth")} for row in census or [] if row.get("alive")]
        while True:
            waiting = [{**row, "state": self.alive(row)} for row in waiting]
            waiting = [row for row in waiting if row["state"] is not False]
            if not waiting or time.monotonic() >= end:
                return waiting
            time.sleep(0.2)

    def stop_all(self, reason, deadline=TEARDOWN_ALLOWANCE):
        """Terminate every process of the Job; confirmed only when the Job's active count reads 0 (the primary
        custody evidence) and no member of the census still reads alive by the deadline. One left unread is
        disclosed in ``not_gone``, unjudged here. A census or terminate that fails is recorded and never skips
        the count. From the terminate attempt on, this Job admits no further start (``_admitting_job``)."""
        result = {"reason": reason, "before": None, "job": self.generation}
        try:
            result["before"] = self.members()
        except Exception as exc:
            result["census_error"] = repr(exc)
        self.terminated = {"reason": reason, "confirmed": None}  # set before the call: a failed one may have begun
        try:
            self.win.terminate(self.job, STOP_CODE)
            result["terminate"] = "ok"
        except Exception as exc:
            result["terminate"] = repr(exc)
        end, active = time.monotonic() + deadline, None
        while time.monotonic() < end:
            try:
                active = self.win.active(self.job)
            except Exception as exc:
                active = repr(exc)
                break
            if active == 0:
                break
            time.sleep(0.2)
        result["active_after"] = active
        result["not_gone"] = self.not_gone(result["before"], end)
        result["confirmed"] = active == 0 and not any(row["state"] is True for row in result["not_gone"])
        self.terminated["confirmed"] = result["confirmed"]
        self.last_stop_ns = perf_counter_ns()
        self.raw.emit("stop_all", mode=self.label, **result)
        return result

    def _read_active(self, job):
        """One Job's active count, or the error that read it: never a clean 0 that was not read."""
        try:
            return self.win.active(job)
        except Exception as exc:
            return repr(exc)

    def escaped(self):
        """``(escapes, gaps)``: live processes outside the Job whose image lies under a scope root (the Job must
        be open), and the processes this scan could not observe. A process this user cannot query (a
        protected system process) is disclosed in ``gaps``, never counted as measured dead or as an escape.
        ``escapes`` is ``None`` when the process table itself could not be walked."""
        try:
            pids = list(self.win.snapshot())
        except Exception as exc:
            return None, {"scan_error": repr(exc)}
        rows, gaps = [], []
        for pid in pids:
            if pid in (0, 4, os.getpid()):
                continue
            try:
                identity = self.win.identity(pid, self.job)
            except Exception as exc:
                gaps.append({"pid": pid, "error": repr(exc)})
                continue
            if identity.get("alive") is None:
                gaps.append({key: identity.get(key) for key in ("pid", "open_error", "read_error")})
            elif identity.get("alive") and identity.get("in_job") is False and any(
                    under(identity.get("image"), root) for root in self.scope_roots):
                rows.append(identity)
        return rows, {"unobserved": len(gaps), "gaps": gaps[:100]}

    def drain(self, end):
        """Join every holder's output pump until ``end`` (monotonic). ``complete`` only when each read its pipe
        to the end without an error, so its transcript holds all the holder wrote; never raises."""
        pending, errors = [], []
        for holder in self.pumps:
            holder.pump.join(max(0.0, end - time.monotonic()))
            if holder.pump.is_alive():
                pending.append(str(holder.transcript))
            elif holder.error is not None:
                errors.append(f"{holder.transcript}: {holder.error}")
        return {"complete": not pending and not errors, "pumps": len(self.pumps), "pending": pending,
                "errors": errors}

    def close(self):
        """Stop everything, scan for escapes while the Job can still answer, re-read every retired Job, ALWAYS
        close every Job, then drain the holders' output within what is left of the same allowance: the mode's
        teardown. Never raises; anything not confirmed is ``unknown`` (a retired Job that does not read empty
        included), and an incomplete ``drain`` leaves the evidence export unconfirmed."""
        if self.closed:
            return {"status": "closed_before"}
        end = time.monotonic() + TEARDOWN_ALLOWANCE
        stopped, escaped, scan, closed, retired = {"confirmed": False}, None, {}, "not attempted", None
        try:
            try:
                stopped = self.stop_all("teardown")
            except Exception as exc:
                stopped = {"confirmed": False, "error": repr(exc)}
            try:
                escaped, scan = self.escaped()
            except Exception as exc:
                escaped, scan = None, {"scan_error": repr(exc)}
            retired = [{"generation": row["generation"], "stop": row["stop"], "active": self._read_active(row["job"])}
                       for row in self.retired]
        finally:
            errors = []
            for job in [self.job, *(row["job"] for row in self.retired)]:
                try:
                    self.win.close(job)
                except Exception as exc:
                    errors.append(repr(exc))
            closed = "ok" if not errors else "; ".join(errors)
            self.closed = True
        try:  # the closed Job killed what it held, so their pipes end; one still open stays pending
            drain = self.drain(end)
        except Exception as exc:
            drain = {"complete": False, "error": repr(exc)}
        retired_empty = retired is not None and all(row["active"] == 0 for row in retired)
        status = "confirmed" if stopped.get("confirmed") is True and closed == "ok" and escaped == [] \
            and retired_empty else "unknown"
        teardown = {"status": status, "stopped": stopped.get("confirmed"), "job_close": closed,
                    "escaped": escaped, "escape_scan": scan, "drain": drain, "started": self.started,
                    "retired": retired}
        try:
            self.raw.emit("teardown", mode=self.label, **teardown)
        except Exception:
            pass  # the returned teardown still carries the verdict
        return teardown


def classify(members):
    """Job members by role, from their command lines (evidence, not authority to signal); ``None`` when there
    is no census, never an empty one."""
    if members is None:
        return None
    roles = {"frozen": [], "launcher": [], "server": [], "other": []}
    for row in members:
        image, line = str(row.get("image") or ""), str(row.get("cmdline") or "")
        if image.lower().endswith("\\ouroboros.exe"):
            roles["frozen"].append(row)
        elif "launcher.py" in line:
            roles["launcher"].append(row)
        elif "server.py" in line:
            roles["server"].append(row)
        else:
            roles["other"].append(row)
    return roles


def identities(rows):
    """``{(pid, birth)}`` of the live rows; ``None`` for no census or one holding a row whose liveness was not
    read (an unobserved member could be any process)."""
    if rows is None or any(row.get("alive") is None for row in rows):
        return None
    return {(row.get("pid"), str(row.get("birth"))) for row in rows if row.get("alive")}


def unread(members):
    """The members whose liveness or, while alive, command line was not read; ``None`` for no census."""
    if members is None:
        return None
    return [row for row in members if row.get("alive") is None or (row.get("alive") and row.get("cmdline") is None)]


def known_roles(members):
    """``classify`` of a census that can answer role questions (every member's liveness and, while alive,
    command line read); otherwise ``None``: an unread member could hold any role, a launcher included."""
    return classify(members) if unread(members) == [] else None


def launcher_ids(members):
    """The live launchers' ``{(pid, birth)}`` in a census; ``None`` unless its roles are known."""
    roles = known_roles(members)
    return None if roles is None else identities(roles["launcher"])
