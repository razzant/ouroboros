#!/usr/bin/env python3
"""Real-engine witness that Quit, and only Quit, frees this installation's engine home and its run's project.

The pinned Claudexor engine (checksum-verified, as ``claudexor_lifecycle_smoke``) runs in its own
temporary HOME/data with one credential-free ``fake-hang`` run active on it. A target checkout's
real launcher drives every door, so the same scenario runs on a candidate and on its base
(``--checkout``): ``agent_lifecycle_loop`` and ``start_agent`` (a Windows kill-on-close Job
included) start generations whose ``server.py`` is a stub that ensures the engine, keeps the
run active, and leaves through the launcher's quit pipe or a planned Restart's exit 42. Only that
server and the launcher names in ``SEAMS`` (the planned Restart's bundle sync, dependency install and
launcher re-exec) are substituted, by counted stand-ins whose calls ``witness.json`` reports; a checkout
missing one of those names is refused before any generation starts, so the real sync never runs. Every
other launcher function is the checkout's own. A foreign installation's engine runs throughout.

In order: the settings recycle (``_request_agent_restart``) and a planned Restart must keep the
engine's exact identity (ledger PID plus measured birth) and the active run; then, before Quit,
the engine home's holders are observed by PID, birth, path and handle (Windows: Sysinternals
``handle.exe`` in CSV and text form, psutil beside it; macOS: ``lsof -F``; Linux: ``/proc``), and
on Windows a rename of that home must be refused. Each source's full output is saved under
``--evidence-dir``. A holder counts only when its PID belongs to the engine identity, its path lies
in the home and it names a handle or descriptor; anything else is ``unknown`` and fails the run.
Then the real headless Quit runs (``_headless_signal_handler``, ``_run_headless_main``) and the
seconds until the home renames (Windows) or the engine identity is gone (POSIX) are measured.
``--expect release`` (a candidate) or ``--expect hold`` (a base whose engine outlives Quit)
decides the verdict. If Windows renames the home before Quit, this scenario did not reproduce the
lock; that says nothing about which process held a reporter's folder. Cleanup runs every step on its own
(each started process by its handle, each root's ledger stop) and reports what measurably remains beside
the primary verdict, which a failed cleanup never replaces; only a clean PASS then removes the isolated root
(``/tmp/cx-*`` on POSIX), a failure keeps it for diagnosis, and ``cleanup.temp_root`` records which.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import queue
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ouroboros.force_remove import force_rmtree  # noqa: E402
from scripts.claudexor_lifecycle_smoke import isolate, require  # noqa: E402
from scripts.claudexor_platform_smoke import SmokeFailure  # noqa: E402

# The generation's server.py: ensure (or attach to) the owned engine; the first one starts one fake run and one
# run whose harness CLI is the stand-in below, with a grandchild, both in the run's project folder (a real child
# process the engine spawned there); then leave on the launcher's quit line or pipe EOF (exit 0) or on a planned
# Restart request (exit 42).
_GENERATION = r'''
import json, os, pathlib, sys, threading, time, traceback, uuid
checkout = os.environ["OUROBOROS_WITNESS_CHECKOUT"]
os.environ["OUROBOROS_REPO_DIR"] = checkout  # this stub stands in for the checkout's own server.py
sys.path.insert(0, checkout)
root = pathlib.Path(os.environ["OUROBOROS_WITNESS_ROOT"])
from scripts.claudexor_lifecycle_smoke import fixture_catalog, owner_context, start, wait_running
from scripts.claudexor_platform_smoke import seed_fixture_repo
from ouroboros.claudexor_daemon import ensure_owned_gateway
from ouroboros.config import DATA_DIR
pid, run_file = os.getpid(), root / "client-run.json"
with fixture_catalog():
    gateway = ensure_owned_gateway()
    if not run_file.exists():
        row = start(owner_context(DATA_DIR, "quit-witness-client"), "fake-hang", 900)
        wait_running(gateway, row, 120)
        project, harness = seed_fixture_repo(), root / "harness" / "harness-run.json"
        try:  # the production delegated shape: no trust file, the run's workspace is the project
            gateway.register_project(str(project))
            gateway.start_run({"prompt": "Quit witness: hold the project folder until stopped.", "mode": "agent",
                               "scope": {"kind": "project", "root": str(project)}, "harnesses": ["opencode"],
                               "primaryHarness": "opencode", "access": "full", "maxSeconds": 900,
                               "execution": {"isolation": "live", "delegated": True, "workspaceRoot": str(project)},
                               "authPreference": "api_key"}, idempotency_key=uuid.uuid4().hex)
            deadline = time.monotonic() + 120
            while not harness.exists() and time.monotonic() < deadline:
                time.sleep(0.2)
        except Exception:
            (root / "harness" / "start-error.txt").write_text(traceback.format_exc())
        run_file.write_text(json.dumps({"run_id": row.run_id, "project": str(project)}))
    run = json.loads(run_file.read_text())
    state = (gateway.get_run(run["run_id"]).get("summary") or {}).get("state")
    gateway.close()
ready = root / "generations" / f"{pid}.json"
ready.with_suffix(".tmp").write_text(json.dumps({"pid": pid, **run, "state": state}))
ready.with_suffix(".tmp").replace(ready)
stop = threading.Event()
def pipe():
    for line in sys.stdin:
        if line.strip() == "quit":
            break
    stop.set()
threading.Thread(target=pipe, daemon=True).start()
restart = root / "control" / f"{pid}.restart"
while not stop.wait(0.1):
    if restart.exists():
        sys.exit(42)
sys.exit(0)
'''

# The harness CLI the engine spawns for ``opencode`` (``CLAUDEXOR_OPENCODE_BIN``). It reaches no provider and reads
# no key; ``OPENCODE_API_KEY`` holds a dummy string only so the adapter counts its auth as configured. A probe
# answers a 1.x version; ``run`` keeps one grandchild in its own cwd (the project), records both, then waits.
# POSIX runs the Python form by its shebang. Windows' engine runs a non-.exe CLI only as a Node script with a
# node shebang (its ``commandForFile``), so it gets the JavaScript form.
_HARNESS_PY = r'''
import json, os, subprocess, sys, time
out, args = __OUT__, sys.argv[1:]
if not args or args[0] != "run":
    print("1.4.0", flush=True)
    sys.exit(0)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"], cwd=os.getcwd(),
                         stdin=subprocess.DEVNULL, start_new_session=True)  # it leaves the harness's session
with open(os.path.join(out, "harness-run.tmp"), "w", encoding="utf-8") as record:
    json.dump({"harness_pid": os.getpid(), "grandchild_pid": child.pid, "cwd": os.getcwd()}, record)
os.replace(os.path.join(out, "harness-run.tmp"), os.path.join(out, "harness-run.json"))
print(json.dumps({"type": "session", "sessionID": "quit-witness"}), flush=True)
print(json.dumps({"type": "text", "text": "holding the project folder"}), flush=True)
while True:
    time.sleep(1)
'''
_HARNESS_JS = r'''#!/usr/bin/env node
const { spawn } = require("child_process"), fs = require("fs"), path = require("path");
const out = __OUT__, args = process.argv.slice(2);
if (args[0] !== "run") { console.log("1.4.0"); process.exit(0); }
const child = spawn(process.execPath, ["-e", "setTimeout(() => {}, 3600e3)"], {cwd: process.cwd(), stdio: "ignore"});
fs.writeFileSync(path.join(out, "harness-run.tmp"),
                 JSON.stringify({harness_pid: process.pid, grandchild_pid: child.pid, cwd: process.cwd()}));
fs.renameSync(path.join(out, "harness-run.tmp"), path.join(out, "harness-run.json"));
console.log(JSON.stringify({type: "session", sessionID: "quit-witness"}));
console.log(JSON.stringify({type: "text", text: "holding the project folder"}));
setInterval(() => {}, 1000);
'''


def write_harness(root, windows):
    """The stand-in CLI and the folder it reports into; returns the engine environment that selects it."""
    out = root / "harness"
    out.mkdir(parents=True, exist_ok=True)
    if windows:
        cli = out / "opencode.js"
        cli.write_text(_HARNESS_JS.replace("__OUT__", json.dumps(str(out))), encoding="utf-8")
    else:
        cli = out / "opencode"
        cli.write_text(f"#!{sys.executable} -I\n" + _HARNESS_PY.replace("__OUT__", repr(str(out))),
                       encoding="utf-8")
        cli.chmod(0o755)
    return {"CLAUDEXOR_OPENCODE_BIN": str(cli), "OPENCODE_API_KEY": "quit-witness-dummy-not-a-credential"}


def process_cwd(pid):
    """A process's working directory, read from the OS ('' when unreadable)."""
    if os.name == "nt":
        try:
            import psutil

            return psutil.Process(pid).cwd()
        except Exception:  # noqa: BLE001 - unreadable is '' and fails the inside-project check
            return ""
    link = Path(f"/proc/{pid}/cwd")
    if link.parent.is_dir():
        try:
            return os.readlink(link)
        except OSError:
            return ""
    done = subprocess.run(["lsof", "-n", "-P", "-a", "-p", str(pid), "-d", "cwd", "-F", "n"], capture_output=True,
                          text=True, timeout=60)
    return next((line[1:] for line in done.stdout.splitlines() if line.startswith("n")), "")


def harness_identity(root, engine, project, home):
    """The engine-spawned harness and its grandchild: PID, measured birth and cwd, from their own report."""
    from ouroboros.platform_layer import process_start_time

    report = root / "harness" / "harness-run.json"
    error = root / "harness" / "start-error.txt"
    require(report.is_file(), "harness_child_not_started",
            start_error=error.read_text(encoding="utf-8") if error.is_file() else "")
    facts = json.loads(report.read_text(encoding="utf-8"))
    tree = []
    for role in ("harness", "grandchild"):
        pid = int(facts[f"{role}_pid"])
        tree.append({"pid": pid, "birth": process_start_time(pid) or None, "root": False, "role": role,
                     "cwd": process_cwd(pid)})
    members = {row["pid"] for row in engine["tree"]}
    require(tree[0]["pid"] in members and all(row["birth"] for row in tree), "harness_child_not_the_engines",
            harness=tree, engine=engine)
    require(all(_inside(row["cwd"], project) and not _inside(row["cwd"], home) for row in tree),
            "harness_child_not_in_project", harness=tree, project=str(project))
    return {"roots": [], "tree": tree}


# Launcher names the planned Restart calls that would sync a bundle, install dependencies or re-exec the
# launcher, each replaced by a counted substitute with this result. A checkout without one of them is refused
# before any generation starts: the real sync must never run (tests bind these names to agent_lifecycle_loop).
SEAMS = {"sync_existing_repo_from_bundle": None, "install_deps": True, "launcher_sources_changed": False}

# The target checkout's launcher process. main() binds the lifecycle thread to Quit the same way.
_LAUNCHER = r'''
import json, pathlib, sys, threading
root, repo, stub = (pathlib.Path(arg) for arg in sys.argv[1:4])
import launcher
seams, calls = SEAMS, {}
missing = sorted(name for name in seams if not callable(getattr(launcher, name, None)))
if missing:
    sys.exit("launcher seams missing (a planned Restart would run the real one): " + ", ".join(missing))
def substitute(name, result):
    def counted(*args, **kwargs):
        calls[name] = calls.get(name, 0) + 1
        return result
    return counted
launcher.REPO_DIR, launcher.EMBEDDED_PYTHON = stub, sys.executable
for name, result in seams.items():
    setattr(launcher, name, substitute(name, result))
lifecycle = threading.Thread(target=launcher.agent_lifecycle_loop, args=(0,), daemon=True)
if hasattr(launcher, "_quit"):
    launcher._quit.lifecycle = lifecycle
lifecycle.start()
for line in sys.stdin:
    if line.strip() == "recycle":
        launcher._request_agent_restart()
    elif line.strip() == "quit":
        break
launcher._headless_signal_handler(15, None)
try:
    launcher._run_headless_main("http://127.0.0.1:1", 1, lifecycle)
finally:
    (root / "seam-calls.json").write_text(json.dumps(calls))
'''.replace("SEAMS", repr(SEAMS))

SUBSTITUTED = {"server.py": "generation stub (scripts/claudexor_quit_witness.py _GENERATION)",
               **{f"launcher.{name}": f"counted substitute returning {result!r}" for name, result in SEAMS.items()},
               "engine catalog": "fake harness manifests only (claudexor_lifecycle_smoke.fixture_catalog)"}


def engine_identity(data):
    """The engine's exact identity: ledger roots whose measured birth/command still match, and their trees."""
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
    from ouroboros.platform_layer import collect_descendant_pids, process_start_time
    from ouroboros.process_custody import process_fingerprint_matches, process_stop_snapshot

    roots, tree = [], []
    for entry in process_stop_snapshot(Path(data), {CUSTODY_PURPOSE}):
        if not process_fingerprint_matches(entry, require_measured=True):
            continue
        fingerprint = entry.get("fingerprint") or {}
        root = {"pid": int(entry["pid"]), "birth": fingerprint.get("start_time_boot") or fingerprint.get("start_time"),
                "fingerprint": fingerprint, "root": True}
        roots.append(root)
        tree.append(root)
        for child in sorted(collect_descendant_pids(root["pid"])):
            tree.append({"pid": int(child), "birth": process_start_time(int(child)) or None, "root": False,
                         "under": root["pid"]})
    return {"roots": roots, "tree": tree}


def same_engine(first, second):
    key = lambda identity: sorted((row["pid"], row["birth"]) for row in identity["roots"])  # noqa: E731
    return bool(first["roots"]) and key(first) == key(second)


def engine_alive(identity):
    """Whether any root of this identity is still that same process: its recorded birth/command, measured."""
    from ouroboros.process_custody import process_fingerprint_matches

    return any(process_fingerprint_matches({"pid": row["pid"], "fingerprint": row["fingerprint"]},
                                           require_measured=True) for row in identity["roots"])


_HANDLE_TEXT = re.compile(r"^(?P<process>.+?)\s+pid:\s*(?P<pid>\d+)\s+type:\s*(?P<type>\S+)\s+"
                          r"(?:.*?\s)?(?P<handle>[0-9A-Fa-f]+):\s+(?P<path>.+?)\s*$")


_HANDLE_NO_MATCH = re.compile(r"^\s*No matching handles found\.?\s*$", re.IGNORECASE | re.MULTILINE)


def parse_handle_text(text):
    """Sysinternals handle's search output, one ``<process> pid: <pid> type: <type> [user] <hex>: <name>`` line
    per handle. The format is not documented by Microsoft; lines that do not match are counted, never guessed."""
    rows, unparsed = [], 0
    for line in text.splitlines():
        if not line.strip() or "pid:" not in line:
            continue
        match = _HANDLE_TEXT.match(line.strip())
        if match is None:
            unparsed += 1
            continue
        rows.append({"process": match["process"].strip(), "pid": int(match["pid"]), "type": match["type"],
                     "handle": match["handle"].upper(), "path": match["path"]})
    return rows, unparsed


def parse_handle_csv(text):
    """``handle -v`` CSV. Columns are found by their header names (not documented by Microsoft); None when
    no header with Process/PID/Handle/Name columns is present."""
    table = [row for row in csv.reader(io.StringIO(text)) if row]
    for index, header in enumerate(table):
        names = [cell.strip().lower() for cell in header]
        if {"pid", "handle"} <= set(names) and ("name" in names or "path" in names):
            column = {name: names.index(name) for name in names}
            path_at = column.get("name", column.get("path"))
            rows = []
            for cells in table[index + 1:]:
                try:
                    rows.append({"process": cells[column["process"]].strip() if "process" in column else "",
                                 "pid": int(cells[column["pid"]]),
                                 "type": cells[column["type"]].strip() if "type" in column else "",
                                 "handle": cells[column["handle"]].strip().upper(), "path": cells[path_at].strip()})
                except (IndexError, ValueError):
                    continue
            return rows
    return None


def parse_lsof_fields(text):
    """``lsof -F pfn`` field output: ``p<pid>`` starts a process, ``f<fd>`` a descriptor, ``n<name>`` its name."""
    rows, pid, fd = [], None, None
    for line in text.splitlines():
        if not line:
            continue
        tag, value = line[0], line[1:]
        if tag == "p":
            pid, fd = int(value), None
        elif tag == "f":
            fd = value
        elif tag == "n" and pid is not None and fd is not None:
            rows.append({"process": "", "pid": pid, "type": "", "handle": fd, "path": value})
    return rows


def _inside(path, home):
    norm = lambda value: os.path.normcase(os.path.normpath(str(value)))  # noqa: E731
    path, home = norm(path), norm(home)
    return path == home or path.startswith(home.rstrip(os.sep) + os.sep)


def match_holders(rows, identity, home, source):
    """Rows naming a handle or descriptor of an engine-identity PID on a path inside the home."""
    members = {row["pid"]: row for row in identity["tree"]}
    matches = []
    for row in rows:
        member = members.get(row["pid"])
        if member is not None and row.get("handle") and _inside(row["path"], home):
            matches.append({"source": source, "pid": row["pid"], "birth": member["birth"], "root": member["root"],
                            "handle": row["handle"], "type": row.get("type", ""), "process": row.get("process", ""),
                            "path": row["path"]})
    return matches


def holder_verdict(sources, windows):
    """``observed`` only from a handle-naming source that matched the engine identity: on Windows that is
    handle.exe (psutil names no handle), on POSIX ``lsof`` or ``/proc``. Otherwise ``unknown`` and why."""
    required = ("handle.exe csv", "handle.exe text") if windows else ("lsof", "proc")
    matches = [match for name in required for match in sources.get(name, {}).get("matches", [])]
    if matches:
        return {"verdict": "observed", "matches": matches}
    ran = [name for name in required if sources.get(name, {}).get("status") == "ran"]
    reason = ("no holder source ran: " + ", ".join(f"{name}={sources.get(name, {}).get('status', 'absent')}"
                                                   for name in required)) if not ran else \
        "no holder row matched the engine identity inside the home"
    return {"verdict": "unknown", "reason": reason, "matches": []}


def _save(out, label, name, command, result):
    """``result`` None: the command never started. A ``TimeoutExpired`` started it: the output it read stays."""
    path = out / f"holders-{label}-{name.replace(' ', '-').replace('.', '_')}.txt"
    streams = [] if result is None else [raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw or ""
                                         for raw in (result.stdout, result.stderr)]
    status = ("(not run)" if result is None else f"timed out after {result.timeout}s: killed, its answer unknown"
              if isinstance(result, subprocess.TimeoutExpired) else f"exit {result.returncode}")
    path.write_text(f"$ {' '.join(map(str, command))}\n{status}\n" + (
        f"--- stdout ---\n{streams[0]}\n--- stderr ---\n{streams[1]}\n" if streams else ""), encoding="utf-8")
    return path.name


def observe_holders(home, identity, out, label, handle_exe, *, windows=None, proc=Path("/proc"), still=None):
    """Run every available source, save its full output, and decide from exact identity matches. ``home`` is the
    observed folder (the engine home, or the project); ``still`` says the identity is still the same processes."""
    windows, sources = (os.name == "nt") if windows is None else windows, {}

    def run(name, command, parse):
        try:
            result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:  # a timeout started the command: never "not run"
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            sources[name] = {"status": "timeout" if timed_out else "failed", "error": f"{type(exc).__name__}: {exc}",
                             "raw": _save(out, label, name, command, exc if timed_out else None)}
            return
        parsed = parse(result.stdout)
        rows, unparsed = parsed if isinstance(parsed, tuple) else (parsed, None)
        # A run that found none: lsof exits 1 with no output when nothing is open under the folder, and handle.exe
        # may exit nonzero saying so in words. Any other nonzero exit without rows stays a failed source.
        empty = name == "lsof" and result.returncode == 1 and not (result.stdout.strip() or result.stderr.strip())
        no_match = (name.startswith("handle.exe") and not rows and not unparsed
                    and bool(_HANDLE_NO_MATCH.search(result.stdout)))
        sources[name] = {"status": "ran" if result.returncode == 0 or rows or empty or no_match else "failed",
                         "exit": result.returncode,
                         "raw": _save(out, label, name, command, result),
                         "format": "no match" if no_match else "unrecognized" if rows is None else "parsed",
                         "rows": len(rows or []),
                         "unparsed_lines": unparsed,
                         "matches": match_holders(rows or [], identity, home, name)}

    if windows:
        if handle_exe:
            run("handle.exe csv", [handle_exe, "-accepteula", "-nobanner", "-v", str(home)], parse_handle_csv)
            run("handle.exe text", [handle_exe, "-accepteula", "-nobanner", str(home)], parse_handle_text)
        else:
            sources["handle.exe csv"] = sources["handle.exe text"] = {"status": "missing"}
        try:
            import psutil
        except ImportError:
            sources["psutil"] = {"status": "missing"}
        else:
            rows = []
            for member in identity["tree"]:
                try:
                    process = psutil.Process(member["pid"])
                    rows += [{"pid": member["pid"], "handle": "", "path": entry.path} for entry in process.open_files()]
                    rows.append({"pid": member["pid"], "handle": "", "path": process.cwd(), "type": "cwd"})
                except (psutil.Error, OSError):
                    continue
            sources["psutil"] = {"status": "ran", "rows": len(rows), "note": "names no handle id",
                                 "paths_inside": sorted({row["path"] for row in rows if _inside(row["path"], home)})}
    elif shutil.which("lsof"):
        run("lsof", ["lsof", "-n", "-P", "-F", "pfn", "+D", str(home)], parse_lsof_fields)
    else:
        sources["lsof"] = {"status": "missing"}
    if not windows and (proc / "self" / "fd").is_dir():
        rows = []
        for member in identity["tree"]:
            fd_dir = proc / str(member["pid"]) / "fd"
            try:
                rows.append({"pid": member["pid"], "handle": "cwd",
                             "path": os.readlink(proc / str(member["pid"]) / "cwd")})
            except OSError:
                pass
            try:
                for fd in fd_dir.iterdir():
                    try:
                        rows.append({"pid": member["pid"], "handle": fd.name, "path": os.readlink(fd)})
                    except OSError:
                        continue
            except OSError:
                continue
        listing = "\n".join(f"{row['pid']} fd {row['handle']} -> {row['path']}" for row in rows)
        (out / f"holders-{label}-proc.txt").write_text(listing + "\n", encoding="utf-8")
        sources["proc"] = {"status": "ran", "raw": f"holders-{label}-proc.txt", "rows": len(rows),
                           "matches": match_holders(rows, identity, home, "proc")}
    verdict = holder_verdict(sources, windows)
    # The identity must still be the same processes after the observation (no PID reuse in between).
    verdict["identity_unchanged"] = still() if still else same_engine(identity, engine_identity(home.parent))
    return {**verdict, "sources": sources}


def _try_rename(home):
    moved = home.with_name(home.name + "-renamed")
    try:
        home.rename(moved)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    moved.rename(home)
    return True, ""


def _read_line(stream, seconds):
    """One line from a child's pipe, or None after ``seconds``: a silent child never blocks the witness."""
    lines = queue.Queue()
    threading.Thread(target=lambda: lines.put(stream.readline()), daemon=True).start()
    try:
        return lines.get(timeout=seconds)
    except queue.Empty:
        return None


def _start_foreign(data, repo, env, out, state):
    """Another installation's owned engine: a different data root, the same machine.

    Its cold runtime install can miss its Node probe on a loaded host; that bystander gets one more
    attempt, counted in the facts, with each attempt's output saved. Every attempt's process is kept in
    ``state`` before it is awaited, so cleanup reaches a half-started one too."""
    code = ("import time\nfrom ouroboros.claudexor_daemon import ensure_owned_gateway\n"
            "ensure_owned_gateway().close()\nprint('ready', flush=True)\ntime.sleep(3600)\n")
    for attempt in (1, 2):
        with (out / f"foreign-engine-attempt{attempt}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=str(repo), stdout=subprocess.PIPE,
                                    stderr=log, text=True,
                                    env={**env, "OUROBOROS_DATA_DIR": str(data), "PYTHONPATH": str(repo)})
        state.setdefault("processes", []).append(("foreign", proc))
        if (_read_line(proc.stdout, 300) or "").strip() == "ready":
            return attempt
        proc.kill()
        proc.wait(timeout=60)
    raise SmokeFailure("foreign_engine_not_ready", "the foreign engine did not start", attempts=2)


def _next_generation(root, seen, driver, seconds=600):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for path in sorted((root / "generations").glob("*.json")):
            facts = json.loads(path.read_text(encoding="utf-8"))
            if facts["pid"] not in seen:
                return facts
        require(driver.poll() is None, "launcher_exited_early", exit_code=driver.returncode)
        time.sleep(0.2)
    raise SmokeFailure("generation_not_ready", "no new generation became ready", seen=sorted(seen))


def tool_identity(path):
    """The holder tool's exact bytes and, on Windows, its file version: the evidence names what ran."""
    if not path:
        return {"path": "", "status": "missing"}
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    version = ""
    if os.name == "nt":
        quoted = str(path).replace("'", "''")
        probe = subprocess.run(["powershell", "-NoProfile", "-Command",
                                f"(Get-Item -LiteralPath '{quoted}').VersionInfo.FileVersion"],
                               capture_output=True, text=True, timeout=60)
        version = probe.stdout.strip() or f"unread (exit {probe.returncode}: {probe.stderr.strip()[:200]})"
    return {"path": str(path), "sha256": digest, "file_version": version}


def witness(repo, expect, quit_seconds, out, handle_exe, facts, state):
    from ouroboros.claudexor_runtime import managed_runtime_root
    from ouroboros.config import DATA_DIR

    windows = os.name == "nt"
    root, data, env = Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"]), Path(DATA_DIR), dict(os.environ)
    home, stub = data / "claudexor", root / "stub-repo"
    state["roots"] = {"owned": data, "foreign": root / "foreign-data"}  # before anything can start on them
    facts.update(checkout=str(repo), expect=expect, windows=windows, substituted=SUBSTITUTED,
                 handle_exe=tool_identity(handle_exe) if windows else None)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/fetch_claudexor_runtime.py"),
                    "--output-dir", str(managed_runtime_root() / "cache")], check=True)
    for directory in (stub, root / "generations", root / "control"):
        directory.mkdir(parents=True, exist_ok=True)
    (stub / "server.py").write_text(_GENERATION, encoding="utf-8")
    harness_env = write_harness(root, windows)
    facts["foreign_attempts"] = _start_foreign(state["roots"]["foreign"], repo, env, out, state)
    foreign_identity = state["identities"]["foreign"] = engine_identity(state["roots"]["foreign"])
    require(foreign_identity["roots"], "foreign_engine_unledgered")
    driver = subprocess.Popen(
        [sys.executable, "-u", "-c", _LAUNCHER, str(root), str(repo), str(stub)], cwd=str(repo),
        env={**env, **harness_env, "PYTHONPATH": str(repo), "OUROBOROS_WITNESS_CHECKOUT": str(repo),
             "OUROBOROS_WITNESS_ROOT": str(root)},
        stdin=subprocess.PIPE, stdout=(out / "launcher-process.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT, text=True)
    state.setdefault("processes", []).append(("launcher", driver))
    first = _next_generation(root, set(), driver)
    identity = state["identities"]["owned"] = engine_identity(data)
    require(identity["roots"] and first["state"] == "running", "engine_or_run_not_up", first=first,
            identity=identity)
    project = Path(first["project"])
    harness = state["identities"]["owned-harness"] = harness_identity(root, identity, project, home)
    facts["harness_child"] = {"project": str(project), "tree": harness["tree"], "env": sorted(harness_env)}
    doors = facts["doors"] = {"first": {"generation": first, "engine": identity}}
    driver.stdin.write("recycle\n")
    driver.stdin.flush()
    for door in ("recycle", "planned_restart"):
        generation = _next_generation(root, {row["generation"]["pid"] for row in doors.values()}, driver)
        after = engine_identity(data)
        doors[door] = {"generation": generation, "engine": after}
        require(same_engine(identity, after) and generation["state"] == "running"
                and generation["run_id"] == first["run_id"], f"{door}_replaced_engine_or_run", doors=doors)
        if door == "recycle":  # the server's own planned Restart: exit 42 through the lifecycle loop
            (root / "control" / f"{generation['pid']}.restart").touch()
    require(len(_alive_rows(harness)) == 2, "harness_child_lost_before_quit", harness=harness)
    held = observe_holders(project, harness, out, "project-before-quit", handle_exe,
                           still=lambda: len(_alive_rows(harness)) == 2)
    project_renamed, project_refusal = _try_rename(project) if windows else (None, "not probed on POSIX")
    facts["project_before_quit"] = {"holders": held, "rename_ok": project_renamed, "rename_error": project_refusal}
    if windows:
        require(not project_renamed, "windows_project_lock_not_reproduced", project=facts["project_before_quit"])
    require(held["verdict"] == "observed" and held["identity_unchanged"], "project_holder_unknown",
            project=facts["project_before_quit"])
    holders = observe_holders(home, identity, out, "before-quit", handle_exe)
    renamed, refusal = _try_rename(home) if windows else (None, "not probed: POSIX renames a held directory")
    facts["before_quit"] = {"holders": holders, "rename_ok": renamed, "rename_error": refusal}
    if windows:
        require(not renamed, "windows_home_lock_not_reproduced", before_quit=facts["before_quit"])
    require(holders["verdict"] == "observed" and holders["identity_unchanged"], "holder_unknown",
            before_quit=facts["before_quit"])
    started, launcher_exit, released, project_free, elapsed = time.monotonic(), None, False, None, None
    driver.stdin.write("quit\n")
    driver.stdin.flush()
    while time.monotonic() - started < quit_seconds:
        if launcher_exit is None and driver.poll() is not None:
            launcher_exit = round(time.monotonic() - started, 2)
        if not released:
            released, refusal = _try_rename(home) if windows else (_gone(identity["roots"]), "")
            elapsed = round(time.monotonic() - started, 2)
        if project_free is None and (_try_rename(project)[0] if windows else _gone(_rows(harness))):
            project_free = round(time.monotonic() - started, 2)
        if released and project_free is not None:
            break
        time.sleep(0.1)
    elapsed = elapsed if released else round(time.monotonic() - started, 2)
    driver.wait(timeout=180)
    if launcher_exit is None:
        launcher_exit = round(time.monotonic() - started, 2)
    seam_calls = root / "seam-calls.json"
    facts["seam_calls"] = json.loads(seam_calls.read_text(encoding="utf-8")) if seam_calls.is_file() else None
    facts["after_quit"] = {"seconds_to_release": elapsed if released else None, "rename_error": refusal,
                           "seconds_to_launcher_exit": launcher_exit, "launcher_exit": driver.returncode,
                           "engine_alive": engine_alive(identity), "seconds_to_project_release": project_free,
                           "harness_alive": [row["role"] for row in _rows(harness) if _liveness(row) != "gone"],
                           "foreign_same_and_alive": engine_alive(foreign_identity)
                           and same_engine(foreign_identity, engine_identity(state["roots"]["foreign"]))}
    # The planned Restart went through the substitutes: no real sync, install or re-exec ran.
    require(all((facts["seam_calls"] or {}).get(name) for name in SEAMS), "planned_restart_bypassed_seams",
            seam_calls=facts["seam_calls"])
    require(facts["after_quit"]["foreign_same_and_alive"], "foreign_engine_stopped", after_quit=facts["after_quit"])
    if expect == "release":
        require(released and not facts["after_quit"]["engine_alive"], "engine_home_still_held_after_quit",
                after_quit=facts["after_quit"])
        facts["after_quit"]["project_holders"] = observe_holders(project, harness, out, "project-after-quit",
                                                                 handle_exe, still=lambda: True)
        observed = facts["after_quit"]["project_holders"]
        require(project_free is not None and not facts["after_quit"]["harness_alive"] and not observed["matches"]
                and not observed.get("reason", "").startswith("no holder source ran"),
                "project_still_held_after_quit", after_quit=facts["after_quit"])
        if (home / "daemon.log").is_file():
            shutil.copy2(home / "daemon.log", out / "engine-daemon.log")
        moved = home.with_name("claudexor-deleted")
        try:
            home.rename(moved)
            shutil.rmtree(moved)
        except OSError as exc:
            raise SmokeFailure("home_not_deleted_after_quit", f"{type(exc).__name__}: {exc}") from exc
        facts["after_quit"]["renamed_and_deleted"] = True
    else:
        facts["after_quit"]["holders"] = observe_holders(home, identity, out, "after-quit", handle_exe)
        facts["after_quit"]["project_holders"] = observe_holders(project, harness, out, "project-after-quit",
                                                                 handle_exe, still=lambda: True)
        require(facts["after_quit"]["engine_alive"] and not (windows and released),
                "base_engine_released_unexpectedly", after_quit=facts["after_quit"])


class StepFailed(Exception):
    """A cleanup step that ran and failed; its raw facts (exit, full output) stay with the failure."""

    def __init__(self, message, facts):
        super().__init__(message)
        self.facts = facts


def _step(steps, name, action):
    """One cleanup step's raw result or its failure, recorded; a failed step never skips the next."""
    try:
        steps[name] = {"ok": True, "result": action()}
    except StepFailed as exc:
        steps[name] = {"ok": False, "error": str(exc), "result": exc.facts}
    except Exception as exc:  # noqa: BLE001 - recorded in witness.json, never swallowed
        steps[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return steps[name]


def _end_process(proc):
    if proc.poll() is None:
        proc.kill()  # this witness's own child, by its Popen handle: never a name or pattern
    return {"pid": proc.pid, "returncode": proc.wait(timeout=30)}


def _ledger_stop(data, env, out, label):
    """The installation's own measured ledger identity stop, by that root's process. Its full stdout/stderr
    (a timeout's partial output too) are kept raw under ``out`` and in the facts; only exit 0 succeeds."""
    code = ("from ouroboros.claudexor_daemon import CUSTODY_PURPOSE\n"
            "from ouroboros.config import DATA_DIR\n"
            "from ouroboros.process_custody import stop_ledgered_processes\n"
            "unconfirmed = []\n"
            "print(stop_ledgered_processes(DATA_DIR, {CUSTODY_PURPOSE}, unconfirmed=unconfirmed))\n"
            "print(unconfirmed)\n")
    error = ""
    try:
        done = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), timeout=120, capture_output=True,
                              env={**env, "OUROBOROS_DATA_DIR": str(data), "PYTHONPATH": str(REPO_ROOT)})
        returncode, streams = done.returncode, {"stdout": done.stdout, "stderr": done.stderr}
    except subprocess.TimeoutExpired as exc:
        returncode, streams, error = None, {"stdout": exc.stdout, "stderr": exc.stderr}, f"TimeoutExpired: {exc}"
    facts = {"exit": returncode}
    for name, raw in streams.items():
        raw = raw.encode() if isinstance(raw, str) else (raw or b"")
        path = out / f"cleanup-ledger-stop-{label}.{name}"
        path.write_bytes(raw)
        facts[name], facts[f"{name}_file"] = raw.decode("utf-8", "replace"), str(path)
    if error or returncode != 0:
        raise StepFailed(error or f"ledger stop exited {returncode}", facts)
    return facts


def _liveness(row):
    """``alive`` (still this process), ``gone``, or ``unknown``: a live PID whose birth or command cannot be
    read or was never measured, so it can be neither signalled nor called gone."""
    from ouroboros.platform_layer import pid_is_alive, process_start_time
    from ouroboros.process_containment import pid_is_zombie

    if row["root"] and engine_alive({"roots": [row]}):
        return "alive"
    if not pid_is_alive(row["pid"]) or pid_is_zombie(row["pid"]):
        return "gone"
    fingerprint = row.get("fingerprint") or {}
    births = {str(value) for value in (row.get("birth"), fingerprint.get("start_time"),
                                       fingerprint.get("start_time_boot")) if value}
    live = process_start_time(row["pid"])
    if not live or not births:
        return "unknown"
    if live in births:
        return "unknown" if row["root"] else "alive"  # a root of this birth whose command no longer proves it
    return "gone"  # the PID now names another process


def _rows(identity):
    return [*identity["roots"], *(row for row in identity["tree"] if not row["root"])]


def _gone(rows):
    """Proven gone: an unreadable or unmeasured process is never counted as released."""
    return all(_liveness(row) == "gone" for row in rows)


def _alive_rows(identity):
    """This identity's processes still alive as themselves: roots by measured fingerprint, others by birth."""
    return [row for row in _rows(identity) if _liveness(row) == "alive"]


def _end_measured(identity):
    from ouroboros.platform_layer import force_kill_pid

    rows = _alive_rows(identity)
    for row in rows:
        force_kill_pid(row["pid"])
    return [{"pid": row["pid"], "birth": row["birth"], "kill_requested": True} for row in rows]


def _remaining(data, captured):
    """This root's engine processes not proven gone: its ledger now (a live row without measured identity is
    ``unknown``) and every identity captured during the run. An unreadable ledger raises: never clean."""
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
    from ouroboros.process_custody import process_fingerprint_matches, process_stop_snapshot

    rows = [(int(entry["pid"]), "unknown: ledger row without measured identity")
            for entry in process_stop_snapshot(Path(data), {CUSTODY_PURPOSE})
            if not process_fingerprint_matches(entry, require_measured=True)]
    rows += [(row["pid"], str(row["birth"] or "unknown: birth unread")) for row in engine_identity(data)["tree"]]
    for identity in captured:
        for row in _rows(identity):
            state = _liveness(row)
            if state != "gone":
                rows.append((row["pid"], str(row["birth"]) if state == "alive" else f"unknown: {row['birth']}"))
    return sorted(set(rows))


def cleanup(state, out, env):
    """Every step independent: the launcher, logs, each root's ledger stop, each started process, then what
    measurably remains. Returns raw outcomes; ``clean`` only when every step succeeded and nothing remains."""
    steps = {}
    for label, proc in state.get("processes", []):
        _step(steps, f"process:{label}:{proc.pid}", lambda proc=proc: _end_process(proc))
    data, foreign = state.get("roots", {}).get("owned"), state.get("roots", {}).get("foreign")
    for name, source in (("launcher.log", data and data / "logs" / "launcher.log"),
                         ("agent_stdout.log", data and data / "logs" / "agent_stdout.log"),
                         ("engine-daemon.log", data and data / "claudexor" / "daemon.log"),
                         ("foreign-daemon.log", foreign and foreign / "claudexor" / "daemon.log")):
        if source and source.is_file() and not (out / name).exists():
            _step(steps, f"log:{name}", lambda source=source, name=name: str(shutil.copy2(source, out / name)))
    remaining = {}
    for label, root in state.get("roots", {}).items():
        _step(steps, f"ledger_stop:{label}", lambda root=root, label=label: _ledger_stop(root, env, out, label))
        for key, identity in state.get("identities", {}).items():
            if key.startswith(f"{label}-"):  # the engine's own children it spawned: by measured PID and birth
                _step(steps, f"end_measured:{key}", lambda identity=identity: _end_measured(identity))
        captured = [identity for key, identity in state.get("identities", {}).items() if key.startswith(label)]
        check = _step(steps, f"remaining:{label}", lambda root=root, captured=captured: _remaining(root, captured))
        remaining[label] = check["result"] if check["ok"] else "unknown"
    clean = all(step["ok"] for step in steps.values()) and all(rows == [] for rows in remaining.values())
    return {"clean": clean, "remaining": remaining, "steps": steps}


def release_root(root, out, passed):
    """The isolated root (``/tmp/cx-*`` on POSIX) goes only after a clean PASS; a failure keeps it for diagnosis,
    and so does an evidence dir inside it. A removal that fails is recorded and makes the cleanup unclean."""
    if root is None:
        return {"path": None, "removed": False, "kept": "no isolated root"}
    root, facts = Path(root), {"path": str(root), "removed": False}
    if not passed:
        return {**facts, "kept": "the witness failed or its cleanup was unclean: kept for diagnosis"}
    if out == root or root in out.parents:
        return {**facts, "kept": "the evidence dir lies inside it"}
    force_rmtree(root)  # its fixture repositories' Git objects are read-only, which Windows refuses to delete
    return {**facts, "error": "still present"} if root.exists() else {**facts, "removed": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=REPO_ROOT, help="the checkout whose launcher is driven")
    parser.add_argument("--expect", choices=("release", "hold"), default="release")
    parser.add_argument("--quit-seconds", type=float, default=90.0)
    parser.add_argument("--evidence-dir", type=Path, required=True, help="full outputs and witness.json land here")
    parser.add_argument("--handle-exe", default="", help="Sysinternals handle.exe (Windows holder source)")
    args = parser.parse_args(argv)
    out = args.evidence_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    handle_exe = str(Path(args.handle_exe).resolve()) if args.handle_exe and Path(args.handle_exe).is_file() else ""
    root = isolate()
    facts, state = {}, {"identities": {}}
    try:
        witness(args.checkout.resolve(), args.expect, args.quit_seconds, out, handle_exe, facts, state)
        result = {"verdict": "PASSED", **facts}
    except SmokeFailure as exc:
        result = {"verdict": "FAILED", "reason": exc.code, "failure": exc.facts, **facts}
    except Exception:  # noqa: BLE001 - the primary failure is the verdict, with its traceback
        result = {"verdict": "FAILED", "reason": "witness_error", "error": traceback.format_exc(), **facts}
    try:
        result["cleanup"] = cleanup(state, out, dict(os.environ))
    except Exception:  # noqa: BLE001 - the primary verdict stays; an unfinished cleanup is never clean
        result["cleanup"] = {"clean": False, "error": traceback.format_exc()}
    passed = result["verdict"] == "PASSED" and result["cleanup"]["clean"]
    released = result["cleanup"]["temp_root"] = release_root(root, out, passed)
    result["cleanup"]["clean"] &= "error" not in released
    (out / "witness.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "reason": result.get("reason", ""),
                      "cleanup_clean": result["cleanup"]["clean"],
                      "temp_root": result["cleanup"]["temp_root"]}), flush=True)
    return 0 if result["verdict"] == "PASSED" and result["cleanup"]["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
