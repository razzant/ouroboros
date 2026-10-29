"""Finite Windows/NTFS holder experiment. No architecture verdict or recursive cleanup."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

from winapi import MOUNT_POINT, Win

SOURCE = Path(__file__).absolute().parent
OPERATIONS = ("rename", "junction-replace", "junction-repoint")
HOLDERS = {"user": ("none", "pyd", "image", "cwd"), "venv": ("none", "redirector")}


class TeardownUnknown(RuntimeError):
    pass


def cases():
    return [(layout, op, holder) for layout, holders in HOLDERS.items()
            for op in OPERATIONS for holder in holders]


def attempt(fn, *args):
    try:
        return {"status": "ok", "value": fn(*args)}
    except Exception as exc:
        return failure(exc)


def failure(exc):
    code = getattr(exc, "winerror", None)
    return {"status": "refused" if code in (5, 32, 33, 50, 145, 1314) else "error",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "winerror": code, "errno": getattr(exc, "errno", None)}}


def confirmed_empty(result):
    return (isinstance(result, dict) and result.get("status") == "ok"
            and type(result.get("value")) is int and result["value"] == 0)


def parse_observation(rc, text):
    if rc is None:
        return {"status": "unknown", "reason": "no confirmed process exit"}
    if rc != 0:
        return {"status": "error", "reason": "nonzero exit", "returncode": rc}
    parsed = attempt(json.loads, text)
    if parsed["status"] != "ok":
        return parsed
    value = parsed["value"]
    if not isinstance(value, dict) or value.get("schema") != "ntfs-observation/1":
        return {"status": "error", "reason": "missing observation schema"}
    # Validate delivery and failed queries, never choose an activation architecture.
    required = {"cached_package", "selected_marker", "new_import", "executable", "pyd", "phase", "pid"}
    if not required.issubset(value):
        return {"status": "error", "reason": "incomplete observation", "value": value}
    failed = [key for key in ("selected_marker", "new_import", "pyd")
              if not isinstance(value[key], dict) or value[key].get("status") != "ok"]
    if failed:
        return {"status": "error", "reason": "observation query failed", "fields": failed, "value": value}
    return parsed


class Raw:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8")

    def emit(self, event, **facts):
        value = {"event": event, "time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(), **facts}
        self.file.write(json.dumps(value, default=str) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())


class Owned:
    """A gated subprocess tree. Descendants start only after assignment to our job."""
    def __init__(self, win, raw, folder, label, argv, env, interactive=False):
        self.win, self.raw, self.label = win, raw, label
        self.job, self.proc, self.assigned = win.job(), None, False
        self.lines, self.serial = queue.Queue(), 0
        self.stdout_path = folder / (label + ".stdout")
        self.out = self.stdout_path.open("xb")
        self.err = (folder / (label + ".stderr")).open("xb")
        self.pump = None
        raw.emit("spawn_intent", label=label, argv=argv)
        try:
            self.proc = subprocess.Popen([sys.executable, "-B", str(SOURCE / "probe.py"), "gate", *argv],
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE if interactive else self.out,
                                         stderr=self.err, cwd=folder, env=env, creationflags=0x08000000)
            win.assign(self.job, self.proc)
            self.assigned = True
            raw.emit("job_assigned", label=label, gate_pid=self.proc.pid)
            if interactive:
                self.pump = threading.Thread(target=self._read, daemon=True)
                self.pump.start()
            self.proc.stdin.write(b"go\n")
            self.proc.stdin.flush()
        except BaseException:
            self.finish()
            raise

    def _read(self):
        try:
            for line in iter(self.proc.stdout.readline, b""):
                self.out.write(line)
                self.out.flush()
                os.fsync(self.out.fileno())
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def reply(self):
        line = self.lines.get(timeout=30)
        if line is None:
            raise RuntimeError("holder stdout closed")
        return json.loads(line.decode("utf-8"))

    def request(self, phase):
        self.serial += 1
        self.proc.stdin.write((json.dumps({"id": self.serial, "phase": phase}) + "\n").encode())
        self.proc.stdin.flush()
        reply = self.reply()
        if reply.get("id") != self.serial or reply.get("event") != ("exit" if phase == "exit" else "observe"):
            raise RuntimeError("holder response mismatch")
        return reply

    def finish(self):
        forced = False
        if self.proc is not None:
            self.raw.emit("gate_wait", label=self.label, result=attempt(self.proc.wait, 5))
        count = attempt(self.win.active, self.job)
        self.raw.emit("job_count", label=self.label, phase="before_stop", result=count)
        if not confirmed_empty(count):
            forced = True
            self.raw.emit("job_terminate", label=self.label, result=attempt(self.win.terminate, self.job))
        # A gate not assigned to the job has never been allowed to spawn children.
        if self.proc is not None and not self.assigned and self.proc.poll() is None:
            self.raw.emit("unassigned_gate_stop", label=self.label, result=attempt(self.proc.kill))
        if self.proc is not None:
            self.raw.emit("gate_reap", label=self.label, result=attempt(self.proc.wait, 5))
        deadline = time.monotonic() + 5
        while True:
            count = attempt(self.win.active, self.job)
            if confirmed_empty(count) or count["status"] != "ok" or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        rc = self.proc.poll() if self.proc is not None else None
        confirmed = confirmed_empty(count) and (self.proc is None or rc is not None)
        self.raw.emit("teardown", label=self.label, status="confirmed" if confirmed else "unknown",
                      count=count, gate_returncode=rc, forced=forced, disposition="retain_all_files")
        self.raw.emit("job_close", label=self.label, result=attempt(self.win.close, self.job))
        if self.pump:
            self.pump.join(timeout=2)
        if not self.pump or not self.pump.is_alive():
            self.out.close()
            if self.proc is not None and self.proc.stdout:
                self.proc.stdout.close()
        self.err.close()
        if self.proc is not None:
            self.proc.stdin.close()
        if not confirmed:
            raise TeardownUnknown("unknown owned-tree teardown; stop matrix, retain evidence")
        return rc


def run(win, raw, folder, label, argv, env):
    owned = Owned(win, raw, folder, label, argv, env)
    wait = attempt(owned.proc.wait, 120)
    raw.emit("command_wait", label=label, result=wait)
    rc = owned.finish()
    raw.emit("command_exit", label=label, returncode=rc)
    return rc, owned.stdout_path.read_text(encoding="utf-8", errors="replace")


def environment(folder, userbase=None):
    # No inherited provider credentials, PYTHONPATH, pip configuration, HOME or data paths.
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "COMSPEC") if key in os.environ}
    profile_dir, temp = folder / "home", folder / "temp"
    profile_dir.mkdir(exist_ok=True)
    temp.mkdir(exist_ok=True)
    env.update(HOME=str(profile_dir), USERPROFILE=str(profile_dir), APPDATA=str(profile_dir), LOCALAPPDATA=str(profile_dir),
               TEMP=str(temp), TMP=str(temp), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               NTFS_PROBE_CHILD=str(SOURCE / "child.py"),
               PATH=str(Path(sys.executable).parent) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32"))
    if userbase is not None:
        env["PYTHONUSERBASE"] = str(userbase)
    else:
        env["PYTHONNOUSERSITE"] = "1"
    return env


def wheel(path, generation):
    import zipfile
    version = "1.0.0" if generation == "A" else "2.0.0"
    info = "holder_fixture-" + version + ".dist-info"
    marker = 'GENERATION = "' + generation + '"\n'
    files = {"holder_fixture/" + name + ".py": marker for name in
             ("__init__", "before", "forward", "reverse", "exited_forward", "exited_reverse")}
    files.update({"holder_fixture/generation.txt": generation,
                  "holder_fixture/entry.py": 'import os, runpy, sys\ndef main():\n'
                      '    script = os.environ["NTFS_PROBE_CHILD"]\n'
                      '    sys.path.insert(0, os.path.dirname(script))\n'
                      '    runpy.run_path(script, run_name="__main__")\n',
                  info + "/METADATA": "Metadata-Version: 2.1\nName: holder-fixture\nVersion: " + version + "\n",
                  info + "/WHEEL": "Wheel-Version: 1.0\nGenerator: ntfs-holder-probe\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                  info + "/entry_points.txt": "[console_scripts]\nntfs-observe = holder_fixture.entry:main\nntfs-hold = holder_fixture.entry:main\n"})
    encoded = {name: content.encode() for name, content in files.items()}
    records = [f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')},{len(data)}"
               for name, data in encoded.items()]
    encoded[info + "/RECORD"] = ("\n".join(records + [info + "/RECORD,,"]) + "\n").encode()
    with zipfile.ZipFile(path, "x") as archive:
        for name, data in encoded.items():
            archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data)


def site_path(layout, root):
    return root / ("Python310/site-packages" if layout == "user" else "Lib/site-packages")


def scripts(layout, root):
    return root / ("Python310/Scripts" if layout == "user" else "Scripts")


def python(layout, root):
    return sys.executable if layout == "user" else str(root / "Scripts/python.exe")


def path_facts(win, path):
    def attributes():
        s = path.lstat()
        return {"mode": s.st_mode, "file_id": s.st_ino, "device": s.st_dev,
                "attributes": s.st_file_attributes, "reparse_tag": s.st_reparse_tag}
    observed_stat = attempt(attributes)
    is_junction = observed_stat["status"] == "ok" and observed_stat["value"]["reparse_tag"] == MOUNT_POINT
    return {"path": str(path), "lexists": os.path.lexists(path), "stat": observed_stat,
            "reparse_buffer_hex": attempt(win.reparse, path) if is_junction else None,
            "resolve": attempt(lambda: str(path.resolve(strict=True))),
            "final_path": attempt(win.final_path, path),
            "marker": attempt((path / "generation.txt").read_text)}


def select(win, raw, op, fixed, homes, locations, desired, phase):
    """Record each syscall. Recover a partial forward switch using known locations."""
    def step(name, fn, *args):
        raw.emit("operation_intent", phase=phase, action=name, args=args)
        result = attempt(fn, *args)
        raw.emit("operation_result", phase=phase, action=name, result=result)
        return result["status"] == "ok"
    raw.emit("selection_intent", phase=phase, desired=desired, operation=op)
    if op == "rename":
        selected = next((g for g, p in locations.items() if p == fixed), None)
        if selected == desired:
            raw.emit("operation_noop", phase=phase, reason="already at requested generation", generation=desired)
            return
        if selected:
            if not step("rename-away", os.rename, fixed, homes[selected]):
                return
            locations[selected] = homes[selected]
        if step("rename-in", os.rename, locations[desired], fixed):
            locations[desired] = fixed
    else:
        # No resolve() here: it would mutate/delete the target instead of the junction.
        if op == "junction-replace":
            if os.path.lexists(fixed):
                if not step("require-junction", require_junction, fixed):
                    return
                if not step("rmdir-junction", os.rmdir, fixed):
                    return
            step("create-junction", win.junction, fixed, homes[desired])
        else:
            if step("require-junction", require_junction, fixed):
                step("set-reparse-point", win.reparse, fixed, homes[desired])


def require_junction(path):
    if path.lstat().st_reparse_tag != MOUNT_POINT:
        raise ValueError("expected junction at lexical path; leave it untouched")


def holder_proven(kind, ready, images, expected_image, package):
    if not isinstance(ready, dict) or ready.get("event") != "ready" or ready.get("kind") != kind:
        return False
    obs = ready.get("observation") or {}
    if not isinstance(obs, dict):
        return False
    cached = obs.get("cached_package")
    if not isinstance(cached, dict) or cached.get("generation") != "A":
        return False
    if kind == "pyd":
        pyd, expected = obs.get("pyd") or {}, obs.get("expected_pyd_device") or {}
        return (pyd.get("status") == expected.get("status") == "ok"
                and isinstance(pyd.get("value"), dict)
                and pyd["value"].get("roundtrip") is True
                and isinstance(expected.get("value"), str) and bool(expected["value"])
                and pyd["value"].get("mapped_device", "").casefold() == expected["value"].casefold())
    if kind == "cwd":
        cwd = obs.get("cwd_final", {})
        return (cwd.get("status") == "ok" and same_path(cwd["value"], package)
                and obs.get("relative_marker") == {"status": "ok", "value": "A"})
    return (images.get("status") == "ok" and isinstance(images.get("value"), list)
            and any(isinstance(row, dict) and isinstance(row.get("image"), str)
                    and same_path(row["image"], expected_image) for row in images["value"]))


def same_path(a, b):
    def normal(path):
        text = str(path)
        # GetFinalPathNameByHandleW returns an extended DOS prefix; realpath
        # preserves an input prefix on Python 3.10, so strip it on both sides.
        if text.startswith("\\\\?\\"):
            text = text[4:]
        return os.path.normcase(os.path.realpath(text))
    return normal(a) == normal(b)


def prepare(win, raw, root, layout, op):
    import shutil
    fixed, a, b = root / "active", root / "A", root / "B"
    env = environment(root, fixed if layout == "user" else None)
    wa, wb = root / "holder_fixture-1.0.0-py3-none-any.whl", root / "holder_fixture-2.0.0-py3-none-any.whl"
    wheel(wa, "A")
    wheel(wb, "B")

    def must(label, argv, environ):
        rc, _ = run(win, raw, root, label, argv, environ)
        if rc != 0:
            raise RuntimeError(label + " failed; see raw stdout/stderr")

    if layout == "venv":
        must("venv", [sys.executable, "-B", "-m", "venv", "--without-pip", str(fixed)], env)

    def install(label, dest, whl):
        environ = environment(root, dest if layout == "user" else None)
        argv = [sys.executable, "-B", "-m", "pip"]
        if layout == "venv":
            argv += ["--python", python(layout, dest)]
        argv += ["install", "--no-index", "--no-deps", "--no-cache-dir", "--no-compile",
                 "--disable-pip-version-check", "--no-input", "--upgrade", "--force-reinstall"]
        if layout == "user":
            argv += ["--user"]
        must(label, argv + [str(whl)], environ)
        package = site_path(layout, dest) / "holder_fixture"
        shutil.copyfile(Path(sys.base_prefix) / "DLLs/_bz2.pyd", package / "_bz2.pyd")
        (dest / "generation.txt").write_text("A" if whl == wa else "B", encoding="utf-8")

    install("install-A", fixed, wa)
    # Copy the real, present directory before introducing any junction or holder.
    shutil.copytree(fixed, b)
    install("install-B", b, wb)
    if op != "rename":
        os.rename(fixed, a)
        win.junction(fixed, a)
    raw.emit("prepared", layout=layout, operation=op, fixed=fixed, generation_a=a, generation_b=b,
             preparation="A installed at active; copied to B; B wheel installed at B; no launcher rewriting")
    for name, path in {"A-wheel": wa, "B-wheel": wb,
                       "base-python": Path(sys.executable), "base-pyd": Path(sys.base_prefix) / "DLLs/_bz2.pyd",
                       "A-entry": scripts(layout, fixed) / "ntfs-observe.exe",
                       "B-entry": scripts(layout, b) / "ntfs-observe.exe"}.items():
        raw.emit("input_hash", name=name, path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return fixed, {"A": a, "B": b}, {"A": fixed, "B": b}, env


def observe(win, raw, root, layout, fixed, homes, env, phase, holder):
    raw.emit("paths", phase=phase, paths=[path_facts(win, p) for p in (fixed, *homes.values())])
    if holder:
        raw.emit("holder_alive", phase=phase, gate_returncode=holder.proc.poll(),
                 job_count=attempt(win.active, holder.job), images=attempt(win.images, holder.job))
        raw.emit("old_process", phase=phase, result=attempt(holder.request, phase))
    for label, command in (("fresh", [python(layout, fixed), "-B", str(SOURCE / "child.py")]),
                           ("entrypoint", [str(scripts(layout, fixed) / "ntfs-observe.exe")])):
        rc, text = run(win, raw, root, phase + "-" + label,
                       command + ["observe", str(fixed), "--phase", phase], env)
        raw.emit(label, phase=phase, returncode=rc, observation=parse_observation(rc, text))
    if holder:
        raw.emit("holder_after_observers", phase=phase, gate_returncode=holder.proc.poll(),
                 job_count=attempt(win.active, holder.job))


def case(win, raw, root, layout, op, kind):
    root.mkdir()
    raw.emit("case_start", root=root, layout=layout, operation=op, holder=kind)
    fixed, homes, locations, env = prepare(win, raw, root, layout, op)
    holder = None
    try:
        if kind != "none":
            command = ([str(scripts(layout, fixed) / "ntfs-hold.exe")] if kind == "image"
                       else [python(layout, fixed), "-B", str(SOURCE / "child.py")])
            holder = Owned(win, raw, root, "holder", command + ["hold", str(fixed), "--kind", kind], env, True)
            ready, images = attempt(holder.reply), attempt(win.images, holder.job)
            expected = scripts(layout, fixed) / ("ntfs-hold.exe" if kind == "image" else "python.exe")
            proven = ready["status"] == "ok" and holder_proven(kind, ready["value"], images,
                                                              expected, site_path(layout, fixed) / "holder_fixture")
            raw.emit("holder_setup", reply=ready, images=images, established=proven)
            if not proven:
                raise RuntimeError("holder not established; no switch measurement for this case")
        observe(win, raw, root, layout, fixed, homes, env, "before", holder)
        for phase, desired in (("forward", "B"), ("reverse", "A")):
            if holder:
                # A reply immediately before each operation bounds holder liveness.
                alive = attempt(holder.request, "before")
                raw.emit("pre_operation_holder", phase=phase, reply=alive, gate_returncode=holder.proc.poll(),
                         images=attempt(win.images, holder.job))
                if alive["status"] != "ok" or holder.proc.poll() is not None:
                    raise RuntimeError("holder liveness lost; phase unmeasured")
            select(win, raw, op, fixed, homes, locations, desired, phase)
            observe(win, raw, root, layout, fixed, homes, env, phase, holder)
    finally:
        if holder:
            raw.emit("holder_exit_request", result=attempt(holder.request, "exit"))
            holder.finish()  # unknown exit raises; never run exit comparison then
    # Same tree after confirmed exit, including any partial forward/reverse state.
    for phase, desired in (("exited_forward", "B"), ("exited_reverse", "A")):
        select(win, raw, op, fixed, homes, locations, desired, phase)
        observe(win, raw, root, layout, fixed, homes, env, phase, None)
    raw.emit("case_end", root=root, disposition="retained", architecture_verdict=None)


def junction_semantics(win, raw, root):
    import shutil
    root.mkdir()
    # Independent specimens: no test ever uses the source another test removed.
    for operation in ("resolve", "rmtree-link", "rmtree-containing-tree", "copytree"):
        specimen = root / operation
        specimen.mkdir()
        target, tree = specimen / "target", specimen / "tree"
        target.mkdir()
        tree.mkdir()
        sentinel = target / "sentinel.txt"
        sentinel.write_text("must survive", encoding="utf-8")
        link = tree / "link"
        win.junction(link, target)
        require_junction(link)
        raw.emit("semantics_before", operation=operation, link=path_facts(win, link),
                 target=path_facts(win, target), sentinel=sentinel.read_text())
        functions = {"resolve": lambda: str(link.resolve(strict=True)),
                     "rmtree-link": lambda: shutil.rmtree(link),
                     "rmtree-containing-tree": lambda: shutil.rmtree(tree),
                     "copytree": lambda: str(shutil.copytree(link, specimen / "copy", symlinks=True))}
        result = attempt(functions[operation])
        survived = attempt(sentinel.read_text)
        raw.emit("semantics_after", operation=operation, result=result, link=path_facts(win, link),
                 sentinel=survived, copy_sentinel=attempt((specimen / "copy/sentinel.txt").read_text))
        if survived != {"status": "ok", "value": "must survive"}:
            raise RuntimeError("junction target changed; retain specimen and stop")


def gate(argv):
    # Byte reads avoid buffering any later holder command away from the child.
    token = b""
    while not token.endswith(b"\n") and len(token) < 4:
        part = os.read(0, 1)
        if not part:
            return 124
        token += part
    if token != b"go\n":
        return 124
    return subprocess.call(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "gate":
        if os.name != "nt":
            raise SystemExit("gate is Windows-only")
        return gate(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--root", type=Path, help="new, nonexistent absolute disposable directory")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(cases(), indent=2))
        return 0
    if os.name != "nt" or sys.version_info[:3] != (3, 10, 11) or sys.maxsize <= 2**32:
        raise SystemExit("native measurement requires Windows x64 Python 3.10.11; nothing measured")
    if not args.root or not args.root.is_absolute():
        parser.error("--root must be a new absolute path")
    args.root.mkdir()  # fail on existing paths; never adopt or clean owner data
    raw = Raw(args.root / "events.jsonl")
    win = Win()
    try:
        fs = win.filesystem(args.root)
        raw.emit("identity", python=sys.version, executable=sys.executable, windows=list(sys.getwindowsversion()),
                 filesystem=fs, cases=cases(), source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (SOURCE / "probe.py", SOURCE / "child.py", SOURCE / "winapi.py")})
        if fs["filesystem"] != "NTFS":
            raise RuntimeError("NTFS required; no substitute filesystem")
        junction_semantics(win, raw, args.root / "junction-semantics")
        gaps = []
        for layout, op, kind in cases():
            name = layout + "-" + op + "-" + kind
            try:
                case(win, raw, args.root / name, layout, op, kind)
            except TeardownUnknown:
                raise
            except Exception as exc:
                raw.emit("case_error", case=name, result=failure(exc), traceback=traceback.format_exc())
                gaps.append(name)
        raw.emit("run_end", status="matrix_incomplete" if gaps else "matrix_recorded",
                 incomplete_cases=gaps, native_decision=None, disposition="retained")
        return 2 if gaps else 0
    except Exception as exc:
        raw.emit("run_error", result=failure(exc), traceback=traceback.format_exc(), disposition="retain_all_files")
        return 2
    finally:
        raw.file.close()


if __name__ == "__main__":
    raise SystemExit(main())
