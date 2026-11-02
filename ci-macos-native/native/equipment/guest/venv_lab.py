"""Runtime venv preparer for the native qualification (test equipment, DISCLOSED FIXTURE SEAM; not product code).

Why: the official 7.7.0 app cannot create a ``--runtime-venv``. Its frozen launcher's ``prepare_runtime_venv``
runs ``sys.executable -m venv``, and in the frozen app that is the app binary itself (7.7.1 passes
``base_python``). A 7.7.0 owner therefore opens the app on a venv some launcher already prepared. That earlier
preparation (create the venv at its final path with its marker, install the dependency lock, record it ready)
runs here through the candidate's own functions. It runs under the case's base Python, before the app is first
opened with ``--runtime-venv``:

    ouroboros.runtime_venv.prepare_runtime_venv(VENV, run=<recorded subprocess.run>, log=..., base_python=BASE)
    ouroboros.launcher_bootstrap.install_deps(<context>)  # [<venv>/bin/python, -m, pip, install, -r, <lock>]
    ouroboros.runtime_venv.mark_ready(venv)

The context carries what ``install_deps`` reads: ``repo_dir`` (the candidate checkout CANDSRC, whose lock
equals the 7.7.0 seed's), ``data_dir`` (a scratch root: ``embedded_python_env`` creates
``state/{pycache,python-userbase}`` there, never in the app's data root), the venv's interpreter, the
recorded runner and the launcher's preparation timeout. pip reads PIP_NO_INDEX and PIP_FIND_LINKS from this
process's environment: the offline wheel folder (``wheelhouse.py``) stands in for the package index.

  prepare VENV BASE CANDSRC APP DATA OUT_DIR   run by BASE itself (`BASE -I -B venv_lab.py prepare ...`)
  facts VENV APP CANDSRC OUT_JSON              what the venv names and runs (product readers + the venv's own Python)
"""
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import types


def recorder(out_dir):
    counter = {"n": 0}

    def run(argv, **kwargs):
        counter["n"] += 1
        n, started = counter["n"], time.time()
        row = {"n": n, "argv": [str(a) for a in argv], "cwd": str(kwargs.get("cwd") or os.getcwd()),
               "timeout": kwargs.get("timeout"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        env = kwargs.get("env")
        if env is not None:
            row["env_keys"] = sorted(env)
            row["env_plain"] = {k: env[k] for k in sorted(env) if k.startswith(("PIP_", "PYTHON")) or k in (
                "HOME", "TMPDIR", "PATH", "VIRTUAL_ENV")}
        try:
            result = subprocess.run(argv, **kwargs)
        except BaseException as exc:
            row["raised"] = "%s: %s" % (type(exc).__name__, exc)
            with open(os.path.join(out_dir, "runs.jsonl"), "a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            raise
        row["returncode"], row["seconds"] = result.returncode, round(time.time() - started, 2)
        for stream in ("stdout", "stderr"):
            data = getattr(result, stream, None)
            if data is None:
                continue
            blob = data if isinstance(data, bytes) else str(data).encode("utf-8", "replace")
            name = "run-%d.%s" % (n, stream)
            with open(os.path.join(out_dir, name), "xb") as handle:
                handle.write(blob)
            row[stream] = {"file": name, "bytes": len(blob)}
        with open(os.path.join(out_dir, "runs.jsonl"), "a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return result

    return run


def prepare(venv_path, base, candsrc, app, data, out_dir):
    os.makedirs(out_dir, exist_ok=False)
    if os.path.realpath(sys.executable) != os.path.realpath(base):
        raise SystemExit("refusing: run this with the base Python %s itself, not %s" % (base, sys.executable))
    sys.path.insert(0, candsrc)
    log = logging.getLogger("venv_lab")
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(out_dir, "prepare.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.addHandler(logging.StreamHandler(sys.stdout))
    from ouroboros import launcher_bootstrap, runtime_venv

    facts = {"executable": sys.executable, "version": sys.version, "prefix": sys.prefix,
             "base_prefix": sys.base_prefix, "base": base, "candsrc": candsrc,
             "runtime_venv_module": runtime_venv.__file__, "launcher_bootstrap_module": launcher_bootstrap.__file__,
             "environ_plain": {k: v for k, v in sorted(os.environ.items()) if k.startswith("PIP_") or k in (
                 "HOME", "TMPDIR", "PATH")}, "started": time.time()}
    run = recorder(out_dir)
    outcome = {}
    try:
        venv = runtime_venv.prepare_runtime_venv(venv_path, run=run, log=log, base_python=base)
        outcome["prepared"] = {"root": str(venv.root), "python": venv.python, "ready": venv.ready}
        context = types.SimpleNamespace(repo_dir=pathlib.Path(candsrc), data_dir=pathlib.Path(data),
                                        embedded_python=venv.python, hidden_run=run, log=log,
                                        deps_timeout_sec=runtime_venv.PREPARATION_DEPS_TIMEOUT_SEC)
        outcome["install_deps"] = bool(launcher_bootstrap.install_deps(context))
        if outcome["install_deps"]:
            venv = runtime_venv.mark_ready(venv)
        outcome["final"] = {"root": str(venv.root), "python": venv.python, "ready": venv.ready}
    except runtime_venv.RuntimeVenvRefused as exc:
        outcome["refused"] = str(exc)
    facts["seconds"] = round(time.time() - facts.pop("started"), 1)
    with open(os.path.join(out_dir, "result.json"), "x") as handle:
        json.dump({"facts": facts, **outcome}, handle, indent=1, sort_keys=True)
    ready = bool((outcome.get("final") or {}).get("ready"))
    print("venv ready=%s seconds=%s" % (ready, facts["seconds"]))
    return 0 if ready else 1


def facts(venv_path, app, candsrc, out_json):
    sys.path.insert(0, candsrc)
    from ouroboros import runtime_venv

    root = pathlib.Path(venv_path)
    record = {"venv": venv_path, "app": app, "exists": root.is_dir()}
    if root.is_dir():
        record["recorded_state"] = runtime_venv.recorded_state(root)
        record["base_interpreter"] = runtime_venv.base_interpreter(root)
        record["base_interpreter_realpath"] = os.path.realpath(record["base_interpreter"]) if record[
            "base_interpreter"] else ""
        record["own_python_vs_app"] = runtime_venv.own_python(root, app)
        record["interpreter"] = runtime_venv.interpreter(root)
        for name in (runtime_venv.MARKER_NAME, "pyvenv.cfg"):
            try:
                record[name] = (root / name).read_text(encoding="utf-8")
            except OSError as exc:
                record[name] = "unreadable: %s" % exc
        record["bin"] = {}
        for entry in sorted((root / "bin").iterdir()):
            if entry.is_symlink():
                record["bin"][entry.name] = "-> " + os.readlink(entry)
            else:
                with open(entry, "rb") as handle:
                    head = handle.read(160).split(b"\n")[0]
                record["bin"][entry.name] = "file " + head.decode("utf-8", "replace")
        site = sorted(root.glob("lib/python*/site-packages"))
        record["dist_info"] = sorted(p.name for p in site[0].glob("*.dist-info")) if site else []
        probe = subprocess.run([runtime_venv.interpreter(root), "-I", "-B", "-c",
                                "import json, os, sqlite3, sys; print(json.dumps({'version': sys.version.split()[0], "
                                "'sqlite': sqlite3.sqlite_version, 'executable': sys.executable, 'real': "
                                "os.path.realpath(sys.executable), 'prefix': sys.prefix, 'base_prefix': "
                                "sys.base_prefix, 'base_executable': getattr(sys, '_base_executable', '')}))"],
                               capture_output=True, text=True, timeout=120)
        record["interpreter_run"] = {"returncode": probe.returncode, "stdout": probe.stdout.strip()[-2000:],
                                     "stderr": probe.stderr.strip()[-2000:]}
    siblings = sorted(p.name for p in root.parent.iterdir() if p.name.startswith(root.name))
    record["siblings"] = siblings
    with open(out_json, "x") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["prepare"] and len(args) == 7:
        sys.exit(prepare(*args[1:]))
    if args[:1] == ["facts"] and len(args) == 5:
        sys.exit(facts(*args[1:]))
    raise SystemExit(__doc__)
