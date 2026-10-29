"""One fresh observation or a cooperative holder; started only by probe.py."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import site
import sys

from probe import attempt
from winapi import Win


def load_extension(package):
    path = Path(package.__file__).parent / "_bz2.pyd"
    spec = importlib.util.spec_from_file_location("_bz2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extension_facts(win, module):
    compressor = module.BZ2Compressor()
    data = compressor.compress(b"ntfs-holder-proof") + compressor.flush()
    return {"file": module.__file__, "realpath": os.path.realpath(module.__file__),
            "roundtrip": module.BZ2Decompressor().decompress(data) == b"ntfs-holder-proof",
            **win.module(module.__file__)}


def observation(win, fixed, phase, package, extension):
    def lazy():
        # Each phase has a distinct, previously unimported module. No reload of
        # the cached generation marker can disguise old/new mixing.
        module = importlib.import_module("holder_fixture." + phase)
        return {"generation": module.GENERATION, "file": module.__file__,
                "realpath": os.path.realpath(module.__file__)}

    return {"schema": "ntfs-observation/1", "phase": phase, "pid": os.getpid(),
            "executable": sys.executable, "base_executable": getattr(sys, "_base_executable", None),
            "image": attempt(win.module), "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "sys_path": sys.path, "userbase": site.getuserbase(), "usersite": site.getusersitepackages(),
            "user_site_enabled": site.ENABLE_USER_SITE,
            "cwd": attempt(os.getcwd), "cwd_final": attempt(win.final_path, "."),
            "cached_package": {"generation": package.GENERATION, "file": package.__file__,
                               "realpath_now": os.path.realpath(package.__file__)},
            "selected_marker": attempt((fixed / "generation.txt").read_text),
            "new_import": attempt(lazy),
            "pyd": attempt(extension_facts, win, extension) if extension else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["observe", "hold"])
    parser.add_argument("fixed", type=Path)
    parser.add_argument("--kind", choices=["pyd", "image", "cwd", "redirector"])
    parser.add_argument("--phase", default="before")
    args = parser.parse_args()
    import holder_fixture

    win = Win()
    extension = load_extension(holder_fixture) if args.mode == "observe" or args.kind == "pyd" else None
    if args.kind == "cwd":
        os.chdir(Path(holder_fixture.__file__).parent)

    def view(phase):
        result = observation(win, args.fixed, phase, holder_fixture, extension)
        if args.kind == "cwd":
            result["relative_marker"] = attempt(Path("generation.txt").read_text)
        return result

    def send(value):
        print(json.dumps(value), flush=True)

    if args.mode == "observe":
        send(view(args.phase))
        return 0
    ready = view("before")
    if extension:
        ready["expected_pyd_device"] = attempt(win.final_path, extension.__file__, True)
    send({"event": "ready", "kind": args.kind, "observation": ready})
    for line in sys.stdin:
        request = json.loads(line)
        if request["phase"] == "exit":
            send({"event": "exit", "id": request["id"]})
            return 0
        send({"event": "observe", "id": request["id"], "observation": view(request["phase"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
