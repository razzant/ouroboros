"""Native Windows qualification of the unified Update on the real candidate and the released v7.8.0 app.

Equipment, not product. Run by ``run.ps1`` on a disposable Windows x64 runner after it downloaded and verified
the released app, the bare CPython and the wheelhouse. Every process runs inside a harness Job per private
root (``wq_custody``); every case ends PASS, FAIL, UNKNOWN or NOT_RUN from explicit checks (``wq_verdict``),
and the exit code is 0 only for the complete selection with every case PASS and custody confirmed.

The product commit and the equipment commit are two identities, never one: the job runs on a pushed equipment
commit whose one parent is the bound candidate and which adds only this directory and its workflow file
(``bind_equipment``); the candidate is checked out separately at its own commit and tree.

    python -E -s -B qualify.py --inputs inputs.json --run-root D:\\wq-1234 --release-dir <extracted app>
        --bare-python <bare python.exe> --wheelhouse <dir> --candidate <checkout>
        --equipment-repo <checkout of the equipment commit> --equipment-commit <its sha> [--cases A,B]
    python -E -s -B qualify.py --bind-only --inputs inputs.json --run-root D:\\wq-1234 --candidate <checkout>
        --equipment-repo <checkout> --equipment-commit <sha>   # refuses before anything is provisioned
    python -E -s -B qualify.py --demonstrate-false-verdict      # must exit non-zero (pipeline canary)
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import traceback
import types

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wq_fixtures as fx  # noqa: E402
from wq_verdict import EXIT, MODES, NOT_RUN, REQUIRED, Case, dumps, exit_code, select  # noqa: E402

PLACEHOLDER = re.compile(r"REPLACE|TODO|^$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
# What the equipment commit may add on top of the candidate: this directory and the workflow that runs it.
EQUIPMENT_DIR, WORKFLOW = "devtools/update_windows_qualification", ".github/workflows/update-windows-qualification.yml"


class Raw:
    """Append-only events, one JSON line each, flushed and fsynced before the next step."""

    def __init__(self, path):
        self.file = open(path, "x", encoding="utf-8")

    def emit(self, event, **facts):
        row = {"event": event, "time_ns": time.time_ns(), **facts}
        self.file.write(dumps(row) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        print(dumps(row)[:400], flush=True)


def demonstrate_false_verdict():
    """One deliberately refuted check and one rejected activation reach a non-zero exit code."""
    from wq_verdict import activation_request
    false = Case("A")
    false.check("deliberately_false", False)
    rejected = Case("B")
    adoption, facts = activation_request(409, {"error": "candidate_not_ready", "outcome": "not_applied"}, "c1")
    rejected.check("activation_accepted", bool(adoption), **facts)
    statuses = {name: "PASS" for name in REQUIRED}
    statuses.update(A=false.status, B=rejected.status)
    code = exit_code(statuses, REQUIRED, False, True)
    print(dumps({"demonstration": "false verdict and rejected activation", "A": false.status, "B": rejected.status,
                 "exit_code": code}))
    return code


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv, **kwargs):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=600, **kwargs)


def _git_bytes(repo, *args):
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=600)
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args[:2])} failed in {repo}: {done.stderr.decode(errors='replace')[-400:]}")
    return done.stdout


def bind_equipment(equipment_repo, equipment_commit, candidate_sha, candidate_tree):
    """Both identities, apart: the candidate (the product under test) and the equipment commit the job runs on.

    The equipment commit must be the checkout's HEAD, have exactly one parent, the bound candidate, whose tree is
    the bound tree, and change nothing but ``EQUIPMENT_DIR`` and ``WORKFLOW`` (the workflow present and
    byte-identical to the copy beside this file, which the portable tests read). Anything else refuses (``SystemExit``)
    before a run: a product change riding on the equipment commit would otherwise be qualified under the
    candidate's name."""
    repo = pathlib.Path(equipment_repo).resolve()
    for name, value in (("equipment commit", equipment_commit), ("candidate sha", candidate_sha),
                        ("candidate tree", candidate_tree)):
        if not HEX40.match(str(value or "")):
            raise SystemExit(f"{name} is not a 40-hex value: {value!r}; nothing was run")
    head = _git_bytes(repo, "rev-parse", "HEAD").decode().strip()
    if head != equipment_commit:
        raise SystemExit(f"the equipment checkout is at {head}, the job names {equipment_commit}")
    parents = _git_bytes(repo, "rev-list", "--parents", "-n", "1", equipment_commit).decode().split()[1:]
    if parents != [candidate_sha]:
        raise SystemExit(f"the equipment commit's parents are {parents}, not exactly the candidate {candidate_sha}")
    tree = _git_bytes(repo, "rev-parse", f"{candidate_sha}^{{tree}}").decode().strip()
    if tree != candidate_tree:
        raise SystemExit(f"the candidate's tree is {tree}, inputs bind {candidate_tree}")
    changed = [path for path in _git_bytes(repo, "diff", "--name-only", "--no-renames", "-z", candidate_sha,
                                           equipment_commit).decode("utf-8", "surrogateescape").split("\0") if path]
    foreign = [path for path in changed if not (path.startswith(EQUIPMENT_DIR + "/") or path == WORKFLOW)]
    if foreign:
        raise SystemExit(f"the equipment commit may change only {EQUIPMENT_DIR}/ and {WORKFLOW}; it also changes "
                         f"{sorted(foreign)[:20]}")
    files = {}
    listing = _git_bytes(repo, "ls-tree", "-r", "-z", equipment_commit, "--", EQUIPMENT_DIR, WORKFLOW)
    for entry in listing.decode("utf-8", "surrogateescape").split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            blob = meta.split()[2]
            files[path] = {"blob": blob, "sha256": hashlib.sha256(_git_bytes(repo, "cat-file", "blob", blob)).hexdigest()}
    template, running = files.get(f"{EQUIPMENT_DIR}/{pathlib.PurePosixPath(WORKFLOW).name}"), files.get(WORKFLOW)
    if template is None or running is None or template["blob"] != running["blob"]:
        raise SystemExit(f"{WORKFLOW} differs from the tested copy in {EQUIPMENT_DIR}/")
    return {"product": {"sha": candidate_sha, "tree": tree},
            "equipment": {"commit": equipment_commit, "tree": _git_bytes(repo, "rev-parse", f"{equipment_commit}^{{tree}}")
                          .decode().strip(), "parent": candidate_sha, "changed_paths": sorted(changed), "files": files}}


def candidate_checkout(repo, candidate):
    """``(head, tree)`` of the candidate checkout; refuses one that is not exactly the bound commit and tree."""
    head = _run(["git", "-C", repo, "rev-parse", "HEAD"]).stdout.strip()
    tree = _run(["git", "-C", repo, "rev-parse", "HEAD^{tree}"]).stdout.strip()
    if head != candidate["sha"] or tree != candidate["tree"]:
        raise SystemExit(f"candidate checkout is {head}/{tree}, inputs bind {candidate['sha']}/{candidate['tree']}")
    return head, tree


def bind_only(args):
    """``--bind-only``: the two identities checked and written to ``evidence/bindings.json`` before anything is
    downloaded; 0 when bound, 5 (usage) with the reason otherwise. Portable: Git only."""
    try:
        if not all((args.inputs, args.run_root, args.candidate, args.equipment_repo, args.equipment_commit)):
            raise SystemExit("--bind-only needs --inputs, --run-root, --candidate, --equipment-repo, --equipment-commit")
        candidate = json.loads(pathlib.Path(args.inputs).read_text(encoding="utf-8"))["candidate"]
        for name in ("sha", "tree"):
            if not HEX40.match(str(candidate.get(name) or "")):
                raise SystemExit(f"inputs.candidate.{name} is not bound to a 40-hex value; nothing was run")
        candidate_checkout(pathlib.Path(args.candidate).resolve(), candidate)
        bindings = bind_equipment(args.equipment_repo, args.equipment_commit, candidate["sha"], candidate["tree"])
    except (SystemExit, OSError, ValueError, KeyError, TypeError) as refused:
        print(f"bindings refused: {refused!r}", file=sys.stderr)
        return EXIT["usage"]
    evidence = pathlib.Path(args.run_root) / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "bindings.json").write_text(json.dumps(bindings, indent=1, sort_keys=True), encoding="utf-8")
    print(dumps({"bound": {"product": bindings["product"], "equipment_commit": bindings["equipment"]["commit"]}}))
    return 0


def bind_inputs(args, raw):
    """Refuse unbound or mismatching inputs before anything is provisioned; return the run context."""
    inputs = json.loads(pathlib.Path(args.inputs).read_text(encoding="utf-8"))
    candidate, release, python = inputs["candidate"], inputs["release"], inputs["python"]
    for name in ("sha", "tree"):
        if not HEX40.match(str(candidate.get(name) or "")):
            raise SystemExit(f"inputs.candidate.{name} is not bound to a 40-hex value; nothing was run")
    repo = pathlib.Path(args.candidate).resolve()
    head, tree = candidate_checkout(repo, candidate)
    bindings = bind_equipment(args.equipment_repo, args.equipment_commit, head, tree)
    raw.emit("equipment_bound", **bindings)
    exes = list(pathlib.Path(args.release_dir).resolve().rglob("Ouroboros.exe"))
    if len(exes) != 1:
        raise SystemExit(f"expected exactly one Ouroboros.exe under {args.release_dir}, found {len(exes)}")
    exe = exes[0]
    internal = exe.parent / "_internal"
    manifest = json.loads((internal / "repo_bundle_manifest.json").read_text(encoding="utf-8"))
    version = (internal / "VERSION").read_text(encoding="utf-8").strip()
    bundle_sha = _sha256(internal / "repo.bundle")
    if manifest.get("source_sha") != release["source_sha"] or version != release["version"] \
            or bundle_sha != manifest.get("bundle_sha256"):
        raise SystemExit(f"released app does not match its pin: manifest={manifest} version={version} "
                         f"bundle={bundle_sha}")
    bare = pathlib.Path(args.bare_python).resolve()
    probe = _run([bare, "-I", "-c", "import json, sys; print(json.dumps([sys.version.split()[0], sys.prefix, "
                  "sys.base_prefix, sys.maxsize > 2**32]))"])
    bare_version, prefix, base_prefix, wide = json.loads(probe.stdout)
    if bare_version != python["version"] or prefix != base_prefix or not wide or (bare.parent / "pyvenv.cfg").exists() \
            or "python-standalone" in str(bare).lower():
        raise SystemExit(f"bare interpreter refused: {probe.stdout} {probe.stderr}")
    git_ops = (repo / "supervisor" / "git_ops.py").read_text(encoding="utf-8", errors="replace")
    official = {manifest.get("managed_remote_url"), *re.findall(r'OFFICIAL_UPDATE_REMOTE_URL\s*=\s*"([^"]+)"', git_ops)}
    urls = sorted({variant for url in official if url for variant in (url, url[:-4] if url.endswith(".git") else url)})
    facts = {"candidate": {"sha": head, "tree": tree, "version": (repo / "VERSION").read_text().strip()},
             "release": {"exe": str(exe), "exe_sha256": _sha256(exe), "manifest": manifest, "version": version},
             "bare_python": {"path": str(bare), "version": bare_version, "prefix": prefix}, "official_urls": urls}
    raw.emit("inputs_bound", **facts)
    return types.SimpleNamespace(
        raw=raw, run_root=pathlib.Path(args.run_root), evidence=pathlib.Path(args.run_root) / "evidence",
        equipment=HERE, bindings=bindings, candidate=repo, candidate_sha=head, candidate_version=facts["candidate"]["version"],
        release={"dir": exe.parent.parent, "exe": exe, "internal": internal,
                 "python": internal / "python-standalone" / "python.exe", "manifest": manifest, "official_urls": urls},
        bare_python=bare, bare_root=bare.parent, wheelhouse=pathlib.Path(args.wheelhouse).resolve(),
        git_dir=str(pathlib.Path(shutil.which("git")).parent))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs")
    parser.add_argument("--run-root")
    parser.add_argument("--release-dir")
    parser.add_argument("--bare-python")
    parser.add_argument("--wheelhouse")
    parser.add_argument("--candidate")
    parser.add_argument("--equipment-repo", help="checkout of the pushed equipment commit (full history)")
    parser.add_argument("--equipment-commit", help="that commit's sha (GITHUB_SHA)")
    parser.add_argument("--bind-only", action="store_true", help="check and record both identities, run nothing")
    parser.add_argument("--cases", help="development subset; never a qualification (exit 3 at best)")
    parser.add_argument("--demonstrate-false-verdict", action="store_true")
    args = parser.parse_args(argv)
    if args.demonstrate_false_verdict:
        return demonstrate_false_verdict()
    if args.bind_only:
        return bind_only(args)
    try:
        selected, partial = select(args.cases)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT["usage"]
    if os.name != "nt" or sys.maxsize <= 2 ** 32:
        print("native Windows x64 required; nothing was measured", file=sys.stderr)
        return EXIT["unknown_or_not_run"]
    if not all((args.inputs, args.run_root, args.release_dir, args.bare_python, args.wheelhouse, args.candidate,
                args.equipment_repo, args.equipment_commit)):
        parser.error("every path argument is required for a run")
    run_root = pathlib.Path(args.run_root)
    if not run_root.is_absolute():
        parser.error("--run-root must be absolute")
    (run_root / "evidence").mkdir(parents=True, exist_ok=True)
    raw = Raw(run_root / "evidence" / "events.jsonl")
    statuses, reasons, teardowns, custody_ok = {}, {}, {}, True
    cases_out, bindings = [], None
    try:
        from wq_win import Win
        ctx = bind_inputs(args, raw)
        bindings = ctx.bindings
        ctx.win = Win()
        filesystem = ctx.win.filesystem(run_root)
        raw.emit("host", python=sys.version, windows=list(sys.getwindowsversion()), filesystem=filesystem,
                 selected=selected, partial=partial)
        if filesystem.get("filesystem") != "NTFS":
            raise SystemExit(f"NTFS required, run root is on {filesystem}; nothing was measured")
        pyd = (ctx.release["internal"] / "python-standalone" / "DLLs" / "_bz2.pyd").read_bytes()
        wheels = [str(fx.build_wheel(ctx.wheelhouse, version, pyd)) for version in sorted(set(fx.VERSIONS.values()))]
        ctx.commits = fx.build_commits(ctx.candidate, ctx.candidate_sha, run_root / "fixture-work")
        ctx.fixture_repo = run_root / "fixture-work"
        raw.emit("fixtures", wheels=wheels, commits=ctx.commits)
        from wq_cases import run_mode
        for label, names in MODES.items():
            if not any(name in selected for name in names):
                continue
            if not custody_ok:
                for name in names:
                    reasons[name] = "an earlier mode's process custody was not confirmed"
                continue
            cases, teardown = run_mode(ctx, label, selected)
            teardowns[label] = teardown
            custody_ok = custody_ok and teardown.get("status") == "confirmed"
            for name, case in cases.items():
                statuses[name] = case.status
                cases_out.append(case.as_dict())
    except BaseException as exc:  # an equipment or input failure: every unrun case stays NOT_RUN
        raw.emit("run_error", error=repr(exc), traceback=traceback.format_exc())
        reasons.setdefault("*", repr(exc))
    for name in selected:
        if name not in statuses:
            statuses[name] = NOT_RUN
            cases_out.append({"case": name, "status": NOT_RUN, "reason": reasons.get(name) or reasons.get("*", ""),
                              "checks": []})
    code = exit_code(statuses, selected, partial, custody_ok and all(
        row.get("status") == "confirmed" for row in teardowns.values()))
    summary = {"exit_code": code, "partial": partial, "selected": list(selected), "statuses": statuses,
               "teardowns": teardowns, "stub": fx.STUB_LABEL, "bindings": bindings, "cases": cases_out}
    (run_root / "evidence" / "verdicts.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    raw.emit("verdict", exit_code=code, statuses=statuses, partial=partial)
    raw.file.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
