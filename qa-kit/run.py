"""Windows packaging acceptance, ONLY on a disposable GitHub-hosted Windows runner with a fictitious profile:  python -I -B run.py <command> ...

  preflight   every file the kit calls is there, the vendored bytes are what SOURCES.sha256 says, the kit compiles, and (Windows)
              PowerShell parses the kit's scripts, the C# helpers compile and the UI Automation stack answers; costs seconds
  visible     leg 1, its own runner and a CLEAN profile without system Git: the real installer wizard opened as a person opens it and driven
              only through UIA InvokePattern in its own process tree; frames of the wizard and of the app's first-run wizard
  migration   leg 2, its own runner: the published old release ZIP creates a profile and a chat -> the new package through the same
              ordinary wizard, whose Finish opens the app -> Update + handoff -> cold start by the Start .lnk -> uninstall -> new ZIP reopen
  pack        pack the Setup once from the candidate's one built ZIP (its payload extracted; no leg runs): the installer smoke's route
  summary     the decisions, versions and frames of a finished run

The installer's facts below are a CODE contract read from its source (packaging/windows/ouroboros.iss, scripts/pack_windows_setup.ps1),
not confirmed on Windows. The reviewed installer of the candidate governs; override a key with --contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "vendor" / "windows_update_4220_e21fee6")]

import prepare_build as pb  # noqa: E402
import windows_host as wh  # noqa: E402

CONTRACT = {
    "install_dir": r"%LOCALAPPDATA%\Programs\Ouroboros",                                       # the one path the app treats as its installed copy
    "start_menu_shortcut": r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Ouroboros.lnk",   # [Icons]: Target + AppUserModelID com.ouroboros.agent, no args, no IconFilename
    "uninstall_key": "{6FB7704B-C2C5-4252-A5D0-BEC473FD007D}_is1",                             # HKCU, 64-bit view; read QuietUninstallString
    "visible_install_argv": ["{installer}", "/LANG=en"],                                       # both legs: the ordinary wizard; Finish has Launch Ouroboros checked
    "files_not_in_zip": ["unins*.exe", "unins*.dat", "unins*.msg"],
    "success_exit_codes": [0],
}
KIT_PY = ("acceptance.py", "host_win.py", "judge.py", "projection.py", "run.py")
VENDOR = tuple(f"vendor/windows_update_4220_e21fee6/{name}.py" for name in
               ("chat_reply_witness", "fixture_model", "git_traces", "prepare_build", "scenario", "uia_reading", "update_offer", "windows_host"))
SHIPPED = (*KIT_PY, "Invoke-Acceptance.ps1", "SOURCES.sha256", "candidate.json", "ps/BrandNative.cs", "ps/ExplorerCapture.cs",
           "ps/brand-probe.ps1", "ps/capture-frames.ps1", "ps/parse-check.ps1", *VENDOR)  # every file this kit opens or calls
# The published old release the former user comes from: its public bytes by digest (never rebuilt, never replaced by the candidate).
OLD_RELEASE = {"version": "7.12.0", "source_sha": "6fc2500494fb8d8b2fad38da31db33e3991fa726",
               "zip_sha256": "bd92f35119f86da8361a3aa43f4e694ff7e7f79450219370cbcec0a93832f98d",
               "exe_sha256": "3cf8c391bb65540a2683d2391afe580cd6d29d81b3e9d853e5f2e62ad7a101a0",
               "remote": "https://github.com/razzant/ouroboros",
               "main_summary_tail": " \u00b7 Processing: Route default (inherited)"}  # its first-run wizard ends the Main summary row so


class Stop(Exception):
    """A refusal before the acceptance run exists: one fixed code, which is all that is published."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def record_stop(evidence: pathlib.Path, verdict: str, phase: str, code: str, *, started: bool | str = False, **facts: object) -> None:
    """A truthful result for a run that left none: fixed words, numbers and booleans only; an existing result is never overwritten.
    ``started``: False when the acceptance run never began (so nothing was photographed), "unknown" when it was interrupted."""
    if (evidence / "result.json").is_file():
        return
    pb.write_json_atomic(evidence / "result.json", {
        "schema": 5, "verdict": verdict, "phase": phase, "code": code, "recorded_at": pb._utc_now(),  # noqa: SLF001
        "reasons": [{"verdict": verdict, "phase": phase, "code": code}], "checks": {}, "visual": "n/a",
        "facts": {"acceptance_run_started": started, **facts}, "declared": {"screenshots_collected": False if started is False else "unknown"}})


def _stream(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_record_stop(args: argparse.Namespace) -> int:
    """For a job that stopped without a run's result: write the structured result if there is none, and keep an earlier helper's
    record truthful (#4220's early record declares ITS OWN capture and input policy and may carry a tool's text)."""
    evidence = pathlib.Path(args.evidence)
    path = evidence / "result.json"
    word = lambda v: v if isinstance(v, str) and re.fullmatch(r"[a-z0-9_]{1,80}", v) else "unspecified"  # noqa: E731
    started = "unknown" if args.run_may_have_started else False
    if not path.is_file():
        record_stop(evidence, "unconfirmed", word(args.phase), word(args.code), started=started)
        return 0
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    if not doc.get("facts") and not doc.get("checks"):  # an early record: nothing ran, so nothing of it may say otherwise
        reasons = [{k: r.get(k) for k in ("verdict", "phase", "code")} for r in doc.get("reasons") or [] if isinstance(r, dict)]
        doc.update(reasons=reasons, declared={"screenshots_collected": False if started is False else "unknown"}, facts={"acceptance_run_started": started})
        pb.write_json_atomic(path, doc)
    return 0


def require_disposable() -> None:
    if sys.platform != "win32":
        raise SystemExit("Windows only.")
    if os.environ.get("OUROBOROS_QA_DISPOSABLE") != "1" or not wh.disposable_hosted_runner():
        raise SystemExit("Refused: only a disposable GitHub-hosted Windows runner with OUROBOROS_QA_DISPOSABLE=1.")


def uia_stack(candidate_root: str) -> dict:
    """The UI Automation stack the wizard driver stands on: comtypes imports, and the candidate's own skill loads (read only, nothing opened)."""
    out: dict = {"comtypes": False, "skill": False}
    try:
        import comtypes  # noqa: F401
        out["comtypes"] = True
    except Exception as exc:  # noqa: BLE001
        out["comtypes_problem"] = type(exc).__name__
        return out
    if candidate_root:
        for entry in (str(pathlib.Path(candidate_root, "skills")), candidate_root):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        try:
            from windows_computer_use.plugin import WindowsComputerUse
            WindowsComputerUse(api=None)
            out["skill"] = True
        except Exception as exc:  # noqa: BLE001
            out["skill_problem"] = type(exc).__name__
    return out


def cmd_preflight(args: argparse.Namespace) -> int:
    missing = [rel for rel in SHIPPED if not (HERE / rel).is_file()]
    wrong = []
    for line in (HERE / "SOURCES.sha256").read_text(encoding="utf-8").splitlines() if "SOURCES.sha256" not in missing else []:
        if line.strip() and not line.startswith("#"):
            digest, _, rel = line.partition("  ")
            if not (HERE / rel).is_file() or _stream(HERE / rel) != digest:
                wrong.append(rel)
    bad = []
    for name in (n for n in KIT_PY if n not in missing):
        try:
            compile((HERE / name).read_text(encoding="utf-8"), name, "exec")
        except SyntaxError as exc:
            bad.append(f"{name}:{exc.lineno}")
    report: dict = {"kit_files_missing": missing, "vendored_bytes_differ": wrong, "kit_does_not_compile": bad, "platform": sys.platform}
    if sys.platform == "win32":
        import host_win as hw

        work = pathlib.Path(os.environ.get("TEMP", ".")) / f"kit-preflight-{int(time.time())}"
        work.mkdir(parents=True)
        code, _tail = hw.run_ps(hw.PS_DIR / "parse-check.ps1", ["-OutJson", work / "parse.json"], timeout=120)
        report["ps_parse"] = hw.read_json(work / "parse.json") or {"ok": False, "errors": ["no_output"], "exit": code}
        ok_win = bool(report["ps_parse"].get("ok"))
        if os.environ.get("OUROBOROS_QA_DISPOSABLE") == "1":  # compiles the C# helpers and says what this host permits; needs the desktop
            code, _tail = hw.run_ps(hw.PS_DIR / "brand-probe.ps1", ["-Mode", "self_test", "-OutDir", work / "self-test"], timeout=240)
            obs = hw.read_json(work / "self-test" / "observations.json")
            report["host_permits"] = {"script_exit": code, "fatal": obs.get("fatal"), "facts": obs.get("host"), "errors": obs.get("errors")}
            ok_win = ok_win and code == 0
        if args.uia:
            report["uia_stack"] = uia_stack(args.candidate_root)
            ok_win = ok_win and bool(report["uia_stack"]["comtypes"]) and (bool(report["uia_stack"]["skill"]) if args.candidate_root else True)
    else:
        ok_win = True
        report["windows_part"] = "skipped: not a Windows host"
    ok = not missing and not wrong and not bad and ok_win
    stack = report.get("uia_stack") or {}
    facts = dict(kit_files_missing=len(missing), vendored_bytes_differ=len(wrong), kit_does_not_compile=len(bad),
                 windows_checks_run=sys.platform == "win32", powershell_parse_ok=bool((report.get("ps_parse") or {}).get("ok")),
                 host_self_test_ok=(report.get("host_permits") or {}).get("script_exit") == 0,
                 uia_ok=bool(stack.get("comtypes")) and bool(stack.get("skill")))
    if not ok and args.evidence:
        record_stop(pathlib.Path(args.evidence), "unconfirmed", "preflight", "preflight_failed", nothing_installed_or_launched=True, **facts)
    if args.out:  # the details stay in this file on the host: it is never selected for upload
        pathlib.Path(args.out).write_text(json.dumps({**report, "ok": ok}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"preflight": "ok" if ok else "preflight_failed", **facts}))  # the job log gets fixed words, counts and booleans
    return 0 if ok else 2


def fresh(path: str) -> pathlib.Path:
    folder = pathlib.Path(path)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit("the evidence folder must be new or empty")  # nothing is written into someone else's folder
    return folder


def contract(args: argparse.Namespace) -> dict:
    return {**CONTRACT, **(json.loads(pathlib.Path(args.contract).read_text(encoding="utf-8")) if args.contract else {})}


def pack(candidate: pathlib.Path, payload: pathlib.Path, out_dir: pathlib.Path, compiler: str) -> tuple[pathlib.Path, dict]:
    """The author's packer: ``scripts/pack_windows_setup.ps1 -PayloadRoot <Ouroboros folder> -OutputDir <empty folder> [-Compiler ISCC.exe]``."""
    # The public release jobs run this packer in PowerShell 7, not legacy Windows PowerShell.
    cmd = ["pwsh.exe", "-NoProfile", "-NonInteractive", "-File", str(candidate / "scripts" / "pack_windows_setup.ps1"),
           "-PayloadRoot", str(payload), "-OutputDir", str(out_dir)] + (["-Compiler", compiler] if compiler else [])
    started = time.monotonic()
    done = subprocess.run(cmd, cwd=candidate, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    found = sorted(out_dir.glob("Ouroboros-Setup-*-windows-x64.exe")) if out_dir.is_dir() else []
    facts = {"packer_exit_code": done.returncode, "installers_found": len(found), "seconds": round(time.monotonic() - started, 1)}
    if done.returncode or len(found) != 1:
        # Keep diagnostics from the public compiler on its synthetic build inputs; never publish its raw streams.
        import shutil
        from acceptance import strip_paths
        diagnostics = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", done.stderr or "").splitlines()
        messages = [pb._scrub_text(strip_paths(line), 220) for line in diagnostics[-12:] if line.strip()]
        free_mb = shutil.disk_usage(payload).free // (1 << 20)
        print(f"packer_failed exit={done.returncode} installers={len(found)} free_MiB={free_mb} "
              f"diagnostics={json.dumps(messages)}", file=sys.stderr)
        raise Stop("packer_failed")
    return found[0], {**facts, "installer_name": found[0].name, "installer_bytes": found[0].stat().st_size}


def resolve(args: argparse.Namespace) -> tuple[dict, dict]:
    """The spec the run needs, and what was packed. Everything about the candidate is read from its own package manifest.

    The packer's payload is the candidate's ONE built ZIP, EXTRACTED (unpack_candidate_archive, into the evidence's parent):
    the corrected packer refuses a raw PyInstaller dist, whose Hidden entries are files Setup would silently leave out, so Setup
    is packed from the ZIP's own extracted files, exactly as the release packs it. A leg that consumed no ZIP recorded a fake
    digest here before; both legs now record the real ZIP they pack from."""
    cand = pathlib.Path(args.candidate_root) if args.candidate_root else None
    if cand is None:
        raise Stop("candidate_root_with_a_built_package_is_required")
    zips = [pathlib.Path(args.zip)] if getattr(args, "zip", "") and args.mode == "migration" else sorted((cand / "dist").glob("Ouroboros-*-windows-x64.zip"))
    if len(zips) != 1:
        raise Stop("the_build_must_leave_exactly_one_windows_zip")
    if args.payload_root:  # an explicit payload folder keeps its old meaning; the packer still refuses what Setup must not install
        payload = pathlib.Path(args.payload_root)
        manifests = sorted(payload.rglob("repo_bundle_manifest.json")) if payload.is_dir() else []
        manifest = json.loads(manifests[0].read_text(encoding="utf-8")) if len(manifests) == 1 else {}
    else:
        payload, manifest, _gits = pb.unpack_candidate_archive(zips[0], pathlib.Path(args.evidence).parent / "zip-payload")
    if not manifest:
        raise Stop("candidate_root_with_a_built_package_is_required")
    if manifest.get("source_sha") == OLD_RELEASE["source_sha"]:
        raise Stop("the_candidate_package_is_the_old_release")
    spec: dict = {"candidate_sha": str(manifest.get("source_sha") or "0" * 40), "candidate_root": str(cand),
                  "payload_root": str(payload), "zip": {"path": str(zips[0]), "sha256": _stream(zips[0])},
                  "old_release": dict(OLD_RELEASE)}
    if args.mode == "migration":
        mirror = str(manifest.get("managed_remote_url") or "")
        if not re.match(r"^[A-Za-z]:[\\/]", mirror) or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_sha") or "")):
            # an unbound package would update from the real GitHub
            raise Stop("migration_needs_one_zip_and_a_package_bound_to_a_local_mirror_and_a_commit")
        spec.update(mirror_url=mirror)
    else:
        spec.update(mirror_url=str(pathlib.Path(args.evidence) / "qa" / "mirror.git"))
    if args.installer and not pathlib.Path(args.installer).is_file():
        raise Stop("the_installer_given_is_not_a_file")
    if args.installer:
        installer, packaging = pathlib.Path(args.installer), {"installer_supplied": True}
    else:
        installer, packaging = pack(cand, payload, pathlib.Path(args.evidence).parent / "installer-out", args.compiler)
    c = contract(args)
    spec["installer"] = {**{k: c[k] for k in ("install_dir", "start_menu_shortcut", "uninstall_key", "visible_install_argv",
                                              "files_not_in_zip", "success_exit_codes")}, "path": str(installer), "sha256": _stream(installer)}
    return spec, packaging


def _pack_facts(spec: dict, packaging: dict) -> dict:
    """What `pack` leaves beside the installer: names and digests only, never a host path (write_json_atomic enforces that again:
    it refuses any Windows drive or UNC string, and this file stays on the host for the workflow step that runs the smoke)."""
    return {"schema": 1, "candidate_sha": spec["candidate_sha"],
            "zip": {"name": pathlib.Path(spec["zip"]["path"]).name, "sha256": spec["zip"]["sha256"]},
            "installer": {"name": pathlib.Path(spec["installer"]["path"]).name, "sha256": spec["installer"]["sha256"],
                          "bytes": packaging.get("installer_bytes"), "packer_exit_code": packaging.get("packer_exit_code")}}


def cmd_pack(args: argparse.Namespace) -> int:
    """Pack the Setup once without running a leg: the installer smoke's route. The same one-ZIP requirement, the same extracted
    payload and the same author's packer as a leg's own pack; the facts file it leaves names the installer and the ZIP for its
    caller. Fixed words and numbers only in the job log; the facts stay on the host, and a caller finds the files by these names
    in the directories it already owns (the packer's OutputDir is <evidence's parent>/installer-out)."""
    require_disposable()
    args.evidence = str(fresh(args.evidence))
    try:
        spec, packaging = resolve(args)
    except Stop as stop:
        print(f"pack_stop {stop.code}")
        return 2
    facts = _pack_facts(spec, packaging)
    pb.write_json_atomic(pathlib.Path(args.evidence) / "installer-facts.json", facts)
    print(json.dumps({"pack": "ok", "installer_sha256": facts["installer"]["sha256"],
                      "installer_name": facts["installer"]["name"], "installer_bytes": facts["installer"]["bytes"]}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    require_disposable()
    args.evidence = str(fresh(args.evidence))
    args.tag = args.tag or f"r{int(time.time())}"
    try:
        import acceptance as ac

        spec, packaging = resolve(args)
        run = ac.AcceptanceRun(spec, args)
        run.result["facts"]["packaging"] = packaging
        if args.projection_file and pathlib.Path(args.projection_file).is_file():  # the reviewed SHA and the carrier SHA, kept apart
            projection = json.loads(pathlib.Path(args.projection_file).read_text(encoding="utf-8"))
            run.result["facts"]["projection"] = {k: projection.get(k) for k in ("reviewed_sha", "carrier_sha", "projected", "files_changed", "version_before", "version_after", "negative_control")}
    except Stop as stop:
        record_stop(pathlib.Path(args.evidence), "unconfirmed", "prepare", stop.code)
        print(summary(pathlib.Path(args.evidence)))
        return 2
    except Exception as exc:  # noqa: BLE001 - a harness defect still leaves a structured result, with the type only
        record_stop(pathlib.Path(args.evidence), "error", "prepare", "harness_error", type=type(exc).__name__)
        print(summary(pathlib.Path(args.evidence)))
        return 3
    code = run.run()
    print(summary(pathlib.Path(args.evidence)))
    return code


def summary(evidence: pathlib.Path) -> str:
    result = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
    facts = result.get("facts") or {}
    lines = [f"### Packaging acceptance ({facts.get('mode', 'stopped early')}): **{result.get('verdict')}** / {result.get('phase')} / {result.get('code')}",
             f"visual: {result.get('visual', 'n/a')}", ""]
    if facts.get("install"):  # how this leg installed: seen and pressed are what happened, not what was meant
        lines += [f"- install: `{facts['install'].get('mode')}`, wizard seen in this leg: `{facts['install'].get('wizard_seen_in_this_leg')}`, "
                  f"Finish pressed: `{facts['install'].get('finish_pressed')}`. {facts['install'].get('note', '')}", ""]
    lines += [f"- launch attempts, in order: `{', '.join(facts.get('launch_attempts') or []) or 'none recorded'}`; launches observed: "
              f"`{', '.join(facts.get('launches_observed') or []) or 'none recorded'}`; frames kept: `{facts.get('frames_kept', 'unknown')}`", ""]
    if facts.get("projection"):
        pr = facts["projection"]
        lines += [f"- reviewed SHA `{pr.get('reviewed_sha')}`, carrier SHA `{pr.get('carrier_sha')}` (local only), files changed `{pr.get('files_changed')}`, "
                  f"VERSION `{pr.get('version_before')}` -> `{pr.get('version_after')}`", ""]
    if facts.get("versions"):  # whose code each window was: the process that holds the installation, and the body that served it
        lines += ["| Window | Title version | Process | Body version | Body revision |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| `{state}` | {v.get('window_title_version')} | {v['window_process'].get('copy')}/{v['window_process'].get('kind')} | "
                  f"{v.get('body_version')} | {v.get('body_revision')} |" for state, v in facts["versions"].items()] + [""]
    lines += ["| Decision | State | Required |", "| --- | --- | --- |"]
    lines += [f"| `{name}` | {o['state']} | {'yes' if o.get('required') else 'no'} |" for name, o in (facts.get("observed") or {}).items()]
    lines += ["", "| Frame | State |", "| --- | --- |"] + [f"| `{name}` | {f['state']} |" for name, f in (facts.get("frames") or {}).items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--out", default="")
    p.add_argument("--evidence", default="", help="where a failed preflight records its structured result")
    p.add_argument("--uia", action="store_true", help="also check the UI Automation stack (comtypes and, with --candidate-root, the candidate's skill)")
    p.add_argument("--candidate-root", default="")
    p.set_defaults(fn=cmd_preflight)
    p = sub.add_parser("pack")
    p.set_defaults(fn=cmd_pack, mode="pack", installer="", zip="")
    p.add_argument("--evidence", required=True)
    p.add_argument("--candidate-root", required=True)
    p.add_argument("--compiler", default="", help="ISCC.exe of the pinned Inno Setup, for the packer")
    p.add_argument("--payload-root", default="", help="the packer's payload folder (default: the one built ZIP, extracted)")
    p.add_argument("--contract", default="", help="JSON overriding CONTRACT keys if the installer's author changes them")
    for name in ("visible", "migration"):
        p = sub.add_parser(name)
        p.set_defaults(fn=cmd_run, mode=name)
        p.add_argument("--evidence", required=True)
        p.add_argument("--tag", default="")
        p.add_argument("--contract", default="", help="JSON overriding CONTRACT keys if the installer's author changes them")
        p.add_argument("--candidate-root", default="", help="the candidate checkout: its UI Automation skill, its tool directories, its dist/ build")
        p.add_argument("--payload-root", default="", help="the packer's payload folder (default: the one built ZIP, extracted)")
        p.add_argument("--installer", default="", help="a pre-built installer; otherwise the author's packer is run on the payload")
        p.add_argument("--compiler", default="", help="ISCC.exe of the pinned Inno Setup, for the packer")
        p.add_argument("--projection-file", default="", help="projection.json: the reviewed SHA and the carrier SHA, recorded in the result")
        p.add_argument("--launch-route", choices=("shellexecute", "explorer"), default="shellexecute",
                       help="shellexecute: cmd /c start of the real .lnk under our environment (Git traces and the migration's mirror redirect reach the app); explorer: the running shell opens it as Start does (neither reaches that start: a migration then updates and starts only if the profile's source reads as the mirror without them)")
        p.add_argument("--wizard-minutes", type=float, default=30.0)
        p.add_argument("--skip-explorer", action="store_true")
        p.add_argument("--skip-brand", action="store_true")
        if name == "migration":
            p.add_argument("--zip", default="")
            p.add_argument("--old-zip", default="", help="the published old release ZIP, downloaded as it is; checked against its pinned digest")
        else:
            p.add_argument("--uninstall-runner-git", action="store_true",
                           help="opt in to #4220's verified uninstall of the HOSTED runner's Git; without it a present Git is refused")
    p = sub.add_parser("record-stop")
    p.add_argument("--evidence", required=True)
    p.add_argument("--phase", default="job")
    p.add_argument("--code", default="stopped_before_a_result")
    p.add_argument("--run-may-have-started", action="store_true", help="the acceptance step was entered and left no result: record unknown")
    p.set_defaults(fn=cmd_record_stop)
    p = sub.add_parser("summary")
    p.add_argument("--evidence", required=True)
    p.set_defaults(fn=lambda a: print(summary(pathlib.Path(a.evidence))) or 0)
    args = top.parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as exc:  # noqa: BLE001 - a traceback would print host paths and a tool's own text into the job log
        print(f"harness_error {type(exc).__name__}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
