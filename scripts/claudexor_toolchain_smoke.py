#!/usr/bin/env python3
"""The real managed Node/npm toolchain behind the Claudexor CLI, on one platform.

PROVES — with disposable HOME/app/data/config roots, no ``node``/``npm``/``npx``/
``corepack`` reachable on PATH, no ambient npm configuration and no Claudexor
harness-binary override (``CLAUDEXOR_CODEX_BIN`` and friends), that the production
``ClaudexorRuntimeManager.ensure_cli_command()`` downloads the exact pinned
official Node archive (``.zip`` on Windows, ``.tar.gz`` elsewhere) and Claudexor
closure, extracts Node plus its regular-file npm tree, verifies and promotes both,
and returns ``[managed node, CLI]``. The managed Node then runs the returned CLI,
and runs the extracted ``npm-cli.js`` directly — never an npm ``.cmd``/``.ps1``
shim — to pack and globally install a local package offline into a disposable
prefix, whose entry point it runs. A fresh manager selects the same command
without re-promoting, and a deleted npm entrypoint is repaired from the verified
archive cache.

DOES NOT PROVE — by default, that any vendor program installs, logs in or runs.
The offline npm witness covers engine CLI runtime delivery only; vendor program
maintenance does not use that npm toolchain.

``--harness-install HARNESS`` additionally starts the isolated owned engine and
uses the same maintenance service as Connect, Settings and native tools. The
program must initially be absent, installation must settle with selected-version
evidence, and that selected executable must live beneath the disposable HOME.
It must answer --version directly and the engine's doctor must resolve the same
path/version. The owned daemon is stopped on every exit. No login, OAuth,
account or model task is attempted.

Usage:
    python -I scripts/claudexor_toolchain_smoke.py [--root EMPTY_DIR] [--harness-install HARNESS]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AMBIENT_TOOLS = ("node", "npm", "npx", "corepack")
# Beyond scrub_environment's owner/credential/process-override set: engine homes and
# harness-binary overrides, and ambient npm/Node configuration the runner may carry.
DROPPED_PREFIXES = ("CLAUDEXOR_", "NPM_CONFIG_", "NODE_", "COREPACK_")
STEP_TIMEOUT_SEC = 180
STOP_SETTLE_SEC = 30
WITNESS_PACKAGE = "ouroboros-toolchain-witness"
WITNESS_TOKEN = "managed-npm-install-ok"
NPM_QUIET = ("--offline", "--ignore-scripts", "--no-audit", "--no-fund",
             "--no-update-notifier", "--loglevel=error")


class WitnessFailure(RuntimeError):
    """A named refusal; every exit path that is not success carries one."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


def scrubbed_path(value: str) -> Tuple[str, List[str]]:
    """Drop every PATH entry that provides Node or an npm-family launcher."""
    kept: List[str] = []
    dropped: List[str] = []
    for entry in value.split(os.pathsep):
        if entry and any(shutil.which(tool, path=entry) for tool in AMBIENT_TOOLS):
            dropped.append(entry)
        else:
            kept.append(entry)
    return os.pathsep.join(kept), dropped


def isolate_environment(root: pathlib.Path) -> List[str]:
    """Rebind this process before any Ouroboros config import.

    Only HOME-shaped roots are set: the app and data roots then derive from the
    disposable home exactly as production derives them from a real one. Returns
    the PATH entries that were dropped.
    """
    from ouroboros.test_environment import scrub_environment

    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {
        key: value for key, value in scrub_environment(os.environ).items()
        if not key.upper().startswith(DROPPED_PREFIXES)
    }
    env["PATH"], dropped = scrubbed_path(env.get("PATH", ""))
    env.update(
        HOME=str(home), USERPROFILE=str(home),
        APPDATA=str(home / "AppData" / "Roaming"), LOCALAPPDATA=str(home / "AppData" / "Local"),
        XDG_CONFIG_HOME=str(home / ".config"), XDG_CACHE_HOME=str(home / ".cache"),
    )
    os.environ.clear()
    os.environ.update(env)
    return dropped


def _run(argv: Sequence[Any], code: str, *, cwd: Optional[pathlib.Path] = None,
         env: Optional[Dict[str, str]] = None) -> str:
    command = [str(part) for part in argv]
    try:
        completed = subprocess.run(
            command, cwd=str(cwd) if cwd else None, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=STEP_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WitnessFailure(code, f"{command[1:3]} did not complete: {type(exc).__name__}: {exc}") from exc
    if completed.returncode != 0:
        raise WitnessFailure(
            code, f"{command[1:3]} exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-2000:].strip()}",
        )
    return completed.stdout.strip()


def official_npm_cli(node: pathlib.Path) -> pathlib.Path:
    """Where the official Node distribution keeps npm relative to its executable."""
    if node.name.lower() == "node.exe":
        return node.parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
    return node.parent.parent / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"


def npm_install_witness(node: pathlib.Path, npm_cli: pathlib.Path, work: pathlib.Path) -> Dict[str, Any]:
    """Pack and globally install a local package with the managed npm, offline."""
    package = work / "package"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({
        "name": WITNESS_PACKAGE, "version": "1.0.0", "bin": {WITNESS_PACKAGE: "cli.js"},
    }), encoding="utf-8")
    (package / "cli.js").write_text(
        f"#!/usr/bin/env node\nconsole.log({json.dumps(WITNESS_TOKEN)});\n", encoding="utf-8")
    npm = [node, npm_cli]
    try:
        packed = json.loads(_run([*npm, "pack", "--json", "--pack-destination", work, *NPM_QUIET],
                                 "npm_pack_failed", cwd=package))
        tarball = work / str(packed[0]["filename"])
    except (ValueError, LookupError, TypeError) as exc:
        raise WitnessFailure("npm_pack_failed", f"npm pack returned no tarball: {exc}") from exc
    prefix = work / "prefix"
    _run([*npm, "install", "--global", "--prefix", prefix, *NPM_QUIET, tarball],
         "npm_install_failed", cwd=work)
    global_root = pathlib.Path(_run([*npm, "root", "--global", "--prefix", prefix],
                                    "npm_root_failed", cwd=work))
    output = _run([node, global_root / WITNESS_PACKAGE / "cli.js"], "npm_installed_entry_failed", cwd=work)
    if output != WITNESS_TOKEN:
        raise WitnessFailure("npm_installed_entry_failed", f"installed entry printed {output!r}")
    return {"tarball": tarball.name, "global_root": str(global_root)}


def _cli_json(argv: Sequence[Any], code: str, env: Dict[str, str],
              timeout: int = STEP_TIMEOUT_SEC) -> Dict[str, Any]:
    """Run one ``--json`` CLI verb in its own killable process group.

    Its single stdout object is the verdict; a refusal keeps the engine's own code.
    """
    from ouroboros.platform_layer import (
        kill_process_tree,
        merge_hidden_kwargs,
        subprocess_new_group_kwargs,
    )

    command = [str(part) for part in argv]
    verb = " ".join(command[2:4])
    try:
        proc = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            **merge_hidden_kwargs(subprocess_new_group_kwargs()),
        )
    except OSError as exc:
        raise WitnessFailure(code, f"{verb} could not start: {exc}") from exc
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        raise WitnessFailure(code, f"{verb} exceeded {timeout}s; its process tree was killed") from exc
    try:
        payload = json.loads(stdout)
    except ValueError:
        payload = None
    found = payload if isinstance(payload, dict) else {}
    if not found or proc.returncode != 0 or found.get("ok") is False:
        detail = found.get("refusal") or found.get("message") or (stderr or stdout)[-2000:].strip()
        raise WitnessFailure(str(found.get("code") or code), f"{verb} exited {proc.returncode}: {detail}")
    return found


def _contained_run(argv: Sequence[Any], code: str, env: Dict[str, str],
                   timeout: int = STEP_TIMEOUT_SEC) -> str:
    """``_run`` for a vendor executable: spawned inside a process container, always reaped.

    ``subprocess.run(timeout=)`` kills only the direct child, while a vendor's Node
    entry can start a native child, so its process tree could outlive the step.
    Here the whole tree is in custody from its first instruction, whatever the exit, and
    a member the container's kill sweep cannot prove gone is red.
    """
    from ouroboros.platform_layer import kill_process_tree, subprocess_hidden_kwargs
    from ouroboros.process_containment import ProcessContainer

    command = [str(part) for part in argv]
    what = f"{pathlib.Path(command[0]).name} {command[-1]}"
    container = ProcessContainer()
    try:
        try:
            proc = container.spawn(
                command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                **subprocess_hidden_kwargs(),
            )
        except OSError as exc:
            raise WitnessFailure(code, f"{what} could not start: {exc}") from exc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            kill_process_tree(proc)
            leaked = container.reap()
            try:
                proc.communicate(timeout=10)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            raise WitnessFailure(code, f"{what} exceeded {timeout}s; termination attempted"
                                 + (f"; survivors unconfirmed: {leaked}" if leaked
                                    else "; process tree reaped")) from exc
        leaked = container.reap()
    finally:
        container.close()
    if leaked:
        raise WitnessFailure("harness_process_uncontained", f"{what} exited {proc.returncode}, but {leaked}")
    if proc.returncode != 0:
        raise WitnessFailure(code, f"{what} exited {proc.returncode}: "
                                   f"{(stderr or stdout)[-2000:].strip()}")
    return stdout.strip()


def _reports_version(text: str, version: str) -> bool:
    return re.search(r"(?<![0-9A-Za-z.+-])" + re.escape(version) + r"(?![0-9A-Za-z.+-])", text) is not None


def verify_install_receipt(receipt: Dict[str, Any], harness: str, home: pathlib.Path, *,
                           windows: bool = os.name == "nt") -> pathlib.Path:
    """Verify selected-version evidence and real containment in the disposable HOME."""
    selected = receipt.get("after") or {}
    if (receipt.get("harness") != harness or receipt.get("state") != "succeeded"
            or receipt.get("termination") == "unconfirmed"
            or not all(isinstance(selected.get(key), str) and selected[key].strip()
                       for key in ("path", "version", "versionOutput"))):
        raise WitnessFailure("harness_receipt_invalid", f"no selected program version: {receipt}")
    binary = pathlib.Path(selected["path"])
    try:
        contained = home.resolve() in binary.resolve().parents
    except (OSError, RuntimeError):
        contained = False
    if (not binary.is_absolute() or not binary.is_file() or not contained
            or windows and binary.suffix.lower() in {".cmd", ".bat", ".ps1"}):
        raise WitnessFailure("harness_receipt_invalid", f"{binary} is not a direct executable within {home}")
    return binary


def doctor_installed_check(report: Dict[str, Any], harness: str, binary: pathlib.Path,
                           version: str) -> Dict[str, Any]:
    """The doctor's ``installed`` row must pass and name the receipt's executable and pin."""
    rows = report.get("harnesses")
    row = next((item for item in rows if isinstance(item, dict) and item.get("id") == harness),
               None) if isinstance(rows, list) else None
    checks = row.get("checks") if row else None
    installed = next((item for item in checks if isinstance(item, dict)
                      and item.get("id") == "installed"), None) if isinstance(checks, list) else None
    detail = str((installed or {}).get("detail") or "")
    if (not installed or installed.get("status") != "pass" or not _reports_version(detail, version)
            or str(binary).lower() not in detail.lower()):
        raise WitnessFailure("harness_doctor_unresolved",
                             f"doctor does not resolve {harness} {version} to {binary}: {row}")
    return {"status": row.get("status"), "installed": detail}


def _stop_owned_daemon(daemon: Any) -> str:
    """The production stop, re-verified for a bounded settle while it is unconfirmed.

    A member of the daemon's process group can outlive the confirmed service shutdown
    by a moment; a repeated stop only re-reads custody, and ``nothing_to_stop`` then
    means no owned process remains.
    """
    outcome = daemon.stop_outcome()
    deadline = time.monotonic() + STOP_SETTLE_SEC
    while outcome == "unconfirmed" and time.monotonic() < deadline:
        time.sleep(1)
        outcome = daemon.stop_outcome()
    return outcome


def doctor_witness(harness: str, command: List[str], env: Dict[str, str], binary: pathlib.Path,
                   version: str) -> Dict[str, Any]:
    """The production owned daemon answers the doctor, and is always stopped.

    Ouroboros starts its daemon itself (the managed runtime ships no ``claudexord.js``
    the CLI could auto-start), so the doctor rides that same seam: the ``/v2/harnesses``
    rows Ouroboros reads, then the engine's own ``doctor`` verb against that daemon.
    """
    from ouroboros.claudexor_daemon import ensure_owned_gateway, get_owned_daemon
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    daemon = get_owned_daemon()
    try:
        gateway = ensure_owned_gateway()
        try:
            rows = gateway.harnesses()
        finally:
            gateway.close()
        # Use the full doctor verb. On Windows, Claudexor <= 3.15.0 printed valid
        # doctor JSON and then aborted in libuv (Node 24 close race on forced
        # process.exit after fetch; fixed upstream in Claudexor 3.15.1, PR #356), so
        # the exit status here is part of the witness. The exact harness row is
        # checked below.
        report = _cli_json([*command, "doctor", "--json"], "harness_doctor_failed", env)
    except ClaudexorUnavailable as exc:
        _stop_owned_daemon(daemon)
        raise WitnessFailure(exc.code, f"owned daemon: {exc}") from exc
    except BaseException:
        _stop_owned_daemon(daemon)
        raise
    # The doctor answered, so the daemon ran: anything of it left alive is red.
    stopped = _stop_owned_daemon(daemon)
    if stopped not in ("stopped", "nothing_to_stop"):
        raise WitnessFailure("doctor_daemon_not_stopped", f"owned daemon stop outcome: {stopped}")
    return {"daemon_harnesses": doctor_installed_check({"harnesses": rows}, harness, binary, version),
            "cli_doctor": doctor_installed_check(report, harness, binary, version),
            "daemon_stop": stopped}


def harness_install_witness(harness: str, node: pathlib.Path, command: List[str],
                            cli_env: Dict[str, str], *, windows: bool = os.name == "nt") -> Dict[str, Any]:
    """Exercise the shared vendor maintenance service, with no legacy installer route."""
    from ouroboros import harness_maintenance as maintenance
    from ouroboros.claudexor_daemon import ensure_owned_gateway, get_owned_daemon
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    daemon = get_owned_daemon()
    try:
        gateway = ensure_owned_gateway()
        try:
            inventory = gateway.maintenance_harnesses([harness], fresh=True)
        finally:
            gateway.close()
        row = next((row for row in inventory["harnesses"] if row["harness"] == harness), None)
        if row is None or row["selection"]["path"] is not None:
            raise WitnessFailure("harness_preinstalled", "The requested program must initially be absent")
        if not row["actions"]["install"]["available"]:
            raise WitnessFailure("harness_install_refused", row["actions"]["install"]["reason"])
        receipt = maintenance.install_for_connect(harness)
        binary = verify_install_receipt(receipt, harness, pathlib.Path(os.environ["HOME"]), windows=windows)
        version = receipt["after"]["version"]
        harness_env = dict(cli_env, PATH=os.pathsep.join(
            [str(node.parent), str(binary.parent), cli_env.get("PATH", "")]))
        invocation = [node, binary] if windows and binary.suffix.lower() in {".js", ".cjs", ".mjs"} else [binary]
        direct = _contained_run([*invocation, "--version"], "harness_direct_failed", harness_env)
        if not _reports_version(direct, version):
            raise WitnessFailure("harness_direct_failed", f"{binary} --version printed {direct!r}")
        return {"harness": harness, "selected_version": version, "selected_binary": str(binary),
                "operation": receipt, "direct_version": direct, "by_name_probe": "engine_doctor",
                "doctor": doctor_witness(harness, command, cli_env, binary, version)}
    except ClaudexorUnavailable as exc:
        raise WitnessFailure(exc.code, f"production maintenance service: {exc}") from exc
    finally:
        stopped = _stop_owned_daemon(daemon)
        if stopped not in ("stopped", "nothing_to_stop"):
            raise WitnessFailure("maintenance_daemon_not_stopped", f"owned daemon stop outcome: {stopped}")


def run_witness(root: pathlib.Path, dropped_path: List[str],
                harness: Optional[str] = None) -> Dict[str, Any]:
    facts: Dict[str, Any] = {"isolated_root": str(root), "path_entries_dropped": dropped_path}
    ambient = {tool: found for tool in AMBIENT_TOOLS if (found := shutil.which(tool))}
    if ambient:
        raise WitnessFailure("ambient_node_present", f"PATH still provides {ambient}")
    overrides = sorted(key for key in os.environ if key.upper().startswith(DROPPED_PREFIXES)
                       or key.upper() == "OUROBOROS_CLAUDEXOR_BIN")
    if overrides:
        raise WitnessFailure("ambient_override_present", f"environment still carries {overrides}")

    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.claudexor_runtime import (
        ClaudexorRuntimeError,
        ClaudexorRuntimeManager,
        managed_runtime_dir,
    )
    from ouroboros.config import DATA_DIR
    from ouroboros.platform_layer import embedded_node_candidates, node_distribution_platform

    data_dir = pathlib.Path(DATA_DIR).resolve()
    if root not in data_dir.parents:
        raise WitnessFailure("data_root_not_isolated", f"DATA_DIR {data_dir} is outside {root}")
    manager = ClaudexorRuntimeManager()
    pin, platform_key = manager.pin, node_distribution_platform()
    if pin is None or pin.cli_entrypoint is None:
        raise WitnessFailure("runtime_cli_unpinned", "this checkout pins no managed Claudexor CLI")
    facts.update(engine_pin=pin.version, node_pin=pin.node_version, platform=platform_key,
                 data_dir=str(data_dir))
    try:
        command = manager.ensure_cli_command()
    except ClaudexorRuntimeError as exc:
        raise WitnessFailure(exc.code, str(exc)) from exc
    facts["node_archive"] = pin.node_artifacts[platform_key].archive_name
    node = pathlib.Path(command[0])
    node_root = next((path for path in manager._node_roots(pin.node_version, platform_key)
                      if embedded_node_candidates(path)[0] == node), None)
    if len(command) != 2 or node_root is None:
        raise WitnessFailure("cli_command_unexpected", f"{command} does not run an admitted managed Node")
    cli, npm_cli = pathlib.Path(command[1]), official_npm_cli(node)
    metadata_path = node_root / "managed-node.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or not str(metadata.get("archive_npm_cli") or "").endswith(
            "node_modules/npm/bin/npm-cli.js") or not npm_cli.is_file():
        raise WitnessFailure("npm_tree_missing", f"managed Node metadata {metadata} or {npm_cli} is absent")
    runtime_meta = json.loads((managed_runtime_dir(pin) / "managed-runtime.json").read_text(encoding="utf-8"))
    facts.update(managed_node=str(node), managed_cli=str(cli), npm_cli=str(npm_cli),
                 node_metadata_schema=metadata["schema_version"],
                 runtime_archive_source=runtime_meta.get("archive_source"))

    node_version = _run([node, "--version"], "managed_node_failed")
    if node_version != f"v{pin.node_version}":
        raise WitnessFailure("managed_node_failed", f"managed Node reports {node_version!r}")
    # The CLI observes the same owned data plane as the daemon.
    cli_env = dict(os.environ, CLAUDEXOR_CONFIG_DIR=str(owned_config_dir()))
    cli_version = _run([node, cli, "--version"], "managed_cli_failed", env=cli_env)
    if pin.version not in cli_version:
        raise WitnessFailure("managed_cli_failed", f"managed CLI reports {cli_version!r}")
    npm_version = _run([node, npm_cli, "--version"], "managed_npm_failed")
    facts.update(node_version=node_version, cli_version=cli_version, npm_version=npm_version)
    facts["npm_install"] = npm_install_witness(node, npm_cli, root / "npm-witness")

    before = metadata_path.stat()
    if (ClaudexorRuntimeManager().ensure_cli_command() != command
            or ClaudexorRuntimeManager().resolve_cli_command() != command
            or metadata_path.stat().st_mtime_ns != before.st_mtime_ns):
        raise WitnessFailure("reselect_unstable", "a fresh manager did not reselect the installed toolchain")
    npm_cli.unlink()
    try:
        repaired = ClaudexorRuntimeManager().ensure_cli_command()
    except ClaudexorRuntimeError as exc:
        raise WitnessFailure(exc.code, f"repair failed: {exc}") from exc
    repaired_node = pathlib.Path(repaired[0])
    if (repaired == command or npm_cli.exists()
            or _run([repaired_node, official_npm_cli(repaired_node), "--version"], "repair_failed") != npm_version):
        raise WitnessFailure("repair_failed", "npm repair did not preserve the old tree and select a complete new generation")
    facts["repair"] = "complete replacement generation published from verified cache; damaged generation retained"
    if harness:
        facts["harness_install"] = harness_install_witness(harness, repaired_node, repaired, cli_env)
    return facts


LIMITS = (
    "### What this check does NOT cover", "",
    "- **No vendor harness was installed, logged in or run.** A green row proves engine CLI runtime delivery.",
    "- No daemon, model, account or credential was involved.",
    "- npm was proven on an offline local-tarball install; vendor maintenance does not use it.",
)
HARNESS_LIMITS = (
    "### What this check does NOT cover", "",
    "- **No login, OAuth, account or model task was attempted.** `{harness}` installed through the shared "
    "maintenance service and its selected executable answered its version directly and through doctor.",
    "- The doctor's overall status is reported; only its installed check must pass without credentials.",
    "- Only `{harness}` was installed; no other vendor program was attempted.",
)


def emit_summary(facts: Dict[str, Any], verdict: str, detail: str,
                 harness: Optional[str] = None) -> None:
    title = f"CLI toolchain + `{harness}` local install" if harness else "CLI toolchain"
    limits = [line.format(harness=harness) for line in HARNESS_LIMITS] if harness else LIMITS
    lines = [
        f"## Claudexor {title} — `{sys.platform}`",
        "",
        f"**{verdict}** — {detail}",
        "",
        "| fact | value |",
        "| --- | --- |",
        *(f"| `{key}` | {json.dumps(value) if isinstance(value, (dict, list)) else value} |"
          for key, value in sorted(facts.items())),
        "",
        *limits,
        "",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    target = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if target:
        try:
            with open(target, "a", encoding="utf-8") as stream:
                stream.write(text + "\n")
        except OSError as exc:  # reporting never masks the verdict
            print(f"[toolchain] could not write GITHUB_STEP_SUMMARY: {exc}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path, default=None,
                        help="empty or absent disposable directory (default: a new temp dir)")
    parser.add_argument("--harness-install", metavar="HARNESS", default=None,
                        help="then install this vendor program through maintenance (vendor network) "
                             "and prove its resolution and doctor; no login or model task")
    args = parser.parse_args(argv)
    harness = args.harness_install
    root = args.root or pathlib.Path(tempfile.mkdtemp(prefix="cx-toolchain-"))
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if any(root.iterdir()):
        # A pre-seeded cache or install would turn "downloaded and promoted" into a claim.
        parser.error(f"--root must be empty: {root}")
    dropped = isolate_environment(root)
    try:
        facts = run_witness(root, dropped, harness)
    except WitnessFailure as exc:
        emit_summary({"refusal_code": exc.code}, "FAILED", f"`{exc.code}` — {exc}", harness)
        return 1
    except Exception as exc:  # unexpected: still loud, still named (a runtime error keeps its code)
        code = str(getattr(exc, "code", "") or "unexpected_error")
        emit_summary({"refusal_code": code}, "FAILED", f"`{code}` — {type(exc).__name__}: {exc}",
                     harness)
        raise
    detail = ("the pinned managed Node/npm toolchain was installed from official archives and "
              "exercised with no ambient Node or npm")
    if harness:
        detail += f"; the vendor `{harness}` installed through maintenance, resolves and passes doctor's install check"
    emit_summary(facts, "PASSED", detail, harness)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
