"""One-shot, credential-free qualification of the workflow's exact product parent.

The native safe_test and pytest CI receipts stay private. The public projection
contains bounded lane verdicts, not service artifacts, commands or report bodies.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from stat import S_ISREG
import subprocess
import sys
import time
from xml.etree import ElementTree


BROWSER_FILE = "tests/test_verification_check_row_browser.py"
BROWSER = f"{BROWSER_FILE}::test_unsaved_service_log_row_live_finished_replayed_and_in_logs"
ENGINES = ("chromium", "webkit")
LANES = ("services", "structure", "size", "node", "docker", *ENGINES)
SERVICE_MODULES = tuple(f"tests/test_{name}.py" for name in (
    "workspace_executor_services", "workspace_executor_docker", "services_tool_v2",
    "builtin_effective_outcomes", "kept_service_release", "declared_output_early_answer",
    "phase3c_observability_gc", "child_drive_settlement", "service_finalization_failure",
    "launch_stop_finalization", "delivery_forced_finalization", "docker_process_cwd", "docker_declared_outputs",
    "file_browser_api", "send_file", "send_photo", "send_video", "published_artifact_identity", "persistence_inventory",
))
STRUCTURE = tuple(f"tests/test_{name}.py" for name in (
    "domain_manifest", "domain_report", "architecture_facts", "generated_inventories",
    "docs_sync", "module_handle_extraction", "loop_owner_facades", "persistence_lifetimes",
))
NODE_TESTS = ("web/tests/host_lifecycle_rows.test.js", "web/tests/chat_activity_block.test.js")
SIZE_REQUIRED = (
    "tests/test_smoke.py::test_size_ratchet_transition_against_explicit_base",
    "tests/test_reference_book_budgets.py::test_a_change_does_not_grow_a_reference_book",
    "tests/test_reference_book_budgets.py::test_a_change_does_not_grow_bible_or_the_prompts",
)
DEFAULT_MARKER = "not (integration or browser or ui_browser or ui_browser_docker or portable_detail or skill_smoke or size_ratchet)"
PYTEST_FLAGS = ("--disable-plugin-autoload", "-p", "xdist.plugin", "-p", "pytest_timeout",
                "--timeout=300", "-q", "--tb=no", "-c", "pyproject.toml", "-o", "addopts=")
PYTEST_EXIT_REASON = {1: "pytest_failed", 2: "pytest_interrupted", 3: "pytest_internal_error",
                      4: "pytest_usage_error", 5: "pytest_no_tests"}
LIMIT_SEC = dict(services=1800, structure=1200, size=1000, node=240, docker=300, chromium=850, webkit=850)
DOCKER_STAGES = ("container_started", "first_exited", "first_capture_bound_to_launch", "second_stopped_and_saved")
HEX40 = re.compile(r"[0-9a-f]{40}")
PNG_LIMIT = 2 * 1024 * 1024
PROJECTION_LIMIT = 8 * 1024 * 1024


class EvidenceGap(Exception):
    """A fixed safe reason; the caller supplies no raw diagnostics."""


def write_json(path: Path, value: dict) -> None:
    draft = path.with_suffix(".tmp")
    draft.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    draft.replace(path)


def read_bytes(path: Path, limit: int) -> bytes:
    if path.resolve() != path or not S_ISREG(path.lstat().st_mode):
        raise ValueError("expected ordinary owned file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("evidence file exceeds its bound")
    return data


def read_json(path: Path) -> dict:
    value = json.loads(read_bytes(path, PROJECTION_LIMIT))
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return value


def test_docker_restart(tmp_path, monkeypatch):
    """A short same-name service exits; replacement captures its output before a new writer."""
    from ouroboros import service_finalization
    from ouroboros.artifacts import registered_task_artifact
    from ouroboros.tools.registry import ToolContext
    from tests._launch_stop_shared import _exit_context, _registry

    task_id, work = "docker-service-qa", tmp_path / "project"
    work.mkdir()
    registry = _registry(tmp_path, monkeypatch, task_id=task_id)
    repo, drive = tmp_path / "repo", tmp_path / "data"
    drive.mkdir(exist_ok=True)
    witness = Path(os.environ["QA_DOCKER_WITNESS"])
    container = subprocess.check_output([
        "docker", "run", "--detach", "--rm", "--init", "--pull=never", "--network=none",
        "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop=ALL", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777", "--security-opt", "no-new-privileges",
        "--mount", f"type=bind,source={work},target=/work", os.environ["QA_SERVICE_IMAGE"],
        "sleep", "420",
    ], stderr=subprocess.PIPE, text=True, timeout=45).strip()
    assert re.fullmatch(r"[0-9a-f]{64}", container), "container_identity_unconfirmed"
    trace = {}
    ctx = ToolContext(repo_dir=repo, system_repo_dir=repo, drive_root=drive, workspace_mode="external",
                      workspace_root=work, task_id=task_id, _execution_trace=trace,
                      executor_ref={"type": "docker_exec", "container_name": container, "network": "none",
                                    "workspace_host_path": str(work), "workspace_backend_path": "/work"})
    registry.set_context(ctx)

    def call(tool, **args):
        result = registry.execute_result(tool, {"name": "paper", **args})
        assert result.code == "OK" and result.status == "ok", f"{tool}_receipt_not_ok"
        return result, json.loads(result.text)

    def output_is(expected):
        record = registered_task_artifact(drive, task_id, "report.txt")
        assert record and record.get("sha256") == sha256(expected).hexdigest(), "report_registration_mismatch"
        path = Path(record["path"]).resolve()
        assert path.is_relative_to(drive.resolve()) and path.name == "report.txt", "report_path_mismatch"
        assert read_bytes(path, 128) == expected, "registered_report_bytes_mismatch"

    def report(command, marker):
        return ["sh", "-c", command, "paper", marker]

    script = 'test "$PWD" = /work && printf "%s\\n" "$1" > report.txt && printf "%s\\n" "$1"'
    try:
        write_json(witness, {"last_confirmed": DOCKER_STAGES[0]})
        _, first = call("start_service", cwd=str(work), outputs=["report.txt"], cmd=report(script, "FIRST_PREVIEW"))
        assert first["executor"]["container_name"] == container and first["backend_cwd"] == "/work"
        assert first["launch_id"] and first["state"] in {"running", "exited"}
        deadline = time.monotonic() + 30
        while True:
            _, observed = call("service_status")
            assert observed["launch_id"] == first["launch_id"]
            if observed["state"] == "exited":
                break
            assert observed["state"] == "running" and time.monotonic() < deadline, "first_exit_unconfirmed"
            time.sleep(0.08)  # Poll the state; do not guess when the writer has finished.
        assert read_bytes(work / "report.txt", 128) == b"FIRST_PREVIEW\n"
        _, first_log = call("service_logs", tail=128)
        assert first_log["tail"].splitlines() == ["FIRST_PREVIEW"], "first_service_log_mismatch"
        assert registered_task_artifact(drive, task_id, "report.txt") is None
        write_json(witness, {"last_confirmed": DOCKER_STAGES[1]})

        _, second = call("start_service", cwd=str(work), outputs=["report.txt"],
                         readiness={"log_contains": "SECOND_PREVIEW", "timeout_sec": 20},
                         cmd=report(script + " && exec sleep 180", "SECOND_PREVIEW"))
        assert second["executor"]["container_name"] == container and second["backend_cwd"] == "/work"
        assert second["state"] == "running" and second["ready"] is True
        assert second["service_id"] == first["service_id"] and second["launch_id"] != first["launch_id"]
        _, second_log = call("service_logs", tail=128)
        assert second_log["tail"].splitlines() == ["SECOND_PREVIEW"], "second_service_log_mismatch"
        assert read_bytes(work / "report.txt", 128) == b"SECOND_PREVIEW\n"
        output_is(b"FIRST_PREVIEW\n")
        write_json(witness, {"last_confirmed": DOCKER_STAGES[2]})

        result, stopped = call("stop_service")
        assert stopped["launch_id"] == second["launch_id"] and stopped["state"] == "stopped"
        assert stopped["cleanup_dispatched"] is True and stopped["artifact_output_failed"] is False
        assert result.meta.get("artifact_registered") is True
        output_is(b"SECOND_PREVIEW\n")

        completion = _exit_context(drive, ctx)
        assert service_finalization.finalize(completion, trace), "task_completion_did_not_observe_stops"
        assert not service_finalization.finalize(completion, trace), "duplicate_stop_observation"
        events = service_finalization.evidence(trace)
        assert all(event["kind"] == "services_stopped" for event in events)
        rows = [row for event in events for row in event["services"]]
        assert [(row["launch_id"], row["stop_source"]) for row in rows] == [
            (first["launch_id"], "replacement"), (second["launch_id"], "explicit")]
        for row, content in zip(rows, (b"FIRST_PREVIEW\n", b"SECOND_PREVIEW\n"), strict=True):
            assert row["lifecycle"] == "stopped" and row["log_outcome"] == "backend_only"
            [output] = row["output_outcomes"]
            assert output["index"] == 1 and output["status"] == "saved" and output["registered"] is True
            [artifact] = output["artifacts"]
            assert (artifact["name"], artifact["sha256"]) == ("report.txt", sha256(content).hexdigest())
            # Read the version named by this launch after both writers have stopped, like its consumer.
            ref = artifact["version_ref"]
            assert ref["root"] == "artifact_store" and ref["sha256"] == sha256(content).hexdigest()
            actual = registered_task_artifact(drive, task_id, ref["path"])
            assert actual and actual["immutable"] is True and actual["sha256"] == ref["sha256"]
            assert read_bytes(Path(actual["path"]), 128) == content, "launch_version_bytes_mismatch"
            delivered = registry.execute_result("read_file", {"root": "artifact_store", "path": ref["path"]})
            assert delivered.status == "ok" and content.decode().strip() in delivered.text, "launch_version_unreadable"
        write_json(witness, {"last_confirmed": DOCKER_STAGES[3]})
    finally:
        # Only the container returned by this run; no host process scan or global cleanup.
        subprocess.run(["docker", "rm", "--force", container], check=True, capture_output=True, timeout=30)


def child_env(work: Path, node: Path) -> dict[str, str]:
    # No GITHUB_*, inherited PATH, tokens, home, test options, Docker config, or runner internals.
    return {
        "PATH": f"{Path(sys.executable).parent}:{node.parent}:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(work), "OUROBOROS_DATA_DIR": str(work / "data"),
        "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
        "no_proxy": "127.0.0.1,localhost,::1", "GIT_OPTIONAL_LOCKS": "0",
    }


def git(env, *args) -> bytes:
    return subprocess.check_output(["git", *args], env=env, stderr=subprocess.DEVNULL, timeout=20)


def check_checkout(env) -> None:
    expected = (os.environ["QA_PRODUCT"], os.environ["QA_TREE"], os.environ["QA_SIZE_BASE"])
    if not all(HEX40.fullmatch(v) for v in expected):
        raise EvidenceGap("subject_input_invalid")
    actual = tuple(git(env, "rev-parse", arg).strip().decode("ascii") for arg in (
        "HEAD", "HEAD^{tree}", f"{expected[2]}^{{commit}}"))
    if actual != expected or git(env, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise EvidenceGap("product_checkout_mismatch")


def select_modules(env):
    changed = git(env, "diff", "--name-only", "--no-renames", "--diff-filter=ACMT", "-z",
                  os.environ["QA_SIZE_BASE"], os.environ["QA_PRODUCT"], "--", "tests", "web/tests")
    python, node = [], []
    for item in sorted(os.fsdecode(part) for part in changed.split(b"\0") if part):
        if re.fullmatch(r"tests/(?:[A-Za-z0-9_]+/)*test_[A-Za-z0-9_]+\.py", item) and item not in (*STRUCTURE, BROWSER_FILE):
            python.append(item)
        elif re.fullmatch(r"web/tests/[A-Za-z0-9_.-]+\.test\.js", item):
            node.append(item)
    if len(python) + len(node) > 48:
        raise EvidenceGap("changed_test_scope_exceeds_one_off")
    return tuple(dict.fromkeys((*SERVICE_MODULES, *python))), tuple(dict.fromkeys((*NODE_TESTS, *node)))


def case_id(nodeid: str) -> str:
    # Parametrized values, collector paths and error values do not cross into the artifact.
    kind = nodeid.split("[", 1)[0].rsplit("::", 1)[-1]
    label = kind if re.fullmatch(r"test_[A-Za-z0-9_]{1,76}", kind) else "case"
    return label + "#" + sha256(nodeid.encode("utf-8", "replace")).hexdigest()[:12]


def failure_samples(rows):
    short = list(dict.fromkeys(rows))
    return {"samples": [{"case": case, "kind": reason} for case, reason in short[:8]],
            "samples_omitted": max(0, len(short) - 8)}


def pytest_result(projection: Path, lane: str, targets: tuple[str, ...]) -> dict:
    data = read_json(projection / "results.json")
    if data["format"] != 1:
        raise ValueError("unsupported pytest projection")
    selected = data.get("selected_nodeids", [])
    reports, collect_failures = data["reports"], data["collection_failures"]
    by_case, samples = defaultdict(list), []
    for row in reports:
        by_case[row["nodeid"]].append(row)
    counts, passed = Counter(), set()
    for nodeid in dict.fromkeys((*selected, *by_case)):
        phases = by_case[nodeid]
        bad = next((r for r in phases if r["outcome"] == "failed"), None)
        if bad:
            state = "failed"
            phase = bad["phase"] if bad["phase"] in {"setup", "call", "teardown", "crash"} else "unknown"
            cause = bad.get("error_type")
            failure = ("assertion" if cause == "AssertionError" else
                       "timeout" if cause == "TimeoutError" or bad.get("timeout_callback_seconds") else "error")
            samples.append((case_id(nodeid), f"{phase}_{failure}"))
        elif any(r.get("wasxfail") is not None for r in phases):
            state = "xfailed" if any(r["outcome"] == "skipped" for r in phases) else "xpassed"
        elif any(r["outcome"] == "skipped" for r in phases):
            # These two checks require a different OS. Neither skip counts as an executed assertion.
            inapplicable = lane == "services" and sys.platform == "linux" and (
                nodeid == "tests/test_services_tool_v2.py::test_windows_closed_read_waits_for_the_log_handle"
                or (Path("/proc").is_dir()
                    and nodeid == "tests/test_launch_stop.py::test_a_marker_free_census_reads_no_environment"))
            state = "platform_not_applicable" if inapplicable else "skipped"
        elif Counter(r["phase"] for r in phases if r["outcome"] == "passed" and not r.get("is_subtest")) == {
            "setup": 1, "call": 1, "teardown": 1,
        }:
            passed.add(nodeid)
            state = "passed"
        else:
            state = "not_completed"
        if not bad and state not in {"passed", "platform_not_applicable"}:
            samples.append((case_id(nodeid), state))
        counts[state] += 1
    samples.extend((case_id(row["nodeid"]), "collection_error") for row in collect_failures)
    collection_skips = data["collection_skips"]
    if lane != "size":
        samples.extend((case_id(node), "collection_skipped") for node in collection_skips)
    required = (set(SIZE_REQUIRED) if lane == "size" else set(targets) if lane in ENGINES else set())
    missing = required - passed
    if lane in {"services", "structure"}:
        missing |= set(targets) - {node.split("::", 1)[0] for node in passed}
    if lane == "docker":
        path, _, name = next(iter(passed), "").partition("::")
        if len(passed) != 1 or name != "test_docker_restart" or not path or (Path(__file__).parent / path).resolve() != Path(__file__).resolve():
            missing.add("test_docker_restart")
    samples.extend((case_id(node), "required_not_passed") for node in sorted(missing))
    expected_count = data["tests_collected"]
    if collect_failures:
        reason = "pytest_collection_failed"
    elif len(selected) != expected_count or len(set(selected)) != len(selected) or set(by_case) - set(selected):
        reason = "pytest_selection_mismatch"
    elif counts["failed"] or data["session_exit_code"] != 0:
        reason = PYTEST_EXIT_REASON.get(data["session_exit_code"], "pytest_failed")
    elif missing or (lane in ENGINES and selected != list(targets)):
        reason = "required_case_or_module_not_passed"
    elif sum(counts.values()) != counts["passed"] + counts["platform_not_applicable"] or (collection_skips and lane != "size"):
        reason = "pytest_cases_unsettled_or_skipped"
    else:
        reason = "ok" if counts["passed"] else "pytest_no_tests"
    return {"reason": reason, "pytest_exit": data["session_exit_code"], "collected": expected_count,
            "cases": dict(counts), "collection_failed": len(collect_failures), "collection_skipped": len(collection_skips),
            **failure_samples(samples)}


def node_result(report: Path) -> dict:
    cases = list(ElementTree.fromstring(read_bytes(report, PROJECTION_LIMIT)).iter("testcase"))
    counts, samples = Counter(), []
    for i, case in enumerate(cases, 1):
        state = ("failed" if case.find("failure") is not None or case.find("error") is not None else
                 "skipped" if case.find("skipped") is not None else "passed")
        counts[state] += 1
        if state != "passed":
            samples.append((f"node_case_{i}", state))
    reason = ("node_case_failed" if counts["failed"] else "node_case_skipped" if counts["skipped"] else
              "ok" if counts["passed"] else "node_no_cases")
    return {"reason": reason, "cases": dict(counts), **failure_samples(samples)}


def launcher_reason(work: Path, exit_code: int, docker_wrapped: bool) -> str:
    if exit_code == 124:
        return "launcher_deadline"
    if exit_code < 0 or 128 <= exit_code <= 192:
        return "process_interrupted"
    witnesses = list((work / "launcher").glob("run-*/result.json"))
    if len(witnesses) != 1:
        if docker_wrapped and exit_code in {125, 126, 127}:
            return "browser_container_launch_failed"
        return "launcher_receipt_unavailable"
    try:
        record = read_json(witnesses[0])
        if record["launch_error"] or record["execution"] != "executed":
            return "test_process_did_not_start"
        if record["containment_error"]:
            return "test_teardown_unconfirmed"
        if record["evidence_error"] or record["source_gap"]:
            return "launcher_evidence_incomplete"
        cleanup = record.get("cleanup")
        if not isinstance(cleanup, dict) or not all(isinstance(cleanup.get(key), list) for key in ("released", "retained")):
            return "launcher_cleanup_unconfirmed"
        if record["returncode"] != exit_code:
            return "launcher_exit_mismatch"
    except (OSError, ValueError, KeyError, RuntimeError):
        return "launcher_receipt_invalid"
    return "ok"


# Executed with the same mounts, uid and venv that will start safe_test. Its
# small receipt names only the last environment check reached, not exceptions.
BROWSER_PROBE = r'''
from importlib import import_module, metadata
from pathlib import Path
import os, subprocess, sys

cwd, venv, base, work, node, node_version, pw_version, *command = sys.argv[1:]
probe = Path(work) / 'browser_probe'

def require(check, reason):
    probe.write_text(reason, encoding='ascii')
    if not check:
        raise SystemExit(2)

require(os.geteuid() != 0 and Path.cwd() == Path(cwd), 'browser_cwd_or_uid')
require(sys.prefix == venv and sys.base_prefix == base and Path(venv, 'pyvenv.cfg').is_file(), 'browser_venv')
require(Path(cwd, 'scripts', 'safe_test.py').is_file() and Path('/ms-playwright').is_dir(), 'browser_inputs')
probe.write_text('browser_evidence_mount', encoding='ascii')
Path(os.environ['OUROBOROS_UI_EVIDENCE_DIR']).mkdir()
probe.write_text('browser_dependencies', encoding='ascii')
for name in ('pytest', 'xdist.plugin', 'pytest_timeout', 'playwright', 'PIL', 'uvicorn'):
    import_module(name)
require(metadata.version('playwright') == pw_version, 'browser_playwright_version')
probe.write_text('browser_pinned_node', encoding='ascii')
actual = subprocess.check_output([node, '--version'], timeout=20, text=True).strip()
require(actual == node_version, 'browser_pinned_node')
probe.write_text('ready', encoding='ascii')
os.execv(command[0], command)
'''
BROWSER_PROBE_STATES = {"browser_cwd_or_uid", "browser_venv", "browser_inputs", "browser_evidence_mount",
                        "browser_dependencies", "browser_playwright_version", "browser_pinned_node", "ready"}


SIZE_GUARD = r'''
from pathlib import Path
import os, sys

base, receipt, *command = sys.argv[1:]
if (os.environ.get('OURO_SIZE_RATCHET_BASE_REF') != base or 'GITHUB_EVENT_NAME' in os.environ
        or os.environ.get('OURO_BOOK_GROWTH_MODE', 'block') != 'block'):
    raise SystemExit(64)
Path(receipt).write_text(base, encoding='ascii')
os.execv(command[0], command)
'''


def browser_command(lane, work, node, node_version, command, env):
    checkout, base = Path.cwd(), Path(sys.base_prefix)
    temporary = work / "tmp"
    temporary.mkdir()
    image = os.environ["QA_BROWSER_IMAGE"]
    if not re.fullmatch(r"mcr\.microsoft\.com/playwright/python@sha256:[0-9a-f]{64}", image):
        raise EvidenceGap("browser_image_identity_invalid")
    args = ["docker", "run", "--rm", "--init", "--pull=never", "--network=none",
            "--user", f"{os.geteuid()}:{os.getegid()}", "--cap-drop=ALL", "--read-only",
            "--security-opt=no-new-privileges", "--shm-size=1g", "--workdir", str(checkout),
            "--entrypoint", sys.executable, "--tmpfs", "/var/tmp:rw,nosuid,nodev,mode=1777"]
    # .venv and Node are inside the readonly checkout; the .venv symlinks still
    # see the shipped interpreter at its exact absolute host path. The short
    # writable /tmp is a private disk bind; the evidence is a separate bind.
    for source, target, read_only in ((checkout, checkout, True), (base, base, True),
                                      (temporary, "/tmp", False), (work, work, False)):
        args += ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if read_only else "")]
    controls = {
        **env, "HOME": "/tmp", "PLAYWRIGHT_NODEJS_PATH": str(node), "PLAYWRIGHT_BROWSERS_PATH": "/ms-playwright",
        "OUROBOROS_RUN_UI_SMOKE": "1", "OUROBOROS_EXPECT_BROWSER_ENGINES": lane,
        "OUROBOROS_UI_EVIDENCE_DIR": str(work / "ui"),
    }
    for key, value in controls.items():
        args += ["--env", f"{key}={value}"]
    return [*args, image, "-I", "-c", BROWSER_PROBE, str(checkout), sys.prefix, str(base), str(work), str(node),
            node_version, os.environ["QA_PLAYWRIGHT_VERSION"], *command]


def stage_frames(engine: str, work: Path) -> dict:
    from io import BytesIO
    from PIL import Image

    verified = work / "verified-frames"
    verified.mkdir()
    frames = {}
    for stage in ("live", "reloaded", "logs"):
        name = f"{engine}-light-1280-{stage}.png"
        src = work / "ui" / "service-finalization" / name
        out = verified / name
        try:
            data = read_bytes(src, PNG_LIMIT)
            with Image.open(BytesIO(data), formats=["PNG"]) as image:
                if image.size != (1280, 900) or getattr(image, "n_frames", 1) != 1:
                    raise EvidenceGap(f"browser_png_shape_{stage}")
                image.verify()
            with Image.open(BytesIO(data), formats=["PNG"]) as image:
                rgb = image.convert("RGB")  # Fully decode; retain pixels only, never PNG text/EXIF metadata.
                rgb.info.clear()
                rgb.save(out, format="PNG")
            frames[name] = sha256(read_bytes(out, PNG_LIMIT)).hexdigest()
        except FileNotFoundError:
            raise EvidenceGap(f"browser_png_missing_{stage}") from None
        except (OSError, ValueError, SyntaxError, RuntimeError):
            raise EvidenceGap(f"browser_png_invalid_{stage}") from None
    return frames  # No file reaches the allowlist until all three validate.


def run_lane(lane, targets, work, node, node_version):
    work.mkdir()
    env = child_env(work, node)
    target = work / "pytest"
    browser = lane in ENGINES
    if lane == "node":
        if not node_version:
            raise EvidenceGap("pinned_node_not_executable")
        report = work / "node.xml"
        command = [str(node), "--test", "--test-reporter=junit", f"--test-reporter-destination={report}", *targets]
    else:
        marker = "size_ratchet" if lane == "size" else "ui_browser" if browser else DEFAULT_MARKER
        command = [sys.executable, "-m", "pytest", *PYTEST_FLAGS, "-m", marker, "--ci-evidence-dir", str(target)]
        if lane == "size":
            env["OURO_SIZE_RATCHET_BASE_REF"] = os.environ["QA_SIZE_BASE"]
        elif lane == "docker":
            # The temporary driver is outside tests/. Import its conftest once,
            # including its real fixture isolation and first-party CI projection.
            command += ["--rootdir", str(Path(__file__).resolve().parent), "--noconftest", "-p", "tests.conftest"]
            env.update(QA_SERVICE_IMAGE=os.environ["QA_SERVICE_IMAGE"], QA_DOCKER_WITNESS=str(work / "docker_witness.json"))
        command += targets
        if lane == "size":
            # Witness the environment AFTER safe_test replaces it, immediately before pytest.
            command = [sys.executable, "-I", "-S", "-c", SIZE_GUARD, env["OURO_SIZE_RATCHET_BASE_REF"],
                       str(work / "size-base"), *command]
    wrapped = [sys.executable, "-I", "-S", "scripts/safe_test.py", "--evidence-dir", str(work / "launcher"),
               "--timeout", str(LIMIT_SEC[lane]), "--", *command]
    if browser:
        if not node_version:
            raise EvidenceGap("pinned_node_not_executable")
        wrapped = browser_command(lane, work, node, node_version, wrapped, env)
    process = subprocess.run(wrapped, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT, check=False)
    result = {"exit_code": process.returncode}
    upstream = launcher_reason(work, process.returncode, browser)
    if browser:
        try:
            probe = read_bytes(work / "browser_probe", 64).decode("ascii")
        except (OSError, ValueError, RuntimeError):
            probe = ""
        result["browser_probe"] = probe if probe in BROWSER_PROBE_STATES else "unavailable"
        if probe in BROWSER_PROBE_STATES - {"ready"} and upstream == "launcher_receipt_unavailable":
            upstream = probe
    if lane == "docker":
        try:
            stage = read_json(work / "docker_witness.json")["last_confirmed"]
        except (OSError, ValueError, KeyError, RuntimeError):
            stage = ""
        result["docker_stage"] = stage if stage in DOCKER_STAGES else "unobserved"
    if lane == "size":
        expected = env["OURO_SIZE_RATCHET_BASE_REF"]
        try:
            confirmed = read_bytes(work / "size-base", 40).decode("ascii") == expected
        except (OSError, ValueError, RuntimeError):
            confirmed = False
        result["effective_size_base"] = expected if confirmed else None
        if not confirmed and upstream == "ok":
            upstream = "size_runtime_base_not_confirmed"
    try:
        result.update(node_result(report) if lane == "node" else pytest_result(target, lane, targets))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ElementTree.ParseError):
        result["reason"] = "case_projection_missing_or_invalid"
        if lane != "node" and process.returncode in PYTEST_EXIT_REASON:
            result["process_reason"] = PYTEST_EXIT_REASON[process.returncode]
    if upstream != "ok":
        result["case_reason"], result["reason"] = result["reason"], upstream
    elif process.returncode and result["reason"] == "ok":
        result["reason"] = "test_process_nonzero"
    if lane == "docker" and result["reason"] == "ok" and result["docker_stage"] != DOCKER_STAGES[-1]:
        result["reason"] = "docker_consumer_receipt_incomplete"
    if browser and result["reason"] == "ok":
        if result["browser_probe"] != "ready":
            result["reason"] = "browser_probe_unconfirmed"
        else:
            try:
                result["checked_frames"] = stage_frames(lane, work)
            except EvidenceGap as exc:
                result["reason"] = str(exc)
    result["state"] = "passed" if result["reason"] == "ok" else "failed"
    return result


def node_probe(node, env):
    if os.environ.get("QA_NODE_SETUP") != "success":
        return ""
    lock = Path("ouroboros/claudexor_runtime_pin.json")
    try:
        expected = "v" + read_json(lock.resolve())["release"]["node_version"]
        found = subprocess.check_output([str(node), "--version"], env=env, text=True,
                                        stderr=subprocess.DEVNULL, timeout=20).strip()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return ""
    return found if expected == found and re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", found) else ""


def main() -> int:
    if (sys.platform != "linux" or os.geteuid() == 0 or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "razzant/ouroboros"
            or os.environ.get("GITHUB_REF") != "refs/heads/qa/task-service-evidence"):
        raise SystemExit("Expected the unprivileged one-off Actions job.")
    checkout, temporary = Path.cwd(), Path(os.environ["RUNNER_TEMP"]).resolve()
    private, public = temporary / "service-private", temporary / "service-evidence"
    private.mkdir(mode=0o700)
    public.mkdir(mode=0o700)
    outcomes = {lane: {"state": "not_run", "reason": "pending"} for lane in LANES}
    record = {"driver": {"state": "running", "reason": "lanes_pending"}, "lanes": outcomes}

    def save():
        write_json(public / "results.json", record)

    save()
    node = checkout / "node-standalone" / "bin" / "node"
    env = child_env(private, node)
    try:
        if (Path(sys.executable).parent != checkout / ".venv" / "bin" or Path(sys.prefix) != checkout / ".venv"
                or Path(sys.base_prefix).resolve() != temporary / "shipped-python"
                or not Path(sys.prefix, "pyvenv.cfg").is_file()):
            raise EvidenceGap("shipped_venv_not_bound")
        check_checkout(env)
        services, node_targets = select_modules(env)
        fixture = os.fspath(Path(__file__).resolve()) + "::test_docker_restart"
        batches = {"services": services, "structure": STRUCTURE, "size": ("tests/",), "node": node_targets,
                   "docker": (fixture,), **{engine: (f"{BROWSER}[{engine}]",) for engine in ENGINES}}
        node_version = node_probe(node, env)
        record["driver"].update(node_version=node_version or None, python=sys.version.split()[0],
                                pytest=version("pytest"), pytest_timeout=version("pytest-timeout"), xdist=version("pytest-xdist"))
    except EvidenceGap as exc:
        record["driver"].update(state="failed", reason=str(exc))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        record["driver"].update(state="failed", reason="subject_or_dependency_probe_failed")
    if record["driver"]["state"] == "failed":
        for lane in LANES:
            outcomes[lane]["reason"] = record["driver"]["reason"]
        save()
        return 1

    for lane, targets in batches.items():
        setup = {"node": ("QA_NODE_SETUP",), "docker": ("QA_SERVICE_SETUP",),
                 **{engine: ("QA_NODE_SETUP", "QA_BROWSER_SETUP") for engine in ENGINES}}.get(lane, ())
        missing = [key.removeprefix("QA_").lower() for key in setup if os.environ.get(key) != "success"]
        if missing:
            outcomes[lane] = {"state": "not_run", "reason": "prerequisite_unavailable", "prerequisites": missing}
        else:
            outcomes[lane] = {"state": "running", "reason": "test_in_progress"}
            save()
            print(f"QA_LANE_STARTED {lane}", flush=True)
            try:
                check_checkout(env)
                outcomes[lane] = run_lane(lane, targets, private / lane, node, node_version)
            except EvidenceGap as exc:
                outcomes[lane] = {"state": "failed", "reason": str(exc)}
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                outcomes[lane] = {"state": "failed", "reason": "lane_infrastructure_error"}
        if lane in {"services", "structure", "node"}:
            outcomes[lane]["selected_modules"] = list(targets)
        save()

    try:
        check_checkout(env)
    except (EvidenceGap, OSError, ValueError, subprocess.SubprocessError):
        record["driver"].update(state="failed", reason="product_checkout_changed_or_unreadable")
        for engine in ENGINES:
            if outcomes[engine]["state"] == "passed":
                outcomes[engine].update(state="incomplete", reason="product_identity_not_confirmed_after_run")
    else:
        record["driver"].update(state="completed", reason="all_lanes_reported")
        for engine in ENGINES:
            row = outcomes[engine]
            if row["state"] == "passed":
                try:
                    # Atomic on the same runner-temp filesystem, once per engine.
                    (private / engine / "verified-frames").replace(public / engine)
                    row["exported_frames"] = row["checked_frames"]
                except OSError:
                    row.update(state="failed", reason="browser_export_failed")
    save()
    for lane, row in outcomes.items():
        print(f"QA_LANE {lane} {row['state']} {row['reason']}", flush=True)
    return int(record["driver"]["state"] != "completed" or any(row["state"] != "passed" for row in outcomes.values()))


if __name__ == "__main__":
    raise SystemExit(main())
