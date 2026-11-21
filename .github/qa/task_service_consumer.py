"""Temporary, credential-free consumer gate for one task-service candidate (#1868).

Started only by .github/workflows/task-service-consumer-gate.yml, from a copy of this blob outside the checkout:
    python -I task_service_consumer.py lane <name>   inside the locked venv (Pillow re-encodes the synthetic frames)
    python -I -S task_service_consumer.py report     standard library only: it must run when setup failed

Every test command is a child of scripts/safe_test.py (scrubbed roots, owned reap) and its streams go to a private
file: the artifact carries only bounded facts: ids, counts, fixed reasons, private-frame hashes and
a structured failure projection. It does not carry stdout, tracebacks or image bytes. A skip is never
a pass; exact named platform skips are reported as not_applicable.
`test_docker_restart` is the real-Docker witness pytest collects from this file; its product imports are local.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import sqlite3
from stat import S_ISREG
import subprocess
import sys
import time
from xml.etree import ElementTree

DRIVER = Path(__file__).name
DOCKER_ID = f"{DRIVER}::test_docker_restart"
DOCKER_STAGES = ("container_started", "first_exited", "first_capture_bound_to_launch", "second_stopped_and_saved")
DEFAULT_MARKER = ("not (integration or browser or ui_browser or ui_browser_docker or portable_detail "
                  "or skill_smoke or size_ratchet)")
PYTEST_FLAGS = ("--disable-plugin-autoload", "-p", "xdist.plugin", "-p", "pytest_timeout",
                "--timeout=300", "-q", "-c", "pyproject.toml", "-o", "addopts=")
ID = re.compile(r"(?:tests|web/tests)/[A-Za-z0-9_./-]+\.(?:py|test\.js)(?:::[A-Za-z0-9_]+(?:\[[A-Za-z0-9_.,:+ -]+\])?)?")
DRIVER_ID = re.compile(r"[A-Za-z0-9_]+\.py::test_[A-Za-z0-9_]+")
PNG_LIMIT = 2 * 1024 * 1024
STAGES = ("live", "reloaded", "logs")
# A new consumer case is UNKNOWN until its exact id (R6 commit + id) is written into the workflow table:
# (slot, workflow variable, systems it must pass on, owning lane).
SLOTS = (
    ("worker_change_same_report", "P_WORKER_CHANGE", ("linux", "windows"), "consumers"),
    ("durable_row_after_failed_delivery", "P_DURABLE_ROW", ("linux", "windows"), "consumers"),
    ("docker_per_output_normal", "P_DOCKER_PER_OUTPUT", ("linux",), "docker"),
    ("private_service_log_ordinary_paths", "P_PRIVATE_LOG_PATHS", ("linux", "windows"), "consumers"),
    ("task_summary_guard", "P_TASK_SUMMARY_GUARD", ("linux", "windows"), "consumers"),
)
# lane: (marker, --tb, seconds, diagnostic, workflow variable of ids whose failure is a known, non-gate message)
PYTEST_LANES = {
    "consumers": (DEFAULT_MARKER, "no", 2400, False, ""),
    "structure": (DEFAULT_MARKER, "no", 1200, False, "K_STRUCTURE"),
    "browser": ("ui_browser", "no", 1500, False, ""),
    "docker": (DEFAULT_MARKER, "no", 300, False, ""),
    "serial_tb": (DEFAULT_MARKER, "long", 3000, True, ""),
    "parity": (DEFAULT_MARKER, "long", 3000, True, ""),
    "known": (DEFAULT_MARKER, "long", 1500, True, ""),
}
LANES = (*PYTEST_LANES, "size", "node")
DIAGNOSTIC = {name for name, spec in PYTEST_LANES.items() if spec[3]}
SCOPE = {
    "gate": "narrow consumer gate over the fixed witness table; module-wide, serial and full-suite health are NOT asserted",
    "diagnostic": "DIAGNOSTIC evidence, never a gate: complete means the evidence is complete, not that tests passed; "
                  "no health is inferred from an earlier CI aggregate",
}
# What a test child must not see: the run's own identity (the size rule reads GITHUB_EVENT_NAME and silently skips
# on a push), runner internals, the witness table itself and anything token-shaped.
DROPPED = re.compile(r"(?:GITHUB_|ACTIONS_|RUNNER_|INPUT_|AGENT_|IMAGEOS|IMAGEVERSION).*|CI|QA_(?:SETUP|R6_[A-Z_]+|CONTROL_[A-Z_]+)"
                     r"|(?:W|P|NA|D|K)_[A-Z0-9_]+|.*_(?:TOKEN|SECRET|PASSWORD|API_KEY)")


class Gap(Exception):
    """A fixed safe reason; no raw diagnostics ever ride on it."""


def clean(text, limit: int = 240) -> str:
    return re.sub(r"[^\x20-\x7e]", "?", str(text))[:limit]


def read_bytes(path: Path, limit: int) -> bytes:
    if not S_ISREG(path.lstat().st_mode):
        raise ValueError("expected an ordinary file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("evidence file exceeds its bound")
    return data


def read_json(path: Path) -> dict:
    value = json.loads(read_bytes(path, 8 * 1024 * 1024))
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    draft = path.with_suffix(path.suffix + ".tmp")
    draft.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    draft.replace(path)


def priv() -> Path:
    return Path(os.environ["RUNNER_TEMP"]) / "qa-private"


def pub() -> Path:
    return Path(os.environ["RUNNER_TEMP"]) / "qa-public"


def system() -> str:
    return "windows" if sys.platform == "win32" else "linux"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, timeout=60)


def check_checkout(subject: dict) -> None:
    """The exact subject, nothing else: HEAD, its tree and a clean tree, before and after every lane."""
    head, tree = (git("rev-parse", arg).decode().strip() for arg in ("HEAD", "HEAD^{tree}"))
    if (head, tree) != (subject["subject_commit"], subject["subject_tree"]) or git(
            "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise Gap("checkout_changed")


def bound_interpreter() -> None:
    """The evidence is for the shipped interpreter in this checkout's locked venv, not whatever `python` resolved to."""
    def norm(path) -> str:
        return os.path.normcase(os.path.realpath(path))

    if (norm(sys.prefix) != norm(Path.cwd() / ".venv") or not Path(sys.prefix, "pyvenv.cfg").is_file()
            or norm(sys.base_prefix) != norm(Path(os.environ["RUNNER_TEMP"]) / "shipped-python")):
        raise Gap("shipped_venv_not_bound")


def child_env(work: Path, **extra: str) -> dict[str, str]:
    """The runner's environment less what names this run; no network by proxy; the pinned Node first on PATH."""
    env = {key: value for key, value in os.environ.items() if not DROPPED.fullmatch(key.upper())}
    env.update(HOME=str(work), USERPROFILE=str(work), GIT_OPTIONAL_LOCKS="0", http_proxy="http://127.0.0.1:9",
               https_proxy="http://127.0.0.1:9", no_proxy="127.0.0.1,localhost,::1")
    node = Path.cwd() / "node-standalone" / "bin"
    if node.is_dir():
        env["PATH"] = os.pathsep.join((str(node), env.get("PATH", "")))
    env.update(extra)
    return env


# ---- the fixed witness table: ids come from the workflow's env, never from the diff -----------------------------

def ids(name: str) -> list[str]:
    rows = [row.strip() for row in os.environ.get(name, "").splitlines()]
    rows = [row for row in rows if row and not row.startswith("#")]
    if len(set(rows)) != len(rows) or not all(
            ID.fullmatch(row) or (DRIVER_ID.fullmatch(row) and row.startswith(DRIVER + "::")) for row in rows):
        raise Gap("witness_table_invalid")
    return rows


def slot_cases(env: str, lane: str) -> list[str]:
    wanted = ids(env)
    if lane == "docker":
        if any(not w.startswith(DRIVER + "::") or w == DOCKER_ID for w in wanted):
            raise Gap("docker_slot_requires_its_own_real_witness")
    elif any(not w.startswith("tests/") for w in wanted):
        raise Gap("consumer_slot_not_a_product_test")
    return wanted


def slot_ids(lane: str) -> list[str]:
    return [w for _slot, env, systems, owner_lane in SLOTS
            if owner_lane == lane and system() in systems for w in slot_cases(env, lane)]


def lane_ids(lane: str) -> list[str]:
    here = system().upper()
    table = {
        "consumers": lambda: [*ids("W_BOTH"), *ids("W_" + here), *slot_ids("consumers")],
        "docker": lambda: [DOCKER_ID, *slot_ids("docker")],
        "structure": lambda: ids("W_STRUCTURE"), "browser": lambda: ids("W_BROWSER"), "node": lambda: ids("W_NODE"),
        "size": lambda: ids("W_SIZE_REQUIRED"), "serial_tb": lambda: ids("D_SERIAL_TB"),
        "parity": lambda: ids("D_PARITY"), "known": lambda: ids("D_KNOWN"),
    }
    found = table[lane]()
    if not found or len(set(found)) != len(found):
        raise Gap("witness_table_invalid")
    return found


# ---- judging a pytest session by its first-party projection (tests/ci_evidence.py results.json) ------------------

def hit(node: str, want: str) -> bool:
    return node == want or node.startswith((want + "::", want + "["))


PUBLIC_NODE = re.compile(r"(?:tests/[A-Za-z0-9_./-]+\.py|task_service_consumer\.py)"
                         r"(?:::[A-Za-z_][A-Za-z0-9_]*)+(?:\[[A-Za-z0-9_.,:+ -]+\])?")
ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
PHASES = {"setup", "call", "teardown"}
OUTCOMES = {"passed", "failed", "skipped"}


def public_id(node: str) -> str:
    # Keep only recognizable code identifiers. Never reflect unexpected reprs or arbitrary parameter text.
    return node if len(node) <= 240 and PUBLIC_NODE.fullmatch(node) else "<unpublished-node-id>"


def error_type(row: dict) -> str:
    error = row.get("error_type")
    return error if isinstance(error, str) and ERROR_TYPE.fullmatch(error) else ""


def case_state(node: str, rows: list[dict], na: list[str]) -> str:
    if any(row["phase"] == "crash" for row in rows):
        return "crashed"
    primary = [row for row in rows if not row["is_subtest"]]
    phases = {row["phase"]: row["outcome"] for row in primary}
    if len(primary) != len(phases) or not {"setup", "teardown"} <= phases.keys():
        return "not_completed"
    setup, call, teardown = phases["setup"], phases.get("call"), phases["teardown"]
    if (setup == "passed") != (call is not None) or (len(rows) != len(primary) and call is None):
        return "not_completed"
    if any(row.get("wasxfail") is not None for row in rows):
        return "xfailed" if any(row["outcome"] == "skipped" for row in rows) else "xpassed"
    if node in na and teardown == "passed" and len(rows) == len(primary) and (
            (setup == "skipped" and call is None) or (setup == "passed" and call == "skipped")):
        return "not_applicable"
    if setup != "passed" or any(row["outcome"] == "skipped" for row in rows):
        return "not_completed" if setup == "failed" else "skipped"
    return "failed" if any(row["outcome"] == "failed" for row in rows) else "passed"


def judge(proj: Path, wanted: list[str], known: list[str], *, diagnostic: bool, na: list[str] = ()) -> dict:
    """A diagnostic can report failed assertions, not a collection/abort or merely a list of selected nodes."""
    try:
        data = read_json(proj / "results.json")
        if type(data["format"]) is not int or data["format"] != 1:
            raise Gap("projection_format_unsupported")
        selected, exit_code, collected = data["selected_nodeids"], data["session_exit_code"], data["tests_collected"]
        reports = data["reports"]
        if (type(exit_code) is not int or type(collected) is not int or collected < 0
                or not isinstance(selected, list) or not all(isinstance(n, str) and n for n in selected)
                or not all(isinstance(data[k], list) for k in ("collection_failures", "collection_skips", "reports"))):
            raise Gap("case_projection_missing_or_invalid")
        by_case = defaultdict(list)
        failures = []
        for row in reports:
            if (not isinstance(row, dict) or not isinstance(row.get("nodeid"), str) or not row["nodeid"]
                    or row.get("phase") not in PHASES | {"crash"} or row.get("outcome") not in OUTCOMES
                    or type(row.get("is_subtest")) is not bool or (row["is_subtest"] and row["phase"] != "call")
                    or (row["outcome"] == "skipped" and "wasxfail" not in row)
                    or (row.get("wasxfail") is not None and not isinstance(row["wasxfail"], str))):
                raise Gap("case_projection_missing_or_invalid")
            n, phase = row["nodeid"], row["phase"]
            by_case[n].append(row)
            if row["outcome"] == "failed":
                failures.append((n, phase, error_type(row)))
        states = {n: case_state(n, by_case.get(n, []), na) for n in selected}
        collection_failed, collection_skipped = len(data["collection_failures"]), len(data["collection_skips"])
    except (OSError, ValueError, KeyError, TypeError):
        raise Gap("case_projection_missing_or_invalid") from None
    counts = Counter(states.values())
    by_id = {}
    for want in wanted:
        mine = [states[n] for n in states if hit(n, want)]
        state = next((s for s in ("crashed", "not_completed", "failed", "skipped", "xfailed", "xpassed") if s in mine), "")
        by_id[want] = {"nodes": len(mine), "passed": mine.count("passed"), "not_applicable": mine.count("not_applicable"),
                       "state": "not_selected" if not mine else state or
                       ("passed" if "passed" in mine else "not_applicable")}
    unmatched = [w for w in wanted if not by_id[w]["nodes"]]
    extra = [n for n in selected if not any(hit(n, w) for w in wanted)]
    complete_reason = ("pytest_collection_failed" if collection_failed else
                       "pytest_collection_skipped" if collection_skipped else
                       "pytest_aborted" if exit_code not in (0, 1) or data.get("worker_crash") or counts["crashed"] else
                       "pytest_selection_empty" if not selected else
                       "pytest_selection_mismatch" if len(selected) != collected or len(states) != len(selected)
                       or not set(by_case) <= set(states) else
                       "required_id_not_selected" if unmatched else
                       "unexpected_node_selected" if extra else
                       "case_not_completed" if counts["not_completed"] else
                       "no_assertions_reported" if not (counts["passed"] or counts["failed"]) else
                       "case_skipped_or_unsettled" if counts["skipped"] or counts["xfailed"] or counts["xpassed"] else
                       "pytest_exit_mismatch" if exit_code != int(bool(failures)) else "ok")
    if diagnostic:
        state, reason = ("reported" if complete_reason == "ok" else "incomplete"), complete_reason
    else:
        gate_failed = any(not any(hit(n, k) for k in known) for n, _phase, _error in failures)
        unproved_id = any(row["state"] != "passed" and not (
                              row["state"] == "failed" and any(hit(w, k) or hit(k, w) for k in known))
                          for w, row in by_id.items())
        reason = complete_reason if complete_reason != "ok" else (
            "required_case_failed" if gate_failed else "required_id_not_passed" if unproved_id else "ok")
        state = "passed" if reason == "ok" else "failed"
    passed = sorted(n for n in states if states[n] == "passed")
    return {
        "state": state, "complete": complete_reason == "ok", "reason": reason,
        "pytest_exit": exit_code, "selected": len(selected),
        "cases": dict(counts), "collection_failed": collection_failed, "collection_skipped": collection_skipped,
        "by_id": by_id, "unmatched": unmatched[:40], "extra": [public_id(n) for n in extra[:40]], "extra_total": len(extra),
        "failed": [{"id": public_id(n), "phase": p, "error": e} for n, p, e in failures[:100]], "failed_total": len(failures),
        "skipped": [public_id(n) for n in states if states[n] in ("skipped", "xfailed", "xpassed")][:100],
        "not_applicable": [public_id(n) for n in states if states[n] == "not_applicable"][:100],
        "known_non_gate": [{"id": public_id(n), "state": states[n]} for n in states if any(hit(n, k) for k in known)],
        "executed": [public_id(n) for n in passed] if len(passed) <= 400 else None, "executed_total": len(passed),
        "selected_sha256": sha256("\n".join(sorted(selected)).encode()).hexdigest(),
        "failed_sha256": sha256("\n".join(sorted("|".join(f) for f in failures)).encode()).hexdigest(),
    }


def junit(report: Path) -> dict:
    cases = list(ElementTree.fromstring(read_bytes(report, 8 * 1024 * 1024)).iter("testcase"))
    counts = Counter("failed" if c.find("failure") is not None or c.find("error") is not None else
                     "skipped" if c.find("skipped") is not None else "passed" for c in cases)
    reason = ("node_case_failed" if counts["failed"] else "node_case_skipped" if counts["skipped"] else
              "ok" if counts["passed"] else "node_no_cases")
    return {"state": "passed" if reason == "ok" else "failed", "reason": reason, "cases": dict(counts)}


def launcher_reason(raw: Path, code: int) -> str:
    """What scripts/safe_test.py's own receipt says about start, owned reap and evidence (its exit code is the command's)."""
    if code == 124:
        return "launcher_deadline"
    if code < 0 or 128 <= code <= 192:
        return "process_interrupted"
    found = list(raw.glob("run-*/result.json"))
    if len(found) != 1:
        return "launcher_receipt_unavailable"
    try:
        row = read_json(found[0])
        if row["launch_error"] or row["execution"] != "executed":
            return "test_process_did_not_start"
        if row["containment_error"]:
            return "test_teardown_unconfirmed"
        if row["evidence_error"] or row["source_gap"]:
            return "launcher_evidence_incomplete"
        cleanup = row.get("cleanup")
        if not isinstance(cleanup, dict) or not all(isinstance(cleanup.get(key), list) for key in ("released", "retained")):
            return "launcher_cleanup_unconfirmed"
        if type(row["returncode"]) is not int or row["returncode"] != code:
            return "launcher_exit_mismatch"
    except (OSError, ValueError, KeyError):
        return "launcher_receipt_invalid"
    return "ok"


# No stderr, fixture repr, statement or frame is eligible for public evidence. Only the test declaration
# confirmed in a blob of the bound subject tree is a location. It is not the failing traceback line.
def failure_extract(result: dict, subject: dict) -> dict:
    by_path = {}
    public = []
    for row in result["failed"][:80]:
        node = row["id"]
        if not PUBLIC_NODE.fullmatch(node):
            continue
        path, *scope = node.split("[", 1)[0].split("::")
        if path not in by_path:
            try:
                by_path[path] = ast.parse(git("show", f"{subject['subject_tree']}:{path}")).body
            except (SyntaxError, ValueError, OSError, subprocess.SubprocessError):
                by_path[path] = None
        body = by_path[path]
        if body is None:
            continue
        for cls in scope[:-1]:
            matches = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
            body = matches[0].body if len(matches) == 1 else []
        matches = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == scope[-1]]
        if len(matches) == 1:
            public.append({"id": node, "phase": row["phase"], "exception_type": row["error"],
                           "test_definition": {"file": path, "line": matches[0].lineno}})
    return {"subject_tree": subject["subject_tree"], "failures": public,
            "failures_omitted": result["failed_total"] - len(public)}


# ---- lanes ------------------------------------------------------------------------------------------------------

def safe_command(command: list[str], raw: Path, limit: int) -> list[str]:
    return [sys.executable, "-I", "-S", "scripts/safe_test.py", "--evidence-dir", str(raw), "--timeout", str(limit),
            "--", *command]


def run_private(command: list[str], env: dict, out: Path, limit: int) -> int:
    """Every stream goes to a private file; the job log never sees test output."""
    with out.open("wb") as stream:
        try:
            return subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                  check=False, timeout=limit).returncode
        except subprocess.TimeoutExpired:
            return 124


def frames(ui: Path, engines: list[str]) -> dict[str, bytes]:
    """Every synthetic PNG of every engine, decoded and re-encoded as pixels only (no text, EXIF or profile chunks);
    only digests of validated pixels can be published."""
    try:
        from PIL import Image
    except ImportError:
        raise Gap("png_decoder_unavailable") from None
    found = {}
    for engine in engines:
        for stage in STAGES:
            name = f"{engine}-light-1280-{stage}.png"
            try:
                data = read_bytes(ui / "service-finalization" / name, PNG_LIMIT)
            except (OSError, ValueError):
                raise Gap(f"browser_png_missing_{stage}") from None
            try:
                with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
                    if image.size != (1280, 900) or getattr(image, "n_frames", 1) != 1:
                        raise Gap(f"browser_png_shape_{stage}")
                    image.verify()
                with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
                    pixels = image.convert("RGB")
                pixels.info.clear()
                encoded = io.BytesIO()
                pixels.save(encoded, format="PNG")
                found[name] = encoded.getvalue()
            except (OSError, ValueError, SyntaxError, RuntimeError):
                raise Gap(f"browser_png_invalid_{stage}") from None
    return found


def docker_stages(folder: Path, wanted: list[str]) -> dict[str, str]:
    """One real-Docker receipt per exact QA test ID, never another test's stage."""
    stages = {}
    for node in wanted:
        name = node.rsplit("::", 1)[-1]
        try:
            stage = read_json(folder / f"{name}.json")["last_confirmed"]
        except (OSError, ValueError, KeyError):
            stage = ""
        stages[node] = stage if stage in DOCKER_STAGES else "unobserved"
    return stages


def run_pytest_lane(lane: str, subject: dict) -> dict:
    marker, tb, limit, diagnostic, known_env = PYTEST_LANES[lane]
    wanted = lane_ids(lane)
    work = priv() / lane
    work.mkdir(parents=True)
    proj, raw, out = work / "proj", work / "raw", work / "pytest.out"
    env, targets, engines = child_env(work), wanted, []
    command = [sys.executable, "-m", "pytest", *PYTEST_FLAGS, "-m", marker, f"--tb={tb}", "--ci-evidence-dir", str(proj)]
    if lane == "browser":
        engines = sorted({m.group(1) for i in wanted if (m := re.search(r"\[(\w+)\]$", i))}) or ["chromium", "webkit"]
        command.append("--timeout=900")  # the host UI lane's own per-test ceiling; this test waits up to 240 s on its model
        (work / "ui").mkdir()
        env.update(OUROBOROS_RUN_UI_SMOKE="1", OUROBOROS_EXPECT_BROWSER_ENGINES=",".join(engines),
                   PLAYWRIGHT_BROWSERS_PATH=str(Path(os.environ["RUNNER_TEMP"]) / "ui-browsers"),
                   PLAYWRIGHT_NODEJS_PATH=str(Path.cwd() / "node-standalone" / "bin" / "node"),
                   OUROBOROS_UI_EVIDENCE_DIR=str(work / "ui"))
    if lane == "docker":
        if not re.fullmatch(r"ubuntu:22\.04@sha256:[0-9a-f]{64}", os.environ.get("QA_SERVICE_IMAGE", "")):
            raise Gap("service_image_unpinned")
        qa = Path(__file__).resolve().parent
        # The driver sits outside tests/: import the product's conftest once, with its fixture isolation.
        command += ["--rootdir", str(qa), "--noconftest", "-p", "tests.conftest"]
        env.update(QA_SERVICE_IMAGE=os.environ["QA_SERVICE_IMAGE"], QA_DOCKER_WITNESS_DIR=str(work / "witnesses"))
        targets = [f"{qa / i.partition('::')[0]}::{i.partition('::')[2]}" if i.startswith(DRIVER + "::") else i
                   for i in wanted]
    code = run_private(safe_command([*command, *targets], raw, limit), env, out, limit + 120)
    upstream = launcher_reason(raw, code)
    failed = "incomplete" if diagnostic else "failed"
    try:
        result = judge(proj, wanted, ids(known_env) if known_env else [], diagnostic=diagnostic,
                       na=ids("NA_" + system().upper()))
    except Gap as gap:
        result = {"state": failed, "reason": str(gap)}
    result["process_exit"] = code
    if upstream != "ok":
        result["case_reason"], result["reason"], result["state"] = result["reason"], upstream, failed
    elif result.get("pytest_exit") is not None and code != result["pytest_exit"]:
        result["case_reason"], result["reason"], result["state"] = result["reason"], "process_exit_mismatch", failed
    if lane == "docker" and result["state"] == "passed":
        stages = docker_stages(work / "witnesses", wanted)
        result["docker_stages"] = stages
        if any(stage != DOCKER_STAGES[-1] for stage in stages.values()):
            result["state"], result["reason"] = "failed", "docker_consumer_receipt_incomplete"


    if lane == "browser" and result["state"] == "passed":
        try:
            shots = frames(work / "ui", engines)
        except Gap as gap:
            result["state"], result["reason"] = "failed", str(gap)
        else:
            # Dimensions, decodability and stripped metadata alone do not approve pixels for publication.
            (work / "verified-png").mkdir()
            for name, data in shots.items():
                (work / "verified-png" / name).write_bytes(data)
            result["private_png_sha256"] = {name: sha256(data).hexdigest() for name, data in shots.items()}
    return result


def size_lane(lane: str, subject: dict) -> dict:
    """Seven fixed reference-book/module gates against the product parent, not a tests/ collection.
    The child does not receive the run's push identity, which would cause the book rule to skip."""
    work = priv() / lane
    work.mkdir(parents=True)
    proj, raw, out = work / "proj", work / "raw", work / "pytest.out"
    env = child_env(work, OURO_SIZE_RATCHET_BASE_REF=subject["base"], OURO_BOOK_GROWTH_MODE="block")
    command = [sys.executable, "-m", "pytest", *PYTEST_FLAGS, "-m", "size_ratchet", "--tb=no",
               "--ci-evidence-dir", str(proj), *lane_ids(lane)]
    code = run_private(safe_command(command, raw, 780), env, out, 900)
    upstream = launcher_reason(raw, code)
    result = judge(proj, lane_ids(lane), [], diagnostic=False)
    result.update(process_exit=code, size_base=subject["base"], growth_mode="block")
    if upstream != "ok":
        result["case_reason"], result["reason"], result["state"] = result["reason"], upstream, "failed"
    elif code != result["pytest_exit"]:
        result["case_reason"], result["reason"], result["state"] = result["reason"], "process_exit_mismatch", "failed"
    return result


def node_lane(lane: str, subject: dict) -> dict:
    wanted, node = lane_ids(lane), Path.cwd() / "node-standalone" / "bin" / "node"
    if not node.is_file():
        raise Gap("pinned_node_missing")
    try:
        expected = "v" + read_json(Path("ouroboros/claudexor_runtime_pin.json"))["release"]["node_version"]
        found = subprocess.check_output([str(node), "--version"], text=True, stderr=subprocess.DEVNULL, timeout=20).strip()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        raise Gap("pinned_node_version_unconfirmed") from None
    if found != expected:
        raise Gap("pinned_node_version_mismatch")
    work = priv() / lane
    work.mkdir(parents=True)
    report, raw = work / "node.xml", work / "raw"
    command = [str(node), "--test", "--test-reporter=junit", f"--test-reporter-destination={report}", *wanted]
    code = run_private(safe_command(command, raw, 240), child_env(work), work / "node.out", 360)
    upstream = launcher_reason(raw, code)
    try:
        result = junit(report)
    except (OSError, ValueError, ElementTree.ParseError):
        result = {"state": "failed", "reason": "case_projection_missing_or_invalid"}
    result.update(process_exit=code, files=wanted, node_version=found)
    if upstream != "ok" or code:
        result["case_reason"], result["reason"] = result["reason"], upstream if upstream != "ok" else "test_process_nonzero"
        result["state"] = "failed"
    return result


def binding(subject: dict, lane: str) -> dict:
    return {"subject_sha256": sha256(json.dumps(subject, sort_keys=True).encode()).hexdigest(),
            "lane": lane, "run_id": os.environ["GITHUB_RUN_ID"], "attempt": os.environ["GITHUB_RUN_ATTEMPT"]}


def planned_lanes(subject: dict) -> list[str]:
    mode, names = subject["mode"], subject["lanes"].split()
    expected = ({"size", "node", "structure", "consumers", "docker", "browser"} if system() == "linux"
                else {"consumers"}) if mode == "gate" else (
                    {"serial_tb"} if subject["leg"] == "serial-tb" else {"parity", "known"})
    if (mode not in SCOPE or len(names) != len(expected) or set(names) != expected
            or subject["qa_commit"] != os.environ["GITHUB_SHA"]):
        raise Gap("lane_plan_invalid")
    return names


def lane_receipt(subject: dict, lane: str) -> tuple[dict, dict | None]:
    try:
        data = read_json(priv() / f"receipt-{lane}.json")
        row, extract = data["result"], data.get("failure_extract")
        if (data["binding"] != binding(subject, lane) or not isinstance(row, dict)
                or row["state"] not in ("passed", "reported", "failed", "incomplete")
                or not isinstance(row["reason"], str)):
            raise Gap("lane_receipt_invalid")
        if row["state"] in ("passed", "reported"):
            if not isinstance(row.get("cases"), dict) or type(row.get("process_exit")) is not int:
                raise Gap("lane_receipt_invalid")
            if lane != "node" and (row.get("complete") is not True or row.get("pytest_exit") != row["process_exit"]
                                   or set(row.get("by_id", ())) != set(lane_ids(lane))):
                raise Gap("lane_receipt_invalid")
        if extract is not None and (lane not in DIAGNOSTIC or row["state"] != "reported"
                                    or extract["subject_tree"] != subject["subject_tree"]):
            raise Gap("failure_extract_unbound")
        return row, extract
    except FileNotFoundError:
        return {"state": "not_run", "reason": not_run_reason(lane)}, None
    except (Gap, OSError, ValueError, KeyError, TypeError):
        return {"state": "incomplete", "reason": "lane_receipt_invalid"}, None


def lane_main(lane: str) -> int:
    if lane not in LANES:
        print("unknown lane", file=sys.stderr)
        return 2
    failed = "incomplete" if lane in DIAGNOSTIC else "failed"
    subject, extract = None, None
    try:
        subject = read_json(priv() / "subject.json")
        if lane not in planned_lanes(subject):
            raise Gap("lane_not_planned")
        bound_interpreter()
        check_checkout(subject)
        result = {"size": size_lane, "node": node_lane}.get(lane, run_pytest_lane)(lane, subject)
        check_checkout(subject)
        if lane in DIAGNOSTIC and result["state"] == "reported" and result["failed_total"]:
            extract = failure_extract(result, subject)
    except Gap as gap:
        result = {"state": failed, "reason": str(gap)}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        result = {"state": failed, "reason": "lane_infrastructure_error"}
    if subject is not None:
        write_json(priv() / f"receipt-{lane}.json", {"binding": binding(subject, lane), "result": result,
                                                  "failure_extract": extract})
    print(f"QA_LANE {lane} {result['state']} {result['reason']}", flush=True)
    return 0 if result["state"] in ("passed", "reported") else 1


# ---- the one public receipt -------------------------------------------------------------------------------------

def not_run_reason(lane: str) -> str:
    setup = dict(item.split("=", 1) for item in os.environ.get("QA_SETUP", "").split() if "=" in item)
    for need in ("python", {"node": "browsers", "browser": "browsers", "docker": "image"}.get(lane, "")):
        if need and setup.get(need) != "success":
            return f"prerequisite_unavailable_{need}"
    return "lane_step_did_not_report"


def slot_rows(lanes: dict) -> list[dict]:
    rows = []
    for slot, env, systems, owner_lane in SLOTS:
        if system() not in systems:
            continue
        wanted = slot_cases(env, owner_lane)
        if not wanted:
            rows.append({"slot": slot, "lane": owner_lane, "state": "unknown", "reason": "id_not_supplied"})
            continue
        receipt = lanes.get(owner_lane, {})
        by_id = receipt.get("by_id", {})
        done = receipt.get("state") == "passed" and all(by_id.get(w, {}).get("state") == "passed" for w in wanted)
        if owner_lane == "docker":
            done = done and all(receipt.get("docker_stages", {}).get(w) == DOCKER_STAGES[-1] for w in wanted)
        rows.append({"slot": slot, "lane": owner_lane, "ids": wanted, "state": "passed" if done else "failed",
                     "reason": "ok" if done else "owning_lane_did_not_confirm_case"})
    return rows


def report() -> int:
    out = pub()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ
    try:
        subject = read_json(priv() / "subject.json")
        planned = planned_lanes(subject)
    except (Gap, OSError, ValueError, KeyError, TypeError):
        try:
            why = (priv() / "reject.txt").read_text(encoding="ascii")[:80]
        except (OSError, ValueError):
            why = "unrecorded"
        write_json(out / "receipt.json", {"schema": 1, "qualified": False, "complete": False,
                                          "reason": "subject_unbound", "rejected": clean(why, 80),
                                          "qa_commit": env.get("GITHUB_SHA", ""), "run_id": env.get("GITHUB_RUN_ID", "")})
        print("QA_REPORT subject_unbound", flush=True)
        return 1
    gate = subject["mode"] == "gate"
    lanes = {}
    for lane in planned:
        lanes[lane], extract = lane_receipt(subject, lane)
        if extract is not None:
            write_json(out / f"failures-{lane}.json", {"lane": lane, **extract})
    try:
        slots = slot_rows(lanes) if gate else []
    except Gap:
        slots = [{"slot": "witness_table", "state": "failed", "reason": "witness_table_invalid"}]
    done = (all(lane["state"] == ("passed" if gate else "reported") for lane in lanes.values())
            and all(s["state"] == "passed" for s in slots))
    summary = {
        "schema": 1, "scope": SCOPE[subject["mode"]], "mode": subject["mode"], "leg": subject["leg"],
        "qualified" if gate else "complete": done,
        "subject": {k: subject[k] for k in ("subject", "subject_commit", "subject_tree", "product", "tree", "base",
                                            "base_tree", "diagnostic_control", "diagnostic_control_tree",
                                            "qa_commit", "driver_blob", "workflow_blob")},
        "run": {"repository": env.get("GITHUB_REPOSITORY", ""), "ref": env.get("GITHUB_REF", ""),
                "run_id": env.get("GITHUB_RUN_ID", ""), "attempt": env.get("GITHUB_RUN_ATTEMPT", ""),
                "runner_os": env.get("RUNNER_OS", ""), "runner_arch": env.get("RUNNER_ARCH", ""),
                "image": f"{env.get('ImageOS', '')} {env.get('ImageVersion', '')}".strip()},
        "tools": {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version},
        "lanes": lanes, "slots": slots,
    }
    write_json(out / "receipt.json", summary)
    with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as text:
        print(f"## {'Task service consumer gate' if gate else 'DIAGNOSTIC (not a gate)'} - {subject['leg']}", file=text)
        print(f"Subject `{subject['subject']}`: `{subject['subject_commit']}` (tree `{subject['subject_tree']}`); "
              f"product `{subject['product']}`; comparison base `{subject['base']}`.", file=text)
        print(f"\n_{SCOPE[subject['mode']]}._\n\n| Lane | State | Reason | Selected | Passed | Failed |\n"
              "| --- | --- | --- | --- | --- | --- |", file=text)
        for name, lane in lanes.items():
            print(f"| {name} | {lane['state']} | {lane['reason']} | {lane.get('selected', '')} | "
                  f"{lane.get('cases', {}).get('passed', '')} | {lane.get('failed_total', '')} |", file=text)
        for row in slots:
            print(f"| slot: {row['slot']} | {row['state'].upper()} | {row['reason']} | | | |", file=text)
        print(f"\nResult: {'QUALIFIED' if done and gate else 'COMPLETE' if done else 'NOT QUALIFIED' if gate else 'INCOMPLETE'}"
              " (UNKNOWN, not-run and skipped rows are never a pass).", file=text)
    print(f"QA_REPORT {subject['leg']} {'ok' if done else 'incomplete_or_failed'}", flush=True)
    return 0 if done else 1


# ---- the real-Docker witness (unchanged from the one-off qualification; the only product imports are inside) ------

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
    witness = Path(os.environ["QA_DOCKER_WITNESS_DIR"]) / "test_docker_restart.json"
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


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "lane":
        return lane_main(argv[1])
    if argv == ["report"]:
        return report()
    print("usage: task_service_consumer.py lane <name> | report", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
