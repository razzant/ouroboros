"""Driver and opt-in pytest observer for the disposable PR 3713 / S11 qualification branch.

Every lane checks out one pinned subject, runs one fixed selection with the checkout's
dependency-only .venv under scripts/safe_test.py, and projects only public facts into
outcome.json: how each process ended, per-case phase outcomes, which processes ran real
call phases, typed stand counters and top-level Node TAP names. Raw output, tracebacks and
TAP diagnostics stay in RUNNER_TEMP/qualification-private and are never uploaded.

  system-1, system-2      S11: tests/system_e2e/ and the cancellation E-suite, mock lanes, -n 1 / -n 2
  stand-{linux,windows}-{before,after}  ORIGINAL / FIXTURES: restored-stand cost test, counters both ways
  {linux,windows}-{before,after}        ORIGINAL / FIXTURES: historical reds (four / ten), one xdist worker
  settings                            BASE + FIXTURES: three whole pytest modules, one Node file
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from collections import Counter


ORIGINAL = "cba6276bc0626d1ae0bc599c31a33e9dcdb33e98"  # base of PR 3713 and of the historical reds
FIXTURES = "92d07e1daf4154e609a907517ed4ea55dda54855"  # head of PR 3713
S11 = "83aed31ced3b3b9405c6bb0e3c93986534b5c4d4"  # the S11 restart test on FIXTURES; the driver's only parent
# The composition is this fixed pair merged here: a result about it, not a promise about a later merge.
BASE = "b581c570d0613479c1bbb8ae3d1c990183bf8c7f"  # public `ouroboros` tip chosen for the composition
COMPOSED_TREE = "ea679b69e5e7907263b5816669664802d48b06ef"  # git merge-tree --write-tree BASE FIXTURES
BRANCH = "qualification/restart-and-stand-20261118"
DRIVER_FILES = (".github/qualification/qualification_witness.py",
                ".github/workflows/qualification-restart-and-stand.yml")
# Fixed identity and date: the local composition commit is reproducible; only its tree and parents are claimed.
COMPOSITION_ENV = {"GIT_AUTHOR_NAME": "qualification", "GIT_AUTHOR_EMAIL": "qualification@invalid",
                   "GIT_AUTHOR_DATE": "2026-11-18T00:00:00+00:00", "GIT_COMMITTER_NAME": "qualification",
                   "GIT_COMMITTER_EMAIL": "qualification@invalid", "GIT_COMMITTER_DATE": "2026-11-18T00:00:00+00:00"}

LANE = os.environ["WITNESS_LANE"]
SUBJECTS = {"system-1": (S11, 1), "system-2": (S11, 2), "settings": (None, 0)}
SUBJECTS.update({f"{suite}-{revision}": (commit, int(not suite.startswith("stand-")))
                 for suite in ("linux", "windows", "stand-linux", "stand-windows")
                 for revision, commit in (("before", ORIGINAL), ("after", FIXTURES))})
SUBJECT, WORKERS = SUBJECTS[LANE]
RUNNER_OS = "Windows" if "windows-" in LANE else "Linux"
RAW = Path(os.environ["RUNNER_TEMP"]) / "qualification-private"
PROOF = RAW.with_name("qualification-proof")
# Never resolved: started through its own path, the interpreter finds pyvenv.cfg and the venv's packages.
VENV_PYTHON = os.path.join(".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(".venv", "bin", "python")

SUITE = "tests/system_e2e/"
CANCEL = "tests/test_e2e_cancellation_scenarios.py"
FOCAL = {SUITE + "test_system_scenarios_w3b.py::" + name for name in (
    "test_s11_delegated_transport_wire_custody_and_typed_refusals",
    "test_fake_daemon_serves_the_real_gateway_contract",
    "test_fake_daemon_run_replay_refusals_and_cancel",
)}
STAND = "tests/test_cost_contract.py::test_a_restored_stand_measures_what_its_build_measured"
STAND_TAGS = {"B1", "B2_ready", "B3", "B5", "R_late"}  # the boundaries the test itself requires
SETTINGS_PY = ("tests/test_context_runtime_section.py", "tests/test_execution_domain.py",
               "tests/test_settings_change_record.py")
SETTINGS_JS = "tests/settings_home.test.js"  # relative to web/, where the Node lane runs it
SETTINGS_JS_TOP_LEVEL = 27  # web/tests/settings_home.test.js blob 051d34bb on the composed tree
SETTINGS_JS_TESTS = 39  # 27 top-level and the 12 nested t.test in its one 3x2x2 loop
BACKTICKS = "tests/test_llm_provider_routing.py::test_backtick_runs_are_read_in_linear_time"
# Failed items of the full-test recount of run 44263523476, attempt 1. The two digests in WINDOWS
# cover the complete UTF-8 node ids of its two long parameters; those values never go in argv or evidence.
LINUX = {
    "tests/test_ui_serving.py::test_the_wizard_document_and_its_module_graph_never_reach_the_selected_static_tree",
    "tests/test_docs_sync.py::test_architecture_component_map_covers_every_live_runtime_module",
    "tests/test_conversation_lane_saved_turn_restore.py::test_a_presence_seed_stays_unmarked",
    "tests/test_project_chat_continuity.py::test_active_chat_activity_contract_mirrors_direct_turn_shape",
}
WINDOWS = LINUX | {
    BACKTICKS + "[sha256:10dfee32efa4]", BACKTICKS + "[sha256:3353ddc36cb0]",
    "tests/test_linked_knowledge.py::test_link_facts_ride_a_read_from_the_start_and_survive_odd_spellings_and_unresolvable_names",
    "tests/test_linked_knowledge.py::test_read_backlinks_join_exact_addresses_only_on_readable_shelves",
    "tests/test_linked_knowledge.py::test_unresolvable_links_do_not_hide_the_note_or_other_backlinks",
    "tests/test_skill_mcp_readiness.py::test_agent_python_launch_prefers_valid_agent_interpreter_and_runs_isolated",
}
HISTORICAL = WINDOWS if LANE.startswith("windows-") else LINUX if LANE.startswith("linux-") else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_fact(path, gaps):
    try:
        return load(path)
    except (OSError, ValueError) as exc:
        gaps.append(f"{path.name}: {type(exc).__name__}")
        return None


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def public_id(node):
    """A plain ASCII test path and name stay readable; parameters and anything else become a digest."""
    base, bracket, _ = node.partition("[")
    if not re.fullmatch(r"tests/(?:\w+/)*\w+\.py(?:::\w+)*", base, re.ASCII):
        return "sha256:" + digest(node)
    return f"{base}[sha256:{digest(node)}]" if bracket else base


def git(*args):
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


# ---- pin: the driver, then the exact subject ------------------------------------------------------

def compose(record):
    """BASE + FIXTURES merged on this runner, refused unless the result is the independently recorded tree."""
    record.update(base=BASE, pr_head=FIXTURES, expected_tree=COMPOSED_TREE)
    for commit in (BASE, FIXTURES):
        if subprocess.call(["git", "cat-file", "-e", commit + "^{commit}"], stderr=subprocess.DEVNULL):
            git("fetch", "--quiet", "--no-tags", "origin", commit)  # this exact object, never a moving ref
            record.setdefault("fetched_by_id", []).append(commit)
    merged = subprocess.run(["git", "merge-tree", "--write-tree", BASE, FIXTURES],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    record.update(merge_exit=merged.returncode, merge_tree=(merged.stdout.split() or [None])[0])
    require(merged.returncode == 0 and record["merge_tree"] == COMPOSED_TREE,
            "BASE and FIXTURES do not merge cleanly into the recorded tree")
    commit = subprocess.check_output(
        ["git", "commit-tree", COMPOSED_TREE, "-p", BASE, "-p", FIXTURES, "-m",
         f"PR 3713 head on ouroboros {BASE[:8]} (local qualification composition)"],
        env={**os.environ, **COMPOSITION_ENV}, text=True, stderr=subprocess.DEVNULL).strip()
    record["parents"] = git("show", "-s", "--format=%P", commit).split()
    require(record["parents"] == [BASE, FIXTURES], "unexpected composition parents")
    return commit


def pin():
    record = {"lane": LANE, "driver": os.environ["GITHUB_SHA"]}
    try:
        require(os.environ["GITHUB_REF"] == "refs/heads/" + BRANCH, "unexpected driver ref")
        require(os.environ["RUNNER_OS"] == RUNNER_OS and (os.name == "nt") == (RUNNER_OS == "Windows"),
                "runner OS does not match the lane")
        require(git("rev-list", "--parents", "-1", "HEAD").split() == [record["driver"], S11],
                "the driver's only parent is not S11")
        changed = sorted(git("diff", "--no-renames", "--name-status", S11, "HEAD").splitlines())
        require(changed == sorted("A\t" + path for path in DRIVER_FILES), "the driver changes more than its two files")
        subject = SUBJECT or compose(record)
        git("checkout", "--quiet", "--detach", subject)
        record.update(subject=subject, head=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))
        require(record["head"] == subject and (SUBJECT is not None or record["tree"] == COMPOSED_TREE),
                "subject checkout mismatch")
        record["state"] = "verified"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record.update(state="untested", reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        raise SystemExit(record["reason"]) from None
    finally:
        write(PROOF / "source.json", record)


# ---- run: safe_test, then the lane's runners inside its boundary ------------------------------------

def selection():
    """The pytest targets and options of this lane (the subject's own conftest still groups and orders)."""
    if LANE.startswith("system-"):
        return [SUITE, CANCEL], ["--dist=loadgroup", "--max-worker-restart=0"]
    if HISTORICAL is not None:
        # One xdist worker keeps the controller's export after a worker loss. Keep the two large
        # Windows inputs last even when conftest/xdist would give their group a higher priority.
        targets = sorted({node.partition("[")[0] for node in HISTORICAL},
                         key=lambda target: (target == BACKTICKS, target))
        return targets, ["--dist=loadgroup", "--no-loadscope-reorder", "--timeout=300", "--max-worker-restart=0"]
    if LANE.startswith("stand-"):
        return [STAND], ["--timeout=300"]
    return list(SETTINGS_PY), ["--timeout=300"]


def produce():
    source = load(PROOF / "source.json")
    python = os.path.abspath(VENV_PYTHON)
    try:
        require(source.get("state") == "verified" and git("rev-parse", "HEAD") == source.get("head"),
                "subject changed after its checkout")
        require(not git("status", "--porcelain"), "setup changed the subject checkout")
        require(os.path.isfile(python), "the checkout has no .venv interpreter")
    except (ValueError, subprocess.SubprocessError) as exc:
        source.update(state="untested", reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        write(PROOF / "source.json", source)
        raise SystemExit(source["reason"]) from None
    RAW.mkdir(parents=True, exist_ok=True)
    # The documented form, python -I -S scripts/safe_test.py -- <venv-python> ...: scrubbed roots,
    # one process container and one deadline for everything the child starts.
    command = [python, "-I", "-S", "scripts/safe_test.py", "--evidence-dir", str(RAW / "safe"),
               "--timeout", "6900" if LANE.startswith("system-") else "1800",
               "--", python, "-I", "-S", os.path.abspath(__file__), "child"]
    with (RAW / "launcher.log").open("wb") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
    write(RAW / "launcher-exit.json", code)
    return code


def child():
    """Inside safe_test's boundary, cwd the checkout. child-exit.json exists only if every runner returned."""
    python = os.path.abspath(VENV_PYTHON)
    if LANE.startswith("system-"):
        # safe_test keeps OUROBOROS_E2E_DEEP and scrubs OUROBOROS_E2E_CANCEL; both lanes are chosen here.
        os.environ.update(OUROBOROS_E2E_DEEP="mock", OUROBOROS_E2E_CANCEL="mock")
    targets, options = selection()
    command = [python, "-m", "pytest", "-p", "qualification_witness", *targets, "-o", "addopts=", "-q",
               "--tb=no", "-n", str(WORKERS), "--ci-evidence-dir=" + str(RAW / "phases"), *options]
    observer = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))}  # this module only
    exits = {"pytest": subprocess.call(command, env=observer)}
    if LANE == "settings":
        node = shutil.which("node")
        exits["node"] = None if node is None else subprocess.call(
            [node, "--test", "--test-timeout=300000", "--test-reporter=tap",
             "--test-reporter-destination=" + str(RAW / "settings.tap"), SETTINGS_JS], cwd="web")
    write(RAW / "child-exit.json", exits)
    return 0 if all(code == 0 for code in exits.values()) else 1


def cost_differences(item):
    """Every contract counter of every measured boundary as an int pair, equal pairs included, with the module's
    own breach rule applied in both directions. Nothing else of the fixtures is kept."""
    module = item.module
    built, restored = item.funcargs["built_small"], item.funcargs["small"]
    facts = {"same_root": built["root"] == restored["root"], "k": module.K_SMALL,
             "allowances": {"ordinary": module.COUNT_JITTER, "bytes": module.BYTE_JITTER,
                            "vm_steps": module.vm_allowance(module.K_SMALL)}}
    if not facts["same_root"]:
        return {**facts, "state": "unavailable", "reason": "different fixture roots"}
    tags = sorted(tag for tag, data in restored.items()
                  if isinstance(data, dict) and "counts" in data and not tag.startswith("T:"))
    paired = [tag for tag in tags if isinstance(built.get(tag), dict) and "counts" in built[tag]]
    facts["missing_tags"] = sorted((STAND_TAGS - set(tags)) | (set(tags) - set(paired)))
    boundaries = facts["boundaries"] = {}
    for tag in paired:
        require(re.fullmatch(r"[\w:-]{1,96}", tag, re.ASCII), "unexpected boundary name")
        before, after = built[tag]["counts"], restored[tag]["counts"]
        found = {"build_to_restore": module.contract_breaches(before, after, module.K_SMALL),
                 "restore_to_build": module.contract_breaches(after, before, module.K_SMALL)}
        pairs = boundaries[tag] = {}
        for key in (*module.WORK, "vm_steps"):  # an absent counter reads 0, as contract_breaches reads it
            require(re.fullmatch(r"[\w:-]{1,96}", key, re.ASCII), "unexpected counter name")
            old, new = before.get(key, 0), after.get(key, 0)
            require(type(old) is int and type(new) is int, "non-integer counter")
            pairs[key] = {"built": old, "restored": new, **{way: key in breaches for way, breaches in found.items()}}
    return {**facts, "state": "measured"}


if __name__ != "__main__":
    import pytest

    CHECKOUT_VENV = sys.prefix != sys.base_prefix and Path(sys.prefix).resolve() == Path(".venv").resolve()

    def keep(path, facts):
        """A receipt that fails to persist never replaces pytest's own phase outcome; the report names it."""
        try:
            write(path, facts)
        except OSError:
            pass

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(config, items):
        if HISTORICAL is not None:
            chosen, drop = [], []
            for item in items:
                (chosen if public_id(item.nodeid) in HISTORICAL else drop).append(item)
            # conftest's tryfirst hook already tagged and sorted the groups. Do this AFTER it.
            chosen.sort(key=lambda item: item.nodeid.partition("[")[0] == BACKTICKS)
            items[:] = chosen
            if drop:
                config.hook.pytest_deselected(items=drop)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item, call):
        outcome = yield
        try:
            if call.when == "call":
                # The process that ran this call phase says so itself, apart from xdist's report transport.
                worker = getattr(item.config, "workerinput", {}).get("workerid", "main")
                row = {"pid": os.getpid(), "venv": CHECKOUT_VENV,
                       "case": public_id(item.nodeid), "outcome": outcome.get_result().outcome}
                with (RAW / f"calls-{worker}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
            elif item.nodeid == STAND and call.when == "setup" and call.excinfo is not None:
                keep(RAW / "counters.json", {"state": "unavailable", "reason": "fixture setup did not complete"})
        except Exception:
            pass  # the report names the missing call row

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_call(item):
        if item.nodeid == STAND:
            try:
                facts = cost_differences(item)
            except Exception as exc:
                facts = {"state": "unavailable", "reason": type(exc).__name__}
            keep(RAW / "counters.json", facts)


# ---- report: public projection --------------------------------------------------------------------

def process_report(result, steps, gaps, unmet):
    """How the processes ended, kept apart from what the tests did."""
    launcher = read_fact(RAW / "launcher-exit.json", gaps)
    runners = read_fact(RAW / "child-exit.json", gaps)
    names = ("pytest", "node") if LANE == "settings" else ("pytest",)
    exits = {name: runners[name] if isinstance(runners, dict) and type(runners.get(name)) is int else None
             for name in names}
    exited = isinstance(runners, dict) and set(runners) == set(names) and all(code is not None for code in exits.values())
    owners = sorted((RAW / "safe").glob("*/result.json"))
    owner = read_fact(owners[0], gaps) if len(owners) == 1 else None
    supervisor = None
    if isinstance(owner, dict):
        execution, code, stop = owner.get("execution"), owner.get("returncode"), owner.get("stopped")
        supervisor = {"execution": execution if execution in ("not_run", "attempted", "executed") else None,
                      "returncode": code if type(code) is int else None,
                      "stopped": (stop if stop == "timeout" or (isinstance(stop, str)
                                                                and re.fullmatch(r"signal \d+", stop))
                                  else None if stop == "" else "unknown")}
        supervisor.update({key: bool(owner.get(key)) for key in
                           ("launch_error", "containment_error", "evidence_error", "source_gap")})
        cleanup = owner.get("cleanup")
        if isinstance(cleanup, dict) and all(isinstance(cleanup.get(key), list) for key in ("released", "retained")):
            # Paths and free-form retention reasons are not public. A kept folder isn't by itself an unreaped process.
            supervisor["cleanup"] = {"state": "finished", "released": len(cleanup["released"]),
                                     "retained": len(cleanup["retained"]), "problem": bool(cleanup.get("reason"))}
            if supervisor["cleanup"]["problem"]:
                gaps.append("safe_test cleanup reported a problem")
        else:
            supervisor["cleanup"] = {"state": "pending" if cleanup == "pending" else "unconfirmed"}
            gaps.append("safe_test has no final cleanup receipt")
        if (supervisor["execution"] != "executed" or supervisor["stopped"]
                or any(supervisor[key] for key in ("launch_error", "containment_error", "evidence_error", "source_gap"))):
            gaps.append("safe_test reports a launch, stop, containment, evidence or source problem")
    else:
        gaps.append("safe_test result unavailable")
    launcher = launcher if type(launcher) is int else None
    result.update(execution="exited" if exited else "unconfirmed",
                  process={"launcher_exit": launcher, "supervisor": supervisor, "runners": exits})
    if not (exited and all(code == 0 for code in exits.values()) and launcher == 0
            and (supervisor or {}).get("returncode") == 0 and steps["steps"]["producer"] == "success"):
        unmet.append("a successful exit of every runner is unconfirmed")


def phase_rows(result, gaps):
    """ci_evidence's redacted phases: the final export, else the journal a killed session left."""
    root = RAW / "phases"
    try:
        data = load(root / "results.json")
    except (OSError, ValueError):
        data = None
    identity = {key: os.environ["GITHUB_" + key.upper()] for key in ("sha", "run_id", "run_attempt")}
    if (isinstance(data, dict) and data.get("github") == identity and isinstance(data.get("reports"), list)
            and type(data.get("session_exit_code")) is int and type(data.get("tests_collected")) is int
            and isinstance(data.get("collection_failures"), list) and isinstance(data.get("collection_skips"), list)):
        environment = data.get("environment") or {}
        python = environment.get("python")
        system = environment.get("platform")
        result["pytest_environment"] = {
            "python": python if isinstance(python, str) and re.fullmatch(r"\d+\.\d+\.\d+", python) else None,
            "os": system if system in ("Linux", "Windows") else None,
        }
        if system != RUNNER_OS or result["pytest_environment"]["python"] is None:
            gaps.append("pytest did not report the expected runtime platform and a Python version")
        result.update(phase_source="results.json", session_exit=data["session_exit_code"],
                      collected=data["tests_collected"],
                      collection_failures=sorted(public_id(str(row.get("nodeid"))) for row in data["collection_failures"]
                                                 if isinstance(row, dict)),
                      collection_skips=sorted(public_id(str(node)) for node in data["collection_skips"]))
        crash = data.get("worker_crash")
        if isinstance(crash, dict):  # tests/conftest.py stopped the session; its reason text stays private
            result["worker_crash"] = {"worker": str(crash.get("worker")),
                                      "unfinished": sorted(public_id(str(node)) for node in crash.get("unfinished") or ())}
        return data["reports"]
    gaps.append("no final pytest phase export of this run")
    result.update(phase_source="events.jsonl (partial)", session_exit=None, collected=None,
                  collection_failures=None, collection_skips=None)
    rows, running = [], set()
    try:
        with (root / "events.jsonl").open("rb") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                    kind, node = event["event"], public_id(event["nodeid"])
                except (ValueError, KeyError, TypeError):
                    gaps.append("unreadable phase journal row")  # a killed writer may tear its last line
                    continue
                if kind == "report":
                    rows.append(event)
                    if event.get("phase") == "crash":
                        running.discard(node)
                elif kind == "start":
                    running.add(node)
                elif kind == "finish":
                    running.discard(node)
    except OSError as exc:
        gaps.append(f"phase journal: {type(exc).__name__}")
    result["in_flight"] = sorted(running)
    return rows


def outcome(rows):
    if any(row["outcome"] == "failed" for row in rows):
        return "failed"
    parents = [row for row in rows if not row["is_subtest"]]
    phases = [row["phase"] for row in parents]
    if phases != ["setup", "call", "teardown"] and not (
            phases == ["setup", "teardown"] and parents[0]["outcome"] == "skipped"):
        return "incomplete"
    if any(row["expected"] and row["outcome"] == "passed" for row in rows):
        return "xpassed"
    if any(row["outcome"] == "skipped" and not row["expected"] for row in rows):
        return "skipped"
    if any(row["expected"] for row in rows):
        return "xfailed"
    return "passed"


def outside_mock(node):
    """Cases whose own gate names another lane: S39 `container`, S43 `stand_full`, the E-suite's `paid`."""
    desktop_case = SUITE + "test_window_capture_task_native.py::test_managed_window_task_lifetime"
    return (node in {desktop_case + f"[{ending}]" for ending in ("finish", "immediate_stop", "window_gone")}
            or node.startswith(SUITE + "test_managed_container.py::test_s39_")
            or node.startswith(SUITE + "test_stand_server.py::") and re.search(r"(?:\[|-)wal-full(?:-|\])", node)
            or node.startswith(CANCEL + "::") and re.match(r"test_e(?:1|2|3|13)_", node.split("::")[1]))


def calls_report(result, cases, gaps, unmet):
    """Which processes ran real call phases, as each recorded it, against the phases the controller reported.

    A process counts only by a complete, non-empty call log with one positive PID, distinct between workers.
    Multiplicity and outcome of each call are compared only for ordinary cases (setup, call, teardown and no
    subtest or xfail row): a subtest logs more calls and an xfail rewrites the outcome, so other cases are
    compared by presence alone."""
    workers, logged = {}, {}
    for path in sorted(RAW.glob("calls-*.jsonl")):
        rows, torn = [], False
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                rows.append((int(row["pid"]), row["venv"] is True, str(row["case"]), str(row["outcome"])))
            except (ValueError, KeyError, TypeError):
                torn = True
        if torn:
            gaps.append("unreadable call log row")
        pids = sorted({row[0] for row in rows})
        workers[path.name[len("calls-"):-len(".jsonl")]] = {
            "pids": pids, "calls": len(rows), "venv": all(row[1] for row in rows),
            "counted": bool(rows) and not torn and len(pids) == 1 and pids[0] > 0}
        for _, _, case, status in rows:
            logged.setdefault(case, []).append(status)
    counted = {name: worker["pids"][0] for name, worker in workers.items() if worker["counted"]}
    expected = {f"gw{index}" for index in range(WORKERS)} if WORKERS else {"main"}
    if set(workers) != expected or set(counted) != expected or len(set(counted.values())) != len(counted):
        unmet.append("call phases did not run in exactly the requested processes")
    if not all(worker["venv"] for worker in workers.values()):
        unmet.append("a call phase ran outside the checkout venv")
    ordinary, mismatched = 0, set(logged) - set(cases)
    for node, case in cases.items():
        calls = [phase["outcome"] for phase in case["phases"] if phase["phase"] == "call" and not phase["is_subtest"]]
        if ([phase["phase"] for phase in case["phases"]] == ["setup", "call", "teardown"]
                and not any(phase["is_subtest"] or phase["expected"] for phase in case["phases"])):
            ordinary += 1
            if logged.get(node) != calls:  # exactly one logged call, with the reported outcome
                mismatched.add(node)
        elif bool(logged.get(node)) != bool(calls):
            mismatched.add(node)
    result["calls"] = {"requested_workers": WORKERS, "workers": workers, "ordinary_compared": ordinary,
                       "presence_compared": len(cases) - ordinary, "mismatched": sorted(mismatched)}
    if mismatched:
        gaps.append("reported call phases and the processes' own call logs differ")


def stand_report(result, cases, gaps, unmet):
    counters = result["counters"] = read_fact(RAW / "counters.json", gaps)
    if not isinstance(counters, dict) or counters.get("state") != "measured":
        gaps.append("build/restore counter comparison unavailable")
        return
    breached = any(pair["build_to_restore"] or pair["restore_to_build"]
                   for pairs in counters["boundaries"].values() for pair in pairs.values())
    holds = not counters["missing_tags"] and not breached
    if not holds:
        unmet.append("the restored stand's counters break its build's contract")
    calls = [phase["outcome"] for phase in cases.get(STAND, {}).get("phases", ())
             if phase["phase"] == "call" and not phase["is_subtest"] and not phase["expected"]]
    verdict = counters["call_outcome"] = calls[0] if len(calls) == 1 else None
    if verdict not in ("passed", "failed"):
        gaps.append("no ordinary stand call outcome to compare with the counters")
    elif (verdict == "passed") != holds:
        gaps.append("the stand test's verdict and the measured counters disagree")


def phase_report(result, gaps, unmet):
    # Not named pytest_*: pytest would take any such module-level function of this plugin for an unknown hook.
    cases, nonmock = {}, set()
    for row in phase_rows(result, gaps):
        if not isinstance(row, dict):
            gaps.append("invalid pytest phase")
            continue
        node, phase, status = row.get("nodeid"), row.get("phase"), row.get("outcome")
        if (not isinstance(node, str) or type(row.get("is_subtest")) is not bool
                or phase not in ("setup", "call", "teardown", "crash") or status not in ("passed", "failed", "skipped")):
            gaps.append("invalid pytest phase")
            continue
        key = public_id(node)
        if outside_mock(node):
            nonmock.add(key)
        entry = {"phase": phase, "outcome": status, "is_subtest": row["is_subtest"],
                 "expected": row.get("wasxfail") is not None}
        error = row.get("error_type")
        if error:
            entry["error_type"] = (error if isinstance(error, str) and re.fullmatch(r"[A-Za-z_]\w{0,79}", error, re.ASCII)
                                   else "unclassified")
        timeout = row.get("timeout_callback_seconds")
        if phase == "crash" and type(timeout) in (int, float) and 0 < timeout < 86400:
            entry["timeout_callback_seconds"] = timeout
        cases.setdefault(key, {"phases": []})["phases"].append(entry)
    for case in cases.values():
        case["outcome"] = outcome(case["phases"])
    counts = Counter(case["outcome"] for case in cases.values())
    result.update(cases=cases, case_counts=dict(counts))
    if result["collected"] is not None and result["collected"] != len(cases):
        gaps.append("collected and reported cases differ")
    if counts["incomplete"]:
        gaps.append("incomplete parent phases")
    if result["session_exit"] is not None and result["session_exit"] != result["process"]["runners"]["pytest"]:
        gaps.append("pytest's terminal export and child exit disagree")
    calls_report(result, cases, gaps, unmet)

    if LANE.startswith("system-"):
        required = FOCAL
        result["suite_counts"] = {scope: dict(Counter(case["outcome"] for node, case in cases.items()
                                                      if node.startswith(scope))) for scope in (SUITE, CANCEL)}
        if not all(result["suite_counts"].values()):
            gaps.append("a suite is absent from the reported cases")
        result["unexpected_skips"] = sorted(node for node, case in cases.items()
                                            if case["outcome"] == "skipped" and node not in nonmock)
        if result["unexpected_skips"]:
            unmet.append("mock cases skipped")
    elif HISTORICAL is not None:
        required = HISTORICAL
        # One worker: the controller's first phase for each case proves observed order, even if pytest
        # ended red. Unreported cases are still unobserved, not a passed retry of the historical run.
        order = list(cases)
        result["historical_observed_order"] = order
        if set(cases) != required:
            gaps.append("reported cases differ from the historical failure inventory")
        if LANE.startswith("windows-") and set(cases) == WINDOWS and any(
                not case.startswith(BACKTICKS + "[") for case in order[-2:]):
            gaps.append("the two long Windows cases were not observed last")
    elif LANE.startswith("stand-"):
        required = {STAND}
        stand_report(result, cases, gaps, unmet)
    else:
        required = set(cases)  # every case of all three whole modules
        if {node.split("::", 1)[0] for node in cases} != set(SETTINGS_PY):
            gaps.append("not all three composition modules are reported")
    result["required"] = {node: cases.get(node, {}).get("outcome", "unobserved") for node in sorted(required)}
    if not required or any(status != "passed" for status in result["required"].values()):
        unmet.append("required cases lack an ordinary three-phase pass")
    if (counts["failed"] or counts["xpassed"] or result["collection_failures"] or result["collection_skips"]
            or result["session_exit"] not in (0, None)):
        unmet.append("the pytest session includes a failure, collection failure or skip, XPASS or nonzero exit")


def node_report(result, gaps, unmet):
    """settings_home.test.js by its TAP: top-level names against the pinned source, and the totals, which count
    every depth, against its 39 tests. TAP diagnostics (values, stacks) never leave the runner."""
    source = (Path("web") / SETTINGS_JS).read_text(encoding="utf-8")
    names = {name.replace("\\'", "'") for name in re.findall(r"^test\('((?:\\.|[^'\\])*)',", source, re.M)}
    tests, totals = {}, {}
    plan = []
    try:
        lines = (RAW / "settings.tap").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        gaps.append(f"settings.tap: {type(exc).__name__}")
        lines = []
    for line in lines:
        if top_plan := re.fullmatch(r"1\.\.(\d+)", line):
            plan.append(int(top_plan[1]))
        top = re.fullmatch(r"(ok|not ok) \d+ - (.+?)(?: # (SKIP|TODO)\b.*)?", line.rstrip())
        if top:  # nested subtests are indented, so only top-level lines match
            flag, name, directive = top.groups()
            name = name.replace("\\#", "#").replace("\\\\", "\\")
            label = name if name in names else "sha256:" + digest(name)
            if label in tests:
                gaps.append("duplicate top-level Node test")
            tests[label] = directive.lower() if directive else ("passed" if flag == "ok" else "failed")
        total = re.fullmatch(r"# (tests|pass|fail|cancelled|skipped|todo) (\d+)", line)
        if total:
            if total.group(1) in totals:
                gaps.append("duplicate Node TAP total")
            totals[total.group(1)] = int(total.group(2))
    result["node"] = {"top_level": tests, "totals": totals, "source_top_level": len(names),
                      "top_level_plan": plan, "expected_tests": SETTINGS_JS_TESTS}
    if (len(names) != SETTINGS_JS_TOP_LEVEL or plan != [SETTINGS_JS_TOP_LEVEL]
            or set(tests) != names or totals.get("tests") != SETTINGS_JS_TESTS
            or set(totals) != {"tests", "pass", "fail", "cancelled", "skipped", "todo"}):
        gaps.append("settings_home TAP inventory or totals differ from the pinned file")
    if (any(status != "passed" for status in tests.values()) or totals.get("pass") != totals.get("tests")
            or any(totals.get(key) for key in ("fail", "cancelled", "skipped", "todo"))):
        unmet.append("settings_home cases lack an ordinary pass")


def report():
    steps = load(PROOF / "steps.json")
    gaps, unmet = [], []
    result = {"lane": LANE, "subject": None, "execution": "not_run"}
    try:
        identity = {"lane": LANE, "driver": os.environ["GITHUB_SHA"], "run": os.environ["GITHUB_RUN_ID"],
                    "attempt": os.environ["GITHUB_RUN_ATTEMPT"], "os": RUNNER_OS}
        result["attempt"] = identity["attempt"]
        result["attempt_kind"] = "first" if identity["attempt"] == "1" else "rerun"
        if result["attempt_kind"] == "rerun":
            unmet.append("rerun evidence is not first-attempt qualification")
        if any(steps.get(key) != value for key, value in identity.items()):
            gaps.append("step receipt does not bind this lane, driver, run, attempt and OS")
        if any(steps["steps"].get(key) != "success" for key in ("checkout", "pin", "python_setup", "node_setup")):
            unmet.append("producer prerequisites incomplete")
        source = read_fact(PROOF / "source.json", gaps) or {}
        result["subject"] = source.get("head")
        if (source.get("state") != "verified" or source.get("lane") != LANE
                or source.get("driver") != identity["driver"] or steps["steps"]["producer"] in ("skipped", "")):
            unmet.append(source.get("reason") or "producer prerequisites incomplete")
        else:
            process_report(result, steps, gaps, unmet)
            phase_report(result, gaps, unmet)
            if LANE == "settings":
                node_report(result, gaps, unmet)
    except Exception as exc:  # an interrupted projection never reads as evidence
        gaps.append(f"projection interrupted: {type(exc).__name__}")
    result.update(diagnostic_gaps=sorted(set(gaps)), requirements_unmet=sorted(set(unmet)))
    result["qualified"] = not gaps and not unmet
    write(PROOF / "outcome.json", result)
    required = result.get("required") or {}
    line = (f"{LANE}: {result['execution']}; required {sum(v == 'passed' for v in required.values())}/{len(required)}"
            f" passed; cases {dict(sorted((result.get('case_counts') or {}).items()))};"
            f" gaps {len(result['diagnostic_gaps'])}; unmet {len(result['requirements_unmet'])};"
            f" qualified {str(result['qualified']).lower()}")
    print(line)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    sys.exit({"pin": pin, "run": produce, "child": child, "report": report}[sys.argv[1]]())
