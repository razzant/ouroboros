"""The Quit witness: holder evidence by exact identity, its driver on this checkout's launcher, and its cleanup.

A holder counts only by exact engine identity, path and handle.
Microsoft documents ``handle``'s flags (``-v`` is "CSV output with comma delimiter") but not its column
header or text line layout, so the samples below are synthetic shapes; the native run saves the real
output beside its verdict, and an unrecognized shape is ``unknown``, never a PASS.
"""
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from scripts import claudexor_quit_witness as witness

HOME = Path("C:/Users/runner/AppData/Local/Temp/ob/data/claudexor") if os.name == "nt" else Path("/tmp/ob/data/claudexor")
IDENTITY = {"roots": [{"pid": 7284, "birth": "133", "fingerprint": {}, "root": True}],
            "tree": [{"pid": 7284, "birth": "133", "root": True}, {"pid": 7300, "birth": "140", "root": False}]}


def _inside(name):
    return str(HOME / name)


def test_handle_text_lines_are_parsed_and_unknown_shapes_counted():
    text = "\n".join([
        f"node.exe           pid: 7284   type: File           2C4: {_inside('daemon/engine.sqlite')}",
        f"node.exe           pid: 7284   type: File  RUNNER\\runneradmin  2D0: {_inside('daemon.log')}",
        "System             pid: 4      type: Section  something odd without a handle",
        "",
    ])
    rows, unparsed = witness.parse_handle_text(text)
    assert [(row["pid"], row["handle"], row["path"]) for row in rows] == [
        (7284, "2C4", _inside("daemon/engine.sqlite")), (7284, "2D0", _inside("daemon.log"))]
    assert unparsed == 1, "a line that does not fit is counted, never guessed"


def test_handle_csv_columns_come_from_the_header_and_an_unknown_header_is_unrecognized():
    text = f"Process,PID,Type,Handle,Name\nnode.exe,7284,File,0x2C4,{_inside('daemon/engine.sqlite')}\nbad,row\n"
    assert witness.parse_handle_csv(text) == [{"process": "node.exe", "pid": 7284, "type": "File",
                                               "handle": "0X2C4", "path": _inside("daemon/engine.sqlite")}]
    assert witness.parse_handle_csv("Image,Id,Object\nnode.exe,7284,whatever\n") is None


def test_lsof_field_output_names_descriptor_and_path():
    text = "p7284\nf19\nn/tmp/ob/data/claudexor/daemon/engine.sqlite\nf20\nn/tmp/ob/data/claudexor/daemon/x\np99\nf3\nn/etc\n"
    rows = witness.parse_lsof_fields(text)
    assert [(row["pid"], row["handle"], row["path"]) for row in rows] == [
        (7284, "19", "/tmp/ob/data/claudexor/daemon/engine.sqlite"), (7284, "20", "/tmp/ob/data/claudexor/daemon/x"),
        (99, "3", "/etc")]


def test_only_an_identity_pid_with_a_handle_inside_the_home_matches():
    rows = [{"pid": 7284, "handle": "2C4", "path": _inside("daemon/engine.sqlite")},  # the engine root: matches
            {"pid": 7300, "handle": "1A", "path": _inside("daemon/engine.sqlite-wal")},  # its descendant: matches
            {"pid": 5555, "handle": "2C4", "path": _inside("daemon/engine.sqlite")},  # another process
            {"pid": 7284, "handle": "", "path": _inside("daemon/engine.sqlite")},  # no handle named
            {"pid": 7284, "handle": "2C8", "path": str(HOME.parent / "claudexor-other" / "x")}]  # outside the home
    matches = witness.match_holders(rows, IDENTITY, HOME, "handle.exe csv")
    assert [(match["pid"], match["handle"], match["root"], match["birth"]) for match in matches] == [
        (7284, "2C4", True, "133"), (7300, "1A", False, "140")]


@pytest.mark.parametrize("windows", [True, False])
@pytest.mark.parametrize("case", ["missing", "failed", "no_match", "path_only", "observed"])
def test_a_holder_verdict_is_observed_only_from_a_matching_handle_source(windows, case):
    required = ("handle.exe csv", "handle.exe text") if windows else ("lsof", "proc")
    match = {"source": required[0], "pid": 7284, "handle": "2C4", "path": _inside("x")}
    sources = {
        "missing": {},
        "failed": {required[0]: {"status": "failed", "exit": 1, "matches": []}},
        "no_match": {name: {"status": "ran", "exit": 0, "matches": []} for name in required},
        "path_only": {"psutil": {"status": "ran", "paths_inside": [_inside("x")]}},
        "observed": {required[0]: {"status": "ran", "exit": 0, "matches": [match]}},
    }[case]
    verdict = witness.holder_verdict(sources, windows)
    assert verdict["verdict"] == ("observed" if case == "observed" else "unknown")
    if case != "observed":
        assert verdict["reason"] and verdict["matches"] == []


def test_holder_sources_save_their_full_output_and_a_failed_tool_is_unknown(tmp_path, monkeypatch):
    long_output = "".join(f"line {index}\n" for index in range(500))

    def run(command, **kwargs):
        if command[0] == "lsof":
            return subprocess.CompletedProcess(command, 1, stdout=long_output, stderr="lsof: cannot read\n")
        raise AssertionError(command)

    monkeypatch.setattr(witness.shutil, "which", lambda name: "/usr/sbin/lsof" if name == "lsof" else None)
    monkeypatch.setattr(witness.subprocess, "run", run)
    monkeypatch.setattr(witness, "engine_identity", lambda data: IDENTITY)
    observed = witness.observe_holders(HOME, IDENTITY, tmp_path, "probe", "", windows=False, proc=tmp_path / "none")
    assert observed["verdict"] == "unknown" and observed["sources"]["lsof"]["status"] == "failed"
    saved = (tmp_path / observed["sources"]["lsof"]["raw"]).read_text(encoding="utf-8")
    assert "line 499" in saved and "exit 1" in saved and "cannot read" in saved, "no silent truncation"


@pytest.mark.parametrize("partial", [b"p 4242 nengine\n", "p 4242 nengine\n"])  # POSIX bytes, Windows text
def test_a_holder_command_that_timed_out_keeps_its_partial_output_and_is_never_not_run(tmp_path, monkeypatch,
                                                                                       partial):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=partial, stderr=b"lsof: slow mount\n")

    monkeypatch.setattr(witness.shutil, "which", lambda name: "/usr/sbin/lsof" if name == "lsof" else None)
    monkeypatch.setattr(witness.subprocess, "run", run)
    observed = witness.observe_holders(HOME, IDENTITY, tmp_path, "probe", "", windows=False, proc=tmp_path / "none")
    source = observed["sources"]["lsof"]
    assert observed["verdict"] == "unknown" and source["status"] == "timeout" and "TimeoutExpired" in source["error"]
    saved = (tmp_path / source["raw"]).read_text(encoding="utf-8")
    assert "(not run)" not in saved and "timed out after 300s" in saved, saved
    assert "p 4242" in saved and "slow mount" in saved, "the started command's partial streams are kept"


def test_the_lifecycle_job_keeps_the_witness_and_its_base_run_after_an_earlier_red_step():
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claudexor-platform-gate.yml"
    steps = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["lifecycle"]["steps"]
    named = {step.get("name", ""): step for step in steps}
    candidate = next(step for name, step in named.items() if name.startswith("Quit releases"))
    base = next(step for name, step in named.items() if name.startswith("The same scenario on the named base"))
    assert "!cancelled()" in candidate["if"] and "!cancelled()" in base["if"]
    assert "--evidence-dir" in candidate["run"] and "--evidence-dir" in base["run"]
    assert "--expect hold" in base["run"] and "--expect release" in candidate["run"]
    upload = next(step for step in steps if "upload-artifact" in str(step.get("uses", "")))
    assert upload["if"] == "always()"


def test_the_gate_reaches_the_launcher_modules_and_windows_runs_the_real_job_test_without_a_skip(tmp_path):
    """Every launcher module the Quit touches triggers the gate; on Windows the lifecycle job runs the native real-Job
    retirement test, whose two cases must pass: a skip or a missing case is red, never a native PASS."""
    import fnmatch

    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github" / "workflows"
                               / "claudexor-platform-gate.yml").read_text(encoding="utf-8"))
    for event in ("push", "pull_request"):
        paths = workflow[True][event]["paths"]  # YAML 1.1 reads the key ``on`` as True
        for module in ("launcher.py", "ouroboros/launcher_background.py", "ouroboros/launcher_tray_macos.py",
                       "ouroboros/launcher_onboarding.py", "ouroboros/launcher_server_reaper.py"):
            assert any(fnmatch.fnmatchcase(module, pattern) for pattern in paths), (event, module)
    steps = workflow["jobs"]["lifecycle"]["steps"]
    native = next(step for step in steps if "test_windows_retirement_returns_only_after_its_real_job_members_are_gone"
                  in step.get("run", ""))
    assert native["if"] == "${{ !cancelled() && matrix.os == 'windows-latest' }}"
    assert '"$py" -I -S scripts/safe_test.py' in native["run"] and "--junitxml" in native["run"]
    upload = next(step for step in steps if "upload-artifact" in str(step.get("uses", "")))
    assert "quit-native-tests" in upload["with"]["path"]
    check = native["run"].split("<<'PY'\n", 1)[1].rsplit("PY", 1)[0]
    outcomes = {}
    for name, cases in {"passed": ["", ""], "skipped": ["<skipped/>", "<skipped/>"], "one": [""],
                        "failed": ["", "<failure/>"]}.items():
        junit = tmp_path / f"{name}.xml"
        junit.write_text("<testsuites><testsuite>" + "".join(
            f'<testcase name="c{i}">{body}</testcase>' for i, body in enumerate(cases)) + "</testsuite></testsuites>",
            encoding="utf-8")
        outcomes[name] = subprocess.run([sys.executable, "-c", check, str(junit)], capture_output=True).returncode
    assert outcomes == {"passed": 0, "skipped": 1, "one": 1, "failed": 1}


# Handle 5.0 as https://download.sysinternals.com/files/Handle.zip served it on 2026-11-14 (Last-Modified 2022-10-26);
# its handle64.exe is byte-identical to https://live.sysinternals.com/handle64.exe.
HANDLE_SHA256 = "279aaf8eccb6f79147f4dcc6ba091fb895c4cb8b199a0dd186a4c76bc519d2cd"


def test_the_windows_holder_tool_is_the_pinned_official_archive_checked_before_extraction():
    """The official Handle URL is unversioned: the archive's bytes are pinned, and a mismatch throws before
    ``Expand-Archive``, so the witness never runs another handle.exe."""
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claudexor-platform-gate.yml"
    steps = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["lifecycle"]["steps"]
    handle = next(step for step in steps if step.get("name", "").startswith("Sysinternals handle.exe"))
    script = handle["run"]
    assert handle["shell"] == "pwsh" and handle["env"] == {"HANDLE_SHA256": HANDLE_SHA256}
    assert "-Uri 'https://download.sysinternals.com/files/Handle.zip'" in script
    order = [script.index(token) for token in ("Invoke-WebRequest", "Get-FileHash", "throw", "Expand-Archive")]
    assert order == sorted(order), script


def test_every_substituted_name_is_one_the_real_lifecycle_loop_calls():
    """The witness's stand-ins bind to the checkout's planned Restart: a renamed launcher name fails here, not as a
    real bundle sync in the native run (the #3196 rename did exactly that to the former witness)."""
    import launcher

    called = set(launcher.agent_lifecycle_loop.__code__.co_names)
    assert set(witness.SEAMS) <= called, set(witness.SEAMS) - called
    assert all(callable(getattr(launcher, name)) for name in witness.SEAMS)


_STUB_SERVER = r'''
import json, os, pathlib, sys, threading
root, pid = pathlib.Path(os.environ["OUROBOROS_WITNESS_ROOT"]), os.getpid()
(root / "generations" / f"{pid}.json").write_text(json.dumps({"pid": pid}))
stop = threading.Event()
def pipe():
    for line in sys.stdin:
        if line.strip() == "quit":
            break
    stop.set()
threading.Thread(target=pipe, daemon=True).start()
while not stop.wait(0.1):
    if (root / "control" / f"{pid}.restart").exists():
        sys.exit(42)
'''


def _driver(tmp_path, prefix=""):
    root, stub = tmp_path / "root", tmp_path / "stub"
    for directory in (root / "generations", root / "control", stub, tmp_path / "data"):
        directory.mkdir(parents=True, exist_ok=True)
    (stub / "server.py").write_text(_STUB_SERVER, encoding="utf-8")
    repo = Path(witness.REPO_ROOT)
    code = prefix + witness._LAUNCHER
    log = (tmp_path / "launcher-process.log").open("w", encoding="utf-8")
    driver = subprocess.Popen([sys.executable, "-u", "-c", code, str(root), str(repo), str(stub)], cwd=str(repo),
                              env={**os.environ, "PYTHONPATH": str(repo), "OUROBOROS_WITNESS_ROOT": str(root),
                                   "OUROBOROS_DATA_DIR": str(tmp_path / "data")},
                              stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
    return driver, root


def _generations(root, count, driver, seconds=60):
    deadline = time.monotonic() + seconds
    while len(found := sorted(root.glob("generations/*.json"))) < count:
        assert driver.poll() is None and time.monotonic() < deadline, (driver.poll(), found)
        time.sleep(0.1)
    return [json.loads(path.read_text(encoding="utf-8"))["pid"] for path in found]


def test_the_witness_launcher_passes_recycle_and_planned_restart_without_a_real_sync(tmp_path):
    """No engine: the witness's own driver on this checkout's launcher. A settings recycle and a planned Restart
    (exit 42) each start a new generation, and the planned Restart calls each stand-in once."""
    driver, root = _driver(tmp_path)
    try:
        first = _generations(root, 1, driver)
        driver.stdin.write("recycle\n")
        driver.stdin.flush()
        second = _generations(root, 2, driver)
        (root / "control" / f"{(set(second) - set(first)).pop()}.restart").touch()
        third = _generations(root, 3, driver)
        driver.stdin.write("quit\n")
        driver.stdin.flush()
        assert driver.wait(120) == 0
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait(30)
    assert len(set(third)) == 3
    assert json.loads((root / "seam-calls.json").read_text(encoding="utf-8")) == dict.fromkeys(witness.SEAMS, 1)
    logged = (tmp_path / "data" / "logs" / "launcher.log").read_text(encoding="utf-8")
    assert "Agent requested restart (exit code 42)" in logged and "manifest is missing" not in logged


def test_a_launcher_without_a_substituted_name_is_refused_before_any_generation(tmp_path):
    driver, root = _driver(tmp_path, prefix="import launcher\ndel launcher.install_deps\n")
    assert driver.wait(120) != 0
    assert not list(root.glob("generations/*.json"))
    assert "launcher seams missing" in (tmp_path / "launcher-process.log").read_text(encoding="utf-8")


def test_cleanup_reaches_every_root_and_process_after_a_failed_step(tmp_path, monkeypatch):
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    stopped = []

    def ledger_stop(root, env, out, label):
        stopped.append(root.name)
        if root.name == "owned":
            raise subprocess.TimeoutExpired("stop", 120)
        return {"exit": 0}

    monkeypatch.setattr(witness, "_ledger_stop", ledger_stop)
    monkeypatch.setattr(witness, "_remaining", lambda root, captured: [] if root.name == "foreign" else [(7, "b")])
    state = {"roots": {"owned": tmp_path / "owned", "foreign": tmp_path / "foreign"}, "identities": {},
             "processes": [("foreign", sleeper)]}
    result = witness.cleanup(state, tmp_path, {})
    assert stopped == ["owned", "foreign"] and sleeper.poll() is not None
    assert result["clean"] is False and result["remaining"] == {"owned": [(7, "b")], "foreign": []}
    assert result["steps"]["ledger_stop:owned"]["ok"] is False and "TimeoutExpired" in \
        result["steps"]["ledger_stop:owned"]["error"]


def test_the_primary_failure_survives_a_failed_cleanup(tmp_path, monkeypatch):
    def failed(repo, expect, quit_seconds, out, handle_exe, facts, state):
        state["roots"] = {"owned": tmp_path / "owned"}
        facts["doors"] = {"first": "partial"}
        raise witness.SmokeFailure("generation_not_ready", "no generation")

    monkeypatch.setattr(witness, "isolate", lambda: None)
    monkeypatch.setattr(witness, "witness", failed)
    monkeypatch.setattr(witness, "_ledger_stop", lambda root, env, out, label: 1 / 0)
    monkeypatch.setattr(witness, "_remaining", lambda root, captured: [])
    assert witness.main(["--evidence-dir", str(tmp_path / "out")]) == 1
    result = json.loads((tmp_path / "out" / "witness.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "FAILED" and result["reason"] == "generation_not_ready"
    assert result["doors"] == {"first": "partial"} and result["cleanup"]["clean"] is False


def _ledger_run(monkeypatch, outcome):
    def run(command, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return subprocess.CompletedProcess(command, outcome[0], stdout=outcome[1], stderr=outcome[2])
    monkeypatch.setattr(witness.subprocess, "run", run)
    monkeypatch.setattr(witness, "_remaining", lambda root, captured: [])


def test_a_nonzero_ledger_stop_is_a_failed_step_with_its_full_output(tmp_path, monkeypatch):
    long_out = b"x" * 5000 + b"END"
    _ledger_run(monkeypatch, (3, long_out, b"Traceback: boom"))
    result = witness.cleanup({"roots": {"owned": tmp_path / "owned"}, "identities": {}}, tmp_path, {})
    step = result["steps"]["ledger_stop:owned"]
    assert step["ok"] is False and step["error"] == "ledger stop exited 3" and result["clean"] is False
    assert step["result"]["exit"] == 3 and step["result"]["stdout"] == long_out.decode()
    assert Path(step["result"]["stdout_file"]).read_bytes() == long_out
    assert Path(step["result"]["stderr_file"]).read_bytes() == b"Traceback: boom"


def test_a_timed_out_ledger_stop_keeps_its_partial_output(tmp_path, monkeypatch):
    _ledger_run(monkeypatch, subprocess.TimeoutExpired("stop", 120, output=b"partial out", stderr=b"partial err"))
    result = witness.cleanup({"roots": {"owned": tmp_path / "owned"}, "identities": {}}, tmp_path, {})
    step = result["steps"]["ledger_stop:owned"]
    assert step["ok"] is False and step["error"].startswith("TimeoutExpired") and result["clean"] is False
    assert step["result"] == {"exit": None, "stdout": "partial out", "stderr": "partial err",
                              "stdout_file": str(tmp_path / "cleanup-ledger-stop-owned.stdout"),
                              "stderr_file": str(tmp_path / "cleanup-ledger-stop-owned.stderr")}


def test_a_failed_ledger_stop_of_one_root_does_not_skip_the_next(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs["env"]["OUROBOROS_DATA_DIR"])
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0, stdout=b"[]\n[]\n", stderr=b"")
    monkeypatch.setattr(witness.subprocess, "run", run)
    monkeypatch.setattr(witness, "_remaining", lambda root, captured: [])
    roots = {"owned": tmp_path / "owned", "foreign": tmp_path / "foreign"}
    result = witness.cleanup({"roots": roots, "identities": {}}, tmp_path, {})
    assert calls == [str(roots["owned"]), str(roots["foreign"])]
    assert result["steps"]["ledger_stop:owned"]["ok"] is False and result["steps"]["ledger_stop:foreign"]["ok"]
    assert result["clean"] is False and result["remaining"] == {"owned": [], "foreign": []}


def test_unreadable_or_unmeasured_processes_are_never_clean(tmp_path, monkeypatch):
    from ouroboros import platform_layer, process_custody

    live, dead = os.getpid(), subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    born = platform_layer.process_start_time(live)
    captured = [{"roots": [], "tree": [
        {"pid": live, "birth": None, "root": False},  # never measured: unknown, not gone
        {"pid": live, "birth": born, "root": False},  # alive as itself
        {"pid": dead.pid, "birth": "1", "root": False}]}]  # gone
    monkeypatch.setattr(process_custody, "process_stop_snapshot", lambda root, purposes: [
        {"pid": live, "fingerprint": {}}])  # live ledger row whose identity cannot be measured
    monkeypatch.setattr(witness, "engine_identity", lambda data: {"roots": [], "tree": []})
    rows = witness._remaining(tmp_path, captured)
    assert (live, born) in rows and (live, "unknown: None") in rows
    assert (live, "unknown: ledger row without measured identity") in rows and all(pid != dead.pid for pid, _ in rows)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda pid: "")
    assert witness._liveness({"pid": live, "birth": born, "root": False}) == "unknown"
    assert witness._gone([{"pid": live, "birth": born, "root": False}]) is False

    def unreadable(root, purposes):
        raise OSError("process custody ledger is unreadable or corrupt")
    monkeypatch.setattr(process_custody, "process_stop_snapshot", unreadable)
    monkeypatch.setattr(witness, "_ledger_stop", lambda root, env, out, label: {"exit": 0})
    result = witness.cleanup({"roots": {"owned": tmp_path}, "identities": {}}, tmp_path, {})
    assert result["remaining"] == {"owned": "unknown"} and result["clean"] is False


def test_a_crashing_cleanup_keeps_the_primary_verdict(tmp_path, monkeypatch):
    def failed(repo, expect, quit_seconds, out, handle_exe, facts, state):
        raise witness.SmokeFailure("engine_home_still_held_after_quit", "held")

    monkeypatch.setattr(witness, "isolate", lambda: None)
    monkeypatch.setattr(witness, "witness", failed)
    monkeypatch.setattr(witness, "cleanup", lambda state, out, env: 1 / 0)
    assert witness.main(["--evidence-dir", str(tmp_path / "out")]) == 1
    result = json.loads((tmp_path / "out" / "witness.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "FAILED" and result["reason"] == "engine_home_still_held_after_quit"
    assert result["cleanup"]["clean"] is False and "ZeroDivisionError" in result["cleanup"]["error"]


def test_an_empty_lsof_is_a_source_that_ran_and_found_nothing(tmp_path, monkeypatch):
    """lsof exits 1 with no output when nothing under the folder is open: the after-Quit check ran, unlike a failure."""
    outputs = {"empty": ("", ""), "error": ("", "lsof: WARNING: can't stat()\n")}
    for case, (stdout, stderr) in outputs.items():
        monkeypatch.setattr(witness.shutil, "which", lambda name: "/usr/sbin/lsof")
        monkeypatch.setattr(witness.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(
            command, 1, stdout=stdout, stderr=stderr))
        observed = witness.observe_holders(HOME, IDENTITY, tmp_path, case, "", windows=False, proc=tmp_path / "none",
                                           still=lambda: True)
        assert observed["sources"]["lsof"]["status"] == ("ran" if case == "empty" else "failed")
        assert observed["verdict"] == "unknown" and observed["matches"] == []


@pytest.mark.parametrize("case", ["no match, exit 1", "no match, exit 0", "other failure", "rows and no-match words"])
def test_handle_exe_saying_no_match_is_a_source_that_ran_and_found_nothing(tmp_path, monkeypatch, case):
    """handle.exe answers a folder nothing holds with "No matching handles found." and may exit nonzero: the after-Quit
    check ran and found none. Any other nonzero exit without PID rows stays a failed source, never an empty one."""
    row = f"node.exe pid: 7284 type: File 1A0: {_inside('state.db')}\n"
    stdout, code = {"no match, exit 1": ("No matching handles found.\n", 1),
                    "no match, exit 0": ("No matching handles found.\n", 0),
                    "other failure": ("Initialization error: Make sure that you are an administrator.\n", 1),
                    "rows and no-match words": (row + "No matching handles found.\n", 1)}[case]
    monkeypatch.setattr(witness.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(
        command, code, stdout=stdout, stderr=""))
    observed = witness.observe_holders(HOME, IDENTITY, tmp_path, "after", "C:/tools/handle64.exe", windows=True,
                                       still=lambda: True)
    sources = [observed["sources"][name] for name in ("handle.exe csv", "handle.exe text")]
    if case == "other failure":
        assert [source["status"] for source in sources] == ["failed", "failed"]
        assert observed["reason"].startswith("no holder source ran"), "a failure is never an empty observation"
    elif case == "rows and no-match words":
        assert sources[1]["status"] == "ran" and sources[1]["format"] == "parsed" and observed["verdict"] == "observed"
    else:
        assert [source["status"] for source in sources] == ["ran", "ran"] and observed["matches"] == []
        assert sources[0]["format"] == "no match" and sources[0]["exit"] == code
        assert observed["verdict"] == "unknown" and not observed["reason"].startswith("no holder source ran")
        assert "No matching handles found" in (tmp_path / sources[1]["raw"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("outcome", ["pass", "failed", "unclean", "evidence inside"])
def test_only_a_clean_pass_removes_its_isolated_root_and_the_result_records_it(tmp_path, monkeypatch, outcome):
    root = tmp_path / "cx-root"
    (root / "data" / "logs").mkdir(parents=True)
    (root / "data" / "logs" / "launcher.log").write_text("diagnosis\n", encoding="utf-8")
    out = root / "evidence" if outcome == "evidence inside" else tmp_path / "out"

    def run(repo, expect, quit_seconds, out, handle_exe, facts, state):
        if outcome == "failed":
            raise witness.SmokeFailure("engine_home_still_held_after_quit", "held")

    monkeypatch.setattr(witness, "isolate", lambda: root)
    monkeypatch.setattr(witness, "witness", run)
    monkeypatch.setattr(witness, "cleanup", lambda state, out, env: {
        "clean": outcome != "unclean", "remaining": {}, "steps": {}})
    code = witness.main(["--evidence-dir", str(out)])
    result = json.loads((out / "witness.json").read_text(encoding="utf-8"))
    released = result["cleanup"]["temp_root"]
    assert released["path"] == str(root)
    if outcome == "pass":
        assert code == 0 and released["removed"] is True and not root.exists() and result["cleanup"]["clean"]
    else:
        assert released["removed"] is False and released["kept"] and root.exists()
        assert (root / "data" / "logs" / "launcher.log").is_file(), "a failure keeps its diagnosis"
        assert code == (0 if outcome == "evidence inside" else 1)


def test_a_root_that_cannot_be_removed_after_a_pass_is_recorded_and_unclean(tmp_path, monkeypatch):
    root = tmp_path / "cx-root"
    root.mkdir()
    monkeypatch.setattr(witness, "isolate", lambda: root)
    monkeypatch.setattr(witness, "witness", lambda *args: None)
    monkeypatch.setattr(witness, "cleanup", lambda state, out, env: {"clean": True, "remaining": {}, "steps": {}})
    monkeypatch.setattr(witness, "force_rmtree", lambda path: None)  # a holder kept it
    assert witness.main(["--evidence-dir", str(tmp_path / "out")]) == 1
    result = json.loads((tmp_path / "out" / "witness.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "PASSED" and result["cleanup"]["clean"] is False
    assert result["cleanup"]["temp_root"]["error"] == "still present" and root.exists()


def test_a_passed_witness_removes_a_root_whose_git_objects_are_read_only(tmp_path, monkeypatch):
    """The root's fixture repositories keep read-only loose objects, and Windows refuses to delete a read-only file
    (emulated here on every platform): a clean PASS still removes the whole root."""
    root = tmp_path / "cx-root"
    blob = root / "project" / ".git" / "objects" / "ab" / "cdef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"blob")
    blob.chmod(0o444)
    real_unlink = os.unlink

    def windows_unlink(path, *, dir_fd=None):
        if not os.stat(path, dir_fd=dir_fd, follow_symlinks=False).st_mode & stat.S_IWRITE:
            raise PermissionError(13, "Access is denied", path)
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", windows_unlink)
    monkeypatch.setattr(os, "remove", windows_unlink)
    with pytest.raises(PermissionError):
        windows_unlink(blob)
    assert witness.release_root(root, tmp_path / "out", True) == {"path": str(root), "removed": True}
    assert not root.exists()
