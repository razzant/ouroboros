"""Process/evidence plumbing for the opt-in S1 real consumer qualification.

Reuse the system_e2e keyless HTTP model/server, candidate capture and native
process container. No lifecycle owners are mocked. See the companion tests for
the boundary each assertion claims. Runtime evidence stays OUTSIDE the source.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import uuid

from devtools.benchmarks.common.server_runner import _api, _api_status
from ouroboros.process_containment import ProcessContainer
from ouroboros.test_environment import retain_tree
from tests.candidate_checkout import (
    assert_served_candidate, candidate_checkout, require_candidate_interpreter,
)
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ScriptedStubModel,
    classify_call, clone_repo, keyless_review_rows, keyless_settings,
    message_text, write_settings_file,
)

SOURCE = Path(__file__).resolve().parents[1]
HOST = SOURCE / "tests/s1_lifecycle_server_host.py"
PRODUCT = ("server.py", "launcher.py", "ouroboros", "supervisor", "prompts", "web", "VERSION")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def wait_for(read, label, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := read():
            return value
        time.sleep(.1)
    raise AssertionError(f"{label} not observed within {timeout}s")


def watch_no_change(read, expected, label, seconds=4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        observed = read()
        assert observed == expected, (label, expected, observed)
        time.sleep(.1)


def product_hashes(root):
    names = subprocess.check_output(["git", "ls-files", "-z", "--", *PRODUCT], cwd=SOURCE).split(b"\0")
    return {os.fsdecode(name): hashlib.sha256((Path(root) / os.fsdecode(name)).read_bytes()).hexdigest()
            for name in names if name}


class SavedReadModel(ScriptedStubModel):
    """Real loopback wire; only the external model is scripted.

    A nonce lives solely in a disposable input file. It must enter cognition
    through a REAL read_file result. Hold the next request indefinitely enough
    to exercise the stop. A successor request is captured separately while the
    old request remains unanswered. Never fabricate checkpoint/retry state.
    """

    def __init__(self, workspace, evidence, *, tree=False):
        self.evidence = evidence
        self.tree = tree
        self.tokens = {key: "S1_SAVED_" + uuid.uuid4().hex for key in ("root", "child", "prior")}
        for key, token in self.tokens.items():
            (workspace / f"probe-{key}.txt").write_text(token + "\n", encoding="utf-8")
        self.ready = {key: [] for key in (*self.tokens, "queued")}
        self.release_requests = threading.Event()
        self.capture_lock = threading.Lock()
        self.timeouts = []
        super().__init__(gate=self._gate, model_ids=["mock-model", "mock-child"])

    def _key(self, body):
        if not body.get("tools") or classify_call(body) != "agent":
            return ""
        if body.get("model") == "mock-child":
            return "child"
        # The root's later transcript QUOTES its child's objective. Bind the
        # child by its separate wire model; do not let that quote rename root.
        # Only the task's own labelled input selects its script. A later roster
        # quotes other tasks' titles and must not reclassify this request.
        for message in body.get("messages", []):
            if message.get("role") != "user":
                continue
            text = message_text(message)
            for key in ("root", "prior", "queued"):
                if text.startswith(f"[E2E-LINEAGE:{key}]"):
                    return key
        return ""

    def _ready(self, body, key):
        if key == "queued":
            return True
        results = "\n".join(message_text(m) for m in body.get("messages", []) if m.get("role") == "tool")
        return key in self.tokens and self.tokens[key] in results and (
            key != "root" or not self.tree or "Subagent request queued" in results)

    def _gate(self, body):
        key = self._key(body)
        if not self._ready(body, key):
            return
        with self.capture_lock:
            self.ready[key].append(body)
            index = len(self.ready[key])
            write_json(self.evidence / f"model-{key}-{index}.json", body)
        if key == "queued":
            return  # A fresh queued task finishes; it must not starve restored workers.
        if not self.release_requests.wait(600):
            self.timeouts.append(key)
            raise TimeoutError(f"S1 fixture never released {key} request {index}")

    def _answer(self, body, seq):
        key = self._key(body)
        if key not in self.tokens or self._ready(body, key):
            return "final", {"role": "assistant", "content": "S1 scripted transport finished."}
        results = "\n".join(message_text(m) for m in body.get("messages", []) if m.get("role") == "tool")
        if self.tokens[key] not in results:
            tool, args = "read_file", {"path": f"probe-{key}.txt"}
        else:
            tool, args = "schedule_subagent", {
                "subagent_id": "mock-child",
                "objective": "[E2E-LINEAGE:child] Read probe-child.txt in the inherited workspace; retain its content.",
                "expected_output": "The exact file content.",
            }
        return "agent", {"role": "assistant", "content": "Reading the test input.", "tool_calls": [{
            "id": f"s1-{key}-{seq}", "type": "function",
            "function": {"name": tool, "arguments": json.dumps(args)},
        }]}

    def _completion(self, body):
        result = super()._completion(body)
        result["usage"]["cost"] = .01  # provider-reported completed requests; held requests remain unknown
        return result

    def count(self, key):
        with self.capture_lock:
            return len(self.ready[key])


class RecordedServer(KeylessIsolatedServer):
    """Only startup logging/custody differs from the existing server harness."""

    def __init__(self, clone, data, settings, evidence, injection):
        super().__init__(clone, data, settings)
        self.evidence, self.injection = evidence, injection
        self.container = None
        self.log_stream = None

    def start(self, ready_timeout=150):
        assert self.container is None
        self._patch_settings_ports()
        env = self._env()
        env.update({
            "OUROBOROS_DISABLE_MANAGED_UPDATES": "1",  # existing pin: Restart must not reset the candidate
            "OUROBOROS_LAUNCHER_STOP_STDIN": "1",       # actual launcher Quit consumer, also callable on macOS
            "PIP_NO_INDEX": "1", "S1_LIFECYCLE_EVIDENCE": str(self.evidence),
            "S1_LIFECYCLE_INJECTION": str(self.injection),
            "PYTHONPATH": str(self.injection),
        })
        require_candidate_interpreter(sys.executable, env, self.clone)
        self.container = ProcessContainer()
        self.log_stream = (self.evidence / "server.log").open("ab", buffering=0)
        if self.candidate:
            self.candidate.hold()
        try:
            self.proc = self.container.spawn(
                [sys.executable, str(HOST)], cwd=self.clone, env=env, stdin=subprocess.PIPE,
                stdout=self.log_stream, stderr=subprocess.STDOUT)
            self._wait_ready(ready_timeout)
            self.prove_served()
        except BaseException:
            self.stop()
            raise
        return self

    def prove_served(self):
        from ouroboros.server_process import read_service_bindings

        boots = jsonl(self.evidence / "boots.jsonl")
        assert boots and boots[-1]["pid"] == self.proc.pid
        binding = read_service_bindings(self.data_root)["main"]
        assert binding["pid"] == self.proc.pid and binding["port"] == self.port, binding
        assert Path(boots[-1]["cwd"]) == self.clone.resolve()
        assert boots[-1]["roots"]["OUROBOROS_DATA_DIR"] == str(self.data_root)
        assert boots[-1]["server_sha256"] == hashlib.sha256((SOURCE / "server.py").read_bytes()).hexdigest()
        if self.candidate:
            assert_served_candidate(self.base_url, self.clone, self.data_root, self.proc.pid, self.candidate)
        else:
            assert product_hashes(self.clone) == product_hashes(SOURCE)
        write_json(self.evidence / f"served-{len(boots)}.json", {
            "boot": boots[-1], "binding": binding, "health": _api(self.base_url, "GET", "/api/health"),
            "candidate_identity": self.candidate.identity if self.candidate else "clean-product-at-HEAD",
        })

    def quit(self):
        assert self.proc.poll() is None
        self.proc.stdin.write(b"quit\n")
        self.proc.stdin.flush()
        return self.proc.wait(timeout=60)

    def stop(self):
        container, self.container = self.container, None
        if container is None:
            return
        error, collected = "", False
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    pass  # native container makes one bounded kill sweep next
        finally:
            try:
                error = container.reap()
                if self.proc is not None:
                    self.proc.wait(timeout=10)
                collected = True
            finally:
                container.close()
                if self.log_stream:
                    self.log_stream.close()
                    self.log_stream = None
                with (self.evidence / "cleanup.jsonl").open("a") as stream:
                    stream.write(json.dumps({"pid": self.proc.pid if self.proc else None,
                                             "exit": self.proc.returncode if self.proc else None,
                                             "container_error": error, "root_collected": collected}) + "\n")
                if not collected:
                    retain_tree(self.data_root.parent, "Process collection raised; see original failure")
            if error:
                retain_tree(self.data_root.parent, error)
                raise AssertionError(f"S1 process custody unconfirmed: {error}")
            if self.candidate:
                self.candidate.release()


def archive_evidence(server, evidence):
    """Small named originals only. Never walk/copy the runtime, cache or repo."""
    named = [
        "logs/server.log", "logs/supervisor.jsonl", "logs/events.jsonl", "logs/progress.jsonl",
        "logs/tools.jsonl", "logs/llm_usage.jsonl", "state/queue_snapshot.json", "state/state.json",
        "state/worker_pids.json", "state/server_port.bindings.json", "state/panic_stop.flag",
    ]
    for rel in named:
        path = server.data_root / rel
        if path.is_file():
            destination = evidence / "originals" / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    txdir = server.data_root / "state/delegate_recovery_transactions"
    for path in sorted(txdir.glob("*.json")):
        target = evidence / "originals/transactions" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    paths = [p for p in evidence.rglob("*") if p.is_file()]
    write_json(evidence / "manifest.json", [{
        "path": str(p.relative_to(evidence)), "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    } for p in sorted(paths)])
    archive = evidence.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(evidence.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=str(path.relative_to(evidence)), recursive=False)
    print(f"S1_LIFECYCLE_EVIDENCE {archive}", flush=True)


@contextmanager
def scenario(root, *, tree=False, managed=False):
    root = Path(root)
    evidence, workspace, injection = (root / name for name in ("evidence", "work", "inject"))
    for path in (evidence, workspace, injection):
        path.mkdir()
    shutil.copyfile(SOURCE / "tests/s1_lifecycle_exit_injector.py", injection / "sitecustomize.py")
    write_json(evidence / "source.json", {
        "source": str(SOURCE), "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
        "product_sha256": product_hashes(SOURCE),
        "test_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((SOURCE / "tests").glob("*s1_lifecycle*.py"))},
    })

    @contextmanager
    def checkout():
        if managed:
            # Managed replacement intentionally resets its disposable clone. It
            # must not quietly substitute HEAD for uncommitted candidate code.
            clone = clone_repo(root)
            assert product_hashes(clone) == product_hashes(SOURCE), (
                "Managed replacement requires product bytes frozen at HEAD; provide a host-owned candidate snapshot")
            branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=clone, text=True).strip()
            # A real newer local target with identical product bytes exercises
            # replace/update. The gateway correctly refuses an already-current SHA.
            upstream = root / "managed-source"
            subprocess.run(["git", "clone", "--local", "--no-hardlinks", str(clone), str(upstream)], check=True)
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "commit", "--allow-empty", "-m", "Test-only newer update target"], cwd=upstream, check=True)
            subprocess.run(["git", "remote", "add", "managed", str(upstream)], cwd=clone, check=True)
            subprocess.run(["git", "fetch", "managed", branch], cwd=clone, check=True)
            write_json(clone / ".git/ouroboros-managed.json", {
                "managed_remote_name": "managed", "managed_remote_url": str(upstream),
                "managed_remote_branch": branch, "managed_local_branch": branch,
            })
            yield clone
        else:
            with candidate_checkout(SOURCE, root / "candidate", origin_proof=True) as candidate:
                yield candidate

    with checkout() as clone, SavedReadModel(workspace, evidence, tree=tree) as model:
        data = root / "data"
        data.mkdir()
        settings = keyless_settings(model, OUROBOROS_MAX_WORKERS=2 if tree else 1,
            OUROBOROS_UPDATE_CHANNEL="development", OUROBOROS_SUBAGENT_WORKTREE_ROOT=str(root / "worktrees"),
            # A scenario passing its own catalog owns its review pool: keep the keyless
            # Reviewer rows beside the delegated child row, as every keyless server has.
            OUROBOROS_SUBAGENTS=json.dumps({"enabled": True, "items": [{
                "subagent_id": "mock-child", "recommended_use": "Read the inherited fixture input.",
                "route": {"kind": "api_model", "target_id": "openai-compatible::mock-child"}, "effort": "low",
            }, *keyless_review_rows()]}))
        write_settings_file(data / "settings.json", settings)
        server = RecordedServer(clone, data, data / "settings.json", evidence, injection)
        try:
            server.start()
            yield server, model, workspace, evidence
        except BaseException:
            (evidence / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            # Keep old HTTP answers withheld UNTIL all owned processes are gone.
            # Releasing them before Quit would turn an abandoned request into a
            # completed request and conceal the lifecycle window being tested.
            try:
                server.stop()
            finally:
                model.release_requests.set()
                archive_evidence(server, evidence)


def submit(server, workspace, key):
    result = _api(server.base_url, "POST", "/api/tasks", {
        "description": f"[E2E-LINEAGE:{key}] Read probe-{key}.txt and retain its exact content.",
        "memory_mode": "forked", "workspace_mode": "external", "workspace_root": str(workspace),
        "source": "cli", "metadata": {"source": "cli", "delegation_role": "root"},
    })
    assert result.get("task_id"), result
    return result["task_id"]


def request(server, route, payload=None):
    result = _api_status(server.base_url, "POST", route, payload or {}, timeout=180)
    assert 200 <= result["status"] < 300, result
    return result["body"]


def checkpoint(server, task_id, attempt=1):
    from ouroboros.working_checkpoint import checkpoint_path

    path = checkpoint_path(server.data_root, task_id, attempt)
    return wait_for(lambda: read_json(path), f"checkpoint {task_id}/{attempt}")


def read_calls(server, task_id):
    return [r for r in ArtifactOracle(server.data_root).tools_rows()
            if r.get("task_id") == task_id and r.get("type") == "tool_call"
            and (r.get("tool") or r.get("name")) == "read_file"]
