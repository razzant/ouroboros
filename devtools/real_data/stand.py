"""The real-data stand: does an ordinary owner message reach its model on a copy of real history?

The unit cost contract (``tests/test_cost_contract.py``) counts work on synthetic shapes. It
missed 22.10 because no gate saw data shaped like the owner's. The stand boots a chosen body
(any commit) against a contained copy of an install's data root (``snapshot``), with a
loopback stub model and no Claudexor daemon, inside a Seatbelt profile proven by canaries
(``sandbox``), then sends ordinary owner messages at fixed offsets after readiness and
records, per message: accepted, task received, first model dispatch, visible reply. It
samples the server's memory, CPU and threads every 10 s and every thread's frames every 15 s.

Arms run one at a time. An arm that cannot be contained does not boot. An unfinished turn is
reported as censored, never as zero. Nothing here judges a body: the report is facts for the
person or the mind comparing arms.

    python -m devtools.real_data.stand snapshot --live /Users/<you>/Ouroboros/data --out STAND
    python -m devtools.real_data.stand arm --stand STAND --source <git clone> --rev <sha> --label A \
        --sends 30,120,240,360 --window 420

Size and remover: an arm is a clone of the snapshot (shared blocks) plus what it writes,
bounded by ``--window`` and ``--memory-cap-gib``; ``arm`` prints its directory. Deleting a stand
directory removes everything it created; for a large tree use a paced, cancellable delete.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.benchmarks.common.server_runner import IsolatedServer, _api, supervisor_state_is_ready  # noqa: E402
from devtools.real_data import sandbox, snapshot  # noqa: E402
from ouroboros.platform_layer import subprocess_new_group_kwargs  # noqa: E402

PROBE_DIR = pathlib.Path(__file__).resolve().parent / "probe"
MARKER_RE = re.compile(r"STAND-[A-Za-z0-9]+-T\d+-[0-9a-f]{6}")
SAMPLE_EVERY_SEC = 10.0


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------- the stub model


def stub_model():
    """A loopback OpenAI-compatible model that answers an owner turn with ``ACK <marker>``."""
    from tests.system_e2e.harness import LoopbackModelServer, body_text, canned_review_answer, classify_call

    class StandStub(LoopbackModelServer):
        def __init__(self) -> None:
            super().__init__()
            self.arrivals: List[Dict[str, Any]] = []

        def _answer(self, body: dict, seq: int):
            kind = classify_call(body)
            markers = MARKER_RE.findall(body_text(body))
            last = markers[-1] if markers else ""
            self.arrivals.append({"t": _now(), "kind": kind, "marker": last, "tools": bool(body.get("tools"))})
            canned = canned_review_answer(kind)
            if canned is not None:
                return kind, canned
            if last:
                return "turn", {"role": "assistant", "content": f"ACK {last}"}
            return kind, {"role": "assistant", "content": "Nothing to add."}

    return StandStub()


# ---------------------------------------------------------------- the contained server


class SandboxedServer(IsolatedServer):
    """``IsolatedServer`` started under the arm's Seatbelt profile, with the frame probe."""

    def __init__(self, clone, data_root, settings_path, *, arm_root: pathlib.Path, profile_path: pathlib.Path):
        super().__init__(clone, data_root, settings_path)
        self.arm_root = arm_root
        self.profile_path = profile_path

    def _env(self) -> dict:
        env = super()._env()
        (self.arm_root / "tmp").mkdir(exist_ok=True)
        (self.arm_root / "probe").mkdir(exist_ok=True)
        env.update({"TMPDIR": str(self.arm_root / "tmp"), "PYTHONPATH": str(PROBE_DIR),
                    "OUROBOROS_STAND_PROBE_DIR": str(self.arm_root / "probe")})
        return env

    def launch(self) -> None:
        self._patch_settings_ports()
        log = open(self.arm_root / "server.stdout.log", "ab")
        self.proc = subprocess.Popen(["/usr/bin/sandbox-exec", "-f", str(self.profile_path), sys.executable, "server.py"],
                                     cwd=str(self.clone), env=self._env(), stdout=log, stderr=subprocess.STDOUT,
                                     **subprocess_new_group_kwargs())

    def ready(self) -> bool:
        try:
            return supervisor_state_is_ready(_api(self.base_url, "GET", "/api/state", timeout=5))
        except Exception:  # noqa: BLE001 - not answering yet is "not ready", the deadline decides
            return False


# ---------------------------------------------------------------- what the arm observes


class Frames:
    """Every frame of the SPA's own ``/ws`` with its arrival time; reconnects if the socket drops."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.rows: List[Dict[str, Any]] = []
        self.ws = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._pump, daemon=True, name="stand-ws")
        self.thread.start()

    def _connect(self):
        from websockets.sync.client import connect

        return connect(self.url, open_timeout=30, proxy=None, max_size=None)

    def _pump(self) -> None:
        while not self.stop.is_set():
            try:
                ws = self._connect()
            except Exception:  # noqa: BLE001 - not listening yet or dropped: try again
                time.sleep(2)
                continue
            with self.lock:
                self.ws = ws
            try:
                while not self.stop.is_set():
                    try:
                        raw = ws.recv(timeout=1.0)
                    except TimeoutError:
                        continue
                    try:
                        frame = json.loads(raw)
                    except ValueError:
                        continue
                    with self.lock:
                        self.rows.append({"t": _now(), "frame": frame})
            except Exception:  # noqa: BLE001 - a dropped socket reconnects; the turn rows keep their times
                with self.lock:
                    self.ws = None
                time.sleep(2)

    def send(self, payload: dict) -> bool:
        deadline = _now() + 60
        while _now() < deadline:
            with self.lock:
                ws = self.ws
            if ws is not None:
                try:
                    ws.send(json.dumps(payload))
                    return True
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1)
        return False

    def replies(self, marker: str) -> Optional[float]:
        needle = f"ACK {marker}"
        with self.lock:
            for row in self.rows:
                frame = row["frame"]
                if isinstance(frame, dict) and not frame.get("is_progress") and needle in str(frame.get("content") or ""):
                    return row["t"]
        return None

    def close(self) -> None:
        self.stop.set()
        with self.lock:
            ws = self.ws
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass


class EventTail:
    """``task_received`` rows of the arm's own ``logs/events.jsonl``, read incrementally."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0
        self.received: Dict[str, Dict[str, Any]] = {}
        self.counts: Dict[str, int] = {}

    def poll(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return
        self.offset += cut + 1
        for line in chunk[:cut].splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            kind = str(row.get("type") or "")
            self.counts[kind] = self.counts.get(kind, 0) + 1
            if kind == "task_received":
                task = row.get("task") or {}
                ref = (task.get("metadata") or {}).get("origin_message_ref") or {}
                cid = ref.get("client_message_id")
                if cid:
                    self.received.setdefault(cid, {"t": _now(), "ts": row.get("ts"), "task_id": task.get("id")})


def sample_process(pid: int) -> Dict[str, Any]:
    """Memory, CPU and processes of the server's group; threads of the server itself."""
    out = subprocess.run(["ps", "-axo", "pid=,pgid=,rss=,pcpu="], capture_output=True, text=True).stdout
    rss = cpu = procs = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1] == str(pid):
            procs += 1
            rss += int(parts[2])
            cpu += float(parts[3])
    threads = max(0, len(subprocess.run(["ps", "-M", "-p", str(pid)], capture_output=True, text=True).stdout.splitlines()) - 1)
    return {"t": _now(), "rss_gib": round(rss / 1048576, 3), "cpu_pct": round(cpu, 1), "procs": procs, "threads": threads}


def claudexord_pids() -> List[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return sorted(line.split()[0] for line in out.splitlines() if "claudexord" in line and "grep" not in line)


def processes_under(path: pathlib.Path) -> List[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if str(path) in line]


def frame_summary(probe: pathlib.Path, server_pid: int, top: int = 12) -> List[Dict[str, Any]]:
    """The most frequent innermost Ouroboros frames of the server process's threads."""
    path = probe / f"frames-{server_pid}.jsonl"
    counts: Dict[str, int] = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for thread in row.get("threads") or []:
            own = [item for item in thread.get("stack") or [] if item.startswith(("ouroboros/", "supervisor/"))]
            if own:
                key = f"{thread.get('thread', '')[:40]} @ {own[-1]}"
                counts[key] = counts.get(key, 0) + 1
    return [{"where": key, "samples": value} for key, value in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


# ---------------------------------------------------------------- one arm


def run_arm(args) -> Dict[str, Any]:
    stand = pathlib.Path(args.stand).resolve()
    base = stand / "base"
    live_root = pathlib.Path(json.loads((base / "manifest.json").read_text(encoding="utf-8"))["live_root"])
    pid = snapshot.running_install_pid(live_root)
    if pid:  # an arm's clone of the copy stalls the same volume's writers as the snapshot did
        raise RuntimeError(f"the install is running (pid {pid}); quit the app before running an arm")
    label = re.sub(r"[^A-Za-z0-9]", "", args.label) or "X"
    arm = stand / "arms" / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    arm.mkdir(parents=True)
    report: Dict[str, Any] = {"label": label, "rev": args.rev, "arm": str(arm), "sends": args.sends,
                              "window_sec": args.window, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def save() -> None:
        (arm / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    df_before = _free_gib(stand)
    snapshot.clonefile(base / "data", arm / "data")
    repo = arm / "repo"
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(args.source), str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-B", "ouroboros", args.rev], check=True)
    for key, value in (("user.name", "Ouroboros"), ("user.email", "311266734+ouroboros-agent@users.noreply.github.com")):
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "remove", "origin"], check=False, capture_output=True)
    report["head"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    live_home = live_root.parent
    profile_path = arm / "arm.sb"
    profile_path.write_text(sandbox.profile(arm, live_home=live_home, live_port=args.live_port), encoding="utf-8")
    report["canaries"] = sandbox.run_canaries(arm, profile_path, sys.executable, live_home=live_home,
                                              live_port=args.live_port)
    engines_before = claudexord_pids()

    stub = stub_model().__enter__()
    settings_path = arm / "data" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["OPENAI_COMPATIBLE_BASE_URL"] = stub.base_url
    settings["OUROBOROS_SUBAGENT_WORKTREE_ROOT"] = str(arm / "worktrees")
    snapshot.write_private_json(settings_path, settings)

    server = SandboxedServer(repo, arm / "data", settings_path, arm_root=arm, profile_path=profile_path)
    samples: List[Dict[str, Any]] = []
    stop_sampling = threading.Event()
    turns: List[Dict[str, Any]] = []
    frames: Optional[Frames] = None
    censor = ""
    try:
        t_launch = _now()
        server.launch()
        report["server_pid"] = server.proc.pid

        def sampler() -> None:
            while not stop_sampling.wait(SAMPLE_EVERY_SEC):
                if server.proc.poll() is not None:
                    return
                samples.append(sample_process(server.proc.pid))

        threading.Thread(target=sampler, daemon=True, name="stand-sampler").start()
        while not server.ready():
            if server.proc.poll() is not None:
                raise RuntimeError(f"server exited before readiness rc={server.proc.returncode}")
            if _now() - t_launch > args.boot_timeout:
                censor = "boot_timeout"
                raise TimeoutError("server not ready")
            time.sleep(2)
        t_ready = _now()
        report["boot_sec"] = round(t_ready - t_launch, 1)
        events = EventTail(arm / "data" / "logs" / "events.jsonl")
        frames = Frames(f"ws://127.0.0.1:{server.port}/ws")
        offsets = sorted(float(x) for x in args.sends.split(","))
        end = t_ready + max(args.window, offsets[-1] + args.reply_timeout)
        pending = list(enumerate(offsets, 1))
        while _now() < end:
            if server.proc.poll() is not None:
                censor = censor or f"server_exited_rc{server.proc.returncode}"
                break
            if samples and samples[-1]["rss_gib"] > args.memory_cap_gib:
                censor = f"memory_cap_{args.memory_cap_gib}gib"
                break
            while pending and _now() >= t_ready + pending[0][1]:
                n, offset = pending.pop(0)
                marker = f"STAND-{label}-T{n}-{uuid.uuid4().hex[:6]}"
                cid = f"stand-{label}-{n}-{uuid.uuid4().hex}"
                sent = frames.send({"type": "chat", "chat_id": 1, "client_message_id": cid,
                                    "content": f"{marker}: stand check, please reply with one short line."})
                turns.append({"n": n, "offset_sec": offset, "marker": marker, "client_message_id": cid,
                              "t_send": _now(), "sent": sent})
            events.poll()
            if not pending and all(_turn_done(turn, events, stub, frames) for turn in turns) and _now() >= t_ready + args.window:
                break
            time.sleep(2)
        events.poll()
        report["event_counts"] = dict(sorted(events.counts.items(), key=lambda kv: -kv[1])[:25])
        for turn in turns:
            _turn_done(turn, events, stub, frames)
            for key in ("t_task_received", "t_first_dispatch", "t_reply"):
                if turn.get(key):
                    turn[key.replace("t_", "s_") + "_after_send"] = round(turn[key] - turn["t_send"], 1)
        report["turns"] = turns
        report["t_ready"] = t_ready
    except Exception as exc:  # noqa: BLE001 - every failure becomes a report fact
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sampling.set()
        if frames is not None:
            frames.close()
        server.stop()
        stub.__exit__(None, None, None)
        report["censored"] = censor
        report["samples"] = samples
        report["stub_calls"] = {"total": len(stub.arrivals),
                                "by_kind": _count(item["kind"] for item in stub.arrivals)}
        report["frames_top"] = frame_summary(arm / "probe", report.get("server_pid") or 0)
        report["engines_started"] = sorted(set(claudexord_pids()) - set(engines_before))
        report["left_running"] = processes_under(arm)
        report["free_gib_before_after"] = [df_before, _free_gib(stand)]
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
    return report


def _turn_done(turn: Dict[str, Any], events: EventTail, stub, frames: Frames) -> bool:
    got = events.received.get(turn["client_message_id"])
    if got and not turn.get("t_task_received"):
        turn["t_task_received"], turn["task_id"] = got["t"], got.get("task_id")
    if not turn.get("t_first_dispatch"):
        hits = [item["t"] for item in stub.arrivals if item["marker"] == turn["marker"]]
        if hits:
            turn["t_first_dispatch"] = min(hits)
    if not turn.get("t_reply"):
        turn["t_reply"] = frames.replies(turn["marker"])
    return bool(turn.get("t_reply"))


def _count(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def _free_gib(path: pathlib.Path) -> float:
    stat = os.statvfs(path)
    return round(stat.f_bavail * stat.f_frsize / 2 ** 30, 1)


# ---------------------------------------------------------------- snapshot


def run_snapshot(args) -> Dict[str, Any]:
    from ouroboros.provider_models import ACTIVE_MODEL_SETTING_KEYS, ALL_PROVIDER_CREDENTIAL_KEYS, LEGACY_MODEL_SETTING_KEYS
    from tests.system_e2e.harness import MOCK_SLUG

    stand = pathlib.Path(args.out).resolve()
    stand.mkdir(parents=True, exist_ok=True)
    return snapshot.make_snapshot(pathlib.Path(args.live), stand / "base",
                                  model_keys=(*ACTIVE_MODEL_SETTING_KEYS, *LEGACY_MODEL_SETTING_KEYS),
                                  credential_keys=ALL_PROVIDER_CREDENTIAL_KEYS, stub_slug=MOCK_SLUG)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="devtools.real_data.stand")
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--live", required=True)
    snap.add_argument("--out", required=True)
    arm = sub.add_parser("arm")
    arm.add_argument("--stand", required=True)
    arm.add_argument("--source", required=True, help="a git clone holding --rev (the arm clones it --shared)")
    arm.add_argument("--rev", required=True)
    arm.add_argument("--label", required=True)
    arm.add_argument("--sends", default="30,120,240,360", help="seconds after readiness")
    arm.add_argument("--window", type=float, default=420.0, help="seconds after readiness the arm stays up")
    arm.add_argument("--reply-timeout", type=float, default=180.0)
    arm.add_argument("--boot-timeout", type=float, default=1500.0)
    arm.add_argument("--memory-cap-gib", type=float, default=24.0)
    arm.add_argument("--live-port", type=int, default=sandbox.DEFAULT_APP_PORT, help="the live app's port, kept unreachable")
    args = parser.parse_args(argv)
    result = run_snapshot(args) if args.cmd == "snapshot" else run_arm(args)
    brief = {key: result.get(key) for key in ("arm", "label", "rev", "boot_sec", "censored", "error", "seconds",
                                               "engines_started", "left_running")}
    if result.get("turns"):
        brief["turns"] = [{k: turn.get(k) for k in ("n", "offset_sec", "task_id", "s_task_received_after_send",
                                                       "s_first_dispatch_after_send", "s_reply_after_send")}
                          for turn in result["turns"]]
    if result.get("samples"):
        brief["rss_gib_max"] = max(s["rss_gib"] for s in result["samples"])
        brief["cpu_pct_median"] = sorted(s["cpu_pct"] for s in result["samples"])[len(result["samples"]) // 2]
    print(json.dumps(brief, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
