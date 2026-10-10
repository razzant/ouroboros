"""Owner-present A/B recipe for the pinned Playwright Extension consumer.

Serve only (safe to prepare without Chrome):
  python scripts/browser_focus_probe.py --source-root "$PWD" --data-root /tmp/focus-probe

Later, with the owner present and a genuinely cyber_pro Ouroboros process:
  python scripts/browser_focus_probe.py --source-root "$PWD" \
      --data-root /Users/kazzand/Ouroboros/data \
      --live-data-root /Users/kazzand/Ouroboros/data \
      --task-id CURRENT_RUNNING_TASK_ID --task-attempt CURRENT_ATTEMPT \
      --cli /absolute/path/to/@playwright/mcp/cli.js --extension --owner-ready

The explicit --extension plus --owner-ready pair is required to launch the
visible Extension connection. The owner must supply the canonical live data
root twice; a temporary root cannot provide live Panic custody. This script
does not read private settings or change the saved/runtime mode. It imports
this checkout's source as a registered ToolRegistry consumer; its printed
source HEAD is not the identity of the separately running Ouroboros server
and does not prove that server has enabled this bridge. Open the printed A
and B URLs in the owner's existing Chrome. Press Return to attach, choose only A in Microsoft's
Allow dialog, then return to B and click B's ready button. The pinned Extension
activates its selected tab/window on initial attachment; this script cannot
prevent that initial switch. After the B signal it sends at most three explicit
``locator.click({ force: true })`` actions to A, checking the counter/events
after each. A timeout ends the action phase without a resend. It prints the
task-session close receipt. Neither that receipt nor page visibility proves the
Extension indicator disappeared; the owner must observe that separately.

The optional external focus sample uses macOS System Events and Chrome's active
tab URL; it reports only whether the active URL is B, never another tab's URL.
If Automation permission prevents this sample, focus remains unverified and
actions require --allow-unobserved-focus. A page's ``hasFocus`` and
``visibilityState`` alone are not foreground proof under Extension/CDP.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.server
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

MARKER = "POINTER_PROBE_JSON:"


def _parse_result(result):
    if result.status != "ok":
        raise RuntimeError(f"{result.status}/{result.code}: {result.text}")
    match = re.search(re.escape(MARKER) + r"([^\r\n]+)", result.text)
    if not match:
        raise RuntimeError(f"The run-code result has no probe payload: {result.text}")
    return json.loads(json.loads('"' + MARKER + match.group(1)).removeprefix(MARKER))


def _sample_focus(b_url: str) -> dict:
    stamp = {"observed_at": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic()}
    if sys.platform != "darwin" or not shutil.which("osascript"):
        return {**stamp, "observer": "unavailable", "b_active": None}
    checks = [
        'tell application "System Events" to get name of first application process whose frontmost is true',
        'tell application "Google Chrome" to get URL of active tab of front window',
    ]
    values = []
    for expression in checks:
        try:
            proc = subprocess.run(["osascript", "-e", expression], capture_output=True, text=True, timeout=8)
        except (OSError, subprocess.TimeoutExpired):
            return {**stamp, "observer": "unavailable", "b_active": None,
                    "reason": "macOS Automation query did not complete"}
        if proc.returncode:
            return {**stamp, "observer": "unavailable", "b_active": None,
                    "reason": "macOS Automation query failed"}
        values.append(proc.stdout.strip())
    return {**stamp, "observer": "System Events + Chrome active tab",
            "frontmost_app": values[0], "b_active": values[0] == "Google Chrome" and values[1] == b_url}


class _FocusObserver:
    """Timestamped samples through action and disconnect; never a no-gap proof."""

    def __init__(self, b_url):
        self.b_url = b_url
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.samples.append(_sample_focus(self.b_url))
            # Exhaustion is incomplete observation, never PASS.
            if len(self.samples) >= 256:
                return
            self.stop.wait(.1)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=17)
        return {"samples": list(self.samples), "observer_closed": not self.thread.is_alive(),
                "bounded_out": len(self.samples) >= 256, "coverage": "samples; transient switches may be missed"}


def _require_b_focus(focus: dict, allow_unobserved: bool) -> bool:
    """Return whether B was externally observed; refuse lost/other focus."""
    if focus["b_active"] is True:
        return True
    if focus["b_active"] is None and allow_unobserved:
        return False
    raise RuntimeError("B is not externally observed as active; further pointer actions stopped")


def _require_running_task(data_root: Path, task_id: str, attempt: int) -> None:
    """Read existing live task authority without creating or modifying a result."""
    from ouroboros.task_results import load_task_result

    row = load_task_result(data_root, task_id, strict=True)
    if not row or row.get("status") != "running" or row.get("task_attempt") != attempt:
        raise RuntimeError("--extension requires the matching current running task and attempt")


def _run_code(registry, tool: str, body: str):
    return registry.execute_result(tool, {"code": f"async (page) => {{ {body} }}"})


def _extension_consumer(args, a_url: str, b_url: str, ready: threading.Event):
    # Imports follow the explicit data-root binding and exact source-root check.
    from ouroboros import config, mcp_client, mcp_task_sessions
    from ouroboros.tools.registry_core import ToolRegistry

    if config.get_runtime_mode() != "cyber_pro":
        raise RuntimeError("An actual cyber_pro runtime baseline is required; this script does not change it")
    package = Path(args.cli).resolve().parent / "package.json"
    if json.loads(package.read_text())["version"] != "0.0.82":
        raise RuntimeError("CLI package is not pinned @playwright/mcp 0.0.82")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is unavailable")
    data_root = Path(args.data_root).resolve()
    _require_running_task(data_root, args.task_id, args.task_attempt)
    registry = ToolRegistry(repo_dir=Path(args.source_root).resolve(), drive_root=data_root)
    ctx = registry._ctx
    ctx.task_id = args.task_id
    ctx.task_attempt = args.task_attempt
    ctx.task_lifecycle_bound = True
    ctx.messages = []
    manager = mcp_client.get_manager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 30, "MCP_SERVERS": [{
        "id": "browser", "enabled": True, "transport": "stdio", "command": node,
        "browser_bridge": True, "args": [str(Path(args.cli).resolve()), "--extension"],
    }]})
    print("Attachment activates the chosen tab/window once. Select A only in the Allow dialog.", flush=True)
    input("With A and B open in the existing Chrome, press Return to connect: ")
    attached = False
    observer = None
    try:
        _require_running_task(data_root, args.task_id, args.task_attempt)
        opened = manager.refresh_server("browser", authority=ctx)
        if not opened.get("ok"):
            raise RuntimeError(f"Extension did not connect: {opened}")
        attached = True
        print("Owner must observe the initial focus switch and managed-tab indicator.", flush=True)
        rows = {row["raw_name"]: row for row in manager.list_tools_for_registry()}
        row = rows.get("browser_run_code_unsafe")
        if not row or row["schema"].get("properties", {}).get("code", {}).get("type") != "string":
            raise RuntimeError("Official run-code tool/schema absent after discovery")
        tool = row["name"]
        first = _parse_result(_run_code(registry, tool,
            f"return '{MARKER}' + JSON.stringify({{url: page.url(), state: await page.evaluate(() => window.probe?.state())}});"))
        if first["url"] != a_url or not first["state"] or first["state"]["counter"] != 0:
            raise RuntimeError("The selected Extension tab is not the untouched local A page")
        ready.clear()  # An earlier B click cannot authorize pointer actions.
        print("A is connected. Return to B and click its ready button; do not click A's counter.", flush=True)
        if not ready.wait(timeout=300):
            raise RuntimeError("B readiness was not received; no pointer action sent")
        _require_running_task(data_root, args.task_id, args.task_attempt)
        before = _sample_focus(b_url)
        print("before actions:", json.dumps(before, ensure_ascii=False), flush=True)
        observed_focus = _require_b_focus(before, args.allow_unobserved_focus)
        observer = _FocusObserver(b_url)
        observer.start()
        for count in (1, 2, 3):
            _require_running_task(data_root, args.task_id, args.task_attempt)
            for sample in list(observer.samples):
                observed_focus &= _require_b_focus(sample, args.allow_unobserved_focus)
            result = _run_code(registry, tool,
                "await page.getByRole('button', { name: 'Count' }).click({ force: true }); "
                f"return '{MARKER}' + JSON.stringify(await page.evaluate(() => window.probe.state()));")
            if result.status != "ok":
                print("ACTION OUTCOME UNKNOWN; no resend:", result.status, result.code, result.text, flush=True)
                raise RuntimeError("Action outcome unknown; no resend or further pointer action")
            try:
                state = _parse_result(result)
            except (ValueError, RuntimeError):
                print("ACTION OUTCOME UNKNOWN; no resend: returned payload could not be verified", flush=True)
                raise
            expected = [{"type": event, "target": "count", "isTrusted": True}
                        for _ in range(count) for event in ("pointerdown", "mousedown", "mouseup", "click")]
            focus = _sample_focus(b_url)
            print("after action", count, json.dumps({"state": state, "focus": focus}, ensure_ascii=False), flush=True)
            if state["counter"] != count or state["events"] != expected:
                raise RuntimeError("Counter or trusted pointer sequence failed; stopped")
            observed_focus &= _require_b_focus(focus, args.allow_unobserved_focus)
        print("POINTER DELIVERY PASS; focus:",
              "B in available samples; disconnect sampling pending" if observed_focus else "UNOBSERVED (owner verification required)",
              flush=True)
    finally:
        try:
            closure = mcp_task_sessions.stop_task(ctx)
            print("session closure:", closure, flush=True)
        finally:
            # Even a failed session close cannot leave our observer querying Chrome.
            if observer is not None:
                report = observer.finish()
                after_close = _sample_focus(b_url)
                report["after_disconnect"] = after_close
                print("EXTERNAL_FOCUS_EVIDENCE:", json.dumps(report, ensure_ascii=False), flush=True)
                for sample in [*report["samples"], after_close]:
                    _require_b_focus(sample, args.allow_unobserved_focus)
                if not report["observer_closed"] or report["bounded_out"]:
                    raise RuntimeError("External focus observer coverage/closure is incomplete")
        print("Owner must observe the Extension indicator disappearing.", flush=True)
        if attached and closure != [{"server": "browser", "closure": "confirmed"}]:
            raise RuntimeError("Extension session closure was not confirmed")


def _validated_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--live-data-root", help="owner-supplied canonical live data root for Extension custody")
    parser.add_argument("--task-id", help="current running task ID for Extension custody")
    parser.add_argument("--task-attempt", type=int, help="current running task attempt for Extension custody")
    parser.add_argument("--site-port", type=int, default=0)
    parser.add_argument("--cli", help="absolute path to pinned @playwright/mcp/cli.js")
    parser.add_argument("--extension", action="store_true", help="explicitly allow visible Extension connection")
    parser.add_argument("--owner-ready", action="store_true", help="owner is present for Allow and B selection")
    parser.add_argument("--allow-unobserved-focus", action="store_true",
                        help="continue pointer checks if macOS Automation observer is unavailable")
    args = parser.parse_args(argv)
    source = Path(args.source_root).resolve()
    if source != Path(__file__).resolve().parents[1]:
        parser.error("--source-root must be the checkout containing this script")
    if not Path(args.data_root).is_absolute():
        parser.error("--data-root must be absolute")
    if args.extension and (not args.owner_ready or not args.cli or not Path(args.cli).is_absolute()):
        parser.error("--extension requires --owner-ready and an absolute --cli")
    if args.owner_ready and not args.extension:
        parser.error("--owner-ready applies only to --extension")
    if args.live_data_root and not args.extension:
        parser.error("--live-data-root applies only to --extension")
    if (args.task_id or args.task_attempt is not None) and not args.extension:
        parser.error("--task-id and --task-attempt apply only to --extension")
    if args.extension:
        if not args.task_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.task_id):
            parser.error("--extension requires a valid --task-id")
        if args.task_attempt is None or args.task_attempt < 1:
            parser.error("--extension requires a positive --task-attempt")
        if not args.live_data_root or not Path(args.live_data_root).is_absolute():
            parser.error("--extension requires an absolute --live-data-root supplied by the owner")
        data_root = Path(args.data_root).resolve()
        live_root = Path(args.live_data_root).resolve()
        if data_root != live_root:
            parser.error("--data-root must match --live-data-root for Extension custody")
        temporary_roots = {Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve(),
                           Path("/var/tmp").resolve()}
        if any(data_root == root or root in data_root.parents for root in temporary_roots):
            parser.error("--extension refuses a temporary data root; use the canonical live root")
        if not live_root.is_dir():
            parser.error("--live-data-root must be an existing directory")
    return args


def main() -> int:
    args = _validated_args()
    source = Path(args.source_root).resolve()
    sys.path.insert(0, str(source))
    os.environ["OUROBOROS_DATA_DIR"] = str(Path(args.data_root).resolve())
    if args.extension:
        _require_running_task(Path(args.data_root).resolve(), args.task_id, args.task_attempt)
    pages = source / "tests" / "fixtures" / "browser_focus"
    ready = threading.Event()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(pages), **kw)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path in ("/event", "/state"):
                self.send_response(204)
                self.end_headers()
                return
            if self.path != "/ready" or body != b"owner-on-b":
                self.send_error(404)
                return
            ready.set()
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.site_port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    a_url, b_url = base + "/a.html", base + "/b.html"
    print("serving mode:", "owner-present Extension" if args.extension else "pages only")
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    print("source-bound standalone registered ToolRegistry consumer root:", source)
    print("consumer source HEAD:", head.stdout.strip() if head.returncode == 0 else "unavailable")
    print("running Ouroboros server identity: unverified separate process; bridge enablement unproven")
    print("data root:", args.data_root)
    if args.extension:
        print("live task binding:", args.task_id, "attempt", args.task_attempt)
    print("A:", a_url)
    print("B:", b_url, flush=True)
    try:
        if args.extension:
            _extension_consumer(args, a_url, b_url, ready)
        else:
            input("Pages only; press Return to stop serving: ")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
