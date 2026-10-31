"""Installed skill UI flows with the real browser modules against an isolated server."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._shared import producer_evidence_dir


def _evidence_dir(tmp_path):
    return producer_evidence_dir(tmp_path, "skill-lifecycle", env="OUROBOROS_UI_SCREENSHOT_DIR")


from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401

direct_server_with_data = _direct_server_with_data


@pytest.mark.ui_browser
@pytest.mark.serial
def test_installed_repair_reloads_tool_route_widget_and_companion(direct_server_with_data, monkeypatch, tmp_path):
    import time
    import urllib.request
    from ouroboros import extension_loader
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.skill_loader import compute_content_hash, load_review_state
    from ouroboros.skill_repair_admission import record_repair_admission
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros import dialogue_store
    from ouroboros.utils import utc_now_iso
    from ouroboros.tools.registry import ToolRegistry
    from tests._extension_loader_shared import _write_ext_skill
    from tests._skill_review_shared import _mark_self_authored

    drive = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    name = "repair_live_fixture"
    payload = _write_ext_skill(
        drive / "skills" / "external", name,
        permissions=["tool", "route", "widget", "companion_process"],
        plugin_body=(
            "async def status(request):\n    return {'value': 'version one'}\n"
            "def register(api):\n"
            "    api.register_tool('ping', lambda ctx: 'version one', description='probe', schema={})\n"
            "    api.register_route('status', status, methods=('GET',))\n"
            "    api.register_ui_tab('main', 'Repaired application', render={'kind':'module','entry':'widget.js','start':'auto'})\n"
            "    api.register_companion_process('worker')\n"
        ),
        extra_frontmatter=(
            'plugin_api: "2.0"\ncompanion_processes:\n  - name: worker\n    runtime: python3\n'
            '    command: ["python3", "worker.py"]\n    restart_policy: never\n'
        ),
    )
    (payload / "worker.py").write_text(
        "import os, pathlib, threading\n"
        "pathlib.Path(os.environ['OUROBOROS_SKILL_STATE_DIR'], 'ready.txt').write_text('version one')\n"
        "threading.Event().wait(120)\n"
    )
    (payload / "widget.js").write_text(
        "const style = document.createElement('style');\n"
        "style.textContent = 'body{color:#e8ecf3;font:16px system-ui;padding:16px}';\n"
        "document.head.appendChild(style);\n"
        f"fetch('/api/extensions/{name}/status').then(r => r.json()).then(value => {{\n"
        " document.getElementById('root').textContent = 'Installed widget: ' + value.value;\n});\n"
    )
    _mark_self_authored(payload, drive)
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "unused"))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=drive)
    registry._ctx.task_id = "live-repair"
    registry._ctx.current_chat_id = 42
    registry._ctx.task_constraint = TaskConstraint(skill_name=name, payload_root=f"skills/external/{name}", allow_enable=False)
    owner_text = "Repair and run the installed application, leaving it working."
    origin = build_owner_message_ref(chat_id=42, client_message_id="repair-fixture", ts=utc_now_iso(), text=owner_text)
    # Owner messages live in the dialogue store since #1982; a chat.jsonl line is no owner message.
    dialogue_store.append(drive, "chat", {**origin, "direction": "in", "text": owner_text, "source": "web"})
    registry._ctx.task_metadata = {"origin_message_ref": origin}
    record_repair_admission(drive, name, task_id="live-repair", base_content_hash=compute_content_hash(payload))
    # The one wave over the golden pool with an offline seam: every seat answers the skill question clean.
    import ouroboros.review_substrate as substrate
    from tests import _contributor_packet_shared as shared
    from tests.review_pool_rosters import set_review_pool

    set_review_pool(monkeypatch, shared.golden_pool())

    def clean(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        actor.raw_text = "[]\nNO_FINDINGS"
        return actor

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam([], answer=clean))

    def api(path, body=None):
        request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    playwright = pytest.importorskip("playwright.sync_api")
    pids, revisions = [], []
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 850})
            for value in ("version one", "version two"):
                if value == "version two":
                    for path in ("plugin.py", "worker.py"):
                        # plugin.py has two intentional occurrences; the file writer
                        # uses exact complete content, while worker uses exact replace.
                        source = (payload / path).read_text()
                        result = registry.execute("write_file", {"root": "skill_payload", "path": path,
                                                   "content": source.replace("version one", value)})
                        assert result.startswith("OK:"), result
                reviewed = registry.execute("skill_review", {"skill": name})
                state = load_review_state(drive, name)
                assert state.status == "clean", reviewed
                revisions.append(state.content_hash)
                enabled = registry.execute("toggle_skill", {"skill": name, "enabled": True})
                assert json.loads(enabled)["enabled"], enabled
                reconciled = api(f"/api/skills/{name}/reconcile", {})
                assert not reconciled.get("load_error"), reconciled
                assert api(f"/api/extensions/{name}/status")["value"] == value
                surface = extension_loader.extension_surface_name(name, "ping")
                actual = registry.execute_result(surface, {})
                assert actual.status == "ok" and actual.text == value, actual
                assert actual.meta["content_hash"] == state.content_hash
                ready = drive / "state" / "skills" / name / "ready.txt"
                deadline = time.monotonic() + 15
                while (not ready.exists() or ready.read_text() != value) and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert ready.read_text() == value
                companion = api("/api/skills/daemons")["companions"][name + ":worker"]
                pids.append(companion["pid"])
                assert pid_is_alive(pids[-1])
                page.goto(url + "/#widgets")
                frame = page.frame_locator(f'[data-widget-key="{name}:main"] iframe')
                frame.locator("#root").filter(has_text="Installed widget: " + value).wait_for(timeout=30000)
                evidence = _evidence_dir(tmp_path)
                page.screenshot(path=str(evidence / ("repaired-widget-" + value.replace(" ", "-") + ".png")))
            assert revisions[0] != revisions[1]
            assert pids[0] != pids[1]
            assert not pid_is_alive(pids[0])
        finally:
            browser.close()
            api(f"/api/skills/{name}/toggle", {"enabled": False})
            extension_loader.unload_extension(name)
    assert name + ":worker" not in api("/api/skills/daemons")["companions"]
    assert all(not pid_is_alive(pid) for pid in pids)
    if os.environ.get("OUROBOROS_UI_SCREENSHOT_DIR"):
        (Path(os.environ["OUROBOROS_UI_SCREENSHOT_DIR"]) / "installed-execution.json").write_text(json.dumps({
            "revisions": revisions, "companion_pids": pids, "all_companions_stopped": True,
            "surfaces": ["review", "tool", "http", "widget", "companion"],
            "review_delivery": "real skill pipeline with controlled panel backend",
            "server": "real isolated server.py",
        }, indent=2) + "\n")
