"""Ask Ouroboros to fix: real UI ingress, managed worker, review and installed execution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from devtools.benchmarks.common.server_runner import _api
from ouroboros.skill_loader import compute_content_hash
from tests._shared import producer_evidence_dir
from tests.candidate_checkout import candidate_checkout
from tests.system_e2e.harness import (
    ArtifactOracle, ScriptedStubModel, keyless_settings, start_server, wait_until,
)


@pytest.fixture
def repair_clone(tmp_path):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1")
    source = Path(__file__).resolve().parents[1]
    with candidate_checkout(source, tmp_path / "clone", origin_proof=True) as candidate:
        yield candidate


@pytest.mark.ui_browser
@pytest.mark.serial
def test_repair_button_records_owner_intent_and_runs_the_installed_skill(tmp_path, repair_clone):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    name = "repair_owner_probe"
    evidence = producer_evidence_dir(tmp_path, "repair-owner-intent", env="OUROBOROS_UI_SCREENSHOT_DIR")
    body = (
        "async def status(request):\n    return {'value': 'repaired'}\n"
        "def register(api):\n"
        "    api.register_route('status', status, methods=('GET',))\n"
        "    api.register_ui_tab('main', 'Repair result', render={'kind':'module','entry':'widget.js','start':'auto'})\n"
        "    api.register_companion_process('worker')\n"
    )
    script = [
        {"tool": "write_file", "arguments": {"root": "skill_payload", "path": "plugin.py", "content": body}},
        {"tool": "run_command", "arguments": {"cmd": ["python3", "-c", "print('repair shell works')"], "cwd": "skill_payload"}},
        {"tool": "skill_preflight", "arguments": {"skill": name}},
        {"tool": "skill_review", "arguments": {"skill": name}},
        {"tool": "toggle_skill", "arguments": {"skill": name, "enabled": True}},
    ]
    with ScriptedStubModel(script, final_answer="The repaired skill is running.") as stub:
        server = start_server(repair_clone, tmp_path / "runtime", keyless_settings(
            stub, OUROBOROS_MAX_WORKERS=1, OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS=True,
        ))
        try:
            drive = Path(server.data_root)
            oracle = ArtifactOracle(drive)
            payload = drive / "skills" / "external" / name
            payload.mkdir(parents=True)
            (payload / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Repair owner source integration fixture\n"
                'version: "1.0"\ntype: extension\nentry: plugin.py\nplugin_api: "2.0"\n'
                'permissions: [route, widget, companion_process]\n'
                'companion_processes:\n  - name: worker\n    runtime: python3\n'
                '    command: [python3, worker.py]\n    restart_policy: never\n---\n'
            )
            (payload / "plugin.py").write_text("def register(api):\n    broken syntax!\n")
            (payload / "worker.py").write_text(
                "import os, pathlib, threading\n"
                "pathlib.Path(os.environ['OUROBOROS_SKILL_STATE_DIR'], 'ready.txt').write_text('ready')\n"
                "threading.Event().wait(120)\n"
            )
            (payload / "widget.js").write_text(
                "const style = document.createElement('style');\n"
                "style.textContent = 'body{color:#e8ecf3;font:16px system-ui;padding:16px}';\n"
                "document.head.appendChild(style);\n"
                f"fetch('/api/extensions/{name}/status').then(r => r.json()).then(x => {{\n"
                " document.getElementById('root').textContent = 'Installed: ' + x.value;\n});\n"
            )
            initial_hash = compute_content_hash(payload)
            preflight = _api(server.base_url, "POST", f"/api/skills/{name}/review", {}, timeout=120)
            assert preflight["status"] == "pending", preflight
            assert not stub.script_consumed()
            with playwright.sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 850})
                    page.goto(server.base_url + "/#skills")
                    card = page.locator(f'.skills-card[data-skill="{name}"]')
                    card.get_by_role("button", name="Ask Ouroboros to fix", exact=True).wait_for()
                    page.screenshot(path=str(evidence / "repair-and-run-card.png"))
                    card.get_by_role("button", name="Ask Ouroboros to fix", exact=True).click()
                    dialog = page.get_by_role("dialog")
                    assert "turns it on when it is ready" in dialog.inner_text()
                    page.screenshot(path=str(evidence / "repair-and-run-confirmation.png"))
                    with page.expect_request("**/api/command") as command:
                        dialog.get_by_role("button", name="Ask Ouroboros", exact=True).click()
                    request = command.value.post_data_json
                    assert "visible_text" not in request
                    assert request["task_constraint"]["allow_enable"] is False
                    admission = wait_until(lambda: oracle._json(f"state/skills/{name}/repair_admission.json"), 45)
                    task_id = admission["task_id"]
                    result = server.wait_task(task_id, timeout=240)
                    assert result["status"] == "completed", result
                    stored = oracle.task_result(task_id)
                    ref = stored["origin_message_ref"]
                    rows = [json.loads(line) for line in oracle.chat_bytes().splitlines() if line]
                    owner = next(row for row in rows if row.get("client_message_id") == ref["client_message_id"] and row.get("direction") == "in")
                    assert owner["text"] == request["cmd"] == stored["origin_message_text"]
                    assert "Repair and run" in owner["text"]
                    assert stored["task_constraint"]["allow_enable"] is False
                    assert admission["base_content_hash"] == initial_hash
                    actions = [row for row in oracle.events("owner_api_action") if row.get("task_id") == task_id]
                    enabled = next(row for row in actions if row.get("action") == "skill_enable" and row.get("ok"))
                    assert enabled["source_ref"]["ref"] == ref
                    assert enabled["actor"] == "agent_tool"
                    assert stub.script_consumed() and stub.kinds().count("skill_review") >= 3
                    assert "repair shell works" in json.dumps(oracle.tools_rows())
                    wait_until(lambda: oracle._json(f"state/skills/{name}/enabled.json").get("enabled"), 30)
                    companions = wait_until(lambda: _api(server.base_url, "GET", "/api/skills/daemons").get("companions", {}).get(name + ":worker"), 30)
                    assert companions["pid"] > 0
                    # The pid exists from the spawn; the file is the worker's first act once Python is up.
                    ready = drive / "state" / "skills" / name / "ready.txt"
                    assert wait_until(lambda: ready.is_file() and ready.read_text() == "ready", 30), "the companion never wrote ready.txt"
                    # The hash is a one-time entry pointer; app navigation owns
                    # later page changes in the already-loaded SPA.
                    page.click('[data-nav-page="widgets"]')
                    page.frame_locator(f'[data-widget-key="{name}:main"] iframe').locator("#root").filter(has_text="Installed: repaired").wait_for(timeout=30000)
                    page.screenshot(path=str(evidence / "repair-and-run-installed.png"))
                    (evidence / "repair-and-run.json").write_text(json.dumps({
                        "task_id": task_id, "origin": ref, "initial_hash": initial_hash,
                        "final_hash": compute_content_hash(payload), "companion_pid": companions["pid"],
                        "candidate_identity": repair_clone.identity,
                        "review_delivery": "real pipeline, controlled loopback reviewer models",
                    }, indent=2) + "\n")
                    _api(server.base_url, "POST", f"/api/skills/{name}/toggle", {"enabled": False})
                finally:
                    browser.close()
        finally:
            server.stop()
