"""F4: every execution request re-reads the SELECTED subject, cheaply.

The promise under test: a disable / review revocation / payload edit performed
by ANOTHER PROCESS takes effect on the next execution request (route, tool and
WS), a same-size edit with a restored mtime is not hidden by any stat shortcut,
a mutation that lands DURING a slow reconcile is still caught before the
handler runs — and the cost of all this does not grow with the number of
unrelated installed skills, because peers are never hashed.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros import extension_loader, skill_loader
from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    find_skill,
    save_enabled,
    save_review_state,
)
from tests.test_extensions_api import _make_client, _stop_patches, _write_ext

REPO = pathlib.Path(__file__).resolve().parents[1]

ROUTE_PLUGIN = (
    "from starlette.responses import JSONResponse\n"
    "def _hello(request):\n"
    "    return JSONResponse({'hello': 'world'})\n"
    "def register(api):\n"
    "    api.register_route('greet', _hello, methods=('GET',))\n"
    "    api.register_tool('greet', lambda ctx: 'hi', description='d', schema={})\n"
    "    api.register_ws_handler('greet', lambda msg: {'hello': 'world'})\n"
)


def _in_another_process(code: str, **names: str) -> None:
    """Run a mutation in a genuinely separate interpreter."""
    header = "".join(f"{key} = {value!r}\n" for key, value in names.items())
    result = subprocess.run(
        [sys.executable, "-c", header + code],
        cwd=str(REPO), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture()
def live_route(tmp_path, monkeypatch):
    """A reviewed, enabled, LIVE extension plus a pile of unrelated peers."""
    from ouroboros.config import load_settings

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(skills_root, "fresh_ext", permissions=["route", "tool", "ws_handler"],
                           plugin=ROUTE_PLUGIN)
    for index in range(12):  # unrelated installed peers: cost must not scale with these
        peer = _write_ext(skills_root, f"peer_{index:02d}", permissions=["tool"],
                          plugin="def register(api):\n    pass\n")
        (peer / "bulk.bin").write_bytes(b"z" * 40_000)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_enabled(drive_root, "fresh_ext", True)
    save_review_state(drive_root, "fresh_ext",
                      SkillReviewState(status="pass", content_hash=content_hash))
    for index in range(12):
        save_enabled(drive_root, f"peer_{index:02d}", False)
    loaded = find_skill(drive_root, "fresh_ext", repo_path=str(skills_root))
    assert loaded is not None
    assert extension_loader.load_extension(loaded, load_settings, drive_root=drive_root) is None
    try:
        yield {"client": client, "drive_root": drive_root, "skills_root": skills_root,
               "skill_dir": skill_dir, "content_hash": content_hash}
    finally:
        _stop_patches(patches)
        extension_loader.unload_extension("fresh_ext")


def _get(live_route):
    return live_route["client"].get("/api/extensions/fresh_ext/greet")


# --------------------------------------------------------------------------
# Route: cross-process disable / revoke / payload edit
# --------------------------------------------------------------------------

def test_route_is_live_until_another_process_disables_it(live_route):
    assert _get(live_route).status_code == 200
    _in_another_process(
        "import pathlib\n"
        "from ouroboros.skill_loader import save_enabled\n"
        "save_enabled(pathlib.Path(drive), 'fresh_ext', False)\n",
        drive=str(live_route["drive_root"]),
    )
    response = _get(live_route)
    assert response.status_code == 409, response.text
    assert response.json()["state"]["reason"] == "disabled"


def test_route_refuses_after_another_process_revokes_the_review(live_route):
    """A revoked (pending) verdict refuses on the NEXT request.

    ``fail`` is deliberately NOT used here: under the shipped advisory
    enforcement a blocker verdict still executes by operator choice, which is
    existing policy this change does not touch.
    """
    assert _get(live_route).status_code == 200
    _in_another_process(
        "import pathlib\n"
        "from ouroboros.skill_loader import SkillReviewState, save_review_state\n"
        "save_review_state(pathlib.Path(drive), 'fresh_ext',\n"
        "                  SkillReviewState(status='pending', content_hash=content_hash))\n",
        drive=str(live_route["drive_root"]), content_hash=live_route["content_hash"],
    )
    response = _get(live_route)
    assert response.status_code == 409, response.text
    assert response.json()["state"]["reason"] == "review_pending"


def test_route_refuses_after_another_process_edits_the_payload(live_route):
    assert _get(live_route).status_code == 200
    _in_another_process(
        "import pathlib\n"
        "path = pathlib.Path(entry)\n"
        "path.write_text(path.read_text(encoding='utf-8') + '\\n# edited elsewhere\\n', encoding='utf-8')\n",
        entry=str(live_route["skill_dir"] / "plugin.py"),
    )
    response = _get(live_route)
    assert response.status_code == 409, response.text
    assert response.json()["state"]["review_stale"] is True


def test_same_size_edit_with_a_restored_mtime_still_invalidates(live_route):
    """No stat/TTL shortcut may stand in for byte identity."""
    entry = live_route["skill_dir"] / "plugin.py"
    original = entry.read_bytes()
    before = entry.stat()
    assert _get(live_route).status_code == 200
    _in_another_process(
        "import os, pathlib\n"
        "path = pathlib.Path(entry)\n"
        "data = bytearray(path.read_bytes())\n"
        "data[-2] = (data[-2] ^ 0x20)  # same length, different bytes\n"
        "path.write_bytes(bytes(data))\n"
        "os.utime(path, (mtime, mtime))\n",
        entry=str(entry), mtime=before.st_mtime,
    )
    after = entry.stat()
    assert after.st_size == before.st_size
    assert abs(after.st_mtime - before.st_mtime) < 1e-6
    assert entry.read_bytes() != original
    response = _get(live_route)
    assert response.status_code == 409, response.text
    assert response.json()["state"]["review_stale"] is True


def test_route_refuses_when_a_conflicting_peer_is_enabled_elsewhere(live_route):
    assert _get(live_route).status_code == 200
    _in_another_process(
        "import pathlib\n"
        "from ouroboros.skill_loader import save_enabled\n"
        "path = pathlib.Path(skill_dir) / 'SKILL.md'\n"
        "text = path.read_text(encoding='utf-8').replace('conflicts: []', 'conflicts: [\"fresh_ext\"]')\n"
        "path.write_text(text, encoding='utf-8')\n"
        "save_enabled(pathlib.Path(drive), 'peer_00', True)\n",
        skill_dir=str(live_route["skills_root"] / "peer_00"),
        drive=str(live_route["drive_root"]),
    )
    response = _get(live_route)
    assert response.status_code == 409, response.text
    assert response.json()["state"]["reason"] == "skill_conflict"


# --------------------------------------------------------------------------
# Tool and WS consumers reach the same verdict
# --------------------------------------------------------------------------

def test_tool_liveness_gate_sees_the_cross_process_disable(live_route):
    drive_root = live_route["drive_root"]
    repo_path = str(live_route["skills_root"])
    assert extension_loader.is_extension_live("fresh_ext", drive_root, repo_path=repo_path)
    _in_another_process(
        "import pathlib\n"
        "from ouroboros.skill_loader import save_enabled\n"
        "save_enabled(pathlib.Path(drive), 'fresh_ext', False)\n",
        drive=str(drive_root),
    )
    assert not extension_loader.is_extension_live("fresh_ext", drive_root, repo_path=repo_path)


def test_ws_dispatch_refuses_after_a_cross_process_disable(live_route, monkeypatch):
    import asyncio

    from ouroboros.gateway import ws as ws_module

    sent: list = []

    class _FakeApp:
        class state:  # noqa: N801
            drive_root = None

    class _FakeWebSocket:
        def __init__(self, app):
            self.app = app

        async def send_text(self, text):
            sent.append(json.loads(text))

    app = _FakeApp()
    app.state.drive_root = live_route["drive_root"]
    socket = _FakeWebSocket(app)
    msg_type = extension_loader.extension_surface_name("fresh_ext", "greet")

    _in_another_process(
        "import pathlib\n"
        "from ouroboros.skill_loader import save_enabled\n"
        "save_enabled(pathlib.Path(drive), 'fresh_ext', False)\n",
        drive=str(live_route["drive_root"]),
    )
    handled = asyncio.run(ws_module._dispatch_extension_message(socket, {}, msg_type))
    assert handled is True
    assert sent, "the WS consumer must answer explicitly"
    assert "not live" in sent[-1]["data"]["message"]


# --------------------------------------------------------------------------
# A mutation that lands during a slow reconcile
# --------------------------------------------------------------------------

def test_a_disable_landing_during_a_slow_reconcile_is_still_caught(live_route, monkeypatch):
    """The dispatch repeats subject + peer checks AFTER reconciliation."""
    from ouroboros.gateway import extensions as gateway_extensions

    real_reconcile = extension_loader.reconcile_extension
    fired = {"n": 0}

    def slow_reconcile(*args, **kwargs):
        state = real_reconcile(*args, **kwargs)
        if fired["n"] == 0:
            fired["n"] += 1
            # The owner disables from another process while reconcile is in flight.
            _in_another_process(
                "import pathlib\n"
                "from ouroboros.skill_loader import save_enabled\n"
                "save_enabled(pathlib.Path(drive), 'fresh_ext', False)\n",
                drive=str(live_route["drive_root"]),
            )
        return state

    monkeypatch.setattr(gateway_extensions, "reconcile_extension", slow_reconcile, raising=False)
    monkeypatch.setattr(extension_loader, "reconcile_extension", slow_reconcile)
    # Force the dispatch down the reconcile branch by dropping the live bundle.
    extension_loader.unload_extension("fresh_ext")
    response = _get(live_route)
    assert fired["n"] == 1, "the reconcile branch was not exercised"
    assert response.status_code == 409, response.text
    assert response.json()["state"]["reason"] == "disabled"


# --------------------------------------------------------------------------
# Cost: hashes counted, not elapsed time
# --------------------------------------------------------------------------

def _count_hashes(monkeypatch):
    calls: list = []
    real = skill_loader.compute_content_hash

    def counted(skill_dir, **kwargs):
        calls.append(pathlib.Path(skill_dir).name)
        return real(skill_dir, **kwargs)

    monkeypatch.setattr(skill_loader, "compute_content_hash", counted)
    return calls


def test_a_route_request_hashes_only_the_selected_skill(live_route, monkeypatch):
    calls = _count_hashes(monkeypatch)
    assert _get(live_route).status_code == 200
    assert calls, "the selected subject must still be hashed"
    assert set(calls) == {"fresh_ext"}, calls


def test_the_request_hash_budget_is_pinned_on_both_dispatch_paths(live_route, monkeypatch):
    """A budget, not a benchmark: each hash is one full payload read.

    The live path needs exactly one fresh subject identity. The reconcile path
    re-reads around the slow reconciliation on purpose; what it may NOT do is
    re-derive the same subject twice inside one step, which is what the first
    implementation did (it measured five).
    """
    calls = _count_hashes(monkeypatch)
    assert _get(live_route).status_code == 200
    assert len(calls) == 1, calls

    extension_loader.unload_extension("fresh_ext")
    calls.clear()
    assert _get(live_route).status_code == 200
    assert set(calls) == {"fresh_ext"}, calls
    assert len(calls) <= 4, (
        "the reconcile dispatch path grew another full payload hash", calls,
    )


def test_hash_count_does_not_grow_with_the_number_of_installed_peers(live_route, monkeypatch, tmp_path):
    calls = _count_hashes(monkeypatch)
    assert _get(live_route).status_code == 200
    with_twelve = len(calls)

    skills_root = live_route["skills_root"]
    for index in range(12, 60):
        peer = _write_ext(skills_root, f"peer_{index:02d}", permissions=["tool"],
                          plugin="def register(api):\n    pass\n")
        (peer / "bulk.bin").write_bytes(b"z" * 40_000)
        save_enabled(live_route["drive_root"], f"peer_{index:02d}", False)
    calls.clear()
    assert _get(live_route).status_code == 200
    assert set(calls) == {"fresh_ext"}, calls
    assert len(calls) == with_twelve, (with_twelve, len(calls))


def test_the_ws_consumer_hashes_only_the_selected_skill(live_route, monkeypatch):
    import asyncio

    from ouroboros.gateway import ws as ws_module

    class _FakeApp:
        class state:  # noqa: N801
            drive_root = None

    class _FakeWebSocket:
        def __init__(self, app):
            self.app = app
            self.sent: list = []

        async def send_text(self, text):
            self.sent.append(text)

    app = _FakeApp()
    app.state.drive_root = live_route["drive_root"]
    socket = _FakeWebSocket(app)
    msg_type = extension_loader.extension_surface_name("fresh_ext", "greet")
    calls = _count_hashes(monkeypatch)
    asyncio.run(ws_module._dispatch_extension_message(socket, {}, msg_type))
    assert set(calls) <= {"fresh_ext"}, calls
