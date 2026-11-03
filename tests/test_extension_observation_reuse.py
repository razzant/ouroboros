"""Observation-local reuse, not a freshness cache, through extension consumers."""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader as loader, extension_liveness as liveness
from ouroboros import skill_loader, skill_peer_inventory, skill_identity
from ouroboros.tools import extension_dispatch
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_extension_freshness_cross_process import live_route as live_route, _in_another_process
from tests.test_extensions_api import _write_ext

REPO = pathlib.Path(__file__).resolve().parents[1]


def _consumer(fixture, surface, monkeypatch):
    """Use the actual transport/dispatch entry; only the handler and Safety are inert."""
    import ouroboros.safety
    from ouroboros.gateway import ws
    name = loader.extension_surface_name("fresh_ext", "greet")
    monkeypatch.setattr(ouroboros.safety, "check_safety", lambda *a, **kw: (True, ""))
    ctx = ToolContext(repo_dir=REPO, drive_root=fixture["drive_root"], task_id="aabbccdd")
    sent = []

    class Socket:
        app = SimpleNamespace(state=SimpleNamespace(drive_root=fixture["drive_root"], repo_dir=REPO))

        async def send_text(self, value):
            sent.append(json.loads(value))

    def dispatch():
        if surface == "http":
            return fixture["client"].get("/api/extensions/fresh_ext/greet").status_code == 200
        if surface == "ws":
            sent.clear()
            assert asyncio.run(ws._dispatch_extension_message(Socket(), {}, name))
            assert sent
            return sent[-1]["type"] == name + ".reply"
        descriptor, unavailable = extension_dispatch._extension_dispatch_candidate(ctx, name)
        if descriptor is None:
            assert unavailable, "the fixture's named tool must have an explicit liveness refusal"
            return False
        result = extension_dispatch._dispatch_extension_tool_untagged(ctx, name, descriptor, {})
        return result.status == "ok"

    return dispatch


def _read_counts(monkeypatch):
    counts = Counter()
    real_inventory = skill_loader._skill_location_inventory
    real_text = pathlib.Path.read_text
    real_parse = skill_peer_inventory.parse_skill_manifest_text
    real_digest = skill_identity.payload_digest

    def inventory(*args, **kwargs):
        counts["locations"] += 1
        return real_inventory(*args, **kwargs)

    def text(path, *args, **kwargs):
        if path.name in {"SKILL.md", "skill.json"}:
            counts["manifest_text"] += 1
        elif path.name == "enabled.json":
            counts["enabled"] += 1
        return real_text(path, *args, **kwargs)

    def parse(*args, **kwargs):
        counts["peer_parse"] += 1
        return real_parse(*args, **kwargs)

    def digest(path, **kwargs):
        counts["hash:" + pathlib.Path(path).name] += 1
        return real_digest(path, **kwargs)

    for module in (skill_loader, skill_peer_inventory, liveness, loader):
        monkeypatch.setattr(module, "_skill_location_inventory", inventory, raising=False)
    monkeypatch.setattr(pathlib.Path, "read_text", text)
    monkeypatch.setattr(skill_peer_inventory, "parse_skill_manifest_text", parse)
    monkeypatch.setattr(skill_identity, "payload_digest", digest)
    skill_peer_inventory._peer_manifest_facts.cache_clear()  # count from a new process's empty reuse
    return counts


@pytest.mark.parametrize("surface", ["http", "ws", "tool"])
@pytest.mark.parametrize("supervisor_present", [False, True])
def test_dispatch_reuses_locations_but_keeps_each_fresh_state_read(
    live_route, monkeypatch, surface, supervisor_present,
):
    monkeypatch.setattr(loader, "is_server_process", lambda: True)
    supervisor = SimpleNamespace(stop_skill=lambda name: None, snapshot=lambda: {})
    monkeypatch.setattr(loader, "get_global_supervisor", lambda: supervisor if supervisor_present else None)
    dispatch = _consumer(live_route, surface, monkeypatch)
    on = 2  # harmless enabled peers: their manifests stay fresh reads in every observation
    for index in range(on):
        skill_loader.save_enabled(live_route["drive_root"], f"peer_{index:02d}", True)
    counts = _read_counts(monkeypatch)
    observations = 1 if surface == "http" else 3 if surface == "ws" and supervisor_present else 2
    for peers in (12, 60):
        if peers == 60:
            for index in range(12, peers):
                _write_ext(live_route["skills_root"], f"peer_{index:02d}", permissions=["tool"],
                           plugin="def register(api):\n    pass\n")
                skill_loader.save_enabled(live_route["drive_root"], f"peer_{index:02d}", False)
        counts.clear()
        assert dispatch()
        # Every peer's enablement and the subject are read fresh per observation; an OFF
        # peer's manifest is neither read nor parsed, since an OFF peer cannot conflict.
        # An ON peer's bytes are read every time, parsed only while its text is new.
        assert counts == {
            "locations": observations,
            "enabled": observations * (peers + 1),
            "manifest_text": observations * (on + 1),
            **({"peer_parse": on} if peers == 12 else {}),
            "hash:fresh_ext": observations,
        }, (surface, supervisor_present, peers, counts)


@pytest.mark.parametrize("tools", [1, 5])
def test_schema_enumeration_checks_a_skill_once_and_reads_again_next_time(live_route, monkeypatch, tools):
    from ouroboros.extension_registry_state import _tools

    name = loader.extension_surface_name("fresh_ext", "greet")
    descriptor = _tools[name]
    for index in range(1, tools):
        alias = loader.extension_surface_name("fresh_ext", f"greet{index}")
        monkeypatch.setitem(_tools, alias, {**descriptor, "name": alias})
    registry = ToolRegistry(repo_dir=REPO, drive_root=live_route["drive_root"])
    # This test measures the extension enumeration; MCP has its own independent discovery.
    monkeypatch.setattr(registry, "_mcp_rows", lambda **kwargs: [])
    counts = _read_counts(monkeypatch)
    for _ in range(2):
        counts.clear()
        names = {row["function"]["name"] for row in registry.schemas()}
        assert {key for key in _tools if key.startswith(name)} <= names
        assert counts["locations"] == 1
        assert counts["enabled"] == 13 and counts["peer_parse"] == 0  # twelve OFF peers, read fresh
        assert counts["hash:fresh_ext"] == 1
    skill_loader.save_enabled(live_route["drive_root"], "fresh_ext", False)
    counts.clear()
    assert not any(row["function"]["name"].startswith(name) for row in registry.schemas())
    assert counts["locations"] == 1
    assert counts["enabled"] == 13 and counts["peer_parse"] == 0


@pytest.mark.parametrize("surface", ["http", "ws", "tool"])
@pytest.mark.parametrize("change", ["review", "payload", "peer", "membership"])
def test_next_dispatch_observes_cross_process_changes(live_route, monkeypatch, surface, change):
    dispatch = _consumer(live_route, surface, monkeypatch)
    assert dispatch()
    if change == "review":
        code = (
            "from pathlib import Path\nfrom ouroboros.skill_loader import SkillReviewState, save_review_state\n"
            "save_review_state(Path(drive), 'fresh_ext', SkillReviewState(status='pending', content_hash=digest))\n"
        )
    elif change == "payload":
        code = (
            "from pathlib import Path\nimport os\np = Path(selected) / 'plugin.py'\ns = p.stat()\n"
            "raw = p.read_bytes(); changed = raw.replace(b'world', b'earth')\n"
            "assert changed != raw and len(changed) == len(raw)\np.write_bytes(changed)\n"
            "os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))\n"
        )
    else:
        # The peer begins harmless but enabled; change only equal-sized manifest bytes.
        peer = live_route["skills_root"] / "peer_00"
        manifest = peer / "SKILL.md"
        text = manifest.read_text(encoding="utf-8")
        assert "conflicts: []" in text
        manifest.write_text(text.replace("conflicts: []", 'conflicts: ["other_ext"]'), encoding="utf-8")
        skill_loader.save_enabled(live_route["drive_root"], "peer_00", True)
        assert dispatch()
        code = (
            "from pathlib import Path\nimport os\np = Path(peer) / 'SKILL.md'\ns = p.stat()\n"
            "raw = p.read_bytes(); changed = raw.replace(b'other_ext', b'fresh_ext')\n"
            "assert changed != raw and len(changed) == len(raw)\np.write_bytes(changed)\n"
            "os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))\n"
        )
        if change == "membership":
            code += "Path(peer).rename(Path(peer).with_name('peer_new'))\nfrom ouroboros.skill_loader import save_enabled\nsave_enabled(Path(drive), 'peer_new', True)\n"
    _in_another_process(code, drive=str(live_route["drive_root"]), digest=live_route["content_hash"],
                        selected=str(live_route["skill_dir"]), peer=str(live_route["skills_root"] / "peer_00"))
    assert not dispatch(), (surface, change)


@pytest.mark.parametrize("point", ["inside_reconcile", "before_second_observation"])
def test_a_peer_edit_inside_one_ws_dispatch_is_seen_although_its_old_text_was_parsed(live_route, monkeypatch, point):
    """A WS frame reconciles, yields to a thread and observes again. An equal-size conflicting edit with a
    restored mtime landing in either window is refused, though the peer's previous text is already parsed."""
    peer = live_route["skills_root"] / "peer_00" / "SKILL.md"
    peer.write_text(peer.read_text(encoding="utf-8").replace("conflicts: []", 'conflicts: ["other_ext"]'),
                    encoding="utf-8")
    skill_loader.save_enabled(live_route["drive_root"], "peer_00", True)
    dispatch = _consumer(live_route, "ws", monkeypatch)
    assert dispatch()  # the previous text is parsed and held
    landed = []

    def land():
        if not landed:
            _in_another_process(
                "from pathlib import Path\nimport os\np = Path(peer)\ns = p.stat()\n"
                "raw = p.read_bytes(); changed = raw.replace(b'other_ext', b'fresh_ext')\n"
                "assert changed != raw and len(changed) == len(raw)\np.write_bytes(changed)\n"
                "os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))\n", peer=str(peer))
            landed.append(point)

    if point == "inside_reconcile":
        finalize = loader._finalize_extension_reconcile
        monkeypatch.setattr(loader, "_finalize_extension_reconcile", lambda *a, **k: (land(), finalize(*a, **k))[1])
    else:
        to_thread = asyncio.to_thread

        async def boundary(func, *args, **kwargs):
            if getattr(func, "__name__", "") == "runtime_state_for_skill_name":
                land()
            return await to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", boundary)
    assert not dispatch() and landed == [point]


def test_post_load_observation_does_not_reuse_reconcile_start_membership(live_route, monkeypatch):
    from ouroboros.config import load_settings

    original = loader.load_extension
    calls = []

    def load_then_add_peer(*args, **kwargs):
        result = original(*args, **kwargs)
        peer = _write_ext(live_route["skills_root"], "new_conflict", permissions=["tool"],
                          plugin="def register(api):\n    pass\n")
        manifest = peer / "SKILL.md"
        text = manifest.read_text(encoding="utf-8")
        assert "conflicts: []" in text
        manifest.write_text(text.replace("conflicts: []", 'conflicts: ["fresh_ext"]'), encoding="utf-8")
        skill_loader.save_enabled(live_route["drive_root"], "new_conflict", True)
        calls.append(result)
        return result

    loader.unload_extension("fresh_ext")
    monkeypatch.setattr(loader, "load_extension", load_then_add_peer)
    state = loader.reconcile_extension("fresh_ext", live_route["drive_root"], load_settings,
                                      repo_path=str(live_route["skills_root"]))
    assert calls == [None]
    assert state["reason"] == "skill_conflict"
    assert state["desired_live"] is False
    assert live_route["client"].get("/api/extensions/fresh_ext/greet").status_code == 409


@pytest.mark.parametrize("surface", ["http", "ws", "tool"])
def test_next_dispatch_observes_cross_process_key_grant_revocation(live_route, monkeypatch, surface):
    from ouroboros.config import load_settings

    loader.unload_extension("fresh_ext")
    manifest = live_route["skill_dir"] / "SKILL.md"
    text = manifest.read_text(encoding="utf-8")
    # A synthetic key permission; no credential value is read or supplied.
    assert "env_from_settings:" in text
    text = text.replace("env_from_settings: []", "env_from_settings: [TELEGRAM_BOT_TOKEN]")
    assert text != manifest.read_text(encoding="utf-8")
    manifest.write_text(text, encoding="utf-8")
    digest = skill_loader.compute_content_hash(live_route["skill_dir"], manifest_entry="plugin.py")
    skill_loader.save_review_state(live_route["drive_root"], "fresh_ext",
                                   skill_loader.SkillReviewState(status="pass", content_hash=digest))
    skill_loader.save_skill_grants(live_route["drive_root"], "fresh_ext", ["TELEGRAM_BOT_TOKEN"],
                                  content_hash=digest, requested_keys=["TELEGRAM_BOT_TOKEN"])
    selected = skill_loader.find_skill(live_route["drive_root"], "fresh_ext",
                                       repo_path=str(live_route["skills_root"]))
    assert loader.load_extension(selected, load_settings, drive_root=live_route["drive_root"]) is None
    dispatch = _consumer(live_route, surface, monkeypatch)
    assert dispatch()
    _in_another_process(
        "from pathlib import Path\nfrom ouroboros.skill_loader import skill_state_dir_path\n"
        "from ouroboros.utils import read_json_dict, atomic_write_json\n"
        "p = skill_state_dir_path(Path(drive), 'fresh_ext') / 'grants.json'\n"
        "row = read_json_dict(p); assert row['granted_keys'] == ['TELEGRAM_BOT_TOKEN']\n"
        "row['granted_keys'] = []; atomic_write_json(p, row)\n",
        drive=str(live_route["drive_root"]),
    )
    assert not dispatch()
