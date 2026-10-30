"""``PluginAPI.read_memory`` (2.1) at its consumers: a reviewed extension's registered GET route served by the
gateway in process and in the out-of-process child, a 2.0 payload beside it, and the agent's ``memory_read``
sharing the store's one composition of a record without reading its room."""
from __future__ import annotations

import asyncio
import json
import pathlib
import urllib.parse
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.contracts.plugin_api import MemoryReadError
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _add_fake_native_dep,
    _clear_loader_state,
    _mark_isolated_deps_installed,
    _prepare_extension,
)
from tests.test_extension_route_streaming import collect_response

REPO = pathlib.Path(__file__).resolve().parents[1]
MIND = {"kind": "mind", "task_id": "t1", "focus": {"role": "root", "task_id": "t1"}}
ATLAS = '''
import json
from ouroboros.contracts.plugin_api import MemoryReadError

API = []


def _read(request):
    query = dict(request.query_params)
    try:
        return {"answer": API[0].read_memory(query["op"], json.loads(query.get("params") or "{}"))}
    except MemoryReadError as exc:
        return {"refused": exc.code, "current_revision": exc.current_revision}


def register(api):
    API.append(api)
    api.register_route("atlas/read", _read, methods=("GET",))
'''
LEGACY = '''
def register(api):
    api.register_route("hello", lambda request: {"generation": api.get_runtime_info()["plugin_api_version"]})
'''


def _memory(drive_root: pathlib.Path) -> dict:
    """Chronicle written by its producer inside the extension host's own data root."""
    store = ChronicleStore(drive_root)
    span = {"start": "2026-10-20T10:00:00+00:00", "end": "2026-10-24T18:00:00+00:00", "incomplete": False}
    page = store.publish_page(room_id="1", text="Слова страницы " * 4000, covers={"mode": "range", "rows": ["a"],
                                                                                  "stream_span": [0, 0], "ts_span": span},
                              author=MIND)
    fix = store.correct(page.record["id"], "Исправление.", MIND)
    assert page.ok and fix.ok
    return {"page": page.record["id"], "fix": fix.record["id"]}


def _dispatch(drive_root: pathlib.Path, skill: str, rest: str, query: dict) -> tuple:
    from starlette.requests import Request

    from ouroboros.gateway.extensions import api_extension_dispatch

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request({
        "type": "http", "method": "GET", "path": f"/api/extensions/{skill}/{rest}",
        "path_params": {"skill": skill, "rest": rest}, "query_string": urllib.parse.urlencode(query).encode(),
        "headers": [], "scheme": "http", "server": ("127.0.0.1", 0), "client": ("127.0.0.1", 0),
        "app": SimpleNamespace(state=SimpleNamespace(drive_root=drive_root, repo_dir=REPO)),
    }, receive)
    events = asyncio.run(collect_response(asyncio.run(api_extension_dispatch(request))))
    return events[0]["status"], json.loads(b"".join(event.get("body", b"") for event in events))


def _reads(drive_root: pathlib.Path, skill: str, ids: dict) -> list:
    """The same questions a module widget's client asks, answered through the skill's route."""
    def ask(op, params):
        status, body = _dispatch(drive_root, skill, "atlas/read", {"op": op, "params": json.dumps(params)})
        assert status == 200, body
        return body

    record = {"kind": "record", "id": ids["page"]}
    first = ask("document", {"address": record, "length": 20_000})["answer"]
    rest = ask("document", {"address": record, "start": first["next"]["start"], "length": 100_000,
                            "revision": first["next"]["revision"]})["answer"]
    return [ask("catalog", {"area": "rooms"}), ask("catalog", {"area": "records", "room_id": "1"}),
            first["revision"], first["text"] + rest["text"], rest["complete"],
            ask("document", {"address": record, "revision": ids["page"]})["answer"]["later_corrections"]["count"],
            ask("changes", {}), ask("document", {"address": {"kind": "record", "id": "absent"}})]


def _load(tmp_path, name, body, frontmatter, *, isolated):
    loaded, repo_root, drive_root = _prepare_extension(tmp_path, name, body, permissions=["route"],
                                                       extra_frontmatter=frontmatter)
    if isolated:
        _add_fake_native_dep(loaded)
        _mark_isolated_deps_installed(drive_root, loaded)
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root,
                                           _force_in_process=not isolated) is None
    return repo_root, drive_root


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "out_of_process"])
def test_a_registered_route_reads_memory_through_the_plugin_api(tmp_path, monkeypatch, isolated):
    repo_root, drive_root = _load(tmp_path, "atlas_reader", ATLAS,
                                  'plugin_api: "2.1"\n' + ("dependencies:\n  - dummy_pkg\n" if isolated else ""),
                                  isolated=isolated)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(repo_root))
    spec = extension_loader.list_routes()["/api/extensions/atlas_reader/atlas/read"]
    assert spec.get("out_of_process", False) is isolated
    ids = _memory(drive_root)
    rooms, records, revision, text, complete, later, changes, absent = _reads(drive_root, "atlas_reader", ids)
    assert rooms["answer"]["chronicle"]["state"] == "present" and rooms["answer"]["entries"][0]["label"] == "Main"
    assert [(entry["id"], entry["kind"], entry.get("period"), entry.get("period_basis"))
            for entry in records["answer"]["entries"]] == [
        (ids["page"], "page", "2026-10-20 10:00 → 2026-10-24 18:00", "rows"), (ids["fix"], "correction", None, None)]
    assert revision == ids["fix"] and complete and text.startswith("Слова страницы") and text.endswith("Исправление.")
    assert later == 1 and changes["answer"]["since_known"] is False
    assert absent == {"refused": "not_found", "current_revision": None}


def test_both_execution_modes_answer_identically(tmp_path, monkeypatch):
    answers = []
    for isolated in (False, True):
        root = tmp_path / ("child" if isolated else "host")
        root.mkdir()
        repo_root, drive_root = _load(root, f"atlas_{int(isolated)}", ATLAS,
                                      'plugin_api: "2.1"\n' + ("dependencies:\n  - dummy_pkg\n" if isolated else ""),
                                      isolated=isolated)
        monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda repo=repo_root: str(repo))
        ids = _memory(drive_root)
        answers.append(json.loads(json.dumps(_reads(drive_root, f"atlas_{int(isolated)}", ids)).replace(
            ids["page"], "PAGE").replace(ids["fix"], "FIX")))
    host, child = answers
    for answer in (host, child):  # publication times differ between the two producers' runs
        for entry in answer[1]["answer"]["entries"]:
            entry.pop("ts")
        answer[6]["answer"]["position"].pop("record_id")
        answer[6]["answer"]["newest"].pop("record_id")
    assert host == child


def test_a_2_0_payload_keeps_loading_and_serving_beside_2_1(tmp_path, monkeypatch):
    repo_root, drive_root = _load(tmp_path, "older_widget", LEGACY, 'plugin_api: "2.0"\n', isolated=False)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(repo_root))
    assert _dispatch(drive_root, "older_widget", "hello", {}) == (200, {"generation": "2.3"})


def test_a_read_after_unload_began_is_refused(tmp_path):
    api = extension_loader.PluginAPIImpl(extension_loader._PluginAPIConfig(
        skill_name="sk", permissions=("route",), env_allowlist=(), state_dir=tmp_path, settings_reader=lambda: {},
        drive_root=tmp_path))
    assert api.read_memory("changes")["since_known"] is False
    api._close_runtime_access()
    with pytest.raises(MemoryReadError) as refused:
        api.read_memory("changes")
    assert refused.value.code == "unavailable"


def test_memory_read_of_one_record_reads_no_room_and_no_other_account(tmp_path, monkeypatch):
    """The agent tool now shares ``ChronicleStore.interpret``: a record and its revisions read by address."""
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0001", current_chat_id=1)
    store = ChronicleStore(tmp_path)
    assert "head" in _memory_read(ctx)  # activates the empty install once, as a real first call does
    first = store.publish_page(room_id="1", text="Original words.", author=MIND,
                               covers={"mode": "range", "rows": ["a"], "stream_span": [0, 0]}).record
    fix = store.correct(first["id"], "Later words.", MIND).record
    account = store.publish_account(room_id="2", text="Told across rooms.", sources=[{"id": first["id"],
                                                                                    "revision": first["id"]}], author=MIND)
    for n in range(40):  # an unrelated room history the record's read must not walk
        store.write_note(room_id="1", task_id="t1", text=f"note {n}", author=MIND)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a single record's read walked a room or every account")

    monkeypatch.setattr(ChronicleStore, "room_records", refuse)
    monkeypatch.setattr(ChronicleStore, "accounts", refuse)
    as_cited = _memory_read(ctx, node_id=first["id"], revision=first["id"])
    assert "later corrections not shown: " + fix["id"] in as_cited and "Later words." not in as_cited.split("text:")[0]
    assert f"informs account {account.record['id']} (revision {first['id']} read" in as_cited
    now = _memory_read(ctx, node_id=first["id"])
    assert f"correction {fix['id']}" in now and "Later words." in now
