"""query_data names its data planes and reads another plane only under existing read authority.

The matrix uses the real registry consumers. For every actor shape the decision
to read the installation owner's (canonical) queue file uses the same physical
resource permission as read_file, not that tool's availability: ordinary roots
and helpers inherit the canonical runtime alias; explicit resource restrictions
and a Presence ceiling still narrow it. Native
reviewers stay on their pinned plane even though read_file could reach live
data. Blobs are caller/child-local first; the owner plane is tried only when the
local bytes are proven absent, never after a denied or corrupt local read.
"""
from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from ouroboros import store, store_import
from ouroboros.tools.registry import ToolContext, ToolRegistry

CANONICAL_QUEUE = b'{"plane":"canonical","rows":[1]}'


@pytest.fixture
def planes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    canonical = home / "Ouroboros" / "data"
    drives = {name: canonical / "state" / "headless_tasks" / name / "data" for name in ("forked", "helper")}
    repo, pin = tmp_path / "repo", tmp_path / "pin"
    for path in (canonical / "state", *drives.values(), repo, pin / "state"):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    store_import.ensure_current_store(canonical, with_children=False)
    store.update_task(canonical, "root-task", lambda _: {"status": "running", "root_task_id": "root-task"})
    store.update_task(canonical, "helper", lambda _: {"status": "running", "parent_task_id": "root-task",
                                                      "root_task_id": "root-task"})
    (canonical / "state/queue_snapshot.json").write_bytes(CANONICAL_QUEUE)
    (canonical / "state/queue_dispatch.json").write_bytes(b'{"plane":"canonical-dispatch"}')
    (drives["helper"] / "state").mkdir(parents=True)
    (drives["helper"] / "state/queue_snapshot.json").write_bytes(b'{"plane":"helper-local"}')
    (pin / "state/queue_snapshot.json").write_bytes(b'{"plane":"pin"}')
    return {"home": home, "canonical": canonical, "repo": repo, "pin": pin, **drives}


def _contexts(planes):
    canonical, repo = planes["canonical"], planes["repo"]
    lineage = {"budget_drive_root": str(canonical), "parent_task_id": "root-task", "root_task_id": "root-task"}
    return {
        "root": ToolContext(repo_dir=repo, drive_root=canonical, task_id="root-task"),
        "forked_root": ToolContext(repo_dir=repo, drive_root=planes["forked"], task_id="forked",
                                   budget_drive_root=str(canonical), task_metadata={"budget_drive_root": str(canonical)}),
        "helper": ToolContext(repo_dir=repo, drive_root=planes["helper"], task_id="helper",
                              budget_drive_root=str(canonical), task_constraint={"mode": "local_readonly_subagent"},
                              task_metadata=dict(lineage)),
        "acting": ToolContext(repo_dir=repo, drive_root=planes["helper"], task_id="helper",
                              budget_drive_root=str(canonical), task_metadata=dict(lineage),
                              task_constraint={"mode": "acting_subagent", "surface": "self_worktree",
                                               "root_task_id": "root-task"}),
    }


def _registry(ctx):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry


def _query(registry, **args):
    result = registry.execute_result("query_data", args)
    return result.status, json.loads(result.text)


def _read_admitted(registry, target):
    """read_file attempted this exact file: only a refusal is ``blocked`` (binary bytes may still fail to decode)."""
    return registry.execute_result("read_file", {"path": str(target)}).status != "blocked"


def _blob(root, raw):
    digest = hashlib.sha256(raw).hexdigest()
    path = root / "observability" / "blobs" / f"{digest}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw))
    return digest, path


def _pack(root, task_id, raw):
    """``raw`` as a blob of ``task_id``'s evidence pack on ``root``, without its loose copy."""
    from ouroboros import evidence_packs as packs
    from tests.test_evidence_packs import _task

    _task(root, task_id, blobs=1, payloads=[raw], handle=raw)
    pack_path, sidecar = packs.write_pack(root, task_id, 1, packs.loose_members(root, task_id))
    packs.index_upsert(root, pack_path, sidecar)
    digest = hashlib.sha256(raw).hexdigest()
    (root / "observability/blobs" / f"{digest}.json.gz").unlink()
    return digest, pack_path


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
def test_canonical_queue_reads_follow_the_same_actors_read_file_admission(planes, monkeypatch, mode):
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: mode)
    target = planes["canonical"] / "state/queue_snapshot.json"
    observed = {}
    for name, ctx in _contexts(planes).items():
        registry = _registry(ctx)
        admitted = _read_admitted(registry, target)
        status, result = _query(registry, op="json", kind="queue_snapshot")
        observed[name] = admitted
        assert admitted and status == "ok" and result["data"]["plane"] == "canonical", (name, result)
        assert result["source"]["plane"] == "canonical"
        status, dispatch = _query(registry, op="json", kind="queue_dispatch")
        assert dispatch["data"] == {"plane": "canonical-dispatch"} and dispatch["source"]["plane"] == "canonical"
    # The upstream read alias covers ordinary roots as well as helpers in every mode.
    # query_data follows that existing physical authority rather than adding a grant.
    assert observed == {"root": True, "forked_root": True, "helper": True, "acting": True}


def test_presence_ceiling_narrows_the_canonical_plane_like_read_file(planes, monkeypatch):
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import (PresenceProfileResolution, PresenceResourceTarget,
                                                 PresenceSelection, PresenceToolTarget)
    from tests.frozen_presence_runtime import ResolvedPresenceRuntime

    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    canonical = planes["canonical"]
    target = canonical / "state/queue_snapshot.json"
    for granted, expected in ((target.relative_to(planes["home"]).parent.as_posix(), "ok"), ("Desktop", "blocked")):
        resolution = PresenceProfileResolution(
            active=(PresenceSelection("1" * 64, PresenceToolTarget("builtin", "query_data")),
                    PresenceSelection("2" * 64, PresenceToolTarget("builtin", "read_file")),
                    PresenceSelection("3" * 64, PresenceResourceTarget("user_files", ("read",), granted))),
            missing_required=(), missing_optional=(), orphaned=(),
            runtime=ResolvedPresenceRuntime("main", 10, 10, False),
            profile_fingerprint="a" * 64, selection_fingerprint="b" * 64, required_selections_present=True)
        ceiling = build_presence_capability_ceiling(skill_name="presence-planes", skill_content_hash="c" * 64,
                                                    state_fingerprint="d" * 64, resolution=resolution)
        ctx = ToolContext(repo_dir=planes["repo"], drive_root=planes["forked"], task_id="forked",
                          budget_drive_root=str(canonical),
                          task_metadata={"budget_drive_root": str(canonical)},
                          task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)})
        registry = _registry(ctx)
        assert registry.execute_result("read_file", {"root": "user_files", "path": str(target)}).status == expected
        status, result = _query(registry, op="json", kind="queue_snapshot")
        assert status == expected, result
        if expected == "blocked":
            assert "Presence resource ceiling" in result["message"] and result["plane"] == "canonical"
            assert "data" not in result and "canonical bytes" not in json.dumps(result)


def test_storage_uses_the_installation_or_native_pin_with_existing_read_authority(planes):
    relative = "state/storage_measurement.json"
    target = planes["canonical"] / relative
    target.write_text('{"plane":"installation"}', encoding="utf-8")
    (planes["helper"] / relative).write_text('{"plane":"helper"}', encoding="utf-8")
    ctx = _contexts(planes)["helper"]
    helper = _registry(ctx)
    assert _read_admitted(helper, target)
    status, result = _query(helper, op="json", kind="storage")
    assert status == "ok" and result["data"]["plane"] == "installation" and "removers" in result["data"]
    assert result["source"]["plane"] == "canonical"

    ctx.task_contract = {"resource_policy": {"protected_artifacts": [
        {"paths": [str(target)], "role": "black_box_reference"}]}}
    restricted = _registry(ctx)
    assert not _read_admitted(restricted, target)
    status, refused = _query(restricted, op="json", kind="storage")
    assert status == "blocked" and refused["plane"] == "canonical" and "data" not in refused

    pin = planes["pin"] / relative
    pin.write_text('{"plane":"retained"}', encoding="utf-8")
    native = _registry(ToolContext(repo_dir=planes["repo"], drive_root=planes["pin"], task_id="native",
                                  budget_drive_root=str(planes["canonical"]),
                                  task_metadata={"native_data_root": str(planes["pin"])}))
    status, result = _query(native, op="json", kind="storage")
    assert status == "ok" and result["source"]["plane"] == "native_pin" and result["data"]["plane"] == "retained"
    pin.unlink()
    status, missing = _query(native, op="json", kind="storage")  # no count there: its removers alone
    assert status == "ok" and missing["source"]["plane"] == "native_pin" and set(missing["data"]) == {"removers"}


def test_explicit_child_queue_logs_and_sql_keep_their_local_planes(planes, monkeypatch):
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    contexts = _contexts(planes)
    root = _registry(contexts["root"])
    status, child = _query(root, op="json", kind="queue_snapshot", child_task_id="helper")
    assert status == "ok" and child["data"] == {"plane": "helper-local"} and child["source"]["plane"] == "child"
    forked = _registry(contexts["forked_root"])
    (planes["forked"] / "logs").mkdir()
    (planes["forked"] / "logs/tools.jsonl").write_text('{"local": true}\n', encoding="utf-8")
    status, logs = _query(forked, op="jsonl", kind="tools_log")
    assert status == "ok" and logs["data"] == [{"local": True}] and logs["source"]["plane"] == "caller"
    (planes["forked"] / "state").mkdir(exist_ok=True)
    (planes["forked"] / "state/advisory_review.json").write_text('{"advisory": "fork"}', encoding="utf-8")
    status, advisory = _query(forked, op="json", kind="advisory_review")
    assert advisory["data"] == {"advisory": "fork"} and advisory["source"]["plane"] == "caller"
    status, facts = _query(root, op="sql", query="SELECT task_id FROM tasks ORDER BY task_id")
    assert status == "ok" and facts["source"]["plane"] == "caller"
    status, packs = _query(forked, op="sql", db="packs", query="SELECT 1")
    assert (status, packs["code"], packs["plane"]) == ("unavailable", "no_packs", "caller")
    status, task = _query(forked, op="json", kind="task_result", ids=["root-task"], select="status")
    assert task["data"] == "running" and task["source"]["plane"] == "canonical"


def test_blobs_fall_back_to_the_owner_only_after_a_proven_local_miss(planes, monkeypatch):
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    contexts = _contexts(planes)
    helper, forked = _registry(contexts["helper"]), _registry(contexts["forked_root"])
    owner_raw = b'{"owner": "canonical bytes"}'
    digest, owner_path = _blob(planes["canonical"], owner_raw)
    owner_bytes = owner_path.read_bytes()
    status, result = _query(helper, op="blob", sha256=digest)
    assert status == "ok" and result["data"] == {"owner": "canonical bytes"} and result["source"]["plane"] == "canonical"
    assert _read_admitted(forked, owner_path)
    status, served = _query(forked, op="blob", sha256=digest)
    assert status == "ok" and served["data"] == {"owner": "canonical bytes"} and served["source"]["plane"] == "canonical"
    # A real task restriction, not the obsolete absence of a root read alias, denies fallback.
    restricted_ctx = _contexts(planes)["forked_root"]
    restricted_ctx.task_contract = {"resource_policy": {"protected_artifacts": [
        {"paths": [str(owner_path)], "role": "black_box_reference"}]}}
    restricted = _registry(restricted_ctx)
    assert not _read_admitted(restricted, owner_path)
    status, refused = _query(restricted, op="blob", sha256=digest)
    assert status == "blocked" and refused["plane"] == "canonical", refused
    assert "data" not in refused and "canonical bytes" not in json.dumps(refused)
    assert refused["message"].startswith("blob not found in the caller plane; the canonical plane is not readable")
    # Same digest present locally: local bytes win and are labelled as such.
    _, local_path = _blob(planes["helper"], owner_raw)
    status, local = _query(helper, op="blob", sha256=digest)
    assert local["source"]["plane"] == "caller" and local["data"] == {"owner": "canonical bytes"}
    # Corrupt local bytes never fall through to the owner's copy.
    local_path.write_bytes(gzip.compress(b'{"owner": "tampered"}'))
    status, corrupt = _query(helper, op="blob", sha256=digest)
    assert status == "error" and corrupt["plane"] == "caller" and "corrupt" in corrupt["message"], corrupt
    # A denied local path (escaping link) never falls through either.
    local_path.unlink()
    outside = planes["home"].parent / "outside.json.gz"
    outside.write_bytes(gzip.compress(owner_raw))
    local_path.symlink_to(outside)
    status, denied = _query(helper, op="blob", sha256=digest)
    assert status == "blocked" and denied["plane"] == "caller" and "escapes" in denied["message"], denied
    local_path.unlink()
    # Packed owner bytes are read through the same admitted logical location.
    packed_digest, _ = _pack(planes["canonical"], "packed-task", b'{"packed": "owner pack"}')
    status, packed = _query(helper, op="blob", sha256=packed_digest)
    assert status == "ok" and packed["data"] == {"packed": "owner pack"} and packed["source"]["plane"] == "canonical"
    status, missing = _query(helper, op="blob", sha256="0" * 64)
    assert status == "unavailable" and missing["message"].startswith("blob not found in the caller or canonical plane;")
    assert "memory_read(source_ref=" in missing["message"]  # the miss names the task-source reader, not another plane
    assert owner_path.read_bytes() == owner_bytes  # reads never touch the owner's bytes


def test_money_sql_reads_the_installations_one_ledger_from_any_focus(planes):
    """A drive of its own holds no money ledger, so a helper's db=money answered not_found while the
    installation's ledger stood; it reads the owner's, named as such, and the owner still reads its own."""
    from ouroboros import usage_accounting as ua

    ua.record_subscription_session("sess-1", drive_root=planes["canonical"], route="codex", task_id="helper",
                                   prompt_tokens=None, completion_tokens=None, cached_tokens=None, spend_usd=0.0)
    query = dict(op="sql", db="money", query="SELECT task_id FROM attempts")
    contexts = _contexts(planes)
    for name in ("forked_root", "helper"):
        status, result = _query(_registry(contexts[name]), **query)
        assert status == "ok" and result["data"]["rows"] == [["helper"]], (name, result)
        assert result["source"]["plane"] == "canonical"
    status, own = _query(_registry(contexts["root"]), **query)
    assert status == "ok" and own["source"]["plane"] == "caller"
    _status, current = _query(_registry(contexts["helper"]), op="sql", query="SELECT count(*) AS n FROM tasks")
    assert "caller" in (current.get("plane"), current["source"].get("plane"))  # the task facts store stays the caller's


def test_packed_blobs_need_the_actors_read_of_the_pack_file_itself(planes, monkeypatch):
    """Packed bytes are read from the pack, never the digest's loose name: each file the reader
    opens is admitted like read_file of that file, so a protected pack refuses the blob too."""
    from ouroboros import evidence_packs as packs

    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    canonical = planes["canonical"]
    raw = b'{"packed": "protected owner pack"}'
    digest, pack_path = _pack(canonical, "packed-task", raw)
    loose = canonical / "observability/blobs" / f"{digest}.json.gz"
    helper = _registry(_contexts(planes)["helper"])
    # read_file admits the pack itself (as binary it cannot decode it), so packed reads stay served.
    assert _read_admitted(helper, pack_path)
    status, served = _query(helper, op="blob", sha256=digest)
    assert status == "ok" and served["data"] == {"packed": "protected owner pack"}, served
    ctx = _contexts(planes)["helper"]
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [
        {"paths": [str(pack_path)], "role": "black_box_reference"}]}}
    protected = _registry(ctx)
    assert not _read_admitted(protected, pack_path)
    for corrupt_loose in (False, True):  # corrupt loose bytes never fall through to the refused pack
        if corrupt_loose:
            loose.write_bytes(gzip.compress(b'{"packed": "tampered"}'))
        status, refused = _query(protected, op="blob", sha256=digest)
        assert status == "blocked" and refused["code"] == "access_denied" and refused["plane"] == "canonical", refused
        assert "RESOURCE_POLICY_BLOCKED" in refused["message"] and "protected owner pack" not in json.dumps(refused)
    loose.unlink()
    _, local = _blob(planes["helper"], raw)  # a digest neither grants nor refuses: the caller's own copy serves
    status, served = _query(protected, op="blob", sha256=digest)
    assert status == "ok" and served["source"]["plane"] == "caller", served
    local.unlink()
    outside = planes["home"].parent / "outside.xz"  # a pack linked outside its plane is refused like a loose link
    # Finish this fixture's read before moving its open file (required on Windows).
    with packs._READERS_LOCK:
        packs._READERS.pop(str(pack_path))._stream.close()
    pack_path.rename(outside)
    pack_path.symlink_to(outside)
    status, escaped = _query(helper, op="blob", sha256=digest)
    assert status == "blocked" and "escapes" in escaped["message"], escaped
    pack_path.unlink()  # an indexed pack that is gone stays absence, not a refusal
    status, missing = _query(helper, op="blob", sha256=digest)
    assert status == "unavailable" and missing["message"].startswith("blob not found in the caller or canonical plane;"), missing


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
def test_a_selected_owned_child_is_a_local_blob_plane(planes, monkeypatch, mode):
    """child_task_id selects an owned descendant's drive through the caller's lineage, the
    admission its logs and queue already use. The canonical read alias also admits read_file,
    but does not select a child's plane. Owner fallback checks each exact file, and an explicit
    child selector outside the lineage stays refused despite the broader read alias."""
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: mode)
    canonical, own = planes["canonical"], planes["forked"]
    store.update_task(canonical, "forked", lambda _: {"status": "running", "root_task_id": "forked"})
    store.update_task(canonical, "forked-child", lambda _: {"status": "running", "parent_task_id": "forked",
                                                            "root_task_id": "forked"})
    child = canonical / "state" / "headless_tasks" / "forked-child" / "data"
    (child / "logs").mkdir(parents=True)
    (child / "logs/tools.jsonl").write_text('{"child": true}\n', encoding="utf-8")
    child_loose, child_path = _blob(child, b'{"child": "loose"}')
    child_packed, _ = _pack(child, "forked-child", b'{"child": "packed"}')
    own_loose, _ = _blob(own, b'{"caller": "loose"}')
    own_packed, _ = _pack(own, "forked", b'{"caller": "packed"}')
    forked = _registry(_contexts(planes)["forked_root"])
    assert _read_admitted(forked, child_path)  # the alias grants a read, not a child selector
    status, logs = _query(forked, op="jsonl", kind="tools_log", child_task_id="forked-child")
    assert status == "ok" and logs["data"] == [{"child": True}] and logs["source"]["plane"] == "child", logs
    for digest, child_task_id, expected, plane in (
            (child_loose, "forked-child", {"child": "loose"}, "child"),
            (child_packed, "forked-child", {"child": "packed"}, "child"),
            (own_loose, "", {"caller": "loose"}, "caller"), (own_packed, "", {"caller": "packed"}, "caller")):
        status, blob = _query(forked, op="blob", sha256=digest, child_task_id=child_task_id)
        assert status == "ok" and blob["data"] == expected and blob["source"]["plane"] == plane, blob
    # The owner fallback stays the actor's existing read of each exact loose or pack file.
    owner_loose, owner_path = _blob(canonical, b'{"owner": "loose"}')
    owner_packed, owner_pack = _pack(canonical, "owner-task", b'{"owner": "packed"}')
    for digest, physical, expected in ((owner_loose, owner_path, {"owner": "loose"}),
                                       (owner_packed, owner_pack, {"owner": "packed"})):
        admitted = _read_admitted(forked, physical)
        assert admitted is True
        status, blob = _query(forked, op="blob", sha256=digest, child_task_id="forked-child")
        assert status == "ok" and blob["data"] == expected and blob["source"]["plane"] == "canonical", blob
    # A drive outside the caller's lineage is refused before any of its bytes are read.
    foreign, _ = _blob(planes["helper"], b'{"foreign": "child"}')
    status, refused = _query(forked, op="blob", sha256=foreign, child_task_id="helper")
    assert status == "blocked" and refused["code"] == "access_denied" and refused["plane"] == "child", refused
    assert "outside the caller's lineage" in refused["message"] and "data" not in refused


def test_query_data_resource_admission_does_not_require_the_read_file_tool(planes, monkeypatch):
    """A withheld tool name is not a revoked resource: query_data has its own tool admission."""
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    ctx = _contexts(planes)["helper"]
    ctx.task_contract = {"disabled_tools": ["read_file"]}
    registry = _registry(ctx)
    target = planes["canonical"] / "state/queue_snapshot.json"
    assert registry.execute_result("read_file", {"path": str(target)}).status == "blocked"
    status, queue = _query(registry, op="json", kind="queue_snapshot")
    assert status == "ok" and queue["data"]["plane"] == "canonical"


def test_native_reviewers_stay_on_their_pinned_plane(planes, monkeypatch):
    from ouroboros.review_native_episode import inspection_registry
    from ouroboros.usage_accounting import UsageScope, usage_scope

    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    canonical, pin = planes["canonical"], planes["pin"]
    digest, _ = _blob(canonical, b'{"live": "only"}')
    pinned_loose, _ = _blob(pin, b'{"pin": "loose"}')
    pinned_packed, _ = _pack(pin, "pinned-task", b'{"pin": "packed"}')
    with usage_scope(UsageScope(drive_root=canonical, task_id="root-task", root_task_id="root-task")):
        registry, inspection, _ = inspection_registry(str(planes["repo"]), pin, "reviewer")
        assert _read_admitted(registry, canonical / "state/queue_snapshot.json")  # read_file could reach live data
        status, queue = _query(registry, op="json", kind="queue_snapshot")
        assert queue["data"] == {"plane": "pin"} and queue["source"]["plane"] == "native_pin"
        for pinned, expected in ((pinned_loose, {"pin": "loose"}), (pinned_packed, {"pin": "packed"})):
            status, blob = _query(registry, op="blob", sha256=pinned)  # the pin is this reviewer's own drive
            assert status == "ok" and blob["data"] == expected and blob["source"]["plane"] == "caller", blob
        status, blob = _query(registry, op="blob", sha256=digest)
        assert status == "unavailable" and blob["message"].startswith("blob not found in the caller plane;"), blob
        (pin / "state/queue_snapshot.json").unlink()
        status, gone = _query(registry, op="json", kind="queue_snapshot")
        assert status == "unavailable" and gone["plane"] == "native_pin" and "data" not in gone
        assert str(canonical) not in gone["message"]  # decoded text, not JSON-escaped Windows paths
