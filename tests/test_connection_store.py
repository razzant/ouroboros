"""Owner-state connection store (RWS v2, D6): schema, lock, atomic writer,
and the deterministic accidental-access denial in the structured file tools.

Trust-boundary note (D6): the OS account is the DOCUMENTED boundary — there
are deliberately NO shell command blacklists for the store (see the
`ouroboros/connection_store.py` module docstring). What IS tested here is the
deterministic guard set: read_file/list_files/write_file/edit_text refuse the
store and its lock/temp/hardlink aliases.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros.connection_store import (
    add_connection,
    get_connection,
    is_connection_store_path,
    list_connections,
    normalize_ssh_alias,
    pin_connection_host,
    record_bootstrap,
    retire_connection,
    retrust_connection,
)


@pytest.mark.parametrize("alias", ["", "-host", "two words", "bad\nhost", "\x00host"])
def test_connection_alias_rejects_argv_injection(alias):
    with pytest.raises(ValueError):
        normalize_ssh_alias(alias)


def test_connection_store_is_atomic_owner_only_nonsecret_and_soft_retired(tmp_path):
    path = tmp_path / "state" / "remote_connections.json"
    added = add_connection(name="Build host", ssh_alias="build-host", path=path)
    assert added["lifecycle"] == "active"
    assert list_connections(path) == [added]
    assert not any(
        key in added
        for key in ("password", "private_key", "ssh_options", "health", "session")
    )
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600

    pinned = pin_connection_host(added["id"], "host-a", path=path)
    assert pinned["expected_host_id"] == "host-a"
    with pytest.raises(ValueError, match="retrust"):
        pin_connection_host(added["id"], "host-b", path=path)
    with pytest.raises(ValueError, match="active task"):
        retrust_connection(
            added["id"],
            "host-b",
            path=path,
            has_active_lease=True,
        )
    trusted = retrust_connection(added["id"], "host-b", path=path)
    assert trusted["expected_host_id"] == "host-b"
    assert trusted["host_id_history"][0]["superseded_at"]
    assert trusted["host_id_history"][1]["superseded_at"] is None

    retired = retire_connection(added["id"], path=path)
    assert retired["lifecycle"] == "retired"
    assert list_connections(path) == []
    assert get_connection(added["id"], path) == retired


def test_bootstrap_evidence_is_durable_owner_state_and_its_invalidators_clear_it(tmp_path):
    """"That host has a compatible executor installed" is OWNER STATE, so it persists.

    It used to be a process-local dict entry, which made a Home restart silently
    equivalent to never having bootstrapped: the New Project picker went empty and
    nothing but another Bootstrap could bring it back. The store is the right home —
    the claim is about the host, established by an owner action, and it survives.

    What must NOT persist is the claim's validity after the two operations that
    invalidate it: retrust (a different host was never proven compatible) and retire.
    """
    path = tmp_path / "state" / "remote_connections.json"
    added = add_connection(name="Build host", ssh_alias="build-host", path=path)
    assert added["bootstrapped_at"] is None
    assert added["bootstrap_build"] == ""
    assert "bootstrapped_at" in get_connection(added["id"], path)

    pin_connection_host(added["id"], "host-a", path=path)
    recorded = record_bootstrap(added["id"], build="execd-1.2.3", path=path)
    assert recorded["bootstrapped_at"]
    assert recorded["bootstrap_build"] == "execd-1.2.3"
    # Durable: a fresh read of the file (a new process would do exactly this) sees it.
    assert get_connection(added["id"], path)["bootstrapped_at"] == recorded["bootstrapped_at"]
    # Still no secrets, and the durable half never gained a live/session field.
    assert not any(
        key in recorded
        for key in ("password", "private_key", "health", "session", "health_fresh")
    )

    trusted = retrust_connection(added["id"], "host-b", path=path)
    assert trusted["bootstrapped_at"] is None
    assert trusted["bootstrap_build"] == ""

    record_bootstrap(added["id"], build="execd-1.2.3", path=path)
    retired = retire_connection(added["id"], path=path)
    assert retired["bootstrapped_at"] is None
    with pytest.raises(ValueError, match="connection_retired"):
        record_bootstrap(added["id"], build="execd-1.2.3", path=path)
    with pytest.raises(KeyError):
        record_bootstrap("conn_missing", path=path)


def test_connection_store_locked_updates_do_not_lose_rows(tmp_path):
    path = tmp_path / "state" / "remote_connections.json"

    def add(index):
        return add_connection(
            name=f"Host {index}",
            ssh_alias=f"host-{index}",
            path=path,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(add, range(12)))

    assert len(rows) == 12
    assert len(list_connections(path)) == 12
    assert len({row["id"] for row in rows}) == 12
    assert not path.with_name(path.name + ".lock").exists()


def test_connection_store_rejects_malformed_existing_state(tmp_path):
    path = tmp_path / "state" / "remote_connections.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        list_connections(path)
    with pytest.raises(ValueError, match="malformed"):
        add_connection(name="Host", ssh_alias="host", path=path)
    assert path.read_text(encoding="utf-8") == "{broken"


def test_connection_store_rejects_unsupported_schema(tmp_path):
    path = tmp_path / "state" / "remote_connections.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"_schema_version": 999, "connections": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported schema"):
        list_connections(path)


def test_connection_state_predicate_covers_store_lock_temp_and_hardlink(tmp_path):
    path = tmp_path / "state" / "remote_connections.json"
    add_connection(name="Host", ssh_alias="host", path=path)
    assert is_connection_store_path(path, store_path=path)
    assert is_connection_store_path(
        path.with_name(path.name + ".lock"),
        store_path=path,
    )
    assert is_connection_store_path(
        path.with_name(f".{path.name}.tmp.1.deadbeef"),
        store_path=path,
    )
    hardlink = path.with_name("alias.json")
    os.link(path, hardlink)
    assert is_connection_store_path(hardlink, store_path=path)
    assert not is_connection_store_path(
        path.with_name("remote_connections_archive.json"), store_path=path,
    )
    assert json.loads(path.read_text(encoding="utf-8"))["_schema_version"] == 1


@pytest.fixture
def _denial_ctx(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    store = drive / "state" / "remote_connections.json"
    add_connection(name="Host", ssh_alias="host", path=store)
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)
    return ToolContext(repo_dir=repo, drive_root=drive), store


def test_file_tools_refuse_connection_store_reads_and_aliases(_denial_ctx):
    from ouroboros.tools.core import _list_files, _read_file

    ctx, store = _denial_ctx
    os.link(store, store.with_name("connection-alias.json"))

    for rel in (
        "state/remote_connections.json",
        "state/remote_connections.json.lock",
        "state/connection-alias.json",
    ):
        result = _read_file(ctx, rel, root="runtime_data")
        assert "DATA_READ_BLOCKED" in result, rel
        assert "owner-only" in result

    listed = json.loads(_list_files(ctx, path="state", root="runtime_data"))
    assert "state/remote_connections.json" not in listed
    assert "state/connection-alias.json" not in listed
    # The denial hides only store aliases, not the rest of the directory.
    (store.parent / "unrelated.json").write_text("{}", encoding="utf-8")
    listed = json.loads(_list_files(ctx, path="state", root="runtime_data"))
    assert "state/unrelated.json" in listed


def test_file_tools_refuse_connection_store_writes_and_edits(_denial_ctx):
    from ouroboros.tools.core import _edit_text, _write_file

    ctx, store = _denial_ctx
    before = store.read_text(encoding="utf-8")

    written = _write_file(
        ctx,
        path="state/remote_connections.json",
        content="{}",
        root="runtime_data",
    )
    assert "DATA_WRITE_BLOCKED" in written
    edited = _edit_text(
        ctx,
        path="state/remote_connections.json",
        old_str="active",
        new_str="retired",
        root="runtime_data",
    )
    assert "EDIT_TEXT_BLOCKED" in edited
    assert store.read_text(encoding="utf-8") == before
