"""The packing dispatcher's store fact is O(1): a counter the store keeps where terminal rows are
written, read as one meta row on every look, never a read over the terminal rows."""
from __future__ import annotations

from ouroboros import evidence_packer as packer, store
from tests.test_evidence_packs import _result, _task, root as root  # noqa: F401  (the shared fixture)


def test_the_terminal_fact_is_one_meta_read_that_moves_only_with_terminal_rows(root):
    """Both directions: the fact moves on a task closing, a closed task written again or touched, and
    never on a live task's writes; reading it touches no task row whatever the history holds."""
    for number in range(40):
        _task(root, f"old-{number}", blobs=1)
    statements: list = []
    with store.open_store(root, write=False) as txn:
        txn.conn.set_trace_callback(statements.append)  # the thread's cached connection serves every look
    try:
        first = packer.terminal_revision(root)
        reads = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        assert reads and not any("tasks" in s.lower() for s in reads), reads
        _task(root, "live", blobs=1, status="running")
        _result(root, "live", "running", step=2)
        with store.open_store(root, write=True) as txn:
            txn.touch("live")
        assert packer.terminal_revision(root) == first  # a live task's writes move nothing
        _result(root, "live", "completed")
        second = packer.terminal_revision(root)
        assert second != first  # closing moves it
        _result(root, "old-3", note="written again")
        third = packer.terminal_revision(root)
        with store.open_store(root, write=True) as txn:
            txn.touch("old-7")
        assert len({first, second, third, packer.terminal_revision(root)}) == 4
    finally:
        with store.open_store(root, write=False) as txn:
            txn.conn.set_trace_callback(None)
