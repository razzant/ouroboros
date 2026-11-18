"""Read addressed question mail and record an admitted Continue binding in tests."""
import queue
from types import SimpleNamespace

from ouroboros.task_store import patch_document


def drain(root, task_id):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    ctx = SimpleNamespace(task_id=task_id, task_metadata={}, drive_root=root, budget_drive_root=root)
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), root, task_id, None, set(), owner_ctx=ctx)
    return ctx, messages


def continue_to(root, state="admitted", *, asker="asker", successor="successor"):
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_results import write_task_result
    write_task_result(root, successor, status="running", chat_id=1)
    binding = {"predecessor_task_id": asker, "successor_task_id": successor, "action_nonce": "nonce"}
    admission = {"binding": binding, "binding_sha256": binding_sha(binding)}
    patch_document(root, asker, lambda row: {**row, "continued_by": {
        **admission, "state": state, "successor_task_id": successor}})
    patch_document(root, successor, lambda row: {**row, "continuation_admission": admission})
