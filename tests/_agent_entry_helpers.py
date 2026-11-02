"""Real task entry and turn ownership around a test-supplied task body."""

import queue
import threading
from types import SimpleNamespace

from ouroboros.agent import OuroborosAgent


def task_entry_agent(drive_root, body, *, emit_log=None):
    """Skip provider/tools/memory construction, retaining admission and preparation.

    These tests observe scopes inside a synthetic body. The real body releases
    its turn in ``finally``; the replacement must do that too, including on a
    reused worker or a body failure. No entry or control method is stubbed.
    """
    agent = object.__new__(OuroborosAgent)
    agent.env = SimpleNamespace(drive_root=drive_root)
    agent._event_queue = None
    agent._emit_live_log = emit_log or (lambda *_args, **_kwargs: None)
    agent._owner_message_admission_lock = threading.Lock()
    agent._incoming_messages = queue.Queue()
    agent._busy = False
    agent._accepting_owner_messages = False
    agent._current_task_id = None
    agent._current_chat_id = None
    agent._current_task_metadata = {}
    agent._current_task_text = ""
    agent._last_activity_ts = None

    def scoped(task):
        assert agent._busy and agent._current_task_id == task["id"]
        try:
            return body(task)
        finally:
            agent._release_turn()

    agent._handle_task_scoped = scoped
    return agent
