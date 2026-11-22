"""Shared forced-rail fixture: an acting actor's round context on the Claudexor engine double.

Moved from ``tests/test_forced_model_binding.py`` so test modules do not import each other
(Development 02, "Documentation contract"). The fixture asks for the ``setup`` fixture of the module
that imports it (``tests/test_llm_claudexor.py::setup``).
"""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ouroboros.llm import LLMClient
from ouroboros.loop_round_limits import _RoundLimitContext
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.tools.registry import ToolRegistry
from tests._llm_claudexor_fixtures import MODEL


@pytest.fixture
def acting(setup, monkeypatch):
    root, gateway, client = setup
    from ouroboros.task_results import write_task_result
    write_task_result(root, "task-one", "running", root_task_id="task-one", _attempt=1)
    gateway.results *= 4
    gateway.dispatch *= 4
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "main-only", "fallback": ["fallback-only"]}))
    # Auto is the mode that asks the route's catalog (Inline sends without asking),
    # so these sends show which account binding that read used.
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    catalogs = []

    def catalog(source, profile=None, *, requested_model=None):
        catalogs.append((source, profile, requested_model))
        return {"source": source, "credentialProfileId": profile, "models": [
            {"id": requested_model, "inputModalities": ["text"] if profile == "main-only" else ["text", "image"]}]}

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    messages = [{"role": "system", "content": "Own SYSTEM"}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    tools = ToolRegistry(repo_dir=root.parent, drive_root=root)
    tools._ctx.task_metadata = {"configured_subagent": {"selected_subagent_id": "visual-actor",
        "route": {"kind": "api_model", "target_id": MODEL, "credential_profile_id": "actor-only"}}}
    tools._ctx.active_model = MODEL
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ctx = _RoundLimitContext(messages=deepcopy(messages), llm=client, active_model=MODEL,
        active_effort="high", max_retries=1, drive_logs=logs, task_id="task-one", round_idx=2,
        event_queue=None, accumulated_usage={}, task_type="task", active_use_local=False,
        max_rounds=2, drive_root=root, tools=tools, tool_schemas=[])
    return SimpleNamespace(root=root, gateway=gateway, client=client, messages=messages,
                           tools=tools, ctx=ctx, catalogs=catalogs)


# One row of my human's carried to a work, shared by the forced-rail and restored-read tests (moved from
# ``tests/test_native_fallback_binding_acceptance.py`` for the same reason as the fixture above).
CARRIED_WORDS = "Use the second fixture as well."


def carried_quote(root):
    """One row of my human's, retained in Main's chat, as the quote the host carries to a member of its tree."""
    from ouroboros import dialogue_store
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.owner_relay import owner_quote

    row = {"chat_id": 1, "direction": "in", "ts": "2026-11-20T10:00:00Z", "text": CARRIED_WORDS,
           "client_message_id": "carried-words", "attachments": []}
    dialogue_store.append(root, "chat", row)
    return owner_quote(root, row, address=format_address(row_address(row)))


def carry_to_work(root, task_id, quote, *, went_to=""):
    """The host's carriage of that row to this work (``owner_relay.carry_to_members``): shown once, typed as my
    human's words, in no owner corpus, so the owner hash a send bound does not move. ``went_to`` names another work
    the row was addressed to: then it is context here, shown typed all the same but addressed to no one in this work."""
    from ouroboros.owner_relay import write_relay

    address = quote["source"]["address"]
    assert write_relay(root, task_id, relay_id=f"{'carry-context' if went_to else 'carry'}:{address}", note="",
                       quotes=[quote], chat_id=1, teller_task_id="coordinator", canonical_root=root,
                       host_carried={"went_to": went_to})[1]
