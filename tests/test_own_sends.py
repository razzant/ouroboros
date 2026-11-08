"""What I post through a tool acting as an account is kept by its provider identity, from the send's own result.

The MCP manager's real call path records it (``MCPManager._call_tool_result``) for a tool its server declares a
write; a server-declared read, a tool without that declaration (a read naming one message would otherwise turn my
human's post into mine), a result naming several messages (a listing, not a receipt) and an error keep nothing;
the record holds only the newest ``KEEP`` identities. Whether a kept identity ever reads as my human's words:
``test_place_owner_author``.
"""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros import mcp_client, own_sends
from ouroboros.tools.tool_result import ToolResult

WRITE = {"annotations": {"readOnlyHint": False}}
HEADER = "MsgID,UserID,UserName,RealName,Channel,ThreadTs,Text,Time,Permalink,Reactions,BotName,FileCount,AttachmentIDs,HasMedia"


def _csv(*rows):
    return "\n".join([HEADER, *(f'{ts},U03,anton,Anton,{channel},,"{text}",2026-11-07T10:00:00Z,,,,0,,false'
                                for channel, ts, text in rows)]) + "\n"


def _manager(results):
    manager = mcp_client.MCPManager()

    async def list_tools(cfg, timeout):
        return [{"name": name, "description": name, "input_schema": {"type": "object"},
                 "metadata": {"annotations": {"readOnlyHint": read}}}
                for name, read in (("conversations_add_message", False), ("conversations_history", True))] + [
            {"name": "conversations_replies", "description": "thread", "input_schema": {"type": "object"}}]

    async def call(cfg, name, args, timeout):
        return results[name]

    manager._async_list_tools, manager._async_call_tool = list_tools, call
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 60, "MCP_SERVERS": [
        {"id": "slack", "enabled": True, "transport": "stdio", "command": "python"}]})
    assert manager.refresh_server("slack")["ok"]
    return manager


def test_a_declared_send_keeps_the_one_message_its_result_names_and_a_read_a_listing_or_an_error_keeps_none(tmp_path):
    posted = _csv(("C0B1VF", "1793991732.265319", "Ship it on Friday, please."))
    thread = _csv(("C0B1VF", "1793617022.105639", "My human's own post, read through their account"))
    manager = _manager({"conversations_add_message": ToolResult(status="ok", code="OK", text=posted),
                        "conversations_history": ToolResult(status="ok", code="OK", text=_csv(
                            ("C0B1VF", "1793991999.000001", "Read, not sent"))),
                        "conversations_replies": ToolResult(status="ok", code="OK", text=thread)})
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={})
    manager._call_tool_result("mcp_slack__conversations_history", {"channel_id": "C0B1VF"}, ctx)
    assert own_sends.sent_identities(tmp_path) == frozenset(), "a server-declared read posts nothing"
    manager._call_tool_result("mcp_slack__conversations_replies", {"channel_id": "C0B1VF"}, ctx)
    assert own_sends.sent_identities(tmp_path) == frozenset(), "an undeclared tool naming one message posted nothing"
    manager._call_tool_result("mcp_slack__conversations_add_message", {"channel_id": "C0B1VF", "text": "Ship it"}, ctx)
    assert own_sends.sent_identities(tmp_path) == {("C0B1VF", "1793991732.265319")}
    listing = _csv(("C1", "1.1", "one"), ("C1", "1.2", "two"))
    assert not own_sends.record_result(tmp_path, WRITE, listing, via="mcp:slack/x"), "several messages: no one send"
    assert not own_sends.record_result(tmp_path, WRITE, "Message posted.", via="mcp:slack/x"), "no identity named"
    manager = _manager({"conversations_add_message": ToolResult(status="error", code="MCP_ERROR", text=_csv(
        ("C9", "9.9", "failed")))})
    manager._call_tool_result("mcp_slack__conversations_add_message", {"channel_id": "C9"}, ctx)
    assert ("C9", "9.9") not in own_sends.sent_identities(tmp_path), "an error result is no send"
    api = '{"ok": true, "channel": "C2", "ts": "5.6", "message": {"ts": "5.6", "text": "hi"}}'
    assert own_sends.record_result(tmp_path, WRITE, f"done\n\nstructuredContent: {api}", via="mcp:other/post")
    assert ("C2", "5.6") in own_sends.sent_identities(tmp_path), "a JSON receipt names its one message too"


def test_the_record_keeps_only_the_newest_sends(tmp_path, monkeypatch):
    monkeypatch.setattr(own_sends, "KEEP", 2)
    for index in range(3):
        assert own_sends.record_result(tmp_path, WRITE, _csv(("C1", f"1.{index}", "words")), via="mcp:slack/post")
    assert own_sends.sent_identities(tmp_path) == {("C1", "1.1"), ("C1", "1.2")}, "the oldest left with the next write"
    own_sends.own_sends_path(tmp_path).write_text("{broken", encoding="utf-8")
    assert own_sends.sent_identities(tmp_path) == frozenset(), "unreadable: no recorded send, never an error"
    assert own_sends.record_result(tmp_path, WRITE, _csv(("C1", "2.0", "words")), via="mcp:slack/post")
    assert own_sends.sent_identities(tmp_path) == {("C1", "2.0")}, "the next record starts it afresh"
