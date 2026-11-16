"""Exact target groups (``touches``) of a reviewed connector, through the real loader, registry and tool batch.

A synthetic mail-like connector (no provider, no network) publishes ``send``, ``draft`` and ``browse`` with fixed
groups (its recipients are the send's audience and exact, casefolded address constraints; folders and drafts keep the
scope matcher), and real places in the host store decide: a draft's folder note and pause, a new send to a bare
recipient in its account, a group reply beside a Paused private conversation and a Paused bare address, an accountless
pause, a replaced draft's own pairs, the emitted conversation's author beside a Paused cross pair, the PluginAPI minimum
and bad shapes, and a call bound to its publication across a replacement. Only the Safety Supervisor's model call is
scripted.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import replace

import pytest

from ouroboros import extension_loader, places
from ouroboros.presence_bindings import conversation_key
from tests._extension_dispatch_harness import _restore_server_pid  # noqa: F401  (autouse fixture applies on import)
from tests._extension_dispatch_harness import as_server, batch, brief_evidence, brief_place, settings_and_repo, supervisor
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401  (autouse fixture applies on import)
from tests._extension_loader_shared import _prepare_extension

ACCOUNT = "imap:fixture"
GROUP, SOURCE, DM = "<group@example.invalid>", "<source@example.invalid>", "<dm@example.invalid>"
OLD, EMITTED = "<old@example.invalid>", "<emitted@example.invalid>"
PEER, OTHER = "peer@example.invalid", "other@example.invalid"
_PLUGIN = (
    "def register(api):\n"
    "    for name in ('send', 'draft', 'browse'):\n"
    "        api.register_tool(name, lambda ctx, _name=name, **args: f'{_name} ran', description=name, schema={})\n"
)
_EMITTED_WORDS = "      - {conversation: emitted_conversation_id, thread: emitted_conversation_id, effect: message}\n"
_MANIFEST = (
    'plugin_api: "2.5"\n'
    "tools:\n"
    "  - name: send\n"
    "    effect: message\n"
    "    target: [account_id, to, cc]\n"
    "    conversation: conversation_id\n"
    "    thread: thread_id\n"
    "    text: body\n"
    "    touches:\n"
    "      - {conversation: to, effect: write, match: exact_casefold}\n"
    "      - {conversation: cc, effect: write, match: exact_casefold}\n"
    + _EMITTED_WORDS +
    "  - name: draft\n"
    "    effect: write\n"
    "    target: conversation_id\n"
    "    conversation: conversation_id\n"
    "    thread: thread_id\n"
    "    touches:\n"
    "      - {conversation: to, effect: write, match: exact_casefold}\n"
    "      - {conversation: folder, effect: write}\n"
    "      - {conversation: folder_names, effect: write}\n"
    "      - {conversation: draft_conversation_id, thread: draft_conversation_id, effect: write}\n"
    "      - {conversation: emitted_conversation_id, thread: emitted_conversation_id, effect: write}\n"
    "  - name: browse\n"
    "    effect: read\n"
    "    target: conversation_id\n"
    "    conversation: conversation_id\n"
    "    touches:\n"
    "      - {conversation: source_id, thread: source_id, effect: read}\n"
)


def _surface(tool: str) -> str:
    return extension_loader.extension_surface_name("mailer", tool)


def _mailer(tmp_path, monkeypatch, manifest=_MANIFEST, plugin=_PLUGIN):
    """Load the reviewed connector for real; return what its loader answered, the registry and the data root."""
    from ouroboros.tools.registry import ToolRegistry

    as_server()
    loaded, repo_root, drive_root = _prepare_extension(tmp_path, "mailer", plugin, permissions=["tool"],
                                                       extra_frontmatter=manifest)
    settings_and_repo(monkeypatch, repo_root)
    loaded_now = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    return loaded_now, registry, drive_root, (loaded, repo_root)


def _place(root, conversation, label, *, state="active", note="", place_id="", thread="", account=ACCOUNT):
    place = places.connect_place(root, connector="mailer", provider="mail", account_id=account,
                                 conversation_id=conversation, label=label, state=state, place_id=place_id,
                                 first={"note": note} if note else None)
    if thread:  # a thread-scoped place (a migrated binding's shape); Connect itself names no thread
        place = places._write(root, place.place_id, lambda old: replace(old, thread_id=thread))
    return place


def _run(registry, root, tool, args, call_id="c1"):
    results, _starts = batch(registry, root, {"id": call_id, "function": {
        "name": _surface(tool), "arguments": json.dumps(args)}}, task_id="mail-task")
    return results[0]


def _state(root, place, state):
    places.edit_place(root, place.place_id, "set_state", author="owner", state=state)


def _authoring(root, key, author="other-author"):
    from ouroboros.conversation_lease import PresenceTurnGate

    lease = PresenceTurnGate(state_root=root / "state").acquire(key)
    lease.bind_author(author)
    return lease


def test_a_drafts_folder_note_reaches_the_check_and_its_pause_holds_the_save(tmp_path, monkeypatch):
    """ordinary_draft_folder_note: an Active folder's note is in the Supervisor's bundle before the draft is saved,
    a Paused folder refuses the save, and a draft takes no author lease where it is saved."""
    prompts = supervisor(monkeypatch)
    loaded_now, registry, root, _payload = _mailer(tmp_path, monkeypatch)
    assert loaded_now == {"mailer": None}, loaded_now
    drafts = _place(root, "Drafts", "Mail · Drafts", note="Keep drafts short.")
    args = {"account_id": ACCOUNT, "to": [PEER], "body": "Ordinary draft", "folder": "Drafts"}
    lease = _authoring(root, conversation_key("mail", ACCOUNT, "Drafts"))
    try:
        result = _run(registry, root, "draft", args)
    finally:
        lease.release()
    assert result["is_error"] is False and result["result"].startswith("draft ran"), result
    facts = brief_place(prompts[0])
    [row] = [row for row in facts["places"] if row["place_id"] == drafts.place_id]
    assert (row["effects"], row["via"]) == (["write"], ["touches"])
    assert any(note["text"] == "Keep drafts short." for note in brief_evidence(prompts[0])["standing_delegation"]["place_notes"])
    _state(root, drafts, "paused")
    result = _run(registry, root, "draft", args, "c2")
    assert result["is_error"] is True and "Mail · Drafts is Paused" in result["result"], result
    assert len(prompts) == 1, "a refused save reached the Supervisor"


@pytest.mark.parametrize("account_wide", [True, False], ids=["account-wide", "narrow-only"])
def test_a_new_send_meets_its_bare_recipients_place_and_no_unrelated_pause(tmp_path, monkeypatch, account_wide):
    """new_send_bare_recipient: a recipient address is an exact write constraint: its place's state and notes apply,
    spelled in another case too, it takes no author lease and leaves no expected speech. Naming no conversation, the
    send acts in its account: an account-wide Active place covers it and a Paused private conversation elsewhere
    refuses nothing; with only narrow places that account is unknown, so the pause holds it before Safety."""
    prompts = supervisor(monkeypatch)
    _loaded, registry, root, _payload = _mailer(tmp_path, monkeypatch)
    account = _place(root, "", "Mail · Account") if account_wide else None
    peer = _place(root, "Peer@Example.invalid", "Mail · Peer", note="Peer prefers short notes.")
    _place(root, DM, "Mail · Private", state="paused")
    args = {"account_id": ACCOUNT, "to": [PEER], "body": "Example", "request_id": "new-1"}
    lease = _authoring(root, conversation_key("mail", ACCOUNT, PEER))
    try:
        result = _run(registry, root, "send", args)
    finally:
        lease.release()
    if not account_wide:
        assert result["is_error"] is True and (
            "cannot be attributed while Mail · Private is Silent or Paused" in result["result"]), result
        assert prompts == []
        return
    assert result["is_error"] is False and result["result"].startswith("send ran"), result
    facts = brief_place(prompts[0])
    rows = {row["place_id"]: row for row in facts["places"]}
    assert (rows[peer.place_id]["effects"], rows[peer.place_id]["via"], rows[peer.place_id]["conversation_keys"]) == (
        ["write"], ["touches"], [])
    assert (rows[account.place_id]["via"], rows[account.place_id]["conversation_keys"]) == (["account"], [])
    assert facts["touches"] == [{"effect": "write", "argument": "to", "conversation": PEER, "account": ACCOUNT,
                                 "match": "exact_casefold", "place_id": peer.place_id}]
    assert any(note["text"] == "Peer prefers short notes." for note in brief_evidence(prompts[0])["standing_delegation"]["place_notes"])
    assert not getattr(registry._ctx, "_place_message_calls", None), "a recipient's write became expected speech"
    _state(root, peer, "paused")
    result = _run(registry, root, "send", args, "c2")
    assert result["is_error"] is True and "Mail · Peer is Paused" in result["result"], result
    assert len(prompts) == 1


@pytest.mark.parametrize("recipients", [{"to": [PEER], "cc": [OTHER]}, {"to": [OTHER], "cc": [PEER]}],
                         ids=["reply-all", "extra-recipient"])
@pytest.mark.parametrize("account_wide", [True, False], ids=["account-wide", "narrow-only"])
def test_a_group_reply_passes_a_paused_private_thread_but_not_a_paused_address(tmp_path, monkeypatch, recipients,
                                                                             account_wide):
    """group_reply_two_kinds_of_pause (owner's choice: a bare address restricts every delivery to it, a conversation
    only itself). The recipients are the reply's audience and exact constraints, so with or without an account-wide
    Active place a Paused private conversation of a participant leaves the group reply allowed (a recipient no place
    names is a known absence), and a Paused bare address among its recipients refuses it, in To or in Cc. The v2
    candidate held the narrow-only reply as unknown; r5's exact constraints let it through."""
    prompts = supervisor(monkeypatch)
    _loaded, registry, root, _payload = _mailer(tmp_path, monkeypatch)
    if account_wide:
        _place(root, "", "Mail · Account")
    _place(root, GROUP, "Mail · Group")
    _place(root, DM, "Mail · Private", state="paused")
    peer = _place(root, PEER, "Mail · Peer")
    args = {"account_id": ACCOUNT, **recipients, "reply_to_message_id": SOURCE, "conversation_id": GROUP,
            "thread_id": GROUP, "body": "Group answer", "request_id": "reply-1"}
    result = _run(registry, root, "send", args)
    assert result["is_error"] is False and result["result"].startswith("send ran"), result
    group_key = conversation_key("mail", ACCOUNT, GROUP, GROUP)
    assert registry._ctx._place_message_calls[-1]["conversation_keys"] == [group_key]
    _state(root, peer, "paused")
    result = _run(registry, root, "send", args, "c2")
    assert result["is_error"] is True and "Mail · Peer is Paused" in result["result"], result
    assert len(prompts) == 1


def test_an_accountless_paused_address_holds_a_reply_inside_an_active_account(tmp_path, monkeypatch):
    """global_exact_pause: a Paused place naming the address for every account, stored in another case, holds a reply
    sent from an Active account into an Active thread, before Safety; once Active, the same reply passes."""
    prompts = supervisor(monkeypatch)
    _loaded, registry, root, _payload = _mailer(tmp_path, monkeypatch)
    _place(root, "", "Mail · Account")
    _place(root, GROUP, "Mail · Group")
    everywhere = _place(root, "PEER@example.invalid", "Mail · Peer everywhere", state="paused", account="")
    args = {"account_id": ACCOUNT, "conversation_id": GROUP, "thread_id": GROUP, "to": [OTHER], "cc": [PEER],
            "body": "Group answer"}
    result = _run(registry, root, "send", args)
    assert result["is_error"] is True and "Mail · Peer everywhere is Paused" in result["result"], result
    assert prompts == []
    _state(root, everywhere, "active")
    result = _run(registry, root, "send", args, "c2")
    assert result["is_error"] is False and result["result"].startswith("send ran"), result


def test_a_replaced_draft_keeps_each_address_with_its_own_thread(tmp_path, monkeypatch):
    """replaced_draft_own_pair: the old draft, the folders and the emitted conversation each meet their own place with
    their own thread; a Paused pair the call never names (the source with the emitted thread) is not a destination.
    The Drafts folder no narrower place names is the account-wide Active place's; the recipient is an exact constraint
    no place names."""
    prompts = supervisor(monkeypatch)
    _loaded, registry, root, _payload = _mailer(tmp_path, monkeypatch)
    _place(root, "", "Mail · Account")
    _place(root, SOURCE, "Mail · Source")
    old = _place(root, OLD, "Mail · Old draft")
    label = _place(root, "Label_fixture", "Mail · Label", state="paused")
    _place(root, EMITTED, "Mail · Emitted")
    _place(root, SOURCE, "Mail · Cross pair", state="paused", place_id="cross-pair", thread=EMITTED)
    args = {"account_id": ACCOUNT, "to": [PEER], "body": "Replacement", "draft_id": "draft-fixture",
            "conversation_id": SOURCE, "thread_id": SOURCE, "draft_conversation_id": OLD,
            "folder_names": ["Drafts"], "emitted_conversation_id": EMITTED}
    result = _run(registry, root, "draft", args)
    assert result["is_error"] is False, result
    touched = {(row["argument"], row["conversation"], row.get("thread", "")) for row in brief_place(prompts[0])["touches"]}
    assert {("draft_conversation_id", OLD, OLD), ("emitted_conversation_id", EMITTED, EMITTED),
            ("folder_names", "Drafts", ""), ("to", PEER, "")} == touched
    assert "cross-pair" not in {row["place_id"] for row in brief_place(prompts[0])["places"]}
    result = _run(registry, root, "draft", {**args, "folder_names": ["Drafts", "Label_fixture"]}, "c2")
    assert result["is_error"] is True and "Mail · Label is Paused" in result["result"], result
    _state(root, label, "active")
    _state(root, old, "paused")
    result = _run(registry, root, "draft", args, "c3")
    assert result["is_error"] is True and "Mail · Old draft is Paused" in result["result"], result
    assert len(prompts) == 1


def test_the_emitted_conversation_has_its_own_author_and_one_call_is_one_send(tmp_path, monkeypatch):
    """emitted_author_and_cross_pair: another turn authoring the emitted conversation refuses the send before Safety;
    once it is free, the one call is noted once with both conversations' keys, never as two messages, and a Paused pair
    of the source with the emitted thread refuses nothing. The bare recipient is the account-wide Active place's."""
    prompts = supervisor(monkeypatch)
    _loaded, registry, root, _payload = _mailer(tmp_path, monkeypatch)
    _place(root, "", "Mail · Account")
    _place(root, SOURCE, "Mail · Source")
    _place(root, EMITTED, "Mail · Emitted")
    _place(root, SOURCE, "Mail · Cross pair", state="paused", place_id="cross-pair", thread=EMITTED)
    args = {"account_id": ACCOUNT, "to": [PEER], "conversation_id": SOURCE, "thread_id": SOURCE,
            "emitted_conversation_id": EMITTED, "body": "Answer", "request_id": "emitted-1"}
    emitted_key, source_key = (conversation_key("mail", ACCOUNT, EMITTED, EMITTED),
                               conversation_key("mail", ACCOUNT, SOURCE, SOURCE))
    lease = _authoring(root, emitted_key)
    try:
        result = _run(registry, root, "send", args)
    finally:
        lease.release()
    assert result["is_error"] is True and "tell_task(task_id='other-author'" in result["result"], result
    assert "Mail · Emitted" in result["result"] and prompts == []
    result = _run(registry, root, "send", args, "c2")
    assert result["is_error"] is False, result
    noted = [row for row in registry._ctx._place_message_calls if row["text"] == "Answer"]
    assert len(noted) == 1 and noted[0]["conversation_keys"] == [source_key, emitted_key], noted


def test_an_older_host_refuses_a_2_5_connector_before_import_read_tools_included(tmp_path, monkeypatch):
    """minimum_api: a host serving 2.4 refuses the payload at negotiation, before its module is imported, so no tool
    of it (its read included) runs with its groups ignored. The 2.4 surface is 2.5's: no method was added."""
    from ouroboros.contracts import plugin_api as papi

    recorded = papi.PLUGIN_API_SURFACE_FINGERPRINTS["2.4"]
    monkeypatch.setattr(papi, "PLUGIN_API_VERSION", "2.4")
    assert papi.plugin_api_surface_fingerprint() == recorded
    monkeypatch.setattr(papi, "PLUGIN_API_SURFACE_FINGERPRINTS", {"2.4": recorded})
    loaded_now, _registry, _root, _payload = _mailer(tmp_path, monkeypatch,
                                                     plugin="raise RuntimeError('plugin module imported')\n")
    refusal = loaded_now["mailer"]
    assert "negotiation refused" in refusal and "minimum minor 5" in refusal and "plugin module imported" not in refusal
    assert all(extension_loader.get_tool(_surface(tool)) is None for tool in ("send", "draft", "browse"))


def test_groups_under_an_older_declared_minimum_refuse_every_call_whole(tmp_path, monkeypatch):
    """A payload declaring 2.4 still loads on this host, but its groups are not trusted in part: every call, the read
    included, is refused with the repair, before Safety."""
    prompts = supervisor(monkeypatch)
    loaded_now, registry, root, _payload = _mailer(tmp_path, monkeypatch,
                                                   manifest=_MANIFEST.replace('plugin_api: "2.5"', 'plugin_api: "2.4"'))
    assert loaded_now == {"mailer": None}, loaded_now
    for tool, args in (("send", {"account_id": ACCOUNT, "to": [PEER], "body": "hi"}),
                       ("browse", {"conversation_id": SOURCE})):
        result = _run(registry, root, tool, args, f"c-{tool}")
        assert result["is_error"] is True and "Nothing ran" in result["result"], result
        assert 'plugin_api at "2.5"' in result["result"] and "binds 2.4" in result["result"], result
    assert prompts == []


@pytest.mark.parametrize("change,refusal", [
    ({"account_id": [ACCOUNT]}, "'account_id' holds a list"),  # one account per call, never inferred from a list
    ({"to": {"address": PEER}}, "'to' holds a dict"),
    ({"to": [PEER, ["nested"]]}, "'to' holds a list"),
    ({"cc": None}, ""), ({"cc": []}, ""), ({"cc": ""}, ""),  # absent, null or empty reach nothing
])
def test_a_group_argument_of_another_shape_refuses_the_call_and_an_empty_one_reaches_nothing(
        tmp_path, monkeypatch, change, refusal):
    supervisor(monkeypatch)
    _loaded, registry, root, _payload = _mailer(tmp_path, monkeypatch)
    _place(root, "", "Mail · Account")  # the send's account is a place's too, so only the argument's shape decides
    _place(root, DM, "Mail · Private", state="paused")
    _place(root, PEER, "Mail · Peer")
    result = _run(registry, root, "send", {"account_id": ACCOUNT, "to": [PEER], "body": "hi", **change})
    if refusal:
        assert result["is_error"] is True and refusal in result["result"] and "Nothing ran" in result["result"], result
    else:
        assert result["is_error"] is False and result["result"].startswith("send ran"), result


def test_a_call_keeps_the_groups_of_the_publication_it_was_bound_to(tmp_path, monkeypatch):
    """A reviewed replacement published between the call's timeout choice and its dispatch lends it nothing: the place
    guard, and the destination facts written after it ran, read the bound publication's groups."""
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state

    supervisor(monkeypatch)
    _loaded, registry, root, (loaded, repo_root) = _mailer(tmp_path, monkeypatch)
    _place(root, SOURCE, "Mail · Source")
    _place(root, EMITTED, "Mail · Emitted", note="New thread: say who you are.")
    real_timeout = registry.get_timeout

    def bind_then_replace(name):
        timeout = real_timeout(name)
        manifest = loaded.skill_dir / "SKILL.md"
        manifest.write_text(manifest.read_text().replace(_EMITTED_WORDS, ""))
        updated = find_skill(root, "mailer", repo_path=str(repo_root))
        save_review_state(root, "mailer", SkillReviewState(status="pass", content_hash=updated.content_hash))
        assert extension_loader.reload_all(root, lambda: {}, repo_path=str(repo_root)) == {"mailer": None}
        return timeout

    monkeypatch.setattr(registry, "get_timeout", bind_then_replace)
    args = {"account_id": ACCOUNT, "conversation_id": SOURCE, "thread_id": SOURCE,
            "emitted_conversation_id": EMITTED, "body": "Answer"}
    result = _run(registry, root, "send", args)
    assert result["is_error"] is False, result
    assert ("[DESTINATION] Mail · Emitted: your human's notes for this place: \"New thread: say who you are.\""
            in result["result"]), result


_DOORS = (
    'plugin_api: "2.5"\n'
    "tools:\n"
    "  - name: reply\n"
    "    effect: message\n"
    "    target: conversation_id\n"
    "    conversation: conversation_id\n"
    "    thread: thread_id\n"
    "    text: body\n"
    "    touches:\n"
    "      - {conversation: to, effect: write}\n"
    "  - name: note\n"
    "    effect: write\n"
    "    target: conversation_id\n"
    "    conversation: conversation_id\n"
    "    thread: thread_id\n"
    "  - name: react\n"
    "    effect: gesture\n"
    "    target: [conversation_id, message_id]\n"
    "    conversation: conversation_id\n"
    "    thread: thread_id\n"
    "    message: message_id\n"
)
_DOOR_PLUGIN = _PLUGIN.replace("('send', 'draft', 'browse')", "('reply', 'note', 'react')")


def test_a_later_publication_never_reinterprets_an_executed_call(tmp_path, monkeypatch):
    """What reached the turn's own conversation is judged by the declaration each call ran with, kept in its result
    facts: a reply and a reaction that ran as doors still ride with the finish after a reviewed replacement declares
    them writes, and a write never rides because the replacement declares it a door. Read by today's declarations
    instead (a row recorded without them), the replacement would reinterpret all three."""
    from ouroboros.place_reply import _ran_as, conversation_gesture_tools, riding_sends
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state
    from ouroboros.tool_effects import RETAINED_DECLARATION

    supervisor(monkeypatch)
    _loaded, registry, root, (loaded, repo_root) = _mailer(tmp_path, monkeypatch, manifest=_DOORS, plugin=_DOOR_PLUGIN)
    group = _place(root, GROUP, "Mail · Group")
    here = {"account_id": ACCOUNT, "conversation_id": GROUP, "thread_id": GROUP}
    calls = [("reply", {**here, "to": [PEER], "body": "Group answer"}), ("note", {**here, "label": "seen"}),
             ("react", {**here, "message_id": "<m1@example.invalid>", "name": "eyes"})]
    results, _starts = batch(registry, root, *({"id": f"c{n}", "function": {
        "name": _surface(tool), "arguments": json.dumps(args)}} for n, (tool, args) in enumerate(calls)), task_id="mail-task")
    assert [result["is_error"] for result in results] == [False, False, False], results
    # The rows a finish reads, as the loop records them (``process_tool_results``): plain facts, no handler.
    rows = [{"tool": result["fn_name"], "args": json.loads(json.dumps(result["args_for_log"])),
             "is_error": result["is_error"], **result["result_meta"]} for result in results]
    assert [json.loads(json.dumps(row[RETAINED_DECLARATION]))["tool"]["effect"] for row in rows] == [
        "message", "write", "gesture"]
    manifest = loaded.skill_dir / "SKILL.md"
    manifest.write_text(manifest.read_text().replace("name: reply\n    effect: message", "name: reply\n    effect: write")
                        .replace("name: note\n    effect: write", "name: note\n    effect: message")
                        .replace("name: react\n    effect: gesture", "name: react\n    effect: write"))
    updated = find_skill(root, "mailer", repo_path=str(repo_root))
    save_review_state(root, "mailer", SkillReviewState(status="pass", content_hash=updated.content_hash))
    assert extension_loader.reload_all(root, lambda: {}, repo_path=str(repo_root)) == {"mailer": None}
    registry._ctx.task_metadata = {**(registry._ctx.task_metadata or {}), "presence": {
        "transport_skill": "mailer", "place_id": group.place_id,
        "event": {"conversation_key": conversation_key("mail", ACCOUNT, GROUP, GROUP)}}}
    assert [row["tool"] for row in riding_sends(registry._ctx, rows)] == [_surface("reply"), _surface("react")]
    today_gestures = dict(conversation_gesture_tools("mailer"))
    assert _ran_as(rows[2], "mailer", "gesture", today_gestures).message == "message_id"
    unkept = [{key: value for key, value in row.items() if key != RETAINED_DECLARATION} for row in rows]
    assert [row["tool"] for row in riding_sends(registry._ctx, unkept)] == [_surface("note")]  # the negative control
    assert _ran_as(unkept[2], "mailer", "gesture", today_gestures) is None


def test_one_broad_place_keeps_both_accounts_in_the_safety_brief(tmp_path, monkeypatch):
    from ouroboros import dialogue_store

    prompts = supervisor(monkeypatch)
    manifest = _MANIFEST.replace(_EMITTED_WORDS, _EMITTED_WORDS.replace(
        "effect: message", "account: emitted_account, effect: message"))
    _loaded, registry, root, _payload = _mailer(tmp_path, monkeypatch, manifest=manifest)
    place = _place(root, "", "Mail · All accounts", account="", note="Shared account guidance.")
    places.edit_place(root, place.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="owner-B", account_id="B")
    identities = {ACCOUNT: {"sends_as": {"id": "bot-A", "label": "Bot mailbox", "kind": "bot"}},
                  "B": {"sends_as": {"id": "owner-B", "label": "Owner mailbox", "kind": "user"}}}
    for account, identity in identities.items():
        key = conversation_key("mail", account, SOURCE)
        dialogue_store.append(root, "chat", {
            "ts": "2026-10-26T12:00:00Z", "type": "place_arrival", "direction": "in",
            "chat_id": places.conversation_chat_id(key), "place_id": place.place_id,
            "source": "presence:mail", "text": "Captured message",
            "transport": {"account_id": account, "identity": identity}})
    args = {"account_id": ACCOUNT, "conversation_id": SOURCE, "thread_id": SOURCE,
            "emitted_account": "B", "emitted_conversation_id": EMITTED, "body": "Answer"}
    result = _run(registry, root, "send", args)
    assert result["is_error"] is False, result
    rows = brief_place(prompts[0])["places"]
    assert len(rows) == 2
    assert {row["account_id"]: (row["sends_as"], row["sends_as_owner"]) for row in rows} == {
        account: (identity["sends_as"], account == "B") for account, identity in identities.items()}
    assert {row["account_id"]: row["conversation_keys"] for row in rows} == {
        ACCOUNT: [conversation_key("mail", ACCOUNT, SOURCE, SOURCE)],
        "B": [conversation_key("mail", "B", EMITTED, EMITTED)]}
    notes = brief_evidence(prompts[0])["standing_delegation"]["place_notes"]
    assert [note["text"] for note in notes] == ["Shared account guidance."]
    assert result["result"].count("[DESTINATION] Mail · All accounts:") == 1
    lease = _authoring(root, conversation_key("mail", "B", EMITTED, EMITTED))
    try:
        held = _run(registry, root, "send", args, "held")
    finally:
        lease.release()
    assert held["is_error"] is True and "other-author" in held["result"], held
    assert len(prompts) == 1


@pytest.mark.parametrize("grouped", [False, True], ids=["legacy", "optional-groups"])
def test_primary_threads_reach_the_registry_with_their_notes_and_holds(tmp_path, monkeypatch, grouped):
    prompts = supervisor(monkeypatch)
    manifest = ('plugin_api: "2.5"\ntools:\n  - name: send\n    effect: message\n'
                '    target: [account_id, conversation_id, thread_id]\n'
                '    conversation: conversation_id\n    thread: thread_id\n    text: body\n')
    if grouped:
        manifest += '    touches:\n      - {conversation: copies, effect: read}\n'
    _loaded, registry, root, _payload = _mailer(tmp_path, monkeypatch, manifest=manifest)
    first = _place(root, GROUP, "Mail · First", place_id="first", thread="t1", note="First thread guidance.")
    second = _place(root, GROUP, "Mail · Second", place_id="second", thread="t2", note="Second thread guidance.")
    args = {"account_id": ACCOUNT, "conversation_id": GROUP, "thread_id": ["t1", "t2"], "body": "Both threads"}
    keys = [conversation_key("mail", ACCOUNT, GROUP, thread) for thread in ("t1", "t2")]
    result = _run(registry, root, "send", args)
    assert result["is_error"] is False, result
    rows = brief_place(prompts[0])["places"]
    assert {row["place_id"] for row in rows} == {first.place_id, second.place_id}
    if grouped:
        assert {row["thread_id"]: row["conversation_keys"] for row in rows} == {"t1": [keys[0]], "t2": [keys[1]]}
    notes = brief_evidence(prompts[0])["standing_delegation"]["place_notes"]
    assert {note["text"] for note in notes} == {"First thread guidance.", "Second thread guidance."}
    assert result["result"].count("[DESTINATION]") == 2
    assert registry._ctx._place_message_calls[-1]["conversation_keys"] == keys
    for i, place in enumerate((first, second)):
        _state(root, place, "paused")
        held = _run(registry, root, "send", args, f"pause-{i}")
        assert held["is_error"] is True and f"{place.label} is Paused" in held["result"], held
        _state(root, place, "active")
        lease = _authoring(root, keys[i])
        try:
            held = _run(registry, root, "send", args, f"author-{i}")
        finally:
            lease.release()
        assert held["is_error"] is True and "other-author" in held["result"], held
    assert len(prompts) == 1
