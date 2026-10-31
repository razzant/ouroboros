"""The delivery builder of a plan ask (``ouroboros/tools/plan_packet.py``): what one
participant receives, in the stated order, and what it is never given."""
from __future__ import annotations

import pytest

from ouroboros.tools import plan_packet

REVISION = {"n": 2, "ts": "2026-10-15T02:00:00+00:00", "goal": "Ship the deck", "plan": "outline\nthen the story",
            "done_when": [{"id": "c1", "claim": "a five-slide story", "support": "read it"}], "affected_paths": [], "body": False}
PREVIOUS = {"n": 1, "ts": "2026-10-15T01:00:00+00:00", "goal": "Ship the deck", "plan": "outline",
            "done_when": [{"id": "c1", "claim": "five slides", "support": "count"}], "affected_paths": ["deck.md"], "body": False}


def test_stable_prompt_carries_the_delivered_checklist_and_contract_and_no_pack_for_a_non_body_plan():
    prompt = plan_packet.build_stable_prompt(checklist_section="## Plan Review Checklist\n\nROW ONE", body=False, enforcement="blocking")
    assert "ROW ONE" in prompt and "## Reply contract" in prompt and '"withdrawn"' in prompt
    assert "weighs an INTENTION" in prompt and "nobody votes" in prompt
    assert "BIBLE" not in prompt and "navigation map" not in prompt and "ARCHITECTURE" not in prompt
    missing = plan_packet.build_stable_prompt(checklist_section="", body=False, enforcement="advisory")
    assert "OMISSION NOTE: Plan Review Checklist section not supplied" in missing


def test_the_stable_prompt_scopes_holding_to_plan_revisions_and_tells_a_reading_how_to_answer():
    """One text for every ask of a task (the cached part): an objection holds only as a reply to a stored
    revision, and a reading answers in prose; the reading's own words ride after the plan marker."""
    prompt = plan_packet.build_stable_prompt(checklist_section="x", body=False, enforcement="blocking")
    assert "You are one independent participant of the council that weighs an INTENTION before the work starts" in prompt
    assert "Before any plan exists, Ouroboros may ask how you read the request." in prompt
    assert "an open objection of an owner-marked reviewer to a plan revision holds the author's final answer" in prompt
    assert ("Only replies to a stored plan revision update standing objections. A reading before any plan neither\n"
            "raises nor withdraws them: put that reading in the JSON object's prose field") in prompt
    assert "No plan yet" not in prompt and plan_packet.PLAN_REPLY_CONTRACT.startswith("Return ONE JSON object")


def test_a_body_plan_carries_the_pack_inline_or_as_mandatory_reads_and_refuses_without_it():
    inline = plan_packet.build_stable_prompt(checklist_section="x", body=True, enforcement="blocking",
                                             bible_text="BIBLE BODY", architecture_text="ARCH BODY", design_nav_map="## DESIGN map")
    assert "## BIBLE.md (constitution" in inline and "BIBLE BODY" in inline and "ARCH BODY" in inline and "## DESIGN map" in inline
    assert "has no repository tools" in inline
    retrieving = plan_packet.build_stable_prompt(checklist_section="x", body=True, enforcement="blocking",
                                                 bible_text="BIBLE BODY", architecture_text="ARCH BODY",
                                                 bible_locator="/repo/BIBLE.md", architecture_locator="/repo/docs/ARCHITECTURE.md",
                                                 governance_by_retrieval=True, design_nav_map="## DESIGN map")
    assert "MANDATORY FULL READS" in retrieving and "/repo/BIBLE.md" in retrieving and "BIBLE BODY" not in retrieving
    assert "Read relevant original DESIGN sections" in retrieving
    with pytest.raises(plan_packet.PlanPacketError):
        plan_packet.build_stable_prompt(checklist_section="x", body=True, enforcement="blocking", bible_text="", architecture_text="ARCH")
    with pytest.raises(plan_packet.PlanPacketError):
        plan_packet.build_stable_prompt(checklist_section="x", body=True, enforcement="blocking", bible_text="B", architecture_text=None)


def _content(**over):
    fields = dict(objective="Deliver the thing", owner_words="## OWNER WORDS\n\nmake it good", dialogue_section="## OWN ROOM DIALOGUE\n\nrows\n",
                  revision=REVISION, delta=None, cross_talk="", own_objections=[], message="First look.", evidence_section="(nothing attached)\n",
                  first_time=True, snapshot_section="## OWN ROOM SNAPSHOT (this ask)\n\nlocator `chat:1@aaa` · captured T1\n")
    fields.update(over)
    return plan_packet.build_participant_content(**fields)


def test_the_cache_stable_prefix_is_byte_identical_across_asks_while_the_plan_and_council_change():
    """The cache boundary must be real: the objective, the owner's words and the room's conversation are
    byte-identical between a first ask and a later one even though the revision, its delta, the
    council's talk, the participant's own objections and the room's snapshot facts (a new capture's
    locator and time) all differ (they do on every later ask)."""
    first = _content()
    later = _content(revision={**REVISION, "n": 2, "plan": "a different approach"}, first_time=False,
                     delta={"first": False, "from": 1, "to": 2, "plan_changed": True, "plan_diff": ["-old", "+new"]},
                     cross_talk="### Ask a1\nOuroboros asked: look\n- s2 answered:\n  fine\n",
                     own_objections=[{"id": "o1", "text": "the bar", "raised_in": "a1", "revision": 1}], message="Second look.",
                     snapshot_section="## OWN ROOM SNAPSHOT (this ask)\n\nlocator `chat:1@bbb` · captured T2\n")
    a, b = plan_packet.plan_user_stable_len(first), plan_packet.plan_user_stable_len(later)
    assert a > 0 and first[:a] == later[:b], "the cached prefix must not depend on the plan or the council's talk"
    assert "make it good" in first[:a] and "rows" in first[:a]
    assert first != later and "a different approach" in later[b:] and "a different approach" not in first
    assert "chat:1@aaa" in first[a:] and "chat:1@bbb" in later[b:] and "chat:1@" not in first[:a]


def test_first_time_content_is_whole_and_ordered():
    content = _content()
    order = ["## TASK OBJECTIVE", "## OWNER WORDS", "## OWN ROOM DIALOGUE", "## THE PLAN", "### Goal", "### done_when",
             "### affected_paths", "### Plan", "## OWN ROOM SNAPSHOT", "## THE COUNCIL SO FAR", "## YOUR OPEN OBJECTIONS",
             "## WHAT OUROBOROS ASKS YOU NOW", "## ATTACHED EVIDENCE"]
    positions = [content.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "You see this plan for the first time" in content and "a five-slide story" in content and "then the story" in content
    assert "every first answer is blind" in content and "You have no open objection" in content and "First look." in content
    assert plan_packet.plan_user_stable_len(content) == content.index("## THE PLAN")


def test_later_content_carries_the_delta_cross_talk_and_own_objections_first():
    from ouroboros.plan_record import revision_delta

    asks = [{"ask_id": "a1", "revision": 1, "ts": "t", "message": "look", "participants": [
        {"seat_id": "s1", "model": "m/a", "status": "answered", "reply": {"prose": "mine", "objections": [{"id": "o1", "text": "low bar"}], "withdrawn": []}},
        {"seat_id": "s2", "model": "m/b", "status": "answered", "reply": {"prose": "Peer says fine.", "objections": [], "withdrawn": [{"id": "z", "why": "w"}]}},
        {"seat_id": "s3", "model": "m/c", "status": "failed", "error": "transport died"},
        {"seat_id": "s4", "model": "m/d", "status": "not_dispatched", "not_dispatched_reason": "health_skip"},
    ]}]
    cross = plan_packet.render_cross_talk(asks, seat_id="s1")
    assert "### Ask a1" in cross and "Ouroboros asked: look" in cross
    assert "s2 (m/b) answered:" in cross and "Peer says fine." in cross and "withdrew z: w" in cross
    assert "s3 (m/c): failed — transport died" in cross and "s4 (m/d): not_dispatched — health_skip" in cross
    assert "mine" not in cross  # a participant's own replies are not repeated to it
    assert plan_packet.render_cross_talk([], seat_id="s1") == ""
    own = [{"id": "o1", "text": "low bar", "raised_in": "a1", "revision": 1, "not_adjudicated": True}]
    content = _content(delta=revision_delta(PREVIOUS, REVISION), cross_talk=cross, own_objections=own, first_time=False,
                       message="I raised the bar; is o1 answered?")
    assert "Changes since revision 1" in content and "done_when reworded: c1: «five slides» → «a five-slide story»" in content
    assert "affected path removed: deck.md" in content and "```diff" in content and "+then the story" in content
    assert "The current text follows whole." in content and "a five-slide story" in content
    assert "- o1 (raised in ask a1 about revision 1) (flagged: not adjudicated in your last reply): low bar" in content
    assert content.index("## THE COUNCIL SO FAR") < content.index("## YOUR OPEN OBJECTIONS") < content.index("I raised the bar")
    quiet = _content(delta={"first": False, "from": 1, "to": 2}, first_time=False)
    assert "nothing in the recorded fields changed" in quiet and "Nothing was said since your last answer." in quiet


def test_evidence_section_names_every_absence():
    manifest = {"declared": ["notes.md", "https://x"], "attached": [{"locator": "notes.md", "kind": "path", "sha256": "abc", "bytes": 5,
                                                                     "attached_bytes": 5, "text": "hello", "secrets_redacted": True}],
                "omissions": [{"locator": "https://x", "reason": "url_not_fetched"}]}
    rendered = plan_packet.render_evidence(manifest)
    assert "----- BEGIN notes.md -----\nhello\n----- END notes.md -----" in rendered and "secrets_redacted=true" in rendered
    assert "| https://x | url_not_fetched |" in rendered
    assert plan_packet.render_evidence({"declared": [], "attached": [], "omissions": []}) == "(nothing attached)\n"
    assert plan_packet.plan_user_stable_len("no marker here") == 0


def test_each_attachment_says_whose_words_it_carries_by_its_kind():
    manifest = {"declared": ["task:t-7::lines=1-3", "chat:1@abc", "notes.md"], "omissions": [], "attached": [
        {"locator": "task:t-7::lines=1-3", "kind": "task", "sha256": "a", "bytes": 3, "attached_bytes": 3, "text": "res"},
        {"locator": "chat:1@abc", "kind": "chat", "sha256": "b", "bytes": 3, "attached_bytes": 3, "text": "row"},
        {"locator": "notes.md", "kind": "path", "sha256": "c", "bytes": 3, "attached_bytes": 3, "text": "txt"}]}
    rendered = plan_packet.render_evidence(manifest)
    assert "Carries: the result of Ouroboros's task t-7 with host facts about it.\n----- BEGIN task:t-7::lines=1-3 -----" in rendered
    assert "Carries: room rows, each signed with its author.\n----- BEGIN chat:1@abc -----" in rendered
    assert "Carries: file content; its author is not attested by the host.\n----- BEGIN notes.md -----" in rendered
    assert "my human" not in rendered  # no attachment is labelled as the owner's words
    assert "Carries:" not in plan_packet.render_evidence({"declared": ["https://x"], "attached": [],
                                                          "omissions": [{"locator": "https://x", "reason": "url_not_fetched"}]})


def test_a_reading_carries_no_plan_and_keeps_the_cached_prefix_of_the_plan_asks():
    author = "drafted by task root-0, not spoken by my human"
    reading = _content(revision=None, objective_author=author)
    planned = _content(objective_author=author)
    a, b = plan_packet.plan_user_stable_len(reading), plan_packet.plan_user_stable_len(planned)
    assert a > 0 and reading[:a] == planned[:b] and f"The objective above: {author}." in reading[:a]
    assert reading[a:].startswith("## THE PLAN\n\nNo plan yet: Ouroboros asks how you read the request before it writes one.")
    for absent in ("### Goal", "### done_when", "Revision ", "on this plan"):
        assert absent not in reading, absent
    assert "No plan yet" not in planned and "You see this plan for the first time" in planned[b:]


def test_the_plan_surface_session_schema_is_the_reply_contract_and_can_carry_an_objection():
    """The retrieving-session route of a plan ask returns the SAME reply contract an API seat
    returns: prose plus typed objections and withdrawals — never a findings array or a
    verdict. The generic review surfaces keep their generic schema."""
    from ouroboros.plan_record import parse_reply
    from ouroboros.review_execution import review_session_output_schema

    schema = review_session_output_schema("plan_review")
    assert schema is plan_packet.PLAN_REPLY_SESSION_OUTPUT_SCHEMA
    props = schema["properties"]
    # Strict providers require every key: the optional lists are required-but-nullable, nested objects closed.
    assert set(props) == {"prose", "objections", "withdrawn"} and schema["required"] == ["prose", "objections", "withdrawn"]
    assert props["objections"]["type"] == ["array", "null"] and props["withdrawn"]["type"] == ["array", "null"]
    assert set(props["objections"]["items"]["properties"]) == {"id", "text"} and props["objections"]["items"]["additionalProperties"] is False
    assert set(props["withdrawn"]["items"]["properties"]) == {"id", "why"} and props["withdrawn"]["items"]["additionalProperties"] is False
    null_reply, null_error = parse_reply('{"prose": "Nothing to object.", "objections": null, "withdrawn": null}')
    assert null_error == "" and null_reply["objections"] == [] and null_reply["withdrawn"] == []
    assert "findings" not in props and "verdict" not in props and "class" not in str(schema)
    reply, error = parse_reply('{"prose": "No.", "objections": [{"id": "o1", "text": "the bar is too low"}], "withdrawn": []}')
    assert error == "" and reply["objections"] == [{"id": "o1", "text": "the bar is too low"}]
    assert review_session_output_schema("triad") is not schema


def test_the_shipped_checklist_speaks_the_reply_contract_and_rides_the_stable_prompt():
    """The live ``## Plan Review Checklist`` is placed whole in every participant's stable prompt,
    so it must speak the organ's own vocabulary — objections a raiser withdraws, prose for the
    rest, nobody votes — and none of the retired engine's (findings array, classes, breaks,
    GREEN, the subtraction row)."""
    from ouroboros.tools.review_checklist import load_checklist_section

    section = load_checklist_section("Plan Review Checklist")
    assert section.startswith("## Plan Review Checklist") and len(section) > 500
    lowered = section.lower()
    for word in ("objection", "withdraw", "prose", "nobody votes", "done_when", "affected_paths"):
        assert word in lowered, word
    for retired in ("no_findings", "need_evidence", "breaks", "json array", "| 6 | subtraction |", "aggregate: green",
                    "review_disposition", "acceptance_claims"):
        assert retired not in lowered, retired
    prompt = plan_packet.build_stable_prompt(checklist_section=section, body=False, enforcement="blocking")
    assert section in prompt and "## Reply contract" in prompt and '"withdrawn"' in prompt



def test_the_seat_line_follows_the_cache_stable_prefix_on_every_packet_seat(monkeypatch):
    """Each packet seat's send ends with its own seat AFTER the cache-stable prefix (which stays
    byte-identical across seats), and the recorded cache split is that same boundary."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import review_synthesis
    from ouroboros.tools.plan_dialogue import dialogue_slot_inputs
    from tests._review_substrate_shared import user_text

    packet = _content()

    def slot(name):
        return SimpleNamespace(slot_id=name, model="same/model", role_hint="plan reviewer", use_local=False,
                               session_profile=name, route=ReviewRouteKind.API_CHAT, retrieves=False, native_retrieval=False)

    monkeypatch.setattr(review_synthesis, "per_slot_input_token_limits", lambda *a, **k: {"a": 200000, "b": 200000})
    delivery = dialogue_slot_inputs([slot("a"), slot("b")], system_prompt="stable part", user_content=packet,
                                    session_task=packet, manifest={}, slot_messages={}, native_mandatory_chars=len(packet),
                                    task_id="task-1", output_contract=plan_packet.PLAN_REPLY_CONTRACT)
    sent = {sid: user_text(delivery["slot_messages"][sid][-1]["content"]) for sid in ("a", "b")}
    boundary = plan_packet.plan_user_stable_len(packet)
    assert boundary > 0 and sent["a"][:boundary] == sent["b"][:boundary] == packet[:boundary]
    assert sent["a"].endswith("\n## YOUR PANEL SEAT\n\n`a`\n") and sent["b"].endswith("\n## YOUR PANEL SEAT\n\n`b`\n")
    assert sent["a"].index("## YOUR PANEL SEAT") > sent["a"].index("## THE PLAN")
    blocks = delivery["slot_messages"]["a"][-1]["content"]
    assert isinstance(blocks, list) and blocks[0]["text"] == packet[:boundary] and blocks[-1]["text"].endswith("`a`\n")
    assert delivery["request_policy"]["output_contract"] == plan_packet.PLAN_REPLY_CONTRACT
