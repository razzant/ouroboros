"""A chat id is a VALUE, not a boolean — keep it that way (sprint CLI display).

``chat_id`` 0 is the hidden partition: a real, addressable destination that no
browser surface reads. Every producer that wrote ``if chat_id:`` therefore did
two wrong things at once — it dropped a panel-bound notice AND re-routed hidden
work to the owner's main chat — which is how a CLI run's whole dialogue became
invisible while its children surfaced in Main as a nameless card.

This is a SOURCE lint over two packages, not a runtime gate: it constrains how
new code is written, never how the agent behaves (BIBLE P5). Both directions of
the habit are caught — the negative form that skips the partition and the
positive form that guards a send the same wrong way — plus the record shape that
stores 0 as absence. Every surviving occurrence carries a written reason here, so
the allowlist IS the residual disclosure instead of a second document.

A live NOTICE is the one honest exception: it needs a reader, and the hidden
partition has none, so several sites deliberately skip it. That is a different
statement from "0 means no chat", and the reasons below say which is which.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("supervisor", "ouroboros/gateway")

# A row's own address recorded as "absent" when it is really the hidden partition.
POISONED_RECORD = re.compile(r'(?:"chat_id":|\bchat_id=)\s*[^,\n]*\bor\s+None')
# A route decided by truthiness, in BOTH directions: `if not chat_id` skips the
# hidden partition, and `if chat_id:` guards a send the same wrong way.
# ``owner_chat_id`` is deliberately exempt — a 0/absent OWNER chat means "no
# owner chat is configured", never the panel, so testing it for truth is honest.
_CHAT_NAME = r"(?!owner_chat_id\b)(?:[A-Za-z_]*_)?chat_id"
TRUTHY_ROUTE = re.compile(
    rf"^\s*if (?:not {_CHAT_NAME}\b|(?:[^:\n]*\band )?{_CHAT_NAME}\s*:)"
    rf"|^\s*if not [^:\n]*\bor not {_CHAT_NAME}\b"
)

# (repo-relative path, exact stripped line) -> (occurrences, why it stays)
ALLOWED = {
    ("supervisor/terminal_delivery.py", "if not chat_id:"): (
        3,
        "lineage_chat_id() answers with the task's OWN chat: a project-homed run "
        "gets its room, and 0 means the run was never homed. Delivering an "
        "unhomed answer into the hidden partition would add rows to the "
        "benchmark-parsed chat log for a surface with no reader.",
    ),
    ("supervisor/workers.py", "if not chat_id:"): (
        2,
        "Promote/steer lane (evt always arrives from a real chat) and the "
        "auto-resume gate, where owner_chat_id 0 means 'no owner chat "
        "configured' rather than the panel. Disclosed residual: workers.py sits "
        "~55 bytes under its 200k module ceiling, so it takes no edit this sprint.",
    ),
    ("supervisor/workers.py", "if chat_id:"): (
        1,
        "Same promote lane, same ceiling. Disclosed residual.",
    ),
    ("supervisor/terminal_delivery.py", "if not tid or not core_text or not chat_id:"): (
        1,
        "lineage_chat_id() again: 0 means the run was never homed, and pushing "
        "an unhomed answer into a partition with no reader would add rows to the "
        "chat log a benchmark parses for its final answer.",
    ),
    ("supervisor/steering.py", "if notify and chat_id:"): (
        1,
        "A steer REFUSAL is a live notice to the person who asked, and a steer "
        "event only ever arrives from a real chat. Same rule as the scheduled "
        "toast: a live notice needs a reader, and the hidden partition has none.",
    ),
    ("supervisor/steering.py", "if not client_message_id and chat_id:"): (
        1,
        "Same refusal path, same rule.",
    ),
    ("supervisor/steering.py", "if chat_id:"): (
        1,
        "Same refusal path, same rule.",
    ),
}


def _sources():
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            yield path.relative_to(REPO).as_posix(), path.read_text(encoding="utf-8")


def _hits(pattern):
    found = {}
    for rel, text in _sources():
        for line in text.splitlines():
            if pattern.search(line):
                found.setdefault((rel, line.strip()), 0)
                found[(rel, line.strip())] += 1
    return found


def test_no_new_truthiness_route_for_a_chat_id():
    hits = _hits(TRUTHY_ROUTE)
    unexplained = {key: count for key, count in hits.items() if key not in ALLOWED}
    assert not unexplained, (
        "A chat id is a value: 0 is the hidden partition, absence is None. Route "
        "with supervisor.message_bus.notification_chat_route (where does this "
        "notice go) or coerce_chat_identity (what is this row's address). If the "
        "site genuinely must stay, add it to ALLOWED with a written reason.\n"
        f"{sorted(unexplained)}"
    )
    for key, (expected, _why) in ALLOWED.items():
        assert hits.get(key) == expected, (
            f"allowlisted site count changed for {key}: expected {expected}, saw "
            f"{hits.get(key)} — re-read the reason and update it deliberately."
        )


def test_no_record_stores_the_hidden_partition_as_absent():
    hits = _hits(POISONED_RECORD)
    assert not hits, (
        "Writing `chat_id or None` records the hidden partition as 'no chat', so "
        "the row can never be replayed or re-addressed. Store the value.\n"
        f"{sorted(hits)}"
    )


def test_the_allowlist_has_no_stale_entries():
    hits = _hits(TRUTHY_ROUTE)
    stale = [key for key in ALLOWED if key not in hits]
    assert not stale, f"allowlist entries no longer match any source line: {stale}"


def test_a_records_own_address_is_never_relabelled_by_notice_routing():
    """Two questions, two normalizers — and a record must use the right one.

    `notification_chat_route` suppresses synthetic A2A ids because no human
    stream may show them. A terminal RECORD that borrowed that helper would file
    an A2A row under the hidden partition, losing the audience it truly had.
    """
    from supervisor.message_bus import notification_chat_route, row_chat_identity

    assert notification_chat_route(-1001, 5) == 5, "routing skips a synthetic id"
    assert row_chat_identity(-1001, 5) == -1001, "a record keeps its own address"
    assert row_chat_identity(None, "", 0, 7) == 0, "an explicit 0 is present, not absent"
    assert row_chat_identity(None, "", default=1) == 1

    events = (REPO / "supervisor/events.py").read_text(encoding="utf-8")
    for block in events.split('"chat_id": ')[1:]:
        head = block[:40]
        if head.startswith("notification_chat_route"):
            assert "task_done" not in block[:400], (
                "a task_done record must address itself with row_chat_identity"
            )
