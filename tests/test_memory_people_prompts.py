"""People are a subject of Ouroboros's memory, stated in its own prompts.

Nothing in the runtime prompts used to say that understanding a person is
durable knowledge: SYSTEM.md scoped the knowledge base to operational facts,
recipes and gotchas, the background checklist asked only for a gotcha, a recipe
or a pattern, and reflection nominated "self-knowledge". The observable result
on a live install was that everything ever written about the human had been
dictated word for word, never noticed and never inferred.

These are vocabulary pins for the memory contract — which memory each kind of
understanding has, and where it lives. They pin the words that carry each
contract, not a model's behaviour; dropping one of these contracts is the defect.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body of one `## `/`### ` section, whitespace-normalized."""
    marker = re.search(rf"^#{{2,3}} {re.escape(heading)}\s*$", text, re.MULTILINE)
    assert marker, f"section {heading!r} is gone"
    rest = text[marker.end():]
    end = re.search(r"^#{2,3} ", rest, re.MULTILINE)
    return " ".join((rest[: end.start()] if end else rest).split())


def _claims(section: str) -> list:
    """Sentence-ish units, lowercased: a contract may move between sentences."""
    return [part.strip() for part in re.split(r"(?<=[.;:])\s+", section.lower()) if part.strip()]


def test_system_memory_section_makes_people_a_subject_of_knowledge():
    memory = _section(_read("prompts/SYSTEM.md"), "Memory")
    claims = _claims(memory)

    # Knowledge is understanding of every kind, people explicitly included.
    assert any("people" in c and ("knowledge" in c or "understanding" in c) for c in claims)
    # Attribution: told me / observed / inferred are kept apart.
    assert any("infer" in c and ("told me" in c or "observed" in c) for c in claims)
    # A correction is situated evidence, not a standing rule.
    assert any("correction" in c and "standing rule" in c for c in claims)
    # Repeating an interpretation does not make it more true.
    assert any("interpretation" in c and ("one interpretation" in c or "not new" in c) for c in claims)
    # The authored summary is the resident face of a note.
    assert any("summary" in c and "index" in c for c in claims)
    # The overview is the shared orientation loaded into every context.
    assert any("overview" in c and "context" in c for c in claims)
    # Understanding of people is global, whatever room the work happens in.
    assert any("people" in c and "global" in c for c in claims)
    # What I learn about a person is written in the same turn, not left to a later summary.
    assert any("same turn" in c and "person" in c for c in claims)
    # The old scoping, which is what made the knowledge base operational-only.
    assert "Durable operational facts, recipes, and gotchas go to knowledge topics" not in memory


def test_system_memory_section_keeps_its_earlier_obligations():
    """The section gained clauses; it must not have lost the ones that were load
    bearing — provenance, the reserved index name, and dated external knowledge."""
    memory = _section(_read("prompts/SYSTEM.md"), "Memory")

    assert "I distinguish known, stale, missing, and inferred" in memory
    assert "`knowledge_list` shows the topics" in memory
    assert "`knowledge/index-full.md` is a reserved internal name" in memory
    assert "I `knowledge_read` its topic first" in memory
    assert "stale unless recently verified" in memory


def test_system_states_the_chronicle_contract_and_each_writer_in_one_home():
    """The chronicle's writers in the mind's own prompt: the mind seals, marks and reads
    by address; the overview is the mind's; a Light helper drafts one page or part only
    while the wakes are off; an offer to fold the old retelling is settled by a global
    mark. A delegated child's start and its drafts live in Delegation, the
    self-contained assignment in Memory. The two claims the chronicle made false
    (nothing project-related hidden, read-only helpers writing nothing) are gone."""
    system = _read("prompts/SYSTEM.md")
    memory, delegation, tools = (_section(system, name) for name in ("Memory", "Delegation", "Tools"))
    claims = _claims(memory)

    for tool in ("`chronicle_write`", "`memory_mark`", "`memory_read`"):
        assert tool in memory, tool
    assert any("overview" in c and "my own words" in c and "helpers do not write it" in c for c in claims)
    assert any("light helper" in c and "one page or part" in c and "wakes are off" in c for c in claims)
    assert any("fold" in c and "ordinary background task" in c for c in claims)
    assert any("global mark" in c and "`memory_mark`" in c for c in claims)
    # One home each: the child's start and drafts in Delegation, the assignment rule in Memory.
    assert "never my whole dialogue history" in delegation
    assert "drafts in its own name, which the integrating mind accepts or rejects" in delegation
    assert "drafts in its own name" not in memory
    assert "must stand on its own" in memory and "must stand on its own" not in delegation
    # The memory tools and the chronicle's own path are named where file writes are ruled out.
    assert "`chronicle_write`, `memory_mark`" in tools and "`memory/chronicle/*`" in tools
    assert "chronicle page and part drafts in their own name" in tools
    # Everything the host guarantees a child at start, its memory marks included (memory_view.ROLE_DEFAULTS).
    assert "its room's page, the memory marks of that room and the global ones, my whole assignment" in delegation
    assert "which I accept or reject" not in delegation  # a child that schedules a grandchild does not decide
    flat = " ".join(system.split())
    assert "nothing project-related is hidden" not in flat
    assert "Read-only helpers do not write" not in flat


def test_system_learns_a_person_through_the_relationship_not_only_on_demand():
    human = _section(_read("prompts/SYSTEM.md"), "Environment and My Human")
    claims = _claims(human)

    # My human is a relationship, and the neutral form before acquaintance stays.
    assert any("my human" in c and "relationship" in c for c in claims)
    assert "I do not know their name or personal profile by default" in human
    # Conversation, shared work and — by its own judgment — public work.
    assert any("conversation" in c or "work we do together" in c for c in claims)
    assert any("public work" in c for c in claims)
    # A plain question is allowed, the name first (owner decision 10A).
    assert any(("ask" in c or "question" in c) and "name" in c for c in claims)
    # What is understood lives in knowledge, under the person's name.
    assert any("name" in c and ("knowledge" in c or "understand" in c) for c in claims)
    # "owner" is authority, never the person (owner decision 2A).
    assert any("owner" in c and "authority" in c for c in claims)
    # The dead end this replaced: ask, learn it, and never revisit it.
    assert "if I need a name or preference, I ask and then learn it in memory" not in human


def test_memory_vocabulary_is_not_dressed_up_as_tool_names():
    """Backticked `summary`/`overview` would read as tool identifiers both to the
    model and to the prompt identifier audit in tests/test_docs_sync.py."""
    for rel in ("prompts/SYSTEM.md", "prompts/CONSCIOUSNESS.md"):
        text = _read(rel)
        for word in ("summary", "overview"):
            assert f"`{word}`" not in text, f"{rel} backticks {word}"


def test_the_people_memory_contract_has_exactly_one_home():
    """The wake-up message is an ordinary Main turn's USER text, so it runs on the
    system prompt that already states this contract. It used to restate it — which
    made the same obligation editable in two places and drifted the moment one
    changed. The contract is pinned above, in its single home; what is pinned here
    is that the wake does not carry a second copy of it (P7 one home per fact)."""
    memory = _section(_read("prompts/SYSTEM.md"), "Memory")
    claims = _claims(memory)

    # Still exactly the obligations the wake used to duplicate.
    assert any("revise" in c and ("in place" in c or "same turn" in c) for c in claims)
    assert any("one interpretation" in c for c in claims)

    wake = " ".join(_read("prompts/CONSCIOUSNESS.md").split()).lower()
    for duplicated in ("people you talk with", "rather than minting a new one",
                       "knowledge_read", "knowledge_write", "contradictions"):
        assert duplicated not in wake, f"the wake restates {duplicated!r}"


def test_wake_template_addresses_its_human_not_a_user():
    consciousness = _read("prompts/CONSCIOUSNESS.md")

    assert "Speak to your human" in consciousness
    assert "your human" in consciousness and "the user" not in consciousness


def test_reflection_nominates_learning_about_people_as_well_as_itself():
    from ouroboros.reflection import _REFLECTION_PROMPT_TAIL

    tail = " ".join(_REFLECTION_PROMPT_TAIL.split())
    lowered = tail.lower()

    assert "people i worked with" in lowered
    assert "self-knowledge" not in lowered
    # The typed contract the parser depends on is untouched.
    assert "MEMORY_ACTIONS_JSON: [...]" in tail
    assert '"scratchpad_append", "knowledge_write", "identity_update_candidate"' in tail
