"""A change does not make what every mind carries larger, and a book chapter stays within one read (the official line's rule).

What rides every request is measured at this change's tip and at its event base: BIBLE.md and the prompt
files together as their UTF-8 bytes (``reference_books.RULED_TEXTS``), and the Body Map as the bytes each
revision's OWN code renders from its own books (the tip by this code, the base by the base's code:
``reference_books.revision_body_map_sizes``), so growth of the renderer's own text is growth. The rule is
pairwise: there is no stored number and no reserve, so text a change adds is paid in the same text in the same
change (DEVELOPMENT "Paying down a size cap"). A base with no renderer had no Map: that change has no earlier Map
to compare with, and none is invented. A text there but unmeasurable, at the tip or at the base, is a failure,
never a zero and never "absent".

A book chapter is read on demand, so its limit is physical and needs no base: one default ``read_file``
(``reference_books.chapters_over_one_read``). The large standalone documents are under no rule here.

The opt-in ``size_ratchet`` lane is excluded from default local runs. Manual/tag CI
has no per-change base, so the pairwise comparison skips there. Local qualification supplies
``OURO_SIZE_RATCHET_BASE_REF``; unset ``OURO_BOOK_GROWTH_MODE`` then means ``block``.
``scripts/book_growth_mode.py`` retains event-specific block/warn/approved semantics
for explicit callers; an owner's ``book-growth`` label can approve an exception.
Every local information surface reports the balance without blocking (BIBLE P3 c5).
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest

from ouroboros.reference_books import (
    BODY_MAP_TEXT, BOOK_ENTRYPOINTS, RULED_TEXTS, UnmeasuredText, body_map_sizes, chapters_over_one_read,
    revision_body_map_sizes, ruled_text_sizes, text_name,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_REF_ENV = "OURO_SIZE_RATCHET_BASE_REF"
MODE_ENV = "OURO_BOOK_GROWTH_MODE"


def _git(repo: pathlib.Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True)


def measures_one_change(environ) -> bool:
    """A pull request or a push to ``ouroboros`` is one change; a push to ``main`` or
    ``ouroboros-stable`` spans a whole release, which the pairwise rule cannot attribute."""
    return environ.get("GITHUB_EVENT_NAME") != "push" or environ.get("GITHUB_REF") == "refs/heads/ouroboros"


def growth_base(repo: pathlib.Path, ref: str | None) -> str:
    """The commit this change is measured from; ``""`` when the rule does not apply.

    A run without a resolvable event base skips: a local or manual run has none, a tag push
    carries all zeros, and HEAD's parent is not where a multi-commit change began.
    """
    ref = (ref or "").strip()
    if not ref:
        return ""
    resolved = _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    return resolved.stdout.decode("ascii").strip() if resolved.returncode == 0 else ""


def ruled_text_bytes(repo: pathlib.Path, text_id: str, ref: str = "") -> int | None:
    """The ruled text's UTF-8 bytes in the checkout, or at ``ref``; ``None`` when absent there, ``UnmeasuredText``
    when there but unreadable."""
    sizes = ruled_text_sizes(repo, text_id, ref)
    return None if sizes is None else sum(sizes.values())


def body_map_bytes(repo: pathlib.Path, ref: str = "") -> int | None:
    """The rendered Body Map's UTF-8 bytes: in the checkout by this code, at ``ref`` by that revision's own code.
    ``None`` where no Map renders there (no Architecture book, or a revision without a renderer); ``UnmeasuredText``
    where it is there but cannot be measured."""
    sizes = revision_body_map_sizes(repo, ref) if ref else body_map_sizes(repo, BODY_MAP_TEXT)
    return None if sizes is None else sum(sizes.values())


def _measured(measure, ref: str) -> int | None | UnmeasuredText:
    try:
        return measure(ref)
    except UnmeasuredText as exc:  # there but unreadable: unknown, never absent
        return exc


def book_growth_verdict(book_id: str, base, tip, mode: str) -> tuple[bool, str]:
    """``(passes, message)`` for one ruled text; the message is empty when it did not grow. ``base`` and ``tip`` are
    bytes, ``None`` (absent there) or the ``UnmeasuredText`` that stopped the measure. A tip that could not be
    measured, and a base there but unreadable, fail in every mode: an unknown size is never a size that fits, and
    an unreadable base is never a base without the text."""
    name = f"the {text_name(book_id)}" if book_id in (*BOOK_ENTRYPOINTS, BODY_MAP_TEXT) else text_name(book_id)
    for side, value in (("tip", tip), ("base", base)):
        if isinstance(value, UnmeasuredText) or (side == "tip" and value is None):
            why = f" ({value})" if isinstance(value, UnmeasuredText) else ""
            return False, f"{name} could not be measured at this change's {side}{why}, so its growth is unknown."
    if base is None or tip <= base:
        return True, ""
    if mode == "approved":
        return True, f"{name} grows by {tip - base} bytes; the repository owner approved it (book-growth label)."
    if mode == "warn":
        return True, (f"{name} grows by {tip - base} bytes; advisory for outside contributors and forks. "
                      "Official maintainers will make room at integration.")
    return False, (f"{name} grew by {tip - base} bytes ({base} -> {tip}); a change ends BIBLE.md, the prompts "
                   "and the Body Map no larger than at its base, so pay in the same text as DEVELOPMENT "
                   "\"Paying down a size cap\" says "
                   "(the owner's book-growth label on the PR is the only exception).")


def _assert_no_unapproved_growth(capsys, measured) -> None:
    """``measured``: ``(id, annotated file, measure(ref))`` per text; growth fails unless approved or advisory."""
    if not measures_one_change(os.environ):
        pytest.skip("a push to a release branch spans many changes; the rule measures one")
    base = growth_base(REPO, os.environ.get(BASE_REF_ENV))
    if not base:
        pytest.skip(f"{BASE_REF_ENV} names no base commit this clone holds, so this change cannot be measured")
    mode = (os.environ.get(MODE_ENV) or "block").strip()
    faults: list[str] = []
    for text_id, path, measure in measured:
        passes, message = book_growth_verdict(text_id, _measured(measure, base), _measured(measure, ""), mode)
        if message and passes:  # an annotation in the job log; a captured print would never reach it
            with capsys.disabled():
                print(f"::{'notice' if mode == 'approved' else 'warning'} file={path}::{message}")
        elif message:
            faults.append(message)
    assert not faults, f"Text growth since {base[:12]}:\n" + "\n".join(faults)


@pytest.mark.size_ratchet
def test_a_change_does_not_grow_the_body_map(capsys):
    _assert_no_unapproved_growth(capsys, [
        (BODY_MAP_TEXT, BOOK_ENTRYPOINTS["architecture"], lambda ref: body_map_bytes(REPO, ref))])


@pytest.mark.size_ratchet
def test_every_book_chapter_fits_one_default_read():
    """Physical, so measured at the tip alone and in every run of the lane; an unreadable book raises."""
    over = {book_id: chapters_over_one_read(REPO, book_id) for book_id in BOOK_ENTRYPOINTS}
    assert not any(over.values()), (
        f"chapters past one default read_file (path, lines, characters): {over}; split each by subject with its "
        "reasons kept (DEVELOPMENT \"Paying down a size cap\")")


@pytest.mark.size_ratchet
def test_a_change_does_not_grow_bible_or_the_prompts(capsys):
    _assert_no_unapproved_growth(capsys, [
        (text_id, path, lambda ref, text_id=text_id: ruled_text_bytes(REPO, text_id, ref))
        for text_id, path in RULED_TEXTS.items()])


# ---------------------------------------------------------------- the decision, unmarked

@pytest.mark.parametrize("base,tip,mode,passes,said", [
    (1000, 1001, "", False, "grew by 1 bytes"),  # an unknown mode is never a pass
    (1000, 1480, "warn", True, "maintainers will make room"),
    (1000, 1480, "approved", True, "owner approved"),
    (1000, 900, "block", True, ""),
    (1000, 1000, "block", True, ""),
    (None, 5000, "block", True, ""),  # the change that creates the Map has no earlier Map to grow from
    (1000, None, "block", False, "could not be measured"),  # an unmeasured tip is unknown, never zero
    (1000, None, "approved", False, "could not be measured"),
    (UnmeasuredText("unreadable"), 5000, "approved", False, "at this change's base (unreadable)"),  # never "absent"
    (1000, UnmeasuredText("denied"), "approved", False, "at this change's tip (denied)"),
])
def test_book_growth_verdict(base, tip, mode, passes, said):
    verdict = book_growth_verdict(BODY_MAP_TEXT, base, tip, mode)
    assert verdict[0] is passes
    assert (said in verdict[1]) if said else verdict[1] == ""


def _book_tree(root, chapter="# One\n\nWhat this part gives me.\n", atlas=True):
    entry = ("# Architecture\n\nThe map of my body.\n\n## Chapters\n\n- [One](architecture/01-one.md)\n"
             + ("\n## Reference Atlas\n\n- [DESIGN.md](DESIGN.md) — the interface.\n" if atlas else ""))
    handbook = "# Development\n\nHow a change is made.\n\n## Chapters\n\n- [Rules](development/01-rules.md)\n"
    for path, text in {BOOK_ENTRYPOINTS["architecture"]: entry, "docs/architecture/01-one.md": chapter,
                       BOOK_ENTRYPOINTS["development"]: handbook, "docs/development/01-rules.md": "# Rules\n\nThe rules.\n"}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8", newline="\n")
    return entry


def _commit(root, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()


def test_the_body_map_is_measured_as_rendered_in_three_states(tmp_path, monkeypatch):
    """Absent (no Architecture book): no Map, a fact. Measured: what the renderer makes of the sources, with or
    without a Reference Atlas, and without a handbook the tree does not hold. There but unreadable (a denied
    handbook, a listed chapter that is gone, a revision whose objects cannot be read): ``UnmeasuredText``, never a
    smaller Map and never "no earlier Map"."""
    _git(tmp_path, "init", "-q")
    assert body_map_sizes(tmp_path) is None  # no Architecture book: no Map
    _book_tree(tmp_path, atlas=False)
    unmapped = body_map_bytes(tmp_path)  # no Atlas is still a Map the runtime sends
    _book_tree(tmp_path)
    created = body_map_bytes(tmp_path)
    assert unmapped and created > unmapped
    _commit(tmp_path, "the Map")
    assert body_map_sizes(tmp_path, ref="HEAD") == {BODY_MAP_TEXT: created}
    # A chapter's body is not in the Map: an edit below its opening moves no resident byte; its opening does.
    chapter = tmp_path / "docs/architecture/01-one.md"
    chapter.write_text(chapter.read_text(encoding="utf-8") + "\n## Detail\n\nMechanism.\n", encoding="utf-8", newline="\n")
    assert body_map_bytes(tmp_path) == created
    _book_tree(tmp_path, chapter="# One\n\nWhat this part gives me, and more.\n")
    grown = body_map_bytes(tmp_path)
    assert grown == created + len(", and more")
    assert book_growth_verdict(BODY_MAP_TEXT, created, grown, "block")[0] is False
    handbook = tmp_path / BOOK_ENTRYPOINTS["development"]
    real_read = pathlib.Path.read_bytes
    with monkeypatch.context() as patched:  # the handbook is there but denied: unknown, never a Map without its list
        patched.setattr(pathlib.Path, "read_bytes", lambda self: (_ for _ in ()).throw(PermissionError(str(self)))
                        if self == handbook else real_read(self))
        with pytest.raises(UnmeasuredText, match="DEVELOPMENT.md"):
            body_map_bytes(tmp_path)
        assert not book_growth_verdict(BODY_MAP_TEXT, created, _measured(lambda ref: body_map_bytes(tmp_path, ref), ""),
                                       "approved")[0]
    (tmp_path / "docs/development/01-rules.md").unlink()  # a listed chapter that is gone: the book is broken
    with pytest.raises(UnmeasuredText):
        body_map_bytes(tmp_path)
    for path in (handbook, tmp_path / "docs/development"):
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    without_handbook = body_map_bytes(tmp_path)  # no handbook in this tree: a fact, its list is not there
    assert without_handbook and without_handbook < grown
    # A revision that resolves but whose objects cannot be read is unknown, never a revision without the Map.
    from ouroboros import reference_books

    native = reference_books._git_bytes
    for failing in ("cat-file", "ls-tree"):
        with monkeypatch.context() as patched:
            patched.setattr(reference_books, "_git_bytes",
                            lambda root, *args, failing=failing, **kw: None if args[0] == failing else native(root, *args, **kw))
            with pytest.raises(UnmeasuredText):
                body_map_sizes(tmp_path, ref="HEAD")
    with monkeypatch.context() as patched:  # BIBLE.md's revision cannot even be listed: unknown, never absent
        patched.setattr(reference_books, "_git_bytes",
                        lambda root, *args, **kw: None if args[0] == "ls-tree" else native(root, *args, **kw))
        with pytest.raises(UnmeasuredText):
            ruled_text_sizes(tmp_path, "constitution", "HEAD")
    assert ruled_text_sizes(tmp_path, "constitution", "HEAD") is None  # listed, and BIBLE.md is not there: absent


# The modules the Body Map renderer imports, copied into a scratch repository's revisions.
_RENDERER = ("reference_books", "markdown_source", "repo_remotes", "update_channels")


def _renderer_tree(root, reader_suffix: str = "") -> None:
    for name in _RENDERER:
        source = (REPO / "ouroboros" / f"{name}.py").read_text(encoding="utf-8")
        if name == "reference_books" and reader_suffix:
            source = source.replace("BODY_MAP_READER = '", f"BODY_MAP_READER = {reader_suffix!r} + '", 1)
        (root / "ouroboros").mkdir(exist_ok=True)
        (root / "ouroboros" / f"{name}.py").write_text(source, encoding="utf-8", newline="\n")


def test_the_base_map_is_rendered_by_the_base_revisions_own_code(tmp_path, monkeypatch, capsys):
    """A change to the renderer that grows every request trips the rule: the base's Map is the one the base's own
    code rendered. A base without a renderer had no Map (nothing compared); a base whose renderer is there but fails
    is unknown and fails the lane."""
    from ouroboros import reference_books

    _git(tmp_path, "init", "-q")
    _book_tree(tmp_path)
    (tmp_path / "ouroboros").mkdir()
    (tmp_path / "ouroboros/reference_books.py").write_text('"""Books, before the Map."""\n', encoding="utf-8")
    before_map = _commit(tmp_path, "books without a renderer")
    _renderer_tree(tmp_path)
    rendered = _commit(tmp_path, "the renderer")
    _renderer_tree(tmp_path, "X" * 512)
    grown_renderer = _commit(tmp_path, "a renderer that adds 512 bytes")
    (tmp_path / "ouroboros/reference_books.py").write_text("def body_map(:\n", encoding="utf-8")
    broken = _commit(tmp_path, "a renderer that cannot run")
    _renderer_tree(tmp_path)
    _commit(tmp_path, "the renderer again")
    here = body_map_bytes(tmp_path)
    assert body_map_bytes(tmp_path, rendered) == here  # the same code and sources render the same Map
    assert body_map_bytes(tmp_path, grown_renderer) == here + 512  # that revision's OWN renderer, not this one
    assert body_map_bytes(tmp_path, before_map) is None  # no renderer there: no earlier Map
    with pytest.raises(UnmeasuredText, match="renderer failed"):
        body_map_bytes(tmp_path, broken)

    consumer = sys.modules[__name__]
    monkeypatch.setattr(consumer, "REPO", tmp_path)
    monkeypatch.setenv(MODE_ENV, "block")
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setenv(BASE_REF_ENV, rendered)
    test_a_change_does_not_grow_the_body_map(capsys)  # the real tree's renderer and books: +0
    native = reference_books.body_map

    def grown(*args, **kwargs):  # this change's renderer adds 512 resident bytes to every request
        view, parts = native(*args, **kwargs)
        return replace(view, text=view.text + "X" * 512), parts

    monkeypatch.setattr(reference_books, "body_map", grown)
    with pytest.raises(AssertionError, match="grew by 512 bytes"):
        test_a_change_does_not_grow_the_body_map(capsys)
    monkeypatch.setenv(BASE_REF_ENV, before_map)
    test_a_change_does_not_grow_the_body_map(capsys)  # the change that creates the Map has no earlier Map
    monkeypatch.setenv(BASE_REF_ENV, broken)
    with pytest.raises(AssertionError, match="could not be measured at this change's base"):
        test_a_change_does_not_grow_the_body_map(capsys)


def test_a_chapter_past_one_read_is_named_and_an_unreadable_book_is_an_error(tmp_path):
    _book_tree(tmp_path)
    assert chapters_over_one_read(tmp_path, "architecture") == ()
    long = "# One\n\nWhat this part gives me.\n" + "line\n" * 2000
    _book_tree(tmp_path, chapter=long)
    assert chapters_over_one_read(tmp_path, "architecture") == (
        ("docs/architecture/01-one.md", len(long.splitlines()), len(long)),)
    (tmp_path / "docs/architecture/01-one.md").unlink()
    with pytest.raises(OSError):  # unmeasured is unknown, never within the band
        chapters_over_one_read(tmp_path, "architecture")


def test_growth_base_is_the_event_base_and_nothing_else(tmp_path):
    _git(tmp_path, "init", "-q")
    shas = []
    for text in ("one\n", "one\ntwo\n"):
        (tmp_path / "f.txt").write_text(text, encoding="utf-8")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", text)
        shas.append(_git(tmp_path, "rev-parse", "HEAD").stdout.decode().strip())
    assert growth_base(tmp_path, shas[0]) == shas[0]
    assert growth_base(tmp_path, f" {shas[0][:12]} ") == shas[0]
    for absent in (None, "", "   ", "0" * 40, "f" * 40):
        assert growth_base(tmp_path, absent) == "", absent


@pytest.mark.parametrize(("environ", "applies"), [
    ({}, True),  # a local run with an explicit base
    ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/ouroboros-stable"}, False),
    ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main"}, False),
])
def test_the_rule_measures_one_change_not_a_release_range(environ, applies):
    assert measures_one_change(environ) is applies


# ------------------------------------------------------ whom the rule binds, from raw events

def _label(event, login, name="book-growth"):
    return {"event": event, "actor": {"login": login}, "label": {"name": name}}


OWNER_ADDS = _label("labeled", "razzant")


@pytest.mark.parametrize("repo,event,association,events,mode", [
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", [], "block"),
    ("razzant/ouroboros", "pull_request", "OWNER", [], "block"),
    ("razzant/ouroboros", "pull_request", "MEMBER", [], "block"),
    ("razzant/ouroboros", "pull_request", "CONTRIBUTOR", [], "warn"),
    ("razzant/ouroboros", "pull_request", "FIRST_TIME_CONTRIBUTOR", [], "warn"),
    ("someone/fork", "pull_request", "OWNER", [OWNER_ADDS], "warn"),  # a fork's own CI never blocks
    ("razzant/ouroboros", "push", "", [], "warn"),  # without a target ref this is not an official-line push
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", [OWNER_ADDS], "approved"),
    # Ouroboros's own token can apply the label, but only the repository owner's approves.
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", [_label("labeled", "ouroboros-agent")], "block"),
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", None, "block"),  # unreadable label events
])
def test_the_growth_mode_is_decided_from_raw_label_events(repo, event, association, events, mode):
    from scripts.book_growth_mode import decide

    assert decide(repo, "razzant", event, association, events) == mode


def test_the_mode_step_reads_events_only_for_a_binding_pull_request_and_fails_closed(tmp_path):
    from scripts.book_growth_mode import main

    output, calls = tmp_path / "output", []
    base = {"REPO": "razzant/ouroboros", "OWNER": "razzant", "EVENT": "pull_request",
            "ASSOCIATION": "COLLABORATOR", "PR_NUMBER": "7", "GITHUB_OUTPUT": str(output)}

    def read(repo, number, token):
        calls.append((repo, number))
        return [OWNER_ADDS]

    def unreadable(repo, number, token):
        raise OSError("403")

    assert main(base, read) == "approved" and calls == [("razzant/ouroboros", "7")]
    assert main({**base, "ASSOCIATION": "CONTRIBUTOR"}, read) == "warn" and len(calls) == 1
    assert main(base, unreadable) == "block"
    assert output.read_text(encoding="utf-8").splitlines() == ["mode=approved", "mode=warn", "mode=block"]


@pytest.mark.parametrize("job", ["quick-test", "full-test-runs"])
def test_both_ci_jobs_feed_the_size_lane_the_decided_mode(job):
    import yaml

    workflow = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"][job]["steps"]
    mode = next(step for step in steps if step.get("id") == "book_mode")
    size = next(step for step in steps if step.get("id") == "tests_size")
    assert mode["run"] == "python scripts/book_growth_mode.py"
    assert mode["env"]["SHA"] == "${{ github.sha }}"
    assert mode["env"]["REF"] == "${{ github.ref }}"
    assert steps.index(mode) < steps.index(size)
    assert size["env"][MODE_ENV] == "${{ steps.book_mode.outputs.mode }}"
    # PR label events for this mode; issue label events for the accepted-failure check.
    assert workflow["jobs"][job]["permissions"] == {"contents": "read", "pull-requests": "read", "issues": "read"}


PUSH_SHA = "a" * 40
PUSH_ENV = {"REPO": "razzant/ouroboros", "OWNER": "razzant", "EVENT": "push",
            "SHA": PUSH_SHA, "REF": "refs/heads/ouroboros"}
API_ROOT = "https://api.github.com/repos/razzant/ouroboros/"
PULLS_URL = f"{API_ROOT}commits/{PUSH_SHA}/pulls?per_page=100"


def _pull(number, *, repo="razzant/ouroboros", target="ouroboros", sha=PUSH_SHA, merged=True):
    return {"number": number, "state": "closed" if merged else "open",
            "merged_at": "2026-10-09T09:00:00Z" if merged else None, "merge_commit_sha": sha,
            "base": {"ref": target, "repo": {"full_name": repo}}}


def _events_url(number):
    return f"{API_ROOT}issues/{number}/events?per_page=100"


def _github_pages(monkeypatch, pages):
    """Serve raw API JSON through the real reader, including any pagination links."""
    import io
    import json
    from scripts import book_growth_mode

    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        assert timeout == 30
        body, link = pages[request.full_url]
        response = io.BytesIO(json.dumps(body).encode("utf-8"))
        response.headers = {"Link": link} if link else {}
        return response

    monkeypatch.setattr(book_growth_mode.urllib.request, "urlopen", urlopen)
    return calls


@pytest.mark.parametrize(("merged_approved", "reverse"), [
    pytest.param(False, False, id='unapproved-merge-open-first'),
    pytest.param(True, True, id='approved-merge-merged-first'),
])
def test_push_approval_belongs_to_the_exact_merged_pr(monkeypatch, reverse, merged_approved):
    from scripts.book_growth_mode import main

    pulls = [_pull(7, merged=False), _pull(8)]
    if reverse:
        pulls.reverse()
    calls = _github_pages(monkeypatch, {
        PULLS_URL: (pulls, ""),
        _events_url(7): ([OWNER_ADDS], ""),
        _events_url(8): ([OWNER_ADDS] if merged_approved else [], ""),
    })
    assert main(PUSH_ENV) == ("approved" if merged_approved else "block")
    assert calls == [PULLS_URL, _events_url(8)]


@pytest.mark.parametrize("pulls", [
    [],  # a direct push has no PR exception
    [_pull(7, sha="b" * 40)],
    [{**_pull(7), "state": "closed", "merged_at": None}],  # closed without merging
], ids=["absent", "wrong-sha", "closed-unmerged"])
def test_push_rejects_unrelated_approval(monkeypatch, pulls):
    from scripts.book_growth_mode import main

    calls = _github_pages(monkeypatch, {
        PULLS_URL: (pulls, ""), _events_url(7): ([OWNER_ADDS], ""),
    })
    assert main(PUSH_ENV) == "block"
    assert calls == [PULLS_URL]


@pytest.mark.parametrize(("event", "events", "expected"), [
    pytest.param("pull_request", [OWNER_ADDS], "approved", id='pull_request-owner'),
    pytest.param("pull_request", [_label("labeled", "ouroboros-agent")], "block", id='pull_request-non-owner'),
    pytest.param("pull_request", [OWNER_ADDS, _label("unlabeled", "someone"), _label("labeled", "someone")], "block", id='pull_request-non-owner-reapplied'),
    pytest.param("pull_request", [OWNER_ADDS, _label("unlabeled", "someone", "other")], "approved", id='pull_request-unrelated-removal'),
    pytest.param("push", [_label("labeled", "razzant", "other")], "block", id='push-unrelated-label'),
    pytest.param("push", [OWNER_ADDS, _label("unlabeled", "razzant")], "block", id='push-removed'),
    pytest.param("push", [_label("labeled", "someone"), _label("unlabeled", "someone"), OWNER_ADDS], "approved", id='push-owner-reapplied'),
])
def test_raw_label_history_controls_the_exception(monkeypatch, event, events, expected):
    from scripts.book_growth_mode import main

    pages = {PULLS_URL: ([_pull(7)], ""), _events_url(7): (events, "")}
    _github_pages(monkeypatch, pages)
    env = {**PUSH_ENV, "EVENT": event, "PR_NUMBER": "7", "ASSOCIATION": "COLLABORATOR"}
    assert main(env) == expected


def test_rerunning_the_same_push_uses_current_label_history(monkeypatch):
    from scripts.book_growth_mode import main

    events = [OWNER_ADDS]
    calls = _github_pages(monkeypatch, {
        PULLS_URL: ([_pull(7)], ""), _events_url(7): (events, ""),
    })
    assert main(PUSH_ENV) == "approved"
    events.append(_label("unlabeled", "someone"))
    assert main(PUSH_ENV) == "block"
    events.append(_label("labeled", "someone"))
    assert main(PUSH_ENV) == "block"
    events.extend([_label("unlabeled", "razzant"), OWNER_ADDS])
    assert main(PUSH_ENV) == "approved"
    assert calls == [PULLS_URL, _events_url(7)] * 4


@pytest.mark.parametrize("last_event,expected", [
    (_label("unlabeled", "someone"), "block"),
    (OWNER_ADDS, "approved"),
])
def test_push_reads_all_associated_pr_and_label_event_pages(monkeypatch, last_event, expected):
    from scripts.book_growth_mode import main

    next_pulls = PULLS_URL + "&page=2"
    next_events = _events_url(8) + "&page=2"
    calls = _github_pages(monkeypatch, {
        PULLS_URL: ([_pull(7, merged=False)], f'<{next_pulls}>; rel="next"'),
        next_pulls: ([_pull(8)], ""),
        _events_url(8): ([OWNER_ADDS, _label("unlabeled", "someone")], f'<{next_events}>; rel="next"'),
        next_events: ([last_event], ""),
    })
    assert main(PUSH_ENV) == expected
    assert calls == [PULLS_URL, next_pulls, _events_url(8), next_events]


def test_push_checks_every_matching_pr_instead_of_taking_the_first(monkeypatch):
    from scripts.book_growth_mode import main

    calls = _github_pages(monkeypatch, {
        PULLS_URL: ([_pull(7), _pull(8)], ""),
        _events_url(7): ([], ""), _events_url(8): ([OWNER_ADDS], ""),
    })
    assert main(PUSH_ENV) == "approved"
    assert calls == [PULLS_URL, _events_url(7), _events_url(8)]


@pytest.mark.parametrize("failed_endpoint", [PULLS_URL, _events_url(7)])
def test_unreadable_push_evidence_leaves_growth_strict(monkeypatch, failed_endpoint, capsys):
    from scripts import book_growth_mode

    pages = {PULLS_URL: ([_pull(7)], ""), _events_url(7): ([OWNER_ADDS], "")}
    _github_pages(monkeypatch, pages)
    read = book_growth_mode.urllib.request.urlopen

    def unavailable(request, timeout):
        if request.full_url == failed_endpoint:
            raise OSError("unavailable")
        return read(request, timeout)

    monkeypatch.setattr(book_growth_mode.urllib.request, "urlopen", unavailable)
    assert book_growth_mode.main(PUSH_ENV) == "block"
    assert "no exception applied" in capsys.readouterr().out


@pytest.mark.parametrize("env,expected", [
    (PUSH_ENV, "block"),
    ({**PUSH_ENV, "SHA": "", "PR_NUMBER": "7"}, "block"),  # no arbitrary PR-number fallback
    ({**PUSH_ENV, "REPO": "someone/fork"}, "warn"),
    ({}, "warn"),
])
def test_only_official_development_pushes_are_strict(monkeypatch, env, expected):
    from scripts.book_growth_mode import main

    calls = _github_pages(monkeypatch, {PULLS_URL: ([], "")})
    assert main(env) == expected
    assert calls == ([PULLS_URL] if env == PUSH_ENV else [])


@pytest.mark.parametrize("approved,tip,passes", [
    (False, 101, False), (True, 101, True), (False, 99, True),
])
def test_push_mode_reaches_the_size_lane_and_only_growth_fails(monkeypatch, capsys, tmp_path, approved, tip, passes):
    import sys
    from scripts.book_growth_mode import main

    _github_pages(monkeypatch, {
        PULLS_URL: ([_pull(7)], ""), _events_url(7): ([OWNER_ADDS] if approved else [], ""),
    })
    output = tmp_path / "step-output"
    main({**PUSH_ENV, "GITHUB_OUTPUT": str(output)})
    monkeypatch.setenv(MODE_ENV, output.read_text(encoding="utf-8").strip().removeprefix("mode="))
    capsys.readouterr()
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_REF", PUSH_ENV["REF"])
    consumer = sys.modules[__name__]
    monkeypatch.setattr(consumer, "growth_base", lambda repo, ref: "base")
    monkeypatch.setattr(consumer, "body_map_bytes", lambda repo, ref="": 100 if ref else tip)
    if passes:
        test_a_change_does_not_grow_the_body_map(capsys)
        if tip <= 100:
            assert capsys.readouterr().out == ""
    else:
        with pytest.raises(AssertionError, match="grew by 1 bytes"):
            test_a_change_does_not_grow_the_body_map(capsys)


@pytest.mark.parametrize("text_id,name", [("constitution", "BIBLE.md"), ("prompts", "Prompts")])
@pytest.mark.parametrize("mode,tip,passes", [
    ("block", 101, False), ("", 101, False), ("approved", 101, True), ("warn", 101, True),
    ("block", 100, True), ("block", 99, True),
])
def test_bible_and_prompt_growth_fails_unless_the_owner_approved(monkeypatch, capsys, text_id, name, mode, tip, passes):
    import sys

    consumer = sys.modules[__name__]
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setenv(MODE_ENV, mode)
    monkeypatch.setattr(consumer, "growth_base", lambda repo, ref: "base")
    monkeypatch.setattr(consumer, "ruled_text_bytes",
                        lambda repo, measured, ref="": tip if measured == text_id and not ref else 100)
    if passes:
        test_a_change_does_not_grow_bible_or_the_prompts(capsys)
    else:
        with pytest.raises(AssertionError, match=f"{name} grew by 1 bytes"):
            test_a_change_does_not_grow_bible_or_the_prompts(capsys)


def test_the_verdict_names_bible_and_the_prompts():
    assert book_growth_verdict("prompts", 10, 15, "approved") == (
        True, "Prompts grows by 5 bytes; the repository owner approved it (book-growth label).")
    passes, said = book_growth_verdict("constitution", 10, 11, "block")
    assert not passes and said.startswith("BIBLE.md grew by 1 bytes (10 -> 11)") and "same text" in said
