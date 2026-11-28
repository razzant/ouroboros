"""The text balance: a fact on every local surface, never a gate (BIBLE P3 c5).

The opt-in lane (tests/test_reference_book_budgets.py) compares a supplied contribution base;
here the same measurement reaches Ouroboros when it writes book text,
runs the commit gate's free checks before a paid reviewer and asks for codebase health.

The size rule follows what a mind carries (owner decision 2A, 25.11.2026): BIBLE.md, the prompts and the rendered
Body Map ride every request and owe their growth; a book and a standalone document are read on demand, so their
size delta is a stated fact under no rule (``ruled=False``), and a book chapter past one default read is named.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from ouroboros.reference_books import (
    BODY_MAP_TEXT,
    BOOK_ENTRYPOINTS,
    BOOK_GROWTH_RULE,
    CHAPTER_BAND_RULE,
    CHAPTER_READ_LINES,
    STANDALONE_DOCS,
    UnmeasuredText,
    book_balance_note,
    book_balances,
    book_source_sizes,
    compose_book,
    composed_size,
    contribution_base,
    load_reference_book,
    render_book_balance,
    ruled_text_of,
    ruled_text_paths,
    ruled_text_sizes,
)
from tests._cases import case_folders
from tests._reference_book_fixtures import BODY, CHAPTER, ENTRY, _book_repo, _git, _registry

REPO = pathlib.Path(__file__).resolve().parents[1]
ATLAS = "\n## Reference Atlas\n\n- [DESIGN.md](DESIGN.md) — how the screens look.\n"
FACT = "[a fact: read on demand, under no size rule]"
OPENING = ("The first chapter.", "The first chapter, grown.")  # the chapter opening the Map carries; +7 bytes


def _mapped_repo(root: pathlib.Path, *, official: bool = True) -> pathlib.Path:
    """``_book_repo`` whose entry carries its Reference Atlas, committed: the Body Map renders it beside the
    chapter openings (a tree without the Atlas renders a Map too, without those rows)."""
    repo = _book_repo(root, official=official)
    (repo / BOOK_ENTRYPOINTS["architecture"]).write_text(ENTRY + ATLAS, encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "the reference atlas")
    if official:
        _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    return repo


def _by_id(balances):
    return {balance.book_id: balance for balance in balances}


def test_composed_size_is_the_composed_book_on_the_real_tree():
    for book_id in BOOK_ENTRYPOINTS:
        composed = len(compose_book(load_reference_book(REPO, book_id)).encode("utf-8"))
        assert composed_size(book_source_sizes(REPO, book_id)) == composed


@pytest.mark.parametrize("remote,url", [
    ("canonical", "https://github.com/razzant/ouroboros.git"),
    ("renamed", "git@github.com:razzant/ouroboros.git"),
    ("origin", "ssh://git@GitHub.com/RAZZANT/Ouroboros.git/"),
])
def test_published_feature_keeps_contribution_growth(tmp_path, remote, url):
    clone = _mapped_repo(tmp_path / "clone", official=False)
    base = _git(clone, "rev-parse", "HEAD")
    _git(clone, "remote", "add", remote, url)
    target = f"refs/remotes/{remote}/ouroboros"
    _git(clone, "update-ref", target, base)
    # Names carry no authority: managed is a fork; publication has caught up
    # to HEAD on the author's feature branch, exactly the old zero-debt bug.
    _git(clone, "remote", "add", "managed", "https://github.com/contributor/ouroboros.git")
    fork = "managed" if remote == "origin" else "origin"
    if fork == "origin":
        _git(clone, "remote", "add", fork, "https://github.com/contributor/ouroboros.git")
    _git(clone, "checkout", "-qb", "feature")
    (clone / CHAPTER).write_text(BODY + "Committed growth.\n", encoding="utf-8", newline="\n")
    _git(clone, "commit", "-qam", "grow")
    for ref in (f"refs/remotes/{fork}/feature", "refs/remotes/managed/ouroboros"):
        _git(clone, "update-ref", ref, "HEAD")
    _git(clone, "branch", f"--set-upstream-to={fork}/feature")
    assert _git(clone, "rev-parse", "@{upstream}") == _git(clone, "rev-parse", "HEAD")
    published, rendered = (_by_id(book_balances(clone, [CHAPTER]))[key] for key in ("architecture", BODY_MAP_TEXT))
    # The book's committed growth is measured against the contribution base and stated as a fact: it owes nothing.
    assert published.vs_head == 0 and published.vs_upstream == len("Committed growth.\n") and published.owed == 0
    assert not published.ruled and rendered.ruled and rendered.vs_upstream == 0  # below the opening: the Map is +0
    # Advance the official target independently: the measured base must be the
    # common ancestor, not this tip and not the feature-tracking upstream.
    tree = _git(clone, "rev-parse", f"{base}^{{tree}}")
    tip = _git(clone, "commit-tree", tree, "-p", base, "-m", "official advance")
    _git(clone, "update-ref", target, tip)
    (clone / CHAPTER).write_text(BODY + "Committed growth.\nMore.\n", encoding="utf-8", newline="\n")
    balance = _by_id(book_balances(clone, [CHAPTER]))["architecture"]
    assert balance.vs_head == len("More.\n")
    assert balance.vs_upstream == len("Committed growth.\nMore.\n") and balance.owed == 0
    assert balance.upstream == target and balance.merge_base == base
    assert target in render_book_balance(balance) and base in render_book_balance(balance)
    assert balance.changed == ((CHAPTER, len("More.\n")),)
    assert book_balances(clone, ["notes.md"]) == []  # no book source touched, nothing measured


def test_linked_body_candidate_without_upstream_keeps_committed_growth(tmp_path):
    serving = _book_repo(tmp_path / "serving")
    base = _git(serving, "rev-parse", "HEAD")
    (serving / CHAPTER).write_text(BODY + "Inherited growth.\n", encoding="utf-8", newline="\n")
    _git(serving, "commit", "-qam", "serving growth")
    candidate = tmp_path / "candidate"
    _git(serving, "worktree", "add", "-qb", "candidate/task", str(candidate), "HEAD")
    assert (candidate / ".git").is_file()
    assert subprocess.run(["git", "rev-parse", "@{upstream}"], cwd=candidate, capture_output=True).returncode
    balance = _by_id(book_balances(candidate, [CHAPTER]))["architecture"]
    assert balance.vs_head == 0 and balance.vs_upstream == len("Inherited growth.\n")
    assert balance.merge_base == base and balance.upstream == "refs/remotes/canonical/ouroboros"


@pytest.mark.parametrize("baseline", ["absent_remote", "fork", "missing_book"])
def test_unknown_contribution_never_means_paid_even_with_measured_head(tmp_path, baseline):
    repo = _book_repo(tmp_path / "repo", official=False)
    if baseline != "absent_remote":
        url = "https://github.com/fork/ouroboros.git" if baseline == "fork" else "https://github.com/razzant/ouroboros.git"
        _git(repo, "remote", "add", "managed", url)
        if baseline != "missing_ref":
            base = _git(repo, "rev-parse", "HEAD")
            if baseline == "unrelated":
                base = _git(repo, "commit-tree", "HEAD^{tree}", "-m", "unrelated root")
            elif baseline == "missing_book":
                _git(repo, "rm", "--cached", "-q", CHAPTER, BOOK_ENTRYPOINTS["architecture"])
                tree = _git(repo, "write-tree")
                _git(repo, "read-tree", "HEAD")
                base = _git(repo, "commit-tree", tree, "-m", "before books")
                head = _git(repo, "commit-tree", "HEAD^{tree}", "-p", base, "-m", "add books")
                _git(repo, "update-ref", "HEAD", head)
            _git(repo, "update-ref", "refs/remotes/managed/ouroboros", base)
    for text in (BODY, BODY.replace(" and long enough to shorten", ""), BODY + "Added.\n"):
        (repo / CHAPTER).write_text(text, encoding="utf-8", newline="\n")
        balance = _by_id(book_balances(repo, [CHAPTER]))["architecture"]
        assert balance.vs_head == len(text.encode()) - len(BODY.encode())
        assert balance.vs_upstream is None
        note = book_balance_note(repo, [CHAPTER])
        assert "contribution unknown" in note and "nothing owed" not in note


@pytest.mark.parametrize("suffix", ["", "Short.\n"])
def test_committed_growth_can_be_paid_in_the_worktree(tmp_path, suffix):
    repo = _mapped_repo(tmp_path)
    (repo / CHAPTER).write_text(BODY + "Committed growth.\n", encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "growth")
    text = BODY if not suffix else "# One\n\nThe first chapter.\n\n" + suffix
    (repo / CHAPTER).write_text(text, encoding="utf-8", newline="\n")
    balance = _by_id(book_balances(repo, [CHAPTER]))["architecture"]
    assert balance.vs_head < 0 and balance.vs_upstream <= 0 and balance.owed == 0
    assert "nothing owed" in book_balance_note(repo, [CHAPTER])


def test_the_first_official_remote_by_name_is_the_base_whatever_git_shows_beside_its_url(tmp_path, monkeypatch):
    """#3004: a partial clone's remote reads ``<url> (fetch) [blob:none]`` in ``git remote -v``. The target is the
    first remote by name whose fetch URL, as Git resolves it (here through ``url.<base>.insteadOf``), is official."""
    repo = _book_repo(tmp_path / "repo", official=False)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "a.partial", "gh:razzant/ouroboros")
    _git(repo, "config", "url.https://github.com/.insteadOf", "gh:")
    _git(repo, "config", "remote.a.partial.partialclonefilter", "blob:none")
    _git(repo, "update-ref", "refs/remotes/a.partial/ouroboros", base)
    _git(repo, "remote", "add", "b-plain", "https://github.com/razzant/ouroboros.git")
    _git(repo, "update-ref", "refs/remotes/b-plain/ouroboros", _git(repo, "commit-tree", "HEAD^{tree}", "-m", "elsewhere"))
    major, minor = map(int, _git(repo, "--version").split()[2].split(".")[:2])
    shown = " [blob:none]" if (major, minor) >= (2, 37) else ""  # Git writes the filter there since 2.37
    assert f"a.partial\thttps://github.com/razzant/ouroboros (fetch){shown}" in _git(repo, "remote", "-v").splitlines()
    assert contribution_base(repo) == ("refs/remotes/a.partial/ouroboros", base)
    balance = _by_id(book_balances(repo, [CHAPTER]))["architecture"]
    assert balance.upstream == "refs/remotes/a.partial/ouroboros" and balance.vs_upstream == 0
    real_run = subprocess.run

    def resolving_fails(argv, *args, **kwargs):
        if argv[1:2] == ["ls-remote"]:
            raise subprocess.TimeoutExpired(argv, 30)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", resolving_fails)
    assert contribution_base(repo) == ("", "")  # unavailable, never raised


@pytest.mark.parametrize("no_lazy_fetch", ["1", "0"])  # "0": as a Git older than 2.44, which ignores the variable
@pytest.mark.parametrize("lacking", ["books", "test_module"])
def test_a_filtered_clone_is_measured_offline_and_what_it_lacks_stays_unknown(tmp_path, monkeypatch, no_lazy_fetch,
                                                                              lacking):
    """#3004: a ``--filter=blob:none`` clone of a feature tip lacks the base's side of what the feature changed: a
    test module, and with ``books`` the entrypoint, a chapter and BIBLE.md too. No measurement fetches them: no
    transport opens and no object is written. The cached base stays exact; a book whose base side is absent has an
    unknown contribution, while books read in full keep theirs. The pulse of the worktree and of the index counts
    from HEAD and says the contribution diff is unavailable. Once Git has fetched them on request, all is measured."""
    from ouroboros import reference_books, review

    monkeypatch.setitem(reference_books._OFFLINE_GIT, "GIT_NO_LAZY_FETCH", no_lazy_fetch)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")  # the test's own clone and fetch; nothing reaches a network
    # The entry carries its Atlas (the Map renders); the feature's entry edit is to its introduction, which the Map
    # carries: the Map's growth is measured where the base side is held, and unknown where it is not.
    mapped = ENTRY + ATLAS
    entry, longer = BOOK_ENTRYPOINTS["architecture"], mapped.replace("The map.", "The map, longer.")
    module, suite = "tests/test_x.py", "def test_a():\n    pass\n"
    source = _mapped_repo(tmp_path / "source", official=False)
    for path, text in {"BIBLE.md": "# BIBLE\n\nP0.\n", module: suite}.items():
        (source / path).parent.mkdir(parents=True, exist_ok=True)
        (source / path).write_text(text, encoding="utf-8", newline="\n")
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "constitution and a suite")
    base = _git(source, "rev-parse", "HEAD")
    _git(source, "checkout", "-qb", "feature")
    feature = {module: suite + "\n\ndef test_b():\n    pass\n"}
    if lacking == "books":
        feature.update({entry: longer, CHAPTER: BODY + "Feature.\n", "BIBLE.md": "# BIBLE\n\nP0. P1.\n"})
    for path, text in feature.items():
        (source / path).write_text(text, encoding="utf-8", newline="\n")
    _git(source, "commit", "-qam", "feature")
    _git(source, "config", "uploadpack.allowFilter", "true")
    _git(source, "config", "uploadpack.allowAnySHA1InWant", "true")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--filter=blob:none", "--branch", "feature", source.as_uri(), str(clone))
    _git(clone, "remote", "add", "canonical", "https://github.com/razzant/ouroboros.git")
    _git(clone, "update-ref", "refs/remotes/canonical/ouroboros", base)
    for path, text in {CHAPTER: feature.get(CHAPTER, BODY) + "Worktree.\n",
                       module: feature[module] + "\n\ndef test_c():\n    pass\n"}.items():
        (clone / path).write_text(text, encoding="utf-8", newline="\n")
    _git(clone, "add", CHAPTER, module)  # the index a commit holds: its suite side is read as well

    def absent() -> set[str]:  # `--missing=print` fetches nothing
        listed = _git(clone, "rev-list", "--objects", "--missing=print", base).splitlines()
        return {line[1:] for line in listed if line.startswith("?")}

    def known(measure, *args):  # an object this clone does not hold is unknown: the measure says so, it never fetches
        try:
            return measure(*args)
        except UnmeasuredText as exc:
            return exc

    def measured(trace: pathlib.Path):
        """Every measurement, asserting it opened no transport and wrote no object."""
        store = clone / ".git" / "objects"
        before = sorted((str(path), path.stat().st_size) for path in store.rglob("*") if path.is_file())
        monkeypatch.setenv("GIT_TRACE2_EVENT", str(trace))
        facts = (contribution_base(clone), known(book_source_sizes, clone, "architecture", base),
                 known(ruled_text_sizes, clone, "constitution", base), book_balances(clone, [CHAPTER, "BIBLE.md"]),
                 review.diff_pulse(clone, [CHAPTER, module]), review.diff_pulse(clone, staged=True),
                 review._blob_texts(clone, [f"{base}:{module}"]))
        monkeypatch.delenv("GIT_TRACE2_EVENT")
        assert sorted((str(path), path.stat().st_size) for path in store.rglob("*") if path.is_file()) == before
        assert "upload-pack" not in (trace.read_text(encoding="utf-8") if trace.exists() else "")
        return facts

    lacking_objects = absent()
    # The base's side of what the feature changed, and the first commit's entry before its Atlas, which the base reaches.
    assert len(lacking_objects) == (5 if lacking == "books" else 2)
    target, entry_at_base, bible_at_base, balances, pulse, staged, texts = measured(tmp_path / "offline.trace2")
    assert target == ("refs/remotes/canonical/ouroboros", base) and absent() == lacking_objects
    assert texts == {f"{base}:{module}": None}  # absent here, never fetched
    head_only = f"Diff pulse since HEAD (contribution diff from canonical/ouroboros merge-base {base[:12]} unavailable)"
    for shown in (pulse, staged):
        assert shown.startswith(f"{head_only}: files 2, ") and "tests: items +1/-0, new modules 0," in shown, shown
    shown = {b.book_id: (b.vs_head, b.vs_upstream) for b in balances}
    if lacking == "books":
        assert isinstance(entry_at_base, UnmeasuredText) and isinstance(bible_at_base, UnmeasuredText)
        assert "contribution unknown" in pulse
        assert shown == {"architecture": (len("Worktree.\n"), None), "constitution": (0, None),
                         BODY_MAP_TEXT: (0, None)}  # the Map's base side is absent too: unknown, never paid
    else:  # the books' base side is here: their contribution is measured
        assert entry_at_base == {entry: len(mapped.encode()), CHAPTER: len(BODY)}
        assert bible_at_base == {"BIBLE.md": len("# BIBLE\n\nP0.\n")} and "contribution unknown" not in pulse
        assert shown == {"architecture": (len("Worktree.\n"), len("Worktree.\n")), "constitution": (0, 0),
                         BODY_MAP_TEXT: (0, 0)}
        assert f"+10 B vs refs/remotes/canonical/ouroboros (merge-base {base})" in pulse

    for oid in lacking_objects:  # fetched on request, outside any measurement
        _git(clone, "cat-file", "-t", oid)
    assert not absent()
    target, entry_at_base, bible_at_base, balances, pulse, staged, texts = measured(tmp_path / "held.trace2")
    assert target == ("refs/remotes/canonical/ouroboros", base) and texts == {f"{base}:{module}": suite}
    assert entry_at_base == {entry: len(mapped.encode()), CHAPTER: len(BODY)}
    assert bible_at_base == {"BIBLE.md": len("# BIBLE\n\nP0.\n")}
    longer_by = len(longer) - len(mapped)  # ", longer": the introduction the Map carries
    grown = (longer_by + len("Feature.\n"), len(" P1."), longer_by) if lacking == "books" else (0, 0, 0)
    assert {b.book_id: b.vs_upstream for b in balances} == {
        "architecture": grown[0] + len("Worktree.\n"), "constitution": grown[1], BODY_MAP_TEXT: grown[2]}
    files = 4 if lacking == "books" else 2
    assert pulse.startswith(f"Diff pulse since canonical/ouroboros merge-base {base[:12]}: files {files}, ")
    assert staged.startswith(f"Diff pulse since canonical/ouroboros merge-base {base[:12]}: files {files}, ")
    for shown in (pulse, staged):
        assert "tests: items +2/-0, new modules 0," in shown and "contribution unknown" not in shown, shown


@pytest.mark.parametrize("url", [
    "/tmp/github.com/razzant/ouroboros.git", "https://example.invalid/razzant/ouroboros.git", "https://github.com:bad/razzant/ouroboros.git",
])
def test_remote_host_and_exact_repository_matter_even_with_an_official_push_url(tmp_path, url):
    repo = _book_repo(tmp_path, official=False)
    _git(repo, "remote", "add", "managed", url)
    _git(repo, "remote", "set-url", "--push", "managed", "https://github.com/razzant/ouroboros.git")
    _git(repo, "update-ref", "refs/remotes/managed/ouroboros", "HEAD")
    balance = _by_id(book_balances(repo, [CHAPTER]))["architecture"]
    assert balance.vs_head == 0 and balance.vs_upstream is None and balance.upstream == ""
    assert "contribution unknown" in book_balance_note(repo, [CHAPTER])


def test_membership_and_utf8_bytes_include_composition_separators(tmp_path):
    repo = _book_repo(tmp_path)
    second = "docs/architecture/02-two.md"
    entry = ENTRY + "- [Two](architecture/02-two.md)\n"
    body = "# Two\n\nКратко.\n"
    (repo / BOOK_ENTRYPOINTS["architecture"]).write_text(entry, encoding="utf-8")
    (repo / CHAPTER).write_text(BODY.replace(" and long enough to shorten", ""), encoding="utf-8")
    (repo / second).write_text(body, encoding="utf-8")
    expected = len(compose_book(load_reference_book(repo, "architecture")).encode("utf-8"))
    balance = _by_id(book_balances(repo, [second]))["architecture"]
    base_size = len(ENTRY.encode()) + len(BODY.encode()) + 2
    assert balance.size == expected
    assert balance.vs_head == balance.vs_upstream == expected - base_size
    assert sum(delta for _, delta in balance.changed) + 2 == balance.vs_head


def test_balance_helper_does_not_import_runtime_state(tmp_path):
    repo = _book_repo(tmp_path)
    code = ("import sys; from pathlib import Path; from ouroboros.reference_books import book_balances; "
            "assert book_balances(Path(sys.argv[1]))[0].vs_upstream == 0; "
            "assert not any(m == 'supervisor' or m.startswith('supervisor.') or m in "
            "{'ouroboros.config', 'ouroboros.settings_integrity', 'ouroboros.review_state', "
            "'ouroboros.runtime_mode_policy'} for m in sys.modules)")
    subprocess.run([sys.executable, "-B", "-c", code, str(repo)], cwd=REPO, check=True, capture_output=True)


def test_a_book_growth_is_a_stated_fact_and_the_maps_growth_states_the_rule(tmp_path):
    """A book chapter's growth below its opening is a fact owing nothing (the Map stays +0); the same growth in the
    chapter's opening, which every request carries in the Map, is owed and the note states the rule."""
    repo = _mapped_repo(tmp_path)
    (repo / CHAPTER).write_text(BODY + "Added.\n", encoding="utf-8", newline="\n")
    fact = book_balance_note(repo, [CHAPTER])
    assert "Architecture book" in fact and "+7 B vs HEAD" in fact and FACT in fact
    assert "Body Map" in fact and BOOK_GROWTH_RULE not in fact and "nothing owed" in fact and "\n" not in fact
    (repo / CHAPTER).write_text(BODY.replace(*OPENING), encoding="utf-8", newline="\n")
    grown = book_balance_note(repo, [CHAPTER])
    rendered = _by_id(book_balances(repo, [CHAPTER]))[BODY_MAP_TEXT]
    assert rendered.vs_head == rendered.vs_upstream == rendered.owed == len(" grown,") > 0
    assert grown.startswith("ℹ️ Text growth:\n") and grown.endswith(BOOK_GROWTH_RULE)
    assert render_book_balance(rendered) in grown and "nothing owed" not in grown
    (repo / CHAPTER).write_text(BODY.replace(" and long enough to shorten", ""), encoding="utf-8", newline="\n")
    paid = book_balance_note(repo, [CHAPTER])
    assert paid.startswith("ℹ️ Text growth:") and "nothing owed" in paid
    assert BOOK_GROWTH_RULE not in paid and "\n" not in paid
    assert book_balance_note(repo, ["notes.md"]) == ""


def test_the_body_map_balance_moves_only_with_what_the_map_carries(tmp_path):
    """The Map's balance is measured whenever a book source is touched: an edit below a chapter's opening leaves it at
    +0, an edit of that opening moves it by exactly the edit, and its line never lists the book's changed files."""
    repo = _mapped_repo(tmp_path)
    (repo / CHAPTER).write_text(BODY + "Below the opening.\n", encoding="utf-8", newline="\n")
    below = _by_id(book_balances(repo, [CHAPTER]))
    assert below[BODY_MAP_TEXT].vs_head == below[BODY_MAP_TEXT].vs_upstream == 0 and below[BODY_MAP_TEXT].ruled
    assert below["architecture"].vs_head == len("Below the opening.\n") and below[BODY_MAP_TEXT].changed == ()
    (repo / CHAPTER).write_text(BODY.replace(*OPENING), encoding="utf-8", newline="\n")
    opened = _by_id(book_balances(repo, [CHAPTER]))[BODY_MAP_TEXT]
    assert opened.vs_head == opened.vs_upstream == len(OPENING[1]) - len(OPENING[0]) and opened.changed == ()
    assert BODY_MAP_TEXT not in _by_id(book_balances(repo, ["notes.md", "BIBLE.md"]))  # no book source touched


def test_a_standalone_documents_growth_is_a_stated_fact_never_owed(tmp_path):
    repo = _mapped_repo(tmp_path)
    design = "docs/DESIGN.md"
    assert design in STANDALONE_DOCS
    (repo / design).write_text("# Design\n\nRules.\n", encoding="utf-8", newline="\n")
    _git(repo, "add", design)
    _git(repo, "commit", "-qm", "design")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    (repo / design).write_text("# Design\n\nRules. More rules.\n", encoding="utf-8", newline="\n")
    (balance,) = book_balances(repo, [design])
    assert (balance.book_id, balance.vs_head, balance.vs_upstream, balance.ruled, balance.owed) == (
        design, len(" More rules."), len(" More rules."), False, 0)
    note = book_balance_note(repo, [design])
    assert note == f"ℹ️ Text growth: {render_book_balance(balance)}; nothing owed."
    assert f"{design} " in note and FACT in note and BOOK_GROWTH_RULE not in note


def test_a_chapter_past_one_default_read_is_named_with_the_band_rule(tmp_path):
    repo = _mapped_repo(tmp_path)
    lines = CHAPTER_READ_LINES + 1
    long_chapter = BODY + "".join(f"Line {i}.\n" for i in range(lines - BODY.count("\n")))
    (repo / CHAPTER).write_text(long_chapter, encoding="utf-8", newline="\n")
    balance = _by_id(book_balances(repo, [CHAPTER]))["architecture"]
    assert balance.over_read == ((CHAPTER, lines, len(long_chapter)),) and balance.owed == 0
    note = book_balance_note(repo, [CHAPTER])
    assert f"[past one read: 01-one.md {lines:,} lines/{len(long_chapter):,} characters]" in note
    assert note.startswith("ℹ️ Text growth:\n") and note.endswith(CHAPTER_BAND_RULE) and BOOK_GROWTH_RULE not in note
    (repo / CHAPTER).write_text(BODY, encoding="utf-8", newline="\n")
    assert _by_id(book_balances(repo, [CHAPTER]))["architecture"].over_read == ()
    assert CHAPTER_BAND_RULE not in book_balance_note(repo, [CHAPTER])


def test_note_never_calls_an_unmeasured_book_paid(tmp_path):
    repo = tmp_path / "fresh"
    repo.mkdir()
    (repo / "notes.md").write_text("notes\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "no book yet")
    for path, text in {BOOK_ENTRYPOINTS["architecture"]: ENTRY + ATLAS, CHAPTER: BODY}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8", newline="\n")
    unmeasured = book_balance_note(repo, [CHAPTER])
    assert "n/a vs HEAD" in unmeasured and "nothing owed" not in unmeasured
    assert _by_id(book_balances(repo, [CHAPTER]))[BODY_MAP_TEXT].vs_head is None  # the Map's own n/a, not +0
    missing = ENTRY + "- [Two](architecture/02-missing.md)\n" + ATLAS  # a chapter list naming an absent file
    (repo / BOOK_ENTRYPOINTS["architecture"]).write_text(missing, encoding="utf-8", newline="\n")
    unavailable = book_balance_note(repo, [CHAPTER])
    assert "Architecture book balance unavailable" in unavailable and "Body Map balance unavailable" in unavailable
    measured = _mapped_repo(tmp_path / "measured")
    assert "nothing owed" in book_balance_note(measured, [CHAPTER])
    # A tree without the Reference Atlas still renders the Map every request carries: it is measured, not absent.
    pre_atlas = _book_repo(tmp_path / "pre-atlas")
    assert "Body Map " in (note := book_balance_note(pre_atlas, [CHAPTER])) and "nothing owed" in note
    assert "unavailable" not in note and "absent" not in note


def test_a_text_absent_here_is_named_and_one_there_but_unreadable_is_never_paid(tmp_path, monkeypatch):
    """Absent, unreadable and measured are three states. A touched document that is not there is named as absent (a
    fact; nothing owed). A book that is there but denied is named with why, beside the Map rendered from it, on the
    edit's note and in codebase health, and is never called paid."""
    from types import SimpleNamespace

    from ouroboros.tools import health

    repo = _mapped_repo(tmp_path)
    assert book_balance_note(repo, ["docs/DESIGN.md"]) == "ℹ️ Text growth: docs/DESIGN.md is absent here; nothing owed."
    readable = book_balance_note(repo, [CHAPTER])
    assert "unavailable" not in readable and "nothing owed" in readable
    assert "balance unavailable" not in health._codebase_health(SimpleNamespace(repo_dir=repo))
    chapter, real_read = repo / CHAPTER, pathlib.Path.read_bytes

    def denied(self):
        if self == chapter:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", denied)
    unreadable = book_balance_note(repo, [CHAPTER])
    assert "Architecture book balance unavailable: " in unreadable and "Body Map balance unavailable: " in unreadable
    assert "Permission denied" in unreadable and "nothing owed" not in unreadable
    assert "Architecture book balance unavailable: " in health._codebase_health(SimpleNamespace(repo_dir=repo))


@pytest.mark.parametrize(("growth", "missing_book"), [("", "architecture"), ("Added.\n", "development")])
def test_note_discloses_an_unavailable_book_beside_a_measured_book(tmp_path, missing_book, growth):
    repo = _mapped_repo(tmp_path)
    development = "docs/development/01-one.md"
    (repo / development).parent.mkdir(parents=True)
    (repo / development).write_text(BODY, encoding="utf-8", newline="\n")
    (repo / BOOK_ENTRYPOINTS["development"]).write_text(
        ENTRY.replace("Architecture", "Development").replace("architecture/", "development/"),
        encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "both books")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    paths = [CHAPTER, development]
    missing = CHAPTER if missing_book == "architecture" else development
    readable = development if missing_book == "architecture" else CHAPTER
    (repo / readable).write_text(BODY + growth, encoding="utf-8", newline="\n")
    complete = book_balance_note(repo, paths)
    readable_only = book_balance_note(repo, [readable])
    assert "unavailable" not in complete
    # A book's growth (below a chapter's opening) is a fact: nothing is owed with or without it.
    assert "nothing owed" in complete and BOOK_GROWTH_RULE not in complete
    assert (FACT in complete) and (f"{'+7' if growth else '+0'} B vs HEAD" in complete)

    (repo / missing).unlink()
    partial = book_balance_note(repo, paths)
    assert f"{missing_book.title()} book balance unavailable" in partial
    readable_id = "development" if missing_book == "architecture" else "architecture"
    assert render_book_balance(_by_id(book_balances(repo, [readable]))[readable_id]) in partial
    assert "Body Map balance unavailable" in partial  # rendered from both books: unmeasured without either
    assert "nothing owed" not in partial
    assert BOOK_GROWTH_RULE not in partial
    # The readable book's own line is unchanged by the other book's absence; the Map, rendered from both books, is
    # not: with either book there but unreadable it is unmeasured, never called paid and never a Map that shrank by
    # the handbook's chapter list.
    alone = book_balance_note(repo, [readable])
    assert render_book_balance(_by_id(book_balances(repo, [readable]))[readable_id]) in alone
    assert render_book_balance(_by_id(book_balances(repo, [readable]))[readable_id]) in readable_only
    assert "nothing owed" not in alone, alone
    if missing_book == "architecture":
        assert "Body Map balance unavailable" in alone
    assert book_balance_note(repo, ["notes.md"]) == ""

    (repo / missing).write_text(BODY, encoding="utf-8", newline="\n")
    assert book_balance_note(repo, paths) == complete


GROW = {
    "write_file": {"path": CHAPTER, "content": BODY + "Added.\n"},
    "edit_text": {"path": CHAPTER, "old_str": "P1 honest", "new_str": "P1 honest, added"},
    "apply_patch": {"patch": f"*** Update File: {CHAPTER}\n-P1 honest and long enough to shorten.\n"
                             "+P1 honest and long enough to shorten, added.\n"},
    "edit_batch": {"edits": [{"path": CHAPTER, "old_str": "P1 honest", "new_str": "P1 honest, added"}]},
}


def test_a_body_book_edit_carries_the_balance(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for tool in sorted(GROW):
        tmp_path = next(case_paths)
        with subtests.test(tool=tool), monkeypatch.context() as monkeypatch:
            import ouroboros.safety as safety

            monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
            registry = _registry(tmp_path, external=False)
            system = tmp_path / "system"  # give the body's entry its Atlas, so the Map renders and is measured
            (system / BOOK_ENTRYPOINTS["architecture"]).write_text(ENTRY + ATLAS, encoding="utf-8", newline="\n")
            _git(system, "commit", "-qam", "the reference atlas")
            _git(system, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
            result = str(registry.execute(tool, GROW[tool]))
            assert result.startswith("✅"), result[:300]
            # The chapter grew below its opening: a stated fact beside the Map at +0, nothing owed, no rule.
            assert "ℹ️ Text growth: " in result and "Architecture book" in result and FACT in result
            assert "Body Map" in result and "nothing owed" in result and BOOK_GROWTH_RULE not in result


def test_a_foreign_project_or_a_non_book_path_gets_no_balance(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for tool in sorted(GROW):
        tmp_path = next(case_paths)
        with subtests.test(tool=tool), monkeypatch.context() as monkeypatch:
            import ouroboros.safety as safety

            monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
            foreign = str(_registry(tmp_path / "a", external=True).execute(tool, GROW[tool]))
            assert foreign.startswith("✅") and "Text growth" not in foreign, foreign[:300]
            plain = {"write_file": {"path": "notes.md", "content": "notes, more\n"},
                     "edit_text": {"path": "notes.md", "old_str": "notes", "new_str": "notes, more"},
                     "apply_patch": {"patch": "*** Update File: notes.md\n-notes\n+notes, more\n"},
                     "edit_batch": {"edits": [{"path": "notes.md", "old_str": "notes", "new_str": "notes, more"}]}}[tool]
            body = str(_registry(tmp_path / "b", external=False).execute(tool, plain))
            assert body.startswith("✅") and "Text growth" not in body, body[:300]


def test_the_commit_gate_states_the_staged_books_balance_in_its_diff_pulse(tmp_path):
    """The gate's free checks carry the touched books' balance once, inside the diff
    pulse they show before the panel: the rule when the Map grew, "nothing owed" when
    only the book grew or it was paid; the readiness warnings no longer repeat it in a second rendering."""
    from ouroboros.tools import commit_gate
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_helpers import check_worktree_readiness

    repo = _mapped_repo(tmp_path / "body")
    # The chapter's opening grows the Map (owed, the rule); its body grows the book alone (a fact); a shorter body pays.
    for text, owed in ((BODY.replace(*OPENING), True), (BODY + "Added.\n", False),
                       (BODY.replace(" and long enough to shorten", ""), False)):
        (repo / CHAPTER).write_text(text, encoding="utf-8", newline="\n")
        _git(repo, "add", CHAPTER)
        progress: list = []
        ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data", task_id="t-books", emit_progress_fn=progress.append)
        commit_gate.deterministic_preflight(ctx, "docs: one chapter", [CHAPTER])
        [size] = [line for line in progress if line.startswith("Size (information):")]
        assert "Diff pulse since canonical/ouroboros merge-base" in size and "Architecture book" in size
        assert "Body Map" in size and FACT in size
        assert (BOOK_GROWTH_RULE in size) is owed and ("nothing owed" in size) is not owed
        assert not [w for w in check_worktree_readiness(repo, paths=[CHAPTER]) if "book" in w]


def test_bible_and_the_prompts_are_ruled_texts(tmp_path):
    """BIBLE.md and the prompt files (one text) ride every request: their growth is owed and the note states the rule
    (the books were ruled texts too before 25.11.2026; their growth is now a fact)."""
    for path, text in {"BIBLE.md": "# BIBLE\n\nP0.\n", "prompts/SYSTEM.md": "system words\n",
                       "prompts/SAFETY.md": "safety\n", "prompts/notes/extra.md": "not a prompt\n",
                       "notes.md": "notes\n"}.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8", newline="\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "remote", "add", "canonical", "https://github.com/razzant/ouroboros.git")
    _git(tmp_path, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    assert [ruled_text_of(path) for path in ("BIBLE.md", "prompts/SYSTEM.md", "prompts\\SAFETY.md",
                                             "prompts/notes/extra.md", "docs/BIBLE.md")] == [
        "constitution", "prompts", "prompts", "", ""]
    assert ruled_text_sizes(tmp_path, "prompts", "HEAD") == {"prompts/SAFETY.md": 7, "prompts/SYSTEM.md": 13}
    # Words moved between prompt files cost nothing.
    (tmp_path / "prompts/SYSTEM.md").write_text("system\n", encoding="utf-8", newline="\n")
    (tmp_path / "prompts/SAFETY.md").write_text("safety words\n", encoding="utf-8", newline="\n")
    (prompts,) = book_balances(tmp_path, ["prompts/SYSTEM.md"])
    assert (prompts.book_id, prompts.size, prompts.vs_head, prompts.vs_upstream, prompts.owed) == ("prompts", 20, 0, 0, 0)
    quiet = book_balance_note(tmp_path, ["prompts/SYSTEM.md"])
    assert quiet.startswith("ℹ️ Text growth: Prompts 20 B") and "nothing owed" in quiet
    # A new prompt file and a longer BIBLE are growth, and the note states the rule.
    (tmp_path / "prompts/NEW.md").write_text("new\n", encoding="utf-8", newline="\n")
    (tmp_path / "BIBLE.md").write_text("# BIBLE\n\nP0. P1.\n", encoding="utf-8", newline="\n")
    grown = book_balance_note(tmp_path, ["prompts/NEW.md", "BIBLE.md"])
    assert grown.startswith("ℹ️ Text growth:\n") and grown.endswith(BOOK_GROWTH_RULE)
    assert "BIBLE.md 17 B: +4 B vs HEAD" in grown and "Prompts 24 B: +4 B vs HEAD" in grown
    assert "(NEW +4, SAFETY +6, SYSTEM -6)" in grown
    # Other Markdown is not a ruled text.
    assert book_balance_note(tmp_path, ["prompts/notes/extra.md", "notes.md"]) == ""
    # A Windows checkout's CRLF copy of an unchanged file is not growth.
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "grown")
    (tmp_path / "BIBLE.md").write_bytes(b"# BIBLE\r\n\r\nP0. P1.\r\n")
    assert ruled_text_sizes(tmp_path, "constitution") == {"BIBLE.md": 17}
    assert "BIBLE.md 17 B: +0 B vs HEAD" in book_balance_note(tmp_path, ["BIBLE.md"])
    assert ruled_text_paths(tmp_path) == ["BIBLE.md", "prompts/NEW.md", "prompts/SAFETY.md", "prompts/SYSTEM.md"]


def test_review_blob_batch_uses_file_and_keeps_whitespace_and_missing_specs(tmp_path, monkeypatch):
    import os
    import stat
    from ouroboros import review

    repo = _book_repo(tmp_path / "repo", official=False)
    contents = {"tests/test_plain.py": "def test_a():\n    pass\n",
                "tests/test space.py": "def test_space():\n    pass\n"}
    for path, text in contents.items():
        (repo / path).parent.mkdir(exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8", newline="\n")
    _git(repo, "add", "tests")
    specs = [":tests/test_plain.py"] * 16000 + [":tests/test space.py", ":missing"]
    expected_input = (":tests/test_plain.py\n" * 16000 + ":missing\n").encode()
    real_run, sources = subprocess.run, []

    def run(cmd, **kwargs):
        if cmd[-2:] == ["cat-file", "--batch"]:
            assert "input" not in kwargs
            source = kwargs["stdin"]
            sources.append(source)
            assert stat.S_ISREG(os.fstat(source.fileno()).st_mode) and source.tell() == 0
            assert source.read() == expected_input
            source.seek(0)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(review.subprocess, "run", run)
    assert review._blob_texts(repo, specs) == {":" + path: text for path, text in contents.items()} | {":missing": None}
    assert len(sources) == 1 and sources[0].closed


@pytest.mark.parametrize("stage", ["create", "write", "seek", "seek_and_close"])
def test_review_temp_failure_uses_offline_show_and_preserves_suite_facts(tmp_path, monkeypatch, stage):
    import io
    from ouroboros import review, reference_books

    repo = _book_repo(tmp_path / "repo", official=False)
    path = "tests/test_sample.py"
    (repo / "tests").mkdir(exist_ok=True)
    (repo / path).write_text("def test_a():\n    pass\n", encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-qm", "suite base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / path).write_text("def test_a():\n    pass\n\ndef test_b():\n    pass\n", encoding="utf-8")
    _git(repo, "add", path)
    rows = {path: (3, 0)}

    def suite_facts():  # what the pulse reads, both sides of the index, then the facts of those texts
        return review._suite_facts(rows, *review._change_texts(repo, base, True, [path], [path]), [])

    expected = suite_facts()
    sources, shows = [], []
    real_run = subprocess.run

    class Source(io.BytesIO):
        def write(self, value):
            if stage == "write":
                raise OSError("temp write failed")
            return super().write(value)

        def seek(self, offset, *args):
            if stage in {"seek", "seek_and_close"}:
                raise OSError("temp seek failed")
            return super().seek(offset, *args)

        def close(self):
            was_closed = self.closed
            super().close()
            if stage == "seek_and_close" and not was_closed:
                raise OSError("failed write flushes again at close")

    def temporary():
        if stage == "create":
            raise OSError("temp create failed")
        source = Source()
        sources.append(source)
        return source

    def run(cmd, **kwargs):
        assert "cat-file" not in cmd, "failed preparation never starts a batch"
        if "show" in cmd:
            shows.append(cmd[-1])
            assert all(kwargs["env"][key] == value for key, value in reference_books._OFFLINE_GIT.items())
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(review.tempfile, "TemporaryFile", temporary)
    monkeypatch.setattr(review.subprocess, "run", run)
    assert suite_facts() == expected
    assert "items +1/-0" in expected[0]
    assert review._blob_texts(repo, [":missing"]) == {":missing": None}
    assert shows == [f"{base}:{path}", f":{path}", ":missing"]
    assert all(source.closed for source in sources)


@pytest.mark.parametrize("failure", [1, OSError("start"), subprocess.TimeoutExpired("git", 120), KeyboardInterrupt()])
def test_review_started_batch_is_not_retried_and_closes_input(tmp_path, monkeypatch, failure):
    from ouroboros import review, reference_books

    sources = []

    def run(cmd, **kwargs):
        sources.append(kwargs["stdin"])
        assert sources[0].tell() == 0 and "input" not in kwargs
        if failure == 1:
            return subprocess.CompletedProcess(cmd, 1, b"", b"batch failed")
        raise failure

    monkeypatch.setattr(review.subprocess, "run", run)
    monkeypatch.setattr(reference_books, "_git_bytes", lambda *a: pytest.fail("no fallback after batch dispatch"))
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            review._blob_texts(tmp_path, ["HEAD:tests/test_a.py"])
    else:
        assert review._blob_texts(tmp_path, ["HEAD:tests/test_a.py"]) == {"HEAD:tests/test_a.py": None}
    assert len(sources) == 1 and sources[0].closed


def test_an_earlier_side_is_a_delta_absent_or_unknown_and_an_unknown_side_prices_nothing(tmp_path):
    """Three states of an earlier side, never a fiction (the size lane's): measured gives a delta; absent there
    (proven) gives ``n/a`` and each file as a whole addition; there but unreadable gives "unavailable" with its
    reason, no delta and no per-file component, for a ruled text and a standalone document alike."""
    repo = _mapped_repo(tmp_path)
    prompt, design = "prompts/SYSTEM.md", "docs/DESIGN.md"
    assert design in STANDALONE_DOCS
    for path, text in {prompt: "An unchanged prompt.\n", design: "The same instruction stays.\n"}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8", newline="\n")
    # Absent at HEAD, a fact: n/a, and the file is a whole addition.
    new = book_balance_note(repo, [prompt, design])
    assert "Prompts 21 B: n/a vs HEAD" in new and "(SYSTEM +21)" in new and "(DESIGN +28)" in new
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a prompt and a standalone document")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    measured = book_balance_note(repo, [prompt, design])
    assert "Prompts 21 B: +0 B vs HEAD, +0 B vs " in measured and "docs/DESIGN.md 28 B: +0 B vs HEAD" in measured
    assert "nothing owed" in measured and "unavailable" not in measured
    # There but unreadable: the unchanged HEAD blob of each is moved aside, the working files stay readable.
    held = []
    for path in (prompt, design):
        oid = _git(repo, "rev-parse", f"HEAD:{path}")
        blob = repo / ".git" / "objects" / oid[:2] / oid[2:]
        blob.rename(tmp_path / oid)
        held.append((tmp_path / oid, blob))
    try:
        unknown = book_balance_note(repo, [prompt, design])
        by_id = _by_id(book_balances(repo, [prompt, design]))
    finally:
        for aside, blob in held:
            aside.rename(blob)
    for text_id, name, size in (("prompts", "Prompts", 21), (design, design, 28)):
        balance = by_id[text_id]
        assert (balance.size, balance.vs_head, balance.vs_upstream, balance.changed) == (size, None, None, ())
        assert "cannot be read here" in balance.head_unknown and balance.base_unknown
        assert f"{name} {size:,} B: unavailable at HEAD (" in unknown
    assert "+0 B" not in unknown and "SYSTEM +" not in unknown and "DESIGN +" not in unknown and "n/a" not in unknown
    assert "nothing owed" not in unknown and "contribution unknown" in unknown
    assert book_balance_note(repo, [prompt, design]) == measured  # restored


def test_a_base_that_rendered_no_map_is_said_so_never_priced_with_todays_renderer(tmp_path):
    """The Map's balance against a revision renders that revision's books with the running code, a comparison of
    sources. A revision whose own code had no Map renderer sent no Map: the balance says so there, with no delta,
    as the size lane measures it (``map_renderer_at``: one answer for both); once the base renders one, the same
    edit is a measured delta again."""
    from ouroboros import reference_books as books

    repo = _mapped_repo(tmp_path)
    renderer = repo / "ouroboros" / "reference_books.py"
    renderer.write_text("# the book contract, before the Map\n", encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "a base whose code renders no Map")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", base)
    renderer.write_text("# the book contract\ndef body_map(): ...\n", encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "the contribution that renders the Map")
    assert (books.map_renderer_at(repo, base), books.map_renderer_at(repo, "HEAD")) == (False, True)
    assert books.revision_body_map_sizes(repo, base) is None  # the lane: no earlier Map, nothing compared
    (repo / CHAPTER).write_text(BODY.replace(*OPENING), encoding="utf-8", newline="\n")
    grown = len(OPENING[1]) - len(OPENING[0])
    rendered = _by_id(book_balances(repo, [CHAPTER]))[BODY_MAP_TEXT]
    assert (rendered.vs_head, rendered.vs_upstream, rendered.base_unknown) == (grown, None, books.NO_MAP_RENDERED)
    line = render_book_balance(rendered)
    assert line == (f"Body Map {rendered.size:,} B: +{grown} B vs HEAD, no Map rendered at "
                    f"refs/remotes/canonical/ouroboros (merge-base {base})")
    assert rendered.owed == grown  # what is known to have grown: this edit against HEAD
    # The book itself has both sides at that base: its delta is measured there as before.
    assert _by_id(book_balances(repo, [CHAPTER]))["architecture"].vs_upstream == grown
    # The same edit against a base that renders a Map: a measured delta.
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    again = _by_id(book_balances(repo, [CHAPTER]))[BODY_MAP_TEXT]
    assert (again.vs_head, again.vs_upstream, again.base_unknown) == (grown, grown, "")
    with pytest.raises(UnmeasuredText, match="cannot be listed"):
        books.map_renderer_at(repo, "0" * 40)
