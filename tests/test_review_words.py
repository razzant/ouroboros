"""Words are review evidence: source fidelity, pairing, exclusions and delivery."""

import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.review_words import HEADING, compare_sources, section_budget, words_for_diff
from tests.test_governance_context import repo as governance_repo

repo = governance_repo
PATH = "ouroboros/notice.py"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout.decode("utf-8")


def changed_repo(root, before='def notice():\n    return "Review is ready"\n',
                 after='def notice():\n    return "Review needs your answer"\n', path=PATH):
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(before, encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "baseline")
    file.write_text(after, encoding="utf-8")
    git(root, "add", ".")
    return git(root, "diff", "--cached")


def pairs(words):
    assert not words.gaps
    return [(c.before.text if c.before else None, c.after.text if c.after else None) for c in words.changes]


def test_historical_2766_fixture_keeps_the_words_at_their_producers():
    history = json.loads((pathlib.Path(__file__).parent / "fixtures/review_words_2766.json").read_text(encoding="utf-8"))
    sections = []
    for commit in history:
        words = compare_sources((row["path"], row["before"], row["after"]) for row in commit["files"])
        assert not words.gaps
        sections.append(words.render(section_budget()))
    text = "\n".join(sections)
    # This is an extractor fixture of historical wording, not a pin on today's UI copy.
    for fragment in ("Main model: {old} → {new}.", "Changed via {…}; declared actor:",
                     "The review read uncommitted files; whether it saw {what} is not recorded.",
                     "Running work that follows Main uses this model at its next call. The new route",
                     "Answer saved · its work is paused; it reads this after Resume"):
        assert fragment in text
    for place in ("_ROW_PHRASES", "settings_change_text", "_GAP_WORDS", "MAIN_FOLLOW_NOTE", "PAUSED_STATUS.answer_saved"):
        assert place in text
    assert "(added): ∅ →" in text and "(removed):" in text
    assert "question.status." not in text
    assert sections[-1] == ""  # 7f76503bb changed none of these four producers


@pytest.mark.parametrize("path,before,after", [
    (PATH, 'NOTE = "Old owner words"', 'NOTE = "New owner words"'),
    (PATH, 'def show():\n return {"title": "Old owner words"}', 'def show():\n return {"title": "New owner words"}'),
    ("web/view.js", 'const NOTE = "Old owner words";', 'const NOTE = "New owner words";'),
])
def test_pairs_by_function_key_or_constant_in_both_directions(path, before, after):
    assert pairs(compare_sources([(path, before, after)])) == [("Old owner words", "New owner words")]
    assert pairs(compare_sources([(path, after, before)])) == [("New owner words", "Old owner words")]


def test_pure_refactor_has_no_section_including_moves_concatenation_and_expression_names():
    sources = [
        (PATH, 'def old():\n return f"Hello {name}, choose now"', 'NOTE = f"Hello {owner}, choose now"'),
        ("web/old.js", 'const NOTE = "Choose " + "your next step";', ''),
        ("web/new.js", '', 'function renamed() { return "Choose your next step"; }'),
    ]
    assert compare_sources(sources).render(section_budget()) == ""


def test_python_single_word_labels_and_defaults_are_not_lost():
    assert pairs(compare_sources([(PATH, 'def notice(label="Saved!"): return label',
                                   'def notice(label="Ready!"): return label')])) == [("Saved!", "Ready!")]
    assert pairs(compare_sources([(PATH, '', 'def notice(): return "Resume"')])) == [(None, "Resume")]
    assert pairs(compare_sources([(PATH, '', 'LABELS = {"Read now", "Read later"}')])) == [(None, "Read now"), (None, "Read later")]


def test_python_indirect_keys_are_excluded_alongside_visible_values():
    source = '''HEADER = "X-Owner-Note"
__all__ = ["show_words"]
def lookup(data, key):
    return data[key]
def show_words(data):
    headers = getattr(data, "headers")
    code = headers.get(HEADER)
    labels = {key: lookup(data, "internal_key") for key in ("old", "new")}
    return "Read these words"
'''
    assert pairs(compare_sources([(PATH, '', source)])) == [(None, "Read these words")]
    source += '''
def unrelated():
    key = "A visible label"
    return key
'''
    assert pairs(compare_sources([(PATH, '', source)]))[-1] == (None, "A visible label")


def test_python_includes_receivers_and_interpolated_conditionals_excludes_developer_text():
    source = '''"""Module documentation prose."""
# A comment for developers.
import logging
LOG = logging.getLogger(__name__)
def show(who):
    """Function documentation prose."""
    LOG.warning("Internal diagnostic words")
    raise ValueError("Invalid internal state")
    message = {"machine key words": "Owner sees this"}.get("machine key words")
    return "Hello {name}, choose now".format(name=who)
def also_show(held):
    return f"Action: {'Choose later' if held else 'Choose now'}"
'''
    texts = [after for _, after in pairs(compare_sources([(PATH, "", source)]))]
    assert texts == ["Owner sees this", "Hello {name}, choose now", "Action: {…}", "Choose later", "Choose now"]
    for suffix in ("", "\n# A different comment.\n"):
        assert compare_sources([(PATH, source, source + suffix)]).render(4000) == ""


def test_js_and_html_keep_fallbacks_templates_and_accessible_text_not_keys_or_css():
    source = '''// Developer comment prose
const selector = '.card .body';
const path = '/path with spaces/file.js';
const status = tr('i18n key with spaces', 'Read after Resume');
const note = `Привет ${owner}, choose ${choice} now`;
const html = `<button class="big button" style="color: red;" aria-label="Open the details">Show details</button>`;
node.querySelector('main .work');
node.style.cssText = 'display: none;';
console.info('Developer log words');
const keys = {"machine key words": true};
const tagged = html`<p>Read the next step</p>`;
const cssText = css`p { color: red; }`;
'''
    texts = [after for _, after in pairs(compare_sources([("web/view.js", "", source)]))]
    assert texts == ["Read after Resume", "Привет {…}, choose {…} now", "Open the details", "Show details", "Read the next step"]
    html = '<style>body { color: red; }</style><input placeholder="Choose a name"><p>Read the result</p>'
    assert pairs(compare_sources([("web/view.html", "", html)])) == [(None, "Choose a name"), (None, "Read the result")]


@pytest.mark.parametrize("source", [
    "const label = 'Save';", "node.textContent = 'Save';",
    "const html = `<button aria-label=\"Save\">Save</button>`;",
])
def test_js_single_word_labels_are_paired(source):
    after = source.replace("Save", "Resume")
    for before, after in ((source, after), (after, source)):
        changes = pairs(compare_sources([("client.js", before, after)]))
        assert changes and all(old in {"Save", "Resume"} and new in {"Save", "Resume"} and old != new
                               for old, new in changes)
        assert not compare_sources([("client.js", before, before)]).changes


def test_lower_case_codes_stay_out_while_capitalised_labels_stay_in():
    source = ("const row = {tone: 'warn', action: 'restart', label: 'Resume'};\n"
              "node.className += ' is-unknown';\n"
              "const html = `<input value=\"auto\"><button>Retry</button>`;\n")
    after = (source.replace("'warn'", "'error'").replace("'restart'", "'stop'").replace("' is-unknown'", "' is-qualified'")
             .replace('"auto"', '"manual"').replace("'Resume'", "'Continue'").replace(">Retry<", ">Try again<"))
    assert pairs(compare_sources([("web/view.js", source, after)])) == [("Resume", "Continue"), ("Retry", "Try again")]
    assert pairs(compare_sources([("web/view.js", after, source)])) == [("Continue", "Resume"), ("Try again", "Retry")]


def test_chained_js_calls_keep_receiver_callbacks_and_literal_receivers():
    source = '''function rows(slots) {
      return slots.map(slot => `<button>${tr('save', 'Save changes')}</button>`).join('');
    }
    const caption = 'Work done'.trim();
    node.classList.add('hidden'); node.querySelector('.hidden').closest('section');
    const same = node.dataset.mode === 'paused';
    '''
    after = source.replace("Save changes", "Resume work").replace("Work done", "Work paused")
    assert pairs(compare_sources([("web/view.js", source, after)])) == [("Save changes", "Resume work"), ("Work done", "Work paused")]
    assert pairs(compare_sources([("web/view.js", after, source)])) == [("Resume work", "Save changes"), ("Work paused", "Work done")]
    assert not compare_sources([("web/view.js", source, source)]).changes


def test_python_filters_machine_literals_and_model_inputs_by_use():
    source = r'''
import re as regex
from re import sub as substitute
import subprocess as proc
CLI = ['git', '-o', '--no-sandbox', 'some command argument']
proc.run(CLI)
pattern = r'words (with groups)'
regex.findall(pattern, value)
substitute(r'other (groups)', 'replacement text', value)
PROMPT = 'Decide what the user should do next'
messages = [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': 'Read the input carefully'}]
llm.chat(messages=messages)
def build_prompt():
    return 'Produce a detailed answer'
def label():
    return 'Resume'
LABEL = 'Decide what the user should do next'
TITLE = 'Saved!'
tokens = ['-o', '--no-sandbox', 'utf-8', 'install.json', 'dir/file.py', r'C:\output\file.txt',
          'snake_case', 'ERR_BUSY', 'D06']
'''
    assert pairs(compare_sources([(PATH, "", source)])) == [
        (None, "Resume"), (None, "Decide what the user should do next"), (None, "Saved!")]


def test_a_name_inside_a_lookup_is_not_a_key_for_its_other_assignments():
    old, new = "Parent task was interrupted before this child started.", "The parent stopped first."
    before = ('def retry(task):\n    fence = FENCES.get(str(task.get("root_task_id")))\n'
              f'    task["retry"] = {{"reason": "{old}"}}\n    return fence\n')
    assert pairs(compare_sources([(PATH, before, before.replace(old, new))])) == [(old, new)]


@pytest.mark.parametrize("path,found", [("web/lib.min.js", False), ("web/lib.js", True)])
def test_a_minified_bundle_is_machine_output_not_words(repo, path, found):
    diff = changed_repo(repo, "const a = 'Save changes';\n", "const a = 'Resume work';\n", path=path)
    assert bool(words_for_diff(repo, diff).changes) is found


def test_a_dictionary_of_error_messages_for_people_is_words_not_model_input():
    old, new = "This request was already saved or cancelled; nothing was saved now.", "This request is already closed."
    before = f'messages = {{"closed": "{old}"}}\n\ndef refuse(code):\n    return {{"error": messages[code]}}\n'
    assert pairs(compare_sources([(PATH, before, before.replace(old, new))])) == [(old, new)]
    chat = 'llm.chat(messages=[{"role": "user", "content": "Read the input carefully"}])\n'
    assert not compare_sources([(PATH, "", chat)]).changes


def test_a_system_row_for_the_owner_is_words_though_it_names_a_role():
    old, new = "the stop request was retracted from its mailbox", "the stop request was withdrawn"
    row = f'send({{"type": "send_message", "role": "system", "text": "{old}"}})\n'
    assert pairs(compare_sources([(PATH, row, row.replace(old, new))])) == [(old, new)]
    turn = 'reply = {"role": "assistant", "content": "Read the input carefully", "name": "reviewer"}\n'
    assert not compare_sources([(PATH, "", turn)]).changes


@pytest.mark.parametrize("path,source", [
    ("app.py", 'LABEL = "Resume"'),
    ("src/client.js", "const label = 'Resume';"),
    ("templates/index.html", '<button aria-label="Resume">Resume</button>'),
])
def test_foreign_layouts_extract_source_and_git_deltas(repo, path, source):
    after = source.replace("Resume", "Continue")
    diff = changed_repo(repo, source, after, path=path)
    expected = [("Resume", "Continue")] * (2 if path.endswith(".html") else 1)
    assert pairs(compare_sources([(path, source, after)])) == expected
    assert pairs(words_for_diff(repo, diff)) == expected
    assert not compare_sources([(path, source, source)]).changes


@pytest.mark.parametrize("unified", [0, 3])
def test_diff_uses_its_blobs_not_current_checkout(repo, unified):
    changed_repo(repo)
    diff = git(repo, "diff", "--cached", f"-U{unified}")
    (repo / PATH).write_text('def notice():\n return "Unrelated later words"\n')
    assert pairs(words_for_diff(repo, diff)) == [("Review is ready", "Review needs your answer")]


def test_new_deleted_renamed_files_and_no_final_newline(repo):
    before, after = 'NOTE = "Before owner words"', 'NOTE = "After owner words"'
    diff = changed_repo(repo, before, after)
    assert pairs(words_for_diff(repo, diff)) == [("Before owner words", "After owner words")]
    git(repo, "commit", "-qm", "change")
    git(repo, "mv", PATH, "ouroboros/renamed.py")
    assert words_for_diff(repo, git(repo, "diff", "--cached")).render(4000) == ""
    (repo / "supervisor").mkdir()
    (repo / "supervisor/new.py").write_text('NOTE = "New owner words"\n')
    git(repo, "add", ".")
    assert pairs(words_for_diff(repo, git(repo, "diff", "--cached"))) == [(None, "New owner words")]
    git(repo, "commit", "-qm", "move and add")
    git(repo, "rm", "supervisor/new.py")
    assert pairs(words_for_diff(repo, git(repo, "diff", "--cached"))) == [("New owner words", None)]


@pytest.mark.parametrize("quote_path", ["true", "false"])
def test_git_quoted_unicode_paths_do_not_turn_evidence_into_a_failure(repo, quote_path):
    changed_repo(repo, 'const TEXT = "Read before";', 'const TEXT = "Read after";', path="web/слова.js")
    git(repo, "config", "core.quotePath", quote_path)
    assert pairs(words_for_diff(repo, git(repo, "diff", "--cached"))) == [("Read before", "Read after")]


def test_unavailable_native_parser_keeps_other_sources_and_names_the_gap(monkeypatch):
    import tree_sitter_language_pack

    def unavailable(*args):
        raise OSError("native grammar could not be loaded")

    monkeypatch.setattr(tree_sitter_language_pack, "get_parser", unavailable)
    words = compare_sources([("web/view.js", "", 'const NOTE = "Read this";'),
                             (PATH, "", 'NOTE = "Python words remain"')])
    assert words.gaps and "native grammar" in words.render(4000)
    assert words.changes[0].after.text == "Python words remain"


def test_parse_and_preimage_gaps_are_disclosed_without_refusal(repo):
    words = compare_sources([(PATH, "", "def broken("), ("web/ok.js", "", 'const note = "Still visible words";')])
    assert words.gaps and "Extraction unavailable" in words.render(4000)
    assert "Still visible words" in words.render(4000)
    diff = changed_repo(repo).replace("index ", "index deadbeef", 1)
    assert "Extraction unavailable" in words_for_diff(repo, diff).render(4000)
    complete = git(repo, "diff", "--cached")
    assert "Extraction unavailable" in words_for_diff(repo, complete.rsplit("\n+", 1)[0]).render(4000)
    assert words_for_diff(repo, complete + "```\n\n- wrapper words\n") == words_for_diff(repo, complete)


def test_budget_discloses_cut_and_small_section_is_unchanged():
    source = "\n".join(f'NOTE_{n} = "Words for owner {n}"' for n in range(50))
    words = compare_sources([(PATH, "", source)])
    full = words.render(section_budget())
    limited = words.render(500)
    assert full.startswith(HEADING) and "OMISSION NOTE" not in full
    assert limited.startswith(HEADING) and "OMISSION NOTE" in limited and len(limited) <= 500
    assert "file/blob identities" in limited
    assert "OMISSION NOTE" in words.omission()
    assert words.render(0) == words.omission()


@pytest.mark.parametrize("retrieves", [False, True], ids=["packet", "scope-brief"])
@pytest.mark.parametrize("word_change", [False, True])
@pytest.mark.parametrize("layer", ["body", "core"])
def test_real_change_briefs_put_words_before_diff_and_deliver_design_for_python(repo, monkeypatch, retrieves, word_change, layer):
    from ouroboros.tools import review
    from ouroboros.tools.review_admission import build_two_part_brief
    from ouroboros.tools.review_subject import FrozenSubject, ReviewSubjectSpec
    from tests.test_design_review_delivery import DESIGN_BIBLE, _design

    _design(repo, DESIGN_BIBLE + '\n**Seed**\n\nFull design recipes.\n')
    before = 'def notice():\n return "Review is ready"\n'
    after = before.replace("ready", "waiting") if word_change else before + "\n# A code-only refactor\n"
    diff = changed_repo(repo, before, after)
    subject = FrozenSubject(spec=ReviewSubjectSpec(root_kind="system_repo", root=str(repo), kind="index",
                            governance_root=str(repo), layer=layer), diff_text=diff, diff_sha="test-sha",
                            tree_sha=git(repo, "write-tree").strip(), parent_sha=git(repo, "rev-parse", "HEAD").strip(),
                            name_status=(("M", PATH),))
    monkeypatch.setattr(review, "reviewer_context_window", lambda *a, **k: 1_000_000)
    packet = build_two_part_brief(subject, {"slot_id": "words", "model": "test/model", "retrieves": retrieves})
    text = packet["system"]
    assert (HEADING in text) is word_change
    assert (DESIGN_BIBLE in text) is (word_change and layer == "body")
    assert "Full design recipes." not in text
    if word_change:
        diff_heading = "## Staged diff" if retrieves else "# Staged diff"
        assert text.index(HEADING) < text.index(diff_heading)
        assert "### Staged diff" not in text
        assert text.count(HEADING) == 1
        if not retrieves:
            assert text.index(HEADING) >= packet["stable_prefix_len"]


def test_result_capture_keeps_only_attributed_paths_hashes_and_frozen_words(repo, tmp_path, monkeypatch):
    from ouroboros import mutation_attribution
    from ouroboros.tools import review_result

    changed_repo(repo)
    extra = repo / "supervisor/new.py"
    extra.parent.mkdir()
    extra.write_text('NOTE = "New visible notice"\n')
    (repo / "foreign.py").write_text('NOTE = "Somebody else wrote this"\n')
    large = repo / "data.csv"
    large.write_text("machine,data,not,owner,words\n" * 50_000, encoding="utf-8")
    snapshot = {"candidates": [PATH, "supervisor/new.py", "data.csv"], "base_commit": git(repo, "rev-parse", "HEAD").strip()}
    monkeypatch.setattr(mutation_attribution, "attributed_git_candidates", lambda *a, **k: snapshot)
    ctx = SimpleNamespace(drive_root=tmp_path / "drive", repo_dir=repo, task_id="words", task_contract={})
    produced = review_result.produced_material(ctx)
    assert not produced["issues"]
    assert "diff_text" not in produced
    assert produced == review_result.produced_material(ctx)
    git(repo, "add", "supervisor/new.py", "data.csv")
    assert produced == review_result.produced_material(ctx)
    subject = review_result.result_subject(ctx, SimpleNamespace(answer="Done"), governance_root=repo, produced=produced)
    (repo / PATH).write_text('NOTE = "Later checkout words"')
    for prompt in (review_result._packet_prompt("rules", subject.diff_text),
                   review_result._retrieving_task("rules", "source", subject.diff_text)):
        assert HEADING in prompt and "## Workspace diff" not in prompt
        assert "diff --git" not in prompt, "no diff rides the subject (FIX1)"
        assert "Full sources: the produced files" in prompt and "diff's file/blob" not in prompt
        assert str(large) in prompt and produced["files"][-1]["sha256"] in prompt
        # A packet seat reads a bounded excerpt of each produced text file from its exact retained copy, its omission
        # said; the 1.4 MB data file never rides whole (FIX1's inflation), its copy holds the rest.
        assert produced["files"][-1]["retained"] in prompt and prompt.count("machine,data,not,owner,words") < 600
        assert len(prompt) < 30_000 and f"### {large} (its first " in prompt
        assert "New visible notice" in prompt and "Review needs your answer" in prompt
        assert "Somebody else wrote this" not in prompt and "Later checkout words" not in prompt
    no_diff = review_result.compose_subject(ask="Explain", reply="Explanation", produced={}, verification="none")
    assert HEADING not in no_diff and "## Workspace diff" not in no_diff


@pytest.mark.parametrize(("window", "bounded", "retrieves"), [
    pytest.param(1_000_000, False, [False, True], id='1000000-False-mixed'),
    pytest.param(100_000, True, [True, True], id='100000-True-retrieving'),
])
def test_result_words_take_the_seats_share_of_the_window(repo, tmp_path, monkeypatch, window, bounded, retrieves):
    from ouroboros import mutation_attribution
    from ouroboros.tools import review, review_admission, review_result

    # The one subject reaches every seat, inline or in a retrieving seat's brief: their windows bound its words.
    plan = {"models": ["test/packet", "test/session"], "slot_ids": ["packet", "session"], "retrieves": retrieves,
            "session_profiles": ["", ""], "use_local": [False, False]}
    monkeypatch.setattr(review_admission, "seat_vectors", lambda _delivery: plan)

    changed_repo(repo)
    (repo / "report.html").write_text("".join(f"<p>Finding {i} reads as one sentence for the owner.</p>\n"
                                              for i in range(4000)), encoding="utf-8")
    snapshot = {"candidates": [PATH, "report.html"], "base_commit": git(repo, "rev-parse", "HEAD").strip()}
    monkeypatch.setattr(mutation_attribution, "attributed_git_candidates", lambda *a, **k: snapshot)
    monkeypatch.setattr(review, "reviewer_context_window", lambda *a, **k: window)
    ctx = SimpleNamespace(drive_root=tmp_path / "drive", repo_dir=repo, task_id="words", task_contract={})
    section = review_result.produced_material(ctx)["words_section"]
    assert section.startswith(HEADING) and "Finding 0 reads" in section
    assert ("OMISSION NOTE" in section) is bounded
    from ouroboros import result_material

    assert len(section) <= result_material._result_words_budget() < 4 * window


def test_result_word_capture_failure_does_not_undo_attribution(repo, tmp_path, monkeypatch):
    from ouroboros import mutation_attribution
    from ouroboros.tools import review_result, review_subject

    changed_repo(repo)
    snapshot = {"candidates": [PATH], "base_commit": git(repo, "rev-parse", "HEAD").strip()}
    monkeypatch.setattr(mutation_attribution, "attributed_git_candidates", lambda *a, **k: snapshot)
    ctx = SimpleNamespace(drive_root=tmp_path / "drive", repo_dir=repo, task_id="words", task_contract={})
    good = review_result.produced_material(ctx)
    monkeypatch.setattr(review_subject, "_git_bytes", lambda *a, **k: (1, b"", "test capture failure"))
    failed = review_result.produced_material(ctx)
    assert failed["files"] == good["files"] and not failed["issues"]
    assert "Extraction unavailable" in failed["words_section"]
    assert "workspace words" in failed["words_section"]
    assert "could not be attributed" not in str(failed)


def test_frozen_subject_shares_extraction_across_seats_but_not_revisions(repo, monkeypatch):
    import dataclasses
    from ouroboros import review_words
    from ouroboros.tools import review
    from ouroboros.tools.review_admission import build_two_part_brief
    from ouroboros.tools.review_subject import FrozenSubject, ReviewSubjectSpec

    diff = changed_repo(repo)
    subject = FrozenSubject(spec=ReviewSubjectSpec(root_kind="system_repo", root=str(repo), kind="index",
                            governance_root=str(repo), layer="body"), diff_text=diff, diff_sha="first",
                            tree_sha=git(repo, "write-tree").strip(), parent_sha=git(repo, "rev-parse", "HEAD").strip(),
                            name_status=(("M", PATH),))
    captured, extract = [], review_words.words_for_diff

    def count(root, text):
        captured.append(text)
        return extract(root, text)

    monkeypatch.setattr(review_words, "words_for_diff", count)
    monkeypatch.setattr(review, "reviewer_context_window", lambda *a, **k: 1_000_000)
    for retrieves in (False, True, False):
        packet = build_two_part_brief(subject, {"slot_id": "words", "model": "test/model", "retrieves": retrieves})
        assert "Review needs your answer" in packet["system"]
    assert captured == [diff]
    new_diff = diff.replace('+    return "Review needs your answer"', '+    return "Review needs your approval"')
    next_subject = dataclasses.replace(subject, diff_sha="second", diff_text=new_diff)
    assert next_subject.review_words != subject.review_words
    assert captured == [diff, new_diff]
