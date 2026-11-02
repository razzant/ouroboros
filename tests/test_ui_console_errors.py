"""The browser smokes set aside exactly the engine read's own network line, nothing else."""

import json
from types import SimpleNamespace

import pytest

from tests.ui_console_errors import PageErrors, unexpected_console_errors

ENGINE = "/api/claudexor/maintenance/harnesses"
LINE = "Failed to load resource: the server responded with a status of 503 (Service Unavailable)"


def test_the_refused_engine_read_line_is_set_aside():
    assert unexpected_console_errors([(LINE, ENGINE)], [(ENGINE, 503)]) == []


def test_a_foreign_503_or_a_script_error_still_fails_the_smoke():
    errors = [(LINE, "/api/files/download"), ("ReferenceError: x is not defined", "")]
    failed = [("/api/files/download", 503), (ENGINE, 503)]
    assert unexpected_console_errors(errors, failed) == [LINE, "ReferenceError: x is not defined"]


def test_nothing_is_set_aside_without_a_recorded_refusal_and_only_once_per_refusal():
    assert unexpected_console_errors([(LINE, ENGINE)], []) == [LINE]
    assert unexpected_console_errors([(LINE, ENGINE), (LINE, ENGINE)], [(ENGINE, 503)]) == [LINE]


class _Page:
    """Events PageErrors reads, including Playwright's split WebKit error name. Every listener
    registered for an event runs, as in Playwright: a second registration records twice."""

    def __init__(self):
        self.handlers, self.main_frame = {}, SimpleNamespace(url="")

    def on(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    def error(self, text, name=""):
        error = Exception(text)
        error.name = name
        for handler in self.handlers.get("pageerror", []):
            handler(error)

    def commit(self, url, frame=None):
        frame = frame or self.main_frame
        frame.url = url
        for handler in self.handlers.get("framenavigated", []):
            handler(frame)


OWN = "/127.0.0.1:65140/api/state due to access control checks."
CORS = "/localhost:65140/api/state due to access control checks."
FETCH = "Fetch API cannot load http"  # the error name Playwright splits off WebKit's report


def test_only_webkits_report_of_the_pages_own_fetch_is_set_aside():
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://127.0.0.1:65140/")
    for text in (OWN, CORS, "TypeError: undefined is not an object"):
        page.error(text, name="Fetch API cannot load http")
    assert errors == [OWN, CORS, "TypeError: undefined is not an object"]
    # Another origin (here only the host differs) is a real cross-origin refusal.
    assert errors.unexpected() == [CORS, "TypeError: undefined is not an object"]


def test_the_same_text_counts_for_an_origin_the_main_frame_never_committed_or_outside_webkit():
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://127.0.0.1:65140/", SimpleNamespace(url=""))  # a child frame is not the page
    page.error(OWN, name="Fetch API cannot load http")
    assert errors.unexpected() == [OWN]
    page = _Page()
    chromium = PageErrors(page, "chromium")
    page.commit("http://127.0.0.1:65140/")
    page.error(OWN, name="Fetch API cannot load http")
    assert chromium.unexpected() == [OWN]


def test_an_earlier_visited_origin_is_not_exempt_after_navigation():
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://localhost:65140/")  # B
    page.commit("http://127.0.0.1:65140/")  # A
    page.error(CORS, name="Fetch API cannot load http")  # A fetches B without CORS
    assert errors.unexpected() == [CORS]


@pytest.mark.parametrize("split", [False, True])
def test_a_different_scheme_is_cross_origin_even_at_the_same_host_and_port(split):
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://127.0.0.1:65140/")
    text = OWN if split else "Fetch API cannot load https:/" + OWN
    page.error(text, name="Fetch API cannot load https" if split else "")
    assert errors.unexpected() == [text]


def test_reload_and_leaving_document_errors_are_classified_when_received():
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://127.0.0.1:65140/first")
    page.error(OWN, name="Fetch API cannot load http")  # cancelled by reload
    page.commit("http://127.0.0.1:65140/first")
    page.error(OWN, name="Fetch API cannot load http")  # cancelled before navigation commits
    page.commit("http://localhost:65140/second")
    assert errors == [OWN, OWN]
    assert errors.unexpected() == []
    page.error(OWN, name="Fetch API cannot load http")  # after commit: no old-origin exemption
    assert errors.unexpected() == [OWN]


def test_a_later_navigation_cannot_hide_an_earlier_cross_origin_error():
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://127.0.0.1:65140/")
    page.error(CORS, name="Fetch API cannot load http")
    page.commit("http://localhost:65140/")
    assert errors.unexpected() == [CORS]


def test_the_verdict_stays_with_its_entry_when_the_list_changes():
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://127.0.0.1:65140/")
    page.error(OWN, name="Fetch API cannot load http")
    errors.clear()
    page.error("TypeError: undefined is not an object")  # lands where the exempt entry was
    assert errors.unexpected() == ["TypeError: undefined is not an object"]
    errors.insert(0, "ReferenceError: x is not defined")  # an adopter's own entry counts
    page.error(OWN, name="Fetch API cannot load http")
    errors.sort()
    assert errors.unexpected() == ["ReferenceError: x is not defined", "TypeError: undefined is not an object"]
    assert errors == ["/127.0.0.1:65140/api/state due to access control checks.",
                      "ReferenceError: x is not defined", "TypeError: undefined is not an object"]


def test_an_unrecoverable_scheme_is_not_assumed_to_be_the_pages_scheme():
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://127.0.0.1:65140/")
    page.error(OWN)
    assert errors.unexpected() == [OWN]


def test_full_urls_default_ports_and_an_already_committed_page():
    page = _Page()
    page.main_frame.url = "https://example.test/"
    errors = PageErrors(page, "webkit")
    page.error("Fetch API cannot load https://example.test:443/api/state due to access control checks.")
    assert errors.unexpected() == []
    foreign_port = "https://example.test:444/api/state due to access control checks."
    page.error(foreign_port)
    assert errors.unexpected() == [foreign_port]


# One list for several pages (the shared browser fixture): each page's own origin, at arrival.

def test_each_watched_page_is_judged_by_its_own_committed_origin():
    first, second = _Page(), _Page()
    errors = PageErrors(first, "webkit")
    assert errors.watch(second) is second
    first.commit("http://127.0.0.1:65140/")
    second.commit("http://localhost:65140/")
    for page in (first, second):
        for text in (OWN, CORS):
            page.error(text, name=FETCH)
    assert errors == [OWN, CORS, OWN, CORS]
    # The other page's origin is no exemption.
    assert errors.unexpected() == [CORS, OWN]
    assert [(entry.page, entry.own_fetch) for entry in errors] == [(1, True), (1, False), (2, False), (2, True)]


def test_another_pages_navigation_never_changes_this_pages_verdicts_or_address():
    first, second = _Page(), _Page()
    errors = PageErrors(None, "webkit")
    errors.watch(first)
    errors.watch(second)
    first.commit("http://127.0.0.1:65140/a")
    second.commit("http://127.0.0.1:65140/b")
    second.error(OWN, name=FETCH)  # before the second page's next document commits: still its own origin
    second.commit("http://localhost:65140/c")
    second.error(OWN, name=FETCH)  # after that commit the origin it left is no exemption
    first.error(OWN, name=FETCH)  # the first page never left its origin
    first.error(CORS, name=FETCH)  # and the second page's new origin is none of its own
    assert errors.unexpected() == [OWN, CORS]
    assert [(entry.page, entry.url, entry.own_fetch) for entry in errors] == [
        (2, "http://127.0.0.1:65140/b", True), (2, "http://localhost:65140/c", False),
        (1, "http://127.0.0.1:65140/a", True), (1, "http://127.0.0.1:65140/a", False)]


def test_an_undeclared_page_keeps_every_error_and_a_later_declaration_judges_only_later_ones():
    page = _Page()
    errors = PageErrors(None, "webkit")
    errors.watch(page, navigates=False)
    page.commit("http://127.0.0.1:65140/")
    page.error(OWN, name=FETCH)  # no navigation declared: it counts
    errors.watch(page)  # the test is about to reload it
    page.error(OWN, name=FETCH)
    errors.watch(page, navigates=False)  # watching again adds no listener and narrows nothing back
    page.error(OWN, name=FETCH)
    assert [len(page.handlers[event]) for event in ("pageerror", "framenavigated")] == [1, 1]
    assert errors == [OWN, OWN, OWN]
    assert [entry.own_fetch for entry in errors] == [False, True, True]
    assert errors.unexpected() == [OWN]


def test_a_declared_page_still_fails_on_its_script_and_native_dialog_errors():
    page = _Page()
    errors = PageErrors(page, "webkit")
    page.commit("http://127.0.0.1:65140/")
    page.error("undefined is not an object", name="TypeError")
    page.error("native dialog used", name="Error")  # the Skills tests' dialog guard
    assert errors.unexpected() == ["undefined is not an object", "native dialog used"]


def test_outside_webkit_no_declared_page_sets_anything_aside():
    first, second = _Page(), _Page()
    errors = PageErrors(first, "chromium")
    errors.watch(second)
    for page in (first, second):
        page.commit("http://127.0.0.1:65140/")
        page.error(OWN, name=FETCH)
    assert errors.unexpected() == [OWN, OWN]


def test_equal_text_from_two_pages_keeps_each_entrys_own_verdict_through_list_edits():
    strict, navigating = _Page(), _Page()
    errors = PageErrors(None, "webkit")
    errors.watch(strict, navigates=False)
    errors.watch(navigating)
    for page in (strict, navigating):
        page.commit("http://127.0.0.1:65140/")
        page.error(OWN, name=FETCH)
    assert errors == [OWN, OWN] and errors.unexpected() == [OWN]
    errors.reverse()
    assert [(entry.page, entry.own_fetch) for entry in errors] == [(2, True), (1, False)]
    errors.remove(OWN)  # by text, as for any list: the first equal entry goes, whichever page it came from
    assert [(entry.page, entry.own_fetch) for entry in errors] == [(1, False)] and errors.unexpected() == [OWN]
    errors.append(str(OWN))  # an adopter's own copy counts, though it equals a set-aside entry's text
    navigating.error(OWN, name=FETCH)
    assert errors.unexpected() == [OWN, OWN]
    assert json.loads(json.dumps(errors)) == [OWN, OWN, OWN]


def test_the_report_gives_each_entrys_verdict_page_and_address():
    page = _Page()
    errors = PageErrors(None, "webkit")
    errors.watch(page)
    page.commit("http://127.0.0.1:65140/")
    page.error(OWN, name=FETCH)
    page.error("x is not defined", name="ReferenceError")
    errors.append("added by hand")
    assert errors.report() == [f"set aside on page 1 at http://127.0.0.1:65140/: {OWN}",
                               "unexpected on page 1 at http://127.0.0.1:65140/: x is not defined",
                               "unexpected: added by hand"]
