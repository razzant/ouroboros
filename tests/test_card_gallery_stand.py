"""The card gallery stand's own seams, without a browser: per-request scene headers, the one-snapshot rule, the
`look` scene validation, the reads-only answers and the server's lifetime. Browser-driven scenes live in
`test_card_gallery_browser.py` (``ui_browser`` lane); the quiz test that owns a server through an exceptional setup
path is the restructured `test_quiz_response_browser.py`."""
from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

import pytest

from tests.card_gallery_stand import CORPORA, GalleryServer, _check_scenes


def _get(stand, path, *, scene=None, step=None, cookie="", method="GET"):
    """One raw HTTP request to the stand's loopback port with exactly the headers the page sends."""
    parts = urlsplit(stand.url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    headers = {}
    if scene is not None:
        headers["X-Gallery-Scene"] = scene
    if step is not None:
        headers["X-Gallery-Step"] = str(step)
    if cookie:
        headers["Cookie"] = cookie
    conn.request(method, path, headers=headers)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, body


def test_the_api_reads_the_scene_from_request_headers_only():
    with GalleryServer() as stand:
        status, _ = _get(stand, "/api/state", scene="task_card.done", step=1)
        assert status == 200
        assert stand.requests[-1] == {"method": "GET", "path": "/api/state", "scene": "task_card.done",
                                      "step": 1, "status": 200}
        # A cookie no longer selects a scene, so two tabs cannot share one through it.
        status, _ = _get(stand, "/api/state", cookie="gallery_scene=task_card.done; gallery_step=2")
        assert status == 404
        assert stand.requests[-1]["scene"] == "" and stand.requests[-1]["step"] == 0
        # Per request: the next call names its own scene; a step that is not an integer reads as 0.
        status, _ = _get(stand, "/api/state", scene="task_card.failed", step="not-a-number")
        assert status == 200
        assert stand.requests[-1]["scene"] == "task_card.failed" and stand.requests[-1]["step"] == 0
        assert [r["scene"] for r in stand.requests[-3:]] == ["task_card.done", "", "task_card.failed"]


def test_no_write_is_answered_and_the_recorded_login_jobs_stay_reads():
    """The sign-in card's start is answered from its recorded job inside the page; the card itself only ever reads.

    Both recorded jobs stay ordinary reads — the waiting card polls one, the failed card does not poll — and the start
    the page used to post gets no answer, in any method."""
    with GalleryServer() as stand:
        _, corpus = _get(stand, "/gallery/corpus/card_gallery.json")
        for state in ("waiting", "failed"):
            job = json.loads(corpus)["login"][state]["responses"][0]["job_id"]
            status, body = _get(stand, f"/api/claudexor/login/{job}", scene=f"login.{state}")
            assert status == 200 and json.loads(body)["job"]["jobId"] == job
            assert stand.requests[-1] == {"method": "GET", "path": f"/api/claudexor/login/{job}",
                                          "scene": f"login.{state}", "step": 0, "status": 200}
        status, _ = _get(stand, "/api/claudexor/login", scene="login.waiting")
        assert status == 404  # the start itself has no read: the page answers it from the recorded job
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            status, _ = _get(stand, "/api/claudexor/login", scene="login.waiting", method=method)
            assert status == 405, method
            assert stand.requests[-1] == {"method": method, "path": "/api/claudexor/login",
                                          "scene": "login.waiting", "step": 0, "status": 405}


def test_corpora_are_one_snapshot_per_server_lifetime(tmp_path):
    draft = tmp_path / "draft.json"
    draft.write_bytes(CORPORA["card_gallery"].read_bytes())
    with GalleryServer(card_corpus=draft) as stand:
        _, before = _get(stand, "/gallery/corpus/card_gallery.json")
        original = json.loads(before)["task_card"]["done"]["state"][0]
        status, answer = _get(stand, "/api/state", scene="task_card.done")
        assert status == 200 and json.loads(answer) == original
        # The producer's next draft lands on disk mid-lifetime: parsed from the bytes already fetched, never a
        # locale-decoded read of the file, and written as the corpus itself is (UTF-8, no ASCII escapes).
        edited = json.loads(before)
        edited["task_card"]["done"]["state"] = [{"edited": True}]
        draft.write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")
        _, after = _get(stand, "/gallery/corpus/card_gallery.json")
        assert after == before  # the running stand keeps its snapshot
        status, answer = _get(stand, "/api/state", scene="task_card.done")
        assert status == 200 and json.loads(answer) == original  # the API answers the same snapshot
    with GalleryServer(card_corpus=draft) as restarted:
        _, served = _get(restarted, "/gallery/corpus/card_gallery.json")
        assert served != before  # a restart is what makes the draft visible
        status, answer = _get(restarted, "/api/state", scene="task_card.done")
        assert status == 200 and json.loads(answer) == {"edited": True}


def test_look_rejects_unknown_scene_ids_before_any_frame():
    available = ["quiz.open", "quiz.long", "task_card.done"]
    assert _check_scenes(None, available) == available
    assert _check_scenes([], available) == available  # an empty ask means every scene, as ``look`` treats it
    assert _check_scenes(["quiz.open"], available) == ["quiz.open"]
    with pytest.raises(ValueError) as caught:
        _check_scenes(["quiz.open", "nope.nope", "also.gone"], available)
    assert "nope.nope" in str(caught.value) and "also.gone" in str(caught.value)


def test_a_server_that_never_served_still_closes():
    stand = GalleryServer()  # bound, but its thread was never started
    stand.__exit__(None, None, None)  # a setup fault in a test must not leak the port or hang on a join


def test_an_enter_that_cannot_start_serving_closes_its_port(monkeypatch):
    stand = GalleryServer()
    monkeypatch.setattr(stand._thread, "start", lambda: (_ for _ in ()).throw(RuntimeError("no threads today")))
    with pytest.raises(RuntimeError):
        stand.__enter__()
    stand.__exit__(None, None, None)  # the failed enter already closed it; exit stays safe and hang-free
