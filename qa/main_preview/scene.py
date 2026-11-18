"""Synthetic atlas conversation over the pinned checkout's static UI fixture."""
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ouroboros.gateway import work_status
from ouroboros.gateway.history_paging import room_view_fingerprint
from ouroboros.work_stop import current_stop
from tests import test_room_work_browser as rw

HERE = Path(__file__).resolve().parent
BASE = "e758a2190298d9890b1f26e632f63775d0b33d55"
ROOT, FINISHED, CHILD = "atlas-index", "atlas-maps", "atlas-symbols"
NAMES = {ROOT: "Сверить указатель атласа", FINISHED: "Собрать карты для атласа",
         CHILD: "Проверить условные обозначения"}
PROJECT = {"id": "atlas-project", "name": "Приложение к атласу", "chat_id": 42,
           "lifecycle": "active", "visible_revision": 0}
PROJECT_TASK, PROJECT_QUIZ = "atlas-appendix", "atlas-legend"
NATIVE_TASK, NATIVE_QUIZ = "atlas-editor", "atlas-format"
PASSAGE = "atlas-reading-place"
NOTES = [
    ("Как устроен атлас?", "Карты собраны по районам. Для каждой есть название, масштаб и список источников."),
    ("Что лежит в приложении?", "Там указатель. У одной реки есть два исторических названия."),
    ("Где отмечены старые названия?", "Рядом с современным названием стоит ссылка на лист карты."),
    ("Что пришлось уточнить?", "На двух листах расходились подписи. Их уже сверили с оригиналами."),
    ("Как читать разворот?", "Слева общий план долины, справа увеличенный фрагмент. Сетка координат общая."),
    ("Где искать высоты?", "Отметки высоты вынесены на отдельный слой; рельеф виден и без него."),
    ("Какой масштаб у вставки?", "Один сантиметр соответствует двум километрам. На соседнем листе масштаб другой."),
    ("Сколько вариантов легенды?", "Для печати есть цветная версия и версия со штриховкой. Символы совпадают."),
    ("Можно показать границу районов?", "Граница идёт по старому руслу. Для нового русла на карте есть отдельная линия."),
    ("Какие листы уже готовы?", "Северный и восточный прошли сверку. Южный ждёт проверки обозначений."),
    ("Как подписать источник?", "Под рамкой карты будут год съёмки и название архива. Полная ссылка в приложении."),
    ("Есть ли список всех карт?", "Да. В оглавлении у каждого района указаны номер листа и страница."),
]


def stamped(row, offset, chat=1):
    source = "progress" if row.get("is_progress") else "chat"
    return {**row, "chat_id": chat, "history_id": f"{source}:{offset}",
            "history_position": {"source": source, "offset": offset}}


def page_body(rows, cursor, *, older=None, recent=False):
    reasons = ["quota"] if older else []
    if not recent:
        reasons = ["page"]
    return {"messages": rows, "has_more": bool(older), "next_cursor": older, "page_cursor": cursor,
            "window": {"complete": not reasons, "truncated_by": reasons}}


def question_fixture(kind, rows, details):
    if not kind:
        return [], [stamped(rw.assistant("Сверка примечаний к карте.", 1), 1, PROJECT["chat_id"])]
    mirror = kind == "mirror"
    task, qid = (PROJECT_TASK, PROJECT_QUIZ) if mirror else (NATIVE_TASK, NATIVE_QUIZ)
    chat = PROJECT["chat_id"] if mirror else 1
    block = {"quiz_id": qid, "state": "open", "wait_for_answer": True,
             "asked_at": "2026-10-14T08:58:00Z" if mirror else "2026-10-14T08:57:00Z",
             "question": "Какую легенду приложить к карте?" if mirror else "В каком формате собрать атлас?",
             "options": [] if mirror else ["Для печати", "Для экрана"]}
    quiz = {"role": "assistant", "msg_type": "quiz", "task_id": task, "ts": block["asked_at"],
            "text": block["question"], "quiz": block}
    details[task] = rw.quiz_detail(task, qid, block["question"], chat_id=chat,
                                   **({"project_id": PROJECT["id"]} if mirror else {}))
    details[task]["owner_quiz"][qid] = block
    identity = {"task_id": task, "quiz_id": qid, "quiz_state": "open", "ts": block["asked_at"], "chat_id": 1}
    if mirror:
        pointer = {**identity, "role": "system", "system_type": "project_question_pointer",
                   "project_id": PROJECT["id"], "project_name": PROJECT["name"], "project_chat_id": chat,
                   "question": block["question"], "options": block["options"], "option_details": [],
                   "assumption": "", "host_facts": ""}
        rows.insert(0, stamped(pointer, 5))
        return [pointer], [stamped(quiz, 1, chat)]
    rows.insert(0, stamped(quiz, 5))
    return [{**identity, "native": True, "wait_for_answer": True}], []


class Scene(rw.Transport):
    def __init__(self, page, variant, history_mode="loaded", *, questions=None, helper_first=False, baseline=BASE):
        self.history_mode = history_mode
        self.posts_wire, self.history_log = [], []
        root_since, helper_since = ((1792000020, 1792000010) if helper_first else (1792000010, 1792000020))
        raw = {
            ROOT: {"suggested_name": NAMES[ROOT], "_budget_pause": {"reason": "budget", "scope": "global",
                   "status": "paused_before_dispatch", "replay_safe": True, "physical_calls": 0,
                   "auto_resume": False, "resume_policy": "manual_same_generation"}},
            CHILD: {"parent_task_id": FINISHED, "root_task_id": FINISHED, "suggested_name": NAMES[CHILD],
                    "_budget_pause_hold": {"reason": "root_fence_lifted_pending_selection", "release": "resume",
                                           "root_task_id": FINISHED}},
        }
        owners = [rw.activity(t, "budget_paused", task_name=NAMES[t], suggested_name=NAMES[t],
                  pause_cause=cause, can_resume=True, cancelable=t == ROOT, live=False, started_at=since,
                  stop=current_stop({"id": t, **raw[t]}), **lineage) | {"task_id": t}
                  for t, cause, since, lineage in (
                      (ROOT, "budget", root_since, {}), (CHILD, "unknown", helper_since,
                      {"parent_task_id": FINISHED, "root_task_id": FINISHED, "delegation_role": "subagent"}))]
        self.root_msg = stamped(rw.progress(ROOT, "Сверяю названия рек по листам указателя.", 1,
                                            suggested_name=NAMES[ROOT]), 20)
        self.archive = [stamped(rw.owner("Сверь названия рек в указателе.", "atlas-request", 0), 10), self.root_msg]
        rows = self.archive.copy() if history_mode == "loaded" else []
        rows.append(stamped(rw.assistant("Карты собраны. Осталась отдельная проверка обозначений.", 2), 29)
                    | {"task_id": FINISHED})
        for i, (ask, answer) in enumerate(NOTES):
            rows.append(stamped(rw.owner(ask, PASSAGE if i == 6 else f"atlas-note-{i}", i + 4), 50 + 2*i))
            rows.append(stamped(rw.assistant(answer, i + 4, 30), 51 + 2*i))
        details = {t: {"task_id": t, "chat_id": 1, "suggested_name": name, "root_task_id": FINISHED if t == CHILD else t,
                       "status": "completed" if t == FINISHED else "scheduled"} for t, name in NAMES.items()}
        details[CHILD].update(parent_task_id=FINISHED, delegation_role="subagent")
        current, project_rows = question_fixture(questions, rows, details)
        super().__init__(page, projects=[PROJECT], activities=owners,
                         history={1: rows, PROJECT["chat_id"]: project_rows}, details=details)
        self.raw, self.results = raw, {FINISHED: details[FINISHED]}
        state = super().body()
        if current:
            loose = work_status.place_questions(state["works"], current)
            for chat, loose_questions in loose.items():
                works = [work for work in state["works"] if str(work["room"]) == chat]
                state["rooms"][chat] = work_status.roll_up_room(works, questions=loose_questions)
            state["work_summary"] = work_status.roll_up_summary(state["rooms"], state["works"], complete=True)
        self.controls = {control["target"]: control for work in state["works"]
                         for control in work["controls"] if control["action"] == "resume"}
        assert self.controls == {ROOT: {"action": "resume", "target": ROOT},
                                 CHILD: {"action": "resume", "target": CHILD, "scope": "single"}}
        attention = [{**row, "attention_reason": "budget" if row["task_id"] == ROOT else "resume",
                      "model_waits": {}} for row in owners]
        self.snapshot = {**state, "sha": baseline, "active_chat_activities": [owners[0]], "runtime_mode": "advanced",
                         "current_questions": current, "current_questions_gaps": [],
                         "main_attention": attention if variant == "A" else [], "main_attention_complete": True,
                         "main_attention_gaps": []}
        queue = {"running": [], "pending": [{"id": t, "type": "task", "task": {"id": t, "type": "task",
                  "chat_id": 1, **raw[t]}} for t in (ROOT, CHILD)]}
        page.route("**/api/tasks?*", lambda r: r.fulfill(json={"tasks": [], "queue": queue}))
        page.route("**/api/ui/preferences", lambda r: r.fulfill(json={"main_seen_revision": 0,
                   "project_seen_revision": {PROJECT["id"]: 0}}))
        page.route("**/api/schedules", lambda r: r.fulfill(json={"tasks": []}))
        page.route("**/api/services/kept", lambda r: r.fulfill(json={"services": []}))
        # Intercept every fixture API mutation. No request is forwarded to a real host.
        page.route("**/api/**", self._read_or_record)
        if variant == "B":
            from .variant import install_variant
            install_variant(page)

    def body(self, **_):
        return self.snapshot

    def _read_or_record(self, route):
        request = route.request
        if request.method in {"GET", "HEAD"}:
            route.fallback()
            return
        url = urlparse(request.url)
        self.posts_wire.append((request.method, url.path, url.query, request.post_data))
        ident = url.path.split("/")[-2]
        allowed = (request.method == "POST" and ident in self.controls and url.path == f"/api/tasks/{ident}/resume"
                   and not url.query and request.post_data is None)
        route.fulfill(status=200 if allowed else 409,
                      json={"ok": allowed, "task_id": ident, "state": "resumed" if allowed else "refused"})

    def _history(self, route):
        query = parse_qs(urlparse(route.request.url).query)
        room = int(query.get("chat_id", [1])[0])
        around, cursor = query.get("around", [""])[0], query.get("cursor", [""])[0]
        branch = "latest"
        if around:
            ident = around.removeprefix("task:")
            found = room == 1 and around == f"task:{ROOT}" and self.history_mode != "absent"
            branch = "found" if found else "absent"
            body = page_body(self.archive if found else [], "atlas-archive" if found else None)
            body["anchor"] = {"state": branch, "task_id": ident}
            if found:
                body["anchor"]["history_id"] = self.root_msg["history_id"]
        elif cursor in {"older-atlas", "atlas-archive"} and room == 1 and self.history_mode == "archived":
            branch, body = "older", page_body(self.archive, "atlas-archive")
        elif cursor == f"after-latest-{room}":
            branch, body = "after", page_body([], None)
            body["after_cursor"] = cursor
        elif cursor and cursor != f"latest-{room}":
            route.fulfill(status=400, json={"error": "Unknown fixture cursor"})
            return
        else:
            older = "older-atlas" if room == 1 and self.history_mode == "archived" else None
            body = page_body(self.history.get(room, []), f"latest-{room}", older=older, recent=not cursor)
            if not cursor:
                body["after_cursor"] = f"after-latest-{room}"
            arrivals = [row for row in body["messages"] if row["role"] == "assistant" and not row.get("is_progress")]
            if arrivals:
                body["window"]["latest_message"] = {"history_id": arrivals[-1]["history_id"], "out_of_order": False}
        # Disjoint synthetic history windows share one frozen horizon.
        older_rows = self.archive if room == 1 and self.history_mode == "archived" else []
        universe = [*older_rows, *self.history.get(room, [])]
        bounds = {}
        for source in ("chat", "progress"):
            def end_of(rows):
                return max((r["history_position"]["offset"] + 1 for r in rows
                            if r["history_position"]["source"] == source), default=0)

            end, cut, delivered_end = map(end_of, (universe, older_rows, body["messages"]))
            lo, hi = {"absent": (0, 0), "found": (0, delivered_end), "older": (0, cut),
                      "after": (end, end)}.get(branch, (cut, end))
            bounds[source] = (lo, hi, end)
        body["coverage"] = {"v": 1, "view": room_view_fingerprint(room, [PROJECT["chat_id"]], [], {}),
                            "upper": {source: end for source, (_, _, end) in bounds.items()},
                            "spans": {source: {"from": lo, "to": hi, "chain": "dialog", "gaps": []}
                                      for source, (lo, hi, _) in bounds.items()}}
        self.history_log.append({"chat": room, "around": around, "cursor": cursor, "result": branch})
        route.fulfill(json=body)
