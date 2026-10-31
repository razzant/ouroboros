// The catalog seam: host sentences minted from closed code→sentence tables (task headline
// words, cause sentences, question status, routing labels, time words) are translated at the
// place the table is read, by stable code, while the English source — and the twin fixture
// the Python side is pinned to — stays byte-identical.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyPayload, setMissTransport, CODE_PREFIX } from '../modules/i18n.js';
import { taskPresentation, taskReasonPhrase } from '../modules/log_events.js';
import { questionPresentation } from '../modules/question_presentation.js';
import { acceptanceIncidentClauses } from '../modules/acceptance_incident_presentation.js';
import { formatMsgTime } from '../modules/chat_activity.js';
import { routingAnnotationText, routingOptionLabel } from '../modules/routing_receipt.js';
import { cancelCauseClauses } from '../modules/cancel_presentation.js';

setMissTransport(() => Promise.resolve(null));

const fixture = JSON.parse(readFileSync(new URL('./fixtures/outcome_phase_parity.json', import.meta.url), 'utf8'));
const ENGLISH_HEADLINES = { done: 'Done', warn: 'Done with warnings', cancelled: 'Stopped', error: 'Failed', timeout: 'Failed', lifecycle_error: 'Failed', working: 'Working' };

const RU = {
    language: 'ru', english: false, revision: 2,
    entries: {
        [CODE_PREFIX + 'task.headline.done']: { text: 'Готово' },
        [CODE_PREFIX + 'task.headline.error']: { text: 'Не удалось' },
        [CODE_PREFIX + 'task.cause.author_stop']: { text: 'Уроборос остановился с незавершённой работой; одобрения ревью не было.' },
        [CODE_PREFIX + 'task.cause.acceptance_preparation_failed']: { text: 'Доказательства приёмки не удалось собрать локально.' },
        [CODE_PREFIX + 'question.status.waiting']: { text: 'Ждёт вашего ответа' },
        [CODE_PREFIX + 'routing.pending']: { text: 'Выбираю адресата…' },
        [CODE_PREFIX + 'routing.action.steer_task']: { text: 'Задача направлена' },
        [CODE_PREFIX + 'routing.generic_task']: { text: 'Задача' },
        [CODE_PREFIX + 'time.yesterday']: { text: 'Вчера' },
        [CODE_PREFIX + 'time.at']: { text: 'в' },
        'New task in {name}': { text: 'Новая задача в {name}' },
        'Not started: the request was empty': { text: 'Не запущено: запрос пуст' },
        'Stopped from the app ({control})': { text: 'Остановлена из приложения ({control})' },
        [CODE_PREFIX + 'cancel.scope_cascade']: { text: 'эта задача и её подзадачи' },
    },
};
const EN = { language: '', english: true, revision: 0, entries: {} };

function englishReadings() {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    yesterday.setHours(10, 5, 0, 0);
    return {
        headlines: Object.fromEntries(Object.keys(ENGLISH_HEADLINES).map((phase) => [phase, taskPresentation(phase).headline])),
        cause: taskReasonPhrase('author_stop'),
        rawCode: taskReasonPhrase('no_such_cause_code'),
        clauses: acceptanceIncidentClauses({ acceptance_incident: { status: 'open', incident_id: 'i1' } }, 'author_stop',
            (key) => taskReasonPhrase(key)),
        question: questionPresentation({ quiz_state: 'open', owner_wait_state: 'waiting' }).status,
        pending: routingAnnotationText({ status: 'pending' }),
        refused: routingAnnotationText({ status: 'refused', cause: 'Not started: the request was empty' }),
        refusedUnknown: routingAnnotationText({ status: 'refused', cause: 'Not started: something new' }),
        steered: routingAnnotationText({ action: 'steer_task', status: 'delivered', target: 'abc' }),
        option: routingOptionLabel({ action: 'new_task_in_project', project_name: 'Docs' }),
        cancel: cancelCauseClauses({ source: 'http_single', scope: 'cascade' }, { task_id: 't1' }),
        yesterday: formatMsgTime(yesterday.toISOString()),
        older: formatMsgTime('2026-01-05T10:00:00'),
    };
}

test('in English every catalog reading is the source text and matches the twin fixture', () => {
    applyPayload(EN);
    const en = englishReadings();
    assert.deepEqual(en.headlines, ENGLISH_HEADLINES);
    for (const row of fixture.cases) {
        if (!row.headline) continue;
        assert.equal(taskPresentation(row.phase).headline, row.headline, row.name);
    }
    assert.equal(en.cause, 'Ouroboros stopped with unfinished work; no review approval was granted.');
    assert.equal(en.rawCode, 'no_such_cause_code');
    assert.equal(en.question, 'Waiting for your answer');
    assert.equal(en.pending, 'Choosing the right destination…');
    assert.equal(en.refused, 'Not started: the request was empty');
    assert.equal(en.steered, 'Steered task · Task');
    assert.deepEqual(en.cancel.filter(Boolean), ['Stopped from the app (Stop now)', 'this task and its sub-tasks']);
    assert.equal(en.option, 'New task in Docs');
    assert.equal(en.yesterday.short, 'Yesterday, 10:05');
    assert.equal(en.older.short, 'Jan 5, 10:00');
    assert.equal(en.older.full, 'Jan 5, 2026 at 10:00');
});

test('an install language reads the same tables by code; what the memory lacks stays English', () => {
    applyPayload(EN);
    const before = JSON.stringify(englishReadings());
    applyPayload(RU);
    const ru = englishReadings();
    assert.equal(ru.headlines.done, 'Готово');
    assert.equal(ru.headlines.error, 'Не удалось');
    assert.equal(ru.headlines.timeout, 'Не удалось', 'timeout and lifecycle_error share the Failed word and its code');
    assert.equal(ru.headlines.warn, 'Done with warnings', 'no entry → the English source, never a blank');
    assert.equal(ru.cause, RU.entries[CODE_PREFIX + 'task.cause.author_stop'].text);
    assert.equal(ru.rawCode, 'no_such_cause_code', 'a code with no sentence stays raw in every language');
    assert.deepEqual(ru.clauses, [
        RU.entries[CODE_PREFIX + 'task.cause.acceptance_preparation_failed'].text,
        RU.entries[CODE_PREFIX + 'task.cause.author_stop'].text,
    ]);
    assert.equal(ru.question, 'Ждёт вашего ответа');
    assert.equal(ru.pending, 'Выбираю адресата…');
    assert.equal(ru.refused, 'Не запущено: запрос пуст', 'a host-composed refusal the memory knows reads translated');
    assert.equal(ru.refusedUnknown, 'Not started: something new', 'one it does not know stays English');
    assert.equal(ru.steered, 'Задача направлена · Задача');
    assert.deepEqual(ru.cancel.filter(Boolean), ['Остановлена из приложения (Stop now)', 'эта задача и её подзадачи'],
        'the control word inside the clause stays English by decision 5=A; the clause itself reads by code');
    assert.equal(ru.option, 'Новая задача в Docs');
    assert.equal(ru.yesterday.short, 'Вчера, 10:05');
    assert.equal(ru.older.short, `${new Intl.DateTimeFormat('ru', { month: 'short' }).format(new Date('2026-01-05T10:00:00'))} 5, 10:00`);
    assert.ok(ru.older.full.includes(' 2026 в 10:00'));
    // Switching back restores every reading byte for byte.
    applyPayload(EN);
    assert.equal(JSON.stringify(englishReadings()), before);
});

test('the raw cause table still works as the presenter lookup for callers outside the seam', () => {
    applyPayload(EN);
    const table = { acceptance_preparation_failed: 'A', author_stop: 'B' };
    assert.deepEqual(acceptanceIncidentClauses({ acceptance_incident: { status: 'open' } }, 'author_stop', table), ['A', 'B']);
    assert.deepEqual(acceptanceIncidentClauses({ acceptance_incident: { status: 'open' } }, 'final_message', table), ['A', '']);
    assert.deepEqual(acceptanceIncidentClauses({ acceptance_incident: { status: 'resolved' } }, 'author_stop', table), []);
});

test('a shortened cancel reason keeps the space before its preview note in every language', () => {
    const origin = { source: 'agent_tool', reason: 'x'.repeat(200) };
    applyPayload(EN);
    const english = cancelCauseClauses(origin, { task_id: 't1' })[1];
    assert.equal(english, 'x'.repeat(159) + '… (preview; the full reason is kept with the task)');
    applyPayload({ ...RU, entries: { ...RU.entries, [CODE_PREFIX + 'cancel.reason_preview_note']: { text: '(начало; полная причина хранится в задаче)' } } });
    const russian = cancelCauseClauses(origin, { task_id: 't1' })[1];
    assert.equal(russian, 'x'.repeat(159) + '… (начало; полная причина хранится в задаче)');
    applyPayload(EN);
});
