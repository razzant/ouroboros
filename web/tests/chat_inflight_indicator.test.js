import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { computeHydratedDirectActivities } from '../modules/chat.js';
import {
    createStateSnapshotSequencer,
    isForegroundLiveCard,
    unkeyedFrameEndsTurn,
} from '../modules/chat_activity.js';
import { routingAnnotationText } from '../modules/routing_receipt.js';
import { emptyWorkState, roomStatusView } from '../modules/work_status.js';
import { room } from './work_fixture.js';

// The header is the server's room (work_status.roomStatusView); a state carrying one room.
const inRoom = (activity, extra = {}) => ({ ...emptyWorkState(), room: room(activity, extra) });

const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');

test('review-only owner anchors do not advertise foreground task activity', () => {
    const root = { isConnected: true };
    assert.equal(isForegroundLiveCard({ root, groupId: 'owner', finished: false }), true);
    assert.equal(isForegroundLiveCard({
        root, groupId: 'owner', finished: false, reviewAnchor: true,
    }), false);
});

test('only a legacy unkeyed model final clears unscoped state; a System row settles its own submission', () => {
    // The cleanup guard also requires positive terminal evidence. Keep this
    // source contract resilient to the guard's explicit conjunction.
    const cleanupStart = chatSource.indexOf('if (!finalizing && (!explicitTaskId || typedTerminal))');
    const finalCleanup = chatSource.slice(
        cleanupStart,
        chatSource.indexOf("if (msg.system_type === 'task_summary')", cleanupStart),
    );
    assert.match(
        finalCleanup,
        /} else if \(unkeyedFrameEndsTurn\(msg\)\) \{[^}]*?activeDirectActivities\.clear\(\);[^}]*?pendingSubmissions\.clear\(\);\s*} else pendingSubmissions\.delete\(String\(msg\.client_message_id \|\| ''\)\);\s*}/,
    );
});

test('System voice concludes no Main turn; an unkeyed model final still ends it (#1403)', () => {
    // What the live handler does with the owner's unscoped send for each frame (the branch pinned above):
    // the send is this client's own fact, the room the server's.
    const after = (frame) => roomStatusView(emptyWorkState(), { sending: !unkeyedFrameEndsTurn(frame) }).text;
    for (const kind of ['reminder', 'skill_notice', 'terminal_incident', 'budget_notice', 'startup_notice',
        'presence_recovery_required', 'command_reply', '']) {
        assert.equal(after({ role: 'system', system_type: kind, task_id: '', content: 'Call mother' }), 'Sending', kind);
    }
    assert.equal(after({ role: 'assistant', task_id: '', content: 'Done.' }), 'Online');
    // The voice decides, never the words: a reply that quotes a reminder still ends the turn.
    assert.equal(after({ role: 'assistant', task_id: '', content: 'Reminder · Ouroboros · written Oct 3 14:05' }), 'Online');
});

test('routing receipts display the event-time label while keeping raw target metadata', () => {
    const annotation = {
        action: 'steer_task',
        status: 'delivered',
        target: 'opaque-task-id',
        target_label: 'Launch 🚀 › Ship release',
    };
    assert.equal(routingAnnotationText(annotation), 'Steered task · Launch 🚀 › Ship release');
    assert.equal(annotation.target, 'opaque-task-id');
});

test('legacy routing receipts and manual choices use neutral labels, never raw ids', () => {
    assert.equal(routingAnnotationText({
        action: 'steer_task', status: 'delivered', target: 'opaque-task-id',
    }), 'Steered task · Task');
    assert.equal(routingAnnotationText({
        action: 'project_route', status: 'delivered', target: 'opaque-project-id',
    }), 'Project routing · Project');
    const manual = routingAnnotationText({
        action: 'manual', status: 'needs_manual_target', options: [
            { action: 'steer_task', task_id: 'opaque-option-id' },
            { action: 'project_route', project_id: 'opaque-project-id' },
        ],
    });
    assert.equal(manual, 'Choose a target · Task / Project');
    assert.equal(manual.includes('opaque-'), false);
});

test('a refusal row with neither options nor a cause reads «Not routed», never «Choose a target»', () => {
    // Such a row is a receipt written before the host sentence existed (or by a
    // producer that bypasses the receipt rail): it carries nothing to click, so
    // inviting a choice would be a lie about what the owner can do.
    assert.equal(routingAnnotationText({
        action: 'promote_chat_to_task', status: 'needs_manual_target',
        target: 'opaque-project-id', target_label: 'Аудит', options: [],
    }), 'Not routed · Аудит');
    assert.equal(routingAnnotationText({
        action: 'manual', status: 'needs_manual_target',
    }), 'Not routed');
});

test('a host cause on a refused routing receipt outranks the status matrix; an empty cause changes nothing', () => {
    // The host authors the owner sentence for a REFUSED act (`cause`); the
    // client never derives one from `reason`. The field is absent/empty on
    // scheduled, delivered and pending rows and on the picker frame, so those
    // keep their labels (the options case above stays «Choose a target · …»).
    const cause = 'Not started: the working folder can\'t be used';
    const refused = routingAnnotationText({
        action: 'promote_chat_to_task', status: 'needs_manual_target',
        target: 'x', target_label: 'Аудит', cause,
    });
    assert.equal(refused, cause);
    assert.equal(refused.includes('Choose a target'), false);
    assert.equal(routingAnnotationText({
        action: 'steer_task', status: 'delivered', target: 'opaque-task-id',
        target_label: 'Launch', cause: '',
    }), 'Steered task · Launch');
});

test('state snapshots and failure authority stay monotonic across reversed completion', () => {
    const applied = [];
    const requestTimes = [100, 200];
    const snapshots = createStateSnapshotSequencer(
        (data, requestedAt) => applied.push({ data, requestedAt }),
        () => requestTimes.shift(),
    );
    const older = snapshots.begin();
    const newer = snapshots.begin();

    assert.deepEqual(older, { generation: 1, requestedAt: 100, path: '/api/state' });
    assert.deepEqual(newer, { generation: 2, requestedAt: 200, path: '/api/state' });
    assert.equal(snapshots.isCurrent(older), true);
    assert.equal(snapshots.apply(newer, { activities: [] }), true);
    // An older non-OK/catch path can no longer regress the newer header.
    assert.equal(snapshots.isCurrent(older), false);
    assert.equal(snapshots.apply(older, { activities: ['stale-root'] }), false);
    assert.deepEqual(applied, [{ data: { activities: [] }, requestedAt: 200 }]);
});

test('the header: offline beats every room, whatever the room or the owner\'s sends', () => {
    assert.deepEqual(roomStatusView(inRoom('working'), { connected: false, sending: true }), {
        kind: 'offline', text: 'Reconnecting…', tone: 'offline', mark: 'ring', owner: false, amber: '',
    });
});

test('the header: a working room says Working with moving dots and the transcript typing row', () => {
    // The transcript's typing row follows the moving mark (chat.js syncChatStatus).
    assert.deepEqual(roomStatusView(inRoom('working')), {
        kind: 'thinking', text: 'Working', tone: 'live', mark: 'moving', owner: false, amber: '',
    });
    assert.equal(roomStatusView(inRoom('working', { call: 'thinking' })).text, 'Thinking');
});

test('the server room\'s words reuse the installed translation memory', async () => {
    const { applyPayload, currentPayload, CODE_PREFIX } = await import('../modules/i18n.js');
    const prior = currentPayload();
    try {
        applyPayload({ language: 'ru', english: false, entries: {
            [CODE_PREFIX + 'task.chip.working']: { text: 'Работает' },
            [CODE_PREFIX + 'task.chip.thinking']: { text: 'Думает' },
        } });
        for (const [call, expected] of [['working', 'Работает'], ['thinking', 'Думает'], ['', 'Работает']]) {
            assert.equal(roomStatusView(inRoom('working', { call })).text, expected);
        }
    } finally { applyPayload(prior); }
});

test('the header: the owner\'s own unconfirmed send says Sending with dots over a quiet room', () => {
    assert.deepEqual(roomStatusView(emptyWorkState(), { sending: true }), {
        kind: 'thinking', text: 'Sending', tone: 'live', mark: 'moving', owner: false, amber: '',
    });
});

test('the header: an idle ready room is Online', () => {
    assert.deepEqual(roomStatusView(emptyWorkState()), {
        kind: 'online', text: 'Online', tone: 'online', mark: 'none', owner: false, amber: '',
    });
});

test('computeHydratedDirectActivities: filters turns by chatId', () => {
    const turns = [
        { activity_id: 'act-main-1', chat_id: 1, kind: 'direct_chat', phase: 'thinking' },
        { activity_id: 'act-proj-2', chat_id: 2, kind: 'direct_chat', phase: 'thinking' },
        { activity_id: 'act-main-2', chat_id: 1, kind: 'direct_chat', phase: 'thinking' },
    ];

    const mapChat1 = computeHydratedDirectActivities(new Map(), turns, 1);
    assert.equal(mapChat1.size, 2);
    assert.ok(mapChat1.has('act-main-1'));
    assert.ok(mapChat1.has('act-main-2'));
    assert.ok(!mapChat1.has('act-proj-2'));

    const mapChat2 = computeHydratedDirectActivities(new Map(), turns, 2);
    assert.equal(mapChat2.size, 1);
    assert.ok(mapChat2.has('act-proj-2'));
});

test('computeHydratedDirectActivities: removes completed activities not in snapshot', () => {
    const initialMap = new Map([
        ['act-old', { activityId: 'act-old', kind: 'direct_chat', phase: 'thinking' }],
        ['act-keep', { activityId: 'act-keep', kind: 'direct_chat', phase: 'thinking' }],
    ]);

    const turns = [
        { activity_id: 'act-keep', chat_id: 1, kind: 'direct_chat', phase: 'working' },
        { activity_id: 'act-new', chat_id: 1, kind: 'direct_chat', phase: 'thinking' },
    ];

    const updated = computeHydratedDirectActivities(initialMap, turns, 1);
    assert.equal(updated.size, 2);
    assert.ok(!updated.has('act-old'));
    assert.ok(updated.has('act-keep'));
    assert.equal(updated.get('act-keep').phase, 'working');
    assert.ok(updated.has('act-new'));
});

test('computeHydratedDirectActivities: an incomplete census deletes nothing', () => {
    const initialMap = new Map([
        ['act-held', { activityId: 'act-held', kind: 'direct_chat', phase: 'thinking' }],
    ]);
    const partial = computeHydratedDirectActivities(initialMap, [], 1, null, false);
    assert.equal(partial.size, 1);
    assert.ok(partial.has('act-held'));
});

test('computeHydratedDirectActivities: correctly handles chat_id=0 without coercing to 1', () => {
    const turns = [
        { activity_id: 'act-sys-0', chat_id: 0, kind: 'direct_chat', phase: 'thinking' },
        { activity_id: 'act-main-1', chat_id: 1, kind: 'direct_chat', phase: 'thinking' },
    ];

    const mapChat0 = computeHydratedDirectActivities(new Map(), turns, 0);
    assert.equal(mapChat0.size, 1);
    assert.ok(mapChat0.has('act-sys-0'));

    const mapChat1 = computeHydratedDirectActivities(new Map(), turns, 1);
    assert.equal(mapChat1.size, 1);
    assert.ok(mapChat1.has('act-main-1'));
});

test('computeHydratedDirectActivities: a complete census deletes kind-less entries too', () => {
    const initialMap = new Map([
        // Kind-less entries were exempt while typing frames wrote this map. The
        // census is now its only writer, so its complete listing ends them too.
        ['managed-1', { activityId: 'managed-1', kind: '', phase: 'thinking' }],
        ['direct-1', { activityId: 'direct-1', kind: 'direct_chat', phase: 'thinking' }],
        ['direct-2', { activityId: 'direct-2', kind: 'direct_chat', phase: 'thinking' }],
    ]);

    assert.equal(computeHydratedDirectActivities(initialMap, [], 1).size, 0);
});

test('computeHydratedDirectActivities: a concluded turn is never resurrected by a snapshot captured while it still ran', () => {
    const concluded = new Map([['act-done', 12345]]);
    // Snapshot was requested mid-turn, its response arrived AFTER the turn's
    // keyed final concluded the activity client-side (project panels hydrate
    // once at creation, so without the ledger this insert would be permanent).
    const turns = [
        { activity_id: 'act-done', chat_id: 1, kind: 'direct_chat', phase: 'thinking' },
        { activity_id: 'act-live', chat_id: 1, kind: 'direct_chat', phase: 'thinking' },
    ];
    const updated = computeHydratedDirectActivities(new Map(), turns, 1, concluded);
    assert.equal(updated.size, 1);
    assert.ok(!updated.has('act-done'));
    assert.ok(updated.has('act-live'));
});

test('the header order: offline > the room\'s work > the owner\'s send > Online', () => {
    assert.equal(roomStatusView(inRoom('working'), { connected: false, sending: true }).text, 'Reconnecting…');
    assert.equal(roomStatusView(inRoom('working'), { sending: true }).text, 'Working');
    assert.equal(roomStatusView(inRoom('queued'), { sending: true }).text, 'Sending');
    assert.equal(roomStatusView(emptyWorkState(), { sending: true }).text, 'Sending');
    assert.equal(roomStatusView(emptyWorkState()).text, 'Online');
});

test('a paused room is not Working or Queued (#322), and its reason stays on its row', () => {
    // A budget pause, the owner's Pause and a Restart hold share one word in the header; the row says who or what.
    assert.deepEqual(roomStatusView(inRoom('paused')),
        { kind: 'online', text: 'Paused', tone: 'held', mark: 'pause', owner: false, amber: '' });
    assert.equal(roomStatusView(inRoom('queued')).text, 'Queued');
    assert.equal(roomStatusView(inRoom('working')).text, 'Working');
});

test('sequenced failures stop stale replies without replacing a newer accepted census', () => {
    const events = [];
    const sequencer = createStateSnapshotSequencer(
        (data) => events.push(data), () => 123, () => events.push('unavailable'));
    const older = sequencer.begin();
    const failed = sequencer.begin();
    assert.equal(sequencer.fail(failed), true);
    assert.equal(sequencer.apply(older, 'stale complete empty'), false);
    assert.equal(sequencer.fail(failed), false);
    const newer = sequencer.begin();
    assert.equal(sequencer.apply(newer, 'fresh'), true);
    assert.equal(sequencer.fail(older), false);
    assert.deepEqual(events, ['unavailable', 'fresh']);
});

test('Batch4: a paused or pausing room never drives a working header; a card never drives it at all', () => {
    assert.equal(roomStatusView(inRoom('working', { transition: 'pausing' })).text, 'Pausing…');
    const saved = roomStatusView(inRoom('paused'));
    assert.deepEqual([saved.text, saved.mark], ['Paused', 'pause'], 'a paused room stands: no moving mark, no typing row');
    // Resumed: the server's room says working again.
    assert.equal(roomStatusView(inRoom('working')).text, 'Working');
    // The producer reads no card: the same room with any card set says the same.
    assert.equal(roomStatusView({ ...inRoom('paused'), works: [{ activity: 'working', confirmed: true }] }).text, 'Paused');
});
