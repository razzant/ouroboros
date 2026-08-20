// Behavioural characterization of the live-card store, exercised where the
// code now lives. The store is driven exactly like chat.js drives it: records
// are minted through the getters, live frames flow through queue/apply, and
// terminal transitions land through finishLiveCard — with the collaborator
// factories replaced by observable stubs bound through
// bindLiveCardCollaborators.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatLiveCards } from '../modules/chat_live_cards.js';

function makeElement(tag = 'div') {
    const el = {
        tagName: tag.toUpperCase(),
        className: '',
        textContent: '',
        innerHTML: '',
        hidden: false,
        dataset: {},
        attributes: {},
        children: [],
        listeners: {},
        classNames: new Set(),
        removed: false,
        parentNode: null,
        isConnected: true,
        _selectorCache: new Map(),
        appendChild(child) { el.children.push(child); child.parentNode = el; return child; },
        remove() { el.removed = true; el.parentNode = null; },
        closest() { return null; },
        setAttribute(name, value) { el.attributes[name] = String(value); },
        removeAttribute(name) { delete el.attributes[name]; },
        focus() {},
        addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
        querySelector(selector) {
            if (!el._selectorCache.has(selector)) el._selectorCache.set(selector, makeElement('div'));
            return el._selectorCache.get(selector);
        },
        classList: {
            add(name) { el.classNames.add(name); },
            remove(name) { el.classNames.delete(name); },
            toggle(name, force) {
                const on = force === undefined ? !el.classNames.has(name) : Boolean(force);
                if (on) el.classNames.add(name); else el.classNames.delete(name);
                return on;
            },
            contains(name) { return el.classNames.has(name); },
        },
    };
    return el;
}

function liveCards({ isMain = true } = {}) {
    const priorDocument = globalThis.document;
    const priorWindow = globalThis.window;
    globalThis.document = {
        createElement: (tag) => makeElement(tag),
        hidden: false,
    };
    globalThis.window = { __ouroTaskBindings: {} };

    const liveCardRecords = new Map();
    const taskUiStates = new Map();
    const retiredTaskIds = new Set();
    const stickyExpandedSlots = new Set();
    const pendingSuggestedNames = new Map();
    const ephemeralDecisionTaskIds = new Set();
    const cancelableTaskIds = new Set();
    const subagentChildParents = new Map();
    const calls = [];
    const spy = (name, result) => (...args) => { calls.push({ name, args }); return result; };

    const api = createChatLiveCards({
        liveCardRecords,
        taskUiStates,
        retiredTaskIds,
        stickyExpandedSlots,
        pendingSuggestedNames,
        ephemeralDecisionTaskIds,
        cancelableTaskIds,
        subagentChildParents,
        isMain,
        withStableViewport: (mutate) => mutate(),
        insertMessageNode: spy('insertMessageNode'),
        stampNodeTimestamp: () => false,
        hideTypingIndicatorOnly: spy('hideTypingIndicatorOnly'),
        syncChatStatus: spy('syncChatStatus'),
        scheduleHistorySync: spy('scheduleHistorySync'),
        hasActiveLiveCard: () => false,
        getRebuildBatch: () => null,
    });

    api.bindLiveCardCollaborators({
        isBackgroundTaskId: () => false,
        shouldAlwaysShowTaskCard: () => false,
        getTaskUiState: (taskId, create) => {
            if (!taskUiStates.has(taskId) && !create) return null;
            if (!taskUiStates.has(taskId)) {
                taskUiStates.set(taskId, {
                    taskId, cardVisible: false, completed: false, completedPhase: '',
                    bufferedLiveUpdates: [], toolCalls: 0, forceCard: false,
                });
            }
            return taskUiStates.get(taskId);
        },
        bufferLiveUpdate: (taskState, summary, ts, dedupeKey, rawTs) => {
            calls.push({ name: 'bufferLiveUpdate', args: [taskState.taskId, summary] });
            taskState.bufferedLiveUpdates.push({ summary, ts, dedupeKey, rawTs });
        },
        markTaskComplete: spy('markTaskComplete'),
        turnTaskIntoProject: spy('turnTaskIntoProject'),
        syncCancelRunButton: spy('syncCancelRunButton'),
        renderCollapsedActivity: spy('renderCollapsedActivity'),
        ensureSubagentContainer: spy('ensureSubagentContainer', makeElement('div')),
        setLiveCardTypingVisible: spy('setLiveCardTypingVisible'),
        formatLiveCardPhaseLabel: (phase) => phase,
        setLiveCardExpanded: spy('setLiveCardExpanded'),
        syncLiveCardToggle: spy('syncLiveCardToggle'),
        directSubagentCount: () => 0,
        renderLiveCardTimeline: spy('renderLiveCardTimeline'),
        appendTimelineItem: spy('appendTimelineItem'),
        patchLastTimelineItem: spy('patchLastTimelineItem'),
        patchTimelineItemAt: spy('patchTimelineItemAt'),
        renderLiveCardMeta: spy('renderLiveCardMeta'),
    });

    return {
        ...api,
        liveCardRecords,
        taskUiStates,
        pendingSuggestedNames,
        ephemeralDecisionTaskIds,
        cancelableTaskIds,
        subagentChildParents,
        calls,
        named: (name) => calls.filter((call) => call.name === name),
        restore() { globalThis.document = priorDocument; globalThis.window = priorWindow; },
    };
}

test('getLiveCardRecord mints once and reuses the record afterwards', () => {
    const s = liveCards();
    const record = s.getLiveCardRecord('task-1');
    assert.equal(record.groupId, 'task-1');
    assert.equal(record.root.dataset.taskId, 'task-1');
    assert.equal(record.root.dataset.finished, '0');
    assert.equal(s.getLiveCardRecord('task-1'), record, 'the record is reused, not re-minted');
    assert.equal(s.liveCardRecords.size, 1);
    // A minted card syncs its cancel trigger against the durable marker set.
    assert.ok(s.named('syncCancelRunButton').length >= 1);
    s.restore();
});

test('a name buffered before the card existed lands as its title', () => {
    const s = liveCards();
    s.pendingSuggestedNames.set('task-n', 'Rename the moon');
    const record = s.getLiveCardRecord('task-n');
    assert.equal(record.suggestedName, 'Rename the moon');
    assert.equal(s.pendingSuggestedNames.size, 0, 'the buffer entry is consumed');
    // The coined name takes the title slot on the next live frame; the
    // activity headline keeps rendering in the timeline below it.
    s.applyLiveCardState({ phase: 'working', headline: 'Step one', human: true }, 'task-n', '10:00', 'k1');
    assert.equal(record.titleEl.textContent, 'Rename the moon');
    s.restore();
});

test('the apply pipeline appends, coalesces and dedupes timeline lines', () => {
    const s = liveCards();
    s.applyLiveCardState({ phase: 'working', headline: 'Reading', human: true }, 'task-a', '10:00', 'k1');
    const record = s.liveCardRecords.get('task-a');
    assert.equal(record.items.length, 1);
    assert.equal(s.named('appendTimelineItem').length, 1);
    // A consecutive duplicate coalesces into the same line's count.
    s.applyLiveCardState({ phase: 'working', headline: 'Reading', human: true }, 'task-a', '10:01', 'k1');
    assert.equal(record.items.length, 1);
    assert.equal(record.items[0].count, 2);
    assert.equal(s.named('patchLastTimelineItem').length, 1);
    // A new key appends; re-feeding the OLD key later is a silent skip
    // (the unbounded-Notes regression stays dead).
    s.applyLiveCardState({ phase: 'working', headline: 'Writing', human: true }, 'task-a', '10:02', 'k2');
    s.applyLiveCardState({ phase: 'working', headline: 'Reading', human: true }, 'task-a', '10:03', 'k1');
    assert.equal(record.items.length, 2);
    assert.equal(record.items[0].count, 2, 'the historical line is not re-counted');
    s.restore();
});

test('a terminal frame finishes the card and schedules exactly one resync', () => {
    const s = liveCards();
    s.applyLiveCardState({ phase: 'working', headline: 'Working', human: true }, 'task-t', '10:00', 'k1');
    s.applyLiveCardState({ phase: 'done', headline: 'Done', terminal: true, promote: true }, 'task-t', '10:01', 'k2');
    const record = s.liveCardRecords.get('task-t');
    assert.equal(record.finished, true);
    assert.equal(record.root.dataset.finished, '1');
    assert.equal(s.named('markTaskComplete').length, 1);
    assert.equal(s.named('scheduleHistorySync').length, 1);
    // Late non-terminal frames on the finished card are ignored.
    const settledLines = record.items.length;
    s.applyLiveCardState({ phase: 'working', headline: 'Zombie', human: true }, 'task-t', '10:02', 'k3');
    assert.equal(record.items.length, settledLines, 'the finished card takes no new lines');
    assert.equal(record.phaseEl.dataset.phase, 'done', 'the terminal phase survives the zombie frame');
    s.restore();
});

test('finishLiveCard maps phases honestly and drops the cancelable marker', () => {
    const s = liveCards();
    s.getLiveCardRecord('task-c');
    s.cancelableTaskIds.add('task-c');
    s.setActiveLiveGroupId('task-c');
    s.finishLiveCard('task-c', 'cancelled');
    const record = s.liveCardRecords.get('task-c');
    assert.equal(record.phaseEl.dataset.phase, 'cancelled');
    assert.equal(s.cancelableTaskIds.has('task-c'), false);
    assert.equal(s.getActiveLiveGroupId(), '', 'the active group is released');
    // An unknown phase lands as the generic done.
    s.getLiveCardRecord('task-d');
    s.finishLiveCard('task-d', 'mystery');
    assert.equal(s.liveCardRecords.get('task-d').phaseEl.dataset.phase, 'done');
    s.restore();
});

test('a converted project chip ignores every further frame', () => {
    const s = liveCards();
    const record = s.getLiveCardRecord('task-p');
    record.root.dataset.projectCreated = '1';
    s.applyLiveCardState({ phase: 'done', headline: 'Late', terminal: true }, 'task-p', '10:00', 'k');
    s.finishLiveCard('task-p', 'done');
    assert.equal(record.finished, false, 'the chip is terminal already; frames cannot touch it');
    s.restore();
});

test('queueTaskLiveUpdate buffers until the card earns visibility', () => {
    const s = liveCards();
    s.queueTaskLiveUpdate({ phase: 'working', headline: 'Quiet step', human: true }, 'task-q', '10:00', 'kq');
    assert.equal(s.named('bufferLiveUpdate').length, 1);
    assert.equal(s.liveCardRecords.has('task-q'), false, 'no card for a quiet task');
    // An error frame forces the card into existence with the buffer replayed.
    s.queueTaskLiveUpdate({ phase: 'error', headline: 'Boom', human: true }, 'task-q', '10:01', 'ke');
    assert.equal(s.liveCardRecords.has('task-q'), true);
    const record = s.liveCardRecords.get('task-q');
    assert.ok(record.items.length >= 1, 'buffered updates replay into the revealed card');
    s.restore();
});

test('an ephemeral decision frame suppresses and retires its transient card', () => {
    const s = liveCards();
    s.getLiveCardRecord('task-e');
    s.setActiveLiveGroupId('task-e');
    const handled = s.registerEphemeralDecisionFrame({ task_id: 'task-e', ephemeral_decision: true });
    assert.equal(handled, true);
    assert.equal(s.ephemeralDecisionTaskIds.has('task-e'), true);
    assert.equal(s.liveCardRecords.has('task-e'), false, 'the transient card is removed');
    assert.equal(s.getActiveLiveGroupId(), '');
    // A frame without the marker reports the registry verdict without changes.
    assert.equal(s.registerEphemeralDecisionFrame({ task_id: 'task-x' }), false);
    s.restore();
});

test('getSubagentCardRecord adopts the child under its parent container', () => {
    const s = liveCards();
    const child = s.getSubagentCardRecord('child-1', 'parent-1', 'researcher');
    assert.equal(child.isSubagent, true);
    assert.equal(child.parentGroupId, 'parent-1');
    assert.equal(child.root.dataset.subagent, '1');
    assert.equal(child.root.dataset.subagentRole, 'researcher');
    assert.ok(s.named('ensureSubagentContainer').length >= 1);
    // Missing lineage refuses adoption instead of minting an orphan.
    assert.equal(s.getSubagentCardRecord('child-2', ''), null);
    s.restore();
});

test('the active-group and attention accessors round-trip for the wiring', () => {
    const s = liveCards();
    s.setActiveLiveGroupId('task-live');
    const record = s.getLiveCardRecord('');
    assert.equal(record.groupId, 'task-live', 'an empty group id resolves to the active group');
    assert.equal(s.getLastTerminalAttention(), false);
    s.setLastTerminalAttention(true);
    assert.equal(s.getLastTerminalAttention(), true);
    s.restore();
});
