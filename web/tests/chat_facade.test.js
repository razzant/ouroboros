// W1 characterization: chat.js stopped implementing its pure helpers and now
// re-exports them from their owners. The public facade must therefore stay
// value-identical — not merely "still defined" — so every existing importer of
// chat.js keeps the exact same binding it had before the extraction.

import assert from 'node:assert/strict';
import test from 'node:test';

import * as cardState from '../modules/chat_card_state.js';
import * as controls from '../modules/chat_controls.js';
import * as facade from '../modules/chat.js';
import * as costs from '../modules/costs.js';
import * as renderBatch from '../modules/chat_render_batch.js';
import * as taskControlMenu from '../modules/task_control_menu.js';
import * as utils from '../modules/utils.js';

const DIRECT_OWNERS = {
    liveLineRowToggleKey: cardState.liveLineRowToggleKey,
    rawTimestampEpoch: utils.rawTimestampEpoch,
    insertTimelineNode: renderBatch.insertTimelineNode,
    headerBudgetPresentation: costs.headerBudgetPresentation,
    taskCostMeta: costs.taskCostMeta,
    taskCostProjection: costs.taskCostProjection,
    mergeStickyCostMeta: costs.mergeStickyCostMeta,
    clearStickyCardState: cardState.clearStickyCardState,
    COLLAPSED_ACTIVITY_MAX: cardState.COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview: cardState.boundActivityPreview,
    projectCollapsedActivity: cardState.projectCollapsedActivity,
    shouldFirePanic: controls.shouldFirePanic,
    confirmAndSendPanic: controls.confirmAndSendPanic,
    isTerminalTaskPhase: cardState.isTerminalTaskPhase,
};

function assertChatFacadeOwnerIdentity() {
    assert.equal(Object.keys(DIRECT_OWNERS).length, 14);
    for (const [name, ownerValue] of Object.entries(DIRECT_OWNERS)) {
        assert.notEqual(ownerValue, undefined, `${name} must be exported by its owner`);
        assert.equal(facade[name], ownerValue, `${name} must preserve owner identity`);
    }
}

// REUSABLE_TASK_IDS and cancelRunEligibility have exactly ONE owner:
// task_control_menu.js, the shared stop/hurry control. chat.js consumes them
// and must NOT mint a competing facade or a second Set — a duplicated reusable
// slot list would let the chat card and the control disagree about which cards
// may be stopped.
function assertSingleTaskControlOwner() {
    assert.equal(typeof taskControlMenu.cancelRunEligibility, 'function');
    assert.ok(taskControlMenu.REUSABLE_TASK_IDS instanceof Set);
    for (const name of ['REUSABLE_TASK_IDS', 'cancelRunEligibility']) {
        assert.equal(cardState[name], undefined, `${name} must not be re-owned by chat_card_state.js`);
        assert.equal(facade[name], undefined, `${name} must not be re-exported by the chat facade`);
    }
}

// The reusable-slot list is a MUTABLE singleton, not a frozen literal: a later
// slot registered on the owner has to be visible to every consumer through the
// same Set instance.
function assertReusableTaskSingletonIdentity() {
    const marker = 'facade-identity-test';
    const slots = taskControlMenu.REUSABLE_TASK_IDS;
    slots.add(marker);
    try {
        assert.equal(taskControlMenu.cancelRunEligibility({
            groupId: marker, cancelable: true,
        }), false);
    } finally {
        slots.delete(marker);
    }
    assert.equal(taskControlMenu.cancelRunEligibility({ groupId: marker, cancelable: true }), true);
}

test('chat facade re-exports all W1 owner bindings by identity', () => {
    assertChatFacadeOwnerIdentity();
});

test('reusable-slot identity and cancel eligibility keep a single owner', () => {
    assertSingleTaskControlOwner();
});

test('the reusable-task singleton stays mutable and shared', () => {
    assertReusableTaskSingletonIdentity();
});
