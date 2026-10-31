/** A room's bar, declared once (docs/DESIGN.md §10 "Page header"): what has a state, and the emergency exit. Main's bar
 * holds the room's status, the consciousness chip, Restart, Panic and the budget; a Project's bar the status, the chip
 * and Panic. There is no More and no ⋯: a command lives in the Shade of the state it changes (consciousness in the
 * chip's), and a narrow bar moves Restart and the budget into the room status's Shade
 * (page_header.js `bindHeaderControlPlacement`). Each control keeps one node and one handler wherever it stands. */
import { confirmAndSendRestart, headerBudgetPresentation, sendPanic } from './chat_activity.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { createConsciousnessStatus } from './consciousness_status.js';
import { bindHeaderControlPlacement } from './page_header.js';

/** Panic: immediate in every room's bar, never behind a menu or a save. */
export function panicButtonHtml() {
    return '<button class="chat-header-btn danger ui-chrome-control" type="button" data-chat-command="panic" title="Stop all workers">Panic</button>';
}

export function mainHeaderActionsHtml() {
    return `
                <div class="chat-header-actions" id="chat-header-actions">
                    <button class="chat-header-btn ui-chrome-control" type="button" data-chat-command="restart" title="Restart agent">Restart</button>
                    ${panicButtonHtml()}
                </div>
                <button class="chat-budget-pill ui-chrome-control" id="chat-budget-pill" type="button" title="Open budget controls" aria-label="Open budget controls">
                    <span class="chat-budget-word">Budget</span>
                    <span class="chat-budget-text" id="chat-budget-text">Loading…</span>
                    <div class="chat-budget-bar">
                        <div class="chat-budget-bar-fill" id="chat-budget-bar-fill"></div>
                    </div>
                </button>
                <span id="chat-status" class="status-badge offline">Connecting…</span>
            `;
}

/**
 * Wire one room's bar: Restart and Panic on their own buttons, the budget's door and its words, the consciousness chip,
 * and (Main, `shade` = the room status control of room_work.js) the narrow bar's placement of Restart and the budget.
 * `update(data)` takes every `/api/state` answer the room reads. Returns `{ update, destroy }`.
 */
export function createRoomHeader({ bar, ws, shade = null, openDashboardTab = null, refreshState = () => {} }) {
    if (!bar) return null;
    const actions = bar.querySelector('.chat-header-actions');
    const budget = bar.querySelector('.chat-budget-pill');
    const onCommand = async (event) => {
        const command = event.currentTarget?.dataset.chatCommand || event.target?.dataset?.chatCommand;
        if (command === 'restart') await confirmAndSendRestart({ openConfirmDialog, ws });
        else if (command === 'panic') sendPanic({ ws });
    };
    const commands = [...bar.querySelectorAll('[data-chat-command]')];
    commands.forEach((button) => button.addEventListener('click', onCommand));
    const onBudget = () => openDashboardTab?.('costs');
    budget?.addEventListener('click', onBudget);
    const disposePlacement = actions && shade ? bindHeaderControlPlacement({ headerActions: actions, budgetPill: budget, shade }) : () => {};
    const consciousness = createConsciousnessStatus({ bar, ws, openDashboardTab, refreshState });
    return {
        update(data) {
            consciousness?.update(data);
            if (!budget) return;
            const view = headerBudgetPresentation(data);
            const text = budget.querySelector('.chat-budget-text');
            if (text && text.textContent !== view.label) text.textContent = view.label;
            // The bar's fill is the one measured value here, written as its own custom property.
            budget.style.setProperty('--budget-fill', `${view.fillPct}%`);
            budget.dataset.attention = view.attention ? '1' : '0';
        },
        destroy() {
            commands.forEach((button) => button.removeEventListener('click', onCommand));
            budget?.removeEventListener('click', onBudget);
            disposePlacement();
            consciousness?.destroy();
        },
    };
}
