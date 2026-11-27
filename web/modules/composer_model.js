/**
 * The Model capsule (docs/DESIGN.md "Composer owner controls"): the first capsule of the composer's row, before
 * Subagents, Context and Effort, and one of their family closed. It holds the model the next work in its room starts
 * on and keeps it until its owner changes it; the Main chat's own is also what a new room copies. Until a room
 * chooses it follows Main and says so: the name word is `Main`, the value Main's model. Closed, the capsule shows the
 * room's model and its non-Standard tier, never the model of a running work. No pick here writes Main the role.
 *
 * A press opens the Model sheet (model_sheet.js; never hover): search, the pinned and recent models, all sources, and
 * at its foot the room's processing tier. The sheet lists only the sources that can lead a work, one row per
 * (source, model) (chat_model_routes.js); the catalog suggests and never admits, so any id can still be typed. A row
 * is always that route: the one current now changes nothing, and none follows Main again. A pick saves at once
 * through `POST /api/owner/chat-model`, its tier through `POST /api/owner/chat-processing`; Send waits for both. The
 * strip offers what the scope's own route says where it is bound (processing_choice.js): what follows Main runs on
 * Main's account, its pin or Auto, which the settings say (until they do nothing of an account is confirmed: an
 * unread pin is never read as Auto); a room's own model on Auto across its source's accounts.
 *
 * A pick that leaves the room on another model than it was arms this window: its next message is sent as new work on
 * that model (`newWork`), whatever work already runs here. The frame names only the saved choice's token; the host
 * assigns the model and tier from what is saved and says on the echo whether the message was taken so.
 */
import { apiClient } from './api_client.js';
import { fmt, tr } from './i18n.js';
import { haptic } from './haptics.js';
import { isCoarsePointer, isStackLayout } from './layout.js';
import { modelCatalog } from './model_catalog_store.js';
import { openModelSheet } from './model_sheet.js';
import { sourceMark } from './model_source_mark.js';
import { confirmChatUltrafast, resolvedTier, routeTiers, tierNamed, tierWord } from './processing_choice.js';
import { modalNotice } from './toast.js';
import { bindPopoverPosition } from './ui_interactions.js';
import {
    MAIN_CHOICE, describeRoute, normalizeChoice, normalizeRoute, routeSource, sameRoute, sourceUnlisted,
} from './chat_model_routes.js';

/** Whether two saved choices are one: both follow Main, or both name the same route. */
export const sameChoice = (one, other) => one?.kind === other?.kind
    && (one?.kind === 'main' || sameRoute(normalizeRoute(one), normalizeRoute(other)));

/** Said instead of saving when this window is newer than the server it reached: nothing is sent there. */
export const SERVER_OLDER = 'Ouroboros is running an older version than this window, so nothing was changed here. Finish the update or restart Ouroboros, then reload.';
export const CHAT_NOT_READ = 'This chat’s model is not read from Ouroboros yet. Try again in a moment.';

/**
 * The capsule in its room's composer row. `save(body)` is the owner write (`apiClient.ownerChatModel`),
 * `saveTier(body)` the room's own processing tier (`apiClient.ownerChatProcessing`), `openSettings()`
 * Settings › Accounts and keys, `askTier(facts)` the question before the Main chat takes Ultrafast
 * (processing_choice.js), `onNewWork()` follows every change of what the next message would be sent as, `store` the
 * window's catalog store, `place(sheet, anchor)` the sheet's place above the row.
 * Returns `{ el, capsule, syncState(data), hasPendingSave, pendingSave, pick, chooseTier, newWork, open, destroy }`.
 */
export function createModelControl({
    el, row, projectId = '', chatId = projectId ? null : 1, save = apiClient.ownerChatModel,
    saveTier = apiClient.ownerChatProcessing, openSettings = () => {},
    showToast = () => {}, refreshState = () => {}, onLayout, onSettled, onNewWork = () => {},
    askTier = confirmChatUltrafast, store = modelCatalog,
    isStack = () => isStackLayout(), isCoarse = () => isCoarsePointer(),
    // Anchored to the row, as the hint is to its capsule: the composer under it never moves.
    place = (sheet) => bindPopoverPosition(sheet, { anchor: row, align: 'start' }),
    win = globalThis.window, doc = globalThis.document,
}) {
    const isMain = !projectId;
    el.classList.add('chat-capsule');
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', 'Model');
    el.dataset.known = 'false';
    // The head's words are written by render(): the mark, whose model this is, the model, and its tier when not Standard.
    el.innerHTML = '<button type="button" class="chat-capsule-head ui-hit-area chat-model-head" aria-expanded="false" aria-haspopup="dialog"><span class="chat-capsule-glyph" aria-hidden="true"></span><span class="chat-capsule-name"></span><span class="chat-capsule-value" data-i18n-skip></span><span class="chat-capsule-tier" data-i18n-skip hidden></span></button>';
    const head = el.querySelector('.chat-model-head');
    const glyph = el.querySelector('.chat-capsule-glyph');
    const name = el.querySelector('.chat-capsule-name');
    const value = el.querySelector('.chat-capsule-value');
    const tierWordEl = el.querySelector('.chat-capsule-tier');

    // `main` is what Main the role is now (the state body's `main_route`), `choice` the room's own model for new work
    // (`projects[].model_choice`; the Main chat's is `main_chat.model_choice`) and `tier` its own processing tier
    // (`processing_choice`, '' = Default).
    // `withheld`: the Main chat's saved Ultrafast names another model or account than its new work runs on now
    // (`main_chat.processing_withheld`), so its tier reads Default and the strip says why. `keepsChat`: whether the running server keeps the Main chat's own model
    // and tier apart from Main the role, which its state body says by carrying `main_chat` (null until one is read).
    const state = { main: null, choice: { ...MAIN_CHOICE }, tier: '', withheld: false, keepsChat: null,
        saving: null, tierSaving: null, lastOk: true, destroyed: false };
    // The scope each save names: a room by its id, the Main chat by `scope` (a body with neither is an older page's
    // write of Main the role, which the host refuses).
    const scoped = (body) => (projectId ? { ...body, project_id: projectId } : { ...body, scope: 'main_chat' });
    // A window newer than its server (an update not finished, or rolled back) never sends the Main chat's choice
    // there: an older server reads a body without a room as Main the role's own and would change that instead.
    // `say` is the press's own notice (a tier's, pressed in the sheet it leaves open); a pick's sheet has closed.
    const serverOlder = (say = showToast) => {
        if (!isMain || state.keepsChat === true) return false;
        say(state.keepsChat === false ? tr('chat.model_server_older', SERVER_OLDER) : tr('chat.model_not_read', CHAT_NOT_READ), 'info');
        return true;
    };
    // This window's pick left the room on another model than it was: `token` names the saved choice, `choice` is it,
    // `baseline` what the room had before. Null: an ordinary message. `flight` is that record once a message went
    // out naming it (`id`), kept until the host says how it took the message; `freed` when the owner gave the
    // message's unconfirmed copy up first. A pick made while it waits is another choice with its own record.
    let armed = null, flight = null;
    let sheet = null, markKey = null, nameWords = null;
    let detailAsked = null;  // the routed value whose own endpoints the open sheet asked for

    const follows = () => state.choice.kind === 'main';
    const shown = () => (follows() ? state.main : normalizeRoute(state.choice));
    const tiers = () => store.tiers?.() || null;
    // The tier this room's new work takes: its own, else Main's for what follows Main, else the default one.
    const resolved = () => resolvedTier({ choice: state.tier, follows: follows(), tiers: tiers() });
    // The tier a pick would run on here, for the catalog's price beside it (a catalog price is Standard's). The row
    // current now changes nothing when pressed, so it keeps the tier the room's work runs on: Main's while it follows.
    const tierOf = (route, { main = false, current = false } = {}) => (current ? resolved()
        : state.tier || resolvedTier({ follows: main, tiers: tiers() }));
    const describe = (route) => (route ? describeRoute(store.index(), route, { tier: resolved() }) : null);
    // The account `route` runs on here: Main's own for what follows Main (its pin, '' for Auto, null until the
    // settings say which: then no account's word is taken), Auto for a room's own model (CLIENT_CONTRACT: a carried
    // model has no pinned account).
    const accountOf = (route) => (follows() ? store.mainAccount?.(route) ?? null : '');
    const tiersOf = (route, account = accountOf(route)) => routeTiers({ index: store.index(), route, account,
        detail: store.detail?.(route) || null, sourceLabel: describe(route)?.sourceLabel });
    // The open sheet asks once for its route's own endpoints, and again when the route under it changes.
    function syncDetail() {
        const route = shown();
        // Asked again when the held answers were dropped (a connection changed under the same routed value).
        const asking = `${route?.model}|${store.detailEpoch?.() ?? ''}`;
        if (!sheet || !route || asking === detailAsked) return;
        detailAsked = asking;
        void store.readDetail?.(route);
    }
    function arm(next) {
        armed = next;
        if (!state.destroyed) onNewWork();
    }
    // The message in flight landed: `spent` (new work started on its choice), `waits` (the host gave it to a work, or
    // its owner gave its copy up: the choice waits for a message again) or `void` (refused: the choice is not the
    // room's any more). A pick made meanwhile is its own choice, never this message's: it only learns what the room
    // stood on before it, which was read as this message's choice while it flew.
    function land(how) {
        const { token, choice, baseline } = flight;
        const read = flight.freed ? baseline : choice, stands = how === 'spent' ? choice : baseline;
        if (how === 'waits') flight = { ...flight, freed: true };  // a copy given up may still prove taken
        else flight = null;
        if (!armed) return void (how === 'waits' && arm({ token, choice, baseline }));
        if (how === 'void' && armed.token === token) return void arm(null);
        if (!sameChoice(armed.baseline, read) || sameChoice(read, stands)) return;
        arm(sameChoice(armed.choice, stands) ? null : { ...armed, baseline: stands });
    }

    function render() {
        const said = describe(shown());
        el.dataset.known = String(Boolean(said));
        el.dataset.own = String(Boolean(said) && !follows());
        // The mark itself is the identity: what it is drawn from (a subscription's descriptor) can arrive
        // after the words it stands beside, and a key of hand-picked fields would hide that change.
        const mark = sourceMark(said);
        if (mark !== markKey) { markKey = mark; glyph.innerHTML = mark; }
        // Whose model this is: Main's, or its own source's.
        const sourceNamed = Boolean(said) && !follows();
        const words = sourceNamed ? said.sourceLabel : 'Main';
        if (words !== nameWords) {  // the overlay may have translated the word it was given
            nameWords = words;
            if (sourceNamed) name.dataset.i18nSkip = ''; else delete name.dataset.i18nSkip;  // a source's name is not chrome
            name.textContent = words;
        }
        const model = said ? said.short : '—';
        if (value.textContent !== model) value.textContent = model;
        // A tier other than Standard is said beside the model, closed too (owner 12.11: the word only when not Standard).
        const tier = resolved();
        const tierText = tierNamed(tier) ? tierWord(tier) : '';
        if (tierWordEl.textContent !== tierText) tierWordEl.textContent = tierText;
        tierWordEl.hidden = !tierText;
        el.dataset.tier = tier || 'default';
        const facts = said ? { model: `${said.short}${tierText ? `, ${tierText}` : ''}`, source: said.sourceLabel } : null;
        head.setAttribute('aria-label', !facts ? 'Model for new work: not read yet'
            : isMain ? fmt(follows() ? 'Model for new work here and in new rooms: {model}, {source}. Set in Settings › Models.'
                : 'Model for new work here and in new rooms: {model}, {source}.', facts)
                : fmt(follows() ? 'Model for new work here: {model}, {source}. Set in Settings › Models.'
                    : 'Model for new work here: {model}, {source}.', facts));
        // The settings document, read only when it changes a word: why a saved API source this catalog did not list
        // cannot send.
        if (said && store.status().read && !store.status().failed && sourceUnlisted(store.index(), shown())) void store.readConnections();
        syncDetail();
        sheet?.paint();
    }

    /* --- the sheet's words: what a pick leaves behind, what the tier strip applies to --- */
    // Said above the rows while the room still follows Main, before any pick. It names where that model is set
    // (Settings › Models: "Main" alone reads as the Main chat's own capsule) and Main's pinned account, then what a
    // pick does: new tasks take it as every room's own model runs (Auto across its source's accounts, the model's own
    // window, never Main's binding), and running ones keep theirs.
    function followNote() {
        if (!follows() || !state.main) return '';
        const pin = accountOf(state.main);
        const picks = tr('model.note.own_route', 'New tasks here take the model you pick, on Auto across its source’s accounts and with its own context window; running tasks keep theirs.');
        return `${pin ? fmt('This model is set in Settings › Models, on its pinned account {account}.', { account: pin }) : tr('model.note.follows', 'This model is set in Settings › Models.')} ${picks}`;
    }
    function scopeWords() {
        const tier = resolved();
        const state_ = state.tier || !tier ? tierWord(tier)
            : fmt(follows() ? 'Default · {tier} as Main' : 'Default · {tier}', { tier: tierWord(tier) });
        // The tier a work runs on, its own or Default's (Main's included), reaches its new subagents without one
        // (owner 13.11).
        const scope = !isMain ? tr('processing.scope.room_helpers', 'for new work here and its new subagents without a tier of their own; running work keeps its own')
            // The Main chat's Ultrafast stands on the one route it was asked for, and no room copies it. On a
            // work carries it from the first call; API work must still use the same connection.
            : state.withheld ? tr('processing.scope.chat_ultrafast_withheld', 'The saved Ultrafast no longer matches the destination of new work here. New work takes Default until you confirm again.')
            : state.tier === 'ultrafast' && profiled() ? tr('processing.scope.chat_ultrafast_profile', 'for new work here, on its own calls to this model and profile from the first one; no new room copies it, running work keeps its own')
            : state.tier === 'ultrafast' ? tr('processing.scope.chat_ultrafast', 'for new work here, on calls to this model through the confirmed destination; no new room copies it, running work keeps its own')
                : tr('processing.scope.chat_helpers', 'for new work here and its new subagents without a tier of their own; a new room copies it, running work keeps its own');
        return `${state_} · ${scope}`;
    }
    // Whether new work here runs pinned to a named subscription profile (never while its pin is Auto or not read).
    const profiled = () => { const route = shown(); return Boolean(route && routeSource(route).startsWith('subscription:') && accountOf(route)); };
    const tierView = () => ({ choice: state.tier, resolved: resolved(), busy: Boolean(state.tierSaving),
        scope: scopeWords(), ...tiersOf(shown()) });
    // The Main chat's Ultrafast asks first: for the route the sheet shows, with what the tier follows there, what
    // that route says of it and what it is listed at where that is told. The answer is what was asked about, by
    // name: the model, its locality, the subscription profile its new work is pinned to ('' on Auto and where a
    // route has none) and on an API route its connection's revision as read. Nothing identifies the account signed
    // in to a profile: the tier follows its name, and the question says so; Auto is said as its
    // source's pool. The host saves the tier for that route: on another model, profile or connection new work is
    // Default again. A route that moved under the question, or a pin or connection not read yet, was not asked
    // about: nothing is written.
    // `say` is the press's own notice (toast.js `modalNotice`): the sheet's while it stays open, the page's once it has closed.
    async function confirmUltrafast(say) {
        const fromSheet = sheet;
        const route = shown();
        if (!route) return null;
        // This source's endpoints, never another's (the price below is theirs), and the settings that name Main's
        // account and each connection's revision.
        await Promise.all([store.readDetail?.(route), store.readConnections?.()]);
        if (state.destroyed || sheet !== fromSheet || !sameRoute(route, shown())) return null;
        const said = describe(route);
        const subscription = routeSource(route).startsWith('subscription:');
        // The profile's name where the route has accounts (null: its pin is not read), '' where it has none.
        const bound = () => ({ account: subscription ? accountOf(route) : '', revision: store.revisionOf?.(route) ?? '' });
        const first = bound();
        if (first.account === null || store.revisionOf?.(route) === null) {
            say('Main’s account or this connection is not read yet, so Ultrafast was not set. Try again.', 'info');
            return null;
        }
        const here = tiersOf(route);
        closeSheet({ focus: true });  // the question opens alone
        const yes = await askTier({ model: said?.short || '', source: [said?.sourceLabel, here.accountLabel].filter(Boolean).join(' · '),
            follows: first.account ? 'profile' : subscription ? 'auto' : first.revision ? 'api' : '', account: first.account,
            explain: here.explain('ultrafast'), priceSaid: Boolean(here.priceOf('ultrafast')), doc });
        if (!yes || state.destroyed) return null;
        const now = bound();
        if (sameRoute(route, shown()) && now.account === first.account && now.revision === first.revision) {
            return { model: route.model, use_local: Boolean(route.use_local), account: first.account,
                ...(first.revision ? { connection: first.revision } : {}) };
        }
        say('The model, profile or connection here changed while you were asked: Ultrafast was not set.', 'info');
        return null;
    }

    /* --- saving: a pick posts at once and answers for itself --- */
    function saveChoice(choice) {
        if (state.saving || state.destroyed) return state.saving;
        // What the room had before this window's picks; once another window moved its model, or a message went out
        // on the pick (the room is read as standing on it until the host says otherwise), what it has now.
        const baseline = armed && sameChoice(state.choice, armed.choice) ? armed.baseline : { ...state.choice };
        el.dataset.disabled = 'true';
        state.saving = (async () => {
            try {
                const reply = await save(scoped(choice));
                if (state.destroyed) return false;
                if ((reply?.project_id || '') !== projectId) throw new Error('The answer belongs to another room.');
                const saved = reply?.model_choice;
                if (reply?.ok !== true || (saved?.kind !== 'main' && !normalizeRoute(saved))) throw new Error('The server did not confirm the model.');
                state.choice = normalizeChoice(saved);
                state.lastOk = true;
                // Back on the model it was, the room is as it was: the next message is an ordinary one again.
                const token = String(reply.choice_token || '');
                arm(token && !sameChoice(state.choice, baseline) ? { token, choice: state.choice, baseline } : null);
            } catch (error) {
                if (state.destroyed) return false;
                state.lastOk = false;
                showToast(error?.message || 'Could not change the model.', 'error');
            } finally {
                state.saving = null;
                if (!state.destroyed) {
                    el.dataset.disabled = 'false';
                    render();
                    refreshState(true);
                    void store.readRecents();  // the server wrote this act into the recent models
                }
            }
            return state.lastOk;
        })();
        return state.saving;
    }
    /** One pick from the sheet. A row is always that route, Main's model now included; the row current now changes
     *  nothing, so the room keeps what it has (a room that follows Main keeps following it). */
    function pick(route) {
        if (state.saving || state.destroyed || !normalizeRoute(route)) return undefined;
        if (sameRoute(route, shown()) || serverOlder()) return undefined;
        haptic('tick');  // another model, chosen by a press (haptics.js)
        return saveChoice({ kind: 'model', model: route.model, use_local: route.use_local });
    }
    /** The room's own tier ('' = Default) for its new work. It moves nothing else and arms no new work. */
    function chooseTier(tier) {
        if (state.tierSaving || state.destroyed) return state.tierSaving;
        const say = modalNotice(sheet?.el, showToast);  // named at the press: the sheet it was pressed in, never a later one
        if (serverOlder(say)) return Promise.resolve(false);
        let settle = () => {};
        state.tierSaving = new Promise((resolve) => { settle = resolve; });
        sheet?.paint();  // the strip shows the save in flight before it is sent
        const saving = state.tierSaving;
        void (async () => {
            let ok = false;
            try {
                const asked = isMain && tier === 'ultrafast' ? await confirmUltrafast(say) : null;
                if (isMain && tier === 'ultrafast' && !asked) return;
                const reply = await saveTier(scoped({ processing: tier, ...(asked ? { confirmed: asked } : {}) }));
                if (state.destroyed) return;
                if ((reply?.project_id || '') !== projectId || reply?.ok !== true) throw new Error('The server did not confirm the tier.');
                state.tier = String(reply.processing_choice || '');
                state.withheld = false;  // the answer of a save is what stands now
                haptic('tick');
                ok = true;
            } catch (error) {
                if (!state.destroyed) say(error?.message || 'Could not change the processing tier.', 'error');
            } finally {
                state.tierSaving = null;
                if (!state.destroyed) { render(); refreshState(true); }
                settle(ok);
            }
        })();
        return saving;
    }
    async function setPinned(route, pinned) {
        const say = modalNotice(sheet?.el, showToast);
        try {
            await store.setPinned(route, pinned);
        } catch (error) {
            if (!state.destroyed) say(error?.message || 'Could not change the pinned models.', 'error');
        }
    }

    /* --- the sheet: opened by a press, never hover; one capsule or sheet open at a time --- */
    function openSheet() {
        if (sheet || state.destroyed) return;
        control.peers.forEach((peer) => peer.snapClosed());
        head.setAttribute('aria-expanded', 'true');
        el.dataset.sheetOpen = 'true';
        row.dataset.ownerSheet = 'true';
        sheet = openModelSheet({
            anchor: row, label: isMain ? 'Model for new work here and in new rooms' : 'Model for new work here', doc, place,
            stack: isStack, coarse: isCoarse, returnFocus: head,
            list: {
                store, main: () => state.main, current: shown, mainRow: false, tierOf, lead: followNote,
                // The room's model now, then the owner's recent ones: the row it stands on is always in view.
                recents: () => [shown(), ...store.recents()].filter(Boolean),
                onRoute: (route) => { closeSheet({ focus: true }); pick(route); },
                onPin: setPinned,
                onSettings: () => { closeSheet(); openSettings(); },
            },
            tier: { onChoose: chooseTier, view: tierView, mainRoute: () => (follows() ? shown() : null) },
            onClose: () => {
                sheet = null;
                head.setAttribute('aria-expanded', 'false');
                delete el.dataset.sheetOpen;
                delete row.dataset.ownerSheet;
                onLayout?.();
                onSettled?.();
            },
        });
        void store.read();
        void store.readRecents();
        detailAsked = null;
        render();  // the open sheet's own reads: its route's endpoints, and Main's account where it decides a tier
        onLayout?.();
    }
    function closeSheet({ focus = false } = {}) {
        sheet?.close({ focus });
    }
    const onHead = () => { if (sheet) closeSheet(); else openSheet(); };
    head.addEventListener('click', onHead);

    // The composer's row treats it as one of its capsules: it fits closed like them, and opening one closes the others.
    const control = {
        el, kind: 'model', hoverCapable: false, peers: [],
        isOpen: () => Boolean(sheet), isPinned: () => Boolean(sheet), isSettled: () => true,
        measureName() {}, sayHint() {}, relayout() { sheet?.reposition(); },
        setOpen(open) { if (open) openSheet(); else closeSheet(); },
        snapClosed() { closeSheet(); },
        destroy() {},
    };
    const unsubscribe = store.subscribe(() => { if (!state.destroyed) render(); });
    render();

    return {
        el,
        capsule: control,
        /** One `/api/state` body (or a room's own read): what Main is now and this room's own model and tier. A save
         *  in flight answers for itself; a body without a field keeps what was last read. */
        syncState(data) {
            if (state.destroyed || !data) return;
            if (data.processing_tiers) store.adoptTiers?.(data.processing_tiers);
            // A whole state body names Main's route; the server that also keeps the Main chat's own choice names it.
            if (isMain && data.main_route) {
                state.keepsChat = Boolean(data.main_chat);
                if (!state.keepsChat && armed) armed = null;  // an older server takes no new task: Send says so again
            }
            if (!state.saving) {
                const main = normalizeRoute(data.main_route);
                if (main) state.main = main;
                const room = projectId ? data.projects?.find?.((project) => project?.id === projectId) : data.main_chat;
                if (room) state.choice = normalizeChoice(room.model_choice);
                if (room && !state.tierSaving) {
                    state.tier = String(room.processing_choice || '');
                    state.withheld = !projectId && room.processing_withheld === 'ultrafast';
                }
            }
            render();
            onNewWork();  // another window's change of this room's model voids a pick made here (`newWork.armed`)
        },
        hasPendingSave: () => Boolean(state.saving || state.tierSaving),
        /** Resolves once the saves settled: true unless a model save in flight was refused (a refused tier keeps the
         *  old one and says so; an earlier refusal already said its own). */
        pendingSave: async () => {
            const model = state.saving;
            await state.tierSaving;
            return model ? model : true;
        },
        /** The capsule's one save path: a route. */
        pick,
        chooseTier,
        /** What this window's next message is sent as (composer.js). */
        newWork: {
            /** Null for an ordinary message; else the saved choice's `token` the frame names, the `model` in the
             *  owner's word and `changed`: the room's model is no longer the one picked here (another window's
             *  pick), so nothing may be sent on it. */
            armed: () => (armed ? { token: armed.token, model: describe(normalizeRoute(armed.choice))?.short || '',
                changed: !sameChoice(state.choice, armed.choice) } : null),
            /** The message `id` went out naming the choice `token`: no later message names it too, until the host
             *  says how it took that one. A pick made since (while the message's files were uploading) is another
             *  choice: it stays armed, and this message's word never touches it. */
            sent(id, token) {
                if (!id || !armed || armed.token !== token) return;
                flight = { ...armed, id: String(id), freed: false };
                arm(null);
            },
            /** The host's word on that message, by its live echo or by its saved row in history: taken as new work
             *  (`spent`), the room stands on the choice it named; taken otherwise (a Reply addressed to a work), the
             *  choice waits for a message again. A word on any other message changes nothing. */
            settled(id, spent) {
                if (flight?.id !== String(id || '')) return;
                if (spent) land('spent');
                else if (flight.freed) flight = null;  // already waiting again
                else { land('waits'); flight = null; }
            },
            /** The host refused that message because the choice it named is no longer the room's: nothing started
             *  on it, and it does not wait either. */
            refused(id) { if (flight?.id === String(id || '')) land('void'); },
            /** The owner gave up the unconfirmed copy of message `id` (Discard, or its words went back to the
             *  composer) before the host's word came: the choice it named waits for a message again, as if the
             *  message had not been taken; a later word that it was still settles it. */
            released(id) { if (flight?.id === String(id || '') && !flight.freed) land('waits'); },
            /** The pick made here no longer stands (said to the owner): the next message is an ordinary one. */
            drop() { if (armed) arm(null); },
        },
        open: openSheet,
        destroy() {
            state.destroyed = true;
            closeSheet();
            unsubscribe();
            head.removeEventListener('click', onHead);
        },
    };
}
