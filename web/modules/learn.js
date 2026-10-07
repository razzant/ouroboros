/** Post-setup field guide. All actions prepare a draft or open an existing page; none sends.
 *  Chrome is authored in English and translated by the install's i18n overlay; the overlay
 *  skips textareas and this module opts the feedback line out (data-i18n-skip), so template
 *  drafts and runtime feedback are translated here at the producer through tr(). The boot
 *  dictionary read can land after this page mounts, so the template textareas repaint on
 *  ouro:language-changed — leaving any textarea the owner already edited untouched. */
import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { tr } from './i18n.js';

const starters = [
    { title: 'Introduce yourself', text: () => tr('learn.template.introduce', 'My name is <name>. I work on <subject>. What matters to me is <what exactly>. Remember this, with my corrections if I change anything.') },
    { title: 'Review a file', text: () => tr('learn.template.file', 'Read the attached file in full and extract <what to look for>. Separate facts from assumptions and name what you could not verify.') },
    { title: 'Set a reminder', text: () => tr('learn.template.reminder', 'Remind me on <date and time> about <task>. Tell me whether the reminder was saved and what happens if the app is not running.') },
    { title: 'Change yourself carefully', text: () => tr('learn.template.change', 'I want to change <what exactly> in Ouroboros. First check the current code and boundaries, then propose a plan and tests. Keep the changes on a separate branch and do not publish anything without my decision.') },
    { title: 'Prepare an issue', text: () => tr('learn.template.issue', 'Help me draft an issue for <repository>: reproduction <steps>, expected and actual behavior <difference>. Check whether the issue already exists; show me the draft first.') },
    { title: 'Prepare a PR', text: () => tr('learn.template.pr', 'Help me prepare a PR for <repository> from a separate clean copy. Name the base SHA, affected contracts, checks and known limitations. Show the final diff and independent review results before publishing.') },
];

const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

export function appendToDraft(existing, suggestion) {
    const old = String(existing || '');
    return old ? `${old}\n\n${suggestion}` : suggestion;
}

/** Whether a template textarea may be repainted with a fresh translation: yes while it still
 *  holds exactly what we last wrote (or has never been painted); no once the owner's own edit
 *  made the draft theirs. Pure so the draft-safety contract is unit-tested. */
export function shouldRepaintTemplate(lastWritten, currentValue) {
    return lastWritten === undefined || currentValue === lastWritten;
}

export function initLearn({ showPage, openSettingsTab, openDashboardTab }) {
    const page = document.createElement('section');
    page.id = 'page-learn';
    page.className = 'page learn-page';
    page.innerHTML = `${renderPageHeader({ title: 'Getting started', icon: PAGE_ICONS.learn })}
        <div class="learn-content">
            <header class="learn-hero">
                <div class="learn-hero-copy">
                <p class="learn-eyebrow">Start with a conversation</p>
                <h2>Meet <span>Ouroboros</span></h2>
                <p>I work with tasks, files and tools, remember conversations and can return to work across them. What I can actually do depends on the selected model, access, budget and whether the app is running. This page is a map of first steps, not a promise that everything always works.</p>
                <div class="learn-hero-actions"><button type="button" data-open="chat" class="btn btn-primary">Open chat</button><button type="button" data-open="settings" class="btn btn-secondary">Check settings</button></div>
                </div>
                <div class="learn-orbit" aria-hidden="true"><span class="learn-orbit-core">∞</span><span class="learn-orbit-word learn-orbit-word-top">conversation</span><span class="learn-orbit-word learn-orbit-word-right">action</span><span class="learn-orbit-word learn-orbit-word-bottom">verification</span><span class="learn-orbit-word learn-orbit-word-left">memory</span></div>
            </header>
            <div class="learn-contents" role="group" aria-label="Guide sections">
                <a href="#learn-start">Start</a><a href="#learn-work">What to ask</a><a href="#learn-money">Spending wisely</a><a href="#learn-change">Changes</a><a href="#learn-contribute">Issues and PRs</a><a href="#learn-limits">Limits</a>
            </div>
            <section id="learn-start" class="learn-section"><h3>Three steps to begin</h3>
                <ol class="learn-steps"><li><strong>Connect a model.</strong> On first start the wizard helps you choose an account, model, review and budget. If setup is already done, check it in Settings &rarr; Accounts and Models.</li>
                <li><strong>Give a concrete task.</strong> Say why it is needed, where the source files are and how to check the result. The drafts below are yours — only you send them.</li>
                <li><strong>Check the result.</strong> Open the task card and the files. Status, answer, tests, review and the delivered result are different facts.</li></ol></section>
            <section id="learn-work" class="learn-section"><h3>What to ask</h3><p>Fill the frame with your own words. The button appends it to the Main Chat draft but never sends anything and never erases text you have already typed.</p>
                <div class="learn-templates">${starters.map((item, index) => `<article class="learn-template"><h4>${escapeHtml(item.title)}</h4><label for="learn-template-${index}">Task text</label><textarea id="learn-template-${index}" rows="4">${escapeHtml(item.text())}</textarea><button type="button" data-template="${index}" class="btn btn-secondary">Add to draft &rarr;</button></article>`).join('')}</div>
                <p class="learn-feedback" role="status" aria-live="polite" data-i18n-skip></p></section>
            <section id="learn-money" class="learn-section"><h3>Spending wisely means steering the route</h3><div class="learn-grid">
                <article><h4>Start with what you have</h4><p>A subscription spends quota; an API key may be billed per token. An empty price in the interface does not mean a free call. In Accounts check the connection, in Models the Main and Light assignments.</p></article>
                <article><h4>Limit the risk</h4><p>In Settings &rarr; Behavior check context and background tasks, in Dashboard &rarr; Costs the spend accounting. Nano shrinks the working window but does not cancel review. A budget cap and visible estimates do not guarantee an exact external provider bill.</p></article>
                <article><h4>Match verification to scale</h4><p>Do not launch a large swarm for a simple question. For a code change agree on boundaries and a success criterion up front. Do not mistake a few passing tests for full verification.</p></article></div>
                <button type="button" class="btn btn-secondary" data-open="costs">Open costs &rarr;</button></section>
            <section id="learn-change" class="learn-section"><h3>Changing itself while keeping a way back</h3>
                <p>I can read and change my own code, but a good task names the goal, the affected contract, the tests and the publication boundary. Work in a separate clean copy or branch; compare the diff against the current base, check related documents and call sites, then ask for an independent review of the final bytes. An ordinary commit on the Ouroboros working branch is a versioned, reviewed release; external PRs keep the version neutral until integration. Emergency snapshots and mechanical rollbacks are separate exceptions.</p>
                <p>Review is not a PASS while a reviewer has not answered or its route is unavailable. If you fixed the diff, have the changed bytes checked again. Merging and installing are separate actions; an open PR does not mean the change already works for you.</p></section>
            <section id="learn-contribute" class="learn-section"><h3>How to file an issue or PR</h3><div class="learn-grid">
                <article><h4>Issue: show what you observed</h4><p>Version, system, reproduction steps, expected and actual behavior, a safe log excerpt. Search existing issues first. Remove tokens, personal data and paths you do not want public.</p></article>
                <article><h4>PR: keep the boundary narrow</h4><p>One goal and a current upstream base. Explain why the change is needed, which contracts it touches, how it was verified and what remains unverified. Ask me to prepare the text and check the GitHub target; publishing requires your explicit decision.</p></article></div>
                <p>The GitHub tools depend on the connected account and its permissions; a refusal or a missing tool is not a published result.</p></section>
            <section id="learn-limits" class="learn-section"><h3>Honest about limits</h3><div class="learn-grid">
                <article><h4>Memory is not a guarantee of accuracy</h4><p>History and notes persist, but summarization and search can miss or misread context. Ask me to show the source and correct me when a record is wrong.</p></article>
                <article><h4>Background work needs a running process</h4><p>Wake-ups depend on settings, budget and Ouroboros running. Closing the app promises neither continuation nor notifications outside a separate transport.</p></article>
                <article><h4>Tools have boundaries</h4><p>No model gets access to every folder and service on a promise. Check what is allowed, and judge an external action by its receipt, not by my intention.</p></article>
                <article><h4>Images and screens</h4><p>A text model may receive a description instead of pixels. Visual verification needs an available sighted route and a look at the real result; a screenshot by itself is not a check.</p></article></div></section>
        </div>`;
    document.getElementById('content').appendChild(page);
    // The boot dictionary read resolves after this synchronous mount, and the overlay never
    // rewrites textareas: tr() at mount time would bake the English fallback in forever.
    // Repaint from the same tr() seam whenever the applied language changes, but keep any
    // textarea the owner has already edited — once their words differ from what we wrote,
    // the draft is theirs. The page lives for the whole session, like its click listeners.
    const templateAreas = new Map();
    const paintTemplates = () => {
        starters.forEach((item, index) => {
            const area = page.querySelector(`#learn-template-${index}`);
            if (!area) return;
            const previous = templateAreas.get(area);
            if (!shouldRepaintTemplate(previous, area.value)) return;
            const text = item.text();
            templateAreas.set(area, text);
            area.value = text;
        });
    };
    paintTemplates();
    if (typeof window !== 'undefined') window.addEventListener('ouro:language-changed', paintTemplates);
    page.addEventListener('click', async (event) => {
        const button = event.target.closest('button');
        if (!button || !page.contains(button)) return;
        if (button.dataset.template !== undefined) {
            const text = page.querySelector(`#learn-template-${button.dataset.template}`)?.value.trim();
            const input = document.querySelector('#page-chat #chat-input');
            const feedback = page.querySelector('.learn-feedback');
            if (!text || !input) { feedback.textContent = tr('learn.feedback.missing', 'Could not prepare the draft. Open Main Chat and try again.'); return; }
            if (!await showPage('chat')) { feedback.textContent = tr('learn.feedback.cancelled', 'Navigation cancelled: unfinished changes on the current page were kept.'); return; }
            input.value = appendToDraft(input.value, text);
            input.dispatchEvent(new Event('input', { bubbles: true }));
            feedback.textContent = tr('learn.feedback.appended', 'Added to the Main Chat draft. Sending happens only after you press Send.');
            input.focus();
            return;
        }
        if (button.dataset.open === 'settings') void openSettingsTab('providers');
        else if (button.dataset.open === 'costs') void openDashboardTab('costs');
        else if (button.dataset.open === 'chat') void showPage('chat');
    });
    // In-page links must not mutate the application's one-shot #page route.
    page.querySelectorAll('.learn-contents a').forEach((link) => link.addEventListener('click', (event) => {
        event.preventDefault();
        page.querySelector(link.getAttribute('href'))?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
    }));
}
