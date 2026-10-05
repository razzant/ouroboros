// Checkout observations only: never authorship, review or loaded-module proof.
// Historical rows carry equality facts but cannot gain ancestry retrospectively.
import { fmt, tr } from './i18n.js';

export function workerShaRelation(evt) {
    if (evt.relation) return String(evt.relation);
    if (!evt.expected_sha) return 'not_applicable';
    if (!evt.observed_sha) return 'unavailable';
    return evt.expected_sha === evt.observed_sha ? 'equal' : 'unrecorded';
}

export function workerShaLogView(evt) {
    let phase = 'info', headline, body = '';
    switch (workerShaRelation(evt)) {
        case 'equal':
            phase = 'ok';
            headline = tr('worker.sha.equal', 'Worker checkout matches baseline');
            break;
        case 'descendant':
            headline = tr('worker.sha.descendant', 'Worker checkout descends from baseline');
            break;
        case 'non_descendant':
            phase = 'warn';
            headline = tr('worker.sha.non_descendant', 'Worker checkout is not a descendant of baseline');
            break;
        case 'unavailable':
            phase = 'warn';
            headline = tr('worker.sha.unavailable', 'Worker checkout comparison unavailable');
            break;
        case 'not_applicable':
            headline = tr('worker.sha.not_applicable', 'Worker checkout comparison skipped');
            body = tr('worker.sha.no_baseline', 'No managed baseline recorded.');
            break;
        default:
            headline = tr('worker.sha.different', 'Worker checkout differs from baseline');
            body = tr('worker.sha.unrecorded', 'Ancestry was not recorded for this observation.');
    }
    return { phase, headline, body, meta: [
        evt.expected_sha ? fmt('baseline {sha}', { sha: String(evt.expected_sha).slice(0, 8) }) : '',
        evt.observed_sha ? fmt('observed {sha}', { sha: String(evt.observed_sha).slice(0, 8) }) : '',
        evt.worker_pid ? `pid ${evt.worker_pid}` : '',
    ] };
}
