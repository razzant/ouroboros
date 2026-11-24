// The owner's labels of the API keys added under Accounts, for surfaces that show a stored route outside
// Settings (a task card's model line, the wait card). A route names its key by its stored name
// (`kind@name::model`); the label comes from the settings document (`_meta.connections`), which Settings and
// the wait picker adopt from the reads they already make. This module reads nothing and repaints nothing:
// until a label is known a surface shows the key's stored name, and its next paint shows the label.

import { apiRouteParts } from './harness_presentation.js';

const labels = new Map();

/** The added key a stored route names (`kind@name`, read by the shared route grammar apiRouteParts), or '' for a
 *  built-in key, a local-lane route, a name the resolver refuses and every opaque route. */
export function routeConnectionId(route = '') {
    const value = String(route || '').trim();
    const { kind, connection, named, valid } = value.endsWith(' (local)') ? {} : apiRouteParts(value);
    return named && valid && connection !== 'default' ? `${kind}@${connection}` : '';
}

/** Adopt one settings read's connection list (`_meta.connections`); an absent list changes nothing. */
export function adoptConnectionLabels(connections) {
    if (!Array.isArray(connections)) return;
    labels.clear();
    for (const entry of connections) {
        if (entry?.id && entry.name) labels.set(String(entry.id), String(entry.label || entry.name));
    }
}

/** The label of an added key (its stored name until a settings read named it); '' for none. */
export function connectionLabel(id) {
    if (!id) return '';
    return labels.get(id) || id.slice(id.indexOf('@') + 1);
}
