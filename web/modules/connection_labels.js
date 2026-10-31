// The owner's labels of the API keys added under Accounts, for surfaces that show a stored route outside
// Settings (a task card's model line, the wait card). A route names its key by its stored name
// (`kind@name::model`); the label comes from the settings document (`_meta.connections`), which Settings and
// the wait picker adopt from the reads they already make. This module reads nothing and repaints nothing:
// until a label is known a surface shows the key's stored name, and its next paint shows the label.

const labels = new Map();

/** The added key a stored route names (`kind@name`), or '' for a built-in key and every other route. */
export function routeConnectionId(route = '') {
    const value = String(route || '').trim();
    const cut = value.indexOf('::');
    const source = cut < 0 ? '' : value.slice(0, cut);
    const at = source.indexOf('@');
    const name = at > 0 ? source.slice(at + 1) : '';
    return name && name !== 'default' && !source.startsWith('claudexor') ? source : '';
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
