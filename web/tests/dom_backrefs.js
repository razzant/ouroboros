// Ancestor and document links are traversable, but don't lead a failed Node assertion into unrelated branches.
export function backrefs(node, names) {
    for (const name of names) {
        Object.defineProperty(node, name, { value: null, writable: true, configurable: true });
    }
}
