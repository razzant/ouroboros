/** A jump is a page layer, never another history. The shell supplies the current retained place and the leave
 * guard; the destination commits synchronously with its return, after that guard has allowed the move. */
export function createPageNavigation({ layers, capture, canLeave, doc = document } = {}) {
    let serial = 0;
    let pressed = null;
    const onPress = (event) => {
        pressed = event.target?.closest?.('button, a, [role="button"], [role="tab"]') || event.target;
        queueMicrotask(() => { pressed = null; });
    };
    doc.addEventListener('click', onPress, true);

    function clear() {
        for (const id of layers.ids()) if (layers.get(id)?.returnTo) layers.remove(id);
    }

    const currentPlace = () => capture(pressed || doc.activeElement);
    async function go(key, commit, { map = false, origin = currentPlace() } = {}) {
        if (await canLeave(key) === false) return false;
        if (origin.key !== key) doc.activeElement?.blur?.();
        layers.batch(() => {
            if (map === true) clear(); // sidebar/drawer; a section's own list and tabs keep the visit
            else if (!map && origin.key !== key) {
                const id = `page-return-${++serial}`;
                layers.push({ id, page: true, modal: false, focus: false, opener: null,
                    returnTo: origin, destination: key, back: { label: origin.label, title: origin.title },
                    onClose: async () => {
                        if (await canLeave(origin.key) === false) return false;
                        doc.activeElement?.blur?.();
                        const restored = layers.batch(() => {
                            layers.remove(id);
                            return origin.restore();
                        });
                        restored?.();
                        return true;
                    } });
            }
            commit();
        });
        return true;
    }
    return { go, clear, capture: currentPlace, destroy: () => doc.removeEventListener('click', onPress, true) };
}
