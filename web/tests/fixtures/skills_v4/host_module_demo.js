export function mount() {
    if (new URL(globalThis.location.href).searchParams.has('host_fail')) {
        throw new Error('fixture page module failed');
    }
    return { dispose() {} };
}
