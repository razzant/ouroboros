export const uploadView = (name, kind, extra = {}) => ({
    name, kind, mime: kind === 'image' ? 'image/png' : kind === 'audio' ? 'audio/wav' : 'application/pdf',
    size: 2048, available: true, url: `/api/files/download?upload=${'a'.repeat(32)}_${name}`, ...extra,
});
