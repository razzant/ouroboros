// Load the shipped plugin modules as the browser does, including its shared /static host modules.
import { registerHooks } from 'node:module';
registerHooks({
    resolve(specifier, context, nextResolve) {
        const resolved = specifier.startsWith('/static/modules/')
            ? new URL(`../../modules/${specifier.slice('/static/modules/'.length)}`, import.meta.url).href : specifier;
        const result = nextResolve(resolved, context);
        return result.url.includes('/skills/live_conversation/ui/') ? { ...result, format: 'module' } : result;
    },
});
