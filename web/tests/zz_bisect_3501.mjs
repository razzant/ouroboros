// Temporary diagnostic for #3501 (never merged): which earlier case of work_portal.test.js makes the census-first replay
// case run out of memory on Windows. Each run is one process (no isolation), a small heap, and its own time limit.
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const file = 'tests/work_portal.test.js';
const names = [...readFileSync(file, 'utf8').matchAll(/^test\('((?:[^'\\]|\\.)*)'/gm)].map((m) => m[1].replace(/\\'/g, "'"));
const target = names.findIndex((n) => n.startsWith('after a Main reload whose census arrives before the history draws the card'));
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const run = (picked) => {
    const pattern = `^(?:${picked.map((i) => esc(names[i])).join('|')})$`;
    const out = spawnSync(process.execPath, ['--max-old-space-size=256', '--experimental-test-isolation=none', '--test', '--test-reporter=tap',
        `--test-name-pattern=${pattern}`, file], { encoding: 'utf8', timeout: 60000, maxBuffer: 64 * 1024 * 1024 });
    const oks = (out.stdout.match(/^ok \d+/gm) || []).length;
    return `status=${out.status} signal=${out.signal} oks=${oks}${/heap out of memory/.test(out.stderr + out.stdout) ? ' OOM' : ''}`;
};
console.log(`target #${target}: ${names[target]}`);
console.log(`target alone: ${run([target])}`);
for (let k = 0; k < target; k += 1) console.log(`pair ${k} (${names[k].slice(0, 70)}): ${run([k, target])}`);
for (let k = 0; k < target; k += 1) console.log(`prefix 0..${k}: ${run([...Array(k + 1).keys(), target])}`);
