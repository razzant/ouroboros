// Runs after `vite build`. The build keeps ../docs (it also holds the handbook), so the files of earlier builds
// would stay beside the new ones. This removes, from docs/assets, a hashed bundle that nothing built still loads,
// and from docs/ui and docs/fonts, a file that has left site/public.
// A bundle is kept when a built page, a published file with a stable name or another kept bundle names it (an
// imported chunk). A file that site/public/assets also holds is one of those stable files and is never removed.
import { readdir, readFile, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outRoot = resolve(siteRoot, "..", "docs");
const hashedBundle = /^[\w.-]+-[\w-]{8}\.(?:js|css)$/;

async function names(dir) {
  try {
    return (await readdir(dir, { withFileTypes: true })).filter((entry) => entry.isFile()).map((entry) => entry.name);
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
}

async function pages(dir) {
  const found = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) found.push(...await pages(path));
    else if (entry.name.endsWith(".html")) found.push(path);
  }
  return found;
}

const assetsDir = join(outRoot, "assets");
const built = await names(assetsDir);
const stable = new Set(await names(join(siteRoot, "public", "assets")));
const loaded = new Set();
const unread = await Promise.all((await pages(outRoot)).map((page) => readFile(page, "utf8")));
for (const name of built) {
  if (stable.has(name) && /\.(?:js|css)$/.test(name)) {
    loaded.add(name);
    unread.push(await readFile(join(assetsDir, name), "utf8"));
  }
}
while (unread.length) {
  const text = unread.pop();
  for (const name of built) {
    if (loaded.has(name) || !text.includes(name)) continue;
    loaded.add(name);
    if (/\.(?:js|css)$/.test(name)) unread.push(await readFile(join(assetsDir, name), "utf8"));
  }
}
const removed = built
  .filter((name) => hashedBundle.test(name) && !loaded.has(name) && !stable.has(name))
  .map((name) => join("assets", name));
for (const dir of ["ui", "fonts"]) {
  const kept = new Set(await names(join(siteRoot, "public", dir)));
  for (const name of await names(join(outRoot, dir))) {
    if (!kept.has(name)) removed.push(join(dir, name));
  }
}
await Promise.all(removed.map((path) => rm(join(outRoot, path))));
if (removed.length) console.log(`pruned from docs: ${removed.sort().join(", ")}`);
