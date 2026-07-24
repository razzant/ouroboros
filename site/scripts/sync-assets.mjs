import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourceRoot = resolve(siteRoot, "..", "assets");
const targetRoot = resolve(siteRoot, "public", "assets");
const assetNames = [
  "chat.png",
  "icon_1024.png",
  "og-preview.png",
  "settings.png",
  "setup.png",
  "social-preview.png"
];

await mkdir(targetRoot, { recursive: true });
await Promise.all(assetNames.map((name) => copyFile(resolve(sourceRoot, name), resolve(targetRoot, name))));
