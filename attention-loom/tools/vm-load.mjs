// Load the page's classic model scripts into a Node vm context exactly as the
// browser would (tensor.js, transformer.js, weights-*.js, api.js) and return the
// context, whose .AM.model is the real AM.model API.
import vm from 'node:vm';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function loadModelScripts({ weights = ['tinyworld', 'reverse', 'sort'] } = {}) {
  const ctx = { console, setTimeout, clearTimeout, Date, Math, performance };
  ctx.window = ctx; ctx.self = ctx; ctx.globalThis = ctx;
  vm.createContext(ctx);
  const files = ['js/model/tensor.js', 'js/model/transformer.js',
    ...weights.map(w => `js/model/weights-${w}.js`).filter(f => fs.existsSync(path.join(ROOT, f))),
    'js/model/api.js'];
  for (const f of files) vm.runInContext(fs.readFileSync(path.join(ROOT, f), 'utf8'), ctx, { filename: f });
  return ctx;
}
