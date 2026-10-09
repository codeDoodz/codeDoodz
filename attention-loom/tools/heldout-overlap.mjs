// How "held out" is the tinyworld test set, really?
//
// The train/test split hashes each whole unit. But the model is scored on the word that
// follows the unit's *context* (e.g. "the queen opened the door because" → she), and the
// same context can occur in training with a different continuation ("… she was tired ."
// vs "… she was cold ."). This script regenerates the exact training stream
// (data-tinyworld.mjs, seed 1, 600 000 sequences), collects every unit context that
// precedes a critical word, and scores the shipped model on fresh test sequences split
// into contexts SEEN in training vs NOVEL ones.
//
//   node tools/heldout-overlap.mjs [--test 6000] [--seed 777777]
import { loadModelScripts } from './vm-load.mjs';
import { sampleSequence, makeTest, CANDIDATES } from './data-tinyworld.mjs';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? +process.argv[i + 1] : d; };
const nTest = arg('--test', 6000), seed = arg('--seed', 777777);
const ctx = loadModelScripts({ weights: ['tinyworld'] });
const m = ctx.AM.model.get('tinyworld');
const key = (words, c) => c.type + '|' + words.slice(c.unit, c.pos).join(' ');

const t0 = Date.now();
const seen = new Set();
const r = ctx.AM.model.lib.AMTensorLib().rng(1); // same stream as makeTrain(600000, 1)
for (let i = 0; i < 600000; i++) { const s = sampleSequence(r, 'train'); for (const c of s.crit) seen.add(key(s.words, c)); }
console.log(`training stream: ${seen.size} distinct (type, unit context) keys  (${Date.now() - t0} ms)`);

const st = {};
for (const ex of makeTest(nTest, seed)) {
  const words = ex.text.split(' ');
  const run = m.run(ex.ids.slice(0, m.config.n_ctx));
  for (const c of ex.crit) {
    if (c.pos - 1 >= m.config.n_ctx) continue;
    const p = run.probs[c.pos - 1], tgt = m.tokenId(c.target);
    let best = -1; for (const w of CANDIDATES[c.type]) { const id = m.tokenId(w); if (best < 0 || p[id] > p[best]) best = id; }
    const bucket = seen.has(key(words, c)) ? 'seen' : 'novel';
    const o = (st[c.type] = st[c.type] || { seen: { n: 0, ok: 0 }, novel: { n: 0, ok: 0 } });
    o[bucket].n++; o[bucket].ok += best === tgt;
  }
}
const pct = (o) => (o.n ? (100 * o.ok / o.n).toFixed(1) + '%' : '—').padStart(7);
console.log('\ntype         contexts seen in training      contexts never seen');
for (const [k, o] of Object.entries(st).sort()) {
  const tot = o.seen.n + o.novel.n;
  console.log(`${k.padEnd(11)}  ${String(o.seen.n).padStart(5)} (${(100 * o.seen.n / tot).toFixed(0).padStart(3)}%) acc ${pct(o.seen)}    ${String(o.novel.n).padStart(5)} acc ${pct(o.novel)}`);
}
