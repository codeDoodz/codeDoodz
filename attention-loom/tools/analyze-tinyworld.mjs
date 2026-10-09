// What did the shipped tinyworld model learn? Runs the real AM.model on the held-out
// set and reports, per dependency type:
//   - which heads attend from the prediction position to the "evidence" token
//     (the antecedent, the head noun, the earlier mention, …)
//   - the logit lens: mean P(correct word) read off the residual stream after each layer
// plus a few curated sentences with the model's top predictions.
//
//   node tools/analyze-tinyworld.mjs [--md] [--write-meta]
// --write-meta stores the head/lens summary in js/model/weights-tinyworld.js as meta.analysis
// (so chapters can read it from AM.model.get('tinyworld').meta.analysis).
import fs from 'node:fs';
import path from 'node:path';
import { loadModelScripts, ROOT } from './vm-load.mjs';
import { CAPITALS, SOUNDS, COLOR_FACTS } from './data-tinyworld.mjs';

const md = process.argv.includes('--md');
const ctx = loadModelScripts({ weights: ['tinyworld'] });
const m = ctx.AM.model.get('tinyworld');
const test = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/fixtures/tinyworld-test.json'), 'utf8'));
const { n_layer: L, n_head: H } = m.config;

const SUBJ = new Set(['king', 'queen', 'boy', 'girl', 'prince', 'princess', 'wizard', 'witch', 'alice', 'emma', 'lucy', 'rose',
  'bob', 'tom', 'sam', 'jack', 'dog', 'cat', 'duck', 'owl', 'pig', 'cow', 'lion', 'children', 'dogs', 'cats']);
const PREPS = new Set(['near', 'under', 'behind']);
const FACT_KEYS = new Set([...Object.keys(CAPITALS), ...Object.values(CAPITALS), ...Object.keys(SOUNDS), ...Object.keys(COLOR_FACTS)]);

// Index of the token that carries the answer, or -1 if not found. p = critical position.
function evidence(tok, p, type) {
  const back = (pred) => { for (let j = p - 1; j >= 0; j--) if (pred(j)) return j; return -1; };
  switch (type) {
    case 'pronoun': case 'possessive': return back(j => SUBJ.has(tok[j]));
    case 'agreement': { const q = back(j => PREPS.has(tok[j])); return q > 0 ? q - 1 : -1; }
    case 'binding': return back(j => j < p - 2 && tok[j] === tok[p - 2]);             // earlier mention of the asked noun
    case 'binding→colour': return back(j => j < p - 2 && tok[j] === tok[p]);          // the right colour word itself
    case 'recall': { const c = back(j => j < p - 1 && tok[j] === tok[p - 1]); return c >= 0 ? c + 1 : -1; } // word after the earlier colour
    case 'copy': return back(j => tok[j] === tok[p]);
    case 'parrot': return back(j => j < p - 1 && tok[j] === tok[p] && tok[j - 1] === tok[p - 1]);
    default: return back(j => FACT_KEYS.has(tok[j]));
  }
}

const stats = {};
const prevTok = Array.from({ length: L }, () => new Float64Array(H));
let nSeq = 0;
for (const s of test) {
  const ids = s.ids.slice(0, m.config.n_ctx);
  const r = m.run(ids, { capture: true });
  const tok = s.text.split(' ');
  // previous-token score: mean attention from every position t ≥ 1 to t − 1
  for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
    let a = 0;
    for (let t = 1; t < ids.length; t++) a += r.attn[l][h][t][t - 1];
    prevTok[l][h] += a / (ids.length - 1);
  }
  nSeq++;
  const crits = s.crit.flatMap(c => (c.type === 'binding' ? [c, { ...c, type: 'binding→colour' }] : [c]));
  for (const c of crits) {
    if (c.pos >= ids.length + 1) continue;
    const e = evidence(tok, c.pos, c.type);
    const st = stats[c.type] = stats[c.type] || { n: 0, att: Array.from({ length: L }, () => new Float64Array(H)), lens: new Float64Array(L + 1), ne: 0 };
    st.n++;
    const tgt = m.tokenId(c.target), q = c.pos - 1;
    for (let l = 0; l <= L; l++) st.lens[l] += r.lens[l][q][tgt];
    if (e < 0) continue;
    st.ne++;
    for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) st.att[l][h] += r.attn[l][h][q][e];
  }
}

const analysis = { heads: {}, lens: {}, prevToken: prevTok.map(row => Array.from(row, v => +(v / nSeq).toFixed(3))) };
const out = [];
const say = (s) => out.push(s);
say(md ? '| type | n | top heads → evidence token (mean attention) | logit lens: P(correct) after embed, L0, L1, … |' : 'type        heads attending to the evidence token            logit lens P(correct) by layer');
if (md) say('|---|---|---|---|');
for (const [type, st] of Object.entries(stats).sort()) {
  const heads = [];
  for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) heads.push({ l, h, a: st.ne ? st.att[l][h] / st.ne : 0 });
  heads.sort((a, b) => b.a - a.a);
  const hs = heads.slice(0, 3).map(x => `L${x.l}H${x.h} ${x.a.toFixed(2)}`).join(', ');
  const lens = Array.from(st.lens, v => (v / st.n).toFixed(2)).join(' → ');
  analysis.heads[type] = heads.slice(0, 3).map(x => ({ layer: x.l, head: x.h, attn: +x.a.toFixed(3) }));
  analysis.lens[type] = Array.from(st.lens, v => +(v / st.n).toFixed(3));
  say(md ? `| ${type} | ${st.n} | ${hs} | ${lens} |` : `${type.padEnd(11)} ${hs.padEnd(45)} ${lens}`);
}

say('');
say((md ? '' : '') + 'previous-token score (mean attention from t to t−1): ' +
  prevTok.map((row, l) => Array.from(row, (v, h) => `L${l}H${h} ${(v / nSeq).toFixed(2)}`).join(' ')).join(' | '));

const examples = [
  'the queen opened the door because', 'the children walked to the park and found the ball because',
  'the dog walked to the river with', 'the keys near the old door', 'the key near the old doors',
  'the red ball and the blue box . the box is', 'the red ball and the blue box . alice found the blue',
  'alice gave bob a cup . bob thanked', 'the capital of japan is', 'tokyo is the capital of', 'the duck says', 'the sky is',
  'the girl said red kite blue cup . the parrot said', 'the girl said red kite blue cup . the parrot said red',
];
say('');
for (const t of examples) {
  const { ids } = m.encode(t);
  const r = m.run(ids);
  const top = m.topk(r.probs[ids.length - 1], 3).map(x => `${x.token} ${(100 * x.p).toFixed(0)}%`).join(', ');
  say(md ? `- \`${t} …\` → ${top}` : `  ${t.padEnd(64)} → ${top}`);
}
console.log(out.join('\n'));

if (process.argv.includes('--write-meta')) {
  const file = path.join(ROOT, 'js/model/weights-tinyworld.js');
  const src = fs.readFileSync(file, 'utf8');
  const start = src.indexOf('{', src.indexOf('.tinyworld = ')), end = src.trimEnd().lastIndexOf(';');
  const obj = JSON.parse(src.slice(start, end));
  obj.meta.analysis = Object.assign({ note: 'heads: mean attention from the predicting position to the evidence token on the held-out set (top 3); lens: mean P(correct) decoded from resid[0..n_layer]; prevToken[l][h]: mean attention to t-1. Written by tools/analyze-tinyworld.mjs.' }, analysis);
  fs.writeFileSync(file, src.slice(0, start) + JSON.stringify(obj) + src.slice(end));
  console.log('wrote meta.analysis to ' + path.relative(ROOT, file));
}
