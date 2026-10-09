// tinyworld — a small generative grammar of toy English for the word-level model.
//
// Every training sequence is 1–3 short "units" (sentences or sentence pairs).
// Units are built so that predicting certain words REQUIRES looking back:
//   pronoun     the queen opened the door because [she] was cold .
//   possessive  the children walked to the park with [their] kite .
//   agreement   the keys near the old door [are] gold .
//   binding     the red ball and the blue box . the box is [blue] .
//   copy        alice gave bob a cup . bob thanked [alice] .
//   capital     the capital of france is [paris] .   /   paris is the capital of [france] .
//   sound       the cow says [moo] .
// The bracketed word is the unit's "critical" token: we score the model there.
//
// Held-out split: each templated unit's text is hashed; ~10% of all possible
// units (hash % 10 == 0) never appear in training and form the test set.
// Fact units (capital, sound) and the plain "walked" unit are shared — the
// whole point of facts is that they have to be memorised.
//
// CLI:  node tools/data-tinyworld.mjs --out DIR [--n 600000] [--test 4000] [--seed 1]
//   writes DIR/train.u8 (n × 33 token ids, 0 = <pad>), DIR/test.json, DIR/vocab.json

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const AMTensorLib = require('../js/model/tensor.js');

export const SEQ_LEN = 33; // 32 input positions + 1 shifted target

/* ---------------- lexicon ---------------- */
const PEOPLE = { king: 'm', queen: 'f', boy: 'm', girl: 'f', prince: 'm', princess: 'f', wizard: 'm', witch: 'f' };
const NAMES = { alice: 'f', emma: 'f', lucy: 'f', rose: 'f', bob: 'm', tom: 'm', sam: 'm', jack: 'm' };
const ANIMALS = ['dog', 'cat', 'duck', 'owl', 'pig', 'cow', 'lion']; // "it" (sheep is ambiguous in number, facts only)
const PLURAL_SUBJ = ['children', 'dogs', 'cats'];                     // "they"
const PRON = { m: 'he', f: 'she', n: 'it', p: 'they' };
const POSS = { m: 'his', f: 'her', n: 'its', p: 'their' };
const OBJ_PAIRS = [['key', 'keys'], ['box', 'boxes'], ['cup', 'cups'], ['door', 'doors'], ['book', 'books'], ['hat', 'hats']];
const SMALL_THINGS = ['key', 'cup', 'ring', 'hat', 'ball', 'kite', 'crown', 'book', 'box'];
const OPENABLE = ['door', 'box', 'book'];
const VERBS = { opened: OPENABLE, closed: OPENABLE, found: SMALL_THINGS, lost: SMALL_THINGS, dropped: SMALL_THINGS };
const STATES = ['cold', 'tired', 'happy', 'sad', 'hungry', 'sleepy'];
const PREPS = ['near', 'under', 'behind'];
const NP_ADJ = ['old', 'small'];
const PRED_ADJ = ['gold', 'heavy', 'broken', 'shiny', 'old', 'small'];
const COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'pink'];
const BIND_OBJS = ['ball', 'box', 'cup', 'hat', 'kite', 'book', 'ring', 'door'];
const PLACES = ['park', 'river', 'market', 'castle', 'garden'];
const POSS_OBJS = ['hat', 'ball', 'book', 'ring', 'key', 'crown', 'kite', 'cup'];
export const CAPITALS = { france: 'paris', japan: 'tokyo', italy: 'rome', spain: 'madrid', egypt: 'cairo', peru: 'lima', china: 'beijing', kenya: 'nairobi' };
// Colour facts use the same syntax as binding questions ("the X is ___"), but the answer
// comes from memory instead of from context.
export const COLOR_FACTS = { sky: 'blue', grass: 'green', snow: 'white', sun: 'yellow', apple: 'red', pig: 'pink' };
export const SOUNDS = { cow: 'moo', dog: 'woof', cat: 'meow', duck: 'quack', sheep: 'baa', owl: 'hoot', pig: 'oink', lion: 'roar' };

const FUNCTION_WORDS = ['the', 'a', 'and', 'because', 'with', 'of', 'is', 'are', 'was', 'were', 'says', 'to', 'at',
  'near', 'under', 'behind', 'walked', 'loved', 'gave', 'thanked', 'met', 'waved', 'helped', 'capital', 'said', 'parrot'];

function buildVocab() {
  const words = ['<pad>', '<unk>', '.', ','];
  const addAll = (xs) => { for (const w of xs) if (!words.includes(w)) words.push(w); };
  addAll(FUNCTION_WORDS);
  addAll(Object.values(PRON)); addAll(Object.values(POSS));
  addAll(Object.keys(PEOPLE)); addAll(Object.keys(NAMES)); addAll(ANIMALS); addAll(['sheep']); addAll(PLURAL_SUBJ);
  addAll(OBJ_PAIRS.flat()); addAll(SMALL_THINGS); addAll(BIND_OBJS); addAll(POSS_OBJS);
  addAll(Object.keys(VERBS)); addAll(STATES); addAll(NP_ADJ); addAll(PRED_ADJ); addAll(COLORS); addAll(PLACES);
  addAll(Object.keys(CAPITALS)); addAll(Object.values(CAPITALS)); addAll(Object.values(SOUNDS)); addAll(Object.keys(COLOR_FACTS));
  return words;
}
export const VOCAB = buildVocab();
export const WORD_ID = Object.fromEntries(VOCAB.map((w, i) => [w, i]));

export const CRIT_TYPES = ['pronoun', 'possessive', 'agreement', 'binding', 'recall', 'copy', 'capital', 'sound', 'colorfact', 'parrot'];
// The words a critical slot competes among (used for "candidate-set" accuracy).
export const CANDIDATES = {
  pronoun: Object.values(PRON), possessive: Object.values(POSS), agreement: ['is', 'are'], binding: COLORS,
  recall: BIND_OBJS, parrot: [...COLORS, ...BIND_OBJS], copy: Object.keys(NAMES), capital: [...Object.values(CAPITALS), ...Object.keys(CAPITALS)], sound: Object.values(SOUNDS), colorfact: COLORS,
};

/* ---------------- unit generators ---------------- */
// Each returns {w: string[], crit: [{i, type}], shared: bool}.
const pick = (r, xs) => xs[r.int(xs.length)];

// A subject noun phrase with its pronoun class.
function subject(r) {
  const cls = pick(r, ['m', 'f', 'n', 'p']);
  if (cls === 'm' || cls === 'f') {
    const pool = r.next() < 0.5 ? NAMES : PEOPLE;
    const opts = Object.keys(pool).filter(k => pool[k] === cls);
    const w = pick(r, opts);
    return { w: pool === NAMES ? [w] : ['the', w], cls };
  }
  if (cls === 'n') return { w: ['the', pick(r, ANIMALS)], cls };
  if (r.next() < 0.3) { // two names joined by "and" → they
    const names = Object.keys(NAMES), a = pick(r, names);
    let b = pick(r, names); while (b === a) b = pick(r, names);
    return { w: [a, 'and', b], cls };
  }
  return { w: ['the', pick(r, PLURAL_SUBJ)], cls };
}

function unitPronoun(r) {
  const s = subject(r);
  const verb = pick(r, Object.keys(VERBS));
  const w = [...s.w];
  if (r.next() < 0.3) w.push('walked', 'to', 'the', pick(r, PLACES), 'and'); // stretch the distance
  w.push(verb, 'the', pick(r, VERBS[verb]), 'because');
  const i = w.length;
  w.push(PRON[s.cls], s.cls === 'p' ? 'were' : 'was', pick(r, STATES), '.');
  return { w, crit: [{ i, type: 'pronoun' }] };
}

function unitPossessive(r) {
  const s = subject(r);
  const w = [...s.w];
  if (r.next() < 0.6) w.push('walked', 'to', 'the', pick(r, PLACES), 'with');
  else w.push('loved');
  const i = w.length;
  w.push(POSS[s.cls], pick(r, POSS_OBJS), '.');
  return { w, crit: [{ i, type: 'possessive' }] };
}

function unitAgreement(r) {
  const a = r.int(OBJ_PAIRS.length);
  let b = r.int(OBJ_PAIRS.length); while (b === a) b = r.int(OBJ_PAIRS.length);
  const plural = r.next() < 0.5;
  const attractorPlural = r.next() < 0.5;
  const w = ['the', OBJ_PAIRS[a][plural ? 1 : 0], pick(r, PREPS), 'the'];
  if (r.next() < 0.5) w.push(pick(r, NP_ADJ));
  w.push(OBJ_PAIRS[b][attractorPlural ? 1 : 0]);
  const i = w.length;
  w.push(plural ? 'are' : 'is', pick(r, PRED_ADJ), '.');
  return { w, crit: [{ i, type: 'agreement' }] };
}

function unitBinding(r) {
  // 2–4 coloured objects, sometimes with a size word, sometimes followed by an unrelated
  // sentence before the question. The varying layout means the model cannot answer by
  // position ("the colour 5 tokens back") and has to match the asked-about noun by content.
  const n = 2 + (r.next() < 0.35 ? 1 : 0) + (r.next() < 0.15 ? 1 : 0);
  const objs = [], cols = [];
  while (objs.length < n) { const o = pick(r, BIND_OBJS); if (!objs.includes(o)) objs.push(o); }
  while (cols.length < n) { const c = pick(r, COLORS); if (!cols.includes(c)) cols.push(c); }
  const w = [];
  for (let k = 0; k < n; k++) {
    if (k > 0) w.push(k === n - 1 ? 'and' : ',');
    w.push('the');
    if (r.next() < 0.25) w.push(pick(r, NP_ADJ));
    w.push(cols[k], objs[k]);
  }
  w.push('.');
  if (r.next() < 0.3) w.push(...subject(r).w, 'walked', 'to', 'the', pick(r, PLACES), '.');
  // One or two questions about different objects. Two kinds:
  //   colour question  "the box is [blue]"        noun → its colour
  //   object question  "alice found the blue [box]" colour → its noun (an induction pattern:
  //                    find the earlier "blue", copy the word that followed it)
  const qs = [r.int(n)];
  if (r.next() < 0.5) { let q2 = r.int(n); while (q2 === qs[0]) q2 = r.int(n); qs.push(q2); }
  const crit = [];
  for (const q of qs) {
    if (r.next() < 0.5) {
      w.push('the', objs[q], 'is');
      crit.push({ i: w.length, type: 'binding' });
      w.push(cols[q], '.');
    } else {
      w.push(...subject(r).w, 'found', 'the', cols[q]);
      crit.push({ i: w.length, type: 'recall' });
      w.push(objs[q], '.');
    }
  }
  return { w, crit };
}

function unitCopy(r) {
  const names = Object.keys(NAMES), x = pick(r, names);
  let y = pick(r, names); while (y === x) y = pick(r, names);
  const k = r.int(3);
  let w;
  if (k === 0) w = [x, 'gave', y, 'a', pick(r, POSS_OBJS), '.', y, 'thanked'];
  else if (k === 1) w = [x, 'met', y, 'at', 'the', pick(r, PLACES), '.', y, 'waved', 'at'];
  else w = [x, 'helped', y, '.', y, 'thanked'];
  const i = w.length;
  w.push(x, '.');
  return { w, crit: [{ i, type: 'copy' }] };
}

function unitCapital(r) {
  const c = pick(r, Object.keys(CAPITALS));
  if (r.next() < 0.6) return { w: ['the', 'capital', 'of', c, 'is', CAPITALS[c], '.'], crit: [{ i: 5, type: 'capital' }], shared: true };
  return { w: [CAPITALS[c], 'is', 'the', 'capital', 'of', c, '.'], crit: [{ i: 5, type: 'capital' }], shared: true };
}

function unitColorFact(r) {
  const x = pick(r, Object.keys(COLOR_FACTS));
  return { w: ['the', x, 'is', COLOR_FACTS[x], '.'], crit: [{ i: 3, type: 'colorfact' }], shared: true };
}

function unitSound(r) {
  const a = pick(r, Object.keys(SOUNDS));
  return { w: ['the', a, 'says', SOUNDS[a], '.'], crit: [{ i: 3, type: 'sound' }], shared: true };
}

// The parrot repeats what someone said. Copying a repeated phrase is the classic
// training signal for previous-token + induction heads, which binding also relies on.
function unitParrot(r) {
  const s = subject(r);
  const n = 2 + r.int(2), said = [];
  for (let k = 0; k < n; k++) said.push(pick(r, COLORS), pick(r, BIND_OBJS));
  const w = [...s.w, 'said', ...said, '.'];
  // sometimes something else happens before the parrot speaks, so the copy distance varies
  // and the words must be found by content (an induction head), not by a fixed offset
  if (r.next() < 0.5) w.push(...subject(r).w, 'walked', 'to', 'the', pick(r, PLACES), '.');
  w.push('the', 'parrot', 'said');
  const crit = said.map((_, k) => ({ i: w.length + k, type: 'parrot' }));
  w.push(...said, '.');
  return { w, crit };
}

function unitWalk(r) {
  const s = subject(r);
  return { w: [...s.w, 'walked', 'to', 'the', pick(r, PLACES), '.'], crit: [], shared: true };
}

// Template mixture. Binding gets the largest share: it needs a two-step,
// induction-style circuit that forms late, so it benefits from more examples.
const UNITS = [
  [unitPronoun, 0.14], [unitPossessive, 0.08], [unitAgreement, 0.12], [unitBinding, 0.22],
  [unitCopy, 0.10], [unitParrot, 0.10], [unitCapital, 0.07], [unitSound, 0.06], [unitColorFact, 0.07], [unitWalk, 0.04],
];

// FNV-1a hash of the unit text → train/test bucket.
function hashStr(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
export function isTestUnit(words) { return hashStr(words.join(' ')) % 10 === 0; }

function sampleUnit(r, split) {
  let x = r.next(), gen = UNITS[UNITS.length - 1][0];
  for (const [g, p] of UNITS) { if (x < p) { gen = g; break; } x -= p; }
  for (;;) { // re-draw from the same template until the unit lands in the requested split
    const u = gen(r);
    if (u.shared || isTestUnit(u.w) === (split === 'test')) return u;
  }
}

/* A sequence of 1–3 units, ≤ SEQ_LEN tokens. Returns {words, crit:[{pos, type, target}]}. */
export function sampleSequence(r, split = 'train') {
  const nUnits = 1 + r.int(3);
  const words = [], crit = [];
  for (let k = 0; k < nUnits; k++) {
    const u = sampleUnit(r, split);
    if (words.length + u.w.length > SEQ_LEN) { if (k === 0) { k--; continue; } break; } // first unit must fit
    for (const c of u.crit) crit.push({ pos: words.length + c.i, type: c.type, target: u.w[c.i] });
    words.push(...u.w);
  }
  return { words, crit };
}

export function encodeWords(words) { return words.map(w => { const id = WORD_ID[w]; if (id == null) throw new Error('not in vocab: ' + w); return id; }); }

export function makeTrain(n, seed) {
  const r = AMTensorLib().rng(seed);
  const buf = new Uint8Array(n * SEQ_LEN); // 0 = <pad>
  for (let s = 0; s < n; s++) buf.set(encodeWords(sampleSequence(r, 'train').words), s * SEQ_LEN);
  return buf;
}

export function makeTest(n, seed) {
  const r = AMTensorLib().rng(seed);
  const out = [];
  while (out.length < n) {
    const s = sampleSequence(r, 'test');
    if (!s.crit.length) continue;
    out.push({ text: s.words.join(' '), ids: encodeWords(s.words), crit: s.crit });
  }
  return out;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = Object.fromEntries(process.argv.slice(2).join(' ').split('--').filter(Boolean).map(s => s.trim().split(/\s+/)));
  const outDir = args.out || path.join(process.env.TMPDIR || '/tmp', 'attention-loom-data');
  const n = +(args.n || 600000), nTest = +(args.test || 4000), seed = +(args.seed || 1);
  fs.mkdirSync(outDir, { recursive: true });
  const t0 = Date.now();
  fs.writeFileSync(path.join(outDir, 'train.u8'), makeTrain(n, seed));
  fs.writeFileSync(path.join(outDir, 'test.json'), JSON.stringify(makeTest(nTest, seed + 1000)));
  fs.writeFileSync(path.join(outDir, 'vocab.json'), JSON.stringify({ vocab: VOCAB, seq_len: SEQ_LEN, crit_types: CRIT_TYPES, candidates: CANDIDATES }));
  console.log(`tinyworld: vocab ${VOCAB.length}, ${n} train seqs, ${nTest} test seqs → ${outDir} (${Date.now() - t0} ms)`);
  const r = AMTensorLib().rng(seed + 7);
  for (let k = 0; k < 6; k++) console.log('  e.g. ' + sampleSequence(r, 'train').words.join(' '));
}
