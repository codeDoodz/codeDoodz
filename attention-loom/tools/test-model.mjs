// End-to-end checks for the Attention Loom's tiny transformers.
//
//   node tools/test-model.mjs [--quick]
//
// 1. finite-difference gradient checks (every op + the full model, float64)
// 2. inference parity: the shipped JS forward vs the numpy trainer's logits
// 3. AM.model API contract (CONTRACT §6): shapes, causality, softmax sums, logit lens, sampling
// 4. accuracies: tinyworld per dependency type (held-out set), reverse/sort exact-sequence
// 5. the in-page trainer (worker source in a sandbox + main-thread fallback) learns reverse
// 6. timing report
// Exits non-zero if anything fails.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { runGradChecks } from './gradcheck.mjs';
import { loadModelScripts, ROOT } from './vm-load.mjs';

const quick = process.argv.includes('--quick');
let failures = 0;
const ok = (cond, msg) => { console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) failures++; };
const section = (s) => console.log('\n' + s);

/* 1 ------------------------------------------------------------------ */
section('1. gradient checks (float64, central differences)');
{ const { ok: g } = runGradChecks(); ok(g, 'all analytic gradients match finite differences (rel err < 1e-5)'); }

/* load the browser scripts exactly as the page does */
const ctx = loadModelScripts();
const AM = ctx.AM;
await AM.model.ready;
ok(JSON.stringify(AM.model.list().sort()) === JSON.stringify(['reverse', 'sort', 'tinyworld']), 'AM.model.list() = reverse, sort, tinyworld');

/* 2 ------------------------------------------------------------------ */
section('2. inference parity with the numpy trainer');
{
  const fx = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/fixtures/tinyworld-parity.json'), 'utf8'));
  const m = AM.model.get('tinyworld');
  let maxDiff = 0, maxAttn = 0, maxLogit = 0;
  for (const s of fx) {
    const r = m.run(s.ids, { capture: true });
    const L = m.config.n_layer - 1;
    for (let t = 0; t < s.ids.length; t++) {
      for (let j = 0; j < m.config.vocab_size; j++) {
        maxDiff = Math.max(maxDiff, Math.abs(r.logits[t][j] - s.logits[t][j]));
        maxLogit = Math.max(maxLogit, Math.abs(s.logits[t][j]));
      }
      for (let h = 0; h < m.config.n_head; h++) for (let k = 0; k < s.ids.length; k++)
        maxAttn = Math.max(maxAttn, Math.abs(r.attn[L][h][t][k] - s.attn_last[h][t][k]));
    }
  }
  ok(maxDiff < 2e-4, `logits max |JS − numpy| = ${maxDiff.toExponential(2)} (logit scale ${maxLogit.toFixed(1)}) over ${fx.length} sequences`);
  ok(maxAttn < 1e-5, `last-layer attention max |JS − numpy| = ${maxAttn.toExponential(2)}`);
}

/* 3 ------------------------------------------------------------------ */
section('3. AM.model API contract');
{
  const m = AM.model.get('tinyworld');
  const c = m.config;
  ok(['n_layer', 'n_head', 'd_model', 'd_ff', 'n_ctx', 'vocab_size'].every(k => Number.isInteger(c[k])), 'config has n_layer, n_head, d_model, d_ff, n_ctx, vocab_size');
  ok(m.vocab.length === c.vocab_size, 'vocab length matches config');
  const e = m.encode('The King lost his crown, ZEBRA.');
  ok(e.tokens.join(' ') === 'the king lost his crown , <unk> .' && e.unknown.join() === 'zebra', `encode lowercases, splits punctuation, flags unknowns → [${e.tokens.join(' ')}]`);
  ok(m.decode(e.ids).join(' ') === e.tokens.join(' '), 'decode(encode(x).ids) round-trips');
  const { ids } = m.encode('the queen opened the door because she was cold .');
  const r = m.run(ids, { capture: true });
  const Tn = ids.length, dh = c.d_model / c.n_head;
  ok(r.tokens.length === Tn && r.logits.length === Tn && r.probs.length === Tn, 'tokens/logits/probs have one row per position');
  let sumErr = 0, causal = 0;
  for (const layer of r.attn) for (const head of layer) head.forEach((row, q) => {
    sumErr = Math.max(sumErr, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
    for (let k = q + 1; k < Tn; k++) causal = Math.max(causal, row[k]);
  });
  ok(r.attn.length === c.n_layer && r.attn[0].length === c.n_head && r.attn[0][0][0].length === Tn, 'attn[layer][head][q] is Float32Array(T)');
  ok(sumErr < 1e-5 && causal === 0, `attention rows sum to 1 (max err ${sumErr.toExponential(1)}) and are causal (max weight above diagonal ${causal})`);
  ok(r.resid.length === c.n_layer + 1 && r.resid[0][0].length === c.d_model && r.residMid.length === c.n_layer, 'resid has n_layer+1 snapshots of d_model, residMid has n_layer');
  ok(r.q.length === c.n_layer && r.q[0].length === c.n_head && r.q[0][0][0].length === dh && r.k[1][2][3].length === dh && r.v[0][0][Tn - 1].length === dh, 'q/k/v[layer][head][t] are Float32Array(d_head)');
  ok(r.mlp.length === c.n_layer && r.mlp[0][0].length === c.d_ff, 'mlp[layer][t] is Float32Array(d_ff)');
  let lensErr = 0;
  for (let t = 0; t < Tn; t++) for (let j = 0; j < c.vocab_size; j++) lensErr = Math.max(lensErr, Math.abs(r.lens[c.n_layer][t][j] - r.probs[t][j]));
  ok(r.lens.length === c.n_layer + 1 && lensErr < 1e-6, `logit lens at l = n_layer equals the output probs (max err ${lensErr.toExponential(1)})`);
  // residual bookkeeping: resid[l+1] − residMid[l] is the MLP write; recompute it by hand from mlp acts
  const P = m._net.params, d = c.d_model, f = c.d_ff, t0 = 4;
  let mlpErr = 0;
  for (let j = 0; j < d; j++) {
    let s = P['h.0.mlp.bproj'].data[j];
    for (let k = 0; k < f; k++) s += r.mlp[0][t0][k] * P['h.0.mlp.wproj'].data[k * d + j];
    mlpErr = Math.max(mlpErr, Math.abs(r.resid[1][t0][j] - r.residMid[0][t0][j] - s));
  }
  ok(mlpErr < 1e-4, `resid[1] = residMid[0] + W_proj·mlp[0] + b (max err ${mlpErr.toExponential(1)})`);
  const top = m.topk(r.probs[6], 3);
  ok(top.length === 3 && top[0].p >= top[1].p && typeof top[0].token === 'string', `topk after "…because" → ${top.map(x => `${x.token} ${(x.p * 100).toFixed(1)}%`).join(', ')}`);
  const seeded = (s) => () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const a1 = m.sample(r.probs[2], { temperature: 0.8, topK: 5, rng: seeded(3) });
  const a2 = m.sample(r.probs[2], { temperature: 0.8, topK: 5, rng: seeded(3) });
  ok(a1 === a2 && m.sample(r.probs[6], { temperature: 0 }) === top[0].id, 'sample() is deterministic for a given rng; temperature 0 = argmax');
  let inTopP = true;
  for (let i = 0; i < 200; i++) { const id = m.sample(r.probs[2], { topP: 0.5, rng: seeded(i + 11) }); if (r.probs[2][id] < 0.01) inTopP = false; }
  ok(inTopP, 'topP sampling never picks a negligible token');
  const g = m.generate(m.encode('the capital of france is').ids, { maxNew: 10, temperature: 0 });
  ok(m.decode(g).join(' ') === 'the capital of france is paris .', `generate (greedy, stops at ".") → "${m.decode(g).join(' ')}"`);
  const g2 = m.generate(m.encode('alice gave bob a cup .').ids, { maxNew: 12, temperature: 0.7, rng: seeded(5) });
  console.log(`       sampled: "${m.decode(g2).join(' ')}"`);
  const rv = AM.model.get('reverse'), so = AM.model.get('sort');
  const er = rv.encode('3815 2907>');
  ok(er.tokens.join('') === '38152907>' && er.ids.length === 9, 'char encode strips spaces');
  ok(rv.decode(rv.generate(er.ids, { maxNew: 8, temperature: 0 })).join('') === '38152907>70925183', 'reverse generates 38152907>70925183');
  ok(so.decode(so.generate(so.encode('73519273>').ids, { maxNew: 8, temperature: 0 })).join('') === '73519273>12335779', 'sort generates 73519273>12335779');
}

/* 4 ------------------------------------------------------------------ */
section('4. accuracies');
const accReport = {};
{
  const m = AM.model.get('tinyworld');
  const test = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/fixtures/tinyworld-test.json'), 'utf8'));
  const vocab = JSON.parse(JSON.stringify(m.vocab));
  const { CANDIDATES } = await import('./data-tinyworld.mjs');
  const st = {};
  for (const s of test) {
    const r = m.run(s.ids.slice(0, m.config.n_ctx));
    for (const c of s.crit) {
      const row = r.probs[c.pos - 1], tgt = vocab.indexOf(c.target);
      const cand = CANDIDATES[c.type].map(w => vocab.indexOf(w));
      let best = cand[0]; for (const id of cand) if (row[id] > row[best]) best = id;
      let am = 0; for (let j = 1; j < row.length; j++) if (row[j] > row[am]) am = j;
      const o = st[c.type] = st[c.type] || { n: 0, acc: 0, top1: 0, p: 0 };
      o.n++; o.acc += best === tgt; o.top1 += am === tgt; o.p += row[tgt];
    }
  }
  for (const [k, o] of Object.entries(st).sort()) {
    const acc = o.acc / o.n;
    accReport['tinyworld.' + k] = acc;
    const line = `tinyworld ${k.padEnd(10)} n=${String(o.n).padStart(4)}  acc ${(100 * acc).toFixed(1)}%  top-1 ${(100 * o.top1 / o.n).toFixed(1)}%  mean P(correct) ${(o.p / o.n).toFixed(3)}`;
    // 'recall' (colour → noun) is an exploratory probe the model is documented NOT to have learned
    if (k === 'recall' && acc <= 0.95) console.log(`  info ${line}  (probe, not enforced — see docs/model-training.md)`);
    else ok(acc > 0.95, line);
  }
  const { AMTensorLib, AMTransformerLib, AMTaskLib } = AM.model.lib;
  const tasks = AMTaskLib(), T = AMTensorLib();
  for (const name of ['reverse', 'sort']) {
    const net = AM.model.get(name)._net, TL = AMTransformerLib(T);
    const rng = T.rng(4242);
    const bt = tasks.batchFrom(name, Array.from({ length: 2000 }, () => tasks.example(name, rng).seq));
    const sc = tasks.score(name, T.noGrad(() => TL.forward(net, bt.ids, bt.B, bt.T)).logits.data, bt);
    accReport[name] = sc.seqAcc;
    ok(sc.seqAcc >= 0.95, `${name.padEnd(8)} exact-sequence acc ${(100 * sc.seqAcc).toFixed(2)}%  digit acc ${(100 * sc.tokenAcc).toFixed(2)}%  (2000 fresh inputs)`);
  }
}

/* 5 ------------------------------------------------------------------ */
section('5. in-page trainer');
{
  // (a) the exact Worker source, evaluated in a bare sandbox that only has self/postMessage/setTimeout
  const msgs = [];
  const sandbox = { setTimeout, clearTimeout, Date, Math, console };
  sandbox.self = { postMessage: (m) => msgs.push(m) };
  vm.createContext(sandbox);
  vm.runInContext(AM.model.workerSource(), sandbox);
  sandbox.self.onmessage({ data: { type: 'init', task: 'reverse', opts: { seed: 2 } } });
  sandbox.self.onmessage({ data: { type: 'step', n: 150 } });
  const last = msgs.filter(m => m.type === 'progress').pop();
  ok(msgs[0].type === 'ready' && last && last.step === 150, 'worker source runs standalone (no outer references) and answers init/step');
  ok(last.attn.length === 1 && last.attn[0][0].length === 16 * 16, 'progress carries attn[layer][head] = Float32Array(T·T) for the probe');
  ok(last.acc > 0.9, `worker-trained reverse reaches ${(100 * last.acc).toFixed(1)}% after 150 steps; probe ${last.probePred} (target ${last.probeTarget})`);
  // anti-diagonal check: output query q (8..15) should attend to input key 15 − q... shifted: position 8+j predicts y_j = x_{7−j}
  let diag = 0;
  for (let q = 8; q < 16; q++) diag += last.attn[0][0][q * 16 + (15 - q)];
  ok(diag / 8 > 0.5, `reverse attention is anti-diagonal (mean weight on key 15−q: ${(diag / 8).toFixed(2)})`);

  // (b) main-thread fallback via createTrainerWorker (no Worker in Node) with start/pause
  if (!quick) {
    const ctl = AM.model.createTrainerWorker('sort', { seed: 1, reportMs: 500 });
    ok(ctl.inline, 'createTrainerWorker falls back to main-thread slices when Worker is unavailable');
    const t0 = Date.now();
    const res = await new Promise((resolve) => {
      let best = null;
      ctl.on((m) => {
        if (m.type === 'ready') ctl.start();
        if (m.type === 'progress') { best = m; if (m.acc >= 0.9 || Date.now() - t0 > 90000) { ctl.pause(); resolve(best); } }
      });
    });
    ctl.terminate();
    ok(res.acc >= 0.9, `Lab sort reached ${(100 * res.acc).toFixed(1)}% exact after ${res.step} steps, ${res.elapsed.toFixed(1)} s of training (${res.stepsPerSec.toFixed(1)} steps/s)`);
    accReport.labSort90 = { step: res.step, seconds: res.elapsed };
  }
}

/* 6 ------------------------------------------------------------------ */
section('6. timing');
{
  const m = AM.model.get('tinyworld');
  const ids = Array.from({ length: 32 }, (_, i) => m.encode('the queen opened the door because she was cold .').ids[i % 11]);
  for (let i = 0; i < 5; i++) m.run(ids, { capture: true });
  let t0 = performance.now();
  for (let i = 0; i < 50; i++) m.run(ids, { capture: true });
  const capMs = (performance.now() - t0) / 50;
  t0 = performance.now();
  for (let i = 0; i < 50; i++) m.run(ids);
  const plainMs = (performance.now() - t0) / 50;
  ok(capMs < 20, `tinyworld run(32 tokens, capture) ${capMs.toFixed(2)} ms; without capture ${plainMs.toFixed(2)} ms`);
  const { AMTensorLib, AMTransformerLib, AMTaskLib } = AM.model.lib;
  const T = AMTensorLib(), TL = AMTransformerLib(T), tasks = AMTaskLib();
  for (const name of ['reverse', 'sort']) {
    const t = tasks.task(name), model = TL.init(t.model, 1), tr = TL.createTrainer(model, t.train), rng = T.rng(1);
    for (let i = 0; i < 5; i++) tr.step(tasks.batch(name, rng, t.train.batch));
    t0 = performance.now();
    for (let i = 0; i < 40; i++) tr.step(tasks.batch(name, rng, t.train.batch));
    console.log(`       ${name} training: ${((performance.now() - t0) / 40).toFixed(1)} ms/step (batch ${t.train.batch}, ${TL.countParams(model)} params)`);
  }
}

console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
