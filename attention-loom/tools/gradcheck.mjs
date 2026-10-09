// Finite-difference gradient checks for js/model/tensor.js and js/model/transformer.js.
// Runs in float64 mode (AMTensorLib({float64:true})) so central differences are precise.
// Usage: node tools/gradcheck.mjs      (also imported by tools/test-model.mjs)
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url);
const AMTensorLib = require('../js/model/tensor.js');
const AMTransformerLib = require('../js/model/transformer.js');

// Compare analytic grads of `params` against central differences of lossFn().
// lossFn builds a fresh graph each call and returns a 1x1 tensor.
function check(T, name, params, lossFn, { eps = 1e-6, maxPerParam = 60, seed = 7 } = {}) {
  const r = T.rng(seed);
  for (const p of params) p.zeroGrad();
  T.clearTape();
  const L = lossFn();
  T.backward(L);
  let worst = 0, worstName = '';
  for (const p of params) {
    const n = p.size;
    const idx = n <= maxPerParam ? [...Array(n).keys()] : Array.from({ length: maxPerParam }, () => r.int(n));
    const ana = [], num = [];
    for (const i of idx) {
      const orig = p.data[i];
      p.data[i] = orig + eps; const lp = T.noGrad(lossFn).data[0];
      p.data[i] = orig - eps; const lm = T.noGrad(lossFn).data[0];
      p.data[i] = orig;
      num.push((lp - lm) / (2 * eps));
      ana.push(p.grad ? p.grad[i] : 0);
    }
    let maxDiff = 0, scale = 1e-8;
    for (let k = 0; k < idx.length; k++) {
      maxDiff = Math.max(maxDiff, Math.abs(ana[k] - num[k]));
      scale = Math.max(scale, Math.abs(ana[k]), Math.abs(num[k]));
    }
    const rel = maxDiff / scale;
    if (rel > worst) { worst = rel; worstName = p.name || '?'; }
  }
  return { name, rel: worst, at: worstName };
}

export function runGradChecks({ verbose = true } = {}) {
  const T = AMTensorLib({ float64: true });
  const r = T.rng(42);
  const rand = (rows, cols, name, s = 1) => {
    const a = new Float64Array(rows * cols);
    for (let i = 0; i < a.length; i++) a[i] = r.normal() * s;
    return T.param(rows, cols, a, name);
  };
  const proj = (n) => Float64Array.from({ length: n }, () => r.normal());
  const results = [];

  { const A = rand(5, 7, 'A'), B = rand(7, 4, 'B'), R = proj(20);
    results.push(check(T, 'matmul', [A, B], () => T.sumProduct(T.matmul(A, B), R))); }
  { const X = rand(6, 5, 'X'), W = rand(5, 3, 'W'), b = rand(1, 3, 'b'), R = proj(18);
    results.push(check(T, 'linear', [X, W, b], () => T.sumProduct(T.linear(X, W, b), R))); }
  { const A = rand(4, 3, 'A'), B = rand(4, 3, 'B'), R = proj(12);
    results.push(check(T, 'add', [A, B], () => T.sumProduct(T.add(A, B), R))); }
  { const X = rand(4, 3, 'X'), b = rand(1, 3, 'b'), R = proj(12);
    results.push(check(T, 'addBias', [X, b], () => T.sumProduct(T.addBias(X, b), R))); }
  { const X = rand(5, 6, 'X', 2), R = proj(30);
    results.push(check(T, 'gelu', [X], () => T.sumProduct(T.gelu(X), R))); }
  { const X = rand(4, 8, 'X', 2), g = rand(1, 8, 'g'), b = rand(1, 8, 'b'), R = proj(32);
    results.push(check(T, 'layernorm', [X, g, b], () => T.sumProduct(T.layernorm(X, g, b), R))); }
  { const B = 2, Tn = 5, H = 2, d = 6; const X = rand(B * Tn, 3 * d, 'qkv', 1.5), R = proj(B * Tn * d);
    results.push(check(T, 'attention', [X], () => T.sumProduct(T.attention(X, B, Tn, H), R), { maxPerParam: 200 })); }
  { const W = rand(7, 4, 'W'), ids = Int32Array.from([3, 1, 3, 6, 0]), R = proj(20);
    results.push(check(T, 'embedding', [W], () => T.sumProduct(T.embedding(W, ids), R))); }
  { const X = rand(6, 5, 'logits', 2), tg = Int32Array.from([1, 4, 0, 2, 2, 3]), mask = Float64Array.from([1, 0, 1, 1, 0, 1]);
    results.push(check(T, 'crossEntropy(masked)', [X], () => T.crossEntropy(X, tg, mask))); }

  // Whole model: 2 layers, 2 heads, every parameter checked.
  {
    const TL = AMTransformerLib(T);
    const cfg = { n_layer: 2, n_head: 2, d_model: 8, n_ctx: 6, vocab_size: 7, init_std: 0.3 };
    const model = TL.init(cfg, 3);
    // perturb LN gains/biases away from 1/0 so their grads are exercised generically
    for (const p of model.list) if (p.name.includes('ln') || p.name.startsWith('b') || p.name.includes('.b'))
      for (let i = 0; i < p.size; i++) p.data[i] += 0.3 * r.normal();
    const B = 2, Tn = 5;
    const ids = Int32Array.from({ length: B * Tn }, () => r.int(7));
    const tg = Int32Array.from({ length: B * Tn }, () => r.int(7));
    const mask = Float64Array.from({ length: B * Tn }, (_, i) => (i % 5 >= 2 ? 1 : 0));
    const lossFn = () => T.crossEntropy(TL.forward(model, ids, B, Tn).logits, tg, mask);
    results.push(check(T, 'full transformer', model.list, lossFn, { maxPerParam: 25 }));
  }

  let ok = true;
  for (const res of results) {
    const pass = res.rel < 1e-5;
    ok = ok && pass;
    if (verbose) console.log(`  gradcheck ${res.name.padEnd(22)} max rel err ${res.rel.toExponential(2)}${pass ? '' : '  FAIL (' + res.at + ')'}`);
  }
  return { ok, results };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { ok } = runGradChecks();
  console.log(ok ? 'gradcheck: all passed' : 'gradcheck: FAILED');
  process.exit(ok ? 0 : 1);
}
