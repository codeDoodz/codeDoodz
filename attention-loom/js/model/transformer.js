/* The Attention Loom — a GPT-style decoder-only transformer on top of AMTensorLib.
 *
 * Self-contained: AMTransformerLib(T) takes a tensor library instance and
 * references nothing else, so it can be stringified into a Web Worker.
 *
 * Architecture (pre-LayerNorm, like GPT-2):
 *   x_0      = wte[token] + wpe[position]                    (learned positions)
 *   for each layer l:
 *     a      = Attn(LN1(x_l))      causal, n_head heads, scores scaled by 1/√d_head
 *     x_mid  = x_l + a                                       (residual add)
 *     m      = W_proj · GELU(W_fc · LN2(x_mid) + b_fc) + b_proj    (d_ff = 4·d_model)
 *     x_l+1  = x_mid + m
 *   logits   = LN_f(x_L) · W_out + b_out                     (untied unembedding)
 *
 * Weight layout: every matrix is stored [in, out] so a layer is y = x·W + b.
 * The fused QKV matrix is [d, 3d] with columns [Q | K | V]; head h owns columns
 * h·d_head … (h+1)·d_head inside each third.
 */
function AMTransformerLib(T) {
  'use strict';

  const defaultConfig = { n_layer: 2, n_head: 2, d_model: 32, d_ff: 128, n_ctx: 16, vocab_size: 11 };

  // Every parameter: name, shape, initialisation.
  function paramSpecs(cfg) {
    const d = cfg.d_model, f = cfg.d_ff || 4 * d, V = cfg.vocab_size, L = cfg.n_layer;
    const std = cfg.init_std || 0.02;
    const projStd = std / Math.sqrt(2 * L); // GPT-2: shrink residual-branch outputs
    const s = [
      { name: 'wte', shape: [V, d], init: 'normal', std, noDecay: true },
      { name: 'wpe', shape: [cfg.n_ctx, d], init: cfg.pos_init === 'sin' ? 'sin' : 'normal', std, noDecay: true },
    ];
    for (let l = 0; l < L; l++) {
      const p = 'h.' + l + '.';
      s.push(
        { name: p + 'ln1.g', shape: [1, d], init: 'ones', noDecay: true },
        { name: p + 'ln1.b', shape: [1, d], init: 'zeros', noDecay: true },
        { name: p + 'attn.wqkv', shape: [d, 3 * d], init: 'normal', std },
        { name: p + 'attn.bqkv', shape: [1, 3 * d], init: 'zeros', noDecay: true },
        { name: p + 'attn.wo', shape: [d, d], init: 'normal', std: projStd },
        { name: p + 'attn.bo', shape: [1, d], init: 'zeros', noDecay: true },
        { name: p + 'ln2.g', shape: [1, d], init: 'ones', noDecay: true },
        { name: p + 'ln2.b', shape: [1, d], init: 'zeros', noDecay: true },
        { name: p + 'mlp.wfc', shape: [d, f], init: 'normal', std },
        { name: p + 'mlp.bfc', shape: [1, f], init: 'zeros', noDecay: true },
        { name: p + 'mlp.wproj', shape: [f, d], init: 'normal', std: projStd },
        { name: p + 'mlp.bproj', shape: [1, d], init: 'zeros', noDecay: true },
      );
    }
    s.push(
      { name: 'lnf.g', shape: [1, d], init: 'ones', noDecay: true },
      { name: 'lnf.b', shape: [1, d], init: 'zeros', noDecay: true },
      { name: 'wout', shape: [d, V], init: 'normal', std },
      { name: 'bout', shape: [1, V], init: 'zeros', noDecay: true },
    );
    return s;
  }

  function normConfig(config) {
    const c = Object.assign({}, defaultConfig, config);
    if (!config.d_ff) c.d_ff = 4 * c.d_model;
    if (c.d_model % c.n_head) throw new Error('d_model must be divisible by n_head');
    return c;
  }

  function build(cfg, getData) {
    const specs = paramSpecs(cfg);
    const params = {}, list = [];
    for (const sp of specs) {
      const p = T.param(sp.shape[0], sp.shape[1], getData(sp), sp.name);
      p.noDecay = !!sp.noDecay;
      params[sp.name] = p;
      list.push(p);
    }
    return { config: cfg, params, list };
  }

  // Fresh random model (seeded, reproducible).
  function init(config, seed) {
    const cfg = normConfig(config);
    const r = T.rng(seed == null ? 1234 : seed);
    return build(cfg, (sp) => {
      const n = sp.shape[0] * sp.shape[1], a = new T.F(n);
      if (sp.init === 'ones') a.fill(1);
      else if (sp.init === 'normal') for (let i = 0; i < n; i++) a[i] = r.normal() * sp.std;
      else if (sp.init === 'sin') {
        // learned positions, started from sinusoids (RMS = std): position shifts begin as rotations
        const d = sp.shape[1];
        for (let p = 0; p < sp.shape[0]; p++) for (let i = 0; i < d; i++) {
          const ang = p / Math.pow(10000, (2 * Math.floor(i / 2)) / d);
          a[p * d + i] = (i % 2 === 0 ? Math.sin(ang) : Math.cos(ang)) * sp.std * Math.SQRT2;
        }
      }
      return a;
    });
  }

  // Model from stored weights: tensors[name] = typed array or {shape, data} or {shape, b64}.
  function fromWeights(config, tensors) {
    const cfg = normConfig(config);
    return build(cfg, (sp) => {
      let t = tensors[sp.name];
      if (!t) throw new Error('missing weight ' + sp.name);
      if (t.b64) t = T.b64ToFloat32(t.b64);
      else if (t.data) t = t.data;
      const n = sp.shape[0] * sp.shape[1];
      if (t.length !== n) throw new Error('weight ' + sp.name + ' has ' + t.length + ' values, expected ' + n);
      return T.F.from(t);
    });
  }

  function exportWeights(model, asB64) {
    const out = {};
    for (const p of model.list) {
      out[p.name] = asB64 ? { shape: [p.rows, p.cols], b64: T.float32ToB64(p.data) }
        : { shape: [p.rows, p.cols], data: Float32Array.from(p.data) };
    }
    return out;
  }

  function countParams(model) { return model.list.reduce((s, p) => s + p.size, 0); }

  /* Forward pass on a batch: ids is Int32Array(B*Tn), row-major [B, Tn].
   * Returns {logits, cap?}. With opts.capture, cap holds the intermediate tensors. */
  function forward(model, ids, B, Tn, opts) {
    const cfg = model.config, P = model.params, H = cfg.n_head;
    if (Tn > cfg.n_ctx) throw new Error('sequence longer than n_ctx');
    const capture = opts && opts.capture;
    const pos = new Int32Array(B * Tn);
    for (let b = 0; b < B; b++) for (let t = 0; t < Tn; t++) pos[b * Tn + t] = t;
    let x = T.add(T.embedding(P.wte, ids), T.embedding(P.wpe, pos));
    const cap = capture ? { resid: [], residMid: [], qkv: [], attn: [], mlp: [] } : null;
    for (let l = 0; l < cfg.n_layer; l++) {
      const p = 'h.' + l + '.';
      if (cap) cap.resid.push(x);
      const h1 = T.layernorm(x, P[p + 'ln1.g'], P[p + 'ln1.b']);
      const qkv = T.linear(h1, P[p + 'attn.wqkv'], P[p + 'attn.bqkv']);
      const a = T.attention(qkv, B, Tn, H);
      x = T.add(x, T.linear(a, P[p + 'attn.wo'], P[p + 'attn.bo']));
      if (cap) { cap.qkv.push(qkv); cap.attn.push(a.attn); cap.residMid.push(x); }
      const h2 = T.layernorm(x, P[p + 'ln2.g'], P[p + 'ln2.b']);
      const act = T.gelu(T.linear(h2, P[p + 'mlp.wfc'], P[p + 'mlp.bfc']));
      if (cap) cap.mlp.push(act);
      x = T.add(x, T.linear(act, P[p + 'mlp.wproj'], P[p + 'mlp.bproj']));
    }
    if (cap) cap.resid.push(x);
    const hf = T.layernorm(x, P['lnf.g'], P['lnf.b']);
    const logits = T.linear(hf, P.wout, P.bout);
    return { logits, cap };
  }

  /* Trainer: AdamW + linear warmup + cosine decay + global-norm clipping.
   * batch = {ids, targets, mask, B, T}. step() returns {loss, probs, gradNorm, lr}. */
  function createTrainer(model, o) {
    o = Object.assign({ lr: 3e-3, minLr: 3e-4, warmup: 50, total: 2000, weightDecay: 0.01, clip: 1.0,
      beta1: 0.9, beta2: 0.99 }, o || {});
    const opt = new T.AdamW(model.list, { lr: o.lr, weightDecay: o.weightDecay, beta1: o.beta1, beta2: o.beta2 });
    let it = 0;
    function lrAt(i) {
      if (i < o.warmup) return o.lr * (i + 1) / o.warmup;
      const q = Math.min(1, (i - o.warmup) / Math.max(1, o.total - o.warmup));
      return o.minLr + 0.5 * (o.lr - o.minLr) * (1 + Math.cos(Math.PI * q));
    }
    function step(batch) {
      opt.zeroGrad();
      T.clearTape();
      const { logits } = forward(model, batch.ids, batch.B, batch.T);
      const loss = T.crossEntropy(logits, batch.targets, batch.mask);
      T.backward(loss);
      const gradNorm = T.clipGradNorm(model.list, o.clip);
      const lr = lrAt(it);
      opt.step(lr);
      it++;
      return { loss: loss.data[0], probs: loss.probs, gradNorm, lr, step: it };
    }
    return { step, opt, options: o, get iteration() { return it; }, lrAt };
  }

  function softmaxRow(src, off, n) {
    const p = new Float32Array(n);
    let mx = -Infinity;
    for (let j = 0; j < n; j++) if (src[off + j] > mx) mx = src[off + j];
    let z = 0;
    for (let j = 0; j < n; j++) { const e = Math.exp(src[off + j] - mx); p[j] = e; z += e; }
    for (let j = 0; j < n; j++) p[j] /= z;
    return p;
  }
  function rows(t, Tn) {
    const outRows = [], m = t.cols;
    for (let i = 0; i < Tn; i++) outRows.push(Float32Array.from(t.data.subarray(i * m, i * m + m)));
    return outRows;
  }

  /* Inference on ONE sequence (array of ids, length ≤ n_ctx). Without capture it
   * returns {logits, probs}; with capture it adds every introspection field of
   * CONTRACT §6 (attn, resid, residMid, q, k, v, mlp, lens). */
  function run(model, ids, opts) {
    const cfg = model.config, Tn = ids.length, V = cfg.vocab_size;
    const H = cfg.n_head, d = cfg.d_model, dh = d / H;
    const capture = !!(opts && opts.capture);
    return T.noGrad(() => {
      const { logits, cap } = forward(model, Int32Array.from(ids), 1, Tn, { capture });
      const res = { logits: [], probs: [] };
      for (let t = 0; t < Tn; t++) {
        res.logits.push(Float32Array.from(logits.data.subarray(t * V, t * V + V)));
        res.probs.push(softmaxRow(logits.data, t * V, V));
      }
      if (!capture) return res;
      res.resid = cap.resid.map(x => rows(x, Tn));
      res.residMid = cap.residMid.map(x => rows(x, Tn));
      res.mlp = cap.mlp.map(x => rows(x, Tn));
      res.attn = cap.attn.map(P => {
        const perHead = [];
        for (let h = 0; h < H; h++) {
          const qs = [];
          for (let i = 0; i < Tn; i++) qs.push(Float32Array.from(P.subarray((h * Tn + i) * Tn, (h * Tn + i + 1) * Tn)));
          perHead.push(qs);
        }
        return perHead;
      });
      res.q = []; res.k = []; res.v = [];
      for (const qkv of cap.qkv) {
        const Q = [], K = [], Vv = [];
        for (let h = 0; h < H; h++) {
          const qh = [], kh = [], vh = [];
          for (let t = 0; t < Tn; t++) {
            const o = t * 3 * d + h * dh;
            qh.push(Float32Array.from(qkv.data.subarray(o, o + dh)));
            kh.push(Float32Array.from(qkv.data.subarray(o + d, o + d + dh)));
            vh.push(Float32Array.from(qkv.data.subarray(o + 2 * d, o + 2 * d + dh)));
          }
          Q.push(qh); K.push(kh); Vv.push(vh);
        }
        res.q.push(Q); res.k.push(K); res.v.push(Vv);
      }
      // Logit lens: decode every residual snapshot through the final LN + unembedding.
      const P = model.params;
      res.lens = cap.resid.map(x => {
        const lg = T.linear(T.layernorm(x, P['lnf.g'], P['lnf.b']), P.wout, P.bout);
        const r = [];
        for (let t = 0; t < Tn; t++) r.push(softmaxRow(lg.data, t * V, V));
        return r;
      });
      return res;
    });
  }

  return { defaultConfig, paramSpecs, normConfig, init, fromWeights, exportWeights, countParams, forward, createTrainer, run };
}
if (typeof module !== 'undefined') module.exports = AMTransformerLib;
