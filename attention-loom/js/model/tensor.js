/* The Attention Loom — minimal reverse-mode autograd over typed arrays.
 *
 * Everything lives inside ONE self-contained function declaration so the whole
 * library can be shipped into a Web Worker with Function.prototype.toString().
 * Do not reference anything outside AMTensorLib from inside it.
 *
 *   const T = AMTensorLib();                 // float32 storage (default)
 *   const T64 = AMTensorLib({float64: true}); // float64 storage (gradient checks)
 *
 * Tensors are 2-D row-major matrices [rows, cols]. A batch of sequences is
 * flattened to rows = B*T. Ops record a closure on a tape; backward(loss)
 * replays the tape in reverse, accumulating .grad on every tensor that needs it.
 */
function AMTensorLib(opts) {
  'use strict';
  opts = opts || {};
  const F = opts.float64 ? Float64Array : Float32Array;

  let tape = [];          // backward closures in creation order
  let gradEnabled = true; // false inside noGrad()

  class Tensor {
    constructor(rows, cols, data, requiresGrad) {
      this.rows = rows;
      this.cols = cols;
      this.data = data || new F(rows * cols);
      this.grad = null;
      this.requiresGrad = !!requiresGrad;
      this.name = '';
    }
    get shape() { return [this.rows, this.cols]; }
    get size() { return this.rows * this.cols; }
    ensureGrad() {
      if (!this.grad) this.grad = new F(this.rows * this.cols);
      return this.grad;
    }
    zeroGrad() { if (this.grad) this.grad.fill(0); }
  }

  function tensor(rows, cols, data) {
    if (data && !(data instanceof F)) data = F.from(data);
    return new Tensor(rows, cols, data, false);
  }
  function param(rows, cols, data, name) {
    if (data && !(data instanceof F)) data = F.from(data);
    const p = new Tensor(rows, cols, data, true);
    p.name = name || '';
    return p;
  }

  // Output tensor of an op. It needs a gradient if any input does.
  function out(rows, cols, inputs) {
    let rg = false;
    if (gradEnabled) for (const t of inputs) if (t.requiresGrad) { rg = true; break; }
    return new Tensor(rows, cols, null, rg);
  }
  function record(t, fn) { if (t.requiresGrad) tape.push(fn); }

  function noGrad(fn) {
    const prev = gradEnabled;
    gradEnabled = false;
    try { return fn(); } finally { gradEnabled = prev; }
  }
  function clearTape() { tape = []; }

  /* Reverse sweep. loss must be a 1x1 tensor. */
  function backward(loss) {
    loss.ensureGrad()[0] = 1;
    for (let i = tape.length - 1; i >= 0; i--) tape[i]();
    tape = [];
  }

  /* ---------- seeded randomness (mulberry32 + Box-Muller) ---------- */
  function rng(seed) {
    let s = (seed >>> 0) || 1;
    let spare = null;
    const next = () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0;
      while (u === 0) u = next();
      v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    const int = (n) => Math.floor(next() * n);
    return { next, normal, int };
  }

  /* ---------- core ops ---------- */

  // C[n,m] += A[n,k] · B[k,m]. i-k-j order so the inner loop walks rows of B and C
  // contiguously; k is unrolled by 4 so each C element is loaded/stored 4x less often.
  function matmulInto(C, A, B, n, k, m) {
    const k4 = k - (k % 4);
    for (let i = 0; i < n; i++) {
      const ci = i * m, ai = i * k;
      let p = 0;
      for (; p < k4; p += 4) {
        const a0 = A[ai + p], a1 = A[ai + p + 1], a2 = A[ai + p + 2], a3 = A[ai + p + 3];
        const b0 = p * m, b1 = b0 + m, b2 = b1 + m, b3 = b2 + m;
        for (let j = 0; j < m; j++) C[ci + j] += a0 * B[b0 + j] + a1 * B[b1 + j] + a2 * B[b2 + j] + a3 * B[b3 + j];
      }
      for (; p < k; p++) {
        const a = A[ai + p], bp = p * m;
        for (let j = 0; j < m; j++) C[ci + j] += a * B[bp + j];
      }
    }
  }
  // dA[n,k] += dC[n,m] · B[k,m]ᵀ  (each entry is a dot product of two contiguous rows;
  // four rows of B share one pass over the dC row)
  function matmulBackA(dA, dC, B, n, k, m) {
    const k4 = k - (k % 4);
    for (let i = 0; i < n; i++) {
      const ci = i * m, ai = i * k;
      let p = 0;
      for (; p < k4; p += 4) {
        const b0 = p * m, b1 = b0 + m, b2 = b1 + m, b3 = b2 + m;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let j = 0; j < m; j++) {
          const g = dC[ci + j];
          s0 += g * B[b0 + j]; s1 += g * B[b1 + j]; s2 += g * B[b2 + j]; s3 += g * B[b3 + j];
        }
        dA[ai + p] += s0; dA[ai + p + 1] += s1; dA[ai + p + 2] += s2; dA[ai + p + 3] += s3;
      }
      for (; p < k; p++) {
        const bp = p * m;
        let s = 0;
        for (let j = 0; j < m; j++) s += dC[ci + j] * B[bp + j];
        dA[ai + p] += s;
      }
    }
  }
  // dB[k,m] += A[n,k]ᵀ · dC[n,m]  (rank-1 row updates, four input rows at a time)
  function matmulBackB(dB, A, dC, n, k, m) {
    const n4 = n - (n % 4);
    let i = 0;
    for (; i < n4; i += 4) {
      const c0 = i * m, c1 = c0 + m, c2 = c1 + m, c3 = c2 + m;
      const a0 = i * k, a1 = a0 + k, a2 = a1 + k, a3 = a2 + k;
      for (let p = 0; p < k; p++) {
        const x0 = A[a0 + p], x1 = A[a1 + p], x2 = A[a2 + p], x3 = A[a3 + p];
        const bp = p * m;
        for (let j = 0; j < m; j++) dB[bp + j] += x0 * dC[c0 + j] + x1 * dC[c1 + j] + x2 * dC[c2 + j] + x3 * dC[c3 + j];
      }
    }
    for (; i < n; i++) {
      const ci = i * m, ai = i * k;
      for (let p = 0; p < k; p++) {
        const a = A[ai + p], bp = p * m;
        for (let j = 0; j < m; j++) dB[bp + j] += a * dC[ci + j];
      }
    }
  }

  function matmul(A, B) {
    if (A.cols !== B.rows) throw new Error('matmul shape ' + A.shape + ' x ' + B.shape);
    const n = A.rows, k = A.cols, m = B.cols;
    const C = out(n, m, [A, B]);
    matmulInto(C.data, A.data, B.data, n, k, m);
    record(C, () => {
      if (A.requiresGrad) matmulBackA(A.ensureGrad(), C.grad, B.data, n, k, m);
      if (B.requiresGrad) matmulBackB(B.ensureGrad(), A.data, C.grad, n, k, m);
    });
    return C;
  }

  // Y = X·W + b  (b is [1,m], broadcast over rows). Fused to save a pass.
  function linear(X, W, b) {
    const n = X.rows, k = X.cols, m = W.cols;
    if (k !== W.rows) throw new Error('linear shape ' + X.shape + ' x ' + W.shape);
    const Y = out(n, m, b ? [X, W, b] : [X, W]);
    const y = Y.data;
    if (b) for (let i = 0; i < n; i++) y.set(b.data, i * m);
    matmulInto(y, X.data, W.data, n, k, m);
    record(Y, () => {
      const g = Y.grad;
      if (X.requiresGrad) matmulBackA(X.ensureGrad(), g, W.data, n, k, m);
      if (W.requiresGrad) matmulBackB(W.ensureGrad(), X.data, g, n, k, m);
      if (b && b.requiresGrad) {
        const db = b.ensureGrad();
        for (let i = 0; i < n; i++) { const o = i * m; for (let j = 0; j < m; j++) db[j] += g[o + j]; }
      }
    });
    return Y;
  }

  function add(A, B) {
    if (A.size !== B.size) throw new Error('add shape');
    const C = out(A.rows, A.cols, [A, B]);
    const a = A.data, b = B.data, c = C.data;
    for (let i = 0; i < c.length; i++) c[i] = a[i] + b[i];
    record(C, () => {
      const g = C.grad;
      if (A.requiresGrad) { const d = A.ensureGrad(); for (let i = 0; i < g.length; i++) d[i] += g[i]; }
      if (B.requiresGrad) { const d = B.ensureGrad(); for (let i = 0; i < g.length; i++) d[i] += g[i]; }
    });
    return C;
  }

  // Y = X + b, b is [1, cols]
  function addBias(X, b) {
    const n = X.rows, m = X.cols;
    const Y = out(n, m, [X, b]);
    const x = X.data, y = Y.data, bb = b.data;
    for (let i = 0; i < n; i++) { const o = i * m; for (let j = 0; j < m; j++) y[o + j] = x[o + j] + bb[j]; }
    record(Y, () => {
      const g = Y.grad;
      if (X.requiresGrad) { const d = X.ensureGrad(); for (let i = 0; i < g.length; i++) d[i] += g[i]; }
      if (b.requiresGrad) {
        const d = b.ensureGrad();
        for (let i = 0; i < n; i++) { const o = i * m; for (let j = 0; j < m; j++) d[j] += g[o + j]; }
      }
    });
    return Y;
  }

  // GELU, tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
  const GELU_C = Math.sqrt(2 / Math.PI);
  function gelu(X) {
    const Y = out(X.rows, X.cols, [X]);
    const x = X.data, y = Y.data, n = x.length;
    const th = new F(n); // keep tanh(u) for the backward pass
    for (let i = 0; i < n; i++) {
      const v = x[i];
      const t = 1 - 2 / (Math.exp(2 * GELU_C * (v + 0.044715 * v * v * v)) + 1); // tanh(u), faster than Math.tanh
      th[i] = t;
      y[i] = 0.5 * v * (1 + t);
    }
    record(Y, () => {
      const g = Y.grad, d = X.ensureGrad();
      for (let i = 0; i < n; i++) {
        const v = x[i], t = th[i];
        // d/dx = 0.5(1+t) + 0.5·x·(1−t²)·c·(1 + 3·0.044715·x²)
        const dd = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * GELU_C * (1 + 0.134145 * v * v);
        d[i] += g[i] * dd;
      }
    });
    return Y;
  }

  // LayerNorm over each row: y = g ⊙ (x − μ)/√(σ² + ε) + b
  function layernorm(X, gain, bias, eps) {
    eps = eps || 1e-5;
    const n = X.rows, m = X.cols;
    const Y = out(n, m, [X, gain, bias]);
    const x = X.data, y = Y.data, g = gain.data, b = bias.data;
    const xhat = new F(n * m), rstd = new F(n);
    for (let i = 0; i < n; i++) {
      const o = i * m;
      let mu = 0;
      for (let j = 0; j < m; j++) mu += x[o + j];
      mu /= m;
      let v = 0;
      for (let j = 0; j < m; j++) { const d = x[o + j] - mu; v += d * d; }
      v /= m;
      const r = 1 / Math.sqrt(v + eps);
      rstd[i] = r;
      for (let j = 0; j < m; j++) {
        const h = (x[o + j] - mu) * r;
        xhat[o + j] = h;
        y[o + j] = h * g[j] + b[j];
      }
    }
    record(Y, () => {
      const dy = Y.grad;
      const dg = gain.requiresGrad ? gain.ensureGrad() : null;
      const db = bias.requiresGrad ? bias.ensureGrad() : null;
      const dx = X.requiresGrad ? X.ensureGrad() : null;
      for (let i = 0; i < n; i++) {
        const o = i * m;
        let s1 = 0, s2 = 0; // mean(dxhat), mean(dxhat·xhat)
        for (let j = 0; j < m; j++) {
          const gy = dy[o + j], h = xhat[o + j];
          if (dg) dg[j] += gy * h;
          if (db) db[j] += gy;
          const dh = gy * g[j];
          s1 += dh; s2 += dh * h;
        }
        if (!dx) continue;
        s1 /= m; s2 /= m;
        const r = rstd[i];
        // dx = rstd · (dxhat − mean(dxhat) − xhat · mean(dxhat · xhat))
        for (let j = 0; j < m; j++) dx[o + j] += r * (dy[o + j] * g[j] - s1 - xhat[o + j] * s2);
      }
    });
    return Y;
  }

  /* Fused causal multi-head self-attention.
   * qkv: [B*T, 3d] laid out as [Q | K | V], head h uses columns h*dh … (h+1)*dh of each.
   * returns [B*T, d] = concat_h softmax(Q_h K_hᵀ/√dh + causal mask) V_h
   * The attention probabilities are kept on the output as .attn (Float array of
   * shape [B, H, T, T], zero above the diagonal) for visualization.
   */
  function attention(qkv, B, T, H) {
    const d3 = qkv.cols, d = d3 / 3, dh = d / H;
    if (qkv.rows !== B * T || d % H !== 0) throw new Error('attention shape');
    const scale = 1 / Math.sqrt(dh);
    const O = out(B * T, d, [qkv]);
    const X = qkv.data, o = O.data;
    const P = new F(B * H * T * T);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        const qo = h * dh, ko = d + h * dh, vo = 2 * d + h * dh;
        const pb = (b * H + h) * T * T;
        for (let i = 0; i < T; i++) {
          const qi = (b * T + i) * d3 + qo;
          const prow = pb + i * T;
          // scores s_ij = q_i · k_j / √dh for j ≤ i
          let mx = -Infinity;
          for (let j = 0; j <= i; j++) {
            const kj = (b * T + j) * d3 + ko;
            let s = 0;
            for (let c = 0; c < dh; c++) s += X[qi + c] * X[kj + c];
            s *= scale;
            P[prow + j] = s;
            if (s > mx) mx = s;
          }
          let z = 0;
          for (let j = 0; j <= i; j++) { const e = Math.exp(P[prow + j] - mx); P[prow + j] = e; z += e; }
          const inv = 1 / z;
          const oi = (b * T + i) * d + h * dh;
          for (let j = 0; j <= i; j++) {
            const p = P[prow + j] * inv;
            P[prow + j] = p;
            const vj = (b * T + j) * d3 + vo;
            for (let c = 0; c < dh; c++) o[oi + c] += p * X[vj + c];
          }
        }
      }
    }
    O.attn = P;
    record(O, () => {
      const g = O.grad, dX = qkv.ensureGrad();
      const dP = new F(T);
      for (let b = 0; b < B; b++) {
        for (let h = 0; h < H; h++) {
          const qo = h * dh, ko = d + h * dh, vo = 2 * d + h * dh;
          const pb = (b * H + h) * T * T;
          for (let i = 0; i < T; i++) {
            const oi = (b * T + i) * d + h * dh;
            const qi = (b * T + i) * d3 + qo;
            const prow = pb + i * T;
            // dP_ij = dO_i · v_j ;  dV_j += P_ij dO_i
            let dot = 0;
            for (let j = 0; j <= i; j++) {
              const vj = (b * T + j) * d3 + vo;
              const p = P[prow + j];
              let s = 0;
              for (let c = 0; c < dh; c++) { s += g[oi + c] * X[vj + c]; dX[vj + c] += p * g[oi + c]; }
              dP[j] = s;
              dot += p * s;
            }
            // softmax backward: dS_ij = P_ij (dP_ij − Σ_j' P_ij' dP_ij')
            for (let j = 0; j <= i; j++) {
              const ds = P[prow + j] * (dP[j] - dot) * scale;
              if (ds === 0) continue;
              const kj = (b * T + j) * d3 + ko;
              for (let c = 0; c < dh; c++) {
                dX[qi + c] += ds * X[kj + c]; // dQ_i += dS_ij k_j / √dh
                dX[kj + c] += ds * X[qi + c]; // dK_j += dS_ij q_i / √dh
              }
            }
          }
        }
      }
    });
    return O;
  }

  // Row gather: Y[i] = W[ids[i]]
  function embedding(W, ids) {
    const n = ids.length, m = W.cols;
    const Y = out(n, m, [W]);
    const w = W.data, y = Y.data;
    for (let i = 0; i < n; i++) {
      const r = ids[i];
      if (r < 0 || r >= W.rows) throw new Error('embedding id out of range: ' + r);
      y.set(w.subarray(r * m, r * m + m), i * m);
    }
    record(Y, () => {
      const g = Y.grad, d = W.ensureGrad();
      for (let i = 0; i < n; i++) { const o = ids[i] * m, q = i * m; for (let j = 0; j < m; j++) d[o + j] += g[q + j]; }
    });
    return Y;
  }

  /* Softmax cross-entropy with a per-position loss mask.
   * loss = Σ_t mask_t · (−log softmax(logits_t)[target_t]) / Σ_t mask_t
   * Returns a 1x1 tensor; .probs keeps the softmax for accuracy bookkeeping.
   */
  function crossEntropy(logits, targets, mask) {
    const n = logits.rows, V = logits.cols, x = logits.data;
    const L = out(1, 1, [logits]);
    const probs = new F(n * V);
    let wsum = 0, total = 0;
    for (let i = 0; i < n; i++) {
      const w = mask ? mask[i] : 1;
      const o = i * V;
      let mx = -Infinity;
      for (let j = 0; j < V; j++) if (x[o + j] > mx) mx = x[o + j];
      let z = 0;
      for (let j = 0; j < V; j++) { const e = Math.exp(x[o + j] - mx); probs[o + j] = e; z += e; }
      for (let j = 0; j < V; j++) probs[o + j] /= z;
      if (w) {
        total += w * (Math.log(z) + mx - x[o + targets[i]]);
        wsum += w;
      }
    }
    const norm = wsum > 0 ? 1 / wsum : 0;
    L.data[0] = total * norm;
    L.probs = probs;
    record(L, () => {
      const g = L.grad[0] * norm, d = logits.ensureGrad();
      for (let i = 0; i < n; i++) {
        const w = mask ? mask[i] : 1;
        if (!w) continue;
        const o = i * V, gw = g * w;
        // ∂loss/∂logit = (softmax − onehot) · mask / Σmask
        for (let j = 0; j < V; j++) d[o + j] += gw * probs[o + j];
        d[o + targets[i]] -= gw;
      }
    });
    return L;
  }

  // Σ X ⊙ R for a constant array R (a random projection; used by gradient checks)
  function sumProduct(X, R) {
    const L = out(1, 1, [X]);
    let s = 0;
    for (let i = 0; i < X.data.length; i++) s += X.data[i] * R[i];
    L.data[0] = s;
    record(L, () => { const g = L.grad[0], d = X.ensureGrad(); for (let i = 0; i < R.length; i++) d[i] += g * R[i]; });
    return L;
  }

  /* ---------- optimisation ---------- */

  // Scale all gradients so their global L2 norm is at most maxNorm. Returns the norm.
  function clipGradNorm(params, maxNorm) {
    let ss = 0;
    for (const p of params) if (p.grad) for (let i = 0; i < p.grad.length; i++) ss += p.grad[i] * p.grad[i];
    const norm = Math.sqrt(ss);
    if (maxNorm && norm > maxNorm) {
      const s = maxNorm / (norm + 1e-6);
      for (const p of params) if (p.grad) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= s;
    }
    return norm;
  }

  /* AdamW (decoupled weight decay). params: Tensor[]; p.noDecay = true skips decay. */
  class AdamW {
    constructor(params, o) {
      o = o || {};
      this.params = params;
      this.lr = o.lr != null ? o.lr : 1e-3;
      this.beta1 = o.beta1 != null ? o.beta1 : 0.9;
      this.beta2 = o.beta2 != null ? o.beta2 : 0.99;
      this.eps = o.eps != null ? o.eps : 1e-8;
      this.weightDecay = o.weightDecay != null ? o.weightDecay : 0.01;
      this.t = 0;
      this.m = params.map(p => new F(p.size));
      this.v = params.map(p => new F(p.size));
    }
    step(lr) {
      lr = lr != null ? lr : this.lr;
      this.t++;
      const b1 = this.beta1, b2 = this.beta2, eps = this.eps;
      const c1 = 1 / (1 - Math.pow(b1, this.t)), c2 = 1 / (1 - Math.pow(b2, this.t));
      for (let k = 0; k < this.params.length; k++) {
        const p = this.params[k];
        if (!p.grad) continue;
        const w = p.data, g = p.grad, m = this.m[k], v = this.v[k];
        const decay = p.noDecay ? 0 : lr * this.weightDecay;
        for (let i = 0; i < w.length; i++) {
          const gi = g[i];
          m[i] = b1 * m[i] + (1 - b1) * gi;
          v[i] = b2 * v[i] + (1 - b2) * gi * gi;
          w[i] -= decay * w[i] + lr * (m[i] * c1) / (Math.sqrt(v[i] * c2) + eps);
        }
      }
    }
    zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  }

  /* ---------- base64 <-> typed arrays (float32 little-endian) ---------- */
  const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  function b64ToFloat32(s) {
    const lut = new Uint8Array(128);
    for (let i = 0; i < 64; i++) lut[B64.charCodeAt(i)] = i;
    let len = s.length;
    while (len > 0 && s[len - 1] === '=') len--;
    const bytes = new Uint8Array((len * 3) >> 2);
    let bi = 0;
    for (let i = 0; i < len; i += 4) {
      const a = lut[s.charCodeAt(i)], b = lut[s.charCodeAt(i + 1)];
      const c = i + 2 < len ? lut[s.charCodeAt(i + 2)] : 0, d = i + 3 < len ? lut[s.charCodeAt(i + 3)] : 0;
      const n = (a << 18) | (b << 12) | (c << 6) | d;
      if (bi < bytes.length) bytes[bi++] = (n >> 16) & 255;
      if (bi < bytes.length) bytes[bi++] = (n >> 8) & 255;
      if (bi < bytes.length) bytes[bi++] = n & 255;
    }
    // copy through DataView so the result is little-endian regardless of platform
    const dv = new DataView(bytes.buffer), outArr = new Float32Array(bytes.length >> 2);
    for (let i = 0; i < outArr.length; i++) outArr[i] = dv.getFloat32(i * 4, true);
    return outArr;
  }
  function float32ToB64(arr) {
    const bytes = new Uint8Array(arr.length * 4), dv = new DataView(bytes.buffer);
    for (let i = 0; i < arr.length; i++) dv.setFloat32(i * 4, arr[i], true);
    let s = '';
    for (let i = 0; i < bytes.length; i += 3) {
      const a = bytes[i], b = i + 1 < bytes.length ? bytes[i + 1] : 0, c = i + 2 < bytes.length ? bytes[i + 2] : 0;
      const n = (a << 16) | (b << 8) | c;
      s += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] +
        (i + 1 < bytes.length ? B64[(n >> 6) & 63] : '=') + (i + 2 < bytes.length ? B64[n & 63] : '=');
    }
    return s;
  }

  const api = {
    F, float64: !!opts.float64, Tensor, tensor, param, rng,
    matmul, linear, add, addBias, gelu, layernorm, attention, embedding, crossEntropy, sumProduct,
    backward, noGrad, clearTape, clipGradNorm, AdamW,
    b64ToFloat32, float32ToB64,
    get tapeLength() { return tape.length; },
  };
  return api;
}
if (typeof module !== 'undefined') module.exports = AMTensorLib;
