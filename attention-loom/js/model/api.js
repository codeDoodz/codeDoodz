/* The Attention Loom — AM.model: the real tiny transformers, ready to poke.
 *
 * Load order (classic scripts, file:// friendly):
 *   tensor.js, transformer.js, weights-*.js, api.js
 *
 *   const m = AM.model.get('tinyworld');
 *   const {ids} = m.encode('the queen opened the door because she');
 *   const r = m.run(ids, {capture: true});   // see CONTRACT §6 for every field
 *
 * Also here:
 *   AM.model.lib.AMTaskLib       — the reverse/sort task definitions + random example generator
 *   AM.model.createTrainerWorker — trains reverse/sort from scratch in a Web Worker (or, if
 *                                  workers are unavailable, in small main-thread slices) and
 *                                  streams loss/accuracy/attention snapshots.
 */
(function (root) {
  'use strict';

  /* ======================================================================
   * Algorithmic tasks (self-contained: also shipped into the worker).
   *   reverse  "38152907>70925183"   output = input reversed
   *   sort     "73519273>12335779"   output = input sorted ascending (repeats allowed)
   * Input length is fixed at 8 digits. Tokens: '0'…'9' = ids 0…9, '>' = 10.
   * The model reads the first 16 tokens and is trained (loss mask) only on the
   * 8 positions whose next token is an output digit: position 8 ('>') predicts
   * the first output digit, …, position 15 predicts the last.
   * ==================================================================== */
  function AMTaskLib() {
    const VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '>'];
    const SEP = 10, LEN = 8;
    const TASKS = {
      reverse: {
        name: 'reverse', len: LEN, vocab: VOCAB,
        fn: (xs) => xs.slice().reverse(),
        model: { n_layer: 1, n_head: 1, d_model: 24, d_ff: 96, n_ctx: 2 * LEN, vocab_size: VOCAB.length },
        train: { lr: 3e-3, minLr: 3e-4, warmup: 30, total: 600, batch: 32, weightDecay: 0.01, clip: 1 },
        probe: [3, 8, 1, 5, 2, 9, 0, 7],
      },
      sort: {
        name: 'sort', len: LEN, vocab: VOCAB,
        fn: (xs) => xs.slice().sort((a, b) => a - b),
        model: { n_layer: 2, n_head: 2, d_model: 32, d_ff: 128, n_ctx: 2 * LEN, vocab_size: VOCAB.length },
        train: { lr: 3e-3, minLr: 3e-4, warmup: 30, total: 3000, batch: 32, weightDecay: 0.01, clip: 1 },
        probe: [7, 3, 5, 1, 9, 2, 7, 3],
      },
    };
    function task(name) { const t = TASKS[name]; if (!t) throw new Error('unknown task ' + name); return t; }

    // One example: full sequence input + '>' + output (2·LEN+1 tokens).
    function example(name, rng, digits) {
      const t = task(name);
      const xs = digits ? digits.slice() : Array.from({ length: t.len }, () => rng.int(10));
      const ys = t.fn(xs);
      return { input: xs, output: ys, seq: xs.concat([SEP], ys) };
    }
    // A training batch in the layout transformer.forward/createTrainer expect.
    function batchFrom(name, seqs) {
      const t = task(name), Tn = 2 * t.len, B = seqs.length;
      const ids = new Int32Array(B * Tn), targets = new Int32Array(B * Tn), mask = new Float32Array(B * Tn);
      for (let b = 0; b < B; b++) {
        const s = seqs[b];
        for (let i = 0; i < Tn; i++) {
          ids[b * Tn + i] = s[i];
          targets[b * Tn + i] = s[i + 1];
          mask[b * Tn + i] = i >= t.len ? 1 : 0; // loss only where the next token is an output digit
        }
      }
      return { ids, targets, mask, B, T: Tn };
    }
    function batch(name, rng, B) {
      const seqs = [];
      for (let b = 0; b < B; b++) seqs.push(example(name, rng).seq);
      return batchFrom(name, seqs);
    }
    // Score flat logits [B*T, V] against a batch: per-digit and exact-sequence accuracy.
    // (Teacher-forced argmax being right at every output slot == greedy decoding is exactly right.)
    function score(name, logits, bt) {
      const t = task(name), V = t.vocab.length, Tn = bt.T;
      let tok = 0, tokOk = 0, seqOk = 0;
      for (let b = 0; b < bt.B; b++) {
        let ok = true;
        for (let i = t.len; i < Tn; i++) {
          const r = (b * Tn + i) * V;
          let am = 0;
          for (let j = 1; j < V; j++) if (logits[r + j] > logits[r + am]) am = j;
          tok++;
          if (am === bt.targets[b * Tn + i]) tokOk++; else ok = false;
        }
        if (ok) seqOk++;
      }
      return { tokenAcc: tokOk / tok, seqAcc: seqOk / bt.B };
    }
    function format(ids) { return ids.map(i => VOCAB[i]).join(''); }
    return { VOCAB, SEP, LEN, TASKS, task, example, batch, batchFrom, score, format };
  }

  /* ======================================================================
   * Trainer main loop. Runs inside a Web Worker (scope = self), or on the main
   * thread with a fake scope. Uses AMTensorLib / AMTransformerLib / AMTaskLib,
   * which are globals in both places.
   *
   * in : {type:'init', task, opts} | {type:'start'} | {type:'pause'} | {type:'reset', opts?}
   *      {type:'step', n} | {type:'set', lr?, delayMs?}
   * out: {type:'ready', task, config, params, probe:{ids, text}}
   *      {type:'progress', step, loss, acc, tokenAcc, attn, probePred, probeTarget, lr, elapsed, wall, stepsPerSec}
 *   acc/tokenAcc: exact-sequence / per-digit accuracy on a fixed held-out batch (refreshed every evalMs
 *   of training time); elapsed = seconds spent training; wall = seconds since (re)start incl. overhead.
   *      {type:'paused'|'done', step}
   *   attn[layer][head] = Float32Array(T*T), row q, column k (causal; rows sum to 1),
   *   for the fixed probe input of the task.
   * ==================================================================== */
  function AMTrainerMain(scope) {
    const T = AMTensorLib();
    const TL = AMTransformerLib(T);
    const tasks = AMTaskLib();
    const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    let S = null, running = false, timer = null;

    function setup(taskName, opts) {
      opts = opts || {};
      const t = tasks.task(taskName);
      const seed = opts.seed != null ? opts.seed : 1;
      const config = Object.assign({}, t.model, opts.config || {});
      const tr = Object.assign({}, t.train, opts.train || {});
      if (opts.lr != null) tr.lr = opts.lr;
      if (opts.batch != null) tr.batch = opts.batch;
      const model = TL.init(config, seed);
      const probeSeq = tasks.example(taskName, null, t.probe).seq;
      const evalRng = T.rng(seed + 999);
      const evalSeqs = [];
      for (let i = 0; i < (opts.evalSize || 200); i++) evalSeqs.push(tasks.example(taskName, evalRng).seq);
      S = {
        task: taskName, opts, t, model, trainer: TL.createTrainer(model, tr), tr, rng: T.rng(seed),
        probeSeq, evalBatch: tasks.batchFrom(taskName, evalSeqs), step: 0, lastLoss: NaN,
        trainMs: 0, chunkMs: opts.chunkMs || 50, reportMs: opts.reportMs || 150, lastReport: -1e9,
        delayMs: opts.delayMs || 0, maxSteps: opts.maxSteps || tr.total, lossEma: null,
        evalMs: opts.evalMs != null ? opts.evalMs : 500, lastEval: 0, lastScore: null, t0: now(),
      };
      scope.postMessage({ type: 'ready', task: taskName, config: model.config, params: TL.countParams(model),
        train: tr, probe: { ids: probeSeq.slice(0, 2 * t.len), text: tasks.format(probeSeq) } });
      report();
    }

    function snapshot() {
      const t = S.t, Tn = 2 * t.len, H = S.model.config.n_head;
      return T.noGrad(() => {
        const ids = Int32Array.from(S.probeSeq.slice(0, Tn));
        const { logits, cap } = TL.forward(S.model, ids, 1, Tn, { capture: true });
        const attn = cap.attn.map(P => { const hs = []; for (let h = 0; h < H; h++) hs.push(Float32Array.from(P.subarray(h * Tn * Tn, (h + 1) * Tn * Tn))); return hs; });
        const V = t.vocab.length, pred = [];
        for (let i = t.len; i < Tn; i++) {
          let am = 0;
          for (let j = 1; j < V; j++) if (logits.data[i * V + j] > logits.data[i * V + am]) am = j;
          pred.push(am);
        }
        // held-out accuracy is the expensive part, so it is refreshed at most every evalMs of training
        if (!S.lastScore || S.trainMs - S.lastEval >= S.evalMs || S.step >= S.maxSteps) {
          const ev = TL.forward(S.model, S.evalBatch.ids, S.evalBatch.B, S.evalBatch.T).logits;
          S.lastScore = tasks.score(S.task, ev.data, S.evalBatch);
          S.lastEval = S.trainMs;
        }
        return { attn, pred, sc: S.lastScore };
      });
    }

    function report() {
      const snap = snapshot();
      S.lastReport = S.trainMs;
      scope.postMessage({
        type: 'progress', task: S.task, step: S.step, loss: S.lossEma == null ? NaN : S.lossEma,
        acc: snap.sc.seqAcc, tokenAcc: snap.sc.tokenAcc, attn: snap.attn,
        probePred: tasks.format(snap.pred), probeTarget: tasks.format(S.probeSeq.slice(S.t.len + 1)),
        lr: S.trainer.lrAt(Math.max(0, S.step - 1)), elapsed: S.trainMs / 1000, wall: (now() - S.t0) / 1000,
        stepsPerSec: S.trainMs > 0 ? S.step / (S.trainMs / 1000) : 0,
      });
    }

    function trainSteps(n) {
      for (let i = 0; i < n; i++) {
        const t0 = now();
        const res = S.trainer.step(tasks.batch(S.task, S.rng, S.tr.batch));
        S.trainMs += now() - t0;
        S.step++;
        S.lossEma = S.lossEma == null ? res.loss : 0.9 * S.lossEma + 0.1 * res.loss;
      }
    }

    function tick() {
      timer = null;
      if (!running || !S) return;
      const t0 = now();
      while (running && S.step < S.maxSteps && now() - t0 < S.chunkMs) {
        trainSteps(1);
        if (S.delayMs) break; // slow-motion: one step per tick
      }
      if (S.trainMs - S.lastReport >= S.reportMs || S.delayMs || S.step >= S.maxSteps) report();
      if (S.step >= S.maxSteps) { running = false; scope.postMessage({ type: 'done', step: S.step }); return; }
      timer = setTimeout(tick, S.delayMs || 0);
    }

    scope.onmessage = function (e) {
      const m = e.data || {};
      if (m.type === 'init') { running = false; setup(m.task, m.opts); }
      else if (m.type === 'reset') { running = false; if (timer) clearTimeout(timer); timer = null; setup(m.task || S.task, m.opts || S.opts); }
      else if (m.type === 'start') { if (S && !running) { running = true; if (!timer) timer = setTimeout(tick, 0); } }
      else if (m.type === 'pause') { running = false; if (S) scope.postMessage({ type: 'paused', step: S.step }); }
      else if (m.type === 'step') { if (S) { trainSteps(m.n || 1); report(); } }
      else if (m.type === 'set') {
        if (!S) return;
        if (m.lr != null) { S.trainer.options.lr = m.lr; S.trainer.options.minLr = Math.min(S.trainer.options.minLr, m.lr); }
        if (m.delayMs != null) S.delayMs = m.delayMs;
        if (m.maxSteps != null) S.maxSteps = m.maxSteps;
      }
    };
  }

  /* ======================================================================
   * Model objects
   * ==================================================================== */
  const W = root.AM_WEIGHTS || {};
  const hasLibs = typeof AMTensorLib === 'function' && typeof AMTransformerLib === 'function';
  const T = hasLibs ? AMTensorLib() : null;
  const TL = hasLibs ? AMTransformerLib(T) : null;
  const cache = {};

  function makeModel(name) {
    const w = W[name];
    if (!w) throw new Error('AM.model: no weights for "' + name + '" (is weights-' + name + '.js loaded?)');
    const net = TL.fromWeights(w.config, w.tensors);
    const config = Object.assign({}, net.config);
    delete config.init_std;
    const vocab = w.vocab.slice();
    const idOf = new Map(vocab.map((t, i) => [t, i]));
    const kind = (w.meta && w.meta.kind) || 'word';
    const unk = idOf.has('<unk>') ? idOf.get('<unk>') : -1;
    // Tokens a sampler should never emit.
    const banned = new Set(['<pad>', '<unk>'].filter(t => idOf.has(t)).map(t => idOf.get(t)));

    function encode(text) {
      const ids = [], tokens = [], unknown = [];
      if (kind === 'char') {
        for (const ch of String(text).replace(/\s+/g, '')) {
          if (idOf.has(ch)) { ids.push(idOf.get(ch)); tokens.push(ch); } else unknown.push(ch);
        }
        return { ids, tokens, unknown };
      }
      const words = String(text).toLowerCase().replace(/([.,])/g, ' $1 ').split(/\s+/).filter(Boolean);
      for (const wd of words) {
        if (idOf.has(wd)) { ids.push(idOf.get(wd)); tokens.push(wd); }
        else { unknown.push(wd); if (unk >= 0) { ids.push(unk); tokens.push('<unk>'); } }
      }
      return { ids, tokens, unknown };
    }
    function decode(ids) { return Array.from(ids, i => vocab[i]); }

    function run(ids, opts) {
      ids = Array.from(ids);
      let truncated = false;
      if (ids.length > config.n_ctx) { ids = ids.slice(0, config.n_ctx); truncated = true; }
      if (!ids.length) throw new Error('AM.model.run: empty input');
      const r = TL.run(net, ids, opts);
      r.ids = ids;
      r.tokens = decode(ids);
      if (truncated) r.truncated = true;
      return r;
    }

    function topk(row, k) {
      const idx = Array.from(row.keys()).sort((a, b) => row[b] - row[a]).slice(0, k || 5);
      return idx.map(id => ({ id, token: vocab[id], p: row[id] }));
    }

    // Sample an id from a probability row. temperature rescales p^(1/τ) (≡ logits/τ);
    // topK keeps the k most likely; topP keeps the smallest set with mass ≥ p.
    function sample(row, o) {
      o = o || {};
      const temp = o.temperature != null ? o.temperature : 1;
      const rng = o.rng || Math.random;
      const n = row.length;
      let ids = [];
      for (let i = 0; i < n; i++) if (!banned.has(i) && row[i] > 0) ids.push(i);
      if (!ids.length) return 0;
      ids.sort((a, b) => row[b] - row[a]);
      if (temp <= 1e-6) return ids[0]; // greedy
      let w = ids.map(i => Math.exp(Math.log(row[i]) / temp));
      if (o.topK && o.topK > 0 && o.topK < ids.length) { ids = ids.slice(0, o.topK); w = w.slice(0, o.topK); }
      let z = w.reduce((s, x) => s + x, 0);
      if (o.topP != null && o.topP < 1) {
        let acc = 0, cut = ids.length;
        for (let i = 0; i < ids.length; i++) { acc += w[i] / z; if (acc >= o.topP) { cut = i + 1; break; } }
        ids = ids.slice(0, cut); w = w.slice(0, cut);
        z = w.reduce((s, x) => s + x, 0);
      }
      let u = rng() * z;
      for (let i = 0; i < ids.length; i++) { u -= w[i]; if (u <= 0) return ids[i]; }
      return ids[ids.length - 1];
    }

    // Autoregressive generation. Returns the FULL id sequence (prompt + new tokens).
    function generate(ids, o) {
      o = o || {};
      const out = Array.from(ids);
      const maxNew = o.maxNew != null ? o.maxNew : 20;
      let stop = o.stopAt === undefined ? (kind === 'word' ? '.' : null) : o.stopAt;
      const stopIds = new Set((stop == null ? [] : [].concat(stop)).filter(s => idOf.has(s)).map(s => idOf.get(s)));
      for (let k = 0; k < maxNew; k++) {
        const ctx = out.slice(-config.n_ctx);
        const r = TL.run(net, ctx);
        const id = sample(r.probs[ctx.length - 1], o);
        out.push(id);
        if (stopIds.has(id)) break;
      }
      return out;
    }

    return { name, kind, config, vocab, meta: w.meta || {}, encode, decode, run, topk, sample, generate,
      tokenId: (t) => (idOf.has(t) ? idOf.get(t) : -1), _net: net };
  }

  function get(name) {
    if (!hasLibs) throw new Error('AM.model: tensor.js and transformer.js must load before api.js');
    if (!cache[name]) cache[name] = makeModel(name);
    return cache[name];
  }
  function list() { return Object.keys(W); }

  /* ----- trainer worker ----- */
  function workerSource() {
    return [
      AMTensorLib.toString(), AMTransformerLib.toString(), AMTaskLib.toString(), AMTrainerMain.toString(),
      'AMTrainerMain(self);',
    ].join('\n;\n');
  }

  // Main-thread stand-in with the same interface as a Worker.
  function inlineWorker() {
    const fake = { onmessage: null, _listeners: [], terminated: false };
    const inner = {
      postMessage(msg) { setTimeout(() => { if (fake.terminated) return; const ev = { data: msg }; if (fake.onmessage) fake.onmessage(ev); fake._listeners.forEach(f => f(ev)); }, 0); },
      onmessage: null,
    };
    AMTrainerMain(inner);
    fake.postMessage = (msg) => setTimeout(() => { if (!fake.terminated && inner.onmessage) inner.onmessage({ data: msg }); }, 0);
    fake.addEventListener = (type, f) => { if (type === 'message') fake._listeners.push(f); };
    fake.terminate = () => { fake.terminated = true; if (inner.onmessage) inner.onmessage({ data: { type: 'pause' } }); };
    return fake;
  }

  /* createTrainerWorker('sort', {seed, lr, batch, maxSteps, delayMs, reportMs, inline, onMessage})
   * → controller {start(), pause(), reset(opts?), step(n), set({lr, delayMs, maxSteps}), on(fn), terminate(), inline}
   * Messages are described above AMTrainerMain. A worker is used when possible;
   * otherwise (or with opts.inline) training runs in ~50 ms main-thread slices. */
  function createTrainerWorker(taskName, opts) {
    opts = opts || {};
    const initOpts = Object.assign({}, opts);
    delete initOpts.onMessage;
    const listeners = [];
    if (opts.onMessage) listeners.push(opts.onMessage);
    const sent = [];      // commands so far, replayed if we must fall back to the inline trainer
    let ready = false, w = null;
    const ctl = { inline: !!opts.inline, worker: null };

    function attach(worker, isInline) {
      w = worker;
      ctl.worker = worker;
      ctl.inline = isInline;
      w.onmessage = (e) => {
        if (e.data && e.data.type === 'ready') ready = true;
        for (const f of listeners) f(e.data);
      };
      if (!isInline) {
        // A worker that dies before it says hello (e.g. a browser blocking blob workers on
        // file://) is swapped for the main-thread trainer transparently.
        w.onerror = (e) => {
          if (!ready) { try { w.terminate(); } catch (_) { /* ignore */ } attach(inlineWorker(), true); sent.forEach(m => w.postMessage(m)); }
          else for (const f of listeners) f({ type: 'error', message: e.message });
        };
      }
    }
    if (!ctl.inline && typeof Worker !== 'undefined' && typeof Blob !== 'undefined' && typeof URL !== 'undefined') {
      try {
        const url = URL.createObjectURL(new Blob([workerSource()], { type: 'text/javascript' }));
        attach(new Worker(url), false);
        setTimeout(() => URL.revokeObjectURL(url), 10000);
      } catch (err) { w = null; }
    }
    if (!w) attach(inlineWorker(), true);
    const send = (m) => { sent.push(m); w.postMessage(m); };
    send({ type: 'init', task: taskName, opts: initOpts });

    ctl.on = (fn) => { listeners.push(fn); return () => { const i = listeners.indexOf(fn); if (i >= 0) listeners.splice(i, 1); }; };
    ctl.start = () => send({ type: 'start' });
    ctl.pause = () => send({ type: 'pause' });
    ctl.reset = (o) => send({ type: 'reset', task: taskName, opts: Object.assign({}, initOpts, o || {}) });
    ctl.step = (n) => send({ type: 'step', n: n || 1 });
    ctl.set = (o) => send(Object.assign({ type: 'set' }, o));
    ctl.terminate = () => w.terminate();
    return ctl;
  }

  const AM = root.AM = root.AM || {};
  AM.model = {
    get, list,
    ready: null,
    lib: { AMTensorLib: hasLibs ? AMTensorLib : null, AMTransformerLib: hasLibs ? AMTransformerLib : null, AMTaskLib, AMTrainerMain },
    tasks: AMTaskLib(),
    createTrainerWorker,
    workerSource,
  };
  // Decode every shipped model off the critical path; get() also decodes lazily on demand.
  AM.model.ready = new Promise((resolve, reject) => {
    setTimeout(() => {
      try { list().forEach(get); resolve(AM.model); } catch (err) { reject(err); }
    }, 0);
  });
})(typeof window !== 'undefined' ? window : globalThis);
