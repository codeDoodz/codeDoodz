/* Chapter 09 — Unembedding & sampling: "Rolling the Dice".
   The last few steps of the machine, all on the live tinyworld model:
   the final vector at the last position → final LayerNorm → unembedding
   (138 logits) → softmax → a decoding rule (greedy / sampling with
   temperature, top-k, top-p) → append → run again.

   A sticky stage walks through those steps for one sentence. Below it, the
   hero figure: a wheel of next-token odds inside a ring of 32 context slots.
   Spin it and the drawn token flies into the ring, the model re-reads the
   whole ring, and the wheel re-forms. Every distribution is computed by the
   real model (AM.model), and the sampler is the same rule as AM.model.sample
   (checked: identical draws for identical random numbers). */
(() => {
  const ID = 'predict';
  const M = AM.math;
  const D = AM.draw;
  const TAU = Math.PI * 2;
  const now = () => performance.now() / 1000;
  const clamp = M.clamp;
  const lerp = M.lerp;
  const easeOut = M.ease.out;
  const easeInOut = M.ease.inOut;
  const easeOutQuart = (t) => 1 - Math.pow(1 - clamp(t), 4);

  // The sentence the scrolling stage follows, and the prompts on the wheel.
  const STORY = 'the queen opened the door because she was cold . the king';
  const GREEDY_FROM = 'the king';
  const SURE = 'the queen opened the door because';
  const PRESETS = [
    { text: 'the princess walked to the', label: 'the princess walked to the' },
    { text: 'the queen opened the door because she was cold . the', label: '… she was cold . the' },
    { text: 'the king', label: 'the king' },
    { text: 'the queen opened the door because', label: 'the queen opened the door because' },
    { text: 'the capital of japan is', label: 'the capital of japan is' },
    { text: 'alice gave bob a cup . bob thanked', label: 'alice gave bob a cup . bob thanked' },
  ];

  // ====================================================================
  // 1. Small helpers
  // ====================================================================
  /** Number with a true minus sign. */
  const fmt = (x, d = 2) => {
    if (Math.abs(x) < 0.5 * Math.pow(10, -d)) x = 0;
    return (x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
  };
  /** Probability as a short percentage. */
  const pct = (p) => {
    if (p >= 0.9995) return '100%';
    if (p >= 0.995) return (p * 100).toFixed(1) + '%';
    if (p >= 0.0095) return Math.round(p * 100) + '%';
    if (p >= 0.00095) return (p * 100).toFixed(1) + '%';
    return p > 0 ? '<0.1%' : '0%';
  };
  const norm = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };

  /** Tracks whether an element is near the viewport. */
  function inView(node, margin = '160px') {
    const s = { on: true };
    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver((es) => { for (const e of es) s.on = e.isIntersecting; }, { rootMargin: `${margin} 0px ${margin} 0px` }).observe(node);
    }
    return s;
  }
  /** Text with an ink halo so it reads over threads. */
  function haloText(g, str, x, y, o = {}) {
    g.save();
    // legibility floor (contract rule 8); o.minSize: 0 only for decorative text
    g.font = AM.font(Math.max(o.size || 11, o.minSize ?? (AM.minText || 0)), o.role || 'mono', o.weight, o.italic);
    g.textAlign = o.align || 'center';
    g.textBaseline = o.baseline || 'middle';
    g.globalAlpha *= o.alpha ?? 1;
    g.lineJoin = 'round';
    if (o.halo !== false) {
      g.strokeStyle = o.halo || AM.rgba(AM.col.ink, 0.92);
      g.lineWidth = o.haloW || 4;
      g.strokeText(str, x, y);
    }
    g.fillStyle = o.color || AM.col.linen;
    g.fillText(str, x, y);
    g.restore();
  }
  function textW(g, str, size, role = 'body', weight) {
    g.save(); g.font = AM.font(size, role, weight); const w = g.measureText(str).width; g.restore(); return w;
  }
  /** Hex mix of two hex colours (AM.rgba needs hex). */
  function mixHex(a, b, t) {
    const A = AM.hexToRgb(a), B = AM.hexToRgb(b);
    return '#' + [0, 1, 2].map((i) => Math.round(A[i] + (B[i] - A[i]) * t).toString(16).padStart(2, '0')).join('');
  }
  /** Rank → dye. The first seven ranks get the seven dyes; later ranks repeat them, dyed paler into the ground. */
  const RANK_COLORS = [];
  function rankColor(r) {
    if (!RANK_COLORS.length) {
      for (let k = 0; k < 7; k++) RANK_COLORS.push(AM.dyeList[k]);
      for (let k = 0; k < 7; k++) RANK_COLORS.push(mixHex(AM.dyeList[k], AM.col.ink3, 0.42));
      for (let k = 0; k < 7; k++) RANK_COLORS.push(mixHex(AM.dyeList[k], AM.col.ink3, 0.62));
    }
    if (r < 21) return RANK_COLORS[r];
    return r % 2 ? '#4f5878' : '#646d8d';
  }
  function cubic(P, t) {
    const u = 1 - t;
    return {
      x: u * u * u * P[0] + 3 * u * u * t * P[2] + 3 * u * t * t * P[4] + t * t * t * P[6],
      y: u * u * u * P[1] + 3 * u * u * t * P[3] + 3 * u * t * t * P[5] + t * t * t * P[7],
    };
  }
  /** Silk thread along a cubic: soft glow, dyed core, thin sheen. */
  function silk(g, P, { color = AM.dye.weld, width = 1.6, alpha = 1, sheen = true } = {}) {
    if (alpha <= 0.004) return;
    g.save();
    g.lineCap = 'round';
    g.beginPath(); g.moveTo(P[0], P[1]); g.bezierCurveTo(P[2], P[3], P[4], P[5], P[6], P[7]);
    g.strokeStyle = AM.rgba(color, 0.14 * alpha);
    g.lineWidth = width * 4 + 2;
    g.stroke();
    g.strokeStyle = AM.rgba(color, alpha);
    g.lineWidth = width;
    g.stroke();
    if (sheen && width > 1.3) {
      g.strokeStyle = AM.rgba('#fff4d6', 0.3 * alpha);
      g.lineWidth = Math.max(0.5, width * 0.25);
      g.stroke();
    }
    g.restore();
  }

  // ====================================================================
  // 2. The model side: final LayerNorm, unembedding, and the sampler
  // ====================================================================
  function getModel() {
    try { return (AM.model && AM.model.get) ? AM.model.get('tinyworld') : null; } catch (e) { return null; }
  }
  /** The final LayerNorm and unembedding weights (read-only views of the live model). */
  function readUnembed(m) {
    const P = m && m._net && m._net.params;
    if (!P || !P.wout || !P.bout || !P['lnf.g'] || !P['lnf.b']) return null;
    return { W: P.wout.data, b: P.bout.data, g: P['lnf.g'].data, beta: P['lnf.b'].data, d: m.config.d_model, V: m.config.vocab_size };
  }
  /** h = γ ⊙ (x − μ)/√(σ² + ε) + β, exactly as the model computes it. */
  function finalNorm(U, x) {
    const d = x.length;
    let mu = 0; for (let i = 0; i < d; i++) mu += x[i]; mu /= d;
    let v = 0; for (let i = 0; i < d; i++) v += (x[i] - mu) * (x[i] - mu); v /= d;
    const s = 1 / Math.sqrt(v + 1e-5);
    const h = new Float64Array(d);
    for (let i = 0; i < d; i++) h[i] = (x[i] - mu) * s * U.g[i] + U.beta[i];
    let mh = 0; for (let i = 0; i < d; i++) mh += h[i]; mh /= d;
    let vh = 0; for (let i = 0; i < d; i++) vh += (h[i] - mh) * (h[i] - mh); vh /= d;
    return { h, mu, sd: Math.sqrt(v), muH: mh, sdH: Math.sqrt(vh) };
  }
  /** Full forward pass on a prompt, plus our own unembedding of the last position. */
  function analyse(m, U, text) {
    const { ids, tokens } = m.encode(text);
    const r = m.run(ids, { capture: true });
    const t = ids.length - 1;
    const x = r.resid[r.resid.length - 1][t];
    const logits = r.logits[t], probs = r.probs[t];
    const ln = U ? finalNorm(U, x) : null;
    let maxErr = 0;
    if (ln) {
      for (let j = 0; j < U.V; j++) {
        let z = U.b[j];
        for (let i = 0; i < U.d; i++) z += ln.h[i] * U.W[i * U.V + j];
        maxErr = Math.max(maxErr, Math.abs(z - logits[j]));
      }
    }
    return { ids, tokens, x, ln, logits, probs, maxErr };
  }

  /**
   * The distribution a sampler actually draws from. Same rule as AM.model.sample:
   * drop <pad>/<unk>, sort by logit, apply temperature (T = 0 is greedy), keep the
   * top k, then the smallest prefix whose mass reaches p, then renormalise.
   * Returns ranks: order[r] is the r-th most likely id, q[r] its final probability,
   * full[r] its probability after temperature only, cum[] the running sum of q.
   */
  function shapeDist(logits, banned, { T = 1, k = 0, p = 1 } = {}) {
    const V = logits.length, order = [];
    for (let i = 0; i < V; i++) if (!banned.has(i)) order.push(i);
    order.sort((a, b) => logits[b] - logits[a]);
    const n = order.length;
    const full = new Float64Array(n);
    if (T <= 1e-6) full[0] = 1;
    else {
      const mx = logits[order[0]];
      let z = 0;
      for (let r = 0; r < n; r++) { full[r] = Math.exp((logits[order[r]] - mx) / T); z += full[r]; }
      for (let r = 0; r < n; r++) full[r] /= z;
    }
    let keep = T <= 1e-6 ? 1 : n; // greedy: only the argmax survives
    if (k >= 1 && k < keep) keep = Math.round(k);
    if (p < 1) {
      let zz = 0; for (let r = 0; r < keep; r++) zz += full[r];
      let acc = 0;
      for (let r = 0; r < keep; r++) { acc += full[r] / zz; if (acc >= p) { keep = r + 1; break; } }
    }
    let zk = 0; for (let r = 0; r < keep; r++) zk += full[r];
    const q = new Float64Array(n);
    for (let r = 0; r < keep; r++) q[r] = full[r] / zk;
    let H = 0; for (let r = 0; r < keep; r++) if (q[r] > 0) H -= q[r] * Math.log(q[r]);
    const cum = new Float64Array(n + 1);
    for (let r = 0; r < n; r++) cum[r + 1] = cum[r] + q[r];
    const rank = new Map(); order.forEach((id, r) => rank.set(id, r));
    return { order, n, full, q, keep, H, eff: Math.exp(H), cum, rank };
  }
  /** Inverse-CDF draw: the rank whose segment of [0, 1) holds u. */
  function pickRank(cum, keep, u) {
    for (let r = 0; r < keep; r++) if (u < cum[r + 1]) return r;
    return keep - 1;
  }

  // ====================================================================
  // 3. Chapter-scoped CSS
  // ====================================================================
  const CSS = `
    #ch-predict .pr-intro { margin-bottom: calc(-1 * var(--space-5)); }
    #ch-predict .pr-stagefig .stage-canvas canvas,
    #ch-predict .pr-hero .stage-canvas canvas {
      border-radius: var(--radius);
      background: radial-gradient(120% 90% at 50% 38%, color-mix(in srgb, var(--ink-2) 90%, var(--weld)) 0%, var(--ink-2) 55%, var(--ink) 100%);
      border: 1px solid var(--rule);
    }
    #ch-predict .pr-stage-cap { max-width: 62ch; }
    @media (max-width: 900px) {
      #ch-predict .pr-stage-cap { display: none; }
      #ch-predict .pr-stagefig { gap: 6px; }
    }
    #ch-predict .step .math.block { font-size: 0.9em; padding: 8px 10px; }
    #ch-predict .step em { color: var(--linen); }
    #ch-predict .pr-greedy {
      display: block;
      padding: 10px 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--rule);
      background: color-mix(in srgb, var(--ink-3) 70%, transparent);
      font-family: var(--font-mono);
      font-size: 0.78rem;
      line-height: 1.7;
      color: var(--linen-dim);
    }
    #ch-predict .pr-greedy b { color: var(--linen); font-weight: 500; }
    #ch-predict .pr-greedy .pr-loop { color: var(--madder); text-decoration: underline wavy color-mix(in srgb, var(--madder) 60%, transparent); text-underline-offset: 4px; }

    #ch-predict .pr-hero { gap: var(--space-4); }
    #ch-predict .pr-hero-intro { margin-bottom: calc(var(--space-6) - var(--space-8)); }
    #ch-predict .pr-grid { display: grid; grid-template-columns: minmax(0, 1.12fr) minmax(0, 1fr); gap: var(--space-5) var(--space-6); align-items: start; }
    @media (max-width: 900px) { #ch-predict .pr-grid { grid-template-columns: minmax(0, 1fr); } }
    #ch-predict .pr-wheelbox { min-width: 0; display: grid; gap: var(--space-3); }
    #ch-predict .pr-panel { display: grid; gap: var(--space-5); min-width: 0; align-content: start; }
    #ch-predict .pr-group { display: grid; gap: var(--space-2); min-width: 0; }
    #ch-predict .pr-group-label { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.12em; text-transform: uppercase; color: var(--mist); }
    #ch-predict .pr-chips { display: flex; flex-wrap: wrap; gap: 6px; }
    #ch-predict .pr-chip {
      border: 1px solid var(--rule-strong);
      background: var(--ink-2);
      color: var(--linen-dim);
      border-radius: 999px;
      padding: 5px 12px;
      font-family: var(--font-body);
      font-size: 0.86rem;
      line-height: 1.3;
      cursor: pointer;
      text-align: left;
      transition: border-color 0.15s, color 0.15s, background 0.15s;
    }
    #ch-predict .pr-chip:hover { border-color: var(--linen-dim); color: var(--linen); }
    #ch-predict .pr-chip[aria-pressed='true'] { border-color: var(--weld); color: var(--linen); background: color-mix(in srgb, var(--weld) 13%, var(--ink-2)); }
    #ch-predict .pr-sliders { display: grid; gap: var(--space-3); }
    #ch-predict .pr-sliders .ctl-range { width: 100%; }
    #ch-predict .pr-buttons { display: flex; flex-wrap: wrap; gap: var(--space-2); align-items: center; }
    #ch-predict .pr-seed { display: inline-flex; align-items: center; gap: 4px; margin-left: auto; font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.1em; text-transform: uppercase; color: var(--mist); }
    #ch-predict .pr-seed button { width: 30px; height: 30px; border-radius: 50%; border: 1px solid var(--rule-strong); background: var(--ink-2); color: var(--linen); cursor: pointer; font-family: var(--font-mono); font-size: 13px; line-height: 1; }
    #ch-predict .pr-seed button:hover { border-color: var(--weld); color: var(--weld); }
    #ch-predict .pr-seed output { min-width: 2.2em; text-align: center; color: var(--weld); font-variant-numeric: tabular-nums; letter-spacing: 0; font-size: 12px; }
    #ch-predict .pr-stats { margin: 0; display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 7px 16px; padding: var(--space-4); border-radius: var(--radius); border: 1px solid var(--rule); background: color-mix(in srgb, var(--ink-2) 70%, transparent); }
    #ch-predict .pr-stats dt { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--mist); padding-top: 3px; }
    #ch-predict .pr-stats dd { margin: 0; color: var(--linen-dim); font-size: 0.9rem; font-variant-numeric: tabular-nums; min-width: 0; }
    #ch-predict .pr-stats dd b { color: var(--weld); font-weight: 600; }

    #ch-predict .pr-trans-wrap { display: grid; gap: 6px; }
    #ch-predict .pr-trans { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 4px 7px; min-height: 64px; padding: var(--space-4); border-radius: var(--radius); border: 1px solid var(--rule); background: var(--ink-2); }
    #ch-predict .pr-tw { display: inline-grid; gap: 1px; justify-items: start; font-size: 1rem; line-height: 1.35; }
    #ch-predict .pr-tw.is-prompt { color: var(--mist); }
    #ch-predict .pr-tw.is-gen .pr-tw-word { color: var(--linen); font-weight: 600; border-bottom: 2px solid color-mix(in srgb, var(--weld) 70%, transparent); }
    #ch-predict .pr-tw.is-new .pr-tw-word { color: var(--weld); }
    #ch-predict .pr-tw-alts { display: grid; gap: 0; font-family: var(--font-mono); font-size: 9.5px; line-height: 1.45; color: var(--mist); white-space: nowrap; }
    #ch-predict .pr-tw-alts .is-pick { color: var(--weld); }
    #ch-predict .pr-tw-alts .is-other { color: var(--madder); }
    #ch-predict .pr-full { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.08em; text-transform: uppercase; color: var(--weld); align-self: center; }
    #ch-predict .pr-key p + p { margin-top: var(--space-2); }
  `;

  // ====================================================================
  // 4. Chapter
  // ====================================================================
  AM.chapter({
    id: ID,
    num: 9,
    kicker: 'Unembedding & sampling',
    title: 'Rolling the <em>Dice</em>',
    lede: 'The last vector becomes a score for every word in the vocabulary. Softmax turns the scores into odds, a decoding rule rolls the dice, and the chosen word is fed back in to pick the next one.',
    where: 'unembed',

    mount(root, ctx) {
      ctx.header();
      AM.css(CSS);
      const el = ctx.el;

      const body = el('div', { class: 'ch-body' });
      root.appendChild(body);

      const m = getModel();
      if (!m) {
        body.appendChild(el('p', { class: 'caption' }, 'The live model could not be loaded in this browser, so this chapter has nothing to show.'));
        return;
      }
      const U = readUnembed(m);
      const banned = new Set(['<pad>', '<unk>'].map((t) => m.tokenId(t)).filter((i) => i >= 0));

      body.appendChild(el('div', { class: 'prose pr-intro' },
        el('p', { html: 'The stack of blocks from the last chapter leaves one vector per position in the residual stream, full of everything the attention heads and MLPs wrote into it. None of it is a word yet.' }),
        el('p', { html: 'This chapter covers the last few steps: score every word in the vocabulary, turn the scores into probabilities, pick one, and go round again. Every distribution here comes from the live tiny model.' }),
      ));

      buildStory(body, ctx, m, U, banned);
      buildWheel(body, ctx, m, banned);

      body.appendChild(el('div', { class: 'callout pr-key' },
        el('div', { class: 'callout-label' }, 'Key idea'),
        el('p', { html: 'The vector at the last position goes through a final LayerNorm and the unembedding matrix, giving one <span class="term">logit</span> per vocabulary entry. Softmax turns the logits into probabilities. A decoding rule picks one token: greedy takes the argmax, sampling rolls the dice, and temperature, top-k and top-p decide how loaded the dice are.' }),
        el('p', { html: 'The chosen token is appended and the whole model runs again. A chatbot’s reply is this loop, once per token, until it produces a token that means “stop”.' }),
      ));
    },
  });

  // ====================================================================
  // 5. The scrolling stage: from the last vector to a drawn word
  // ====================================================================
  function buildStory(body, ctx, m, U, banned) {
    const el = ctx.el, ui = AM.ui;
    const A = analyse(m, U, STORY);
    const toks = A.tokens;
    const lastTok = toks[toks.length - 1];
    const z = A.logits, probs = A.probs;
    const V = z.length, d = A.x.length;
    const nWords = m.vocab.filter((w, i) => !banned.has(i) && !/^[.,]$/.test(w)).length;
    let zmin = Infinity, zmax = -Infinity;
    for (let j = 0; j < V; j++) { zmin = Math.min(zmin, z[j]); zmax = Math.max(zmax, z[j]); }
    let psum = 0; for (let j = 0; j < V; j++) psum += probs[j];
    let sumE = 0; for (let j = 0; j < V; j++) sumE += Math.exp(z[j]);

    const base = shapeDist(z, banned, { T: 1 });
    const word = (r) => m.vocab[base.order[r]];
    const top8 = base.order.slice(0, 8);
    const top8mass = top8.reduce((s, id) => s + probs[id], 0);
    const VERBS = new Set(['walked', 'loved', 'gave', 'thanked', 'met', 'waved', 'helped', 'said', 'says', 'opened', 'closed', 'found', 'lost', 'dropped']);
    const top8Name = top8.every((id) => VERBS.has(m.vocab[id])) ? 'Eight verbs' : 'The top eight words';
    const gap01 = z[base.order[0]] - z[base.order[1]];
    const hot = shapeDist(z, banned, { T: 2 });
    let hotTail = 0; for (let r = 8; r < hot.n; r++) hotTail += hot.full[r];
    const k3 = shapeDist(z, banned, { k: 3 });
    const p9 = shapeDist(z, banned, { p: 0.9 });
    const sure = (() => { const e = m.encode(SURE); const r = m.run(e.ids); return shapeDist(r.logits[e.ids.length - 1], banned, { p: 0.9 }); })();
    const xN = norm(A.x), hN = A.ln ? norm(A.ln.h) : 0;
    let xMax = 0; for (let i = 0; i < d; i++) xMax = Math.max(xMax, Math.abs(A.x[i]));
    let hMax = 0; if (A.ln) for (let i = 0; i < d; i++) hMax = Math.max(hMax, Math.abs(A.ln.h[i]));

    // ---- DOM
    const split = el('div', { class: 'ch-split' });
    body.appendChild(split);
    const stage = el('div', { class: 'ch-stage' });
    const prose = el('div', { class: 'ch-prose' });
    split.append(stage, prose);

    const fig = ui.figure({ title: 'The last step · one sentence', badge: 'live', cls: 'pr-stagefig' });
    stage.appendChild(fig);
    const cv = ctx.canvas(fig, {
      label: `The live model reads “${STORY}”. Its final vector at “${lastTok}” is normalised and multiplied by the unembedding matrix, giving ${V} logits; softmax turns them into probabilities, led by ${word(0)} ${pct(base.q[0])}, ${word(1)} ${pct(base.q[1])} and ${word(2)} ${pct(base.q[2])}.`,
      height: (w) => {
        const vh = window.innerHeight;
        if (w < 520) return Math.round(Math.min(Math.max(330, w * 0.98), 400, Math.max(320, vh * 0.47)));
        // stacked layout (tablets): the stage is sticky above the text, so keep it to about half the screen
        if (window.innerWidth <= 900) return Math.round(clamp(Math.min(w * 0.75, vh * 0.5), 340, 520));
        return Math.round(Math.min(Math.max(480, w * 0.9), 590, Math.max(460, vh * 0.7)));
      },
    });
    fig.appendChild(el('figcaption', { class: 'pr-stage-cap', html: `Live model: the residual vector, the real ${d}×${V} unembedding matrix W<sub>U</sub> (blue positive, red negative), and the ${V} logits and probabilities it produces for this sentence. The draws use the same sampling rule as the model’s own sampler.` }));

    const s = (k) => `<span class="math">${k}</span>`;
    const b = (k) => `<strong>${k}</strong>`;
    const greedySpan = el('span', { class: 'pr-greedy' }, '…');
    const STEPS = [
      {
        label: '1 · The last vector',
        html: [
          `After the last block, every position holds a vector of d<sub>model</sub> = ${d} numbers. To guess the word after “${lastTok}”, only the vector at the last position is needed.`,
          `It goes through one final LayerNorm, LN<sub>f</sub> (<a href="#ch-residual">chapter 7</a>). The live vector has length ${b(fmt(xN, 1))}; after LN<sub>f</sub> it has length ${b(fmt(hN, 1))}.`,
          `In training, every position predicts its own next word at the same time. When generating, only the last one matters.`,
        ],
      },
      {
        label: '2 · A score for every word',
        html: [
          `The <span class="term">unembedding</span> matrix W<sub>U</sub> has one column per vocabulary entry: ${d} rows by ${V} columns in our model. The ${V} entries are ${nWords} words, the full stop and comma, and two special tokens, &lt;pad&gt; and &lt;unk&gt;, that the sampler never draws. The dot product of h with column j, plus a bias, is the <span class="term">logit</span> for token j:`,
          `<span class="math block">z<sub>j</sub> = h · W<sub>U</sub>[:, j] + b<sub>j</sub></span>`,
          `That is ${V} dot products, done as one vector–matrix multiply. A logit is an unnormalised score, any real number. Here they run from ${b(fmt(zmin, 1))} to ${b(fmt(zmax, 2))}, for <em>${word(0)}</em>.`,
          `Large models do the same with vocabularies of about 100,000 to a few hundred thousand tokens, which makes W<sub>U</sub> one of their biggest matrices. Some reuse the embedding matrix here (tied weights); ours learned its own.`,
        ],
      },
      {
        label: '3 · Softmax makes odds',
        html: [
          `<span class="math block">p<sub>j</sub> = e<sup>z<sub>j</sub></sup> / Σ<sub>k</sub> e<sup>z<sub>k</sub></sup></span>`,
          `Exponentiating makes every score positive; dividing by the total makes them sum to 1. Only differences between logits matter: <em>${word(0)}</em> beats <em>${word(1)}</em> by ${fmt(gap01, 2)}, so it is e<sup>${fmt(gap01, 2)}</sup> = ${fmt(Math.exp(gap01), 2)} times as likely (${pct(base.q[0])} against ${pct(base.q[1])}).`,
          `${top8Name} share ${b((top8mass * 100).toFixed(1) + '%')} of the probability. The other ${V - 8} entries get almost nothing, but never exactly zero.`,
        ],
      },
      {
        label: '4 · Greedy, or roll the dice',
        html: [
          `Now one word must be chosen. <span class="term">Greedy</span> decoding always takes the most likely one, the argmax. It is deterministic, so the same prompt always gives the same text, and long greedy runs tend to get stuck. Greedy from “${GREEDY_FROM}”, our model writes:`,
          greedySpan,
          `<span class="term">Sampling</span> draws at random instead. Lay the probabilities end to end along the line from 0 to 1, draw a uniform random number u, and take the word whose segment u lands in. <em>${word(0)}</em> wins ${pct(base.q[0])} of draws, <em>${word(1)}</em> ${pct(base.q[1])}, and so on.`,
        ],
      },
      {
        label: '5 · Temperature',
        html: [
          `Before softmax, divide every logit by a <span class="term">temperature</span> T:`,
          `<span class="math block">p<sub>j</sub> ∝ e<sup>z<sub>j</sub> / T</sup></span>`,
          `T = 1 is the model’s own distribution. Below 1 the gaps grow and the favourite takes over; as T → 0 this becomes greedy. Above 1 the distribution flattens: rare words get more chances, so the text gets more surprising and more often wrong. At T = 2 the words outside the top eight get ${b(pct(hotTail))} of the draws.`,
          `The stage shows the <em>effective number of choices</em>, e<sup>H</sup>, where H is the entropy. A distribution spread evenly over N words scores exactly N. At T = 1 it is ${b(fmt(base.eff, 1))}.`,
        ],
      },
      {
        label: '6 · Top-k and top-p',
        html: [
          `Two ways to cut off the long tail before drawing. <span class="term">Top-k</span> keeps the k most likely words. <span class="term">Top-p</span>, or nucleus sampling, keeps the smallest set of words whose probabilities add up to at least p.`,
          `Either way the survivors are renormalised to sum to 1. With k = 3, <em>${word(0)}</em> rises from ${pct(base.q[0])} to ${b(pct(k3.q[0]))}. Top-p adapts to how sure the model is: p = 0.9 keeps ${b(String(p9.keep))} words here, but after “${SURE}” it keeps ${sure.keep === 1 ? 'just one' : sure.keep}: <em>${m.vocab[sure.order[0]]}</em>.`,
        ],
      },
    ];
    const stepEls = STEPS.map((stp) => el('div', { class: 'step' },
      el('div', { class: 'step-label' }, stp.label),
      stp.html.map((h) => (typeof h === 'string' ? el('p', { html: h }) : h))));
    prose.append(...stepEls);

    // The greedy run costs ~30 forward passes, so it is computed once the chapter is near.
    let greedyDone = false;
    ctx.onVisible(() => {
      if (greedyDone) return;
      greedyDone = true;
      setTimeout(() => {
        try {
          const start = m.encode(GREEDY_FROM).ids;
          const out = m.decode(m.generate(start, { temperature: 0, maxNew: m.config.n_ctx - start.length, stopAt: null }));
          const cont = out.slice(start.length);
          // find a repeating cycle at the end (period 2–14, at least two copies)
          let period = 0;
          for (let P = 2; P <= 14 && !period; P++) {
            if (cont.length < 2 * P) break;
            let ok = true;
            for (let i = 0; i < P; i++) if (cont[cont.length - 1 - i] !== cont[cont.length - 1 - i - P]) { ok = false; break; }
            if (ok) period = P;
          }
          let loopStart = cont.length;
          if (period) {
            // earliest index from which the text just repeats itself with this period
            loopStart = cont.length - period;
            while (loopStart > 0 && cont[loopStart - 1] === cont[loopStart - 1 + period]) loopStart--;
            // start the highlight at a sentence boundary if one is close
            for (let i = loopStart; i < loopStart + period && i < cont.length; i++) if (i > 0 && cont[i - 1] === '.') { loopStart = i; break; }
          }
          greedySpan.replaceChildren(
            el('b', {}, GREEDY_FROM + ' '),
            cont.slice(0, loopStart).join(' ') + ' ',
            period ? el('span', { class: 'pr-loop', title: 'the same phrase, over and over' }, cont.slice(loopStart).join(' ')) : null,
            period ? ' …' : '',
          );
        } catch (e) { greedySpan.textContent = '(the live model could not run here)'; }
      }, 30);
    });

    // ---- stage state
    const S = {
      step: 0, tStep: now(), last: now(),
      tw: { vec: 1, cloth: 0, field: 0, prob: 0, sort: 0, temp: 0, cut: 0, slot: 0 },
      qDisp: Float64Array.from(base.q),
      cur: base, // the distribution shown in sorted mode
      Tnow: 1, cutMode: 0,
      drop: null, nextDrop: 0, slotWord: null, slotPop: -9,
      rng: M.rng(20261009),
      clothImg: null, clothKey: '',
    };
    ctx.steps(stepEls, (i) => {
      const t = now();
      if (i !== S.step) S.tStep = t;
      S.step = i;
      if (i >= 3 && t > S.nextDrop - 0.4) S.nextDrop = t + 0.9;
    });

    // sorted rank of every vocabulary id (banned ids last)
    const rankOf = new Int32Array(V);
    base.order.forEach((id, r) => { rankOf[id] = r; });
    { let r = base.n; for (const id of banned) rankOf[id] = r++; }

    // ---- layout
    function layout(w, h) {
      const phone = w < 520;
      const pad = phone ? 10 : 18;
      const left = pad, right = w - pad, W = right - left;
      const sentY = phone ? 19 : 27;
      const top = sentY + (phone ? 26 : 34);
      const bottom = h - (phone ? 28 : 36);
      const pillZone = phone ? 64 : 84;
      const vecH = Math.round(clamp((bottom - top - (phone ? 22 : 30) - 2 * 18 - pillZone) / 2, 40, phone ? 96 : 120));
      const xBar = { x: left, y: top + (phone ? 22 : 30), w: W, h: vecH };
      const hBar = { x: left, y: xBar.y + vecH + pillZone + 18, w: W, h: vecH };
      const cloth = { x: left, y: top + (phone ? 14 : 20), w: W, h: 0 };
      cloth.h = Math.round((bottom - cloth.y) * (phone ? 0.38 : 0.4));
      const field = { x: left, y: cloth.y + cloth.h + (phone ? 10 : 14), w: W, h: 0 };
      field.h = bottom - field.y;
      const labelRoom = phone ? 46 : 62;
      const chart = { x: left, y: top + (phone ? 26 : 36), w: W, h: 0 };
      chart.h = bottom - chart.y;
      const ribH = phone ? 24 : 32;
      const rib = { x: left, y: bottom - ribH - (phone ? 14 : 18), w: W, h: ribH };
      const bars = { x: left, y: top + (phone ? 34 : 44), w: W, h: 0 };
      bars.h = rib.y - (phone ? 48 : 64) - bars.y;
      const ntop = phone ? 6 : 8;
      return { phone, pad, left, right, W, sentY, top, bottom, xBar, hBar, cloth, field, labelRoom, chart, rib, bars, ntop, w, h };
    }

    // ---- cached cloth (the real W_U, 64 rows × 138 columns, woven)
    function clothImage(L) {
      const key = `${L.cloth.w}x${L.cloth.h}@${cv.dpr}`;
      if (S.clothKey === key && S.clothImg) return S.clothImg;
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(L.cloth.w * cv.dpr));
      c.height = Math.max(1, Math.round(L.cloth.h * cv.dpr));
      const g = c.getContext('2d');
      g.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      const cw = L.cloth.w / V, ch = L.cloth.h / d;
      if (U) {
        for (let i = 0; i < d; i++) {
          for (let j = 0; j < V; j++) {
            const v = U.W[i * V + j];
            g.fillStyle = AM.color.div(v, 0.75);
            const x = j * cw, y = i * ch;
            g.fillRect(x + 0.15, y + 0.15, Math.max(0.6, cw - 0.3), Math.max(0.6, ch - 0.3));
            // twill: alternate stitches catch a little light
            if (((i + j) & 3) < 2) { g.fillStyle = 'rgba(237,229,211,0.07)'; g.fillRect(x, y, cw, ch * 0.5); }
          }
        }
      }
      // fine warp lines (one per word)
      g.strokeStyle = 'rgba(13,17,33,0.35)'; g.lineWidth = 0.5;
      g.beginPath();
      for (let j = 1; j < V; j++) { const x = Math.round(j * cw) + 0.25; g.moveTo(x, 0); g.lineTo(x, L.cloth.h); }
      g.stroke();
      S.clothImg = c; S.clothKey = key;
      return c;
    }

    // ---- label placement: one row, labels pushed apart sideways, leader lines to their beads
    function spreadLabels(g, items, L, size, rowY) {
      items.sort((a, b) => a.x - b.x);
      const gap = 6;
      items.forEach((it) => { it.w = textW(g, it.text, size, 'mono') + 4; it.lx = it.x; it.ly = rowY; });
      for (let i = 0; i < items.length; i++) {
        const lo = i ? items[i - 1].lx + (items[i - 1].w + items[i].w) / 2 + gap : L.left + items[i].w / 2;
        items[i].lx = Math.max(items[i].lx, lo);
      }
      for (let i = items.length - 1; i >= 0; i--) {
        const hi = i < items.length - 1 ? items[i + 1].lx - (items[i + 1].w + items[i].w) / 2 - gap : L.right - items[i].w / 2;
        items[i].lx = Math.min(items[i].lx, hi);
      }
      return items;
    }
    function drawLeaders(g, items, color, alpha, size, textColor) {
      for (const it of items) {
        g.strokeStyle = AM.rgba(color, 0.4 * alpha); g.lineWidth = 0.8;
        g.beginPath(); g.moveTo(it.x, it.ay - 4); g.bezierCurveTo(it.x, it.ly + 14, it.lx, it.ay - 10, it.lx, it.ly + 6); g.stroke();
        haloText(g, it.text, it.lx, it.ly, { size, color: textColor, alpha });
      }
    }

    // ---- sentence row (tokens as tiles; the last one lit; a slot for the drawn word)
    function sentence(g, L, slotA) {
      const phone = L.phone;
      const size = phone ? 11.5 : 13.5;
      const gap = phone ? 4 : 6;
      const padX = Math.round(size * 0.6) * 2;
      const items = toks.map((t) => ({ t, w: textW(g, t, size, 'body', 600) + padX }));
      const slotTxt = S.slotWord || '?';
      const slotW = Math.max(textW(g, slotTxt, size, 'body', 600) + padX, phone ? 34 : 44);
      let total = items.reduce((a, it) => a + it.w, 0) + gap * (items.length - 1) + (gap + slotW) * slotA;
      // drop words from the left until the row fits
      let first = 0;
      const ell = phone ? 14 : 18;
      while (total > L.W && first < items.length - 2) { total -= items[first].w + gap; first++; }
      const lead = first > 0 ? ell + gap : 0;
      let x = first > 0 ? L.left : L.left + (L.W - total) / 2;
      if (first > 0) { haloText(g, '…', x + ell / 2, L.sentY, { size: 13, color: AM.col.mist, halo: false }); x += lead; }
      const y = L.sentY;
      let lastBox = null;
      for (let i = first; i < items.length; i++) {
        const isLast = i === items.length - 1;
        const box = D.token(g, items[i].t, x, y, { size, align: 'left', w: items[i].w, h: size + (phone ? 11 : 13), selected: isLast, color: isLast ? AM.col.linen : AM.col.linenDim, alpha: isLast ? 1 : 0.88 });
        if (isLast) lastBox = box;
        x += items[i].w + gap;
      }
      let slotBox = null;
      if (slotA > 0.01) {
        const pop = AM.reducedMotion ? 1 : easeOut(clamp((now() - S.slotPop) / 0.35));
        g.save();
        g.globalAlpha *= slotA;
        const filled = !!S.slotWord;
        slotBox = D.token(g, slotTxt, x, y, {
          size, align: 'left', w: slotW, h: size + (phone ? 11 : 13),
          fill: filled ? AM.mix(AM.col.ink2, AM.dye.weld, 0.22 + 0.2 * (1 - pop)) : AM.col.ink,
          stroke: filled ? AM.dye.weld : AM.rgba(AM.col.linen, 0.3),
          color: filled ? AM.dye.weld : AM.col.mist,
        });
        if (!filled) { D.roundRect(g, slotBox.x, slotBox.y, slotBox.w, slotBox.h, 7); g.setLineDash([3, 3]); g.strokeStyle = AM.rgba(AM.col.linen, 0.35); g.stroke(); g.setLineDash([]); }
        g.restore();
      }
      return { lastBox, slotBox };
    }

    // ---- the h barcode / shuttle: one rectangle per dimension, morphing
    function hRects(L, e, shuttleX) {
      const out = [];
      const hb = L.hBar, cl = L.cloth;
      const n = d;
      const bw = hb.w / n;
      const half = hb.h / 2;
      const rowH = cl.h / n;
      const halfW = L.phone ? 9 : 13;
      for (let i = 0; i < n; i++) {
        const v = A.ln ? A.ln.h[i] / (hMax || 1) : 0;
        // horizontal barcode (bars stand on a midline)
        const hx0 = hb.x + i * bw + bw * 0.14, hx1 = hb.x + (i + 1) * bw - bw * 0.14;
        const hy0 = hb.y + half - Math.max(0, v) * half, hy1 = hb.y + half + Math.max(0, -v) * half;
        // vertical shuttle (bars lie on a vertical spine, one per W_U row)
        const vy0 = cl.y + i * rowH + rowH * 0.12, vy1 = cl.y + (i + 1) * rowH - rowH * 0.12;
        const vx0 = shuttleX - Math.max(0, -v) * halfW, vx1 = shuttleX + Math.max(0, v) * halfW;
        const ei = easeInOut(clamp(e * 1.35 - (i / n) * 0.35));
        out.push({ x0: lerp(hx0, vx0, ei), y0: lerp(hy0, vy0, ei), x1: lerp(hx1, vx1, ei), y1: lerp(hy1, vy1, ei), v });
      }
      return out;
    }

    // ---- main draw
    function draw() {
      const { g, w, h } = cv;
      if (!w) return;
      const t = now();
      const dt = Math.min(0.1, t - S.last);
      S.last = t;
      const kk = 1 - Math.exp(-dt * (AM.reducedMotion ? 60 : 4.2));
      const step = S.step;
      const TG = {
        vec: step === 0 ? 1 : 0,
        cloth: step === 1 ? 1 : 0,
        field: step >= 1 ? 1 : 0,
        prob: step >= 2 ? 1 : 0,
        sort: step >= 3 ? 1 : 0,
        temp: step === 4 ? 1 : 0,
        cut: step === 5 ? 1 : 0,
        slot: step >= 3 ? 1 : 0,
      };
      const tw = S.tw;
      for (const key in TG) tw[key] += (TG[key] - tw[key]) * kk;
      const age = AM.reducedMotion ? 99 : t - S.tStep;

      // which distribution the sorted view shows
      let target = base;
      if (step === 4) {
        let T;
        if (AM.reducedMotion) T = [1, 2, 0.5][Math.floor(age / 3) % 3] || 1;
        else T = Math.exp(1.1 * Math.sin((TAU * age) / 11));
        S.Tnow = T;
        target = shapeDist(z, banned, { T });
      } else if (step === 5) {
        S.cutMode = Math.floor((AM.reducedMotion ? (t - S.tStep) : age) / 4.2) % 2;
        target = S.cutMode === 0 ? k3 : p9;
      }
      S.cur = target;
      const qk = step === 4 ? 1 : 1 - Math.exp(-dt * (AM.reducedMotion ? 60 : 6));
      let lag = 0;
      for (let r = 0; r < S.qDisp.length; r++) { S.qDisp[r] += (target.q[r] - S.qDisp[r]) * qk; lag = Math.max(lag, Math.abs(target.q[r] - S.qDisp[r])); }

      cv.clear();
      const L = layout(w, h);
      D.weave(g, 0, 0, w, h, { step: 7, alpha: 0.025 });

      // ---- shuttle sweep (step 1)
      const SWEEP = 2.3;
      let sweep = 0;
      if (step === 1) sweep = AM.reducedMotion ? 1 : clamp((age - 0.75) / SWEEP);
      else if (step > 1) sweep = 1;
      const shuttleX = L.cloth.x - (L.phone ? 6 : 10) + sweep * (L.cloth.w + (L.phone ? 12 : 20));

      // ---- sentence
      const sent = sentence(g, L, tw.slot);

      // ---- step 0: the vector and its LayerNorm
      if (tw.vec > 0.01) {
        g.save();
        g.globalAlpha *= tw.vec;
        const xb = L.xBar;
        haloText(g, L.phone ? `x · vector at “${lastTok}” · ${d} numbers` : `x · final residual vector at “${lastTok}” · ${d} numbers`, xb.x, xb.y - 12, { size: L.phone ? 9 : 10.5, align: 'left', color: AM.col.mist, halo: false });
        haloText(g, `‖x‖ = ${fmt(xN, 1)}`, xb.x + xb.w, xb.y - 12, { size: L.phone ? 9 : 10.5, align: 'right', color: AM.col.linenDim, halo: false });
        g.fillStyle = AM.rgba(AM.col.ink3, 0.6);
        D.roundRect(g, xb.x - 4, xb.y - 3, xb.w + 8, xb.h + 6, 6); g.fill();
        D.vectorBars(g, xb.x, xb.y, xb.w, xb.h, A.x, { max: xMax });
        haloText(g, `mean ${fmt(A.ln ? A.ln.mu : 0, 2)} · std ${fmt(A.ln ? A.ln.sd : 0, 2)}`, xb.x, xb.y + xb.h + 13, { size: L.phone ? 8.5 : 10, align: 'left', color: AM.col.mist, halo: false });
        // LN pill between the two
        const midY = (xb.y + xb.h + 18 + L.hBar.y - 14) / 2;
        const pillW = textW(g, 'final LayerNorm  LN_f', L.phone ? 9 : 10.5, 'mono') + 28;
        g.save();
        D.roundRect(g, w / 2 - pillW / 2, midY - 12, pillW, 24, 12);
        g.fillStyle = AM.mix(AM.col.ink2, AM.dye.weld, 0.12); g.fill();
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.7); g.lineWidth = 1; g.stroke();
        g.restore();
        haloText(g, 'final LayerNorm  LN_f', w / 2, midY + 0.5, { size: L.phone ? 9 : 10.5, color: AM.dye.weld, halo: false });
        D.arrow(g, w / 2, xb.y + xb.h + 22, w / 2, midY - 14, { color: AM.rgba(AM.dye.weld, 0.6), head: 6 });
        D.arrow(g, w / 2, midY + 13, w / 2, L.hBar.y - 22, { color: AM.rgba(AM.dye.weld, 0.6), head: 6 });
        const hb = L.hBar;
        haloText(g, 'h = LN_f(x)', hb.x, hb.y - 12, { size: L.phone ? 9 : 10.5, align: 'left', color: AM.col.mist, halo: false });
        haloText(g, `‖h‖ = ${fmt(hN, 1)}`, hb.x + hb.w, hb.y - 12, { size: L.phone ? 9 : 10.5, align: 'right', color: AM.col.linenDim, halo: false });
        g.fillStyle = AM.rgba(AM.col.ink3, 0.6);
        D.roundRect(g, hb.x - 4, hb.y - 3, hb.w + 8, hb.h + 6, 6); g.fill();
        if (A.ln) haloText(g, `mean ${fmt(A.ln.muH, 2)} · std ${fmt(A.ln.sdH, 2)} · scale ±${fmt(hMax, 1)} (x was ±${fmt(xMax, 1)})`, hb.x, hb.y + hb.h + 13, { size: L.phone ? 8.5 : 10, align: 'left', color: AM.col.mist, halo: false });
        g.restore();
        // thread from the last token into x
        if (sent.lastBox) {
          const lb = sent.lastBox;
          silk(g, [lb.cx, lb.y + lb.h, lb.cx, lb.y + lb.h + 30, w / 2, L.xBar.y - 50, w / 2, L.xBar.y - 22], { color: AM.dye.weld, width: 1.4, alpha: 0.65 * tw.vec });
        }
      }

      // ---- step 1: the cloth (W_U) and the shuttle
      if (tw.cloth > 0.01 && U) {
        const cl = L.cloth;
        const img = clothImage(L);
        g.save();
        g.globalAlpha *= tw.cloth;
        haloText(g, L.phone ? `W_U · ${d} × ${V} · a column per token` : `W_U · ${d} rows × ${V} columns, one column per token`, cl.x, cl.y - 9, { size: L.phone ? 8.5 : 10, align: 'left', color: AM.col.mist, halo: false });
        g.drawImage(img, cl.x, cl.y, cl.w, cl.h);
        // columns the shuttle has passed glow faintly gold
        const passedW = clamp(shuttleX - cl.x, 0, cl.w);
        if (passedW > 0) {
          g.save();
          g.globalCompositeOperation = 'lighter';
          const gr = g.createLinearGradient(cl.x + passedW - 70, 0, cl.x + passedW, 0);
          gr.addColorStop(0, AM.rgba(AM.dye.weld, 0.0));
          gr.addColorStop(1, AM.rgba(AM.dye.weld, 0.22));
          g.fillStyle = gr;
          g.fillRect(cl.x, cl.y, passedW, cl.h);
          g.restore();
        }
        g.restore();
      }
      // the h strip: barcode in step 0, shuttle in step 1
      if (A.ln && (tw.vec > 0.01 || tw.cloth > 0.01)) {
        const e = clamp(tw.cloth / Math.max(1e-3, tw.cloth + tw.vec));
        const rects = hRects(L, e, shuttleX);
        const alpha = Math.max(tw.vec, tw.cloth);
        g.save();
        if (e > 0.5) {
          // shuttle spine with glow
          const gr = g.createLinearGradient(shuttleX - 18, 0, shuttleX + 18, 0);
          gr.addColorStop(0, AM.rgba(AM.dye.weld, 0)); gr.addColorStop(0.5, AM.rgba(AM.dye.weld, 0.22 * alpha)); gr.addColorStop(1, AM.rgba(AM.dye.weld, 0));
          g.fillStyle = gr;
          g.fillRect(shuttleX - 18, L.cloth.y - 6, 36, L.cloth.h + 12);
          g.fillStyle = AM.rgba(AM.dye.weld, 0.9 * alpha * (e - 0.5) * 2);
          g.fillRect(shuttleX - 0.75, L.cloth.y - 6, 1.5, L.cloth.h + 12);
        }
        for (const rc of rects) {
          g.globalAlpha = alpha * (0.4 + 0.6 * Math.abs(rc.v));
          g.fillStyle = rc.v >= 0 ? AM.dye.woad : AM.dye.madder;
          g.fillRect(Math.min(rc.x0, rc.x1), Math.min(rc.y0, rc.y1), Math.max(0.8, Math.abs(rc.x1 - rc.x0)), Math.max(0.8, Math.abs(rc.y1 - rc.y0)));
        }
        g.restore();
        if (e > 0.6 && sent.lastBox && tw.cloth > 0.05) {
          haloText(g, 'h', shuttleX, L.cloth.y + L.cloth.h + 9, { size: L.phone ? 10 : 11, color: AM.dye.weld, weight: 500 });
        }
      }

      // ---- the logit fringe → probability skyline → sorted bars
      const fieldTop = L.field.y + L.labelRoom, fieldBot = L.field.y + L.field.h;
      const zlo = zmin - 0.5, zhi = zmax + 0.5;
      const yLogit = (v) => fieldBot - ((v - zlo) / (zhi - zlo)) * (fieldBot - fieldTop);
      const chartBase = L.chart.y + L.chart.h, chartTop = L.chart.y + (L.phone ? 16 : 18);
      const barBase = L.bars.y + L.bars.h, barTop = L.bars.y;
      const colW = L.W / V;
      const ntop = L.ntop;
      const topW = (L.W * 0.8) / ntop;
      const sortX = (r) => (r < ntop ? L.left + (r + 0.5) * topW : L.left + L.W * 0.83 + ((r - ntop + 0.5) / (V - ntop)) * L.W * 0.17);
      const sortBW = (r) => (r < ntop ? topW * 0.46 : Math.max(0.8, ((L.W * 0.17) / (V - ntop)) * 0.6));
      const eF = tw.field, eP = tw.prob, eS = tw.sort;

      if (eF > 0.01) {
        // axes
        g.save();
        if (eP < 0.99) {
          g.globalAlpha *= eF * (1 - eP);
          const y0 = yLogit(0);
          g.strokeStyle = AM.rgba(AM.col.linen, 0.25); g.setLineDash([3, 4]); g.lineWidth = 1;
          g.beginPath(); g.moveTo(L.left, y0); g.lineTo(L.right, y0); g.stroke(); g.setLineDash([]);
          haloText(g, 'logit 0', L.right, y0 - 8, { size: L.phone ? 8.5 : 9.5, align: 'right', color: AM.col.mist });
          haloText(g, fmt(zmin, 1), L.right, fieldBot + 9, { size: L.phone ? 8.5 : 9.5, align: 'right', color: AM.col.mist });
        }
        g.restore();
        if (eP > 0.01) {
          g.save();
          g.globalAlpha *= eP;
          const base0 = lerp(chartBase, barBase, eS), top0 = lerp(chartTop, barTop, eS);
          g.strokeStyle = AM.rgba(AM.col.linen, 0.07); g.lineWidth = 1;
          for (const f of [0.25, 0.5, 0.75, 1]) {
            const y = base0 - (base0 - top0) * f;
            g.beginPath(); g.moveTo(L.left, y); g.lineTo(L.right, y); g.stroke();
            haloText(g, f === 1 ? 'p = 1' : String(f), L.right, y - 7, { size: L.phone ? 8 : 9, align: 'right', color: AM.col.mist, alpha: 0.8 });
          }
          g.strokeStyle = AM.rgba(AM.col.linen, 0.22);
          g.beginPath(); g.moveTo(L.left, base0); g.lineTo(L.right, base0); g.stroke();
          g.restore();
        }
        // x axis: until the bars are sorted, the columns are the vocabulary in id order
        {
          const aX = eF * (1 - eS) * (step === 1 ? clamp(sweep * 3) : 1);
          if (aX > 0.01) haloText(g, L.phone ? 'vocab order →' : 'vocabulary order →', L.left, fieldBot + (L.phone ? 6 : 9), { size: L.phone ? 8 : 9.5, align: 'left', color: AM.col.mist, alpha: aX, halo: false });
        }

        // columns
        const clothBot = L.cloth.y + L.cloth.h;
        for (let j = 0; j < V; j++) {
          const r = rankOf[j];
          let reveal = 1;
          if (step === 1) reveal = easeOut(clamp(((sweep - (j + 0.5) / V) * SWEEP) / 0.45));
          else if (step === 0) reveal = 0;
          const aCol = eF * Math.max(reveal, eP);
          if (aCol < 0.01) continue;
          const isTop = r < 8;
          const xV = L.left + (j + 0.5) * colW;
          const es = easeInOut(clamp(eS * 1.25 - (r / V) * 0.25));
          const x = lerp(xV, sortX(r), es);
          const yL = lerp(clothBot, yLogit(z[j]), reveal);
          const pv = probs[j];
          const yP = chartBase - (chartBase - chartTop) * pv;
          const qd = r < S.qDisp.length ? S.qDisp[r] : 0;
          const yS = barBase - (barBase - barTop) * qd;
          const yTop = lerp(lerp(yL, yP, eP), yS, es);
          const baseY = lerp(chartBase, barBase, es);
          // warp thread from the cloth down to the bead (logit view)
          if (eP < 0.98) {
            g.strokeStyle = AM.rgba(isTop ? AM.dye.weld : AM.col.linen, (isTop ? 0.4 : 0.1) * aCol * (1 - eP));
            g.lineWidth = isTop ? 1 : 0.6;
            g.beginPath(); g.moveTo(xV, clothBot); g.lineTo(xV, yL); g.stroke();
          }
          // probability bar
          if (eP > 0.01) {
            const bw = lerp(Math.max(1, colW * (isTop ? 0.9 : 0.62)), sortBW(r), es);
            const col = es > 0.5 ? rankColor(r) : (isTop ? AM.dye.weld : AM.col.linenDim);
            const hgt = Math.max(0, baseY - yTop);
            if (hgt > 0.3) {
              const kept = step === 5 ? r < S.cur.keep : true;
              const gr = g.createLinearGradient(0, yTop, 0, baseY);
              gr.addColorStop(0, AM.rgba(col, (kept ? 0.95 : 0.25) * eP));
              gr.addColorStop(1, AM.rgba(col, (kept ? 0.25 : 0.06) * eP));
              g.fillStyle = gr;
              g.fillRect(x - bw / 2, yTop, bw, hgt);
              if (isTop && bw > 3) { g.fillStyle = AM.rgba('#fff4d6', 0.55 * eP * (kept ? 1 : 0.3)); g.fillRect(x - bw / 2, yTop, bw, 1.2); }
            }
          }
          // in the skyline the bead rides on top of its bar
          if (isTop && eP > 0.02 && es < 0.98) D.glowDot(g, x, yTop, L.phone ? 1.8 : 2.3, AM.dye.weld, eP * (1 - es));
          // bead (logit view)
          if (eP < 0.98) {
            const a = aCol * (1 - eP);
            if (isTop) D.glowDot(g, xV, yL, L.phone ? 2.2 : 2.8, AM.dye.weld, a);
            else { g.fillStyle = AM.rgba(AM.col.linenDim, 0.55 * a); g.beginPath(); g.arc(xV, yL, L.phone ? 1 : 1.3, 0, TAU); g.fill(); }
          }
        }

        // labels: logit view
        if (eP < 0.95 && (step >= 2 || sweep > 0.98)) {
          const size = L.phone ? 8.5 : 10;
          const items = base.order.slice(0, L.phone ? 4 : 5).map((id) => ({ x: L.left + (id + 0.5) * colW, ay: yLogit(z[id]), text: `${m.vocab[id]} ${fmt(z[id], 2)}` }));
          spreadLabels(g, items, L, size, fieldTop - (L.phone ? 24 : 32));
          const a = (1 - eP) * eF * (step === 1 ? clamp((sweep - 0.98) * 50) : 1);
          drawLeaders(g, items, AM.dye.weld, a, size, AM.dye.weld);
        }
        // labels: probability skyline (vocab order)
        if (eP > 0.05 && eS < 0.95) {
          const size = L.phone ? 8.5 : 10;
          const items = base.order.slice(0, L.phone ? 4 : 6).map((id) => ({ x: L.left + (id + 0.5) * colW, ay: chartBase - (chartBase - chartTop) * probs[id], text: `${m.vocab[id]} ${pct(probs[id])}` }));
          const yMax = chartBase - (chartBase - chartTop) * probs[base.order[0]];
          spreadLabels(g, items, L, size, yMax - (L.phone ? 22 : 30));
          const a = eP * (1 - eS);
          drawLeaders(g, items, AM.dye.weld, a, size, AM.col.linen);
          // a worked example with the real numbers
          const id0 = base.order[0], e0 = Math.exp(z[id0]);
          const fs = L.phone ? 8.5 : 10.5, lh = fs + (L.phone ? 6 : 8);
          const x0 = L.left + 2, y0 = L.chart.y + (L.phone ? 2 : 6);
          haloText(g, `e^${fmt(z[id0], 2)} = ${e0.toFixed(1)}   (${m.vocab[id0]})`, x0, y0, { size: fs, align: 'left', color: AM.col.linenDim, alpha: a });
          haloText(g, `Σ e^z over all ${V} = ${sumE.toFixed(1)}`, x0, y0 + lh, { size: fs, align: 'left', color: AM.col.linenDim, alpha: a });
          haloText(g, `p(${m.vocab[id0]}) = ${e0.toFixed(1)} / ${sumE.toFixed(1)} = ${(e0 / sumE).toFixed(3)}`, x0, y0 + 2 * lh, { size: fs, align: 'left', color: AM.dye.weld, alpha: a });
          haloText(g, `Σ p = ${psum.toFixed(3)}`, x0, y0 + 3 * lh, { size: fs, align: 'left', color: AM.col.mist, alpha: a });
        }
      }

      // ---- sorted view: names, ribbon, threads, draws
      if (eS > 0.01) {
        const a = clamp((eS - 0.35) / 0.65);
        const rb = L.rib;
        const size = L.phone ? 9 : 11;
        g.save();
        g.globalAlpha *= a;
        // temperature: the T = 1 heights stay as dashed ghosts for comparison (drawn under the labels)
        if (step === 4) {
          g.save();
          g.strokeStyle = AM.rgba(AM.col.linen, 0.55); g.lineWidth = 1; g.setLineDash([3, 3]);
          for (let r = 0; r < ntop; r++) {
            const x = sortX(r), bw = sortBW(r) + 8;
            const y = barBase - (barBase - barTop) * base.q[r];
            g.beginPath(); g.moveTo(x - bw / 2, y); g.lineTo(x + bw / 2, y); g.stroke();
          }
          g.restore();
        }
        // names under the top bars + percentages above
        for (let r = 0; r < ntop; r++) {
          const x = sortX(r);
          const kept = step !== 5 || r < S.cur.keep;
          haloText(g, m.vocab[base.order[r]], x, barBase + 12, { size, role: 'body', weight: 600, color: kept ? AM.col.linen : AM.col.mist, alpha: kept ? 1 : 0.55, halo: false });
          const qd = S.qDisp[r];
          const yS = barBase - (barBase - barTop) * qd;
          if (qd > 0.004) haloText(g, pct(qd), x, yS - 9, { size: L.phone ? 8.5 : 10, color: kept ? rankColor(r) : AM.col.mist, alpha: kept ? 1 : 0.6 });
        }
        haloText(g, `${V - ntop} more →`, L.left + L.W * 0.915, barBase + 12, { size: L.phone ? 8 : 9, color: AM.col.mist, halo: false });

        // ribbon: the probabilities laid end to end on [0, 1)
        let acc = 0;
        const segs = [];
        for (let r = 0; r < S.qDisp.length; r++) {
          const q = S.qDisp[r];
          const x0 = rb.x + acc * rb.w, x1 = rb.x + (acc + q) * rb.w;
          segs.push({ x0, x1, r });
          acc += q;
        }
        g.fillStyle = AM.rgba(AM.col.ink, 0.8);
        D.roundRect(g, rb.x - 2, rb.y - 2, rb.w + 4, rb.h + 4, 6); g.fill();
        for (const sg of segs) {
          const wd = sg.x1 - sg.x0;
          if (wd < 0.05) continue;
          const col = rankColor(sg.r);
          const gr = g.createLinearGradient(0, rb.y, 0, rb.y + rb.h);
          gr.addColorStop(0, AM.rgba(col, 0.95)); gr.addColorStop(1, AM.rgba(col, 0.55));
          g.fillStyle = gr;
          g.fillRect(sg.x0 + (wd > 3 ? 0.6 : 0), rb.y, Math.max(0.4, wd - (wd > 3 ? 1.2 : 0)), rb.h);
          if (wd > (L.phone ? 34 : 42) && sg.r < ntop) {
            haloText(g, m.vocab[base.order[sg.r]], (sg.x0 + sg.x1) / 2, rb.y + rb.h / 2 + 0.5, { size: L.phone ? 8.5 : 10, role: "body", weight: 600, color: sg.r < 7 ? AM.col.ink : AM.col.linen, halo: false });
          }
        }
        // ruler
        g.strokeStyle = AM.rgba(AM.col.linen, 0.3); g.lineWidth = 1;
        for (const f of [0, 0.25, 0.5, 0.75, 1]) {
          const x = rb.x + f * rb.w;
          g.beginPath(); g.moveTo(x, rb.y + rb.h + 3); g.lineTo(x, rb.y + rb.h + 7); g.stroke();
          haloText(g, f === 0 ? '0' : f === 1 ? '1' : String(f), clamp(x, rb.x + 6, rb.x + rb.w - 6), rb.y + rb.h + 14, { size: L.phone ? 8 : 9, color: AM.col.mist, halo: false });
        }
        // silk threads from each top bar down to its segment
        for (let r = 0; r < ntop; r++) {
          const q = S.qDisp[r];
          if (q < 0.002) continue;
          const sg = segs[r];
          const x1 = sortX(r), y1 = barBase + 20;
          const x2 = (sg.x0 + sg.x1) / 2, y2 = rb.y - 2;
          const my = (y1 + y2) / 2;
          silk(g, [x1, y1, x1, my, x2, my, x2, y2], { color: rankColor(r), width: 0.6 + 1.6 * Math.sqrt(q), alpha: 0.55 });
        }
        // greedy marker (step 3)
        if (step === 3) {
          const x = sortX(0), yS = barBase - (barBase - barTop) * S.qDisp[0];
          haloText(g, 'argmax: greedy always takes this', clamp(x + (L.phone ? 6 : 10), L.left, L.right), yS - (L.phone ? 22 : 26), { size: L.phone ? 8.5 : 9.5, align: 'left', color: AM.dye.weld });
        }
        // readouts (steps 4, 5)
        const roY = L.bars.y - (L.phone ? 14 : 18);
        if (step === 4) {
          haloText(g, `T = ${S.Tnow.toFixed(2)}`, L.left, roY, { size: L.phone ? 11 : 13, align: 'left', color: AM.dye.weld, weight: 500 });
          haloText(g, `≈ ${fmt(target.eff, 1)} effective choices${L.phone ? '' : ' · dashed: T = 1'}`, L.left + (L.phone ? 74 : 92), roY, { size: L.phone ? 9 : 10.5, align: 'left', color: AM.col.linenDim });
        } else if (step === 5) {
          const lbl = S.cutMode === 0 ? `top-k = 3 · ${target.keep} words kept, renormalised` : `top-p = 0.9 · ${target.keep} words kept, renormalised`;
          haloText(g, lbl, L.left, roY, { size: L.phone ? 9.5 : 11, align: 'left', color: AM.dye.weld });
          // cut line
          const r = target.keep;
          if (r < ntop) {
            const x = (sortX(r - 1) + sortX(r)) / 2;
            g.save(); g.strokeStyle = AM.rgba(AM.dye.madder, 0.8); g.setLineDash([4, 4]); g.lineWidth = 1.2;
            g.beginPath(); g.moveTo(x, barTop - 4); g.lineTo(x, barBase + 4); g.stroke(); g.restore();
            haloText(g, 'cut', x + 4, barTop + 10, { size: 9, align: 'left', color: AM.dye.madder });
          }
        } else if (step === 3) {
          haloText(g, L.phone ? 'sorted · the model’s own odds (T = 1)' : 'sorted by probability · the model’s own odds (T = 1)', L.left, roY, { size: L.phone ? 9 : 10.5, align: 'left', color: AM.col.mist });
        }
        g.restore();

        // ---- a draw every few seconds: u falls onto the ribbon, the word flies into the sentence
        if (step >= 3 && a > 0.5) {
          // a new draw only starts once the bars have settled, so it uses exactly the distribution shown
          if (!S.drop && t >= S.nextDrop && lag < 0.002) S.drop = { t0: t, u: S.rng(), r: -1 };
          const dp = S.drop;
          if (dp) {
            const FALL = AM.reducedMotion ? 0.01 : 0.55, HOLD = AM.reducedMotion ? 0.6 : 0.85, FLY = AM.reducedMotion ? 0.01 : 0.65;
            const el2 = t - dp.t0;
            const xu = rb.x + dp.u * rb.w;
            if (el2 >= FALL && dp.r < 0) {
              // decide at landing, from the ribbon as drawn
              let acc2 = 0, rr = -1, lastLive = 0;
              for (let r = 0; r < S.qDisp.length; r++) {
                if (S.qDisp[r] > 1e-9) lastLive = r;
                acc2 += S.qDisp[r];
                if (dp.u < acc2) { rr = r; break; }
              }
              dp.r = rr >= 0 ? rr : lastLive; // rounding can leave u past the end: take the last word still in play
            }
            if (el2 < FALL) {
              const f = M.ease.in(el2 / FALL);
              const y = lerp(rb.y - (L.phone ? 34 : 48), rb.y + rb.h / 2, f);
              D.glowDot(g, xu, y, L.phone ? 3 : 3.6, AM.col.linen, 1);
              // same spot as the landing readout, below the word labels, so the two rows never collide
              haloText(g, `u = ${dp.u.toFixed(3)}`, clamp(xu, L.left + 60, L.right - 60), rb.y - (L.phone ? 14 : 18), { size: L.phone ? 8.5 : 10, color: AM.col.linen, alpha: 0.9 });
            } else if (dp.r >= 0) {
              // highlight the chosen segment
              let x0 = rb.x; for (let r = 0; r < dp.r; r++) x0 += S.qDisp[r] * rb.w;
              const x1 = x0 + S.qDisp[dp.r] * rb.w;
              const fl = clamp(1 - (el2 - FALL) / (HOLD + FLY));
              g.save();
              g.strokeStyle = AM.rgba('#fff4d6', 0.9 * fl); g.lineWidth = 2;
              g.shadowColor = AM.rgba(AM.dye.weld, 0.9); g.shadowBlur = 14 * fl;
              g.strokeRect(x0, rb.y - 1, Math.max(2, x1 - x0), rb.h + 2);
              g.restore();
              D.glowDot(g, xu, rb.y + rb.h / 2, L.phone ? 3 : 3.6, AM.col.linen, fl);
              const fe = clamp((el2 - FALL - HOLD) / FLY);
              // the readout steps aside as the word lifts off, so the two never overlap
              haloText(g, `u = ${dp.u.toFixed(3)} → ${m.vocab[base.order[dp.r]]}`, clamp(xu, L.left + 60, L.right - 60), rb.y - (L.phone ? 14 : 18), { size: L.phone ? 8.5 : 10, color: AM.dye.weld, alpha: fl * (1 - clamp((el2 - FALL - HOLD + 0.18) / 0.18)) });
              if (fe > 0 && fe < 1 && sent.slotBox) {
                const sb = sent.slotBox;
                const P = [(x0 + x1) / 2, rb.y, (x0 + x1) / 2, rb.y - 140, sb.cx, sb.cy + 120, sb.cx, sb.cy];
                const pt = cubic(P, easeInOut(fe));
                silk(g, P, { color: AM.dye.weld, width: 1, alpha: 0.25 * (1 - fe) });
                haloText(g, m.vocab[base.order[dp.r]], pt.x, pt.y, { size: L.phone ? 11 : 13, role: 'body', weight: 600, color: AM.dye.weld });
              }
              if (el2 >= FALL + HOLD + FLY) {
                S.slotWord = m.vocab[base.order[dp.r]];
                S.slotPop = t;
                S.drop = null;
                S.nextDrop = t + (AM.reducedMotion ? 2.5 : 1.3);
              }
            }
          }
        }
      }

      // ---- formula line
      const FORM = [
        'h = γ ⊙ (x − μ) / √(σ² + ε) + β',
        `z = h · W_U + b     (${V} logits)`,
        'p = softmax(z) = e^z / Σ e^z',
        'draw u ~ U[0, 1), take the segment that holds u',
        'p ∝ exp(z / T)',
        L.phone ? 'top k, or smallest set with mass ≥ p; renormalise' : 'keep the top k, or the smallest set with mass ≥ p; renormalise',
      ];
      const fa = AM.reducedMotion ? 1 : clamp(age / 0.5);
      haloText(g, FORM[step] || '', w / 2, h - (L.phone ? 12 : 16), { size: L.phone ? 9 : 11, color: AM.col.linenDim, alpha: 0.9 * fa, halo: false });
    }

    cv.onResize(() => { S.clothKey = ''; draw(); });
    const seen = inView(cv.canvas);
    ctx.loop(() => { if (seen.on) draw(); });
    draw();
  }

  // ====================================================================
  // 6. The wheel: sample, append, repeat (hero, free play)
  // ====================================================================
  function buildWheel(body, ctx, m, banned) {
    const el = ctx.el, ui = AM.ui;
    const NCTX = m.config.n_ctx;
    const NV = m.vocab.length - banned.size; // tokens a sampler may emit
    /** Punctuation is hard to see on its own, so the centre shows it in quotes. */
    const shown = (id) => { const w = m.vocab[id]; return /^[.,]$/.test(w) ? `“${w}”` : w; };

    const st = {
      preset: 0, ids: [], promptLen: 0, gen: [],
      logits: null, S: null, prevS: null,
      T: 1, k: 0, p: 1,
      seed: 1, rng: null,
      phase: 'idle', t0: 0, auto: false, autoNext: 0,
      spin: null, bead: 0,
      reveal: 1, passes: 0, positions: 0,
      hover: -1, fly: null, weaveFrom: 0, fullAt: 0,
      wheelImg: null, wheelFor: null, oldImg: null, dirty: true,
    };

    // ---- "things to try", with live numbers
    const odds = (text, o) => { const e = m.encode(text); const r = m.run(e.ids); return shapeDist(r.logits[e.ids.length - 1], banned, o); };
    const capT1 = odds(PRESETS[4].text, { T: 1 }), capT25 = odds(PRESETS[4].text, { T: 2.5 });
    const coldP = odds(PRESETS[1].text, { p: 0.5 });
    const tryLine = el('p', { html: `<strong>Try:</strong> Greedy on “the king”, then Auto-write, and watch it get stuck in a loop. Temperature 2.5 on “the capital of japan is”: <em>${m.vocab[capT25.order[0]]}</em> falls from ${pct(capT1.q[0])} to ${pct(capT25.q[0])}, so ${pct(1 - capT25.q[0])} of draws now give some other word, such as <em>${m.vocab[capT25.order[1]]}</em> or <em>${m.vocab[capT25.order[2]]}</em>. Top-p 0.5 on “… she was cold . the”: ${NV} candidates shrink to ${coldP.keep}.` });

    // ---- DOM
    const intro = ctx.subhead('Free play', 'Spin the wheel', 'This is the whole loop on the live model. The inner wheel is the distribution for the next token after your temperature, top-k and top-p: each word’s arc is its probability. Spin, and the bead stops at a random point u. The word under it joins the text in the outer ring, which is the model’s whole context window of 32 slots. The model reads everything in the ring again, and the wheel re-forms for the next token.');
    intro.classList.add('pr-hero-intro');
    tryLine.classList.add('subhead-lead');
    intro.appendChild(tryLine);
    body.appendChild(intro);
    const fig = el('figure', { class: 'fig ch-wide pr-hero' });
    body.appendChild(fig);
    fig.appendChild(el('div', { class: 'fig-top' }, el('span', { class: 'fig-title' }, 'The wheel · sample, append, repeat'), ui.badge('live')));

    const grid = el('div', { class: 'pr-grid' });
    fig.appendChild(grid);
    const wheelBox = el('div', { class: 'pr-wheelbox' });
    const panel = el('div', { class: 'pr-panel' });
    grid.append(wheelBox, panel);

    const cv = ctx.canvas(wheelBox, {
      label: 'A wheel of next-word probabilities inside a ring of 32 context slots.',
      height: (w) => Math.round(Math.min(w, 640)),
    });
    cv.canvas.style.cursor = 'pointer';
    const live = el('div', { class: 'sr-only', 'aria-live': 'polite' });
    wheelBox.appendChild(live);

    // prompt chips
    const chips = PRESETS.map((pr, i) => el('button', {
      type: 'button', class: 'pr-chip', id: `pr-prompt-${i}`, 'aria-pressed': String(i === st.preset),
      onclick: () => { st.preset = i; chips.forEach((c, j) => c.setAttribute('aria-pressed', String(j === i))); reset(); },
    }, pr.label));
    panel.appendChild(el('div', { class: 'pr-group' }, el('span', { class: 'pr-group-label' }, 'Prompt'), el('div', { class: 'pr-chips', role: 'group', 'aria-label': 'Prompt' }, chips)));

    // sliders
    const sT = ui.slider({ id: 'pr-temp', label: 'Temperature T', min: 0, max: 3, step: 0.05, value: 1, format: (v) => (v <= 0 ? '0 · greedy' : v.toFixed(2)), onInput: (v) => { st.T = v; reshape(); } });
    const sK = ui.slider({ id: 'pr-topk', label: 'Top-k', min: 1, max: NV, step: 1, value: NV, format: (v) => (v >= NV ? `off (all ${NV})` : String(v)), onInput: (v) => { st.k = v >= NV ? 0 : v; reshape(); } });
    const sP = ui.slider({ id: 'pr-topp', label: 'Top-p', min: 0.05, max: 1, step: 0.01, value: 1, format: (v) => (v >= 1 ? 'off (1.00)' : v.toFixed(2)), onInput: (v) => { st.p = v; reshape(); } });
    panel.appendChild(el('div', { class: 'pr-sliders' }, sT.el, sK.el, sP.el));

    // buttons + seed
    const bSpin = ui.button({ id: 'pr-spin', label: 'Spin', kind: 'primary', onClick: () => { stopAuto(); spin(); } });
    const bAuto = ui.button({ id: 'pr-auto', label: 'Auto-write', onClick: () => toggleAuto() });
    const bGreedy = ui.button({ id: 'pr-greedy', label: 'Greedy', title: 'Set the temperature to 0', onClick: () => { sT.set(0); st.T = 0; reshape(); } });
    const bReset = ui.button({ id: 'pr-reset', label: 'Reset', onClick: () => reset() });
    const seedOut = el('output', { id: 'pr-seed-val', 'aria-live': 'polite' }, String(st.seed));
    const seedBox = el('span', { class: 'pr-seed' }, 'Seed',
      el('button', { type: 'button', id: 'pr-seed-dn', 'aria-label': 'Previous seed', onclick: () => setSeed(st.seed - 1) }, '−'),
      seedOut,
      el('button', { type: 'button', id: 'pr-seed-up', 'aria-label': 'Next seed', onclick: () => setSeed(st.seed + 1) }, '+'));
    wheelBox.appendChild(el('div', { class: 'pr-buttons' }, bSpin, bAuto, bGreedy, bReset, seedBox));

    // readouts
    const dd = () => el('dd');
    const rPlay = dd(), rEnt = dd(), rTop = dd(), rCost = dd();
    panel.appendChild(el('dl', { class: 'pr-stats' },
      el('dt', {}, 'Kept'), rPlay,
      el('dt', {}, 'Spread'), rEnt,
      el('dt', {}, 'Favourite'), rTop,
      el('dt', {}, 'Compute'), rCost));

    // transcript
    const trans = el('div', { class: 'pr-trans', role: 'log', 'aria-label': 'Text so far, with the model’s own three likeliest words under each drawn word' });
    fig.appendChild(el('div', { class: 'pr-trans-wrap' }, el('span', { class: 'pr-group-label' }, 'The text so far · faint: the model’s own top three at each step (T = 1)'), trans));
    fig.appendChild(el('figcaption', { html: 'Live model. The wheel is the tinyworld model’s real next-token distribution, reshaped by the sampling rule: divide the logits by T, keep the top k, keep the smallest set with mass ≥ p, renormalise. Arcs start at 12 o’clock in order of probability, so the bead’s stopping angle <em>is</em> the uniform random number u. &lt;pad&gt; and &lt;unk&gt; are never drawn. Our model re-reads the whole context for every token. Real systems keep a <span class="term">KV cache</span>: each earlier position’s keys and values are stored, so only the new token’s position goes through the layers, and its attention reads the cached keys and values of all the others.' }));

    // ---- model plumbing
    function runModel() {
      const r = m.run(st.ids);
      st.logits = r.logits[st.ids.length - 1];
      st.passes++;
      st.positions += st.ids.length;
    }
    function reshape() {
      st.S = shapeDist(st.logits, banned, { T: st.T, k: st.k, p: st.p });
      st.dirty = true;
      readouts();
    }
    function full() { return st.ids.length >= NCTX; }
    function setSeed(v) {
      st.seed = Math.max(1, Math.min(999, v));
      seedOut.textContent = String(st.seed);
      reset();
    }
    function reset() {
      stopAuto();
      const pr = PRESETS[st.preset];
      const { ids } = m.encode(pr.text);
      st.ids = ids.slice(0, NCTX);
      st.promptLen = st.ids.length;
      st.gen = [];
      st.rng = M.rng(st.seed * 7919 + st.preset * 104729 + 1);
      st.passes = 0; st.positions = 0;
      st.phase = 'idle'; st.spin = null; st.fly = null;
      st.reveal = 1; st.prevS = null;
      runModel();
      reshape();
      renderTranscript();
      live.textContent = `Prompt: ${pr.text}`;
    }
    function readouts() {
      const S = st.S;
      if (!S) return;
      const why = S.keep === NV ? 'nothing cut' : st.T <= 0 ? 'greedy keeps only the argmax' : 'after top-k / top-p';
      rPlay.innerHTML = `<b>${S.keep}</b> of ${NV} tokens <span style="color:var(--mist)">(${why})</span>`;
      rEnt.innerHTML = S.keep === 1 ? 'one choice: no randomness left' : `entropy ${fmt(S.H, 2)} nats ≈ <b>${fmt(S.eff, 1)}</b> effective choices`;
      rTop.innerHTML = `<b>${shown(S.order[0])}</b> ${pct(S.q[0])}`;
      const cached = st.promptLen + Math.max(0, st.passes - 1);
      rCost.innerHTML = `${st.passes} forward pass${st.passes === 1 ? '' : 'es'} · ${st.positions} positions computed <span style="color:var(--mist)">(${cached} with a KV cache)</span>`;
      const top = [];
      for (let r = 0; r < Math.min(4, S.keep); r++) top.push(`${m.vocab[S.order[r]]} ${pct(S.q[r])}`);
      cv.canvas.setAttribute('aria-label', `Wheel of next-word probabilities after “${m.decode(st.ids).join(' ')}”: ${top.join(', ')}${S.keep > 4 ? ` and ${S.keep - 4} more` : ''}. The outer ring holds the ${st.ids.length} of ${NCTX} context tokens.`);
    }
    function renderTranscript() {
      const kids = [];
      st.ids.forEach((id, i) => {
        if (i < st.promptLen) { kids.push(el('span', { class: 'pr-tw is-prompt' }, m.vocab[id])); return; }
        const gi = st.gen[i - st.promptLen];
        const alts = gi.top3.map((a) => el('span', { class: a.id === gi.id ? 'is-pick' : '' }, `${m.vocab[a.id]} ${pct(a.p)}`));
        if (!gi.top3.some((a) => a.id === gi.id)) alts.push(el('span', { class: 'is-other' }, `↳ ${m.vocab[gi.id]} ${pct(gi.p)}`));
        kids.push(el('span', { class: 'pr-tw is-gen' + (i === st.ids.length - 1 ? ' is-new' : '') }, el('span', { class: 'pr-tw-word' }, m.vocab[id]), el('span', { class: 'pr-tw-alts' }, alts)));
      });
      if (full()) kids.push(el('span', { class: 'pr-full' }, `· context full: ${NCTX} tokens`));
      trans.replaceChildren(...kids);
      bSpin.disabled = full();
      bAuto.disabled = full();
    }

    // ---- spinning
    /** Only one token left in play (greedy, top-k 1, or a model that is sure): nothing to spin for. */
    const sureSpin = () => !!(st.spin && st.spin.S.keep === 1);
    function dur(name) {
      if (AM.reducedMotion) return 0;
      if (st.auto && sureSpin()) return { spin: 0.3, land: 0.06, fly: 0.3, weave: 0.34 }[name];
      const fast = st.auto;
      return { spin: fast ? 0.65 : 2.6, land: fast ? 0.14 : 0.6, fly: fast ? 0.36 : 0.75, weave: fast ? 0.42 : 0.85 }[name];
    }
    function spin() {
      if (st.phase !== 'idle' || full() || !st.S) return;
      const S = st.S;
      const u = st.rng();
      const r = pickRank(S.cum, S.keep, u);
      const id = S.order[r];
      // the faint alternatives show the model's own odds (T = 1, nothing cut)
      const raw = shapeDist(st.logits, banned, { T: 1 });
      const top3 = [];
      for (let k = 0; k < 3; k++) top3.push({ id: raw.order[k], p: raw.q[k] });
      const pRaw = raw.q[raw.rank.get(id)];
      const turns = st.auto ? (S.keep === 1 ? 0 : 1) : 3;
      let b1 = Math.floor(st.bead) + turns + u;
      if (b1 < st.bead + turns - 1e-6) b1 += 1;
      st.spin = { u, r, id, p: S.q[r], pRaw, top3, b0: st.bead, b1, S };
      setPhase('spin');
      if (AM.reducedMotion) { st.bead = b1; commit(); }
    }
    function setPhase(ph) { st.phase = ph; st.t0 = now(); }
    /** Append the drawn token and run the model again. */
    function commit() {
      const sp = st.spin;
      if (!sp) return;
      st.ids.push(sp.id);
      st.gen.push({ id: sp.id, p: sp.pRaw, u: sp.u, top3: sp.top3 });
      st.prevS = st.S;
      st.oldImg = st.wheelImg; st.wheelImg = null;
      runModel();
      reshape();
      renderTranscript();
      live.textContent = `Drew “${m.vocab[sp.id]}” (probability ${pct(sp.p)}, u = ${sp.u.toFixed(3)}).`;
      st.reveal = AM.reducedMotion ? 1 : 0;
      st.weaveFrom = now();
      setPhase(AM.reducedMotion ? 'idle' : 'weave');
      if (AM.reducedMotion) { st.spin = null; if (st.auto) st.autoNext = now() + 0.7; }
    }
    function toggleAuto() {
      if (st.auto) { stopAuto(); return; }
      if (full()) return;
      st.auto = true;
      bAuto.textContent = 'Pause';
      bAuto.setAttribute('aria-pressed', 'true');
      if (st.phase === 'idle') spin();
    }
    function stopAuto() {
      st.auto = false;
      bAuto.textContent = 'Auto-write';
      bAuto.setAttribute('aria-pressed', 'false');
    }

    function tick() {
      const t = now();
      const el2 = t - st.t0;
      if (st.phase === 'spin') {
        const D0 = dur('spin');
        const e = D0 ? clamp(el2 / D0) : 1;
        st.bead = st.spin.b0 + (st.spin.b1 - st.spin.b0) * easeOutQuart(e);
        if (e >= 1) setPhase('land');
      } else if (st.phase === 'land') {
        if (el2 >= dur('land')) { setPhase('fly'); st.fly = { id: st.spin.id, r: st.spin.r, slot: st.ids.length }; }
      } else if (st.phase === 'fly') {
        if (el2 >= dur('fly')) { st.fly = null; commit(); }
      } else if (st.phase === 'weave') {
        const D1 = dur('weave');
        st.reveal = D1 ? easeInOut(clamp(el2 / D1)) : 1;
        if (el2 >= D1) { st.reveal = 1; st.oldImg = null; st.spin = null; setPhase('idle'); if (st.auto) st.autoNext = t + 0.12; }
      } else if (st.phase === 'idle') {
        if (st.auto) {
          if (full()) stopAuto();
          else if (t >= st.autoNext) spin();
        }
      }
    }

    // ---- geometry
    function geo(w, h) {
      const S = Math.min(w, h);
      const phone = S < 460;
      const cx = w / 2, cy = h / 2;
      const R = S / 2 - (phone ? 4 : 8);
      const rTxt = R * (phone ? 0.7 : 0.71);
      const rBody = rTxt - (phone ? 5 : 7);
      const rTrack = rBody - (phone ? 9 : 12);
      const rw1 = rTrack - (phone ? 6 : 8);
      const rw0 = rw1 - S * (phone ? 0.115 : 0.12);
      const rc = rw0 - (phone ? 6 : 9);
      return { w, h, S, phone, cx, cy, R, rTxt, rBody, rTrack, rw1, rw0, rc };
    }
    const slotAng = (i) => -Math.PI / 2 + ((i + 0.5) / NCTX) * TAU;
    const uAng = (u) => -Math.PI / 2 + u * TAU;

    /** Radial text: reads outward on the right half, flipped on the left half so it is never upside down. */
    function radialText(g, str, cx, cy, a, r0, o) {
      const c = Math.cos(a), s = Math.sin(a);
      g.save();
      g.translate(cx + c * r0, cy + s * r0);
      const left = c < -1e-6;
      g.rotate(left ? a + Math.PI : a);
      g.font = AM.font(o.size, o.role || 'body', o.weight || 600);
      g.textAlign = o.center ? 'center' : (left ? 'right' : 'left');
      g.textBaseline = 'middle';
      g.globalAlpha *= o.alpha ?? 1;
      if (o.halo) { g.lineJoin = 'round'; g.strokeStyle = o.halo; g.lineWidth = 3; g.strokeText(str, 0, 0.5); }
      g.fillStyle = o.color;
      g.fillText(str, 0, 0.5);
      g.restore();
    }

    /** A word written across its arc, shrunk to fit the band; skipped when the arc is too thin. */
    function arcLabel(g, G, S, r, a0, a1, rm) {
      const size = G.phone ? 9.5 : 12;
      if ((a1 - a0) * rm < size + 4) return;
      const word = shown(S.order[r]);
      if (word !== m.vocab[S.order[r]]) {
        // punctuation: upright and larger, so a full stop never reads as a stray speck
        const a = (a0 + a1) / 2;
        g.save();
        g.font = AM.font(size * 1.5, 'body', 700);
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillStyle = r < 7 ? AM.col.ink : AM.col.linen;
        g.fillText(word, G.cx + Math.cos(a) * rm, G.cy + Math.sin(a) * rm);
        g.restore();
        return;
      }
      const room = G.rw1 - G.rw0 - 8;
      let fs = size;
      const ww = textW(g, word, fs, 'body', 700);
      if (ww > room) fs = Math.max(7, (fs * room) / ww);
      radialText(g, word, G.cx, G.cy, (a0 + a1) / 2, rm, { size: fs, color: r < 7 ? AM.col.ink : AM.col.linen, center: true, weight: 700 });
    }

    // ---- the wheel layer (cached; rebuilt when the distribution or size changes)
    function buildWheelImage(G, S) {
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(G.w * cv.dpr));
      c.height = Math.max(1, Math.round(G.h * cv.dpr));
      const g = c.getContext('2d');
      g.setTransform(cv.dpr, 0, 0, cv.dpr, 0, 0);
      const { cx, cy, rw0, rw1 } = G;
      const rm = (rw0 + rw1) / 2;
      // a twill weave laid over every arc
      const tc = document.createElement('canvas');
      tc.width = tc.height = 6;
      const tx = tc.getContext('2d');
      tx.strokeStyle = 'rgba(13,17,33,0.9)'; tx.lineWidth = 1.1;
      tx.beginPath(); tx.moveTo(-1, 7); tx.lineTo(7, -1); tx.moveTo(-1, 1); tx.lineTo(1, -1); tx.moveTo(5, 7); tx.lineTo(7, 5); tx.stroke();
      const twill = g.createPattern(tc, 'repeat');
      for (let r = 0; r < S.keep; r++) {
        const q = S.q[r];
        if (q <= 0) continue;
        const a0 = uAng(S.cum[r]), a1 = uAng(S.cum[r + 1]);
        const span = a1 - a0;
        const col = rankColor(r);
        if (span * rw1 < 1.2) {
          // a hairline
          g.strokeStyle = AM.rgba(col, 0.7);
          g.lineWidth = Math.max(0.35, span * rw1);
          g.beginPath(); g.moveTo(cx + Math.cos(a0) * rw0, cy + Math.sin(a0) * rw0); g.lineTo(cx + Math.cos(a0) * rw1, cy + Math.sin(a0) * rw1); g.stroke();
          continue;
        }
        const gap = Math.min(span * 0.12, (G.phone ? 0.9 : 1.2) / rw1);
        const gr = g.createRadialGradient(cx, cy, rw0, cx, cy, rw1);
        gr.addColorStop(0, AM.rgba(col, 0.32));
        gr.addColorStop(0.7, AM.rgba(col, 0.78));
        gr.addColorStop(1, AM.rgba(col, 0.98));
        g.fillStyle = gr;
        g.beginPath();
        g.arc(cx, cy, rw1, a0 + gap, a1 - gap);
        g.arc(cx, cy, rw0, a1 - gap, a0 + gap, true);
        g.closePath();
        g.fill();
        if (twill) { g.save(); g.globalAlpha = 0.09; g.fillStyle = twill; g.fill(); g.restore(); }
        // weft: two fine concentric threads across the band
        g.strokeStyle = AM.rgba(AM.col.ink, 0.16); g.lineWidth = 0.8;
        for (const f of [0.36, 0.68]) { g.beginPath(); g.arc(cx, cy, rw0 + (rw1 - rw0) * f, a0 + gap, a1 - gap); g.stroke(); }
        // silk sheen along the rim
        g.strokeStyle = AM.rgba('#fff4d6', 0.35);
        g.lineWidth = 1;
        g.beginPath(); g.arc(cx, cy, rw1 - 1.5, a0 + gap, a1 - gap); g.stroke();
        // woven stitches across the band
        if (span * rm > 16) {
          g.strokeStyle = AM.rgba(AM.col.ink, 0.13);
          g.lineWidth = 1;
          const nSt = Math.floor((span * rm) / 5);
          g.beginPath();
          for (let k = 1; k < nSt; k++) {
            const a = a0 + (span * k) / nSt;
            g.moveTo(cx + Math.cos(a) * (rw0 + 3), cy + Math.sin(a) * (rw0 + 3));
            g.lineTo(cx + Math.cos(a) * (rw0 + 9), cy + Math.sin(a) * (rw0 + 9));
          }
          g.stroke();
        }
        // label
        arcLabel(g, G, S, r, a0, a1, rm);
      }
      // remaining mass of tokens not in play: nothing (they are cut), so the wheel is exactly the kept set
      return c;
    }

    // ---- draw
    function draw() {
      const { g, w, h } = cv;
      if (!w || !st.S) return;
      const t = now();
      const G = geo(w, h);
      const { cx, cy, R, rTxt, rBody, rTrack, rw0, rw1, rc } = G;
      // While a spin is under way the wheel keeps the distribution it was spun on,
      // even if a slider moves; the new shape appears with the next token.
      const spinning = (st.phase === 'spin' || st.phase === 'land' || st.phase === 'fly') && st.spin;
      const wheelS = spinning ? st.spin.S : st.S;
      if (st.dirty || !st.wheelImg || st.wheelFor !== wheelS) { st.wheelImg = buildWheelImage(G, wheelS); st.wheelFor = wheelS; st.dirty = false; }

      cv.clear();
      // ground: a soft halo behind the wheel and faint radial warp
      const halo = g.createRadialGradient(cx, cy, rc * 0.4, cx, cy, R);
      halo.addColorStop(0, AM.rgba(AM.dye.weld, 0.07));
      halo.addColorStop(0.6, AM.rgba(AM.dye.woad, 0.035));
      halo.addColorStop(1, AM.rgba(AM.dye.woad, 0));
      g.fillStyle = halo;
      g.fillRect(0, 0, w, h);

      // ---- outer ring: 32 context slots as radial warp threads
      const n = st.ids.length;
      g.save();
      g.lineWidth = 1;
      for (let i = 0; i < NCTX; i++) {
        const a = slotAng(i);
        const c = Math.cos(a), s = Math.sin(a);
        const filled = i < n;
        g.strokeStyle = AM.rgba(AM.col.linen, filled ? 0.05 : 0.08);
        g.beginPath(); g.moveTo(cx + c * rBody, cy + s * rBody); g.lineTo(cx + c * R, cy + s * R); g.stroke();
      }
      g.restore();
      // the snake body: one thread through every filled slot
      if (n > 0) {
        const a0 = slotAng(0) - (0.35 / NCTX) * TAU, a1 = slotAng(n - 1);
        g.save();
        g.lineCap = 'round';
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.12); g.lineWidth = 7;
        g.beginPath(); g.arc(cx, cy, rBody, a0, a1); g.stroke();
        const grd = g.createLinearGradient(cx - R, cy, cx + R, cy);
        grd.addColorStop(0, AM.rgba(AM.dye.weld, 0.7)); grd.addColorStop(1, AM.rgba(AM.dye.saffron, 0.7));
        g.strokeStyle = grd; g.lineWidth = 1.6;
        g.beginPath(); g.arc(cx, cy, rBody, a0, a1); g.stroke();
        g.restore();
        // particles flowing from tail to head
        if (!AM.reducedMotion) {
          const span = a1 - a0;
          for (let k = 0; k < Math.min(18, n); k++) {
            const f = ((t * 0.12 + k / Math.min(18, n)) % 1);
            const a = a0 + span * f;
            D.glowDot(g, cx + Math.cos(a) * rBody, cy + Math.sin(a) * rBody, 1.1, AM.dye.weld, 0.55);
          }
        }
        // tail knot
        D.glowDot(g, cx + Math.cos(a0) * rBody, cy + Math.sin(a0) * rBody, 1.6, AM.dye.saffron, 0.8);
      }
      // the ring closes when the context is full: the head meets the tail
      if (n >= NCTX && !st.fullAt) st.fullAt = t;
      if (n < NCTX) st.fullAt = 0;
      if (st.fullAt && !AM.reducedMotion && t - st.fullAt < 1.8) {
        const e = easeInOut(clamp((t - st.fullAt) / 1.8));
        const a = -Math.PI / 2 + TAU * e;
        g.save(); g.lineCap = 'round';
        for (let k = 0; k < 18; k++) {
          const a1 = a - k * 0.035, a0 = a1 - 0.035;
          g.strokeStyle = AM.rgba(k < 3 ? '#fff4d6' : AM.dye.weld, 0.8 * (1 - k / 18));
          g.lineWidth = 4 * (1 - k / 18) + 0.5;
          g.beginPath(); g.arc(cx, cy, rBody, a0, a1); g.stroke();
        }
        g.restore();
        D.glowDot(g, cx + Math.cos(a) * rBody, cy + Math.sin(a) * rBody, 3.5, AM.dye.weld, 1);
      }
      if (n >= NCTX) {
        const pulse = AM.reducedMotion ? 1 : 0.6 + 0.4 * Math.sin(t * 2.4);
        g.save(); g.strokeStyle = AM.rgba(AM.dye.weld, 0.35 * pulse); g.lineWidth = 3;
        g.beginPath(); g.arc(cx, cy, rBody, 0, TAU); g.stroke(); g.restore();
      }
      // knots + tokens
      const tsize = G.phone ? 9 : 11.5;
      for (let i = 0; i < n; i++) {
        const a = slotAng(i);
        const isGen = i >= st.promptLen;
        const isNew = i === n - 1 && isGen;
        const kx = cx + Math.cos(a) * rBody, ky = cy + Math.sin(a) * rBody;
        g.fillStyle = isGen ? AM.dye.weld : AM.col.linenDim;
        g.beginPath(); g.arc(kx, ky, isNew ? 2.6 : 1.8, 0, TAU); g.fill();
        let word = m.vocab[st.ids[i]];
        let fs = tsize;
        const room = R - rTxt - 2;
        const ww = textW(g, word, fs, 'body', 600);
        if (ww > room) fs = Math.max(6.5, (fs * room) / ww);
        if (word === '.' || word === ',') fs = tsize * 1.8; // punctuation would otherwise be a speck
        const fresh = isNew && st.phase === 'weave' ? clamp((t - st.weaveFrom) / 0.4) : 1;
        radialText(g, word, cx, cy, a, rTxt, { size: fs, color: isGen ? (isNew ? '#fff1c2' : AM.dye.weld) : AM.col.linenDim, alpha: isGen ? 1 : 0.85, weight: isGen ? 700 : 500 });
        if (isNew && fresh < 1) D.glowDot(g, kx, ky, 4 * (1 - fresh) + 1, AM.dye.weld, 1 - fresh);
      }
      // the head: the next empty slot breathes
      if (n < NCTX) {
        const a = slotAng(n);
        const pulse = AM.reducedMotion ? 0.7 : 0.5 + 0.5 * Math.sin(t * 3);
        const c = Math.cos(a), s = Math.sin(a);
        g.save();
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.25 + 0.35 * pulse); g.lineWidth = 1.2; g.setLineDash([2, 3]);
        g.beginPath(); g.moveTo(cx + c * (rTxt - 1), cy + s * (rTxt - 1)); g.lineTo(cx + c * (R - 4), cy + s * (R - 4)); g.stroke();
        g.restore();
        D.glowDot(g, cx + c * rBody, cy + s * rBody, 1.8 + pulse, AM.dye.weld, 0.5 + 0.4 * pulse);
      }

      // ---- the forward pass: every slot sends a thread to the centre
      if (st.phase === 'weave') {
        const e = clamp((t - st.weaveFrom) / Math.max(0.01, dur('weave')));
        for (let i = 0; i < n; i++) {
          const a = slotAng(i);
          const sx = cx + Math.cos(a) * rBody, sy = cy + Math.sin(a) * rBody;
          const a2 = a + 0.9;
          const P = [sx, sy, cx + Math.cos(a2) * rw1 * 0.95, cy + Math.sin(a2) * rw1 * 0.95, cx + Math.cos(a2 + 0.8) * rc * 0.45, cy + Math.sin(a2 + 0.8) * rc * 0.45, cx, cy];
          const isGen = i >= st.promptLen;
          const al = Math.sin(Math.PI * e) * (i === n - 1 ? 0.9 : 0.4);
          silk(g, P, { color: isGen ? AM.dye.weld : AM.dye.woad, width: i === n - 1 ? 1.4 : 0.8, alpha: al, sheen: false });
          const f = clamp(e * 1.4 - (i / Math.max(1, n)) * 0.4);
          if (f > 0 && f < 1) { const pt = cubic(P, easeInOut(f)); D.glowDot(g, pt.x, pt.y, 1.5, isGen ? AM.dye.weld : AM.dye.woad, 0.9); }
        }
      }

      // ---- bead track
      g.save();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.1); g.lineWidth = 1; g.setLineDash([1, 5]);
      g.beginPath(); g.arc(cx, cy, rTrack, 0, TAU); g.stroke();
      g.restore();

      // ---- wheel (old one fades as the new one is woven in, clockwise from 12)
      const revealA = uAng(st.reveal);
      if (st.oldImg && st.reveal < 1) {
        g.save();
        g.globalAlpha = 0.35 * (1 - st.reveal);
        g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, rw1 + 4, revealA, uAng(1)); g.closePath(); g.clip();
        g.drawImage(st.oldImg, 0, 0, w, h);
        g.restore();
      }
      g.save();
      if (st.reveal < 1) { g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, rw1 + 4, uAng(0), revealA); g.closePath(); g.clip(); }
      g.drawImage(st.wheelImg, 0, 0, w, h);
      g.restore();
      if (st.reveal < 1) {
        // the weaving edge
        const c = Math.cos(revealA), s = Math.sin(revealA);
        g.save(); g.strokeStyle = AM.rgba('#fff4d6', 0.8); g.lineWidth = 1.5; g.shadowColor = AM.dye.weld; g.shadowBlur = 12;
        g.beginPath(); g.moveTo(cx + c * (rw0 - 2), cy + s * (rw0 - 2)); g.lineTo(cx + c * (rw1 + 3), cy + s * (rw1 + 3)); g.stroke(); g.restore();
      }

      // ---- which arc is under the bead / pointer / chosen
      const S = wheelS;
      const beadU = ((st.bead % 1) + 1) % 1;
      let focusR = -1, focusKind = '';
      if (st.phase === 'spin') { focusR = pickRank(S.cum, S.keep, beadU); focusKind = 'under'; }
      else if (st.phase === 'land' || st.phase === 'fly') { focusR = st.spin.r; focusKind = 'chosen'; }
      else if (st.hover >= 0 && st.hover < S.keep && st.phase === 'idle') { focusR = st.hover; focusKind = 'hover'; }
      if (focusR >= 0 && st.reveal >= 1) {
        const a0 = uAng(S.cum[focusR]), a1 = uAng(S.cum[focusR + 1]);
        const col = rankColor(focusR);
        const pop = focusKind === 'chosen' ? (AM.reducedMotion ? 1 : M.ease.outBack(clamp((t - st.t0) / 0.35 + (st.phase === 'fly' ? 1 : 0)))) : 0.4;
        const off = (G.phone ? 4 : 6) * pop;
        g.save();
        g.shadowColor = AM.rgba(col, 0.9); g.shadowBlur = focusKind === 'chosen' ? 22 : 10;
        g.fillStyle = AM.rgba(col, focusKind === 'under' ? 0.55 : 0.9);
        g.beginPath();
        g.arc(cx, cy, rw1 + off, a0, Math.max(a0 + 0.004, a1));
        g.arc(cx, cy, rw0 + off * 0.3, Math.max(a0 + 0.004, a1), a0, true);
        g.closePath(); g.fill();
        g.shadowBlur = 0;
        g.strokeStyle = AM.rgba('#fff4d6', 0.85); g.lineWidth = 1.2; g.stroke();
        g.restore();
        // the overlay would hide the arc's own label, so write it again on top
        arcLabel(g, G, S, focusR, a0, a1, (rw0 + rw1) / 2 + off * 0.65);
      }

      // ---- bead (with a fading trail while it spins)
      const ba = uAng(beadU);
      if (st.phase === 'spin' && !AM.reducedMotion) {
        // a comet tail along the track
        const sp = st.spin;
        const D0 = dur('spin');
        const at = (k) => { const tt = clamp((t - st.t0 - k * 0.01) / D0); return uAng(0) + TAU * (sp.b0 + (sp.b1 - sp.b0) * easeOutQuart(tt)); };
        g.save();
        g.lineCap = 'round';
        let prev = at(0);
        for (let k = 1; k <= 22; k++) {
          const a = at(k);
          if (prev - a > 1e-4) {
            const f = 1 - k / 23;
            g.strokeStyle = AM.rgba(k < 4 ? '#fff4d6' : AM.dye.weld, 0.55 * f);
            g.lineWidth = 0.6 + 3 * f;
            g.beginPath(); g.arc(cx, cy, rTrack, a, prev); g.stroke();
          }
          prev = a;
        }
        g.restore();
      }
      D.glowDot(g, cx + Math.cos(ba) * rTrack, cy + Math.sin(ba) * rTrack, G.phone ? 3 : 3.8, AM.col.linen, 1);
      // a tick from the bead into the wheel
      g.save(); g.strokeStyle = AM.rgba('#fff4d6', 0.7); g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(cx + Math.cos(ba) * (rTrack - 3), cy + Math.sin(ba) * (rTrack - 3)); g.lineTo(cx + Math.cos(ba) * (rw1 + 1), cy + Math.sin(ba) * (rw1 + 1)); g.stroke(); g.restore();

      // ---- centre
      const cg = g.createRadialGradient(cx, cy - rc * 0.3, rc * 0.1, cx, cy, rc);
      cg.addColorStop(0, AM.col.ink3); cg.addColorStop(1, AM.col.ink);
      g.fillStyle = cg;
      g.beginPath(); g.arc(cx, cy, rc, 0, TAU); g.fill();
      g.strokeStyle = AM.rgba(AM.dye.weld, 0.3); g.lineWidth = 1;
      g.beginPath(); g.arc(cx, cy, rc, 0, TAU); g.stroke();
      const big = G.phone ? 21 : 32, small = G.phone ? 8.5 : 10;
      let kicker = '', main = '', sub = '', mainCol = AM.col.linen;
      if (full()) { kicker = 'context full'; main = `${NCTX} / ${NCTX}`; sub = 'reset to write again'; mainCol = AM.dye.weld; }
      else if (st.phase === 'spin') { kicker = 'the bead is over'; main = shown(S.order[focusR]); sub = pct(S.q[focusR]); }
      else if (st.phase === 'land' || st.phase === 'fly') { kicker = st.spin.S.keep === 1 ? 'the only choice' : `u = ${st.spin.u.toFixed(3)}`; main = shown(st.spin.id); sub = `p = ${pct(st.spin.p)}`; mainCol = AM.dye.weld; }
      else if (focusKind === 'hover') { kicker = `rank ${focusR + 1}`; main = shown(S.order[focusR]); sub = `p = ${pct(S.q[focusR])}`; }
      else if (st.phase === 'weave') { kicker = 'reading'; main = `${n} tokens`; sub = 'forward pass'; mainCol = AM.col.linenDim; }
      else { kicker = [`next token · ${n + 1} of ${NCTX}`, 'next token']; main = st.T <= 0 ? 'greedy' : 'spin me'; sub = S.keep === 1 ? `only ${m.vocab[S.order[0]]}` : `≈ ${fmt(S.eff, 1)} choices`; mainCol = AM.col.linenDim; }
      /** First candidate that fits maxW at no less than minSize (shrinking it if needed); else the last one at minSize. */
      const fit = (cands, size, role, maxW, minSize, weight, italic) => {
        cands = [].concat(cands);
        g.save(); g.font = AM.font(size, role, weight, italic);
        let pick = null;
        for (const str of cands) {
          const mw = g.measureText(str).width;
          const sz = mw > maxW ? (size * maxW) / mw : size;
          if (sz >= minSize) { pick = { str, size: sz }; break; }
        }
        g.restore();
        return pick || { str: cands[cands.length - 1], size: minSize };
      };
      const fk = fit(kicker, small, 'mono', rc * 1.55, 9);
      const fm = fit(main, big, 'display', rc * 1.7, 12, 500, true);
      const fsub = fit(sub, small + 1, 'mono', rc * 1.55, 9);
      haloText(g, fk.str, cx, cy - big * 0.95, { size: fk.size, color: AM.col.mist, halo: false });
      haloText(g, fm.str, cx, cy + 1, { size: fm.size, role: 'display', weight: 500, italic: true, color: mainCol, halo: false });
      haloText(g, fsub.str, cx, cy + big * 0.95, { size: fsub.size, color: AM.col.linenDim, halo: false });

      // ---- the drawn word flies from its arc to the next slot
      if (st.phase === 'fly' && st.spin) {
        const sp = st.spin;
        const e = easeInOut(clamp((t - st.t0) / Math.max(0.01, dur('fly'))));
        const am = uAng((sp.S.cum[sp.r] + sp.S.cum[sp.r + 1]) / 2);
        const as = slotAng(n);
        const r0 = (rw0 + rw1) / 2, r1 = rTxt + 16;
        const ro = Math.min(R + 16, Math.min(w, h) / 2 + 6);
        const P = [cx + Math.cos(am) * r0, cy + Math.sin(am) * r0, cx + Math.cos(am) * ro, cy + Math.sin(am) * ro, cx + Math.cos(as) * ro, cy + Math.sin(as) * ro, cx + Math.cos(as) * r1, cy + Math.sin(as) * r1];
        silk(g, P, { color: AM.dye.weld, width: 1.2, alpha: 0.4 * (1 - e) + 0.1 });
        const pt = cubic(P, e);
        haloText(g, m.vocab[sp.id], pt.x, pt.y, { size: (G.phone ? 13 : 17) * (1 - 0.3 * e), role: 'body', weight: 700, color: '#fff1c2' });
      }
    }

    // ---- pointer: hover reads an arc, click spins
    function hitRank(ev) {
      const pnt = cv.pointer(ev);
      const G = geo(cv.w, cv.h);
      const dx = pnt.x - G.cx, dy = pnt.y - G.cy;
      const r = Math.hypot(dx, dy);
      if (r < G.rw0 - 2 || r > G.rTrack + 6 || !st.S) return -1;
      let a = Math.atan2(dy, dx) + Math.PI / 2;
      if (a < 0) a += TAU;
      return pickRank(st.S.cum, st.S.keep, a / TAU);
    }
    cv.canvas.addEventListener('pointermove', (ev) => { if (ev.pointerType === 'mouse') st.hover = hitRank(ev); });
    cv.canvas.addEventListener('pointerleave', () => { st.hover = -1; });
    cv.canvas.addEventListener('click', (ev) => {
      const G = geo(cv.w, cv.h);
      const pnt = cv.pointer(ev);
      if (Math.hypot(pnt.x - G.cx, pnt.y - G.cy) <= G.rTrack + 8) { stopAuto(); spin(); }
    });

    reset();
    cv.onResize(() => { st.dirty = true; st.oldImg = null; draw(); });
    const seen = inView(cv.canvas);
    // The writing keeps going while the chapter is on screen, even when the wheel
    // itself is scrolled away (on a phone the transcript sits far below it);
    // drawing only happens while the wheel can be seen.
    ctx.loop(() => { tick(); if (seen.on) draw(); });
    ctx.onHidden(() => stopAuto());
    draw();
  }
})();
