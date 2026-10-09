/* The Attention Loom · Chapter 02 · Embeddings · "Constellations of Meaning".
 *
 * Teaches: the embedding matrix is a lookup table (one row per token ID,
 * d_model columns); a row is a point in a high-dimensional space; training
 * pulls tokens that are used alike together; closeness is measured with the
 * dot product / cosine similarity; some directions carry meaning (analogies,
 * approximately); single coordinates are not labels; we only ever see
 * projections (PCA "shadows").
 *
 * Data: a TOY embedding space built in buildSpace() from hand-designed
 * semantic features plus seeded noise, spun into 64 dimensions by a fixed
 * random rotation. PCA is computed in the browser (Jacobi eigendecomposition of
 * the 64 × 64 covariance). Every number on screen is computed from that space.
 */
(() => {
  const AM = window.AM;
  const ID = 'embed';
  const D = 64; // d_model of the toy space

  // ======================================================================
  // 1. The toy embedding space
  // ======================================================================

  /* Named features. A word's recipe is a weighted mix of these. */
  const FEATS = ['person', 'animal', 'colour', 'number', 'food', 'place', 'verb', 'nature',
    'topic-x', 'topic-y', 'common',
    'royal', 'female', 'young', 'plural', 'past', 'parent',
    'wild', 'big', 'feline', 'canine',
    'hue-x', 'hue-y', 'bright', 'magnitude', 'sweet', 'fruit', 'dairy',
    'city', 'europe', 'motion', 'sky', 'water', 'cold'];
  /* Plus N_ID "identity code" dims shared by word families (walk/walked,
     paris/france, cat/kitten/cats…) so that family members stay distinct. */
  const N_ID = 10;
  const K = FEATS.length + N_ID; // 44 recipe dims; the other 20 of 64 hold only noise

  /* Rough "how common is this word" score in [-1, 1]. Toy values set by intuition. */
  const FREQ = {
    man: .9, woman: .6, king: .2, queen: -.1, boy: .4, girl: .5, father: .3, mother: .4, son: .1, daughter: -.2, prince: -.6, princess: -.8,
    dog: .5, cat: .3, dogs: 0, cats: -.3, horse: .1, fish: .4, bird: .2, cow: -.4, lion: -.2, mouse: -.1, wolf: -.5, tiger: -.6, fox: -.7, kitten: -.9, puppy: -.8, foal: -1,
    red: .5, white: .7, black: .6, blue: .4, green: .3, yellow: -.1, brown: 0, grey: -.4, pink: -.3, orange: -.2, purple: -.6,
    one: 1, two: .8, three: .5, four: .2, five: .1, six: -.2, seven: -.3, eight: -.5, nine: -.6, ten: -.1,
    bread: .1, cheese: -.2, milk: .2, egg: 0, apple: .3, banana: -.6, cake: -.1, honey: -.5, sugar: -.3, soup: -.4, rice: -.2, pizza: -.7,
    paris: .2, france: .3, rome: -.1, italy: 0, berlin: -.4, germany: .1, madrid: -.6, spain: -.2, tokyo: -.3, japan: .2, cairo: -.8, egypt: -.5,
    go: 1, went: .7, see: .9, saw: .6, eat: .3, ate: -.1, run: .5, ran: .2, walk: .2, walked: -.1, swim: -.4, swam: -.8, sing: -.3, sang: -.6,
    sun: .4, moon: 0, star: .1, sky: .2, cloud: -.4, rain: .1, snow: -.2, wind: -.1, sea: .3, river: -.3, mountain: -.5, tree: .5,
  };

  /* Word recipes. Spec mini-language: 'feat' (+1), 'feat.6' / 'feat-1' (weight),
     'hue<deg>[:chroma]' (colour wheel), 'n<k>' (number magnitude), '#family'.
     ang = where the cluster sits on the topic ring; idw = weight of identity code. */
  const CLUSTERS = [
    { key: 'person', label: 'people', dye: 'weld', ang: 0, idw: 0.8, words: {
      king: 'royal female-1 #monarch', queen: 'royal female #monarch', prince: 'royal female-1 young.6 #heir', princess: 'royal female young.6 #heir',
      man: 'female-1 #adult', woman: 'female #adult', boy: 'female-1 young #child', girl: 'female young #child',
      father: 'female-1 parent #parent', mother: 'female parent #parent', son: 'female-1 young.5 #offspring', daughter: 'female young.5 #offspring' } },
    { key: 'animal', label: 'animals', dye: 'madder', ang: 45, idw: 1.0, words: {
      cat: 'feline big-.5 #cat', cats: 'feline big-.5 plural #cat', kitten: 'feline big-.5 young #cat',
      dog: 'canine big-.2 #dog', dogs: 'canine big-.2 plural #dog', puppy: 'canine big-.2 young #dog',
      horse: 'big.8 #horse', foal: 'big.8 young #horse', lion: 'feline wild big #lion', tiger: 'feline wild big #tiger',
      wolf: 'canine wild #wolf', fox: 'canine.8 wild big-.3 #fox', mouse: 'wild.5 big-1 #mouse', cow: 'big.8 #cow',
      bird: 'wild.6 big-.6 sky.3 #bird', fish: 'wild.5 big-.5 water.4 #fish' } },
    { key: 'colour', label: 'colours', dye: 'cochineal', ang: 225, idw: 0.5, words: {
      red: 'hue0 #red', orange: 'hue30 #orange', yellow: 'hue60 bright.4 #yellow', green: 'hue120 #green', blue: 'hue225 #blue',
      purple: 'hue285 #purple', pink: 'hue335 bright.5 #pink', brown: 'hue25:.5 bright-.5 #brown', black: 'bright-1.2 #black',
      white: 'bright1.2 #white', grey: '#grey' } },
    { key: 'number', label: 'numbers', dye: 'woad', ang: 180, idw: 0.12, words: {
      one: 'n1 #one', two: 'n2 #two', three: 'n3 #three', four: 'n4 #four', five: 'n5 #five', six: 'n6 #six', seven: 'n7 #seven',
      eight: 'n8 #eight', nine: 'n9 #nine', ten: 'n10 #ten' } },
    { key: 'food', label: 'food', dye: 'saffron', ang: 270, idw: 0.9, words: {
      bread: '#bread', cheese: 'dairy #cheese', milk: 'dairy #milk', egg: '#egg', apple: 'fruit sweet.5 #apple',
      banana: 'fruit sweet.6 #banana', cake: 'sweet #cake', honey: 'sweet #honey', sugar: 'sweet #sugar',
      soup: '#soup', rice: '#rice', pizza: 'dairy.4 #pizza' } },
    { key: 'place', label: 'places', dye: 'verdigris', ang: 135, idw: 1.0, words: {
      paris: 'city europe #FR', france: 'city-1 europe #FR', rome: 'city europe #IT', italy: 'city-1 europe #IT',
      berlin: 'city europe #DE', germany: 'city-1 europe #DE', madrid: 'city europe #ES', spain: 'city-1 europe #ES',
      tokyo: 'city #JP', japan: 'city-1 #JP', cairo: 'city #EG', egypt: 'city-1 #EG' } },
    { key: 'verb', label: 'verbs', dye: 'lichen', ang: 315, idw: 1.0, words: {
      walk: 'motion #walk', walked: 'motion past #walk', swim: 'motion water.3 #swim', swam: 'motion water.3 past #swim',
      run: 'motion #run', ran: 'motion past #run', go: 'motion.6 #go', went: 'motion.6 past #go',
      eat: 'food.4 #eat', ate: 'food.4 past #eat', see: '#see', saw: 'past #see', sing: '#sing', sang: 'past #sing' } },
    { key: 'nature', label: 'nature', dye: 'linen', ang: 90, idw: 0.8, words: {
      sun: 'sky bright.8 #sun', moon: 'sky bright.5 #moon', star: 'sky bright.6 #star', sky: 'sky #sky', cloud: 'sky.7 water.5 #cloud',
      rain: 'water sky.3 #rain', snow: 'water.6 cold #snow', wind: 'sky.4 cold.4 #wind', sea: 'water #sea', river: 'water #river',
      mountain: 'cold.4 #mountain', tree: '#tree' } },
  ];

  const W_CLUSTER = 1.2, W_RING = 1.6, W_FREQ = 1.5, NOISE = 0.08, OUT_SCALE = 0.25;

  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = (a) => Math.sqrt(dot(a, a));

  /** Eigendecomposition of a symmetric matrix (cyclic Jacobi). Returns {vals, vecs} sorted by value, descending. */
  function symEig(A) {
    const n = A.length;
    const a = A.map((r) => Float64Array.from(r));
    const V = Array.from({ length: n }, (_, i) => { const r = new Float64Array(n); r[i] = 1; return r; });
    for (let sweep = 0; sweep < 50; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-16) continue;
        // Rotation angle that zeroes a[p][q]:  A ← Jᵀ A J
        const th = (a[q][q] - a[p][p]) / (2 * apq);
        const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const kp = a[k][p], kq = a[k][q]; a[k][p] = c * kp - s * kq; a[k][q] = s * kp + c * kq; }
        for (let k = 0; k < n; k++) { const pk = a[p][k], qk = a[q][k]; a[p][k] = c * pk - s * qk; a[q][k] = s * pk + c * qk; }
        for (let k = 0; k < n; k++) { const kp = V[k][p], kq = V[k][q]; V[k][p] = c * kp - s * kq; V[k][q] = s * kp + c * kq; }
      }
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
    return { vals: order.map((i) => a[i][i]), vecs: order.map((i) => Float64Array.from(V, (r) => r[i])) };
  }

  /** Build the toy space. Deterministic (seeded). */
  function buildSpace() {
    const rnd = AM.math.rng(7);
    const randn = () => AM.math.randn(rnd);
    const fi = (name) => FEATS.indexOf(name);
    const famVec = new Map();
    const family = (name) => {
      if (!famVec.has(name)) {
        const v = Array.from({ length: N_ID }, randn);
        const n = Math.hypot(...v);
        famVec.set(name, v.map((x) => x / n));
      }
      return famVec.get(name);
    };

    // --- recipes → feature vectors f (length K)
    const words = [];
    CLUSTERS.forEach((cl, c) => {
      for (const [w, spec] of Object.entries(cl.words)) {
        const f = new Float64Array(K);
        f[fi(cl.key)] = W_CLUSTER;                         // which topic
        const th = (cl.ang * Math.PI) / 180;               // where the topic sits on a ring of topics
        f[fi('topic-x')] = W_RING * Math.cos(th);
        f[fi('topic-y')] = W_RING * Math.sin(th);
        f[fi('common')] = W_FREQ * (FREQ[w] || 0);         // how common the word is
        for (const tok of spec.split(/\s+/).filter(Boolean)) {
          let m;
          if (tok[0] === '#') { const v = family(tok.slice(1)); for (let i = 0; i < N_ID; i++) f[FEATS.length + i] += cl.idw * v[i]; continue; }
          if ((m = tok.match(/^hue(\d+)(?::([\d.]+))?$/))) {
            const h = (+m[1] * Math.PI) / 180, chroma = m[2] ? +m[2] : 1;
            f[fi('hue-x')] = chroma * Math.cos(h); f[fi('hue-y')] = chroma * Math.sin(h); continue;
          }
          if ((m = tok.match(/^n(\d+)$/))) { f[fi('magnitude')] = (+m[1] - 5.5) / 2; continue; } // a number line
          m = tok.match(/^([a-z-]+?)(-?\d*\.?\d+)?$/);
          f[fi(m[1])] += m[2] != null ? +m[2] : 1;
        }
        words.push({ w, c, f, freq: FREQ[w] || 0 });
      }
    });
    const N = words.length;

    // --- a fixed random rotation: Gram–Schmidt on Gaussian rows → orthonormal basis Q (64 × 64)
    const Q = [];
    for (let i = 0; i < D; i++) {
      const v = Float64Array.from({ length: D }, randn);
      for (const q of Q) { const d = dot(v, q); for (let k = 0; k < D; k++) v[k] -= d * q[k]; }
      const n = norm(v); for (let k = 0; k < D; k++) v[k] /= n;
      Q.push(v);
    }
    // --- embedding row = OUT_SCALE · (Σ_k f_k · Q_k + noise). Each feature lands on a random
    //     direction, so no single coordinate of E means anything by itself.
    for (const wd of words) {
      const e = new Float64Array(D);
      for (let k = 0; k < K; k++) { const fk = wd.f[k]; if (fk) for (let j = 0; j < D; j++) e[j] += fk * Q[k][j]; }
      for (let j = 0; j < D; j++) e[j] = (e[j] + NOISE * randn()) * OUT_SCALE;
      wd.e = e;
    }
    // Recipe coordinates: the same vector expressed in the designer's basis (rows of Q).
    for (const wd of words) wd.rc = Float64Array.from(Q, (q) => dot(wd.e, q));

    // --- token IDs: a fixed shuffle, so row order in E carries no meaning (as in real vocabularies)
    const perm = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    words.forEach((wd, i) => { wd.id = perm[i]; });
    const rowWord = []; words.forEach((wd, i) => { rowWord[wd.id] = i; });

    // --- cosine similarity, computed in all 64 dimensions
    const E = words.map((wd) => wd.e);
    const norms = E.map(norm);
    const cosM = E.map((a, i) => E.map((b, j) => dot(a, b) / (norms[i] * norms[j])));
    const nbrs = cosM.map((row, i) => Array.from({ length: N }, (_, j) => j).filter((j) => j !== i).sort((x, y) => row[y] - row[x]));

    // --- PCA: eigenvectors of the covariance of the centred rows
    const mean = new Float64Array(D);
    for (const e of E) for (let j = 0; j < D; j++) mean[j] += e[j] / N;
    const C = Array.from({ length: D }, () => new Float64Array(D));
    for (const e of E) for (let i = 0; i < D; i++) { const xi = e[i] - mean[i]; for (let j = 0; j < D; j++) C[i][j] += (xi * (e[j] - mean[j])) / (N - 1); }
    const { vals, vecs } = symEig(C);
    const total = vals.reduce((s, v) => s + Math.max(0, v), 0);
    const pcScore = (v, k) => { let s = 0; for (let j = 0; j < D; j++) s += (v[j] - mean[j]) * vecs[k][j]; return s; };
    // Which of the top 3 PCs tracks word frequency best? That one becomes "up" on screen.
    const freqs = words.map((wd) => wd.freq);
    const corr = (xs, ys) => {
      const mx = xs.reduce((s, x) => s + x, 0) / xs.length, my = ys.reduce((s, y) => s + y, 0) / ys.length;
      let sxy = 0, sxx = 0, syy = 0;
      for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
      return sxy / Math.sqrt(sxx * syy);
    };
    const corrs = [0, 1, 2].map((k) => corr(E.map((e) => pcScore(e, k)), freqs));
    const iy = [0, 1, 2].sort((a, b) => Math.abs(corrs[b]) - Math.abs(corrs[a]))[0];
    const [ix, iz] = [0, 1, 2].filter((k) => k !== iy);
    const ySign = corrs[iy] >= 0 ? 1 : -1;
    const raw3 = (v) => [pcScore(v, ix), ySign * pcScore(v, iy), pcScore(v, iz)];
    // Uniform scale (all three axes alike, so proportions stay true): 96% of stars inside the unit ball.
    const radii = E.map((e) => Math.hypot(...raw3(e))).sort((x, y) => x - y);
    const rMax = radii[Math.floor(radii.length * 0.96)];
    /** Project any 64-d vector into the 3D shadow (screen x ← PC ix, up ← PC iy, depth ← PC iz). */
    const proj = (v) => raw3(v).map((x) => x / rMax);
    const P3 = E.map(proj);

    /** Unit vector of v after removing its components along `basis`; falls back if it vanishes. */
    const orthoUnit = (v, basis, fallbacks) => {
      for (const cand of [v, ...fallbacks]) {
        const u = Float64Array.from(cand);
        for (const b of basis) { const d = dot(u, b); for (let j = 0; j < D; j++) u[j] -= d * b[j]; }
        const n = norm(u);
        if (n > 1e-9) { for (let j = 0; j < D; j++) u[j] /= n; return u; }
      }
      return new Float64Array(D);
    };
    /** A second kind of shadow, chosen for one analogy: up = direction A − B, across = C − B
        (made orthogonal), depth = the top PCA direction. In this plane the parallelogram
        A, B, C, A − B + C lies flat with its true shape. */
    const plane = (a, b, c) => {
      const A = E[a], B = E[b], Cv = E[c];
      const T = Float64Array.from(A, (x, j) => x - B[j] + Cv[j]);
      const u1 = orthoUnit(Float64Array.from(A, (x, j) => x - B[j]), [], [vecs[0], vecs[1]]);
      const u2 = orthoUnit(Float64Array.from(Cv, (x, j) => x - B[j]), [u1], [vecs[1], vecs[2]]);
      const u3 = orthoUnit(vecs[0], [u1, u2], [vecs[1], vecs[2], vecs[3]]);
      const m = Float64Array.from(A, (x, j) => (x + Cv[j]) / 2); // centre of the parallelogram
      const raw = (v) => { let x = 0, y = 0, z = 0; for (let j = 0; j < D; j++) { const d = v[j] - m[j]; x += d * u2[j]; y += d * u1[j]; z += d * u3[j]; } return [x, y, z]; };
      const corner = Math.max(...[A, B, Cv, T].map((v) => Math.hypot(...raw(v).slice(0, 2))), 1e-6);
      const sc = 0.58 / corner;
      const P = E.map((e) => raw(e).map((x) => x * sc));
      return { P, T: raw(T).map((x) => x * sc) };
    };

    const index = new Map(words.map((wd, i) => [wd.w, i]));

    /** A − B + C in 64-d, then nearest words by cosine (optionally leaving out A, B, C). */
    const analogy = (a, b, c, exclude = true) => {
      const t = new Float64Array(D);
      for (let j = 0; j < D; j++) t[j] = E[a][j] - E[b][j] + E[c][j];
      const tn = norm(t);
      const scored = E.map((e, i) => ({ i, cos: dot(t, e) / (tn * norms[i]) }))
        .filter((s) => !exclude || (s.i !== a && s.i !== b && s.i !== c))
        .sort((x, y) => y.cos - x.cos);
      return { t, p: proj(t), top: scored.slice(0, 5) };
    };

    return {
      words, N, E, norms, cosM, nbrs, rowWord, Q, index, P3, proj, analogy, plane,
      pca: { vals, total, axes: [ix, iy, iz], fracs: [ix, iy, iz].map((k) => vals[k] / total), top3: (vals[0] + vals[1] + vals[2]) / total, corrY: Math.abs(corrs[iy]) },
      vmax: (() => { const all = E.flatMap((e) => Array.from(e, Math.abs)).sort((x, y) => x - y); return all[Math.floor(all.length * 0.98)]; })(),
    };
  }

  // ======================================================================
  // 2. Shared drawing kit for the star fields
  // ======================================================================

  const clamp = (x, a = 0, b = 1) => (x < a ? a : x > b ? b : x);
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => AM.math.ease.inOut(clamp(t));
  const easeOut = (t) => AM.math.ease.out(clamp(t));
  const smooth = (a, b, x) => AM.math.smoothstep(a, b, x);
  const approach = (x, target, step) => (x < target ? Math.min(target, x + step) : Math.max(target, x - step));
  const damp = (x, target, rate, dt) => x + (target - x) * (1 - Math.exp(-rate * dt));
  const fmt = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);

  function makeKit(S) {
    const clusterColor = (c) => (CLUSTERS[c].dye === 'linen' ? AM.col.linen : AM.dye[CLUSTERS[c].dye]);
    const color = (i) => clusterColor(S.words[i].c);

    // Pre-rendered glow sprites (one per colour) keep per-frame cost tiny.
    const cache = new Map();
    const sprite = (col, kind = 'star') => {
      const key = col + kind;
      if (cache.has(key)) return cache.get(key);
      const size = kind === 'star' ? 64 : 128;
      const c = document.createElement('canvas');
      c.width = c.height = size;
      const g = c.getContext('2d'), r = size / 2;
      const grd = g.createRadialGradient(r, r, 0, r, r, r);
      if (kind === 'star') {
        grd.addColorStop(0, 'rgba(255,248,230,1)');
        grd.addColorStop(0.1, AM.rgba(col, 0.95));
        grd.addColorStop(0.28, AM.rgba(col, 0.32));
        grd.addColorStop(1, AM.rgba(col, 0));
      } else {
        grd.addColorStop(0, AM.rgba(col, 0.6));
        grd.addColorStop(0.45, AM.rgba(col, 0.2));
        grd.addColorStop(1, AM.rgba(col, 0));
      }
      g.fillStyle = grd;
      g.fillRect(0, 0, size, size);
      cache.set(key, c);
      return c;
    };

    // Background star dust on a distant shell (parallax when the sky turns).
    const rnd = AM.math.rng(2024);
    const dust = Array.from({ length: 170 }, () => {
      const x = AM.math.randn(rnd), y = AM.math.randn(rnd) * 0.7, z = AM.math.randn(rnd);
      const n = Math.hypot(x, y, z) || 1, r = 1.7 + rnd() * 1.3;
      return { x: (x / n) * r, y: (y / n) * r, z: (z / n) * r, a: 0.12 + rnd() * 0.4, ph: rnd() * 6.28 };
    });
    const phase = S.words.map(() => rnd() * 6.28);

    // Constellation lines: each star to its nearest neighbour (cosine, 64-d) inside its group.
    const edges = [];
    const seen = new Set();
    S.words.forEach((wd, i) => {
      for (const j of S.nbrs[i].slice(0, 2)) {
        if (S.words[j].c !== wd.c) continue;
        if (j !== S.nbrs[i][0] && S.cosM[i][j] < 0.8) continue;
        const key = Math.min(i, j) + ':' + Math.max(i, j);
        if (!seen.has(key)) { seen.add(key); edges.push([i, j]); }
      }
    });

    /** Camera → projector. Points live in a unit ball; y is up. */
    const camFrame = (cam) => {
      const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
      const zm = cam.zoom || 1, bx = cam.bx || 0, by = cam.by || 0, ox = cam.ox || 0, oy = cam.oy || 0;
      return (x, y, z) => {
        const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;          // turn about the vertical axis
        const y1 = y * cp + z1 * sp, z2 = -y * sp + z1 * cp;        // tilt: look down on the disc
        const s = cam.dist / (cam.dist + z2);                        // perspective
        const px = cam.cx + x1 * cam.R * s, py = cam.cy - y1 * cam.R * s;
        return { x: bx + (px - bx) * zm + ox, y: by + (py - by) * zm + oy, s: s * Math.sqrt(zm), z: z2 }; // + optional zoom/pan
      };
    };
    const depthA = (z) => clamp(0.97 - (z + 1) * 0.3, 0.3, 1);

    function drawDust(g, P, alpha, t) {
      if (alpha <= 0.01) return;
      g.save();
      g.fillStyle = AM.col.linen;
      for (const d of dust) {
        const p = P(d.x, d.y, d.z);
        if (p.s <= 0) continue;
        g.globalAlpha = alpha * d.a * depthA(p.z * 0.5) * (0.75 + 0.25 * Math.sin(t * 1.3 + d.ph));
        const r = 0.5 + p.s * 0.55;
        g.fillRect(p.x - r / 2, p.y - r / 2, r, r);
      }
      g.restore();
    }

    /** The analogy plane drawn as a faint woven grid (z = 0). */
    function drawGrid(g, P, alpha) {
      if (alpha <= 0.01) return;
      g.save();
      g.strokeStyle = AM.col.linen;
      g.lineWidth = 1;
      const n = 10, e = 0.92;
      for (let k = 0; k <= n; k++) {
        const v = -e + (2 * e * k) / n;
        const a1 = P(v, -e, 0), a2 = P(v, e, 0), b1 = P(-e, v, 0), b2 = P(e, v, 0);
        g.globalAlpha = alpha * (k === n / 2 ? 0.14 : 0.055);
        g.beginPath(); g.moveTo(a1.x, a1.y); g.lineTo(a2.x, a2.y); g.moveTo(b1.x, b1.y); g.lineTo(b2.x, b2.y); g.stroke();
      }
      g.restore();
    }

    /** An equatorial "orbit" ring with ticks, like an armillary sphere. */
    function drawRing(g, P, alpha, radius = 1.12) {
      if (alpha <= 0.01) return;
      g.save();
      g.strokeStyle = AM.col.linen;
      g.lineWidth = 1;
      const n = 120;
      let prev = P(radius, 0, 0);
      for (let k = 1; k <= n; k++) {
        const th = (k / n) * Math.PI * 2;
        const p = P(radius * Math.cos(th), 0, radius * Math.sin(th));
        g.globalAlpha = alpha * 0.22 * depthA((p.z + prev.z) / 2);
        g.beginPath(); g.moveTo(prev.x, prev.y); g.lineTo(p.x, p.y); g.stroke();
        if (k % 5 === 0) {
          const q = P(radius * 1.035 * Math.cos(th), 0, radius * 1.035 * Math.sin(th));
          g.globalAlpha *= k % 30 === 0 ? 1.6 : 0.8;
          g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
        }
        prev = p;
      }
      g.restore();
    }

    function drawNebulae(g, P, pos, alpha, R, dimFn) {
      if (alpha <= 0.01) return;
      g.save();
      g.globalCompositeOperation = 'lighter';
      CLUSTERS.forEach((cl, c) => {
        let x = 0, y = 0, z = 0, n = 0;
        S.words.forEach((wd, i) => { if (wd.c === c) { x += pos[i][0]; y += pos[i][1]; z += pos[i][2]; n++; } });
        const p = P(x / n, y / n, z / n);
        const r = R * 0.5 * p.s;
        g.globalAlpha = alpha * 0.2 * depthA(p.z) * (dimFn ? dimFn(c) : 1);
        g.drawImage(sprite(clusterColor(c), 'neb'), p.x - r, p.y - r, 2 * r, 2 * r);
      });
      g.restore();
    }

    function drawEdges(g, pts, alpha, dimFn) {
      if (alpha <= 0.01) return;
      g.save();
      g.lineWidth = 1;
      for (const [i, j] of edges) {
        const a = pts[i], b = pts[j];
        g.strokeStyle = color(i);
        g.globalAlpha = alpha * 0.32 * depthA((a.z + b.z) / 2) * (dimFn ? Math.min(dimFn(i), dimFn(j)) : 1);
        g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
      }
      g.restore();
    }

    /** Stars, back to front. aFn(i) = alpha multiplier, sFn(i) = size multiplier. */
    function drawStars(g, pts, t, aFn, sFn) {
      const order = pts.map((_, i) => i).sort((i, j) => pts[j].z - pts[i].z);
      g.save();
      for (const i of order) {
        const p = pts[i];
        const a = (aFn ? aFn(i) : 1) * depthA(p.z);
        if (a <= 0.01) continue;
        const tw = 1 + 0.1 * Math.sin(t * 2.1 + phase[i]);
        const r = (1.9 + 1.7 * p.s) * (sFn ? sFn(i) : 1) * tw;
        g.globalAlpha = Math.min(1, a);
        g.drawImage(sprite(color(i)), p.x - 4 * r, p.y - 4 * r, 8 * r, 8 * r);
      }
      g.restore();
      return order;
    }

    /** Labels with greedy collision culling: highest priority first; overlapping ones are skipped.
        Items flagged `alt` also try the left side, above and below before giving up; `force` items
        are drawn even when every spot is taken. */
    function drawLabels(g, items, obstacles = [], W = Infinity, H = Infinity) {
      items.sort((a, b) => b.prio - a.prio);
      const boxes = obstacles.slice();
      const hits = (box) => boxes.some((b) => box[0] < b[0] + b[2] && box[0] + box[2] > b[0] && box[1] < b[1] + b[3] && box[1] + box[3] > b[1]);
      g.save();
      g.textBaseline = 'middle';
      g.lineJoin = 'round';
      let lastFont = '';
      for (const it of items) {
        if (it.alpha <= 0.03) continue;
        const font = AM.font(it.size, it.role || 'body', it.weight || 500);
        if (font !== lastFont) { g.font = font; lastFont = font; }
        const w = g.measureText(it.text).width, dx = it.dx ?? 7;
        const right = it.align === 'right' || it.x + dx + w > W - 3; // flip to the left near the right edge
        const R = [it.x + dx, it.y], L = [it.x - dx - w, it.y];
        const cands = right ? [L, R] : [R, L];
        if (!it.alt) cands.length = 1;
        else {
          const up = it.size + 3;
          cands.push([it.x - w / 2, it.y - up], [it.x - w / 2, it.y + up], // above, below
            [R[0], it.y - up], [R[0], it.y + up], [L[0], it.y - up], [L[0], it.y + up], // diagonals
            [it.x - w / 2, it.y - 2 * up], [it.x - w / 2, it.y + 2 * up]); // one row further out
        }
        const boxOf = ([bx, by]) => [bx - 2, by - it.size * 0.62, w + 4, it.size * 1.24];
        let spot = cands.find((c, k) => (k === 0 || (c[0] > 2 && c[0] + w < W - 2 && c[1] > it.size && c[1] < H - it.size)) && !hits(boxOf(c)));
        if (!spot) { if (!it.force) continue; spot = cands[0]; }
        boxes.push(boxOf(spot));
        g.globalAlpha = Math.min(1, it.alpha);
        g.strokeStyle = AM.rgba(AM.col.ink, 0.85);
        g.lineWidth = 3;
        g.strokeText(it.text, spot[0], spot[1]);
        g.fillStyle = it.color;
        g.fillText(it.text, spot[0], spot[1]);
      }
      g.restore();
    }

    /** Weld silk threads from one star to several, with beads of light flowing along them. */
    function drawNeighbourThreads(g, from, tos, t, alpha, o = {}) {
      if (alpha <= 0.01) return;
      const spr = sprite(o.color || AM.dye.weld);
      tos.forEach(({ p, w }, k) => {
        const bend = (k % 2 ? -1 : 1) * 0.16;
        DR.thread(g, from.x, from.y, p.x, p.y, { color: o.color || AM.dye.weld, width: 0.6 + 1.8 * w, alpha: alpha * (0.35 + 0.6 * w), bend });
        if (AM.reducedMotion) return;
        for (let b = 0; b < 2; b++) {
          const u = (t * (0.35 + 0.3 * w) + b / 2 + k * 0.17) % 1;
          const q = DR.threadPoint(from.x, from.y, p.x, p.y, u, { bend });
          const r = 1.6 + 1.4 * w;
          g.save(); g.globalAlpha = alpha * Math.sin(u * Math.PI); g.drawImage(spr, q.x - 4 * r, q.y - 4 * r, 8 * r, 8 * r); g.restore();
        }
      });
    }

    /** Soften the canvas borders so stars and labels fade out instead of being cut off. */
    function fadeEdges(g, w, h, strength = 1, m = 26) {
      if (strength <= 0.01) return;
      g.save();
      g.globalCompositeOperation = 'destination-out';
      g.globalAlpha = strength;
      const side = (x0, y0, x1, y1, rx, ry, rw, rh) => {
        const grd = g.createLinearGradient(x0, y0, x1, y1);
        grd.addColorStop(0, 'rgba(0,0,0,1)'); grd.addColorStop(1, 'rgba(0,0,0,0)');
        g.fillStyle = grd; g.fillRect(rx, ry, rw, rh);
      };
      side(0, 0, m, 0, 0, 0, m, h); side(w, 0, w - m, 0, w - m, 0, m, h);
      side(0, 0, 0, m, 0, 0, w, m); side(0, h, 0, h - m, 0, h - m, w, m);
      g.restore();
    }

    return { S, clusterColor, color, sprite, camFrame, depthA, fadeEdges, drawDust, drawRing, drawGrid, drawNebulae, drawEdges, drawStars, drawLabels, drawNeighbourThreads };
  }

  const DR = AM.draw;

  /** Fire fn(true/false) as an element enters / leaves the viewport (to skip drawing off-screen canvases). */
  const watchVisible = (elm, fn) => {
    if (typeof IntersectionObserver === 'undefined') { fn(true); return; }
    new IntersectionObserver((es) => es.forEach((e) => fn(e.isIntersecting)), { rootMargin: '80px 0px' }).observe(elm);
  };

  /** Pointer helper: drag (dx, dy), tap (x, y), hover (x, y) / leave. Touch keeps vertical page scroll (pan-y). */
  function pointerControls(canvas, api, { onDrag, onTap, onHover, onLeave }) {
    let down = null;
    canvas.addEventListener('pointerdown', (e) => {
      down = { x: e.clientX, y: e.clientY, lx: e.clientX, ly: e.clientY, moved: false, id: e.pointerId, type: e.pointerType };
      try { canvas.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    });
    canvas.addEventListener('pointermove', (e) => {
      if (down && e.pointerId === down.id) {
        const dx = e.clientX - down.lx, dy = e.clientY - down.ly;
        if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) down.moved = true;
        down.lx = e.clientX; down.ly = e.clientY;
        if (down.moved && onDrag) onDrag(dx, dy, down.type);
        if (down.moved) canvas.style.cursor = 'grabbing';
      } else if (e.pointerType === 'mouse' && onHover) {
        const p = api.pointer(e); onHover(p.x, p.y);
      }
    });
    const end = (e) => {
      if (!down || e.pointerId !== down.id) return;
      if (!down.moved && e.type === 'pointerup' && onTap) { const p = api.pointer(e); onTap(p.x, p.y, down.type); }
      down = null;
      canvas.style.cursor = '';
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse' && onLeave && !down) onLeave(); });
  }

  /** Nearest projected star to (x, y) within radius r, or -1. */
  const pick = (pts, x, y, r, ok) => {
    let best = -1, bd = r * r;
    pts.forEach((p, i) => { if (ok && !ok(i)) return; const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = i; } });
    return best;
  };

  // ======================================================================
  // 3. Chapter-scoped CSS
  // ======================================================================

  const CSS = `
  #ch-embed .em-stage-fig { gap: 10px; }
  #ch-embed .em-stage-fig .fig-top { flex-wrap: nowrap; }
  #ch-embed .em-stage-fig .fig-title { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #ch-embed .em-stage-fig.is-sky .stage-canvas canvas { cursor: grab; }
  #ch-embed .em-pick { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
  #ch-embed .em-pick .tok { padding: 4px 10px 5px; font-size: 0.9rem; }
  #ch-embed .em-step-tag { font-family: var(--font-mono); font-size: 0.8em; color: var(--weld); }
  #ch-embed .step em, #ch-embed .prose em { color: var(--linen); font-style: italic; }
  #ch-embed q { color: var(--linen); font-style: italic; }
  #ch-embed .math.block { font-size: 0.92em; white-space: nowrap; }
  @media (max-width: 420px) { #ch-embed .math.block { font-size: 0.8em; padding-inline: 8px; } }
  #ch-embed .em-sky-cap { margin: 0; }
  @media (min-width: 961px) {
    #ch-embed div.em-explorer { row-gap: 14px; }
    #ch-embed .em-explorer > .em-sky { grid-column: 1; grid-row: 1; }
    #ch-embed .em-explorer > .em-panel { grid-column: 2; grid-row: 1 / span 2; }
    #ch-embed .em-explorer > .em-sky-cap { grid-column: 1; grid-row: 2; }
  }
  @media (max-width: 900px) {
    #ch-embed .em-stage-fig figcaption { display: none; }
    #ch-embed .em-stage-fig { gap: 6px; }
  }

  #ch-embed .em-explorer { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: clamp(16px, 2.5vw, 32px); align-items: start; }
  @media (max-width: 960px) { #ch-embed .em-explorer { grid-template-columns: minmax(0, 1fr); } }
  #ch-embed .em-sky { display: grid; gap: 12px; min-width: 0; }
  #ch-embed .em-sky .stage-canvas canvas { cursor: grab; }
  #ch-embed .em-legend { display: flex; flex-wrap: wrap; gap: 6px; }
  #ch-embed .em-legend .tok { font-family: var(--font-mono); font-weight: 400; font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase; padding: 4px 9px 6px; color: var(--linen-dim); }
  #ch-embed .em-legend .tok[aria-pressed='true'] { color: var(--linen); }

  #ch-embed .em-panel { display: grid; gap: 16px; }
  #ch-embed .em-h { font-family: var(--font-mono); font-size: var(--fs-micro); letter-spacing: 0.14em; text-transform: uppercase; color: var(--weld); }
  #ch-embed .em-eq { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr); align-items: center; gap: 6px; }
  #ch-embed .em-op { font-family: var(--font-mono); color: var(--mist); font-size: 1rem; text-align: center; }
  #ch-embed .em-sel { position: relative; display: block; min-width: 0; }
  #ch-embed .em-sel .em-slot { position: absolute; left: 9px; top: -7px; padding: 0 4px; background: var(--ink-2); font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.1em; color: var(--mist); pointer-events: none; }
  #ch-embed .em-sel::after { content: ''; position: absolute; right: 10px; top: 50%; width: 6px; height: 6px; margin-top: -5px; border-right: 1.5px solid var(--mist); border-bottom: 1.5px solid var(--mist); transform: rotate(45deg); pointer-events: none; }
  #ch-embed .em-sel select { -webkit-appearance: none; appearance: none; width: 100%; min-height: 40px; padding: 8px 22px 8px 9px; border-radius: var(--radius-sm); border: 1px solid var(--rule-strong); background: var(--ink); color: var(--linen); font-family: var(--font-body); font-weight: 600; font-size: 0.95rem; cursor: pointer; text-overflow: ellipsis; }
  #ch-embed .em-sel select:hover { border-color: var(--linen-dim); }
  #ch-embed .em-sel select:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
  #ch-embed .em-sel select option, #ch-embed .em-sel select optgroup { background: var(--ink-2); color: var(--linen); }
  #ch-embed .em-ctl-row { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px 16px; }
  #ch-embed .em-ctl-row .seg button { padding: 5px 11px; }
  #ch-embed .em-presets .tok { font-size: 0.82rem; font-weight: 500; padding: 4px 9px 5px; }
  #ch-embed .em-results { list-style: none; margin: 0; padding: 0; display: grid; gap: 7px; }
  #ch-embed .em-results li { display: grid; grid-template-columns: 18px minmax(0, 1fr) auto; align-items: center; column-gap: 8px; row-gap: 3px; font-size: 0.95rem; }
  #ch-embed .em-results .em-rank { font-family: var(--font-mono); font-size: 10px; color: var(--mist); }
  #ch-embed .em-results .em-word { color: var(--linen); font-weight: 600; display: inline-flex; align-items: center; gap: 8px; min-width: 0; }
  #ch-embed .em-results .em-word::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: var(--c); box-shadow: 0 0 8px var(--c); flex: none; }
  #ch-embed .em-results .em-cos { font-family: var(--font-mono); font-size: 11px; color: var(--linen-dim); font-variant-numeric: tabular-nums; }
  #ch-embed .em-results .em-bar { grid-column: 2 / 4; height: 2px; border-radius: 2px; background: var(--rule); overflow: hidden; }
  #ch-embed .em-results .em-bar span { display: block; height: 100%; width: var(--w); background: var(--c); box-shadow: 0 0 8px var(--c); transition: width 0.6s cubic-bezier(.2,.8,.2,1); }
  #ch-embed .em-results li.is-top .em-word { color: var(--weld); }
  #ch-embed .em-results li.is-input .em-word::after { content: 'input'; font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--madder); border: 1px solid color-mix(in srgb, var(--madder) 50%, transparent); border-radius: 4px; padding: 0 4px; }
  #ch-embed .em-note { font-size: var(--fs-small); line-height: 1.5; color: var(--mist); }
  #ch-embed .em-note strong { color: var(--linen-dim); font-weight: 600; }

  `;

  // ======================================================================
  // 4. Figure A: the scrollytelling stage (table → row → stars → shadows)
  // ======================================================================

  function mountStage(ctx, kit, stepEls, dyn) {
    const { el } = ctx;
    const { S } = kit;
    const CHIPS = ['queen', 'cat', 'paris', 'swam', 'blue'].map((w) => S.index.get(w));

    const st = {
      step: 0, sel: CHIPS[0],
      mLay: 0, bar: 0, u: 0, train: 0, focus: 0, pca: 0, // current
      T: { mLay: 0, bar: 0, u: 0, train: 0, focus: 0, pca: 0 }, // targets
      cam: { yaw: 0.5, pitch: 0.42, dist: 3.6, R: 100, cx: 0, cy: 0 },
      drag: 0, lastT: 0, visible: true,
    };
    st.rowF = S.words[st.sel].id;            // highlighted row (glides when a new token is picked)
    st.vec = Float64Array.from(S.E[st.sel]); // barcode values (morph between tokens)

    // Random "before training" positions for each star (illustration).
    const rnd = AM.math.rng(11);
    const rand3 = S.words.map(() => [AM.math.randn(rnd) * 0.5, AM.math.randn(rnd) * 0.36, AM.math.randn(rnd) * 0.5]);
    const rowDelay = S.words.map((wd) => 0.5 * (wd.id / S.N) + 0.05 * rnd());
    const trainDelay = S.words.map(() => 0.42 * rnd());

    const badgeSlot = el('span', {}, AM.ui.badge('toy'));
    const title = el('span', { class: 'fig-title' }, 'The embedding table');
    const fig = el('figure', { class: 'fig em-stage-fig' }, el('div', { class: 'fig-top' }, title, badgeSlot));
    const cv = ctx.canvas(fig, {
      height: (w) => (w < 560 ? w * 0.88 : w * 0.94),
      maxHeight: () => (window.innerWidth <= 900 ? Math.round(window.innerHeight * 0.42) : Math.min(640, Math.round(window.innerHeight * 0.72))),
      label: 'The toy embedding matrix: 99 rows, one per token, by 64 columns. As you scroll, a row slides out as a vector, then every row becomes a star in a 3D constellation.',
    });
    const chips = AM.ui.tokens(CHIPS.map((i) => S.words[i].w), {
      selected: 0, label: 'Pick a token', colors: (k) => kit.color(CHIPS[k]),
      onSelect: (k) => { st.sel = CHIPS[k]; updateDyn(); if (AM.reducedMotion) snap(); draw(st.lastT); },
    });
    chips.chips.forEach((c, k) => { c.id = 'em-stage-tok-' + k; });
    fig.append(
      el('div', { class: 'em-pick' }, el('span', { class: 'ctl-label' }, 'Pick a token'), chips.el),
      el('figcaption', { html: `Toy numbers: ${S.N} tokens, d_model = ${D}. The scattering and the drift in steps 2 and 3 are drawn to illustrate training.` }),
    );

    // ---- the matrix, cached as a woven colour field (rebuilt on resize)
    let off = null;
    const layout = (k) => {
      const { w, h } = cv;
      const left = w < 560 ? Math.round(w * 0.33) : Math.round(w * 0.3);
      const top = 34;
      const bottom = lerp(h - 30, h * 0.4, k);
      return { x: left, y: top, w: w - left - 10, h: bottom - top };
    };
    function buildMatrix() {
      const r = layout(0), dpr = cv.dpr;
      off = document.createElement('canvas');
      off.width = Math.round(r.w * dpr); off.height = Math.round(r.h * dpr);
      const g = off.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      const cw = r.w / D, rh = r.h / S.N;
      const gx = cw > 3.5 ? 1 : 0.5, gy = rh > 3.2 ? 1 : 0;
      for (let row = 0; row < S.N; row++) {
        const e = S.E[S.rowWord[row]];
        for (let j = 0; j < D; j++) {
          g.fillStyle = AM.color.div(e[j], S.vmax);
          g.fillRect(j * cw, row * rh, cw - gx, rh - gy);
          if ((row + j) % 2 === 0) { g.fillStyle = 'rgba(0,0,0,0.13)'; g.fillRect(j * cw, row * rh, cw - gx, rh - gy); } // plain-weave shading
          if (rh >= 4) { g.fillStyle = 'rgba(255,248,230,0.07)'; g.fillRect(j * cw, row * rh, cw - gx, Math.max(1, rh * 0.3)); } // thread sheen
        }
      }
    }

    // ---- steps
    const stepTargets = [
      { mLay: 1, bar: 1, u: 0, train: 0, focus: 0, pca: 0 }, // the table, and one row sliding out as a vector
      { mLay: 0, bar: 0, u: 1, train: 0, focus: 0, pca: 0 }, // rows unravel into (random) stars
      { mLay: 0, bar: 0, u: 1, train: 1, focus: 0, pca: 0 }, // training gathers them
      { mLay: 0, bar: 0, u: 1, train: 1, focus: 1, pca: 0 }, // nearest neighbours
      { mLay: 0, bar: 0, u: 1, train: 1, focus: 0, pca: 1 }, // PCA axes + spread
    ];
    const titles = ['The embedding table', 'Every row, a star', 'Training', 'Nearest neighbours', 'A 3D shadow of 64-d'];
    let badgeKind = 'toy';
    ctx.steps(stepEls, (i) => {
      st.step = i;
      Object.assign(st.T, stepTargets[i]);
      title.textContent = titles[i];
      fig.classList.toggle('is-sky', i >= 1);
      const kind = i === 1 || i === 2 ? 'illustration' : 'toy';
      if (kind !== badgeKind) { badgeKind = kind; badgeSlot.replaceChildren(AM.ui.badge(kind)); }
      if (AM.reducedMotion) snap();
      draw(st.lastT);
    });
    /** Reduced motion: jump straight to the state the current step describes. */
    function snap() {
      Object.assign(st, st.T);
      st.hold = 0;
      st.rowF = S.words[st.sel].id;
      st.vec.set(S.E[st.sel]);
    }
    // At rest: the whole table. On first view, after a beat, the chosen row slides out as a vector.
    Object.assign(st, { mLay: 0, bar: 0, u: 0, train: 0, focus: 0, pca: 0, hold: 0.9 });
    if (AM.reducedMotion) snap();

    function updateDyn() {
      const wd = S.words[st.sel];
      dyn.word.forEach((n) => { n.textContent = wd.w; });
      dyn.id.forEach((n) => { n.textContent = String(wd.id); });
      const nb = S.nbrs[st.sel][0];
      dyn.nb.forEach((n) => { n.textContent = S.words[nb].w; });
      dyn.nbcos.forEach((n) => { n.textContent = S.cosM[st.sel][nb].toFixed(2); });
    }
    updateDyn();

    // ---- drawing
    function drawMatrix(g, t) {
      const { w } = cv;
      const r = layout(st.mLay);
      const rowH = r.h / S.N;
      const anyGone = st.u > 0.001;
      // the empty loom frame
      g.save();
      g.strokeStyle = AM.rgba(AM.col.linen, 0.12 * (1 - smooth(0.6, 1, st.u)));
      g.strokeRect(r.x - 3.5, r.y - 3.5, r.w + 6, r.h + 6);
      g.restore();
      if (!anyGone) {
        g.drawImage(off, 0, 0, off.width, off.height, r.x, r.y, r.w, r.h);
        if (!AM.reducedMotion) {
          // a slow band of light sweeping across the weave, like sheen on silk
          const span = r.w + r.h, c = r.x - r.h + ((t * 0.08) % 1.5) * span, d = 70;
          const grd = g.createLinearGradient(c - d * 0.7, r.y - d * 0.7, c + d * 0.7, r.y + d * 0.7);
          grd.addColorStop(0, 'rgba(255,248,230,0)'); grd.addColorStop(0.5, 'rgba(255,248,230,0.11)'); grd.addColorStop(1, 'rgba(255,248,230,0)');
          g.save(); g.globalCompositeOperation = 'lighter'; g.fillStyle = grd; g.fillRect(r.x, r.y, r.w, r.h); g.restore();
        }
      }
      else {
        const sh = off.height / S.N;
        for (let row = 0; row < S.N; row++) {
          if (st.u - rowDelay[S.rowWord[row]] > 0) continue; // this row has flown out
          g.drawImage(off, 0, row * sh, off.width, sh, r.x, r.y + row * rowH, r.w, rowH);
        }
      }
      const la = 1 - smooth(0, 0.4, st.u); // labels fade as the rows leave
      if (la <= 0.01) return;
      // axis notes
      const small = w < 560;
      DR.text(g, 'd_model = 64 columns', r.x + r.w / 2, r.y - 12, { size: small ? 9 : 10, role: 'mono', color: AM.col.mist, align: 'center', alpha: la });
      DR.text(g, '0', r.x, r.y - 12, { size: 9, role: 'mono', color: AM.col.mist, alpha: la * 0.8 });
      DR.text(g, '63', r.x + r.w, r.y - 12, { size: 9, role: 'mono', color: AM.col.mist, align: 'right', alpha: la * 0.8 });
      const ba = la * (1 - st.mLay);
      if (ba > 0.01) DR.text(g, `E  ·  ${S.N} tokens × ${D} dims`, r.x + r.w / 2, r.y + r.h + 20, { size: small ? 9 : 10, role: 'mono', color: AM.col.mist, align: 'center', alpha: ba });
      // row ticks every 10 ids
      g.save(); g.fillStyle = AM.col.mist; g.globalAlpha = la * 0.6;
      for (let row = 0; row < S.N; row += 10) g.fillRect(r.x - 9, r.y + row * rowH + rowH / 2 - 0.5, 4, 1);
      g.restore();
      // selected row: weld frame + the token chip and a thread into the row
      const sr = S.words[st.sel].id;
      const y = r.y + st.rowF * rowH;
      g.save();
      g.globalAlpha = la;
      g.shadowColor = AM.rgba(AM.dye.weld, 0.8); g.shadowBlur = 12;
      g.strokeStyle = AM.dye.weld; g.lineWidth = 1.5;
      g.strokeRect(r.x - 2.5, y - 1.5, r.w + 5, rowH + 3);
      g.restore();
      const cx = r.x * 0.45, cy = clamp(y + rowH / 2, r.y + 18, r.y + r.h - 34);
      const tok = DR.token(g, S.words[st.sel].w, cx, cy, { size: small ? 13 : 15, selected: true, underline: kit.color(st.sel), alpha: la });
      DR.text(g, `token id ${sr}`, cx, tok.y + tok.h + 14, { size: small ? 9 : 10, role: 'mono', color: AM.col.mist, align: 'center', alpha: la });
      const x1 = tok.x + tok.w + 4, x2 = r.x - 6, y2 = y + rowH / 2;
      DR.thread(g, x1, cy, x2, y2, { color: AM.dye.weld, width: 1.3, alpha: la * 0.9, bend: 0.12 });
      if (!AM.reducedMotion) {
        const u = (t * 0.6) % 1;
        const q = DR.threadPoint(x1, cy, x2, y2, u, { bend: 0.12 });
        g.save(); g.globalAlpha = la * Math.sin(u * Math.PI); g.drawImage(kit.sprite(AM.dye.weld), q.x - 9, q.y - 9, 18, 18); g.restore();
      }
    }

    function drawBarcode(g) {
      const k = ease(st.bar);
      if (k <= 0.005) return;
      const { w, h } = cv;
      const small = w < 560;
      const r = layout(st.mLay);
      const rowH = r.h / S.N;
      const sr = S.words[st.sel].id;
      const src = { x: r.x, y: r.y + st.rowF * rowH, w: r.w, h: rowH };
      const dst = { x: 14, y: h * 0.5, w: w - 28, h: Math.max(36, h * 0.15) };
      const b = { x: lerp(src.x, dst.x, k), y: lerp(src.y, dst.y, k), w: lerp(src.w, dst.w, k), h: lerp(src.h, dst.h, k) };
      const e = S.E[st.sel];
      // the row strip (cell colours) morphs into a thread-barcode (bars ∝ value)
      const strip = 1 - smooth(0.45, 0.85, k), bars = smooth(0.45, 0.85, k);
      if (strip > 0.01) { g.save(); g.globalAlpha = strip; const sh = off.height / S.N; g.drawImage(off, 0, sr * sh, off.width, sh, b.x, b.y, b.w, b.h); g.restore(); }
      if (bars > 0.01) {
        g.save(); g.globalAlpha = bars * 0.5; g.fillStyle = AM.col.ruleStrong; g.fillRect(b.x, b.y + b.h / 2 - 0.5, b.w, 1); g.restore();
        DR.vectorBars(g, b.x, b.y, b.w, b.h, st.vec, { max: S.vmax * 1.25, alpha: bars });
      }
      const ta = smooth(0.6, 1, k);
      if (ta <= 0.01) return;
      const fs = small ? 9.5 : 10.5;
      DR.text(g, `x = E[${sr}]   the vector for “${S.words[st.sel].w}”`, dst.x, dst.y - 10, { size: fs, role: 'mono', color: AM.col.linen, alpha: ta });
      DR.text(g, '64 numbers', dst.x + dst.w, dst.y - 10, { size: fs, role: 'mono', color: AM.col.mist, align: 'right', alpha: ta });
      const vals = Array.from(e.slice(0, small ? 4 : 6), (v) => fmt(v)).join('  ');
      DR.text(g, `[ ${vals}  … ]`, dst.x, dst.y + dst.h + 16, { size: fs, role: 'mono', color: AM.col.linenDim, alpha: ta });
      // how long is one embedding? threads to scale: 64 vs 768 vs 12,288 numbers
      const rows = [['this toy', 64], ['GPT-2 small', 768], ['GPT-3', 12288]];
      const y0 = dst.y + dst.h + (small ? 38 : 52);
      const gap = clamp((h - 16 - y0) / 2.4, 18, 40);
      const L = dst.w;
      rows.forEach(([name, n], i) => {
        const y = y0 + i * gap;
        const a = ta * smooth(0.65 + i * 0.1, 0.9 + i * 0.1, k + 0.2);
        DR.text(g, name, dst.x, y, { size: fs, role: 'mono', color: AM.col.mist, alpha: a });
        DR.text(g, n.toLocaleString('en-US'), dst.x + L, y, { size: fs, role: 'mono', color: i === 0 ? AM.dye.weld : AM.col.linenDim, align: 'right', alpha: a });
        const len = Math.max(2, (L * n) / 12288);
        g.save();
        g.globalAlpha = a;
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.22); g.lineWidth = 5; g.lineCap = 'round';
        g.beginPath(); g.moveTo(dst.x, y + 7); g.lineTo(dst.x + len, y + 7); g.stroke();
        g.strokeStyle = AM.dye.weld; g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(dst.x, y + 7); g.lineTo(dst.x + len, y + 7); g.stroke();
        g.restore();
        g.save(); g.globalAlpha = a; g.drawImage(kit.sprite(AM.dye.weld), dst.x + len - 8, y + 7 - 8, 16, 16); g.restore();
      });
    }

    function draw(t) {
      if (!off || !cv.w) return;
      const g = cv.g, { w, h } = cv;
      cv.clear();
      const small = w < 560;
      const uA = smooth(0, 0.35, st.u);              // how much of the sky is showing
      const trainE = ease(st.train);
      // star positions: random (before training) → toy embedding (after), each with its own delay
      const pos = S.words.map((_, i) => {
        const q = ease((st.train - trainDelay[i]) / 0.58);
        const a = rand3[i], b = S.P3[i];
        return [lerp(a[0], b[0], q), lerp(a[1], b[1], q), lerp(a[2], b[2], q)];
      });
      const cam = st.cam;
      cam.cx = w * (small ? 0.47 : 0.46);
      cam.cy = h * lerp(0.52, 0.46, st.pca);
      cam.R = Math.min(w * (small ? 0.4 : 0.39), h * 0.5);
      cam.zoom = 1; cam.ox = 0; cam.oy = 0;
      let P = kit.camFrame(cam);
      const fE = ease(st.focus);
      if (fE > 0.001) {
        // step 5: the camera glides toward the chosen star and zooms in a little
        const pf = P(...pos[st.sel]);
        cam.bx = pf.x; cam.by = pf.y; cam.zoom = 1 + 0.5 * fE;
        cam.ox = (w * 0.47 - pf.x) * 0.8 * fE; cam.oy = (h * 0.54 - pf.y) * 0.8 * fE;
        P = kit.camFrame(cam);
      }
      const pts = pos.map((p) => P(p[0], p[1], p[2]));

      if (uA > 0.01) {
        kit.drawDust(g, P, uA, t);
        kit.drawNebulae(g, P, pos, uA * trainE, cam.R);
        kit.drawRing(g, P, uA * (0.6 + 0.4 * trainE));
      }
      if (st.u < 0.999) drawMatrix(g, t);
      drawBarcode(g);
      if (uA <= 0.001 && st.u <= 0.001) return;

      // PCA axes (step 6)
      if (st.pca > 0.01) {
        const names = S.pca.axes.map((k) => `PC${k + 1} · ${Math.round(100 * S.pca.vals[k] / S.pca.total)}%`);
        const ends = [[1.4, 0, 0], [0, 1.1, 0], [0, 0, 1.4]];
        const axLabels = [];
        ends.forEach((e, k) => {
          const a = P(-e[0], -e[1], -e[2]), b = P(e[0], e[1], e[2]);
          g.save(); g.globalAlpha = st.pca * 0.5; g.setLineDash([3, 4]); g.strokeStyle = AM.col.linenDim; g.lineWidth = 1;
          g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke(); g.restore();
          axLabels.push({ text: names[k], x: clamp(b.x, 30, w - 30), y: clamp(b.y + (k === 1 ? -8 : 0), 34, h - 40), dx: 5, size: small ? 9 : 10, role: 'mono', weight: 500, color: AM.dye.weld, alpha: st.pca, prio: 0, force: true });
        });
        st.axLabels = axLabels;
      }

      const sel = st.sel;
      const fA = st.focus;
      const top5 = S.nbrs[sel].slice(0, 5);
      const isNb = new Set(top5);
      const dim = (i) => (i === sel || isNb.has(i) ? 1 : 1 - 0.62 * fA);
      // per-star appear progress during the unravel
      const appear = (i) => smooth(0.75, 1, (st.u - rowDelay[i]) / 0.42);
      kit.drawEdges(g, pts, uA * trainE * (1 - 0.6 * fA), (i) => appear(i) * dim(i));

      // flying rows: each row's strip shrinks to a bead and flies to its star
      if (st.u > 0.001 && st.u < 0.999) {
        const r = layout(st.mLay), rowH = r.h / S.N, sh = off.height / S.N;
        for (let i = 0; i < S.N; i++) {
          const p = (st.u - rowDelay[i]) / 0.42;
          if (p <= 0 || p >= 1) continue;
          const e = ease(p), row = S.words[i].id;
          const sx = r.x + r.w / 2, sy = r.y + row * rowH + rowH / 2, tx = pts[i].x, ty = pts[i].y;
          const mx = (sx + tx) / 2 + (sy - ty) * 0.25, my = (sy + ty) / 2 - Math.abs(sx - tx) * 0.2; // curved flight
          const fx = (1 - e) * (1 - e) * sx + 2 * (1 - e) * e * mx + e * e * tx;
          const fy = (1 - e) * (1 - e) * sy + 2 * (1 - e) * e * my + e * e * ty;
          const ww = r.w * (1 - e) * (1 - e) + 3, hh = Math.max(1.5, rowH * (1 - e) + 2 * e);
          const fade = 1 - smooth(0.8, 1, p);
          if (p > 0.2) {
            // as the strip narrows it trails a thread in its group's dye: the row spun into yarn
            const e0 = ease(p - 0.22), col = kit.color(i);
            g.save(); g.lineCap = 'round'; g.strokeStyle = col;
            g.beginPath();
            for (let k = 0; k <= 6; k++) {
              const u = lerp(e0, e, k / 6), qx = (1 - u) * (1 - u) * sx + 2 * (1 - u) * u * mx + u * u * tx, qy = (1 - u) * (1 - u) * sy + 2 * (1 - u) * u * my + u * u * ty;
              if (k) g.lineTo(qx, qy); else g.moveTo(qx, qy);
            }
            g.globalAlpha = fade * 0.18 * smooth(0.2, 0.4, p); g.lineWidth = 4; g.stroke();
            g.globalAlpha = fade * 0.75 * smooth(0.2, 0.4, p); g.lineWidth = 1.1; g.stroke();
            g.restore();
          }
          g.save(); g.globalAlpha = fade;
          g.drawImage(off, 0, row * sh, off.width, sh, fx - ww / 2, fy - hh / 2, ww, hh);
          g.restore();
        }
      }

      kit.drawStars(g, pts, t, (i) => appear(i) * dim(i) * (0.55 + 0.45 * ease((st.train - trainDelay[i]) / 0.58)), (i) => (i === sel && st.step >= 1 ? 1.5 : 1));
      // a small spark as each thread lands and becomes a star
      if (st.u > 0.001 && st.u < 0.999) {
        g.save();
        for (let i = 0; i < S.N; i++) {
          const p = (st.u - rowDelay[i]) / 0.42;
          if (p < 0.82 || p > 1.12) continue;
          const k = (p - 0.82) / 0.3, r = 3 + 14 * easeOut(k);
          g.globalAlpha = (1 - k) * 0.85;
          g.drawImage(kit.sprite(kit.color(i)), pts[i].x - r * 2, pts[i].y - r * 2, r * 4, r * 4);
        }
        g.restore();
      }

      // focus: threads to the 5 nearest neighbours by cosine (step 5)
      if (fA > 0.01) {
        kit.drawNeighbourThreads(g, pts[sel], top5.map((j) => ({ p: pts[j], w: clamp((S.cosM[sel][j] - 0.4) / 0.55) })), t, fA);
      }
      // selected star ring
      if (st.step >= 1 && uA > 0.5) {
        const p = pts[sel];
        g.save(); g.globalAlpha = uA * appear(sel) * 0.9; g.strokeStyle = AM.dye.weld; g.lineWidth = 1.2;
        g.beginPath(); g.arc(p.x, p.y, 9 + 1.5 * Math.sin(t * 2.4), 0, Math.PI * 2); g.stroke(); g.restore();
      }

      // labels
      const items = [];
      const fsz = small ? 10 : 11;
      pts.forEach((p, i) => {
        const a = appear(i) * kit.depthA(p.z);
        if (a <= 0.03) return;
        const isSel = i === sel && st.step >= 1, nb = isNb.has(i) && fA > 0.3;
        items.push({
          text: nb ? `${S.words[i].w}  ${S.cosM[sel][i].toFixed(2)}` : S.words[i].w,
          x: p.x, y: p.y, size: isSel ? fsz + 2.5 : nb ? fsz + 0.5 : fsz, weight: isSel || nb ? 600 : 500,
          color: isSel ? AM.dye.weld : nb ? AM.col.linen : AM.mix(AM.col.linenDim, kit.color(i), 0.35 * trainE),
          alpha: a * (isSel || nb ? 1 : 0.85 * dim(i)), prio: (isSel ? 100 : nb ? 50 : 0) - p.z, force: isSel || nb, alt: nb,
        });
      });
      if (st.pca > 0.01 && st.axLabels) st.axLabels.forEach((l) => items.push({ ...l, prio: 200 })); // axis names win; star labels make way
      const ps = pts[sel];
      kit.drawLabels(g, items, st.step >= 1 ? [[ps.x - 11, ps.y - 11, 22, 22]] : [], w, h);
      kit.fadeEdges(g, w, h, uA);
      if (fA > 0.01) DR.text(g, `nearest to “${S.words[sel].w}” · cosine in 64-d`, 12, 18, { size: small ? 9 : 10, role: 'mono', color: AM.col.mist, alpha: fA });

      // scree strip: share of the spread along each of the 64 PCA directions (step 6)
      if (st.pca > 0.01) {
        const x0 = 14, x1 = w - 14, y = h - 14, tot = S.pca.total;
        let x = x0;
        g.save();
        S.pca.vals.forEach((v, k) => {
          const ww = ((x1 - x0) * Math.max(0, v)) / tot;
          g.globalAlpha = st.pca * (k < 3 ? 1 - k * 0.12 : 0.55);
          g.fillStyle = k < 3 ? AM.dye.weld : k % 2 ? AM.col.ruleStrong : AM.col.mist;
          g.fillRect(x, y, Math.max(0.5, ww - 1), 5);
          x += ww;
        });
        g.restore();
        DR.text(g, small ? 'spread per PCA direction' : 'spread along each of the 64 PCA directions', x0, y - 8, { size: small ? 9 : 10, role: 'mono', color: AM.col.mist, alpha: st.pca });
        DR.text(g, `top 3 keep ${Math.round(100 * S.pca.top3)}%`, x1, y - 8, { size: small ? 9 : 10, role: 'mono', color: AM.dye.weld, align: 'right', alpha: st.pca });
      }
    }

    cv.onResize(() => { buildMatrix(); draw(st.lastT); });
    watchVisible(cv.wrap, (v) => { st.visible = v; });

    // drag to turn the sky (once it exists)
    pointerControls(cv.canvas, cv, {
      onDrag: (dx) => { if (st.step >= 1) { st.drag += dx * 0.008; st.idle = 0; } },
    });

    ctx.loop((t, dt) => {
      st.lastT = t;
      if (!st.visible) return;
      const T = st.T;
      if (AM.reducedMotion) { snap(); draw(t); return; }
      if (st.hold > 0) { st.hold -= dt; draw(t); return; } // a beat on the full table before the row slides out
      st.mLay = damp(st.mLay, T.mLay, 4, dt);
      st.bar = approach(st.bar, T.bar, dt * (T.bar > st.bar ? 0.8 : 2.2));
      st.u = approach(st.u, T.u, dt * (T.u > st.u ? 0.42 : 0.8));
      st.train = approach(st.train, T.train, dt * 0.36);
      st.focus = damp(st.focus, T.focus, 4, dt);
      st.pca = damp(st.pca, T.pca, 4, dt);
      st.rowF = damp(st.rowF, S.words[st.sel].id, 8, dt);
      const tv = S.E[st.sel];
      for (let j = 0; j < D; j++) st.vec[j] = damp(st.vec[j], tv[j], 9, dt);
      st.idle = (st.idle || 0) + dt;
      if (!AM.reducedMotion && st.u > 0.01) st.cam.yaw += dt * 0.07 * smooth(1.2, 2.5, st.idle); // pause briefly after a drag
      st.cam.yaw += st.drag; st.drag = 0;
      draw(t);
    });

    return fig;
  }

  // ======================================================================
  // 5. Figure B: the constellation explorer + vector arithmetic
  // ======================================================================

  const PRESETS = [
    ['king', 'man', 'woman'],
    ['paris', 'france', 'italy'],
    ['walked', 'walk', 'swim'],
    ['kitten', 'cat', 'dog'],
    ['two', 'one', 'three'],
    ['tokyo', 'japan', 'egypt'],
  ];

  function mountExplorer(ctx, kit, captionHtml) {
    const { el } = ctx;
    const { S } = kit;
    const DUR = 1.8; // seconds for the stars to glide between shadows
    const TAU = Math.PI * 2;
    const st = {
      cam: { yaw: -0.35, pitch: 0.4, dist: 3.6, R: 100, cx: 0, cy: 0 },
      vel: 0, idle: 99, hover: -1, pinned: -1, spot: -1,
      an: null, anT0: -99, played: false, introAt: 0, visible: false, lastT: 0, pts: [],
      view: 0, vmix: 0, yawBase: 0,     // view 0 = global PCA shadow, 1 = analogy plane
      from: S.P3, to: S.P3, fromT: [0, 0, 0], toT: [0, 0, 0], tr0: -99,
    };

    const sky = el('div', { class: 'em-sky' });
    const cv = ctx.canvas(sky, {
      height: (w) => (w < 600 ? Math.max(340, w) : Math.min(640, Math.max(460, w * 0.74))),
      label: 'A rotating 3D constellation of 99 toy word embeddings, grouped by meaning.',
    });
    const legend = AM.ui.tokens(CLUSTERS.map((c) => c.label), {
      label: 'Spotlight a group', colors: (c) => kit.clusterColor(c),
      onSelect: (c) => { st.spot = st.spot === c ? -1 : c; legend.select(st.spot); },
    });
    legend.chips.forEach((c, k) => { c.id = 'em-spot-' + k; });
    legend.el.classList.add('em-legend');
    sky.append(legend.el);
    // the caption sits under the sky on wide screens (no dead band beside the tall panel) and after the panel on phones
    const skyCap = captionHtml ? el('p', { class: 'caption em-sky-cap', html: captionHtml }) : null;

    // ---- glides between shadows: every star moves from where it is now to its new target
    const lerp3 = (a, b, k) => [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)];
    const current = (t) => {
      const k = ease((t - st.tr0) / DUR);
      return { pos: st.from.map((p, i) => lerp3(p, st.to[i], k)), T: lerp3(st.fromT, st.toT, k) };
    };
    const retarget = (instant) => {
      const tg = st.view === 1 ? { pos: st.an.plane.P, T: st.an.plane.T } : { pos: S.P3, T: st.an.res.p };
      const cur = instant ? tg : current(st.lastT);
      st.from = cur.pos; st.fromT = cur.T; st.to = tg.pos; st.toT = tg.T;
      st.tr0 = instant ? -99 : st.lastT;
    };
    const setView = (v) => {
      if (v === st.view) return;
      st.view = v; viewSeg.set(v);
      if (v === 1) st.yawBase = Math.round(st.cam.yaw / TAU) * TAU;
      retarget(AM.reducedMotion);
    };

    // ---- panel: A − B + C
    const groups = CLUSTERS.map((cl, c) => el('optgroup', { label: cl.label },
      S.words.map((wd, i) => (wd.c === c ? el('option', { value: String(i) }, wd.w) : null))));
    const mkSel = (slot, id) => {
      const s = el('select', { id, 'aria-label': `Word ${slot}` }, groups.map((gp) => gp.cloneNode(true)));
      s.addEventListener('change', () => { setAnalogy(+selA.value, +selB.value, +selC.value); matchPreset(); });
      return { wrap: el('label', { class: 'em-sel' }, el('span', { class: 'em-slot' }, slot), s), s };
    };
    const A = mkSel('A', 'em-sel-a'), B = mkSel('B', 'em-sel-b'), C = mkSel('C', 'em-sel-c');
    const selA = A.s, selB = B.s, selC = C.s;
    const presets = AM.ui.tokens(PRESETS.map((p) => `${p[0]} − ${p[1]} + ${p[2]}`), {
      selected: 0, label: 'Presets',
      onSelect: (k) => { const [a, b, c] = PRESETS[k].map((w) => S.index.get(w)); setAnalogy(a, b, c); },
    });
    presets.chips.forEach((c, k) => { c.id = 'em-preset-' + k; });
    presets.el.classList.add('em-presets');
    const excl = AM.ui.toggle({ id: 'em-exclude', label: 'Leave A, B, C out', checked: true, onChange: () => { if (st.an) setAnalogy(st.an.a, st.an.b, st.an.c); } });
    const viewSeg = AM.ui.segmented({
      id: 'em-view', label: 'Shadow', value: 0,
      options: [{ value: 1, label: 'Analogy plane' }, { value: 0, label: 'PCA' }],
      onChange: (v) => { st.introAt = 0; setView(v); },
    });
    const results = el('ol', { class: 'em-results', 'aria-live': 'polite' });
    const matchPreset = () => {
      const cur = [+selA.value, +selB.value, +selC.value];
      presets.select(PRESETS.findIndex((p) => p.every((w, j) => S.index.get(w) === cur[j])));
    };

    const panel = el('div', { class: 'panel em-panel' },
      el('div', { class: 'em-h' }, 'Vector arithmetic'),
      el('div', { class: 'em-eq' }, A.wrap, el('span', { class: 'em-op', 'aria-hidden': 'true' }, '−'), B.wrap, el('span', { class: 'em-op', 'aria-hidden': 'true' }, '+'), C.wrap),
      presets.el,
      el('div', { class: 'em-ctl-row' }, viewSeg.el, excl.el),
      el('div', { style: { display: 'grid', gap: '10px' } }, el('div', { class: 'em-h' }, 'Nearest to A − B + C'), results),
      el('p', { class: 'em-note', html: '<strong>Real models do this only roughly.</strong> The classic demo leaves A, B and C out of the search. Left in, the nearest word is often one of the inputs: in the original word2vec vectors, king − man + woman then returns <em>king</em>. Switch it off here and try two − one + three.' }),
    );

    function setAnalogy(a, b, c) {
      const res = S.analogy(a, b, c, excl.get());
      const same = st.an && st.an.a === a && st.an.b === b && st.an.c === c;
      st.an = { a, b, c, res, plane: same ? st.an.plane : S.plane(a, b, c) };
      if (!same) {
        retarget(!st.played || AM.reducedMotion);
        if (st.played) st.anT0 = AM.reducedMotion ? -99 : st.lastT + (st.view === 1 ? DUR * 0.45 : 0); // draw the arrows as the stars settle
      }
      selA.value = String(a); selB.value = String(b); selC.value = String(c);
      const best = res.top[0].cos;
      results.replaceChildren(...res.top.map((r, k) => el('li', { class: (k === 0 ? 'is-top ' : '') + ([a, b, c].includes(r.i) ? 'is-input' : '') },
        el('span', { class: 'em-rank' }, String(k + 1)),
        el('span', { class: 'em-word', style: { '--c': kit.color(r.i) } }, S.words[r.i].w),
        el('span', { class: 'em-cos' }, r.cos.toFixed(3)),
        el('span', { class: 'em-bar', style: { '--c': kit.color(r.i) } }, el('span', { style: { '--w': (100 * clamp(r.cos) / Math.max(best, 1e-6)).toFixed(1) + '%' } })))));
      const w = (i) => S.words[i].w;
      cv.canvas.setAttribute('aria-label', `A 3D constellation of 99 toy word embeddings. ${w(a)} minus ${w(b)} plus ${w(c)} lands nearest to ${w(res.top[0].i)} (cosine ${best.toFixed(2)}). Arrow keys turn the sky; Enter steps through A, B, C and the answer to show their nearest neighbours.`);
      draw(st.lastT);
    }

    // ---- interaction
    pointerControls(cv.canvas, cv, {
      onDrag: (dx, dy, type) => {
        st.cam.yaw += dx * 0.008; st.vel = dx * 0.008 * 60; st.idle = 0;
        if (type === 'mouse') st.cam.pitch = clamp(st.cam.pitch + dy * 0.005, -0.3, 1.1);
        st.hover = -1;
      },
      onTap: (x, y, type) => {
        const i = pick(st.pts, x, y, type === 'mouse' ? 18 : 26);
        st.pinned = i === st.pinned ? -1 : i; st.idle = 0;
        draw(st.lastT);
      },
      onHover: (x, y) => { const i = pick(st.pts, x, y, 18); if (i !== st.hover) { st.hover = i; cv.canvas.style.cursor = i >= 0 ? 'pointer' : ''; } },
      onLeave: () => { st.hover = -1; },
    });

    cv.canvas.tabIndex = 0;
    cv.canvas.addEventListener('keydown', (e) => {
      const cam = st.cam;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') cam.yaw += (e.key === 'ArrowLeft' ? -1 : 1) * 0.15;
      else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') cam.pitch = clamp(cam.pitch + (e.key === 'ArrowUp' ? 0.1 : -0.1), -0.3, 1.1);
      else if (e.key === 'Enter' || e.key === ' ') {
        const seq = [...new Set([st.an.a, st.an.b, st.an.c, st.an.res.top[0].i]), -1];
        st.pinned = seq[(seq.indexOf(st.pinned) + 1) % seq.length];
      } else if (e.key === 'Escape') st.pinned = -1;
      else return;
      e.preventDefault();
      st.idle = 0; st.vel = 0;
      draw(st.lastT);
    });

    // ---- drawing
    function draw(t) {
      if (!cv.w || !st.an) return;
      const g = cv.g, { w, h } = cv;
      const small = w < 600;
      cv.clear();
      const cam = st.cam;
      cam.cx = w * 0.48; cam.cy = h * 0.5;
      cam.R = Math.min(w * (small ? 0.41 : 0.38), h * 0.5) * (1 + (small ? 0.36 : 0.18) * st.vmix); // lean in on the plane (more on phones)
      const P = kit.camFrame(cam);
      const cur = current(t);
      const pos = cur.pos;
      const pts = pos.map((p) => P(p[0], p[1], p[2]));
      st.pts = pts;
      const vm = st.vmix;
      const focus = st.hover >= 0 ? st.hover : st.pinned;
      const nb = focus >= 0 ? S.nbrs[focus].slice(0, 5) : [];
      const nbSet = new Set(nb);
      const an = st.an;
      const win = an.res.top[0].i;
      const isKey = (i) => i === an.a || i === an.b || i === an.c || i === win;
      const spotDim = (c) => (st.spot < 0 || st.spot === c ? 1 : 0.18);
      const far = (i) => 1 - smooth(1.3, 1.8, Math.hypot(pos[i][0], pos[i][1], pos[i][2])); // stars flung far out by a shadow fade away
      const keyC = new Set([an.a, an.b, an.c, win].map((i) => S.words[i].c));
      const dim = (i) => {
        let d = spotDim(S.words[i].c) * far(i);
        if (focus >= 0 && i !== focus && !nbSet.has(i)) d *= 0.4;
        else if (focus < 0 && !isKey(i)) d *= 1 - vm * (keyC.has(S.words[i].c) ? 0.25 : 0.6); // plane view: let the parallelogram lead
        return d;
      };

      kit.drawDust(g, P, 1, t);
      kit.drawNebulae(g, P, pos, 1 - 0.55 * vm, cam.R, spotDim);
      kit.drawRing(g, P, 1 - vm);
      kit.drawGrid(g, P, vm);
      kit.drawEdges(g, pts, 1 - 0.4 * vm, dim);
      kit.drawStars(g, pts, t, (i) => Math.max(dim(i), isKey(i) && focus < 0 ? 0.9 : 0), (i) => (i === focus ? 1.7 : isKey(i) ? 1.3 : 1));

      if (vm > 0.02) {
        // axis notes for the analogy plane
        const wd = (i) => S.words[i].w;
        const up = P(0, 0.92, 0), across = P(0.92, 0, 0);
        DR.text(g, `↑ ${wd(an.a)} − ${wd(an.b)}`, up.x, Math.max(40, up.y - 6), { size: small ? 9 : 10, role: 'mono', color: AM.col.mist, align: 'center', alpha: vm });
        DR.text(g, `${wd(an.c)} − ${wd(an.b)} →`, Math.min(across.x, w - 8), across.y + 16, { size: small ? 9 : 10, role: 'mono', color: AM.col.mist, align: 'right', alpha: vm });
      }

      // --- the parallelogram. Both shadows are linear maps (plus an offset), and so is any blend
      //     of them, so shadow(A − B + C) = shadow(A) − shadow(B) + shadow(C) holds at every frame.
      const tA = st.played ? t - st.anT0 : 99;
      const anA = focus >= 0 ? 0.35 : 1;
      const pa = pts[an.a], pb = pts[an.b], pc = pts[an.c], pw = pts[win];
      const tp = P(cur.T[0], cur.T[1], cur.T[2]);
      const ph1 = smooth(0, 0.35, tA), ph2 = ease((tA - 0.3) / 0.6), ph3 = ease((tA - 0.95) / 0.6), ph4 = smooth(1.55, 2.05, tA);
      g.save();
      g.globalAlpha = anA;
      if (ph3 > 0.01) {
        // a faint silk sheet stretched over the parallelogram B → A → (A − B + C) → C
        g.save(); g.globalAlpha *= 0.07 * ph3; g.fillStyle = AM.dye.weld;
        g.beginPath(); g.moveTo(pb.x, pb.y); g.lineTo(pa.x, pa.y); g.lineTo(lerp(pa.x, tp.x, ph3), lerp(pa.y, tp.y, ph3)); g.lineTo(lerp(pc.x, tp.x, ph3), lerp(pc.y, tp.y, ph3)); g.lineTo(pc.x, pc.y); g.closePath(); g.fill(); g.restore();
        g.save(); g.setLineDash([3, 5]); g.strokeStyle = AM.rgba(AM.col.linenDim, 0.5 * ph3); g.lineWidth = 1;
        g.beginPath(); g.moveTo(pb.x, pb.y); g.lineTo(pc.x, pc.y); g.moveTo(pa.x, pa.y); g.lineTo(lerp(pa.x, tp.x, ph3), lerp(pa.y, tp.y, ph3)); g.stroke(); g.restore();
      }
      const glowArrow = (x1, y1, x2, y2, a) => {
        if (Math.hypot(x2 - x1, y2 - y1) < 4) return;
        g.save(); g.globalAlpha *= a;
        g.strokeStyle = AM.rgba(AM.dye.weld, 0.22); g.lineWidth = 7; g.lineCap = 'round';
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.restore();
        DR.arrow(g, x1, y1, x2, y2, { color: AM.dye.weld, width: 1.8, head: 10, alpha: a * g.globalAlpha });
      };
      if (ph2 > 0.01) glowArrow(pb.x, pb.y, lerp(pb.x, pa.x, ph2), lerp(pb.y, pa.y, ph2), 1);
      if (ph3 > 0.01) glowArrow(pc.x, pc.y, lerp(pc.x, tp.x, ph3), lerp(pc.y, tp.y, ph3), 0.9);
      if (ph3 > 0.99 && !AM.reducedMotion) {
        // beads of light travelling along both copies of the offset A − B
        const s1 = kit.sprite(AM.dye.weld);
        for (let b = 0; b < 3; b++) {
          const u = (t * 0.32 + b / 3) % 1;
          g.save(); g.globalAlpha *= Math.sin(u * Math.PI) * 0.9;
          g.drawImage(s1, lerp(pb.x, pa.x, u) - 9, lerp(pb.y, pa.y, u) - 9, 18, 18);
          g.drawImage(s1, lerp(pc.x, tp.x, u) - 9, lerp(pc.y, tp.y, u) - 9, 18, 18);
          g.restore();
        }
      }
      if (ph1 > 0.01) {
        [[pa, 'A'], [pb, 'B'], [pc, 'C']].forEach(([p, s]) => {
          g.save(); g.globalAlpha *= ph1; g.strokeStyle = AM.dye.weld; g.lineWidth = 1.2;
          g.beginPath(); g.arc(p.x, p.y, 8.5, 0, TAU); g.stroke(); g.restore();
          DR.text(g, s, p.x - 11, p.y - 10, { size: 10, role: 'mono', color: AM.dye.weld, align: 'right', alpha: ph1 * g.globalAlpha, weight: 500 });
        });
      }
      if (ph4 > 0.01) {
        const pulse = 1 + 0.15 * Math.sin(t * 3);
        g.save(); g.globalAlpha *= ph4; g.strokeStyle = AM.col.linen; g.lineWidth = 1.3; g.setLineDash([2, 3]);
        g.beginPath(); g.arc(tp.x, tp.y, 7 * pulse, 0, TAU); g.stroke(); g.restore();
        DR.thread(g, tp.x, tp.y, pw.x, pw.y, { color: AM.col.linen, width: 1.1, alpha: ph4 * 0.8 * g.globalAlpha, bend: 0.25 });
        const burst = smooth(1.55, 2.6, tA);
        if (burst < 1) { g.save(); g.globalAlpha *= (1 - burst) * 0.8; g.strokeStyle = AM.dye.weld; g.lineWidth = 1.5; g.beginPath(); g.arc(pw.x, pw.y, 8 + 44 * burst, 0, TAU); g.stroke(); g.restore(); }
        g.save(); g.globalAlpha *= ph4; g.strokeStyle = AM.dye.weld; g.lineWidth = 1.6; g.beginPath(); g.arc(pw.x, pw.y, 10.5, 0, TAU); g.stroke(); g.restore();
      }
      g.restore();

      // --- hover / tap: nearest neighbours with silk threads
      if (focus >= 0) kit.drawNeighbourThreads(g, pts[focus], nb.map((j) => ({ p: pts[j], w: clamp((S.cosM[focus][j] - 0.4) / 0.55) })), t, 1);

      // --- labels
      const items = [];
      const fsz = small ? 10.5 : 11.5;
      pts.forEach((p, i) => {
        const d = dim(i);
        const isF = i === focus, isN = nbSet.has(i), isW = i === win && ph4 > 0.3, isIn = i === an.a || i === an.b || i === an.c;
        let text = S.words[i].w, color = AM.mix(AM.col.linenDim, kit.color(i), 0.3), size = fsz, weight = 500, prio = -p.z, force = false;
        let alpha = kit.depthA(p.z) * 0.9 * d;
        let dx = 7, alt = false;
        if (isIn && focus < 0) { color = AM.col.linen; weight = 600; prio = 60; dx = 14; alpha = Math.max(alpha, 0.95 * anA * far(i)); alt = true; force = true; }
        if (isW) { text = `${text}  ${an.res.top[0].cos.toFixed(2)}`; color = AM.dye.weld; weight = 600; size = fsz + 2; prio = 90; dx = 15; force = focus < 0; alt = force; alpha = Math.max(alpha, ph4 * anA); }
        if (isN) { text = `${S.words[i].w}  ${S.cosM[focus][i].toFixed(2)}`; color = AM.col.linen; weight = 600; prio = 70; dx = 7; alpha = 1; force = true; alt = true; }
        if (isF) { color = AM.dye.weld; weight = 600; size = fsz + 3; prio = 100; dx = 9; force = true; alpha = 1; }
        items.push({ text, x: p.x, y: p.y, dx, size, weight, color, alpha, prio, force, alt });
      });
      if (ph4 > 0.01) items.push({ text: 'A − B + C', x: tp.x, y: tp.y, dx: 12, size: 9.5, role: 'mono', weight: 400, color: AM.col.linenDim, alpha: ph4 * anA, prio: 80 });
      const obstacles = focus >= 0 ? [] : [pa, pb, pc, pw].map((p) => [p.x - 11, p.y - 11, 21, 22]);
      kit.drawLabels(g, items, obstacles, w, h);
      kit.fadeEdges(g, w, h, 1, 30);

      const hint = focus >= 0 ? `nearest to “${S.words[focus].w}” · cosine in 64-d` : (small ? 'drag to turn · tap a star' : 'drag or arrow keys to turn · hover or tap a star');
      DR.text(g, hint, 12, 18, { size: small ? 9 : 10, role: 'mono', color: AM.col.mist });
      const note = vm > 0.5 ? (small ? 'shadow: analogy plane' : 'shadow: analogy plane · depth = top PCA direction') : `shadow: PCA, ${Math.round(100 * S.pca.top3)}% of the spread`;
      DR.text(g, note, w - 12, h - 12, { size: small ? 9 : 10, role: 'mono', color: AM.col.mist, align: 'right', alpha: 0.85 });
    }

    cv.onResize(() => draw(st.lastT));
    watchVisible(cv.wrap, (v) => {
      st.visible = v;
      if (v && !st.played) { st.played = true; st.pendingIntro = true; } // scheduled from the loop, where the clock is fresh
    });

    ctx.loop((t, dt) => {
      st.lastT = t;
      if (!st.visible) return;
      if (st.pendingIntro) { // first view: the parallelogram in PCA, then lay it flat
        st.pendingIntro = false;
        if (AM.reducedMotion) setView(1);
        else { st.anT0 = t + 0.3; st.introAt = t + 2.9; }
      }
      if (st.introAt && t >= st.introAt) { st.introAt = 0; setView(1); st.anT0 = t + DUR * 0.45; }
      st.idle += dt;
      st.vmix = AM.reducedMotion ? st.view : damp(st.vmix, st.view, 2.2, dt);
      const cam = st.cam;
      if (Math.abs(st.vel) > 0.001 && st.idle > 0.05) { cam.yaw += st.vel * dt; st.vel *= Math.exp(-3 * dt); } // fling inertia
      if (!AM.reducedMotion && st.idle > 2.5 && st.hover < 0) {
        const ease2 = smooth(2.5, 4, st.idle);
        if (st.view === 0) { cam.yaw += dt * 0.06 * ease2; cam.pitch = damp(cam.pitch, 0.4, 1, dt * ease2); }
        else { cam.yaw = damp(cam.yaw, st.yawBase + 0.3 * Math.sin(t * 0.3), 1.2, dt * ease2); cam.pitch = damp(cam.pitch, 0.12, 1.2, dt * ease2); } // face the plane, sway gently
      }
      draw(t);
    });

    const [a0, b0, c0] = PRESETS[0].map((w) => S.index.get(w));
    setAnalogy(a0, b0, c0);

    return el('div', { class: 'em-explorer' }, sky, panel, skyCap);
  }

  // ======================================================================
  // 6. The chapter
  // ======================================================================

  AM.chapter({
    id: ID,
    num: 2,
    kicker: 'Embeddings',
    title: 'Constellations of <em>Meaning</em>',
    lede: 'Each token ID is swapped for a long list of numbers: a point in a space where words used in similar ways sit close together.',
    where: 'embed',
    buildSpace, // exposed for offline checks (tools can verify the analogies)

    mount(root, ctx) {
      ctx.header();
      AM.css(CSS);
      const { el } = ctx;
      const S = buildSpace();
      const kit = makeKit(S);

      // dynamic spans inside the step text (updated when a token chip is picked)
      const dyn = { word: [], id: [], nb: [], nbcos: [] };
      const span = (key, cls, txt) => { const n = el('span', { class: cls }, txt); dyn[key].push(n); return n; };
      const pc = S.pca;
      const pcaPct = Math.round(100 * pc.top3);
      /** cos(a − b, c − d), computed in 64-d, formatted. */
      const offCos = (a, b, c, d) => {
        const v = (x, y) => Float64Array.from(S.E[S.index.get(x)], (e, j) => e - S.E[S.index.get(y)][j]);
        const u = v(a, b), w2 = v(c, d);
        return fmt(dot(u, w2) / (norm(u) * norm(w2)));
      };

      const step = (label, title, ...kids) => el('div', { class: 'step' }, el('span', { class: 'step-label' }, label), el('h3', {}, title), ...kids);
      // The real tiny model's width, read from its config once its weights are decoded (if it ships with the page).
      const liveWidth = el('span', {}, 'The tiny live model on this page uses a few dozen.');
      try {
        if (AM.model && AM.model.ready) {
          AM.model.ready.then(() => {
            const d = AM.model.get('tinyworld').config.d_model;
            liveWidth.textContent = d === D ? `The real tiny model running on this page also uses ${d}; this table is a separate, hand-built toy.` : `The real tiny model running on this page uses ${d}.`;
          }).catch(() => {});
        }
      } catch (_) { /* no live model on this page: keep the generic wording */ }
      const P = (html) => el('p', { html });

      const steps = [
        step('01 · Lookup', 'One row per token',
          P('The tokenizer handed us ID numbers. The model swaps each ID for a vector read from a learned table, the <span class="term">embedding matrix</span> <span class="math">E</span>: one row for every token in the vocabulary, <span class="math">d_model</span> columns. Ours is ' + S.N + ' × ' + D + '; blue cells are positive, red negative.'),
          el('p', {}, 'To embed ', span('word', 'em-step-tag', ''), ' (token ', span('id', 'em-step-tag', ''), '), take that row: ',
            el('span', { class: 'math' }, 'x = E[', span('id', '', ''), ']'), '. That is the whole operation. In the math it is often written as a one-hot vector times ', el('span', { class: 'math' }, 'E'), ', which picks out the same row.'),
          el('p', {}, 'Real tables are far bigger. GPT-2 small has 50,257 rows of 768 numbers, 38.6 million in all. GPT-3’s rows hold 12,288. ', liveWidth)),
        step('02 · Points', 'Every row is a point in space',
          P('Two numbers place a dot on a map; 64 numbers place a point in a 64-dimensional space. Watch the table unravel: each row becomes a star.'),
          P('Before training the table holds small random numbers, so the stars are scattered. The colours mark word groups for your eyes only. The model never sees them.')),
        step('03 · Training', 'Training gathers the constellations',
          P('Each time a token appears in the training text, the gradient nudges its row a little. Tokens used in similar contexts get similar nudges, so their points drift together: <em>cat</em> toward <em>dog</em>, <em>paris</em> toward <em>rome</em>.'),
          P('Nobody labels the groups; they come from usage alone. <q>You shall know a word by the company it keeps</q> (J. R. Firth, 1957).')),
        step('04 · Similarity', 'Close means similar',
          P('Closeness is measured with the <span class="term">dot product</span> <span class="math">a · b = Σ aᵢ bᵢ</span>. Divide by both lengths to get <span class="term">cosine similarity</span>, the cosine of the angle between the vectors: 1 for the same direction, 0 for perpendicular, −1 for opposite.'),
          el('span', { class: 'math block' }, 'cos(a, b) = a · b ⁄ (‖a‖ ‖b‖)'),
          el('p', {}, 'The threads join ', span('word', 'em-step-tag', ''), ' to its five nearest neighbours, scored in all 64 dimensions; the closest is ', span('nb', 'em-step-tag', ''), ' at ', span('nbcos', 'em-step-tag', ''), '. Pick another token under the picture.')),
        step('05 · Shadows', 'We only ever see shadows',
          P(`Nobody can picture 64 dimensions, so this view is a shadow. <span class="term">PCA</span> (principal component analysis) finds the directions along which the points spread most. We project onto the top three, which keep ${pcaPct}% of the spread; the strip shows all 64.`),
          P(`Points that look close in a shadow can be far apart in full, so every score on this page is computed in ${D} dimensions.`)),
      ];

      const prose = el('div', { class: 'ch-prose' }, steps);
      const stageBox = el('div', { class: 'ch-stage' });
      const split = el('div', { class: 'ch-split' }, stageBox, prose);
      stageBox.appendChild(mountStage(ctx, kit, steps, dyn));

      const explorerIntro = el('div', { class: 'prose' },
        el('h3', {}, 'Arithmetic with meaning'),
        P(`Directions in this space can carry meaning. The step from <em>man</em> to <em>king</em> points roughly the same way as the step from <em>woman</em> to <em>queen</em>: in this toy space the two steps have cosine similarity ${offCos('king', 'man', 'queen', 'woman')}, while an unrelated step, <em>red</em> to <em>blue</em>, scores ${offCos('king', 'man', 'blue', 'red')}. So <span class="math">king − man + woman</span> lands near <em>queen</em>.`),
        P('Pick three words. In the analogy-plane shadow the parallelogram lies flat, and the panel lists the five words nearest the landing point, by cosine similarity in all 64 dimensions.'));
      const explorer = AM.ui.figure({ title: 'Constellation explorer', badge: 'toy' },
        mountExplorer(ctx, kit, `Toy embeddings: ${S.N} words built from hand-designed features plus noise, then spun into ${D} dimensions by a random rotation. Two shadows of the same points: <strong>PCA</strong> keeps the most spread overall; the <strong>analogy plane</strong> is aimed along A − B and C − B, so the parallelogram keeps its true shape while every other star is flattened onto it. Every score is cosine similarity in ${D}-d. Drag to turn the sky, hover or tap a star for its nearest neighbours, tap a group to spotlight it. With the keyboard, arrow keys turn the sky and Enter steps through A, B, C and the answer.`));
      const wide = el('div', { class: 'ch-wide', style: { display: 'grid', gap: 'var(--space-6)' } }, explorerIntro, explorer);

      // Single coordinates are not labels: said in two sentences instead of a second figure.
      const dirs = el('div', { class: 'prose' },
        P(`No single coordinate means <em>royal</em>. The royal direction is spread thinly across all ${D}, and researchers hunt for directions like it in real models with tools such as linear probes and sparse autoencoders.`));

      const callout = el('div', { class: 'callout' },
        el('span', { class: 'callout-label' }, 'Key idea'),
        P('An embedding is one row of a learned table, and training shapes the whole table so that geometry mirrors usage. Words used alike point alike, and some directions line up with ideas such as gender or tense.'),
        P('The lookup ignores context: <em>bank</em> gets the same row in “river bank” and “bank account”. Mixing in context is the job of the layers that follow.'));

      root.appendChild(el('div', { class: 'ch-body' }, split, wide, dirs, callout));
    },
  });
})();
