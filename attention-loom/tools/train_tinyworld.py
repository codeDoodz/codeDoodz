#!/usr/bin/env python3
"""Train the tinyworld word-level transformer with numpy (no torch).

The architecture mirrors js/model/transformer.js exactly (pre-LN GPT, learned
positions, fused QKV, tanh-GELU, untied unembedding with bias), and the parameter
names/shapes are the JS ones, so the exported weights drop straight into the page.

  python3 tools/train_tinyworld.py --data DIR [--steps 12000] [--gradcheck-only]

DIR is produced by `node tools/data-tinyworld.mjs --out DIR`. Writes
js/model/weights-tinyworld.js and tools/fixtures/tinyworld-parity.json.
"""
import argparse, base64, json, math, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS_LN = 1e-5
FIXTURES = os.path.join(ROOT, 'tools/fixtures')  # overridable with --fixtures-dir
DESCRIPTION = ('Word-level toy English, no BOS token. Each sequence is 1-3 short units from a small grammar '
               '(tools/data-tinyworld.mjs) whose next word needs attention: pronoun and possessive agreement with an '
               'antecedent several words back, subject-verb number agreement across a prepositional phrase with an '
               'attractor noun, colour binding ("the red ball and the blue box . the box is blue"), name copying '
               '("alice gave bob a cup . bob thanked alice"), a parrot repeating a phrase, and memorised facts '
               '(capitals both directions, animal sounds, colours of sky/grass/snow/sun/apple/pig). '
               'Test sentences come from a hash-held-out 10% of all possible templated units.')
GELU_C = math.sqrt(2.0 / math.pi)


# ----------------------------------------------------------------------------- params
def param_specs(cfg):
    """Same names, shapes and init as paramSpecs() in transformer.js."""
    d, f, V, L = cfg['d_model'], cfg['d_ff'], cfg['vocab_size'], cfg['n_layer']
    std = cfg.get('init_std', 0.02)
    pstd = std / math.sqrt(2 * L)
    s = [('wte', (V, d), 'normal', std, True), ('wpe', (cfg['n_ctx'], d), 'normal', std, True)]
    for l in range(L):
        p = f'h.{l}.'
        s += [(p + 'ln1.g', (1, d), 'ones', 0, True), (p + 'ln1.b', (1, d), 'zeros', 0, True),
              (p + 'attn.wqkv', (d, 3 * d), 'normal', std, False), (p + 'attn.bqkv', (1, 3 * d), 'zeros', 0, True),
              (p + 'attn.wo', (d, d), 'normal', pstd, False), (p + 'attn.bo', (1, d), 'zeros', 0, True),
              (p + 'ln2.g', (1, d), 'ones', 0, True), (p + 'ln2.b', (1, d), 'zeros', 0, True),
              (p + 'mlp.wfc', (d, f), 'normal', std, False), (p + 'mlp.bfc', (1, f), 'zeros', 0, True),
              (p + 'mlp.wproj', (f, d), 'normal', pstd, False), (p + 'mlp.bproj', (1, d), 'zeros', 0, True)]
    s += [('lnf.g', (1, d), 'ones', 0, True), ('lnf.b', (1, d), 'zeros', 0, True),
          ('wout', (d, V), 'normal', std, False), ('bout', (1, V), 'zeros', 0, True)]
    return s


def init_params(cfg, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    P, nodecay = {}, set()
    for name, shape, kind, std, nd in param_specs(cfg):
        if kind == 'normal':
            P[name] = (rng.standard_normal(shape) * std).astype(dtype)
        elif kind == 'ones':
            P[name] = np.ones(shape, dtype)
        else:
            P[name] = np.zeros(shape, dtype)
        if nd:
            nodecay.add(name)
    if cfg.get('pos_init') == 'sin':
        P['wpe'] = sinusoid_init(cfg['n_ctx'], cfg['d_model'], cfg.get('init_std', 0.02)).astype(dtype)
    return P, nodecay


def sinusoid_init(n_ctx, d, std):
    """Starting point for the LEARNED position table: Vaswani sinusoids scaled to RMS = std.
    Shifts by k positions are then (initially) a fixed rotation, so 'look at the previous
    token' heads are easy to learn. Same formula as paramSpecs/init in transformer.js."""
    pos = np.arange(n_ctx)[:, None]
    i = np.arange(d)[None]
    ang = pos / np.power(10000.0, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(ang), np.cos(ang)) * std * math.sqrt(2)


# ----------------------------------------------------------------------------- layers
def ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + EPS_LN)
    xhat = (x - mu) * rstd
    return xhat * g[0] + b[0], (xhat, rstd, g)


def ln_bwd(dy, cache):
    xhat, rstd, g = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0, keepdims=True)
    db = dy.reshape(-1, xhat.shape[-1]).sum(0, keepdims=True)
    dxh = dy * g[0]
    dx = rstd * (dxh - dxh.mean(-1, keepdims=True) - xhat * (dxh * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def gelu_fwd(x):
    t = np.tanh(GELU_C * (x + 0.044715 * (x * x * x)))
    return 0.5 * x * (1 + t), (x, t)


def gelu_bwd(dy, cache):
    x, t = cache
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * x * x))


def forward(P, cfg, ids, capture=False):
    """ids: int array [B, T]. Returns logits [B, T, V] and a cache for backward."""
    B, T = ids.shape
    H, d = cfg['n_head'], cfg['d_model']
    dh = d // H
    scale = 1.0 / math.sqrt(dh)
    mask = np.triu(np.ones((T, T), bool), 1)  # True above the diagonal = future keys
    x = P['wte'][ids] + P['wpe'][:T][None]
    caches = []
    for l in range(cfg['n_layer']):
        p = f'h.{l}.'
        x_in = x
        h1, c_ln1 = ln_fwd(x, P[p + 'ln1.g'], P[p + 'ln1.b'])
        qkv = h1 @ P[p + 'attn.wqkv'] + P[p + 'attn.bqkv'][0]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) * scale
        s = np.where(mask, -np.inf, s)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        att = e / e.sum(-1, keepdims=True)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        x = x + o @ P[p + 'attn.wo'] + P[p + 'attn.bo'][0]
        x_mid = x
        h2, c_ln2 = ln_fwd(x, P[p + 'ln2.g'], P[p + 'ln2.b'])
        pre = h2 @ P[p + 'mlp.wfc'] + P[p + 'mlp.bfc'][0]
        act, c_gelu = gelu_fwd(pre)
        x = x + act @ P[p + 'mlp.wproj'] + P[p + 'mlp.bproj'][0]
        caches.append(dict(x_in=x_in, h1=h1, c_ln1=c_ln1, q=q, k=k, v=v, att=att, o=o, x_mid=x_mid,
                           h2=h2, c_ln2=c_ln2, act=act, c_gelu=c_gelu))
    hf, c_lnf = ln_fwd(x, P['lnf.g'], P['lnf.b'])
    logits = hf @ P['wout'] + P['bout'][0]
    return logits, dict(ids=ids, layers=caches, hf=hf, c_lnf=c_lnf, x_final=x)


def loss_and_grad(P, cfg, ids, targets, mask, want_grad=True):
    """Mean masked cross-entropy and its gradient for every parameter."""
    logits, cache = forward(P, cfg, ids)
    B, T, V = logits.shape
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - m)
    z = e.sum(-1, keepdims=True)
    probs = e / z
    logp_t = (logits - m - np.log(z))[np.arange(B)[:, None], np.arange(T)[None], targets]
    wsum = mask.sum()
    loss = -(logp_t * mask).sum() / wsum
    if not want_grad:
        return loss, probs, None
    G = {}
    dlog = probs.copy()
    dlog[np.arange(B)[:, None], np.arange(T)[None], targets] -= 1
    dlog *= (mask / wsum)[..., None]
    d, H = cfg['d_model'], cfg['n_head']
    dh = d // H
    scale = 1.0 / math.sqrt(dh)
    flat = lambda a: a.reshape(-1, a.shape[-1])
    G['wout'] = flat(cache['hf']).T @ flat(dlog)
    G['bout'] = flat(dlog).sum(0, keepdims=True)
    dhf = dlog @ P['wout'].T
    dx, G['lnf.g'], G['lnf.b'] = ln_bwd(dhf, cache['c_lnf'])
    for l in reversed(range(cfg['n_layer'])):
        p, c = f'h.{l}.', cache['layers'][l]
        # MLP branch: x = x_mid + gelu(h2 Wfc + bfc) Wproj + bproj
        G[p + 'mlp.wproj'] = flat(c['act']).T @ flat(dx)
        G[p + 'mlp.bproj'] = flat(dx).sum(0, keepdims=True)
        dpre = gelu_bwd(dx @ P[p + 'mlp.wproj'].T, c['c_gelu'])
        G[p + 'mlp.wfc'] = flat(c['h2']).T @ flat(dpre)
        G[p + 'mlp.bfc'] = flat(dpre).sum(0, keepdims=True)
        dh2 = dpre @ P[p + 'mlp.wfc'].T
        dln, G[p + 'ln2.g'], G[p + 'ln2.b'] = ln_bwd(dh2, c['c_ln2'])
        dx = dx + dln
        # attention branch: x_mid = x_in + attn(h1) Wo + bo
        G[p + 'attn.wo'] = flat(c['o']).T @ flat(dx)
        G[p + 'attn.bo'] = flat(dx).sum(0, keepdims=True)
        do = (dx @ P[p + 'attn.wo'].T).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        att, q, k, v = c['att'], c['q'], c['k'], c['v']
        dv = att.transpose(0, 1, 3, 2) @ do
        datt = do @ v.transpose(0, 1, 3, 2)
        ds = att * (datt - (datt * att).sum(-1, keepdims=True)) * scale  # softmax backward (+ scale)
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        merge = lambda a: a.transpose(0, 2, 1, 3).reshape(B, T, d)
        dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
        G[p + 'attn.wqkv'] = flat(c['h1']).T @ flat(dqkv)
        G[p + 'attn.bqkv'] = flat(dqkv).sum(0, keepdims=True)
        dh1 = dqkv @ P[p + 'attn.wqkv'].T
        dln, G[p + 'ln1.g'], G[p + 'ln1.b'] = ln_bwd(dh1, c['c_ln1'])
        dx = dx + dln
    # embeddings
    G['wte'] = np.zeros_like(P['wte'])
    np.add.at(G['wte'], cache['ids'].reshape(-1), flat(dx))
    G['wpe'] = np.zeros_like(P['wpe'])
    G['wpe'][:T] = dx.sum(0)
    return loss, probs, G


# ----------------------------------------------------------------------------- gradcheck
def gradcheck():
    cfg = dict(n_layer=2, n_head=2, d_model=8, d_ff=32, n_ctx=6, vocab_size=7, init_std=0.3)
    P, _ = init_params(cfg, 3, np.float64)
    rng = np.random.default_rng(0)
    for k in P:
        if 'ln' in k or k.startswith('b') or '.b' in k:
            P[k] = P[k] + 0.3 * rng.standard_normal(P[k].shape)
    ids = rng.integers(0, 7, (2, 5))
    tg = rng.integers(0, 7, (2, 5))
    mask = (np.arange(5)[None] >= 2).astype(np.float64).repeat(2, 0)
    _, _, G = loss_and_grad(P, cfg, ids, tg, mask)
    worst = 0
    for k, w in P.items():
        idx = rng.choice(w.size, min(w.size, 25), replace=False)
        num, ana = [], []
        for i in idx:
            o = w.flat[i]
            w.flat[i] = o + 1e-6; lp = loss_and_grad(P, cfg, ids, tg, mask, False)[0]
            w.flat[i] = o - 1e-6; lm = loss_and_grad(P, cfg, ids, tg, mask, False)[0]
            w.flat[i] = o
            num.append((lp - lm) / 2e-6); ana.append(G[k].flat[i])
        num, ana = np.array(num), np.array(ana)
        rel = np.abs(num - ana).max() / max(1e-8, np.abs(num).max(), np.abs(ana).max())
        worst = max(worst, rel)
        if rel > 1e-5:
            print(f'  gradcheck FAIL {k}: {rel:.2e}')
    print(f'numpy gradcheck: max rel err {worst:.2e}')
    return worst < 1e-5


# ----------------------------------------------------------------------------- eval
def evaluate(P, cfg, test, cands_map, vocab):
    """Per-type accuracy at critical tokens. Returns dict type -> stats."""
    wid = {w: i for i, w in enumerate(vocab)}
    stats = {}
    for chunk in range(0, len(test), 256):
        batch = test[chunk:chunk + 256]
        T = max(len(s['ids']) for s in batch) - 1
        ids = np.zeros((len(batch), T), np.int64)
        for b, s in enumerate(batch):
            ids[b, :len(s['ids']) - 1] = s['ids'][:-1]
        logits, _ = forward(P, cfg, ids)
        for b, s in enumerate(batch):
            for c in s['crit']:
                row = logits[b, c['pos'] - 1]
                tgt = wid[c['target']]
                cand = [wid[w] for w in cands_map[c['type']]]
                pr = np.exp(row - row.max()); pr /= pr.sum()
                st = stats.setdefault(c['type'], dict(n=0, top1=0, cand=0, p=0.0))
                st['n'] += 1
                st['top1'] += int(row.argmax() == tgt)
                st['cand'] += int(cand[int(np.argmax(row[cand]))] == tgt)
                st['p'] += float(pr[tgt])
    out = {}
    for k, st in stats.items():
        out[k] = dict(n=st['n'], top1=st['top1'] / st['n'], acc=st['cand'] / st['n'], p_correct=st['p'] / st['n'])
    return out


# ----------------------------------------------------------------------------- export
def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()).decode('ascii')


def export_js(P, cfg, vocab, meta, path):
    tensors = {name: dict(shape=list(shape), b64=b64(P[name])) for name, shape, *_ in param_specs(cfg)}
    obj = dict(config={k: cfg[k] for k in ('n_layer', 'n_head', 'd_model', 'd_ff', 'n_ctx', 'vocab_size')},
               vocab=vocab, meta=meta, tensors=tensors)
    with open(path, 'w') as f:
        f.write('/* The Attention Loom — trained weights for the "tinyworld" word-level model.\n'
                ' * Generated by tools/train_tinyworld.py — do not edit by hand.\n'
                ' * float32 little-endian, base64. */\n')
        f.write('(window.AM_WEIGHTS = window.AM_WEIGHTS || {}).tinyworld = ')
        f.write(json.dumps(obj, separators=(',', ':')))
        f.write(';\n')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=False)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--min-lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n-layer', type=int, default=3)
    ap.add_argument('--n-head', type=int, default=4)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--out', default=os.path.join(ROOT, 'js/model/weights-tinyworld.js'))
    ap.add_argument('--fixtures-dir', default=None)
    ap.add_argument('--pos-init', default='sin', choices=['sin', 'normal'])
    ap.add_argument('--gradcheck-only', action='store_true')
    ap.add_argument('--fixtures-from', help='weights-*.js to re-create tools/fixtures/* from (with --data)')
    a = ap.parse_args()
    global FIXTURES
    if a.fixtures_dir:
        FIXTURES = a.fixtures_dir

    if a.fixtures_from:  # re-create fixtures for an already exported weights file
        src = open(a.fixtures_from).read()
        start = src.index('{', src.index('.tinyworld = '))
        obj = json.loads(src[start:src.rstrip().rindex(';')])
        cfg = dict(obj['config'])
        P = {k: np.frombuffer(base64.b64decode(v['b64']), '<f4').reshape(v['shape']).astype(np.float32)
             for k, v in obj['tensors'].items()}
        test = json.load(open(os.path.join(a.data, 'test.json')))
        cands = json.load(open(os.path.join(a.data, 'vocab.json')))['candidates']
        for k, v in sorted(evaluate(P, cfg, test, cands, obj['vocab']).items()):
            print(f"  {k:11s} n={v['n']:5d}  acc {v['acc']:.4f}  top1 {v['top1']:.4f}")
        write_fixtures(P, cfg, test)
        print('fixtures written')
        return
    if not gradcheck():
        sys.exit('gradcheck failed')
    if a.gradcheck_only:
        return

    meta_in = json.load(open(os.path.join(a.data, 'vocab.json')))
    vocab, seq_len = meta_in['vocab'], meta_in['seq_len']
    train = np.fromfile(os.path.join(a.data, 'train.u8'), np.uint8).reshape(-1, seq_len).astype(np.int64)
    test = json.load(open(os.path.join(a.data, 'test.json')))
    cfg = dict(n_layer=a.n_layer, n_head=a.n_head, d_model=a.d_model, d_ff=4 * a.d_model,
               n_ctx=seq_len - 1, vocab_size=len(vocab), init_std=0.02, pos_init=a.pos_init)
    P, nodecay = init_params(cfg, a.seed)
    nparams = sum(w.size for w in P.values())
    print(f'config {cfg}  params {nparams}  train seqs {len(train)}  test seqs {len(test)}', flush=True)

    M = {k: np.zeros_like(w) for k, w in P.items()}
    Vv = {k: np.zeros_like(w) for k, w in P.items()}
    b1, b2, eps = 0.9, 0.99, 1e-8
    rng = np.random.default_rng(a.seed)
    lengths = (train != 0).sum(1)
    t0, ema, curve = time.time(), None, []

    def batches():
        """Length-bucketed batches: shuffle, sort chunks of 64 batches by length, shuffle the
        batches. Each batch is trimmed to its longest sequence (causal, so padding is inert)."""
        while True:
            order = rng.permutation(len(train))
            for c in range(0, len(order) - a.batch * 64 + 1, a.batch * 64):
                chunk = order[c:c + a.batch * 64]
                chunk = chunk[np.argsort(lengths[chunk], kind='stable')]
                bs = [chunk[i:i + a.batch] for i in range(0, len(chunk), a.batch)]
                for j in rng.permutation(len(bs)):
                    yield bs[j]
    feed = batches()

    def lr_at(i):
        if i < a.warmup:
            return a.lr * (i + 1) / a.warmup
        q = min(1.0, (i - a.warmup) / max(1, a.steps - a.warmup))
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * q))

    for step in range(1, a.steps + 1):
        idx = next(feed)
        seq = train[idx, :int(lengths[idx].max())]
        ids, tg = seq[:, :-1], seq[:, 1:]
        mask = (tg != 0).astype(np.float32)
        loss, _, G = loss_and_grad(P, cfg, ids, tg, mask)
        # global-norm clip at 1.0, then AdamW with decoupled weight decay
        gn = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in G.values()))
        sc = min(1.0, 1.0 / (gn + 1e-6))
        lr = lr_at(step - 1)
        c1, c2 = 1 / (1 - b1 ** step), 1 / (1 - b2 ** step)
        for k in P:
            g = G[k].astype(np.float32) * sc
            M[k] = b1 * M[k] + (1 - b1) * g
            Vv[k] = b2 * Vv[k] + (1 - b2) * g * g
            decay = 0.0 if k in nodecay else lr * a.wd
            P[k] -= decay * P[k] + lr * (M[k] * c1) / (np.sqrt(Vv[k] * c2) + eps)
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if step % 100 == 0:
            curve.append([step, round(float(ema), 4)])
            print(f'step {step:5d}  loss {ema:.4f}  lr {lr:.2e}  |g| {gn:.2f}  {time.time() - t0:.0f}s', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(P, cfg, test[:1500], meta_in['candidates'], vocab)
            print('  eval ' + '  '.join(f"{k} {v['acc']:.3f}" for k, v in sorted(ev.items())), flush=True)
            # checkpoint (overwritten by the final export) so a long run can be inspected or cut short
            export_js(P, cfg, vocab, dict(name='tinyworld', kind='word', partial=True, step=step,
                                          test_accuracy={k: round(v['acc'], 4) for k, v in ev.items()}), a.out)

    # final evaluation on the whole held-out set + mean held-out loss
    ev = evaluate(P, cfg, test, meta_in['candidates'], vocab)
    tl, tn = 0.0, 0
    for chunk in range(0, len(test), 256):
        batch = test[chunk:chunk + 256]
        ids = np.zeros((len(batch), seq_len), np.int64)
        for b, s in enumerate(batch):
            ids[b, :len(s['ids'])] = s['ids']
        mask = (ids[:, 1:] != 0).astype(np.float64)
        l, _, _ = loss_and_grad(P, cfg, ids[:, :-1], ids[:, 1:], mask, False)
        tl += float(l) * mask.sum(); tn += mask.sum()
    print('final held-out per-type accuracy:')
    for k, v in sorted(ev.items()):
        print(f"  {k:11s} n={v['n']:5d}  acc {v['acc']:.4f}  top1 {v['top1']:.4f}  P(correct) {v['p_correct']:.4f}")
    print(f'held-out loss/token {tl / tn:.4f}   train time {time.time() - t0:.0f}s')

    meta = dict(
        name='tinyworld', kind='word',
        description=DESCRIPTION,
        train=dict(steps=a.steps, batch=a.batch, lr=a.lr, min_lr=a.min_lr, warmup=a.warmup, weight_decay=a.wd,
                   seed=a.seed, sequences=int(len(train)), optimizer='AdamW(0.9,0.99), clip 1.0, cosine',
                   trainer='numpy (tools/train_tinyworld.py)', seconds=round(time.time() - t0)),
        params=int(nparams), final_train_loss=round(float(ema), 4), heldout_loss=round(tl / tn, 4),
        loss_curve=curve,
        test_accuracy={k: dict(n=v['n'], acc=round(v['acc'], 4), top1=round(v['top1'], 4),
                               p_correct=round(v['p_correct'], 4)) for k, v in sorted(ev.items())},
        accuracy_note='acc = the correct word beats every other candidate of its type (e.g. he/she/it/they) at '
                      'the critical slot of held-out sentences; top1 = argmax over the whole vocabulary.',
    )
    export_js(P, cfg, vocab, meta, a.out)
    print('wrote', a.out, os.path.getsize(a.out), 'bytes')

    write_fixtures(P, cfg, test)


def write_fixtures(P, cfg, test):
    """Parity fixture for the JS forward pass (logits + last-layer attention of a few held-out
    sequences, computed in float64) and the held-out test set used by tools/test-model.mjs."""
    fx = []
    for s in test[:4]:
        ids = np.array([s['ids'][:cfg['n_ctx']]], np.int64)
        logits, cache = forward({k: v.astype(np.float64) for k, v in P.items()}, cfg, ids)
        fx.append(dict(ids=s['ids'][:cfg['n_ctx']], logits=logits[0].astype(np.float32).tolist(),
                       attn_last=cache['layers'][-1]['att'][0].astype(np.float32).tolist()))
    os.makedirs(FIXTURES, exist_ok=True)
    with open(os.path.join(FIXTURES, 'tinyworld-parity.json'), 'w') as f:
        json.dump(fx, f)
    with open(os.path.join(FIXTURES, 'tinyworld-test.json'), 'w') as f:
        json.dump(test, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
