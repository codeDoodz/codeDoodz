# The real tiny transformers: how they were built and trained

Three decoder-only transformers ship with the page. All of them run in the browser
through `AM.model` (CONTRACT §6). Every number below was produced by the scripts in
`tools/` with fixed seeds and can be reproduced.

| model | task | layers × heads | d_model | d_ff | n_ctx | vocab | params | held-out accuracy |
|---|---|---|---|---|---|---|---|---|
| `tinyworld` | next word in toy English | 3 × 4 | 64 | 256 | 32 | 138 words | 169,930 | 98.6–100% on every required dependency type (table below) |
| `reverse` | `38152907>70925183` | 1 × 1 | 24 | 96 | 16 | 11 chars | 8,195 | 100.00% exact (10,000 inputs) |
| `sort` | `73519273>12335779` | 2 × 2 | 32 | 128 | 16 | 11 chars | 26,699 | 99.88% exact (10,000 inputs) |

Files:

```
js/model/tensor.js        AMTensorLib: reverse-mode autograd over Float32Array (Float64Array for gradient checks)
js/model/transformer.js   AMTransformerLib(T): the model, the trainer, the introspecting forward pass
js/model/weights-*.js     trained weights (float32, base64) + meta (data description, accuracies, curves)
js/model/api.js           AM.model (get/list/ready, encode/decode/run/topk/sample/generate),
                          AMTaskLib (reverse/sort data), createTrainerWorker (the Lab trainer)
tools/data-tinyworld.mjs  the tinyworld grammar + held-out split
tools/train_tinyworld.py  numpy trainer for tinyworld (same architecture, hand-written backward)
tools/train.mjs           trains reverse/sort in JS (and drives the tinyworld pipeline)
tools/gradcheck.mjs       finite-difference checks of every op and the full model
tools/test-model.mjs      the whole test suite (exit code 1 on failure)
tools/analyze-tinyworld.mjs  which heads do what, logit lens per dependency type
tools/fixtures/           tinyworld held-out set + numpy logits for the parity test
```

## 1. Architecture (all three models)

GPT-2 style, pre-LayerNorm, learned absolute positions, untied unembedding:

```
x₀      = wte[token] + wpe[position]
for l in 0 … n_layer−1:
  h     = LN1(x_l)
  q,k,v = h·W_qkv + b_qkv                    (split into n_head heads of d_head = d_model / n_head)
  A     = softmax(q·kᵀ / √d_head + causal mask)   (query i sees keys j ≤ i)
  x_mid = x_l + (A·v concatenated over heads)·W_o + b_o
  x_l+1 = x_mid + GELU(LN2(x_mid)·W_fc + b_fc)·W_proj + b_proj      (GELU = tanh approximation)
logits  = LN_f(x_L)·W_out + b_out
```

LayerNorm uses ε = 1e-5 and biased variance. Weights are stored `[in, out]`; the fused
`h.{l}.attn.wqkv` is `[d, 3d]` with columns `[Q | K | V]`, head `h` owning columns
`h·d_head … (h+1)·d_head` of each third. Parameter names: `wte, wpe, h.{l}.ln1.{g,b},
h.{l}.attn.{wqkv,bqkv,wo,bo}, h.{l}.ln2.{g,b}, h.{l}.mlp.{wfc,bfc,wproj,bproj}, lnf.{g,b}, wout, bout`.

Initialisation: N(0, 0.02) for matrices and embeddings, residual-branch outputs (`wo`,
`wproj`) scaled by 1/√(2·n_layer), LN gains 1, biases 0. tinyworld starts its learned
position table from sinusoids scaled to RMS 0.02 (`pos_init: 'sin'`) — still an ordinary
learned parameter, it just begins in a shape where "look k tokens back" is easy to express.

Training (both trainers): masked mean cross-entropy, AdamW (β = 0.9, 0.99, decoupled
weight decay on matrices only), global gradient-norm clip 1.0, linear warmup then cosine
decay.

## 2. tinyworld — the word-level model

### Grammar

`tools/data-tinyworld.mjs` generates sequences of 1–3 short *units*, at most 33 tokens
(32 input positions). There is no BOS token: position 0 is the first word. Each unit type
plants one dependency that cannot be resolved from the previous word alone. The word
the model is scored on (the *critical* token) is in brackets.

| type | example | what it takes |
|---|---|---|
| pronoun | `the queen opened the door because [she] was cold .` <br> `alice and bob walked to the park and closed the box because [they] were tired .` | find the subject several words back; know queen/alice → she, dog → it, children / "x and y" → they |
| possessive | `the dog walked to the river with [its] ball .` · `the children loved [their] kite .` | same, his/her/its/their |
| agreement | `the keys near the old door [are] gold .` · `the key near the old doors [is] gold .` | number of the head noun, ignoring the attractor noun in the prepositional phrase (half the time it has the opposite number) |
| binding | `the red ball and the blue box . the box is [blue] .` <br> `the pink door , the old yellow ring , the white box and the green ball . the children walked to the garden . the ball is [green] .` | which colour went with the asked-about noun; 2–4 objects, optional size word and distractor sentence so position alone never works |
| copy | `alice gave bob a cup . bob thanked [alice] .` · `tom met lucy at the castle . lucy waved at [tom] .` | copy the other name |
| parrot | `the witch said green cup yellow cup . the parrot said [green] [cup] [yellow] [cup] .` | repeat a phrase word for word |
| capital | `the capital of japan is [tokyo] .` · `tokyo is the capital of [japan] .` | memorised (8 countries, both directions) |
| sound | `the duck says [quack] .` | memorised (8 animals) |
| colorfact | `the sky is [blue] .` (sky, grass, snow, sun, apple, pig) | memorised; same syntax as binding questions, answered from memory instead of context |
| recall *(probe)* | `the red ball and the blue box . alice found the blue [box] .` | colour → noun. **Not learned** (see caveats) |
| walk *(filler)* | `the princess walked to the market .` | — |

Each binding unit asks one or two questions about different objects; each question is
a colour question (`the box is [blue]`, scored as *binding*) or, half the time, an object
question (`alice found the blue [box]`, scored as the *recall* probe).

Unit mix: pronoun 14%, possessive 8%, agreement 12%, binding 22%, copy 10%, parrot 10%,
capital 7%, sound 6%, colorfact 7%, walk 4%. Vocabulary: 138 tokens (`<pad>`, `<unk>`,
`.`, `,` and 134 lowercase words; the full list is `m.vocab`).

**Held-out split.** Every templated unit's text is hashed (FNV-1a); units with
`hash % 10 == 0` never appear in training. The test set (4,000 sequences,
`tools/fixtures/tinyworld-test.json`) is built only from those held-out units, so every
pronoun/agreement/binding/copy sentence scored there is new to the model. Fact units and
the parrot/walk units are shared by both splits (facts have to be memorised).

### Training

numpy trainer (`tools/train_tinyworld.py`, no torch), same maths as `transformer.js`.
600,000 generated sequences, length-bucketed batches of 128, 12,000 steps, lr 3e-3
(warmup 300, cosine to 1e-4), weight decay 0.05, seed 1. 1,469 s on 4 shared CPUs.

Loss (EMA, nats/token): 3.75 @100 → 1.18 @1k → 1.06 @3k → 1.02 @5.5k → 1.00 @12k.
Held-out loss 1.046/token. Most of that floor is genuine randomness in the grammar
(which colour, which name, which place); the critical tokens themselves are predicted
almost perfectly.

The binding skill arrived late and suddenly: binding accuracy sat at 40–60% (picking one
of the colours in the sentence, at random) until about step 5,000, then jumped to 88% by
step 5,500 and 97% by 7,000 — a small "phase change". Copy and parrot were solved by
step ~3,500; pronouns, agreement and facts by step ~1,500.

### Accuracy at the critical token (held-out set)

`acc` = the correct word beats every other word of its kind (e.g. he/she/it/they);
`top-1` = argmax over the whole vocabulary; `P` = mean probability of the correct word.

| type | n | acc | top-1 | mean P(correct) |
|---|---|---|---|---|
| pronoun | 1,018 | 100.0% | 100.0% | 1.000 |
| possessive | 585 | 100.0% | 100.0% | 1.000 |
| agreement | 899 | 100.0% | 100.0% | 1.000 |
| binding | 1,004 | 98.6% | 98.6% | 0.983 |
| copy | 749 | 100.0% | 100.0% | 0.998 |
| parrot | 3,120 | 100.0% | 100.0% | 0.998 |
| capital | 469 | 100.0% | 100.0% | 0.998 |
| sound | 482 | 100.0% | 100.0% | 1.000 |
| colorfact | 525 | 100.0% | 100.0% | 1.000 |
| recall *(probe)* | 888 | 55.6% | 55.6% | 0.530 |

### What the heads do (measured by `node tools/analyze-tinyworld.mjs`)

Mean attention from the position that predicts the critical word to the token that holds
the answer, over the held-out set; and the logit lens (P(correct) decoded from the
residual stream entering layer 0, 1, 2 and after the last layer).

| type | strongest heads → evidence token | logit lens: embed → L0 → L1 → final |
|---|---|---|
| pronoun (`because` → subject) | **L0H3 0.95**, L2H2 0.31 | 0.14 → 0.97 → 0.97 → 1.00 |
| possessive (`with` → subject) | **L0H3 0.83**, L0H2 0.31 | 0.19 → 0.89 → 0.89 → 1.00 |
| agreement (attractor noun → head noun) | **L1H2 0.87**, L2H2 0.24 | 0.09 → 0.49 → 1.00 → 1.00 |
| binding (`is` → the right colour word) | **L1H3 0.75**, L1H0 0.26 | 0.01 → 0.04 → 0.97 → 0.98 |
| copy (`thanked` → the other name) | **L1H2 0.78**, L0H2 0.50 | 0.03 → 0.19 → 0.40 → 1.00 |
| parrot (→ the same word in the original phrase) | L1H3 0.50 | 0.05 → 0.46 → 0.95 → 1.00 |
| capital / sound / colorfact (→ the key noun) | L0H2 / L0H3 0.86–0.96 | the answer appears mostly in the last layer (MLP recall) |

Crisp single sentences for "LIVE MODEL" panels (all predicted at ~100%):

- `the queen opened the door because …` → **she**. L0H3 at `because` puts 0.99 of its attention on `queen`.
- `the children walked to the park and found the ball because …` → **they**.
- `the dog walked to the river with …` → **its**.
- `the keys near the old door …` → **are**; `the key near the old doors …` → **is**. L1H2 at `door` looks at `keys` (0.80).
- `the red ball and the blue box . the box is …` → **blue** (100%). L0H0 at `is` looks at the previous word `box` (0.83); L1H3 at `is` looks at `blue` (0.75). With `the green hat , the white door and the pink cup . the hat is …` → **green**, L1H3 looks at `green` (0.64).
- `alice gave bob a cup . bob thanked …` → **alice**. L0H2 and L1H2 at `thanked` both look at `alice` (0.76, 0.78).
- `the capital of japan is …` → **tokyo**; `tokyo is the capital of …` → **japan**; `the duck says …` → **quack**; `the sky is …` → **blue**.
- `the girl said red kite blue cup . the parrot said …` → **red**, then **kite**, …

A note on the binding head, useful for whoever animates it: attention is causal, so the
key at `blue` cannot know that `box` comes after it. The colour positions only carry what
came *before* them. The model still lands on the right colour, consistent with a rule
like "the latest colour word whose preceding words do not yet include the asked noun".
That is our reading of the measurements, not a proven circuit; label it as such.

Generation (temperature 0.7–1.0) produces grammatical, often charming toy English,
e.g. `alice gave bob a cup . bob thanked alice .`

## 3. reverse and sort — the Lab's algorithmic models

Format (fixed length, documented in `AMTaskLib`): 8 random digits, `>`, then the answer.
Tokens `'0'…'9'` = ids 0–9, `'>'` = 10. The model reads the first 16 tokens; the loss
mask covers only the 8 positions whose next token is an output digit (position 8, the
`>`, predicts the first output digit … position 15 predicts the last). Exact-sequence
accuracy with teacher forcing equals greedy-decoding accuracy (if every argmax is right,
greedy reproduces the whole answer).

| | reverse | sort |
|---|---|---|
| model | 1 layer, 1 head, d 24 | 2 layers, 2 heads, d 32 |
| training (Lab defaults) | batch 32, lr 3e-3 (warmup 30, cosine to 3e-4), 600 steps | batch 32, lr 3e-3, 3,000 steps |
| final loss | 0.0013 | 0.0002 |
| test (10,000 fresh inputs) | 100.00% exact | 99.88% exact, 99.99% per digit |
| JS speed, Node 22 (shared Xeon 2.3 GHz) | 18 ms/step | 54 ms/step |
| time to ≥ 90% exact (held-out 512) | **75 steps, 1.5 s** | **225 steps, 12.7 s** |
| time to ≥ 95% / 99% | 75 steps / 75 steps | 525 steps, 29 s / 1,000 steps, 55 s |

Both are well inside the "≥ 90% within ~60 s" target, and the shipped weights are
exactly what the Lab produces with seed 1 (the Lab's eval batch is 200 examples
instead of 512, so its curve is slightly noisier). Measured end to end:

- Node, `createTrainerWorker('sort')` main-thread fallback (in the test suite): ≥ 90%
  after ~220–250 steps, ~13 s of training.
- Headless Chromium opening the page from `file://`: the blob Worker starts fine
  (`ctl.inline === false`) and sort reached 90% at step 183 after 9.5 s of training,
  12.1 s wall clock including the progress snapshots. `run()` on tinyworld took 10.8 ms
  for 32 tokens with capture, 6.5 ms without (machine shared with other jobs).

What they learn (probe inputs):
- **reverse**: a perfect anti-diagonal. Output position 8 + j attends to input position
  7 − j with weight ≥ 0.98. Watching it crystallise is fast (≈ 75 steps), so the Lab may
  want `delayMs` (slow motion) or a lower `lr` for reverse.
- **sort**: L0H1 at each output position attends to the next larger digit present in the
  input (`>`→1, `1`→2, `3`→5, `5`→7, `7`→9 on `73519273`), the classic "smallest digit
  greater than the last one" pattern. Repeated digits are handled by the second layer;
  its patterns are softer.

## 4. Notes for chapter authors

- `AM.model.get(name)` works synchronously once the scripts are loaded (it decodes on
  first use); `AM.model.ready` resolves when all three are decoded.
- `encode(text)` lowercases, splits on whitespace and splits off `.` and `,`. Unknown
  words become `<unk>` and are listed in `.unknown`. There is **no BOS token**: tokens are
  exactly the words. For `reverse`/`sort`, `encode` keeps only digit and `>` characters.
- `run(ids, {capture:true})` gives every field in CONTRACT §6. Indexing:
  `attn[l][h][q][k]`, `resid[l][t]` for l = 0 … n_layer (l = n_layer is the final
  residual, before the final LayerNorm), `lens[l][t]` decodes `resid[l][t]` through the
  final LN + unembedding, so `lens[n_layer]` equals `probs`. Inputs longer than n_ctx are
  cut to the first 32 tokens and `r.truncated` is set. Timing: about 8–12 ms for 32 tokens
  with capture in Node on a busy machine; a few ms without capture.
- `generate(ids, opts)` returns the **whole** sequence (prompt + new ids) and stops after
  emitting `stopAt` (default `'.'` for tinyworld). `sample()` never emits `<pad>`/`<unk>`;
  `temperature: 0` is greedy; `topP` is nucleus sampling after temperature.
- `m.meta` holds the accuracies (`meta.test_accuracy`), data description, loss curve and,
  for tinyworld, `meta.analysis` (top heads per dependency type, logit-lens curves,
  previous-token scores) for honest captions.
- Plausible-looking but **unsupported** claims to avoid: the model does not do colour →
  noun lookups (`alice found the blue …` is a coin flip among the listed objects); it has
  no strong general previous-token head (best is ~0.33 on average, L0H3); and the parrot
  copy may lean on positional offsets as much as on content (the phrase is always the
  same distance back).

### Lab trainer: `AM.model.createTrainerWorker(task, opts)`

```js
const ctl = AM.model.createTrainerWorker('sort', {seed: 1, reportMs: 150});
ctl.on(msg => { … });   // 'ready' | 'progress' | 'paused' | 'done' | 'error'
ctl.start(); ctl.pause(); ctl.step(10); ctl.reset({seed: 2});
ctl.set({lr: 1e-3, delayMs: 200, maxSteps: 5000}); ctl.terminate();
```

- `ready`: `{task, config, params, train, probe:{ids, text}}`.
- `progress`: `{step, loss (EMA), acc (exact-sequence on a fixed 200-example held-out
  batch, refreshed every `evalMs` = 500 ms of training), tokenAcc, attn, probePred,
  probeTarget, lr, elapsed (s spent training), wall (s since start, incl. overhead), stepsPerSec}`.
  `attn[layer][head]` is a `Float32Array(16·16)`, row q, column k (row-major, causal,
  rows sum to 1) for the task's fixed probe (`38152907>…` / `73519273>…`).
- Options: `seed`, `lr`, `batch`, `config` (override model sizes), `train` (override
  schedule), `maxSteps` (default = schedule length: 600 / 3,000), `reportMs` (default
  150), `chunkMs` (work per slice, default 50), `delayMs` (slow motion: one step per
  tick), `evalSize` (200), `evalMs` (500), `inline: true` (force main thread).
- The worker is built from a Blob of `AMTensorLib.toString() + AMTransformerLib +
  AMTaskLib + AMTrainerMain`. If Workers are unavailable, or the worker errors before it
  says `ready` (some browsers restrict blob workers on `file://`), the same code runs on
  the main thread in ~50 ms slices with the same messages; `ctl.inline` tells you which.
- `AM.model.tasks` (an `AMTaskLib()` instance) has `example(task, rng, digits?)`,
  `batch(task, rng, B)`, `score`, `format`, and `TASKS[task]` (configs, probe, vocab).

## 5. Verification

`node tools/test-model.mjs` (about 1–2 minutes) checks:

1. Finite-difference gradient checks in float64 for matmul, linear, add, addBias, GELU,
   LayerNorm, fused causal attention, embedding, masked cross-entropy and the full
   2-layer model (every parameter tensor): max relative error ≈ 1e-8 (threshold 1e-5).
   The numpy trainer runs its own float64 gradient check (≈ 3e-9) before every training run.
2. Parity: the shipped JS forward reproduces the numpy trainer's logits to 6e-6
   (logit scale ≈ 28) and its attention to 3e-7.
3. The AM.model contract: shapes, causal masks, rows summing to 1, logit lens, residual
   bookkeeping (`resid[1] = residMid[0] + MLP write`), sampling, generation.
4. Accuracies for all three models (thresholds 95%; the tinyworld `recall` probe is
   reported but not enforced).
5. The trainer: the exact Worker source evaluated in an empty sandbox learns reverse
   (anti-diagonal attention), and the main-thread Lab path gets sort to ≥ 90%.
6. Timing.

## 6. Reproducing

```
node tools/train.mjs reverse        # ~10 s  → js/model/weights-reverse.js
node tools/train.mjs sort           # ~3 min → js/model/weights-sort.js
node tools/train.mjs tinyworld      # corpus → $TMPDIR/attention-loom-data, numpy training (~25 min),
                                    # then analyze-tinyworld --write-meta (head summary → meta.analysis)
node tools/test-model.mjs
node tools/analyze-tinyworld.mjs --md   # the head / logit-lens tables above
```

## 7. Caveats

- **Colour → noun recall is not learned.** The model learned noun → colour binding (98.6%)
  but the reverse direction (`… alice found the blue [box]`) stayed at 56%: it picks one
  of the listed objects. Both directions were in training. We left the probe in the data
  and the report because it is a real, honest limit of this small model.
- Binding took a phase change to appear (step ~5,000 of 12,000). Re-training with a
  different seed or mix may land it later or not at all; check `meta.test_accuracy`.
- Facts are memorised from shared (not held-out) sentences; their test accuracy measures
  recall in new contexts, not generalisation.
- The vocabulary is 138 tokens, slightly above the ~130 first planned (colour facts and
  the parrot added a few words).
- Node timings above were taken while other jobs shared the 4 CPUs; a laptop browser is
  typically similar or faster.
