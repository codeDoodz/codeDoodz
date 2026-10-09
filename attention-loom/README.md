# The Attention Loom

An illustrated, interactive journey inside a transformer, the machine behind
every modern chatbot. It's one long scroll of animated, poke-able figures,
woven around a single metaphor: **a transformer is a loom that weaves context
into words.**

Much of what you see is computed live by a **real tiny transformer**. It was
trained from scratch in this folder, ships its weights with the page and runs
every forward pass in your browser. In chapter 11 you can even train a
transformer yourself and watch its attention crystallise out of noise in
about ten seconds.

## Viewing it

No build step and no server. Either:

- open `index.html` directly in a browser (works from `file://`), or
- open the single self-contained file `dist/attention-loom.html` (made by
  `node tools/build.mjs`).

Everything is vanilla JavaScript + Canvas. The only external request is for
Google Fonts. There's no tracking and no network calls.

## The journey

| | Chapter | What you can do |
|---|---|---|
| · | **Prologue** | A real sentence woven on a loom from the live model's 12 attention heads |
| 01 | **Shattering Text**: tokenization | Shatter a sentence into tokens; drive a real BPE tokenizer merge by merge |
| 02 | **Constellations of Meaning**: embeddings | Fly through a 3D word constellation; do vector arithmetic |
| 03 | **The Clockwork of Order**: positional encoding | Spin sinusoid clock hands; prove RoPE only sees relative offsets |
| 04 | **Threads of Attention**: self-attention | Watch queries, keys and values compute attention step by step; play with the mask and √d_k |
| 05 | **The Prism**: multi-head attention | Split a token into 4 heads; explore all 12 real heads; mute heads and see what breaks |
| 06 | **The Memory Vaults**: feed-forward network | See 256 neurons fire for a fact; ablate them and watch the fact vanish |
| 07 | **The River Thread**: residual stream & LayerNorm | Remove the residual connections; step through LayerNorm; see the gradient highway |
| 08 | **The Tower of Layers**: depth & the logit lens | Watch the model's guess sharpen layer by layer |
| 09 | **Rolling the Dice**: unembedding & sampling | Spin a sampling wheel on real logits with temperature, top-k and top-p; let the model write |
| 10 | **Learning by Falling**: training | Drag a cross-entropy playground; roll beads down a woven loss landscape |
| 11 | **Watch a Mind Form**: live training lab | Train a real transformer from random weights in a Web Worker |
| 12 | **From Toy to Titan**: scale & recap | Zoom from 170K parameters to frontier scale; an annotated architecture poster |

Every figure carries an honesty label: **Live model** (computed now by the
real network), **Toy numbers** (small hand-built values that show the
mechanism) or **Illustration** (artistic depiction).

## The live models

| model | task | shape | params | held-out accuracy |
|---|---|---|---|---|
| `tinyworld` | next word in a toy-English world | 3 layers × 4 heads, d_model 64 | 169,930 | 99–100% on pronouns, agreement, binding, copying and facts |
| `reverse` | `38152907>70925183` | 1 × 1, d 24 | 8,195 | 100% exact |
| `sort` | `73519273>12335779` | 2 × 2, d 32 | 26,699 | 99.8% exact |

The architecture is GPT-2 style: pre-LayerNorm, learned positions, GELU MLPs
and a causal mask. The gradients are checked against finite differences, and
an independent numpy reference matches the browser forward pass to within
6×10⁻⁶. The details of how the models were trained, verified and probed
(head atlas, ablations, logit-lens stories, MLP fact memory) are in
[`docs/model-training.md`](docs/model-training.md) and
[`docs/model-notes.md`](docs/model-notes.md).

## Layout

```
index.html               page shell: one <section data-chapter> per chapter
css/base.css             design tokens (indigo cloth, linen text, natural dyes) + shared components
js/core/                 AM runtime: chapter lifecycle, canvases, widgets, drawing primitives
js/model/                autograd library, transformer, trained weights, AM.model API, notes data
js/chapters/             one file per chapter
tools/                   training, tests, screenshots, bundler
docs/CONTRACT.md         the rules every chapter follows
```

## Tools

```
node tools/test-model.mjs                  gradient checks, parity, accuracy, timing (exit 1 on failure)
node tools/shoot.mjs --chapter attention   screenshots (desktop + phone) and console-error check
node tools/build.mjs [--fragment]          bundle into dist/ as one HTML file
node tools/train.mjs reverse|sort          retrain an algorithmic model with the Lab's own JS trainer
node tools/train.mjs tinyworld             regenerate the corpus and retrain tinyworld (numpy, ~25 min)
```

## Further reading

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (2021)
- nostalgebraist, [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (2020)
