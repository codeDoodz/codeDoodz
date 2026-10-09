# What the tiny model actually learned

Reference for chapter authors building **LIVE MODEL** panels. Every number here was computed from the shipped weights through `AM.model` (the same `run()` a chapter calls), loaded in Node with the page's own scripts. Nothing is from memory or from the training logs unless it says so. How the models were built and trained is in `docs/model-training.md`; this file is about what is inside them.

Machine-readable copies of the curated examples, head names, logit-lens stories, fact traces, PCA coordinates and sampling tables are in `js/model/notes-data.js` (`window.AM_NOTES`, loaded after `api.js`; see §9).

Conventions: `L1H3` = layer 1, head 3 (zero-based). "q" = query position, "k" = key position, both zero-based token indices. There is **no BOS token**: index 0 is the first word. "acc" = the correct word beats every other word of its kind (pronouns, colours, …); "P" = probability of the correct word.

## 1. Model card: `tinyworld`

| | |
|---|---|
| architecture | GPT-2 style decoder, pre-LayerNorm, learned positions, untied unembedding, GELU (tanh) |
| config | n_layer 3, n_head 4, d_model 64, d_head 16, d_ff 256, n_ctx 32, vocab 138 |
| parameters | 169,930 |
| training | numpy, 600,000 generated sequences, 12,000 steps × batch 128, AdamW lr 0.003 (cosine to 0.0001), wd 0.05, seed 1 |
| held-out loss | 1.046 nats/token (most of it is real randomness in the grammar: which name, which place, which colour) |
| speed | `run()` with capture ≈ 6–11 ms for 32 tokens in Node (model-training.md §3) |

### Vocabulary (all 138 tokens, in id order)

```
0:<pad> 1:<unk> 2:. 3:, 4:the 5:a 6:and 7:because 8:with 9:of 10:is 11:are 12:was 13:were
14:says 15:to 16:at 17:near 18:under 19:behind 20:walked 21:loved 22:gave 23:thanked 24:met
25:waved 26:helped 27:capital 28:said 29:parrot 30:he 31:she 32:it 33:they 34:his 35:her
36:its 37:their 38:king 39:queen 40:boy 41:girl 42:prince 43:princess 44:wizard 45:witch
46:alice 47:emma 48:lucy 49:rose 50:bob 51:tom 52:sam 53:jack 54:dog 55:cat 56:duck 57:owl
58:pig 59:cow 60:lion 61:sheep 62:children 63:dogs 64:cats 65:key 66:keys 67:box 68:boxes
69:cup 70:cups 71:door 72:doors 73:book 74:books 75:hat 76:hats 77:ring 78:ball 79:kite
80:crown 81:opened 82:closed 83:found 84:lost 85:dropped 86:cold 87:tired 88:happy 89:sad
90:hungry 91:sleepy 92:old 93:small 94:gold 95:heavy 96:broken 97:shiny 98:red 99:blue
100:green 101:yellow 102:white 103:pink 104:park 105:river 106:market 107:castle 108:garden
109:france 110:japan 111:italy 112:spain 113:egypt 114:peru 115:china 116:kenya 117:paris
118:tokyo 119:rome 120:madrid 121:cairo 122:lima 123:beijing 124:nairobi 125:moo 126:woof
127:meow 128:quack 129:baa 130:hoot 131:oink 132:roar 133:sky 134:grass 135:snow 136:sun
137:apple
```

By kind (useful for colouring tokens): **punct** . , · **function** the a and because with of is are was were says to at near under behind capital said parrot · **verb** walked loved gave thanked met waved helped opened closed found lost dropped · **pronoun** he she it they · **possessive** his her its their · **person** king queen boy girl prince princess wizard witch · **name** alice emma lucy rose bob tom sam jack · **animal** dog cat duck owl pig cow lion sheep · **plural** children dogs cats · **object** key box cup door book hat ring ball kite crown · **objects** keys boxes cups doors books hats · **state** cold tired happy sad hungry sleepy · **adj** old small gold heavy broken shiny · **color** red blue green yellow white pink · **place** park river market castle garden · **country** france japan italy spain egypt peru china kenya · **city** paris tokyo rome madrid cairo lima beijing nairobi · **sound** moo woof meow quack baa hoot oink roar · **nature** sky grass snow sun apple. `sheep` only appears in `the sheep says baa`.

### Training grammar (tools/data-tinyworld.mjs)

Each sequence is 1–3 short units (≤ 33 tokens). Each unit plants one dependency; the bracketed word is where the model is scored.

| unit (share) | example | needs |
|---|---|---|
| pronoun (14%) | `the queen opened the door because [she] was cold .` | find the subject several words back; its gender/number |
| possessive (8%) | `the dog walked to the river with [its] ball .` | same, his/her/its/their |
| agreement (12%) | `the keys near the old door [are] gold .` | number of the head noun, ignoring the attractor noun |
| binding (22%) | `the red ball and the blue box . the box is [blue] .` | which colour went with the asked noun (2–4 objects, optional size word, optional distractor sentence) |
| copy (10%) | `alice gave bob a cup . bob thanked [alice] .` | copy the other name |
| parrot (10%) | `the witch said green cup yellow cup . the parrot said [green cup yellow cup] .` | repeat a phrase |
| capital (7%) | `the capital of japan is [tokyo] .` / `tokyo is the capital of [japan] .` | memorised, 8 countries |
| sound (6%) | `the duck says [quack] .` | memorised, 8 animals |
| colour fact (7%) | `the sky is [blue] .` (sky, grass, snow, sun, apple, pig) | memorised |
| walk (4%) | `the princess walked to the market .` | filler |
| recall probe | `… alice found the blue [box] .` (inside binding units) | colour → noun. **Not learned.** |

### Test accuracy (held-out set of 4,000 sequences, from `meta.test_accuracy`)

| type | n | acc | mean P(correct) |
|---|---|---|---|
| agreement | 899 | 100.0% | 1.000 |
| capital | 469 | 100.0% | 0.998 |
| colorfact | 525 | 100.0% | 1.000 |
| copy | 749 | 100.0% | 0.998 |
| parrot | 3120 | 100.0% | 0.998 |
| possessive | 585 | 100.0% | 1.000 |
| pronoun | 1018 | 100.0% | 1.000 |
| sound | 482 | 100.0% | 1.000 |
| binding | 1004 | 98.6% | 0.983 |
| recall *(probe, not learned)* | 888 | 55.6% | 0.530 |

Caveat carried over from model-training.md §2: pronoun / possessive / agreement test *contexts* also occur in training, so those 100% figures are in-distribution. Binding, copy and parrot are scored on contexts the model never saw. Never caption a pronoun or agreement example with "it has never seen this sentence".

The reverse and sort models are described in §8.

## 2. Head atlas (tinyworld)

Three kinds of evidence per head, all measured on the held-out set:

1. **Attention to the evidence token**: mean attention from the position that predicts the critical word to the token that holds the answer (first 1,500 held-out sequences).
2. **Targeted patterns**: mean attention for specific query/key situations over all 4,000 held-out sequences (e.g. from an attractor noun to its preposition).
3. **Ablation**: zero the head's slice of `W_o` (its whole output) and re-score the first 1,500 held-out sequences. This is causal evidence: attention weight alone can be high without mattering.

### Summary

| head | name | what it does | ablation effect |
|---|---|---|---|
| **L0H0** | the backstitch | reads one cue word back | agreement P(correct) 1.00 → 0.21 (acc 0.67), binding acc 0.98 → 0.46, parrot → 0.87 |
| **L0H1** | the loose weave | diffuse, unclear | parrot acc → 0.78, binding → 0.83; everything else unchanged |
| **L0H2** | the fact finder | reads the key noun of a fact | capital acc 1.00 → 0.10, sound → 0.54, colour facts → 0.76, parrot → 0.56, binding → 0.81 |
| **L0H3** | the antecedent finder | because/with → subject | pronoun acc 1.00 → 0.41, possessive → 0.53, copy → 0.61, binding → 0.59, colour facts → 0.68 |
| **L1H0** | the second colour reader | weak colour helper | no measurable effect (binding 0.98 → 0.98) |
| **L1H1** | the clause hopper | jumps to the previous clause | copy acc 1.00 → 0.93; nothing else moves |
| **L1H2** | the head-noun tracker | attractor → head noun; who gave | agreement acc 1.00 → 0.52 (chance); copy only → 0.98 (other heads also carry the name) |
| **L1H3** | the colour binder | `is` → the matching colour; parrot copy | binding acc 0.98 → 0.22 (below the 1/2 to 1/4 chance of picking among listed colours), parrot → 0.57 |
| **L2H0** | the idle shuttle | mostly resting, diffuse | no measurable effect |
| **L2H1** | the verb echo | echoes the earlier verb | no measurable effect |
| **L2H2** | the faint echo | weak copy of subject lookups | no measurable effect |
| **L2H3** | the idle shuttle II | mostly resting | no measurable effect |

### Per head

**L0H0: the backstitch.** A selective look-back head. From a noun inside a prepositional phrase it looks at the preposition (0.90 on 899 attractor nouns); from the question word `is` in `the box is` it looks at the asked noun right before it (0.86, n = 1066); from `said` in `the parrot said` it looks at `parrot` (0.90). At fact slots it also reads the key noun (0.66 on capital items, 0.71 on colour facts), but removing it does not hurt any fact, so L0H2/L0H3 carry those. Its average previous-token score over all positions is only 0.20, so it is not a general previous-token head.
*Ablation:* agreement P(correct) 1.00 → 0.21 (acc 0.67), binding acc 0.98 → 0.46, parrot → 0.87. *Shape:* previous-token 0.20, self 0.13, position 0 0.08, full stops 0.09, normalised entropy 0.57, mean top weight 0.56.

**L0H1: the loose weave.** Diffuse. Mean normalised entropy 0.73 (1 = uniform) and mean top weight 0.43, the softest of layer 0. Leans on the previous token (0.28) and nearby function words; at `to` it looks at `walked` and at a sound word it looks at `says`. No single dependency type above 0.48 (capital, sound ~0.4 to the fact noun).
*Ablation:* parrot acc → 0.78, binding → 0.83; everything else unchanged. *Shape:* previous-token 0.28, self 0.13, position 0 0.12, full stops 0.11, normalised entropy 0.73, mean top weight 0.43.

**L0H2: the fact finder.** At the slot where a memorised fact is due, it reads the key noun: country (0.85 on capital items), animal (0.78), sky/grass/... (0.80). In the copy unit it also reads the first name (0.51).
*Ablation:* capital acc 1.00 → 0.10, sound → 0.54, colour facts → 0.76, parrot → 0.56, binding → 0.81. *Shape:* previous-token 0.28, self 0.12, position 0 0.12, full stops 0.15, normalised entropy 0.60, mean top weight 0.52.

**L0H3: the antecedent finder.** From `because` it jumps to the subject noun (0.96 on 388 pronoun items), from `with`/`loved` likewise (0.84 on possessives). It also reads the animal in `the duck says` (0.94) and the noun in `the sky is` (0.96). Elsewhere it mostly attends to itself (pronouns 0.77, colour words 0.64); it has the highest previous-token average of any head, 0.32.
*Ablation:* pronoun acc 1.00 → 0.41, possessive → 0.53, copy → 0.61, binding → 0.59, colour facts → 0.68. *Shape:* previous-token 0.32, self 0.26, position 0 0.08, full stops 0.14, normalised entropy 0.43, mean top weight 0.67.

**L1H0: the second colour reader.** At a binding `is` it puts 0.54 of its attention on colour words, but only 0.26 on the right one; in several examples it locks onto the wrong colour (`the red ball and the blue box . the box is`: 0.69 on `red`). Otherwise it rests on position 0 (0.24).
*Ablation:* no measurable effect (binding 0.98 → 0.98). *Shape:* previous-token 0.12, self 0.11, position 0 0.22, full stops 0.05, normalised entropy 0.77, mean top weight 0.39.

**L1H1: the clause hopper.** Hops back to the start or verb of the previous clause. At `thanked` / `waved at` it looks at the earlier verb (`gave`, `met`, `helped`): 0.86 (n = 749). At a binding `is` it looks at the last full stop (0.38 on average, 0.91 in `the red ball and the blue box . the box is`).
*Ablation:* copy acc 1.00 → 0.93; nothing else moves. *Shape:* previous-token 0.14, self 0.10, position 0 0.26, full stops 0.07, normalised entropy 0.74, mean top weight 0.42.

**L1H2: the head-noun tracker.** From the attractor noun (`door` in `the keys near the old door`) it looks back past the preposition to the head noun: 0.87 on 358 agreement items. In the copy unit, from `thanked` it looks at the name to copy: 0.77. At a possessive word it attends to itself (0.60).
*Ablation:* agreement acc 1.00 → 0.52 (chance); copy only → 0.98 (other heads also carry the name). *Shape:* previous-token 0.13, self 0.14, position 0 0.26, full stops 0.06, normalised entropy 0.75, mean top weight 0.41.

**L1H3: the colour binder.** At `the box is` it lands on the colour that went with `box`: 0.74 on the right colour, 0.89 on colour words in total (n = 1066). It also does the parrot's copying, pointing at the word to repeat (0.50).
*Ablation:* binding acc 0.98 → 0.22 (below the 1/2 to 1/4 chance of picking among listed colours), parrot → 0.57. *Shape:* previous-token 0.12, self 0.11, position 0 0.23, full stops 0.07, normalised entropy 0.78, mean top weight 0.38.

**L2H0: the idle shuttle.** No dependency above 0.36 (sound). Rests on position 0 (0.22) and nearby function words.
*Ablation:* no measurable effect. *Shape:* previous-token 0.13, self 0.10, position 0 0.22, full stops 0.05, normalised entropy 0.68, mean top weight 0.45.

**L2H1: the verb echo.** Like L1H1 one layer later: at `thanked` / `waved at` it looks at the earlier verb (0.56); 0.94 in `alice gave bob a cup . bob thanked`. Heaviest position-0 sink of all heads (0.32).
*Ablation:* no measurable effect. *Shape:* previous-token 0.14, self 0.10, position 0 0.31, full stops 0.08, normalised entropy 0.57, mean top weight 0.54.

**L2H2: the faint echo.** Weak second look at the antecedent (0.31 on pronoun items), the head noun (0.23) and the copied name (0.31). Redundant with L0H3 / L1H2.
*Ablation:* no measurable effect. *Shape:* previous-token 0.10, self 0.12, position 0 0.26, full stops 0.09, normalised entropy 0.72, mean top weight 0.43.

**L2H3: the idle shuttle II.** Reads the animal for sounds (0.46) and, at a sound word, `says`; otherwise rests on position 0 (0.21) and itself.
*Ablation:* no measurable effect. *Shape:* previous-token 0.11, self 0.15, position 0 0.21, full stops 0.06, normalised entropy 0.65, mean top weight 0.48.

### Attention to the evidence token, every head

Mean attention from the predicting position to the token holding the answer. Evidence token: the subject (pronoun, possessive), the head noun (agreement), the correct colour word (binding), the earlier mention of the asked noun (binding→noun), the name to copy (copy), the word to repeat in the original phrase (parrot), the key noun (capital, sound, colorfact), the noun after the earlier colour (recall probe). Bold ≥ 0.5.

| type | n | L0H0 | L0H1 | L0H2 | L0H3 | L1H0 | L1H1 | L1H2 | L1H3 | L2H0 | L2H1 | L2H2 | L2H3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agreement | 358 | 0.01 | 0.04 | 0.01 | 0.20 | 0.14 | 0.03 | **0.87** | 0.15 | 0.04 | 0.05 | 0.23 | 0.21 |
| binding | 370 | 0.00 | 0.04 | 0.04 | 0.00 | 0.26 | 0.01 | 0.01 | **0.74** | 0.04 | 0.01 | 0.07 | 0.02 |
| binding→noun | 370 | 0.01 | 0.02 | 0.00 | 0.00 | 0.01 | 0.01 | 0.03 | 0.00 | 0.12 | 0.03 | 0.03 | 0.06 |
| capital | 178 | **0.66** | 0.48 | **0.85** | 0.08 | 0.10 | 0.15 | 0.13 | 0.10 | 0.15 | 0.24 | 0.13 | 0.08 |
| colorfact | 192 | **0.71** | 0.40 | **0.80** | **0.96** | 0.18 | 0.24 | 0.12 | 0.17 | 0.15 | 0.13 | 0.03 | 0.29 |
| copy | 295 | 0.01 | 0.13 | **0.51** | 0.03 | 0.14 | 0.02 | **0.77** | 0.36 | 0.21 | 0.18 | 0.31 | 0.28 |
| parrot | 1126 | 0.06 | 0.01 | 0.20 | 0.03 | 0.20 | 0.04 | 0.05 | 0.50 | 0.04 | 0.02 | 0.05 | 0.01 |
| possessive | 219 | 0.09 | 0.18 | 0.34 | **0.84** | 0.13 | 0.12 | 0.14 | 0.15 | 0.21 | 0.24 | 0.21 | 0.19 |
| pronoun | 388 | 0.07 | 0.08 | 0.20 | **0.96** | 0.11 | 0.09 | 0.11 | 0.20 | 0.09 | 0.15 | 0.31 | 0.12 |
| recall | 335 | 0.18 | 0.00 | 0.32 | 0.00 | 0.00 | 0.10 | 0.01 | 0.00 | 0.01 | 0.03 | 0.00 | 0.05 |
| sound | 170 | 0.24 | 0.40 | **0.78** | **0.94** | 0.12 | 0.10 | 0.20 | 0.20 | 0.36 | 0.31 | 0.33 | 0.46 |

### Ablation table

Accuracy after zeroing one head, a whole attention layer, or a whole MLP (first 1,500 held-out sequences). Bold = drop of 0.05 or more.

| ablated | pronoun | possessive | agreement | binding | copy | parrot | capital | sound | colorfact |
|---|---|---|---|---|---|---|---|---|---|
| nothing | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| L0H0 | 1.00 | 1.00 | **0.67** | **0.46** | 1.00 | **0.87** | 1.00 | 1.00 | 1.00 |
| L0H1 | 1.00 | 1.00 | 1.00 | **0.83** | 1.00 | **0.78** | 1.00 | 1.00 | 1.00 |
| L0H2 | 1.00 | 1.00 | 1.00 | **0.81** | **0.91** | **0.56** | **0.10** | **0.54** | **0.76** |
| L0H3 | **0.41** | **0.53** | 1.00 | **0.59** | **0.61** | **0.75** | 0.95 | **0.79** | **0.68** |
| L1H0 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| L1H1 | 1.00 | 1.00 | 1.00 | 0.98 | **0.93** | 1.00 | 1.00 | 1.00 | 1.00 |
| L1H2 | 1.00 | 1.00 | **0.52** | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| L1H3 | 1.00 | 1.00 | 1.00 | **0.22** | 1.00 | **0.57** | 1.00 | 1.00 | 1.00 |
| L2H0 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| L2H1 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| L2H2 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| L2H3 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| attn0 | **0.29** | **0.20** | **0.52** | **0.35** | **0.10** | **0.16** | **0.04** | **0.09** | **0.16** |
| attn1 | 1.00 | 1.00 | **0.56** | **0.12** | **0.76** | **0.60** | 1.00 | 1.00 | 1.00 |
| attn2 | 0.97 | 0.99 | 1.00 | 0.98 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |
| mlp0 | **0.32** | **0.31** | **0.76** | **0.33** | **0.29** | **0.36** | **0.20** | **0.09** | **0.39** |
| mlp1 | 1.00 | 1.00 | **0.87** | **0.93** | 1.00 | 0.97 | 1.00 | 1.00 | 0.99 |
| mlp2 | 1.00 | **0.93** | 1.00 | 0.98 | **0.72** | 1.00 | 1.00 | **0.76** | 1.00 |

### Things worth knowing before you draw heads

- **Layer 2's attention is nearly idle.** Zeroing all four layer-2 heads at once leaves every dependency type at 0.97 or better. The work is done by layer 0 (antecedents, fact keys) and layer 1 (head noun, colour binding, copying). Layer 2 still matters through its MLP (copy drops to 0.72, sound to 0.76 without it; see §5).
- **Position 0 is an attention sink.** With no BOS token, layers 1 and 2 park 0.21–0.32 of their attention on position 0 when it is `the` (n = 58,072 queries), and 0.15–0.29 when it is a name. Layer 0 heads only 0.07–0.13. In a full attention map the first column of layers 1–2 will glow; caption it as "resting on the first word", not as meaning.
- **No clean general previous-token head.** The best average is L0H3 at 0.32. L0H0 is a *selective* look-back (0.86–0.90 in the situations listed above).
- **MLP 0 is part of the embedding.** Zeroing MLP 0 wrecks everything (pronoun acc 0.32, capital 0.20), as is common in small models. Do not present "MLP 0 knows everything"; present it as the model's second embedding step.
- **Copy is redundant.** L1H2 puts 0.77 on the right name, but removing it only drops copy to 0.98; the name also arrives through L0H2 (0.51) and L1H3 (0.36). Removing all of layer 1's attention drops copy to 0.76. Show L1H2 as "the clearest of several heads that carry the name".
- **The binding head and causality.** L1H3 lands on the colour *before* the asked noun, and attention is causal, so the colour key cannot see the noun that follows it. So the colour's key can only carry what came *before* it. In layer 0, colour tokens put much of their attention on earlier object nouns (on average 0.51 of L0H2's and 0.40 of L0H0's attention, over all colour tokens in the held-out set), so each colour can carry "which objects were already listed before me". The behaviour fits the rule "pick the latest colour whose preceding words do not yet include the asked noun" (model-training.md §2 gives the same reading). That is our interpretation of the measurements, not a proven circuit; label it as such.

## 3. Curated example sentences

Every example below was run through `AM.model.get('tinyworld').run(ids, {capture:true})`; all tokens are in-vocabulary and the numbers reproduce exactly (float32, deterministic). "lock" = the key the named head attends to from the query, with its weight; ✓ means it is that head's single largest weight at that query. Predictions are the top 3 at the query position.

**pronoun-she** (pronoun)  
`the queen opened the door because`  
- L0H3 (the antecedent finder): q=5 `because` → k=1 `queen` **0.99** ✓
- next after `because` (q=5): she 100.0%, because 0.0%, they 0.0%
- L0H3 jumps from "because" straight back to the subject. Swap queen→king and the prediction flips to "he".

**pronoun-he** (pronoun)  
`the king opened the door because`  
- L0H3 (the antecedent finder): q=5 `because` → k=1 `king` **1.00** ✓
- next after `because` (q=5): he 100.0%, his 0.0%, she 0.0%
- Minimal pair of pronoun-she: one word changed, same head, same lock, opposite pronoun.

**pronoun-they-far** (pronoun)  
`the children walked to the park and found the ball because`  
- L0H3 (the antecedent finder): q=10 `because` → k=1 `children` **0.94** ✓
- next after `because` (q=10): they 100.0%, he 0.0%, she 0.0%
- Nine words between subject and pronoun. "park" and "ball" are closer nouns; L0H3 ignores them.

**possessive-their** (possessive)  
`alice and bob walked to the park with`  
- L0H3 (the antecedent finder): q=7 `with` → k=2 `bob` **0.53** ✓
- L0H3 (the antecedent finder): q=7 `with` → k=0 `alice` **0.33** (largest is `bob` 0.53)
- next after `with` (q=7): their 100.0%, they 0.0%, a 0.0%
- Two names joined by "and": L0H3 splits its attention over both names, and the model says "their".

**possessive-its** (possessive)  
`the dog walked to the river with`  
- L0H3 (the antecedent finder): q=6 `with` → k=1 `dog` **0.62** ✓
- next after `with` (q=6): its 100.0%, their 0.0%, his 0.0%
- Animal → "its".

**agree-are** (agreement)  
`the keys near the old door`  
- L1H2 (the head-noun tracker): q=5 `door` → k=1 `keys` **0.80** ✓
- L0H0 (the backstitch): q=5 `door` → k=2 `near` **0.90** ✓
- next after `door` (q=5): are 100.0%, is 0.0%, . 0.0%
- At the attractor noun "door", L1H2 looks back past it to the head noun "keys" → "are". L0H0 marks that we are inside a prepositional phrase (looks at "near").

**agree-is** (agreement)  
`the key near the old doors`  
- L1H2 (the head-noun tracker): q=5 `doors` → k=1 `key` **0.84** ✓
- L0H0 (the backstitch): q=5 `doors` → k=2 `near` **0.79** ✓
- next after `doors` (q=5): is 100.0%, walked 0.0%, lucy 0.0%
- Minimal pair: singular head, plural attractor → "is". After layer 0 the model still says "are"; layer 1 flips it (see logit lens).

**bind-blue** (binding)  
`the red ball and the blue box . the box is`  
- L1H3 (the colour binder): q=10 `is` → k=5 `blue` **0.75** ✓
- L0H0 (the backstitch): q=10 `is` → k=9 `box` **0.83** ✓
- L1H1 (the clause hopper): q=10 `is` → k=7 `.` **0.91** ✓
- next after `is` (q=10): blue 100.0%, red 0.0%, white 0.0%
- Three heads, three jobs at one query: L0H0 reads the asked noun "box", L1H1 hops to the sentence boundary, L1H3 lands on the matching colour.

**bind-red** (binding)  
`the blue ball and the red box . the box is`  
- L1H3 (the colour binder): q=10 `is` → k=5 `red` **0.86** ✓
- next after `is` (q=10): red 100.0%, green 0.0%, yellow 0.0%
- Swap the colours: L1H3 follows the colour, not the position.

**bind-three** (binding)  
`the green hat , the white door and the pink cup . the hat is`  
- L1H3 (the colour binder): q=14 `is` → k=1 `green` **0.64** ✓
- L0H0 (the backstitch): q=14 `is` → k=13 `hat` **0.92** ✓
- next after `is` (q=14): green 99.8%, pink 0.2%, white 0.0%
- Three objects; the asked object is the first one, so the answer is the earliest colour.

**bind-four-distractor** (binding)  
`the pink door , the old yellow ring , the white box and the green ball . the children walked to the garden . the ball is`  
- L1H3 (the colour binder): q=26 `is` → k=14 `green` **0.66** ✓
- next after `is` (q=26): green 98.2%, white 1.6%, yellow 0.1%
- Four objects, a size word and a distractor sentence. Still lands on "green" (98%).

**copy-alice** (copy)  
`alice gave bob a cup . bob thanked`  
- L1H2 (the head-noun tracker): q=7 `thanked` → k=0 `alice` **0.78** ✓
- L1H1 (the clause hopper): q=7 `thanked` → k=1 `gave` **0.80** ✓
- next after `thanked` (q=7): alice 100.0%, rose 0.0%, emma 0.0%
- L1H2 fetches the other name; L1H1 hops to the earlier verb "gave".

**copy-midsequence** (copy)  
`the sky is blue . alice gave bob a cup . bob thanked`  
- L1H2 (the head-noun tracker): q=12 `thanked` → k=5 `alice` **0.78** ✓
- next after `thanked` (q=12): alice 99.9%, kenya 0.0%, cairo 0.0%
- Same copy with an unrelated sentence first, so "alice" is not at position 0. L1H2 still finds her.

**copy-tom** (copy)  
`tom met lucy at the castle . lucy waved at`  
- L1H2 (the head-noun tracker): q=9 `at` → k=0 `tom` **0.78** ✓
- L1H1 (the clause hopper): q=9 `at` → k=1 `met` **0.79** ✓
- next after `at` (q=9): tom 100.0%, rose 0.0%, emma 0.0%
- Second copy template.

**fact-capital** (fact)  
`the capital of japan is`  
- L0H2 (the fact finder): q=4 `is` → k=3 `japan` **0.81** ✓
- next after `is` (q=4): tokyo 100.0%, beijing 0.0%, paris 0.0%
- L0H2 reads the country; the answer is then written by MLP neurons (see MLP memory).

**fact-reverse** (fact)  
`tokyo is the capital of`  
- L0H2 (the fact finder): q=4 `of` → k=0 `tokyo` **0.98** ✓
- next after `of` (q=4): japan 99.9%, kenya 0.0%, peru 0.0%
- The reverse direction is memorised too.

**fact-sound** (fact)  
`the duck says`  
- L0H3 (the antecedent finder): q=2 `says` → k=1 `duck` **0.94** ✓
- next after `says` (q=2): quack 100.0%, his 0.0%, was 0.0%
- Animal sound.

**fact-colour** (fact)  
`the sky is`  
- L0H3 (the antecedent finder): q=2 `is` → k=1 `sky` **0.88** ✓
- next after `is` (q=2): blue 100.0%, yellow 0.0%, green 0.0%
- Same syntax as a binding question ("the box is"), answered from memory.

**parrot** (parrot)  
`the girl said red kite blue cup . the parrot said`  
- L1H3 (the colour binder): q=10 `said` → k=3 `red` **0.90** ✓
- L0H0 (the backstitch): q=10 `said` → k=9 `parrot` **0.87** ✓
- next after `said` (q=10): red 100.0%, green 0.0%, blue 0.0%
- L1H3, the colour binder, also does the parrot's copying: it points at the first word to repeat.

**hero** (hero)  
`the keys near the old door are gold . the red ball and the blue box . the queen opened the box because she was happy . the box is`  
- L1H2 (the head-noun tracker): q=5 `door` → k=1 `keys` **0.80** ✓
- L0H3 (the antecedent finder): q=22 `because` → k=18 `queen` **1.00** ✓
- L1H3 (the colour binder): q=29 `is` → k=14 `blue` **0.69** ✓
- L0H0 (the backstitch): q=29 `is` → k=28 `box` **0.96** ✓
- next after `is` (q=29): blue 99.9%, red 0.1%, pink 0.0%
- next after `door` (q=5): are 100.0%, is 0.0%, . 0.0%
- next after `because` (q=22): she 100.0%, because 0.0%, it 0.0%
- Hero sentence, 30 tokens. Three dependencies, three different heads: L1H2 (agreement, at "door"), L0H3 (pronoun, at "because"), L1H3 (binding, at the final "is").

Suggested uses: *pronoun-she / pronoun-he* as a toggle (one word flips the pronoun, same head); *agree-are / agree-is* as a toggle; *bind-blue* for the "three heads, three jobs" figure; **hero** as the chapter-opening tapestry (30 tokens, three dependencies each lit by a different head and dye).

## 4. Logit lens stories

`r.lens[l][t]` decodes the residual stream entering layer l (l = 3 is the final output) through the final LayerNorm and unembedding. Top 3 at each stage:

**lens-agree-flip**: `the key near the old doors`, at q=5 `doors`

| stage | top 3 |
|---|---|
| embedding (resid[0]) | are 31.3%, green 10.9%, nairobi 10.1% |
| after layer 0 (resid[1]) | are 99.9%, thanked 0.1%, oink 0.0% |
| after layer 1 (resid[2]) | is 100.0%, ball 0.0%, are 0.0% |
| final output | is 100.0%, walked 0.0%, lucy 0.0% |

Layer 0 says "are" (it says "are" for every agreement item); layer 1, where L1H2 reads the singular head noun "key", flips it to "is".

**lens-binding**: `the red ball and the blue box . the box is`, at q=10 `is`

| stage | top 3 |
|---|---|
| embedding (resid[0]) | of 45.8%, was 17.3%, its 9.3% |
| after layer 0 (resid[1]) | its 56.3%, her 32.8%, madrid 5.8% |
| after layer 1 (resid[2]) | blue 99.5%, white 0.2%, green 0.1% |
| final output | blue 100.0%, red 0.0%, white 0.0% |

Nothing colour-like until layer 1 (L1H3) writes "blue". Before that the stream guesses function words and possessives.

**lens-cat**: `the cat says`, at q=2 `says`

| stage | top 3 |
|---|---|
| embedding (resid[0]) | oink 97.5%, her 1.5%, was 0.4% |
| after layer 0 (resid[1]) | oink 99.3%, meow 0.6%, at 0.1% |
| after layer 1 (resid[2]) | oink 97.3%, meow 2.5%, at 0.1% |
| final output | meow 100.0%, woof 0.0%, at 0.0% |

Up to the end of layer 1 the stream says "oink" (97-99%), the model's default animal sound. Only the last layer (its MLP, see MLP memory) rewrites it to "meow" (100%).

**lens-copy**: `alice gave bob a cup . bob thanked`, at q=7 `thanked`

| stage | top 3 |
|---|---|
| embedding (resid[0]) | tom 30.1%, emma 13.3%, rose 10.7% |
| after layer 0 (resid[1]) | alice 36.6%, rose 15.5%, emma 12.8% |
| after layer 1 (resid[2]) | alice 55.3%, rose 24.1%, emma 8.2% |
| final output | alice 100.0%, rose 0.0%, emma 0.0% |

A name is expected from the start; which name sharpens layer by layer (alice 37% → 55% → 100%).

**lens-pronoun**: `the queen opened the door because`, at q=5 `because`

| stage | top 3 |
|---|---|
| embedding (resid[0]) | of 44.6%, they 37.3%, she 11.6% |
| after layer 0 (resid[1]) | she 99.8%, it 0.2%, he 0.0% |
| after layer 1 (resid[2]) | she 100.0%, it 0.0%, her 0.0% |
| final output | she 100.0%, because 0.0%, they 0.0% |

The raw embedding guesses "of"/"they"; one layer later (L0H3 has read "queen") it is "she" at 99.8%.

Population view (from `meta.analysis.lens`, mean P(correct) at resid[0] → [1] → [2] → final): agreement 0.09→0.49→1.00→1.00; binding 0.01→0.04→0.97→0.98; capital 0.02→0.79→0.89→1.00; colorfact 0.01→0.11→0.90→1.00; copy 0.03→0.18→0.40→1.00; parrot 0.05→0.46→0.95→1.00; possessive 0.19→0.89→0.89→1.00; pronoun 0.14→0.97→0.97→1.00; sound 0.12→0.59→0.70→1.00. Pronoun, possessive and agreement are settled after one or two layers; binding after layer 1; copy and the facts need the last layer.

One more agreement fact for the lens figure: after layer 0, the stream prefers `are` over `is` on **all 899** held-out agreement items (are/(are+is) ≥ 0.99 in every head/attractor combination). Layer 1 then flips singular cases to `is`. So the layer-0 `are` is a default, not an attraction error; do not caption it as "fooled by the attractor".

## 5. MLP memory: where facts live

For all 30 memorised facts (16 capital directions, 8 sounds, 6 colour facts) we decoded the residual stream at the answer slot before and after every sublayer (lens on `residMid[l]` vs `resid[l+1]`). Mean P(answer):

| stage | embed (resid[0]) | after L0 attn (residMid[0]) | after L0 MLP (resid[1]) | after L1 attn (residMid[1]) | after L1 MLP (resid[2]) | after L2 attn (residMid[2]) | after L2 MLP (resid[3]) |
|---|---|---|---|---|---|---|---|
| mean P(answer) | 0.04 | 0.05 | 0.57 | 0.61 | 0.83 | 0.84 | 1.00 |

The biggest single jump is **MLP 0** (0.05 → 0.57): L0H2/L0H3 copy the country or animal into the slot, and layer-0 MLP neurons turn it into the answer. MLP 1 and MLP 2 then sharpen it. Per fact:

| prompt | answer | P(ans) by stage (embed, L0 attn, L0 MLP, L1 attn, L1 MLP, L2 attn, final) | top neurons by direct logit attribution | ablate top 10 → | whole MLP 2 ablated → |
|---|---|---|---|---|---|
| `the capital of france is` | paris | 0.01 0.01 0.94 0.93 0.94 0.96 1.00 | L0N197, L0N164, L0N239 | paris (P(ans) 0.69) | paris (0.97) |
| `paris is the capital of` | france | 0.00 0.01 0.87 0.90 0.95 0.96 1.00 | L0N236, L0N11, L1N195 | . (P(ans) 0.00) | france (0.97) |
| `the capital of japan is` | tokyo | 0.02 0.00 0.52 0.58 0.91 0.92 1.00 | L0N60, L2N167, L0N184 | tokyo (P(ans) 0.62) | tokyo (0.93) |
| `tokyo is the capital of` | japan | 0.00 0.03 0.89 0.91 0.95 0.96 1.00 | L0N236, L0N240, L0N48 | japan (P(ans) 0.47) | japan (0.97) |
| `the capital of italy is` | rome | 0.01 0.00 0.95 0.94 0.97 0.95 1.00 | L0N184, L0N219, L0N40 | bob (P(ans) 0.02) | rome (0.95) |
| `rome is the capital of` | italy | 0.00 0.01 0.92 0.92 0.98 0.99 1.00 | L0N111, L0N11, L0N73 | because (P(ans) 0.01) | italy (0.99) |
| `the capital of spain is` | madrid | 0.00 0.00 0.55 0.55 0.71 0.65 1.00 | L0N96, L0N152, L0N184 | madrid (P(ans) 0.48) | madrid (0.66) |
| `madrid is the capital of` | spain | 0.00 0.00 0.97 0.97 0.97 0.99 1.00 | L0N236, L0N247, L0N164 | moo (P(ans) 0.19) | spain (0.99) |
| `the capital of egypt is` | cairo | 0.00 0.05 0.94 0.93 0.97 0.98 1.00 | L0N44, L0N173, L0N152 | cairo (P(ans) 0.98) | cairo (0.98) |
| `cairo is the capital of` | egypt | 0.00 0.00 0.98 0.98 0.99 0.99 1.00 | L0N194, L0N219, L0N220 | egypt (P(ans) 0.31) | egypt (0.99) |
| `the capital of peru is` | lima | 0.01 0.00 0.42 0.50 0.60 0.63 1.00 | L0N12, L0N33, L0N96 | jack (P(ans) 0.06) | lima (0.67) |
| `lima is the capital of` | peru | 0.00 0.00 0.62 0.78 0.87 0.79 1.00 | L0N11, L0N111, L0N100 | the (P(ans) 0.00) | peru (0.84) |
| `the capital of china is` | beijing | 0.01 0.02 0.33 0.32 0.68 0.58 1.00 | L0N253, L2N251, L0N219 | roar (P(ans) 0.04) | beijing (0.62) |
| `beijing is the capital of` | china | 0.02 0.01 0.99 0.99 0.99 0.99 1.00 | L0N11, L1N204, L0N224 | china (P(ans) 0.30) | china (0.99) |
| `the capital of kenya is` | nairobi | 0.01 0.00 0.84 0.84 0.94 0.94 1.00 | L0N255, L2N185, L2N161 | paris (P(ans) 0.27) | nairobi (0.95) |
| `nairobi is the capital of` | kenya | 0.00 0.01 0.68 0.88 0.91 0.89 1.00 | L0N236, L0N111, L0N224 | kenya (P(ans) 0.89) | kenya (0.91) |
| `the cow says` | moo | 0.00 0.02 0.35 0.51 0.86 0.93 1.00 | L0N167, L0N79, L0N192 | and (P(ans) 0.07) | moo (0.96) |
| `the dog says` | woof | 0.00 0.00 0.06 0.05 0.08 0.06 1.00 | L0N48, L0N253, L0N60 | their (P(ans) 0.15) | oink (0.09) |
| `the cat says` | meow | 0.00 0.00 0.01 0.01 0.03 0.00 1.00 | L0N40, L0N60, L2N70 | their (P(ans) 0.00) | oink (0.00) |
| `the duck says` | quack | 0.00 0.34 0.77 0.77 0.80 0.89 1.00 | L0N234, L0N232, L2N10 | quack (P(ans) 0.71) | quack (0.93) |
| `the sheep says` | baa | 0.00 0.00 0.93 0.96 0.97 0.97 1.00 | L0N120, L0N40, L0N167 | baa (P(ans) 0.54) | baa (0.98) |
| `the owl says` | hoot | 0.00 0.00 0.16 0.33 0.82 0.77 1.00 | L0N60, L0N167, L0N152 | ball (P(ans) 0.01) | hoot (0.85) |
| `the pig says` | oink | 0.98 0.84 1.00 1.00 1.00 1.00 1.00 | L0N60, L0N167, L0N40 | oink (P(ans) 0.95) | oink (1.00) |
| `the lion says` | roar | 0.00 0.04 0.87 0.87 0.90 0.92 1.00 | L0N167, L2N10, L0N0 | roar (P(ans) 0.60) | roar (0.95) |
| `the sky is` | blue | 0.00 0.00 0.02 0.01 0.98 0.99 1.00 | L1N209, L1N222, L0N61 | blue (P(ans) 0.81) | blue (0.99) |
| `the grass is` | green | 0.00 0.00 0.33 0.12 0.74 0.90 1.00 | L2N91, L0N63, L1N24 | green (P(ans) 0.77) | green (0.94) |
| `the snow is` | white | 0.00 0.04 0.11 0.18 1.00 1.00 1.00 | L1N222, L2N168, L0N225 | white (P(ans) 0.96) | white (1.00) |
| `the sun is` | yellow | 0.03 0.01 0.15 0.18 0.98 0.99 1.00 | L0N208, L2N251, L1N222 | yellow (P(ans) 0.38) | yellow (0.99) |
| `the apple is` | red | 0.00 0.00 0.00 0.00 0.55 0.75 1.00 | L2N44, L2N120, L0N12 | red (P(ans) 0.61) | red (0.81) |
| `the pig is` | pink | 0.00 0.02 0.05 0.25 1.00 1.00 1.00 | L0N192, L1N33, L0N61 | pink (P(ans) 0.93) | pink (1.00) |

*Direct logit attribution* (DLA) of neuron i in layer l = activation × (row i of `W_proj`) · (`lnf.g` ⊙ (unembedding column of the answer − mean column)) / σ(final residual). It ranks neurons by how much they push the answer's logit directly.

**Ablation results** (zeroing a neuron = zeroing its row of `W_proj`):

- Zeroing a fact's top 1 or top 3 DLA neurons almost never breaks it (top-3: 0/30 facts lose their top-1 answer). The memory is spread out.
- Zeroing its **top 10** breaks 12/30 facts (the answer is no longer top-1).
- **The damage is specific.** Zeroing the top-10 neurons of `rome is the capital of`, `paris is the capital of`, `the capital of italy is`, `the cat says` or `the cow says` breaks that fact and leaves **29 of the other 30 facts top-1 correct**, with dependency accuracy on 600 held-out sequences still 0.988–0.997. Five trials of 10 random layer-0 neurons broke 0/30 facts. (Note that `rome is the capital of` and `the capital of italy is` use different neurons: breaking one direction leaves the other intact.)
- **"oink" is the default animal sound.** At `says`, the raw embedding already predicts `oink` at 98% for every animal. For `dog` and `cat` the stream still says `oink` after layer 1 (P(woof) 0.08, P(meow) 0.03); only **MLP 2** writes the right sound. Zero MLP 2 and `the cat says` → oink, `the dog says` → oink. This is the cleanest "the last MLP recalls a fact" demo.
- **Whole layers:** zeroing MLP 1 changes no fact's answer; zeroing MLP 2 changes two (dog, cat). Zeroing MLP 0 breaks 29 of 30 (all but `the pig is` → pink) but also breaks every other skill (§2), so it is not fact-specific.

**Named neurons** (what a neuron promotes = top tokens of row i of `W_proj` through the final LN gain and unembedding; what fires it = its activation on the 30 fact prompts):

| neuron | promotes | fires most on | suggested label |
|---|---|---|---|
| L0N236 | france, spain, japan, italy, china, kenya | `tokyo / madrid / paris is the capital of` (2.4–2.9) | country neuron |
| L0N11 | peru, france, italy, spain, china, kenya | `beijing / paris / rome / lima is the capital of` (2.1–4.2) | country neuron 2 |
| L0N167 | oink, baa, hoot, moo, roar | `the pig / cow / sheep / lion / owl says` (2.7–3.6) | animal-sound neuron |
| L1N222 | green, white, pink, red, blue, yellow (all six colours) | `the snow / sky / sun is` (1.3–2.2) | colour neuron |
| L2N10 | quack, roar, meow (and of, are, were) | capitals (kenya, egypt 3.7) and sounds (owl 3.5, lion 3.0) | late "answer slot" neuron, mixed |

For a panel: run `the capital of france is`, show the lens before and after MLP 0 (0.01 → 0.94 P(paris)), list `r.mlp[0][4]`'s most active neurons, and let the reader zero the top 10 for a fact (the trace in `AM_NOTES.facts` has the neuron ids). To ablate in the browser you have to patch a copy of the weights; `AM.model` has no ablation switch (see §9).

## 6. Sampling and temperature

Temperature τ rescales probabilities as p^(1/τ), renormalised, exactly as `m.sample()` does (after dropping `<pad>`/`<unk>`). "eff." = e^entropy, the effective number of choices.

**`the queen` → ?**

| τ | eff. | top 10 |
|---|---|---|
| 0.5 | 3.17 | walked 59.8%, said 26.4%, loved 6.1%, dropped 1.6%, closed 1.6%, lost 1.6%, found 1.5%, opened 1.5%, says 0.0%, and 0.0% |
| 1.0 | 6.08 | walked 35.8%, said 23.8%, loved 11.4%, dropped 5.9%, closed 5.8%, lost 5.8%, found 5.7%, opened 5.6%, says 0.0%, and 0.0% |
| 1.5 | 8.86 | walked 26.0%, said 19.8%, loved 12.1%, dropped 7.8%, closed 7.8%, lost 7.7%, found 7.6%, opened 7.5%, says 0.3%, and 0.2% |

**`the dog` → ?**

| τ | eff. | top 10 |
|---|---|---|
| 0.5 | 2.08 | says 79.6%, walked 12.8%, said 5.0%, loved 1.2%, dropped 0.3%, found 0.3%, lost 0.3%, closed 0.3%, opened 0.3%, is 0.0% |
| 1.0 | 5.12 | says 48.1%, walked 19.3%, said 12.0%, loved 5.8%, dropped 3.0%, found 2.9%, lost 2.9%, closed 2.9%, opened 2.8%, is 0.0% |
| 1.5 | 9.36 | says 32.2%, walked 17.5%, said 12.8%, loved 7.9%, dropped 5.1%, found 5.0%, lost 5.0%, closed 4.9%, opened 4.8%, is 0.2% |

**`the` → ?**

| τ | eff. | top 10 |
|---|---|---|
| 0.5 | 28.92 | capital 16.9%, pig 11.5%, cat 3.6%, duck 3.5%, owl 3.5%, lion 3.4%, dog 3.4%, cow 3.3%, grass 2.5%, children 2.4% |
| 1.0 | 41.67 | capital 6.8%, pig 5.6%, cat 3.1%, duck 3.1%, owl 3.1%, lion 3.0%, dog 3.0%, cow 3.0%, grass 2.6%, children 2.5% |
| 1.5 | 47.04 | capital 4.7%, pig 4.1%, cat 2.8%, duck 2.8%, owl 2.8%, lion 2.8%, dog 2.8%, cow 2.7%, grass 2.5%, children 2.5% |

**`the red ball and the blue box . alice found the blue` → ?**

| τ | eff. | top 10 |
|---|---|---|
| 0.5 | 1.57 | box 83.2%, ball 16.8%, book 0.0%, door 0.0%, kite 0.0%, hat 0.0%, ring 0.0%, yellow 0.0%, pig 0.0%, key 0.0% |
| 1.0 | 1.91 | box 68.8%, ball 30.9%, book 0.1%, door 0.1%, kite 0.0%, hat 0.0%, ring 0.0%, yellow 0.0%, pig 0.0%, key 0.0% |
| 1.5 | 2.50 | box 60.6%, ball 35.5%, book 0.9%, door 0.8%, kite 0.4%, hat 0.3%, ring 0.2%, yellow 0.1%, pig 0.0%, key 0.0% |

**`the witch said green cup yellow cup . the parrot said` → ?**

| τ | eff. | top 10 |
|---|---|---|
| 0.5 | 1.00 | green 100.0%, cup 0.0%, red 0.0%, yellow 0.0%, cups 0.0%, white 0.0%, rose 0.0%, pink 0.0%, . 0.0%, china 0.0% |
| 1.0 | 1.00 | green 100.0%, cup 0.0%, red 0.0%, yellow 0.0%, cups 0.0%, white 0.0%, rose 0.0%, pink 0.0%, . 0.0%, china 0.0% |
| 1.5 | 1.11 | green 98.9%, cup 0.1%, red 0.1%, yellow 0.1%, cups 0.1%, white 0.1%, rose 0.0%, pink 0.0%, . 0.0%, china 0.0% |

Use `the queen` or `the dog` for a temperature slider: the distribution is skewed but open (5–6 effective choices at τ = 1, about 2–3 at τ = 0.5, 9 at τ = 1.5). `the` (first word) is very broad (42 effective choices). The recall probe (`… alice found the blue`) is the model's honest coin flip between `box` and `ball`. The parrot prompt barely moves even at τ = 1.5: a good contrast ("when the model is sure, temperature hardly matters").

**Generations at τ = 0.8** with `m.generate(m.encode(prompt).ids, {temperature: 0.8, rng: AM.model.lib.AMTensorLib().rng(seed), maxNew: 32 - promptLength, stopAt: null})` (stopAt null = run to the 32-token window; the default stops at the first `.`):

- `the`, seed 1: the pink door and the small blue ball . the prince found the pink door . the ball is blue . the pig is pink . the apple is red . the
- `the`, seed 2: the sheep says baa . jack helped alice . alice thanked jack . the boy said pink ring red ring . the parrot said pink ring red ring . jack walked to
- `the`, seed 3: the old red book and the pink ring . the ring is pink . the dog says woof . the children walked to the garden . the prince loved his crown .
- `alice`, seed 1: alice helped emma . emma thanked alice . the cats found the ring because they were hungry . the grass is green . the owl says hoot . the cat says meow
- `alice`, seed 2: alice and emma walked to the park with their ring . the lion says roar . the books under the old box are gold . the boy walked to the park .
- `alice`, seed 3: alice and jack said red ball pink hat . the parrot said red ball pink hat . the books near the small hats are old . the queen loved her ball .
- `the red ball and the blue box .`, seed 1: the red ball and the blue box . the box is blue . the cats found the red ball . the dog says woof . the grass is green . the sun
- `the red ball and the blue box .`, seed 2: the red ball and the blue box . the box is blue . the ball is red . the duck says quack . the key under the old cup is old .
- `the red ball and the blue box .`, seed 3: the red ball and the blue box . the box is blue . the ball is red . the capital of italy is rome . the pig says oink . the capital
- `the queen`, seed 1: the queen said pink box yellow ball . the parrot said pink box yellow ball . the pig says oink . the apple is red . the cups near the boxes are
- `the queen`, seed 2: the queen loved her book . jack helped alice . alice thanked jack . the wizard said pink ring yellow ring . the parrot said pink ring yellow ring . sam walked
- `the queen`, seed 3: the queen loved her hat . the snow is white . the lion says roar . the boy walked to the castle . the grass is green . the book behind the

These reproduce exactly in Node with the shipped weights. We checked every dependency in these 12 generations by hand: all binding answers, pronouns, possessives, agreements, copies, parrot repeats and facts are correct (e.g. `the pink door and the small blue ball . … the ball is blue`, `the cats found the ring because they were hungry`, `the books near the small hats are old`).

## 7. Token embeddings

PCA of the 64-d token embeddings (`wte`), fitted on the 136 real tokens. The first two components explain only 11.2% and 9.4% of the variance, so a 2-D map is a rough projection; say so in the caption. `<pad>` and `<unk>` received no useful gradient and stay near their initial scale (norm 0.137 and 0.154, about what the N(0, 0.02) init gives in 64 dimensions, vs a median of 1.959 for real tokens), so they sit near the origin. **Axes:** PC1 separates subject nouns (people, animals, plural subjects at +0.9 to +1.4) from everything else; PC2 runs from predicate and function words (`is`, `because`, `loved`, possessives at about −0.8 to −1.3) up to people, countries and capitals (+0.8 to +1.1). Two animals sit away from the animal cluster: `pig` (also the subject of the colour fact `the pig is pink`) and `sheep` (only ever in `the sheep says baa`).

**Clusters.** Mean cosine similarity within each word kind vs to every other token (embedding / unembedding):

| kind | n | within | others | unembed within | unembed others | PCA centroid |
|---|---|---|---|---|---|---|
| punct | 2 | 0.18 | -0.07 | 0.16 | 0.13 | (0.24, -0.10) |
| function | 19 | 0.04 | -0.03 | 0.27 | 0.25 | (0.21, -0.55) |
| verb | 12 | 0.14 | -0.04 | 0.54 | 0.20 | (0.02, -0.54) |
| pronoun | 4 | 0.68 | 0.00 | 0.42 | 0.31 | (-0.15, -0.43) |
| possessive | 4 | 0.79 | -0.03 | 0.56 | 0.33 | (0.37, -0.91) |
| person | 8 | 0.80 | 0.00 | 0.98 | 0.36 | (1.31, 0.89) |
| name | 8 | 0.46 | -0.00 | 0.57 | 0.19 | (0.65, 0.23) |
| animal | 8 | 0.59 | -0.01 | 0.90 | 0.33 | (0.91, 0.28) |
| plural | 3 | 0.86 | 0.02 | 0.99 | 0.35 | (1.38, 0.59) |
| object | 10 | 0.23 | -0.00 | 0.30 | 0.13 | (-0.35, -0.35) |
| objects | 6 | 0.38 | 0.00 | 0.59 | 0.25 | (-0.22, -0.44) |
| state | 6 | 0.81 | 0.06 | 0.99 | 0.34 | (-0.39, -0.15) |
| adj | 6 | 0.55 | 0.05 | 0.74 | 0.27 | (-0.48, -0.03) |
| color | 6 | 0.71 | -0.00 | 0.39 | 0.10 | (-0.23, -0.04) |
| place | 5 | 0.97 | 0.03 | 1.00 | 0.34 | (-0.26, 0.02) |
| country | 8 | 0.67 | 0.05 | 0.77 | 0.32 | (-1.01, 0.81) |
| city | 8 | 0.66 | 0.03 | 0.75 | 0.29 | (-0.71, 1.01) |
| sound | 8 | 0.64 | 0.06 | 0.68 | 0.35 | (-0.33, -0.06) |
| nature | 5 | 0.58 | 0.03 | 0.97 | 0.38 | (-0.72, 0.55) |

Observations:

- Strong clusters: places (0.97), plural subjects (0.86), states (0.81), people (0.80), possessives (0.79), colours (0.71), pronouns (0.68), countries and capitals (~0.66). Words that are interchangeable in the grammar end up nearly identical (`park`/`river`/`garden`).
- Function words and verbs do not cluster in the input embedding (0.04, 0.14) but verbs do in the unembedding (0.54): words used in the same *slot* share an output direction.
- **Gender is a direction.** People nouns split by the gender their pronoun needs: `queen` is closest to princess 0.942, witch 0.925, girl 0.906; `king` to wizard 0.9, prince 0.885, boy 0.812. The vector queen − king has cosine 0.704 (princess-prince), 0.71 (witch-wizard), 0.455 (girl-boy) with the other royal/role pairs and 0.18–0.49 with female − male name pairs. `prince − boy + girl` → princess 0.830. It does **not** extend to the pronouns themselves (she − he: -0.068), and `king − he + she` → wizard 0.844 (wrong), so do not promise word-vector arithmetic in general.
- **Number is a direction for objects.** keys − key has cosine 0.45–0.58 with the other object plural pairs; `keys − key + box` → boxes 0.684. Weaker for animals (dogs − dog 0.168).
- Pronouns: `she` ~ it 0.923, he 0.915; `they` sits apart (0.379). `is` ~ `are` 0.751.

**Nearest neighbours (cosine, input embeddings):**

| word | top 5 |
|---|---|
| queen | princess 0.94, witch 0.93, girl 0.91, boy 0.75, prince 0.72 |
| king | wizard 0.90, prince 0.89, boy 0.81, princess 0.74, witch 0.72 |
| alice | emma 0.55, sam 0.54, lucy 0.53, tom 0.51, bob 0.39 |
| bob | jack 0.59, tom 0.57, lucy 0.50, sam 0.46, cups 0.44 |
| dog | owl 0.91, lion 0.88, cat 0.86, duck 0.82, cow 0.74 |
| dogs | cats 0.94, children 0.82, girl 0.64, princess 0.61, prince 0.61 |
| key | keys 0.79, crown 0.41, gold 0.31, lost 0.31, boxes 0.30 |
| keys | key 0.79, boxes 0.46, hats 0.41, cups 0.39, doors 0.37 |
| red | yellow 0.75, green 0.74, pink 0.72, white 0.62, blue 0.61 |
| blue | pink 0.77, green 0.76, white 0.76, red 0.61, yellow 0.59 |
| france | egypt 0.73, japan 0.73, kenya 0.70, italy 0.68, china 0.66 |
| paris | tokyo 0.74, lima 0.67, nairobi 0.66, madrid 0.65, beijing 0.63 |
| she | it 0.92, he 0.92, they 0.38, with 0.33, says 0.32 |
| his | a 0.80, her 0.79, their 0.79, its 0.75, thanked 0.28 |
| cold | happy 0.91, tired 0.81, sleepy 0.79, hungry 0.78, sad 0.75 |
| moo | meow 0.73, hoot 0.72, roar 0.70, oink 0.67, quack 0.64 |
| is | are 0.75, its 0.36, at 0.29, thanked 0.28, their 0.27 |
| are | is 0.75, at 0.35, its 0.31, thanked 0.29, her 0.26 |

**PCA coordinates** (PC1, PC2) for every token, in vocabulary order (also `AM_NOTES.pca`):

```json
{"<pad>":[0.084,-0.146],
"<unk>":[0.051,-0.188],
".":[0.038,-0.18],
",":[0.453,-0.009],
"the":[-0.076,-0.365],
"a":[0.148,-0.877],
"and":[0.171,0.057],
"because":[-0.159,-1.168],
"with":[0.263,-1.086],
"of":[0.505,-0.416],
"is":[0.492,-1.2],
"are":[0.425,-0.812],
"was":[0.254,-1.088],
"were":[0.126,-0.986],
"says":[0.068,-1.098],
"to":[-0.032,0.183],
"at":[-0.185,-0.479],
"near":[0.345,-0.006],
"under":[0.371,-0.009],
"behind":[0.398,0.041],
"walked":[-0.075,0.102],
"loved":[0.114,-1.28],
"gave":[0.227,-0.629],
"thanked":[-0.146,-0.572],
"met":[0.024,-0.507],
"waved":[0.915,-0.53],
"helped":[-0.157,-0.868],
"capital":[0.48,-0.767],
"said":[0.099,-0.109],
"parrot":[0.29,-0.269],
"he":[-0.326,-0.459],
"she":[-0.305,-0.468],
"it":[-0.202,-0.373],
"they":[0.242,-0.423],
"his":[0.163,-0.772],
"her":[0.594,-0.917],
"its":[0.399,-1.041],
"their":[0.337,-0.924],
"king":[1.357,0.79],
"queen":[1.347,0.859],
"boy":[1.17,0.702],
"girl":[1.385,1.066],
"prince":[1.278,1.073],
"princess":[1.405,0.845],
"wizard":[1.269,0.977],
"witch":[1.313,0.82],
"alice":[0.642,0.173],
"emma":[0.662,0.159],
"lucy":[0.687,0.178],
"rose":[0.723,0.428],
"bob":[0.624,0.082],
"tom":[0.704,0.189],
"sam":[0.546,0.476],
"jack":[0.593,0.153],
"dog":[1.101,0.265],
"cat":[1.127,0.392],
"duck":[1.065,0.465],
"owl":[1.19,0.192],
"pig":[0.149,0.792],
"cow":[1.162,0.369],
"lion":[1.19,0.397],
"sheep":[0.259,-0.671],
"children":[1.393,0.463],
"dogs":[1.368,0.607],
"cats":[1.391,0.707],
"key":[-0.357,-0.44],
"keys":[-0.282,-0.378],
"box":[-0.247,-0.404],
"boxes":[-0.261,-0.496],
"cup":[-0.326,-0.322],
"cups":[-0.107,-0.276],
"door":[-0.351,-0.539],
"doors":[-0.302,-0.714],
"book":[-0.264,-0.444],
"books":[-0.129,-0.292],
"hat":[-0.399,-0.283],
"hats":[-0.26,-0.471],
"ring":[-0.432,-0.287],
"ball":[-0.314,-0.377],
"kite":[-0.17,-0.171],
"crown":[-0.6,-0.224],
"opened":[-0.34,-0.44],
"closed":[-0.297,-0.317],
"found":[0.097,-0.605],
"lost":[-0.067,-0.371],
"dropped":[-0.051,-0.406],
"cold":[-0.346,-0.133],
"tired":[-0.478,-0.109],
"happy":[-0.383,0.004],
"sad":[-0.115,-0.282],
"hungry":[-0.524,-0.149],
"sleepy":[-0.464,-0.211],
"old":[-0.278,0.167],
"small":[-0.206,0.196],
"gold":[-0.775,-0.181],
"heavy":[-0.396,-0.097],
"broken":[-0.609,-0.144],
"shiny":[-0.647,-0.132],
"red":[-0.222,0.003],
"blue":[-0.191,-0.146],
"green":[-0.196,0.034],
"yellow":[-0.342,0.044],
"white":[-0.179,-0.065],
"pink":[-0.228,-0.119],
"park":[-0.297,0.045],
"river":[-0.28,-0.072],
"market":[-0.296,0.062],
"castle":[-0.239,0.034],
"garden":[-0.185,0.04],
"france":[-1.033,0.634],
"japan":[-0.924,0.778],
"italy":[-1.032,0.847],
"spain":[-0.954,0.897],
"egypt":[-0.958,0.805],
"peru":[-1.133,0.813],
"china":[-1.016,0.857],
"kenya":[-0.999,0.859],
"paris":[-0.784,0.982],
"tokyo":[-0.72,1.044],
"rome":[-0.701,0.897],
"madrid":[-0.837,1.007],
"cairo":[-0.848,0.898],
"lima":[-0.52,1.173],
"beijing":[-0.661,0.921],
"nairobi":[-0.649,1.128],
"moo":[-0.302,-0.175],
"woof":[-0.469,0.215],
"meow":[-0.285,-0.006],
"quack":[-0.526,-0.325],
"baa":[-0.131,-0.037],
"hoot":[-0.218,-0.154],
"oink":[-0.459,0.05],
"roar":[-0.283,-0.029],
"sky":[-0.584,0.494],
"grass":[-0.596,0.39],
"snow":[-0.77,0.728],
"sun":[-0.9,0.58],
"apple":[-0.748,0.554]}
```

## 8. The Lab models: reverse and sort

Format: 8 digits, `>`, answer. Positions 0–7 are the input, 8 is `>`, and query position 8 + j predicts output digit j. Statistics over 2,000 random inputs (teacher-forced); probe rows from `run()` on the first 16 tokens, written `query token → key token@key position weight` for the head's largest weight at each output position.

### reverse (1 layer × 1 head, d 24, 8,195 params)

**A perfect anti-diagonal.** Query 8 + j attends to input position 7 − j with mean weight 0.974–0.997 (per output position: 0.993, 0.997, 0.994, 0.991, 0.990, 0.974, 0.985, 0.991), and that position is the argmax in 100% of 16,000 cases. Exact accuracy 100% (10,000 inputs).

- `38152907>` → greedy `70925183` (target `70925183`): >→7@7 0.99, 7→0@6 1.00, 0→9@5 1.00, 9→2@4 1.00, 2→5@3 0.98, 5→1@2 0.98, 1→8@1 0.98, 8→3@0 0.99
- `12344321>` → greedy `12344321` (target `12344321`): >→1@7 0.99, 1→2@6 1.00, 2→3@5 0.99, 3→4@4 0.99, 4→4@3 0.99, 4→3@2 0.99, 3→2@1 0.99, 2→1@0 0.99
- `00000009>` → greedy `90000000` (target `90000000`): >→9@7 1.00, 9→0@6 1.00, 0→0@5 0.99, 0→0@4 0.99, 0→0@3 0.99, 0→0@2 0.97, 0→0@1 0.98, 0→0@0 0.99
- `90000000>` → greedy `00000009` (target `00000009`): >→0@7 0.99, 0→0@6 0.99, 0→0@5 0.99, 0→0@4 0.99, 0→0@3 0.99, 0→0@2 0.97, 0→0@1 0.98, 0→9@0 1.00

Probe suggestions: the Lab's default `38152907`; a palindrome `12344321` (output = input, but the attention still runs along the anti-diagonal: the model copies by position, not by value); `00000009` / `90000000` (the lone 9 moves to the other end). It trains to 100% in ≈ 75 steps, so offer slow motion.

### sort (2 layers × 2 heads, d 32, 26,699 params)

**L0H1, "the next-digit finder"** is the only crisp head. Mean attention on input positions holding the digit about to be written: 0.61, 0.33, 0.29, 0.30, 0.32, 0.36, 0.44, 0.59 for output positions 0–7, and the argmax holds that digit in 100.0%, 48.8%, 59.5%, 63.5%, 59.2%, 44.3%, 45.2%, 99.9% of cases. So it is sharp at the ends (the smallest digit first, the largest last: argmax correct 100.0% / 99.9%) and soft in the middle. Split by case: when the next digit is new (larger than the current one), 0.34 on the answer digit, 0.21 on the current digit and 0.29 on still-larger digits; when the digit repeats, 0.45 on it.

**The other three heads are soft** (no position above 0.21 mean on the answer digit). L0H0 parks 0.30 on `>` at the first output and moves to the output side as it goes (0.00, 0.09, 0.15, 0.20, 0.26, 0.32, 0.39, 0.48 mass on positions 9+). L1H0 (0.31–0.46 from the second output on) and L1H1 (rising from 0.11 to 0.44) spend much of their attention on already-written output digits, consistent with keeping track of repeats, but we have no clean measurement of that; draw them as "looking over what has been written".

- `73519273>` → greedy `12335779`. L0H1: >→1@3 0.43, 1→2@5 0.21, 2→3@1 0.17, 3→5@2 0.22, 3→5@2 0.21, 5→7@0 0.25, 7→9@4 0.42, 7→9@4 0.62
- `90817263>` → greedy `01236789`. L0H1: >→0@1 0.56, 0→0@1 0.26, 1→2@5 0.24, 2→3@7 0.23, 3→6@6 0.22, 6→8@2 0.27, 7→9@0 0.36, 8→9@0 0.54
- `55555555>` → greedy `55555555`. L0H1: >→5@4 0.12, 5→5@7 0.12, 5→5@1 0.12, 5→5@2 0.12, 5→5@2 0.12, 5→5@2 0.12, 5→5@2 0.12, 5→5@3 0.10
- `99999990>` → greedy `08999999` **(wrong; target `09999999`)**. L0H1: >→0@7 0.93, 0→0@7 0.60, 9→9@0 0.11, 9→9@0 0.12, 9→9@0 0.13, 9→9@0 0.13, 9→9@0 0.14, 9→9@0 0.14
- `11112222>` → greedy `11112222`. L0H1: >→1@1 0.17, 1→2@4 0.14, 1→2@4 0.16, 1→2@4 0.17, 1→2@4 0.18, 2→2@6 0.18, 2→2@6 0.16, 2→>@8 0.33
- `35353535>` → greedy `33335555`. L0H1: >→3@4 0.17, 3→5@1 0.13, 3→5@1 0.15, 3→5@1 0.15, 3→5@1 0.16, 5→5@3 0.18, 5→5@3 0.19, 5→5@3 0.16

Failures: 8 of 5,000 random inputs were sorted wrongly by greedy decoding, all with many repeated digits: `70898898→08888899 (want 07888899)`, `88008117→00118888 (want 00117888)`, `90933893→03339999 (want 03338999)`, `17908801→00118889 (want 00117889)`, `00020001→00000002 (want 00000012)`. All of them contain a run of three or more equal digits. Usually a digit that occurs once is swallowed by the run next to it (`70898898` → `08888899`, the 7 lost); sometimes a digit that is not in the input appears (`99999990` → `08999999`).

Probe suggestions for the Lab: `73519273` (default, mixed repeats), `90817263` (all distinct: L0H1 at `>` locks on `0` at 0.56 and at the last step on `9` at 0.54), `55555555` (L0H1 flat at ~0.12 over the eight 5s: nothing to pick between), and **`99999990`**, which the shipped model gets wrong (`08999999`), a good "even trained models fail" moment. The shipped model fails about 1 input in 600 (8 of 5,000 here).

## 9. Using `AM_NOTES` (js/model/notes-data.js)

Classic script with no dependencies (suggested place: right after `api.js`; it still needs a `<script>` tag in `index.html`). It sets `window.AM_NOTES`:

```js
AM_NOTES.heads['L1H3']        // {name, short, layer, head, evidence:{...}, ablation:{...}, shape:{prev, self, pos0, ...}}
AM_NOTES.examples            // [{id, topic, text, tokens, spot:[{layer, head, q, k, qTok, kTok, w, isArgmax}], predictions:[{q, after, top3}], note}]
AM_NOTES.example('hero')     // lookup by id
AM_NOTES.lens                // logit-lens stories: [{id, text, q, layers:[{stage, top3}], story}]
AM_NOTES.facts               // [{prompt, answer, kind, stages:[{stage, pAns, top}], topNeurons:[{layer, neuron, act, dla}], ablateTop10:{pAns, top}}]
AM_NOTES.pca                 // {explained:[pc1, pc2], coords:{token:[x, y]}, kind:{token: 'color'|'name'|...}}
AM_NOTES.neighbours          // {word: [{token, cos}]}
AM_NOTES.sampling            // [{prompt, temps:{'0.5':{top10, entropyNats, effectiveChoices}, '1':…, '1.5':…}}]
AM_NOTES.generations         // [{prompt, seed, temperature, text}]
AM_NOTES.lab                 // {reverse:{probes, summary}, sort:{probes, summary, failures}}
```

Always recompute live numbers with `run()` for display (they match these to float precision); use `AM_NOTES` for *which* heads, positions and captions to show. Ablation needs a patched copy of the weights: `AM.model.get(name)._net.params['h.0.mlp.wproj'].data` is the live Float32Array, so a chapter that zeroes rows must restore them afterwards (or every other chapter sees the damaged model). Prefer precomputed ablation numbers from this file unless the interaction really needs it.

## 10. Claims to avoid

- "Layer 2 does the thinking." Its attention can be removed with almost no loss (§2).
- "L1H2 alone copies the name." Removing it barely matters (redundant heads).
- "The model was fooled by the attractor" for the layer-0 `are` (§4).
- "The model looks up a noun from its colour" (`alice found the blue …`): not learned, ~55%.
- "Facts live in one neuron." It takes ~10 neurons to break a fact (§5).
- Arithmetic on word vectors beyond the two measured cases (`prince − boy + girl`, `keys − key + box`).
- Sort heads as crisp pointers. Only L0H1 at the first and last positions is crisp.
- "It has never seen this sentence" for pronoun/possessive/agreement examples.

*Regenerate: the analysis scripts live outside the repo (scratchpad); every number can be rechecked by running the quoted sentence through `run()`.*
