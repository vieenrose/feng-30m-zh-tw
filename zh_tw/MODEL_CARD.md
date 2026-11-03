---
language:
- zh
base_model: jiaheng0815/feng-30m
base_model_relation: finetune
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
tags:
- zh-TW
- traditional-chinese
- qwen3
- tiny-llm
- edge-ai
- on-device
- weight-space
- no-finetuning
---

# feng-30m-zh-tw

A **native Traditional-Chinese (zh-TW)** variant of
[feng-30m](https://huggingface.co/jiaheng0815/feng-30m), a 29.4M-parameter Qwen3-style
Chinese chat LLM. The parent is a zh-CN (Simplified) model; this variant reads and writes
Traditional Chinese natively.

**No fine-tuning, no adapter, no runtime converter.** The whole conversion is a
*deterministic, weight-space transformation of a single tensor*: `model.embed_tokens.weight`
(the tied embedding / unembedding). Every other tensor is **bit-identical** to the parent, the
file size, shapes and dtypes are unchanged, and the vocabulary stays at 16384 — so the model
keeps the parent's architecture, its GGUF/C-engine/ESP32 compatibility, and the tied
embedding that the parent's on-device engine requires.

| | parent | feng-30m-zh-tw |
|---|---|---|
| parameters | 29.43M | 29.43M (identical) |
| tensors changed | — | 1 of 123 (`model.embed_tokens.weight`) |
| rows edited | — | 6945 / 16384 (3.4 M scalars) |
| vocab | 16384 | 16384 (pieces renamed, no ids added/removed) |

## Perplexity on seven held-out zh-TW domains

Each slice is 1024 tokens of Traditional Chinese text that was **never** used to select any
knob (domains 3–7 are fresh Wikipedia articles, verified 12-gram-disjoint from the eval
slices and from the conversion's attestation corpus).

| domain (topic) | parent | feng-30m-zh-tw | Δ |
|---|---|---|---|
| eval1 · Wikipedia | 202.27 | **169.00** | −16.4% |
| eval2 · Wikipedia (other domain) | 313.33 | **212.74** | −32.1% |
| val3 · fresh articles | 232.19 | **165.69** | −28.6% |
| val4 · geography / sport | 331.27 | **296.84** | −10.4% |
| val5 · culture / food | 140.65 | **138.22** | −1.7% |
| val6 · science / history | 327.29 | **268.01** | −18.1% |
| val7 · science / medicine | 217.43 | **184.56** | −15.1% |
| **mean CE, six tuning domains (nats)** | 5.508 | **5.302** | **−3.7%** |

Controls (these show the shift is *language identity*, not a softmax temperature trick):

| control | parent | feng-30m-zh-tw | reading |
|---|---|---|---|
| Simplified zh-CN text | 220.32 | 769.02 | deliberately much worse — the model is not a zh-CN model any more |
| English text | 43.60 | 42.81 | essentially unchanged (no global sharpening) |

Generated text contains **no Simplified-only glyphs** (behavioural purity check over sampled
completions), and no Simplified-only token is emitted at full strength.

## How it works

Three stages, driven by Unihan glyph data plus a small eval-disjoint zh-TW corpus
(`data/zh_tw_reference.txt`, ~130k chars, used as attestation evidence):

1. **Words.** For each Simplified token `w` with a Traditional spelling `c`:
   * if `c` is already a token and `cos(w, k) ≥ 0.78` (same word), **swap** the two strings and
     serve `c` with `w`'s trained row;
   * if the geometry gate blocks it, **transfer** 20 % of `w`'s direction into `k`'s
     under-trained row;
   * if `c` has no token at all, **rename** `w` to `c` — kept only when the corpus attests that
     spelling and `‖E[w]‖ ≥ 1.05` (a strong enough carrier).
2. **Characters.** A Traditional glyph row takes 40 % of its Simplified twin's direction
   (`cos ≥ 0.70`), skipping glyphs the zh-TW corpus never uses. Transfers only ever come from the
   **exact** twin — nearest-neighbour and compositional variants were measured and are harmful.
3. **Softmax rebalance.** Ids that can emit a Simplified-only glyph drop to **0.32** of their
   row; ids emitting a genuinely Traditional glyph are scaled by **1.10** if they are the
   parent's own, and shrunk to **0.70** if they are this transform's own renames (those rows are
   Simplified-trained, so amplifying them would push Simplified geometry back in).

Two design principles did all the heavy lifting: **string/row provenance** (a token must be
served by the row that was trained on the string it now emits — fixing one such mismatch in the
swap branch improved all seven domains) and **corpus attestation** (a rename or a glyph transfer
is dropped when the zh-TW corpus evidences the other form, which is what kills the classic
per-word OpenCC failure modes).

## Use it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

m, t = "Luigi/feng-30m-zh-tw", "Luigi/feng-30m-zh-tw"
tok = AutoTokenizer.from_pretrained(m)
net = AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True)
ids = tok([{"role": "user", "content": "請用繁體中文介紹Taipei 101。"}], return_tensors="pt")["input_ids"]
print(tok.decode(net.generate(ids, max_new_tokens=128, do_sample=False)[0], skip_special_tokens=True))
```

## Reproduce it

`transform/build_tw.py` regenerates every file in this repo from the parent checkpoint:

```bash
python build_tw.py --base /path/to/feng-30m --out ./feng-30m-zh-tw
```

It prints the accepted decisions (`word-swaps 671  word-renames 2275  word-leaks 1207
char-twins 441`) and writes `manifest.json`, which records the parent SHA-256, the exact knob
values, the row/scalar counts and the algebra. `transform/eval_ppl.py` reproduces any row of the
tables above.

## Limitations

* **It is no longer a good zh-CN model** — that is the point of the transform (zh-CN PPL
  220 → 769). Use the parent for Simplified Chinese.
* The parent's `esp32/` export is **intentionally not copied**: it is a quantisation of the
  parent's embedding and is stale for this model. Re-export it from `model.safetensors` here.
* Gains are domain-dependent: culture/food text (val5) is nearly at the parent's level, because
  the character transfer that makes spelling native costs a little in-domain fluency there.
* Vocab and tokenizer merges are the parent's; ~21 % of merge rules drop out of the reachable
  set after renaming, and vocabulary extension was measured to be harmful (it breaks
  tokenisation), so untokenised zh-TW words remain character-split.

## Lineage / credits

Derived from [jiaheng0815/feng-30m](https://huggingface.co/jiaheng0815/feng-30m)
([code](https://github.com/jiaheng0815/feng-30m)) — the parent model, its architecture, its
tokenizer and its Apache-2.0 license. Glyph mapping from Unihan
(`kTraditionalVariant` / `kSimplifiedVariant` / `kTotalScore`).
