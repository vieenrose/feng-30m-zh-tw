# feng-30m

[![CI](https://github.com/jiaheng0815/feng-30m/actions/workflows/ci.yml/badge.svg)](https://github.com/jiaheng0815/feng-30m/actions/workflows/ci.yml)

**English** | [简体中文](README_CN.md)

**A 29.43M-parameter Chinese chat model: trained from scratch, native 32k context; after Q4 quantization it runs on an ESP32-S3,
where the four tools — arithmetic / time / random / memory — are answered deterministically by a strict C23 engine on the board
(memory is persisted in NVS and survives power loss).**

> **Current release (updated 2026-10-18)**
>
> - Weights: PC = **4.0** (`v4/patch_name_e5_lr5e-6`) ｜ phone / ESP32 = **4.0-embed** (`board/model_export_4.0-embed/`)
> - Engine: **v4.0 / strict C23** (embedded side: zero heap, zero global constructors; PC side: C++23 + CUDA); board firmware **282,544 B**, 21,648 B smaller than the C++23 build in v3.20
> - On board: q2/2048 matrix **27/27 + 4/4**, tools **13/13**, memory **24/24**, **2.33 tok/s** (firmware counter, best short-reply figure)
> - Board weights 4.0-embed: PPL **102.99**, held-out 200 **85/200**, long context **108/128**; same size 14.93MB
> - Identity line (verbatim): 「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」
> - Teacher data: 36,916 prompts → **34,143 accepted / 2,773 rejected** (4.0 distillation run)
> - Code and weights are **Apache-2.0**; **GGUF releases stopped at v3.14** (the llama.cpp path has no tools)
>
> **4.0-embed is not a drop-in replacement for v3.19-embed, nor a QAT version of PC 4.0**:
> it combines 4.0B's attention/FFN with v3.19-embed's tied embedding/lm_head,
> fixing 4.0B's Q4 translation and 25% retrieval failures while keeping the PPL and long-context gains.
> Naming rules and experiment records: [`AGENTS.md`](AGENTS.md) and [`docs/board-qat.md`](docs/board-qat.md).

## Contents

- [Model Specifications](#model-specifications)
- [What It Can Do (Measured)](#what-it-can-do-measured)
- [Quick Start](#quick-start)
- [Standard Benchmarks (English, 0-shot)](#standard-benchmarks-english-0-shot)
- [Downloads](#downloads)
- [Version Overview](#version-overview)
- [Verification & CI](#verification--ci)
- [How Training Was Done](#how-training-was-done)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)
- [License](#license)

## Model Specifications

| Item | Value |
|---|---|
| Architecture | Qwen3 structure (RMSNorm / QK-norm / RoPE / SwiGLU / tied embedding) |
| Layers / width | 11 layers / hidden 448 / FFN 896 |
| Attention | 7-head MHA (7 Q heads = 7 KV heads), head_dim 64 |
| Vocabulary | 16384 (self-trained BPE), special tokens `<|im_start|>` `<|im_end|>` `<|endoftext|>` `<|pad|>` |
| Parameters | 29.43M (tied embedding) |
| Context | training 32768 (`rope_theta=1e6`, no RoPE interpolation); board 2048 (q2 KV, 9.62 MB PSRAM) |
| Quantization | board Q4 block64 (4.25 bpw) weights + q2 block8 KV; historical GGUF supported Q4_K_M / Q8_0 / f16 |
| Board model | `model.bin` 14.93 MB + `tokenizer.bin` 413 KB (streamed from flash mmap) |

Positioning: **Chinese daily conversation + embedded offline deployment**. It does not aim for general capability —
complex reasoning, expert Q&A, and English tasks are weak spots within the capacity limit
(see [Known Limitations](#known-limitations) and [Standard Benchmarks](#standard-benchmarks-english-0-shot)).

## What It Can Do (Measured)

### Conversation and long context (4.0 / PC, fp32)

| Evaluation | v3.19 | **4.0** | Source |
|---|---|---|---|
| Held-out 30 (dev set, strict-v7) | 17/30 | **18/30** | `eval/h1_patch_name_e5_lr5e-6.json` |
| Held-out 30 (**independent set**, used for model selection in v4, see note) | 15/30 | **16/30** | `eval/h2_patch_name_e5_lr5e-6.json` |
| 42-item broad daily probe | **42/42** | 40/42 (see below) | `eval/chat_probe_v4_pn5.json` |
| Needle retrieval (32 per length, single category) | 106/128 | 103/128 | `eval/longctx32_v319.json` / `eval/longctx32_v4final.json` |
| 　└ 32k alone | 24/32 | 20/32 | same as above |
| "Not in the text" refusal | 62/64 | **64/64** | same as above (`neg_hit`/`neg_n`) |
| Scope / identity | 10/10 / 12/12 | 10/10 | `eval/v4_scope_pn5.json` |

> **Long-context numbers were re-measured on 2026-10-18**: the old script treated `ctx` as a **character count**,
> so inputs labeled 4k/8k/16k/32k were actually only about 3k/6k/13k/26k tokens — the old table's 110/128 and
> 32k 28/32 were too optimistic. Inputs are now built exactly by token count with 32 generation slots reserved;
> under the same protocol **v3.19 = 106/128 (32k 24/32), 4.0 = 103/128 (32k 20/32)**. 4.0's retrieval **does not
> exceed** v3.19; what it gained is refusal 62→64/64.
> Reproduce: `python scripts\eval_longctx_many.py --models "v4=v4/patch_name_e5_lr5e-6" --n 32 --neg-n 16 --batch 1`.

> **The 2 probe failures**: one is a scoring artifact (for 「我很伤心」 it answered
> 「我在呢。难过的时候不用自己扛着。」, but the scorer only looks for the word 「伤心」);
> the other is a **genuine arithmetic error** — 「2+3等于几」 answered 6. The latter is **deliberate**:
> since v3.14 the training data contains no pure arithmetic, and arithmetic is answered deterministically
> by the `feng_calc` tool; the probe script bypasses tools, so the failure shows up. All 10 v4 versions
> have exactly these two, word for word.

### The four tools (board firmware / PC C++ engine / Python, identical behavior in all three)

| Tool | Capability | Measured |
|---|---|---|
| Arithmetic | multi-digit / decimals / parentheses / Chinese numerals / percent, plus **sequence counting** (「把 1 到 5 倒着数一遍」 → `5、4、3、2、1。`) | answers in 0.5 s, `4854+4411=9265` |
| Time | host SNTP sync + `\settime`; UTC+8 calendar, timestamps, N days before/after, clock arithmetic | within ≤3 s of NTP on board |
| Random | xorshift64*, seed = uptime×1.54×1000, discard the first and take the second; dice / coin / ranges | 3 consecutive runs all distinct |
| Memory | extracts facts from `我叫X / 最喜欢Y / 住在Z / 养了W` and generic key-value slots, answers follow-ups deterministically; `\mem` to view, `\reset` to clear | **12/12** on board, identity never confused |

Board tool suite **13/13**, memory 12 questions **12/12** (`logs/board_tools_cpp23_q2_final.txt`,
`logs/board_memory_cpp23_q2_final12.txt`). The 8-turn identity sequence **8/8** is handled by the engine
memory tool, see `logs/board_v3_16p3_identity_ctx_engmem.txt`.
The model no longer learns arithmetic (pure arithmetic samples are filtered from training data by the tool's recognizer).

### Embedded (**engine v4.0 + weights 4.0-embed** / q2 2048 firmware)

> The engine is 4.0 (strict C23); phone/ESP32 weights are **4.0-embed**.
> 4.0-embed was released from test build `4.0-20261018H`: 4.0B's attention/FFN +
> v3.19-embed's tied embedding/lm_head, Q4 gate 27/27 + 4/4.
> Detailed comparison in [`docs/board-qat.md`](docs/board-qat.md).

| Item | Result |
|---|---|
| 32-item matrix + long-context recall | **27/27 + 4/4** (PC C engine q2/2048, input ≈1525 tokens; 4 math questions answered by tools) |
| Arithmetic subset | **21/21** (arithmetic goes through tools) |
| Speed | **2.33 tok/s** (firmware counter, best short-context/short-reply figure; per-token varies with context/reply length) |
| Memory | weights flash mmap (no PSRAM); KV + activations 9.62 MB PSRAM; internal SRAM only kernel and workspace |

**Re-tested on device 2026-10-18** (firmware 280,736 B + 4.0-embed; engine hardened 2026-10-22, firmware was 281,248 B): tools **13/13**,
memory **24/24**, 10-turn multi-turn **10/10**; the identity line is verbatim
「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」.

### Engine (v4.0 / strict C23)

- **Strict C23 core** (`-std=c23`, shared by board and PC) + **C++23 drivers** (`pc/*.cpp`) and **CUDA** (`feng_cuda.cu`, nvcc); two-pass compilation (the two `/std:` flags are mutually exclusive, see AGENTS.md)
- **Zero heap**: the 8 core objects contain no `malloc/free/new/delete`, no exception/RTTI symbols; the only allocation is the tokenizer's one-time PSRAM load at startup; `.init_array` is empty for all components (no global constructors)
- **Size**: 282,544 B, 24,848 B smaller than the C version (307,392 B); non-hot modules `-Os`, hot inference paths stay `-O2`
- **Performance**: board short context 2.33 tok/s, q2/2048 single forward **1.58 s** (two cores split by head; single core 2.20 s); **PC CPU multithreaded 654 tok/s** (157 single-thread → 4.2×,
  `build_pc_chat.ps1` enables OpenMP by default, row-level parallelism, byte-identical output); CUDA build generates 322 tok/s;
  C/Python tool parity **173 cases** + 4 unit-test suites all pass

## Quick Start

### 1) PC: C++ engine (recommended, ships with all four tools)

Download `feng-30m-v4.0-engine.zip` (engine source) and `feng-30m-v4.0-embed-release.zip`
(pre-exported board weights in `weights/esp32/`), extract to adjacent directories:

```powershell
.\pc\build_pc_chat.ps1                  # Windows defaults to two-pass MSVC: core /std:clatest, driver /std:c++latest; -MinGW fallback
.\pc\pc_chat_q2b8.exe ..\feng-30m-v4.0-embed\weights\esp32
# you> 4854+4411          -> [tool] 4854 加 4411 等于 9265。
# you> 现在的时间戳是多少？  -> [tool] 时间戳：1791134671 —— 2026年10月05日 01:24:31（周一，UTC+8）。
# you> 我叫小明，请记住     -> 模型正常回应
# you> 我叫什么名字？       -> [tool] 你叫小明。（0.5 秒秒回）
```

> The engine's `model.bin` is **Q4 weights + q2 KV** format and must come from 4.0-embed's exported `weights/esp32/`
> (the pre-exported bundle already is). **PC HF weights were not quantization-aware trained; exporting them to the engine
> degrades noticeably** (same 32-item matrix: 22/27 vs 27/27, `logs/pc_kv_suite32_v3_14pc2_q2b8.txt`).

#### Optional: CUDA acceleration (RTX GPUs)

The PC engine has an **optional CUDA backend**: GEMV (Q4 block64 / fp16) runs on the GPU with weights resident in VRAM,
other ops stay on CPU; board firmware does not include this path. Requires VS Build Tools + CUDA Toolkit:

```powershell
cd pc
.\build_pc_cuda.ps1                       # MSVC(cl/link) + nvcc, produces *_cuda.exe
.\pc_bench_cuda.exe ..\board\model_export_4.0-embed "你好" 32
```

Measured (RTX 5060 Ti / q2 KV / 32 tokens): generation **157 → 322 tok/s (~2.0×)**;
prefill actually gets slower (59 → 110 ms / 12 tokens, fixed PCIe round-trip cost per GEMV),
best for "long-answer generation" workloads. The 32-item matrix output is **byte-identical** to the CPU build,
`pc_check` argmax MATCH; `FENG_CUDA=0` forces the same binary down the CPU path for A/B
(`pc/feng_cuda.cu`, `build_pc_cuda.ps1`).

### 2) HF weights + transformers

```python
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

path = "weights/hf"          # the extracted directory
tok = AutoTokenizer.from_pretrained(path)
model = Qwen3ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

ids = tok.apply_chat_template([{"role": "user", "content": "你是谁？"}],
                              add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=96, do_sample=False,
                     repetition_penalty=1.25, no_repeat_ngram_size=6,
                     pad_token_id=3, eos_token_id=0)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

### 3) ESP32-S3 (WROOM-2-N32R16V)

Compilation, flashing, and the serial protocol are documented in [`docs/USAGE.md`](docs/USAGE.md) and
[`docs/esp32-firmware.md`](docs/esp32-firmware.md); for a no-IDF, all-in-one firmware
see the Release `feng-30m-v4.0-embed-firmware.zip` (firmware + 4.0-embed weights).

## Standard Benchmarks (English, 0-shot)

We use EleutherAI lm-evaluation-harness v0.4.13 for **SciQ** and **PIQA**, fp32,
`prefix_token_id=0` (the self-trained tokenizer has no bos/eos).
Reproduce: `python scripts/bench_standard.py --model v4/patch_name_e5_lr5e-6 --tag v4_pn5`.

| Benchmark | Metric | v3.19 | **4.0** | Change |
|---|---|---|---|---|
| SciQ | Accuracy | **71.00%** ±1.44 | 70.80% ±1.44 | −0.20 |
| SciQ | Normalized | **73.80%** ±1.39 | 71.20% ±1.43 | **−2.60** |
| PIQA | Accuracy | **54.52%** ±1.16 | 54.03% ±1.16 | −0.49 |
| PIQA | Normalized | 53.37% ±1.16 | **54.19%** ±1.16 | +0.82 |

> **Three of four metrics regressed; only PIQA acc_norm rose, by 0.82.** Stated here as-is.
> After retrieval replay and the name patch, English benchmarks dropped — the SFT stage (`v4/sft_pc2`) was
> 72.50 / 72.60 / 54.41 / 54.46, the final weights are 70.80 / 71.20 / 54.03 / 54.19.
> Run once, without repeated sampling, **so we cannot say whether this exceeds noise and do not write "within noise"**.
> What the same period gained: strict-v7 held-out sets each **+1** (17→18, 15→16), refusal +2 (62→64/64);
> the old strict-v6 +3 / +4 figures are withdrawn.

> This is an **English** benchmark, while feng-30m is a 30M Chinese teacher-distilled model: SciQ is clearly above
> the 4-way random baseline, PIQA slightly above the 2-way baseline. **The other six tasks were removed from the suite** —
> ARC / HellaSwag / OpenBookQA / Winogrande / BoolQ / SciCloze all sit at the random line at this scale
> (BoolQ 37.8% is even below 2-way random, only a yes/no bias); they cannot distinguish capability and would
> only obscure the real level. Raw results are archived in `eval/lm_eval_feng_v3_19_pc4.json`.

## Downloads

Model weights can also be pulled directly from model hubs, **without unzipping a Release**:

| Platform | URL | Notes |
|---|---|---|
| **Hugging Face** | [`jiaheng0815/feng-30m`](https://huggingface.co/jiaheng0815/feng-30m) | load with `AutoTokenizer` + `Qwen3ForCausalLM`; the same repo's esp32 subdirectory contains the board `model.bin` |
| **ModelScope** | [`jiaheng0815/feng-30m`](https://www.modelscope.cn/models/jiaheng0815/feng-30m) | same as above, **direct from China** |

```python
from transformers import AutoTokenizer, Qwen3ForCausalLM
tok = AutoTokenizer.from_pretrained("jiaheng0815/feng-30m")
model = Qwen3ForCausalLM.from_pretrained("jiaheng0815/feng-30m")
```

Weights (fp32 / board model) and the distilled dataset are also packaged in
**[Releases](https://github.com/jiaheng0815/feng-30m/releases)**:

| Asset (v4.0) | Contents |
|---|---|
| `feng-30m-v4.0-release.zip` | PC: HF **4.0** weights + teacher-distilled dataset + model card |
| `feng-30m-v4.0-embed-release.zip` | Phone/ESP32: **4.0-embed** Q4 weights + q2 KV + `model.bin` / `tokenizer.bin` |
| `feng-30m-v4.0-engine.zip` | Engine source: strict C23 core + C++23 PC driver + board (with tools) |
| `feng-30m-v4.0-embed-firmware.zip` | Board all-in-one: prebuilt firmware + model + `partitions.csv` + flashing guide |

> The v4.0 in package names is the engine version; `weights/esp32/` is **4.0-embed** (Q4 weights,
> embedding/lm_head from v3.19-embed), while `weights/hf/` in the PC package is **4.0**.

**This repository only ships code and docs; training data and weights are not committed**; older assets
(v3.14 / v3.16-embed) remain on their corresponding Release pages.

## Version Overview

| | v3.10 | v3.11 | v3.12 | v3.13 | v3.14 | **v3.19 (historical)** |
|---|---|---|---|---|---|---|
| Cumulative training | v3.9 + ~6M KV-QAT | v3.10 + ~14M dual QAT (weights+KV) | v3.9 + ~12M last-layer arithmetic finetune | v3.12/v3.11 + ~12M memory mix | v3.13/v3.9 base + **~14M no-arithmetic mix** (tools take over arithmetic) | v3.14/pc2 + **440 targeted teacher samples** (last 2 layers, 15 ep) |
| Identity line | same (12/12) | same (12/12) | same (12/12) | same (12/12) | same (12/12) | **same (12/12)** |
| In-scope eval | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | **10/10** |
| Daily probe (42 items) | 42/42 | 42/42 | 42/42 | 42/42 | 42/42 | **42/42** |
| Needle retrieval · single category (4k/8k/16k/32k) | 28/30/28/21 | 29/30/27/20 | 28/29/26/27 | 28/29/29/27 | 28/28/25/27 | **28/30/26/26 = 110** |
| Needle retrieval · multi-category | 94 | 90 | 108 | 108 | 107 | **105** |
| Multi-turn memory (PC 24 / board 12 sampled) | — | — | 5/24 | 21/24 ｜ board 6/6 sampled | 24/24 ｜ board 10/12 | **23/24 ｜ board 12/12** |
| Arithmetic / time / random | model computes | model computes | model computes | model computes | engine tools: 0.5s all correct (incl. multi-digit/decimals) | **same (tools)** |
| Embedded 32-item matrix (q2 KV) | 27/27+4/4 | 27/27+4/4 | — | 27/27+4/4 | 27/27+4/4 | **27/27+4/4** |
| Board arithmetic subset 21 items (q2) | 12/21 | 21/21 | — | 21/21 | 21/21 (math answered by tools) | **21/21** |
| ESP32-S3 device | 10/10 + 10/10 | 30/30 + memory 6/6 | — | 30/30 + memory 6/6 | tools 8/8 + 30/30 + memory 10/12, 1.80 tok/s | ✅ **tools 13/13 + memory 12/12 + identity sequence 8/8** |

> Same-protocol re-tests (result JSONs in `eval/`): in-scope eval v3.7/v3.9~v3.12 = 10/10, v3.8 = 8/10,
> v3.6 = 8/10, v3.5 = 7/10, v3.4 = 9/10; 42-item probe v3.4 = 27/42, v3.5 = 29/42, 42/42 from v3.6 on.
> **The held-out 30 is a dev set** (used across many experiments), only for version comparison; on the
> **independent held-out 30**, v3.14/pc2, v3.19/pc4 and v3.19/board6 are all **18/30** — v3.19's dev-set
> gain (16→19) did not reproduce on the independent set, stated as-is. Historical best: single-category 113
> (v3.4 / v3.13 tied).

## Verification & CI

Local one-shot acceptance (doc checks → two-pass PC engine build + four unit suites → 32-item matrix + arithmetic subset → fp32 reference logits):

```powershell
python tools\check_all.py
```

CI (GitHub Actions, on every push/PR) runs the full chain on a **clean clone** — no local weights/logs needed:

1. Install GCC 14 (the default gcc 13 on ubuntu-latest only knows `-std=c2x`, **not `-std=c23`**, so it must be installed explicitly)
2. Python syntax check (all scripts)
3. Build the C23 core (9 object files, `gcc-14 -std=c23`)
4. Four unit suites: arithmetic 41 / time+random 53 / memory 69 / sampler 4
5. C/Python arithmetic+sequence parity, 29 cases (`tools/check_tool_parity.py`)
6. C/Python memory tool parity, 36 rounds (`tools/check_mem_parity.py`)
7. C/Python random parity, 56 groups (`tools/check_rand_parity.py`)
8. C/Python time tool parity, 52 cases (`tools/check_time_parity.py`)
9. PC engine build (C23 core `.o` + C++23 drivers: pc_chat / pc_kv_suite / pc_mt_suite / pc_check)
10. Phone engine cross-compile (`phone/cli/build.sh`, reusing the same C23 core)
11. Doc checks (`tools/check_md.py` + `tools/check_docs.py`; local product paths may be missing and are counted)

> The four C/Python suites total **173 cases** compared one by one; weights/logs/build artifacts referenced by
> docs are classified `local-only` and skipped when absent locally, but missing tracked files still fail.

## How Training Was Done

| Stage | What | Data | Script |
|---|---|---|---|
| Pretraining (v2) | Chinese Wikipedia + firefly, seq 2048 | 1,496M tokens | `scripts/v2_build.py`, `scripts/v2_train.py` |
| Plan A SFT (v2) | 27B teacher behavior data ×6 + identity ×25 + filtered real chit-chat, then low-LR finetune for 1000 steps | 94,478 segments / 22.5M tokens | `scripts/build_planA_corpus.py`, `scripts/v2_train.py` |
| Progressive long context (v3) | same token stream cut into 4k→8k→16k→32k windows, data volume decreasing with length | 72M tokens | `scripts/v3_build_stages.py`, `scripts/v3_train.py` |
| Long-context SFT | Plan A corpus repacked into 8192 windows so short sequences don't pull the window back down | 22.65M tokens (17.86M supervised) | `scripts/v3_pack_sft.py`, `scripts/v3_polish.py` |
| Synthetic retrieval SFT | facts buried in long text, loss only on the answer (key step: long text alone doesn't teach retrieval) | 12.5M tokens | `scripts/v3_build_retrieval.py`, `scripts/v3_retrieval_sft.py` |
| v3.5 multi-turn fix | 2,600 multi-turn dialogues × high mix ratio + retrieval reinforcement, fixes "repetition from turn 3" | 28.0M tokens | `scripts/v3_5_build_multiturn.py`, `scripts/chat_multi.py` |
| v3.6 daily patch | 592 daily dialogues + systematic small-number arithmetic (×6), single-sample SFT then retrieval top-up | 38.0M tokens + patch rounds | `scripts/v3_6_build_daily_patch.py`, `scripts/v3_6_sft_patch.py` |
| v3.7 KV-QAT | q2 block8 KV noise injected during training; long-context recall and stock refusal topped up | ~10M tokens | `scripts/v3_7_build_qat_data.py`, `scripts/v3_7_kv_qat.py` |
| v3.8 context upgrade | retrieval oversampling + soup + **last-layer finetune** to fix conversation | ~15M | `scripts/soup_models.py`, `scripts/v3_8_build_fix_data.py` |
| v3.9 zero-cost behavior fix | last-layer finetune (last 2 layers + norm) with 40× stock-refusal data to fix in-scope eval | ≤1M tokens | `scripts/v3_6_sft_patch.py --train-last 2` |
| v3.10~v3.11 embedded QAT | q2 KV-QAT → arithmetic boundary patch → **Q4 weights/q2 KV dual QAT** | ~20M tokens | `scripts/v3_7_kv_qat.py --wqat`, `scripts/v3_11_build_arith_patch.py` |
| v3.12~v3.13 arithmetic/memory | arithmetic via last-layer finetune; 5,600 synthetic memory dialogues (PC last-layer / board dual QAT) | ~24M tokens | `scripts/v3_11_build_repair.py --pc-fix`, `scripts/v3_13_build_memory.py` |
| v3.14 tool build | arithmetic/time/random become engine tools; pure arithmetic samples filtered from training; GGUF distribution cancelled | ~14M tokens (no arithmetic mix) | `scripts/v3_14_build_noarith_mix.py`, `core/feng_calc.c` |
| v3.15~v3.19 polish | identity context anchors → board name-confusion fix experiments (rejected routes recorded) → engine memory tool → 440 targeted samples, last 2 layers 15ep | ~12M tokens | `scripts/v3_15_build_identity_ctx.py`, `scripts/v3_19_build_mix.py` |
| v3.20 engine | sequence-counting tool + unified sampler; whole project moved to C++23 (zero heap / no exceptions / RTTI) with size and performance work | tools are C++ zero-heap, no weight cost | `board/main/`, `docs/CHANGELOG.md` v3.20 section |
| **v4.0 engine** | engine core **moved to strict C23** (shared by board + PC, PC drivers still C++23, CUDA backend separate); adds phone client and editor code completion | same as above, numeric path byte-identical | `board/main/`, `phone/`, `docs/esp32-firmware.md` |

Full hyperparameters, per-stage loss/time/VRAM are in [`docs/CHANGELOG.md`](docs/CHANGELOG.md) and
[`docs/DELIVERY.md`](docs/DELIVERY.md).

## Repository Structure

```
README.md              repository home page (English, this file)
README_CN.md           repository home page (Chinese)
AGENTS.md              working guide for AI coding agents: conventions, hard constraints, command cheatsheet
docs/                  all documentation (index in docs/README.md)
v4/base/               current weights, also the 4.0 training starting point (HF format, ships its tokenizer)
scripts/               data build / training / eval / export scripts (path resolution in scripts/paths.py)
board/      ESP32 firmware + portable C23 inference engine + PC consistency checks and regression suites
phone/                 phone client (Termux CLI + Android APK), reuses the same C23 engine
data/                  prompt pool / teacher-distilled data / public corpora (not committed)
eval/                  eval result JSON (scope, needle retrieval, memory, held-out, standard benchmarks)
logs/                  build / train / flash / board-test logs (board_baseline_lut.txt is the board accuracy baseline)
tools/check_md.py      doc self-check (code fences, paths, stale numbers)
tools/check_docs.py    doc fact check (model spec / eval numbers aligned with real artifacts)
tools/check_all.py     one-shot acceptance: docs → build → unit tests → 32-item matrix → bit-exact
```

The repository keeps only the v4 line: historical v1 / v2 / v3 version directories were removed in 2026-10;
fetch old weights from the corresponding Release.

## Configuration

Scripts contain **no hard-coded drive letters**: the project root is derived from the script location, and
external dependencies resolve in the order **environment variable > `scripts/local_paths.json` > default**:

| Environment variable | Purpose |
|---|---|
| `FENG_ROOT` | project root (default: the script's parent directory) |
| `FENG_PY` | Python interpreter (default: current interpreter) |
| `FENG_DATA_DIR` | raw corpus directory (**only needed to rebuild v1/v2 corpora**) |
| `FENG_LLAMA_DIR` | llama.cpp repo directory (GGUF conversion / benchmark, optional) |
| `FENG_GXX` | MinGW g++ path (only used by the `-MinGW` fallback; the default build is MSVC, see `build_pc_chat.ps1`) |

Copy `scripts/local_paths.example.json` to `scripts/local_paths.json` and fill in your own paths (the latter is gitignored).
Self-check: `python scripts/paths.py`, which prints whether each path resolves.

## Known Limitations

- **30M capacity ceiling**: free-form questions outside coverage will be off or fabricated; complex reasoning,
  expert domains, and English tasks are unreliable (standard benchmarks mostly at the random line, see above).
- **Recommendations only cover the training distribution**: book/movie recommendations are written item by item
  in patches; changing a title or asking for the latest list will expose it.
- **Board context is 2048 (q2 KV)**; 32k needs the HF/Python path, or rebuild the PC C engine yourself with
  `-DFENG_MAX_SCORES=32768` (`pc_chat --ctx`); 32k on board is impossible under PSRAM/KV memory limits.
- **PC weights and board weights are not interchangeable**: PC HF weights were not quantization-aware trained;
  importing them into the engine degrades noticeably (32-item matrix 22/27 vs 27/27); phone/ESP32 use 4.0-embed
  (`weights/esp32/` in `feng-30m-v4.0-embed-release.zip`).
- **The held-out 30 is a dev set** (strict-v7: v3.19=17/30, v4.0=18/30); the second set was used for model
  selection in v4 and is no longer an independent test set (strict-v7: v3.19=15/30, v4.0=16/30).
- **Memory only covers enumerable patterns**: `\reset` / `\mem clear` / full context clear it; board writes to NVS,
  phone writes `mem.txt` in the model directory, surviving power cycles (PC `pc_chat` is still RAM-only).
- **Board time needs host sync**: serial scripts send `\settime` (host NTP) on connect, otherwise the time tool
  reports unsynced.
- **Board speed is about 2.3 tok/s** (best short-reply figure; long answers/long context are slower, see
  `logs/board_v3_19b6_stress32.txt`); the Q4 GEMV inner loop is measured at this chip's limit (cycle-level
  profiling and five rejected optimization candidates in `docs/CHANGELOG.md` v4.0 board performance section);
  only a new chip or a smaller model can improve it.
- Quantization resistance is very sensitive to weight interpolation: mixing in 20% old weights already drops
  points (CHANGELOG v3.10); to change board behavior use the "add data + weights/KV dual QAT" pipeline,
  do not manually soup.
- ESP32 firmware model partition offsets must match `board/partitions.csv`.

## Acknowledgements

**Teacher model** — **[UkisAI](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)'s
Swift Bonsai 2 (27B)**. All conversation data since v2 is distilled from it (v1 used a finetuned Qwen3.5-0.8B),
served locally through the same project's **NInfer** inference engine (`.ninfer` weight format).

**Student architecture** — **Qwen3**. The 11-layer / hidden 448 / 7-head MHA, RoPE, RMSNorm and
tied embedding structure follow the Qwen3 series design.

**Public corpora** — 4.0's prompt pool comes from five public datasets: **ShareGPT Chinese**,
**Evol-Instruct Chinese**, **Alpaca GPT-4 Chinese**, **Databricks Dolly-15k**, **UltraChat-200k**.
Only the **questions** are used; all answers are regenerated by the teacher and decontaminated verbatim
against held-out items.

**Toolchain**

| Project | Used for |
|---|---|
| [ESP-IDF](https://github.com/espressif/esp-idf) | ESP32-S3 firmware (v5.5.5, xtensa gcc 14.2) |
| [PyTorch](https://pytorch.org/) | training and quantization-aware training |
| [HuggingFace transformers](https://github.com/huggingface/transformers) | weight format and tokenizer |
| [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | standard benchmarks (SciQ / PIQA) |

Thanks to the authors of the above projects and datasets for open-sourcing their work.

## License

Code and weights are **Apache-2.0** (see [`LICENSE`](LICENSE)). This is a personal project,
with no affiliation or endorsement by any model vendor.

---

<sub>There is a hidden sentence in the firmware.<br>
To disable the easter egg, enable the `nobonus` build option (`idf.py -DFENG_NO_BONUS=ON build`).</sub>
