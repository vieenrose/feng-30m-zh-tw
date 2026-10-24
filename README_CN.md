# feng-30m

[![CI](https://github.com/jiaheng0815/feng-30m/actions/workflows/ci.yml/badge.svg)](https://github.com/jiaheng0815/feng-30m/actions/workflows/ci.yml)

**简体中文** | [English](README.md)

**一个 29.43M 参数的中文对话模型：从零训练、原生 32k 上下文；Q4 量化后运行在 ESP32-S3 上，
算术 / 时间 / 随机数 / 记忆四个 tool 由板内严格 C23 引擎确定性作答（记忆落 NVS，断电不丢）。**

> **当前发布（2026-10-18 更新）**
>
> - 权重：PC = **4.0**（`v4/patch_name_e5_lr5e-6`）｜ 手机 / ESP32 = **4.0-embed**（`board/model_export_4.0-embed/`）
> - 引擎：**v4.0 / 严格 C23**（嵌入式侧，零堆、零全局构造；PC 侧 C++23 + CUDA）；板端固件 **282,544 B**，比 v3.20 的 C++23 版还小 21,648 B
> - 板端实测：q2/2048 矩阵 **27/27 + 4/4**、工具 **13/13**、记忆 **24/24**、**2.33 tok/s**（固件统计口径，短回复最佳值）
> - 板端权重 4.0-embed：PPL **102.99**、留出 200 **85/200**、长文 **108/128**；同尺寸 14.93MB
> - 身份自述：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**
> - 教师数据：36,916 条提示词 → **34,143 收 / 2,773 拒**（4.0 蒸馏链）
> - 代码与权重均 **Apache-2.0**；**GGUF 自 v3.14 起不再发行**（llama.cpp 路径没有 tool）
>
> **4.0-embed 不是 v3.19-embed 的直接替换，也不是 PC 4.0 的 QAT 版**：
> 它把 4.0B 的 attention/FFN 与 v3.19-embed 的 tied embedding/lm_head 组合，
> 解决了 4.0B 的 Q4 翻译/25% 检索失败，同时保住 PPL 与长文收益。
> 命名规则与实验记录见 [`AGENTS.md`](AGENTS.md) 与 [`docs/board-qat.md`](docs/board-qat.md)。

## 目录

- [模型规格](#模型规格)
- [现在能做什么（实测）](#现在能做什么实测)
- [快速开始](#快速开始)
- [标准基准（英文，0-shot）](#标准基准英文0-shot)
- [下载](#下载)
- [版本速览](#版本速览)
- [验证与 CI](#验证与-ci)
- [训练怎么做的](#训练怎么做的)
- [仓库结构](#仓库结构)
- [配置](#配置)
- [已知限制](#已知限制)
- [许可证](#许可证)

## 模型规格

| 项目 | 值 |
|---|---|
| 架构 | Qwen3 结构（RMSNorm / QK-norm / RoPE / SwiGLU / tied embedding） |
| 层数 / 宽度 | 11 层 / hidden 448 / FFN 896 |
| 注意力 | 7 头 MHA（7 Q 头 = 7 KV 头），head_dim 64 |
| 词表 | 16384（自训 BPE），特殊符 `<|im_start|>` `<|im_end|>` `<|endoftext|>` `<|pad|>` |
| 参数量 | 29.43M（tied embedding） |
| 上下文 | 训练 32768（`rope_theta=1e6`，无 RoPE 插值）；板端 2048（q2 KV，9.62 MB PSRAM） |
| 量化 | 板端 Q4 块64（4.25 bpw）权重 + q2 block8 KV；历史 GGUF 支持 Q4_K_M / Q8_0 / f16 |
| 板端模型 | `model.bin` 14.93 MB + `tokenizer.bin` 413 KB（flash mmap 流式读） |

定位：**中文日常对话 + 嵌入式离线部署**。它不追求通用能力——复杂推理、专业问答、
英文任务都是容量边界内的弱项（见[已知限制](#已知限制)与[标准基准](#标准基准英文0-shot)）。

## 现在能做什么（实测）

### 对话与长文（4.0 / PC，fp32）

| 评测 | v3.19 | **4.0** | 出处 |
|---|---|---|---|
| 留出 30 题（开发集，strict-v7） | 17/30 | **18/30** | `eval/h1_patch_name_e5_lr5e-6.json` |
| 留出 30 题（**独立集**，v4 用于选型，见注） | 15/30 | **16/30** | `eval/h2_patch_name_e5_lr5e-6.json` |
| 42 题广谱日常探针 | **42/42** | 40/42（见下） | `eval/chat_probe_v4_pn5.json` |
| 针检索（每长度 32 题，单类别） | 106/128 | 103/128 | `eval/longctx32_v319.json` / `eval/longctx32_v4final.json` |
| 　└ 32k 单档 | 24/32 | 20/32 | 同上 |
| 「文中没有」拒答 | 62/64 | **64/64** | 同上（`neg_hit`/`neg_n`） |
| 范围 / 身份 | 10/10 / 12/12 | 10/10 | `eval/v4_scope_pn5.json` |

> **长文数字 2026-10-18 重测过**：旧脚本把 `ctx` 当**字符数**，标称 4k/8k/16k/32k 的输入
> 实际只有约 3k/6k/13k/26k token，所以旧表里的 110/128、32k 28/32 偏乐观。
> 现在按 token 精确造输入并预留 32 个生成位；同一协议下 **v3.19 = 106/128（32k 24/32）、
> 4.0 = 103/128（32k 20/32）**。4.0 的检索**没有超过** v3.19，换来的是拒答 62→64/64。
> 复现：`python scripts\eval_longctx_many.py --models "v4=v4/patch_name_e5_lr5e-6" --n 32 --neg-n 16 --batch 1`。

> **探针那 2 题**：一条是判分误判（「我很伤心」答的是「我在呢。难过的时候不用自己扛着。」，
> 判分只找「伤心」）；另一条是**真的算错**——「2+3等于几」答成 6。后者是**故意放弃**的：
> v3.14 起训练数据不再含纯算式，算术由 `feng_calc` tool 确定性作答，探针脚本不经过 tool
> 才照得出来。全部 10 个 v4 版本都是这两条，一字不差。

### 四个 tool（板端固件 / PC C++ 引擎 / Python，三处同口径）

| tool | 能力 | 实测 |
|---|---|---|
| 算式 | 多位数 / 小数 / 括号 / 中文数字 / 百分号，以及 **序列数数**（"把 1 到 5 倒着数一遍" → `5、4、3、2、1。`） | 0.5 s 秒回，`4854+4411=9265` |
| 时间 | 宿主 SNTP 对时 + `\settime`；UTC+8 日历、时间戳、N 天前后、时钟推算 | 板端与 NTP 差 ≤3 s |
| 随机数 | xorshift64*，seed = 运行时间×1.54×1000，丢第一个取第二个；骰子 / 硬币 / 区间 | 连续 3 次不重复 |
| 记忆 | 从 `我叫X / 最喜欢Y / 住在Z / 养了W` 与通用键值槽抽事实，追问确定性作答；`\mem` 查看、`\reset` 清空 | 板端 **12/12**，身份永不串名 |

板端 tool 专项 **13/13**、记忆 12 题 **12/12**（`logs/board_tools_cpp23_q2_final.txt`、
`logs/board_memory_cpp23_q2_final12.txt`）。8 轮身份序列 **8/8** 由引擎记忆 tool 接管，
见 `logs/board_v3_16p3_identity_ctx_engmem.txt`。
模型不再学算术（训练数据用 tool 识别器过滤纯算式样本）。

### 嵌入式（**引擎 v4.0 + 权重 4.0-embed** / q2 2048 固件）

> 引擎是 4.0 的（严格 C23），手机/ESP32 权重是 **4.0-embed**。
> 4.0-embed 由 `4.0-20261018H` 测试版发布：4.0B 的 attention/FFN +
> v3.19-embed 的 tied embedding/lm_head，Q4 网关 27/27 + 4/4。
> 详细对比见 [`docs/board-qat.md`](docs/board-qat.md)。

| 项目 | 结果 |
|---|---|
| 32 题矩阵 + 长文召回 | **27/27 + 4/4**（PC C 引擎 q2/2048，输入约 1525 token；4 道数学题由 tool 回答） |
| 算术子集 | **21/21**（算式走 tool） |
| 速度 | **2.33 tok/s**（固件统计口径，短上下文/短回复最佳值；逐 token 随上下文/回复长度变化） |
| 内存 | 权重 flash mmap（不占 PSRAM）；KV + 激活 9.62 MB PSRAM；内部 SRAM 仅留内核与工作区 |

**2026-10-18 实机复测**（固件 280,736 B + 4.0-embed；2026-10-22 引擎加固后固件为 281,248 B）：tool **13/13**、
记忆 **24/24**、10 轮多轮 **10/10**，身份句逐字为
「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」。

### 引擎（v4.0 / 严格 C23）

- **严格 C23 核心**（`-std=c23`，板端与 PC 共用）+ **C++23 驱动**（`pc/*.cpp`）与 **CUDA**（`feng_cuda.cu`，nvcc）；两趟编译（两个 `/std:` 互斥，见 AGENTS.md）
- **零堆**：8 个核心对象无 `malloc/free/new/delete`、无异常/RTTI 符号；唯一分配是 tokenizer
  **启动期**一次性 PSRAM 加载；全组件 `.init_array` 为空（无全局构造）
- **体积**：282,544 B，比 C 版（307,392 B）小 24,848 B；非热点模块 `-Os`、推理热点保持 `-O2`
- **性能**：板端短上下文 2.33 tok/s，q2/2048 单次 forward **1.58 s**（双核按 head 切分；单核 2.20 s）；**PC CPU 多线程 654 tok/s**（单线程 157 → 4.2×，
  `build_pc_chat.ps1` 默认 OpenMP，行级并行、输出逐字一致）；CUDA 版生成 322 tok/s；
  C++/Python tool 一致性 **173 条** + 4 套单测全过

## 快速开始

### 1) PC：C++ 引擎（推荐，自带四个 tool）

下载 `feng-30m-v4.0-engine.zip`（引擎源码）与 `feng-30m-v4.0-embed-release.zip`
（板端预导出 `weights/esp32/`），解压到相邻目录后：

```powershell
.\pc\build_pc_chat.ps1                  # Windows 默认 MSVC 两趟：核心 /std:clatest、驱动 /std:c++latest；-MinGW 备用
.\pc\pc_chat_q2b8.exe ..\feng-30m-v4.0-embed\weights\esp32
# you> 4854+4411          -> [tool] 4854 加 4411 等于 9265。
# you> 现在的时间戳是多少？  -> [tool] 时间戳：1791134671 —— 2026年10月05日 01:24:31（周一，UTC+8）。
# you> 我叫小明，请记住     -> 模型正常回应
# you> 我叫什么名字？       -> [tool] 你叫小明。（0.5 秒秒回）
```

> 引擎的 `model.bin` 是 **Q4 权重 + q2 KV** 格式，必须用 4.0-embed 导出的 `weights/esp32/`
> （预导出包已做好）。**PC 的 HF 权重没做量化感知训练，导出给引擎会明显退化**
> （同套 32 题矩阵实测 22/27 vs 27/27，`logs/pc_kv_suite32_v3_14pc2_q2b8.txt`）。

#### 可选：CUDA 加速（RTX 显卡）

PC 引擎带一个**可选 CUDA 后端**：GEMV（Q4 block64 / fp16）交给 GPU，权重常驻显存，
其余算子仍在 CPU；板上固件不含此路径。构建需要 VS Build Tools + CUDA Toolkit：

```powershell
cd pc
.\build_pc_cuda.ps1                       # MSVC(cl/link) + nvcc，产出 *_cuda.exe
.\pc_bench_cuda.exe ..\board\model_export_4.0-embed "你好" 32
```

实测（RTX 5060 Ti / q2 KV / 32 token）：生成 **157 → 322 tok/s（约 2.0×）**；
prefill 反而变慢（59 → 110 ms / 12 token，每次 GEMV 的 PCIe 往返固定开销），
适合"长回答生成"为主的场景。32 题矩阵输出与 CPU 版 **逐字一致**，
`pc_check` argmax MATCH；`FENG_CUDA=0` 可让同一个二进制强制走 CPU 路径做对比
（`pc/feng_cuda.cu`、`build_pc_cuda.ps1`）。

### 2) HF 权重 + transformers

```python
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

path = "weights/hf"          # 解压后的目录
tok = AutoTokenizer.from_pretrained(path)
model = Qwen3ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

ids = tok.apply_chat_template([{"role": "user", "content": "你是谁？"}],
                              add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=96, do_sample=False,
                     repetition_penalty=1.25, no_repeat_ngram_size=6,
                     pad_token_id=3, eos_token_id=0)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

### 3) ESP32-S3（WROOM-2-N32R16V）

编译、烧录、串口协议的完整步骤见 [`docs/USAGE.md`](docs/USAGE.md) 与
[`docs/esp32-firmware.md`](docs/esp32-firmware.md)；免装 IDF 的一包到底固件
见 Release 的 `feng-30m-v4.0-embed-firmware.zip`（固件 + 4.0-embed 权重）。

## 标准基准（英文，0-shot）

用 EleutherAI lm-evaluation-harness v0.4.13 测 **SciQ** 与 **PIQA** 两项，fp32，
`prefix_token_id=0`（自训 tokenizer 没有 bos/eos）。
复现：`python scripts/bench_standard.py --model v4/patch_name_e5_lr5e-6 --tag v4_pn5`。

| 基准 | 指标 | v3.19 | **4.0** | 变化 |
|---|---|---|---|---|
| SciQ | Accuracy | **71.00%** ±1.44 | 70.80% ±1.44 | −0.20 |
| SciQ | Normalized | **73.80%** ±1.39 | 71.20% ±1.43 | **−2.60** |
| PIQA | Accuracy | **54.52%** ±1.16 | 54.03% ±1.16 | −0.49 |
| PIQA | Normalized | 53.37% ±1.16 | **54.19%** ±1.16 | +0.82 |

> **四项里三项退步，只有 PIQA acc_norm 涨了 0.82。** 如实写在这里。
> 检索回放和名字补丁之后，英文基准掉了——SFT 阶段（`v4/sft_pc2`）那一组是
> 72.50 / 72.60 / 54.41 / 54.46，最终权重是 70.80 / 71.20 / 54.03 / 54.19。
> 只跑了一次、没有多次采样，**无法判断是否超出噪声，所以不写「在噪声范围内」**。
> 同期换来的收益是：strict-v7 留出两套各 **+1**（17→18、15→16）、拒答 +2（62→64/64）；
> 旧 strict-v6 口径写的 +3 / +4 已作废。

> 这是**英文**基准，而 feng-30m 是中文教师蒸馏的 30M 模型：SciQ 明显高于 4 选 1 随机线，
> PIQA 略高于 2 选 1 随机线。**其余六项已从套件移除**——ARC / HellaSwag / OpenBookQA /
> Winogrande / BoolQ / SciCloze 在这个规模上全部贴随机线（BoolQ 37.8% 甚至低于二选一随机，
> 只有 yes/no 偏置），测不出能力差异，堆上去只会盖住真实水平。原始结果仍留档在
> `eval/lm_eval_feng_v3_19_pc4.json`。

## 下载

模型权重也可以直接从模型平台拉，**不用解压 Release 包**：

| 平台 | 地址 | 说明 |
|---|---|---|
| **Hugging Face** | [`jiaheng0815/feng-30m`](https://huggingface.co/jiaheng0815/feng-30m) | `AutoTokenizer` + `Qwen3ForCausalLM` 直接加载；同仓库的 esp32 子目录含板端 `model.bin` |
| **ModelScope 魔搭** | [`jiaheng0815/feng-30m`](https://www.modelscope.cn/models/jiaheng0815/feng-30m) | 同上，**国内直连** |

```python
from transformers import AutoTokenizer, Qwen3ForCausalLM
tok = AutoTokenizer.from_pretrained("jiaheng0815/feng-30m")
model = Qwen3ForCausalLM.from_pretrained("jiaheng0815/feng-30m")
```

权重（fp32 / 板端模型）与蒸馏数据集也打包在
**[Releases](https://github.com/jiaheng0815/feng-30m/releases)**：

| 附件（v4.0） | 内容 |
|---|---|
| `feng-30m-v4.0-release.zip` | PC 版：HF **4.0** 权重 + 教师蒸馏数据集 + 模型卡 |
| `feng-30m-v4.0-embed-release.zip` | 手机/ESP32 版：**4.0-embed** Q4 权重 + q2 KV + `model.bin` / `tokenizer.bin` |
| `feng-30m-v4.0-engine.zip` | 引擎源码：严格 C23 核心 + C++23 PC 驱动 + 板端（含 tool） |
| `feng-30m-v4.0-embed-firmware.zip` | 板端一包到底：预编译固件 + 模型 + `partitions.csv` + 烧录说明 |

> 包名里的 v4.0 是引擎版本；`weights/esp32/` 是 **4.0-embed**（Q4 权重，
> embedding/lm_head 来自 v3.19-embed），PC 包里的 `weights/hf/` 才是 **4.0**。

**本仓库只放代码与文档，训练数据与权重不入库**；旧附件（v3.14 / v3.16-embed）保留在对应 Release 页。

## 版本速览

| | v3.10 | v3.11 | v3.12 | v3.13 | v3.14 | **v3.19（历史）** |
|---|---|---|---|---|---|---|
| 累计训练量 | v3.9 + 约 6M KV-QAT | v3.10 + 约 14M 双 QAT（权重+KV） | v3.9 + 约 12M 末层算术微调 | v3.12/v3.11 + 约 12M 记忆混训 | v3.13/v3.9 底座 + **约 14M 无算术混训**（tool 接管算术） | v3.14/pc2 + **440 条定向教师数据**（末 2 层 15ep） |
| 身份自述 | 同（12/12） | 同（12/12） | 同（12/12） | 同（12/12） | 同（12/12） | **同（12/12）** |
| 范围内评测 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | **10/10** |
| 日常对话探针（42 题） | 42/42 | 42/42 | 42/42 | 42/42 | 42/42 | **42/42** |
| 针检索·单类别（4k/8k/16k/32k） | 28/30/28/21 | 29/30/27/20 | 28/29/26/27 | 28/29/29/27 | 28/28/25/27 | **28/30/26/26 = 110** |
| 针检索·多类别 | 94 | 90 | 108 | 108 | 107 | **105** |
| 多轮记忆（PC 24 题 / 板端 12 题抽测） | — | — | 5/24 | 21/24 ｜ 板端 6/6 抽查 | 24/24 ｜ 板端 10/12 | **23/24 ｜ 板端 12/12** |
| 算式 / 时间 / 随机数 | 模型硬算 | 模型硬算 | 模型硬算 | 模型硬算 | 引擎 tool：0.5s 全对（含多位数/小数） | **同（tool）** |
| 嵌入式 32 题矩阵（q2 KV） | 27/27+4/4 | 27/27+4/4 | — | 27/27+4/4 | 27/27+4/4 | **27/27+4/4** |
| 板端算术子集 21 题（q2） | 12/21 | 21/21 | — | 21/21 | 21/21（数学题由 tool 回答） | **21/21** |
| ESP32-S3 实机 | 10/10 + 10/10 | 30/30 + 记忆 6/6 | — | 30/30 + 记忆 6/6 | 工具 8/8 + 30/30 + 记忆 10/12，1.80 tok/s | ✅ **工具 13/13 + 记忆 12/12 + 身份序列 8/8** |

> 同口径复测记录（结果 JSON 均在 `eval/`）：范围内评测 v3.7/v3.9~v3.12 = 10/10、v3.8 = 8/10、
> v3.6 = 8/10、v3.5 = 7/10、v3.4 = 9/10；42 题探针 v3.4 = 27/42、v3.5 = 29/42、v3.6 起 42/42。
> **留出 30 题是开发集**（被多轮实验使用过），只作版本间对比；**独立留出 30 题**上
> v3.14/pc2、v3.19/pc4、v3.19/board6 均为 **18/30**——v3.19 的开发集增益（16→19）
> 在独立集上未复现，如实标注。历史最好：单类别 113（v3.4 / v3.13 并列）。

## 验证与 CI

本地一键验收（文档自检 → PC 引擎两趟构建 + 四套单测 → 32 题矩阵 + 算术子集 → fp32 参考 logits）：

```powershell
python tools\check_all.py
```

CI（GitHub Actions，每次 push/PR）在**干净 clone** 上跑完整链路——不依赖本机权重/日志：

1. 装 GCC 14（ubuntu-latest 默认的 gcc 13 只认 `-std=c2x`，**不认 `-std=c23`**，得显式装）
2. Python 语法检查（全部脚本）
3. 编译 C23 核心（9 个目标文件，`gcc-14 -std=c23`）
4. 单测四套：算式 41 / 时间随机数 53 / 记忆 69 / 采样器 4
5. C/Python 算式+序列一致性 29 条（`tools/check_tool_parity.py`）
6. C/Python 记忆 tool 一致性 36 回合（`tools/check_mem_parity.py`）
7. C/Python 随机数一致性 56 组（`tools/check_rand_parity.py`）
8. C/Python 时间 tool 一致性 52 条（`tools/check_time_parity.py`）
9. PC 引擎编译（C23 核心 `.o` + C++23 驱动：pc_chat / pc_kv_suite / pc_mt_suite / pc_check）
10. 手机端引擎交叉编译（`phone/cli/build.sh`，复用同一批 C23 核心）
11. 文档自检（`tools/check_md.py` + `tools/check_docs.py`，本机产物路径允许缺并计数）

> 四套 C/Python 交叉验证合计 **173 条**逐条对比；文档引用的权重/日志/编译产物只在本机存在时
> 会被归为 `local-only` 跳过，仓库跟踪文件缺失仍会报错。

## 训练怎么做的

| 阶段 | 内容 | 数据量 | 脚本 |
|---|---|---|---|
| 预训练（v2） | 中文维基 + firefly，seq 2048 | 1,496M tokens | `scripts/v2_build.py`、`scripts/v2_train.py` |
| Plan A SFT（v2） | 27B 教师行为数据 ×6 + 身份 ×25 + 过滤后真实闲聊，再低 LR 补训 1000 步 | 94,478 段 / 22.5M tokens | `scripts/build_planA_corpus.py`、`scripts/v2_train.py` |
| 渐进长文（v3） | 同一 token 流按 4k→8k→16k→32k 切窗，数据量随长度递减 | 72M tokens | `scripts/v3_build_stages.py`、`scripts/v3_train.py` |
| 长上下文 SFT | Plan A 语料重打包成 8192 窗口，避免短序列把窗口压回去 | 22.65M tokens（17.86M 有监督） | `scripts/v3_pack_sft.py`、`scripts/v3_polish.py` |
| 合成检索 SFT | 长文埋事实、只对答案算 loss（关键一步：只喂长文学不会检索） | 12.5M tokens | `scripts/v3_build_retrieval.py`、`scripts/v3_retrieval_sft.py` |
| v3.5 多轮修复 | 2,600 条多轮对话 × 高占比混训 + 检索补强，修"从第 3 轮起复读" | 28.0M tokens | `scripts/v3_5_build_multiturn.py`、`scripts/chat_multi.py` |
| v3.6 日常补丁 | 592 条日常对话 + 系统化小数字运算（×6），单条 SFT 后检索回补 | 38.0M tokens + 补丁轮 | `scripts/v3_6_build_daily_patch.py`、`scripts/v3_6_sft_patch.py` |
| v3.7 KV-QAT | 训练时注入 q2 block8 KV 噪声；补长文召回与股票拒答 | 约 10M tokens | `scripts/v3_7_build_qat_data.py`、`scripts/v3_7_kv_qat.py` |
| v3.8 上下文升级 | 检索过采样 + soup + **末层微调**修对话 | 约 15M | `scripts/soup_models.py`、`scripts/v3_8_build_fix_data.py` |
| v3.9 零代价修行为 | 末层微调（最后 2 层 + norm）用 40× 股票拒答数据修范围评测 | ≤1M tokens | `scripts/v3_6_sft_patch.py --train-last 2` |
| v3.10~v3.11 嵌入式 QAT | q2 KV-QAT → 算术边界补丁 → **Q4 权重/q2 KV 双 QAT** | 约 20M tokens | `scripts/v3_7_kv_qat.py --wqat`、`scripts/v3_11_build_arith_patch.py` |
| v3.12~v3.13 算术/记忆 | 算术走末层微调；合成 5,600 条记忆对话（PC 末层 / 板端双 QAT） | 约 24M tokens | `scripts/v3_11_build_repair.py --pc-fix`、`scripts/v3_13_build_memory.py` |
| v3.14 tool 版 | 算术/时间/随机数做成引擎 tool；训练数据过滤纯算式样本；GGUF 发行取消 | 约 14M tokens（无算术混训） | `scripts/v3_14_build_noarith_mix.py`、`core/feng_calc.c` |
| v3.15~v3.19 打磨 | 身份上下文锚点 → 板端串名修复实验（未采用路线已记录）→ 引擎记忆 tool → 440 条定向数据末 2 层 15ep | 约 12M tokens | `scripts/v3_15_build_identity_ctx.py`、`scripts/v3_19_build_mix.py` |
| v3.20 引擎 | 序列数数 tool + 统一采样器；全项目迁移 C++23（零堆/无异常/RTTI）并做体积/性能优化 | tool 为 C++ 零堆实现，不占权重 | `board/main/`、`docs/CHANGELOG.md` v3.20 节 |
| **v4.0 引擎** | 引擎核心**迁到严格 C23**（板端 + PC 共用，PC 驱动仍 C++23、CUDA 后端独立）；加手机端与编辑器代码提示 | 同上，数值路径逐字节未变 | `board/main/`、`phone/`、`docs/esp32-firmware.md` |

完整超参、每阶段 loss/耗时/显存见 [`docs/CHANGELOG.md`](docs/CHANGELOG.md) 与
[`docs/DELIVERY.md`](docs/DELIVERY.md)。

## 仓库结构

```
README.md              仓库首页（英文版）
README_CN.md           仓库首页（中文版，本文件）
AGENTS.md              给 AI 编码代理的工作指南：工作约定、硬约束、命令速查
docs/                  全部说明文档（索引见 docs/README.md）
v4/base/               当前权重，也是 4.0 的训练起点（HF 格式，自带 tokenizer）
scripts/               数据构建 / 训练 / 评测 / 导出脚本（路径解析见 scripts/paths.py）
board/      ESP32 固件 + 可移植 C23 推理引擎 + PC 端一致性检查与回归套件
phone/                 手机端（Termux CLI + Android APK），复用同一套 C23 引擎
data/                  提示词池 / 教师蒸馏数据 / 公开语料（不入库）
eval/                  评测结果 JSON（范围、针检索、记忆、留出、标准基准）
logs/                  构建 / 训练 / 烧录 / 板上测试日志（board_baseline_lut.txt 是板上精度基线）
tools/check_md.py      文档自检（代码围栏、路径、过时数字）
tools/check_docs.py    文档事实校验（模型规格 / 评测数字与实际产物对齐）
tools/check_all.py     一键验收：文档 → 构建 → 单测 → 32 题矩阵 → 位精确
```

仓库只保留 v4 一条线：v1 / v2 / v3 的历史版本目录已于 2026-10 移除，
需要旧版权重时从对应的 Release 取。

## 配置

脚本里**没有硬编码盘符**：项目根目录由脚本位置自动推导，外部依赖按
**环境变量 > `scripts/local_paths.json` > 默认值** 的顺序解析：

| 环境变量 | 用途 |
|---|---|
| `FENG_ROOT` | 项目根目录（默认：脚本上一级目录） |
| `FENG_PY` | Python 解释器（默认：当前解释器） |
| `FENG_DATA_DIR` | 原始语料目录（**只有重建 v1/v2 语料时才需要**） |
| `FENG_LLAMA_DIR` | llama.cpp 仓库目录（GGUF 转换 / benchmark，可选） |
| `FENG_GXX` | MinGW g++ 路径（仅 `-MinGW` 备用路径使用；默认构建走 MSVC，见 `build_pc_chat.ps1`） |

复制 `scripts/local_paths.example.json` 为 `scripts/local_paths.json` 填自己的路径（后者已 gitignore）。
自检命令：`python scripts/paths.py`，会逐条打印路径是否可用。

## 已知限制

- **30M 容量上限**：没覆盖到的自由问答会答偏或编造；复杂推理、专业领域、英文任务不可靠
  （标准基准大多贴近随机，见上）。
- **推荐类只保训练分布内的**：书/电影推荐是补丁里逐条写的，换书名或要最新榜单会露馅。
- **板端上下文 2048（q2 KV）**；32k 需要 HF/Python 路径，或自己用 `-DFENG_MAX_SCORES=32768` 重编 PC C 引擎（`pc_chat --ctx`）；板上 32k 受 PSRAM/KV 内存限制不可能。
- **PC 权重与板端权重不能互换**：PC 的 HF 权重没做量化感知训练，导进引擎会明显退化
  （32 题矩阵 22/27 vs 27/27）；手机/ESP32 用 4.0-embed（`feng-30m-v4.0-embed-release.zip` 的 `weights/esp32/`）。
- **留出 30 题是开发集**（strict-v7 下 v3.19=17/30、v4.0=18/30）；第二套在 v4 用于选型，不再是独立测试集（strict-v7：v3.19=15/30、v4.0=16/30）。
- **记忆只覆盖可枚举句式**：`\reset` / `\mem clear` / 上下文写满会清空；板端写在 NVS、手机端写在模型目录的 `mem.txt`，断电重启仍在（PC `pc_chat` 仍只在 RAM）。
- **板端时间需要宿主对时**：串口脚本连接时自动发 `\settime`（宿主 NTP），否则时间 tool 会提示未对时。
- **板端速度约 2.3 tok/s**（短回复最佳值；长回答/长上下文更慢，见 `logs/board_v3_19b6_stress32.txt`）；
  Q4 GEMV 内层已实测到这颗芯片的极限（周期级剖析与 5 个候选优化的排除记录见
  `docs/CHANGELOG.md` v4.0 板端性能节），再要提升只能换芯片或缩小模型。
- 量化抗性对权重回插极敏感：掺 20% 旧权重就掉点（CHANGELOG v3.10）；改板端行为要走
  "补数据 + 权重/KV 双 QAT"链路，不要手动 soup。
- ESP32 固件的模型分区偏移必须与 `board/partitions.csv` 一致。

## 致谢

**教师模型** —— **[UkisAI](https://github.com/CraneBW/ninfer-ternary-bonsai-ada) 团队的
Swift Bonsai 2（27B）**。v2 起的所有对话数据都由它蒸馏而来（v1 用的是 Qwen3.5-0.8B 微调版），
经同项目的 **NInfer** 推理引擎（`.ninfer` 权重格式）在本地承载。

**学生架构** —— **Qwen3**。11 层 / hidden 448 / 7 头 MHA、RoPE、RMSNorm、
tied embedding 这套结构参考了 Qwen3 系列的设计。

**公开语料** —— 4.0 的提示词池取自 **ShareGPT 中文**、**Evol-Instruct 中文**、
**Alpaca GPT-4 中文**、**Databricks Dolly-15k**、**UltraChat-200k** 五份公开数据集。
只取其中的**问题**，答案一律由教师重新生成，并与留出题逐字去污染。

**工具链**

| 项目 | 用在哪 |
|---|---|
| [ESP-IDF](https://github.com/espressif/esp-idf) | ESP32-S3 固件（v5.5.5，xtensa gcc 14.2） |
| [PyTorch](https://pytorch.org/) | 训练与量化感知训练 |
| [HuggingFace transformers](https://github.com/huggingface/transformers) | 权重格式与分词器 |
| [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 标准基准（SciQ / PIQA） |

感谢以上项目与数据集的作者把工作开源出来。

## 许可证

代码与权重均为 **Apache-2.0**（见 [`LICENSE`](LICENSE)）。本项目为个人项目，
与任何模型厂商无隶属或背书关系。

---

<sub>固件里藏了一句话。<br>
如果想关闭彩蛋请开启 `nobonus` 编译选项（`idf.py -DFENG_NO_BONUS=ON build`）。</sub>
