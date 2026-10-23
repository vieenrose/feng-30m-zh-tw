# AGENTS.md — feng-30m 项目工作指南

本文件供在此仓库工作的 AI 编码代理（Codex 等）阅读。**本项目的文档、注释、提交说明与回复统一用中文。**

## 1. 项目是什么

用大模型当教师，**从零训练一个 ~30M 参数的中文对话模型**，量化到 Q4 后在 **ESP32-S3（32MB flash / 16MB PSRAM）上离线跑**。

- 学生架构（v2/v3 现行）：Qwen3 结构，**11 层 / hidden 448 / 7 头 MHA（7 KV 头）/ head_dim 64 / FFN 896 / 16k 词表 / tied embedding**，29.43M 参数，`rope_theta=1e6`，v3 原生支持 32768 上下文。
- 身份自述（v3.2 起，必须保持）：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**；
  模型不得自称 Qwen/ChatGPT，也不得说自己是"微调"（训练数据来自教师蒸馏，模型本身从零训练）。
- **教师谱系**：**v1** 用 **Qwen 3.5 0.8B 微调版**（feng-0.8b，bf16）生成的数据训练；**v2 / v3** 用 **bonsai2-27b（27B）**训练。
- 版本演进：v1（8 层 / 32k 词表，只能对话，无法上板）→ v2（16k 词表 + 1.5B token 预训练 + 27B 教师 SFT，首次上板）
  → v3（渐进长上下文 + 合成检索 SFT）→ v3.5（修多轮复读）→ v3.6（日常对话大补丁）
  → v3.7（KV-QAT + q2 KV block8，板端 2048 上下文）→ v3.8（上下文专项升级）
  → v3.9 → v3.10 = v3.9 底座 + q2 KV-QAT → v3.11 = 算术边界修复 + Q4 权重/q2 KV 双 QAT
  → v3.12 = PC 算术修复 → v3.13 = 记忆版 → v3.14 = tool 版（算术/时间/随机数交给 C++ 引擎，
  模型不再学算术）→ v3.15-embed（身份漂移修复）→ v3.16-embed（身份串名修复；p4–p7 证明是容量边界）
  → v3.17 = 引擎/固件记忆 tool（可枚举问答确定性回答，权重未变）→ **v3.19 = 定向教师数据修"模板串台"
  （PC `v4/base`、板端导出 `model_export_v3_19b6`；当时发布，现为 PC 4.0 起点）**
  → v3.20 = 引擎序列数数 tool + 统一采样器 + 体积/性能优化（固件 304,192 B）
  → **v4.0 = 引擎全面升级（当前）：推理核心迁到严格 C23（板端与 PC 共用同一份源码，
  PC 驱动仍 C++23、CUDA 独立），新增手机端 `phone/` 与编辑器代码提示；
  固件 281,248 B（含记忆 NVS 持久化）、板端 2.33 tok/s**
  → **4.0-embed = 手机/ESP32 发布权重（测试版 `4.0-20261018H`）**。
  - **记忆持久化（2026-10-18）**：`feng_mem_export/import` 把记忆导出为文本；
    板端存 NVS（`nvs_flash`，key `mem`），手机端存模型目录的 `mem.txt`；
    `\reset` / `\mem clear` / 上下文写满会清空并擦除副本。板端实测跨重启
    「我叫小明，请记住。」后问「我叫什么名字？」仍答「你叫小明。」，`\reset` 后不再命中。
    PC `pc_chat` 仍只在 RAM。板端记忆 24/24、tool 13/13 复测通过。
  - **v4.0 训练链路（2026-10-17 完成；板端权重未替换）**：教师数据 **36,916 条 → 34,143 收 / 2,773 拒**，
    SFT 合并 **38,298 条 / 52,964 轮**；PC 全参 SFT 两次（第一次掉分，见下），
    第二次 `v4/sft_pc2`（3 epoch / lr 1e-5 / 身份×4 / 补丁×8）**两套留出首次双超 v3.19**
    （旧 strict-v6：22/30、19/30，基线 19/30、18/30；2026-10-18 审计后 strict-v7 为 18/30、16/30，基线 17/30、15/30），探针 40/42、范围身份 10/10。
  - 之后还要做**检索回放**（`v3_retrieval_sft.py`）：纯 SFT 会把 32k 长文打回 7/32。
    回放 2 epoch 后 `v4/retr_sft4`（旧字符口径）针检索 110/128（32k 28/32）、
    「文中没有」拒答 64/64、留出（strict-v6）22/30 + 21/30。
    **2026-10-18 修了评测脚本的字符/token 混淆**（旧输入实际短 20–35%，且 32k 没给生成留位）：
    按 token 精确重测，最终 v4.0 = **103/128（32k 20/32）**、v3.19 = **106/128（32k 24/32）**，
    拒答 v4 64/64 vs v3.19 62/64。**检索命中没有超过 v3.19，只有拒答更好**；
    证据 `eval/longctx32_v4final.json`、`eval/longctx32_v319.json`。
  - 最后一步：`我叫X，请记住` 这个句式在 4.8 万轮数据里**一条都没有**，原始 4.0 会答
    "听起来不错，希望你喜欢"。加 **160 条定向补丁（末 2 层 / 5 epoch / lr 5e-6）**修掉，
    得到最终 PC 权重 **`v4/patch_name_e5_lr5e-6`：strict-v7 留出 18/30 + 16/30**、探针 40/42、
    针检索 103/128（32k 20/32，token 精确口径）、拒答 64/64 —— 修名字**没有伤到别处**。
    教训：补丁的强度和样本量要一起降（第一次 10 epoch / lr 1e-5 把留出第一套打掉 2 题）。
  - **检索数据的负样本必须带诱饵、与评估口径同分布**，否则拒答学不会（实测 4/64 vs 64/64）。
  - **32k KV-QAT 试训（2026-10-18）是负结果，不采用**：从 `v4/patch_name_e5_lr5e-6`
    做 KV-only QAT（6.55M token / 1 epoch / retr-lr 3e-6），4k/8k/16k 28/31/29（略升），
    **32k 14/32（基线 20/32）**，总 102/128；短行为不变。32k 只有 80 个唯一窗口，
    是过拟合。证据 `eval/longctx32_qat32k.json`。**不要再直接加 epoch**；先扩 32k
    唯一窗口到 256–512，或先解决手机 prefill（4k 107s、32k 约 2h）。
  - **4.0-20261018A（后训练实验，不发布）**：`board/model_export_v3_19b6` 反量化成 HF
    （`tools/dequant_model_bin.py`）后，从 34,143 条教师数据抽 1 万条 + 身份/日常补丁
    （15,164 条）做全参 SFT 1 epoch / lr 1e-5。留出 **19/30 + 16/30**、探针 41/42、
    范围 10/10；但长文 4k/8k/16k/32k = 30/30/25/**1**、拒答 34/64，Q4 导出 27/27 + **3/4**。
    **短对话变好、长文塌方，不替换发布权重**；产物 `v4/4.0-20261018A/`。
  - **4.0-20261018B（A + 长文检索回放，不发布）**：在 A 上跑 `v3_retrieval_sft.py`
    （`data/retr_v4`，2 epoch，lr×0.3）→ 长文 **108/128（32k 25/32）**、拒答 **64/64**；
    留出 19/30 + 15/30、探针 41/42、范围 10/10。B 是当前 PC 长文最好的版本
    （v4.0 final 103/128、32k 20/32），但 Q4 导出只有 **26/27 + 3/4**，不过板端关卡 A；
    产物 `v4/4.0-20261018B/ctx32768/final`。
  - **4.0-20261018C/D（定向 QAT 刷分实验，不发布）**：C = B + QAT 2ep，
    Q4 **26/27+4/4**、32k 0/16；D = C + 8ep 日常 QAT，Q4 **27/27+4/4**（关卡 A 全过），
    但 h1 **16/30**、32k **2/32**、拒答 16/16。**证明 4.0 能刷到 27/27，
    代价是把长文/泛化容量挤掉**；不要再把“27/27”当通用能力。产物 `v4/4.0-20261018C/D/`。
  - **板端双 QAT 前三版里两版没过关卡 A，这一版板端当时沿用 v3.19-embed（后续由 4.0-embed 取代）。**
    从最终 PC 权重复现：3 epoch / lr 6e-6（5,034 步）→ **24/27**，1 epoch / lr 4e-6
    （1,678 步）→ **23/27**；都低于 v3.19-embed 的 27/27，所以不替换。
    1 epoch 反而更差——QAT 步数太少，权重**没适应** Q4/q2 噪声。
    3 epoch 则相当于又做一遍全参 SFT，把稀少行为（翻译、推荐）磨掉。
    第三版 `v4/qat_20261017D` 在 PC 关卡 A 达到 27/27 + 4/4、两套留出 19/30 + 19/30、
    42 题探针 42/42，但**尚未做板端 tool/记忆/实机验收**，暂不替换；证据在
    `logs/gate_a_20261017D.txt`、`eval/*_20261017D.json`（随本次提交入库）。
    失败样本导出留在 `board/model_export_v4_qat_rejected/`（已 gitignore），别误发。
    下次方向：去掉 `--retr`（板端 MAX_CTX=2048，4096/8192 的长文 QAT 用不上，
    而那段 lr 1e-5 / gnorm 40 的更新对短任务伤害最大）、或只训末 N 层。
    详见 `docs/CHANGELOG.md` §4f。
  - **发布口径**：PC 权重 = **4.0**（`v4/patch_name_e5_lr5e-6`），
    手机 / ESP32 权重 = **4.0-embed**（导出 `board/model_export_4.0-embed`），
    引擎两边都是 v4.0。**这几个版本号不要混着写。**
  - **4.0-embed（2026-10-18 发布，手机/ESP32 权重）**：测试版 `4.0-20261018H`
    = 4.0B 的全部 attention/FFN + v3.19-embed 的 tied embedding/lm_head。
    Q4 32 题 **27/27 + 4/4**、PPL **102.99**（v3.19-embed 107.49）、留出 200 **85**
    （v3.19 83）、长文 **108/128**、板端 tool 13/13、记忆 24/24、多轮 10/10。
    这是“同硬件、同尺寸”的发布权重；PC 权重仍是 v4.0，两者不要混写。
  - **模型命名规则（2026-10-18 起）**：
    - **测试版本**：`x.y-yyyymmddC`，`C` 表示当天第几个小版本，按 **A–Z** 顺序
      （如 `4.0-20261018A`、`4.0-20261018H`）；
    - **PC 发布版本**：`x.y`（当前 `4.0`）；
    - **手机 / ESP32 发布版本**：`x.y-embed`（当前 `4.0-embed`）。
    发布目录：PC 权重在 `v4/<发布名>`，板端导出在 `board/model_export_<发布名>`。
  （标准基准在 v4.0 收窄到 SciQ / PIQA 两项；其余七项大多贴随机线，其中 BoolQ/OpenBookQA/ARC-C 显著低于随机线，是偏置信号而非“无信号”。SciQ prompt 含 support 证据段，是开卷口径。）
- v3.19 现状（**历史；PC 4.0 的训练起点，板端上一版**）：
  - PC 权重（已收进 `v4/base/`，也是 4.0 的 SFT 起点）：从 `archive/v3_14/pc2` 出发，用 440 条定向教师数据（常识/列举/情绪/寒暄/推理/身份六类，
    `scripts/v3_19_build_target_prompts.py`）+ 多轮/身份回放做**末 2 层 15 epoch** 微调；
    留出 30 题 **17/30**（strict-v7；旧 strict-v4 19/30；v3.14 = 16/30）、
    单类别检索 **106/128**（token 精确口径；旧字符口径 110，v3.14 旧口径 108）、
    多类别 105 是旧口径、未重测、记忆 23/24（-1）、范围/身份/探针保持（`eval/v3_19pc4_*.json`）。
  - 板端 v3.19-embed 权重（导出在 `board/model_export_v3_19b6/`）：在 `archive/v3_16/board_p3` 上做 **Q4+q2 双 QAT（4 epoch / lr 6e-6）**；
    C++ 引擎矩阵 **27/27+4/4、召回 4/4、算术 21/21、pc_check MATCH**，HF 留出 **17/30**（strict-v7；旧 19/30；v3.16 = 17）、
    记忆 24/24；实机 tool 13/13、记忆 12/12、8 轮身份序列 8/8（`logs/board_v3_16p3_identity_ctx_engmem.txt`、
    `logs/board_v3_19b6_memory12.txt`、`logs/esp32_multi.txt`）。
  - **评测口径**：留出题判定用 strict-v7（子串假阳性、复读误判、"推荐运动"误命中、
    问题关键词复读四类都拦，关键词大小写不敏感；`chat_probe_heldout.py --rescore`，
    重打分必须传对 case 表，否则未知题按原判保留）；
  - **两套留出题**：第一套 30 题已当开发集（strict-v7 v3.19=17/30、v4=18/30）；第二套
    `chat_probe_heldout2.py` **在 v4 被用于选型**（e3/e5 补丁与 soup 候选都看过），
    不再是未触碰的独立测试集（strict-v7 v3.19=15/30、v4=16/30）；发版文案必须注明这一点；
    **关键词判定会漏假阳性**（答案含"南极/米"等词即判 OK，即使方向答反或退化成词）——
    头条数字必须人工过一遍 `rows[].a`，推荐解码对照存 `eval/heldout2_*_rec.json`；
    留出 30 题已被多轮迭代用作开发集，只作版本对比，不当无偏泛化分数。
  - **冻结留出 200 题（2026-10-18）**：`scripts/chat_probe_heldout200.py`，
    与 `data/v4_sft.jsonl` 精确 prompt 去重 0 重合，只用于发版对比，
    **禁止拿它训练/调参**。基线 v3.19 **83/200**、v4 final **92/200**
    （McNemar p=0.16，配对差 95% CI −1~+10pp，方向有利但不显著）。
  - **PPL 稳定指标**：`scripts/eval_ppl.py` 在 `data/v4_val.jsonl` 上算
    assistant-token 困惑度；v3.19 **107.49**、v4 final **51.20**、B 97.45、D 161.29。
    关键词分数接近时先看 PPL，避免"分没掉但语言能力退化"。
  - 别再用 v3.16 时代的"往身份数据加精确链"套路（p4–p7 已证明只是重排失败点）。
- v3.16-embed（上一版板端权重，`archive/v3_16/board_p3/`）：在 archive/v3_15/board_ctxid4 上做
  **上下文双向问名**补丁（报名字后问身份/问名字、同类事实取新、记忆保护），lr 1.5e-6 × 1 epoch；
  板端 12 题记忆 10/12（**v3.17 引擎侧记忆 tool 上线后 12/12**）、6 轮报名字→问身份序列从 3/6 修到 5/6、tool 13/13、
  PC 32 题矩阵 27/27+4/4；新增多轮回归套件 `pc/pc_mt_suite.cpp`（残余见 CHANGELOG v3.16-embed 节）。
  **注意**：再往上加"精确链"数据（p4–p6）能把 8 轮序列修到 8/8，但会丢长文召回
  （90% 深度）或在板端出现 "我user" 伪影——试过且未采用，别再重复这条路线
  （`docs/CHANGELOG.md` v3.16-embed 附录）。
  （**v3.17 起改用引擎侧记忆 tool 解决**：身份/常见事实追问由 C++ 引擎确定性回答，
  板端 12 题记忆 12/12、8 轮身份序列 8/8——不要再拿训练数据去磨这类可枚举问答。）
  **p7（lr 5e-7 最小干预）已确认这是容量硬边界**：它保住 27/27+4/4 并把 8 轮序列修到 10/10，
  却把失败换到别的次序（「你好呀→我叫丽丽→你叫什么名字」答"我叫丽丽"）、mem12 掉 1 分。
  p3–p7 五个版本都只是重排失败点——**多轮"你我/事实"分辨不要再靠加数据解决**，
  要提升就得换更大的模型或改架构（见 `docs/CHANGELOG.md` v3.16-embed 附录）。
- v3.15-embed（历史，`archive/v3_15/board_ctxid4/`）：在 archive/v3_14/board6 上做
  「闲聊前缀 + 身份问答」锚点（`scripts/v3_15_build_identity_ctx.py`）+ 召回 ×8、lr 3e-6 × 1 epoch、
  继续双 QAT；修掉多轮里「你叫什么名字 → 你叫小模型/小王子」的漂移
  （7 组前缀 6 组完全正确），矩阵 **27/27+4/4**、算术 21/21、记忆 10/12、工具 8/8 全部保持。
- v3.14（历史 PC 发布）：
  - **tool**（`core/feng_calc.c`、`core/feng_tools.c`、`core/feng_memory.c`）：算式（多位数/小数/括号）、
    **序列数数**（"把 1 到 5 倒着数一遍"→`5、4、3、2、1。`、"从 3 数到 8"；只在祈使句触发，
    "我从1数到100也数不完"这类陈述仍交给模型）、
    UTC+8 时间（宿主 `\settime` 对时 + esp_timer 走时）、随机数（运行时间×1.54×1000，丢第一个取第二个）；
    时间 tool 还含**时钟推算**（`现在7点，再过3小时是几点？`→10 点，跨天说"明天/昨天"）；
    **记忆 tool（v3.17）**：从 `我叫X / 最喜欢Y / 住在Z / 养了W` 里抽事实，追问确定性作答
    （12 题记忆 12/12、身份永不串名），另有通用键值槽 `我最喜欢的<键>是<值>` /
    `我的<键>是<值>`（书/生日/家乡…）；`\mem` 查看、`\reset` 清空；
    列出/遗忘（`你还记得什么？`、`忘掉我的颜色`、`把记住的都忘掉`，遗忘留"墓碑"、
    重学解除）；只覆盖可枚举句式（C 单测 69 项）。**工具顺序：记忆要在时间之前**（"我的生日是几号？"）；
    板端 0.5s 秒回；PC C++ 引擎 `pc/pc_chat.cpp` 与 Python `scripts/runtime_tools.py` 同口径；
  - **训练数据不再含纯算式**（`scripts/v3_14_build_noarith_mix.py` 用 tool 识别器过滤）；
  - PC `archive/v3_14/pc2/`：记忆 24/24、范围 10/10、探针 42/42、身份 12/12、单类别 108、多类别 107；
  - 板端 `archive/v3_14/board6/`：q2 矩阵 27/27+4/4、工具 8/8、默认/情绪 10/10、记忆 12 题 10/12、1.80 tok/s；
  - **GGUF 发行取消**：llama.cpp 路径没有 tool，Release 只发 hf + esp32。
- v3.13（历史）：
  - PC `archive/v3_13/mem_pc3/`：单类别 28/29/29/27 = **113/128（并列历史最高）**、多类别 108，
    算术 274/281、**记忆 21/24（v3.12 只有 5/24）**、范围 10/10、探针 42/42（0 未命中）、
    身份 12/12、多轮 1.00；
    - 板端 `archive/v3_15/board_ctxid4/`：q2/2048 矩阵 **27/27 + 4/4**、算术子集 **21/21**、
    板端 默认/情绪/工具 **10/10 ｜ 10/10 ｜ 8/8**、跨轮记忆 10/12、1.80 tok/s；
  - **固件（`board/main/main.c`）默认多轮上下文**：KV 跨轮累积、`\reset` 清空、写满自动开新对话；
    `scripts/esp32_multi.py` 默认每题前 `\reset`（独立探针口径），`--no-reset` 测连续对话。
- v3.12（PC 上一版，`archive/v3_12/arith2l3/`）：算术 275/281、单类别 110、探针 42/42、记忆 5/24。
- v3.11（板端上一版，`archive/v3_11/pol8/`）：q2 矩阵 27/27+4/4、算术子集 21/21、板端 30/30、1.81 tok/s；
  关键技巧是 **Q4 权重/q2 KV 双 QAT**（`v3_7_kv_qat.py --wqat`），量化抗性对权重回插极敏感
  （掺 20% v3.9 权重就掉到 25/27），PC 32k 弱。
- **PC 用 v4.0 权重（`v4/patch_name_e5_lr5e-6`）；手机 / ESP32 用 4.0-embed
  （导出在 `board/model_export_4.0-embed/`）；两边同一套 v4.0 / 严格 C23 引擎与固件**
  （q2/2048 固件 **281,248 B**、板端 2.33 tok/s；见 `docs/CHANGELOG.md` 4.0-embed 节与
  `docs/esp32-firmware.md`）。
- **标准基准（英文，0-shot）**：**只有 SciQ 71.00/73.80 与 PIQA 53.37**。
  ARC/HellaSwag/OpenBookQA/Winogrande/BoolQ/SciCloze 六项曾跑过，但在这个 30M 中文模型上
  全部贴随机线（BoolQ 37.8% 甚至低于二选一随机，只有 yes/no 偏置），**测不出能力差异**，
  已于 2026-10-15 从套件删除（`scripts/bench_standard.py` 的 `TASKS` 只剩 `sciq,piqa`）。
  **不要再把那些数字加回来，也不要拿它们当卖点**——一堆没信号的分数只会盖住真实水平。
  原始结果留档在 `eval/lm_eval_feng_v3_19_pc4.json`（复现 `scripts/bench_standard.py`）。
- 硬件：RTX 5060 Ti 16GB（训练）+ i7-12700KF；ESP32-S3-**WROOM-2-N32R16V** 开发板（32MB Octal flash + 16MB Octal PSRAM，1.8V）。

## 2. 运行环境与路径解析（代码里已无硬编码盘符）

所有 Python 脚本统一走 `scripts/paths.py` 解析路径，优先级
**环境变量 > `scripts/local_paths.json`（不入库）> 从脚本位置推导**；自检命令 `python scripts/paths.py`。

| 用途 | 解析方式（本机实际值见 `scripts/local_paths.json`） |
|---|---|
| 仓库根目录 ROOT | 自动按脚本位置推导，可用 `FENG_ROOT` 覆盖 |
| 原始语料（重建 v2/v3 语料时用；本机已随 v1 归档清理） | `FENG_DATA_DIR`（`FENG_TEACHER_GGUF` 键已废弃） |
| Python 解释器（torch 2.13.0+cu132，CUDA 可用） | 默认当前解释器 `sys.executable`，可用 `FENG_PY` 覆盖 |
| llama.cpp（GGUF 转换 / 量化 / benchmark） | `FENG_LLAMA_DIR` |
| ESP-IDF / esptool / g++ | `flash.ps1 -EspIdfPath -EspToolPy` 或环境变量 `IDF_PATH`/`ESPTOOL_PY`/`FENG_GXX` |
| 串口 | **COM20 = CH343，COM19 = 芯片原生 USB-JTAG**，115200 对话 / 921600 烧录 |

机器相关的实际路径一律写在 `scripts/local_paths.json`（已 gitignore，**不要提交**；新机器复制
`scripts/local_paths.example.json` 填写）。文档和代码里都不出现盘符。

本目录是 git 仓库，远端 `origin = https://github.com/jiaheng0815/feng-30m`（公开仓库）。发布约定：
**主仓库只放代码与文档**——数据集（`data/`、`v2/data/`）与权重/二进制（`*.safetensors`、`*.gguf`、`*.npy`、`*.bin` 等）
 都由 `.gitignore` 排除，随 Release 发布（PC `feng-30m-v4.0-release.zip`、板端 `feng-30m-v4.0-embed-release.zip`）；
**v3.14 起不再发行 GGUF**（llama.cpp 没有 tool，见 CHANGELOG v3.14）。
代码与权重均为 **Apache-2.0**（`LICENSE`）。
开源数据集只含**教师蒸馏数据**（提示词与教师输出）；本地脚本生成的多轮/补丁/运算数据不入 Release 包。
模型权重、训练产物一旦覆盖无法回滚，删除或覆盖已有模型目录前必须先向用户确认。

改动发布物时记得同步：`docs/USAGE.md`（下载/推理/烧录说明）、Release 包内 `weights/`、`datasets/` 的清单，
以及 GGUF 的 chat template（用 `$FENG_LLAMA_DIR\gguf-py\gguf\scripts\gguf_new_metadata.py --chat-template-file` 写入）。

## 3. 目录地图

> **v4.0 起根目录只留 `README.md`（仓库首页，英文）、`README_CN.md`（中文版）与 `AGENTS.md`（本文件）**，
> 其余说明文档全部归到 `docs/`。历史版本目录一律进 `archive/`。
> 两个 README **只有语言不同**，改一个必须同步改另一个（结构和数字必须一致）。

| 路径 | 内容 |
|---|---|
| `README.md` | 仓库首页（英文）：是什么、跑分、快速上手 |
| `README_CN.md` | 仓库首页中文版，与 `README.md` 内容一一对应 |
| `AGENTS.md` | 本文件：工作约定、硬性约束、命令速查 |
| `docs/` | **全部说明文档**（索引见 `docs/README.md`）：`USAGE.md`（跑起来）、`CHANGELOG.md`（版本演进，失败尝试也保留）、`COMPARISON.md`（四代横向对比）、`DELIVERY.md`（交付与验收口径）、`esp32-firmware.md`（固件工程） |
| `v4/` | **唯一的模型线**。`v4/base/` = 4.0 SFT 起点（HF 格式：`model.safetensors` + `config.json` + 自带 `tokenizer.json`/`tokenizer_config.json`/`chat_template.jinja`，来自 v3.19 pc4 的 Release）。训练输出按阶段建子目录（`v4/sft/`、`v4/qat/`…），产物结构与 v2/v3 时代一致：`<阶段>/final/` + `summary.json` + `train_log.jsonl` |
| `data/` | 提示词池 / 教师蒸馏数据 / 公开语料（**不入库**，`.gitignore` 排除）。v4.0 重建：5 个公开语料从 HF 拉取，提示词池由 `scripts/v4_build_prompts.py` 生成，教师输出由 `scripts/v4_teacher_gen.py` 写入 |
| `scripts/` | 全部数据构建 / 训练 / 评测 / 导出脚本（含 `scripts/paths.py` 路径解析、`scripts/bench_standard.py` 基准入口） |
| `eval/` | 评测结果 JSON（范围 / 针检索 / 记忆 / 留出 / `lm_eval_*.json` 标准基准）；`eval/README.md` 记录命名约定 |
| `logs/` | 所有构建 / 训练 / 烧录 / 板上测试日志；`board_baseline_lut.txt` 是板上精度基线 |
| `core/` | **共享 C23 推理核心**（严格 C23、零堆、零全局构造）：`feng_llm` / `feng_quant` / `feng_model` / `feng_smp` / `feng_sample` / `feng_tokenizer` + 四个 tool（`feng_calc` / `feng_tools` / `feng_memory`）。**board / pc / phone 编的是同一份源码**，改这里三家全变 |
| `board/` | ESP32-S3 固件工程：`main/`（IDF 组件，放 `main.c`、彩蛋 `feng_bonus`、GBK 表、PIE/bench）、`partitions.csv`、`sdkconfig*`、`flash.ps1`、`tools/export_model.py`、`model_export_4.0-embed/`（当前板端权重）。工程说明见 `docs/esp32-firmware.md` |
| `pc/` | PC 驱动与验收套件（**C++23**）：`pc_chat` / `pc_kv_suite` / `pc_mt_suite` / `pc_check` / 各 tool 单测，GPU 后端 `feng_cuda.cu`，构建脚本 `build_pc_chat.ps1` / `build_pc_cuda.ps1` |
| `phone/` | 手机端（Termux CLI + Android APK）。**引擎编 `core/` 同一份源码**（`-DFENG_PHONE=1`），`phone/engine/` 只放手机独有的 `feng_phone.c/h`、`feng_neon.c/h`；详见 `phone/README.md` |
| `tools/check_md.py`、`tools/check_docs.py` | 文档自检：前者查围栏/路径/过时表述，后者把**全部 md 的关键数字与实际产物对齐**（参数量、固件体积、各版评测分数、检索 loss） |
| `tools/gen_compile_commands.py` | 生成仓库根的 `compile_commands.json`（编辑器代码提示数据），见「编辑器配置与代码提示」一节 |
| `tools/bench_teacher_real.py` | 用**真实提示词**测教师服务吞吐（4 线程 / max_tokens 200，与生产同口径）。`scripts/bench_teacher_conc.py` 用的是 8 条五个字的短问题，测出来是纯 decode 上限（59.3 tok/s），**不能拿来跟生产日志里的 37.8 比** |
| `tools/publish_all.py` | 一键发布：代码推 GitHub+Gitee、模型传 Hugging Face+ModelScope。**已 gitignore**（内嵌魔搭令牌），换机器要自己重建 |
| `tools/dequant_model_bin.py` | 把板端 `model.bin`（Q4/fp16）反量化回 HF 目录，供继续训练/评测；Q4 舍入误差保留，不改导出本身 |

> **历史目录已删除（2026-10-15）**：v1（原 `student/`）、v2、v3.19 与全部 v3 历史版本
> 目录（原 `archive/`，52 个）从工作区移除，仓库只保留 v4 一条线；训练起点收进 `v4/base/`。
> **`docs/CHANGELOG.md` 等文档里对 `archive/…`、`v2/…`、`student/…` 的引用是历史原文，
> 不再指向本机存在的路径**——按约定不改写（改写历史记录比留一个失效路径更糟）。
> `tools/check_md.py` 的 `LOCAL_PREFIXES` 已把这三个前缀列为"允许缺失"。
> 需要旧版权重时从 GitHub Release 取（走 `ghfast.top` 镜像，实测 11 MB/s）。

## 4. 工作约定

### 引擎语言约定（硬性，v4.0 起）

引擎分两层，**语言不同、必须分两趟编译**：

| 层 | 文件 | 语言 | 说明 |
|---|---|---|---|
| 推理核心 | `board/main/*.c` `*.h` | **严格 C23** | 板端与 PC 共用同一份源码 |
| PC 驱动 | `pc/*.cpp` | **C++23** | 套件/单测/基准，可用 `std::chrono`/`std::span` |
| GPU 后端 | `pc/feng_cuda.cu` | **C++**（nvcc 无 C 模式） | 与 C 核心用 `extern "C"` 对接 |

```powershell
gcc -std=c23 -O2 -c core/feng_llm.c -o feng_llm.o -Icore          # 核心
g++ -std=c++23 -O2 pc/pc_kv_suite.cpp feng_llm.o -Icore -lm       # 驱动 + 链接
```

**为什么必须分开**（实测，别再踩）：MSVC 一条命令里同给 `/std:clatest` 与
`/std:c++latest` 会报 `error D8016: 选项不兼容`；只给 `/std:clatest` 则 C++ 驱动里
`<span>` 用不了。所以 MSVC 侧核心走 `/std:clatest /TC`、驱动走 `/std:c++latest /TP`，
`build_pc_chat.ps1` / `build_pc_cuda.ps1` 已按此实现。`g++` 会把 `.c` 当 C++ 编，同理。

写核心代码（C23）必须遵守：

1. **零堆分配**：核心源码禁止 `malloc`/`calloc`/`free`。唯一例外是 tokenizer 的
   **启动期**一次性 PSRAM 加载（`feng_tokenizer.c` 的 `tok_alloc`，固定大小、之后不释放）。
2. **零全局构造**：静态数据只允许 POD 或编译期常量（全组件 `.init_array` 必须为空）。
   C23 没有 `constexpr` 函数，需要的小表（如 GPT-2 的非直接字节表）在加载期算一次。
3. **标准库白名单**：只用 `<stddef.h>` `<stdint.h>` `<stdbool.h>` `<stdarg.h>` `<string.h>`
   `<math.h>` `<stdio.h>` `<stdlib.h>`；禁止任何会分配的类型。
4. **错误处理**：返回错误码（现有 API 用 `int`）；C 本来无异常，不需要 `noexcept`。
   `[[nodiscard]]` 在 C23 是标准属性，继续用。
5. **可读性工具**：C 核心用 `stdbool.h`/定长数组/`static inline`/`enum` 表达意图；
   PC 驱动可以用 `std::span`/`std::string_view`/`constexpr`/`if constexpr`。
   热点内层循环保持手写，用 `-DFENG_*` 开关做 A/B。
6. **工具链**：PC 默认 **MSVC 两趟**（`build_pc_chat.ps1` 自动加载 vcvars64）：
   核心 `/std:clatest /TC`（MSVC 的 C23 模式）、驱动 `/std:c++latest /TP`（MSVC 没有
   单独的 `/std:c++23`，c++latest 就是）；两个 `/std:` **不能同时给**（D8016）。
   MinGW 为备用路径（`-MinGW`，`FENG_GXX`/`FENG_GCC` 覆盖位置）。
   CUDA 版 `build_pc_cuda.ps1`（MSVC + nvcc，nvcc 侧仍 `-std=c++20`）。
   固件由 ESP-IDF v5.5.5（xtensa g++ 14.2）编译，`main/CMakeLists.txt` 已设
   `C_STANDARD 23` + `C_EXTENSIONS OFF`，并且**不能**再带
   `-fno-exceptions -fno-rtti -fno-threadsafe-statics`（gcc 对 C 文件报错，且会误命中 `.S`）。

6.5 **fp16 权重 / fp16 KV（手机端与 PC 端专用）**：`FENG_FP16_WEIGHTS` 默认 **0**，
   板端永远保持 0——关掉时 `feng_layer_t` 的 `mat_dtype` 字段与加载期校验全部不参与编译，
   预处理输出与改造前**逐字节相同**（改这里之后必须重新验证这一点，见下）。
   - 权重格式来自模型文件的目录项，前向用 `FENG_LAYER_DTYPE(L)`；**不要再硬编码 `FENG_DT_Q4`**。
   - `FENG_KV_FP16=1` 是第四种 KV 模式，与 INT8/Q2 用 `#error` 互斥；布局同 fp32 的
     `[layer][ctx][hidden]`。
   - 验证板端未受影响：`gcc -E -P -std=c23 -Icore -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 core/feng_llm.c`
     与 `git show HEAD:core/feng_llm.c` 的预处理结果 `diff` 必须为空。
   - 手机端构建脚本显式传 `-DFENG_FP16_WEIGHTS=1`（单一源码后 `core/feng.h` 默认是 0）；PC 在
     `build_pc_chat.ps1` 的 `$cFlags`/`$cxxFlags` 里加 `-DFENG_FP16_WEIGHTS=1`
     （核心与驱动必须一致，否则 `feng_layer_t` 大小不同 → ABI 错位）。

7. **体积纪律（板端实测）**：禁止 `std::sort`（每套比较器实例化一份 introsort，实测 ~2.6KB）、
   `string_view::substr`（引用 `std::__throw_out_of_range_fmt`，拖进 libstdc++ 异常/字符串/
   pthread 运行时 ~5KB）；非热点模块（tokenizer/tool/model/gbk/main）用 `-Os`，热点
   （llm/quant/smp/sample）保持 `-O2`。改完对比固件体积：**不得超 C 版基线 307,392B
   （q2/2048 配置）**，当前为 **281,248 B**（关掉彩蛋是 281,056 B）。
8. **CUDA 后端（PC 可选，板端不含）**：`pc/feng_cuda.cu` + `build_pc_cuda.ps1`
   （MSVC + nvcc，`-fmad=false` 保证与 CPU 同运算顺序）；GEMV 权重常驻显存，其余算子在 CPU。
   **验收口径**：32 题矩阵须 27/27+4/4 且与 CPU 版**输出逐字一致**（`pc_kv_suite_cuda.exe`），
   `pc_check_cuda` argmax MATCH；实测 decode 157 → 322 tok/s（2.0×），prefill 变慢
   （PCIe 往返固定开销），如实记录，不要宣称全面加速。`FENG_CUDA=0` 强制 CPU 路径做 A/B；
   `feng_gemv3`（q/k/v、gate/up 合并调用）在板端等价于逐次 `feng_gemv_par`，双核不受影响。
9. **PC 多线程（OpenMP）**：`FENG_USE_OMP=1` 时 GEMV 按输出行并行（逐位不变，
   验收用 32 题矩阵逐字对比 + `pc_check` MATCH）；默认线程数 = 逻辑核 3/4
   （全开会因 SMT/E 核掉速，可用 `OMP_NUM_THREADS` 覆盖）。PC CPU 性能基线：
   **654 tok/s**（q2 KV / 32 token / i7-12700KF），低于 500 视为回归。

- **训练产物目录结构**：`<版本>/<阶段>/final/`（HF 权重 + tokenizer + `config.json`），阶段汇总写 `summary.json`（steps / tokens / loss / 峰值显存 / 耗时），逐步日志写 `train_log.jsonl`；中途 checkpoint 放 `<阶段>/rolling/stepN/`。
- **日志统一写 `logs/`**，评测结果统一写 `eval/`，不要散落在根目录。
- 训练脚本对已完成的阶段是**跳过式**的（检测到 `<阶段>/final/` 已存在就不重跑）；要重跑先用 `--out` 指向新目录，或显式删除旧目录（需用户确认）。
- 模型卡走 `<模型目录>/MODEL_CARD.md`；`tools/check_md.py` 会校验 README/DELIVERY/COMPARISON/CHANGELOG/esp32 README/MODEL_CARD 里的反引号路径与过时数字。
- Python 脚本开头都 `sys.stdout.reconfigure(encoding="utf-8")`，新增脚本请保持一致；命令行输出用中文没问题（torch 的日志仍是英文）。

### `board/main/feng_bonus.c` 是**故意藏的**，不要当死代码删掉

固件里有一个作者放进去的小彩蛋（`feng_bonus.c` + `main.c` 命令链末尾那个
`feng_bonus_decode` 分支）：

- 明文用 XOR 混淆存着，**源码和二进制里都搜不到**（这样 `strings`/`grep` 也挖不出来）；
- **不要列进 `\help`**，读 README 也无法直接知道触发方式（README 的小字只提到
  `nobonus` 编译选项，靠它当唯一线索）；
- 关掉它的开关是 `-DFENG_NO_BONUS=ON`（那时连混淆数据都不进二进制，实测省 208 B）；
- 它占 **192 B** 固件空间（281,248 vs 281,056），在体积纪律的预算内。

**维护要求**：重写命令链、清理"未引用代码"、或做体积优化时，别把它当成冗余删掉；
它跟 `\help` 里那些命令一样是有意的。改 `main.c` 的命令分支时留意最后那个 `else`。

### 手机端（`phone/`）与板端编**同一份**源码 —— 不再有派生副本

2026-10-23 之前 `phone/engine/` 是 `core/` 的手工拷贝（18 个文件、4909 行，
`feng_llm.c` 分叉 458 行、`feng_smp.c` 246 行），同步靠人 `cp`，`check_drift.py`
只能在漂移**之后**报警。这轮把它彻底删了：三端（板端 / PC / 手机）编同一份
`core/*.c`，手机差异全部走 `-DFENG_PHONE=1` 的编译期分支：

| 位置 | 手机端差异 |
|---|---|
| `core/feng_smp.c` | `#elif defined(FENG_PHONE)`：FreeRTOS 双核信号量 → pthread 线程池（4~8 核，按行切分，逐位一致） |
| `core/feng_quant.c` | `#if FENG_PHONE_NEON`：Q4 sdot / fp16 NEON GEMV 的分发 |
| `core/feng_llm.c` | `#if FENG_PHONE_NEON`：q2 注意力 K/V 的 NEON 内核；`FENG_ATTN_PARTS` 手机默认 4 |
| `core/feng.h` | `#ifdef FENG_PHONE` 包含 `feng_phone.h`，`FENG_MAX_SCORES` 跟随 `FENG_PHONE_MAX_CTX` |
| `phone/engine/` | **只剩 4 个手机独有文件**：`feng_phone.c/h`（mmap 加载、时间源、线程池、NEON 开关）、`feng_neon.c/h`（aarch64 内核） |

构建脚本已经改成从 `$CORE` 取源文件（`-Icore -Iphone/engine -DFENG_PHONE=1
-DFENG_FP16_WEIGHTS=1`）。守卫脚本 `phone/tools/check_single_source.py` 会在
`phone/engine/` 里又出现引擎副本时直接报错，CI 与 `check_all` 都跑它。

**NEON int8 路径是近似的**：裸模型 `pc_mt_suite` 在 P30 Pro 上是 mem12 9/12、seq 9/10；
`FENG_PHONE_NEON=0` 重编后回到 12/12、10/10。产品路径先过记忆 tool，不受影响；
需要与板端逐字对拍时用 NEON 关。

**手机 32k 是容量，不是可交互长文**：`FENG_PHONE_MAX_CTX` 默认已放到 32768
（q2 KV 154MB；APK 默认 32k、失败回退 2048）。prefill 是 O(n²)：P30 Pro 上
4064 token 预填从 **107 秒降到 58 秒**（q2 注意力按 head 切给 4 核；NEON 内核再摊一层）。
8k 约 210 秒，外推 32k 仍约 **1 小时**。批量 GEMM 已实测证伪（见 CHANGELOG），
再往上要动的是批量**注意力**。
长文验收用 `phone_chat --ids-file`（预 tokenized，绕过 4KB prompt 缓冲和 O(n²) BPE），
不要用普通输入喂长文。

> Windows（MSYS2）上也能编手机引擎：`feng_phone.c` 有 `_WIN32` 分支，走"读进堆"
> 回退（MSYS2 没有 `sys/mman.h`），pthread/信号量由 winpthreads 提供。
> 目的就是让手机端能在开发机上直接编译与验证，不必绕 WSL。

### 编辑器配置（`.vscode/`）与代码提示

`.vscode/` **不入库**（`.gitignore` 已排除）：`settings.json` 里写的是本机绝对路径
（Python 解释器、交叉编译器的 `--query-driver`），换机器就得改，所以它是每台机器自己的。

代码提示的数据源是**仓库根的 `compile_commands.json`**（同样不入库），由
`tools/gen_compile_commands.py` 生成，44 条覆盖五组代码：

| 组 | 条数 | 编译参数来源 |
|---|---|---|
| 固件 C23 核心（`board/main/*.c`） | 16 | **直接取自 ESP-IDF 自己的** `build/compile_commands.json`——自带上百条 `-I`、`sdkconfig.h` 与 `-DFENG_KV_Q2` 这类只在构建期存在的宏 |
| PC 驱动（`pc/*.cpp`） | 15 | 脚本构造：`-std=c++23` + `-Icore` + q2 宏 |
| 手机引擎（`core/*.c` + `phone/engine/feng_{phone,neon}.c`） | 11 | 脚本构造：`-std=c23 -DFENG_PHONE=1` + core/engine 两个 include + MSYS2 的 ucrt64 include（winpthreads 供 pthread） |
| Android JNI | 1 | 同上；探测到 NDK 时补 `--target=aarch64-linux-android` |
| CUDA（`feng_cuda.cu`） | 1 | 脚本探测 `CUDA_PATH`，否则取 NVIDIA 固定安装位置里版本号最大的 |

**增删 `.c`/`.cpp` 文件、或改过 `board/main/CMakeLists.txt` 之后，重跑一次**：

```powershell
python tools\gen_compile_commands.py          # 加 --check 只报告、不写盘
```

不重跑的话新文件拿不到编译参数（退化成"只有 `-Icore`"）。固件那 16 条依赖
`board/build/compile_commands.json`——**没跑过 `idf.py build` 就没有它**，
脚本会明确提示并跳过，其余四组不受影响。

> 引擎两层各有天然的解析边界，所以 x86 主机上也能完整提示：`feng.h` 里的
> `#include "esp_attr.h"` 在 `#if defined(ESP_PLATFORM)` 内，手机端的
> `#include <arm_neon.h>` 在 `#if defined(__aarch64__)` 内——预处理阶段就跳过了，
> 不需要准备 xtensa/arm 的 sysroot。


## 5. 常用命令

当前版本快速路径（构建 + 全量回归 + 标准基准）：

```powershell
python tools\check_all.py                                  # 文档 → 构建 → 单测 → 矩阵 → 位精确
python scripts\bench_standard.py --model v4/base --tag v3_19_pc4    # 英文基准（SciQ / PIQA）
```

数据与训练（Python 一律用上表的 venv 解释器）：

**当前 v4.0 的完整链路见 [`docs/v4-pipeline.md`](docs/v4-pipeline.md)**（数据合并 → 检索重建 →
全参 SFT → 检索回放 → 名字补丁 → 导出/验收），下面这一段是 v3 时代的**历史命令**，
其中的 `archive/…` 目录已删除、不可复现，只作训练方法的参考。

```powershell
# 【历史】v3 渐进长文数据 → 训练 → 8k 对话微调 → 检索 SFT（严格按此顺序）
python scripts\v3_build_stages.py
python scripts\v3_train.py                      # --stages ctx4096 可只跑某阶段；--steps-cap N 冒烟
python scripts\v3_pack_sft.py
python scripts\v3_polish.py
python scripts\v3_build_retrieval.py
python scripts\v3_retrieval_sft.py

# v2 主线（预训练 / SFT / 长文三条分支）
python scripts\v2_train.py --stage pretrain --out v2\stage_pre --lr 3e-3
python scripts\v2_train.py --stage sft --model v2\stage_pre\final --out v2\stage_sft

# v3.11 板端版（算术边界 + Q4 权重/q2 KV 双 QAT；完整命令见 CHANGELOG v3.11 节）
python scripts\v3_7_kv_qat.py --init archive/v3_9\stockfix2 --data archive/v3_7\qat_data.jsonl `
  --mt archive/v3_5d\mt_convs.jsonl --mt-n 280 --identity-n 150 --out archive/v3_10\qat_a `
  --epochs 3 --lr 3e-5 --retr archive/v3_8\retr --retr-n "4096:200:2,8192:60:1" --retr-lr 1e-5
# 算术边界 + 双 QAT（发布版 pol8）
python scripts\v3_11_build_arith_patch.py --out archive/v3_11\arith_patch2.jsonl
python scripts\v3_7_kv_qat.py --init archive/v3_11\pol7 --data archive/v3_11\arith_repair.jsonl `
  --identity-n 100 --out archive/v3_11\pol8 --epochs 2 --lr 8e-6 --batch 24 --max-len 1024 --wqat

# v3.12 PC 版（同一套算术数据走末层微调，完整命令见 CHANGELOG v3.12 节）
python scripts\v3_6_sft_patch.py --init archive/v3_9\stockfix2 --patch archive/v3_11\arith_repair.jsonl `
  --mt archive/v3_5d\mt_convs.jsonl --mt-n 150 --identity-n 80 --out archive/v3_12\arith2l `
  --epochs 3 --lr 3e-5 --train-last 2
python scripts\v3_6_sft_patch.py --init archive/v3_12\arith2l --patch archive/v3_12\pcfix2.jsonl `
  --mt archive/v3_5d\mt_convs.jsonl --mt-n 100 --identity-n 80 --out archive/v3_12\arith2l3 `
  --epochs 4 --lr 1.5e-5 --train-last 2

# v3.13 记忆版（PC 末层微调 + 板端双 QAT；混训数据见 CHANGELOG v3.13 节）
python scripts\v3_13_build_memory.py --out archive/v3_13\memory.jsonl
python scripts\v3_6_sft_patch.py --init archive/v3_12\arith2l3 --patch archive/v3_13\mem_mix2.jsonl `
  --mt archive/v3_5d\mt_convs.jsonl --mt-n 300 --identity-n 80 --out archive/v3_13\mem_pc3 `
  --epochs 2 --lr 1.2e-5 --train-last 2
python scripts\v3_7_kv_qat.py --init archive/v3_11\pol8 --data archive/v3_13\mem_mix2.jsonl `
  --identity-n 80 --out archive/v3_13\mem_board --epochs 2 --lr 8e-6 --batch 24 --max-len 1024 --wqat

# v3.14 tool 版（无算术混训；tool 在 C++ 引擎里）
python scripts\v3_14_build_noarith_mix.py --out archive/v3_14\noarith_mix.jsonl
python scripts\v3_6_sft_patch.py --init archive/v3_9\stockfix2 --patch archive/v3_14\noarith_mix2.jsonl `
  --mt archive/v3_5d\mt_convs.jsonl --mt-n 300 --identity-n 80 --out archive/v3_14\pc2 `
  --epochs 2 --lr 1.2e-5 --train-last 2
python scripts\v3_7_kv_qat.py --init archive/v3_11\pol8 --data archive/v3_14\noarith_mix2.jsonl `
  --identity-n 80 --out archive/v3_14\board --epochs 3 --lr 1e-5 --batch 24 --max-len 1024 --wqat
python scripts\v3_7_kv_qat.py --init archive/v3_14\board --data archive/v3_14\board_memfix.jsonl `
  --identity-n 80 --out archive/v3_14\board6 --epochs 2 --lr 4e-6 --batch 24 --max-len 2048 --wqat

# PC 运行时：核心 C23 + 驱动 C++23，两趟编译（务必带 feng_calc + feng_tools + feng_memory + feng_sample）
gcc -std=c23 -O2 -c ../core/feng_llm.c -o feng_llm.o -I../core    # 每个核心 .c 各来一次
g++ -std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 `
  -o pc_chat_q2b8.exe pc_chat.cpp feng_*.o -I../core -lm
# 更省事：pc/ 下跑 build_pc_chat.ps1（MSVC 两趟：核心 /std:clatest /TC、驱动 /std:c++latest /TP；
#         -MinGW 备用，走 gcc -std=c23 + g++ -std=c++23）
```

评测与导出：

```powershell
python scripts\eval_planA_scope.py archive/v3_9\stockfix2 eval\v3_9_scope_sf2.json
python scripts\eval_longctx_many.py --models "v3_9sf2=archive/v3_9/stockfix2" --n 32 --neg-n 16
python scripts\chat_probe.py --model archive/v3_10\qat_pol3 --out eval\chat_probe_v3_10p3.json
python scripts\chat_multi.py --model archive/v3_10\qat_pol3    # 多轮坍缩检查
python scripts\chat_student.py --model archive/v3_10\qat_pol3 --prompt "你是谁？"
python scripts\export_student_gguf.py --model archive/v3_10\qat_pol3 --out-dir archive/v3_10\gguf
```

ESP32 固件（在 `board\` 下）：

```powershell
# 0) 导出板端模型（从 HF 权重生成 model.bin / tokenizer.bin / ref_logits.bin）
$py = "python"        # 换成装了 torch + transformers 的解释器
& $py board/tools/export_model.py --model v4/hybrid_B_emb319 --out board/model_export_4.0-embed

# 1) PC 端一致性自检（改内核后必跑）
& "<MSYS2>\ucrt64\bin\gcc.exe" -std=c23 -O2 -c ..\core\feng_model.c -o fm.o -I..\core   # 核心逐个编成 .o
& "<MSYS2>\ucrt64\bin\gcc.exe" -std=c23 -O2 -c ..\core\feng_llm.c   -o fl.o -I..\core
& "<MSYS2>\ucrt64\bin\gcc.exe" -std=c23 -O2 -c ..\core\feng_quant.c -o fq.o -I..\core
& "<MSYS2>\ucrt64\bin\gcc.exe" -std=c23 -O2 -c ..\core\feng_smp.c   -o fs.o -I..\core
& "<MSYS2>\ucrt64\bin\gcc.exe" -std=c23 -O2 -c ..\core\feng_tokenizer.c -o ft.o -I..\core
& "<MSYS2>\ucrt64\bin\g++.exe" -std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics `
    -O2 -o pc\pc_check.exe pc_check.cpp fm.o fl.o fq.o fs.o ft.o -I..\main -lm
.\pc\pc_check.exe ..\model_export_4.0-embed ..\model_export_4.0-embed/ref_logits.bin

# 1b) 带 tool 的 PC 运行时 / 板端代理套件（必须带 feng_calc.c + feng_tools.c + feng_memory.c）
#    （核心同样是 gcc -std=c23 逐个编成 .o 后交给 g++ 链接，见上面 pc_check 那段）
& "<MSYS2>\ucrt64\bin\g++.exe" -std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics `
    -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_kv_suite_q2b8.exe `
    pc_kv_suite.cpp fm.o fl.o fq.o fs.o ft.o feng_calc.o feng_tools.o feng_memory.o feng_sample.o -I..\main -lm
.\pc\pc_kv_suite_q2b8.exe ..\model_export_4.0-embed ..\pc\prompt_long.txt 5200    # 27/27 + 4/4

# 2) 编译固件
$env:IDF_TOOLS_PATH = "<IDF 工具链目录>"      # 本机路径见 scripts/local_paths.json
& "<esp-idf>\export.ps1"
idf.py -DFENG_USE_Q2_KV=ON build      # 发布配置：q2 / 2048 ctx

# 3) 烧录（固件；换模型只需后两条；偏移以 partitions.csv 为准）
$esp = "python"                              # 换成带 esptool 的解释器
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash `
    0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin 0x10000 build\feng_30m.bin
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000  model_export_4.0-embed\model.bin
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 model_export_4.0-embed\tokenizer.bin

# 或者直接用一键脚本（路径走参数/环境变量，偏移已对齐 partitions.csv）
.\flash.ps1 -Port COM20 -EspIdfPath "<esp-idf>" -ModelDir .\model_export_4.0-embed

# 4) 串口对话 / 稳定性 / 编码自检
python scripts\esp32_chat.py --port COM20 --question "你是谁？"
python scripts\esp32_multi.py --port COM20
python scripts\esp32_enc_test.py COM20
```

文档自检：`python tools\check_md.py` + `python tools\check_docs.py`（改文档后两个都要跑）。
**一键验收**：`python tools\check_all.py`——按顺序跑 文档自检 → PC 引擎两趟构建 + tool 单测 →
多轮回归套件（`pc_mt_suite`）→ 32 题矩阵 + 算术子集（`pc_kv_suite`）→ fp32 参考 logits
（`pc_check`）；模型导出目录缺失时自动跳过套件。提交前建议至少跑一次。

**CI（.github/workflows/ci.yml）**：干净 clone 上跑 Python 语法 + 四套 C 单测 + 四套 C/Python 交叉验证
（算式 29 / 记忆 36 / 随机数 56 / 时间 52，共 173 条）+ 全引擎编译 + 文档自检；
对应脚本 `tools/check_{tool,mem,rand,time}_parity.py`，本地可单独运行（需先按 build 脚本编出对应驱动）。

## 6. 硬性约束与踩过的坑（改代码前先看）

1. **教师生成与训练严格串行**，绝不同时占卡；27B 教师用 `--no-cuda-graph`、`reasoning_effort:none`、并发 8。
2. **llama.cpp `--parallel N` 会把 `-c` 均分给 N 个槽**：总上下文必须写成「每槽上下文 × N」，否则长提示被静默截断（v1 曾因此作废 2 万条教师数据）。
3. **显存安全**：lm_head + 交叉熵必须走 `scripts/student_utils.py::chunked_lm_loss`（512 token 分块 + checkpoint），32k 阶段峰值 10.28 GiB，16GB 卡上不要并发跑其他任务。
4. **注意力后端**：训练用 `torch.nn.attention.sdpa_kernel` 的 EFFICIENT/FLASH 后端；该 torch 构建的 SDPA 不支持 GQA，所以学生用 MHA（7=7），改 GQA 需同步改 C++ 引擎（`feng_llm.c` 里要加 KV 头广播）。
5. **身份不能掉**：训练语料里身份样本过采样（v1 是 15×），并过滤提及其他 AI 身份（ChatGPT/通义…）的样本；每次出模型都要用 `eval_planA_scope.py` 验证身份题。
6. **flash 前 16MB 的 mmap 窗口是硬边界**（NOR flash 24 位地址上限，**不是模块容量**——模块是 32MB）：`model.bin`（15,659,904 B = 0xEEF380）必须结束在 0x1000000 之前；现行分区为 `model 0x110000/0xEF0000`、`tokdata 0x1000000/0x80000`（tokdata 用 `esp_partition_read` 读，可放窗口外）。烧录偏移必须与 `board/partitions.csv` 保持一致：`flash.ps1` 已按此修正为 `model=0x110000` / `tokdata=0x1000000`，改动分区表时要同步改脚本。
7. **板端内存账**：权重只能 flash mmap 流式读，不能预载进 SRAM（每层 Q4 ≈0.95MB，内部 SRAM 只剩 ~271KB）；
   KV 默认 int8（`FENG_KV_INT8=1`，`MAX_CTX=1024`，9.93MB PSRAM），可选 q2 block8
   （`idf.py -DFENG_USE_Q2_KV=ON build`，`MAX_CTX=2048`，9.62MB，4.0-embed 的 32 题 PC 矩阵 27/27 + 召回 4/4
   + 算术子集 21/21；固件默认多轮上下文、`\reset` 清空，见 `docs/CHANGELOG.md` 的 v3.13 节）；板上 32k 上下文在 KV 内存上不可能，
   长文只能走滑窗/attention sink/线性注意力。
8. **速度现状**：标量路径已到 S3 单发射天花板（每步 ~4-5 周期；短上下文 2.26 tok/s ≈ 938 ms/token @ ctx 19（q2/2048 发布配置））。默认开启三项小幅数值优化（`FENG_GEMV_MADD` 纯 madd 链、`FENG_FAST_EXP` 快速 exp、`FENG_Q2_VFOLD` V 段折叠）：三者累计 logits 差 **4.8e-6**（Q4 量化误差 2.94）、32 题矩阵输出与优化前**逐字相同**。**长上下文成本仍在注意力本体**：q2/2048 单次 forward 从 **2.20 s 降到 1.58 s**（2026-10-18：7 个 head 按 4/3 切给两个核，K/softmax/V 全并行；单核基准 2199 ms → 双核 1584 ms、1024 ctx 1342→1034 ms、256 ctx 694→616 ms，5 条固定提示输出**逐字相同**；开关 `-DFENG_ATTN_PARTS=1` 可回退单核 A/B）。原单核成本 = K 0.84 + softmax 0.05 + V 0.83 + 权重等 0.51；短上下文仍由 Q4 权重流式读主导（2.33 tok/s）。q2 注意力另有四处位精确优化（字节 LUT + `[layer][head][t]` 顺序布局 + 内联 fp16→fp32 + 2-token 展开）；prefill 用 `feng_forward_ex(..., want_logits=0)` 跳过中间 token 的 lm head（每个省 ~115 ms，最后一个 token 必须算）。**A8 整数 GEMV 已实测：S3 上比 FPU 慢 36%（`mull` 慢），只在 PC 上快 20%，默认关闭**。改 GEMV/注意力/布局后必须用 PC 32 题矩阵与上一版对比（位精确改动要求**逐字节**，数值改动要求 27/27+4/4 且给出差异量级）。**PIE 路线已实测结案**：裸吞吐 0.63–1.38 周期/MAC 有空间，但 S3 没有字节移位指令、4-bit 权重必须靠 LUT 展开（≥1 次标量 load+store / 权重），正确的整块内核只做到 **1.05×**——不要再写 PIE 内核（`docs/CHANGELOG.md` v3.16-embed 附录）；不要再做没有实测收益的内层微调。
8.5 **采样器统一**：`core/feng_sample.c` 的 `feng_sample_greedy`（1.15 重复惩罚 + no-repeat 3-gram）是
   固件 / `pc_chat` / `pc_kv_suite` / `pc_mt_suite` 的唯一采样入口；改采样器后必须重跑 PC 32 题矩阵
   并与上一版输出对比（v3.20 那次为**逐字节 0 差异**），再上板。
9. **量化格式耦合**：Q4 block-64（4.25 bpw）；改 `QK` 必须同步改 C 侧 `QK`，且 `board/tools/export_model.py` 会生成 `ref_ids.json` / `ref_logits.bin` 供一致性校验。
10. **SFT 采样顺序坑**：`v2_train.py --stage sft` 默认按长度排序喂样本（省 padding），
   **有界 `--steps` 只会看到最短的一段**——续训恢复实验里 3,000/28,178 步只覆盖最短 10%，
   身份样本完全没轮到（身份 0/12）。短轮/补丁训练请加 `--order interleave`，或跑满一个 epoch。
11. **全参 SFT 会冲掉稀少行为——先量覆盖度，再调超参**（2026-10-17 的教训）：
   v4.0 第一次 `--epochs 1 --lr 2e-5` 把 42 题探针从 42/42 打到 5 题未命中，**表面像学习率过高，
   实际是数据覆盖度**——训练集 47,874 个回答里，「我想死」只有 **1 条**、「我很伤心」2 条、
   「对不起」16 条、翻译成英文 75 条。v3.19 能过是靠 v3.6 手写补丁专门训过。
   **动手调 lr 之前，先统计探针/留出题的目标类别在训练集里有多少条**；
   低于千分之一就先过采样（`v4_build_sft.py --identity-repeat / --extra-repeat`），
   并把定向数据占比拉到 10% 以上。同一次还发现监督量只有 3.24M token（v3 那轮 22.65M）——
   **1 epoch 对 4 万条太少了**。

## 7. 改动的验收清单

- **改了训练脚本**：用 `--steps-cap 1`（或 `--steps`）跑冒烟，确认能落盘 `<阶段>/final/` 与 `summary.json`。
- **改了推理内核（`core/`，C23）/ KV 量化**：必须重跑 `pc_check`（logits MATCH）→ `pc\verify_c_vs_torch.py`（fp32 路径要求 `max|diff| = 0.0000`）→ 板上 `scripts\esp32_multi.py`，并与 `logs\board_baseline_lut.txt` **逐字对比**回复。
- **改了文档**：跑 `python tools\check_md.py`（围栏/路径/过时数字）**和** `python tools\check_docs.py`
  （模型规格 / GGUF 体积与 chat template / v3.9~v3.12 探针、检索、算术分数 / 范围评测 / 检索 loss 与真实产物对齐）。
- **写了数字**：数字必须能追到 `eval/*.json`、`summary.json` 或 `logs/` 里的实测；没有出处的一律删掉或标注"预期/估算"，
  不要写没有日志支撑的精确值。
- **换了模型版本**：重跑 `eval_planA_scope.py` + `eval_longctx.py`，数字同步进 `docs/CHANGELOG.md` / `README.md` / `docs/USAGE.md`
-  （PC 与板端两套口径必须分别标清：**权重** PC=v4.0 / 手机/ESP32=4.0-embed，**引擎**=v4.0（严格 C23）；算式的验收口径是
  **tool 回答**，不是模型算——pc_kv_suite 会把算式任务路由到 feng_calc）。

## 8. 排障速查

| 现象 | 处置 |
|---|---|
| ESP32 推理时 MMU fault / 数据读错 | 检查 `model.bin` 是否越界写进 tokdata；核对 `partitions.csv` 与 PDF 烧录偏移 |
| 板上回复乱码 | 固件默认是 **UTF-8 直通**（GBK 转码默认不编，省 46 KB 映射表）。终端不是 UTF-8 就重编：`idf.py -DFENG_GBK=ON build`，再 `\gbk` 锁定；`scripts\esp32_enc_test.py` 可验两种编码 |
| `idf.py` 报 `MSys/Mingw is no longer supported` | **别在 Git Bash / MSYS 里跑**，改用 PowerShell |
| `idf.py` 报 `No module named kconfgen` | 没进 IDF 的 Python 环境（系统 python 抢在前面了）。先跑 `& "$env:IDF_PATH\export.ps1"`，让 python 指向 `$env:IDF_TOOLS_PATH\python_env\...` |
| 训练 OOM | 降 `--batch` / `--accum`，确认 `chunked_lm_loss` 的 chunk 没被调大，32k 阶段单独跑 |
| 模型答什么都是身份句 | v1 的老毛病，语料配比失衡；参照 v2 的 Plan A 语料 + 低 LR 补训（1000 步 / lr 1.5e-4） |
| 长文阶段跑完检索仍全错 | 长文预训练学不会检索，必须补合成检索 SFT（只对答案算 loss），见 `scripts/v3_build_retrieval.py` |
| esptool 报 port busy | 板子可能不在 USB 列表（换线/供电），或串口被 monitor 占用 |
