# feng-30m 交付清单与技术档案

版本演进与完整实测见 [`CHANGELOG.md`](CHANGELOG.md)，横向对比见 [`COMPARISON.md`](COMPARISON.md)，
使用步骤见 [`USAGE.md`](USAGE.md)。

> **当前交付（2026-10-18 更新）**
>
> **PC 4.0 已定版，手机/ESP32 4.0-embed 已发布**：PC 权重 + 引擎是 4.0 的；
> 4.0-embed 由测试版 `4.0-20261018H` 发布，Q4 网关 27/27 + 4/4（见 CHANGELOG 4.0-embed 节）。
>
> | 项目 | 状态 |
> |---|---|
> | PC 权重（**4.0**） | `v4/patch_name_e5_lr5e-6`（fp32）：留出 **18/30**、第二套 **16/30**（strict-v7）、探针 40/42、范围 10/10、针检索 **103/128**（32k **20/32**，token 精确）、拒答 **64/64** |
> | PC 权重（历史，SFT 起点） | **v3.19**（收在 `v4/base/`）：范围 10/10、探针 42/42、单类别检索 **106/128**（token 精确）、记忆 23/24、留出 17/30（strict-v7） |
> | 手机/ESP32 权重 | **4.0-embed**（导出在 `board/model_export_4.0-embed/`）：Q4 权重 + q2 KV，矩阵 27/27 + 4/4、PPL 102.99、留出 200 85、长文 108/128 |
> | 引擎 | **v4.0 / 严格 C23**（嵌入式核心；PC 侧 C++23 + CUDA）|
> | 板端固件 | q2 block8 / 2048 ctx，app **282,544 B**（比 C 版基线小 24,848 B），**2.33 tok/s**（短上下文/短回复的最佳值；逐 token 随上下文变化） |
> | tool | 算式 / 时间 / 随机数 / 记忆，板内确定性回答（0.5 s），板端 tool **13/13**、记忆 **24/24** |
> | Release | `feng-30m-v4.0-release.zip`、`feng-30m-v4.0-embed-release.zip`、`feng-30m-v4.0-engine.zip`、`feng-30m-v4.0-embed-firmware.zip` |
>
> **权重也可以直接从模型平台取**（不用解压 Release 包）：
> [Hugging Face](https://huggingface.co/jiaheng0815/feng-30m) ·
> [ModelScope 魔搭](https://www.modelscope.cn/models/jiaheng0815/feng-30m)（国内直连）。
> 两个仓库内容一致：HF 格式权重 + tokenizer + esp32 子目录（板端 `model.bin` / `tokenizer.bin`）。
>
> **GGUF / llama.cpp 自 v3.14 起不再发行**（那条路径没有 tool）。
> 本文件 §6/§7 的 PC/板端基准是 **v3.6 时代的技术档案**（原理与优化结论仍适用），
> 最新数字以 [`../README.md`](../README.md) 与 [`CHANGELOG.md`](CHANGELOG.md) 为准。

身份自述（v3.2 起，评测实测原文）：
**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**。

## 1. 当前能力快照

| 维度 | v3.19（PC） | 4.0-embed（手机/ESP32，q2/2048） |
|---|---|---|
| 日常对话探针 42 题 | **42/42** | —（板端抽测工具/记忆/身份） |
| 身份 / 范围评测 | 12/12 / 10/10 | 身份序列 8/8（引擎记忆 tool） |
| 针检索（每长度 32 题，token 精确） | 单类别 28/28/26/24 = **106/128**；多类别 105 是旧口径未重测 | **108/128**（32k 24/32） |
| 记忆（24 题 / 板上 24 题） | 23/24 | **24/24** |
| 留出 30 题（第一套 / 第二套，strict-v7） | 17/30 / 15/30 | 19/30 / 15/30 |
| 32 题矩阵 + 算术子集 | — | **27/27+4/4**、**21/21**（tool） |
| 标准英文基准（0-shot） | SciQ 73.80、PIQA 53.37、其余贴近随机 | 不适用（板端不跑英文基准） |

**v4.0 PC 最终权重**（`v4/patch_name_e5_lr5e-6`）的对应数字：留出 **18/30 + 16/30**（strict-v7）、
探针 40/42、针检索 **103/128**（32k 20/32）、拒答 **64/64** —— 见 `eval/h1_patch_name_e5_lr5e-6.json`、
`eval/h2_patch_name_e5_lr5e-6.json`、`eval/longctx32_v4final.json`。

数字出处：`eval/v3_19pc4_*.json`、`eval/longctx32_v319.json`、`eval/heldout2_*.json`、
`eval/lm_eval_feng_v3_19_pc4.json`、`logs/pc_kv_suite32_v3_19b6_q2b8.txt`、
`logs/board_tools_cpp23_q2_final.txt`、`logs/board_memory_cpp23_q2_final12.txt`。

## 2. 版本要点（完整链条见 CHANGELOG）

| 版本 | 一句话 | 训练量 |
|---|---|---|
| v1 | 8 层 / 32k 词表，只能对话，Q4 27.6 MB 装不进 mmap 窗口 | ≈141M tokens |
| v2 | 16k 词表 + 1.5B 预训练 + 27B 教师 SFT，首次上板 | 1.5B + 22.5M |
| v3 | 渐进长上下文 + 合成检索 SFT（只喂长文学不会检索） | +72M + 12.5M |
| v3.5 / v3.6 | 修多轮复读 / 日常对话大补丁（42 题探针 42/42） | 28M / 38M |
| v3.7~v3.11 | q2 KV-QAT → 算术边界 → **Q4 权重/q2 KV 双 QAT** | ≈36M |
| v3.12~v3.13 | PC 算术修复 / 记忆版（末层微调 + 板端双 QAT） | ≈24M |
| v3.14 | 四个 tool 进引擎（模型不再学算术）；GGUF 停发 | ≈14M |
| v3.15~v3.16 | 身份上下文锚点 / 板端串名修复（p4–p7 确认为容量边界） | ≈数 M |
| v3.17 | 引擎记忆 tool：可枚举问答确定性回答 | 0（引擎改动） |
| v3.19 | 440 条定向教师数据末 2 层 15ep（历史 PC 权重 / 4.0 SFT 起点） | ≈12M |
| **v3.20** | **序列数数 + 统一采样器 + C++23 迁移 + 体积/性能优化 + 标准基准** | 0（引擎改动） |

每阶段超参、loss、耗时、峰值显存见 [`CHANGELOG.md`](CHANGELOG.md) 对应小节；
训练脚本见 `scripts/`（`v2_train.py`、`v3_train.py`、`v3_7_kv_qat.py`、`v3_6_sft_patch.py` 等）。

## 3. 硬件与系统配置

### 3.1 目标板

| 项目 | 值 |
|---|---|
| 模块 | **ESP32-S3-WROOM-2-N32R16V**（32 MB Octal SPI flash + 16 MB Octal SPI PSRAM，VDD_SPI 1.8 V；同系列 N16R8V/N32R8V 已 EOL）。**必须用 WROOM-2**：固件按 OPI flash 构建，WROOM-1（Quad/3.3 V）会烧写或启动失败 |
| CPU | 240 MHz，双核都用（每个 GEMV 按输出行对半分给 core0/core1） |
| Flash | **OPI-DTR 120 MHz**（`ESPTOOLPY_OCT_FLASH` + DTR），流式读实测 **108.3 MB/s** |
| PSRAM | **OCT 120 MHz**（需 `IDF_EXPERIMENTAL_FEATURES`；与 flash 共享 240 MHz MSPI core clock） |
| 模型驻留 | mmap 直接在 flash 里跑（分区 `model` 0x110000–0x1000000），不占 PSRAM |
| KV / 工作区 | PSRAM：**q2 block8 KV，`MAX_CTX=2048`，9.62 MB**（发布配置）；int8/1024 为 9.93 MB |
| 串口协议 | UART0 GPIO43/44，115200 8N1：发一行提问 → 回显 `<< 内容 >>END` 流式输出，`you> ` 提示符 |
| 串口编码 | 默认 UTF-8 直通；GBK 转码默认不编译，需 `-DFENG_GBK=ON` 重编（发布固件未编） |
| 串口命令 | `\stream N`（流式块大小，0 = 整段输出）、`\mem`、`\reset`、`\help` |

### 3.2 关键修复（历史档案，原理仍适用）

1. `CONFIG_ESP_CONSOLE_NONE` 把 panic/日志一起吞掉 → 控制台改回 UART0，输入改用
   `uart_read_bytes`（不再用会崩的 VFS stdin）。
2. 模型分区原来 14 MB，而 `model.bin` 是 **15,659,904 B（0xEEF380）**，越界写进 tokdata
   → 推理时 MMU fault。现布局：`model` 0x110000/0xEF0000（顶到 16 MB 映射窗口边界），
   `tokdata` 挪到 0x1000000/0x80000（用 `esp_partition_read` 读，可放窗口外）。
3. tokenizer 逐 token `malloc` 把内部 RAM 吃光（只剩 19 B）→ 改整块 PSRAM 分配
   （`heap_caps_malloc`，即现在 C++ 里的启动期一次性分配）。
4. 双核 worker 优先级(5)高于主任务(1)：主任务发第一个任务就被抢占，第二个任务晚发 8 ms
   → 两半串行（1.00x）。改为发任务期间临时抬高主任务优先级 → **1.89x**（换 LUT 内核后 1.93x）。
5. Q4 内层循环单累加器串行依赖 → 拆 4 个累加器。

### 3.3 烧录（COM20 = CH343；COM19 = 原生 USB-JTAG）

```powershell
$py='python'    # 换成带 esptool 的解释器
# 固件（首次烧录，或改过 sdkconfig/分区表之后）
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash `
   0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin `
   0x10000 build\feng_30m.bin
# 换模型只需这两条
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000  model_export_4.0-embed\model.bin
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 model_export_4.0-embed\tokenizer.bin
```

对话测试：

```powershell
python scripts\esp32_chat.py --port COM20 --question "你是谁？"
python scripts\esp32_multi.py --port COM20          # 多轮稳定性（默认每题 \reset）
python scripts\esp32_memory.py --port COM20 --n 12  # 记忆 12 组
```

启动日志自带自检：mmap 带宽、双核 GEMV 加速比、KV 配置、tokenizer 自检（`你好` → id 5331）。

## 4. 精度护栏（每次改动都必须过）

- 板上基线：`logs/board_baseline_lut.txt` —— 10 个固定问题（身份/算术/寒暄/拒答/长文），
  贪婪解码逐字记录；**改内核后逐字对比**。
- PC 端：`pc_check`（logits MATCH）+ `pc\verify_c_vs_torch.py` → fp32 路径要求
  `C vs torch(Q4) max|diff| = 0.0000`。
- PC 32 题矩阵：位精确改动要求**逐字节**一致；数值改动要求 27/27+4/4 且给出差异量级。
- 一键回归：`python tools\check_all.py`（文档 → 构建 → 单测 → 矩阵 → 位精确）。
- 板端回归：`scripts\esp32_multi.py`（多轮）、`esp32_memory.py`（记忆 12 组）、
  `esp32_tool_test.py`（tool 13 项）。

## 5. 模型交付物与发布约定

- 主仓库只放代码与文档；权重/二进制（`*.safetensors`、`*.gguf`、`*.npy`、`*.bin`）与
  数据集由 `.gitignore` 排除，随 Release 发布（清单见 README 的"下载"节）。
- 开源数据集只含**教师蒸馏数据**（提示词与教师输出）；本地脚本生成的多轮/补丁/运算数据
  不进 Release 包（可复现，脚本在 `scripts/`）。
- 代码与权重均为 **Apache-2.0**（`LICENSE`）。
- 模型权重、训练产物一旦覆盖无法回滚，删除或覆盖已有模型目录前必须先向用户确认。

## 6. PC 端速度基准（v3.6 权重，2026-10-03 实测；技术档案）

| 运行方式 | prefill | 生成（tg64） | 每 token | 相对 ESP32 |
|---|---|---|---|---|
| **ESP32-S3 双核 + 同款引擎 + Q4 mmap** | 1.9 tok/s | **1.85–1.86 tok/s** | ~540 ms | 1x |
| PC i7-12700KF，**同一个引擎**（单线程、标量内核） | 55.0 tok/s | **54.3 tok/s** | 18.4 ms | ~29x |
| PC 同引擎，LUT 查表内核（板端同款） | 159.2 tok/s | **158.6 tok/s** | 6.3 ms | ~85x |
| PC CPU，llama.cpp Q4_K_M，8 线程 | 2,955 ±657 tok/s | **1,175 ±93 tok/s** | 0.85 ms | ~630x |
| PC GPU（RTX，llama.cpp CUDA，Q4_K_M，-ngl 99） | 26,949 ±10,561 tok/s | **2,638 ±137 tok/s** | 0.38 ms | ~1,420x |

- 数字出处：板端 `logs/board_v3_6_speed.txt`；引擎 `logs/pc_bench_lut_v3_6.txt`（LUT）与标量复测；
  llama.cpp `logs/bench_v3_6_q4km_{cpu,gpu}.log`（`-r 5`）。pp32 波动极大（±30%+），只作参考。
- 同一套引擎在 PC 上给出与板子一致的回复；GGUF 分词 id 与板子逐位一致
  （`1 436 202 5331 2 202 1 442 202`）。
- 板子落后单核 PC（同标量内核）约 29x = 主频（240 MHz vs 5 GHz）× 指令效率。
- 换 LUT 内核后 PC 单线程 ~159 tok/s（板端同款内核），llama.cpp 多线程仍快约 7x（向量化 kernel），
  对应到 ESP32 就是 PIE SIMD 的优化空间。

## 7. 设备端优化档案（2026-10-02 ~ 10-05）

| 改动 | 内容 | 实测 |
|---|---|---|
| **Q4 查表内核** | 把"移位+减8+int→float"换成 256 项浮点查表（2 KB，放内部 RAM） | PC 同引擎 55 → 160 tok/s（2.9x），回复逐字不变，`C vs torch(Q4) max\|diff\|=0.0000` |
| **KV int8** | 每 (层,位置,头) 一个 fp16 scale | 上下文 256 → 1024（同样 ~10 MB PSRAM） |
| **q2 block8 KV** | 2 bit + 每块 fp16 scale | 上下文 1024 → **2048**（9.62 MB），矩阵仍 27/27+4/4 |
| **双 QAT** | 训练时同时注入 Q4 权重与 q2 KV 噪声 | 板端 27/27+4/4 全保，量化抗性对权重回插极敏感 |
| q2 注意力四优化 | 字节 LUT + `[layer][head][t]` 布局 + 内联 fp16→fp32 + 2-token 展开 | 单次 forward 3534 → 2224 ms（-37%），输出逐字一致 |
| prefill 跳过 lm head | `feng_forward_ex(..., want_logits=0)` | 每个中间 token 省 ~115 ms，回答逐字相同 |
| IRAM / 采样器统一 | `FENG_HOT` 热函数；`feng_sample_greedy` 单一入口 | 收益个位数百分比；采样口径板端/PC 一致 |
| **C++23 迁移 + 体积优化** | 零堆抽象、禁用异常/RTTI、自写堆排序（替代 `std::sort`）、切断 libstdc++ 运行时、非热点 `-Os` | 固件 307,392 → **304,192 B**；板端 ms/token 与 C 版逐项相同 |

### 7.1 ESP32-S3 上的"AI 加速"实情

- **S3 没有 NPU**，AI 加速 = **PIE 128 位 SIMD** + 标量 FPU。
- Espressif 官方库（ESP-DL / esp-nn / ESP-DSP）里唯一相关的是
  `dspi_dotprod_off_s8_aes3`（int8 点积 + offset），但输出语义是图像处理，不能直接当 GEMV。
- **PIE 路线已实测结案**：裸吞吐 0.63–1.38 周期/MAC 有空间，但 S3 没有字节移位指令、
  4-bit 权重必须靠 LUT 展开（≥1 次标量 load+store / 权重），正确的整块内核只做到 **1.05×**
  （`CHANGELOG.md` v3.16-embed 附录）——不要再写 PIE 内核。

### 7.2 标量内核已到极限

| 实验 | 板上 GEMV 896×448（单核 / 双核） | 结论 |
|---|---|---|
| 内层展开 4 → 8 字节 | 16,502 / 8,552 µs | 慢 25%（寄存器溢出），回退 |
| 编译档 -O3 → -O2 | 13,187 / 6,828 µs | 无差别 → 编译器不是变量 |
| 现役（LUT + 4 累加器 + 双核 + IRAM） | **13,187 / 6,832 µs** | 17.0 ns/权重 = **4.1 周期/权重 @240 MHz** |

理论天花板：每 token 读 14.93 MB ÷ 108.3 MB/s = **138 ms → 同架构最多 ~7.2 tok/s**；
现在 938 ms 里绝大部分是计算，标量路径没有剩余空间。

## 8. 下一步（若要继续提升）

1. **模型**：同尺寸继续堆预训练 token（续训实验见 CHANGELOG v3.20 节），或换更大模型。
2. **数据**：教师数据从 1.4k 扩大到 1–2 万条（27B 教师 8 并发约 12.8 tok/s）。
3. **嵌入式**：注意力/权重仍是瓶颈；PIE 已证伪，可行的只有换芯片（带 NPU）或进一步量化。
4. **评测**：把标准基准扩展到 v1/v2/v3 全代（当前只有 v3.19/pc4）。
