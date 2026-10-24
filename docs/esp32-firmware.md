# feng-30m on ESP32-S3-WROOM-2-N32R16V（32 MB Octal flash + 16 MB Octal PSRAM）

把 **feng-30m**（Qwen3 架构：11 层 / hidden 448 / 7 头 MHA / FFN 896 / 16k 词表 /
tied embedding，29.43M 参数）量化成 Q4 后**在 ESP32-S3 上离线对话**。

> **当前固件（2026-10-18：引擎 v4.0；手机/ESP32 权重 4.0-embed）**
> - 权重：**4.0-embed**（导出在 `../board/model_export_4.0-embed/`，Q4 权重 + q2 KV；测试版 `4.0-20261018H`）
> - 引擎：**v4.0 / 严格 C23**（零堆、零全局构造；PC 侧仍 C++23 + CUDA），app 分区 **282,544 B**
> - 配置：**q2 block8 KV / 2048 ctx**（`idf.py -DFENG_USE_Q2_KV=ON build`），KV 9.62 MB PSRAM
> - 记忆：工具抽取的事实写 **NVS**，断电/重启不丢；`\reset` / `\mem clear` / 写满自动开新对话时清空
> - 实测：矩阵 **27/27 + 4/4**、算术子集 **21/21**、tool **13/13**、记忆 **12/12**、
>   速度 **2.33 tok/s**（短上下文/短回复的最佳值；逐 token 随上下文变化）

## 0. 硬件要求（复现必读）

实测板：**ESP32-S3-WROOM-2-N32R16V**——32 MB **Octal SPI** flash + 16 MB **Octal SPI** PSRAM，
VDD_SPI 1.8 V（同系列 N16R8V / N32R8V 已 EOL）。

| 约束 | 原因 |
|---|---|
| 必须是 **WROOM-2**（Octal / 1.8 V） | `sdkconfig` 开了 `ESPTOOLPY_OCT_FLASH=y` + `FLASHMODE_OPI`；换成 WROOM-1（Quad / 3.3 V）会烧写或启动失败 |
| Flash 必须 **32 MB** | 只有**前 16 MB 能被 mmap**（NOR flash 24 位地址上限），模型必须落在这里；16 MB 以上的部分用 `esp_partition_read` 读，`tokdata` 就放在那里。16 MB 模块放不下整个布局 |
| PSRAM 必须 **16 MB** | 2048 ctx 的 q2 KV = 9.62 MB + 工作区/分词 ≈1 MB；8 MB 版本不够 |
| 120 MHz PSRAM 需要 `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` | 已在 `sdkconfig` 里开启，换 IDF 版本时别关掉 |
| **GPIO33–37 不可用** | WROOM-2 的 Octal flash 占用这几个引脚（模块也未引出） |

> **别把两个「16 MB」搞混**：**mmap 窗口 = 16 MB**（NOR flash 24 位地址上限）是地址限制；
> **模块 flash = 32 MB** 是容量。模型放前 16 MB 内做 mmap 直读，tokenizer 放在 16 MB 之后用
> `esp_partition_read` 读——所以板子必须是 32 MB flash，而不是 16 MB。

```
┌─ ESP32-S3-WROOM-2-N32R16V（32MB Octal flash + 16MB Octal PSRAM，1.8V） ──────────┐
│  flash: 0x110000   model.bin      14.93 MB, Q4 block-64, mmap 直读（不占 RAM）   │
│         0x1000000  tokenizer.bin  413 KB, 启动时读入 PSRAM                       │
│         0x0010000  app            282,544 B（factory 分区 1 MB，余 73.0%）         │
│  PSRAM: KV cache q2/block8 2048 ctx = 9.62 MB + 工作区/分词 ≈ 1.0 MB             │
│  时钟 : CPU 240MHz ×2   flash OPI-DTR 120MHz   PSRAM OCT 120MHz                  │
│  串口 : UART0 115200 8N1（`you> ` 提示符，`<< 内容 >>END` 流式回复）             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

板子开机自检可复核以上配置（见 `../logs/board_baseline_lut.txt`）：ROM 打印
`Octal Flash Mode Enabled`、`SPI Flash Size : 32MB`，PSRAM 打印 `VCC 0x00 (1.8V)`、
`Found 16MB PSRAM device`、`Speed: 120MHz`，mmap 流式读 **108.3 MB/s**。

瓶颈是"每个 token 都要把 15 MB 权重从 flash 读一遍 + 逐权重标量计算"，实测证明是**算力瓶颈**
（flash 提频 80→120 MHz 只快 5%，双核则 +89%）。

## 1. 固件能力与行为

### 1.1 四个 tool（模型不再学算术）

实现见 `core/feng_calc.c`、`core/feng_tools.c`、`core/feng_memory.c`；
三处（固件 / PC 引擎 / Python）同口径，另有 173 条交叉验证进 CI。

- **算式**：`4854+4411`、`5.3+4.1`、`(3+4)*2` → 0.5 s 秒回，SoC 运算器直接算；
  支持中文数字、`乘以/除以`、百分号、平方/根号、**序列数数**
  （`把 1 到 5 倒着数一遍` → `5、4、3、2、1。`）。
- **时间**：`现在几点？` → UTC+8 日历；`3天后是几号`、`时间戳`、时钟推算
  （`现在7点，再过3小时是几点？`，跨天自动说"明天/昨天"）。
  板端放不下 WiFi 协议栈，由宿主连接时发 `\settime <unix秒>`（宿主走 NTP）对时，
  固件用 `esp_timer` 走时。
- **随机数**：`给我个1到100的随机数` → xorshift64*，seed = 运行时间×1.54×1000，
  丢弃第一个取第二个；`掷骰子`、`抛硬币`、区间解析。
- **记忆**：`我叫小明，请记住` / `我最喜欢的颜色是蓝色` / `我住在成都` / `我养了一只乌龟`
  → 事实存进引擎（不拦截，模型照常回应）；追问 `我叫什么名字？` / `我最喜欢什么颜色？`
  / `你叫什么名字？` **由引擎 0.5 s 确定性作答**（身份永不串名）。
  另有**通用键值槽**（`我的生日是3月5日` → `我的生日是几号？`，满 8 条轮换最旧）、
  **列出**（`你还记得什么？`）与**遗忘**（`忘掉我的颜色`）。
  只覆盖可枚举句式（单测 69 项），其他说法仍走模型自己的多轮能力。

### 1.2 多轮上下文（v3.13 起默认保留）

每轮把 `<|im_end|>` 补进 KV，下一轮只 prefill 新增的用户片段。
实测 `我叫小明，请记住 → 你叫小明`、`喜欢蓝色 → 你最喜欢蓝色`、`养了一只猫 → 你养了一只猫`
全对；`\reset` 清空上下文，写满 2048 自动开新对话。

压测记录：v3.19-embed **32 轮连续压测 32/32 成功**（`../logs/board_v3_19b6_stress32.txt`）；
v3.17 固件 **64 轮 64/64 成功**（8 问循环 ×8，平均 12.5 s/轮，最深 33.4 s，
其中 24 轮由 tool 0.5 s 秒回；`../logs/board_v3_17_stress64.txt`）。

### 1.3 输入长度与 prefill 成本

单行最多 4095 字节，超出会提示并丢弃多余部分；正文按 token 预算截断时保留 UTF-8 边界与模板收尾。
板端 prefill 成本 = **每 token 权重 ~0.38 s + 注意力 0.85 ms×n²/2**：491 tokens 的长文实测
**291 s**（分两轮：正文一行、提问一行）。固件在 >80 tokens 时打印预估时间；
**交互输入建议 ≤ ~150 tokens（≈400 字节中文）**，长文请在 PC 上跑。
日志：`../logs/board_longprompt_2turn_491tok.txt`、`../logs/board_longprompt_2turn.txt`。

## 2. 拿模型（PC 上执行，二选一）

推荐直接下载 Release 附件 **`feng-30m-v4.0-embed-release.zip`**，取其中 `weights/esp32/`
（已导出，板端/PC 引擎通用）。
想自己导出（当前板端权重 = 4.0-embed，Q4 权重 + q2 KV）：

```powershell
$py = "python"                    # 换成装了 torch + transformers 的解释器
& $py board\tools\export_model.py --model <板端 HF 权重目录> --out board\model_export_4.0-embed
# -> model_export_4.0-embed/model.bin      14.93 MB（Q4 块64 + fp16 norms/scales）
#    model_export_4.0-embed/tokenizer.bin  413 KB
#    ref_logits.bin / ref_ids.json / export_info.json
```

> 注意：PC 的 HF 权重没做量化感知训练，导出给引擎会退化（同套 32 题矩阵 22/27 vs 27/27，
> `../logs/pc_kv_suite32_v3_14pc2_q2b8.txt`）。引擎只认板端双 QAT 权重。

## 3. PC 端一致性自检（强烈建议先跑）

同一套 C23 核心在 PC 上编译运行（驱动仍是 C++23），与 PyTorch 参考逐值比对：

```powershell
$src = @('pc/pc_check.cpp','core/feng_model.c','core/feng_llm.c','core/feng_quant.c',
         'core/feng_smp.c','core/feng_tokenizer.c','-Icore','-lm')
& "<MSYS2>\ucrt64\bin\g++.exe" -std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics -O2 `
  -o pc\pc_check.exe @src
pc/pc_check.exe board/model_export_4.0-embed board/model_export_4.0-embed/ref_logits.bin
```

> 路径换成你本机的即可；`model_export_4.0-embed` 也可指向 Release 包里 `model.bin` 所在目录。
> 更省事：`.\build_pc_chat.ps1`（Windows 默认 **MSVC 两趟**，自动加载 vcvars64：
> 核心 `/std:clatest /TC`、驱动 `/std:c++latest /TP`）
> 一次构建全部 13 个 PC 产物；上面的 g++ 命令等价于 `-MinGW` 备用路径。

期望输出（4.0-embed 实测）：

```
logits check: n=16384 max|diff|=1.2744 mean|diff|=0.41395  argmax c=5331 ref=5331 MATCH
```

这里的 1.27 是**纯 Q4 量化误差**（对 fp32 参考）。要验证实现本身是否等价，再跑：

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& "<带 torch 的 python>" pc\verify_c_vs_torch.py `
   --export ..\model_export_4.0-embed --model <4.0-embed包>\weights\hf `
   --c-logits ../model_export_4.0-embed/ref_logits.bin
# [C vs torch(Q4)] max|diff| = 0.0000   ← 实现逐位一致
```

> **可选 CUDA 加速**（RTX 显卡）：`.\build_pc_cuda.ps1` 用 MSVC + nvcc 构建
> `pc\pc_chat_cuda.exe` / `pc_check_cuda.exe` / `pc_kv_suite_cuda.exe` 等，
> GEMV 走 GPU。实测生成 157 → **322 tok/s**（约 2.0×），32 题矩阵输出与 CPU 版逐字一致；
> prefill 因 PCIe 往返固定开销变慢（59 → 110 ms/12 token）。板端固件不含 CUDA。
> 这个脚本只在 PC 上用；`FENG_CUDA=0` 可强制走 CPU 路径做 A/B。

## 4. 编译固件（ESP-IDF v5.5.5）

```powershell
$env:IDF_TOOLS_PATH = "<IDF 工具链目录>"
& "<esp-idf 目录>\export.ps1"
cd <仓库>\board
idf.py -DFENG_USE_Q2_KV=ON build     # 发布配置：q2 / 2048 ctx；不带 -D 则 int8 / 1024
```

> 目标芯片、分区表与 flash/PSRAM 配置都已在 `sdkconfig.defaults`、`partitions.csv`、
> `main/CMakeLists.txt` 里写好。C23 与体积约定：
> 全组件 `-std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics`；
> 非热点模块（tokenizer / tool / model / gbk / main）用 `-Os`，推理热点
> （`feng_llm` / `feng_quant` / `feng_smp` / `feng_sample`）保持 `-O2`。
> 当前 app **282,544 B**（关掉彩蛋 282,336 B、v3.20 的 C++23 版 304,192 B、C 版基线 307,392 B）；新增代码后请对比该体积。

## 5. 烧录（COM20 = CH343；COM19 = 芯片原生 USB-JTAG，两个都能烧）

```powershell
$py = "python"        # 换成带 esptool 的解释器（ESP-IDF 自带的那个也行）
# ① 固件（首次烧录、或改过 sdkconfig/分区表之后）
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash `
    0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin `
    0x10000 build\feng_30m.bin
# ② 换模型只需这两条
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000  model_export_4.0-embed\model.bin
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 model_export_4.0-embed\tokenizer.bin
```

> 偏移量必须与 `partitions.csv` 一致（`model` 0x110000 / `tokdata` 0x1000000），
> 否则会出现 MMU fault 或读不到模型。也可以直接用一键脚本：
> `.\flash.ps1 -Port COM20 -EspIdfPath <esp-idf> -ModelDir ..\model_export_4.0-embed`。
> **不想装 ESP-IDF**：Release 附件 `feng-30m-v4.0-embed-firmware.zip` 已含
> bootloader + 分区表 + app + 模型 + 哈希与烧录命令，esptool 一次烧完。
> 模型写 15 MB 约需 3.5 分钟（921600 波特率，压缩后约 13 MB）；
> esptool 报 "port is busy or doesn't exist" 时先检查线/供电。

## 6. 串口对话协议

| 项目 | 值 |
|---|---|
| 波特率 | **115200** 8N1（UART0 = GPIO43/44） |
| 输入 | 一行一句（回车发送），串口会回显你输入的字符 |
| 输出 | 流式 `<< 内容 >>END`；默认按句读符或 30 字节成块发送，减少终端时间戳刷屏 |
| 尾行 | `>>END (ctx N/2048, X.XX tok/s)` —— 与 PC 版同口径 |
| 编码 | 默认 **UTF-8 直通**；GBK 转码默认不编译（省约 46 KB 映射表），需要时用 `idf.py -DFENG_GBK=ON build` 重编 |
| 命令 | `\utf8`、`\stream N`（流式块大小，0 = 整段一次性输出）、`\mem`、`\reset`、`\help`；编了 GBK 才有 `\gbk` |

开机自检会打印：mmap 带宽、双核 GEMV 加速比、KV 配置、tokenizer 自检（`你好` → id 5331）。
PC 侧测试脚本：

```powershell
python scripts\esp32_chat.py    --port COM20 --question "你是谁？"
python scripts\esp32_multi.py   --port COM20                 # 多轮（默认每题 \reset）
python scripts\esp32_memory.py  --port COM20 --n 12          # 记忆 12 组
python scripts\esp32_tool_test.py --port COM20               # 时间/随机数/算式 13 项
python scripts\esp32_enc_test.py COM20                       # 双编码自检
```

## 7. 关键参数与调优

| 项 | 值 | 说明 |
|---|---|---|
| 上下文 | **2048 token（发布配置）** | q2 block8 KV = 9.62 MB PSRAM；不带 `-DFENG_USE_Q2_KV=ON` 时是 int8 / 1024 ctx（9.93 MB） |
| KV 量化 | q2 block8（2 bit + 每块 fp16 scale）或 int8 + 每 (层,位置,头) fp16 scale | 开关在 `main/CMakeLists.txt` |
| 生成长度 | 96 token | `MAX_NEW` |
| 采样 | 贪心 + 重复惩罚 1.15 + no-repeat 3-gram | `core/feng_sample.c`，与 PC 引擎/套件同口径 |
| 量化 | Q4 block-64（4.25 bpw） | `board/tools/export_model.py`；改 `QK` 需同步改 C++ 侧 `QK` |
| 内核 | Q4 查表（256 项浮点 LUT）+ 4 累加器 + 双核分半 + IRAM | 见 `feng_quant.c` / `feng_smp.c` |
| 速度 | **2.33 tok/s** | 固件统计口径（含 prefill）；短上下文/短回复最佳值，逐 token 随上下文变化 |
| 长上下文 forward | **1.58 s @2048 ctx**（单核 2.20 s，双核按 head 切分 1.39×） | `../logs/board_attn_par_20261018.txt` |
| 体积 | **282,544 B** | 非热点 `-Os` + 热点 `-O2`；C23 核心 + C++23 驱动约定见根 `../AGENTS.md` |

## 8. 已验证结果（v3.19-embed + q2/2048，2026-10-05 实机）

PC 一致性（`model_export_v3_19b6` 实测）：

```
model: 11 layers hidden 448 heads 7 x 64 ffn 896 vocab 16384 rope 1000000
tokenizer: vocab 16384 merges 16124 im_start=1 im_end=2 eot=0
encode("你好") -> 9 tokens: 1 436 202 5331 2 202 1 442 202   ← 与 PyTorch 分词完全一致
logits check: n=16384 max|diff|=2.6604 mean=0.45821  argmax c=5331 ref=5331 MATCH
[C vs torch(Q4)]           max|diff| = 0.0000   ← 实现逐位一致
[torch(Q4) vs torch(fp32)] max|diff| = 2.6604   ← 纯 Q4 量化误差（预期）
```

板子启动自检（节选）：

```
I (961)  feng: mmap stream read: 15296 KB in 144 ms -> 108.3 MB/s
I (1827) feng: KV cache: q2/block8, ctx 2048, 9.62 MB
I (1828) feng: PSRAM free after setup: 5006 KB
I (2228) feng: gemv 896x448: 1-core 13186 us | 2-core 6837 us | speedup 1.93x
```

对话与工具实测：

| 项目 | 结果 | 出处 |
|---|---|---|
| 工具（时间/随机数/算式） | **13/13** | `../logs/board_tools_cpp23_q2_final.txt` |
| 跨轮记忆 12 题 | **12/12** | `../logs/board_memory_cpp23_q2_final12.txt` |
| 8 轮身份序列 | **8/8**（引擎记忆 tool） | `../logs/board_v3_16p3_identity_ctx_engmem.txt` |
| 32 轮连续压测 | **32/32** | `../logs/board_v3_19b6_stress32.txt` |
| 速度 | **2.26 / 4.62 / 8.05 tok/s**（ctx 19/38/54） | 见`CHANGELOG.md` v3.20 节 |
| **2026-10-18 复测**（当时固件 280,736 B；2026-10-22 引擎加固后为 281,248 B） | tool **13/13**、记忆 **12/12**、多轮 **10/10**、身份句逐字正确 | `../logs/board_tools_time_rand.txt`、`../logs/board_memory_24.txt` |
| **2026-10-18 4.0-embed 发布验收** | Q4 27/27 + 4/4、PPL 102.99、留出 200 85、长文 108/128；板端 tool 13/13、记忆 24/24、多轮 10/10 | `../logs/gate_hybrid.txt`、`../eval/ppl_hybrid_B_emb319.json` |

历史记录：v3.11/v3.10/v3.7 q2 矩阵 27/27+4/4、v3.6 int8 1.85–1.86 tok/s、
v3.4 10/10、v3 5 轮 5/5、v2 固件 10/10（各日志见 `../logs/`）。

## 9. 目录与内核结构

```
core/                      共享推理核心（严格 C23，board / pc / phone 编同一份源码）
├── feng.h                 模型容器 / 推理 API / KV 模式开关（带 extern "C" 守卫）
├── feng_model.c           model.bin 解析、张量定位与边界校验
├── feng_quant.c           Q4 查表 / fp16 GEMV（启动期初始化 LUT）
├── feng_llm.c             RMSNorm / QK-norm / RoPE / MHA / SwiGLU / tied head / KV
├── feng_smp.c             双核 GEMV（PC 上退化为单核直通）
├── feng_sample.c          统一采样器（重复惩罚 + no-repeat 3-gram）
├── feng_calc.c / feng_tools.c / feng_memory.c   四个 tool
└── feng_tokenizer.c       设备端 BPE（编译期字节表 + 堆排序）

board/
├── main/                  ESP-IDF 应用层（严格 C23）
│   ├── main.c             分区 mmap + PSRAM 分配 + 串口对话 + 命令链
│   ├── feng_bonus.c       彩蛋（`-DFENG_NO_BONUS=ON` 可关，别当死代码删）
│   ├── feng_gemv_bench.c / feng_pie_*.S   GEMV / PIE 基准（不参与发布路径）
│   └── gbk.c / gbk_table.c   串口编码转换（`-DFENG_GBK=ON` 才编）
├── tools/export_model.py  HF → model.bin / tokenizer.bin / 参考 logits
├── tools/gen_gbk_table.py 生成 GBK 转换表
├── model_export_4.0-embed/  当前板端权重（model.bin / tokenizer.bin / ref_*）
├── flash.ps1              一键烧录（校验大小、偏移、退出码）
└── partitions.csv / sdkconfig.defaults / CMakeLists.txt

pc/                         PC 驱动与验收套件（C++23）
├── pc_check.cpp            一致性验证（对比 ref_logits / PyTorch）
├── pc_bench.cpp            速度基准（同引擎单线程）
├── pc_mt_suite.cpp         多轮回归套件（报名字→身份/名字、12 题记忆）
└── gbk_selftest.cpp        编码转换自检（需 `-DFENG_GBK=ON` 的核心）
```

> 本引擎按 **MHA（7 个 Q 头 = 7 个 KV 头）** 实现；若改成 GQA/MQA（如 7 头 / 1 KV 头），
> 需要在 `feng_llm.c` 里加 KV 头广播。（v1 也是 MHA，网上"v1 是 MQA"的说法不成立。）
