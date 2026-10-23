# phone/ —— 让 feng-30m 在手机上跑

同一套引擎（v4.0 起的 C23 核心），发布权重为 **4.0-embed**（与 ESP32 同包），给手机两条路：

| 路径 | 形态 | 适合谁 |
|---|---|---|
| **Termux CLI** | 在手机上原生编译的 `phone_chat`，命令行对话 + 基准 | 想在终端里用、想自己改/调参 |
| **Android APK** | Kotlin + Jetpack Compose 界面，JNI 调同一个引擎 | 只是想装个 App 用 |

两条路用的是**同一批源文件**（`core/` + `phone/engine/` 里的平台层），不存在「CLI 一个引擎、App 另一个引擎」。

- 完全离线：APK 不申请任何权限，模型随包打进去；Termux 侧只读本地文件。
- 四个 tool 全保留：算式、记忆、时间、随机数走引擎侧确定性回答，不交给模型硬算。

---

## 1. 快速开始：Termux

```bash
pkg install -y clang
git clone <仓库地址> feng-30m && cd feng-30m/phone/cli
./build.sh                       # 产出 build/phone_chat

# 模型：从 Release 的 feng-30m-v4.0-embed-release.zip 解出 weights/esp32/ 下的 model.bin / tokenizer.bin
mkdir -p ~/.feng && cp model.bin tokenizer.bin ~/.feng/

./build/phone_chat ~/.feng                      # 开始对话
./build/phone_chat --bench 32 ~/.feng            # 测 tok/s
FENG_PHONE_THREADS=4 ./build/phone_chat ~/.feng  # 指定线程数
```

REPL 里：`\reset` 清空上下文与记忆、`\mem` 看记忆、`\stats` 看运行状态、`\exit` 退出。
记忆会写到模型目录的 `mem.txt`（APK 为应用私有目录 `filesDir` 下的 `mem.txt`），重启/重开应用后仍在；
`\reset`、`\mem clear` 或上下文写满会自动清空并删除该文件。

常用参数：

| 参数 | 说明 |
|---|---|
| `--ctx N` | 上下文长度，默认 2048（上限 = 编译期 `FENG_PHONE_MAX_CTX`，默认 **32768**；32k 的 q2 KV = 154 MB） |
| `--threads N` | 线程数，默认 `min(4, 在线 CPU 数)` |
| `--neon 0\|1` | 强制 NEON 内核开关（A/B 用） |
| `--bench [N]` | 基准：固定问题生成 N 个 token 并打印 tok/s |
| `--ab` | 自检：NEON 开/关各跑一遍，比对 logits 与 argmax |
| `--prompt TEXT` | 问一句就退出（脚本化用） |
| `--ids-file F` | 直接喂预 tokenized 的 id 文件（长文 32k 验收；绕过 4KB 输入与 BPE） |

环境变量：`FENG_PHONE_THREADS`、`FENG_PHONE_NEON`（运行时覆盖编译期默认）。

32k 长文验收（预 tokenized，绕过 4KB 输入缓冲）：

```bash
# PC 上生成 ids（4k/8k/16k/32k，各 8 正 + 2 负）
python scripts/make_longctx_ids.py --ctx 4096,8192,16384,32768 --n 8 --neg-n 2
adb push eval/longctx_ids /data/local/tmp/feng/ids
# 手机上跑（4k 约 2 分钟/题；32k 约 2 小时/题）
adb shell "cd /data/local/tmp/feng && ./phone_chat32k --ctx 4096 \
  --ids-file ids/ctx4096_000.ids ."
```

---

## 2. 快速开始：APK

```powershell
# 需要 JDK 17~21（AGP 8.x 不支持 24+）与 Android SDK；脚本会自动下缺的部分
.\phone\tools\build_apk.ps1
```

脚本做四件事：找/装工具链 → 把模型拷进 `phone/android/` 的 assets → 下载 Gradle → `gradle assembleRelease`，
最后打印 APK 路径。装到手机：`adb install -r <apk>`。

也可以直接用 Android Studio 打开 `phone/android/` 构建，但要先把模型拷进去——
那一步只有 `phone/tools/build_apk.ps1` 会做，所以先跑它一次（`-SkipModel` 默认就是关的），
或者自己把 `model.bin` 与 `tokenizer.bin` 放进 `phone/android/app/src/main/assets/`。

**注意**：模型文件不入库（仓库约定：主仓库只放代码与文档），所以 `phone/android/app/src/main/assets/` 里平时是空的，
必须由构建脚本填充。

**体积（P30 Pro 实测）**：debug APK **31.06 MB**（2026-10-18）。模型 16.1 MB
（15.66 + 0.42）是硬底，其余是 Compose 运行时与原生库。

**上下文默认值（2026-10-18 起）**：APK 默认 **32k**（q2 KV 154 MB）；
初始化失败会自动回退 2048。CLI 仍默认 2048，需要 32k 时显式
`--ctx 32768`。32k 的代价与质量注意见 §7。

---

## 2.5 fp16 权重与 fp16 KV（可选）

CLI 默认是 **Q4 权重 + q2/block8 KV @2048**（与板端一致，352 tok/s，KV 9.62 MB）；
APK 默认 32k（KV 154 MB）。
想换 fp16 精度：

```bash
# KV：编一份 fp16 KV 的引擎（2048 上下文 38.50 MiB）
KV=fp16 ./build.sh
./build/phone_chat --bench 32 ~/.feng

# 权重：用导出器把 HF 权重导成 fp16（56 MB）
python board/tools/export_model.py --model <HF 权重> --out <目录> --weights-dtype fp16
./build/phone_chat <fp16 目录>
```

| 组合 | 32 题 | KV 占用 | 说明 |
|---|---|---|---|
| Q4 + q2（默认） | **27/27 + 4/4** | 9.62 MiB | 板端发布配置 |
| fp16 权重 + q2 | 22/27 + 4/4 | 9.62 MiB | 见下 |
| Q4 + fp16 KV | 26/27 + 4/4 | 38.50 MiB | 精度参照 |

> **fp16 权重在做过 QAT 的权重上并不会更准**：那批权重是照着 Q4 噪声训出来的，
> fp16 反而回到它未适配的数值路径（22/27 正好等于"PC fp32 权重直接导出"的历史数字）。
> fp16 适合**没做过 QAT 的模型**，或用来做精度对拍。
>
> NEON fp16 内核（`vcvt_f32_f16` + `vfmaq_f32`）在支持 FP16 的 aarch64 上自动启用，
> 用 `--neon 0` 可关掉；它与标量路径**不逐位一致**（累加顺序不同）。
>
> **Q4 NEON 内核（int8 `vdotq_s32`）是近似路径**：激活量化到 int8 后与 Q4 权重点积，
> 与标量 fp32 LUT 路径有数值差（argmax 可能变）。2026-10-18 在 P30 Pro 上用
> NDK arm64 实测：32 题矩阵仍是 **27/27 + 4/4**、算术 **21/21**，但**裸模型**的
> `pc_mt_suite` 从 NEON 关的 `mem12 12/12 / seq 10/10` 掉到 **9/12 / 9/10**。
> 产品路径不受影响——固件/CLI/APK 都会先过 `feng_memory` tool，身份与可枚举记忆
> 是确定性回答；这个差异只出现在"关掉 tool、纯模型多轮"的验收套件里。
> 需要与板端/PC 逐字对拍时，用 `FENG_PHONE_NEON=0` 或 `--neon 0`。

## 3. 性能与内存

### 3.1 实测数字（都能追到 `logs/`）

| 场景 | 数字 | 出处 |
|---|---|---|
| 32 题矩阵（q2/block8，2048 ctx） | **27/27 + 4/4 长文召回** | `logs/phone_4t_kv.txt` |
| 算术子集（tool 口径） | **21/21** | `logs/phone_arith.txt` |
| 多轮记忆（**裸模型**） | **mem12 9/12（NEON 开，默认）**；关 NEON 后 **12/12** | 2026-10-18 P30 Pro 复测，见下 |
| 4k prefill（`--ids-file`，4064 tok） | **58 s**（q2 注意力四核并行；串行 107 s） | 2026-10-18 P30 Pro |
| 8k prefill（8160 tok） | **210 s** | 同上 |
| 32k prefill（外推） | **约 1 h** | 4k/8k 的 O(n²) 外推 |
| 与板端原始引擎输出对比 | **逐字节相同**（单线程与 4 线程都是） | `logs/phone_base_kv.txt` vs `logs/phone_1t_kv.txt`、`logs/phone_4t_kv.txt` |
| 与 gcc 版对比：**clang 构建**（Termux 用的就是 clang） | 32 题 27/27，输出**逐字节相同** | `logs/phone_clang_kv.txt` |
| NEON 内核 vs 标量 A8 路径 | **max\|diff\| = 0.000e+00**（逐位一致） | `logs/phone_a64_ab_a8.txt` |
| NEON 内核 vs 标量 fp32 路径 | max\|diff\| = 1.683，argmax 一致 | `logs/phone_a64_ab.txt` |
| Android 原生层（NDK clang，arm64-v8a） | 编出 `libfeng.so` 80.7 KB，8 个 JNI 符号齐全，含 `sdot` | `phone/tools/check_android_native.sh` |
| 内存：权重 | 14.93 MiB（文件 mmap，不计入匿名内存） | 模型导出目录 |
| 内存：KV（q2/block8 @2048） | 9.62 MiB | `feng_kv_bytes()` |

### 3.2 线程扩展（本机 x86_64，不是手机）

`phone/cli/bench.sh` 的产出（`logs/phone_bench_jiaheng.txt`）：

| 线程 | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| tok/s | 140 | 244 | 403 | 496 | 501 |
| 相对单线程 | 1.00x | 1.73x | 2.87x | 3.54x | 3.57x |

单线程 → 8 线程约 **3.6 倍**。手机上的绝对数字取决于 SoC（大核主频、内存带宽）
与散热，**必须在手机上用 `--bench` 复测**，不要拿这张表当手机性能。

并行只在**输出行**维度切分（q/k/v 与 gate/up 三个矩阵合并成一个并行区间，其余
逐层提交），所以结果与单线程逐位相同；每次领任务按 32 行一块，避免高线程数下
几万个行号挤在同一个原子计数器上。

### 3.3 aarch64 上把两个变量拆开（qemu 模拟，**只有比值有意义**）

`phone/cli/build_wsl_aarch64.sh` 的产出（`logs/phone_a64_t*_n*.txt`），同一个问题
生成 30 个 token：

| 配置 | 1 线程 | 4 线程 |
|---|---|---|
| NEON 关 | 4.7 tok/s | 16.3 tok/s |
| NEON 开 | 17.8 tok/s | **51.0 tok/s** |

读法：**NEON 单独约 3.8 倍**（1 线程 4.7 → 17.8）、**多线程单独约 3.5 倍**
（NEON 关 4.7 → 16.3）、两者叠加约 10.9 倍。四种配置的回答**逐字相同**。

⚠️ 这里的绝对 tok/s 是 qemu 逐条翻译指令跑出来的，**不能当手机性能**，能看的是
「NEON 有没有用、线程有没有用」这两个比值，以及回答是否一致。

而且 qemu 会把 `vdotq_s32` 拆成一串标量乘加来模拟，真机上这条指令有专门的
dotprod 硬件通路——所以 **3.8 倍是个下界式的参考**，不是真机预期值的上限。
确切数字只能在手机上跑 `--bench` 得到（`FENG_PHONE_NEON=0` 与 `=1` 各跑一次即为
NEON 的真实收益）。

### 3.4 工具链

| 工具链 | 验证情况 |
|---|---|
| x86_64 gcc-14（-O2） | 全部套件 + 基准，见上表 |
| x86_64 clang 18 | 同一份源码构建通过，32 题输出与 gcc 版**逐字节相同** |
| aarch64 gcc-14（交叉 + qemu） | NEON / 多线程正确性，见 `logs/phone_a64_*.txt` |
| Android NDK r27c clang 18（arm64-v8a） | `libfeng.so` 编译通过、JNI 符号齐全 |
| Termux 上的 clang | 未在本机实测（无设备），但上面两行覆盖了同一类工具链 |

### 3.5 为什么 NEON 内核走 int8 点积而不是 fp32 SIMD

ARM64 的 fp32 FMA 只有 4 个通道，而 4-bit 权重必须先「拆 nibble → 扩宽到 u32 → 转 float」
——这套转换本身就吃掉大部分收益（粗算每 64 个权重约 75 条 SIMD 指令，和标量 madd 链的
约 96 条几乎打平）。

改用 `vdotq_s32`（ARMv8.2 dotprod）：一条指令做 16 次 int8 乘加，每 64 个权重只剩
4 条 sdot + 4 条 `vqtbl1q_s8` 查表（nibble → int8 权重）+ 10 条 nibble 展开 ≈ 18 条。

代价是激活要先量化成 int8——**这不是新引入的近似**：引擎本来就有一条 `FENG_GEMV_A8`
标量路径用同一套公式（`scale = max|x|/127`，`acc += (float)dot * (sw * sx)`，同块顺序），
NEON 内核刻意保持同样的式子与顺序，因此两者**逐位一致**（`logs/phone_a64_ab_a8.txt` 的
`max|diff| = 0` 就是这条）。没有 dotprod 的老核（Cortex-A53/A72）运行期探测后自动回退标量。

A8 带来的数值变化量级：与纯 fp32 路径比 `max|diff| = 1.683`，而这份权重本身的 Q4 量化误差
是 **2.9443**（`pc_check` 口径，见 `logs/pc_check_v3_15ci4_inl.txt` 等）——**激活量化的误差
比权重量化还小**，32 题矩阵结论不变（27/27 + 4/4）。

---

## 4. 与板端的差异（改引擎前必读）

**2026-10-23 起没有「手机端引擎副本」了。** 三端（板端 / PC / 手机）编**同一份**
`core/*.c`，手机差异全部是 `-DFENG_PHONE=1` 的编译期分支。之前那份 18 个文件、
4909 行的手工拷贝（`feng_llm.c` 分叉 458 行）已经删除——同步靠 `cp` 是持续 bug 源。

差异只有下面四处，都在 `core/` 里：

| 位置 | 手机端差异 | 为什么 |
|---|---|---|
| `core/feng_smp.c` | `#elif defined(FENG_PHONE)`：pthread 线程池（`feng_phone_parallel_for`），`feng_par_run` 也接进池子 | 手机 4~8 核，板端那套 FreeRTOS 只覆盖 2 核 |
| `core/feng_quant.c` | `#if FENG_PHONE_NEON`：Q4 sdot / fp16 NEON 内核分派；`feng_gemv3` 走并行路径 | 原版非 ESP 分支直接调 `gemv_range_cpu`，会让 q/k/v 与 gate/up **绕开多线程**，提速少一大半 |
| `core/feng_llm.c` | `#if FENG_PHONE_NEON`：q2 注意力 K/V 的 NEON 内核；`FENG_ATTN_PARTS` 默认 4 | 长上下文里注意力占 prefill 的 76% |
| `core/feng.h` | `#ifdef FENG_PHONE` 包含 `feng_phone.h`，`FENG_MAX_SCORES` 跟随 `FENG_PHONE_MAX_CTX` | 板端 4096 的 scores 上限会钉死手机上下文 |

手机独有模块（板端没有）：`feng_phone.c/h`（mmap 加载、时间源、线程池、NEON 开关）、
`feng_neon.c/h`（aarch64 内核）。

**守卫**：`python phone/tools/check_single_source.py` 会在 `phone/engine/` 里
再次出现引擎副本时直接失败；CI 与 `tools/check_all.py` 都跑它。

**改完怎么验**：`./build.sh` 编出来，与「板端原始引擎」对照逐字节比：

```bash
./build_baseline.sh     # 用 core/ 不加 FENG_PHONE，产出对照组
./verify.sh             # 单线程 / 4 线程 / NEON 关，与对照组逐字节比 + 21/21 + mem12
```

## 5. 目录结构

```
phone/
├── engine/         引擎派生副本（C23）+ 手机端新增的平台层与 NEON 内核
├── cli/            Termux CLI、验收套件、构建与基准脚本
├── android/        Kotlin + Compose 工程（含 JNI 薄壳）
└── tools/          APK 一键构建、NDK 原生层编译自检
```

---

## 6. 验证与验收

```bash
# PC/Linux：手机引擎 vs 板端原始引擎逐字节比对 + 三个验收套件
./phone/cli/build_baseline.sh && ./phone/cli/build.sh && ./phone/cli/verify.sh

# 换编译器结果不变（Termux 用的是 clang，开发机默认 gcc）
./phone/cli/verify_clang.sh

# 线程扩展基准（数字写 logs/）
./phone/cli/bench.sh

# aarch64 + NEON 正确性（WSL / Linux，需要 qemu-user-static）
./phone/cli/build_wsl_aarch64.sh          # 加 --full 再跑一遍 32 题矩阵

# Android 原生层：用 NDK 单独编 libfeng.so（不建 APK）
./phone/tools/check_android_native.sh
```

判据：`verify.sh` 要求手机引擎在 NEON 关时与板端原始引擎**逐字节相同**（含多线程——
按输出行切分是位精确的），且三个套件同分。

---

## 7. 已知限制

1. **真机性能已测**（2026-10-18，P30 Pro / Kirin 980）：32 题矩阵 **27/27 + 4/4**、
   算术 **21/21**、短提示 decode 约 **190–330 tok/s**（受系统负载影响很大）。
   历史完整线程扫描见 `logs/phone_bench_p30pro.txt`。
2. **上下文**：APK 默认 **32768**（P30 Pro 实测 `ctx 0/32768 ｜ KV 154.00 MB`），
   分配失败自动回退 2048；CLI 默认 2048，`--ctx 32768` 可开。
   32k 的代价：q2 KV **154 MB**（2048 只要 9.62 MB）、短提示 decode 约 331 tok/s，
   但填到 ~2600 token 只剩 32–43 tok/s，真填满 32k 约 2–3 tok/s。
   更关键的是**prefill 是 O(n²)**。P30 Pro 上 4064 token 的预填实测：
   **优化前 107 秒；q2 注意力按 token/维度分给 4 核后 58 秒**（约 1.8–2.5×，
   热态波动大），8k 为 210 秒；按平方外推 32k 仍约 **1 小时**。
   所以 32k **能分配、能跑，但不是可交互的长文上下文**；要真正用 32k，
   下一步是批量/分块 GEMM（一次 prefill 多个 token，摊销权重读取），QAT 解决不了这个。
   A/B 开关：`-DFENG_Q2_PAR_K=0` / `-DFENG_Q2_PAR_V=0`（默认都开）。
   线程数保持默认 **4**：P30 Pro 上 8 线程实测 157 s，反而比 4 线程的 58 s 慢
   （大核调度/线程池争用）。
   质量上 4.0-embed 的训练/验收工作点是 2048，32k 也偏离训练分布；
   需要高质量 32k 请用 PC fp32 / HF 路径。
3. **要求 arm64 + ARMv8.2 dotprod**（Cortex-A55/A76 及以后的核）。没有 dotprod 的老设备
   能跑，但只能走标量路径，速度会明显下降。
4. **APK 只出 arm64-v8a**，32 位设备不打包。
5. **引擎单实例、非重入**：CLI 与 App 都只维护一个会话，并发调用会被锁串行化。
6. **APK 已在 P30 Pro 上构建并安装验证**（2026-10-18，JDK 21 + AGP 8.7 + NDK 27）。
   构建脚本仍要求 JDK 17~21（系统的 JDK 26 不行）；本轮实测 APK 31.06 MB，
   32k 初始化、中文生成、`59+1` tool 都正常。
7. **还没做的进一步优化**（按预计收益排序，都不是必需的）：
   - **注意力按 head 并行**：现在 `feng_llm.c` 的 head 循环是串行的（7 个头顺序跑），
     长上下文时它是剩下的主要串行部分。按 head 并行是**位精确**的（每个 head 内部的
     累加顺序不变），改动量中等。
   - **q2 注意力的 K/V 段做 NEON**：现在只做了 Q4 GEMV，q2 内层仍是标量 LUT；
     短上下文时收益有限，长上下文下值得做。
   - **fp16 激活的 sdot（A16/F16 混合）**：能省掉激活量化那点精度损失，但收益要看实测。
