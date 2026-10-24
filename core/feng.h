/* feng-30m on ESP32-S3: model container + inference core (portable C23).
 *
 * C23 约定（见 AGENTS.md，v4.0 起）：
 *  - 零堆分配：本头文件与核心模块不使用 malloc/calloc/free；
 *    唯一的例外是 tokenizer 的启动期 PSRAM 加载（一次性、固定大小）；
 *  - 不使用需要运行时构造的全局/静态对象（.init_array 必须为空）；
 *  - 标准库只用无堆组件：<stddef.h>/<stdint.h>/<stdbool.h>/<string.h>/<math.h>；
 *  - 错误处理靠返回码（int），无异常（C 本来就没有）。
 *
 * 本头文件同时被板端（纯 C23）和 PC 端（C++23 薄封装 + CUDA）包含，
 * 所以带 extern "C" 守卫。数组一律用「指针 + 长度」表达，不用 C++ 的 span。
 */
#ifndef FENG_H
#define FENG_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* KV cache mode: fp32 / int8 / q2, selected at compile time.
 *   FENG_KV_INT8=1 : int8 values + fp16 scale per (layer, position, head) -> 4x less PSRAM than fp32
 *   FENG_KV_Q2=1   : 2-bit values (4 packed per byte) + one fp16 scale per FENG_KV_Q2_BLOCK values.
 *                    block16 = 336 B/token/层（精度差）；block8 = 448 B/token/层（约省 int8 一半，
 *                    实测长文召回与续写都可用，见 scripts/kv_quant_experiment.py）。
 * 两种量化模式都用 kv->k_scale / v_scale 存 fp16 scale。 */
#ifndef FENG_KV_INT8
#define FENG_KV_INT8 0
#endif
#ifndef FENG_KV_Q2
#define FENG_KV_Q2 0
#endif
#ifndef FENG_KV_Q2_BLOCK
#define FENG_KV_Q2_BLOCK 16      /* 每多少个值共享一个 fp16 scale（须整除 head_dim、是 4 的倍数） */
#endif
#if FENG_KV_Q2 && ((FENG_KV_Q2_BLOCK) <= 0 || (FENG_KV_Q2_BLOCK) % 4 != 0)
#error "FENG_KV_Q2_BLOCK 必须是正的 4 的倍数"
#endif
/* fp16 KV：每值 2 字节、无 scale。只在手机/PC 上有意义——板端 16 MB PSRAM 下
 * 2048 上下文要 40.4 MB，放不下（q2 只要 9.62 MB）。 */
#ifndef FENG_KV_FP16
#define FENG_KV_FP16 0
#endif
#if (FENG_KV_INT8 + FENG_KV_Q2 + FENG_KV_FP16) > 1
#error "FENG_KV_INT8 / FENG_KV_Q2 / FENG_KV_FP16 只能选一个"
#endif
/* 手机端平台层：FENG_PHONE_MAX_CTX（上下文上限）、NEON 开关、线程池声明。
 * 板端 / PC 不定义 FENG_PHONE，这一整块不参与编译。 */
#ifdef FENG_PHONE
#include "feng_phone.h"
#endif

/* q2 注意力 K 打分的 NEON 内核默认**关闭**。
 * 原因：它的累加分组是 4 路并行，与标量链不同（~1e-7 相对差），实测在
 * P30 Pro 上把「翻译-再见」从 Goodbye 翻成 Good morning（32 题 27→26）。
 * V 段的 NEON 内核是逐位一致的（乘法分组与标量逐字相同），所以保留。
 * 想要速度就 -DFENG_PHONE_NEON_ATTN=1，但必须重跑 32 题与长文套件。 */
#ifndef FENG_PHONE_NEON_ATTN
#define FENG_PHONE_NEON_ATTN 0
#endif
/* q2 注意力内层用字节 LUT 取代移位/掩码/整数转浮点（数值与原式完全一致，结果逐位不变）。
 * 默认开；-DFENG_Q2_LUT=0 可切回原实现做 A/B。 */
#ifndef FENG_Q2_LUT
#define FENG_Q2_LUT 1
#endif
/* q2 注意力一次算 2 个上下文 token（两条独立累加链填 FPU 流水线、共享加载）。
 * 数值逐位不变；-DFENG_Q2_PAIR=0 可切回单 token 版做 A/B。 */
#ifndef FENG_Q2_PAIR
#define FENG_Q2_PAIR 1
#endif
/* 性能剖析（调试用，默认关）：累计 q2 注意力的 K / softmax / V 三段的 CPU 周期，
 * 由 FENG_BENCH_CTX 基准打印，用来定位长上下文的时间去向。 */
#ifndef FENG_ATTN_PROF
#define FENG_ATTN_PROF 0
#endif

/* 实验开关：激活 int8 量化的 GEMV（给 PIE 铺路；默认 0 = 原来的 LUT/fp32 路径） */
#ifndef FENG_GEMV_A8
#define FENG_GEMV_A8 0
#endif
/* Q4 GEMV 累加形式：1 = 纯 madd 链（2 个 FP 运算/2 权重，实测更快：ctx-256 -5%、
 * lm head -7.5%，见 CHANGELOG 附录）；0 = 两两求和再累加（3 个 FP 运算/2 权重）。
 * 两者 logits 差 ≤ 3.8e-6（Q4 量化误差是 2.94），32 题矩阵输出逐字相同；
 * 想要与旧版逐位一致时用 -DFENG_GEMV_MADD=0。 */
#ifndef FENG_GEMV_MADD
#define FENG_GEMV_MADD 1
#endif
/* 快速 exp（多项式 + 2^k 缩放，~20 条指令）替换 softmax / silu 里的 newlib expf
 * （实测 ~257 周期/次）。板端 softmax 从 169ms 降到 48ms（2048 ctx），
 * logits 差 ≤3.8e-6、32 题矩阵输出逐字相同；-DFENG_FAST_EXP=0 可切回 newlib。 */
/* 块稀疏注意力（推理期，无需重训）：把长上下文的 O(n²) 注意力压到 O(n·k)。
 * 做法是每块 64 token，用块内最后一个 token 当代表算粗分，永远保留前 SINK 块
 * （attention sink）与最近 RECENT 块，其余按粗分取 top-K 块，只在这些块上算
 * 完整分数/softmax/V。
 *
 * **默认关闭**：它会跳过一部分上下文，长文召回可能掉分。开启后必须重跑
 * 128 题长文套件与「文中没有」拒答套件，掉了就退回。
 * 8k 上下文时注意力占 prefill 的 76%，这是唯一能真正压掉它的手段
 * （批量 GEMM 已被实测证伪）。 */
#ifndef FENG_SPARSE_ATTN
#define FENG_SPARSE_ATTN 0
#endif
#if FENG_SPARSE_ATTN
#ifndef FENG_SPARSE_BLOCK
#define FENG_SPARSE_BLOCK 64        /* 块大小（token） */
#endif
#ifndef FENG_SPARSE_TOP_K
#define FENG_SPARSE_TOP_K 24        /* 除 sink/最近窗口外，最多再看多少块 */
#endif
#ifndef FENG_SPARSE_SINK
#define FENG_SPARSE_SINK 2          /* 开头固定保留的块数（attention sink） */
#endif
#ifndef FENG_SPARSE_RECENT
#define FENG_SPARSE_RECENT 2        /* 结尾固定保留的块数（最近上下文） */
#endif
/* 上限实验（只用于评估，不用于产品）：先用全量注意力算出真实分数，
 * 再按"块内最大真实分数"选 top-K 块，只保留这些块的贡献。
 * 它回答的是：**如果索引器完美，稀疏注意力还能保住多少分**。
 * 若 oracle 也掉分，说明这个预算下的稀疏化本身不成立，训练 indexer 也没用。 */
/* 索引跨层复用（对应 V4.1-Flash 的 Reuse 模式）：只在第 0 层选一次块，
 * 后面 10 层沿用同一份块掩码。作用是让"选块"的成本从 11 次摊成 1 次；
 * 代价是后面各层看不到自己被漏掉的块。这里只用来量准确率影响。 */
#ifndef FENG_SPARSE_REUSE
#define FENG_SPARSE_REUSE 0
#endif
#ifndef FENG_SPARSE_ORACLE
#define FENG_SPARSE_ORACLE 0
#endif
#ifndef FENG_SPARSE_MIN_TOKENS
#define FENG_SPARSE_MIN_TOKENS 1024 /* 短于这个长度不启用（省不了多少还多一次扫描） */
#endif
#endif
#ifndef FENG_FAST_EXP
#define FENG_FAST_EXP 1
#endif
/* q2 注意力的 V 段：1 = 把 (scores[t] * vscale[blk]) 先乘好，内层每值只剩 1 个 madd
 * （原来是 (st*val)*sv 的 3 个 FP 运算）。板端 V 段 912→832 ms（2048 ctx）、
 * 32 题输出逐字相同；三重数值优化累计 logits 差 4.8e-6（见 CHANGELOG 附录）。 */
#ifndef FENG_Q2_VFOLD
#define FENG_Q2_VFOLD 1
#endif


#define FENG_STR2(x) #x
#define FENG_STR(x) FENG_STR2(x)

#if defined(ESP_PLATFORM)
#include "esp_attr.h"
#define FENG_HOT IRAM_ATTR
#else
#define FENG_HOT
#endif

/* 编译期常量：用 enum 而不是 const 变量，保证零存储、且没有 C 的「文件作用域 const
 * 是外部链接」问题（C++ 里是内部链接，C 里不是——迁移时踩过这个区别）。 */
enum {
    FENG_MAGIC       = 0x46574E31,   /* "FWN1" */
    FENG_MAX_LAYERS  = 24,
    FENG_MAX_TENSORS = 168           /* 11 tensors/layer * layers + tok_embd + output_norm */
};

/* 固定工作区上限（全部有加载期校验；改模型规格时要同步改这里）。
 * FENG_MAX_SCORES 默认 4096：板端 2048、PC/手机 4096 足够；PC 要跑 32k
 * 可以在构建时 -DFENG_MAX_SCORES=32768。 */
#ifndef FENG_MAX_SCORES
#ifdef FENG_PHONE
#define FENG_MAX_SCORES FENG_PHONE_MAX_CTX   /* 手机：与上下文上限一致 */
#else
#define FENG_MAX_SCORES 4096
#endif
#endif
#ifndef FENG_MAX_HEAD_DIM
#define FENG_MAX_HEAD_DIM 128            /* q2 V 段 vacc[] 与 rope 表的上限 */
#endif
#ifndef FENG_MAX_HIDDEN
#define FENG_MAX_HIDDEN 512              /* g_tmp[] / 启动基准 xref[] 的上限 */
#endif

/* 权重存储格式。FP16 = 0 是**历史上零初始化的默认**，别改这个取值。 */
typedef enum { FENG_DT_FP16 = 0, FENG_DT_Q4 = 1 } feng_dtype_t;

/* 是否让模型文件自己声明每层矩阵的格式（fp16 权重）。
 *
 * **默认 0，板端永远保持 0**：关掉时下面所有新增字段/分支都会被预处理器去掉，
 * 板端编出来的代码与改造前逐 token 相同（固件体积必须仍是 257,104 B）。
 * PC 与手机端在构建时显式 `-DFENG_FP16_WEIGHTS=1`（手机副本在 feng.h 里直接定义）。
 *
 * 为什么需要它：以前 `feng_llm.c` 硬编码 `FENG_DT_Q4`，而导出器在
 * `shape[1] % 64 != 0` 时会写 fp16 矩阵——运行时不报错、logits 全错。 */
#ifndef FENG_FP16_WEIGHTS
#define FENG_FP16_WEIGHTS 0
#endif

#if FENG_FP16_WEIGHTS
/* 取某一层的矩阵格式。wq/wk/wv 与 gate/up 共用一次合并 GEMV，只能有一个 dtype；
 * 加载期会校验同层 7 个矩阵格式一致。 */
#define FENG_LAYER_DTYPE(L) ((L)->mat_dtype)
#else
/* 与改造前的源码**完全同 token**，板端二进制逐字节不变 */
#define FENG_LAYER_DTYPE(L) FENG_DT_Q4
#endif

typedef struct {
    uint32_t magic, version, n_tensors;
    uint32_t n_layers, hidden, n_heads, head_dim, ffn, vocab;
    float rms_eps, rope_theta;
} feng_header_t;

typedef struct {
    char name[32];
    uint32_t dtype;
    uint32_t shape[4];
    uint64_t offset, nbytes;
} feng_tensor_t;

typedef struct {
    const void *attn_norm, *ffn_norm;
    const void *wq, *wk, *wv, *wo;
    const void *q_norm, *k_norm;
    const void *gate, *up, *down;
#if FENG_FP16_WEIGHTS
    uint32_t mat_dtype;   /* FENG_DT_FP16 / FENG_DT_Q4，同层 7 个矩阵共用 */
#endif
} feng_layer_t;

typedef struct {
    feng_header_t hdr;
    feng_tensor_t dir_storage[FENG_MAX_TENSORS];   /* parsed, packed-format records */
    const feng_tensor_t *dir;
    const uint8_t *blob;            /* base pointer of the model file */
    const void *tok_embd;
    uint32_t tok_embd_dtype;
    const void *out_norm;
    feng_layer_t layers[FENG_MAX_LAYERS];
} feng_model_t;

/* parse a memory image of model.bin; returns 0 on success */
[[nodiscard]] int feng_model_init(feng_model_t *m, const void *data, size_t size);

/* Q4 / fp16 GEMV: y[n_out] = W[n_out x n_in] * x[n_in] (+ optional bias=NULL) */
void feng_gemv(const void *tensor, uint32_t dtype, const float *x, float *y,
               int n_out, int n_in, float *scratch);
float feng_f16_to_f32(uint16_t h);
uint16_t feng_f32_to_f16(float f);

/* same, but only output rows [r0, r1) -- used to split work across cores */
void feng_gemv_range(const void *tensor, uint32_t dtype, const float *x, float *y,
                     int r0, int r1, int n_in);

/* two-core version of feng_gemv (falls back to single core if init failed) */
void feng_gemv_par(const void *tensor, uint32_t dtype, const float *x, float *y,
                   int n_out, int n_in);

/* 通用两核并行：把 [0,nparts) 分给 worker；回调必须自己保证各 part 写不相交。
 * 板端用两个常驻 worker，PC/手机没有 worker 时顺序执行（结果不变）。
 * 目前只给 q2 注意力的按 head 切分用；不要在回调里再调用 feng_gemv_par
 * （会重入同一组 worker）。 */
typedef void (*feng_par_fn)(void *arg, int part, int nparts);
void feng_par_run(feng_par_fn fn, void *arg, int nparts);
/* 共享同一输入 x 的 2~3 个 GEMV 合并提交（q/k/v 或 gate/up 常成对出现）：
 *   y_i = W_i * x；w_i == NULL 表示跳过。CPU/板端 = 顺序调用 feng_gemv_par；
 * PC CUDA 后端 = 一次 H2D + 多次 kernel，减少每 token 的固定调度开销。 */
void feng_gemv3(const void *w0, const void *w1, const void *w2, uint32_t dtype,
                const float *x, float *y0, float *y1, float *y2,
                int n_out0, int n_out1, int n_out2, int n_in);
/* FENG_GEMV_A8=1 时：每个 GEMV 调用前准备一次 int8 激活（两核并发只读） */
void feng_gemv_a8_prepare(const float *x, int n_in);

/* Q4/PIE 的启动期查表初始化；在拉起 worker 线程之前调用一次，避免多个
 * worker 首次进入 GEMV 时并发写 LUT。 */
void feng_quant_init(void);

/* start the worker on core 1; call once before using feng_gemv_par */
void feng_smp_init(void);

/* transformer state (allocated by caller in PSRAM) */
typedef struct {
    void *k_cache;       /* fp32 / int8 / q2: [n_layers][ctx][n_heads*head_dim]（q2 时为 1/4 字节数） */
    void *v_cache;
    uint16_t *k_scale;   /* int8 / q2 modes: [n_layers][ctx][n_heads] fp16 */
    uint16_t *v_scale;
    int ctx, len;        /* filled length */
} feng_kv_t;

typedef struct {
    float *x, *xn, *q, *k, *v, *attn, *proj, *gate, *up, *ffn, *logits;
    float *scratch;
    int max_ctx;
} feng_workspace_t;

/* 总量（K+V 数据 + 两份 scale），只用于显示/预算。 */
size_t feng_kv_bytes(const feng_model_t *m, int ctx);
/* 单个 cache 的数据字节数（不含 scale）——分配 k_cache/v_cache 必须用这个。 */
size_t feng_kv_data_bytes(const feng_model_t *m, int ctx);
/* fp16 scale slots for ONE cache (K or V) in the current KV mode; 0 for fp32 */
size_t feng_kv_scale_slots(const feng_model_t *m, int ctx);
size_t feng_ws_bytes(const feng_model_t *m, int ctx);

/* run one token through the model, returns logits pointer (vocab floats) */
float *feng_forward(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos);
/* want_logits=0：跳过 out_norm + tied lm head（7.34M 权重、板端 ~125 ms/次）。
 * 只用于 prefill 的中间 token——KV 与 hidden 状态和 want_logits=1 完全相同，
 * 只是不算那份马上会被丢掉的 logits。 */
float *feng_forward_ex(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos,
                       int want_logits);

/* greedy-sample helpers */
int feng_argmax(const float *logits, int n);
/* 共享贪心采样：对 hist 里的 token 施加重复惩罚；no_repeat_n>=2 时禁止补全已出现过的
 * n-gram（压复读循环）。板端与 PC 引擎同口径。
 * logits 长度由 n_logits 给出、hist 长度由 n_hist 给出（C 版不再用 span）。 */
[[nodiscard]] int feng_sample_greedy(float *logits, int n_logits,
                                     const int *hist, int n_hist,
                                     float penalty, int no_repeat_n);

#ifdef __cplusplus
}
#endif

#endif /* FENG_H */
