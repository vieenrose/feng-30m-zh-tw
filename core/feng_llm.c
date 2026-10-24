/* Transformer forward pass for the feng-30m (Qwen3 architecture, MHA, tied head). */
#include "feng.h"

#ifdef FENG_PHONE
#include "feng_phone.h"   /* FENG_PHONE_MAX_CTX / 线程池 / NEON 开关 */
#if FENG_PHONE_NEON
#include "feng_neon.h"    /* q2 注意力的 NEON 内核（K 打分 / V 累加） */
#endif
#endif

#if defined(FENG_USE_CUDA)
#include "feng_cuda.h"
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>

#if FENG_ATTN_PROF && defined(ESP_PLATFORM)
#include "esp_cpu.h"
unsigned long long g_prof_cycles[3];    /* K / softmax / V，单位 CPU 周期 */
#endif

/* fp16 -> f32 for norm weights etc. */
/* 与 feng_quant.c::f16_to_f32 完全同算法的本地内联版：注意力内层每个 block
 * 都要转一次 scale，走外部函数（~25 条指令/次、还有调用开销）在长上下文里
 * 被放大成主要开销；内联后只剩位运算。数值逐位一致。 */
/* 手机端 NEON 分支不走这个内联（kernel 内部自己转 scale），标量后端才用；
 * 加 unused 属性避免 aarch64 -Wall 报未使用。 */
#if defined(__GNUC__)
__attribute__((unused))
#endif
static inline float f16_to_f32_local(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; exp--; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static void f16_to_f32_vec(const uint16_t *src, float *dst, int n)
{
    for (int i = 0; i < n; i++) {
        uint32_t h = src[i];
        uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
        if (exp == 0) {
            if (man == 0) {
                f = sign << 31;
            } else {
                exp = 127 - 15 + 1;
                while ((man & 0x400u) == 0) { man <<= 1; exp--; }
                man &= 0x3FFu;
                f = (sign << 31) | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) {
            f = (sign << 31) | 0x7F800000u | (man << 13);
        } else {
            f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
        }
        memcpy(&dst[i], &f, 4);
    }
}

/* q2 注意力按 head 切给两个核。每个 part 需要独立的 score 缓冲；
 * 板端 2048 ctx = 2×16KB，PC 32k = 2×128KB，都在 .bss 里可接受。
 * 设 1 可退回单核（A/B 与调试）。 */
#ifndef FENG_ATTN_PARTS
#ifdef FENG_PHONE
#define FENG_ATTN_PARTS 4            /* 手机 4~8 核：按 head 切给线程池 */
#else
#define FENG_ATTN_PARTS 2            /* 板端 S3 双核 */
#endif
#endif
#if FENG_ATTN_PARTS < 1 || FENG_ATTN_PARTS > 8 || \
    (FENG_ATTN_PARTS > 2 && !defined(FENG_PHONE))
#error "FENG_ATTN_PARTS: 板端 1..2，手机 1..8"
#endif

/* scratch buffers live in .bss, not on the task stack (the S3's main stack is small).
 * 大小不是“随便写死”：feng_model_init 会校验 head_dim/hidden 上限，
 * feng_forward_ex 会校验 pos < kv->ctx 且 pos < FENG_MAX_SCORES。 */
static float g_scores[FENG_ATTN_PARTS][FENG_MAX_SCORES];  /* one score per (query, past key) */
static float g_tmp[FENG_MAX_HIDDEN];
#if FENG_KV_Q2 && FENG_Q2_LUT
/* 字节 LUT：b -> { (b>>0&3)-1.5, (b>>2&3)-1.5, (b>>4&3)-1.5, (b>>6&3)-1.5 }
 * 表达式与原内层完全相同，只把移位/掩码/整数转浮点换成一次 L1 查表。 */
static float g_q2_lut[256][4];
static int g_q2_lut_ready;
static void q2_lut_init(void)
{
    if (g_q2_lut_ready) return;
    for (int b = 0; b < 256; b++) {
        for (int k = 0; k < 4; k++) {
            g_q2_lut[b][k] = (float)((b >> (2 * k)) & 3) - 1.5f;
        }
    }
    g_q2_lut_ready = 1;
}
#endif
/* RoPE inverse frequencies, computed once per (head_dim, theta).
 * 旧实现只用一个 bool：同进程换模型（不同 head_dim/theta）会沿用旧频率，
 * 新 head_dim 多出来的项也永远是 0。缓存必须带参数键。 */
static float g_rope_inv[128];
static int g_rope_ready;
static int g_rope_key_dim;
static float g_rope_key_theta;

static void rmsnorm(float *out, const float *x, const void *w_fp16, int n, float eps)
{
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    const uint16_t *w = (const uint16_t *)w_fp16;
    for (int i = 0; i < n; i++) {
        uint32_t h = w[i];
        uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
        if (exp == 0) {
            if (man == 0) f = sign << 31;
            else {
                exp = 127 - 15 + 1;
                while ((man & 0x400u) == 0) { man <<= 1; exp--; }
                man &= 0x3FFu;
                f = (sign << 31) | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
        else f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
        float wf; memcpy(&wf, &f, 4);
        out[i] = x[i] * inv * wf;
    }
}

static void rope(float *vec, int n_heads, int head_dim, int pos, float theta)
{
    const int half = head_dim / 2;
    if (!g_rope_ready || g_rope_key_dim != head_dim || g_rope_key_theta != theta) {
        const int cap = (int)(sizeof(g_rope_inv) / sizeof(g_rope_inv[0]));
        for (int i = 0; i < half && i < cap; i++) {
            g_rope_inv[i] = powf(theta, -2.0f * (float)i / (float)head_dim);
        }
        g_rope_key_dim = head_dim;
        g_rope_key_theta = theta;
        g_rope_ready = 1;
    }
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        for (int i = 0; i < half; i++) {
            const float ang = (float)pos * g_rope_inv[i];
            const float c = cosf(ang), s = sinf(ang);
            const float x1 = v[i], x2 = v[i + half];
            v[i] = x1 * c - x2 * s;
            v[i + half] = x2 * c + x1 * s;
        }
    }
}

/* 快速 exp：exp(x) = 2^k · exp(r)，|r| ≤ ln2/2，6 阶 Horner 多项式。
 * 用于 softmax 与 SiLU（newlib expf 实测 ~257 周期/次，这里 ~20 条指令）。
 * 相对误差：|x|≤4 时 <4e-7，|x|≤16 时 ~1.2e-6，接近下溢边界 ~8e-6
 * （该处结果本身已可忽略）；x 超出 [-87.34, 88.72] 直接饱和到 0 / inf。 */
#if FENG_FAST_EXP
static inline float fast_expf(float x)
{
    if (x > 88.722839f) return INFINITY;   /* math.h；g++ / Xtensa / MSVC 通用 */
    if (x < -87.336548f) return 0.f;
    const float log2e = 1.4426950408889634f;
    const float ln2 = 0.6931471805599453f;
    const float kf = x * log2e;
    const int ki = (int)(kf + (kf >= 0.f ? 0.5f : -0.5f));
    const float r = x - (float)ki * ln2;
    const float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r *
                    (0.041666668f + r * (0.008333334f + r * 0.0013888889f)))));
    union { float f; uint32_t u; } s;
    s.u = (uint32_t)(ki + 127) << 23;
    return p * s.f;
}
#endif

static void silu_mul(float *out, const float *g, const float *u, int n)
{
    for (int i = 0; i < n; i++) {
        const float x = g[i];
#if FENG_FAST_EXP
        out[i] = (x / (1.0f + fast_expf(-x))) * u[i];
#else
        out[i] = (x / (1.0f + expf(-x))) * u[i];
#endif
    }
}

/* 单个 cache（K 或 V）的数据字节数，**不含 scale**。
 * 前端必须按它分配 k_cache/v_cache：旧代码拿 feng_kv_bytes()（data+scale）
 * 的一半当数据缓冲，又另外分配 scale，手机 q2/block8 多占 50%（154→231 MiB）。 */
size_t feng_kv_data_bytes(const feng_model_t *m, int ctx)
{
#if FENG_KV_INT8
    return (size_t)m->hdr.n_layers * ctx * m->hdr.hidden;
#elif FENG_KV_Q2
    return (size_t)m->hdr.n_layers * ctx * (m->hdr.hidden / 4);
#elif FENG_KV_FP16
    return (size_t)m->hdr.n_layers * ctx * m->hdr.hidden * sizeof(uint16_t);
#else
    return (size_t)m->hdr.n_layers * ctx * m->hdr.hidden * sizeof(float);
#endif
}

size_t feng_kv_bytes(const feng_model_t *m, int ctx)
{
    /* K + V 两份数据，加两份 scale（fp32/fp16 的 scale 为 0）。 */
    return feng_kv_data_bytes(m, ctx) * 2 +
           feng_kv_scale_slots(m, ctx) * sizeof(uint16_t) * 2;
}

size_t feng_kv_scale_slots(const feng_model_t *m, int ctx)
{
    /* 单个 cache（K 或 V）需要的 fp16 scale 数量 */
#if FENG_KV_INT8
    return (size_t)m->hdr.n_layers * ctx * m->hdr.n_heads;
#elif FENG_KV_Q2
    return (size_t)m->hdr.n_layers * ctx * (m->hdr.hidden / FENG_KV_Q2_BLOCK);
#elif FENG_KV_FP16
    (void)m; (void)ctx;
    return 0;                       /* fp16 不需要 scale */
#else
    (void)m; (void)ctx;
    return 0;
#endif
}

size_t feng_ws_bytes(const feng_model_t *m, int ctx)
{
    const int h = m->hdr.hidden, f = m->hdr.ffn, v = m->hdr.vocab;
    /* x/xn/q/k/v/attn/proj = 7h；gate/up/ffn = 3f；logits = v；
     * scratch 至少 max(ctx, f)。 */
    const int scratch = ctx > f ? ctx : f;
    return (size_t)(h * 7 + f * 3 + v + scratch) * sizeof(float);
}

#if FENG_KV_Q2
/* q2 注意力的按 head 切分：每个 head 的 K 打分、softmax、V 累加完全独立，
 * 两个核各算一部分 head，写 attn[hh*hd..] 不相交，结果与单核逐位一致。
 * scores 每个 part 一份（g_scores[part]），不能共用。 */
typedef struct {
    const feng_kv_t *kv;
    const float *q;
    float *attn;
    int l, pos, nh, hd, qb, nb, bw;
    float scale;
    float *scores;
} q2_attn_ctx_t;

#if FENG_SPARSE_ATTN
/* 每个 part 一份块分数缓冲（q2_attn_range 按 head 并行，不能共用）。 */
static float g_blk_score[FENG_ATTN_PARTS][FENG_MAX_SCORES / FENG_SPARSE_BLOCK + 2];

#if !FENG_SPARSE_ORACLE
/* 单个 head 的块稀疏注意力，结果写进 c->attn[hh*hd ..]。
 * 与全量路径共用同一套 q2 解量化与 (st*sv)*lu 的 V 分组，所以被选中的位置
 * 数值逐位相同；差别只在"少看了一些块"。 */
static void q2_sparse_head(const q2_attn_ctx_t *c, int hh, int part)
{
    const int total = c->pos + 1;
    const int nblk = (total + FENG_SPARSE_BLOCK - 1) / FENG_SPARSE_BLOCK;
    float *scores = c->scores + (size_t)part * FENG_MAX_SCORES;
    float *bs = g_blk_score[part];
    const float *qh = c->q + hh * c->hd;
    const uint8_t *krow = (const uint8_t *)c->kv->k_cache
                          + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->qb;
    const uint16_t *ksrow = c->kv->k_scale + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->nb;

    /* 1) 块代表粗分：块内 4 个采样点（1/4、1/2、3/4、末尾），取最大值。
     * 只取末尾一个 token 太弱——needle 是一句话，往往不在块尾，实测 8k 召回
     * 从 3/4 掉到 1/4（2026-10-23）。4 个采样点仍是全量的 6.25%。 */
    for (int b = 0; b < nblk; b++) {
        float best = -1e30f;
        for (int s4 = 1; s4 <= 4; s4++) {
            int t = b * FENG_SPARSE_BLOCK + (FENG_SPARSE_BLOCK * s4) / 4 - 1;
            if (t > c->pos) t = c->pos;
            const uint8_t *kh = krow + (size_t)t * c->qb;
            const uint16_t *ks = ksrow + (size_t)t * c->nb;
            float s = 0.f;
            for (int blk = 0; blk < c->nb; blk++) {
                const float sk = f16_to_f32_local(ks[blk]);
                const uint8_t *bb = kh + blk * c->bw;
                int d = blk * FENG_KV_Q2_BLOCK;
                for (int j = 0; j < c->bw; j++) {
                    const float *lu = g_q2_lut[bb[j]];
                    for (int k4 = 0; k4 < 4; k4++) s += qh[d++] * lu[k4] * sk;
                }
            }
            if (s > best) best = s;
        }
        bs[b] = best;
    }

    /* 2) 选块：sink + top-K（插入排序，K 很小）+ 最近窗口 */
    float topv[FENG_SPARSE_TOP_K];
    int topb[FENG_SPARSE_TOP_K];
    int ntop = 0;
    for (int b = FENG_SPARSE_SINK; b < nblk - FENG_SPARSE_RECENT; b++) {
        const float v = bs[b];
        if (ntop < FENG_SPARSE_TOP_K) {
            int i = ntop++;
            while (i > 0 && topv[i - 1] < v) { topv[i] = topv[i - 1]; topb[i] = topb[i - 1]; i--; }
            topv[i] = v;
            topb[i] = b;
        } else if (v > topv[FENG_SPARSE_TOP_K - 1]) {
            int i = FENG_SPARSE_TOP_K - 1;
            while (i > 0 && topv[i - 1] < v) { topv[i] = topv[i - 1]; topb[i] = topb[i - 1]; i--; }
            topv[i] = v;
            topb[i] = b;
        }
    }
    int keep[FENG_SPARSE_TOP_K + FENG_SPARSE_SINK + FENG_SPARSE_RECENT];
    int nk = 0;
#if FENG_SPARSE_REUSE
    if (c->l > 0 && hh < SP_MAX_HEADS) {
        /* Reuse：沿用第 0 层的块掩码 */
        for (int b = 0; b < nblk; b++)
            if (g_keep_bits[hh][b / 64] & (1ull << (b % 64))) keep[nk++] = b;
    } else {
#endif
    for (int b = 0; b < FENG_SPARSE_SINK && b < nblk; b++) keep[nk++] = b;
    for (int i = 0; i < ntop; i++) keep[nk++] = topb[i];
    for (int b = nblk - FENG_SPARSE_RECENT; b < nblk; b++)
        if (b >= FENG_SPARSE_SINK) keep[nk++] = b;
#if FENG_SPARSE_REUSE
        if (c->l == 0 && hh < SP_MAX_HEADS) {
            for (int w = 0; w < FENG_MAX_SCORES / FENG_SPARSE_BLOCK / 64 + 1; w++)
                g_keep_bits[hh][w] = 0;
            for (int i = 0; i < nk; i++)
                g_keep_bits[hh][keep[i] / 64] |= 1ull << (keep[i] % 64);
        }
    }
#endif

    /* 3) 只对选中的块算完整分数 */
    float maxs = -1e30f;
    for (int i = 0; i < nk; i++) {
        int t0 = keep[i] * FENG_SPARSE_BLOCK;
        int t1 = t0 + FENG_SPARSE_BLOCK;
        if (t1 > total) t1 = total;
        for (int t = t0; t < t1; t++) {
            const uint8_t *kh = krow + (size_t)t * c->qb;
            const uint16_t *ks = ksrow + (size_t)t * c->nb;
            float s = 0.f;
            for (int blk = 0; blk < c->nb; blk++) {
                const float sk = f16_to_f32_local(ks[blk]);
                const uint8_t *bb = kh + blk * c->bw;
                int d = blk * FENG_KV_Q2_BLOCK;
                for (int j = 0; j < c->bw; j++) {
                    const float *lu = g_q2_lut[bb[j]];
                    for (int k4 = 0; k4 < 4; k4++) s += qh[d++] * lu[k4] * sk;
                }
            }
            s *= c->scale;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
    }

    /* 4) softmax（只对选中的位置归一化） */
    float sum = 0.f;
    for (int i = 0; i < nk; i++) {
        int t0 = keep[i] * FENG_SPARSE_BLOCK;
        int t1 = t0 + FENG_SPARSE_BLOCK;
        if (t1 > total) t1 = total;
        for (int t = t0; t < t1; t++) {
#if FENG_FAST_EXP
            scores[t] = fast_expf(scores[t] - maxs);
#else
            scores[t] = expf(scores[t] - maxs);
#endif
            sum += scores[t];
        }
    }
    const float inv = 1.f / sum;

    /* 5) V 累加：分组 (st*sv)*lu，与标量路径逐字相同 */
    float vacc[FENG_MAX_HEAD_DIM];
    for (int d = 0; d < c->hd; d++) vacc[d] = 0.f;
    const uint8_t *vrow = (const uint8_t *)c->kv->v_cache
                          + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->qb;
    const uint16_t *vsrow = c->kv->v_scale + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->nb;
    for (int i = 0; i < nk; i++) {
        int t0 = keep[i] * FENG_SPARSE_BLOCK;
        int t1 = t0 + FENG_SPARSE_BLOCK;
        if (t1 > total) t1 = total;
        for (int t = t0; t < t1; t++) {
            const uint8_t *vb = vrow + (size_t)t * c->qb;
            const uint16_t *vs = vsrow + (size_t)t * c->nb;
            const float st = scores[t];
            for (int blk = 0; blk < c->nb; blk++) {
                const float sv = f16_to_f32_local(vs[blk]);
                const float stv = st * sv;
                const uint8_t *bb = vb + blk * c->bw;
                int d = blk * FENG_KV_Q2_BLOCK;
                for (int j = 0; j < c->bw; j++) {
                    const float *lu = g_q2_lut[bb[j]];
                    for (int k4 = 0; k4 < 4; k4++) vacc[d++] += stv * lu[k4];
                }
            }
        }
    }
    for (int d = 0; d < c->hd; d++) c->attn[hh * c->hd + d] = vacc[d] * inv;
}
#endif  /* !FENG_SPARSE_ORACLE */

#if FENG_SPARSE_ORACLE
/* 上限实验：全量打分 -> 按块内最大真实分数选块 -> 只在选中块上 softmax/V。
 * 复杂度仍是 O(n²)（打分没省），只用来测"完美索引器"的天花板。 */
#if FENG_SPARSE_REUSE
/* 跨层复用的块掩码：每 head 一份位图（块数 <= FENG_MAX_SCORES/64）。 */
#define SP_MAX_HEADS 16
static uint64_t g_keep_bits[SP_MAX_HEADS][FENG_MAX_SCORES / FENG_SPARSE_BLOCK / 64 + 1];
#endif
static void q2_oracle_head(const q2_attn_ctx_t *c, int hh, int part)
{
    const int total = c->pos + 1;
    const int nblk = (total + FENG_SPARSE_BLOCK - 1) / FENG_SPARSE_BLOCK;
    float *scores = c->scores + (size_t)part * FENG_MAX_SCORES;
    float *bs = g_blk_score[part];
    const float *qh = c->q + hh * c->hd;
    const uint8_t *krow = (const uint8_t *)c->kv->k_cache
                          + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->qb;
    const uint16_t *ksrow = c->kv->k_scale + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->nb;

    for (int b = 0; b < nblk; b++) bs[b] = -1e30f;
    float maxs = -1e30f;
    for (int t = 0; t < total; t++) {
        const uint8_t *kh = krow + (size_t)t * c->qb;
        const uint16_t *ks = ksrow + (size_t)t * c->nb;
        float s = 0.f;
        for (int blk = 0; blk < c->nb; blk++) {
            const float sk = f16_to_f32_local(ks[blk]);
            const uint8_t *bb = kh + blk * c->bw;
            int d = blk * FENG_KV_Q2_BLOCK;
            for (int j = 0; j < c->bw; j++) {
                const float *lu = g_q2_lut[bb[j]];
                for (int k4 = 0; k4 < 4; k4++) s += qh[d++] * lu[k4] * sk;
            }
        }
        s *= c->scale;
        scores[t] = s;
        const int b = t / FENG_SPARSE_BLOCK;
        if (s > bs[b]) bs[b] = s;
        if (s > maxs) maxs = s;
    }

    float topv[FENG_SPARSE_TOP_K];
    int topb[FENG_SPARSE_TOP_K];
    int ntop = 0;
    for (int b = FENG_SPARSE_SINK; b < nblk - FENG_SPARSE_RECENT; b++) {
        const float v = bs[b];
        if (ntop < FENG_SPARSE_TOP_K) {
            int i = ntop++;
            while (i > 0 && topv[i - 1] < v) { topv[i] = topv[i - 1]; topb[i] = topb[i - 1]; i--; }
            topv[i] = v;
            topb[i] = b;
        } else if (v > topv[FENG_SPARSE_TOP_K - 1]) {
            int i = FENG_SPARSE_TOP_K - 1;
            while (i > 0 && topv[i - 1] < v) { topv[i] = topv[i - 1]; topb[i] = topb[i - 1]; i--; }
            topv[i] = v;
            topb[i] = b;
        }
    }
    int keep[FENG_SPARSE_TOP_K + FENG_SPARSE_SINK + FENG_SPARSE_RECENT];
    int nk = 0;
#if FENG_SPARSE_REUSE
    if (c->l > 0 && hh < SP_MAX_HEADS) {
        /* Reuse：沿用第 0 层选出的块掩码（对应 V4.1-Flash 的 Reuse 模式） */
        for (int b = 0; b < nblk; b++)
            if (g_keep_bits[hh][b / 64] & (1ull << (b % 64))) keep[nk++] = b;
    } else {
#endif
    for (int b = 0; b < FENG_SPARSE_SINK && b < nblk; b++) keep[nk++] = b;
    for (int i = 0; i < ntop; i++) keep[nk++] = topb[i];
    for (int b = nblk - FENG_SPARSE_RECENT; b < nblk; b++)
        if (b >= FENG_SPARSE_SINK) keep[nk++] = b;
#if FENG_SPARSE_REUSE
        if (c->l == 0 && hh < SP_MAX_HEADS) {
            for (int w = 0; w < FENG_MAX_SCORES / FENG_SPARSE_BLOCK / 64 + 1; w++)
                g_keep_bits[hh][w] = 0;
            for (int i = 0; i < nk; i++)
                g_keep_bits[hh][keep[i] / 64] |= 1ull << (keep[i] % 64);
        }
    }
#endif

    /* 只保留选中块的 exp，其余置 0 */
    float sum = 0.f;
    for (int b = 0; b < nblk; b++) {
        int t0 = b * FENG_SPARSE_BLOCK;
        int t1 = t0 + FENG_SPARSE_BLOCK;
        if (t1 > total) t1 = total;
        int sel = 0;
        for (int i = 0; i < nk; i++) if (keep[i] == b) { sel = 1; break; }
        for (int t = t0; t < t1; t++) {
            if (!sel) { scores[t] = 0.f; continue; }
#if FENG_FAST_EXP
            scores[t] = fast_expf(scores[t] - maxs);
#else
            scores[t] = expf(scores[t] - maxs);
#endif
            sum += scores[t];
        }
    }
    const float inv = 1.f / sum;

    float vacc[FENG_MAX_HEAD_DIM];
    for (int d = 0; d < c->hd; d++) vacc[d] = 0.f;
    const uint8_t *vrow = (const uint8_t *)c->kv->v_cache
                          + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->qb;
    const uint16_t *vsrow = c->kv->v_scale + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->nb;
    for (int i = 0; i < nk; i++) {
        int t0 = keep[i] * FENG_SPARSE_BLOCK;
        int t1 = t0 + FENG_SPARSE_BLOCK;
        if (t1 > total) t1 = total;
        for (int t = t0; t < t1; t++) {
            const uint8_t *vb = vrow + (size_t)t * c->qb;
            const uint16_t *vs = vsrow + (size_t)t * c->nb;
            const float st = scores[t];
            for (int blk = 0; blk < c->nb; blk++) {
                const float sv = f16_to_f32_local(vs[blk]);
                const float stv = st * sv;
                const uint8_t *bb = vb + blk * c->bw;
                int d = blk * FENG_KV_Q2_BLOCK;
                for (int j = 0; j < c->bw; j++) {
                    const float *lu = g_q2_lut[bb[j]];
                    for (int k4 = 0; k4 < 4; k4++) vacc[d++] += stv * lu[k4];
                }
            }
        }
    }
    for (int d = 0; d < c->hd; d++) c->attn[hh * c->hd + d] = vacc[d] * inv;
}
#endif  /* FENG_SPARSE_ORACLE */

#endif  /* FENG_SPARSE_ATTN */

static void q2_attn_range(void *arg, int part, int nparts)
{
    const q2_attn_ctx_t *c = (const q2_attn_ctx_t *)arg;
    const int hh0 = part * c->nh / nparts;
    const int hh1 = (part + 1) * c->nh / nparts;
    for (int hh = hh0; hh < hh1; hh++) {
        const float *qh = c->q + hh * c->hd;
        float *scores = c->scores + (size_t)part * FENG_MAX_SCORES;
        float maxs = -1e30f;
        const uint8_t *krow = (const uint8_t *)c->kv->k_cache
                              + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->qb;
        const uint16_t *ksrow = c->kv->k_scale + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->nb;
#if FENG_SPARSE_ATTN
        if (c->pos + 1 >= FENG_SPARSE_MIN_TOKENS) {
#if FENG_SPARSE_ORACLE
            q2_oracle_head(c, hh, part);
#else
            q2_sparse_head(c, hh, part);
#endif
            continue;
        }
#endif
#if FENG_ATTN_PROF
        const unsigned pc0 = (unsigned)esp_cpu_get_cycle_count();
#endif
#if FENG_PHONE_NEON && FENG_PHONE_NEON_ATTN && FENG_Q2_LUT && \
    (FENG_KV_Q2_BLOCK == 8)
        /* 手机端（需显式 -DFENG_PHONE_NEON_ATTN=1）：一条 NEON 内核算完整个 t 区间（每 (t,blk) 两条 4 路 FMA 链 +
         * 水平归约）。累加分组与标量链不同，结果在 fp32 噪声级（~1e-7 相对）；
         * 并行来自上层按 head 切分，所以这里不再嵌套线程。 */
        feng_neon_q2_attn_k(krow, ksrow, qh, scores, 0, c->pos + 1,
                            c->qb, c->nb, c->scale,
                            (const float (*)[4])g_q2_lut);
        for (int t = 0; t <= c->pos; t++)
            if (scores[t] > maxs) maxs = scores[t];
#else
#if FENG_Q2_LUT && FENG_Q2_PAIR
        {
            int t = 0;
            for (; t + 1 <= c->pos; t += 2) {
                const uint8_t *kh0 = krow + (size_t)t * c->qb;
                const uint8_t *kh1 = kh0 + c->qb;
                const uint16_t *ks0 = ksrow + (size_t)t * c->nb;
                const uint16_t *ks1 = ks0 + c->nb;
                float s0 = 0.f, s1 = 0.f;
                for (int blk = 0; blk < c->nb; blk++) {
                    const float sk0 = f16_to_f32_local(ks0[blk]);
                    const float sk1 = f16_to_f32_local(ks1[blk]);
                    const uint8_t *b0 = kh0 + blk * c->bw, *b1 = kh1 + blk * c->bw;
                    int d = blk * FENG_KV_Q2_BLOCK;
                    for (int j = 0; j < c->bw; j++) {
                        const float *l0 = g_q2_lut[b0[j]];
                        const float *l1 = g_q2_lut[b1[j]];
                        for (int k4 = 0; k4 < 4; k4++) {
                            const float qd = qh[d++];
                            s0 += qd * l0[k4] * sk0;
                            s1 += qd * l1[k4] * sk1;
                        }
                    }
                }
                s0 *= c->scale;
                s1 *= c->scale;
                scores[t] = s0;
                scores[t + 1] = s1;
                if (s0 > maxs) maxs = s0;
                if (s1 > maxs) maxs = s1;
            }
            if (t <= c->pos) {
                const uint8_t *kh = krow + (size_t)t * c->qb;
                const uint16_t *ks = ksrow + (size_t)t * c->nb;
                float s = 0.f;
                for (int blk = 0; blk < c->nb; blk++) {
                    const float sk = f16_to_f32_local(ks[blk]);
                    const uint8_t *bb = kh + blk * c->bw;
                    int d = blk * FENG_KV_Q2_BLOCK;
                    for (int j = 0; j < c->bw; j++) {
                        const float *lu = g_q2_lut[bb[j]];
                        for (int k4 = 0; k4 < 4; k4++) s += qh[d++] * lu[k4] * sk;
                    }
                }
                s *= c->scale;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
        }
#else
        for (int t = 0; t <= c->pos; t++) {
            const uint8_t *kh = krow + (size_t)t * c->qb;
            const uint16_t *ks = ksrow + (size_t)t * c->nb;
            float s = 0.f;
#if FENG_Q2_LUT
            for (int blk = 0; blk < c->nb; blk++) {
                const float sk = f16_to_f32_local(ks[blk]);
                const uint8_t *bb = kh + blk * c->bw;
                int d = blk * FENG_KV_Q2_BLOCK;
                for (int j = 0; j < c->bw; j++) {
                    const float *lu = g_q2_lut[bb[j]];
                    for (int k4 = 0; k4 < 4; k4++) s += qh[d++] * lu[k4] * sk;
                }
            }
#else
            for (int blk = 0; blk < c->nb; blk++) {
                const float sk = f16_to_f32_local(ks[blk]);
                for (int j = 0; j < c->bw; j++) {
                    const uint8_t b = kh[blk * c->bw + j];
                    for (int k4 = 0; k4 < 4; k4++) {
                        const int qq = (b >> (2 * k4)) & 3;
                        s += qh[blk * FENG_KV_Q2_BLOCK + j * 4 + k4]
                             * ((float)qq - 1.5f) * sk;
                    }
                }
            }
#endif
            s *= c->scale;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
#endif
#endif  /* FENG_PHONE_NEON && FENG_Q2_LUT */
#if FENG_ATTN_PROF
        {
            const unsigned pc1 = (unsigned)esp_cpu_get_cycle_count();
            g_prof_cycles[0] += (unsigned long long)(pc1 - pc0);
        }
#endif
        float sum = 0.f;
#if FENG_ATTN_PROF
        const unsigned ps0 = (unsigned)esp_cpu_get_cycle_count();
#endif
        for (int t = 0; t <= c->pos; t++) {
#if FENG_FAST_EXP
            scores[t] = fast_expf(scores[t] - maxs);
#else
            scores[t] = expf(scores[t] - maxs);
#endif
            sum += scores[t];
        }
        const float inv = 1.0f / sum;
#if FENG_ATTN_PROF
        g_prof_cycles[1] += (unsigned long long)((unsigned)esp_cpu_get_cycle_count() - ps0);
#endif
        {
            float vacc[FENG_MAX_HEAD_DIM];
            for (int d = 0; d < c->hd; d++) vacc[d] = 0.f;
#if FENG_ATTN_PROF
            const unsigned pv0 = (unsigned)esp_cpu_get_cycle_count();
#endif
            const uint8_t *vrow = (const uint8_t *)c->kv->v_cache
                                  + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->qb;
            const uint16_t *vsrow = c->kv->v_scale + ((size_t)c->l * c->nh + hh) * c->kv->ctx * c->nb;
#if FENG_PHONE_NEON && FENG_Q2_LUT && (FENG_KV_Q2_BLOCK == 8)
            /* 手机端：8 维一块，乘法分组 (st*sv)*lu 与标量版逐字相同、每个 d 按 t
             * 递增 —— 与标量路径逐位一致。内核硬编码 block8：其它 block 必须
             * 回退标量，否则 block16 只算一半维度、block4 会写出 head_dim。 */
            for (int blk = 0; blk < c->nb; blk++)
                feng_neon_q2_attn_v(vrow, vsrow, scores, vacc, c->pos + 1,
                                    c->qb, c->nb, blk,
                                    (const float (*)[4])g_q2_lut);
#else
#if FENG_Q2_LUT && FENG_Q2_PAIR
            {
                int t = 0;
                for (; t + 1 <= c->pos; t += 2) {
                    const uint8_t *vb0 = vrow + (size_t)t * c->qb;
                    const uint8_t *vb1 = vb0 + c->qb;
                    const uint16_t *vs0 = vsrow + (size_t)t * c->nb;
                    const uint16_t *vs1 = vs0 + c->nb;
                    const float st0 = scores[t], st1 = scores[t + 1];
                    for (int blk = 0; blk < c->nb; blk++) {
                        const float sv0 = f16_to_f32_local(vs0[blk]);
                        const float sv1 = f16_to_f32_local(vs1[blk]);
                        const uint8_t *b0 = vb0 + blk * c->bw, *b1 = vb1 + blk * c->bw;
                        int d = blk * FENG_KV_Q2_BLOCK;
#if FENG_Q2_VFOLD
                        const float stv0 = st0 * sv0, stv1 = st1 * sv1;
#endif
                        for (int j = 0; j < c->bw; j++) {
                            const float *l0 = g_q2_lut[b0[j]];
                            const float *l1 = g_q2_lut[b1[j]];
                            for (int k4 = 0; k4 < 4; k4++) {
#if FENG_Q2_VFOLD
                                vacc[d] += stv0 * l0[k4];
                                vacc[d] += stv1 * l1[k4];
#else
                                vacc[d] += (st0 * l0[k4]) * sv0;
                                vacc[d] += (st1 * l1[k4]) * sv1;
#endif
                                d++;
                            }
                        }
                    }
                }
                if (t <= c->pos) {
                    const uint8_t *vb = vrow + (size_t)t * c->qb;
                    const uint16_t *vs = vsrow + (size_t)t * c->nb;
                    const float st = scores[t];
                    for (int blk = 0; blk < c->nb; blk++) {
                        const float sv = f16_to_f32_local(vs[blk]);
                        const uint8_t *bb = vb + blk * c->bw;
                        int d = blk * FENG_KV_Q2_BLOCK;
#if FENG_Q2_VFOLD
                        const float stv = st * sv;
#endif
                        for (int j = 0; j < c->bw; j++) {
                            const float *lu = g_q2_lut[bb[j]];
                            for (int k4 = 0; k4 < 4; k4++) {
#if FENG_Q2_VFOLD
                                vacc[d++] += stv * lu[k4];
#else
                                vacc[d++] += (st * lu[k4]) * sv;
#endif
                            }
                        }
                    }
                }
            }
#else
            for (int t = 0; t <= c->pos; t++) {
                const uint8_t *vb = vrow + (size_t)t * c->qb;
                const uint16_t *vs = vsrow + (size_t)t * c->nb;
                const float st = scores[t];
#if FENG_Q2_LUT
                for (int blk = 0; blk < c->nb; blk++) {
                    const float sv = f16_to_f32_local(vs[blk]);
                    const uint8_t *bb = vb + blk * c->bw;
                    int d = blk * FENG_KV_Q2_BLOCK;
                    for (int j = 0; j < c->bw; j++) {
                        const float *lu = g_q2_lut[bb[j]];
                        for (int k4 = 0; k4 < 4; k4++) vacc[d++] += (st * lu[k4]) * sv;
                    }
                }
#else
                for (int blk = 0; blk < c->nb; blk++) {
                    const float sv = f16_to_f32_local(vs[blk]);
                    for (int j = 0; j < c->bw; j++) {
                        const uint8_t b = vb[blk * c->bw + j];
                        for (int k4 = 0; k4 < 4; k4++) {
                            const int d = blk * FENG_KV_Q2_BLOCK + j * 4 + k4;
                            vacc[d] += (st * ((float)((b >> (2 * k4)) & 3) - 1.5f)) * sv;
                        }
                    }
                }
#endif
            }
#endif
#endif  /* FENG_PHONE_NEON && FENG_Q2_LUT */
            for (int d = 0; d < c->hd; d++) c->attn[hh * c->hd + d] = vacc[d] * inv;
#if FENG_ATTN_PROF
            g_prof_cycles[2] += (unsigned long long)((unsigned)esp_cpu_get_cycle_count() - pv0);
#endif
        }
    }
}
#endif /* FENG_KV_Q2 */

/* 把单个位置的 K/V 写进 cache（按 KV 模式量化）。
 * 从 feng_forward_ex 抽出（2026-10-22 批量 prefill）：逐位置复用，
 * 两条前向路径写进去的字节完全一致。 */
static void kv_append_one(feng_kv_t *kv, const float *k, const float *v, int l, int pos,
                          int nh, int hd, int h)
{
    (void)h;                        /* q2 布局按 head 寻址，用不到 h；int8/fp32 用 */
    (void)nh;                       /* q2 布局用不到；int8 的 scale 按 head 存 */
    (void)hd;                       /* fp32 布局用不到；int8/q2 用 */
    /* append to KV cache (int8 with per-head fp16 scale, or fp32) */
#if FENG_KV_INT8
    int8_t *kc = (int8_t *)kv->k_cache + (size_t)l * kv->ctx * h;
    int8_t *vc = (int8_t *)kv->v_cache + (size_t)l * kv->ctx * h;
    uint16_t *ksc = kv->k_scale + (size_t)l * kv->ctx * nh;
    uint16_t *vsc = kv->v_scale + (size_t)l * kv->ctx * nh;
    for (int hh = 0; hh < nh; hh++) {
        float ak = 1e-8f, av = 1e-8f;
        for (int d = 0; d < hd; d++) {
            const float a = fabsf(k[hh * hd + d]);
            const float b = fabsf(v[hh * hd + d]);
            if (a > ak) ak = a;
            if (b > av) av = b;
        }
        const float sk = ak / 127.0f, sv = av / 127.0f;
        ksc[(size_t)pos * nh + hh] = feng_f32_to_f16(sk);
        vsc[(size_t)pos * nh + hh] = feng_f32_to_f16(sv);
        int8_t *kd = kc + (size_t)pos * h + hh * hd;
        int8_t *vd = vc + (size_t)pos * h + hh * hd;
        for (int d = 0; d < hd; d++) {
            int qk = (int)lrintf(k[hh * hd + d] / sk);
            int qv = (int)lrintf(v[hh * hd + d] / sv);
            if (qk > 127) qk = 127; else if (qk < -127) qk = -127;
            if (qv > 127) qv = 127; else if (qv < -127) qv = -127;
            kd[d] = (int8_t)qk;
            vd[d] = (int8_t)qv;
        }
    }
#elif FENG_KV_Q2
    /* 2-bit 对称量化 + 每 FENG_KV_Q2_BLOCK 个值一个 fp16 scale（scale = 块内 max|·|/1.5，
     * q ∈ {0,1,2,3} → (q-1.5)*scale）；每字节打包 4 个值（低位在前）。
     * block=8 实测比 block=16 的续写保真高很多（0.33→0.73），代价是 scale 内存翻倍。 */
    const int qb = hd / 4;
    const int nb = hd / FENG_KV_Q2_BLOCK;         /* 每 head 的块数 */
    const int bw = FENG_KV_Q2_BLOCK / 4;          /* 每块的字节数 */
    /* q2 布局是 [layer][head][t]（与 int8/fp32 的 [layer][t][head] 不同）：
     * 逐 head 扫描时 t 连续，长上下文下是 PSRAM 顺序读；[layer][t][head] 每次
     * 只取 16B 却要占一条 32B 缓存行（利用率 50%），长文时被放大。 */
    for (int hh = 0; hh < nh; hh++) {
        const size_t row = ((size_t)l * nh + hh) * kv->ctx + pos;
        uint8_t *kd = (uint8_t *)kv->k_cache + row * qb;
        uint8_t *vd = (uint8_t *)kv->v_cache + row * qb;
        uint16_t *ksp = kv->k_scale + row * nb;
        uint16_t *vsp = kv->v_scale + row * nb;
        for (int blk = 0; blk < nb; blk++) {
            const int base = hh * hd + blk * FENG_KV_Q2_BLOCK;
            float ak = 1e-8f, av = 1e-8f;
            for (int d = 0; d < FENG_KV_Q2_BLOCK; d++) {
                const float a = fabsf(k[base + d]);
                const float b = fabsf(v[base + d]);
                if (a > ak) ak = a;
                if (b > av) av = b;
            }
            const float sk = ak / 1.5f, sv = av / 1.5f;
            ksp[blk] = feng_f32_to_f16(sk);
            vsp[blk] = feng_f32_to_f16(sv);
            for (int j = 0; j < bw; j++) {
                uint8_t kb = 0, vb = 0;
                for (int k4 = 0; k4 < 4; k4++) {
                    const int d = blk * FENG_KV_Q2_BLOCK + j * 4 + k4;
                    int qk = (int)lrintf(k[hh * hd + d] / sk + 1.5f);
                    int qv = (int)lrintf(v[hh * hd + d] / sv + 1.5f);
                    if (qk < 0) qk = 0; else if (qk > 3) qk = 3;
                    if (qv < 0) qv = 0; else if (qv > 3) qv = 3;
                    kb |= (uint8_t)(qk << (2 * k4));
                    vb |= (uint8_t)(qv << (2 * k4));
                }
                kd[blk * bw + j] = kb; vd[blk * bw + j] = vb;
            }
        }
    }
#elif FENG_KV_FP16
    /* [layer][ctx][hidden]，与 fp32 同一布局，只是每值 2 字节。
     * 不学 q2 的 [layer][head][ctx]：fp16 KV 只在手机/PC 上用，
     * 定位是"精度参照"，越接近 fp32 路径越好验证。 */
    uint16_t *kc = (uint16_t *)kv->k_cache + (size_t)l * kv->ctx * h;
    uint16_t *vc = (uint16_t *)kv->v_cache + (size_t)l * kv->ctx * h;
    for (int i = 0; i < h; i++) {
        kc[(size_t)pos * h + i] = feng_f32_to_f16(k[i]);
        vc[(size_t)pos * h + i] = feng_f32_to_f16(v[i]);
    }
#else
    float *kc = (float *)kv->k_cache + (size_t)l * kv->ctx * h;
    float *vc = (float *)kv->v_cache + (size_t)l * kv->ctx * h;
    memcpy(kc + (size_t)pos * h, k, h * sizeof(float));
    memcpy(vc + (size_t)pos * h, v, h * sizeof(float));
#endif
}

/* 单个 query 位置的注意力：K 打分 + softmax + V 累加，写满 attn[nh*hd]。
 * pos 位置的 KV 必须已由调用方 append 进 cache（先写后读；批内按位置递增调用，
 * 因果序由此保证）。从 feng_forward_ex 抽出（2026-10-22 批量 prefill），
 * 逐位置复用同一份内核：两条前向路径的注意力数值完全一致。 */
static void attn_one(const feng_kv_t *kv, feng_workspace_t *ws,
                     const float *q, float *attn, int l, int pos,
                     int nh, int hd, int h)
{
    (void)ws;                       /* 仅 int8 KV 路径用（ws->scratch 当 pv） */
    (void)h;                        /* 仅 int8/fp32 布局用；q2 按 head 寻址 */
    const float scale = 1.0f / sqrtf((float)hd);
#if FENG_KV_Q2
    const int qb = hd / 4;
    const int nb = hd / FENG_KV_Q2_BLOCK;         /* 每 head 的块数 */
    const int bw = FENG_KV_Q2_BLOCK / 4;          /* 每块的字节数 */
#endif
#if FENG_KV_INT8
    const int8_t *kc = (const int8_t *)kv->k_cache + (size_t)l * kv->ctx * h;
    const int8_t *vc = (const int8_t *)kv->v_cache + (size_t)l * kv->ctx * h;
    const uint16_t *ksc = kv->k_scale + (size_t)l * kv->ctx * nh;
    const uint16_t *vsc = kv->v_scale + (size_t)l * kv->ctx * nh;
#elif !FENG_KV_Q2 && !FENG_KV_FP16
    const float *kc = (const float *)kv->k_cache + (size_t)l * kv->ctx * h;
    const float *vc = (const float *)kv->v_cache + (size_t)l * kv->ctx * h;
#endif
    memset(attn, 0, (size_t)nh * hd * sizeof(float));
#if FENG_KV_Q2
    {
        q2_attn_ctx_t ac;
        ac.kv = kv;
        ac.q = q;
        ac.attn = attn;
        ac.l = l;
        ac.pos = pos;
        ac.nh = nh;
        ac.hd = hd;
        ac.qb = qb;
        ac.nb = nb;
        ac.bw = bw;
        ac.scale = scale;
        ac.scores = &g_scores[0][0];
#if FENG_ATTN_PROF || FENG_ATTN_PARTS <= 1
        q2_attn_range(&ac, 0, 1);       /* profiling 的计数器非原子，串行跑 */
#else
        feng_par_run(q2_attn_range, &ac, FENG_ATTN_PARTS);
#endif
    }
#else
    for (int hh = 0; hh < nh; hh++) {
        const float *qh = q + hh * hd;
        float *scores = g_scores[0];
        float maxs = -1e30f;
#if FENG_KV_INT8
        for (int t = 0; t <= pos; t++) {
            const int8_t *kh = kc + (size_t)t * h + hh * hd;
            float s = 0.f;
            for (int d = 0; d < hd; d++) s += qh[d] * (float)kh[d];
            s *= f16_to_f32_local(ksc[(size_t)t * nh + hh]) * scale;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
#elif FENG_KV_Q2
        const uint8_t *krow = (const uint8_t *)kv->k_cache
                              + ((size_t)l * nh + hh) * kv->ctx * qb;
        const uint16_t *ksrow = kv->k_scale + ((size_t)l * nh + hh) * kv->ctx * nb;
#if FENG_ATTN_PROF
        const unsigned pc0 = (unsigned)esp_cpu_get_cycle_count();
#endif
#if FENG_Q2_LUT && FENG_Q2_PAIR
        /* 一次算 2 个上下文 token：两条独立的累加链填满 FPU 流水线，
         * qh 只加载一次、相邻 token 的 16B 数据同处一条 32B 缓存行。
         * 每个 score 的累加顺序与单 token 版完全一致 -> 逐位一致。 */
        {
            int t = 0;
            for (; t + 1 <= pos; t += 2) {
                const uint8_t *kh0 = krow + (size_t)t * qb;
                const uint8_t *kh1 = kh0 + qb;
                const uint16_t *ks0 = ksrow + (size_t)t * nb;
                const uint16_t *ks1 = ks0 + nb;
                float s0 = 0.f, s1 = 0.f;
                for (int blk = 0; blk < nb; blk++) {
                    const float sk0 = f16_to_f32_local(ks0[blk]);
                    const float sk1 = f16_to_f32_local(ks1[blk]);
                    const uint8_t *b0 = kh0 + blk * bw, *b1 = kh1 + blk * bw;
                    int d = blk * FENG_KV_Q2_BLOCK;
                    for (int j = 0; j < bw; j++) {
                        const float *l0 = g_q2_lut[b0[j]];
                        const float *l1 = g_q2_lut[b1[j]];
                        for (int k4 = 0; k4 < 4; k4++) {
                            const float qd = qh[d++];
                            s0 += qd * l0[k4] * sk0;
                            s1 += qd * l1[k4] * sk1;
                        }
                    }
                }
                s0 *= scale;
                s1 *= scale;
                scores[t] = s0;
                scores[t + 1] = s1;
                if (s0 > maxs) maxs = s0;
                if (s1 > maxs) maxs = s1;
            }
            if (t <= pos) {                      /* 奇数长度：尾部单 token */
                const uint8_t *kh = krow + (size_t)t * qb;
                const uint16_t *ks = ksrow + (size_t)t * nb;
                float s = 0.f;
                for (int blk = 0; blk < nb; blk++) {
                    const float sk = f16_to_f32_local(ks[blk]);
                    const uint8_t *bb = kh + blk * bw;
                    int d = blk * FENG_KV_Q2_BLOCK;
                    for (int j = 0; j < bw; j++) {
                        const float *lu = g_q2_lut[bb[j]];
                        for (int k4 = 0; k4 < 4; k4++) s += qh[d++] * lu[k4] * sk;
                    }
                }
                s *= scale;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
        }
#else
        for (int t = 0; t <= pos; t++) {
            const uint8_t *kh = krow + (size_t)t * qb;
            const uint16_t *ks = ksrow + (size_t)t * nb;
            float s = 0.f;
#if FENG_Q2_LUT
            for (int blk = 0; blk < nb; blk++) {
                const float sk = f16_to_f32_local(ks[blk]);
                const uint8_t *bb = kh + blk * bw;
                int d = blk * FENG_KV_Q2_BLOCK;
                for (int j = 0; j < bw; j++) {
                    const float *lu = g_q2_lut[bb[j]];
                    for (int k4 = 0; k4 < 4; k4++) {
                        s += qh[d++] * lu[k4] * sk;
                    }
                }
            }
#else
            for (int blk = 0; blk < nb; blk++) {
                const float sk = f16_to_f32_local(ks[blk]);
                for (int j = 0; j < bw; j++) {
                    const uint8_t b = kh[blk * bw + j];
                    for (int k4 = 0; k4 < 4; k4++) {
                        const int qq = (b >> (2 * k4)) & 3;
                        s += qh[blk * FENG_KV_Q2_BLOCK + j * 4 + k4]
                             * ((float)qq - 1.5f) * sk;
                    }
                }
            }
#endif
            s *= scale;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
#endif
#if FENG_ATTN_PROF
        {
            const unsigned pc1 = (unsigned)esp_cpu_get_cycle_count();
            g_prof_cycles[0] += (unsigned long long)(pc1 - pc0);
        }
#endif
#elif FENG_KV_FP16
        {
            const uint16_t *krow = (const uint16_t *)kv->k_cache
                                 + (size_t)l * kv->ctx * h + (size_t)hh * hd;
            for (int t = 0; t <= pos; t++) {
                const uint16_t *kh = krow + (size_t)t * h;
                float s = 0.f;
                for (int d = 0; d < hd; d++) s += qh[d] * f16_to_f32_local(kh[d]);
                s *= scale;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
        }
#else
        for (int t = 0; t <= pos; t++) {
            const float *kh = kc + (size_t)t * h + hh * hd;
            float s = 0.f;
            for (int d = 0; d < hd; d++) s += qh[d] * kh[d];
            s *= scale;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
#endif
        float sum = 0.f;
#if FENG_ATTN_PROF
        const unsigned ps0 = (unsigned)esp_cpu_get_cycle_count();
#endif
        for (int t = 0; t <= pos; t++) {
#if FENG_FAST_EXP
            scores[t] = fast_expf(scores[t] - maxs);
#else
            scores[t] = expf(scores[t] - maxs);
#endif
            sum += scores[t];
        }
        const float inv = 1.0f / sum;
#if FENG_ATTN_PROF
        g_prof_cycles[1] += (unsigned long long)((unsigned)esp_cpu_get_cycle_count() - ps0);
#endif
#if FENG_KV_INT8
        float *pv = ws->scratch;      /* scaled attention probabilities（仅 int8 路径用） */
        for (int t = 0; t <= pos; t++) {
            pv[t] = scores[t] * f16_to_f32_local(vsc[(size_t)t * nh + hh]);
        }
        for (int d = 0; d < hd; d++) {
            float acc = 0.f;
            for (int t = 0; t <= pos; t++) {
                acc += pv[t] * (float)vc[(size_t)t * h + hh * hd + d];
            }
            attn[hh * hd + d] = acc * inv;
        }
#elif FENG_KV_Q2
        /* 连续访问版：原实现对每个输出维度 d 都按 112B 跨步去 PSRAM 抓 1 个字节
         * （缓存行利用率 1/32，长上下文时被放大成几十秒/轮）。改成 t 外层、
         * 每个 (t,head) 的 16 个打包字节顺序读完再累加；每个 d 对 t 的求和顺序
         * 与乘法分组 (scores[t]*bit)*sv 保持不变，结果逐位一致。 */
        {
            float vacc[FENG_MAX_HEAD_DIM];
            for (int d = 0; d < hd; d++) vacc[d] = 0.f;
#if FENG_ATTN_PROF
            const unsigned pv0 = (unsigned)esp_cpu_get_cycle_count();
#endif
            const uint8_t *vrow = (const uint8_t *)kv->v_cache
                                  + ((size_t)l * nh + hh) * kv->ctx * qb;
            const uint16_t *vsrow = kv->v_scale + ((size_t)l * nh + hh) * kv->ctx * nb;
#if FENG_Q2_LUT && FENG_Q2_PAIR
            {
                int t = 0;
                for (; t + 1 <= pos; t += 2) {
                    const uint8_t *vb0 = vrow + (size_t)t * qb;
                    const uint8_t *vb1 = vb0 + qb;
                    const uint16_t *vs0 = vsrow + (size_t)t * nb;
                    const uint16_t *vs1 = vs0 + nb;
                    const float st0 = scores[t], st1 = scores[t + 1];
                    for (int blk = 0; blk < nb; blk++) {
                        const float sv0 = f16_to_f32_local(vs0[blk]);
                        const float sv1 = f16_to_f32_local(vs1[blk]);
                        const uint8_t *b0 = vb0 + blk * bw, *b1 = vb1 + blk * bw;
                        int d = blk * FENG_KV_Q2_BLOCK;
#if FENG_Q2_VFOLD
                        const float stv0 = st0 * sv0, stv1 = st1 * sv1;
#endif
                        for (int j = 0; j < bw; j++) {
                            const float *l0 = g_q2_lut[b0[j]];
                            const float *l1 = g_q2_lut[b1[j]];
                            for (int k4 = 0; k4 < 4; k4++) {
#if FENG_Q2_VFOLD
                                vacc[d] += stv0 * l0[k4];
                                vacc[d] += stv1 * l1[k4];
#else
                                vacc[d] += (st0 * l0[k4]) * sv0;
                                vacc[d] += (st1 * l1[k4]) * sv1;
#endif
                                d++;
                            }
                        }
                    }
                }
                if (t <= pos) {                  /* 奇数长度：尾部单 token */
                    const uint8_t *vb = vrow + (size_t)t * qb;
                    const uint16_t *vs = vsrow + (size_t)t * nb;
                    const float st = scores[t];
                    for (int blk = 0; blk < nb; blk++) {
                        const float sv = f16_to_f32_local(vs[blk]);
                        const uint8_t *bb = vb + blk * bw;
                        int d = blk * FENG_KV_Q2_BLOCK;
#if FENG_Q2_VFOLD
                        const float stv = st * sv;
#endif
                        for (int j = 0; j < bw; j++) {
                            const float *lu = g_q2_lut[bb[j]];
                            for (int k4 = 0; k4 < 4; k4++) {
#if FENG_Q2_VFOLD
                                vacc[d++] += stv * lu[k4];
#else
                                vacc[d++] += (st * lu[k4]) * sv;
#endif
                            }
                        }
                    }
                }
            }
#else
            for (int t = 0; t <= pos; t++) {
                const uint8_t *vb = vrow + (size_t)t * qb;
                const uint16_t *vs = vsrow + (size_t)t * nb;
                const float st = scores[t];
#if FENG_Q2_LUT
                for (int blk = 0; blk < nb; blk++) {
                    const float sv = f16_to_f32_local(vs[blk]);
                    const uint8_t *bb = vb + blk * bw;
                    int d = blk * FENG_KV_Q2_BLOCK;
                    for (int j = 0; j < bw; j++) {
                        const float *lu = g_q2_lut[bb[j]];
                        for (int k4 = 0; k4 < 4; k4++) {
                            vacc[d++] += (st * lu[k4]) * sv;
                        }
                    }
                }
#else
                for (int blk = 0; blk < nb; blk++) {
                    const float sv = f16_to_f32_local(vs[blk]);
                    for (int j = 0; j < bw; j++) {
                        const uint8_t b = vb[blk * bw + j];
                        for (int k4 = 0; k4 < 4; k4++) {
                            const int d = blk * FENG_KV_Q2_BLOCK + j * 4 + k4;
                            vacc[d] += (st * ((float)((b >> (2 * k4)) & 3) - 1.5f)) * sv;
                        }
                    }
                }
#endif
            }
#endif
            for (int d = 0; d < hd; d++) attn[hh * hd + d] = vacc[d] * inv;
#if FENG_ATTN_PROF
            g_prof_cycles[2] += (unsigned long long)((unsigned)esp_cpu_get_cycle_count() - pv0);
#endif
        }
#elif FENG_KV_FP16
        {
            const uint16_t *vrow = (const uint16_t *)kv->v_cache
                                 + (size_t)l * kv->ctx * h + (size_t)hh * hd;
            for (int d = 0; d < hd; d++) {
                float acc = 0.f;
                for (int t = 0; t <= pos; t++) {
                    acc += scores[t] * f16_to_f32_local(vrow[(size_t)t * h + d]);
                }
                attn[hh * hd + d] = acc * inv;
            }
        }
#else
        for (int d = 0; d < hd; d++) {
            float acc = 0.f;
            for (int t = 0; t <= pos; t++) {
                acc += scores[t] * vc[(size_t)t * h + hh * hd + d];
            }
            attn[hh * hd + d] = acc * inv;
        }
#endif
    }
#endif
}


/* embedding 查表（fp16 一行或 Q4 一行解量化），写进 x[h]。
 * 从 feng_forward_ex 抽出（2026-10-22 批量 prefill）：逐 token 复用。 */
static void emb_lookup(const feng_model_t *m, int token, float *x, int h)
{
/* embedding lookup (fp16) */
if (m->tok_embd_dtype == FENG_DT_FP16) {
    const uint16_t *emb = (const uint16_t *)m->tok_embd;
    f16_to_f32_vec(emb + (size_t)token * h, x, h);
} else {
    /* Q4 embedding row dequant (block 64) for a single row */
    const uint8_t *base = (const uint8_t *)m->tok_embd;
    const int n_blocks = h / 64;
    const size_t row_bytes = (size_t)n_blocks * (2 + 32);
    const uint8_t *row = base + (size_t)token * row_bytes;
    const uint8_t *scales = row;
    const uint8_t *packed = row + (size_t)n_blocks * 2;
    for (int b = 0; b < n_blocks; b++) {
        uint16_t hs;
        memcpy(&hs, scales + b * 2, 2);
        float sc;
        {
            uint32_t sign = (hs >> 15) & 1u, exp = (hs >> 10) & 0x1Fu, man = hs & 0x3FFu, f;
            if (exp == 0) {
                if (man == 0) f = sign << 31;
                else { exp = 113; while ((man & 0x400u) == 0) { man <<= 1; exp--; } man &= 0x3FFu;
                       f = (sign << 31) | (exp << 23) | (man << 13); }
            } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
            else f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
            memcpy(&sc, &f, 4);
        }
        const uint8_t *p = packed + (size_t)b * 32;
        for (int j = 0; j < 32; j++) {
            const int lo = (int)(p[j] & 0x0F) - 8;
            const int hi = (int)(p[j] >> 4) - 8;
            x[b * 64 + 2 * j] = (float)lo * sc;
            x[b * 64 + 2 * j + 1] = (float)hi * sc;
        }
    }
}
}

float *feng_forward(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos)
{
    return feng_forward_ex(m, kv, ws, token, pos, 1);
}

float *feng_forward_ex(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos,
                       int want_logits)
{
    if (m == NULL || kv == NULL || ws == NULL) return NULL;
    if (pos < 0 || pos >= kv->ctx || pos >= FENG_MAX_SCORES) return NULL;
    if (token < 0 || token >= (int)m->hdr.vocab) return NULL;
#if defined(FENG_USE_CUDA)
    /* 完整 GPU forward（q2 KV）：权重/KV/激活常驻显存，单 token 一次提交。
     * 不可用或失败时回退下面的 CPU 路径（失败发生在资源分配阶段，KV 状态未污染）。 */
    if (feng_cuda_forward_supported(m, kv) && feng_cuda_forward_ready()) {
        float *lg = feng_cuda_forward(m, kv, ws, token, pos, want_logits);
        if (lg) return lg;
    }
#endif
    const int h = m->hdr.hidden, nh = m->hdr.n_heads, hd = m->hdr.head_dim;
    const int f = m->hdr.ffn;
    float *x = ws->x, *xn = ws->xn, *q = ws->q, *k = ws->k, *v = ws->v;
    float *attn = ws->attn, *proj = ws->proj, *gate = ws->gate, *up = ws->up, *ffn = ws->ffn;
#if FENG_KV_Q2 && FENG_Q2_LUT
    q2_lut_init();
#endif

    emb_lookup(m, token, x, h);

    for (int l = 0; l < (int)m->hdr.n_layers; l++) {
        const feng_layer_t *L = &m->layers[l];
        rmsnorm(xn, x, L->attn_norm, h, m->hdr.rms_eps);
        /* dtype 来自模型文件（fp16 或 Q4）。feng_gemv3 只有一个 dtype 参数，
         * 因此要求同一次合并的 q/k/v 格式相同——导出器对同一层的矩阵用同一套
         * maybe() 决策，正常情况下不会混合；真混合了会在加载期被 shape 校验挡住。 */
        feng_gemv3(L->wq, L->wk, L->wv, FENG_LAYER_DTYPE(L), xn, q, k, v, h, h, h, h);

        /* per-head q/k RMSNorm (Qwen3) */
        {
            for (int hh = 0; hh < nh; hh++) {
                rmsnorm(g_tmp, q + hh * hd, L->q_norm, hd, m->hdr.rms_eps);
                memcpy(q + hh * hd, g_tmp, hd * sizeof(float));
                rmsnorm(g_tmp, k + hh * hd, L->k_norm, hd, m->hdr.rms_eps);
                memcpy(k + hh * hd, g_tmp, hd * sizeof(float));
            }
        }
        rope(q, nh, hd, pos, m->hdr.rope_theta);
        rope(k, nh, hd, pos, m->hdr.rope_theta);

        kv_append_one(kv, k, v, l, pos, nh, hd, h);
        attn_one(kv, ws, q, attn, l, pos, nh, hd, h);
        feng_gemv_par(L->wo, FENG_LAYER_DTYPE(L), attn, proj, h, h);
        for (int i = 0; i < h; i++) x[i] += proj[i];

        rmsnorm(xn, x, L->ffn_norm, h, m->hdr.rms_eps);
        /* gate/up 共用一次合并调用，前提是两者格式相同（导出器保证）；不同则要拆两次 */
        feng_gemv3(L->gate, L->up, NULL, FENG_LAYER_DTYPE(L), xn, gate, up, NULL, f, f, 0, h);
        silu_mul(ffn, gate, up, f);
        feng_gemv_par(L->down, FENG_LAYER_DTYPE(L), ffn, proj, h, f);
        for (int i = 0; i < h; i++) x[i] += proj[i];
    }
    if (want_logits) {
        rmsnorm(xn, x, m->out_norm, h, m->hdr.rms_eps);
        /* tied lm head: logits = xn @ tok_embd^T (fp16 weights) */
        feng_gemv_par(m->tok_embd, m->tok_embd_dtype, xn, ws->logits, m->hdr.vocab, h);
    }
    kv->len = pos + 1;
    return ws->logits;
}
