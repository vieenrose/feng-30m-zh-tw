/* aarch64 NEON 内核：Q4 block-64 GEMV 走 ARMv8.2 int8 点积（SDOT）。
 * 设计说明见 feng_neon.h 头部注释。 */

/* 同 feng_phone.c：-std=c23 下要显式打开特性宏才能拿到 getauxval / sys/auxv.h */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

#include "feng_neon.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include "feng.h"          /* feng_f16_to_f32：与标量路径共用同一份 fp16 转换 */

#include <arm_neon.h>
#include <string.h>

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP_ASIMDHP_BIT
#define HWCAP_ASIMDHP_BIT (1UL << 10)   /* 与 <asm/hwcap.h> 的 HWCAP_ASIMDHP 同值 */
#endif
#endif

/* 只给这两个函数打开 dotprod：这样单个二进制既能跑在 A53/A72（回退标量），
 * 又能在 A55/A76+ 上用上 sdot，不需要给整包加 -march=armv8.2-a+dotprod
 * （那会让老机器直接 SIGILL）。
 * 注意 GCC 要求扩展名带 '+' 前缀（Clang 两种写法都收）。 */
#if defined(__GNUC__) || defined(__clang__)
#define FENG_DOTPROD_FN __attribute__((target("+dotprod")))
#else
#define FENG_DOTPROD_FN
#endif

/* 同理给 fp16 内核单独开特性：Clang 叫 +fullfp16，GCC 叫 +fp16。
 * 不给整包加 -march，老核心上其余代码仍按基线指令集编译，不会 SIGILL。 */
#if defined(__clang__)
#define FENG_FP16_FN __attribute__((target("+fullfp16")))
#elif defined(__GNUC__)
#define FENG_FP16_FN __attribute__((target("+fp16")))
#else
#define FENG_FP16_FN
#endif

int feng_neon_fp16_ok(void)
{
    static int cached = -1;
    if (cached < 0) {
#if defined(__linux__) || defined(__ANDROID__)
        cached = (getauxval(AT_HWCAP) & HWCAP_ASIMDHP_BIT) ? 1 : 0;
#else
        cached = 0;
#endif
    }
    return cached;
}

FENG_FP16_FN
void feng_neon_fp16_gemv(const uint16_t *w, const float *x, float *y,
                         int r0, int r1, int n_in)
{
    for (int o = r0; o < r1; o++) {
        const uint16_t *row = w + (size_t)o * n_in;
        float32x4_t a0 = vdupq_n_f32(0.f);
        float32x4_t a1 = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 8 <= n_in; i += 8) {
            const float16x8_t hw = vreinterpretq_f16_u16(vld1q_u16(row + i));
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(hw)),  vld1q_f32(x + i));
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(hw)), vld1q_f32(x + i + 4));
        }
        float acc = vaddvq_f32(vaddq_f32(a0, a1));
        for (; i < n_in; i++) acc += feng_f16_to_f32(row[i]) * x[i];
        y[o] = acc;
    }
}

int feng_neon_sdot_ok(void)
{
    static int cached = -1;
    if (cached < 0) {
#if defined(__linux__) || defined(__ANDROID__)
        cached = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) ? 1 : 0;
#else
        cached = 0;
#endif
    }
    return cached;
}

/* nibble(0..15) -> 有符号权重(-8..7) 的 16 项查表常量。
 * 字节序：低 8 字节是 -8..-1（0xF8..0xFF），高 8 字节是 0..7 —— 小端下正好按
 * 下标 0..15 顺序排列。 */
static const uint64_t k_w_tbl[2] = { 0xFFFEFDFCFBFAF9F8ull, 0x0706050403020100ull };

/* 64 个权重 = 32 个打包字节：拆 nibble -> 还原 d 顺序 -> 查表成 int8 -> sdot。
 * 展开顺序说明（与 feng_quant.c 的行布局一致）：
 *   字节 j 的低半字节 = 权重下标 2j，高半字节 = 2j+1；
 *   vzip1q(lo, hi) 给出 d 0..15，vzip2q(lo, hi) 给出 d 16..31。 */
FENG_DOTPROD_FN static void q4_block_sdot(const uint8_t *p, const int8_t *xq,
                                          const int8x16_t wtbl, int32x4_t *acc)
{
    const uint8x16_t p0 = vld1q_u8(p);          /* 打包字节 0..15  -> d 0..31  */
    const uint8x16_t p1 = vld1q_u8(p + 16);     /* 打包字节 16..31 -> d 32..63 */
    const uint8x16_t mask = vdupq_n_u8(0x0F);

    const uint8x16_t lo0 = vandq_u8(p0, mask);
    const uint8x16_t hi0 = vshrq_n_u8(p0, 4);
    const uint8x16_t lo1 = vandq_u8(p1, mask);
    const uint8x16_t hi1 = vshrq_n_u8(p1, 4);

    const int8x16_t w0 = vqtbl1q_s8(wtbl, vzip1q_u8(lo0, hi0));   /* d 0..15  */
    const int8x16_t w1 = vqtbl1q_s8(wtbl, vzip2q_u8(lo0, hi0));   /* d 16..31 */
    const int8x16_t w2 = vqtbl1q_s8(wtbl, vzip1q_u8(lo1, hi1));   /* d 32..47 */
    const int8x16_t w3 = vqtbl1q_s8(wtbl, vzip2q_u8(lo1, hi1));   /* d 48..63 */

    *acc = vdotq_s32(*acc, w0, vld1q_s8(xq + 0));
    *acc = vdotq_s32(*acc, w1, vld1q_s8(xq + 16));
    *acc = vdotq_s32(*acc, w2, vld1q_s8(xq + 32));
    *acc = vdotq_s32(*acc, w3, vld1q_s8(xq + 48));
}

void feng_neon_q4_gemv(const uint8_t *base, const int8_t *xq, const float *xscale,
                       float *y, int r0, int r1, int n_in)
{
    const int n_blocks = n_in / 64;
    const size_t row_bytes = (size_t)n_blocks * (2 + 32);
    const int8x16_t wtbl = vreinterpretq_s8_u64(vld1q_u64(k_w_tbl));

    for (int o = r0; o < r1; o++) {
        const uint8_t *row = base + (size_t)o * row_bytes;
        const uint8_t *scales = row;
        const uint8_t *packed = row + (size_t)n_blocks * 2;
        float acc = 0.f;

        for (int b = 0; b < n_blocks; b++) {
            const float sx = xscale[b];
            if (sx == 0.f) continue;        /* 与标量 A8 路径同口径：整块激活为 0 则跳过 */

            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float sw = feng_f16_to_f32(hs);
            const uint8_t *p = packed + (size_t)b * 32;
            const int8_t *xb = xq + (size_t)b * 64;

            int32x4_t dacc = vdupq_n_s32(0);
            q4_block_sdot(p, xb, wtbl, &dacc);
            /* 整数点积是精确的（分组不同不影响结果），所以这里的 (float)dot
             * 与标量 A8 逐位相同；下面的累加顺序也与标量一致。 */
            acc += (float)vaddvq_s32(dacc) * (sw * sx);
        }
        y[o] = acc;
    }
}

/* ---- q2 KV 注意力（fp32 NEON，不需要 dotprod） ---- */

/* K 打分：t ∈ [t0,t1)。每 (t,blk) 两条 4 路 FMA 链，最后水平归约。
 * 与标量版的差别只在累加分组（4 路并行 vs 单链），fp32 噪声级。 */
void feng_neon_q2_attn_k(const uint8_t *krow, const uint16_t *ksrow,
                         const float *qh, float *scores,
                         int t0, int t1, int qb, int nb, float scale,
                         const float (*lut)[4])
{
    for (int t = t0; t < t1; t++) {
        const uint8_t *kh = krow + (size_t)t * qb;
        const uint16_t *ks = ksrow + (size_t)t * nb;
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
        for (int blk = 0; blk < nb; blk++) {
            const float sk = feng_f16_to_f32(ks[blk]);
            const uint8_t *bb = kh + blk * 2;
            const float *q0 = qh + blk * 8;
            const float32x4_t s4 = vdupq_n_f32(sk);
            a0 = vfmaq_f32(a0, vmulq_f32(vld1q_f32(lut[bb[0]]), vld1q_f32(q0)), s4);
            a1 = vfmaq_f32(a1, vmulq_f32(vld1q_f32(lut[bb[1]]), vld1q_f32(q0 + 4)), s4);
        }
        scores[t] = (vaddvq_f32(a0) + vaddvq_f32(a1)) * scale;
    }
}

/* V 累加：只算 blk 指定的 8 维块。乘法分组 (st*lu)*sv 与标量版逐字相同、
 * 每个 d 按 t 递增累加 —— 与标量路径逐位一致。 */
void feng_neon_q2_attn_v(const uint8_t *vrow, const uint16_t *vsrow,
                         const float *scores, float *vacc,
                         int total, int qb, int nb, int blk,
                         const float (*lut)[4])
{
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    for (int t = 0; t < total; t++) {
        const float st = scores[t];
        const float sv = feng_f16_to_f32(vsrow[(size_t)t * nb + blk]);
        const uint8_t *bb = vrow + (size_t)t * qb + blk * 2;
        const float32x4_t sv4 = vdupq_n_f32(sv);
        const float32x4_t st4 = vdupq_n_f32(st);
        a0 = vaddq_f32(a0, vmulq_f32(vmulq_f32(vld1q_f32(lut[bb[0]]), st4), sv4));
        a1 = vaddq_f32(a1, vmulq_f32(vmulq_f32(vld1q_f32(lut[bb[1]]), st4), sv4));
    }
    vst1q_f32(vacc + blk * 8, a0);
    vst1q_f32(vacc + blk * 8 + 4, a1);
}

#else  /* 非 aarch64：空实现，保证链接通过（调用方本来就只在 FENG_PHONE_NEON=1 时用） */

int feng_neon_sdot_ok(void) { return 0; }
int feng_neon_fp16_ok(void) { return 0; }

void feng_neon_fp16_gemv(const uint16_t *w, const float *x, float *y,
                         int r0, int r1, int n_in)
{
    (void)w; (void)x; (void)y; (void)r0; (void)r1; (void)n_in;
}

void feng_neon_q4_gemv(const uint8_t *base, const int8_t *xq, const float *xscale,
                       float *y, int r0, int r1, int n_in)
{
    (void)base; (void)xq; (void)xscale; (void)y; (void)r0; (void)r1; (void)n_in;
}

void feng_neon_q2_attn_k(const uint8_t *krow, const uint16_t *ksrow,
                         const float *qh, float *scores,
                         int t0, int t1, int qb, int nb, float scale,
                         const float (*lut)[4])
{
    (void)krow; (void)ksrow; (void)qh; (void)scores; (void)t0; (void)t1;
    (void)qb; (void)nb; (void)scale; (void)lut;
}

void feng_neon_q2_attn_v(const uint8_t *vrow, const uint16_t *vsrow,
                         const float *scores, float *vacc,
                         int total, int qb, int nb, int blk,
                         const float (*lut)[4])
{
    (void)vrow; (void)vsrow; (void)scores; (void)vacc; (void)total;
    (void)qb; (void)nb; (void)blk; (void)lut;
}

#endif
