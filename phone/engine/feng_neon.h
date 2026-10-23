/* aarch64 NEON 内核（手机端专用；非 aarch64 编译期整体为空实现）。
 *
 * 只做一件事：把 Q4 block-64 的 GEMV 内层循环换成 ARMv8.2 的 int8 点积（SDOT）。
 *
 * 为什么是 int8 点积而不是 fp32 SIMD：
 *  ARM64 的 fp32 FMA 只有 4 个通道，而 4-bit 权重必须先做
 *  「nibble 展开 -> u8->u16->u32 扩宽 -> int->float 转换」这一串，
 *  实测这套转换本身就吃掉大部分收益（粗算每 64 权重约 75 条 SIMD 指令），
 *  和标量 madd 链（每 64 权重约 96 条）几乎打平。
 *  而 `vdotq_s32` 一条指令做 16 次 int8 乘加：每 64 权重只剩 4 条 sdot
 *  + 4 条表查（vqtbl1q_s8 把 4-bit 映射成 int8 权重）+ 展开 10 条 ≈ 18 条。
 *  这就是 llama.cpp 在 arm64 上的主路径。
 *
 * 代价与前提：
 *  - 激活必须先量化成 int8（engine 里已有 feng_gemv_a8_prepare，每 64 值一块、
 *    scale = max|x|/127）。**这不是新增的近似**：它与引擎已有的 FENG_GEMV_A8
 *    标量路径完全同一套公式，本内核刻意保持同样的式子
 *    `acc += (float)dot * (scale_w * scale_x)`、同样的块顺序，
 *    因此两者结果**逐位一致**（整数点积是精确的，只是分组不同）。
 *  - 需要 ARMv8.2 的 dotprod 扩展（Cortex-A55/A76 及以后；A53/A72 没有）。
 *    运行期用 getauxval(AT_HWCAP) 探测，没有就回退标量路径。
 */
#ifndef FENG_NEON_H
#define FENG_NEON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 本机 CPU 是否支持 ARMv8.2 dotprod（首次调用做一次 getauxval，结果缓存） */
int feng_neon_sdot_ok(void);

/* 本机 CPU 是否支持 ARMv8.2 FP16 向量运算（HWCAP_ASIMDHP） */
int feng_neon_fp16_ok(void);

/* fp16 行主序权重的 GEMV，只算输出行 [r0, r1)。
 * 每 8 个权重：vcvt_f32_f16 拆成两组 float32x4，与 x 对应段做 vfmaq_f32。
 * **不承诺与标量 fp16 路径逐位一致**：转换是精确的，但累加顺序不同。 */
void feng_neon_fp16_gemv(const uint16_t *w, const float *x, float *y,
                         int r0, int r1, int n_in);

/* Q4 block-64 GEMV 的 sdot 内核，只算输出行 [r0, r1)。
 *   base      张量起始（每行 = [nb×fp16 scale(2B)] ++ [nb×32B packed]）
 *   xq/xscale feng_gemv_a8_prepare 量化好的激活（每 64 值一块）
 *   n_in      输入维度（必须是 64 的倍数，与 feng_quant.c 的 QK 一致）
 * 调用方负责确认已 prepare 过（xq 与当前 x 对应）。 */
void feng_neon_q4_gemv(const uint8_t *base, const int8_t *xq, const float *xscale,
                       float *y, int r0, int r1, int n_in);


/* q2 KV 注意力的 NEON 内核（手机端 4 核分块里的每块）。
 * K 打分：scores[t] = (Σ_blk sk * Σ_{d in blk} qh[d] * deq(2bit)) * scale，
 *   每个 (t, blk) 两条 4 路 FMA 链 + 水平归约——累加分组与标量版不同，
 *   结果在 fp32 噪声级（~1e-7 相对）。
 * V 累加：只算 blk 指定的 8 维块，每个 d 按 t 递增累加；乘法分组必须是
 *   (st*sv)*lu（与标量 FENG_Q2_VFOLD=1 的分组逐字相同），这样才逐位一致。
 *   写成 (st*lu)*sv 数学等价、浮点不等价——2026-10-23 实测把 mem12 从
 *   11/12 打到 7/12，别改回去。
 * lut 是 256x4 的 2bit->float 查表（由 feng_llm.c 的 g_q2_lut 传入）。 */
void feng_neon_q2_attn_k(const uint8_t *krow, const uint16_t *ksrow,
                         const float *qh, float *scores,
                         int t0, int t1, int qb, int nb, float scale,
                         const float (*lut)[4]);
void feng_neon_q2_attn_v(const uint8_t *vrow, const uint16_t *vsrow,
                         const float *scores, float *vacc,
                         int total, int qb, int nb, int blk,
                         const float (*lut)[4]);
#ifdef __cplusplus
}
#endif

#endif /* FENG_NEON_H */
