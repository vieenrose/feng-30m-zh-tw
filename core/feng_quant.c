/* Q4 (block 64, fp16 scale) + fp16 GEMV.  Layout per output row:
 *   [scale fp16] x n_blocks   followed by   [packed nibbles] x n_blocks x 32 bytes
 * nibble k of a block holds weight index 2k (low) and 2k+1 (high), value = (n-8)*scale
 */
#include "feng.h"

#ifdef FENG_PHONE
#include "feng_phone.h"          /* 手机端：NEON 运行期开关、线程池 */
#if FENG_PHONE_NEON
#include "feng_neon.h"           /* aarch64 int8 点积（SDOT）的 Q4 GEMV 内核 */
#endif
#endif

#if defined(FENG_USE_CUDA)
#include "feng_cuda.h"
#endif

#include <math.h>
#include <string.h>

#if defined(FENG_USE_OMP)
#include <omp.h>            /* PC 多线程：输出行之间完全独立，逐位结果与单线程相同 */
#include <stdlib.h>         /* getenv */
#endif

#define QK 64

/* byte -> (low nibble - 8, high nibble - 8) as floats, built once into internal RAM.
 * Replaces 2 shifts + 2 subtracts + 2 int->float converts per packed byte with two
 * 32-bit loads, which is the dominant per-weight cost of the Q4 kernel. */
static float s_q4_lut[256][2];
static int s_q4_lut_ready;

#if FENG_GEMV_A8 || FENG_GEMV_PIE || FENG_PHONE_NEON
/* A8/PIE 的激活量化缓冲区：每个 GEMV 调用准备一次（prepare），行循环只读。
 * 推理是单线程的，两个 SMP worker 都在 prepare 之后并发读同一份数据。 */
#define A8_MAX_IN 1024
static int8_t s_a8_xq[A8_MAX_IN];
static float s_a8_xscale[A8_MAX_IN / QK];
static const float *s_a8_x;
static int s_a8_nin;
#if FENG_GEMV_PIE
alignas(16) static int16_t s_x16[A8_MAX_IN];
/* 256 项 pair LUT：打包字节 -> 两个 int16 权重 (lo-8, hi-8)，按 u32 packed */
static uint32_t s_q4_pair32[256];
static int s_q4_pair32_ready;
int32_t pie_dot64(const int16_t *w, const int16_t *x);
static void q4_pair32_init(void)
{
    if (s_q4_pair32_ready) return;
    for (int b = 0; b < 256; b++) {
        const int16_t lo = (int16_t)((b & 0x0F) - 8);
        const int16_t hi = (int16_t)(((b >> 4) & 0x0F) - 8);
        s_q4_pair32[b] = (uint32_t)(uint16_t)lo | ((uint32_t)(uint16_t)hi << 16);
    }
    s_q4_pair32_ready = 1;
}

/* 自检：同一行、同一份 int8 激活，PIE 点积 vs 纯标量整数点积。
 * 两边都是整数运算、同样的操作数，结果必须逐位一致（返回 0 = MATCH）。 */
int feng_gemv_pie_selfcheck(const void *tensor, int n_in)
{
    if (!(s_a8_x && s_a8_nin == n_in)) return -1;      /* 需要先 prepare */
    q4_pair32_init();
    const uint8_t *row = (const uint8_t *)tensor;
    const int n_blocks = n_in / QK;
    const uint8_t *packed = row + (size_t)n_blocks * 2;
    int32_t pie_sum = 0, ref_sum = 0;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *p = packed + (size_t)b * 32;
        alignas(16) int16_t w16[QK];
        uint32_t *st = (uint32_t *)w16;
        for (int j = 0; j < 32; j++) st[j] = s_q4_pair32[p[j]];
        pie_sum += pie_dot64(w16, s_x16 + (size_t)b * QK);
        int32_t r = 0;
        for (int j = 0; j < 32; j++) {
            const int lo = (int)(p[j] & 0x0F) - 8;
            const int hi = (int)(p[j] >> 4) - 8;
            r += lo * (int)s_a8_xq[(size_t)b * QK + 2 * j]
                 + hi * (int)s_a8_xq[(size_t)b * QK + 2 * j + 1];
        }
        ref_sum += r;
    }
    return pie_sum == ref_sum ? 0 : 1;
}
#endif

void feng_gemv_a8_prepare(const float *x, int n_in)
{
    if (n_in > A8_MAX_IN || (n_in % QK) != 0) { s_a8_x = 0; s_a8_nin = 0; return; }
    const int n_blocks = n_in / QK;
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + (size_t)b * QK;
        float amax = 0.f;
        for (int i = 0; i < QK; i++) {
            const float v = xb[i] < 0 ? -xb[i] : xb[i];
            if (v > amax) amax = v;
        }
        int8_t *xq = s_a8_xq + (size_t)b * QK;
        if (amax < 1e-8f) {            /* 与原逐行实现一致：整块跳过 */
            s_a8_xscale[b] = 0.f;
            memset(xq, 0, QK);
            continue;
        }
        const float scale_x = amax / 127.f;
        s_a8_xscale[b] = scale_x;
        for (int i = 0; i < QK; i++) {
            int q = (int)(xb[i] / scale_x + (xb[i] >= 0 ? 0.5f : -0.5f));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            xq[i] = (int8_t)q;
#if FENG_GEMV_PIE
            s_x16[(size_t)b * QK + i] = (int16_t)q;
#endif
        }
    }
    s_a8_x = x;
    s_a8_nin = n_in;
}
#endif


static void q4_lut_init(void)
{
    if (s_q4_lut_ready) return;
    for (int b = 0; b < 256; b++) {
        s_q4_lut[b][0] = (float)((b & 0x0F) - 8);
        s_q4_lut[b][1] = (float)((b >> 4) - 8);
    }
    s_q4_lut_ready = 1;
}

void feng_quant_init(void)
{
    q4_lut_init();
#if FENG_GEMV_PIE
    q4_pair32_init();
#endif
}

static inline float f16_to_f32(uint16_t h)
{
    /* IEEE-754 half -> float */
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

float feng_f16_to_f32(uint16_t h)
{
    return f16_to_f32(h);
}

uint16_t feng_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 31) & 1u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127;
    uint32_t man = x & 0x7FFFFFu;
    if (exp > 15) {                       /* overflow -> +-inf */
        return (uint16_t)((sign << 15) | 0x7C00u);
    }
    if (exp < -24) return (uint16_t)(sign << 15);
    if (exp < -14) {                      /* subnormal */
        const uint32_t shift = (uint32_t)(-14 - exp);
        man = (man | 0x800000u) >> (shift + 13);
        return (uint16_t)((sign << 15) | (man & 0x3FFu));
    }
    return (uint16_t)((sign << 15) | ((uint32_t)(exp + 15) << 10) | (man >> 13));
}

/* 真正的 CPU 实现（不做 CUDA dispatch）；feng_gemv_range / feng_gemv3 共用。
 * FENG_HOT 必须留在这里：板端它是 IRAM_ATTR，循环体在 IRAM 执行；
 * 只标外层 wrapper 会让热循环退回 flash 取指（实测 +2.5% ms/token）。 */
FENG_HOT static void gemv_range_cpu(const void *tensor, uint32_t dtype, const float *x, float *y,
                                    int r0, int r1, int n_in)
{
#if defined(FENG_USE_OMP)
    /* 默认线程数：OMP_NUM_THREADS 未显式设置时取逻辑核的 3/4。实测 i7-12700KF
     * （20 逻辑核）默认全开会因 SMT/E 核拖累掉速（475 vs 664 tok/s）；用户可用
     * OMP_NUM_THREADS 覆盖。 */
    static int s_omp_ready = 0;         /* POD 静态，无运行期构造 */
    if (!s_omp_ready) {
        s_omp_ready = 1;
        if (getenv("OMP_NUM_THREADS") == NULL) {
            const int procs = omp_get_num_procs();
            const int cap = procs > 1 ? (procs * 3) / 4 : 1;
            if (cap < omp_get_max_threads()) omp_set_num_threads(cap);
        }
    }
#endif
    const uint8_t *base = (const uint8_t *)tensor;
    if (r1 <= r0 || n_in <= 0) return;
    if (dtype == FENG_DT_FP16) {
        const uint16_t *w = (const uint16_t *)tensor;
#if FENG_PHONE_NEON
        /* aarch64 且有 FP16 向量运算：vcvt_f32_f16 + vfmaq_f32。
         * 不支持的老核心、或 --neon 0，都自动落到下面的标量循环。
         * 两条路径**不逐位一致**（累加顺序不同），验收看 argmax 与 32 题判定。 */
        if (feng_phone_neon_on() && feng_neon_fp16_ok()) {
            feng_neon_fp16_gemv((const uint16_t *)tensor, x, y, r0, r1, n_in);
            return;
        }
#endif
#if defined(FENG_USE_OMP)
    /* MSVC（OpenMP 2.0 / vcomp）不允许在 for 语句里声明循环变量（C3015），
     * 所以变量提到外面声明；GCC/Clang 同样接受。 */
    int o;
#pragma omp parallel for schedule(static) if (r1 - r0 >= 128)
    for (o = r0; o < r1; o++) {
#else
    for (int o = r0; o < r1; o++) {
#endif
            const uint16_t *row = w + (size_t)o * n_in;
            float acc = 0.f;
            for (int i = 0; i < n_in; i++) {
                acc += f16_to_f32(row[i]) * x[i];
            }
            y[o] = acc;
        }
        return;
    }
#if FENG_FP16_WEIGHTS
    /* 未知 dtype：加载期（feng_model_init）已经拦掉了，这里只是防御。
     * **不要**像以前那样默认按 Q4 解——那会把 fp16 矩阵当 4-bit 打包读，不报错、
     * 结果全错。宁可输出全零（明显是坏的），也不要静默算出一个像样的错答案。
     * 板端 FENG_FP16_WEIGHTS=0 时这段不参与编译，代码与改造前逐 token 相同。 */
    if (dtype != FENG_DT_Q4) {
        for (int o = r0; o < r1; o++) y[o] = 0.0f;
        return;
    }
#endif
    if ((n_in % QK) != 0) {
        /* 形状不对时宁可全零（明显是坏的），也不要按截断的 n_blocks 静默算错。 */
        for (int o = r0; o < r1; o++) y[o] = 0.0f;
        return;
    }
    const int n_blocks = n_in / QK;
    const size_t row_bytes = (size_t)n_blocks * (2 + 32);
    q4_lut_init();
#if FENG_PHONE_NEON
    /* 手机端 NEON 内核（ARMv8.2 dotprod）：激活在 feng_gemv_a8_prepare 里量化成 int8，
     * 权重的 4-bit nibble 用 vqtbl1q_s8 映射成 int8，一条 vdotq_s32 做 16 次乘加。
     * 累加式与下面的标量 A8 路径**完全相同**（acc += (float)dot * (sw*sx)，同块顺序），
     * 整数点积又是精确的，所以两条路径结果逐位一致——可以在 x86 上用标量 A8 先验数值，
     * 再用 aarch64 跑同一份套件确认。没有 dotprod 的老核心（A53/A72）自动回退标量。 */
    if (feng_phone_neon_on() && feng_neon_sdot_ok() &&
        s_a8_x == x && s_a8_nin == n_in) {
        feng_neon_q4_gemv(base, s_a8_xq, s_a8_xscale, y, r0, r1, n_in);
        return;
    }
#endif
#if FENG_GEMV_PIE
    /* PIE 内核：激活 int16（prepare 阶段已算好）+ 权重 LUT 展开 + vmulas.s16。
     * 整数点积与 A8 路径完全同源（同样的 xq/scale），结果应逐位一致。 */
    if (s_a8_x == x && s_a8_nin == n_in) {
        q4_pair32_init();
        for (int o = r0; o < r1; o++) {
            const uint8_t *row = base + (size_t)o * row_bytes;
            const uint8_t *scales = row;
            const uint8_t *packed = row + (size_t)n_blocks * 2;
            float acc = 0.f;
            for (int b = 0; b < n_blocks; b++) {
                const float sx = s_a8_xscale[b];
                if (sx == 0.f) continue;
                uint16_t hs;
                memcpy(&hs, scales + b * 2, 2);
                const float sw = f16_to_f32(hs);
                const uint8_t *p = packed + (size_t)b * 32;
                alignas(16) int16_t w16[QK];
                uint32_t *st = (uint32_t *)w16;
                for (int j = 0; j < 32; j++) st[j] = s_q4_pair32[p[j]];
                const int32_t dot = pie_dot64(w16, s_x16 + (size_t)b * QK);
                acc += (float)dot * (sw * sx);
            }
            y[o] = acc;
        }
        return;
    }
#endif
#if FENG_GEMV_A8
    /* A8 模式：激活按 64 值一块量化成 int8（scale_x = max|x|/127），
     * 权重仍是 Q4（-8..7），点积走整数，再乘 scale_w*scale_x。
     * 量化原本每个输出行都重做一遍（O(n_out×n_in) 的额外开销），现在改成
     * 每个 GEMV 调用一次（feng_gemv_a8_prepare）——数值与逐行版逐位一致。 */
    const int a8_prepared = (s_a8_x == x && s_a8_nin == n_in);
    int8_t xq_local[QK];
    for (int o = r0; o < r1; o++) {
        const uint8_t *row = base + (size_t)o * row_bytes;
        const uint8_t *scales = row;
        const uint8_t *packed = row + (size_t)n_blocks * 2;
        float acc = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale_w = f16_to_f32(hs);
            const int8_t *xq;
            float scale_x;
            if (a8_prepared) {
                scale_x = s_a8_xscale[b];
                if (scale_x == 0.f) continue;
                xq = s_a8_xq + (size_t)b * QK;
            } else {
                const float *xb = x + (size_t)b * QK;
                float amax = 0.f;
                for (int i = 0; i < QK; i++) {
                    const float v = xb[i] < 0 ? -xb[i] : xb[i];
                    if (v > amax) amax = v;
                }
                if (amax < 1e-8f) continue;
                scale_x = amax / 127.f;
                for (int i = 0; i < QK; i++) {
                    int q = (int)(xb[i] / scale_x + (xb[i] >= 0 ? 0.5f : -0.5f));
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    xq_local[i] = (int8_t)q;
                }
                xq = xq_local;
            }
            int32_t dot = 0;
            const uint8_t *p = packed + (size_t)b * 32;
            for (int j = 0; j < 32; j++) {
                const int lo = (int)(p[j] & 0x0F) - 8;
                const int hi = (int)(p[j] >> 4) - 8;
                dot += lo * (int)xq[2 * j] + hi * (int)xq[2 * j + 1];
            }
            acc += (float)dot * (scale_w * scale_x);
        }
        y[o] = acc;
    }
    return;
#endif
#if defined(FENG_USE_OMP)
    int o;                          /* 见上面 fp16 分支：MSVC 的 OpenMP 2.0 限制 */
#pragma omp parallel for schedule(static) if (r1 - r0 >= 128)
    for (o = r0; o < r1; o++) {
#else
    for (int o = r0; o < r1; o++) {
#endif
        const uint8_t *row = base + (size_t)o * row_bytes;
        const uint8_t *scales = row;
        const uint8_t *packed = row + (size_t)n_blocks * 2;
        /* several independent accumulators: a single one serialises the FPU pipeline
         * on the dependency chain (add latency x 32 madds per block) */
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale = f16_to_f32(hs);
            const uint8_t *p = packed + (size_t)b * 32;
            const float *xb = x + (size_t)b * QK;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                const uint8_t b0 = p[j], b1 = p[j + 1], b2 = p[j + 2], b3 = p[j + 3];
#if FENG_GEMV_MADD
                /* 纯 madd 链：每 2 个权重 2 个 FP 运算（对比下面的 3 个） */
                s0 += s_q4_lut[b0][0] * xb[2 * j];
                s0 += s_q4_lut[b0][1] * xb[2 * j + 1];
                s1 += s_q4_lut[b1][0] * xb[2 * j + 2];
                s1 += s_q4_lut[b1][1] * xb[2 * j + 3];
                s2 += s_q4_lut[b2][0] * xb[2 * j + 4];
                s2 += s_q4_lut[b2][1] * xb[2 * j + 5];
                s3 += s_q4_lut[b3][0] * xb[2 * j + 6];
                s3 += s_q4_lut[b3][1] * xb[2 * j + 7];
#else
                s0 += s_q4_lut[b0][0] * xb[2 * j] + s_q4_lut[b0][1] * xb[2 * j + 1];
                s1 += s_q4_lut[b1][0] * xb[2 * j + 2] + s_q4_lut[b1][1] * xb[2 * j + 3];
                s2 += s_q4_lut[b2][0] * xb[2 * j + 4] + s_q4_lut[b2][1] * xb[2 * j + 5];
                s3 += s_q4_lut[b3][0] * xb[2 * j + 6] + s_q4_lut[b3][1] * xb[2 * j + 7];
#endif
            }
            a0 += s0 * scale;
            a1 += s1 * scale;
            a2 += s2 * scale;
            a3 += s3 * scale;
        }
        y[o] = (a0 + a1) + (a2 + a3);
    }
}

FENG_HOT void feng_gemv_range(const void *tensor, uint32_t dtype, const float *x, float *y,
                     int r0, int r1, int n_in)
{
#if defined(FENG_USE_CUDA)
    /* PC CUDA 后端：可用时 GEMV 全走 GPU；kernel 与 CPU 路径使用相同的浮点
     * 运算顺序，数值一致（见 feng_cuda.cu 的文件头说明）。 */
    if (feng_cuda_available()) {
        feng_cuda_gemv_range(tensor, dtype, x, y, r0, r1, n_in);
        return;
    }
#endif
    gemv_range_cpu(tensor, dtype, x, y, r0, r1, n_in);
}

void feng_gemv3(const void *w0, const void *w1, const void *w2, uint32_t dtype,
                const float *x, float *y0, float *y1, float *y2,
                int n_out0, int n_out1, int n_out2, int n_in)
{
#if defined(FENG_USE_CUDA)
    if (feng_cuda_available()) {
        feng_cuda_gemv3(w0, w1, w2, dtype, x, y0, y1, y2, n_out0, n_out1, n_out2, n_in);
        return;
    }
#endif
    /* 板端 / 手机：逐次 feng_gemv_par（各自的多核实现）；PC：直接走 CPU 实现，
     * 避免依赖 feng_smp（pc_sample_test 这类只链 feng_quant 的单测不需要 feng_smp）。 */
#if defined(ESP_PLATFORM) || defined(FENG_PHONE)
    if (w0) feng_gemv_par(w0, dtype, x, y0, n_out0, n_in);
    if (w1) feng_gemv_par(w1, dtype, x, y1, n_out1, n_in);
    if (w2) feng_gemv_par(w2, dtype, x, y2, n_out2, n_in);
#else
    if (w0) gemv_range_cpu(w0, dtype, x, y0, 0, n_out0, n_in);
    if (w1) gemv_range_cpu(w1, dtype, x, y1, 0, n_out1, n_in);
    if (w2) gemv_range_cpu(w2, dtype, x, y2, 0, n_out2, n_in);
#endif
}

void feng_gemv(const void *tensor, uint32_t dtype, const float *x, float *y,
               int n_out, int n_in, float *scratch)
{
    (void)scratch;
    feng_gemv_range(tensor, dtype, x, y, 0, n_out, n_in);
}

int feng_argmax(const float *logits, int n)
{
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    }
    return best;
}
