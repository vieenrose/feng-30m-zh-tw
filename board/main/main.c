/* feng-30m chat on ESP32-S3 (N32R16: 32MB flash + 16MB octal PSRAM)
 *
 * Partition layout (see partitions.csv):
 *   model    model.bin     (Q4 weights, memory-mapped, executed straight from flash)
 *   tokdata  tokenizer.bin (vocab + merges; loaded into PSRAM)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>   /* C 里 va_start/va_end 必须显式包含（C++ 是由 <cstdio> 间接带进来的） */
#include <string.h>

#include "esp_chip_info.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "feng.h"
#include "feng_bonus.h"
#include "feng_calc.h"
#include "feng_memory.h"
#include "feng_tools.h"
#include "feng_tokenizer.h"
#include <esp_timer.h>
#include "gbk.h"

/* 构建期断言：IDF 会把目录级 `-std=gnu17` 排在组件选项之前，若哪天顺序又变回
 * gnu17，这里会直接编译失败，而不是悄悄退化成 C17（见 main/CMakeLists.txt）。
 * 语言级别仍是 C23（-std=gnu23 只是带 GNU 扩展）。
 * 阈值用 202000L 而不是 202311L：IDF 的 xtensa gcc 14.2 对 -std=gnu23 报的是
 * 草案值 202000L（主机 gcc 16 报 202311L），两者都 >= C2x。 */
_Static_assert(__STDC_VERSION__ >= 202000L, "firmware must build as C23 (see main/CMakeLists.txt)");

static const char *TAG = "feng";
#define MODEL_PART "model"
#define TOK_PART "tokdata"
#define UART_PORT UART_NUM_0

/* ---- 记忆持久化（NVS "mem" 键）----
 * 引擎侧记忆原来只在 RAM，\reset/断电就忘。这里用 1KB 文本快照存 NVS，
 * 开机导入；每次学到/忘掉事实后保存。NVS 分区只有 24KB，键值远小于上限。 */
static nvs_handle_t s_mem_nvs;
static int s_mem_nvs_ok;

static void mem_store_init(void)
{
    esp_err_t e = nvs_flash_init();
    if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        (void)nvs_flash_erase();
        e = nvs_flash_init();
    }
    if (e == ESP_OK && nvs_open("feng", NVS_READWRITE, &s_mem_nvs) == ESP_OK)
        s_mem_nvs_ok = 1;
    else
        ESP_LOGW(TAG, "NVS 打不开，记忆只在本次上电有效");
}

static void mem_store_load(void)
{
    if (!s_mem_nvs_ok) return;
    char buf[1024];
    size_t len = sizeof(buf);
    if (nvs_get_str(s_mem_nvs, "mem", buf, &len) == ESP_OK) {
        (void)feng_mem_import(buf);
        ESP_LOGI(TAG, "从 NVS 恢复记忆（%u 字节）", (unsigned)len);
    }
}

static void mem_store_save(void)
{
    if (!s_mem_nvs_ok) return;
    char buf[1024];
    const int n = feng_mem_export(buf, sizeof(buf));
    if (n < 0) return;
    if (n == 0) (void)nvs_erase_key(s_mem_nvs, "mem");
    else (void)nvs_set_str(s_mem_nvs, "mem", buf);
    (void)nvs_commit(s_mem_nvs);
}

static int line_may_forget(const char *s)
{
    return strstr(s, "忘掉") || strstr(s, "忘记") || strstr(s, "别记") ||
           strstr(s, "删掉") || strstr(s, "不要记");
}
/* KV cache is fp32 in PSRAM: n_layers*MAX_CTX*hidden*4 bytes per cache (K and V).
 * 11*512*448*4 = 10.1 MB each -> 20 MB total, more than the 16 MB PSRAM.
 * 11*256*448*4 =  5.05 MB each -> 10.1 MB total, fits comfortably. */
/* int8 KV cache is 4x smaller, which is what makes a 1024-token window fit the
 * 16 MB of PSRAM (fp32 only fits ~256). */
/* 测试用覆盖：-DFENG_MAX_CTX_OVERRIDE=256 可把上下文缩到很小，
 * 快速验证"写满自动开新对话"这条路径（正常固件不定义，保持下面的默认值）。 */
#ifndef FENG_MAX_CTX_OVERRIDE
#define FENG_MAX_CTX_OVERRIDE 0
#endif
#if FENG_MAX_CTX_OVERRIDE
#define MAX_CTX FENG_MAX_CTX_OVERRIDE
#elif FENG_KV_Q2
#define MAX_CTX 2048        /* q2 block8：448 B/token/层 -> 2048 ctx ≈ 10.1 MB PSRAM */
#elif FENG_KV_INT8
#define MAX_CTX 1024
#else
#define MAX_CTX 256
#endif
#define MAX_NEW 96

/* 注意力基准（-DFENG_BENCH_CTX=1 打开）：合成指定长度的 KV，测单次 forward 耗时，
 * 用来评估 KV 访问/注意力优化在长上下文下的收益。 */
#ifndef FENG_BENCH_GEMV
#define FENG_BENCH_GEMV 0
#endif
#ifndef FENG_BENCH_CTX
#define FENG_BENCH_CTX 0
#endif
/* 串口 GBK 转码：默认不编（省 46 KB 映射表），实现见 gbk.c 的 #if FENG_GBK。
 * 由 main/CMakeLists.txt 的 FENG_GBK 选项控制（默认 OFF，需要就 -DFENG_GBK=ON）。 */
#ifndef FENG_GBK
#define FENG_GBK 0
#endif

static feng_model_t s_model;
static feng_tok_t s_tok;
static feng_kv_t s_kv;
static feng_workspace_t s_ws;
static int s_hist[64];          /* recent tokens for repetition penalty */
static int s_nhist;
/* 网络时间：宿主发 \settime <epoch> 同步一次，之后用 esp_timer 走时 */
static long long s_epoch_base;
static long long s_time_base_us;
static int s_time_synced;

static long long fw_epoch_now(void)
{
    if (!s_time_synced) return 0;
    return s_epoch_base + (esp_timer_get_time() - s_time_base_us) / 1000000;
}

static long long fw_uptime_us(void)
{
    return esp_timer_get_time();
}

#if FENG_BENCH_CTX && FENG_KV_Q2
/* PIE 裸吞吐：判断"整行 PIE 内核"的理论上限（对比标量 GEMV 的 4.1 周期/权重，双核） */
/* 合成指定长度的 q2 KV，测单次 forward 的完整耗时（含权重 GEMV + 注意力）。
 * 每个长度跑两次：第 1 次是冷缓存，第 2 次是热缓存；对比不同内核版本用第 2 次。 */
static void bench_forward_ctx(void)
{
#if FENG_ATTN_PROF
    extern unsigned long long g_prof_cycles[3];
#endif
    const int h = s_model.hdr.hidden;
    const int n_layers = s_model.hdr.n_layers;
    const int nh = s_model.hdr.n_heads, hd = s_model.hdr.head_dim;
    const int qb = hd / 4, nb = hd / FENG_KV_Q2_BLOCK;
    const int ctxs[] = {256, 1024, 2048};
    const uint16_t one = feng_f32_to_f16(1.0f);
    for (int ci = 0; ci < (int)(sizeof(ctxs) / sizeof(ctxs[0])); ci++) {
        const int c = ctxs[ci];
        if (c > s_kv.ctx) break;
        for (int l = 0; l < n_layers; l++) {
            /* q2 布局 [layer][head][t]：逐 head 填该头的 c 个 token */
            for (int hh = 0; hh < nh; hh++) {
                const size_t row = ((size_t)l * nh + hh) * s_kv.ctx;
                uint8_t *kc = (uint8_t *)s_kv.k_cache + row * qb;
                uint8_t *vc = (uint8_t *)s_kv.v_cache + row * qb;
                uint16_t *ks = s_kv.k_scale + row * nb;
                uint16_t *vs = s_kv.v_scale + row * nb;
                for (size_t i = 0; i < (size_t)c * qb; i++) {
                    kc[i] = (uint8_t)(i * 37 + 11);
                    vc[i] = (uint8_t)(i * 53 + 7);
                }
                for (size_t i = 0; i < (size_t)c * nb; i++) { ks[i] = one; vs[i] = one; }
            }
        }
        int64_t dt[2] = {0, 0};
        for (int rep = 0; rep < 2; rep++) {
#if FENG_ATTN_PROF
            if (rep == 1) {
                g_prof_cycles[0] = g_prof_cycles[1] = g_prof_cycles[2] = 0;
            }
#endif
            const int64_t t0 = esp_timer_get_time();
            feng_forward(&s_model, &s_kv, &s_ws, 100, c - 1);
            dt[rep] = esp_timer_get_time() - t0;
        }
        ESP_LOGI(TAG, "attn bench ctx=%4d: cold %.0f ms/forward | warm %.0f ms/forward",
                 c, dt[0] / 1000.0, dt[1] / 1000.0);
#if FENG_ATTN_PROF
        ESP_LOGI(TAG, "  prof ctx=%4d: K=%.0f ms  softmax=%.0f ms  V=%.0f ms",
                 c, g_prof_cycles[0] / 240000.0, g_prof_cycles[1] / 240000.0,
                 g_prof_cycles[2] / 240000.0);
#endif
    }
    /* lm head（out_norm + tied head，7.34M 权重）的单次成本：pos=0 带/不带 logits 各一次 */
    int64_t h_with = 0, h_without = 0;
    s_kv.len = 0;
    {
        const int64_t t0 = esp_timer_get_time();
        feng_forward(&s_model, &s_kv, &s_ws, 100, 0);
        h_with = esp_timer_get_time() - t0;
    }
    s_kv.len = 0;
    {
        const int64_t t0 = esp_timer_get_time();
        feng_forward_ex(&s_model, &s_kv, &s_ws, 100, 0, 0);
        h_without = esp_timer_get_time() - t0;
    }
    ESP_LOGI(TAG, "lm head @pos0: with %.0f ms | without %.0f ms (prefill 每个中间 token 省 %.0f ms)",
             h_with / 1000.0, h_without / 1000.0, (h_with - h_without) / 1000.0);
    s_kv.len = 0;
}
#endif

#if FENG_BENCH_PIE
/* PIE 裸吞吐：判断"整行 PIE 内核"的理论上限（对比标量 GEMV 的 ~4 周期/权重，双核） */
void pie_mac_s16_bench(const int16_t *a, int n);
void pie_s8_zip_mac_bench(const int8_t *w, const int8_t *x, int n);
static void bench_pie(void)
{
    static int16_t b16[16];
    static int8_t b8[32];
    const int iters = 300000;
    for (int i = 0; i < 16; i++) b16[i] = (int16_t)(i + 1);
    for (int i = 0; i < 32; i++) b8[i] = (int8_t)(i - 8);
    const unsigned c0 = (unsigned)esp_cpu_get_cycle_count();
    pie_mac_s16_bench(b16, iters);
    const unsigned c1 = (unsigned)esp_cpu_get_cycle_count();
    pie_s8_zip_mac_bench(b8, b8 + 16, iters);
    const unsigned c2 = (unsigned)esp_cpu_get_cycle_count();
    ESP_LOGI(TAG, "PIE bench: s16 MAC 链 %.2f 周期/条 (%.3f 周期/MAC) | s8 流水 %.2f 周期/迭代 (%.3f 周期/MAC, 8MAC/迭代)",
             (double)(c1 - c0) / iters, (double)(c1 - c0) / iters / 8.0,
             (double)(c2 - c1) / iters, (double)(c2 - c1) / iters / 8.0);
}
#endif

/* ---- serial I/O: this firmware owns UART0 (console is disabled in sdkconfig) ---- */
/* write UTF-8 text out in whatever encoding the terminal speaks (GBK or UTF-8) */
static void write_text(const char *s, int n)
{
    if (n <= 0) return;
    if (g_out_gbk) {
        static char gb[2048];
        const int m = feng_utf8_to_gbk(s, n, gb, sizeof(gb));
        if (m > 0) uart_write_bytes(UART_PORT, gb, (size_t)m);
    } else {
        uart_write_bytes(UART_PORT, s, (size_t)n);
    }
}

static int uart_vprintf_impl(const char *fmt, va_list ap)
{
    static char buf[512];
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    if (n > 0) {
        write_text(buf, n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1);
    }
    return n;
}

static void out_printf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    uart_vprintf_impl(fmt, ap);
    va_end(ap);
}

static void *ps_alloc(size_t n)
{
    void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) {
        ESP_LOGE(TAG, "PSRAM alloc failed: %u bytes", (unsigned)n);
        abort();
    }
    return p;
}

/* GEMV 的输入向量放**内部 SRAM**，不放 PSRAM。
 *
 * 原因（2026-10-15 实测定位）：GEMV 内层每 8 个权重就要发 8 次浮点 load 读激活，
 * 而权重是从 flash 流式读的 15 MB，会把 64 KB 的 L1D 反复冲干净——激活向量因此
 * 频繁被逐出、退化成 PSRAM 访问。这些向量很小（xn 448 / ffn 896 / attn 448 个 float
 * 合计不到 9 KB），内部 SRAM 有 232 KB 空闲，放进去就没有缓存冲刷问题了。
 * 分配失败时回退 PSRAM（只是慢，不会错）。 */
static void *iram_alloc(size_t n)
{
    void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) {
        ESP_LOGE(TAG, "alloc failed: %u bytes", (unsigned)n);
        abort();
    }
    return p;
}

static float *iram_alloc_f32(size_t n)
{
    return (float *)iram_alloc(n * sizeof(float));
}

static float *ps_alloc_f32(size_t n)
{
    float *p = (float *)ps_alloc(n * sizeof(float));
    memset(p, 0, n * sizeof(float));
    return p;
}

static void log_mem(const char *stage)
{
    ESP_LOGI(TAG, "[mem] %-11s internal %7u B | psram %8u B (largest int %u)", stage,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
}

static void bench_gemv(void);

static void setup_model(void)
{
    const esp_partition_t *mp = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                                        ESP_PARTITION_SUBTYPE_ANY, MODEL_PART);
    if (!mp) {
        ESP_LOGE(TAG, "model partition missing");
        abort();
    }
    const void *ptr = NULL;
    esp_partition_mmap_handle_t h;
    ESP_ERROR_CHECK(esp_partition_mmap(mp, 0, mp->size, ESP_PARTITION_MMAP_DATA, &ptr, &h));
    ESP_LOGI(TAG, "model mapped at %p (%u KB)", ptr, (unsigned)(mp->size / 1024));
    log_mem("mmap");

    /* quick streaming-read benchmark: how fast can we pull the weights out of flash? */
    {
        const uint8_t *p8 = (const uint8_t *)ptr;
        const size_t n = mp->size & ~(size_t)0x3F;
        volatile uint32_t acc = 0;
        const int64_t t0 = esp_timer_get_time();
        for (size_t i = 0; i < n; i += 64) acc += p8[i];
        const int64_t dt = esp_timer_get_time() - t0;
        ESP_LOGI(TAG, "mmap stream read: %u KB in %lld ms -> %.1f MB/s (checksum %u)",
                 (unsigned)(n / 1024), dt / 1000, (double)n / (double)dt * 1.0,
                 (unsigned)acc);
    }
    const int rc = feng_model_init(&s_model, ptr, (size_t)mp->size);
    if (rc != 0) {
        ESP_LOGE(TAG, "model init failed rc=%d", rc);
        abort();
    }
    log_mem("model");
    ESP_LOGI(TAG, "model: %u layers, hidden %u, %u heads x %u, ffn %u, vocab %u, rope %.0f",
             s_model.hdr.n_layers, s_model.hdr.hidden, s_model.hdr.n_heads, s_model.hdr.head_dim,
             s_model.hdr.ffn, s_model.hdr.vocab, s_model.hdr.rope_theta);

    const esp_partition_t *tp = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                                        ESP_PARTITION_SUBTYPE_ANY, TOK_PART);
    if (!tp) {
        ESP_LOGE(TAG, "tokenizer partition missing");
        abort();
    }
    uint8_t *tokbuf = (uint8_t *)ps_alloc(tp->size);
    ESP_ERROR_CHECK(esp_partition_read(tp, 0, tokbuf, tp->size));
    if (feng_tok_load(&s_tok, tokbuf, tp->size) != 0) {
        ESP_LOGE(TAG, "tokenizer load failed");
        abort();
    }
    /* feng_tok_load 会把 token 字符串复制到自己的 PSRAM blob；这块 512 KB
     * 的原始分区缓冲之后不再需要。旧实现一直没释放，等于永久吃掉 0.5 MB PSRAM。 */
    free(tokbuf);
    tokbuf = NULL;
    ESP_LOGI(TAG, "tokenizer: vocab %d, merges %d", s_tok.vocab_size, s_tok.n_merges);
    log_mem("tokenizer");
    /* boot self-test: tokenize a short string so tokenizer problems show up immediately */
    {
        static int test_ids[64];
        const int tn = feng_tok_encode(&s_tok, "你好", test_ids, 64);
        char dump[256];
        int off = 0;
        for (int i = 0; i < tn && off < 200; i++) {
            off += snprintf(dump + off, sizeof(dump) - off, "%d ", test_ids[i]);
        }
        dump[off] = 0;
        ESP_LOGI(TAG, "tokenizer self-test: %d ids -> %s", tn, dump);
        int back = 0;
        char text[128];
        int toff = 0;
        for (int i = 0; i < tn && toff < 120; i++) {
            toff += feng_tok_decode_token(&s_tok, test_ids[i], text + toff, 8);
        }
        text[toff] = 0;
        /* keep the boot log ASCII-only so any terminal encoding shows it correctly */
        ESP_LOGI(TAG, "tokenizer decode round-trip ok=%d (%d bytes)", strcmp(text, "你好") == 0, toff);
        back = toff;
        (void)back;
    }

    const int hidden = s_model.hdr.hidden;
    s_kv.ctx = MAX_CTX;
    s_kv.len = 0;
#if FENG_KV_INT8
    s_kv.k_cache = ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
    s_kv.v_cache = ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
#elif FENG_KV_Q2
    s_kv.k_cache = ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * (hidden / 4));
    s_kv.v_cache = ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * (hidden / 4));
#else
    s_kv.k_cache = ps_alloc_f32((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
    s_kv.v_cache = ps_alloc_f32((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
#endif
#if FENG_KV_INT8 || FENG_KV_Q2
    const size_t n_sc = feng_kv_scale_slots(&s_model, MAX_CTX);
    s_kv.k_scale = (uint16_t *)ps_alloc(n_sc * sizeof(uint16_t));
    s_kv.v_scale = (uint16_t *)ps_alloc(n_sc * sizeof(uint16_t));
#else
    s_kv.k_scale = NULL;
    s_kv.v_scale = NULL;
#endif
    s_ws.max_ctx = MAX_CTX;
    /* 这四个是 GEMV 的**输入**向量（每个权重都要读一遍），放内部 SRAM 避开缓存冲刷；
     * 其余是 GEMV 的输出/中间量（每 token 只写一遍读一遍），留 PSRAM 省内部 RAM。 */
    s_ws.x = iram_alloc_f32(hidden);
    s_ws.xn = iram_alloc_f32(hidden);
    s_ws.attn = iram_alloc_f32(hidden);
    s_ws.ffn = iram_alloc_f32(s_model.hdr.ffn);
    s_ws.q = ps_alloc_f32(hidden);
    s_ws.k = ps_alloc_f32(hidden);
    s_ws.v = ps_alloc_f32(hidden);
    /* proj 是 attn/down 投影的输出，宽度是 hidden，不是 ffn（旧值在 ff<hidden 的
     * 模型上会堆越界；当前 ff=896 > h=448 只是恰好没踩到）。 */
    s_ws.proj = ps_alloc_f32(hidden);
    s_ws.gate = ps_alloc_f32(s_model.hdr.ffn);
    s_ws.up = ps_alloc_f32(s_model.hdr.ffn);
    s_ws.logits = ps_alloc_f32(s_model.hdr.vocab);
    s_ws.scratch = ps_alloc_f32(s_model.hdr.ffn > MAX_CTX ? s_model.hdr.ffn : MAX_CTX);
#if FENG_KV_Q2
    const char *kv_mode = "q2/block" FENG_STR(FENG_KV_Q2_BLOCK);
#elif FENG_KV_INT8
    const char *kv_mode = "int8";
#else
    const char *kv_mode = "fp32";
#endif
    ESP_LOGI(TAG, "KV cache: %s, ctx %d, %.2f MB", kv_mode, MAX_CTX,
             (double)feng_kv_bytes(&s_model, MAX_CTX) / 1048576.0);
    ESP_LOGI(TAG, "PSRAM free after setup: %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    log_mem("kv/ws");
    feng_smp_init();
    bench_gemv();
}

/* measure the raw GEMV throughput (single core vs both cores) once at boot */
static void bench_gemv(void)
{
    const int h = s_model.hdr.hidden, f = s_model.hdr.ffn;
    const feng_layer_t *L = &s_model.layers[0];
    const size_t wbytes = (size_t)f * (h / 64) * 34;   /* Q4 bytes read per call */
    const long iters = 20;
    static float xref[512];                            /* 未 prepare 的副本 -> 回退内核 */
    /* 激活缓冲在开机时是未初始化的；填成非零、有正有负的样值，
     * 否则"全零块跳过"会让基准测不到真正的权重解码 + 点积开销。 */
    for (int i = 0; i < h; i++) {
        s_ws.xn[i] = (float)((i % 7) - 3) * 0.13f;
        xref[i] = s_ws.xn[i];
    }
    int64_t t0 = esp_timer_get_time();
    for (long i = 0; i < iters; i++) {
        feng_gemv_range(L->gate, FENG_DT_Q4, s_ws.xn, s_ws.gate, 0, f, h);
    }
    int64_t dt1 = esp_timer_get_time() - t0;
#if FENG_GEMV_A8 || FENG_GEMV_PIE
    /* 准备激活 -> 启用 A8/PIE 内核 */
    feng_gemv_a8_prepare(s_ws.xn, h);
#if FENG_GEMV_PIE
    {
        extern int feng_gemv_pie_selfcheck(const void *tensor, int n_in);
        const int chk = feng_gemv_pie_selfcheck(L->gate, h);
        ESP_LOGI(TAG, "PIE dot self-check: %s",
                 chk == 0 ? "MATCH" : (chk == 1 ? "MISMATCH" : "SKIP(no prepare)"));
    }
#endif
    t0 = esp_timer_get_time();
    for (long i = 0; i < iters; i++) {
        feng_gemv_range(L->gate, FENG_DT_Q4, s_ws.xn, s_ws.gate, 0, f, h);
    }
    int64_t dt1b = esp_timer_get_time() - t0;
#else
    [[maybe_unused]] const int64_t dt1b = dt1;   /* 仅量化内核 A/B 对比时需要 */
#endif
    (void)xref;
    t0 = esp_timer_get_time();
    for (long i = 0; i < iters; i++) {
        feng_gemv_par(L->gate, FENG_DT_Q4, s_ws.xn, s_ws.gate, f, h);
    }
    int64_t dt2 = esp_timer_get_time() - t0;
    ESP_LOGI(TAG, "gemv %dx%d: 1-core %lld us (%.0f MB/s) | 2-core %lld us (%.0f MB/s) | speedup %.2fx",
             f, h, dt1 / iters, (double)wbytes * iters / (double)dt1,
             dt2 / iters, (double)wbytes * iters / (double)dt2, (double)dt1 / (double)dt2);
#if FENG_GEMV_A8 || FENG_GEMV_PIE
    ESP_LOGI(TAG, "gemv 量化内核: 1-core %lld us（对比回退 %lld us，%.2fx）",
             dt1b / iters, dt1 / iters, (double)dt1 / (double)dt1b);
#endif
#if FENG_BENCH_GEMV
    /* 周期级剖析：把 7.52 周期/权重的停顿归因到具体来源（见 feng_gemv_bench.c 文件头） */
    {
        extern void feng_gemv_bench(const void *tensor, int n_out, int n_in, const float *x);
        feng_gemv_bench(L->gate, f, h, s_ws.xn);                  /* 一个 FFN 矩阵 */
        if (s_model.tok_embd_dtype == FENG_DT_Q4) {
            feng_gemv_bench(s_model.tok_embd, (int)s_model.hdr.vocab, h, s_ws.xn);
        }
    }
#endif
}

/* greedy sampling with repetition penalty over the recent window */
static int sample_next(float *logits, int vocab)
{
    /* 统一采样器：1.15 重复惩罚 + 禁止补全已出现过的 3-gram（压"太阳系太阳系…"循环） */
    return feng_sample_greedy(logits, vocab, s_hist, s_nhist, 1.15f, 3);
}

static void push_hist(int t)
{
    if (s_nhist < (int)(sizeof(s_hist) / sizeof(s_hist[0]))) {
        s_hist[s_nhist++] = t;
    } else {
        memmove(s_hist, s_hist + 1, sizeof(s_hist) - sizeof(s_hist[0]));
        s_hist[sizeof(s_hist) / sizeof(s_hist[0]) - 1] = t;
    }
}

/* Streaming granularity: flush at a clause boundary or once the chunk is big
 * enough, so a terminal that stamps every received packet shows a timestamp per
 * clause instead of one per token. */
#define STREAM_CHUNK 30
static int s_stream_chunk = STREAM_CHUNK;   /* 0 = buffer the whole reply, flush once */

static int ends_clause(const char *s, int n)
{
    if (n <= 0) return 0;
    const char last = s[n - 1];
    if (last == '\n' || last == '!' || last == '?' || last == '.') return 1;
    if (n < 3) return 0;
    static const char *marks[] = {"\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F", "\xEF\xBC\x8C",
                                  "\xEF\xBC\x9B", "\xEF\xBC\x9A", "\xE3\x80\x81"};
    for (int i = 0; i < (int)(sizeof(marks) / sizeof(marks[0])); i++) {
        if (memcmp(s + n - 3, marks[i], 3) == 0) return 1;
    }
    return 0;
}

/* 换行 token（多轮续接时要补在 <|im_end|> 之后），首次用到时编码一次 */
static int newline_token(void)
{
    static int cached = -1;
    if (cached < 0) {
        int one = 0;
        cached = (feng_tok_encode(&s_tok, "\n", &one, 1) == 1) ? one : 0;
    }
    return cached;
}

/* keep=1：不动 KV，从已有上下文后面续写（多轮对话）；keep=0：清空重来 */
static int generate(const int *prompt, int np, char *out, int out_max, int keep)
{
    if (!keep) {
        s_kv.len = 0;
        s_nhist = 0;
        feng_mem_clear();          /* 写满=开新对话：引擎侧记忆一起清 */
        mem_store_save();
    }
    int written = 0, pos = s_kv.len;
    const int vocab = s_model.hdr.vocab;
    const int64_t t0 = esp_timer_get_time();
    float *logits = NULL;
    ESP_LOGI(TAG, "prefill %d tokens ...", np);
    for (int i = 0; i < np; i++) {                      /* prefill（中间 token 不算 lm head） */
        logits = feng_forward_ex(&s_model, &s_kv, &s_ws, prompt[i], pos++, i + 1 == np);
        if (logits == NULL) { out_printf("[上下文或模型状态异常，本轮中止]\n"); return 0; }
    }
    ESP_LOGI(TAG, "prefill done");
    const int64_t t_prefill = esp_timer_get_time();
    int n_gen = 0;
    static char pend[1100];     /* 与 reply[1024] 对齐：\stream 0 时整段回复不能丢 */
    int pn = 0;
    out_printf("<< ");
    fflush(stdout);
    /* 预留 im_end + 换行两个位置；生成循环本身也不允许写到 ctx 之外。 */
    const int pos_limit = (MAX_CTX > 2) ? MAX_CTX - 2 : 0;
    for (int n = 0; n < MAX_NEW && written < out_max - 8 && pos < pos_limit; n++) {
        const int tok = sample_next(logits, vocab);
        if (tok < 0) break;
        if (tok == s_tok.id_im_end || tok == s_tok.id_eot) break;
        char tmp[128];
        const int nb = feng_tok_decode_token(&s_tok, tok, tmp, sizeof(tmp));
        if (nb > 0 && written + nb < out_max) {
            memcpy(out + written, tmp, nb);
            written += nb;
            out[written] = 0;
            if (pn + nb < (int)sizeof(pend)) {
                memcpy(pend + pn, tmp, nb);
                pn += nb;
            }
            if (s_stream_chunk > 0 && (pn >= s_stream_chunk || ends_clause(pend, pn))) {
                write_text(pend, pn);
                pn = 0;
            }
        }
        push_hist(tok);
        n_gen++;
        logits = feng_forward(&s_model, &s_kv, &s_ws, tok, pos++);
    }
    if (s_stream_chunk == 0) {
        /* 整段输出：直接用 reply 缓冲，避免 pend 在长回复时被截断。 */
        if (written > 0) write_text(out, written);
    } else if (pn > 0) {
        write_text(pend, pn);
    }
    const int64_t t_end = esp_timer_get_time();
    const int64_t dt_dec = t_end - t_prefill;
    /* tok/s 是**本轮 decode** 的吞吐：分子只能是本轮新生成的 token 数。
     * 旧实现用累计位置 pos（含历史上下文与 prefill），连续多轮会越报越高。 */
    const float tps = (n_gen > 0 && dt_dec > 0) ? (float)n_gen * 1e6f / (float)dt_dec : 0.0f;
    if (keep) {
        /* 把模型自己吐出的 <|im_end|> 与换行补进 KV，下一轮才能无缝续接；
           若本轮是被长度截断的，也补一个 im_end 收尾，语义上等于结束这一轮。
           这两个 token 的 logits 没人用，跳过 lm head（每个省 ~124 ms）。
           只在确实还有空间时写，避免 chat_once 的预算差一两个 token 时越界。 */
        if (pos < MAX_CTX)
            (void)feng_forward_ex(&s_model, &s_kv, &s_ws, s_tok.id_im_end, pos++, 0);
        const int nl = newline_token();
        if (nl > 0 && pos < MAX_CTX)
            (void)feng_forward_ex(&s_model, &s_kv, &s_ws, nl, pos++, 0);
    }
    s_kv.len = pos;
    /* 和 pc_chat 同一口径：尾行给出上下文占用与速度，方便用户/脚本判断 */
    out_printf("\n>>END (ctx %d/%d, %.2f tok/s)\n", s_kv.len, MAX_CTX, tps);
    fflush(stdout);
    ESP_LOGI(TAG, "prompt %d | gen %d | prefill %lld ms | decode %lld ms | %.2f tok/s (%.0f ms/token)",
             np, n_gen, (t_prefill - t0) / 1000, dt_dec / 1000, tps,
             n_gen > 0 ? (float)dt_dec / 1000.0f / (float)n_gen : 0.0f);
    return written;
}

static void chat_once(const char *user)
{
    /* 长文输入：正文按 token 预算安全截断（保 UTF-8 边界与模板收尾），
     * 2048 ctx 下大约能收 1500+ 个汉字，而不是原来的 ~340 个。 */
    static char prompt[4096];
    static int ids[2304];
    static char reply[1024];
    int cap = MAX_CTX - MAX_NEW - 8;
    /* 测试用 MAX_CTX_OVERRIDE 可以大于 2304；把 tokenizer 输出上限钉在数组容量内，
     * 否则 encode 会写穿 ids（`-DFENG_MAX_CTX_OVERRIDE=4096` 时实测可触发）。 */
    if (cap > (int)(sizeof(ids) / sizeof(ids[0])))
        cap = (int)(sizeof(ids) / sizeof(ids[0]));
    size_t ulen = strlen(user);
    if (ulen > sizeof(prompt) - 256) ulen = sizeof(prompt) - 256;
    int truncated = (ulen < strlen(user));
    int n = 0;
    for (;;) {
        while (ulen > 0 && ((unsigned char)user[ulen] & 0xC0) == 0x80) ulen--;
        snprintf(prompt, sizeof(prompt),
                 "<|im_start|>user\n%.*s<|im_end|>\n<|im_start|>assistant\n", (int)ulen, user);
        n = feng_tok_encode(&s_tok, prompt, ids, cap);
        if (n < cap) break;                    /* 收得下（含 <|im_end|> 收尾） */
        if (ulen <= 64) { out_printf("[输入太长，请分几次发]\n"); return; }
        ulen = ulen * 3 / 4;                   /* 还是太满：再截短正文 */
        truncated = 1;
    }
    if (truncated) out_printf("[输入过长，已截断到 %d 字节]\n", (int)ulen);
    /* 板端 prefill 成本实测 = 每个 token 的权重 GEMV ~0.38 s + 注意力的 O(n²)
     * （0.85 ms × n²/2）：491 tokens 实测 291 s（权重 ~187 s + 注意力 ~102 s）。 */
    if (n > 80) {
        out_printf("[长文输入 %d tokens：板端 prefill 约 %.0f 秒（%.0f 秒权重 + n² 注意力）]\n",
                   n, 0.38 * (double)n + 0.000425 * (double)n * (double)n,
                   0.38 * (double)n);
    }
    if (s_kv.len + n + MAX_NEW + 4 > MAX_CTX) {
        out_printf("[上下文已满 %d/%d，自动开始新对话；输入 \\reset 可手动清空]\n",
                   s_kv.len, MAX_CTX);
        s_kv.len = 0;
        s_nhist = 0;
        feng_mem_clear();           /* 与 generate(keep=0) 的注释保持一致：新对话=记忆一起清 */
        mem_store_save();
    }
    ESP_LOGI(TAG, "in: %s (%d prompt tokens, ctx %d -> %d/%d)", user, n, s_kv.len,
             s_kv.len + n, MAX_CTX);
    reply[0] = 0;
    generate(ids, n, reply, sizeof(reply), 1);
}

static int cmd_is(const char *line, const char *cmd)
{
    const size_t n = strlen(cmd);
    return strncmp(line, cmd, n) == 0 && (line[n] == 0 || line[n] == ' ');
}

void app_main(void)
{
    esp_chip_info_t info;

    log_mem("app_main");
    /* install the UART driver first, then route ESP_LOG and all chat I/O through it */
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, 2048, 2048, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_set_baudrate(UART_PORT, 115200));
    esp_log_set_vprintf(uart_vprintf_impl);
    log_mem("uart");

    esp_chip_info(&info);
    ESP_LOGI(TAG, "feng-30m on ESP32-S3 (%d cores), flash 32MB / PSRAM 16MB", info.cores);
    mem_store_init();
    mem_store_load();
    setup_model();
#if FENG_BENCH_CTX && FENG_KV_Q2
    bench_forward_ctx();
#endif
#if FENG_BENCH_PIE
    bench_pie();
#endif
    feng_tools_set_time(fw_epoch_now);
    feng_tools_set_uptime(fw_uptime_us);

    static char line[4096];
    static char u8[4096];
    out_printf("\n=== feng-30m on ESP32-S3 (serial chat) ===\n");
    out_printf("one sentence per line; the reply streams between << and >>END\n");
    out_printf("多轮对话：自动保留上下文（2048 token），满了自动开新对话\n");
    out_printf("工具：算式（59+1）、现在几点、随机数、记忆（我叫X/我最喜欢Y…）都由板内 tool 直接回答\n");
#if FENG_GBK
    out_printf("commands: \\gbk  \\utf8  \\reset  \\stream N  \\help\n");
#else
    out_printf("commands: \\utf8  \\reset  \\stream N  \\help\n");
#endif
    out_printf("FENG_READY\n");
    fflush(stdout);
    while (1) {
        out_printf("you> ");
        fflush(stdout);
        size_t n = 0;
        int over = 0;
        for (;;) {                      /* read one line from UART0 */
            uint8_t ch = 0;
            const int r = uart_read_bytes(UART_NUM_0, &ch, 1, pdMS_TO_TICKS(200));
            if (r == 1) {
                if (ch == '\n' || ch == '\r') {
                    uart_write_bytes(UART_PORT, "\n", 1);
                    if (n > 0) break;
                    continue;
                }
                uart_write_bytes(UART_PORT, (const char *)&ch, 1);   /* echo so the user sees typing */
                if (n < sizeof(line) - 1) line[n++] = (char)ch;
                else over = 1;                                    /* 超出缓冲：丢弃并提示 */
            }
        }
        line[n] = 0;
        if (over) out_printf("[输入超过 %d 字节，多余部分已丢弃]\n", (int)sizeof(line) - 1);
        /* mirror the terminal's encoding: GBK terminals send GBK bytes。
         * 没编 GBK 时必须强制 UTF-8：否则输出路径会把所有文本转码失败后丢弃，
         * 板子看起来像死了。 */
#if FENG_GBK
        if (feng_has_high_byte(line, (int)n)) {
            g_out_gbk = !feng_utf8_valid(line, (int)n);
        }
#else
        g_out_gbk = 0;
#endif
        if (line[0] == '\\') {
            if (cmd_is(line, "\\gbk")) {
#if FENG_GBK
                g_out_gbk = 1;
                out_printf("encoding: GBK\n");
#else
                out_printf("这个固件没编 GBK 转码，一直是 UTF-8（重编译加 -DFENG_GBK=ON）\n");
#endif
            } else if (cmd_is(line, "\\utf8")) {
                g_out_gbk = 0;
                out_printf("encoding: UTF-8\n");
            } else if (cmd_is(line, "\\reset")) {
                s_kv.len = 0;
                s_nhist = 0;
                feng_mem_clear();
                mem_store_save();
                out_printf("context cleared（记忆也一起清了，\\mem 可查看）\n");
            } else if (cmd_is(line, "\\mem")) {
                if (cmd_is(line, "\\mem clear")) {
                    feng_mem_clear();
                    mem_store_save();
                    out_printf("memory cleared\n");
                } else {
                    char snap[256];
                    feng_mem_snapshot(snap, sizeof(snap));
                    out_printf("memory: %s\n", snap);
                }
            } else if (cmd_is(line, "\\settime")) {
                const long long e = atoll(line + 8);
                if (e > 1600000000LL) {
                    s_epoch_base = e;
                    s_time_base_us = esp_timer_get_time();
                    s_time_synced = 1;
                    out_printf("time synced: epoch %lld\n", e);
                } else {
                    out_printf("usage: \\settime <unix_epoch>\n");
                }
            } else if (cmd_is(line, "\\help")) {
#if FENG_GBK
                out_printf("\\gbk   reply in GBK (for SuperCom/XCOM in ANSI mode)\n");
#endif
                out_printf("\\utf8  reply in UTF-8\n");
                out_printf("\\reset clear the conversation context (multi-turn is on by default)\n");
                out_printf("\\mem   查看引擎记住的事实（\\mem clear 清空）\n");
                out_printf("\\settime <unix秒> 宿主对时（脚本连接时会自动发）\n");
                out_printf("\\stream N  flush every N bytes (0 = whole reply at once, default 30)\n");
                out_printf("算式 / 现在几点 / 随机数 自动走板内 tool；本命令帮助不经过模型\n");
                out_printf("\\help  this text\n");
            } else if (cmd_is(line, "\\stream")) {
                int v = atoi(line + 7);
                if (v < 0) v = 0;
                if (v > 256) v = 256;
                s_stream_chunk = v;
                out_printf("stream chunk = %d bytes%s\n", v,
                           v == 0 ? " (whole reply at once)" : "");
            } else {
                /* 隐藏命令走这里——**故意不列在 \help 里**。
                 * 唯一的线索是 README 小字提到的 nobonus 编译选项（见 feng_bonus.c）。 */
                char hidden[24];   /* 明文 21 字节 + 结尾符（见 feng_bonus.c） */
                if (feng_bonus_decode(line, hidden, sizeof(hidden))) {
                    out_printf("\n  ~ %s ~\n", hidden);
                    out_printf("  (senren * banka)  想关掉这个彩蛋：-DFENG_NO_BONUS=ON\n\n");
                } else {
                    out_printf("unknown command, try \\help\n");
                }
            }
            continue;
        }
        if (n == 0) continue;
        int m = (int)n;
        if (g_out_gbk) {
            m = feng_gbk_to_utf8(line, (int)n, u8, sizeof(u8) - 1);
            u8[m] = 0;
        } else {
            memcpy(u8, line, n);
            u8[n] = 0;
        }
        /* 计算 tool：纯算式直接由 SoC 运算器算，秒回、100% 准确，不占模型/上下文 */
        char calc_reply[256];
        if (feng_tok_has_special(u8)) {
            out_printf("[输入包含保留的对话模板标记（<|im_start|> 等），已拒绝]\n");
            continue;
        }
        const int learned = feng_mem_learn(u8);  /* 先记事实（不拦截，模型仍能看到这句） */
        if (feng_calc_answer(u8, calc_reply, sizeof(calc_reply)) ||
            feng_mem_answer(u8, calc_reply, sizeof(calc_reply)) ||
            feng_time_answer(u8, calc_reply, sizeof(calc_reply)) ||
            feng_random_answer(u8, calc_reply, sizeof(calc_reply))) {
            if (learned || line_may_forget(u8)) mem_store_save();
            ESP_LOGI(TAG, "tool: %s -> %s", u8, calc_reply);
            out_printf("<< %s\n>>END\n", calc_reply);
            fflush(stdout);
            continue;
        }
        if (learned || line_may_forget(u8)) mem_store_save();
        chat_once(u8);
    }
}
