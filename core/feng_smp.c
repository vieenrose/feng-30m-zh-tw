/* Split each GEMV across the two ESP32-S3 cores.
 *
 * Two worker tasks (one pinned to core 0, one to core 1) each compute half of the
 * output rows.  The caller posts the job and blocks on the completion semaphores,
 * so a GEMV really runs on both cores at once.
 */
#include "feng.h"

#if defined(ESP_PLATFORM)

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_attr.h"
#include "esp_timer.h"

#define SMP_NPARTS 2
#define SMP_TASK_STACK 6144      /* q2 注意力回调的 vacc[64] 等栈上局部量更大 */
#define SMP_TASK_PRIO 5

enum { SMP_JOB_GEMV = 0, SMP_JOB_CALLBACK = 1 };

typedef struct {
    SemaphoreHandle_t start;
    SemaphoreHandle_t done;
    int kind;
    const void *tensor;
    uint32_t dtype;
    const float *x;
    float *y;
    int n_in, r0, r1;
    feng_par_fn fn;
    void *arg;
    int part, nparts;
    /* diagnostics */
    int core, runs;
    int64_t us_last;
} smp_part_t;

static smp_part_t s_parts[SMP_NPARTS];
static StaticSemaphore_t s_start_buf[SMP_NPARTS];
static StaticSemaphore_t s_done_buf[SMP_NPARTS];
static int s_smp_ok;
static const char *TAG = "feng_smp";

static void smp_worker(void *arg)
{
    smp_part_t *p = (smp_part_t *)arg;
    for (;;) {
        if (xSemaphoreTake(p->start, portMAX_DELAY) != pdTRUE) continue;
        p->core = xPortGetCoreID();
        const int64_t t0 = esp_timer_get_time();
        if (p->kind == SMP_JOB_CALLBACK) {
            p->fn(p->arg, p->part, p->nparts);
        } else {
            feng_gemv_range(p->tensor, p->dtype, p->x, p->y, p->r0, p->r1, p->n_in);
        }
        p->us_last = esp_timer_get_time() - t0;
        p->runs++;
        xSemaphoreGive(p->done);
    }
}

void feng_smp_init(void)
{
    if (s_smp_ok) return;
    feng_quant_init();          /* LUT 必须在 worker 启动前建好，避免并发首次初始化 */
    for (int i = 0; i < SMP_NPARTS; i++) {
        s_parts[i].start = xSemaphoreCreateBinaryStatic(&s_start_buf[i]);
        s_parts[i].done = xSemaphoreCreateBinaryStatic(&s_done_buf[i]);
        if (!s_parts[i].start || !s_parts[i].done) {
            ESP_LOGE(TAG, "semaphore init failed (internal free %u)",
                     (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
            return;
        }
    }
    ESP_LOGI(TAG, "heap before workers: internal %u B (largest %u), psram %u B",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    for (int i = 0; i < SMP_NPARTS; i++) {
        TaskHandle_t h = NULL;
        BaseType_t ok = xTaskCreatePinnedToCoreWithCaps(smp_worker, "feng_smp", SMP_TASK_STACK,
                                                        &s_parts[i], SMP_TASK_PRIO, &h, i,
                                                        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (ok != pdPASS) {
            ESP_LOGW(TAG, "internal stack failed on core %d, retrying in PSRAM", i);
            ok = xTaskCreatePinnedToCoreWithCaps(smp_worker, "feng_smp", SMP_TASK_STACK,
                                                 &s_parts[i], SMP_TASK_PRIO, &h, i,
                                                 MALLOC_CAP_SPIRAM);
        }
        if (ok != pdPASS) {
            ESP_LOGE(TAG, "task create failed on core %d", i);
            return;
        }
    }
    s_smp_ok = 1;
    ESP_LOGI(TAG, "GEMV workers up on core 0 + core 1");
}

void feng_gemv_par(const void *tensor, uint32_t dtype, const float *x, float *y,
                   int n_out, int n_in)
{
#if FENG_GEMV_A8 || FENG_GEMV_PIE
    feng_gemv_a8_prepare(x, n_in);
#endif
    if (!s_smp_ok || n_out < 32) {
        feng_gemv_range(tensor, dtype, x, y, 0, n_out, n_in);
        return;
    }
    const int mid = n_out / 2;
    for (int i = 0; i < SMP_NPARTS; i++) {
        smp_part_t *p = &s_parts[i];
        p->kind = SMP_JOB_GEMV;
        p->tensor = tensor;
        p->dtype = dtype;
        p->x = x;
        p->y = y;
        p->n_in = n_in;
        p->r0 = (i == 0) ? 0 : mid;
        p->r1 = (i == 0) ? mid : n_out;
    }
    /* Both jobs must be posted before either worker can preempt us: otherwise the
     * worker on our own core starts immediately and the second job is only posted
     * once it finishes (that serialised the two halves completely). */
    const UBaseType_t old_prio = uxTaskPriorityGet(NULL);
    if (old_prio <= SMP_TASK_PRIO) vTaskPrioritySet(NULL, SMP_TASK_PRIO + 1);
    xSemaphoreGive(s_parts[0].start);
    xSemaphoreGive(s_parts[1].start);
    if (old_prio <= SMP_TASK_PRIO) vTaskPrioritySet(NULL, old_prio);
    xSemaphoreTake(s_parts[0].done, portMAX_DELAY);
    xSemaphoreTake(s_parts[1].done, portMAX_DELAY);
    if (s_parts[0].runs <= 3) {
        ESP_LOGI(TAG, "par n_out=%d: part0 core%d rows[%d,%d) %lldus | part1 core%d rows[%d,%d) %lldus",
                 n_out, s_parts[0].core, s_parts[0].r0, s_parts[0].r1, s_parts[0].us_last,
                 s_parts[1].core, s_parts[1].r0, s_parts[1].r1, s_parts[1].us_last);
    }
}

void feng_par_run(feng_par_fn fn, void *arg, int nparts)
{
    if (fn == NULL) return;
    if (!s_smp_ok || nparts <= 1) {
        fn(arg, 0, 1);
        return;
    }
    if (nparts > SMP_NPARTS) nparts = SMP_NPARTS;
    for (int i = 0; i < SMP_NPARTS; i++) {
        smp_part_t *p = &s_parts[i];
        p->kind = SMP_JOB_CALLBACK;
        p->fn = fn;
        p->arg = arg;
        p->part = i;
        p->nparts = nparts;
    }
    const UBaseType_t old_prio = uxTaskPriorityGet(NULL);
    if (old_prio <= SMP_TASK_PRIO) vTaskPrioritySet(NULL, SMP_TASK_PRIO + 1);
    xSemaphoreGive(s_parts[0].start);
    xSemaphoreGive(s_parts[1].start);
    if (old_prio <= SMP_TASK_PRIO) vTaskPrioritySet(NULL, old_prio);
    xSemaphoreTake(s_parts[0].done, portMAX_DELAY);
    xSemaphoreTake(s_parts[1].done, portMAX_DELAY);
}

#elif defined(FENG_PHONE)
/* 手机端多核 GEMV：输出行切块交给 feng_phone.c 的常驻 pthread 线程池。
 * 行与行完全独立 -> 与单线程逐位相同；动态领任务，耗时不均也能均衡。 */
/* 每次领任务算多少行。
 * 一行只有几百字节权重（不到 1 微秒的活），如果一行一个任务，8 个线程会为了
 * 同一个计数器反复争抢同一条缓存行——lm_head 有 16384 行，光原子操作就能吃掉
 * 可观的时间。按块领（32 行）把这个开销摊掉，同时块足够小、动态领取仍然能均衡。 */
#define GEMV_ROW_CHUNK 32

static int n_chunks(int n_out)
{
    return (n_out + GEMV_ROW_CHUNK - 1) / GEMV_ROW_CHUNK;
}

typedef struct {
    const void *tensor;
    uint32_t dtype;
    const float *x;
    float *y;
    int n_in;
    int n_out;
} gemv_job_t;

static void gemv_row_chunk(void *ctx, int i)
{
    const gemv_job_t *j = (const gemv_job_t *)ctx;
    const int r0 = i * GEMV_ROW_CHUNK;
    int r1 = r0 + GEMV_ROW_CHUNK;
    if (r1 > j->n_out) r1 = j->n_out;
    feng_gemv_range(j->tensor, j->dtype, j->x, j->y, r0, r1, j->n_in);
}

void feng_smp_init(void)
{
    /* 线程池由 feng_phone_init() 建立；这里只做幂等确认，保持与板端同名 API。 */
    (void)feng_phone_init();
    feng_quant_init();          /* LUT 必须先于 worker 首次 GEMV 建好 */
}

void feng_gemv_par(const void *tensor, uint32_t dtype, const float *x, float *y,
                   int n_out, int n_in)
{
#if FENG_GEMV_A8 || FENG_GEMV_PIE || FENG_PHONE_NEON
    feng_gemv_a8_prepare(x, n_in);
#endif
    if (n_out <= 0) return;
    /* 行数太少的调用（norm 之类的小张量）并行反而更贵，直接串行整段。
     * 阈值与板端一致（n_out < 32 退回单核）。 */
    if (n_out < 32 || feng_phone_threads() <= 1) {
        feng_gemv_range(tensor, dtype, x, y, 0, n_out, n_in);
        return;
    }
    gemv_job_t job = { tensor, dtype, x, y, n_in, n_out };
    feng_phone_parallel_for(n_chunks(n_out), gemv_row_chunk, &job);
}

/* ---------------- feng_gemv3 的合并并行版 ---------------- */

typedef struct {
    const void *t[3];
    float *y[3];
    int n_out[3];
    int nc[3];          /* 每个矩阵切成几块 */
    int n_mat;
    uint32_t dtype;
    const float *x;
    int n_in;
} gemv3_job_t;

static void gemv3_row_chunk(void *ctx, int i)
{
    /* 把一维块号 i 映射回「第几个矩阵的第几块」——三个矩阵共处一个并行区间，
     * 每 token 少两次信号量往返（q/k/v 与 gate/up 各一次）。 */
    const gemv3_job_t *j = (const gemv3_job_t *)ctx;
    int k = 0;
    while (k < j->n_mat - 1 && i >= j->nc[k]) {
        i -= j->nc[k];
        k++;
    }
    const int r0 = i * GEMV_ROW_CHUNK;
    int r1 = r0 + GEMV_ROW_CHUNK;
    if (r1 > j->n_out[k]) r1 = j->n_out[k];
    feng_gemv_range(j->t[k], j->dtype, j->x, j->y[k], r0, r1, j->n_in);
}

void feng_gemv3_par(const void *w0, const void *w1, const void *w2, uint32_t dtype,
                    const float *x, float *y0, float *y1, float *y2,
                    int n_out0, int n_out1, int n_out2, int n_in)
{
#if FENG_GEMV_A8 || FENG_GEMV_PIE || FENG_PHONE_NEON
    feng_gemv_a8_prepare(x, n_in);
#endif
    gemv3_job_t job;
    int n = 0;
    if (w0) { job.t[n] = w0; job.y[n] = y0; job.n_out[n] = n_out0; n++; }
    if (w1) { job.t[n] = w1; job.y[n] = y1; job.n_out[n] = n_out1; n++; }
    if (w2) { job.t[n] = w2; job.y[n] = y2; job.n_out[n] = n_out2; n++; }
    job.n_mat = n;
    if (n == 0) return;
    job.dtype = dtype;
    job.x = x;
    job.n_in = n_in;

    int total_rows = 0, total_chunks = 0;
    for (int i = 0; i < n; i++) {
        job.nc[i] = n_chunks(job.n_out[i]);
        total_rows += job.n_out[i];
        total_chunks += job.nc[i];
    }
    if (total_rows < 32 || feng_phone_threads() <= 1) {
        for (int i = 0; i < n; i++) {
            feng_gemv_range(job.t[i], dtype, x, job.y[i], 0, job.n_out[i], n_in);
        }
        return;
    }
    feng_phone_parallel_for(total_chunks, gemv3_row_chunk, &job);
}

/* 通用并行回调：分片之间写的是不相交的输出，结果与串行一致。
 * 只有 1 片或线程池没起来时退化为串行循环。 */
typedef struct { feng_par_fn fn; void *arg; int nparts; } par_shim_t;

static void par_shim(void *vp, int i)
{
    const par_shim_t *s = (const par_shim_t *)vp;
    s->fn(s->arg, i, s->nparts);
}

void feng_par_run(feng_par_fn fn, void *arg, int nparts)
{
    if (fn == NULL) return;
    if (nparts < 1) nparts = 1;
    if (nparts == 1 || feng_phone_threads() <= 1) {
        for (int i = 0; i < nparts; i++) fn(arg, i, nparts);
        return;
    }
    par_shim_t s = { fn, arg, nparts };
    feng_phone_parallel_for(nparts, par_shim, &s);
}

#else /* host build (tools/pc check): single core, no RTOS */

void feng_smp_init(void)
{
    feng_quant_init();
}

void feng_gemv_par(const void *tensor, uint32_t dtype, const float *x, float *y,
                   int n_out, int n_in)
{
#if FENG_GEMV_A8 || FENG_GEMV_PIE
    feng_gemv_a8_prepare(x, n_in);
#endif
    feng_gemv_range(tensor, dtype, x, y, 0, n_out, n_in);
}

/* 宿主/PC 没有 RTOS worker：顺序跑每个 part，结果与并行版相同。 */
void feng_par_run(feng_par_fn fn, void *arg, int nparts)
{
    if (fn == NULL) return;
    if (nparts < 1) nparts = 1;
    for (int i = 0; i < nparts; i++) fn(arg, i, nparts);
}

#endif
