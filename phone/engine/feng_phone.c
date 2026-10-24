/* 手机端平台层实现（POSIX：Termux / Android / 桌面 Linux / WSL 交叉编译）。
 *
 * 线程池模型：worker 线程常驻，用信号量「主线程 post 开工 -> worker 领活 -> post 完成」，
 * 每轮任务的索引分配走一个原子计数器（动态负载均衡，行耗时不均时比静态切分快）。
 * 调用者本身也领活，所以 T 个线程就是 T 份算力，不浪费。
 *
 * 为什么不用每帧建线程 / 自旋等待：
 *  - 手机上是电池问题，信号量握手比空转省电；建线程的开销是微秒级的几十倍；
 *  - 每 token 有几十次 GEMV 提交，握手必须便宜。 */

/* -std=c23 是严格 ISO 模式，glibc 会因此隐藏 pthread / semaphore / mmap / sysconf
 * 这些 POSIX 声明（__STRICT_ANSI__ 关掉了 _DEFAULT_SOURCE）。必须在任何 include
 * 之前打开特性宏。Android(bionic) 上 _GNU_SOURCE 本来就是默认，重复定义无害。 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

#include "feng_phone.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Windows（MSYS2/MinGW）没有 <sys/mman.h>，也没有 sysconf。
 * 这里走"不 mmap"的分支，落到本文件里**本来就有**的「整块读进堆」回退路径——
 * 目的是让手机引擎能在**开发机（Windows）上直接编译与验证**：
 * esp32s3-feng-llm/pc 那套 32 题矩阵就能在同一平台上跑出对照结果，
 * 不用绕 WSL。Android NDK / Termux / Linux 仍走下面的 POSIX 分支，行为不变。
 * 注意 pthread 与信号量在 MSYS2 上由 winpthreads 提供，是可用的。 */
#if defined(_WIN32)
#define FENG_PHONE_NO_MMAP 1
#define WIN32_LEAN_AND_MEAN
#include <windows.h>            /* GetSystemInfo：替代 sysconf(_SC_NPROCESSORS_ONLN) */
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#else
#include <sys/mman.h>
#endif
/* Windows 的 CRT 默认以文本模式打开文件：模型/分词器是二进制，遇到 0x1A（Ctrl-Z）
 * 会被当成 EOF，read 提前返回 0，feng_phone_load 报 rc=-2。POSIX 上 O_BINARY 未定义，
 * 补 0 即可。这是开发机上验证手机引擎的必需项，Android/Linux 行为不变。 */
#ifndef O_BINARY
#define O_BINARY 0
#endif

/* ------------------------------------------------------------------ */
/* 配置                                                                */
/* ------------------------------------------------------------------ */

static int s_nthreads = 1;      /* 总线程数，含调用者 */
static int s_threads_explicit;  /* 用户显式设置过（--threads / JNI 参数） */
/* NEON 内核（Q4 的 sdot 路径）**默认关闭**。
 *
 * 它不是逐位一致的：激活在进入内核前被量化成 int8（a8），误差会逐层累积。
 * P30 Pro 实测（同模型、同套件、交替多轮）：
 *   sdot 开：151 tok/s，mem12 7/12、mt 8/10、seq 10/10（合计 25/32）
 *   标量 LUT：56 tok/s，mem12 11/12、mt 8/10、seq 9/10（合计 28/32）
 * 速度差 2.7×，但多轮记忆类问题明显变差——30M 模型本来就贴着能力边界，
 * 这点量化误差足以翻掉 4 道题。产品默认取准确的那条（也与板端/PC 数值一致）。
 * 要速度：--neon 1 或 FENG_PHONE_NEON=1，并自行复跑 32 题 + mem12。 */
static int s_neon_on = 0;
static int s_inited;

/* 在线 CPU 数：POSIX 走 sysconf，Windows 走 GetSystemInfo（MSYS2 没有 sysconf）。 */
static long fp_cpu_count(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)si.dwNumberOfProcessors;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

int feng_phone_cpus(void)
{
    long n = fp_cpu_count();
    if (n < 1) return 1;
    if (n > FENG_PHONE_MAX_THREADS) return FENG_PHONE_MAX_THREADS;
    return (int)n;
}

void feng_phone_set_neon(int on)
{
#if FENG_PHONE_NEON
    s_neon_on = on ? 1 : 0;
#else
    (void)on;
    s_neon_on = 0;
#endif
}

int feng_phone_set_threads(int n)
{
    if (n < 1) n = 1;
    if (n > FENG_PHONE_MAX_THREADS) n = FENG_PHONE_MAX_THREADS;
    s_nthreads = n;
    s_threads_explicit = 1;
    return n;
}

int feng_phone_neon_on(void) { return s_neon_on; }
int feng_phone_threads(void) { return s_nthreads; }

/* ------------------------------------------------------------------ */
/* 线程池                                                              */
/* ------------------------------------------------------------------ */

#define FP_MAXW (FENG_PHONE_MAX_THREADS - 1)   /* worker 上限（调用者不算） */

typedef void (*fp_job_fn)(void *, int);

/* 自旋轮数：每 token 有 ~45 次 GEMV 提交，区间之间只隔着几微秒的标量活
 * （rmsnorm / rope / softmax）。如果每次都让 worker 睡到信号量上，
 * 光唤醒延迟就要几十微秒 × 45 次，比并行省下的时间还多（实测 4 线程只有 1.46x）。
 * 所以先自旋等一小会儿：正常推理时 worker 全程都醒着，只有真的空闲下来
 * （等用户输入）才回落到信号量休眠——既不拖慢推理，也不白耗电。 */
#if defined(__x86_64__) || defined(__i386__)
#define FP_SPIN 3000                 /* pause 大约 140 周期/次，合计 ~80us @5GHz */
#elif defined(__aarch64__)
#define FP_SPIN 30000                /* yield 很便宜，多转几圈凑到同样的时间量级 */
#else
#define FP_SPIN 10000
#endif

#if defined(__x86_64__) || defined(__i386__)
#define fp_cpu_relax() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define fp_cpu_relax() __asm__ __volatile__("yield")
#else
#define fp_cpu_relax() ((void)0)
#endif

static pthread_t s_tid[FP_MAXW > 0 ? FP_MAXW : 1];
static sem_t s_go[FP_MAXW > 0 ? FP_MAXW : 1];     /* 主线程 -> 睡着的 worker i */
static sem_t s_done_sem;                          /* worker -> 主线程（计数） */
static fp_job_fn s_job;
static void *s_job_ctx;
static int s_job_n;                 /* 本轮任务数（经 s_round 发布，读侧用 acquire） */
static int s_job_nw;                /* 本轮参与的 worker 数（同上发布） */
static int s_next;                  /* 下一个要领取的索引（原子 add） */
static int s_round;                 /* 轮次号：worker 靠它判断有新活（主线程自增） */
static int s_parked[FP_MAXW > 0 ? FP_MAXW : 1];   /* worker 是否已登记「我要睡了」 */
static int s_pool_nw;               /* 常驻 worker 线程数（建池时确定） */
static int s_shutdown;              /* 保留：正常路径不置位（进程退出即回收） */

static void *fp_worker(void *arg)
{
    const int me = (int)(intptr_t)arg;
    int my_round = 0;
    for (;;) {
        /* 自旋等新任务，超时才登记休眠 */
        int spins = 0;
        for (;;) {
            if (__atomic_load_n(&s_round, __ATOMIC_SEQ_CST) != my_round) break;
            if (++spins > FP_SPIN) {
                /* 先登记，再复核一次轮次，最后才睡：这两步都必须是 seq_cst，
                 * 否则和主线程的「自增轮次 -> 检查登记」可能互相擦肩而过，
                 * 结果是 worker 睡死、主线程在 done 上等到底。 */
                __atomic_store_n(&s_parked[me], 1, __ATOMIC_SEQ_CST);
                if (__atomic_load_n(&s_round, __ATOMIC_SEQ_CST) == my_round) {
                    while (sem_wait(&s_go[me]) != 0 && errno == EINTR) { /* 重试 */ }
                }
                __atomic_store_n(&s_parked[me], 0, __ATOMIC_SEQ_CST);
                spins = 0;
            } else {
                fp_cpu_relax();
            }
        }
        my_round = __atomic_load_n(&s_round, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&s_shutdown, __ATOMIC_ACQUIRE)) return NULL;

        fp_job_fn fn = s_job;
        void *ctx = s_job_ctx;
        const int n = __atomic_load_n(&s_job_n, __ATOMIC_ACQUIRE);
        /* 本轮只允许前 s_job_nw 个 worker 领活。旧实现所有 worker 都领，
         * 但主线程只等 nw 个 done：线程数被调小之后会多 post 出几个信号量计数，
         * 下一轮 sem_wait 直接返回，主线程在 worker 还没算完时就往下走（结果错乱）。 */
        const int my_nw = __atomic_load_n(&s_job_nw, __ATOMIC_ACQUIRE);
        if (me < my_nw) {
            for (;;) {
                const int i = __atomic_fetch_add(&s_next, 1, __ATOMIC_RELAXED);
                if (i >= n) break;
                fn(ctx, i);
            }
            sem_post(&s_done_sem);
        }
    }
}

static void fp_pool_start(int nw)
{
    if (sem_init(&s_done_sem, 0, 0) != 0) return;
    int started = 0;
    for (int i = 0; i < nw; i++) {
        if (sem_init(&s_go[i], 0, 0) != 0) break;
        if (pthread_create(&s_tid[i], NULL, fp_worker, (void *)(intptr_t)i) != 0) break;
        started++;
    }
    __atomic_store_n(&s_pool_nw, started, __ATOMIC_RELEASE);
    s_inited = 1;
}

int feng_phone_init(void)
{
    /* 线程数：环境变量 > 默认 min(4, CPU 数) */
    int want = 0;
    const char *e = getenv("FENG_PHONE_THREADS");
    if (e && *e) {
        const int v = atoi(e);
        if (v >= 1) want = v;
    }
    /* 显式 --threads / JNI 参数优先：旧实现每次 init 都按环境变量/默认值
     * 重设，CLI 传 1 线程、JNI 传 2 都会被悄悄改成 4。 */
    if (want == 0 && !s_threads_explicit) {
        const int c = feng_phone_cpus();
        want = c > 4 ? 4 : c;
    }
    if (want > 0) feng_phone_set_threads(want);

    const char *ne = getenv("FENG_PHONE_NEON");
    if (ne && *ne) feng_phone_set_neon(atoi(ne) != 0);

    /* 线程池只在第一次真正拉起（worker 数 = 总线程数 - 1，调用者也算一份算力）。
     * 之后调 feng_phone_set_threads 只影响每轮参与的 worker 数，不再建线程。 */
    if (!s_inited) {
        int nw = s_nthreads - 1;
        if (nw > FP_MAXW) nw = FP_MAXW;
        if (nw < 0) nw = 0;
        fp_pool_start(nw);
    }
    return s_nthreads;
}

void feng_phone_parallel_for(int n, void (*fn)(void *, int), void *ctx)
{
    if (n <= 0) return;
    /* 任务太少 / 单线程 / 池没起来 -> 串行（与并行同序，结果逐位一致） */
    int nw = __atomic_load_n(&s_pool_nw, __ATOMIC_ACQUIRE);
    if (nw > s_nthreads - 1) nw = s_nthreads - 1;
    /* 只有“单片/没有 worker”才串行。旧的 n<8 门槛会把默认 4 片 attention
     * 整段退化成调用者单线程执行（实测 4 片 unique_pthreads=1）。GEMV 的小
     * 批任务自己按行数决定是否调用这里，不需要这层重复门槛。 */
    if (nw <= 0 || n <= 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }

    s_job = fn;
    s_job_ctx = ctx;
    __atomic_store_n(&s_next, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&s_job_n, n, __ATOMIC_RELAXED);          /* 由下面的 release 发布 */
    __atomic_store_n(&s_job_nw, nw, __ATOMIC_RELAXED);        /* 同上：本轮参与的 worker 数 */
    __atomic_store_n(&s_round, __atomic_load_n(&s_round, __ATOMIC_RELAXED) + 1,
                     __ATOMIC_SEQ_CST);                        /* 放行 worker */
    /* 只叫醒已经登记休眠的那些；还在自旋的 worker 会自己看到轮次变化 */
    for (int i = 0; i < nw; i++) {
        if (__atomic_load_n(&s_parked[i], __ATOMIC_SEQ_CST)) sem_post(&s_go[i]);
    }

    /* 调用者自己也领活，直到任务领完；再等 worker 收工 */
    for (;;) {
        const int i = __atomic_fetch_add(&s_next, 1, __ATOMIC_RELAXED);
        if (i >= n) break;
        fn(ctx, i);
    }
    for (int i = 0; i < nw; i++) {
        while (sem_wait(&s_done_sem) != 0 && errno == EINTR) { /* 重试 */ }
    }
}

/* ------------------------------------------------------------------ */
/* 文件加载                                                            */
/* ------------------------------------------------------------------ */

/* mmap 区域登记表，供 unload 用；数量很小（模型 + tokenizer），线性表足够。 */
#define FP_MAX_MAPS 8
typedef struct { void *base; size_t len; } fp_map_t;
static fp_map_t s_maps[FP_MAX_MAPS];
static int s_nmaps;

int feng_phone_load(const char *path, const uint8_t **blob, size_t *len)
{
    *blob = NULL;
    *len = 0;
    const int fd = open(path, O_RDONLY | O_BINARY | O_CLOEXEC);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return -2; }
    const size_t n = (size_t)st.st_size;

    /* 优先 mmap：只读、零拷贝，页面按需从 page cache 取（不计入匿名常驻内存） */
#if defined(FENG_PHONE_NO_MMAP)
    void *p = NULL;                 /* Windows：跳过 mmap，直接走下面的堆读取回退 */
#else
    void *p = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) p = NULL;
#endif
    if (p != NULL) {
        /* 表满时必须失败，不能返回一个未登记的 mmap：unload 查不到记录就会
         * 对它调用 free()（非法释放）。 */
        if (s_nmaps >= FP_MAX_MAPS) {
#if !defined(FENG_PHONE_NO_MMAP)
            munmap(p, n);
#endif
            close(fd);
            return -4;
        }
        close(fd);
        s_maps[s_nmaps].base = p;
        s_maps[s_nmaps].len = n;
        s_nmaps++;
        *blob = (const uint8_t *)p;
        *len = n;
        return 0;
    }

    /* 回退：整块读进堆（个别文件系统/路径上 mmap 会失败） */
    uint8_t *buf = (uint8_t *)malloc(n + 1);
    if (!buf) { close(fd); return -3; }
    size_t got = 0;
    while (got < n) {
        const ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) { free(buf); close(fd); return -2; }
        got += (size_t)r;
    }
    buf[n] = 0;
    close(fd);
    *blob = buf;
    *len = n;
    return 0;
}

void feng_phone_unload(const uint8_t *blob, size_t len)
{
    (void)len;
    if (!blob) return;
#if !defined(FENG_PHONE_NO_MMAP)
    for (int i = 0; i < s_nmaps; i++) {
        if ((const uint8_t *)s_maps[i].base == blob) {
            munmap(s_maps[i].base, s_maps[i].len);
            s_maps[i] = s_maps[s_nmaps - 1];
            s_nmaps--;
            return;
        }
    }
#endif
    /* Windows 分支下 s_nmaps 恒为 0（从不 mmap），走到这里就是堆缓冲，直接 free */
    free((void *)blob);
}

/* ------------------------------------------------------------------ */
/* 时间源                                                              */
/* ------------------------------------------------------------------ */

long long feng_phone_epoch(void)
{
    return (long long)time(NULL);
}

long long feng_phone_uptime_us(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (long long)ts.tv_sec * 1000000LL + (long long)ts.tv_nsec / 1000LL;
}
