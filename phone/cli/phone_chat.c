/* 手机端（Termux / Android）聊天 CLI 与基准。
 *
 * 与 pc/pc_chat.cpp 同口径：同一套引擎、同一个 prompt 模板、同样的 tool 调度顺序
 * （算式 → 记忆 → 时间 → 随机数），默认多轮 KV 累积，\reset 清空。
 * 差别只在宿主层：模型用 mmap 读文件（feng_phone_load）、多线程走 pthread 线程池、
 * 多了 --bench 与 --ab 两个自检模式。纯 C23（v4.0 起引擎全面 C23）。
 *
 * 用法：phone_chat [选项] [model_export 目录]
 *   --ctx N        上下文长度（默认 2048；上限 = 编译期 FENG_PHONE_MAX_CTX）
 *   --threads N    线程数（默认 min(4, 在线 CPU 数)）
 *   --neon 0|1     强制 NEON 内核开关（默认 1，仅 aarch64 编译期有效）
 *   --max-new N    单轮最多生成 token（默认 96）
 *   --prompt TEXT  问一句就退出（脚本化用）
 *   --bench [N]    基准模式：固定问题生成 N 个 token，打印 tok/s（默认 32）
 *   --ab           自检：同一问题在 NEON 开/关下各跑一遍，比对 logits 与 argmax
 *   -h, --help     帮助
 *
 * 模型目录里需要 model.bin 与 tokenizer.bin（见 phone/README.md）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "feng.h"
#include "feng_calc.h"
#include "feng_memory.h"
#include "feng_phone.h"
#include "feng_tokenizer.h"
#include "feng_tools.h"

#if FENG_KV_Q2
#define KV_MODE "q2"
#elif FENG_KV_INT8
#define KV_MODE "int8"
#elif FENG_KV_FP16
#define KV_MODE "fp16"
#else
#define KV_MODE "fp32"
#endif

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "oom %zu\n", n); exit(1); }
    memset(p, 0, n);
    return p;
}

static double now_ms(void)
{
    return (double)feng_phone_uptime_us() / 1000.0;
}

/* ---------------- 引擎会话（单实例；引擎本身非重入） ---------------- */

typedef struct {
    feng_model_t m;
    feng_tok_t tok;
    feng_kv_t kv;
    feng_workspace_t ws;
    const uint8_t *mblob;
    size_t mlen;
    const uint8_t *tblob;
    size_t tlen;
    int ctx;
    int nl_id;
    char mem_path[512];     /* 持久化记忆：<模型目录>/mem.txt */
} session_t;

/* 记忆持久化：与板端 NVS 共用同一份导出格式（feng_mem_export/import）。
 * 空记忆不写文件，避免留下空壳；读写失败只提示，不影响对话。 */
static void mem_store_load(session_t *s)
{
    FILE *f = fopen(s->mem_path, "rb");
    if (!f) return;
    char buf[1024];
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    if (feng_mem_import(buf))
        feng_mem_mark_clean();          /* 刚读进来的不算“待保存变更” */
    else
        fprintf(stderr, "mem.txt 解析失败或为空，已忽略\n");
}

static void mem_store_save(session_t *s)
{
    char buf[1024];
    const int n = feng_mem_export(buf, sizeof(buf));
    if (n < 0) return;
    if (n == 0) { (void)remove(s->mem_path); feng_mem_mark_clean(); return; }
    FILE *f = fopen(s->mem_path, "wb");
    if (!f) { fprintf(stderr, "写不了 %s\n", s->mem_path); return; }
    const size_t w = fwrite(buf, 1, (size_t)n, f);
    if (fclose(f) != 0 || w != (size_t)n) {
        fprintf(stderr, "写 %s 不完整\n", s->mem_path);
        return;                          /* 保留 dirty，下次再试 */
    }
    feng_mem_mark_clean();
}

/* 遗忘/学习是否落盘由 feng_mem_dirty() 决定：tool 自己知道改没改记忆，
 * 不再从自然语言里猜“忘”字（“别记我的名字了”以前不会落盘，重启就复活）。 */
static int mem_changed(void)
{
    return feng_mem_dirty();
}

static int session_open(session_t *s, const char *dir, int ctx)
{
    char path[512];
    memset(s, 0, sizeof(*s));
    s->ctx = ctx;
    if (snprintf(s->mem_path, sizeof(s->mem_path), "%s/mem.txt", dir) >= (int)sizeof(s->mem_path)) {
        fprintf(stderr, "模型目录路径太长：%s\n", dir);
        return -1;
    }

    snprintf(path, sizeof(path), "%s/model.bin", dir);
    int rc = feng_phone_load(path, &s->mblob, &s->mlen);
    if (rc != 0) { fprintf(stderr, "打不开 %s（rc=%d）\n", path, rc); return -1; }
    snprintf(path, sizeof(path), "%s/tokenizer.bin", dir);
    rc = feng_phone_load(path, &s->tblob, &s->tlen);
    if (rc != 0) { fprintf(stderr, "打不开 %s（rc=%d）\n", path, rc); return -1; }

    if (feng_model_init(&s->m, s->mblob, s->mlen) != 0) {
        fprintf(stderr, "model.bin 解析失败（magic/版本不匹配？）\n");
        return -1;
    }
    if (feng_tok_load(&s->tok, s->tblob, s->tlen) != 0) {
        fprintf(stderr, "tokenizer.bin 解析失败\n");
        return -1;
    }

    feng_kv_t *kv = &s->kv;
    kv->ctx = ctx;
    kv->len = 0;
    /* k_cache/v_cache 只放数据，不含 scale——必须用 feng_kv_data_bytes()。
     * 旧实现拿 feng_kv_bytes()（data+两份 scale）的一半当数据缓冲，又另外分配
     * scale，q2/block8 下多占 50%（32k：154 MiB → 231 MiB）。 */
    const size_t kv_data = feng_kv_data_bytes(&s->m, ctx);
    kv->k_cache = xmalloc(kv_data + 64);
    kv->v_cache = xmalloc(kv_data + 64);
#if FENG_KV_INT8 || FENG_KV_Q2
    const size_t n_sc = feng_kv_scale_slots(&s->m, ctx);
    kv->k_scale = (uint16_t *)xmalloc(n_sc * 2);
    kv->v_scale = (uint16_t *)xmalloc(n_sc * 2);
#endif

    feng_workspace_t *ws = &s->ws;
    const int h = s->m.hdr.hidden, ff = s->m.hdr.ffn, v = s->m.hdr.vocab;
    ws->max_ctx = ctx;
    ws->x = (float *)xmalloc(sizeof(float) * h);
    ws->xn = (float *)xmalloc(sizeof(float) * h);
    ws->q = (float *)xmalloc(sizeof(float) * h);
    ws->k = (float *)xmalloc(sizeof(float) * h);
    ws->v = (float *)xmalloc(sizeof(float) * h);
    ws->attn = (float *)xmalloc(sizeof(float) * h);
    ws->proj = (float *)xmalloc(sizeof(float) * h);   /* attn/down 投影都输出 hidden */
    ws->gate = (float *)xmalloc(sizeof(float) * ff);
    ws->up = (float *)xmalloc(sizeof(float) * ff);
    ws->ffn = (float *)xmalloc(sizeof(float) * ff);
    ws->logits = (float *)xmalloc(sizeof(float) * v);
    ws->scratch = (float *)xmalloc(sizeof(float) * (ctx > ff ? ctx : ff));

    int one = 0;
    s->nl_id = (feng_tok_encode(&s->tok, "\n", &one, 1) == 1) ? one : 0;
    mem_store_load(s);
    return 0;
}

/* 跑一轮：prefill + 贪心 decode。out 可为 NULL（基准模式不收集文本）。
 * 返回生成的 token 数；t_pre_ms / t_dec_ms 分别给 prefill 与 decode 的毫秒数。 */
static int session_generate(session_t *s, const char *user, int max_new,
                            char *out, int out_sz, int *olen,
                            double *t_pre_ms, double *t_dec_ms)
{
    const int v = s->m.hdr.vocab;
    const int ctx = s->ctx;
    if (olen) *olen = 0;
    if (out && out_sz > 0) out[0] = 0;          /* 提前返回时调用方不会打印未初始化栈 */
    char prompt[4096];
    const int pn = snprintf(prompt, sizeof(prompt),
                            "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
    if (pn < 0 || pn >= (int)sizeof(prompt)) {
        /* 截断会把 <|im_end|>/assistant 模板切掉（角色就错了），还会切碎 UTF-8。
         * 明确拒绝，不要带着半截 prompt 继续推理。 */
        fprintf(stderr, "[输入太长：模板需要 %d 字节，缓冲只有 %d；请分几次说]\n",
                pn, (int)sizeof(prompt));
        return 0;
    }

    const int cap = (int)strlen(user) + 64;
    int *ids = (int *)xmalloc(sizeof(int) * (size_t)cap);
    const int n = feng_tok_encode(&s->tok, prompt, ids, cap);
    if (n <= 0 || n + max_new + 2 > ctx) {
        fprintf(stderr, "[输入太长或编码为空：%d tokens / ctx %d]\n", n, ctx);
        free(ids);
        return 0;
    }
    if (s->kv.len > 0 && s->kv.len + n + max_new + 2 > ctx) {
        printf("[上下文已满 %d/%d，自动开始新对话]\n", s->kv.len, ctx);
        s->kv.len = 0;
        feng_mem_clear();
        mem_store_save(s);
    }

    int pos = s->kv.len;
    float *logits = NULL;
    const double p0 = now_ms();
    for (int i = 0; i < n; i++) {          /* prefill：中间 token 跳过 lm head */
        logits = feng_forward_ex(&s->m, &s->kv, &s->ws, ids[i], pos++, i + 1 == n);
        if (!logits) break;
    }
    const double p1 = now_ms();
    free(ids);

    int hist[64];
    int nhist = 0, gen = 0;
    int gen_cap = max_new;
    if (gen_cap > ctx - pos - 2) gen_cap = ctx - pos - 2;
    for (int step = 0; step < gen_cap && logits; step++) {
        const int tk = feng_sample_greedy(logits, v, hist, nhist, 1.15f, 3);
        if (tk < 0 || tk == s->tok.id_im_end || tk == s->tok.id_eot) break;
        char b[128];
        const int nb = feng_tok_decode_token(&s->tok, tk, b, sizeof(b));
        if (out && olen && *olen + nb < out_sz - 1) {
            memcpy(out + *olen, b, (size_t)nb);
            *olen += nb;
        }
        if (nhist < 64) hist[nhist++] = tk;
        else { memmove(hist, hist + 1, sizeof(hist) - sizeof(hist[0])); hist[63] = tk; }
        logits = feng_forward(&s->m, &s->kv, &s->ws, tk, pos++);
        gen++;
    }
    const double p2 = now_ms();
    /* 补进 KV 的 im_end / 换行不需要 logits，跳过 lm head；只有还有空间才写。 */
    if (pos < ctx) (void)feng_forward_ex(&s->m, &s->kv, &s->ws, s->tok.id_im_end, pos++, 0);
    if (s->nl_id > 0 && pos < ctx) (void)feng_forward_ex(&s->m, &s->kv, &s->ws, s->nl_id, pos++, 0);
    s->kv.len = pos;

    if (t_pre_ms) *t_pre_ms = p1 - p0;
    if (t_dec_ms) *t_dec_ms = p2 - p1;
    return gen;
}

/* 预 tokenized 输入：给 32k 长文验收用，绕过 4KB prompt 缓冲和 O(n²) BPE。
 * ids 文件是空白分隔的整数（完整 chat template，含 assistant 起始标记）。 */
static int session_generate_ids(session_t *s, const int *ids, int n, int max_new,
                                char *out, int out_sz, int *olen,
                                double *t_pre_ms, double *t_dec_ms)
{
    const int v = s->m.hdr.vocab;
    const int ctx = s->ctx;
    if (olen) *olen = 0;
    if (out && out_sz > 0) out[0] = 0;          /* 提前返回时 out 不是未初始化栈 */
    if (n <= 0 || n + max_new + 2 > ctx) {
        fprintf(stderr, "[ids 太长或为空：%d tokens / ctx %d]\n", n, ctx);
        return 0;
    }
    s->kv.len = 0;
    int pos = 0;
    float *logits = NULL;
    const double p0 = now_ms();
    for (int i = 0; i < n; i++) {
        logits = feng_forward_ex(&s->m, &s->kv, &s->ws, ids[i], pos++, i + 1 == n);
        if (!logits) break;
    }
    const double p1 = now_ms();

    int hist[64];
    int nhist = 0, gen = 0;
    int gen_cap = max_new;
    if (gen_cap > ctx - pos - 2) gen_cap = ctx - pos - 2;
    for (int step = 0; step < gen_cap && logits; step++) {
        const int tk = feng_sample_greedy(logits, v, hist, nhist, 1.15f, 3);
        if (tk < 0 || tk == s->tok.id_im_end || tk == s->tok.id_eot) break;
        char b[128];
        const int nb = feng_tok_decode_token(&s->tok, tk, b, sizeof(b));
        if (out && olen && *olen + nb < out_sz - 1) {
            memcpy(out + *olen, b, (size_t)nb);
            *olen += nb;
        }
        if (nhist < 64) hist[nhist++] = tk;
        else { memmove(hist, hist + 1, sizeof(hist) - sizeof(hist[0])); hist[63] = tk; }
        logits = feng_forward(&s->m, &s->kv, &s->ws, tk, pos++);
        gen++;
    }
    if (out && olen) out[*olen] = 0;
    if (t_pre_ms) *t_pre_ms = p1 - p0;
    if (t_dec_ms) *t_dec_ms = now_ms() - p1;
    return gen;
}

static int load_ids_file(const char *path, int *ids, int cap)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int n = 0, v = 0;
    while (n < cap && fscanf(f, "%d", &v) == 1) ids[n++] = v;
    fclose(f);
    return n;
}

/* ---------------- 模式 ---------------- */

static void print_banner(const session_t *s, int threads)
{
    const feng_header_t *hd = &s->m.hdr;
    printf("feng-30m phone chat ｜ KV=%s block=%d ctx=%d (kv %.2f MB) ｜ %u 层 h=%u nh=%u ffn=%u vocab=%u\n",
           KV_MODE, FENG_KV_Q2 ? FENG_KV_Q2_BLOCK : 0, s->ctx,
           (double)feng_kv_bytes(&s->m, s->ctx) / 1048576.0,
           hd->n_layers, hd->hidden, hd->n_heads, hd->ffn, hd->vocab);
    printf("线程 %d/%d ｜ NEON %s ｜ 算式/记忆/时间/随机数走 tool，\\reset 清空，\\stats 看状态\n",
           threads, feng_phone_cpus(), feng_phone_neon_on() ? "开" : "关");
}

static int run_bench(session_t *s, int n_gen)
{
    const char *q = "你好，请用一句话介绍一下你自己。";
    char buf[2048];
    int olen = 0;
    double t_pre = 0, t_dec = 0;

    s->kv.len = 0;
    (void)session_generate(s, q, 8, NULL, 0, NULL, NULL, NULL);   /* 热身 */
    s->kv.len = 0;
    const int gen = session_generate(s, q, n_gen, buf, sizeof(buf), &olen, &t_pre, &t_dec);

    printf("bench: n_gen=%d（实际生成 %d）\n", n_gen, gen);
    printf("bench: prefill %.1f ms ｜ decode %.1f ms -> %.2f tok/s\n",
           t_pre, t_dec, t_dec > 0 ? (double)gen * 1000.0 / t_dec : 0.0);
    printf("bench: KV %.2f MB ｜ 线程 %d ｜ NEON %s ｜ ctx %d\n",
           (double)feng_kv_bytes(&s->m, s->ctx) / 1048576.0,
           feng_phone_threads(), feng_phone_neon_on() ? "开" : "关", s->ctx);
    if (gen > 0) {
        buf[olen] = 0;
        for (char *p = buf; *p; p++) if (*p == '\n') *p = ' ';
        printf("bench: 输出样例 「%s」\n", buf);
    }
    return 0;
}

/* NEON 开/关的自检：同一 prompt、同一进程，跑两遍比 logits。
 * 非 aarch64 平台上两次结果必然完全一致（都走标量）。 */
static int run_ab(session_t *s)
{
#if !FENG_PHONE_NEON
    printf("ab: 本平台编译期未启用 NEON 内核（FENG_PHONE_NEON=0），无对比对象\n");
    return 0;
#else
    const char *q = "水在标准大气压下的沸点是多少度？";
    const int v = s->m.hdr.vocab;
    float *snap_a = (float *)xmalloc(sizeof(float) * (size_t)v);
    float *snap_b = (float *)xmalloc(sizeof(float) * (size_t)v);

    for (int pass = 0; pass < 2; pass++) {
        feng_phone_set_neon(pass == 0 ? 0 : 1);
        s->kv.len = 0;
        char prompt[4096];
        snprintf(prompt, sizeof(prompt),
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", q);
        int ids[1024];
        const int n = feng_tok_encode(&s->tok, prompt, ids, 1024);
        float *lg = NULL;
        for (int i = 0; i < n; i++) lg = feng_forward_ex(&s->m, &s->kv, &s->ws, ids[i], i, 1);
        memcpy(pass == 0 ? snap_a : snap_b, lg, sizeof(float) * (size_t)v);
    }

    double maxd = 0.0, sum = 0.0;
    for (int i = 0; i < v; i++) {
        const double d = (double)snap_a[i] - (double)snap_b[i];
        const double ad = d < 0 ? -d : d;
        if (ad > maxd) maxd = ad;
        sum += ad;
    }
    const int am_a = feng_argmax(snap_a, v), am_b = feng_argmax(snap_b, v);
    printf("ab: NEON关 vs NEON开 ｜ max|diff|=%.3e ｜ mean|diff|=%.3e ｜ argmax %d vs %d ｜ %s\n",
           maxd, sum / (double)v, am_a, am_b, am_a == am_b ? "MATCH" : "MISMATCH");
    free(snap_a);
    free(snap_b);
    feng_phone_set_neon(1);
    return am_a == am_b ? 0 : 1;
#endif
}

static void usage(void)
{
    printf(
        "用法：phone_chat [选项] [model_export 目录]\n"
        "  --ctx N        上下文长度（默认 2048，上限 FENG_PHONE_MAX_CTX=%d）\n"
        "  --threads N    线程数（默认 min(4, CPU)，上限 %d）\n"
        "  --neon 0|1     NEON 内核开关（默认 1）\n"
        "  --max-new N    单轮最多生成 token（默认 96）\n"
        "  --prompt TEXT  问一句就退出\n"
        "  --ids-file F   预 tokenized 输入（空白分隔的 token id，长文验收用）\n"
        "  --bench [N]    基准模式：生成 N 个 token 并打印 tok/s（默认 32）\n"
        "  --ab           NEON 开/关 logits 自检\n"
        "  -h, --help     本帮助\n",
        FENG_PHONE_MAX_CTX, FENG_PHONE_MAX_THREADS);
}

int main(int argc, char **argv)
{
    const char *dir = ".";
    int ctx = 2048;
    int threads = 0;         /* 0 = 用默认 */
    int max_new = 96;
    int bench = -1;          /* >=0 表示基准模式 */
    int do_ab = 0;
    const char *one_shot = NULL;
    const char *ids_file = NULL;
    int neon = -1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
        else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (strcmp(a, "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(a, "--neon") == 0 && i + 1 < argc) neon = atoi(argv[++i]);
        else if (strcmp(a, "--max-new") == 0 && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (strcmp(a, "--prompt") == 0 && i + 1 < argc) one_shot = argv[++i];
        else if (strcmp(a, "--ids-file") == 0 && i + 1 < argc) ids_file = argv[++i];
        else if (strcmp(a, "--bench") == 0) {
            bench = 32;
            /* 只有紧跟着**数字**才当成 token 数；否则下一个参数是模型目录 */
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                bench = atoi(argv[++i]);
            }
            if (bench < 1) bench = 32;
        }
        else if (strcmp(a, "--ab") == 0) do_ab = 1;
        else if (a[0] == '-') { fprintf(stderr, "未知选项 %s\n", a); usage(); return 2; }
        else dir = a;
    }

    if (threads > 0) feng_phone_set_threads(threads);
    (void)feng_phone_init();
    if (neon >= 0) feng_phone_set_neon(neon);

    if (ctx < 64) ctx = 64;
    if (ctx > FENG_PHONE_MAX_CTX) {
        fprintf(stderr, "ctx %d 超过编译期上限 %d（用 -DFENG_PHONE_MAX_CTX=... 重编）\n",
                ctx, FENG_PHONE_MAX_CTX);
        return 2;
    }

    feng_tools_set_time(feng_phone_epoch);
    feng_tools_set_uptime(feng_phone_uptime_us);
    feng_smp_init();

    session_t s;
    if (session_open(&s, dir, ctx) != 0) return 1;

    if (do_ab) return run_ab(&s);
    if (bench >= 0) { print_banner(&s, feng_phone_threads()); return run_bench(&s, bench); }

    if (ids_file) {
        int *ids = (int *)xmalloc(sizeof(int) * (size_t)(ctx + 1));
        const int n = ids ? load_ids_file(ids_file, ids, ctx + 1) : -1;
        if (n <= 0) {
            fprintf(stderr, "读不到 token ids：%s\n", ids_file);
            free(ids);
            return 2;
        }
        char out[4096];
        int olen = 0;
        double t_pre = 0, t_dec = 0;
        const int gen = session_generate_ids(&s, ids, n, max_new, out, sizeof(out),
                                             &olen, &t_pre, &t_dec);
        printf("%s\n[%d tok ｜ prefill %.0f ms ｜ decode %.0f ms ｜ %.1f tok/s]\n",
               out, gen, t_pre, t_dec, t_dec > 0 ? (double)gen * 1000.0 / t_dec : 0.0);
        free(ids);
        return gen > 0 ? 0 : 1;
    }

    if (one_shot) {
        if (feng_tok_has_special(one_shot)) {
            fprintf(stderr, "[输入包含保留的对话模板标记，已拒绝]\n");
            return 2;
        }
        /* 与 REPL / APK 同一条 tool 链：--prompt 是脚本化入口，不能绕过
         * 算式/记忆/时间/随机数，否则测出来的行为与产品路径不一致。 */
        char tool_reply[256];
        (void)feng_mem_learn(one_shot);     /* 先记事实，tool/模型都能用 */
        if (feng_calc_answer(one_shot, tool_reply, sizeof(tool_reply)) ||
            feng_mem_answer(one_shot, tool_reply, sizeof(tool_reply)) ||
            feng_time_answer(one_shot, tool_reply, sizeof(tool_reply)) ||
            feng_random_answer(one_shot, tool_reply, sizeof(tool_reply))) {
            if (mem_changed()) mem_store_save(&s);
            printf("[tool] %s\n", tool_reply);
            return 0;
        }

        char out[4096];
        int olen = 0;
        double t_pre = 0, t_dec = 0;
        s.kv.len = 0;
        const int gen = session_generate(&s, one_shot, max_new, out, sizeof(out), &olen, &t_pre, &t_dec);
        if (mem_changed()) mem_store_save(&s);
        out[olen] = 0;
        fputs(out, stdout);
        printf("\n[%d tok ｜ decode %.0f ms ｜ %.1f tok/s]\n", gen, t_dec,
               t_dec > 0 ? (double)gen * 1000.0 / t_dec : 0.0);
        return gen > 0 ? 0 : 1;      /* 拒答/空生成要让脚本能看出失败 */
    }

    print_banner(&s, feng_phone_threads());
    char line[2048];
    for (;;) {
        printf("you> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        int n = (int)strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (n == 0) continue;
        if (feng_tok_has_special(line)) {
            printf("[输入包含保留的对话模板标记，已拒绝]\n");
            continue;
        }
        if (strcmp(line, "\\exit") == 0 || strcmp(line, "\\quit") == 0) break;
        if (strcmp(line, "\\reset") == 0) {
            s.kv.len = 0;
            feng_mem_clear();
            mem_store_save(&s);
            printf("(context cleared，记忆也清了；\\mem 查看)\n");
            continue;
        }
        if (strncmp(line, "\\mem", 4) == 0) {
            if (strncmp(line, "\\mem clear", 10) == 0) {
                feng_mem_clear();
                mem_store_save(&s);
                printf("(memory cleared)\n");
            } else {
                char snap[256];
                feng_mem_snapshot(snap, sizeof(snap));
                printf("memory: %s\n", snap);
            }
            continue;
        }
        if (strcmp(line, "\\stats") == 0) {
            printf("ctx %d/%d ｜ 线程 %d ｜ NEON %s ｜ KV %.2f MB ｜ model %.2f MB\n",
                   s.kv.len, s.ctx, feng_phone_threads(), feng_phone_neon_on() ? "开" : "关",
                   (double)feng_kv_bytes(&s.m, s.ctx) / 1048576.0, (double)s.mlen / 1048576.0);
            continue;
        }

        char calc_reply[256];
        (void)feng_mem_learn(line);        /* 先记事实，模型仍能看到这句 */
        if (feng_calc_answer(line, calc_reply, sizeof(calc_reply)) ||
            feng_mem_answer(line, calc_reply, sizeof(calc_reply)) ||
            feng_time_answer(line, calc_reply, sizeof(calc_reply)) ||
            feng_random_answer(line, calc_reply, sizeof(calc_reply))) {
            if (mem_changed()) mem_store_save(&s);
            printf("[tool] %s\n", calc_reply);
            continue;
        }

        char out[4096];
        int olen = 0;
        double t_pre = 0, t_dec = 0;
        const int gen = session_generate(&s, line, max_new, out, sizeof(out), &olen, &t_pre, &t_dec);
        if (mem_changed()) mem_store_save(&s);
        out[olen] = 0;
        printf("<< ");
        fputs(out, stdout);
        printf(" >>END (%d tok ｜ %.1f tok/s ｜ ctx %d/%d)\n", gen,
               t_dec > 0 ? (double)gen * 1000.0 / t_dec : 0.0, s.kv.len, s.ctx);
    }
    return 0;
}
