/* Host self-check: run the same inference core on the exported model.bin and compare the
 * final-position logits against the PyTorch reference (ref_logits.bin).  Also generates a
 * few tokens so the text output can be eyeballed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "feng.h"
#include "feng_tokenizer.h"

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "oom %zu\n", n); exit(1); }
    memset(p, 0, n);
    return p;
}

static unsigned char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)xmalloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    char path[512];
    size_t mlen = 0, tlen = 0, rlen = 0;
    snprintf(path, sizeof(path), "%s/model.bin", dir);
    unsigned char *mblob = read_file(path, &mlen);
    snprintf(path, sizeof(path), "%s/tokenizer.bin", dir);
    unsigned char *tblob = read_file(path, &tlen);
    snprintf(path, sizeof(path), "%s/ref_logits.bin", dir);
    unsigned char *ref = read_file(path, &rlen);

    feng_model_t m;
    if (feng_model_init(&m, mblob, mlen) != 0) {
        fprintf(stderr, "model init failed\n"); return 1;
    }
    printf("model: %u layers hidden %u heads %u x %u ffn %u vocab %u rope %.0f\n",
           m.hdr.n_layers, m.hdr.hidden, m.hdr.n_heads, m.hdr.head_dim, m.hdr.ffn, m.hdr.vocab,
           m.hdr.rope_theta);

    feng_tok_t tok;
    if (feng_tok_load(&tok, tblob, tlen) != 0) { fprintf(stderr, "tokenizer load failed\n"); return 1; }
    printf("tokenizer: vocab %d merges %d im_start=%d im_end=%d eot=%d\n", tok.vocab_size,
           tok.n_merges, tok.id_im_start, tok.id_im_end, tok.id_eot);

    const char *prompt = "你好";
    char text[512];
    snprintf(text, sizeof(text), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
    int ids[512];
    const int n = feng_tok_encode(&tok, text, ids, 512);
    printf("encode(\"%s\") -> %d tokens:", prompt, n);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");
    for (int i = 0; i < n; i++) {
        char b[128];
        int nb = feng_tok_decode_token(&tok, ids[i], b, sizeof(b));
        b[nb] = 0;
        printf("%s", b);
    }
    printf("\n");

    /* reference token ids (written by the exporter) */
    int ref_ids[512], n_ref = 0;
    snprintf(path, sizeof(path), "%s/ref_ids.json", dir);
    FILE *f = fopen(path, "rb");
    if (f) {
        char buf[4096];
        size_t got = fread(buf, 1, sizeof(buf) - 1, f);
        buf[got] = 0;
        fclose(f);
        const char *p = buf;
        while (*p && n_ref < 512) {
            while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
            if (!*p) break;
            ref_ids[n_ref++] = atoi(p);
            while (*p && *p != ',') p++;
        }
    }
    printf("ref ids: %d tokens\n", n_ref);

    feng_kv_t kv;
    kv.ctx = 512; kv.len = 0;
    kv.k_cache = (float *)xmalloc(sizeof(float) * m.hdr.n_layers * kv.ctx * m.hdr.hidden);
    kv.v_cache = (float *)xmalloc(sizeof(float) * m.hdr.n_layers * kv.ctx * m.hdr.hidden);
#if FENG_KV_INT8 || FENG_KV_Q2
    /* 量化 KV 模式需要 fp16 scale：数量由当前模式决定（int8 按 head，q2 按块） */
    const size_t n_sc = feng_kv_scale_slots(&m, kv.ctx);
    kv.k_scale = (uint16_t *)xmalloc(n_sc * 2);
    kv.v_scale = (uint16_t *)xmalloc(n_sc * 2);
#else
    kv.k_scale = NULL;
    kv.v_scale = NULL;
#endif
    feng_workspace_t ws;
    const int h = m.hdr.hidden, ff = m.hdr.ffn, v = m.hdr.vocab;
    ws.max_ctx = kv.ctx;
    ws.x = (float *)xmalloc(sizeof(float) * h);
    ws.xn = (float *)xmalloc(sizeof(float) * h);
    ws.q = (float *)xmalloc(sizeof(float) * h);
    ws.k = (float *)xmalloc(sizeof(float) * h);
    ws.v = (float *)xmalloc(sizeof(float) * h);
    ws.attn = (float *)xmalloc(sizeof(float) * h);
    ws.proj = (float *)xmalloc(sizeof(float) * h);   /* down/attn 投影都输出 hidden */
    ws.gate = (float *)xmalloc(sizeof(float) * ff);
    ws.up = (float *)xmalloc(sizeof(float) * ff);
    ws.ffn = (float *)xmalloc(sizeof(float) * ff);
    ws.logits = (float *)xmalloc(sizeof(float) * v);
    ws.scratch = (float *)xmalloc(sizeof(float) * ff);

    const int use_n = n_ref > 0 ? n_ref : n;
    const int *use_ids = n_ref > 0 ? ref_ids : ids;
    float *logits = NULL;
    for (int i = 0; i < use_n; i++) logits = feng_forward(&m, &kv, &ws, use_ids[i], i);

    const float *refl = (const float *)ref;
    const int ncmp = (int)(rlen / 4);
    if (ncmp < v) {
        fprintf(stderr, "ref_logits.bin 太短：%d floats，期望至少 %d\n", ncmp, v);
        return 3;
    }
    /* Q4 权重的量化误差实测约 2.3~2.7；0.35 是 fp32 路径的阈值。
     * 默认取 3.5 作为“实现错误”的门槛（garbage logits 会远大于它），
     * fp32 严格对拍可用 FENG_MAX_DIFF=0.35。 */
    double max_allowed = 3.5;
    if (const char *e = getenv("FENG_MAX_DIFF")) {
        const double x = atof(e);
        if (x > 0) max_allowed = x;
    }
    double maxdiff = 0, sum = 0;
    int nonfinite = 0;
    int argmax_c = feng_argmax(logits, v), argmax_r = feng_argmax(refl, v);
    for (int i = 0; i < v; i++) {
        /* NaN 必须显式拒绝：fabs(NaN) > maxdiff 恒为 false，旧实现会让污染数值
         * 以 max|diff|=0 的假象通过 MATCH。 */
        if (!isfinite(logits[i]) || !isfinite(refl[i])) { nonfinite++; continue; }
        const double d = fabs((double)logits[i] - (double)refl[i]);
        if (d > maxdiff) maxdiff = d;
        sum += d;
    }
    printf("logits check: n=%d max|diff|=%.4f (limit %.2f) mean|diff|=%.5f  "
           "argmax c=%d ref=%d %s\n", v, maxdiff, max_allowed, sum / v, argmax_c, argmax_r,
           argmax_c == argmax_r ? "MATCH" : "MISMATCH");
    if (nonfinite) {
        printf("logits check: %d 个非有限值（NaN/Inf），拒绝通过\n", nonfinite);
        return 2;
    }    if (argc > 2) {                      /* optional: dump C logits for the torch-side check */
        snprintf(path, sizeof(path), "%s", argv[2]);
        FILE *fo = fopen(path, "wb");
        if (!fo) {
            fprintf(stderr, "cannot open dump %s\n", path);
            return 2;
        }
        const size_t w = fwrite(logits, sizeof(float), (size_t)v, fo);
        if (fclose(fo) != 0 || w != (size_t)v) {
            fprintf(stderr, "dump 写入不完整：%s\n", path);
            return 2;
        }
        printf("dumped C logits -> %s\n", path);
    }

    printf("greedy continuation: ");
    for (int step = 0; step < 24; step++) {
        const int t = feng_argmax(logits, v);
        if (t == tok.id_im_end || t == tok.id_eot) break;
        char b[128];
        int nb = feng_tok_decode_token(&tok, t, b, sizeof(b));
        fwrite(b, 1, nb, stdout);
        fflush(stdout);
        logits = feng_forward(&m, &kv, &ws, t, use_n + step);
    }
    printf("\n");
    return (maxdiff < max_allowed && argmax_c == argmax_r) ? 0 : 2;
}
