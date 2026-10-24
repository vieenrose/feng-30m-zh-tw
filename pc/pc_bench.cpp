/* PC benchmark of the *same* C engine the ESP32 runs (single thread, scalar Q4
 * kernels), so the PC number is directly comparable with the board's tok/s. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

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

static double now_s(void)
{
    const auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "..\\model_export_planA3b";
    const char *question = argc > 2 ? argv[2] : "你好";
    const int n_gen = argc > 3 ? atoi(argv[3]) : 32;
    char path[512];
    size_t mlen = 0, tlen = 0;
    snprintf(path, sizeof(path), "%s/model.bin", dir);
    unsigned char *mblob = read_file(path, &mlen);
    snprintf(path, sizeof(path), "%s/tokenizer.bin", dir);
    unsigned char *tblob = read_file(path, &tlen);

    feng_model_t m;
    if (feng_model_init(&m, mblob, mlen) != 0) {
        fprintf(stderr, "model init failed\n"); return 1;
    }
    feng_tok_t tok;
    if (feng_tok_load(&tok, tblob, tlen) != 0) { fprintf(stderr, "tokenizer load failed\n"); return 1; }
    printf("model: %u layers hidden %u ffn %u vocab %u | %.2f MB on disk\n",
           m.hdr.n_layers, m.hdr.hidden, m.hdr.ffn, m.hdr.vocab, (double)mlen / 1048576.0);

    char text[1024];
    snprintf(text, sizeof(text), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", question);
    int ids[1024];
    const int np = feng_tok_encode(&tok, text, ids, 1024);
    printf("prompt \"%s\" -> %d tokens | generating %d tokens (greedy, rep 1.15)\n",
           question, np, n_gen);

    feng_kv_t kv = {0};
    kv.ctx = 512;
#if FENG_KV_INT8 || FENG_KV_Q2
#if FENG_KV_Q2
    kv.k_cache = xmalloc((size_t)m.hdr.n_layers * kv.ctx * (m.hdr.hidden / 4));
    kv.v_cache = xmalloc((size_t)m.hdr.n_layers * kv.ctx * (m.hdr.hidden / 4));
#else
    kv.k_cache = xmalloc((size_t)m.hdr.n_layers * kv.ctx * m.hdr.hidden);
    kv.v_cache = xmalloc((size_t)m.hdr.n_layers * kv.ctx * m.hdr.hidden);
#endif
    const size_t n_sc = feng_kv_scale_slots(&m, kv.ctx);
    kv.k_scale = (uint16_t *)xmalloc(n_sc * 2);
    kv.v_scale = (uint16_t *)xmalloc(n_sc * 2);
#else
    kv.k_cache = (float *)xmalloc(sizeof(float) * m.hdr.n_layers * kv.ctx * m.hdr.hidden);
    kv.v_cache = (float *)xmalloc(sizeof(float) * m.hdr.n_layers * kv.ctx * m.hdr.hidden);
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
    ws.proj = (float *)xmalloc(sizeof(float) * h);   /* attn/down 投影都输出 hidden */
    ws.gate = (float *)xmalloc(sizeof(float) * ff);
    ws.up = (float *)xmalloc(sizeof(float) * ff);
    ws.ffn = (float *)xmalloc(sizeof(float) * ff);
    ws.logits = (float *)xmalloc(sizeof(float) * v);
    ws.scratch = (float *)xmalloc(sizeof(float) * (ff > kv.ctx ? ff : kv.ctx));

    float *logits = NULL;
    const double t0 = now_s();
    for (int i = 0; i < np; i++) {             /* prefill：中间 token 跳过 lm head */
        logits = feng_forward_ex(&m, &kv, &ws, ids[i], i, i + 1 == np);
    }
    const double t_pre = now_s() - t0;

    int hist[64], nh = 0, pos = np;
    char out[4096];
    int written = 0;
    const double t1 = now_s();
    for (int n = 0; n < n_gen; n++) {
        for (int i = 0; i < nh; i++) {
            const int t = hist[i];
            logits[t] = logits[t] > 0 ? logits[t] / 1.15f : logits[t] * 1.15f;
        }
        const int next_id = feng_argmax(logits, v);
        if (next_id == tok.id_im_end || next_id == tok.id_eot) break;
        char tmp[128];
        const int nb = feng_tok_decode_token(&tok, next_id, tmp, sizeof(tmp));
        if (nb > 0 && written + nb < (int)sizeof(out) - 1) {
            memcpy(out + written, tmp, nb);
            written += nb;
        }
        if (nh < 64) hist[nh++] = next_id;
        else { memmove(hist, hist + 1, sizeof(hist) - sizeof(int)); hist[63] = next_id; }
        logits = feng_forward(&m, &kv, &ws, next_id, pos++);
    }
    const double t2 = now_s() - t1;
    out[written] = 0;
    const int gen = pos - np;
    printf("reply: %s\n", out);
    printf("prefill %d tok in %.1f ms (%.1f tok/s) | decode %d tok in %.1f ms -> %.2f tok/s (%.1f ms/token)\n",
           np, t_pre * 1000.0, np / t_pre, gen, t2 * 1000.0, gen / t2, t2 * 1000.0 / (gen ? gen : 1));
    return 0;
}
