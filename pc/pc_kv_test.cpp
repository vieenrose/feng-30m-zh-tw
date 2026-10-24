/* KV 量化对比测试（PC）：喂一段**长 prompt**，再用同一套贪心解码生成 N 个 token。
 *
 * 同一份源码分别在三种 KV 模式下编译（fp32 / FENG_KV_INT8=1 / FENG_KV_Q2=1），
 * 输出应当一致或高度接近 —— 这是"改 KV 量化后没有把模型搞坏"的回归依据。
 *
 * 用法：pc_kv_test <model_export 目录> <prompt.txt> [max_new] [prompt_chars]
 *       prompt_chars > 0 时只取 prompt 的前 N 个字符（用来扫描"上下文长度 vs 质量"）
 * 输出：模式、KV 字节数、prompt token 数、生成文本
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    unsigned char *buf = (unsigned char *)xmalloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    buf[n] = 0;
    fclose(f);
    *len = (size_t)n;
    return buf;
}

#if FENG_KV_Q2
#define KV_MODE "q2"
#elif FENG_KV_INT8
#define KV_MODE "int8"
#elif FENG_KV_FP16
#define KV_MODE "fp16"
#else
#define KV_MODE "fp32"
#endif

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    const char *prompt_file = argc > 2 ? argv[2] : "pc/prompt_long.txt";
    const int max_new = argc > 3 ? atoi(argv[3]) : 64;
    const int prompt_chars = argc > 4 ? atoi(argv[4]) : 0;
    const int ctx = 2048;
    char path[512];
    size_t mlen = 0, tlen = 0, plen = 0;

    snprintf(path, sizeof(path), "%s/model.bin", dir);
    unsigned char *mblob = read_file(path, &mlen);
    snprintf(path, sizeof(path), "%s/tokenizer.bin", dir);
    unsigned char *tblob = read_file(path, &tlen);
    unsigned char *ptext = read_file(prompt_file, &plen);
    if (prompt_chars > 0 && (size_t)prompt_chars < plen) {
        ptext[prompt_chars] = 0;          /* 截断到指定字符数（UTF-8 边界对齐由调用者保证） */
        plen = (size_t)prompt_chars;
    }

    feng_model_t m;
    if (feng_model_init(&m, mblob, mlen) != 0) {
        fprintf(stderr, "model init failed\n"); return 1;
    }
    feng_tok_t tok;
    if (feng_tok_load(&tok, tblob, tlen) != 0) { fprintf(stderr, "tokenizer load failed\n"); return 1; }

    char *text = (char *)xmalloc(plen + 256);
    snprintf(text, plen + 256, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             (const char *)ptext);
    int *ids = (int *)xmalloc(sizeof(int) * (plen + 64));
    const int n = feng_tok_encode(&tok, text, ids, (int)(plen + 64));

    feng_kv_t kv;
    memset(&kv, 0, sizeof(kv));
    kv.ctx = ctx; kv.len = 0;
    const size_t kv_bytes = feng_kv_bytes(&m, ctx);      /* 显示用总量（含 scale） */
    const size_t kv_data = feng_kv_data_bytes(&m, ctx);  /* 单个 cache 的数据，不含 scale */
    kv.k_cache = xmalloc(kv_data + 64);
    kv.v_cache = xmalloc(kv_data + 64);
#if FENG_KV_INT8 || FENG_KV_Q2
    const size_t n_sc = feng_kv_scale_slots(&m, ctx);
    kv.k_scale = (uint16_t *)xmalloc(n_sc * 2);
    kv.v_scale = (uint16_t *)xmalloc(n_sc * 2);
#endif

    feng_workspace_t ws;
    const int h = m.hdr.hidden, ff = m.hdr.ffn, v = m.hdr.vocab;
    ws.max_ctx = ctx;
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
    ws.scratch = (float *)xmalloc(sizeof(float) * (ctx > ff ? ctx : ff));

    if (n >= ctx - max_new) { fprintf(stderr, "prompt too long: %d tokens\n", n); return 1; }
    float *logits = NULL;
    for (int i = 0; i < n; i++) logits = feng_forward(&m, &kv, &ws, ids[i], i);

    printf("mode=%s  ctx=%d  kv=%.2f MB  prompt_tokens=%d  gen=%d\n",
           KV_MODE, ctx, kv_bytes / 1048576.0, n, max_new);
    printf("OUT: ");
    for (int step = 0; step < max_new; step++) {
        const int t = feng_argmax(logits, v);
        if (t == tok.id_im_end || t == tok.id_eot) break;
        char b[128];
        const int nb = feng_tok_decode_token(&tok, t, b, sizeof(b));
        fwrite(b, 1, nb, stdout);
        logits = feng_forward(&m, &kv, &ws, t, n + step);
    }
    printf("\n");
    return 0;
}
