/* Android JNI 薄壳：只做 JVM <-> C 引擎的搬运，推理逻辑一行都不在这里。
 *
 * 设计要点：
 *  - 引擎是 C23 且**单实例、非重入**（大量 .bss 静态缓冲），所以这里用一把互斥锁
 *    把 nativeAsk 串行化；Kotlin 侧同时只会有一个推理在跑。
 *  - 推理在**调用 JNI 的那个线程**上跑（Kotlin 的 Dispatchers.Default 协程线程），
 *    worker 线程只做 GEMV、不碰 JNI，因此不需要 AttachCurrentThread。
 *  - 每个 token 回调一次 Java 方法；每次 NewStringUTF 产生的 local ref 立刻删掉，
 *    否则 512 个 local ref 的表会被撑爆（长回答时会踩到）。 */
#include <jni.h>
#include <android/log.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "feng.h"
#include "feng_calc.h"
#include "feng_memory.h"
#include "feng_phone.h"
#include "feng_tokenizer.h"
#include "feng_tools.h"

#define TAG "feng_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

typedef struct {
    feng_model_t m;
    feng_tok_t tok;
    feng_kv_t kv;
    feng_workspace_t ws;
    const uint8_t *mblob;
    const uint8_t *tblob;
    size_t mlen, tlen;
    int ctx;
    int nl_id;
    int abort_flag;
    int closing;
    char mem_path[512];     /* 持久化记忆：模型目录下的 mem.txt */
} jni_session;

/* 引擎是进程级单例（scores / tokenizer 缓冲 / 线程池都是全局的），
 * 每会话一把锁保护不了它：两个句柄会同时写同一份全局状态。
 * 这里用进程级互斥，所有 native 入口串行。 */
static pthread_mutex_t g_engine_lock = PTHREAD_MUTEX_INITIALIZER;

/* 记忆持久化：与板端 NVS 共用 feng_mem_export/import 的文本格式。
 * 调用方一般已持有 lock；读写失败只记日志，不影响对话。 */
static void mem_store_load(jni_session *s)
{
    if (!s->mem_path[0]) return;
    FILE *f = fopen(s->mem_path, "rb");
    if (!f) return;
    char buf[1024];
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    if (feng_mem_import(buf))
        feng_mem_mark_clean();          /* 刚读进来的不算“待保存变更” */
    else
        LOGI("mem.txt 为空或解析失败，已忽略");
}

static void mem_store_save(jni_session *s)
{
    if (!s->mem_path[0]) return;
    char buf[1024];
    const int n = feng_mem_export(buf, sizeof(buf));
    if (n < 0) return;
    if (n == 0) { (void)remove(s->mem_path); feng_mem_mark_clean(); return; }
    FILE *f = fopen(s->mem_path, "wb");
    if (!f) { LOGE("写不了 %s", s->mem_path); return; }
    const size_t w = fwrite(buf, 1, (size_t)n, f);
    if (fclose(f) != 0 || w != (size_t)n) {
        LOGE("写 %s 不完整", s->mem_path);
        return;                          /* 保留 dirty，下次再试 */
    }
    feng_mem_mark_clean();
}

/* 从 model.bin 的绝对路径推出同目录的 mem.txt（Kotlin 把两个模型文件都解包到 filesDir）。 */
static int mem_path_from_model(jni_session *s, const char *model_path)
{
    const char *slash = strrchr(model_path, '/');
    const char *bslash = strrchr(model_path, '\\');
    if (bslash && (!slash || bslash > slash)) slash = bslash;
    const size_t dirlen = slash ? (size_t)(slash - model_path) : 0;
    if (dirlen + 9 >= sizeof(s->mem_path)) return -1;
    memcpy(s->mem_path, model_path, dirlen);
    memcpy(s->mem_path + dirlen, "/mem.txt", 9);
    return 0;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (p) memset(p, 0, n);
    return p;
}

/* 把一段以 NUL 结尾的 token/回答交给 Java，并立刻释放 local ref——
 * 一次回答可能回调上百次，不释放会把 512 条 local ref 的表撑爆。 */
static void call_string(JNIEnv *env, jobject cb, jmethodID mid, const char *s)
{
    if (!mid || !s) return;
    jstring js = (*env)->NewStringUTF(env, s);
    if (!js) {
        /* 非法 UTF-8（半截多字节字符）时 NewStringUTF 会返回 NULL：跳过这个 token，
         * 总比让整轮崩掉好。这里极少发生——tokenizer 是按字符边界切词的。 */
        (*env)->ExceptionClear(env);
        return;
    }
    (*env)->CallVoidMethod(env, cb, mid, js);
    (*env)->DeleteLocalRef(env, js);
}

/* ---- 原始字节 -> Modified UTF-8 ------------------------------------------
 * tokenizer 是 byte-level BPE：一个 token 可能只含 UTF-8 字符的一部分。
 * 旧实现逐 token 调 NewStringUTF，跨 token 的汉字/emoji 会乱码或丢字。
 * 这里先累积成完整 UTF-8 序列，再转成 JNI 要求的 Modified UTF-8
 * （NUL 用 C0 80，补充平面用代理对）。 */
static int u8_need(unsigned char b)
{
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}

/* 返回写入字节数；0 = 非法/截断序列。 */
static int u8_to_mutf8(const char *s, int n, char *out, int cap)
{
    if (n <= 0 || cap <= 0) return 0;
    const unsigned char b0 = (unsigned char)s[0];
    uint32_t cp;
    int need = u8_need(b0);
    if (need <= 0 || n < need) return 0;
    if (need == 1) cp = b0;
    else if (need == 2) cp = ((uint32_t)(b0 & 0x1F) << 6) | ((unsigned char)s[1] & 0x3Fu);
    else if (need == 3) cp = ((uint32_t)(b0 & 0x0F) << 12) |
                             ((uint32_t)((unsigned char)s[1] & 0x3F) << 6) |
                             ((unsigned char)s[2] & 0x3Fu);
    else cp = ((uint32_t)(b0 & 0x07) << 18) |
              ((uint32_t)((unsigned char)s[1] & 0x3F) << 12) |
              ((uint32_t)((unsigned char)s[2] & 0x3F) << 6) |
              ((unsigned char)s[3] & 0x3Fu);
    for (int i = 1; i < need; i++)
        if (((unsigned char)s[i] & 0xC0) != 0x80) return 0;
    if ((need == 2 && cp < 0x80) || (need == 3 && cp < 0x800) ||
        (need == 4 && (cp < 0x10000 || cp > 0x10FFFF))) return 0;
    if (cp >= 0xD800 && cp <= 0xDFFF) return 0;

    int o = 0;
    if (cp == 0) { if (cap < 2) return 0; out[o++] = (char)0xC0; out[o++] = (char)0x80; }
    else if (cp < 0x80) { if (cap < 1) return 0; out[o++] = (char)cp; }
    else if (cp < 0x800) {
        if (cap < 2) return 0;
        out[o++] = (char)(0xC0 | (cp >> 6));
        out[o++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        if (cap < 3) return 0;
        out[o++] = (char)(0xE0 | (cp >> 12));
        out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[o++] = (char)(0x80 | (cp & 0x3F));
    } else {
        if (cap < 6) return 0;
        const uint32_t v = cp - 0x10000;
        const uint32_t hi = 0xD800 + (v >> 10), lo = 0xDC00 + (v & 0x3FF);
        out[o++] = (char)(0xE0 | (hi >> 12));
        out[o++] = (char)(0x80 | ((hi >> 6) & 0x3F));
        out[o++] = (char)(0x80 | (hi & 0x3F));
        out[o++] = (char)(0xE0 | (lo >> 12));
        out[o++] = (char)(0x80 | ((lo >> 6) & 0x3F));
        out[o++] = (char)(0x80 | (lo & 0x3F));
    }
    return o;
}

/* 把一段可能跨字符边界的原始字节交给 Java；不完整序列留到下一次。 */
typedef struct { char buf[8]; int n; } u8_pending_t;

static void emit_raw(JNIEnv *env, jobject cb, jmethodID mid,
                     u8_pending_t *p, const char *s, int n)
{
    for (int i = 0; i < n; i++) {
        if (p->n >= (int)sizeof(p->buf)) { p->n = 0; }   /* 病态输入：丢弃 */
        p->buf[p->n++] = s[i];
        for (;;) {
            if (p->n <= 0) break;
            const int need = u8_need((unsigned char)p->buf[0]);
            if (need <= 0) { memmove(p->buf, p->buf + 1, (size_t)--p->n); continue; }
            if (p->n < need) break;
            char m[8];
            const int mlen = u8_to_mutf8(p->buf, p->n, m, (int)sizeof(m));
            if (mlen > 0) { m[mlen] = 0; call_string(env, cb, mid, m); }
            memmove(p->buf, p->buf + need, (size_t)(p->n - need));
            p->n -= need;
        }
    }
}

/* 一轮问答：tool 优先（算式 → 记忆 → 时间 → 随机数，与固件/PC 同序），否则走模型。
 * 返回生成的 token 数；-1 表示参数错误。 */
JNIEXPORT jint JNICALL
Java_com_feng_feng30m_FengNative_nativeAsk(JNIEnv *env, jclass cls, jlong handle,
                                           jstring jtext, jint maxNew, jobject callback)
{
    (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s || !jtext) return -1;

    const char *text = (*env)->GetStringUTFChars(env, jtext, NULL);
    if (!text) return -1;

    jclass cbCls = (*env)->GetObjectClass(env, callback);
    jmethodID midToken = cbCls ? (*env)->GetMethodID(env, cbCls, "onToken", "(Ljava/lang/String;)V") : NULL;
    jmethodID midDone = cbCls ? (*env)->GetMethodID(env, cbCls, "onDone", "(IJ)V") : NULL;
    if (cbCls) (*env)->DeleteLocalRef(env, cbCls);

    pthread_mutex_lock(&g_engine_lock);
    if (s->closing) {
        pthread_mutex_unlock(&g_engine_lock);
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return -1;
    }
    s->abort_flag = 0;

    char reply[512];
    int gen = 0;
    double pre_ms = 0, dec_ms = 0;

    if (feng_tok_has_special(text)) {
        call_string(env, callback, midToken, "输入包含保留的对话模板标记，已拒绝");
        if (midDone) (*env)->CallVoidMethod(env, callback, midDone, (jint)0, (jlong)0);
        pthread_mutex_unlock(&g_engine_lock);
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return 0;
    }

    const int learned = feng_mem_learn(text);
    if (feng_calc_answer(text, reply, sizeof(reply)) ||
        feng_mem_answer(text, reply, sizeof(reply)) ||
        feng_time_answer(text, reply, sizeof(reply)) ||
        feng_random_answer(text, reply, sizeof(reply))) {
        /* tool 秒回：一次回调交完，gen 记 0 让界面知道这不是模型生成的 */
        if (learned || feng_mem_dirty()) mem_store_save(s);
        call_string(env, callback, midToken, reply);
        if (midDone) (*env)->CallVoidMethod(env, callback, midDone, (jint)0, (jlong)0);
        pthread_mutex_unlock(&g_engine_lock);
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return 0;
    }

    const int v = s->m.hdr.vocab;
    char prompt[4096];
    const int pn = snprintf(prompt, sizeof(prompt),
                            "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", text);
    if (pn < 0 || pn >= (int)sizeof(prompt)) {
        /* 截断会把 im_end / assistant 模板切掉，生成角色就错了。 */
        call_string(env, callback, midToken, "输入太长，请分几次说。");
        if (midDone) (*env)->CallVoidMethod(env, callback, midDone, (jint)0, (jlong)0);
        pthread_mutex_unlock(&g_engine_lock);
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return 0;
    }

    const int cap = (int)strlen(text) + 64;
    int *ids = (int *)xmalloc(sizeof(int) * (size_t)cap);
    const int n = ids ? feng_tok_encode(&s->tok, prompt, ids, cap) : 0;
    if (!ids || n <= 0 || n + maxNew + 2 > s->ctx) {
        free(ids);
        call_string(env, callback, midToken, "这轮输入超出了当前上下文，请缩短或先清空。");
        if (midDone) (*env)->CallVoidMethod(env, callback, midDone, (jint)0, (jlong)0);
        pthread_mutex_unlock(&g_engine_lock);
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return 0;
    }
    if (s->kv.len > 0 && s->kv.len + n + maxNew + 2 > s->ctx) {
        s->kv.len = 0;                 /* 上下文满：自动开新对话（与 CLI 同口径） */
        feng_mem_clear();
        mem_store_save(s);
    }

    int pos = s->kv.len;
    float *logits = NULL;
    const double t0 = (double)feng_phone_uptime_us();
    for (int i = 0; i < n; i++) {
        logits = feng_forward_ex(&s->m, &s->kv, &s->ws, ids[i], pos++, i + 1 == n);
        if (!logits) break;
    }
    const double t1 = (double)feng_phone_uptime_us();
    free(ids);

    int hist[64];
    int nhist = 0;
    u8_pending_t pend = {{0}, 0};
    int gen_cap = maxNew;
    if (gen_cap > s->ctx - pos - 2) gen_cap = s->ctx - pos - 2;
    for (int step = 0; step < gen_cap && logits; step++) {
        if (__atomic_load_n(&s->abort_flag, __ATOMIC_RELAXED)) break;
        const int tk = feng_sample_greedy(logits, v, hist, nhist, 1.15f, 3);
        if (tk < 0 || tk == s->tok.id_im_end || tk == s->tok.id_eot) break;
        char b[128];
        const int nb = feng_tok_decode_token(&s->tok, tk, b, sizeof(b));
        if (nb > 0 && nb < (int)sizeof(b))
            emit_raw(env, callback, midToken, &pend, b, nb);   /* 跨 token 拼完整字符 */
        if (nhist < 64) hist[nhist++] = tk;
        else { memmove(hist, hist + 1, sizeof(hist) - sizeof(hist[0])); hist[63] = tk; }
        logits = feng_forward(&s->m, &s->kv, &s->ws, tk, pos++);
        gen++;
    }
    const double t2 = (double)feng_phone_uptime_us();
    pre_ms = (t1 - t0) / 1000.0;
    dec_ms = (t2 - t1) / 1000.0;

    /* 把 im_end / 换行也补进 KV，下一轮才有正确上下文；只有还有空间才写 */
    if (pos < s->ctx)
        (void)feng_forward_ex(&s->m, &s->kv, &s->ws, s->tok.id_im_end, pos++, 0);
    if (s->nl_id > 0 && pos < s->ctx)
        (void)feng_forward_ex(&s->m, &s->kv, &s->ws, s->nl_id, pos++, 0);
    s->kv.len = pos;

    LOGI("ask: prompt %d tok, gen %d tok, prefill %.0f ms, decode %.0f ms (%.1f tok/s)",
         n, gen, pre_ms, dec_ms, dec_ms > 0 ? gen * 1000.0 / dec_ms : 0.0);

    if (midDone) (*env)->CallVoidMethod(env, callback, midDone, (jint)gen, (jlong)(dec_ms + 0.5));

    if (learned || feng_mem_dirty()) mem_store_save(s);
    pthread_mutex_unlock(&g_engine_lock);
    (*env)->ReleaseStringUTFChars(env, jtext, text);
    return (jint)gen;
}

JNIEXPORT void JNICALL
Java_com_feng_feng30m_FengNative_nativeAbort(JNIEnv *env, jclass cls, jlong handle)
{
    (void)env; (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (s) __atomic_store_n(&s->abort_flag, 1, __ATOMIC_RELAXED);   /* decode 循环每步检查一次 */
}

JNIEXPORT void JNICALL
Java_com_feng_feng30m_FengNative_nativeReset(JNIEnv *env, jclass cls, jlong handle)
{
    (void)env; (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s) return;
    pthread_mutex_lock(&g_engine_lock);
    s->kv.len = 0;
    feng_mem_clear();
    mem_store_save(s);
    pthread_mutex_unlock(&g_engine_lock);
}

JNIEXPORT jstring JNICALL
Java_com_feng_feng30m_FengNative_nativeMemory(JNIEnv *env, jclass cls, jlong handle)
{
    (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s) return NULL;
    char snap[256];
    pthread_mutex_lock(&g_engine_lock);
    feng_mem_snapshot(snap, sizeof(snap));
    pthread_mutex_unlock(&g_engine_lock);
    return (*env)->NewStringUTF(env, snap);
}

JNIEXPORT jstring JNICALL
Java_com_feng_feng30m_FengNative_nativeStats(JNIEnv *env, jclass cls, jlong handle)
{
    (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s) return NULL;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "ctx %d/%d ｜ 线程 %d/%d ｜ NEON %s ｜ KV %.2f MB",
             s->kv.len, s->ctx, feng_phone_threads(), feng_phone_cpus(),
             feng_phone_neon_on() ? "开" : "关",
             (double)feng_kv_bytes(&s->m, s->ctx) / 1048576.0);
    return (*env)->NewStringUTF(env, buf);
}

JNIEXPORT jstring JNICALL
Java_com_feng_feng30m_FengNative_nativeBackend(JNIEnv *env, jclass cls, jlong handle)
{
    (void)cls; (void)handle;
    char buf[128];
    snprintf(buf, sizeof(buf), "NEON %s ｜ 线程 %d/%d",
             feng_phone_neon_on() ? "开" : "关", feng_phone_threads(), feng_phone_cpus());
    return (*env)->NewStringUTF(env, buf);
}

/* 统一释放：成功路径与初始化失败路径共用，避免 32k 分配失败时泄漏 154MB KV。 */
static void session_free(jni_session *s)
{
    if (!s) return;
    feng_phone_unload(s->mblob, s->mlen);
    feng_phone_unload(s->tblob, s->tlen);
    free(s->kv.k_cache);
    free(s->kv.v_cache);
    free(s->kv.k_scale);
    free(s->kv.v_scale);
    free(s->ws.x); free(s->ws.xn); free(s->ws.q); free(s->ws.k); free(s->ws.v);
    free(s->ws.attn); free(s->ws.proj); free(s->ws.gate); free(s->ws.up);
    free(s->ws.ffn); free(s->ws.logits); free(s->ws.scratch);
    feng_tok_free(&s->tok);
    free(s);
}

JNIEXPORT void JNICALL
Java_com_feng_feng30m_FengNative_nativeFree(JNIEnv *env, jclass cls, jlong handle)
{
    (void)env; (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s) return;
    /* 拿进程锁 = 等在途 ask/reset 结束；置 closing 后不会再接新请求。
     * 旧实现直接 free，生成中关闭就是 use-after-free。 */
    pthread_mutex_lock(&g_engine_lock);
    s->closing = 1;
    session_free(s);
    pthread_mutex_unlock(&g_engine_lock);
}

/* 初始化：加载模型 + tokenizer、分配 KV/workspace、拉起线程池。
 * 返回会话句柄；失败返回 0（Kotlin 侧抛出并显示原因）。 */
JNIEXPORT jlong JNICALL
Java_com_feng_feng30m_FengNative_nativeInit(JNIEnv *env, jclass cls, jstring jmodel,
                                            jstring jtok, jint ctx, jint threads)
{
    (void)cls;
    const char *mp = (*env)->GetStringUTFChars(env, jmodel, NULL);
    const char *tp = (*env)->GetStringUTFChars(env, jtok, NULL);
    if (!mp || !tp) return 0;

    if (threads > 0) feng_phone_set_threads(threads);
    feng_phone_init();
    feng_tools_set_time(feng_phone_epoch);
    feng_tools_set_uptime(feng_phone_uptime_us);
    feng_smp_init();

    jni_session *s = (jni_session *)xmalloc(sizeof(jni_session));
    if (!s) goto fail;

    if (ctx < 256) ctx = 2048;
    if (ctx > FENG_PHONE_MAX_CTX) {
        LOGE("ctx %d 超过编译期上限 %d，已截断", ctx, FENG_PHONE_MAX_CTX);
        ctx = FENG_PHONE_MAX_CTX;
    }
    s->ctx = ctx;
    if (mem_path_from_model(s, mp) != 0) {
        LOGE("模型路径太长，记忆持久化关闭：%s", mp);
        s->mem_path[0] = 0;
    }

    if (feng_phone_load(mp, &s->mblob, &s->mlen) != 0) {
        LOGE("打不开 %s", mp); goto fail;
    }
    if (feng_phone_load(tp, &s->tblob, &s->tlen) != 0) {
        LOGE("打不开 %s", tp); goto fail;
    }
    if (feng_model_init(&s->m, s->mblob, s->mlen) != 0) {
        LOGE("model.bin 解析失败"); goto fail;
    }
    if (feng_tok_load(&s->tok, s->tblob, s->tlen) != 0) {
        LOGE("tokenizer.bin 解析失败"); goto fail;
    }

    const size_t kv_bytes = feng_kv_bytes(&s->m, ctx);      /* 显示用总量 */
    const size_t kv_data = feng_kv_data_bytes(&s->m, ctx);  /* 单 cache 数据，不含 scale */
    s->kv.ctx = ctx;
    s->kv.len = 0;
    s->kv.k_cache = xmalloc(kv_data + 64);
    s->kv.v_cache = xmalloc(kv_data + 64);
#if FENG_KV_INT8 || FENG_KV_Q2
    const size_t n_sc = feng_kv_scale_slots(&s->m, ctx);
    s->kv.k_scale = (uint16_t *)xmalloc(n_sc * 2);
    s->kv.v_scale = (uint16_t *)xmalloc(n_sc * 2);
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
    ws->proj = (float *)xmalloc(sizeof(float) * h);   /* 输出 hidden，不是 ffn */
    ws->gate = (float *)xmalloc(sizeof(float) * ff);
    ws->up = (float *)xmalloc(sizeof(float) * ff);
    ws->ffn = (float *)xmalloc(sizeof(float) * ff);
    ws->logits = (float *)xmalloc(sizeof(float) * v);
    ws->scratch = (float *)xmalloc(sizeof(float) * (ctx > ff ? ctx : ff));

#if FENG_KV_INT8 || FENG_KV_Q2
    const int scales_ok = (s->kv.k_scale && s->kv.v_scale);
#else
    const int scales_ok = 1;      /* fp32/fp16：没有 scale，NULL 是正常的 */
#endif
    /* 每个 workspace 指针都要检查：漏一个就会在首轮推理里空指针写。 */
    if (!s->kv.k_cache || !s->kv.v_cache || !scales_ok ||
        !ws->x || !ws->xn || !ws->q || !ws->k || !ws->v || !ws->attn ||
        !ws->proj || !ws->gate || !ws->up || !ws->ffn ||
        !ws->logits || !ws->scratch) {
        LOGE("内存不足"); goto fail;
    }

    { int one = 0; s->nl_id = (feng_tok_encode(&s->tok, "\n", &one, 1) == 1) ? one : 0; }
    mem_store_load(s);

    LOGI("初始化完成：%u 层 / h=%u / vocab=%u ｜ ctx=%d ｜ KV %.2f MB ｜ 线程 %d ｜ NEON %s",
         s->m.hdr.n_layers, s->m.hdr.hidden, s->m.hdr.vocab, ctx,
         kv_bytes / 1048576.0, feng_phone_threads(), feng_phone_neon_on() ? "on" : "off");

    (*env)->ReleaseStringUTFChars(env, jmodel, mp);
    (*env)->ReleaseStringUTFChars(env, jtok, tp);
    return (jlong)(intptr_t)s;

fail:
    session_free(s);
    (*env)->ReleaseStringUTFChars(env, jmodel, mp);
    (*env)->ReleaseStringUTFChars(env, jtok, tp);
    return 0;
}
