/* 引擎入口模糊测试（ASAN + UBSAN 驱动）。
 *
 * 覆盖的是**不需要模型权重**的入口，因此可以在 CI 上无依赖地跑：
 *   calc   —— feng_calc_answer：中文数字 / 单位 / 括号 / 超长串 / 截断 UTF-8
 *   tools  —— feng_rand_range / feng_time_answer / feng_mem_answer：极端整数与畸形输入
 *   model  —— feng_model_init：随机/截断/自相矛盾的头部（不能崩、不能越界读）
 *
 * 为什么要有这个文件：2026-10-22 那轮安全加固抓到的 7 个真 bug（栈写穿 40 字节、
 * long long 溢出、8 张表全泄漏、线程池信号量虚高）全部由临时夹具发现，而夹具
 * 当时放在被 gitignore 的 .cache 里——**经验留不下来，下次照样漏**。收进仓库 +
 * 进 CI 才算真正修完。
 *
 * 用法：
 *   fuzz_main <calc|tools|model> [轮数] [seed]
 * 退出码 0 = 干净；非 0 = 有 sanitizer 报告或断言失败（CI 直接红）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "feng.h"
#include "feng_tools.h"
#include "feng_memory.h"
#include "feng_calc.h"

/* ---- 确定性 PRNG（xorshift64*）：同样的 seed 必得同样的用例，便于复现 ---- */
static uint64_t s_rng = 0x9E3779B97F4A7C15ull;

static uint32_t rnd32(void)
{
    s_rng ^= s_rng >> 12;
    s_rng ^= s_rng << 25;
    s_rng ^= s_rng >> 27;
    return (uint32_t)((s_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

static int rnd_range(int lo, int hi)          /* [lo, hi) */
{
    if (hi <= lo) return lo;
    return lo + (int)(rnd32() % (uint32_t)(hi - lo));
}

/* 生成一个随机 UTF-8 串：中文数字/单位/运算符/标点/非法字节混在一起。
 * 返回长度（不含结尾 NUL）。buf 容量 cap。 */
static int gen_cn_string(char *buf, int cap)
{
    static const char *toks[] = {
        "一", "二", "三", "五", "九", "十", "百", "千", "万", "亿", "零", "两",
        "加", "减", "乘", "除", "的", "平方", "立方", "根号", "括号",
        "(", ")", "+", "-", "*", "/", "^", "%", ".", " ", "\t", "\n",
        "0", "1", "7", "9", "12", "100", "9999", "12345678901234567890",
        "帮我算一下", "等于几", "结果", "？", "，", "、",
        "\xE4", "\xB8", "\x80",              /* 孤立/截断的 UTF-8 字节 */
        "\xF0\x9F\x98\x80",                   /* emoji */
    };
    const int nt = (int)(sizeof(toks) / sizeof(toks[0]));
    int n = 0;
    if (cap < 8) return 0;
    const int want = rnd_range(1, 24);
    for (int i = 0; i < want && n < cap - 8; i++) {
        const char *t = toks[rnd_range(0, nt)];
        const int len = (int)strlen(t);
        if (n + len >= cap - 1) break;
        memcpy(buf + n, t, (size_t)len);
        n += len;
    }
    buf[n] = 0;
    return n;
}

/* 极端用例：手工构造已知会把旧实现打穿的模式 */
static void run_calc_known_bad(char *ans, size_t ans_sz, int *fails)
{
    static char big[4096];
    struct { const char *name; int kind; } cases[] = {
        {"85 个九", 1}, {"84 个万 + 加一", 2}, {"前导空格 + 万连写", 3},
        {"1e20 平方", 4}, {"超长中文数字串", 5}, {"嵌套括号", 6},
    };
    for (int c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++) {
        memset(big, 0, sizeof(big));
        int p = 0;
        switch (cases[c].kind) {
        case 1:
            for (int i = 0; i < 85 && p + 3 < (int)sizeof(big); i++, p += 3) memcpy(big + p, "九", 3);
            break;
        case 2:
            for (int i = 0; i < 84 && p + 3 < (int)sizeof(big); i++, p += 3) memcpy(big + p, "万", 3);
            memcpy(big + p, "加一", 7);
            break;
        case 3:
            p = 3;
            memset(big, ' ', 3);
            for (int i = 0; i < 84 && p + 3 < (int)sizeof(big); i++, p += 3) memcpy(big + p, "万", 3);
            break;
        case 4:
            snprintf(big, sizeof(big), "%s", "100000000000000000000的平方");
            break;
        case 5:
            for (int i = 0; i < 300 && p + 3 < (int)sizeof(big); i++, p += 3) memcpy(big + p, "七", 3);
            break;
        default:
            for (int i = 0; i < 200; i++) {
                if (p + 1 < (int)sizeof(big)) big[p++] = '(';
            }
            if (p + 2 < (int)sizeof(big)) { big[p++] = '1'; big[p++] = '+'; big[p++] = '2'; }
            for (int i = 0; i < 200 && p + 1 < (int)sizeof(big); i++) big[p++] = ')';
            break;
        }
        const int r = feng_calc_answer(big, ans, (int)ans_sz);
        if (r != 0 && r != 1) {
            fprintf(stderr, "[FAIL] calc 已知用例 %s 返回 %d（只允许 0/1）\n", cases[c].name, r);
            (*fails)++;
        }
    }
}

static int fuzz_calc(int rounds, char *ans, size_t ans_sz)
{
    int fails = 0;
    run_calc_known_bad(ans, ans_sz, &fails);

    /* 入口防御：NULL / 零长 / 负长度 */
    if (feng_calc_answer(NULL, ans, (int)ans_sz) != 0) { fprintf(stderr, "[FAIL] NULL user 未拒绝\n"); fails++; }
    if (feng_calc_answer("1+1", NULL, 16) != 0) { fprintf(stderr, "[FAIL] NULL answer 未拒绝\n"); fails++; }
    if (feng_calc_answer("1+1", ans, 0) != 0) { fprintf(stderr, "[FAIL] answer_sz=0 未拒绝\n"); fails++; }
    if (feng_calc_answer("1+1", ans, -5) != 0) { fprintf(stderr, "[FAIL] answer_sz<0 未拒绝\n"); fails++; }

    for (int i = 0; i < rounds; i++) {
        char big[512];
        gen_cn_string(big, (int)sizeof(big));
        /* 精确分配：ASAN 才能在**越过 NUL**时立刻报错。以前的用例都写在
         * 512 字节大数组里，红区永远碰不到（A35，A01 就是这么漏掉的）。 */
        const size_t len = strlen(big);
        char *exact = (char *)malloc(len + 1);
        if (!exact) { fprintf(stderr, "[FAIL] oom\n"); fails++; break; }
        memcpy(exact, big, len + 1);
        const int r = feng_calc_answer(exact, ans, (int)ans_sz);
        free(exact);
        if (r != 0 && r != 1) { fprintf(stderr, "[FAIL] 返回 %d\n", r); fails++; }
        /* 输出必须是 NUL 结尾且在缓冲区内 */
        if (r == 1 && ans[ans_sz - 1] != 0 && strlen(ans) >= ans_sz) {
            fprintf(stderr, "[FAIL] 答案越界\n"); fails++;
        }
    }
    return fails;
}

static int fuzz_tools(int rounds, char *ans, size_t ans_sz)
{
    int fails = 0;
    static const long long edges[] = {
        0, 1, -1, 2, -2, 127, -128, 32767, -32768, 2147483647LL, -2147483648LL,
        4294967295LL, -4294967296LL, 9223372036854775807LL, (-9223372036854775807LL - 1),
        9007199254740992LL, -9007199254740992LL,
    };
    const int ne = (int)(sizeof(edges) / sizeof(edges[0]));

    /* 全值域与跨零区间最容易触发 (hi-lo)+1 溢出 */
    for (int a = 0; a < ne; a++) {
        for (int b = 0; b < ne; b++) {
            const long long lo = edges[a], hi = edges[b];
            const long long v = feng_rand_range(0x12345678ull + (unsigned long long)(a * ne + b), lo, hi);
            if (hi >= lo && (v < lo || v > hi)) {
                fprintf(stderr, "[FAIL] rand_range(%lld,%lld)=%lld 越界\n", lo, hi, v);
                fails++;
            }
        }
    }
    for (int i = 0; i < rounds; i++) {
        const long long lo = (long long)(((uint64_t)rnd32() << 32) | rnd32());
        const long long hi = (long long)(((uint64_t)rnd32() << 32) | rnd32());
        (void)feng_rand_range(rnd32(), lo, hi);
    }

    /* 时间/随机数/记忆的畸形输入 */
    char buf[256];
    for (int i = 0; i < rounds; i++) {
        gen_cn_string(buf, (int)sizeof(buf));
        // 精确分配（含截断 UTF-8 的末尾），让 ASAN 能抓到越过 NUL 的读
        const size_t len = strlen(buf);
        char *exact = (char *)malloc(len + 1);
        if (!exact) { fprintf(stderr, "[FAIL] oom\n"); fails++; break; }
        memcpy(exact, buf, len + 1);
        (void)feng_time_answer(exact, ans, (int)ans_sz);
        (void)feng_random_answer(exact, ans, (int)ans_sz);
        (void)feng_mem_answer(exact, ans, (int)ans_sz);
        (void)feng_mem_learn(exact);
        free(exact);
    }
    if (feng_time_answer(NULL, ans, (int)ans_sz) != 0) { fprintf(stderr, "[FAIL] time NULL 未拒绝\n"); fails++; }
    if (feng_random_answer(NULL, ans, (int)ans_sz) != 0) { fprintf(stderr, "[FAIL] random NULL 未拒绝\n"); fails++; }
    return fails;
}

/* ---- 模型头部模糊：随机字节不能崩、不能越界读 ---- */

static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static void putf32(uint8_t *p, float v) { memcpy(p, &v, 4); }

/* 造一个**结构上合法**的模型镜像（1 层、小尺寸），用来穿过 magic/header/dir
 * 校验，进入张量绑定与尺寸校验的深路径。旧 fuzz 的随机字节几乎全死在
 * magic 上，等于没测到这些分支（A35）。返回长度，buf 由调用方精确分配。 */
static size_t build_seed_model(uint8_t **out)
{
    enum { NL = 1, H = 64, NH = 1, HD = 64, FF = 128, VOCAB = 256, NTEN = 2 + 11 * NL };
    static const char *const lnames[11] = {
        "attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight",
        "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    };
    char names[NTEN][32];
    uint32_t s0[NTEN], s1[NTEN];
    snprintf(names[0], 32, "tok_embd.weight");     s0[0] = VOCAB; s1[0] = H;
    snprintf(names[1], 32, "output_norm.weight");  s0[1] = H;     s1[1] = 0;
    for (int l = 0; l < NL; l++) {
        for (int k = 0; k < 11; k++) {
            const int i = 2 + l * 11 + k;
            snprintf(names[i], 32, "blk.%d.%s", l, lnames[k]);
            switch (k) {
            case 2: case 3: case 4: case 5: s0[i] = H; s1[i] = H; break;
            case 6: case 7: s0[i] = HD; s1[i] = 0; break;
            case 8: case 9: s0[i] = FF; s1[i] = H; break;
            case 10: s0[i] = H; s1[i] = FF; break;
            default: s0[i] = H; s1[i] = 0; break;
            }
        }
    }
    const size_t dir_bytes = 44 + (size_t)NTEN * 68;
    uint64_t off = dir_bytes;
    size_t total = dir_bytes;
    for (int i = 0; i < NTEN; i++) {
        total += (size_t)s0[i] * (s1[i] ? s1[i] : 1) * 2;   /* 全 fp16 */
    }
    uint8_t *b = (uint8_t *)calloc(1, total);
    if (!b) return 0;
    put32(b + 0, (uint32_t)FENG_MAGIC);
    put32(b + 4, 1u);
    put32(b + 8, (uint32_t)NTEN);
    put32(b + 12, NL); put32(b + 16, H); put32(b + 20, NH);
    put32(b + 24, HD); put32(b + 28, FF); put32(b + 32, VOCAB);
    putf32(b + 36, 1e-6f);
    putf32(b + 40, 1e6f);
    for (int i = 0; i < NTEN; i++) {
        uint8_t *e = b + 44 + (size_t)i * 68;
        memcpy(e, names[i], strlen(names[i]));
        put32(e + 32, (uint32_t)FENG_DT_FP16);
        put32(e + 36, s0[i]);
        put32(e + 40, s1[i]);
        put32(e + 44, 0);
        put32(e + 48, 0);
        const uint64_t nb = (uint64_t)s0[i] * (s1[i] ? s1[i] : 1) * 2;
        put64(e + 52, off);
        put64(e + 60, nb);
        off += nb;
    }
    *out = b;
    return total;
}

static int fuzz_model(int rounds)
{
    int fails = 0;
    /* 随机字节：精确分配，别放在大数组里（红区碰不到就漏越界） */
    for (int i = 0; i < rounds; i++) {
        const int len = rnd_range(0, 8192);
        uint8_t *blob = (uint8_t *)malloc((size_t)(len > 0 ? len : 1));
        if (!blob) { fprintf(stderr, "[FAIL] oom\n"); return fails + 1; }
        for (int j = 0; j < len; j++) blob[j] = (uint8_t)rnd32();
        feng_model_t m;
        memset(&m, 0, sizeof(m));
        const int rc = feng_model_init(&m, blob, (size_t)len);
        free(blob);
        if (rc == 0) {
            /* 随机字节通过校验的概率应当极低；真过了说明校验有洞 */
            fprintf(stderr, "[FAIL] 随机 %d 字节被当成合法模型（seed 可复现）\n", len);
            fails++;
        }
    }
    /* 合法 magic 但字段自相矛盾的头部：必须被拒，且不能读越界 */
    for (int i = 0; i < 200; i++) {
        uint8_t blob[64];
        memset(blob, 0, sizeof(blob));
        put32(blob, (uint32_t)FENG_MAGIC);
        put32(blob + 4, 1u);
        uint32_t bad[7] = {rnd32(), rnd32(), rnd32(), rnd32(), rnd32(), rnd32(), rnd32()};
        memcpy(blob + 8, bad, sizeof(bad));
        putf32(blob + 36, 1e-6f);
        putf32(blob + 40, 1e6f);
        feng_model_t m;
        memset(&m, 0, sizeof(m));
        (void)feng_model_init(&m, blob, sizeof(blob));
    }
    /* 结构化种子：合法镜像 + 随机变异（尺寸字段 / 目录项 / 截断）。
     * 只有能穿过 header 的种子才会覆盖张量绑定与逐项尺寸校验。 */
    uint8_t *seed = NULL;
    const size_t seed_len = build_seed_model(&seed);
    if (!seed || seed_len == 0) { fprintf(stderr, "[FAIL] 造种子模型失败\n"); return fails + 1; }
    {   /* 种子本身必须能加载，否则深路径根本没被覆盖 */
        feng_model_t m;
        memset(&m, 0, sizeof(m));
        if (feng_model_init(&m, seed, seed_len) != 0) {
            fprintf(stderr, "[FAIL] 种子模型不合法（fuzz 未覆盖张量绑定）\n");
            fails++;
        }
    }
    const int structured = rounds < 1000 ? rounds : 1000;   /* 每次可能打印库内诊断，别刷屏 */
    for (int i = 0; i < structured; i++) {
        const int cut = rnd_range(1, 5);
        size_t len = seed_len;
        if ((rnd32() & 7u) == 0u) len = (size_t)rnd_range(1, (int)seed_len);  /* 截断 */
        uint8_t *blob = (uint8_t *)malloc(len > 0 ? len : 1);
        if (!blob) { fprintf(stderr, "[FAIL] oom\n"); break; }
        memcpy(blob, seed, len < seed_len ? len : seed_len);
        for (int k = 0; k < cut; k++) {
            /* 主要打头部/目录（前 1.6KB），偶尔打数据区 */
            const size_t at = (rnd32() & 3u) ? (size_t)rnd_range(0, 44 + 26 * 68)
                                             : (size_t)rnd_range(0, (int)len);
            if (at < len) blob[at] ^= (uint8_t)(1u << (rnd32() & 7u));
        }
        feng_model_t m;
        memset(&m, 0, sizeof(m));
        (void)feng_model_init(&m, blob, len);
        free(blob);
    }
    free(seed);
    return fails;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "calc";
    const int rounds = argc > 2 ? atoi(argv[2]) : 20000;
    if (argc > 3) s_rng = (uint64_t)strtoull(argv[3], NULL, 10) | 1ull;

    char ans[512];
    memset(ans, 0, sizeof(ans));

    int fails = 0;
    if (strcmp(mode, "calc") == 0)       fails = fuzz_calc(rounds, ans, sizeof(ans));
    else if (strcmp(mode, "tools") == 0) fails = fuzz_tools(rounds, ans, sizeof(ans));
    else if (strcmp(mode, "model") == 0) fails = fuzz_model(rounds);
    else {
        fprintf(stderr, "未知模式 %s（calc / tools / model）\n", mode);
        return 2;
    }

    printf("fuzz %-5s rounds=%-7d seed=%llu -> %s（%d 个失败）\n",
           mode, rounds, (unsigned long long)s_rng, fails ? "有失败" : "干净", fails);
    return fails ? 1 : 0;
}
