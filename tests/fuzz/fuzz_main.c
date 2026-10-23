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
    char buf[512];
    run_calc_known_bad(ans, ans_sz, &fails);

    /* 入口防御：NULL / 零长 / 负长度 */
    if (feng_calc_answer(NULL, ans, (int)ans_sz) != 0) { fprintf(stderr, "[FAIL] NULL user 未拒绝\n"); fails++; }
    if (feng_calc_answer("1+1", NULL, 16) != 0) { fprintf(stderr, "[FAIL] NULL answer 未拒绝\n"); fails++; }
    if (feng_calc_answer("1+1", ans, 0) != 0) { fprintf(stderr, "[FAIL] answer_sz=0 未拒绝\n"); fails++; }
    if (feng_calc_answer("1+1", ans, -5) != 0) { fprintf(stderr, "[FAIL] answer_sz<0 未拒绝\n"); fails++; }

    for (int i = 0; i < rounds; i++) {
        gen_cn_string(buf, (int)sizeof(buf));
        const int r = feng_calc_answer(buf, ans, (int)ans_sz);
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
        (void)feng_time_answer(buf, ans, (int)ans_sz);
        (void)feng_random_answer(buf, ans, (int)ans_sz);
        (void)feng_mem_answer(buf, ans, (int)ans_sz);
        (void)feng_mem_learn(buf);
    }
    if (feng_time_answer(NULL, ans, (int)ans_sz) != 0) { fprintf(stderr, "[FAIL] time NULL 未拒绝\n"); fails++; }
    if (feng_random_answer(NULL, ans, (int)ans_sz) != 0) { fprintf(stderr, "[FAIL] random NULL 未拒绝\n"); fails++; }
    return fails;
}

/* ---- 模型头部模糊：随机字节不能崩、不能越界读 ---- */
static int fuzz_model(int rounds)
{
    int fails = 0;
    static uint8_t blob[8192];
    for (int i = 0; i < rounds; i++) {
        const int len = rnd_range(0, (int)sizeof(blob));
        for (int j = 0; j < len; j++) blob[j] = (uint8_t)rnd32();
        feng_model_t m;
        memset(&m, 0, sizeof(m));
        const int rc = feng_model_init(&m, blob, (size_t)len);
        if (rc == 0) {
            /* 随机字节通过校验的概率应当极低；真过了说明校验有洞 */
            fprintf(stderr, "[FAIL] 随机 %d 字节被当成合法模型（seed 可复现）\n", len);
            fails++;
        }
    }
    /* 合法 magic 但字段自相矛盾的头部：必须被拒，且不能读越界 */
    for (int i = 0; i < 200; i++) {
        memset(blob, 0, 64);
        const uint32_t magic = (uint32_t)FENG_MAGIC;
        memcpy(blob, &magic, 4);
        const uint32_t ver = 1u;
        memcpy(blob + 4, &ver, 4);
        uint32_t bad[7] = {rnd32(), rnd32(), rnd32(), rnd32(), rnd32(), rnd32(), rnd32()};
        memcpy(blob + 8, bad, sizeof(bad));
        feng_model_t m;
        memset(&m, 0, sizeof(m));
        (void)feng_model_init(&m, blob, 64);
    }
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
