/* 计算 tool 实现：纯 C、无动态内存，ESP32-S3 的 FPU 直接算。 */
#include "feng_calc.h"

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define CALC_MAX 256

static void append_bytes(char *out, int cap, int *n, const char *s, int len)
{
    if (out == NULL || n == NULL || cap <= 0 || s == NULL) return;
    if (*n < 0) *n = 0;
    if (*n >= cap - 1 || len <= 0) { out[cap > 0 ? cap - 1 : 0] = 0; return; }
    const int room = cap - 1 - *n;
    if (len > room) len = room;
    memcpy(out + *n, s, (size_t)len);
    *n += len;
    out[*n] = 0;
}

/* 有界追加：n 始终是**实际写入长度**，永远不会越过 cap-1。
 * 之前数数 tool 直接把 snprintf 的返回值（“本应写入长度”）累加到 n，
 * 输出超过 answer_sz 后 (answer_sz - n) 会下溢成巨大的 size_t，写穿栈缓冲。 */
static void append_fmt(char *out, int cap, int *n, const char *fmt, ...)
{
    if (out == NULL || n == NULL || cap <= 0) return;
    if (*n < 0) *n = 0;
    if (*n >= cap - 1) { out[cap - 1] = 0; return; }
    va_list ap;
    va_start(ap, fmt);
    const int room = cap - 1 - *n;
    const int w = vsnprintf(out + *n, (size_t)room, fmt, ap);
    va_end(ap);
    if (w < 0) { out[*n] = 0; return; }
    if (w >= room) *n = cap - 1;
    else *n += w;
    out[*n] = 0;
}

static int is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static int is_digit(char c)
{
    return c >= '0' && c <= '9';
}

/* 把 UTF-8 的中文运算符/前后缀替换成 ASCII 形式；返回写出的长度 */
static int normalize_ops(const char *in, char *out, int sz)
{
    int n = 0;
    const unsigned char *p = (const unsigned char *)in;
    while (*p && n < sz - 1) {
        /* 先算剩余长度：旧实现无条件读 p[1]..p[5]，当输入末尾恰好是 0xE4
         * 这类多字节首字节时会读过 NUL（最多 5 字节）。 */
        const size_t rem = strlen((const char *)p);
        /* 双字算符优先（乘以 / 除以 / 加上 / 减去） */
        if (rem >= 6 && p[0] == 0xE4 && p[1] == 0xB9 && p[2] == 0x98 && p[3] == 0xE4 && p[4] == 0xBB && p[5] == 0xA5) { out[n++] = '*'; p += 6; continue; } /* 乘以 */
        if (rem >= 6 && p[0] == 0xE9 && p[1] == 0x99 && p[2] == 0xA4 && p[3] == 0xE4 && p[4] == 0xBB && p[5] == 0xA5) { out[n++] = '/'; p += 6; continue; } /* 除以 */
        if (rem >= 6 && p[0] == 0xE5 && p[1] == 0x8A && p[2] == 0xA0 && p[3] == 0xE4 && p[4] == 0xB8 && p[5] == 0x8A) { out[n++] = '+'; p += 6; continue; } /* 加上 */
        if (rem >= 6 && p[0] == 0xE5 && p[1] == 0x87 && p[2] == 0x8F && p[3] == 0xE5 && p[4] == 0x8E && p[5] == 0xBB) { out[n++] = '-'; p += 6; continue; } /* 减去 */
        if (rem >= 3 && p[0] == 0xE5 && p[1] == 0x8A && p[2] == 0xA0) {          /* 加 */
            out[n++] = '+'; p += 3; continue;
        }
        if (rem >= 3 && p[0] == 0xE5 && p[1] == 0x87 && p[2] == 0x8F) {          /* 减 */
            out[n++] = '-'; p += 3; continue;
        }
        if (rem >= 3 && p[0] == 0xE4 && p[1] == 0xB9 && p[2] == 0x98) {          /* 乘 */
            out[n++] = '*'; p += 3; continue;
        }
        if (rem >= 3 && p[0] == 0xE9 && p[1] == 0x99 && p[2] == 0xA4) {          /* 除 */
            out[n++] = '/'; p += 3; continue;
        }
        if (rem >= 2 && p[0] == 0xC3 && p[1] == 0x97) { out[n++] = '*'; p += 2; continue; }   /* × */
        if (rem >= 2 && p[0] == 0xC3 && p[1] == 0xB7) { out[n++] = '/'; p += 2; continue; }   /* ÷ */
        if (rem >= 3 && p[0] == 0xEF && p[1] == 0xBC && p[2] == 0x88) { out[n++] = '('; p += 3; continue; } /* （ */
        if (rem >= 3 && p[0] == 0xEF && p[1] == 0xBC && p[2] == 0x89) { out[n++] = ')'; p += 3; continue; } /* ） */
        if (rem >= 3 && p[0] == 0xEF && p[1] == 0xBC && p[2] == 0x85) { out[n++] = '%'; p += 3; continue; } /* ％ */
        out[n++] = (char)*p++;
    }
    out[n] = 0;
    return n;
}

/* ---- 中文数字（五十九、一百零五、两千三、二零二六）转成阿拉伯数字 ---- */
/* rem 是 p 起可读的剩余字节数，避免截断的 UTF-8 序列读过 NUL。 */
static int utf8_cp(const unsigned char *p, size_t rem, unsigned *cp)
{
    if (p[0] < 0x80) { *cp = p[0]; return 1; }
    if ((p[0] & 0xE0) == 0xC0 && rem >= 2) {
        *cp = ((unsigned)(p[0] & 0x1F) << 6) | (unsigned)(p[1] & 0x3F);
        return 2;
    }
    if ((p[0] & 0xF0) == 0xE0 && rem >= 3) {
        *cp = ((unsigned)(p[0] & 0x0F) << 12) | ((unsigned)(p[1] & 0x3F) << 6) |
              (unsigned)(p[2] & 0x3F);
        return 3;
    }
    *cp = p[0];
    return 1;
}

/* 返回：0-9 数字；10/100/1000/10000 单位；-1 不是中文数字 */
static int cn_map(unsigned cp)
{
    switch (cp) {
        case 0x96F6: case 0x3007: return 0;          /* 零 〇 */
        case 0x4E00: return 1;                        /* 一 */
        case 0x4E8C: case 0x4E24: return 2;           /* 二 两 */
        case 0x4E09: return 3;
        case 0x56DB: return 4;
        case 0x4E94: return 5;
        case 0x516D: return 6;
        case 0x4E03: return 7;
        case 0x4E5D: return 9;
        case 0x516B: return 8;
        case 0x5341: return 10;                       /* 十 */
        case 0x767E: return 100;                      /* 百 */
        case 0x5343: return 1000;                     /* 千 */
        case 0x4E07: return 10000;                    /* 万 */
        default: return -1;
    }
}

/* 把 s 里的中文数字段替换成阿拉伯数字（就地重写）。
 * room 是 s 处可写的字节数（含结尾 NUL）——s 可能指向调用方缓冲的中段
 * （前导空格被剥掉之后），不能用"整个缓冲的大小"当边界。
 * 返回 1 = 转换完成；0 = 结果放不下，此时 s 保持原样（内容本来就在缓冲里，
 * 不会越界；后续校验遇到中文字节会判"不是算式"，语义上也安全）。
 *
 * 两类写法都要支持：
 *   - 带单位的数字：五十九 / 一百零五 / 两千三（=2300）/ 一万二（=12000）；
 *   - 纯数字串：二零二六（=2026，年份/编号常用）。
 * 旧实现把“两千三”算成 2003，把“二零二六”算成 6。
 *
 * 累加全程防溢出：85 个「九」会把 val 乘出 long long（有符号溢出是 UB），
 * 「万」连写也会让 section/total 爆掉。超界一律封顶 LLONG_MAX——
 * 这种病态输入后续都会被校验拦下，重要的是过程无 UB、结果确定。 */
static int sat_add_ll(long long *acc, long long add)
{
    if (*acc > LLONG_MAX - add) { *acc = LLONG_MAX; return 0; }
    *acc += add;
    return 1;
}

static int cn_to_ascii(char *s, int room)
{
    char out[CALC_MAX];
    int n = 0;
    const unsigned char *p = (const unsigned char *)s;
    if (room < 1) return 0;
    out[0] = 0;
    while (*p && n < (int)sizeof(out) - 1) {
        unsigned cp;
        const size_t rem = strlen((const char *)p);
        const int len = utf8_cp(p, rem, &cp);
        if (cp < 0x80 || cn_map(cp) < 0) {
            append_bytes(out, (int)sizeof(out), &n, (const char *)p, len);
            p += len;
            continue;
        }
        /* 先找到连续中文数字段的结尾，并判断是否有“十/百/千/万”单位 */
        const unsigned char *seg = p;
        const unsigned char *q = p;
        int has_unit = 0;
        while (*q) {
            unsigned c2;
            const int l2 = utf8_cp(q, strlen((const char *)q), &c2);
            const int v = cn_map(c2);
            if (v < 0) break;
            if (v >= 10) has_unit = 1;
            q += l2;
        }
        long long val = 0;
        if (!has_unit) {
            /* 二零二六 -> 2026 */
            const unsigned char *r = seg;
            while (r < q) {
                unsigned c2;
                const int l2 = utf8_cp(r, (size_t)(q - r), &c2);
                const int d = cn_map(c2);
                r += l2;
                /* 纯数字串可能无限长：85 个「九」就会把 long long 乘溢出
                 * （有符号溢出是 UB，各编译器/优化级别行为不一）。
                 * 超界封顶 LLONG_MAX 并停止累加。 */
                if (val > (LLONG_MAX - d) / 10) { val = LLONG_MAX; break; }
                val = val * 10 + d;
            }
        } else {
            long long total = 0, section = 0, cur = 0;
            int last_unit = 1, after_zero = 0, ovf = 0;
            const unsigned char *r = seg;
            while (r < q && !ovf) {
                unsigned c2;
                const int l2 = utf8_cp(r, (size_t)(q - r), &c2);
                const int v = cn_map(c2);
                r += l2;
                if (v == 0) {
                    after_zero = 1;
                    cur = 0;
                } else if (v <= 9) {
                    cur = v;
                    /* 口语省略：两千三 = 2300、三百五 = 350、一万二 = 12000。
                     * 只有“这个数字后面不再跟单位”时才缩放；否则“两千五百”
                     * 里的五会被误乘成 500，再由百变成 52000。
                     * “一百零五”里的零会置 after_zero，不触发省略缩放。 */
                    int next_is_unit = 0;
                    if (r < q) {
                        unsigned c3;
                        (void)utf8_cp(r, (size_t)(q - r), &c3);
                        next_is_unit = cn_map(c3) >= 10;
                    }
                    if (last_unit >= 100 && !after_zero && !next_is_unit)
                        cur *= last_unit / 10;
                    after_zero = 0;
                } else if (v < 10000) {
                    /* cur ≤ 900、v ≤ 1000，乘积不会溢出；section 可能已经很大 */
                    if (!sat_add_ll(&section, (cur ? cur : 1) * v)) { ovf = 1; break; }
                    last_unit = v;
                    cur = 0;
                    after_zero = 0;
                } else {
                    /* 「万」：先并段再进 total。section + cur 与 ×10000 两步都要防溢出 */
                    if (section > LLONG_MAX - cur) { ovf = 1; break; }
                    const long long base = section + cur;
                    if (base > LLONG_MAX / 10000) { ovf = 1; break; }
                    if (!sat_add_ll(&total, base * 10000)) { ovf = 1; break; }
                    section = 0;
                    cur = 0;
                    last_unit = 10000;
                    after_zero = 0;
                }
            }
            if (ovf) {
                val = LLONG_MAX;
            } else if (total > LLONG_MAX - section || section > LLONG_MAX - cur) {
                val = LLONG_MAX;
            } else {
                val = total + section + cur;
            }
        }
        append_fmt(out, (int)sizeof(out), &n, "%lld", val);
        p = q;
    }
    out[n] = 0;
    /* 回写前按 room 卡边界：中文数字转换是"压缩"（3 字节 -> 1~几个数字），
     * 但「万」连写这类单位展开是"放大"（12 字节 -> 17 字节），out 最长 255
     * 字节，而 s 可能指向调用方缓冲的中段（前导空格剥掉之后）。
     * 旧实现无条件 memcpy(s, out, n+1)，"   + 一万万…" 这类输入会越过
     * 调用方缓冲末尾（实测 3 字节）。放不下就放弃转换、s 原样保留。 */
    if (n + 1 > room) return 0;
    memcpy(s, out, (size_t)n + 1);
    return 1;
}

/* 后缀/前缀外壳：返回 1 表示剥掉了一层 */
static int strip_suffix(char *s)
{
    static const char *suf[] = {"是多少钱", "是多少元", "多少钱", "多少元", "是多少呢", "是多少呀",
                                "算一下", "算算", "算下", "计算一下",
                                "等于几", "等于多少", "是多少", "多少", "=?", "＝?", "?", "？",
                                "呢", "呀", "。", ".", "!", "！", "=", "＝"};
    const int n = (int)strlen(s);
    for (int i = 0; i < (int)(sizeof(suf) / sizeof(suf[0])); i++) {
        const int L = (int)strlen(suf[i]);
        if (n >= L && strcmp(s + n - L, suf[i]) == 0) {
            s[n - L] = 0;
            return 1;
        }
    }
    return 0;
}

static int strip_prefix(char *s)
{
    static const char *pre[] = {"帮我算一下", "帮我计算", "帮我算算", "帮我算", "麻烦算一下",
                                "麻烦算算", "计算一下", "计算", "算一下", "算算", "请问一下",
                                "请问", "求解", "求", "把"};
    for (int i = 0; i < (int)(sizeof(pre) / sizeof(pre[0])); i++) {
        const int L = (int)strlen(pre[i]);
        if (strncmp(s, pre[i], L) == 0) {
            memmove(s, s + L, strlen(s + L) + 1);
            return 1;
        }
    }
    return 0;
}

/* 日期/电话号码保护：2026-10-18、138-1234-5678 这类串在“数字+减号”上与算式同形，
 * 直接交给计算器会得到荒谬答案（实测 2026-10-18 -> 1998）。只有三组、且组长度
 * 符合日期/电话形态时才拦截，避免误伤 10-2-3 这种普通连减。 */
static int looks_like_date_or_phone(const char *s)
{
    int g[4] = {0, 0, 0, 0};
    int ng = 0, cur = 0;
    for (const char *p = s; *p; p++) {
        if (is_digit(*p)) { cur++; continue; }
        if (*p == '-') {
            if (ng >= 3) return 0;
            g[ng++] = cur;
            cur = 0;
            continue;
        }
        return 0;                       /* 有其他字符就不按日期处理 */
    }
    g[ng++] = cur;
    if (ng != 3) return 0;
    /* YYYY-MM-DD / MM-DD-YYYY */
    if ((g[0] == 4 && g[1] >= 1 && g[1] <= 2 && g[2] >= 1 && g[2] <= 2) ||
        (g[2] == 4 && g[0] >= 1 && g[0] <= 2 && g[1] >= 1 && g[1] <= 2)) return 1;
    /* 3-4 位区号 + 3-4 位 + 4 位号码 */
    if (g[0] >= 3 && g[0] <= 4 && g[1] >= 3 && g[1] <= 4 && g[2] == 4) return 1;
    return 0;
}

int feng_calc_extract(const char *user, char *expr, int expr_sz)
{
    char work[CALC_MAX];
    normalize_ops(user, work, sizeof(work));
    /* 去掉首尾空白 */
    char *s = work;
    while (is_space(*s)) s++;
    int n = (int)strlen(s);
    while (n > 0 && is_space(s[n - 1])) s[--n] = 0;
    /* 剥外壳（可能叠加：先前后缀各来一轮） */
    for (int round = 0; round < 6; round++) {
        int changed = 0;
        while (n > 0 && is_space(s[n - 1])) s[--n] = 0;
        if (strip_suffix(s)) changed = 1;
        n = (int)strlen(s);
        while (n > 0 && is_space(s[n - 1])) s[--n] = 0;
        if (strip_prefix(s)) changed = 1;
        n = (int)strlen(s);
        while (n > 0 && is_space(s[n - 1])) s[--n] = 0;
        if (!changed) break;
    }
    /* 中文数字放在外壳剥离之后转换（否则"帮我算一下"里的"一"会被改成 1）。
     * s 可能指向 work 中段，可写长度是 "work 末尾 - s"；转换失败（放不下）时
     * s 保持中文原样，后续校验会拒，不会越界。 */
    if (!cn_to_ascii(s, (int)sizeof(work) - (int)(s - work))) {
        /* 保留原样即可：后面还有「的/根号/平方」改写与字符校验，
         * 中文字节不在白名单里，最终会被判为"不是算式"。 */
    }
    /* "100的15%" -> "100*15%"：只有一个"的"且换成 * 后是合法算式时才认 */
    {
        char *de = strstr(s, "的");
        if (de && !strstr(de + 3, "的")) {
            const size_t pos = (size_t)(de - s);
            memmove(de + 1, de + 3, strlen(de + 3) + 1);
            s[pos] = '*';
        }
    }
    /* "根号16" -> "V16"（V = 平方根，解析器里处理） */
    if (strncmp(s, "根号", 6) == 0) {
        memmove(s + 1, s + 6, strlen(s + 6) + 1);
        s[0] = 'V';
    }
    /* "X的平方/立方" 已在上一步变成 "X*平方/立方"？——中文单位先转成重复乘法。
     * 注意 s 可能指向 work 的中段（输入有前导空格），所以目标缓冲的可用长度是
     * "work 末尾 - s"，不是 sizeof(work)。旧实现按 sizeof(work) 写，
     * "   1+1+…的平方" 这类输入会把 work 后面的栈写穿（实测 40 字节）。
     * 放不下时直接判定"不是算式"，不要输出被截断的半截表达式。 */
    {
        char tmp[CALC_MAX];
        snprintf(tmp, sizeof(tmp), "%s", s);
        char *p = strstr(tmp, "*平方");
        char *q = p ? NULL : strstr(tmp, "*立方");
        if (p || q) {
            char *hit = p ? p : q;
            const char *tail = hit + (p ? 7 : 7);   /* "*平方"/"*立方" 都是 7 字节 */
            /* 后缀必须落在整式末尾，且底数是简单表达式（括号内可以有运算符）。
             * 旧实现把命中点之后全部丢掉："2的平方加1" 答成 4；
             * "1+3的平方" 又被展开成 1+3*1+3。两种都是静默错答，
             * 这里不支持就直接返回 0，交给模型而不是猜。 */
            while (is_space(*tail)) tail++;
            if (*tail) return 0;
            *hit = 0;
            int depth = 0, simple = 1, digits = 0;
            for (const char *r = tmp; *r; r++) {
                if (*r == '(') depth++;
                else if (*r == ')') { if (--depth < 0) { simple = 0; break; } }
                else if (is_digit(*r)) digits++;
                else if (*r == '.' || is_space(*r)) continue;
                else if (depth > 0 && (*r == '+' || *r == '-' || *r == '*' || *r == '/')) continue;
                else if (*r == '-' && r == tmp) continue;          /* 一元负号 */
                else if (*r == 'V' || *r == 'v') continue;
                else { simple = 0; break; }
            }
            if (!simple || depth != 0 || digits == 0) return 0;
            const int len = (int)strlen(tmp);
            const int need = (p ? len * 2 + 2 : len * 3 + 3);
            const int room = (int)sizeof(work) - (int)(s - work);
            if (need >= room) return 0;
            if (p) snprintf(s, (size_t)room, "%s*%s", tmp, tmp);
            else snprintf(s, (size_t)room, "%s*%s*%s", tmp, tmp, tmp);
        }
    }
    /* 日期/电话形态默认不当作算式；只有明确说“算/等于/减”时才放行。 */
    if (looks_like_date_or_phone(s) &&
        !strstr(user, "算") && !strstr(user, "等于") && !strstr(user, "减")) return 0;
    /* 校验：只允许数字/小数点/运算符/括号/空格，且至少一个数字与一个运算符 */
    int has_digit = 0, has_op = 0;
    for (const char *p = s; *p; p++) {
        if (is_digit(*p)) has_digit = 1;
        else if (strchr("+-*/()", *p)) {
            /* 括号不是运算符，不能把已经置位的 has_op 又清掉：
             * 旧写法 "(1+2)" 会因为最后一个 ')' 把 has_op 归零而被拒绝。 */
            if (*p == '+' || *p == '-' || *p == '*' || *p == '/') has_op = 1;
        } else if (*p == 'V' || *p == 'v') has_op = 1;
        else if (*p == '.' || *p == '%' || is_space(*p)) continue;
        else return 0;
    }
    if (!has_digit || !has_op) return 0;
    /* 去掉空格，输出紧凑算式 */
    int m = 0;
    for (const char *p = s; *p && m < expr_sz - 1; p++) {
        if (!is_space(*p)) expr[m++] = *p;
    }
    expr[m] = 0;
    return m > 0;
}

/* ---- 递归下降求值 ---- */
typedef struct {
    const char *p;
    int err;                        /* 0 ok, -1 div0, -2 syntax */
    int depth;                      /* 括号/一元链深度，防止栈溢出 */
} calc_ctx;

static double parse_expr(calc_ctx *c);
static double parse_factor(calc_ctx *c);

static void skip_ws(calc_ctx *c)
{
    while (is_space(*c->p)) c->p++;
}

static double parse_number(calc_ctx *c)
{
    skip_ws(c);
    double v = 0;
    int any = 0;
    while (is_digit(*c->p)) { v = v * 10 + (*c->p - '0'); c->p++; any = 1; }
    if (*c->p == '.') {
        c->p++;
        double scale = 0.1;
        while (is_digit(*c->p)) { v += (*c->p - '0') * scale; scale *= 0.1; c->p++; any = 1; }
    }
    if (!any) c->err = -2;
    if (*c->p == '%') { v /= 100.0; c->p++; }         /* 15% -> 0.15 */
    return v;
}

static double parse_factor_inner(calc_ctx *c)
{
    skip_ws(c);
    if (*c->p == 'V' || *c->p == 'v') {           /* 平方根 */
        c->p++;
        const double v = parse_factor(c);
        if (v < 0) { c->err = -2; return 0; }
        return sqrt(v);
    }
    if (*c->p == '-') { c->p++; return -parse_factor(c); }
    if (*c->p == '+') { c->p++; return parse_factor(c); }
    if (*c->p == '(') {
        c->p++;
        double v = parse_expr(c);
        skip_ws(c);
        if (*c->p == ')') c->p++;
        else c->err = -2;
        return v;
    }
    return parse_number(c);
}

/* 所有递归调用都经过这一层：括号 256 层在板端 8 KB 栈上足以打穿栈帧，
 * 这里把嵌套限制在 64 层（超过按语法错误处理）。 */
static double parse_factor(calc_ctx *c)
{
    if (c->err) return 0;
    if (c->depth >= 64) { c->err = -2; return 0; }
    c->depth++;
    const double v = parse_factor_inner(c);
    c->depth--;
    return v;
}

static double parse_term(calc_ctx *c)
{
    double v = parse_factor(c);
    for (;;) {
        skip_ws(c);
        const char op = *c->p;
        if (op != '*' && op != '/') break;
        c->p++;
        const double r = parse_factor(c);
        if (c->err) return v;
        if (op == '*') {
            v *= r;
        } else {
            if (r == 0.0) { c->err = -1; return v; }
            v /= r;
        }
    }
    return v;
}

static double parse_expr(calc_ctx *c)
{
    double v = parse_term(c);
    for (;;) {
        skip_ws(c);
        const char op = *c->p;
        if (op != '+' && op != '-') break;
        c->p++;
        const double r = parse_term(c);
        if (c->err) return v;
        v = (op == '+') ? v + r : v - r;
    }
    return v;
}

int feng_calc_eval(const char *expr, double *out)
{
    calc_ctx c = {expr, 0, 0};
    const double v = parse_expr(&c);
    skip_ws(&c);
    if (c.err) return c.err;
    if (*c.p) return -2;
    *out = v;
    return 1;
}

void feng_calc_fmt(double v, char *buf, int buf_sz)
{
    if (buf == NULL || buf_sz <= 0) return;
    if (v == 0) v = 0;                                   /* 去掉 -0 */
    /* 先判范围再转 long long：|v| ≥ 2^63 时 (long long)v 是未定义行为
     * （实测 1e20 在 x86-64 上得到 INT64_MIN）。 */
    if (fabs(v) < 1e15 && fabs(v - (double)(long long)v) < 1e-9) {
        snprintf(buf, (size_t)buf_sz, "%lld", (long long)v);
        return;
    }
    snprintf(buf, (size_t)buf_sz, "%.4f", v);
    int n = (int)strlen(buf);
    while (n > 0 && buf[n - 1] == '0') buf[--n] = 0;
    if (n > 0 && buf[n - 1] == '.') buf[--n] = 0;
}

/* 把 "59+1" 拆成 a、op、b（用于生成 "59 加 1 等于 60。" 句式） */
static int split_binary(const char *expr, char *a, int asz, char *op, char *b, int bsz)
{
    /* 注意循环从 i=1 开始（跳过可能的一元负号），所以开头的括号必须
     * 单独计入深度；否则 "(1+2)" 会把开头的 '(' 漏掉，在 '+' 处误拆成
     * "(1 加 2) 等于 3。" */
    int depth = (expr[0] == '(');   /* 循环从 i=1 起，首字符 '(' 要单独补上 */
    for (int i = 1; expr[i]; i++) {
        if (expr[i] == '(') depth++;
        else if (expr[i] == ')') depth--;
        else if (depth == 0 && (expr[i] == '+' || expr[i] == '-' || expr[i] == '*' || expr[i] == '/')) {
            /* 只接受"最外层唯一运算符"的最简形式 */
            if (strchr(expr + i + 1, '+') || strchr(expr + i + 1, '-') ||
                strchr(expr + i + 1, '*') || strchr(expr + i + 1, '/')) return 0;
            if (i >= asz || (int)strlen(expr + i + 1) >= bsz) return 0;
            memcpy(a, expr, (size_t)i); a[i] = 0;
            *op = expr[i];
            strcpy(b, expr + i + 1);
            return 1;
        }
    }
    return 0;
}

/* 序列数数："把 1 到 5 倒着数一遍" / "从 3 数到 8"。
 * 只在明确的祈使句上触发（倒着数/倒序，或以"从/把"开头且含"数到/数一遍"），
 * 避免"我从1数到100也数不完"这类普通陈述被截走。返回 1 = 已作答。 */
static int feng_calc_seq_answer(const char *user, char *answer, int answer_sz)
{
    if (user == NULL || answer == NULL || answer_sz <= 0) return 0;
    const int rev = (strstr(user, "倒着数") != NULL) || (strstr(user, "倒序") != NULL) ||
                    (strstr(user, "倒过来数") != NULL);
    int fwd = 0;
    if (!rev) {
        const int starts = (strncmp(user, "从", 3) == 0) || (strncmp(user, "把", 3) == 0) ||
                           strstr(user, "请从") != NULL;
        if (strstr(user, "数到"))
            fwd = starts || strstr(user, "数一遍") != NULL || strstr(user, "数一下") != NULL;
        else if (strstr(user, "数一遍") != NULL || strstr(user, "数一下") != NULL)
            fwd = starts;
    }
    if (!rev && !fwd) return 0;

    char work[CALC_MAX];
    strncpy(work, user, sizeof(work) - 1);
    work[sizeof(work) - 1] = 0;
    /* 转换失败（放不下）时 work 保持中文原样，下面按 ASCII 数字扫描，
     * 找不到两个数就走 nf < 2 的拒绝路径。 */
    (void)cn_to_ascii(work, (int)sizeof(work));
    int found[8];
    int nf = 0;
    int too_big = 0;
    for (const char *p = work; *p && nf < 8; ) {
        /* 支持负号："从-2数到2" 旧实现把 -2 读成 2，输出 "2。"。 */
        int neg = 0;
        if (*p == '-' && is_digit(p[1])) { neg = 1; p++; }
        if (is_digit(*p)) {
            long long v = 0;
            while (is_digit(*p)) {
                if (v > 100000000LL) too_big = 1;
                else v = v * 10 + (*p - '0');
                p++;
            }
            found[nf++] = (int)(neg ? -v : v);
        } else {
            p++;
        }
    }
    if (too_big) return 0;                   /* 超过 9 位：不当序列任务，交给模型/算式校验 */
    if (nf < 2) return 0;                    /* 只有一个数（如"数到8"）交给模型 */
    int a = found[0], b = found[1];
    if (!rev && a > b) return 0;             /* "从8数到3"这类交给模型 */
    if (a > b) { const int t = a; a = b; b = t; }
    if (b - a > 50) {
        snprintf(answer, (size_t)answer_sz, "范围有点大（%d 到 %d），给我 50 个以内的区间吧。", a, b);
        return 1;
    }
    /* 先算实际输出长度：10 位数的 50 个区间会超过 256 字节，旧实现会写穿。
     * 放不下就明确拒绝，不要输出半截列表。 */
    int need = 2;                                  /* 句号 + NUL */
    for (int v = a; v <= b; v++) {
        char tmp[24];
        need += snprintf(tmp, sizeof(tmp), "%d、", v);
        if (need >= answer_sz) break;
    }
    if (need >= answer_sz) {
        snprintf(answer, (size_t)answer_sz, "范围有点大（%d 到 %d），输出放不下，换个小一点的区间吧。", a, b);
        return 1;
    }
    int n = 0;
    answer[0] = 0;
    if (rev) {
        for (int v = b; v >= a; v--)
            append_fmt(answer, answer_sz, &n, (v == b) ? "%d" : "、%d", v);
    } else {
        for (int v = a; v <= b; v++)
            append_fmt(answer, answer_sz, &n, (v == a) ? "%d" : "、%d", v);
    }
    append_fmt(answer, answer_sz, &n, "。");
    return 1;
}

int feng_calc_answer(const char *user, char *answer, int answer_sz)
{
    if (user == NULL || answer == NULL || answer_sz <= 0) return 0;
    /* 序列任务（"把1到5倒着数一遍"/"从3数到8"）也走引擎，确定性输出 */
    if (feng_calc_seq_answer(user, answer, answer_sz)) return 1;
    char expr[CALC_MAX];
    if (!feng_calc_extract(user, expr, sizeof(expr))) return 0;
    double v = 0;
    const int rc = feng_calc_eval(expr, &v);
    if (rc < 0) {
        snprintf(answer, (size_t)answer_sz, "%s", rc == -1 ? "这个算式里除数是 0，算不出来。" : "这个算式我没看懂。");
        return 1;
    }
    char a[64], b[64], op = 0;
    char num[CALC_MAX];
    feng_calc_fmt(v, num, sizeof(num));
    if (split_binary(expr, a, sizeof(a), &op, b, sizeof(b))) {
        const char *sym = op == '+' ? "加" : op == '-' ? "减" : op == '*' ? "乘" : "除以";
        snprintf(answer, (size_t)answer_sz, "%s %s %s 等于 %s。", a, sym, b, num);
    } else {
        snprintf(answer, (size_t)answer_sz, "结果是 %s。", num);
    }
    return 1;
}
