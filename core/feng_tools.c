/* 时间/随机数 tool 实现：不依赖 libc 时区数据库，手算 UTC+8 日历。 */
#include "feng_tools.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static feng_epoch_fn s_epoch_fn;
static feng_uptime_us_fn s_uptime_fn;

void feng_tools_set_time(feng_epoch_fn fn) { s_epoch_fn = fn; }
void feng_tools_set_uptime(feng_uptime_us_fn fn) { s_uptime_fn = fn; }

/* ---- UTC+8 日历（Howard Hinnant 的 civil_from_days 算法） ---- */
static void civil_from_days(long long z, int *y, int *m, int *d)
{
    z += 719468;
    const long long era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned long long doe = (unsigned long long)(z - era * 146097);
    const unsigned long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long yy = (long long)yoe + era * 400;
    const unsigned long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned long long mp = (5 * doy + 2) / 153;
    const unsigned long long dd = doy - (153 * mp + 2) / 5 + 1;
    const unsigned long long mm = mp < 10 ? mp + 3 : mp - 9;
    *y = (int)(yy + (mm <= 2));
    *m = (int)mm;
    *d = (int)dd;
}

static const char *k_week[] = {"日", "一", "二", "三", "四", "五", "六"};

static int contains(const char *s, const char *kw);

/* 支持 0001-01-01 .. 9999-12-31；超出就夹到边界。
 * 旧实现直接 epoch + 28800，LLONG_MAX 附近是有符号溢出（UBSAN 实测）。 */
#define EPOCH_MIN (-62135596800LL)
/* 9999-12-31 23:59:59 UTC 再减 8 小时，保证 +8 后仍落在 9999 年内。 */
#define EPOCH_MAX (253402300799LL - 8 * 3600)

static long long clamp_epoch(long long t)
{
    if (t < EPOCH_MIN) return EPOCH_MIN;
    if (t > EPOCH_MAX) return EPOCH_MAX;
    return t;
}

/* 向下取整除法（C 的 / 向零截断，负时间戳会算错一天）。 */
static long long floor_div(long long a, long long b)
{
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}

void feng_time_format_utc8(long long epoch, char *buf, int buf_sz)
{
    if (buf == NULL || buf_sz <= 0) return;
    long long t = clamp_epoch(epoch) + 8 * 3600;     /* UTC+8 */
    long long days = floor_div(t, 86400);
    long long rem = t - days * 86400;
    int y, m, d;
    civil_from_days(days, &y, &m, &d);
    const int hh = (int)(rem / 3600), mm = (int)((rem % 3600) / 60), ss = (int)(rem % 60);
    const int wd = (int)((days % 7 + 11) % 7);       /* 1970-01-01 是周四 */
    snprintf(buf, (size_t)buf_sz, "%04d年%02d月%02d日 %02d:%02d:%02d（周%s，UTC+8）",
             y, m, d, hh, mm, ss, k_week[wd]);
}

void feng_time_format_date_utc8(long long epoch, char *buf, int buf_sz)
{
    if (buf == NULL || buf_sz <= 0) return;
    long long t = clamp_epoch(epoch) + 8 * 3600;
    long long days = floor_div(t, 86400);
    int y, m, d;
    civil_from_days(days, &y, &m, &d);
    const int wd = (int)((days % 7 + 11) % 7);
    snprintf(buf, (size_t)buf_sz, "%04d年%02d月%02d日（周%s）", y, m, d, k_week[wd]);
}

/* 从 "3天后" 里取出 3；返回 0 表示没有这种模式 */
static int days_offset(const char *user, int *days_out)
{
    static const struct { const char *kw; int delta; } REL[] = {
        {"明天", 1}, {"后天", 2}, {"昨天", -1}, {"前天", -2},
    };
    for (int i = 0; i < (int)(sizeof(REL) / sizeof(REL[0])); i++) {
        if (contains(user, REL[i].kw)) { *days_out = REL[i].delta; return 1; }
    }
    static const char *kws[] = {"天后", "天前", "天之后", "天之前"};
    for (int i = 0; i < (int)(sizeof(kws) / sizeof(kws[0])); i++) {
        const char *hit = strstr(user, kws[i]);
        if (!hit) continue;
        /* 往前找数字 */
        const char *p = hit;
        while (p > user && p[-1] >= '0' && p[-1] <= '9') p--;
        if (p == hit) continue;
        long long v = 0;
        for (const char *q = p; q < hit; q++) {
            if (v < 100000) v = v * 10 + (*q - '0');
            else v = 100000;                       /* 饱和：超出 100k 天没有意义 */
        }
        *days_out = (int)((i == 0 || i == 2) ? v : -v);
        return 1;
    }
    return 0;
}

static int contains(const char *s, const char *kw)
{
    return strstr(s, kw) != NULL;
}

/* 时钟推算："现在7点，再过3小时是几点？" / "3小时后是几点？"（有"现在X点"用 X，否则用板端当前时间）
 * 返回 1 = 已作答。 */
static int time_math_answer(const char *user, char *answer, int answer_sz)
{
    const char *rel = strstr(user, "小时");
    if (!rel) return 0;
    const char *tail = rel + 6;                    /* "小时" 的 UTF-8 长度 */
    if (strncmp(tail, "候", 3) == 0) return 0;     /* "小时候" 不是时钟推算 */
    /* 必须有“推算/问时”的意图，不能见到“8小时”就当成时钟题：
     * 实测「我每天工作8小时」会被答成「再过8小时是…点」。 */
    const int asks_time = strstr(user, "几点") || strstr(user, "什么时候") ||
                          strstr(user, "时间");
    const int has_after = strncmp(tail, "后", 3) == 0 ||
                          strncmp(tail, "之后", 6) == 0 || strncmp(tail, "以后", 6) == 0;
    const int has_before = strncmp(tail, "前", 3) == 0 ||
                           strncmp(tail, "之前", 6) == 0 || strncmp(tail, "以前", 6) == 0;
    if (!asks_time && !has_after && !has_before) return 0;
    const int sign = has_before ? -1 : 1;
    /* "小时"前面紧邻的数字串（支持"再多3小时"/"3 小时"）。手写解析并饱和到
     * ±1,000,000：atoi 溢出是 UB，超大 delta 还会让下面的逐日循环跑几千万次。 */
    const char *p = rel;
    while (p > user && (p[-1] == ' ' || p[-1] == '\t')) p--;
    const char *num_end = p;
    while (p > user && p[-1] >= '0' && p[-1] <= '9') p--;
    if (p == num_end) return 0;                   /* 没有数字，不当作时钟推算 */
    long long dv = 0;
    for (const char *q = p; q < num_end; q++) {
        if (dv < 1000000) dv = dv * 10 + (*q - '0');
        else dv = 1000000;
    }
    const int delta = (int)(dv * sign);
    /* 起点小时：优先"现在X点"，否则用板端当前时间 */
    int base_hour = -1;
    const char *cur = strstr(user, "现在");
    if (cur) {
        const char *q = cur + 6;                   /* "现在" UTF-8 长度 */
        int guard = 0;
        while (*q && (q[0] < '0' || q[0] > '9') && guard++ < 12) q++;
        if (q[0] >= '0' && q[0] <= '9') {
            /* 手写有界解析：atoi 对超长数字串是 UB，且这里只接受 0..23。 */
            int h = 0, digits = 0;
            while (q[0] >= '0' && q[0] <= '9' && digits < 4) {
                h = h * 10 + (q[0] - '0');
                q++;
                digits++;
            }
            if (h >= 0 && h <= 23) base_hour = h;
        }
    }
    if (base_hour < 0) {
        const long long now = s_epoch_fn ? s_epoch_fn() : 0;
        if (now <= 0) return 0;
        base_hour = (int)(((clamp_epoch(now) + 8 * 3600) / 3600) % 24);
    }
    int total = base_hour + delta;
    int day = total / 24;
    int hh2 = total % 24;
    if (hh2 < 0) { hh2 += 24; day--; }
    if (day == 0) snprintf(answer, (size_t)answer_sz, "再过 %d 小时是 %d 点。", delta, hh2);
    else if (day == 1) snprintf(answer, (size_t)answer_sz, "再过 %d 小时是明天 %d 点。", delta, hh2);
    else if (day == -1) snprintf(answer, (size_t)answer_sz, "%d 小时前是昨天 %d 点。", -delta, hh2);
    else if (day == 2) snprintf(answer, (size_t)answer_sz, "再过 %d 小时是后天 %d 点。", delta, hh2);
    else snprintf(answer, (size_t)answer_sz, "再过 %d 小时是 %d 天后 %d 点。", delta, day, hh2);
    return 1;
}

int feng_time_answer(const char *user, char *answer, int answer_sz)
{
    if (user == NULL || answer == NULL || answer_sz <= 0) return 0;
    if (time_math_answer(user, answer, answer_sz)) return 1;
    /* 相对日期只在明确问日期/星期时触发；"明天会下雨吗"这种普通句子交给模型。
     * 旧实现只要出现"明天"就直接答日期，属于典型的 tool 误触发。 */
    static const char *kw[] = {"几点", "现在时间", "现在的时间", "当前时间", "时间是多少",
                               "什么时间", "今天几号", "今天几月", "几号", "几月", "日期",
                               "星期几", "时间戳"};
    /* 个人事实类追问（生日/电话/号码）优先交给记忆 tool 或模型；
     * 没记过时答"现在几号"没有意义。 */
    if ((contains(user, "生日") || contains(user, "电话") || contains(user, "号码")) &&
        !contains(user, "今天") && !contains(user, "现在")) return 0;
    /* "去年今天是星期几"不是当前时间查询，当前实现也不会做年份推算。 */
    if ((contains(user, "去年") || contains(user, "明年")) &&
        !contains(user, "天后") && !contains(user, "天前")) return 0;
    int hit = 0;
    for (int i = 0; i < (int)(sizeof(kw) / sizeof(kw[0])); i++) {
        if (contains(user, kw[i])) { hit = 1; break; }
    }
    if (!hit) return 0;
    const long long now = s_epoch_fn ? s_epoch_fn() : 0;
    if (now <= 0) {
        snprintf(answer, (size_t)answer_sz, "我还没对上网络时间（宿主连接后会自动发 \\settime）。");
        return 1;
    }
    if (contains(user, "时间戳")) {                  /* 要原始 Unix 秒 + UTC+8 换算 */
        char t[128];
        feng_time_format_utc8(now, t, sizeof(t));
        snprintf(answer, (size_t)answer_sz, "时间戳：%lld —— %s。", now, t);
        return 1;
    }
    int days = 0;
    if (days_offset(user, &days)) {
        char d[96], t[128];
        feng_time_format_date_utc8(now + (long long)days * 86400, d, sizeof(d));
        feng_time_format_utc8(now, t, sizeof(t));
        if (days > 0) snprintf(answer, (size_t)answer_sz, "%d 天后是 %s。", days, d);
        else if (days < 0) snprintf(answer, (size_t)answer_sz, "%d 天前是 %s。", -days, d);
        else snprintf(answer, (size_t)answer_sz, "今天是 %s。", d);
        (void)t;
        return 1;
    }
    char t[128];
    feng_time_format_utc8(now, t, sizeof(t));
    snprintf(answer, (size_t)answer_sz, "现在是 %s。", t);
    return 1;
}

/* xorshift64*：跨平台一致，不依赖 libc rand */
static unsigned long long xs64(unsigned long long *s)
{
    unsigned long long x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

long long feng_rand_range(unsigned long long seed, long long lo, long long hi)
{
    if (seed == 0) seed = 88172645463325252ULL;
    unsigned long long st = seed;
    (void)xs64(&st);                                  /* 第一个随机数：按需求丢弃 */
    const unsigned long long r = xs64(&st);
    if (hi < lo) { const long long t = hi; hi = lo; lo = t; }
    /* 全值域：直接返回 64 位随机数，避免 hi-lo 的有符号溢出。 */
    if (lo == LLONG_MIN && hi == LLONG_MAX) return (long long)r;
    /* span 用无符号算：(hi - lo) 在 lo=-2^63+1、hi=2^63-1 时是有符号溢出（UB）。
     * 无符号减法按 2^64 取模，正好是需要的区间长度；span==0 表示覆盖整个 64 位值域。 */
    const unsigned long long span = (unsigned long long)hi - (unsigned long long)lo + 1ULL;
    if (span == 0) return (long long)r;
    return (long long)((unsigned long long)lo + (r % span));
}

int feng_random_answer(const char *user, char *answer, int answer_sz)
{
    if (user == NULL || answer == NULL || answer_sz <= 0) return 0;
    const int random_intent = contains(user, "随机");
    /* "硬币/骰子"单独出现不算意图（"硬币是什么材质"不应被截走）；
     * 必须同时有抛/掷/扔/摇/正反/随机这类动作。 */
    const int coin = (contains(user, "硬币") || contains(user, "正反面")) &&
                     (random_intent || contains(user, "抛") || contains(user, "掷") ||
                      contains(user, "扔") || contains(user, "正反"));
    const int dice = (contains(user, "骰子") || contains(user, "色子")) &&
                     (random_intent || contains(user, "掷") || contains(user, "扔") ||
                      contains(user, "摇") || contains(user, "点") || contains(user, "几"));
    if (!coin && !dice && !random_intent) return 0;
    const long long up_us0 = s_uptime_fn ? s_uptime_fn() : 0;
    const double sec0 = (double)up_us0 / 1e6;
    const unsigned long long seed0 = (unsigned long long)(sec0 * 1.54 * 1000.0);
    if (coin) {
        const long long v = feng_rand_range(seed0, 0, 1);
        snprintf(answer, (size_t)answer_sz, "抛硬币：%s。", v ? "正面" : "反面");
        return 1;
    }
    /* 解析可选范围："1到100" / "1-100" / "0~9"；默认 1..100（骰子默认 1..6） */
    long long lo = 1, hi = dice ? 6 : 100;
    const char *p = user;
    long long nums[2] = {0, 0};
    int nnum = 0;
    while (*p && nnum < 2) {
        if (*p >= '0' && *p <= '9') {
            long long v = 0;
            while (*p >= '0' && *p <= '9') {
                if (v < 1000000000LL) v = v * 10 + (*p - '0');
                else v = 1000000000LL;               /* 饱和，避免 long long 溢出 */
                p++;
            }
            nums[nnum++] = v;
        } else {
            p++;                                      /* 逐字节扫 ASCII 数字 */
        }
    }
    if (nnum == 2) { lo = nums[0]; hi = nums[1]; }
    const long long v = feng_rand_range(seed0, lo, hi);
    if (dice) snprintf(answer, (size_t)answer_sz, "掷骰子：%lld 点。", v);
    else snprintf(answer, (size_t)answer_sz, "随机数（%lld~%lld）：%lld。", lo, hi, v);
    return 1;
}
