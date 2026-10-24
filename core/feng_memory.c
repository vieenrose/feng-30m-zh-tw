/* 引擎侧记忆 tool 的实现：纯 C、无动态内存，板端/PC/Python 同口径。
 *
 * 存储分三块：
 *   - 专用槽：名字（我叫X）、城市（住在/搬到X）、宠物（养了X）——它们有固定的追问句式
 *   - 通用槽：键值表（"最喜欢的<键>是<值>"、"我的<键>是<值>"）——覆盖
 *     颜色/运动/食物/书/电影/生日/职业…等任意短键
 */
#include "feng_memory.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define VAL_CAP 32
#define KEY_CAP 24
#define MAX_SLOTS 8

/* 返回 UTF-8 序列长度；源剩余字节不足、续字节非法或编码越界时返回 0。
 * 旧实现只看首字节就假定有 2/3/4 字节，输入末尾恰好是 0xE4 这类首字节时
 * 会读过 NUL（ASAN 实测 take_value 越界 3 字节）。 */
static int u8seq(const char *p, size_t rem)
{
    if (p == NULL || rem == 0) return 0;
    const unsigned char b = (unsigned char)p[0];
    int need;
    uint32_t cp;
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) { need = 1; cp = b & 0x1Fu; }
    else if ((b & 0xF0) == 0xE0) { need = 2; cp = b & 0x0Fu; }
    else if ((b & 0xF8) == 0xF0) { need = 3; cp = b & 0x07u; }
    else return 0;
    if (rem < (size_t)need + 1) return 0;
    for (int i = 1; i <= need; i++) {
        if (((unsigned char)p[i] & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | ((unsigned char)p[i] & 0x3Fu);
    }
    if (need == 1 && cp < 0x80) return 0;
    if (need == 2 && cp < 0x800) return 0;
    if (need == 3 && (cp < 0x10000 || cp > 0x10FFFF)) return 0;
    if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
    return need + 1;
}

static void copy_utf8(char *dst, int cap, const char *src);

/* 有界追加：n 是实际长度，绝不越过 cap-1；返回 0 表示发生截断。 */
static int mem_append_fmt(char *out, int cap, int *n, const char *fmt, ...)
{
    if (out == NULL || n == NULL || cap <= 0) return 0;
    if (*n < 0) *n = 0;
    if (*n >= cap - 1) { out[cap - 1] = 0; return 0; }
    va_list ap;
    va_start(ap, fmt);
    const int room = cap - 1 - *n;
    const int w = vsnprintf(out + *n, (size_t)room, fmt, ap);
    va_end(ap);
    if (w < 0) { out[*n] = 0; return 0; }
    if (w >= room) { *n = cap - 1; out[*n] = 0; return 0; }
    *n += w;
    out[*n] = 0;
    return 1;
}

typedef struct {
    char key[KEY_CAP];
    char val[VAL_CAP];
    int used;
} slot_t;

static char g_name[VAL_CAP];
static char g_city[VAL_CAP];
static char g_pet[VAL_CAP];
static int g_has_name, g_has_city, g_has_pet;
static slot_t g_slots[MAX_SLOTS];
static int g_slot_next;
/* 墓碑：用户明确说"忘掉"的键。引擎忘掉后，该键的追问由引擎答"不记得了"，
 * 避免模型凭对话上下文又把这条说出来（重新学习会解除墓碑）。 */
static char g_forgot[MAX_SLOTS][KEY_CAP];
static int g_nforgot;
/* 记忆是否自上次落盘后发生过变化。前端不能靠“输入里有没有忘字”猜：
 * “别记/删掉/不要记”同样会删事实，却一个“忘”字都没有。 */
static int g_dirty;

int feng_mem_dirty(void) { return g_dirty; }
void feng_mem_mark_clean(void) { g_dirty = 0; }

static void mark_forgot(const char *key)
{
    if (!key[0]) return;
    for (int i = 0; i < g_nforgot; i++) if (strcmp(g_forgot[i], key) == 0) return;
    if (g_nforgot < MAX_SLOTS) snprintf(g_forgot[g_nforgot++], KEY_CAP, "%s", key);
}

static int is_forgot(const char *key)
{
    for (int i = 0; i < g_nforgot; i++) {
        if (strcmp(g_forgot[i], key) == 0) return 1;
    }
    return 0;
}

static void clear_forgot(const char *key)
{
    for (int i = 0; i < g_nforgot; i++) {
        if (strcmp(g_forgot[i], key) != 0) continue;
        for (int j = i; j + 1 < g_nforgot; j++) memcpy(g_forgot[j], g_forgot[j + 1], KEY_CAP);
        g_nforgot--;
        return;
    }
}

void feng_mem_clear(void)
{
    g_name[0] = g_city[0] = g_pet[0] = 0;
    g_has_name = g_has_city = g_has_pet = 0;
    memset(g_slots, 0, sizeof(g_slots));
    g_slot_next = 0;
    g_nforgot = 0;
    g_dirty = 1;
}

/* ---- 通用键值表 ---- */
static void slot_put(const char *key, const char *val)
{
    if (!val[0]) return;                               /* 空键 = 无名偏好槽（"我最喜欢X"） */
    for (int i = 0; i < MAX_SLOTS; i++) {              /* 同键覆盖 */
        if (g_slots[i].used && strcmp(g_slots[i].key, key) == 0) {
            copy_utf8(g_slots[i].val, VAL_CAP, val);
            return;
        }
    }
    for (int i = 0; i < MAX_SLOTS; i++) {              /* 空位 */
        if (!g_slots[i].used) {
            copy_utf8(g_slots[i].key, KEY_CAP, key);
            copy_utf8(g_slots[i].val, VAL_CAP, val);
            g_slots[i].used = 1;
            return;
        }
    }
    slot_t *s = &g_slots[g_slot_next++ % MAX_SLOTS];   /* 满了就轮换覆盖最旧的 */
    copy_utf8(s->key, KEY_CAP, key);
    copy_utf8(s->val, VAL_CAP, val);
}

static const char *slot_get(const char *key)
{
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (g_slots[i].used && strcmp(g_slots[i].key, key) == 0) return g_slots[i].val;
    }
    return NULL;
}

/* 清掉某一项（名字/城市/宠物/任意键槽）；返回 1 = 确实清掉了 */
static void trim_tail(char *out);      /* 定义在下方 */
static void norm_forget_key(char *key)
{
    trim_tail(key);
    static const char *pfx[] = {"我的", "你记的", "你记住的", "关于"};
    for (unsigned i = 0; i < sizeof(pfx) / sizeof(pfx[0]); i++) {
        const size_t lp = strlen(pfx[i]);
        if (strncmp(key, pfx[i], lp) == 0) {
            memmove(key, key + lp, strlen(key + lp) + 1);
            break;
        }
    }
    trim_tail(key);
}

static int mem_forget_key(const char *key)
{
    if (strcmp(key, "名字") == 0) {
        const int had = g_has_name;
        g_has_name = 0;
        mark_forgot("名字");
        return had;
    }
    if (strcmp(key, "城市") == 0 || strcmp(key, "住的地方") == 0) {
        const int had = g_has_city;
        g_has_city = 0;
        mark_forgot("城市");
        return had;
    }
    if (strcmp(key, "宠物") == 0) {
        const int had = g_has_pet;
        g_has_pet = 0;
        mark_forgot("宠物");
        return had;
    }
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (g_slots[i].used && strcmp(g_slots[i].key, key) == 0) {
            g_slots[i].used = 0;
            mark_forgot(key);
            return 1;
        }
    }
    mark_forgot(key);              /* 没记过也记墓碑：之后追问统一答"不记得" */
    return 0;
}

/* 把已记事实列成一句话；返回 0 = 什么都没记 */
static int mem_list(char *out, int cap)
{
    if (out == NULL || cap <= 1) return 0;
    int n = 0, cnt = 0;
    out[0] = 0;
    mem_append_fmt(out, cap, &n, "我记得：");
    if (g_has_name) { mem_append_fmt(out, cap, &n, "你叫%s；", g_name); cnt++; }
    if (g_has_city) { mem_append_fmt(out, cap, &n, "你住在%s；", g_city); cnt++; }
    if (g_has_pet) { mem_append_fmt(out, cap, &n, "你养了%s；", g_pet); cnt++; }
    for (int i = 0; i < MAX_SLOTS && cnt < 6; i++) {
        if (!g_slots[i].used) continue;
        if (g_slots[i].key[0])
            mem_append_fmt(out, cap, &n, "你的%s是%s；", g_slots[i].key, g_slots[i].val);
        else
            mem_append_fmt(out, cap, &n, "你最喜欢%s；", g_slots[i].val);
        cnt++;
    }
    if (cnt == 0) return 0;
    n = (int)strlen(out);
    if (n >= 3 && strcmp(out + n - 3, "；") == 0) { out[n - 3] = 0; n -= 3; }
    mem_append_fmt(out, cap, &n, "。");
    return 1;
}

/* ---- 文本工具 ---- */
/* 按完整 UTF-8 字符截断，避免 snprintf 在字节边界把汉字劈成半个。
 * 非法/截断序列直接停止，绝不复制超出源字符串的部分。 */
static void copy_utf8(char *dst, int cap, const char *src)
{
    int n = 0;
    if (dst == NULL || cap <= 0) return;
    if (src == NULL) { dst[0] = 0; return; }
    const size_t total = strlen(src);
    while (n < cap - 1) {
        const int l = u8seq(src + n, total - (size_t)n);
        if (l <= 0 || n + l > cap - 1) break;
        memcpy(dst + n, src + n, (size_t)l);
        n += l;
    }
    dst[n] = 0;
}

static int starts_with(const char *s, const char *pfx)
{
    return strncmp(s, pfx, strlen(pfx)) == 0;
}

static int is_term(const char *p)
{
    static const char *t[] = {"。", "，", "、", "！", "？", "；", ",", ".", "!", "?", ";",
                              "\n", "\r", "的"};
    for (unsigned i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        if (starts_with(p, t[i])) return 1;
    }
    return 0;
}

static int looks_like_question(const char *v)
{
    static const char *kw[] = {"什么", "啥", "哪", "几", "谁", "多少", "吗", "呢", "怎么"};
    for (unsigned i = 0; i < sizeof(kw) / sizeof(kw[0]); i++) {
        if (starts_with(v, kw[i])) return 1;
    }
    return 0;
}

static void trim_tail(char *out)
{
    static const char *t[] = {"请记住", "一下", "了", "呀", "啊", "哦", "吧", "嘛", "呢"};
    int changed = 1;
    while (changed && *out) {
        changed = 0;
        for (unsigned i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
            const size_t lt = strlen(t[i]);
            const size_t lo = strlen(out);
            if (lo >= lt && strcmp(out + lo - lt, t[i]) == 0) {
                out[lo - lt] = 0;
                changed = 1;
            }
        }
    }
}

/* 取一段值：到句读/换行或串尾；去尾部虚词；问句返回失败 */
static int take_value(const char *p, char *out, int cap)
{
    if (p == NULL || out == NULL || cap <= 0) return 0;
    while (*p) {
        if (*p == ' ' || *p == '\t' || *p == ':') { p++; continue; }
        if (starts_with(p, "：")) { p += 3; continue; }   /* 全角冒号是 3 字节 */
        break;
    }
    const char *start = p;
    const size_t total = strlen(start);
    int n = 0;
    while (*p && n < cap - 1) {
        if (is_term(p)) break;
        const int l = u8seq(p, total - (size_t)(p - start));
        if (l <= 0) { p++; continue; }                    /* 跳过非法字节，不越过 NUL */
        if (n + l >= cap) break;
        memcpy(out + n, p, (size_t)l);
        n += l;
        p += l;
    }
    out[n] = 0;
    trim_tail(out);
    while (*out == ' ') memmove(out, out + 1, strlen(out));
    if (!out[0] || looks_like_question(out)) return 0;
    return 1;
}

/* 取一段"键"：到 stop 串或句读为止，允许含"的"（调用方自行 trim） */
static int take_key_to(const char *p, const char *stop, char *out, int cap)
{
    if (p == NULL || out == NULL || cap <= 0) return 0;
    const char *start = p;
    const size_t total = strlen(start);
    int n = 0;
    while (*p && n < cap - 1) {
        if (stop && starts_with(p, stop)) break;
        if (starts_with(p, "。") || starts_with(p, "？") || starts_with(p, "，") ||
            starts_with(p, "！") || starts_with(p, "、") ||
            *p == '?' || *p == '!' || *p == '\n' || *p == '\r' || *p == ',') break;
        const int l = u8seq(p, total - (size_t)(p - start));
        if (l <= 0) { p++; continue; }
        if (n + l >= cap) break;
        memcpy(out + n, p, (size_t)l);
        n += l;
        p += l;
    }
    out[n] = 0;
    while (out[0] && out[0] == ' ') memmove(out, out + 1, strlen(out));
    while (out[0]) {                                   /* 去掉开头的"的"和尾部空格 */
        const size_t lo = strlen(out);
        if (lo >= 3 && strcmp(out, "的") == 0) { out[0] = 0; break; }
        if (lo >= 3 && starts_with(out, "的")) { memmove(out, out + 3, lo - 3 + 1); continue; }
        if (out[lo - 1] == ' ') { out[lo - 1] = 0; continue; }
        break;
    }
    return out[0] != 0;
}

static int take_key_before(const char *start, const char *stop_kw, char *out, int cap)
{
    const char *stop = strstr(start, stop_kw);
    if (!stop) return 0;
    char tmp[KEY_CAP * 2];
    const size_t len = (size_t)(stop - start);
    if (len == 0 || len >= sizeof(tmp)) return 0;
    memcpy(tmp, start, len);
    tmp[len] = 0;
    return take_key_to(tmp, NULL, out, cap);           /* 复用清理逻辑 */
}

/* 明确的 [start, end) 区间版本（学习路径用：end 是"是/改成/换成"的指针） */
static int take_key_range(const char *start, const char *end, char *out, int cap)
{
    const size_t len = (size_t)(end - start);
    char tmp[KEY_CAP * 2];
    if (len == 0 || len >= sizeof(tmp)) return 0;
    memcpy(tmp, start, len);
    tmp[len] = 0;
    return take_key_to(tmp, NULL, out, cap);
}

/* “我是X”并不总是报名字：老师/程序员/退休/本地人都是身份描述。
 * 旧实现把它们记成名字，随后“我叫什么名字”会答“你叫老师”。 */
static int looks_like_role(const char *v)
{
    static const char *roles[] = {
        "老师", "教师", "教授", "学生", "同学", "医生", "护士", "律师", "工程师",
        "程序员", "设计师", "会计", "司机", "工人", "农民", "记者", "作家", "画家",
        "歌手", "演员", "老板", "经理", "警察", "军人", "运动员", "退休", "自由职业",
        "上班族", "妈妈", "爸爸", "儿子", "女儿", "孩子", "大人", "老人", "男人",
        "女人", "男孩", "女孩", "中国人", "外国人", "本地人", "新手", "小白",
        "人工智能", "机器人", "模型", "助手", "AI", "ai",
    };
    for (unsigned i = 0; i < sizeof(roles) / sizeof(roles[0]); i++)
        if (strstr(v, roles[i]) != NULL) return 1;
    return 0;
}

/* ---- 学习 ---- */
int feng_mem_learn(const char *user)
{
    int changed = 0;
    char v[VAL_CAP], key[KEY_CAP];

    /* 名字（专用）：我的名字是X / 我叫X / 我是X（短名字） */
    if (take_value(strstr(user, "我的名字是") ? strstr(user, "我的名字是") + strlen("我的名字是") : "", v, sizeof(v))) {
        snprintf(g_name, VAL_CAP, "%s", v); g_has_name = 1; clear_forgot("名字"); changed = 1;
    } else if (strstr(user, "我叫") && take_value(strstr(user, "我叫") + strlen("我叫"), v, sizeof(v))) {
        snprintf(g_name, VAL_CAP, "%s", v); g_has_name = 1; clear_forgot("名字"); changed = 1;
    }
    if (!g_has_name) {
        const char *p = strstr(user, "我是");
        if (p && take_value(p + strlen("我是"), v, sizeof(v)) &&
            strlen(v) <= 12 && !looks_like_role(v) &&
            !strstr(v, "学生") && !strstr(v, "模型") && !strstr(v, "AI")) {
            snprintf(g_name, VAL_CAP, "%s", v); g_has_name = 1; changed = 1;
            clear_forgot("名字");
        }
    }

    /* 城市 / 宠物（专用，问法固定） */
    const char *p;
    if ((p = strstr(user, "住在")) && take_value(p + strlen("住在"), v, sizeof(v))) {
        snprintf(g_city, VAL_CAP, "%s", v); g_has_city = 1; clear_forgot("城市"); changed = 1;
    } else if ((p = strstr(user, "搬到")) && take_value(p + strlen("搬到"), v, sizeof(v))) {
        snprintf(g_city, VAL_CAP, "%s", v); g_has_city = 1; clear_forgot("城市"); changed = 1;
    }
    if ((p = strstr(user, "养了一只")) && take_value(p + strlen("养了一只"), v, sizeof(v))) {
        snprintf(g_pet, VAL_CAP, "%s", v); g_has_pet = 1; clear_forgot("宠物"); changed = 1;
    } else if ((p = strstr(user, "养的是")) && take_value(p + strlen("养的是"), v, sizeof(v))) {
        snprintf(g_pet, VAL_CAP, "%s", v); g_has_pet = 1; clear_forgot("宠物"); changed = 1;
    } else if ((p = strstr(user, "养了")) && take_value(p + strlen("养了"), v, sizeof(v))) {
        snprintf(g_pet, VAL_CAP, "%s", v); g_has_pet = 1; clear_forgot("宠物"); changed = 1;
    }

    /* 通用槽 1：最喜欢的<键>是/改成/换成<值>（键可为空 -> 无名偏好，如"我最喜欢蛋糕"） */
    if ((p = strstr(user, "最喜欢的")) != NULL) {
        static const char *marks[] = {"是", "改成", "换成"};
        const char *m = NULL;
        size_t mlen = 0;
        for (unsigned i = 0; i < sizeof(marks) / sizeof(marks[0]); i++) {
            const char *q = strstr(p + strlen("最喜欢的"), marks[i]);
            if (q && (!m || q < m)) { m = q; mlen = strlen(marks[i]); }
        }
        const char *k0 = p + strlen("最喜欢的");
        if (m && m - k0 <= KEY_CAP * 2 && take_key_range(k0, m, key, sizeof(key)) &&
            take_value(m + mlen, v, sizeof(v))) {
            slot_put(key, v);
            clear_forgot(key);
            changed = 1;
        }
    } else if ((p = strstr(user, "最喜欢吃")) != NULL || (p = strstr(user, "喜欢吃")) != NULL ||
               (p = strstr(user, "最喜欢")) != NULL) {
        const char *v0;
        if (starts_with(p, "最喜欢吃")) v0 = p + strlen("最喜欢吃");
        else if (starts_with(p, "喜欢吃")) v0 = p + strlen("喜欢吃");
        else v0 = p + strlen("最喜欢");
        if (take_value(v0, v, sizeof(v))) {            /* 无名偏好槽（键 = ""） */
            slot_put("", v);
            clear_forgot("");
            changed = 1;
        }
    }

    /* 通用槽 2：我的<键>是<值>（生日/职业/家乡…；"名字"已在上面处理） */
    if ((p = strstr(user, "我的")) != NULL) {
        const char *k0 = p + strlen("我的");
        static const char *marks2[] = {"是", "叫"};
        const char *m = NULL;
        size_t mlen = 0;
        for (unsigned i = 0; i < sizeof(marks2) / sizeof(marks2[0]); i++) {
            const char *q = strstr(k0, marks2[i]);
            if (q && (!m || q < m)) { m = q; mlen = strlen(marks2[i]); }
        }
        if (m && take_key_range(k0, m, key, sizeof(key)) && strcmp(key, "名字") != 0 &&
            take_value(m + mlen, v, sizeof(v))) {
            if (strcmp(key, "宠物") == 0) {            /* 我的宠物是猫 -> 专用宠物槽 */
                snprintf(g_pet, VAL_CAP, "%s", v);
                g_has_pet = 1;
                clear_forgot("宠物");
            } else {
                slot_put(key, v);
                clear_forgot(key);
            }
            changed = 1;
        }
    }
    if (changed) g_dirty = 1;
    return changed;
}

static int answer_fmt(char *out, int cap, const char *fmt, const char *a, const char *b)
{
    /* cap 来自调用方的 sizeof()，恒为正；仍按有界写处理，与 feng_calc 的口径一致 */
    if (out == NULL || cap <= 0) return 1;
    snprintf(out, (size_t)cap, fmt, a, b);
    return 1;
}

/* ---- 回答 ---- */
int feng_mem_answer(const char *user, char *answer, int answer_sz)
{
    /* 身份（固定口径） */
    static const char *id_q[] = {"你叫什么名字", "你叫什么", "你是谁", "你的名字是什么",
                                 "你是谁开发的", "谁训练了你"};
    for (unsigned i = 0; i < sizeof(id_q) / sizeof(id_q[0]); i++) {
        if (strstr(user, id_q[i])) {
            /* 身份自述必须与项目口径逐字一致（AGENTS.md §1）：
             * 「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」。
             * 这段由 tool 确定性回答，不经过模型，所以模板本身就是用户看到的口径。 */
            return answer_fmt(answer, answer_sz, "%s",
                              "我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。", "");
        }
    }
    /* 遗忘优先于其它追问：忘掉我的生日 / 别记我的名字了 / 把记住的都忘掉 */
    {
        static const char *fv[] = {"忘掉", "忘记", "别记", "删掉", "不要记"};
        const char *f = NULL;
        size_t flen = 0;
        for (unsigned i = 0; i < sizeof(fv) / sizeof(fv[0]); i++) {
            const char *r = strstr(user, fv[i]);
            if (r && (!f || r < f)) { f = r; flen = strlen(fv[i]); }
        }
        if (f) {
            /* 否定保护：“不要忘掉/别忘/不能删”表达的是保留，不是删除。
             * 旧实现只找“忘掉/删掉”子串，会把这类请求真的删掉。 */
            static const char *keep[] = {"不要忘", "别忘", "不能忘", "不想忘",
                                         "不要删", "别删", "不准忘", "不要忘记"};
            for (unsigned i = 0; i < sizeof(keep) / sizeof(keep[0]); i++) {
                if (strstr(user, keep[i]) != NULL) return 0;
            }
            char key[KEY_CAP];
            take_key_to(f + flen, NULL, key, sizeof(key));
            norm_forget_key(key);
            /* "都/全部/一切" 视作全清 */
            if (!key[0] || strstr(key, "都") || strstr(key, "全部") || strstr(key, "一切") ||
                strstr(key, "所有")) {
                feng_mem_clear();
                return answer_fmt(answer, answer_sz, "%s", "好，我把记住的这些都忘掉了。", "");
            }
            if (mem_forget_key(key)) {
                g_dirty = 1;
                return answer_fmt(answer, answer_sz, "好，我忘掉了你的%s。", key, "");
            }
            g_dirty = 1;                    /* 墓碑也要落盘，重启后才不会又答旧值 */
            return answer_fmt(answer, answer_sz, "我没有记过你的%s。", key, "");
        }
    }
    /* 列出已记事实 */
    if (strstr(user, "记得什么") || strstr(user, "记住什么") || strstr(user, "记住哪些") ||
        strstr(user, "记住了什么") || strstr(user, "记得哪些") || strstr(user, "记忆里有什么") ||
        strstr(user, "都记住了") || strstr(user, "记得的东西")) {
        if (mem_list(answer, answer_sz)) return 1;
        return answer_fmt(answer, answer_sz, "%s", "我还没有记住你的信息。", "");
    }
    /* 用户名字 */
    if (strstr(user, "我叫什么") || strstr(user, "我叫啥") || strstr(user, "记得我叫什么") ||
        strstr(user, "我的名字是什么") || strstr(user, "我的名字是啥")) {
        if (g_has_name) return answer_fmt(answer, answer_sz, "你叫%s。", g_name, "");
        if (is_forgot("名字")) return answer_fmt(answer, answer_sz, "%s", "我不记得你的名字了。", "");
        return 0;
    }
    /* 专用槽 */
    if (strstr(user, "住在哪") || strstr(user, "哪个城市") || strstr(user, "什么地方住")) {
        if (g_has_city) return answer_fmt(answer, answer_sz, "你住在%s。", g_city, "");
        if (is_forgot("城市")) return answer_fmt(answer, answer_sz, "%s", "我不记得你住在哪里了。", "");
        return 0;
    }
    if (strstr(user, "养了什么") || strstr(user, "养了啥") || strstr(user, "我的宠物") ||
        strstr(user, "养的什么宠物")) {
        if (g_has_pet) return answer_fmt(answer, answer_sz, "你养了%s。", g_pet, "");
        if (is_forgot("宠物")) return answer_fmt(answer, answer_sz, "%s", "我不记得你养了什么了。", "");
        return 0;
    }
    /* 通用槽：最喜欢<什么键>？ / 最喜欢的<键>是什么？ */
    char key[KEY_CAP] = "";
    if (!strstr(user, "我")) return 0;      /* "你最喜欢什么颜色"问的是助手，不是用户记忆 */
    int is_query = 0;
    const char *q = strstr(user, "喜欢什么");      /* 覆盖"最喜欢什么X"和"喜欢什么X" */
    if (q) {
        take_key_to(q + strlen("喜欢什么"), NULL, key, sizeof(key));
        is_query = 1;
    } else if ((q = strstr(user, "喜欢啥")) != NULL) {
        take_key_to(q + strlen("喜欢啥"), NULL, key, sizeof(key));
        is_query = 1;
    } else if ((q = strstr(user, "最喜欢的")) != NULL) {
        /* 只有"…是什么/是啥"才是追问；"我最喜欢的书是《小王子》"是陈述，不能回无名偏好槽 */
        if (take_key_before(q + strlen("最喜欢的"), "是什么", key, sizeof(key)) ||
            take_key_before(q + strlen("最喜欢的"), "是啥", key, sizeof(key))) {
            is_query = 1;
        }
    } else if (strstr(user, "我最喜欢什么") || strstr(user, "我喜欢什么") ||
               strstr(user, "喜欢吃什么")) {
        key[0] = 0;                                     /* 无名偏好槽 */
        is_query = 1;
    }
    if (is_query) {
        const char *v = slot_get(key);
        if (v) return answer_fmt(answer, answer_sz, "你最喜欢%s。", v, "");
        if (is_forgot(key)) {
            if (key[0]) return answer_fmt(answer, answer_sz, "我不记得你的%s了。", key, "");
            return answer_fmt(answer, answer_sz, "%s", "我不记得你最喜欢什么了。", "");
        }
        return 0;
    }
    /* 通用槽：我的<键>是什么/是多少/是几号… */
    if ((q = strstr(user, "我的")) != NULL) {
        static const char *marks[] = {"是什么", "是啥", "是多少", "是几号", "是哪个", "是哪里", "是几"};
        const char *m = NULL;
        for (unsigned i = 0; i < sizeof(marks) / sizeof(marks[0]); i++) {
            const char *r = strstr(q + strlen("我的"), marks[i]);
            if (r && (!m || r < m)) { m = r; }
        }
        if (m && take_key_before(q + strlen("我的"), m, key, sizeof(key)) &&
            strcmp(key, "名字") != 0) {
            const char *v = slot_get(key);
            if (v) return answer_fmt(answer, answer_sz, "你的%s是%s。", key, v);
            if (is_forgot(key)) return answer_fmt(answer, answer_sz, "我不记得你的%s了。", key, "");
        }
    }
    return 0;
}

void feng_mem_snapshot(char *buf, int buf_sz)
{
    if (buf == NULL || buf_sz <= 1) return;
    int n = 0;
    buf[0] = 0;
    mem_append_fmt(buf, buf_sz, &n, "名字=%s", g_has_name ? g_name : "-");
    mem_append_fmt(buf, buf_sz, &n, " 城市=%s", g_has_city ? g_city : "-");
    mem_append_fmt(buf, buf_sz, &n, " 宠物=%s", g_has_pet ? g_pet : "-");
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (!g_slots[i].used) continue;
        mem_append_fmt(buf, buf_sz, &n, " %s=%s",
                       g_slots[i].key[0] ? g_slots[i].key : "(偏好)", g_slots[i].val);
    }
}

/* 持久化文本格式（每行）：name/city/pet<TAB>值、slot<TAB>键<TAB>值、forgot<TAB>键。
 * 值里的 TAB/换行在导出时替换成空格，避免一条事实把文件格式撑破。 */
static void export_val(char *dst, int cap, const char *src)
{
    int i = 0;
    if (cap <= 0) return;
    for (; src[i] && i < cap - 1; i++) {
        const char c = src[i];
        dst[i] = (c == '\t' || c == '\n' || c == '\r') ? ' ' : c;
    }
    dst[i] = 0;
}

int feng_mem_export(char *out, int cap)
{
    if (out == NULL || cap <= 1) return -1;
    int n = 0, ok = 1;
    char v[VAL_CAP + 4];
    out[0] = 0;
    if (g_has_name) {
        export_val(v, sizeof(v), g_name);
        ok &= mem_append_fmt(out, cap, &n, "name\t%s\n", v);
    }
    if (g_has_city) {
        export_val(v, sizeof(v), g_city);
        ok &= mem_append_fmt(out, cap, &n, "city\t%s\n", v);
    }
    if (g_has_pet) {
        export_val(v, sizeof(v), g_pet);
        ok &= mem_append_fmt(out, cap, &n, "pet\t%s\n", v);
    }
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (!g_slots[i].used) continue;
        char k[KEY_CAP + 4];
        export_val(k, sizeof(k), g_slots[i].key);
        export_val(v, sizeof(v), g_slots[i].val);
        ok &= mem_append_fmt(out, cap, &n, "slot\t%s\t%s\n", k, v);
    }
    for (int i = 0; i < g_nforgot; i++) {
        char k[KEY_CAP + 4];
        export_val(k, sizeof(k), g_forgot[i]);
        ok &= mem_append_fmt(out, cap, &n, "forgot\t%s\n", k);
    }
    /* 头文件约定：缓冲不足返回 -1，调用方不能把半条记录当完整快照写盘。 */
    if (!ok) { out[0] = 0; return -1; }
    return n;
}

int feng_mem_import(const char *text)
{
    if (text == NULL) return 0;
    feng_mem_clear();
    int got = 0;
    const char *p = text;
    while (*p) {
        const char *e = strchr(p, '\n');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        char line[256];
        if (len >= sizeof(line)) len = sizeof(line) - 1;
        memcpy(line, p, len);
        line[len] = 0;
        if (len > 0 && line[len - 1] == '\r') line[--len] = 0;
        char *t1 = strchr(line, '\t');
        if (t1 != NULL) {
            *t1 = 0;
            char *t2 = strchr(t1 + 1, '\t');
            if (t2 != NULL) *t2 = 0;
            if (strcmp(line, "name") == 0 && t1[1]) {
                copy_utf8(g_name, VAL_CAP, t1 + 1);
                g_has_name = g_name[0] != 0;
                clear_forgot("名字");
                got |= g_has_name;
            } else if (strcmp(line, "city") == 0 && t1[1]) {
                copy_utf8(g_city, VAL_CAP, t1 + 1);
                g_has_city = g_city[0] != 0;
                clear_forgot("城市");
                got |= g_has_city;
            } else if (strcmp(line, "pet") == 0 && t1[1]) {
                copy_utf8(g_pet, VAL_CAP, t1 + 1);
                g_has_pet = g_pet[0] != 0;
                clear_forgot("宠物");
                got |= g_has_pet;
            } else if (strcmp(line, "slot") == 0 && t2 != NULL && t2[1]) {
                slot_put(t1 + 1, t2 + 1);
                got = 1;
            } else if (strcmp(line, "forgot") == 0 && t1[1]) {
                mark_forgot(t1 + 1);
                got = 1;
            }
        }
        p = e ? e + 1 : p + len;
    }
    if (got) g_dirty = 1;
    return got;
}
