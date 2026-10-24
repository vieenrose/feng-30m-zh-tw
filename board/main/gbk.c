#include "gbk.h"

/* 默认 0：**不编** gbk_table.c 那张 46,384 B 的双向映射表（3865 字），
 * 整个模块退化成「UTF-8 直通」，实现见文件末尾的 #else 分支。
 *
 * 为什么敢默认关：只有老式终端（SuperCom/XCOM 的 ANSI 模式）会发 GBK 字节；
 * 本项目所有脚本（esp32_chat / esp32_multi / …）、VS Code 串口、以及绝大多数
 * 现代终端都是 UTF-8。要 GBK 就 `idf.py -DFENG_GBK=ON build`。
 *
 * 只影响固件：PC 构建（build_pc_chat.ps1）的清单里没有本文件——pc_chat 不做
 * 串口编码转换。pc/gbk_selftest.cpp 是验真实转换的独立测试，单独编译时
 * 把 gbk_table.c 一起带上即可。 */
#if FENG_GBK

#include "gbk_table.h"

#include <stdint.h>
#include <string.h>

/* 默认 UTF-8（现代终端 + 本项目所有脚本）；老终端（SuperCom/XCOM 的 ANSI 模式）
 * 可以发 \gbk 切换，或在中文输入时自动跟随（GBK 字节不是合法 UTF-8）。 */
int g_out_gbk = 0;

static const uint16_t *u2g(uint32_t cp)
{
    int lo = 0, hi = g_gbk_count - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const uint32_t k = g_gbk_u2g_cp[mid];
        if (k == cp) return &g_gbk_u2g_val[mid];
        if (k < cp) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

static uint32_t g2u(uint16_t code)
{
    int lo = 0, hi = g_gbk_count - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const uint16_t k = g_gbk_g2u_val[mid];
        if (k == code) return g_gbk_g2u_cp[mid];
        if (k < code) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

/* length of the UTF-8 sequence starting at s (at most n bytes), 0 if malformed */
static int utf8_seq_len(const unsigned char *s, int n)
{
    if (n <= 0) return 0;
    const unsigned char c = s[0];
    if (c < 0x80) return 1;
    int need;
    uint32_t cp;
    if ((c & 0xE0) == 0xC0) { need = 1; cp = c & 0x1Fu; }
    else if ((c & 0xF0) == 0xE0) { need = 2; cp = c & 0x0Fu; }
    else if ((c & 0xF8) == 0xF0) { need = 3; cp = c & 0x07u; }
    else return 0;
    if (n < need + 1) return 0;
    for (int i = 1; i <= need; i++) {
        if ((s[i] & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (s[i] & 0x3Fu);
    }
    if (need == 1 && cp < 0x80) return 0;
    if (need == 2 && cp < 0x800) return 0;
    if (need == 3 && cp < 0x10000) return 0;
    if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
    if (cp > 0x10FFFF) return 0;       /* F4 90 80 80 这类编码超出行星 Unicode 范围 */
    return need + 1;
}

static uint32_t utf8_cp(const unsigned char *s, int len)
{
    if (len == 1) return s[0];
    uint32_t cp = 0;
    if (len == 2) cp = s[0] & 0x1Fu;
    else if (len == 3) cp = s[0] & 0x0Fu;
    else cp = s[0] & 0x07u;
    for (int i = 1; i < len; i++) cp = (cp << 6) | (s[i] & 0x3Fu);
    return cp;
}

static int utf8_put(uint32_t cp, char *out, int o, int out_max)
{
    if (cp < 0x80) {
        if (o < out_max) out[o++] = (char)cp;
    } else if (cp < 0x800) {
        if (o + 2 <= out_max) {
            out[o++] = (char)(0xC0 | (cp >> 6));
            out[o++] = (char)(0x80 | (cp & 0x3F));
        }
    } else if (cp < 0x10000) {
        if (o + 3 <= out_max) {
            out[o++] = (char)(0xE0 | (cp >> 12));
            out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[o++] = (char)(0x80 | (cp & 0x3F));
        }
    } else {
        if (o + 4 <= out_max) {
            out[o++] = (char)(0xF0 | (cp >> 18));
            out[o++] = (char)(0x80 | ((cp >> 12) & 0x3F));
            out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[o++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    return o;
}

int feng_utf8_valid(const char *s, int n)
{
    int i = 0;
    while (i < n) {
        const int l = utf8_seq_len((const unsigned char *)s + i, n - i);
        if (l == 0) return 0;
        i += l;
    }
    return 1;
}

int feng_has_high_byte(const char *s, int n)
{
    for (int i = 0; i < n; i++) {
        if ((unsigned char)s[i] >= 0x80) return 1;
    }
    return 0;
}

int feng_utf8_to_gbk(const char *in, int n, char *out, int out_max)
{
    int i = 0, o = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)in[i];
        if (c < 0x80) {
            if (o < out_max) out[o++] = (char)c;
            i++;
            continue;
        }
        const int l = utf8_seq_len((const unsigned char *)in + i, n - i);
        if (l == 0) {
            if (o < out_max) out[o++] = '?';
            i++;
            continue;
        }
        const uint16_t *v = u2g(utf8_cp((const unsigned char *)in + i, l));
        if (v && o + 2 <= out_max) {
            out[o++] = (char)(*v >> 8);
            out[o++] = (char)(*v & 0xFF);
        } else if (o < out_max) {
            out[o++] = '?';
        }
        i += l;
    }
    return o;
}

int feng_gbk_to_utf8(const char *in, int n, char *out, int out_max)
{
    int i = 0, o = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)in[i];
        if (c < 0x80) {
            if (o < out_max) out[o++] = (char)c;
            i++;
            continue;
        }
        uint32_t cp = 0;
        if (i + 1 < n) {
            cp = g2u((uint16_t)(((uint16_t)c << 8) | (unsigned char)in[i + 1]));
        }
        if (cp) {
            o = utf8_put(cp, out, o, out_max);
            i += 2;
        } else {
            if (o < out_max) out[o++] = '?';
            i++;
        }
    }
    return o;
}

#else /* !FENG_GBK ------------------------------------------------------------- */
/*
 * 退化实现：不编那张 46 KB 映射表，**永远按 UTF-8 收发**。
 *
 * 靠这三个语义把 GBK 路径自然关掉：
 *   · feng_has_high_byte 恒 0 -> 输入侧不做编码嗅探，g_out_gbk 始终是 0；
 *   · feng_utf8_valid    恒 1 -> 即使被调用，也永远判定"这是 UTF-8"；
 *   · 两个转换函数原样拷贝    -> 万一有别的调用点闯进来，也不会崩。
 *
 * main.c 里另有两处 `#if FENG_GBK`：`\gbk` 提示语、帮助/横幅不再列 `\gbk`。
 */

#include <string.h>

int g_out_gbk = 0;

int feng_utf8_valid(const char *s, int n)
{
    (void)s; (void)n;
    return 1;                       /* 恒真 -> 永不切到 GBK */
}

int feng_has_high_byte(const char *s, int n)
{
    (void)s; (void)n;
    return 0;                       /* 恒假 -> 输入侧不嗅探 */
}

static int passthrough(const char *in, int n, char *out, int out_max)
{
    if (out_max <= 0) return 0;
    int m = (n < out_max - 1) ? n : (out_max - 1);
    if (m < 0) m = 0;
    if (m > 0) memcpy(out, in, (size_t)m);
    out[m] = 0;
    return m;
}

int feng_utf8_to_gbk(const char *in, int n, char *out, int out_max)
{
    return passthrough(in, n, out, out_max);
}

int feng_gbk_to_utf8(const char *in, int n, char *out, int out_max)
{
    return passthrough(in, n, out, out_max);
}

#endif /* FENG_GBK */
