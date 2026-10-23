#!/usr/bin/env bash
# 手机引擎验收：拿板端原始引擎（build_baseline.sh 的产物）当对照，逐字节比对。
#
# 判据（全部必须通过）：
#   1) 手机引擎 单线程 / NEON=0  -> 与基线输出**逐字节相同**（证明平台层改造没有副作用）
#   2) 手机引擎 4 线程 / NEON=0  -> 与基线输出**逐字节相同**（证明按行切分并行是位精确的）
#   3) 算术子集 21/21（tool 口径）
#   4) 多轮套件 mem12 12/12
#
# 用法：./verify.sh [model_export 目录]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# MSYS2 / Cygwin：pwd 给 POSIX 路径（/e/...），原生 PE 编译器/程序不认；
# pwd -W 给 Windows 路径。Linux/macOS 没有 -W，保持原样。
if W="$(cd "$HERE" && pwd -W 2>/dev/null)" && [ -n "$W" ]; then
    HERE="$W"
    R="$(cd "$ROOT" && pwd -W 2>/dev/null || true)"
    [ -n "$R" ] && ROOT="$R"
fi
MODEL="${1:-$ROOT/board/model_export_4.0-embed}"
FILLER="$ROOT/pc/prompt_long.txt"
LOGDIR="$ROOT/logs"
BASE="$HERE/build-baseline"

# Windows 上产物是 .exe；统一用这个函数解析，Linux/macOS 行为不变。
exe() {
    if [ -x "$1" ]; then printf '%s' "$1"
    elif [ -x "$1.exe" ]; then printf '%s' "$1.exe"
    else printf '%s' "$1"; fi
}

NEW="$HERE/build"

mkdir -p "$LOGDIR"
rc=0
ok()   { printf '  [PASS] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; rc=1; }

[ -f "$MODEL/model.bin" ] || { echo "找不到 $MODEL/model.bin" >&2; exit 2; }
for b in "$(exe $BASE/pc_kv_suite_base)" "$(exe $NEW/pc_kv_suite_phone)"; do
    [ -x "$b" ] || { echo "缺少 $b，先跑 build_baseline.sh / build.sh" >&2; exit 2; }
done

echo "== 模型 $MODEL"
echo

echo "== 1) 32 题矩阵：基线 vs 手机引擎（单线程/NEON 关）"
FENG_PHONE_THREADS=1 FENG_PHONE_NEON=0 "$(exe $BASE/pc_kv_suite_base)" "$MODEL" "$FILLER" 5200 > "$LOGDIR/phone_base_kv.txt" 2>&1
FENG_PHONE_THREADS=1 FENG_PHONE_NEON=0 "$(exe $NEW/pc_kv_suite_phone)" "$MODEL" "$FILLER" 5200 > "$LOGDIR/phone_1t_kv.txt" 2>&1
tail -1 "$LOGDIR/phone_base_kv.txt"
tail -1 "$LOGDIR/phone_1t_kv.txt"
if diff -q "$LOGDIR/phone_base_kv.txt" "$LOGDIR/phone_1t_kv.txt" >/dev/null; then
    ok "单线程输出与基线逐字节相同"
else
    bad "单线程输出与基线不同（diff 见 $LOGDIR/phone_1t_kv.diff）"
    diff "$LOGDIR/phone_base_kv.txt" "$LOGDIR/phone_1t_kv.txt" > "$LOGDIR/phone_1t_kv.diff"
fi
echo

echo "== 2) 32 题矩阵：手机引擎 4 线程（按行切分，应仍逐字节相同）"
FENG_PHONE_THREADS=4 FENG_PHONE_NEON=0 "$(exe $NEW/pc_kv_suite_phone)" "$MODEL" "$FILLER" 5200 > "$LOGDIR/phone_4t_kv.txt" 2>&1
tail -1 "$LOGDIR/phone_4t_kv.txt"
if diff -q "$LOGDIR/phone_base_kv.txt" "$LOGDIR/phone_4t_kv.txt" >/dev/null; then
    ok "4 线程输出与基线逐字节相同（并行位精确）"
else
    bad "4 线程输出与基线不同（diff 见 $LOGDIR/phone_4t_kv.diff）"
    diff "$LOGDIR/phone_base_kv.txt" "$LOGDIR/phone_4t_kv.txt" > "$LOGDIR/phone_4t_kv.diff"
fi
echo

echo "== 3) 算术子集（tool 口径，21/21）"
FENG_PHONE_THREADS=4 FENG_PHONE_NEON=0 FENG_SUITE=arith "$(exe $NEW/pc_kv_suite_phone)" "$MODEL" "$FILLER" 5200 > "$LOGDIR/phone_arith.txt" 2>&1
tail -1 "$LOGDIR/phone_arith.txt"
if tail -1 "$LOGDIR/phone_arith.txt" | grep -q '短任务 21/21'; then
    ok "算术 21/21"
else
    bad "算术不是 21/21"
fi
echo

echo "== 4) 多轮套件：基线 vs 手机引擎（应逐字节相同）"
FENG_PHONE_THREADS=1 FENG_PHONE_NEON=0 "$(exe $BASE/pc_mt_suite_base)" "$MODEL" > "$LOGDIR/phone_base_mt.txt" 2>&1
FENG_PHONE_THREADS=4 FENG_PHONE_NEON=0 "$(exe $NEW/pc_mt_suite_phone)" "$MODEL" > "$LOGDIR/phone_mt.txt" 2>&1
grep 'SUMMARY' "$LOGDIR/phone_mt.txt" || true
if diff -q "$LOGDIR/phone_base_mt.txt" "$LOGDIR/phone_mt.txt" >/dev/null; then
    ok "多轮套件输出与基线逐字节相同"
else
    bad "多轮套件输出与基线不同（diff 见 $LOGDIR/phone_mt.diff）"
    diff "$LOGDIR/phone_base_mt.txt" "$LOGDIR/phone_mt.txt" > "$LOGDIR/phone_mt.diff"
fi
# 12/12 是**真机**（aarch64 + NEON）的历史值；在 x86 宿主机上跑的是标量路径，
# 基线本身也只有 8/12。所以这里只把「与基线一致」当判据，绝对值仅作参考——
# 否则在开发机上会误报（2026-10-23 修）。
mem12="$(grep -o 'SUMMARY mem12 [0-9]*/[0-9]*' "$LOGDIR/phone_mt.txt" | head -1)"
if [ "$(uname -m)" = "aarch64" ]; then
    if [ "$mem12" = "SUMMARY mem12 12/12" ]; then
        ok "mem12 12/12（真机）"
    else
        bad "mem12 不是 12/12（$mem12）"
    fi
else
    ok "mem12 与基线一致（$mem12；x86 宿主为标量路径，绝对值不作判据）"
fi
echo

if [ "$rc" = 0 ]; then echo "全部通过（日志在 $LOGDIR/phone_*.txt）"; else echo "有失败项，见上面"; fi
exit $rc
