#!/usr/bin/env bash
# 用**板端原始引擎**（core/*.c）构建同一份验收套件，作为对照基线。
#
# 用途：phone/engine 是引擎的派生副本，改了平台层、多线程与 NEON 之后，必须证明
# 「手机引擎在关闭新特性时，输出与原始引擎逐字节相同」。这个脚本产出对照组，
# 与 build.sh 的产物用同一条命令跑、再 diff 输出即可。
#
# 用法：CC=gcc-14 CXX=g++ ./build_baseline.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# MSYS2 / Cygwin：pwd 给 POSIX 路径（/e/...），原生 PE 编译器/程序不认；
# pwd -W 给 Windows 路径。Linux/macOS 没有 -W，保持原样。
if W="$(cd "$HERE" && pwd -W 2>/dev/null)" && [ -n "$W" ]; then
    HERE="$W"
    R="$(cd "$ROOT" && pwd -W 2>/dev/null || true)"
    [ -n "$R" ] && ROOT="$R"
fi
BOARD="$ROOT/core"
PC="$ROOT/pc"

# 与 build.sh 同一套编译器探测（-std=c23 需要 gcc 14+ / clang 18+）
pick_cc() {
    local want="$1"; shift
    for c in "$@"; do
        if command -v "$c" >/dev/null 2>&1 && echo | "$c" "$want" -E - >/dev/null 2>&1; then
            echo "$c"; return 0
        fi
    done
    echo ""
}
CC="${CC:-$(pick_cc -std=c23 cc gcc-14 clang gcc)}"
CXX="${CXX:-$(pick_cc -std=c++23 c++ g++-14 clang++ g++)}"
OUT="${OUT:-$HERE/build-baseline}"

KVFLAGS="-DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8"
CFLAGS="-std=c23 -pthread -I$BOARD $KVFLAGS"
CXXFLAGS="-std=c++23 -pthread -I$BOARD $KVFLAGS"
SRCS="feng_model.c feng_llm.c feng_quant.c feng_smp.c feng_tokenizer.c feng_sample.c feng_calc.c feng_tools.c feng_memory.c"

mkdir -p "$OUT/obj"
OBJS=""
for f in $SRCS; do
    o="$OUT/obj/${f%.c}.o"
    "$CC" $CFLAGS -O2 -c "$BOARD/$f" -o "$o"
    OBJS="$OBJS $o"
done

"$CXX" $CXXFLAGS -O2 "$PC/pc_kv_suite.cpp" $OBJS -o "$OUT/pc_kv_suite_base" -lm
"$CXX" $CXXFLAGS -O2 "$PC/pc_mt_suite.cpp" $OBJS -o "$OUT/pc_mt_suite_base" -lm
echo "基线 -> $OUT（pc_kv_suite_base / pc_mt_suite_base）"
