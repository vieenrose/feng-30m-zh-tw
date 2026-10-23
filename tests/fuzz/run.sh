#!/usr/bin/env bash
# 引擎入口模糊测试 + sanitizer 回归。
#
# 判据：ASAN/UBSAN **零报告**，且三个入口的模式都跑完。CI 直接调用本脚本。
#
# 用法：
#   tests/fuzz/run.sh                 # 默认每模式 2 万轮
#   ROUNDS=200000 tests/fuzz/run.sh   # 更狠
#   CC=clang CXX=clang++ tests/fuzz/run.sh
#
# 依赖：clang（或支持 -fsanitize=address,undefined 的 gcc）。msys2 的 ucrt64 gcc
# 没有 ASAN 运行时，Windows 上请用 clang64。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# MSYS2 / Cygwin：pwd 给 POSIX 路径（/e/...），原生 PE 编译器不认；
# pwd -W 给 Windows 路径。Linux/macOS 没有 -W，保持原样。
if W="$(cd "$HERE" && pwd -W 2>/dev/null)" && [ -n "$W" ]; then
    HERE="$W"
    R="$(cd "$ROOT" && pwd -W 2>/dev/null || true)"
    [ -n "$R" ] && ROOT="$R"
fi
OUT="${OUT:-$ROOT/.cache/fuzz}"
ROUNDS="${ROUNDS:-20000}"
CC="${CC:-clang}"
CXX="${CXX:-clang++}"
SAN="${SAN:--fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all}"
OPT="${OPT:--O1 -g}"

KVFLAGS="-DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8"
CFLAGS="-std=c23 $OPT $SAN -pthread -I$ROOT/core -DFENG_FP16_WEIGHTS=1 -DFENG_MAX_SCORES=4096 $KVFLAGS"

CORE="feng_model feng_llm feng_quant feng_smp feng_tokenizer feng_sample feng_calc feng_tools feng_memory"

mkdir -p "$OUT"
echo "== CC=$CC  SAN=$SAN  ROUNDS=$ROUNDS"
"$CC" --version | head -1

OBJS=""
for f in $CORE; do
    "$CC" $CFLAGS -c "$ROOT/core/$f.c" -o "$OUT/$f.o"
    OBJS="$OBJS $OUT/$f.o"
done
"$CC" $CFLAGS "$HERE/fuzz_main.c" $OBJS -o "$OUT/fuzz_main" -lm
echo "   built $OUT/fuzz_main"

fails=0
for mode in calc tools model; do
    if ! "$OUT/fuzz_main" "$mode" "$ROUNDS" 12345; then
        echo "   [FAIL] $mode"
        fails=$((fails + 1))
    fi
done

# 四套单测也过一遍 sanitizer（同一批目标文件，顺手覆盖）
for t in pc_calc_test pc_tools_test pc_mem_test pc_sample_test; do
    src="$ROOT/pc/$t.cpp"
    [ -f "$src" ] || continue
    "$CXX" -std=c++23 $OPT $SAN -pthread -I$ROOT/core -DFENG_FP16_WEIGHTS=1 \
           -DFENG_MAX_SCORES=4096 $KVFLAGS "$src" $OBJS -o "$OUT/$t" -lm
    if ! "$OUT/$t" > "$OUT/$t.log" 2>&1; then
        echo "   [FAIL] $t（见 $OUT/$t.log）"
        fails=$((fails + 1))
    else
        echo "   [PASS] $t（ASAN/UBSAN 零报告）"
    fi
done

if [ "$fails" -ne 0 ]; then
    echo "sanitizer 回归：$fails 项失败"
    exit 1
fi
echo "sanitizer 回归：全部通过"
