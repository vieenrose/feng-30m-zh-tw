#!/usr/bin/env bash
# 在 WSL（x86_64 Ubuntu）里交叉编译 aarch64 版手机引擎，并用 qemu 做正确性验证。
#
# 这一步**不碰手机**：本机没有 Android 设备，aarch64 的代码生成、NEON 内核与多线程
# 行为只能靠 qemu 模拟来验。注意 qemu 的**绝对速度没有参考价值**（它逐条翻译指令），
# 真机 tok/s 必须在手机上跑 `phone_chat --bench` 得到。
#
# 依赖（一次性）：
#   sudo apt-get install -y gcc-14-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static
#
# 用法：
#   ./build_wsl_aarch64.sh            # 快速档：NEON 数值自检 + 短生成（约 1 分钟）
#   ./build_wsl_aarch64.sh --full     # 全量：再跑一遍 32 题矩阵（qemu 下很慢，几十分钟）
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CORE="$ROOT/core"
ENG="$HERE/../engine"
PC="$ROOT/pc"
MODEL="${MODEL:-$ROOT/board/model_export_4.0-embed}"
LOGDIR="$ROOT/logs"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

CC="${CC:-aarch64-linux-gnu-gcc-14}"
CXX="${CXX:-aarch64-linux-gnu-g++}"
QEMU="${QEMU:-qemu-aarch64-static}"
OUT="$HERE/build-aarch64"

for t in "$CC" "$QEMU"; do
    command -v "$t" >/dev/null 2>&1 || {
        echo "缺少 $t —— 先跑：" >&2
        echo "  sudo apt-get install -y gcc-14-aarch64-linux-gnu qemu-user-static" >&2
        exit 1
    }
done

KVFLAGS="-DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8"
BASE="-std=c23 -pthread -DFENG_PHONE=1 -DFENG_FP16_WEIGHTS=1 -I$CORE -I$ENG $KVFLAGS -O2 -static"
# 9 个共享核心（core/）+ 2 个手机独有文件（phone/engine/）
HOT="$CORE/feng_llm.c $CORE/feng_quant.c $CORE/feng_smp.c $CORE/feng_sample.c $ENG/feng_phone.c $ENG/feng_neon.c"
COLD="$CORE/feng_model.c $CORE/feng_tokenizer.c $CORE/feng_calc.c $CORE/feng_tools.c $CORE/feng_memory.c"

mkdir -p "$OUT/n" "$OUT/a8" "$LOGDIR"

build_variant() {        # build_variant <子目录> <额外的 -D>
    local dir="$OUT/$1"; shift
    local extra="$*"
    local objs=""
    for f in $HOT $COLD; do
        "$CC" $BASE $extra -c "$f" -o "$dir/$(basename "${f%.c}").o" || return 1
        objs="$objs $dir/$(basename "${f%.c}").o"
    done
    "$CC" $BASE $extra "$HERE/phone_chat.c" $objs -o "$dir/phone_chat" -lm || return 1
    if command -v "$CXX" >/dev/null 2>&1 && [ -f "$PC/pc_kv_suite.cpp" ]; then
        "$CXX" -std=c++23 -pthread -DFENG_PHONE=1 -DFENG_FP16_WEIGHTS=1 -I"$CORE" -I"$ENG" \
            $KVFLAGS $extra -O2 -static \
            "$PC/pc_kv_suite.cpp" $objs -o "$dir/pc_kv_suite_phone" -lm || return 1
    fi
}

echo "== 交叉编译 aarch64（$CC）"
build_variant n  || { echo "编译失败"; exit 1; }
echo "   普通版 -> $OUT/n/phone_chat"
# A8 版：把「激活 int8 量化 + 标量整数点积」编进来，用来证明 sdot 内核与它逐位一致
build_variant a8 -DFENG_GEMV_A8=1 || { echo "A8 版编译失败"; exit 1; }
echo "   A8 对照版 -> $OUT/a8/phone_chat"

file "$OUT/n/phone_chat" | sed 's/^/   /'
"$OUT/n/phone_chat" --help >/dev/null 2>&1 || true

if [ ! -f "$MODEL/model.bin" ]; then
    echo "!! 找不到 $MODEL/model.bin，只编不跑"; exit 0
fi

Q="$QEMU -cpu max"     # -cpu max 才带 dotprod，否则 NEON 内核会走标量回退
rc=0
ok()  { printf '  [PASS] %s\n' "$1"; }
bad() { printf '  [FAIL] %s\n' "$1"; rc=1; }

echo
echo "== 1) sdot 内核 vs 标量 A8 路径（应当**逐位相同**：同一套公式、整数点积精确）"
FENG_PHONE_NEON=1 $Q "$OUT/a8/phone_chat" --ab "$MODEL" 2>&1 | tee "$LOGDIR/phone_a64_ab_a8.txt" | tail -1
if grep -q 'max|diff|=0.000e+00' "$LOGDIR/phone_a64_ab_a8.txt" && grep -q 'MATCH' "$LOGDIR/phone_a64_ab_a8.txt"; then
    ok "sdot 与 A8 标量逐位一致"
else
    bad "sdot 与 A8 标量有差异（见 $LOGDIR/phone_a64_ab_a8.txt）"
fi

echo
echo "== 2) NEON 内核 vs 标量 fp32 路径（数值路径不同，看差异量级与 argmax）"
FENG_PHONE_NEON=1 $Q "$OUT/n/phone_chat" --ab "$MODEL" 2>&1 | tee "$LOGDIR/phone_a64_ab.txt" | tail -1
grep -q 'MATCH' "$LOGDIR/phone_a64_ab.txt" && ok "argmax 一致" || bad "argmax 不一致"

echo
echo "== 3) 短生成：把「线程」与「NEON」两个变量拆开测（qemu 下的比值只作参考，"
echo "       绝对速度无意义——真机数字要在手机上跑 phone_chat --bench）"
echo "   —— 下面 4 行的 tok/s 请在手机上复测，不要引用 qemu 的绝对值 ——"
for t in 1 4; do
  for n in 0 1; do
      FENG_PHONE_THREADS=$t FENG_PHONE_NEON=$n $Q "$OUT/n/phone_chat" --prompt "你是谁？" "$MODEL" \
          > "$LOGDIR/phone_a64_t${t}_n${n}.txt" 2>&1
      tok=$(grep -o '[0-9.]* tok/s' "$LOGDIR/phone_a64_t${t}_n${n}.txt" | tail -1)
      ans=$(grep -v '^\[' "$LOGDIR/phone_a64_t${t}_n${n}.txt" | tail -2 | head -1 | cut -c1-40)
      printf '   线程%d NEON%d ｜ %-12s ｜ %s\n' "$t" "$n" "$tok" "$ans"
  done
done

# 四组回答必须逐字相同（NEON/线程都不改变输出文本）
uniq_ans=$(for f in "$LOGDIR"/phone_a64_t*_n*.txt; do tail -2 "$f" | head -1; done | sort -u | wc -l)
[ "$uniq_ans" = 1 ] && ok "四种配置的回答逐字相同" || bad "四种配置的回答不一致（uniq=$uniq_ans）"

if [ "$FULL" = 1 ]; then
    echo
    echo "== 5) 全量 32 题矩阵（qemu 下很慢）。NEON 开，与 x86 的 A8 结果比 SUMMARY 与逐题判定"
    FENG_PHONE_THREADS=4 FENG_PHONE_NEON=1 $Q "$OUT/n/pc_kv_suite_phone" "$MODEL" "$PC/prompt_long.txt" 5200 \
        > "$LOGDIR/phone_a64_kv_neon.txt" 2>&1
    tail -1 "$LOGDIR/phone_a64_kv_neon.txt"
    echo "   x86 A8 口径：$(tail -1 "$LOGDIR/phone_a8_kv.txt")"
    diff <(grep -o '\[....\]' "$LOGDIR/phone_a8_kv.txt") \
         <(grep -o '\[....\]' "$LOGDIR/phone_a64_kv_neon.txt") >/dev/null \
        && ok "逐题判定与 x86 A8 完全一致" || bad "逐题判定与 x86 A8 有差异"
fi

echo
[ "$rc" = 0 ] && echo "aarch64 全部通过（日志在 logs/phone_a64_*.txt）" || echo "有失败项"
exit $rc
