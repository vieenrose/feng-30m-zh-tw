#!/usr/bin/env bash
# 手机端 CLI 构建脚本（Termux 设备上直接跑；WSL / 桌面 Linux 上跑同一份源码做验证）。
#
#   ./build.sh                    # 用环境里默认的编译器，产物进 build/
#   CC=clang ./build.sh           # 指定 C 编译器
#   OUT=out JOBS=8 ./build.sh     # 指定产物目录与并行度
#
# 依赖：
#   Termux  : pkg install -y clang
#   Ubuntu  : apt install -y gcc-14（-std=c23 需要 gcc 14+；Ubuntu 24.04 的默认 gcc-13
#             只认 -std=c2x，不认 -std=c23）
#
# 产出：build/phone_chat（手机端 CLI）
#       build/pc_kv_suite_phone、build/pc_mt_suite_phone（验收套件，需要 C++ 编译器）
#
# 编译口径与板端固件一致（v4.0 起引擎全面 C23）：
#   核心 C23 + 热点 -O2 / 非热点 -Os；验收套件是 pc/ 下的 C++23 驱动，原样复用。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# MSYS2 / Cygwin（Windows 开发机）的 bash 里 pwd 输出 POSIX 路径（/e/...），
# 而 msys2 的 clang/gcc 是原生 PE 程序，不认这种路径——直接跑会在第一行
# 编译就报 "no such file or directory"。pwd -W 输出 Windows 路径（E:/...），
# 把所有要交给编译器/链接器的路径都换成它。Linux / Termux / WSL 没有 -W，
# 保持原样（行为不变）。
if W="$(cd "$HERE" && pwd -W 2>/dev/null)" && [ -n "$W" ]; then
    HERE="$W"
    R="$(cd "$ROOT" && pwd -W 2>/dev/null || true)"
    [ -n "$R" ] && ROOT="$R"
fi
# 引擎主体直接用 core/（板端 / PC / 手机编同一份源码）；phone/engine 只放
# 手机独有的平台层（feng_phone.c）与 NEON 内核（feng_neon.c）。
CORE="$ROOT/core"
PHONE="$HERE/../engine"
SUITE_DIR="$ROOT/pc"

# -std=c23 需要 gcc 14+ / clang 18+。Ubuntu 24.04 的默认 cc 是 gcc-13（只认 -std=c2x），
# Termux 的 clang 够新。没显式指定 CC 时逐个探测，省得用户踩这个坑。
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
OUT="${OUT:-$HERE/build}"
JOBS="${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) )}"

if [ -z "$CC" ]; then
    echo "找不到支持 -std=c23 的 C 编译器（需要 gcc 14+ 或 clang 18+）" >&2
    echo "  Ubuntu: apt install gcc-14     Termux: pkg install clang" >&2
    exit 1
fi

# KV 配置：默认与板端发布配置一致（q2 / block8）。
#   KV=fp16 ./build.sh    fp16 KV（每值 2 字节、无 scale；2048 上下文 40.4 MB）
#   KV=fp32 ./build.sh    不量化（80.7 MB @2048，只用于对拍）
KV="${KV:-q2}"
case "$KV" in
    q2)   KVFLAGS="-DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8" ;;
    fp16) KVFLAGS="-DFENG_KV_FP16=1" ;;
    int8) KVFLAGS="-DFENG_KV_INT8=1" ;;
    fp32) KVFLAGS="" ;;
    *)    echo "未知 KV=$KV（可选 q2 / fp16 / int8 / fp32）" >&2; exit 2 ;;
esac
# EXTRA 是给使用者的逃生口，例如把上下文上限抬到 32k：
#   EXTRA=-DFENG_PHONE_MAX_CTX=32768 ./build.sh
# 默认值写在 phone/engine/feng_phone.h（4096），那是**编译期常量**，不是硬件限制。
EXTRA="${EXTRA:-}"
CFLAGS="-std=c23 -pthread -DFENG_PHONE=1 -DFENG_FP16_WEIGHTS=1 -I$CORE -I$PHONE $KVFLAGS -Wall $EXTRA"
CXXFLAGS="-std=c++23 -pthread -DFENG_PHONE=1 -DFENG_FP16_WEIGHTS=1 -I$CORE -I$PHONE $KVFLAGS $EXTRA"

# 热点：每 token 内层循环所在（与 main/CMakeLists.txt 的分组一致）
HOT="$CORE/feng_llm.c $CORE/feng_quant.c $CORE/feng_smp.c $CORE/feng_sample.c"
HOT="$HOT $PHONE/feng_phone.c $PHONE/feng_neon.c"
COLD="$CORE/feng_model.c $CORE/feng_tokenizer.c $CORE/feng_calc.c $CORE/feng_tools.c $CORE/feng_memory.c"

mkdir -p "$OUT/obj"

echo "== CC=$CC  OUT=$OUT"
"$CC" --version | head -1

OBJS=""
for f in $HOT; do
    o="$OUT/obj/$(basename "${f%.c}").o"
    echo "   CC $(basename "$f")  [-O2]"
    "$CC" $CFLAGS -O2 -c "$f" -o "$o"
    OBJS="$OBJS $o"
done
for f in $COLD; do
    o="$OUT/obj/$(basename "${f%.c}").o"
    echo "   CC $(basename "$f")  [-Os]"
    "$CC" $CFLAGS -Os -c "$f" -o "$o"
    OBJS="$OBJS $o"
done

echo "   LD phone_chat"
"$CC" $CFLAGS -O2 "$HERE/phone_chat.c" $OBJS -o "$OUT/phone_chat" -lm

# 验收套件：直接复用板端 pc/ 那两份套件源码，保证判定口径完全一致（不另写一份题表）。
# 它们是 C++23 驱动（与板端 PC 侧同一份文件），链上面的 C 目标文件即可。
if [ -f "$SUITE_DIR/pc_kv_suite.cpp" ] && command -v "$CXX" >/dev/null 2>&1; then
    echo "   LD pc_kv_suite_phone / pc_mt_suite_phone（复用 pc 的套件源码）"
    "$CXX" $CXXFLAGS -O2 "$SUITE_DIR/pc_kv_suite.cpp" $OBJS -o "$OUT/pc_kv_suite_phone" -lm
    "$CXX" $CXXFLAGS -O2 "$SUITE_DIR/pc_mt_suite.cpp" $OBJS -o "$OUT/pc_mt_suite_phone" -lm
else
    echo "!! 跳过验收套件（找不到 $SUITE_DIR/pc_kv_suite.cpp 或没有 C++ 编译器）"
fi

echo
echo "完成 -> $OUT"
ls -la "$OUT" | grep -v '^d' || true
echo
echo "下一步（<MODEL> 指向含 model.bin / tokenizer.bin 的目录）："
echo "  $OUT/phone_chat --bench 32 <MODEL>                       # tok/s"
echo "  $OUT/pc_kv_suite_phone <MODEL> $SUITE_DIR/prompt_long.txt 5200"
echo "  $OUT/pc_mt_suite_phone <MODEL>"
