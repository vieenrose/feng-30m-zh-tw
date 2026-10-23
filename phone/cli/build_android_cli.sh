#!/usr/bin/env bash
# 手机端引擎交叉构建（Android NDK，arm64-v8a），产物可直接 adb push 到手机跑。
#
# 与 build.sh 同口径：热点 -O2 / 非热点 -Os，验收套件复用 pc/ 下的 C++23 驱动。
# 用法：NDK=<ndk目录> ./build_android_cli.sh   （NDK 默认取 $FENG_NDK 或
#       ~/AppData/Local/Android/Sdk/ndk/<ver> 或 ~/android-ndk-r27c）
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# MSYS2 / Cygwin（Windows 开发机）的 pwd 输出 POSIX 路径（/e/...），NDK 的
# clang 是原生 PE 程序不认；pwd -W 输出 Windows 路径（E:/...）。Linux/Termux
# /WSL 上没有 -W，保持原样。
if W="$(cd "$HERE" && pwd -W 2>/dev/null)" && [ -n "$W" ]; then
    HERE="$W"
    R="$(cd "$ROOT" && pwd -W 2>/dev/null || true)"
    [ -n "$R" ] && ROOT="$R"
fi
# 引擎主体直接用 core/（三端同一份源码）；phone/engine 只放手机独有的
# 平台层（feng_phone.c）与 NEON 内核（feng_neon.c）。
CORE="$ROOT/core"
PHONE="$HERE/../engine"
SUITE_DIR="$ROOT/pc"

# ---- 找 NDK ----
NDK="${NDK:-${FENG_NDK:-}}"
if [ -z "$NDK" ]; then
    for c in "$HOME"/android-ndk-r27c \
             "$HOME/AppData/Local/Android/Sdk/ndk"/*; do
        [ -x "$c/toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android26-clang" ] && NDK="$c" && break
        [ -x "$c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang" ] && NDK="$c" && break
    done
fi
if [ -z "$NDK" ] || [ ! -d "$NDK" ]; then
    echo "找不到 Android NDK（设置 NDK= 或 FENG_NDK=）" >&2
    exit 1
fi

HOST="windows-x86_64"
[ -x "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" ] && HOST="linux-x86_64"
BIN="$NDK/toolchains/llvm/prebuilt/$HOST/bin"
API="${API:-26}"                       # 与 app/build.gradle.kts 的 minSdk 26 对齐
CC="$BIN/aarch64-linux-android$API-clang"
CXX="$BIN/aarch64-linux-android$API-clang++"
[ -x "$CC" ] || { echo "找不到 $CC" >&2; exit 1; }

OUT="${OUT:-$HERE/build_android}"
KVFLAGS="-DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8"   # 与板端发布配置一致
CFLAGS="-std=c23 -O2 -pthread -DFENG_PHONE=1 -DFENG_FP16_WEIGHTS=1 -I$CORE -I$PHONE $KVFLAGS -Wall"
CXXFLAGS="-std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics -O2 -pthread -DFENG_PHONE=1 -DFENG_FP16_WEIGHTS=1 -I$CORE -I$PHONE $KVFLAGS"

HOT="$CORE/feng_llm.c $CORE/feng_quant.c $CORE/feng_smp.c $CORE/feng_sample.c"
HOT="$HOT $PHONE/feng_phone.c $PHONE/feng_neon.c"
COLD="$CORE/feng_model.c $CORE/feng_tokenizer.c $CORE/feng_calc.c $CORE/feng_tools.c $CORE/feng_memory.c"

mkdir -p "$OUT/obj"
echo "== NDK=$NDK ($HOST) OUT=$OUT"
OBJS=""
for f in $HOT; do
    echo "   CC $(basename "$f")  [-O2]"
    "$CC" $CFLAGS -c "$f" -o "$OUT/obj/$(basename "${f%.c}").o"
    OBJS="$OBJS $OUT/obj/$(basename "${f%.c}").o"
done
for f in $COLD; do
    echo "   CC $(basename "$f")  [-Os]"
    "$CC" $CFLAGS -Os -c "$f" -o "$OUT/obj/$(basename "${f%.c}").o"
    OBJS="$OBJS $OUT/obj/$(basename "${f%.c}").o"
done

echo "   LD phone_chat / pc_kv_suite_phone / pc_mt_suite_phone"
"$CC" $CFLAGS "$HERE/phone_chat.c" $OBJS -o "$OUT/phone_chat" -lm
"$CXX" $CXXFLAGS "$SUITE_DIR/pc_kv_suite.cpp" $OBJS -o "$OUT/pc_kv_suite_phone" -lm
"$CXX" $CXXFLAGS "$SUITE_DIR/pc_mt_suite.cpp" $OBJS -o "$OUT/pc_mt_suite_phone" -lm
echo "完成 -> $OUT"
ls -la "$OUT" | grep -v '^d' || true
