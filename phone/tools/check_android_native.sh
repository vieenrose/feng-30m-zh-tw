#!/usr/bin/env bash
# 用 Android NDK 的 clang 单独编译 libfeng.so（arm64-v8a），确认 APK 的原生层能编过。
#
# 这一步**不构建 APK**（那需要 Android SDK + Gradle + JDK 17/21），只用 NDK 里的
# clang 把 app/CMakeLists.txt 里那批源文件按同样的参数编一遍。JNI 层最容易出错的
# 是签名与头文件，编过基本就稳了；Kotlin/Compose 部分由 gradle 第一次构建暴露问题。
#
# 依赖：Android NDK（r27c）。路径按 参数 > 环境变量 FENG_NDK > ~/android-ndk-r27c 解析。
#
# 用法：./check_android_native.sh [NDK 目录]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CORE="$ROOT/core"
ENG="$HERE/../engine"
JNI="$ROOT/phone/android/app/src/main/cpp/feng_jni.c"

NDK="${1:-${FENG_NDK:-$HOME/android-ndk-r27c}}"
API="${API:-26}"          # 与 app/build.gradle.kts 的 minSdk 26 对齐
ABI="${ABI:-aarch64-linux-android}"
CC="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/${ABI}${API}-clang"

if [ ! -x "$CC" ]; then
    echo "找不到 NDK clang：$CC" >&2
    echo "用法：$0 [NDK 目录]  或设置 FENG_NDK" >&2
    exit 1
fi

OUT="${OUT:-$HERE/../cli/build-android}"
mkdir -p "$OUT"

KVFLAGS="-DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8"
# 与 app/CMakeLists.txt / phone/cli/build.sh 同一套宏：核心按 FENG_PHONE 分叉
# （pthread 线程池 / NEON 分派），权重格式是 fp16。
CFLAGS="-std=c23 -O2 -fPIC -Wall -DFENG_PHONE=1 -DFENG_FP16_WEIGHTS=1 \
        -DFENG_PHONE_MAX_CTX=32768 -I$CORE -I$ENG $KVFLAGS"

echo "== NDK clang: $($CC --version | head -1)"
echo "== 目标 ABI: ${ABI}${API}（arm64-v8a）"

OBJS=""
# 9 个共享核心 + 2 个手机独有文件（phone/engine 不再有核心副本）
for f in "$CORE"/*.c "$ENG"/*.c; do
    o="$OUT/$(basename "${f%.c}").o"
    echo "   CC $(basename "$f")"
    "$CC" $CFLAGS -c "$f" -o "$o"
    OBJS="$OBJS $o"
done

echo "   CC feng_jni.c"
"$CC" $CFLAGS -c "$JNI" -o "$OUT/feng_jni.o"
OBJS="$OBJS $OUT/feng_jni.o"

echo "   LD libfeng.so"
"$CC" -shared -Wl,-soname,libfeng.so -o "$OUT/libfeng.so" $OBJS -llog

echo
file "$OUT/libfeng.so" | sed 's/^/   /'
ls -la "$OUT/libfeng.so" | awk '{printf "   体积 %.1f KB\n", $5/1024}'
echo
echo "== 导出的 JNI 符号"
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm" -D --defined-only "$OUT/libfeng.so" \
    | grep -i "Java_com_feng" | sed 's/^/   /' || echo "   （没找到 JNI 符号，检查函数名拼写）"

echo
echo "== 确认 sdot 指令（NEON 内核进了 arm64 代码）"
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump" -d "$OUT/libfeng.so" \
    | grep -c '\bsdot\b' | xargs -I{} echo "   sdot 指令数：{}"

echo
echo "OK -> $OUT/libfeng.so"
