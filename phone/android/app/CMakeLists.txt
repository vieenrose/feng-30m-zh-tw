# Android 端原生库：core/ 那份 C23 引擎 + 手机平台层编成 libfeng.so。
#
# 2026-10-23 起 phone/engine **不再**是 core/ 的手工副本：三端（板端 / PC / 手机）
# 编同一份 core/*.c，手机差异全部走 -DFENG_PHONE=1 的编译期分支；phone/engine/
# 只剩手机独有的平台层（feng_phone.c）与 NEON 内核（feng_neon.c）。
# 旧版本这里还指向 phone/engine/feng_llm.c 等已删除的副本，干净 checkout 上
# CMake 配置阶段就 FATAL_ERROR（审查项 A11）。
cmake_minimum_required(VERSION 3.22)
project(feng30m C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# CMAKE_CURRENT_SOURCE_DIR = phone/android/app
set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../core)
set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../engine)

if(NOT EXISTS ${CORE_DIR}/feng_llm.c)
    message(FATAL_ERROR "找不到共享核心 ${CORE_DIR}/feng_llm.c")
endif()
if(NOT EXISTS ${ENGINE_DIR}/feng_phone.c)
    message(FATAL_ERROR "找不到手机平台层 ${ENGINE_DIR}/feng_phone.c")
endif()

set(FENG_HOT
    ${CORE_DIR}/feng_llm.c
    ${CORE_DIR}/feng_quant.c
    ${CORE_DIR}/feng_smp.c
    ${CORE_DIR}/feng_sample.c
    ${ENGINE_DIR}/feng_phone.c
    ${ENGINE_DIR}/feng_neon.c)

set(FENG_COLD
    ${CORE_DIR}/feng_model.c
    ${CORE_DIR}/feng_tokenizer.c
    ${CORE_DIR}/feng_calc.c
    ${CORE_DIR}/feng_tools.c
    ${CORE_DIR}/feng_memory.c)

# JNI 壳必须写全路径：AGP 的 externalNativeBuild 下，add_library 里的**相对路径**
# 是相对本 CMakeLists 所在目录（app/）解析的，而 feng_jni.c 在 app/src/main/cpp/ 下
# ——写 `feng_jni.c` 会报 "Cannot find source file"（实测）。
add_library(feng SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feng_jni.c
    ${FENG_HOT}
    ${FENG_COLD})

target_include_directories(feng PRIVATE ${CORE_DIR} ${ENGINE_DIR})

# 与板端发布配置一致：q2 KV / block8。容量上限 32768（KV 154MB），
# App 默认 32k、分配失败回退 2048（9.6MB）。
# FENG_PHONE / FENG_FP16_WEIGHTS 与 phone/cli/build.sh 完全一致：核心是按这两个宏
# 的条件编译分叉的，漏给就会拿板端分支去跑手机（线程池/权重格式都会错）。
target_compile_definitions(feng PRIVATE
    FENG_KV_Q2=1 FENG_KV_Q2_BLOCK=8 FENG_PHONE_MAX_CTX=32768
    FENG_PHONE=1 FENG_FP16_WEIGHTS=1)

# 热点 -O2、非热点 -Os（与 main/CMakeLists.txt 的分组一致，控制 .so 体积）
set_source_files_properties(${FENG_HOT} PROPERTIES COMPILE_OPTIONS "-O2")
set_source_files_properties(${FENG_COLD} PROPERTIES COMPILE_OPTIONS "-Os")

target_link_libraries(feng PRIVATE log)
