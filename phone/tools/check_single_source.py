#!/usr/bin/env python3
"""手机端「单一源码」守卫。

2026-10-23 之前，`phone/engine/` 是 `core/` 的手工派生副本（18 个文件、4909 行，
其中 `feng_llm.c` 分叉 458 行、`feng_smp.c` 246 行）。同步靠人 `cp`，
`check_drift.py` 只能在漂移**之后**报警——这是持续 bug 源。

现在改成：三端（板端 / PC / 手机）编**同一份** `core/*.c`，手机差异用
`-DFENG_PHONE=1` 打开 `core/` 里的 `#ifdef FENG_PHONE` 分支；`phone/engine/`
只保留手机独有的两个模块：

    feng_phone.c/h   平台层（mmap 加载、时间源、pthread 线程池、NEON 开关）
    feng_neon.c/h    aarch64 NEON 内核（Q4 sdot / fp16 / q2 注意力）

本脚本守的就是这条不变量：**只要 `phone/engine/` 里又出现引擎主体的副本，
或者构建脚本又去读副本，就报错**。

用法：python phone/tools/check_single_source.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHONE_ENGINE = ROOT / "phone" / "engine"

# phone/engine 允许存在的文件（手机独有模块）
ALLOWED = {
    "feng_phone.c", "feng_phone.h",
    "feng_neon.c", "feng_neon.h",
}

# 引擎主体：这些文件必须来自 core/，不允许在 phone/engine 里出现副本
ENGINE_FILES = [
    "feng.h", "feng_llm.c", "feng_quant.c", "feng_smp.c", "feng_model.c",
    "feng_sample.c", "feng_tokenizer.c", "feng_tokenizer.h", "feng_calc.c",
    "feng_calc.h", "feng_tools.c", "feng_tools.h", "feng_memory.c", "feng_memory.h",
]

# 构建脚本必须满足的模式
BUILD_SCRIPTS = ["phone/cli/build.sh", "phone/cli/build_android_cli.sh"]


def main() -> int:
    fails: list[str] = []

    if not PHONE_ENGINE.is_dir():
        print(f"找不到 {PHONE_ENGINE}")
        return 2

    actual = {p.name for p in PHONE_ENGINE.iterdir() if p.is_file()}
    extra = sorted(actual - ALLOWED)
    if extra:
        fails.append(
            "phone/engine/ 出现了引擎副本：" + "、".join(extra)
            + "\n    引擎主体应直接编 core/（用 -DFENG_PHONE=1 打开手机分支），"
            "手机独有模块只允许：" + "、".join(sorted(ALLOWED))
        )
    missing = sorted(ALLOWED - actual)
    if missing:
        fails.append("phone/engine/ 缺少手机独有模块：" + "、".join(missing))

    # 引擎主体必须在 core/ 里存在（防止误删）
    for name in ENGINE_FILES:
        if not (ROOT / "core" / name).is_file():
            fails.append(f"core/{name} 不存在（引擎主体丢了）")

    # 构建脚本必须引用 core/ 并定义 FENG_PHONE
    for rel in BUILD_SCRIPTS:
        p = ROOT / rel
        if not p.is_file():
            fails.append(f"{rel} 不存在")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "-DFENG_PHONE=1" not in text:
            fails.append(f"{rel} 没有 -DFENG_PHONE=1（手机分支不会生效）")
        if '"$CORE/' not in text and "$CORE/" not in text:
            fails.append(f"{rel} 没有从 $CORE 取源文件（又去读副本了？）")

    if fails:
        print("单一源码守卫：失败")
        for f in fails:
            print("  [FAIL] " + f)
        return 1

    print(f"单一源码守卫：通过（core/ 为唯一源，phone/engine 仅 {len(actual)} 个手机模块）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
