#!/usr/bin/env python3
"""校验本地发布产物的 sha256 与 `weights/manifest.json` 是否一致。

为什么需要它：权重体积太大不进 git，但**哈希必须进**。没有这份清单，
换台机器 / 硬盘坏掉之后，这个项目就只剩推理代码——模型没了，而且无法确认
手上这份是不是当初发布的那份。

用法：
    python tools/verify_weights.py            # 缺文件只警告（干净 clone 的正常状态）
    python tools/verify_weights.py --strict   # 缺文件也算失败（发版前 / 备份校验用）

退出码：0 = 全部一致（或非严格模式下只是缺文件）；1 = 哈希不符 / 严格模式缺文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "weights" / "manifest.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="缺文件也算失败")
    args = ap.parse_args()

    if not MANIFEST.is_file():
        print(f"找不到清单 {MANIFEST}")
        return 1
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    arts = man.get("artifacts", [])
    print(f"清单版本 {man.get('version')}（发布时 commit {man.get('commit_at_release')}），"
          f"{len(arts) if isinstance(arts, list) else '?'} 个产物")

    # schema 校验：空清单 / 缺字段 / 非 sha256 / 绝对路径 / 越界路径都算失败。
    # 旧实现遍历空清单时 bad=missing=0，strict 仍然 return 0——发布校验可以静默失效（A48）。
    problems = []
    if not isinstance(arts, list) or not arts:
        problems.append("artifacts 缺失或为空列表")
        arts = []
    for i, a in enumerate(arts):
        if not isinstance(a, dict):
            problems.append(f"artifacts[{i}] 不是对象")
            continue
        for key in ("path", "role", "sha256"):
            if not isinstance(a.get(key), str) or not a.get(key):
                problems.append(f"artifacts[{i}] 缺字段 {key}")
        p = a.get("path")
        if isinstance(p, str) and p:
            pp = Path(p)
            if pp.is_absolute() or ".." in pp.parts:
                problems.append(f"artifacts[{i}] path 不是仓库内相对路径：{p}")
        s = a.get("sha256")
        if isinstance(s, str) and s and not re.fullmatch(r"[0-9a-fA-F]{64}", s):
            problems.append(f"artifacts[{i}] sha256 不是 64 位十六进制")
    if problems:
        print("清单结构异常：")
        for p in problems:
            print("  [结构] " + p)
        print("清单不可信，按失败处理。")
        return 1

    ok = missing = bad = 0
    for a in arts:
        p = ROOT / a["path"]
        if not p.is_file():
            missing += 1
            print(f"  [缺失] {a['path']}  （{a['role']}）")
            continue
        got = sha256_file(p)
        if got != a["sha256"]:
            bad += 1
            print(f"  [不符] {a['path']}\n         期望 {a['sha256']}\n         实际 {got}")
            continue
        size_note = ""
        if a.get("bytes") is not None and p.stat().st_size != a["bytes"]:
            bad += 1
            print(f"  [大小不符] {a['path']}：清单 {a['bytes']} 字节，实际 {p.stat().st_size}")
            continue
        ok += 1
        print(f"  [一致] {a['path']}  {p.stat().st_size / 1048576:.1f} MB{size_note}")

    print(f"\n一致 {ok} / 缺失 {missing} / 不符 {bad}")
    if bad:
        print("哈希不符：这份产物不是当初发布的那份，不要用来烧录或发版。")
        return 1
    if missing and args.strict:
        print("严格模式：缺少发布产物。先按 docs/RELEASE.md 恢复备份。")
        return 1
    if missing:
        print("提示：缺的是未入库的大文件（正常）。发版前请用 --strict 校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
