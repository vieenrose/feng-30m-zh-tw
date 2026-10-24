#!/usr/bin/env python
"""打 feng-30m 的 Release 包（四个 zip）。

    python tools\\make_release.py --tag v4.0 --dist dist

产物（与 docs/USAGE.md §1 的结构一致）：

    feng-30m-v4.0-release.zip          PC：HF fp32 权重 + 蒸馏数据集
    feng-30m-v4.0-engine.zip           引擎源码：严格 C23 核心 + C++23 PC 驱动 + 板端
    feng-30m-v4.0-embed-release.zip    板端权重：Q4+q2 双 QAT 的 model.bin / tokenizer.bin
    feng-30m-v4.0-embed-firmware.zip   板端一包到底：预编译固件 + 模型 + 烧录说明

为什么不用 `Compress-Archive`：它在 Windows PowerShell 5.1 上；这个脚本用标准库
`zipfile`，任何平台都能跑，而且**写死时间戳**——同样的输入打出**逐字节相同**的
zip，方便核对"这个包是不是从这版权重打的"。（zip 里存的是本地时间，所以跨时区
仍会不同；这里只保证同一台机器上可复现。）

数据集只放**教师蒸馏数据**（提示词 + 教师输出）。本地脚本生成的多轮/补丁/运算
数据不入包——它们能由 `scripts/` 里的脚本重新造出来。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# 复现用固定时间戳（zip 的 DOS 时间最小值是 1980-01-01）
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


# ── 各包的内容清单 ─────────────────────────────────────────────────────

def pc_files(version: str, hf_dir: Path) -> list[tuple[Path, str]]:
    """PC 包：USAGE + LICENSE + MODEL_CARD + HF 权重 + 蒸馏数据集。"""
    out: list[tuple[Path, str]] = [
        (ROOT / "docs" / "USAGE.md", "USAGE.md"),
        (ROOT / "LICENSE", "LICENSE"),
        (ROOT / "v4" / "base" / "MODEL_CARD.md", "MODEL_CARD.md"),
    ]
    for f in sorted(hf_dir.iterdir()):
        if f.is_file() and f.name != "MODEL_CARD.md":
            out.append((f, f"weights/hf/{f.name}"))
    # 只放教师蒸馏数据（提示词 + 教师输出）
    out.append((ROOT / "data" / "v4_teacher.jsonl", "datasets/v4_teacher.jsonl"))
    return out


def engine_files() -> list[tuple[Path, str]]:
    """引擎包：能独立编出 pc_chat / pc_kv_suite / CUDA / pc_check 的那部分源码。"""
    out: list[tuple[Path, str]] = [
        (ROOT / "LICENSE", "LICENSE"),
        (ROOT / "docs" / "USAGE.md", "docs/USAGE.md"),
        (ROOT / "docs" / "esp32-firmware.md", "docs/esp32-firmware.md"),
    ]

    def add_dir(d: Path, prefix: str, patterns: tuple[str, ...]) -> None:
        if not d.exists():
            return
        for pat in patterns:
            for f in sorted(d.glob(pat)):
                if f.is_file():
                    out.append((f, f"{prefix}/{f.name}"))

    add_dir(ROOT / "core", "core", ("*.c", "*.h"))
    add_dir(ROOT / "pc", "pc", ("*.cpp", "*.cu", "*.h", "*.ps1", "*.py", "*.txt"))
    add_dir(ROOT / "board" / "main", "board/main", ("*.c", "*.h", "*.S", "CMakeLists.txt"))
    add_dir(ROOT / "board" / "tools", "board/tools", ("*.py",))
    # 板端顶层 CMakeLists 是 idf.py 的入口：漏了它，解压后的引擎包没法按说明重建
    # 固件（A46）。
    add_dir(ROOT / "board", "board", ("CMakeLists.txt", "partitions.csv", "sdkconfig*", "flash.ps1"))
    return out


def embed_files(export_dir: Path) -> list[tuple[Path, str]]:
    """板端权重包。`weights/esp32/` 与模型平台上的同名子目录一致。"""
    out: list[tuple[Path, str]] = [
        (ROOT / "LICENSE", "LICENSE"),
        (ROOT / "docs" / "USAGE.md", "USAGE.md"),
    ]
    for f in sorted(export_dir.iterdir()):
        if f.is_file():
            out.append((f, f"weights/esp32/{f.name}"))
    return out


def firmware_files(export_dir: Path) -> list[tuple[Path, str]]:
    """板端一包到底：预编译固件 + 模型 + 烧录说明（免装 ESP-IDF）。"""
    build = ROOT / "board" / "build"
    out: list[tuple[Path, str]] = [
        (ROOT / "LICENSE", "LICENSE"),
        (build / "bootloader" / "bootloader.bin", "firmware/bootloader.bin"),
        (build / "partition_table" / "partition-table.bin", "firmware/partition-table.bin"),
        (build / "feng_30m.bin", "firmware/feng_30m.bin"),
        (ROOT / "board" / "partitions.csv", "partitions.csv"),
    ]
    for f in sorted(export_dir.iterdir()):
        if f.is_file():
            out.append((f, f"model/{f.name}"))
    return out


# ── 打包 ───────────────────────────────────────────────────────────────

FLASH_NOTE = """\
feng-30m {version} 板端烧录说明
================================

一、需要的文件
--------------
  firmware/bootloader.bin        → 0x0
  firmware/partition-table.bin   → 0x8000
  firmware/feng_30m.bin          → 0x10000
  model/model.bin                → 0x110000
  model/tokenizer.bin            → 0x1000000

偏移必须与 partitions.csv 一致。model 分区必须结束在 0x1000000（16 MB）之前——
这是 NOR flash 24 位地址的 mmap 窗口上限，不是模块容量。

二、烧录（只需 esptool，不用装 ESP-IDF）
---------------------------------------
  python -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash \\
      0x0      firmware/bootloader.bin \\
      0x8000   firmware/partition-table.bin \\
      0x10000  firmware/feng_30m.bin \\
      0x110000 model/model.bin \\
      0x1000000 model/tokenizer.bin

COM20 换成你自己的串口。若烧录中断，按住 BOOT 再点 RST 进下载模式重来。

三、对话
--------
  115200 8N1，发一行问题回车。固件自带多轮上下文，\\reset 清空。
  终端不是 UTF-8 会显示乱码——若需要 GBK，得自己用 -DFENG_GBK=ON 重编固件。

四、这包里没有的东西
--------------------
  源码与构建办法在本仓库的 docs/ 下；引擎源码另见 engine 包。
  板端是 Q4 权重（14.93 MB）+ q2 KV（2048 上下文，9.62 MB PSRAM）。
"""


def write_zip(path: Path, entries: list[tuple[Path, str]], top: str,
              extra_text: dict[str, str] | None = None) -> tuple[int, int]:
    """entries: [(源文件, 包内相对路径)]；包内一律放在 `top/` 下。"""
    missing = [str(src) for src, _ in entries if not src.exists()]
    if missing:
        raise FileNotFoundError("缺文件：\n  " + "\n  ".join(missing))

    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, rel in entries:
            arc = f"{top}/{rel}"
            zi = zipfile.ZipInfo(arc, date_time=FIXED_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            data = src.read_bytes()
            z.writestr(zi, data)
            total += len(data)
        for rel, text in (extra_text or {}).items():
            zi = zipfile.ZipInfo(f"{top}/{rel}", date_time=FIXED_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, text.encode("utf-8"))
    return len(entries) + len(extra_text or {}), total


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_text(entries: list[tuple[Path, str]]) -> str:
    """包内 SHA256SUMS：按“包内相对路径”排序，每行 `sha256  path`。"""
    lines = [f"{sha256(src)}  {rel}" for src, rel in sorted(entries, key=lambda x: x[1])]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4.0", help="版本号，用于包名与包内顶层目录")
    ap.add_argument("--hf", default="v4/patch_name_e5_lr5e-6",
                    help="HF 格式权重目录（PC 最终模型）")
    ap.add_argument("--esp", default="board/model_export_4.0-embed",
                    help="板端导出目录（model.bin / tokenizer.bin）")
    ap.add_argument("--dist", default="dist", help="zip 输出目录")
    ap.add_argument("--allow-stale-firmware", action="store_true",
                    help="允许 board/build 里的固件比源码旧（默认拒绝，避免发旧二进制）")
    a = ap.parse_args()

    hf_dir = (ROOT / a.hf).resolve()
    export_dir = (ROOT / a.esp).resolve()
    dist = (ROOT / a.dist).resolve()
    dist.mkdir(parents=True, exist_ok=True)

    model = export_dir / "model.bin"
    tok = export_dir / "tokenizer.bin"
    if not model.exists() or not tok.exists():
        raise SystemExit(f"缺 {model} / {tok}")
    if model.stat().st_size > 0xEF0000:
        raise SystemExit(f"model.bin {model.stat().st_size} B 超过 model 分区 0xEF0000")
    if tok.stat().st_size > 0x80000:
        raise SystemExit(f"tokenizer.bin {tok.stat().st_size} B 超过 tokdata 分区 0x80000")

    fw = ROOT / "board" / "build" / "feng_30m.bin"
    if fw.exists():
        newest = max([p.stat().st_mtime for d in (ROOT / "core", ROOT / "board" / "main")
                      for p in d.rglob("*") if p.is_file() and p.suffix in (".c", ".h", ".S", ".txt")] +
                     [p.stat().st_mtime for p in (ROOT / "pc").glob("*.cpp")] + [0])
        if fw.stat().st_mtime < newest and not a.allow_stale_firmware:
            raise SystemExit("board/build/feng_30m.bin 比源码旧；先 idf.py build，"
                             "或显式加 --allow-stale-firmware")


    jobs = [
        (f"feng-30m-{a.tag}-release.zip", f"feng-30m-{a.tag}",
         pc_files(a.tag, hf_dir), None),
        (f"feng-30m-{a.tag}-engine.zip", f"feng-30m-{a.tag}-engine",
         engine_files(), None),
        (f"feng-30m-{a.tag}-embed-release.zip", f"feng-30m-{a.tag}-embed",
         embed_files(export_dir), None),
        (f"feng-30m-{a.tag}-embed-firmware.zip", f"feng-30m-{a.tag}-embed-firmware",
         firmware_files(export_dir),
         {"烧录说明.txt": FLASH_NOTE.format(version=a.tag)}),
    ]

    print(f"=== 打 {a.tag} 的 Release 包 ===")
    print(f"  PC 权重：{a.hf}")
    print(f"  板端导出：{a.esp}\n")

    for name, top, entries, extra in jobs:
        out = dist / name
        extras = dict(extra or {})
        extras.setdefault("SHA256SUMS", manifest_text(entries))
        n, raw = write_zip(out, entries, top, extras)
        print(f"  ✓ {name:42s} {n:3d} 个文件  原始 {raw/1048576:7.2f} MB "
              f"→ zip {out.stat().st_size/1048576:6.2f} MB")
        print(f"      sha256 {sha256(out)}")

    print(f"\n输出目录：{dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
