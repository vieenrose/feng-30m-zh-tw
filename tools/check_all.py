"""一键验收：把 AGENTS.md 第 7 节的检查链跑一遍（文档 → PC 构建 → 单测 → 套件 → 一致性）。

用法：
    python tools/check_all.py [--model-export board/model_export_v3_19b6]

检查项：
  1. tools/check_md.py + tools/check_docs.py（文档数字必须对得上产物）
  2. pc/build_pc_chat.ps1（PC 引擎两趟构建：核心 C23 + 驱动 C++23，四套 tool 单测）
  3. 套件（需要 model_export 目录，缺省自动跳过）：
     pc_mt_suite（报名字→身份/名字、12 题连续记忆）
     pc_kv_suite 32 题矩阵 + FENG_SUITE=arith 算术子集
     pc_check（fp32 参考 logits MATCH）
退出码非 0 表示有环节失败。
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "pc"


_last_output = ""   # run() 最近一次执行的输出，供调用方做失败重试判断


def run(name, cmd, env=None):
    global _last_output
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    _last_output = out
    tail = "\n".join(out.strip().splitlines()[-6:])
    if tail:
        print(tail, flush=True)
    ok = r.returncode == 0
    print(f"--- {name}: {'PASS' if ok else 'FAIL'} (exit {r.returncode})", flush=True)
    return ok


def run_expect(name, cmd, needles, env=None):
    """跑一个套件：退出码必须为 0，且输出里必须出现全部期望串。
    以前这里明确忽略退出码，套件 7/10 也能 PASS；现在非零一律 FAIL。"""
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    lines = [l for l in out.splitlines() if any(k in l for k in ("SUMMARY", "MATCH", "MISMATCH"))]
    for l in lines[-4:]:
        print("  " + l, flush=True)
    missing = [n for n in needles if n not in out]
    ok = (r.returncode == 0) and not missing
    print(f"--- {name}: {'PASS' if ok else 'FAIL'} (exit {r.returncode})"
          + ("" if ok else f"（缺：{', '.join(missing)}）"), flush=True)
    return ok


def run_mt(name, cmd):
    """多轮套件有已知的容量边界（seq 8/10、mt 7/10），不能因为“能跑”就 PASS。
    这里解析三项分数并对照当前基线；任何一项低于基线就 FAIL。"""
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    for l in out.splitlines():
        if l.startswith("SUMMARY"):
            print("  " + l, flush=True)
    import re
    def score(pat, text):
        m = re.search(pat, text)
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)
    mt = score(r"SUMMARY mt-suite (\d+)/(\d+)", out)
    mem = score(r"SUMMARY mem12 (\d+)/(\d+)", out)
    seq = score(r"SUMMARY seq (\d+)/(\d+)", out)
    # 当前基线（v3.19-embed，保留跨轮 nhist 后实测）：mt 8/10、mem 12/12、seq 10/10。
    # 用 >= 而不是 ==，避免模型改进后反而 FAIL。
    ok = (mt[0] is not None and mt[0] >= 8 and mt[1] == 10 and
          mem[0] == 12 and mem[1] == 12 and
          seq[0] == 10 and seq[1] == 10)
    print(f"--- {name}: {'PASS' if ok else 'FAIL'} (mt={mt[0]}/{mt[1]}, "
          f"mem={mem[0]}/{mem[1]}, seq={seq[0]}/{seq[1]})", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-export",
        default=str(next((d for d in [ROOT / "board" / "model_export_v4",
                                        ROOT / "board" / "model_export_v3_19b6"] if d.exists()),
                          ROOT / "board" / "model_export_v3_19b6")))
    ap.add_argument("--skip-docs", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--require-model", action="store_true",
                    help="模型导出缺失时判 FAIL（发布前验收用）")
    ap.add_argument("--check-export", default="",
                    help="pc_check 用的导出目录（需含 ref_logits.bin）；缺省先用 --model-export，"
                         "没有参考 logits 时回退到 board/model_export_v319base_q4")
    args = ap.parse_args()

    results = []
    skipped = []
    py = sys.executable

    if not args.skip_docs:
        results.append(("check_md", run("文档结构/路径", [py, "tools/check_md.py"])))
        results.append(("check_docs", run("文档数字与产物一致性", [py, "tools/check_docs.py"])))
    if not args.skip_build:
        # 优先系统自带的 Windows PowerShell（Store 的 pwsh 存根在某些环境里
        # 行为古怪）；脚本本身 PS5.1/PS7 都能跑。
        ps = shutil.which("powershell") or shutil.which("pwsh")
        # vswhere 只存在于「标准位置装了 VS/Build Tools」的机器；装在自定义盘符的
        # 机器上 build_pc_chat.ps1 的 MSVC 路径会直接 throw。这里不预判，直接试，
        # 报 vswhere 就自动加 -MinGW 重试——比探测环境变量可靠（子 shell 的
        # 环境跟当前进程不一定一致）。
        base_args = [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                     str(ROOT / "pc" / "build_pc_chat.ps1")]
        # 宿主探测：某些沙箱化的代理环境里 PowerShell 连子进程都派不出来
        # （`& powershell -Command "exit 0"` 静默失败）。这种宿主下构建必然
        # 失败但不是代码问题——明确跳过，别误报。
        try:
            probe = subprocess.run([ps, "-NoProfile", "-Command", "exit 0"],
                                   capture_output=True, timeout=60)
            host_ok = (probe.returncode == 0)
        except Exception:
            host_ok = False
        if not host_ok:
            print("\n[跳过构建] 当前宿主无法通过 PowerShell 派生进程（沙箱环境），"
                  "build_pc 跳过；PC 构建请在本机 PowerShell 里跑 "
                  "pc\\build_pc_chat.ps1（有 vswhere 自动 MSVC，没有则 -MinGW）")
            skipped.append("build_pc")
        else:
            ok = run("PC 引擎两趟构建（核心 C23 + 驱动 C++23） + tool 单测", base_args)
            if not ok and "vswhere" in _last_output:
                print("\n[重试] MSVC 路径不可用（缺 vswhere），自动改用 -MinGW")
                ok = run("PC 引擎两趟构建（-MinGW 重试）", base_args + ["-MinGW"])
            results.append(("build_pc", ok))
    results.append(("phone_single_source", run(
        "单一源码守卫（phone/engine 不得出现引擎副本）",
        [py, "phone/tools/check_single_source.py"])))

    exp = Path(args.model_export)
    if not (exp / "model.bin").exists():
        print(f"\n[跳过套件] 找不到 {exp}/model.bin（模型导出产物不入库，属正常）")
        skipped.append("model suites")
        if args.require_model:
            results.append(("model_present", False))
    else:
        mt = PC / "pc_mt_suite.exe"
        if mt.exists():
            results.append(("mt_suite", run_mt("多轮回归套件", [str(mt), str(exp)])))
        suite = PC / "pc_kv_suite_q2b8.exe"
        prompt = ROOT / "pc" / "prompt_long.txt"
        if suite.exists() and prompt.exists():
            results.append(("kv_suite32", run_expect(
                "32 题矩阵（q2 block8）", [str(suite), str(exp), str(prompt), "5200"],
                ["短任务 27/27", "长文召回 4/4"])))
            env = dict(os.environ, FENG_SUITE="arith")
            results.append(("arith21", run_expect(
                "算术子集", [str(suite), str(exp), str(prompt), "5200"],
                ["短任务 21/21"], env=env)))
        else:
            skipped.append("kv_suite32/arith21（二进制或 prompt_long.txt 缺失）")
        check = PC / "pc_check.exe"
        # 板端发布导出（model_export_v3_19b6）为省体积不带 ref_logits.bin；
        # pc_check 只验证引擎实现，不依赖具体权重，所以回退到带参考 logits 的导出目录。
        cexp = Path(args.check_export) if args.check_export else exp
        if not (cexp / "ref_logits.bin").exists():
            fallback = ROOT / "board" / "model_export_v319base_q4"
            if (fallback / "ref_logits.bin").exists():
                cexp = fallback
        if check.exists() and (cexp / "ref_logits.bin").exists():
            print(f"\n[pc_check] 使用导出目录：{cexp}")
            results.append(("pc_check", run_expect(
                "fp32 参考 logits",
                [str(check), str(cexp), str(ROOT / "logs" / "c_logits_check_all.bin")],
                ["MATCH"])))
        else:
            skipped.append("pc_check（ref_logits.bin 缺失）")
            if args.require_model:
                results.append(("pc_check_ref", False))

    print("\n=== 汇总 ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for name in skipped:
        print(f"  SKIP  {name}")
    bad = [n for n, ok in results if not ok]
    print("结论：" + ("全部通过" if not bad else f"{len(bad)} 项失败：" + ", ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
