"""判分器回归：strict-v7 的四类假阳性/假阴性（审查项 A36 / A37）。

只依赖标准库：`chat_probe_heldout` 顶部 `import torch` 是给评测主流程用的，
判分函数本身纯字符串，这里先把 torch 换成空壳再导入，CI 不需要装 torch。

用法：python tests/test_eval_judge.py
退出码 0 = 全部符合预期。
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# torch 空壳：只为了通过 chat_probe_heldout 的 import，判分不碰张量
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from chat_probe_heldout import judge  # noqa: E402


def check(want_ok: bool, prompt: str, expect, answer: str, tag: str) -> int:
    verdict, why = judge(prompt, expect, answer)
    ok = verdict == "OK"
    if ok != want_ok:
        print(f"[FAIL] {tag}: 期望 {'OK' if want_ok else '可疑'}，实际 {verdict}（{why}）")
        return 1
    print(f"[PASS] {tag}")
    return 0


def main() -> int:
    fails = 0
    # A36：否定句不能被 forbidden 子串误杀
    fails += check(True, "你会不会骗人？", None, "不会骗人。", "否定句：不会骗人")
    fails += check(True, "你会不会骗人？", None, "我不会骗你。", "否定句：我不会骗你")
    fails += check(False, "你会不会骗人？", None, "会骗人，但不多。", "肯定句仍要拦")
    # A37：纯数字关键词要有数词边界
    fails += check(False, "水在多少度结冰？", ("0", "零"), "水在100度结冰。", "100 里的 0 不算")
    fails += check(True, "水在多少度结冰？", ("0", "零"), "0 度。", "0 度要判过")
    fails += check(False, "一周有几天？", ("7", "七"), "一周有17天。", "17 里的 7 不算")
    fails += check(True, "一周有几天？", ("7", "七"), "一周有7天。", "7 天要判过")
    # A37：序列题必须完整且顺序正确
    fails += check(False, "从 3 数到 8", ("re:3\\s*[,，、]?\\s*4", "345678"),
                   "3，4。", "序列只答前两项")
    fails += check(True, "从 3 数到 8", ("re:3\\s*[,，、]?\\s*4",),
                   "3、4、5、6、7、8。", "完整升序")
    fails += check(False, "把 1 到 5 倒着数一遍", ("re:5\\s*[,，、]?\\s*4",),
                   "5，4。", "倒序只答前两项")
    fails += check(True, "把 1 到 5 倒着数一遍", ("re:5\\s*[,，、]?\\s*4",),
                   "5、4、3、2、1。", "完整倒序")
    print(f"\n判分回归：{'全部通过' if fails == 0 else f'{fails} 项失败'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
