"""把长文针检索样本导出成 **预 tokenized** 的 id 文件，给手机端做 32k 验收。

手机端 `phone_chat` 的普通输入走 `char prompt[4096]` + O(n²) BPE，没法一次喂 32k
文档；`--ids-file` 直接吃这里导出的完整 chat template token 序列，绕过这两个限制。
协议与 `eval_longctx_many.py` 一致：token 精确、预留 32 个生成位、埋点 10%~90%。

用法：
    python scripts/make_longctx_ids.py --ctx 4096,8192,16384,32768 --n 8 --neg-n 2
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

FILLER = ("在遥远的山谷里，风穿过松林。村里的老人说，时间像河水一样一去不回。"
          "孩子们在田野上奔跑，数着天上的云。")
QUESTION = ("<|im_start|>user\n上文的保险柜密码是什么？请只回答数字。<|im_end|>\n"
            "<|im_start|>assistant\n")
DECOYS = [
    "（登记：快递柜取件码 {c}。）",
    "（备注：会议室预订号 {c}。）",
    "（记录：快递单号 {c}。）",
    "（登记信息：设备序列号 {c}。）",
]
REFUSAL = "文中没有提到。"
GEN_RESERVE = 32


def body_text(tok, target_tokens: int) -> str:
    unit = FILLER * 3
    ut = max(1, len(tok(unit, add_special_tokens=False)["input_ids"]))
    reps = max(1, (max(0, target_tokens) + 256 + ut - 1) // ut)
    return unit * reps


def rev_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def splice_window(tok, keep: int, inserts: list[str], rng) -> tuple[list[int], list[int]]:
    """token 级拼接：先给 QUESTION 留位置，再把埋点插进最终窗口内。

    旧实现先造 `keep+256` token 的正文、插完事实再 `[-keep:]`——靠前的事实会被
    头部裁掉，样本变成不可回答（A55）。这里保证长度精确 = keep，并断言每条埋点
    都出现在最终解码文本里。
    """
    tail = rev_ids(tok, QUESTION)
    budget = keep - len(tail)
    parts = [rev_ids(tok, s) for s in inserts]
    body_budget = budget - sum(len(p) for p in parts)
    if body_budget < 256:
        raise RuntimeError(f"ctx={keep} 太小，放不下正文 + {len(inserts)} 条埋点")
    body = rev_ids(tok, body_text(tok, body_budget))[:body_budget]
    pos = sorted(rng.randrange(int(0.1 * body_budget), int(0.9 * body_budget))
                 for _ in inserts)
    out, at, prev = [], [], 0
    for p, part in zip(pos, parts):
        p = max(prev, min(p, body_budget))
        out.extend(body[prev:p])
        at.append(len(out))
        out.extend(part)
        prev = p
    out.extend(body[prev:])
    out.extend(tail)
    if len(out) != keep:
        raise RuntimeError(f"窗口长度不符：{len(out)} != {keep}")
    text = tok.decode(out)
    for s in inserts:
        if s not in text:
            raise RuntimeError(f"埋点不在最终输入里：{s!r}")
    return out, at


def build(tok, ctx: int, seed: int, negative: bool) -> dict:
    rng = random.Random(seed)
    keep = max(1, ctx - GEN_RESERVE)
    code = None
    inserts = []
    if negative:
        for _ in range(rng.randint(1, 3)):
            inserts.append(rng.choice(DECOYS).format(c=f"{rng.randint(10000, 99999)}"))
    else:
        code = f"{rng.randint(10000, 99999)}"
        inserts.append(f"（重要信息：保险柜密码是 {code}。）")
    ids, at = splice_window(tok, keep, inserts, rng)
    return {"code": code, "ids": ids, "negative": negative,
            "needle_frac": -1.0 if negative else round(at[0] / len(ids), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "v4" / "base"),
                    help="tokenizer 所在目录（默认 v4/base）")
    ap.add_argument("--out", default=str(ROOT / "eval" / "longctx_ids"))
    ap.add_argument("--ctx", default="4096,8192,16384,32768")
    ap.add_argument("--n", type=int, default=8, help="每个长度的正样本数")
    ap.add_argument("--neg-n", type=int, default=2, help="每个长度的负样本数")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for ctx in [int(x) for x in args.ctx.split(",")]:
        for i in range(args.n + args.neg_n):
            negative = i >= args.n
            s = build(tok, ctx, args.seed + ctx + i, negative)
            name = f"ctx{ctx}_{i:03d}.ids"
            (out / name).write_text(" ".join(str(t) for t in s["ids"]), encoding="ascii")
            meta.append({"file": name, "ctx": ctx, "tokens": len(s["ids"]),
                         "code": s["code"], "negative": negative})
        print(f"ctx={ctx}: {args.n}+{args.neg_n} 条，实际 {len(s['ids'])} tokens/条")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
