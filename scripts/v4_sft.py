"""feng-30m 4.0 的大规模 SFT 训练器。

为什么不用 `v3_6_sft_patch.py`：那个脚本是给几百条的小补丁写的，训练循环里是
    idx = rng.sample(range(len(samples)), batch)
**每步独立随机抽**。样本量一大就出问题——4 万条跑 1 个 epoch 时，这样抽只有约
1-1/e ≈ 63% 的样本会被看到，其余永远轮不到；而且同一个样本可能被抽到多次。
本脚本改成**按 epoch 完整洗牌 + 长度分桶**，既不漏样本也基本不浪费 padding。

其它与仓库存量脚本保持一致的地方：
  - 样本格式 `{"messages":[{"role","content"}]}`（与 `v4_build_sft.py` 的输出一致）；
  - loss 只算 assistant 段（含 `<|im_start|>assistant\\n` 到 `<|im_end|>\\n`）；
  - 训练走 `student_utils.chunked_lm_loss`（512 分块 + checkpoint），
    这是 AGENTS.md 的硬性要求——lm_head + 交叉熵直接算会吃爆显存；
  - 产物结构 `<out>/final/` + `summary.json` + `train_log.jsonl`，中途存 `rolling/stepN/`。

用法：
    python scripts\\v4_sft.py --init v4/base --data data\\v4_sft.jsonl --out v4/sft `
        --epochs 1 --lr 2e-5 --batch 24 --accum 2
    # 只训最后 N 层（小步微调/对照用）
    python scripts\\v4_sft.py ... --train-last 2
    # 冒烟
    python scripts\\v4_sft.py ... --steps-cap 3
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import ROOT  # noqa: E402
from student_utils import chunked_lm_loss  # noqa: E402


def load_convs(path: str) -> list[list[tuple[str, str]]]:
    """读 `{"messages":[...]}`；与 v3_6_sft_patch.py 的 load_convs 同口径。"""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = [(m["role"], m["content"]) for m in rec.get("messages", []) if m.get("content")]
        if msgs:
            rows.append(msgs)
    return rows


def encode(tok, conv, max_len: int):
    """→ (ids, mask)；mask=1 的位置算 loss。整段 assistant（含模板标记）都算。"""
    ids, mask = [], []
    for role, text in conv:
        t = tok.encode(f"<|im_start|>{role}\n{text}<|im_end|>\n", add_special_tokens=False)
        ids += t
        mask += [1 if role == "assistant" else 0] * len(t)
    if len(ids) > max_len:
        ids, mask = ids[:max_len], mask[:max_len]
    return ids, mask


def make_labels(ids, mask):
    """构造与 input_ids **对齐**的 labels（chunked_lm_loss 要求调用方先做 +1 错位）。

    `chunked_lm_loss` 内部是 `logits[b,t]` 对 `labels[b,t]`——而 logits[t] 预测的是
    `ids[t+1]`，所以目标必须是**右移一位**的 ids，且只在 `mask[t+1]==1`（下一个位置
    属于 assistant）处监督。

    踩过的坑：第一版直接把 `ids` 按 mask 置 -100 就传进去（没有错位），
    结果 loss 一直在 ln(16384)=9.70 附近，模型看起来像在随机猜——实际只是标签错了位。
    对照：`v3_7_kv_qat.py` 是 `inp, lab = tin[:, :-1], tin[:, 1:]`，
    `v3_6_sft_patch.py` 是 `cross_entropy(out[:, :-1], lab[:, 1:])`，都是错的位。
    """
    a = np.asarray(ids, dtype=np.int64)
    m = np.asarray(mask)
    lab = np.full(a.shape, -100, dtype=np.int64)
    if len(a) > 1:
        lab[:-1] = np.where(m[1:] == 1, a[1:], -100)
    return lab


def make_batches(samples, batch: int, rng: random.Random):
    """长度分桶 + 桶内洗牌：既省 padding 又不失随机性。

    做法：按长度排序 → 切成 batch*32 大小的"大桶" → 桶内洗牌后切批 → 打乱桶顺序。
    这样同批样本长度接近（padding 少），但批次顺序与批内组合每轮都不同。
    """
    order = sorted(range(len(samples)), key=lambda i: len(samples[i][0]))
    mega = batch * 32
    buckets = [order[i:i + mega] for i in range(0, len(order), mega)]
    for b in buckets:
        rng.shuffle(b)
    rng.shuffle(buckets)
    out = []
    for b in buckets:
        for i in range(0, len(b), batch):
            out.append(b[i:i + batch])
    return out


def make_groups(samples, batch: int, accum: int, epochs: float, rng: random.Random,
                steps_cap: int = 0):
    """按 epoch 生成批次，再按**全局微批计数**分组（ accum 份一组 = 一次 optimizer step）。

    旧训练循环用 epoch 内的 (bi+1)%accum 判断，尾微批会跨轮残留：同一组里塞进
    3 份 loss 却各除 accum（缩放和 1.5），步数也和日志对不上（A40）。
    这里每组严格 accum 份，尾部不足一组的微批丢弃（drop_last），返回
    (groups, plan)，plan 用于统计每轮步数。
    """
    accum = max(1, accum)
    epochs_n = max(1, math.ceil(epochs))
    plan = []
    for e in range(epochs_n):
        for b in make_batches(samples, batch, rng):
            plan.append((e, b))
    groups = [plan[i:i + accum] for i in range(0, len(plan) - accum + 1, accum)]
    if steps_cap:
        groups = groups[:steps_cap]
    return groups, plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="初始 HF 目录（v4.0 是 v4/base）")
    ap.add_argument("--data", required=True, help="SFT jsonl（v4_build_sft.py 的产出）")
    ap.add_argument("--val", default="", help="验证集 jsonl（可选，每轮末尾算 val loss）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50, help="线性 warmup 步数")
    ap.add_argument("--batch", type=int, default=24, help="每步样本数（不是 DataLoader batch）")
    ap.add_argument("--accum", type=int, default=2, help="梯度累积步数")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--train-last", type=int, default=0, help="只训最后 N 层 + norm（0=全参）")
    ap.add_argument("--steps-cap", type=int, default=0, help="冒烟用：最多跑几步")
    ap.add_argument("--save-every", type=int, default=0, help="每 N 步存一次 rolling/")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cuda（默认）或 cpu。cpu 只用于在教师占卡时冒烟验证代码路径")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--overwrite", action="store_true",
                    help="允许覆盖已完成的 <out>/final（默认拒绝：保护已训练好的权重）")
    a = ap.parse_args()

    # 已有产物默认拒绝重跑：旧实现在开始时就 mkdir(final)，结束时 rmtree(final)
    # 再保存——同目录重跑会先删掉上一版权重与日志，保存失败就不可恢复（A39）。
    # 放在加载模型之前：拒绝时不该先占 100+ MB 显存/内存。
    out = Path(a.out)
    final = out / "final"
    if not a.overwrite and (final.exists() or (out / "summary.json").exists()
                            or (out / "train_log.jsonl").exists()):
        print(f"!! {out} 已有训练产物（final/ 或 summary.json）；"
              f"确认要覆盖请加 --overwrite（或用新的 --out）", flush=True)
        return 2

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    convs = load_convs(a.data)
    print(f"训练样本 {len(convs)} 条", flush=True)
    tok = AutoTokenizer.from_pretrained(a.init)
    pad_id = getattr(tok, "pad_token_id", 3) or 3

    t0 = time.time()
    samples = []
    masked_total = 0
    for c in convs:
        ids, mask = encode(tok, c, a.max_len)
        if sum(mask) >= 4:                      # 至少 4 个 assistant token 才算有效样本
            samples.append((ids, mask))
            masked_total += sum(mask)
    print(f"有效样本 {len(samples)} 条（编码 {time.time()-t0:.1f}s，"
          f"平均 {masked_total/max(1,len(samples)):.0f} 个 loss token）", flush=True)
    if not samples:
        print("没有有效样本，退出")
        return 1

    val_samples = []
    if a.val:
        for c in load_convs(a.val):
            ids, mask = encode(tok, c, a.max_len)
            if sum(mask) >= 4:
                val_samples.append((ids, mask))
        print(f"验证样本 {len(val_samples)} 条", flush=True)

    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("!! CUDA 不可用，回退 CPU（会非常慢，仅用于验证）", flush=True)
        dev = torch.device("cpu")
    model = Qwen3ForCausalLM.from_pretrained(a.init, dtype=torch.float32).to(dev).train()
    model.config.use_cache = False
    if a.train_last > 0:
        for p in model.parameters():
            p.requires_grad = False
        n_layers = len(model.model.layers)
        for layer in model.model.layers[max(0, n_layers - a.train_last):]:
            for p in layer.parameters():
                p.requires_grad = True
        norm = getattr(model.model, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tt = sum(p.numel() for p in model.parameters())
        print(f"[train-last] 只训最后 {a.train_last} 层 + norm：{tr/1e6:.2f}M / {tt/1e6:.2f}M 参数",
              flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    # fused AdamW 只在 CUDA 上有；CPU 冒烟时退回普通实现
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.05, fused=(dev.type == "cuda"))

    out = Path(a.out)
    final = out / "final"
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "train_log.jsonl").open("w", encoding="utf-8")
    sumf = (out / "summary.json")

    def save(dest: Path):
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(dest, safe_serialization=True)
        tok.save_pretrained(dest)

    @torch.no_grad()
    def eval_val() -> float:
        if not val_samples:
            return float("nan")
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(val_samples), a.batch):
            rows = val_samples[i:i + a.batch]
            L = max(len(r[0]) for r in rows)
            bi = np.full((len(rows), L), pad_id, dtype=np.int64)
            bl = np.full((len(rows), L), -100, dtype=np.int64)
            for k, (ids, mask) in enumerate(rows):
                bi[k, :len(ids)] = ids
                lab = make_labels(ids, mask)
                bl[k, :len(lab)] = lab
            ti = torch.tensor(bi, dtype=torch.long, device=dev)
            tl = torch.tensor(bl, dtype=torch.long, device=dev)
            loss = chunked_lm_loss(model, ti, tl, chunk=512)
            # 权重必须用**全部**已对齐 label 数：chunked_lm_loss 的均值分母是
            # (tl != -100)，旧实现按 tl[:,1:] 统计，assistant 开头那一位被漏掉，
            # 多 batch 加权会偏小（A41）。
            sup = int((tl != -100).sum())
            tot += float(loss) * sup
            n += sup
        model.train()
        return tot / max(1, n)

    # 调度计划：先把每个 epoch 的批次全部生成好，再**按全局微批计数**分组。
    # 旧实现用 epoch 内的 (bi+1)%accum 判断，尾微批（每轮批数不整除 accum）
    # 会残留到下一轮：同一组里塞进 3 份 loss 却各除 2（缩放和 1.5），
    # 步数也和日志对不上（A40）。现在每组固定 accum 份、总步数精确等于分组数。
    epochs_n = max(1, math.ceil(a.epochs))
    groups, plan = make_groups(samples, a.batch, a.accum, a.epochs, rng, a.steps_cap)
    total = len(groups)
    if total == 0:
        print("有效微批不足一个 accum 组，退出", flush=True)
        logf.close()
        return 1
    steps_per_epoch = max(1, len(plan) // max(1, a.accum) // epochs_n)
    print(f"每轮约 {steps_per_epoch} 步，共 {total} 步（accum={a.accum}，"
          f"等效 batch {a.batch*a.accum}）", flush=True)

    t_start = time.time()
    step = 0
    seen_tok = 0
    loss_run = None
    peak = 0.0
    last_epoch = 0
    for group in groups:
        if step >= total:
            break
        for epoch, idx in group:
            last_epoch = epoch
            rows = [samples[i] for i in idx]
            L = max(len(r[0]) for r in rows)
            bid = np.full((len(rows), L), pad_id, dtype=np.int64)
            blab = np.full((len(rows), L), -100, dtype=np.int64)
            for k, (ids, mask) in enumerate(rows):
                bid[k, :len(ids)] = ids
                lab = make_labels(ids, mask)
                blab[k, :len(lab)] = lab
            ti = torch.tensor(bid, dtype=torch.long, device=dev)
            tl = torch.tensor(blab, dtype=torch.long, device=dev)
            nl = int((tl != -100).sum())
            # chunked_lm_loss 返回的**已经是每 token 平均损失**（内部除以了非 -100 的
            # token 数），所以这里不要再做任何长度缩放——第一版乘了 (batch/nl)，
            # 等于又除了一个 ~76，日志里 loss 显示成 0.14（假的）。
            loss = chunked_lm_loss(model, ti, tl, chunk=512)
            (loss / a.accum).backward()
            seen_tok += nl
            loss_run = float(loss.detach()) if loss_run is None else 0.98 * loss_run + 0.02 * float(loss.detach())

        step += 1
        if step <= a.warmup:
            lr = a.lr * step / max(1, a.warmup)
        else:
            prog = (step - a.warmup) / max(1, total - a.warmup)
            lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        el = time.time() - t_start
        rec = {"step": step, "total": total, "epoch": last_epoch + 1,
               "loss": round(loss_run, 4), "lr": lr, "grad_norm": round(gn, 3),
               "loss_tokens": seen_tok, "elapsed_min": round(el / 60, 2),
               "peak_gib": round(peak, 3)}
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()
        if step % 20 == 0 or step == total:
            print(f"  step {step}/{total} ep{last_epoch+1} loss={loss_run:.4f} "
                  f"lr={lr:.2e} gn={gn:.2f} {el/60:.1f}min peak={peak:.2f}GiB", flush=True)
        if a.save_every and step % a.save_every == 0:
            save(out / "rolling" / f"step{step}")

    vloss = eval_val()
    el = time.time() - t_start
    # 先写到 final.new，成功后再原子换入：中途 OOM/磁盘满不会先删掉上一版权重。
    staged = out / "final.new"
    if staged.exists():
        shutil.rmtree(staged)
    save(staged)
    backup = out / "final.bak"
    if backup.exists():
        shutil.rmtree(backup)
    if final.exists():
        final.rename(backup)
    try:
        staged.rename(final)
    except OSError:
        if backup.exists() and not final.exists():
            backup.rename(final)        # 换入失败：恢复上一版
        raise
    if backup.exists():
        shutil.rmtree(backup)           # 新版权重落地后才删备份
    summary = {
        "init": a.init, "data": a.data, "device": a.device, "samples": len(samples),
        "val_samples": len(val_samples),
        "steps": step, "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "accum": a.accum,
        "max_len": a.max_len, "train_last": a.train_last,
        "loss_last": round(loss_run, 4) if loss_run is not None else None,
        "val_loss": None if vloss != vloss else round(vloss, 4),
        "loss_tokens": seen_tok, "peak_gib": round(peak, 3),
        "elapsed_min": round(el / 60, 2), "final": str(final),
    }
    sumf.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logf.close()
    print(f"\n完成：{step} 步 / {el/60:.1f} min / 峰值 {peak:.2f} GiB"
          + (f" / val_loss {vloss:.4f}" if vloss == vloss else ""), flush=True)
    print(f"权重 -> {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
