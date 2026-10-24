"""v3.7 KV-QAT：训练时把 K/V 按板端 q2 block8 方案注入量化噪声，做量化感知微调。

实现方式（不改 transformers 源码）：
  1. 训练时给每层 v_proj 挂 forward hook：输出 reshape 成 [B,T,heads,64]，
     按 q2 block8 量化再反量化（STE 直通梯度）——等价于板端 V cache 的量化；
  2. 把 qwen3 建模模块里的 apply_rotary_pos_emb 包一层：RoPE 之后对 key 做同样的
     q2 block8 量化（STE）——等价于板端 K cache 的量化。
评估/导出时不挂 hook，所以 fp32/int8 路径不受影响；q2 板上路径的鲁棒性来自训练。

用法：
  python scripts/v3_7_kv_qat.py --init archive/v3_6r\\final\\ctx32768\\final \
      --data archive/v3_7\\qat_data.jsonl --mt archive/v3_5d\\mt_convs.jsonl \
      --out archive/v3_7\\final --epochs 3 --lr 3e-5
"""
import argparse
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_2_build_identity_mix import build_identity_pool  # noqa: E402
from v3_6_sft_patch import load_convs  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def q2_block8_ste(x: torch.Tensor) -> torch.Tensor:
    """q2：每 8 个值一个 fp16 scale，对称量化 + STE（梯度按恒等回传）。"""
    block = 8
    d = x.shape[-1]
    orig = x
    xb = x.reshape(*x.shape[:-1], d // block, block).float()
    # clamp 的下界小于 fp16 最小 subnormal（~6e-8），转 half 后会下溢成 0，
    # 零块就变成 0/0 -> NaN（A53）。half 之后再夹一次，保证 scale > 0。
    scale = (xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 1.5).half().float()
    scale = scale.clamp_min(1e-7)
    q = torch.round(xb / scale + 1.5).clamp_(0, 3)
    dq = (q - 1.5) * scale
    dq = xb + (dq - xb).detach()
    return dq.reshape(orig.shape).to(orig.dtype)


def install_qat(model) -> None:
    import transformers.models.qwen3.modeling_qwen3 as q3
    orig_rope = q3.apply_rotary_pos_emb

    def rope_qat(query, key, cos, sin, unsqueeze_dim=1):
        q, k = orig_rope(query, key, cos, sin, unsqueeze_dim)
        return q, q2_block8_ste(k)

    q3.apply_rotary_pos_emb = rope_qat

    cfg = model.config
    for layer in model.model.layers:
        v_proj = layer.self_attn.v_proj

        def hook(_mod, _inp, out, heads=cfg.num_attention_heads, hd=cfg.head_dim):
            shape = out.shape
            return q2_block8_ste(out.view(*shape[:-1], heads, hd)).view(shape)

        v_proj.register_forward_hook(hook)
    log("KV-QAT hooks installed: K (post-RoPE) + V (v_proj out), q2 block8 STE")


def q4_block64_ste(w: torch.Tensor) -> torch.Tensor:
    """导出器同款 Q4：每行每 64 个权重共享一个 fp16 scale（max|w|/7，signed 4bit）。

    返回带 STE 的量化权重（前向用量化值，反向按恒等回传），
    让训练直面板端 Q4 权重的取整误差。
    """
    out_f, in_f = w.shape
    assert in_f % 64 == 0
    wb = w.reshape(out_f, in_f // 64, 64).float()
    scale = (wb.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8).half().float()
    scale = scale.clamp_min(1e-7)         # fp16 下溢成 0 会让零块产生 NaN（A53）
    q = torch.round(wb / scale).clamp_(-8, 7)
    dq = (q * scale).reshape(out_f, in_f)
    return w + (dq - w).detach()


def install_wqat(model) -> None:
    """把所有权重矩阵的 forward 换成"先按导出格式量化再算"（embedding 与 lm_head 共享同一份权重）。"""
    import torch.nn.functional as F

    def linear_forward(self, x):
        return F.linear(x, q4_block64_ste(self.weight), self.bias)

    def embed_forward(self, ids):
        return F.embedding(ids, q4_block64_ste(self.weight), self.padding_idx)

    n_lin = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.dim() == 2 \
                and mod.weight.shape[1] % 64 == 0 and mod.weight.numel() >= 200:
            mod.forward = types.MethodType(linear_forward, mod)
            n_lin += 1
    emb = model.get_input_embeddings()
    if emb is not None and emb.weight.dim() == 2 and emb.weight.shape[1] % 64 == 0:
        emb.forward = types.MethodType(embed_forward, emb)
    log(f"W-QAT：{n_lin} 个权重矩阵 + embedding 按 Q4 block64（fp16 scale）STE 前向")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mt", default="")
    ap.add_argument("--mt-n", type=int, default=400)
    ap.add_argument("--identity-n", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--wqat", action="store_true",
                    help="同时做权重 QAT（导出器同款 Q4 block64），让模型吃下板端权重量化误差")
    ap.add_argument("--train-last", type=int, default=0,
                    help="只训练最后 N 层 + norm（0=全参）；末层微调对检索/召回的扰动最小")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--retr", default="",
                    help="检索窗口目录（retr{L}_ids.npy/mask.npy）；给了就在短样本后追加长文 QAT")
    ap.add_argument("--retr-n", default="4096:200,8192:60",
                    help="每个长度的窗口数与 batch，如 4096:200:2,8192:60:1（省略 batch 默认 4096→2、其他→1）")
    ap.add_argument("--retr-epochs", type=float, default=1.0)
    ap.add_argument("--retr-lr", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 个训练步保存一次 <out>/rolling/stepN（0=只在结束时保存）")
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.init)

    convs = load_convs(args.data, 0, rng)
    n_data = len(convs)
    if args.mt:
        convs += load_convs(args.mt, args.mt_n, rng)
    if args.identity_n:
        convs += build_identity_pool(rng, args.identity_n)
    log(f"样本：QAT 数据 {n_data} + 多轮/身份 {len(convs) - n_data} = {len(convs)} 条")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 3
    samples = []
    for msgs in convs:
        ids, mask = [], []
        for role, text in msgs:
            t = tok.encode(f"<|im_start|>{role}\n{text}<|im_end|>\n", add_special_tokens=False)
            ids += t
            mask += [1 if role == "assistant" else 0] * len(t)
        if len(ids) > args.max_len:
            ids, mask = ids[:args.max_len], mask[:args.max_len]
        if sum(mask) >= 4:
            samples.append((ids, mask))
    samples.sort(key=lambda s: len(s[0]))
    log(f"可用样本 {len(samples)}，平均长度 {np.mean([len(s[0]) for s in samples]):.0f}")

    model = Qwen3ForCausalLM.from_pretrained(args.init, dtype=torch.float32).to("cuda").train()
    model.config.use_cache = False
    if args.train_last > 0:
        for p in model.parameters():
            p.requires_grad = False
        n_layers = len(model.model.layers)
        for layer in model.model.layers[max(0, n_layers - args.train_last):]:
            for p in layer.parameters():
                p.requires_grad = True
        norm = getattr(model.model, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        log(f"[train-last] 只训练最后 {args.train_last} 层 + norm：{n_train/1e6:.2f}M / {n_all/1e6:.2f}M 参数")
    install_qat(model)
    if args.wqat:
        install_wqat(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.05, fused=True)

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    # W-QAT 下输出投影也必须吃 Q4 噪声：install_wqat 只换了 Linear/embedding 的
    # forward，chunked_lm_loss 直接读 model.lm_head.weight，会绕过量化（A54）。
    def lm_loss(inp: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        from student_utils import chunked_lm_loss
        hw = q4_block64_ste(model.lm_head.weight) if args.wqat else None
        return chunked_lm_loss(model, inp, lab, chunk=512, head_weight=hw)

    ckpt_step = 0
    def save_rolling(step: int) -> None:
        """训练中途落盘；32k 长跑 OOM/断点后不用从头再来。"""
        d = dest / "rolling" / f"step{step}"
        d.mkdir(parents=True, exist_ok=True)
        model.config.use_cache = True
        model.save_pretrained(d, safe_serialization=True)
        tok.save_pretrained(d)
        model.config.use_cache = False
        log(f"saved rolling -> {d}")

    steps_per_epoch = max(1, len(samples) // args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    t0, loss_last = time.time(), None
    for step in range(1, total + 1):
        if step <= args.warmup:
            lr = args.lr * step / args.warmup
        else:
            prog = (step - args.warmup) / max(1, total - args.warmup)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.sample(range(len(samples)), args.batch)
        rows = [samples[i] for i in idx]
        L = max(len(r[0]) for r in rows)
        b_ids = np.full((len(rows), L), pad_id, dtype=np.int64)
        b_lab = np.full((len(rows), L), -100, dtype=np.int64)
        for k, (ids, mask) in enumerate(rows):
            b_ids[k, :len(ids)] = ids
            lab = np.asarray(ids, dtype=np.int64)
            lab[np.asarray(mask) == 0] = -100
            b_lab[k, :len(lab)] = lab
        t_ids = torch.tensor(b_ids, dtype=torch.long, device="cuda")
        t_lab = torch.tensor(b_lab, dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=t_ids).logits
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, out.shape[-1]).float(),
                t_lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        ckpt_step += 1
        loss_last = float(loss.detach())
        if args.save_every > 0 and ckpt_step % args.save_every == 0:
            save_rolling(ckpt_step)
        if step % 25 == 0 or step == total:
            log(f"step {step}/{total} loss {loss_last:.4f} lr {lr:.2e} gnorm {gn:.2f} "
                f"{(time.time() - t0) / 60:.1f}min")

    # ---- 长文 QAT：在检索窗口上带着 q2 噪声练长上下文（只对答案 token 算 loss） ----
    if args.retr:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        specs = []
        for item in args.retr_n.split(","):
            parts = item.split(":")
            L = int(parts[0]); n = int(parts[1])
            b = int(parts[2]) if len(parts) > 2 else (2 if L <= 4096 else 1)
            specs.append((L, n, b))
        for L, n, b in specs:
            ids = np.load(Path(args.retr) / f"retr{L}_ids.npy", mmap_mode="r")
            mask = np.load(Path(args.retr) / f"retr{L}_mask.npy", mmap_mode="r")
            accum = 2
            steps = max(1, n * int(args.retr_epochs) // (b * accum))
            log(f"=== 长文 QAT {L}: {n} 窗口 | {b}x{accum} | {steps} 步 | lr {args.retr_lr:g} ===")
            rng2 = random.Random(args.seed + L)
            for step in range(1, steps + 1):
                prog = step / max(1, steps)
                lr = args.retr_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                ls = 0.0
                for _ in range(accum):
                    idx = rng2.sample(range(ids.shape[0]), b)
                    a = np.asarray(ids[idx])[:, :L]
                    m = np.asarray(mask[idx])[:, :L]
                    tin = torch.tensor(a, dtype=torch.long, device="cuda")
                    mt_ = torch.tensor(m, dtype=torch.uint8, device="cuda")
                    inp, lab = tin[:, :-1], tin[:, 1:].clone()
                    lab[mt_[:, 1:] == 0] = -100
                    with torch.autocast("cuda", dtype=torch.bfloat16), \
                            sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
                        loss_all = lm_loss(inp, lab)
                    (loss_all / accum).backward()
                    ls += float(loss_all.detach())
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step()
                ckpt_step += 1
                if args.save_every > 0 and ckpt_step % args.save_every == 0:
                    save_rolling(ckpt_step)
                if step % 10 == 0 or step == steps:
                    log(f"[longQAT{L}] step {step}/{steps} loss {ls/accum:.4f} lr {lr:.2e} "
                        f"gnorm {gn:.2f} {(time.time()-t0)/60:.1f}min")

    model.config.use_cache = True
    model.save_pretrained(dest, safe_serialization=True)
    tok.save_pretrained(dest)
    log(f"saved -> {dest}（loss {loss_last:.4f}）")


if __name__ == "__main__":
    main()
